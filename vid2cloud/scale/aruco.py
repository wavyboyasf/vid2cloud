"""Detekcja markerów ArUco na pełnej rozdzielczości klatki + przeliczenie na wejście modelu.

Detekcja idzie po pełnej klatce (4K), bo marker 12 cm z 3 m ma przy 512 px około
15 pikseli. Dopiero gotowe narożniki skalujemy w dół — patrz `rescale_corners()`
i (następna sesja) `scale/lift.py`.

Przekształcenie wejścia MASt3R-SLAM (sprawdzone w kodzie upstreamu, nie zgadywane)
--------------------------------------------------------------------------------
Ścieżka klatki w silniku:

* `third_party/mast3r-slam/mast3r_slam/dataloader.py:25` — `self.img_size = 512`
  (stała dla wszystkich datasetów monokularnych, także MP4);
* `third_party/mast3r-slam/mast3r_slam/frame.py:112` — każda klatka idzie przez
  `resize_img(img, img_size)` w `create_frame()`; ten sam wywołuje
  `dataloader.py:55` w `get_img_shape()`, żeby poznać kształt wejścia;
* `third_party/mast3r-slam/mast3r_slam/mast3r_utils.py:244` — `resize_img()`:
  dla `size == 512` skaluje **dłuższy bok do 512** (`_resize_pil_image()`,
  `mast3r_utils.py:234`, `PIL.Image.resize`, każdy bok zaokrąglany osobno:
  `int(round(x * 512 / S))`), a potem **przycina centralnie** do wielokrotności
  16 px na każdym boku:

      cx, cy = W // 2, H // 2
      halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
      img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

  (`mast3r_utils.py:256-264`). Dla materiału 16:9 i 4:3 crop wychodzi zerowy
  (3840x2160 -> 512x288, 4000x3000 -> 512x384), ale dla nietypowych proporcji nie:
  3840x1634 -> 512x218 -> crop do 512x208, czyli 5 px zdjęte z góry. Dlatego
  `rescale_corners()` odtwarza całe przekształcenie, nie samą skalę.

Konwencja pikseli. Narożniki z `cv2.aruco` są w konwencji środka piksela, więc
przeliczenie to `x' = (x + 0.5) * W/W1 - 0.5 - left`. Upstream w `Intrinsics`
(`dataloader.py:292`) liczy punkt główny bez tej półpikselowej poprawki
(`K[0,2] / scale_w - half_crop_w`) — różnica wynosi 0.5 * (W/W1 - 1), czyli ok.
0.43 px przy 4K -> 512. To dotyczy tylko trybu z kalibracją; tu trzymamy się
konwencji geometrycznie poprawnej dla próbkowania pointmapy. Pozycja w docs/TODO.md.

Ograniczenie: `frame.py:117-120` potrafi jeszcze przerzedzić siatkę pointmapy
(`config["dataset"]["img_downsample"]`, domyślnie 1 w `config/base.yaml:5`).
Ten moduł zakłada wartość 1.

Zależności (opencv-contrib-python) instalujemy w env vid2cloud-tools, nie mast3r-slam.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import time
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_DICT = "DICT_4X4_50"

#: Dłuższy bok wejścia modelu; `dataloader.py:25`.
MODEL_INPUT_SIZE = 512

#: Kolejność narożników z cv2.aruco: zgodnie z ruchem wskazówek zegara od lewego
#: górnego rogu markera (w jego własnym układzie, nie w układzie obrazu).
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")

#: Kąt z display matrix (konwencja ffprobe) -> obrót OpenCV do orientacji wyświetlania.
ROTATE_CODE = {
    -90: cv2.ROTATE_90_CLOCKWISE,
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    -180: cv2.ROTATE_180,
}


# --------------------------------------------------------------------------- #
# Detekcja
# --------------------------------------------------------------------------- #


def get_dictionary(dict_name: str = DEFAULT_DICT) -> cv2.aruco.Dictionary:
    """Słownik ArUco po nazwie, np. "DICT_4X4_50"."""
    if not dict_name.startswith("DICT_"):
        raise ValueError(f"nazwa słownika musi zaczynać się od 'DICT_', dostałem {dict_name!r}")
    value = getattr(cv2.aruco, dict_name, None)
    if value is None:
        raise ValueError(f"cv2.aruco nie zna słownika {dict_name!r}")
    return cv2.aruco.getPredefinedDictionary(value)


def default_parameters() -> cv2.aruco.DetectorParameters:
    """Parametry detektora: domyślne OpenCV + subpikselowe dociąganie narożników.

    `CORNER_REFINE_SUBPIX` wywołuje wewnątrz `cv2.cornerSubPix` na obrazie w skali
    szarości, z oknem `min(cornerRefinementWinSize, relativeCornerRefinmentWinSize *
    odległość do sąsiedniego narożnika)` — czyli okno kurczy się przy małych markerach,
    czego stały `cornerSubPix` po naszej stronie by nie robił.
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def detect(
    image_bgr: np.ndarray,
    dict_name: str = DEFAULT_DICT,
    parameters: Optional[cv2.aruco.DetectorParameters] = None,
) -> dict[int, np.ndarray]:
    """Wykrywa markery na pełnej klatce. Zwraca {id: ndarray(4, 2) float64}.

    Współrzędne są w pikselach obrazu wejściowego (konwencja środka piksela),
    w kolejności `CORNER_ORDER`. Obraz może być BGR albo już w skali szarości.

    Podnosi `ValueError`, jeśli ten sam ID wypadł na klatce więcej niż raz — przy
    definiowaniu datum skali dwa markery o tym samym ID to nie jest coś, co wolno
    rozstrzygnąć po cichu.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("pusty obraz")
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(
        get_dictionary(dict_name), parameters if parameters is not None else default_parameters()
    )
    corners, ids, _ = detector.detectMarkers(gray)

    out: dict[int, np.ndarray] = {}
    if ids is None:
        return out
    duplicates = []
    for quad, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id in out:
            duplicates.append(marker_id)
            continue
        out[marker_id] = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    if duplicates:
        raise ValueError(f"ten sam ID wykryty wielokrotnie na jednej klatce: {sorted(set(duplicates))}")
    return out


# --------------------------------------------------------------------------- #
# Przeliczenie na rozdzielczość wejścia modelu
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InputTransform:
    """Pełne przekształcenie pełna klatka -> wejście modelu (skala + crop centralny).

    Pola `scale_x`/`scale_y` to mnożniki (W/W1, H/H1), a `left`/`top` to lewy górny
    róg cropu we współrzędnych obrazu **po** skalowaniu.
    """

    scale_x: float
    scale_y: float
    left: float
    top: float
    resized_shape: tuple[int, int]  # (H, W) po skalowaniu, przed cropem
    out_shape: tuple[int, int]  # (H, W) wejścia modelu

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Punkty (..., 2) w pikselach pełnej klatki -> piksele wejścia modelu."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != 2:
            raise ValueError(f"oczekuję punktów (..., 2), dostałem {pts.shape}")
        out = np.empty_like(pts)
        out[..., 0] = (pts[..., 0] + 0.5) * self.scale_x - 0.5 - self.left
        out[..., 1] = (pts[..., 1] + 0.5) * self.scale_y - 0.5 - self.top
        return out


def input_transform(
    from_shape: tuple[int, int],
    size: int = MODEL_INPUT_SIZE,
    square_ok: bool = False,
) -> InputTransform:
    """Odtwarza `resize_img()` upstreamu dla klatki o kształcie `from_shape` = (H, W).

    Wierna reimplementacja `mast3r_utils.py:244-264` dla `size == 512`: dłuższy bok
    do `size`, potem crop centralny do wielokrotności 16 px. Zwraca `InputTransform`.
    """
    if size != 512:
        # mast3r_utils.py:245 dopuszcza jeszcze 224 (inna ścieżka: krótszy bok + crop
        # do kwadratu), ale silnik go nie używa — nie implementuję czegoś, czego nie
        # da się porównać z przebiegiem.
        raise ValueError(f"upstream używa wyłącznie size=512 (dataloader.py:25), dostałem {size}")

    h1, w1 = int(from_shape[0]), int(from_shape[1])
    if h1 <= 0 or w1 <= 0:
        raise ValueError(f"niepoprawny kształt klatki: {from_shape}")

    longest = max(w1, h1)
    w = int(round(w1 * size / longest))
    h = int(round(h1 * size / longest))

    cx, cy = w // 2, h // 2
    halfw = ((2 * cx) // 16) * 8
    halfh = ((2 * cy) // 16) * 8
    if not square_ok and w == h:
        halfh = 3 * halfw / 4  # mast3r_utils.py:263, przypadek kwadratowego wejścia

    # PIL.Image.crop zaokrągla współrzędne pudełka, stąd round() na wysokości.
    out_w = int(round(2 * halfw))
    out_h = int(round(2 * halfh))

    return InputTransform(
        scale_x=w / w1,
        scale_y=h / h1,
        left=float(cx - halfw),
        top=float(cy - halfh),
        resized_shape=(h, w),
        out_shape=(out_h, out_w),
    )


def rescale_corners(
    corners,
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
):
    """Narożniki z pełnej rozdzielczości -> rozdzielczość wejścia modelu.

    `corners` to albo pojedyncza tablica (..., 2), albo słownik {id: ndarray(4, 2)}
    — zwracany jest ten sam typ. `from_shape` to (H, W) pełnej klatki, `to_shape`
    to (H, W) wejścia modelu, czyli to, co silnik zwraca jako `true_shape`
    (`dataloader.get_img_shape()`, `dataloader.py:52-56`).

    `to_shape` nie wystarcza do odtworzenia cropu (offset jest w nim nie do odczytania),
    więc przekształcenie liczymy z `from_shape` przez `input_transform()`, a `to_shape`
    służy do sprawdzenia, czy to, co wyszło, zgadza się z tym, co ma silnik. Niezgodność
    to `ValueError`, nie ciche dopasowanie.
    """
    tf = input_transform(from_shape)
    expected = (int(to_shape[0]), int(to_shape[1]))
    if tf.out_shape != expected:
        raise ValueError(
            f"kształt wejścia modelu liczony z {tuple(from_shape)} to {tf.out_shape}, "
            f"a podano {expected} — sprawdź from_shape albo img_downsample w configu silnika"
        )
    if isinstance(corners, dict):
        return {int(k): tf.apply(v) for k, v in corners.items()}
    return tf.apply(corners)


# --------------------------------------------------------------------------- #
# Rysowanie i zapis
# --------------------------------------------------------------------------- #


def draw(image_bgr: np.ndarray, detections: dict[int, np.ndarray]) -> np.ndarray:
    """Kopia obrazu z narysowanymi narożnikami, ID i kierunkiem (lewy górny na czerwono)."""
    canvas = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    # Grubości skalowane rozmiarem obrazu — na 4K linia 1 px jest niewidoczna.
    thick = max(1, int(round(max(canvas.shape[:2]) / 900)))

    for marker_id, quad in sorted(detections.items()):
        pts = np.round(quad).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=thick)
        for i, (x, y) in enumerate(pts):
            color = (0, 0, 255) if i == 0 else (255, 0, 0)  # lewy górny czerwony
            cv2.circle(canvas, (int(x), int(y)), 2 * thick, color, -1)
        # Etykieta obok markera, nie na nim — wzór ma zostać widoczny do oceny okiem.
        center = quad.mean(axis=0)
        label_at = quad[0] + (quad[0] - center) * 0.25
        cv2.putText(
            canvas, str(marker_id), (int(label_at[0]), int(label_at[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * thick, (0, 0, 255), thick, cv2.LINE_AA,
        )
    return canvas


def to_json(
    detections: dict[int, np.ndarray],
    source: str,
    image_shape: tuple[int, int],
    dict_name: str,
    frame_index: Optional[int] = None,
) -> dict:
    """Wynik detekcji jako struktura do zapisu: narożniki pełne i przeliczone na wejście modelu."""
    tf = input_transform(image_shape[:2])
    markers = []
    for marker_id, quad in sorted(detections.items()):
        side = float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]))
        markers.append(
            {
                "id": marker_id,
                "corners": quad.tolist(),
                "corners_model_input": tf.apply(quad).tolist(),
                "mean_side_px": side,
                "mean_side_px_model_input": side * (tf.scale_x + tf.scale_y) / 2,
            }
        )
    return {
        "source": source,
        "frame": frame_index,
        "dict": dict_name,
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "corner_order": list(CORNER_ORDER),
        "model_input": asdict(tf) | {"size": MODEL_INPUT_SIZE},
        "n_detected": len(markers),
        "markers": markers,
    }


# --------------------------------------------------------------------------- #
# Rotacja z metadanych
# --------------------------------------------------------------------------- #


def parse_display_matrix(data: bytes) -> int:
    """Kąt obrotu [stopnie] z macierzy wyświetlania MP4; konwencja jak `ffprobe`.

    `data` to 36 bajtów side data typu DISPLAYMATRIX: dziewięć int32 little-endian
    w zapisie stałoprzecinkowym 16.16. Kąt liczymy jak `av_display_rotation_get()`
    w ffmpeg — `-atan2(m[1], m[0])` — i zaokrąglamy do wielokrotności 90 stopni,
    bo tylko takie obroty robi telefon, a tylko takie umie `cv2.rotate`.
    """
    if len(data) != 36:
        raise ValueError(f"display matrix ma 36 bajtów, dostałem {len(data)}")
    values = struct.unpack("<9i", data)
    a, b = values[0] / 65536.0, values[1] / 65536.0
    if a == 0.0 and b == 0.0:
        return 0
    snapped = int(round(-math.degrees(math.atan2(b, a)) / 90.0)) * 90
    snapped %= 360
    return snapped - 360 if snapped > 180 else snapped


def frame_rotation(frame) -> int:
    """Kąt rotacji klatki PyAV z jej side data; 0, gdy metadanych nie ma.

    Rotacja siedzi w kontenerze, nie w obrazie: PyAV wydaje piksele tak, jak leżą
    w pliku, a `cv2.VideoCapture` (czyli dekoder upstreamu) stosuje ją sam. Żeby
    obie ścieżki patrzyły na ten sam kadr, musimy ją odtworzyć po swojej stronie.
    """
    for side_data in frame.side_data:
        name = getattr(getattr(side_data, "type", None), "name", "") or str(getattr(side_data, "type", ""))
        if "DISPLAYMATRIX" in name.upper():
            return parse_display_matrix(bytes(side_data))
    return 0


def apply_rotation(image_bgr: np.ndarray, rotation: int) -> np.ndarray:
    """Obrót klatki do orientacji wyświetlania. `rotation` jak z `frame_rotation()`.

    Na długość boku markera w pikselach obrót nie ma wpływu (jest obrotem o wielokrotność
    90 stopni, bez interpolacji), ale na współrzędne narożników i na kształt klatki — ma.
    """
    if rotation == 0:
        return image_bgr
    try:
        return cv2.rotate(image_bgr, ROTATE_CODE[rotation])
    except KeyError:
        raise ValueError(f"nie umiem obrócić o {rotation} stopni; umiem {sorted(ROTATE_CODE)}") from None


def video_rotation(path: Path) -> int:
    """Kąt rotacji nagrania: dekoduje pierwszą klatkę i czyta jej display matrix."""
    import av  # lokalnie: bez wideo moduł nie potrzebuje PyAV

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            return frame_rotation(frame)
    raise ValueError(f"{path}: nie udało się zdekodować żadnej klatki")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def read_video_frame(path: Path, index: int, rotate: bool = True) -> np.ndarray:
    """Klatka nr `index` z pliku wideo, dekodowana sekwencyjnie PyAV (CLAUDE.md: bez seeka).

    Przy `rotate=True` stosujemy rotację z metadanych kontenera (`frame_rotation()`),
    czyli dajemy ten sam kadr, który widzi `cv2.VideoCapture` w dekoderze upstreamu.
    To jest narzędzie diagnostyczne — przy dużym `--frame` po prostu czeka.
    """
    import av  # lokalnie: bez --video moduł nie potrzebuje PyAV

    if index < 0:
        raise ValueError("--frame musi być >= 0")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i == index:
                image = frame.to_ndarray(format="bgr24")
                return apply_rotation(image, frame_rotation(frame)) if rotate else image
    raise ValueError(f"{path} ma mniej niż {index + 1} klatek")


def _label(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float,
           color=(255, 255, 255), bg=(0, 0, 0)) -> None:
    """Tekst na ciemnym prostokącie — czytelny niezależnie od tła sceny."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(canvas, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), bg, -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def render_video(
    video: Path,
    out_path: Path,
    dict_name: str = DEFAULT_DICT,
    width: int = 1920,
    min_px: float = 15.0,
    rotate: bool = True,
) -> dict:
    """Film podglądu z obrysami markerów na każdej klatce + CSV per klatka obok.

    Dekodowanie sekwencyjne PyAV, detekcja na pełnej klatce (`detect()`), liczby z
    `to_json()` — te same co w trybie pojedynczej klatki. Obrysy są rysowane dopiero na
    klatce zmniejszonej tak, żeby dłuższy bok miał `width`. Kolor to tylko wizualizacja
    progu `min_px` boku przy 512 px (zielony >= próg, czerwony poniżej); wartości są w CSV.

    Przy `rotate=True` (domyślnie) klatka jest obracana zgodnie z metadanymi kontenera,
    zanim pójdzie do detekcji — czyli pracujemy na tym kadrze, który widzi `cv2.VideoCapture`
    w dekoderze upstreamu. Rotację czytamy raz, z pierwszej klatki. Na bok markera w pikselach
    obrót nie wpływa; na współrzędne narożników i na kształt kadru — wpływa.
    """
    import av

    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    green, red = (60, 200, 60), (40, 40, 230)
    per_id: dict[int, dict] = {}
    rotation: Optional[int] = None
    n_frames, t0 = 0, time.monotonic()

    with av.open(str(video)) as src, av.open(str(out_path), "w") as dst, \
            open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        ist = src.streams.video[0]
        ist.thread_type = "AUTO"
        rate = Fraction(ist.average_rate).limit_denominator(1001) if ist.average_rate else Fraction(30)
        total = ist.frames or 0
        ost = None
        log = csv.writer(fcsv)
        log.writerow(["frame", "time_s", "id", "mean_side_px", "side_px_512", "error"])

        for i, frame in enumerate(src.decode(ist)):
            image = frame.to_ndarray(format="bgr24")
            if rotate:
                if rotation is None:
                    rotation = frame_rotation(frame)
                image = apply_rotation(image, rotation)
            h, w = image.shape[:2]
            if ost is None:
                shrink = min(1.0, width / max(w, h))
                out_w = int(round(w * shrink)) // 2 * 2
                out_h = int(round(h * shrink)) // 2 * 2
                ost = dst.add_stream("libx264", rate=rate)
                ost.width, ost.height, ost.pix_fmt = out_w, out_h, "yuv420p"
                ost.options = {"crf": "20", "preset": "veryfast"}
            t = frame.time if frame.time is not None else float(i / rate)

            error = ""
            try:
                detections = detect(image, dict_name=dict_name)
            except ValueError as exc:  # duplikat ID — nie przerywamy całego filmu
                detections, error = {}, str(exc)
            markers = to_json(detections, str(video), (h, w), dict_name, i)["markers"]

            canvas = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
            k = out_w / w
            for m in markers:
                side512 = m["mean_side_px_model_input"]
                color = green if side512 >= min_px else red
                pts = np.round(np.asarray(m["corners"]) * k).astype(np.int32)
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)
                cv2.circle(canvas, tuple(int(v) for v in pts[0]), 5, (0, 0, 255), -1)
                x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 8
                _label(canvas, f"ID {m['id']}  {side512:.1f} px przy 512", (max(x, 4), max(y, 24)), 0.7, color)
                log.writerow([i, f"{t:.4f}", m["id"], f"{m['mean_side_px']:.2f}", f"{side512:.2f}", error])
                stats = per_id.setdefault(m["id"], {"frames": 0, "min_512": side512, "max_512": side512, "below": 0})
                stats["frames"] += 1
                stats["min_512"] = min(stats["min_512"], side512)
                stats["max_512"] = max(stats["max_512"], side512)
                stats["below"] += side512 < min_px
            if not markers:
                log.writerow([i, f"{t:.4f}", "", "", "", error])

            hud = f"klatka {i}" + (f"/{total - 1}" if total else "") + f"  {t:6.2f} s  wykryto {len(markers)}"
            _label(canvas, hud, (12, 34), 0.9)
            _label(canvas, f"prog {min_px:g} px przy 512: zielony >=, czerwony <", (12, 72), 0.6)
            if error:
                _label(canvas, error, (12, 106), 0.6, red)

            vf = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            vf.pts, vf.time_base = i, 1 / rate
            for packet in ost.encode(vf):
                dst.mux(packet)
            n_frames = i + 1
            if n_frames % 100 == 0:
                print(f"  klatka {n_frames}" + (f"/{total}" if total else "")
                      + f"  ({time.monotonic() - t0:.0f} s)", flush=True)

        if ost is not None:
            for packet in ost.encode():
                dst.mux(packet)

    return {"frames": n_frames, "seconds": time.monotonic() - t0, "csv": csv_path,
            "per_id": per_id, "rotation": rotation or 0}


def main(argv: Optional[list[str]] = None) -> int:
    """Punkt wejścia `python -m vid2cloud.scale.aruco`.

    Argparse, a nie typer (CLAUDE.md), bo to jest wejście diagnostyczne uruchamiane
    w env `vid2cloud-tools`, gdzie pakiet nie jest instalowany — ma działać z samym
    opencv i numpy. Właściwy interfejs projektu, `vid2cloud/cli.py`, zostaje na typerze.
    """
    parser = argparse.ArgumentParser(
        prog="python -m vid2cloud.scale.aruco",
        description="Detekcja markerów ArUco na zdjęciu albo na klatce wideo.",
    )
    parser.add_argument("image", nargs="?", type=Path, help="Zdjęcie wejściowe (JPG/PNG).")
    parser.add_argument("--video", type=Path, help="Zamiast zdjęcia: plik wideo.")
    parser.add_argument("--frame", type=int, default=0, help="Numer klatki przy --video (od 0).")
    parser.add_argument("--dict", dest="dict_name", default=DEFAULT_DICT, help=f"Słownik (domyślnie {DEFAULT_DICT}).")
    parser.add_argument("--out", type=Path, help="Plik PNG/JPG z podglądem detekcji.")
    parser.add_argument("--json", dest="json_path", type=Path, help="Plik JSON; domyślnie obok --out.")
    parser.add_argument("--render", type=Path, metavar="OUT.mp4",
                        help="Z --video: film podglądu z obrysami na każdej klatce (+ CSV obok).")
    parser.add_argument("--width", type=int, default=1920,
                        help="Dłuższy bok filmu podglądu w px (domyślnie 1920).")
    parser.add_argument("--no-rotate", dest="rotate", action="store_false",
                        help="Nie stosuj rotacji z metadanych wideo (domyślnie stosujemy).")
    parser.add_argument("--min-px", type=float, default=15.0,
                        help="Próg boku przy 512 px dla koloru obrysu (domyślnie 15).")
    args = parser.parse_args(argv)

    if (args.image is None) == (args.video is None):
        parser.error("podaj albo zdjęcie, albo --video (dokładnie jedno)")

    if args.render is not None:
        if args.video is None:
            parser.error("--render wymaga --video")
        if not args.video.exists():
            parser.error(f"nie ma pliku {args.video}")
        print(f"{args.video} -> {args.render}")
        summary = render_video(args.video, args.render, args.dict_name, args.width, args.min_px, args.rotate)
        print(f"klatek: {summary['frames']}  czas: {summary['seconds']:.0f} s")
        print(f"rotacja z metadanych: {summary['rotation']} deg"
              + ("" if args.rotate else " (NIE zastosowana, --no-rotate)"))
        print(f"film:   {args.render}\nCSV:    {summary['csv']}")
        for marker_id, st in sorted(summary["per_id"].items()):
            print(f"  ID {marker_id:>3}  klatek {st['frames']:>5}  px@512 {st['min_512']:6.2f}-{st['max_512']:6.2f}"
                  f"  poniżej {args.min_px:g}: {st['below']}")
        return 0

    if args.video is not None:
        if not args.video.exists():
            parser.error(f"nie ma pliku {args.video}")
        try:
            image = read_video_frame(args.video, args.frame, args.rotate)
        except ValueError as exc:
            parser.error(str(exc))
        source, frame_index = str(args.video), args.frame
    else:
        if not args.image.exists():
            parser.error(f"nie ma pliku {args.image}")
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            parser.error(f"OpenCV nie wczytał {args.image}")
        source, frame_index = str(args.image), None

    detections = detect(image, dict_name=args.dict_name)
    result = to_json(detections, source, image.shape[:2], args.dict_name, frame_index)

    h, w = image.shape[:2]
    print(f"{source}" + (f" (klatka {frame_index})" if frame_index is not None else "") + f"  {w}x{h}")
    print(f"wykryto: {result['n_detected']}  słownik: {args.dict_name}")
    for m in result["markers"]:
        print(f"  ID {m['id']:>3}  bok {m['mean_side_px']:7.2f} px  -> {m['mean_side_px_model_input']:6.2f} px przy 512")
    if not detections:
        print("  (nic — sprawdź słownik, ostrość i czy marker nie jest za mały)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.out), draw(image, detections)):
            print(f"nie udało się zapisać {args.out}", file=sys.stderr)
            return 1
        print(f"podgląd: {args.out}")

    json_path = args.json_path or (args.out.with_suffix(".json") if args.out is not None else None)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON:     {json_path}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
