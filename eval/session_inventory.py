#!/usr/bin/env python3
"""Inwentaryzacja sesji pomiarowej: który marker to który ID, gdzie stał, ile razy go widać.

Wejście: CSV per klatka z `python -m vid2cloud.scale.aruco --video ... --render`
(katalogi `runs/<sesja>/<nazwa>/podglad.csv`) oraz zdjęcia z katalogu sesji.
Wyjście w `runs/<sesja>/inventory/`:

* `ids/ID_XX.jpg`   — dla każdego wykrytego ID kadr, w którym ten marker jest największy
                      (najbliższy widok), obwiedziony i podpisany;
* `overview/*.jpg`  — kadry z największą liczbą markerów naraz (pokazują rozstawienie);
* `ids_summary.csv` — zestawienie per ID: czy rozstawiony, czy wydrukowany, ile wykryć,
                      boki przy 512 px;
* `photos.csv`      — surowe wykrycia na zdjęciach z `photos_targets/` (cache detekcji).

Dodatkowo tabele do ręcznego uzupełnienia w `notes/` katalogu sesji i konwerter
`--to-targets`, który z uzupełnionego `pomiary_tasma.csv` robi `targets.json` w schemacie
czytanym przez `vid2cloud.scale.estimate.load_targets()`.

Deklaracja ID (`--ids 3,8,9,10,11,12`, zapamiętywana w `notes/rozstawione_id.txt`) mówi,
które markery **faktycznie stały w scenie**. Bez niej nie da się odróżnić fałszywego
wykrycia od prawdziwego: ID 7 jest w `boards.json`, ale w sesji 21.09 nie był rozstawiony,
a mimo to wyszedł raz na szybie okiennej z bokiem 40 px przy 512.

Jednostki. W `pomiary_tasma.csv` wpisuje się to, co pokazała taśma — liczbę w kolumnie
`odleglosc` i jednostkę (`mm`/`cm`/`m`) obok. Na milimetry, w których jest `targets.json`,
przelicza `to_mm()`, a nie człowiek w pamięci. `sigma_mm` jest zawsze w milimetrach.

Uwaga o liczbach: `bok512_*` i `udzial_ponizej_15px` liczymy po **wszystkich** wykryciach
danego ID — i z nagrań, i ze zdjęć — bo pytanie jest o to, jak marker wychodzi w materiale,
a nie o pojedyncze źródło. Liczby wykryć zostają rozbite na kolumny per nagranie i zdjęcia.

Env: `vid2cloud-tools` (cv2, numpy, av). Uruchamiać z katalogu repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vid2cloud.scale import aruco  # noqa: E402

#: Dłuższy bok obrazków w `ids/` i `overview/`.
VIEW_LONG_SIDE = 1600

#: Próg boku przy 512 px, względem którego liczymy odsetek małych wykryć.
MIN_PX_512 = 15.0

#: Ile najlepszych kadrów trafia do `overview/`.
OVERVIEW_COUNT = 5

#: Minimalny odstęp między kadrami z tego samego nagrania w `overview/` [klatki].
OVERVIEW_MIN_GAP = 30

#: Ile zdjęć z `photos_targets/` musi pokazywać marker, żeby nie był zgłoszony jako rzadki.
PHOTOS_FLAG_THRESHOLD = 5

#: Domyślna niepewność pomiaru taśmą [mm] wpisywana do tabeli.
DEFAULT_SIGMA_MM = 2.0

#: Jednostki dozwolone w kolumnie `jednostka` tabeli pomiarów, przelicznik na milimetry.
#: `targets.json` jest w milimetrach (schemat `scale/estimate.py`) — tu wpisuje się to,
#: co pokazała taśma, a przeliczenie robi konwerter, nie człowiek.
UNITS_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}

#: Jednostka wpisywana do świeżej tabeli. Notatki z taśmy są w centymetrach.
DEFAULT_UNIT = "cm"

#: Zakres długości odcinka wzorcowego, poza którym konwerter zgłasza podejrzenie
#: pomyłki w jednostce (1 cm – 50 m). Nie poprawia wartości, tylko ją wypisuje.
SUSPICIOUS_MM = (10.0, 50_000.0)

#: Plik z ID zadeklarowanymi przez autora jako faktycznie rozstawione w tej sesji.
DECLARED_IDS_FILE = "rozstawione_id.txt"

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --------------------------------------------------------------------------- #
# Struktury
# --------------------------------------------------------------------------- #


@dataclass
class Detection:
    """Jedno wykrycie markera: skąd, jak duży."""

    source: str          # nazwa nagrania albo pliku zdjęcia
    kind: str            # "video" albo "photo"
    frame: Optional[int] # numer klatki (None dla zdjęcia)
    marker_id: int
    side_px: float
    side_px_512: float


@dataclass
class ViewKey:
    """Kadr: plik + numer klatki (None dla zdjęcia)."""

    source: str
    kind: str
    frame: Optional[int]

    def key(self) -> tuple:
        return (self.kind, self.source, self.frame)

    def label(self) -> str:
        return self.source if self.frame is None else f"{self.source}  klatka {self.frame}"

    def slug(self) -> str:
        return Path(self.source).stem if self.frame is None else f"{Path(self.source).stem}_f{self.frame}"


@dataclass
class Inventory:
    """Wszystko, co wyszło z CSV-ek i zdjęć."""

    detections: list[Detection] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    photos: list[str] = field(default_factory=list)

    def by_id(self) -> dict[int, list[Detection]]:
        out: dict[int, list[Detection]] = defaultdict(list)
        for d in self.detections:
            out[d.marker_id].append(d)
        return dict(out)

    def by_view(self) -> dict[tuple, list[Detection]]:
        """Wykrycia pogrupowane po kadrze — do liczenia, gdzie widać najwięcej ID."""
        out: dict[tuple, list[Detection]] = defaultdict(list)
        for d in self.detections:
            out[(d.kind, d.source, d.frame)].append(d)
        return dict(out)


# --------------------------------------------------------------------------- #
# Wejście: wydrukowane ID
# --------------------------------------------------------------------------- #


def load_boards(boards_dir: Path) -> dict[int, dict]:
    """ID faktycznie wydrukowane, z wszystkich `boards/*/boards.json`.

    Zwraca {id: {"nominal_size_mm": float, "board": nazwa katalogu, "dict": nazwa}}.
    Ten sam ID może siedzieć w dwóch zestawach (np. 3 w `test_A4_150mm` i `boards_A3_120mm`)
    — wtedy zapamiętujemy wszystkie zestawy, bo to jest informacja dla autora, nie błąd.
    """
    out: dict[int, dict] = {}
    for path in sorted(boards_dir.glob("*/boards.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        board = path.parent.name
        for entry in data.get("markers", []):
            marker_id = int(entry["id"])
            size = float(entry.get("nominal_size_mm", data.get("nominal_size_mm", 0.0)))
            if marker_id in out:
                out[marker_id]["board"] += f"+{board}"
                out[marker_id]["nominal_size_mm_all"].append(size)
                continue
            out[marker_id] = {
                "nominal_size_mm": size,
                "nominal_size_mm_all": [size],
                "board": board,
                "dict": str(entry.get("dict", data.get("dict", aruco.DEFAULT_DICT))),
            }
    return out


def parse_declared_ids(text: str) -> set[int]:
    """ID zadeklarowane przez autora: liczby po przecinkach lub spacjach, `#` to komentarz.

    Czysta funkcja, żeby format deklaracji dało się przetestować bez pliku na dysku.
    """
    out: set[int] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for token in line.replace(",", " ").split():
            try:
                out.add(int(token))
            except ValueError:
                raise ValueError(f"deklaracja ID: {token!r} nie jest liczbą") from None
    if not out:
        raise ValueError("deklaracja ID jest pusta")
    return out


def read_declared_ids(path: Path) -> Optional[set[int]]:
    """Deklaracja z pliku; `None`, gdy pliku nie ma (nie jest to błąd, tylko brak deklaracji)."""
    if not path.exists():
        return None
    return parse_declared_ids(path.read_text(encoding="utf-8"))


def write_declared_ids(ids: set[int], path: Path, session: str = "") -> None:
    """Zapis deklaracji, żeby kolejne uruchomienie nie wymagało powtarzania `--ids`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "# ID markerow faktycznie rozstawionych w tej sesji — deklaracja autora.\n"
    if session:
        head += f"# sesja: {session}\n"
    head += "# Wszystko spoza tej listy jest w zestawieniu traktowane jak falszywe wykrycie.\n"
    path.write_text(head + ", ".join(str(i) for i in sorted(ids)) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Wejście: CSV per klatka z nagrań
# --------------------------------------------------------------------------- #


def read_video_csv(path: Path, source: str) -> list[Detection]:
    """Wykrycia z `podglad.csv` (format z `aruco.render_video`). Wiersze bez ID pomijamy."""
    out: list[Detection] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("id"):
                continue
            out.append(
                Detection(
                    source=source,
                    kind="video",
                    frame=int(row["frame"]),
                    marker_id=int(row["id"]),
                    side_px=float(row["mean_side_px"]),
                    side_px_512=float(row["side_px_512"]),
                )
            )
    return out


def read_runs(runs_dir: Path) -> tuple[list[Detection], list[str]]:
    """Wszystkie `*/podglad.csv` w katalogu przebiegów sesji."""
    detections: list[Detection] = []
    videos: list[str] = []
    for csv_path in sorted(runs_dir.glob("*/podglad.csv")):
        source = csv_path.parent.name
        videos.append(source)
        detections.extend(read_video_csv(csv_path, source))
    return detections, videos


# --------------------------------------------------------------------------- #
# Wejście: detekcja na zdjęciach
# --------------------------------------------------------------------------- #


def detect_photos(photos_dir: Path, dict_name: str = aruco.DEFAULT_DICT,
                  progress_every: int = 10) -> list[dict]:
    """Detekcja ArUco na każdym zdjęciu z katalogu; pełna rozdzielczość, jak w nagraniach.

    Zwraca listę wierszy do `photos.csv`: plik, ID, bok w px, bok przy 512, błąd.
    Zdjęcie bez wykrycia dostaje jeden wiersz z pustym ID — żeby było widać, że było sprawdzone.
    """
    rows: list[dict] = []
    files = sorted(p for p in photos_dir.iterdir() if p.suffix in PHOTO_SUFFIXES)
    for n, path in enumerate(files, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append({"file": path.name, "id": "", "mean_side_px": "", "side_px_512": "",
                         "error": "OpenCV nie wczytał pliku"})
            continue
        error = ""
        try:
            detections = aruco.detect(image, dict_name=dict_name)
        except ValueError as exc:  # duplikat ID na jednym zdjęciu
            detections, error = {}, str(exc)
        result = aruco.to_json(detections, path.name, image.shape[:2], dict_name, None)
        if not result["markers"]:
            rows.append({"file": path.name, "id": "", "mean_side_px": "", "side_px_512": "", "error": error})
        for m in result["markers"]:
            rows.append({
                "file": path.name,
                "id": m["id"],
                "mean_side_px": f"{m['mean_side_px']:.2f}",
                "side_px_512": f"{m['mean_side_px_model_input']:.2f}",
                "error": error,
            })
        if progress_every and n % progress_every == 0:
            print(f"  zdjęcie {n}/{len(files)}", flush=True)
    return rows


def write_photo_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "id", "mean_side_px", "side_px_512", "error"])
        w.writeheader()
        w.writerows(rows)


def photo_rows_to_detections(rows: Iterable[dict]) -> tuple[list[Detection], list[str]]:
    out: list[Detection] = []
    files: list[str] = []
    for row in rows:
        if row["file"] not in files:
            files.append(row["file"])
        if not str(row.get("id", "")).strip():
            continue
        out.append(
            Detection(
                source=row["file"],
                kind="photo",
                frame=None,
                marker_id=int(row["id"]),
                side_px=float(row["mean_side_px"]),
                side_px_512=float(row["side_px_512"]),
            )
        )
    return out, files


# --------------------------------------------------------------------------- #
# Wybór kadrów
# --------------------------------------------------------------------------- #


def best_view_per_id(inv: Inventory) -> dict[int, tuple[ViewKey, Detection]]:
    """Dla każdego ID kadr, w którym ten marker ma największy bok (najbliższy widok)."""
    best: dict[int, tuple[ViewKey, Detection]] = {}
    for d in inv.detections:
        current = best.get(d.marker_id)
        if current is None or d.side_px_512 > current[1].side_px_512:
            best[d.marker_id] = (ViewKey(d.source, d.kind, d.frame), d)
    return best


def overview_views(inv: Inventory, count: int = OVERVIEW_COUNT,
                   min_gap: int = OVERVIEW_MIN_GAP) -> list[tuple[ViewKey, list[int]]]:
    """Kadry z największą liczbą różnych ID naraz.

    Kadry z tego samego nagrania muszą być od siebie dalej niż `min_gap` klatek, inaczej
    dostalibyśmy pięć razy tę samą scenę. Kadr, którego zestaw ID jest podzbiorem już
    wybranego, pomijamy — nie wnosi nic nowego o rozstawieniu.
    """
    candidates = []
    for (kind, source, frame), dets in inv.by_view().items():
        ids = sorted({d.marker_id for d in dets})
        candidates.append((len(ids), ViewKey(source, kind, frame), ids))
    candidates.sort(key=lambda c: (-c[0], c[1].source, c[1].frame if c[1].frame is not None else -1))

    chosen: list[tuple[ViewKey, list[int]]] = []
    for _, view, ids in candidates:
        if len(chosen) >= count:
            break
        too_close = any(
            v.kind == view.kind and v.source == view.source
            and v.frame is not None and view.frame is not None
            and abs(v.frame - view.frame) <= min_gap
            for v, _ in chosen
        )
        if too_close:
            continue
        if any(set(ids) <= set(prev_ids) for _, prev_ids in chosen):
            continue
        chosen.append((view, ids))
    return chosen


def extract_frames(video_path: Path, indices: Iterable[int], rotate: bool = True) -> dict[int, np.ndarray]:
    """Wybrane klatki z nagrania, jednym przebiegiem sekwencyjnym (CLAUDE.md: bez seeka).

    Rotację z metadanych stosujemy tak jak `aruco.render_video`, więc numery klatek
    i boki zgadzają się z CSV-ką podglądu.
    """
    import av

    want = set(int(i) for i in indices)
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rotation: Optional[int] = None
        for i, frame in enumerate(container.decode(stream)):
            if rotation is None:
                rotation = aruco.frame_rotation(frame)
            if i in want:
                image = frame.to_ndarray(format="bgr24")
                out[i] = aruco.apply_rotation(image, rotation) if rotate else image
                want.discard(i)
                if not want:
                    break
    return out


# --------------------------------------------------------------------------- #
# Rysowanie
# --------------------------------------------------------------------------- #


def _label(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float,
           color=(255, 255, 255), bg=(0, 0, 0)) -> None:
    """Tekst na ciemnym prostokącie — czytelny niezależnie od tła sceny.

    Etykieta jest dociskana do kadru: marker przy prawej krawędzi ma podpis przesunięty
    w lewo, zamiast uciętego na brzegu obrazka.
    """
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    h, w = canvas.shape[:2]
    x = min(max(org[0], 4), max(4, w - tw - 6))
    y = min(max(org[1], th + 8), h - base - 4)
    cv2.rectangle(canvas, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), bg, -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def render_view(image: np.ndarray, caption: list[str], highlight: Optional[int] = None,
                dict_name: str = aruco.DEFAULT_DICT, long_side: int = VIEW_LONG_SIDE) -> np.ndarray:
    """Kadr zmniejszony do `long_side` z obwódkami markerów, ID i podpisem w rogu.

    Detekcję powtarzamy na pełnej klatce (CSV nie trzyma narożników), a obrysy rysujemy
    już na zmniejszonym obrazie — inaczej po zmniejszeniu linia znika. `highlight` dostaje
    grubszą, żółtą obwódkę; pozostałe markery są zielone, dla kontekstu.
    """
    h, w = image.shape[:2]
    k = min(1.0, long_side / max(h, w))
    canvas = cv2.resize(image, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)

    try:
        detections = aruco.detect(image, dict_name=dict_name)
    except ValueError:
        detections = {}
    markers = aruco.to_json(detections, "", image.shape[:2], dict_name, None)["markers"]

    for m in markers:
        is_target = highlight is not None and m["id"] == highlight
        color = (40, 230, 230) if is_target else (60, 200, 60)
        pts = np.round(np.asarray(m["corners"]) * k).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=4 if is_target else 2)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 8
        _label(canvas, f"ID {m['id']}  {m['mean_side_px_model_input']:.1f} px przy 512",
               (max(x, 4), max(y, 26)), 0.8 if is_target else 0.6, color)

    y = 32
    for line in caption:
        _label(canvas, line, (12, y), 0.7)
        y += 34
    return canvas


# --------------------------------------------------------------------------- #
# Zestawienie per ID
# --------------------------------------------------------------------------- #


def summary_rows(inv: Inventory, boards: dict[int, dict], min_px: float = MIN_PX_512,
                 declared: Optional[set[int]] = None) -> list[dict]:
    """Wiersze `ids_summary.csv`: jeden na każdy wykryty ID.

    `declared` to lista ID, które autor zadeklarował jako rozstawione w tej sesji. Kolumna
    `zadeklarowany` jest ważniejsza od `w_boards`: ID 7 był wydrukowany, ale nie wyszedł
    w teren, więc jego wykrycia są fałszywe mimo wpisu w `boards.json`.
    """
    best = best_view_per_id(inv)
    rows = []
    for marker_id, dets in sorted(inv.by_id().items()):
        sides = [d.side_px_512 for d in dets]
        per_video = Counter(d.source for d in dets if d.kind == "video")
        photos = {d.source for d in dets if d.kind == "photo"}
        board = boards.get(marker_id)
        row = {
            "id": marker_id,
            "zadeklarowany": "?" if declared is None else ("tak" if marker_id in declared else "nie"),
            "w_boards": "tak" if board else "nie",
            "wzorzec": board["board"] if board else "",
            "nominal_mm": f"{board['nominal_size_mm']:.1f}" if board else "",
            "zdjec_targets": len(photos),
            "klatek_razem": sum(per_video.values()),
            "bok512_sredni": f"{float(np.mean(sides)):.2f}",
            "bok512_min": f"{min(sides):.2f}",
            "bok512_max": f"{max(sides):.2f}",
            f"udzial_ponizej_{min_px:g}px": f"{sum(s < min_px for s in sides) / len(sides):.3f}",
            "najlepszy_widok": best[marker_id][0].label(),
        }
        for video in inv.videos:
            row[f"klatek_{video}"] = per_video.get(video, 0)
        rows.append(row)
    return rows


def write_summary(rows: list[dict], path: Path, videos: list[str], min_px: float = MIN_PX_512) -> None:
    fields = ["id", "zadeklarowany", "w_boards", "wzorzec", "nominal_mm", "zdjec_targets", "klatek_razem"] \
        + [f"klatek_{v}" for v in videos] \
        + ["bok512_sredni", "bok512_min", "bok512_max", f"udzial_ponizej_{min_px:g}px", "najlepszy_widok"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rare_in_photos(inv: Inventory, expected: Iterable[int],
                   threshold: int = PHOTOS_FLAG_THRESHOLD) -> list[tuple[int, int]]:
    """Oczekiwane ID widoczne na mniej niż `threshold` zdjęciach z `photos_targets/`.

    `expected` to deklaracja autora, a gdy jej nie ma — wszystkie wydrukowane ID. Różnica
    jest praktyczna: bez deklaracji flaga wypisuje cały zestaw A3, którego w sesji nie było.
    """
    photos_per_id: dict[int, set[str]] = defaultdict(set)
    for d in inv.detections:
        if d.kind == "photo":
            photos_per_id[d.marker_id].add(d.source)
    return [(marker_id, len(photos_per_id.get(marker_id, ()))) for marker_id in sorted(expected)
            if len(photos_per_id.get(marker_id, ())) < threshold]


# --------------------------------------------------------------------------- #
# Tabele do ręcznego uzupełnienia
# --------------------------------------------------------------------------- #


def video_metadata(path: Path) -> dict:
    """Metadane nagrania z PyAV + rotacja z display matrix (`aruco.video_rotation`)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(Fraction(stream.average_rate)) if stream.average_rate else float("nan")
        duration = float(stream.duration * stream.time_base) if stream.duration else float("nan")
        meta = {
            "plik": path.name,
            "czas_s": f"{duration:.2f}",
            "rozdzielczosc": f"{stream.codec_context.width}x{stream.codec_context.height}",
            "fps": f"{rate:.3f}",
            "kodek": stream.codec_context.name,
            "klatek": stream.frames or "",
        }
    rotation = aruco.video_rotation(path)
    meta["rotacja"] = rotation
    w, h = (int(v) for v in meta["rozdzielczosc"].split("x"))
    meta["rozdzielczosc_po_rotacji"] = f"{h}x{w}" if rotation % 180 else f"{w}x{h}"
    return meta


NAGRANIA_FIELDS = ["plik", "nazwa", "czas_s", "rozdzielczosc", "rozdzielczosc_po_rotacji", "fps",
                   "kodek", "klatek", "rotacja", "typ", "start_rowny_koncowi", "opis_trasy", "uwagi"]

#: Kolumny `nagrania.csv` wypełniane ręcznie — przy ponownym uruchomieniu przenoszone
#: ze starej wersji pliku, żeby skrypt nie kasował pracy autora. `nazwa` to czytelny alias
#: nagrania (np. `daleko_wolno`) używany na wykresach i w raportach. Pliki zostają pod
#: nazwami z telefonu, bo te siedzą w config.json, logach upstreamu i podpisach obrazków.
NAGRANIA_MANUAL = ("nazwa", "typ", "start_rowny_koncowi", "opis_trasy", "uwagi")


def read_existing(path: Path, key: tuple[str, ...]) -> dict[tuple, dict]:
    """Poprzednia wersja tabeli, zaindeksowana kluczem. Brak pliku to pusty słownik."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {tuple(str(row.get(k, "")).strip() for k in key): row for row in csv.DictReader(f)}


def carry_over(row: dict, old: dict[tuple, dict], key: tuple[str, ...],
               columns: Iterable[str]) -> dict:
    """Przepisanie ręcznie uzupełnionych kolumn ze starej wersji wiersza."""
    previous = old.get(tuple(str(row.get(k, "")).strip() for k in key))
    if previous:
        for column in columns:
            if str(previous.get(column, "")).strip():
                row[column] = previous[column]
    return row


def write_nagrania(video_dir: Path, path: Path) -> list[dict]:
    """`nagrania.csv`: metadane z pliku + kolumny do opisania trasy ręcznie.

    Kolumny ręczne z poprzedniej wersji pliku są przepisywane — tak samo jak
    `eval/test_c_start_point.py` robi z kolumnami pomiarowymi w `results.csv`.
    """
    old = read_existing(path, ("plik",))
    rows = []
    for video in sorted(video_dir.glob("*.mp4")):
        row = video_metadata(video)
        row.update({column: "" for column in NAGRANIA_MANUAL})
        rows.append(carry_over(row, old, ("plik",), NAGRANIA_MANUAL))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NAGRANIA_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def covisible_pairs(inv: Inventory, only_ids: Optional[set[int]] = None) -> dict[tuple[int, int], Counter]:
    """Pary ID widoczne w tym samym kadrze: ile razy razem w klatkach, ile razy na zdjęciach.

    Funkcja czysta (liczy na strukturach, nie na plikach) — jest pod testem.
    """
    out: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for (kind, _source, _frame), dets in inv.by_view().items():
        ids = sorted({d.marker_id for d in dets}
                     if only_ids is None else {d.marker_id for d in dets if d.marker_id in only_ids})
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                out[(a, b)]["klatek" if kind == "video" else "zdjec"] += 1
    return dict(out)


POMIARY_FIELDS = ["id_a", "id_b", "wspolwidocznosc_klatek", "wspolwidocznosc_zdjec",
                  "odleglosc", "jednostka", "sigma_mm", "punkty_pomiaru", "uwagi"]

#: Kolumny `pomiary_tasma.csv` wypełniane ręcznie — przenoszone przy ponownym uruchomieniu.
POMIARY_MANUAL = ("odleglosc", "jednostka", "sigma_mm", "punkty_pomiaru", "uwagi")


def write_pomiary(inv: Inventory, expected: Iterable[int], path: Path,
                  all_pairs: bool = False) -> list[dict]:
    """`pomiary_tasma.csv`: wiersz na parę ID do wpisania odległości z taśmy.

    `expected` to ID zadeklarowane jako rozstawione — pary z ID spoza tej listy nie mają
    po co powstawać, bo nie ma czego mierzyć. Domyślnie wypisujemy tylko pary
    **współwidoczne** (gdzieś w materiale oba markery są w jednym kadrze), posortowane po
    współwidoczności; `all_pairs=True` daje pełny iloczyn. Wiersze już uzupełnione ręcznie
    zostają — także wtedy, gdy para przestała być współwidoczna po zmianie deklaracji.
    """
    expected = set(expected)
    usable = {d.marker_id for d in inv.detections} & expected
    counts = covisible_pairs(inv, only_ids=usable)
    old = read_existing(path, ("id_a", "id_b"))

    if all_pairs:
        ordered = sorted(usable)
        pairs = [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:]]
    else:
        pairs = sorted(counts, key=lambda p: (-(counts[p]["zdjec"] + counts[p]["klatek"]), p))

    # Pary bez współwidoczności, ale z wpisanym pomiarem — autor zmierzył, więc zostają.
    filled = {
        (int(row["id_a"]), int(row["id_b"]))
        for key, row in old.items()
        if str(row.get("odleglosc", row.get("odleglosc_mm", ""))).strip() and key[0] and key[1]
    }
    pairs += sorted(filled - set(pairs))

    rows = []
    for a, b in pairs:
        c = counts.get((a, b), Counter())
        row = {
            "id_a": a,
            "id_b": b,
            "wspolwidocznosc_klatek": c["klatek"],
            "wspolwidocznosc_zdjec": c["zdjec"],
            "odleglosc": "",
            "jednostka": DEFAULT_UNIT,
            "sigma_mm": f"{DEFAULT_SIGMA_MM:g}",
            "punkty_pomiaru": "",
            "uwagi": "",
        }
        rows.append(carry_over(row, old, ("id_a", "id_b"), POMIARY_MANUAL))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=POMIARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


# --------------------------------------------------------------------------- #
# Konwerter: pomiary_tasma.csv -> targets.json
# --------------------------------------------------------------------------- #


def to_mm(value: str, unit: str) -> float:
    """Liczba z tabeli na milimetry. Przecinek dziesiętny dozwolony, jednostka obowiązkowa.

    Osobna funkcja czysta, bo to jedyne miejsce, w którym w całym module zmienia się
    jednostka — a pomyłka tutaj przechodzi prosto do współczynnika skali.
    """
    text = str(value).strip().replace(",", ".")
    unit = str(unit).strip().lower()
    if unit not in UNITS_MM:
        raise ValueError(f"nieznana jednostka {unit!r}; dozwolone: {', '.join(UNITS_MM)}")
    try:
        return float(text) * UNITS_MM[unit]
    except ValueError:
        raise ValueError(f"{value!r} nie jest liczbą") from None


def suspicious_lengths(segments: Iterable[dict], limits: tuple[float, float] = SUSPICIOUS_MM) -> list[dict]:
    """Segmenty poza rozsądnym zakresem długości — typowy objaw pomylonej jednostki.

    Zwraca je do wypisania; wartości nie poprawiamy i wiersza nie odrzucamy (CLAUDE.md,
    zasada 4: dziwny wynik się zgłasza, a nie „naprawia").
    """
    low, high = limits
    return [s for s in segments if not (low <= s["length_mm"] <= high)]


def pomiary_to_targets(rows: Iterable[dict], boards: Optional[dict[int, dict]] = None,
                       dict_name: str = aruco.DEFAULT_DICT, date: str = "",
                       measured_by: str = "tasma", source: str = "",
                       declared: Optional[set[int]] = None) -> dict:
    """Słownik w schemacie `targets.json` z wierszy `pomiary_tasma.csv`.

    Wiersze bez odległości pomijamy — mierzona była tylko część par. Odległość czytamy
    z kolumny `odleglosc` razem z `jednostka` (mm/cm/m) i przeliczamy na milimetry, bo
    `targets.json` jest w milimetrach; starszy nagłówek `odleglosc_mm` też jest przyjmowany.
    `punkty_pomiaru` idzie do pola `reference`, bo to ono mówi, w co klikać w CloudCompare.
    Nominały z `boards.json` wchodzą wyłącznie jako informacja o markerze (`nominal_size_mm`),
    nigdy jako `side_mm` — wydruk nie jest wzorcem skali (docs/decisions.md, 2026-09-15).

    `declared` (ID rozstawione w sesji) jest sprawdzane twardo: wzorzec skali oparty na
    markerze, którego w scenie nie było, to nie jest coś, co wolno przepuścić dalej.
    """
    segments = []
    used_ids: set[int] = set()
    for i, row in enumerate(rows, start=1):
        legacy = "odleglosc" not in row and "odleglosc_mm" in row
        raw = str(row.get("odleglosc", row.get("odleglosc_mm", ""))).strip()
        if not raw:
            continue
        unit = "mm" if legacy else str(row.get("jednostka", "")).strip() or DEFAULT_UNIT
        try:
            length_mm = to_mm(raw, unit)
        except ValueError as exc:
            raise ValueError(f"wiersz {i}: odległość — {exc}") from None
        try:
            a, b = int(str(row["id_a"]).strip()), int(str(row["id_b"]).strip())
        except (KeyError, ValueError):
            raise ValueError(f"wiersz {i}: brak poprawnych 'id_a'/'id_b'") from None
        if declared is not None and not {a, b} <= declared:
            obce = sorted({a, b} - declared)
            raise ValueError(
                f"wiersz {i}: ID {obce} nie są na liście rozstawionych "
                f"({sorted(declared)}). Popraw wiersz albo deklarację (--ids)."
            )

        sigma_raw = str(row.get("sigma_mm", "")).strip().replace(",", ".")
        sigma_mm = float(sigma_raw) if sigma_raw else DEFAULT_SIGMA_MM
        segments.append({
            "name": f"{min(a, b)}-{max(a, b)}",
            "length_mm": length_mm,
            "sigma_mm": sigma_mm,
            "kind": "aruco",
            "marker_a": min(a, b),
            "marker_b": max(a, b),
            "reference": str(row.get("punkty_pomiaru", "")).strip(),
            "measured_by": measured_by,
            "date": date,
        })
        used_ids.update((a, b))

    if not segments:
        raise ValueError("żaden wiersz nie ma wpisanej 'odleglosc' — nie ma z czego zrobić targets.json")

    markers = {}
    for marker_id in sorted(used_ids):
        entry = (boards or {}).get(marker_id)
        if entry is None:
            continue
        markers[str(marker_id)] = {
            "nominal_size_mm": entry["nominal_size_mm"],
            "board": entry["board"],
            "uwaga": "nominał wydruku, NIE wzorzec skali",
        }
    out = {"dict": dict_name, "markers": markers, "segments": segments}
    if source:
        out["source_csv"] = source
    return out


def read_pomiary_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Przebieg inwentaryzacji
# --------------------------------------------------------------------------- #


def build_inventory(runs_dir: Path, photos_dir: Optional[Path], out_dir: Path,
                    dict_name: str = aruco.DEFAULT_DICT, redetect: bool = False) -> Inventory:
    """CSV-ki nagrań + detekcja na zdjęciach (z cache w `photos.csv`)."""
    detections, videos = read_runs(runs_dir)
    print(f"nagrania: {len(videos)}  wykryć w CSV: {len(detections)}")

    photos: list[str] = []
    photos_csv = out_dir / "photos.csv"
    if photos_dir is not None and photos_dir.is_dir():
        if photos_csv.exists() and not redetect:
            print(f"zdjęcia: czytam gotowe {photos_csv} (--redetect-photos, żeby policzyć od nowa)")
            with open(photos_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            print(f"zdjęcia: detekcja w {photos_dir}")
            rows = detect_photos(photos_dir, dict_name=dict_name)
            write_photo_rows(rows, photos_csv)
        photo_dets, photos = photo_rows_to_detections(rows)
        detections.extend(photo_dets)
        print(f"zdjęcia: {len(photos)}  wykryć: {len(photo_dets)}")

    return Inventory(detections=detections, videos=videos, photos=photos)


def write_views(inv: Inventory, video_dir: Path, photos_dir: Optional[Path], out_dir: Path,
                dict_name: str = aruco.DEFAULT_DICT) -> tuple[int, int]:
    """Obrazki `ids/ID_XX.jpg` i `overview/*.jpg`. Każde nagranie czytane jednym przebiegiem."""
    best = best_view_per_id(inv)
    overview = overview_views(inv)

    # Co trzeba wyciągnąć z którego nagrania — zbieramy najpierw, dekodujemy raz.
    needed: dict[str, set[int]] = defaultdict(set)
    for view, _ in list(best.values()) + [(v, None) for v, _ in overview]:
        if view.kind == "video":
            needed[view.source].add(int(view.frame))

    frames: dict[tuple[str, int], np.ndarray] = {}
    for source, indices in needed.items():
        path = video_dir / f"{source}.mp4"
        if not path.exists():
            print(f"UWAGA: nie ma {path} — pomijam kadry {sorted(indices)}", file=sys.stderr)
            continue
        print(f"wyciągam {len(indices)} klatek z {path.name}: {sorted(indices)}", flush=True)
        for index, image in extract_frames(path, indices).items():
            frames[(source, index)] = image

    def image_for(view: ViewKey) -> Optional[np.ndarray]:
        if view.kind == "video":
            return frames.get((view.source, int(view.frame)))
        if photos_dir is None:
            return None
        return cv2.imread(str(photos_dir / view.source), cv2.IMREAD_COLOR)

    ids_dir, overview_dir = out_dir / "ids", out_dir / "overview"
    ids_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)

    n_ids = 0
    for marker_id, (view, det) in sorted(best.items()):
        image = image_for(view)
        if image is None:
            print(f"UWAGA: brak obrazu dla ID {marker_id} ({view.label()})", file=sys.stderr)
            continue
        caption = [
            f"ID {marker_id}   bok {det.side_px_512:.1f} px przy 512   ({det.side_px:.0f} px na pelnej klatce)",
            view.label(),
        ]
        canvas = render_view(image, caption, highlight=marker_id, dict_name=dict_name)
        cv2.imwrite(str(ids_dir / f"ID_{marker_id:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_ids += 1

    n_overview = 0
    for rank, (view, ids) in enumerate(overview, start=1):
        image = image_for(view)
        if image is None:
            print(f"UWAGA: brak obrazu dla overview {view.label()}", file=sys.stderr)
            continue
        caption = [f"{len(ids)} markerow: {', '.join(str(i) for i in ids)}", view.label()]
        canvas = render_view(image, caption, dict_name=dict_name)
        cv2.imwrite(str(overview_dir / f"{rank:02d}_{view.slug()}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_overview += 1
    return n_ids, n_overview


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python eval/session_inventory.py",
        description="Inwentaryzacja sesji pomiarowej: obrazki per ID, zestawienie, tabele.",
    )
    parser.add_argument("--session", type=Path, default=Path("data/2026-09-21"),
                        help="Katalog sesji (video/, photos_targets/, notes/...).")
    parser.add_argument("--runs", type=Path, default=Path("runs/2026-09-21"),
                        help="Katalog przebiegów z podglądami ArUco (*/podglad.csv).")
    parser.add_argument("--boards", type=Path, default=Path("boards"),
                        help="Katalog z wydrukami (*/boards.json).")
    parser.add_argument("--dict", dest="dict_name", default=aruco.DEFAULT_DICT, help="Słownik ArUco.")
    parser.add_argument("--ids", help="ID faktycznie rozstawione w sesji, np. 3,8,9,10,11,12. "
                                      f"Zapisywane do notes/{DECLARED_IDS_FILE} i stamtąd czytane "
                                      "przy kolejnych uruchomieniach.")
    parser.add_argument("--ids-file", type=Path,
                        help=f"Plik z deklaracją ID (domyślnie <sesja>/notes/{DECLARED_IDS_FILE}).")
    parser.add_argument("--min-px", type=float, default=MIN_PX_512,
                        help=f"Próg boku przy 512 px w zestawieniu (domyślnie {MIN_PX_512:g}).")
    parser.add_argument("--redetect-photos", action="store_true",
                        help="Policz detekcję na zdjęciach od nowa, nie czytaj photos.csv.")
    parser.add_argument("--no-views", action="store_true", help="Pomiń obrazki ids/ i overview/.")
    parser.add_argument("--all-pairs", action="store_true",
                        help="W pomiary_tasma.csv wypisz wszystkie pary, nie tylko współwidoczne.")
    parser.add_argument("--to-targets", type=Path, metavar="POMIARY.CSV",
                        help="Tylko konwersja uzupełnionego pomiary_tasma.csv na targets.json.")
    parser.add_argument("--out", type=Path, help="Plik wyjściowy targets.json (z --to-targets).")
    parser.add_argument("--date", default="", help="Data pomiaru wpisywana do targets.json (RRRR-MM-DD).")
    args = parser.parse_args(argv)

    boards = load_boards(args.boards)
    ids_file = args.ids_file or (args.session / "notes" / DECLARED_IDS_FILE)
    try:
        declared = parse_declared_ids(args.ids) if args.ids else read_declared_ids(ids_file)
    except ValueError as exc:
        parser.error(str(exc))
    if args.ids:
        write_declared_ids(declared, ids_file, session=args.session.name)
        print(f"zadeklarowane ID: {sorted(declared)} -> {ids_file}")
    elif declared is not None:
        print(f"zadeklarowane ID (z {ids_file}): {sorted(declared)}")
    else:
        print(f"UWAGA: nie ma deklaracji ID (--ids albo {ids_file}). Za rozstawione uznaję "
              f"wszystko z boards.json: {sorted(boards)}", file=sys.stderr)
    expected = declared if declared is not None else set(boards)

    if args.to_targets is not None:
        if not args.to_targets.exists():
            parser.error(f"nie ma pliku {args.to_targets}")
        rows = read_pomiary_csv(args.to_targets)
        try:
            targets = pomiary_to_targets(rows, boards, args.dict_name, args.date,
                                         source=str(args.to_targets), declared=declared)
        except ValueError as exc:
            print(f"BŁĄD: {exc}", file=sys.stderr)
            return 1
        out = args.out or args.to_targets.with_name("targets.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(targets, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{args.to_targets} -> {out}")
        print(f"segmentów: {len(targets['segments'])}"
              f"  (pominięto wierszy bez odległości: {len(rows) - len(targets['segments'])})")
        for seg in targets["segments"]:
            print(f"  {seg['name']:>8}  {seg['length_mm']:9.1f} mm  +-{seg['sigma_mm']:g}  {seg['reference']}")
            if not seg["reference"]:
                print(f"           UWAGA: pusta kolumna 'punkty_pomiaru' — nie wiadomo, w co klikać "
                      f"w CloudCompare")
        odstajace = suspicious_lengths(targets["segments"])
        if odstajace:
            print(f"\nPODEJRZANE DŁUGOŚCI (poza {SUSPICIOUS_MM[0]:g}-{SUSPICIOUS_MM[1]:g} mm — "
                  f"sprawdź kolumnę 'jednostka'):", file=sys.stderr)
            for seg in odstajace:
                print(f"  {seg['name']}: {seg['length_mm']:.1f} mm", file=sys.stderr)
        return 0

    out_dir = args.runs / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    photos_dir = args.session / "photos_targets"

    inv = build_inventory(args.runs, photos_dir, out_dir, args.dict_name, args.redetect_photos)
    if not inv.detections:
        print("BŁĄD: zero wykryć — czy przebiegi --render są w --runs?", file=sys.stderr)
        return 1

    rows = summary_rows(inv, boards, args.min_px, declared)
    write_summary(rows, out_dir / "ids_summary.csv", inv.videos, args.min_px)
    print(f"\nzestawienie: {out_dir / 'ids_summary.csv'}  ({len(rows)} ID)")
    for row in rows:
        print(f"  ID {row['id']:>3}  rozstawiony: {row['zadeklarowany']:3}  boards: {row['w_boards']:3}"
              f"  klatek {row['klatek_razem']:>5}"
              f"  zdjęć {row['zdjec_targets']:>3}  bok512 śr {row['bok512_sredni']:>6}"
              f"  min {row['bok512_min']:>6}  najlepszy: {row['najlepszy_widok']}")

    if not args.no_views:
        n_ids, n_overview = write_views(inv, args.session / "video", photos_dir, out_dir, args.dict_name)
        print(f"\nobrazki: {n_ids} w {out_dir / 'ids'}, {n_overview} w {out_dir / 'overview'}")

    notes_dir = args.session / "notes"
    nagrania = write_nagrania(args.session / "video", notes_dir / "nagrania.csv")
    pomiary = write_pomiary(inv, expected, notes_dir / "pomiary_tasma.csv", args.all_pairs)
    print(f"\ntabele do uzupełnienia:")
    print(f"  {notes_dir / 'nagrania.csv'}      ({len(nagrania)} nagrań)")
    print(f"  {notes_dir / 'pomiary_tasma.csv'} ({len(pomiary)} par"
          + ("" if args.all_pairs else ", tylko współwidoczne") + ")")

    detected = {d.marker_id for d in inv.detections}
    obce = sorted(detected - expected)
    if obce:
        print(f"\nID WYKRYTE, A NIE ROZSTAWIONE (fałszywe wykrycia): {obce}")
        for marker_id in obce:
            row = next(r for r in rows if r["id"] == marker_id)
            w_boards = " (jest w boards.json!)" if row["w_boards"] == "tak" else ""
            print(f"  ID {marker_id:>3}  klatek {row['klatek_razem']:>5}  zdjęć {row['zdjec_targets']:>3}"
                  f"  bok512 max {row['bok512_max']:>6}{w_boards}")
    brakujace = sorted(expected - detected)
    if brakujace:
        print(f"\nID ROZSTAWIONE, A NIEWYKRYTE NIGDZIE: {brakujace}")

    rare = rare_in_photos(inv, expected)
    if rare:
        print(f"\nROZSTAWIONE ID NA MNIEJ NIŻ {PHOTOS_FLAG_THRESHOLD} ZDJĘCIACH z photos_targets/:")
        for marker_id, n in rare:
            board = boards.get(marker_id, {}).get("board", "brak w boards.json")
            print(f"  ID {marker_id:>3}  zdjęć: {n}  ({board})")
    else:
        print(f"\nKażdy rozstawiony ID jest na co najmniej {PHOTOS_FLAG_THRESHOLD} zdjęciach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
