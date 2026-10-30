"""Selekcja klatek: miary jakości klatki i dobór klatek do silnika (komponent 1 pracy).

Funkcje czyste (numpy / OpenCV / PIL), bez czytania plików i bez GPU:

    model_input_rgb()   — klatka RGB w pełnej rozdzielczości -> obraz wejścia sieci (512 px)
    model_input_gray()  — to samo w skali szarości; na tym liczone są wszystkie miary
    tenengrad()         — średnia (Gx² + Gy²), Sobel 3×3, float32       [miara główna]
    laplacian_var()     — wariancja Laplasjanu, cv2.Laplacian ksize=3   [do porównania]
    exposure()          — mediana jasności, udział pikseli < 5 i > 250
    fixed_step()        — stały krok k od przesunięcia `offset`
    select_in_windows() — w każdym oknie k klatek argmax miary spośród dopuszczonych
    thresholds()        — maska dopuszczenia: próg względny ostrości + zakres jasności
    parse_frame_list() / format_frame_list() — plik listy klatek (jeden indeks w linii)

Rozdzielczość miar. Sieć widzi klatkę po `resize_img()` upstreamu (dłuższy bok do 512,
LANCZOS, crop centralny do wielokrotności 16 px), więc miary liczymy na tym samym obrazie.
Geometria (rozmiar po skalowaniu, crop) pochodzi z `scale.aruco.input_transform()` —
tej samej funkcji, którą sprawdzają testy względem PIL i upstreamu; tu dochodzi tylko
resampling pikseli tym samym filtrem co upstream (`mast3r_utils.py:234-241`). Rotacja
z metadanych jest stosowana wcześniej, w `input/sources.py::FileSource`.

Progi `thresholds()` (THR_*) ustalono 2026-10-28, PRZED jakimkolwiek przebiegiem z selekcją
(docs/decisions.md). Nie wolno ich stroić pod wynik (CLAUDE.md, zasada 4).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
import PIL.Image

from vid2cloud.scale.aruco import MODEL_INPUT_SIZE, input_transform

#: Progi odrzucające — ustalone 2026-10-28 przed przebiegami, nie do strojenia.
THR_REL = 0.5  # tenengrad < THR_REL × mediana tenengradu w oknie czasowym -> odrzut
THR_WIN_S = 2.0  # półszerokość okna czasowego [s] wokół klatki
THR_MED_LO = 40.0  # mediana jasności poniżej -> odrzut (0–255)
THR_MED_HI = 220.0  # mediana jasności powyżej -> odrzut (0–255)

#: Progi pikseli „ciemnych" i „prześwietlonych" w `exposure()` (0–255, ostre nierówności).
DARK_LT = 5
BRIGHT_GT = 250


# --------------------------------------------------------------------------- #
# Obraz wejścia sieci
# --------------------------------------------------------------------------- #


def model_input_rgb(rgb: np.ndarray) -> np.ndarray:
    """Klatka RGB uint8 (H, W, 3) -> obraz wejścia sieci uint8 (h, w, 3), jak `resize_img()`.

    Upstream (`mast3r_utils.py:244`) robi `PIL.Image.fromarray(np.uint8(img * 255))` z klatki
    float32 = rgb / 255; ten obieg jest tożsamościowy dla wszystkich 256 wartości uint8
    (sprawdzone 2026-10-28), więc PIL dostaje tu tę samą klatkę wprost. Filtr: LANCZOS przy
    zmniejszaniu, BICUBIC przy powiększaniu (`_resize_pil_image()`, `mast3r_utils.py:234`).
    Zgodność piksel w piksel z upstreamem pilnuje test w env `mast3r-slam`.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"oczekuję RGB uint8 (H, W, 3), dostałem {rgb.dtype} {rgb.shape}")
    tf = input_transform(rgb.shape[:2])
    h, w = tf.resized_shape
    interp = PIL.Image.LANCZOS if max(rgb.shape[:2]) > MODEL_INPUT_SIZE else PIL.Image.BICUBIC
    img = PIL.Image.fromarray(np.ascontiguousarray(rgb)).resize((w, h), interp)
    out_h, out_w = tf.out_shape
    img = img.crop((tf.left, tf.top, tf.left + out_w, tf.top + out_h))
    out = np.asarray(img)
    if out.shape[:2] != (out_h, out_w):
        raise AssertionError(f"crop dał {out.shape[:2]}, a input_transform() mówi {tf.out_shape}")
    return out


def model_input_gray(rgb: np.ndarray) -> np.ndarray:
    """Klatka RGB uint8 w pełnej rozdzielczości -> skala szarości uint8 w rozdzielczości sieci.

    Najpierw skalowanie RGB (to, co widzi sieć), potem konwersja `cv2.COLOR_RGB2GRAY`.
    """
    return cv2.cvtColor(model_input_rgb(rgb), cv2.COLOR_RGB2GRAY)


# --------------------------------------------------------------------------- #
# Miary jakości (na obrazie z model_input_gray)
# --------------------------------------------------------------------------- #


def _gray_float32(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray)
    if g.ndim != 2:
        raise ValueError(f"oczekuję obrazu w skali szarości (H, W), dostałem {g.shape}")
    return g.astype(np.float32, copy=False)


def tenengrad(gray: np.ndarray) -> float:
    """Średnia z Gx² + Gy² po wszystkich pikselach; Sobel 3×3 na float32 (jasność 0–255).

    Brzeg obrazu: domyślne `BORDER_REFLECT_101` OpenCV. Średnia akumulowana w float64.
    """
    g = _gray_float32(gray)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy, dtype=np.float64))


def laplacian_var(gray: np.ndarray) -> float:
    """Wariancja odpowiedzi `cv2.Laplacian(ksize=3)` na float32 (jasność 0–255).

    Uwaga: w OpenCV ksize=3 to suma drugich pochodnych Sobela, czyli jądro
    [[2, 0, 2], [0, -8, 0], [2, 0, 2]]; klasyczne jądro 4-sąsiedztwa [[0, 1, 0], [1, -4, 1],
    [0, 1, 0]] to ksize=1. Brzeg: `BORDER_REFLECT_101`. Wariancja w float64 (ddof = 0).
    """
    g = _gray_float32(gray)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(np.var(lap, dtype=np.float64))


def exposure(gray: np.ndarray) -> dict[str, float]:
    """Mediana jasności (0–255) i udziały pikseli < 5 (`frac_dark`) i > 250 (`frac_bright`).

    Klucze są nazwami kolumn `quality.csv`. Wejście: uint8, bo progi są w skali 0–255.
    """
    g = np.asarray(gray)
    if g.ndim != 2 or g.dtype != np.uint8:
        raise ValueError(f"oczekuję uint8 (H, W), dostałem {g.dtype} {g.shape}")
    return {
        "bright_median": float(np.median(g)),
        "frac_dark": float(np.mean(g < DARK_LT)),
        "frac_bright": float(np.mean(g > BRIGHT_GT)),
    }


# --------------------------------------------------------------------------- #
# Dobór klatek
# --------------------------------------------------------------------------- #


def fixed_step(n_frames: int, k: int, offset: int = 0) -> list[int]:
    """Stały krok: offset, offset + k, offset + 2k, ... < n_frames.

    Liczba klatek to ceil((n_frames − offset) / k), więc przy offset > 0 może być o jedną
    mniejsza niż przy offset = 0.
    """
    if k < 1:
        raise ValueError(f"k musi być >= 1, dostałem {k}")
    if not 0 <= offset < k:
        raise ValueError(f"offset musi być w [0, k), dostałem offset={offset}, k={k}")
    if n_frames < 0:
        raise ValueError(f"n_frames musi być >= 0, dostałem {n_frames}")
    return list(range(offset, n_frames, k))


def select_in_windows(
    scores: Sequence[float],
    k: int,
    keep: Optional[Sequence[bool]] = None,
) -> tuple[list[int], dict]:
    """W każdym oknie [0, k), [k, 2k), ... wybiera klatkę o największej mierze.

    Kandydatami w oknie są tylko klatki z `keep == True` (przy `keep=None` — wszystkie).
    Okno bez żadnego kandydata jest POMIJANE (nic z niego nie trafia do listy) i ląduje
    w raporcie z powodem — wyboru nie wymuszamy. Remis: wygrywa wcześniejsza klatka
    (`np.argmax`).

    Ostatnie okno, gdy n nie dzieli się przez k, jest krótsze ([k·⌊n/k⌋, n)) i jest
    traktowane tak samo jak pozostałe: argmax spośród jego klatek. Dzięki temu okien jest
    ceil(n / k) = len(fixed_step(n, k, 0)), a przy `keep=None` tyle samo jest wybranych klatek.

    Zwraca (indeksy rosnąco, raport), raport = {"k", "n_frames", "n_windows", "n_selected",
    "skipped": [{"window", "start", "stop", "reason"}, ...]} (stop wyłącznie).
    """
    if k < 1:
        raise ValueError(f"k musi być >= 1, dostałem {k}")
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"scores musi być 1-D, dostałem {s.shape}")
    if not np.all(np.isfinite(s)):
        bad = np.flatnonzero(~np.isfinite(s))[:10].tolist()
        raise ValueError(f"scores ma wartości nieskończone/NaN (klatki {bad}) — nie zgaduję, co z nimi")
    n = s.size
    if keep is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != s.shape:
            raise ValueError(f"keep ma kształt {mask.shape}, scores {s.shape}")

    indices: list[int] = []
    skipped: list[dict] = []
    n_windows = 0
    for window, start in enumerate(range(0, n, k)):
        n_windows += 1
        stop = min(start + k, n)
        cand = np.flatnonzero(mask[start:stop])
        if cand.size == 0:
            skipped.append({"window": window, "start": start, "stop": stop,
                            "reason": "brak klatki z keep=True w oknie"})
            continue
        indices.append(int(start + cand[np.argmax(s[start:stop][cand])]))
    report = {"k": k, "n_frames": n, "n_windows": n_windows, "n_selected": len(indices), "skipped": skipped}
    return indices, report


def threshold_parts(
    df: Mapping,
    rel: float = THR_REL,
    win_s: float = THR_WIN_S,
    med_lo: float = THR_MED_LO,
    med_hi: float = THR_MED_HI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Składniki `thresholds()`: (sharp_ok, bright_ok, mediana tenengradu w oknie).

    Rozbite po to, żeby raport mógł podać, ile klatek odpadło z którego powodu.
    """
    pts = np.asarray(df["pts_s"], dtype=np.float64)
    ten = np.asarray(df["tenengrad"], dtype=np.float64)
    med = np.asarray(df["bright_median"], dtype=np.float64)
    if not (pts.shape == ten.shape == med.shape) or pts.ndim != 1:
        raise ValueError(f"kolumny o różnych kształtach: pts {pts.shape}, tenengrad {ten.shape}, "
                         f"bright_median {med.shape}")
    for name, arr in (("pts_s", pts), ("tenengrad", ten), ("bright_median", med)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} ma wartości nieskończone/NaN")
    if pts.size > 1 and not np.all(np.diff(pts) > 0):
        raise ValueError("pts_s musi rosnąć ściśle (kolejność dekodowania z PTS)")

    lo = np.searchsorted(pts, pts - win_s, side="left")
    hi = np.searchsorted(pts, pts + win_s, side="right")
    local = np.array([np.median(ten[a:b]) for a, b in zip(lo, hi)], dtype=np.float64)
    sharp_ok = ten >= rel * local
    bright_ok = (med >= med_lo) & (med <= med_hi)
    return sharp_ok, bright_ok, local


def thresholds(
    df: Mapping,
    rel: float = THR_REL,
    win_s: float = THR_WIN_S,
    med_lo: float = THR_MED_LO,
    med_hi: float = THR_MED_HI,
) -> np.ndarray:
    """Maska `keep` (bool, jedna wartość na klatkę). Klatka odpada, gdy:

    * tenengrad < rel × mediana tenengradu klatek z |pts_j − pts_i| ≤ win_s (okno domknięte,
      klatka i jest w nim; przy początku i końcu nagrania okno jest ucięte, nie przesuwane), ALBO
    * mediana jasności poza domkniętym przedziałem [med_lo, med_hi].

    `df` — cokolwiek, co daje kolumny przez `df["pts_s"]`, `df["tenengrad"]`,
    `df["bright_median"]` (słownik tablic, wynik `read_quality_csv()`, DataFrame).
    Wartości domyślne to progi ustalone przed przebiegami (THR_*), nie do strojenia.
    """
    sharp_ok, bright_ok, _ = threshold_parts(df, rel=rel, win_s=win_s, med_lo=med_lo, med_hi=med_hi)
    return sharp_ok & bright_ok


# --------------------------------------------------------------------------- #
# Lista klatek (plik tekstowy, jeden indeks w linii)
# --------------------------------------------------------------------------- #


def format_frame_list(indices: Sequence[int]) -> str:
    """Treść pliku listy: jeden indeks klatki w linii, zakończona znakiem nowej linii."""
    return "".join(f"{int(i)}\n" for i in indices)


def parse_frame_list(text: str) -> list[int]:
    """Treść pliku listy -> indeksy. Puste linie są pomijane; reszta musi być liczbą
    całkowitą >= 0, a całość ściśle rosnąca i niepusta — inaczej `ValueError`."""
    out: list[int] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = int(line)
        except ValueError:
            raise ValueError(f"linia {lineno}: {line!r} nie jest indeksem klatki") from None
        if value < 0:
            raise ValueError(f"linia {lineno}: indeks ujemny {value}")
        if out and value <= out[-1]:
            raise ValueError(f"linia {lineno}: {value} po {out[-1]} — lista musi rosnąć ściśle")
        out.append(value)
    if not out:
        raise ValueError("lista klatek jest pusta")
    return out
