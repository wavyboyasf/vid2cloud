"""Zrzuty stanu klatek kluczowych do .npz.

Dwa zrzuty na klatkę kluczową k: `kf_{k:03d}_created.npz` (stan w chwili powstania)
i `kf_{k:03d}_final.npz` (stan na końcu przebiegu). Różnią się, bo tracker uśrednia
pointmapę ostatniej klatki kluczowej z kolejnymi klatkami, a backend przesuwa pozy.

Zawartość pliku (klucze `KEYFRAME_NPZ_KEYS`):
    kf              int    — indeks klatki kluczowej
    frame_index     int    — indeks klatki źródła, z której powstała
    pts_s           float  — PTS tej klatki [s]
    source          str    — "init" | "tracker" | "reloc"
    T_WC            (8,)   — Sim(3): tx ty tz qx qy qz qw s (float64 z float32 silnika)
    X_canon         (h, w, 3) float32 — pointmapa w układzie kamery, jednostki sieci
    C               (h, w) float32    — SUMA ufności z uśredniania; średnia to C / N
    N               int    — liczba uśrednionych pointmap (0 możliwe dla relokalizacji)
    N_updates       int
    img_true_shape  (2,)   — (h, w) wejścia modelu
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

KEYFRAME_NPZ_KEYS = (
    "kf", "frame_index", "pts_s", "source", "T_WC", "X_canon", "C", "N", "N_updates", "img_true_shape",
)
KEYFRAME_SOURCES = ("init", "tracker", "reloc")
KEYFRAME_STAGES = ("created", "final")


def keyframe_npz_path(kf_dir: Path, kf: int, stage: str) -> Path:
    """Ścieżka zrzutu: kf_dir / kf_{kf:03d}_{stage}.npz."""
    if stage not in KEYFRAME_STAGES:
        raise ValueError(f"stage musi być jednym z {KEYFRAME_STAGES}, dostałem {stage!r}")
    return Path(kf_dir) / f"kf_{kf:03d}_{stage}.npz"


def write_keyframe_npz(
    path: Path,
    *,
    kf: int,
    frame_index: int,
    pts_s: float,
    source: str,
    T_WC: np.ndarray,
    X_canon: np.ndarray,
    C: np.ndarray,
    N: int,
    N_updates: int,
    img_true_shape: tuple[int, int],
) -> Path:
    """Zapisuje zrzut klatki kluczowej. Kształty są sprawdzane, nie dopasowywane."""
    if source not in KEYFRAME_SOURCES:
        raise ValueError(f"source musi być jednym z {KEYFRAME_SOURCES}, dostałem {source!r}")
    T_WC = np.asarray(T_WC, dtype=np.float64).reshape(-1)
    if T_WC.shape != (8,):
        raise ValueError(f"T_WC ma mieć 8 liczb Sim(3), ma {T_WC.shape}")
    h, w = int(img_true_shape[0]), int(img_true_shape[1])
    X_canon = np.asarray(X_canon, dtype=np.float32)
    C = np.asarray(C, dtype=np.float32)
    if X_canon.shape != (h, w, 3):
        raise ValueError(f"X_canon ma kształt {X_canon.shape}, oczekiwano {(h, w, 3)}")
    if C.shape != (h, w):
        raise ValueError(f"C ma kształt {C.shape}, oczekiwano {(h, w)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        kf=np.int64(kf),
        frame_index=np.int64(frame_index),
        pts_s=np.float64(pts_s),
        source=np.str_(source),
        T_WC=T_WC,
        X_canon=X_canon,
        C=C,
        N=np.int64(N),
        N_updates=np.int64(N_updates),
        img_true_shape=np.array([h, w], dtype=np.int64),
    )
    return path


def read_keyframe_npz(path: Path) -> dict:
    """Wczytuje zrzut jako słownik; skalary jako typy Pythona, tablice jako numpy."""
    with np.load(path, allow_pickle=False) as data:
        missing = set(KEYFRAME_NPZ_KEYS) - set(data.files)
        if missing:
            raise ValueError(f"{path}: brak kluczy {sorted(missing)}")
        out = {}
        for key in KEYFRAME_NPZ_KEYS:
            value = data[key]
            if value.ndim == 0:
                value = value.item()
            out[key] = value
    return out
