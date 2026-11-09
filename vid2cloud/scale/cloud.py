"""Odbudowa chmury z zrzutów klatek kluczowych (`kf_*_final.npz`) i chmury w metrach.

Reguła doboru punktów jest ta sama co przy zapisie `cloud.ply` przez upstream:
`third_party/mast3r-slam/mast3r_slam/evaluate.py:47-70` (`save_reconstruction`, wołane
z `vid2cloud/engine/mast3r_slam.py::finalize()` z progiem `CLOUD_C_CONF_THRESHOLD` = 1,5):

* klatki kluczowe po kolei, k = 0 … K−1 (`evaluate.py:52`);
* punkt świata `T_WC.act(X_canon)` = s·R·X + t (`evaluate.py:59`), piksele wiersz po wierszu;
* zostają piksele ze średnią ufnością `get_average_conf()` = C / N **ostro większą** od progu
  (`evaluate.py:61-64`, `frame.py:107-108`); próg chmury to `>`, próg liftu `C_MIN` to `>=`;
* `use_calib` = False w `config/base.yaml:1`, więc gałąź `constrain_points_to_ray`
  (`evaluate.py:54-58`) nie działa i X_canon idzie bez zmian;
* zapis float32 x y z + uchar red green blue (`evaluate.py:88-106`).

C / N liczone jak w torchu na GPU: dzielnik jest skalarem Pythona (`frame.py:265`), a dla
skalara CUDA mnoży przez odwrotność w float32 — tu tak samo: C · (1/N) w float32. Na trzech
przebiegach z 03.10 zwykłe dzielenie float32 daje tę samą liczbę punktów (sprawdzone 04.11).

Zrzuty .npz nie mają kolorów. Jeśli odbudowana chmura punkt po punkcie zgadza się z
`cloud.ply` (ta sama liczba punktów i maks. różnica współrzędnych < `MATCH_TOL`), kolory
są brane z `cloud.ply` po indeksie; jeśli nie — chmury w metrach są zapisywane bez kolorów.

Chmura w metrach (krok 4 sesji S9): dla wektora współczynników f_k klatek kluczowych
położenia klatek składa się jak ciąg poligonowy (`vid2cloud/scale/factors.py::keyframe_positions`),
a punkt klatki k to P = p_k + f_k · s_k · R_k · X. Przy jednym f dla wszystkich klatek
P = f · (s_k · R_k · X + t_k), czyli chmura upstreamu × f.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from vid2cloud.export.plyio import read_ply, write_ply, xyz
from vid2cloud.scale.lift import quat_to_matrix

#: Próg średniej ufności chmury upstreamu (`engine/mast3r_slam.py::CLOUD_C_CONF_THRESHOLD`).
CLOUD_C_THRESHOLD = 1.5
#: Maks. różnica współrzędnych odbudowa ↔ cloud.ply, przy której kolejność uznajemy za tę samą.
MATCH_TOL = 1e-5


@dataclass
class KeyframeCloud:
    """Piksele klatki kluczowej k, które wchodzą do chmury, w układzie kamery k."""

    kf: int
    T_WC: np.ndarray  # (8,) tx ty tz qx qy qz qw s — poza końcowa
    X: np.ndarray  # (n, 3) float64, jednostki sieci

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.T_WC[:3], dtype=np.float64)

    @property
    def s(self) -> float:
        return float(self.T_WC[7])

    @property
    def R(self) -> np.ndarray:
        return quat_to_matrix(self.T_WC[3:7])


def average_conf_mask(C_sum: np.ndarray, N: int, threshold: float = CLOUD_C_THRESHOLD) -> np.ndarray:
    """Maska pikseli chmury: C · (1/N) > próg, w float32 jak torch (N = 0 -> inf/nan jak upstream)."""
    C32 = np.asarray(C_sum, dtype=np.float32).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.float32(1.0) / np.float32(int(N))
        avg = (C32 * inv).astype(np.float32)
    return avg > np.float32(threshold)


def keyframe_clouds(final_dumps: Sequence[dict], threshold: float = CLOUD_C_THRESHOLD) -> list[KeyframeCloud]:
    """Z zrzutów końcowych (w kolejności k) piksele chmury i pozy. Kolejność k jest sprawdzana."""
    out = []
    for i, d in enumerate(final_dumps):
        if int(d["kf"]) != i:
            raise ValueError(f"zrzut na pozycji {i} ma kf = {d['kf']} — oczekiwano kolejnych klatek od 0")
        X = np.asarray(d["X_canon"]).reshape(-1, 3)
        mask = average_conf_mask(d["C"], d["N"], threshold)
        out.append(KeyframeCloud(i, np.asarray(d["T_WC"], dtype=np.float64).reshape(-1), X[mask].astype(np.float64)))
    return out


def assemble(kclouds: Sequence[KeyframeCloud], f: Optional[Sequence[float]] = None,
             p: Optional[np.ndarray] = None) -> np.ndarray:
    """Chmura (N, 3) float64. Bez f i p: s_k·R_k·X + t_k (jak upstream); z nimi: p_k + f_k·s_k·R_k·X."""
    if (f is None) != (p is None):
        raise ValueError("podaj oba: f i p, albo żadne")
    if f is not None and (len(f) != len(kclouds) or len(p) != len(kclouds)):
        raise ValueError(f"{len(kclouds)} klatek, {len(f)} współczynników, {len(p)} położeń")
    parts = []
    for k, kc in enumerate(kclouds):
        rot = kc.s * kc.X @ kc.R.T
        if f is None:
            parts.append(rot + kc.t)
        else:
            parts.append(np.asarray(p[k], dtype=np.float64) + float(f[k]) * rot)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))


def compare_with_upstream(points: np.ndarray, cloud_ply: Path) -> tuple[dict, Optional[np.ndarray]]:
    """Odbudowa ↔ cloud.ply punkt po punkcie. Zwraca (raport, kolory (N, 3) uint8 albo None)."""
    v = read_ply(cloud_ply)
    up = xyz(v)
    report: dict = {"cloud_ply": str(cloud_ply), "n_rebuilt": int(len(points)), "n_upstream": int(len(up)),
                    "max_abs_diff": None, "match_tol": MATCH_TOL, "same_order": False}
    if len(up) == len(points) and len(up):
        diff = float(np.max(np.abs(up - points)))
        report["max_abs_diff"] = diff
        report["same_order"] = diff < MATCH_TOL
    colors = None
    if report["same_order"] and all(c in v.dtype.names for c in ("red", "green", "blue")):
        colors = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
    report["colors"] = "z cloud.ply po indeksie" if colors is not None else "brak (kolejność nie zgodna z cloud.ply)"
    return report, colors


def write_cloud(path: Path, points: np.ndarray, colors: Optional[np.ndarray], comments: tuple[str, ...] = ()) -> None:
    """PLY jak upstream: float32 x y z (+ uchar red green blue, gdy kolory są)."""
    dt = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    v = np.zeros(len(points), dtype=dt)
    v["x"], v["y"], v["z"] = points[:, 0], points[:, 1], points[:, 2]
    if colors is not None:
        v["red"], v["green"], v["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    write_ply(path, v, comments)
