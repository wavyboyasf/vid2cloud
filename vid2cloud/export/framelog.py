"""Log per klatka do CSV — obowiązkowy dla każdego przebiegu (zasada 3 z CLAUDE.md).

    FRAME_LOG_COLUMNS — stały zestaw kolumn `log.csv`.
    FrameLogWriter    — writer CSV z tymi kolumnami.
    depth_stats()     — mediana i p95 głębi z pointmapy klatki + udział pikseli niskiej ufności.

Ten plik jest podstawą rozdziału wyników — bez niego nie ma czym pokazać, gdzie pipeline
traci czas ani jak zachowuje się głębia klatka po klatce. Kolumny ostrości i selekcji
dojdą z `input/selection.py` (komponent 1); na razie każda klatka idzie do silnika.

Głębia to współrzędna z pointmapy w układzie kamery tej klatki, w jednostkach sieci
(bez skali Sim(3) i bez skali metrycznej). Liczona ze wszystkich pikseli, bez filtrowania.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

FRAME_LOG_COLUMNS = (
    "index", "pts_s", "t_decode_ms", "t_engine_ms", "t_backend_wait_ms",
    "is_kf", "kf_index", "mode", "n_kf", "depth_median", "depth_p95", "low_conf_frac",
)


def depth_stats(
    pointmap: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    conf_threshold: float,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(mediana z, p95 z, udział pikseli z conf < conf_threshold). None, gdy brak danych."""
    median = p95 = low = None
    if pointmap is not None:
        z = np.asarray(pointmap, dtype=np.float64).reshape(-1, 3)[:, 2]
        median = float(np.median(z))
        p95 = float(np.percentile(z, 95))
    if conf is not None:
        c = np.asarray(conf).reshape(-1)
        low = float(np.mean(c < conf_threshold))
    return median, p95, low


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


class FrameLogWriter:
    """CSV z kolumnami `FRAME_LOG_COLUMNS`; brakujące wartości jako puste pola."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(FRAME_LOG_COLUMNS)
        self.n_rows = 0

    def write(self, row: dict) -> None:
        unknown = set(row) - set(FRAME_LOG_COLUMNS)
        if unknown:
            raise ValueError(f"nieznane kolumny logu: {sorted(unknown)}")
        self._writer.writerow([_cell(row.get(col)) for col in FRAME_LOG_COLUMNS])
        self.n_rows += 1
        if self.n_rows % 50 == 0:
            self._file.flush()

    def close(self) -> None:
        self._file.close()
