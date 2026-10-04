"""Zapis trajektorii kamery i historii póz klatek kluczowych — ze skalą Sim(3).

    write_trajectory_sim3() — `trajectory_sim3.txt`: jedna linia na klatkę kluczową,
                              `pts_s tx ty tz qx qy qz qw s`. Upstreamowy `save_traj`
                              (evaluate.py:40) zapisuje `as_SE3(T_WC)`, czyli GUBI skalę;
                              ten format ją zachowuje. Pierwsze 8 kolumn jest jak w TUM.
    PosesHistoryWriter      — `poses_history.csv`: po każdym zdarzeniu (powstanie klatki
                              kluczowej, koniec optymalizacji backendu, relokalizacja,
                              koniec przebiegu) wiersz na KAŻDĄ klatkę kluczową.

Liczby z silnika są float32; zapis `%.9g` odtwarza je bez straty.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from vid2cloud.engine.base import SIM3_FIELDS

POSES_HISTORY_COLUMNS = ("event_id", "event", "frame_index", "n_kf", "kf", *SIM3_FIELDS)
POSES_EVENTS = ("kf_created", "backend_done", "reloc", "final")
TRAJECTORY_SIM3_HEADER = "# pts_s " + " ".join(SIM3_FIELDS)


def _fmt(x: float) -> str:
    return f"{float(x):.9g}"


def _as_poses(poses) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"pozy mają mieć kształt (n, 8), mają {arr.shape}")
    return arr


def write_trajectory_sim3(path: Path, pts_s: Sequence[float], poses) -> Path:
    """Trajektoria klatek kluczowych: `pts_s tx ty tz qx qy qz qw s`, nagłówek z '#'."""
    poses = _as_poses(poses)
    if len(pts_s) != len(poses):
        raise ValueError(f"{len(pts_s)} czasów na {len(poses)} póz")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(TRAJECTORY_SIM3_HEADER + "\n")
        for t, pose in zip(pts_s, poses):
            f.write(" ".join([f"{float(t):.6f}", *(_fmt(v) for v in pose)]) + "\n")
    return path


def read_trajectory_sim3(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (pts_s (n,), poses (n, 8))."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != 9:
        raise ValueError(f"{path}: oczekuję 9 kolumn, jest {data.shape[1]}")
    return data[:, 0], data[:, 1:]


class PosesHistoryWriter:
    """Dopisuje zdarzenia do `poses_history.csv`; każdy wiersz to jedna klatka kluczowa."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(POSES_HISTORY_COLUMNS)
        self.n_events = 0

    def write_event(self, event: str, frame_index: int, poses) -> int:
        """Zapisuje zdarzenie z pozami wszystkich klatek kluczowych; zwraca event_id."""
        if event not in POSES_EVENTS:
            raise ValueError(f"event musi być jednym z {POSES_EVENTS}, dostałem {event!r}")
        poses = _as_poses(poses)
        event_id = self.n_events
        for kf, pose in enumerate(poses):
            self._writer.writerow([event_id, event, int(frame_index), len(poses), kf, *(_fmt(v) for v in pose)])
        self._file.flush()
        self.n_events += 1
        return event_id

    def close(self) -> None:
        self._file.close()


def read_poses_history(path: Path) -> list[dict]:
    """Wiersze `poses_history.csv` jako słowniki z liczbami (int / float)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != POSES_HISTORY_COLUMNS:
            raise ValueError(f"{path}: kolumny {reader.fieldnames}, oczekiwano {POSES_HISTORY_COLUMNS}")
        for row in reader:
            out: dict = {"event": row["event"]}
            for key in ("event_id", "frame_index", "n_kf", "kf"):
                out[key] = int(row[key])
            for key in SIM3_FIELDS:
                out[key] = float(row[key])
            rows.append(out)
    return rows
