"""Źródła klatek: plik wideo i kamera na żywo.

    FileSource    — dekodowanie pliku przez PyAV, sekwencyjnie, jednym przebiegiem, bez seeków.
                    Zwraca (index, pts_s, rgb uint8 HxWx3) w pełnej rozdzielczości, już
                    obrócone do orientacji wyświetlania.
    CameraSource  — (jeszcze nie ma) strumień na żywo; klatki spóźnione są porzucane,
                    nie kolejkowane.

Uwaga: dekodowanie idzie wyłącznie przez PyAV. OpenCV służy do operacji na obrazie
(ArUco, miary ostrości), nigdy do czytania wideo — patrz CLAUDE.md.

Rotacja. PyAV wydaje piksele tak, jak leżą w pliku, a `cv2.VideoCapture` w dekoderze
upstreamu stosuje display matrix sam (docs/decisions.md, wpis z 2026-10-02). Źródło
obraca więc klatkę samo, tymi samymi funkcjami co `scale/aruco.py`, żeby silnik, detekcja
ArUco i upstream patrzyły na ten sam kadr.

Czas. Zawsze z PTS klatki (`frame.pts * time_base`), nigdy `index / fps` — nagrania
z telefonu bywają VFR. PTS nie jest przesuwany do zera: pierwsza klatka może mieć
np. 0,0243 s, tak samo jak w `podglad.csv` z `scale/aruco.py --render`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from vid2cloud.scale.aruco import apply_rotation, frame_rotation


class FileSource:
    """Sekwencyjny dekoder pliku wideo (PyAV). Iteracja daje (index, pts_s, rgb).

    `rgb` to uint8 (H, W, 3) w pełnej rozdzielczości, po rotacji z metadanych
    (przy `rotate=True`). `index` to numer klatki w kolejności dekodowania (od 0).
    Po iteracji `rotation` trzyma kąt z pierwszej klatki, a `n_frames` liczbę klatek.
    """

    def __init__(self, path: Path | str, rotate: bool = True):
        self.path = Path(path)
        self.rotate = rotate
        self.rotation: Optional[int] = None
        self.n_frames = 0

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import av  # lokalnie: reszta pakietu nie potrzebuje PyAV

        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time is None:
                    raise ValueError(f"{self.path}: klatka {index} bez PTS — czasu nie zgaduję z fps")
                rgb = frame.to_ndarray(format="rgb24")
                rotation = frame_rotation(frame) if self.rotate else 0
                if self.rotation is None:
                    self.rotation = rotation
                if rotation:
                    rgb = apply_rotation(rgb, rotation)
                self.n_frames = index + 1
                yield index, float(frame.time), rgb
