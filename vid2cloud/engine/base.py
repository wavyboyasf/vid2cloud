"""Wspólny interfejs silnika rekonstrukcji.

    InputTransform — przekształcenie pełna klatka -> wejście modelu, tak jak liczy je silnik.
    FrameResult    — wynik dla jednej klatki: poza Sim(3), pointmapa w układzie kamery,
                     ufność, rozdzielczość wejścia modelu, znacznik klatki kluczowej.
    Engine         — protokół silnika: input_transform(), push_frame(), finalize().

Interfejs jest po to, żeby reszta pakietu nie znała szczegółów MASt3R-SLAM i żeby
dało się podstawić inny silnik (CUT3R, SLAM3R) bez ruszania warstwy wejścia i skali.
Moduł nie importuje torcha — FrameResult niesie tablice numpy.

Konwencja pozy: 8 liczb Sim(3) w kolejności lietorch, [tx, ty, tz, qx, qy, qz, qw, s]
(`lietorch_utils.as_SE3` w upstreamie dzieli wektor na [3, 4, 1]). Punkt świata to
`s * R(q) @ X + t` dla punktu X w układzie kamery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

#: Nazwy pól pozy Sim(3) w kolejności `Sim3.data`.
SIM3_FIELDS = ("tx", "ty", "tz", "qx", "qy", "qz", "qw", "s")


@dataclass(frozen=True)
class InputTransform:
    """Przekształcenie pełna klatka -> wejście modelu, w liczbach podanych przez silnik.

    Pola mają te same nazwy i znaczenie co `scale.aruco.InputTransform` (mnożniki
    W/W1 i H/H1, lewy górny róg cropu po skalowaniu, kształty (H, W)), żeby dało się je
    porównać testem i żeby run.py mógł przeliczać narożniki konwencją modułu skali.
    Silnik podaje same liczby; konwencja piksela (środek czy róg) należy do scale/.
    """

    scale_x: float
    scale_y: float
    left: float
    top: float
    resized_shape: tuple[int, int]  # (H, W) po skalowaniu, przed cropem
    out_shape: tuple[int, int]  # (H, W) wejścia modelu


@dataclass
class FrameResult:
    """Wynik silnika dla jednej klatki źródła.

    `T_WC` — poza tej klatki po przetworzeniu (Sim(3), 8 liczb, `SIM3_FIELDS`).
    `pointmap` — (h*w, 3) w układzie kamery tej klatki, w jednostkach sieci; `conf` —
    (h*w,) średnia ufność (C / N). Oba w rozdzielczości `input_shape` = (h, w).
    `new_keyframes` — klatki kluczowe, które pojawiły się w trakcie tego wywołania, jako
    pary (kf, frame_index); przy relokalizacji w osobnym procesie frame_index może być
    wcześniejszą klatką niż bieżąca.
    """

    index: int
    pts_s: float
    is_keyframe: bool
    kf_index: Optional[int]
    mode: str
    T_WC: np.ndarray
    input_shape: tuple[int, int]
    pointmap: Optional[np.ndarray] = None
    conf: Optional[np.ndarray] = None
    n_kf: int = 0
    new_keyframes: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    t_backend_wait_ms: float = 0.0


class Engine(Protocol):
    """Protokół silnika rekonstrukcji klatka po klatce."""

    def input_transform(self, full_h: int, full_w: int) -> InputTransform:
        """Przekształcenie klatki (full_h, full_w) na wejście modelu, liczone przez silnik."""
        ...

    def push_frame(self, rgb: np.ndarray, index: int, pts_s: float) -> FrameResult:
        """Przetwarza jedną klatkę RGB uint8 (H, W, 3) w pełnej rozdzielczości."""
        ...

    def finalize(self) -> dict[str, Path]:
        """Kończy przebieg, zapisuje wyniki i zwraca ścieżki do nich."""
        ...
