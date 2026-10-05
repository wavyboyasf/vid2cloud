"""Lift punktów markera z 2D (wejście modelu, 512 px) do 3D z pointmapy klatki kluczowej.

Funkcje czyste (numpy), bez silnika i bez GPU. Wejściem są narożniki markera przeliczone
na wejście modelu (`aruco_keyframes.csv`, kolumny `*_512`) oraz pointmapa X (h, w, 3)
i ŚREDNIA ufność C (h, w) z dumpu klatki kluczowej (`kf_*_{created,final}.npz`; średnia to
C / N — `mean_confidence()` odmawia przy N = 0, które zdarza się po relokalizacji).

Konwencja pikseli
-----------------
Ta sama co w `scale/aruco.py::rescale_corners()`: środek piksela (wiersz i, kolumna j)
ma współrzędne (x = j, y = i). Pointmapa jest wyrównana z obrazem wejścia modelu piksel
w piksel, więc X[i, j] to punkt dla (x = j, y = i). Sprawdzenie na danych: wpis
w docs/decisions.md z 2026-10-05.

Metody
------
* `lift_marker()` — środek markera: przecięcie przekątnych czworokąta, punkt 3D = mediana
  (po współrzędnych) X z pikseli wewnątrz czworokąta zmniejszonego do 50 % wokół środka,
  tylko C >= c_min. Zwraca też liczbę pikseli i rozrzut (MAD odległości od mediany).
  Odporność na krawędź planszy i tło: brzeg markera nie wchodzi do próbki.
* `fit_plane_ray()` + `PlaneRayModel.point()` — narożnik albo inny punkt na planszy:
  płaszczyzna dopasowana (SVD) do tych samych pikseli wnętrza, przecięta z promieniem
  w pikselu punktu. Promień r(x, y) = X / X_z dopasowany afinicznie w (x, y) z tych samych
  pikseli — dla kamery otworkowej (z w przód) X / X_z = K⁻¹ [x, y, 1] jest dokładnie
  afiniczne, a dla płaskiej planszy przecięcie jest dokładne. Brzeg markera nie jest
  próbkowany, tylko wyznaczany z wnętrza.
* `lift_window()` — diagnostyka: mediana X w oknie 3×3 przy najbliższym pikselu.

Punkt odniesienia odcinka (`parse_reference()`) wynika z pola `reference` w targets.json:
narożniki w kolejności ArUco w układzie samego markera (0 lewy-górny, 1 prawy-górny,
2 prawy-dolny, 3 lewy-dolny — potwierdzone przez autora na podglądzie 05.10), środek
krawędzi = środek odcinka między jej dwoma narożnikami W 3D (po lifcie, nie w 2D),
„krawędź pod kątem” = górny róg tej krawędzi (uwaga w `pomiary_tasma.csv`). Taśma była
mierzona od narożników czarnego kwadratu, więc narożniki z ArUco to punkty pomiaru.
Nieznana nazwa punktu albo kwalifikator to błąd, nie zgadywanie.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Domyślny próg średniej ufności piksela (= próg chmury `WindowMsg.C_conf_threshold`).
C_MIN = 1.5
#: Czworokąt próbkowania: narożniki przysunięte do środka do tego ułamka.
SHRINK = 0.5
#: Poniżej tylu pikseli lift nie zwraca punktu (bez progu na rozrzut).
MIN_PX = 5

CORNER_INDEX = {"lewy-gorny": 0, "prawy-gorny": 1, "prawy-dolny": 2, "lewy-dolny": 3}
EDGE_CORNERS = {
    "srodek-gornej": (0, 1),
    "srodek-prawej": (1, 2),
    "srodek-dolnej": (3, 2),
    "srodek-lewej": (0, 3),
}
#: „krawędź pod kątem” -> górny róg tej krawędzi (decyzja autora 05.10, uwaga w pomiary_tasma.csv).
EDGE_TOP_CORNER = {"prawa-krawedz": 1, "lewa-krawedz": 0}
CENTER_NAMES = ("srodek",)
QUALIFIERS = ("kwadrat", "max", "pion", "poziom", "pod katem")

_PL = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


# --------------------------------------------------------------------------- #
# Punkt odniesienia z targets.json
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReferencePoint:
    """Punkt odniesienia jednego końca odcinka.

    `kind`: "center" | "corner" | "edge_mid" | "edge_top_corner"; `corners` — indeksy
    narożników ArUco, których punkty 3D się uśrednia (1 albo 2; pusty dla środka).
    """

    marker_id: int
    name: str
    kind: str
    corners: tuple[int, ...]

    @property
    def used(self) -> str:
        """Opis tego, co faktycznie jest liftowane — do kolumny CSV."""
        if self.kind == "center":
            return f"{self.marker_id}:srodek(mediana)"
        if self.kind == "edge_mid":
            a, b = self.corners
            return f"{self.marker_id}:srodek3D(n{a},n{b})"
        return f"{self.marker_id}:n{self.corners[0]}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_PL).strip().lower())


def parse_reference(reference: str, marker_a: int, marker_b: int) -> tuple[ReferencePoint, ReferencePoint]:
    """`"12:srodek-gornej -> 11:srodek-dolnej; kwadrat; max, pion"` -> dwa punkty odniesienia.

    Kolejność jak w tekście. Markery w tekście muszą być dokładnie {marker_a, marker_b}.
    Wymagany kwalifikator „kwadrat” (punkty na czarnym kwadracie). Każda nieznana nazwa
    punktu albo kwalifikator to `ValueError`.
    """
    head, _, tail = reference.partition(";")
    if "->" not in head:
        raise ValueError(f"reference {reference!r}: brak '->' między punktami odniesienia")
    ends = [e for e in head.split("->")]
    if len(ends) != 2:
        raise ValueError(f"reference {reference!r}: oczekiwano dokładnie dwóch punktów 'ID:punkt'")

    qualifiers = [_norm(q) for q in re.split(r"[;,]", tail) if q.strip()]
    unknown = [q for q in qualifiers if q not in QUALIFIERS]
    if unknown:
        raise ValueError(f"reference {reference!r}: nieobsługiwane kwalifikatory {unknown}; znam {list(QUALIFIERS)}")
    if "kwadrat" not in qualifiers:
        raise ValueError(f"reference {reference!r}: brak 'kwadrat' — obsługuję tylko punkty na czarnym kwadracie markera")

    points = []
    for end in ends:
        marker, sep, name = end.partition(":")
        if not sep:
            raise ValueError(f"reference {reference!r}: punkt {end.strip()!r} nie ma postaci 'ID:punkt'")
        try:
            marker_id = int(marker.strip())
        except ValueError:
            raise ValueError(f"reference {reference!r}: {marker.strip()!r} nie jest ID markera") from None
        name = _norm(name)
        if name in CORNER_INDEX:
            points.append(ReferencePoint(marker_id, name, "corner", (CORNER_INDEX[name],)))
        elif name in EDGE_CORNERS:
            points.append(ReferencePoint(marker_id, name, "edge_mid", EDGE_CORNERS[name]))
        elif name in EDGE_TOP_CORNER:
            if "pod katem" not in qualifiers:
                raise ValueError(f"reference {reference!r}: {name!r} bez 'pod katem' — nie wiem, który punkt krawędzi")
            points.append(ReferencePoint(marker_id, name, "edge_top_corner", (EDGE_TOP_CORNER[name],)))
        elif name in CENTER_NAMES:
            points.append(ReferencePoint(marker_id, name, "center", ()))
        else:
            known = [*CORNER_INDEX, *EDGE_CORNERS, *EDGE_TOP_CORNER, *CENTER_NAMES]
            raise ValueError(f"reference {reference!r}: nieobsługiwany punkt {name!r}; znam {known}")

    ids = {p.marker_id for p in points}
    if ids != {int(marker_a), int(marker_b)} or len(ids) != 2:
        raise ValueError(
            f"reference {reference!r}: markery {sorted(p.marker_id for p in points)} "
            f"nie zgadzają się z marker_a/marker_b = {marker_a}, {marker_b}"
        )
    return points[0], points[1]


# --------------------------------------------------------------------------- #
# Geometria czworokąta i próbka pikseli
# --------------------------------------------------------------------------- #


def _as_quad(corners) -> np.ndarray:
    q = np.asarray(corners, dtype=np.float64)
    if q.shape != (4, 2):
        raise ValueError(f"narożniki mają mieć kształt (4, 2), mają {q.shape}")
    return q


def quad_center(corners) -> np.ndarray:
    """Środek czworokąta = przecięcie przekątnych 0–2 i 1–3."""
    q = _as_quad(corners)
    d1, d2 = q[2] - q[0], q[3] - q[1]
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("przekątne czworokąta są równoległe — zdegenerowany czworokąt")
    t, _ = np.linalg.solve(A, q[1] - q[0])
    return q[0] + t * d1


def shrink_quad(corners, factor: float = SHRINK) -> np.ndarray:
    """Narożniki przysunięte do środka (przecięcia przekątnych) do ułamka `factor`."""
    q = _as_quad(corners)
    c = quad_center(q)
    return c + factor * (q - c)


def pixels_in_quad(corners, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(wiersze, kolumny) pikseli, których środki (x = j, y = i) leżą w czworokącie wypukłym."""
    q = _as_quad(corners)
    h, w = int(shape[0]), int(shape[1])
    j0 = max(int(np.floor(q[:, 0].min())), 0)
    j1 = min(int(np.ceil(q[:, 0].max())), w - 1)
    i0 = max(int(np.floor(q[:, 1].min())), 0)
    i1 = min(int(np.ceil(q[:, 1].max())), h - 1)
    if j1 < j0 or i1 < i0:
        return np.empty(0, int), np.empty(0, int)
    ii, jj = np.mgrid[i0:i1 + 1, j0:j1 + 1]
    x, y = jj.ravel().astype(np.float64), ii.ravel().astype(np.float64)
    signs = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        signs.append((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]))
    s = np.array(signs)
    inside = np.all(s >= -1e-9, axis=0) | np.all(s <= 1e-9, axis=0)
    return ii.ravel()[inside], jj.ravel()[inside]


def mean_confidence(C_sum: np.ndarray, N: int) -> np.ndarray:
    """Średnia ufność C / N z dumpu. N <= 0 (relokalizacja) to `ValueError`, nie dzielenie przez zero."""
    if int(N) <= 0:
        raise ValueError(f"N = {int(N)}: średnia ufność C / N nieokreślona (klatka z relokalizacji)")
    return np.asarray(C_sum, dtype=np.float64) / int(N)


def _check_maps(X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    C = np.asarray(C)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(f"pointmapa ma mieć kształt (h, w, 3), ma {X.shape}")
    if C.shape != X.shape[:2]:
        raise ValueError(f"ufność ma kształt {C.shape}, pointmapa {X.shape[:2]}")
    return X, C


def marker_samples(corners, X, C, c_min: float = C_MIN, shrink: float = SHRINK):
    """Piksele wnętrza markera: (uv (n, 2), P (n, 3) float64, n_w_czworokacie).

    Wnętrze = czworokąt zmniejszony do `shrink`; zostają piksele z C >= c_min i skończonym X.
    """
    X, C = _check_maps(X, C)
    rows, cols = pixels_in_quad(shrink_quad(corners, shrink), X.shape[:2])
    P = X[rows, cols].astype(np.float64)
    ok = (C[rows, cols] >= c_min) & np.all(np.isfinite(P), axis=1)
    uv = np.stack([cols[ok], rows[ok]], axis=1).astype(np.float64)
    return uv, P[ok], int(rows.size)


def _mad(P: np.ndarray, center: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(P - center, axis=1)))


# --------------------------------------------------------------------------- #
# Lift
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LiftResult:
    """Punkt 3D w układzie kamery klatki kluczowej + diagnostyka. `point is None` -> `reason`."""

    point: Optional[np.ndarray]
    n_px: int
    mad: Optional[float]
    reason: str = ""


def lift_marker(corners_512, X, C, c_min: float = C_MIN, shrink: float = SHRINK,
                min_px: int = MIN_PX) -> LiftResult:
    """Środek markera: mediana X w czworokącie 50 % wokół przecięcia przekątnych, C >= c_min."""
    _, P, n_quad = marker_samples(corners_512, X, C, c_min, shrink)
    n = len(P)
    if n < min_px:
        return LiftResult(None, n, None, f"{n} px z C >= {c_min} (z {n_quad} w czworokącie {shrink:.0%}) < {min_px}")
    point = np.median(P, axis=0)
    return LiftResult(point, n, _mad(P, point))


@dataclass(frozen=True)
class PlaneRayModel:
    """Płaszczyzna planszy (p0, normal) i afiniczny model promienia r(x, y) = X / X_z."""

    p0: np.ndarray
    normal: np.ndarray
    ray_coef: np.ndarray  # (3, 2): [x, y, 1] @ ray_coef = (X/X_z, Y/X_z)
    n_px: int
    mad: float
    plane_rms: float
    ray_rms: float

    def ray(self, uv) -> np.ndarray:
        x, y = np.asarray(uv, dtype=np.float64)
        rxy = np.array([x, y, 1.0]) @ self.ray_coef
        return np.array([rxy[0], rxy[1], 1.0])

    def point(self, uv) -> np.ndarray:
        """Przecięcie promienia w pikselu `uv` (x, y) z płaszczyzną planszy."""
        r = self.ray(uv)
        denom = float(self.normal @ r)
        if abs(denom) < 1e-12:
            raise ValueError(f"promień w {tuple(uv)} równoległy do płaszczyzny planszy")
        return r * float(self.normal @ self.p0) / denom


def fit_plane_ray(corners_512, X, C, c_min: float = C_MIN, shrink: float = SHRINK,
                  min_px: int = MIN_PX) -> tuple[Optional[PlaneRayModel], str]:
    """Model płaszczyzna + promień z pikseli wnętrza markera. Zwraca (model, "") albo (None, powód)."""
    uv, P, n_quad = marker_samples(corners_512, X, C, c_min, shrink)
    front = P[:, 2] > 0
    n_behind = int((~front).sum())
    uv, P = uv[front], P[front]
    n = len(P)
    if n < min_px:
        extra = f", {n_behind} z z <= 0" if n_behind else ""
        return None, f"{n} px z C >= {c_min} (z {n_quad} w czworokącie {shrink:.0%}{extra}) < {min_px}"
    A = np.column_stack([uv, np.ones(n)])
    if np.linalg.matrix_rank(A) < 3:
        return None, "piksele wnętrza współliniowe — nie da się dopasować promienia"

    p0 = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - p0, full_matrices=False)
    normal = vt[-1]
    plane_rms = float(np.sqrt(np.mean(((P - p0) @ normal) ** 2)))

    rays = P[:, :2] / P[:, 2:3]
    coef, *_ = np.linalg.lstsq(A, rays, rcond=None)
    ray_rms = float(np.sqrt(np.mean(np.sum((A @ coef - rays) ** 2, axis=1))))
    median = np.median(P, axis=0)
    return PlaneRayModel(p0, normal, coef, n, _mad(P, median), plane_rms, ray_rms), ""


def lift_window(uv, X, C, c_min: float = C_MIN, half: int = 1) -> LiftResult:
    """Diagnostyka: mediana X w oknie (2·half+1)² wokół najbliższego piksela, tylko C >= c_min."""
    X, C = _check_maps(X, C)
    h, w = X.shape[:2]
    j, i = (int(round(v)) for v in np.asarray(uv, dtype=np.float64))
    i0, i1, j0, j1 = max(i - half, 0), min(i + half, h - 1), max(j - half, 0), min(j + half, w - 1)
    if i1 < i0 or j1 < j0:
        return LiftResult(None, 0, None, f"okno wokół {tuple(uv)} poza obrazem {h}x{w}")
    P = X[i0:i1 + 1, j0:j1 + 1].reshape(-1, 3).astype(np.float64)
    ok = (C[i0:i1 + 1, j0:j1 + 1].reshape(-1) >= c_min) & np.all(np.isfinite(P), axis=1)
    P = P[ok]
    if len(P) == 0:
        return LiftResult(None, 0, None, f"brak pikseli z C >= {c_min} w oknie wokół {tuple(uv)}")
    point = np.median(P, axis=0)
    return LiftResult(point, len(P), _mad(P, point))


@dataclass(frozen=True)
class ReferenceLift:
    """Punkt odniesienia po lifcie: metoda główna (`point`) i diagnostyczna mediana 3×3."""

    ref: ReferencePoint
    point: Optional[np.ndarray]
    point_win3: Optional[np.ndarray]
    n_px: int
    mad: Optional[float]
    plane_rms: Optional[float]
    ray_rms: Optional[float]
    n_px_win3: int
    reason: str = ""


def lift_reference(ref: ReferencePoint, corners_512, X, C, c_min: float = C_MIN,
                   shrink: float = SHRINK, min_px: int = MIN_PX) -> ReferenceLift:
    """Punkt odniesienia w 3D (układ kamery klatki kluczowej).

    Środek markera — `lift_marker()`; narożnik / górny róg krawędzi — płaszczyzna + promień;
    środek krawędzi — środek odcinka między dwoma narożnikami po lifcie (w 3D). Obok, jako
    diagnostyka, to samo z mediany w oknie 3×3 (dla środka krawędzi też środek w 3D).
    """
    q = _as_quad(corners_512)
    if ref.kind == "center":
        main = lift_marker(q, X, C, c_min, shrink, min_px)
        win = lift_window(quad_center(q), X, C, c_min)
        return ReferenceLift(ref, main.point, win.point, main.n_px, main.mad, None, None, win.n_px, main.reason)

    model, reason = fit_plane_ray(q, X, C, c_min, shrink, min_px)
    wins = [lift_window(q[k], X, C, c_min) for k in ref.corners]
    point_win3 = None
    if all(wl.point is not None for wl in wins):
        point_win3 = np.mean([wl.point for wl in wins], axis=0)
    n_win = min(wl.n_px for wl in wins)
    if model is None:
        n_px = len(marker_samples(q, X, C, c_min, shrink)[1])
        return ReferenceLift(ref, None, point_win3, n_px, None, None, None, n_win, reason)
    point = np.mean([model.point(q[k]) for k in ref.corners], axis=0)
    return ReferenceLift(ref, point, point_win3, model.n_px, model.mad, model.plane_rms, model.ray_rms, n_win)


# --------------------------------------------------------------------------- #
# Układ świata
# --------------------------------------------------------------------------- #


def quat_to_matrix(q) -> np.ndarray:
    """Kwaternion (qx, qy, qz, qw) — kolejność lietorch — na macierz obrotu 3×3."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def to_world(T_WC, X) -> np.ndarray:
    """Punkty (..., 3) z układu kamery do świata pozą Sim(3) [tx ty tz qx qy qz qw s]: s·R·X + t."""
    T = np.asarray(T_WC, dtype=np.float64).reshape(-1)
    if T.shape != (8,):
        raise ValueError(f"poza Sim(3) ma 8 liczb, ma {T.shape}")
    R = quat_to_matrix(T[3:7])
    X = np.asarray(X, dtype=np.float64)
    return T[7] * X @ R.T + T[:3]
