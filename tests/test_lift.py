"""scale/lift.py na syntetycznej pointmapie: kamera otworkowa, płaska plansza w znanym miejscu.

Konwencja pikseli jak w module: środek piksela (i, j) to (x = j, y = i), więc rzut
x = f·X/Z + cx daje współrzędne wprost w tej konwencji.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from vid2cloud.scale.lift import (
    fit_plane_ray,
    lift_marker,
    lift_reference,
    lift_window,
    mean_confidence,
    parse_reference,
    pixels_in_quad,
    quad_center,
    shrink_quad,
    to_world,
)

H, W = 288, 512
F, CX, CY = 400.0, 255.5, 143.5
TARGETS = Path(__file__).resolve().parents[1] / "data" / "2026-09-21" / "notes" / "targets.json"


def rays() -> np.ndarray:
    ii, jj = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.stack([(jj - CX) / F, (ii - CY) / F, np.ones_like(jj)], axis=-1)


def plane_pointmap(normal, dist) -> np.ndarray:
    """X (H, W, 3) dla płaszczyzny n·X = dist widzianej przez kamerę otworkową."""
    d = rays()
    n = np.asarray(normal, dtype=np.float64)
    return d * (dist / (d @ n))[..., None]


def project(P) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    return np.stack([F * P[..., 0] / P[..., 2] + CX, F * P[..., 1] / P[..., 2] + CY], axis=-1)


def square(center_px, side_px) -> np.ndarray:
    x, y = center_px
    h = side_px / 2
    return np.array([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]], dtype=np.float64)


C_OK = np.full((H, W), 2.0)


# --------------------------------------------------------------------------- #
# Środek markera (mediana) — płaszczyzna czołowa w znanej odległości
# --------------------------------------------------------------------------- #


def test_center_segment_on_fronto_parallel_plane():
    z = 2.5
    X = plane_pointmap([0, 0, 1], z)
    qa, qb = square((100, 120), 24), square((400, 150), 30)
    la, lb = lift_marker(qa, X, C_OK), lift_marker(qb, X, C_OK)
    true_a = np.array([(100 - CX) / F, (120 - CY) / F, 1.0]) * z
    true_b = np.array([(400 - CX) / F, (150 - CY) / F, 1.0]) * z
    np.testing.assert_allclose(la.point, true_a, atol=1e-12)
    np.testing.assert_allclose(lb.point, true_b, atol=1e-12)
    assert abs(np.linalg.norm(la.point - lb.point) - np.linalg.norm(true_a - true_b)) < 1e-6
    assert la.n_px == 13 * 13  # czworokąt 50 %: 12 px boku, środki pikseli 94..106
    assert la.mad > 0 and la.reason == ""


def test_lift_marker_too_few_pixels():
    X = plane_pointmap([0, 0, 1], 2.0)
    res = lift_marker(square((200, 100), 3), X, C_OK)  # czworokąt 50 %: 1,5 px boku
    assert res.point is None and res.n_px < 5
    assert "< 5" in res.reason


def test_lift_marker_ignores_low_confidence():
    X = plane_pointmap([0, 0, 1], 2.0)
    C = C_OK.copy()
    C[:, :200] = 1.0  # lewa część obrazu poniżej progu 1.5
    q = square((200, 100), 24)  # czworokąt 50 %: kolumny 194..206
    full, part = lift_marker(q, X, C_OK), lift_marker(q, X, C)
    assert part.n_px == 13 * 7  # zostają kolumny 200..206
    assert part.n_px < full.n_px
    res = lift_marker(q, X, np.ones((H, W)))
    assert res.point is None and "C >= 1.5" in res.reason


# --------------------------------------------------------------------------- #
# Narożniki i środki krawędzi (płaszczyzna + promień) — plansza skośna
# --------------------------------------------------------------------------- #


def tilted_board(center3d, side_m, normal):
    """Kwadrat na płaszczyźnie o normalnej `normal`: narożniki 3D w kolejności ArUco (TL, TR, BR, BL)."""
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    right = np.cross([0, 1, 0], n)
    right /= np.linalg.norm(right)
    if right[0] < 0:
        right = -right  # w obrazie w prawo (x kamery w prawo)
    down = np.cross(n, right)  # w obrazie w dół (y kamery w dół)
    if down[1] < 0:
        down = -down
    h = side_m / 2
    c = np.asarray(center3d, dtype=np.float64)
    return np.array([c - h * right - h * down, c + h * right - h * down, c + h * right + h * down, c - h * right + h * down])


@pytest.fixture(scope="module")
def tilted():
    normal = np.array([np.sin(np.radians(35)), np.sin(np.radians(-20)), -np.cos(np.radians(35))])
    normal /= np.linalg.norm(normal)
    a3d = tilted_board([-0.4, 0.05, 3.0], 0.15, normal)
    b3d = tilted_board([0.5, -0.1, 3.0], 0.15, normal)
    # Obie plansze na tej samej płaszczyźnie: n·X = n·(środek) — dla b przesuwamy wzdłuż normalnej.
    dist = float(normal @ a3d.mean(axis=0))
    b3d = b3d - normal * (normal @ b3d.mean(axis=0) - dist)
    X = plane_pointmap(normal, dist)
    return X, a3d, b3d


def test_plane_ray_corners_reproduce_segment(tilted):
    X, a3d, b3d = tilted
    qa, qb = project(a3d), project(b3d)
    ref_a, ref_b = parse_reference("3:lewy-gorny -> 10:prawy-gorny; kwadrat; max, poziom", 3, 10)
    la, lb = lift_reference(ref_a, qa, X, C_OK), lift_reference(ref_b, qb, X, C_OK)
    np.testing.assert_allclose(la.point, a3d[0], atol=1e-9)
    np.testing.assert_allclose(lb.point, b3d[1], atol=1e-9)
    true = np.linalg.norm(a3d[0] - b3d[1])
    assert abs(np.linalg.norm(la.point - lb.point) - true) < 1e-6
    assert la.plane_rms < 1e-9 and la.ray_rms < 1e-9
    assert la.n_px >= 5 and la.n_px_win3 > 0


def test_edge_midpoint_is_3d_midpoint_of_lifted_corners(tilted):
    X, a3d, b3d = tilted
    qa, qb = project(a3d), project(b3d)
    ref_a, ref_b = parse_reference("12:srodek-gornej -> 11:srodek-dolnej; kwadrat; max, pion", 11, 12)
    la, lb = lift_reference(ref_a, qa, X, C_OK), lift_reference(ref_b, qb, X, C_OK)
    np.testing.assert_allclose(la.point, (a3d[0] + a3d[1]) / 2, atol=1e-9)
    np.testing.assert_allclose(lb.point, (b3d[3] + b3d[2]) / 2, atol=1e-9)
    # Środek krawędzi w 2D (na obrazie) to nie ten sam punkt — perspektywa go przesuwa.
    model, _ = fit_plane_ray(qa, X, C_OK)
    mid_2d = model.point((qa[0] + qa[1]) / 2)
    assert np.linalg.norm(mid_2d - la.point) > 1e-6


def test_edge_top_corner(tilted):
    X, a3d, b3d = tilted
    ref_a, ref_b = parse_reference("8:prawa-krawedz -> 3: lewa-krawedz; kwadrat, pod katem", 8, 3)
    la = lift_reference(ref_a, project(a3d), X, C_OK)
    lb = lift_reference(ref_b, project(b3d), X, C_OK)
    np.testing.assert_allclose(la.point, a3d[1], atol=1e-9)  # prawa krawędź -> prawy górny
    np.testing.assert_allclose(lb.point, b3d[0], atol=1e-9)  # lewa krawędź -> lewy górny


def test_plane_ray_float32_pointmap(tilted):
    X, a3d, _ = tilted
    model, reason = fit_plane_ray(project(a3d), X.astype(np.float32), C_OK)
    assert reason == ""
    err = np.linalg.norm(model.point(project(a3d)[0]) - a3d[0])
    assert err < 1e-5  # pointmapy w dumpach są float32


def test_window_median_at_pixel_center():
    X = plane_pointmap([0, 0, 1], 2.0)
    res = lift_window((300.0, 50.0), X, C_OK)
    np.testing.assert_allclose(res.point, X[50, 300], atol=1e-12)
    assert res.n_px == 9
    edge = lift_window((0.2, 0.4), X, C_OK)  # przycięte do obrazu
    assert edge.n_px == 4


# --------------------------------------------------------------------------- #
# Parser reference
# --------------------------------------------------------------------------- #


def test_parse_real_references():
    if not TARGETS.exists():
        pytest.skip(f"brak {TARGETS}")
    got = {}
    for seg in json.loads(TARGETS.read_text(encoding="utf-8"))["segments"]:
        a, b = parse_reference(seg["reference"], seg["marker_a"], seg["marker_b"])
        got[seg["name"]] = (a.used, b.used)
    assert got == {
        "11-12": ("12:srodek3D(n0,n1)", "11:srodek3D(n3,n2)"),
        "3-10": ("3:n0", "10:n1"),
        "3-8": ("8:n1", "3:n0"),
        "10-11": ("10:n1", "11:n0"),
        "8-9": ("9:n0", "8:n1"),
    }


@pytest.mark.parametrize(
    "reference, msg",
    [
        ("3:lewy-gorny -> 10:srodek-planszy; kwadrat", "nieobsługiwany punkt"),
        ("3:lewy-gorny -> 10:prawy-gorny; kartka", "nieobsługiwane kwalifikatory"),
        ("3:lewy-gorny -> 10:prawy-gorny; max", "brak 'kwadrat'"),
        ("3:lewy-gorny -> 9:prawy-gorny; kwadrat", "nie zgadzają się"),
        ("3:prawa-krawedz -> 10:lewa-krawedz; kwadrat", "bez 'pod katem'"),
        ("3 lewy-gorny -> 10:prawy-gorny; kwadrat", "'ID:punkt'"),
        ("3:lewy-gorny, 10:prawy-gorny; kwadrat", "brak '->'"),
    ],
)
def test_parse_reference_rejects(reference, msg):
    with pytest.raises(ValueError, match=msg):
        parse_reference(reference, 3, 10)


def test_parse_reference_center_and_diacritics():
    a, b = parse_reference("3: Środek -> 10:lewy-górny; kwadrat", 3, 10)
    assert a.kind == "center" and b.corners == (0,)


# --------------------------------------------------------------------------- #
# Pomocnicze
# --------------------------------------------------------------------------- #


def test_mean_confidence_refuses_n_zero():
    C = np.full((2, 2), 8.0)
    np.testing.assert_allclose(mean_confidence(C, 4), 2.0)
    with pytest.raises(ValueError, match="N = 0"):
        mean_confidence(C, 0)


def test_quad_center_is_diagonal_intersection():
    q = np.array([[10.0, 10.0], [50.0, 14.0], [46.0, 60.0], [8.0, 40.0]])
    c = quad_center(q)
    for a, b in ((q[0], q[2]), (q[1], q[3])):
        d = b - a
        assert abs(d[0] * (c[1] - a[1]) - d[1] * (c[0] - a[0])) < 1e-9
    np.testing.assert_allclose(quad_center(shrink_quad(q, 0.5)), c, atol=1e-9)


def test_pixels_in_quad_orientation_independent():
    q = square((20, 30), 10)
    r1, c1 = pixels_in_quad(q, (H, W))
    r2, c2 = pixels_in_quad(q[::-1], (H, W))
    assert len(r1) == 11 * 11
    assert set(zip(r1, c1)) == set(zip(r2, c2))


def test_to_world_sim3():
    X = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]])
    np.testing.assert_allclose(to_world([0, 0, 0, 0, 0, 0, 1, 1], X), X)
    s, t = 2.0, np.array([0.5, -1.0, 4.0])
    q = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # 90° wokół z
    out = to_world([*t, *q, s], X)
    np.testing.assert_allclose(out, [[0.5, 1.0, 4.0], [-3.5, -1.0, 10.0]], atol=1e-12)
