"""Zgrubne dopasowanie z par punktów (Umeyama): znane podobieństwo + szum 1 mm.

Punkty jak w sali: naroża i krawędzie w prostopadłościanie 20 × 10 × 8 m (TLS, metry),
rekonstrukcja w jednostkach sieci (skala ok. 1,5 m / jedn., jak factor_final z 03.10).
Początek układu rekonstrukcji (kamera pierwszej klatki kluczowej) leży w sali: t = obraz
tego punktu w TLS. Translacja jest porównywana w tym punkcie — poza salą błąd obrotu
mnożyłby się przez odległość i test mierzyłby ekstrapolację, nie dopasowanie.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "coarse_from_pairs", Path(__file__).resolve().parents[1] / "eval" / "coarse_from_pairs.py")
cfp = importlib.util.module_from_spec(_SPEC)
sys.modules["coarse_from_pairs"] = cfp
_SPEC.loader.exec_module(cfp)

SIGMA = 0.001  # 1 mm na każdej osi, na punktach TLS


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


S_TRUE = 1.53
R_TRUE = _rotation([0.3, -1.0, 0.4], np.radians(117.0))
T_TRUE = np.array([6.0, 4.0, 1.5])  # kamera kf 0 w sali, 1,5 m nad podłogą


def _pairs(n, seed=0):
    rng = np.random.default_rng(seed)
    tls = rng.uniform([0, 0, 0], [20, 10, 8], size=(n, 3))
    rec = (tls - T_TRUE) @ R_TRUE / S_TRUE  # odwrotność y = s R x + t
    return rec, tls + rng.normal(0, SIGMA, size=tls.shape)


def test_identity_without_noise():
    rec, _ = _pairs(8)
    tls = cfp.apply(cfp.to_matrix(S_TRUE, R_TRUE, T_TRUE), rec)
    s, R, t = cfp.umeyama(rec, tls)
    assert s == pytest.approx(S_TRUE, rel=1e-12)
    np.testing.assert_allclose(R, R_TRUE, atol=1e-12)
    np.testing.assert_allclose(t, T_TRUE, atol=1e-10)


def _errors(n, seed):
    rec, tls = _pairs(n, seed)
    s, R, t = cfp.umeyama(rec, tls)
    assert np.linalg.det(R) == pytest.approx(1.0)
    angle = np.arccos(np.clip((np.trace(R.T @ R_TRUE) - 1) / 2, -1, 1))
    return abs(s / S_TRUE - 1), angle, np.linalg.norm(t - T_TRUE)


@pytest.mark.parametrize("seed", range(5))
def test_recovered_to_1e3(seed):
    """50 par: skala (względnie), obrót [rad] i translacja [m] odtworzone do 1e-3."""
    ds, angle, dt = _errors(50, seed)
    assert ds < 1e-3
    assert angle < 1e-3
    assert dt < 1e-3


@pytest.mark.parametrize("seed", range(5))
def test_eight_pairs(seed):
    """8 par (jak w instrukcji): skala i obrót do 1e-3; translacja na poziomie szumu.

    Błąd translacji to co najmniej błąd środka ciężkości, σ·√3/√n = 0,61 mm przy n = 8,
    więc 1e-3 m nie jest tu osiągalne z definicji (w 20 losowaniach 0,3–1,8 mm). Granica
    to trzykrotność tej wartości.
    """
    ds, angle, dt = _errors(8, seed)
    assert ds < 1e-3
    assert angle < 1e-3
    assert dt < 3 * SIGMA * np.sqrt(3) / np.sqrt(8)


def test_residuals_match_noise():
    """RMS odchyłek ≈ σ·√3·√(1 − 7/(3n)) (3n obserwacji, 7 parametrów); n = 400 dla stabilności."""
    n = 400
    rec, tls = _pairs(n, seed=11)
    rep = cfp.coarse_report(rec, tls)
    expected = SIGMA * np.sqrt(3) * np.sqrt(1 - 7 / (3 * n))
    assert rep["rms_m"] == pytest.approx(expected, rel=0.1)
    assert len(rep["residuals_m"]) == n
    assert rep["scale"] == pytest.approx(cfp.matrix_scale(np.array(rep["matrix"])), rel=1e-12)


def test_mirror_is_not_returned():
    """Dla płaskiego układu (wszystkie punkty na podłodze) wynik ma det(R) = +1, nie odbicie."""
    rng = np.random.default_rng(3)
    tls = np.column_stack([rng.uniform(0, 20, 8), rng.uniform(0, 10, 8), np.zeros(8)])
    rec = (tls - T_TRUE) @ R_TRUE / S_TRUE
    s, R, t = cfp.umeyama(rec, tls)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(cfp.apply(cfp.to_matrix(s, R, t), rec), tls, atol=1e-9)


def test_collinear_and_too_few_rejected():
    with pytest.raises(ValueError, match="co najmniej"):
        cfp.umeyama(np.zeros((2, 3)), np.zeros((2, 3)))
    line = np.outer(np.arange(5.0), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="współliniowe"):
        cfp.umeyama(line, line * 2)


def test_read_cloudcompare_exports(tmp_path):
    """x,y,z; indeks,x,y,z; etykieta,x,y,z — brane są trzy ostatnie pola, kolejność z pliku."""
    (tmp_path / "a.txt").write_text("1.5,2.5,3.5\n-1,0,2e-3\n\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("0,1.5,2.5,3.5\n1,-1,0,0.002\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("# komentarz\nPoint #0,1.5,2.5,3.5\nPoint #1,-1,0,0.002\n", encoding="utf-8")
    expected = np.array([[1.5, 2.5, 3.5], [-1, 0, 0.002]])
    for name in "abc":
        np.testing.assert_allclose(cfp.read_picked_points(tmp_path / f"{name}.txt"), expected)
    (tmp_path / "bad.txt").write_text("1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mniej niż 3"):
        cfp.read_picked_points(tmp_path / "bad.txt")


def test_matrix_roundtrip(tmp_path):
    M = cfp.to_matrix(S_TRUE, R_TRUE, T_TRUE)
    cfp.write_matrix(tmp_path / "m.txt", M)
    np.testing.assert_allclose(cfp.read_matrix(tmp_path / "m.txt"), M, rtol=1e-11, atol=1e-11)
    assert len((tmp_path / "m.txt").read_text().splitlines()) == 4


def test_cli_writes_matrix_and_report(tmp_path):
    rec, tls = _pairs(7, seed=2)
    np.savetxt(tmp_path / "r.txt", rec, delimiter=",", fmt="%.9f")
    np.savetxt(tmp_path / "t.txt", tls, delimiter=",", fmt="%.9f")
    out = tmp_path / "coarse" / "nagranie.txt"
    assert cfp.main(["nagranie", "--rec", str(tmp_path / "r.txt"), "--tls", str(tmp_path / "t.txt"),
                     "--out", str(out)]) == 0
    M = cfp.read_matrix(out)
    assert cfp.matrix_scale(M) == pytest.approx(S_TRUE, rel=1e-3)
    assert (tmp_path / "coarse" / "nagranie_report.json").exists()


def test_metric_coarse_maps_scaled_cloud_like_base():
    """M_m · (f·x) = M_c · x dla dowolnego podobieństwa M_c; skala macierzy dzieli się przez f."""
    spec = importlib.util.spec_from_file_location(
        "coarse_metric", Path(__file__).resolve().parents[1] / "eval" / "coarse_metric.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    rng = np.random.default_rng(3)
    s, R, t = 1.48, _rotation(rng.normal(size=3), 0.7), rng.normal(size=3)
    M_c = np.eye(4)
    M_c[:3, :3], M_c[:3, 3] = s * R, t
    x = rng.normal(size=(20, 3))
    f = 1.6473
    M_m = cm.metric_coarse(M_c, f)
    hom = lambda M, P: P @ M[:3, :3].T + M[:3, 3]  # noqa: E731
    np.testing.assert_allclose(hom(M_m, f * x), hom(M_c, x), rtol=1e-12, atol=1e-12)
    assert cm.matrix_scale(M_m) == pytest.approx(s / f)
