"""scale/factors.py i scale/cloud.py: skala osobno dla każdej klatki kluczowej, na syntetyce.

Wartości oczekiwane policzone ręcznie (komentarze przy testach).
"""

import math

import numpy as np
import pytest

from vid2cloud.export.plyio import write_ply
from vid2cloud.scale import factors as fx
from vid2cloud.scale.cloud import (
    KeyframeCloud,
    assemble,
    average_conf_mask,
    compare_with_upstream,
    keyframe_clouds,
)
from vid2cloud.scale.estimate import parse_targets

E = math.e


def obs(*items):
    return [fx.Observation(k, s, f) for k, s, f in items]


# --------------------------------------------------------------------------- #
# Obserwacje
# --------------------------------------------------------------------------- #


def test_observations_take_only_ok_with_factor_final():
    rows = [{"kf": 0, "segment": "a", "status": "ok", "factor_final": 1.5},
            {"kf": 0, "segment": "b", "status": "final: 3: za mało pikseli", "factor_final": None},
            {"kf": 1, "segment": "a", "status": "ok", "factor_final": None},
            {"kf": 2, "segment": "b", "status": "ok", "factor_final": 1.6}]
    assert fx.observations(rows) == obs((0, "a", 1.5), (2, "b", 1.6))


def test_frame_medians_even_count_is_mean_of_middle():
    """kf 0: [1.0, 1.2] -> 1.1; kf 3: [1, 2, 9] -> 2."""
    m = fx.frame_medians(obs((0, "a", 1.0), (3, "a", 1.0), (0, "b", 1.2), (3, "b", 2.0), (3, "c", 9.0)))
    assert m == {0: pytest.approx(1.1), 3: pytest.approx(2.0)}


# --------------------------------------------------------------------------- #
# Wariant globalny
# --------------------------------------------------------------------------- #


TARGETS = parse_targets({"segments": [
    {"name": "a", "length_mm": 1000.0, "marker_a": 1, "marker_b": 2},
    {"name": "b", "length_mm": 2000.0, "marker_a": 2, "marker_b": 3},
]})


def _row(kf, seg, L_mm, s, d, status="ok"):
    return {"kf": kf, "segment": seg, "status": status, "s_final": s, "d_canon_final": d,
            "factor_final": L_mm / 1000.0 / (s * d)}


def test_global_factor_median_of_segment_medians():
    """a: s·d = [0.5, 0.8, 1.0] -> mediana 0.8 -> 1.25; b: s·d = [1.0] -> 2.0; mediana(1.25, 2.0) = 1.625."""
    rows = [_row(0, "a", 1000, 1.0, 0.5), _row(1, "a", 1000, 2.0, 0.4), _row(2, "a", 1000, 0.5, 2.0),
            _row(1, "b", 2000, 1.0, 1.0), _row(2, "b", 2000, 1.0, 9.0, status="final: brak")]
    f, how, est = fx.global_factor(rows, TARGETS)
    assert f == pytest.approx(1.625)
    assert [s.name for s in est.segments] == ["a", "b"]
    assert est.factor("a").factor == pytest.approx(1.25) and est.factor("a").n == 3
    assert "mediana z 2" in how


def test_global_factor_without_pairs():
    with pytest.raises(fx.NoPairs):
        fx.global_factor([_row(0, "a", 1000, 1.0, 1.0, status="created: N = 0")], TARGETS)


# --------------------------------------------------------------------------- #
# Metoda A
# --------------------------------------------------------------------------- #


def test_interp_factors_hand_example():
    """pts 0..4; kf 1: f = 1, kf 3: f = e^0.2 -> kf 2 = e^0.1 (połowa w log), kf 0 = 1, kf 4 = e^0.2."""
    f = fx.interp_factors([0.0, 1.0, 2.0, 3.0, 4.0], obs((1, "a", 1.0), (3, "a", E ** 0.2)))
    np.testing.assert_allclose(f, [1.0, 1.0, E ** 0.1, E ** 0.2, E ** 0.2], rtol=1e-12)


def test_interp_factors_uneven_time_and_median_node():
    """pts [0, 1, 4]; kf 0: mediana [1.0, 1.2] = 1.1, kf 2: 2.2 -> kf 1 = 1.1 · 2^(1/4)."""
    f = fx.interp_factors([0.0, 1.0, 4.0], obs((0, "a", 1.0), (0, "b", 1.2), (2, "a", 2.2)))
    assert f[0] == 1.1 and f[2] == 2.2
    assert f[1] == pytest.approx(1.1 * 2 ** 0.25, rel=1e-12)


def test_interp_factors_rejects_non_increasing_time():
    with pytest.raises(ValueError):
        fx.interp_factors([0.0, 2.0, 1.0], obs((0, "a", 1.0)))


# --------------------------------------------------------------------------- #
# Metoda B
# --------------------------------------------------------------------------- #


def test_lsq_two_keyframes_hand_solution():
    """K = 2, pts [0, 1], u_0 = a, u_1 = b, σ_obs = τ = 1 -> wszystkie wagi 1.

    AᵀPA = [[2, −1], [−1, 2]], AᵀPl = [a, b] -> u = [(2a + b)/3, (a + 2b)/3];
    (AᵀPA)⁻¹ = [[2, 1], [1, 2]]/3 -> σ_u = √(2/3); v = ±(b − a)/3 (3 równania) ->
    vᵀPv = (b − a)²/3, r = 1, σ0 = |b − a|/√3.
    """
    a, b = 0.1, 0.4
    res = fx.lsq_factors([0.0, 1.0], obs((0, "x", E ** a), (1, "x", E ** b)), 1.0, 1.0)
    np.testing.assert_allclose(res.u, [(2 * a + b) / 3, (a + 2 * b) / 3], rtol=1e-12)
    np.testing.assert_allclose(res.sigma_u, [math.sqrt(2 / 3)] * 2, rtol=1e-12)
    assert res.dof == 1 and res.n_eq == 3 and res.n_obs == 2
    assert res.vtpv == pytest.approx((b - a) ** 2 / 3)
    assert res.sigma0_post == pytest.approx(abs(b - a) / math.sqrt(3))
    np.testing.assert_allclose(res.f, np.exp(res.u))


def test_lsq_weights_follow_time_gap():
    """Waga różnicy 1/(τ²·Δt): Δt = 4, τ = 0,5 -> waga 1; σ_obs = 1 -> ten sam układ co wyżej."""
    a, b = -0.2, 0.3
    res = fx.lsq_factors([10.0, 14.0], obs((0, "x", E ** a), (1, "x", E ** b)), 1.0, 0.5)
    np.testing.assert_allclose(res.u, [(2 * a + b) / 3, (a + 2 * b) / 3], rtol=1e-12)


def test_lsq_unobserved_keyframe_between_is_bridged():
    """K = 3, obserwacje tylko w kf 0 (u = 0) i kf 2 (u = 0.3), Δt równe: środek dokładnie 0.15
    (symetria), a jego σ_u większe niż klatek z obserwacją."""
    res = fx.lsq_factors([0.0, 1.0, 2.0], obs((0, "x", 1.0), (2, "x", E ** 0.3)), 0.05, 0.1)
    assert res.u[1] == pytest.approx(0.15, abs=1e-12)
    assert res.sigma_u[1] > res.sigma_u[0] and res.sigma_u[0] == pytest.approx(res.sigma_u[2])


def test_lsq_single_observation_has_no_sigma0_post():
    res = fx.lsq_factors([0.0, 5.0], obs((1, "x", 1.4)), 0.05, 0.01)
    np.testing.assert_allclose(res.f, [1.4, 1.4], rtol=1e-12)
    assert res.dof == 0 and res.sigma0_post is None


def test_lsq_without_observations():
    with pytest.raises(fx.NoPairs):
        fx.lsq_factors([0.0, 1.0], [], 0.05, 0.01)


# --------------------------------------------------------------------------- #
# Położenia klatek
# --------------------------------------------------------------------------- #


def test_keyframe_positions_hand_example():
    """t = [0, (1,0,0), (1,2,0)], f = [1, 2, 3]: p1 = 1,5·(1,0,0); p2 = p1 + 2,5·(0,2,0) = (1,5; 5; 0)."""
    p = fx.keyframe_positions([[0, 0, 0], [1, 0, 0], [1, 2, 0]], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(p, [[0, 0, 0], [1.5, 0, 0], [1.5, 5, 0]])


def test_keyframe_positions_constant_factor_scales_translations():
    t = np.random.default_rng(0).normal(size=(6, 3))
    np.testing.assert_allclose(fx.keyframe_positions(t, [1.7] * 6), 1.7 * t, rtol=1e-12)


# --------------------------------------------------------------------------- #
# σ_obs i τ
# --------------------------------------------------------------------------- #


def test_sigma_obs_estimate_hand_example():
    """Grupa [e^−0.1, 1, e^0.1]: mediana 1, reszty [−0.1, 0, 0.1], Σr² = 0.02, n = 3, m = 1, r = 2 -> σ = 0.1.
    Grupa jednoelementowa nie wchodzi."""
    out = fx.sigma_obs_estimate([[E ** -0.1, 1.0, E ** 0.1], [5.0]])
    assert out["sigma_obs"] == pytest.approx(0.1)
    assert (out["n_residuals"], out["n_frames"], out["dof"]) == (3, 1, 2)


def test_sigma_obs_estimate_pair_uses_log_of_median():
    """Para [1, 3]: mediana 2, reszty ln(1/2), ln(3/2); n = 2, m = 1, r = 1."""
    out = fx.sigma_obs_estimate([[1.0, 3.0]])
    assert out["sigma_obs"] == pytest.approx(math.sqrt(math.log(0.5) ** 2 + math.log(1.5) ** 2))


def test_tau_estimate_hand_example():
    """Δu² = [0.04, 0.01, 0], Δt = [2, 1, 2] -> τ² = 0.05 / 5 = 0.01; kolejność po pts, nie po wejściu."""
    out = fx.tau_estimate([[(2.0, E ** 0.2), (0.0, 1.0)], [(0.0, 1.0), (1.0, E ** -0.1), (3.0, E ** -0.1)]])
    assert out["tau2"] == pytest.approx(0.01) and out["tau"] == pytest.approx(0.1)
    assert out["n_increments"] == 3 and out["sum_dt"] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# Chmura: odbudowa i warianty (punkt (d) sesji S9)
# --------------------------------------------------------------------------- #


def _quat(axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return [*(axis * math.sin(angle / 2)), math.cos(angle / 2)]


def _dumps(K=4, h=6, w=8, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(K):
        T = np.array([*rng.normal(size=3), *_quat(rng.normal(size=3), rng.uniform(0, 1)), rng.uniform(0.5, 2.0)])
        out.append({"kf": k, "T_WC": T, "X_canon": rng.normal(size=(h, w, 3)).astype(np.float32),
                    "C": rng.uniform(0, 4, size=(h, w)).astype(np.float32), "N": int(rng.integers(1, 5))})
    return out


def test_average_conf_mask_strict_and_n_zero_like_upstream():
    C = np.array([[1.5, 1.6], [3.0, 2.9]], dtype=np.float32)
    assert average_conf_mask(C, 1).tolist() == [False, True, True, True]  # > 1,5, nie >=
    assert average_conf_mask(C, 2).tolist() == [False, False, False, False]
    assert average_conf_mask(np.array([0.0, 1.0], dtype=np.float32), 0).tolist() == [False, True]  # nan, inf


def test_assemble_without_factors_is_sim3_of_each_keyframe():
    dumps = _dumps()
    kc = keyframe_clouds(dumps)
    P = assemble(kc)
    from vid2cloud.scale.lift import to_world
    exp = np.concatenate([to_world(d["T_WC"], np.asarray(d["X_canon"]).reshape(-1, 3)[
        average_conf_mask(d["C"], d["N"])]) for d in dumps])
    np.testing.assert_allclose(P, exp, rtol=1e-12, atol=1e-12)


def test_methods_with_equal_factors_give_global_variant():
    """Wszystkie f_kj = c: A i B dają f_k = c, wariant globalny f = c, a chmura każdego = c × odbudowa."""
    c = 1.37
    dumps = _dumps(K=5)
    kc = keyframe_clouds(dumps)
    pts = [0.0, 1.0, 2.5, 7.0, 7.5]
    o = obs((1, "a", c), (1, "b", c), (3, "a", c))
    rows = [_row(1, "a", 1000, 1.0, 1000 / 1000 / c), _row(1, "b", 2000, 2.0, 2000 / 1000 / (2.0 * c)),
            _row(3, "a", 1000, 0.5, 1000 / 1000 / (0.5 * c))]
    f_glob, _, _ = fx.global_factor(rows, TARGETS)
    assert f_glob == pytest.approx(c, rel=1e-12)
    raw = assemble(kc)
    t = np.array([k.t for k in kc])
    f_a = fx.interp_factors(pts, o)
    f_b = fx.lsq_factors(pts, o, 0.05, 0.01).f
    for f in (np.full(5, f_glob), f_a, f_b):
        np.testing.assert_allclose(f, c, rtol=1e-12)
        np.testing.assert_allclose(assemble(kc, f, fx.keyframe_positions(t, f)), c * raw, rtol=1e-9, atol=1e-12)


def test_variant_point_formula():
    """Punkt klatki k: p_k + f_k·s_k·R_k·X — dla jednej klatki z R = I, s = 2, p = (1, 0, 0), f = 3."""
    kc = [KeyframeCloud(0, np.array([5.0, 5.0, 5.0, 0, 0, 0, 1.0, 2.0]), np.array([[1.0, 2.0, 3.0]]))]
    np.testing.assert_allclose(assemble(kc, [3.0], np.array([[1.0, 0, 0]])), [[7.0, 12.0, 18.0]])


def test_compare_with_upstream_same_and_permuted_order(tmp_path):
    P = np.random.default_rng(2).normal(size=(50, 3))
    v = np.zeros(50, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = P.T
    v["red"] = np.arange(50)
    write_ply(tmp_path / "cloud.ply", v)
    rep, col = compare_with_upstream(P, tmp_path / "cloud.ply")
    assert rep["same_order"] and rep["max_abs_diff"] < 1e-6 and col[:, 0].tolist() == list(range(50))
    rep, col = compare_with_upstream(P[::-1], tmp_path / "cloud.ply")
    assert not rep["same_order"] and col is None
    rep, col = compare_with_upstream(P[:49], tmp_path / "cloud.ply")
    assert rep["max_abs_diff"] is None and col is None
