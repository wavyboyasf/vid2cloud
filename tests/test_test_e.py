"""Test E: rangi i ρ Spearmana bez scipy. Wartości oczekiwane policzone ręcznie."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location("test_e_mod", Path(__file__).resolve().parents[1] / "eval" / "test_e.py")
te = importlib.util.module_from_spec(_SPEC)
sys.modules["test_e_mod"] = te
_SPEC.loader.exec_module(te)


def test_rankdata_ties_average():
    """[10, 20, 20, 5] -> rangi [2, 3.5, 3.5, 1]."""
    np.testing.assert_allclose(te.rankdata([10, 20, 20, 5]), [2, 3.5, 3.5, 1])


def test_spearman_monotonic_and_reversed():
    x = [1, 2, 3, 4, 5]
    assert te.spearman(x, [2, 4, 8, 16, 32]) == pytest.approx(1.0)
    assert te.spearman(x, [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_hand_value():
    """Bez remisów: ρ = 1 − 6Σd²/(n(n²−1)); x = 1..5, y-rangi [2, 1, 4, 3, 5] -> Σd² = 4, ρ = 0.8."""
    assert te.spearman([1, 2, 3, 4, 5], [20, 10, 40, 30, 50]) == pytest.approx(0.8)


def test_spearman_with_ties_matches_pearson_of_ranks():
    """x = [1, 2, 2, 3] -> rangi [1, 2.5, 2.5, 4]; y = [1, 3, 2, 4]: Pearson rang = 0.9486833."""
    assert te.spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(0.9 / np.sqrt(0.9), rel=1e-6)


def test_spearman_degenerate():
    assert te.spearman([1, 2], [1, 2]) is None
    assert te.spearman([1, 1, 1], [1, 2, 3]) is None


def test_correlations_groups_and_range():
    rows = [{"g": "a", "x": 1.0, "y": 3.0}, {"g": "a", "x": 2.0, "y": 2.0}, {"g": "a", "x": 3.0, "y": 1.0},
            {"g": "b", "x": 1.0, "y": 4.0}, {"g": "b", "x": 2.0, "y": None}]
    out = te.correlations(rows, "g", "x", "y")
    assert [o["g"] for o in out] == ["a", "b", "łącznie"]
    assert out[0]["rho"] == pytest.approx(-1.0) and out[0]["n"] == 3
    assert out[1]["rho"] is None and out[1]["n"] == 1
    assert out[2]["n"] == 4 and out[2]["y_min"] == 1.0 and out[2]["y_max"] == 4.0


def test_within_keyframe_pairs_hand_example():
    """Klatka (r, 0): x = [1, 2, 3], y = [10, 12, 9] -> pary (0,1): +·+ zgodna, (0,2): +·− niezgodna,
    (1,2): +·− niezgodna; mediana y = 10 -> odchylenia [0, +20, −10] %. Klatka (r, 1): x równe -> remis.
    Klatka (s, 0) ma 1 wiersz, (s, 1) ma drugi wiersz bez y — obie nie wchodzą."""
    rows = [{"p": "r", "kf": 0, "o": "a", "x": 1.0, "y": 10.0}, {"p": "r", "kf": 0, "o": "b", "x": 2.0, "y": 12.0},
            {"p": "r", "kf": 0, "o": "c", "x": 3.0, "y": 9.0},
            {"p": "r", "kf": 1, "o": "a", "x": 5.0, "y": 1.0}, {"p": "r", "kf": 1, "o": "b", "x": 5.0, "y": 2.0},
            {"p": "s", "kf": 0, "o": "a", "x": 1.0, "y": 1.0},
            {"p": "s", "kf": 1, "o": "a", "x": 1.0, "y": 1.0}, {"p": "s", "kf": 1, "o": "b", "x": 2.0, "y": None}]
    pairs = te.within_keyframe_pairs(rows, ("p", "kf"), "x", "y", label="o")
    assert [(q["kf"], q["o_i"], q["o_j"], q["znak"]) for q in pairs] == [
        (0, "a", "b", "zgodna"), (0, "a", "c", "niezgodna"), (0, "b", "c", "niezgodna"), (1, "a", "b", "remis")]
    assert pairs[0]["dev_i_pct"] == pytest.approx(0.0) and pairs[0]["dev_j_pct"] == pytest.approx(20.0)
    assert pairs[1]["dev_j_pct"] == pytest.approx(-10.0)
    assert pairs[3]["median_y"] == pytest.approx(1.5) and pairs[3]["dev_i_pct"] == pytest.approx(-100 / 3)

    summ = te.within_summary(pairs, "p", ("p", "kf"), ["r", "s"])
    assert summ[0] == {"p": "r", "klatki_ge2": 2, "pary": 4, "zgodne": 1, "niezgodne": 2, "remisy": 1,
                       "odch_min_pct": pytest.approx(-100 / 3), "odch_max_pct": pytest.approx(100 / 3)}
    assert summ[1]["pary"] == 0 and summ[1]["odch_min_pct"] is None
    assert summ[2]["p"] == "łącznie" and summ[2]["pary"] == 4
