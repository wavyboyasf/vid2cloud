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
