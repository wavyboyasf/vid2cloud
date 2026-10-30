"""Pomocnicze funkcje przebiegu wstępnego (`vid2cloud/input/quality.py`) na syntetyce.

Bez nagrań i bez GPU: korelacja rang, odstępy PTS, odcinki odrzuceń, zapis/odczyt
quality.csv i komenda `select` na sztucznym quality.csv.
"""

import csv

import numpy as np
import pytest

from vid2cloud.input import quality as qm
from vid2cloud.input import selection as sel


def test_spearman_monotonic_and_reversed():
    a = np.arange(10.0)
    assert qm.spearman(a, a ** 3 + 7) == pytest.approx(1.0)
    assert qm.spearman(a, -np.exp(a)) == pytest.approx(-1.0)


def test_spearman_with_ties():
    """b = [5, 6, 7, 8, 7] -> rangi [1, 2, 3,5, 5, 3,5]; rangi a = [1..5].
    Odchylenia od średniej 3: a [−2, −1, 0, 1, 2], b [−2, −1, 0,5, 2, 0,5].
    r = 8 / sqrt(10 · 9,5) = 0,820782..."""
    assert qm.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(8 / np.sqrt(95), rel=1e-12)


def test_gap_stats():
    """Klatki co 1/30 s, jedna pominięta: dt = [33,3; 33,3; 66,7; 33,3] ms, mediana 33,3,
    próg 1,5 × 33,3 = 50 ms -> jedna luka, przed klatką 3, równa 66,7 ms."""
    pts = np.array([0, 1, 2, 4, 5]) / 30.0
    g = qm.gap_stats(pts)
    assert g["dt_median_ms"] == pytest.approx(1000 / 30)
    assert g["gap_threshold_ms"] == pytest.approx(50.0)
    assert g["n_gaps"] == 1
    assert g["gap_frames"] == [3]
    assert g["gap_max_ms"] == pytest.approx(2000 / 30)


def test_gap_stats_without_gaps():
    g = qm.gap_stats(np.arange(100) / 30.0)
    assert g["n_gaps"] == 0 and g["gap_max_ms"] is None and g["gap_frames"] == []


def test_false_runs():
    assert qm.false_runs([True, False, False, True, False]) == [(1, 3), (4, 5)]
    assert qm.false_runs([True, True]) == []
    assert qm.false_runs([False]) == [(0, 1)]


def _rows(n=20):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append({
            "frame": i, "pts_s": 0.0083 + i / 30.0, "dt_ms": None if i == 0 else 1000 / 30,
            "tenengrad": float(rng.uniform(100, 200)), "lapvar": float(rng.uniform(50, 80)),
            "bright_median": 120.0, "frac_dark": 0.0, "frac_bright": 0.001,
            "keep": i not in (3, 4, 5), "t_measure_ms": 50.0,
        })
    return rows


def test_quality_csv_roundtrip(tmp_path):
    rows = _rows()
    path = tmp_path / "quality.csv"
    qm.write_quality_csv(path, rows)
    q = qm.read_quality_csv(path)
    assert q["frame"].tolist() == list(range(20))
    assert np.isnan(q["dt_ms"][0]) and q["dt_ms"][1] == pytest.approx(33.3333, abs=1e-4)
    assert q["keep"].tolist() == [r["keep"] for r in rows]
    assert np.allclose(q["tenengrad"], [r["tenengrad"] for r in rows], rtol=1e-8)


def test_run_select_writes_lists_and_manifest(tmp_path):
    """20 klatek, k = 3: stały krok 7/7/6, okna 7 (ostatnie jednoelementowe), okno 1 = klatki 3–5
    w całości keep = False -> sharp_thr_k3 ma 6 klatek i jedno okno pominięte."""
    rows = _rows()
    qcsv = tmp_path / "quality.csv"
    qm.write_quality_csv(qcsv, rows)
    rep = qm.run_select(qcsv, 3, tmp_path / "frames")
    out = tmp_path / "frames"
    with open(out / "manifest.csv", newline="", encoding="utf-8") as f:
        manifest = {r["wariant"]: r for r in csv.DictReader(f)}
    assert {k: int(v["n_klatek"]) for k, v in manifest.items()} == {
        "fixed_k3_o0": 7, "fixed_k3_o1": 7, "fixed_k3_o2": 6, "sharp_k3": 7, "sharp_thr_k3": 6,
    }
    assert manifest["fixed_k3_o1"]["offset"] == "1" and manifest["fixed_k3_o1"]["n_okien_pominietych"] == ""
    assert manifest["sharp_thr_k3"]["offset"] == "" and manifest["sharp_thr_k3"]["n_okien_pominietych"] == "1"
    assert sel.parse_frame_list((out / "fixed_k3_o2.txt").read_text()) == [2, 5, 8, 11, 14, 17]
    ten = np.array([r["tenengrad"] for r in rows])
    sharp = sel.parse_frame_list((out / "sharp_k3.txt").read_text())
    assert sharp == [int(np.argmax(ten[s:s + 3])) + s for s in range(0, 20, 3)]
    with open(out / "sharp_thr_k3_pominiete_okna.csv", newline="", encoding="utf-8") as f:
        skipped = list(csv.DictReader(f))
    assert [(r["start"], r["stop"]) for r in skipped] == [("3", "6")]
    assert rep["common_sharp_k3_fixed_k3_o0"] == len(set(sharp) & set(range(0, 20, 3)))
    with pytest.raises(FileExistsError):
        qm.run_select(qcsv, 3, out)
