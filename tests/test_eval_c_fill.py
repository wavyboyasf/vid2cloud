"""Część czysta testu C: regresja, --fill i regeneracja results.csv (bez ffmpeg i silnika)."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "eval" / "test_c_start_point.py"
_spec = importlib.util.spec_from_file_location("test_c_start_point", _path)
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["clip"]: r for r in csv.DictReader(f)}


def _results(out, starts, extra=None):
    header = tc.RESULT_COLUMNS + (["note"] if extra else [])
    rows = [{"clip": f"clip_{s}", "start_s": str(s), "end_s": "30", "length_s": str(30 - s),
             "note": (extra or {}).get(s, "")} for s in starts]
    tc.write_csv(out / "results.csv", header, rows)


def test_ols_exact_line():
    b, a = tc.ols([1.0, 10.0, 19.0], [1.01, 1.10, 1.19])
    assert b == pytest.approx(0.01)
    assert a == pytest.approx(1.0)
    assert tc.ols([1.0], [2.0]) == (None, None)
    assert tc.ols([3.0, 3.0], [1.0, 2.0]) == (None, None)


def test_fill_scales_and_summary(tmp_path):
    _results(tmp_path, [0, 9, 18], extra={9: "rozmyte"})
    # separator ';' z przecinkiem dziesiętnym
    (tmp_path / "ref.csv").write_text("segment;ref_m\nW1;1,5\nW2;2,0\n", encoding="utf-8")
    # W1: scale = 1 + 0.01 * (19 - start); W2 brak dla clip_9
    w1 = {s: 1.5 / (1 + 0.01 * (19 - s)) for s in (0, 9, 18)}
    (tmp_path / "meas.csv").write_text(
        "clip,W1_measured_m,W2_measured_m\n"
        f"clip_0,{w1[0]!r},1.0\nclip_9,{w1[9]!r},\nclip_18,{w1[18]!r},2.5\n",
        encoding="utf-8")

    tc.fill(tmp_path, tmp_path / "meas.csv", tmp_path / "ref.csv")

    rows = _rows(tmp_path / "results.csv")
    assert rows["clip_0"]["W1_scale"] == "1.190000"
    assert rows["clip_18"]["W1_scale"] == "1.010000"
    assert rows["clip_0"]["W2_scale"] == "2.000000"
    assert rows["clip_9"]["W2_measured_m"] == rows["clip_9"]["W2_scale"] == ""
    assert rows["clip_9"]["note"] == "rozmyte"

    summary = (tmp_path / "results_summary.txt").read_text(encoding="utf-8")
    w1_block, w2_block = summary.split("\nW1 ")[1].split("\nW2 ")
    assert "slope b   0.010000" in w1_block
    assert "intercept 1.000000" in w1_block
    assert "n = 2" in w2_block


def test_fill_rejects_unknown_clip(tmp_path):
    _results(tmp_path, [18])
    (tmp_path / "ref.csv").write_text("segment,ref_m\nW1,1.5\nW2,2.0\n", encoding="utf-8")
    (tmp_path / "meas.csv").write_text("clip,W1_measured_m,W2_measured_m\nclip_99,1,1\n",
                                       encoding="utf-8")
    with pytest.raises(SystemExit):
        tc.fill(tmp_path, tmp_path / "meas.csv", tmp_path / "ref.csv")


def test_write_results_keeps_measurements(tmp_path):
    clip = tmp_path / "clip_18"
    clip.mkdir()
    (clip / "config.json").write_text(json.dumps(
        {"clip": "clip_18", "start_s": 18.0, "end_s": 30.0, "length_s": 12.0,
         "reencoded": True, "n_frames": 720}), encoding="utf-8")
    tc.write_csv(tmp_path / "results.csv", tc.RESULT_COLUMNS,
                 [{"clip": "clip_18", "W1_measured_m": "1.457", "W1_scale": "1.148936"}])

    tc.write_results(tmp_path)

    row = _rows(tmp_path / "results.csv")["clip_18"]
    assert (row["n_frames"], row["cloud_path"]) == ("720", "")
    assert (row["W1_measured_m"], row["W1_scale"]) == ("1.457", "1.148936")
