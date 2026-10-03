"""Zestawienie skali z kilku przebiegów: liczby w scale_summary.csv.

Tylko funkcja czysta `summary_rows()`; rysowanie (matplotlib) nie jest testowane.
Oczekiwane wartości policzone ręcznie w komentarzach.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "scale_report", Path(__file__).resolve().parents[1] / "eval" / "scale_report.py")
sr = importlib.util.module_from_spec(_SPEC)
sys.modules["scale_report"] = sr
_SPEC.loader.exec_module(sr)


def _run(segments, median, spread):
    return {"segments": segments, "summary": {"median": median, "spread_pct": spread}}


def _seg(name, measured, sigma_measured, factor, sigma_factor=0.01, ref_mm=1000.0, n=4):
    return {"name": name, "ref_mm": ref_mm, "sigma_ref_mm": 2.0, "measured": measured,
            "sigma_measured": sigma_measured, "factor": factor, "sigma_factor": sigma_factor, "n": n}


def test_summary_rows_repeatability_and_deviation():
    """Powtarzalność = sigma/measured: 0.02/0.80 = 2.50 %. Odchylenie: 1.80/1.50 - 1 = +20.00 %."""
    runs = {"A": _run([_seg("8-9", 0.80, 0.02, 1.80), _seg("3-8", 0.50, 0.001, 1.50)], 1.50, 20.0)}
    rows = {r["segment"]: r for r in sr.summary_rows(runs)}
    assert rows["8-9"]["powtarzalnosc_pct"] == "2.50"
    assert rows["8-9"]["odchylenie_od_mediany_pct"] == "+20.00"
    assert rows["3-8"]["odchylenie_od_mediany_pct"] == "+0.00"
    assert rows["3-8"]["rozrzut_nagrania_pct"] == "20.00"


def test_summary_rows_single_click_has_no_repeatability():
    """Jeden pomiar bez sigmy: estimate daje None — tu ma zostać puste pole, nie zero."""
    runs = {"A": _run([_seg("8-9", 0.80, None, 1.80, sigma_factor=None, n=1)], 1.80, 0.0)}
    row = sr.summary_rows(runs)[0]
    assert row["sigma_klikniec"] == "" and row["powtarzalnosc_pct"] == "" and row["sigma_factor"] == ""


def test_summary_rows_keeps_recordings_apart():
    runs = {"A": _run([_seg("8-9", 1.0, 0.01, 1.6)], 1.6, 0.0),
            "B": _run([_seg("8-9", 1.0, 0.01, 1.3)], 1.3, 0.0)}
    rows = sr.summary_rows(runs)
    assert [(r["nagranie"], r["mediana_nagrania"]) for r in rows] == [("A", "1.600000"), ("B", "1.300000")]


def test_load_aliases_reads_name_column_and_skips_empty(tmp_path):
    path = tmp_path / "nagrania.csv"
    path.write_text(
        "plik,nazwa,uwagi\n"
        "VID_20260921_125357.mp4,daleko_wolno,\n"
        "VID_20260921_133034.mp4,,inne rozstawienie\n", encoding="utf-8")
    assert sr.load_aliases(path) == {"VID_20260921_125357": "daleko_wolno"}
    assert sr.load_aliases(tmp_path / "brak.csv") == {}


def test_display_name_keeps_number_next_to_alias():
    """Numer zostaje zawsze — po nim nagranie znajduje się w runs/ i w config.json."""
    aliases = {"VID_20260921_125357": "daleko_wolno"}
    assert sr.display_name("VID_20260921_125357", aliases) == "daleko_wolno (125357)"
    assert sr.display_name("VID_20260921_125437", aliases) == "125437"


def test_summary_rows_carries_alias():
    runs = {"VID_X_1": _run([_seg("8-9", 1.0, 0.01, 1.6)], 1.6, 0.0),
            "VID_X_2": _run([_seg("8-9", 1.0, 0.01, 1.3)], 1.3, 0.0)}
    rows = sr.summary_rows(runs, {"VID_X_1": "daleko_wolno"})
    assert [r["nazwa"] for r in rows] == ["daleko_wolno", ""]
