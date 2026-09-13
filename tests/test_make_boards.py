"""Funkcje czyste generatora plansz: parsowanie ID, rozmiar rastra, zawartość boards.json."""

import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "tools" / "make_boards.py"
_spec = importlib.util.spec_from_file_location("make_boards", _path)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)


def test_parse_ids():
    assert mb.parse_ids("0-7") == [0, 1, 2, 3, 4, 5, 6, 7]
    assert mb.parse_ids("0,3,5") == [0, 3, 5]
    assert mb.parse_ids("0-3,8-9") == [0, 1, 2, 3, 8, 9]
    assert mb.parse_ids("5, 5 ,4") == [4, 5]  # powtórzenia i spacje
    for bad in ("5-2", "abc", "", ",", "-1"):
        with pytest.raises(ValueError):
            mb.parse_ids(bad)


@pytest.mark.parametrize("size_mm", [20.0, 70.0, 120.0, 297.0])
def test_marker_pixels_multiple_of_six_and_over_600dpi(size_mm):
    """Bok w pikselach musi być wielokrotnością 6 (4 komórki + 2 bordera) i dawać >= 600 dpi."""
    px = mb.marker_pixels(size_mm)
    assert px % mb.CELLS_PER_SIDE == 0
    assert px * 25.4 / size_mm >= mb.MIN_DPI
    # zaokrąglenie w górę, ale nie o więcej niż jedną komórkę
    assert (px - mb.CELLS_PER_SIDE) * 25.4 / size_mm < mb.MIN_DPI


def test_render_marker_is_bilevel_and_exact_size():
    img = mb.render_marker(mb.get_dictionary(mb.DEFAULT_DICT), 3, 60)
    assert img.mode == "1"
    assert img.size == (60, 60)
    with pytest.raises(ValueError, match="wielokrotnością"):
        mb.render_marker(mb.get_dictionary(mb.DEFAULT_DICT), 3, 61)


def test_build_writes_pdf_and_metadata(tmp_path):
    out = tmp_path / "boards.pdf"
    meta = mb.build(out, [0, 3], 60.0, "A4", mb.DEFAULT_DICT, scalebar=False)

    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:5] == b"%PDF-"
    assert meta["markers"] == [
        {"id": 0, "dict": "DICT_4X4_50", "nominal_size_mm": 60.0, "page": 1},
        {"id": 3, "dict": "DICT_4X4_50", "nominal_size_mm": 60.0, "page": 2},
    ]
    assert meta["render_dpi"] >= mb.MIN_DPI
    assert meta["page_mm"] == [210.0, 297.0]
    json.dumps(meta)  # musi się serializować bez sztuczek


def test_build_rejects_bad_input(tmp_path):
    args = (tmp_path / "x.pdf",)
    with pytest.raises(ValueError, match="ID 0..49"):
        mb.build(*args, [50], 60.0, "A4", mb.DEFAULT_DICT, scalebar=False)
    with pytest.raises(ValueError, match="nie mieści się"):
        mb.build(*args, [0], 400.0, "A4", mb.DEFAULT_DICT, scalebar=False)
    with pytest.raises(ValueError, match="nieznany format"):
        mb.build(*args, [0], 60.0, "A9", mb.DEFAULT_DICT, scalebar=False)
