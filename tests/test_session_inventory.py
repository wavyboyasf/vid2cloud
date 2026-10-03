"""Inwentaryzacja sesji: konwerter pomiarów taśmą na targets.json i wybór kadrów.

Same funkcje czyste (CLAUDE.md, zasada 2): detekcja na zdjęciach i dekodowanie wideo
nie są tu testowane. Najważniejszy test to ten, który sprawdza, że wynik konwertera
wczytuje `estimate.load_targets()` — czyli że dwie strony schematu się nie rozjechały.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vid2cloud.scale import estimate as est

_SPEC = importlib.util.spec_from_file_location(
    "session_inventory", Path(__file__).resolve().parents[1] / "eval" / "session_inventory.py"
)
si = importlib.util.module_from_spec(_SPEC)
# Rejestracja w sys.modules przed wykonaniem: bez tego @dataclass w module nie widzi
# własnego modułu i wywala się na AttributeError.
sys.modules["session_inventory"] = si
_SPEC.loader.exec_module(si)


# --------------------------------------------------------------------------- #
# Pomocnicze
# --------------------------------------------------------------------------- #


def _row(id_a, id_b, odleglosc="", jednostka="mm", punkty_pomiaru="", sigma_mm="2", uwagi=""):
    return {
        "id_a": str(id_a),
        "id_b": str(id_b),
        "wspolwidocznosc_klatek": "0",
        "wspolwidocznosc_zdjec": "0",
        "odleglosc": str(odleglosc),
        "jednostka": jednostka,
        "sigma_mm": str(sigma_mm),
        "punkty_pomiaru": punkty_pomiaru,
        "uwagi": uwagi,
    }


def _det(marker_id, source="VID_A", kind="video", frame=0, side_px_512=20.0):
    return si.Detection(
        source=source, kind=kind, frame=frame, marker_id=marker_id,
        side_px=side_px_512 * 7.5, side_px_512=side_px_512,
    )


BOARDS = {
    8: {"nominal_size_mm": 150.0, "nominal_size_mm_all": [150.0], "board": "boards_A4_150mm",
        "dict": "DICT_4X4_50"},
    48: {"nominal_size_mm": 70.0, "nominal_size_mm_all": [70.0], "board": "scalebar_A4_70mm",
         "dict": "DICT_4X4_50"},
    49: {"nominal_size_mm": 70.0, "nominal_size_mm_all": [70.0], "board": "scalebar_A4_70mm",
         "dict": "DICT_4X4_50"},
}


# --------------------------------------------------------------------------- #
# Konwerter pomiary_tasma.csv -> targets.json
# --------------------------------------------------------------------------- #


def test_to_targets_is_readable_by_load_targets(tmp_path):
    """Definicja „działa" tej sesji: dwa uzupełnione wiersze -> targets.json bez błędu.

    Trzeci wiersz jest pusty (par mierzonych było mniej niż wykrytych) i ma zostać
    pominięty, a nie wysadzić konwerter.
    """
    rows = [
        _row(48, 49, 500.0, punkty_pomiaru="srodek-srodek", sigma_mm=2),
        _row(8, 10, 1234.5, punkty_pomiaru="narozniki zewnetrzne lewy-dolny/prawy-dolny", sigma_mm=3),
        _row(8, 48),
    ]
    data = si.pomiary_to_targets(rows, BOARDS, date="2026-09-21")
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    targets = est.load_targets(path)
    assert sorted(targets.segments) == ["48-49", "8-10"]

    seg = targets.segment("48-49")
    assert seg.length_mm == pytest.approx(500.0)
    assert seg.sigma_mm == pytest.approx(2.0)
    assert (seg.marker_a, seg.marker_b) == (48, 49)
    assert seg.reference == "srodek-srodek"
    assert seg.measured_by == "tasma" and seg.date == "2026-09-21"
    assert targets.segment("8-10").sigma_mm == pytest.approx(3.0)


def test_to_targets_names_pair_in_rising_order():
    """Nazwa i pola segmentu nie zależą od kolejności ID w wierszu."""
    data = si.pomiary_to_targets([_row(49, 48, 500.0)], BOARDS)
    seg = data["segments"][0]
    assert seg["name"] == "48-49"
    assert (seg["marker_a"], seg["marker_b"]) == (48, 49)


def test_to_targets_accepts_decimal_comma():
    """Arkusz kalkulacyjny zapisuje 500,5 — to ma przejść, a nie wywalić się na float()."""
    data = si.pomiary_to_targets([_row(48, 49, "500,5", sigma_mm="1,5")], BOARDS)
    assert data["segments"][0]["length_mm"] == pytest.approx(500.5)
    assert data["segments"][0]["sigma_mm"] == pytest.approx(1.5)


def test_to_targets_default_sigma_when_column_empty():
    data = si.pomiary_to_targets([_row(48, 49, 500.0, sigma_mm="")], BOARDS)
    assert data["segments"][0]["sigma_mm"] == pytest.approx(si.DEFAULT_SIGMA_MM)


def test_to_targets_rejects_file_without_any_distance():
    """Pusta tabela to błąd: lepiej krzyknąć niż zapisać targets.json bez segmentów."""
    with pytest.raises(ValueError, match="odleglosc"):
        si.pomiary_to_targets([_row(48, 49), _row(8, 48)], BOARDS)


def test_to_targets_rejects_non_numeric_distance():
    with pytest.raises(ValueError, match="nie jest liczbą"):
        si.pomiary_to_targets([_row(48, 49, "ok. 50 cm")], BOARDS)


def test_to_targets_never_writes_nominal_as_side_mm():
    """Nominał wydruku nie jest wzorcem skali (docs/decisions.md, 2026-09-15).

    Dlatego w `markers` wolno mu siedzieć tylko jako informacja — pole `side_mm`,
    które `estimate.parse_targets()` traktuje jako wymiar markera, musi zostać puste.
    """
    data = si.pomiary_to_targets([_row(48, 49, 500.0)], BOARDS)
    assert set(data["markers"]) == {"48", "49"}
    for entry in data["markers"].values():
        assert "side_mm" not in entry
        assert entry["nominal_size_mm"] == pytest.approx(70.0)


# --------------------------------------------------------------------------- #
# Jednostki
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value,unit,expected", [
    ("109,1", "cm", 1091.0),
    ("1091", "mm", 1091.0),
    ("1.0911", "m", 1091.1),
    ("109.1", "CM", 1091.0),
])
def test_to_mm_converts_what_the_tape_showed(value, unit, expected):
    """Wpisuje się odczyt z taśmy i jednostkę; przeliczenie jest po stronie kodu."""
    assert si.to_mm(value, unit) == pytest.approx(expected)


def test_to_mm_rejects_unknown_unit():
    with pytest.raises(ValueError, match="nieznana jednostka"):
        si.to_mm("109,1", "cale")


def test_to_targets_uses_unit_column():
    """109,1 cm między 11 a 12 ma wejść do targets.json jako 1091 mm."""
    data = si.pomiary_to_targets([_row(11, 12, "109,1", "cm", punkty_pomiaru="srodek-srodek")], BOARDS)
    assert data["segments"][0]["length_mm"] == pytest.approx(1091.0)


def test_to_targets_sigma_stays_in_mm_regardless_of_unit():
    """`sigma_mm` jest w milimetrach zawsze — kolumna `jednostka` jej nie dotyczy."""
    data = si.pomiary_to_targets([_row(11, 12, "109,1", "cm", sigma_mm="2")], BOARDS)
    assert data["segments"][0]["sigma_mm"] == pytest.approx(2.0)


def test_to_targets_reads_legacy_millimetre_column():
    """Stary nagłówek `odleglosc_mm` (bez kolumny jednostki) ma dalej działać."""
    row = {"id_a": "48", "id_b": "49", "odleglosc_mm": "500", "sigma_mm": "2", "punkty_pomiaru": ""}
    data = si.pomiary_to_targets([row], BOARDS)
    assert data["segments"][0]["length_mm"] == pytest.approx(500.0)


def test_suspicious_lengths_catches_unit_mistake():
    """1091 wpisane jako metry to 1,09 km — ma zostać zgłoszone, nie poprawione."""
    data = si.pomiary_to_targets([_row(11, 12, "1091", "m"), _row(8, 10, "109,1", "cm")], BOARDS)
    odstajace = si.suspicious_lengths(data["segments"])
    assert [s["name"] for s in odstajace] == ["11-12"]
    assert data["segments"][0]["length_mm"] == pytest.approx(1_091_000.0)


# --------------------------------------------------------------------------- #
# Deklaracja rozstawionych ID
# --------------------------------------------------------------------------- #


def test_parse_declared_ids_accepts_commas_spaces_and_comments():
    text = "# ID rozstawione 21.09\n3, 8, 9\n10 11 12  # reszta zostala w teczce\n"
    assert si.parse_declared_ids(text) == {3, 8, 9, 10, 11, 12}


def test_parse_declared_ids_rejects_garbage_and_emptiness():
    with pytest.raises(ValueError, match="nie jest liczbą"):
        si.parse_declared_ids("3, 8, dziewiec")
    with pytest.raises(ValueError, match="pusta"):
        si.parse_declared_ids("# tylko komentarz\n")


def test_declared_ids_round_trip(tmp_path):
    path = tmp_path / "rozstawione_id.txt"
    assert si.read_declared_ids(path) is None  # brak pliku to brak deklaracji, nie błąd
    si.write_declared_ids({12, 3, 8}, path, session="2026-09-21")
    assert si.read_declared_ids(path) == {3, 8, 12}


def test_to_targets_rejects_id_outside_declaration():
    """Sedno sprawy: wzorzec skali nie może stać na markerze, którego w scenie nie było."""
    with pytest.raises(ValueError, match=r"ID \[7\] nie są na liście rozstawionych"):
        si.pomiary_to_targets([_row(7, 10, "109,1", "cm")], BOARDS, declared={3, 8, 9, 10, 11, 12})


def test_summary_marks_printed_but_not_deployed_id():
    """ID 7 jest w boards.json, a nie był rozstawiony — kolumny mają to rozróżniać."""
    inv = si.Inventory(detections=[_det(8, frame=0), _det(7, frame=1)], videos=["VID_A"])
    boards = dict(BOARDS)
    boards[7] = {"nominal_size_mm": 120.0, "nominal_size_mm_all": [120.0],
                 "board": "boards_A3_120mm", "dict": "DICT_4X4_50"}
    rows = {row["id"]: row for row in si.summary_rows(inv, boards, declared={8})}
    assert (rows[7]["w_boards"], rows[7]["zadeklarowany"]) == ("tak", "nie")
    assert (rows[8]["w_boards"], rows[8]["zadeklarowany"]) == ("tak", "tak")


def test_summary_without_declaration_leaves_question_mark():
    inv = si.Inventory(detections=[_det(8, frame=0)], videos=["VID_A"])
    assert si.summary_rows(inv, BOARDS)[0]["zadeklarowany"] == "?"


# --------------------------------------------------------------------------- #
# Tabele: ręczne wpisy przeżywają ponowne uruchomienie
# --------------------------------------------------------------------------- #


def test_write_pomiary_keeps_manual_columns_and_measured_pairs(tmp_path):
    """Drugie uruchomienie nie ma prawa skasować wpisanych odległości.

    Para 11-12 jest uzupełniona; po zawężeniu deklaracji do ID bez 11 i 12 wiersz z pomiarem
    i tak zostaje, bo autor to zmierzył — znika tylko z listy par do zmierzenia.
    """
    path = tmp_path / "pomiary_tasma.csv"
    dets = [_det(11, frame=0), _det(12, frame=0), _det(8, frame=0)]
    inv = si.Inventory(detections=dets, videos=["VID_A"])

    si.write_pomiary(inv, {8, 11, 12}, path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert ("11", "12") in {(r["id_a"], r["id_b"]) for r in rows}
    assert all(r["jednostka"] == si.DEFAULT_UNIT for r in rows)

    text = path.read_text(encoding="utf-8").replace(
        "11,12,1,0,,cm,2,,", "11,12,1,0,109.1,cm,2,srodek-srodek,")
    path.write_text(text, encoding="utf-8")

    si.write_pomiary(inv, {8}, path)
    rows = {(r["id_a"], r["id_b"]): r for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert rows[("11", "12")]["odleglosc"] == "109.1"
    assert rows[("11", "12")]["punkty_pomiaru"] == "srodek-srodek"


def test_carry_over_does_not_resurrect_empty_values():
    old = {("a.mp4",): {"plik": "a.mp4", "typ": "", "uwagi": "pierwsza petla"}}
    row = si.carry_over({"plik": "a.mp4", "typ": "petla", "uwagi": ""}, old, ("plik",),
                        ("typ", "uwagi"))
    assert row["typ"] == "petla"          # stare puste nie nadpisuje nowego
    assert row["uwagi"] == "pierwsza petla"


# --------------------------------------------------------------------------- #
# Wybór kadrów
# --------------------------------------------------------------------------- #


def test_best_view_per_id_takes_largest_marker():
    inv = si.Inventory(
        detections=[
            _det(48, frame=0, side_px_512=9.0),
            _det(48, frame=10, side_px_512=19.5),
            _det(48, frame=20, side_px_512=12.0),
            _det(49, source="IMG_1.jpg", kind="photo", frame=None, side_px_512=30.0),
        ],
        videos=["VID_A"],
        photos=["IMG_1.jpg"],
    )
    best = si.best_view_per_id(inv)
    assert best[48][0].frame == 10
    assert best[48][1].side_px_512 == pytest.approx(19.5)
    assert best[49][0].kind == "photo" and best[49][0].frame is None


def test_overview_prefers_frames_with_most_ids_and_keeps_distance():
    """Pięć kadrów ma być różnymi scenami, nie pięcioma kopiami tej samej.

    Klatki 100 i 105 widzą te same trzy markery — druga nie wnosi nic i jest bliżej
    niż `OVERVIEW_MIN_GAP`, więc do overview wchodzi tylko jedna z nich.
    """
    dets = []
    for frame in (100, 105):
        dets += [_det(i, frame=frame) for i in (8, 10, 11)]
    dets += [_det(i, frame=500) for i in (48, 49)]
    inv = si.Inventory(detections=dets, videos=["VID_A"])

    chosen = si.overview_views(inv, count=5)
    frames = [v.frame for v, _ in chosen]
    assert frames[0] == 100
    assert 105 not in frames
    assert 500 in frames


def test_overview_skips_subset_of_already_chosen_ids():
    """Kadr z podzbiorem ID już pokazanych nie mówi nic nowego o rozstawieniu."""
    dets = [_det(i, frame=0) for i in (8, 10, 11)] + [_det(i, frame=400) for i in (8, 10)]
    inv = si.Inventory(detections=dets, videos=["VID_A"])
    chosen = si.overview_views(inv, count=5)
    assert [v.frame for v, _ in chosen] == [0]


# --------------------------------------------------------------------------- #
# Współwidoczność par
# --------------------------------------------------------------------------- #


def test_covisible_pairs_counts_frames_and_photos_separately():
    dets = [
        _det(48, frame=0), _det(49, frame=0),            # para 48-49 w klatce
        _det(48, frame=1), _det(49, frame=1),            # i w drugiej
        _det(48, source="IMG_1.jpg", kind="photo", frame=None),
        _det(49, source="IMG_1.jpg", kind="photo", frame=None),
        _det(8, frame=900),                              # sam, bez pary
    ]
    pairs = si.covisible_pairs(si.Inventory(detections=dets, videos=["VID_A"]))
    assert pairs[(48, 49)]["klatek"] == 2
    assert pairs[(48, 49)]["zdjec"] == 1
    assert (8, 48) not in pairs


def test_covisible_pairs_respects_only_ids():
    """Fałszywe wykrycie (ID nie z wydruku) nie ma prawa wejść do tabeli pomiarów."""
    dets = [_det(48, frame=0), _det(49, frame=0), _det(17, frame=0)]
    pairs = si.covisible_pairs(si.Inventory(detections=dets, videos=["VID_A"]),
                               only_ids={48, 49})
    assert list(pairs) == [(48, 49)]


# --------------------------------------------------------------------------- #
# Wydrukowane ID
# --------------------------------------------------------------------------- #


def test_load_boards_merges_sets_sharing_an_id(tmp_path):
    """ID 3 jest i w zestawie testowym, i w planszach — oba zestawy mają zostać widoczne."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "boards.json").write_text(json.dumps({
        "dict": "DICT_4X4_50", "nominal_size_mm": 150.0,
        "markers": [{"id": 3, "nominal_size_mm": 150.0}, {"id": 8, "nominal_size_mm": 150.0}],
    }), encoding="utf-8")
    (tmp_path / "b" / "boards.json").write_text(json.dumps({
        "dict": "DICT_4X4_50", "nominal_size_mm": 120.0,
        "markers": [{"id": 3, "nominal_size_mm": 120.0}],
    }), encoding="utf-8")

    boards = si.load_boards(tmp_path)
    assert sorted(boards) == [3, 8]
    assert boards[3]["board"] == "a+b"
    assert boards[3]["nominal_size_mm_all"] == [150.0, 120.0]


def test_rare_in_photos_flags_printed_ids_with_few_photos():
    dets = [_det(48, source=f"IMG_{i}.jpg", kind="photo", frame=None) for i in range(6)]
    dets += [_det(49, source="IMG_0.jpg", kind="photo", frame=None)]
    dets += [_det(8, frame=0)]  # tylko na nagraniu, zero zdjęć
    inv = si.Inventory(detections=dets, videos=["VID_A"])
    assert si.rare_in_photos(inv, BOARDS) == [(8, 0), (49, 1)]
    # Z deklaracją flaga pilnuje tylko tego, co faktycznie rozstawiono.
    assert si.rare_in_photos(inv, {48, 49}) == [(49, 1)]
