"""Estymacja skali na syntetyce: odtworzenie znanego czynnika, rozrzut, propagacja, walidacja.

Same funkcje czyste (CLAUDE.md, zasada 2) — MASt3R-SLAM nie jest tu w ogóle dotykany,
a `scale_ply()` (open3d, I/O) nie jest testowany jednostkowo. Liczby oczekiwane są
policzone ręcznie w docstringach testów, nie przepisane z wyjścia kodu.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from vid2cloud.scale import estimate as est

#: Pięć point-pickingów W1 w chmurze MASt3R-SLAM z czerwca (data/own/reference_segments.md).
W1_MEASUREMENTS = [1.456, 1.471, 1.436, 1.461, 1.457]


def _segment(name, length_mm, sigma_mm=None, **extra):
    seg = {"name": name, "length_mm": length_mm, **extra}
    if sigma_mm is not None:
        seg["sigma_mm"] = sigma_mm
    return seg


def _targets(*segments, **top):
    data = {"dict": "DICT_4X4_50", "segments": list(segments)}
    data.update(top)
    return est.parse_targets(data, source="<test>")


# --------------------------------------------------------------------------- #
# Test 1: odtworzenie znanego czynnika
# --------------------------------------------------------------------------- #


def test_recovers_known_factor_from_scaled_cloud():
    """Chmura w jednostkach modelu + wzorzec o znanej długości -> ten sam czynnik.

    Odcinek o długości rzeczywistej 1234,5 mm = 1,2345 m kładziemy w „chmurze" tak, żeby
    mierzył 1,2345 / 0,734 jednostek modelu. estimate_scale ma odtworzyć 0,734, a
    apply_scale tym czynnikiem ma wrócić do 1,2345 m.
    """
    true_factor = 0.734
    ref_mm = 1234.5

    direction = np.array([2.0, -3.0, 6.0]) / 7.0  # norma dokładnie 1
    p0 = np.array([0.1, -0.2, 0.3])
    cloud = np.stack([p0, p0 + direction * (ref_mm / 1000.0 / true_factor)])
    measured = float(np.linalg.norm(cloud[1] - cloud[0]))

    out = est.estimate_scale(
        [{"segment": "syn", "measured": measured}],
        _targets(_segment("syn", ref_mm, 1.0)),
    )

    assert out.summary.n_segments == 1
    assert out.segments[0].n == 1
    assert out.segments[0].factor == pytest.approx(true_factor, abs=1e-9)
    assert out.summary.median == pytest.approx(true_factor, abs=1e-9)
    assert out.summary.spread_pct == pytest.approx(0.0, abs=1e-12)
    # jeden pomiar bez podanej sigmy: niepewności nie znamy, więc None (nie zero)
    assert out.segments[0].sigma_measured is None
    assert out.segments[0].sigma_factor is None
    assert out.summary.std is None

    scaled = est.apply_scale(cloud, out.segments[0].factor)
    assert float(np.linalg.norm(scaled[1] - scaled[0])) == pytest.approx(ref_mm / 1000.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Test 2: kilka segmentów o różnych czynnikach
# --------------------------------------------------------------------------- #


def test_different_factors_median_and_spread():
    """Trzy wzorce, każdy mierzy w chmurze 1,0 j.m., długości rzeczywiste 1000/1200/1900 mm.

    Czynniki wychodzą 1,0 / 1,2 / 1,9. Ręcznie:
      mediana  = 1,2
      średnia  = (1,0 + 1,2 + 1,9) / 3 = 1,36666667
      std (ddof=1): odchylenia -0,36666667, -0,16666667, 0,53333333;
                    suma kwadratów 0,44666667; /2 = 0,22333333; sqrt = 0,47258156
      rozrzut  = (1,9 - 1,0) / 1,2 * 100 = 75,0 %
    """
    targets = _targets(
        _segment("A", 1000.0, 1.0),
        _segment("B", 1200.0, 1.0),
        _segment("C", 1900.0, 1.0),
    )
    observations = [{"segment": n, "measured": 1.0} for n in ("A", "B", "C")]
    out = est.estimate_scale(observations, targets)

    assert [s.factor for s in out.segments] == pytest.approx([1.0, 1.2, 1.9])
    assert out.summary.median == pytest.approx(1.2)
    assert out.summary.mean == pytest.approx(1.36666667, rel=1e-8)
    assert out.summary.min == pytest.approx(1.0)
    assert out.summary.max == pytest.approx(1.9)
    assert out.summary.std == pytest.approx(0.47258156, rel=1e-8)
    assert out.summary.spread_pct == pytest.approx(75.0, rel=1e-12)
    assert out.summary.n_segments == 3
    # kolejność wyniku = kolejność obserwacji; outliera nikt nie usunął
    assert [s.name for s in out.segments] == ["A", "B", "C"]

    # wybór jednej liczby jest jawny i opisany
    factor, how = est.select_factor(out, use="median")
    assert factor == pytest.approx(1.2)
    assert "mediana" in how and "3" in how
    factor, how = est.select_factor(out, segment="C")
    assert factor == pytest.approx(1.9)
    assert "C" in how


def test_table_warns_above_spread_threshold():
    """Rozrzut powyżej progu dopisuje ostrzeżenie, ale nie zmienia liczb."""
    out = est.estimate_scale(
        [{"segment": n, "measured": 1.0} for n in ("A", "B")],
        _targets(_segment("A", 1000.0, 1.0), _segment("B", 1200.0, 1.0)),
    )
    text = est.format_table(out)
    assert "UWAGA" in text
    # czynniki 1,0 i 1,2 -> mediana 1,1; rozrzut (1,2 - 1,0) / 1,1 * 100 = 18,18 %
    assert out.summary.spread_pct == pytest.approx(18.18181818, rel=1e-8)

    quiet = est.estimate_scale(
        [{"segment": n, "measured": 1.0} for n in ("A", "B")],
        _targets(_segment("A", 1000.0, 1.0), _segment("B", 1020.0, 1.0)),
    )
    assert "UWAGA" not in est.format_table(quiet)  # rozrzut 1,98 % < 5 %


# --------------------------------------------------------------------------- #
# Test 3: propagacja niepewności
# --------------------------------------------------------------------------- #


def test_uncertainty_propagation_hand_example():
    """W1 z pomiarów 11.09: ref 1674,0 mm ± 12,0; w chmurze 1,457 j.m. ± 0,0128.

    Rachunek ręczny:
      factor            = 1,674 / 1,457          = 1,14893617
      sigma_ref / ref   = 12 / 1674              = 0,00716846
      sigma_meas / meas = 0,0128 / 1,457         = 0,00878517
      pierwiastek sumy kwadratów                 = 0,01133869
      sigma_factor      = 1,14893617 * 0,01133869 = 0,01302744
    """
    out = est.estimate_scale(
        [{"segment": "W1", "measured": 1.457, "sigma_measured": 0.0128}],
        _targets(_segment("W1", 1674.0, 12.0)),
    )
    row = out.segments[0]
    assert row.ref_mm == 1674.0
    assert row.sigma_ref_mm == 12.0
    assert row.factor == pytest.approx(1.14893617, rel=1e-8)
    assert row.sigma_factor == pytest.approx(0.01302744, rel=1e-6)


def test_default_sigma_mm_when_absent():
    """Brak sigma_mm w targets.json -> DEFAULT_SIGMA_MM, nie zero i nie cisza."""
    targets = _targets(_segment("X", 1000.0))
    assert targets.segment("X").sigma_mm == est.DEFAULT_SIGMA_MM


# --------------------------------------------------------------------------- #
# Test 4: powtórzenia pomiaru
# --------------------------------------------------------------------------- #


def test_repeated_measurements_median_and_sample_sigma():
    """Pięć kliknięć W1: wartością jest mediana, sigma to odchylenie próbki (ddof=1).

    1,456  1,471  1,436  1,461  1,457 -> posortowane 1,436  1,456  1,457  1,461  1,471,
    czyli mediana 1,457 (nie średnia 1,4562).
    Odchylenia od średniej: -0,0002  0,0148  -0,0202  0,0048  0,0008;
    suma kwadratów 6,50800e-4; / (5-1) = 1,62700e-4; sqrt = 0,01275539.
    factor = 1,674 / 1,457 = 1,14893617 — tyle, ile w reference_segments.md (x1,149).
    """
    out = est.estimate_scale(
        [{"segment": "W1", "measurements": W1_MEASUREMENTS}],
        _targets(_segment("W1", 1674.0, 12.0)),
    )
    row = out.segments[0]
    assert row.n == 5
    assert row.measured == pytest.approx(1.457, abs=1e-12)
    assert row.sigma_measured == pytest.approx(0.01275539, rel=1e-6)
    assert row.factor == pytest.approx(1.14893617, rel=1e-8)
    # sigma z próbki weszła do propagacji
    assert row.sigma_factor == pytest.approx(1.14893617 * np.hypot(12.0 / 1674.0, 0.01275539 / 1.457), rel=1e-6)


def test_explicit_sigma_wins_over_sample():
    """Jawne sigma_measured ma pierwszeństwo przed sigmą policzoną z próbki."""
    out = est.estimate_scale(
        [{"segment": "W1", "measurements": W1_MEASUREMENTS, "sigma_measured": 0.05}],
        _targets(_segment("W1", 1674.0, 12.0)),
    )
    assert out.segments[0].sigma_measured == pytest.approx(0.05)
    assert out.segments[0].n == 5


# --------------------------------------------------------------------------- #
# Test 5: walidacja targets.json i obserwacji
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "segments, fragment",
    [
        ([_segment("W1", 1000.0), _segment("W1", 1200.0)], "unikalne"),
        ([_segment("W1", 0.0)], "length_mm"),
        ([_segment("W1", -5.0)], "length_mm"),
        ([_segment("W1", 1000.0, 0.0)], "sigma_mm"),
        ([_segment("W1", 1000.0, -1.0)], "sigma_mm"),
        ([_segment("W1", 1000.0, 1.0, marker_a=20, marker_b=20)], "marker"),
        ([{"length_mm": 1000.0}], "name"),
        ([_segment("W1", "dużo")], "liczby"),
        ([], "niepustą"),
    ],
)
def test_bad_targets_raise_value_error(segments, fragment):
    """Błędny targets.json to ValueError z nazwą segmentu — nie ciche pominięcie."""
    with pytest.raises(ValueError) as exc:
        _targets(*segments)
    assert fragment in str(exc.value)
    if segments and segments[0].get("name"):
        assert "W1" in str(exc.value)


def test_unknown_segment_names_itself():
    targets = _targets(_segment("W1", 1000.0))
    with pytest.raises(ValueError) as exc:
        est.estimate_scale([{"segment": "W9", "measured": 1.0}], targets)
    assert "W9" in str(exc.value) and "W1" in str(exc.value)


@pytest.mark.parametrize(
    "observations, fragment",
    [
        ([{"segment": "W1"}], "dokładnie jedno"),
        ([{"segment": "W1", "measured": 1.0, "measurements": [1.0]}], "dokładnie jedno"),
        ([{"segment": "W1", "measured": 0.0}], "> 0"),
        ([{"segment": "W1", "measured": -1.0}], "> 0"),
        ([{"segment": "W1", "measurements": []}], "puste"),
        ([{"measured": 1.0}], "segment"),
        ([{"segment": "W1", "measured": 1.0}, {"segment": "W1", "measured": 1.1}], "więcej niż raz"),
        ([], "brak obserwacji"),
    ],
)
def test_bad_observations_raise_value_error(observations, fragment):
    with pytest.raises(ValueError) as exc:
        est.estimate_scale(observations, _targets(_segment("W1", 1000.0)))
    assert fragment in str(exc.value)


def test_apply_scale_validates_shape_and_factor():
    with pytest.raises(ValueError):
        est.apply_scale(np.zeros((4, 2)), 1.0)
    with pytest.raises(ValueError):
        est.apply_scale(np.zeros((4, 3)), 0.0)
    with pytest.raises(ValueError):
        est.apply_scale(np.zeros((4, 3)), -1.0)


# --------------------------------------------------------------------------- #
# Wczytywanie plików
# --------------------------------------------------------------------------- #


def test_load_targets_roundtrip(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "dict": "DICT_4X4_50",
                "markers": {"3": {"side_mm": 145.9, "board": "test_A4_150mm"}},
                "segments": [_segment("W1", 1674.0, 12.0, kind="control", reference="narożniki")],
            }
        ),
        encoding="utf-8",
    )
    targets = est.load_targets(path)
    assert targets.dict_name == "DICT_4X4_50"
    assert targets.markers["3"]["side_mm"] == 145.9
    assert targets.segment("W1").length_mm == 1674.0
    assert targets.segment("W1").reference == "narożniki"


def test_load_targets_rejects_broken_json(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text("{ to nie jest JSON", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        est.load_targets(path)
    assert "JSON" in str(exc.value)


def test_read_measured_csv_groups_repetitions(tmp_path):
    """Wiele wierszy tego samego segmentu = powtórzenia pomiaru, w kolejności pojawienia się."""
    path = tmp_path / "measured.csv"
    path.write_text(
        "segment,measured\n"
        "W1,1.456\n"
        "W2,1.110\n"
        "W1,1.471\n"
        "# komentarz terenowy,\n"
        "\n"
        "W1,1.436\n",
        encoding="utf-8",
    )
    observations = est.read_measured_csv(path)
    assert [o["segment"] for o in observations] == ["W1", "W2"]
    assert observations[0]["measurements"] == [1.456, 1.471, 1.436]
    assert observations[1]["measurements"] == [1.110]


def test_read_measured_csv_rejects_sigma_with_repetitions(tmp_path):
    path = tmp_path / "measured.csv"
    path.write_text("segment,measured,sigma_measured\nW1,1.456,0.01\nW1,1.471,\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        est.read_measured_csv(path)
    assert "sigma" in str(exc.value)


def test_read_measured_csv_requires_columns(tmp_path):
    path = tmp_path / "measured.csv"
    path.write_text("odcinek,dlugosc\nW1,1.456\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        est.read_measured_csv(path)
    assert "segment" in str(exc.value)


# --------------------------------------------------------------------------- #
# Zapis obok przeskalowanej chmury
# --------------------------------------------------------------------------- #


def test_scale_record_has_everything_needed_to_redo_it():
    """`<out>.scale.json` musi wystarczyć do odtworzenia skalowania.

    Samo `scale_ply()` (open3d, I/O) zostaje do ręcznego sprawdzenia na chmurze —
    tu testujemy tylko treść zapisu, bo to ona decyduje o odtwarzalności.
    """
    out = est.estimate_scale(
        [
            {"segment": "W1", "measurements": W1_MEASUREMENTS},
            {"segment": "W2", "measured": 1.107},
        ],
        _targets(_segment("W1", 1674.0, 12.0), _segment("W2", 1107.0, 20.0)),
    )
    factor, how = est.select_factor(out, use="median")
    record = est.build_scale_record(
        factor,
        "runs/x/cloud.ply",
        "runs/x/cloud_m.ply",
        123,
        {"how": how, "targets": "targets.json", "measured": "measured.csv", **out.as_dict()},
        now=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc),
    )

    assert record["factor"] == pytest.approx(out.summary.median)
    assert "mediana" in record["how"]
    assert record["n_points"] == 123
    assert record["source_ply"].endswith("cloud.ply")
    assert record["written"].startswith("2026-09-15T12:00:00")
    assert [s["name"] for s in record["segments"]] == ["W1", "W2"]
    assert record["summary"]["spread_pct"] == pytest.approx(out.summary.spread_pct)
    json.dumps(record)  # musi dać się zapisać do JSON-a
