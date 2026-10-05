"""Estymacja współczynnika skali z odległości między wzorcami fizycznymi.

Chmura z MASt3R-SLAM jest w jednostkach umownych. Wzorcem jest odcinek o **zmierzonej**
długości rzeczywistej: odległość między środkami dwóch markerów ArUco na łacie albo
odcinek kontrolny na obiekcie. Jego długość w chmurze pochodzi na razie z ręcznego
point-pickingu w CloudCompare (5 niezależnych kliknięć, mediana) — automatyczny pomiar
wymaga pointmapy silnika i czeka na `scale/lift.py` oraz adapter `engine/mast3r_slam.py`.

Współczynnik NIE jest jedną liczbą dla chmury
---------------------------------------------
Zmierzone na chmurze z czerwca (`data/own/reference_segments.md`, pomiary 11.09.2026):
przy kotwicy (odcinek W1) wychodzi ok. x1,15, kilka metrów dalej przy stoliku (W2) ok. x1,00
— **ta sama mapa**. Dlatego ten moduł liczy i raportuje współczynnik osobno dla każdego
segmentu i podaje rozrzut. Uśrednienie do jednej liczby jest dopuszczalne wyłącznie jawnie
(`--use median`) i zawsze ląduje w zapisie `<out>.scale.json` razem z rozkładem per segment.
Outlierów moduł nie odrzuca — raportuje je; przy rozrzucie powyżej
`SPREAD_WARN_PCT` dopisuje ostrzeżenie, ale wyniku nie zmienia.

Jednostki
---------
* `targets.json` — **milimetry** (tak się mierzy taśmą i suwmiarką);
* pomiary z chmury — jednostki modelu, bezwymiarowe;
* `factor` — **metry na jednostkę modelu** (`factor = length_mm / 1000 / measured`).

Zamiana mm -> m jest w jednym miejscu: stała `MM_PER_M` w `_segment_factor()`.

Schemat `targets.json`
----------------------
::

    {
      "dict": "DICT_4X4_50",
      "markers": {
        "3": {"side_mm": 145.9, "board": "test_A4_150mm", "note": "wydruk domowy, zmierzony"}
      },
      "segments": [
        {"name": "lata-1", "kind": "scalebar", "marker_a": 20, "marker_b": 21,
         "length_mm": 1002.5, "sigma_mm": 2.0,
         "reference": "środki markerów, taśma stalowa 2x", "measured_by": "BZ",
         "date": "2026-09-16"},
        {"name": "W1", "kind": "control", "length_mm": 1674.0, "sigma_mm": 12.0,
         "reference": "dolne wewnętrzne narożniki czerwonego panelu", "date": "2026-09-11"}
      ]
    }

`side_mm` jest w schemacie dla porządku (ślad po wydruku), ale **nie służy do liczenia
skali**: błąd narożnika małego markera jest rzędu 1 px, a sam wydruk potrafi być o 2,7 %
mniejszy od nominału — patrz `docs/decisions.md`, wpisy z 2026-09-13 i 2026-09-15.

Zależności: numpy (rdzeń). `scale_ply()` dokłada open3d, importowany dopiero przy użyciu,
bo env `vid2cloud-tools` go nie ma i cała reszta modułu działa bez niego.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

DEFAULT_DICT = "DICT_4X4_50"

#: Przelicznik jednostek `targets.json` (mm) na metry. Jedyne miejsce tej zamiany.
MM_PER_M = 1000.0

#: Niepewność długości wzorca, jeśli w `targets.json` nie podano `sigma_mm` [mm].
DEFAULT_SIGMA_MM = 1.0

#: Powyżej tego rozrzutu współczynników CLI dopisuje ostrzeżenie (samego wyniku nie zmienia).
SPREAD_WARN_PCT = 5.0


# --------------------------------------------------------------------------- #
# Struktury
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Segment:
    """Odcinek wzorcowy: długość rzeczywista z pomiaru w terenie."""

    name: str
    length_mm: float
    sigma_mm: float
    kind: str = "control"
    marker_a: Optional[int] = None
    marker_b: Optional[int] = None
    reference: str = ""
    measured_by: str = ""
    date: str = ""


@dataclass(frozen=True)
class Targets:
    """Zawartość `targets.json` po walidacji."""

    segments: dict[str, Segment]
    dict_name: str = DEFAULT_DICT
    markers: dict[str, dict] = field(default_factory=dict)
    source: str = "<dict>"

    def segment(self, name: str) -> Segment:
        """Segment po nazwie; brak segmentu to błąd, nie ciche pominięcie."""
        try:
            return self.segments[name]
        except KeyError:
            known = ", ".join(self.segments) or "(plik nie ma żadnego)"
            raise ValueError(f"{self.source}: nie znam segmentu {name!r}; są: {known}") from None


@dataclass(frozen=True)
class SegmentFactor:
    """Współczynnik policzony dla jednego segmentu."""

    name: str
    kind: str
    ref_mm: float
    sigma_ref_mm: float
    measured: float
    sigma_measured: Optional[float]
    factor: float
    sigma_factor: Optional[float]
    n: int


@dataclass(frozen=True)
class Summary:
    """Zbiorczy obraz rozrzutu współczynników. To NIE jest „współczynnik chmury"."""

    median: float
    mean: float
    min: float
    max: float
    std: Optional[float]
    spread_pct: float
    n_segments: int


@dataclass(frozen=True)
class ScaleEstimate:
    """Wynik `estimate_scale()`: lista per segment + zbiorczo."""

    segments: tuple[SegmentFactor, ...]
    summary: Summary

    def as_dict(self) -> dict:
        return {"segments": [asdict(s) for s in self.segments], "summary": asdict(self.summary)}

    def factor(self, name: str) -> SegmentFactor:
        for s in self.segments:
            if s.name == name:
                return s
        known = ", ".join(s.name for s in self.segments)
        raise ValueError(f"nie ma wyniku dla segmentu {name!r}; policzone: {known}")


# --------------------------------------------------------------------------- #
# Wczytanie targets.json
# --------------------------------------------------------------------------- #


def _as_float(value: Any, what: str) -> float:
    """Liczba albo ValueError z nazwą pola — żadnych cichych konwersji."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: oczekiwano liczby, dostałem {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{what}: liczba musi być skończona, dostałem {value!r}")
    return out


def _as_int(value: Any, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: oczekiwano liczby całkowitej, dostałem {value!r}") from None


def parse_targets(data: Any, source: str = "<dict>") -> Targets:
    """Walidacja i konwersja słownika ze schematu `targets.json`.

    Osobno od `load_targets()`, żeby dało się testować bez pliku na dysku.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{source}: na wierzchu oczekiwano obiektu JSON")

    dict_name = str(data.get("dict", DEFAULT_DICT))

    markers_raw = data.get("markers") or {}
    if not isinstance(markers_raw, dict):
        raise ValueError(f"{source}: 'markers' musi być obiektem {{id: {{...}}}}")
    markers: dict[str, dict] = {}
    for key, entry in markers_raw.items():
        what = f"{source}: marker {key!r}"
        if not isinstance(entry, dict):
            raise ValueError(f"{what}: oczekiwano obiektu")
        if entry.get("side_mm") is not None:
            side = _as_float(entry["side_mm"], f"{what}: side_mm")
            if side <= 0:
                raise ValueError(f"{what}: side_mm musi być > 0, dostałem {side}")
        markers[str(key)] = dict(entry)

    segments_raw = data.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ValueError(f"{source}: 'segments' musi być niepustą listą")

    segments: dict[str, Segment] = {}
    for i, raw in enumerate(segments_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: segment #{i}: oczekiwano obiektu")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source}: segment #{i}: brak niepustego pola 'name'")
        name = name.strip()
        what = f"{source}: segment {name!r}"
        if name in segments:
            raise ValueError(f"{what}: nazwa segmentu powtarza się — nazwy muszą być unikalne")

        if raw.get("length_mm") is None:
            raise ValueError(f"{what}: brak pola 'length_mm'")
        length_mm = _as_float(raw["length_mm"], f"{what}: length_mm")
        if length_mm <= 0:
            raise ValueError(f"{what}: length_mm musi być > 0, dostałem {length_mm}")

        if raw.get("sigma_mm") is None:
            sigma_mm = DEFAULT_SIGMA_MM
        else:
            sigma_mm = _as_float(raw["sigma_mm"], f"{what}: sigma_mm")
            if sigma_mm <= 0:
                raise ValueError(f"{what}: sigma_mm musi być > 0, dostałem {sigma_mm}")

        marker_a = None if raw.get("marker_a") is None else _as_int(raw["marker_a"], f"{what}: marker_a")
        marker_b = None if raw.get("marker_b") is None else _as_int(raw["marker_b"], f"{what}: marker_b")
        if marker_a is not None and marker_a == marker_b:
            raise ValueError(f"{what}: marker_a i marker_b to ten sam marker ({marker_a})")

        segments[name] = Segment(
            name=name,
            length_mm=length_mm,
            sigma_mm=sigma_mm,
            kind=str(raw.get("kind", "control")),
            marker_a=marker_a,
            marker_b=marker_b,
            reference=str(raw.get("reference", "")),
            measured_by=str(raw.get("measured_by", "")),
            date=str(raw.get("date", "")),
        )

    return Targets(segments=segments, dict_name=dict_name, markers=markers, source=source)


def load_targets(path: Path | str) -> Targets:
    """Wczytanie i walidacja `targets.json`."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: niepoprawny JSON ({exc})") from None
    return parse_targets(data, source=str(path))


# --------------------------------------------------------------------------- #
# Wczytanie pomiarów
# --------------------------------------------------------------------------- #


def read_measured_csv(path: Path | str) -> list[dict]:
    """`segment,measured` -> lista obserwacji dla `estimate_scale()`.

    Wiele wierszy z tą samą nazwą segmentu to powtórzenia pomiaru tego samego odcinka —
    trafiają do jednej obserwacji jako `measurements`. Puste wiersze i wiersze zaczynające
    się od `#` są pomijane (miejsce na notatki terenowe). Opcjonalna kolumna
    `sigma_measured` ma sens tylko przy pojedynczym pomiarze; przy powtórzeniach sigma
    liczy się z próbki i podana wartość byłaby dwuznaczna.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: plik nie ma żadnych wierszy z danymi")
    if "segment" not in (rows[0].keys()) or "measured" not in (rows[0].keys()):
        raise ValueError(f"{path}: oczekiwano kolumn 'segment' i 'measured', są: {list(rows[0])}")

    order: list[str] = []
    values: dict[str, list[float]] = {}
    sigmas: dict[str, list[Optional[float]]] = {}
    for i, row in enumerate(rows, start=2):  # +2: nagłówek i numeracja od 1
        name = (row.get("segment") or "").strip()
        if not name or name.startswith("#"):
            continue
        raw = (row.get("measured") or "").strip()
        if not raw:
            raise ValueError(f"{path}:{i}: segment {name!r} bez wartości 'measured'")
        value = _as_float(raw, f"{path}:{i}: measured")
        raw_sigma = (row.get("sigma_measured") or "").strip() if "sigma_measured" in row else ""
        sigma = _as_float(raw_sigma, f"{path}:{i}: sigma_measured") if raw_sigma else None
        if name not in values:
            order.append(name)
            values[name] = []
            sigmas[name] = []
        values[name].append(value)
        sigmas[name].append(sigma)

    if not order:
        raise ValueError(f"{path}: same komentarze i puste wiersze, brak pomiarów")

    observations = []
    for name in order:
        given = [s for s in sigmas[name] if s is not None]
        if given and len(values[name]) > 1:
            raise ValueError(
                f"{path}: segment {name!r} ma {len(values[name])} pomiarów i jawne "
                f"'sigma_measured' — przy powtórzeniach sigma liczy się z próbki, usuń kolumnę"
            )
        obs: dict[str, Any] = {"segment": name, "measurements": values[name]}
        if given:
            obs["sigma_measured"] = given[0]
        observations.append(obs)
    return observations


# --------------------------------------------------------------------------- #
# Estymacja (funkcje czyste)
# --------------------------------------------------------------------------- #


def _segment_factor(segment: Segment, measured: float, sigma_measured: Optional[float], n: int) -> SegmentFactor:
    """Współczynnik dla jednego segmentu: metry na jednostkę modelu.

    Propagacja względna: sigma_factor = factor * sqrt((sigma_ref/ref)^2 + (sigma_meas/meas)^2).
    Gdy niepewności pomiaru nie znamy (jeden pomiar, brak `sigma_measured`), zwracamy None —
    zero byłoby liczbą wymyśloną.
    """
    factor = (segment.length_mm / MM_PER_M) / measured
    if sigma_measured is None:
        sigma_factor = None
    else:
        rel = math.hypot(segment.sigma_mm / segment.length_mm, sigma_measured / measured)
        sigma_factor = factor * rel
    return SegmentFactor(
        name=segment.name,
        kind=segment.kind,
        ref_mm=segment.length_mm,
        sigma_ref_mm=segment.sigma_mm,
        measured=measured,
        sigma_measured=sigma_measured,
        factor=factor,
        sigma_factor=sigma_factor,
        n=n,
    )


def estimate_scale(observations: Iterable[dict], targets: Targets) -> ScaleEstimate:
    """Współczynnik skali osobno dla każdego zmierzonego segmentu + rozrzut.

    `observations`: lista słowników
        {"segment": nazwa, "measured": x [, "sigma_measured": s]}
      albo
        {"segment": nazwa, "measurements": [x1, x2, ...] [, "sigma_measured": s]}

    Przy `measurements` wartością jest **mediana**, a sigma — odchylenie standardowe próbki
    (ddof=1, czyli rozrzut pojedynczego kliknięcia, nie niepewność mediany; celowo
    ostrożniej). Jawne `sigma_measured` ma pierwszeństwo przed sigmą z próbki.

    Funkcja czysta: bez I/O, bez odrzucania outlierów, bez sprowadzania wyniku do jednej
    liczby. Wybór jednej liczby jest osobną, jawną decyzją — `select_factor()`.
    """
    rows: list[SegmentFactor] = []
    seen: set[str] = set()

    for i, obs in enumerate(observations):
        if not isinstance(obs, dict):
            raise ValueError(f"obserwacja #{i}: oczekiwano słownika, dostałem {type(obs).__name__}")
        name = obs.get("segment")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"obserwacja #{i}: brak niepustego pola 'segment'")
        name = name.strip()
        if name in seen:
            raise ValueError(
                f"segment {name!r} występuje w obserwacjach więcej niż raz — "
                f"powtórzenia pomiaru złóż w jedną pozycję z listą 'measurements'"
            )
        seen.add(name)
        segment = targets.segment(name)  # nieznana nazwa -> ValueError z nazwą

        has_single = obs.get("measured") is not None
        has_many = obs.get("measurements") is not None
        if has_single == has_many:
            raise ValueError(f"segment {name!r}: podaj dokładnie jedno z 'measured' / 'measurements'")

        if has_many:
            series = obs["measurements"]
            if isinstance(series, (str, bytes)) or not isinstance(series, Sequence):
                raise ValueError(f"segment {name!r}: 'measurements' musi być listą liczb")
            values = [_as_float(v, f"segment {name!r}: measurements[{k}]") for k, v in enumerate(series)]
            if not values:
                raise ValueError(f"segment {name!r}: 'measurements' jest puste")
            measured = float(np.median(values))
            sigma_measured = float(np.std(values, ddof=1)) if len(values) > 1 else None
            n = len(values)
        else:
            measured = _as_float(obs["measured"], f"segment {name!r}: measured")
            sigma_measured = None
            n = 1

        if measured <= 0:
            raise ValueError(f"segment {name!r}: zmierzona długość musi być > 0, dostałem {measured}")

        if obs.get("sigma_measured") is not None:
            sigma_measured = _as_float(obs["sigma_measured"], f"segment {name!r}: sigma_measured")
            if sigma_measured < 0:
                raise ValueError(f"segment {name!r}: sigma_measured musi być >= 0, dostałem {sigma_measured}")

        rows.append(_segment_factor(segment, measured, sigma_measured, n))

    if not rows:
        raise ValueError("brak obserwacji — nie ma z czego liczyć skali")

    factors = np.array([r.factor for r in rows], dtype=np.float64)
    median = float(np.median(factors))
    summary = Summary(
        median=median,
        mean=float(np.mean(factors)),
        min=float(np.min(factors)),
        max=float(np.max(factors)),
        std=float(np.std(factors, ddof=1)) if len(factors) > 1 else None,
        spread_pct=float((np.max(factors) - np.min(factors)) / median * 100.0),
        n_segments=len(rows),
    )
    return ScaleEstimate(segments=tuple(rows), summary=summary)


def select_factor(estimate: ScaleEstimate, use: str = "median", segment: Optional[str] = None) -> tuple[float, str]:
    """Jedna liczba do przeskalowania chmury + zapis, skąd się wzięła.

    Zwraca `(factor, how)`, gdzie `how` trafia do `<out>.scale.json` — żeby z samego pliku
    dało się odczytać, czy chmurę przeskalowano jednym wzorcem, czy uśrednieniem kilku.
    """
    if segment is not None:
        row = estimate.factor(segment)
        return row.factor, f"segment {segment!r} (n={row.n})"
    if use == "median":
        return estimate.summary.median, f"mediana z {estimate.summary.n_segments} segmentów"
    if use == "mean":
        return estimate.summary.mean, f"średnia z {estimate.summary.n_segments} segmentów"
    raise ValueError(f"nie znam sposobu wyboru {use!r}; dostępne: median, mean albo nazwa segmentu")


def apply_scale(points: np.ndarray, factor: float) -> np.ndarray:
    """Przeskalowanie tablicy (N,3) jednostek modelu na metry. Zwraca nową tablicę."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"oczekiwano tablicy (N,3), dostałem {array.shape}")
    factor = _as_float(factor, "factor")
    if factor <= 0:
        raise ValueError(f"factor musi być > 0, dostałem {factor}")
    return array * factor


# --------------------------------------------------------------------------- #
# Zapis przeskalowanej chmury
# --------------------------------------------------------------------------- #


def build_scale_record(
    factor: float,
    in_path: Path | str,
    out_path: Path | str,
    n_points: int,
    extra: dict,
    now: Optional[datetime] = None,
) -> dict:
    """Zawartość `<out>.scale.json` — funkcja czysta, żeby dało się ją sprawdzić testem.

    To jest jedyny ślad po tym, czym i na jakiej podstawie przeskalowano chmurę, więc
    idzie tu komplet: współczynnik, sposób wyboru, wszystkie policzone segmenty i data.
    """
    return {
        "factor": float(factor),
        "source_ply": str(in_path),
        "scaled_ply": str(out_path),
        "n_points": int(n_points),
        "units": "factor przelicza jednostki modelu na metry",
        "written": (now or datetime.now().astimezone()).isoformat(timespec="seconds"),
        **extra,
    }


def scale_ply(in_path: Path | str, out_path: Path | str, factor: float, record: dict) -> Path:
    """Przeskalowanie chmury PLY i zapis `<out>.scale.json` obok wyniku.

    Bez pliku z zapisem przeskalowana chmura jest nie do odtworzenia — nie widać, jakim
    współczynnikiem i z którego wzorca. Zwraca ścieżkę tego pliku.

    Uwaga: open3d przenosi współrzędne, kolory i normalne; **niestandardowe właściwości PLY
    (np. ufność per punkt) przy przejściu przez open3d giną**. Skalowanie jest jednorodne,
    więc normalne zostają bez zmian.
    """
    try:
        import open3d as o3d
    except ImportError as exc:  # env vid2cloud-tools nie ma open3d
        raise RuntimeError(
            "scale_ply() wymaga open3d (`pip install open3d` w env vid2cloud-tools); "
            "reszta modułu działa bez niego"
        ) from exc

    in_path, out_path = Path(in_path), Path(out_path)
    if not in_path.exists():
        raise ValueError(f"nie ma pliku {in_path}")
    cloud = o3d.io.read_point_cloud(str(in_path))
    if len(cloud.points) == 0:
        raise ValueError(f"{in_path}: open3d nie wczytał żadnego punktu")

    cloud.points = o3d.utility.Vector3dVector(apply_scale(np.asarray(cloud.points), factor))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(out_path), cloud):
        raise RuntimeError(f"open3d nie zapisał {out_path}")

    record_path = Path(str(out_path) + ".scale.json")
    payload = build_scale_record(factor, in_path, out_path, len(cloud.points), record)
    record_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return record_path


# --------------------------------------------------------------------------- #
# Raport tekstowy
# --------------------------------------------------------------------------- #


def _pm(value: Optional[float], fmt: str) -> str:
    """„± sigma" albo „± ?", gdy niepewności nie znamy."""
    return f"± {value:{fmt}}" if value is not None else "± ?     "


def format_table(estimate: ScaleEstimate) -> str:
    """Tabela per segment + wiersz zbiorczy. Bez wniosków — same liczby."""
    lines = [
        f"{'segment':<12}{'rodzaj':<10}{'ref [mm]':>20}{'w chmurze [j.m.]':>24}"
        f"{'factor [m/j.m.]':>24}{'n':>4}",
        "-" * 94,
    ]
    for row in estimate.segments:
        lines.append(
            f"{row.name:<12}{row.kind:<10}"
            f"{row.ref_mm:>12.1f} {_pm(row.sigma_ref_mm, '5.1f'):>7}"
            f"{row.measured:>14.5f} {_pm(row.sigma_measured, '7.5f'):>9}"
            f"{row.factor:>13.6f} {_pm(row.sigma_factor, '8.6f'):>10}"
            f"{row.n:>4}"
        )
    s = estimate.summary
    std = f"{s.std:.6f}" if s.std is not None else "?"
    lines += [
        "-" * 94,
        f"zbiorczo ({s.n_segments} segm.): mediana {s.median:.6f}  średnia {s.mean:.6f}  "
        f"min {s.min:.6f}  max {s.max:.6f}  std {std}",
        f"rozrzut (max-min)/mediana: {s.spread_pct:.2f} %",
    ]
    if s.spread_pct > SPREAD_WARN_PCT:
        lines.append(
            f"UWAGA: rozrzut {s.spread_pct:.2f} % > {SPREAD_WARN_PCT:.0f} % — współczynnik nie jest "
            f"jedną liczbą dla tej chmury. Wynik zostaje bez zmian (ostrzeżenie, nie korekta)."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    """Punkt wejścia `python -m vid2cloud.scale.estimate`.

    Argparse, nie typer — jak w `scale/aruco.py`: to wejście uruchamiane w env
    `vid2cloud-tools`, gdzie pakiet nie jest instalowany.
    """
    parser = argparse.ArgumentParser(
        prog="python -m vid2cloud.scale.estimate",
        description="Współczynnik skali z odległości między wzorcami — osobno dla każdego wzorca.",
    )
    parser.add_argument("--targets", type=Path, required=True, help="targets.json z długościami wzorców [mm].")
    parser.add_argument("--measured", type=Path, required=True, help="CSV: segment,measured (powtórzenia = wiele wierszy).")
    parser.add_argument("--apply", dest="apply_ply", type=Path, help="Chmura PLY do przeskalowania.")
    parser.add_argument("--out", type=Path, help="Wynikowy PLY (wymagany przy --apply).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-segment", metavar="NAZWA", help="Skaluj współczynnikiem tego jednego wzorca.")
    group.add_argument("--use", choices=("median", "mean"), default="median", help="Uśrednienie (domyślnie median).")
    parser.add_argument("--json", dest="json_path", type=Path, help="Zapisz wynik (per segment + zbiorczo) do pliku.")
    args = parser.parse_args(argv)

    if args.apply_ply is not None and args.out is None:
        parser.error("--apply wymaga --out")
    for path in (args.targets, args.measured):
        if not path.exists():
            parser.error(f"nie ma pliku {path}")

    try:
        targets = load_targets(args.targets)
        observations = read_measured_csv(args.measured)
        estimate = estimate_scale(observations, targets)
    except ValueError as exc:
        print(f"błąd: {exc}", file=sys.stderr)
        return 1

    print(f"targets:  {args.targets}")
    print(f"pomiary:  {args.measured}")
    print(format_table(estimate))

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(estimate.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON:     {args.json_path}")

    if args.apply_ply is None:
        # Bez --apply wypisujemy wybraną liczbę tylko wtedy, gdy autor sam o nią poprosił
        # (--use-segment). Domyślnie zostaje tabela: ta lista jest wynikiem, nie mediana.
        if args.use_segment is not None:
            try:
                factor, how = select_factor(estimate, segment=args.use_segment)
            except ValueError as exc:
                print(f"błąd: {exc}", file=sys.stderr)
                return 1
            print(f"\nwybrany factor: {factor:.6f} m/j.m.  ({how})  — bez --apply nic nie zapisano")
        return 0

    try:
        factor, how = select_factor(estimate, use=args.use, segment=args.use_segment)
        record = {
            "how": how,
            "targets": str(args.targets),
            "measured": str(args.measured),
            **estimate.as_dict(),
        }
        record_path = scale_ply(args.apply_ply, args.out, factor, record)
    except (ValueError, RuntimeError) as exc:
        print(f"błąd: {exc}", file=sys.stderr)
        return 1

    print(f"\nużyty factor: {factor:.6f} m/j.m.  ({how})")
    print(f"chmura:   {args.out}")
    print(f"zapis:    {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
