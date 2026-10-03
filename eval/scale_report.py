#!/usr/bin/env python3
"""Zestawienie skali z kilku przebiegów: tabela per odcinek i wykres współczynników.

Wejście: `runs/<sesja>/<nagranie>/scale.json` (wynik `python -m vid2cloud.scale.estimate
--json`). Wyjście w katalogu sesji:

* `scale_summary.csv` — wiersz na (nagranie, odcinek): długość z taśmy, odległość
  w chmurze (mediana kliknięć), powtarzalność kliknięć bezwzględna i względna,
  współczynnik ± sigma i jego odchylenie od mediany tego nagrania;
* `scale_factors.png` — dwa panele na wspólnej osi odcinków: (A) współczynnik ± sigma
  z medianą nagrania, (B) względna powtarzalność kliknięć `sigma_measured / measured`.

Nazwy nagrań na wykresie i w tabeli biorą się z kolumny `nazwa` w
`<sesja>/notes/nagrania.csv` (np. `daleko_wolno`); bez niej zostaje numer z nazwy pliku.
Pliki i katalogi się nie zmieniają — alias jest tylko etykietą.

Nic tu nie jest wykluczane ani progowane: wszystkie nagrania idą tą samą drogą.
Skrypt niczego nie interpretuje — liczy i rysuje.

Env: `vid2cloud-tools` (numpy, matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

#: Kolory serii — pierwsze trzy sloty palety kategorycznej (walidowane dla wszystkich par
#: w trybie jasnym); kształt znacznika jest drugim kodowaniem, więc kolor nie niesie
#: tożsamości sam.
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "D")]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load_runs(runs_dir: Path, names: Optional[list[str]] = None) -> dict[str, dict]:
    """{nagranie: zawartość scale.json} — w kolejności podanej albo alfabetycznej."""
    paths = ([runs_dir / n / "scale.json" for n in names] if names
             else sorted(runs_dir.glob("*/scale.json")))
    out = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"nie ma {path} — najpierw scale.estimate --json")
        out[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def load_aliases(path: Path) -> dict[str, str]:
    """{nazwa pliku bez .mp4: alias} z kolumny `nazwa` w `nagrania.csv`; pusty, gdy brak."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {Path(row["plik"]).stem: row["nazwa"].strip()
                for row in csv.DictReader(f) if (row.get("nazwa") or "").strip()}


def display_name(name: str, aliases: Optional[dict[str, str]] = None) -> str:
    """Etykieta nagrania: alias z numerem w nawiasie, albo sam numer (końcówka nazwy pliku).

    Numer zostaje zawsze — po nim nagranie da się znaleźć w runs/, logach i config.json.
    """
    short = name.rsplit("_", 1)[-1]
    alias = (aliases or {}).get(name)
    return f"{alias} ({short})" if alias else short


def summary_rows(runs: dict[str, dict], aliases: Optional[dict[str, str]] = None) -> list[dict]:
    """Wiersze `scale_summary.csv`. Funkcja czysta — liczy na słownikach z scale.json."""
    rows = []
    for name, data in runs.items():
        median = data["summary"]["median"]
        for seg in data["segments"]:
            sm = seg["sigma_measured"]
            rows.append({
                "nagranie": name,
                "nazwa": (aliases or {}).get(name, ""),
                "segment": seg["name"],
                "ref_mm": seg["ref_mm"],
                "sigma_ref_mm": seg["sigma_ref_mm"],
                "n_klikniec": seg["n"],
                "w_chmurze": f"{seg['measured']:.5f}",
                "sigma_klikniec": "" if sm is None else f"{sm:.5f}",
                "powtarzalnosc_pct": "" if sm is None else f"{100 * sm / seg['measured']:.2f}",
                "factor": f"{seg['factor']:.6f}",
                "sigma_factor": "" if seg["sigma_factor"] is None else f"{seg['sigma_factor']:.6f}",
                "odchylenie_od_mediany_pct": f"{100 * (seg['factor'] / median - 1):+.2f}",
                "mediana_nagrania": f"{median:.6f}",
                "rozrzut_nagrania_pct": f"{data['summary']['spread_pct']:.2f}",
            })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(runs: dict[str, dict], path: Path, title: str = "",
         aliases: Optional[dict[str, str]] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Oś odcinków w kolejności z targets.json (pierwsze nagranie) — dla sesji 21.09 to
    # łańcuch statywów 9-8-3-10-11-12 od lewej do prawej.
    first = next(iter(runs.values()))
    order = [s["name"] for s in first["segments"]]
    ref = {s["name"]: (s["ref_mm"], s["sigma_ref_mm"]) for s in first["segments"]}
    labels = [f"{n}\n{ref[n][0]:.0f} mm ±{ref[n][1]:g}" for n in order]
    x = list(range(len(order)))
    width = 0.22
    offsets = [(i - (len(runs) - 1) / 2) * width for i in range(len(runs))]

    fig, (ax_f, ax_r) = plt.subplots(
        2, 1, figsize=(9.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]},
        facecolor=SURFACE)
    for ax in (ax_f, ax_r):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=9)

    for (name, data), (color, marker), dx in zip(runs.items(), SERIES, offsets):
        segs = {s["name"]: s for s in data["segments"]}
        xs = [xi + dx for xi, n in zip(x, order) if n in segs]
        present = [n for n in order if n in segs]
        f = [segs[n]["factor"] for n in present]
        sf = [segs[n]["sigma_factor"] or 0.0 for n in present]
        rel = [100 * (segs[n]["sigma_measured"] or 0.0) / segs[n]["measured"] for n in present]
        label = display_name(name, aliases)
        med, spread = data["summary"]["median"], data["summary"]["spread_pct"]

        ax_f.axhline(med, color=color, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.8, zorder=1)
        ax_f.errorbar(xs, f, yerr=sf, fmt=marker, color=color, ms=7, mec=SURFACE, mew=1.5,
                      elinewidth=1.6, capsize=0, zorder=3,
                      label=f"{label}\nmediana {med:.3f} · rozrzut {spread:.1f} %")
        ax_r.plot(xs, rel, marker, color=color, ms=7, mec=SURFACE, mew=1.5, linestyle="none", zorder=3)
        for xi, r in zip(xs, rel):
            ax_r.plot([xi, xi], [0, r], color=color, linewidth=1.6, zorder=2)

    ax_f.set_ylabel("współczynnik skali [m / j. chmury]", color=INK_2, fontsize=10)
    # Legenda nad panelem, nie w nim — w środku zasłaniała punkty przy 8-9 i 3-8.
    handles, names = ax_f.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK, handletextpad=0.4, columnspacing=2.0)
    ax_r.set_ylabel("powtarzalność\nkliknięć [%]", color=INK_2, fontsize=10)
    ax_r.set_ylim(bottom=0)
    ax_r.set_xticks(x, labels)
    ax_r.set_xlabel("odcinek wzorcowy (taśma)", color=INK_2, fontsize=10)
    fig.text(0.01, 0.975, title or "Współczynnik skali per odcinek", color=INK, fontsize=11,
             ha="left", va="center")
    fig.text(0.99, 0.975, "linia przerywana = mediana nagrania", color=MUTED, fontsize=8,
             ha="right", va="center")
    fig.text(0.01, 0.005,
             "Wąsy w panelu górnym: sigma współczynnika (taśma + rozrzut kliknięć). "
             "Panel dolny: sigma_kliknięć / mediana kliknięć, 4–5 kliknięć na odcinek.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.89))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python eval/scale_report.py", description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs/2026-09-21"), help="Katalog sesji w runs/.")
    p.add_argument("--recordings", nargs="*", help="Nazwy nagrań (domyślnie wszystkie z scale.json).")
    p.add_argument("--session", type=Path, default=Path("data/2026-09-21"),
                   help="Katalog sesji; aliasy nagrań z notes/nagrania.csv (kolumna 'nazwa').")
    p.add_argument("--title", default="", help="Tytuł wykresu.")
    args = p.parse_args(argv)

    runs = load_runs(args.runs, args.recordings)
    if len(runs) > len(SERIES):
        raise SystemExit(f"wykres mieści {len(SERIES)} nagrania; podaj --recordings")
    aliases = load_aliases(args.session / "notes" / "nagrania.csv")
    rows = summary_rows(runs, aliases)
    csv_path, png_path = args.runs / "scale_summary.csv", args.runs / "scale_factors.png"
    write_csv(rows, csv_path)
    plot(runs, png_path, args.title, aliases)
    print(f"tabela: {csv_path}  ({len(rows)} wierszy)\nwykres: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
