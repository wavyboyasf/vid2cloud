"""Test E — współczynnik skali sieci względem głębokości odcinka w układzie kamery.

Tabela 1 (pary taśmy, sesja 21.09): wiersze `scale_per_keyframe.csv` trzech przebiegów adaptera
z 03.10 — przebieg, kf, odcinek, z_mid_m, net_cr (= factor_net_created), final
(= factor_final). Spearman ρ(net_cr, z_mid_m) osobno na nagranie i łącznie, zakres z_mid_m,
wykres net_cr względem z_mid_m (kolor = nagranie).

Tabela 2 („bok znacznika, mieszkanie”, VID_ODLEGLOSC): `marker_side_scale.csv` z
eval/marker_side_scale.py — inny wzorzec i inny pokój, więc NIE jest łączona z tabelą 1.
Spearman tylko informacyjnie (3 plansze).

Skrypt nie interpretuje — wypisuje liczby. Env vid2cloud-tools.

    python eval/test_e.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER = REPO_ROOT / "runs" / "2026-09-21" / "adapter"
DEFAULT_RUNS = [ADAPTER / n for n in ("daleko_wolno", "blisko_z_gory", "daleko_szybko_pion")]
DEFAULT_OUT = REPO_ROOT / "runs" / "2026-09-21" / "test_e"
DEFAULT_MARKER = REPO_ROOT / "runs" / "odleglosc" / "adapter" / "marker_side_scale.csv"
DEFAULT_MARKER_OUT = REPO_ROOT / "runs" / "odleglosc" / "test_e"

PAIR_COLUMNS = ("przebieg", "nagranie", "kf", "odcinek", "z_mid_m", "net_cr", "final", "z_mid_canon",
                "net_cr_median_run")


# --------------------------------------------------------------------------- #
# Funkcje czyste
# --------------------------------------------------------------------------- #


def rankdata(x) -> np.ndarray:
    """Rangi 1..n, remisy — średnia ranga (jak scipy.stats.rankdata, method='average')."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(x, y) -> float | None:
    """ρ Spearmana = korelacja Pearsona rang. None, gdy n < 3 albo jedna ze zmiennych stała."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("różne długości")
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return None if den == 0 else float((rx * ry).sum() / den)


def _f(v: str):
    return float(v) if v not in ("", None) else None


def pair_rows(run_dir: Path) -> list[dict]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rec = Path(cfg["video"]).stem
    out = []
    with open(run_dir / "scale_per_keyframe.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "z_mid_m" not in r:
                raise ValueError(f"{run_dir}: scale_per_keyframe.csv bez z_mid_m — przelicz eval/scale_per_keyframe.py")
            if r["status"] != "ok":
                continue
            out.append({"przebieg": run_dir.name, "nagranie": rec, "kf": int(r["kf"]), "odcinek": r["segment"],
                        "z_mid_m": _f(r["z_mid_m"]), "net_cr": _f(r["factor_net_created"]),
                        "final": _f(r["factor_final"]), "z_mid_canon": _f(r["z_mid_canon"]),
                        "net_cr_median_run": _f(r["net_cr_median_run"])})
    return out


def correlations(rows: list[dict], group: str, x: str, y: str) -> list[dict]:
    """ρ(x, y) dla każdej grupy i łącznie + zakres y."""
    out = []
    groups = sorted({r[group] for r in rows}, key=lambda g: [r[group] for r in rows].index(g))
    for g in [*groups, None]:
        sel = [r for r in rows if g is None or r[group] == g]
        sel = [r for r in sel if r[x] is not None and r[y] is not None]
        ys = [r[y] for r in sel]
        out.append({group: "łącznie" if g is None else g, "n": len(sel),
                    "rho": spearman([r[x] for r in sel], ys),
                    f"{y}_min": min(ys) if ys else None, f"{y}_max": max(ys) if ys else None})
    return out


# --------------------------------------------------------------------------- #
# Zapis, wykres
# --------------------------------------------------------------------------- #


def _cell(v) -> str:
    if v is None:
        return ""
    return f"{v:.9g}" if isinstance(v, float) else str(v)


def write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in columns])


def _fmt(v, nd=4):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def md_table(columns, rows, nd=4) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c), nd) for c in columns) + " |")
    return "\n".join(lines)


# Paleta referencyjna (skill dataviz, palette.md), tryb jasny — rysunki do druku. Kategorie w stałej
# kolejności slotów 1–5; kształt znacznika to drugie kodowanie (aqua i żółty < 3:1 do tła).
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "D", "^", "v"]


def plot_scatter(rows: list[dict], group: str, x: str, y: str, xlabel: str, ylabel: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = list(dict.fromkeys(r[group] for r in rows))
    for i, g in enumerate(groups):
        sel = [r for r in rows if r[group] == g and r[x] is not None and r[y] is not None]
        ax.plot([r[x] for r in sel], [r[y] for r in sel], MARKERS[i % len(MARKERS)], ls="none", ms=6.5,
                color=CATEGORICAL[i % len(CATEGORICAL)], markeredgecolor=SURFACE, markeredgewidth=1.0,
                label=f"{g} (n = {len(sel)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tabela 2 — bok znacznika
# --------------------------------------------------------------------------- #

MARKER_COLUMNS = ("kf", "frame_index", "marker_id", "z_canon", "z_m", "side_px_512_mean", "factor_net_median",
                  "side_spread_pct", "factor_final_median", "status")


def marker_rows(path: Path) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {k: r.get(k) for k in MARKER_COLUMNS}
            for k in ("z_canon", "z_m", "side_px_512_mean", "factor_net_median", "side_spread_pct",
                      "factor_final_median"):
                row[k] = _f(r.get(k, ""))
            row["kf"], row["marker_id"] = int(r["kf"]), int(r["marker_id"])
            row["plansza"] = f"ID {row['marker_id']}"
            out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--marker-csv", type=Path, default=DEFAULT_MARKER)
    ap.add_argument("--marker-out", type=Path, default=DEFAULT_MARKER_OUT)
    args = ap.parse_args(argv)

    # Tabela 1
    rows = [r for p in args.runs for r in pair_rows(p)]
    write_csv(args.out / "test_e_pary.csv", PAIR_COLUMNS, rows)
    corr = correlations(rows, "przebieg", "net_cr", "z_mid_m")
    write_csv(args.out / "test_e_spearman.csv", ("przebieg", "n", "rho", "z_mid_m_min", "z_mid_m_max"), corr)
    plot_scatter(rows, "przebieg", "z_mid_m", "net_cr",
                 "głębokość środka odcinka z_mid_m [m] (z_mid_canon · mediana net_cr przebiegu)",
                 "net_cr = L / d_canon przy powstaniu kf [m / jedn. sieci]", args.out / "test_e_net_cr_z_mid.png")
    print("Tabela 1 — pary taśmy, sesja 21.09 (status ok)\n")
    print(md_table(("przebieg", "kf", "odcinek", "z_mid_m", "net_cr", "final"), rows))
    print("\nSpearman ρ(net_cr, z_mid_m)\n")
    print(md_table(("przebieg", "n", "rho", "z_mid_m_min", "z_mid_m_max"), corr, 3))
    print(f"\nzapisano {args.out}/test_e_pary.csv, test_e_spearman.csv, test_e_net_cr_z_mid.png")

    # Tabela 2
    if not args.marker_csv.exists():
        print(f"\nTabela 2: brak {args.marker_csv} — nie liczono")
        return 0
    mrows = marker_rows(args.marker_csv)
    ok = [r for r in mrows if r["status"] == "ok"]
    write_csv(args.marker_out / "test_e_bok_znacznika.csv", MARKER_COLUMNS, mrows)
    # Informacyjnie: per plansza (po klatkach kluczowych), per klatka kluczowa (po planszach) i łącznie.
    mcorr = [{"grupa": f"plansza {c['plansza']}" if c["plansza"] != "łącznie" else "łącznie", **c}
             for c in correlations(ok, "plansza", "factor_net_median", "z_m")]
    for r in ok:
        r["klatka"] = f"kf {r['kf']}"
    mcorr[-1:-1] = [{"grupa": c["klatka"], **c} for c in correlations(ok, "klatka", "factor_net_median", "z_m")
                    if c["klatka"] != "łącznie"]
    cols = ("grupa", "n", "rho", "z_m_min", "z_m_max")
    write_csv(args.marker_out / "test_e_bok_znacznika_spearman.csv", cols, mcorr)
    plot_scatter(ok, "plansza", "z_m", "factor_net_median",
                 "głębokość środka planszy z_m [m] (z_canon · mediana współczynnika przebiegu)",
                 "149,7 mm / bok, mediana 4 boków [m / jedn.]",
                 args.marker_out / "test_e_bok_znacznika.png")
    print("\nTabela 2 — bok znacznika, mieszkanie (VID_ODLEGLOSC; nie łączyć z tabelą 1)\n")
    print(md_table(MARKER_COLUMNS, mrows))
    print("\nSpearman ρ(współczynnik, z_m) — informacyjnie\n")
    print(md_table(cols, mcorr, 3))
    print(f"\nzapisano {args.marker_out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
