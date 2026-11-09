"""Test E — współczynnik skali sieci względem głębokości odcinka w układzie kamery.

Tabela 1 (pary taśmy, sesja 21.09): wiersze `scale_per_keyframe.csv` trzech przebiegów adaptera
z 03.10 — przebieg, kf, odcinek, z_mid_m, net_cr (= factor_net_created), final
(= factor_final). Spearman ρ(net_cr, z_mid_m) osobno na nagranie i łącznie, zakres z_mid_m,
wykres net_cr względem z_mid_m (kolor = nagranie).

Tabela 2 („bok znacznika, mieszkanie”, VID_ODLEGLOSC): `marker_side_scale.csv` z
eval/marker_side_scale.py — inny wzorzec i inny pokój, więc NIE jest łączona z tabelą 1.
Spearman tylko informacyjnie (3 plansze).

Łączny ρ (tabele 1 i 2) liczony jest między klatkami kluczowymi, więc jest obciążony skalą
klatki — każda klatka ma własne s_k i własny błąd skali sieci (decyzja autora 2026-10-30,
dopisana 2026-11-04). Dlatego obok są tabele w obrębie klatki kluczowej (`within_keyframe_pairs`):
w każdej klatce z ≥ 2 odcinkami (planszami) wszystkie pary i < j, znak (y_j − y_i)·(x_j − x_i)
— > 0 zgodna, < 0 niezgodna, = 0 remis (jak w τ Kendalla) — i odchylenie y od mediany klatki [%]:
    1b (21.09): x = z_mid_m, y = net_cr -> test_e_w_klatce_pary.csv, test_e_w_klatce_podsumowanie.csv;
        informacyjnie y = final -> test_e_w_klatce_pary_final.csv, test_e_w_klatce_podsumowanie_final.csv;
    2b (VID_ODLEGLOSC): x = z_m, y = factor_net_median, grupy po kf -> test_e_bok_znacznika_w_klatce_*.csv;
        wiersze z side_spread_pct > 5 % mają flagę (nie są usuwane).

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


RHO_NOTE = "między klatkami kluczowymi — obciążony skalą klatki (decyzja 2026-10-30)"
SPREAD_FLAG_PCT = 5.0


def within_keyframe_pairs(rows: list[dict], key, x: str, y: str, label: str | None = None) -> list[dict]:
    """Pary i < j w każdej grupie `key` (np. (przebieg, kf)) z ≥ 2 wierszami, w kolejności wierszy.

    znak = sign((y_j − y_i)·(x_j − x_i)): "zgodna" (> 0), "niezgodna" (< 0), "remis" (= 0).
    dev_*_pct = (y − mediana y grupy) / mediana · 100. Wiersze bez x albo y nie wchodzą do grup.
    `row_i`, `row_j` — indeksy wierszy w `rows` (do policzenia zakresu odchyleń bez powtórzeń).
    """
    key = (key,) if isinstance(key, str) else tuple(key)
    groups: dict[tuple, list[int]] = {}
    for idx, r in enumerate(rows):
        if r.get(x) is None or r.get(y) is None:
            continue
        groups.setdefault(tuple(r[k] for k in key), []).append(idx)
    out = []
    for g, members in groups.items():
        if len(members) < 2:
            continue
        med = float(np.median([rows[i][y] for i in members]))
        dev = {i: (rows[i][y] - med) / med * 100.0 for i in members}
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                dx, dy = rows[j][x] - rows[i][x], rows[j][y] - rows[i][y]
                prod = dx * dy
                pair = dict(zip(key, g))
                if label is not None:
                    pair.update({f"{label}_i": rows[i][label], f"{label}_j": rows[j][label]})
                pair.update({"row_i": i, "row_j": j, "x_i": rows[i][x], "x_j": rows[j][x],
                             "y_i": rows[i][y], "y_j": rows[j][y], "dx": dx, "dy": dy,
                             "znak": "zgodna" if prod > 0 else ("niezgodna" if prod < 0 else "remis"),
                             "dev_i_pct": dev[i], "dev_j_pct": dev[j], "median_y": med, "n_w_klatce": len(members)})
                out.append(pair)
    return out


def within_summary(pairs: list[dict], by: str, key, order: list) -> list[dict]:
    """Na wartość `by` (np. przebieg) i łącznie: klatki z ≥ 2 wierszami, pary, zgodne, niezgodne, remisy,
    zakres odchyleń % (każdy wiersz raz)."""
    key = (key,) if isinstance(key, str) else tuple(key)
    out = []
    for g in [*order, None]:
        sel = [p for p in pairs if g is None or p[by] == g]
        devs = {}
        for p in sel:
            devs[p["row_i"]], devs[p["row_j"]] = p["dev_i_pct"], p["dev_j_pct"]
        out.append({by: "łącznie" if g is None else g,
                    "klatki_ge2": len({tuple(p[k] for k in key) for p in sel}),
                    "pary": len(sel),
                    "zgodne": sum(p["znak"] == "zgodna" for p in sel),
                    "niezgodne": sum(p["znak"] == "niezgodna" for p in sel),
                    "remisy": sum(p["znak"] == "remis" for p in sel),
                    "odch_min_pct": min(devs.values()) if devs else None,
                    "odch_max_pct": max(devs.values()) if devs else None})
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

WITHIN_PAIR_COLUMNS = ("przebieg", "nagranie", "kf", "n_w_klatce", "odcinek_i", "odcinek_j", "x_i", "x_j",
                       "y_i", "y_j", "dx", "dy", "znak", "median_y", "dev_i_pct", "dev_j_pct")
WITHIN_SUMMARY_COLUMNS = ("przebieg", "klatki_ge2", "pary", "zgodne", "niezgodne", "remisy",
                          "odch_min_pct", "odch_max_pct", "nagranie")
WITHIN_MARKER_COLUMNS = ("nagranie", "kf", "n_w_klatce", "marker_id_i", "marker_id_j", "x_i", "x_j", "y_i", "y_j",
                         "dx", "dy", "znak", "median_y", "dev_i_pct", "dev_j_pct", "spread_i_pct", "spread_j_pct",
                         "flaga_rozrzut_gt_5pct")

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
    print(f"\nSpearman ρ(net_cr, z_mid_m) — {RHO_NOTE}\n")
    print(md_table(("przebieg", "n", "rho", "z_mid_m_min", "z_mid_m_max"), corr, 3))
    print(f"\nzapisano {args.out}/test_e_pary.csv, test_e_spearman.csv, test_e_net_cr_z_mid.png")

    # Tabela 1b — w obrębie klatki kluczowej (net_cr) i informacyjnie final
    order = list(dict.fromkeys(r["przebieg"] for r in rows))
    rec = {r["przebieg"]: r["nagranie"] for r in rows}
    for p in args.runs:  # przebieg bez wierszy ok też ma swój wiersz podsumowania (zera)
        rec.setdefault(p.name, Path(json.loads((p / "config.json").read_text(encoding="utf-8"))["video"]).stem)
        if p.name not in order:
            order.append(p.name)
    for y, suffix, title in (("net_cr", "", "Tabela 1b — w obrębie klatki kluczowej, y = net_cr"),
                             ("final", "_final", "Tabela 1b (informacyjnie) — w obrębie klatki kluczowej, y = final")):
        pairs = within_keyframe_pairs(rows, ("przebieg", "kf"), "z_mid_m", y, label="odcinek")
        for pr in pairs:
            pr["nagranie"] = rec[pr["przebieg"]]
        summ = within_summary(pairs, "przebieg", ("przebieg", "kf"), order)
        for sm in summ:
            sm["nagranie"] = rec.get(sm["przebieg"], "")
        write_csv(args.out / f"test_e_w_klatce_pary{suffix}.csv", WITHIN_PAIR_COLUMNS, pairs)
        write_csv(args.out / f"test_e_w_klatce_podsumowanie{suffix}.csv", WITHIN_SUMMARY_COLUMNS, summ)
        print(f"\n{title} (x = z_mid_m)\n")
        print(md_table(("przebieg", "kf", "odcinek_i", "odcinek_j", "x_i", "x_j", "y_i", "y_j", "znak",
                        "dev_i_pct", "dev_j_pct"), pairs))
        print()
        print(md_table(WITHIN_SUMMARY_COLUMNS[:-1], summ, 2))
    print(f"\nzapisano {args.out}/test_e_w_klatce_{{pary,podsumowanie}}{{,_final}}.csv")

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
    print(f"\nSpearman ρ(współczynnik, z_m) — informacyjnie; łącznie: {RHO_NOTE}\n")
    print(md_table(cols, mcorr, 3))

    # Tabela 2b — w obrębie klatki kluczowej, po planszach
    for r in ok:
        r["rozrzut_gt_5pct"] = r["side_spread_pct"] is not None and r["side_spread_pct"] > SPREAD_FLAG_PCT
    mpairs = within_keyframe_pairs(ok, ("kf",), "z_m", "factor_net_median", label="marker_id")
    for pr in mpairs:
        ri, rj = ok[pr["row_i"]], ok[pr["row_j"]]
        pr.update({"spread_i_pct": ri["side_spread_pct"], "spread_j_pct": rj["side_spread_pct"],
                   "flaga_rozrzut_gt_5pct": ri["rozrzut_gt_5pct"] or rj["rozrzut_gt_5pct"]})
    for pr in mpairs:
        pr["nagranie"] = "VID_ODLEGLOSC"
    msumm = within_summary(mpairs, "nagranie", ("kf",), ["VID_ODLEGLOSC"])[:1]  # jedno nagranie = łącznie
    msumm[0]["pary_z_flaga"] = sum(pr["flaga_rozrzut_gt_5pct"] for pr in mpairs)
    write_csv(args.marker_out / "test_e_bok_znacznika_w_klatce_pary.csv", WITHIN_MARKER_COLUMNS, mpairs)
    write_csv(args.marker_out / "test_e_bok_znacznika_w_klatce_podsumowanie.csv",
              ("nagranie", *WITHIN_SUMMARY_COLUMNS[1:-1], "pary_z_flaga"), msumm)
    print("\nTabela 2b — w obrębie klatki kluczowej (x = z_m, y = factor_net_median; flaga: rozrzut boków > 5 %)\n")
    print(md_table(("kf", "marker_id_i", "marker_id_j", "x_i", "x_j", "y_i", "y_j", "znak", "dev_i_pct", "dev_j_pct",
                    "spread_i_pct", "spread_j_pct", "flaga_rozrzut_gt_5pct"), mpairs))
    print()
    print(md_table(("nagranie", *WITHIN_SUMMARY_COLUMNS[1:-1], "pary_z_flaga"), msumm, 2))
    print(f"\nzapisano {args.marker_out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
