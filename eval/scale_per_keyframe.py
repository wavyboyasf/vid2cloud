"""Skala per klatka kluczowa: współczynnik każdego odcinka taśmy rozłożony na sieć i graf pozycji.

Dla każdej klatki kluczowej k i każdego odcinka z targets.json, którego OBA markery są w niej
wykryte (`aruco_keyframes.csv`), punkty odniesienia (pole `reference`) są liftowane do 3D
z pointmapy klatki k (`scale/lift.py`) w dwóch stanach — przy powstaniu (`kf_k_created.npz`)
i na końcu (`kf_k_final.npz`). d_canon to odległość punktów w układzie kamery klatki k
(jednostki sieci), s_k — skala Sim(3) klatki k z `poses_history.csv`. Współczynniki w metrach
na jednostkę chmury (jak `scale_summary.csv` i `scale/estimate.py`), L = length_mm / 1000:

    factor_net_created = L / d_canon_created                 (sama sieć, bez grafu)
    factor_online      = L / (s_created · d_canon_created)   (chmura w chwili powstania kf)
    factor_final       = L / (s_final · d_canon_final)       (chmura na końcu, cloud.ply)

Diagnostyka: d z mediany w oknie 3×3 zamiast płaszczyzny + promienia i różnica
`diff_win3_mm = L_mm · (d_win3 / d − 1)` — bez progu.

Głębokość odcinka (test E): z_a_canon, z_b_canon — współrzędna z końców odcinka w układzie
kamery klatki k (jednostki sieci), z_mid_canon — z środka odcinka; wszystkie ze stanu przy
powstaniu, czyli z tych samych punktów co d_canon_created i factor_net_created. Na metry:
z_mid_m = z_mid_canon · mediana(factor_net_created) WSZYSTKICH odcinków przebiegu
(kolumna net_cr_median_run) — nie współczynnikiem tej samej pary, bo wtedy z_mid_m i
factor_net_created byłyby zależne z definicji.

Wyjście w każdym katalogu przebiegu: scale_per_keyframe.csv, s_history.csv,
fig_a_factor_online_final.png, fig_b_s_history.png, fig_c_factor_net_final.png,
markers_world.ply (środki markerów w świecie, T_WC_final · X, kolor = kf). Obok katalogów
przebiegów: porownanie_scale_summary.csv (mediana factor_final per odcinek obok pomiaru
ręcznego). Wejście: tylko przebiegi z single_thread = true. Env vid2cloud-tools.

Liczenie (wczytanie przebiegu, lift, współczynniki, zapis CSV/PLY) jest od 2026-11-04
w `vid2cloud/scale/per_keyframe.py` — wspólne z `vid2cloud scale`; tutaj zostały rysunki
i porównanie z pomiarem ręcznym.

    python eval/scale_per_keyframe.py            # trzy przebiegi z 03.10, targets z sesji 21.09
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vid2cloud.scale.estimate import Segment, load_targets  # noqa: E402
from vid2cloud.scale.lift import C_MIN  # noqa: E402
# Liczenie i zapis przeniesione 2026-11-04 do vid2cloud/scale/per_keyframe.py; nazwy zostają tu
# dostępne dla eval/half_pixel_check.py i innych skryptów, które importują je z tego modułu.
from vid2cloud.scale.per_keyframe import (  # noqa: E402,F401
    COLUMNS,
    S_HISTORY_COLUMNS,
    STATES,
    Run,
    add_depth_m,
    load_run,
    marker_centers_world,
    s_at,
    s_history_rows,
    segment_rows,
    segments_with_refs,
    write_csv,
    write_markers_ply,
)

ADAPTER = REPO_ROOT / "runs" / "2026-09-21" / "adapter"
DEFAULT_RUNS = [ADAPTER / n for n in ("daleko_wolno", "blisko_z_gory", "daleko_szybko_pion")]
DEFAULT_TARGETS = REPO_ROOT / "data" / "2026-09-21" / "notes" / "targets.json"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "2026-09-21" / "scale_summary.csv"


# --------------------------------------------------------------------------- #
# Rysunki
# --------------------------------------------------------------------------- #

# Paleta referencyjna (skill dataviz, palette.md), tryb jasny — rysunki są do druku.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
#: Rampa sekwencyjna (niebieska) od kroku 250 do 700 — indeks klatki kluczowej.
KF_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5,
        "legend.frameon": False,
    })


def kf_colors(n: int):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("kf", KF_RAMP)
    return cmap, [cmap(i / max(n - 1, 1)) for i in range(n)]


def _panels(plt, segments: list[Segment], rows: list[dict], series, title: str, ylabel: str, path: Path) -> None:
    """Małe panele: jeden na odcinek; `series` = [(kolumna, etykieta, styl dict)], łączone pionową kreską."""
    ok = [r for r in rows if r["status"] == "ok"]
    vals = [r[c] for r in ok for c, _, _ in series if r.get(c) is not None]
    fig, axes = plt.subplots(1, len(segments), figsize=(2.6 * len(segments), 3.4), sharey=True, sharex=True)
    t_max = max((r["pts_s"] for r in rows), default=1.0)
    for ax, seg in zip(np.atleast_1d(axes), segments):
        mine = sorted((r for r in ok if r["segment"] == seg.name), key=lambda r: r["pts_s"])
        ax.set_title(f"{seg.name}  (L = {seg.length_mm:.0f} mm)", color=INK, loc="left")
        for r in mine:
            ys = [r[c] for c, _, _ in series]
            ax.plot([r["pts_s"]] * len(ys), ys, color=AXIS, lw=0.8, zorder=1)
        for c, label, style in series:
            ax.plot([r["pts_s"] for r in mine], [r[c] for r in mine], ls="none", label=label, zorder=3, **style)
        if not mine:
            ax.text(0.5, 0.5, "brak klatki kluczowej\nz oboma markerami", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED, fontsize=8)
        ax.set_xlim(-0.03 * t_max, t_max * 1.03 + 1)
        ax.set_xlabel("czas nagrania [s]")
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    if vals:
        lo, hi = min(vals), max(vals)
        pad = 0.08 * (hi - lo) if hi > lo else 0.05 * hi
        np.atleast_1d(axes)[0].set_ylim(lo - pad, hi + pad)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(series), bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_run(run: Run, segments: list[Segment], rows: list[dict], s_hist: list[dict]) -> list[Path]:
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    ring = {"markeredgecolor": SURFACE, "markeredgewidth": 1.0}
    title = f"{run.name} ({run.recording})"
    out = []

    p = run.path / "fig_a_factor_online_final.png"
    _panels(plt, segments, rows, [
        ("factor_online", "przy powstaniu kf (online)",
         {"marker": "o", "ms": 6.5, "mfc": SURFACE, "mec": ORANGE, "mew": 1.6}),
        ("factor_final", "na końcu (final)", {"marker": "o", "ms": 6.5, "color": BLUE, **ring}),
    ], f"{title} — współczynnik skali odcinka w klatkach kluczowych", "współczynnik [m / jedn. chmury]", p)
    out.append(p)

    p = run.path / "fig_c_factor_net_final.png"
    _panels(plt, segments, rows, [
        ("factor_final", "na końcu (final)", {"marker": "o", "ms": 6.5, "mfc": SURFACE, "mec": MUTED, "mew": 1.4}),
        ("factor_net_created", "sama sieć, przy powstaniu kf", {"marker": "D", "ms": 5.5, "color": VIOLET, **ring}),
    ], f"{title} — sama sieć (L / d_canon) na tle współczynnika końcowego", "współczynnik [m / jedn.]", p)
    out.append(p)

    # b) s_k(t): krzywa schodkowa każdej klatki kluczowej po kolejnych zdarzeniach backendu
    p = run.path / "fig_b_s_history.png"
    n = len(run.kfs)
    cmap, colors = kf_colors(n)
    t_end = max(run.frame_pts.values())
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for kd in run.kfs:
        k = kd["kf"]
        pts = sorted(((r["pts_s"], r["s"]) for r in s_hist if r["kf"] == k), key=lambda x: x[0])
        s_created = s_at(run.history, k, "created")
        xs = [kd["pts_s"]] + [t for t, _ in pts] + [t_end]
        ys = [s_created] + [s for _, s in pts] + [s_at(run.history, k, "final")]
        ax.plot(xs, ys, drawstyle="steps-post", color=colors[k], lw=1.3, zorder=2)
        ax.plot(kd["pts_s"], s_created, "o", ms=6, mfc=SURFACE, mec=colors[k], mew=1.4, zorder=3)
    ax.plot([], [], "o", ms=6, mfc=SURFACE, mec=INK2, mew=1.4, ls="none", label="s przy powstaniu (przed optymalizacją)")
    ax.plot([], [], color=INK2, lw=1.3, label="s po kolejnych optymalizacjach backendu")
    ax.legend(loc="best")
    ax.set_xlabel("czas nagrania [s] (PTS klatki, przy której zaszło zdarzenie)")
    ax.set_ylabel("skala Sim(3) klatki kluczowej  s_k")
    ax.set_title(f"{title} — s_k(t), kf 0 przypięta (s = 1)", loc="left", color=INK)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, max(n - 1, 1)))
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("indeks klatki kluczowej k")
    cb.locator = matplotlib.ticker.MaxNLocator(integer=True)
    cb.update_ticks()
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Porównanie z pomiarem ręcznym
# --------------------------------------------------------------------------- #


def comparison(results: dict[str, tuple[Run, list[dict]]], segments: list[Segment], summary_csv: Path) -> list[dict]:
    manual = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            manual[(r["nagranie"], r["segment"])] = r
    out = []
    for run, rows in results.values():
        for seg in segments:
            vals = [r["factor_final"] for r in rows if r["segment"] == seg.name and r.get("factor_final") is not None]
            m = manual.get((run.recording, seg.name), {})
            out.append({
                "nagranie": run.recording, "nazwa": run.name, "segment": seg.name,
                "n_kf": len(vals),
                "kf": " ".join(str(r["kf"]) for r in rows if r["segment"] == seg.name and r.get("factor_final") is not None),
                "factor_final_mediana": float(np.median(vals)) if vals else None,
                "factor_final_min": float(min(vals)) if vals else None,
                "factor_final_max": float(max(vals)) if vals else None,
                "factor_reczny": float(m["factor"]) if m.get("factor") else None,
                "sigma_factor_reczny": float(m["sigma_factor"]) if m.get("sigma_factor") else None,
                "powtarzalnosc_klikniec_pct": float(m["powtarzalnosc_pct"]) if m.get("powtarzalnosc_pct") else None,
            })
    return out


COMPARISON_COLUMNS = ("nagranie", "nazwa", "segment", "n_kf", "kf", "factor_final_mediana", "factor_final_min",
                      "factor_final_max", "factor_reczny", "sigma_factor_reczny", "powtarzalnosc_klikniec_pct")


def markdown_table(rows: list[dict]) -> str:
    def f(v, fmt="{:.4f}"):
        return "—" if v is None else fmt.format(v)

    lines = ["| nagranie | odcinek | n kf (kf) | factor_final mediana (min–max) | ręcznie (σ) |", "|---|---|---|---|---|"]
    for r in rows:
        rng = "" if r["n_kf"] < 2 else f" ({f(r['factor_final_min'])}–{f(r['factor_final_max'])})"
        lines.append(f"| {r['nazwa']} | {r['segment']} | {r['n_kf']} ({r['kf'] or '—'}) | "
                     f"{f(r['factor_final_mediana'])}{rng} | {f(r['factor_reczny'])} ({f(r['sigma_factor_reczny'])}) |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path, nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="scale_summary.csv (pomiar ręczny)")
    parser.add_argument("--c-min", type=float, default=C_MIN)
    parser.add_argument("--comparison-out", type=Path, default=ADAPTER / "porownanie_scale_summary.csv")
    args = parser.parse_args(argv)

    targets = load_targets(args.targets)
    marker_ids = {int(k) for k in targets.markers}
    segments, refs = segments_with_refs(targets)

    results = {}
    for path in args.runs:
        run = load_run(path, marker_ids)
        rows = segment_rows(run, segments, refs, args.c_min)
        write_csv(run.path / "scale_per_keyframe.csv", COLUMNS, rows)
        s_hist = s_history_rows(run)
        write_csv(run.path / "s_history.csv", S_HISTORY_COLUMNS, s_hist)
        centers = marker_centers_world(run, args.c_min)
        _, cols = kf_colors(len(run.kfs))
        rgb = {k: tuple(int(round(255 * v)) for v in c[:3]) for k, c in enumerate(cols)}
        write_markers_ply(run.path / "markers_world.ply", centers, rgb)
        figs = plot_run(run, segments, rows, s_hist)
        n_ok = sum(r["status"] == "ok" for r in rows)
        print(f"{run.name}: {len(run.kfs)} kf, {len(rows)} wierszy odcinków ({n_ok} ok), "
              f"{len(s_hist)} wierszy s_history, {len(centers)} środków markerów, {len(figs)} rysunki")
        for r in rows:
            if r["status"] != "ok":
                print(f"  kf {r['kf']} {r['segment']}: {r['status']}")
        results[run.name] = (run, rows)

    table = comparison(results, segments, args.summary)
    write_csv(args.comparison_out, COMPARISON_COLUMNS, table)
    print(f"\n{markdown_table(table)}\n\nzapisano {args.comparison_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
