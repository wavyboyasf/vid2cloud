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

Wyjście w każdym katalogu przebiegu: scale_per_keyframe.csv, s_history.csv,
fig_a_factor_online_final.png, fig_b_s_history.png, fig_c_factor_net_final.png,
markers_world.ply (środki markerów w świecie, T_WC_final · X, kolor = kf). Obok katalogów
przebiegów: porownanie_scale_summary.csv (mediana factor_final per odcinek obok pomiaru
ręcznego). Wejście: tylko przebiegi z single_thread = true. Env vid2cloud-tools.

    python eval/scale_per_keyframe.py            # trzy przebiegi z 03.10, targets z sesji 21.09
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vid2cloud.export.keyframes import read_keyframe_npz  # noqa: E402
from vid2cloud.export.trajectory import read_poses_history  # noqa: E402
from vid2cloud.scale.estimate import MM_PER_M, Segment, load_targets  # noqa: E402
from vid2cloud.scale.lift import (  # noqa: E402
    C_MIN,
    ReferencePoint,
    lift_marker,
    lift_reference,
    mean_confidence,
    parse_reference,
    to_world,
)

ADAPTER = REPO_ROOT / "runs" / "2026-09-21" / "adapter"
DEFAULT_RUNS = [ADAPTER / n for n in ("daleko_wolno", "blisko_z_gory", "daleko_szybko_pion")]
DEFAULT_TARGETS = REPO_ROOT / "data" / "2026-09-21" / "notes" / "targets.json"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "2026-09-21" / "scale_summary.csv"
STATES = ("created", "final")

COLUMNS = (
    "kf", "pts_s", "segment", "L_mm", "d_canon_created", "d_canon_final", "s_created", "s_final",
    "factor_net_created", "factor_online", "factor_final",
    "n_px_a_created", "n_px_b_created", "mad_a_created", "mad_b_created",
    "n_px_a_final", "n_px_b_final", "mad_a_final", "mad_b_final",
    "frame_index", "ref_a", "ref_b", "sigma_mm",
    "plane_rms_a_final", "plane_rms_b_final",
    "d_win3_created", "d_win3_final", "diff_win3_mm_created", "diff_win3_mm_final",
    "status",
)
S_HISTORY_COLUMNS = ("event_id", "event", "frame_index", "pts_s", "n_kf", "kf", "s")


# --------------------------------------------------------------------------- #
# Wczytanie przebiegu
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    path: Path
    name: str
    recording: str
    kfs: list[dict]  # {"kf", "frame_index", "pts_s", "created": dump, "final": dump}
    markers: dict[int, dict[int, np.ndarray]]  # kf -> {id: narożniki 512 (4, 2)}
    history: list[dict]
    frame_pts: dict[int, float] = field(default_factory=dict)  # indeks klatki -> pts_s (log.csv)


def load_run(path: Path, marker_ids: set[int]) -> Run:
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if cfg.get("engine", {}).get("single_thread") is not True:
        raise ValueError(f"{path}: single_thread != true — wejściem są tylko przebiegi deterministyczne")
    if cfg.get("status") != "done":
        raise ValueError(f"{path}: przebieg nie jest zakończony (status {cfg.get('status')!r})")

    kfs = []
    for created in sorted((path / "keyframes").glob("kf_*_created.npz")):
        k = int(created.name.split("_")[1])
        dumps = {st: read_keyframe_npz(path / "keyframes" / f"kf_{k:03d}_{st}.npz") for st in STATES}
        if dumps["created"]["frame_index"] != dumps["final"]["frame_index"]:
            raise ValueError(f"{path}: kf {k} — różne frame_index w created i final")
        kfs.append({"kf": k, "frame_index": dumps["created"]["frame_index"],
                    "pts_s": dumps["created"]["pts_s"], **dumps})
    if [d["kf"] for d in kfs] != list(range(len(kfs))):
        raise ValueError(f"{path}: luki w numeracji klatek kluczowych")

    markers: dict[int, dict[int, np.ndarray]] = {d["kf"]: {} for d in kfs}
    with open(path / "aruco_keyframes.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["id"] or int(row["id"]) not in marker_ids:
                continue  # brak wykrycia albo ID spoza targets.json (fałszywe wykrycia)
            q = np.array([float(row[f"{c}{i}_512"]) for i in range(4) for c in "xy"]).reshape(4, 2)
            markers[int(row["kf"])][int(row["id"])] = q

    frame_pts = {}
    with open(path / "log.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_pts[int(row["index"])] = float(row["pts_s"])
    recording = Path(cfg["video"]).stem
    return Run(path, path.name, recording, kfs, markers, read_poses_history(path / "poses_history.csv"), frame_pts)


def s_at(history: list[dict], kf: int, stage: str) -> float:
    """s klatki kf: przy powstaniu (wiersz kf_created/reloc z n_kf = kf + 1) albo z wiersza final."""
    if stage == "created":
        rows = [r for r in history if r["event"] in ("kf_created", "reloc") and r["kf"] == kf and r["n_kf"] == kf + 1]
    else:
        rows = [r for r in history if r["event"] == "final" and r["kf"] == kf]
    if len(rows) != 1:
        raise ValueError(f"poses_history: {len(rows)} wierszy {stage} dla kf {kf}, oczekiwano 1")
    return rows[0]["s"]


# --------------------------------------------------------------------------- #
# Liczenie
# --------------------------------------------------------------------------- #


def _f(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)


def segment_rows(run: Run, segments: list[Segment], refs: dict[str, tuple[ReferencePoint, ReferencePoint]],
                 c_min: float = C_MIN) -> list[dict]:
    rows = []
    for kd in run.kfs:
        k = kd["kf"]
        found = run.markers[k]
        s = {st: s_at(run.history, k, st) for st in STATES}
        for st in STATES:  # zgodność historii z dumpem — rozjazd to błąd w danych, nie w pomiarze
            if not np.isclose(s[st], kd[st]["T_WC"][7], rtol=1e-6):
                raise ValueError(f"{run.name} kf {k}: s {st} z poses_history {s[st]} != z dumpu {kd[st]['T_WC'][7]}")
        for seg in segments:
            if seg.marker_a not in found or seg.marker_b not in found:
                continue
            ra, rb = refs[seg.name]
            row: dict = {
                "kf": k, "pts_s": kd["pts_s"], "segment": seg.name, "L_mm": seg.length_mm,
                "frame_index": kd["frame_index"], "ref_a": ra.used, "ref_b": rb.used, "sigma_mm": seg.sigma_mm,
                "s_created": s["created"], "s_final": s["final"],
            }
            reasons = []
            for st in STATES:
                dump = kd[st]
                try:
                    C = mean_confidence(dump["C"], dump["N"])
                except ValueError as exc:
                    reasons.append(f"{st}: {exc}")
                    continue
                la = lift_reference(ra, found[ra.marker_id], dump["X_canon"], C, c_min)
                lb = lift_reference(rb, found[rb.marker_id], dump["X_canon"], C, c_min)
                row.update({f"n_px_a_{st}": la.n_px, f"n_px_b_{st}": lb.n_px,
                            f"mad_a_{st}": _f(la.mad), f"mad_b_{st}": _f(lb.mad)})
                if st == "final":
                    row.update({"plane_rms_a_final": _f(la.plane_rms), "plane_rms_b_final": _f(lb.plane_rms)})
                if la.point is None or lb.point is None:
                    reasons.append(f"{st}: " + "; ".join(f"{r.ref.marker_id}: {r.reason}" for r in (la, lb) if r.point is None))
                    continue
                d = float(np.linalg.norm(la.point - lb.point))
                row[f"d_canon_{st}"] = d
                if la.point_win3 is not None and lb.point_win3 is not None:
                    d_win = float(np.linalg.norm(la.point_win3 - lb.point_win3))
                    row[f"d_win3_{st}"] = d_win
                    row[f"diff_win3_mm_{st}"] = seg.length_mm * (d_win / d - 1.0)
            L = seg.length_mm / MM_PER_M
            if row.get("d_canon_created"):
                row["factor_net_created"] = L / row["d_canon_created"]
                row["factor_online"] = L / (s["created"] * row["d_canon_created"])
            if row.get("d_canon_final"):
                row["factor_final"] = L / (s["final"] * row["d_canon_final"])
            row["status"] = "ok" if not reasons else " | ".join(reasons)
            rows.append(row)
    return rows


def s_history_rows(run: Run) -> list[dict]:
    """s_k po każdym zdarzeniu backendu (backend_done; reloc to też optymalizacja)."""
    pts = {kd["frame_index"]: kd["pts_s"] for kd in run.kfs}
    out = []
    for r in run.history:
        if r["event"] not in ("backend_done", "reloc"):
            continue
        out.append({"event_id": r["event_id"], "event": r["event"], "frame_index": r["frame_index"],
                    "pts_s": pts.get(r["frame_index"], run.frame_pts.get(r["frame_index"])),
                    "n_kf": r["n_kf"], "kf": r["kf"], "s": r["s"]})
    return out


def marker_centers_world(run: Run, c_min: float = C_MIN) -> list[dict]:
    """Środki markerów (mediana, stan final) w układzie świata: T_WC_final · X."""
    out = []
    for kd in run.kfs:
        dump = kd["final"]
        if int(dump["N"]) <= 0:
            continue
        C = mean_confidence(dump["C"], dump["N"])
        for mid, quad in sorted(run.markers[kd["kf"]].items()):
            res = lift_marker(quad, dump["X_canon"], C, c_min)
            if res.point is None:
                continue
            out.append({"kf": kd["kf"], "marker_id": mid, "xyz": to_world(dump["T_WC"], res.point)})
    return out


# --------------------------------------------------------------------------- #
# Zapis
# --------------------------------------------------------------------------- #


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.9g}"
    return str(v)


def write_csv(path: Path, columns, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in columns])


def write_markers_ply(path: Path, centers: list[dict], colors: dict[int, tuple[int, int, int]]) -> None:
    """Binarny PLY: x y z, red green blue, kf, marker_id (dwa ostatnie jako pola skalarne w CC)."""
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("kf", "<i4"), ("marker_id", "<i4")])
    data = np.zeros(len(centers), dtype=dt)
    for i, c in enumerate(centers):
        data[i] = (*c["xyz"], *colors[c["kf"]], c["kf"], c["marker_id"])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment vid2cloud: srodki markerow ArUco, T_WC_final * X (lift_marker, mediana), kolor = kf\n"
        f"element vertex {len(data)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int kf\nproperty int marker_id\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


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
    segments = [s for s in targets.segments.values() if s.marker_a is not None and s.marker_b is not None]
    # Każdy reference parsowany z góry: nieobsługiwany to błąd przed liczeniem, nie pominięcie.
    refs = {s.name: parse_reference(s.reference, s.marker_a, s.marker_b) for s in segments}

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
