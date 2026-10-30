"""Przebieg wstępny po pliku (bez GPU): miary jakości każdej klatki, wykres, listy klatek.

    measure_video()  — FileSource sekwencyjnie -> wiersze `quality.csv` (+ keep z `thresholds()`)
    gap_stats()      — odstępy PTS: mediana i odstępy > 1,5 × mediana (klatki pominięte w zapisie)
    spearman()       — korelacja rang Spearmana (rangi średnie przy remisach)
    plot_sharpness() — `fig_ostrosc.png`
    run_quality()    — komenda `vid2cloud quality <mp4> --out <katalog>`
    run_select()     — komenda `vid2cloud select <quality.csv> --k 3 --out <katalog>`

Miary liczone są na obrazie wejścia sieci (`selection.model_input_gray()`), po rotacji
z `FileSource`. Env `vid2cloud-tools`.

Kolumny `quality.csv`:
    frame          indeks klatki w kolejności dekodowania (ten sam co `index` w log.csv przebiegu)
    pts_s          PTS [s], bez przesuwania do zera (jak log.csv)
    dt_ms          pts_s − pts_s poprzedniej klatki [ms]; puste dla klatki 0
    tenengrad      `selection.tenengrad()`
    lapvar         `selection.laplacian_var()`
    bright_median, frac_dark, frac_bright  — `selection.exposure()`
    keep           1/0 z `selection.thresholds()` z progami domyślnymi (THR_*)
    t_measure_ms   czas: skalowanie do 512 + szarość + trzy miary (bez dekodowania i rotacji)

Pliki z `run_select()` (w `--out`): `<wariant>.txt` (jeden indeks w linii), `manifest.csv`
(wariant, k, offset, n_klatek, n_okien_pominietych — puste pole = nie dotyczy: offset nie
dotyczy wariantów okienkowych, okna nie dotyczą stałego kroku), `<wariant>_pominiete_okna.csv`
dla wariantów okienkowych, `select_report.json`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from vid2cloud.input import selection as sel
from vid2cloud.input.sources import FileSource

QUALITY_COLUMNS = (
    "frame", "pts_s", "dt_ms", "tenengrad", "lapvar",
    "bright_median", "frac_dark", "frac_bright", "keep", "t_measure_ms",
)
#: Odstęp PTS większy niż GAP_FACTOR × mediana liczony jako luka (pominięte klatki w zapisie).
GAP_FACTOR = 1.5
#: k dla pionowych kresek na wykresie (select_in_windows bez progów).
FIG_K = 3
MANIFEST_COLUMNS = ("wariant", "k", "offset", "n_klatek", "n_okien_pominietych")
SKIPPED_COLUMNS = ("window", "start", "stop", "reason")

_FMT = {
    "frame": "{:d}", "pts_s": "{:.6f}", "dt_ms": "{:.4f}", "tenengrad": "{:.9g}", "lapvar": "{:.9g}",
    "bright_median": "{:.1f}", "frac_dark": "{:.9g}", "frac_bright": "{:.9g}", "t_measure_ms": "{:.3f}",
}


# --------------------------------------------------------------------------- #
# Pomiar
# --------------------------------------------------------------------------- #


def measure_frame(rgb: np.ndarray) -> tuple[dict, tuple[int, int]]:
    """Miary jednej klatki RGB w pełnej rozdzielczości. Zwraca (wiersz, kształt obrazu sieci)."""
    t0 = time.perf_counter()
    gray = sel.model_input_gray(rgb)
    row = {"tenengrad": sel.tenengrad(gray), "lapvar": sel.laplacian_var(gray), **sel.exposure(gray)}
    row["t_measure_ms"] = 1000.0 * (time.perf_counter() - t0)
    return row, gray.shape


def columns(rows: Sequence[dict]) -> dict[str, np.ndarray]:
    """Wiersze -> słownik tablic (dt_ms: NaN dla klatki bez poprzedniczki)."""
    out = {}
    for c in QUALITY_COLUMNS:
        if c == "keep" and not all(c in r for r in rows):
            continue
        vals = [r.get(c) for r in rows]
        if c == "frame":
            out[c] = np.array(vals, dtype=np.int64)
        elif c == "keep":
            out[c] = np.array(vals, dtype=bool)
        else:
            out[c] = np.array([np.nan if v is None else v for v in vals], dtype=np.float64)
    return out


def measure_video(video: Path, progress_every: int = 200) -> tuple[list[dict], dict]:
    """Wszystkie klatki pliku, sekwencyjnie. Zwraca (wiersze quality.csv, informacje o przebiegu)."""
    source = FileSource(video)
    rows: list[dict] = []
    prev_pts: Optional[float] = None
    shape: Optional[tuple[int, int]] = None
    full_shape: Optional[tuple[int, int]] = None
    t_decode = 0.0
    t0 = time.perf_counter()
    it = iter(source)
    while True:
        t1 = time.perf_counter()
        try:
            index, pts_s, rgb = next(it)
        except StopIteration:
            break
        t_decode += time.perf_counter() - t1
        m, shape = measure_frame(rgb)
        full_shape = full_shape or tuple(rgb.shape[:2])
        rows.append({
            "frame": index,
            "pts_s": pts_s,
            "dt_ms": None if prev_pts is None else 1000.0 * (pts_s - prev_pts),
            **m,
        })
        prev_pts = pts_s
        if progress_every and len(rows) % progress_every == 0:
            el = time.perf_counter() - t0
            print(f"  klatka {len(rows)}  {len(rows) / el:.1f} kl/s", flush=True)
    if not rows:
        raise ValueError(f"{video}: nie zdekodowano żadnej klatki")
    keep = sel.thresholds(columns(rows))
    for r, k in zip(rows, keep):
        r["keep"] = bool(k)
    info = {
        "n_frames": len(rows),
        "rotation_applied": source.rotation,
        "full_shape_after_rotation": list(full_shape),
        "model_input_shape": list(shape),
        "decode_total_s": t_decode,
        "wall_s": time.perf_counter() - t0,
    }
    return rows, info


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _cell(col: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if col == "keep":
        return str(int(bool(value)))
    return _FMT[col].format(value)


def write_quality_csv(path: Path, rows: Sequence[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(QUALITY_COLUMNS)
        for r in rows:
            w.writerow([_cell(c, r.get(c)) for c in QUALITY_COLUMNS])


def read_quality_csv(path: Path) -> dict[str, np.ndarray]:
    """`quality.csv` -> słownik tablic (frame int, keep bool, reszta float; puste -> NaN)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(QUALITY_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: brak kolumn {sorted(missing)}")
        raw = list(reader)
    out = {}
    for c in QUALITY_COLUMNS:
        vals = [r[c] for r in raw]
        if c == "frame":
            out[c] = np.array([int(v) for v in vals], dtype=np.int64)
        elif c == "keep":
            out[c] = np.array([v == "1" for v in vals], dtype=bool)
        else:
            out[c] = np.array([float(v) if v != "" else np.nan for v in vals], dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Statystyki do raportu
# --------------------------------------------------------------------------- #


def gap_stats(pts_s: Sequence[float], factor: float = GAP_FACTOR) -> dict:
    """Odstępy PTS [ms]: mediana, min, max i odstępy > factor × mediana.

    `gap_frames` — indeksy klatek, PRZED którymi wypada luka (klatka i ma dt_i > próg).
    `gap_max_ms` — największy z tych odstępów; None, gdy luk nie ma.
    """
    pts = np.asarray(pts_s, dtype=np.float64)
    if pts.size < 2:
        raise ValueError("potrzeba co najmniej dwóch klatek")
    dt = np.diff(pts) * 1000.0
    med = float(np.median(dt))
    thr = factor * med
    gap_idx = np.flatnonzero(dt > thr) + 1
    return {
        "dt_median_ms": med,
        "dt_min_ms": float(dt.min()),
        "dt_max_ms": float(dt.max()),
        "gap_factor": factor,
        "gap_threshold_ms": thr,
        "n_gaps": int(gap_idx.size),
        "gap_max_ms": float(dt[gap_idx - 1].max()) if gap_idx.size else None,
        "gap_frames": [int(i) for i in gap_idx],
    }


def _rank(x: np.ndarray) -> np.ndarray:
    """Rangi 1..n, przy remisach średnia rang (jak `scipy.stats.rankdata(method='average')`)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Współczynnik korelacji rang Spearmana (Pearson na rangach średnich)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError(f"potrzeba dwóch wektorów 1-D tej samej długości >= 2, dostałem {a.shape} i {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("wartości nieskończone/NaN")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def false_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Ciągłe odcinki z mask == False jako (start, stop) — stop wyłącznie."""
    m = np.asarray(mask, dtype=bool)
    bad = np.r_[False, ~m, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(bad))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


# --------------------------------------------------------------------------- #
# Wykres
# --------------------------------------------------------------------------- #

#: Kolory: dwie serie (kategoryczne 1 i 2), tło odrzuceń, kreski wybranych klatek, siatka.
COLOR_TEN = "#2a78d6"
COLOR_LAP = "#eb6834"
COLOR_REJECT = "#dcdcd8"
COLOR_TICK = "#52514e"
COLOR_GRID = "#e9e9e6"


def plot_sharpness(q: dict[str, np.ndarray], selected: Sequence[int], keep: Sequence[bool], path: Path) -> None:
    """Tenengrad i lapvar / mediana nagrania w funkcji czasu; szare tło = odrzucone przez progi;
    pod spodem pasek z pionowymi kreskami klatek wybranych. Bez tytułu, PNG 150 dpi."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pts = q["pts_s"]
    ten = q["tenengrad"] / np.median(q["tenengrad"])
    lap = q["lapvar"] / np.median(q["lapvar"])
    half = 0.5 * float(np.median(np.diff(pts))) if pts.size > 1 else 0.0

    fig, (ax, ax_sel) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 4.2), dpi=150,
        gridspec_kw={"height_ratios": [10, 1], "hspace": 0.06},
    )
    for a in (ax, ax_sel):
        for start, stop in false_runs(keep):
            a.axvspan(pts[start] - half, pts[stop - 1] + half, color=COLOR_REJECT, lw=0, zorder=0)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax.plot(pts, ten, color=COLOR_TEN, lw=0.9, label="Tenengrad", zorder=3)
    ax.plot(pts, lap, color=COLOR_LAP, lw=0.9, label="wariancja Laplasjanu", zorder=2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("miara / mediana nagrania")
    ax.grid(axis="y", color=COLOR_GRID, lw=0.6, zorder=1)
    ax.tick_params(axis="x", which="both", bottom=False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=COLOR_REJECT, edgecolor="none"))
    labels.append("odrzucone przez progi")
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=8, ncol=3)

    ax_sel.vlines(pts[np.asarray(selected, dtype=int)], 0, 1, color=COLOR_TICK, lw=0.35)
    ax_sel.set_ylim(0, 1)
    ax_sel.set_yticks([])
    ax_sel.set_ylabel(f"k = {FIG_K}", rotation=0, ha="right", va="center", fontsize=8)
    ax_sel.spines["left"].set_visible(False)
    ax_sel.set_xlim(pts[0] - half, pts[-1] + half)
    ax_sel.set_xlabel("czas [s]")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Komendy
# --------------------------------------------------------------------------- #


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _code_state() -> dict:
    from vid2cloud.run import REPO_ROOT, _git_state  # bez torcha: run.py importuje silnik leniwie

    return _git_state(REPO_ROOT)


def _refuse_overwrite(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"już istnieje: {', '.join(existing)} — nie nadpisuję (--overwrite)")


def run_quality(video: Path, out: Path, overwrite: bool = False) -> dict:
    """quality.csv, fig_ostrosc.png i quality_report.json w katalogu `out`. Zwraca raport."""
    video, out = Path(video).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = {n: out / n for n in ("quality.csv", "fig_ostrosc.png", "quality_report.json")}
    _refuse_overwrite(list(files.values()), overwrite)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    rows, info = measure_video(video)
    write_quality_csv(files["quality.csv"], rows)
    q = columns(rows)
    sharp_ok, bright_ok, _ = sel.threshold_parts(q)
    keep = q["keep"]
    selected, sel_report = sel.select_in_windows(q["tenengrad"], FIG_K)
    plot_sharpness(q, selected, keep, files["fig_ostrosc.png"])

    report = {
        "video": str(video),
        "out": str(out),
        "started_at": started_at,
        "vid2cloud": _code_state(),
        **info,
        **gap_stats(q["pts_s"]),
        "spearman_tenengrad_lapvar": spearman(q["tenengrad"], q["lapvar"]),
        "thresholds": {"rel": sel.THR_REL, "win_s": sel.THR_WIN_S,
                       "med_lo": sel.THR_MED_LO, "med_hi": sel.THR_MED_HI},
        "n_keep": int(keep.sum()),
        "n_reject": int((~keep).sum()),
        "n_reject_sharpness": int((~sharp_ok).sum()),
        "n_reject_brightness": int((~bright_ok).sum()),
        "n_reject_both": int((~sharp_ok & ~bright_ok).sum()),
        "t_measure_ms_median": float(np.median(q["t_measure_ms"])),
        "fig": {"k": FIG_K, "n_selected": sel_report["n_selected"], "file": str(files["fig_ostrosc.png"])},
        "files": {k: str(v) for k, v in files.items()},
    }
    _write_json(files["quality_report.json"], report)
    return report


def run_select(quality_csv: Path, k: int, out: Optional[Path] = None, overwrite: bool = False) -> dict:
    """Listy klatek (stały krok × k przesunięć, argmax w oknach bez i z progami) + manifest."""
    quality_csv = Path(quality_csv).resolve()
    out = Path(out).resolve() if out else quality_csv.parent / "frames"
    q = read_quality_csv(quality_csv)
    n = int(q["frame"].size)
    if not np.array_equal(q["frame"], np.arange(n)):
        raise ValueError(f"{quality_csv}: kolumna frame nie jest ciągła 0..{n - 1} — listy byłyby przesunięte")

    variants: list[tuple[str, Optional[int], list[int], Optional[dict]]] = []
    for o in range(k):
        variants.append((f"fixed_k{k}_o{o}", o, sel.fixed_step(n, k, o), None))
    idx, rep = sel.select_in_windows(q["tenengrad"], k)
    variants.append((f"sharp_k{k}", None, idx, rep))
    idx, rep = sel.select_in_windows(q["tenengrad"], k, q["keep"])
    variants.append((f"sharp_thr_k{k}", None, idx, rep))

    paths = [out / f"{name}.txt" for name, *_ in variants] + [out / "manifest.csv", out / "select_report.json"]
    paths += [out / f"{name}_pominiete_okna.csv" for name, _, _, rep in variants if rep is not None]
    out.mkdir(parents=True, exist_ok=True)
    _refuse_overwrite(paths, overwrite)

    manifest = []
    for name, offset, indices, rep in variants:
        (out / f"{name}.txt").write_text(sel.format_frame_list(indices), encoding="utf-8")
        manifest.append({
            "wariant": name, "k": k,
            "offset": "" if offset is None else offset,
            "n_klatek": len(indices),
            "n_okien_pominietych": "" if rep is None else len(rep["skipped"]),
        })
        if rep is not None:
            with open(out / f"{name}_pominiete_okna.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=SKIPPED_COLUMNS)
                w.writeheader()
                w.writerows(rep["skipped"])
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(manifest)

    lists = {name: indices for name, _, indices, _ in variants}
    common = len(set(lists[f"sharp_k{k}"]) & set(lists[f"fixed_k{k}_o0"]))
    report = {
        "quality_csv": str(quality_csv),
        "quality_csv_sha256": hashlib.sha256(quality_csv.read_bytes()).hexdigest(),
        "out": str(out),
        "k": k,
        "n_frames": n,
        "manifest": manifest,
        f"common_sharp_k{k}_fixed_k{k}_o0": common,
        "vid2cloud": _code_state(),
    }
    _write_json(out / "select_report.json", report)
    return report


def format_quality_report(r: dict) -> str:
    gap_max = "brak" if r["gap_max_ms"] is None else f"{r['gap_max_ms']:.2f} ms"
    return "\n".join([
        f"quality: {r['video']}",
        f"  klatek {r['n_frames']}, rotacja {r['rotation_applied']}, obraz sieci {r['model_input_shape']}",
        f"  dt: mediana {r['dt_median_ms']:.3f} ms (min {r['dt_min_ms']:.3f}, max {r['dt_max_ms']:.3f}); "
        f"odstępów > {r['gap_factor']} × mediana: {r['n_gaps']}, największy {gap_max}",
        f"  Spearman tenengrad ↔ lapvar: {r['spearman_tenengrad_lapvar']:.4f}",
        f"  keep: {r['n_keep']} / odrzucone {r['n_reject']} (ostrość {r['n_reject_sharpness']}, "
        f"jasność {r['n_reject_brightness']}, oba {r['n_reject_both']})",
        f"  t_measure mediana {r['t_measure_ms_median']:.1f} ms, całość {r['wall_s']:.0f} s -> {r['out']}",
    ])


def format_select_report(r: dict) -> str:
    k = r["k"]
    lines = [f"select: {r['quality_csv']} (k = {k}, {r['n_frames']} klatek) -> {r['out']}"]
    for m in r["manifest"]:
        skipped = "" if m["n_okien_pominietych"] == "" else f", okien pominiętych {m['n_okien_pominietych']}"
        lines.append(f"  {m['wariant']:<16} {m['n_klatek']:>5} klatek{skipped}")
    lines.append(f"  wspólnych sharp_k{k} i fixed_k{k}_o0: {r[f'common_sharp_k{k}_fixed_k{k}_o0']}")
    return "\n".join(lines)
