"""Referencja TLS z sesji 21.09: inwentaryzacja archiwów, scalenie skanów sali w CloudCompare.

Archiwa `data/2026-09-21/reference_lidar/ZamkeKrolewski-*.zip` zawierają pliki .e57 z trzech
sal (Koncertowa, Tronowa, Senacka), po jednym stanowisku na plik. Skrypt:

    inventory   lista wszystkich .e57 ze wszystkich archiwów -> eval_cc/tls_inventory.csv
    build       wypakowanie plików sali do reference_lidar/<sala>/ (bez kasowania zipów),
                CloudCompare: -O każdego skanu, -MERGE_CLOUDS, -SS SPATIAL <krok> -> PLY,
                raport (liczba stanowisk, punktów przed/po, prostopadłościan otaczający)
    dims        jak build, ale pliki są wypakowywane do katalogu tymczasowego poza repo
                i usuwane po pomiarze (sala tylko do inwentaryzacji i wymiarów)

    python eval/tls_reference.py inventory
    python eval/tls_reference.py build --room Koncertowa
    python eval/tls_reference.py dims --room Tronowa

Liczba punktów „przed” to suma `recordCount` z nagłówków XML plików E57 oraz liczba punktów
wczytanych przez CloudCompare (CC pomija punkty z cartesianInvalidState != 0). Wymiary to
prostopadłościan otaczający w osiach układu TLS liczony z chmury po podpróbkowaniu
(podpróbkowanie przestrzenne wybiera istniejące punkty, więc bbox jest najwyżej o krok węższy).
Env vid2cloud-tools; parametry w eval/eval_config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from plyio import read_ply  # noqa: E402

LIDAR = REPO_ROOT / "data" / "2026-09-21" / "reference_lidar"
EVAL_CC = REPO_ROOT / "runs" / "2026-09-21" / "eval_cc"
CONFIG = REPO_ROOT / "eval" / "eval_config.json"
CC_SH = REPO_ROOT / "eval" / "cc.sh"
E57_NS = "{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}"


# --------------------------------------------------------------------------- #
# E57: nagłówek i sekcja XML (bez dekodowania punktów)
# --------------------------------------------------------------------------- #


def e57_logical_to_physical(offset: int, page: int = 1024) -> int:
    """Strony E57 mają 4 B CRC na końcu: logiczne bajty to page − 4 z każdej strony."""
    data = page - 4
    return (offset // data) * page + offset % data


def read_e57_xml(fileobj) -> str:
    """Sekcja XML pliku E57 (ASTM E2807): nagłówek 48 B, potem strony z CRC."""
    fileobj.seek(0)
    sig, _major, _minor, _plen, xml_off, xml_len, page = struct.unpack("<8sIIQQQQ", fileobj.read(48))
    if sig != b"ASTM-E57":
        raise ValueError(f"to nie jest plik E57 (sygnatura {sig!r})")
    out = bytearray()
    phys = xml_off
    while len(out) < xml_len:
        start = phys // page * page
        fileobj.seek(start)
        chunk = fileobj.read(page)[: page - 4]
        out += chunk[phys - start:]
        phys = start + page
    return bytes(out[:xml_len]).decode("utf-8")


def e57_scans(xml: str) -> list[dict]:
    """Per data3D: nazwa, recordCount, poza (kwaternion w-x-y-z, translacja), cartesianBounds."""
    root = ET.fromstring(xml)
    out = []
    for child in root.find(f"{E57_NS}data3D"):
        def num(path, default=0.0):
            el = child.find(path)
            return float(el.text) if el is not None and el.text else default  # pusty element = 0 w E57

        pts = child.find(f"{E57_NS}points")
        rot = [num(f"{E57_NS}pose/{E57_NS}rotation/{E57_NS}{k}") for k in "wxyz"]
        tr = [num(f"{E57_NS}pose/{E57_NS}translation/{E57_NS}{k}") for k in "xyz"]
        b = {k: num(f"{E57_NS}cartesianBounds/{E57_NS}{k}") for k in
             ("xMinimum", "xMaximum", "yMinimum", "yMaximum", "zMinimum", "zMaximum")}
        name = child.find(f"{E57_NS}name")
        out.append({"name": name.text if name is not None else "", "points": int(pts.get("recordCount")),
                    "rotation_wxyz": rot, "translation": tr, "cartesian_bounds_local": b})
    return out


# --------------------------------------------------------------------------- #
# Inwentaryzacja i wypakowanie
# --------------------------------------------------------------------------- #


def inventory(lidar_dir: Path = LIDAR) -> list[dict]:
    rows = []
    for z in sorted(lidar_dir.glob("ZamkeKrolewski-*.zip")):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".e57"):
                    parts = Path(info.filename).parts
                    rows.append({"archiwum": z.name, "plik": info.filename, "rozmiar": info.file_size,
                                 "rozmiar_skompresowany": info.compress_size,
                                 "katalog": parts[-2] if len(parts) > 1 else ""})
    return rows


def room_files(rows: list[dict], room: str) -> list[dict]:
    """Pliki .e57 sali: `room` w nazwie pliku (np. Koncertowa, Tronowa)."""
    return sorted((r for r in rows if room.lower() in Path(r["plik"]).name.lower()), key=lambda r: r["plik"])


def extract(files: list[dict], dest: Path, lidar_dir: Path = LIDAR) -> list[Path]:
    """Wypakowuje pliki do `dest` (bez ścieżki w archiwum); istniejący plik o tym rozmiarze zostaje."""
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for r in files:
        target = dest / Path(r["plik"]).name
        if not (target.exists() and target.stat().st_size == r["rozmiar"]):
            with zipfile.ZipFile(lidar_dir / r["archiwum"]) as zf, zf.open(r["plik"]) as src, \
                    open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
        out.append(target)
    return out


# --------------------------------------------------------------------------- #
# CloudCompare
# --------------------------------------------------------------------------- #


def cc_merge_subsample_args(scans: list[Path], step_m: float, out_ply: Path) -> list[str]:
    args = ["-AUTO_SAVE", "OFF", "-NO_TIMESTAMP", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE"]
    for p in scans:
        args += ["-O", str(p)]
    args += ["-MERGE_CLOUDS", "-SS", "SPATIAL", f"{step_m:g}", "-SAVE_CLOUDS", "FILE", str(out_ply)]
    return args


def run_cc(args: list[str], log_path: Path, commands_path: Path) -> str:
    """Uruchamia eval/cc.sh, dopisuje dokładną komendę do `commands_path`, zwraca log CC."""
    cmd = ["bash", str(CC_SH), *args]
    with open(commands_path, "a", encoding="utf-8") as f:
        f.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n" + " ".join(_q(a) for a in cmd) + "\n")
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 or "Processed finished" not in proc.stdout:
        raise RuntimeError(f"CloudCompare: kod {proc.returncode}, log {log_path}")
    print(f"  CloudCompare {time.time() - t0:.0f} s, log {log_path}")
    return proc.stdout


def _q(a: str) -> str:
    return f"'{a}'" if re.search(r"[\s'\"]", a) else a


def cc_loaded_counts(log: str) -> list[int]:
    """Liczby punktów wczytanych przez CC — wiersze 'Found one cloud with N points'/'[E57] ... points'."""
    return [int(m) for m in re.findall(r"Found one cloud with (\d+) points", log)]


def bbox(points: np.ndarray, percentiles=(1, 99)) -> dict:
    lo, hi = points.min(axis=0), points.max(axis=0)
    plo, phi = np.percentile(points, percentiles, axis=0)
    return {"min": lo.tolist(), "max": hi.tolist(), "size_m": (hi - lo).tolist(),
            f"p{percentiles[0]}": plo.tolist(), f"p{percentiles[1]}": phi.tolist(),
            f"size_p{percentiles[0]}_p{percentiles[1]}_m": (phi - plo).tolist()}


def build(room: str, step_m: float, scans: list[Path], out_ply: Path, xml_info: list[dict], files: list[dict]) -> dict:
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    stem = out_ply.with_suffix("")
    log = run_cc(cc_merge_subsample_args(scans, step_m, out_ply), Path(f"{stem}.cc.log"),
                 out_ply.parent / "cc_commands.txt")
    loaded = cc_loaded_counts(log)
    m = re.findall(r"Result: (\d+) points", log)
    ply = read_ply(out_ply)
    xyz = xyz_of(ply)
    report = {
        "room": room,
        "files": [{"archiwum": r["archiwum"], "plik": r["plik"], "rozmiar": r["rozmiar"]} for r in files],
        "n_scans_files": len(scans),
        "n_scans_data3d": sum(len(x["scans"]) for x in xml_info),
        "points_e57_recordcount": sum(s["points"] for x in xml_info for s in x["scans"]),
        "points_cc_loaded_per_scan": loaded,
        "points_cc_loaded": int(sum(loaded)) if loaded else None,
        "subsample_spatial_m": step_m,
        "points_after_subsample_cc_log": int(m[-1]) if m else None,
        "points_after_subsample_ply": int(len(xyz)),
        "bbox_tls_axes": bbox(xyz),
        "wall_aligned_extra": wall_aligned(xyz[:: max(1, len(xyz) // 2_000_000)]),
        "ply": str(out_ply.relative_to(REPO_ROOT)) if out_ply.is_relative_to(REPO_ROOT) else str(out_ply),
        "ply_properties": list(ply.dtype.names),
        "scans_e57": xml_info,
    }
    return report


def wall_aligned(points: np.ndarray, percentiles=(1, 99), step_deg: float = 0.1) -> dict:
    """Wymiary w osiach obróconych wokół Z: kąt θ ∈ [0°, 90°) minimalizujący pole prostokąta
    percentyli 1–99 w XY (przegląd co `step_deg`). Dodatek do bbox w osiach TLS — sala nie musi
    leżeć wzdłuż osi skanera; percentyle, bo przez okna i drzwi skaner widzi punkty poza salą."""
    P = np.asarray(points, dtype=np.float64)
    best = None
    for th in np.arange(0.0, 90.0, step_deg):
        c, s_ = np.cos(np.radians(th)), np.sin(np.radians(th))
        u = P[:, 0] * c + P[:, 1] * s_
        v = -P[:, 0] * s_ + P[:, 1] * c
        lo, hi = np.percentile(np.column_stack([u, v]), percentiles, axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, th, lo, hi, u, v)
    _, th, lo, hi, u, v = best
    zlo, zhi = np.percentile(P[:, 2], percentiles)
    return {"angle_deg": float(th), "step_deg": step_deg, "percentiles": list(percentiles),
            "size_p_m": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(zhi - zlo)],
            "size_minmax_m": [float(u.max() - u.min()), float(v.max() - v.min()), float(P[:, 2].max() - P[:, 2].min())]}


def top_view(ply_path: Path, png: Path, cell_m: float = 0.05) -> None:
    """Widok z góry: liczba punktów w komórkach XY (skala log), bbox i percentyle 1–99 — diagnostyka."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle

    P = xyz_of(read_ply(ply_path))
    b = bbox(P)
    nx, ny = (int(np.ceil((b["max"][i] - b["min"][i]) / cell_m)) for i in (0, 1))
    H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=(nx, ny),
                               range=((b["min"][0], b["max"][0]), (b["min"][1], b["max"][1])))
    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    im = ax.pcolormesh(xe, ye, np.ma.masked_equal(H.T, 0), cmap=ramp, norm=LogNorm(), rasterized=True)
    ax.add_patch(Rectangle((b["min"][0], b["min"][1]), b["size_m"][0], b["size_m"][1], fill=False,
                           ec="#52514e", lw=1.0, label="prostopadłościan otaczający"))
    ax.add_patch(Rectangle((b["p1"][0], b["p1"][1]), b["size_p1_p99_m"][0], b["size_p1_p99_m"][1], fill=False,
                           ec="#eb6834", lw=1.2, ls="--", label="percentyle 1–99"))
    ax.set_aspect("equal")
    ax.set_xlabel("X TLS [m]")
    ax.set_ylabel("Y TLS [m]")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(f"punkty w komórce {cell_m * 100:.0f} × {cell_m * 100:.0f} cm")
    cb.outline.set_visible(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def xyz_of(v) -> np.ndarray:
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)


def xml_for(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append({"file": p.name, "scans": e57_scans(read_e57_xml(f))})
    return out


def print_report(r: dict) -> None:
    b = r["bbox_tls_axes"]
    print(f"{r['room']}: {r['n_scans_files']} plików, {r['n_scans_data3d']} stanowisk (data3D)")
    print(f"  punkty: E57 recordCount {r['points_e57_recordcount']:,}, wczytane przez CC {r['points_cc_loaded']}, "
          f"po SS {r['subsample_spatial_m']} m: {r['points_after_subsample_ply']:,}")
    print("  bbox [m]: " + ", ".join(f"{a} {lo:.3f}..{hi:.3f} ({s:.3f})"
                                     for a, lo, hi, s in zip("XYZ", b["min"], b["max"], b["size_m"])))
    print("  p1–p99 [m]: " + ", ".join(f"{a} {s:.3f}" for a, s in zip("XYZ", b["size_p1_p99_m"])))
    w = r.get("wall_aligned_extra")
    if w:
        print(f"  dodatek, osie obrócone o {w['angle_deg']:.1f}° wokół Z: p1–p99 "
              + " × ".join(f"{v:.3f}" for v in w["size_p_m"]) + " m; min–max "
              + " × ".join(f"{v:.3f}" for v in w["size_minmax_m"]) + " m")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("inventory", "build", "dims", "top"))
    ap.add_argument("--room", default="Koncertowa")
    ap.add_argument("--out", type=Path, help="PLY wynikowy (build); domyślnie eval_cc/tls/tls_<sala>_1cm.ply")
    args = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    step = float(cfg["tls_subsample_m"])
    EVAL_CC.mkdir(parents=True, exist_ok=True)

    rows = inventory()
    inv = EVAL_CC / "tls_inventory.csv"
    with open(inv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["archiwum", "plik", "rozmiar", "rozmiar_skompresowany", "katalog"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} plików .e57 w {len({r['archiwum'] for r in rows})} archiwach -> {inv}")
    if args.action == "inventory":
        return 0

    files = room_files(rows, args.room)
    if not files:
        print(f"brak plików z '{args.room}' w nazwie")
        return 1
    tag = f"{args.room.lower()}_{round(step * 100):d}cm"
    if args.action == "top":
        ply = args.out or EVAL_CC / "tls" / f"tls_{tag}.ply"
        top_view(ply, ply.with_name(ply.stem + "_top.png"))
        print(f"-> {ply.with_name(ply.stem + '_top.png')}")
        return 0
    if args.action == "build":
        scans = extract(files, LIDAR / args.room.lower())
        out_ply = args.out or EVAL_CC / "tls" / f"tls_{tag}.ply"
        report = build(args.room, step, scans, out_ply, xml_for(scans), files)
        report_path = out_ply.with_suffix(".json")
        top_view(out_ply, out_ply.with_name(out_ply.stem + "_top.png"))
    else:
        tmp = Path(tempfile.mkdtemp(prefix=f"vid2cloud_{args.room.lower()}_", dir=Path.home() / ".cache"))
        try:
            scans = extract(files, tmp)
            out_ply = tmp / f"tls_{tag}.ply"
            report = build(args.room, step, scans, out_ply, xml_for(scans), files)
            report["ply"] = "nie zachowano (katalog tymczasowy usunięty po pomiarze)"
            top_view(out_ply, EVAL_CC / "tls" / f"tls_{tag}_dims_top.png")
            shutil.copy(out_ply.with_suffix(".cc.log"), EVAL_CC / "tls" / f"tls_{tag}_dims.cc.log")
            with open(tmp / "cc_commands.txt", encoding="utf-8") as f:
                (EVAL_CC / "tls" / f"tls_{tag}_dims.cc_commands.txt").write_text(f.read(), encoding="utf-8")
        finally:
            shutil.rmtree(tmp)
        report_path = EVAL_CC / "tls" / f"tls_{tag}_dims.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_report(report)
    print(f"raport -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
