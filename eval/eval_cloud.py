"""Ocena kształtu jednej chmury rekonstrukcji względem TLS (CloudCompare w trybie wsadowym + numpy).

    python eval/eval_cloud.py <chmura.ply> --name <nagranie> --out <katalog>

Kroki (parametry z eval/eval_config.json, ustalone przed wynikami):
 a) zgrubne podobieństwo nagrania M_c (eval_cc/coarse/<nagranie>.txt z eval/coarse_from_pairs.py);
 b) prostopadłościan: percentyle 1–99 współrzędnych chmury po M_c (osobno X, Y, Z) ± 0,20 m,
    liczony RAZ na nagranie z runs/2026-09-21/adapter/<nagranie>/cloud.ply i zapisany do
    eval_cc/crop/<nagranie>.json; tym samym prostopadłościanem wycinane są rekonstrukcja i TLS
    (TLS po wycięciu: eval_cc/crop/<nagranie>_tls.ply, wspólny dla wariantów nagrania);
 c) SOR na rekonstrukcji (CloudCompare -SOR k σ);
 d) ICP ze skalą rekonstrukcji do TLS (CloudCompare -ICP -ADJUST_SCALE …) -> M_icp;
 e) C2C rekonstrukcja (compared) -> TLS (reference), bez modelu lokalnego (-C2C_DIST);
 f) numpy: n, średnia, mediana, RMS, p95, odsetek w progach; s_icp = cbrt(det((M_icp · M_c)[:3, :3]));
    RMS z ICP (jak wypisuje CloudCompare); liczba punktów przed i po SOR;
 g) jeśli obok chmury jest scale_per_keyframe.csv: s_mod = mediana factor_final, s_mod / s_icp − 1;
    chmura po SOR przeskalowana przez s_mod, start z (M_icp · M_c) bez członu skali, ICP sztywne,
    C2C -> RMS (kształt w skali z taśmy).

Wyjście w <out>: c2c_stats.json, c2c.ply (chmura po d) z polem odległości C2C — do zrzutów
z legendą w GUI), c2c_smod.ply (gałąź g), cc_commands.txt, cc_*.log, pliki pośrednie;
jeden wiersz w eval_cc/c2c_summary.csv (zastępowany przy ponownym przeliczeniu tej samej chmury).
Env vid2cloud-tools.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from coarse_from_pairs import matrix_scale, read_matrix  # noqa: E402
from plyio import read_ply, write_ply, xyz  # noqa: E402

CONFIG = REPO_ROOT / "eval" / "eval_config.json"
CC_SH = REPO_ROOT / "eval" / "cc.sh"
SESSION = REPO_ROOT / "runs" / "2026-09-21"
EVAL_CC = SESSION / "eval_cc"
DEFAULT_TLS_NAME = "tls/tls_koncertowa_1cm_static.ply"

SUMMARY_COLUMNS = (
    "name", "cloud", "out", "evaluated_at", "tls",
    "n_cloud", "n_crop", "n_before_sor", "n_after_sor",
    "s_coarse", "s_icp", "icp_rms_cc", "icp_n_final",
    "c2c_n", "c2c_mean", "c2c_median", "c2c_rms", "c2c_p95",
    "frac_le_0.05", "frac_le_0.10", "frac_le_0.20",
    "s_mod", "s_mod_n", "s_mod_vs_icp_pct", "rigid_icp_rms_cc", "rigid_c2c_rms", "rigid_c2c_mean",
    "rigid_c2c_median", "rigid_c2c_p95",
)


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Funkcje czyste
# --------------------------------------------------------------------------- #


def transform(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.asarray(P, dtype=np.float64) @ M[:3, :3].T + M[:3, 3]


def crop_box(points_aligned: np.ndarray, percentiles=(1, 99), margin: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Prostopadłościan: percentyle współrzędnych osobno na każdej osi, rozszerzone o margines."""
    lo, hi = np.percentile(np.asarray(points_aligned, dtype=np.float64), percentiles, axis=0)
    return lo - margin, hi + margin


def in_box(P: np.ndarray, lo, hi) -> np.ndarray:
    P = np.asarray(P)
    return np.all((P >= np.asarray(lo)) & (P <= np.asarray(hi)), axis=1)


def distance_stats(d, thresholds) -> dict:
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    out = {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
           "rms": float(np.sqrt(np.mean(d ** 2))), "p95": float(np.percentile(d, 95)), "max": float(d.max())}
    out["fraction_le"] = {f"{t:.2f}": float(np.mean(d <= t)) for t in thresholds}
    return out


def parse_icp_log(log: str) -> dict:
    """RMS i liczba punktów ostatniego kroku z wypisu `-ICP` CloudCompare 2.13."""
    rms = re.findall(r"^\[[^\]]*\] RMS: ([0-9eE.+-]+)", log, flags=re.M)
    n = re.findall(r"Number of points used for final step: (\d+)", log)
    if not rms:
        raise ValueError("w logu CloudCompare nie ma wiersza 'RMS:' z ICP")
    return {"rms": float(rms[-1]), "n_final": int(n[-1]) if n else None}


def similarity_parts(M: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """M = [s R | t] -> s, R, t."""
    s = matrix_scale(M)
    return s, M[:3, :3] / s, M[:3, 3].copy()


def s_mod_from_csv(path: Path) -> tuple[float | None, int]:
    """Mediana factor_final z scale_per_keyframe.csv (wiersze z wartością); (None, 0), gdy brak."""
    vals = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("factor_final"):
                vals.append(float(r["factor_final"]))
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)


def c2c_field(vertices) -> str:
    names = [n for n in vertices.dtype.names if "c2c" in n.lower()]
    if len(names) != 1:
        raise ValueError(f"oczekiwano jednego pola C2C w PLY, są: {vertices.dtype.names}")
    return names[0]


def cc_base_args() -> list[str]:
    return ["-AUTO_SAVE", "OFF", "-NO_TIMESTAMP", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE"]


def icp_args(cfg: dict, adjust_scale: bool) -> list[str]:
    icp = cfg["icp"]
    args = ["-ICP", "-MIN_ERROR_DIFF", f"{icp['min_error_diff']:g}",
            "-RANDOM_SAMPLING_LIMIT", str(int(icp["random_sampling_limit"])),
            "-OVERLAP", f"{icp['final_overlap_pct']:g}"]
    return args + (["-ADJUST_SCALE"] if adjust_scale else [])


def cc_main_args(cfg: dict, rec: Path, tls: Path, sor_out: Path, c2c_out: Path) -> list[str]:
    """c) SOR, d) ICP ze skalą, e) C2C. Rekonstrukcja wczytana pierwsza = data / compared."""
    return [*cc_base_args(), "-O", str(rec), "-SOR", str(int(cfg["sor_knn"])), f"{cfg['sor_sigma']:g}",
            "-SAVE_CLOUDS", "FILE", str(sor_out), "-O", str(tls), *icp_args(cfg, True),
            "-C2C_DIST", "-POP_CLOUDS", "-SAVE_CLOUDS", "FILE", str(c2c_out)]


def cc_rigid_args(cfg: dict, rec: Path, tls: Path, c2c_out: Path) -> list[str]:
    """g) ICP bez skali (te same pozostałe parametry), C2C."""
    return [*cc_base_args(), "-O", str(rec), "-O", str(tls), *icp_args(cfg, False),
            "-C2C_DIST", "-POP_CLOUDS", "-SAVE_CLOUDS", "FILE", str(c2c_out)]


# --------------------------------------------------------------------------- #
# CloudCompare
# --------------------------------------------------------------------------- #


def _q(a: str) -> str:
    return f"'{a}'" if re.search(r"[\s'\"]", a) else a


def run_cc(args: list[str], log_path: Path, commands_path: Path) -> str:
    cmd = ["bash", str(CC_SH), *args]
    with open(commands_path, "a", encoding="utf-8") as f:
        f.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n" + " ".join(_q(a) for a in cmd) + "\n")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 or "Processed finished" not in proc.stdout:
        raise RuntimeError(f"CloudCompare: kod {proc.returncode}, log {log_path}")
    return proc.stdout


def take_registration_matrix(out: Path, dest: Path) -> np.ndarray:
    """CC zapisuje macierz ICP jako <ostatnia ścieżka zapisu chmury data>_REGISTRATION_MATRIX*.txt
    (np. rec_sor.ply_REGISTRATION_MATRIX.txt). Po każdym wywołaniu przenoszona do `dest`, więc
    w katalogu ma być dokładnie jedna."""
    found = list(out.glob("*REGISTRATION_MATRIX*.txt"))
    if len(found) != 1:
        raise RuntimeError(f"oczekiwano jednej macierzy ICP *REGISTRATION_MATRIX*.txt w {out}, są: {found}")
    found[0].replace(dest)
    return np.loadtxt(dest).reshape(4, 4)


# --------------------------------------------------------------------------- #
# Prostopadłościan i TLS po wycięciu — raz na nagranie
# --------------------------------------------------------------------------- #


def _file_id(p: Path) -> dict:
    st = Path(p).stat()
    return {"path": str(p), "size": st.st_size, "mtime": st.st_mtime}


def get_crop(name: str, eval_cc: Path, crop_source: Path, M_c: np.ndarray, coarse_path: Path, cfg: dict) -> dict:
    path = eval_cc / "crop" / f"{name}.json"
    if path.exists():
        box = json.loads(path.read_text(encoding="utf-8"))
        if not np.allclose(np.array(box["coarse_matrix"]), M_c, rtol=0, atol=1e-12):
            raise RuntimeError(f"{path}: liczony z inną macierzą zgrubną niż {coarse_path} — "
                               "usuń go (i <nazwa>_tls.ply), jeśli macierz zgrubna została zmieniona")
        return box
    P = transform(M_c, xyz(read_ply(crop_source)))
    lo, hi = crop_box(P, cfg["crop_percentiles"], cfg["crop_margin_m"])
    box = {"name": name, "source": str(crop_source), "coarse_file": str(coarse_path), "coarse_matrix": M_c.tolist(),
           "percentiles": cfg["crop_percentiles"], "margin_m": cfg["crop_margin_m"],
           "min": lo.tolist(), "max": hi.tolist(), "size_m": (hi - lo).tolist(),
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(box, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return box


def get_tls_crop(name: str, eval_cc: Path, tls: Path, box: dict) -> tuple[Path, dict]:
    ply = eval_cc / "crop" / f"{name}_tls.ply"
    meta_path = eval_cc / "crop" / f"{name}_tls.json"
    want = {"tls": _file_id(tls), "min": box["min"], "max": box["max"]}
    if ply.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if {k: meta.get(k) for k in want} == want:
            return ply, meta
    v = read_ply(tls)
    keep = in_box(xyz(v), box["min"], box["max"])
    write_ply(ply, v[keep], comments=(f"vid2cloud: TLS {tls.name} wyciety prostopadloscianem {name}",))
    meta = {**want, "n_tls": int(len(v)), "n_crop": int(keep.sum())}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ply, meta


# --------------------------------------------------------------------------- #
# Ocena jednej chmury
# --------------------------------------------------------------------------- #


def evaluate(cloud: Path, name: str, out: Path, *, tls: Path, coarse: Path, crop_source: Path, eval_cc: Path,
             summary: Path, scale_csv: Path | None, cfg: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*REGISTRATION_MATRIX*.txt"):  # resztki przerwanego przebiegu
        stale.unlink()
    commands = out / "cc_commands.txt"
    commands.write_text("", encoding="utf-8")
    thresholds = cfg["thresholds_m"]

    # a) zgrubne dopasowanie, b) prostopadłościan
    M_c = read_matrix(coarse)
    box = get_crop(name, eval_cc, crop_source, M_c, coarse, cfg)
    tls_crop, tls_meta = get_tls_crop(name, eval_cc, tls, box)
    v = read_ply(cloud)
    P = transform(M_c, xyz(v))
    keep = in_box(P, box["min"], box["max"])
    rec = v[keep].copy()
    rec["x"], rec["y"], rec["z"] = P[keep, 0], P[keep, 1], P[keep, 2]
    rec_path = out / "rec_coarse_crop.ply"
    write_ply(rec_path, rec, comments=(f"vid2cloud: {cloud} po macierzy zgrubnej {coarse.name}, wyciety",))

    # c) SOR, d) ICP ze skalą, e) C2C — CloudCompare
    sor_path, c2c_path = out / "rec_sor.ply", out / "c2c.ply"
    log = run_cc(cc_main_args(cfg, rec_path, tls_crop, sor_path, c2c_path), out / "cc_main.log", commands)
    M_icp = take_registration_matrix(out, out / "icp_matrix.txt")
    icp = parse_icp_log(log)
    c2c = read_ply(c2c_path)
    stats = distance_stats(c2c[c2c_field(c2c)], thresholds)

    # f) skala całkowita
    M_tot = M_icp @ M_c
    np.savetxt(out / "total_matrix.txt", M_tot, fmt="%.12g")
    s_icp = matrix_scale(M_tot)
    n_sor = len(read_ply(sor_path))
    result = {
        "name": name, "cloud": str(cloud), "out": str(out), "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tls": str(tls), "tls_crop": str(tls_crop), "n_tls_crop": tls_meta.get("n_crop"),
        "coarse_file": str(coarse), "crop": box,
        "n_cloud": int(len(v)), "n_crop": int(keep.sum()), "n_before_sor": int(keep.sum()), "n_after_sor": n_sor,
        "M_coarse": M_c.tolist(), "M_icp": M_icp.tolist(), "M_total": M_tot.tolist(),
        "s_coarse": matrix_scale(M_c), "s_icp_step": matrix_scale(M_icp), "s_icp": s_icp,
        "icp_rms_cc": icp["rms"], "icp_n_final": icp["n_final"],
        "c2c": stats, "config": {k: cfg[k] for k in cfg if not k.startswith("_")},
    }

    # g) skala z taśmy
    result["scale_csv"] = str(scale_csv) if scale_csv else None
    s_mod, n_mod = s_mod_from_csv(scale_csv) if scale_csv and scale_csv.exists() else (None, 0)
    result.update({"s_mod": s_mod, "s_mod_n": n_mod})
    if s_mod is not None:
        result["s_mod_vs_icp_pct"] = (s_mod / s_icp - 1.0) * 100.0
        _, R, t = similarity_parts(M_tot)
        sor = read_ply(sor_path)
        X = transform(np.linalg.inv(M_c), xyz(sor))  # z powrotem do jednostek chmury
        Y0 = (s_mod * X) @ R.T + t                    # skala z taśmy, start z d) bez skali
        init = sor.copy()
        init["x"], init["y"], init["z"] = Y0[:, 0], Y0[:, 1], Y0[:, 2]
        init_path = out / "rec_smod_init.ply"
        write_ply(init_path, init, comments=(f"vid2cloud: po SOR, skala s_mod = {s_mod:.9g}, start z M_total bez skali",))
        log_r = run_cc(cc_rigid_args(cfg, init_path, tls_crop, out / "c2c_smod.ply"), out / "cc_rigid.log", commands)
        M_r = take_registration_matrix(out, out / "icp_rigid_matrix.txt")
        c2c_r = read_ply(out / "c2c_smod.ply")
        result.update({"rigid_M_icp": M_r.tolist(), "rigid_det": float(np.linalg.det(M_r[:3, :3])),
                       "rigid_icp_rms_cc": parse_icp_log(log_r)["rms"],
                       "rigid_c2c": distance_stats(c2c_r[c2c_field(c2c_r)], thresholds)})

    (out / "c2c_stats.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_summary(summary, summary_row(result))
    return result


def summary_row(r: dict) -> dict:
    c, rg = r["c2c"], r.get("rigid_c2c") or {}
    row = {k: r.get(k) for k in SUMMARY_COLUMNS}
    row.update({"c2c_n": c["n"], "c2c_mean": c["mean"], "c2c_median": c["median"], "c2c_rms": c["rms"],
                "c2c_p95": c["p95"], "rigid_c2c_rms": rg.get("rms"), "rigid_c2c_mean": rg.get("mean"),
                "rigid_c2c_median": rg.get("median"), "rigid_c2c_p95": rg.get("p95")})
    for t, v in c["fraction_le"].items():
        row[f"frac_le_{t}"] = v
    return row


def _cell(v) -> str:
    if v is None:
        return ""
    return f"{v:.9g}" if isinstance(v, float) else str(v)


def update_summary(path: Path, row: dict) -> None:
    """Jeden wiersz na chmurę: ten sam `cloud` zastępuje poprzedni wiersz."""
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["cloud"] != row["cloud"]]
    rows.append({k: _cell(row.get(k)) for k in SUMMARY_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cloud", type=Path)
    ap.add_argument("--name", required=True, help="nagranie (daleko_wolno, blisko_z_gory, daleko_szybko_pion, …)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-cc", type=Path, default=EVAL_CC)
    ap.add_argument("--tls", type=Path, help=f"domyślnie <eval-cc>/{DEFAULT_TLS_NAME}")
    ap.add_argument("--coarse", type=Path, help="domyślnie <eval-cc>/coarse/<nagranie>.txt")
    ap.add_argument("--crop-source", type=Path,
                    help="chmura, z której liczony jest prostopadłościan (domyślnie runs/2026-09-21/adapter/<nagranie>/cloud.ply)")
    ap.add_argument("--scale-csv", type=Path, help="domyślnie scale_per_keyframe.csv obok chmury")
    ap.add_argument("--summary", type=Path, help="domyślnie <eval-cc>/c2c_summary.csv")
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tls = args.tls or args.eval_cc / DEFAULT_TLS_NAME
    coarse = args.coarse or args.eval_cc / "coarse" / f"{args.name}.txt"
    crop_source = args.crop_source or SESSION / "adapter" / args.name / "cloud.ply"
    scale_csv = args.scale_csv or args.cloud.parent / "scale_per_keyframe.csv"
    summary = args.summary or args.eval_cc / "c2c_summary.csv"
    for label, p in (("chmura", args.cloud), ("TLS", tls), ("macierz zgrubna", coarse), ("źródło prostopadłościanu", crop_source)):
        if not p.exists():
            print(f"brak: {label} {p}")
            return 2
    t0 = time.time()
    r = evaluate(args.cloud, args.name, args.out, tls=tls, coarse=coarse, crop_source=crop_source,
                 eval_cc=args.eval_cc, summary=summary, scale_csv=scale_csv, cfg=cfg)
    c = r["c2c"]
    print(f"{args.cloud} ({args.name}): {r['n_cloud']} pkt, po wycięciu {r['n_crop']}, po SOR {r['n_after_sor']}")
    print(f"  s_coarse {r['s_coarse']:.6f}, s_icp {r['s_icp']:.6f}, RMS ICP (CC) {r['icp_rms_cc']:.4f}")
    print(f"  C2C: n {c['n']}, średnia {c['mean']:.4f}, mediana {c['median']:.4f}, RMS {c['rms']:.4f}, "
          f"p95 {c['p95']:.4f} m; " + ", ".join(f"≤{k} m: {v * 100:.1f} %" for k, v in c["fraction_le"].items()))
    if r.get("s_mod") is not None:
        print(f"  s_mod {r['s_mod']:.6f} (n {r['s_mod_n']}), s_mod/s_icp − 1 = {r['s_mod_vs_icp_pct']:+.2f} %, "
              f"RMS C2C sztywnie w skali s_mod {r['rigid_c2c']['rms']:.4f} m")
    print(f"  {time.time() - t0:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
