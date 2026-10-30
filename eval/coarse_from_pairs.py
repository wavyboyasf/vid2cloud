"""Zgrubne dopasowanie rekonstrukcji do TLS z par punktów wskazanych ręcznie (Umeyama, 7 parametrów).

Wejście: dwa pliki z eksportu „Point list picking” CloudCompare — punkty na chmurze rekonstrukcji
i te same punkty, w tej samej kolejności, na TLS. Wyjście: przekształcenie przez podobieństwo
y ≈ s · R · x + t (x — rekonstrukcja, y — TLS) jako macierz 4×4 w formacie, który czyta
`-APPLY_TRANS` CloudCompare (4 wiersze po 4 liczby), oraz raport: skala, odchyłka na każdej
parze |s R x + t − y| (w jednostkach TLS, czyli metrach) i ich RMS.

    python eval/coarse_from_pairs.py daleko_wolno
    python eval/coarse_from_pairs.py daleko_wolno --rec pary_rec.txt --tls pary_tls.txt

Domyślne ścieżki: runs/2026-09-21/eval_cc/pairs/<nazwa>_rec.txt i <nazwa>_tls.txt,
wynik do runs/2026-09-21/eval_cc/coarse/<nazwa>.txt (+ <nazwa>_report.json).

Format eksportu: każdy wiersz to „x,y,z”, „indeks,x,y,z” albo „etykieta,x,y,z” (separator
przecinek, średnik, tabulator albo spacja). Brane są trzy ostatnie pola; wiersze puste
i zaczynające się od # albo // są pomijane. Pary łączone są wyłącznie po kolejności.

Literatura: Umeyama 1991, Least-squares estimation of transformation parameters between two
point patterns, IEEE TPAMI 13(4), 376–380.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CC = REPO_ROOT / "runs" / "2026-09-21" / "eval_cc"
MIN_PAIRS = 3


def read_picked_points(path) -> np.ndarray:
    """Punkty z eksportu Point list picking — (n, 3), w kolejności z pliku."""
    pts = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        fields = [f for f in re.split(r"[,;\t ]+", s) if f]
        if len(fields) < 3:
            raise ValueError(f"{path}:{lineno}: mniej niż 3 pola: {line!r}")
        try:
            pts.append([float(v) for v in fields[-3:]])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: ostatnie trzy pola nie są liczbami: {line!r}") from exc
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def umeyama(src, dst, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimalizujące Σ |dst_i − (s R src_i + t)|² (Umeyama 1991, wzory 40–42)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"oczekiwano dwóch tablic (n, 3) tego samego kształtu, są {src.shape} i {dst.shape}")
    n = len(src)
    if n < MIN_PAIRS:
        raise ValueError(f"{n} par — potrzeba co najmniej {MIN_PAIRS}")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = float((xs ** 2).sum() / n)
    if var_s <= 0:
        raise ValueError("wszystkie punkty rekonstrukcji w jednym miejscu")
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    if np.linalg.matrix_rank(cov, tol=1e-12 * max(D.max(), 1e-300)) < 2:
        raise ValueError("punkty współliniowe — obrót nie jest wyznaczony")
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def to_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = s * np.asarray(R)
    M[:3, 3] = t
    return M


def apply(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    return P @ M[:3, :3].T + M[:3, 3]


def matrix_scale(M: np.ndarray) -> float:
    """Skala podobieństwa z macierzy 4×4: pierwiastek trzeciego stopnia z det(M[:3, :3])."""
    return float(np.cbrt(np.linalg.det(np.asarray(M)[:3, :3])))


def residuals(M: np.ndarray, src, dst) -> np.ndarray:
    return np.linalg.norm(apply(M, src) - np.asarray(dst, dtype=np.float64), axis=1)


def write_matrix(path, M: np.ndarray) -> None:
    """4 wiersze po 4 liczby, separator spacja — format `-APPLY_TRANS` CloudCompare."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(" ".join(f"{v:.12g}" for v in row) for row in M) + "\n", encoding="utf-8")


def read_matrix(path) -> np.ndarray:
    vals = [float(v) for v in Path(path).read_text(encoding="utf-8").split()]
    if len(vals) != 16:
        raise ValueError(f"{path}: {len(vals)} liczb zamiast 16")
    return np.array(vals).reshape(4, 4)


def coarse_report(src, dst) -> dict:
    s, R, t = umeyama(src, dst)
    M = to_matrix(s, R, t)
    res = residuals(M, src, dst)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return {
        "n_pairs": int(len(src)), "scale": s, "rotation_angle_deg": angle, "translation": t.tolist(),
        "matrix": M.tolist(),
        "residuals_m": res.tolist(), "rms_m": float(np.sqrt(np.mean(res ** 2))),
        "max_m": float(res.max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", help="nazwa nagrania (katalog w runs/2026-09-21/adapter/)")
    ap.add_argument("--rec", type=Path, help="punkty na rekonstrukcji (domyślnie eval_cc/pairs/<nazwa>_rec.txt)")
    ap.add_argument("--tls", type=Path, help="punkty na TLS (domyślnie eval_cc/pairs/<nazwa>_tls.txt)")
    ap.add_argument("--out", type=Path, help="macierz 4×4 (domyślnie eval_cc/coarse/<nazwa>.txt)")
    args = ap.parse_args(argv)
    rec = args.rec or EVAL_CC / "pairs" / f"{args.name}_rec.txt"
    tls = args.tls or EVAL_CC / "pairs" / f"{args.name}_tls.txt"
    out = args.out or EVAL_CC / "coarse" / f"{args.name}.txt"

    src, dst = read_picked_points(rec), read_picked_points(tls)
    if len(src) != len(dst):
        print(f"różna liczba punktów: {rec} ma {len(src)}, {tls} ma {len(dst)} — pary łączone po kolejności")
        return 1
    rep = coarse_report(src, dst)
    rep.update({"name": args.name, "rec_file": str(rec), "tls_file": str(tls), "matrix_file": str(out)})
    write_matrix(out, np.array(rep["matrix"]))
    rep_path = out.with_name(out.stem + "_report.json")
    rep_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{args.name}: {rep['n_pairs']} par, skala {rep['scale']:.6f} [m / jedn. chmury], "
          f"obrót {rep['rotation_angle_deg']:.3f}°")
    print("  para  odchyłka [m]")
    for i, r in enumerate(rep["residuals_m"]):
        print(f"  {i + 1:4d}  {r:.4f}")
    print(f"  RMS {rep['rms_m']:.4f} m, max {rep['max_m']:.4f} m")
    print(f"macierz -> {out}\nraport  -> {rep_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
