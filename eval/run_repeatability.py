"""Powtarzalność przebiegów: s_k (final) per klatka kluczowa między dwoma lub więcej przebiegami.

Czyta `keyframes/kf_*_final.npz` z każdego katalogu przebiegu (`vid2cloud run`). Pierwszy
katalog jest odniesieniem; dla każdego następnego wypisuje wiersz na klatkę kluczową:
indeks klatki źródła w obu przebiegach, s_k w obu, różnicę i różnicę względną.
Klatki kluczowe parowane są po indeksie k — jeśli liczby klatek kluczowych albo ich
klatki źródła się różnią, widać to w kolumnach frame_index i w nagłówku.

Bez torcha (env vid2cloud-tools wystarczy). Liczby się zapisuje, nie interpretuje.

    python eval/run_repeatability.py runs/.../blisko_z_gory runs/.../blisko_z_gory_powt2 \\
        runs/.../blisko_z_gory_powt3_mt --out runs/.../powtarzalnosc_125437.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vid2cloud.export.keyframes import read_keyframe_npz  # noqa: E402

COLUMNS = ("run", "single_thread", "kf", "frame_index_ref", "frame_index", "s_ref", "s",
           "s_diff", "s_rel_diff_pct")


def load_run(run_dir: Path) -> tuple[list[dict], dict]:
    files = sorted(Path(run_dir, "keyframes").glob("kf_*_final.npz"))
    if not files:
        raise FileNotFoundError(f"{run_dir}: brak keyframes/kf_*_final.npz")
    kfs = [read_keyframe_npz(f) for f in files]
    cfg = json.loads(Path(run_dir, "config.json").read_text(encoding="utf-8"))
    return kfs, cfg


def compare(ref: list[dict], other: list[dict]) -> list[dict]:
    rows = []
    for k in range(max(len(ref), len(other))):
        a = ref[k] if k < len(ref) else None
        b = other[k] if k < len(other) else None
        s_a = float(a["T_WC"][7]) if a else None
        s_b = float(b["T_WC"][7]) if b else None
        diff = s_b - s_a if (a and b) else None
        rows.append({
            "kf": k,
            "frame_index_ref": a["frame_index"] if a else None,
            "frame_index": b["frame_index"] if b else None,
            "s_ref": s_a,
            "s": s_b,
            "s_diff": diff,
            "s_rel_diff_pct": 100.0 * diff / s_a if diff is not None else None,
        })
    return rows


def _fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path, nargs="+", help="katalogi przebiegów; pierwszy = odniesienie")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if len(args.runs) < 2:
        parser.error("potrzebne co najmniej dwa przebiegi")

    ref, ref_cfg = load_run(args.runs[0])
    print(f"odniesienie: {args.runs[0]}  (single_thread={ref_cfg['engine']['single_thread']}, "
          f"kf={len(ref)})")
    all_rows = []
    for run_dir in args.runs[1:]:
        other, cfg = load_run(run_dir)
        st = cfg["engine"]["single_thread"]
        print(f"\n{run_dir}  (single_thread={st}, kf={len(other)})")
        print("  ".join(COLUMNS[2:]))
        for row in compare(ref, other):
            row = {"run": run_dir.name, "single_thread": st} | row
            all_rows.append(row)
            print("  ".join(_fmt(row[c]) for c in COLUMNS[2:]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for row in all_rows:
                w.writerow([_fmt(row[c]) for c in COLUMNS])
        print(f"\nzapisano {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
