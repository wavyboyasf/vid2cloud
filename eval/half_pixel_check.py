"""Konwencja półpiksela: narożniki z pełnej klatki przeliczone na 512 vs wykryte wprost na obrazie 512.

Obraz wejścia sieci jest zapisany jako `keyframes/kf_NNN.png` (uimg) i jest wyrównany z pointmapą
piksel w piksel (ten sam crop, głowica sieci daje wartość na każdy piksel wejścia). ArUco wykryte
na nim to więc niezależny pomiar położenia narożnika w układzie pointmapy. Porównujemy go z:
    konwencja środka piksela (rescale_corners):  x' = (x + 0,5)·s − 0,5 − left
    konwencja narożnika piksela:                 x' = x·s − left
Obie różnią się o 0,5·(s − 1) (−0,433 px przy 4K -> 512). Dodatkowo: o ile zmienia się d_canon
odcinków (stan final), gdy narożniki przeliczyć konwencją narożnika zamiast środka.

Wypisuje liczby, nie wybiera konwencji. Env vid2cloud-tools.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from scale_per_keyframe import DEFAULT_RUNS, DEFAULT_TARGETS, load_run  # noqa: E402
from vid2cloud.scale import aruco  # noqa: E402
from vid2cloud.scale.estimate import load_targets  # noqa: E402
from vid2cloud.scale.lift import lift_reference, mean_confidence, parse_reference  # noqa: E402


def main() -> int:
    targets = load_targets(DEFAULT_TARGETS)
    ids = {int(k) for k in targets.markers}
    segments = [s for s in targets.segments.values() if s.marker_a is not None]
    out_rows, rel = [], []
    for path in DEFAULT_RUNS:
        run = load_run(path, ids)
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        s = cfg["input_transform"]["scale_x"]
        shift = 0.5 * (s - 1.0)  # konwencja narożnika minus konwencja środka
        for kd in run.kfs:
            k = kd["kf"]
            img = cv2.imread(str(path / "keyframes" / f"kf_{k:03d}.png"), cv2.IMREAD_GRAYSCALE)
            det = aruco.detect(img)
            for mid, q512 in run.markers[k].items():
                if mid not in det:
                    continue
                d = det[mid] - q512  # wykryte na 512 minus konwencja środka
                for c in range(4):
                    out_rows.append({"run": run.name, "kf": k, "id": mid, "corner": c,
                                     "dx_center": d[c, 0], "dy_center": d[c, 1],
                                     "dx_corner": d[c, 0] - shift, "dy_corner": d[c, 1] - shift})
            # wrażliwość d_canon (final) na konwencję
            dump = kd["final"]
            C = mean_confidence(dump["C"], dump["N"])
            for seg in segments:
                if seg.marker_a not in run.markers[k] or seg.marker_b not in run.markers[k]:
                    continue
                ra, rb = parse_reference(seg.reference, seg.marker_a, seg.marker_b)
                dd = []
                for sh in (0.0, shift):
                    la = lift_reference(ra, run.markers[k][ra.marker_id] + sh, dump["X_canon"], C)
                    lb = lift_reference(rb, run.markers[k][rb.marker_id] + sh, dump["X_canon"], C)
                    dd.append(np.linalg.norm(la.point - lb.point))
                rel.append({"run": run.name, "kf": k, "segment": seg.name,
                            "d_center": dd[0], "d_corner": dd[1], "rel_pct": 100 * (dd[1] / dd[0] - 1)})

    out = REPO_ROOT / "runs" / "2026-09-21" / "adapter" / "half_pixel_check.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    a = {k: np.array([r[k] for r in out_rows]) for k in ("dx_center", "dy_center", "dx_corner", "dy_corner")}
    n_markers = len({(r["run"], r["kf"], r["id"]) for r in out_rows})
    print(f"narożniki wykryte na 512 i na pełnej klatce: {len(out_rows)} ({n_markers} markerów)")
    for k, v in a.items():
        print(f"  {k:10s} średnia {v.mean():+.3f} px  mediana {np.median(v):+.3f} px  "
              f"std {v.std(ddof=1):.3f} px  |·| mediana {np.median(np.abs(v)):.3f} px")
    r = np.array([x["rel_pct"] for x in rel])
    print(f"d_canon (final), konwencja narożnika vs środka: {len(r)} odcinków, zmiana względna "
          f"mediana {np.median(r):+.4f} %, min {r.min():+.4f} %, max {r.max():+.4f} %")
    print(f"zapisano {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
