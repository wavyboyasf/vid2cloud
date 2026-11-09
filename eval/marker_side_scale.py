"""Współczynnik skali z boku znacznika ArUco w każdej klatce kluczowej (VID_ODLEGLOSC, mieszkanie).

W tym nagraniu nie ma odcinków taśmy między znacznikami, a odległości kamera–plansza z
docs/decisions.md (2026-09-19) to szacunek — nie są tu używane. Wzorcem jest bok czarnego
kwadratu: 149,7 mm (zmierzony bok wydruku, odpowiedź autora 28.10), plansze ID 11, 8, 10.

Dla każdej klatki kluczowej k i każdej wykrytej planszy: cztery narożniki ArUco liftowane tak
jak w sesji B (`scale/lift.py`: płaszczyzna planszy z pikseli wnętrza + promień w pikselu
narożnika), długości czterech boków w mapie klatki k, w dwóch stanach:

    factor_net_i   = L / bok_i (created)              (sama sieć — jak factor_net_created)
    factor_final_i = L / (s_final · bok_i (final))    (chmura na końcu)

i mediany z czterech boków. Rozrzut boków = (max − min) / mediana czterech boków [%];
powyżej 5 % wiersz ma flagę `spread_gt_5pct` (lift boku niestabilny — zgłosić, nie
interpretować). Bok w pikselach przy 512 px z narożników `*_512`. Głębokość: z_canon —
z środka planszy (mediana X w czworokącie 50 %, `lift_marker`) w układzie kamery klatki k,
stan created; z_m = z_canon · mediana factor_net_median wszystkich wierszy przebiegu.

    python eval/marker_side_scale.py               # runs/odleglosc/adapter
Wyjście: <przebieg>/marker_side_scale.csv. Env vid2cloud-tools.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vid2cloud.scale.lift import C_MIN, fit_plane_ray, lift_marker, mean_confidence  # noqa: E402
from vid2cloud.scale.per_keyframe import STATES, load_run, s_at, write_csv  # noqa: E402

DEFAULT_RUN = REPO_ROOT / "runs" / "odleglosc" / "adapter"
SIDE_MM = 149.7
MARKER_IDS = (11, 8, 10)
SPREAD_FLAG_PCT = 5.0

COLUMNS = (
    "kf", "frame_index", "pts_s", "marker_id", "L_mm",
    "side_px_512_mean", "side_px_512_min", "side_px_512_max",
    *(f"side_canon_created_{i}" for i in range(4)), *(f"factor_net_{i}" for i in range(4)),
    "factor_net_median", "side_spread_pct", "spread_gt_5pct",
    "s_created", "s_final", *(f"side_canon_final_{i}" for i in range(4)),
    "factor_final_median", "side_spread_final_pct",
    "z_canon", "net_median_run", "z_m",
    "n_px_created", "mad_created", "plane_rms_created", "n_px_center", "status",
)


def quad_sides(P: np.ndarray) -> np.ndarray:
    """Długości boków czworokąta 0-1, 1-2, 2-3, 3-0 (2D albo 3D)."""
    P = np.asarray(P, dtype=np.float64)
    return np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)


def spread_pct(sides) -> float:
    sides = np.asarray(sides, dtype=np.float64)
    return float((sides.max() - sides.min()) / np.median(sides) * 100.0)


def marker_rows(run, side_mm: float = SIDE_MM, ids=MARKER_IDS, c_min: float = C_MIN) -> list[dict]:
    L = side_mm / 1000.0
    rows = []
    for kd in run.kfs:
        k = kd["kf"]
        s = {st: s_at(run.history, k, st) for st in STATES}
        for mid in ids:
            q = run.markers[k].get(mid)
            if q is None:
                continue
            px = quad_sides(q)
            row = {"kf": k, "frame_index": kd["frame_index"], "pts_s": kd["pts_s"], "marker_id": mid, "L_mm": side_mm,
                   "side_px_512_mean": float(px.mean()), "side_px_512_min": float(px.min()),
                   "side_px_512_max": float(px.max()), "s_created": s["created"], "s_final": s["final"]}
            reasons = []
            for st in STATES:
                dump = kd[st]
                try:
                    C = mean_confidence(dump["C"], dump["N"])
                except ValueError as exc:
                    reasons.append(f"{st}: {exc}")
                    continue
                model, reason = fit_plane_ray(q, dump["X_canon"], C, c_min)
                if model is None:
                    reasons.append(f"{st}: {reason}")
                    continue
                sides = quad_sides(np.array([model.point(q[i]) for i in range(4)]))
                for i in range(4):
                    row[f"side_canon_{st}_{i}"] = float(sides[i])
                if st == "created":
                    f = L / sides
                    row.update({**{f"factor_net_{i}": float(f[i]) for i in range(4)},
                                "factor_net_median": float(np.median(f)), "side_spread_pct": spread_pct(sides),
                                "n_px_created": model.n_px, "mad_created": model.mad,
                                "plane_rms_created": model.plane_rms})
                    row["spread_gt_5pct"] = row["side_spread_pct"] > SPREAD_FLAG_PCT
                    center = lift_marker(q, dump["X_canon"], C, c_min)
                    row["n_px_center"] = center.n_px
                    if center.point is None:
                        reasons.append(f"środek: {center.reason}")
                    else:
                        row["z_canon"] = float(center.point[2])
                else:
                    row["factor_final_median"] = float(np.median(L / (s["final"] * sides)))
                    row["side_spread_final_pct"] = spread_pct(sides)
            row["status"] = "ok" if not reasons else " | ".join(reasons)
            rows.append(row)
    net = [r["factor_net_median"] for r in rows if r.get("factor_net_median") is not None]
    med = float(np.median(net)) if net else None
    for r in rows:
        r["net_median_run"] = med
        if med is not None and r.get("z_canon") is not None:
            r["z_m"] = r["z_canon"] * med
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN)
    ap.add_argument("--side-mm", type=float, default=SIDE_MM)
    args = ap.parse_args(argv)
    run = load_run(args.run, set(MARKER_IDS))
    rows = marker_rows(run, args.side_mm)
    out = run.path / "marker_side_scale.csv"
    write_csv(out, COLUMNS, rows)
    print(f"{run.name}: {len(run.kfs)} kf, {len(rows)} wierszy plansza × kf -> {out}")
    print("| kf | ID | bok 512 px śr. (min–max) | factor_net mediana | rozrzut boków % | factor_final mediana | z_canon | z_m | status |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        def f(key, nd=4):
            v = r.get(key)
            return "—" if v is None else f"{v:.{nd}f}"
        flag = " **> 5 %**" if r.get("spread_gt_5pct") else ""
        print(f"| {r['kf']} | {r['marker_id']} | {f('side_px_512_mean', 2)} ({f('side_px_512_min', 2)}–{f('side_px_512_max', 2)}) "
              f"| {f('factor_net_median')} | {f('side_spread_pct', 2)}{flag} | {f('factor_final_median')} "
              f"| {f('z_canon')} | {f('z_m', 3)} | {r['status']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
