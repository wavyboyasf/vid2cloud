"""σ_obs i τ metody B (MNK) — policzone RAZ z trzech nagrań 21.09 i zamrożone w params.json.

    σ_obs² = Σ_k Σ_j (log f_kj − log med_k)² / (n − m)
             po klatkach kluczowych z ≥ 2 odcinkami (n reszt, m klatek), łącznie z trzech nagrań;
    τ²     = Σ (Δu)² / Σ Δt
             po kolejnych (w czasie) klatkach z obserwacjami w obrębie nagrania,
             Δu = log med_k' − log med_k, Δt = pts_k' − pts_k, łącznie z trzech nagrań;

gdzie f_kj = factor_final pary (klatka k, odcinek j) ze statusem ok, med_k = mediana f_kj
klatki k (kolumna f_obs w scale_keyframes.csv). Wartości trafiają do
vid2cloud/scale/params.json i są domyślne dla każdego przebiegu (`vid2cloud scale`,
`vid2cloud run --targets`); opcje --sigma-obs / --tau je nadpisują, a scale.json zapisuje,
które użyto. Bez --overwrite istniejący params.json nie jest nadpisywany. Env vid2cloud-tools.

    python eval/scale_params.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vid2cloud.scale import factors as fx  # noqa: E402
from vid2cloud.scale.estimate import load_targets  # noqa: E402
from vid2cloud.scale.lift import C_MIN  # noqa: E402
from vid2cloud.scale.per_keyframe import load_run, segment_rows, segments_with_refs  # noqa: E402

ADAPTER = REPO_ROOT / "runs" / "2026-09-21" / "adapter"
DEFAULT_RUNS = [ADAPTER / n for n in ("daleko_wolno", "blisko_z_gory", "daleko_szybko_pion")]
DEFAULT_TARGETS = REPO_ROOT / "data" / "2026-09-21" / "notes" / "targets.json"

FORMULAS = {
    "sigma_obs": "sigma_obs^2 = sum_k sum_j (ln f_kj - ln med_k)^2 / (n - m); klatki z >= 2 odcinkami, "
                 "n reszt, m klatek, laczne z nagran",
    "tau2": "tau^2 = sum (ln med_k' - ln med_k)^2 / sum (pts_k' - pts_k); kolejne w czasie klatki "
            "z obserwacjami w obrebie nagrania, laczne z nagran",
    "f_kj": "factor_final = L / (s_k * d_canon_final), pary ze statusem ok; med_k = mediana f_kj klatki k",
}


def _rel(p: Path) -> str:
    """Ścieżka względem repozytorium (bez resolve — data/ bywa dowiązaniem poza repo)."""
    p = Path(p).absolute()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    ap.add_argument("--out", type=Path, default=fx.PARAMS_PATH)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.out.exists() and not args.overwrite:
        print(f"{args.out} już istnieje — wartości są zamrożone; --overwrite tylko świadomie", file=sys.stderr)
        return 1

    targets = load_targets(args.targets)
    segments, refs = segments_with_refs(targets)
    groups, series, frames, incr = [], [], [], []
    for path in args.runs:
        run = load_run(path, {int(k) for k in targets.markers})
        obs = fx.observations(segment_rows(run, segments, refs, C_MIN))
        med = fx.frame_medians(obs)
        pts = {kd["kf"]: kd["pts_s"] for kd in run.kfs}
        for k in med:
            fk = [o.f for o in obs if o.kf == k]
            frames.append({"przebieg": run.name, "kf": k, "pts_s": pts[k], "n": len(fk), "f": fk, "med": med[k]})
            groups.append(fk)
        s = sorted((pts[k], med[k], k) for k in med)
        series.append([(t, m) for t, m, _ in s])
        for (t0, m0, k0), (t1, m1, k1) in zip(s, s[1:]):
            incr.append({"przebieg": run.name, "kf": [k0, k1], "dt_s": t1 - t0, "du": math.log(m1 / m0)})

    so = fx.sigma_obs_estimate(groups)
    ta = fx.tau_estimate(series)
    notes = []
    if so["n_frames"] < 3:
        notes.append(f"sigma_obs z {so['n_frames']} klatek (< 3)")
    if ta["n_increments"] < 3:
        notes.append(f"tau z {ta['n_increments']} przyrostów (< 3)")
    record = {
        "sigma_obs": so["sigma_obs"],
        "tau": ta["tau"],
        "tau2": ta["tau2"],
        "units": {"sigma_obs": "ln f (≈ błąd względny)", "tau": "ln f / sqrt(s)", "tau2": "(ln f)^2 / s"},
        "date": date.today().isoformat(),
        "source": "eval/scale_params.py, nagrania 21.09 (" + ", ".join(p.name for p in args.runs) + ")",
        "runs": [_rel(p) for p in args.runs],
        "targets": {"path": _rel(args.targets), "sha256": hashlib.sha256(args.targets.read_bytes()).hexdigest()},
        "c_min": C_MIN,
        "formulas": FORMULAS,
        "components": {
            "sigma_obs": {k: so[k] for k in ("n_residuals", "n_frames", "dof", "sum_sq")} | {
                "frames": [f for f in frames if f["n"] >= 2]},
            "tau": {k: ta[k] for k in ("n_increments", "sum_du2", "sum_dt")} | {"increments": incr},
        },
        "notes": notes,
    }
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"σ_obs = {so['sigma_obs']:.6g}  (n = {so['n_residuals']} reszt, m = {so['n_frames']} klatek, "
          f"stopni swobody {so['dof']}, Σ r² = {so['sum_sq']:.6g})")
    print(f"τ     = {ta['tau']:.6g}  τ² = {ta['tau2']:.6g} /s  ({ta['n_increments']} przyrostów, "
          f"Σ Δu² = {ta['sum_du2']:.6g}, Σ Δt = {ta['sum_dt']:.4f} s)")
    print("\n| przebieg | kf | pts [s] | n | med f |")
    print("|---|---|---|---|---|")
    for f in frames:
        print(f"| {f['przebieg']} | {f['kf']} | {f['pts_s']:.3f} | {f['n']} | {f['med']:.6f} |")
    print("\n| przebieg | kf → kf' | Δt [s] | Δu |")
    print("|---|---|---|---|")
    for i in incr:
        print(f"| {i['przebieg']} | {i['kf'][0]} → {i['kf'][1]} | {i['dt_s']:.3f} | {i['du']:+.6f} |")
    for n in notes:
        print(f"UWAGA: {n}")
    print(f"\nzapisano {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
