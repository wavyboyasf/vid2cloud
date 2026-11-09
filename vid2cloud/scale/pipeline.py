"""Skala osobno dla każdej klatki kluczowej na gotowym katalogu przebiegu.

Jedna funkcja, `scale_run()`, woła ją `vid2cloud scale <katalog>` (env vid2cloud-tools,
bez GPU) i `vid2cloud run ... --targets` po `finalize()` silnika. Kroki:

1. `scale_per_keyframe.csv` — współczynnik każdej pary (klatka kluczowa, odcinek), lift jak
   w `scale/per_keyframe.py` (bez zmian względem `eval/scale_per_keyframe.py`);
2. obserwacje f_kj = factor_final par ze statusem ok; wektor f_k w trzech wariantach
   (`scale/factors.py`): globalny (jedna liczba), A — interpolacja, B — MNK;
3. chmura odbudowana z `kf_*_final.npz` regułą upstreamu (`scale/cloud.py`), porównana
   z `cloud.ply`; położenia klatek jak ciąg poligonowy, punkt P = p_k + f_k·s_k·R_k·X;
4. zapis: `scale_keyframes.csv`, `cloud_m_global.ply`, `cloud_m_kf_interp.ply`,
   `cloud_m_kf_mnk.ply`, `scale.json`.

Bez żadnej pary ze statusem ok: tylko `scale_per_keyframe.csv` i `scale.json` ze statusem
„brak par” — to wynik, nie błąd. Outliery nie są odrzucane; rozrzut > 5 % daje ostrzeżenie
w logu i w `scale.json`, liczb nie zmienia.

Bez `overwrite` istniejące `scale.json`, `scale_keyframes.csv` i `cloud_m_*.ply` są błędem
przed liczeniem. `scale_per_keyframe.csv` pisze też `eval/scale_per_keyframe.py`, więc
istniejący plik jest błędem tylko wtedy, gdy jego treść różni się od policzonej.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from vid2cloud.scale import factors as fx
from vid2cloud.scale.cloud import CLOUD_C_THRESHOLD, assemble, compare_with_upstream, keyframe_clouds, write_cloud
from vid2cloud.scale.estimate import SPREAD_WARN_PCT, load_targets
from vid2cloud.scale.lift import C_MIN
from vid2cloud.scale.per_keyframe import COLUMNS, csv_text, load_run, segment_rows, segments_with_refs, write_csv

PER_PAIR_CSV = "scale_per_keyframe.csv"
KEYFRAMES_CSV = "scale_keyframes.csv"
SCALE_JSON = "scale.json"
VARIANTS = ("global", "kf_interp", "kf_mnk")
CLOUDS = {v: f"cloud_m_{v}.ply" for v in VARIANTS}
EXCLUSIVE = (SCALE_JSON, KEYFRAMES_CSV, *CLOUDS.values())
KEYFRAME_COLUMNS = ("kf", "pts_s", "n_par", "f_obs", "f_interp", "f_mnk", "sigma_u_mnk", "f_global")
STATUS_OK, STATUS_NO_PAIRS = "ok", "brak par"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_params(sigma_obs: Optional[float], tau: Optional[float], params_path: Path = fx.PARAMS_PATH) -> dict:
    """σ_obs i τ: z opcji, jeśli podane, inaczej zamrożone z params.json; zapis mówi, skąd która."""
    frozen = fx.load_params(params_path)
    out = {}
    for key, given in (("sigma_obs", sigma_obs), ("tau", tau)):
        if given is not None:
            if not given > 0:
                raise ValueError(f"--{key.replace('_', '-')} musi być > 0, jest {given}")
            out[key] = {"value": float(given), "source": "opcja CLI"}
        else:
            out[key] = {"value": float(frozen[key]),
                        "source": f"params.json z {frozen.get('date', '?')} (eval/scale_params.py)"}
    out["params_json"] = {"path": str(params_path), "sha256": _sha256(params_path)}
    return out


def _spread_pct(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float((v.max() - v.min()) / np.median(v) * 100.0)


def scale_run(run_dir: Path, targets_path: Path, c_min: float = C_MIN, sigma_obs: Optional[float] = None,
              tau: Optional[float] = None, overwrite: bool = False,
              log: Callable[[str], None] = print) -> dict:
    """Skala per klatka kluczowa dla katalogu przebiegu. Zwraca zawartość `scale.json`."""
    t0 = time.perf_counter()
    run_dir, targets_path = Path(run_dir), Path(targets_path)
    if not overwrite:
        existing = [n for n in EXCLUSIVE if (run_dir / n).exists()]
        if existing:
            raise FileExistsError(f"{run_dir}: już są {', '.join(existing)} — nie nadpisuję (--overwrite)")

    targets = load_targets(targets_path)
    segments, refs = segments_with_refs(targets)
    params = resolve_params(sigma_obs, tau)
    run = load_run(run_dir, {int(k) for k in targets.markers})

    rows = segment_rows(run, segments, refs, c_min)
    text = csv_text(COLUMNS, rows)
    per_pair = run_dir / PER_PAIR_CSV
    if per_pair.exists() and not overwrite and per_pair.read_bytes() != text.encode("utf-8"):
        raise FileExistsError(f"{per_pair}: istnieje i różni się od policzonego — nie nadpisuję (--overwrite)")

    obs = fx.observations(rows)
    K = len(run.kfs)
    record: dict = {
        "written": _now(),
        "run_dir": str(run_dir.absolute()),
        "recording": run.recording,
        "targets": {"path": str(targets_path.absolute()), "sha256": _sha256(targets_path)},
        "c_min": c_min,
        "cloud_c_threshold": CLOUD_C_THRESHOLD,
        "units": "f przelicza jednostki chmury na metry",
        "n_keyframes": K,
        "n_pairs": len(rows),
        "n_pairs_ok": len(obs),
        "n_keyframes_with_obs": len({o.kf for o in obs}),
        "params": params,
        "warnings": [],
        "files": {"scale_per_keyframe": str(per_pair)},
    }
    for r in rows:
        if r["status"] != "ok":
            log(f"  kf {r['kf']} {r['segment']}: {r['status']}")

    if not obs:
        write_csv(per_pair, COLUMNS, rows)
        record.update({"status": STATUS_NO_PAIRS,
                       "message": f"żadna z {K} klatek kluczowych nie ma pary odcinka ze statusem ok "
                                  f"({len(rows)} par z oboma markerami)",
                       "t_s": time.perf_counter() - t0})
        (run_dir / SCALE_JSON).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        record["files"]["scale_json"] = str(run_dir / SCALE_JSON)
        log(f"skala: {STATUS_NO_PAIRS} — {record['message']}")
        return record

    # --- współczynniki --------------------------------------------------- #
    pts = np.array([kd["pts_s"] for kd in run.kfs], dtype=np.float64)
    f_glob, how, est = fx.global_factor(rows, targets)
    f_vec = {
        "global": np.full(K, f_glob),
        "kf_interp": fx.interp_factors(pts, obs),
    }
    lsq = fx.lsq_factors(pts, obs, params["sigma_obs"]["value"], params["tau"]["value"])
    f_vec["kf_mnk"] = lsq.f

    # --- chmury ---------------------------------------------------------- #
    kclouds = keyframe_clouds([kd["final"] for kd in run.kfs])
    raw = assemble(kclouds)
    rebuild, colors = compare_with_upstream(raw, run_dir / "cloud.ply")
    del raw
    if not rebuild["same_order"]:
        record["warnings"].append(f"odbudowa nie odtwarza cloud.ply punkt po punkcie ({rebuild}) — chmury bez kolorów")
    t = np.array([kc.t for kc in kclouds])
    pos = {v: fx.keyframe_positions(t, f_vec[v]) for v in VARIANTS}
    for v in VARIANTS:
        out = run_dir / CLOUDS[v]
        P = assemble(kclouds, f_vec[v], pos[v])
        write_cloud(out, P, colors, comments=(
            f"vid2cloud: chmura w metrach, wariant {v}; P = p_k + f_k * s_k * R_k * X",
            f"zrodlo: {run_dir.name}/keyframes/kf_*_final.npz, C/N > {CLOUD_C_THRESHOLD}",
        ))
        record["files"][f"cloud_m_{v}"] = str(out)
    n_points = int(sum(len(kc.X) for kc in kclouds))

    # --- tabela klatek --------------------------------------------------- #
    med, cnt = fx.frame_medians(obs), fx.frame_counts(obs)
    krows = [{"kf": k, "pts_s": float(pts[k]), "n_par": cnt.get(k, 0), "f_obs": med.get(k),
              "f_interp": float(f_vec["kf_interp"][k]), "f_mnk": float(lsq.f[k]),
              "sigma_u_mnk": float(lsq.sigma_u[k]), "f_global": f_glob} for k in range(K)]
    write_csv(run_dir / KEYFRAMES_CSV, KEYFRAME_COLUMNS, krows)
    write_csv(per_pair, COLUMNS, rows)
    record["files"]["scale_keyframes"] = str(run_dir / KEYFRAMES_CSV)

    # --- rozrzut: ostrzeżenie, bez zmiany liczb --------------------------- #
    spread_obs = _spread_pct([o.f for o in obs])
    for what, value in (("współczynników odcinków (wariant globalny)", est.summary.spread_pct),
                        ("f_kj wszystkich par", spread_obs)):
        if value > SPREAD_WARN_PCT:
            record["warnings"].append(f"rozrzut {what} {value:.2f} % > {SPREAD_WARN_PCT:.0f} % — "
                                      f"liczby bez zmian, outliery nieodrzucane")

    def shift(v):
        return float(np.max(np.linalg.norm(pos[v] - pos["global"], axis=1)))

    record.update({
        "status": STATUS_OK,
        "global": {"f": f_glob, "how": how, "estimate": est.as_dict()},
        "kf_interp": {"f_min": float(f_vec["kf_interp"].min()), "f_max": float(f_vec["kf_interp"].max()),
                      "max_shift_vs_global_m": shift("kf_interp")},
        "kf_mnk": {"f_min": float(lsq.f.min()), "f_max": float(lsq.f.max()),
                   "max_shift_vs_global_m": shift("kf_mnk"),
                   "sigma0_prior": 1.0, "sigma0_post": lsq.sigma0_post, "dof": lsq.dof,
                   "n_obs_eq": lsq.n_obs, "n_eq": lsq.n_eq, "vtpv": lsq.vtpv},
        "spread_obs_pct": spread_obs,
        "rebuild": rebuild | {"n_points_written": n_points},
        "keyframes": krows,
        "t_s": time.perf_counter() - t0,
    })
    (run_dir / SCALE_JSON).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    record["files"]["scale_json"] = str(run_dir / SCALE_JSON)
    for w in record["warnings"]:
        log(f"UWAGA: {w}")
    return record


def _num(v, nd=4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def format_report(record: dict) -> str:
    """Tabela f_k (A, B ± błąd, globalny) i liczby z scale.json — bez wniosków."""
    lines = [f"{record['run_dir']}  ({record.get('recording', '?')})",
             f"status: {record['status']}  —  klatek kluczowych {record['n_keyframes']}, "
             f"z obserwacjami {record['n_keyframes_with_obs']}, par ok {record['n_pairs_ok']} / {record['n_pairs']}"]
    if record["status"] != STATUS_OK:
        lines.append(record.get("message", ""))
        return "\n".join(lines)
    p = record["params"]
    lines.append(f"σ_obs = {p['sigma_obs']['value']:.6g} ({p['sigma_obs']['source']}), "
                 f"τ = {p['tau']['value']:.6g} ({p['tau']['source']})")
    lines += ["", "| kf | pts [s] | n par | f obs (mediana) | f A (interp) | f B (MNK) ± σ_u | f global |",
              "|---|---|---|---|---|---|---|"]
    for r in record["keyframes"]:
        lines.append(f"| {r['kf']} | {r['pts_s']:.3f} | {r['n_par']} | {_num(r['f_obs'])} | {_num(r['f_interp'])} "
                     f"| {_num(r['f_mnk'])} ± {_num(r['sigma_u_mnk'])} | {_num(r['f_global'])} |")
    b = record["kf_mnk"]
    lines += ["",
              f"f_global = {record['global']['f']:.6f} ({record['global']['how']})",
              f"maks. przesunięcie położenia klatki względem globalnego: A {record['kf_interp']['max_shift_vs_global_m']:.4f} m, "
              f"B {b['max_shift_vs_global_m']:.4f} m",
              f"MNK: σ0 a posteriori {_num(b['sigma0_post'])}, stopni swobody {b['dof']}",
              f"odbudowa: {record['rebuild']['n_rebuilt']} pkt (cloud.ply {record['rebuild']['n_upstream']}), "
              f"maks. różnica {record['rebuild']['max_abs_diff']}, kolory: {record['rebuild']['colors']}",
              f"ostrzeżeń: {len(record['warnings'])} (w logu powyżej i w scale.json)",
              f"czas {record['t_s']:.1f} s"]
    return "\n".join(lines)
