"""Skala per klatka kluczowa: współczynnik każdego odcinka taśmy w każdej klatce kluczowej.

Przeniesione 2026-11-04 z `eval/scale_per_keyframe.py` (rysunki i porównanie z pomiarem
ręcznym zostały w eval/), żeby z tych samych funkcji korzystały `vid2cloud scale`,
`vid2cloud run --targets` i skrypty ewaluacyjne. Bez matplotlib — działa w obu env.

Dla każdej klatki kluczowej k i każdego odcinka z targets.json, którego OBA markery są w niej
wykryte (`aruco_keyframes.csv`), punkty odniesienia (pole `reference`) są liftowane do 3D
z pointmapy klatki k (`scale/lift.py`) w dwóch stanach — przy powstaniu (`kf_k_created.npz`)
i na końcu (`kf_k_final.npz`). d_canon to odległość punktów w układzie kamery klatki k
(jednostki sieci), s_k — skala Sim(3) klatki k z `poses_history.csv`. Współczynniki w metrach
na jednostkę chmury (jak `scale/estimate.py`), L = length_mm / 1000:

    factor_net_created = L / d_canon_created                 (sama sieć, bez grafu)
    factor_online      = L / (s_created · d_canon_created)   (chmura w chwili powstania kf)
    factor_final       = L / (s_final · d_canon_final)       (chmura na końcu, cloud.ply)

Diagnostyka: d z mediany w oknie 3×3 zamiast płaszczyzny + promienia i różnica
`diff_win3_mm = L_mm · (d_win3 / d − 1)` — bez progu.

Głębokość odcinka (test E): z_a_canon, z_b_canon — współrzędna z końców odcinka w układzie
kamery klatki k (jednostki sieci), z_mid_canon — z środka odcinka; wszystkie ze stanu przy
powstaniu, czyli z tych samych punktów co d_canon_created i factor_net_created. Na metry:
z_mid_m = z_mid_canon · mediana(factor_net_created) WSZYSTKICH odcinków przebiegu
(kolumna net_cr_median_run) — nie współczynnikiem tej samej pary, bo wtedy z_mid_m i
factor_net_created byłyby zależne z definicji.

Wejście: tylko przebiegi z single_thread = true (zrzuty odpowiadają zdarzeniom, nie tempu GPU).
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from vid2cloud.export.keyframes import read_keyframe_npz
from vid2cloud.export.trajectory import read_poses_history
from vid2cloud.scale.estimate import MM_PER_M, Segment, Targets
from vid2cloud.scale.lift import (
    C_MIN,
    ReferencePoint,
    lift_marker,
    lift_reference,
    mean_confidence,
    parse_reference,
    to_world,
)

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
    "z_a_canon", "z_b_canon", "z_mid_canon", "net_cr_median_run", "z_mid_m",
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


def segments_with_refs(targets: Targets) -> tuple[list[Segment], dict[str, tuple[ReferencePoint, ReferencePoint]]]:
    """Odcinki między dwoma markerami + ich punkty odniesienia.

    Każdy `reference` parsowany z góry: nieobsługiwany to błąd przed liczeniem, nie pominięcie.
    """
    segments = [s for s in targets.segments.values() if s.marker_a is not None and s.marker_b is not None]
    refs = {s.name: parse_reference(s.reference, s.marker_a, s.marker_b) for s in segments}
    return segments, refs


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
                if st == "created":  # głębokość z tych samych punktów co factor_net_created
                    row.update({"z_a_canon": float(la.point[2]), "z_b_canon": float(lb.point[2]),
                                "z_mid_canon": float((la.point[2] + lb.point[2]) / 2)})
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
    return add_depth_m(rows)


def add_depth_m(rows: list[dict]) -> list[dict]:
    """z_mid_m = z_mid_canon · mediana factor_net_created wszystkich wierszy przebiegu (nie tej pary)."""
    net = [r["factor_net_created"] for r in rows if r.get("factor_net_created") is not None]
    med = float(np.median(net)) if net else None
    for r in rows:
        r["net_cr_median_run"] = med
        if med is not None and r.get("z_mid_canon") is not None:
            r["z_mid_m"] = r["z_mid_canon"] * med
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


def csv_text(columns, rows: list[dict]) -> str:
    """Zawartość CSV jako tekst (ten sam format co `write_csv`) — do porównania z plikiem na dysku."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(columns)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in columns])
    return buf.getvalue()


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
