"""Przebieg `vid2cloud run`: FileSource -> silnik -> zrzuty, log per klatka, ArUco per klatka kluczowa.

To jest jedyne miejsce, które łączy silnik (`engine/`) z modułem skali (`scale/`):
silnik nie importuje scale/, scale/ nie importuje engine/.

Pliki w katalogu przebiegu (`--out`):
    config.json           — źródło (ffprobe), commit upstreamu, sha256 configu, czasy całkowite
    log.csv               — wiersz na każdą klatkę (export/framelog.py)
    aruco_keyframes.csv   — markery na pełnej klatce każdej nowej klatki kluczowej,
                            narożniki w pełnej rozdzielczości i przeliczone na wejście modelu
    poses_history.csv, trajectory_sim3.txt, cloud.ply, keyframes/  — z silnika

Narożniki przeliczamy transformacją podaną przez silnik (`engine.input_transform()`,
liczoną z `resize_img()` upstreamu) w konwencji środka piksela z `scale/aruco.py`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vid2cloud.export.framelog import FrameLogWriter, depth_stats
from vid2cloud.input.sources import FileSource
from vid2cloud.scale import aruco

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Próg „niskiej ufności" w logu = próg chmury (`WindowMsg.C_conf_threshold`, 1.5).
LOW_CONF_THRESHOLD = 1.5
#: Ile ostatnich pełnych klatek trzymać dla ArUco. Przy single_thread nowa klatka kluczowa
#: to zawsze bieżąca klatka; bez single_thread relokalizacja w backendzie może się spóźnić.
FULL_FRAME_CACHE = 8

ARUCO_COLUMNS = (
    "kf", "frame_index", "id",
    *(f"{c}{i}" for i in range(4) for c in ("x", "y")),
    *(f"{c}{i}_512" for i in range(4) for c in ("x", "y")),
    "side_px_512", "error",
)


def ffprobe(path: Path) -> dict:
    """Parametry strumienia wideo z ffprobe (z liczeniem klatek przez dekodowanie)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,nb_read_frames,avg_frame_rate,r_frame_rate,"
        "time_base,start_time,duration:stream_side_data=rotation",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream = json.loads(out.stdout)["streams"][0]
    rotation = 0
    for sd in stream.pop("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    stream["rotation"] = rotation
    for key in ("width", "height", "nb_frames", "nb_read_frames"):
        if key in stream:
            stream[key] = int(stream[key])
    for key in ("start_time", "duration"):
        if key in stream:
            stream[key] = float(stream[key])
    return stream


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_state(path: Path) -> dict:
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def ply_vertex_count(path: Path) -> int:
    """Liczba punktów z nagłówka PLY."""
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"{path}: brak 'element vertex' w nagłówku")


def aruco_rows(kf: int, frame_index: int, rgb: np.ndarray, tf: aruco.InputTransform) -> list[list]:
    """Wiersze `aruco_keyframes.csv` dla jednej klatki kluczowej (pełna klatka RGB)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    try:
        detections = aruco.detect(gray)
    except ValueError as exc:  # duplikat ID — zapisujemy, nie rozstrzygamy
        return [[kf, frame_index, "", *([""] * 17), str(exc)]]
    if not detections:
        return [[kf, frame_index, "", *([""] * 17), ""]]
    rows = []
    for marker_id, quad in sorted(detections.items()):
        q512 = tf.apply(quad)
        side = float(np.mean([np.linalg.norm(q512[i] - q512[(i + 1) % 4]) for i in range(4)]))
        rows.append([
            kf, frame_index, marker_id,
            *(f"{v:.4f}" for v in quad.reshape(-1)),
            *(f"{v:.4f}" for v in q512.reshape(-1)),
            f"{side:.4f}", "",
        ])
    return rows


def run_video(
    video: Path,
    out: Path,
    config_path: Optional[Path] = None,
    single_thread: bool = True,
    max_frames: Optional[int] = None,
    progress_every: int = 100,
) -> dict:
    """Pełny przebieg nagrania przez MASt3R-SLAM. Zwraca słownik zapisany jako config.json."""
    t_start = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    video = Path(video).resolve()
    out = Path(out).resolve()
    for name in ("log.csv", "cloud.ply", "config.json"):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} już istnieje — nie nadpisuję wyników przebiegu")
    out.mkdir(parents=True, exist_ok=True)

    # Ciężki import (torch, upstream) dopiero tutaj — `vid2cloud --help` ma działać bez GPU.
    from vid2cloud.engine import mast3r_slam as m3

    config_path = Path(config_path).resolve() if config_path else m3.DEFAULT_CONFIG
    probe = ffprobe(video)
    engine = m3.Mast3rSlamEngine(out, config_path=config_path, single_thread=single_thread)

    record: dict = {
        "command": sys.argv,
        "started_at": started_at,
        "video": str(video),
        "out": str(out),
        "source": {"ffprobe": probe},
        "engine": {
            "name": engine.name,
            "upstream": m3.upstream_commit(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "config": dict(m3.config),
            "checkpoint": str(engine.checkpoint),
            "single_thread": single_thread,
            "cloud_c_conf_threshold": m3.CLOUD_C_CONF_THRESHOLD,
        },
        "vid2cloud": _git_state(REPO_ROOT),
        "env": {
            "python": platform.python_version(),
            "torch": m3.torch.__version__,
            "gpu": m3.torch.cuda.get_device_name(0) if m3.torch.cuda.is_available() else None,
        },
        "max_frames": max_frames,
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "status": "running",
    }
    _write_json(out / "config.json", record)  # nadpisywany na końcu; przerwany przebieg też go ma

    source = FileSource(video)
    log = FrameLogWriter(out / "log.csv")
    f_aruco = open(out / "aruco_keyframes.csv", "w", newline="", encoding="utf-8")
    w_aruco = csv.writer(f_aruco)
    w_aruco.writerow(ARUCO_COLUMNS)

    cache: OrderedDict[int, np.ndarray] = OrderedDict()
    tf_aruco: Optional[aruco.InputTransform] = None
    t_decode = t_engine = t_aruco = 0.0
    n_frames = 0
    pts_first = pts_last = None
    missing_full_frames: list[int] = []
    t_loop = time.perf_counter()

    it = iter(source)
    while max_frames is None or n_frames < max_frames:
        t0 = time.perf_counter()
        try:
            index, pts_s, rgb = next(it)
        except StopIteration:
            break
        dt_decode = time.perf_counter() - t0

        if tf_aruco is None:
            h, w = rgb.shape[:2]
            tf_engine = engine.input_transform(h, w)
            tf_aruco = aruco.InputTransform(**asdict(tf_engine))
            record["input_transform"] = asdict(tf_engine) | {"full_shape": [h, w]}
            engine.start(h, w)
            t_loop = time.perf_counter()

        t1 = time.perf_counter()
        res = engine.push_frame(rgb, index, pts_s)
        dt_engine = time.perf_counter() - t1

        cache[index] = rgb
        while len(cache) > FULL_FRAME_CACHE:
            cache.popitem(last=False)
        t2 = time.perf_counter()
        for kf, frame_index in res.new_keyframes:
            full = cache.get(frame_index)
            if full is None:
                missing_full_frames.append(frame_index)
                w_aruco.writerow([kf, frame_index, "", *([""] * 17), "pełna klatka poza buforem"])
                continue
            w_aruco.writerows(aruco_rows(kf, frame_index, full, tf_aruco))
        if res.new_keyframes:
            f_aruco.flush()
        t_aruco += time.perf_counter() - t2

        median, p95, low = depth_stats(res.pointmap, res.conf, LOW_CONF_THRESHOLD)
        log.write({
            "index": index,
            "pts_s": f"{pts_s:.6f}",
            "t_decode_ms": 1000.0 * dt_decode,
            "t_engine_ms": 1000.0 * dt_engine,
            "t_backend_wait_ms": res.t_backend_wait_ms,
            "is_kf": res.is_keyframe,
            "kf_index": res.kf_index,
            "mode": res.mode,
            "n_kf": res.n_kf,
            "depth_median": median,
            "depth_p95": p95,
            "low_conf_frac": low,
        })

        t_decode += dt_decode
        t_engine += dt_engine
        n_frames += 1
        pts_first = pts_s if pts_first is None else pts_first
        pts_last = pts_s
        if progress_every and n_frames % progress_every == 0:
            elapsed = time.perf_counter() - t_loop
            print(f"  klatka {n_frames}/{probe.get('nb_read_frames', '?')}  kf={res.n_kf}  "
                  f"{n_frames / elapsed:.2f} kl/s", flush=True)

    log.close()
    f_aruco.close()
    if n_frames == 0:
        raise ValueError(f"{video}: nie zdekodowano żadnej klatki")

    paths = engine.finalize()
    wall = time.perf_counter() - t_start
    record.update({
        "status": "done",
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": record["source"] | {
            "decoded_frames": n_frames,
            "rotation_applied": source.rotation,
            "pts_first_s": pts_first,
            "pts_last_s": pts_last,
            "decoder": "PyAV (FileSource), sekwencyjnie",
        },
        "times_s": {
            "wall_total": wall,
            "setup": engine.timings.get("setup_s"),
            "decode_total": t_decode,
            "engine_total": t_engine,
            "aruco_total": t_aruco,
            "finalize": engine.timings.get("finalize_s"),
        },
        "results": {
            "n_frames": n_frames,
            "n_keyframes": len(list(Path(paths["keyframes"]).glob("kf_*_final.npz"))),
            "n_points_cloud": ply_vertex_count(paths["cloud"]),
            "poses_history_events": engine.history.n_events,
            "missing_full_frames_for_aruco": missing_full_frames,
            **engine.counts,
        },
        "files": {k: str(v) for k, v in paths.items()} | {
            "log": str(out / "log.csv"),
            "aruco_keyframes": str(out / "aruco_keyframes.csv"),
        },
    })
    _write_json(out / "config.json", record)
    return record
