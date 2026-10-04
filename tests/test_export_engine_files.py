"""Formaty plików przebiegu: zrzuty klatek kluczowych (.npz), poses_history.csv,
trajectory_sim3.txt i log.csv. Funkcje czyste, bez silnika i bez GPU."""

import csv

import numpy as np
import pytest

from vid2cloud.export.framelog import FRAME_LOG_COLUMNS, FrameLogWriter, depth_stats
from vid2cloud.export.keyframes import (
    KEYFRAME_NPZ_KEYS,
    keyframe_npz_path,
    read_keyframe_npz,
    write_keyframe_npz,
)
from vid2cloud.export.trajectory import (
    POSES_HISTORY_COLUMNS,
    PosesHistoryWriter,
    read_poses_history,
    read_trajectory_sim3,
    write_trajectory_sim3,
)

H, W = 288, 512


def _snapshot(rng, **over):
    snap = dict(
        kf=3,
        frame_index=417,
        pts_s=13.9031,
        source="tracker",
        T_WC=np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 1.25], dtype=np.float32),
        X_canon=rng.normal(size=(H, W, 3)).astype(np.float32),
        C=rng.uniform(1, 20, size=(H, W)).astype(np.float32),
        N=4,
        N_updates=5,
        img_true_shape=(H, W),
    )
    snap.update(over)
    return snap


def test_keyframe_npz_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    snap = _snapshot(rng)
    path = write_keyframe_npz(keyframe_npz_path(tmp_path / "keyframes", 3, "created"), **snap)
    assert path.name == "kf_003_created.npz"

    back = read_keyframe_npz(path)
    assert tuple(back) == KEYFRAME_NPZ_KEYS
    assert back["kf"] == 3 and back["frame_index"] == 417 and back["N"] == 4 and back["N_updates"] == 5
    assert back["source"] == "tracker"
    assert back["pts_s"] == pytest.approx(13.9031)
    assert back["T_WC"].shape == (8,) and back["T_WC"].dtype == np.float64
    np.testing.assert_array_equal(back["T_WC"], snap["T_WC"].astype(np.float64))  # float32 -> float64 bez straty
    assert back["X_canon"].shape == (H, W, 3) and back["X_canon"].dtype == np.float32
    assert back["C"].shape == (H, W) and back["C"].dtype == np.float32
    np.testing.assert_array_equal(back["X_canon"], snap["X_canon"])
    np.testing.assert_array_equal(back["C"], snap["C"])
    assert tuple(back["img_true_shape"]) == (H, W)


@pytest.mark.parametrize(
    "over",
    [
        {"X_canon": np.zeros((H * W, 3), np.float32)},  # płaska, nie (h, w, 3)
        {"C": np.zeros((H, W, 1), np.float32)},
        {"T_WC": np.zeros(7)},
        {"source": "loop"},
    ],
)
def test_keyframe_npz_rejects_bad_input(tmp_path, over):
    with pytest.raises(ValueError):
        write_keyframe_npz(tmp_path / "x.npz", **_snapshot(np.random.default_rng(1), **over))


def test_keyframe_npz_path_stage():
    assert keyframe_npz_path("kfs", 12, "final").name == "kf_012_final.npz"
    with pytest.raises(ValueError):
        keyframe_npz_path("kfs", 0, "after_ga")


def test_poses_history_columns_and_rows(tmp_path):
    path = tmp_path / "poses_history.csv"
    writer = PosesHistoryWriter(path)
    p1 = np.array([[0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)
    p2 = np.array([[0, 0, 0, 0, 0, 0, 1, 1], [0.5, 0.1, -0.2, 0.0, 0.1, 0.0, 0.99498744, 1.0731]], dtype=np.float32)
    assert writer.write_event("kf_created", 0, p1) == 0
    assert writer.write_event("kf_created", 57, p2) == 1
    assert writer.write_event("backend_done", 57, p2) == 2
    assert writer.write_event("final", 99, p2) == 3
    with pytest.raises(ValueError):
        writer.write_event("ga", 99, p2)
    writer.close()

    with open(path, newline="") as f:
        header = next(csv.reader(f))
    assert tuple(header) == POSES_HISTORY_COLUMNS
    assert header == ["event_id", "event", "frame_index", "n_kf", "kf",
                      "tx", "ty", "tz", "qx", "qy", "qz", "qw", "s"]

    rows = read_poses_history(path)
    assert len(rows) == 1 + 2 + 2 + 2
    assert [r["event_id"] for r in rows] == [0, 1, 1, 2, 2, 3, 3]
    assert [r["kf"] for r in rows] == [0, 0, 1, 0, 1, 0, 1]
    assert [r["n_kf"] for r in rows] == [1, 2, 2, 2, 2, 2, 2]
    last = rows[-1]
    assert last["event"] == "final" and last["frame_index"] == 99
    # %.9g odtwarza float32 bez straty
    assert np.float32(last["s"]) == p2[1, 7]
    assert np.float32(last["qw"]) == p2[1, 6]


def test_trajectory_sim3_roundtrip(tmp_path):
    poses = np.array([[0, 0, 0, 0, 0, 0, 1, 1], [1.5, -2.25, 0.125, 0.1, 0.2, 0.3, 0.9273618, 0.8125]],
                     dtype=np.float32)
    pts = [0.0243, 9.3]
    path = write_trajectory_sim3(tmp_path / "trajectory_sim3.txt", pts, poses)
    first = path.read_text().splitlines()[0]
    assert first == "# pts_s tx ty tz qx qy qz qw s"
    t, back = read_trajectory_sim3(path)
    np.testing.assert_allclose(t, pts)
    np.testing.assert_array_equal(back.astype(np.float32), poses)
    with pytest.raises(ValueError):
        write_trajectory_sim3(tmp_path / "bad.txt", [0.0], poses)


def test_frame_log_columns(tmp_path):
    path = tmp_path / "log.csv"
    log = FrameLogWriter(path)
    log.write({"index": 0, "pts_s": "0.024300", "is_kf": True, "kf_index": 0, "mode": "INIT", "n_kf": 1,
               "t_decode_ms": 12.5, "t_engine_ms": 300.25, "t_backend_wait_ms": 0.0,
               "depth_median": 2.0, "depth_p95": 3.5, "low_conf_frac": 0.1})
    log.write({"index": 1, "pts_s": "0.057600", "is_kf": False, "kf_index": None, "mode": "TRACKING", "n_kf": 1})
    with pytest.raises(ValueError):
        log.write({"index": 2, "sharpness": 1.0})
    log.close()
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert tuple(rows[0]) == FRAME_LOG_COLUMNS
    assert rows[1][FRAME_LOG_COLUMNS.index("is_kf")] == "1"
    assert rows[2][FRAME_LOG_COLUMNS.index("kf_index")] == ""
    assert rows[2][FRAME_LOG_COLUMNS.index("depth_median")] == ""


def test_depth_stats():
    z = np.arange(1, 101, dtype=np.float32)  # 1..100
    pointmap = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    conf = np.where(z <= 25, 1.0, 2.0)
    median, p95, low = depth_stats(pointmap, conf, 1.5)
    assert median == pytest.approx(50.5)
    assert p95 == pytest.approx(np.percentile(z, 95))
    assert low == pytest.approx(0.25)
    assert depth_stats(None, None, 1.5) == (None, None, None)
