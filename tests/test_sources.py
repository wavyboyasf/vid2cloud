"""FileSource na krótkim klipie: rotacja z metadanych, PTS, liczba klatek.

Klip: pierwsze 2 s z VID_20260921_125536 (rotation=-90 w display matrix), wycięte
`ffmpeg -i ... -map 0:v:0 -t 2 -c copy data/clips/VID_20260921_125536_0-2s.mp4`.
Start od 0 s, więc copy nie dokłada pre-rollu. Bez klipu (data/ jest w .gitignore) test
jest pomijany.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av")

from vid2cloud.input.sources import FileSource  # noqa: E402

CLIP = Path(__file__).resolve().parents[1] / "data" / "clips" / "VID_20260921_125536_0-2s.mp4"

pytestmark = [
    pytest.mark.skipif(not CLIP.exists(), reason=f"brak klipu {CLIP}"),
    pytest.mark.skipif(shutil.which("ffprobe") is None, reason="brak ffprobe"),
]


def _ffprobe_frames(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,nb_frames,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["streams"][0]


@pytest.fixture(scope="module")
def decoded():
    source = FileSource(CLIP)
    items = [(index, pts, rgb.shape, rgb.dtype) for index, pts, rgb in source]
    return source, items


def test_frame_count_matches_ffprobe(decoded):
    source, items = decoded
    probe = _ffprobe_frames(CLIP)
    assert len(items) == int(probe["nb_read_frames"]) == int(probe["nb_frames"])
    assert source.n_frames == len(items)
    assert [i for i, *_ in items] == list(range(len(items)))


def test_portrait_after_rotation(decoded):
    source, items = decoded
    probe = _ffprobe_frames(CLIP)
    assert (probe["width"], probe["height"]) == (3840, 2160)  # w pliku leży poziomo
    assert source.rotation == -90
    for _, _, shape, dtype in items:
        assert shape == (3840, 2160, 3)  # po obrocie: pion, H > W
        assert dtype == np.uint8


def test_pts_strictly_increasing(decoded):
    _, items = decoded
    pts = np.array([p for _, p, *_ in items])
    assert np.all(np.isfinite(pts))
    assert np.all(np.diff(pts) > 0)


def test_no_rotation_flag_keeps_file_layout():
    source = FileSource(CLIP, rotate=False)
    index, pts, rgb = next(iter(source))
    assert index == 0
    assert rgb.shape == (2160, 3840, 3)
    assert source.rotation == 0
