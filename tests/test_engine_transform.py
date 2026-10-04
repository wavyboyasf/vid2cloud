"""InputTransform z adaptera (liczony `resize_img()` upstreamu) == reimplementacja w scale/aruco.py.

Wymaga upstreamu (torch, mast3r_slam), więc działa tylko w env `mast3r-slam`;
w `vid2cloud-tools` jest pomijany. GPU niepotrzebne — `resize_img` to PIL.
Jeśli test padnie, to rozjazd trzeba ZGŁOSIĆ, a nie poprawiać aruco.py pod wynik.
"""

from dataclasses import asdict

import pytest

pytest.importorskip("torch")
pytest.importorskip("mast3r_slam")

from vid2cloud.engine.mast3r_slam import upstream_input_transform  # noqa: E402
from vid2cloud.scale.aruco import input_transform  # noqa: E402


@pytest.mark.parametrize("full_hw", [(2160, 3840), (3840, 2160)], ids=["3840x2160", "2160x3840"])
def test_adapter_transform_equals_aruco(full_hw):
    engine_tf = upstream_input_transform(*full_hw)
    aruco_tf = input_transform(full_hw)
    for key, value in asdict(aruco_tf).items():
        got = getattr(engine_tf, key)
        if isinstance(value, tuple):
            assert tuple(got) == value, key
        else:
            assert got == pytest.approx(value, abs=1e-9, rel=0), key
