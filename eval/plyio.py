"""Cienki import: PLY bez open3d żyje w `vid2cloud/export/plyio.py` (od 2026-11-04).

Skrypty z eval/ robią `from plyio import ...` z eval/ na sys.path — ten plik zostaje,
żeby ich importy działały bez zmian.
"""

import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vid2cloud.export.plyio import read_ply, write_ply, xyz, xyz_vertices  # noqa: E402,F401
