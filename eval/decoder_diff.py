"""Ta sama klatka przez upstreamowy MP4Dataset i przez FileSource: różnica w `frame.img`.

MP4Dataset (env bez torchcodec) dekoduje `cv2.VideoCapture` z seekiem na każdą klatkę
(`dataloader.py:257`), FileSource — PyAV sekwencyjnie. Obie ścieżki dają float32 0..1,
które idzie do `create_frame()` upstreamu. Porównujemy `frame.img` (wejście sieci po
ImgNorm, zakres -1..1) i dla kontekstu surowe RGB w pełnej rozdzielczości (poziomy 0..255).
Dla klatki i porównujemy MP4Dataset[i] z FileSource[i-1], [i], [i+1] — gdyby seek trafiał
w sąsiednią klatkę, widać to od razu.

Env `mast3r-slam`, bez GPU (create_frame na CPU). Liczby się zapisuje, nie interpretuje.

    python eval/decoder_diff.py data/2026-09-21/video/VID_20260921_125437.mp4 --frames 0 500 \\
        --out runs/2026-09-21/adapter/decoder_diff_125437.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vid2cloud.engine.mast3r_slam import DEFAULT_CONFIG, IMG_SIZE  # noqa: E402  (ładuje upstream)

import lietorch  # noqa: E402
from mast3r_slam.config import load_config  # noqa: E402
from mast3r_slam.dataloader import MP4Dataset  # noqa: E402
from mast3r_slam.frame import create_frame  # noqa: E402

from vid2cloud.input.sources import FileSource  # noqa: E402

#: frame.img = (x - 0.5) / 0.5, więc 1 poziom uint8 to 2/255 w jednostkach frame.img.
IMG_UNITS_PER_LEVEL = 2.0 / 255.0


def model_input(img01: np.ndarray, index: int) -> np.ndarray:
    frame = create_frame(index, img01, lietorch.Sim3.Identity(1, device="cpu"), img_size=IMG_SIZE, device="cpu")
    return frame.img.numpy().astype(np.float64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 500])
    parser.add_argument("--out", type=Path, default=None, help="CSV z wynikami")
    args = parser.parse_args(argv)

    load_config(str(DEFAULT_CONFIG))
    dataset = MP4Dataset(str(args.video))
    wanted = sorted({j for i in args.frames for j in (i - 1, i, i + 1) if j >= 0})

    pyav: dict[int, np.ndarray] = {}
    for index, _pts, rgb in FileSource(args.video):
        if index in wanted:
            pyav[index] = rgb
        if index >= wanted[-1]:
            break

    rows = []
    for i in args.frames:
        # get_image = __getitem__ bez znacznika czasu (lista timestamps rośnie tylko przy
        # odczycie po kolei, dataloader.py:263); seek CAP_PROP_POS_FRAMES jest ten sam.
        up01 = dataset.get_image(i)  # float32 0..1, dataloader.py:46-50
        up_img = model_input(up01, i)
        up_rgb = up01.astype(np.float64) * 255.0
        for j in (i - 1, i, i + 1):
            if j not in pyav:
                continue
            fs01 = pyav[j].astype(np.float32) / 255.0
            fs_img = model_input(fs01, j)
            if fs01.shape != up01.shape:
                raise ValueError(f"kształty różne: MP4Dataset {up01.shape}, FileSource {fs01.shape}")
            d_img = np.abs(up_img - fs_img)
            d_rgb = np.abs(up_rgb - pyav[j].astype(np.float64))
            rows.append({
                "mp4dataset_frame": i,
                "filesource_frame": j,
                "full_shape": "x".join(map(str, up01.shape)),
                "img_shape": "x".join(map(str, up_img.shape)),
                "img_mean_abs": d_img.mean(),
                "img_max_abs": d_img.max(),
                "img_mean_abs_levels": d_img.mean() / IMG_UNITS_PER_LEVEL,
                "img_max_abs_levels": d_img.max() / IMG_UNITS_PER_LEVEL,
                "rgb_full_mean_abs": d_rgb.mean(),
                "rgb_full_max_abs": d_rgb.max(),
            })

    cols = list(rows[0])
    print("  ".join(cols))
    for r in rows:
        print("  ".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"zapisano {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
