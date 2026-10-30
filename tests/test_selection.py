"""Miary jakości klatki i dobór klatek (`vid2cloud/input/selection.py`) na syntetyce.

Funkcje czyste, bez GPU i bez nagrań. Wyjątek: `test_model_input_matches_upstream` porównuje
obraz wejścia sieci z `resize_img()` upstreamu piksel w piksel — potrzebuje `mast3r_slam`
(env `mast3r-slam`), w `vid2cloud-tools` jest pomijany. Oczekiwane liczby są policzone
ręcznie w komentarzach, nie przepisane z wyjścia kodu.
"""

import cv2
import numpy as np
import pytest

from vid2cloud.input import selection as sel

SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _checkerboard(h: int, w: int, square: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy // square) + (xx // square)) % 2 * 255).astype(np.uint8)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


# --------------------------------------------------------------------------- #
# Ostrość
# --------------------------------------------------------------------------- #


def test_sharpness_decreases_with_blur_at_model_resolution():
    """Szachownica 512×288, pola 16 px, rozmywana Gaussem: obie miary maleją ściśle."""
    board = _checkerboard(288, 512, 16)
    ten = [sel.tenengrad(_blur(board, s)) for s in SIGMAS]
    lap = [sel.laplacian_var(_blur(board, s)) for s in SIGMAS]
    assert all(a > b for a, b in zip(ten, ten[1:])), ten
    assert all(a > b for a, b in zip(lap, lap[1:])), lap


def test_sharpness_decreases_with_blur_through_model_input():
    """Ta sama własność na pełnej ścieżce: klatka 4K RGB rozmyta w pełnej rozdzielczości
    -> `model_input_gray()` (512×288) -> miary. Pola 120 px w 4K = 16 px po skalowaniu."""
    board = _checkerboard(2160, 3840, 120)
    ten, lap = [], []
    for s in (0.0, 4.0, 8.0, 16.0, 32.0):
        rgb = np.repeat(_blur(board, s)[:, :, None], 3, axis=2)
        gray = sel.model_input_gray(rgb)
        assert gray.shape == (288, 512)
        ten.append(sel.tenengrad(gray))
        lap.append(sel.laplacian_var(gray))
    assert all(a > b for a, b in zip(ten, ten[1:])), ten
    assert all(a > b for a, b in zip(lap, lap[1:])), lap


def test_sharpness_of_flat_image_is_zero():
    flat = np.full((64, 64), 128, dtype=np.uint8)
    assert sel.tenengrad(flat) == 0.0
    assert sel.laplacian_var(flat) == 0.0


def test_tenengrad_of_ramp():
    """Rampa w poziomie o nachyleniu 1/px: Sobel x = 8 w środku obrazu (jądro [1,2,1]ᵀ×[−1,0,1]
    daje 2·4·1), Sobel y = 0. Brzeg REFLECT_101 zeruje Gx w pierwszej i ostatniej kolumnie,
    więc średnia Gx² = 64 · (W − 2) / W."""
    w = 100
    ramp = np.tile(np.arange(w, dtype=np.float32), (20, 1))
    assert sel.tenengrad(ramp) == pytest.approx(64.0 * (w - 2) / w, rel=1e-6)


# --------------------------------------------------------------------------- #
# Ekspozycja
# --------------------------------------------------------------------------- #


def test_exposure_dark_and_bright():
    dark = sel.exposure(np.full((10, 10), 2, dtype=np.uint8))
    assert dark == {"bright_median": 2.0, "frac_dark": 1.0, "frac_bright": 0.0}
    bright = sel.exposure(np.full((10, 10), 253, dtype=np.uint8))
    assert bright == {"bright_median": 253.0, "frac_dark": 0.0, "frac_bright": 1.0}


def test_exposure_half_and_half_and_bounds():
    """Połowa 0, połowa 255: mediana (0 + 255) / 2 = 127,5. Wartości 5 i 250 nie liczą się
    ani jako ciemne, ani jako jasne (nierówności ostre)."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:, 5:] = 255
    assert sel.exposure(img) == {"bright_median": 127.5, "frac_dark": 0.5, "frac_bright": 0.5}
    edge = np.array([[5, 250]], dtype=np.uint8)
    e = sel.exposure(edge)
    assert e["frac_dark"] == 0.0 and e["frac_bright"] == 0.0


def test_exposure_rejects_float():
    with pytest.raises(ValueError):
        sel.exposure(np.zeros((4, 4), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Stały krok
# --------------------------------------------------------------------------- #


def test_fixed_step_offsets():
    assert sel.fixed_step(10, 3, 0) == [0, 3, 6, 9]
    assert sel.fixed_step(10, 3, 1) == [1, 4, 7]
    assert sel.fixed_step(10, 3, 2) == [2, 5, 8]
    assert sel.fixed_step(10, 1, 0) == list(range(10))
    assert sel.fixed_step(0, 3, 0) == []


@pytest.mark.parametrize("n, counts", [(987, (329, 329, 329)), (1376, (459, 459, 458)), (1290, (430, 430, 430))])
def test_fixed_step_counts_for_recordings(n, counts):
    """ceil((n − o) / 3): 987 -> 329/329/329, 1376 -> 459/459/458, 1290 -> 430/430/430."""
    assert tuple(len(sel.fixed_step(n, 3, o)) for o in range(3)) == counts


def test_fixed_step_rejects_bad_offset():
    with pytest.raises(ValueError):
        sel.fixed_step(10, 3, 3)
    with pytest.raises(ValueError):
        sel.fixed_step(10, 0, 0)


# --------------------------------------------------------------------------- #
# Okna
# --------------------------------------------------------------------------- #

SCORES = [1, 5, 2, 7, 3, 9, 4, 4, 0, 8]  # okna k=3: [0,3) [3,6) [6,9) [9,10)


def test_select_in_windows_argmax_and_partial_last_window():
    idx, rep = sel.select_in_windows(SCORES, 3)
    # [1,5,2] -> 1; [7,3,9] -> 5; [4,4,0] -> 6 (remis, wcześniejsza); [8] -> 9 (okno niepełne)
    assert idx == [1, 5, 6, 9]
    assert rep["n_windows"] == 4 and rep["n_selected"] == 4 and rep["skipped"] == []


def test_select_in_windows_respects_keep_and_skips_empty_window():
    keep = [True] * 10
    keep[1] = False  # argmax okna 0 odpada -> wygrywa 2 (2 > 1)
    keep[3] = keep[4] = keep[5] = False  # całe okno 1 odpada
    idx, rep = sel.select_in_windows(SCORES, 3, keep)
    assert idx == [2, 6, 9]
    assert rep["n_windows"] == 4 and rep["n_selected"] == 3
    assert len(rep["skipped"]) == 1
    skipped = rep["skipped"][0]
    assert (skipped["window"], skipped["start"], skipped["stop"]) == (1, 3, 6)
    assert skipped["reason"]


def test_select_in_windows_partial_last_window_skipped():
    keep = [True] * 9 + [False]
    idx, rep = sel.select_in_windows(SCORES, 3, keep)
    assert idx == [1, 5, 6]
    assert [(s["start"], s["stop"]) for s in rep["skipped"]] == [(9, 10)]


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 11, 12, 987, 1376])
def test_select_in_windows_count_equals_fixed_step(n, k):
    rng = np.random.default_rng(n * 10 + k)
    idx, rep = sel.select_in_windows(rng.random(n), k)
    assert len(idx) == len(sel.fixed_step(n, k, 0)) == rep["n_windows"]
    # Po jednej klatce z każdego okna, rosnąco.
    assert [i // k for i in idx] == list(range(len(idx)))


def test_select_in_windows_rejects_nan():
    with pytest.raises(ValueError):
        sel.select_in_windows([1.0, float("nan"), 2.0], 3)


# --------------------------------------------------------------------------- #
# Progi
# --------------------------------------------------------------------------- #


def _df(n=300, fps=30.0, ten=100.0, med=128.0):
    return {
        "pts_s": np.arange(n) / fps + 0.0083,
        "tenengrad": np.full(n, ten, dtype=np.float64),
        "bright_median": np.full(n, med, dtype=np.float64),
    }


def test_thresholds_default_values():
    """Progi ustalone przed przebiegami — test pilnuje, żeby nikt ich po cichu nie zmienił."""
    assert (sel.THR_REL, sel.THR_WIN_S, sel.THR_MED_LO, sel.THR_MED_HI) == (0.5, 2.0, 40.0, 220.0)


def test_thresholds_relative_sharpness():
    """Stały tenengrad 100, krótki spadek: do 40 (< 0,5 × 100) odpada, do 60 zostaje.
    Mediana w oknie ±2 s (121 klatek) jest dalej 100, bo spadek ma 6 klatek."""
    df = _df()
    df["tenengrad"][100:106] = 40.0
    df["tenengrad"][200:206] = 60.0
    keep = sel.thresholds(df)
    assert keep.dtype == bool and keep.shape == (300,)
    assert not keep[100:106].any()
    assert keep[200:206].all()
    assert keep.sum() == 300 - 6


def test_thresholds_window_is_in_seconds_not_frames():
    """Okno liczone z PTS, nie z liczby klatek. 90 klatek z tenengradem 300, potem 25 ze 100.

    29,97 kl./s, żeby krawędź okna (±2 s = ±59,94 klatki) nie wypadała dokładnie na klatce
    (przy 30 kl./s i klatce równo 2,0 s dalej o przynależności decyduje zaokrąglenie float).
    PTS ciągłe: klatka 90 ma w oknie klatki 31–114, czyli 59 × 300 i 25 × 100 -> mediana 300,
    a 100 < 0,5 · 300 -> odpada; klatka 114: klatki 55–114, 35 × 300 i 25 × 100 -> mediana 300
    -> odpada; tak samo wszystkie pomiędzy. Z luką przed klatką 90 (t = 12,0 s) w jej oknie są
    tylko klatki ze 100 -> zostaje, a klatki z 300 mają w oknie same 300 -> zostają. Bez luki
    klatki z 300 też zostają (300 ≥ 0,5 · mediana, a mediana ≤ 300)."""
    fps = 29.97
    ten = np.r_[np.full(90, 300.0), np.full(25, 100.0)]
    med = np.full(115, 128.0)
    pts_cont = np.arange(115) / fps
    pts_gap = np.r_[np.arange(90) / fps, 12.0 + np.arange(25) / fps]
    keep_gap = sel.thresholds({"pts_s": pts_gap, "tenengrad": ten, "bright_median": med})
    keep_cont = sel.thresholds({"pts_s": pts_cont, "tenengrad": ten, "bright_median": med})
    assert keep_gap.all()
    assert keep_cont[:90].all()
    assert not keep_cont[90:].any()


def test_thresholds_brightness_bounds_inclusive():
    df = _df(n=6)
    df["bright_median"][:] = [40.0, 220.0, 39.5, 220.5, 128.0, 0.0]
    keep = sel.thresholds(df)
    assert keep.tolist() == [True, True, False, False, True, False]


def test_threshold_parts_split():
    df = _df()
    df["tenengrad"][10] = 1.0
    df["bright_median"][20] = 250.0
    sharp_ok, bright_ok, local = sel.threshold_parts(df)
    assert not sharp_ok[10] and bright_ok[10]
    assert sharp_ok[20] and not bright_ok[20]
    assert np.all(local == 100.0)
    assert np.array_equal(sel.thresholds(df), sharp_ok & bright_ok)


def test_thresholds_rejects_unsorted_pts():
    df = _df(n=5)
    df["pts_s"][3] = df["pts_s"][1]
    with pytest.raises(ValueError):
        sel.thresholds(df)


# --------------------------------------------------------------------------- #
# Plik listy
# --------------------------------------------------------------------------- #


def test_frame_list_roundtrip():
    idx = [0, 3, 6, 1375]
    text = sel.format_frame_list(idx)
    assert text == "0\n3\n6\n1375\n"
    assert sel.parse_frame_list(text) == idx
    assert sel.parse_frame_list("1\n\n4\n") == [1, 4]


@pytest.mark.parametrize("text", ["", "\n", "3\n1\n", "1\n1\n", "-1\n", "1.5\n", "a\n"])
def test_frame_list_rejects_bad(text):
    with pytest.raises(ValueError):
        sel.parse_frame_list(text)


# --------------------------------------------------------------------------- #
# Obraz wejścia sieci
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hw, out", [((2160, 3840), (288, 512)), ((3840, 2160), (512, 288))])
def test_model_input_shape(hw, out):
    rgb = np.zeros((*hw, 3), dtype=np.uint8)
    assert sel.model_input_rgb(rgb).shape == (*out, 3)
    assert sel.model_input_gray(rgb).shape == out


@pytest.mark.parametrize("hw", [(2160, 3840), (3840, 2160), (1080, 1920), (3000, 4000), (1634, 3840)])
def test_model_input_matches_upstream(hw):
    """Obraz 512 px z `model_input_rgb()` == `resize_img()` upstreamu, bajt w bajt.

    Wejście upstreamu jak w adapterze: rgb.astype(float32) / 255 (`dataloader.py:50`).
    Szum losowy, bo na nim różnica filtra albo cropu wychodzi od razu.
    """
    pytest.importorskip("torch")
    mast3r_utils = pytest.importorskip("mast3r_slam.mast3r_utils")
    rng = np.random.default_rng(hw[0] + hw[1])
    rgb = rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8)
    expected = mast3r_utils.resize_img(rgb.astype(np.float32) / 255.0, 512)["unnormalized_img"]
    got = sel.model_input_rgb(rgb)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)
