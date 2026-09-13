"""Detekcja ArUco na syntetyce: perspektywa, mały marker, przeliczenie na wejście modelu.

Testujemy tylko funkcje czyste — MASt3R-SLAM nie jest tu w ogóle dotykany (CLAUDE.md,
zasada 2). Reimplementacja przekształcenia wejścia silnika jest sprawdzana względem
PIL, czyli tej samej biblioteki, której używa `mast3r_utils.resize_img()`.
"""

import cv2
import numpy as np
import PIL.Image
import pytest

from vid2cloud.scale import aruco

#: Najmniejszy bok markera [px], przy którym detekcja działa na czystej syntetyce dla
#: każdego z testowanych ID. Wartość zmierzona — patrz docs/decisions.md (2026-09-13).
#: To granica optymistyczna: bez szumu, rozmycia, kompresji i pod kątem 0°.
MIN_MARKER_PX = 12

#: Bok markera 12 cm widzianego z 3 m przy wejściu modelu 512 px (z pola widzenia
#: telefonu ~65°) — rozmiar z prompta sesji, ten musi działać.
FIELD_MARKER_PX = 15

MARKER_IDS = (0, 7, 23, 49)


def _big_scene(marker_id: int, side: int = 600, quiet: float = 1.0) -> np.ndarray:
    """Marker na białym tle z otuliną `quiet` boku markera z każdej strony."""
    marker = cv2.aruco.generateImageMarker(aruco.get_dictionary(), marker_id, side)
    pad = int(round(side * quiet))
    canvas = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    canvas[pad : pad + side, pad : pad + side] = marker
    return canvas


def _expected_corners(offset: float, side: float) -> np.ndarray:
    """Narożniki markera w konwencji środka piksela.

    Czarny obszar zajmuje piksele [offset, offset+side), więc jego zewnętrzna krawędź
    leży na współrzędnej ciągłej `offset`, czyli `offset - 0.5` licząc od środków pikseli.
    Kolejność jak w `aruco.CORNER_ORDER`.
    """
    a, b = offset - 0.5, offset + side - 0.5
    return np.array([[a, a], [b, a], [b, b], [a, b]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Test 1: marker pod perspektywą
# --------------------------------------------------------------------------- #


def test_detect_marker_under_perspective():
    """Marker wklejony na białe tło i zniekształcony perspektywicznie: ten sam ID,
    narożniki w granicach 1 px od przekształconych narożników oczekiwanych."""
    marker_id, side, offset, n = 7, 300, 300, 900
    marker = cv2.aruco.generateImageMarker(aruco.get_dictionary(), marker_id, side)
    canvas = np.full((n, n), 255, np.uint8)
    canvas[offset : offset + side, offset : offset + side] = marker

    # Homografia: skos w obie strony, na tyle mocny, żeby marker nie był już prostokątem.
    src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    dst = np.float32([[60, 30], [n - 20, 90], [n - 45, n - 35], [95, n - 70]])
    h = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        canvas, h, (n, n), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )

    detections = aruco.detect(warped)
    assert set(detections) == {marker_id}

    expected = cv2.perspectiveTransform(
        _expected_corners(offset, side).reshape(-1, 1, 2), h
    ).reshape(4, 2)
    error = np.abs(detections[marker_id] - expected).max()
    assert error < 1.0, f"największy błąd narożnika {error:.3f} px"


def test_detect_returns_empty_and_rejects_duplicates():
    """Białe tło bez markera -> pusty słownik; ten sam ID dwa razy -> ValueError."""
    assert aruco.detect(np.full((400, 400), 255, np.uint8)) == {}

    marker = cv2.aruco.generateImageMarker(aruco.get_dictionary(), 7, 180)
    canvas = np.full((400, 800), 255, np.uint8)
    canvas[110:290, 60:240] = marker
    canvas[110:290, 560:740] = marker
    with pytest.raises(ValueError, match="wielokrotnie"):
        aruco.detect(canvas)


# --------------------------------------------------------------------------- #
# Test 2: mały marker — gdzie jest próg
# --------------------------------------------------------------------------- #


def _detect_at(scene: np.ndarray, side_px: int, marker_id: int) -> bool:
    """Skaluje scenę tak, żeby bok markera miał `side_px`, i próbuje wykryć."""
    n = scene.shape[0]
    big = n // 3  # _big_scene: otulina = 1 bok markera z każdej strony
    out = int(round(n * side_px / big))
    small = cv2.resize(scene, (out, out), interpolation=cv2.INTER_AREA)
    return marker_id in aruco.detect(small)


@pytest.mark.parametrize("marker_id", MARKER_IDS)
def test_detect_small_marker_threshold(marker_id):
    """Ile pikseli boku potrzebuje detekcja na czystej syntetyce.

    Progiem nazywamy najmniejszy bok, od którego detekcja działa **nieprzerwanie**
    aż do 40 px — pojedyncze trafienie przy 6 px to przypadek, nie zdolność.
    """
    scene = _big_scene(marker_id)
    sizes = range(6, 41)
    hits = {s for s in sizes if _detect_at(scene, s, marker_id)}

    threshold = next((s for s in sizes if all(x in hits for x in range(s, 41))), None)
    assert threshold is not None, f"ID {marker_id}: brak detekcji w żadnym stabilnym zakresie"
    assert threshold <= MIN_MARKER_PX, (
        f"ID {marker_id}: próg podskoczył do {threshold} px "
        f"(udokumentowane {MIN_MARKER_PX} px) — zaktualizuj docs/decisions.md"
    )
    assert _detect_at(scene, FIELD_MARKER_PX, marker_id), (
        f"ID {marker_id}: marker {FIELD_MARKER_PX} px (12 cm z 3 m przy 512) niewykryty"
    )


def test_small_marker_corner_accuracy():
    """Przy 15 px narożniki nadal mieszczą się w 1 px — bez tego lift do 3D nie ma sensu."""
    marker_id, side, quiet = 7, 600, 1.0
    scene = _big_scene(marker_id, side, quiet)
    n = scene.shape[0]
    out = int(round(n * FIELD_MARKER_PX / side))
    small = cv2.resize(scene, (out, out), interpolation=cv2.INTER_AREA)

    detections = aruco.detect(small)
    assert marker_id in detections

    s = out / n
    pad = int(round(side * quiet))
    expected = _expected_corners(pad * s, side * s)
    error = np.abs(detections[marker_id] - expected).max()
    assert error < 1.0, f"największy błąd narożnika przy {FIELD_MARKER_PX} px: {error:.3f} px"


# --------------------------------------------------------------------------- #
# Przeliczenie na wejście modelu
# --------------------------------------------------------------------------- #


def _pil_reference(w1: int, h1: int, size: int = 512):
    """Dosłowne powtórzenie `mast3r_utils.resize_img()` na PIL — wzorzec do porównania."""
    img = PIL.Image.new("L", (w1, h1))
    s = max(img.size)
    img = img.resize(tuple(int(round(x * size / s)) for x in img.size))
    w, h = img.size
    cx, cy = w // 2, h // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if w == h:
        halfh = 3 * halfw / 4
    cropped = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return (h, w), (cx - halfw, cy - halfh), (cropped.size[1], cropped.size[0])


@pytest.mark.parametrize(
    "w1,h1",
    [(3840, 2160), (1920, 1080), (4000, 3000), (4032, 3024), (1080, 1920),
     (3840, 1634), (1000, 1000), (640, 480), (1919, 1079)],
)
def test_input_transform_matches_upstream(w1, h1):
    """Skala, offset cropu i kształt wyjścia zgadzają się z PIL-owym oryginałem."""
    resized, left_top, out_shape = _pil_reference(w1, h1)
    tf = aruco.input_transform((h1, w1))

    assert tf.resized_shape == resized
    assert (tf.left, tf.top) == pytest.approx(left_top)
    assert tf.out_shape == out_shape
    assert tf.scale_x == pytest.approx(resized[1] / w1)
    assert tf.scale_y == pytest.approx(resized[0] / h1)


def test_rescale_corners_scale_crop_and_shape_check():
    """Skalowanie 4K (bez cropu) i format z cropem; niezgodne `to_shape` -> ValueError."""
    corners = {7: np.array([[0.0, 0.0], [3839.0, 0.0], [3839.0, 2159.0], [0.0, 2159.0]])}
    out = aruco.rescale_corners(corners, (2160, 3840), (288, 512))
    # 3840x2160 -> 512x288 bez cropu: skala 512/3840 w obu osiach, środki pikseli
    k = 512 / 3840
    assert out[7][0] == pytest.approx([0.5 * k - 0.5, 0.5 * k - 0.5])
    assert out[7][2] == pytest.approx([3839.5 * k - 0.5, 2159.5 * (288 / 2160) - 0.5])

    # 3840x1634 -> 512x218 -> crop do 512x208, czyli 5 px z góry
    point = np.array([[1920.0, 817.0]])
    shifted = aruco.rescale_corners(point, (1634, 3840), (208, 512))
    assert shifted[0, 1] == pytest.approx((817.5) * (218 / 1634) - 0.5 - 5.0)

    with pytest.raises(ValueError, match="kształt wejścia modelu"):
        aruco.rescale_corners(corners, (2160, 3840), (384, 512))


def test_rescale_corners_accepts_array_and_dict():
    array_out = aruco.rescale_corners(np.zeros((4, 2)), (2160, 3840), (288, 512))
    dict_out = aruco.rescale_corners({3: np.zeros((4, 2))}, (2160, 3840), (288, 512))
    assert isinstance(array_out, np.ndarray) and array_out.shape == (4, 2)
    assert isinstance(dict_out, dict) and set(dict_out) == {3}
    assert dict_out[3] == pytest.approx(array_out)
