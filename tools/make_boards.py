"""Generator plansz z markerami ArUco do druku (PDF w milimetrach).

Dwa tryby:

* plansze (domyślnie) — po jednym markerze na stronę, wycentrowanym, z podpisem
  "ID 3 · 120 mm · DICT_4X4_50"; do rozłożenia w scenie jako punkty kontrolne;
* ``--scalebar`` — strona A4 z markerem i krzyżem osiowym; dwa takie wydruki
  (dwa różne ID) idą na końce listwy, a odległość między środkami markerów jest
  jedynym wzorcem skali w projekcie.

Obok PDF zapisywany jest ``boards.json`` z listą ``{id, dict, nominal_size_mm}``.

Dlaczego marker jest rastrem, a nie wektorem: bok markera w pikselach musi być
wielokrotnością 6 (DICT_4X4_50 to 4 komórki danych + 2 komórki bordera), inaczej
krawędzie komórek wypadają między pikselami i przy rasteryzacji do druku robi się
z nich szarość. Obraz jest renderowany w >= 600 dpi dla zadanego rozmiaru fizycznego
i wstawiany do PDF jako bitmapa — drawImage nie skaluje go dalej.

UWAGA: drukarki skalują wydruk o 1-3 %. Bok każdego wydrukowanego markera trzeba
zmierzyć suwmiarką i wpisać rzeczywistą wartość do targets.json — wartość nominalna
z boards.json jest tylko punktem odniesienia.

Środowisko: env vid2cloud-tools (opencv-contrib-python, reportlab, pillow).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
from reportlab.lib import pagesizes
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas

DEFAULT_DICT = "DICT_4X4_50"

#: Komórek na bok obrazu markera: 4 komórki danych + 2 komórki czarnego bordera
#: (markerBorderBits = 1 z każdej strony). Bok w pikselach musi być wielokrotnością.
CELLS_PER_SIDE = 6

#: Minimalna rozdzielczość rastra markera w PDF.
MIN_DPI = 600.0

PAGES = {name: getattr(pagesizes, name) for name in ("A0", "A1", "A2", "A3", "A4", "A5")}


# --------------------------------------------------------------------------- #
# Marker -> bitmapa
# --------------------------------------------------------------------------- #


def marker_pixels(size_mm: float, dpi: float = MIN_DPI) -> int:
    """Bok markera w pikselach: >= `dpi` dla `size_mm`, zaokrąglony w górę do 6."""
    needed = size_mm / 25.4 * dpi
    return int(math.ceil(needed / CELLS_PER_SIDE)) * CELLS_PER_SIDE


def render_marker(dictionary, marker_id: int, side_px: int) -> PIL.Image.Image:
    """Bitmapa markera jako obraz 1-bitowy (bez półtonów na krawędziach komórek)."""
    if side_px % CELLS_PER_SIDE:
        raise ValueError(f"bok {side_px} px nie jest wielokrotnością {CELLS_PER_SIDE}")
    array = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    # generateImageMarker zwraca czyste 0/255, więc konwersja do "1" nic nie gubi
    # i daje w PDF bezstratny, mały obraz zamiast ośmiobitowego.
    return PIL.Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("1")


# --------------------------------------------------------------------------- #
# Strony PDF
# --------------------------------------------------------------------------- #


def _caption_size(size_mm: float) -> float:
    """Stopień pisma podpisu w punktach — rośnie z markerem, ale w rozsądnych granicach."""
    return max(9.0, min(20.0, size_mm * 0.12))


def _centred_text(pdf, x_center, baseline, text, font_size, knockout=False):
    """Tekst wyśrodkowany; `knockout` kładzie pod nim biały prostokąt.

    Na stronie łaty skali przez środek biegnie linia osiowa — bez wybicia tła
    przecinałaby podpis w poprzek.
    """
    pdf.setFont("Helvetica", font_size)
    if knockout:
        width = pdf.stringWidth(text, "Helvetica", font_size)
        pad = 0.3 * font_size
        pdf.setFillColorRGB(1, 1, 1)
        pdf.rect(x_center - width / 2 - pad, baseline - 0.3 * font_size - pad,
                 width + 2 * pad, font_size + 2 * pad, stroke=0, fill=1)
        pdf.setFillColorRGB(0, 0, 0)
    pdf.drawCentredString(x_center, baseline, text)


def draw_board_page(pdf, dictionary, marker_id, size_mm, page_wh, dict_name,
                    gap_mm=8.0, knockout=False):
    """Jedna strona plansz: marker wycentrowany, podpis pod nim. Zwraca marginesy [mm]."""
    page_w, page_h = page_wh
    side = size_mm * mm
    x = (page_w - side) / 2.0
    y = (page_h - side) / 2.0

    pdf.drawImage(
        ImageReader(render_marker(dictionary, marker_id, marker_pixels(size_mm))),
        x, y, width=side, height=side,
    )

    font_size = _caption_size(size_mm)
    caption = f"ID {marker_id} \u00b7 {size_mm:g} mm \u00b7 {dict_name}"
    _centred_text(pdf, page_w / 2.0, y - gap_mm * mm - font_size, caption, font_size, knockout)

    return {
        "left_right_mm": round((page_w - side) / 2.0 / mm, 2),
        "top_mm": round((page_h - side) / 2.0 / mm, 2),
        "bottom_mm": round(((page_h - side) / 2.0 - gap_mm * mm - font_size) / mm, 2),
    }


def draw_scalebar_page(pdf, dictionary, marker_id, size_mm, page_wh, dict_name, gap_mm=8.0):
    """Strona łaty skali: krzyż osiowy przez środek markera + znaczniki na krawędziach.

    Znaczniki służą do przyklejenia obu wydruków na listwie tak, żeby środki markerów
    leżały na jednej osi — mierzona jest odległość między środkami, nie między bokami.
    Krzyż idzie pod marker (nieprzezroczysty raster go zakryje), a podpisy mają wybite
    białe tło, więc linia nigdzie nie przecina tekstu.
    """
    page_w, page_h = page_wh
    cx, cy = page_w / 2.0, page_h / 2.0
    tick = 12 * mm

    pdf.setLineWidth(0.4)
    pdf.line(0, cy, page_w, cy)
    pdf.line(cx, 0, cx, page_h)
    # Pogrubione znaczniki tuż przy krawędziach — po nich przykłada się wydruk do listwy.
    pdf.setLineWidth(1.2)
    for x0, y0, x1, y1 in (
        (0, cy, tick, cy), (page_w - tick, cy, page_w, cy),
        (cx, 0, cx, tick), (cx, page_h - tick, cx, page_h),
    ):
        pdf.line(x0, y0, x1, y1)

    margins = draw_board_page(pdf, dictionary, marker_id, size_mm, page_wh, dict_name,
                              gap_mm, knockout=True)

    _centred_text(
        pdf, cx, 14 * mm,
        "LATA SKALI \u00b7 sklej oba wydruki na listwie po krzyzu osiowym \u00b7 "
        "mierz odleglosc miedzy srodkami markerow",
        9.0, knockout=True,
    )
    return margins


# --------------------------------------------------------------------------- #
# Wejście
# --------------------------------------------------------------------------- #


def parse_ids(spec: str) -> list[int]:
    """"0-7", "0,3,5", "0-3,8-9" -> posortowana lista bez powtórzeń."""
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part.lstrip("-"):
                lo, _, hi = part.partition("-")
                lo_i, hi_i = int(lo), int(hi)
                if hi_i < lo_i:
                    raise ValueError(f"zakres {part!r} ma koniec przed początkiem")
                ids.update(range(lo_i, hi_i + 1))
            else:
                ids.add(int(part))
        except ValueError as exc:
            if "invalid literal" in str(exc):
                raise ValueError(f"{part!r} to nie liczba ani zakres; oczekuję np. \"0-7\", \"0,3,5\"") from exc
            raise
    if not ids:
        raise ValueError(f"pusta lista ID: {spec!r}")
    if min(ids) < 0:
        raise ValueError(f"ID musi być >= 0, dostałem {min(ids)}")
    return sorted(ids)


def get_dictionary(dict_name: str):
    value = getattr(cv2.aruco, dict_name, None)
    if not dict_name.startswith("DICT_") or value is None:
        raise ValueError(f"cv2.aruco nie zna słownika {dict_name!r}")
    return cv2.aruco.getPredefinedDictionary(value)


# --------------------------------------------------------------------------- #
# Główna funkcja
# --------------------------------------------------------------------------- #


def build(out_pdf: Path, ids, size_mm: float, page: str, dict_name: str, scalebar: bool) -> dict:
    """Generuje PDF i zwraca zawartość boards.json (nie zapisuje go)."""
    if page not in PAGES:
        raise ValueError(f"nieznany format {page!r}; dostępne: {', '.join(PAGES)}")
    page_wh = PAGES[page]
    dictionary = get_dictionary(dict_name)

    n_markers = int(dictionary.bytesList.shape[0])
    too_big = [i for i in ids if i >= n_markers]
    if too_big:
        raise ValueError(f"{dict_name} ma ID 0..{n_markers - 1}, a podano {too_big}")

    side = size_mm * mm
    if side >= min(page_wh):
        raise ValueError(
            f"marker {size_mm:g} mm nie mieści się na {page} "
            f"({page_wh[0] / mm:.0f} x {page_wh[1] / mm:.0f} mm)"
        )

    side_px = marker_pixels(size_mm)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfcanvas.Canvas(str(out_pdf), pagesize=page_wh)
    pdf.setTitle(f"vid2cloud {'scalebar' if scalebar else 'boards'} {dict_name} {size_mm:g} mm")

    draw = draw_scalebar_page if scalebar else draw_board_page
    margins = {}
    for marker_id in ids:
        margins = draw(pdf, dictionary, marker_id, size_mm, page_wh, dict_name)
        pdf.showPage()
    pdf.save()

    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tool": "tools/make_boards.py",
        "mode": "scalebar" if scalebar else "boards",
        "pdf": out_pdf.name,
        "dict": dict_name,
        "page": page,
        "page_mm": [round(page_wh[0] / mm, 2), round(page_wh[1] / mm, 2)],
        "nominal_size_mm": size_mm,
        "marker_px": side_px,
        "render_dpi": round(side_px * 25.4 / size_mm, 1),
        "white_margin_mm": margins,
        "markers": [
            {"id": i, "dict": dict_name, "nominal_size_mm": size_mm, "page": n}
            for n, i in enumerate(ids, start=1)
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/make_boards.py",
        description="PDF z markerami ArUco w zadanym rozmiarze fizycznym.",
        epilog="Po wydruku zmierz bok suwmiarką i wpisz do targets.json — drukarki skalują.",
    )
    parser.add_argument("--ids", default=None, help="ID markerów: \"0-7\", \"0,3,5\", \"0-3,8-9\".")
    parser.add_argument("--size-mm", type=float, default=None, help="Bok markera w mm (plansze 120, łata 80).")
    parser.add_argument("--page", default=None, choices=sorted(PAGES), help="Format strony (plansze A3, łata A4).")
    parser.add_argument("--dict", dest="dict_name", default=DEFAULT_DICT, help=f"Słownik (domyślnie {DEFAULT_DICT}).")
    parser.add_argument("--scalebar", action="store_true", help="Tryb łaty skali: krzyż osiowy, domyślnie A4.")
    parser.add_argument("--out", type=Path, default=None, help="Plik PDF; boards.json ląduje obok.")
    args = parser.parse_args(argv)

    # Domyślne inne dla plansz i dla łaty. 70 mm to największy bok, przy którym biały
    # margines na A4 (210 mm) wychodzi równy bokowi markera: (210 - 70) / 2 = 70.
    size_mm = args.size_mm if args.size_mm is not None else (70.0 if args.scalebar else 120.0)
    page = args.page if args.page is not None else ("A4" if args.scalebar else "A3")
    ids_spec = args.ids if args.ids is not None else ("48-49" if args.scalebar else "0-7")

    if args.scalebar and args.page not in (None, "A4"):
        print(f"uwaga: łata skali na {args.page} zamiast A4 — upewnij się, że ksero nie przeskaluje")

    try:
        ids = parse_ids(ids_spec)
        mode = "scalebar" if args.scalebar else "boards"
        # boards.json leży obok PDF pod stałą nazwą, więc każdy przebieg dostaje własny
        # katalog — inaczej druga komenda skasowałaby opis pierwszego wydruku.
        out_pdf = args.out or Path("boards") / f"{mode}_{page}_{size_mm:g}mm" / f"{mode}.pdf"
        meta = build(out_pdf, ids, size_mm, page, args.dict_name, args.scalebar)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # nieosiągalne, parser.error kończy proces

    json_path = out_pdf.with_name("boards.json")
    if json_path.exists():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8")).get("pdf")
        except (json.JSONDecodeError, OSError):
            previous = None
        if previous and previous != out_pdf.name:
            print(f"UWAGA: nadpisuję {json_path}, które opisywało {previous} — ten wydruk "
                  f"nie ma już swojego boards.json")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PDF:        {out_pdf}  ({len(ids)} stron, ID {ids[0]}..{ids[-1]})")
    print(f"boards.json {json_path}")
    print(f"marker:     {size_mm:g} mm = {meta['marker_px']} px przy {meta['render_dpi']:.0f} dpi "
          f"({meta['marker_px'] // CELLS_PER_SIDE} px na komórkę)")
    margins = meta["white_margin_mm"]
    print(f"biały margines [mm]: boki {margins['left_right_mm']}, góra {margins['top_mm']}, dół {margins['bottom_mm']}")

    if min(margins.values()) < size_mm:
        print(
            f"UWAGA: najwęższy biały margines to {min(margins.values()):g} mm, "
            f"mniej niż bok markera ({size_mm:g} mm). Detekcja z ostrego kąta może na tym "
            f"ucierpieć — albo weź większy format, albo zostaw wokół planszy jasne tło."
        )
    print("UWAGA: po wydruku zmierz bok suwmiarką i wpisz do targets.json — drukarki skalują o 1-3 %.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
