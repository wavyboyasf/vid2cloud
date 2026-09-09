"""Generator plansz z markerami ArUco do druku (PDF w milimetrach).

Będzie tu:
    - generowanie PDF (reportlab) z markerami DICT_4X4_50 o zadanym boku w mm,
      po jednym markerze na stronę, wycentrowanym, z białym marginesem >= 1 bok markera
      i podpisem "ID n · <bok> mm · DICT_4X4_50",
    - tryb --scalebar: markery na końce łaty skali,
    - zapis boards.json obok PDF: {id, dict, nominal_size_mm}.

UWAGA: drukarki skalują wydruk o 1-3%. Bok każdego wydrukowanego markera trzeba
zmierzyć suwmiarką i wpisać rzeczywistą wartość do targets.json — wartość nominalna
z boards.json jest tylko punktem odniesienia.

Implementacja: sesja 3.
"""
