"""vid2cloud — gęsta chmura punktów z wideo, klatka po klatce.

Warstwy pakietu:
    input/   — dekodowanie wideo, bufor klatek, selekcja klatek do rekonstrukcji
    engine/  — abstrakcja silnika rekonstrukcji + adapter do MASt3R-SLAM
    scale/   — moduł skali metrycznej (ArUco -> lift do 3D -> estymacja skali)
    export/  — zapis chmury, trajektorii i logu per klatka
    viz/     — podgląd 3D (viser)
    cli.py   — interfejs wiersza poleceń (typer)
"""

__version__ = "0.1.0"
