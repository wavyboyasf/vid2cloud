"""Podgląd 3D w przeglądarce (viser).

Będzie tu:
    serve()  — uruchomienie serwera viser: chmura punktów, trajektoria kamery,
               zaznaczone pozycje wykrytych markerów ArUco.

Podgląd jest opcjonalny (extra `viz`) i nie może być wymagany do przetwarzania —
pipeline musi działać headless, bo tak leci na GPU w tle.
"""
