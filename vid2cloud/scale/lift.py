"""Lift narożników markera z 2D do 3D przy użyciu pointmapy silnika.

Będzie tu:
    lift_points()  — dla współrzędnych 2D w rozdzielczości wejścia modelu pobiera
                     XYZ z pointmapy (interpolacja + odrzucenie pikseli o niskiej ufności).

Pointmapa jest rzadka względem pełnej klatki, więc 3D narożnika jest interpolowane.
Dlatego skalę liczymy z odległości MIĘDZY markerami na łacie (~1 m), a nie z boku
pojedynczego markera — błąd interpolacji rozkłada się wtedy na dłuższy odcinek.
"""
