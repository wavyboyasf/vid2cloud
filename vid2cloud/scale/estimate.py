"""Estymacja współczynnika skali z odległości między wzorcami.

Będzie tu:
    estimate_scale()  — dla par punktów o znanej odległości rzeczywistej (targets.json)
                        liczy współczynnik skali wraz z rozrzutem (mediana, min/max, std).
    apply_scale()      — przeskalowanie chmury i trajektorii, zapis użytego współczynnika.

Skala pochodzi ZAWSZE z fizycznego wzorca zmierzonego w terenie. Żadnych stałych
poprawek ani prioru modelu — patrz CLAUDE.md. Funkcje czyste, testowane na syntetyce.
"""
