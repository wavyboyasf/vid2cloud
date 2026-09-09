"""Zapis chmury punktów do PLY / LAS.

Będzie tu:
    write_ply()  — zapis XYZ + RGB (+ opcjonalnie ufność jako atrybut skalarny).
    write_las()  — wariant dla świata GIS-owego, z układem odniesienia jeśli znany.

Chmura zapisywana jest zawsze razem z config.json przebiegu, żeby dało się odtworzyć,
z jakich ustawień powstała.
"""
