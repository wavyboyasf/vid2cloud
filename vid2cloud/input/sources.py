"""Źródła klatek: plik wideo i kamera na żywo.

Będzie tu:
    FrameSource   — wspólny protokół: iterator zwracający (index, pts_s, frame_bgr).
    FileSource    — dekodowanie pliku przez PyAV, sekwencyjnie, jednym przebiegiem.
                    Bez seek-per-frame: seek tylko raz, przy ustawianiu offsetu startu.
    CameraSource  — strumień na żywo; klatki spóźnione są porzucane, nie kolejkowane.

Uwaga: dekodowanie idzie wyłącznie przez PyAV. OpenCV służy do operacji na obrazie
(ArUco, miary ostrości), nigdy do czytania wideo — patrz CLAUDE.md.
"""
