"""Log per klatka do CSV — obowiązkowy dla każdego przebiegu.

Będzie tu:
    FrameLogWriter  — writer CSV o stałym zestawie kolumn:
                      frame_idx, pts_s, sharpness, selected, decode_ms, infer_ms,
                      depth_median, depth_p95, low_conf_ratio.

Ten plik jest podstawą rozdziału wyników — bez niego nie ma czym uzasadnić selekcji
klatek ani pokazać, gdzie pipeline traci czas.
"""
