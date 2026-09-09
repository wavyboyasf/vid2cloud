"""Adapter do upstreamowego MASt3R-SLAM z third_party/mast3r-slam/.

Będzie tu:
    Mast3rSlamEngine — implementacja Engine z engine/base.py: uruchomienie silnika,
                       podawanie klatek, odbiór pointmap i póz, zamknięcie sesji.
    _to_frame_result() — tłumaczenie struktur upstreamu na FrameResult.

To JEDYNE miejsce w projekcie, które wolno wiązać z API MASt3R-SLAM. Upstreamu nie
modyfikujemy — jeśli czegoś brakuje, obchodzimy to tutaj, nie łatką w third_party/.
Adapter musi też odnotować rozdzielczość wejścia modelu (pointmapy są w niej, nie w
rozdzielczości oryginalnej klatki) — korzysta z tego scale/lift.py.
"""
