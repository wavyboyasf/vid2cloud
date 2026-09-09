"""Wspólny interfejs silnika rekonstrukcji.

Będzie tu:
    FrameResult   — wynik dla jednej klatki: pointmapa, mapa ufności, poza kamery,
                    rozdzielczość wejścia modelu, znacznik klatki kluczowej.
    Engine        — protokół silnika; kluczowa metoda push_frame(frame) -> FrameResult.
                    Dodatkowo: finalize() zwracające scaloną chmurę i trajektorię.

Interfejs jest po to, żeby reszta pakietu nie znała szczegółów MASt3R-SLAM i żeby
dało się podstawić inny silnik (CUT3R, SLAM3R) bez ruszania warstwy wejścia i skali.
"""
