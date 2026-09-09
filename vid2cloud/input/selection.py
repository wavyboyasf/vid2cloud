"""Selekcja klatek: które klatki w ogóle trafiają do rekonstrukcji.

Będzie tu:
    sharpness()      — miara ostrości klatki (wariancja Laplasjanu lub Tenengrad);
                       funkcja czysta, testowana jednostkowo na syntetyce.
    select_frames()  — reguła wyboru: odrzucenie klatek rozmytych, kontrola
                       minimalnego odstępu czasowego / parallaksy między wyborami.

Każda klatka — wybrana czy nie — musi zostawić wiersz w logu CSV (indeks, PTS,
miara ostrości, czy wybrana). Bez tego nie ma z czego pisać rozdziału wyników.
"""
