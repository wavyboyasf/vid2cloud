"""Test C — zależność współczynnika skali od punktu startu sekwencji.

Będzie tu skrypt, który z jednego nagrania tnie N klipów o różnych offsetach startu
(ffmpeg), uruchamia silnik na każdym z nich i zbiera wyniki do runs/test_c/results.csv.
Pomiar odcinka odniesienia robi autor ręcznie w CloudCompare — skrypt zostawia na to
puste kolumny i dopiero w trybie --fill liczy współczynniki i ich rozrzut.

Hipoteza: MASt3R-SLAM dziedziczy skalę mapy z pierwszej klatki kluczowej, więc
współczynnik może zależeć od tego, gdzie zaczyna się klip.

Implementacja: sesja 2.
"""
