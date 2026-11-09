# Algorytm 2. Przebieg nagrania: dekodowanie, silnik, markery ArUco na klatkach kluczowych

Moduł: `vid2cloud/run.py` (komenda `vid2cloud run`); źródło klatek: `vid2cloud/input/sources.py`.

**Wejście:** plik wideo V, katalog wyjściowy O, przełącznik `single_thread`; opcjonalnie targets.json.

**Wyjście:** log per klatka, tabela markerów na klatkach kluczowych, opis przebiegu
(parametry źródła, wersja silnika, czasy) oraz wyjścia Algorytmu 1.

1. Odczytaj parametry strumienia V (rozdzielczość, rotacja, liczba klatek, czas trwania).
   Zapisz opis przebiegu ze statusem „w toku”.
2. **Dla każdej** klatki V w kolejności dekodowania, bez przewijania:
   1. (i, t_i, I_i) ← kolejna klatka; t_i ← PTS · podstawa czasu; obróć I_i zgodnie
      z macierzą wyświetlania kontenera.
   2. **Jeśli** i jest pierwszą klatką: τ ← przekształcenie pełna klatka → wejście sieci
      podane przez silnik dla kształtu I_i; uruchom silnik.
   3. R ← Algorytm 1 dla (I_i, i, t_i); zmierz czas dekodowania i czas silnika.
   4. **Dla każdej** nowej klatki kluczowej (k, j) w R:
      M ← detekcja ArUco na pełnej klatce I_j;
      **dla każdego** markera m ∈ M zapisz (k, j, id_m, q_m, τ(q_m), średni bok τ(q_m)),
      gdzie q_m — cztery narożniki w pełnej rozdzielczości.
   5. Zapisz wiersz logu: i, t_i, czasy, czy klatka kluczowa, k, tryb silnika, |KF|,
      mediana i 95. percentyl głębi z_i z pointmapy X_i, udział pikseli z ufnością < 1,5.
3. Zakończ silnik (Algorytm 1, kroki 11–13); policz punkty chmury.
4. Uzupełnij opis przebiegu o czasy całkowite i liczby wyników; status ← „zakończony”.
5. **Jeśli** podano wzorce (targets.json, sprawdzony przed krokiem 1): Algorytm 4 na O;
   wynik albo komunikat błędu → sekcja „scale” opisu przebiegu (błąd nie usuwa wyników kroków 1–4).

**Przekształcenie narożników** (konwencja środka piksela):
τ(x, y) = ((x + ½) · s_x − ½ − o_x, (y + ½) · s_y − ½ − o_y),
gdzie s_x = W / W_1, s_y = H / H_1 (skala do dłuższego boku 512), a (o_x, o_y) to lewy górny
róg cropu po skalowaniu — wszystkie cztery liczby pochodzą z silnika.
