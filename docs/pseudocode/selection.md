# Algorytm 4. Dobór klatek w oknach z progami odrzucającymi

Moduły: `vid2cloud/input/selection.py` (miary, dobór), `vid2cloud/input/quality.py` (przebieg
wstępny, listy), `vid2cloud/run.py` (przebieg silnika z listą klatek).

**Wejście:** nagranie V o n klatkach; szerokość okna k; progi ustalone przed przebiegami:
r = 0,5, w = 2 s, [b_lo, b_hi] = [40, 220].

**Wyjście:** miary jakości każdej klatki; listy indeksów klatek: stały krok F_o (o = 0, …, k − 1),
argmax ostrości S, argmax ostrości z progami S_p; okna pominięte w S_p z powodem.

**Miary (przebieg wstępny, bez GPU)**

1. **Dla każdej** klatki V w kolejności dekodowania, bez przewijania:
   1. (i, t_i, I_i) ← kolejna klatka; t_i ← PTS · podstawa czasu; obróć I_i zgodnie
      z macierzą wyświetlania kontenera.
   2. J_i ← I_i przeskalowana jak wejście sieci: dłuższy bok do 512 px (filtr Lanczosa),
      crop centralny do wielokrotności 16 px; G_i ← J_i w skali szarości.
   3. T_i ← średnia po pikselach (G_x² + G_y²), gdzie G_x, G_y — splot G_i z jądrem Sobela 3 × 3.
   4. L_i ← wariancja splotu G_i z jądrem [[2, 0, 2], [0, −8, 0], [2, 0, 2]].
   5. B_i ← mediana jasności G_i; udziały pikseli o jasności < 5 i > 250.

**Progi odrzucające**

2. **Dla każdej** klatki i: M_i ← mediana { T_j : |t_j − t_i| ≤ w } (okno w czasie z PTS,
   przy końcach nagrania ucięte).
3. keep_i ← (T_i ≥ r · M_i) ∧ (b_lo ≤ B_i ≤ b_hi).

**Dobór**

4. F_o ← { o, o + k, o + 2k, … } ∩ [0, n), o = 0, …, k − 1.
5. Okna W_m ← [m·k, min((m + 1)·k, n)), m = 0, …, ⌈n / k⌉ − 1 (ostatnie może być krótsze
   i jest traktowane jak pozostałe).
6. **Dla każdego** okna W_m:
   1. S ← S ∪ { argmax_{i ∈ W_m} T_i } (remis: wcześniejsza klatka).
   2. C_m ← { i ∈ W_m : keep_i }.
   3. **Jeśli** C_m = ∅: zapisz W_m jako pominięte z powodem; **inaczej**
      S_p ← S_p ∪ { argmax_{i ∈ C_m} T_i }.
7. |S| = ⌈n / k⌉ = |F_0|; |S_p| = ⌈n / k⌉ − liczba okien pominiętych; |F_o| = ⌈(n − o) / k⌉.

**Przebieg silnika z listą P** (`vid2cloud run --frames`)

8. **Dla każdej** klatki V w kolejności dekodowania: zdekoduj (i, t_i, I_i); **jeśli** i ∈ P —
   Algorytm 2, kroki 2.2–2.5, i fed_i ← 1; **inaczej** zapisz w logu (i, t_i, czas dekodowania)
   i fed_i ← 0. Silnik startuje na pierwszej klatce z P.
