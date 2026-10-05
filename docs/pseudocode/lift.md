# Algorytm 3. Lift punktów markera i współczynnik skali per klatka kluczowa

Moduły: `vid2cloud/scale/lift.py` (lift), `eval/scale_per_keyframe.py` (współczynniki).

**Wejście:** dla klatki kluczowej k — pointmapa X_k (h × w × 3) w układzie kamery k, suma ufności
C_k i liczba uśrednień N_k (stan przy powstaniu i na końcu), skala s_k z historii póz;
narożniki markerów q_m (4 × 2) w pikselach wejścia sieci; odcinki taśmy (marker a, marker b,
długość L, opis punktów odniesienia); próg ufności c_min = 1,5.

**Wyjście:** dla każdej pary (k, odcinek), w której oba markery są wykryte: d_canon (stan przy
powstaniu i końcowy) oraz współczynniki f_sieć, f_online, f_końcowy.

**Lift punktu odniesienia** (piksel (i, j) ma środek w (x = j, y = i))

1. **Jeśli** N_k = 0: odrzuć klatkę (średnia ufność nieokreślona). Inaczej c ← C_k / N_k.
2. o ← przecięcie przekątnych czworokąta q; q′ ← narożniki przysunięte do o do 50 %.
3. P ← { X_k(i, j) : środek (j, i) ∈ q′, c(i, j) ≥ c_min }. **Jeśli** |P| < 5: brak punktu, podaj powód.
4. Środek markera: mediana P po współrzędnych; rozrzut = mediana ‖p − mediana‖.
5. Narożnik n: płaszczyzna Π ← najmniejszy kierunek SVD (P − średnia);
   promień r(x, y) ← dopasowanie afiniczne (X/Z, Y/Z) w (x, y) na pikselach P;
   punkt ← przecięcie promienia r(q_n) z Π.
6. Środek krawędzi (n₁, n₂): środek odcinka między punktami 3D narożników n₁ i n₂ (po lifcie).
7. Diagnostycznie: to samo z mediany X w oknie 3 × 3 wokół q_n (piksele z c ≥ c_min).

**Współczynnik odcinka w klatce k**

8. Punkty odniesienia (a, b) ← opis odcinka: narożnik, środek krawędzi albo górny róg krawędzi
   (dla pomiaru „pod kątem”); nieznany opis — błąd przed liczeniem.
9. **Dla** stanu e ∈ {przy powstaniu, końcowy}: d_e ← ‖lift(a) − lift(b)‖ w układzie kamery k.
10. f_sieć ← L / d_powstanie; f_online ← L / (s_k,powstanie · d_powstanie);
    f_końcowy ← L / (s_k,koniec · d_koniec)   [L w metrach, f w m na jednostkę chmury].
11. Różnica metod: Δ ← L · (d_okno3×3 / d − 1) [mm], bez progu.

**Nakładka:** środek markera m w klatce k w świecie: s_k · R(q_k) · o_m + t_k (poza końcowa).
