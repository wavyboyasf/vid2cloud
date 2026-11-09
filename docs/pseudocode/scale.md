# Algorytm 4. Skala osobno dla każdej klatki kluczowej

Moduły: `vid2cloud/scale/pipeline.py` (komendy `vid2cloud scale`, `vid2cloud run --targets`),
`scale/factors.py` (współczynniki), `scale/cloud.py` (chmura), `scale/per_keyframe.py` (Algorytm 3).

**Wejście:** dla klatek kluczowych k = 0 … K−1 (kolejność powstania) — poza końcowa
T_k = (t_k, R_k, s_k), pointmapa X_k, ufność C_k, N_k (stan końcowy), czas pts_k; obserwacje
f_kj = factor_final par (k, odcinek j) ze statusem ok z Algorytmu 3; długości L_j; σ_obs, τ
(zamrożone, `params.json`); próg chmury c = 1,5.

**Wyjście:** wektory f_k trzech wariantów (globalny, A, B ± σ_u), położenia klatek p_k,
chmury w metrach, maks. ‖p_k − p_k^glob‖ dla A i B.

1. **Jeśli** brak obserwacji: status „brak par”, koniec (wynik, nie błąd).
2. med_k ← mediana { f_kj } dla klatek z obserwacjami; O ← zbiór takich k.

**Wariant globalny**

3. Dla każdego odcinka j: m_j ← mediana { s_k · d_k,j,final } po klatkach; f_j ← L_j / m_j.
4. f^glob ← mediana { f_j }; f_k^glob ← f^glob dla każdego k.

**Metoda A — interpolacja**

5. **Dla każdego** k: **jeśli** k ∈ O: f_k^A ← med_k;
   inaczej ln f_k^A ← interpolacja liniowa ln med w pts między najbliższymi k′ ∈ O
   (przed pierwszym i za ostatnim k′ ∈ O — wartość skrajna).

**Metoda B — wyrównanie pośredniczące** (niewiadome u_k = ln f_k)

6. Równania: u_k = ln f_kj, waga 1 / σ_obs²; u_(k+1) − u_k = 0, waga 1 / (τ² · (pts_(k+1) − pts_k)).
7. u ← (AᵀPA)⁻¹ AᵀPl;  σ_u,k ← √[(AᵀPA)⁻¹]_kk (σ0 a priori = 1);
   v ← Au − l;  r ← liczba równań − K;  σ0 ← √(vᵀPv / r) (jeśli r > 0).
8. f_k^B ← exp(u_k).

**Położenia i chmura** (wspólne dla wariantów, różni się tylko f)

9. p_0 ← f_0 · t_0;  p_k ← p_(k−1) + ½ (f_(k−1) + f_k) · (t_k − t_(k−1)).
10. **Dla każdego** k i piksela (i, j) z C_k(i, j) / N_k > c (kolejno k, wiersz po wierszu):
    P ← p_k + f_k · s_k · R_k · X_k(i, j).
11. Kontrola: chmura z kroku 10 przy f_k = 1, p_k = t_k ma odtworzyć chmurę upstreamu punkt
    po punkcie; **jeśli** tak — kolory z niej po indeksie, inaczej bez kolorów.
12. Δ_A ← max_k ‖p_k^A − p_k^glob‖, Δ_B ← max_k ‖p_k^B − p_k^glob‖.
    Rozrzut f_j albo f_kj > 5 % → ostrzeżenie (liczby bez zmian).

**σ_obs i τ** (raz, z nagrań 21.09): σ_obs² = Σ (ln f_kj − ln med_k)² / (n − m) po klatkach
z ≥ 2 odcinkami; τ² = Σ (ln med_k′ − ln med_k)² / Σ (pts_k′ − pts_k) po kolejnych k, k′ ∈ O
w obrębie nagrania.

Przy wszystkich f_k równych f kroki 9–10 dają P = f · (s_k R_k X + t_k), czyli wariant globalny.
