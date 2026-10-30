# Algorytm 5. Ocena kształtu chmury względem referencji

Moduły: `eval/coarse_from_pairs.py` (krok 0), `eval/eval_cloud.py` (kroki a–g),
`eval/eval_all.sh` (wszystkie chmury). SOR, ICP i C2C liczy CloudCompare 2.13.2 (`eval/cc.sh`).

**Wejście:** chmura rekonstrukcji X = {x_i} (jednostki chmury) nagrania r; referencja TLS
Y = {y_j} [m] (podpróbkowana do 1 cm, bez elementów ruchomych); pary punktów wskazane ręcznie
(p_k ↔ q_k), k = 1, …, m, m ≥ 3; chmura bazowa nagrania X_r (przebieg adaptera); opcjonalnie
współczynniki taśmy f_l (factor_final) tego przebiegu. Parametry ustalone przed wynikami:
percentyle (1, 99), margines δ = 0,20 m, SOR (k = 6, σ = 1), ICP (Δ = 10⁻⁶, 50 000 próbek,
nakrycie 90 %), progi τ = (0,05; 0,10; 0,20) m.

**Wyjście:** odległości d_i; n, średnia, mediana, RMS, p95 i odsetki d_i ≤ τ; skala s_icp;
RMS ICP; s_mod, s_mod / s_icp − 1, RMS C2C w skali s_mod.

0. **Zgrubnie, raz na nagranie** (Umeyama): (s_c, R_c, t_c) ← argmin Σ_k ‖q_k − (s R p_k + t)‖²;
   M_c ← [s_c R_c | t_c]; odchyłki e_k ← ‖M_c p_k − q_k‖ i ich RMS.
1. **(a)** X′ ← { M_c x_i }.
2. **(b) Prostopadłościan, raz na nagranie:** dla każdej osi a ∈ {X, Y, Z}:
   lo_a ← P₁({(M_c x)_a : x ∈ X_r}) − δ, hi_a ← P₉₉(…) + δ.
   B ← [lo, hi]; X″ ← X′ ∩ B; Y_B ← Y ∩ B.
3. **(c) SOR:** dla każdego x ∈ X″: μ(x) ← średnia odległość do k najbliższych sąsiadów w X″;
   X‴ ← { x : μ(x) ≤ mean(μ) + σ · std(μ) }.
4. **(d) ICP ze skalą** X‴ → Y_B (X‴ ruchoma, Y_B nieruchoma): powtarzaj — losuj ≤ 50 000
   punktów X‴, przyporządkuj najbliższe punkty Y_B, odrzuć 10 % najdalszych par, wyznacz
   podobieństwo i zastosuj — aż spadek błędu < Δ. Wynik M_icp; RMS_icp z CloudCompare.
5. **(e) C2C:** dla każdego x ∈ X‴: d(x) ← min_{y ∈ Y_B} ‖M_icp x − y‖ (bez modelu lokalnego).
6. **(f)** M ← M_icp · M_c; s_icp ← ∛det(M[1:3, 1:3]); statystyki d; |X″|, |X‴|.
7. **(g) Jeśli** są współczynniki taśmy:
   1. s_mod ← mediana_l f_l; zapisz s_mod i (s_mod / s_icp − 1) · 100 %.
   2. M = [s_icp R | t] → T₀ ← [R | t]; Z ← { T₀ (s_mod · M_c⁻¹ x) : x ∈ X‴ }.
   3. ICP sztywne Z → Y_B (jak krok 4, bez skali); C2C jak krok 5; RMS d — kształt w skali z taśmy.
