# Algorytm 1. Silnik: MASt3R-SLAM klatka po klatce z zapisem stanu klatek kluczowych

Moduł: `vid2cloud/engine/mast3r_slam.py` (adapter do MASt3R-SLAM, Murai i in. 2025).

**Wejście:** ciąg klatek (I_i, i, t_i) — obraz RGB w pełnej rozdzielczości, indeks, czas PTS;
konfiguracja silnika; przełącznik `single_thread` ∈ {prawda, fałsz}.

**Wyjście:** dla każdej klatki — poza T_i ∈ Sim(3), pointmapa X_i i ufność C_i w układzie kamery,
tryb pracy; dla każdej klatki kluczowej k — stan w chwili powstania i na końcu
(T_k, X_k, C_k, N_k); historia póz {T_k} po każdym zdarzeniu; chmura punktów; trajektoria (t_k, T_k).
Poza T = (t, q, s): punkt świata = s · R(q) · X + t.

**Inicjalizacja**

1. (h, w) ← kształt wejścia sieci: dłuższy bok I_0 do 512 px, crop centralny do wielokrotności 16.
2. Utwórz pamięć współdzieloną: zbiór klatek kluczowych KF, stan bieżącej klatki S, kolejkę zadań Q.
   tryb ← INIT. Wczytaj sieć f.
3. Uruchom proces backendu B. B w pętli: dla zadania k z Q — dobierz sąsiadów (k − 1 oraz klatki
   z wyszukiwania obrazowego), dodaj krawędzie grafu, optymalizuj Sim(3) wszystkich klatek
   kluczowych poza k = 0 (Gauss-Newton po promieniach); w trybie RELOC — relokalizacja.

**Dla każdej klatki (I_i, i, t_i)**

4. F ← przeskalowane I_i / 255; T_F ← tożsamość dla pierwszej klatki, inaczej poza poprzedniej.
5. **Jeśli** tryb = INIT: (X_F, C_F) ← f(F, F); KF ← KF ∪ {F}; ZRZUT(k, „created”);
   Q ← Q ∪ {k}; tryb ← TRACKING.
6. **Jeśli** tryb = TRACKING: (nowa, zgubiona) ← ŚLEDŹ(F, ostatnia klatka kluczowa K):
   dopasowanie f(F, K), optymalizacja T_F (Sim(3)) względem K, uśrednienie X_K ważone ufnością.
   **Jeśli** zgubiona: tryb ← RELOC.
7. **Jeśli** tryb = RELOC: (X_F, C_F) ← f(F, F); zleć relokalizację w B; **jeśli** single_thread —
   czekaj na jej koniec. Udana relokalizacja dokłada F do KF w procesie B.
8. **Jeśli** nowa: KF ← KF ∪ {F}; ZRZUT(k, „created”); Q ← Q ∪ {k};
   **jeśli** single_thread — czekaj, aż Q = ∅, potem HISTORIA(„backend_done”).
9. Klatki kluczowe dołożone przez B (wzrost |KF|): ZRZUT(k, „created”), HISTORIA(„reloc”).
   **Jeśli** nie single_thread i B zakończył zadanie od ostatniego sprawdzenia: HISTORIA(„backend_done”).
10. Zwróć (T_F, X_F, C_F / N_F, tryb, k — jeśli F jest klatką kluczową).

**Zakończenie**

11. tryb ← TERMINATED; czekaj na zakończenie B.
12. Dla każdego k ∈ KF: ZRZUT(k, „final”). HISTORIA(„final”).
13. Chmura ← ⋃_k { T_k · X_k(p) : C_k(p) / N_k > 1,5 }. Trajektoria ← (t_k, T_k) z zachowaną skalą s_k.

**Procedury**

- ZRZUT(k, etap): zapisz (i_k, t_k, T_k, X_k, C_k, N_k) klatki kluczowej k.
- HISTORIA(e): dopisz T_k wszystkich klatek kluczowych z etykietą zdarzenia e.

Kroki 4–8 i 11 odtwarzają pętlę `main.py` upstreamu; kroki ZRZUT i HISTORIA tylko czytają stan.
