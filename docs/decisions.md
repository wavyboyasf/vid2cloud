# Decyzje architektoniczne

Każdy wpis: **co** / **dlaczego** / **alternatywa**. Krótko — trzy zdania na sekcję wystarczą.

---

## 2026-09-09 — MASt3R-SLAM jako submodule, cały kontakt przez adapter w `engine/`

**Co.** Upstream MASt3R-SLAM wchodzi do repozytorium jako submodule w `third_party/mast3r-slam/`
i nie jest modyfikowany. Jedynym miejscem w projekcie, które zna jego API, jest adapter
`vid2cloud/engine/mast3r_slam.py`, implementujący wspólny interfejs z `vid2cloud/engine/base.py`.
Reszta pakietu — wejście, skala, eksport — widzi tylko ten interfejs.

**Dlaczego.** Submodule przypina konkretny commit upstreamu, więc widać dokładnie, na jakiej
wersji silnika powstały wyniki, a `git diff` repozytorium pokazuje wyłącznie wkład własny —
przy pracy dyplomowej to argument przy obronie, nie tylko higiena. Adapter daje drugą korzyść:
jeśli trzeba będzie porównać silniki (CUT3R, SLAM3R), podmienia się jedną klasę zamiast
przepisywać pipeline.

**Alternatywa: fork.** Fork byłby wygodniejszy, bo uruchomienie MASt3R-SLAM na RTX 5060 Ti
(sm_120, torch 2.11+cu128) wymagało poprawek w źródłach upstreamu — gencode `sm_120` w `setup.py`,
`.type()` → `.scalar_type()` w kernelach CUDA, `at::linalg_norm` zamiast `torch::linalg::linalg_norm`,
`weights_only=False` przy wczytywaniu checkpointów NAVER. Odrzucony, bo fork zaciera granicę między
kodem cudzym a własnym i wymaga ręcznego nadążania za upstreamem. **Konsekwencja do rozwiązania:**
te poprawki trzeba trzymać poza submodule — jako serię patchy nakładanych skryptem przy
instalacji środowiska. Sposób nie jest jeszcze wybrany, pozycja jest w `docs/TODO.md`.
