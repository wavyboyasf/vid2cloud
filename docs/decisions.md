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

---

## 2026-09-11 — Test C: cięcie klipów copy-albo-re-encode z weryfikacją ffprobe, wyniki hardlinkami

**Co.** `eval/test_c_start_point.py` tnie klip `-c copy` tylko wtedy, gdy pierwsza zdekodowana
klatka jest keyframe'em w odległości ≤ 1 klatki od startu i nie ma pre-rollu z flagą discard;
w przeciwnym razie robi re-encode `libx264 -crf 18 -fps_mode passthrough -frames:v N`, gdzie N to
liczba klatek źródła w [start, end). Chmura i trajektoria z `third_party/mast3r-slam/logs/test_c_<X>/`
trafiają do `runs/test_c/clip_<X>/` jako hardlinki, a `results.csv` jest budowany ze wszystkich
`clip_*/config.json` i przenosi kolumny pomiarowe ze swojej poprzedniej wersji.

**Dlaczego.** Zmienną eksperymentu jest start, więc pierwsza klatka musi być dokładna: `-c copy`
zaczyna od keyframe'u *przed* startem i chowa pre-roll listą edycji MP4, a OpenCV (dekoder silnika
bez torchcodec) i tak liczy go w długości klipu. Z kolei `-to` przy re-encode przepuszcza jedną
klatkę po końcu (trim liczy czas od pierwszej zachowanej klatki), stąd `-frames:v N` — dzięki temu
wszystkie klipy kończą się na tej samej klatce źródła. Hardlinki chronią wyniki w `runs/` przed
upstreamem, który na starcie kasuje `logs/<save-as>/input.ply`, i nie podwajają zajętości dysku.

**Alternatywa: re-encode wszystkich klipów.** Dałoby jednolity kodek; odrzucone na razie, bo
projekt testu przewiduje najpierw copy. **Konsekwencja do decyzji autora:** w VID20263.mp4
keyframe'y są co ~1,0165 s, więc spośród startów 0,3,…,18 s tylko 0 s przechodzi przez copy —
`clip_0` zostaje w oryginalnym HEVC, pozostałe są w x264 CRF 18.
