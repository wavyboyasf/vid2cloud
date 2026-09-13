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
projekt testu przewiduje najpierw copy. (Zmienione 2026-09-13 — patrz wpis niżej: re-encode
wszystkiego jest teraz domyślny, copy-first został pod `--no-reencode-all`.) **Konsekwencja do decyzji autora:** w VID20263.mp4
keyframe'y są co ~1,0165 s, więc spośród startów 0,3,…,18 s tylko 0 s przechodzi przez copy —
`clip_0` zostaje w oryginalnym HEVC, pozostałe są w x264 CRF 18.

---

## 2026-09-13 — Test C: wszystkie klipy przekodowane jednakowo, x264 z GOP 60

**Co.** `eval/test_c_start_point.py` domyślnie (`--reencode-all`) przekodowuje każdy klip tymi
samymi parametrami, także ten, którego start wypada na keyframie, i wymusza `-g 60` zamiast
domyślnych 250 w libx264. Próba `-c copy` została pod flagą `--no-reencode-all`.

**Dlaczego.** Klipy porównujemy między sobą, więc kodek nie może być zmienną: przy copy-first
`clip_0` zostawał w oryginalnym HEVC, a reszta szła w x264. GOP skrócono z powodu zmierzonej luki
I/O upstreamu: przebieg na `clip_18` (12 s, 720 klatek, GOP 250) trwał **480 s przy obciążeniu GPU
ok. 2 % i CPU ok. 1200 %** — `MP4Dataset` bez torchcodec ustawia `CAP_PROP_POS_FRAMES` na każdą
klatkę (`dataloader.py:257`), więc z lektury kodu OpenCV przewija do poprzedniego keyframe'u
i dekoduje średnio pół GOP-u 4K na każdą wydaną klatkę. GOP 60 zbliża klip do źródła (keyframe co
~61 klatek). Czy faktycznie skraca przebieg — **nie zmierzone**, do sprawdzenia na pierwszym
przebiegu silnika po tej zmianie.

**Alternatywa: torchcodec w env silnika.** Upstream użyłby go automatycznie zamiast OpenCV, ale
instalacja ruszałaby zależności torcha w env `mast3r-slam`, czego CLAUDE.md zabrania. Druga
alternatywa — własne dekodowanie PyAV przez adapter `engine/mast3r_slam.py` — jest szersza niż
test C i czeka w `docs/TODO.md`.

---

## 2026-09-13 — ArUco: detekcja na pełnej klatce, przeliczenie odtwarza `resize_img()` upstreamu

**Co.** `vid2cloud/scale/aruco.py` wykrywa markery na pełnej rozdzielczości klatki
(`detect()`, `cv2.aruco.ArucoDetector` z `CORNER_REFINE_SUBPIX`), a osobna funkcja
`rescale_corners()` przelicza narożniki na rozdzielczość wejścia modelu. Przeliczenie
nie jest samą skalą: reimplementuje całe `resize_img()` z
`third_party/mast3r-slam/mast3r_slam/mast3r_utils.py:244` — dłuższy bok do 512, każdy
bok zaokrąglany osobno, a potem **crop centralny** do wielokrotności 16 px
(`mast3r_utils.py:256-264`). `to_shape` służy do sprawdzenia wyniku, nie do jego wyliczenia.

**Dlaczego.** Marker 12 cm z 3 m ma przy 512 px około 15 px, więc detekcja musi iść po
pełnej klatce — inaczej traci się dokładność narożników, na której stoi cała skala. Crop
trzeba uwzględnić, bo dla proporcji innych niż 16:9 i 4:3 nie jest zerowy: 3840x1634
skaluje się do 512x218, a potem traci 5 px z góry i 5 z dołu. Reimplementacja została
sprawdzona względem PIL (`tests/test_aruco.py::test_input_transform_matches_upstream`,
9 formatów, zgodność skali, offsetu cropu i kształtu wyjścia).

**Alternatywa: wywołać `resize_img()` upstreamu.** Dałoby zgodność z definicji, ale
wciągnęłoby `mast3r_utils` (a z nim torcha i checkpointy) do env `vid2cloud-tools`, gdzie
detekcja ArUco ma działać bez GPU. Odrzucone; ceną jest test porównawczy, który trzeba
przejrzeć przy zmianie commitu submodule.

**Konsekwencja do rozstrzygnięcia przy lifcie.** Konwencja półpiksela. Narożniki
z `cv2.aruco` są w konwencji środka piksela, więc `rescale_corners()` liczy
`x' = (x + 0.5) * W/W1 - 0.5 - left`. Upstream w `Intrinsics` (`dataloader.py:292`) liczy
punkt główny bez tej poprawki — różnica to 0.5 * (W/W1 - 1), czyli ok. 0.43 px przy
4K -> 512. Dotyczy tylko trybu z kalibracją, ale jeśli lift będzie próbkował pointmapę
przez interpolację, to jest to pół piksela w miejscu, gdzie 15-pikselowy marker ma
definiować skalę. Pozycja w `docs/TODO.md`.

---

## 2026-09-13 — Minimalny rozmiar markera: zmierzony próg to 12 px boku na czystej syntetyce

**Co.** Prompt sesji pytał, czy detekcja działa przy boku ~15 px (tyle ma marker 12 cm
z 3 m przy wejściu 512 px). **Działa.** Zmierzony próg stabilnej detekcji — najmniejszy
bok, od którego marker jest wykrywany nieprzerwanie aż do 40 px — zależy od wzoru ID
i wyniósł: ID 0 — 12 px, ID 7 — 9 px, ID 23 — 8 px, ID 49 — 8 px. Największy z nich, 12 px,
jest w `tests/test_aruco.py` jako `MIN_MARKER_PX` i test pilnuje, żeby próg nie podskoczył.

**Warunki pomiaru.** Marker wygenerowany `generateImageMarker` w 600 px, otulina białego
tła równa bokowi markera, cała scena zmniejszona `INTER_AREA`, kąt 0°, bez kompresji.
Największy błąd narożnika względem prawdy w tych warunkach to 0,19 px niezależnie od
rozmiaru (0,09 px przy 60 px boku). Po dodaniu szumu gaussowskiego σ = 8 i rozmycia
σ = 0,6 px detekcja nadal działa od 10 px, ale błąd narożnika rośnie i przy 15 px sięgnął
0,99 px — przy 60 px jest to 0,18 px.

**To jest granica optymistyczna, nie robocza.** Brak w pomiarze: kąta patrzenia, rozmycia
ruchu, kompresji wideo, nierównomiernego oświetlenia i tego, że wydruk nie jest idealnie
czarno-biały. Liczba mówi tylko, że przy 15 px nie zderzamy się ze ścianą algorytmu —
ile trzeba **naprawdę**, pokaże pomiar na nagraniu. Nie wyciągam z tego wniosku o odległości
roboczej od markera.

**Konsekwencja.** Skoro błąd narożnika przy małym markerze i szumie jest rzędu 1 px,
a narożnik jest potem liftowany do 3D, to przy definiowaniu datum skali lepiej opierać się
na odległości między środkami dwóch markerów łaty (uśrednienie po 4 narożnikach każdego)
niż na boku pojedynczego markera. Do sprawdzenia w sesji o lifcie.
