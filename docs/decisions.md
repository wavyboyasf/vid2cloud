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

---

## 2026-09-15 — Pomiar wydruku próbnego A4: drukarka domowa zmniejsza o ok. 2,7 %

**Co zmierzono.** Wydruk `boards/test_A4_150mm/test.pdf` (jedna strona A4, ID 3, nominał
150,0 mm) na drukarce domowej autora. Bok czarnego kwadratu od zewnętrznej krawędzi do
zewnętrznej krawędzi, linijką Maped 15 cm z podziałką milimetrową, zdjęcia pomiaru
w `~/Pobrane/POMIAR CM\`:

| kierunek | odczyt | / 150,0 mm | różnica |
|---|---|---|---|
| pion | 145,9 mm | 0,972667 (−2,733 %) | −4,1 mm |
| poziom | 146,0 mm | 0,973333 (−2,667 %) | −4,0 mm |

Różnica między osiami to 0,1 mm (+0,069 %). Rozdzielczość odczytu linijki milimetrowej
to ok. 0,5 mm, czyli ok. 0,33 % przy 150 mm — anizotropia jest **poniżej** tej granicy
i z tego pomiaru nie da się orzec, czy w ogóle istnieje.

**Czego ten pomiar NIE rozstrzyga.** Przyczyny. −2,7 % to wartość typowa zarówno dla
ustawienia „dopasuj do obszaru drukowania" w sterowniku (skalowanie programowe, do
wyłączenia), jak i dla mechanicznego skalowania samego urządzenia (nie do wyłączenia).
Rozstrzygnąłby to drugi wydruk z jawnie ustawioną skalą 100 % / „rozmiar rzeczywisty"
i ponownym pomiarem — **nie wykonany**. Wniosek zostawiam autorowi.

**Konsekwencja, która nie zależy od przyczyny.** `nominal_size_mm` z `boards.json` nie
nadaje się na wzorzec skali — 2,7 % to o rząd wielkości więcej niż dokładność, o którą
w tej pracy chodzi. Każdy wydruk musi mieć własny wpis w `targets.json` ze **zmierzoną**
wartością. To potwierdza ostrzeżenie wpisane do README i do samego generatora.

**Konsekwencja dla łaty skali: żadna.** Wzorcem jest odległość między środkami dwóch
markerów na listwie, mierzona po sklejeniu, a nie bok wydrukowanego markera. Skalowanie
wydruku nie wchodzi do tego rachunku, a środek czworokąta (średnia czterech narożników)
nie przesuwa się przy skalowaniu ani przy anizotropii. Łatę można więc drukować w domu;
ksero zostaje potrzebne wyłącznie do dużych plansz.

---

## 2026-09-15 — Skala: współczynnik osobno dla każdego wzorca, jedna liczba tylko jawnie

**Co.** `vid2cloud/scale/estimate.py` liczy współczynnik dla każdego segmentu wzorcowego
osobno i zwraca listę per segment plus wiersz zbiorczy (mediana, średnia, min, max, std,
rozrzut). Sprowadzenie do jednej liczby jest oddzielnym, jawnym krokiem (`select_factor()`,
w CLI `--use median` albo `--use-segment NAZWA`), a jego opis — „mediana z N segmentów"
albo „segment 'W1'" — trafia razem z kompletem wyników per segment do `<out>.scale.json`
obok przeskalowanej chmury. Jednostki: `targets.json` w milimetrach, pomiar z chmury
bezwymiarowy, `factor` przelicza jednostki modelu na **metry**; zamiana mm→m siedzi
w jednym miejscu (`MM_PER_M` w `_segment_factor()`).

**Dlaczego.** Na chmurze z czerwca ten sam przebieg daje ×1,1489 przy kotwicy (W1)
i ×1,0000 kilka metrów dalej przy stoliku (W2) — rozrzut 13,86 % wewnątrz jednej mapy
(uruchomione dziś na `data/own/targets_test.json` + `measured_test.csv`, dane z pomiarów
11.09). Moduł zwracający jedną liczbę zgubiłby dokładnie ten fakt, o który w tej pracy
chodzi. Przy rozrzucie powyżej 5 % (`SPREAD_WARN_PCT`) CLI dopisuje linię ostrzeżenia,
ale liczb nie zmienia i outlierów nie odrzuca.

**Alternatywa: średnia ważona 1/σ².** Dałaby jedną liczbę z niepewnością i wyglądałaby
poważniej, lecz zakłada, że wszystkie segmenty mierzą tę samą wielkość zaburzoną błędem
losowym — a różnica między W1 i W2 jest systematyczna (skala zmienia się wzdłuż mapy).
Odrzucona; gdyby wróciła, to jako pozycja obok listy per segment, nie zamiast niej.

**Niepewność pomiaru: odchylenie próbki, nie błąd mediany.** Przy powtórzeniach
(5 point-pickingów) `sigma_measured` to odchylenie standardowe próbki, ddof=1 — dla W1
1,28 cm. Nie σ/√n (≈ 0,6 cm), bo kliknięcia nie są niezależnymi losowaniami: to ten sam
operator celujący w ten sam, niejednoznacznie zdefiniowany narożnik, więc dzielenie przez
√n zaniżałoby niepewność. Propagacja jest względna,
`sigma_factor = factor * sqrt((sigma_ref/ref)² + (sigma_meas/meas)²)`; przy jednym pomiarze
bez podanej sigmy `sigma_factor` to `None`, nie zero — liczby, której nie znamy, moduł
nie wymyśla.

**Czego nie uruchomiono.** `scale_ply()` (open3d) — open3d nie jest zainstalowany ani
w `vid2cloud-tools`, ani w `mast3r-slam`, więc import jest leniwy, a ścieżka `--apply`
kończy się czytelnym błędem. Testem objęta jest tylko czysta część zapisu
(`build_scale_record()`). Pozycja w `docs/TODO.md`.

---

## 2026-09-17 — Pierwszy pomiar ArUco na nagraniu: bok markera przy 512 px na 2–3 m (wnętrze)

**Co zmierzono.** Nagranie `data/own/proba_generalna.mp4` (oryginał `VID_4k_poziom.mp4`: h264,
3840×2160 bez rotacji w metadanych, 30 fps, 39,87 s, 1196 klatek), klatka 30 (1,01 s).
Detekcja `python -m vid2cloud.scale.aruco` na pełnej klatce, bok przeliczony na wejście modelu
512×288 (skala 512/3840 = 0,1333 w obu osiach). **Odległości kamera–plansza to szacunek autora,
nie pomiar.** Wymiary wzorców są nominalne — boku tych wydruków nie mierzono.

| ID | wzorzec (nominał) | odległość (szac.) | bok 4K: śr. / min–max | przy 512: śr. / min–max | min/max boku |
|---|---|---|---|---|---|
| 8 | plansza A4, 150 mm | poniżej 2 m | 153,69 / 96,02–214,78 px | 20,49 / 12,80–28,64 px | 0,45 |
| 10 | plansza A4, 150 mm | ok. 2,2 m | 118,79 / 102,57–136,16 px | 15,84 / 13,68–18,15 px | 0,75 |
| 48 | łata, 70 mm | ok. 2,5–3 m | 75,56 / 68,86–82,22 px | 10,08 / 9,18–10,96 px | 0,84 |
| 49 | łata, 70 mm | ok. 2,5–3 m | 56,74 / 54,66–58,94 px | 7,57 / 7,29–7,86 px | 0,93 |

**Warunki.** Plansze A4 leżą płasko na podłodze i są widziane skośnie — stąd rozrzut boków
(kolumna min/max). Markery łaty są przyklejone na końcach tuby leżącej na łóżku. Kamera z ręki.

**Całe nagranie** (`--render`, `runs/proba/podglad.csv`): średni bok w 4K w klatkach z wykryciem —
ID 3: 178,64–266,00 px (111 klatek), ID 8: 127,94–232,04 (278), ID 9: 174,32–221,46 (92),
ID 10: 116,03–166,63 (170), ID 11: 252,69–415,07 (128), ID 48: 68,36–145,89 (385),
ID 49: 56,22–168,51 (399); ID 12 nie wykryty ani razu. Fałszywe wykrycia ID 17 i 37 — w `docs/TODO.md`.
Klatki bez wykrycia obejrzane ręcznie: w 200 oba markery łaty są całe w kadrze, ale rozmyte
ruchem; w 400 widoczne markery są ucięte krawędzią kadru.

**Czego ten pomiar NIE rozstrzyga.**
- Zależności bok(odległość) ani ogniskowej — odległości są szacunkiem, a plansze na podłodze
  skrócone perspektywą. Nie wyprowadzam z tego żadnej z tych liczb.
- Progu detekcji: najmniejszy średni bok prawdziwego markera w całym nagraniu to 56,22 px w 4K,
  a detektor pracuje na 4K — do 12 px z syntetyki (wpis z 2026-09-13) to nagranie nie dochodzi.
- Czy 15 px przy 512 wystarcza do liftu — to pytanie do sesji o lifcie. Tu jest tylko zestawienie
  z tym progiem: średni bok ID 8 i 10 jest powyżej, ID 48 i 49 poniżej; najkrótszy bok ID 8 i 10
  (12,80 i 13,68 px) też jest poniżej.
- Zasięgu w terenie — to wnętrze i najwyżej ok. 3 m.

---

## 2026-09-19 — Nagranie zasięgowe: trzy plansze 150 mm na ok. 1,0 / 3,0 / 4,5 m

**Co zmierzono.** `data/own/VID_ODLEGLOSC.mp4` (kopia z `~/Pobrane`; h264, 3840×2160 bez rotacji,
30 fps, 15,69 s, 465 klatek). Detekcja: `python -m vid2cloud.scale.aruco --video ... --render
runs/odleglosc/podglad.mp4`, liczby per klatka w `runs/odleglosc/podglad.csv`. Klatki 0–23 to
jedyny fragment, w którym widać wszystkie trzy plansze naraz; poniżej klatka 0.

**Odległości — szacunek autora, po korekcie.** Autor mierzył je od framugi, przez którą potem
przeszedł, i po obejrzeniu nagrania skorygował: do każdej wartości trzeba dodać ok. 0,5 m.
Wartości przyjęte tutaj (0,5 / 2,5 / 4 m + 0,5 m) to nadal szacunek, nie pomiar taśmą.

| ID | odległość (szac.) | bok 4K: śr. / min–max | przy 512: śr. | min/max boku |
|---|---|---|---|---|
| 11 | ok. 1,0 m (blat) | 237,89 / 236,59–239,01 px | 31,72 px | 0,99 |
| 8 | ok. 3,0 m (framuga) | 125,27 / 125,04–125,57 px | 16,70 px | 1,00 |
| 10 | ok. 4,5 m (drzwi w korytarzu) | 87,74 / 87,40–88,17 px | 11,70 px | 0,99 |

Wszystkie trzy są w tej klatce praktycznie na wprost (min/max ≈ 1), więc skos nie skraca boku.

Odległości są przybliżone, więc tego zapisu nie należy używać do wyprowadzania geometrii
(ogniskowej, odległości z rozmiaru markera) — twardą częścią pomiaru są boki w pikselach.

**Czego ten pomiar NIE rozstrzyga.** Progu detekcji: wszystkie trzy plansze są wykryte na pełnej
klatce, najmniejsza ma 87,74 px boku w 4K. Przydatności do liftu też nie — jest tylko zestawienie
z progiem 15 px przy 512 z planu autora: ID 10 jest poniżej (11,70 px), ID 8 i 11 powyżej.
W całym nagraniu ID 10 wypadł poniżej progu w 125 z 348 klatek z wykryciem.

---

## 2026-10-02 — Rotacja z metadanych: stosujemy ją u siebie, bo dekoder upstreamu stosuje ją sam

**Co.** `vid2cloud/scale/aruco.py` czyta kąt obrotu z display matrix kontenera
(`parse_display_matrix()`, `frame_rotation()`, `video_rotation()`) i domyślnie obraca klatkę
**przed** detekcją — i w `read_video_frame()`, i w `render_video()`. Wyłącza to `--no-rotate`.
Wcześniej moduł rotacji nie stosował w ogóle (wpis z 2026-09-13 opisywał ją jako nieistotną,
bo nie zmienia boku markera w pikselach).

**Dlaczego.** Dwa z sześciu nagrań z 21.09 mają `rotation=-90` (`VID_20260921_125536`,
`VID_20260921_133034`) — telefon trzymany pionowo zapisuje kadr jako 3840x2160 i dokłada
macierz obrotu. PyAV wydaje piksele tak, jak leżą w pliku, a `cv2.VideoCapture` — czyli
dekoder `MP4Dataset` upstreamu — stosuje orientację **automatycznie**. Bez tej poprawki nasze
narożniki byłyby w innym układzie współrzędnych niż pointmapa silnika, co psuje lift, a nie
tylko podgląd. Kierunek obrotu jest zmierzony, nie zgadnięty: klatka 0 z PyAV obrócona
`ROTATE_90_CLOCKWISE` zgadza się z klatką z `cv2.VideoCapture` (średnie |diff| 1,7 na kanał,
różnica konwersji barw), a obrócona w drugą stronę nie (90,1).

**Na zmierzone boki markerów to nie wpływa.** Dłuższy bok kadru to 3840 niezależnie od
orientacji, więc skala do wejścia modelu jest ta sama (512/3840), a 288 i 512 są
wielokrotnościami 16, więc crop jest zerowy w obu układach. Liczby z wpisów 2026-09-17
i 2026-09-19 zostają w mocy; zmienia się kształt kadru i współrzędne narożników.

**Alternatywa: obracać dopiero narożniki, po detekcji.** Taniej (bez obrotu 4K na każdej
klatce), ale wymaga pamiętania o obrocie w każdym miejscu, które dotyka współrzędnych —
podgląd, `ids/`, lift, przyszłe ROI. Odrzucone: jedna transformacja przy wejściu jest
trudniejsza do zgubienia niż cztery poprawki w dalszych krokach. Koszt: obrót o wielokrotność
90° to przepisanie bufora bez interpolacji, w pomiarze nie wyszedł poza szum czasu renderu.

---

## 2026-10-03 — Sesja 21.09: skala ręcznie na 3 nagraniach; nic nie wykluczamy
Co: 15 odcinków taśmy (5 × 3 nagrania) zmierzone w chmurach upstreamowego main.py; rozrzut
współczynnika wewnątrz nagrania 11,1 / 15,3 / 12,9 % raportowany jako wynik. Żadne nagranie ani
odcinek nie jest odrzucany; jakość odcinka opisuje powtarzalność kliknięć (sigma_klikniec / mediana),
a dwumodalne kliknięcia oznaczają zdublowaną geometrię (pęknięcie mapy), nie dryf.
Dlaczego: rozrzut skali jest badanym zjawiskiem (dryf skali w monokularnym SLAM-ie) — odrzucanie
nagrań z dużym rozrzutem usuwałoby z próby dokładnie to, co mierzymy. Próg 5 % z wcześniejszej
propozycji nie był ustalony przed pomiarem i nie jest używany.
Alternatywa: kryterium jakości po rozrzucie (odrzucone — miesza zjawisko z błędem pomiaru).
Literatura: Strasdat, Montiel, Davison 2010 (Scale drift-aware large scale monocular SLAM, RSS);
Murai, Dexheimer, Davison 2025 (MASt3R-SLAM, CVPR) — Sim(3) i pin pierwszej klatki kluczowej.
