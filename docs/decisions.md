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

---

## 2026-10-03 — Adapter MASt3R-SLAM: pętla `main.py` odtworzona u nas, backend upstreamu bez kopii

**Co.** `vid2cloud/engine/mast3r_slam.py` powtarza pętlę `main.py:233-310` (INIT / TRACKING / RELOC,
kolejka optymalizacji, czekanie przy `single_thread`), a wszystkie jej elementy importuje z upstreamu.
Proces backendu uruchamia oryginalne `main.run_backend`: katalog upstreamu jest dopisany **na końcu**
`sys.path`, a proces `spawn` dziedziczy `sys.path` rodzica i importuje `main` po nazwie — sprawdzone
przed napisaniem adaptera (dziecko weszło do `run_backend`). Kopia `run_backend`/`relocalization`
nie była potrzebna. Do pętli dochodzą wyłącznie odczyty pod `keyframes.lock`: zrzut klatki kluczowej
przy powstaniu (nowe wykrywane po `len(keyframes)`, więc także z relokalizacji) i na końcu
(`keyframes/kf_*_{created,final}.npz`) oraz pozy Sim(3) wszystkich klatek kluczowych po każdym
zdarzeniu (`poses_history.csv`).

**Dlaczego.** Komponenty 4 i 5 potrzebują pointmapy klatki kluczowej w chwili powstania i skali s_k
po każdej optymalizacji, a upstream nie zapisuje pointmap, a `save_traj` gubi skalę (`as_SE3`).
Odtworzenie pętli to mniej kodu niż łatka w `third_party/` i nie łamie zasady nietykania upstreamu.
`single_thread` jest domyślnie włączone dla pliku: po każdej nowej klatce kluczowej pętla czeka na
koniec optymalizacji (`main.py:301`), więc zrzuty odpowiadają zdarzeniom, a nie temu, jak szybko
liczy GPU. `--no-single-thread` daje zachowanie upstreamu (tryb na żywo). Sprawdzone 03.10 na 125437: dwa
przebiegi z single_thread dały bajt w bajt identyczne `cloud.ply`, `poses_history.csv` i wszystkie
zrzuty `.npz`; przebieg bez single_thread wybrał te same klatki kluczowe, a s_k różniło się od
pierwszego o 0 do 0,556 % (`runs/2026-09-21/adapter/powtarzalnosc_125437.csv`).

**Odstępstwa od `main.py` — wokół pętli, nie w niej.** (1) `single_thread` nadpisywane flagą CLI
(base.yaml ma False). (2) Backend startuje z cwd = katalog upstreamu, bo `load_retriever` czyta
checkpoint ścieżką względną (`mast3r_utils.py:26`); model główny ładowany ścieżką bezwzględną.
(3) `finalize()` dołącza backend **przed** zapisem; upstream zapisuje przed `join()`, co bez
single_thread ściga się z ostatnią optymalizacją. (4) Bez single_thread zdarzenie „backend_done”
jest próbkowane po każdej klatce — kilka optymalizacji między dwiema klatkami zlewa się w jedno.
(5) Pozę klatki kluczowej z relokalizacji widać dopiero po optymalizacji wewnątrz `relocalization()`,
stąd osobne zdarzenie „reloc” zamiast „kf_created”.

**Alternatywa: łatka z hakami w `main.py`/`frame.py` albo kopia `run_backend` z dopisanymi hakami.**
Odrzucone: pierwsza modyfikuje upstream, druga zmienia logikę skopiowanego kodu. Jedyne, czego
w obecnym podejściu brakuje, to poza klatki z relokalizacji sprzed optymalizacji.

Literatura: Murai, Dexheimer, Davison 2025 (MASt3R-SLAM, CVPR) — podział frontend/backend, graf
klatek kluczowych, Sim(3) z pinem pierwszej klatki; sposób podpięcia — brak, decyzja inżynierska.

---

## 2026-10-03 — Źródło klatek: PyAV w env silnika, czas z PTS kontenera, rotacja w źródle

**Co.** `vid2cloud/input/sources.py::FileSource` dekoduje PyAV sekwencyjnie (bez seeków), obraca
klatkę wg display matrix funkcjami z `scale/aruco.py` i zwraca PTS w sekundach **bez przesuwania
do zera** (pierwsza klatka 125437 ma 0,0243 s — tak samo jak w `podglad.csv` z `--render`).
Do env `mast3r-slam` doinstalowano `av 18.1.0` i `pytest 9.1.1` (z nimi `pluggy 1.6.0`,
`iniconfig 2.3.0`); `pip install --dry-run` pokazał, że torch 2.11.0+cu128 i numpy 1.26.4 zostają,
i po instalacji to sprawdzono.

**Dlaczego.** Dekoder musi siedzieć w procesie silnika, a PyAV jest jedynym dozwolonym dekoderem.
Czas z PTS, bo `MP4Dataset` liczy `idx / fps` i zakłada stały fps. Rotacja w źródle, bo
`cv2.VideoCapture` upstreamu stosuje ją sam (wpis z 2026-10-02) — bez tego silnik widziałby
inny kadr niż w przebiegach z 02.10, a ArUco inny niż silnik. pytest w env silnika jest potrzebny
tylko dla testu porównującego transformację wejścia adaptera z `scale/aruco.py`.

**Alternatywa: dekodowanie w osobnym procesie w env `vid2cloud-tools`.** Odrzucone — klatka 4K RGB
to ok. 25 MB do przesłania między procesami, a zysku nie ma, skoro `av` nie rusza torcha ani numpy.

Literatura: brak — decyzja inżynierska.

---

## 2026-10-05 — Konwencja półpiksela w lifcie: środek piksela, sprawdzona na obrazie wejścia sieci

**Co.** `scale/lift.py` przyjmuje konwencję z `scale/aruco.py::rescale_corners()`:
x' = (x + 0,5)·s − 0,5 − left, a piksel pointmapy (wiersz i, kolumna j) ma środek w (x = j, y = i).
Sprawdzone na danych (`eval/half_pixel_check.py`): ArUco wykryte bezpośrednio na obrazie wejścia
sieci (`keyframes/kf_NNN.png`, ten sam crop co pointmapa) porównane z narożnikami z pełnej klatki
przeliczonymi na 512 — 152 narożniki, 38 markerów, trzy przebiegi z 03.10. Konwencja środka:
średnia różnica +0,033 px (x) / −0,023 px (y), mediana +0,012 / 0,000 px. Konwencja narożnika
piksela (x' = x·s − left, jak `Intrinsics` upstreamu): średnia +0,467 / +0,410 px, mediana
+0,445 / +0,433 px, przy oczekiwanym przesunięciu 0,5·(1 − s) = 0,433 px. Odchylenie standardowe
różnic jest w obu konwencjach takie samo: 0,83 px (x) i 0,40 px (y).

**Czego to nie sprawdza.** Wyrównania pointmapy z obrazem piksel w piksel — pointmapa nie niesie
wzoru markera. To przyjmujemy z konstrukcji (sieć zwraca wartość dla każdego piksela wejścia).

**Wrażliwość.** Przeliczenie narożników konwencją narożnika zamiast środka zmienia d_canon 17
odcinków (stan końcowy) o −0,42 … +0,33 % (mediana −0,024 %).

**Alternatywa: konwencja narożnika piksela.** Odrzucona — powyższe różnice jej przeczą.
Pozycja „Konwencja półpiksela” w `docs/TODO.md` zamknięta.

Literatura: brak — sprawdzenie na danych.

---

## 2026-10-05 — Lift punktów odniesienia taśmy: płaszczyzna planszy + promień, narożniki ArUco

**Co.** W `targets.json` z 21.09 żaden odcinek nie jest mierzony środek–środek: punkty odniesienia
to narożniki („lewy-gorny”), środki krawędzi („srodek-gornej”) i krawędzie „pod kątem”. Decyzje
autora z 05.10: (1) narożnik liftowany jako przecięcie promienia w pikselu narożnika z płaszczyzną
planszy; płaszczyzna (SVD) i promień r = (X/Z, Y/Z), afiniczny w (x, y), dopasowane do pikseli
wnętrza markera (czworokąt 50 % wokół przecięcia przekątnych, C/N ≥ 1,5, co najmniej 5 pikseli);
(2) narożniki w kolejności ArUco w układzie markera: 0 lewy-górny, 1 prawy-górny, 2 prawy-dolny,
3 lewy-dolny — potwierdzone przez autora na `runs/2026-09-21/adapter/podglad_narozniki.png`
(daleko_wolno, kf 3, markery 3, 8, 10, 11, 12); narożnik 0 jest wizualnie lewym górnym w 40 z 40
wykryć ID z targets.json w klatkach kluczowych; (3) środek krawędzi = środek odcinka 3D między
jej narożnikami po lifcie, nie środek 2D na obrazie; (4) „prawa/lewa-krawedz, pod katem”
(3-8, 10-11) = górny róg tej krawędzi, zgodnie z uwagą w `pomiary_tasma.csv` i z klikaniem
w CloudCompare. Taśma była mierzona od narożników czarnego kwadratu, więc narożniki ArUco są
punktami pomiaru bez przesunięcia. Środek markera (nakładka `markers_world.ply`) liczony medianą
X w tym samym czworokącie 50 %. Nieznana nazwa punktu albo kwalifikator to błąd przed liczeniem.

**Dlaczego.** Piksel narożnika leży przy 512 px ok. 3–7 px od krawędzi kartki, za którą jest tło
oddalone o metry; płaszczyzna + promień wyznacza narożnik z wnętrza markera i brzegu nie próbkuje.
Dla płaskiej planszy i kamery otworkowej (X/Z afiniczne w pikselach) przecięcie jest dokładne:
`tests/test_lift.py` odtwarza odcinek na płaszczyźnie skośnej do 1e-6 (float64; dla pointmapy
float32 błąd narożnika < 1e-5). Rozrzut (MAD) to mediana odległości pikseli wnętrza od ich
mediany — mierzy także rozmiar plamy pikseli, nie tylko szum; obok jest RMS do płaszczyzny.

**Diagnostyka bez progu.** Ta sama odległość z mediany X w oknie 3×3 przy narożniku, różnica
`L · (d_3×3 / d − 1)`: przy powstaniu mediana −2,3 mm (|·| 5,2 mm, od −65,5 do +43,1 mm), na końcu
mediana +0,9 mm (|·| 3,8 mm, od −5,1 do +28,7 mm), 17 odcinków.

**Alternatywy.** Mediana w oknie przy narożniku — została jako kolumna diagnostyczna. Afiniczna
ekstrapolacja X(x, y) — niedokładna przy perspektywie (plansze „pod kątem”). Tylko środek–środek
— żaden z pięciu odcinków nie dałby wyniku.

Literatura: Hartley, Zisserman 2004 (Multiple View Geometry in Computer Vision, wyd. 2) — model
kamery otworkowej i przecięcie promienia z płaszczyzną.

---

## 2026-10-28 — Selekcja klatek: miary na obrazie wejścia sieci, progi względne ustalone przed przebiegami

**Co.** `vid2cloud/input/selection.py` liczy miary jakości klatki na tym obrazie, który widzi
sieć: klatka po rotacji z metadanych (`FileSource`), przeskalowana jak `resize_img()` upstreamu
(dłuższy bok do 512 px, filtr Lanczosa z PIL, crop centralny do wielokrotności 16 px), w skali
szarości. Geometrię daje istniejąca `scale/aruco.input_transform()`; dopisany jest tylko resampling
pikseli tym samym filtrem co upstream (`model_input_rgb()`). Zgodność z `resize_img()` upstreamu
sprawdza test w env `mast3r-slam` — bajt w bajt dla 5 formatów, w tym 3840×1634 z niezerowym cropem.
Dwie miary ostrości: **Tenengrad** (średnia Gx² + Gy², Sobel 3×3, float32) jako główna i **wariancja
Laplasjanu** (`cv2.Laplacian`, ksize=3 — w OpenCV to jądro [[2, 0, 2], [0, −8, 0], [2, 0, 2]],
klasyczne jądro 4-sąsiedztwa to ksize=1) do porównania; w raporcie przebiegu wstępnego jest ich
korelacja Spearmana. Progi odrzucające, ustalone dziś **przed** jakimkolwiek przebiegiem z selekcją:
klatka odpada, gdy Tenengrad < 0,5 × mediana Tenengradu klatek z |Δt| ≤ 2 s (okno w czasie z PTS),
albo gdy mediana jasności jest poza [40, 220]. Dobór: w każdym oknie k kolejnych klatek argmax
Tenengradu spośród klatek dopuszczonych; okno bez dopuszczonej klatki jest **pomijane** i trafia
do raportu z powodem. Okien jest ⌈n/k⌉, czyli tyle, ile klatek ma stały krok od przesunięcia 0.
`vid2cloud run --frames <lista>` dekoduje wszystkie klatki po kolei i pomija niewybrane **po**
dekodowaniu (bez seeka); bez `--frames` wyniki są identyczne jak przed zmianą (sprawdzone na 10 s
z 125357: 10/10 plików wynikowych, w tym `cloud.ply`, bajt w bajt, log.csv poza kolumnami
czasowymi i klucze config.json bez zmian).

**Dlaczego.** Miary na pełnym 4K oceniałyby szczegół, którego sieć nie dostaje — rozmycie 2 px
w 4K to ok. 0,27 px przy 512 — więc ranking klatek mógłby rozjechać się z tym, co ma znaczenie dla
silnika. Dwie miary z różnych rodzin operatorów (gradientowa i laplasjanowa), żeby wynik porównania
nie był artefaktem jednej z nich. Próg względny w oknie czasowym, a nie bezwzględny, bo poziom
Tenengradu zależy od treści kadru (tekstura, odległość), a nie tylko od rozmycia; okno z PTS, a nie
z liczby klatek, bo zapis telefonu bywa nierówny. Progi ustalone przed przebiegiem, bo dobrane po
wyniku byłyby dopasowaniem do danych (CLAUDE.md, zasada 4); test pilnuje wartości domyślnych.
Okno puste jest pomijane, bo wymuszony wybór wpuściłby klatkę, którą kryterium właśnie uznało za
złą — ceną jest to, że lista z progami może być krótsza od stałego kroku, a różnica jest jawna
w `manifest.csv`. Ta sama liczba klatek co stały krok, bo inaczej porównanie mierzyłoby gęstość
próbkowania, a nie jakość wyboru; szumem odniesienia jest rozrzut między przesunięciami stałego
kroku (offset 0, 1, …, k−1), bo powtórzenie przebiegu z single_thread daje bajt w bajt to samo.

**Alternatywy.** (1) Miary na pełnej rozdzielczości — odrzucone (wyżej). (2) `cv2.resize` z
`INTER_AREA` zamiast PIL — prostsze, ale to nie jest obraz, który widzi sieć; PIL nie jest na liście
zależności z CLAUDE.md, ale już jest w repo (`tools/make_boards.py`, `tests/test_aruco.py`), siedzi
w obu env jako zależność matplotlib i tylko nim da się odtworzyć wejście sieci bajt w bajt.
(3) Wymuszenie wyboru w pustym oknie (najlepsza z odrzuconych) — odrzucone (wyżej). (4) Próg
bezwzględny albo percentyl z całego nagrania — odrzucone: jeden fragment z bogatą teksturą
podnosiłby próg dla całej reszty.

Literatura: Pertuz, Puig, Garcia 2013 (Analysis of focus measure operators for shape-from-focus,
Pattern Recognition 46) — porównanie rodzin operatorów ostrości; Sieberth, Wackrow, Chandler 2016
(Automatic detection of blurred images in UAV image sets, ISPRS J. Photogramm. Remote Sens. 122) —
wykrywanie rozmytych zdjęć w fotogrametrii; Torresani, Remondino 2019 (Videogrammetry vs
photogrammetry for heritage 3D reconstruction, ISPRS Archives XLII-2/W15) — dobór klatek z wideo.

---

## 2026-10-30 — Ocena kształtu chmur względem TLS: CloudCompare wsadowo, parametry ustalone przed wynikami

**Co.** Każda chmura rekonstrukcji jest oceniana względem skanu TLS sali koncertowej jedną
procedurą (`eval/eval_cloud.py`, wszystkie chmury `eval/eval_all.sh`, pseudokod
`docs/pseudocode/eval.md`): macierz zgrubna z par punktów (Umeyama, `eval/coarse_from_pairs.py`)
→ wycięcie prostopadłościanem → SOR → ICP ze skalą → C2C → statystyki w numpy. Parametry,
zapisane w `eval/eval_config.json` 30.10 **przed** jakimkolwiek wynikiem i pilnowane testem:
TLS podpróbkowany przestrzennie do 0,01 m; SOR k = 6, σ = 1,0 (domyślne CloudCompare); ICP
`-ADJUST_SCALE`, `-MIN_ERROR_DIFF 1e-6`, `-RANDOM_SAMPLING_LIMIT 50000`, `-OVERLAP 90`; progi
0,05 / 0,10 / 0,20 m; prostopadłościan = percentyle 1–99 współrzędnych chmury po dopasowaniu
zgrubnym, osobno X, Y, Z, plus 0,20 m. C2C liczone **od rekonstrukcji (compared) do TLS
(reference)**, bez modelu lokalnego. **Jedna macierz zgrubna i jeden prostopadłościan na
nagranie** — liczone z `runs/2026-09-21/adapter/<nagranie>/cloud.ply` i używane dla wszystkich
wariantów (adapter, powtórzenia, selekcja); tym samym prostopadłościanem wycinany jest TLS.
Kontrola skali: s_mod = mediana factor_final przebiegu, porównana z s_icp = ∛det((M_icp·M_c)[:3,:3]),
oraz RMS C2C chmury przeskalowanej przez s_mod i dopasowanej sztywnie.

**Dlaczego.** Kierunek rekonstrukcja → TLS, bo TLS pokrywa całą salę, a rekonstrukcja jej
fragment: odwrotny kierunek mierzyłby głównie brak pokrycia. Bez modelu lokalnego, bo przy
TLS 1 cm odległość do najbliższego punktu różni się od odległości do powierzchni o ułamek kroku
(na syntetyce z siatką 5 mm: RMS C2C 10,2 mm przy szumie 10 mm), a wynik nie zależy od wyboru
modelu i jego promienia. Jeden prostopadłościan i jedna macierz na nagranie, bo inaczej różnice
między wariantami selekcji mieszałyby się z różnicą wycinka i z rozrzutem ręcznego klikania;
warianty zaczynają się od tej samej lub sąsiedniej klatki, więc mają prawie ten sam układ
(kotwica = pierwsza klatka kluczowa), a resztę zbiera ICP. ICP ze skalą, bo chmura monokularna
nie ma skali metrycznej, a pytanie oceny to kształt; skala z taśmy jest osobną gałęzią (g), żeby
dało się porównać kształt „najlepiej dopasowany” z kształtem w skali, którą daje moduł skali.
Parametry ustalone przed wynikami z tego samego powodu co progi selekcji (wpis z 2026-10-28):
dobrane po wyniku byłyby dopasowaniem do danych.

**Podział pracy CloudCompare / numpy.** CloudCompare liczy SOR, ICP i C2C (to są algorytmy,
na które powołuje się tekst pracy); macierz zgrubna, prostopadłościan i wycięcie są w numpy
(deterministyczne, z testem), a do CC trafiają gotowe pliki PLY. Komendy każdej chmury są
w `<out>/cc_commands.txt`. Alternatywa — `-APPLY_TRANS` i `-CROP` w CC — dałaby to samo,
ale `-CROP` działa na wszystkie wczytane chmury naraz, więc kolejność komend musiałaby to
obchodzić.

**CloudCompare w trybie wiersza poleceń.** Flatpak `org.cloudcompare.CloudCompare` **2.13.2**
(metainfo: wydanie 2024-07-11; commit flatpaka fd2774230ee4…, runtime org.kde.Platform 5.15-24.08).
Nagłówek PLY zapisanego przez CC podaje „CloudCompare v2.14.alpha (Nov 11 2011)” — to napis
z biblioteki I/O, wersję bierzemy z metadanych flatpaka. Na flathubie jest już nowsza kompilacja
(runtime KDE 6.11): `flatpak update` zmieniłby wersję między przebiegami — do końca ocen nie
aktualizować (albo `flatpak mask org.cloudcompare.CloudCompare`). Uruchomienie przez
`eval/cc.sh` wymaga dwóch zabiegów, które nie zmieniają obliczeń: (1) Qt `offscreen` bez X11 —
flatpak ma runtime GL `nvidia-595-84`, host sterownik 595.91.07, więc GLX w piaskownicy nie
działa i CC kończy się kodem 134 („Could not initialize GLX”); (2) bez zmiennych condy —
wtyczka PythonRuntime bierze `CONDA_PREFIX` za swoje środowisko i pokazuje QMessageBox, na
którym `-SILENT` wisi bez końca (ustalone z logu Qt: okno pojawia się zaraz po załadowaniu
`libPythonRuntime.so`). Wszystkie przełączniki z promptu są w tej wersji; żadnego nie zastąpiono.
RMS wypisywany przez `-ICP` nie jest RMS-em z pola C2C (syntetyka: 0,0192 wobec 0,0102 m) —
zapisujemy oba, osobno nazwane.

**TLS sali koncertowej** (`eval/tls_reference.py`). 9 archiwów, 25 plików .e57 (Koncertowa 6,
Tronowa 4, Senacka 15; `eval_cc/tls_inventory.csv`). Sala koncertowa: 6 stanowisk (jeden skan
na plik, Leica, projekt Cyclone REGISTER 360, stanowisko 01 w początku układu), 236 068 044
punktów (suma `recordCount` = liczba wczytana przez CC), po `-MERGE_CLOUDS -SS SPATIAL 0.01`
9 744 033 punkty z RGB i intensywnością. Prostopadłościan otaczający w osiach TLS:
28,871 × 40,215 × 11,896 m (X × Y × Z); percentyle 1–99: 21,318 × 20,516 × 10,052 m. Widok
z góry (`tls_koncertowa_1cm_top.png`): sala jest obrócona względem osi X TLS, a przez okna
i drzwi skaner sięga daleko poza nią — dlatego obok jest dodatek: osie obrócone o 31,5° wokół Z
(kąt minimalizujący pole prostokąta percentyli 1–99 w XY), percentyle 1–99:
19,438 × 14,279 × 10,051 m. Sala Tronowa (tylko wymiary; pliki wypakowane tymczasowo i usunięte):
4 stanowiska, 146 614 282 punkty, po 1 cm 7 184 163; prostopadłościan 46,655 × 67,034 × 12,207 m,
percentyle 1–99: 28,688 × 35,168 × 6,484 m — według widoku z góry obie liczby obejmują
pomieszczenia sąsiednie widziane przez drzwi, więc nie są wymiarami sali.

**Test E — głębokość odcinka.** `eval/scale_per_keyframe.py` zapisuje z_a_canon, z_b_canon,
z_mid_canon (stan przy powstaniu kf — te same punkty, z których liczony jest factor_net_created)
i z_mid_m = z_mid_canon · mediana factor_net_created **wszystkich** odcinków przebiegu, a nie
współczynnikiem tej samej pary — inaczej z_mid_m i net_cr byłyby zależne z definicji.
Pozostałe kolumny i pliki przeliczenia trzech przebiegów z 03.10 są bajt w bajt takie jak przed
zmianą. Spearman liczony w numpy (rangi średnie przy remisach; scipy nie ma w env).

**Alternatywy.** (1) M3C2 zamiast C2C — mniej wrażliwe na gęstość, ale wymaga normalnych
i skali projekcji, czyli kolejnych parametrów; C2C jest punktem odniesienia, M3C2 można dodać
obok. (2) Open3D zamiast CloudCompare — odrzucone bez decyzji autora (tekst pracy mówi
o CloudCompare; open3d nie ma też w żadnym env). (3) Prostopadłościan per wariant — odrzucone
(wyżej). (4) Dopasowanie bez macierzy zgrubnej (samo ICP z identyczności) — układy
rekonstrukcji i TLS różnią się dowolnym obrotem i skalą ok. 1,5, ICP by nie zbiegł.

Literatura: Lague, Brodu, Leroux 2013 (Accurate 3D comparison of complex topography with
terrestrial laser scanner: application to the Rangitikei canyon (N-Z), ISPRS J. Photogramm.
Remote Sens. 82) — M3C2, punkt odniesienia dla C2C; McNally i in. 2026 — ICP ze skalą dla
rekonstrukcji obrazowych, stała skala dla lidaru; Umeyama 1991 (Least-squares estimation of
transformation parameters between two point patterns, IEEE TPAMI 13(4)) — przekształcenie
przez podobieństwo z par punktów.

---

## 2026-11-04 — Skala osobno dla każdej klatki kluczowej: położenia jak ciąg poligonowy, warianty A i B, globalny do porównania

**Co.** Moduł skali poprawia skalę osobno dla każdej klatki kluczowej (`vid2cloud scale <przebieg>`,
`vid2cloud run --targets`; `vid2cloud/scale/pipeline.py`, pseudokod `docs/pseudocode/scale.md`).
Obserwacją jest f_kj = factor_final pary (klatka k, odcinek j) ze statusem ok, wektor f_k ma trzy
warianty: **A** — mediana f_kj w klatce z obserwacjami, log f interpolowany liniowo w czasie (pts_k)
między nimi, stały przed pierwszą i za ostatnią; **B** — wyrównanie pośredniczące (wpis niżej);
**globalny** — jedna liczba jak w wpisie z 2026-09-15 (`estimate_scale` z s_k·d_final po klatkach,
mediana segmentów). Chmura jest składana z `kf_*_final.npz` regułą `save_reconstruction` upstreamu
(`evaluate.py:47-70`: C/N > 1,5, k po kolei, piksele wiersz po wierszu); położenia klatek liczone są jak
ciąg poligonowy, p_0 = f_0·t_0, p_k = p_(k−1) + ½(f_(k−1) + f_k)(t_k − t_(k−1)), a punkt klatki k to
p_k + f_k·s_k·R_k·X. Odbudowa bez skalowania odtwarza `cloud.ply` trzech przebiegów z 03.10 co do punktu
(564 165 / 992 966 / 2 736 139, maks. różnica 1,05e-6 / 1,21e-6 / 7,25e-7 jednostki), więc kolory są
brane z `cloud.ply` po indeksie. Wariant globalny = `cloud.ply` × f (sprawdzone: ≤ 1,9e-6 m).

**Dlaczego.** Wybór silnika w pracy (3.4, 4.2, 4.6) stoi na tym, że każda klatka kluczowa ma własną
skalę Sim(3) — dryf skali jest zjawiskiem per klatka, a jeden współczynnik dla całej chmury zaciera
dokładnie to, co moduł ma poprawiać. Położenia z uśrednieniem f sąsiednich klatek (reguła trapezów
na wektorze t_k − t_(k−1)), bo samo f_k·t_k przesuwałoby klatkę o błąd skali całej drogi od kotwicy,
a nie tylko ostatniego odcinka. Przy wszystkich f_k równych wzory sprowadzają się do wariantu globalnego
(test `tests/test_scale_factors.py::test_methods_with_equal_factors_give_global_variant`).

**Alternatywa: jeden współczynnik dla chmury.** Zostaje jako wariant `cloud_m_global.ply` — punkt
odniesienia, względem którego `scale.json` podaje maks. przesunięcie położenia klatki dla A i B.

Literatura: Strasdat, Montiel, Davison 2010 (Scale drift-aware large scale monocular SLAM, RSS);
Murai, Dexheimer, Davison 2025 (MASt3R-SLAM, CVPR).

---

## 2026-11-04 — Metoda B: wyrównanie pośredniczące log f_k, σ_obs i τ zamrożone przed oceną kształtu

**Co.** Niewiadome u_k = ln f_k, k = 0 … K−1. Równania obserwacyjne u_k = ln f_kj z wagą 1/σ_obs²
i pseudoobserwacje u_(k+1) − u_k = 0 z wagą 1/(τ²·(pts_(k+1) − pts_k)) (błądzenie losowe log skali
w czasie). u = (AᵀPA)⁻¹AᵀPl, błąd średni u_k = √diag((AᵀPA)⁻¹) przy σ0 a priori = 1; `scale.json`
podaje też σ0 a posteriori = √(vᵀPv / r), r = liczba równań − K. f_k = exp(u_k), błąd względny ≈ σ_u.
σ_obs i τ policzone **raz** (`eval/scale_params.py`, 04.11, przed oceną kształtu) z trzech nagrań 21.09
i zapisane w `vid2cloud/scale/params.json` — to wartości domyślne każdego przebiegu, `--sigma-obs` /
`--tau` je nadpisują, a `scale.json` zapisuje, które użyto:

| wielkość | wzór | wartość | składniki |
|---|---|---|---|
| σ_obs | Σ (ln f_kj − ln med_k)² / (n − m), klatki z ≥ 2 odcinkami | **0,0473097** | n = 10 reszt, m = 4 klatki, 6 stopni swobody |
| τ | √(Σ (Δ ln med)² / Σ Δt), kolejne klatki z obserwacjami w nagraniu | **0,0205277 /√s** (τ² = 4,21385·10⁻⁴ /s) | 8 przyrostów, Σ Δt = 69,08 s |

med_k = mediana f_kj klatki k (ta sama co w metodzie A). Dzielnik n − m, bo każda klatka zużywa
stopień swobody na swoją medianę; z dzielnikiem n wyszłoby σ_obs = 0,036646. Obie wielkości mają
≥ 3 składniki. Przyrosty Δu liczone są z median klatek, więc obok τ²·Δt zawierają też szum obu median —
estymator tego nie odejmuje.

**Dlaczego.** Wyrównanie daje to, czego nie ma interpolacja: klatki bez obserwacji dostają wartość
z rosnącym błędem średnim, klatki z obserwacjami są ściągane ku sąsiadom w miarę, jak blisko są w czasie,
a σ0 a posteriori mówi, czy przyjęte σ_obs i τ pasują do danego przebiegu. Parametry zamrożone przed
oceną kształtu z tego samego powodu co progi selekcji i parametry CloudCompare (wpisy z 2026-10-28
i 2026-10-30): dobrane po wyniku byłyby dopasowaniem do danych.

**Alternatywy.** (1) Tylko interpolacja — zostaje jako metoda A. (2) σ_obs per odcinek z powtarzalności
kliknięć albo z sigma_mm — odrzucone na razie: obserwacją jest lift z pointmapy, nie kliknięcie, a rozrzut
między odcinkami jednej klatki obejmuje oba źródła błędu. (3) τ i σ_obs estymowane w każdym przebiegu
(np. VCE) — przy 1–4 obserwacjach na klatkę niestabilne.

Literatura: Ghilani C. D. (2017) Adjustment Computations: Spatial Data Analysis, wyd. 6, Wiley —
wyrównanie pośredniczące, wagi, macierz kowariancji.

---

## 2026-11-04 — PLY bez open3d (`export/plyio.py`), komenda `eval` usunięta z CLI

**Co.** Odczyt i zapis PLY w numpy przeniesiony z `eval/plyio.py` do `vid2cloud/export/plyio.py`
(w `eval/` został cienki import); chmury w metrach zapisuje ten moduł — float32 x y z + uchar RGB,
jak upstream. `scale_ply()` z open3d w `scale/estimate.py` zostaje nietknięty i nieużywany przez
aplikację. Komenda `vid2cloud eval` (zaślepka „not implemented”) usunięta — decyzja autora z 02.11;
ocena kształtu jest w `eval/eval_cloud.py` + `eval/eval_all.sh`.

**Dlaczego.** open3d nie ma w żadnym env, a instalacja w env silnika mogłaby ruszyć numpy; własny
zapis jest kilkadziesiąt linii, ma test (`tests/test_eval_cloud.py`) i zachowuje każdą właściwość
PLY, którą open3d by zgubił. `eval` w CLI obiecywał coś, czego CLI nie robi.

**Alternatywa: `pip install open3d` w vid2cloud-tools.** Odrzucone — nowa zależność bez potrzeby
(CLAUDE.md, zasada 8), a i tak potrzebna tylko do zapisu tablicy.

Literatura: brak — decyzja inżynierska.

---

## 2026-11-04 — Test E w obrębie klatki kluczowej (decyzja autora 2026-10-30, dopisana 2026-11-04)

**Co.** `eval/test_e.py` dostaje tabele w obrębie klatki kluczowej (`within_keyframe_pairs`): w każdej
klatce z ≥ 2 odcinkami (planszami) wszystkie pary i < j, znak (y_j − y_i)·(x_j − x_i) — zgodna / niezgodna /
remis jak w τ Kendalla — i odchylenie y od mediany klatki w %. Tabela 1b (21.09): x = z_mid_m, y = net_cr,
osobno informacyjnie y = final; tabela 2b (VID_ODLEGLOSC): x = z_m, y = factor_net_median, wiersze
z rozrzutem boków > 5 % z flagą, nieusunięte. Istniejące tabele i ρ łączny bez zmian (pliki bajt w bajt),
ρ łączny wypisywany z dopiskiem „między klatkami kluczowymi — obciążony skalą klatki”.

**Dlaczego.** Para odcinków z jednej klatki dzieli s_k i ten sam przebieg sieci, więc różnica ich
współczynników zależy od głębokości, a nie od skali klatki; między klatkami skala klatki dominuje.
Liczby podane przez autora: net_cr w daleko_szybko_pion rozrzut 44 % ((max − min)/min = 44,5 %;
w konwencji `estimate.py`, (max − min)/mediana — 38,3 %); w VID_ODLEGLOSC net_cr planszy ID 10 spada
z 1,08 do 0,63, przy final 1,08–1,15.

**Alternatywa: ρ łączny jako jedyna miara.** Zostaje, oznaczony jako obciążony skalą klatki.

Literatura: Kendall M. G. (1938), A new measure of rank correlation, Biometrika 30(1–2), 81–93.

---

## 2026-11-05 — Ocena chmur w metrach: gałąź „sztywna, s = 1” zamiast przeskalowania przez s_mod

**Co.** `eval/eval_cloud.py --metric` (chmury `cloud_m_*.ply` z `vid2cloud scale`; `eval/eval_all.sh`
uruchamia je z tą flagą) zastępuje gałąź g) gałęzią h): ta sama chmura po SOR co w gałęzi głównej,
start z macierzy zgrubnej **bez członu skali** (R, t z `similarity_parts(M_c)`, s = 1;
`rigid_start()`), ICP sztywne z parametrami `icp_rigid` z `eval_config.json`, C2C, te same
statystyki i progi. Zapis: `c2c_rigid_s1.ply` i pola `rigid_s1_{mean,rms,p95,frac_5cm,frac_10cm,frac_20cm}`
w `c2c_stats.json`, obok `metric_scale_error` = s_icp − 1 z gałęzi głównej (błąd skali chmury w metrach).
`--metric` zastępuje `--no-scale-check` z 04.11 i wyklucza `--scale-csv`. Dla chmur w jednostkach sieci
kroki a)–g) bez zmian. Sprawdzone przez odtworzenie nagranych wyjść CloudCompare: 23/23 plików
syntetycznego przypadku identycznych poza znacznikami czasu, te same argumenty wywołań CC. Wprost
bajt w bajt się nie da, bo ICP CloudCompare z `-RANDOM_SAMPLING_LIMIT` daje inny wynik przy każdym
przebiegu (`docs/TODO.md`). Syntetyka (`tests/test_eval_cloud.py::test_synthetic_metric_branch_rigid_s1`,
σ = 1 cm): przy braku błędu skali gałąź s = 1 daje RMS 0,01022 m, przy chmurze, której do TLS brakuje
skali 1,05, RMS 0,08362 m przy s_icp 1,04981 (gałąź główna w obu przypadkach 0,0102 m).

**Dlaczego.** Chmura w metrach ma być oceniana w skali, którą dał jej moduł skali; ICP ze skalą tę skalę
poprawia, więc w gałęzi głównej błąd skali modułu przechodzi do s_icp i znika z RMS. Gałąź s = 1 jest
rejestracją z 6 parametrami (obrót + przesunięcie, procedura z 2.7 pracy), więc błąd skali zostaje
w odległościach. Start z macierzy zgrubnej, a nie z wyniku ICP ze skalą, żeby gałąź nie zależała od
rozwiązania z dopasowaną skalą.

**Alternatywa: tylko ICP ze skalą.** Odrzucona — ukrywa błąd skali modułu; zostaje jako gałąź główna,
z której pochodzi `metric_scale_error`.

Literatura: brak — decyzja inżynierska (procedura z 2.7: rejestracja z 6 parametrami).

---

## 2026-11-05 — Dzielnik σ_obs: n − m potwierdzony przez autora

**Co.** σ_obs metody B (wpis z 2026-11-04, `vid2cloud/scale/params.json`) zostaje z dzielnikiem n − m
(10 reszt, 4 klatki, 6 stopni swobody): σ_obs = 0,0473097. Wartość z dzielnikiem n (0,036646) nie jest
używana. `params.json` bez zmian.

**Dlaczego.** Decyzja autora z 2026-11-05: to estymator nieobciążony, jak m0² = vᵀPv / (n − u)
w wyrównaniu — każda klatka zużywa jeden stopień swobody na swoją medianę (u = m niewiadomych).

**Alternatywa: dzielnik n.** Odrzucona — obciążony w dół, najmocniej przy parach (σ zaniżone o √2).

Literatura: Ghilani C. D. (2017) Adjustment Computations: Spatial Data Analysis, wyd. 6, Wiley.

---

## 2026-11-09 — TLS bez usuwania elementów ruchomych

**Co.** Referencją oceny kształtu jest cały TLS sali koncertowej po podpróbkowaniu do 1 cm, razem
z meblami, pulpitami i wyposażeniem sceny. Ręcznej segmentacji z kroku a) `docs/instrukcja_cloudcompare.md`
nie ma. `runs/2026-09-21/eval_cc/tls/tls_koncertowa_1cm_static.ply`, którego domyślnie używają
`eval/eval_cloud.py` i `eval/eval_all.sh`, to dowiązanie symboliczne do `tls_koncertowa_1cm.ply`
(nazwa w kodzie bez zmian, ten sam plik).

**Dlaczego.** C2C liczone jest od rekonstrukcji do TLS (wpis z 2026-10-30), więc usunięcie mebli
tylko z TLS zwiększałoby odległości punktów rekonstrukcji leżących na meblach, które nie zmieniły
położenia między 23.02 a 21.09 — ich najbliższym punktem stałby się dalszy fragment ściany czy podłogi.
Meble przestawione dają duże odległości niezależnie od tego, czy są w TLS, więc usunięcie ich
z referencji tego nie usuwa.

**Alternatywa: strefy wykluczenia w układzie TLS stosowane do obu chmur** (te same prostopadłościany
albo wieloboki wycinane z TLS i z rekonstrukcji po dopasowaniu). Odrzucona przed zamrożeniem liczb
(15.11) — do dalszych prac.

Literatura: brak — decyzja inżynierska.
