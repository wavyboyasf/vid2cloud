# TODO

Rzeczy zauważone poza zakresem bieżącej sesji. Nie robimy ich od ręki — trafiają tutaj
i czekają na swoją sesję (zasada 1 z CLAUDE.md).

## Środowisko

- [ ] **Nie ma env `mast3r-slam` na natywnym Linuksie.** Działający stack (conda, torch 2.11.0+cu128,
      zbudowane backendy sm_120, checkpointy, `datasets/own/`) siedzi w obrazie WSL2:
      `/media/wavy/DATA/WSL/Ubuntu-24.04/ext4.vhdx` (36 GB). Trzeba go zamontować i wyciągnąć
      dane albo odtworzyć środowisko od zera wg `JAK_URUCHOMIC_MAST3R.md`.
      **Blokuje test C (czwartek).**
- [ ] **Nagrania testowe** — `VID20263_cut.mp4` i reszta `datasets/own/` są w tym samym vhdx.
      Wyciągnąć przed testem C.

## Silnik

- [ ] **Poprawki sm_120 poza submodule.** Patrz `docs/decisions.md` (wpis z 2026-09-09):
      MASt3R-SLAM nie zbuduje się na RTX 5060 Ti bez czterech zmian w źródłach. Skoro nie
      modyfikujemy `third_party/`, potrzebny jest katalog `patches/` i skrypt nakładający je
      przy instalacji środowiska. Do zaprojektowania.
- [ ] Sprawdzić, na którym branchu upstreamu pracujemy — poprzednia konfiguracja używała
      brancha `windows` (wyłącza multiprocessing problematyczny w WSL). Na natywnym Linuksie
      może być niepotrzebny; zweryfikować przed przypięciem commitu submodule.
- [ ] **Dekodowanie MP4 w upstreamie bez torchcodec.** W env `mast3r-slam` nie ma `torchcodec`, więc
      `MP4Dataset` robi `cv2.VideoCapture` + `CAP_PROP_POS_FRAMES`, czyli seek na każdą klatkę
      (`mast3r_slam/dataloader.py:257`). Koszt zależy od GOP-u — stąd `-g 60` przy cięciu klipów
      testu C (docs/decisions.md, wpis z 2026-09-13); prawdziwym rozwiązaniem byłby torchcodec albo
      własne dekodowanie przez adapter. Nie sprawdzono też, czy przy klipie VFR skopiowanym
      `-c copy` OpenCV trafia zawsze w dobrą klatkę, czy potrafi ją zdublować albo pominąć.
      *03.10: `vid2cloud run` dekoduje PyAV sekwencyjnie (adapter). Na 125437 klatki 0 i 500 z
      `MP4Dataset` i z `FileSource` to ta sama klatka (sąsiednie dają kilka razy większą różnicę),
      liczby w `runs/2026-09-21/adapter/decoder_diff_125437.csv`. Klipy testu C nie były przeliczane.*
- [ ] **Znaczniki czasu trajektorii upstreamu przesunięte o 1 klatkę** (z lektury kodu, bez weryfikacji
      przebiegiem): `get_img_shape()` (`main.py:172`) wywołuje `read_img(0)`, które dopisuje znacznik
      (`dataloader.py:264`), więc dla MP4 `timestamps[i] = (i-1)/fps` dla i ≥ 1. Dotyczy `input.txt`
      i nazw PNG keyframe'ów. Ma znaczenie przy mapowaniu keyframe'ów na czas nagrania.
      *Adaptera nie dotyczy: `trajectory_sim3.txt` bierze PTS klatki źródła klatki kluczowej.*
      *03.10, 125437: upstream (02.10) i adapter dały po 7 klatek kluczowych; dla każdej z 6
      klatek k ≥ 1 `t_upstream · CAP_PROP_FPS` (30,000247) = `frame_index` adaptera − 1, co do
      klatki (279/280, 479/480, 672/673, 846/847, 1167/1168, 1302/1303) — zgodne z tym wpisem.*
- [ ] **Półpiksel cropu w `Intrinsics`/`resize_img(return_transformation=True)` upstreamu.** Upstream
      liczy połowę cropu jako `(W - W_out) / 2`, a PIL tnie od `cx - halfw`; przy nieparzystym boku
      po skalowaniu wychodzi 0,5 px różnicy (sprawdzone 03.10: 3840x1640 i 3840x1643 -> bok 219,
      `top` 5,5 u upstreamu, 5,0 w `scale/aruco.py`). Dla 3840x2160 / 2160x3840 / 4000x3000 /
      1920x1080 / 3840x1634 zgodne. `run.py` przelicza narożniki liczbami z silnika, czyli przy
      takim formacie byłby o 0,5 px nie tam; test pilnuje tylko formatów z 21.09.
- [ ] **Klatka kluczowa z relokalizacji ma N = 0** (z lektury kodu): backend dokłada
      `states.get_frame()` (`main.py:42`), a `SharedStates.get_frame()` nie przenosi N
      (`frame.py:169-183`), więc `get_average_conf() = C / 0` do pierwszej aktualizacji przez
      tracker. Zrzuty `kf_*_created.npz` zapisują to, co jest w silniku. Lift (sesja B) musi
      traktować N = 0 jawnie, a `save_reconstruction` przepuściłby wtedy całą klatkę (inf > 1,5).
- [ ] **Ścieżka relokalizacji w adapterze nie jest sprawdzona przebiegiem.** W pięciu przebiegach
      z 03.10 (`runs/2026-09-21/adapter/`) tracker ani razu nie zgubił śledzenia (`tracking_lost` = 0,
      `reloc_kf` = 0), więc zrzut klatki kluczowej dokładanej przez backend (zdarzenie „reloc”) jest
      napisany i nietknięty. Sprawdzić na pierwszym nagraniu z relokalizacją.
- [ ] **Bez single_thread historia póz jest próbkowana co klatkę** (`poses_history.csv`, zdarzenie
      `backend_done`): kilka optymalizacji między dwiema klatkami zlewa się w jeden wiersz, a moment
      zapisu zależy od tempa GPU. Do rozdziału o s_k(t) brać przebiegi z `--single-thread`.

## Wejście

- [ ] **Rotacja z display matrix siedzi w `scale/aruco.py`**, a `input/sources.py` ją stamtąd
      importuje (wejście zależy od modułu skali i od OpenCV). Przenieść `frame_rotation()` /
      `apply_rotation()` do `input/`, gdy powstanie `CameraSource`.

## Skala

- [ ] **Konwencja półpiksela przy przeliczaniu narożników.** `scale/aruco.rescale_corners()`
      liczy `x' = (x + 0.5) * W/W1 - 0.5 - left` (środki pikseli), upstream w `Intrinsics`
      (`dataloader.py:292`) pomija poprawkę — różnica ok. 0,43 px przy 4K -> 512. Przy lifcie
      sprawdzić, którą konwencję zakłada próbkowanie pointmapy, i czy różnica w ogóle wychodzi
      poza szum. Patrz `docs/decisions.md`, wpis z 2026-09-13.
- [ ] **`img_downsample` > 1 nie jest obsłużony.** `frame.py:117-120` przerzedza siatkę pointmapy
      o `config["dataset"]["img_downsample"]` (domyślnie 1 w `config/base.yaml:5`). `input_transform()`
      zakłada 1 i tego nie sprawdza — jeśli kiedyś zmienimy config, narożniki wylądują nie tam.
- [ ] **Próg wykrywalności markera zmierzony tylko na syntetyce** (12 px boku, warunki w
      `docs/decisions.md`). Brakuje pomiaru na nagraniu: kąt, rozmycie ruchu, kompresja, oświetlenie.
      Do zrobienia razem z pierwszym nagraniem z planszami.
      *Pierwsze nagranie (17.09, wnętrze, 2–3 m): `docs/decisions.md`, wpis z 2026-09-17 — najmniejszy
      prawdziwy marker miał 56 px w 4K, więc progu detekcji nie osiąga; rozmycie ruchu obserwowane
      w klatce 200.*
- [ ] **Fałszywe wykrycia ArUco na nagraniu** `data/own/proba_generalna.mp4` (17.09, `runs/proba/podglad.csv`):
      ID 17 w klatkach 164, 657, 1060, 1071 i ID 37 w klatkach 88, 1091; bok 29–37 px w 4K
      (3,9–5,0 px przy 512). Obejrzane klatki 88, 164 i 1091: płaskie trapezy na nakrętce butelki,
      szklance i etykiecie — tych ID nie ma w żadnym wydruku. Parametrów detektora nie zmieniano.
      Do rozstrzygnięcia przy estymacji/lifcie: przyjmować tylko ID z `targets.json`, czy też
      minimalny bok.
- [ ] **Biały margines na planszy A3 jest węższy niż bok markera** — przy 120 mm na A3 wychodzi
      88,5 mm zamiast 120 mm i `make_boards.py` o tym ostrzega. Albo A2 na te markery, albo
      świadomie zostawić i pilnować jasnego tła wokół planszy w scenie.

- [ ] **`scale_ply()` nie został uruchomiony — brak open3d.** Nie ma go ani w `vid2cloud-tools`,
      ani w `mast3r-slam`, więc ścieżka `--apply/--out` w `python -m vid2cloud.scale.estimate`
      jest napisana i nietknięta. Przy instalacji (`pip install open3d` w `vid2cloud-tools`,
      nigdy w env silnika) sprawdzić dwie rzeczy: czy pip nie rusza numpy 1.26.4 i co zostaje
      z chmury MASt3R-SLAM po przejściu przez open3d — kolory i normalne przechodzą, ale
      niestandardowe właściwości PLY (np. ufność per punkt) giną.
- [ ] **Wpis łaty skali w `targets.json` czeka na pomiar po sklejeniu** (środa 16.09).
      `data/own/targets_test.json` ma na razie wyłącznie odcinki kontrolne W1/W2 z 11.09 —
      nominału z `boards.json` tam nie wpisywać, wartość ma być zmierzona taśmą.
- [ ] **Dwie drogi do tej samej liczby: `--fill` w `eval/test_c_start_point.py` i moduł skali.**
      Pomiary W1/W2 w siedmiu chmurach testu C można wprowadzać albo do `results.csv` przez
      `--fill`, albo do `measured.csv` przez `scale/estimate.py`. Wybrać jedną i trzymać się jej,
      inaczej będą dwie wersje tych samych liczb.

- [ ] **Fałszywe wykrycia ArUco na nagraniach z 21.09** (`runs/2026-09-21/*/podglad.csv`,
      zestawienie w `runs/2026-09-21/inventory/ids_summary.csv`). Przy zadeklarowanych ID
      3, 8, 9, 10, 11, 12 detektor zwrócił dodatkowo 7, 16, 17, 22, 24, 28, 34, 37, 38, 44 —
      od 1 do 72 klatek, bok przy 512 px zwykle 3–10 px, ale ID 7 miał 40 px (szyba okienna,
      `ids/ID_07.jpg`), ID 16 — 28,8 px, ID 37 — 16,7 px. Deklaracja rozstawionych ID
      (`notes/rozstawione_id.txt`, flaga `--ids`) odsiewa je w zestawieniu i w tabeli pomiarów,
      ale **nie w samej detekcji** — `scale/aruco.py` nadal je wypisuje do CSV i rysuje
      w podglądzie. Do rozstrzygnięcia przy lifcie, czy filtr ma zejść do detekcji i czy obok
      listy ID potrzebny jest drugi warunek (minimalny bok, ciągłość w czasie): sam bok nie
      wystarczy, bo 40 px fałszywki to więcej niż 4,32 px najmniejszego prawdziwego ID 9.
- [ ] **Łata skali (ID 48/49) nie była na sesji 21.09.** Nie ma jej na żadnym z 71 zdjęć
      z `photos_targets/` ani w nagraniach. Wzorcem skali dla tej sesji mogą więc być tylko
      odległości między planszami mierzone taśmą (`notes/pomiary_tasma.csv`), a te są mierzone
      między markerami stojącymi na statywach — punkt odniesienia trzeba wpisać w kolumnie
      `punkty_pomiaru`, inaczej nie wiadomo, w co klikać w CloudCompare.

- [ ] **Sesja 21.09 ma dwa rozstawienia plansz.** Wg autora odległości z `notes/targets.json`
      dotyczą tylko nagrań 125357, 125437, 125536; nagrania 133034, 133043, 133224 są z innym
      rozstawieniem. `eval/session_inventory.py` nie zna tego podziału — współwidoczność w
      `pomiary_tasma.csv` i liczby w `ids_summary.csv` sumuje po wszystkich nagraniach, a
      `targets.json` nie mówi, do których nagrań się odnosi. Potrzebne: druga tabela pomiarów
      dla drugiego rozstawienia i przypisanie targets.json do listy nagrań (np. kolumna
      `rozstawienie` w `nagrania.csv`). Nie wiadomo, przy którym rozstawieniu są `photos_targets/`.

## Ewaluacja

- [ ] **Zasięg ArUco w terenie nie jest zmierzony.** Z geometrii (4K, HFOV ~68°): marker 30 cm
      z 20 m ma ok. 43 px na pełnej klatce, ale tylko ok. 5,7 px na wejściu modelu 512 px.
      Detekcja idzie po pełnej klatce, więc wykrycie nie jest wąskim gardłem — jest nim to,
      czy pointmapa w takim miejscu daje sensowny punkt 3D. Nie sprawdzone; do pomiaru na
      pierwszym nagraniu zewnętrznym.
      *Wnętrze, 2–3 m (17.09): bok przy 512 px w `docs/decisions.md`, wpis z 2026-09-17.*
- [ ] **Brak `log.csv` per klatka w przebiegach testu C.** Test C uruchamia upstreamowy `main.py`
      bez hooków, więc `runs/test_c/clip_*/` mają `config.json`, `log.txt` i `cloud.ply`, ale nie
      logu per klatka z zasady 3 CLAUDE.md. Wymaga przebiegu przez adapter `engine/mast3r_slam.py`.
      *Adapter jest od 03.10 (`vid2cloud run`); klipy testu C nie zostały przez niego przeliczone.
      Kolumn ostrości i selekcji w `log.csv` jeszcze nie ma — przyjdą z `input/selection.py`.*

- [ ] **Skany TLS sali koncertowej leżą nierozpakowane** w `data/2026-09-21/reference_lidar/`
      (9 archiwów, 8,4 GB, w środku `.e57` z 23.02.2026, np. `ZK-Koncertowa-23022026-05.e57`).
      To są stanowiska lidarowe pokoju jako referencja — czyli trzecie źródło prawdy terenowej
      obok nalotu dronowego z szachownicami i rekonstrukcji COLMAP zespołu. Do ustalenia, **zanim**
      powstanie test skali: które z tych źródeł jest referencją dla czego i skąd każde z nich bierze
      skalę metryczną. Rozpakowanie i przegląd — osobna sesja.
- [ ] **Rekonstrukcje COLMAP zespołu** (`data/2026-09-21/reference_colmap/`, 190 plików): nie wiadomo,
      czy mają skalę metryczną i z czego. Jeśli z tych samych plansz ArUco co nasza chmura, to
      porównanie z nimi nie mierzy skali, tylko zgodność dwóch rekonstrukcji.
