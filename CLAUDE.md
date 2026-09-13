# vid2cloud — kontekst dla agenta kodującego

## Czym jest ten projekt (5 zdań)
Praca inżynierska (geoinformatyka, PW): aplikacja, która z wideo (plik lub strumień na żywo) generuje **gęstą chmurę punktów klatka po klatce**, używając sieci z priorami geometrycznymi. Silnikiem rekonstrukcji jest **MASt3R-SLAM** (upstream, nietykany). Wkład własny to warstwa wejścia (sekwencyjne dekodowanie, bufor, selekcja klatek), **moduł skali metrycznej** (ArUco → definicja datum) oraz skrypty ewaluacyjne. Wyniki są walidowane względem Agisoft Metashape, lidaru i pomiarów tachimetrycznych w CloudCompare. Termin: grudzień 2026 — kod ma **działać i być obronny**, nie być ładny.

## Decyzje, których nie negocjujemy
- Silnik: MASt3R-SLAM jako submodule w `third_party/mast3r-slam/`. **Nie modyfikujemy upstreamu.** Wszystko, co dotyka upstreamu, żyje w `vid2cloud/engine/mast3r_slam.py` (adapter).
- Wideo: **PyAV** sekwencyjnie (bez seek-per-frame). OpenCV tylko do obrazu/ArUco, nie do dekodowania.
- Interfejs: CLI (`typer`) + podgląd 3D w `viser`. Bez GUI desktopowego.
- Skala: **zawsze z fizycznego wzorca** (odległość między markerami na łacie, `targets.json`). Nigdy ze „stałej poprawki" ani z prioru modelu.
- Detekcja ArUco na pełnej rozdzielczości klatki; współrzędne przeskalowane do rozdzielczości wejścia modelu (512 px) przed liftem do 3D.

## Środowisko
- Linux, RTX 5060 Ti 16 GB, conda env `mast3r-slam` (Python 3.11, torch+CUDA z upstreamu). Nie instaluj do tego env niczego, co może zmienić wersję torch/numpy. Narzędzia bez GPU (ArUco, generator plansz, ewaluacja) mogą iść w osobnym env `vid2cloud-tools`.
- Dane testowe: `data/` (gitignore). Krótkie klipy do testów: `data/clips/*.mp4`, ≤ 30 s.
- Wyniki przebiegów: `runs/<nazwa>/` (gitignore), zawsze z `config.json` i `log.csv`.

## Struktura
```
vid2cloud/
  input/      sources.py (FileSource, CameraSource), buffer.py (RingBuffer, producent), selection.py
  engine/     base.py (Engine.push_frame -> FrameResult), mast3r_slam.py (adapter)
  scale/      aruco.py (detekcja), lift.py (2D->3D z pointmapy), estimate.py (skala z odległości)
  export/     ply/las, trajektoria, log per klatka
  viz/        viser
  cli.py
eval/         test_c_start_point.py, test_e_distance.py, cloudcompare_batch.sh
tools/        make_boards.py (PDF plansz w mm)
tests/        pytest — funkcje czyste (miary ostrości, estymacja skali na syntetyce, bufor)
docs/decisions.md   każda decyzja architektoniczna: co / dlaczego / alternatywa (3 zdania)
```

## Zasady pracy
1. **Jedna sesja = jeden moduł = jedna definicja „działa"** podana na początku sesji. Nie rozszerzaj zakresu; jeśli widzisz coś do zrobienia obok — wpisz do `docs/TODO.md`, nie rób.
2. **Funkcje czyste najpierw z testem** (`tests/`), integracja z GPU testowana ręcznie na klipie 10 s. Nie mockuj MASt3R-SLAM w testach — po prostu nie testuj go jednostkowo.
3. **Logowanie per klatka** do CSV: indeks klatki, PTS, miara ostrości, czy wybrana, czas dekodowania, czas inferencji, mediana i p95 głębi, udział pikseli niskiej ufności. Bez tego nie ma rozdziału wyników.
4. **Nie wymyślaj liczb.** Jeśli test nie został uruchomiony, napisz „nie uruchomiono". Jeśli wynik jest dziwny, zgłoś, nie „napraw" go zmianą progu.
5. **Nie interpretuj wyników pomiarów** — zapisz je; wnioski wyciąga autor.
6. Po każdej sesji: wpis w `docs/decisions.md` (jeśli była decyzja) + krótki `git commit` z opisem *co działa* (nie *co zmieniono*).
7. Kod komentowany po polsku lub angielsku — jednolicie w pliku. Nazwy po angielsku.
8. Nie dodawaj zależności bez powodu. Dozwolone: numpy, opencv-contrib-python, av, typer, viser, open3d, matplotlib, pytest, reportlab.

## Czego agent NIE robi
- Nie edytuje niczego w `third_party/`.
- Nie zmienia rozdzielczości wejścia modelu ani jego hiperparametrów „żeby lepiej wyszło".
- Nie usuwa wyników przebiegów z `runs/`.
- Nie pisze tekstu pracy dyplomowej.
- Nie commituje samemu na githuba - podsuwa polecenia użytkownikowi żeby sam mógł zdecydować o pushach
