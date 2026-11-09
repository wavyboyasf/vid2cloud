# vid2cloud

Aplikacja, która z wideo — pliku albo strumienia na żywo — generuje gęstą chmurę punktów
klatka po klatce, używając sieci z priorami geometrycznymi. Silnikiem rekonstrukcji jest
MASt3R-SLAM podpięty jako submodule i nietykany; wkładem własnym są warstwa wejścia
(sekwencyjne dekodowanie, bufor, selekcja klatek), moduł skali metrycznej opartej na
fizycznych wzorcach ArUco oraz skrypty ewaluacyjne. Wyniki walidowane są względem
Reality Scan, lidaru i pomiarów tachimetrycznych w CloudCompare.

## Instalacja

Pakiet instalujemy w trybie edytowalnym, w środowisku, w którym stoi już MASt3R-SLAM:

```bash
conda activate mast3r-slam
pip install -e .
vid2cloud --help
```

Zależności rdzenia to tylko `typer` i `numpy`, bez wersji minimalnych — instalacja nie
może podmienić wersji `torch` ani `numpy` pochodzących z upstreamu. Po instalacji warto
to sprawdzić:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Narzędzia bez GPU (generator plansz, detekcja ArUco, ewaluacja) idą do **osobnego**
środowiska, żeby nie mieszać ich zależności ze środowiskiem silnika:

```bash
python -m venv .venv-tools && source .venv-tools/bin/activate
pip install -e ".[tools]"
```

Pozostałe extras: `viz` (podgląd 3D w viser), `video` (PyAV do dekodowania).

## Wzorce skali (ArUco)

Skala metryczna bierze się **wyłącznie** z fizycznego wzorca, nigdy z prioru modelu.
Plansze do druku generuje `tools/make_boards.py` (env `vid2cloud-tools`):

```bash
# plansze kontrolne: 8 stron A3, marker 120 mm, po jednym na stronę
python tools/make_boards.py --size-mm 120 --ids 0-7 --page A3

# łata skali: 2 strony A4 z krzyżem osiowym, do naklejenia na końce listwy
python tools/make_boards.py --scalebar
```

Każdy przebieg zapisuje PDF i `boards.json` (`{id, dict, nominal_size_mm}`) we własnym
podkatalogu `boards/`. Marker jest wstawiany do PDF jako bitmapa ≥ 600 dpi o boku będącym
wielokrotnością 6 pikseli (4 komórki danych + 2 komórki bordera DICT_4X4_50) — wektor
przeskalowany przez drukarkę rozmywałby krawędzie komórek.

> **Po wydruku zmierz bok markera suwmiarką i wpisz rzeczywistą wartość do `targets.json`.**
> Drukarki i kserokopiarki skalują wydruk o 1–3 % („dopasuj do strony", marginesy sprzętowe),
> więc `nominal_size_mm` z `boards.json` jest tylko punktem odniesienia, a nie wymiarem
> wzorca. Cała skala chmury wisi na tej jednej liczbie.

Detekcja na zdjęciu wydruku albo na klatce wideo — do sprawdzenia, czy wydruk w ogóle
się czyta:

```bash
python -m vid2cloud.scale.aruco zdjecie.jpg --out debug.jpg     # narożniki, ID, JSON
python -m vid2cloud.scale.aruco --video klip.mp4 --frame 120 --out debug.jpg
python -m vid2cloud.scale.aruco --video klip.mp4 --render podglad.mp4     # film z obrysami + CSV per klatka
```

## Skala metryczna osobno dla każdej klatki kluczowej

Wzorcem są odcinki między markerami zmierzone taśmą (`targets.json`, milimetry). Moduł skali
liczy współczynnik każdej pary (klatka kluczowa, odcinek), a z nich wektor f_k w trzech
wariantach: **A** — interpolacja log f w czasie, **B** — wyrównanie MNK (σ_obs i τ zamrożone
w `vid2cloud/scale/params.json`), **globalny** — jedna liczba do porównania. Na gotowym
katalogu przebiegu, bez GPU (env `vid2cloud-tools`):

```bash
python -m vid2cloud.cli scale runs/2026-09-21/adapter/daleko_wolno --targets data/2026-09-21/notes/targets.json
```

albo od razu po silniku: `vid2cloud run nagranie.mp4 --out runs/<nazwa> --targets targets.json`
(błąd skali nie psuje wyników silnika; config.json dostaje sekcję `"scale"`). W katalogu przebiegu
powstają `scale_per_keyframe.csv` (pary), `scale_keyframes.csv` (f_k: obserwacje, A, B ± σ_u, globalny),
`cloud_m_global.ply`, `cloud_m_kf_interp.ply`, `cloud_m_kf_mnk.ply` (metry, kolory z `cloud.ply`)
i `scale.json` (parametry, sha256 targets.json, przesunięcia położeń klatek względem wariantu
globalnego, ostrzeżenia). Bez `--overwrite` istniejące wyniki nie są nadpisywane. Algorytm:
[docs/pseudocode/scale.md](docs/pseudocode/scale.md); decyzje: [docs/decisions.md](docs/decisions.md),
wpisy z 2026-11-04.

## Silnik

Upstream nie jest częścią tego repozytorium — dodaje się go jako submodule:

```bash
git submodule add https://github.com/rmurai0610/MASt3R-SLAM third_party/mast3r-slam
```

Uzasadnienie tego wyboru: [docs/decisions.md](docs/decisions.md).

Przebieg nagrania przez silnik (env `mast3r-slam`, GPU) — adapter odtwarza pętlę `main.py`
upstreamu z własnym dekoderem PyAV i zapisuje stan klatek kluczowych:

```bash
vid2cloud run nagranie.mp4 --out runs/<nazwa>              # single_thread: zrzuty deterministyczne
vid2cloud run nagranie.mp4 --out runs/<nazwa> --no-single-thread   # zachowanie upstreamu
```

W `runs/<nazwa>/`: `cloud.ply`, `trajectory_sim3.txt` (pozy klatek kluczowych **ze skalą** s),
`poses_history.csv` (pozy po każdym zdarzeniu), `keyframes/kf_NNN_{created,final}.npz`
(pointmapa, ufność, poza), `aruco_keyframes.csv`, `log.csv` (wiersz na klatkę), `config.json`.
Algorytm: [docs/pseudocode/engine.md](docs/pseudocode/engine.md), [run.md](docs/pseudocode/run.md).

## Selekcja klatek

Przebieg wstępny bez GPU (env `vid2cloud-tools`) — miary ostrości i ekspozycji każdej klatki
na obrazie wejścia sieci, potem listy klatek do `run --frames`:

```bash
python -m vid2cloud.cli quality nagranie.mp4 --out runs/<nazwa>/quality      # quality.csv, fig_ostrosc.png
python -m vid2cloud.cli select runs/<nazwa>/quality/quality.csv --k 3        # frames/*.txt, manifest.csv
vid2cloud run nagranie.mp4 --out runs/<nazwa>/fixed_k3_o0 --frames runs/<nazwa>/quality/frames/fixed_k3_o0.txt
```

Z `--frames` dekodowane są wszystkie klatki, a do silnika trafiają tylko te z listy (`log.csv`
dostaje kolumnę `fed`). Progi są ustalone z góry: [docs/decisions.md](docs/decisions.md), wpis
z 2026-10-28; algorytm: [docs/pseudocode/selection.md](docs/pseudocode/selection.md).
