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
```

## Silnik

Upstream nie jest częścią tego repozytorium — dodaje się go jako submodule:

```bash
git submodule add https://github.com/rmurai0610/MASt3R-SLAM third_party/mast3r-slam
```

Uzasadnienie tego wyboru: [docs/decisions.md](docs/decisions.md).
