# vid2cloud

Aplikacja, która z wideo — pliku albo strumienia na żywo — generuje gęstą chmurę punktów
klatka po klatce, używając sieci z priorami geometrycznymi. Silnikiem rekonstrukcji jest
MASt3R-SLAM podpięty jako submodule i nietykany; wkładem własnym są warstwa wejścia
(sekwencyjne dekodowanie, bufor, selekcja klatek), moduł skali metrycznej opartej na
fizycznych wzorcach ArUco oraz skrypty ewaluacyjne. Wyniki walidowane są względem
Agisoft Metashape, lidaru i pomiarów tachimetrycznych w CloudCompare.

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

## Silnik

Upstream nie jest częścią tego repozytorium — dodaje się go jako submodule:

```bash
git submodule add https://github.com/rmurai0610/MASt3R-SLAM third_party/mast3r-slam
```

Uzasadnienie tego wyboru: [docs/decisions.md](docs/decisions.md).
