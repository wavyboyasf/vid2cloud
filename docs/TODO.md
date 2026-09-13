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
      (`mast3r_slam/dataloader.py:257`). Koszt zależy od GOP-u (x264 domyślnie keyint 250, źródło
      z telefonu ~61). Nie sprawdzono, czy przy klipie VFR skopiowanym `-c copy` (clip_0 testu C)
      OpenCV trafia zawsze w dobrą klatkę, czy potrafi ją zdublować albo pominąć.
- [ ] **Znaczniki czasu trajektorii upstreamu przesunięte o 1 klatkę** (z lektury kodu, bez weryfikacji
      przebiegiem): `get_img_shape()` (`main.py:172`) wywołuje `read_img(0)`, które dopisuje znacznik
      (`dataloader.py:264`), więc dla MP4 `timestamps[i] = (i-1)/fps` dla i ≥ 1. Dotyczy `input.txt`
      i nazw PNG keyframe'ów. Ma znaczenie przy mapowaniu keyframe'ów na czas nagrania.

## Ewaluacja

- [ ] **Brak `log.csv` per klatka w przebiegach testu C.** Test C uruchamia upstreamowy `main.py`
      bez hooków, więc `runs/test_c/clip_*/` mają `config.json`, `log.txt` i `cloud.ply`, ale nie
      logu per klatka z zasady 3 CLAUDE.md. Wymaga przebiegu przez adapter `engine/mast3r_slam.py`.
