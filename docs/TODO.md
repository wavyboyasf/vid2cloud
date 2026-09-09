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
