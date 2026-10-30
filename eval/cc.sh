#!/usr/bin/env bash
# CloudCompare w trybie wiersza poleceń (-SILENT), bez okna. Wszystkie argumenty idą do CC.
#
#   bash eval/cc.sh -AUTO_SAVE OFF -O chmura.ply -SS SPATIAL 0.01 -SAVE_CLOUDS FILE wynik.ply
#
# Instalacja: flatpak org.cloudcompare.CloudCompare 2.13.2 (flathub). Dwie rzeczy są konieczne,
# żeby CLI w ogóle ruszył (sprawdzone 2026-10-30, docs/decisions.md):
#  1. Qt "offscreen" bez gniazda X11 — flatpak ma runtime GL nvidia-595-84, a host sterownik
#     595.91.07, więc GLX w piaskownicy nie działa i CC przerywa się (kod 134) jeszcze przed
#     przetwarzaniem komend. Przy ustawionym DISPLAY Qt offscreen też próbuje GLX, stąd DISPLAY=.
#  2. Bez zmiennych condy/venv — wtyczka PythonRuntime traktuje CONDA_PREFIX jako swoje
#     środowisko Pythona i pokazuje QMessageBox; w -SILENT nikt go nie zamknie i CC wisi.
# Żaden z tych zabiegów nie zmienia parametrów obliczeń, tylko sposób uruchomienia.
set -euo pipefail
unset_vars=()
for v in $(env | sed -n 's/^\(_\?CONDA[A-Z_]*\|VIRTUAL_ENV\|PYTHONHOME\|PYTHONPATH\)=.*/\1/p'); do
    unset_vars+=(-u "$v")
done
exec env "${unset_vars[@]}" flatpak run \
    --nosocket=x11 --nosocket=fallback-x11 --env=DISPLAY= --env=QT_QPA_PLATFORM=offscreen \
    org.cloudcompare.CloudCompare -SILENT "$@"
