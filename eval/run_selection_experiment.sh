#!/usr/bin/env bash
# Komponent 1: stały krok vs selekcja przy tej samej liczbie klatek.
# 15 przebiegów `vid2cloud run --frames` (3 nagrania × 5 list klatek), single_thread, jeden po drugim.
#
# Start (w sesji 2026-10-28 NIE uruchamiany):
#   mkdir -p runs/2026-09-21/selection && nohup bash eval/run_selection_experiment.sh > runs/2026-09-21/selection/nohup.out 2>&1 &
# Postęp:  tail -f runs/2026-09-21/selection/postep.log
#
# Wejście: listy z `vid2cloud select --k 3` w runs/2026-09-21/quality/<nazwa>/frames/<wariant>.txt.
# Wyjście: runs/2026-09-21/selection/<nazwa>/<wariant>/ — wyniki `vid2cloud run` + stdout.log,
#          a po każdym przebiegu eval/scale_per_keyframe.py na jego katalogu (env vid2cloud-tools):
#          scale_per_keyframe.csv, s_history.csv, rysunki, markers_world.ply, scale.log
#          i porownanie_scale_summary.csv W KATALOGU PRZEBIEGU (domyślne --comparison-out
#          nadpisałoby plik sesji A w runs/2026-09-21/adapter/).
# Przebieg z istniejącym config.json jest pomijany (run_video i tak nie nadpisuje wyników).
#
# Czas — SZACUNEK, nie pomiar: ok. 30 min na 15 przebiegów silnika + eval/scale_per_keyframe.py
# (jego czasu nie mierzono). Model: czas poza silnikiem z sesji A bez zmian (dekodowane są wszystkie
# klatki), czas silnika × (klatki z listy / klatki nagrania): daleko_wolno 5 × ok. 99 s,
# blisko_z_gory 5 × ok. 131 s, daleko_szybko_pion 5 × ok. 132 s. Sprawdzone na jednym klipie 10 s
# (fixed_k3_o0, 100/300 klatek): model 38,0 s, zmierzone 37,3 s. Sama proporcja całego czasu
# dałaby ok. 22 min, ale na klipie zaniżyła (24,2 s wobec 37,3 s).
#
# Uwaga (28.10): w daleko_wolno i blisko_z_gory progi nie odrzuciły żadnej klatki, więc
# sharp_thr_k3.txt == sharp_k3.txt i te dwa przebiegi powtórzą wynik sharp_k3 bajt w bajt.
# Szum odniesienia to rozrzut między fixed_k3_o0/o1/o2 — powtarzanie przebiegu nic nie daje
# (single_thread jest deterministyczne, sesja A).
set -u
REPO=/home/wavy/dev/vid2cloud
RUN_BIN=/home/wavy/miniforge3/envs/mast3r-slam/bin/vid2cloud
TOOLS_PY=/home/wavy/miniforge3/envs/vid2cloud-tools/bin/python
V=$REPO/data/2026-09-21/video
Q=$REPO/runs/2026-09-21/quality
OUT=$REPO/runs/2026-09-21/selection
LOG=$OUT/postep.log
RECORDINGS="daleko_wolno:125357 blisko_z_gory:125437 daleko_szybko_pion:125536"
VARIANTS="fixed_k3_o0 fixed_k3_o1 fixed_k3_o2 sharp_k3 sharp_thr_k3"
cd "$REPO" || exit 1
mkdir -p "$OUT"
log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

# Wszystkie 15 list musi istnieć — lepiej paść od razu niż po dwóch godzinach.
missing=0
for pair in $RECORDINGS; do
    name=${pair%%:*}
    for var in $VARIANTS; do
        [ -f "$Q/$name/frames/$var.txt" ] || { log "BRAK listy $Q/$name/frames/$var.txt"; missing=1; }
    done
done
[ "$missing" -eq 0 ] || { log "przerwane: brak list klatek"; exit 1; }

run() {  # nazwa nagranie wariant
    local name=$1 rec=$2 var=$3
    local dst=$OUT/$name/$var list=$Q/$name/frames/$var.txt
    if [ -e "$dst/config.json" ]; then
        log "[$name/$var] jest config.json — pomijam"
        return
    fi
    mkdir -p "$dst"
    log "[$name/$var] start  VID_20260921_$rec  lista $(wc -l < "$list") klatek  sha256 $(sha256sum "$list" | cut -c1-12)"
    PYTHONUNBUFFERED=1 "$RUN_BIN" run "$V/VID_20260921_$rec.mp4" --out "$dst" --single-thread --frames "$list" \
        < /dev/null > "$dst/stdout.log" 2>&1
    local code=$?
    log "[$name/$var] koniec kod $code  $(grep gotowe "$dst/stdout.log" | tail -1)"
    [ "$code" -eq 0 ] || return
    "$TOOLS_PY" eval/scale_per_keyframe.py "$dst" --comparison-out "$dst/porownanie_scale_summary.csv" \
        < /dev/null > "$dst/scale.log" 2>&1
    log "[$name/$var] scale_per_keyframe kod $?  $(head -1 "$dst/scale.log")"
}

log "START  commit $(git -C "$REPO" rev-parse --short HEAD)  $(git -C "$REPO" status --porcelain --untracked-files=no | wc -l) zmienionych plików"
i=0
for pair in $RECORDINGS; do
    name=${pair%%:*}
    rec=${pair##*:}
    for var in $VARIANTS; do
        i=$((i + 1))
        log "--- przebieg $i/15"
        run "$name" "$rec" "$var"
    done
done
log KONIEC
