#!/usr/bin/env bash
# Ocena kształtu wszystkich chmur sesji 21.09 względem TLS: eval/eval_cloud.py po kolei.
#
#   bash eval/eval_all.sh --dry-run     # tylko lista chmur i tego, czego brakuje
#   bash eval/eval_all.sh               # liczy; chmura z istniejącym c2c_stats.json jest pomijana
#   bash eval/eval_all.sh --force       # liczy wszystko od nowa
#
# Chmury:
#   runs/2026-09-21/adapter/*/cloud.ply          nagranie z config.json (VID_20260921_<nr> -> nazwa)
#   runs/2026-09-21/selection/<nazwa>/*/cloud.ply
#   runs/2026-09-21/realityscan/*.ply            nazwa = nazwa pliku; prostopadłościan z tej samej chmury
#                                                 (osobna macierz zgrubna: eval_cc/coarse/<nazwa>.txt)
# Wyniki: runs/2026-09-21/eval_cc/c2c/<źródło>/<…>/ + jeden wiersz na chmurę w eval_cc/c2c_summary.csv.
set -u
REPO=/home/wavy/dev/vid2cloud
PY=/home/wavy/miniforge3/envs/vid2cloud-tools/bin/python
S=$REPO/runs/2026-09-21
ECC=$S/eval_cc
TLS=$ECC/tls/tls_koncertowa_1cm_static.ply
DRY=0
FORCE=0
for a in "$@"; do
    case $a in
        --dry-run) DRY=1 ;;
        --force) FORCE=1 ;;
        *) echo "nieznany argument: $a"; exit 2 ;;
    esac
done
cd "$REPO" || exit 1

recording_name() {  # katalog przebiegu adaptera -> nazwa nagrania
    local nr
    nr=$(sed -n 's/.*"video": *"[^"]*VID_20260921_\([0-9]*\)\.mp4".*/\1/p' "$1/config.json" | head -1)
    case $nr in
        125357) echo daleko_wolno ;;
        125437) echo blisko_z_gory ;;
        125536) echo daleko_szybko_pion ;;
        *) echo "?$nr" ;;
    esac
}

n=0; n_ready=0; n_done=0
evaluate() {  # chmura nazwa out [crop_source]
    local cloud=$1 name=$2 out=$3 src=${4:-$S/adapter/$2/cloud.ply}
    local missing=""
    [ -f "$ECC/coarse/$name.txt" ] || missing="$missing macierz:eval_cc/coarse/$name.txt"
    [ -f "$TLS" ] || missing="$missing TLS:${TLS#$REPO/}"
    [ -f "$src" ] || missing="$missing prostopadłościan:${src#$REPO/}"
    local state="gotowa"
    [ -f "$out/c2c_stats.json" ] && state="policzona"
    [ -n "$missing" ] && state="brak:$missing"
    n=$((n + 1))
    printf '%3d  %-20s %-62s %s\n' "$n" "$name" "${cloud#$REPO/}" "$state"
    [ "$DRY" -eq 1 ] && return
    if [ -n "$missing" ]; then return; fi
    if [ "$state" = "policzona" ] && [ "$FORCE" -eq 0 ]; then return; fi
    mkdir -p "$out"
    "$PY" eval/eval_cloud.py "$cloud" --name "$name" --out "$out" --crop-source "$src" > "$out/eval_cloud.log" 2>&1
    local code=$?
    echo "     kod $code  $(tail -1 "$out/eval_cloud.log")"
}

echo "# $(date -Iseconds)  dry-run=$DRY  TLS=${TLS#$REPO/}"
for d in "$S"/adapter/*/; do
    d=${d%/}
    [ -f "$d/cloud.ply" ] || continue
    name=$(recording_name "$d")
    evaluate "$d/cloud.ply" "$name" "$ECC/c2c/adapter/$(basename "$d")"
done
for d in "$S"/selection/*/*/; do
    d=${d%/}
    [ -f "$d/cloud.ply" ] || continue
    name=$(basename "$(dirname "$d")")
    evaluate "$d/cloud.ply" "$name" "$ECC/c2c/selection/$name/$(basename "$d")"
done
if compgen -G "$S/realityscan/*.ply" > /dev/null; then
    for f in "$S"/realityscan/*.ply; do
        stem=$(basename "$f" .ply)
        evaluate "$f" "$stem" "$ECC/c2c/realityscan/$stem" "$f"
    done
else
    echo "     (brak runs/2026-09-21/realityscan/*.ply)"
fi
echo "# chmur: $n"
