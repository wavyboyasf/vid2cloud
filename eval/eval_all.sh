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
#   runs/2026-09-21/adapter/*/cloud_m_<wariant>.ply   chmury w metrach z `vid2cloud scale` (global, kf_interp,
#                                                 kf_mnk); nazwa = <nagranie>_m_<wariant>; macierz zgrubna =
#                                                 macierz nagrania · diag(1/f_global) (eval/coarse_metric.py,
#                                                 te same pary punktów co dla cloud.ply); prostopadłościan
#                                                 z cloud_m_global.ply (= f_global · cloud.ply, więc po złożeniu
#                                                 ten sam co nagrania); --metric: zamiast g) gałąź h)
#                                                 (ICP sztywne przy s = 1, metric_scale_error = s_icp − 1)
#                                                 katalog wariantu (np. <nagranie>_kf050, inne f_global):
#                                                 nazwa = <katalog>_m_<wariant>, prostopadłościan nagrania
#                                                 z eval_cc/crop/<nagranie>.json (--crop-name)
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

evaluate_metric() {  # katalog_przebiegu nagranie wariant
    local d=$1 rec=$2 v=$3 run
    run=$(basename "$d")
    # Katalog nagrania (E1): nazwa <nagranie>_m_<wariant>, prostopadłościan z cloud_m_global.ply (= f · cloud.ply).
    # Katalog wariantu (np. _kf050): własna nazwa i macierz (inne f_global), prostopadłościan nagrania.
    local cloud=$d/cloud_m_$v.ply name=${run}_m_$v out=$ECC/c2c/adapter_m/$run/$v
    local base=$ECC/coarse/$rec.txt coarse=$ECC/coarse/${run}_m_$v.txt
    local crop=(--crop-source "$d/cloud_m_global.ply")
    [ "$run" = "$rec" ] || crop=(--crop-name "$rec")
    local missing=""
    [ -f "$base" ] || missing="$missing macierz:eval_cc/coarse/$rec.txt"
    [ -f "$TLS" ] || missing="$missing TLS:${TLS#$REPO/}"
    [ -f "$d/scale.json" ] || missing="$missing scale.json"
    if [ "$run" = "$rec" ]; then
        [ -f "$d/cloud_m_global.ply" ] || missing="$missing prostopadłościan:${d#$REPO/}/cloud_m_global.ply"
    else
        [ -f "$ECC/crop/$rec.json" ] || missing="$missing prostopadłościan:eval_cc/crop/$rec.json"
    fi
    local state="gotowa"
    [ -f "$out/c2c_stats.json" ] && state="policzona"
    [ -n "$missing" ] && state="brak:$missing"
    n=$((n + 1))
    printf '%3d  %-28s %-62s %s\n' "$n" "$name" "${cloud#$REPO/}" "$state"
    [ "$DRY" -eq 1 ] && return
    if [ -n "$missing" ]; then return; fi
    if [ "$state" = "policzona" ] && [ "$FORCE" -eq 0 ]; then return; fi
    "$PY" eval/coarse_metric.py "$base" "$d/scale.json" "$coarse" || return
    mkdir -p "$out"
    "$PY" eval/eval_cloud.py "$cloud" --name "$name" --out "$out" --coarse "$coarse" \
        "${crop[@]}" --metric > "$out/eval_cloud.log" 2>&1
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
for d in "$S"/adapter/*/; do
    d=${d%/}
    compgen -G "$d/cloud_m_*.ply" > /dev/null || continue
    name=$(recording_name "$d")
    for v in global kf_interp kf_mnk; do
        evaluate_metric "$d" "$name" "$v"
    done
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
