"""Test C — zależność współczynnika skali od punktu startu sekwencji.

Hipoteza: MASt3R-SLAM dziedziczy skalę mapy z pierwszej klatki kluczowej, więc współczynnik
skali zmierzony w stałym miejscu sceny może zależeć od tego, gdzie zaczyna się klip.
Projekt: wszystkie klipy kończą się w tym samym miejscu (--end), różnią się tylko startem;
każdy zawiera oba wzorce (W1 ok. 19 s, W2 ok. 24 s nagrania źródłowego), więc zmienną jest
odległość czasowa kotwica (pierwsza klatka) -> wzorzec.

Tryby
-----
    python eval/test_c_start_point.py --video data/own/VID20263.mp4 --starts 0,3,6 --end 30
        cięcie klipów -> runs/test_c/clip_<X>/{input.mp4,config.json}, potem silnik na każdym
        klipie sekwencyjnie, potem runs/test_c/results.csv
    ... --dry-run
        tylko cięcie i results.csv (bez silnika)
    python eval/test_c_start_point.py --fill measured.csv --reference reference.csv
        wpisuje ręczne pomiary z CloudCompare do results.csv, liczy współczynniki
        i results_summary.txt; niczego nie tnie i nie uruchamia silnika

Skrypt używa tylko biblioteki standardowej (+ ffmpeg/ffprobe z systemu), więc działa
w dowolnym env; silnik idzie przez --engine-python (pełna ścieżka, nie PATH).

Cięcie (ffmpeg)
---------------
Domyślnie (--reencode-all) każdy klip jest przekodowany tymi samymi parametrami — także ten,
którego start wypada na keyframie — żeby kodek był jednolity między porównywanymi klipami:
`-c:v libx264 -crf 18 -an -fps_mode passthrough -g 60 -frames:v N`. passthrough = każda klatka
źródła dokładnie raz, bez duplikowania i gubienia; N = liczba klatek źródła w [X, END), bez N
filtr trim liczy (END - X) od pierwszej zachowanej klatki, nie od X, i przepuszcza jedną klatkę
po END. -g 60 (X264_KEYINT) zamiast domyślnych 250, bo silnik przewija do każdej klatki osobno
i koszt dekodowania rośnie z długością GOP-u (docs/decisions.md, wpis z 2026-09-13).
Weryfikacja: n_frames (zdekodowane) vs n_frames_expected (źródło) w config.json.

Z --no-reencode-all skrypt najpierw próbuje `ffmpeg -ss X -to END -i src -c copy -an`. Kopia jest
przyjmowana, gdy pierwsza zdekodowana klatka jest keyframe'em, jej pts mieści się w 1 klatce od 0
(0 = X po cięciu) i pierwszy pakiet nie ma flagi discard. Ten ostatni warunek jest potrzebny, bo
przy -c copy ffmpeg zaczyna od keyframe'u *przed* X i chowa pre-roll listą edycji MP4 (ujemne pts,
flaga D): ffprobe pokazuje wtedy pierwszą klatkę z pts 0, ale nagłówek (nb_frames, z którego
OpenCV bierze długość klipu) liczy też pre-roll. Sprawdzone na VID20263.mp4: keyframe'y co
~1,0165 s, więc z typowych startów tylko 0 s trafia w keyframe (i kończy się dokładnie,
1801 = 1801 klatek).
Rotacja: re-encode stosuje display matrix (domyślne -autorotate) i wychodzi 2160x3840 bez
metadanych rotacji; -c copy zostawia 3840x2160 + rotation=-90. config.json zapisuje wymiary
kodowane, rotację z metadanych, wymiary po rotacji oraz to, co faktycznie zwraca pierwsza
klatka cv2.VideoCapture z env silnika (engine_decoder) — to jest obraz, który dostaje silnik.

Silnik
------
    cd <engine-dir> && <engine-python> main.py --dataset <abs input.mp4> \
        --config <config> --save-as test_c_<X> --no-viz
Gdzie upstream zapisuje wyniki (third_party/mast3r-slam @ e6f4e3d):
  - mast3r_slam/evaluate.py:14-20  prepare_savedir(): save_dir = Path("logs") / args.save_as,
                                   seq_name = dataset.dataset_path.stem  (tu: "input")
  - main.py:216                    recon_file = save_dir / f"{seq_name}.ply"  (kasowany na starcie)
  - main.py:315-320                eval.save_reconstruction(save_dir, f"{seq_name}.ply", keyframes,
                                   last_msg.C_conf_threshold)
    => chmura:       <engine-dir>/logs/test_c_<X>/input.ply
  - main.py:314 + evaluate.py:23-44  save_traj(): jeden wiersz na klatkę kluczową
    => trajektoria:  <engine-dir>/logs/test_c_<X>/input.txt; liczba jej wierszy to n_keyframes
       (silnik nie wypisuje liczby keyframe'ów na stdout)
  - przy --no-viz próg ufności zapisanej chmury = domyślne WindowMsg.C_conf_threshold = 1.5
    (mast3r_slam/visualization.py:38)
  - bez torchcodec MP4Dataset czyta klatki przez cv2.VideoCapture z seekiem na każdą klatkę,
    długość klipu = CAP_PROP_FRAME_COUNT (mast3r_slam/dataloader.py:236-261)
  - load_dataset() wybiera loader po składowych ścieżki ("tum", "euroc", ...;
    dataloader.py:320-338) — skrypt odmawia, gdy --out zawiera katalog o takiej nazwie
Chmura i trajektoria są hardlinkowane (fallback: kopia) do clip_<X>/cloud.ply i
clip_<X>/trajectory.txt, więc ponowne uruchomienie silnika z tym samym --save-as nie zmienia
plików w runs/. Nie nadpisujemy wyników: klip, który ma cloud.ply, jest pomijany; katalog
klipu z innymi parametrami to błąd; log.txt jest dopisywany.

--fill
------
measured.csv:  clip, W1_measured_m, W2_measured_m  (clip jak w results.csv, np. clip_18;
               puste pole = brak pomiaru)
reference.csv: segment, ref_m  (wiersze W1 i W2)
Separator ',', ';' albo tab; przy ';' dozwolony przecinek dziesiętny.
W<k>_scale = ref_W<k> / W<k>_measured_m. results_summary.txt: dla W1 i W2 osobno n, min, max,
mean, std (próbkowe, n-1) oraz nachylenie regresji liniowej MNK współczynnika względem
x = t_W<k> - start_s, gdzie t_W1 = 19 s, t_W2 = 24 s (SEGMENT_TIME_S).
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

# Pierwsze 7 kolumn wynika z clip_*/config.json. Pozostałe (pomiary, ewentualne dopiski
# autora) przy regeneracji results.csv są przenoszone ze starej wersji pliku.
RESULT_COLUMNS = [
    "clip", "start_s", "end_s", "length_s", "reencoded", "n_frames", "cloud_path",
    "W1_measured_m", "W2_measured_m", "W1_scale", "W2_scale",
]
SEGMENTS = ("W1", "W2")
# Czas w nagraniu źródłowym [s], w którym wzorzec jest w kadrze; oś X regresji: t - start_s.
SEGMENT_TIME_S = {"W1": 19.0, "W2": 24.0}
EXPECTED_DISPLAY_WH = (2160, 3840)
# -g dla libx264. Domyślne 250 dawało 480 s przebiegu silnika na 12-sekundowym klipie:
# upstreamowy MP4Dataset bez torchcodec przewija do każdej klatki osobno, więc koszt rośnie
# z długością GOP-u. 60 jest zbliżone do źródła (keyframe co ~61 klatek).
X264_KEYINT = 60
ENGINE_PYTHON_DEFAULT = "~/miniforge3/envs/mast3r-slam/bin/python"
# Składowe ścieżki, po których upstreamowy load_dataset() wybrałby inny loader niż MP4.
ENGINE_PATH_KEYWORDS = {"tum", "euroc", "eth3d", "7-scenes", "realsense", "webcam"}
FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
FFPROBE = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json"]

# Uruchamiane interpreterem silnika: co zwraca cv2.VideoCapture, którego używa MP4Dataset.
CV2_PROBE = """
import importlib.util, json, sys, cv2
cap = cv2.VideoCapture(sys.argv[1])
ok, img = cap.read()
print(json.dumps({
    "cv2": cv2.__version__,
    "torchcodec_installed": importlib.util.find_spec("torchcodec") is not None,
    "frame_hw": list(img.shape[:2]) if ok else None,
    "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
    "fps": cap.get(cv2.CAP_PROP_FPS),
}))
"""


# ---------------------------------------------------------------- narzędzia


def fmt_num(x: float) -> str:
    return f"{x:g}"


def clip_name(start: float) -> str:
    return f"clip_{fmt_num(start)}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str], capture: bool = True, **kw) -> str:
    proc = subprocess.run(cmd, capture_output=capture, text=True, **kw)
    if proc.returncode != 0:
        err = proc.stderr.strip() if capture else "(stderr powyżej)"
        raise RuntimeError(f"kod {proc.returncode}: {' '.join(cmd)}\n{err}")
    return proc.stdout if capture else ""


def ffprobe(path: Path, *args: str) -> dict:
    return json.loads(run(FFPROBE + list(args) + [str(path)]))


@functools.cache
def ffmpeg_version() -> str:
    return run(["ffmpeg", "-version"]).splitlines()[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        raise FileExistsError(f"{dst} już istnieje — nie nadpisuję")
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


# ---------------------------------------------------------------- sondy wideo


def probe_source(src: Path) -> dict:
    """Czasy prezentacji klatek źródła (z pakietów, bez dekodowania) i fps."""
    stream = ffprobe(src, "-show_entries", "stream=avg_frame_rate,duration")["streams"][0]
    packets = ffprobe(src, "-show_entries", "packet=pts_time,flags")["packets"]
    return {
        "fps": float(Fraction(stream["avg_frame_rate"])),
        "duration_s": float(stream["duration"]),
        "frame_times": sorted(float(p["pts_time"]) for p in packets
                              if p.get("pts_time", "N/A") != "N/A" and "D" not in p["flags"]),
    }


def first_frame(path: Path) -> dict | None:
    # 200 pakietów pokrywa z zapasem pre-roll całego GOP-a źródła (~61 klatek).
    frames = ffprobe(path, "-read_intervals", "%+#200",
                     "-show_entries", "frame=key_frame,pts_time,pict_type").get("frames", [])
    return frames[0] if frames else None


def first_packet(path: Path) -> dict | None:
    packets = ffprobe(path, "-read_intervals", "%+#1",
                      "-show_entries", "packet=pts_time,flags").get("packets", [])
    return packets[0] if packets else None


def probe_clip(path: Path) -> dict:
    s = ffprobe(path, "-count_frames", "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration"
                ":stream_side_data=rotation")["streams"][0]
    rotation = 0
    for side_data in s.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(side_data["rotation"])
    w, h = int(s["width"]), int(s["height"])
    display = (h, w) if abs(rotation) % 180 == 90 else (w, h)
    return {
        "codec": s["codec_name"],
        "coded_wh": [w, h],
        "display_rotation_deg": rotation,
        "display_wh": list(display),
        # piksele obrócone przez ffmpeg: kodowane 2160x3840 i brak rotacji w metadanych
        "rotation_applied": rotation == 0 and (w, h) == EXPECTED_DISPLAY_WH,
        "portrait_2160x3840": display == EXPECTED_DISPLAY_WH,
        "avg_frame_rate": s["avg_frame_rate"],
        "duration_s": float(s["duration"]),
        "nb_frames_header": int(s["nb_frames"]) if "nb_frames" in s else None,
        "nb_frames_decoded": int(s["nb_read_frames"]),
        "first_frame": first_frame(path),
    }


def probe_engine_decoder(engine_python: Path, path: Path) -> dict:
    try:
        return json.loads(run([str(engine_python), "-c", CV2_PROBE, str(path)], timeout=300))
    except Exception as e:  # sonda pomocnicza — brak env silnika nie blokuje cięcia
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- cięcie


def prepare_clip(src: Path, src_info: dict, start: float, end: float, clip_dir: Path,
                 engine_python: Path, reencode_all: bool) -> dict:
    name = clip_dir.name
    cfg_path, mp4 = clip_dir / "config.json", clip_dir / "input.mp4"
    if cfg_path.exists() and mp4.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if (cfg.get("source"), cfg.get("start_s"), cfg.get("end_s")) == (str(src), start, end):
            print(f"[{name}] input.mp4 już przygotowany — używam istniejącego")
            return cfg
        raise SystemExit(f"{clip_dir} zawiera klip o innych parametrach (source/start/end) — "
                         "nie nadpisuję; zmień --out albo usuń katalog ręcznie")
    clip_dir.mkdir(parents=True, exist_ok=True)

    frame_s = 1.0 / src_info["fps"]
    in_range = [t for t in src_info["frame_times"] if start <= t < end]
    if not in_range:
        raise SystemExit(f"[{name}] brak klatek źródła w [{start}, {end})")
    if in_range[0] - start > frame_s:
        print(f"[{name}] UWAGA: pierwsza klatka źródła >= start jest w {in_range[0]:.6f} s")

    tmp = clip_dir / "input.tmp.mp4"
    cut_in = ["-ss", str(start), "-to", str(end), "-i", str(src)]
    copy_cmd, ff, fp = None, None, None
    if reencode_all:
        # Jednolity kodek między klipami: nie próbujemy -c copy nawet przy starcie na keyframie.
        reasons = ["--reencode-all"]
    else:
        copy_cmd = FFMPEG + cut_in + ["-c", "copy", "-an", str(tmp)]
        run(copy_cmd)
        ff, fp = first_frame(tmp), first_packet(tmp)
        reasons = []
        if ff is None or int(ff.get("key_frame", 0)) != 1:
            reasons.append("pierwsza zdekodowana klatka nie jest keyframe'em")
        if ff is not None and abs(float(ff["pts_time"])) > frame_s:
            reasons.append("pts pierwszej klatki odbiega od startu o > 1 klatkę")
        if fp is None or "D" in fp.get("flags", ""):
            reasons.append("pre-roll od keyframe'u przed startem (pakiety z flagą discard)")

    reencode_cmd = None
    if reasons:
        if copy_cmd is not None:
            print(f"[{name}] -c copy odrzucone: {'; '.join(reasons)}")
        print(f"[{name}] re-encode libx264 CRF 18, GOP {X264_KEYINT} "
              f"({end - start:g} s 4K, to chwilę potrwa)")
        # -frames:v: filtr trim liczy (END - X) od pierwszej zachowanej klatki, nie od X,
        # więc samo -to wpuszcza klatkę źródła tuż po END (dla X=18: 30,0108 s).
        reencode_cmd = FFMPEG + ["-stats"] + cut_in + [
            "-c:v", "libx264", "-crf", "18", "-an", "-fps_mode", "passthrough",
            "-g", str(X264_KEYINT), "-frames:v", str(len(in_range)), str(tmp)]
        run(reencode_cmd, capture=False)
    else:
        print(f"[{name}] -c copy przyjęte (start na keyframe'ie)")

    video = probe_clip(tmp)
    os.replace(tmp, mp4)
    decoder = probe_engine_decoder(engine_python, mp4)

    n_expected = len(in_range)
    if not (video["nb_frames_decoded"] == video["nb_frames_header"] == n_expected):
        print(f"[{name}] UWAGA: liczba klatek — zdekodowane {video['nb_frames_decoded']}, "
              f"nagłówek {video['nb_frames_header']}, źródło w [start, end) {n_expected}")
    if not video["portrait_2160x3840"]:
        print(f"[{name}] UWAGA: wymiary po rotacji {video['display_wh']}, oczekiwano 2160x3840")
    if decoder.get("frame_hw") not in (None, [EXPECTED_DISPLAY_WH[1], EXPECTED_DISPLAY_WH[0]]):
        print(f"[{name}] UWAGA: cv2 w env silnika zwraca klatkę HxW = {decoder['frame_hw']}")
    if "error" in decoder:
        print(f"[{name}] UWAGA: sonda cv2 w env silnika nie zadziałała: {decoder['error']}")

    cfg = {
        "test": "C",
        "clip": name,
        "source": str(src),
        "source_size_bytes": src.stat().st_size,
        "source_fps": src_info["fps"],
        "start_s": start,
        "end_s": end,
        "length_s": end - start,
        "reencoded": bool(reasons),
        "n_frames": video["nb_frames_decoded"],
        "n_frames_expected": n_expected,
        "first_frame_src_s": in_range[0],
        "cut": {
            "ffmpeg_version": ffmpeg_version(),
            "copy_attempted": copy_cmd is not None,
            "copy_cmd": copy_cmd,
            "copy_first_frame": ff,
            "copy_first_packet": fp,
            "copy_rejected_because": reasons,
            "reencode_cmd": reencode_cmd,
        },
        "video": video,
        "engine_decoder": decoder,
        "prepared_at": now_iso(),
    }
    write_json(cfg_path, cfg)
    return cfg


# ---------------------------------------------------------------- silnik


def engine_setup(args: argparse.Namespace, engine_python: Path) -> dict:
    engine_dir = Path(args.engine_dir).resolve()
    config_path = engine_dir / args.config
    missing = [str(p) for p in (engine_dir / "main.py", engine_python, config_path)
               if not p.exists()]
    if missing:
        raise SystemExit("brak: " + ", ".join(missing))
    try:
        commit = run(["git", "-C", str(engine_dir), "rev-parse", "HEAD"]).strip()
        diff = run(["git", "-C", str(engine_dir), "diff", "HEAD"])
        upstream = {"commit": commit, "dirty": bool(diff),
                    "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}
    except (RuntimeError, OSError) as e:
        upstream = {"error": str(e)}
    return {"dir": engine_dir, "python": engine_python, "config_arg": args.config,
            "config_path": config_path, "config_sha256": sha256_file(config_path),
            "upstream": upstream}


def is_fresh(path: Path, t0: float) -> bool:
    # Plik starszy niż start przebiegu to pozostałość po poprzednim uruchomieniu.
    return path.exists() and path.stat().st_mtime >= t0 - 1.0


def run_engine(cfg: dict, clip_dir: Path, engine: dict) -> bool:
    name = cfg["clip"]
    if (clip_dir / "cloud.ply").exists():
        print(f"[{name}] cloud.ply już istnieje — pomijam silnik (nie nadpisuję wyników)")
        return True
    dataset = (clip_dir / "input.mp4").resolve()
    save_as = f"test_c_{fmt_num(cfg['start_s'])}"
    engine_out = engine["dir"] / "logs" / save_as           # evaluate.py:14-20
    cloud_src = engine_out / f"{dataset.stem}.ply"          # main.py:216, 315-320
    traj_src = engine_out / f"{dataset.stem}.txt"           # main.py:314, evaluate.py:23-44
    cmd = [str(engine["python"]), "main.py", "--dataset", str(dataset),
           "--config", engine["config_arg"], "--save-as", save_as, "--no-viz"]
    log_path = clip_dir / "log.txt"
    print(f"[{name}] silnik: {' '.join(cmd)}")
    print(f"[{name}] log: {log_path}")

    started_at = now_iso()
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"### {started_at}  cwd={engine['dir']}\n### {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=engine["dir"], stdin=subprocess.DEVNULL, stdout=log,
                              stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
    wall_s = time.time() - t0

    ok = proc.returncode == 0 and is_fresh(cloud_src, t0)
    result = {
        "cmd": cmd,
        "cwd": str(engine["dir"]),
        "config_path": str(engine["config_path"]),
        "config_sha256": engine["config_sha256"],
        "upstream": engine["upstream"],
        "started_at": started_at,
        "wall_s": round(wall_s, 1),
        "returncode": proc.returncode,
        "cloud_src": str(cloud_src),
        "cloud_link": None,
        "trajectory_src": str(traj_src),
        "n_keyframes": None,
    }
    if ok:
        result["cloud_link"] = link_or_copy(cloud_src, clip_dir / "cloud.ply")
        if is_fresh(traj_src, t0):
            link_or_copy(traj_src, clip_dir / "trajectory.txt")
            with open(traj_src, encoding="utf-8") as f:
                result["n_keyframes"] = sum(1 for line in f if line.strip())
        print(f"[{name}] OK: {wall_s:.0f} s, keyframe'y: {result['n_keyframes']}")
    else:
        print(f"[{name}] BŁĄD: kod {proc.returncode}, chmura z tego przebiegu "
              f"{'jest' if is_fresh(cloud_src, t0) else 'nie powstała'} — patrz {log_path}")
    cfg["engine"] = result
    write_json(clip_dir / "config.json", cfg)
    return ok


# ---------------------------------------------------------------- results.csv


def write_results(out_dir: Path) -> Path:
    """results.csv ze wszystkich clip_*/config.json; pomiary i dopiski przenoszone ze starego pliku."""
    path = out_dir / "results.csv"
    old_header, old_rows = read_csv(path) if path.exists() else ([], [])
    old = {r["clip"]: r for r in old_rows}
    header = RESULT_COLUMNS + [c for c in old_header if c not in RESULT_COLUMNS]
    rows = []
    for cfg_path in out_dir.glob("clip_*/config.json"):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cloud = cfg_path.parent / "cloud.ply"
        row = {c: "" for c in header}
        row.update(old.get(cfg["clip"], {}))
        row.update({
            "clip": cfg["clip"],
            "start_s": fmt_num(cfg["start_s"]),
            "end_s": fmt_num(cfg["end_s"]),
            "length_s": fmt_num(cfg["length_s"]),
            "reencoded": str(cfg["reencoded"]),
            "n_frames": str(cfg["n_frames"]),
            "cloud_path": str(cloud.resolve()) if cloud.exists() else "",
        })
        rows.append(row)
    rows.sort(key=lambda r: float(r["start_s"]))
    write_csv(path, header, rows)
    return path


# ---------------------------------------------------------------- --fill


def read_table(path: Path, required: list[str]) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"{path}: pusty plik")
    delimiter = max([",", ";", "\t"], key=lines[0].count)
    reader = csv.DictReader(lines, delimiter=delimiter)
    fields = [f.strip() for f in reader.fieldnames or []]
    if missing := [c for c in required if c not in fields]:
        raise SystemExit(f"{path}: brak kolumn {missing} (są: {fields})")
    rows = []
    for i, r in enumerate(reader, start=1):
        if None in r:
            raise SystemExit(f"{path}: wiersz danych {i} ma więcej pól niż nagłówek "
                             "(przecinek dziesiętny przy separatorze ','?)")
        rows.append({k.strip(): (v or "").strip() for k, v in r.items()})
    return rows


def parse_num(value: str, where: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        raise SystemExit(f"{where}: '{value}' nie jest liczbą") from None


def ols(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    """Nachylenie i wyraz wolny regresji liniowej MNK y = a + b*x."""
    n = len(xs)
    if n < 2:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def fill(out_dir: Path, measured_path: Path, reference_path: Path) -> None:
    results_path = out_dir / "results.csv"
    if not results_path.exists():
        raise SystemExit(f"brak {results_path} — najpierw uruchom cięcie (np. z --dry-run)")
    header, rows = read_csv(results_path)

    ref: dict[str, float | None] = {}
    for r in read_table(reference_path, ["segment", "ref_m"]):
        if r["segment"] in ref:
            raise SystemExit(f"{reference_path}: segment {r['segment']} powtórzony")
        ref[r["segment"]] = parse_num(r["ref_m"], f"{reference_path}: {r['segment']}")
    for seg in SEGMENTS:
        if ref.get(seg) is None or ref[seg] <= 0:
            raise SystemExit(f"{reference_path}: brak dodatniej wartości ref_m dla {seg}")

    measured: dict[str, dict[str, float | None]] = {}
    cols = [f"{seg}_measured_m" for seg in SEGMENTS]
    for r in read_table(measured_path, ["clip"] + cols):
        if r["clip"] in measured:
            raise SystemExit(f"{measured_path}: klip {r['clip']} powtórzony")
        measured[r["clip"]] = {
            seg: parse_num(r[f"{seg}_measured_m"], f"{measured_path}: {r['clip']} {seg}")
            for seg in SEGMENTS
        }
    if unknown := sorted(set(measured) - {row["clip"] for row in rows}):
        raise SystemExit(f"{measured_path}: klipy spoza results.csv: {unknown}")

    # Kolumny pomiarowe liczone od nowa z measured.csv — klip bez wiersza dostaje puste pola.
    points: dict[str, list[tuple[str, float, float, float]]] = {seg: [] for seg in SEGMENTS}
    for row in rows:
        m = measured.get(row["clip"], {})
        for seg in SEGMENTS:
            value = m.get(seg)
            if value is not None and value <= 0:
                raise SystemExit(f"{measured_path}: {row['clip']} {seg} = {value} (<= 0)")
            scale = None if value is None else ref[seg] / value
            row[f"{seg}_measured_m"] = "" if value is None else repr(value)
            row[f"{seg}_scale"] = "" if scale is None else f"{scale:.6f}"
            if scale is not None:
                points[seg].append((row["clip"], float(row["start_s"]), value, scale))
    write_csv(results_path, header, rows)

    def f6(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.6f}"

    lines = [
        "Test C — współczynnik skali vs punkt startu klipu",
        f"wygenerowano: {now_iso()}",
        f"results.csv:  {results_path.resolve()}",
        f"pomiary:      {measured_path.resolve()}  sha256={sha256_file(measured_path)}",
        f"referencja:   {reference_path.resolve()}  sha256={sha256_file(reference_path)}",
        "",
        "scale = ref_m / measured_m;  std = odchylenie standardowe próbkowe (n-1)",
        "regresja liniowa MNK: scale = a + b*x,  x = t_wzorca - start_s [s]"
        " (odległość czasowa kotwica -> wzorzec)",
    ]
    for seg in SEGMENTS:
        pts = sorted(points[seg], key=lambda p: p[1])
        t = SEGMENT_TIME_S[seg]
        lines += ["", f"{seg}   ref_m = {ref[seg]:g}   t_wzorca = {t:g} s   n = {len(pts)}"]
        if not pts:
            lines.append("  brak pomiarów")
            continue
        xs = [t - p[1] for p in pts]
        ys = [p[3] for p in pts]
        b, a = ols(xs, ys)
        lines += [
            f"  min       {f6(min(ys))}",
            f"  max       {f6(max(ys))}",
            f"  mean      {f6(statistics.fmean(ys))}",
            f"  std       {f6(statistics.stdev(ys) if len(ys) > 1 else None)}",
            f"  slope b   {f6(b)}  [1/s]",
            f"  intercept {f6(a)}",
            f"  {'clip':<10}{'start_s':>9}{'x_s':>9}{'measured_m':>12}{'scale':>11}",
        ]
        lines += [f"  {c:<10}{s:>9g}{t - s:>9g}{v:>12g}{sc:>11.6f}" for c, s, v, sc in pts]
    summary = "\n".join(lines) + "\n"
    (out_dir / "results_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"zapisano: {results_path} i {out_dir / 'results_summary.txt'}")


# ---------------------------------------------------------------- main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", help="nagranie źródłowe (wymagane poza --fill)")
    p.add_argument("--starts", default="0,3,6,9,12,15,18", help="starty klipów [s], po przecinku")
    p.add_argument("--end", type=float, default=30.0, help="wspólny koniec klipów [s]")
    p.add_argument("--out", default="runs/test_c")
    p.add_argument("--engine-dir", default="third_party/mast3r-slam")
    p.add_argument("--config", default="config/base.yaml", help="względem --engine-dir")
    p.add_argument("--engine-python", default=ENGINE_PYTHON_DEFAULT)
    p.add_argument("--reencode-all", action=argparse.BooleanOptionalAction, default=True,
                   help="przekoduj każdy klip tymi samymi parametrami, także gdy start wypada "
                        "na keyframie (domyślnie); --no-reencode-all przywraca próbę -c copy")
    p.add_argument("--dry-run", action="store_true", help="tylko cięcie i results.csv")
    p.add_argument("--fill", metavar="MEASURED_CSV", help="ręczne pomiary: clip,W1_measured_m,W2_measured_m")
    p.add_argument("--reference", metavar="REFERENCE_CSV", help="segment,ref_m (z --fill)")
    return p.parse_args(argv)


def parse_starts(text: str) -> list[float]:
    try:
        starts = [float(s) for s in text.split(",") if s.strip()]
    except ValueError:
        raise SystemExit(f"--starts: nie da się sparsować '{text}'") from None
    names = [clip_name(s) for s in starts]
    if len(set(names)) != len(names):
        raise SystemExit(f"--starts: powtórzone starty {names}")
    return starts


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # postęp widoczny także przy przekierowaniu do pliku
    args = parse_args(argv)
    out_dir = Path(args.out)

    if args.fill:
        if not args.reference:
            raise SystemExit("--fill wymaga --reference")
        fill(out_dir, Path(args.fill), Path(args.reference))
        return 0

    if not args.video:
        raise SystemExit("brak --video")
    src = Path(args.video).resolve()
    if not src.is_file():
        raise SystemExit(f"nie ma pliku {src}")
    if bad := ENGINE_PATH_KEYWORDS & set(out_dir.resolve().parts):
        raise SystemExit(f"--out zawiera katalog {bad}: upstreamowy load_dataset() "
                         "potraktowałby klip jako inny dataset niż MP4")
    starts = parse_starts(args.starts)
    end = args.end
    src_info = probe_source(src)
    if end > src_info["duration_s"]:
        raise SystemExit(f"--end {end} > długość nagrania {src_info['duration_s']:.3f} s")
    for s in starts:
        if not 0 <= s < end:
            raise SystemExit(f"start {s} poza [0, {end})")
        for seg, t in SEGMENT_TIME_S.items():
            if not s <= t < end:
                print(f"UWAGA: {clip_name(s)} nie obejmuje {seg} (t = {t:g} s)")

    engine_python = Path(os.path.expanduser(args.engine_python))
    engine = None if args.dry_run else engine_setup(args, engine_python)

    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = [prepare_clip(src, src_info, s, end, out_dir / clip_name(s), engine_python,
                         args.reencode_all) for s in starts]
    print(f"results.csv: {write_results(out_dir)}")
    if engine is None:
        return 0

    failed = []
    for cfg in cfgs:
        if not run_engine(cfg, out_dir / cfg["clip"], engine):
            failed.append(cfg["clip"])
        write_results(out_dir)
    if failed:
        print(f"silnik nie dał chmury dla: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
