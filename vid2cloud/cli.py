"""Interfejs wiersza poleceń vid2cloud.

Komendy odpowiadające etapom pracy z materiałem:
    run     — pełny przebieg: wideo -> chmura punktów (opcjonalnie tylko klatki z listy);
              z --targets na końcu ten sam krok skali co `scale`
    quality — przebieg wstępny bez GPU: miary jakości każdej klatki, wykres ostrości
    select  — listy klatek (stały krok, argmax ostrości w oknach) z quality.csv
    scale   — skala metryczna osobno dla każdej klatki kluczowej gotowego przebiegu
              (dwie metody + wariant globalny), chmury w metrach

`run` działa w env mast3r-slam (vid2cloud/run.py); `quality`, `select` i `scale` w env
vid2cloud-tools (`python -m vid2cloud.cli ...`; vid2cloud/input/quality.py,
vid2cloud/scale/pipeline.py). Ocena względem TLS jest w eval/ (eval_cloud.py, eval_all.sh) —
komenda `eval` została usunięta 2026-11-04 (decyzja autora z 02.11).
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

from vid2cloud.scale.lift import C_MIN

app = typer.Typer(
    name="vid2cloud",
    help="Gęsta chmura punktów z wideo, klatka po klatce.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def run(
    video: Annotated[Path, typer.Argument(help="Plik wideo wejściowy.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Katalog przebiegu (runs/<nazwa>).")] = Path("runs/default"),
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Plik konfiguracyjny silnika (domyślnie base.yaml upstreamu).")] = None,
    max_frames: Annotated[Optional[int], typer.Option("--max-frames", help="Ogranicz liczbę zdekodowanych klatek (do testów).")] = None,
    single_thread: Annotated[bool, typer.Option(
        "--single-thread/--no-single-thread",
        help="Czekaj na backend po każdej klatce kluczowej (deterministyczne zrzuty). "
             "--no-single-thread = zachowanie upstreamu, jak na żywo.",
    )] = True,
    frames: Annotated[Optional[Path], typer.Option(
        "--frames",
        help="Plik z listą indeksów klatek (jeden w linii, rosnąco). Dekodowane są wszystkie klatki, "
             "do silnika trafiają tylko te z listy; log.csv dostaje kolumnę fed.",
    )] = None,
    targets: Annotated[Optional[Path], typer.Option(
        "--targets", "-t",
        help="targets.json ze zmierzonymi odcinkami: po zakończeniu silnika skala osobno dla każdej klatki "
             "kluczowej (jak `vid2cloud scale`). Błąd skali nie psuje wyników silnika, ale kod wyjścia ≠ 0.",
    )] = None,
) -> None:
    """Pełny przebieg MASt3R-SLAM: chmura, zrzuty klatek kluczowych, historia póz Sim(3), log per klatka.

    Uruchamiać w env `mast3r-slam`. Bez --frames każda klatka idzie do silnika. Bez --targets
    przebieg jak dotąd (bez sekcji "scale" w config.json).
    """
    from vid2cloud.run import run_video  # lokalnie: torch i upstream tylko dla tej komendy

    record = run_video(video, out, config_path=config, single_thread=single_thread, max_frames=max_frames,
                       frames=frames, targets=targets)
    times, results = record["times_s"], record["results"]
    fed = f" ({results['n_frames_fed']} podanych do silnika)" if "n_frames_fed" in results else ""
    typer.echo(
        f"gotowe: {results['n_frames']} klatek{fed}, {results['n_keyframes']} klatek kluczowych, "
        f"{results['n_points_cloud']} punktów, {times['wall_total']:.0f} s -> {record['out']}"
    )
    if "scale" in record:
        sc = record["scale"]
        typer.echo(f"skala: {sc['status']}  ({times.get('scale', 0):.1f} s)"
                   + (f" — {sc['message']}" if sc.get("message") else ""))
        scale_json = sc.get("files", {}).get("scale_json")
        if scale_json:
            import json

            from vid2cloud.scale.pipeline import format_report

            typer.echo(format_report(json.loads(Path(scale_json).read_text(encoding="utf-8"))))
        if sc["status"] == "error":
            raise typer.Exit(code=1)


@app.command()
def quality(
    video: Annotated[Path, typer.Argument(help="Plik wideo wejściowy.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Katalog na quality.csv, fig_ostrosc.png, quality_report.json.")],
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Nadpisz istniejące pliki w --out.")] = False,
) -> None:
    """Przebieg wstępny bez GPU: miary ostrości i ekspozycji każdej klatki na obrazie wejścia sieci.

    Uruchamiać w env `vid2cloud-tools`: `python -m vid2cloud.cli quality <mp4> --out <katalog>`.
    """
    from vid2cloud.input.quality import format_quality_report, run_quality

    typer.echo(format_quality_report(run_quality(video, out, overwrite=overwrite)))


@app.command()
def select(
    quality_csv: Annotated[Path, typer.Argument(help="quality.csv z `vid2cloud quality`.")],
    k: Annotated[int, typer.Option("--k", help="Krok / szerokość okna w klatkach.")] = 3,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Katalog na listy; domyślnie <katalog quality.csv>/frames.")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Nadpisz istniejące listy w --out.")] = False,
) -> None:
    """Listy klatek dla `run --frames`: stały krok (k przesunięć), argmax tenengradu w oknach bez i z progami.

    Uruchamiać w env `vid2cloud-tools`. Bez dekodowania — tylko z quality.csv.
    """
    from vid2cloud.input.quality import format_select_report, run_select

    typer.echo(format_select_report(run_select(quality_csv, k, out, overwrite=overwrite)))


@app.command()
def scale(
    run_dir: Annotated[Path, typer.Argument(help="Katalog gotowego przebiegu (runs/<nazwa>, single_thread).")],
    targets: Annotated[Path, typer.Option("--targets", "-t", help="targets.json ze zmierzonymi odcinkami (mm).")],
    c_min: Annotated[float, typer.Option("--c-min", help="Próg średniej ufności pikseli liftu.")] = C_MIN,
    sigma_obs: Annotated[Optional[float], typer.Option(
        "--sigma-obs", help="σ_obs metody B (ln f); domyślnie zamrożone z vid2cloud/scale/params.json.")] = None,
    tau: Annotated[Optional[float], typer.Option(
        "--tau", help="τ metody B (ln f / √s); domyślnie zamrożone z vid2cloud/scale/params.json.")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Nadpisz scale.json, scale_keyframes.csv, cloud_m_*.ply.")] = False,
) -> None:
    """Skala osobno dla każdej klatki kluczowej: f_k metodą A (interpolacja) i B (MNK) + wariant globalny.

    Zapisuje w katalogu przebiegu scale_per_keyframe.csv, scale_keyframes.csv, cloud_m_global.ply,
    cloud_m_kf_interp.ply, cloud_m_kf_mnk.ply i scale.json; wypisuje tabelę f_k. Bez GPU —
    env `vid2cloud-tools`: `python -m vid2cloud.cli scale <katalog> --targets <json>`.
    """
    from vid2cloud.scale.pipeline import format_report, scale_run

    try:
        record = scale_run(run_dir, targets, c_min=c_min, sigma_obs=sigma_obs, tau=tau, overwrite=overwrite,
                           log=typer.echo)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"błąd: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(format_report(record))


if __name__ == "__main__":
    app()
