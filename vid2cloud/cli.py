"""Interfejs wiersza poleceń vid2cloud.

Komendy odpowiadające etapom pracy z materiałem:
    run     — pełny przebieg: wideo -> chmura punktów (opcjonalnie tylko klatki z listy)
    quality — przebieg wstępny bez GPU: miary jakości każdej klatki, wykres ostrości
    select  — listy klatek (stały krok, argmax ostrości w oknach) z quality.csv
    scale   — nadanie skali metrycznej gotowej chmurze na podstawie wzorców
    eval    — porównanie wyniku z referencją i zestawienie metryk

`run` działa w env mast3r-slam (vid2cloud/run.py); `quality` i `select` w env vid2cloud-tools
(`python -m vid2cloud.cli ...`, vid2cloud/input/quality.py). `scale` i `eval` na razie tylko
wypisują swoje argumenty i "not implemented" — doprecyzuje je sesja, która je implementuje.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="vid2cloud",
    help="Gęsta chmura punktów z wideo, klatka po klatce.",
    add_completion=False,
    no_args_is_help=True,
)


def _not_implemented(command: str, **kwargs: object) -> None:
    """Wypisuje nazwę komendy i przekazane argumenty. Miejsce na przyszłą logikę."""
    typer.echo(f"{command}: not implemented")
    for key, value in kwargs.items():
        typer.echo(f"  {key} = {value!r}")


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
) -> None:
    """Pełny przebieg MASt3R-SLAM: chmura, zrzuty klatek kluczowych, historia póz Sim(3), log per klatka.

    Uruchamiać w env `mast3r-slam`. Bez --frames każda klatka idzie do silnika.
    """
    from vid2cloud.run import run_video  # lokalnie: torch i upstream tylko dla tej komendy

    record = run_video(video, out, config_path=config, single_thread=single_thread, max_frames=max_frames,
                       frames=frames)
    times, results = record["times_s"], record["results"]
    fed = f" ({results['n_frames_fed']} podanych do silnika)" if "n_frames_fed" in results else ""
    typer.echo(
        f"gotowe: {results['n_frames']} klatek{fed}, {results['n_keyframes']} klatek kluczowych, "
        f"{results['n_points_cloud']} punktów, {times['wall_total']:.0f} s -> {record['out']}"
    )


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
    cloud: Annotated[Path, typer.Argument(help="Chmura punktów do przeskalowania (PLY).")],
    targets: Annotated[Path, typer.Option("--targets", "-t", help="Plik targets.json ze zmierzonymi wzorcami.")] = Path("targets.json"),
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Chmura wyjściowa; domyślnie obok wejściowej.")] = None,
) -> None:
    """Nadaje chmurze skalę metryczną na podstawie fizycznych wzorców z targets.json."""
    _not_implemented("scale", cloud=cloud, targets=targets, out=out)


@app.command("eval")
def eval_(
    run_dir: Annotated[Path, typer.Argument(help="Katalog przebiegu do oceny (runs/<nazwa>).")],
    reference: Annotated[Optional[Path], typer.Option("--reference", "-r", help="Chmura referencyjna (lidar / Metashape).")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Plik na zestawienie metryk.")] = None,
) -> None:
    """Porównuje wynik przebiegu z referencją i zapisuje zestawienie metryk."""
    _not_implemented("eval", run_dir=run_dir, reference=reference, out=out)


if __name__ == "__main__":
    app()
