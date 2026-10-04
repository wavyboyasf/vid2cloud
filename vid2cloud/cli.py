"""Interfejs wiersza poleceń vid2cloud.

Cztery komendy odpowiadające etapom pracy z materiałem:
    run     — pełny przebieg: wideo -> chmura punktów
    select  — sama selekcja klatek, bez rekonstrukcji (diagnostyka materiału)
    scale   — nadanie skali metrycznej gotowej chmurze na podstawie wzorców
    eval    — porównanie wyniku z referencją i zestawienie metryk

`run` działa (vid2cloud/run.py, env mast3r-slam). Pozostałe komendy na razie tylko
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
    max_frames: Annotated[Optional[int], typer.Option("--max-frames", help="Ogranicz liczbę klatek (do testów).")] = None,
    single_thread: Annotated[bool, typer.Option(
        "--single-thread/--no-single-thread",
        help="Czekaj na backend po każdej klatce kluczowej (deterministyczne zrzuty). "
             "--no-single-thread = zachowanie upstreamu, jak na żywo.",
    )] = True,
) -> None:
    """Pełny przebieg MASt3R-SLAM: chmura, zrzuty klatek kluczowych, historia póz Sim(3), log per klatka.

    Uruchamiać w env `mast3r-slam`. Każda klatka idzie do silnika (selekcja klatek jeszcze nie istnieje).
    """
    from vid2cloud.run import run_video  # lokalnie: torch i upstream tylko dla tej komendy

    record = run_video(video, out, config_path=config, single_thread=single_thread, max_frames=max_frames)
    times, results = record["times_s"], record["results"]
    typer.echo(
        f"gotowe: {results['n_frames']} klatek, {results['n_keyframes']} klatek kluczowych, "
        f"{results['n_points_cloud']} punktów, {times['wall_total']:.0f} s -> {record['out']}"
    )


@app.command()
def select(
    video: Annotated[Path, typer.Argument(help="Plik wideo wejściowy.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Katalog na log CSV i podgląd wybranych klatek.")] = Path("runs/select"),
    min_sharpness: Annotated[Optional[float], typer.Option("--min-sharpness", help="Próg miary ostrości.")] = None,
    min_interval: Annotated[Optional[float], typer.Option("--min-interval", help="Minimalny odstęp między wyborami [s].")] = None,
) -> None:
    """Sama selekcja klatek — bez rekonstrukcji. Wynikiem jest log CSV per klatka."""
    _not_implemented("select", video=video, out=out, min_sharpness=min_sharpness, min_interval=min_interval)


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
