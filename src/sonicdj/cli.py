from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

from sonicdj.config import settings
from sonicdj.db.repository import get_db, TrackRepository, PlaylistRepository
from sonicdj.scanner.file_scanner import LibraryScanner
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.metadata.models import TrackMetadata, normalize_key_to_camelot
from sonicdj.metadata.djay_exporter import DjayProExporter

app = typer.Typer(
    name="sonicdj",
    help="SonicDJ — AI-powered DJ curation, metadata enrichment, and djay Pro companion.",
    add_completion=False,
)
console = Console()


@app.command()
def scan(
    directory: Path = typer.Argument(..., help="Path to local folder containing audio files"),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Custom SQLite database file"),
):
    """Scan and index local audio files into the SonicDJ library."""
    db = get_db(f"sqlite:///{db_path}" if db_path else None)
    scanner = LibraryScanner(db)

    directory = directory.resolve()
    if not directory.exists():
        console.print(f"[bold red]Error:[/] Directory '{directory}' does not exist.")
        raise typer.Exit(1)

    console.print(Panel.fit(f"[bold cyan]SonicDJ Library Scanner[/]\nTarget Directory: {directory}", border_style="cyan"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning audio files...", total=None)

        def on_progress(file_path: Path, stats):
            progress.update(task, total=stats.total_found, completed=stats.scanned, description=f"Scanning: {file_path.name[:30]}")

        stats = scanner.scan_directory(directory, progress_callback=on_progress)

    # Summary table
    table = Table(title="Scan Completed Summary", border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Count", style="cyan")
    table.add_row("Total Files Discovered", str(stats.total_found))
    table.add_row("Newly Added Tracks", str(stats.added))
    table.add_row("Updated Tracks", str(stats.updated))
    table.add_row("Failed / Corrupt Files", str(stats.failed))
    console.print(table)


@app.command(name="list")
def list_tracks(
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Filter by genre"),
    camelot: Optional[str] = typer.Option(None, "--camelot", "-k", help="Filter by Camelot key (e.g. 8A, 11B)"),
    min_bpm: Optional[float] = typer.Option(None, "--min-bpm", help="Minimum BPM"),
    max_bpm: Optional[float] = typer.Option(None, "--max-bpm", help="Maximum BPM"),
    min_energy: Optional[float] = typer.Option(None, "--min-energy", help="Minimum energy score (0.0 - 1.0)"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search title, artist, album, comments"),
    limit: int = typer.Option(25, "--limit", "-l", help="Number of tracks to display"),
):
    """List indexed tracks with DJ metadata (BPM, Camelot Key, Energy)."""
    db = get_db()
    repo = TrackRepository(db)

    tracks, total = repo.list_tracks(
        genre=genre,
        camelot=camelot,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        min_energy=min_energy,
        search_query=search,
        limit=limit,
    )

    if not tracks:
        console.print("[yellow]No tracks found matching your query criteria.[/yellow]")
        return

    table = Table(title=f"SonicDJ Library ({len(tracks)} of {total} tracks)", border_style="magenta")
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Artist", style="bold green")
    table.add_column("Title", style="bold white")
    table.add_column("Key", style="bold yellow", justify="center")
    table.add_column("BPM", justify="right", style="cyan")
    table.add_column("Energy", justify="right", style="magenta")
    table.add_column("Genre", style="blue")
    table.add_column("Duration", justify="right", style="dim")

    for t in tracks:
        dur_min = int(t.duration_sec // 60)
        dur_sec = int(t.duration_sec % 60)
        dur_str = f"{dur_min}:{dur_sec:02d}"
        key_display = f"{t.camelot} ({t.key_raw})" if (t.camelot and t.key_raw and t.camelot != t.key_raw) else (t.camelot or t.key_raw or "-")
        bpm_display = f"{t.bpm:.1f}" if t.bpm else "-"
        energy_display = f"{int(t.energy * 100)}%" if t.energy else "-"

        table.add_row(
            str(t.id),
            t.artist or "Unknown",
            t.title or "Unknown",
            key_display,
            bpm_display,
            energy_display,
            t.genre or "-",
            dur_str,
        )

    console.print(table)


@app.command()
def tag(
    file_path: Path = typer.Argument(..., help="Path to audio file"),
    bpm: Optional[float] = typer.Option(None, "--bpm", help="Set BPM"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Set musical key / Camelot (e.g. 8A, Am)"),
    energy: Optional[float] = typer.Option(None, "--energy", "-e", help="Set energy rating (0.0 - 1.0)"),
    rating: Optional[int] = typer.Option(None, "--rating", "-r", help="Set 1-5 star rating"),
    comment: Optional[str] = typer.Option(None, "--comment", "-c", help="Set DJ comment notes"),
):
    """Enrich an audio file with djay Pro compatible ID3/Vorbis tags."""
    file_path = file_path.resolve()
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File '{file_path}' does not exist.")
        raise typer.Exit(1)

    meta = AudioTagEngine.read_metadata(file_path)
    if bpm is not None:
        meta.bpm = bpm
    if key is not None:
        cam, raw_k = normalize_key_to_camelot(key)
        meta.camelot = cam
        meta.key_raw = raw_k
    if energy is not None:
        meta.energy = energy
    if rating is not None:
        meta.rating = rating
    if comment is not None:
        meta.comments = comment

    success = AudioTagEngine.write_djay_pro_tags(file_path, meta)
    if success:
        console.print(f"[bold green]✓[/] Successfully tagged '{file_path.name}' for djay Pro!")
        # Update in database if indexed
        db = get_db()
        repo = TrackRepository(db)
        repo.upsert_track({
            "file_path": str(file_path),
            "file_hash": meta.file_hash,
            "title": meta.title,
            "artist": meta.artist,
            "bpm": meta.bpm,
            "camelot": meta.camelot,
            "key_raw": meta.key_raw,
            "energy": meta.energy,
            "rating": meta.rating,
            "comments": meta.comments,
        })
    else:
        console.print(f"[bold red]Failed to write tags to '{file_path.name}'[/]")


@app.command()
def export_djay(
    output_xml: Path = typer.Argument(..., help="Target path for Rekordbox XML file (e.g. djay_collection.xml)"),
    m3u8: bool = typer.Option(False, "--m3u8", help="Also generate an accompanying M3U8 playlist"),
):
    """Export the entire SonicDJ library into a Rekordbox XML collection for Algoriddim djay Pro."""
    db = get_db()
    repo = TrackRepository(db)
    tracks, total = repo.list_tracks(limit=100000)

    if not tracks:
        console.print("[yellow]Library is empty. Scan audio files first using 'sonicdj scan <path>'.[/yellow]")
        return

    xml_path = DjayProExporter.export_rekordbox_xml(tracks, output_xml)
    console.print(f"[bold green]✓[/] Exported {len(tracks)} tracks to Rekordbox XML for djay Pro: [cyan]{xml_path}[/]")

    if m3u8:
        m3u_target = xml_path.with_suffix(".m3u8")
        DjayProExporter.export_m3u8(tracks, m3u_target)
        console.print(f"[bold green]✓[/] Exported M3U8 playlist: [cyan]{m3u_target}[/]")


@app.command()
def info(file_path: Path = typer.Argument(..., help="Path to audio file")):
    """Inspect all extracted audio properties and tags of a file."""
    file_path = file_path.resolve()
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File '{file_path}' does not exist.")
        raise typer.Exit(1)

    meta = AudioTagEngine.read_metadata(file_path)
    table = Table(title=f"Metadata: {file_path.name}", border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="yellow")

    table.add_row("Title", meta.title)
    table.add_row("Artist", meta.artist)
    table.add_row("Album", meta.album)
    table.add_row("Genre", meta.genre)
    table.add_row("BPM", f"{meta.bpm:.2f}" if meta.bpm else "Not set")
    table.add_row("Camelot Key", meta.camelot or "Not set")
    table.add_row("Musical Key", meta.key_raw or "Not set")
    table.add_row("Energy Score", f"{int(meta.energy * 100)}%" if meta.energy else "Not set")
    table.add_row("Rating", f"{meta.rating} Stars" if meta.rating else "0")
    table.add_row("Duration", f"{meta.duration_sec:.1f}s")
    table.add_row("Format / Sample Rate", f"{meta.format.upper()} @ {meta.sample_rate}Hz, {meta.channels}ch")
    table.add_row("Comments", meta.comments or "None")

    console.print(table)


if __name__ == "__main__":
    app()
