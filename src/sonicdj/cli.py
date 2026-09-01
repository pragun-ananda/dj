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
def analyze(
    target_path: Path = typer.Argument(..., help="Audio file or folder to analyze"),
    auto_tag: bool = typer.Option(True, "--tag/--no-tag", help="Write analyzed tags directly to audio files"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel analyzer workers"),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Custom SQLite database file"),
):
    """Run deep MIR analysis (BPM, Camelot Key, Phrasing, Energy, Vocals, Auto-Cues)."""
    from sonicdj.analysis.batch_analyzer import AudioAnalyzer
    db = get_db(f"sqlite:///{db_path}" if db_path else None)
    analyzer = AudioAnalyzer(db)

    target_path = target_path.resolve()
    if not target_path.exists():
        console.print(f"[bold red]Error:[/] Path '{target_path}' does not exist.")
        raise typer.Exit(1)

    if target_path.is_file():
        console.print(Panel.fit(f"[bold cyan]Analyzing Single Track:[/] {target_path.name}", border_style="cyan"))
        _, analysis = analyzer.analyze_and_enrich_track(target_path, auto_tag_file=auto_tag)

        table = Table(title=f"Analysis: {target_path.name}", border_style="green")
        table.add_column("Metric", style="bold")
        table.add_column("Extracted Value", style="yellow")

        table.add_row("Camelot Key", f"{analysis.key_info.camelot} ({analysis.key_info.musical_key}) [conf: {int(analysis.key_info.confidence*100)}%]")
        table.add_row("Tuning / Drift", f"{analysis.key_info.tuning_hz} Hz ({analysis.key_info.pitch_drift_cents:+0.1f} cents)")
        table.add_row("Tempo (BPM)", f"{analysis.phrasing_info.bpm:.2f} BPM [conf: {int(analysis.phrasing_info.confidence*100)}%]")
        table.add_row("Downbeat (Bar 1)", f"{analysis.phrasing_info.first_downbeat_sec:.3f}s")
        table.add_row("Overall Energy", f"{int(analysis.vocal_energy_info.overall_energy * 100)}%")
        table.add_row("Vocal Presence", f"{analysis.vocal_energy_info.vocal_presence_percent}% (Intro inst: {analysis.vocal_energy_info.instrumental_intro_sec}s)")
        table.add_row("Generated Cues", f"{len(analysis.generated_cues)} Hot Cues")
        table.add_row("djay Pro Comment", analysis.summary_comment)
        console.print(table)

        # Print Cue Breakdown Table
        cue_table = Table(title="Generated djay Pro Hot Cues", border_style="magenta")
        cue_table.add_column("Hot Cue", justify="center", style="bold")
        cue_table.add_column("Name", style="bold white")
        cue_table.add_column("Time", justify="right", style="cyan")
        cue_table.add_column("Type", style="dim")

        for idx, cue in enumerate(analysis.generated_cues):
            cue_letter = chr(ord('A') + (cue.hot_cue_index if cue.hot_cue_index is not None else idx))
            cue_time = f"{cue.timestamp_ms / 1000.0:.2f}s"
            cue_table.add_row(f"Pad {cue_letter}", cue.name, cue_time, cue.cue_type)
        console.print(cue_table)

    else:
        console.print(Panel.fit(f"[bold cyan]SonicDJ Batch Audio Analyzer[/]\nDirectory: {target_path}", border_style="cyan"))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing audio collection...", total=None)

            def on_progress(file_path: Path, stats):
                progress.update(task, total=stats.total_found, completed=stats.analyzed, description=f"Analyzed: {file_path.name[:30]}")

            stats = analyzer.batch_analyze_directory(
                target_path, auto_tag_file=auto_tag, max_workers=workers, progress_callback=on_progress
            )

        # Summary
        table = Table(title="Batch Analysis Summary", border_style="green")
        table.add_column("Metric", style="bold")
        table.add_column("Count", style="cyan")
        table.add_row("Total Files Discovered", str(stats.total_found))
        table.add_row("Successfully Analyzed", str(stats.analyzed))
        table.add_row("Tags Embedded into Files", str(stats.tagged))
        table.add_row("Failed Analyses", str(stats.failed))
        console.print(table)


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


@app.command()
def search(
    query: Optional[str] = typer.Argument(None, help="Natural language description of music, vibe, or mood"),
    ref_id: Optional[int] = typer.Option(None, "--ref", help="Reference track ID to find compatible mix companions"),
    camelot: Optional[str] = typer.Option(None, "--key", "-k", help="Target Camelot key (e.g. 8A, 9B)"),
    bpm: Optional[float] = typer.Option(None, "--bpm", "-b", help="Target BPM tempo"),
    bpm_tolerance: float = typer.Option(6.0, "--tolerance", "-t", help="BPM tolerance window (+/- BPM)"),
    min_energy: Optional[float] = typer.Option(None, "--min-energy", help="Minimum energy score (0.0 to 1.0)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of search results"),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Custom SQLite database file"),
):
    """Semantic vector search over audio collection with real-time harmonic DJ constraints."""
    from sonicdj.search.hybrid_engine import HybridSearchEngine

    db = get_db(f"sqlite:///{db_path}" if db_path else None)
    engine = HybridSearchEngine(db)

    header = f"🔍 Semantic & Harmonic DJ Search"
    if query:
        header += f" | Prompt: '{query}'"
    if camelot:
        header += f" | Key: {camelot}"
    if bpm:
        header += f" | BPM: {bpm:.1f} (±{bpm_tolerance:.1f})"

    console.print(Panel.fit(f"[bold cyan]{header}[/]", border_style="cyan"))

    results = engine.search(
        prompt=query,
        reference_track_id=ref_id,
        target_camelot=camelot,
        target_bpm=bpm,
        bpm_tolerance=bpm_tolerance,
        min_energy=min_energy,
        limit=limit,
    )

    if not results:
        console.print("[yellow]No matching tracks found matching your harmonic/semantic criteria.[/yellow]")
        return

    table = Table(title=f"Ranked Mix Results ({len(results)} matches)", border_style="green")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Match", justify="right", style="bold green")
    table.add_column("Track (Artist - Title)", style="bold white")
    table.add_column("Key", justify="center", style="bold yellow")
    table.add_column("BPM", justify="right", style="cyan")
    table.add_column("Energy", justify="right", style="magenta")
    table.add_column("Subgenre", style="dim")
    table.add_column("Mix Recommendation", style="italic yellow")

    for idx, r in enumerate(results):
        t = r.track
        match_pct = f"{int(r.composite_score * 100)}%"
        bpm_str = f"{t.bpm:.1f}" if t.bpm else "—"
        energy_str = f"{int(t.energy * 100)}%" if t.energy else "—"
        track_str = f"{t.artist} - {t.title}"

        table.add_row(
            f"#{idx+1}",
            match_pct,
            track_str,
            f"{t.camelot or '—'}",
            bpm_str,
            energy_str,
            r.subgenre_info.primary_subgenre,
            r.mix_recommendation,
        )

    console.print(table)


@app.command(name="mix-path")
def mix_path(
    start_id: int = typer.Argument(..., help="Track ID of opening song"),
    target_id: int = typer.Argument(..., help="Track ID of target climax song"),
    max_hops: int = typer.Option(5, "--hops", "-h", help="Maximum intermediate track steps"),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Custom SQLite database file"),
):
    """Find the optimal harmonic setlist path from Track A to Track B."""
    from sonicdj.sandbox.transition_graph import TransitionGraphEngine

    db = get_db(f"sqlite:///{db_path}" if db_path else None)
    engine = TransitionGraphEngine(db)

    steps = engine.find_optimal_set_path(start_id, target_id, max_hops=max_hops)
    if not steps:
        console.print("[bold red]Error:[/] Could not compute a path between the specified tracks.")
        raise typer.Exit(1)

    console.print(Panel.fit(f"[bold cyan]🔀 Harmonic DJ Setlist Path: Track #{start_id} ➔ Track #{target_id}[/]", border_style="cyan"))

    table = Table(title=f"Optimal Set Progression ({len(steps)} Tracks)", border_style="green")
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Track (Artist - Title)", style="bold white")
    table.add_column("Key", justify="center", style="bold yellow")
    table.add_column("BPM", justify="right", style="cyan")
    table.add_column("Energy", justify="right", style="magenta")
    table.add_column("Transition Advice", style="italic yellow")

    for step in steps:
        t = step.track
        bpm_str = f"{t.bpm:.1f}" if t.bpm else "—"
        energy_str = f"{int(t.energy * 100)}%" if t.energy else "—"
        advice = step.transition_from_prev.explanation if step.transition_from_prev else "[bold green]▶ Opening Track[/]"

        table.add_row(
            f"Step {step.step_number}",
            f"{t.artist} - {t.title}",
            t.camelot or "—",
            bpm_str,
            energy_str,
            advice,
        )

    console.print(table)


@app.command()
def audition(
    track_a: Path = typer.Argument(..., help="Path to Deck A audio file (outgoing track)"),
    track_b: Path = typer.Argument(..., help="Path to Deck B audio file (incoming track)"),
    bars: int = typer.Option(16, "--bars", "-b", help="Length of audition transition in bars (8, 16, or 32)"),
    output: Optional[Path] = typer.Option(None, "--out", "-o", help="Output WAV file path for rendered mix preview"),
):
    """Audition a seamless 16/32-bar DJ mix transition between two audio files."""
    from sonicdj.sandbox.quick_mix_engine import QuickMixAuditionEngine

    track_a = track_a.resolve()
    track_b = track_b.resolve()

    if not track_a.exists() or not track_b.exists():
        console.print(f"[bold red]Error:[/] One or both audio files do not exist.")
        raise typer.Exit(1)

    out_path = output or Path("quick_mix_audition.wav")
    console.print(Panel.fit(f"[bold cyan]🎧 Rendering {bars}-Bar Virtual DJ Mix Transition[/]\nDeck A: {track_a.name}\nDeck B: {track_b.name}", border_style="cyan"))

    mix, sr = QuickMixAuditionEngine.render_16bar_audition(
        track_a, track_b, output_wav_path=out_path, num_bars=bars
    )

    duration_sec = len(mix) / float(sr)
    console.print(f"[bold green]✓[/] Rendered {duration_sec:.1f}s transition preview to [cyan]{out_path}[/]")


if __name__ == "__main__":
    app()
