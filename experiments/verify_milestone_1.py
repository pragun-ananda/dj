import sys
import wave
import struct
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import soundfile as sf
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sonicdj.config import Settings
from sonicdj.db.repository import DatabaseManager, TrackRepository, PlaylistRepository
from sonicdj.scanner.file_scanner import LibraryScanner
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.metadata.models import TrackMetadata, CueMetadata, normalize_key_to_camelot
from sonicdj.metadata.djay_exporter import DjayProExporter

console = Console()


def run_milestone_1_experiment():
    console.print(Panel.fit("[bold green]🧪 Milestone 1 End-to-End Verification Experiment[/]", border_style="green"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        music_dir = tmp_path / "audio_library"
        music_dir.mkdir()
        db_path = tmp_path / "experiment_library.db"

        # -------------------------------------------------------------
        # STEP 1: Generate Multi-Format Audio Files (FLAC, WAV, MP3)
        # -------------------------------------------------------------
        console.print("\n[bold cyan]1. Generating Multi-Format Test Audio Files...[/]")
        
        # 1. FLAC Track
        flac_path = music_dir / "Afrohouse" / "Black Coffee - Subconsciously.flac"
        flac_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(flac_path), np.zeros((44100 * 2, 2), dtype=np.float32), 44100, format="FLAC")

        # 2. WAV Track
        wav_path = music_dir / "Afrohouse" / "THEMBA - Sound of Freedom.wav"
        with wave.open(str(wav_path), "wb") as f:
            f.setnchannels(2); f.setsampwidth(2); f.setframerate(44100)
            f.writeframes(struct.pack("<h", 0) * 44100 * 2)

        # 3. MP3 Track
        mp3_path = music_dir / "DeepTech" / "Enoo Napa - Drones.mp3"
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mp3_path, "wb") as f:
            f.write(b"\xff\xfb\x90\x04" + b"\x00" * 414)

        console.print(f"  ✓ Created FLAC: {flac_path.name}")
        console.print(f"  ✓ Created WAV:  {wav_path.name}")
        console.print(f"  ✓ Created MP3:  {mp3_path.name}")

        # -------------------------------------------------------------
        # STEP 2: Write Enriched djay Pro Tags directly to files
        # -------------------------------------------------------------
        console.print("\n[bold cyan]2. Writing djay Pro Tags (Camelot, BPM, Energy, Cue Markers)...[/]")
        
        # Tag FLAC
        AudioTagEngine.write_djay_pro_tags(flac_path, TrackMetadata(
            title="Subconsciously",
            artist="Black Coffee",
            album="Subconsciously",
            genre="Afro House",
            year=2021,
            bpm=122.0,
            camelot="8A",
            key_raw="Am",
            energy=0.85,
            rating=5,
            comments="Peak vocal energy",
            cues=[CueMetadata(name="Intro", timestamp_ms=0), CueMetadata(name="Main Drop", timestamp_ms=64000)]
        ))

        # Tag WAV
        AudioTagEngine.write_djay_pro_tags(wav_path, TrackMetadata(
            title="Sound of Freedom",
            artist="THEMBA",
            album="Modern Rituals",
            genre="Afro House",
            year=2022,
            bpm=123.0,
            camelot="9A",
            key_raw="Em",
            energy=0.90,
            rating=5,
            comments="Heavy bassline drop",
            cues=[CueMetadata(name="Intro", timestamp_ms=0), CueMetadata(name="Drop", timestamp_ms=45000)]
        ))

        # Tag MP3
        AudioTagEngine.write_djay_pro_tags(mp3_path, TrackMetadata(
            title="Drones",
            artist="Enoo Napa",
            album="Drones EP",
            genre="Deep Tech",
            year=2023,
            bpm=124.0,
            camelot="8A",
            key_raw="Am",
            energy=0.80,
            rating=4,
            comments="Hypnotic rolling groove",
            cues=[CueMetadata(name="Intro", timestamp_ms=0), CueMetadata(name="Build", timestamp_ms=32000)]
        ))
        console.print("  ✓ All files enriched with djay Pro frames (TKEY, TBPM, POPM, COMM, Camelot)")

        # -------------------------------------------------------------
        # STEP 3: Run Recursive Library Scanner & Database Indexer
        # -------------------------------------------------------------
        console.print("\n[bold cyan]3. Executing Recursive Audio Library Scanner & DB Ingestion...[/]")
        db = DatabaseManager(f"sqlite:///{db_path}")
        scanner = LibraryScanner(db)
        stats = scanner.scan_directory(music_dir)

        console.print(f"  ✓ Files Discovered: {stats.total_found}")
        console.print(f"  ✓ Tracks Ingested:  {stats.added}")
        console.print(f"  ✓ Failed Files:     {stats.failed}")
        assert stats.added == 3, f"Expected 3 added tracks, got {stats.added}"

        # -------------------------------------------------------------
        # STEP 4: Query Engine & Harmonic Multi-Criteria DJ Filtering
        # -------------------------------------------------------------
        console.print("\n[bold cyan]4. Testing Real-Time Harmonic DJ Filtering (Query: Camelot 8A & BPM 120-125)...[/]")
        repo = TrackRepository(db)
        matches, total_matches = repo.list_tracks(camelot="8A", min_bpm=120.0, max_bpm=125.0)

        table = Table(title="Queried Compatible Tracks (Camelot 8A)", border_style="yellow")
        table.add_column("Artist", style="bold green")
        table.add_column("Title", style="bold white")
        table.add_column("Key", style="bold yellow", justify="center")
        table.add_column("BPM", justify="right", style="cyan")
        table.add_column("Energy", justify="right", style="magenta")
        table.add_column("Rating", justify="center", style="yellow")

        for m in matches:
            table.add_row(m.artist, m.title, f"{m.camelot} ({m.key_raw})", f"{m.bpm:.1f}", f"{int(m.energy * 100)}%", f"⭐ {m.rating}")

        console.print(table)
        assert total_matches == 2, f"Expected 2 matches for 8A, got {total_matches}"

        # -------------------------------------------------------------
        # STEP 5: Export to Algoriddim djay Pro Rekordbox XML & M3U8
        # -------------------------------------------------------------
        console.print("\n[bold cyan]5. Generating Rekordbox XML Collection for Algoriddim djay Pro...[/]")
        all_tracks, _ = repo.list_tracks(limit=100)
        xml_file = tmp_path / "djay_collection.xml"
        m3u_file = tmp_path / "djay_playlist.m3u8"

        DjayProExporter.export_rekordbox_xml(all_tracks, xml_file)
        DjayProExporter.export_m3u8(all_tracks, m3u_file)

        console.print(f"  ✓ Rekordbox XML Generated: {xml_file} ({xml_file.stat().st_size} bytes)")
        console.print(f"  ✓ M3U8 Playlist Generated: {m3u_file} ({m3u_file.stat().st_size} bytes)")

        # Validate XML structure
        tree = ET.parse(xml_file)
        root = tree.getroot()
        assert root.tag == "DJ_PLAYLISTS"
        entries = root.find("COLLECTION").findall("TRACK")
        assert len(entries) == 3, f"Expected 3 tracks in XML, got {len(entries)}"

        for track_node in entries:
            name = track_node.attrib["Name"]
            tonality = track_node.attrib["Tonality"]
            bpm = track_node.attrib["AverageBpm"]
            cues = track_node.findall("POSITION_MARK")
            console.print(f"    • XML Track: [bold white]{name}[/] | Key: [yellow]{tonality}[/] | BPM: [cyan]{bpm}[/] | Hot Cues: [green]{len(cues)} cues[/]")
            assert tonality in ("8A", "9A"), f"Unexpected tonality: {tonality}"
            assert len(cues) >= 1, "Missing hot cue points"

        # -------------------------------------------------------------
        # SUCCESS SUMMARY
        # -------------------------------------------------------------
        console.print(Panel.fit("[bold green]✅ ALL EXPERIMENT ASSERTIONS PASSED SUCCESSFULLY![/]\n"
                                "• Multi-format tag read/write verified (FLAC, WAV, MP3)\n"
                                "• SQLite repository with Camelot/BPM filtering verified\n"
                                "• Algoriddim djay Pro Rekordbox XML & M3U8 compatibility verified", border_style="green"))


if __name__ == "__main__":
    run_milestone_1_experiment()
