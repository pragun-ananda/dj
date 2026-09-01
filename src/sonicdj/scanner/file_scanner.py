import os
from pathlib import Path
from typing import Callable, List, Optional, Generator
from dataclasses import dataclass

from sonicdj.config import settings
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.db.repository import TrackRepository, DatabaseManager


@dataclass
class ScanStats:
    total_found: int = 0
    scanned: int = 0
    added: int = 0
    updated: int = 0
    failed: int = 0
    skipped_duplicates: int = 0


class LibraryScanner:
    """Recursively scans folders, reads metadata, and indexes tracks into SQLite."""

    def __init__(self, db: DatabaseManager, supported_extensions: Optional[set[str]] = None):
        self.db = db
        self.repo = TrackRepository(db)
        self.supported_extensions = supported_extensions or settings.supported_extensions

    def find_audio_files(self, root_dir: Path) -> Generator[Path, None, None]:
        """Yield all supported audio files in root directory recursively."""
        root_dir = Path(root_dir).resolve()
        if not root_dir.exists():
            return

        if root_dir.is_file():
            if root_dir.suffix.lower() in self.supported_extensions:
                yield root_dir
            return

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.supported_extensions:
                    yield Path(dirpath) / filename

    def scan_directory(
        self,
        directory: Path,
        progress_callback: Optional[Callable[[Path, ScanStats], None]] = None,
    ) -> ScanStats:
        """Scan a directory and index all audio tracks into the database."""
        directory = Path(directory).resolve()
        stats = ScanStats()

        files_to_scan = list(self.find_audio_files(directory))
        stats.total_found = len(files_to_scan)

        for file_path in files_to_scan:
            try:
                # 1. Read metadata and audio technical properties
                meta = AudioTagEngine.read_metadata(file_path)

                # 2. Prepare dictionary for database upsert
                track_data = {
                    "file_path": str(file_path),
                    "file_hash": meta.file_hash,
                    "file_size_bytes": meta.file_size_bytes,
                    "format": meta.format,
                    "duration_sec": meta.duration_sec,
                    "sample_rate": meta.sample_rate,
                    "bitrate_kbps": meta.bitrate_kbps,
                    "channels": meta.channels,
                    "title": meta.title,
                    "artist": meta.artist,
                    "album": meta.album,
                    "album_artist": meta.album_artist,
                    "genre": meta.genre,
                    "year": meta.year,
                    "isrc": meta.isrc,
                    "bpm": meta.bpm,
                    "key_raw": meta.key_raw,
                    "camelot": meta.camelot,
                    "energy": meta.energy,
                    "rating": meta.rating,
                    "comments": meta.comments,
                }

                cue_data = [
                    {
                        "name": c.name,
                        "timestamp_ms": c.timestamp_ms,
                        "cue_type": c.cue_type,
                        "hot_cue_index": c.hot_cue_index,
                        "color_hex": c.color_hex,
                    }
                    for c in meta.cues
                ]

                # Check if existing
                existing = self.repo.get_track_by_path(str(file_path))
                self.repo.upsert_track(track_data, cues=cue_data)

                if existing:
                    stats.updated += 1
                else:
                    stats.added += 1

            except Exception as e:
                stats.failed += 1

            stats.scanned += 1
            if progress_callback:
                progress_callback(file_path, stats)

        return stats
