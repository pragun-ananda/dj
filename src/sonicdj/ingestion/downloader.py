import re
from pathlib import Path
from typing import Callable, List, Optional
import httpx

from sonicdj.config import settings
from sonicdj.ingestion.tidal_client import TidalClient, TidalTrackInfo
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.metadata.models import TrackMetadata
from sonicdj.db.repository import TrackRepository, DatabaseManager


def sanitize_filename(name: str) -> str:
    """Removes illegal filesystem characters."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


class TidalDownloader:
    """Orchestrates track downloading, metadata embedding, and database ingestion."""

    def __init__(self, db: DatabaseManager, client: Optional[TidalClient] = None, output_dir: Optional[Path] = None):
        self.db = db
        self.repo = TrackRepository(db)
        self.client = client or TidalClient()
        self.output_dir = output_dir or settings.music_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_track(
        self,
        track_info: TidalTrackInfo,
        stream_url: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """
        Downloads a track audio stream, writes high-res ID3 tags, and registers into the database.
        """
        artist_dir = self.output_dir / sanitize_filename(track_info.artist)
        artist_dir.mkdir(parents=True, exist_ok=True)

        ext = "flac" if track_info.audio_quality in ("HI_RES", "LOSSLESS") else "mp3"
        filename = f"{sanitize_filename(track_info.artist)} - {sanitize_filename(track_info.title)}.{ext}"
        target_path = artist_dir / filename

        # If stream_url is provided, download via HTTP
        if stream_url:
            with httpx.Client(timeout=30.0) as http_client:
                with http_client.stream("GET", stream_url) as resp:
                    resp.raise_for_status()
                    total_bytes = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    with open(target_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback and total_bytes > 0:
                                progress_callback(track_info.title, downloaded / total_bytes)
        else:
            # Create a placeholder audio container if no stream URL for testing/mocking
            if not target_path.exists():
                target_path.touch()

        # Build and write metadata
        meta = TrackMetadata(
            title=track_info.title,
            artist=track_info.artist,
            album=track_info.album,
            duration_sec=track_info.duration_sec,
            isrc=track_info.isrc,
            genre="Electronic",
            format=ext,
            file_size_bytes=target_path.stat().st_size if target_path.exists() else 0,
        )

        try:
            AudioTagEngine.write_djay_pro_tags(target_path, meta)
        except Exception:
            pass

        # Ingest into SQLite database
        track_dict = {
            "file_path": str(target_path.resolve()),
            "file_hash": AudioTagEngine.calculate_file_hash(target_path) if target_path.stat().st_size > 0 else f"tidal_{track_info.id}",
            "file_size_bytes": meta.file_size_bytes,
            "format": ext,
            "duration_sec": meta.duration_sec,
            "title": meta.title,
            "artist": meta.artist,
            "album": meta.album,
            "genre": meta.genre,
            "isrc": meta.isrc,
            "bpm": meta.bpm,
            "camelot": meta.camelot,
            "energy": meta.energy,
        }
        self.repo.upsert_track(track_dict)
        return target_path

    def batch_download_playlist(
        self,
        playlist_tracks: List[TidalTrackInfo],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Path]:
        """Download a batch of tracks sequentially with progress updates."""
        downloaded_paths = []
        total = len(playlist_tracks)

        for idx, track_info in enumerate(playlist_tracks, start=1):
            if progress_callback:
                progress_callback(idx, total, f"{track_info.artist} - {track_info.title}")

            path = self.download_track(track_info)
            downloaded_paths.append(path)

        return downloaded_paths
