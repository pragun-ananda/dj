from pathlib import Path
from typing import Callable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

from sonicdj.config import settings
from sonicdj.analysis.audio_loader import AudioLoader
from sonicdj.analysis.key_detector import KeyDetector, KeyAnalysisResult
from sonicdj.analysis.phrasing_engine import PhrasingEngine, PhrasingAnalysisResult
from sonicdj.analysis.vocal_energy import VocalEnergyProfiler, VocalEnergyResult
from sonicdj.analysis.cue_generator import CueGenerator, CompleteTrackAnalysis
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.metadata.models import TrackMetadata
from sonicdj.db.repository import TrackRepository, DatabaseManager
from sonicdj.db.vector_store import VectorStore
from sonicdj.embeddings.audio_encoder import MultimodalAudioEncoder


@dataclass
class BatchAnalysisStats:
    total_found: int = 0
    analyzed: int = 0
    failed: int = 0
    tagged: int = 0


class AudioAnalyzer:
    """
    Orchestrates end-to-end MIR analysis (Key, Camelot, BPM, Phrasing, Energy, Vocals, Auto-Cues)
    and handles multi-threaded batch library enrichment.
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db
        self.repo = TrackRepository(db) if db else None
        self.vector_store = VectorStore(db) if db else None

    def analyze_file(
        self,
        file_path: Path,
        target_sr: int = 22050,
        max_duration_sec: Optional[float] = 600.0,
    ) -> CompleteTrackAnalysis:
        """
        Runs deep MIR analysis on a single audio file.
        """
        file_path = Path(file_path).resolve()
        audio, sr = AudioLoader.load_audio(
            file_path, target_sr=target_sr, max_duration_sec=max_duration_sec
        )

        # 1. Harmonic Key & Camelot
        key_info = KeyDetector.detect_key(audio, sr=sr)

        # 2. BPM, Beatgrid & Phrasing
        phrasing_info = PhrasingEngine.analyze_phrasing(audio, sr=sr)

        # 3. Energy Curve & Vocal Timeline
        vocal_energy_info = VocalEnergyProfiler.detect_vocal_activity(audio, sr=sr)

        # 4. Synthesize DJ Cue Points
        cues = CueGenerator.generate_cues(phrasing_info, vocal_energy_info, key_info)
        summary_comment = CueGenerator.generate_summary_comment(
            key_info, phrasing_info, vocal_energy_info, cues
        )

        return CompleteTrackAnalysis(
            key_info=key_info,
            phrasing_info=phrasing_info,
            vocal_energy_info=vocal_energy_info,
            generated_cues=cues,
            summary_comment=summary_comment,
        )

    def analyze_and_enrich_track(
        self,
        file_path: Path,
        auto_tag_file: bool = True,
    ) -> Tuple[Path, CompleteTrackAnalysis]:
        """
        Analyzes a track, updates the SQLite database, and optionally writes ID3 tags to disk.
        """
        file_path = Path(file_path).resolve()
        analysis = self.analyze_file(file_path)

        # Read existing metadata or construct new
        meta = AudioTagEngine.read_metadata(file_path)
        meta.bpm = analysis.phrasing_info.bpm
        meta.camelot = analysis.key_info.camelot
        meta.key_raw = analysis.key_info.musical_key
        meta.energy = analysis.vocal_energy_info.overall_energy
        meta.comments = analysis.summary_comment
        meta.cues = analysis.generated_cues

        # 1. Write tags to physical audio file for djay Pro
        if auto_tag_file:
            try:
                AudioTagEngine.write_djay_pro_tags(file_path, meta)
            except Exception:
                pass

        # 2. Update database record
        if self.repo:
            cue_dicts = [
                {
                    "name": c.name,
                    "timestamp_ms": c.timestamp_ms,
                    "cue_type": c.cue_type,
                    "hot_cue_index": c.hot_cue_index,
                    "color_hex": c.color_hex,
                }
                for c in analysis.generated_cues
            ]
            track_dict = {
                "file_path": str(file_path),
                "file_hash": meta.file_hash,
                "file_size_bytes": meta.file_size_bytes,
                "format": meta.format,
                "duration_sec": meta.duration_sec,
                "title": meta.title,
                "artist": meta.artist,
                "album": meta.album,
                "genre": meta.genre,
                "bpm": meta.bpm,
                "camelot": meta.camelot,
                "key_raw": meta.key_raw,
                "energy": meta.energy,
                "comments": meta.comments,
                "analyzed_at": datetime.utcnow(),
            }
            db_track = self.repo.upsert_track(track_dict, cues=cue_dicts)

            if self.vector_store and db_track:
                try:
                    audio, sr = AudioLoader.load_audio(file_path, target_sr=22050, max_duration_sec=120.0)
                    emb_vec = MultimodalAudioEncoder.encode_audio(
                        audio,
                        sr=sr,
                        bpm=meta.bpm,
                        camelot=meta.camelot,
                        energy=meta.energy,
                        has_vocals=analysis.vocal_energy_info.has_vocals,
                    )
                    self.vector_store.upsert_embedding(db_track.id, emb_vec)
                except Exception:
                    pass

        return file_path, analysis

    def batch_analyze_directory(
        self,
        directory: Path,
        auto_tag_file: bool = True,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[Path, BatchAnalysisStats], None]] = None,
    ) -> BatchAnalysisStats:
        """
        Analyzes all audio files in a directory using a multi-threaded worker pool.
        """
        from sonicdj.scanner.file_scanner import LibraryScanner
        scanner = LibraryScanner(self.db or DatabaseManager())
        files_to_analyze = list(scanner.find_audio_files(directory))

        stats = BatchAnalysisStats(total_found=len(files_to_analyze))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.analyze_and_enrich_track, f, auto_tag_file): f
                for f in files_to_analyze
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    future.result()
                    stats.analyzed += 1
                    if auto_tag_file:
                        stats.tagged += 1
                except Exception:
                    stats.failed += 1

                if progress_callback:
                    progress_callback(file_path, stats)

        return stats
