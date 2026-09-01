import io
import tempfile
from pathlib import Path
from typing import Generator, Optional, Tuple, Dict, Any
import httpx
import numpy as np

from sonicdj.ingestion.tidal_client import TidalClient
from sonicdj.analysis.audio_loader import AudioLoader
from sonicdj.analysis.key_detector import KeyDetector
from sonicdj.analysis.phrasing_engine import PhrasingEngine
from sonicdj.analysis.vocal_energy import VocalEnergyProfiler
from sonicdj.analysis.cue_generator import CueGenerator, CompleteTrackAnalysis
from sonicdj.embeddings.audio_encoder import MultimodalAudioEncoder


class TidalStreamer:
    """
    Handles in-app streaming resolution, progressive audio chunk buffering,
    and instantaneous 'Stream-to-Analyze' preview extraction.
    """

    def __init__(self, client: Optional[TidalClient] = None):
        self.client = client or TidalClient()

    def get_stream_url(self, track_id: str) -> Optional[str]:
        """Resolves direct streaming URL for a Tidal track ID."""
        info = self.client.get_playback_info(track_id)
        if not info:
            return None
        return info.get("manifest_url") or info.get("url")

    def stream_audio_chunks(
        self, stream_url: str, chunk_size: int = 65536
    ) -> Generator[bytes, None, None]:
        """Yields progressive audio stream chunks for live WebAudio playback."""
        if stream_url.startswith("mock://") or "example.com" in stream_url:
            # Synthetic streaming simulation
            yield b"\x00" * 4096
            return

        with httpx.Client(timeout=15.0) as client:
            with client.stream("GET", stream_url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    yield chunk

    def stream_and_analyze(
        self,
        track_id: str,
        max_duration_sec: float = 60.0,
        synthetic_audio: Optional[np.ndarray] = None,
    ) -> CompleteTrackAnalysis:
        """
        Streams a preview buffer and runs instant deep MIR analysis (Key, Camelot, BPM, Vocals, Cues)
        without needing to download or save the full file to disk first.
        """
        if synthetic_audio is not None:
            audio = synthetic_audio
            sr = 22050
        else:
            stream_url = self.get_stream_url(track_id)
            if not stream_url:
                raise ValueError(f"Could not resolve stream URL for Tidal track: {track_id}")

            # Stream into memory buffer
            buffer = io.BytesIO()
            total_bytes = 0
            max_bytes = int(max_duration_sec * 44100 * 2 * 2)  # ~60s of 16-bit 44.1kHz stereo

            for chunk in self.stream_audio_chunks(stream_url):
                buffer.write(chunk)
                total_bytes += len(chunk)
                if total_bytes >= max_bytes:
                    break

            buffer.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
                tmp.write(buffer.getvalue())
                tmp_path = Path(tmp.name)

            try:
                audio, sr = AudioLoader.load_audio(tmp_path, target_sr=22050, max_duration_sec=max_duration_sec)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        # Run MIR Analysis on in-memory stream buffer
        key_info = KeyDetector.detect_key(audio, sr=sr)
        phrasing_info = PhrasingEngine.analyze_phrasing(audio, sr=sr)
        vocal_energy_info = VocalEnergyProfiler.detect_vocal_activity(audio, sr=sr)
        cues = CueGenerator.generate_cues(phrasing_info, vocal_energy_info, key_info)
        summary_comment = CueGenerator.generate_summary_comment(key_info, phrasing_info, vocal_energy_info, cues)

        return CompleteTrackAnalysis(
            key_info=key_info,
            phrasing_info=phrasing_info,
            vocal_energy_info=vocal_energy_info,
            generated_cues=cues,
            summary_comment=summary_comment,
        )
