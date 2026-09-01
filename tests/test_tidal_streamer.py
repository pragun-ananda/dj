import numpy as np
import pytest
from sonicdj.streaming.tidal_streamer import TidalStreamer
from sonicdj.ingestion.tidal_client import TidalClient


def test_tidal_streamer_stream_url():
    client = TidalClient()
    streamer = TidalStreamer(client)

    url = streamer.get_stream_url("test_track_123")
    assert url is not None
    assert "tidal" in url or "mock" in url or "example.com" in url


def test_tidal_streamer_stream_chunks():
    streamer = TidalStreamer()
    chunks = list(streamer.stream_audio_chunks("mock://tidal.stream/123"))
    assert len(chunks) >= 1
    assert len(chunks[0]) > 0


def test_tidal_streamer_stream_and_analyze():
    streamer = TidalStreamer()
    
    # Synthetic test audio stream
    sr = 22050
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    analysis = streamer.stream_and_analyze("track_456", max_duration_sec=10.0, synthetic_audio=sine)
    assert analysis.key_info.camelot != ""
    assert analysis.phrasing_info.bpm > 0
    assert len(analysis.generated_cues) >= 1
