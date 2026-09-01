import numpy as np
from sonicdj.analysis.key_detector import KeyDetector


def test_key_detector_synthetic_tones():
    sr = 22050
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # 1. Generate A Minor Chord (A4=440Hz, C5=523.25Hz, E5=659.25Hz)
    a_minor = (
        0.4 * np.sin(2 * np.pi * 440.0 * t) +
        0.3 * np.sin(2 * np.pi * 523.25 * t) +
        0.3 * np.sin(2 * np.pi * 659.25 * t)
    ).astype(np.float32)

    res = KeyDetector.detect_key(a_minor, sr=sr)
    assert res.camelot in ("8A", "9A", "7A", "8B")  # Closely related harmonic matches
    assert res.confidence > 0.3
    assert len(res.chroma_profile) == 12
    assert abs(res.pitch_drift_cents) < 15.0


def test_key_detector_pitch_drift_432hz():
    sr = 22050
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # A4 tuned to 432 Hz instead of 440 Hz (approx -31.8 cents)
    tone_432 = (0.5 * np.sin(2 * np.pi * 432.0 * t)).astype(np.float32)
    res = KeyDetector.detect_key(tone_432, sr=sr)
    assert res.tuning_hz < 438.0  # Drift detected downwards
    assert res.pitch_drift_cents < -10.0


def test_key_detector_empty_and_short():
    res = KeyDetector.detect_key(np.array([]))
    assert res.musical_key == "Unknown"
    assert res.confidence == 0.0

    res_short = KeyDetector.detect_key(np.zeros(100, dtype=np.float32))
    assert len(res_short.chroma_profile) == 12
