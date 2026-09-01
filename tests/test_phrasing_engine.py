import numpy as np
from sonicdj.analysis.phrasing_engine import PhrasingEngine


def test_phrasing_engine_rhythm():
    sr = 22050
    duration_sec = 60.0
    total_samples = int(sr * duration_sec)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Generate 120 BPM 4/4 Kick pulses (2 beats per second = 0.5s interval)
    beat_interval_samples = int(sr * 0.5)
    for i in range(0, total_samples, beat_interval_samples):
        # 50ms burst of 60Hz sub-bass kick
        kick_len = min(int(sr * 0.05), total_samples - i)
        t_kick = np.linspace(0, 0.05, kick_len, endpoint=False)
        audio[i : i + kick_len] += 0.8 * np.sin(2 * np.pi * 60.0 * t_kick)

    res = PhrasingEngine.analyze_phrasing(audio, sr=sr)
    assert 115.0 <= res.bpm <= 125.0
    assert res.confidence > 0.3
    assert len(res.beat_timestamps) >= 100
    assert len(res.downbeat_timestamps) >= 25
    assert len(res.sections) >= 1


def test_phrasing_engine_short_audio():
    res = PhrasingEngine.analyze_phrasing(np.zeros(100, dtype=np.float32))
    assert res.bpm == 120.0
    assert res.confidence == 0.0
    assert len(res.sections) == 0
