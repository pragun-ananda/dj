import numpy as np
from sonicdj.analysis.vocal_energy import VocalEnergyProfiler


def test_vocal_energy_profiler():
    sr = 22050
    duration_sec = 20.0
    total_samples = int(sr * duration_sec)
    audio = np.zeros(total_samples, dtype=np.float32)

    # 1. 0s to 5s: Low energy sub-bass (instrumental intro)
    t1 = np.linspace(0, 5, sr * 5, endpoint=False)
    audio[: sr * 5] = 0.2 * np.sin(2 * np.pi * 60 * t1)

    # 2. 5s to 15s: Strong vocal formant frequencies (800 Hz + 1500 Hz)
    t2 = np.linspace(0, 10, sr * 10, endpoint=False)
    vocal_signal = 0.6 * np.sin(2 * np.pi * 800 * t2) + 0.4 * np.sin(2 * np.pi * 1500 * t2)
    audio[sr * 5 : sr * 15] = vocal_signal

    # 3. 15s to 20s: Instrumental outro
    t3 = np.linspace(0, 5, sr * 5, endpoint=False)
    audio[sr * 15 :] = 0.2 * np.sin(2 * np.pi * 60 * t3)

    res = VocalEnergyProfiler.detect_vocal_activity(audio, sr=sr)
    assert res.overall_energy > 0.1
    assert len(res.energy_curve) >= 19
    assert res.has_vocals is True
    assert res.vocal_presence_percent > 20.0
    assert res.first_vocal_sec is not None
    assert 3.5 <= res.first_vocal_sec <= 7.0
    assert res.instrumental_intro_sec >= 3.0


def test_vocal_energy_short_or_silent():
    res = VocalEnergyProfiler.detect_vocal_activity(np.zeros(100, dtype=np.float32))
    assert res.has_vocals is False
    assert res.vocal_presence_percent == 0.0
