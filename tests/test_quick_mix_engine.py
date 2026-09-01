from pathlib import Path
import numpy as np
import soundfile as sf
from sonicdj.sandbox.quick_mix_engine import QuickMixAuditionEngine


def test_quick_mix_audition_rendering(tmp_path):
    sr = 22050
    t = np.linspace(0, 15, sr * 15, endpoint=False)
    
    # Track A: 120 BPM tone
    sine_a = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    path_a = tmp_path / "track_a.wav"
    sf.write(str(path_a), sine_a, sr, format="WAV")

    # Track B: 124 BPM tone
    sine_b = (0.5 * np.sin(2 * np.pi * 523.25 * t)).astype(np.float32)
    path_b = tmp_path / "track_b.wav"
    sf.write(str(path_b), sine_b, sr, format="WAV")

    out_mix = tmp_path / "rendered_mix.wav"

    mix_audio, out_sr = QuickMixAuditionEngine.render_16bar_audition(
        path_a, path_b, output_wav_path=out_mix, num_bars=8, sr=sr
    )

    assert out_sr == sr
    assert len(mix_audio) > 0
    assert out_mix.exists()
    assert out_mix.stat().st_size > 1000
    assert np.max(np.abs(mix_audio)) > 0.1
