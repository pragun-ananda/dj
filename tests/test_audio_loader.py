import wave
import struct
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

from sonicdj.analysis.audio_loader import AudioLoader


def test_audio_loader_valid_wav(tmp_path):
    wav_path = tmp_path / "test.wav"
    sr = 44100
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 440 Hz Sine wave
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.column_stack([sine, sine])
    sf.write(str(wav_path), stereo, sr, format="WAV")

    # Load with downsampling to 22050
    audio, out_sr = AudioLoader.load_audio(wav_path, target_sr=22050, mono=True)
    assert out_sr == 22050
    assert audio.ndim == 1
    assert len(audio) == 22050
    assert np.max(np.abs(audio)) > 0.5


def test_audio_loader_max_duration(tmp_path):
    wav_path = tmp_path / "test_long.wav"
    sr = 22050
    data = np.zeros((sr * 5, 1), dtype=np.float32)
    sf.write(str(wav_path), data, sr, format="WAV")

    audio, out_sr = AudioLoader.load_audio(wav_path, target_sr=22050, max_duration_sec=2.0)
    assert len(audio) == sr * 2


def test_audio_loader_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        AudioLoader.load_audio(tmp_path / "non_existent.wav")


def test_audio_loader_wave_module_fallback(tmp_path):
    wav_path = tmp_path / "raw_wave.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        data = struct.pack("<h", 1000) * 22050 * 2
        wf.writeframes(data)

    from unittest.mock import patch
    with patch("soundfile.read", side_effect=Exception("Soundfile decode error")):
        audio, sr = AudioLoader.load_audio(wav_path, target_sr=22050)
        assert sr == 22050
        assert len(audio) == 22050


def test_audio_loader_unreadable_fallback(tmp_path):
    dummy_file = tmp_path / "corrupt_audio.wav"
    dummy_file.write_bytes(b"NOT_A_VALID_AUDIO_FILE")

    audio, sr = AudioLoader.load_audio(dummy_file, target_sr=22050)
    assert sr == 22050
    assert len(audio) > 0
