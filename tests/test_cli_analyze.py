from pathlib import Path
import numpy as np
import soundfile as sf
from typer.testing import CliRunner
from sonicdj.cli import app

runner = CliRunner()


def test_cli_analyze_single_file_and_directory(tmp_path):
    # Create test wav
    sr = 22050
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_file = tmp_path / "dance_track.wav"
    sf.write(str(wav_file), sine, sr, format="WAV")

    # 1. Analyze single file
    res_single = runner.invoke(app, ["analyze", str(wav_file)])
    assert res_single.exit_code == 0
    assert "Analyzing Single Track:" in res_single.output
    assert "Camelot Key" in res_single.output
    assert "Generated djay Pro Hot Cues" in res_single.output

    # 2. Analyze directory
    res_dir = runner.invoke(app, ["analyze", str(tmp_path), "--workers", "2"])
    assert res_dir.exit_code == 0
    assert "Batch Analysis Summary" in res_dir.output

    # 3. Analyze invalid path
    res_invalid = runner.invoke(app, ["analyze", "/non/existent/audio.wav"])
    assert res_invalid.exit_code == 1
