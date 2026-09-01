import numpy as np
import soundfile as sf
from typer.testing import CliRunner
from sonicdj.cli import app
from sonicdj.db.repository import TrackRepository

runner = CliRunner()


def test_cli_mix_path_and_audition(temp_db, tmp_path):
    repo = TrackRepository(temp_db)
    db_file = temp_db.db_url.replace("sqlite:///", "")

    t1 = repo.upsert_track({"file_path": "/m/1.wav", "file_hash": "1", "file_size_bytes": 10, "format": "wav", "duration_sec": 180, "title": "Track 1", "artist": "A", "bpm": 120.0, "camelot": "8A", "energy": 0.6})
    t2 = repo.upsert_track({"file_path": "/m/2.wav", "file_hash": "2", "file_size_bytes": 10, "format": "wav", "duration_sec": 180, "title": "Track 2", "artist": "B", "bpm": 124.0, "camelot": "9A", "energy": 0.8})

    # 1. Test mix-path command
    res_path = runner.invoke(app, ["mix-path", str(t1.id), str(t2.id), "--db", db_file])
    assert res_path.exit_code == 0
    assert "Harmonic DJ Setlist Path" in res_path.output
    assert "Track 1" in res_path.output
    assert "Track 2" in res_path.output

    # 2. Test audition command
    sr = 22050
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    f_a = tmp_path / "a.wav"
    f_b = tmp_path / "b.wav"
    sf.write(str(f_a), sine, sr, format="WAV")
    sf.write(str(f_b), sine, sr, format="WAV")

    out_file = tmp_path / "out.wav"
    res_audition = runner.invoke(app, ["audition", str(f_a), str(f_b), "--bars", "8", "--out", str(out_file)])
    assert res_audition.exit_code == 0
    assert "Rendered" in res_audition.output
    assert out_file.exists()
