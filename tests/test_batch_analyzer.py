from pathlib import Path
import wave
import struct
import numpy as np
import soundfile as sf
from sonicdj.analysis.batch_analyzer import AudioAnalyzer
from sonicdj.db.repository import TrackRepository, DatabaseManager


def test_audio_analyzer_single_and_batch(temp_db, tmp_path):
    analyzer = AudioAnalyzer(temp_db)
    repo = TrackRepository(temp_db)

    # 1. Create 2 test WAV files with sound
    sr = 22050
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    f1 = tmp_path / "track_alpha.wav"
    f2 = tmp_path / "track_beta.wav"
    sf.write(str(f1), sine, sr, format="WAV")
    sf.write(str(f2), sine, sr, format="WAV")

    # 2. Test analyze_file directly
    res = analyzer.analyze_file(f1)
    assert res.key_info.camelot != ""
    assert res.phrasing_info.bpm > 0
    assert len(res.generated_cues) >= 1

    # 3. Test analyze_and_enrich_track
    analyzer.analyze_and_enrich_track(f1, auto_tag_file=True)
    db_track = repo.get_track_by_path(str(f1))
    assert db_track is not None
    assert db_track.bpm > 0
    assert db_track.camelot != ""
    assert len(db_track.cues) >= 1

    # 4. Test batch_analyze_directory with progress callback
    progress_records = []
    def on_prog(f, stats):
        progress_records.append((f.name, stats.analyzed))

    stats = analyzer.batch_analyze_directory(
        tmp_path, auto_tag_file=True, max_workers=2, progress_callback=on_prog
    )
    assert stats.total_found == 2
    assert stats.analyzed == 2
    assert stats.failed == 0
    assert len(progress_records) == 2
