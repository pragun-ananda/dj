import wave
import struct
import tempfile
from pathlib import Path
import pytest
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TBPM, TKEY, TCON

from sonicdj.db.repository import DatabaseManager, TrackRepository, PlaylistRepository


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = DatabaseManager(f"sqlite:///{db_path}")
    yield db
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_audio_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        
        # Create a valid minimal WAV file
        wav_path = dir_path / "track_test.wav"
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            # Write 1 second of silence
            data = struct.pack("<h", 0) * (44100 * 2)
            wav_file.writeframes(data)

        # Create a second dummy file for multi-file scan
        wav_path2 = dir_path / "subfolder" / "track_afrohouse.wav"
        wav_path2.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path2), "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            data = struct.pack("<h", 0) * (44100 * 2)
            wav_file.writeframes(data)

        yield dir_path
