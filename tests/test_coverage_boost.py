import json
import wave
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import soundfile as sf
import numpy as np
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TCON, TBPM, TKEY, TDRC, POPM, COMM, TXXX
from mutagen.flac import FLAC
from typer.testing import CliRunner

from sonicdj.config import settings, Settings
from sonicdj.db.schema import Base, Track, CuePoint, Playlist, PlaylistTrack, IngestionJob
from sonicdj.db.repository import DatabaseManager, TrackRepository, PlaylistRepository, get_db
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.metadata.models import TrackMetadata, CueMetadata, normalize_key_to_camelot
from sonicdj.metadata.djay_exporter import DjayProExporter
from sonicdj.scanner.file_scanner import LibraryScanner, ScanStats
from sonicdj.ingestion.tidal_client import TidalClient, TidalTrackInfo
from sonicdj.ingestion.downloader import TidalDownloader
from sonicdj.cli import app

runner = CliRunner()


# -------------------------------------------------------------
# 1. Config & Schema Tests
# -------------------------------------------------------------
def test_config_methods(tmp_path):
    custom_settings = Settings(app_dir=tmp_path / "app", music_dir=tmp_path / "music")
    custom_settings.ensure_directories()
    assert (tmp_path / "app").exists()
    assert (tmp_path / "music").exists()
    assert "sqlite:///" in custom_settings.db_url
    assert custom_settings.db_path == tmp_path / "app" / "library.db"


def test_schema_and_track_methods():
    track = Track(
        file_path="/test/path.mp3",
        file_hash="hash123",
        title="Test Title",
        artist="Test Artist",
        album="Test Album",
        genre="Afro House",
        bpm=124.0,
        camelot="8A",
        key_raw="Am",
        energy=0.85,
        rating=5,
        duration_sec=300.0,
    )
    # Test subgenres getter & setter
    track.subgenres = ["Afro House", "Deep Tech"]
    assert track.subgenres == ["Afro House", "Deep Tech"]
    
    # Test subgenres with invalid JSON
    track.subgenres_json = "INVALID_JSON{"
    assert track.subgenres == []

    # Test to_dict
    d = track.to_dict()
    assert d["title"] == "Test Title"
    assert d["artist"] == "Test Artist"
    assert d["bpm"] == 124.0
    assert d["camelot"] == "8A"
    assert d["cues_count"] == 0


# -------------------------------------------------------------
# 2. Database & Repository Edge Cases
# -------------------------------------------------------------
def test_repository_edge_cases(temp_db):
    repo = TrackRepository(temp_db)
    pl_repo = PlaylistRepository(temp_db)

    # 1. Test session_scope rollback on exception
    with pytest.raises(ValueError):
        with temp_db.session_scope() as session:
            session.add(Track(file_path="/bad/path.mp3", file_hash="hash_bad"))
            raise ValueError("Forced error for rollback")

    # Verify track was rolled back
    assert repo.get_track_by_path("/bad/path.mp3") is None

    # 2. Add track and test get_track_by_path & count_all
    t = repo.upsert_track({"file_path": "/good/path.mp3", "file_hash": "hash_good", "title": "Good Track"})
    assert repo.get_track_by_path("/good/path.mp3") is not None
    assert repo.count_all() == 1

    # 3. Create playlist and update existing playlist
    pl1 = pl_repo.create_playlist("Crate 1", description="Initial")
    assert pl1.description == "Initial"
    pl1_updated = pl_repo.create_playlist("Crate 1", description="Updated Description")
    assert pl1_updated.description == "Updated Description"

    # 4. Add track with explicit position
    pl_repo.add_track_to_playlist(pl1.id, t.id, position=5)
    tracks = pl_repo.get_playlist_tracks(pl1.id)
    assert len(tracks) == 1

    # 5. Test global get_db helper
    db_inst = get_db(temp_db.db_url)
    assert db_inst is not None


# -------------------------------------------------------------
# 3. Comprehensive Multi-Format Tag Writing & Reading
# -------------------------------------------------------------
def test_id3_engine_mp3_writing_and_reading(tmp_path):
    mp3_file = tmp_path / "test_track.mp3"
    # Create minimal mp3 file with valid frame
    with open(mp3_file, "wb") as f:
        f.write(b"\xff\xfb\x90\x04" + b"\x00" * 414)

    meta = TrackMetadata(
        title="Solar Groove",
        artist="DJ Pulse",
        album="Sunrise EP",
        genre="Afro House",
        year=2024,
        bpm=123.0,
        camelot="8A",
        key_raw="Am",
        energy=0.88,
        rating=5,
        comments="Vocal transition marker",
        cues=[CueMetadata(name="Intro", timestamp_ms=0), CueMetadata(name="Drop", timestamp_ms=32000)],
    )

    # Write tags
    assert AudioTagEngine.write_djay_pro_tags(mp3_file, meta) is True

    # Read back and verify
    read_meta = AudioTagEngine.read_metadata(mp3_file)
    assert read_meta.title == "Solar Groove"
    assert read_meta.artist == "DJ Pulse"
    assert read_meta.album == "Sunrise EP"
    assert read_meta.genre == "Afro House"
    assert read_meta.year == 2024
    assert read_meta.bpm == 123.0
    assert read_meta.camelot == "8A"
    assert read_meta.rating == 5


def test_id3_engine_flac_writing_and_reading(tmp_path):
    flac_file = tmp_path / "test_track.flac"
    data = np.zeros((4410, 2), dtype=np.float32)
    sf.write(str(flac_file), data, 44100, format="FLAC")

    meta = TrackMetadata(
        title="Desert Mirage",
        artist="THEMBA",
        album="Deep Roots",
        genre="Afro House",
        year=2023,
        bpm=124.0,
        camelot="9A",
        key_raw="Em",
        energy=0.91,
        rating=4,
        comments="Main drop at 1:15",
        cues=[CueMetadata(name="Intro", timestamp_ms=0)],
    )

    assert AudioTagEngine.write_djay_pro_tags(flac_file, meta) is True

    read_meta = AudioTagEngine.read_metadata(flac_file)
    assert read_meta.title == "Desert Mirage"
    assert read_meta.artist == "THEMBA"
    assert read_meta.year == 2023
    assert read_meta.bpm == 124.0
    assert read_meta.camelot == "9A"
    assert read_meta.rating == 4


def test_id3_engine_popm_ratings(tmp_path):
    # Test all rating levels 1-5 for POPM
    for rating_val in [1, 2, 3, 4, 5]:
        mp3_file = tmp_path / f"test_rating_{rating_val}.mp3"
        with open(mp3_file, "wb") as f:
            f.write(b"\xff\xfb\x90\x04" + b"\x00" * 414)
        
        meta = TrackMetadata(title=f"Rating {rating_val}", rating=rating_val)
        AudioTagEngine.write_djay_pro_tags(mp3_file, meta)
        read_meta = AudioTagEngine.read_metadata(mp3_file)
        assert read_meta.rating == rating_val


def test_id3_engine_errors_and_fallbacks(tmp_path):
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        AudioTagEngine.read_metadata(tmp_path / "non_existent.mp3")

    with pytest.raises(FileNotFoundError):
        AudioTagEngine.write_djay_pro_tags(tmp_path / "non_existent.mp3", TrackMetadata())

    # Unsupported extension
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Hello")
    assert AudioTagEngine.write_djay_pro_tags(txt_file, TrackMetadata()) is False

    meta = AudioTagEngine.read_metadata(txt_file)
    assert meta.format == "txt"
    assert meta.title == "notes"


# -------------------------------------------------------------
# 4. Scanner Edge Cases
# -------------------------------------------------------------
def test_scanner_edge_cases(temp_db, tmp_path):
    scanner = LibraryScanner(temp_db)

    # 1. Non-existent path
    assert list(scanner.find_audio_files(tmp_path / "non_existent_dir")) == []

    # 2. Single file passed directly
    single_wav = tmp_path / "single.wav"
    with wave.open(str(single_wav), "wb") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(44100)
        f.writeframes(struct.pack("<h", 0) * 100)

    found = list(scanner.find_audio_files(single_wav))
    assert len(found) == 1
    assert found[0] == single_wav

    # 3. Non-audio file passed directly
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("not audio")
    assert list(scanner.find_audio_files(txt_file)) == []

    # 4. Scan with corrupt / unreadable audio file error handling
    corrupt_mp3 = tmp_path / "corrupt.mp3"
    corrupt_mp3.write_bytes(b"INVALID_HEADER_GARBAGE")

    with patch.object(AudioTagEngine, "read_metadata", side_effect=Exception("Read error")):
        stats = scanner.scan_directory(tmp_path)
        assert stats.failed > 0


# -------------------------------------------------------------
# 5. Tidal Client & Downloader Mock HTTP Responses
# -------------------------------------------------------------
def test_tidal_client_mock_search_and_playlists(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({
        "access_token": "valid_token",
        "refresh_token": "refresh",
        "user_id": "12345",
        "country_code": "US",
    }))

    client = TidalClient(token_file=token_file)
    assert client.is_authenticated

    mock_search_json = {
        "items": [
            {
                "id": 1234567,
                "title": "Afro Tribe",
                "artist": {"name": "Kelela"},
                "album": {"title": "Solaris", "cover": "abc-123-def"},
                "duration": 345,
                "isrc": "US12345678",
                "streamStartDate": "2024-01-01",
                "audioQuality": "LOSSLESS",
            }
        ]
    }

    # Test search_tracks HTTP 200
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_search_json

    with patch("httpx.Client.get", return_value=mock_resp):
        tracks = client.search_tracks("Afro Tribe")
        assert len(tracks) == 1
        assert tracks[0].title == "Afro Tribe"
        assert tracks[0].artist == "Kelela"
        assert "640x640.jpg" in str(tracks[0].cover_url)

    # Test search_tracks HTTP 401
    mock_err_resp = MagicMock()
    mock_err_resp.status_code = 401
    with patch("httpx.Client.get", return_value=mock_err_resp):
        assert client.search_tracks("Afro Tribe") == []

    # Test get_playlist_tracks HTTP 200
    with patch("httpx.Client.get", return_value=mock_resp):
        pl_tracks = client.get_playlist_tracks("playlist-uuid")
        assert len(pl_tracks) == 1
        assert pl_tracks[0].title == "Afro Tribe"

    # Test get_playlist_tracks HTTP 500
    with patch("httpx.Client.get", return_value=mock_err_resp):
        assert client.get_playlist_tracks("playlist-uuid") == []


def test_tidal_downloader_with_streaming_mock(temp_db, tmp_path):
    downloader = TidalDownloader(db=temp_db, output_dir=tmp_path)
    track_info = TidalTrackInfo(
        id="999",
        title="Streaming Track",
        artist="Audio Pro",
        album="Stream Album",
        duration_sec=200.0,
        audio_quality="LOSSLESS",
    )

    # Mock httpx streaming response
    mock_stream_resp = MagicMock()
    mock_stream_resp.headers = {"content-length": "100"}
    mock_stream_resp.iter_bytes.return_value = [b"chunk1", b"chunk2"]

    progress_calls = []
    def on_prog(title, ratio):
        progress_calls.append((title, ratio))

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.stream.return_value.__enter__.return_value = mock_stream_resp

    with patch("httpx.Client", return_value=mock_client):
        out_path = downloader.download_track(
            track_info, stream_url="https://audio.tidal.com/stream.flac", progress_callback=on_prog
        )
        assert out_path.exists()
        assert len(progress_calls) >= 1


# -------------------------------------------------------------
# 6. djay Pro Exporter with Playlists
# -------------------------------------------------------------
def test_djay_exporter_with_playlists(tmp_path):
    t1 = Track(id=1, file_path="/Music/Track1.mp3", title="Song 1", format="mp3")
    t2 = Track(id=2, file_path="/Music/Track2.flac", title="Song 2", format="flac")
    
    pl = Playlist(id=1, name="Warmup Crate")
    pl.tracks = [
        PlaylistTrack(playlist_id=1, track_id=1, position=1),
        PlaylistTrack(playlist_id=1, track_id=2, position=2),
    ]

    xml_out = tmp_path / "rekordbox_playlists.xml"
    DjayProExporter.export_rekordbox_xml([t1, t2], xml_out, playlists=[pl])
    assert xml_out.exists()
    content = xml_out.read_text()
    assert "Warmup Crate" in content
    assert 'Name="Song 1"' in content


# -------------------------------------------------------------
# 7. CLI Edge Cases (Empty library, errors)
# -------------------------------------------------------------
def test_cli_error_paths(tmp_path):
    # 1. Scan on invalid directory
    res1 = runner.invoke(app, ["scan", "/non/existent/path/xyz"])
    assert res1.exit_code == 1

    # 2. Tag on invalid file
    res2 = runner.invoke(app, ["tag", "/non/existent/track.mp3", "--bpm", "120"])
    assert res2.exit_code == 1

    # 3. List with empty library / no matching search
    res3 = runner.invoke(app, ["list", "--search", "NONEXISTENT_QUERY_12345"])
    assert "No tracks found" in res3.output

    # 4. Export with empty library
    empty_db = tmp_path / "empty.db"
    mock_db = DatabaseManager(f"sqlite:///{empty_db}")
    with patch("sonicdj.cli.get_db", return_value=mock_db):
        res4 = runner.invoke(app, ["export-djay", str(tmp_path / "out.xml")])
        assert "Library is empty" in res4.output


# -------------------------------------------------------------
# 8. Remaining Edge Cases & 100% Target Booster
# -------------------------------------------------------------
def test_mp4_writing_with_mock(tmp_path):
    mp4_file = tmp_path / "song.m4a"
    mp4_file.touch()

    mock_mp4_inst = MagicMock()
    mock_mp4_inst.tags = {}

    with patch("sonicdj.metadata.id3_engine.MP4", return_value=mock_mp4_inst):
        meta = TrackMetadata(
            title="Afro Tech Anthem",
            artist="DJ Master",
            album="Summer 2024",
            genre="Afro Tech",
            year=2024,
            bpm=124.0,
            camelot="8A",
            energy=0.9,
            comments="Cue note",
        )
        success = AudioTagEngine.write_djay_pro_tags(mp4_file, meta)
        assert success is True
        assert mock_mp4_inst.tags["\xa9nam"] == ["Afro Tech Anthem"]
        assert mock_mp4_inst.tags["tmpo"] == [124]


def test_repository_filtering_and_auto_position(temp_db):
    repo = TrackRepository(temp_db)
    pl_repo = PlaylistRepository(temp_db)

    t1 = repo.upsert_track({
        "file_path": "/Music/T1.mp3",
        "file_hash": "h1",
        "title": "Low BPM",
        "bpm": 110.0,
        "energy": 0.3,
    })
    t2 = repo.upsert_track({
        "file_path": "/Music/T2.mp3",
        "file_hash": "h2",
        "title": "High BPM",
        "bpm": 130.0,
        "energy": 0.95,
    })

    # Test max_bpm & min_energy filters
    res, count = repo.list_tracks(max_bpm=120.0)
    assert count == 1
    assert res[0].title == "Low BPM"

    res_energy, count_energy = repo.list_tracks(min_energy=0.9)
    assert count_energy == 1
    assert res_energy[0].title == "High BPM"

    # Test auto position in playlist (position=None)
    pl = pl_repo.create_playlist("Auto Position Crate")
    pl_repo.add_track_to_playlist(pl.id, t1.id)  # position 0
    pl_repo.add_track_to_playlist(pl.id, t2.id)  # position 1

    tracks = pl_repo.get_playlist_tracks(pl.id)
    assert len(tracks) == 2


def test_tidal_client_bad_json(tmp_path):
    token_file = tmp_path / "bad.json"
    token_file.write_text("NOT_JSON{{{")
    client = TidalClient(token_file=token_file)
    assert client.is_authenticated is False


def test_normalize_key_unknown():
    cam, raw = normalize_key_to_camelot("UnknownKey123")
    assert cam == ""
    assert raw == "UnknownKey123"


def test_cli_tag_failure(tmp_path):
    # Mock AudioTagEngine.write_djay_pro_tags returning False
    test_file = tmp_path / "test.wav"
    test_file.touch()

    with patch("sonicdj.metadata.id3_engine.AudioTagEngine.write_djay_pro_tags", return_value=False):
        res = runner.invoke(app, ["tag", str(test_file), "--bpm", "125"])
        assert "Failed to write tags" in res.output
