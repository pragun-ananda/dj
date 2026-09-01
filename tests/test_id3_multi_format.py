import aifc
import wave
import struct
from pathlib import Path
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.metadata.models import TrackMetadata, CueMetadata, CAMELOT_TO_KEY, KEY_TO_CAMELOT


def test_aiff_tag_reading_and_writing(tmp_path):
    aiff_path = tmp_path / "test_track.aiff"
    with aifc.open(str(aiff_path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(struct.pack(">h", 0) * 44100)

    meta = TrackMetadata(
        title="Deep Sunset",
        artist="Black Coffee",
        album="Subconsciously",
        genre="Afro House",
        bpm=122.0,
        camelot="10A",
        key_raw="Bm",
        energy=0.82,
        rating=4,
        comments="Vocal at 1:15",
        cues=[CueMetadata(name="Intro", timestamp_ms=0), CueMetadata(name="Drop", timestamp_ms=45000)],
    )

    success = AudioTagEngine.write_djay_pro_tags(aiff_path, meta)
    assert success is True

    read_back = AudioTagEngine.read_metadata(aiff_path)
    assert read_back.title == "Deep Sunset"
    assert read_back.artist == "Black Coffee"
    assert read_back.camelot == "10A"
    assert read_back.bpm == 122.0
    assert read_back.rating == 4


def test_metadata_model_to_dict():
    meta = TrackMetadata(
        title="Test Track",
        artist="Test Artist",
        bpm=125.0,
        camelot="8A",
    )
    assert meta.title == "Test Track"
    assert meta.bpm == 125.0


def test_camelot_full_mapping_consistency():
    for cam, key in CAMELOT_TO_KEY.items():
        assert cam in ("1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B", "6A", "6B",
                       "7A", "7B", "8A", "8B", "9A", "9B", "10A", "10B", "11A", "11B", "12A", "12B")
        assert len(key) >= 1


def test_flac_and_mp4_tag_parsing():
    # Test FLAC reader logic
    meta_flac = TrackMetadata()
    mock_flac = {
        "TITLE": ["Tribal Sunset"],
        "ARTIST": ["Black Coffee"],
        "ALBUM": ["Africa Gets Physical"],
        "GENRE": ["Afro House"],
        "DATE": ["2024"],
        "BPM": ["123.5"],
        "INITIALKEY": ["8A"],
        "ENERGY": ["0.87"],
        "RATING": ["5"],
        "COMMENT": ["Deep percussion"],
    }
    AudioTagEngine._read_flac_tags(mock_flac, meta_flac)
    assert meta_flac.title == "Tribal Sunset"
    assert meta_flac.artist == "Black Coffee"
    assert meta_flac.year == 2024
    assert meta_flac.bpm == 123.5
    assert meta_flac.camelot == "8A"
    assert meta_flac.energy == 0.87
    assert meta_flac.comments == "Deep percussion"

    # Test MP4 reader logic
    class MockMP4:
        tags = {
            "\xa9nam": ["Desert Mirage"],
            "\xa9ART": ["THEMBA"],
            "\xa9alb": ["Modern Rituals"],
            "\xa9gen": ["Afro Tech"],
            "\xa9day": ["2023"],
            "tmpo": [124],
            "----:com.apple.iTunes:CAMELOT": ["9A"],
            "----:com.apple.iTunes:ENERGY": ["0.92"],
            "\xa9cmt": ["Vocal drop at 1:15"],
        }

    meta_mp4 = TrackMetadata()
    AudioTagEngine._read_mp4_tags(MockMP4(), meta_mp4)
    assert meta_mp4.title == "Desert Mirage"
    assert meta_mp4.artist == "THEMBA"
    assert meta_mp4.year == 2023
    assert meta_mp4.bpm == 124.0
    assert meta_mp4.camelot == "9A"
    assert meta_mp4.energy == 0.92
