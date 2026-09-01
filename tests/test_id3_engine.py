from pathlib import Path
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.metadata.models import TrackMetadata, CueMetadata, normalize_key_to_camelot


def test_camelot_normalization():
    assert normalize_key_to_camelot("Am") == ("8A", "Am")
    assert normalize_key_to_camelot("A minor") == ("8A", "Am")
    assert normalize_key_to_camelot("8A") == ("8A", "Am")
    assert normalize_key_to_camelot("8a") == ("8A", "Am")
    assert normalize_key_to_camelot("11B") == ("11B", "A")
    assert normalize_key_to_camelot("F#m") == ("11A", "F#m")
    assert normalize_key_to_camelot("C") == ("8B", "C")
    assert normalize_key_to_camelot("C maj") == ("8B", "C")
    assert normalize_key_to_camelot("8A / Am") == ("8A", "Am")


def test_wav_tag_reading_and_writing(temp_audio_dir):
    wav_file = temp_audio_dir / "track_test.wav"
    
    # Initial read of raw WAV
    meta = AudioTagEngine.read_metadata(wav_file)
    assert meta.format == "wav"
    assert meta.channels == 2
    assert meta.sample_rate == 44100
    assert meta.duration_sec >= 0.9

    # Write enriched djay Pro tags
    new_meta = TrackMetadata(
        title="Sunlit Horizon",
        artist="Kelela & Black Coffee",
        album="Ibiza Nights",
        genre="Afro House",
        bpm=123.0,
        camelot="9A",
        key_raw="Em",
        energy=0.88,
        rating=5,
        comments="Vocal breakdown at 1:30",
        cues=[
            CueMetadata(name="Intro Beat", timestamp_ms=0),
            CueMetadata(name="Vocal Entry", timestamp_ms=32000),
            CueMetadata(name="Main Drop", timestamp_ms=64000),
        ]
    )

    success = AudioTagEngine.write_djay_pro_tags(wav_file, new_meta)
    assert success is True

    # Read back metadata to verify persistence
    read_back = AudioTagEngine.read_metadata(wav_file)
    assert read_back.title == "Sunlit Horizon"
    assert read_back.artist == "Kelela & Black Coffee"
    assert read_back.album == "Ibiza Nights"
    assert read_back.genre == "Afro House"
    assert read_back.bpm == 123.0
    assert read_back.camelot == "9A"
    assert read_back.rating == 5
    assert "Vocal Entry" in read_back.comments
