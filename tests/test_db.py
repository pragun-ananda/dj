import pytest
from sonicdj.db.repository import TrackRepository, PlaylistRepository
from sonicdj.metadata.models import normalize_key_to_camelot


def test_track_repository_crud(temp_db):
    repo = TrackRepository(temp_db)
    
    # 1. Insert Track
    track_data = {
        "file_path": "/Music/Afrohouse_01.flac",
        "file_hash": "a1b2c3d4e5f6",
        "title": "Tribal Journey",
        "artist": "DJ Solar",
        "album": "Desert Mirage",
        "genre": "Afro House",
        "bpm": 122.5,
        "camelot": "8A",
        "key_raw": "Am",
        "energy": 0.85,
        "rating": 5,
        "duration_sec": 360.0,
    }
    cues = [
        {"name": "Intro", "timestamp_ms": 0, "cue_type": "intro", "hot_cue_index": 0},
        {"name": "Main Drop", "timestamp_ms": 64000, "cue_type": "drop", "hot_cue_index": 1},
    ]

    track = repo.upsert_track(track_data, cues=cues)
    assert track.id is not None
    assert track.title == "Tribal Journey"
    assert track.camelot == "8A"

    # 2. Query Track
    fetched = repo.get_track_by_id(track.id)
    assert fetched is not None
    assert len(fetched.cues) == 2
    assert fetched.cues[1].name == "Main Drop"

    # 3. Filter by Camelot & BPM
    results, total = repo.list_tracks(camelot="8A", min_bpm=120.0, max_bpm=125.0)
    assert total == 1
    assert results[0].artist == "DJ Solar"

    # 4. Filter with non-matching query
    results_empty, total_empty = repo.list_tracks(camelot="11B")
    assert total_empty == 0


def test_playlist_repository(temp_db):
    track_repo = TrackRepository(temp_db)
    playlist_repo = PlaylistRepository(temp_db)

    t1 = track_repo.upsert_track({
        "file_path": "/Music/Track1.mp3",
        "file_hash": "hash1",
        "title": "Track One",
        "artist": "Artist A",
    })
    t2 = track_repo.upsert_track({
        "file_path": "/Music/Track2.mp3",
        "file_hash": "hash2",
        "title": "Track Two",
        "artist": "Artist B",
    })

    pl = playlist_repo.create_playlist("Peak Time Crates", description="Warmup to Peak")
    assert pl.id is not None
    assert pl.name == "Peak Time Crates"

    playlist_repo.add_track_to_playlist(pl.id, t1.id, position=1)
    playlist_repo.add_track_to_playlist(pl.id, t2.id, position=2)

    tracks = playlist_repo.get_playlist_tracks(pl.id)
    assert len(tracks) == 2
    assert tracks[0].title == "Track One"
    assert tracks[1].title == "Track Two"
