from sonicdj.db.repository import TrackRepository
from sonicdj.sandbox.transition_graph import TransitionGraphEngine


def test_transition_evaluation(temp_db):
    repo = TrackRepository(temp_db)
    engine = TransitionGraphEngine(temp_db)

    t1 = repo.upsert_track({
        "file_path": "/music/track1.flac", "file_hash": "h1", "file_size_bytes": 1000,
        "format": "flac", "duration_sec": 180.0, "title": "Warmup Groove", "artist": "Artist A",
        "bpm": 122.0, "camelot": "8A", "energy": 0.70, "comments": "Deep instrumental groove"
    })
    t2 = repo.upsert_track({
        "file_path": "/music/track2.flac", "file_hash": "h2", "file_size_bytes": 1000,
        "format": "flac", "duration_sec": 200.0, "title": "Mid Energy Vibe", "artist": "Artist B",
        "bpm": 124.0, "camelot": "9A", "energy": 0.82, "comments": "Afro rhythm"
    })

    eval_res = engine.evaluate_transition(t1, t2)
    assert eval_res.overall_compatibility > 0.8
    assert eval_res.harmonic_score >= 0.90
    assert eval_res.recommended_bars == 16
    assert "Smooth Adjacent" in eval_res.explanation


def test_find_optimal_set_path(temp_db):
    repo = TrackRepository(temp_db)
    engine = TransitionGraphEngine(temp_db)

    # 3-track progression: 8A (120 BPM) -> 9A (124 BPM) -> 10A (128 BPM)
    t1 = repo.upsert_track({"file_path": "/m/1.wav", "file_hash": "1", "file_size_bytes": 10, "format": "wav", "duration_sec": 180, "title": "Opener", "artist": "A", "bpm": 120.0, "camelot": "8A", "energy": 0.6})
    t2 = repo.upsert_track({"file_path": "/m/2.wav", "file_hash": "2", "file_size_bytes": 10, "format": "wav", "duration_sec": 180, "title": "Bridge", "artist": "B", "bpm": 124.0, "camelot": "9A", "energy": 0.75})
    t3 = repo.upsert_track({"file_path": "/m/3.wav", "file_hash": "3", "file_size_bytes": 10, "format": "wav", "duration_sec": 180, "title": "Climax", "artist": "C", "bpm": 128.0, "camelot": "10A", "energy": 0.95})

    # Path from t1 to t3 should find t1 -> t2 -> t3
    steps = engine.find_optimal_set_path(t1.id, t3.id, max_hops=4)
    assert len(steps) == 3
    assert steps[0].track.title == "Opener"
    assert steps[1].track.title == "Bridge"
    assert steps[2].track.title == "Climax"

    # Same track
    steps_same = engine.find_optimal_set_path(t1.id, t1.id)
    assert len(steps_same) == 1

    # Missing track
    steps_missing = engine.find_optimal_set_path(999, 1000)
    assert len(steps_missing) == 0
