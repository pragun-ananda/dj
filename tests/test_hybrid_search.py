import numpy as np
from sonicdj.db.repository import TrackRepository
from sonicdj.db.vector_store import VectorStore
from sonicdj.search.hybrid_engine import HybridSearchEngine
from sonicdj.embeddings.audio_encoder import MultimodalAudioEncoder


def test_harmonic_compatibility():
    # 1. Identical key
    score, tip = HybridSearchEngine.compute_harmonic_compatibility("8A", "8A")
    assert score == 1.0
    assert "Perfect Harmonic Match" in tip

    # 2. Relative Major/Minor
    score_rel, tip_rel = HybridSearchEngine.compute_harmonic_compatibility("8A", "8B")
    assert score_rel == 0.95
    assert "Relative" in tip_rel

    # 3. Adjacent +/- 1
    score_adj, tip_adj = HybridSearchEngine.compute_harmonic_compatibility("8A", "9A")
    assert score_adj == 0.90
    assert "Smooth Adjacent" in tip_adj

    # 4. Diagonal +/- 1
    score_diag, tip_diag = HybridSearchEngine.compute_harmonic_compatibility("8A", "9B")
    assert score_diag == 0.80

    # 5. Energy Boost (+2)
    score_boost, tip_boost = HybridSearchEngine.compute_harmonic_compatibility("8A", "10A")
    assert score_boost == 0.75
    assert "Energy Boost" in tip_boost

    # 6. Dissonant key (e.g. 8A to 2A)
    score_diss, _ = HybridSearchEngine.compute_harmonic_compatibility("8A", "2A")
    assert score_diss == 0.20


def test_hybrid_search_execution(temp_db):
    repo = TrackRepository(temp_db)
    store = VectorStore(temp_db)
    engine = HybridSearchEngine(temp_db)

    # 1. Insert 3 tracks into DB
    t1 = repo.upsert_track({
        "file_path": "/music/afro1.wav", "file_hash": "h1", "file_size_bytes": 1000,
        "format": "wav", "duration_sec": 180.0, "title": "Afro Sun", "artist": "DJ Keeno",
        "bpm": 123.0, "camelot": "8A", "energy": 0.85, "comments": "Tribal drums with vocal"
    })
    t2 = repo.upsert_track({
        "file_path": "/music/techno1.wav", "file_hash": "h2", "file_size_bytes": 1000,
        "format": "wav", "duration_sec": 200.0, "title": "Dark Pulse", "artist": "Stephan",
        "bpm": 130.0, "camelot": "9A", "energy": 0.90, "comments": "Peak techno"
    })
    t3 = repo.upsert_track({
        "file_path": "/music/deep1.wav", "file_hash": "h3", "file_size_bytes": 1000,
        "format": "wav", "duration_sec": 220.0, "title": "Chill Morning", "artist": "Bonobo",
        "bpm": 115.0, "camelot": "8B", "energy": 0.40, "comments": "Hypnotic chill"
    })

    # 2. Insert vector embeddings
    v_afro = MultimodalAudioEncoder.encode_text("tribal afrohouse drums")
    v_techno = MultimodalAudioEncoder.encode_text("driving dark techno")
    v_chill = MultimodalAudioEncoder.encode_text("chill deep house")

    store.upsert_embedding(t1.id, v_afro)
    store.upsert_embedding(t2.id, v_techno)
    store.upsert_embedding(t3.id, v_chill)

    # 3. Query: "tribal afro drums" targeting 8A @ 123 BPM
    results = engine.search(
        prompt="tribal afrohouse drums",
        target_camelot="8A",
        target_bpm=123.0,
        bpm_tolerance=5.0,
        min_energy=0.7,
        limit=5,
    )
    assert len(results) >= 1
    assert results[0].track.title == "Afro Sun"
    assert results[0].composite_score > 0.8

    # 4. Search by reference track
    ref_results = engine.search(reference_track_id=t1.id, limit=5)
    assert len(ref_results) >= 1
    assert t1.id not in [r.track.id for r in ref_results]
