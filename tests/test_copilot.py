from sonicdj.db.repository import TrackRepository
from sonicdj.hardware.flx4 import FLX4Controller
from sonicdj.copilot.live_engine import LiveCopilotEngine


def test_copilot_engine_and_warnings(temp_db):
    repo = TrackRepository(temp_db)
    flx4 = FLX4Controller()
    copilot = LiveCopilotEngine(temp_db, flx4)

    t1 = repo.upsert_track({
        "file_path": "/m/afro1.wav", "file_hash": "1", "file_size_bytes": 100,
        "format": "wav", "duration_sec": 180.0, "title": "Afro Track", "artist": "DJ A",
        "bpm": 123.0, "camelot": "8A", "energy": 0.85, "comments": "Peak vocal lead"
    })
    t2 = repo.upsert_track({
        "file_path": "/m/clash.wav", "file_hash": "2", "file_size_bytes": 100,
        "format": "wav", "duration_sec": 180.0, "title": "Clash Track", "artist": "DJ B",
        "bpm": 124.0, "camelot": "2B", "energy": 0.90, "comments": "Loud vocal hook"
    })

    copilot.load_deck_track(1, t1.id)
    copilot.load_deck_track(2, t2.id)
    copilot.update_deck_pos(1, 45.0)
    copilot.update_deck_pos(2, 10.0)

    # 1. Simulate dual faders up with high low-EQ to trigger warnings
    flx4.state.deck1_volume = 0.9
    flx4.state.deck2_volume = 0.9
    flx4.state.deck1_eq_low = 0.8
    flx4.state.deck2_eq_low = 0.8

    hud = copilot.get_hud_state()
    assert hud.active_master_deck in (1, 2)
    assert hud.deck1_track is not None
    assert hud.deck1_track["title"] == "Afro Track"
    assert len(hud.active_warnings) >= 2

    # Verify specific warning triggers
    warnings_str = " ".join(hud.active_warnings)
    assert "Bass Overload" in warnings_str
    assert "Dissonant Key Clash" in warnings_str
    assert "Vocal Collision" in warnings_str
