import time
import httpx
from sonicdj.db.repository import TrackRepository
from sonicdj.ui.server import SonicDJServer


def test_ui_server_api_and_html(temp_db):
    repo = TrackRepository(temp_db)
    t1 = repo.upsert_track({
        "file_path": "/m/t1.wav", "file_hash": "h1", "file_size_bytes": 1000,
        "format": "wav", "duration_sec": 180.0, "title": "Workstation Groove", "artist": "Keeno",
        "bpm": 124.0, "camelot": "8A", "energy": 0.85, "comments": "Afro percussion"
    })

    # Pick a test port (e.g. 8765)
    port = 8765
    server = SonicDJServer(temp_db, port=port)
    server.start_background()
    time.sleep(0.1)

    base_url = f"http://127.0.0.1:{port}"

    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            # 1. GET HTML Dashboard
            resp_html = client.get("/")
            assert resp_html.status_code == 200
            assert "SONICDJ" in resp_html.text

            # 2. GET /api/tracks
            resp_tracks = client.get("/api/tracks")
            assert resp_tracks.status_code == 200
            assert resp_tracks.json()["total"] >= 1

            # 3. GET /api/search
            resp_search = client.get("/api/search", params={"q": "groove", "key": "8A"})
            assert resp_search.status_code == 200
            assert len(resp_search.json()["results"]) >= 1

            # 4. POST /api/copilot/load
            resp_load = client.post("/api/copilot/load", json={"deck": 1, "track_id": t1.id})
            assert resp_load.status_code == 200
            assert resp_load.json()["status"] == "loaded"

            # 5. POST /api/flx4/event
            resp_flx4 = client.post("/api/flx4/event", json={"crossfader": 0.0, "deck1_vol": 1.0, "deck2_vol": 0.0})
            assert resp_flx4.status_code == 200
            assert resp_flx4.json()["master_deck"] == 1

            # 6. GET /api/copilot
            resp_copilot = client.get("/api/copilot")
            assert resp_copilot.status_code == 200
            data = resp_copilot.json()
            assert data["active_master_deck"] == 1
            assert data["deck1_track"]["title"] == "Workstation Groove"
    finally:
        server.stop()
