import json
from pathlib import Path
import httpx
from sonicdj.ingestion.tidal_client import TidalClient, TidalTrackInfo


def test_tidal_client_token_lifecycle(tmp_path):
    token_file = tmp_path / "test_token.json"
    client = TidalClient(token_file=token_file)
    assert not client.is_authenticated

    # Save token
    token_data = {
        "access_token": "mock_access_token_12345",
        "refresh_token": "mock_refresh_token_67890",
        "user_id": "999888",
        "country_code": "US"
    }
    client.save_token(token_data)

    # Re-initialize to test loading from disk
    client2 = TidalClient(token_file=token_file)
    assert client2.is_authenticated
    assert client2.access_token == "mock_access_token_12345"
    assert client2.user_id == "999888"
    assert client2._get_headers()["Authorization"] == "Bearer mock_access_token_12345"


def test_tidal_unauthenticated_calls(tmp_path):
    token_file = tmp_path / "empty_token.json"
    client = TidalClient(token_file=token_file)
    
    # Should safely return empty list if not authenticated
    assert client.search_tracks("Afro House") == []
    assert client.get_playlist_tracks("some-uuid") == []
