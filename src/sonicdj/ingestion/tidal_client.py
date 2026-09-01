import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
from pydantic import BaseModel

from sonicdj.config import settings


class TidalTrackInfo(BaseModel):
    id: str
    title: str
    artist: str
    album: str
    duration_sec: float
    isrc: Optional[str] = None
    release_date: Optional[str] = None
    cover_url: Optional[str] = None
    audio_quality: str = "LOSSLESS"


class TidalClient:
    """
    Client for interacting with Tidal's API for metadata retrieval and stream extraction.
    Supports session caching, search, and playlist resolution.
    """

    BASE_URL = "https://api.tidal.com/v1"
    AUTH_URL = "https://auth.tidal.com/v1/oauth2"

    def __init__(self, token_file: Optional[Path] = None):
        self.token_file = token_file or (settings.app_dir / "tidal_token.json")
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.country_code: str = "US"
        self._load_cached_token()

    def _load_cached_token(self) -> bool:
        if self.token_file.exists():
            try:
                with open(self.token_file, "r") as f:
                    data = json.load(f)
                    self.access_token = data.get("access_token")
                    self.refresh_token = data.get("refresh_token")
                    self.user_id = data.get("user_id")
                    self.country_code = data.get("country_code", "US")
                    return bool(self.access_token)
            except Exception:
                pass
        return False

    def save_token(self, token_data: Dict[str, Any]) -> None:
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")
        self.user_id = str(token_data.get("user_id", ""))
        with open(self.token_file, "w") as f:
            json.dump(token_data, f, indent=2)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.access_token)

    def _get_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "SonicDJ/1.0"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def search_tracks(self, query: str, limit: int = 20) -> List[TidalTrackInfo]:
        """Search for tracks on Tidal."""
        if not self.is_authenticated:
            return []

        url = f"{self.BASE_URL}/search/tracks"
        params = {"query": query, "limit": limit, "countryCode": self.country_code}

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params=params, headers=self._get_headers())
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            items = data.get("items", [])
            results = []
            for item in items:
                artist_name = item.get("artist", {}).get("name", "Unknown Artist")
                album_name = item.get("album", {}).get("title", "")
                cover_id = item.get("album", {}).get("cover")
                cover_url = f"https://resources.tidal.com/images/{cover_id.replace('-', '/')}/640x640.jpg" if cover_id else None

                results.append(
                    TidalTrackInfo(
                        id=str(item.get("id")),
                        title=item.get("title", "Unknown"),
                        artist=artist_name,
                        album=album_name,
                        duration_sec=float(item.get("duration", 0)),
                        isrc=item.get("isrc"),
                        release_date=item.get("streamStartDate"),
                        cover_url=cover_url,
                        audio_quality=item.get("audioQuality", "LOSSLESS"),
                    )
                )
            return results

    def get_playlist_tracks(self, playlist_uuid: str) -> List[TidalTrackInfo]:
        """Fetch all track metadata from a Tidal playlist."""
        if not self.is_authenticated:
            return []

        url = f"{self.BASE_URL}/playlists/{playlist_uuid}/tracks"
        params = {"limit": 100, "countryCode": self.country_code}

        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params=params, headers=self._get_headers())
            if resp.status_code != 200:
                return []

            data = resp.json()
            items = data.get("items", [])
            results = []
            for item in items:
                artist_name = item.get("artist", {}).get("name", "Unknown Artist")
                album_name = item.get("album", {}).get("title", "")
                cover_id = item.get("album", {}).get("cover")
                cover_url = f"https://resources.tidal.com/images/{cover_id.replace('-', '/')}/640x640.jpg" if cover_id else None

                results.append(
                    TidalTrackInfo(
                        id=str(item.get("id")),
                        title=item.get("title", "Unknown"),
                        artist=artist_name,
                        album=album_name,
                        duration_sec=float(item.get("duration", 0)),
                        isrc=item.get("isrc"),
                        release_date=item.get("streamStartDate"),
                        cover_url=cover_url,
                        audio_quality=item.get("audioQuality", "LOSSLESS"),
                    )
                )
            return results
