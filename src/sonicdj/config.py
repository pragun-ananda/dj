from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration settings for SonicDJ."""

    app_name: str = "SonicDJ"
    app_dir: Path = Field(default_factory=lambda: Path.home() / ".sonicdj")
    db_name: str = "library.db"
    
    # Audio Storage & Ingestion
    music_dir: Path = Field(default_factory=lambda: Path.home() / "Music" / "SonicDJ")
    supported_extensions: set[str] = {".mp3", ".flac", ".m4a", ".aif", ".aiff", ".wav", ".aac"}
    preferred_audio_format: str = "flac"  # "flac" or "mp3"
    mp3_bitrate_kbps: int = 320

    # Tidal API Settings
    tidal_client_id: str = ""
    tidal_client_secret: str = ""
    tidal_quality: str = "LOSSLESS"  # "HI_RES", "LOSSLESS", "HIGH"

    # Analysis Settings
    default_sample_rate: int = 44100
    cue_phrase_bars: list[int] = [8, 16, 32, 64]

    @property
    def db_path(self) -> Path:
        self.app_dir.mkdir(parents=True, exist_ok=True)
        return self.app_dir / self.db_name

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    def ensure_directories(self) -> None:
        """Ensure necessary directories exist."""
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.music_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
