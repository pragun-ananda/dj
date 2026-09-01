import json
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    ForeignKey,
    Boolean,
    Index,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Track(Base):
    """Database model for an indexed audio track."""
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String, unique=True, nullable=False, index=True)
    file_hash = Column(String(64), index=True, nullable=False)
    file_size_bytes = Column(Integer, default=0)
    format = Column(String(16), default="mp3")
    
    # Audio Technical Details
    duration_sec = Column(Float, default=0.0)
    sample_rate = Column(Integer, default=44100)
    bitrate_kbps = Column(Integer, default=0)
    channels = Column(Integer, default=2)

    # Core ID3 Metadata
    title = Column(String(255), default="Unknown Title", index=True)
    artist = Column(String(255), default="Unknown Artist", index=True)
    album = Column(String(255), default="Unknown Album")
    album_artist = Column(String(255), default="")
    genre = Column(String(128), default="Electronic", index=True)
    subgenres_json = Column(Text, default="[]")  # JSON list of detected subgenres
    year = Column(Integer, nullable=True)
    isrc = Column(String(32), nullable=True)

    # DJ Performance Metrics
    bpm = Column(Float, default=0.0, index=True)
    key_raw = Column(String(16), default="")  # e.g., "Am", "F#m", "C"
    camelot = Column(String(8), default="", index=True)  # e.g., "8A", "11B"
    energy = Column(Float, default=0.0, index=True)  # 0.0 to 1.0
    rating = Column(Integer, default=0)  # 0 to 5 stars
    comments = Column(Text, default="")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    analyzed_at = Column(DateTime, nullable=True)

    # Relationships
    cues = relationship("CuePoint", back_populates="track", cascade="all, delete-orphan")
    playlist_entries = relationship("PlaylistTrack", back_populates="track", cascade="all, delete-orphan")

    @property
    def subgenres(self) -> List[str]:
        try:
            return json.loads(self.subgenres_json) if self.subgenres_json else []
        except Exception:
            return []

    @subgenres.setter
    def subgenres(self, value: List[str]):
        self.subgenres_json = json.dumps(value or [])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "format": self.format,
            "duration_sec": self.duration_sec,
            "bitrate_kbps": self.bitrate_kbps,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "subgenres": self.subgenres,
            "year": self.year,
            "bpm": self.bpm,
            "key_raw": self.key_raw,
            "camelot": self.camelot,
            "energy": self.energy,
            "rating": self.rating,
            "comments": self.comments,
            "cues_count": len(self.cues) if self.cues else 0,
        }


class CuePoint(Base):
    """Database model for memory cues, hot cues, and structural markers."""
    __tablename__ = "cue_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(64), default="Cue")
    timestamp_ms = Column(Integer, nullable=False)  # Milliseconds from start
    cue_type = Column(String(32), default="hot_cue")  # intro, drop, breakdown, outro, vocal_start, vocal_end, custom
    hot_cue_index = Column(Integer, nullable=True)  # 0 to 7 (A through H)
    color_hex = Column(String(16), default="#00FFCC")

    track = relationship("Track", back_populates="cues")


class Playlist(Base):
    """Database model for DJ crates and smart playlists."""
    __tablename__ = "playlists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), unique=True, nullable=False)
    description = Column(Text, default="")
    is_smart = Column(Boolean, default=False)
    rules_json = Column(Text, default="{}")  # JSON rules for dynamic smart crates
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    tracks = relationship("PlaylistTrack", back_populates="playlist", cascade="all, delete-orphan")


class PlaylistTrack(Base):
    """Associative model for tracks inside playlists with custom ordering."""
    __tablename__ = "playlist_tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    playlist_id = Column(Integer, ForeignKey("playlists.id", ondelete="CASCADE"), nullable=False)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False)
    position = Column(Integer, default=0)

    playlist = relationship("Playlist", back_populates="tracks")
    track = relationship("Track", back_populates="playlist_entries")


class IngestionJob(Base):
    """Tracking model for background sync / ingestion tasks."""
    __tablename__ = "ingestion_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String(32), default="local_scan")  # "local_scan", "tidal_sync"
    status = Column(String(32), default="pending")  # "pending", "running", "completed", "failed"
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Performance Indexes for fast DJ queries
Index("idx_tracks_bpm_camelot", Track.bpm, Track.camelot)
Index("idx_tracks_energy", Track.energy)
