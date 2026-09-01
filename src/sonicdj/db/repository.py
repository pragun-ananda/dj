import json
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple
from sqlalchemy import create_engine, or_, and_, desc
from sqlalchemy.orm import sessionmaker, Session, joinedload

from sonicdj.config import settings
from sonicdj.db.schema import Base, Track, CuePoint, Playlist, PlaylistTrack, IngestionJob


class DatabaseManager:
    """Manages SQLite database connections and session lifecycles."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.db_url
        self.engine = create_engine(self.db_url, echo=False)
        self.SessionFactory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.init_db()

    def init_db(self) -> None:
        """Create tables if they do not exist."""
        settings.ensure_directories()
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class TrackRepository:
    """Repository handling all Track and CuePoint database operations."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def upsert_track(self, track_data: dict, cues: Optional[List[dict]] = None) -> Track:
        """Insert or update a track by file_path."""
        with self.db.session_scope() as session:
            file_path = track_data.get("file_path")

            existing = session.query(Track).filter(Track.file_path == file_path).first()

            if existing:
                for key, val in track_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, val)
                track = existing
            else:
                track = Track(**track_data)
                session.add(track)

            session.flush()

            if cues is not None:
                # Replace existing cues with newly analyzed cues
                session.query(CuePoint).filter(CuePoint.track_id == track.id).delete()
                for c in cues:
                    cue_entry = CuePoint(
                        track_id=track.id,
                        name=c.get("name", "Cue"),
                        timestamp_ms=c.get("timestamp_ms", 0),
                        cue_type=c.get("cue_type", "hot_cue"),
                        hot_cue_index=c.get("hot_cue_index"),
                        color_hex=c.get("color_hex", "#00FFCC"),
                    )
                    session.add(cue_entry)

            return track

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        with self.db.session_scope() as session:
            return session.query(Track).options(joinedload(Track.cues)).filter(Track.id == track_id).first()

    def get_track_by_path(self, file_path: str) -> Optional[Track]:
        with self.db.session_scope() as session:
            return session.query(Track).options(joinedload(Track.cues)).filter(Track.file_path == file_path).first()

    def list_tracks(
        self,
        genre: Optional[str] = None,
        camelot: Optional[str] = None,
        min_bpm: Optional[float] = None,
        max_bpm: Optional[float] = None,
        min_energy: Optional[float] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Track], int]:
        """Query tracks with multi-criteria DJ filtering."""
        with self.db.session_scope() as session:
            q = session.query(Track).options(joinedload(Track.cues))

            if genre:
                q = q.filter(Track.genre.ilike(f"%{genre}%"))
            if camelot:
                q = q.filter(Track.camelot == camelot)
            if min_bpm is not None:
                q = q.filter(Track.bpm >= min_bpm)
            if max_bpm is not None:
                q = q.filter(Track.bpm <= max_bpm)
            if min_energy is not None:
                q = q.filter(Track.energy >= min_energy)
            if search_query:
                term = f"%{search_query}%"
                q = q.filter(
                    or_(
                        Track.title.ilike(term),
                        Track.artist.ilike(term),
                        Track.album.ilike(term),
                        Track.genre.ilike(term),
                        Track.comments.ilike(term),
                    )
                )

            total_count = q.count()
            tracks = q.order_by(desc(Track.created_at)).offset(offset).limit(limit).all()
            return tracks, total_count

    def count_all(self) -> int:
        with self.db.session_scope() as session:
            return session.query(Track).count()


class PlaylistRepository:
    """Repository handling DJ Crates and Playlists."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create_playlist(self, name: str, description: str = "", is_smart: bool = False, rules: Optional[dict] = None) -> Playlist:
        with self.db.session_scope() as session:
            existing = session.query(Playlist).filter(Playlist.name == name).first()
            if existing:
                existing.description = description
                existing.is_smart = is_smart
                existing.rules_json = json.dumps(rules or {})
                return existing

            playlist = Playlist(
                name=name,
                description=description,
                is_smart=is_smart,
                rules_json=json.dumps(rules or {}),
            )
            session.add(playlist)
            session.flush()
            return playlist

    def add_track_to_playlist(self, playlist_id: int, track_id: int, position: Optional[int] = None) -> None:
        with self.db.session_scope() as session:
            if position is None:
                max_pos = session.query(PlaylistTrack.position).filter(
                    PlaylistTrack.playlist_id == playlist_id
                ).order_by(desc(PlaylistTrack.position)).first()
                position = (max_pos[0] + 1) if max_pos else 0

            entry = PlaylistTrack(playlist_id=playlist_id, track_id=track_id, position=position)
            session.add(entry)

    def get_playlist_tracks(self, playlist_id: int) -> List[Track]:
        with self.db.session_scope() as session:
            tracks = (
                session.query(Track)
                .join(PlaylistTrack, PlaylistTrack.track_id == Track.id)
                .filter(PlaylistTrack.playlist_id == playlist_id)
                .order_by(PlaylistTrack.position)
                .options(joinedload(Track.cues))
                .all()
            )
            return tracks


# Global DB instance helper
_default_db: Optional[DatabaseManager] = None

def get_db(db_url: Optional[str] = None) -> DatabaseManager:
    global _default_db
    if _default_db is None or db_url is not None:
        _default_db = DatabaseManager(db_url)
    return _default_db
