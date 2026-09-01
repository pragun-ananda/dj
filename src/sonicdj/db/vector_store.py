import struct
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime
import numpy as np
from sqlalchemy import Table, Column, Integer, LargeBinary, DateTime, MetaData, select, insert, update, delete
from sonicdj.db.repository import DatabaseManager


class VectorStore:
    """
    Embedded, high-performance zero-server vector store using SQLite blob storage
    and vectorized NumPy matrix operations for millisecond k-NN search.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.metadata = MetaData()
        self.embeddings_table = Table(
            "track_embeddings",
            self.metadata,
            Column("track_id", Integer, primary_key=True),
            Column("dim", Integer, nullable=False),
            Column("vector_blob", LargeBinary, nullable=False),
            Column("updated_at", DateTime, default=datetime.utcnow),
        )
        self.metadata.create_all(self.db.engine)
        self._cache_track_ids: Optional[np.ndarray] = None
        self._cache_matrix: Optional[np.ndarray] = None
        self._dirty_cache = True

    @staticmethod
    def _vector_to_blob(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def _blob_to_vector(blob: bytes, dim: int = 512) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def upsert_embedding(self, track_id: int, vector: np.ndarray) -> None:
        """Saves or updates a track's 512D embedding vector."""
        vec_norm = vector.astype(np.float32)
        norm = np.linalg.norm(vec_norm)
        if norm > 1e-8:
            vec_norm = vec_norm / norm

        blob = self._vector_to_blob(vec_norm)
        dim = len(vec_norm)

        with self.db.session_scope() as session:
            # Check if exists
            stmt = select(self.embeddings_table.c.track_id).where(self.embeddings_table.c.track_id == track_id)
            existing = session.execute(stmt).scalar_one_or_none()

            if existing is not None:
                update_stmt = (
                    update(self.embeddings_table)
                    .where(self.embeddings_table.c.track_id == track_id)
                    .values(dim=dim, vector_blob=blob, updated_at=datetime.utcnow())
                )
                session.execute(update_stmt)
            else:
                insert_stmt = insert(self.embeddings_table).values(
                    track_id=track_id, dim=dim, vector_blob=blob, updated_at=datetime.utcnow()
                )
                session.execute(insert_stmt)

        self._dirty_cache = True

    def batch_upsert_embeddings(self, items: List[Tuple[int, np.ndarray]]) -> None:
        """Batch upserts multiple track embeddings."""
        for track_id, vec in items:
            self.upsert_embedding(track_id, vec)

    def _refresh_cache(self) -> None:
        """Loads all vectors into memory matrix for sub-millisecond batch matrix ops."""
        with self.db.session_scope() as session:
            stmt = select(
                self.embeddings_table.c.track_id,
                self.embeddings_table.c.dim,
                self.embeddings_table.c.vector_blob
            )
            rows = session.execute(stmt).fetchall()

        if not rows:
            self._cache_track_ids = np.array([], dtype=int)
            self._cache_matrix = np.empty((0, 512), dtype=np.float32)
            self._dirty_cache = False
            return

        track_ids = []
        vectors = []
        for track_id, dim, blob in rows:
            track_ids.append(track_id)
            vectors.append(self._blob_to_vector(blob, dim))

        self._cache_track_ids = np.array(track_ids, dtype=int)
        self._cache_matrix = np.vstack(vectors).astype(np.float32)
        self._dirty_cache = False

    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Retrieves a track's embedding vector."""
        with self.db.session_scope() as session:
            stmt = select(self.embeddings_table.c.dim, self.embeddings_table.c.vector_blob).where(
                self.embeddings_table.c.track_id == track_id
            )
            row = session.execute(stmt).fetchone()
            if row:
                dim, blob = row
                return self._blob_to_vector(blob, dim)
        return None

    def search_knn(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        candidate_track_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Executes fast k-NN vector search against all stored tracks.
        Returns list of (track_id, similarity_score) sorted by descending similarity.
        """
        if self._dirty_cache or self._cache_matrix is None or self._cache_track_ids is None:
            self._refresh_cache()

        if len(self._cache_track_ids) == 0:
            return []

        q_norm = query_vector.astype(np.float32)
        norm = np.linalg.norm(q_norm)
        if norm > 1e-8:
            q_norm = q_norm / norm

        # Batch matrix multiplication: (N, 512) @ (512,) -> (N,)
        scores = np.dot(self._cache_matrix, q_norm)

        # Filter by candidate track IDs if provided (e.g. after SQL pre-filtering)
        if candidate_track_ids is not None:
            results = []
            for idx, track_id in enumerate(self._cache_track_ids):
                if track_id in candidate_track_ids:
                    results.append((int(track_id), float(scores[idx])))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        # Top-K sorting
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (int(self._cache_track_ids[idx]), float(scores[idx]))
            for idx in top_indices
        ]
