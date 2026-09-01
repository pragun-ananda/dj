from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from sonicdj.db.repository import DatabaseManager, TrackRepository
from sonicdj.db.schema import Track
from sonicdj.db.vector_store import VectorStore
from sonicdj.embeddings.audio_encoder import MultimodalAudioEncoder
from sonicdj.metadata.subgenres import SubgenreClassifier, SubgenreClassification


@dataclass
class SearchResult:
    track: Track
    semantic_score: float      # 0.0 to 1.0
    harmonic_score: float      # 0.0 to 1.0 (Camelot compatibility)
    bpm_score: float           # 0.0 to 1.0 (Tempo proximity)
    composite_score: float     # 0.0 to 1.0 (Overall DJ match ranking)
    subgenre_info: SubgenreClassification
    mix_recommendation: str    # DJ tip (e.g. "Perfect Harmonic Mix (8A -> 8A)", "Energy Boost (+2 Camelot)")


class HybridSearchEngine:
    """
    Combines multimodal semantic vector similarity with hard musical DJ constraints
    (Camelot Wheel harmonic mixing, BPM tempo matching, energy curves, and vocal windows).
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.repo = TrackRepository(db)
        self.vector_store = VectorStore(db)

    @classmethod
    def compute_harmonic_compatibility(cls, key_a: Optional[str], key_b: Optional[str]) -> Tuple[float, str]:
        """
        Calculates harmonic mixing compatibility between two Camelot keys.
        Returns (compatibility_score_0_to_1, recommendation_string).
        """
        if not key_a or not key_b:
            return 0.5, "Unknown key compatibility"

        key_a = key_a.upper().strip()
        key_b = key_b.upper().strip()

        if key_a == key_b:
            return 1.0, f"Perfect Harmonic Match ({key_a} -> {key_b})"

        try:
            num_a = int(key_a[:-1])
            mode_a = key_a[-1]
            num_b = int(key_b[:-1])
            mode_b = key_b[-1]
        except Exception:
            return 0.5, "Custom key"

        # 1. Relative Major / Minor (e.g. 8A <-> 8B)
        if num_a == num_b and mode_a != mode_b:
            return 0.95, f"Relative Major/Minor Mix ({key_a} -> {key_b})"

        # Calculate circular distance around 12-hour Camelot wheel
        diff = abs(num_a - num_b)
        wheel_dist = min(diff, 12 - diff)

        # 2. Adjacent +/- 1 (e.g. 8A -> 7A or 9A)
        if wheel_dist == 1 and mode_a == mode_b:
            return 0.90, f"Smooth Adjacent Transition ({key_a} -> {key_b})"

        # 3. Diagonal +/- 1 with mode flip (e.g. 8A -> 9B)
        if wheel_dist == 1 and mode_a != mode_b:
            return 0.80, f"Diagonal Key Transition ({key_a} -> {key_b})"

        # 4. Energy Boost (+2 semitones / +2 Camelot, e.g. 8A -> 10A)
        if (num_b - num_a) % 12 == 2 and mode_a == mode_b:
            return 0.75, f"Energy Boost Mix (+2 Camelot: {key_a} -> {key_b})"

        # Incompatible key (harmonic clash)
        return 0.20, f"Dissonant Transition ({key_a} -> {key_b})"

    @classmethod
    def compute_bpm_proximity(cls, bpm_a: Optional[float], bpm_b: Optional[float], max_tolerance: float = 8.0) -> float:
        """Calculates BPM proximity score (1.0 = identical tempo, 0.0 = >max_tolerance bpm apart)."""
        if bpm_a is None or bpm_b is None or bpm_a <= 0 or bpm_b <= 0:
            return 0.5
        diff = abs(bpm_a - bpm_b)
        return float(max(0.0, 1.0 - (diff / max_tolerance)))

    def search(
        self,
        prompt: Optional[str] = None,
        reference_track_id: Optional[int] = None,
        target_camelot: Optional[str] = None,
        target_bpm: Optional[float] = None,
        bpm_tolerance: float = 8.0,
        min_energy: Optional[float] = None,
        max_energy: Optional[float] = None,
        require_instrumental_intro_sec: Optional[float] = None,
        limit: int = 15,
    ) -> List[SearchResult]:
        """
        Executes hybrid semantic vector search + harmonic & BPM constraint ranking.
        """
        # 1. Obtain query vector
        query_vector = None
        ref_track = None

        if reference_track_id is not None:
            ref_track = self.repo.get_track_by_id(reference_track_id)
            if ref_track:
                query_vector = self.vector_store.get_embedding(reference_track_id)
                if not target_camelot:
                    target_camelot = ref_track.camelot
                if not target_bpm:
                    target_bpm = ref_track.bpm

        if query_vector is None and prompt:
            query_vector = MultimodalAudioEncoder.encode_text(prompt)
        elif query_vector is None:
            query_vector = MultimodalAudioEncoder._get_anchor_vector(42)

        # 2. Get vector search candidates from embedded store
        raw_knn_results = self.vector_store.search_knn(query_vector, top_k=max(50, limit * 3))
        knn_score_map = {track_id: score for track_id, score in raw_knn_results}

        # 3. Retrieve track metadata from database
        all_tracks, _ = self.repo.list_tracks(limit=10000)
        ranked_results: List[SearchResult] = []

        for track in all_tracks:
            # Skip self if referencing
            if reference_track_id and track.id == reference_track_id:
                continue

            # Hard filtering checks
            if target_bpm is not None and track.bpm:
                if abs(track.bpm - target_bpm) > bpm_tolerance:
                    continue

            if min_energy is not None and track.energy is not None and track.energy < min_energy:
                continue
            if max_energy is not None and track.energy is not None and track.energy > max_energy:
                continue

            # Semantic score from vector store (normalized 0 to 1)
            raw_sim = knn_score_map.get(track.id, 0.4)
            semantic_score = float(min(1.0, max(0.0, (raw_sim + 1.0) / 2.0)))

            # Harmonic score
            if target_camelot:
                harmonic_score, rec_tip = self.compute_harmonic_compatibility(target_camelot, track.camelot)
            else:
                harmonic_score, rec_tip = 0.8, "Open key selection"

            # BPM score
            if target_bpm:
                bpm_score = self.compute_bpm_proximity(target_bpm, track.bpm, max_tolerance=bpm_tolerance)
            else:
                bpm_score = 0.8

            # Composite DJ Score
            composite_score = round(
                (0.45 * semantic_score) + (0.30 * harmonic_score) + (0.25 * bpm_score), 3
            )

            # Subgenre and mood classification
            subgenre_info = SubgenreClassifier.classify(
                bpm=track.bpm or 124.0,
                camelot=track.camelot or "8A",
                energy=track.energy or 0.75,
                has_vocals="vocal" in (track.comments or "").lower(),
            )

            ranked_results.append(
                SearchResult(
                    track=track,
                    semantic_score=round(semantic_score, 3),
                    harmonic_score=round(harmonic_score, 3),
                    bpm_score=round(bpm_score, 3),
                    composite_score=composite_score,
                    subgenre_info=subgenre_info,
                    mix_recommendation=rec_tip,
                )
            )

        # Sort by composite score descending
        ranked_results.sort(key=lambda x: x.composite_score, reverse=True)
        return ranked_results[:limit]
