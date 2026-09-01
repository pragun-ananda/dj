import re
import math
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.signal import stft


class MultimodalAudioEncoder:
    """
    Multimodal audio & text embedding engine that projects music tracks
    and natural language text prompts into a shared 512-dimensional semantic vector space.
    """

    EMBEDDING_DIM = 512

    # Deterministic pseudo-random projection seeds for semantic anchors
    GENRE_ANCHORS = {
        "afro": 101, "house": 102, "techno": 103, "amapiano": 104,
        "melodic": 105, "deep": 106, "tech": 107, "vocal": 108,
        "dark": 109, "euphoric": 110, "tribal": 111, "drum": 112,
        "bass": 113, "synth": 114, "piano": 115, "drop": 116,
        "hypnotic": 117, "driving": 118, "acid": 119, "chill": 120,
        "minimal": 121, "progressive": 122, "garage": 123, "breakbeat": 124,
    }

    @classmethod
    def _get_anchor_vector(cls, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        v = rng.randn(cls.EMBEDDING_DIM).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    @classmethod
    def encode_audio(
        cls,
        audio: np.ndarray,
        sr: int = 22050,
        bpm: Optional[float] = None,
        camelot: Optional[str] = None,
        energy: Optional[float] = None,
        has_vocals: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Extracts spectral, rhythmic, and harmonic features from audio
        and projects into a 512D unit-normalized embedding vector.
        """
        if len(audio) == 0:
            v = np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)
            v[0] = 1.0
            return v

        # 1. Spectral feature extraction
        n_fft = 2048
        hop_length = 1024
        frequencies, _, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
        mag = np.abs(Zxx)
        mean_spec = np.mean(mag, axis=1)

        # Multi-band energy distribution (Sub-bass, Bass, Low-Mid, Mid, High-Mid, High)
        sub_bass = np.sum(mean_spec[(frequencies >= 20) & (frequencies < 80)])
        bass = np.sum(mean_spec[(frequencies >= 80) & (frequencies < 250)])
        mid = np.sum(mean_spec[(frequencies >= 250) & (frequencies < 2000)])
        high = np.sum(mean_spec[(frequencies >= 2000) & (frequencies < 10000)])
        total = sub_bass + bass + mid + high + 1e-8

        # Build feature vector
        feats = np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)

        # Deterministic projections based on audio characteristics
        # Sub-bass power (Bass / Techno / Groove)
        feats += (sub_bass / total) * cls._get_anchor_vector(cls.GENRE_ANCHORS["bass"])
        # Mid power (Vocal / Chords / Melodic)
        feats += (mid / total) * cls._get_anchor_vector(cls.GENRE_ANCHORS["melodic"])
        # High power (Percussion / Shakers / Acid)
        feats += (high / total) * cls._get_anchor_vector(cls.GENRE_ANCHORS["tribal"])

        if bpm is not None:
            if 115 <= bpm <= 125:
                feats += 0.8 * cls._get_anchor_vector(cls.GENRE_ANCHORS["house"])
                feats += 0.5 * cls._get_anchor_vector(cls.GENRE_ANCHORS["afro"])
            elif 125 < bpm <= 135:
                feats += 0.9 * cls._get_anchor_vector(cls.GENRE_ANCHORS["techno"])
                feats += 0.5 * cls._get_anchor_vector(cls.GENRE_ANCHORS["driving"])
            elif bpm > 135:
                feats += 0.8 * cls._get_anchor_vector(cls.GENRE_ANCHORS["driving"])

        if energy is not None:
            if energy > 0.75:
                feats += energy * cls._get_anchor_vector(cls.GENRE_ANCHORS["drop"])
                feats += energy * cls._get_anchor_vector(cls.GENRE_ANCHORS["driving"])
            else:
                feats += (1.0 - energy) * cls._get_anchor_vector(cls.GENRE_ANCHORS["deep"])
                feats += (1.0 - energy) * cls._get_anchor_vector(cls.GENRE_ANCHORS["hypnotic"])

        if has_vocals:
            feats += 1.2 * cls._get_anchor_vector(cls.GENRE_ANCHORS["vocal"])

        if camelot:
            if camelot.endswith("A"):  # Minor
                feats += 0.5 * cls._get_anchor_vector(cls.GENRE_ANCHORS["dark"])
            else:  # Major
                feats += 0.5 * cls._get_anchor_vector(cls.GENRE_ANCHORS["euphoric"])

        # Normalize to unit sphere
        norm = np.linalg.norm(feats)
        if norm > 1e-8:
            return (feats / norm).astype(np.float32)
        
        fallback = cls._get_anchor_vector(42)
        return fallback.astype(np.float32)

    @classmethod
    def encode_text(cls, query: str) -> np.ndarray:
        """
        Encodes a natural language text query into the shared 512D vector space.
        Example queries: "dark hypnotic afrohouse with heavy bass", "vocal house anthem"
        """
        clean_text = query.lower().strip()
        tokens = re.findall(r"\w+", clean_text)

        if not tokens:
            v = cls._get_anchor_vector(42)
            return v.astype(np.float32)

        emb = np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)
        matched_any = False

        for token in tokens:
            for keyword, seed in cls.GENRE_ANCHORS.items():
                if keyword in token or token in keyword:
                    weight = 1.5 if token == keyword else 0.8
                    emb += weight * cls._get_anchor_vector(seed)
                    matched_any = True

        if not matched_any:
            # Fallback: hash token strings into deterministic embedding
            for token in tokens:
                token_seed = sum(ord(c) for c in token) % 1000 + 200
                emb += cls._get_anchor_vector(token_seed)

        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            return (emb / norm).astype(np.float32)

        v = cls._get_anchor_vector(42)
        return v.astype(np.float32)

    @classmethod
    def compute_similarity(cls, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two normalized vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
