import math
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy.signal import stft

from sonicdj.metadata.models import CAMELOT_TO_KEY, KEY_TO_CAMELOT, normalize_key_to_camelot


@dataclass
class KeyAnalysisResult:
    musical_key: str          # e.g., "Am", "C", "F#m"
    camelot: str              # e.g., "8A", "8B", "11A"
    confidence: float         # 0.0 to 1.0
    tuning_hz: float          # estimated concert pitch (e.g. 440.0 Hz)
    pitch_drift_cents: float  # deviation in cents from 440 Hz (-50 to +50)
    chroma_profile: list[float]  # 12-element normalized chroma vector


class KeyDetector:
    """
    High-accuracy harmonic key, Camelot wheel, and pitch drift detection engine.
    Uses multi-pass chroma feature extraction and harmonic profile correlation (EDMA / Krumhansl).
    """

    PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    # Krumhansl-Schmuckler & EDMA Harmonic Profiles for Major and Minor Keys
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    @classmethod
    def compute_chromagram(
        cls, audio: np.ndarray, sr: int = 22050, n_fft: int = 4096, hop_length: int = 1024
    ) -> Tuple[np.ndarray, float]:
        """
        Computes 12-bin chromagram and detects average tuning deviation (pitch drift in cents).
        """
        if len(audio) < n_fft:
            # Fallback for very short test audio
            return np.ones(12) / 12.0, 0.0

        frequencies, times, Zxx = stft(
            audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None
        )
        magnitude = np.abs(Zxx)

        # Focus on musical fundamentals (A1 = 55Hz to A7 = 3520Hz)
        valid_freq_mask = (frequencies >= 55.0) & (frequencies <= 3520.0)
        valid_freqs = frequencies[valid_freq_mask]
        valid_mag = magnitude[valid_freq_mask, :]

        if len(valid_freqs) == 0:
            return np.ones(12) / 12.0, 0.0

        # Compute pitch drift around 440 Hz
        # MIDI note = 12 * log2(f / 440) + 69
        midi_exact = 12.0 * np.log2(valid_freqs / 440.0) + 69.0
        midi_rounded = np.round(midi_exact)
        cents_offsets = (midi_exact - midi_rounded) * 100.0  # Deviation in cents

        # Energy-weighted pitch drift calculation
        mean_power_per_bin = np.mean(valid_mag, axis=1)
        total_power = np.sum(mean_power_per_bin)
        if total_power > 1e-6:
            pitch_drift = float(np.sum(cents_offsets * mean_power_per_bin) / total_power)
        else:
            pitch_drift = 0.0

        # Accumulate energy into 12 pitch classes (0=C, 1=Db, ..., 11=B)
        pitch_classes = (midi_rounded.astype(int) % 12)
        chroma = np.zeros(12, dtype=np.float64)

        for p_class, p_mag in zip(pitch_classes, mean_power_per_bin):
            chroma[p_class] += p_mag

        # Normalize chroma vector
        chroma_norm = np.linalg.norm(chroma)
        if chroma_norm > 1e-6:
            chroma = chroma / chroma_norm
        else:
            chroma = np.ones(12) / np.sqrt(12)

        return chroma, pitch_drift

    @classmethod
    def detect_key(cls, audio: np.ndarray, sr: int = 22050) -> KeyAnalysisResult:
        """
        Analyzes audio waveform and returns musical key, Camelot notation, confidence, and tuning.
        """
        if len(audio) == 0:
            return KeyAnalysisResult(
                musical_key="Unknown",
                camelot="",
                confidence=0.0,
                tuning_hz=440.0,
                pitch_drift_cents=0.0,
                chroma_profile=[0.0] * 12,
            )

        chroma, pitch_drift = cls.compute_chromagram(audio, sr=sr)

        # Normalize standard profiles for zero-mean Pearson correlation
        major_norm = (cls.MAJOR_PROFILE - np.mean(cls.MAJOR_PROFILE)) / np.std(cls.MAJOR_PROFILE)
        minor_norm = (cls.MINOR_PROFILE - np.mean(cls.MINOR_PROFILE)) / np.std(cls.MINOR_PROFILE)

        chroma_mean = np.mean(chroma)
        chroma_std = np.std(chroma)
        if chroma_std > 1e-6:
            chroma_norm = (chroma - chroma_mean) / chroma_std
        else:
            chroma_norm = chroma

        correlations = {}

        for root_idx in range(12):
            root_name = cls.PITCH_NAMES[root_idx]
            # Roll chroma to align with root note
            rolled_chroma = np.roll(chroma_norm, -root_idx)

            # Major correlation
            r_major = np.dot(rolled_chroma, major_norm) / 12.0
            correlations[f"{root_name}"] = float(r_major)

            # Minor correlation
            r_minor = np.dot(rolled_chroma, minor_norm) / 12.0
            correlations[f"{root_name}m"] = float(r_minor)

        # Find best matching key
        best_key = max(correlations, key=correlations.get)
        best_score = correlations[best_key]

        # Calculate confidence based on gap to second-best candidate
        sorted_scores = sorted(correlations.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) > 0:
            gap = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, max(0.2, float(best_score * 0.7 + gap * 1.5)))
        else:
            confidence = 0.5

        # Normalize to Camelot
        camelot, canonical_key = normalize_key_to_camelot(best_key)

        # Estimated concert pitch (440 * 2^(cents / 1200))
        tuning_hz = 440.0 * (2.0 ** (pitch_drift / 1200.0))

        return KeyAnalysisResult(
            musical_key=canonical_key or best_key,
            camelot=camelot,
            confidence=round(confidence, 2),
            tuning_hz=round(tuning_hz, 1),
            pitch_drift_cents=round(pitch_drift, 1),
            chroma_profile=[round(float(x), 4) for x in chroma],
        )
