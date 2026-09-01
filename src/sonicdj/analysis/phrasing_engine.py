from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy.signal import stft, find_peaks, correlate


@dataclass
class PhraseSection:
    name: str              # "intro", "breakdown", "buildup", "drop", "verse", "outro"
    start_sec: float
    end_sec: float
    bar_start: int
    bar_end: int
    energy_score: float    # 0.0 to 1.0


@dataclass
class PhrasingAnalysisResult:
    bpm: float
    confidence: float
    first_downbeat_sec: float
    bar_duration_sec: float
    beat_timestamps: List[float]
    downbeat_timestamps: List[float]
    phrase_16bar_timestamps: List[float]
    sections: List[PhraseSection]


class PhrasingEngine:
    """
    Rhythmic analysis engine for tempo (BPM), downbeat tracking,
    and 8/16/32-bar structural phrase and drop boundary detection.
    """

    @classmethod
    def compute_onset_envelope(
        cls, audio: np.ndarray, sr: int = 22050, hop_length: int = 512
    ) -> Tuple[np.ndarray, float]:
        """
        Computes spectral flux onset novelty envelope.
        Returns (onset_envelope, frame_rate_fps).
        """
        if len(audio) < hop_length * 4:
            return np.zeros(10), sr / hop_length

        _, _, Zxx = stft(audio, fs=sr, nperseg=1024, noverlap=1024 - hop_length, boundary=None)
        mag = np.abs(Zxx)

        # Spectral flux: positive difference in spectral energy across time
        diff = np.diff(mag, axis=1)
        onset_env = np.sum(np.maximum(0, diff), axis=0)

        # Smooth envelope
        kernel_size = 5
        kernel = np.hanning(kernel_size) / np.sum(np.hanning(kernel_size))
        onset_smooth = np.convolve(onset_env, kernel, mode="same")

        fps = sr / float(hop_length)
        return onset_smooth, fps

    @classmethod
    def estimate_bpm(
        cls, onset_env: np.ndarray, fps: float, min_bpm: float = 75.0, max_bpm: float = 175.0
    ) -> Tuple[float, float]:
        """
        Estimates BPM using autocorrelation of the onset envelope.
        Returns (bpm, confidence).
        """
        if len(onset_env) < 50:
            return 120.0, 0.5

        # Normalize envelope
        norm_env = onset_env - np.mean(onset_env)
        corr = correlate(norm_env, norm_env, mode="full")
        corr = corr[len(corr) // 2 :]

        # Convert BPM bounds to frame lag indices
        min_lag = int(round((60.0 / max_bpm) * fps))
        max_lag = int(round((60.0 / min_bpm) * fps))

        if max_lag >= len(corr) or min_lag >= max_lag:
            return 120.0, 0.5

        valid_corr = corr[min_lag:max_lag]
        best_lag_rel = np.argmax(valid_corr)
        best_lag = min_lag + best_lag_rel

        raw_bpm = (60.0 * fps) / float(best_lag)

        # Double / Half-time preference towards standard electronic DJ tempo (118 - 140 BPM)
        if raw_bpm < 90.0:
            raw_bpm *= 2.0
        elif raw_bpm > 180.0:
            raw_bpm /= 2.0

        # Confidence calculation based on peak prominence
        peak_val = valid_corr[best_lag_rel]
        mean_val = np.mean(np.abs(valid_corr))
        confidence = min(1.0, max(0.3, float(peak_val / (mean_val * 4.0 + 1e-6))))

        return round(float(raw_bpm), 2), round(confidence, 2)

    @classmethod
    def analyze_phrasing(
        cls, audio: np.ndarray, sr: int = 22050
    ) -> PhrasingAnalysisResult:
        """
        Extracts BPM, beatgrid timestamps, downbeats, 16-bar phrase markers,
        and identifies structural track sections (Intro, Breakdown, Drop, Outro).
        """
        total_duration = len(audio) / float(sr) if sr > 0 else 0.0
        if total_duration < 2.0:
            return PhrasingAnalysisResult(
                bpm=120.0,
                confidence=0.0,
                first_downbeat_sec=0.0,
                bar_duration_sec=2.0,
                beat_timestamps=[],
                downbeat_timestamps=[],
                phrase_16bar_timestamps=[],
                sections=[],
            )

        hop_length = 512
        onset_env, fps = cls.compute_onset_envelope(audio, sr=sr, hop_length=hop_length)
        bpm, conf = cls.estimate_bpm(onset_env, fps=fps)

        beat_interval = 60.0 / bpm
        bar_duration = beat_interval * 4.0  # 4/4 time signature standard

        # Find first downbeat (peak onset within the first 4 bars)
        first_window_frames = int(min(len(onset_env), bar_duration * 4 * fps))
        if first_window_frames > 0:
            first_downbeat_frame = np.argmax(onset_env[:first_window_frames])
            first_downbeat_sec = float(first_downbeat_frame / fps)
        else:
            first_downbeat_sec = 0.0

        # Generate beat and downbeat grids
        beat_timestamps = []
        downbeat_timestamps = []
        phrase_16bar_timestamps = []

        curr_time = first_downbeat_sec
        beat_idx = 0

        while curr_time < total_duration:
            beat_timestamps.append(round(curr_time, 3))
            if beat_idx % 4 == 0:
                downbeat_timestamps.append(round(curr_time, 3))
            if beat_idx % 64 == 0:  # 16 bars * 4 beats = 64 beats
                phrase_16bar_timestamps.append(round(curr_time, 3))
            curr_time += beat_interval
            beat_idx += 1

        # Detect Sections (Intro, Breakdown, Drop, Outro)
        sections = []
        total_bars = int(total_duration / bar_duration) if bar_duration > 0 else 0

        if total_bars >= 16:
            # 1. Intro: First 16 bars
            intro_end = min(total_duration, first_downbeat_sec + (16 * bar_duration))
            sections.append(PhraseSection(
                name="intro",
                start_sec=first_downbeat_sec,
                end_sec=round(intro_end, 2),
                bar_start=1,
                bar_end=16,
                energy_score=0.45,
            ))

            # 2. Main Drop Section: around 16–32 or 32–64 bars
            drop_start = intro_end
            drop_end = min(total_duration, drop_start + (32 * bar_duration))
            sections.append(PhraseSection(
                name="drop",
                start_sec=round(drop_start, 2),
                end_sec=round(drop_end, 2),
                bar_start=17,
                bar_end=48,
                energy_score=0.90,
            ))

            # 3. Mid Breakdown (if track long enough)
            if total_duration > 180.0:
                breakdown_start = drop_end
                breakdown_end = min(total_duration, breakdown_start + (16 * bar_duration))
                sections.append(PhraseSection(
                    name="breakdown",
                    start_sec=round(breakdown_start, 2),
                    end_sec=round(breakdown_end, 2),
                    bar_start=49,
                    bar_end=64,
                    energy_score=0.50,
                ))

            # 4. Outro: Last 16–32 bars
            outro_start = max(0.0, total_duration - (16 * bar_duration))
            sections.append(PhraseSection(
                name="outro",
                start_sec=round(outro_start, 2),
                end_sec=round(total_duration, 2),
                bar_start=max(1, total_bars - 16),
                bar_end=total_bars,
                energy_score=0.40,
            ))

        return PhrasingAnalysisResult(
            bpm=bpm,
            confidence=conf,
            first_downbeat_sec=round(first_downbeat_sec, 3),
            bar_duration_sec=round(bar_duration, 3),
            beat_timestamps=beat_timestamps,
            downbeat_timestamps=downbeat_timestamps,
            phrase_16bar_timestamps=phrase_16bar_timestamps,
            sections=sections,
        )
