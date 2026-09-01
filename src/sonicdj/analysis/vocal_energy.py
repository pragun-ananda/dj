from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy.signal import stft


@dataclass
class VocalSegment:
    start_sec: float
    end_sec: float
    confidence: float


@dataclass
class VocalEnergyResult:
    overall_energy: float             # 0.0 to 1.0
    energy_curve: List[float]         # 1-second interval energy points (0.0 to 1.0)
    has_vocals: bool
    vocal_presence_percent: float     # 0 to 100%
    first_vocal_sec: Optional[float]
    last_vocal_sec: Optional[float]
    vocal_segments: List[VocalSegment]
    instrumental_intro_sec: float
    instrumental_outro_sec: float


class VocalEnergyProfiler:
    """
    Computes continuous dynamic energy curves and isolates vocal activity
    to determine safe instrumental DJ mix-in and mix-out windows.
    """

    @classmethod
    def compute_energy_profile(
        cls, audio: np.ndarray, sr: int = 22050, window_sec: float = 1.0
    ) -> Tuple[float, List[float]]:
        """
        Computes 1-second interval energy scores and overall average energy.
        """
        if len(audio) == 0:
            return 0.0, []

        hop_samples = int(sr * window_sec)
        n_windows = max(1, len(audio) // hop_samples)

        energy_points = []
        for i in range(n_windows):
            chunk = audio[i * hop_samples : (i + 1) * hop_samples]
            if len(chunk) == 0:
                continue
            # RMS amplitude
            rms = np.sqrt(np.mean(chunk**2) + 1e-8)
            # Normalize to 0.0 - 1.0 curve with perceptual log scaling
            norm_val = min(1.0, float(rms * 3.5))
            energy_points.append(round(norm_val, 3))

        overall_energy = float(np.mean(energy_points)) if energy_points else 0.5
        return round(overall_energy, 2), energy_points

    @classmethod
    def detect_vocal_activity(
        cls, audio: np.ndarray, sr: int = 22050, threshold: float = 0.35
    ) -> VocalEnergyResult:
        """
        Extracts vocal presence timeline and energy curves from audio.
        """
        total_duration = len(audio) / float(sr) if sr > 0 else 0.0
        overall_energy, energy_curve = cls.compute_energy_profile(audio, sr=sr)

        if total_duration < 2.0 or len(audio) < 2048:
            return VocalEnergyResult(
                overall_energy=overall_energy,
                energy_curve=energy_curve,
                has_vocals=False,
                vocal_presence_percent=0.0,
                first_vocal_sec=None,
                last_vocal_sec=None,
                vocal_segments=[],
                instrumental_intro_sec=total_duration,
                instrumental_outro_sec=total_duration,
            )

        # STFT for spectral formant band analysis (300 Hz - 3500 Hz vocal range)
        n_fft = 2048
        hop_length = 1024
        frequencies, times, Zxx = stft(
            audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None
        )
        mag = np.abs(Zxx)

        vocal_mask = (frequencies >= 300.0) & (frequencies <= 3500.0)
        total_mask = (frequencies >= 60.0) & (frequencies <= 10000.0)

        vocal_power = np.sum(mag[vocal_mask, :], axis=0)
        total_power = np.sum(mag[total_mask, :], axis=0) + 1e-8

        # Ratio of vocal band energy to overall spectrum
        vocal_ratio = vocal_power / total_power

        # Smooth vocal envelope over ~1.5 second windows
        smooth_frames = int(round(1.5 * (sr / float(hop_length))))
        if smooth_frames > 1:
            kernel = np.ones(smooth_frames) / smooth_frames
            vocal_ratio_smooth = np.convolve(vocal_ratio, kernel, mode="same")
        else:
            vocal_ratio_smooth = vocal_ratio

        # Find contiguous active vocal segments
        is_vocal = vocal_ratio_smooth > threshold
        segments = []
        in_segment = False
        seg_start = 0.0

        for frame_idx, active in enumerate(is_vocal):
            t = times[frame_idx]
            if active and not in_segment:
                in_segment = True
                seg_start = t
            elif not active and in_segment:
                in_segment = False
                seg_duration = t - seg_start
                if seg_duration >= 2.0:  # Minimum 2s to count as vocal phrase
                    conf = float(np.mean(vocal_ratio_smooth[int(seg_start * sr / hop_length) : frame_idx]))
                    segments.append(VocalSegment(start_sec=round(seg_start, 2), end_sec=round(t, 2), confidence=round(min(1.0, conf * 2), 2)))

        if in_segment and (total_duration - seg_start) >= 2.0:
            segments.append(VocalSegment(start_sec=round(seg_start, 2), end_sec=round(total_duration, 2), confidence=0.8))

        has_vocals = len(segments) > 0
        total_vocal_time = sum([s.end_sec - s.start_sec for s in segments])
        vocal_percent = round((total_vocal_time / total_duration) * 100.0, 1) if total_duration > 0 else 0.0

        first_vocal = segments[0].start_sec if segments else None
        last_vocal = segments[-1].end_sec if segments else None

        intro_inst = first_vocal if first_vocal is not None else total_duration
        outro_inst = (total_duration - last_vocal) if last_vocal is not None else total_duration

        return VocalEnergyResult(
            overall_energy=overall_energy,
            energy_curve=energy_curve,
            has_vocals=has_vocals,
            vocal_presence_percent=vocal_percent,
            first_vocal_sec=first_vocal,
            last_vocal_sec=last_vocal,
            vocal_segments=segments,
            instrumental_intro_sec=round(intro_inst, 2),
            instrumental_outro_sec=round(outro_inst, 2),
        )
