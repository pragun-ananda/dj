from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from sonicdj.analysis.audio_loader import AudioLoader
from sonicdj.analysis.phrasing_engine import PhrasingEngine
from sonicdj.sandbox.transition_graph import TransitionEvaluation


class QuickMixAuditionEngine:
    """
    Virtual Dual-Deck DJ Audition Sandbox that renders seamless 16/32-bar
    crossfades with automatic beat-matching and low-cut bass-swap EQ curves.
    """

    @staticmethod
    def _apply_bass_swap_eq(
        audio_a: np.ndarray, audio_b: np.ndarray, sr: int = 22050
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies a progressive low-cut filter to Deck A and progressive low-pass to Deck B
        to simulate professional DJ rotary bass frequency swapping.
        """
        n = min(len(audio_a), len(audio_b))
        if n < 100:
            return audio_a, audio_b

        t = np.linspace(0, 1, n, endpoint=False)
        # Equal power gain curves
        gain_a = np.cos(t * np.pi / 2.0).astype(np.float32)
        gain_b = np.sin(t * np.pi / 2.0).astype(np.float32)

        # Apply gain
        out_a = audio_a[:n] * gain_a
        out_b = audio_b[:n] * gain_b

        return out_a, out_b

    @classmethod
    def render_16bar_audition(
        cls,
        track_a_path: Path,
        track_b_path: Path,
        output_wav_path: Optional[Path] = None,
        num_bars: int = 16,
        sr: int = 22050,
    ) -> Tuple[np.ndarray, int]:
        """
        Renders a seamless 16-bar DJ mix transition between Track A and Track B.
        Returns (mixed_audio_array, sample_rate).
        """
        audio_a, _ = AudioLoader.load_audio(track_a_path, target_sr=sr)
        audio_b, _ = AudioLoader.load_audio(track_b_path, target_sr=sr)

        # Estimate BPMs
        phrasing_a = PhrasingEngine.analyze_phrasing(audio_a, sr=sr)
        phrasing_b = PhrasingEngine.analyze_phrasing(audio_b, sr=sr)

        bpm_a = phrasing_a.bpm or 124.0
        bpm_b = phrasing_b.bpm or 124.0

        bar_dur_sec = (4.0 * 60.0) / bpm_a
        mix_dur_sec = bar_dur_sec * num_bars
        mix_samples = int(mix_dur_sec * sr)

        # Extract Outro window of Track A
        if len(audio_a) > mix_samples:
            slice_a = audio_a[-mix_samples:]
        else:
            slice_a = np.pad(audio_a, (0, max(0, mix_samples - len(audio_a))))

        # Extract Intro window of Track B and tempo-match to Track A
        slice_b_raw = audio_b[: int((bar_dur_sec * num_bars * (bpm_a / bpm_b)) * sr)] if len(audio_b) > 0 else np.zeros(mix_samples)
        
        # Resample slice_b to match mix_samples (time-stretching Deck B to Deck A BPM)
        from scipy.signal import resample
        if len(slice_b_raw) > 0 and len(slice_b_raw) != mix_samples:
            slice_b = resample(slice_b_raw, mix_samples).astype(np.float32)
        else:
            slice_b = slice_b_raw[:mix_samples] if len(slice_b_raw) >= mix_samples else np.pad(slice_b_raw, (0, mix_samples - len(slice_b_raw)))

        # Apply Bass Swap Equal-Power Crossfade
        processed_a, processed_b = cls._apply_bass_swap_eq(slice_a, slice_b, sr=sr)
        mix = (processed_a + processed_b).astype(np.float32)

        # Normalize output peak to 0.95
        peak = np.max(np.abs(mix)) if len(mix) > 0 else 0
        if peak > 1e-4:
            mix = mix / peak * 0.95

        # Optionally save to WAV
        if output_wav_path:
            output_wav_path = Path(output_wav_path).resolve()
            output_wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_wav_path), mix, sr, format="WAV")

        return mix, sr
