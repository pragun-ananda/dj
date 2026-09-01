from sonicdj.analysis.key_detector import KeyAnalysisResult
from sonicdj.analysis.phrasing_engine import PhrasingAnalysisResult, PhraseSection
from sonicdj.analysis.vocal_energy import VocalEnergyResult, VocalSegment
from sonicdj.analysis.cue_generator import CueGenerator


def test_cue_generator():
    key_info = KeyAnalysisResult(
        musical_key="Am",
        camelot="8A",
        confidence=0.88,
        tuning_hz=440.0,
        pitch_drift_cents=0.0,
        chroma_profile=[0.0] * 12,
    )
    phrasing = PhrasingAnalysisResult(
        bpm=123.0,
        confidence=0.92,
        first_downbeat_sec=0.5,
        bar_duration_sec=1.95,
        beat_timestamps=[0.5, 1.0, 1.5, 2.0],
        downbeat_timestamps=[0.5, 2.45],
        phrase_16bar_timestamps=[0.5, 31.7],
        sections=[
            PhraseSection("intro", 0.5, 31.7, 1, 16, 0.45),
            PhraseSection("drop", 31.7, 94.1, 17, 48, 0.95),
            PhraseSection("breakdown", 94.1, 125.3, 49, 64, 0.50),
            PhraseSection("outro", 180.0, 211.2, 65, 80, 0.40),
        ],
    )
    vocal_energy = VocalEnergyResult(
        overall_energy=0.82,
        energy_curve=[0.5] * 200,
        has_vocals=True,
        vocal_presence_percent=45.0,
        first_vocal_sec=15.5,
        last_vocal_sec=175.0,
        vocal_segments=[VocalSegment(15.5, 175.0, 0.85)],
        instrumental_intro_sec=15.5,
        instrumental_outro_sec=36.2,
    )

    cues = CueGenerator.generate_cues(phrasing, vocal_energy, key_info)
    assert len(cues) >= 4

    cue_names = [c.name for c in cues]
    assert "Intro Beat" in cue_names
    assert "Vocal Entry" in cue_names
    assert "Main Drop" in cue_names
    assert "Outro Beat" in cue_names

    comment = CueGenerator.generate_summary_comment(key_info, phrasing, vocal_energy, cues)
    assert "[8A | 123.0 BPM]" in comment
    assert "[Energy: 82%]" in comment
    assert "[Vocals: 45.0%]" in comment
    assert "[Intro Beat: 0s]" in comment


def test_cue_generator_instrumental_fallback():
    key_info = KeyAnalysisResult("Am", "8A", 0.9, 440.0, 0.0, [0.0] * 12)
    phrasing = PhrasingAnalysisResult(
        bpm=120.0,
        confidence=0.8,
        first_downbeat_sec=0.0,
        bar_duration_sec=2.0,
        beat_timestamps=[0.0, 0.5, 1.0],
        downbeat_timestamps=[0.0, 2.0],
        phrase_16bar_timestamps=[0.0, 32.0, 64.0],
        sections=[],
    )
    vocal_energy = VocalEnergyResult(
        overall_energy=0.7,
        energy_curve=[0.5] * 100,
        has_vocals=False,
        vocal_presence_percent=0.0,
        first_vocal_sec=None,
        last_vocal_sec=None,
        vocal_segments=[],
        instrumental_intro_sec=100.0,
        instrumental_outro_sec=100.0,
    )
    cues = CueGenerator.generate_cues(phrasing, vocal_energy, key_info)
    cue_names = [c.name for c in cues]
    assert "Intro Beat" in cue_names
    assert "16b Verse" in cue_names
