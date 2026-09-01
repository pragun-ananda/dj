from typing import List, Optional
from dataclasses import dataclass
from sonicdj.metadata.models import CueMetadata
from sonicdj.analysis.key_detector import KeyAnalysisResult
from sonicdj.analysis.phrasing_engine import PhrasingAnalysisResult, PhraseSection
from sonicdj.analysis.vocal_energy import VocalEnergyResult


@dataclass
class CompleteTrackAnalysis:
    key_info: KeyAnalysisResult
    phrasing_info: PhrasingAnalysisResult
    vocal_energy_info: VocalEnergyResult
    generated_cues: List[CueMetadata]
    summary_comment: str


class CueGenerator:
    """
    Synthesizes MIR analysis into standardized, color-coded DJ Hot Cues
    and Memory Points formatted for Algoriddim djay Pro and Rekordbox.
    """

    COLOR_MAP = {
        "intro": "#00FFCC",       # Cyan / Green
        "vocal": "#00AAFF",       # Sky Blue
        "drop": "#FF0055",        # High-energy Magenta / Red
        "breakdown": "#FF8800",   # Amber / Orange
        "outro": "#AA00FF",       # Purple
        "custom": "#FFFF00",      # Yellow
    }

    @classmethod
    def generate_cues(
        cls,
        phrasing: PhrasingAnalysisResult,
        vocal_energy: VocalEnergyResult,
        key_info: Optional[KeyAnalysisResult] = None,
    ) -> List[CueMetadata]:
        """
        Generates 4 to 6 DJ-ready performance cues based on track structure.
        """
        cues = []
        cue_idx = 0

        # 1. Cue A (Intro / First Beat)
        intro_ms = int(round(phrasing.first_downbeat_sec * 1000.0))
        cues.append(
            CueMetadata(
                name="Intro Beat",
                timestamp_ms=intro_ms,
                cue_type="intro",
                hot_cue_index=cue_idx,
                color_hex=cls.COLOR_MAP["intro"],
            )
        )
        cue_idx += 1

        # 2. Cue B (First Vocal Entry or First Verse)
        if vocal_energy.has_vocals and vocal_energy.first_vocal_sec is not None:
            vocal_ms = int(round(vocal_energy.first_vocal_sec * 1000.0))
            if vocal_ms > intro_ms + 2000:
                cues.append(
                    CueMetadata(
                        name="Vocal Entry",
                        timestamp_ms=vocal_ms,
                        cue_type="vocal_start",
                        hot_cue_index=cue_idx,
                        color_hex=cls.COLOR_MAP["vocal"],
                    )
                )
                cue_idx += 1
        elif len(phrasing.phrase_16bar_timestamps) > 1:
            # Fallback: 16-bar mark for instrumental tracks
            verse_ms = int(round(phrasing.phrase_16bar_timestamps[1] * 1000.0))
            cues.append(
                CueMetadata(
                    name="16b Verse",
                    timestamp_ms=verse_ms,
                    cue_type="verse",
                    hot_cue_index=cue_idx,
                    color_hex=cls.COLOR_MAP["vocal"],
                )
            )
            cue_idx += 1

        # 3. Cue C (Main Drop / Peak Energy)
        drop_section = next((s for s in phrasing.sections if s.name == "drop"), None)
        if drop_section:
            drop_ms = int(round(drop_section.start_sec * 1000.0))
            if drop_ms > intro_ms + 5000:
                cues.append(
                    CueMetadata(
                        name="Main Drop",
                        timestamp_ms=drop_ms,
                        cue_type="drop",
                        hot_cue_index=cue_idx,
                        color_hex=cls.COLOR_MAP["drop"],
                    )
                )
                cue_idx += 1

        # 4. Cue D (Breakdown / Bridge)
        breakdown_section = next((s for s in phrasing.sections if s.name == "breakdown"), None)
        if breakdown_section:
            bk_ms = int(round(breakdown_section.start_sec * 1000.0))
            cues.append(
                CueMetadata(
                    name="Breakdown",
                    timestamp_ms=bk_ms,
                    cue_type="breakdown",
                    hot_cue_index=cue_idx,
                    color_hex=cls.COLOR_MAP["breakdown"],
                )
            )
            cue_idx += 1

        # 5. Cue E (Outro / Mix-Out Point)
        outro_section = next((s for s in phrasing.sections if s.name == "outro"), None)
        if outro_section and outro_section.start_sec > 10.0:
            outro_ms = int(round(outro_section.start_sec * 1000.0))
            cues.append(
                CueMetadata(
                    name="Outro Beat",
                    timestamp_ms=outro_ms,
                    cue_type="outro",
                    hot_cue_index=cue_idx,
                    color_hex=cls.COLOR_MAP["outro"],
                )
            )
            cue_idx += 1

        return cues

    @classmethod
    def generate_summary_comment(
        cls,
        key_info: KeyAnalysisResult,
        phrasing: PhrasingAnalysisResult,
        vocal_energy: VocalEnergyResult,
        cues: List[CueMetadata],
    ) -> str:
        """
        Creates a rich, searchable DJ metadata comment string for djay Pro.
        Example: "[8A 123BPM] [Energy 85%] [Vocals: 30%] | [Intro: 0s] [Drop: 32s] [Outro: 180s]"
        """
        tuning_str = f" ({key_info.tuning_hz}Hz)" if abs(key_info.pitch_drift_cents) > 5.0 else ""
        vocal_str = f" [Vocals: {vocal_energy.vocal_presence_percent}%]" if vocal_energy.has_vocals else " [Instrumental]"
        
        cue_parts = " ".join([f"[{c.name}: {c.timestamp_ms // 1000}s]" for c in cues])
        return f"[{key_info.camelot} | {phrasing.bpm:.1f} BPM{tuning_str}] [Energy: {int(vocal_energy.overall_energy * 100)}%]{vocal_str} | {cue_parts}".strip(" |")
