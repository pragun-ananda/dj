from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SubgenreClassification:
    primary_subgenre: str
    secondary_subgenres: List[str]
    moods: List[str]
    confidence: float


# Comprehensive Electronic Subgenre Taxonomy
SUBGENRE_FAMILIES = {
    "House": [
        "Afro House", "Deep House", "Tech House", "Melodic House", "Organic House",
        "Vocal House", "Latin House", "Acid House", "Funky House", "Jackin House",
        "Progressive House", "Soulful House", "Disco House", "Minimal House", "Electro House"
    ],
    "Techno": [
        "Melodic Techno", "Peak Time Techno", "Raw / Hypnotic Techno", "Hard Techno",
        "Industrial Techno", "Dub Techno", "Minimal Techno", "Acid Techno", "Detroit Techno",
        "Dark Techno", "Driving Techno"
    ],
    "Afro / Global": [
        "Amapiano", "3-Step", "Gqom", "Afro Tech", "Tribal House", "Kuduro", "Baile Funk", "Batida"
    ],
    "Bass & Breaks": [
        "UK Garage", "2-Step", "Breakbeat", "Drum & Bass", "Liquid DnB", "Jungle",
        "Dubstep", "Bassline", "Speed Garage", "Future Bass"
    ],
    "Trance & Melodic": [
        "Psytrance", "Uplifting Trance", "Progressive Trance", "Vocal Trance",
        "Hard Trance", "Eurodance", "Neo-Trance"
    ],
    "Downtempo & Ambient": [
        "Organic Downtempo", "Chillout", "Deep Ambient", "Trip Hop", "Lo-Fi House", "Balearic"
    ]
}

# Mood Categories
MOOD_KEYWORDS = {
    "hypnotic": ["hypnotic", "rolling", "monotonous", "trippy", "deep", "groove"],
    "euphoric": ["euphoric", "uplifting", "anthem", "festival", "bright", "peak"],
    "dark": ["dark", "heavy", "industrial", "ominous", "sinister", "raw", "acid"],
    "melancholic": ["melancholic", "emotional", "sad", "reflective", "deep", "minor"],
    "driving": ["driving", "pumping", "fast", "energetic", "relentless", "peak time"],
    "atmospheric": ["atmospheric", "space", "reverb", "ethereal", "lush", "warm"],
    "tribal": ["tribal", "percussive", "shaker", "conga", "afro", "drums"],
    "chill": ["chill", "smooth", "relaxed", "sunset", "lounge", "downtempo"],
}


class SubgenreClassifier:
    """Classifies audio tracks into granular subgenres and mood descriptors."""

    @classmethod
    def classify(
        cls,
        bpm: float,
        camelot: str,
        energy: float,
        has_vocals: bool,
        spectral_contrast: Optional[np.ndarray] = None,
    ) -> SubgenreClassification:
        """
        Classifies subgenre and mood based on extracted MIR features.
        """
        moods = []
        is_minor = camelot.endswith("A") if camelot else True

        # Mood deduction
        if energy >= 0.8:
            moods.append("driving")
            if is_minor:
                moods.append("dark")
            else:
                moods.append("euphoric")
        elif energy <= 0.5:
            moods.append("hypnotic")
            moods.append("chill" if not is_minor else "melancholic")
        else:
            moods.append("atmospheric")
            if is_minor:
                moods.append("hypnotic")

        # Subgenre deduction based on BPM, Energy, and Vocals
        if 112.0 <= bpm < 118.0:
            if energy < 0.6:
                primary = "Organic Downtempo"
                secondaries = ["Deep House", "Chillout"]
            else:
                primary = "Amapiano"
                secondaries = ["Afro House", "3-Step"]
                moods.append("tribal")
        elif 118.0 <= bpm < 125.0:
            if has_vocals and not is_minor:
                primary = "Vocal House"
                secondaries = ["Melodic House", "Disco House"]
                moods.append("euphoric")
            elif energy > 0.75:
                primary = "Afro House"
                secondaries = ["Tech House", "Tribal House"]
                moods.append("tribal")
            else:
                primary = "Deep House"
                secondaries = ["Organic House", "Melodic House"]
        elif 125.0 <= bpm < 130.0:
            if is_minor:
                if energy > 0.75:
                    primary = "Melodic Techno"
                    secondaries = ["Peak Time Techno", "Tech House"]
                else:
                    primary = "Tech House"
                    secondaries = ["Deep Tech", "Melodic Techno"]
            else:
                primary = "Progressive House"
                secondaries = ["Melodic House", "Trance"]
        elif 130.0 <= bpm < 136.0:
            if energy > 0.8:
                primary = "Peak Time Techno"
                secondaries = ["Raw / Hypnotic Techno", "Acid Techno"]
                moods.append("dark")
            else:
                primary = "UK Garage"
                secondaries = ["Breakbeat", "2-Step"]
        elif 136.0 <= bpm < 150.0:
            primary = "Hard Techno"
            secondaries = ["Psytrance", "Acid Techno"]
            moods.append("driving")
        elif bpm >= 150.0:
            primary = "Drum & Bass"
            secondaries = ["Liquid DnB", "Jungle"]
            moods.append("driving")
        else:
            primary = "Electronic"
            secondaries = ["Downtempo"]

        return SubgenreClassification(
            primary_subgenre=primary,
            secondary_subgenres=secondaries[:3],
            moods=list(set(moods))[:4],
            confidence=0.85,
        )
