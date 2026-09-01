import re
from typing import List, Optional
from pydantic import BaseModel, Field


# Canonical Camelot Wheel mapping dictionary
CAMELOT_TO_KEY = {
    "1A": "Abm", "1B": "B",
    "2A": "Ebm", "2B": "F#",
    "3A": "Bbm", "3B": "Db",
    "4A": "Fm",  "4B": "Ab",
    "5A": "Cm",  "5B": "Eb",
    "6A": "Gm",  "6B": "Bb",
    "7A": "Dm",  "7B": "F",
    "8A": "Am",  "8B": "C",
    "9A": "Em",  "9B": "G",
    "10A": "Bm", "10B": "D",
    "11A": "F#m", "11B": "A",
    "12A": "Dbm", "12B": "E",
}

KEY_TO_CAMELOT = {
    # Minor keys
    "abm": "1A", "g#m": "1A", "ab min": "1A", "g# min": "1A", "ab minor": "1A", "g# minor": "1A",
    "ebm": "2A", "d#m": "2A", "eb min": "2A", "d# min": "2A", "eb minor": "2A", "d# minor": "2A",
    "bbm": "3A", "a#m": "3A", "bb min": "3A", "a# min": "3A", "bb minor": "3A", "a# minor": "3A",
    "fm": "4A", "f min": "4A", "f minor": "4A",
    "cm": "5A", "c min": "5A", "c minor": "5A",
    "gm": "6A", "g min": "6A", "g minor": "6A",
    "dm": "7A", "d min": "7A", "d minor": "7A",
    "am": "8A", "a min": "8A", "a minor": "8A",
    "em": "9A", "e min": "9A", "e minor": "9A",
    "bm": "10A", "b min": "10A", "b minor": "10A",
    "f#m": "11A", "gbm": "11A", "f# min": "11A", "gb min": "11A", "f# minor": "11A", "gb minor": "11A",
    "c#m": "12A", "dbm": "12A", "c# min": "12A", "db min": "12A", "c# minor": "12A", "db minor": "12A",

    # Major keys
    "b": "1B", "b maj": "1B", "b major": "1B",
    "f#": "2B", "gb": "2B", "f# maj": "2B", "gb maj": "2B", "f# major": "2B", "gb major": "2B",
    "db": "3B", "c#": "3B", "db maj": "3B", "c# maj": "3B", "db major": "3B", "c# major": "3B",
    "ab": "4B", "g#": "4B", "ab maj": "4B", "g# maj": "4B", "ab major": "4B", "g# major": "4B",
    "eb": "5B", "d#": "5B", "eb maj": "5B", "d# maj": "5B", "eb major": "5B", "d# major": "5B",
    "bb": "6B", "a#": "6B", "bb maj": "6B", "a# maj": "6B", "bb major": "6B", "a# major": "6B",
    "f": "7B", "f maj": "7B", "f major": "7B",
    "c": "8B", "c maj": "8B", "c major": "8B",
    "g": "9B", "g maj": "9B", "g major": "9B",
    "d": "10B", "d maj": "10B", "d major": "10B",
    "a": "11B", "a maj": "11B", "a major": "11B",
    "e": "12B", "e maj": "12B", "e major": "12B",
}


def normalize_key_to_camelot(key_str: str) -> tuple[str, str]:
    """
    Normalizes any key string (e.g. '8A', 'Am', 'A minor', 'F#m', '11B') into
    a tuple of (camelot_code, musical_key_str).
    """
    if not key_str:
        return "", ""

    clean = key_str.strip()
    upper = clean.upper()

    # Check if already in Camelot format e.g. "8A" or "8B"
    camelot_match = re.match(r"^(\d{1,2}[AB])$", upper)
    if camelot_match:
        cam = camelot_match.group(1)
        raw_key = CAMELOT_TO_KEY.get(cam, "")
        return cam, raw_key

    # Check if key is like "8A / Am" or "Am (8A)"
    combo_match = re.search(r"(\d{1,2}[AB])", upper)
    if combo_match:
        cam = combo_match.group(1)
        raw_key = CAMELOT_TO_KEY.get(cam, "")
        return cam, raw_key

    # Otherwise lookup in KEY_TO_CAMELOT
    lookup = clean.lower()
    cam = KEY_TO_CAMELOT.get(lookup, "")
    if cam:
        return cam, CAMELOT_TO_KEY.get(cam, clean)

    return "", clean


class CueMetadata(BaseModel):
    name: str = "Cue"
    timestamp_ms: int = 0
    cue_type: str = "hot_cue"
    hot_cue_index: Optional[int] = None
    color_hex: str = "#00FFCC"


class TrackMetadata(BaseModel):
    title: str = "Unknown Title"
    artist: str = "Unknown Artist"
    album: str = "Unknown Album"
    album_artist: str = ""
    genre: str = "Electronic"
    subgenres: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    isrc: Optional[str] = None

    # DJ Performance
    bpm: float = 0.0
    key_raw: str = ""
    camelot: str = ""
    energy: float = 0.0  # 0.0 - 1.0
    rating: int = 0      # 0 - 5
    comments: str = ""

    # Technical Audio info
    duration_sec: float = 0.0
    bitrate_kbps: int = 0
    sample_rate: int = 44100
    channels: int = 2
    format: str = "mp3"
    file_hash: str = ""
    file_size_bytes: int = 0

    cues: List[CueMetadata] = Field(default_factory=list)
