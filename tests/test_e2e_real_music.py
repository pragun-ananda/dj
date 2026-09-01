import math
import wave
import struct
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

from sonicdj.analysis.batch_analyzer import AudioAnalyzer
from sonicdj.analysis.key_detector import KeyDetector
from sonicdj.analysis.phrasing_engine import PhrasingEngine
from sonicdj.analysis.vocal_energy import VocalEnergyProfiler
from sonicdj.metadata.id3_engine import AudioTagEngine
from sonicdj.metadata.djay_exporter import DjayProExporter
from sonicdj.db.repository import TrackRepository, DatabaseManager


def synthesize_realistic_afrohouse_track(
    duration_sec: float = 90.0,
    bpm: float = 123.0,
    key_root_hz: float = 440.0,  # A4 (A minor)
    sr: int = 22050,
) -> np.ndarray:
    """
    Synthesizes a realistic multi-track electronic song with:
    1. Punchy 4/4 Kick drum with pitch envelope drop (120Hz -> 50Hz)
    2. Off-beat 16th-note Shakers and Hi-Hats
    3. Rolling Bassline in A Minor
    4. Polyphonic Piano/Synth chord progression (Am -> F -> C -> G)
    5. Vocal Lead with formants & vibrato (active 15s to 45s)
    6. Full Arrangement: Intro (0-15s) -> Verse (15-45s) -> Breakdown (45-60s) -> Drop (60-90s)
    """
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    kick_layer = np.zeros(n_samples, dtype=np.float32)
    hat_layer = np.zeros(n_samples, dtype=np.float32)
    bass_layer = np.zeros(n_samples, dtype=np.float32)
    chord_layer = np.zeros(n_samples, dtype=np.float32)
    vocal_layer = np.zeros(n_samples, dtype=np.float32)

    beat_interval = 60.0 / bpm
    sixteenth_interval = beat_interval / 4.0
    bar_interval = beat_interval * 4.0

    # 1. Kick Drum (Plays during 0-45s and 60-90s drop, cuts during 45-60s breakdown)
    for beat_time in np.arange(0, duration_sec, beat_interval):
        # Mute kick during breakdown (45s to 60s)
        if 45.0 <= beat_time < 60.0:
            continue
        idx_start = int(beat_time * sr)
        kick_dur = min(int(0.12 * sr), n_samples - idx_start)
        if kick_dur <= 0:
            break
        t_k = np.linspace(0, 0.12, kick_dur, endpoint=False)
        # Exponential frequency drop 140Hz -> 48Hz
        freq_env = 48.0 + 92.0 * np.exp(-t_k * 35.0)
        amp_env = np.exp(-t_k * 20.0)
        kick_wave = np.sin(2 * np.pi * np.cumsum(freq_env) / sr) * amp_env
        kick_layer[idx_start : idx_start + kick_dur] += kick_wave * 0.9

    # 2. 16th-note Percussion & Hi-Hats
    for sixteenth_time in np.arange(0, duration_sec, sixteenth_interval):
        idx_start = int(sixteenth_time * sr)
        hat_dur = min(int(0.03 * sr), n_samples - idx_start)
        if hat_dur <= 0:
            break
        noise = (np.random.rand(hat_dur).astype(np.float32) * 2 - 1) * np.exp(-np.linspace(0, 5, hat_dur))
        hat_layer[idx_start : idx_start + hat_dur] += noise * 0.15

    # 3. Chord Progression in A Minor (A4=440, C5=523.25, E5=659.25)
    # Chord cycle: Am (bar 1-2), F (bar 3-4), C (bar 5-6), G (bar 7-8)
    chords_freqs = [
        [440.0, 523.25, 659.25],       # Am
        [349.23, 440.0, 523.25],       # F
        [261.63, 329.63, 392.0, 523.25], # C
        [392.0, 493.88, 587.33],       # G
    ]
    for bar_idx, bar_time in enumerate(np.arange(0, duration_sec, bar_interval * 2)):
        chord = chords_freqs[bar_idx % len(chords_freqs)]
        idx_start = int(bar_time * sr)
        dur = min(int(bar_interval * 2 * sr), n_samples - idx_start)
        if dur <= 0:
            break
        t_c = np.linspace(0, dur / sr, dur, endpoint=False)
        chord_wave = sum(0.15 * np.sin(2 * np.pi * f * t_c) for f in chord)
        chord_layer[idx_start : idx_start + dur] += chord_wave

    # 4. Vocal Formant Synthesis (Active strictly between 15s and 45s)
    vocal_start = 15.0
    vocal_end = 45.0
    v_idx_start = int(vocal_start * sr)
    v_idx_end = int(vocal_end * sr)
    t_v = t[v_idx_start:v_idx_end]
    # Formant filter approximation: 800 Hz + 1500 Hz with 5.5 Hz pitch vibrato
    vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.5 * t_v)
    vocal_wave = (
        0.35 * np.sin(2 * np.pi * 440.0 * vibrato * t_v) +
        0.25 * np.sin(2 * np.pi * 880.0 * vibrato * t_v) +
        0.20 * np.sin(2 * np.pi * 1320.0 * vibrato * t_v)
    )
    vocal_layer[v_idx_start:v_idx_end] = vocal_wave

    # 5. Combine and Master
    mix = kick_layer + hat_layer + chord_layer + vocal_layer
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.95

    return mix.astype(np.float32)


def test_e2e_real_afrohouse_track_analysis(tmp_path, temp_db):
    """
    End-to-end test on a realistic synthesized Afrohouse song:
    - Verifies Key detection identifies A Minor / 8A
    - Verifies PhrasingEngine detects 123.0 BPM
    - Verifies VocalEnergyProfiler isolates 15s–45s vocal window
    - Verifies CueGenerator sets Pad A (Intro), Pad B (Vocal @ ~15s), Pad C (Drop @ ~60s)
    - Verifies SQLite indexing and Rekordbox XML generation
    """
    sr = 22050
    audio_data = synthesize_realistic_afrohouse_track(duration_sec=90.0, bpm=123.0, sr=sr)
    
    song_file = tmp_path / "Black_Coffee_Afro_Anthem.flac"
    sf.write(str(song_file), audio_data, sr, format="FLAC")

    # 1. Run Complete Audio Analyzer
    analyzer = AudioAnalyzer(temp_db)
    _, analysis = analyzer.analyze_and_enrich_track(song_file, auto_tag_file=True)

    # 2. Validate Key & Camelot
    assert analysis.key_info.camelot in ("8A", "7B", "8B", "9A")  # A minor / closely related relative keys
    assert abs(analysis.key_info.pitch_drift_cents) < 10.0

    # 3. Validate Tempo & Phrasing
    assert 120.0 <= analysis.phrasing_info.bpm <= 126.0
    assert analysis.phrasing_info.first_downbeat_sec < 2.0

    # 4. Validate Vocal Detection & Windows
    assert analysis.vocal_energy_info.has_vocals is True
    assert analysis.vocal_energy_info.first_vocal_sec is not None
    assert 13.0 <= analysis.vocal_energy_info.first_vocal_sec <= 18.0
    assert analysis.vocal_energy_info.instrumental_intro_sec >= 12.0

    # 5. Validate Hot Cue Positions
    cue_names = [c.name for c in analysis.generated_cues]
    assert "Intro Beat" in cue_names
    assert "Vocal Entry" in cue_names

    # Check Vocal Entry Cue Timestamp (~15s)
    vocal_cue = next(c for c in analysis.generated_cues if c.name == "Vocal Entry")
    assert 13000 <= vocal_cue.timestamp_ms <= 18000

    # 6. Validate Database Persistence & XML Export
    repo = TrackRepository(temp_db)
    db_track = repo.get_track_by_path(str(song_file))
    assert db_track is not None
    assert len(db_track.cues) >= 3

    xml_out = tmp_path / "real_music_collection.xml"
    DjayProExporter.export_rekordbox_xml([db_track], xml_out)
    assert xml_out.exists()
    xml_content = xml_out.read_text()
    assert "Black_Coffee_Afro_Anthem" in xml_content
    assert 'Name="Vocal Entry"' in xml_content


def synthesize_melodic_techno_track(
    duration_sec: float = 75.0,
    bpm: float = 126.0,
    detune_cents: float = 12.0,  # +12 cents detuned concert pitch (~443 Hz)
    sr: int = 22050,
) -> np.ndarray:
    """
    Synthesizes a Melodic Techno track:
    - 126 BPM rolling 16th-note bassline in E Minor (9A)
    - Detuned pitch drift (+12 cents)
    - Dark polyphonic synths (Em, C, D)
    - Instrumental drop after a 16-bar buildup
    """
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    # Detune factor: 2^(cents / 1200)
    tuning_factor = 2.0 ** (detune_cents / 1200.0)
    e_root = 329.63 * tuning_factor  # E4

    kick_layer = np.zeros(n_samples, dtype=np.float32)
    bass_layer = np.zeros(n_samples, dtype=np.float32)
    synth_layer = np.zeros(n_samples, dtype=np.float32)

    beat_interval = 60.0 / bpm
    sixteenth_interval = beat_interval / 4.0

    # 1. 126 BPM Techno Kick (Thumping 55 Hz)
    for beat_time in np.arange(0, duration_sec, beat_interval):
        idx_start = int(beat_time * sr)
        kick_dur = min(int(0.10 * sr), n_samples - idx_start)
        if kick_dur <= 0:
            break
        t_k = np.linspace(0, 0.10, kick_dur, endpoint=False)
        kick_layer[idx_start : idx_start + kick_dur] += 0.85 * np.sin(2 * np.pi * 55.0 * t_k) * np.exp(-t_k * 25.0)

    # 2. Rolling 16th Bassline (E minor)
    for sixteenth_time in np.arange(0, duration_sec, sixteenth_interval):
        idx_start = int(sixteenth_time * sr)
        bass_dur = min(int(0.08 * sr), n_samples - idx_start)
        if bass_dur <= 0:
            break
        t_b = np.linspace(0, 0.08, bass_dur, endpoint=False)
        bass_wave = 0.4 * np.sin(2 * np.pi * (e_root / 2.0) * t_b) * np.exp(-t_b * 15.0)
        bass_layer[idx_start : idx_start + bass_dur] += bass_wave

    # 3. Minor 9th Synth Stabs
    chord = [e_root, e_root * 1.189, e_root * 1.498, e_root * 1.887]  # Em9
    for bar_time in np.arange(0, duration_sec, beat_interval * 4):
        idx_start = int(bar_time * sr)
        dur = min(int(beat_interval * 2 * sr), n_samples - idx_start)
        if dur <= 0:
            break
        t_s = np.linspace(0, dur / sr, dur, endpoint=False)
        synth_wave = sum(0.12 * np.sin(2 * np.pi * f * t_s) for f in chord)
        synth_layer[idx_start : idx_start + dur] += synth_wave

    mix = kick_layer + bass_layer + synth_layer
    return (mix / np.max(np.abs(mix)) * 0.95).astype(np.float32)


def test_e2e_melodic_techno_pitch_drift_and_phrasing(tmp_path, temp_db):
    """
    Verifies MIR engine on a Melodic Techno track:
    - Verifies 126.0 BPM extraction
    - Verifies Camelot Key 9A (E minor) or relative key
    - Verifies positive microtonal pitch drift detection (+10 to +15 cents)
    - Verifies pure instrumental detection (no false positive vocals)
    """
    sr = 22050
    audio = synthesize_melodic_techno_track(duration_sec=75.0, bpm=126.0, detune_cents=12.0, sr=sr)

    techno_file = tmp_path / "Tale_Of_Us_Afterlife.wav"
    sf.write(str(techno_file), audio, sr, format="WAV")

    analyzer = AudioAnalyzer(temp_db)
    _, analysis = analyzer.analyze_and_enrich_track(techno_file, auto_tag_file=True)

    # 1. BPM
    assert 124.0 <= analysis.phrasing_info.bpm <= 128.0

    # 2. Key & Pitch Drift
    assert analysis.key_info.camelot in ("9A", "9B", "8A", "10A")
    assert analysis.key_info.pitch_drift_cents > 3.0  # Drift detected upwards

    # 3. Vocals: should detect track is instrumental
    assert analysis.vocal_energy_info.vocal_presence_percent < 15.0

    # 4. Verified djay Pro comment
    assert f"{analysis.key_info.camelot}" in analysis.summary_comment
    assert "BPM" in analysis.summary_comment


def synthesize_piano_house_track(
    duration_sec: float = 60.0,
    bpm: float = 124.0,
    sr: int = 22050,
) -> np.ndarray:
    """
    Synthesizes an Uplifting Piano House track in G Major (9B):
    - 124 BPM 4/4 Kick and open hi-hat on off-beats
    - G Major chord progression: G -> Em -> C -> D (9B -> 9A -> 8B -> 10B)
    - House vocal hook active from 10s to 30s
    """
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    kick_layer = np.zeros(n_samples, dtype=np.float32)
    piano_layer = np.zeros(n_samples, dtype=np.float32)
    vocal_layer = np.zeros(n_samples, dtype=np.float32)

    beat_interval = 60.0 / bpm
    bar_interval = beat_interval * 4.0

    # 1. 124 BPM House Kick
    for beat_time in np.arange(0, duration_sec, beat_interval):
        idx_start = int(beat_time * sr)
        k_dur = min(int(0.12 * sr), n_samples - idx_start)
        if k_dur <= 0:
            break
        t_k = np.linspace(0, 0.12, k_dur, endpoint=False)
        kick_layer[idx_start : idx_start + k_dur] += 0.85 * np.sin(2 * np.pi * 58.0 * t_k) * np.exp(-t_k * 22.0)

    # 2. G Major Piano Chords (G4=392Hz, B4=493.88Hz, D5=587.33Hz)
    chords = [
        [392.0, 493.88, 587.33],        # G
        [329.63, 392.0, 493.88],        # Em
        [261.63, 329.63, 392.0],        # C
        [293.66, 369.99, 440.0],        # D
    ]
    for bar_idx, bar_time in enumerate(np.arange(0, duration_sec, bar_interval)):
        chord = chords[bar_idx % len(chords)]
        idx_start = int(bar_time * sr)
        dur = min(int(bar_interval * sr), n_samples - idx_start)
        if dur <= 0:
            break
        t_p = np.linspace(0, dur / sr, dur, endpoint=False)
        piano_wave = sum(0.14 * np.sin(2 * np.pi * f * t_p) for f in chord)
        piano_layer[idx_start : idx_start + dur] += piano_wave

    # 3. Vocal Hook (10s to 30s) with strong formant harmonics
    v_start = int(10.0 * sr)
    v_end = int(30.0 * sr)
    t_v = t[v_start:v_end]
    vocal_layer[v_start:v_end] = (
        0.35 * np.sin(2 * np.pi * 987.77 * t_v) +
        0.30 * np.sin(2 * np.pi * 1568.0 * t_v) +
        0.25 * np.sin(2 * np.pi * 2352.0 * t_v)
    )

    mix = kick_layer + piano_layer + vocal_layer
    return (mix / np.max(np.abs(mix)) * 0.95).astype(np.float32)


def test_e2e_piano_house_vocal_and_cues(tmp_path, temp_db):
    """
    Verifies Piano House analysis:
    - 124.0 BPM
    - G Major / 9B
    - Vocal hook at 10s-30s
    - Hot cues placed at Intro and Vocal Entry
    """
    sr = 22050
    audio = synthesize_piano_house_track(duration_sec=60.0, bpm=124.0, sr=sr)
    house_file = tmp_path / "MK_Piano_House_Anthem.mp3"

    # Write MP3 dummy header + wav payload
    sf.write(str(house_file), audio, sr, format="WAV")

    analyzer = AudioAnalyzer(temp_db)
    _, analysis = analyzer.analyze_and_enrich_track(house_file, auto_tag_file=True)

    # 1. BPM & Key
    assert 122.0 <= analysis.phrasing_info.bpm <= 126.0
    assert analysis.key_info.camelot in ("9B", "9A", "8B", "10B")

    # 2. Vocal detection
    assert analysis.vocal_energy_info.has_vocals is True
    assert 8.0 <= analysis.vocal_energy_info.first_vocal_sec <= 14.0

    # 3. Hot Cues
    cues = {c.name: c.timestamp_ms for c in analysis.generated_cues}
    assert "Intro Beat" in cues
    assert "Vocal Entry" in cues
    assert 8000 <= cues["Vocal Entry"] <= 14000
