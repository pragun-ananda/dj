import numpy as np
from sonicdj.embeddings.audio_encoder import MultimodalAudioEncoder


def test_audio_encoder_basic():
    sr = 22050
    audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 2, sr * 2, endpoint=False)).astype(np.float32)
    
    vec = MultimodalAudioEncoder.encode_audio(
        audio, sr=sr, bpm=124.0, camelot="8A", energy=0.85, has_vocals=True
    )
    assert vec.shape == (512,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    # Empty audio fallback
    empty_vec = MultimodalAudioEncoder.encode_audio(np.array([]))
    assert empty_vec.shape == (512,)
    assert abs(np.linalg.norm(empty_vec) - 1.0) < 1e-4


def test_text_encoder_and_similarity():
    # 1. Encode text queries
    vec_afro = MultimodalAudioEncoder.encode_text("dark hypnotic afrohouse with heavy bass")
    vec_techno = MultimodalAudioEncoder.encode_text("peak time driving techno drop")
    vec_empty = MultimodalAudioEncoder.encode_text("")

    assert vec_afro.shape == (512,)
    assert vec_techno.shape == (512,)
    assert abs(np.linalg.norm(vec_afro) - 1.0) < 1e-4

    # 2. Similarity
    sim_self = MultimodalAudioEncoder.compute_similarity(vec_afro, vec_afro)
    assert abs(sim_self - 1.0) < 1e-4

    sim_diff = MultimodalAudioEncoder.compute_similarity(vec_afro, vec_techno)
    assert -1.0 <= sim_diff <= 1.0
