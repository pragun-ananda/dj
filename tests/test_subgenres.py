from sonicdj.metadata.subgenres import SubgenreClassifier


def test_subgenre_classification():
    # 1. Afro House
    res1 = SubgenreClassifier.classify(bpm=123.0, camelot="8A", energy=0.85, has_vocals=False)
    assert res1.primary_subgenre == "Afro House"
    assert "driving" in res1.moods

    # 2. Melodic Techno
    res2 = SubgenreClassifier.classify(bpm=127.0, camelot="9A", energy=0.88, has_vocals=False)
    assert res2.primary_subgenre == "Melodic Techno"
    assert "dark" in res2.moods

    # 3. Vocal House
    res3 = SubgenreClassifier.classify(bpm=124.0, camelot="8B", energy=0.80, has_vocals=True)
    assert res3.primary_subgenre == "Vocal House"
    assert "euphoric" in res3.moods

    # 4. Amapiano
    res4 = SubgenreClassifier.classify(bpm=115.0, camelot="8A", energy=0.70, has_vocals=True)
    assert res4.primary_subgenre == "Amapiano"

    # 5. Drum & Bass
    res5 = SubgenreClassifier.classify(bpm=174.0, camelot="4A", energy=0.95, has_vocals=False)
    assert res5.primary_subgenre == "Drum & Bass"
