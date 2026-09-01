import xml.etree.ElementTree as ET
from pathlib import Path
from sonicdj.db.schema import Track, CuePoint, Playlist
from sonicdj.metadata.djay_exporter import DjayProExporter


def test_rekordbox_xml_export(tmp_path):
    t1 = Track(
        id=1,
        file_path="/Music/Track1.mp3",
        file_hash="hash1",
        title="Deep Rhythm",
        artist="DJ Pulse",
        genre="Afro House",
        bpm=124.0,
        camelot="8A",
        key_raw="Am",
        rating=4,
        duration_sec=320.0,
        comments="Great breakdown",
        cues=[
            CuePoint(name="Intro", timestamp_ms=0, hot_cue_index=0),
            CuePoint(name="Drop", timestamp_ms=45000, hot_cue_index=1),
        ],
    )
    
    xml_out = tmp_path / "rekordbox_test.xml"
    DjayProExporter.export_rekordbox_xml([t1], xml_out)

    assert xml_out.exists()
    
    # Parse and validate XML structure
    tree = ET.parse(xml_out)
    root = tree.getroot()
    assert root.tag == "DJ_PLAYLISTS"
    
    collection = root.find("COLLECTION")
    assert collection is not None
    assert collection.attrib["Entries"] == "1"

    track_elem = collection.find("TRACK")
    assert track_elem is not None
    assert track_elem.attrib["Name"] == "Deep Rhythm"
    assert track_elem.attrib["Tonality"] == "8A"
    assert track_elem.attrib["AverageBpm"] == "124.00"

    # Check Position Marks (Cues)
    cues = track_elem.findall("POSITION_MARK")
    assert len(cues) == 2
    assert cues[0].attrib["Name"] == "Intro"
    assert cues[1].attrib["Name"] == "Drop"
    assert cues[1].attrib["Start"] == "45.000"


def test_m3u8_export(tmp_path):
    t1 = Track(
        id=1,
        file_path="/Music/Track1.mp3",
        title="Deep Rhythm",
        artist="DJ Pulse",
        bpm=124.0,
        camelot="8A",
        duration_sec=180.0,
    )
    m3u_out = tmp_path / "playlist.m3u8"
    DjayProExporter.export_m3u8([t1], m3u_out, playlist_name="Test Crate")

    assert m3u_out.exists()
    content = m3u_out.read_text()
    assert "#PLAYLIST:Test Crate" in content
    assert "#EXT-X-DJAY-KEY:8A" in content
    assert "/Music/Track1.mp3" in content
