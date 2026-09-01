from pathlib import Path
from sonicdj.ingestion.tidal_client import TidalTrackInfo, TidalClient
from sonicdj.ingestion.downloader import TidalDownloader, sanitize_filename
from sonicdj.db.repository import TrackRepository


def test_sanitize_filename():
    assert sanitize_filename('AC/DC: "Thunder"') == "ACDC Thunder"
    assert sanitize_filename("Artist / Track <Remix>") == "Artist  Track Remix"


def test_tidal_downloader_mock(temp_db, tmp_path):
    downloader = TidalDownloader(db=temp_db, output_dir=tmp_path)
    repo = TrackRepository(temp_db)

    track_info = TidalTrackInfo(
        id="123456",
        title="Breathe",
        artist="CamelPhat",
        album="Dark Matter",
        duration_sec=314.0,
        isrc="GBBKS1900123",
        audio_quality="LOSSLESS",
    )

    downloaded_file = downloader.download_track(track_info)
    assert downloaded_file.exists()
    assert "CamelPhat" in str(downloaded_file)

    # Verify track was indexed in SQLite
    tracks, total = repo.list_tracks(search_query="Breathe")
    assert total == 1
    assert tracks[0].artist == "CamelPhat"
    assert tracks[0].title == "Breathe"
