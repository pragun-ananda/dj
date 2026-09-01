from pathlib import Path
from sonicdj.ingestion.tidal_client import TidalTrackInfo
from sonicdj.ingestion.downloader import TidalDownloader
from sonicdj.db.repository import TrackRepository


def test_batch_download_playlist(temp_db, tmp_path):
    downloader = TidalDownloader(db=temp_db, output_dir=tmp_path)
    repo = TrackRepository(temp_db)

    playlist = [
        TidalTrackInfo(
            id="101",
            title="Track Alpha",
            artist="Artist 1",
            album="Album 1",
            duration_sec=300.0,
            audio_quality="LOSSLESS",
        ),
        TidalTrackInfo(
            id="102",
            title="Track Beta",
            artist="Artist 2",
            album="Album 2",
            duration_sec=320.0,
            audio_quality="LOSSLESS",
        ),
    ]

    progress_records = []
    def on_progress(current, total, name):
        progress_records.append((current, total, name))

    paths = downloader.batch_download_playlist(playlist, progress_callback=on_progress)
    assert len(paths) == 2
    assert len(progress_records) == 2
    assert progress_records[0] == (1, 2, "Artist 1 - Track Alpha")
    assert progress_records[1] == (2, 2, "Artist 2 - Track Beta")

    assert repo.count_all() == 2
