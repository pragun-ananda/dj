from sonicdj.scanner.file_scanner import LibraryScanner
from sonicdj.db.repository import TrackRepository


def test_library_scanner_recursive(temp_db, temp_audio_dir):
    scanner = LibraryScanner(temp_db)
    repo = TrackRepository(temp_db)

    # Initial scan of test directory
    stats = scanner.scan_directory(temp_audio_dir)
    assert stats.total_found == 2
    assert stats.added == 2
    assert stats.failed == 0

    # Verify indexed in DB
    tracks, total = repo.list_tracks()
    assert total == 2

    # Second scan should update, not duplicate
    stats2 = scanner.scan_directory(temp_audio_dir)
    assert stats2.total_found == 2
    assert stats2.updated == 2
    assert stats2.added == 0
    assert repo.count_all() == 2
