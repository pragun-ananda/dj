from typer.testing import CliRunner
from sonicdj.cli import app
from sonicdj.db.repository import DatabaseManager, TrackRepository

runner = CliRunner()


def test_cli_search(temp_db):
    repo = TrackRepository(temp_db)
    repo.upsert_track({
        "file_path": "/music/afro.wav", "file_hash": "hash1", "file_size_bytes": 1000,
        "format": "wav", "duration_sec": 180.0, "title": "Afro Groove", "artist": "Black Coffee",
        "bpm": 123.0, "camelot": "8A", "energy": 0.85, "comments": "Peak vocal energy"
    })

    db_file = temp_db.db_url.replace("sqlite:///", "")

    # Test search command
    result = runner.invoke(app, ["search", "afrohouse groove", "--key", "8A", "--bpm", "123", "--db", db_file])
    assert result.exit_code == 0
    assert "Semantic & Harmonic DJ Search" in result.output
    assert "Afro Groove" in result.output

    # Test empty query / open search
    res_empty = runner.invoke(app, ["search", "--db", db_file])
    assert res_empty.exit_code == 0
