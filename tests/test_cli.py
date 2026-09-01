from pathlib import Path
from typer.testing import CliRunner
from sonicdj.cli import app
from sonicdj.db.repository import DatabaseManager, TrackRepository

runner = CliRunner()


def test_cli_scan_and_list(temp_audio_dir, tmp_path):
    db_file = tmp_path / "cli_test.db"
    
    # Run scan
    scan_result = runner.invoke(app, ["scan", str(temp_audio_dir), "--db", str(db_file)])
    assert scan_result.exit_code == 0
    assert "Scan Completed Summary" in scan_result.output

    # Run list
    list_result = runner.invoke(app, ["list"])
    assert list_result.exit_code == 0


def test_cli_info_and_tag(temp_audio_dir):
    wav_file = temp_audio_dir / "track_test.wav"

    # Info command
    info_result = runner.invoke(app, ["info", str(wav_file)])
    assert info_result.exit_code == 0
    assert "Metadata: track_test.wav" in info_result.output

    # Tag command
    tag_result = runner.invoke(app, [
        "tag", str(wav_file),
        "--key", "8A",
        "--bpm", "124.5",
        "--energy", "0.9",
        "--rating", "5",
        "--comment", "Peak time banger"
    ])
    assert tag_result.exit_code == 0
    assert "Successfully tagged" in tag_result.output


def test_cli_export_djay(temp_audio_dir, tmp_path):
    # Scan first
    runner.invoke(app, ["scan", str(temp_audio_dir)])
    
    xml_out = tmp_path / "export_test.xml"
    export_result = runner.invoke(app, ["export-djay", str(xml_out), "--m3u8"])
    assert export_result.exit_code == 0
    assert xml_out.exists()


def test_cli_invalid_file():
    result = runner.invoke(app, ["info", "/non/existent/file.wav"])
    assert result.exit_code == 1
