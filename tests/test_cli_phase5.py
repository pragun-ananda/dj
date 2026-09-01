from typer.testing import CliRunner
from sonicdj.cli import app

runner = CliRunner()


def test_cli_flx4_monitor():
    result = runner.invoke(app, ["flx4-monitor"])
    assert result.exit_code == 0
    assert "Pioneer DDJ-FLX4 Hardware Telemetry" in result.output
    assert "Crossfader:" in result.output
