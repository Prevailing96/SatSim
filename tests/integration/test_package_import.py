"""Integration: package import surface and CLI help."""

from __future__ import annotations

import pytest

import satsim
from satsim.cli import main


@pytest.mark.integration
def test_version_defined() -> None:
    assert satsim.__version__


@pytest.mark.integration
def test_cli_info(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["info"])
    assert code == 0
    out = capsys.readouterr().out
    assert "SatSim" in out


@pytest.mark.integration
def test_cli_run_dry(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--config", "configs/default.yaml", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
