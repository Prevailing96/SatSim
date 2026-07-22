"""Tests for YAML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from satsim.infrastructure.io.config import load_scenario_config, load_yaml


@pytest.mark.unit
def test_load_default_scenario(configs_dir: Path) -> None:
    path = configs_dir / "default.yaml"
    cfg = load_scenario_config(path)
    assert cfg.name == "default_leo_demo"
    assert cfg.duration_s == pytest.approx(600.0)
    assert cfg.dt_s == pytest.approx(1.0)
    assert cfg.seed == 42
    assert cfg.num_steps == 600


@pytest.mark.unit
def test_load_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


@pytest.mark.unit
def test_scenario_runner_dry_run(configs_dir: Path) -> None:
    from satsim.application.simulation.scenario import ScenarioRunner

    cfg = load_scenario_config(configs_dir / "simulation" / "fast_smoke.yaml")
    runner = ScenarioRunner(cfg)
    history = runner.run(dry_run=True)
    assert history == []
    desc = runner.describe()
    assert desc["name"] == "fast_smoke"
