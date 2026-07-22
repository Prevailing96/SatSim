"""Tests for TwoBodyEnvironment and ScenarioRunner execution path."""

from __future__ import annotations

import pytest

from satsim.application.simulation.scenario import ScenarioConfig, ScenarioRunner
from satsim.application.simulation.two_body_env import (
    TwoBodyEnvironment,
    default_demo_constellation,
)


@pytest.mark.unit
def test_default_demo_constellation_spacing() -> None:
    specs = default_demo_constellation(4)
    assert len(specs) == 4
    assert specs[0].satellite_id == "sat-001"
    anomalies = [s.true_anomaly_rad for s in specs]
    assert anomalies[0] == pytest.approx(0.0)
    assert anomalies[2] == pytest.approx(3.141592653589793, rel=1e-9)


@pytest.mark.unit
def test_environment_reset_and_step() -> None:
    env = TwoBodyEnvironment(duration_s=5.0, dt_s=1.0, satellite_specs=default_demo_constellation(2))
    snap0 = env.reset(seed=1)
    assert snap0.time.seconds == pytest.approx(0.0)
    assert len(snap0.states) == 2
    assert snap0.truncated is False

    snap1 = env.step()
    assert snap1.time.seconds == pytest.approx(1.0)
    assert env.step_index == 1
    # States moved
    s0 = snap0.states["sat-001"]
    s1 = snap1.states["sat-001"]
    assert s1.cartesian.position_m.x != s0.cartesian.position_m.x or (
        s1.cartesian.position_m.y != s0.cartesian.position_m.y
    )


@pytest.mark.unit
def test_environment_truncates_at_duration() -> None:
    env = TwoBodyEnvironment(duration_s=3.0, dt_s=1.0, satellite_specs=default_demo_constellation(1))
    env.reset()
    last = None
    for _ in range(3):
        last = env.step()
    assert last is not None
    assert last.truncated is True
    assert last.time.seconds == pytest.approx(3.0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step()


@pytest.mark.unit
def test_scenario_runner_executes() -> None:
    cfg = ScenarioConfig(name="unit_run", duration_s=5.0, dt_s=1.0, seed=7, n_satellites=2)
    runner = ScenarioRunner(cfg)
    history = runner.run(dry_run=False)
    # reset + 5 steps
    assert len(history) == 6
    assert history[0].time.seconds == pytest.approx(0.0)
    assert history[-1].time.seconds == pytest.approx(5.0)
    assert history[-1].truncated is True
    assert len(history[-1].states) == 2


@pytest.mark.unit
def test_scenario_runner_dry_run_empty() -> None:
    cfg = ScenarioConfig(duration_s=10.0, dt_s=1.0)
    history = ScenarioRunner(cfg).run(dry_run=True)
    assert history == []
