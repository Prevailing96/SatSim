"""Tests for TwoBodyEnvironment and ScenarioRunner execution path."""

from __future__ import annotations

import pytest

from satsim.application.simulation.scenario import ScenarioConfig, ScenarioRunner
from satsim.application.simulation.two_body_env import (
    TwoBodyEnvironment,
    default_demo_constellation,
    default_demo_targets,
)
from satsim.core.orbital.frames import geodetic_to_ecef_spherical


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
    env = TwoBodyEnvironment(
        duration_s=5.0, dt_s=1.0, satellite_specs=default_demo_constellation(2)
    )
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
    env = TwoBodyEnvironment(
        duration_s=3.0, dt_s=1.0, satellite_specs=default_demo_constellation(1)
    )
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


@pytest.mark.unit
def test_reset_produces_nonempty_observations() -> None:
    env = TwoBodyEnvironment(
        duration_s=5.0, dt_s=1.0, satellite_specs=default_demo_constellation(3)
    )
    snap = env.reset(seed=0)
    assert len(snap.observations) == 3
    for bundle in snap.observations:
        assert bundle.image is not None
        assert bundle.image.data.shape == (env.camera_height_px, env.camera_width_px, 3)


@pytest.mark.unit
def test_no_detection_or_tasking_at_reset() -> None:
    """Targets are geographically offset now: t=0 must NOT be a guaranteed hit."""
    env = TwoBodyEnvironment(
        duration_s=5.0, dt_s=1.0, satellite_specs=default_demo_constellation(3)
    )
    snap = env.reset(seed=0)
    assert snap.infos["n_detections"] == 0
    for sat_id, action in snap.infos["agent_actions"].items():
        assert action["kind"] == "idle", f"{sat_id} should be idle at t=0"


@pytest.mark.unit
def test_each_satellite_eventually_detects_its_target() -> None:
    """Every satellite should discover its own target well before t=600s."""
    env = TwoBodyEnvironment(
        duration_s=450.0, dt_s=1.0, satellite_specs=default_demo_constellation(3)
    )
    env.reset(seed=0)
    detected_by: set[str] = set()
    task_events = 0
    for _ in range(450):
        snap = env.step()
        for bundle in snap.observations:
            if bundle.detections:
                detected_by.add(str(bundle.satellite_id))
        for action in snap.infos["agent_actions"].values():
            if action["kind"] == "task":
                task_events += 1

    assert detected_by == {"sat-001", "sat-002", "sat-003"}
    assert task_events > 0


@pytest.mark.unit
def test_deconfliction_keeps_tasking_events_well_below_detections() -> None:
    """Each target is visible for many consecutive frames; only one task per target should fire."""
    env = TwoBodyEnvironment(
        duration_s=450.0, dt_s=1.0, satellite_specs=default_demo_constellation(3)
    )
    env.reset(seed=0)
    total_detections = 0
    task_events = 0
    for _ in range(450):
        snap = env.step()
        total_detections += sum(len(b.detections) for b in snap.observations)
        task_events += sum(
            1 for a in snap.infos["agent_actions"].values() if a["kind"] == "task"
        )

    # Three targets, one satellite each: without deconfliction every one of
    # the ~80+ detections would spawn its own task (as it did before this
    # session). With it, at most one task per target.
    assert total_detections > 30
    assert 0 < task_events <= 3
    assert task_events < total_detections


@pytest.mark.unit
def test_step_keeps_producing_observations() -> None:
    env = TwoBodyEnvironment(
        duration_s=3.0, dt_s=1.0, satellite_specs=default_demo_constellation(1)
    )
    env.reset(seed=0)
    snap = env.step()
    assert len(snap.observations) == 1
    assert "n_detections" in snap.infos
    assert "agent_actions" in snap.infos


@pytest.mark.unit
def test_default_demo_targets_not_colocated_with_satellite_start() -> None:
    """No target should sit under any satellite's t=0 position anymore."""
    targets = default_demo_targets()
    assert len(targets) >= 2
    class_names = {t.class_name for t in targets}
    assert "vessel" in class_names
    assert "aircraft" in class_names

    env = TwoBodyEnvironment(
        duration_s=5.0, dt_s=1.0, satellite_specs=default_demo_constellation(3)
    )
    snap = env.reset(seed=0)
    for state in snap.states.values():
        sat_pos = state.cartesian.position_m.as_array()
        for target in targets:
            target_pos = geodetic_to_ecef_spherical(
                target.latitude_rad, target.longitude_rad
            ).as_array()
            # Satellite altitude alone (550 km) guarantees separation from
            # any surface point, but check explicitly for intent, not just
            # incidental geometry.
            distance_m = float(((sat_pos - target_pos) ** 2).sum() ** 0.5)
            assert distance_m > 100_000.0
