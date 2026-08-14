"""Tests for tasking scheduler and satellite agent closed-loop hooks."""

from __future__ import annotations

import math

import pytest

from satsim.application.agents.base import AgentObservation
from satsim.application.agents.satellite_agent import (
    DEFAULT_DECONFLICTION_RADIUS_M,
    SatelliteAgent,
)
from satsim.application.tasking.requests import TaskPriority, TaskRequest, TaskStatus
from satsim.application.tasking.scheduler import PriorityQueueScheduler
from satsim.core.constants import R_EARTH_EQUATORIAL
from satsim.core.orbital.state import OrbitalState
from satsim.core.sensors.observations import Detection2D, ObservationBundle
from satsim.core.types.identifiers import AgentId, SatelliteId, SensorId
from satsim.core.types.time import SimTime
from satsim.core.types.vectors import CartesianState, Vec3


@pytest.mark.unit
def test_priority_scheduler_activates_highest() -> None:
    sched = PriorityQueueScheduler()
    low = TaskRequest(
        satellite_id=SatelliteId("s1"),
        priority=TaskPriority.LOW,
        request_id="low",
    )
    high = TaskRequest(
        satellite_id=SatelliteId("s1"),
        priority=TaskPriority.CRITICAL,
        request_id="high",
    )
    sched.submit(low)
    sched.submit(high)
    active = sched.tick(SimTime(0.0))
    assert len(active) == 1
    assert active[0].request_id == "high"
    assert active[0].status == TaskStatus.IN_PROGRESS


@pytest.mark.unit
def test_satellite_agent_tasks_on_detection() -> None:
    sat = SatelliteId("sat-1")
    agent = SatelliteAgent(
        AgentId("agent-1"),
        sat,
        detection_class_names=frozenset({"vessel"}),
        min_confidence=0.4,
    )
    det = Detection2D(
        class_id=1,
        class_name="vessel",
        confidence=0.9,
        x_min=10,
        y_min=10,
        x_max=40,
        y_max=40,
    )
    bundle = ObservationBundle(
        sensor_id=SensorId("cam"),
        satellite_id=sat,
        time=SimTime(5.0),
        detections=(det,),
    )
    obs = AgentObservation(
        agent_id=agent.agent_id,
        time=SimTime(5.0),
        bundles=(bundle,),
    )
    action = agent.act(obs)
    assert action.kind == "task"
    assert action.payload["submitted_request_ids"]


@pytest.mark.unit
def test_satellite_agent_idle_without_detections() -> None:
    agent = SatelliteAgent(AgentId("a"), SatelliteId("s"))
    obs = AgentObservation(agent_id=agent.agent_id, time=SimTime(0.0))
    action = agent.act(obs)
    assert action.kind == "idle"


_ORBIT_RADIUS_M = 7_000_000.0


def _state_at_angle(satellite_id: str, angle_rad: float, time_s: float = 0.0) -> OrbitalState:
    """Orbital state on a circle of radius ``_ORBIT_RADIUS_M`` at ``angle_rad``.

    ``approximate_target_location`` only cares about *direction*, not
    magnitude (it projects onto Earth's surface), so varying the angle
    rather than the radius is what actually separates two proxy locations.
    """
    return OrbitalState(
        satellite_id=SatelliteId(satellite_id),
        time=SimTime(time_s),
        cartesian=CartesianState(
            position_m=Vec3(
                _ORBIT_RADIUS_M * math.cos(angle_rad), _ORBIT_RADIUS_M * math.sin(angle_rad), 0.0
            ),
            velocity_m_s=Vec3(0.0, 0.0, 0.0),
        ),
    )


def _angle_for_ground_distance(distance_m: float) -> float:
    """Central angle giving a great-circle chord of ``distance_m`` on Earth."""
    ratio = min(distance_m / (2.0 * R_EARTH_EQUATORIAL), 1.0)
    return 2.0 * math.asin(ratio)


def _detection_observation(
    agent: SatelliteAgent, own_state: OrbitalState, *, time_s: float, sensor_id: str = "cam"
) -> AgentObservation:
    det = Detection2D(
        class_id=0,
        class_name="vessel",
        confidence=0.9,
        x_min=10,
        y_min=10,
        x_max=20,
        y_max=20,
    )
    bundle = ObservationBundle(
        sensor_id=SensorId(sensor_id),
        satellite_id=own_state.satellite_id,
        time=SimTime(time_s),
        detections=(det,),
    )
    return AgentObservation(
        agent_id=agent.agent_id,
        time=SimTime(time_s),
        own_state=own_state,
        bundles=(bundle,),
    )


def _agent(
    agent_id: str, sat_id: str, scheduler: PriorityQueueScheduler | None = None
) -> SatelliteAgent:
    return SatelliteAgent(
        AgentId(agent_id),
        SatelliteId(sat_id),
        detection_class_names=frozenset({"vessel"}),
        scheduler=scheduler,
    )


@pytest.mark.unit
def test_shared_scheduler_deconflicts_two_agents_same_target() -> None:
    """Two agents sharing a scheduler must not both task the same nearby target."""
    shared = PriorityQueueScheduler()
    agent_a = _agent("sat-a", "sat-a", shared)
    agent_b = _agent("sat-b", "sat-b", shared)

    # Two different satellites, close together in angle (well inside the
    # default deconfliction radius): they're plausibly looking at the same
    # ground target.
    close_angle = _angle_for_ground_distance(50_000.0)
    action_a = agent_a.act(
        _detection_observation(agent_a, _state_at_angle("sat-a", 0.0), time_s=0.0)
    )
    action_b = agent_b.act(
        _detection_observation(agent_b, _state_at_angle("sat-b", close_angle), time_s=1.0)
    )

    assert action_a.kind == "task"
    assert action_a.payload["submitted_request_ids"]
    assert action_b.kind == "idle"
    assert action_b.payload == {}
    assert len(shared.pending()) == 1


@pytest.mark.unit
def test_agent_does_not_retask_same_target_across_consecutive_detections() -> None:
    """The same satellite re-detecting one target repeatedly should task it once."""
    agent = _agent("sat-1", "sat-1")
    state = _state_at_angle("sat-1", 0.0)

    first = agent.act(_detection_observation(agent, state, time_s=0.0))
    second = agent.act(_detection_observation(agent, state, time_s=1.0))
    third = agent.act(_detection_observation(agent, state, time_s=2.0))

    assert first.kind == "task"
    assert first.payload["submitted_request_ids"]
    assert second.kind == "idle"
    assert third.kind == "idle"
    assert len(agent.scheduler.pending()) == 1


@pytest.mark.unit
def test_far_apart_targets_are_both_tasked() -> None:
    """Deconfliction must not suppress genuinely different, distant targets."""
    shared = PriorityQueueScheduler()
    agent_a = _agent("sat-a", "sat-a", shared)
    agent_b = _agent("sat-b", "sat-b", shared)

    # Opposite sides of the planet — nothing close to the same target.
    action_a = agent_a.act(
        _detection_observation(agent_a, _state_at_angle("sat-a", 0.0), time_s=0.0)
    )
    action_b = agent_b.act(
        _detection_observation(agent_b, _state_at_angle("sat-b", math.pi), time_s=1.0)
    )

    assert action_a.kind == "task"
    assert action_b.kind == "task"
    assert len(shared.pending()) == 2


@pytest.mark.unit
def test_deconfliction_radius_boundary() -> None:
    """A target just inside the radius is deconflicted; just outside, it is not."""
    radius = DEFAULT_DECONFLICTION_RADIUS_M

    shared_inside = PriorityQueueScheduler()
    agent_1 = _agent("a", "a", shared_inside)
    agent_2 = _agent("b", "b", shared_inside)
    agent_1.act(_detection_observation(agent_1, _state_at_angle("a", 0.0), time_s=0.0))
    inside_angle = _angle_for_ground_distance(radius * 0.5)
    inside_action = agent_2.act(
        _detection_observation(agent_2, _state_at_angle("b", inside_angle), time_s=1.0)
    )
    assert inside_action.kind == "idle"

    shared_outside = PriorityQueueScheduler()
    agent_3 = _agent("c", "c", shared_outside)
    agent_4 = _agent("d", "d", shared_outside)
    agent_3.act(_detection_observation(agent_3, _state_at_angle("c", 0.0), time_s=0.0))
    outside_angle = _angle_for_ground_distance(radius * 2.0)
    outside_action = agent_4.act(
        _detection_observation(agent_4, _state_at_angle("d", outside_angle), time_s=1.0)
    )
    assert outside_action.kind == "task"


@pytest.mark.unit
def test_private_schedulers_do_not_share_deconfliction_state() -> None:
    """Agents built without an explicit scheduler are only self-aware, not constellation-aware."""
    agent_a = _agent("sat-a", "sat-a")
    agent_b = _agent("sat-b", "sat-b")
    assert agent_a.scheduler is not agent_b.scheduler

    state = _state_at_angle("sat-a", 0.0)
    action_a = agent_a.act(_detection_observation(agent_a, state, time_s=0.0))
    action_b = agent_b.act(_detection_observation(agent_b, state, time_s=1.0))

    # Same location, but each agent only ever consults its own scheduler.
    assert action_a.kind == "task"
    assert action_b.kind == "task"
