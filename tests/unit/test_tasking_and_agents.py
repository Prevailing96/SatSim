"""Tests for tasking scheduler and satellite agent closed-loop hooks."""

from __future__ import annotations

import pytest

from satsim.application.agents.base import AgentObservation
from satsim.application.agents.satellite_agent import SatelliteAgent
from satsim.application.tasking.requests import TaskPriority, TaskRequest, TaskStatus
from satsim.application.tasking.scheduler import PriorityQueueScheduler
from satsim.core.sensors.observations import Detection2D, ObservationBundle
from satsim.core.types.identifiers import AgentId, SatelliteId, SensorId
from satsim.core.types.time import SimTime


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
