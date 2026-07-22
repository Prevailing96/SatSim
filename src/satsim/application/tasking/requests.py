"""Tasking request value objects.

Task requests are the bridge between mission goals / CV events and the
scheduler that assigns sensors and collection windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.time import SimTime


class TaskPriority(int, Enum):
    """Discrete priority levels for observation / action tasks."""

    LOW = 10
    NORMAL = 50
    HIGH = 80
    CRITICAL = 100


class TaskStatus(str, Enum):
    """Lifecycle status of a task request."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TaskRequest:
    """A request to collect data or perform a sensor action.

    Attributes:
        request_id: Stable unique id (auto-generated if omitted).
        satellite_id: Preferred or required host satellite.
        sensor_id: Preferred sensor (optional; scheduler may choose).
        priority: Discrete priority.
        earliest: Earliest acceptable start time.
        latest: Latest acceptable start time (deadline window).
        target_description: Human / structured target hint (lat/lon, NORAD,
            image crop id, etc.).
        status: Current lifecycle status.
        metadata: Extensible payload (e.g. CV trigger detection id).
    """

    satellite_id: SatelliteId
    priority: TaskPriority = TaskPriority.NORMAL
    sensor_id: SensorId | None = None
    earliest: SimTime | None = None
    latest: SimTime | None = None
    target_description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    request_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["TaskPriority", "TaskRequest", "TaskStatus"]
