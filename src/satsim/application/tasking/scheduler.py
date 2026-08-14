"""Task scheduler protocol and naive stub implementation.

Advanced multi-satellite schedulers (constraint programming, auctions,
RL policies) should implement :class:`TaskScheduler` without changing
downstream consumers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from satsim.application.tasking.requests import TaskRequest, TaskStatus
from satsim.core.types.time import SimTime
from satsim.core.types.vectors import Vec3

#: Non-terminal task statuses — still "active" from a deconfliction
#: standpoint, whether or not they've started executing yet.
_ACTIVE_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.SCHEDULED, TaskStatus.IN_PROGRESS})


@runtime_checkable
class TaskScheduler(Protocol):
    """Assigns and updates task requests over time."""

    def submit(self, request: TaskRequest) -> None:
        """Enqueue a new task request.

        Args:
            request: Task to schedule.
        """
        ...

    def tick(self, time: SimTime) -> list[TaskRequest]:
        """Advance scheduler state to ``time``.

        Args:
            time: Current simulation time.

        Returns:
            Tasks that became active (``IN_PROGRESS``) this tick.
        """
        ...

    def pending(self) -> list[TaskRequest]:
        """Return currently pending / scheduled tasks (copy-safe list)."""
        ...


class PriorityQueueScheduler:
    """Simple priority-ordered FIFO scheduler (scaffold).

    Does not yet model slew constraints, power, or visibility windows — those
    land with the full constellation geometry stack.
    """

    def __init__(self) -> None:
        """Initialize empty queue."""
        self._queue: list[TaskRequest] = []

    def submit(self, request: TaskRequest) -> None:
        """Enqueue a request and mark it scheduled.

        Args:
            request: Incoming task.
        """
        request.status = TaskStatus.SCHEDULED
        self._queue.append(request)
        self._queue.sort(key=lambda r: (-int(r.priority), r.request_id))

    def tick(self, time: SimTime) -> list[TaskRequest]:
        """Activate the highest-priority ready task (at most one per tick).

        Args:
            time: Current time (used for earliest/latest window checks).

        Returns:
            Zero or one newly activated task.
        """
        activated: list[TaskRequest] = []
        for req in self._queue:
            if req.status != TaskStatus.SCHEDULED:
                continue
            if req.earliest is not None and time.seconds < req.earliest.seconds:
                continue
            if req.latest is not None and time.seconds > req.latest.seconds:
                req.status = TaskStatus.FAILED
                continue
            req.status = TaskStatus.IN_PROGRESS
            activated.append(req)
            break
        return activated

    def pending(self) -> list[TaskRequest]:
        """Return non-terminal tasks."""
        terminal = {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
        return [r for r in self._queue if r.status not in terminal]

    def find_active_near(self, location_m: Vec3, radius_m: float) -> TaskRequest | None:
        """Find a non-terminal task whose target is within ``radius_m``.

        This is the constellation's shared-awareness/deconfliction check:
        every :class:`~satsim.application.agents.satellite_agent.SatelliteAgent`
        given the *same* scheduler instance queries it before submitting a
        new task, so a target already being handled — by this satellite or
        any other sharing the scheduler — isn't re-tasked. Deliberately
        simple: linear scan and a single distance gate, no track fusion.

        Args:
            location_m: Query location (scene/ECI frame) [m].
            radius_m: Match radius [m]; tasks with no ``target_location_m``
                are never matched (nothing to compare against).

        Returns:
            The first matching non-terminal task, or ``None``.
        """
        for req in self._queue:
            if req.status not in _ACTIVE_STATUSES:
                continue
            if req.target_location_m is None:
                continue
            if (req.target_location_m - location_m).norm() <= radius_m:
                return req
        return None

    def clear(self) -> None:
        """Remove all tasks, restoring the scheduler to its initial state."""
        self._queue.clear()


__all__ = ["PriorityQueueScheduler", "TaskScheduler"]
