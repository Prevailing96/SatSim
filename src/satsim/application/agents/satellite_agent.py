"""Default satellite agent implementations."""

from __future__ import annotations

from satsim.application.agents.base import AgentAction, AgentObservation
from satsim.application.tasking.requests import TaskPriority, TaskRequest
from satsim.application.tasking.scheduler import PriorityQueueScheduler
from satsim.core.sensors.observations import Detection2D
from satsim.core.types.identifiers import AgentId, SatelliteId


class SatelliteAgent:
    """Simple onboard agent that can raise tasking from CV detections.

    This scaffold demonstrates the closed-loop path:

    **detections → task requests → scheduler**, without claiming a sophisticated
    autonomy policy. Replace :meth:`act` with learned or rule-rich policies as
    the project matures.

    Args:
        agent_id: Agent identity.
        satellite_id: Host satellite.
        detection_class_names: Class names that trigger follow-up tasking.
        min_confidence: Confidence threshold for reactive tasking.
    """

    def __init__(
        self,
        agent_id: AgentId,
        satellite_id: SatelliteId,
        *,
        detection_class_names: frozenset[str] | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self._agent_id = agent_id
        self._satellite_id = satellite_id
        self._detection_class_names = detection_class_names or frozenset()
        self._min_confidence = min_confidence
        self._scheduler = PriorityQueueScheduler()

    @property
    def agent_id(self) -> AgentId:
        """Agent identifier."""
        return self._agent_id

    @property
    def satellite_id(self) -> SatelliteId:
        """Host satellite identifier."""
        return self._satellite_id

    @property
    def scheduler(self) -> PriorityQueueScheduler:
        """Local task scheduler (scaffold)."""
        return self._scheduler

    def reset(self, *, seed: int | None = None) -> None:
        """Clear local scheduler state.

        Args:
            seed: Unused in the deterministic stub.
        """
        del seed
        self._scheduler = PriorityQueueScheduler()

    def act(self, observation: AgentObservation) -> AgentAction:
        """React to high-confidence detections by enqueueing task requests.

        Args:
            observation: Current agent observation including CV bundles.

        Returns:
            Action describing submitted tasks (or idle).
        """
        submitted: list[str] = []
        for bundle in observation.bundles:
            for det in bundle.detections:
                if self._should_task(det):
                    req = TaskRequest(
                        satellite_id=self._satellite_id,
                        sensor_id=bundle.sensor_id,
                        priority=TaskPriority.HIGH,
                        target_description=(
                            f"followup:{det.class_name}@{det.confidence:.2f}"
                        ),
                        metadata={
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                            "bbox": [det.x_min, det.y_min, det.x_max, det.y_max],
                        },
                    )
                    self._scheduler.submit(req)
                    submitted.append(req.request_id)

        active = self._scheduler.tick(observation.time)
        if not submitted and not active:
            return AgentAction(agent_id=self._agent_id, kind="idle")

        return AgentAction(
            agent_id=self._agent_id,
            kind="task",
            payload={
                "submitted_request_ids": submitted,
                "active_request_ids": [t.request_id for t in active],
            },
        )

    def _should_task(self, det: Detection2D) -> bool:
        """Return True if a detection should trigger follow-up tasking."""
        if det.confidence < self._min_confidence:
            return False
        if not self._detection_class_names:
            return True
        return det.class_name in self._detection_class_names


__all__ = ["SatelliteAgent"]
