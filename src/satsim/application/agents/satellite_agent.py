"""Default satellite agent implementations."""

from __future__ import annotations

from satsim.application.agents.base import AgentAction, AgentObservation
from satsim.application.tasking.requests import TaskPriority, TaskRequest
from satsim.application.tasking.scheduler import PriorityQueueScheduler
from satsim.core.constants import R_EARTH_EQUATORIAL
from satsim.core.orbital.state import OrbitalState
from satsim.core.sensors.observations import Detection2D
from satsim.core.types.identifiers import AgentId, SatelliteId
from satsim.core.types.vectors import Vec3

#: Default spatial-deconfliction gate: two detections whose approximate
#: locations fall within this radius are treated as "probably the same
#: target." Because the location proxy is the *satellite's* nadir point
#: (see :func:`approximate_target_location`), not a fixed estimate of the
#: target itself, it drifts along the ground track while one target stays
#: in view across several consecutive frames — at LEO orbital speed
#: (~7.6 km/s) a ~27s pass drifts ~205 km. The radius needs enough margin
#: over that drift, or the *same* satellite re-tasks the *same* target
#: partway through a single pass, which defeats the point. 250 km covers
#: that with room to spare while staying far below the demo scenario's
#: inter-target spacing (thousands of km), so distinct targets never merge.
DEFAULT_DECONFLICTION_RADIUS_M = 250_000.0


def approximate_target_location(own_state: OrbitalState) -> Vec3:
    """Rough proxy for "where a detection roughly is": the satellite's nadir point.

    This is deliberately *not* true geolocation — it does not reverse-project
    the detection's pixel position through the camera model and depth, it
    just takes the sub-satellite point (where the position vector pierces
    Earth's spherical surface). For a narrow-FOV nadir-pointing camera that's
    within one footprint radius of whatever the camera actually saw, which is
    accurate enough to answer "is this probably the same target another
    satellite already tasked" without adding real geolocation math.

    One consequence worth stating plainly: every detection from the same
    observation gets the *same* location estimate (this satellite's nadir
    point), since nothing here distinguishes where in the frame a detection
    sits. Two simultaneously visible targets in one frame would not be told
    apart by this function alone — a limitation, not a bug, and out of scope
    for this session's simple deconfliction rule.

    Args:
        own_state: Observing satellite's own orbital state.

    Returns:
        Approximate target location in the same frame as ``own_state``.
    """
    position = own_state.cartesian.position_m
    radius = position.norm()
    if radius < 1.0:
        return position
    return position * (R_EARTH_EQUATORIAL / radius)


class SatelliteAgent:
    """Simple onboard agent that can raise tasking from CV detections.

    This scaffold demonstrates the closed-loop path:

    **detections → task requests → scheduler**, without claiming a sophisticated
    autonomy policy. Replace :meth:`act` with learned or rule-rich policies as
    the project matures.

    Deconfliction: before submitting a task, the agent asks its scheduler
    whether a non-terminal task already exists near the detection's
    approximate location (see :func:`approximate_target_location`). If so,
    it skips submission. When multiple agents share one
    :class:`~satsim.application.tasking.scheduler.PriorityQueueScheduler`
    instance (the constellation's shared task board — see
    :class:`~satsim.application.simulation.two_body_env.TwoBodyEnvironment`),
    that check sees every other satellite's tasks too, which is what
    prevents two satellites from both tasking the same target. An agent
    built without an explicit ``scheduler`` gets its own private one and
    only deconflicts against itself — fine for standalone use, just not
    constellation-aware.

    Args:
        agent_id: Agent identity.
        satellite_id: Host satellite.
        detection_class_names: Class names that trigger follow-up tasking.
        min_confidence: Confidence threshold for reactive tasking.
        scheduler: Task scheduler to submit into. Pass the *same* instance
            to multiple agents to get shared, constellation-wide
            deconfliction; omit it for a private, agent-owned scheduler.
        deconfliction_radius_m: Distance below which two detections are
            treated as the same target for tasking purposes.
    """

    def __init__(
        self,
        agent_id: AgentId,
        satellite_id: SatelliteId,
        *,
        detection_class_names: frozenset[str] | None = None,
        min_confidence: float = 0.5,
        scheduler: PriorityQueueScheduler | None = None,
        deconfliction_radius_m: float = DEFAULT_DECONFLICTION_RADIUS_M,
    ) -> None:
        self._agent_id = agent_id
        self._satellite_id = satellite_id
        self._detection_class_names = detection_class_names or frozenset()
        self._min_confidence = min_confidence
        # An agent constructed with no explicit scheduler owns (and resets)
        # a private one; an injected/shared scheduler is the caller's to
        # reset, so multiple agents resetting it independently can't stomp
        # on each other or leave stale references after env.reset().
        self._owns_scheduler = scheduler is None
        self._scheduler = scheduler if scheduler is not None else PriorityQueueScheduler()
        self._deconfliction_radius_m = deconfliction_radius_m

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
        """Task scheduler this agent submits into (may be shared)."""
        return self._scheduler

    def reset(self, *, seed: int | None = None) -> None:
        """Clear scheduler state, if this agent owns its scheduler.

        A shared/injected scheduler is left untouched — its owner (e.g. the
        environment) is responsible for clearing it once, centrally, rather
        than having each agent sharing it replace it with a fresh instance
        (which would silently desynchronize the other agents still holding
        the old one).

        Args:
            seed: Unused in the deterministic stub.
        """
        del seed
        if self._owns_scheduler:
            self._scheduler = PriorityQueueScheduler()

    def act(self, observation: AgentObservation) -> AgentAction:
        """React to high-confidence detections by enqueueing task requests.

        Detections that qualify (:meth:`_should_task`) but whose approximate
        location already has a non-terminal task nearby are deconflicted:
        skipped rather than resubmitted. See class docstring.

        Args:
            observation: Current agent observation including CV bundles.

        Returns:
            Action describing submitted tasks (or idle).
        """
        target_location = (
            approximate_target_location(observation.own_state)
            if observation.own_state is not None
            else None
        )

        submitted: list[str] = []
        deconflicted = 0
        for bundle in observation.bundles:
            for det in bundle.detections:
                if not self._should_task(det):
                    continue
                if target_location is not None and self._scheduler.find_active_near(
                    target_location, self._deconfliction_radius_m
                ):
                    deconflicted += 1
                    continue
                req = TaskRequest(
                    satellite_id=self._satellite_id,
                    sensor_id=bundle.sensor_id,
                    priority=TaskPriority.HIGH,
                    target_description=(
                        f"followup:{det.class_name}@{det.confidence:.2f}"
                    ),
                    target_location_m=target_location,
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
                "deconflicted_duplicates": deconflicted,
            },
        )

    def _should_task(self, det: Detection2D) -> bool:
        """Return True if a detection should trigger follow-up tasking."""
        if det.confidence < self._min_confidence:
            return False
        if not self._detection_class_names:
            return True
        return det.class_name in self._detection_class_names


__all__ = ["DEFAULT_DECONFLICTION_RADIUS_M", "SatelliteAgent", "approximate_target_location"]
