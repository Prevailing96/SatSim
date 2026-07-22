"""Agent protocol for autonomy and future multi-agent / RL policies.

Agents consume a structured observation (including CV products) and emit
actions that the environment applies (tasking updates, mode changes, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from satsim.core.orbital.state import OrbitalState
from satsim.core.sensors.observations import ObservationBundle
from satsim.core.types.identifiers import AgentId
from satsim.core.types.time import SimTime


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """What an agent is allowed to see at decision time.

    Attributes:
        agent_id: Decision-making agent.
        time: Simulation time.
        own_state: Own-ship orbital state if applicable.
        bundles: Sensor/perception products available to this agent.
        extras: Policy-specific features (coverage maps, comms, etc.).
    """

    agent_id: AgentId
    time: SimTime
    own_state: OrbitalState | None = None
    bundles: tuple[ObservationBundle, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentAction:
    """Action payload returned by a policy.

    Attributes:
        agent_id: Acting agent.
        kind: Action type tag (e.g. ``\"idle\"``, ``\"task\"``, ``\"slew\"``).
        payload: Structured parameters for the environment.
    """

    agent_id: AgentId
    kind: str = "idle"
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """Autonomous decision entity."""

    @property
    def agent_id(self) -> AgentId:
        """Unique agent identifier."""
        ...

    def act(self, observation: AgentObservation) -> AgentAction:
        """Select an action given the current observation.

        Args:
            observation: Agent-local view of the world.

        Returns:
            Chosen action.
        """
        ...

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal policy state between episodes.

        Args:
            seed: Optional RNG seed.
        """
        ...


__all__ = ["Agent", "AgentAction", "AgentObservation"]
