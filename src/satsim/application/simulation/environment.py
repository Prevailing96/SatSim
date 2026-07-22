"""Simulation environment protocol for closed-loop runs and future RL.

The environment advances world state, collects observations, runs perception,
and applies agent actions. Gymnasium-compatible wrappers can later adapt
:class:`SimulationEnvironment` without changing the core loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from satsim.core.orbital.state import OrbitalState
from satsim.core.sensors.observations import ObservationBundle
from satsim.core.types.time import SimTime


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of a single environment step.

    Attributes:
        time: Simulation time after the step.
        states: Orbital states keyed by satellite id string.
        observations: Perception-ready observation bundles.
        rewards: Optional per-agent rewards (RL).
        infos: Free-form diagnostics.
        done: Whether the episode/scenario has terminated.
        truncated: Whether the run stopped due to time limit / external stop.
    """

    time: SimTime
    states: dict[str, OrbitalState]
    observations: tuple[ObservationBundle, ...] = ()
    rewards: dict[str, float] = field(default_factory=dict)
    infos: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    truncated: bool = False


@runtime_checkable
class SimulationEnvironment(Protocol):
    """Closed-loop simulation environment interface.

    Implementations own the propagator, renderers, perception stack, and
    agent wiring for a scenario.
    """

    def reset(self, *, seed: int | None = None) -> StepResult:
        """Reset to initial conditions.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            Initial step snapshot (often ``step_index == 0`` state).
        """
        ...

    def step(self, actions: dict[str, Any] | None = None) -> StepResult:
        """Advance one simulation step under optional agent actions.

        Args:
            actions: Mapping of agent id → action payload (tasking commands,
                attitude modes, etc.). ``None`` means default policy / idle.

        Returns:
            Step outcomes including observations and termination flags.
        """
        ...

    @property
    def time(self) -> SimTime:
        """Current simulation time."""
        ...


__all__ = ["SimulationEnvironment", "StepResult"]
