"""Minimal two-body constellation environment.

Advances satellites with :class:`~satsim.core.orbital.propagator.TwoBodyPropagator`.
Sensor/perception hooks are reserved (empty observation bundles) so the dynamics
spine can be validated independently of rendering and CV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from satsim.application.simulation.clock import SimulationClock
from satsim.application.simulation.environment import StepResult
from satsim.core.constants import DEFAULT_LEO_ALTITUDE_M, MU_EARTH
from satsim.core.orbital.propagator import PropagatorConfig, TwoBodyPropagator
from satsim.core.orbital.state import OrbitalState
from satsim.core.orbital.twobody import circular_leo_state
from satsim.core.types.identifiers import SatelliteId
from satsim.core.types.time import SimTime, TimeSpan
from satsim.core.types.vectors import CartesianState, Vec3


@dataclass(frozen=True, slots=True)
class SatelliteSpec:
    """Initial-condition specification for one vehicle.

    Attributes:
        satellite_id: Unique id string (stored as :class:`SatelliteId`).
        altitude_m: Circular-orbit altitude above equatorial radius [m].
        inclination_rad: Orbit inclination [rad].
        raan_rad: RAAN [rad].
        true_anomaly_rad: Initial true anomaly [rad].
    """

    satellite_id: str
    altitude_m: float = DEFAULT_LEO_ALTITUDE_M
    inclination_rad: float = 0.0
    raan_rad: float = 0.0
    true_anomaly_rad: float = 0.0


def default_demo_constellation(n: int = 3) -> tuple[SatelliteSpec, ...]:
    """Build a small co-planar LEO demo constellation.

    Satellites share inclination and altitude; true anomaly is evenly spaced.

    Args:
        n: Number of satellites (must be >= 1).

    Returns:
        Tuple of :class:`SatelliteSpec`.

    Raises:
        ValueError: If ``n < 1``.
    """
    if n < 1:
        msg = "constellation size must be >= 1"
        raise ValueError(msg)
    specs: list[SatelliteSpec] = []
    for i in range(n):
        specs.append(
            SatelliteSpec(
                satellite_id=f"sat-{i + 1:03d}",
                altitude_m=DEFAULT_LEO_ALTITUDE_M,
                inclination_rad=math.radians(53.0),
                raan_rad=0.0,
                true_anomaly_rad=2.0 * math.pi * i / n,
            )
        )
    return tuple(specs)


def _spec_to_state(spec: SatelliteSpec, time: SimTime, mu: float) -> OrbitalState:
    """Convert a satellite spec to an :class:`OrbitalState` at ``time``."""
    r, v = circular_leo_state(
        spec.altitude_m,
        mu=mu,
        inclination_rad=spec.inclination_rad,
        raan_rad=spec.raan_rad,
        true_anomaly_rad=spec.true_anomaly_rad,
    )
    return OrbitalState(
        satellite_id=SatelliteId(spec.satellite_id),
        time=time,
        cartesian=CartesianState(
            position_m=Vec3.from_array(r),
            velocity_m_s=Vec3.from_array(v),
            frame="ECI_J2000",
        ),
    )


@dataclass
class TwoBodyEnvironment:
    """Fixed-step two-body constellation simulation.

    Attributes:
        duration_s: Episode length [s]; ``truncated`` becomes True when reached.
        dt_s: Step size [s].
        satellite_specs: Initial condition templates.
        propagator: Orbital propagator (defaults to Earth two-body).
        seed: RNG seed reserved for future stochastic sensors/agents.
    """

    duration_s: float = 600.0
    dt_s: float = 1.0
    satellite_specs: tuple[SatelliteSpec, ...] = field(
        default_factory=lambda: default_demo_constellation(3)
    )
    propagator: TwoBodyPropagator = field(
        default_factory=lambda: TwoBodyPropagator(PropagatorConfig(mu=MU_EARTH))
    )
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate timing parameters and initialize runtime state."""
        if self.duration_s < 0.0:
            msg = "duration_s must be non-negative"
            raise ValueError(msg)
        if self.dt_s <= 0.0:
            msg = "dt_s must be positive"
            raise ValueError(msg)
        if not self.satellite_specs:
            msg = "at least one satellite_spec is required"
            raise ValueError(msg)
        self._clock = SimulationClock.create(start_s=0.0, dt_s=self.dt_s)
        self._states: dict[str, OrbitalState] = {}
        self._initial_states: dict[str, OrbitalState] = {}
        self._closed = False

    @property
    def time(self) -> SimTime:
        """Current simulation time."""
        return self._clock.current

    @property
    def step_index(self) -> int:
        """Number of completed dynamics steps since last reset."""
        return self._clock.step_index

    def reset(self, *, seed: int | None = None) -> StepResult:
        """Reset satellites to initial conditions and clock to epoch.

        Args:
            seed: Optional seed override for this episode.

        Returns:
            Snapshot at ``t = 0`` (before any dynamics step).
        """
        if seed is not None:
            self.seed = seed
        self._clock.reset()
        self._closed = False
        mu = self.propagator.config.mu
        t0 = self._clock.current
        self._initial_states = {
            spec.satellite_id: _spec_to_state(spec, t0, mu) for spec in self.satellite_specs
        }
        self._states = dict(self._initial_states)
        return self._snapshot(
            done=False,
            truncated=False,
            infos={"event": "reset", "seed": self.seed, "n_satellites": len(self._states)},
        )

    def step(self, actions: dict[str, Any] | None = None) -> StepResult:
        """Propagate all satellites by one time step.

        Args:
            actions: Reserved for future tasking / control (ignored for now).

        Returns:
            Post-step snapshot. ``truncated`` is True when ``time >= duration_s``.

        Raises:
            RuntimeError: If called after the episode has already ended without
                :meth:`reset`.
        """
        del actions  # reserved for closed-loop control
        if self._closed:
            msg = "Episode finished; call reset() before stepping again"
            raise RuntimeError(msg)
        if not self._states:
            # Auto-reset if step is called without explicit reset
            self.reset(seed=self.seed)

        dt = TimeSpan(self.dt_s)
        new_states: dict[str, OrbitalState] = {}
        for sat_id, state in self._states.items():
            new_states[sat_id] = self.propagator.propagate(state, dt)
        self._states = new_states
        self._clock.advance()

        truncated = self._clock.current.seconds >= self.duration_s - 1e-12
        if truncated:
            self._closed = True

        return self._snapshot(
            done=False,
            truncated=truncated,
            infos={
                "event": "step",
                "step_index": self._clock.step_index,
                "n_satellites": len(self._states),
            },
        )

    def _snapshot(
        self,
        *,
        done: bool,
        truncated: bool,
        infos: dict[str, Any],
    ) -> StepResult:
        """Build a :class:`StepResult` from current internal state."""
        return StepResult(
            time=self._clock.current,
            states=dict(self._states),
            observations=(),
            rewards={},
            infos=infos,
            done=done,
            truncated=truncated,
        )


__all__ = [
    "SatelliteSpec",
    "TwoBodyEnvironment",
    "default_demo_constellation",
]
