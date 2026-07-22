"""Orbital propagator protocol and two-body implementation.

Concrete high-order / SGP4 backends may live alongside this module later; the
simulation engine depends on the :class:`Propagator` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from satsim.core.constants import MU_EARTH
from satsim.core.orbital.state import OrbitalState
from satsim.core.orbital.twobody import propagate_rv
from satsim.core.types.time import SimTime, TimeSpan
from satsim.core.types.vectors import CartesianState, Vec3


@dataclass(frozen=True, slots=True)
class PropagatorConfig:
    """Shared knobs for propagator backends.

    Attributes:
        mu: Central-body gravitational parameter [m^3 s^-2].
        include_j2: Whether J2 (or higher) geopotential terms are active.
        include_drag: Whether atmospheric drag is modeled.
        max_step_s: Suggested maximum integrator step [s] (numerical backends).
        extras: Backend-specific options (e.g. density model name).
    """

    mu: float = MU_EARTH
    include_j2: bool = False
    include_drag: bool = False
    max_step_s: float = 60.0
    extras: dict[str, float | str | bool] = field(default_factory=dict)


@runtime_checkable
class Propagator(Protocol):
    """Propagate a satellite orbital state forward or backward in time.

    Implementations must be deterministic for a fixed config and initial state
    (seeded stochastic force models should document their RNG policy).
    """

    @property
    def config(self) -> PropagatorConfig:
        """Active propagator configuration."""
        ...

    def propagate(self, state: OrbitalState, dt: TimeSpan) -> OrbitalState:
        """Propagate ``state`` by duration ``dt``.

        Args:
            state: Initial orbital state.
            dt: Propagation span (negative for reverse).

        Returns:
            State valid at ``state.time + dt``.
        """
        ...

    def propagate_to(self, state: OrbitalState, time: SimTime) -> OrbitalState:
        """Propagate ``state`` to an absolute simulation time.

        Args:
            state: Initial orbital state.
            time: Target simulation time.

        Returns:
            State valid at ``time``.
        """
        ...


class TwoBodyPropagator:
    """Analytic two-body propagator (universal-variable method).

    Propagates Cartesian state under point-mass gravity with gravitational
    parameter ``config.mu``. Attitude is carried forward unchanged. J2/drag
    flags on :class:`PropagatorConfig` are ignored (reserved for future
    perturbation models).
    """

    def __init__(self, config: PropagatorConfig | None = None) -> None:
        """Initialize with optional config.

        Args:
            config: Propagator settings; defaults to Earth two-body.
        """
        self._config = config or PropagatorConfig()

    @property
    def config(self) -> PropagatorConfig:
        """Active configuration."""
        return self._config

    def propagate(self, state: OrbitalState, dt: TimeSpan) -> OrbitalState:
        """Propagate state by ``dt`` under two-body dynamics.

        Args:
            state: Initial state (position/velocity in an inertial frame).
            dt: Time span (seconds; may be negative).

        Returns:
            New :class:`OrbitalState` at ``state.time + dt``.

        Raises:
            ValueError: If the state is singular or the solver fails.
        """
        r0, v0 = state.cartesian.as_arrays()
        r1, v1 = propagate_rv(r0, v0, dt.seconds, mu=self._config.mu)
        new_cartesian = CartesianState(
            position_m=Vec3.from_array(r1),
            velocity_m_s=Vec3.from_array(v1),
            frame=state.cartesian.frame,
        )
        return OrbitalState(
            satellite_id=state.satellite_id,
            time=state.time + dt,
            cartesian=new_cartesian,
            attitude=state.attitude,
        )

    def propagate_to(self, state: OrbitalState, time: SimTime) -> OrbitalState:
        """Propagate to absolute time via :meth:`propagate`.

        Args:
            state: Initial state.
            time: Target time.

        Returns:
            Updated state.
        """
        span = time - state.time
        if not isinstance(span, TimeSpan):  # pragma: no cover - defensive
            msg = "SimTime subtraction must yield TimeSpan"
            raise TypeError(msg)
        return self.propagate(state, span)


__all__ = ["Propagator", "PropagatorConfig", "TwoBodyPropagator"]
