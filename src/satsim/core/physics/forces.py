"""Specific force models for numerical (Cowell-style) propagation.

Force models return acceleration in the inertial frame of the state [m/s^2].
They are composed by numerical integrators; analytic two-body paths may ignore
them except for perturbation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from satsim.core.types.time import SimTime
from satsim.core.types.vectors import CartesianState, Vec3


@dataclass(frozen=True, slots=True)
class ForceModelContext:
    """Extra context available to force evaluations.

    Attributes:
        time: Simulation time of evaluation.
        satellite_mass_kg: Wet mass if drag/thrust scaling requires it.
        cross_section_m2: Reference area for drag [m^2].
        cd: Drag coefficient (dimensionless).
    """

    time: SimTime
    satellite_mass_kg: float = 1.0
    cross_section_m2: float = 1.0
    cd: float = 2.2


@runtime_checkable
class ForceModel(Protocol):
    """Computes specific force (acceleration) for a Cartesian state."""

    @property
    def name(self) -> str:
        """Human-readable model name for logging and composition."""
        ...

    def acceleration(self, state: CartesianState, ctx: ForceModelContext) -> Vec3:
        """Return acceleration [m/s^2] in ``state.frame``.

        Args:
            state: Position/velocity sample.
            ctx: Evaluation context (time, mass, aero params).

        Returns:
            Acceleration vector.
        """
        ...


class ZeroForceModel:
    """Null force model (zero acceleration). Useful as a default/no-op."""

    @property
    def name(self) -> str:
        """Model name."""
        return "zero"

    def acceleration(self, state: CartesianState, ctx: ForceModelContext) -> Vec3:
        """Return zero acceleration.

        Args:
            state: Unused state.
            ctx: Unused context.

        Returns:
            Zero vector.
        """
        del state, ctx
        return Vec3(0.0, 0.0, 0.0)


class PointMassGravity:
    """Central two-body gravity acceleration scaffold.

    Attributes:
        mu: Gravitational parameter [m^3 s^-2].
    """

    def __init__(self, mu: float) -> None:
        """Initialize with gravitational parameter.

        Args:
            mu: GM of the central body [m^3 s^-2].
        """
        self._mu = mu

    @property
    def name(self) -> str:
        """Model name."""
        return "point_mass_gravity"

    def acceleration(self, state: CartesianState, ctx: ForceModelContext) -> Vec3:
        """Compute ``-μ r / |r|^3``.

        Args:
            state: Position must be central-body-centered.
            ctx: Unused.

        Returns:
            Gravitational acceleration.

        Raises:
            ValueError: If position norm is near zero.
        """
        del ctx
        r = state.position_m.as_array()
        r_norm = float((r**2).sum() ** 0.5)
        if r_norm < 1e-9:
            msg = "Position norm too small for point-mass gravity"
            raise ValueError(msg)
        scale = -self._mu / (r_norm**3)
        return Vec3(float(r[0] * scale), float(r[1] * scale), float(r[2] * scale))


__all__ = [
    "ForceModel",
    "ForceModelContext",
    "PointMassGravity",
    "ZeroForceModel",
]
