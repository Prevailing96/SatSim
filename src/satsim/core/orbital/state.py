"""Orbital state snapshot for a single vehicle at a simulation time."""

from __future__ import annotations

from dataclasses import dataclass

from satsim.core.types.identifiers import SatelliteId
from satsim.core.types.time import SimTime
from satsim.core.types.vectors import AttitudeQuaternion, CartesianState


@dataclass(frozen=True, slots=True)
class OrbitalState:
    """Full kinematic snapshot used by the simulation loop.

    Attributes:
        satellite_id: Owning vehicle.
        time: Simulation time of validity.
        cartesian: Position/velocity in the tagged inertial/body frame.
        attitude: Optional body attitude relative to a reference frame.
    """

    satellite_id: SatelliteId
    time: SimTime
    cartesian: CartesianState
    attitude: AttitudeQuaternion | None = None


__all__ = ["OrbitalState"]
