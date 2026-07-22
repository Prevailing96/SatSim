"""Shared value objects and identifiers for the SatSim domain."""

from __future__ import annotations

from satsim.core.types.identifiers import AgentId, ConstellationId, SatelliteId, SensorId
from satsim.core.types.time import SimTime, TimeSpan
from satsim.core.types.vectors import AttitudeQuaternion, CartesianState, Vec3

__all__ = [
    "AgentId",
    "AttitudeQuaternion",
    "CartesianState",
    "ConstellationId",
    "SatelliteId",
    "SensorId",
    "SimTime",
    "TimeSpan",
    "Vec3",
]
