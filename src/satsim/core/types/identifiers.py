"""Strongly typed identifiers for satellites, sensors, agents, and fleets.

Using ``NewType`` (and thin wrappers later if needed) prevents accidental
mixing of string IDs across subsystems while remaining cheap at runtime.
"""

from __future__ import annotations

from typing import NewType

#: Unique satellite vehicle identifier within a scenario.
SatelliteId = NewType("SatelliteId", str)

#: Unique sensor instance identifier (may map 1:N to a satellite).
SensorId = NewType("SensorId", str)

#: Logical agent identifier (may coincide with a satellite or ground node).
AgentId = NewType("AgentId", str)

#: Constellation / fleet identifier for multi-swarm scenarios.
ConstellationId = NewType("ConstellationId", str)

__all__ = [
    "AgentId",
    "ConstellationId",
    "SatelliteId",
    "SensorId",
]
