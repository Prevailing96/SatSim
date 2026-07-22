"""Simulation time primitives.

SatSim uses a continuous simulation clock measured in seconds from a scenario
epoch. Absolute civil time / UTC conversion is an infrastructure concern
(ephemeris backends, logging); domain code should prefer :class:`SimTime`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class SimTime:
    """Simulation time as seconds since scenario epoch.

    Attributes:
        seconds: Seconds from the configured scenario epoch (may be fractional).
    """

    seconds: float

    def __add__(self, other: TimeSpan | float) -> SimTime:
        """Advance this time by a span or raw seconds."""
        if isinstance(other, TimeSpan):
            return SimTime(self.seconds + other.seconds)
        return SimTime(self.seconds + float(other))

    def __sub__(self, other: SimTime | TimeSpan | float) -> SimTime | TimeSpan:
        """Subtract a time (→ span) or a span/float (→ earlier time)."""
        if isinstance(other, SimTime):
            return TimeSpan(self.seconds - other.seconds)
        if isinstance(other, TimeSpan):
            return SimTime(self.seconds - other.seconds)
        return SimTime(self.seconds - float(other))


@dataclass(frozen=True, slots=True, order=True)
class TimeSpan:
    """A duration in simulation seconds.

    Attributes:
        seconds: Duration length (may be negative for reverse differences).
    """

    seconds: float

    def __add__(self, other: TimeSpan | float) -> TimeSpan:
        """Add two spans or a span and raw seconds."""
        if isinstance(other, TimeSpan):
            return TimeSpan(self.seconds + other.seconds)
        return TimeSpan(self.seconds + float(other))

    def __mul__(self, scale: float) -> TimeSpan:
        """Scale this duration."""
        return TimeSpan(self.seconds * scale)

    def abs(self) -> TimeSpan:
        """Return the absolute duration."""
        return TimeSpan(abs(self.seconds))


__all__ = ["SimTime", "TimeSpan"]
