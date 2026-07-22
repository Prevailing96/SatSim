"""Discrete simulation clock driven by a fixed or variable time step."""

from __future__ import annotations

from dataclasses import dataclass

from satsim.core.types.time import SimTime, TimeSpan


@dataclass
class SimulationClock:
    """Tracks simulation time and step index.

    Attributes:
        start: Epoch time (usually ``SimTime(0.0)``).
        dt: Nominal step size.
        current: Current simulation time.
        step_index: Number of completed steps.
    """

    start: SimTime
    dt: TimeSpan
    current: SimTime
    step_index: int = 0

    @classmethod
    def create(cls, start_s: float = 0.0, dt_s: float = 1.0) -> SimulationClock:
        """Factory for a clock at ``start_s`` with step ``dt_s``.

        Args:
            start_s: Epoch seconds.
            dt_s: Step size in seconds (must be positive).

        Returns:
            Initialized clock.

        Raises:
            ValueError: If ``dt_s`` is not positive.
        """
        if dt_s <= 0.0:
            msg = "dt_s must be positive"
            raise ValueError(msg)
        start = SimTime(start_s)
        return cls(start=start, dt=TimeSpan(dt_s), current=start, step_index=0)

    def advance(self) -> SimTime:
        """Advance one step and return the new current time.

        Returns:
            Updated :class:`~satsim.core.types.time.SimTime`.
        """
        self.current = self.current + self.dt
        self.step_index += 1
        return self.current

    def reset(self) -> None:
        """Reset to the epoch."""
        self.current = self.start
        self.step_index = 0


__all__ = ["SimulationClock"]
