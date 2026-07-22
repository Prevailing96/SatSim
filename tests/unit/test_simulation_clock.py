"""Tests for the simulation clock."""

from __future__ import annotations

import pytest

from satsim.application.simulation.clock import SimulationClock


@pytest.mark.unit
def test_clock_advance() -> None:
    clock = SimulationClock.create(start_s=0.0, dt_s=2.0)
    assert clock.step_index == 0
    t = clock.advance()
    assert t.seconds == pytest.approx(2.0)
    assert clock.step_index == 1
    clock.reset()
    assert clock.current.seconds == pytest.approx(0.0)
    assert clock.step_index == 0


@pytest.mark.unit
def test_clock_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="positive"):
        SimulationClock.create(dt_s=0.0)
