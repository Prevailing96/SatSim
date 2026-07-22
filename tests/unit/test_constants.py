"""Tests for physical constants and helpers."""

from __future__ import annotations

import math

import pytest

from satsim.core.constants import (
    MU_EARTH,
    R_EARTH_EQUATORIAL,
    circular_orbit_period_s,
)


@pytest.mark.unit
@pytest.mark.orbital
def test_mu_earth_positive() -> None:
    assert MU_EARTH > 0.0


@pytest.mark.unit
@pytest.mark.orbital
def test_circular_orbit_period_scales_with_a() -> None:
    a1 = R_EARTH_EQUATORIAL + 400_000.0
    a2 = R_EARTH_EQUATORIAL + 800_000.0
    t1 = circular_orbit_period_s(a1)
    t2 = circular_orbit_period_s(a2)
    assert t2 > t1
    # Kepler: T ∝ a^{3/2}
    ratio = t2 / t1
    expected = (a2 / a1) ** 1.5
    assert math.isclose(ratio, expected, rel_tol=1e-9)
