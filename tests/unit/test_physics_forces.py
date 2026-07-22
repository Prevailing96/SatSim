"""Tests for force model stubs."""

from __future__ import annotations

import pytest

from satsim.core.constants import MU_EARTH
from satsim.core.physics.forces import ForceModelContext, PointMassGravity, ZeroForceModel
from satsim.core.types.time import SimTime
from satsim.core.types.vectors import CartesianState, Vec3


@pytest.mark.unit
@pytest.mark.orbital
def test_zero_force() -> None:
    model = ZeroForceModel()
    state = CartesianState(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    ctx = ForceModelContext(time=SimTime(0.0))
    a = model.acceleration(state, ctx)
    assert a == Vec3(0.0, 0.0, 0.0)


@pytest.mark.unit
@pytest.mark.orbital
def test_point_mass_gravity_direction() -> None:
    model = PointMassGravity(MU_EARTH)
    r = 7_000_000.0
    state = CartesianState(Vec3(r, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    ctx = ForceModelContext(time=SimTime(0.0))
    a = model.acceleration(state, ctx)
    # Acceleration should be toward origin: negative X, zero Y/Z.
    assert a.x < 0.0
    assert a.y == pytest.approx(0.0)
    assert a.z == pytest.approx(0.0)
    expected_mag = MU_EARTH / (r**2)
    assert abs(a.x) == pytest.approx(expected_mag, rel=1e-9)
