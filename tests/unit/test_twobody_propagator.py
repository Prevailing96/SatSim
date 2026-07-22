"""Tests for analytic two-body propagation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from satsim.core.constants import DEFAULT_LEO_ALTITUDE_M, MU_EARTH, R_EARTH_EQUATORIAL
from satsim.core.orbital.propagator import TwoBodyPropagator
from satsim.core.orbital.state import OrbitalState
from satsim.core.orbital.twobody import (
    circular_leo_state,
    propagate_rv,
    semi_major_axis,
    specific_energy,
    stumpff_c2,
    stumpff_c3,
)
from satsim.core.types.identifiers import SatelliteId
from satsim.core.types.time import SimTime, TimeSpan
from satsim.core.types.vectors import CartesianState, Vec3


@pytest.mark.unit
@pytest.mark.orbital
def test_stumpff_at_zero() -> None:
    assert stumpff_c2(0.0) == pytest.approx(0.5)
    assert stumpff_c3(0.0) == pytest.approx(1.0 / 6.0)


@pytest.mark.unit
@pytest.mark.orbital
def test_circular_energy_negative() -> None:
    r, v = circular_leo_state(DEFAULT_LEO_ALTITUDE_M)
    eps = specific_energy(r, v, MU_EARTH)
    assert eps < 0.0
    a = semi_major_axis(r, v, MU_EARTH)
    assert a == pytest.approx(R_EARTH_EQUATORIAL + DEFAULT_LEO_ALTITUDE_M, rel=1e-9)


@pytest.mark.unit
@pytest.mark.orbital
def test_propagate_zero_dt_identity() -> None:
    r0, v0 = circular_leo_state(DEFAULT_LEO_ALTITUDE_M)
    r1, v1 = propagate_rv(r0, v0, 0.0)
    np.testing.assert_allclose(r1, r0)
    np.testing.assert_allclose(v1, v0)


@pytest.mark.unit
@pytest.mark.orbital
def test_energy_conserved_over_step() -> None:
    r0, v0 = circular_leo_state(DEFAULT_LEO_ALTITUDE_M, inclination_rad=math.radians(51.6))
    e0 = specific_energy(r0, v0)
    r1, v1 = propagate_rv(r0, v0, 120.0)
    e1 = specific_energy(r1, v1)
    assert e1 == pytest.approx(e0, rel=1e-9, abs=1e-6)


@pytest.mark.unit
@pytest.mark.orbital
def test_round_trip_forward_back() -> None:
    r0, v0 = circular_leo_state(
        DEFAULT_LEO_ALTITUDE_M,
        inclination_rad=math.radians(53.0),
        true_anomaly_rad=0.7,
    )
    dt = 300.0
    r1, v1 = propagate_rv(r0, v0, dt)
    r2, v2 = propagate_rv(r1, v1, -dt)
    np.testing.assert_allclose(r2, r0, rtol=1e-8, atol=1e-3)
    np.testing.assert_allclose(v2, v0, rtol=1e-8, atol=1e-5)


@pytest.mark.unit
@pytest.mark.orbital
def test_circular_orbit_closes_after_period() -> None:
    altitude = DEFAULT_LEO_ALTITUDE_M
    r0, v0 = circular_leo_state(altitude)
    a = R_EARTH_EQUATORIAL + altitude
    period = 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH)
    r1, v1 = propagate_rv(r0, v0, period)
    np.testing.assert_allclose(r1, r0, rtol=1e-6, atol=1.0)  # ~1 m
    np.testing.assert_allclose(v1, v0, rtol=1e-6, atol=1e-3)


@pytest.mark.unit
@pytest.mark.orbital
def test_two_body_propagator_updates_time_and_state() -> None:
    r0, v0 = circular_leo_state(DEFAULT_LEO_ALTITUDE_M)
    state = OrbitalState(
        satellite_id=SatelliteId("sat-001"),
        time=SimTime(0.0),
        cartesian=CartesianState(
            position_m=Vec3.from_array(r0),
            velocity_m_s=Vec3.from_array(v0),
        ),
    )
    prop = TwoBodyPropagator()
    out = prop.propagate(state, TimeSpan(60.0))
    assert out.time.seconds == pytest.approx(60.0)
    assert out.satellite_id == state.satellite_id
    # Position should have moved
    assert out.cartesian.position_m.norm() != pytest.approx(state.cartesian.position_m.norm(), abs=0.0) or (
        out.cartesian.position_m.x != state.cartesian.position_m.x
    )
    dist = float(
        np.linalg.norm(
            out.cartesian.position_m.as_array() - state.cartesian.position_m.as_array()
        )
    )
    assert dist > 100.0  # moved more than 100 m in 60 s LEO
