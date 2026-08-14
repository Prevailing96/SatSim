"""Tests for reference-frame conversion helpers."""

from __future__ import annotations

import math

import pytest

from satsim.core.constants import R_EARTH_EQUATORIAL
from satsim.core.orbital.frames import geodetic_to_ecef_spherical


@pytest.mark.unit
def test_equator_prime_meridian_is_on_positive_x_axis() -> None:
    pos = geodetic_to_ecef_spherical(0.0, 0.0)
    assert pos.x == pytest.approx(R_EARTH_EQUATORIAL)
    assert pos.y == pytest.approx(0.0, abs=1e-6)
    assert pos.z == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_north_pole_is_on_positive_z_axis() -> None:
    pos = geodetic_to_ecef_spherical(math.radians(90.0), 0.0)
    assert pos.x == pytest.approx(0.0, abs=1e-6)
    assert pos.y == pytest.approx(0.0, abs=1e-6)
    assert pos.z == pytest.approx(R_EARTH_EQUATORIAL)


@pytest.mark.unit
def test_altitude_extends_radially_outward() -> None:
    surface = geodetic_to_ecef_spherical(math.radians(10.0), math.radians(20.0))
    above = geodetic_to_ecef_spherical(math.radians(10.0), math.radians(20.0), altitude_m=1000.0)
    assert above.norm() == pytest.approx(surface.norm() + 1000.0)


@pytest.mark.unit
def test_every_point_lies_on_the_sphere() -> None:
    for lat_deg, lon_deg in [(0, 0), (45, 45), (-30, 170), (89, -179), (-89, 5)]:
        pos = geodetic_to_ecef_spherical(math.radians(lat_deg), math.radians(lon_deg))
        assert pos.norm() == pytest.approx(R_EARTH_EQUATORIAL, rel=1e-9)
