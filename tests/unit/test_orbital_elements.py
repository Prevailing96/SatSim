"""Tests for Keplerian elements validation and helpers."""

from __future__ import annotations

import math

import pytest

from satsim.core.constants import MU_EARTH, R_EARTH_EQUATORIAL
from satsim.core.orbital.elements import KeplerianElements


def _leo_elements() -> KeplerianElements:
    return KeplerianElements(
        semi_major_axis_m=R_EARTH_EQUATORIAL + 550_000.0,
        eccentricity=0.001,
        inclination_rad=math.radians(53.0),
        raan_rad=0.0,
        arg_perigee_rad=0.0,
        true_anomaly_rad=0.0,
        mu=MU_EARTH,
    )


@pytest.mark.unit
@pytest.mark.orbital
def test_elements_validate_ok() -> None:
    el = _leo_elements()
    el.validate()  # does not raise


@pytest.mark.unit
@pytest.mark.orbital
def test_elements_reject_negative_a() -> None:
    el = KeplerianElements(
        semi_major_axis_m=-1.0,
        eccentricity=0.0,
        inclination_rad=0.0,
        raan_rad=0.0,
        arg_perigee_rad=0.0,
        true_anomaly_rad=0.0,
    )
    with pytest.raises(ValueError, match="semi_major_axis"):
        el.validate()


@pytest.mark.unit
@pytest.mark.orbital
def test_elements_reject_hyperbolic_scaffold() -> None:
    el = KeplerianElements(
        semi_major_axis_m=7_000_000.0,
        eccentricity=1.2,
        inclination_rad=0.0,
        raan_rad=0.0,
        arg_perigee_rad=0.0,
        true_anomaly_rad=0.0,
    )
    with pytest.raises(ValueError, match="eccentricity"):
        el.validate()


@pytest.mark.unit
@pytest.mark.orbital
def test_peri_apo_radius() -> None:
    el = _leo_elements()
    assert el.periapsis_radius_m < el.apoapsis_radius_m
    assert el.period_s > 0.0
