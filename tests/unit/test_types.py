"""Tests for core value objects."""

from __future__ import annotations

import math

import numpy as np
import pytest

from satsim.core.types.time import SimTime, TimeSpan
from satsim.core.types.vectors import AttitudeQuaternion, Vec3


@pytest.mark.unit
def test_sim_time_arithmetic() -> None:
    t0 = SimTime(10.0)
    t1 = t0 + TimeSpan(2.5)
    assert t1.seconds == pytest.approx(12.5)
    span = t1 - t0
    assert isinstance(span, TimeSpan)
    assert span.seconds == pytest.approx(2.5)


@pytest.mark.unit
def test_vec3_norm_and_array_roundtrip() -> None:
    v = Vec3(3.0, 4.0, 0.0)
    assert v.norm() == pytest.approx(5.0)
    arr = v.as_array()
    assert arr.shape == (3,)
    v2 = Vec3.from_array(arr)
    assert v2 == v


@pytest.mark.unit
def test_vec3_from_array_rejects_bad_size() -> None:
    with pytest.raises(ValueError, match="3 elements"):
        Vec3.from_array([1.0, 2.0])


@pytest.mark.unit
def test_quaternion_normalize() -> None:
    q = AttitudeQuaternion(2.0, 0.0, 0.0, 0.0)
    n = q.normalized()
    assert math.isclose(float(np.linalg.norm(n.as_array())), 1.0)
