"""Property-based tests for Vec3 algebra."""

from __future__ import annotations

import math

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from satsim.core.types.vectors import Vec3


finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@pytest.mark.unit
@given(x=finite, y=finite, z=finite)
def test_norm_non_negative(x: float, y: float, z: float) -> None:
    v = Vec3(x, y, z)
    assert v.norm() >= 0.0


@pytest.mark.unit
@given(x=finite, y=finite, z=finite, s=finite)
def test_scale_homogeneity(x: float, y: float, z: float, s: float) -> None:
    v = Vec3(x, y, z)
    scaled = v * s
    assert math.isclose(scaled.norm(), abs(s) * v.norm(), rel_tol=1e-9, abs_tol=1e-6)
