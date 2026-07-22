"""Shared pytest fixtures for SatSim tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.time import SimTime
from satsim.core.types.vectors import CartesianState, Vec3


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def configs_dir(repo_root: Path) -> Path:
    """Return the configs/ directory."""
    return repo_root / "configs"


@pytest.fixture
def sample_sim_time() -> SimTime:
    """A fixed simulation time for deterministic tests."""
    return SimTime(0.0)


@pytest.fixture
def sample_satellite_id() -> SatelliteId:
    """Sample satellite id."""
    return SatelliteId("sat-001")


@pytest.fixture
def sample_sensor_id() -> SensorId:
    """Sample sensor id."""
    return SensorId("cam-001")


@pytest.fixture
def sample_cartesian_state() -> CartesianState:
    """LEO-scale circular-orbit-ish state in ECI (approximate)."""
    # ~6771 km radius along +X, circular velocity along +Y
    r = 6_771_000.0
    v = 7_667.0
    return CartesianState(
        position_m=Vec3(r, 0.0, 0.0),
        velocity_m_s=Vec3(0.0, v, 0.0),
        frame="ECI_J2000",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded NumPy generator."""
    return np.random.default_rng(0)
