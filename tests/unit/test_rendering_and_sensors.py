"""Tests for placeholder renderer and sensor effects."""

from __future__ import annotations

import numpy as np
import pytest

from satsim.core.sensors.models import CameraModel, ToFSensorModel
from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.time import SimTime
from satsim.infrastructure.rendering.renderer import PlaceholderRenderer, RenderRequest
from satsim.infrastructure.rendering.scene import SceneDescriptor
from satsim.infrastructure.sensors.effects import GaussianNoiseEffect, SensorPipeline
from satsim.infrastructure.sensors.tof import ToFArtifactModel


@pytest.mark.unit
@pytest.mark.render
def test_placeholder_renderer_shapes() -> None:
    sat = SatelliteId("s")
    cam = CameraModel(
        sensor_id=SensorId("c"),
        satellite_id=sat,
        width_px=64,
        height_px=48,
        focal_length_mm=25.0,
        pixel_size_um=3.0,
    )
    tof = ToFSensorModel(
        sensor_id=SensorId("t"),
        satellite_id=sat,
        width_px=32,
        height_px=24,
        max_range_m=20.0,
        min_range_m=1.0,
    )
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000"),
        camera=cam,
        tof=tof,
    )
    result = PlaceholderRenderer().render(req)
    assert result.rgb is not None
    assert result.rgb.shape == (48, 64, 3)
    assert result.depth_m is not None
    assert result.depth_m.shape == (24, 32)


@pytest.mark.unit
@pytest.mark.render
def test_gaussian_noise_changes_rgb(rng: np.random.Generator) -> None:
    sat = SatelliteId("s")
    cam = CameraModel(
        sensor_id=SensorId("c"),
        satellite_id=sat,
        width_px=16,
        height_px=16,
        focal_length_mm=10.0,
        pixel_size_um=5.0,
    )
    result = PlaceholderRenderer().render(
        RenderRequest(
            time=SimTime(0.0),
            scene=SceneDescriptor(frame="ECI_J2000"),
            camera=cam,
        )
    )
    assert result.rgb is not None
    pipeline = SensorPipeline([GaussianNoiseEffect(rgb_sigma=0.05, depth_sigma_m=0.0)])
    noisy = pipeline.apply(result, rng)
    assert noisy.rgb is not None
    assert not np.allclose(noisy.rgb, result.rgb)


@pytest.mark.unit
@pytest.mark.render
def test_tof_artifacts_range_limits(rng: np.random.Generator) -> None:
    sat = SatelliteId("s")
    tof = ToFSensorModel(
        sensor_id=SensorId("t"),
        satellite_id=sat,
        width_px=8,
        height_px=8,
        max_range_m=10.0,
        min_range_m=2.0,
    )
    result = PlaceholderRenderer().render(
        RenderRequest(
            time=SimTime(0.0),
            scene=SceneDescriptor(frame="ECI_J2000"),
            tof=tof,
        )
    )
    # Force some out-of-range values
    assert result.depth_m is not None
    depth = result.depth_m.copy()
    depth[0, 0] = 0.1
    depth[0, 1] = 100.0
    from satsim.infrastructure.rendering.renderer import RenderResult

    result = RenderResult(time=result.time, depth_m=depth, metadata={})
    model = ToFArtifactModel(
        flying_pixel_prob=0.0,
        min_range_m=2.0,
        max_range_m=10.0,
    )
    out = model.apply(result, rng)
    assert out.depth_m is not None
    assert np.isnan(out.depth_m[0, 0])
    assert np.isnan(out.depth_m[0, 1])
