"""Tests for placeholder renderer and sensor effects."""

from __future__ import annotations

import math

import numpy as np
import pytest

from satsim.core.constants import R_EARTH_EQUATORIAL
from satsim.core.sensors.models import CameraModel, ToFSensorModel
from satsim.core.sensors.observations import ImageFrame
from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.time import SimTime
from satsim.core.types.vectors import Vec3
from satsim.infrastructure.rendering.renderer import (
    PlaceholderRenderer,
    RenderRequest,
    RenderResult,
)
from satsim.infrastructure.rendering.scene import SceneDescriptor, SceneObject
from satsim.infrastructure.sensors.effects import (
    BitDepthQuantizationEffect,
    DepthEdgeArtifactEffect,
    DepthQuantizationEffect,
    DepthRangeNoiseEffect,
    GaussianBlurEffect,
    GaussianNoiseEffect,
    SensorPipeline,
    ShotReadNoiseEffect,
)
from satsim.infrastructure.sensors.tof import ToFArtifactModel
from satsim.infrastructure.vision.detector import RuleBasedBlobDetector


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


@pytest.mark.unit
@pytest.mark.render
def test_narrow_fov_nadir_camera_frame_is_all_earth() -> None:
    """A narrow-FOV nadir-pointing EO camera should see ground, not space."""
    sat = SatelliteId("s")
    cam = CameraModel(
        sensor_id=SensorId("c"),
        satellite_id=sat,
        width_px=32,
        height_px=24,
        focal_length_mm=35.0,
        pixel_size_um=5.0,
        fov_x_rad=math.radians(10.0),
        fov_y_rad=math.radians(8.0),
    )
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000"),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + 550_000.0, 0.0, 0.0),
    )
    result = PlaceholderRenderer().render(req)
    assert result.rgb is not None
    assert result.depth_m is not None
    # Every ray should hit Earth: no NaN depth, RGB never the pure-black
    # space fill color.
    assert not np.any(np.isnan(result.depth_m))
    assert not np.allclose(result.rgb, 0.0)


@pytest.mark.unit
@pytest.mark.render
def test_wide_fov_camera_shows_earth_disk_against_space() -> None:
    """A very wide FOV should show Earth as a disk surrounded by black space."""
    sat = SatelliteId("s")
    cam = CameraModel(
        sensor_id=SensorId("c"),
        satellite_id=sat,
        width_px=64,
        height_px=64,
        focal_length_mm=35.0,
        pixel_size_um=5.0,
        fov_x_rad=math.radians(170.0),
        fov_y_rad=math.radians(170.0),
    )
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000"),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + 550_000.0, 0.0, 0.0),
    )
    result = PlaceholderRenderer().render(req)
    assert result.depth_m is not None
    assert result.rgb is not None
    # Corners of a 170deg FOV frame should miss the ~67deg-half-angle Earth
    # disk (deep space -> NaN depth), while the center should hit it.
    assert np.isnan(result.depth_m[0, 0])
    assert not np.isnan(result.depth_m[32, 32])
    assert np.allclose(result.rgb[0, 0], 0.0)


@pytest.mark.unit
@pytest.mark.render
def test_scene_object_at_nadir_is_painted_and_centered() -> None:
    """A target directly below a nadir-pointing camera should land near center."""
    sat = SatelliteId("s")
    altitude_m = 550_000.0
    cam = CameraModel(
        sensor_id=SensorId("c"),
        satellite_id=sat,
        width_px=64,
        height_px=48,
        focal_length_mm=35.0,
        pixel_size_um=5.0,
        fov_x_rad=math.radians(20.0),
        fov_y_rad=math.radians(15.0),
    )
    target = SceneObject(
        object_id="t",
        name="vessel",
        position_m=Vec3(R_EARTH_EQUATORIAL, 0.0, 0.0),
    )
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000", objects=(target,)),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + altitude_m, 0.0, 0.0),
    )
    result = PlaceholderRenderer().render(req)
    assert result.rgb is not None
    assert result.instance_ids is not None
    # Marker pixels (bright red) should exist near the image center.
    marker_mask = result.instance_ids == 1
    assert np.any(marker_mask)
    rows, cols = np.nonzero(marker_mask)
    assert abs(int(rows.mean()) - 24) <= 2
    assert abs(int(cols.mean()) - 32) <= 2


def _narrow_camera(*, fov_deg: float = 20.0) -> CameraModel:
    return CameraModel(
        sensor_id=SensorId("c"),
        satellite_id=SatelliteId("s"),
        width_px=48,
        height_px=48,
        focal_length_mm=35.0,
        pixel_size_um=5.0,
        fov_x_rad=math.radians(fov_deg),
        fov_y_rad=math.radians(fov_deg),
    )


@pytest.mark.unit
@pytest.mark.render
def test_lambertian_shading_day_side_brighter_than_night_side() -> None:
    """Fixed sun-direction Lambertian term should darken the night side."""
    altitude_m = 550_000.0
    renderer = PlaceholderRenderer(sun_direction=(1.0, 0.0, 0.0))
    cam = _narrow_camera()

    day_req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000"),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + altitude_m, 0.0, 0.0),
    )
    night_req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000"),
        camera=cam,
        platform_position_m=Vec3(-(R_EARTH_EQUATORIAL + altitude_m), 0.0, 0.0),
    )
    day = renderer.render(day_req)
    night = renderer.render(night_req)
    assert day.rgb is not None
    assert night.rgb is not None
    assert float(day.rgb.mean()) > float(night.rgb.mean()) * 2.0


@pytest.mark.unit
@pytest.mark.render
def test_limb_darkening_center_brighter_than_edge() -> None:
    """Center of a moderately wide frame should be brighter than its edge."""
    altitude_m = 550_000.0
    # Sun roughly aligned with the boresight so the frame is fully lit; the
    # center-to-edge falloff is then dominated by the limb-darkening term.
    renderer = PlaceholderRenderer(sun_direction=(1.0, 0.0, 0.0))
    cam = _narrow_camera(fov_deg=70.0)
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000"),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + altitude_m, 0.0, 0.0),
    )
    result = renderer.render(req)
    assert result.rgb is not None
    center = float(result.rgb[24, 24].sum())
    edge = float(result.rgb[24, 2].sum())
    assert center > edge


@pytest.mark.unit
@pytest.mark.render
def test_atmosphere_glow_softens_wide_fov_disk_edge() -> None:
    """Rays that narrowly miss Earth should glow, not hard-cut to black."""
    altitude_m = 550_000.0
    cam = _narrow_camera(fov_deg=170.0)
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000"),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + altitude_m, 0.0, 0.0),
    )
    result = PlaceholderRenderer().render(req)
    assert result.rgb is not None
    assert result.depth_m is not None
    miss_mask = np.isnan(result.depth_m)
    missed_pixels = result.rgb[miss_mask]
    assert np.any(np.any(missed_pixels > 0.0, axis=-1)), "expected a nonzero glow ring"
    # The far corner is well outside the thin glow shell and stays pure black.
    assert np.allclose(result.rgb[0, 0], 0.0)


@pytest.mark.unit
@pytest.mark.render
def test_target_behind_earth_is_occluded() -> None:
    """A target on the far side of Earth must not be painted or detected."""
    altitude_m = 550_000.0
    cam = _narrow_camera()
    target = SceneObject(
        object_id="t",
        name="vessel",
        position_m=Vec3(-R_EARTH_EQUATORIAL, 0.0, 0.0),
    )
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000", objects=(target,)),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + altitude_m, 0.0, 0.0),
    )
    result = PlaceholderRenderer().render(req)
    assert result.instance_ids is not None
    assert not np.any(result.instance_ids == 1)


@pytest.mark.unit
@pytest.mark.render
@pytest.mark.vision
def test_structured_marker_still_detectable_end_to_end() -> None:
    """The cross-shaped marker (not a flat blob) must still trip the detector."""
    altitude_m = 550_000.0
    cam = _narrow_camera()
    target = SceneObject(
        object_id="t",
        name="vessel",
        position_m=Vec3(R_EARTH_EQUATORIAL, 0.0, 0.0),
    )
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000", objects=(target,)),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + altitude_m, 0.0, 0.0),
    )
    result = PlaceholderRenderer().render(req)
    assert result.rgb is not None
    frame = ImageFrame(
        sensor_id=SensorId("c"),
        satellite_id=SatelliteId("s"),
        time=SimTime(0.0),
        data=result.rgb,
    )
    perception = RuleBasedBlobDetector().predict(frame)
    assert len(perception.detections) == 1
    assert perception.detections[0].class_name == "vessel"


@pytest.mark.unit
@pytest.mark.render
def test_depth_range_noise_increases_with_range(rng: np.random.Generator) -> None:
    """ToF-style depth noise should have a larger spread at longer ranges."""
    effect = DepthRangeNoiseEffect(
        base_sigma_m=0.02, range_coefficient_m=1.0, reference_range_m=1000.0
    )
    near = RenderResult(time=SimTime(0.0), depth_m=np.full((200, 200), 100.0, dtype=np.float32))
    far = RenderResult(time=SimTime(0.0), depth_m=np.full((200, 200), 5000.0, dtype=np.float32))

    near_out = effect.apply(near, rng)
    far_out = effect.apply(far, rng)
    assert near_out.depth_m is not None
    assert far_out.depth_m is not None
    assert float(np.std(far_out.depth_m)) > float(np.std(near_out.depth_m)) * 5.0


@pytest.mark.unit
@pytest.mark.render
def test_depth_range_noise_preserves_nan_pixels(rng: np.random.Generator) -> None:
    """No-return (NaN) depth pixels must stay NaN, not become noisy numbers."""
    depth = np.full((8, 8), 500.0, dtype=np.float32)
    depth[0, 0] = np.nan
    effect = DepthRangeNoiseEffect()
    out = effect.apply(RenderResult(time=SimTime(0.0), depth_m=depth), rng)
    assert out.depth_m is not None
    assert np.isnan(out.depth_m[0, 0])
    assert not np.any(np.isnan(out.depth_m[1:, 1:]))


@pytest.mark.unit
@pytest.mark.render
def test_depth_quantization_snaps_to_grid() -> None:
    depth = np.linspace(0.0, 100.0, 401, dtype=np.float32).reshape(1, -1)
    effect = DepthQuantizationEffect(resolution_m=0.5)
    out = effect.apply(RenderResult(time=SimTime(0.0), depth_m=depth), np.random.default_rng(0))
    assert out.depth_m is not None
    remainder = np.mod(out.depth_m, 0.5)
    on_grid = np.isclose(remainder, 0.0, atol=1e-4) | np.isclose(remainder, 0.5, atol=1e-4)
    assert bool(np.all(on_grid))


@pytest.mark.unit
@pytest.mark.render
def test_depth_edge_artifact_leaves_flat_region_untouched(rng: np.random.Generator) -> None:
    """A perfectly flat depth region has no relative gradient, so no smear."""
    depth = np.full((16, 16), 1000.0, dtype=np.float32)
    effect = DepthEdgeArtifactEffect(flying_pixel_prob=1.0)  # would smear every edge pixel
    out = effect.apply(RenderResult(time=SimTime(0.0), depth_m=depth), rng)
    assert out.depth_m is not None
    assert np.array_equal(out.depth_m, depth)


@pytest.mark.unit
@pytest.mark.render
def test_depth_edge_artifact_only_touches_pixels_near_a_jump(rng: np.random.Generator) -> None:
    depth = np.full((16, 16), 1000.0, dtype=np.float32)
    depth[:, 8:] = 50_000.0  # sharp relative jump halfway across
    effect = DepthEdgeArtifactEffect(flying_pixel_prob=1.0, relative_gradient_threshold=0.02)
    out = effect.apply(RenderResult(time=SimTime(0.0), depth_m=depth), rng)
    assert out.depth_m is not None
    # Far from the jump, values must be untouched.
    assert np.array_equal(out.depth_m[:, :4], depth[:, :4])
    assert np.array_equal(out.depth_m[:, 12:], depth[:, 12:])
    # Right at the jump, some pixels should have been perturbed.
    assert not np.array_equal(out.depth_m[:, 7:9], depth[:, 7:9])


@pytest.mark.unit
@pytest.mark.render
def test_gaussian_blur_softens_sharp_edge() -> None:
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[:, 8:, :] = 1.0  # hard vertical edge
    effect = GaussianBlurEffect(sigma_px=1.5)
    out = effect.apply(RenderResult(time=SimTime(0.0), rgb=image), np.random.default_rng(0))
    assert out.rgb is not None
    edge_col = out.rgb[8, 7, 0]
    assert 0.0 < float(edge_col) < 1.0  # softened, not a hard 0/1 step anymore


@pytest.mark.unit
@pytest.mark.render
def test_shot_read_noise_changes_rgb_and_stays_bounded(rng: np.random.Generator) -> None:
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    effect = ShotReadNoiseEffect(shot_noise_coeff=0.05, read_noise_sigma=0.01)
    out = effect.apply(RenderResult(time=SimTime(0.0), rgb=image), rng)
    assert out.rgb is not None
    assert not np.allclose(out.rgb, image)
    assert float(out.rgb.min()) >= 0.0
    assert float(out.rgb.max()) <= 1.0


@pytest.mark.unit
@pytest.mark.render
def test_bit_depth_quantization_reduces_distinct_levels() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((32, 32, 3)).astype(np.float32)
    effect = BitDepthQuantizationEffect(bits=4)  # coarse: 16 levels
    out = effect.apply(RenderResult(time=SimTime(0.0), rgb=image), rng)
    assert out.rgb is not None
    assert np.unique(out.rgb).size <= 17
    assert np.unique(image).size > 17


@pytest.mark.unit
@pytest.mark.render
@pytest.mark.vision
def test_full_sensor_pipeline_still_detectable_end_to_end() -> None:
    """Blur + shot/read noise + bit-depth quantization must not blind the detector."""
    altitude_m = 550_000.0
    cam = _narrow_camera()
    target = SceneObject(
        object_id="t",
        name="vessel",
        position_m=Vec3(R_EARTH_EQUATORIAL, 0.0, 0.0),
    )
    req = RenderRequest(
        time=SimTime(0.0),
        scene=SceneDescriptor(frame="ECI_J2000", objects=(target,)),
        camera=cam,
        platform_position_m=Vec3(R_EARTH_EQUATORIAL + altitude_m, 0.0, 0.0),
    )
    ideal = PlaceholderRenderer().render(req)
    pipeline = SensorPipeline(
        [
            GaussianBlurEffect(sigma_px=0.5),
            ShotReadNoiseEffect(shot_noise_coeff=0.015, read_noise_sigma=0.004),
            BitDepthQuantizationEffect(bits=8),
            DepthRangeNoiseEffect(),
            DepthEdgeArtifactEffect(),
            DepthQuantizationEffect(resolution_m=1.0),
        ]
    )
    measured = pipeline.apply(ideal, np.random.default_rng(0))
    assert measured.rgb is not None
    assert measured.depth_m is not None
    # Depth is no longer a flat/purely-geometric map: it now carries sensor
    # noise character on top of the ray-traced geometry.
    assert "effect:depth_range_noise" in measured.metadata
    assert "effect:depth_quantization" in measured.metadata

    frame = ImageFrame(
        sensor_id=SensorId("c"),
        satellite_id=SatelliteId("s"),
        time=SimTime(0.0),
        data=measured.rgb,
    )
    perception = RuleBasedBlobDetector().predict(frame)
    assert len(perception.detections) == 1
    assert perception.detections[0].class_name == "vessel"
