"""Tests for perception pipeline stubs and the rule-based detector."""

from __future__ import annotations

import numpy as np
import pytest

from satsim.core.sensors.observations import ImageFrame
from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.time import SimTime
from satsim.infrastructure.vision.detector import RuleBasedBlobDetector
from satsim.infrastructure.vision.pipeline import PerceptionPipeline

MARKER_RGB = (1.0, 0.1, 0.1)


def _paint_cross(
    data: np.ndarray,
    row0: int,
    col0: int,
    size: int = 5,
    color: tuple[float, float, float] = MARKER_RGB,
) -> None:
    """Paint a plus-sign marker like :class:`PlaceholderRenderer` does.

    Mimics the real renderer's marker geometry (a cross over a background
    patch) so detector tests exercise realistic blob shapes rather than
    solid squares, which the shape-aware confidence scoring now correctly
    treats as less marker-like.
    """
    mid = size // 2
    data[row0 : row0 + size, col0 + mid, :] = color
    data[row0 + mid, col0 : col0 + size, :] = color


def _frame(data: np.ndarray) -> ImageFrame:
    return ImageFrame(
        sensor_id=SensorId("c"), satellite_id=SatelliteId("s"), time=SimTime(0.0), data=data
    )


@pytest.mark.unit
@pytest.mark.vision
def test_stub_perception_pipeline_bundle() -> None:
    frame = ImageFrame(
        sensor_id=SensorId("cam"),
        satellite_id=SatelliteId("sat"),
        time=SimTime(1.0),
        data=np.zeros((32, 32, 3), dtype=np.float32),
    )
    pipe = PerceptionPipeline.stub()
    bundle = pipe.process(frame)
    assert bundle.image is frame
    assert bundle.detections == ()
    assert bundle.segmentation is not None
    assert bundle.segmentation.mask.shape == (32, 32)


@pytest.mark.unit
@pytest.mark.vision
def test_rule_based_detector_finds_painted_marker() -> None:
    data = np.zeros((32, 32, 3), dtype=np.float32)
    data[..., 2] = 0.5  # blueish "earth" background, nothing like the marker
    _paint_cross(data, row0=10, col0=12, size=5)  # realistic cross marker

    frame = ImageFrame(
        sensor_id=SensorId("cam"),
        satellite_id=SatelliteId("sat"),
        time=SimTime(2.0),
        data=data,
    )
    detector = RuleBasedBlobDetector()
    result = detector.predict(frame)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.class_name == "vessel"
    # A near-ideal cross-shaped, exact-color match should score confidently.
    assert det.confidence > 0.8
    assert det.x_min == pytest.approx(12.0)
    assert det.x_max == pytest.approx(17.0)
    assert det.y_min == pytest.approx(10.0)
    assert det.y_max == pytest.approx(15.0)


@pytest.mark.unit
@pytest.mark.vision
def test_rule_based_detector_no_marker_no_detection() -> None:
    data = np.zeros((16, 16, 3), dtype=np.float32)
    data[..., 2] = 0.5
    frame = ImageFrame(
        sensor_id=SensorId("cam"),
        satellite_id=SatelliteId("sat"),
        time=SimTime(0.0),
        data=data,
    )
    result = RuleBasedBlobDetector().predict(frame)
    assert result.detections == ()


@pytest.mark.unit
@pytest.mark.vision
def test_rule_based_pipeline_flows_through_pipeline() -> None:
    data = np.zeros((32, 32, 3), dtype=np.float32)
    _paint_cross(data, row0=5, col0=5, size=5)
    frame = ImageFrame(
        sensor_id=SensorId("cam"),
        satellite_id=SatelliteId("sat"),
        time=SimTime(0.0),
        data=data,
    )
    pipe = PerceptionPipeline.rule_based()
    bundle = pipe.process(frame)
    assert len(bundle.detections) == 1
    assert bundle.detections[0].confidence >= pipe.score_threshold


@pytest.mark.unit
@pytest.mark.vision
def test_rule_based_detector_separates_multiple_targets() -> None:
    """Two well-separated marker patches must yield two distinct detections."""
    data = np.zeros((48, 64, 3), dtype=np.float32)
    data[..., 2] = 0.5
    _paint_cross(data, row0=5, col0=5, size=5)  # blob A, top-left
    _paint_cross(data, row0=30, col0=45, size=5)  # blob B, bottom-right

    frame = ImageFrame(
        sensor_id=SensorId("cam"),
        satellite_id=SatelliteId("sat"),
        time=SimTime(0.0),
        data=data,
    )
    result = RuleBasedBlobDetector().predict(frame)
    assert len(result.detections) == 2
    assert result.metadata["num_components"] == 2

    boxes = sorted(result.detections, key=lambda d: d.x_min)
    assert boxes[0].x_max <= 10.0
    assert boxes[1].x_min >= 45.0
    for det in boxes:
        assert det.class_name == "vessel"
        assert det.confidence > 0.5


@pytest.mark.unit
@pytest.mark.vision
def test_hollow_ring_rejected_by_solidity() -> None:
    """A large hollow outline is one connected blob, but mostly empty inside.

    Isolates the solidity gate specifically: a 30x30 one-pixel-thick square
    ring is well above ``min_pixels``, has aspect ratio 1.0 (so the aspect
    gate would never catch it), yet fills only ~13% of its bounding box —
    nothing like the compact, mostly-filled cross a real marker paints.
    """
    data = np.zeros((40, 40, 3), dtype=np.float32)
    size = 30
    r0, c0 = 5, 5
    data[r0, c0 : c0 + size, :] = MARKER_RGB  # top edge
    data[r0 + size - 1, c0 : c0 + size, :] = MARKER_RGB  # bottom edge
    data[r0 : r0 + size, c0, :] = MARKER_RGB  # left edge
    data[r0 : r0 + size, c0 + size - 1, :] = MARKER_RGB  # right edge

    frame = ImageFrame(
        sensor_id=SensorId("cam"), satellite_id=SatelliteId("sat"), time=SimTime(0.0), data=data
    )
    result = RuleBasedBlobDetector().predict(frame)
    assert result.detections == ()
    assert result.metadata["num_components"] == 1
    assert result.metadata["accepted_components"] == 0


@pytest.mark.unit
@pytest.mark.vision
def test_thin_streak_rejected_by_aspect_ratio() -> None:
    """A one-pixel-wide streak is not a compact marker-shaped blob."""
    data = np.zeros((32, 32, 3), dtype=np.float32)
    data[10, 5:25, :] = MARKER_RGB  # 1 x 20 streak: aspect ratio 20

    frame = ImageFrame(
        sensor_id=SensorId("cam"), satellite_id=SatelliteId("sat"), time=SimTime(0.0), data=data
    )
    result = RuleBasedBlobDetector().predict(frame)
    assert result.detections == ()


@pytest.mark.unit
@pytest.mark.vision
def test_oversized_blob_rejected_by_max_pixels() -> None:
    """A runaway/saturated match (e.g. a rendering bug) should not be reported."""
    data = np.zeros((64, 64, 3), dtype=np.float32)
    data[:, :] = MARKER_RGB  # whole frame matches: clearly not a small target

    frame = ImageFrame(
        sensor_id=SensorId("cam"), satellite_id=SatelliteId("sat"), time=SimTime(0.0), data=data
    )
    result = RuleBasedBlobDetector().predict(frame)
    assert result.detections == ()


@pytest.mark.unit
@pytest.mark.vision
def test_confidence_drops_for_off_color_blob_within_tolerance() -> None:
    """Confidence should track color closeness, not just pass/fail the gate."""
    near = np.zeros((32, 32, 3), dtype=np.float32)
    _paint_cross(near, row0=10, col0=10, size=5, color=(1.0, 0.1, 0.1))

    far = np.zeros((32, 32, 3), dtype=np.float32)
    # Still within color_tolerance=0.35 of marker_color, but not exact.
    _paint_cross(far, row0=10, col0=10, size=5, color=(0.75, 0.25, 0.25))

    detector = RuleBasedBlobDetector()
    near_det = detector.predict(_frame(near)).detections[0]
    far_det = detector.predict(_frame(far)).detections[0]
    assert near_det.confidence > far_det.confidence


@pytest.mark.unit
@pytest.mark.vision
def test_confidence_drops_for_irregular_shape_vs_ideal_cross() -> None:
    """Confidence should track shape quality, not just color match."""
    data_cross = np.zeros((32, 32, 3), dtype=np.float32)
    _paint_cross(data_cross, row0=10, col0=10, size=5)

    data_blob = np.zeros((32, 32, 3), dtype=np.float32)
    data_blob[10:20, 10:20, :] = MARKER_RGB  # solid 10x10: same exact color, wrong shape

    detector = RuleBasedBlobDetector()
    cross_det = detector.predict(_frame(data_cross)).detections[0]
    blob_result = detector.predict(_frame(data_blob))
    assert len(blob_result.detections) == 1
    blob_det = blob_result.detections[0]
    assert cross_det.confidence > blob_det.confidence


@pytest.mark.unit
@pytest.mark.vision
def test_detection_metadata_carries_shape_diagnostics() -> None:
    data = np.zeros((32, 32, 3), dtype=np.float32)
    _paint_cross(data, row0=10, col0=10, size=5)
    frame = ImageFrame(
        sensor_id=SensorId("cam"), satellite_id=SatelliteId("sat"), time=SimTime(0.0), data=data
    )
    det = RuleBasedBlobDetector().predict(frame).detections[0]
    assert det.metadata["matched_pixels"] == 9
    assert det.metadata["solidity"] == pytest.approx(9 / 25)
    assert det.metadata["aspect_ratio"] == pytest.approx(1.0)
