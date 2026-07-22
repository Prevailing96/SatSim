"""Observation products: images, depth, detections, and segmentations.

These types form the contract between rendering/sensor infrastructure,
perception models, and tasking / agent logic. Keep them serializable and
free of framework tensors so export and logging stay simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.time import SimTime


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """Raster image observation (RGB or single-channel).

    Attributes:
        sensor_id: Producing sensor.
        satellite_id: Host satellite.
        time: Capture simulation time.
        data: Image array ``(H, W)`` or ``(H, W, C)``. dtype typically
            ``uint8`` or ``float32`` in ``[0, 1]``.
        metadata: Free-form capture metadata (exposure, gain, frame id).
    """

    sensor_id: SensorId
    satellite_id: SatelliteId
    time: SimTime
    data: NDArray[Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""
        return tuple(int(x) for x in self.data.shape)


@dataclass(frozen=True, slots=True)
class DepthFrame:
    """Depth / range image (ideal or ToF-corrupted).

    Attributes:
        sensor_id: Producing sensor.
        satellite_id: Host satellite.
        time: Capture time.
        depth_m: Range per pixel [m]; invalid pixels may be NaN or <= 0.
        confidence: Optional per-pixel confidence in ``[0, 1]``.
        metadata: Capture / corruption metadata.
    """

    sensor_id: SensorId
    satellite_id: SatelliteId
    time: SimTime
    depth_m: NDArray[np.float32]
    confidence: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Detection2D:
    """Axis-aligned 2D detection in image coordinates.

    Attributes:
        class_id: Integer class index.
        class_name: Human-readable label.
        confidence: Score in ``[0, 1]``.
        x_min: Left edge [px].
        y_min: Top edge [px].
        x_max: Right edge [px].
        y_max: Bottom edge [px].
        track_id: Optional multi-frame track id.
    """

    class_id: int
    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    track_id: int | None = None

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y_max - self.y_min


@dataclass(frozen=True, slots=True)
class SegmentationMask:
    """Semantic or instance segmentation product.

    Attributes:
        sensor_id: Source sensor (for alignment with frames).
        time: Inference / capture time.
        mask: Integer label map ``(H, W)`` or instance ids.
        class_names: Mapping from label id to name (semantic).
        scores: Optional per-instance scores.
        metadata: Extra fields (model name, threshold).
    """

    sensor_id: SensorId
    time: SimTime
    mask: NDArray[np.int32]
    class_names: dict[int, str] = field(default_factory=dict)
    scores: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    """All products associated with one sense–perceive cycle for a sensor.

    Attributes:
        sensor_id: Sensor identity.
        satellite_id: Host vehicle.
        time: Bundle time stamp.
        image: Optional EO frame.
        depth: Optional depth / ToF frame.
        detections: Object detections (may be empty).
        segmentation: Optional segmentation product.
    """

    sensor_id: SensorId
    satellite_id: SatelliteId
    time: SimTime
    image: ImageFrame | None = None
    depth: DepthFrame | None = None
    detections: tuple[Detection2D, ...] = ()
    segmentation: SegmentationMask | None = None


__all__ = [
    "DepthFrame",
    "Detection2D",
    "ImageFrame",
    "ObservationBundle",
    "SegmentationMask",
]
