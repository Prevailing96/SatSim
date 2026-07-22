"""Sensor model contracts (optics, ToF, pose).

Domain models describe *what* a sensor is (intrinsics, rates, modalities).
Concrete noise / rendering pipelines live under ``infrastructure.sensors``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.vectors import AttitudeQuaternion, Vec3


class SensorModality(str, Enum):
    """Supported sensing modalities."""

    RGB = "rgb"
    GRAYSCALE = "grayscale"
    MULTISPECTRAL = "multispectral"
    DEPTH = "depth"
    TOF = "tof"
    IR = "ir"


@dataclass(frozen=True, slots=True)
class SensorPose:
    """Extrinsic pose of a sensor relative to the host satellite body.

    Attributes:
        position_body_m: Sensor origin in body frame [m].
        attitude_body: Rotation from sensor frame to body (or as documented
            by the concrete model; convention must be consistent in rendering).
    """

    position_body_m: Vec3
    attitude_body: AttitudeQuaternion


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Pinhole-style camera intrinsics and timing.

    Attributes:
        sensor_id: Unique sensor id.
        satellite_id: Host vehicle.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        focal_length_mm: Focal length [mm].
        pixel_size_um: Pixel pitch [µm].
        fov_x_rad: Horizontal field of view [rad] (optional if derived).
        fov_y_rad: Vertical field of view [rad] (optional if derived).
        frame_rate_hz: Nominal frame rate [Hz].
        modality: Imaging modality tag.
        pose: Mount pose on the satellite body.
    """

    sensor_id: SensorId
    satellite_id: SatelliteId
    width_px: int
    height_px: int
    focal_length_mm: float
    pixel_size_um: float
    frame_rate_hz: float = 1.0
    modality: SensorModality = SensorModality.RGB
    pose: SensorPose | None = None
    fov_x_rad: float | None = None
    fov_y_rad: float | None = None


@dataclass(frozen=True, slots=True)
class ToFSensorModel:
    """Time-of-flight / active depth sensor parameters.

    Attributes:
        sensor_id: Unique sensor id.
        satellite_id: Host vehicle.
        width_px: Depth map width.
        height_px: Depth map height.
        max_range_m: Maximum unambiguous range [m].
        min_range_m: Minimum valid range [m].
        range_resolution_m: Quantization / nominal resolution [m].
        modulation_frequency_hz: ToF modulation frequency [Hz].
        frame_rate_hz: Nominal rate [Hz].
        pose: Mount pose.
    """

    sensor_id: SensorId
    satellite_id: SatelliteId
    width_px: int
    height_px: int
    max_range_m: float
    min_range_m: float = 0.5
    range_resolution_m: float = 0.01
    modulation_frequency_hz: float = 30e6
    frame_rate_hz: float = 10.0
    pose: SensorPose | None = None


@runtime_checkable
class SensorModel(Protocol):
    """Minimal sensor identity protocol shared by concrete models."""

    @property
    def sensor_id(self) -> SensorId:
        """Unique sensor identifier."""
        ...

    @property
    def satellite_id(self) -> SatelliteId:
        """Host satellite identifier."""
        ...


__all__ = [
    "CameraModel",
    "SensorModality",
    "SensorModel",
    "SensorPose",
    "ToFSensorModel",
]
