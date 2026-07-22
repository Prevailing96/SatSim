"""Scene renderer protocol and placeholder implementation.

Ideal (noise-free) RGB and depth are produced here; sensor-effect pipelines
in ``infrastructure.sensors`` corrupt them into realistic measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from satsim.core.sensors.models import CameraModel, ToFSensorModel
from satsim.core.types.time import SimTime
from satsim.infrastructure.rendering.scene import SceneDescriptor


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Parameters for a single render call.

    Attributes:
        time: Simulation time of the capture.
        scene: World content to render.
        camera: Optional EO camera model.
        tof: Optional ToF / depth sensor model.
        extras: Backend-specific options (samples, resolution overrides).
    """

    time: SimTime
    scene: SceneDescriptor
    camera: CameraModel | None = None
    tof: ToFSensorModel | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Ideal synthetic products before sensor corruption.

    Attributes:
        time: Capture time.
        rgb: Optional ``(H, W, 3)`` float32 image in ``[0, 1]``.
        depth_m: Optional ``(H, W)`` depth in meters.
        instance_ids: Optional instance id map for GT segmentation.
        metadata: Render diagnostics (backend name, timings).
    """

    time: SimTime
    rgb: NDArray[np.float32] | None = None
    depth_m: NDArray[np.float32] | None = None
    instance_ids: NDArray[np.int32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SceneRenderer(Protocol):
    """Renders ideal sensor channels from a scene and camera/ToF models."""

    def render(self, request: RenderRequest) -> RenderResult:
        """Produce ideal RGB / depth for the request.

        Args:
            request: Render parameters.

        Returns:
            Ideal (pre-sensor-effect) result.
        """
        ...


class PlaceholderRenderer:
    """Deterministic placeholder that returns solid-color / planar depth.

    Useful for pipeline wiring tests without a full graphics stack. Replace
    with a real Earth/space renderer when the ``render`` extra is integrated.
    """

    def __init__(self, *, fill_rgb: tuple[float, float, float] = (0.1, 0.2, 0.4)) -> None:
        """Initialize with a constant RGB fill color.

        Args:
            fill_rgb: RGB triple in ``[0, 1]``.
        """
        self._fill_rgb = fill_rgb

    def render(self, request: RenderRequest) -> RenderResult:
        """Create placeholder arrays matching sensor resolutions.

        Args:
            request: Must include at least one of ``camera`` or ``tof``.

        Returns:
            Synthetic constant RGB and/or planar depth.

        Raises:
            ValueError: If neither camera nor ToF model is provided.
        """
        rgb: NDArray[np.float32] | None = None
        depth: NDArray[np.float32] | None = None
        instances: NDArray[np.int32] | None = None

        if request.camera is not None:
            h, w = request.camera.height_px, request.camera.width_px
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[..., 0] = self._fill_rgb[0]
            rgb[..., 1] = self._fill_rgb[1]
            rgb[..., 2] = self._fill_rgb[2]
            instances = np.zeros((h, w), dtype=np.int32)

        if request.tof is not None:
            h, w = request.tof.height_px, request.tof.width_px
            # Mid-range planar depth placeholder.
            mid = 0.5 * (request.tof.min_range_m + request.tof.max_range_m)
            depth = np.full((h, w), mid, dtype=np.float32)

        if rgb is None and depth is None:
            msg = "RenderRequest requires camera and/or tof model"
            raise ValueError(msg)

        return RenderResult(
            time=request.time,
            rgb=rgb,
            depth_m=depth,
            instance_ids=instances,
            metadata={
                "backend": "placeholder",
                "num_objects": len(request.scene.objects),
            },
        )


__all__ = [
    "PlaceholderRenderer",
    "RenderRequest",
    "RenderResult",
    "SceneRenderer",
]
