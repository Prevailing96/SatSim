"""Object detection interfaces and a stub detector for pipeline tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from satsim.core.sensors.observations import Detection2D, ImageFrame
from satsim.core.types.time import SimTime


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """Detector (and later fused) perception output for one frame.

    Attributes:
        time: Inference time stamp.
        detections: 2D boxes.
        model_name: Backend identifier.
        metadata: Scores thresholds, device, latency, etc.
    """

    time: SimTime
    detections: tuple[Detection2D, ...] = ()
    model_name: str = "stub"
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObjectDetector(Protocol):
    """2D object detector over an :class:`ImageFrame`."""

    @property
    def name(self) -> str:
        """Model / backend name."""
        ...

    def predict(self, frame: ImageFrame) -> PerceptionResult:
        """Run detection on a single frame.

        Args:
            frame: Input image observation.

        Returns:
            Detection result.
        """
        ...


class StubObjectDetector:
    """Deterministic detector that returns no boxes (wiring placeholder).

    Swap for YOLO / Faster R-CNN / custom torch modules under the ``vision``
    optional extra without changing callers that depend on :class:`ObjectDetector`.
    """

    def __init__(self, name: str = "stub_detector") -> None:
        """Initialize stub.

        Args:
            name: Reported model name.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Backend name."""
        return self._name

    def predict(self, frame: ImageFrame) -> PerceptionResult:
        """Return empty detections.

        Args:
            frame: Input frame (unused beyond timestamp).

        Returns:
            Empty :class:`PerceptionResult`.
        """
        return PerceptionResult(
            time=frame.time,
            detections=(),
            model_name=self._name,
            metadata={"note": "stub returns no detections"},
        )


__all__ = ["ObjectDetector", "PerceptionResult", "StubObjectDetector"]
