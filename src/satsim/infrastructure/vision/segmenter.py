"""Semantic / instance segmentation interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from satsim.core.sensors.observations import ImageFrame, SegmentationMask


@runtime_checkable
class Segmenter(Protocol):
    """Produces a segmentation mask from an image frame."""

    @property
    def name(self) -> str:
        """Model / backend name."""
        ...

    def predict(self, frame: ImageFrame) -> SegmentationMask:
        """Run segmentation.

        Args:
            frame: Input image.

        Returns:
            Segmentation product aligned to the frame.
        """
        ...


class StubSegmenter:
    """Returns an all-background mask matching the frame spatial size."""

    def __init__(self, name: str = "stub_segmenter") -> None:
        """Initialize stub.

        Args:
            name: Reported model name.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Backend name."""
        return self._name

    def predict(self, frame: ImageFrame) -> SegmentationMask:
        """Create a zero (background) mask.

        Args:
            frame: Input frame.

        Returns:
            Background-only segmentation.
        """
        h, w = frame.data.shape[:2]
        mask = np.zeros((h, w), dtype=np.int32)
        return SegmentationMask(
            sensor_id=frame.sensor_id,
            time=frame.time,
            mask=mask,
            class_names={0: "background"},
            metadata={"model": self._name},
        )


__all__ = ["Segmenter", "StubSegmenter"]
