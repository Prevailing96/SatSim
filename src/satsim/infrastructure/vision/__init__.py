"""Vision / perception adapters: detection, segmentation, tracking hooks."""

from __future__ import annotations

from satsim.infrastructure.vision.detector import ObjectDetector, PerceptionResult
from satsim.infrastructure.vision.pipeline import PerceptionPipeline
from satsim.infrastructure.vision.segmenter import Segmenter

__all__ = [
    "ObjectDetector",
    "PerceptionPipeline",
    "PerceptionResult",
    "Segmenter",
]
