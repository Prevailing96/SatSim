"""Perception pipeline combining detection and optional segmentation.

The pipeline is the unit the simulation environment calls after sensor
formation. Outputs feed agents and tasking without depending on torch types.
"""

from __future__ import annotations

from dataclasses import dataclass

from satsim.core.sensors.observations import (
    Detection2D,
    ImageFrame,
    ObservationBundle,
    SegmentationMask,
)
from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.time import SimTime
from satsim.infrastructure.vision.detector import ObjectDetector, StubObjectDetector
from satsim.infrastructure.vision.segmenter import Segmenter, StubSegmenter


@dataclass
class PerceptionPipeline:
    """Runs configured perception models on an image frame.

    Attributes:
        detector: Object detector backend.
        segmenter: Optional segmenter (disabled if None).
        score_threshold: Drop detections below this confidence.
    """

    detector: ObjectDetector
    segmenter: Segmenter | None = None
    score_threshold: float = 0.25

    @classmethod
    def stub(cls) -> PerceptionPipeline:
        """Factory for a fully stubbed pipeline (CI-friendly).

        Returns:
            Pipeline with stub detector and segmenter.
        """
        return cls(
            detector=StubObjectDetector(),
            segmenter=StubSegmenter(),
            score_threshold=0.25,
        )

    def process(
        self,
        frame: ImageFrame,
        *,
        depth_bundle_sensor: SensorId | None = None,
    ) -> ObservationBundle:
        """Run perception and wrap products in an :class:`ObservationBundle`.

        Args:
            frame: Input image.
            depth_bundle_sensor: Unused placeholder for fused depth association.

        Returns:
            Bundle with detections and optional segmentation (no depth attach
            in this scaffold method — environment can merge depth separately).
        """
        del depth_bundle_sensor  # reserved for fused RGB-D pipelines
        det_result = self.detector.predict(frame)
        filtered = tuple(
            d for d in det_result.detections if d.confidence >= self.score_threshold
        )
        seg: SegmentationMask | None = None
        if self.segmenter is not None:
            seg = self.segmenter.predict(frame)

        return ObservationBundle(
            sensor_id=frame.sensor_id,
            satellite_id=frame.satellite_id,
            time=frame.time,
            image=frame,
            depth=None,
            detections=filtered,
            segmentation=seg,
        )

    def process_arrays(
        self,
        *,
        sensor_id: SensorId,
        satellite_id: SatelliteId,
        time: SimTime,
        image: ImageFrame,
    ) -> tuple[tuple[Detection2D, ...], SegmentationMask | None]:
        """Lower-level API returning detections and mask only.

        Args:
            sensor_id: Sensor id (must match frame).
            satellite_id: Satellite id (must match frame).
            time: Time stamp.
            image: Frame to process.

        Returns:
            ``(detections, segmentation_or_none)``.
        """
        del sensor_id, satellite_id, time
        bundle = self.process(image)
        return bundle.detections, bundle.segmentation


__all__ = ["PerceptionPipeline"]
