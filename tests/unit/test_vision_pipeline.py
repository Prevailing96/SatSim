"""Tests for perception pipeline stubs."""

from __future__ import annotations

import numpy as np
import pytest

from satsim.core.sensors.observations import ImageFrame
from satsim.core.types.identifiers import SatelliteId, SensorId
from satsim.core.types.time import SimTime
from satsim.infrastructure.vision.pipeline import PerceptionPipeline


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
