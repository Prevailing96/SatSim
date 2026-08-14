"""Object detection interfaces, a stub, and a first rule-based detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

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


@dataclass(frozen=True, slots=True)
class RuleBasedBlobDetector:
    """Classical (non-ML) detector: color threshold + geometric shape gating.

    :class:`~satsim.infrastructure.rendering.renderer.PlaceholderRenderer`
    paints scene objects as a small cross-shaped marker. This detector looks
    only at the rendered pixels — never at scene ground truth. Pixels within
    ``color_tolerance`` of ``marker_color`` are grouped into connected
    components (so multiple simultaneously visible targets each get their
    own :class:`Detection2D` instead of one box spanning all of them), and
    each component then has to clear a handful of *geometric* gates before
    it is reported at all:

    - ``min_pixels`` / ``max_pixels`` — reject specks and runaway blobs.
    - ``min_solidity`` — ``matched_pixels / bbox_area``; rejects a sparse
      scatter of coincidentally-matching pixels spread across a large box
      (real markers are a compact, mostly-filled cross, not confetti).
    - ``max_aspect_ratio`` — rejects thin one-pixel-wide streaks, which
      color noise is more likely to produce than a compact blob.

    Color threshold alone can't distinguish "a real marker, blurred" from
    "a handful of unrelated pixels that happened to drift into tolerance";
    these gates are cheap, classical stand-ins for that judgment — no
    ground truth is consulted, only the geometry of what actually matched.

    Confidence for a surviving component is a weighted blend of how closely
    it resembles the *expected* marker in four independent ways: mean color
    closeness, pixel-count closeness, solidity closeness, and aspect-ratio
    closeness (see :meth:`_confidence`). It is a heuristic, not a
    calibrated probability — deliberately so, per this detector's scope.

    Every detection is reported under the same fixed ``class_name`` — a
    single-color threshold genuinely cannot tell one marker-colored blob
    from another, so it does not pretend to classify beyond "something is
    here." That's an honest reflection of what this rule works on, not a
    limitation to paper over.

    ``color_tolerance`` is deliberately loose: the sensor pipeline applies
    PSF blur and shot/read noise before this detector ever sees a frame,
    which dilutes marker-colored pixels toward the surrounding background.
    A handful of blurred marker pixels within a wide color tolerance is
    still a strong, unambiguous signal in this synthetic scene (nothing
    else in frame is anywhere near marker-red), so widening the match
    rather than requiring near-pure color keeps detection robust to that
    blur; the shape gates above pick up the robustness slack that a tighter
    color tolerance would otherwise have provided.

    Attributes:
        class_id: Reported class index.
        class_name: Reported class label.
        marker_color: RGB color (float32 ``[0, 1]`` domain) to match.
        color_tolerance: Max per-pixel Euclidean color distance to count as
            a match.
        min_pixels: Minimum matching pixel count (per connected component).
        max_pixels: Maximum matching pixel count (per connected component);
            guards against a runaway/saturated match.
        min_solidity: Minimum ``matched_pixels / bbox_area``.
        max_aspect_ratio: Maximum ``max(w, h) / min(w, h)`` of the bbox.
        expected_pixels: Pixel count a "typical" marker match has, used only
            to score confidence (not a hard gate).
        expected_solidity: Solidity a "typical" marker match has, used only
            to score confidence (not a hard gate).
    """

    class_id: int = 0
    class_name: str = "vessel"
    marker_color: tuple[float, float, float] = (1.0, 0.1, 0.1)
    color_tolerance: float = 0.35
    min_pixels: int = 3
    max_pixels: int = 300
    min_solidity: float = 0.15
    max_aspect_ratio: float = 3.0
    expected_pixels: float = 9.0
    expected_solidity: float = 0.4

    @property
    def name(self) -> str:
        """Backend name."""
        return "rule_based_blob"

    def predict(self, frame: ImageFrame) -> PerceptionResult:
        """Threshold the frame, shape-gate each blob, and score confidence.

        Args:
            frame: Input frame; must be an ``(H, W, 3+)`` RGB-like array.

        Returns:
            Zero or more detections: one per connected component of
            marker-colored pixels that passes the geometric gates.
        """
        data = frame.data
        if data.ndim != 3 or data.shape[2] < 3:
            return PerceptionResult(
                time=frame.time,
                detections=(),
                model_name=self.name,
                metadata={"note": "non-RGB frame"},
            )

        rgb = data[..., :3].astype(np.float32)
        target = np.array(self.marker_color, dtype=np.float32)
        distance = np.linalg.norm(rgb - target, axis=-1)
        mask = distance <= self.color_tolerance
        total_matched = int(mask.sum())
        if total_matched == 0:
            return PerceptionResult(
                time=frame.time,
                detections=(),
                model_name=self.name,
                metadata={"matched_pixels": 0, "num_components": 0, "accepted_components": 0},
            )

        labeled, num_components = ndimage.label(mask)
        detections: list[Detection2D] = []
        for component_id in range(1, num_components + 1):
            component_mask = labeled == component_id
            detection = self._score_component(component_mask, distance)
            if detection is not None:
                detections.append(detection)

        return PerceptionResult(
            time=frame.time,
            detections=tuple(detections),
            model_name=self.name,
            metadata={
                "matched_pixels": total_matched,
                "num_components": num_components,
                "accepted_components": len(detections),
            },
        )

    def _score_component(
        self,
        component_mask: NDArray[np.bool_],
        distance: NDArray[np.float32],
    ) -> Detection2D | None:
        """Apply geometric gates to one connected component and score it.

        Args:
            component_mask: Boolean ``(H, W)`` mask for a single component.
            distance: Per-pixel color distance from ``marker_color``.

        Returns:
            A :class:`Detection2D` if the component passes every gate,
            else ``None``.
        """
        matched = int(component_mask.sum())
        if matched < self.min_pixels or matched > self.max_pixels:
            return None

        rows, cols = np.nonzero(component_mask)
        x_min, x_max = int(cols.min()), int(cols.max()) + 1
        y_min, y_max = int(rows.min()), int(rows.max()) + 1
        width, height = x_max - x_min, y_max - y_min
        bbox_area = width * height
        solidity = matched / bbox_area
        aspect_ratio = max(width, height) / min(width, height)

        if solidity < self.min_solidity or aspect_ratio > self.max_aspect_ratio:
            return None

        mean_color_distance = float(distance[component_mask].mean())
        confidence = self._confidence(matched, solidity, aspect_ratio, mean_color_distance)

        return Detection2D(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=confidence,
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max),
            metadata={
                "matched_pixels": matched,
                "solidity": solidity,
                "aspect_ratio": aspect_ratio,
                "mean_color_distance": mean_color_distance,
            },
        )

    def _confidence(
        self,
        matched_pixels: int,
        solidity: float,
        aspect_ratio: float,
        mean_color_distance: float,
    ) -> float:
        """Blend four closeness-to-expected-marker scores into one heuristic.

        Each sub-score is in ``[0, 1]`` (1 = matches the expected marker
        exactly, 0 = at or beyond the gate boundary). This is a hand-tuned
        heuristic, not a calibrated probability — "reasonable," not
        statistically rigorous, is the explicit target here.

        Args:
            matched_pixels: Component pixel count.
            solidity: ``matched_pixels / bbox_area``.
            aspect_ratio: ``max(w, h) / min(w, h)`` of the bbox, ``>= 1``.
            mean_color_distance: Mean color distance of matched pixels.

        Returns:
            Confidence in ``[0, 1]``.
        """
        color_score = 1.0 - mean_color_distance / self.color_tolerance
        size_score = 1.0 - abs(matched_pixels - self.expected_pixels) / self.expected_pixels
        solidity_score = 1.0 - abs(solidity - self.expected_solidity) / self.expected_solidity
        aspect_span = max(self.max_aspect_ratio - 1.0, 1e-6)
        aspect_score = 1.0 - (aspect_ratio - 1.0) / aspect_span

        weighted = (
            0.4 * color_score + 0.2 * size_score + 0.2 * solidity_score + 0.2 * aspect_score
        )
        return float(np.clip(weighted, 0.0, 1.0))


__all__ = [
    "ObjectDetector",
    "PerceptionResult",
    "RuleBasedBlobDetector",
    "StubObjectDetector",
]
