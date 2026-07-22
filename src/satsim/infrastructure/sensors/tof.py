"""Time-of-flight specific artifact models.

Real ToF sensors exhibit multipath, flying pixels at depth edges, amplitude-
dependent noise, and limited unambiguous range. This module provides a
composable starting point; fidelity will grow with validated models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from satsim.infrastructure.rendering.renderer import RenderResult


@dataclass(frozen=True, slots=True)
class ToFArtifactModel:
    """Apply simplified ToF depth artifacts.

    Attributes:
        flying_pixel_prob: Probability of edge-adjacent flying pixels.
        multipath_bias_m: Additive bias to mimic multipath [m].
        edge_gradient_threshold: Depth gradient threshold for edge detection.
        min_range_m: Ranges below this become invalid (NaN).
        max_range_m: Ranges above this become invalid (NaN).
    """

    flying_pixel_prob: float = 0.02
    multipath_bias_m: float = 0.0
    edge_gradient_threshold: float = 1.0
    min_range_m: float = 0.5
    max_range_m: float = 100.0

    @property
    def name(self) -> str:
        """Effect name."""
        return "tof_artifacts"

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Corrupt depth with range limits, bias, and flying pixels.

        Args:
            result: Must include ``depth_m`` for meaningful work.
            rng: Random generator.

        Returns:
            Result with corrupted depth and metadata notes.
        """
        if result.depth_m is None:
            return result

        depth = result.depth_m.astype(np.float32).copy()
        if self.multipath_bias_m != 0.0:
            depth = depth + np.float32(self.multipath_bias_m)

        # Invalid outside unambiguous / usable range.
        invalid = (depth < self.min_range_m) | (depth > self.max_range_m)
        depth[invalid] = np.nan

        if self.flying_pixel_prob > 0.0:
            depth = self._apply_flying_pixels(depth, rng)

        meta = dict(result.metadata)
        meta["effect:tof_artifacts"] = {
            "flying_pixel_prob": self.flying_pixel_prob,
            "multipath_bias_m": self.multipath_bias_m,
            "min_range_m": self.min_range_m,
            "max_range_m": self.max_range_m,
        }

        return RenderResult(
            time=result.time,
            rgb=result.rgb,
            depth_m=depth,
            instance_ids=result.instance_ids,
            metadata=meta,
        )

    def _apply_flying_pixels(
        self,
        depth: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """Smear a fraction of edge pixels toward mixed depth values.

        Args:
            depth: Depth map [m].
            rng: RNG.

        Returns:
            Depth with flying-pixel corruption.
        """
        # Simple edge map via finite differences.
        gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
        gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        edges = (gx + gy) > self.edge_gradient_threshold
        mask = edges & (rng.random(depth.shape) < self.flying_pixel_prob)
        if not np.any(mask):
            return depth

        # Mix with a random neighbor-like offset.
        noise = rng.normal(0.0, 0.5, size=depth.shape).astype(np.float32)
        out = depth.copy()
        out[mask] = depth[mask] + noise[mask]
        return out


__all__ = ["ToFArtifactModel"]
