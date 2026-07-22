"""Composable sensor-effect pipelines applied to ideal render products.

Effects operate on :class:`~satsim.infrastructure.rendering.renderer.RenderResult`
(or raw arrays) and produce measurement-like frames for perception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from satsim.infrastructure.rendering.renderer import RenderResult


@runtime_checkable
class SensorEffect(Protocol):
    """Single corruption / formation stage in a sensor pipeline."""

    @property
    def name(self) -> str:
        """Effect name for logging."""
        ...

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Apply this effect, returning a new or mutated-equivalent result.

        Args:
            result: Upstream render / partial measurement.
            rng: NumPy generator for stochastic effects.

        Returns:
            Effect-applied result.
        """
        ...


@dataclass(frozen=True, slots=True)
class GaussianNoiseEffect:
    """Add i.i.d. Gaussian noise to RGB and/or depth channels.

    Attributes:
        rgb_sigma: Std-dev for RGB (float32 domain).
        depth_sigma_m: Std-dev for depth [m].
        clip_rgb: Whether to clip RGB to ``[0, 1]`` after noise.
    """

    rgb_sigma: float = 0.01
    depth_sigma_m: float = 0.05
    clip_rgb: bool = True

    @property
    def name(self) -> str:
        """Effect name."""
        return "gaussian_noise"

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Add Gaussian noise to available channels.

        Args:
            result: Input result.
            rng: Random generator.

        Returns:
            Noisy result with updated metadata.
        """
        rgb = result.rgb
        depth = result.depth_m
        meta = dict(result.metadata)
        meta["effect:gaussian_noise"] = {
            "rgb_sigma": self.rgb_sigma,
            "depth_sigma_m": self.depth_sigma_m,
        }

        if rgb is not None and self.rgb_sigma > 0.0:
            noise = rng.normal(0.0, self.rgb_sigma, size=rgb.shape).astype(np.float32)
            rgb = rgb.astype(np.float32) + noise
            if self.clip_rgb:
                rgb = np.clip(rgb, 0.0, 1.0)

        if depth is not None and self.depth_sigma_m > 0.0:
            noise_d = rng.normal(0.0, self.depth_sigma_m, size=depth.shape).astype(
                np.float32
            )
            depth = depth.astype(np.float32) + noise_d

        return RenderResult(
            time=result.time,
            rgb=rgb,
            depth_m=depth,
            instance_ids=result.instance_ids,
            metadata=meta,
        )


class SensorPipeline:
    """Ordered list of :class:`SensorEffect` stages.

    Args:
        effects: Effects applied in order.
    """

    def __init__(self, effects: list[SensorEffect] | None = None) -> None:
        self._effects: list[SensorEffect] = list(effects or [])

    @property
    def effects(self) -> tuple[SensorEffect, ...]:
        """Configured effects in application order."""
        return tuple(self._effects)

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Run all effects sequentially.

        Args:
            result: Ideal or partial result.
            rng: Shared RNG for the capture.

        Returns:
            Fully processed result.
        """
        out = result
        for effect in self._effects:
            out = effect.apply(out, rng)
        return out


class IdentitySensorPipeline(SensorPipeline):
    """Pipeline with no effects (pass-through)."""

    def __init__(self) -> None:
        """Initialize empty effect list."""
        super().__init__(effects=[])


def quantize_uint8(rgb: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Convert float RGB in ``[0, 1]`` to ``uint8``.

    Args:
        rgb: Float image.

    Returns:
        Quantized uint8 image.
    """
    clipped = np.clip(rgb, 0.0, 1.0)
    return (clipped * 255.0 + 0.5).astype(np.uint8)


__all__ = [
    "GaussianNoiseEffect",
    "IdentitySensorPipeline",
    "SensorEffect",
    "SensorPipeline",
    "quantize_uint8",
]
