"""Composable sensor-effect pipelines applied to ideal render products.

Effects operate on :class:`~satsim.infrastructure.rendering.renderer.RenderResult`
(or raw arrays) and produce measurement-like frames for perception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

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


@dataclass(frozen=True, slots=True)
class GaussianBlurEffect:
    """Simple isotropic Gaussian point-spread-function blur on RGB.

    A real optical system never forms a perfectly sharp image; this
    convolves each color channel with a Gaussian kernel as a cheap stand-in
    for lens/diffraction blur, without modeling a full optical MTF.

    Attributes:
        sigma_px: Blur kernel standard deviation [pixels]. ``0`` disables it.
    """

    sigma_px: float = 0.6

    @property
    def name(self) -> str:
        """Effect name."""
        return "gaussian_psf_blur"

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Blur the RGB channel only; depth passes through untouched.

        Args:
            result: Input result.
            rng: Unused (blur is deterministic).

        Returns:
            Result with blurred RGB.
        """
        del rng
        rgb = result.rgb
        if rgb is None or self.sigma_px <= 0.0:
            return result

        blurred = gaussian_filter(rgb, sigma=(self.sigma_px, self.sigma_px, 0.0))
        meta = dict(result.metadata)
        meta["effect:gaussian_psf_blur"] = {"sigma_px": self.sigma_px}
        return RenderResult(
            time=result.time,
            rgb=blurred.astype(np.float32),
            depth_m=result.depth_m,
            instance_ids=result.instance_ids,
            metadata=meta,
        )


@dataclass(frozen=True, slots=True)
class ShotReadNoiseEffect:
    """Signal-dependent shot noise plus a fixed read-noise floor.

    A simplified Poisson-Gaussian sensor noise model: photon shot noise has
    std-dev proportional to ``sqrt(signal)``, on top of a roughly
    signal-independent electronic read-noise floor. This is a closer
    approximation to real camera noise than :class:`GaussianNoiseEffect`'s
    flat, signal-independent variance.

    Attributes:
        shot_noise_coeff: Scales the ``sqrt(signal)`` shot-noise term.
        read_noise_sigma: Fixed-variance read noise [0, 1] domain.
        clip_rgb: Whether to clip output to ``[0, 1]``.
    """

    shot_noise_coeff: float = 0.02
    read_noise_sigma: float = 0.004
    clip_rgb: bool = True

    @property
    def name(self) -> str:
        """Effect name."""
        return "shot_read_noise"

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Add signal-dependent shot noise and read noise to RGB.

        Args:
            result: Input result.
            rng: Random generator.

        Returns:
            Noisy result with updated metadata.
        """
        rgb = result.rgb
        if rgb is None:
            return result

        signal = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        shot_sigma = self.shot_noise_coeff * np.sqrt(signal)
        shot_unit = rng.normal(0.0, 1.0, size=signal.shape).astype(np.float32)
        read_unit = rng.normal(0.0, 1.0, size=signal.shape).astype(np.float32)
        noisy = signal + shot_unit * shot_sigma + read_unit * np.float32(self.read_noise_sigma)
        if self.clip_rgb:
            noisy = np.clip(noisy, 0.0, 1.0)

        meta = dict(result.metadata)
        meta["effect:shot_read_noise"] = {
            "shot_noise_coeff": self.shot_noise_coeff,
            "read_noise_sigma": self.read_noise_sigma,
        }
        return RenderResult(
            time=result.time,
            rgb=noisy.astype(np.float32),
            depth_m=result.depth_m,
            instance_ids=result.instance_ids,
            metadata=meta,
        )


@dataclass(frozen=True, slots=True)
class BitDepthQuantizationEffect:
    """Quantize RGB to a fixed bit depth, like a camera ADC.

    Output stays float32 in ``[0, 1]`` (so downstream consumers are
    unaffected) but is snapped to ``2**bits`` discrete levels.

    Attributes:
        bits: Output bit depth per channel (``1``-``16``).
    """

    bits: int = 8

    @property
    def name(self) -> str:
        """Effect name."""
        return "bit_depth_quantization"

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Snap RGB to ``2**bits`` discrete levels.

        Args:
            result: Input result.
            rng: Unused (quantization is deterministic).

        Returns:
            Result with quantized RGB.
        """
        del rng
        rgb = result.rgb
        if rgb is None:
            return result

        levels = float((1 << self.bits) - 1)
        clipped = np.clip(rgb, 0.0, 1.0)
        quantized = np.round(clipped * levels) / levels

        meta = dict(result.metadata)
        meta["effect:bit_depth_quantization"] = {"bits": self.bits}
        return RenderResult(
            time=result.time,
            rgb=quantized.astype(np.float32),
            depth_m=result.depth_m,
            instance_ids=result.instance_ids,
            metadata=meta,
        )


@dataclass(frozen=True, slots=True)
class DepthRangeNoiseEffect:
    """ToF-style range-dependent depth noise: precision degrades with range.

    Real time-of-flight / LIDAR sensors have SNR that falls off with
    distance (received signal power drops roughly with ``1/range**2``), so
    depth precision degrades at longer ranges. This adds Gaussian noise
    whose std-dev grows quadratically with range relative to a reference
    distance, unlike :class:`GaussianNoiseEffect`'s flat ``depth_sigma_m``.

    Attributes:
        base_sigma_m: Noise floor at zero range [m].
        range_coefficient_m: Extra std-dev contributed at ``reference_range_m``
            [m].
        reference_range_m: Range at which ``range_coefficient_m`` applies.
    """

    base_sigma_m: float = 1.0
    range_coefficient_m: float = 20.0
    reference_range_m: float = 600_000.0

    @property
    def name(self) -> str:
        """Effect name."""
        return "depth_range_noise"

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Add range-dependent Gaussian noise to depth.

        NaN (no-return) pixels are left as NaN; noise std-dev is computed
        from a NaN-substituted array so no comparison ever touches NaN
        directly (that would trip pytest's warnings-as-errors config).

        Args:
            result: Input result.
            rng: Random generator.

        Returns:
            Noisy result with updated metadata.
        """
        depth = result.depth_m
        if depth is None:
            return result

        finite = np.isfinite(depth)
        safe_depth = np.where(finite, depth, 0.0).astype(np.float32)
        sigma = self.base_sigma_m + self.range_coefficient_m * (
            safe_depth / self.reference_range_m
        ) ** 2
        unit_noise = rng.normal(0.0, 1.0, size=depth.shape).astype(np.float32)
        noisy = safe_depth + unit_noise * sigma.astype(np.float32)
        depth_out = np.where(finite, noisy, np.nan).astype(np.float32)

        meta = dict(result.metadata)
        meta["effect:depth_range_noise"] = {
            "base_sigma_m": self.base_sigma_m,
            "range_coefficient_m": self.range_coefficient_m,
            "reference_range_m": self.reference_range_m,
        }
        return RenderResult(
            time=result.time,
            rgb=result.rgb,
            depth_m=depth_out,
            instance_ids=result.instance_ids,
            metadata=meta,
        )


@dataclass(frozen=True, slots=True)
class DepthEdgeArtifactEffect:
    """Flying-pixel / multipath-style smear near depth discontinuities.

    Real ToF sensors produce spurious mixed-return "flying pixels" at depth
    edges (object silhouettes, horizon) where one sensor element integrates
    returns from two different ranges. This smears a random fraction of
    pixels next to a large *relative* depth jump. Using a relative (not
    absolute) gradient threshold keeps it scale-independent, so it behaves
    sensibly whether depth is meters (a close-range ToF request) or hundreds
    of kilometers (a camera's geometric depth fallback).

    Attributes:
        relative_gradient_threshold: Minimum ``|depth change| / depth`` to
            flag a pixel as an edge.
        flying_pixel_prob: Probability an edge pixel gets smeared.
        smear_frac: Std-dev of the smear, as a fraction of local depth.
    """

    relative_gradient_threshold: float = 0.02
    flying_pixel_prob: float = 0.05
    smear_frac: float = 0.01

    @property
    def name(self) -> str:
        """Effect name."""
        return "depth_edge_artifact"

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Smear a random subset of edge-adjacent pixels.

        Args:
            result: Input result.
            rng: Random generator.

        Returns:
            Result with edge-artifact-corrupted depth.
        """
        depth = result.depth_m
        if depth is None:
            return result

        finite = np.isfinite(depth)
        safe = np.where(finite, depth, 0.0).astype(np.float32)
        denom = np.where(safe > 1.0, safe, 1.0)
        gy = np.abs(np.diff(safe, axis=0, prepend=safe[:1, :])) / denom
        gx = np.abs(np.diff(safe, axis=1, prepend=safe[:, :1])) / denom
        edges = finite & ((gx + gy) > self.relative_gradient_threshold)
        mask = edges & (rng.random(depth.shape) < self.flying_pixel_prob)

        depth_out = depth
        if np.any(mask):
            smear_sigma = safe * self.smear_frac
            unit_noise = rng.normal(0.0, 1.0, size=depth.shape).astype(np.float32)
            smeared = (safe + unit_noise * smear_sigma).astype(np.float32)
            depth_out = np.where(mask, smeared, depth).astype(np.float32)

        meta = dict(result.metadata)
        meta["effect:depth_edge_artifact"] = {
            "relative_gradient_threshold": self.relative_gradient_threshold,
            "flying_pixel_prob": self.flying_pixel_prob,
            "matched_pixels": int(np.sum(mask)),
        }
        return RenderResult(
            time=result.time,
            rgb=result.rgb,
            depth_m=depth_out,
            instance_ids=result.instance_ids,
            metadata=meta,
        )


@dataclass(frozen=True, slots=True)
class DepthQuantizationEffect:
    """Quantize depth to discrete range bins, like a ToF ADC / TDC step.

    Attributes:
        resolution_m: Quantization step [m]. ``<= 0`` disables it.
    """

    resolution_m: float = 1.0

    @property
    def name(self) -> str:
        """Effect name."""
        return "depth_quantization"

    def apply(self, result: RenderResult, rng: np.random.Generator) -> RenderResult:
        """Snap depth to multiples of ``resolution_m``.

        Args:
            result: Input result.
            rng: Unused (quantization is deterministic).

        Returns:
            Result with quantized depth.
        """
        del rng
        depth = result.depth_m
        if depth is None or self.resolution_m <= 0.0:
            return result

        finite = np.isfinite(depth)
        safe = np.where(finite, depth, 0.0)
        quantized = np.round(safe / self.resolution_m) * self.resolution_m
        depth_out = np.where(finite, quantized, np.nan).astype(np.float32)

        meta = dict(result.metadata)
        meta["effect:depth_quantization"] = {"resolution_m": self.resolution_m}
        return RenderResult(
            time=result.time,
            rgb=result.rgb,
            depth_m=depth_out,
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
    "BitDepthQuantizationEffect",
    "DepthEdgeArtifactEffect",
    "DepthQuantizationEffect",
    "DepthRangeNoiseEffect",
    "GaussianBlurEffect",
    "GaussianNoiseEffect",
    "IdentitySensorPipeline",
    "SensorEffect",
    "SensorPipeline",
    "ShotReadNoiseEffect",
    "quantize_uint8",
]
