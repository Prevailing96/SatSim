"""Export helpers for observations, labels, and run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from satsim.core.sensors.observations import ObservationBundle


def export_observation_bundle(
    bundle: ObservationBundle,
    directory: Path | str,
    *,
    prefix: str = "obs",
) -> dict[str, Path]:
    """Export an observation bundle to a directory (metadata + arrays).

    Writes:

    * ``{prefix}_meta.json`` — detections, ids, timestamps
    * ``{prefix}_rgb.npy`` — if image present
    * ``{prefix}_depth.npy`` — if depth present
    * ``{prefix}_seg.npy`` — if segmentation present

    Args:
        bundle: Observation products to export.
        directory: Output directory (created if missing).
        prefix: Filename prefix.

    Returns:
        Mapping of artifact role → written path.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    meta: dict[str, Any] = {
        "sensor_id": str(bundle.sensor_id),
        "satellite_id": str(bundle.satellite_id),
        "time_s": bundle.time.seconds,
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox_xyxy": [d.x_min, d.y_min, d.x_max, d.y_max],
                "track_id": d.track_id,
            }
            for d in bundle.detections
        ],
        "has_image": bundle.image is not None,
        "has_depth": bundle.depth is not None,
        "has_segmentation": bundle.segmentation is not None,
    }

    meta_path = out_dir / f"{prefix}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written["meta"] = meta_path

    # Lazy numpy save to avoid forcing import patterns at module import for
    # callers that only want types — but ObservationBundle already uses numpy.
    import numpy as np

    if bundle.image is not None:
        p = out_dir / f"{prefix}_rgb.npy"
        np.save(p, bundle.image.data)
        written["rgb"] = p

    if bundle.depth is not None:
        p = out_dir / f"{prefix}_depth.npy"
        np.save(p, bundle.depth.depth_m)
        written["depth"] = p

    if bundle.segmentation is not None:
        p = out_dir / f"{prefix}_seg.npy"
        np.save(p, bundle.segmentation.mask)
        written["segmentation"] = p

    return written


__all__ = ["export_observation_bundle"]
