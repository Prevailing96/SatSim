"""Scene description for synthetic rendering.

Scenes are geometric + material descriptions independent of the render
backend (raster, path tracer, procedural Earth texture, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from satsim.core.types.vectors import AttitudeQuaternion, Vec3


@dataclass(frozen=True, slots=True)
class SceneObject:
    """A renderable object in the world.

    Attributes:
        object_id: Stable id within the scene.
        name: Human-readable name / class label for CV ground truth.
        position_m: Position in the scene reference frame [m].
        attitude: Orientation in the scene reference frame.
        mesh_uri: Optional path/URI to mesh or procedural primitive tag.
        scale: Uniform or per-axis scale factors.
        metadata: Extra material / semantic fields.
    """

    object_id: str
    name: str
    position_m: Vec3
    attitude: AttitudeQuaternion | None = None
    mesh_uri: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneDescriptor:
    """Full scene graph snapshot at a simulation time.

    Attributes:
        frame: Reference frame name for object poses.
        objects: Renderable objects.
        background: Background model tag (e.g. ``\"earth_texture\"``, ``\"stars\"``).
        lighting: Lighting parameters (sun direction, irradiance, etc.).
        metadata: Free-form scene notes.
    """

    frame: str
    objects: tuple[SceneObject, ...] = ()
    background: str = "earth_texture"
    lighting: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["SceneDescriptor", "SceneObject"]
