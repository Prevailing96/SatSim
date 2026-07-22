"""Synthetic scene generation and image formation backends."""

from __future__ import annotations

from satsim.infrastructure.rendering.renderer import (
    PlaceholderRenderer,
    RenderRequest,
    RenderResult,
    SceneRenderer,
)
from satsim.infrastructure.rendering.scene import SceneDescriptor, SceneObject

__all__ = [
    "PlaceholderRenderer",
    "RenderRequest",
    "RenderResult",
    "SceneDescriptor",
    "SceneObject",
    "SceneRenderer",
]
