"""Infrastructure adapters: rendering, concrete sensors, vision, I/O, logging.

This layer may depend on optional extras (OpenCV, torch, etc.). Core and
application code should depend on protocols, not concrete backends.
"""

from __future__ import annotations
