"""Core domain layer: pure types, orbital mechanics, and sensor contracts.

This package must remain free of heavy optional dependencies (PyTorch, OpenCV
render backends, filesystem-heavy I/O). Application and infrastructure layers
adapt these abstractions to concrete runtimes.
"""

from __future__ import annotations
