"""Vector and rigid-body state value objects.

These are lightweight, immutable carriers. Numerical work should convert to
``numpy`` arrays at algorithm boundaries when convenient; keeping domain
objects explicit improves type clarity in interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Vec3:
    """Three-dimensional Cartesian vector in meters (or m/s when velocity).

    Attributes:
        x: X component.
        y: Y component.
        z: Z component.
    """

    x: float
    y: float
    z: float

    def as_array(self) -> NDArray[np.float64]:
        """Return shape ``(3,)`` float64 array ``[x, y, z]``."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.floating] | list[float] | tuple[float, ...]) -> Vec3:
        """Build a :class:`Vec3` from a length-3 sequence or array.

        Args:
            arr: Sequence of three components.

        Returns:
            New :class:`Vec3`.

        Raises:
            ValueError: If ``arr`` does not have exactly three elements.
        """
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        if flat.size != 3:
            msg = f"Vec3 expects 3 elements, got {flat.size}"
            raise ValueError(msg)
        return cls(float(flat[0]), float(flat[1]), float(flat[2]))

    def __iter__(self) -> Iterator[float]:
        """Iterate components ``x, y, z``."""
        yield self.x
        yield self.y
        yield self.z

    def norm(self) -> float:
        """Euclidean norm."""
        return float(np.linalg.norm(self.as_array()))

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> Vec3:
        return Vec3(self.x * scale, self.y * scale, self.z * scale)

    def __rmul__(self, scale: float) -> Vec3:
        return self.__mul__(scale)


@dataclass(frozen=True, slots=True)
class CartesianState:
    """Inertial (or frame-tagged) position and velocity.

    Attributes:
        position_m: Position vector [m].
        velocity_m_s: Velocity vector [m/s].
        frame: Reference frame label (e.g. ``"ECI_J2000"``, ``"ECEF"``).
    """

    position_m: Vec3
    velocity_m_s: Vec3
    frame: str = "ECI_J2000"

    def as_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(r, v)`` as float64 arrays of shape ``(3,)``."""
        return self.position_m.as_array(), self.velocity_m_s.as_array()


@dataclass(frozen=True, slots=True)
class AttitudeQuaternion:
    """Unit quaternion attitude (scalar-first: ``w, x, y, z``).

    Attributes:
        w: Scalar part.
        x: Vector i component.
        y: Vector j component.
        z: Vector k component.
        frame_from: Body frame name.
        frame_to: Reference frame name this quaternion rotates into.
    """

    w: float
    x: float
    y: float
    z: float
    frame_from: str = "BODY"
    frame_to: str = "ECI_J2000"

    def as_array(self) -> NDArray[np.float64]:
        """Return ``[w, x, y, z]``."""
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    def normalized(self) -> AttitudeQuaternion:
        """Return a copy scaled to unit length.

        Returns:
            Unit quaternion with the same frame tags.

        Raises:
            ValueError: If the quaternion has (near) zero norm.
        """
        arr = self.as_array()
        n = float(np.linalg.norm(arr))
        if n < 1e-15:
            msg = "Cannot normalize near-zero quaternion"
            raise ValueError(msg)
        w, x, y, z = (arr / n).tolist()
        return AttitudeQuaternion(
            w=w,
            x=x,
            y=y,
            z=z,
            frame_from=self.frame_from,
            frame_to=self.frame_to,
        )


__all__ = ["AttitudeQuaternion", "CartesianState", "Vec3"]
