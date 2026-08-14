"""Reference frame labels and conversion hooks.

Concrete DCM / quaternion transforms between frames will live here as the
ephemeris stack matures. For now we define a controlled vocabulary of frame
names used across state objects and sensor pose metadata, plus a first
geodetic conversion helper for placing ground targets.
"""

from __future__ import annotations

import math
from enum import Enum

from satsim.core.constants import R_EARTH_EQUATORIAL
from satsim.core.types.vectors import Vec3


class ReferenceFrame(str, Enum):
    """Common reference frames used in SatSim state and sensor metadata.

    Values are stable string tags suitable for serialization in configs and
    frame fields on :class:`~satsim.core.types.vectors.CartesianState`.
    """

    ECI_J2000 = "ECI_J2000"
    """Earth-centered inertial, J2000 mean equator/equinox."""

    ECEF = "ECEF"
    """Earth-centered Earth-fixed (WGS-84 aligned, engineering approx)."""

    TEME = "TEME"
    """True Equator Mean Equinox (SGP4-native frame)."""

    LVLH = "LVLH"
    """Local-vertical local-horizontal (orbiting body-relative)."""

    BODY = "BODY"
    """Spacecraft body frame."""

    SENSOR = "SENSOR"
    """Sensor optical / ToF frame (boresight conventions per sensor model)."""

    ENU = "ENU"
    """East-North-Up topocentric frame (ground sites)."""


def geodetic_to_ecef_spherical(
    latitude_rad: float,
    longitude_rad: float,
    *,
    radius_m: float = R_EARTH_EQUATORIAL,
    altitude_m: float = 0.0,
) -> Vec3:
    """Convert planetocentric latitude/longitude to a spherical-Earth ECEF position.

    Uses a spherical (not WGS-84 ellipsoidal) body model, consistent with the
    rest of this scaffold's two-body dynamics. Because the simulation does
    not (yet) model sidereal rotation, :data:`ReferenceFrame.ECEF` and
    :data:`ReferenceFrame.ECI_J2000` coincide numerically — a ground target
    placed here can be compared directly against satellite ECI positions
    without an explicit frame rotation. That equivalence should be
    revisited once a rotating-Earth / epoch-aware ephemeris backend lands.

    Args:
        latitude_rad: Planetocentric latitude [rad], positive north.
        longitude_rad: Longitude [rad], positive east.
        radius_m: Body radius [m]; defaults to equatorial Earth radius.
        altitude_m: Height above the spherical surface [m].

    Returns:
        Position in the (currently ECI-equivalent) ECEF frame [m].
    """
    r = radius_m + altitude_m
    cos_lat = math.cos(latitude_rad)
    x = r * cos_lat * math.cos(longitude_rad)
    y = r * cos_lat * math.sin(longitude_rad)
    z = r * math.sin(latitude_rad)
    return Vec3(x, y, z)


__all__ = ["ReferenceFrame", "geodetic_to_ecef_spherical"]
