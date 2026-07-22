"""Reference frame labels and conversion hooks.

Concrete DCM / quaternion transforms between frames will live here as the
ephemeris stack matures. For now we define a controlled vocabulary of frame
names used across state objects and sensor pose metadata.
"""

from __future__ import annotations

from enum import Enum


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


__all__ = ["ReferenceFrame"]
