"""Physical and reference constants used across SatSim.

Values are SI unless noted. Prefer importing named constants over scattering
magic numbers through the codebase. Sources are conventional WGS-84 / IAU
approximations suitable for engineering simulation; higher-fidelity ephemeris
backends may override body parameters via configuration.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Universal / time
# ---------------------------------------------------------------------------

#: Gravitational constant [m^3 kg^-1 s^-2].
G: float = 6.67430e-11

#: Speed of light in vacuum [m s^-1].
C_LIGHT: float = 299_792_458.0

#: Seconds in a Julian day.
SECONDS_PER_DAY: float = 86_400.0

#: Julian Date of the J2000.0 epoch.
JD_J2000: float = 2_451_545.0

# ---------------------------------------------------------------------------
# Earth (WGS-84 / conventional engineering values)
# ---------------------------------------------------------------------------

#: Earth gravitational parameter μ = GM [m^3 s^-2].
MU_EARTH: float = 3.986004418e14

#: Earth equatorial radius [m] (WGS-84).
R_EARTH_EQUATORIAL: float = 6_378_137.0

#: Earth polar radius [m] (WGS-84).
R_EARTH_POLAR: float = 6_356_752.314245

#: Earth flattening factor (WGS-84).
EARTH_FLATTENING: float = 1.0 / 298.257223563

#: Mean Earth radius (spherical approximation) [m].
R_EARTH_MEAN: float = 6_371_000.0

#: Earth sidereal rotation rate [rad s^-1] (approx).
OMEGA_EARTH: float = 7.2921150e-5

#: J2 zonal harmonic (Earth, dimensionless).
J2_EARTH: float = 1.08262668e-3

# ---------------------------------------------------------------------------
# Useful derived scales
# ---------------------------------------------------------------------------

#: Circular LEO reference altitude often used in demos [m] (≈ 550 km).
DEFAULT_LEO_ALTITUDE_M: float = 550_000.0

#: Nominal LEO circular period scale helper: T = 2π √(a³/μ).
def circular_orbit_period_s(semi_major_axis_m: float, mu: float = MU_EARTH) -> float:
    """Return period [s] of a circular Keplerian orbit.

    Args:
        semi_major_axis_m: Semi-major axis in meters.
        mu: Gravitational parameter [m^3 s^-2].

    Returns:
        Orbital period in seconds.
    """
    return 2.0 * math.pi * math.sqrt(semi_major_axis_m**3 / mu)


__all__ = [
    "G",
    "C_LIGHT",
    "SECONDS_PER_DAY",
    "JD_J2000",
    "MU_EARTH",
    "R_EARTH_EQUATORIAL",
    "R_EARTH_POLAR",
    "EARTH_FLATTENING",
    "R_EARTH_MEAN",
    "OMEGA_EARTH",
    "J2_EARTH",
    "DEFAULT_LEO_ALTITUDE_M",
    "circular_orbit_period_s",
]
