"""Analytic two-body propagation via the universal-variable method.

Implements the classical Lagrange f and g formulation with Stumpff functions
(see Vallado, *Fundamentals of Astrodynamics and Applications*). Works for
circular, elliptical, parabolic, and hyperbolic regimes under a point-mass
central body.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from satsim.core.constants import MU_EARTH

# Convergence and numerical guards
_CHI_TOL = 1e-10
_MAX_NEWTON_ITERS = 50
_MIN_RADIUS_M = 1.0  # reject singular states near the origin


def stumpff_c2(z: float) -> float:
    """Stumpff c2(z) function.

    Args:
        z: Argument ``α χ²`` (dimensionless).

    Returns:
        ``c2(z)``.
    """
    if z > 1e-8:
        sqrt_z = math.sqrt(z)
        return (1.0 - math.cos(sqrt_z)) / z
    if z < -1e-8:
        sqrt_neg = math.sqrt(-z)
        return (1.0 - math.cosh(sqrt_neg)) / z
    # Series: 1/2! - z/4! + z²/6! - ...
    return 0.5 - z / 24.0 + (z * z) / 720.0


def stumpff_c3(z: float) -> float:
    """Stumpff c3(z) function.

    Args:
        z: Argument ``α χ²`` (dimensionless).

    Returns:
        ``c3(z)``.
    """
    if z > 1e-8:
        sqrt_z = math.sqrt(z)
        return (sqrt_z - math.sin(sqrt_z)) / (z * sqrt_z)
    if z < -1e-8:
        sqrt_neg = math.sqrt(-z)
        return (math.sinh(sqrt_neg) - sqrt_neg) / ((-z) * sqrt_neg)
    # Series: 1/3! - z/5! + z²/7! - ...
    return (1.0 / 6.0) - z / 120.0 + (z * z) / 5040.0


def specific_energy(
    r: NDArray[np.float64],
    v: NDArray[np.float64],
    mu: float = MU_EARTH,
) -> float:
    """Specific mechanical energy ε = v²/2 − μ/r [m²/s²].

    Args:
        r: Position vector [m], shape ``(3,)``.
        v: Velocity vector [m/s], shape ``(3,)``.
        mu: Gravitational parameter [m³/s²].

    Returns:
        Specific energy (negative for bound orbits).
    """
    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))
    if r_mag < _MIN_RADIUS_M:
        msg = f"Position magnitude too small for energy: {r_mag}"
        raise ValueError(msg)
    return 0.5 * v_mag * v_mag - mu / r_mag


def semi_major_axis(
    r: NDArray[np.float64],
    v: NDArray[np.float64],
    mu: float = MU_EARTH,
) -> float:
    """Semi-major axis from Cartesian state [m].

    Args:
        r: Position [m].
        v: Velocity [m/s].
        mu: Gravitational parameter.

    Returns:
        Semi-major axis ``a`` (negative for hyperbolic).
    """
    eps = specific_energy(r, v, mu)
    if abs(eps) < 1e-18:
        return math.inf
    return -mu / (2.0 * eps)


def propagate_rv(
    r0: NDArray[np.floating] | list[float],
    v0: NDArray[np.floating] | list[float],
    dt_s: float,
    mu: float = MU_EARTH,
    *,
    tol: float = _CHI_TOL,
    max_iters: int = _MAX_NEWTON_ITERS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate a Cartesian state under two-body dynamics.

    Args:
        r0: Initial position [m].
        v0: Initial velocity [m/s].
        dt_s: Propagation time [s] (negative for reverse).
        mu: Central-body gravitational parameter [m³/s²].
        tol: Newton convergence tolerance on universal variable χ.
        max_iters: Maximum Newton iterations.

    Returns:
        ``(r, v)`` at ``t0 + dt_s``, float64 arrays shape ``(3,)``.

    Raises:
        ValueError: If the state is singular or Newton fails to converge.
    """
    if abs(dt_s) < 1e-15:
        return (
            np.asarray(r0, dtype=np.float64).copy(),
            np.asarray(v0, dtype=np.float64).copy(),
        )

    r0_vec = np.asarray(r0, dtype=np.float64).reshape(3)
    v0_vec = np.asarray(v0, dtype=np.float64).reshape(3)
    r0_mag = float(np.linalg.norm(r0_vec))
    if r0_mag < _MIN_RADIUS_M:
        msg = f"Initial radius too small: {r0_mag} m"
        raise ValueError(msg)
    if mu <= 0.0:
        msg = "mu must be positive"
        raise ValueError(msg)

    v0_mag2 = float(np.dot(v0_vec, v0_vec))
    r0_dot_v0 = float(np.dot(r0_vec, v0_vec))
    # α = 1/a = 2/r − v²/μ
    alpha = 2.0 / r0_mag - v0_mag2 / mu

    chi = _initial_chi_guess(dt_s, r0_mag, r0_dot_v0, alpha, mu)
    sqrt_mu = math.sqrt(mu)

    for _ in range(max_iters):
        z = alpha * chi * chi
        c2 = stumpff_c2(z)
        c3 = stumpff_c3(z)
        # Universal Kepler equation residual
        # √μ Δt = χ³ c3 + (r0·v0/√μ) χ² c2 + r0 χ (1 − z c3)
        f_chi = (
            (chi**3) * c3
            + (r0_dot_v0 / sqrt_mu) * (chi**2) * c2
            + r0_mag * chi * (1.0 - z * c3)
            - sqrt_mu * dt_s
        )
        # dF/dχ
        df_dchi = (
            (chi**2) * c2
            + (r0_dot_v0 / sqrt_mu) * chi * (1.0 - z * c3)
            + r0_mag * (1.0 - z * c2)
        )
        if abs(df_dchi) < 1e-30:
            msg = "Singular derivative in universal-variable Newton solve"
            raise ValueError(msg)
        delta = f_chi / df_dchi
        chi = chi - delta
        if abs(delta) < tol:
            break
    else:
        msg = (
            f"Universal-variable Newton failed to converge in {max_iters} "
            f"iterations (dt={dt_s}, |r0|={r0_mag})"
        )
        raise ValueError(msg)

    z = alpha * chi * chi
    c2 = stumpff_c2(z)
    c3 = stumpff_c3(z)

    # Lagrange coefficients
    f = 1.0 - (chi * chi / r0_mag) * c2
    g = dt_s - (chi**3 / sqrt_mu) * c3
    r_vec = f * r0_vec + g * v0_vec
    r_mag = float(np.linalg.norm(r_vec))
    if r_mag < _MIN_RADIUS_M:
        msg = f"Propagated radius too small: {r_mag} m"
        raise ValueError(msg)

    gdot = 1.0 - (chi * chi / r_mag) * c2
    fdot = (sqrt_mu / (r_mag * r0_mag)) * chi * (z * c3 - 1.0)
    v_vec = fdot * r0_vec + gdot * v0_vec

    # f ġ − ḟ g = 1 identity check is soft; return state
    return r_vec.astype(np.float64), v_vec.astype(np.float64)


def _initial_chi_guess(
    dt_s: float,
    r0_mag: float,
    r0_dot_v0: float,
    alpha: float,
    mu: float,
) -> float:
    """Initial guess for the universal variable χ.

    Args:
        dt_s: Time of flight [s].
        r0_mag: Initial radius [m].
        r0_dot_v0: ``r0 · v0`` [m²/s].
        alpha: Reciprocal semi-major axis [1/m].
        mu: Gravitational parameter.

    Returns:
        Starting χ for Newton iteration.
    """
    sqrt_mu = math.sqrt(mu)
    if alpha > 1e-12:
        # Elliptical: χ ≈ √μ α Δt  (Vallado)
        return sqrt_mu * alpha * dt_s
    if alpha < -1e-12:
        # Hyperbolic (Curtis / Vallado)
        a = 1.0 / alpha
        sign_dt = math.copysign(1.0, dt_s)
        denom = r0_dot_v0 + sign_dt * math.sqrt(-mu * a) * (1.0 - r0_mag * alpha)
        if abs(denom) < 1e-30:
            return sign_dt * math.sqrt(-a) * math.log(1.0 + abs(dt_s))
        arg = (-2.0 * mu * alpha * dt_s) / denom
        if arg <= 0.0:
            return sign_dt * sqrt_mu * abs(dt_s) / max(r0_mag, 1.0)
        return sign_dt * math.sqrt(-a) * math.log(arg)
    # Near-parabolic
    return sqrt_mu * dt_s / r0_mag


def circular_leo_state(
    altitude_m: float,
    *,
    mu: float = MU_EARTH,
    inclination_rad: float = 0.0,
    raan_rad: float = 0.0,
    true_anomaly_rad: float = 0.0,
    earth_radius_m: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a circular-orbit Cartesian state in ECI-like inertial frame.

    Places the satellite at true anomaly on a circular orbit of the given
    altitude, with optional inclination and RAAN (simple rotation sequence).

    Args:
        altitude_m: Height above spherical Earth [m].
        mu: Gravitational parameter.
        inclination_rad: Inclination [rad].
        raan_rad: Right ascension of ascending node [rad].
        true_anomaly_rad: True anomaly (argument of latitude for e=0) [rad].
        earth_radius_m: Central body radius [m]; defaults to equatorial R_E.

    Returns:
        ``(r, v)`` float64 arrays in the inertial frame [m], [m/s].
    """
    from satsim.core.constants import R_EARTH_EQUATORIAL

    r_body = R_EARTH_EQUATORIAL if earth_radius_m is None else earth_radius_m
    radius = r_body + altitude_m
    if radius <= 0.0:
        msg = "orbit radius must be positive"
        raise ValueError(msg)

    speed = math.sqrt(mu / radius)
    # Perifocal: r along x', v along y' at ν=0; rotate by true anomaly first
    cos_nu = math.cos(true_anomaly_rad)
    sin_nu = math.sin(true_anomaly_rad)
    r_pqw = np.array([radius * cos_nu, radius * sin_nu, 0.0], dtype=np.float64)
    v_pqw = np.array([-speed * sin_nu, speed * cos_nu, 0.0], dtype=np.float64)

    cos_o = math.cos(raan_rad)
    sin_o = math.sin(raan_rad)
    cos_i = math.cos(inclination_rad)
    sin_i = math.sin(inclination_rad)
    # R3(Ω) R1(i) R3(ω=0)
    rot = np.array(
        [
            [cos_o, -sin_o * cos_i, sin_o * sin_i],
            [sin_o, cos_o * cos_i, -cos_o * sin_i],
            [0.0, sin_i, cos_i],
        ],
        dtype=np.float64,
    )
    r_eci = rot @ r_pqw
    v_eci = rot @ v_pqw
    return r_eci, v_eci


__all__ = [
    "circular_leo_state",
    "propagate_rv",
    "semi_major_axis",
    "specific_energy",
    "stumpff_c2",
    "stumpff_c3",
]
