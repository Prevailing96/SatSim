"""Classical Keplerian orbital elements."""

from __future__ import annotations

from dataclasses import dataclass

from satsim.core.constants import MU_EARTH


@dataclass(frozen=True, slots=True)
class KeplerianElements:
    """Classical orbital elements (Keplerian set).

    Angles are stored in **radians**. Semi-major axis is in **meters**.

    Attributes:
        semi_major_axis_m: Semi-major axis ``a`` [m].
        eccentricity: Eccentricity ``e`` (dimensionless, ``0 <= e < 1`` for
            closed orbits in the two-body sense).
        inclination_rad: Inclination ``i`` [rad].
        raan_rad: Right ascension of ascending node ``Ω`` [rad].
        arg_perigee_rad: Argument of perigee ``ω`` [rad].
        true_anomaly_rad: True anomaly ``ν`` [rad].
        mu: Gravitational parameter of the central body [m^3 s^-2].
    """

    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_perigee_rad: float
    true_anomaly_rad: float
    mu: float = MU_EARTH

    def validate(self) -> None:
        """Validate basic physical constraints.

        Raises:
            ValueError: If elements are outside accepted ranges for a bound
                elliptical orbit (scaffold-level checks).
        """
        if self.semi_major_axis_m <= 0.0:
            msg = "semi_major_axis_m must be positive"
            raise ValueError(msg)
        if self.eccentricity < 0.0:
            msg = "eccentricity must be non-negative"
            raise ValueError(msg)
        if self.eccentricity >= 1.0:
            msg = (
                "eccentricity >= 1 is not supported by the elliptical scaffold; "
                "use a specialized hyperbolic/parabolic path later"
            )
            raise ValueError(msg)
        if self.mu <= 0.0:
            msg = "mu must be positive"
            raise ValueError(msg)

    @property
    def period_s(self) -> float:
        """Keplerian orbital period [s] for an elliptical orbit."""
        import math

        return 2.0 * math.pi * math.sqrt(self.semi_major_axis_m**3 / self.mu)

    @property
    def periapsis_radius_m(self) -> float:
        """Periapsis radius ``a(1-e)`` [m]."""
        return self.semi_major_axis_m * (1.0 - self.eccentricity)

    @property
    def apoapsis_radius_m(self) -> float:
        """Apoapsis radius ``a(1+e)`` [m]."""
        return self.semi_major_axis_m * (1.0 + self.eccentricity)


__all__ = ["KeplerianElements"]
