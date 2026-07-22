"""Scenario configuration and runner orchestration.

``ScenarioRunner`` builds a :class:`TwoBodyEnvironment`, resets it, and steps
until the configured duration (or an early terminal condition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satsim.application.simulation.clock import SimulationClock
from satsim.application.simulation.environment import StepResult
from satsim.application.simulation.two_body_env import (
    TwoBodyEnvironment,
    default_demo_constellation,
)
from satsim.core.types.time import SimTime


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """High-level scenario parameters (validated subset).

    Full YAML maps are richer; this dataclass is the typed core the runner
    depends on after config loading.

    Attributes:
        name: Human-readable scenario name.
        duration_s: Total simulated duration [s].
        dt_s: Integration / step size [s].
        seed: RNG seed for reproducibility.
        constellation_config: Path or embedded constellation settings.
        sensor_config: Path or embedded sensor settings.
        vision_config: Path or embedded vision settings.
        n_satellites: Demo constellation size when no detailed ICs are loaded.
        extras: Pass-through keys not yet modeled as first-class fields.
    """

    name: str = "default"
    duration_s: float = 600.0
    dt_s: float = 1.0
    seed: int = 0
    constellation_config: str | None = None
    sensor_config: str | None = None
    vision_config: str | None = None
    n_satellites: int = 3
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        """Number of discrete steps for a fixed-step run."""
        if self.dt_s <= 0.0:
            msg = "dt_s must be positive"
            raise ValueError(msg)
        return int(self.duration_s // self.dt_s)


class ScenarioRunner:
    """Orchestrates a scenario lifecycle.

    Current wiring:

    1. Validate config.
    2. Construct :class:`TwoBodyEnvironment` with a demo constellation.
    3. ``reset`` → loop ``step`` until duration (truncated) or ``done``.
    4. Return per-step history (including the reset snapshot as index 0).

    Args:
        config: Typed scenario configuration.
        config_path: Optional source path for logging.
        environment: Optional pre-built environment (tests / custom ICs).
    """

    def __init__(
        self,
        config: ScenarioConfig,
        config_path: Path | str | None = None,
        environment: TwoBodyEnvironment | None = None,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path is not None else None
        self.clock = SimulationClock.create(start_s=0.0, dt_s=config.dt_s)
        self._environment = environment

    def build_environment(self) -> TwoBodyEnvironment:
        """Construct the default environment from scenario config.

        Returns:
            Configured :class:`TwoBodyEnvironment`.
        """
        n = max(1, int(self.config.n_satellites))
        return TwoBodyEnvironment(
            duration_s=self.config.duration_s,
            dt_s=self.config.dt_s,
            satellite_specs=default_demo_constellation(n),
            seed=self.config.seed,
        )

    @property
    def environment(self) -> TwoBodyEnvironment | None:
        """Active environment after :meth:`run`, if any."""
        return self._environment

    def run(self, *, dry_run: bool = False) -> list[StepResult]:
        """Execute the scenario.

        Args:
            dry_run: If True, validate config and return an empty history
                without advancing dynamics.

        Returns:
            History list: ``[reset_snapshot, step_1, ..., step_N]``.

        Raises:
            ValueError: If configuration is inconsistent.
        """
        if self.config.duration_s < 0.0:
            msg = "duration_s must be non-negative"
            raise ValueError(msg)
        if self.config.dt_s <= 0.0:
            msg = "dt_s must be positive"
            raise ValueError(msg)

        if dry_run:
            return []

        env = self._environment or self.build_environment()
        self._environment = env

        history: list[StepResult] = []
        result = env.reset(seed=self.config.seed)
        history.append(result)
        self.clock = SimulationClock.create(start_s=0.0, dt_s=self.config.dt_s)

        max_steps = self.config.num_steps
        for _ in range(max_steps):
            result = env.step()
            history.append(result)
            self.clock.advance()
            if result.done or result.truncated:
                break

        return history

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the planned run.

        Returns:
            Summary dict for CLI / logging.
        """
        return {
            "name": self.config.name,
            "duration_s": self.config.duration_s,
            "dt_s": self.config.dt_s,
            "num_steps": self.config.num_steps,
            "seed": self.config.seed,
            "n_satellites": self.config.n_satellites,
            "config_path": str(self.config_path) if self.config_path else None,
            "start_time_s": self.clock.start.seconds,
            "current_time_s": self.clock.current.seconds,
        }

    @property
    def time(self) -> SimTime:
        """Current clock time."""
        return self.clock.current


__all__ = ["ScenarioConfig", "ScenarioRunner"]
