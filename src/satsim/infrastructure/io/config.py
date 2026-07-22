"""YAML configuration loading and mapping into typed scenario configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from satsim.application.simulation.scenario import ScenarioConfig


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary.

    Args:
        path: Filesystem path to a YAML document.

    Returns:
        Parsed mapping (empty dict if the document is empty).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ImportError: If PyYAML is not installed.
        ValueError: If the root document is not a mapping.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        msg = "PyYAML is required to load configs; install the base satsim package."
        raise ImportError(msg) from exc

    file_path = Path(path)
    if not file_path.is_file():
        msg = f"Config file not found: {file_path}"
        raise FileNotFoundError(msg)

    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"Config root must be a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def load_scenario_config(path: Path | str) -> ScenarioConfig:
    """Load a scenario YAML into :class:`ScenarioConfig`.

    Expected keys (all optional with defaults)::

        name: str
        simulation:
          duration_s: float
          dt_s: float
          seed: int
        constellation: path or mapping (stored as path string if str)
        sensors: path or mapping
        vision: path or mapping

    Args:
        path: Path to scenario YAML.

    Returns:
        Typed scenario configuration.
    """
    raw = load_yaml(path)
    sim = raw.get("simulation", {})
    if sim is None:
        sim = {}
    if not isinstance(sim, dict):
        msg = "simulation section must be a mapping"
        raise ValueError(msg)

    def _as_optional_path(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        # Embedded mapping — not expanded here; stash marker in extras.
        return None

    extras: dict[str, Any] = {}
    for key in ("constellation", "sensors", "vision"):
        if key in raw and not isinstance(raw[key], str):
            extras[key] = raw[key]

    n_sats = sim.get("n_satellites", raw.get("n_satellites", 3))
    return ScenarioConfig(
        name=str(raw.get("name", Path(path).stem)),
        duration_s=float(sim.get("duration_s", 600.0)),
        dt_s=float(sim.get("dt_s", 1.0)),
        seed=int(sim.get("seed", 0)),
        constellation_config=_as_optional_path(raw.get("constellation")),
        sensor_config=_as_optional_path(raw.get("sensors")),
        vision_config=_as_optional_path(raw.get("vision")),
        n_satellites=int(n_sats),
        extras=extras,
    )


__all__ = ["load_scenario_config", "load_yaml"]
