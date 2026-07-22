#!/usr/bin/env python3
"""Run a SatSim scenario from a YAML config.

Usage::

    python scripts/run_scenario.py --config configs/default.yaml
    python scripts/run_scenario.py --config configs/simulation/fast_smoke.yaml --dry-run

This script is the operator-facing entry for experiments. As the environment
is completed, it will construct adapters and call
:class:`~satsim.application.simulation.scenario.ScenarioRunner`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for scenario execution."""
    # Ensure src layout works even without editable install in casual use.
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from satsim.application.simulation.scenario import ScenarioRunner
    from satsim.infrastructure.io.config import load_scenario_config
    from satsim.infrastructure.logging.setup import configure_logging, get_logger

    parser = argparse.ArgumentParser(description="Run a SatSim scenario")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=root / "configs" / "default.yaml",
        help="Path to scenario YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print plan without executing dynamics",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    log = get_logger("satsim.scripts.run_scenario")

    config_path = args.config if args.config.is_absolute() else root / args.config
    log.info("Loading config from %s", config_path)
    scenario = load_scenario_config(config_path)
    runner = ScenarioRunner(scenario, config_path=config_path)

    plan = runner.describe()
    log.info("Scenario plan:\n%s", json.dumps(plan, indent=2))

    if args.dry_run:
        log.info("Dry-run complete (no dynamics executed).")
        return 0

    history = runner.run(dry_run=False)
    # history[0] is reset; remaining entries are dynamics steps
    n_dyn = max(0, len(history) - 1)
    final = history[-1] if history else None
    log.info(
        "Completed %d dynamics steps (%d snapshots including reset)",
        n_dyn,
        len(history),
    )
    if final is not None:
        log.info(
            "Final time=%.3f s | satellites=%d | truncated=%s",
            final.time.seconds,
            len(final.states),
            final.truncated,
        )
        for sat_id, state in sorted(final.states.items()):
            r = state.cartesian.position_m.norm()
            v = state.cartesian.velocity_m_s.norm()
            log.info("  %s: |r|=%.3f km  |v|=%.3f m/s", sat_id, r / 1000.0, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
