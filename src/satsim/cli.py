"""Command-line entry point for SatSim.

The CLI is intentionally thin at scaffold time: it exposes version/help and a
placeholder for scenario runs. Heavier orchestration lives in
``satsim.application`` and ``scripts/``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from satsim import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` for the ``satsim`` console
        script.
    """
    parser = argparse.ArgumentParser(
        prog="satsim",
        description=(
            "SatSim — satellite constellation simulator with synthetic imagery "
            "and computer-vision-driven autonomy."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = sub.add_parser(
        "run",
        help="Run a simulation scenario from a YAML config (scaffold stub).",
    )
    run_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to scenario configuration file.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print plan without executing dynamics.",
    )

    sub.add_parser(
        "info",
        help="Print package version and layer overview.",
    )

    return parser


def _cmd_info() -> int:
    """Print a short package overview."""
    print(f"SatSim v{__version__}")
    print("Layers: core | application | infrastructure")
    print("Docs:   see README.md")
    return 0


def _cmd_run(config: str, dry_run: bool) -> int:
    """Load config and run (or dry-run) a scenario.

    Args:
        config: Path to YAML configuration.
        dry_run: If True, validate and describe without dynamics.

    Returns:
        Process exit code (0 on success).
    """
    from pathlib import Path

    from satsim.application.simulation.scenario import ScenarioRunner
    from satsim.infrastructure.io.config import load_scenario_config

    path = Path(config)
    scenario = load_scenario_config(path)
    runner = ScenarioRunner(scenario, config_path=path)
    plan = runner.describe()
    mode = "dry-run" if dry_run else "execute"
    print(f"[satsim] scenario {mode}: name={plan['name']!r} steps={plan['num_steps']}")
    history = runner.run(dry_run=dry_run)
    if dry_run:
        print("[satsim] dry-run complete (no dynamics).")
        return 0
    n_dyn = max(0, len(history) - 1)
    final = history[-1]
    print(
        f"[satsim] done: dynamics_steps={n_dyn} final_t={final.time.seconds:.3f}s "
        f"sats={len(final.states)} truncated={final.truncated}"
    )

    total_detections = sum(len(bundle.detections) for r in history for bundle in r.observations)
    first_detection_t = next(
        (r.time.seconds for r in history if any(b.detections for b in r.observations)),
        None,
    )
    tasking_events = sum(
        1
        for r in history
        for action in r.infos.get("agent_actions", {}).values()
        if action.get("kind") == "task"
    )
    print(
        f"[satsim] perception: total_detections={total_detections} "
        f"first_detection_t={first_detection_t}"
    )
    print(f"[satsim] tasking: reactive_task_events={tasking_events}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entry.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "info":
        return _cmd_info()
    if args.command == "run":
        return _cmd_run(config=args.config, dry_run=args.dry_run)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
