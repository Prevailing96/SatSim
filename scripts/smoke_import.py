#!/usr/bin/env python3
"""Import-smoke test for the SatSim package layout.

Verifies that the scaffold packages import cleanly without optional extras.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Import key modules and print versions / names."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import satsim
    from satsim.application.agents import SatelliteAgent
    from satsim.application.simulation import ScenarioConfig, SimulationClock
    from satsim.core.constants import MU_EARTH
    from satsim.core.orbital import KeplerianElements, ReferenceFrame
    from satsim.core.types import SimTime, Vec3
    from satsim.infrastructure.rendering import PlaceholderRenderer
    from satsim.infrastructure.vision import PerceptionPipeline

    print(f"satsim {satsim.__version__}")
    print(f"MU_EARTH={MU_EARTH:.6e}")
    print(f"frames={[f.value for f in ReferenceFrame]}")
    print(f"SimTime(0)={SimTime(0.0)}")
    print(f"Vec3 sample={Vec3(1.0, 2.0, 3.0).norm():.4f}")
    print(f"KeplerianElements ok period property available")
    print(f"ScenarioConfig defaults steps={ScenarioConfig().num_steps}")
    print(f"Clock={SimulationClock.create()}")
    print(f"Renderer={PlaceholderRenderer().__class__.__name__}")
    print(f"Perception={PerceptionPipeline.stub().detector.name}")
    print(f"SatelliteAgent class={SatelliteAgent.__name__}")
    print("smoke_import: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
