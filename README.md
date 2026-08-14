# SatSim

**Scalable satellite constellation simulation with synthetic imaging, sensor modeling, and computer-vision-driven autonomy.**

SatSim is a professional-grade research and engineering platform for studying how fleets of satellites sense, decide, and act. It couples two-body orbital dynamics with ray-traced, sensor-realistic synthetic imagery — shaded Earth, geographically placed ground targets, RGB and depth channels — then closes the loop through a classical computer-vision perception stack that drives per-satellite tasking and constellation-wide deconfliction.

The full loop runs today: `satsim run -c configs/default.yaml` propagates a 3-satellite constellation, renders and perceives each satellite's camera view every step, and reacts to what it finds.

```text
[satsim] perception: total_detections=83 first_detection_t=385.0
[satsim] tasking: reactive_task_events=3
```

(83 raw detections across three targets and three satellites, but only 3 tasking events — the constellation shares task awareness, so the same target isn't tasked twice. See [Core concepts](#core-concepts).)

---

## Vision

Modern constellation operations are no longer pure orbital mechanics problems. They are **perception–decision–action** problems:

1. **Dynamics** — satellites move under gravity, drag, and control; relative geometry changes continuously.
2. **Sensing** — EO/IR, ToF, and other sensors produce noisy, biased, resolution-limited observations of Earth and of other space objects.
3. **Perception** — detectors and segmenters extract objects, tracks, and scene understanding from those observations.
4. **Autonomy** — tasking, retasking, and multi-agent coordination must react to what the constellation *sees*, not only to what the planner assumed a priori.

SatSim exists to make that full loop **simulatable, reproducible, and extensible** — from single-satellite sensor studies to multi-agent constellation autonomy and, eventually, reinforcement learning over tasking policies.

```text
  ┌──────────────┐     state      ┌────────────────┐     frames     ┌─────────────┐
  │   Orbital    │ ─────────────► │   Synthetic    │ ─────────────► │  Perception │
  │  Dynamics    │                │  Rendering +   │                │  (detect /  │
  │              │ ◄───────────── │  Sensor models │ ◄───────────── │  segment)   │
  └──────────────┘   tasking /    └────────────────┘   detections   └─────────────┘
                     autonomy                           + masks
```

### Design principles

| Principle | What it means in SatSim |
|-----------|-------------------------|
| **Modular by layer** | Domain logic (`core/`) stays pure; use-cases live in `application/`; I/O, rendering, and ML adapters live in `infrastructure/`. |
| **Closed-loop ready** | CV outputs are first-class events that can drive tasking and agent policies — not just offline metrics. |
| **Sensor realism first** | Synthetic data is useful only if noise, PSF, quantization, ToF artifacts, and calibration error are modeled deliberately. |
| **Type-safe & tested** | Strict typing, documented public APIs, and a test layout that grows with fidelity. |
| **RL / multi-agent friendly** | Environment and agent interfaces are designed so Gymnasium / PettingZoo-style wrappers can land later without rewrites. |
| **Config-driven experiments** | Scenario, constellation, sensor, and vision settings live under `configs/` — code stays general. |

---

## Architecture

```text
SatSim/
├── src/satsim/
│   ├── core/                 # Pure domain: math, orbital state, interfaces
│   │   ├── orbital/          # Propagation, frames, elements, ephemeris types
│   │   ├── physics/          # Force models, attitude (stubs → full models)
│   │   ├── sensors/          # Sensor abstract models & measurement types
│   │   └── types/            # Shared value objects, time, IDs, units helpers
│   ├── application/          # Use-cases and orchestration
│   │   ├── simulation/       # Scenario runner, clock, event loop
│   │   ├── tasking/          # Observation requests, prioritization, schedules
│   │   └── agents/           # Satellite / constellation agents & policies
│   ├── infrastructure/       # Adapters: render, ML, I/O, logging
│   │   ├── rendering/        # Synthetic scene generation & image formation
│   │   ├── sensors/          # Concrete sensor effect pipelines (incl. ToF)
│   │   ├── vision/           # Detectors, segmenters, tracking adapters
│   │   ├── io/               # Config load, datasets, ephemeris, export
│   │   └── logging/          # Structured logging setup
│   └── cli.py                # Entry point: `satsim` console script
├── configs/                  # YAML experiment / scenario configs
├── scripts/                  # Operator scripts (smoke runs, demos)
├── tests/                    # unit / integration / property tests
├── pyproject.toml
└── README.md
```

### Layer responsibilities

- **`core`** — Framework-agnostic types and algorithms. No PyTorch, no file I/O, no rendering backends. Safe to unit-test in isolation.
- **`application`** — Orchestrates a simulation step: advance time → propagate → render/sense → perceive → task/act.
- **`infrastructure`** — Pluggable implementations: renderer backends, YOLO/segment-anything adapters, YAML loaders, log sinks.

This separation keeps orbital math free of ML dependencies and keeps vision code free of Keplerian edge cases.

---

## Feature roadmap (initial → mature)

| Area | Where it stands today | Direction |
|------|------------------------|-----------|
| **Orbital dynamics** | Real universal-variable two-body propagator (elliptical/parabolic/hyperbolic); Keplerian elements | High-order integrators, J2/drag, multi-body, maneuver burns |
| **Constellations** | Evenly-phased demo constellation built in code; Walker-Delta config exists but isn't parsed into it yet | Full YAML → constellation wiring, coverage metrics, inter-sat links |
| **Synthetic data** | Ray-traced Earth sphere (Lambertian shading, limb darkening, atmospheric limb glow) + geometrically consistent depth; geographic (lat/lon) multi-target placement with limb occlusion | Textures, path-traced / raster EO, BRDF, cloud layers |
| **Sensors** | Real effect pipeline: PSF blur, shot+read noise, 8-bit quantization (RGB); range-dependent noise, flying-pixel/multipath artifacts, range quantization (depth) | MTF, dedicated ToF sensor requests, calibration error models |
| **Perception** | `RuleBasedBlobDetector`: color threshold + connected components + geometric gating (solidity, aspect ratio, size) + heuristic confidence; multi-target aware | Learned detectors (YOLO / torch) under the `vision` extra, tracking |
| **Autonomy** | `SatelliteAgent` reacts to detections; `PriorityQueueScheduler` shared across the constellation as a task board with spatial deconfliction (no double-tasking the same target) | Richer policies, visibility/power-aware scheduling, task completion lifecycle |
| **RL** | Environment protocol hooks | Gymnasium / PettingZoo wrappers, curriculum scenarios |

---

## Quick start

### Requirements

- Python **3.11+**
- [uv](https://github.com/astral-sh/uv) (recommended) or pip / Poetry

### Install (uv)

```bash
# Clone and enter
cd SatSim

# Create env and install package + dev tools
uv venv
uv pip install -e ".[dev]"

# Optional stacks
uv pip install -e ".[astro,render,vision]"
# or everything:
uv pip install -e ".[all]"
```

### Install (pip)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
```

### Smoke check

```bash
# CLI banner / version
satsim --help

# Or via module
python -m satsim --help

# Run unit tests
pytest -m unit
```

### Run a scenario

```bash
# Via the console script (prints detection/tasking summary)
satsim run -c configs/default.yaml

# Or the operator script (structured logging, same underlying ScenarioRunner)
python scripts/run_scenario.py --config configs/default.yaml
```

---

## Core concepts

### Simulation step (closed loop)

Every `TwoBodyEnvironment.reset()` / `.step()` call runs the full loop, not a stub of it:

1. **Advance** — propagate every satellite's orbital state (real two-body dynamics).
2. **Sense** — ray-trace each satellite's nadir camera against Earth and the scene's ground targets (`PlaceholderRenderer`), then run the sensor-effect pipeline (blur, shot/read noise, quantization for RGB; range-dependent noise, flying-pixel artifacts, quantization for depth).
3. **Perceive** — `RuleBasedBlobDetector` finds and shape-filters candidate targets, scoring each with a heuristic confidence.
4. **Decide** — each satellite's `SatelliteAgent` reacts to qualifying detections, submitting `TaskRequest`s to a `PriorityQueueScheduler` **shared across the whole constellation** — so a satellite checks constellation-wide task state before tasking a target another satellite already has covered.
5. **Act** *(future)* — attitude slews, mode changes, Δv maneuvers.

Perception is not an offline post-process; it is a real input to autonomy every step, and `StepResult.infos` exposes the outcome (`n_detections`, `agent_actions`, `constellation_active_tasks`) for inspection or logging.

### Synthetic imagery & sensor realism

SatSim treats depth as a first-class modality alongside electro-optical imagery, and both are geometrically and radiometrically real rather than placeholders:

- **Rendering** (`infrastructure/rendering`): ray/sphere intersection against Earth with Lambertian shading from a fixed sun direction, linear limb darkening, and a soft atmospheric glow at the disk edge in wide-FOV views; scene objects are projected as marker patches with proper Earth-limb occlusion.
- **Sensor effects** (`infrastructure/sensors`): a composable `SensorPipeline` of effects — PSF blur, signal-dependent shot noise + read noise, bit-depth quantization for RGB; range-dependent noise, flying-pixel/multipath artifacts, and range quantization for depth.
- **Ground truth**: instance ID maps and per-object scene metadata stay available for future dataset export, independent of what the classical detector can actually see.

### Perception products

Detections are structured value objects (`Detection2D`, with an extensible `metadata` dict for diagnostics like solidity/aspect ratio) so that:

- evaluation metrics stay consistent,
- tasking logic subscribes to real perception output every step, not synthetic events,
- datasets can be exported without ad-hoc dicts.

### Multi-satellite coordination

Ground targets are placed by geographic coordinates (`GroundTargetSpec` + `geodetic_to_ecef_spherical`) on the constellation's shared ground track, not trivially under a satellite at `t=0`. Because every `SatelliteAgent` in the environment is given the *same* `PriorityQueueScheduler` instance, `PriorityQueueScheduler.find_active_near()` gives simple, spatial deconfliction: before submitting a task, an agent checks whether a non-terminal task already exists near its detection's approximate location (the satellite's own nadir point — not true geolocation), and skips submission if so. This is what keeps `reactive_task_events` far below `total_detections` in the example at the top of this README.

---

## Configuration

Configs under `configs/` are the primary way to define experiments:

| Path | Role |
|------|------|
| `configs/default.yaml` | Baseline scenario knobs |
| `configs/constellation/` | Walker / custom constellation layouts |
| `configs/sensors/` | Camera, ToF, noise parameters |
| `configs/vision/` | Model checkpoints, thresholds, class maps |
| `configs/simulation/` | Time step, duration, logging, seeds |

Pydantic settings models (in `infrastructure/io` and domain config types) validate configs before a run starts.

Note: `constellation`, `sensors`, and `vision` are currently just path strings carried through `ScenarioConfig` — the demo constellation, ground targets (`TwoBodyEnvironment.default_demo_targets`), camera model, and sensor-effect pipeline are still built in code, not parsed from these YAML fragments yet. `configs/default.yaml` documents the current demo scenario's geometry (target placement, expected detection timing) in comments until that wiring lands.

---

## Development

### Tooling

| Tool | Purpose |
|------|---------|
| **Ruff** | Lint + format (`ruff check`, `ruff format`) |
| **mypy** | Strict static typing |
| **pytest** | Unit, integration, property tests |
| **hypothesis** | Property-based tests for orbital / geometry edge cases |

```bash
ruff check src tests
ruff format src tests
mypy src
pytest
pytest --cov=satsim --cov-report=term-missing
```

### Test layout

```text
tests/
├── unit/           # pure core + small adapters
├── integration/    # multi-module scenarios
├── property/       # hypothesis / invariants
└── conftest.py     # shared fixtures
```

Markers: `unit`, `integration`, `slow`, `gpu`, `orbital`, `vision`, `render`.

### Contributing guidelines (summary)

1. Keep `core` free of heavy optional dependencies.
2. Prefer protocols / ABCs for propagators, renderers, detectors, and agents.
3. Every public function gets type hints and a Google-style docstring.
4. New physics or sensor effects ship with at least one unit test and a config example.
5. Do not commit large weights or raw datasets — document download paths instead.

---

## Optional extras

Declared in `pyproject.toml`:

| Extra | Contents |
|-------|----------|
| `astro` | Astropy, poliastro |
| `render` | Pillow, OpenCV, trimesh |
| `vision` | PyTorch, torchvision, Ultralytics |
| `rl` | Gymnasium, PettingZoo |
| `dev` | pytest, ruff, mypy, pre-commit, … |
| `all` | Union of the above |

Install only what you need for a given research track.

---

## Project status

**Alpha, but the core loop is real.** The repository provides:

- a real two-body propagator and a working `reset()`/`step()` simulation loop,
- ray-traced synthetic imagery (shaded Earth, occlusion, atmosphere) with a real RGB + depth sensor-effect pipeline,
- a classical detector with geometric shape filtering and heuristic confidence — not just a stub returning empty results,
- geographically placed, multi-target scenarios with constellation-wide task deconfliction,
- production project layout, typed module boundaries, and a test suite that exercises all of the above (unit, integration, property).

Still ahead: J2/drag and richer orbital perturbations, YAML-driven constellation/sensor/target configuration (currently built in code), learned detectors under the `vision` extra, task completion lifecycle, and RL environment wrappers. These are expected to land incrementally on the existing architecture without structural rewrites — the `core` / `application` / `infrastructure` separation is what lets a new renderer, sensor model, detector, or coordination rule get added without touching the layers above or below it, which is exactly how the current feature set was built.

---

## Citation / acknowledgement

If you use SatSim in academic work, please cite the repository and note the version. A formal CITATION.cff will be added when the first citable release is tagged.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Contact

Issues and design discussions are welcome via the project issue tracker. For architecture proposals that affect layer boundaries (e.g. new sensor modalities, multi-agent message buses), open an issue before a large PR so interfaces stay coherent.
