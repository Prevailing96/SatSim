# SatSim

**Scalable satellite constellation simulation with synthetic imaging, sensor modeling, and computer-vision-driven autonomy.**

SatSim is a professional-grade research and engineering platform for studying how fleets of satellites sense, decide, and act. It couples high-fidelity orbital dynamics with photorealistic (and sensor-realistic) synthetic imagery — including RGB, multispectral, and time-of-flight / depth channels — then closes the loop through a computer-vision perception stack that can influence tasking and onboard autonomy.

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

| Area | Near-term scaffold | Direction |
|------|--------------------|-----------|
| **Orbital dynamics** | State vectors, Keplerian elements, two-body propagator interface | High-order integrators, J2/drag, multi-body, maneuver burns |
| **Constellations** | Walker / custom slot configs | Coverage metrics, inter-sat links, swarm geometries |
| **Synthetic data** | Image + depth frame contracts, scene descriptors | Path-traced / raster EO, BRDF, atmosphere, cloud layers |
| **Sensors** | RGB + ToF/depth effect interfaces | PSF, MTF, shot/read noise, quantization, ToF multipath & flying pixels |
| **Perception** | Detection & segmentation result schemas | Training data export, online inference, multi-object tracking |
| **Autonomy** | Tasking request / agent policy protocols | Closed-loop retasking, multi-agent coordination |
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

### Example config-driven entry (as the API matures)

```bash
python scripts/run_scenario.py --config configs/default.yaml
```

---

## Core concepts

### Simulation step (closed loop)

A single logical step of the environment is intended to look like:

1. **Advance** simulation clock and orbital state for all vehicles.
2. **Sense** — for each tasked sensor, render a synthetic observation and apply sensor effects (noise, blur, ToF artifacts).
3. **Perceive** — run detection / segmentation (and later tracking) on those observations.
4. **Decide** — agents / tasking policies update schedules from perception products and mission goals.
5. **Act** — apply attitude slews, mode changes, or (later) Δv maneuvers; emit logs and metrics.

Perception is not an offline post-process; it is an input to autonomy.

### Synthetic imagery & ToF

SatSim treats **depth / time-of-flight** as a first-class modality alongside electro-optical imagery. Starter modules define:

- frame metadata (pose, FOV, wavelength / timing),
- ideal rendered channels,
- sensor-effect pipelines that corrupt ideal channels into realistic measurements,
- ground-truth labels aligned for CV training and evaluation.

### Perception products

Detections and masks are structured value objects so that:

- evaluation metrics stay consistent,
- tasking logic can subscribe to perception events,
- datasets can be exported without ad-hoc dicts.

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

**Alpha scaffold.** The repository currently provides:

- production project layout and packaging,
- typed module boundaries and interfaces,
- baseline configs and smoke CLI,
- test skeleton ready for growth.

High-fidelity propagators, full renderers, trained models, and RL environments will land incrementally on this skeleton without structural rewrites.

---

## Citation / acknowledgement

If you use SatSim in academic work, please cite the repository and note the version. A formal CITATION.cff will be added when the first citable release is tagged.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Contact

Issues and design discussions are welcome via the project issue tracker. For architecture proposals that affect layer boundaries (e.g. new sensor modalities, multi-agent message buses), open an issue before a large PR so interfaces stay coherent.
