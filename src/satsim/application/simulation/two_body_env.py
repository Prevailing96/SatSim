"""Minimal two-body constellation environment.

Advances satellites with :class:`~satsim.core.orbital.propagator.TwoBodyPropagator`.
Every reset/step snapshot also renders each satellite's nadir view, runs it
through a sensor-effect pipeline and perception pipeline, and lets a
:class:`~satsim.application.agents.satellite_agent.SatelliteAgent` react to
whatever it detects — closing the perception -> autonomy loop the rest of
the scaffold only declared as protocols.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from satsim.application.agents.base import AgentObservation
from satsim.application.agents.satellite_agent import (
    DEFAULT_DECONFLICTION_RADIUS_M,
    SatelliteAgent,
)
from satsim.application.simulation.clock import SimulationClock
from satsim.application.simulation.environment import StepResult
from satsim.application.tasking.scheduler import PriorityQueueScheduler
from satsim.core.constants import DEFAULT_LEO_ALTITUDE_M, MU_EARTH
from satsim.core.orbital.frames import geodetic_to_ecef_spherical
from satsim.core.orbital.propagator import PropagatorConfig, TwoBodyPropagator
from satsim.core.orbital.state import OrbitalState
from satsim.core.orbital.twobody import circular_leo_state
from satsim.core.sensors.models import CameraModel
from satsim.core.sensors.observations import DepthFrame, ImageFrame, ObservationBundle
from satsim.core.types.identifiers import AgentId, SatelliteId, SensorId
from satsim.core.types.time import SimTime, TimeSpan
from satsim.core.types.vectors import CartesianState, Vec3
from satsim.infrastructure.rendering.renderer import (
    PlaceholderRenderer,
    RenderRequest,
    SceneRenderer,
)
from satsim.infrastructure.rendering.scene import SceneDescriptor, SceneObject
from satsim.infrastructure.sensors.effects import (
    BitDepthQuantizationEffect,
    DepthEdgeArtifactEffect,
    DepthQuantizationEffect,
    DepthRangeNoiseEffect,
    GaussianBlurEffect,
    SensorPipeline,
    ShotReadNoiseEffect,
)
from satsim.infrastructure.vision.pipeline import PerceptionPipeline

#: Detection classes used by the demo ground targets (see
#: :func:`default_demo_targets`); kept in sync with ``reactive_classes`` in
#: ``configs/default.yaml``.
DEMO_TARGET_CLASSES = ("vessel", "aircraft")


@dataclass(frozen=True, slots=True)
class SatelliteSpec:
    """Initial-condition specification for one vehicle.

    Attributes:
        satellite_id: Unique id string (stored as :class:`SatelliteId`).
        altitude_m: Circular-orbit altitude above equatorial radius [m].
        inclination_rad: Orbit inclination [rad].
        raan_rad: RAAN [rad].
        true_anomaly_rad: Initial true anomaly [rad].
    """

    satellite_id: str
    altitude_m: float = DEFAULT_LEO_ALTITUDE_M
    inclination_rad: float = 0.0
    raan_rad: float = 0.0
    true_anomaly_rad: float = 0.0


def default_demo_constellation(n: int = 3) -> tuple[SatelliteSpec, ...]:
    """Build a small co-planar LEO demo constellation.

    Satellites share inclination and altitude; true anomaly is evenly spaced.

    Args:
        n: Number of satellites (must be >= 1).

    Returns:
        Tuple of :class:`SatelliteSpec`.

    Raises:
        ValueError: If ``n < 1``.
    """
    if n < 1:
        msg = "constellation size must be >= 1"
        raise ValueError(msg)
    specs: list[SatelliteSpec] = []
    for i in range(n):
        specs.append(
            SatelliteSpec(
                satellite_id=f"sat-{i + 1:03d}",
                altitude_m=DEFAULT_LEO_ALTITUDE_M,
                inclination_rad=math.radians(53.0),
                raan_rad=0.0,
                true_anomaly_rad=2.0 * math.pi * i / n,
            )
        )
    return tuple(specs)


def _spec_to_state(spec: SatelliteSpec, time: SimTime, mu: float) -> OrbitalState:
    """Convert a satellite spec to an :class:`OrbitalState` at ``time``."""
    r, v = circular_leo_state(
        spec.altitude_m,
        mu=mu,
        inclination_rad=spec.inclination_rad,
        raan_rad=spec.raan_rad,
        true_anomaly_rad=spec.true_anomaly_rad,
    )
    return OrbitalState(
        satellite_id=SatelliteId(spec.satellite_id),
        time=time,
        cartesian=CartesianState(
            position_m=Vec3.from_array(r),
            velocity_m_s=Vec3.from_array(v),
            frame="ECI_J2000",
        ),
    )


@dataclass(frozen=True, slots=True)
class GroundTargetSpec:
    """A ground target specified by geographic coordinates.

    Attributes:
        target_id: Stable id for the underlying :class:`SceneObject`.
        latitude_rad: Planetocentric latitude [rad], positive north.
        longitude_rad: Longitude [rad], positive east.
        altitude_m: Height above the spherical Earth surface [m].
        class_name: Detection class label (e.g. ``"vessel"``, ``"aircraft"``).
    """

    target_id: str
    latitude_rad: float
    longitude_rad: float
    altitude_m: float = 0.0
    class_name: str = "vessel"


def default_demo_targets() -> tuple[GroundTargetSpec, ...]:
    """Three demo ground targets spread across the shared 53-degree ground track.

    :func:`default_demo_constellation` puts every satellite on the same
    orbital plane (RAAN=0, inclination=53 deg), just phased 120 degrees
    apart in true anomaly. Each target here sits ~25 degrees of true
    anomaly ahead of one satellite's starting position, on that same shared
    plane, expressed as geographic coordinates via
    :func:`~satsim.core.orbital.frames.geodetic_to_ecef_spherical`.

    That means no target is under a satellite at ``t=0`` — each is instead
    discovered roughly a quarter of the way into a ~5740s orbit (about 6-7
    minutes at 550 km altitude) as its assigned satellite's ground track
    sweeps over it, which is well inside the default scenario duration.

    Returns:
        Three :class:`GroundTargetSpec`, one loosely associated with each
        satellite in :func:`default_demo_constellation`.
    """
    inclination_rad = math.radians(53.0)
    lead_angle_rad = math.radians(25.0)
    satellite_start_angles_rad = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    classes = ("vessel", "vessel", "aircraft")
    names = ("target-alpha", "target-bravo", "target-charlie")

    targets: list[GroundTargetSpec] = []
    for start_rad, class_name, target_id in zip(
        satellite_start_angles_rad, classes, names, strict=True
    ):
        theta = start_rad + lead_angle_rad
        latitude_rad = math.asin(math.sin(inclination_rad) * math.sin(theta))
        longitude_rad = math.atan2(
            math.sin(theta) * math.cos(inclination_rad), math.cos(theta)
        )
        targets.append(
            GroundTargetSpec(
                target_id=target_id,
                latitude_rad=latitude_rad,
                longitude_rad=longitude_rad,
                class_name=class_name,
            )
        )
    return tuple(targets)


def _build_scene(targets: tuple[GroundTargetSpec, ...]) -> SceneDescriptor:
    """Build the demo scene: Earth plus the given ground targets.

    Args:
        targets: Ground targets to place in the scene.

    Returns:
        Scene with one :class:`SceneObject` per target.
    """
    objects = tuple(
        SceneObject(
            object_id=t.target_id,
            name=t.class_name,
            position_m=geodetic_to_ecef_spherical(
                t.latitude_rad, t.longitude_rad, altitude_m=t.altitude_m
            ),
            metadata={"class_name": t.class_name},
        )
        for t in targets
    )
    return SceneDescriptor(frame="ECI_J2000", objects=objects, background="earth_texture")


def _default_sensor_pipeline() -> SensorPipeline:
    """Build the demo sensor-effect chain.

    Optics -> RGB sensor -> ADC, then an analogous chain for the
    geometrically-derived depth channel.

    Order matters: blur models the optical stage (before any electronics),
    shot/read noise and bit-depth quantization model the RGB sensor and its
    ADC in that order, and the depth chain runs range-dependent noise and
    edge artifacts before a final quantization step, mirroring a real ToF
    front-end -> range-processing -> ADC pipeline.

    Returns:
        Configured :class:`SensorPipeline`.
    """
    return SensorPipeline(
        [
            GaussianBlurEffect(sigma_px=0.5),
            ShotReadNoiseEffect(shot_noise_coeff=0.015, read_noise_sigma=0.004),
            BitDepthQuantizationEffect(bits=8),
            DepthRangeNoiseEffect(),
            DepthEdgeArtifactEffect(),
            DepthQuantizationEffect(resolution_m=1.0),
        ]
    )


@dataclass
class TwoBodyEnvironment:
    """Fixed-step two-body constellation simulation with a sense-decide loop.

    Attributes:
        duration_s: Episode length [s]; ``truncated`` becomes True when reached.
        dt_s: Step size [s].
        satellite_specs: Initial condition templates.
        ground_targets: Ground targets placed via geographic coordinates.
        propagator: Orbital propagator (defaults to Earth two-body).
        seed: RNG seed for sensor noise and future stochastic agents.
        renderer: Scene renderer used for every satellite's nadir camera.
        sensor_pipeline: Effects applied to the ideal render before perception.
        perception_pipeline: Detector (+ optional segmenter) run on each frame.
        camera_width_px: Demo camera resolution (kept small for step speed).
        camera_height_px: Demo camera resolution (kept small for step speed).
        camera_fov_x_deg: Demo camera horizontal field of view [deg].
        camera_fov_y_deg: Demo camera vertical field of view [deg].
        reactive_class_names: Detection classes that trigger follow-up tasking.
        min_detection_confidence: Confidence floor for reactive tasking.
        deconfliction_radius_m: Distance below which two detections are
            treated as the same target, so only one gets tasked. All
            satellites share one scheduler instance (see
            :attr:`~satsim.application.agents.satellite_agent.SatelliteAgent.scheduler`),
            which is what makes this deconfliction constellation-wide rather
            than per-satellite.
    """

    duration_s: float = 600.0
    dt_s: float = 1.0
    satellite_specs: tuple[SatelliteSpec, ...] = field(
        default_factory=lambda: default_demo_constellation(3)
    )
    ground_targets: tuple[GroundTargetSpec, ...] = field(default_factory=default_demo_targets)
    propagator: TwoBodyPropagator = field(
        default_factory=lambda: TwoBodyPropagator(PropagatorConfig(mu=MU_EARTH))
    )
    seed: int = 0
    renderer: SceneRenderer = field(default_factory=PlaceholderRenderer)
    sensor_pipeline: SensorPipeline = field(default_factory=_default_sensor_pipeline)
    perception_pipeline: PerceptionPipeline = field(default_factory=PerceptionPipeline.rule_based)
    camera_width_px: int = 64
    camera_height_px: int = 48
    camera_fov_x_deg: float = 20.0
    camera_fov_y_deg: float = 15.0
    reactive_class_names: frozenset[str] = field(
        default_factory=lambda: frozenset(DEMO_TARGET_CLASSES)
    )
    min_detection_confidence: float = 0.5
    deconfliction_radius_m: float = DEFAULT_DECONFLICTION_RADIUS_M

    def __post_init__(self) -> None:
        """Validate timing parameters and initialize runtime state."""
        if self.duration_s < 0.0:
            msg = "duration_s must be non-negative"
            raise ValueError(msg)
        if self.dt_s <= 0.0:
            msg = "dt_s must be positive"
            raise ValueError(msg)
        if not self.satellite_specs:
            msg = "at least one satellite_spec is required"
            raise ValueError(msg)
        self._clock = SimulationClock.create(start_s=0.0, dt_s=self.dt_s)
        self._states: dict[str, OrbitalState] = {}
        self._initial_states: dict[str, OrbitalState] = {}
        self._closed = False
        self._scene = _build_scene(self.ground_targets)
        self._cameras: dict[str, CameraModel] = {
            spec.satellite_id: CameraModel(
                sensor_id=SensorId(f"{spec.satellite_id}-eo"),
                satellite_id=SatelliteId(spec.satellite_id),
                width_px=self.camera_width_px,
                height_px=self.camera_height_px,
                focal_length_mm=35.0,
                pixel_size_um=5.0,
                fov_x_rad=math.radians(self.camera_fov_x_deg),
                fov_y_rad=math.radians(self.camera_fov_y_deg),
            )
            for spec in self.satellite_specs
        }
        # One scheduler shared by every agent: this is the constellation's
        # task board. Constructed once and never replaced (only cleared, in
        # reset()) so agents' captured references stay valid across resets.
        self._shared_scheduler = PriorityQueueScheduler()
        self._agents: dict[str, SatelliteAgent] = {
            spec.satellite_id: SatelliteAgent(
                AgentId(spec.satellite_id),
                SatelliteId(spec.satellite_id),
                detection_class_names=self.reactive_class_names,
                min_confidence=self.min_detection_confidence,
                scheduler=self._shared_scheduler,
                deconfliction_radius_m=self.deconfliction_radius_m,
            )
            for spec in self.satellite_specs
        }
        self._rng: np.random.Generator = np.random.default_rng(self.seed)

    @property
    def time(self) -> SimTime:
        """Current simulation time."""
        return self._clock.current

    @property
    def step_index(self) -> int:
        """Number of completed dynamics steps since last reset."""
        return self._clock.step_index

    def reset(self, *, seed: int | None = None) -> StepResult:
        """Reset satellites to initial conditions and clock to epoch.

        Args:
            seed: Optional seed override for this episode.

        Returns:
            Snapshot at ``t = 0`` (before any dynamics step), including a
            freshly rendered/perceived observation for every satellite.
        """
        if seed is not None:
            self.seed = seed
        self._clock.reset()
        self._closed = False
        self._rng = np.random.default_rng(self.seed)
        mu = self.propagator.config.mu
        t0 = self._clock.current
        self._initial_states = {
            spec.satellite_id: _spec_to_state(spec, t0, mu) for spec in self.satellite_specs
        }
        self._states = dict(self._initial_states)
        self._shared_scheduler.clear()
        for agent in self._agents.values():
            agent.reset(seed=self.seed)
        return self._snapshot(
            done=False,
            truncated=False,
            infos={"event": "reset", "seed": self.seed, "n_satellites": len(self._states)},
        )

    def step(self, actions: dict[str, Any] | None = None) -> StepResult:
        """Propagate all satellites by one time step and run sense-decide.

        Args:
            actions: Reserved for future tasking / control (ignored for now).

        Returns:
            Post-step snapshot. ``truncated`` is True when ``time >= duration_s``.

        Raises:
            RuntimeError: If called after the episode has already ended without
                :meth:`reset`.
        """
        del actions  # reserved for closed-loop control
        if self._closed:
            msg = "Episode finished; call reset() before stepping again"
            raise RuntimeError(msg)
        if not self._states:
            # Auto-reset if step is called without explicit reset
            self.reset(seed=self.seed)

        dt = TimeSpan(self.dt_s)
        new_states: dict[str, OrbitalState] = {}
        for sat_id, state in self._states.items():
            new_states[sat_id] = self.propagator.propagate(state, dt)
        self._states = new_states
        self._clock.advance()

        truncated = self._clock.current.seconds >= self.duration_s - 1e-12
        if truncated:
            self._closed = True

        return self._snapshot(
            done=False,
            truncated=truncated,
            infos={
                "event": "step",
                "step_index": self._clock.step_index,
                "n_satellites": len(self._states),
            },
        )

    def _observe(self, sat_id: str, state: OrbitalState) -> ObservationBundle:
        """Render, sense, and perceive one satellite's current view.

        Args:
            sat_id: Satellite identifier (key into ``self._states``).
            state: Current orbital state of that satellite.

        Returns:
            Perception-ready :class:`ObservationBundle` with detections and,
            when available, an aligned depth frame.
        """
        camera = self._cameras[sat_id]
        request = RenderRequest(
            time=state.time,
            scene=self._scene,
            camera=camera,
            platform_position_m=state.cartesian.position_m,
            platform_attitude=state.attitude,
        )
        ideal = self.renderer.render(request)
        measured = self.sensor_pipeline.apply(ideal, self._rng)

        rgb = measured.rgb
        if rgb is None:
            rgb = np.zeros((camera.height_px, camera.width_px, 3), dtype=np.float32)
        image = ImageFrame(
            sensor_id=camera.sensor_id,
            satellite_id=SatelliteId(sat_id),
            time=state.time,
            data=rgb,
        )
        bundle = self.perception_pipeline.process(image)

        depth_frame: DepthFrame | None = None
        if measured.depth_m is not None:
            depth_frame = DepthFrame(
                sensor_id=camera.sensor_id,
                satellite_id=SatelliteId(sat_id),
                time=state.time,
                depth_m=measured.depth_m,
                metadata=dict(measured.metadata),
            )
        return replace(bundle, depth=depth_frame)

    def _snapshot(
        self,
        *,
        done: bool,
        truncated: bool,
        infos: dict[str, Any],
    ) -> StepResult:
        """Build a :class:`StepResult`, running sense-perceive-decide first."""
        observations = tuple(
            self._observe(sat_id, state) for sat_id, state in self._states.items()
        )

        agent_actions: dict[str, dict[str, Any]] = {}
        n_detections = 0
        for bundle in observations:
            sat_key = str(bundle.satellite_id)
            n_detections += len(bundle.detections)
            agent = self._agents[sat_key]
            observation = AgentObservation(
                agent_id=agent.agent_id,
                time=self._clock.current,
                own_state=self._states.get(sat_key),
                bundles=(bundle,),
            )
            action = agent.act(observation)
            agent_actions[sat_key] = {"kind": action.kind, "payload": action.payload}

        merged_infos = dict(infos)
        merged_infos["n_detections"] = n_detections
        merged_infos["agent_actions"] = agent_actions
        # Shared-awareness snapshot: how many tasks the whole constellation
        # currently considers active, regardless of which satellite raised
        # them — the number deconfliction is keeping from double-counting.
        merged_infos["constellation_active_tasks"] = len(self._shared_scheduler.pending())

        return StepResult(
            time=self._clock.current,
            states=dict(self._states),
            observations=observations,
            rewards={},
            infos=merged_infos,
            done=done,
            truncated=truncated,
        )


__all__ = [
    "GroundTargetSpec",
    "SatelliteSpec",
    "TwoBodyEnvironment",
    "default_demo_constellation",
    "default_demo_targets",
]
