"""Scene renderer protocol and a minimal geometric implementation.

Ideal (noise-free) RGB and depth are produced here; sensor-effect pipelines
in ``infrastructure.sensors`` corrupt them into realistic measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from satsim.core.constants import R_EARTH_EQUATORIAL
from satsim.core.sensors.models import CameraModel, ToFSensorModel
from satsim.core.types.time import SimTime
from satsim.core.types.vectors import AttitudeQuaternion, Vec3
from satsim.infrastructure.rendering.scene import SceneDescriptor, SceneObject


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Parameters for a single render call.

    Attributes:
        time: Simulation time of the capture.
        scene: World content to render.
        camera: Optional EO camera model.
        tof: Optional ToF / depth sensor model.
        platform_position_m: Observing satellite position in the scene's
            reference frame [m]. Sensor mounting offsets
            (``camera.pose`` / ``tof.pose``) are not yet composed onto this;
            that is reserved for a future increment.
        platform_attitude: Observing satellite attitude, rotating a
            ``+Z`` boresight / ``+Y`` up sensor-frame convention into the
            scene frame. ``None`` means nadir-pointing (boresight toward the
            scene origin, i.e. Earth center) with an arbitrary roll.
        extras: Backend-specific options (samples, resolution overrides).
    """

    time: SimTime
    scene: SceneDescriptor
    camera: CameraModel | None = None
    tof: ToFSensorModel | None = None
    platform_position_m: Vec3 = field(
        default_factory=lambda: Vec3(R_EARTH_EQUATORIAL + 550_000.0, 0.0, 0.0)
    )
    platform_attitude: AttitudeQuaternion | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Ideal synthetic products before sensor corruption.

    Attributes:
        time: Capture time.
        rgb: Optional ``(H, W, 3)`` float32 image in ``[0, 1]``.
        depth_m: Optional ``(H, W)`` depth in meters.
        instance_ids: Optional instance id map for GT segmentation.
        metadata: Render diagnostics (backend name, timings).
    """

    time: SimTime
    rgb: NDArray[np.float32] | None = None
    depth_m: NDArray[np.float32] | None = None
    instance_ids: NDArray[np.int32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SceneRenderer(Protocol):
    """Renders ideal sensor channels from a scene and camera/ToF models."""

    def render(self, request: RenderRequest) -> RenderResult:
        """Produce ideal RGB / depth for the request.

        Args:
            request: Render parameters.

        Returns:
            Ideal (pre-sensor-effect) result.
        """
        ...


def _quat_to_dcm(q: AttitudeQuaternion) -> NDArray[np.float64]:
    """Convert a unit quaternion to a direction cosine matrix.

    Args:
        q: Attitude quaternion (scalar-first).

    Returns:
        ``3x3`` rotation matrix mapping sensor/body-frame vectors into the
        quaternion's ``frame_to``.

    Raises:
        ValueError: If the quaternion is (near) zero norm.
    """
    w, x, y, z = q.w, q.x, q.y, q.z
    n = w * w + x * x + y * y + z * z
    if n < 1e-15:
        msg = "Cannot build rotation from near-zero quaternion"
        raise ValueError(msg)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _camera_basis(
    platform_position_m: NDArray[np.float64],
    attitude: AttitudeQuaternion | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build an orthonormal ``(right, up, forward)`` camera basis in world space.

    Args:
        platform_position_m: Satellite position ``(3,)`` [m].
        attitude: Optional attitude; ``None`` implies nadir-pointing.

    Returns:
        Unit vectors ``(right, up, forward)``.
    """
    if attitude is not None:
        dcm = _quat_to_dcm(attitude)
        forward = dcm @ np.array([0.0, 0.0, 1.0])
        up_ref = dcm @ np.array([0.0, 1.0, 0.0])
    else:
        r_norm = float(np.linalg.norm(platform_position_m))
        forward = (
            np.array([0.0, 0.0, -1.0]) if r_norm < 1.0 else -platform_position_m / r_norm
        )
        north = np.array([0.0, 0.0, 1.0])
        up_ref = north if abs(float(np.dot(forward, north))) < 0.999 else np.array([1.0, 0.0, 0.0])

    right = np.cross(forward, up_ref)
    right_norm = float(np.linalg.norm(right))
    right = right / right_norm if right_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    up = up / float(np.linalg.norm(up))
    forward = forward / float(np.linalg.norm(forward))
    return right, up, forward


def _camera_fov(camera: CameraModel) -> tuple[float, float]:
    """Return ``(fov_x, fov_y)`` in radians, deriving from optics if unset.

    Args:
        camera: Camera intrinsics.

    Returns:
        Horizontal and vertical full field of view [rad].
    """
    if camera.fov_x_rad is not None and camera.fov_y_rad is not None:
        return camera.fov_x_rad, camera.fov_y_rad
    sensor_w_mm = camera.width_px * camera.pixel_size_um * 1e-3
    sensor_h_mm = camera.height_px * camera.pixel_size_um * 1e-3
    fov_x = 2.0 * math.atan2(sensor_w_mm / 2.0, camera.focal_length_mm)
    fov_y = 2.0 * math.atan2(sensor_h_mm / 2.0, camera.focal_length_mm)
    return fov_x, fov_y


def _pixel_directions(
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    right: NDArray[np.float64],
    up: NDArray[np.float64],
    forward: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute unit world-space ray directions for every pixel.

    Args:
        width: Image width [px].
        height: Image height [px].
        fov_x: Horizontal full FOV [rad].
        fov_y: Vertical full FOV [rad].
        right: Camera "right" unit vector.
        up: Camera "up" unit vector.
        forward: Camera boresight unit vector.

    Returns:
        ``(H, W, 3)`` array of unit ray directions.
    """
    tan_x = math.tan(fov_x / 2.0)
    tan_y = math.tan(fov_y / 2.0)
    cols = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    rows = (np.arange(height) + 0.5) / height * 2.0 - 1.0
    x_ndc, y_ndc = np.meshgrid(cols, -rows)
    x_comp = (x_ndc * tan_x)[..., None]
    y_comp = (y_ndc * tan_y)[..., None]
    dirs = x_comp * right[None, None, :] + y_comp * up[None, None, :] + forward[None, None, :]
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    unit_dirs: NDArray[np.float64] = dirs / norms
    return unit_dirs


def _ray_sphere_hit(
    origin: NDArray[np.float64],
    directions: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Intersect per-pixel rays with a sphere centered at the scene origin.

    Args:
        origin: Ray origin ``(3,)`` shared by all pixels [m].
        directions: Unit ray directions ``(H, W, 3)``.
        radius: Sphere radius [m].

    Returns:
        ``(t, hit)`` where ``t`` is the near-intersection distance for every
        pixel (always finite, meaningless where ``hit`` is False) and
        ``hit`` marks pixels with a valid forward intersection. Keeping ``t``
        finite everywhere lets callers do further arithmetic before masking,
        without ever comparing NaN (numpy warns-as-error on that).
    """
    o = origin.astype(np.float64)
    d = directions.astype(np.float64)
    b = np.sum(d * o[None, None, :], axis=-1)
    c = float(np.dot(o, o) - radius * radius)
    disc = b * b - c
    sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
    t = -b - sqrt_disc
    hit = (disc >= 0.0) & (t > 0.0)
    return t, hit


def _project_object(
    origin: NDArray[np.float64],
    point: NDArray[np.float64],
    right: NDArray[np.float64],
    up: NDArray[np.float64],
    forward: NDArray[np.float64],
    fov_x: float,
    fov_y: float,
    width: int,
    height: int,
    earth_radius: float,
) -> tuple[int, int, float] | None:
    """Project a world point into pixel space if visible in the frustum.

    Args:
        origin: Camera position ``(3,)`` [m].
        point: World point to project ``(3,)`` [m].
        right: Camera right axis.
        up: Camera up axis.
        forward: Camera boresight axis.
        fov_x: Horizontal full FOV [rad].
        fov_y: Vertical full FOV [rad].
        width: Image width [px].
        height: Image height [px].
        earth_radius: Occluding sphere radius [m].

    Returns:
        ``(row, col, distance_m)`` if the point is in front of the camera,
        inside the FOV, and not occluded by the Earth limb; else ``None``.
    """
    vec = point - origin
    dist = float(np.linalg.norm(vec))
    if dist < 1.0:
        return None
    direction = vec / dist
    z_cam = float(np.dot(direction, forward))
    if z_cam <= 1e-6:
        return None
    x_cam = float(np.dot(direction, right))
    y_cam = float(np.dot(direction, up))
    x_ndc = (x_cam / z_cam) / math.tan(fov_x / 2.0)
    y_ndc = (y_cam / z_cam) / math.tan(fov_y / 2.0)
    if abs(x_ndc) > 1.0 or abs(y_ndc) > 1.0:
        return None

    b = float(np.dot(origin, direction))
    c = float(np.dot(origin, origin) - earth_radius * earth_radius)
    disc = b * b - c
    if disc >= 0.0:
        t_hit = -b - math.sqrt(disc)
        if 0.0 < t_hit < dist - 1.0:
            return None  # blocked by the Earth limb

    col = int((x_ndc * 0.5 + 0.5) * width)
    row = int((1.0 - (y_ndc * 0.5 + 0.5)) * height)
    col = min(max(col, 0), width - 1)
    row = min(max(row, 0), height - 1)
    return row, col, dist


def _closest_approach_to_center(
    origin: NDArray[np.float64],
    directions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Perpendicular distance from the scene origin to each ray line.

    Args:
        origin: Ray origin ``(3,)`` shared by all pixels [m].
        directions: Unit ray directions ``(H, W, 3)``.

    Returns:
        ``(H, W)`` distances [m]. Always finite: for a unit direction ``d``
        from point ``o``, ``|o - (o.d)d| = sqrt(|o|^2 - (o.d)^2)``, and
        Cauchy-Schwarz guarantees a non-negative radicand, but the value is
        clipped defensively before the square root anyway.
    """
    b = np.sum(directions * origin[None, None, :], axis=-1)
    radicand = float(np.dot(origin, origin)) - b * b
    dist: NDArray[np.float64] = np.sqrt(np.clip(radicand, 0.0, None))
    return dist


def _paint_targets(
    *,
    origin: NDArray[np.float64],
    objects: tuple[SceneObject, ...],
    right: NDArray[np.float64],
    up: NDArray[np.float64],
    forward: NDArray[np.float64],
    fov_x: float,
    fov_y: float,
    width: int,
    height: int,
    earth_radius: float,
    marker_rgb: tuple[float, float, float] | None,
    marker_outline_rgb: tuple[float, float, float] | None,
    marker_radius_px: int,
    rgb: NDArray[np.float32] | None,
    instances: NDArray[np.int32] | None,
    depth: NDArray[np.float32] | None,
) -> None:
    """Paint a small cross-shaped marker for every visible scene object.

    The marker is a plus-sign of ``marker_rgb`` over an outline-colored
    square (rather than a flat color-fill blob), so it reads as a deliberate
    icon/beacon instead of a rendering artifact while remaining a single
    dominant color for simple rule-based detection.

    Args:
        origin: Camera position ``(3,)`` [m].
        objects: Scene objects to attempt to project.
        right: Camera right axis.
        up: Camera up axis.
        forward: Camera boresight axis.
        fov_x: Horizontal full FOV [rad].
        fov_y: Vertical full FOV [rad].
        width: Image width [px].
        height: Image height [px].
        earth_radius: Occluding sphere radius [m].
        marker_rgb: Cross color, or ``None`` to skip RGB painting.
        marker_outline_rgb: Background/outline color behind the cross.
        marker_radius_px: Half-width of the square marker patch.
        rgb: RGB buffer to mutate (or ``None``).
        instances: Instance id buffer to mutate (or ``None``).
        depth: Depth buffer to mutate (or ``None``).
    """
    for idx, obj in enumerate(objects, start=1):
        proj = _project_object(
            origin,
            obj.position_m.as_array(),
            right,
            up,
            forward,
            fov_x,
            fov_y,
            width,
            height,
            earth_radius,
        )
        if proj is None:
            continue
        row, col, dist = proj
        r0, r1 = max(0, row - marker_radius_px), min(height, row + marker_radius_px + 1)
        c0, c1 = max(0, col - marker_radius_px), min(width, col + marker_radius_px + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        if rgb is not None and marker_rgb is not None:
            patch = np.empty((r1 - r0, c1 - c0, 3), dtype=np.float32)
            patch[...] = marker_outline_rgb if marker_outline_rgb is not None else marker_rgb
            patch[row - r0, :, :] = marker_rgb
            patch[:, col - c0, :] = marker_rgb
            rgb[r0:r1, c0:c1] = patch
        if instances is not None:
            instances[r0:r1, c0:c1] = idx
        if depth is not None:
            depth[r0:r1, c0:c1] = np.float32(dist)


class PlaceholderRenderer:
    """Minimal geometric renderer: Earth as a ray-traced, shaded sphere.

    Not a full 3D engine — no textures or multi-bounce global illumination —
    but every pixel is derived from real ray/sphere intersection and camera
    projection math rather than a constant fill, so RGB, depth, and instance
    ids are geometrically consistent with the requesting camera's pose and
    field of view. Shading combines three cheap, physically-motivated terms:

    - **Lambertian sun illumination**: brightness scales with
      ``dot(surface_normal, sun_direction)``, so a fixed sun direction
      produces a day/night terminator across the disk instead of flat fill.
    - **Limb darkening**: a linear darkening law in the camera-view angle
      softens the disk toward its edge, the way an atmosphere thickens the
      optical path at grazing angles.
    - **Atmospheric limb glow**: rays that narrowly miss the Earth sphere
      (within a thin shell above its radius) are blended toward a pale blue
      glow instead of hard-cutting to black space, softening the disk edge
      in wide-FOV views.

    :class:`~satsim.infrastructure.rendering.scene.SceneObject` entries are
    projected as small cross-shaped markers (not flat blobs) when they fall
    inside the camera frustum and are not occluded by the Earth limb.

    A narrow-FOV nadir camera (typical EO imager) sees its whole frame filled
    with Earth; a wide-FOV camera sees Earth as a disk against black space —
    both fall out of the same ray-sphere test, driven by ``CameraModel`` FOV.

    Sensor mounting offsets (``CameraModel.pose`` / ``ToFSensorModel.pose``)
    are not yet composed into the platform pose; reserved for later once
    multi-sensor extrinsics matter.
    """

    def __init__(
        self,
        *,
        earth_rgb: tuple[float, float, float] = (0.15, 0.35, 0.55),
        space_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        marker_rgb: tuple[float, float, float] = (1.0, 0.1, 0.1),
        marker_outline_rgb: tuple[float, float, float] = (0.03, 0.03, 0.03),
        marker_radius_px: int = 2,
        tof_fov_deg: float = 45.0,
        sun_direction: tuple[float, float, float] = (0.6, -0.4, 0.5),
        limb_darkening_coeff: float = 0.6,
        earth_ambient: float = 0.05,
        atmosphere_thickness_frac: float = 0.02,
        atmosphere_rgb: tuple[float, float, float] = (0.55, 0.75, 1.0),
    ) -> None:
        """Initialize renderer appearance and geometry knobs.

        Args:
            earth_rgb: Base (unshaded) color for Earth-hit pixels.
            space_rgb: Fill color for rays that miss the Earth sphere.
            marker_rgb: Cross color painted for visible scene-object markers.
            marker_outline_rgb: Background color behind each marker's cross.
            marker_radius_px: Half-width of the square marker patch.
            tof_fov_deg: Assumed full field of view for ToF requests, which
                (unlike :class:`CameraModel`) carry no explicit FOV field.
            sun_direction: Fixed unit-ish sun direction in the scene frame
                (normalized internally); a real ephemeris-driven sun is
                future work.
            limb_darkening_coeff: Linear limb-darkening coefficient in
                ``[0, 1]``; ``0`` disables the effect.
            earth_ambient: Minimum illumination fraction on the night side
                (keeps it dark but not pure black, as a stand-in for
                Earthshine / city lights).
            atmosphere_thickness_frac: Thickness of the glow shell above
                Earth's radius, as a fraction of that radius.
            atmosphere_rgb: Color of the atmospheric limb glow.
        """
        self._earth_rgb = earth_rgb
        self._space_rgb = space_rgb
        self._marker_rgb = marker_rgb
        self._marker_outline_rgb = marker_outline_rgb
        self._marker_radius_px = marker_radius_px
        self._tof_fov_rad = math.radians(tof_fov_deg)
        sun = np.array(sun_direction, dtype=np.float64)
        sun_norm = float(np.linalg.norm(sun))
        self._sun_direction = sun / sun_norm if sun_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
        self._limb_darkening_coeff = limb_darkening_coeff
        self._earth_ambient = earth_ambient
        self._atmosphere_thickness_frac = atmosphere_thickness_frac
        self._atmosphere_rgb = atmosphere_rgb

    def render(self, request: RenderRequest) -> RenderResult:
        """Ray-trace Earth and project scene objects for the requested sensors.

        Args:
            request: Must include at least one of ``camera`` or ``tof``.

        Returns:
            RGB/instances from ``camera`` (if given). Depth comes from
            ``tof`` if given, else falls back to ``camera`` geometry so a
            camera-only request still returns a geometrically consistent
            depth map.

        Raises:
            ValueError: If neither camera nor ToF model is provided.
        """
        if request.camera is None and request.tof is None:
            msg = "RenderRequest requires camera and/or tof model"
            raise ValueError(msg)

        origin = request.platform_position_m.as_array()
        right, up, forward = _camera_basis(origin, request.platform_attitude)

        rgb: NDArray[np.float32] | None = None
        instances: NDArray[np.int32] | None = None
        depth: NDArray[np.float32] | None = None

        if request.camera is not None:
            cam = request.camera
            h, w = cam.height_px, cam.width_px
            fov_x, fov_y = _camera_fov(cam)
            dirs = _pixel_directions(w, h, fov_x, fov_y, right, up, forward)
            t_earth, hit = _ray_sphere_hit(origin, dirs, R_EARTH_EQUATORIAL)

            hit_points = origin[None, None, :] + t_earth[..., None] * dirs
            normals = hit_points / np.linalg.norm(hit_points, axis=-1, keepdims=True)

            # Lambertian illumination from a fixed sun direction: 0 on the
            # night side, up to 1 facing the sun straight-on.
            sun_cos = np.sum(normals * self._sun_direction[None, None, :], axis=-1)
            sun_illum = np.clip(sun_cos, 0.0, 1.0)
            illum = self._earth_ambient + (1.0 - self._earth_ambient) * sun_illum

            # Linear limb darkening w.r.t. the *camera* view angle: full
            # brightness looking straight down at the surface, darkening
            # toward the grazing limb.
            view_cos = np.clip(np.sum(normals * -dirs, axis=-1), 0.0, 1.0)
            limb = np.clip(1.0 - self._limb_darkening_coeff * (1.0 - view_cos), 0.0, 1.0)

            shade = (illum * limb).astype(np.float32)
            base = np.array(self._earth_rgb, dtype=np.float32)
            shaded = base[None, None, :] * shade[..., None]

            rgb = np.empty((h, w, 3), dtype=np.float32)
            rgb[...] = self._space_rgb
            rgb[hit] = shaded[hit]

            if self._atmosphere_thickness_frac > 0.0:
                atmo_outer = R_EARTH_EQUATORIAL * (1.0 + self._atmosphere_thickness_frac)
                closest = _closest_approach_to_center(origin, dirs)
                in_glow = (~hit) & (closest >= R_EARTH_EQUATORIAL) & (closest <= atmo_outer)
                if np.any(in_glow):
                    proximity = (atmo_outer - closest) / (atmo_outer - R_EARTH_EQUATORIAL)
                    proximity = np.clip(proximity, 0.0, 1.0).astype(np.float32)
                    glow = np.array(self._atmosphere_rgb, dtype=np.float32)
                    glow_layer = glow[None, None, :] * proximity[..., None]
                    rgb[in_glow] = glow_layer[in_glow]

            instances = np.zeros((h, w), dtype=np.int32)

            camera_depth = np.where(hit, t_earth, np.nan).astype(np.float32)
            depth_target = None if request.tof is not None else camera_depth
            _paint_targets(
                origin=origin,
                objects=request.scene.objects,
                right=right,
                up=up,
                forward=forward,
                fov_x=fov_x,
                fov_y=fov_y,
                width=w,
                height=h,
                earth_radius=R_EARTH_EQUATORIAL,
                marker_rgb=self._marker_rgb,
                marker_outline_rgb=self._marker_outline_rgb,
                marker_radius_px=self._marker_radius_px,
                rgb=rgb,
                instances=instances,
                depth=depth_target,
            )
            if request.tof is None:
                depth = camera_depth

        if request.tof is not None:
            tof = request.tof
            h, w = tof.height_px, tof.width_px
            dirs = _pixel_directions(w, h, self._tof_fov_rad, self._tof_fov_rad, right, up, forward)
            t_tof, hit_tof = _ray_sphere_hit(origin, dirs, R_EARTH_EQUATORIAL)
            in_range = hit_tof & (t_tof >= tof.min_range_m) & (t_tof <= tof.max_range_m)
            depth = np.where(in_range, t_tof, np.nan).astype(np.float32)
            _paint_targets(
                origin=origin,
                objects=request.scene.objects,
                right=right,
                up=up,
                forward=forward,
                fov_x=self._tof_fov_rad,
                fov_y=self._tof_fov_rad,
                width=w,
                height=h,
                earth_radius=R_EARTH_EQUATORIAL,
                marker_rgb=None,
                marker_outline_rgb=None,
                marker_radius_px=self._marker_radius_px,
                rgb=None,
                instances=None,
                depth=depth,
            )

        return RenderResult(
            time=request.time,
            rgb=rgb,
            depth_m=depth,
            instance_ids=instances,
            metadata={
                "backend": "geometric_v0",
                "num_objects": len(request.scene.objects),
            },
        )


__all__ = [
    "PlaceholderRenderer",
    "RenderRequest",
    "RenderResult",
    "SceneRenderer",
]
