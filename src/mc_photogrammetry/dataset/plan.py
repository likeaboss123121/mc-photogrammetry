"""Deterministic capture planning: where to stand, what to look at, what to set.

A plan is generated once and then frozen to JSON. The mod only replays it, so the
plan must be reproducible byte-for-byte from the same inputs: the seeded choice
of resolution, FOV, and orbit offset lives here, not in the client. Positions are
derived from a target by inverting the repository's projection convention, so
every planned view aims exactly at the region centre instead of relying on hand
typed angles.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mc_photogrammetry.dataset.hashing import digest_json
from mc_photogrammetry.dataset.framing_contract import (
    MAX_PROJECTED_REGION_ENVELOPE,
    MIN_CAMERA_CLEARANCE,
    MIN_PROJECTED_OCCUPANCY_HEIGHT,
    MIN_PROJECTED_OCCUPANCY_WIDTH,
)
from mc_photogrammetry.dataset.schemas import (
    MAX_PLAN_JOBS,
    CapturePlan,
    CaptureSettings,
    CameraCapture,
    DatasetSchemaError,
    RegionBounds,
)
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_eye,
    camera_projection_matrix,
    project_points,
)

MIN_VIEWS = 4
MAX_VIEWS = 8
DEFAULT_EYE_HEIGHT = 1.62
#: The four candidate vertical FOVs a seeded sweep may choose from.
FOV_CHOICES = (60.0, 70.0, 80.0, 90.0)
RESOLUTION_CHOICES = ((1920, 1080), (1600, 900), (1280, 720))
WEATHER_CHOICES = ("clear", "clear", "clear", "rain")
MAX_CAMERA_DISTANCE_FACTOR = 8.0

# Orbit framing is measured from projected continuous-region corners at each
# selected view's own vertical FOV and image dimensions. The old dominant band
# is a preference; the hard per-axis floors match evaluator policy and the safe
# upper envelope leaves margin from the image edges.
FRAMING_OCCUPANCY_MIN = 0.45
FRAMING_OCCUPANCY_MAX = 0.70
FRAMING_OCCUPANCY_TARGET = 0.58
FRAMING_OCCUPANCY_SAFE_MAX = MAX_PROJECTED_REGION_ENVELOPE
FRAMING_MIN_CAMERA_CLEARANCE = MIN_CAMERA_CLEARANCE
FRAMING_RADIUS_GRID_STEPS = 64
FRAMING_RADIUS_SEARCH_TOLERANCE = 1e-6
FRAMING_ELEVATION_MIN = 0.0
FRAMING_ELEVATION_MAX = 1.4
FRAMING_ELEVATION_STEP = 0.1
# A small deterministic azimuth neighborhood permits face-on views of strongly
# anisotropic regions to move toward a corner view when no elevation can satisfy
# both image-axis floors. Offsets preserve the seeded orbit order and stay local.
FRAMING_AZIMUTH_OFFSETS = (0.0, -0.2, 0.2, -0.4, 0.4)

#: The client's brightness option validates its value in ``[0, 1]``. The planner
#: stratifies requests across that whole supported range (rather than only the low
#: end) so a plan always exercises both a dark and a bright capture.
GAMMA_MIN = 0.0
GAMMA_MAX = 1.0
#: Fixed representative instants spanning a Minecraft day. 6000 is noon and 18000
#: is midnight, so any plan of ``MIN_VIEWS``..``MAX_VIEWS`` views contains both a
#: day and a night capture; 0 (dawn) and 12000 (dusk) keep the transition shading
#: represented too. These are capture *settings*, not a claim about weather or
#: chunk-heightmap sampling.
TIME_OF_DAY_CHOICES = (0, 6000, 12000, 18000)

#: Requested-vs-actual position tolerance in blocks that the planner promises and
#: the finalizer enforces. The Fabric client records both the player's feet and the
#: rendered eye; a capture is accepted only when each is within this distance of
#: the planned feet / ``feet + eye_height``. The mod uses the same bound before it
#: settles, so a larger finalizer value would not add safety.
EYE_POSITION_TOLERANCE = 0.25
#: Requested-vs-actual heading tolerance in degrees for yaw and pitch. Yaw is
#: compared with a circular delta so 359.9 and -0.1 are the same heading.
HEADING_TOLERANCE_DEGREES = 1.0


def yaw_pitch_to_target(eye: Sequence[float], target: Sequence[float]) -> tuple[float, float]:
    """Invert :func:`camera_rotation_matrix`: the yaw/pitch whose forward is eye->target.

    With the repo convention ``forward = (-cos(pitch)sin(yaw), -sin(pitch),
    cos(pitch)cos(yaw))``, so ``pitch = -asin(dy)`` and ``yaw = atan2(-dx, dz)``.
    """
    dx = float(target[0]) - float(eye[0])
    dy = float(target[1]) - float(eye[1])
    dz = float(target[2]) - float(eye[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-9:
        raise DatasetSchemaError("camera must not sit exactly on its target")
    dx, dy, dz = dx / length, dy / length, dz / length
    pitch = -math.degrees(math.asin(max(-1.0, min(1.0, dy))))
    yaw = math.degrees(math.atan2(-dx, dz))
    return yaw, pitch


def camera_from_target(
    position: Sequence[float],
    target: Sequence[float],
    *,
    fov_degrees: float,
    resolution: tuple[int, int],
    eye_height: float = DEFAULT_EYE_HEIGHT,
    roll_degrees: float = 0.0,
) -> CameraModel:
    """Build a camera at ``position`` whose eye looks exactly at ``target``.

    ``position`` is the player's feet, matching how the mod sets a player, so the
    eye is ``position + eye_height``. Intrinsics are attached explicitly at the
    capture resolution; the FOV is the vertical one Minecraft's projection uses.
    """
    from mc_photogrammetry.reconstruction.camera import intrinsics_for

    width, height = resolution
    yaw, pitch = yaw_pitch_to_target(
        (position[0], position[1] + eye_height, position[2]), target)
    placeholder = CameraModel(
        position=(float(position[0]), float(position[1]), float(position[2])),
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        fov_degrees=fov_degrees,
        eye_height=eye_height,
        roll_degrees=roll_degrees,
    )
    return CameraModel(
        position=placeholder.position,
        yaw_degrees=placeholder.yaw_degrees,
        pitch_degrees=placeholder.pitch_degrees,
        fov_degrees=placeholder.fov_degrees,
        eye_height=placeholder.eye_height,
        intrinsics=intrinsics_for(placeholder, width, height),
        roll_degrees=placeholder.roll_degrees,
    )


def target_pixel(camera: CameraModel, target: Sequence[float],
                 resolution: tuple[int, int]) -> tuple[float, float] | None:
    width, height = resolution
    projected = project_points([tuple(float(v) for v in target)], camera,
                               width=width, height=height)
    if not projected or projected[0] is None:
        return None
    return float(projected[0][0]), float(projected[0][1])


def _bounds_corners(bounds: RegionBounds) -> list[tuple[float, float, float]]:
    lo, hi = bounds.minimum, bounds.maximum
    return [(x, y, z)
            for x in (lo[0], hi[0] + 1.0)
            for y in (lo[1], hi[1] + 1.0)
            for z in (lo[2], hi[2] + 1.0)]


def _continuous_bounds_center(bounds: RegionBounds) -> tuple[float, float, float]:
    """Center of the voxel-volume AABB [min, max + 1]."""
    return tuple(
        float(bounds.minimum[axis]) + float(bounds.size[axis]) / 2.0
        for axis in range(3)
    )  # type: ignore[return-value]


def _bounds_projection_metrics(
    camera: CameraModel,
    bounds: RegionBounds,
    resolution: tuple[int, int],
) -> tuple[float, tuple[float, float] | None]:
    """Return half-open corner coverage and projected envelope W/H fractions."""
    width, height = resolution
    projected = project_points(_bounds_corners(bounds), camera, width=width, height=height)
    visible = [point for point in projected if point is not None]
    coverage = sum(
        0.0 <= point[0] < width and 0.0 <= point[1] < height
        for point in visible
    ) / 8.0
    if len(visible) != 8:
        return coverage, None
    occupancy = (
        (max(point[0] for point in visible) - min(point[0] for point in visible)) / width,
        (max(point[1] for point in visible) - min(point[1] for point in visible)) / height,
    )
    return coverage, occupancy


def bounds_projected_occupancy(
    camera: CameraModel,
    bounds: RegionBounds,
    resolution: tuple[int, int],
) -> tuple[float, float] | None:
    """Project the full continuous region AABB and return envelope fractions W/H.

    Every corner must be in front of the camera. The result is only a 2D
    projection measurement; it is not a depth-tested visibility estimate.
    """
    _coverage, occupancy = _bounds_projection_metrics(camera, bounds, resolution)
    return occupancy


def bounds_coverage(camera: CameraModel, bounds: RegionBounds,
                    resolution: tuple[int, int]) -> float:
    """Fraction of the region's 8 corner samples inside the half-open image."""
    return _bounds_projection_metrics(camera, bounds, resolution)[0]


def camera_sees_bounds(camera: CameraModel, bounds: RegionBounds,
                       resolution: tuple[int, int]) -> bool:
    """True when the continuous voxel-volume center is in-frame with corners present."""
    width, height = resolution
    center = _continuous_bounds_center(bounds)
    projected = project_points([center], camera, width=width, height=height)
    if not projected or projected[0] is None:
        return False
    pixel_x, pixel_y, _depth = projected[0]
    inside = 0.0 <= pixel_x < width and 0.0 <= pixel_y < height
    return bool(inside and bounds_coverage(camera, bounds, resolution) > 0.0)


def _point_to_continuous_bounds_distance(
    point: Sequence[float], bounds: RegionBounds,
) -> float:
    """Euclidean distance to the voxel-volume AABB [min, max + 1]."""
    low = bounds.minimum
    high = tuple(value + 1.0 for value in bounds.maximum)
    squared = 0.0
    for axis in range(3):
        coordinate = float(point[axis])
        if coordinate < low[axis]:
            delta = low[axis] - coordinate
        elif coordinate > high[axis]:
            delta = coordinate - high[axis]
        else:
            delta = 0.0
        squared += delta * delta
    return math.sqrt(squared)


def _camera_clearance(camera: CameraModel, bounds: RegionBounds) -> float:
    """Minimum AABB distance for both the player's feet and rendered eye."""
    return min(
        _point_to_continuous_bounds_distance(camera.position, bounds),
        _point_to_continuous_bounds_distance(camera_eye(camera), bounds),
    )


def _orbit_camera_at_radius(
    aim: Sequence[float],
    radius: float,
    angle: float,
    elevation: float,
    *,
    fov_degrees: float,
    resolution: tuple[int, int],
) -> CameraModel:
    horizontal = radius * math.cos(elevation)
    eye = (
        aim[0] + horizontal * math.sin(angle),
        aim[1] + radius * math.sin(elevation),
        aim[2] + horizontal * math.cos(angle),
    )
    position = (eye[0], eye[1] - DEFAULT_EYE_HEIGHT, eye[2])
    return camera_from_target(
        position, aim, fov_degrees=fov_degrees, resolution=resolution)


def _fit_orbit_at_elevation(
    *,
    aim: Sequence[float],
    bounds: RegionBounds,
    angle: float,
    elevation: float,
    fov_degrees: float,
    resolution: tuple[int, int],
    preferred_radius: float,
    scene_id: str,
    view_index: int,
) -> CameraModel:
    """Choose the closest QA-compatible camera at one fixed orbit elevation."""
    max_radius = max(16.0, bounds.diagonal * MAX_CAMERA_DISTANCE_FACTOR)

    def camera_at(radius: float) -> CameraModel | None:
        if radius <= 1e-9:
            return None
        return _orbit_camera_at_radius(
            aim, radius, angle, elevation,
            fov_degrees=fov_degrees, resolution=resolution)

    # Find the first radius that clears both feet and eye from the continuous
    # voxel AABB. This predicate is monotone along an outward ray from an aim
    # inside the bounds, and the fixed bisection makes the minimum deterministic.
    high = min(max_radius, max(0.5, bounds.diagonal * 0.05))
    candidate = camera_at(high)
    while (candidate is None or _camera_clearance(candidate, bounds)
           < FRAMING_MIN_CAMERA_CLEARANCE - FRAMING_RADIUS_SEARCH_TOLERANCE):
        if high >= max_radius:
            raise DatasetSchemaError(
                f"planned view {view_index} for scene {scene_id!r} cannot clear "
                f"the region by {FRAMING_MIN_CAMERA_CLEARANCE:g} blocks within "
                f"{max_radius:g} blocks")
        high = min(max_radius, high * 1.5)
        candidate = camera_at(high)
    low = 0.0
    for _ in range(40):
        middle = (low + high) / 2.0
        middle_camera = camera_at(middle)
        if (middle_camera is not None
                and _camera_clearance(middle_camera, bounds)
                >= FRAMING_MIN_CAMERA_CLEARANCE):
            high = middle
        else:
            low = middle
    min_clear_radius = min(max_radius, high + 1e-4)

    # Sample logarithmically from the clearance limit to the validation cap.
    # Each sample uses all eight actual projected corners for this view's FOV
    # and resolution. A bounded grid avoids assumptions about projected aspect
    # ratio for tall, wide, or deep regions.
    if max_radius <= min_clear_radius * (1.0 + 1e-12):
        radii = [min_clear_radius]
    else:
        log_span = math.log(max_radius / min_clear_radius)
        radii = [
            min_clear_radius * math.exp(log_span * index / (FRAMING_RADIUS_GRID_STEPS - 1))
            for index in range(FRAMING_RADIUS_GRID_STEPS)
        ]
    radii.append(min(max_radius, max(min_clear_radius, preferred_radius)))
    radii = sorted(set(radii))

    records: list[tuple[float, CameraModel, float, tuple[float, float] | None]] = []
    for radius in radii:
        camera = camera_at(radius)
        if camera is None:
            continue
        coverage, occupancy = _bounds_projection_metrics(camera, bounds, resolution)
        records.append((radius, camera, coverage, occupancy))

    # Refine any sampled crossing of the target with a deterministic bisection.
    # Keep both sides as candidates so the final choice remains the closest
    # feasible occupancy if pixel geometry or an explicit off-centre aim causes
    # a small non-monotone interval.
    sampled_records = tuple(sorted(records, key=lambda item: item[0]))
    for first, second in zip(sampled_records, sampled_records[1:]):
        first_occ = None if first[3] is None else max(first[3])
        second_occ = None if second[3] is None else max(second[3])
        if first_occ is None or second_occ is None:
            continue
        if (first_occ - FRAMING_OCCUPANCY_TARGET) * (
                second_occ - FRAMING_OCCUPANCY_TARGET) > 0.0:
            continue
        left_radius, left_occ = first[0], first_occ
        right_radius, right_occ = second[0], second_occ
        if right_radius <= left_radius:
            continue
        for _ in range(24):
            middle_radius = math.sqrt(left_radius * right_radius)
            middle_camera = camera_at(middle_radius)
            if middle_camera is None:
                break
            coverage, occupancy = _bounds_projection_metrics(
                middle_camera, bounds, resolution)
            records.append((middle_radius, middle_camera, coverage, occupancy))
            if occupancy is None:
                break
            middle_occ = max(occupancy)
            if (left_occ - FRAMING_OCCUPANCY_TARGET) * (
                    middle_occ - FRAMING_OCCUPANCY_TARGET) <= 0.0:
                right_radius, right_occ = middle_radius, middle_occ
            else:
                left_radius, left_occ = middle_radius, middle_occ

    feasible = [
        (0 if FRAMING_OCCUPANCY_MIN <= max(occupancy) <= FRAMING_OCCUPANCY_MAX else 1,
         abs(max(occupancy) - FRAMING_OCCUPANCY_TARGET),
         abs(radius - preferred_radius), radius, camera)
        for radius, camera, coverage, occupancy in records
        if occupancy is not None
        and coverage == 1.0
        and _camera_clearance(camera, bounds)
        >= FRAMING_MIN_CAMERA_CLEARANCE - FRAMING_RADIUS_SEARCH_TOLERANCE
        and occupancy[0] + FRAMING_RADIUS_SEARCH_TOLERANCE
        >= MIN_PROJECTED_OCCUPANCY_WIDTH
        and occupancy[1] + FRAMING_RADIUS_SEARCH_TOLERANCE
        >= MIN_PROJECTED_OCCUPANCY_HEIGHT
        and max(occupancy)
        <= FRAMING_OCCUPANCY_SAFE_MAX + FRAMING_RADIUS_SEARCH_TOLERANCE
    ]
    if not feasible:
        sampled_occupancies = [
            max(item[3]) for item in records if item[3] is not None
        ]
        closest_sampled_occupancy = min(
            sampled_occupancies,
            key=lambda value: abs(value - FRAMING_OCCUPANCY_TARGET),
            default=float("nan"),
        )
        raise DatasetSchemaError(
            f"planned view {view_index} for scene {scene_id!r} has no camera "
            f"within {max_radius:g} blocks meeting clearance, full corner coverage, "
            f"and projected width/height floors ({MIN_PROJECTED_OCCUPANCY_WIDTH:.2f}, "
            f"{MIN_PROJECTED_OCCUPANCY_HEIGHT:.2f}) with safe dominant maximum "
            f"{FRAMING_OCCUPANCY_SAFE_MAX:.2f} (preferred dominant band "
            f"[{FRAMING_OCCUPANCY_MIN:.2f}, {FRAMING_OCCUPANCY_MAX:.2f}]; "
            f"closest sampled dominant envelope "
            f"to target {FRAMING_OCCUPANCY_TARGET:.2f}: "
            f"{closest_sampled_occupancy:.3f})")
    feasible.sort(key=lambda item: (item[0], item[1], item[2]))
    return feasible[0][4]


def _fit_orbit_camera(
    *,
    aim: Sequence[float],
    bounds: RegionBounds,
    angle: float,
    elevation: float,
    fov_degrees: float,
    resolution: tuple[int, int],
    preferred_radius: float,
    scene_id: str,
    view_index: int,
) -> CameraModel:
    """Search seeded elevation first, then a bounded deterministic rescue grid.

    The dominant 0.45--0.70 band remains preferred when it can satisfy both
    evaluator width/height floors. If that band is infeasible for the region's
    aspect ratio, the safe upper envelope is allowed up to 0.85 while the same
    hard per-axis floors, corner coverage, and camera clearance remain enforced.
    """
    elevation_count = round(
        (FRAMING_ELEVATION_MAX - FRAMING_ELEVATION_MIN) / FRAMING_ELEVATION_STEP)
    grid = tuple(
        round(FRAMING_ELEVATION_MIN + index * FRAMING_ELEVATION_STEP, 10)
        for index in range(elevation_count + 1)
    )
    elevation_candidates = [float(elevation)] + sorted(
        (candidate for candidate in grid
         if abs(candidate - float(elevation)) > FRAMING_RADIUS_SEARCH_TOLERANCE),
        key=lambda candidate: (abs(candidate - float(elevation)), candidate),
    )
    fallback: list[tuple[float, float, float, float, CameraModel]] = []
    last_error: DatasetSchemaError | None = None
    offset_groups = sorted({abs(value) for value in FRAMING_AZIMUTH_OFFSETS})
    for offset_size in offset_groups:
        preferred_at_offset: list[tuple[float, float, float, float, CameraModel]] = []
        offsets = tuple(value for value in FRAMING_AZIMUTH_OFFSETS
                        if abs(value) == offset_size)
        for angle_offset in offsets:
            candidate_angle = float(angle) + angle_offset
            for candidate_elevation in elevation_candidates:
                try:
                    camera = _fit_orbit_at_elevation(
                        aim=aim,
                        bounds=bounds,
                        angle=candidate_angle,
                        elevation=candidate_elevation,
                        fov_degrees=fov_degrees,
                        resolution=resolution,
                        preferred_radius=preferred_radius,
                        scene_id=scene_id,
                        view_index=view_index,
                    )
                except DatasetSchemaError as error:
                    last_error = error
                    continue
                occupancy = bounds_projected_occupancy(camera, bounds, resolution)
                if occupancy is None:
                    continue
                dominant = max(occupancy)
                radius = math.dist(camera_eye(camera), aim)
                score = (
                    abs(dominant - FRAMING_OCCUPANCY_TARGET),
                    abs(candidate_elevation - float(elevation)),
                    abs(radius - preferred_radius),
                    angle_offset,
                    camera,
                )
                if FRAMING_OCCUPANCY_MIN <= dominant <= FRAMING_OCCUPANCY_MAX:
                    preferred_at_offset.append(score)
                else:
                    fallback.append((
                        abs(dominant - FRAMING_OCCUPANCY_TARGET),
                        abs(angle_offset),
                        abs(candidate_elevation - float(elevation)),
                        abs(radius - preferred_radius),
                        camera,
                    ))
        # Keep seeded azimuths whenever possible. If the preferred occupancy
        # band is impossible there, use the nearest deterministic orbit offset
        # that makes the complete QA contract feasible.
        if preferred_at_offset:
            preferred_at_offset.sort(key=lambda item: item[:4])
            return preferred_at_offset[0][4]
    if not fallback:
        elevations_text = ", ".join(f"{value:.1f}" for value in elevation_candidates)
        azimuth_text = ", ".join(f"{value:+.1f}" for value in FRAMING_AZIMUTH_OFFSETS)
        detail = "" if last_error is None else f"; preferred-ray detail: {last_error}"
        raise DatasetSchemaError(
            f"planned view {view_index} for scene {scene_id!r} has no camera meeting "
            f"occupancy requirements across bounded azimuth offsets [{azimuth_text}] "
            f"and elevations [{elevations_text}], satisfying corner coverage, "
            f"{FRAMING_MIN_CAMERA_CLEARANCE:g}-block clearance, per-axis projected "
            f"floors ({MIN_PROJECTED_OCCUPANCY_WIDTH:.2f}, "
            f"{MIN_PROJECTED_OCCUPANCY_HEIGHT:.2f}), and safe dominant maximum "
            f"{FRAMING_OCCUPANCY_SAFE_MAX:.2f}{detail}")
    fallback.sort(key=lambda item: item[:4])
    return fallback[0][4]


def plan_id_for(dataset_id: str, scene_id: str, jobs: Sequence[CameraCapture]) -> str:
    digest = digest_json({
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "jobs": [job.to_dict() for job in jobs],
    })
    return f"plan-{digest[:16]}"


def view_setting_strata(view_count: int) -> tuple[tuple[float, int, bool], ...]:
    """Deterministic ``(gamma, time_of_day, hud_hidden)`` coverage for one plan.

    Gamma is evenly stratified over the client's supported ``[GAMMA_MIN,
    GAMMA_MAX]`` brightness range, so every allowed view count (4-8) contains both
    the dark and the bright endpoint. Time cycles the fixed representative day and
    night set, and HUD alternates between hidden (clean capture) and visible
    (ordinary screen). The seed only permutes which view receives which stratum;
    the multiset returned here is a pure function of ``view_count``, so coverage
    cannot depend on the seed.
    """
    if view_count < 1:
        raise DatasetSchemaError("view_count must be at least 1")
    if view_count == 1:
        # A single view cannot carry two endpoints; keep it mid-range rather than
        # silently claiming coverage the view count cannot provide.
        levels = [(GAMMA_MIN + GAMMA_MAX) / 2.0]
    else:
        levels = [GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * index / (view_count - 1)
                  for index in range(view_count)]
    return tuple(
        (round(levels[index], 4),
         int(TIME_OF_DAY_CHOICES[index % len(TIME_OF_DAY_CHOICES)]),
         bool(index % 2))
        for index in range(view_count)
    )


def build_plan(
    *,
    dataset_id: str,
    scene_id: str,
    world_identity: str,
    region: RegionBounds,
    seed: int,
    staging_token: str,
    view_count: int = 6,
    target: Sequence[float] | None = None,
    base_settings: CaptureSettings | None = None,
    frame_prefix: str = "view",
) -> CapturePlan:
    """Build one frozen plan for one region with a seeded orbit sweep.

    ``staging_token`` is required: it is the content fingerprint of the staged
    world the plan may run against, and there is no inert default. A plan is only
    ever built from a world that was really staged, so the Fabric mod can read the
    same token out of the copied world's own marker before it mutates anything.
    A synthetic fixture must pass an explicit digest of its own; it must never
    quietly inherit the all-zero placeholder, because a placeholder that reaches a
    client is refused there and a plan that cannot run is worse than no plan.

    Client settings are covered, not merely drawn: see
    :func:`view_setting_strata`. An explicit ``base_settings`` is fixed and
    overrides that variation entirely, which is how a caller pins one exposure or
    time of day for every view.
    """
    if not staging_token or not str(staging_token).strip():
        raise DatasetSchemaError(
            "build_plan requires an explicit staging_token cut from a staged world")
    if not MIN_VIEWS <= view_count <= MAX_VIEWS:
        raise DatasetSchemaError(f"view_count must be between {MIN_VIEWS} and {MAX_VIEWS}")
    if not dataset_id or not scene_id:
        raise DatasetSchemaError("dataset_id and scene_id are required")
    continuous_center = _continuous_bounds_center(region)
    aim = tuple(float(value) for value in (
        target if target is not None else continuous_center))
    if len(aim) != 3:
        raise DatasetSchemaError("target must have three coordinates")

    rng = random.Random(f"{seed}|{scene_id}|{region.minimum}|{region.maximum}")
    # The strata are fixed by the view count; the seed only permutes which view
    # receives which one, so a reseed changes the orbit and the assignment order but
    # can never drop the dark/bright, day/night, or HUD coverage assertions.
    strata = view_setting_strata(view_count)
    assignment = list(range(view_count))
    rng.shuffle(assignment)
    base_radius = max(6.0, region.diagonal * 0.9)
    base_elevation = 0.28
    jobs: list[CameraCapture] = []
    for index in range(view_count):
        # Draw every seeded choice once per view before deterministic framing
        # search; changing its radius cannot reshuffle resolution, FOV, or weather.
        angle = 2.0 * math.pi * index / view_count + rng.uniform(-0.12, 0.12)
        elevation = base_elevation + rng.uniform(-0.08, 0.12)
        radius_jitter = 1.0 + rng.uniform(-0.05, 0.10)
        resolution = RESOLUTION_CHOICES[rng.randrange(len(RESOLUTION_CHOICES))]
        fov = FOV_CHOICES[rng.randrange(len(FOV_CHOICES))]
        if base_settings is not None:
            # An explicit request is fixed: it wins over the stratified variation.
            settings = CaptureSettings(
                gamma=base_settings.gamma,
                time_of_day=base_settings.time_of_day,
                weather=base_settings.weather,
                hud_hidden=base_settings.hud_hidden,
                resource_pack=base_settings.resource_pack,
                shader_pack=base_settings.shader_pack,
            )
        else:
            gamma, time_of_day, hud_hidden = strata[assignment[index]]
            settings = CaptureSettings(
                gamma=gamma,
                time_of_day=time_of_day,
                weather=WEATHER_CHOICES[rng.randrange(len(WEATHER_CHOICES))],
                hud_hidden=hud_hidden,
            )
        preferred_radius = base_radius * radius_jitter
        camera = _fit_orbit_camera(
            aim=aim,
            bounds=region,
            angle=angle,
            elevation=elevation,
            fov_degrees=fov,
            resolution=resolution,
            preferred_radius=preferred_radius,
            scene_id=scene_id,
            view_index=index,
        )
        jobs.append(CameraCapture(
            frame_id=f"{frame_prefix}-{scene_id}-{index:02d}",
            camera=camera,
            resolution=resolution,
            target=aim,
            settings=settings,
        ))

    plan = CapturePlan(
        plan_id=plan_id_for(dataset_id, scene_id, jobs),
        scene_id=scene_id,
        dataset_id=dataset_id,
        world_identity=world_identity,
        region=region,
        jobs=tuple(jobs),
        staging_token=str(staging_token),
        notes=("developer/evaluation capture plan; ground truth only, "
               "never an input to normal reconstruction",),
    )
    validate_plan(plan)
    return plan


def validate_plan(plan: CapturePlan, *, max_jobs: int = MAX_PLAN_JOBS) -> None:
    """Independent sanity pass over a plan before it is written or armed."""
    if len(plan.jobs) > max_jobs:
        raise DatasetSchemaError(f"plan exceeds {max_jobs} jobs")
    for job in plan.jobs:
        camera = job.camera
        if camera.intrinsics is None:
            raise DatasetSchemaError(f"{job.frame_id}: planned camera needs explicit intrinsics")
        if camera.intrinsics.image_size != job.resolution:
            raise DatasetSchemaError(
                f"{job.frame_id}: intrinsics size does not match resolution")
        # Minecraft's client cannot apply roll, so a rolled plan can never be
        # honoured; refusing it here keeps the frozen ground truth honest.
        if float(camera.roll_degrees) != 0.0:
            raise DatasetSchemaError(
                f"{job.frame_id}: roll_degrees must be 0 because the client cannot apply roll")
        aim = tuple(float(value) for value in job.target)
        if not plan.region.contains(tuple(int(math.floor(value)) for value in aim)):
            raise DatasetSchemaError(f"{job.frame_id}: target lies outside the region")
        eye = camera_eye(camera)
        distance = math.dist(eye, aim)
        if distance > max(16.0, plan.region.diagonal * MAX_CAMERA_DISTANCE_FACTOR):
            raise DatasetSchemaError(f"{job.frame_id}: camera is implausibly far from its region")
        pixel = target_pixel(camera, aim, job.resolution)
        if pixel is None:
            raise DatasetSchemaError(f"{job.frame_id}: target is behind the camera")
        width, height = job.resolution
        if not (0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height):
            raise DatasetSchemaError(f"{job.frame_id}: target does not project into frame")
        coverage, occupancy = _bounds_projection_metrics(
            camera, plan.region, job.resolution)
        if coverage != 1.0 or occupancy is None:
            raise DatasetSchemaError(
                f"{job.frame_id}: region corners must all project inside the frame")
        if (occupancy[0] + FRAMING_RADIUS_SEARCH_TOLERANCE
                < MIN_PROJECTED_OCCUPANCY_WIDTH
                or occupancy[1] + FRAMING_RADIUS_SEARCH_TOLERANCE
                < MIN_PROJECTED_OCCUPANCY_HEIGHT):
            raise DatasetSchemaError(
                f"{job.frame_id}: projected region width/height must meet floors "
                f"({MIN_PROJECTED_OCCUPANCY_WIDTH:.2f}, "
                f"{MIN_PROJECTED_OCCUPANCY_HEIGHT:.2f})")
        if max(occupancy) > FRAMING_OCCUPANCY_SAFE_MAX + FRAMING_RADIUS_SEARCH_TOLERANCE:
            raise DatasetSchemaError(
                f"{job.frame_id}: projected region exceeds safe envelope "
                f"{FRAMING_OCCUPANCY_SAFE_MAX:.2f}")
        if _camera_clearance(camera, plan.region) \
                < FRAMING_MIN_CAMERA_CLEARANCE - FRAMING_RADIUS_SEARCH_TOLERANCE:
            raise DatasetSchemaError(
                f"{job.frame_id}: camera lacks "
                f"{FRAMING_MIN_CAMERA_CLEARANCE:g}-block region clearance")
        matrix = camera_projection_matrix(camera, width=width, height=height)
        if not all(math.isfinite(float(value)) for value in matrix.flat):
            raise DatasetSchemaError(f"{job.frame_id}: projection matrix is not finite")


@dataclass(frozen=True, slots=True)
class RegionRequest:
    """A developer-declared evaluation region, optionally with its own view count."""

    name: str
    bounds: RegionBounds
    target: tuple[float, float, float] | None = None
    views: int = 6

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "views": int(self.views),
            "min": list(self.bounds.minimum),
            "max": list(self.bounds.maximum),
        }
        if self.target is not None:
            payload["target"] = [float(value) for value in self.target]
        return payload


def load_region_requests(path: Path | str) -> tuple[RegionRequest, ...]:
    """Read a developer region file; refuses anything that is not explicit bounds."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DatasetSchemaError("region file root must be an object")
    if payload.get("schema_version") != 1:
        raise DatasetSchemaError(
            f"unsupported region file schema_version {payload.get('schema_version')!r}")
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, (list, tuple)) or not raw_regions:
        raise DatasetSchemaError("region file must list at least one region")
    requests: list[RegionRequest] = []
    for raw in raw_regions:
        raw = _as_mapping(raw)
        bounds = RegionBounds.from_dict({"min": raw.get("min"), "max": raw.get("max")})
        target = raw.get("target")
        requests.append(RegionRequest(
            name=str(raw.get("name", "")),
            bounds=bounds,
            target=None if target is None else tuple(float(value) for value in target),
            views=int(raw.get("views", 6)),
        ))
    return tuple(requests)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetSchemaError("region entries must be objects")
    return value


def landscape_region(
    *,
    spawn_x: int,
    spawn_z: int,
    surface_y: int,
    radius: int = 32,
    vertical_below: int = 8,
    vertical_above: int = 24,
) -> RegionBounds:
    """A bounded landscape sample around a spawn point.

    ``surface_y`` is the spawn height used to centre the sample - in the factory's
    default path the save's own ``SpawnY`` - not a sampled terrain heightmap: this
    factory deliberately does not parse Minecraft's chunk format, and a wrong height
    here only mis-centres a camera, never corrupts a world.
    """
    if radius <= 0 or vertical_below < 0 or vertical_above <= 0:
        raise DatasetSchemaError("landscape sample extents must be positive")
    return RegionBounds(
        (int(spawn_x) - radius, int(surface_y) - vertical_below, int(spawn_z) - radius),
        (int(spawn_x) + radius, int(surface_y) + vertical_above, int(spawn_z) + radius),
    )


def spawn_from_save(save_level_dat: Path) -> tuple[int, int]:
    """Read ``SpawnX``/``SpawnZ`` from ``level.dat``; zero when unreadable.

    Display/inventory use only. This degrades quietly, so it must not be used to
    place cameras: :func:`spawn_position_from_save` is the reader that fails loudly.
    """
    try:
        import nbtlib

        root = nbtlib.load(save_level_dat)
        data = root["Data"]
        return int(data.get("SpawnX", 0)), int(data.get("SpawnZ", 0))
    except Exception:  # pragma: no cover - damaged/absent level.dat
        return 0, 0


def spawn_position_from_save(save_level_dat: Path) -> tuple[int, int, int]:
    """Read the full ``SpawnX``/``SpawnY``/``SpawnZ`` spawn position.

    Used to *place* a landscape sample, so a missing or unreadable spawn fails
    loudly instead of silently assuming a terrain height (which would centre the
    cameras on the wrong terrain). The caller may pass an explicit height instead;
    the CLI exposes that as ``--surface-y``. ``nbtlib`` is a project dependency.
    """
    try:
        import nbtlib

        root = nbtlib.load(save_level_dat)
        data = root["Data"]
        return int(data["SpawnX"]), int(data["SpawnY"]), int(data["SpawnZ"])
    except Exception as error:
        raise DatasetSchemaError(
            f"cannot read SpawnX/SpawnY/SpawnZ from {save_level_dat}: {error}") from error
