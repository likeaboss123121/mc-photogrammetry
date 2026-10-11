"""Evaluation-only capture QA and framing helpers.

This module reads finalized capture and ground-truth artifacts for dataset
promotion review. Its metadata must not be passed into normal reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from mc_photogrammetry.dataset.framing_contract import (
    MAX_PROJECTED_REGION_ENVELOPE,
    MIN_CAMERA_CLEARANCE,
    MIN_PROJECTED_OCCUPANCY_HEIGHT,
    MIN_PROJECTED_OCCUPANCY_WIDTH,
)
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_eye,
    intrinsics_for,
    project_points,
)


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Provisional promotion policy defaults, not empirically validated limits.

    Synthetic bright/dark controls freeze the defaults without fitting them to
    a captured scene. Operators should treat them as review gates, not universal
    camera-quality thresholds.
    """

    near_black_level: int = 16
    clipped_level: int = 250
    max_severe_near_black_fraction: float = 0.80
    max_clipped_fraction: float = 0.10
    min_luminance_std: float = 8.0
    min_luminance_entropy_bits: float = 4.0
    top_band_fraction: float = 0.10
    max_top_band_dark_fraction: float = 0.40
    min_projected_occupancy_width: float = MIN_PROJECTED_OCCUPANCY_WIDTH
    min_projected_occupancy_height: float = MIN_PROJECTED_OCCUPANCY_HEIGHT
    min_projected_region_corner_coverage: float = 1.0
    min_projected_gt_center_coverage: float = 0.95
    min_usable_views: int = 4

    def __post_init__(self) -> None:
        if not 0 <= self.near_black_level < self.clipped_level <= 255:
            raise ValueError("luminance levels must be ordered within [0, 255]")
        for name in (
            "max_severe_near_black_fraction",
            "max_clipped_fraction",
            "top_band_fraction",
            "max_top_band_dark_fraction",
            "min_projected_occupancy_width",
            "min_projected_occupancy_height",
            "min_projected_region_corner_coverage",
            "min_projected_gt_center_coverage",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if not math.isfinite(self.min_luminance_std) or self.min_luminance_std < 0:
            raise ValueError("min_luminance_std must be finite and non-negative")
        if (not math.isfinite(self.min_luminance_entropy_bits)
                or self.min_luminance_entropy_bits < 0):
            raise ValueError("min_luminance_entropy_bits must be finite and non-negative")
        if self.min_usable_views < 1:
            raise ValueError("min_usable_views must be positive")


DEFAULT_QUALITY_POLICY = QualityPolicy()

_BLOCK_VOLUME_CORNERS = tuple(
    (x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)
)
_REFRAME_MIN_RADIAL_SCALE = 0.05
_REFRAME_GRID_STEPS = 96


@dataclass(frozen=True, slots=True)
class FrameQuality:
    frame_id: str
    filename: str
    width: int | None
    height: int | None
    capture_integrity_ok: bool
    integrity_issues: tuple[str, ...]
    near_black_fraction: float | None
    clipped_fraction: float | None
    luminance_mean: float | None
    luminance_std: float | None
    luminance_entropy_bits: float | None
    top_band_dark_fraction: float | None
    hud_hidden: bool | None
    projected_region_corner_coverage: float | None
    # Occupancy uses projected occupied voxel-volume support; center coverage is
    # a separate sampling metric and does not define the object envelope.
    projected_gt_occupancy_width: float | None
    projected_gt_occupancy_height: float | None
    projected_gt_center_coverage: float | None
    projected_gt_truncated: bool | None
    adverse_tags: tuple[str, ...]
    promotion_usable: bool
    promotion_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in ((field, getattr(self, field))
                                for field in self.__dataclass_fields__)
        }


@dataclass(frozen=True, slots=True)
class CaptureQualityReport:
    scene_id: str
    capture_integrity_ok: bool
    scene_integrity_issues: tuple[str, ...]
    promotion_eligible: bool
    retained_as_adverse_data: bool
    usable_view_count: int
    required_usable_view_count: int
    reasons: tuple[str, ...]
    frames: tuple[FrameQuality, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "capture_integrity_ok": self.capture_integrity_ok,
            "scene_integrity_issues": list(self.scene_integrity_issues),
            "promotion_eligible": self.promotion_eligible,
            "retained_as_adverse_data": self.retained_as_adverse_data,
            "usable_view_count": self.usable_view_count,
            "required_usable_view_count": self.required_usable_view_count,
            "reasons": list(self.reasons),
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True, slots=True)
class OrbitFraming:
    horizontal_radius: float
    elevation_degrees: float
    predicted_width_fraction: float
    predicted_height_fraction: float
    horizontal_clearance: float


@dataclass(frozen=True, slots=True)
class EvaluationReframeProposal:
    """Read-only, evaluation-only raywise camera proposal with typed abstention.

    This object is not part of capture planning or reconstruction inputs. It
    describes a candidate camera for a reviewer to inspect; it never rewrites
    a finalized camera or capture artifact.
    """

    camera: CameraModel | None
    radial_scale: float | None
    projected_gt_occupancy_width: float | None
    projected_gt_occupancy_height: float | None
    projected_region_occupancy_width: float | None
    projected_region_occupancy_height: float | None
    projected_region_corner_coverage: float | None
    projected_gt_center_coverage: float | None
    feet_clearance: float | None
    eye_clearance: float | None
    abstention_reason: str | None

    @property
    def feasible(self) -> bool:
        return self.camera is not None and self.abstention_reason is None


def _region_corners(region: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    try:
        low = tuple(float(value) for value in region["min"])
        high = tuple(float(value) + 1.0 for value in region["max"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("region must contain three-dimensional min and max bounds") from error
    if (len(low) != 3 or len(high) != 3
            or any(not math.isfinite(v) for v in (*low, *high))
            or any(high[i] <= low[i] for i in range(3))):
        raise ValueError("region bounds must be finite and strictly increasing")
    return [(x, y, z) for x in (low[0], high[0])
            for y in (low[1], high[1]) for z in (low[2], high[2])]


def _occupied_block_centers(ground_truth: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    air_names = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
    blocks = ground_truth.get("blocks", ())
    points: list[tuple[float, float, float]] = []
    for block in blocks:
        if not isinstance(block, Mapping) or block.get("name") in air_names:
            continue
        position = block.get("pos")
        if not isinstance(position, Sequence) or len(position) != 3:
            continue
        point = tuple(float(value) + 0.5 for value in position)
        if all(math.isfinite(value) for value in point):
            points.append(point)
    return points


def _projected_occupied_voxel_envelope(
    centers: Sequence[tuple[float, float, float]],
    camera: CameraModel,
    *,
    width: int,
    height: int,
    blocks_per_batch: int = 1024,
) -> tuple[float | None, float | None, bool]:
    """Project occupied unit-voxel support, returning envelope and truncation.

    A block occupies ``[x, x + 1] × [y, y + 1] × [z, z + 1]``. Projecting its
    corners measures the visible voxel volume instead of treating each block as
    a dimensionless center sample. Batches bound temporary memory for large GT.
    """
    if blocks_per_batch < 1:
        raise ValueError("blocks_per_batch must be positive")
    if not centers:
        return None, None, False

    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    total_corners = len(centers) * len(_BLOCK_VOLUME_CORNERS)
    in_frame_corners = 0
    in_front_corners = 0
    for start in range(0, len(centers), blocks_per_batch):
        batch = centers[start:start + blocks_per_batch]
        points = (
            (center[0] - 0.5 + dx,
             center[1] - 0.5 + dy,
             center[2] - 0.5 + dz)
            for center in batch
            for dx, dy, dz in _BLOCK_VOLUME_CORNERS
        )
        projected = project_points(points, camera, width=width, height=height)
        for point in projected:
            if point is None:
                continue
            in_front_corners += 1
            pixel_x, pixel_y, _depth = point
            min_x = min(min_x, pixel_x)
            max_x = max(max_x, pixel_x)
            min_y = min(min_y, pixel_y)
            max_y = max(max_y, pixel_y)
            if 0.0 <= pixel_x < width and 0.0 <= pixel_y < height:
                in_frame_corners += 1

    if in_front_corners == 0:
        return None, None, True
    return (
        (max_x - min_x) / width,
        (max_y - min_y) / height,
        in_frame_corners != total_corners,
    )


def propose_evaluation_only_reframe(
    camera: CameraModel,
    ground_truth: Mapping[str, Any],
    *,
    width: int,
    height: int,
    policy: QualityPolicy = DEFAULT_QUALITY_POLICY,
) -> EvaluationReframeProposal:
    """Propose an inward raywise camera move for evaluation review only.

    The function reads the evaluator's finalized occupied-block records and
    requested region. It preserves camera orientation, vertical FOV, stored
    intrinsics, eye height, and external capture settings. Candidate feet and
    eye positions must both remain at least two blocks from the continuous
    region; requested-region corners must fit the half-open image with a safe
    projected envelope, while occupied voxel support must meet the frozen QA
    floors and center-coverage policy. A fixed 96-step radial grid makes the
    proposal deterministic and bounded. It never edits plans or captures.
    """
    def abstain(reason: str) -> EvaluationReframeProposal:
        return EvaluationReframeProposal(
            camera=None, radial_scale=None,
            projected_gt_occupancy_width=None,
            projected_gt_occupancy_height=None,
            projected_region_occupancy_width=None,
            projected_region_occupancy_height=None,
            projected_region_corner_coverage=None,
            projected_gt_center_coverage=None,
            feet_clearance=None, eye_clearance=None,
            abstention_reason=reason,
        )

    if width <= 0 or height <= 0:
        return abstain("invalid_image_dimensions")
    try:
        region = ground_truth["region"]
        if not isinstance(region, Mapping):
            return abstain("invalid_requested_region")
        region_corners = _region_corners(region)
        low = tuple(float(value) for value in region["min"])
        high = tuple(float(value) + 1.0 for value in region["max"])
        centers = _occupied_block_centers(ground_truth)
    except (KeyError, TypeError, ValueError):
        return abstain("invalid_evaluation_geometry")
    if not centers:
        return abstain("no_occupied_ground_truth_blocks")

    aim = tuple((low[index] + high[index]) / 2.0 for index in range(3))
    original_eye = camera_eye(camera)
    ray = original_eye - np.asarray(aim, dtype=np.float64)
    ray_length = float(np.linalg.norm(ray))
    if not math.isfinite(ray_length) or ray_length <= 1e-9:
        return abstain("camera_eye_at_region_center")

    # A radial translation is meaningful only when the current optical axis is
    # already aimed at the requested continuous-region center. Verify against
    # the actual calibrated principal point instead of assuming frame midpoint.
    aim_pixel = project_points([aim], camera, width=width, height=height)[0]
    calibration = intrinsics_for(camera, width, height)
    if aim_pixel is None or math.dist(
            aim_pixel[:2], calibration.principal_point_pixels) > 0.5:
        return abstain("camera_not_aimed_at_region_center")

    def point_clearance(point: Sequence[float]) -> float:
        squared = 0.0
        for axis in range(3):
            coordinate = float(point[axis])
            delta = (low[axis] - coordinate if coordinate < low[axis]
                     else coordinate - high[axis] if coordinate > high[axis]
                     else 0.0)
            squared += delta * delta
        return math.sqrt(squared)

    def evaluate_scale(scale: float) -> EvaluationReframeProposal | None:
        eye = np.asarray(aim, dtype=np.float64) + scale * ray
        feet = eye - np.asarray((0.0, camera.eye_height, 0.0), dtype=np.float64)
        candidate = CameraModel(
            position=tuple(float(value) for value in feet),
            yaw_degrees=camera.yaw_degrees,
            pitch_degrees=camera.pitch_degrees,
            fov_degrees=camera.fov_degrees,
            eye_height=camera.eye_height,
            intrinsics=camera.intrinsics,
            roll_degrees=camera.roll_degrees,
        )
        projected_region = project_points(
            region_corners, candidate, width=width, height=height)
        front_region = [point for point in projected_region if point is not None]
        in_frame_region = [
            point for point in front_region
            if 0.0 <= point[0] < width and 0.0 <= point[1] < height
        ]
        region_coverage = len(in_frame_region) / 8.0
        if len(front_region) == 8:
            region_width = (max(point[0] for point in front_region)
                            - min(point[0] for point in front_region)) / width
            region_height = (max(point[1] for point in front_region)
                             - min(point[1] for point in front_region)) / height
        else:
            region_width = region_height = None

        projected_centers = project_points(
            centers, candidate, width=width, height=height)
        in_frame_centers = [
            point for point in projected_centers
            if point is not None and 0.0 <= point[0] < width
            and 0.0 <= point[1] < height
        ]
        center_coverage = len(in_frame_centers) / len(centers)
        occupied_width, occupied_height, occupied_truncated = (
            _projected_occupied_voxel_envelope(
                centers, candidate, width=width, height=height))
        feet_clearance = point_clearance(candidate.position)
        eye_clearance = point_clearance(camera_eye(candidate))
        if (
            len(in_frame_region) != 8
            or region_width is None or region_height is None
            or max(region_width, region_height) > MAX_PROJECTED_REGION_ENVELOPE
            or occupied_width is None or occupied_height is None
            or occupied_truncated
            or occupied_width < policy.min_projected_occupancy_width
            or occupied_height < policy.min_projected_occupancy_height
            or center_coverage < policy.min_projected_gt_center_coverage
            or min(feet_clearance, eye_clearance) < MIN_CAMERA_CLEARANCE
        ):
            return None
        return EvaluationReframeProposal(
            camera=candidate,
            radial_scale=scale,
            projected_gt_occupancy_width=occupied_width,
            projected_gt_occupancy_height=occupied_height,
            projected_region_occupancy_width=region_width,
            projected_region_occupancy_height=region_height,
            projected_region_corner_coverage=region_coverage,
            projected_gt_center_coverage=center_coverage,
            feet_clearance=feet_clearance,
            eye_clearance=eye_clearance,
            abstention_reason=None,
        )

    # Choose the largest feasible scale (the smallest inward move) from a
    # deterministic coarse-to-fine grid. Coverage/safety gates remain hard.
    scales = np.linspace(1.0, _REFRAME_MIN_RADIAL_SCALE, _REFRAME_GRID_STEPS)
    for scale in scales:
        proposal = evaluate_scale(float(scale))
        if proposal is not None:
            return proposal
    return abstain("no_radial_candidate_meets_occupancy_coverage_and_clearance")


def _image_metrics(image: Image.Image, policy: QualityPolicy) -> tuple[
        float, float, float, float, float, float]:
    """Measure 8-bit sRGB luma statistics; these are not linear-light values."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    luminance = np.rint(
        0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    ).astype(np.uint8)
    histogram = np.bincount(luminance.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / float(luminance.size)
    nonzero = probabilities > 0.0
    entropy = float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))
    band_height = max(1, int(math.ceil(luminance.shape[0] * policy.top_band_fraction)))
    return (
        float(np.mean(luminance <= policy.near_black_level)),
        float(np.mean(luminance >= policy.clipped_level)),
        float(np.mean(luminance)),
        float(np.std(luminance)),
        entropy,
        float(np.mean(luminance[:band_height, :] <= policy.near_black_level)),
    )


def _frame_quality(
    scene_dir: Path,
    frame: Mapping[str, Any],
    camera_payload: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    policy: QualityPolicy,
    expected_resolution: Sequence[Any] | None = None,
) -> FrameQuality:
    frame_id = str(frame.get("frame_id", ""))
    filename = str(frame.get("filename", ""))
    issues: list[str] = []
    tags: list[str] = []
    promotion_reasons: list[str] = []
    width: int | None = None
    height: int | None = None
    near_black = clipped = mean = std = entropy = top_dark = None
    region_coverage = occ_width = occ_height = center_coverage = None
    gt_truncated: bool | None = None

    if not filename or Path(filename).name != filename or not filename.lower().endswith(".png"):
        issues.append("invalid_screenshot_filename")
        image_path = None
    else:
        image_path = scene_dir / "captures" / filename
    image: Image.Image | None = None
    if image_path is not None:
        try:
            with Image.open(image_path) as opened:
                opened.load()
                image = opened.convert("RGB")
            width, height = image.size
            near_black, clipped, mean, std, entropy, top_dark = _image_metrics(image, policy)
        except (OSError, ValueError) as error:
            issues.append(f"screenshot_decode_failed:{type(error).__name__}")

    actual = frame.get("actual")
    actual = actual if isinstance(actual, Mapping) else {}
    requested = actual.get("requested")
    requested = requested if isinstance(requested, Mapping) else {}
    if frame.get("captured") is not True:
        issues.append("capture_not_complete")
    errors = frame.get("errors", ())
    if errors:
        issues.append("capture_reported_errors")
    if actual.get("pose_converged") is not True:
        issues.append("camera_pose_not_converged")
    if actual.get("screen_open") is not False:
        issues.append("screen_state_not_closed")
    if width is not None and height is not None:
        expected_sizes: list[tuple[int, int]] = []
        for w_key, h_key, source in (
            ("framebuffer_width", "framebuffer_height", actual),
            ("window_width", "window_height", actual),
            ("window_width", "window_height", requested),
        ):
            if w_key in source and h_key in source:
                expected_sizes.append((int(source[w_key]), int(source[h_key])))
        if expected_resolution is not None and len(expected_resolution) == 2:
            expected_sizes.append((int(expected_resolution[0]), int(expected_resolution[1])))
        raw_intrinsics = camera_payload.get("intrinsics")
        if isinstance(raw_intrinsics, Mapping):
            raw_size = raw_intrinsics.get("image_size")
            if isinstance(raw_size, Sequence) and len(raw_size) == 2:
                expected_sizes.append((int(raw_size[0]), int(raw_size[1])))
        if any(size != (width, height) for size in expected_sizes):
            issues.append("screenshot_dimensions_mismatch")
    expected_hash = actual.get("screenshot_sha256")
    if image_path is not None and image_path.is_file() and isinstance(expected_hash, str):
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_hash:
            issues.append("screenshot_hash_mismatch")
    elif frame.get("captured") is True:
        issues.append("screenshot_hash_missing")

    hud_hidden = actual.get("hud_hidden") if isinstance(actual.get("hud_hidden"), bool) else None
    if hud_hidden is False:
        tags.append("hud_visible_ordinary_input")

    if image is not None and near_black is not None and clipped is not None:
        try:
            camera = CameraModel.from_dict(dict(camera_payload))
            image_width, image_height = image.size
            corners = _region_corners(ground_truth["region"])
            projected_corners = project_points(
                corners, camera, width=image_width, height=image_height)
            front_corners = [point for point in projected_corners if point is not None]
            in_frame_corners = [
                point for point in front_corners
                if 0.0 <= point[0] < image_width and 0.0 <= point[1] < image_height
            ]
            region_coverage = len(in_frame_corners) / 8.0

            centers = _occupied_block_centers(ground_truth)
            projected_centers = project_points(
                centers, camera, width=image_width, height=image_height)
            in_front_centers = [point for point in projected_centers if point is not None]
            in_frame_centers = [
                point for point in in_front_centers
                if 0.0 <= point[0] < image_width and 0.0 <= point[1] < image_height
            ]
            center_coverage = (len(in_frame_centers) / len(centers)) if centers else 0.0
            occ_width, occ_height, voxel_volume_truncated = (
                _projected_occupied_voxel_envelope(
                    centers, camera, width=image_width, height=image_height))
            gt_truncated = (
                len(in_frame_corners) != 8
                or len(in_frame_centers) != len(centers)
                or voxel_volume_truncated
            )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"evaluation_projection_failed:{type(error).__name__}")

    integrity_ok = not issues
    if near_black is not None and near_black >= policy.max_severe_near_black_fraction:
        tags.append("severe_near_black")
        promotion_reasons.append("severe_near_black")
    if ((std is not None and std < policy.min_luminance_std)
            or (entropy is not None and entropy < policy.min_luminance_entropy_bits)):
        tags.append("low_luminance_spread")
        promotion_reasons.append("low_luminance_spread")
    if clipped is not None and clipped > policy.max_clipped_fraction:
        tags.append("clipped_highlights")
        promotion_reasons.append("clipped_highlights")
    if top_dark is not None and top_dark >= policy.max_top_band_dark_fraction:
        tags.append("dark_upper_band")
        promotion_reasons.append("dark_upper_band")
    if (occ_width is None or occ_height is None
            or occ_width < policy.min_projected_occupancy_width
            or occ_height < policy.min_projected_occupancy_height):
        tags.append("small_projected_gt_occupancy")
        promotion_reasons.append("small_projected_gt_occupancy")
    if (region_coverage is None
            or region_coverage < policy.min_projected_region_corner_coverage
            or center_coverage is None
            or center_coverage < policy.min_projected_gt_center_coverage
            or gt_truncated is True):
        tags.append("projected_gt_truncated_or_coverage_below_floor")
        promotion_reasons.append("projected_gt_truncated_or_coverage_below_floor")
    if not integrity_ok:
        promotion_reasons.append("capture_integrity_failure")

    usable = integrity_ok and not promotion_reasons
    return FrameQuality(
        frame_id=frame_id,
        filename=filename,
        width=width,
        height=height,
        capture_integrity_ok=integrity_ok,
        integrity_issues=tuple(issues),
        near_black_fraction=near_black,
        clipped_fraction=clipped,
        luminance_mean=mean,
        luminance_std=std,
        luminance_entropy_bits=entropy,
        top_band_dark_fraction=top_dark,
        hud_hidden=hud_hidden,
        projected_region_corner_coverage=region_coverage,
        projected_gt_occupancy_width=occ_width,
        projected_gt_occupancy_height=occ_height,
        projected_gt_center_coverage=center_coverage,
        projected_gt_truncated=gt_truncated,
        adverse_tags=tuple(dict.fromkeys(tags)),
        promotion_usable=usable,
        promotion_reasons=tuple(dict.fromkeys(promotion_reasons)),
    )


def evaluate_capture_scene(
    scene_dir: str | Path,
    *,
    policy: QualityPolicy = DEFAULT_QUALITY_POLICY,
) -> CaptureQualityReport:
    """Evaluate one finalized scene directory without modifying its artifacts."""
    directory = Path(scene_dir)
    capture = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))
    frames = capture.get("frames", ())
    cameras = capture.get("cameras", ())
    resolutions = capture.get("resolutions", ())
    scene_issues: list[str] = []
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        frames = ()
        scene_issues.append("invalid_capture_frames")
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
        cameras = ()
        scene_issues.append("invalid_capture_cameras")
    if len(frames) != len(cameras):
        scene_issues.append("frame_camera_count_mismatch")
    capture_scene_id = capture.get("scene_id")
    ground_truth_scene_id = ground_truth.get("scene_id")
    if not isinstance(capture_scene_id, str) or not capture_scene_id:
        scene_issues.append("capture_scene_id_missing")
    if not isinstance(ground_truth_scene_id, str) or not ground_truth_scene_id:
        scene_issues.append("ground_truth_scene_id_missing")
    if capture_scene_id != ground_truth_scene_id:
        scene_issues.append("capture_ground_truth_scene_id_mismatch")
    if not _regions_match(capture.get("region"), ground_truth.get("region")):
        scene_issues.append("capture_ground_truth_region_mismatch")
    if not isinstance(resolutions, Sequence) or isinstance(resolutions, (str, bytes)):
        resolutions = ()
        scene_issues.append("invalid_capture_resolutions")
    elif resolutions and len(resolutions) != len(frames):
        scene_issues.append("frame_resolution_count_mismatch")
    frame_reports: list[FrameQuality] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or index >= len(cameras) or not isinstance(cameras[index], Mapping):
            invalid = FrameQuality(
                frame_id=str(getattr(frame, "get", lambda *_: "")("frame_id", "")),
                filename=str(getattr(frame, "get", lambda *_: "")("filename", "")),
                width=None, height=None, capture_integrity_ok=False,
                integrity_issues=("missing_frame_or_camera_metadata",),
                near_black_fraction=None, clipped_fraction=None, luminance_mean=None,
                luminance_std=None, luminance_entropy_bits=None, top_band_dark_fraction=None,
                hud_hidden=None, projected_region_corner_coverage=None,
                projected_gt_occupancy_width=None, projected_gt_occupancy_height=None,
                projected_gt_center_coverage=None, projected_gt_truncated=None,
                adverse_tags=("capture_integrity_failure",), promotion_usable=False,
                promotion_reasons=("capture_integrity_failure",),
            )
            frame_reports.append(invalid)
            continue
        frame_reports.append(_frame_quality(
            directory, frame, cameras[index], ground_truth, policy,
            expected_resolution=(resolutions[index] if index < len(resolutions) else None)))

    usable_count = sum(frame.promotion_usable for frame in frame_reports)
    integrity_ok = (bool(frame_reports) and not scene_issues
                    and all(frame.capture_integrity_ok for frame in frame_reports))
    eligible = integrity_ok and usable_count >= policy.min_usable_views
    reasons: list[str] = []
    if not integrity_ok:
        reasons.append("one or more captures failed artifact integrity checks")
    if scene_issues:
        reasons.append("scene artifact consistency failure: " + ", ".join(scene_issues))
    if usable_count < policy.min_usable_views:
        reasons.append(
            f"only {usable_count} of {len(frame_reports)} views meet promotion quality and framing policy; "
            f"at least {policy.min_usable_views} are required"
        )
    severe_frames = [frame.frame_id for frame in frame_reports
                     if "severe_near_black" in frame.adverse_tags]
    if severe_frames:
        reasons.append("severe near-black exposure: " + ", ".join(severe_frames))
    upper_dark_frames = [frame.frame_id for frame in frame_reports
                         if "dark_upper_band" in frame.adverse_tags]
    if upper_dark_frames:
        reasons.append("dark upper-frame band: " + ", ".join(upper_dark_frames))
    small_frames = [frame.frame_id for frame in frame_reports
                    if "small_projected_gt_occupancy" in frame.adverse_tags]
    if small_frames:
        reasons.append("projected GT envelope below occupancy floor: " + ", ".join(small_frames))
    adverse_tags = {
        "severe_near_black", "low_luminance_spread", "clipped_highlights",
        "dark_upper_band", "small_projected_gt_occupancy",
        "projected_gt_truncated_or_coverage_below_floor",
    }
    retained_adverse = integrity_ok and any(
        adverse_tags.intersection(frame.adverse_tags) for frame in frame_reports
    )
    return CaptureQualityReport(
        scene_id=str(capture.get("scene_id", ground_truth.get("scene_id", ""))),
        capture_integrity_ok=integrity_ok,
        scene_integrity_issues=tuple(scene_issues),
        promotion_eligible=eligible,
        retained_as_adverse_data=retained_adverse,
        usable_view_count=usable_count,
        required_usable_view_count=policy.min_usable_views,
        reasons=tuple(reasons),
        frames=tuple(frame_reports),
    )


def _regions_match(first: Any, second: Any) -> bool:
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return False
    try:
        first_low = tuple(float(value) for value in first["min"])
        second_low = tuple(float(value) for value in second["min"])
        first_high = tuple(float(value) for value in first["max"])
        second_high = tuple(float(value) for value in second["max"])
    except (KeyError, TypeError, ValueError):
        return False
    values = (*first_low, *second_low, *first_high, *second_high)
    return (
        len(first_low) == len(second_low) == len(first_high) == len(second_high) == 3
        and all(math.isfinite(value) for value in values)
        and first_low == second_low
        and first_high == second_high
    )


def derive_orbit_framing(
    region_dimensions: Sequence[float],
    *,
    fov_degrees: float,
    aspect_ratio: float,
    target_width_fraction: float = 0.55,
    target_height_fraction: float = 0.50,
    minimum_clearance: float = 2.0,
) -> OrbitFraming:
    """Derive a camera radius/elevation targeting the GT envelope occupancy.

    FOV is the project's vertical FOV. Horizontal and vertical projected
    extents use a conservative horizontal diagonal approximation. Invalid or
    unsatisfiable inputs raise ValueError so callers can abstain explicitly.
    """
    if len(region_dimensions) != 3:
        raise ValueError("region_dimensions must contain width, height, and depth")
    width, height, depth = (float(value) for value in region_dimensions)
    if not all(math.isfinite(value) and value > 0.0 for value in (width, height, depth)):
        raise ValueError("region dimensions must be finite and positive")
    if not math.isfinite(fov_degrees) or not 1.0 <= fov_degrees < 179.0:
        raise ValueError("vertical FOV must be finite and in [1, 179)")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be finite and positive")
    if not (0.0 < target_width_fraction < 1.0 and 0.0 < target_height_fraction < 1.0):
        raise ValueError("target occupancy fractions must be in (0, 1)")
    if not math.isfinite(minimum_clearance) or minimum_clearance < 0.0:
        raise ValueError("minimum_clearance must be finite and non-negative")

    vertical_half_fov = math.radians(fov_degrees) / 2.0
    horizontal_half_fov = math.atan(math.tan(vertical_half_fov) * aspect_ratio)
    horizontal_extent = math.hypot(width, depth)
    radius_for_width = horizontal_extent / (
        2.0 * target_width_fraction * math.tan(horizontal_half_fov))
    radius_for_height = height / (
        2.0 * target_height_fraction * math.tan(vertical_half_fov))
    clearance_radius = horizontal_extent / 2.0 + minimum_clearance
    radius = max(radius_for_width, radius_for_height, clearance_radius)
    tan_vertical = math.tan(vertical_half_fov)

    def projected_height_fraction(elevation: float) -> float:
        angle = math.radians(elevation)
        projected_extent = height * math.cos(angle) + horizontal_extent * math.sin(angle)
        return projected_extent * math.cos(angle) / (2.0 * radius * tan_vertical)

    elevation = 0.0
    if projected_height_fraction(0.0) < target_height_fraction:
        previous = 0.0
        found = False
        for step in range(1, 601):
            current = step * 0.1
            if projected_height_fraction(current) >= target_height_fraction:
                low, high = previous, current
                for _ in range(32):
                    middle = (low + high) / 2.0
                    if projected_height_fraction(middle) >= target_height_fraction:
                        high = middle
                    else:
                        low = middle
                elevation = high
                found = True
                break
            previous = current
        if not found:
            raise ValueError("target height occupancy is infeasible at the required radius")

    width_fraction = horizontal_extent / (
        2.0 * radius * math.tan(horizontal_half_fov))
    height_fraction = projected_height_fraction(elevation)
    return OrbitFraming(
        horizontal_radius=radius,
        elevation_degrees=elevation,
        predicted_width_fraction=width_fraction,
        predicted_height_fraction=height_fraction,
        horizontal_clearance=radius - horizontal_extent / 2.0,
    )
