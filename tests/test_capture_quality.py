from __future__ import annotations

import hashlib
from itertools import product
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

from mc_photogrammetry.dataset.plan import (
    _fit_orbit_at_elevation,
    bounds_projected_occupancy,
    yaw_pitch_to_target,
)
from mc_photogrammetry.dataset.framing_contract import (
    MIN_PROJECTED_OCCUPANCY_HEIGHT,
    MIN_PROJECTED_OCCUPANCY_WIDTH,
)
from mc_photogrammetry.dataset.quality import (
    QualityPolicy,
    _occupied_block_centers,
    _projected_occupied_voxel_envelope,
    derive_orbit_framing,
    evaluate_capture_scene,
    propose_evaluation_only_reframe,
)
from mc_photogrammetry.dataset.schemas import RegionBounds
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    intrinsics_for,
    project_points,
)


def test_planner_uses_frozen_quality_occupancy_floors():
    policy = QualityPolicy()
    assert policy.min_projected_occupancy_width == MIN_PROJECTED_OCCUPANCY_WIDTH == 0.30
    assert policy.min_projected_occupancy_height == MIN_PROJECTED_OCCUPANCY_HEIGHT == 0.30


def _anisotropic_quality_fixture():
    bounds = RegionBounds((0, 0, 0), (16, 22, 15))  # 17 x 23 x 16 voxels
    resolution = (640, 480)
    camera = _fit_orbit_at_elevation(
        aim=(8.5, 11.5, 8.0),
        bounds=bounds,
        angle=0.4,
        elevation=0.1,
        fov_degrees=70.0,
        resolution=resolution,
        preferred_radius=bounds.diagonal * 0.9,
        scene_id="synthetic-anisotropic-quality",
        view_index=0,
    )
    return bounds, camera, resolution


def test_quality_measures_anisotropic_occupied_voxel_support_with_fixed_settings(tmp_path):
    bounds, camera, resolution = _anisotropic_quality_fixture()
    positions = tuple(product((0, 16), (0, 22), (0, 15)))
    ground_truth = {
        "blocks": [{"name": "minecraft:stone", "pos": list(pos)}
                   for pos in positions]
    }
    centers = _occupied_block_centers(ground_truth)
    center_pixels = [
        point for point in project_points(
            centers, camera, width=resolution[0], height=resolution[1])
        if point is not None
    ]
    center_width = ((max(point[0] for point in center_pixels)
                     - min(point[0] for point in center_pixels)) / resolution[0])
    planned_occupancy = bounds_projected_occupancy(camera, bounds, resolution)
    assert planned_occupancy is not None and center_width < 0.30

    fixed_settings = {
        "gamma": 1.0,
        "time_of_day": 6000,
        "weather": "clear",
        "hud_hidden": True,
        "resource_pack": None,
        "shader_pack": None,
    }
    scene = tmp_path / "anisotropic"
    _write_synthetic_scene(
        scene,
        view_count=1,
        resolution=resolution,
        region={"min": list(bounds.minimum), "max": list(bounds.maximum)},
        block_positions=positions,
        camera=camera,
        settings=fixed_settings,
    )
    report = evaluate_capture_scene(scene, policy=QualityPolicy(min_usable_views=1))
    frame = report.frames[0]
    assert report.capture_integrity_ok
    assert frame.hud_hidden is True
    assert frame.projected_gt_center_coverage == 1.0
    assert frame.projected_region_corner_coverage == 1.0
    assert frame.projected_gt_truncated is False
    assert frame.projected_gt_occupancy_width == pytest.approx(planned_occupancy[0])
    assert frame.projected_gt_occupancy_height == pytest.approx(planned_occupancy[1])
    assert frame.projected_gt_occupancy_width >= 0.30
    assert frame.projected_gt_occupancy_height >= 0.30
    assert frame.promotion_usable


def test_sparse_gt_uses_only_occupied_voxel_corners_not_region_corners():
    bounds, camera, resolution = _anisotropic_quality_fixture()
    region_occupancy = bounds_projected_occupancy(camera, bounds, resolution)
    assert region_occupancy is not None and region_occupancy[0] >= 0.30

    # A lone edge voxel is not evidence that the rest of the declared region is
    # occupied. Its physical block support remains small even though the larger
    # requested region has full projected occupancy.
    sparse_centers = [(0.5, 0.5, 0.5)]
    width, height, truncated = _projected_occupied_voxel_envelope(
        sparse_centers, camera, width=resolution[0], height=resolution[1])
    assert width is not None and height is not None
    assert width < 0.30
    assert height < 0.30
    assert not truncated


def test_projected_occupied_voxel_corner_outside_frame_marks_truncation():
    camera = CameraModel(
        position=(0.0, 0.0, 0.0), yaw_degrees=0.0, pitch_degrees=0.0,
        fov_degrees=70.0, eye_height=0.0,
    )
    center = (-9.5, 1.5, 10.5)
    center_pixel = project_points([center], camera, width=640, height=480)[0]
    assert center_pixel is not None and 0 <= center_pixel[0] < 640
    _width, _height, truncated = _projected_occupied_voxel_envelope(
        [center], camera, width=640, height=480)
    assert truncated


def test_dense_occupied_voxel_envelope_matches_planner_region_projection():
    bounds, camera, resolution = _anisotropic_quality_fixture()
    dense_centers = [
        (x + 0.5, y + 0.5, z + 0.5)
        for x in range(bounds.minimum[0], bounds.maximum[0] + 1)
        for y in range(bounds.minimum[1], bounds.maximum[1] + 1)
        for z in range(bounds.minimum[2], bounds.maximum[2] + 1)
    ]
    voxel_width, voxel_height, truncated = _projected_occupied_voxel_envelope(
        dense_centers, camera, width=resolution[0], height=resolution[1])
    planner_occupancy = bounds_projected_occupancy(camera, bounds, resolution)
    assert planner_occupancy is not None
    assert voxel_width == pytest.approx(planner_occupancy[0])
    assert voxel_height == pytest.approx(planner_occupancy[1])
    assert not truncated


def test_evaluation_reframe_proposal_meets_occupied_and_region_gates_deterministically():
    bounds, camera, resolution = _anisotropic_quality_fixture()
    # A depth-truncated but vertically complete facade sits just below the
    # occupied-width floor, while the requested continuous region has margin.
    positions = tuple(
        (x, y, z)
        for x in range(bounds.minimum[0], bounds.maximum[0] + 1)
        for y in range(bounds.minimum[1], bounds.maximum[1] + 1)
        for z in range(2, 14)
    )
    ground_truth = {
        "region": {"min": list(bounds.minimum), "max": list(bounds.maximum)},
        "blocks": [{"name": "minecraft:stone", "pos": list(pos)}
                   for pos in positions],
    }
    current_width, current_height, _ = _projected_occupied_voxel_envelope(
        _occupied_block_centers(ground_truth), camera,
        width=resolution[0], height=resolution[1])
    assert current_width is not None and current_width < 0.30

    proposal = propose_evaluation_only_reframe(
        camera, ground_truth, width=resolution[0], height=resolution[1])
    repeat = propose_evaluation_only_reframe(
        camera, ground_truth, width=resolution[0], height=resolution[1])
    assert proposal.feasible and proposal.camera is not None
    assert proposal == repeat
    assert proposal.radial_scale is not None and proposal.radial_scale < 1.0
    assert proposal.projected_gt_occupancy_width is not None
    assert proposal.projected_gt_occupancy_height is not None
    assert proposal.projected_gt_occupancy_width >= 0.30
    assert proposal.projected_gt_occupancy_height >= 0.30
    assert proposal.projected_region_occupancy_width is not None
    assert proposal.projected_region_occupancy_height is not None
    assert max(proposal.projected_region_occupancy_width,
               proposal.projected_region_occupancy_height) <= 0.85
    assert proposal.projected_region_corner_coverage == 1.0
    assert proposal.projected_gt_center_coverage is not None
    assert proposal.projected_gt_center_coverage >= 0.95
    assert proposal.feet_clearance is not None and proposal.feet_clearance >= 2.0
    assert proposal.eye_clearance is not None and proposal.eye_clearance >= 2.0
    proposed_camera = proposal.camera
    assert proposed_camera.yaw_degrees == camera.yaw_degrees
    assert proposed_camera.pitch_degrees == camera.pitch_degrees
    assert proposed_camera.fov_degrees == camera.fov_degrees
    assert proposed_camera.eye_height == camera.eye_height
    assert proposed_camera.intrinsics == camera.intrinsics
    assert proposed_camera.roll_degrees == camera.roll_degrees
    assert proposal.projected_gt_occupancy_height != pytest.approx(current_height)


def test_evaluation_reframe_returns_typed_abstention_for_noncentered_input():
    bounds, camera, resolution = _anisotropic_quality_fixture()
    gt = {
        "region": {"min": list(bounds.minimum), "max": list(bounds.maximum)},
        "blocks": [{"name": "minecraft:stone", "pos": [0, 0, 0]}],
    }
    noncentered = CameraModel(
        position=camera.position,
        yaw_degrees=camera.yaw_degrees + 12.0,
        pitch_degrees=camera.pitch_degrees,
        fov_degrees=camera.fov_degrees,
        eye_height=camera.eye_height,
        intrinsics=camera.intrinsics,
        roll_degrees=camera.roll_degrees,
    )
    proposal = propose_evaluation_only_reframe(
        noncentered, gt, width=resolution[0], height=resolution[1])
    assert not proposal.feasible
    assert proposal.camera is None
    assert proposal.abstention_reason == "camera_not_aimed_at_region_center"

    impossible_gt = {
        "region": gt["region"],
        "blocks": [{"name": "minecraft:stone", "pos": [0, 0, 0]}],
    }
    infeasible = propose_evaluation_only_reframe(
        camera, impossible_gt, width=resolution[0], height=resolution[1])
    assert not infeasible.feasible
    assert infeasible.camera is None
    assert infeasible.abstention_reason == (
        "no_radial_candidate_meets_occupancy_coverage_and_clearance")


def test_evaluation_reframe_is_poisoned_from_normal_cli_and_reconstruction():
    source_root = Path(__file__).resolve().parents[1] / "src" / "mc_photogrammetry"
    forbidden_name = "propose_evaluation_only_reframe"
    protected = [source_root / "cli.py", source_root / "reconstruction",
                 source_root / "pipeline"]
    for path in protected:
        files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for source in files:
            text = source.read_text(encoding="utf-8")
            assert forbidden_name not in text, source
            assert "mc_photogrammetry.dataset.quality" not in text, source


def _synthetic_image(width: int, height: int, *, dark: bool = False,
                     dark_top_band: bool = False, saturated: bool = False,
                     flat_value: int | None = None) -> np.ndarray:
    if dark:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if saturated:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    if flat_value is not None:
        return np.full((height, width, 3), flat_value, dtype=np.uint8)
    y, x = np.indices((height, width))
    value = (35 + (x * 13 + y * 9) % 190).astype(np.uint8)
    image = np.repeat(value[..., None], 3, axis=2)
    if dark_top_band:
        image[:max(1, height // 10), :, :] = 0
    return image


def _write_synthetic_scene(
    directory: Path,
    *,
    view_count: int = 4,
    dark_views: tuple[int, ...] = (),
    dark_band_views: tuple[int, ...] = (),
    saturated_views: tuple[int, ...] = (),
    flat_views: tuple[int, ...] = (),
    camera_distance: float = 16.0,
    camera_height: float = 10.0,
    resolution: tuple[int, int] = (640, 480),
    region: dict[str, Any] | None = None,
    block_positions: tuple[tuple[int, int, int], ...] | None = None,
    camera: CameraModel | None = None,
    settings: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "captures").mkdir()
    width, height = resolution
    region = region or {"min": [-5, -4, -5], "max": [4, 3, 4]}
    target = (0.0, 0.0, 0.0)
    if camera is None:
        position = (0.0, camera_height, -camera_distance)
        yaw, pitch = yaw_pitch_to_target(
            (position[0], position[1] + 1.62, position[2]), target)
        base_camera = CameraModel(
            position=position,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            fov_degrees=70.0,
            eye_height=1.62,
        )
        camera = CameraModel(
            position=base_camera.position,
            yaw_degrees=base_camera.yaw_degrees,
            pitch_degrees=base_camera.pitch_degrees,
            fov_degrees=base_camera.fov_degrees,
            eye_height=base_camera.eye_height,
            intrinsics=intrinsics_for(base_camera, width, height),
        )
    frames = []
    for index in range(view_count):
        frame_id = f"synthetic-{index:02d}"
        filename = f"{frame_id}.png"
        array = _synthetic_image(
            width,
            height,
            dark=index in dark_views,
            dark_top_band=index in dark_band_views,
            saturated=index in saturated_views,
            flat_value=(64 if index in flat_views else None),
        )
        path = directory / "captures" / filename
        Image.fromarray(array, mode="RGB").save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_settings = {
            "hud_hidden": False,
            "gamma": [0.0, 0.25, 0.5, 0.75, 1.0][index % 5],
            "time_of_day": [6000, 12000, 18000, 0, 6000][index % 5],
            "weather": "clear",
        }
        if settings is not None:
            actual_settings.update(settings)
        frames.append({
            "frame_id": frame_id,
            "filename": filename,
            "captured": True,
            "errors": [],
            "actual": {
                "framebuffer_width": width,
                "framebuffer_height": height,
                "window_width": width,
                "window_height": height,
                "pose_converged": True,
                "screen_open": False,
                "screenshot_sha256": digest,
                "requested": {"window_width": width, "window_height": height},
                **actual_settings,
            },
        })
    capture = {
        "scene_id": "synthetic-scene",
        "region": region,
        "frames": frames,
        "cameras": [camera.to_dict() for _ in frames],
        "resolutions": [[width, height] for _ in frames],
    }
    if block_positions is None:
        block_positions = tuple(
            (x, y, z)
            for x in range(-5, 5)
            for y in range(-4, 4)
            for z in range(-5, 5)
        )
    blocks = [{"name": "minecraft:stone", "pos": list(position)}
              for position in block_positions]
    ground_truth = {
        "scene_id": "synthetic-scene",
        "region": region,
        "blocks": blocks,
    }
    (directory / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (directory / "ground_truth.json").write_text(
        json.dumps(ground_truth), encoding="utf-8")


def test_synthetic_image_metrics_measure_dark_band_and_clipping(tmp_path):
    _write_synthetic_scene(tmp_path / "scene", view_count=1, dark_band_views=(0,))
    report = evaluate_capture_scene(tmp_path / "scene")
    frame = report.frames[0]
    assert frame.capture_integrity_ok
    assert frame.near_black_fraction == pytest.approx(0.10, abs=0.003)
    assert frame.top_band_dark_fraction == 1.0
    assert frame.clipped_fraction == 0.0
    assert frame.luminance_std is not None and frame.luminance_std > 8.0
    assert frame.luminance_entropy_bits is not None and frame.luminance_entropy_bits > 4.0
    assert "dark_upper_band" in frame.adverse_tags
    assert frame.hud_hidden is False
    assert "hud_visible_ordinary_input" in frame.adverse_tags


def test_saturated_bright_control_triggers_clipping_policy(tmp_path):
    _write_synthetic_scene(tmp_path / "scene", view_count=1, saturated_views=(0,))
    report = evaluate_capture_scene(tmp_path / "scene")
    frame = report.frames[0]
    assert frame.capture_integrity_ok
    assert frame.near_black_fraction == 0.0
    assert frame.clipped_fraction == 1.0
    assert "clipped_highlights" in frame.adverse_tags


def test_nonblack_low_contrast_control_triggers_spread_policy(tmp_path):
    _write_synthetic_scene(tmp_path / "scene", view_count=1, flat_views=(0,))
    frame = evaluate_capture_scene(tmp_path / "scene").frames[0]
    assert frame.capture_integrity_ok
    assert frame.near_black_fraction == 0.0
    assert frame.luminance_std == 0.0
    assert frame.luminance_entropy_bits == 0.0
    assert "low_luminance_spread" in frame.adverse_tags
    assert "severe_near_black" not in frame.adverse_tags


def test_hud_visible_frames_remain_valid_and_gamma_variants_can_coexist(tmp_path):
    _write_synthetic_scene(tmp_path / "scene", view_count=4)
    report = evaluate_capture_scene(tmp_path / "scene")
    assert report.capture_integrity_ok
    assert report.promotion_eligible
    assert report.usable_view_count == 4
    assert all(frame.capture_integrity_ok for frame in report.frames)
    assert all(frame.hud_hidden is False for frame in report.frames)
    assert all(frame.promotion_usable for frame in report.frames)
    assert not report.retained_as_adverse_data


def test_clean_but_insufficient_view_count_is_not_adverse_content(tmp_path):
    _write_synthetic_scene(tmp_path / "scene", view_count=3)
    report = evaluate_capture_scene(tmp_path / "scene")
    assert report.capture_integrity_ok
    assert not report.promotion_eligible
    assert report.usable_view_count == 3
    assert not report.retained_as_adverse_data


def test_one_dark_variant_is_retained_when_four_views_remain_usable(tmp_path):
    _write_synthetic_scene(tmp_path / "scene", view_count=5, dark_views=(2,))
    report = evaluate_capture_scene(tmp_path / "scene")
    assert report.capture_integrity_ok
    assert report.promotion_eligible
    assert report.usable_view_count == 4
    assert report.retained_as_adverse_data
    assert "severe_near_black" in report.frames[2].adverse_tags
    assert not report.frames[2].promotion_usable


def test_frame_camera_cardinality_mismatch_breaks_scene_integrity(tmp_path):
    scene = tmp_path / "scene"
    _write_synthetic_scene(scene, view_count=4)
    capture_path = scene / "capture.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["cameras"].pop()
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    report = evaluate_capture_scene(scene)
    assert not report.capture_integrity_ok
    assert not report.promotion_eligible
    assert not report.retained_as_adverse_data
    assert "frame_camera_count_mismatch" in report.scene_integrity_issues


def test_actual_window_dimensions_are_checked_independently_of_framebuffer(tmp_path):
    scene = tmp_path / "scene"
    _write_synthetic_scene(scene, view_count=4)
    capture_path = scene / "capture.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["frames"][0]["actual"]["window_width"] = 320
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    report = evaluate_capture_scene(scene)
    assert not report.capture_integrity_ok
    assert "screenshot_dimensions_mismatch" in report.frames[0].integrity_issues


@pytest.mark.parametrize("mismatch", ["scene_id", "region"])
def test_capture_ground_truth_scene_and_region_must_match(tmp_path, mismatch):
    scene = tmp_path / "scene"
    _write_synthetic_scene(scene, view_count=4)
    ground_truth_path = scene / "ground_truth.json"
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    if mismatch == "scene_id":
        ground_truth["scene_id"] = "other-scene"
    else:
        ground_truth["region"]["max"][0] += 1
    ground_truth_path.write_text(json.dumps(ground_truth), encoding="utf-8")
    report = evaluate_capture_scene(scene)
    assert not report.capture_integrity_ok
    assert not report.promotion_eligible
    expected = ("capture_ground_truth_scene_id_mismatch" if mismatch == "scene_id"
                else "capture_ground_truth_region_mismatch")
    assert expected in report.scene_integrity_issues


def test_projected_gt_truncation_cannot_pass_by_occupancy_alone(tmp_path):
    _write_synthetic_scene(
        tmp_path / "scene", view_count=1, camera_distance=3.0, camera_height=0.0)
    policy = QualityPolicy(min_usable_views=1)
    report = evaluate_capture_scene(tmp_path / "scene", policy=policy)
    frame = report.frames[0]
    assert frame.projected_gt_truncated
    assert frame.projected_gt_center_coverage is not None
    assert frame.projected_gt_center_coverage < 0.95
    assert "projected_gt_truncated_or_coverage_below_floor" in frame.promotion_reasons
    assert not report.promotion_eligible


def test_projection_reports_gt_occupancy_and_framing_targets_envelope():
    framing = derive_orbit_framing(
        (64, 32, 64),
        fov_degrees=90.0,
        aspect_ratio=16 / 9,
        target_width_fraction=0.55,
        target_height_fraction=0.50,
        minimum_clearance=2.0,
    )
    assert framing.horizontal_radius > math.hypot(64, 64) / 2.0
    assert framing.horizontal_clearance >= 2.0
    assert framing.predicted_width_fraction <= 0.55 + 1e-9
    assert framing.predicted_width_fraction >= 0.50
    assert framing.predicted_height_fraction == pytest.approx(0.50, abs=1e-5)
    assert 0.0 <= framing.elevation_degrees <= 60.0


@pytest.mark.parametrize(
    "dimensions,fov,aspect",
    [
        ((0, 2, 3), 70.0, 16 / 9),
        ((2, float("nan"), 3), 70.0, 16 / 9),
        ((2, 3, 4), 0.0, 16 / 9),
        ((2, 3, 4), 180.0, 16 / 9),
        ((2, 3, 4), 70.0, 0.0),
    ],
)
def test_framing_abstains_for_invalid_region_or_fov(dimensions, fov, aspect):
    with pytest.raises(ValueError):
        derive_orbit_framing(dimensions, fov_degrees=fov, aspect_ratio=aspect)


def test_quality_contracts_are_exported_from_dataset_package():
    from mc_photogrammetry import dataset

    assert dataset.QualityPolicy is QualityPolicy
    assert dataset.evaluate_capture_scene is evaluate_capture_scene
    assert dataset.derive_orbit_framing is derive_orbit_framing
    assert dataset.DEFAULT_QUALITY_POLICY.min_usable_views == 4


def test_train_station_artifact_is_adverse_not_a_building_promotion_holdout():
    scene = (Path(__file__).resolve().parents[1]
             / "exports/datasets/pilot-batch-train-station-2026-09-12/scenes"
             / "train-station-spawn-208-96-orbit-a")
    if not (scene / "capture.json").is_file():
        pytest.skip("read-only Train Station capture artifact is unavailable")
    report = evaluate_capture_scene(scene)
    assert report.capture_integrity_ok
    assert not report.promotion_eligible
    assert report.retained_as_adverse_data
    assert len(report.frames) == 6
    assert report.usable_view_count < report.required_usable_view_count
    assert report.frames[2].near_black_fraction is not None
    assert report.frames[2].near_black_fraction > 0.90
    assert report.frames[2].top_band_dark_fraction == pytest.approx(1.0, abs=0.002)
    assert all(frame.projected_gt_truncated is False for frame in report.frames)
    assert all(frame.projected_region_corner_coverage == 1.0 for frame in report.frames)
    assert [frame.frame_id for frame in report.frames if "hud_visible_ordinary_input"
            in frame.adverse_tags] == [
                "view-train-station-spawn-208-96-orbit-a-00",
                "view-train-station-spawn-208-96-orbit-a-01",
                "view-train-station-spawn-208-96-orbit-a-03",
            ]
