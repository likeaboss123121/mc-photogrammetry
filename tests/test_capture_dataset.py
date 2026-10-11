"""Contract tests for the deterministic real-Minecraft capture dataset factory.

Everything here exercises public outputs: the JSON a plan freezes to, the plan a
mod would replay, and the files a finalizer writes. No test reaches into an
implementation detail, and no test needs Minecraft, a live ``.minecraft``, or a
network connection.
"""
from __future__ import annotations

import base64
import json
import math
import struct
import subprocess
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

from mc_photogrammetry.dataset import (
    BlockRecord,
    CameraCapture,
    CapturePlan,
    CaptureFrameResult,
    CaptureSettings,
    DatasetSchemaError,
    EYE_POSITION_TOLERANCE,
    GAMMA_MAX,
    GAMMA_MIN,
    GAMMA_TOLERANCE,
    GroundTruthExport,
    MAX_VIEWS,
    MIN_VIEWS,
    QualityPolicy,
    RegionBounds,
    RegionRequest,
    SceneRecipe,
    StagingLimitError,
    StagingLimits,
    TIME_OF_DAY_CHOICES,
    angle_delta_degrees,
    build_plan,
    bounds_coverage,
    bounds_projected_occupancy,
    camera_sees_bounds,
    copy_world_to_staging,
    finalize_scene,
    ground_truth_for_volume,
    inventory_saves,
    landscape_region,
    load_region_requests,
    mod_output_dir,
    parse_ground_truth,
    plan_id_for,
    png_dimensions,
    safe_join,
    scene_dir,
    scene_group,
    split_for_scene,
    split_report,
    spawn_position_from_save,
    stage_world,
    staging_fingerprint,
    staging_marker_payload,
    staging_token_for,
    target_pixel,
    validate_plan,
    validate_pose,
    validate_settings,
    view_setting_strata,
    volume_from_ground_truth,
    volume_topology,
    world_marker_path,
    write_dataset_index,
    write_reference_litematic,
    write_scene_plan,
    write_scene_recipe,
)
from mc_photogrammetry.dataset.hashing import sha256_text
from mc_photogrammetry.dataset import plan as plan_module
from mc_photogrammetry.dataset.plan import yaw_pitch_to_target
from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.export.compare import load_litematic
from mc_photogrammetry.reconstruction.camera import camera_eye

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
    "AAAABJRU5ErkJggg==")

ROOT = Path(__file__).resolve().parents[1]

#: A valid but synthetic staged-world fingerprint for plans built without a world.
TEST_STAGING_TOKEN = sha256_text("test-staged-world")

_PNG_CACHE: dict[tuple[int, int], bytes] = {}


def _png(width: int, height: int) -> bytes:
    """A real (if solid) PNG of the requested pixel size, for finalizer fixtures.

    The finalizer checks the IHDR dimensions of every screenshot against the plan,
    so a 1x1 placeholder can no longer stand in for a 1920x1080 capture.
    """
    cached = _PNG_CACHE.get((width, height))
    if cached is not None:
        return cached

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grey
    row = b"\x00" + b"\x00" * width  # filter byte, then one byte per grey pixel
    raw = zlib.compress(row * height, 6)
    payload = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
               + chunk(b"IDAT", raw) + chunk(b"IEND", b""))
    _PNG_CACHE[(width, height)] = payload
    return payload


def _region() -> RegionBounds:
    return RegionBounds((0, 60, 0), (24, 84, 24))


def _plan(seed: int = 11, region: RegionBounds | None = None, views: int = 5) -> CapturePlan:
    region = region or _region()
    return build_plan(dataset_id="ds-test", scene_id="house-a", world_identity="Template 2",
                      region=region, seed=seed, view_count=views,
                      staging_token=TEST_STAGING_TOKEN)


def _volume() -> BlockVolume:
    volume = BlockVolume()
    volume.set((1, 61, 1), BlockObservation(BlockState("minecraft:stone")))
    volume.set((2, 61, 1), BlockObservation(
        BlockState("minecraft:oak_stairs", (("facing", "west"), ("half", "top")))))
    volume.set((3, 61, 1), BlockObservation(BlockState("minecraft:oak_log", (("axis", "x"),))))
    return volume


def _sha256(path: Path) -> str:
    from mc_photogrammetry.dataset.hashing import sha256_file

    return sha256_file(path)


def _mod_actual(plan: CapturePlan, job: CameraCapture, image: Path) -> dict:
    """The exact ``actual`` object the Fabric mod writes for a successful frame.

    Request and observation are separate: ``actual.requested`` is what the plan
    asked the client and world for, and the flat keys are what the client and the
    server reported afterwards.
    """
    eye = camera_eye(job.camera)
    requested_fov = int(round(float(job.camera.fov_degrees)))
    requested = {
        "fov_degrees": requested_fov,
        "gamma": job.settings.gamma,
        "hud_hidden": job.settings.hud_hidden,
        "window_width": job.resolution[0],
        "window_height": job.resolution[1],
        "framebuffer_width": job.resolution[0],
        "framebuffer_height": job.resolution[1],
        "time_of_day": job.settings.time_of_day,
        "weather": job.settings.weather,
        "resource_pack": job.settings.resource_pack,
        "shader_pack": job.settings.shader_pack,
    }
    return {
        "feet_position": list(job.camera.position),
        "eye_position": [float(value) for value in eye],
        "yaw_degrees": job.camera.yaw_degrees,
        "pitch_degrees": job.camera.pitch_degrees,
        "fov_degrees": requested_fov,
        "gamma": job.settings.gamma,
        "hud_hidden": job.settings.hud_hidden,
        "window_width": job.resolution[0],
        "window_height": job.resolution[1],
        "framebuffer_width": job.resolution[0],
        "framebuffer_height": job.resolution[1],
        "time_of_day": job.settings.time_of_day,
        "raining": job.settings.weather in ("rain", "thunder"),
        "thundering": job.settings.weather == "thunder",
        "weather": job.settings.weather,
        "daylight_cycle_frozen": True,
        "screenshot_sha256": _sha256(image),
        "staging_token": plan.staging_token,
        "requested": requested,
        "active_resource_packs": [],
        "resource_pack_applied": False,
        "shader_pack_applied": False,
    }


def _mod_result(plan: CapturePlan, job: CameraCapture, image: Path) -> CaptureFrameResult:
    return CaptureFrameResult(
        frame_id=job.frame_id,
        filename=job.filename,
        captured=True,
        screenshot=f"screenshots/{job.filename}",
        settled_ticks=job.settle_ticks,
        actual=_mod_actual(plan, job, image),
    )


def _write_mod_outputs(staging: Path, plan: CapturePlan,
                       export: GroundTruthExport) -> None:
    """Write exactly the files the Fabric mod contract promises for one plan."""
    screenshots = staging / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    results = []
    for job in plan.jobs:
        image = screenshots / job.filename
        image.write_bytes(_png(*job.resolution))
        results.append(_mod_result(plan, job, image).to_dict())
    out = mod_output_dir(staging, plan.plan_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "ground_truth.json").write_text(json.dumps(export.to_dict(), indent=2),
                                           encoding="utf-8")


def _legacy_level_dat(path: Path, *, level_name: str, spawn_x: int = 3, spawn_z: int = -4,
                      spawn_y: int | None = None) -> None:
    import nbtlib
    from nbtlib import tag

    data = {
        "LevelName": tag.String(level_name),
        "LastPlayed": tag.Long(1_700_000_000_000),
        "SpawnX": tag.Int(spawn_x),
        "SpawnZ": tag.Int(spawn_z),
    }
    # Real saves always carry SpawnY; omitting it here is how the tests build the
    # missing-spawn case the landscape planner must refuse.
    if spawn_y is not None:
        data["SpawnY"] = tag.Int(spawn_y)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbtlib.File(tag.Compound({"Data": tag.Compound(data)}), gzipped=True).save(path)


# --- schemas, determinism, versioning -------------------------------------


def test_plan_round_trips_and_is_byte_stable_from_the_same_inputs():
    first = _plan().to_dict()
    second = _plan().to_dict()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert CapturePlan.from_dict(json.loads(json.dumps(first))).to_dict() == first
    assert "created" not in json.dumps(first).lower()


def test_plan_is_identical_in_a_fresh_process():
    script = (
        "import sys, json; sys.path.insert(0, r'%s');"
        "from mc_photogrammetry.dataset import build_plan, RegionBounds;"
        "p=build_plan(dataset_id='ds-test', scene_id='house-a', world_identity='Template 2',"
        " region=RegionBounds((0,60,0),(24,84,24)), seed=11, view_count=5,"
        " staging_token=%r);"
        "print(json.dumps(p.to_dict(), sort_keys=True))"
        % (ROOT / "src", TEST_STAGING_TOKEN)
    )
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == _plan().to_dict()


def test_plan_must_carry_a_staged_world_token_and_rejects_the_inert_one():
    plan = _plan()
    assert plan.staging_token == TEST_STAGING_TOKEN
    payload = plan.to_dict()
    payload["staging_token"] = ""
    with pytest.raises(DatasetSchemaError, match="staging_token"):
        CapturePlan.from_dict(payload)
    payload["staging_token"] = "not-a-digest"
    with pytest.raises(DatasetSchemaError, match="hex|64-character"):
        CapturePlan.from_dict(payload)
    # The inert all-zero placeholder can no longer be produced by the builder at
    # all: build_plan has no default token. The Python schema still only enforces a
    # 64-character hex digest, so the inert value is refused where it matters - the
    # Fabric mod's PlanAudit / StagingMarker - and the builder never emits it.
    with pytest.raises(TypeError):
        build_plan(dataset_id="ds-test", scene_id="house-a",
                   world_identity="Template 2", region=_region(), seed=11, view_count=4)
    with pytest.raises(DatasetSchemaError, match="explicit staging_token"):
        build_plan(dataset_id="ds-test", scene_id="house-a", world_identity="Template 2",
                   region=_region(), seed=11, view_count=4, staging_token="   ")
    assert not hasattr(__import__("mc_photogrammetry.dataset", fromlist=["x"]),
                       "unstaged_token")


def test_unknown_schema_version_is_rejected_with_a_clear_error():
    payload = _plan().to_dict()
    payload["schema_version"] = 99
    with pytest.raises(DatasetSchemaError, match="unsupported schema_version"):
        CapturePlan.from_dict(payload)
    recipe = SceneRecipe(seed=1, family="house", dimensions=(4, 4, 4)).to_dict()
    recipe["schema_version"] = 2
    with pytest.raises(DatasetSchemaError, match="unsupported schema"):
        SceneRecipe.from_dict(recipe)


def test_plan_caps_jobs_and_requires_safe_unique_frame_ids():
    plan = _plan()
    too_many = plan.to_dict()
    too_many["jobs"] = [plan.jobs[0].to_dict()] * 65
    with pytest.raises(DatasetSchemaError, match="exceed"):
        CapturePlan.from_dict(too_many)

    duplicated = plan.to_dict()
    duplicated["jobs"] = [plan.jobs[0].to_dict(), plan.jobs[0].to_dict()]
    with pytest.raises(DatasetSchemaError, match="unique"):
        CapturePlan.from_dict(duplicated)

    unsafe = plan.jobs[0].to_dict()
    unsafe["frame_id"] = "view/../../escape"
    bad = plan.to_dict()
    bad["jobs"] = [unsafe]
    with pytest.raises(DatasetSchemaError, match="filename-safe"):
        CapturePlan.from_dict(bad)


def test_capture_settings_validate_declared_weather_time_and_gamma():
    with pytest.raises(DatasetSchemaError, match="weather"):
        CaptureSettings(weather="snow")
    with pytest.raises(DatasetSchemaError, match="time_of_day"):
        CaptureSettings(time_of_day=24000)
    with pytest.raises(DatasetSchemaError, match="gamma"):
        CaptureSettings(gamma=5.0)


# --- deterministic setting variation ---------------------------------------


def test_every_allowed_view_count_covers_gamma_time_and_hud():
    """Coverage is a property of the view count, not a lucky draw."""
    for views in range(MIN_VIEWS, MAX_VIEWS + 1):
        plan = _plan(views=views)
        gammas = [job.settings.gamma for job in plan.jobs]
        times = {job.settings.time_of_day for job in plan.jobs}
        huds = {job.settings.hud_hidden for job in plan.jobs}
        assert len(plan.jobs) == views
        # Gamma is stratified across the client's whole supported range, so both
        # endpoints are always present and no level is repeated.
        assert GAMMA_MIN in gammas and GAMMA_MAX in gammas, views
        assert len(set(gammas)) == views, views
        assert all(GAMMA_MIN <= gamma <= GAMMA_MAX for gamma in gammas)
        # Day and night are both represented, always including noon and midnight.
        assert times == set(TIME_OF_DAY_CHOICES), views
        assert 6000 in times and 18000 in times
        # Clean (HUD hidden) and ordinary-screen captures both appear.
        assert huds == {False, True}, views

    for views in range(MIN_VIEWS, MAX_VIEWS + 1):
        strata = view_setting_strata(views)
        assert len(strata) == views
        assert {entry[0] for entry in strata} >= {GAMMA_MIN, GAMMA_MAX}
        assert {entry[1] for entry in strata} == set(TIME_OF_DAY_CHOICES)
        assert {entry[2] for entry in strata} == {False, True}


def test_setting_coverage_survives_a_seed_change_while_the_orbit_moves():
    first = _plan(seed=11, views=6)
    second = _plan(seed=12, views=6)

    def coverage(plan: CapturePlan):
        # The stratified coverage is gamma/time/HUD; weather is still a seeded draw.
        return sorted((job.settings.gamma, job.settings.time_of_day,
                       job.settings.hud_hidden) for job in plan.jobs)

    # Same seed repeats exactly; a new seed still leaves the same settings multiset.
    assert coverage(first) == coverage(_plan(seed=11, views=6))
    assert coverage(first) == coverage(second)
    # The seed does still move the orbit, so the reseed is not a no-op.
    assert [tuple(job.camera.position) for job in first.jobs] != \
        [tuple(job.camera.position) for job in second.jobs]


def test_the_seed_only_permutes_the_stratified_settings():
    strata = view_setting_strata(6)
    orders = set()
    for seed in (0, 1, 11, 12, 999):
        plan = _plan(seed=seed, views=6)
        assigned = [(job.settings.gamma, job.settings.time_of_day, job.settings.hud_hidden)
                    for job in plan.jobs]
        assert sorted(assigned) == sorted(strata)
        orders.add(tuple(assigned))
    # More than one seed really does permute the per-view assignment.
    assert len(orders) > 1


def test_explicit_base_settings_override_the_variation():
    base = CaptureSettings(gamma=0.5, time_of_day=13000, weather="rain", hud_hidden=True)
    plan = build_plan(dataset_id="ds-test", scene_id="house-a",
                      world_identity="Template 2", region=_region(), seed=11,
                      view_count=6, staging_token=TEST_STAGING_TOKEN,
                      base_settings=base)
    assert [job.settings for job in plan.jobs] == [base] * len(plan.jobs)


# --- cameras and regions ---------------------------------------------------


def test_every_planned_camera_centers_its_target_and_covers_the_region():
    region = _region()
    plan = _plan(region=region)
    for job in plan.jobs:
        width, height = job.resolution
        pixel = target_pixel(job.camera, job.target, job.resolution)
        assert pixel is not None
        assert pixel[0] == pytest.approx(width / 2.0, abs=1e-6)
        assert pixel[1] == pytest.approx(height / 2.0, abs=1e-6)
        assert bounds_coverage(job.camera, region, job.resolution) == 1.0
        occupancy = bounds_projected_occupancy(job.camera, region, job.resolution)
        assert occupancy is not None
        policy = QualityPolicy()
        assert occupancy[0] >= policy.min_projected_occupancy_width
        assert occupancy[1] >= policy.min_projected_occupancy_height
        assert 0.45 <= max(occupancy) <= 0.70
        assert _point_to_continuous_aabb_distance(job.camera.position, region) >= 2.0
        assert _point_to_continuous_aabb_distance(camera_eye(job.camera), region) >= 2.0
        assert job.settle_ticks > 0


def _point_to_continuous_aabb_distance(point, region):
    deltas = []
    for axis in range(3):
        low = region.minimum[axis]
        high = region.maximum[axis] + 1.0
        coordinate = float(point[axis])
        deltas.append(max(low - coordinate, 0.0, coordinate - high))
    return sum(delta * delta for delta in deltas) ** 0.5


@pytest.mark.parametrize(
    "dimensions",
    [(8, 40, 12), (60, 8, 12), (10, 15, 48), (17, 23, 16)],
    ids=["tall", "wide", "deep", "canal-town-like"],
)
@pytest.mark.parametrize("resolution", [(1600, 900), (1024, 768), (800, 1200)],
                         ids=["wide-image", "4x3-image", "portrait-image"])
def test_projected_framing_handles_region_shapes_and_aspect_ratios(
    dimensions, resolution, monkeypatch,
):
    monkeypatch.setattr(plan_module, "RESOLUTION_CHOICES", (resolution,))
    monkeypatch.setattr(plan_module, "FOV_CHOICES", (70.0,))
    low = tuple(-(value // 2) for value in dimensions)
    high = tuple(low[axis] + dimensions[axis] - 1 for axis in range(3))
    region = RegionBounds(low, high)
    plan = build_plan(
        dataset_id="synthetic-framing",
        scene_id=(f"shape-{dimensions[0]}-{dimensions[1]}-{dimensions[2]}-"
                  f"image-{resolution[0]}-{resolution[1]}"),
        world_identity="synthetic-only",
        region=region,
        seed=23,
        staging_token=TEST_STAGING_TOKEN,
        view_count=6,
    )
    for job in plan.jobs:
        assert job.resolution == resolution
        assert bounds_coverage(job.camera, region, resolution) == 1.0
        occupancy = bounds_projected_occupancy(job.camera, region, resolution)
        assert occupancy is not None
        policy = QualityPolicy()
        assert occupancy[0] >= policy.min_projected_occupancy_width, occupancy
        assert occupancy[1] >= policy.min_projected_occupancy_height, occupancy
        assert 0.45 <= max(occupancy) <= plan_module.FRAMING_OCCUPANCY_SAFE_MAX, occupancy
        assert _point_to_continuous_aabb_distance(job.camera.position, region) >= 2.0
        assert _point_to_continuous_aabb_distance(camera_eye(job.camera), region) >= 2.0
        assert bounds_coverage(job.camera, region, resolution) == 1.0


def test_canal_town_like_front_aspect_meets_both_frozen_axis_floors():
    region = RegionBounds((-8, -11, -8), (8, 11, 7))  # 17 x 23 x 16 voxels
    aim = (0.5, 0.5, 0.0)
    resolution = (1600, 900)
    camera = plan_module._fit_orbit_camera(
        aim=aim,
        bounds=region,
        angle=0.0,
        elevation=0.28,
        fov_degrees=70.0,
        resolution=resolution,
        preferred_radius=region.diagonal * 0.9,
        scene_id="synthetic-canal-town-front",
        view_index=0,
    )
    occupancy = bounds_projected_occupancy(camera, region, resolution)
    policy = QualityPolicy()
    assert occupancy is not None
    assert occupancy[0] >= policy.min_projected_occupancy_width
    assert occupancy[1] >= policy.min_projected_occupancy_height
    assert max(occupancy) <= plan_module.FRAMING_OCCUPANCY_SAFE_MAX
    assert bounds_coverage(camera, region, resolution) == 1.0
    assert _point_to_continuous_aabb_distance(camera.position, region) >= 2.0
    assert _point_to_continuous_aabb_distance(camera_eye(camera), region) >= 2.0
    assert camera_eye(camera)[0] == pytest.approx(aim[0])


def test_safe_fallback_allows_tall_region_when_preferred_band_cannot_meet_both_axes():
    region = RegionBounds((-8, -11, -8), (8, 11, 7))  # 17 x 23 x 16 voxels
    aim = (0.5, 0.5, 0.0)
    resolution = (1600, 900)
    camera = plan_module._fit_orbit_at_elevation(
        aim=aim,
        bounds=region,
        angle=0.4,
        elevation=0.1,
        fov_degrees=70.0,
        resolution=resolution,
        preferred_radius=region.diagonal * 0.9,
        scene_id="synthetic-canal-town-fallback",
        view_index=0,
    )
    occupancy = bounds_projected_occupancy(camera, region, resolution)
    assert occupancy is not None
    assert occupancy[0] >= QualityPolicy().min_projected_occupancy_width
    assert occupancy[1] >= QualityPolicy().min_projected_occupancy_height
    assert 0.70 < max(occupancy) <= plan_module.FRAMING_OCCUPANCY_SAFE_MAX
    assert bounds_coverage(camera, region, resolution) == 1.0


def test_default_aim_uses_continuous_voxel_aabb_center_and_explicit_target_is_preserved():
    region = RegionBounds((0, 0, 0), (2, 4, 6))
    centered = build_plan(
        dataset_id="synthetic-framing", scene_id="half-block-center",
        world_identity="synthetic-only", region=region, seed=3,
        staging_token=TEST_STAGING_TOKEN, view_count=4,
    )
    assert all(job.target == (1.5, 2.5, 3.5) for job in centered.jobs)

    explicit_target = (1.0, 2.0, 3.0)
    explicit = build_plan(
        dataset_id="synthetic-framing", scene_id="explicit-target",
        world_identity="synthetic-only", region=region, seed=3,
        staging_token=TEST_STAGING_TOKEN, view_count=4, target=explicit_target,
    )
    assert all(job.target == explicit_target for job in explicit.jobs)


def test_camera_sees_bounds_uses_continuous_voxel_aabb_center(monkeypatch):
    job = _plan().jobs[0]
    region = RegionBounds((2, 4, 6), (4, 8, 12))
    width, height = job.resolution
    calls = []

    def center_projection(points, camera, *, width, height):
        values = list(points)
        calls.append(values)
        return [(width / 2.0, height / 2.0, 1.0) for _point in values]

    monkeypatch.setattr(plan_module, "project_points", center_projection)
    assert camera_sees_bounds(job.camera, region, job.resolution)
    assert calls[0] == [(3.5, 6.5, 9.5)]


@pytest.mark.parametrize("border", ["right", "bottom"])
def test_image_bounds_are_half_open_at_the_right_and_bottom_border(monkeypatch, border):
    job = _plan().jobs[0]
    width, height = job.resolution
    border_pixel = ((float(width), float(height) / 2.0) if border == "right"
                    else (float(width) / 2.0, float(height)))

    def border_projection(points, camera, *, width, height):
        return [(border_pixel[0], border_pixel[1], 1.0) for _point in points]

    monkeypatch.setattr(plan_module, "project_points", border_projection)
    assert bounds_coverage(job.camera, _region(), job.resolution) == 0.0
    assert not camera_sees_bounds(job.camera, _region(), job.resolution)


@pytest.mark.parametrize("border", ["right", "bottom"])
def test_plan_validation_rejects_target_on_half_open_image_border(monkeypatch, border):
    plan = _plan()
    job = plan.jobs[0]
    width, height = job.resolution
    border_pixel = ((float(width), float(height) / 2.0) if border == "right"
                    else (float(width) / 2.0, float(height)))
    monkeypatch.setattr(plan_module, "target_pixel",
                        lambda *_args, **_kwargs: border_pixel)
    with pytest.raises(DatasetSchemaError, match="target does not project into frame"):
        validate_plan(plan)


def test_plan_validation_rejects_projected_occupancy_below_evaluator_floor():
    plan = _plan()
    job = plan.jobs[0]
    old_eye = camera_eye(job.camera)
    old_radius = math.dist(old_eye, job.target)
    scale = 2.5
    new_eye = tuple(
        job.target[axis] + (float(old_eye[axis]) - job.target[axis]) * scale
        for axis in range(3)
    )
    new_position = (new_eye[0], new_eye[1] - job.camera.eye_height, new_eye[2])
    distant_camera = replace(job.camera, position=new_position)
    distant_job = replace(job, camera=distant_camera)
    distant_plan = replace(plan, jobs=(distant_job, *plan.jobs[1:]))
    assert math.dist(new_eye, job.target) > old_radius
    with pytest.raises(DatasetSchemaError, match="projected region width/height.*floors"):
        validate_plan(distant_plan)


def test_plan_fails_clearly_when_clearance_and_occupancy_band_are_incompatible(monkeypatch):
    monkeypatch.setattr(plan_module, "RESOLUTION_CHOICES", ((1920, 1080),))
    monkeypatch.setattr(plan_module, "FOV_CHOICES", (90.0,))
    tiny_region = RegionBounds((0, 0, 0), (0, 0, 0))
    with pytest.raises(DatasetSchemaError, match="no camera.*occupancy"):
        build_plan(
            dataset_id="synthetic-framing",
            scene_id="infeasible-small-region",
            world_identity="synthetic-only",
            region=tiny_region,
            seed=17,
            staging_token=TEST_STAGING_TOKEN,
            view_count=4,
        )


def test_yaw_pitch_inversion_matches_the_projection_convention():
    yaw, pitch = yaw_pitch_to_target((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)

    yaw, _pitch = yaw_pitch_to_target((0.0, 0.0, 0.0), (-5.0, 0.0, 0.0))
    assert yaw == pytest.approx(90.0, abs=1e-6)

    _yaw, pitch = yaw_pitch_to_target((0.0, 0.0, 0.0), (0.0, -5.0, 0.0))
    assert pitch == pytest.approx(90.0, abs=1e-6)

    with pytest.raises(DatasetSchemaError, match="on its target"):
        yaw_pitch_to_target((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


def test_plan_validation_rejects_out_of_region_and_mismatched_cameras():
    plan = _plan()
    swapped = plan.to_dict()
    swapped["jobs"][0]["target"] = [500.0, 500.0, 500.0]
    with pytest.raises(DatasetSchemaError, match="outside the region"):
        validate_plan(CapturePlan.from_dict(swapped))

    mismatched = plan.to_dict()
    mismatched["jobs"][0]["resolution"] = [640, 480]
    with pytest.raises(DatasetSchemaError, match="intrinsics size"):
        CapturePlan.from_dict(mismatched)

    far = plan.to_dict()
    far_job = far["jobs"][0]
    far_job["camera"]["position"] = [900.0, 200.0, 900.0]
    with pytest.raises(DatasetSchemaError, match="far"):
        validate_plan(CapturePlan.from_dict(far))


def test_plan_validation_rejects_roll_the_client_cannot_apply():
    plan = _plan()
    rolled = plan.to_dict()
    rolled["jobs"][0]["camera"]["roll_degrees"] = 5.0
    with pytest.raises(DatasetSchemaError, match="roll_degrees must be 0"):
        validate_plan(CapturePlan.from_dict(rolled))


def test_recorded_feet_and_eye_are_checked_against_the_requested_pose():
    plan = _plan()
    job = plan.jobs[0]
    good = CaptureFrameResult(
        frame_id=job.frame_id, filename=job.filename, captured=True,
        actual={"feet_position": list(job.camera.position),
                "eye_position": [float(value) for value in camera_eye(job.camera)],
                "yaw_degrees": job.camera.yaw_degrees,
                "pitch_degrees": job.camera.pitch_degrees})
    assert validate_pose(plan, [good]) == []

    # The corrected contract: position is feet, so a capture that recorded the eye
    # where the feet belong is exactly an eye_height-sized error and must be caught.
    swapped = CaptureFrameResult(
        frame_id=job.frame_id, filename=job.filename, captured=True,
        actual={"feet_position": [float(value) for value in camera_eye(job.camera)],
                "eye_position": list(job.camera.position),
                "yaw_degrees": job.camera.yaw_degrees,
                "pitch_degrees": job.camera.pitch_degrees})
    problems = validate_pose(plan, [swapped])
    assert len(problems) == 2
    assert all("tolerance" in problem for problem in problems)
    assert all("1.62" in problem for problem in problems)

    missing = CaptureFrameResult(frame_id=job.frame_id, filename=job.filename,
                                 captured=True, actual={})
    assert "feet_position and eye_position" in validate_pose(plan, [missing])[0]

    # Exactly at the documented tolerance is still accepted; beyond it is not.
    at_limit = CaptureFrameResult(
        frame_id=job.frame_id, filename=job.filename, captured=True,
        actual={"feet_position": [job.camera.position[0] + EYE_POSITION_TOLERANCE,
                                  job.camera.position[1], job.camera.position[2]],
                "eye_position": [float(value) for value in camera_eye(job.camera)],
                "yaw_degrees": job.camera.yaw_degrees,
                "pitch_degrees": job.camera.pitch_degrees})
    assert validate_pose(plan, [at_limit]) == []
    beyond = CaptureFrameResult(
        frame_id=job.frame_id, filename=job.filename, captured=True,
        actual={"feet_position": [job.camera.position[0] + EYE_POSITION_TOLERANCE + 0.01,
                                  job.camera.position[1], job.camera.position[2]],
                "eye_position": [float(value) for value in camera_eye(job.camera)],
                "yaw_degrees": job.camera.yaw_degrees,
                "pitch_degrees": job.camera.pitch_degrees})
    assert "recorded feet" in validate_pose(plan, [beyond])[0]


def test_recorded_heading_is_checked_with_a_circular_yaw_delta():
    plan = _plan()
    job = plan.jobs[0]

    def result(yaw: float, pitch: float) -> CaptureFrameResult:
        return CaptureFrameResult(
            frame_id=job.frame_id, filename=job.filename, captured=True,
            actual={"feet_position": list(job.camera.position),
                    "eye_position": [float(value) for value in camera_eye(job.camera)],
                    "yaw_degrees": yaw, "pitch_degrees": pitch})

    # A yaw that wrapped past 360 is the same heading, not a 360-degree miss.
    wrapped = job.camera.yaw_degrees + 360.0
    assert validate_pose(plan, [result(wrapped, job.camera.pitch_degrees)]) == []
    assert angle_delta_degrees(wrapped, job.camera.yaw_degrees) == 0.0
    # The shortest arc is taken across the wrap point, not the raw difference.
    assert angle_delta_degrees(350.0, -10.0) == pytest.approx(0.0, abs=1e-9)
    assert angle_delta_degrees(300.0, -50.0) == pytest.approx(10.0, abs=1e-9)
    assert angle_delta_degrees(-10.0, 350.0) == pytest.approx(0.0, abs=1e-9)
    assert angle_delta_degrees(0.0, 180.0) == pytest.approx(180.0, abs=1e-9)
    assert angle_delta_degrees(90.0, -90.0) == pytest.approx(180.0, abs=1e-9)
    assert angle_delta_degrees(10.0, 350.0) == pytest.approx(20.0, abs=1e-9)

    drift = result(job.camera.yaw_degrees + 3.0, job.camera.pitch_degrees)
    assert "recorded yaw" in validate_pose(plan, [drift])[0]
    tilted = result(job.camera.yaw_degrees, job.camera.pitch_degrees + 3.0)
    assert "recorded pitch" in validate_pose(plan, [tilted])[0]

    absent = CaptureFrameResult(
        frame_id=job.frame_id, filename=job.filename, captured=True,
        actual={"feet_position": list(job.camera.position),
                "eye_position": [float(value) for value in camera_eye(job.camera)]})
    assert "yaw_degrees and pitch_degrees" in validate_pose(plan, [absent])[0]


def _settings_result(tmp_path: Path, plan: CapturePlan,
                     job: CameraCapture) -> CaptureFrameResult:
    image = tmp_path / job.filename
    image.write_bytes(_png(*job.resolution))
    return _mod_result(plan, job, image)


def test_png_dimensions_reads_the_ihdr_without_decoding(tmp_path):
    good = tmp_path / "frame.png"
    good.write_bytes(_png(1920, 1080))
    assert png_dimensions(good) == (1920, 1080)
    assert png_dimensions(tmp_path / "missing.png") is None
    not_png = tmp_path / "not.png"
    not_png.write_bytes(b"this is not a png at all")
    assert png_dimensions(not_png) is None


def test_recorded_settings_are_validated_against_the_plan(tmp_path):
    plan = _plan()
    job = plan.jobs[0]
    good = _settings_result(tmp_path, plan, job)
    assert validate_settings(plan, [good]) == []

    # The FOV contract is an integer round: 70.4 still means the requested 70.
    rounded = good.to_dict()
    rounded["actual"]["fov_degrees"] = int(round(job.camera.fov_degrees)) + 0.4
    assert validate_settings(plan, [CaptureFrameResult.from_dict(rounded)]) == []

    # A request the client's brightness option would reject can never be planned.
    with pytest.raises(DatasetSchemaError, match="gamma"):
        CaptureSettings(gamma=-0.2)
    with pytest.raises(DatasetSchemaError, match="gamma"):
        CaptureSettings(gamma=1.5)


_SETTINGS_MISMATCHES = {
    "actual_fov": ("fov", "actual fov"),
    "actual_gamma": ("gamma", "actual gamma"),
    "actual_hud": ("hud_hidden", "actual hud_hidden"),
    "actual_framebuffer": ("framebuffer_width", "actual framebuffer"),
    "actual_time": ("time_of_day", "actual time_of_day"),
    "actual_weather": ("weather", "actual weather"),
    "request_fov": ("requested", "request fov"),
    "request_time": ("requested", "request time_of_day"),
    "request_weather": ("requested", "request weather"),
    "missing_request": ("requested", "actual.requested"),
}


@pytest.mark.parametrize("mismatch", sorted(_SETTINGS_MISMATCHES))
def test_validate_settings_refuses_every_recorded_mismatch(tmp_path, mismatch):
    plan = _plan()
    job = plan.jobs[0]
    result = _settings_result(tmp_path, plan, job).to_dict()
    actual = result["actual"]
    planned_fov = int(round(job.camera.fov_degrees))
    other_weather = "thunder" if job.settings.weather != "thunder" else "clear"

    if mismatch == "actual_fov":
        actual["fov_degrees"] = planned_fov + 1
    elif mismatch == "actual_gamma":
        actual["gamma"] = float(job.settings.gamma) + 0.5
    elif mismatch == "actual_hud":
        actual["hud_hidden"] = not job.settings.hud_hidden
    elif mismatch == "actual_framebuffer":
        actual["framebuffer_width"] = int(job.resolution[0]) + 8
    elif mismatch == "actual_time":
        actual["time_of_day"] = (int(job.settings.time_of_day) + 1) % 24000
    elif mismatch == "actual_weather":
        actual["weather"] = other_weather
        actual["raining"] = other_weather in ("rain", "thunder")
        actual["thundering"] = other_weather == "thunder"
    elif mismatch == "request_fov":
        actual["requested"]["fov_degrees"] = planned_fov + 1
    elif mismatch == "request_time":
        actual["requested"]["time_of_day"] = (int(job.settings.time_of_day) + 1) % 24000
    elif mismatch == "request_weather":
        actual["requested"]["weather"] = other_weather
    elif mismatch == "missing_request":
        actual.pop("requested")

    problems = validate_settings(plan, [CaptureFrameResult.from_dict(result)])
    assert problems, f"{mismatch} was not refused"
    assert all(job.frame_id in problem for problem in problems)
    field, keyword = _SETTINGS_MISMATCHES[mismatch]
    assert any(keyword in problem for problem in problems), problems


def test_weather_contract_accepts_derived_flags_but_refuses_a_contradiction(tmp_path):
    plan = _plan()
    job = plan.jobs[0]
    result = _settings_result(tmp_path, plan, job).to_dict()
    result["actual"].pop("weather")
    # The explicit label is preferred, but the boolean flags alone still describe
    # the weather honestly.
    assert validate_settings(plan, [CaptureFrameResult.from_dict(result)]) == []
    result["actual"]["raining"] = not result["actual"]["raining"]
    result["actual"]["thundering"] = False
    assert validate_settings(plan, [CaptureFrameResult.from_dict(result)])


def test_explicit_region_file_requires_versioned_explicit_bounds(tmp_path):
    good = tmp_path / "regions.json"
    good.write_text(json.dumps({"schema_version": 1, "regions": [
        {"name": "house", "min": [0, 60, 0], "max": [10, 70, 10], "views": 4}]}),
        encoding="utf-8")
    requests = load_region_requests(good)
    assert len(requests) == 1 and isinstance(requests[0], RegionRequest)
    assert requests[0].bounds.size == (11, 11, 11)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 2, "regions": []}), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="unsupported region file"):
        load_region_requests(bad)

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"schema_version": 1, "regions": [{"name": "x"}]}),
                       encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="min"):
        load_region_requests(missing)


def test_landscape_region_bounds_are_bounded_and_use_the_developer_height():
    region = landscape_region(spawn_x=10, spawn_z=-20, surface_y=70, radius=16)
    assert region.minimum == (-6, 62, -36)
    assert region.maximum == (26, 94, -4)
    with pytest.raises(DatasetSchemaError, match="positive"):
        landscape_region(spawn_x=0, spawn_z=0, surface_y=70, radius=0)


def test_region_bounds_reject_inverted_or_oversized_boxes():
    with pytest.raises(DatasetSchemaError, match="below"):
        RegionBounds((5, 0, 0), (4, 1, 1))
    with pytest.raises(DatasetSchemaError, match="edge"):
        RegionBounds((0, 0, 0), (600, 1, 1))
    with pytest.raises(DatasetSchemaError, match="volume"):
        RegionBounds((0, 0, 0), (100, 50, 50))


# --- splits ----------------------------------------------------------------


def test_split_is_scene_level_never_view_level():
    volume = _volume()
    first = split_for_scene(volume=volume)
    assert first in ("train", "dev", "test")
    for _ in range(5):
        assert split_for_scene(volume=volume) == first

    other = BlockVolume()
    other.set((1, 2, 3), BlockObservation(BlockState("minecraft:stone")))
    assert split_for_scene(volume=other) == split_for_scene(volume=other)


def test_identical_topology_keeps_one_split_across_capture_seeds():
    """The defect: reseeding a capture must not move a scene between splits."""
    volume = _volume()
    canonical = split_for_scene(volume=volume)
    for seed in (0, 1, 4, 5, 11, 999):
        for family in ("house", "tower", "landscape", "custom"):
            assert split_for_scene(seed=seed, family=family, volume=volume) == canonical
            assert scene_group(volume_topology(volume)) == scene_group(volume_topology(volume))


def test_splits_cover_every_bucket_over_many_scenes():
    seen = set()
    for index in range(120):
        volume = BlockVolume()
        volume.set((index, 1, 1), BlockObservation(BlockState("minecraft:stone")))
        seen.add(split_for_scene(volume=volume))
    assert seen == {"train", "dev", "test"}


def test_scene_leakage_across_splits_is_rejected():
    group = scene_group(volume_topology(_volume()))
    report = split_report([
        {"scene_id": "a", "split": "train", "scene_group": group},
        {"scene_id": "b", "split": "test", "scene_group": "other"},
    ])
    assert report["scenes"] == 2
    assert report["by_split"] == {"train": 1, "dev": 0, "test": 1}
    assert report["groups"] == 2

    with pytest.raises(DatasetSchemaError, match="both"):
        split_report([
            {"scene_id": "a", "split": "train"},
            {"scene_id": "a", "split": "test"},
        ])
    # The same structure may be recaptured, but never in a second split.
    with pytest.raises(DatasetSchemaError, match="scene group"):
        split_report([
            {"scene_id": "a", "split": "train", "scene_group": group},
            {"scene_id": "b", "split": "test", "scene_group": group},
        ])
    repeated = split_report([
        {"scene_id": "a", "split": "train", "scene_group": group},
        {"scene_id": "b", "split": "train", "scene_group": group},
    ])
    assert repeated["groups"] == 1
    assert repeated["repeated_groups"] == {group: 2}
    with pytest.raises(DatasetSchemaError, match="unknown split"):
        split_report([{"scene_id": "a", "split": "validation"}])


# --- safe paths ------------------------------------------------------------


def test_safe_join_refuses_traversal_and_absolute_components(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert safe_join(root, "saves", "world") == root / "saves" / "world"
    for bad in ("..", "../escape", "a/b", "a\\b", "", ".", "trailing."):
        with pytest.raises(DatasetSchemaError):
            safe_join(root, bad)


# --- inventory and staging -------------------------------------------------


def test_inventory_lists_saves_read_only_and_staging_copy_excludes_the_lock(tmp_path):
    game = tmp_path / "game"
    _legacy_level_dat(game / "saves" / "Alpha" / "level.dat", level_name="Alpha World")
    (game / "saves" / "Alpha" / "region").mkdir(parents=True)
    (game / "saves" / "Alpha" / "region" / "r.0.0.mca").write_bytes(b"region")
    (game / "saves" / "Alpha" / "session.lock").write_bytes(b"lock")
    _legacy_level_dat(game / "saves" / "Beta" / "level.dat", level_name="Beta")

    saves = inventory_saves(game)
    assert [save.name for save in saves] == ["Alpha", "Beta"]
    alpha = saves[0]
    assert alpha.world_identity == "Alpha World"
    assert alpha.last_played_ms == 1_700_000_000_000
    assert alpha.size_bytes > 0
    before = {path: path.stat().st_mtime_ns for path in alpha.path.rglob("*") if path.is_file()}

    staging = tmp_path / "staging"
    staged = copy_world_to_staging(alpha, staging)
    assert staged == staging / "saves" / "Alpha"
    assert (staged / "level.dat").exists()
    assert (staged / "region" / "r.0.0.mca").read_bytes() == b"region"
    assert not (staged / "session.lock").exists()

    after = {path: path.stat().st_mtime_ns for path in alpha.path.rglob("*") if path.is_file()}
    assert before == after, "inventory/staging must not touch the source save"

    with pytest.raises(DatasetSchemaError, match="already exists"):
        copy_world_to_staging(alpha, staging)
    copy_world_to_staging(alpha, staging, force=True)


def test_staging_copy_is_bounded_and_records_both_fingerprints(tmp_path):
    game = tmp_path / "game"
    _legacy_level_dat(game / "saves" / "Alpha" / "level.dat", level_name="Alpha World")
    (game / "saves" / "Alpha" / "region").mkdir(parents=True)
    (game / "saves" / "Alpha" / "region" / "r.0.0.mca").write_bytes(b"region")
    alpha = inventory_saves(game)[0]

    staging = tmp_path / "staging"
    staged = stage_world(alpha, staging)
    assert staged.destination == staging / "saves" / "Alpha"
    # The token *is* the content fingerprint of the copy, and it survives a
    # re-computation over the staged tree.
    assert staged.staging_token == staged.source_fingerprint
    assert staged.staging_token == staging_fingerprint(staged.destination)
    assert staged.staging_token == staging_token_for(staged.source_fingerprint)
    assert staged.file_count == 2
    assert staged.total_bytes == (game / "saves" / "Alpha" / "level.dat").stat().st_size + 6

    # Assigned caps are enforced against the source before anything is copied.
    with pytest.raises(DatasetSchemaError, match="exceeds 1 files"):
        copy_world_to_staging(alpha, tmp_path / "too-many",
                              limits=StagingLimits(max_files=1))
    with pytest.raises(DatasetSchemaError, match="bytes"):
        copy_world_to_staging(alpha, tmp_path / "too-big",
                              limits=StagingLimits(max_bytes=4))
    assert not (tmp_path / "too-many").exists() or not (
        tmp_path / "too-many" / "saves" / "Alpha").exists()
    with pytest.raises(StagingLimitError, match="positive"):
        StagingLimits(max_files=0)


def test_the_world_marker_lives_in_the_copied_world_root_and_survives_game_writes(tmp_path):
    """The mod resolves the open save root, so the proof must live inside the world.

    The marker is written after the copy was verified, so it never perturbs the
    staged fingerprint. The live tree is never rehashed: Minecraft rewrites ordinary
    files (level.dat, session.lock, player data) as soon as the copy is opened, and
    a valid older copy must still validate.
    """
    game = tmp_path / "game"
    (game / "saves" / "Alpha" / "region").mkdir(parents=True, exist_ok=True)
    (game / "saves" / "Alpha" / "region" / "r.0.0.mca").write_bytes(b"region")
    _legacy_level_dat(game / "saves" / "Alpha" / "level.dat", level_name="Alpha World")
    alpha = inventory_saves(game)[0]

    staging = tmp_path / "staging"
    staged = stage_world(alpha, staging)
    # The server reports LevelName, and the save folder can differ from it; the
    # marker is keyed to the destination directory the stager actually produced.
    assert alpha.name == "Alpha"
    assert alpha.world_identity == "Alpha World"
    assert staged.destination == staging / "saves" / "Alpha"

    marker = world_marker_path(staged.destination)
    assert marker == staged.destination / ".mc-photo-capture-staging.json"
    marker.write_text(json.dumps(staging_marker_payload(
        token=staged.staging_token,
        source_fingerprint=staged.source_fingerprint,
        world_identity=alpha.world_identity,
        dataset_id="ds-test", scene_id="alpha-house", plan_id="plan-abc")),
        encoding="utf-8")

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["token"] == staged.staging_token
    assert payload["world_identity"] == "Alpha World"

    # The game updates the live tree; the marker's token and identity are untouched,
    # and the plan cut from the pristine copy still matches the marker the mod reads.
    (staged.destination / "level.dat").write_bytes(b"rewritten-by-minecraft")
    (staged.destination / "session.lock").write_bytes(b"lock")
    reread = json.loads(marker.read_text(encoding="utf-8"))
    assert reread["token"] == staged.staging_token
    assert reread["world_identity"] == alpha.world_identity
    # Re-staging the untouched source reproduces the same token, so a marker written
    # before the game touched the copy remains valid for the plan built from it.
    assert stage_world(alpha, tmp_path / "second-staging", force=True).staging_token \
        == staged.staging_token


def test_staging_refuses_a_symlinked_or_external_entry(tmp_path):
    game = tmp_path / "game"
    _legacy_level_dat(game / "saves" / "Alpha" / "level.dat", level_name="Alpha")
    external = tmp_path / "outside" / "secret.dat"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"external")
    link = game / "saves" / "Alpha" / "external.dat"
    try:
        link.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not permit creating symlinks in the test")
    alpha = inventory_saves(game)[0]
    with pytest.raises(DatasetSchemaError, match="symlink or reparse point"):
        copy_world_to_staging(alpha, tmp_path / "staging")
    assert not (tmp_path / "staging" / "saves" / "Alpha").exists(), (
        "a refused copy must not leave a partial staged world behind")


def test_staging_fingerprint_is_deterministic_and_content_addressed(tmp_path):
    def make_world(name: str, payload: bytes) -> Path:
        world = tmp_path / "saves" / name
        (world / "region").mkdir(parents=True)
        (world / "level.dat").write_bytes(payload)
        (world / "region" / "r.0.0.mca").write_bytes(b"region")
        (world / "session.lock").write_bytes(b"lock")
        return world

    first = make_world("Alpha", b"level-one")
    again = make_world("Alpha-copy", b"level-one")
    other = make_world("Beta", b"level-two")

    fingerprint = staging_fingerprint(first)
    assert fingerprint == staging_fingerprint(first)
    # The same bytes in a directory under another name are the same world: the
    # fingerprint covers relative content, not the absolute path.
    assert fingerprint == staging_fingerprint(again)
    assert staging_fingerprint(other) != fingerprint
    (first / "level.dat").write_bytes(b"level-three")
    assert staging_fingerprint(first) != fingerprint
    assert staging_fingerprint(first) != staging_fingerprint(other)


def test_staging_detects_a_source_that_changed_during_the_copy(tmp_path, monkeypatch):
    game = tmp_path / "game"
    _legacy_level_dat(game / "saves" / "Alpha" / "level.dat", level_name="Alpha")
    (game / "saves" / "Alpha" / "region").mkdir(parents=True)
    region = game / "saves" / "Alpha" / "region" / "r.0.0.mca"
    region.write_bytes(b"region")
    alpha = inventory_saves(game)[0]

    import mc_photogrammetry.dataset.inventory as inventory

    real_copytree = inventory.shutil.copytree

    def racing_copytree(source, destination, *args, **kwargs):
        result = real_copytree(source, destination, *args, **kwargs)
        # A live Minecraft session writing to the source while it is being staged
        # must be reported, not silently staged as a different world.
        region.write_bytes(b"region-changed-by-the-game")
        return result

    monkeypatch.setattr(inventory.shutil, "copytree", racing_copytree)
    with pytest.raises(DatasetSchemaError, match="changed while it was being copied"):
        copy_world_to_staging(alpha, tmp_path / "staging")


def test_inventory_reports_spawn_and_rejects_a_missing_game_dir(tmp_path):
    from mc_photogrammetry.dataset.plan import spawn_from_save

    game = tmp_path / "game"
    _legacy_level_dat(game / "saves" / "Alpha" / "level.dat", level_name="Alpha",
                      spawn_x=7, spawn_z=-9)
    assert spawn_from_save(game / "saves" / "Alpha" / "level.dat") == (7, -9)

    with pytest.raises(DatasetSchemaError, match="no such game directory"):
        inventory_saves(tmp_path / "absent")


def test_spawn_y_reader_returns_the_full_spawn_and_refuses_a_missing_one(tmp_path):
    level_dat = tmp_path / "level.dat"
    _legacy_level_dat(level_dat, level_name="Alpha", spawn_x=7, spawn_z=-9, spawn_y=71)
    assert spawn_position_from_save(level_dat) == (7, 71, -9)

    # The display-only reader still degrades quietly, but the placement reader must
    # never invent a height: a save without SpawnY is refused, not defaulted to y=0.
    _legacy_level_dat(level_dat, level_name="Alpha", spawn_x=7, spawn_z=-9)
    with pytest.raises(DatasetSchemaError, match="SpawnY"):
        spawn_position_from_save(level_dat)

    level_dat.write_bytes(b"this is not a gzipped nbt file")
    with pytest.raises(DatasetSchemaError, match="SpawnY"):
        spawn_position_from_save(level_dat)


def _prepare_module():
    """Load the CLI module so the landscape path can be exercised in-process."""
    import importlib.util

    path = ROOT / "scripts" / "prepare_capture_dataset.py"
    spec = importlib.util.spec_from_file_location("ds17j_prepare_capture_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _landscape_game(tmp_path: Path, *, spawn_y: int | None) -> Path:
    game = tmp_path / "game"
    _legacy_level_dat(game / "saves" / "Alpha" / "level.dat", level_name="Alpha World",
                      spawn_x=3, spawn_z=-4, spawn_y=spawn_y)
    region = game / "saves" / "Alpha" / "region"
    region.mkdir(parents=True, exist_ok=True)
    (region / "r.0.0.mca").write_bytes(b"region")
    return game


def _landscape_arguments(game: Path, dataset: Path, *extra: str) -> list[str]:
    return ["--game-dir", str(game), "--save", "Alpha", "--dataset-id", "ds-test",
            "--seed", "7", "--landscape", "--radius", "4", "--views", "4",
            "--dataset", str(dataset), *extra]


def test_prepare_landscape_defaults_to_spawn_y_and_honours_an_explicit_override(
        tmp_path, capsys):
    prepare = _prepare_module()

    game = _landscape_game(tmp_path, spawn_y=71)
    dataset = tmp_path / "spawn-y"
    assert prepare.main(_landscape_arguments(game, dataset)) == 0
    assert "SpawnY 71" in capsys.readouterr().out
    scenes = [entry for entry in (dataset / "scenes").iterdir() if entry.is_dir()]
    assert len(scenes) == 1
    plan = json.loads((scenes[0] / "plan.json").read_text(encoding="utf-8"))
    assert plan["region"]["min"][1] == 71 - 8
    assert plan["region"]["max"][1] == 71 + 24

    # An explicit developer height overrides the save's own spawn, and is the only
    # path that can run against a save that carries no SpawnY at all.
    override_game = _landscape_game(tmp_path / "override", spawn_y=None)
    override_dataset = tmp_path / "override-ds"
    assert prepare.main(_landscape_arguments(override_game, override_dataset,
                                             "--surface-y", "70")) == 0
    scenes = [entry for entry in (override_dataset / "scenes").iterdir()
              if entry.is_dir()]
    plan = json.loads((scenes[0] / "plan.json").read_text(encoding="utf-8"))
    assert plan["region"]["min"][1] == 70 - 8
    assert plan["region"]["max"][1] == 70 + 24

    # With no SpawnY and no override the planner refuses instead of assuming a height.
    refused = tmp_path / "refused"
    with pytest.raises(SystemExit, match="--surface-y"):
        prepare.main(_landscape_arguments(override_game, refused))
    assert not (refused / "scenes").exists()


def test_prepare_fixed_settings_json_reaches_every_planned_job(tmp_path, capsys):
    prepare = _prepare_module()
    game = _landscape_game(tmp_path, spawn_y=71)
    dataset = tmp_path / "fixed-dataset"
    requested = {
        "gamma": 0.37,
        "time_of_day": 13500,
        "weather": "rain",
        "hud_hidden": True,
        "resource_pack": 'pack "quoted" $data.zip',
        "shader_pack": None,
    }
    assert prepare.main(_landscape_arguments(
        game, dataset, "--fixed-settings-json",
        json.dumps(requested, ensure_ascii=False, separators=(",", ":")))) == 0
    capsys.readouterr()
    scene = next(path for path in (dataset / "scenes").iterdir() if path.is_dir())
    plan = json.loads((scene / "plan.json").read_text(encoding="utf-8"))
    assert len(plan["jobs"]) == 4
    assert all(job["settings"] == requested for job in plan["jobs"])


def test_prepare_without_settings_flag_keeps_deterministic_strata(tmp_path, capsys):
    prepare = _prepare_module()
    game = _landscape_game(tmp_path, spawn_y=71)
    dataset = tmp_path / "stratified-dataset"
    assert prepare.main(_landscape_arguments(game, dataset)) == 0
    capsys.readouterr()
    scene = next(path for path in (dataset / "scenes").iterdir() if path.is_dir())
    plan = json.loads((scene / "plan.json").read_text(encoding="utf-8"))
    assert len(plan["jobs"]) == 4
    assert len({job["settings"]["gamma"] for job in plan["jobs"]}) == 4


@pytest.mark.parametrize("value, message", [
    ('{"gamma":0.5}', "must contain all core keys"),
    ('{"gamma":0.5,"time_of_day":6000,"weather":"clear",'
     '"hud_hidden":"false"}', "hud_hidden must be a boolean"),
    ('{"gamma":0.5,"time_of_day":6000,"weather":"clear",'
     '"hud_hidden":false,"extra":1}', "unsupported keys"),
    ('{"gamma":0.5,"time_of_day":6000.5,"weather":"clear",'
     '"hud_hidden":false}', "time_of_day must be an integer"),
])
def test_prepare_rejects_invalid_fixed_settings_before_staging(
        tmp_path, value, message):
    prepare = _prepare_module()
    game = _landscape_game(tmp_path, spawn_y=71)
    dataset = tmp_path / "must-not-create"
    with pytest.raises(SystemExit, match=message):
        prepare.main(_landscape_arguments(
            game, dataset, "--fixed-settings-json", value))
    assert not dataset.exists()


def test_prepare_rejects_contradictory_settings_modes_before_staging(tmp_path):
    prepare = _prepare_module()
    game = _landscape_game(tmp_path, spawn_y=71)
    dataset = tmp_path / "must-not-create"
    complete = json.dumps({"gamma": 0.0, "time_of_day": 6000,
                           "weather": "clear", "hud_hidden": False})
    with pytest.raises(SystemExit):
        prepare.main(_landscape_arguments(
            game, dataset, "--fixed-settings-json", complete,
            "--stratified-settings"))
    assert not dataset.exists()


# --- ground truth, blocks, litematic --------------------------------------


def test_ground_truth_round_trip_drops_air_and_keeps_directional_states():
    region = _region()
    export = GroundTruthExport(
        scene_id="house-a", world_identity="Template 2", region=region,
        blocks=(
            BlockRecord((1, 61, 1), "minecraft:stone"),
            BlockRecord((2, 61, 1), "minecraft:oak_stairs",
                        {"facing": "west", "half": "top"}),
            BlockRecord((3, 61, 1), "minecraft:air"),
            BlockRecord((4, 61, 1), "minecraft:cave_air"),
        ))
    reloaded = parse_ground_truth(json.loads(json.dumps(export.to_dict())))
    assert reloaded.to_dict() == export.to_dict()

    volume = volume_from_ground_truth(reloaded)
    assert set(volume.blocks) == {(1, 61, 1), (2, 61, 1)}
    stairs = volume.get((2, 61, 1)).state
    assert stairs.name == "minecraft:oak_stairs"
    assert dict(stairs.properties) == {"facing": "west", "half": "top"}


def test_ground_truth_rejects_blocks_outside_the_region_and_duplicates():
    with pytest.raises(DatasetSchemaError, match="outside"):
        GroundTruthExport(
            scene_id="s", world_identity="w", region=RegionBounds((0, 0, 0), (1, 1, 1)),
            blocks=(BlockRecord((9, 9, 9), "minecraft:stone"),))
    with pytest.raises(DatasetSchemaError, match="duplicate"):
        GroundTruthExport(
            scene_id="s", world_identity="w", region=RegionBounds((0, 0, 0), (1, 1, 1)),
            blocks=(BlockRecord((0, 0, 0), "minecraft:stone"),
                    BlockRecord((0, 0, 0), "minecraft:dirt")))
    with pytest.raises(DatasetSchemaError, match="namespaced"):
        BlockRecord((0, 0, 0), "stone")


def test_reference_litematic_round_trips_and_keeps_region_coordinates(tmp_path):
    region = RegionBounds((0, 60, 0), (23, 83, 23))
    export = ground_truth_for_volume(scene_id="house-a", world_identity="Template 2",
                                     region=region, volume=_volume())
    destination = tmp_path / "reference.litematic"
    write_reference_litematic(export, destination, name="house-a")

    snapshot = load_litematic(destination)
    assert snapshot.size == region.size
    assert snapshot.blocks[(1, 1, 1)].name == "minecraft:stone"
    stairs = snapshot.blocks[(2, 1, 1)]
    assert stairs.name == "minecraft:oak_stairs"
    assert dict(stairs.properties)["facing"] == "west"
    assert snapshot.blocks[(3, 1, 1)].name == "minecraft:oak_log"

    with pytest.raises(DatasetSchemaError, match="overwrite"):
        write_reference_litematic(export, destination)


# --- finalization ----------------------------------------------------------


def _prepared_dataset(tmp_path: Path, *, seed: int = 11, views: int = 5):
    dataset = tmp_path / "dataset"
    staging = dataset / "staging-game"
    region = _region()
    plan = _plan(seed=seed, region=region, views=views)
    recipe = SceneRecipe(seed=seed, family="custom", dimensions=region.size,
                         features=("real-minecraft-capture",), source="real-minecraft-capture")
    write_scene_recipe(dataset, plan.scene_id, recipe)
    write_scene_plan(dataset, plan)
    export = ground_truth_for_volume(scene_id=plan.scene_id,
                                     world_identity=plan.world_identity,
                                     region=region, volume=_volume())
    _write_mod_outputs(staging, plan, export)
    return dataset, staging, plan, export


def test_finalize_writes_hashed_scene_files_and_a_deterministic_index(tmp_path):
    dataset, staging, plan, _export = _prepared_dataset(tmp_path)
    entry = finalize_scene(dataset, plan.scene_id, staging)

    scene = scene_dir(dataset, plan.scene_id)
    capture = json.loads((scene / "capture.json").read_text(encoding="utf-8"))
    assert capture["advanced_only"] is True
    assert "never normal reconstruction input" in capture["usage"]
    assert len(capture["frames"]) == len(plan.jobs)
    assert len(capture["cameras"]) == len(plan.jobs)
    assert [row["filename"] for row in capture["screenshots"]] == list(plan.filenames)
    for row in capture["screenshots"]:
        assert row["sha256"] == _sha256(scene / "captures" / row["filename"])

    assert (scene / "reference.litematic").exists()
    assert (scene / "ground_truth.json").exists()
    volume = volume_from_ground_truth(
        parse_ground_truth(json.loads((scene / "ground_truth.json").read_text())))
    assert entry["split"] == split_for_scene(volume=volume)
    assert entry["topology_digest"] == volume_topology(volume)
    assert entry["scene_group"] == scene_group(volume_topology(volume))
    assert entry["hashes"]["reference.litematic"]
    assert entry["screenshots"][0]["source_sha256"] == entry["screenshots"][0]["sha256"]

    write_dataset_index(dataset)
    first = (dataset / "index.json").read_bytes()
    write_dataset_index(dataset, force=True)
    assert (dataset / "index.json").read_bytes() == first
    index = json.loads(first.decode("utf-8"))
    assert index["summary"]["scenes"] == 1
    assert index["summary"]["groups"] == 1
    assert index["scenes"][0]["scene_id"] == plan.scene_id
    assert index["scenes"][0]["provenance"]["advanced_only"] is True
    assert index["scenes"][0]["provenance"]["split_identity"] == "scene-topology"


def test_report_only_provenance_finalizes_unbound_frames_from_executor_plan_path(
        tmp_path, monkeypatch):
    import hashlib
    from mc_photogrammetry.dataset.hashing import canonical_json
    from mc_photogrammetry.dataset.runtime_render_provenance import (
        MAX_SIDECAR_BYTES, seal_sidecar)

    dataset, staging, plan, _export = _prepared_dataset(tmp_path)
    scene = scene_dir(dataset, plan.scene_id)
    staged_plan = staging / "mc-photo-capture" / "plans" / f"{plan.plan_id}.json"
    staged_plan.parent.mkdir(parents=True)
    staged_plan.write_bytes((scene / "plan.json").read_bytes())
    mod_path = staging / "mods" / "mc-photo-capture-test.jar"
    mod_path.parent.mkdir(parents=True)
    mod_path.write_bytes(b"synthetic capture mod")
    mod_sha = _sha256(mod_path)
    dataset_plan_sha = _sha256(scene / "plan.json")
    nonce = "ab" * 16
    expectation = {
        "schema_version": 1, "mode": "report_only", "attempt_nonce": nonce,
        "plan_id": plan.plan_id, "plan_sha256": _sha256(staged_plan),
        "dataset_plan_sha256": dataset_plan_sha, "staging_token": plan.staging_token,
        "scene_id": plan.scene_id, "world_identity": plan.world_identity,
        "staging_game_dir": str(staging.resolve()),
        "capture_mod_sha256": mod_sha, "capture_mod_source_sha256": None,
        "capture_mod_staged_sha256": mod_sha, "capture_mod_expected_sha256": None,
        "capture_mod_source_path": None, "capture_mod_staged_path": str(mod_path),
    }
    expectation_path = scene / "runtime-render-provenance.expect.json"
    expectation_bytes = canonical_json(expectation).encode("utf-8")
    expectation_path.write_bytes(expectation_bytes)
    expectation_sha = hashlib.sha256(expectation_bytes).hexdigest()

    results_path = mod_output_dir(staging, plan.plan_id) / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    sidecar_frames = []
    expected_frames = []
    for result in results:
        actual = result["actual"]
        frame_id = result["frame_id"]
        snapshot = {"frame_id": frame_id,
                    "capture_epoch_status": "capture_epoch_unbound",
                    "resource_epoch_status": "unbound",
                    "ordered_resource_packs": [],
                    "shader_status": "unknown_no_supported_shader_state_adapter",
                    "gl": {"status": "observed_at_recorder_submission_not_readback_bound"}}
        snapshot_sha = hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()
        png = staging / "screenshots" / result["screenshot"].split("/", 1)[1]
        actual["screenshot_bytes"] = png.stat().st_size
        actual["runtime_render_provenance"] = {
            "filename": "runtime-render-provenance.json", "attempt_nonce": nonce,
            "snapshot_sha256": snapshot_sha,
            "capture_epoch_status": "capture_epoch_unbound"}
        sidecar_frames.append({
            "frame_id": frame_id, "filename": result["filename"],
            "screenshot_sha256": actual["screenshot_sha256"],
            "screenshot_bytes": actual["screenshot_bytes"],
            "snapshot_sha256": snapshot_sha,
            "capture_epoch_status": "capture_epoch_unbound", "snapshot": snapshot})
        expected_frames.append({
            "frame_id": frame_id, "filename": result["filename"],
            "screenshot_sha256": actual["screenshot_sha256"],
            "screenshot_bytes": actual["screenshot_bytes"]})
    results_path.write_text(json.dumps(results), encoding="utf-8")
    sidecar = seal_sidecar({
        "schema_version": 1, "attempt_nonce": nonce, "plan_id": plan.plan_id,
        "plan_sha256": expectation["plan_sha256"],
        "dataset_plan_sha256": dataset_plan_sha,
        "staging_token": plan.staging_token, "scene_id": plan.scene_id,
        "world_identity": plan.world_identity, "mode": "report_only",
        "expectation_sha256": expectation_sha,
        "runtime": {"minecraft_version": "1.21.5", "fabric_loader_version": "0.16.10",
                    "java_version": "21", "capture_mod_sha256": mod_sha,
                    "capture_mod_source_sha256": None,
                    "capture_mod_staged_sha256": mod_sha,
                    "capture_mod_loaded_sha256": None, "client_artifacts": [],
                    "identity_status": "incomplete"},
        "resource_status": "unknown", "resource_packs": [],
        "shader_status": "unknown_no_supported_shader_state_adapter",
        "certification": {"status": "unavailable", "unavailable_reasons": [
            "framebuffer_readback_epoch_unbound"]},
        "gl_status": "partial", "frames": sidecar_frames})
    output = mod_output_dir(staging, plan.plan_id)
    sidecar_path = output / "runtime-render-provenance.json"
    sidecar_bytes = canonical_json(sidecar).encode("utf-8")
    sidecar_path.write_bytes(sidecar_bytes)
    status = {
        "runtime_render_provenance_required": True,
        "plan_id": plan.plan_id, "staging_token": plan.staging_token,
        "runtime_render_provenance": {
            "filename": sidecar_path.name, "attempt_nonce": nonce,
            "expectation_sha256": expectation_sha,
            "plan_sha256": expectation["plan_sha256"],
            "capture_mod_source_sha256": None,
            "capture_mod_staged_sha256": mod_sha, "mode": "report_only",
            "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "bytes": len(sidecar_bytes), "seal_sha256": sidecar["seal_sha256"]}}
    (output / "run_status.json").write_text(json.dumps(status), encoding="utf-8")

    sidecar_path.write_bytes(b"x" * (MAX_SIDECAR_BYTES + 1))
    with pytest.raises(DatasetSchemaError, match="byte cap"):
        finalize_scene(dataset, plan.scene_id, staging, provenance_mode="report_only",
                       provenance_expectation_path=expectation_path,
                       provenance_expectation_sha256=expectation_sha)
    assert not (scene / "capture.json").exists()
    sidecar_path.write_bytes(sidecar_bytes)

    import mc_photogrammetry.dataset.runtime_render_provenance as provenance_module
    with monkeypatch.context() as patcher:
        patcher.setattr(provenance_module, "_linklike", lambda _path: True)
        with pytest.raises(DatasetSchemaError, match="cannot be read"):
            finalize_scene(dataset, plan.scene_id, staging, provenance_mode="report_only",
                           provenance_expectation_path=expectation_path,
                           provenance_expectation_sha256=expectation_sha)
    assert not (scene / "capture.json").exists()

    entry = finalize_scene(
        dataset, plan.scene_id, staging, provenance_mode="report_only",
        provenance_expectation_path=expectation_path,
        provenance_expectation_sha256=expectation_sha)
    capture = json.loads((scene / "capture.json").read_text(encoding="utf-8"))
    assert entry["hashes"]["runtime-render-provenance.json"]
    assert capture["runtime_render_provenance"]["status"] == "report_only_incomplete"
    assert capture["runtime_render_provenance"]["certification_status"] == "unavailable"
    assert capture["runtime_render_provenance"]["unavailable_reasons"] == [
        "framebuffer_readback_epoch_unbound"]
    assert len(capture["frames"]) == len(plan.jobs)


def test_finalize_refuses_a_screenshot_whose_bytes_changed(tmp_path):
    dataset, staging, plan, _export = _prepared_dataset(tmp_path)
    out = mod_output_dir(staging, plan.plan_id)
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))

    # A tampered source file no longer matches the digest the mod recorded.
    (staging / "screenshots" / plan.jobs[0].filename).write_bytes(
        _png(*plan.jobs[0].resolution) + b"tampered")
    with pytest.raises(DatasetSchemaError, match="refusing to copy a changed screenshot"):
        finalize_scene(dataset, plan.scene_id, staging)

    # A missing digest is a refusal too, not an unchecked copy.
    (staging / "screenshots" / plan.jobs[0].filename).write_bytes(
        _png(*plan.jobs[0].resolution))
    payload[0]["actual"].pop("screenshot_sha256")
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="screenshot_sha256"):
        finalize_scene(dataset, plan.scene_id, staging)

    # Restoring the recorded digest lets the scene finalize with matching hashes.
    payload[0]["actual"]["screenshot_sha256"] = _sha256(
        staging / "screenshots" / plan.jobs[0].filename)
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    entry = finalize_scene(dataset, plan.scene_id, staging)
    for row in entry["screenshots"]:
        assert row["sha256"] == row["source_sha256"]


def test_finalize_refuses_incomplete_or_failed_captures(tmp_path):
    dataset, staging, plan, export = _prepared_dataset(tmp_path)
    out = mod_output_dir(staging, plan.plan_id)
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))

    missing = payload[:-1]
    (out / "results.json").write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="no result recorded"):
        finalize_scene(dataset, plan.scene_id, staging)

    payload[-1]["captured"] = False
    payload[-1]["errors"] = ["screenshot call failed"]
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="capture failed"):
        finalize_scene(dataset, plan.scene_id, staging)

    payload[-1]["captured"] = True
    payload[-1]["errors"] = []
    payload[-1]["screenshot"] = "screenshots/does-not-exist.png"
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="not found"):
        finalize_scene(dataset, plan.scene_id, staging)


def test_finalize_rejects_a_capture_that_did_not_honor_the_pose(tmp_path):
    dataset, staging, plan, _export = _prepared_dataset(tmp_path)
    out = mod_output_dir(staging, plan.plan_id)
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    # A drift larger than the documented tolerance must never become ground truth.
    payload[0]["actual"]["feet_position"] = [
        payload[0]["actual"]["feet_position"][0] + 2.0,
        payload[0]["actual"]["feet_position"][1],
        payload[0]["actual"]["feet_position"][2],
    ]
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="tolerance"):
        finalize_scene(dataset, plan.scene_id, staging)

    # Removing the recorded positions is also a refusal, not a silent pass.
    payload[0]["actual"].pop("feet_position")
    payload[0]["actual"].pop("eye_position")
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="feet_position and eye_position"):
        finalize_scene(dataset, plan.scene_id, staging)

    # Restoring the measured pose lets the scene finalize.
    payload[0]["actual"]["feet_position"] = list(plan.jobs[0].camera.position)
    payload[0]["actual"]["eye_position"] = [
        float(value) for value in camera_eye(plan.jobs[0].camera)]
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    entry = finalize_scene(dataset, plan.scene_id, staging)
    assert entry["scene_id"] == plan.scene_id


def test_finalize_refuses_a_capture_whose_actual_settings_disagree(tmp_path):
    dataset, staging, plan, _export = _prepared_dataset(tmp_path)
    out = mod_output_dir(staging, plan.plan_id)
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))

    # The scene may not be finalized from a screenshot the client did not take with
    # the requested brightness, even though the pose and the file are perfect.
    payload[0]["actual"]["gamma"] = float(payload[0]["actual"]["gamma"]) + 0.5
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="actual gamma"):
        finalize_scene(dataset, plan.scene_id, staging)

    # Restoring the observation but corrupting the recorded *request* is refused too.
    payload[0]["actual"]["gamma"] = plan.jobs[0].settings.gamma
    payload[0]["actual"]["requested"]["fov_degrees"] += 1
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="request fov"):
        finalize_scene(dataset, plan.scene_id, staging)

    # A missing recorded request is a refusal, not a silent pass.
    payload[0]["actual"].pop("requested")
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="actual.requested"):
        finalize_scene(dataset, plan.scene_id, staging)


def test_finalize_refuses_a_screenshot_whose_pixels_do_not_match_the_plan(tmp_path):
    dataset, staging, plan, _export = _prepared_dataset(tmp_path)
    out = mod_output_dir(staging, plan.plan_id)
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    image = staging / "screenshots" / plan.jobs[0].filename
    width, height = plan.jobs[0].resolution

    # The recorded framebuffer still claims the planned size, so only the file's own
    # IHDR can catch a window resize that did not actually take effect.
    image.write_bytes(_png(width // 2, height))
    payload[0]["actual"]["screenshot_sha256"] = _sha256(image)
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="pixels"):
        finalize_scene(dataset, plan.scene_id, staging)

    # A file that is not a PNG at all is refused by the same check.
    image.write_bytes(b"not a png, but the recorded digest matches it")
    payload[0]["actual"]["screenshot_sha256"] = _sha256(image)
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="pixels"):
        finalize_scene(dataset, plan.scene_id, staging)


def test_finalize_refuses_overwrite_without_a_flag(tmp_path):
    dataset, staging, plan, _export = _prepared_dataset(tmp_path)
    finalize_scene(dataset, plan.scene_id, staging)
    with pytest.raises(DatasetSchemaError, match="overwrite"):
        finalize_scene(dataset, plan.scene_id, staging)
    finalize_scene(dataset, plan.scene_id, staging, force=True)


def test_finalize_refuses_a_dataset_inside_the_game_directory(tmp_path):
    dataset, _staging, plan, _export = _prepared_dataset(tmp_path)
    with pytest.raises(DatasetSchemaError, match="inside the staging game directory"):
        finalize_scene(dataset, plan.scene_id, dataset)


def test_finalize_rejects_world_identity_mismatch(tmp_path):
    dataset, staging, plan, export = _prepared_dataset(tmp_path)
    swapped = GroundTruthExport(scene_id=export.scene_id, world_identity="Some other world",
                                region=export.region, blocks=export.blocks)
    _write_mod_outputs(staging, plan, swapped)
    with pytest.raises(DatasetSchemaError, match="world identity"):
        finalize_scene(dataset, plan.scene_id, staging)


# --- mod contract fixture --------------------------------------------------


def test_mod_contract_fixture_parses_through_public_schemas(tmp_path):
    """A literal fixture, as the Java mod writes it, must parse without adjustment."""
    results_fixture = json.dumps([
        {"schema_version": 1, "frame_id": "view-house-a-00", "filename": "view-house-a-00.png",
         "captured": True, "screenshot": "screenshots/view-house-a-00.png", "settled_ticks": 20,
         "actual": {"position": [10.0, 64.0, 10.0], "yaw_degrees": 0.0, "pitch_degrees": -20.0,
                    "fov_degrees": 70.0, "gamma": 0.0, "hud_hidden": True,
                    "window": [1920, 1080], "time_of_day": 6000, "weather": "clear"},
         "errors": []}])
    parsed = [CaptureFrameResult.from_dict(entry) for entry in json.loads(results_fixture)]
    assert parsed[0].captured and parsed[0].actual["window"] == [1920, 1080]

    ground_truth_fixture = json.dumps({
        "schema_version": 1, "mod_contract_version": 1, "scene_id": "house-a",
        "world_identity": "Template 2", "region": {"min": [0, 60, 0], "max": [23, 83, 23]},
        "blocks": [{"pos": [1, 61, 1], "name": "minecraft:stone", "properties": {}}]})
    export = parse_ground_truth(json.loads(ground_truth_fixture))
    assert export.region.size == (24, 24, 24)
    assert volume_from_ground_truth(export).blocks[(1, 61, 1)].state.name == "minecraft:stone"

    wrong = json.loads(ground_truth_fixture)
    wrong["mod_contract_version"] = 7
    with pytest.raises(DatasetSchemaError, match="mod contract"):
        parse_ground_truth(wrong)


def test_plan_id_is_a_pure_function_of_its_jobs():
    plan = _plan()
    assert plan.plan_id == plan_id_for(plan.dataset_id, plan.scene_id, plan.jobs)
    other = build_plan(dataset_id="ds-test", scene_id="house-a", world_identity="Template 2",
                       region=_region(), seed=12, view_count=5,
                       staging_token=TEST_STAGING_TOKEN)
    assert other.plan_id != plan.plan_id


def test_camera_capture_requires_matching_intrinsics_and_positive_resolution():
    job = _plan().jobs[0]
    payload = job.to_dict()
    payload["resolution"] = [320, 240]
    with pytest.raises(DatasetSchemaError, match="intrinsics size"):
        CameraCapture.from_dict(payload)
    with pytest.raises(DatasetSchemaError, match="positive"):
        CameraCapture(frame_id="bad", camera=job.camera, resolution=(0, 240),
                      target=job.target)
