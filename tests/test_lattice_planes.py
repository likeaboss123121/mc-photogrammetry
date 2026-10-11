"""DS-29 synthetic controls for the image-only plane readings and signatures.

These are public-output controls over the shipped projection: a ray-cast
axis-aligned room with procedural block textures is rendered through
``camera.camera_projection_matrix`` conventions, and the readings/signatures are
checked for determinism, mask purity, the preregistered distance-alternative set,
the declared resource caps, and the structural-share ordering. No manifest,
reference litematic, frozen evaluation mask or annotation is read anywhere.
"""
from __future__ import annotations

import ast
import base64
import cv2
import inspect
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from mc_photogrammetry.reconstruction import (
    lattice, lattice_planes, plane_graph, region_growth,
)
from mc_photogrammetry.reconstruction.boundary_candidates import (
    PlaneProposal,
    PlaneProposalResult,
)
from mc_photogrammetry.reconstruction.calibration import CalibrationResult, axis_directions
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_rotation_matrix,
    intrinsics_for,
)
from mc_photogrammetry.vision.gui_mask import create_gui_mask

WIDTH, HEIGHT = 720, 480
ROOM = (0.0, 12.0, 0.0, 6.0, 0.0, 12.0)
CAMERA = ((3.6, 0.7, 3.6), 10.0, -25.0)

_SCENE: dict[str, object] = {}


def material_texture(material: int, size: int = 16) -> np.ndarray:
    val = np.full((size, size), 150.0 + 6.0 * (material % 4))
    shift = (material * 5) % 5
    val[4 + (shift % 2), :] = 128.0 + 6.0 * (material % 3)
    val[:, 11 - (shift % 2)] = 134.0 + 4.0 * (material % 2)
    val[0, :] = 25.0
    val[:, 0] = 25.0
    return val.astype(np.uint8)


def render_room(camera: CameraModel, width: int, height: int, room, *, materials=None,
                principal_point=None) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = room
    rotation = camera_rotation_matrix(camera)
    intrinsics = intrinsics_for(camera, width, height)
    fx, fy = intrinsics.focal_length_pixels
    cx, cy = intrinsics.principal_point_pixels if principal_point is None else principal_point
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    world = np.stack([(xs - cx) / fx, -(ys - cy) / fy, np.ones_like(xs, dtype=float)],
                     axis=-1) @ rotation
    eye = np.asarray(camera.position, float) + np.asarray([0.0, camera.eye_height, 0.0])
    best = np.full((height, width), np.inf)
    color = np.zeros((height, width), dtype=np.uint8)
    for axis, value in [(0, x0), (0, x1), (1, y0), (1, y1), (2, z0), (2, z1)]:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (value - eye[axis]) / world[..., axis]
        hit = np.isfinite(t) & (t > 1e-6) & (t < best)
        if not hit.any():
            continue
        point = eye[None, None, :] + np.where(hit, t, 0.0)[..., None] * world
        inside = np.ones((height, width), bool)
        bounds = {0: (x0, x1), 1: (y0, y1), 2: (z0, z1)}
        for other in range(3):
            if other == axis:
                continue
            lo, hi = bounds[other]
            inside &= (point[..., other] >= lo - 1e-6) & (point[..., other] <= hi + 1e-6)
        use = hit & inside
        if not use.any():
            continue
        others = [a for a in range(3) if a != axis]
        u, v = point[..., others[0]], point[..., others[1]]
        bx, by = np.floor(u).astype(int), np.floor(v).astype(int)
        chosen = np.zeros_like(bx) if materials is None else materials(bx, by, axis)
        tu = np.clip((np.mod(u, 1.0) * 16).astype(int), 0, 15)
        tv = np.clip((np.mod(v, 1.0) * 16).astype(int), 0, 15)
        for material in np.unique(chosen):
            tex = material_texture(int(material))
            pick = use & (chosen == material)
            if pick.any():
                color[pick] = tex[tv[pick], tu[pick]]
        best[use] = t[use]
    return color


def mixed_material(bx, by, axis):
    del axis
    return (bx * 3 + by * 5) % 3


def calibration(camera: CameraModel, width: int = WIDTH, height: int = HEIGHT,
                principal_point=None) -> CalibrationResult:
    intrinsics = intrinsics_for(camera, width, height)
    return CalibrationResult(
        focal_pixels=float(intrinsics.focal_length_pixels[0]),
        yaw_degrees=float(camera.yaw_degrees % 90.0),
        pitch_degrees=float(camera.pitch_degrees),
        width=width, height=height, inliers=100, segments=200,
        median_residual=0.5, roll_degrees=0.0, principal_point=principal_point,
    )


def _scene(materials=None) -> tuple[np.ndarray, np.ndarray, CameraModel]:
    key = "uniform" if materials is None else "mixed"
    if key not in _SCENE:
        position, yaw, pitch = CAMERA
        camera = CameraModel(position=position, yaw_degrees=yaw, pitch_degrees=pitch,
                             fov_degrees=70.0)
        image = render_room(camera, WIDTH, HEIGHT, ROOM, materials=materials)
        usable = ~create_gui_mask(HEIGHT, WIDTH)
        _SCENE[key] = (image, usable, camera)
    return _SCENE[key]  # type: ignore[return-value]


@pytest.mark.parametrize("along", (0, 1))
def test_rgb_mask_rectification_keeps_uint8_until_signed_profile(along):
    from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
    coordinate = np.arange(257)
    line = np.where(coordinate % 16 < 2, 255, 0).astype(np.uint8)
    grey = np.tile(line, (257, 1))
    if along:
        grey = grey.T.copy()
    rgb = np.repeat(grey[..., None], 3, axis=2)
    usable = np.ones(grey.shape, dtype=bool)
    usable[0, 0] = False
    before = rgb.tobytes(), usable.tobytes()
    reader = measure._reader_rgb(rgb)
    masked = lattice_planes._masked_grey(reader, usable)
    assert reader is rgb and masked.dtype == np.uint8 and masked[0, 0] == 0
    patch = lattice.rectify_plane(masked, 2, np.eye(3), 128., extent=1., samples=257,
                                 centre=(0., 0.), principal_point=(128., 128.))
    assert patch is not None and patch.dtype == np.uint8
    weights = lattice.rectify_plane(usable.astype(np.float32), 2, np.eye(3), 128.,
                                   extent=1., samples=257, principal_point=(128., 128.))
    for direction in (0, 1):
        expected = np.abs(np.diff(patch.astype(np.float64), axis=1-direction))
        assert np.max(expected) == 255
        np.testing.assert_array_equal(lattice._profile(patch, direction), expected.mean(axis=direction))
        np.testing.assert_array_equal(lattice._profile(patch, direction, weights),
                                      lattice._profile(patch.astype(np.float64), direction, weights))
    integer = lattice.estimate_period(patch, along=along, weights=weights)
    floating = lattice.estimate_period(patch.astype(np.float64), along=along, weights=weights)
    assert integer is not None and integer == floating
    assert integer.period_pixels == pytest.approx(16., rel=.02)
    assert (rgb.tobytes(), usable.tobytes()) == before


def _readings(materials=None, image=None, usable=None, frame="f0"):
    cached_image, cached_usable, camera = _scene(materials)
    diagnostics: dict = {}
    found = lattice_planes.frame_plane_readings(
        cached_image if image is None else image,
        cached_usable if usable is None else usable,
        calibration(camera), frame=frame, diagnostics=diagnostics)
    return found, diagnostics


def _signature_key(reading: lattice_planes.PlaneReading):
    return (
        reading.frame, reading.axis, reading.side, reading.period_pixels,
        reading.phase_pixels, reading.extent, reading.samples, reading.polygon,
        reading.coverage_cells, reading.structural_share, reading.confidence,
        tuple((a.blocks_per_period, a.distance_blocks, a.support) for a in reading.alternatives),
        reading.rectification.axis_matrix, reading.rectification.focal,
        reading.rectification.principal_point, reading.rectification.centre,
    )


def test_distance_alternative_set_is_exactly_the_preregistered_B():
    assert lattice_planes.DISTANCE_ALTERNATIVES == (
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0,
        2.0, 3.0, 4.0, 8.0, 16.0,
    )
    assert lattice_planes.SUPPORT_PRUNE == 0.5


def test_distance_alternatives_build_all_sub_block_readings_from_block_supports():
    estimate = lattice.PeriodEstimate(
        period_pixels=40.0, phase_pixels=3.0, strength=0.9,
        block_supports=((2, 1.0), (4, 1.0), (8, 1.0), (16, 1.0)), coarse_support=1.0)
    patch = np.zeros((64, 64), dtype=np.float32)
    alternatives = lattice_planes._distance_alternatives(
        estimate, extent=1.0, samples=512, patch=patch)
    values = [alternative.blocks_per_period for alternative in alternatives]
    assert values == [1.0, 1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0]
    world = 40.0 * lattice.rectified_pixel_step(1.0, 512)
    for alternative in alternatives:
        assert alternative.distance_blocks == pytest.approx(
            alternative.blocks_per_period / world)
        assert alternative.support >= lattice_planes.SUPPORT_PRUNE * 1.0


def test_distance_alternatives_prune_only_by_the_declared_support_floor():
    estimate = lattice.PeriodEstimate(
        period_pixels=30.0, phase_pixels=1.0, strength=0.5,
        block_supports=((2, 1.0), (4, 0.49), (8, 0.5), (16, 0.1)), coarse_support=0.5)
    patch = np.zeros((32, 32), dtype=np.float32)
    alternatives = lattice_planes._distance_alternatives(
        estimate, extent=1.0, samples=256, patch=patch)
    kept = {alternative.blocks_per_period for alternative in alternatives}
    assert 1.0 in kept
    assert 0.5 in kept           # 1/2 support 1.0
    assert 0.25 not in kept      # 1/4 support 0.49 < 0.5 * 1.0
    assert 0.125 in kept         # 1/8 support 0.50 == floor
    assert 0.0625 not in kept    # 1/16 support 0.10 < floor
    assert len(alternatives) == len(kept)


def test_higher_order_support_uses_the_winning_direction_and_proposal_weights(monkeypatch):
    height, width = 80, 128
    x = np.arange(width)
    patch = np.empty((height, width), dtype=np.float32)
    patch[:40] = (((x % 8) < 1).astype(np.float32) * 100.0)[None, :]
    patch[40:] = (((x % 11) < 1).astype(np.float32) * 100.0)[None, :]
    all_pixels = np.ones_like(patch)
    top_only = np.zeros_like(patch)
    top_only[:40] = 1.0
    unweighted = lattice_planes._autocorrelation_support(
        patch, 8.0, 2, along=0, weights=all_pixels)
    weighted = lattice_planes._autocorrelation_support(
        patch, 8.0, 2, along=0, weights=top_only)
    assert weighted > unweighted + 0.25

    calls = []

    def capture(_patch, _period, multiple, *, along=0, weights=None):
        calls.append((multiple, along, weights))
        return 0.75

    monkeypatch.setattr(lattice_planes, "_autocorrelation_support", capture)
    estimate = lattice.PeriodEstimate(
        period_pixels=8.0, phase_pixels=0.0, strength=0.8,
        block_supports=((2, 0.8),), coarse_support=0.0)
    alternatives = lattice_planes._distance_alternatives(
        estimate, extent=1.0, samples=128, patch=patch, weights=top_only,
        profile_direction=1)
    assert alternatives
    assert len(calls) == 5
    assert all(direction == 1 and weights is top_only for _, direction, weights in calls)
    assert [multiple for multiple, _, _ in calls] == [2, 3, 4, 8, 16]


def test_distance_alternatives_abstain_on_invalid_world_scale():
    estimate = lattice.PeriodEstimate(
        period_pixels=float("nan"), phase_pixels=0.0, strength=1.0)
    patch = np.zeros((16, 16), dtype=np.float32)
    assert lattice_planes._distance_alternatives(
        estimate, extent=1.0, samples=16, patch=patch) == ()
    estimate = lattice.PeriodEstimate(period_pixels=8.0, phase_pixels=0.0, strength=1.0)
    assert lattice_planes._distance_alternatives(
        estimate, extent=0.0, samples=16, patch=patch) == ()
    assert lattice_planes._distance_alternatives(
        estimate, extent=1.0, samples=1, patch=patch) == ()


@pytest.mark.parametrize("samples", (3, 9, 17, 64))
def test_distance_alternative_distance_uses_endpoint_pixel_step(samples):
    period_pixels = 4.25
    extent = 1.375
    estimate = lattice.PeriodEstimate(
        period_pixels=period_pixels, phase_pixels=0.0, strength=0.9)
    alternatives = lattice_planes._distance_alternatives(
        estimate, extent=extent, samples=samples,
        patch=np.zeros((16, 16), dtype=np.float32))
    block = next(alternative for alternative in alternatives
                 if alternative.blocks_per_period == 1.0)
    expected = 1.0 / (period_pixels * lattice.rectified_pixel_step(extent, samples))
    assert block.distance_blocks == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("samples", (5, 9, 17))
def test_signature_grid_scale_uses_canonical_endpoint_scaled_alternative(monkeypatch, samples):
    extent = 1.0
    period_pixels = 2.0
    distance = 1.0 / (period_pixels * lattice.rectified_pixel_step(extent, samples))
    reading = lattice_planes.PlaneReading(
        frame="endpoint", axis=2, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, distance, 1.0),),
        period_pixels=period_pixels, phase_pixels=(0.0, 0.0),
        extent=extent, samples=samples, polygon=None, coverage_cells=16,
        structural_share=0.5, confidence=0.8,
        rectification=lattice_planes.PlaneRectification(
            axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            focal=100.0, principal_point=(8.0, 8.0), centre=(0.0, 0.0)))
    calls = []

    def fake_rectify(_image, _axis, _matrix, _focal, *, extent, samples,
                     centre=(0.0, 0.0), principal_point=None, side=1.0):
        calls.append((extent, samples, centre, principal_point, side))
        return np.ones((samples, samples), dtype=np.float32)

    monkeypatch.setattr(lattice, "rectify_plane", fake_rectify)
    signature = lattice_planes.plane_signature(
        np.zeros((16, 16, 3), dtype=np.uint8), reading, 1.0)
    expected_cells = int(np.ceil(2.0 * extent * distance))
    expected_unit_extent = (expected_cells / 2.0) / distance
    assert signature.cells.shape == (expected_cells, expected_cells)
    assert signature.blocks_per_period == 1.0
    assert calls and all(call[0] == pytest.approx(expected_unit_extent) for call in calls)
    assert all(call[1] == expected_cells * lattice_planes.SIGNATURE_PIXELS_PER_CELL
               for call in calls)


def test_bounded_measure_returns_resolution_of_final_executed_round(monkeypatch):
    measured_shapes = []
    simulated_periods = iter((5.0, 4.9, 4.8))

    def fake_rectify(image, axis, axis_matrix, focal, *, extent, samples,
                     centre=(0.0, 0.0), side=1.0, **kwargs):
        del image, axis, axis_matrix, focal, extent, centre, side, kwargs
        measured_shapes.append(samples)
        return np.ones((samples, samples), dtype=np.float32)

    def fake_period(patch, *, weights=None, minimum_period=4.0, tolerance=0.12):
        del patch, weights, minimum_period, tolerance
        return lattice.PeriodEstimate(
            period_pixels=next(simulated_periods), phase_pixels=0.0, strength=0.9), 1

    monkeypatch.setattr(lattice, "rectify_plane", fake_rectify)
    monkeypatch.setattr(lattice, "_plane_period_with_direction", fake_period)
    estimate, samples, patch, weights, direction = lattice_planes._bounded_measure(
        np.zeros((8, 8), dtype=np.uint8), 0, np.eye(3), 100.0,
        mask=np.ones((8, 8), dtype=np.float32), centre=(0.0, 0.0),
        principal_point=(3.0, 4.0), extent=1.0, side=1.0,
        max_rectified_side=2048)

    # Requests are 400 -> 800 -> 1633 -> 2048; the last request is never
    # executed because the adaptive loop is capped at three rounds.
    assert measured_shapes == [400, 400, 800, 800, 1633, 1633]
    assert estimate is not None and estimate.period_pixels == pytest.approx(4.8)
    assert direction == 1
    assert samples == 1633
    assert patch.shape == (samples, samples)
    assert weights.shape == (samples, samples)


def test_off_center_principal_is_shared_by_window_measure_weights_edge_and_signature(monkeypatch):
    principal = (WIDTH / 2.0 + 37.0, HEIGHT / 2.0 - 23.0)
    camera = CameraModel(position=CAMERA[0], yaw_degrees=CAMERA[1],
                         pitch_degrees=CAMERA[2], fov_degrees=70.0)
    grey = render_room(camera, WIDTH, HEIGHT, ROOM, materials=mixed_material,
                       principal_point=principal)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    usable = ~create_gui_mask(HEIGHT, WIDTH)
    result = calibration(camera, principal_point=principal)
    proposals = region_growth.propose_image_planes_v2(rgb, usable, result, max_candidates=12)
    assert proposals.status == "ok" and proposals.proposals

    window_calls = []
    rectify_calls = []
    real_window = lattice.plane_window
    real_rectify = lattice.rectify_plane

    def capture_window(mask, axis, axis_matrix, focal, *, principal_point=None, **kwargs):
        window_calls.append(principal_point)
        return real_window(mask, axis, axis_matrix, focal,
                           principal_point=principal_point, **kwargs)

    def capture_rectify(image, axis, axis_matrix, focal, *args,
                        principal_point=None, **kwargs):
        rectify_calls.append((id(image), np.asarray(image).ndim, principal_point))
        return real_rectify(image, axis, axis_matrix, focal, *args,
                            principal_point=principal_point, **kwargs)

    monkeypatch.setattr(lattice, "plane_window", capture_window)
    monkeypatch.setattr(lattice, "rectify_plane", capture_rectify)
    readings = lattice_planes.frame_plane_readings_from_proposals(
        rgb, usable, result, proposals, frame="off-center")
    assert readings, "off-center synthetic wall should still produce a plane reading"
    assert window_calls and all(point == principal for point in window_calls)
    reading = readings[0]
    assert reading.rectification is not None
    assert reading.rectification.principal_point == principal

    # Reading measurement includes distinct proposal-mask and edge-weight
    # rectifications. Signature then rectifies RGB and coverage with the same
    # recorded calibration principal. Provenance adds a third rectification
    # for the exact source-region mask.
    measurement_rectifications = list(rectify_calls)
    assert measurement_rectifications
    assert len({call[0] for call in measurement_rectifications}) >= 3
    assert all(call[2] == principal for call in measurement_rectifications)
    signature = lattice_planes.plane_signature(
        rgb, reading, reading.alternatives[0].blocks_per_period, usable=usable)
    assert signature.cells.size > 0
    signature_rectifications = rectify_calls[len(measurement_rectifications):]
    assert len(signature_rectifications) == 3
    assert all(call[2] == principal for call in signature_rectifications)


def test_rectification_homography_round_trips_the_shipped_projection():
    camera = CameraModel(position=(2.0, 1.0, -3.0), yaw_degrees=31.0, pitch_degrees=-14.0)
    result = calibration(camera)
    axis_matrix = np.asarray(
        axis_directions(result.yaw_degrees, result.pitch_degrees, result.roll_degrees))
    principal = (WIDTH / 2.0, HEIGHT / 2.0)
    for axis in range(3):
        for side in (-1.0, 1.0):
            distance = 7.5
            homography = lattice_planes.rectification_homography(
                axis, axis_matrix, result.focal_pixels, principal, side, distance)
            for u_unit, v_unit in ((0.0, 0.0), (0.31, -0.22), (-0.5, 0.4)):
                camera_point = lattice.plane_coordinates_to_camera_point(
                    axis, axis_matrix, side, distance, u_unit, v_unit)
                expected = lattice.camera_point_to_pixel(camera_point, result.focal_pixels, principal)
                if expected is None:
                    # The window extends behind the camera for this axis/side;
                    # the homography must agree that the point is invalid.
                    homogeneous = homography @ np.asarray(
                        (u_unit * distance, v_unit * distance, 1.0))
                    assert homogeneous[2] <= 1e-6
                    continue
                homogeneous = homography @ np.asarray(
                    (u_unit * distance, v_unit * distance, 1.0))
                observed = (homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2])
                assert observed[0] == pytest.approx(expected[0], abs=1e-6)
                assert observed[1] == pytest.approx(expected[1], abs=1e-6)


def test_rectification_homography_rejects_non_positive_distance():
    axis_matrix = np.eye(3)
    with pytest.raises(ValueError):
        lattice_planes.rectification_homography(0, axis_matrix, 500.0, (10.0, 10.0), 1.0, 0.0)


def test_frame_plane_readings_are_deterministic_and_bounded():
    readings, diagnostics = _readings(mixed_material)
    assert readings, "the rendered room should yield at least one reading"
    again, _ = _readings(mixed_material)
    assert [_signature_key(r) for r in readings] == [_signature_key(r) for r in again]
    assert len(readings) <= lattice_planes.MAX_READINGS
    per_axis_side: dict[tuple[int, float], int] = {}
    for reading in readings:
        assert 0 <= reading.axis <= 2
        assert reading.side in (-1.0, 1.0)
        assert reading.alternatives
        assert reading.samples <= lattice_planes.MAX_RECTIFIED_SIDE
        assert reading.coverage_cells >= 1
        assert 0.0 <= reading.structural_share <= 1.0
        assert 0.0 <= reading.confidence <= 1.0
        key = (reading.axis, reading.side)
        per_axis_side[key] = per_axis_side.get(key, 0) + 1
        assert per_axis_side[key] <= lattice_planes.MAX_READINGS_PER_AXIS_SIDE
    assert diagnostics["segments"] > 0


def test_frame_plane_readings_are_pure_outside_the_usable_mask():
    readings, _ = _readings(mixed_material)
    cached_image, cached_usable, camera = _scene(mixed_material)
    rng = np.random.default_rng(29)
    mutated = cached_image.copy()
    noise = rng.integers(0, 256, size=mutated.shape, dtype=np.uint8)
    mutated[~cached_usable] = noise[~cached_usable]
    changed, _ = _readings(mixed_material, image=mutated)
    assert [_signature_key(r) for r in changed] == [_signature_key(r) for r in readings]


def _rgb_scene_and_proposals():
    image, usable, camera = _scene(mixed_material)
    rgb = np.repeat(image[..., None], 3, axis=2)
    result = calibration(camera)
    proposals = region_growth.propose_image_planes_v2(rgb, usable, result, max_candidates=12)
    assert proposals.status == "ok" and proposals.proposals
    return rgb, usable, result, proposals


def _v1_region_control(mask=None, sources=("synthetic-region",)):
    shape = (8, 8)
    if mask is None:
        mask = np.zeros(shape, dtype=bool)
        mask[0:6, 0:2] = True
        mask[4:6, 0:5] = True
        mask[6:8, 6:8] = True  # disconnected island
        mask[4, 2] = False     # opening inside the bounding rectangle
    usable = np.ones(shape, dtype=bool)
    usable[1, 0] = False       # excluded source pixel inside the L
    polygon = ((0.0, 0.0), (7.0, 0.0), (7.0, 7.0), (0.0, 7.0))
    proposal = PlaneProposal(
        axis=0, side=-1, polygon=polygon, mask=np.asarray(mask).copy(),
        strength=0.8, sources=tuple(sources))
    calibration_result = CalibrationResult(
        focal_pixels=100.0, yaw_degrees=0.0, pitch_degrees=0.0,
        width=shape[1], height=shape[0], inliers=12, segments=12,
        median_residual=0.0, roll_degrees=0.0, principal_point=(4.0, 4.0))
    return proposal, usable, calibration_result, polygon


def _controlled_v1_readings(monkeypatch, proposals, usable, calibration_result, *, frame,
                            window_side=None, structural_share=None, max_readings=None):
    height, width = usable.shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    segments = np.asarray(
        [(0, index, width - 1, height - 1 - index) for index in range(12)],
        dtype=np.float32)
    estimate = lattice.PeriodEstimate(
        period_pixels=16.0, phase_pixels=0.0, strength=0.8,
        block_supports=((2, 0.8),), coarse_support=0.8)
    patch = np.ones((128, 128), dtype=np.float32)

    monkeypatch.setattr(lattice_planes, "detect_segments", lambda _grey: segments)
    monkeypatch.setattr(
        lattice_planes, "structural_weights",
        lambda _grey, found: np.ones(len(found), dtype=np.float32))
    monkeypatch.setattr(
        lattice_planes, "_weight_image",
        lambda _segments, _weights, h, w: np.ones((h, w), dtype=np.float32))
    monkeypatch.setattr(lattice, "plane_window", lambda mask, *_args, **_kwargs:
                        (0.0, 0.0, 1.0, 1.0 if window_side is None else window_side(mask)))
    def measured_window(*_args, **_kwargs):
        region_growth._scalar_period_finish(
            "period_measurement", "target_period_reached", attempted_rounds=1)
        return estimate, 128, patch, patch.copy(), 0

    monkeypatch.setattr(lattice_planes, "_bounded_measure", measured_window)
    monkeypatch.setattr(
        lattice_planes, "_distance_alternatives",
        lambda *_args, **_kwargs: (
            lattice_planes.DistanceAlternative(1.0, 4.0, 0.9),
            lattice_planes.DistanceAlternative(2.0, 8.0, 0.8),
        ))
    monkeypatch.setattr(lattice_planes, "_structural_share", structural_share or (lambda *_args: 0.5))
    monkeypatch.setattr(lattice_planes, "_phase_pixels", lambda *_args: (0.0, 0.0))

    def sample_mask(source, _axis, _matrix, _focal, *, samples, **_kwargs):
        source = np.asarray(source)
        if source.ndim == 3 or np.all(source > 0.5):
            return np.ones((samples, samples), dtype=np.float32)
        rows = np.minimum(np.arange(samples) * source.shape[0] // samples,
                          source.shape[0] - 1)
        columns = np.minimum(np.arange(samples) * source.shape[1] // samples,
                             source.shape[1] - 1)
        return source[np.ix_(rows, columns)].astype(np.float32)

    monkeypatch.setattr(lattice, "rectify_plane", sample_mask)
    result = PlaneProposalResult("ok", "synthetic V1 controlled reader", tuple(proposals))
    readings = lattice_planes.frame_plane_readings_from_proposals(
        image, usable, calibration_result, result, frame=frame,
        max_readings=(lattice_planes.MAX_READINGS if max_readings is None else max_readings))
    return image, readings


def test_additive_proposal_adapter_preserves_legacy_readings_and_arity(monkeypatch):
    assert tuple(inspect.signature(lattice_planes.frame_plane_readings).parameters) == (
        "image", "usable", "calibration_result", "frame", "max_readings",
        "min_structural_share", "max_rectified_side", "diagnostics",
    )
    assert tuple(inspect.signature(
        lattice_planes.frame_plane_readings_from_proposals).parameters) == (
            "image", "usable", "calibration_result", "proposal_result", "frame",
            "max_readings", "min_structural_share", "max_rectified_side", "diagnostics",
        )

    rgb, usable, result, _c4 = _rgb_scene_and_proposals()
    original = lattice_planes._candidate_regions
    captured: dict = {}

    def capture_candidates(*args, **kwargs):
        candidates = original(*args, **kwargs)
        captured["candidates"] = candidates
        return candidates

    monkeypatch.setattr(lattice_planes, "_candidate_regions", capture_candidates)
    legacy = lattice_planes.frame_plane_readings(rgb, usable, result, frame="parity")
    proposal_result = PlaneProposalResult(
        status="ok",
        reason="synthetic legacy-candidate parity control",
        proposals=tuple(
            PlaneProposal(
                axis=axis,
                side=-1,  # Deliberately ignored: plane_window derives geometric side.
                polygon=polygon,
                mask=mask,
                strength=score,
                sources=("parity-control",),
            )
            for axis, mask, polygon, score in captured["candidates"]
        ),
    )
    proposed = lattice_planes.frame_plane_readings_from_proposals(
        rgb, usable, result, proposal_result, frame="parity")
    assert legacy
    assert [_signature_key(reading) for reading in proposed] == [
        _signature_key(reading) for reading in legacy]


def test_proposal_adapter_keeps_distinct_overlapping_same_axis_side_regions():
    rgb, usable, result, generated = _rgb_scene_and_proposals()
    source = next(proposal for proposal in generated.proposals if proposal.axis == 2)
    first_mask = np.asarray(source.mask).copy()
    second_mask = first_mask.copy()
    second_mask[:, :100] = False
    first = PlaneProposal(
        axis=2, side=-1, polygon=source.polygon, mask=first_mask,
        strength=0.8, sources=("first",),
    )
    second = PlaneProposal(
        axis=2, side=1, polygon=source.polygon, mask=second_mask,
        strength=0.7, sources=("overlap",),
    )
    readings = lattice_planes.frame_plane_readings_from_proposals(
        rgb, usable, result,
        PlaneProposalResult("ok", "overlap control", (first, second)), frame="overlap")
    assert len(readings) == 2
    assert [reading.axis for reading in readings] == [2, 2]
    assert [reading.side for reading in readings] == [1.0, 1.0]
    assert readings[0].extent != readings[1].extent


def test_proposal_adapter_is_usable_mask_pure_and_order_deterministic():
    rgb, usable, result, proposals = _rgb_scene_and_proposals()
    original_masks = [np.asarray(proposal.mask).copy() for proposal in proposals.proposals]
    baseline = lattice_planes.frame_plane_readings_from_proposals(
        rgb, usable, result, proposals, frame="pure")
    reversed_result = PlaneProposalResult(
        "ok", proposals.reason, tuple(reversed(proposals.proposals)), proposals.stats)
    reordered = lattice_planes.frame_plane_readings_from_proposals(
        rgb, usable, result, reversed_result, frame="pure")
    assert [_signature_key(reading) for reading in reordered] == [
        _signature_key(reading) for reading in baseline]

    rng = np.random.default_rng(44)
    poisoned_image = rgb.copy()
    noise = rng.integers(0, 256, size=rgb.shape, dtype=np.uint8)
    poisoned_image[~usable] = noise[~usable]
    poisoned_proposals = []
    for proposal in proposals.proposals:
        mask = np.asarray(proposal.mask).copy()
        mask[~usable] = True
        poisoned_proposals.append(PlaneProposal(
            axis=proposal.axis, side=-proposal.side, polygon=proposal.polygon,
            mask=mask, strength=proposal.strength, sources=proposal.sources,
            coverage=proposal.coverage,
        ))
    poison_result = PlaneProposalResult(
        "ok", proposals.reason, tuple(poisoned_proposals), proposals.stats)
    changed = lattice_planes.frame_plane_readings_from_proposals(
        poisoned_image, usable, result, poison_result, frame="pure")
    assert [_signature_key(reading) for reading in changed] == [
        _signature_key(reading) for reading in baseline]
    for original_mask, proposal in zip(original_masks, proposals.proposals):
        np.testing.assert_array_equal(proposal.mask, original_mask)


def test_region_evidence_round_trips_l_opening_and_disconnected_mask_exactly():
    proposal, usable, _calibration_result, polygon = _v1_region_control()
    exact = np.asarray(proposal.mask, dtype=bool) & usable
    evidence = lattice_planes._make_region_evidence(
        exact, proposal.axis, polygon, proposal.sources)
    duplicate = lattice_planes._make_region_evidence(
        exact.copy(), proposal.axis, polygon, proposal.sources)
    decoded = lattice_planes._decode_region_evidence(
        evidence, proposal.axis, polygon)
    assert evidence == duplicate
    np.testing.assert_array_equal(decoded, exact)
    assert evidence.pixel_count == int(exact.sum())
    assert evidence.encoding == "packbits-base64" and evidence.bit_order == "little"
    assert evidence.sources == ("synthetic-region",)
    assert json.dumps(evidence.to_record(), sort_keys=True) == json.dumps(
        duplicate.to_record(), sort_keys=True)

    other_axis = lattice_planes._make_region_evidence(exact, 1, polygon, proposal.sources)
    other_polygon = lattice_planes._make_region_evidence(
        exact, proposal.axis, ((0.0, 0.0), (6.0, 0.0), (7.0, 7.0), (0.0, 7.0)),
        proposal.sources)
    other_source = lattice_planes._make_region_evidence(
        exact, proposal.axis, polygon, ("other-source",))
    assert len({evidence.source_region_id, other_axis.source_region_id,
                other_polygon.source_region_id, other_source.source_region_id}) == 4
    cyclic_polygon = (polygon[2], polygon[3], polygon[0], polygon[1])
    reversed_polygon = tuple(reversed(polygon))
    assert lattice_planes._make_region_evidence(
        exact, proposal.axis, cyclic_polygon, proposal.sources) == evidence
    assert lattice_planes._make_region_evidence(
        exact, proposal.axis, reversed_polygon, proposal.sources) == evidence

    # 7x11 leaves five useful bits in the final pack byte; all three high
    # padding bits must remain zero and are rejected if altered.
    odd = np.zeros((9, 14), dtype=bool)
    odd[1:8, 2:13] = True
    odd[3, 6] = False
    odd_polygon = ((2.0, 1.0), (12.0, 1.0), (12.0, 7.0), (2.0, 7.0))
    odd_evidence = lattice_planes._make_region_evidence(
        odd, 0, odd_polygon, ("odd-crop",))
    assert odd_evidence.crop_shape == (7, 11)
    packed = bytearray(base64.b64decode(odd_evidence.packed_mask_base64))
    packed[-1] |= 1 << ((7 * 11) % 8)
    tampered = replace(
        odd_evidence,
        packed_mask_base64=base64.b64encode(packed).decode("ascii"))
    with pytest.raises(ValueError, match="padding bits"):
        lattice_planes._decode_region_evidence(tampered, 0, odd_polygon)


def test_proposal_reading_carries_exact_mask_and_b_specific_coverage_counts(monkeypatch):
    proposal, usable, calibration_result, polygon = _v1_region_control()
    image, readings = _controlled_v1_readings(
        monkeypatch, (proposal,), usable, calibration_result, frame="v1")
    assert len(readings) == 1
    reading = readings[0]
    assert reading.region_provenance_status == "available"
    assert reading.region_evidence is not None
    expected_mask = np.asarray(proposal.mask, dtype=bool) & usable
    decoded = lattice_planes._decode_region_evidence(
        reading.region_evidence, reading.axis, reading.polygon)
    np.testing.assert_array_equal(decoded, expected_mask)
    assert reading.region_evidence.source_region_id
    assert reading.phase_evidence is not None
    assert reading.phase_evidence.source_region_id == reading.region_evidence.source_region_id
    assert reading.phase_evidence.patch_centre_uv == reading.rectification.centre
    assert reading.phase_evidence.period_pixels == reading.period_pixels
    assert reading.phase_evidence.extent == reading.extent
    assert reading.phase_evidence.samples == reading.samples
    assert reading.phase_pixels == tuple(
        value if value is not None else 0.0
        for value in reading.phase_evidence.phase_uv_pixels)
    with pytest.raises(ValueError, match="not bound"):
        replace(reading, phase_evidence=replace(
            reading.phase_evidence, source_region_id="f" * 64))
    assert reading.region_evidence.image_shape == usable.shape
    assert reading.region_evidence.bbox_xyxy == (0, 0, 8, 8)

    second_image, second_readings = _controlled_v1_readings(
        monkeypatch, (proposal,), usable, calibration_result, frame="v1")
    assert second_image.shape == image.shape
    assert second_readings[0].region_evidence == reading.region_evidence
    assert json.dumps(reading.region_evidence.to_record(), sort_keys=True) == json.dumps(
        second_readings[0].region_evidence.to_record(), sort_keys=True)

    footprints = []
    for alternative in reading.alternatives:
        signature = lattice_planes.plane_signature(
            image, reading, alternative.blocks_per_period, usable=usable)
        phase_stripped = lattice_planes.plane_signature(
            image, replace(reading, phase_evidence=None),
            alternative.blocks_per_period, usable=usable)
        np.testing.assert_array_equal(signature.cells, phase_stripped.cells)
        np.testing.assert_array_equal(signature.interior, phase_stripped.interior)
        np.testing.assert_array_equal(signature.coverage, phase_stripped.coverage)
        footprint = signature.region_footprint
        assert footprint is not None
        assert footprint.source_region_id == reading.region_evidence.source_region_id
        assert footprint.blocks_per_period == alternative.blocks_per_period
        assert footprint.origin_uv == signature.origin_uv
        assert footprint.grid_shape == signature.cells.shape
        assert footprint.total_samples_per_cell == footprint.subsample_factor ** 2
        inside = np.frombuffer(
            footprint.inside_region_sample_counts, dtype=np.uint8).reshape(
                footprint.grid_shape)
        valid = np.frombuffer(
            footprint.valid_sample_counts, dtype=np.uint8).reshape(
                footprint.grid_shape)
        assert np.all(inside <= valid)
        assert np.all(valid <= footprint.total_samples_per_cell)
        record_a = json.dumps(footprint.to_record(), sort_keys=True)
        record_b = json.dumps(signature.region_footprint.to_record(), sort_keys=True)
        assert record_a == record_b
        footprints.append((signature, inside, valid))

    with pytest.raises(ValueError, match="retained reading alternative"):
        lattice_planes.plane_signature(image, reading, 0.75, usable=usable)

    with pytest.raises(ValueError, match="explicit usable mask"):
        lattice_planes.plane_signature(
            image, reading, reading.alternatives[0].blocks_per_period, usable=None)

    first, second = footprints
    assert first[0].blocks_per_period == reading.alternatives[0].blocks_per_period
    assert second[0].blocks_per_period == reading.alternatives[1].blocks_per_period
    assert first[0].region_footprint.grid_shape == (8, 8)
    assert second[0].region_footprint.grid_shape == (16, 16)
    assert first[1][0, 0] == 64             # L's vertical/horizontal arm
    assert first[1][4, 2] == 0              # opening is not bbox-filled
    assert first[1][6, 6] == 64             # disconnected island is retained
    assert first[1][0, 2] == 0              # empty between separate supports
    assert first[1][1, 0] == first[2][1, 0] == 0  # usable exclusion survives
    assert not np.array_equal(first[1], second[1])


def test_periodic_energy_phase_evidence_uses_u_columns_v_rows_and_zero_is_measured():
    patch = np.zeros((12, 16), dtype=np.float64)
    patch[np.ix_((1, 5, 9), (0, 4, 8, 12))] = 1.0
    evidence = lattice_planes._periodic_energy_phase_evidence(
        patch, 4.0, centre=(2.5, -1.25), extent=3.0, samples=16)
    assert evidence.phase_uv_pixels == pytest.approx((0.0, 1.0))
    assert evidence.unavailable_reason_uv == (None, None)
    assert evidence.response_magnitude_uv == pytest.approx((12.0, 12.0))
    assert evidence.profile_mass_uv == pytest.approx((12.0, 12.0))
    assert evidence.concentration_uv == pytest.approx((1.0, 1.0))
    assert lattice_planes._phase_pixels(patch, 4.0) == pytest.approx((0.0, 1.0))

    reading = lattice_planes.PlaneReading(
        frame="phase", axis=0, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 2.0, 1.0),),
        period_pixels=4.0, phase_pixels=(0.0, 1.0), extent=3.0, samples=16,
        polygon=None, coverage_cells=4, structural_share=0.5, confidence=0.8,
        rectification=lattice_planes.PlaneRectification(
            axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            focal=100.0, principal_point=(8.0, 8.0), centre=(2.5, -1.25)),
        phase_evidence=evidence)
    assert reading.phase_evidence == evidence
    with pytest.raises(ValueError, match="not bound"):
        replace(reading, phase_evidence=replace(evidence, samples=15))
    with pytest.raises(ValueError, match="concentration"):
        replace(evidence, concentration_uv=(1.01, 1.0))


def test_periodic_energy_phase_evidence_has_explicit_unavailable_reasons_and_legacy_sentinel():
    uniform = np.ones((12, 16), dtype=np.float64)
    zero_resultant = lattice_planes._periodic_energy_phase_evidence(
        uniform, 4.0, centre=(0.0, 0.0), extent=1.0, samples=16)
    assert zero_resultant.phase_uv_pixels == (None, None)
    assert zero_resultant.unavailable_reason_uv == ("zero_resultant", "zero_resultant")
    assert zero_resultant.response_magnitude_uv[0] <= 1e-12
    assert lattice_planes._phase_pixels(uniform, 4.0) == (0.0, 0.0)

    nonzero = np.zeros((8, 8), dtype=np.float64)
    nonzero[::2, ::2] = 1.0
    short_period = lattice_planes._periodic_energy_phase_evidence(
        nonzero, 1.0, centre=(0.0, 0.0), extent=1.0, samples=8)
    assert short_period.unavailable_reason_uv == (
        "insufficient_period", "insufficient_period")
    empty = lattice_planes._periodic_energy_phase_evidence(
        np.zeros((8, 8)), 4.0, centre=(0.0, 0.0), extent=1.0, samples=8)
    assert empty.unavailable_reason_uv == ("empty_profile", "empty_profile")
    nonfinite = np.zeros((8, 8))
    nonfinite[0, 0] = np.nan
    unavailable = lattice_planes._periodic_energy_phase_evidence(
        nonfinite, 4.0, centre=(0.0, 0.0), extent=1.0, samples=8)
    assert unavailable.unavailable_reason_uv == (
        "nonfinite_profile", "nonfinite_profile")
    assert unavailable.response_magnitude_uv == (None, None)
    assert lattice_planes._phase_pixels(nonfinite, 4.0) == (0.0, 0.0)

    with pytest.raises(ValueError, match="unavailable"):
        replace(zero_resultant, phase_uv_pixels=(0.0, None))
    with pytest.raises(ValueError, match="zero-resultant"):
        replace(zero_resultant, response_magnitude_uv=(2e-12, 0.0),
                concentration_uv=(2e-12 / 16.0, 0.0))


def test_periodic_energy_mass_response_boundaries_and_near_period_canonicalization(monkeypatch):
    below_or_equal_mass = lattice_planes._periodic_energy_axis(
        np.asarray([1e-9]), 4.0)
    assert below_or_equal_mass[4] == "empty_profile"
    above_mass = lattice_planes._periodic_energy_axis(
        np.asarray([1.0001e-9]), 4.0)
    assert above_mass[0] == 0.0
    assert above_mass[1] > 1e-12

    monkeypatch.setattr(lattice_planes.supported, "_periodic_energy_resultant",
                        lambda _profile, _offsets, _period: complex(1e-12, 0.0))
    exact_response = lattice_planes._periodic_energy_axis(np.asarray([2.0]), 4.0)
    assert exact_response[4] == "zero_resultant"
    monkeypatch.setattr(lattice_planes.supported, "_periodic_energy_resultant",
                        lambda _profile, _offsets, _period: complex(1.0001e-12, 0.0))
    above_response = lattice_planes._periodic_energy_axis(np.asarray([2.0]), 4.0)
    assert above_response[4] is None and above_response[0] == 0.0

    monkeypatch.setattr(lattice_planes.supported, "_periodic_energy_resultant",
                        lambda _profile, _offsets, _period: complex(1.0, 1e-13))
    near_period = lattice_planes._periodic_energy_axis(np.asarray([2.0]), 4.0)
    assert near_period[4] is None and near_period[0] == 0.0

    assert lattice_planes._periodic_energy_axis(np.asarray([1.0]), 0.0)[4] == "invalid_period"
    assert lattice_planes._periodic_energy_axis(np.asarray([-1.0]), 4.0)[4] == "invalid_profile"


@pytest.mark.parametrize("offset", (-1e-9, 1e-9))
def test_region_signature_canonicalizes_nearby_b_to_exact_alternative(monkeypatch, offset):
    proposal, usable, calibration_result, _polygon = _v1_region_control()
    image, readings = _controlled_v1_readings(
        monkeypatch, (proposal,), usable, calibration_result, frame="b-tolerance")
    reading = readings[0]
    alternative = reading.alternatives[0]
    exact = lattice_planes.plane_signature(
        image, reading, alternative.blocks_per_period, usable=usable)
    nearby = lattice_planes.plane_signature(
        image, reading, alternative.blocks_per_period + offset, usable=usable)
    assert nearby.blocks_per_period == alternative.blocks_per_period
    assert nearby.region_footprint.blocks_per_period == alternative.blocks_per_period
    assert nearby.origin_uv == exact.origin_uv
    np.testing.assert_array_equal(nearby.cells, exact.cells)
    np.testing.assert_array_equal(
        nearby.region_footprint.inside_region_sample_counts,
        exact.region_footprint.inside_region_sample_counts)

    duplicate = replace(reading, alternatives=(alternative, alternative))
    with pytest.raises(ValueError, match="exactly one"):
        lattice_planes.plane_signature(
            image, duplicate, alternative.blocks_per_period, usable=usable)


def test_malformed_region_evidence_never_becomes_available(monkeypatch):
    proposal, usable, calibration_result, polygon = _v1_region_control()
    evidence = lattice_planes._make_region_evidence(
        np.asarray(proposal.mask) & usable, proposal.axis, polygon, proposal.sources)
    image, readings = _controlled_v1_readings(
        monkeypatch, (proposal,), usable, calibration_result, frame="malformed")
    reading = readings[0]
    assert reading.region_provenance_status == "available"

    packed = bytearray(base64.b64decode(evidence.packed_mask_base64))
    # A valid payload with a leading empty row and column must be rejected as
    # a nonminimal crop before digest/source-ID checks.
    inner = np.asarray(proposal.mask, dtype=bool)[1:7, 1:7]
    nonminimal_crop = np.zeros((7, 7), dtype=bool)
    nonminimal_crop[1:, 1:] = inner
    nonminimal_packed = base64.b64encode(
        np.packbits(nonminimal_crop.reshape(-1), bitorder="little").tobytes()
    ).decode("ascii")
    malformed = (
        (replace(evidence, packed_mask_base64="%%%"), "base64"),
        (replace(evidence, packed_mask_base64=base64.b64encode(packed[:-1]).decode("ascii")),
         "length"),
        (replace(evidence, mask_sha256="0" * 64), "digest"),
        (replace(evidence, source_region_id="0" * 64), "identity"),
        (replace(evidence, axis=1), "axis"),
        (replace(evidence, bbox_xyxy=(0, 0, 8, 8), crop_shape=(7, 8)), "out of bounds"),
        (replace(evidence, pixel_count=evidence.pixel_count + 1), "pixel count"),
        (replace(evidence, packed_mask_base64=nonminimal_packed,
                 bbox_xyxy=(0, 0, 7, 7), crop_shape=(7, 7),
                 pixel_count=int(inner.sum())), "minimal nonzero bbox"),
        (replace(evidence, encoding="raw"), "encoding"),
        (replace(evidence, bit_order="big"), "encoding"),
    )
    for bad_evidence, error_text in malformed:
        with pytest.raises(ValueError, match=error_text):
            lattice_planes._decode_region_evidence(bad_evidence, reading.axis, reading.polygon)
        with pytest.raises(ValueError):
            replace(reading, region_evidence=bad_evidence)

def test_region_identity_survives_proposal_reordering_and_reading_graph_parity(monkeypatch):
    first, usable, calibration_result, polygon = _v1_region_control(sources=("first",))
    second_mask = np.asarray(first.mask).copy()
    second_mask[6:8, 6:8] = False
    second_mask[6:8, 5:6] = True
    second = PlaneProposal(
        axis=0, side=1, polygon=polygon, mask=second_mask,
        strength=0.7, sources=("second",))
    image, baseline = _controlled_v1_readings(
        monkeypatch, (first, second), usable, calibration_result, frame="same")
    _image2, reversed_readings = _controlled_v1_readings(
        monkeypatch, (second, first), usable, calibration_result, frame="same")
    assert len(baseline) == len(reversed_readings) == 2
    assert [item.region_evidence.source_region_id for item in baseline] == [
        item.region_evidence.source_region_id for item in reversed_readings]
    by_sources = {
        item.region_evidence.sources: lattice_planes._decode_region_evidence(
            item.region_evidence, item.axis, item.polygon)
        for item in baseline
    }
    np.testing.assert_array_equal(by_sources[("first",)], np.asarray(first.mask) & usable)
    np.testing.assert_array_equal(by_sources[("second",)], second_mask & usable)

    reading = baseline[0]
    legacy_equivalent = replace(
        reading, region_evidence=None,
        phase_evidence=replace(reading.phase_evidence, source_region_id=None),
    )
    assert reading.region_provenance_status == "available"
    assert legacy_equivalent.region_provenance_status == "unavailable"
    assert _signature_key(reading) == _signature_key(legacy_equivalent)
    from mc_photogrammetry.reconstruction.frozen_image_runner import _reading_record
    assert _reading_record(reading) == _reading_record(legacy_equivalent)

    enriched_signatures = []
    legacy_signatures = []
    for frame_id in ("a", "b"):
        enriched = replace(reading, frame=frame_id)
        legacy = replace(legacy_equivalent, frame=frame_id)
        enriched_signatures.append(lattice_planes.plane_signature(
            image, enriched, enriched.alternatives[0].blocks_per_period, usable=usable))
        legacy_signatures.append(lattice_planes.plane_signature(
            image, legacy, legacy.alternatives[0].blocks_per_period, usable=usable))
    for enriched, legacy in zip(enriched_signatures, legacy_signatures):
        np.testing.assert_array_equal(enriched.cells, legacy.cells)
        np.testing.assert_array_equal(enriched.interior, legacy.interior)
        np.testing.assert_array_equal(enriched.coverage, legacy.coverage)
        assert enriched.origin_uv == legacy.origin_uv
        assert legacy.region_footprint is None
    enriched_refusals, legacy_refusals = [], []
    enriched_readings = tuple(signature.reading for signature in enriched_signatures)
    legacy_readings = tuple(signature.reading for signature in legacy_signatures)
    enriched_matches = plane_graph.match_plane_graph(
        enriched_readings, enriched_signatures, refusals=enriched_refusals)
    legacy_matches = plane_graph.match_plane_graph(
        legacy_readings, legacy_signatures, refusals=legacy_refusals)
    assert enriched_matches == legacy_matches
    assert enriched_refusals == legacy_refusals


def test_reading_cap_keeps_selected_region_masks_associated_and_drops_excess(monkeypatch):
    base, usable, calibration_result, polygon = _v1_region_control()
    limit = lattice_planes.MAX_READINGS_PER_AXIS_SIDE
    proposals = []
    evidence_by_source = {}
    for index in range(limit + 2):
        mask = np.asarray(base.mask, dtype=bool).copy()
        mask[0, 2 + index] = True
        shifted_polygon = tuple((x, y + index * 0.01) for x, y in polygon)
        source = (f"cap-{index}",)
        proposal = PlaneProposal(
            axis=base.axis, side=base.side, polygon=shifted_polygon,
            mask=mask, strength=base.strength, sources=source)
        proposals.append(proposal)
        evidence_by_source[source] = lattice_planes._make_region_evidence(
            mask & usable, base.axis, shifted_polygon, source)

    image, readings = _controlled_v1_readings(
        monkeypatch, tuple(proposals), usable, calibration_result, frame="cap")
    expected = sorted(
        evidence_by_source.values(),
        key=lambda item: next(proposal.polygon for proposal in proposals
                              if proposal.sources == item.sources),
    )[:limit]
    selected_ids = [reading.region_evidence.source_region_id for reading in readings]
    assert selected_ids == [item.source_region_id for item in expected]
    dropped_ids = {item.source_region_id for item in evidence_by_source.values()} - set(selected_ids)
    assert len(readings) == limit and len(dropped_ids) == 2
    assert not dropped_ids.intersection(selected_ids)
    for reading in readings:
        expected_evidence = evidence_by_source[reading.region_evidence.sources]
        assert reading.region_evidence == expected_evidence
        np.testing.assert_array_equal(
            lattice_planes._decode_region_evidence(
                reading.region_evidence, reading.axis, reading.polygon),
            np.asarray(next(p for p in proposals
                            if p.sources == reading.region_evidence.sources).mask) & usable,
        )


def test_crowded_reading_cap_round_robins_eligible_axis_sides_with_exact_evidence(monkeypatch):
    base, usable, calibration_result, polygon = _v1_region_control()
    proposals = []
    source_to_mask = {}
    for axis in range(3):
        for side in (-1, 1):
            for rank in range(3):
                source = (f"axis-{axis}-side-{side}-rank-{rank}",)
                mask = np.asarray(base.mask, dtype=bool).copy()
                mask[7, 0] = side == -1
                mask[7, 1] = side == 1
                shifted = tuple((x, y + rank * .01) for x, y in polygon)
                proposals.append(PlaneProposal(axis=axis, side=side, polygon=shifted,
                                               mask=mask, strength=.8, sources=source))
                source_to_mask[source] = mask & usable
    def read(proposal_order):
        return _controlled_v1_readings(
            monkeypatch, proposal_order, usable, calibration_result, frame="crowded",
            window_side=lambda mask: -1.0 if mask[7, 0] else 1.0)[1]
    readings = read(tuple(proposals))
    reversed_readings = read(tuple(reversed(proposals)))
    assert lattice_planes.MAX_READINGS == 6
    assert lattice_planes.MAX_READINGS_PER_AXIS_SIDE == 2
    assert len(readings) == 6
    assert {(reading.axis, reading.side) for reading in readings} == {
        (axis, float(side)) for axis in range(3) for side in (-1, 1)}
    assert [(_signature_key(reading), reading.region_evidence.source_region_id)
            for reading in readings] == [
        (_signature_key(reading), reading.region_evidence.source_region_id)
        for reading in reversed_readings]
    assert all(reading.region_evidence.sources[0].endswith("rank-0") for reading in readings)
    for reading in readings:
        source = reading.region_evidence.sources
        assert reading.phase_evidence.source_region_id == reading.region_evidence.source_region_id
        np.testing.assert_array_equal(lattice_planes._decode_region_evidence(
            reading.region_evidence, reading.axis, reading.polygon), source_to_mask[source])

    only_axis_zero = read(tuple(proposal for proposal in proposals if proposal.axis == 0))
    assert len(only_axis_zero) == 4
    assert {reading.axis for reading in only_axis_zero} == {0}
    assert all(sum((reading.axis, reading.side) == (0, float(side))
                   for reading in only_axis_zero) == 2 for side in (-1, 1))


def test_axis_round_robin_does_not_promote_ineligible_reading(monkeypatch):
    base, usable, calibration_result, polygon = _v1_region_control()
    proposals = tuple(PlaneProposal(axis=axis, side=-1, polygon=polygon,
                                    mask=np.asarray(base.mask, dtype=bool).copy(),
                                    strength=.8, sources=(f"eligibility-axis-{axis}",))
                      for axis in (0, 1))
    values = iter((.01, .5))
    _image, selected = _controlled_v1_readings(
        monkeypatch, proposals, usable, calibration_result, frame="eligibility",
        structural_share=lambda *_: next(values), max_readings=1)
    assert len(selected) == 1 and selected[0].axis == 1
    assert selected[0].structural_share == .5
    assert lattice_planes.MIN_STRUCTURAL_SHARE == .15


def test_crowded_axis_selector_closes_scalar_observer_with_exact_dispositions(monkeypatch):
    base, usable, calibration_result, polygon = _v1_region_control()
    proposals = []
    for axis in range(3):
        for side in (-1, 1):
            for rank in range(3):
                mask = np.asarray(base.mask, dtype=bool).copy()
                mask[7, 0] = side == -1
                mask[7, 1] = side == 1
                proposals.append(PlaneProposal(
                    axis=axis, side=side,
                    polygon=tuple((x, y + rank * .01) for x, y in polygon),
                    mask=mask, strength=.8,
                    sources=(f"observer-axis-{axis}-side-{side}-rank-{rank}",)))
    observer = region_growth._ScalarAttritionObserver()
    with region_growth._scalar_attrition_context(observer):
        _image, crowded = _controlled_v1_readings(
            monkeypatch, tuple(proposals), usable, calibration_result,
            frame="crowded-observed", window_side=lambda mask: -1.0 if mask[7, 0] else 1.0)
        axis_zero_proposals = tuple(proposal for proposal in proposals if proposal.axis == 0)
        _image, axis_zero = _controlled_v1_readings(
            monkeypatch, axis_zero_proposals, usable, calibration_result,
            frame="axis0-observed", window_side=lambda mask: -1.0 if mask[7, 0] else 1.0)
    record = observer.record()
    reading_stage = record["stages"]["reading"]
    assert record["status"] == "complete", (record["omitted_accounting_events"],
        {key: {name: stage[name] for name in ("expected", "classified", "unclassified",
          "unexpected", "multiply_classified", "unexpected_events")}
         for key, stage in record["stages"].items()})
    assert record["omitted_accounting_events"] == 0
    assert reading_stage["expected"] == reading_stage["classified"] == len(proposals) + len(axis_zero_proposals)
    assert all(reading_stage[name] == 0 for name in (
        "unclassified", "unexpected", "multiply_classified", "unexpected_events", "omitted_examples"))
    examples = reading_stage["examples"]
    assert len(examples) == reading_stage["expected"]
    assert len({tuple(row["identity"]) for row in examples}) == len(examples)
    counts = {disposition: sum(row["disposition"] == disposition for row in examples)
              for disposition in ("accepted", "reading_cap", "per_axis_side_cap")}
    assert counts == {"accepted": 10, "reading_cap": 12, "per_axis_side_cap": 2}
    assert {(reading.axis, reading.side) for reading in crowded} == {
        (axis, float(side)) for axis in range(3) for side in (-1, 1)}
    assert {reading.axis for reading in axis_zero} == {0}
    source_by_identity = {
        (frame, ordinal): proposal.sources
        for frame, group in (("crowded-observed", proposals), ("axis0-observed", axis_zero_proposals))
        for ordinal, proposal in enumerate(group)
    }
    accepted_sources = {source_by_identity[tuple(row["identity"])] for row in examples
                        if row["disposition"] == "accepted"}
    assert accepted_sources == {reading.region_evidence.sources for reading in crowded + axis_zero}
    for readings in (crowded, axis_zero):
        for canonical_index, reading in enumerate(sorted(readings, key=lattice_planes.reading_order_key)):
            assert reading.phase_evidence.source_region_id == reading.region_evidence.source_region_id
            assert observer.bindings[("reading", id(reading))] == (reading.frame, canonical_index)
            proposal = next(proposal for proposal in proposals
                            if proposal.sources == reading.region_evidence.sources)
            np.testing.assert_array_equal(lattice_planes._decode_region_evidence(
                reading.region_evidence, reading.axis, reading.polygon), proposal.mask & usable)


def test_real_rectifier_footprint_preserves_hole_and_usable_exclusion():
    rgb, usable, calibration_result, proposals = _rgb_scene_and_proposals()
    source = max(proposals.proposals, key=lambda item: int(np.count_nonzero(item.mask)))
    readings = lattice_planes.frame_plane_readings_from_proposals(
        rgb, usable, calibration_result,
        PlaneProposalResult("ok", "real rectifier base", (source,)),
        frame="real-rectifier",
    )
    assert readings
    reading = readings[0]
    original_mask = lattice_planes._decode_region_evidence(
        reading.region_evidence, reading.axis, reading.polygon)
    b = reading.alternatives[0].blocks_per_period
    base_signature = lattice_planes.plane_signature(
        rgb, reading, b, usable=usable)

    # Select two well-supported, well-separated real rectifier samples and map
    # them back to source pixels. The first becomes a region hole; the second
    # becomes a usable-mask exclusion.
    footprint = base_signature.region_footprint
    cells = base_signature.cells.shape[0]
    samples = cells * footprint.subsample_factor
    distance_blocks = reading.alternatives[0].distance_blocks
    unit_extent = (cells / 2.0) / distance_blocks
    unit_centre = ((base_signature.origin_uv[0] + cells / 2.0) / distance_blocks,
                   (base_signature.origin_uv[1] + cells / 2.0) / distance_blocks)
    region_patch = lattice.rectify_plane(
        original_mask.astype(np.float32), reading.axis,
        np.asarray(reading.rectification.axis_matrix), reading.rectification.focal,
        extent=unit_extent, samples=samples, centre=unit_centre,
        principal_point=reading.rectification.principal_point, side=reading.side)
    normal, first, second = lattice.plane_basis(
        reading.axis, np.asarray(reading.rectification.axis_matrix))
    rows, columns = np.mgrid[0:samples, 0:samples]
    u = unit_centre[0] - unit_extent + columns * (2.0 * unit_extent) / (samples - 1)
    v = unit_centre[1] - unit_extent + rows * (2.0 * unit_extent) / (samples - 1)
    points = reading.side * normal + u[..., None] * first + v[..., None] * second
    source_x = (reading.rectification.principal_point[0]
                + points[..., 0] * reading.rectification.focal / points[..., 2])
    source_y = (reading.rectification.principal_point[1]
                - points[..., 1] * reading.rectification.focal / points[..., 2])
    support = cv2.distanceTransform(original_mask.astype(np.uint8), cv2.DIST_L2, 5)
    pixel_x = np.clip(np.rint(source_x).astype(int), 0, original_mask.shape[1] - 1)
    pixel_y = np.clip(np.rint(source_y).astype(int), 0, original_mask.shape[0] - 1)
    candidates = ((region_patch > 0.99) & (points[..., 2] > 1e-6)
                  & (source_x >= 8) & (source_x < original_mask.shape[1] - 8)
                  & (source_y >= 8) & (source_y < original_mask.shape[0] - 8)
                  & (support[pixel_y, pixel_x] >= 8))
    grid_points = np.argwhere(candidates)
    assert len(grid_points) > 2
    grid_centre = np.asarray(region_patch.shape, dtype=float) / 2.0
    first_index = grid_points[np.argmin(np.sum((grid_points - grid_centre) ** 2, axis=1))]
    first_pixel = (int(pixel_x[tuple(first_index)]), int(pixel_y[tuple(first_index)]))
    distance_from_first = ((pixel_x[tuple(grid_points.T)] - first_pixel[0]) ** 2
                           + (pixel_y[tuple(grid_points.T)] - first_pixel[1]) ** 2)
    second_index = grid_points[int(np.argmax(distance_from_first))]
    second_pixel = (int(pixel_x[tuple(second_index)]), int(pixel_y[tuple(second_index)]))
    assert np.linalg.norm(np.asarray(first_pixel) - np.asarray(second_pixel)) > 20

    holed_mask = original_mask.copy()
    hole_x, hole_y = first_pixel
    holed_mask[hole_y - 6:hole_y + 7, hole_x - 6:hole_x + 7] = False
    usable_with_exclusion = usable.copy()
    excluded_x, excluded_y = second_pixel
    usable_with_exclusion[excluded_y - 4:excluded_y + 5,
                          excluded_x - 4:excluded_x + 5] = False
    sources = reading.region_evidence.sources
    holed_evidence = lattice_planes._make_region_evidence(
        holed_mask & usable_with_exclusion, reading.axis, reading.polygon, sources)
    base_evidence = lattice_planes._make_region_evidence(
        original_mask & usable_with_exclusion, reading.axis, reading.polygon, sources)
    holed_reading = replace(
        reading, region_evidence=holed_evidence,
        phase_evidence=replace(reading.phase_evidence,
                               source_region_id=holed_evidence.source_region_id),
    )
    base_reading = replace(
        reading, region_evidence=base_evidence,
        phase_evidence=replace(reading.phase_evidence,
                               source_region_id=base_evidence.source_region_id),
    )
    signature = lattice_planes.plane_signature(
        rgb, holed_reading, b, usable=usable_with_exclusion)
    base_signature = lattice_planes.plane_signature(
        rgb, base_reading, b, usable=usable_with_exclusion)
    footprint = signature.region_footprint
    base_footprint = base_signature.region_footprint
    decoded = lattice_planes._decode_region_evidence(
        holed_reading.region_evidence, reading.axis, reading.polygon)
    np.testing.assert_array_equal(decoded, holed_mask & usable_with_exclusion)
    inside = np.frombuffer(footprint.inside_region_sample_counts, dtype=np.uint8)
    base_inside = np.frombuffer(
        base_footprint.inside_region_sample_counts, dtype=np.uint8)
    valid = np.frombuffer(footprint.valid_sample_counts, dtype=np.uint8)
    assert int(inside.sum()) < int(base_inside.sum())  # the image-space hole survives rectification
    assert int(inside.sum()) <= int(valid.sum())

    all_pixels = np.ones_like(usable_with_exclusion, dtype=bool)
    unexcluded_signature = lattice_planes.plane_signature(
        rgb, holed_reading, b, usable=all_pixels)
    unexcluded = unexcluded_signature.region_footprint
    assert footprint.inside_region_sample_counts == unexcluded.inside_region_sample_counts
    assert int(valid.sum()) < int(np.frombuffer(
        unexcluded.valid_sample_counts, dtype=np.uint8).sum())


def test_legacy_readings_report_region_provenance_unavailable():
    readings, _diagnostics = _readings(mixed_material)
    assert readings
    reading = readings[0]
    assert reading.region_evidence is None
    assert reading.region_provenance_status == "unavailable"
    image, usable, _camera = _scene(mixed_material)
    signature = lattice_planes.plane_signature(
        image, reading, reading.alternatives[0].blocks_per_period, usable=usable)
    assert signature.region_footprint is None


def test_region_provenance_module_has_no_dataset_evaluator_or_reference_imports():
    source = Path(lattice_planes.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.casefold() for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append((node.module or "").casefold())
    forbidden = ("dataset", "evaluation", "evaluator", "manifest", "reference", "ground_truth")
    assert not any(any(term in name for term in forbidden) for name in imported)


def test_frozen_runner_real_image_path_keeps_schema_v1_and_region_signatures(
    tmp_path, monkeypatch
):
    from mc_photogrammetry.reconstruction import frozen_image_runner

    grey, _usable, _camera = _scene(mixed_material)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    png = tmp_path / "rendered-room.png"
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    png.write_bytes(encoded.tobytes())

    region_readings = []
    region_signatures = []
    read_real = lattice_planes.frame_plane_readings_from_proposals
    signature_real = lattice_planes.plane_signature

    def record_readings(*args, **kwargs):
        readings = read_real(*args, **kwargs)
        region_readings.extend(readings)
        return readings

    def record_signatures(*args, **kwargs):
        signature = signature_real(*args, **kwargs)
        region_signatures.append(signature)
        return signature

    monkeypatch.setattr(lattice_planes, "frame_plane_readings_from_proposals", record_readings)
    monkeypatch.setattr(lattice_planes, "plane_signature", record_signatures)
    freeze = tmp_path / "real-path-freeze.json"
    result = frozen_image_runner.run_frozen_image_pipeline([png], freeze)
    loaded = frozen_image_runner.load_and_verify_frozen_image(freeze)
    frame = loaded["frames"][0]
    assert result["schema_version"] == loaded["schema_version"] == 1
    assert loaded["labels_present"] is False
    assert frame["proposal"]["proposals"] and frame["readings"]
    assert frame["signatures"] and frame["signature_abstentions"] == []
    assert region_readings and all(
        reading.region_provenance_status == "available" for reading in region_readings)
    assert region_signatures and all(
        signature.region_footprint is not None for signature in region_signatures)
    assert set(frame["readings"][0]) == frozen_image_runner._READING_KEYS
    assert set(frame["signatures"][0]) == frozen_image_runner._SIGNATURE_KEYS
    assert "region_evidence" not in frame["readings"][0]
    assert "phase_evidence" not in frame["readings"][0]
    assert "region_footprint" not in frame["signatures"][0]
    assert loaded["code_hashes"]["lattice_planes"] == frozen_image_runner._sha256(
        Path(lattice_planes.__file__).read_bytes())


def test_proposal_adapter_returns_existing_typed_abstentions_for_invalid_inputs():
    rgb, usable, result, proposals = _rgb_scene_and_proposals()

    def reasons_for(image, mask, calibration_result, proposal_result):
        diagnostics: dict = {}
        readings = lattice_planes.frame_plane_readings_from_proposals(
            image, mask, calibration_result, proposal_result,
            frame="invalid", diagnostics=diagnostics)
        assert readings == ()
        assert diagnostics["abstentions"]
        reasons = {item.reason for item in diagnostics["abstentions"]}
        assert reasons <= set(lattice_planes.READING_ABSTENTION_REASONS)
        return reasons

    assert reasons_for(rgb[..., 0], usable, result, proposals) == {"no_plane_region"}
    wrong_dimensions = CalibrationResult(
        focal_pixels=result.focal_pixels, yaw_degrees=result.yaw_degrees,
        pitch_degrees=result.pitch_degrees, width=WIDTH - 1, height=HEIGHT,
        inliers=result.inliers, segments=result.segments,
        median_residual=result.median_residual,
    )
    assert reasons_for(rgb, usable, wrong_dimensions, proposals) == {"no_plane_region"}
    malformed_numeric = CalibrationResult(
        focal_pixels="not-a-number", yaw_degrees=result.yaw_degrees,
        pitch_degrees=result.pitch_degrees, width=WIDTH, height=HEIGHT,
        inliers=result.inliers, segments=result.segments,
        median_residual=result.median_residual,
    )
    assert reasons_for(rgb, usable, malformed_numeric, proposals) == {"no_plane_region"}
    empty_result = PlaneProposalResult("abstain", "no image regions", ())
    assert reasons_for(rgb, usable, result, empty_result) == {"no_plane_region"}

    line_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    line_mask[HEIGHT // 2, 20:WIDTH - 20] = True
    bad_masks = PlaneProposalResult("ok", "invalid masks", (
        PlaneProposal(0, 1, ((0, 0), (10, 0), (10, 10)),
                      np.ones((10, 10), dtype=bool)),
        PlaneProposal(1, -1, ((0, 0), (10, 0), (10, 10)),
                      np.zeros((HEIGHT, WIDTH), dtype=bool)),
        PlaneProposal(2, 1, ((0, 0), (WIDTH - 1, 0), (WIDTH - 1, HEIGHT - 1)),
                      line_mask),
        PlaneProposal(0, -1, ((0, 0), (10, 0), (10, 10)),
                      [[True, False], [True]]),
    ))
    assert reasons_for(rgb, usable, result, bad_masks) == {"unmeasurable_plane"}

    source = proposals.proposals[0]
    malformed_sources = PlaneProposalResult("ok", "bad sources", (
        PlaneProposal(source.axis, source.side, source.polygon, source.mask,
                      strength=source.strength, sources=None),
        PlaneProposal(source.axis, source.side, source.polygon, source.mask,
                      strength=source.strength, sources=("c4", 7)),
    ))
    assert reasons_for(rgb, usable, result, malformed_sources) == {"unmeasurable_plane"}


def test_proposal_adapter_types_near_grazing_window_as_unmeasurable(monkeypatch):
    width = height = 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    usable = np.ones((height, width), dtype=bool)
    calibration_result = CalibrationResult(
        focal_pixels=100.0, yaw_degrees=0.0, pitch_degrees=0.0,
        width=width, height=height, inliers=12, segments=12,
        median_residual=0.0)
    segments = np.asarray([(0, index, 99, 99 - index) for index in range(12)], dtype=np.float32)
    monkeypatch.setattr(lattice_planes, "detect_segments", lambda _image: segments)
    monkeypatch.setattr(
        lattice_planes, "structural_weights", lambda _grey, found: np.ones(len(found), dtype=float))
    mask = np.zeros((height, width), dtype=bool)
    mask[20:80, 49:52] = True
    proposal = PlaneProposal(
        axis=0, side=1, polygon=((49, 20), (51, 20), (51, 79), (49, 79)),
        mask=mask, strength=1.0, sources=("grazing-control",))
    diagnostics: dict = {}
    readings = lattice_planes.frame_plane_readings_from_proposals(
        image, usable, calibration_result,
        PlaneProposalResult("ok", "grazing control", (proposal,)),
        frame="grazing", diagnostics=diagnostics)
    assert readings == ()
    assert diagnostics["abstentions"]
    assert diagnostics["abstentions"][0].reason == "unmeasurable_plane"
    assert "plane_window" in diagnostics["abstentions"][0].detail


def test_proposal_adapter_is_not_connected_to_default_cli_or_pipeline():
    source_root = Path(__file__).resolve().parents[1] / "src" / "mc_photogrammetry"
    protected = [source_root / "cli.py", source_root / "pipeline"]
    accepted_consumer = source_root / "pipeline" / "image_derived_lattice_measure.py"
    consumers = []
    for path in protected:
        files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for source in files:
            if "frame_plane_readings_from_proposals" in source.read_text(
                    encoding="utf-8"):
                consumers.append(source)
    assert consumers == [accepted_consumer]


def test_plane_signature_respects_the_declared_caps_and_is_deterministic():
    readings, _ = _readings(mixed_material)
    reading = readings[0]
    alternative = reading.alternatives[0]
    cached_image, cached_usable, _camera = _scene(mixed_material)
    first = lattice_planes.plane_signature(
        cached_image, reading, alternative.blocks_per_period, usable=cached_usable)
    second = lattice_planes.plane_signature(
        cached_image, reading, alternative.blocks_per_period, usable=cached_usable)
    assert np.array_equal(first.cells, second.cells)
    assert np.array_equal(first.interior, second.interior)
    assert np.array_equal(first.coverage, second.coverage)
    assert first.cells.shape == first.interior.shape == first.coverage.shape
    assert first.cells.shape[0] <= lattice_planes.MAX_SIGNATURE_CELLS
    assert first.cells.shape[1] <= lattice_planes.MAX_SIGNATURE_CELLS
    assert first.origin_uv == second.origin_uv
    assert first.cells.dtype == np.float32 and first.coverage.dtype == bool


def test_warp_signature_identity_homography_returns_the_same_cells():
    readings, _ = _readings(mixed_material)
    reading = readings[0]
    cached_image, cached_usable, _camera = _scene(mixed_material)
    signature = lattice_planes.plane_signature(
        cached_image, reading, reading.alternatives[0].blocks_per_period, usable=cached_usable)
    warped, coverage = lattice_planes.warp_signature(signature, np.eye(3), signature.cells.shape)
    assert np.array_equal(warped, signature.cells)
    assert np.array_equal(coverage, signature.coverage)


def test_structural_share_orders_a_uniform_wall_below_a_structured_one():
    """Directly over the measure: an aperiodic patch is not explained by a lattice."""
    uniform = np.zeros((220, 220), dtype=np.float32)
    uniform[::20, :] = 1.0
    uniform[:, ::20] = 1.0
    structured = uniform.copy()
    structured[57:73, 61:79] = 1.0  # an opening at no lattice phase
    uniform_share = lattice_planes._structural_share(uniform, 20.0)
    structured_share = lattice_planes._structural_share(structured, 20.0)
    assert uniform_share < lattice_planes.MIN_STRUCTURAL_SHARE
    assert structured_share > uniform_share


def test_proposal_control_reports_a_typed_abstention_or_a_reading():
    """Never weaken a threshold: a failed proposal must say so in typed form."""
    grey = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
    usable = ~create_gui_mask(HEIGHT, WIDTH)
    position, yaw, pitch = CAMERA
    camera = CameraModel(position=position, yaw_degrees=yaw, pitch_degrees=pitch, fov_degrees=70.0)
    diagnostics: dict = {}
    readings = lattice_planes.frame_plane_readings(
        grey, usable, calibration(camera), frame="blank", diagnostics=diagnostics)
    assert readings == ()
    assert diagnostics["abstentions"], "a failed proposal must record a typed reason"
    for abstention in diagnostics["abstentions"]:
        assert abstention.reason in lattice_planes.READING_ABSTENTION_REASONS
