"""DS-25 controls: graph-wide cycle tracks, boundary and synthetic degeneracy.

The DS-25 diagnostic builds multi-view keypoint tracks from the frozen directed
top-k candidate graphs and resolves competing repeated-texture tracks with a
preregistered label-blind score. These controls pin the properties that make the
result interpretable:

* a clean synthetic session is resolved into one track per world point, and each
  track closes within the frozen 1-degree epipolar residual;
* a session whose repeated-texture mass closes descriptor cycles records the
  frozen variant's honest limit: with one keypoint per view the per-pair closure
  residual is numerically zero, so the impostor mass is kept and this suite does
  not claim a geometric rejection that does not happen;
* frame-order permutation, frame relabelling and repeated invocation leave the
  track set and the derived pairwise match sets unchanged (points are identical);
* sessions with fewer than three views, with no descriptor-consistent cycles, or
  with pure-rotation geometry yield no usable tracks and a typed pose refusal;
* the blind phases (candidate graphs, tracks, pair match sets, DS-23 selection)
  touch no reference accessor: the reference entry points are poisoned and the
  real frozen inputs are still processed to frozen digests;
* the rerun from the frozen outputs reproduces the written `freeze.json`
  digests, so the "hash before scoring" order is enforced and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from mc_photogrammetry.pipeline import graph_tracks as gt  # noqa: E402

import diagnose_ds25_graph_tracks as ds25  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic session builders (public camera API only).
# --------------------------------------------------------------------------- #
def _bit_descriptors(indices: np.ndarray, bits: int = 6, scale: float = 40.0) -> np.ndarray:
    values = (
        (indices[:, None] >> np.arange(bits)[None, :]) & 1
    ).astype(np.float32)
    descriptors = np.zeros((len(indices), 128), dtype=np.float32)
    descriptors[:, :bits] = values * scale
    return descriptors


def _view(
    name: str,
    position,
    world: np.ndarray,
    indices: np.ndarray,
    descriptors: np.ndarray,
    *,
    resolution=(640, 360),
    fov=70.0,
    noise=0.2,
    rng=None,
):
    from mc_photogrammetry.dataset.plan import camera_from_target
    from mc_photogrammetry.pipeline.correspondence import FrameFeatures
    from mc_photogrammetry.reconstruction.camera import intrinsics_for, project_points

    width, height = resolution
    camera = camera_from_target(
        position, (0.0, 0.0, 0.0), fov_degrees=fov, resolution=(width, height)
    )
    projected = project_points(
        [tuple(value) for value in world], camera, width=width, height=height
    )
    pixels = np.asarray([[value[0], value[1]] for value in projected], dtype=np.float64)
    if noise and rng is not None:
        pixels = pixels + rng.normal(0.0, noise, size=pixels.shape)
    features = FrameFeatures(
        frame_id=name,
        points=pixels,
        descriptors=np.asarray(descriptors, dtype=np.float32),
        width=width,
        height=height,
        usable_pixels=width * height,
        sizes=np.full(len(pixels), 6.0),
        boundary_dropped=0,
    )
    intrinsics = intrinsics_for(camera, width, height)
    return gt.SessionView(
        name=name,
        features=features,
        camera=camera,
        intrinsics=(intrinsics.focal_length_pixels, intrinsics.principal_point_pixels),
    )


def _clean_session(names=("alpha", "bravo", "charlie", "delta"), count=24, seed=11):
    rng = np.random.default_rng(seed)
    world = rng.uniform(-1.5, 1.5, size=(count, 3))
    positions = [(4.0, 0.0, 0.0), (0.0, 0.0, 4.0), (-4.0, 0.0, 0.0), (0.0, 0.0, -4.0)]
    descriptors = _bit_descriptors(np.arange(count))
    views = [
        _view(name, positions[index], world, np.arange(count), descriptors, rng=rng)
        for index, name in enumerate(names)
    ]
    return views, world, descriptors


def _tracks_of(views):
    graphs = gt.build_directed_graphs(views)
    return gt.select_tracks(views, graphs)


# --------------------------------------------------------------------------- #
# 1. Clean synthetic session.
# --------------------------------------------------------------------------- #
def test_clean_session_resolves_one_track_per_world_point():
    views, world, _descriptors = _clean_session()
    result = _tracks_of(views)

    assert len(result.tracks) == len(world)
    assert all(track.cardinality == 4 for track in result.tracks)
    for track in result.tracks:
        # A true point carries the same keypoint index in every view.
        assert len(set(track.points)) == 1
    residuals = np.asarray(
        [track.epipolar_residual_degrees for track in result.tracks], dtype=np.float64
    )
    # The frozen 1-degree inlier law is the admission rule; a clean scene closes
    # well inside it and the worst track is reported honestly.
    assert float(np.median(residuals)) <= gt.DEFAULT_EPIPOLAR_DEGREES
    assert float(np.percentile(residuals, 90)) <= 0.5
    assert all(
        track.cycle_support_ratio >= 1.0 / 4.0 for track in result.tracks
    )
    # Derived pair sets are one-to-one and complete for a clean session.
    matches = gt.tracks_to_matches(result.tracks, "alpha", "bravo")
    assert matches.shape == (len(world), 4)
    assert len(np.unique(matches[:, 0])) == len(world)
    assert len(np.unique(matches[:, 2])) == len(world)


# --------------------------------------------------------------------------- #
# 2. Repeated-texture distractor mass.
# --------------------------------------------------------------------------- #
def test_vacuous_closure_does_not_separate_descriptor_repetition_mass():
    """Without session baselines the one-keypoint closure is vacuous.

    Descriptors carry *no* geometric information here: every impostor shares a
    true point's descriptor plus plain zero-mean noise, so the descriptor term
    cannot separate a true track from an impostor. The true points are a shallow
    orbit and each impostor triples points at different depths along one view ray.

    This control records an honest limitation of the frozen variant: a three-view
    track with one keypoint per view has, per view pair, four yaw-hypothesis rows
    (one per class). The pair's own bounded Gram/SVD null vector is close to
    orthogonal to all four rows, so the closure residual collapses to numerical
    zero and every descriptor-consistent cycle passes the 1-degree admission
    test. Geometry therefore drops nothing and the descriptor-consistent impostor
    mass is kept alongside the true points; this test must not claim the frozen
    resolver rejects it.
    """
    from mc_photogrammetry.dataset.plan import camera_from_target
    from mc_photogrammetry.pipeline.correspondence import FrameFeatures
    from mc_photogrammetry.reconstruction.camera import (
        camera_eye,
        intrinsics_for,
        project_points,
    )

    rng = np.random.default_rng(5)
    positions = [(4.0, 0.0, 0.0), (0.0, 0.0, 4.0), (-4.0, 0.0, 0.0)]
    cameras = [
        camera_from_target(position, (0.0, 0.0, 0.0), fov_degrees=70.0,
                           resolution=(640, 360))
        for position in positions
    ]

    def project(world_point, camera):
        point = project_points(
            [tuple(float(value) for value in world_point)],
            camera, width=640, height=360,
        )[0]
        return np.asarray([point[0], point[1]], dtype=np.float64)

    true_targets = rng.uniform(-1.0, 1.0, size=(12, 3))
    true_pixels = np.asarray(
        [[project(target, camera) for camera in cameras] for target in true_targets],
        dtype=np.float64,
    )
    base_descriptors = _bit_descriptors(np.arange(12))
    impostor_pixels = np.zeros((10, 3, 2), dtype=np.float64)
    impostor_descriptors = np.zeros((10, 128), dtype=np.float32)
    for row in range(10):
        base = rng.uniform(-0.8, 0.8, size=3)
        for index, camera in enumerate(cameras):
            eye = camera_eye(camera)
            direction = base - eye
            direction = direction / np.linalg.norm(direction)
            depth = float(np.linalg.norm(base - eye)) * (
                1.0 + 0.06 * (index + 1) * ((-1) ** row)
            )
            point = eye + direction * depth
            impostor_pixels[row, index] = project(point, camera)
        impostor_descriptors[row] = base_descriptors[row % 12] + rng.normal(
            0.0, 0.05, size=128
        ).astype(np.float32)

    points = [
        np.vstack([true_pixels[:, index, :], impostor_pixels[:, index, :]])
        for index in range(3)
    ]
    descriptors = np.vstack([base_descriptors, impostor_descriptors]).astype(np.float32)
    names = ("left", "mid", "right")
    views = []
    for index, name in enumerate(names):
        intrinsics = intrinsics_for(cameras[index], 640, 360)
        viewed = points[index] + rng.normal(0.0, 0.2, size=points[index].shape)
        views.append(
            gt.SessionView(
                name=name,
                features=FrameFeatures(
                    frame_id=name,
                    points=viewed,
                    descriptors=descriptors,
                    width=640,
                    height=360,
                    usable_pixels=640 * 360,
                    sizes=np.full(len(descriptors), 6.0),
                    boundary_dropped=0,
                ),
                camera=cameras[index],
                intrinsics=(
                    intrinsics.focal_length_pixels,
                    intrinsics.principal_point_pixels,
                ),
            )
        )
    result = _tracks_of(views)
    true_indices = set(range(12))
    assert result.tracks, "expected the descriptor-consistent cycles to close"
    kept = {int(track.points[0]) for track in result.tracks}
    # Complete recall: every true descriptor-consistent cycle survives.
    assert true_indices <= kept, f"missing true cycles: {sorted(true_indices - kept)}"
    # The frozen closure is vacuous, so the 10 repeated-texture impostors survive
    # too: the kept set is larger than the true point set. This is a recorded
    # negative of the frozen variant, not a geometric rejection.
    assert len(result.tracks) == 22
    assert len(kept - true_indices) == 10
    # The closure residual is numerically zero (the pair's own null vector is
    # orthogonal to its rows), so the frozen 1-degree admission test cannot
    # separate the impostor mass. A tiny non-zero epsilon is float noise.
    assert max(track.epipolar_residual_degrees for track in result.tracks) < 1e-9


# --------------------------------------------------------------------------- #
# 3. Order / label permutation invariance and determinism.
# --------------------------------------------------------------------------- #
def test_frame_order_permutation_leaves_tracks_and_pair_sets_unchanged():
    views, _world, _descriptors = _clean_session()
    base = _tracks_of(views)
    permuted = [views[2], views[0], views[3], views[1]]
    other = _tracks_of(permuted)
    key = lambda result: sorted(
        (tuple(track.views), tuple(track.points)) for track in result.tracks
    )
    assert key(base) == key(other)
    for left, right in (("alpha", "bravo"), ("charlie", "delta")):
        assert np.array_equal(
            gt.tracks_to_matches(base.tracks, left, right),
            gt.tracks_to_matches(other.tracks, left, right),
        )


def test_frame_relabelling_is_an_isomorphism_on_tracks():
    names = ("alpha", "bravo", "charlie", "delta")
    views, _world, _descriptors = _clean_session(names=names)
    rename = dict(zip(names, ("whiskey", "xray", "yankee", "zeta")))
    # Rebuild each relabelled view from the same geometry as its source view.
    relabelled = []
    for view in views:
        from mc_photogrammetry.pipeline.correspondence import FrameFeatures

        relabelled.append(
            gt.SessionView(
                name=rename[view.name],
                features=FrameFeatures(
                    frame_id=rename[view.name],
                    points=view.features.points,
                    descriptors=view.features.descriptors,
                    width=view.features.width,
                    height=view.features.height,
                    usable_pixels=view.features.usable_pixels,
                    sizes=view.features.sizes,
                    boundary_dropped=view.features.boundary_dropped,
                ),
                camera=view.camera,
                intrinsics=view.intrinsics,
            )
        )
    base = _tracks_of(views)
    other = _tracks_of(relabelled)
    # `other` already carries the new names, so map the BASE track names through
    # `rename` and compare against the already-relabelled `other` tracks. Mapping
    # `other` again would treat an already-new name as a base name.
    mapped = sorted(
        (tuple(rename[name] for name in track.views), tuple(track.points))
        for track in base.tracks
    )
    expected = sorted(
        (tuple(track.views), tuple(track.points)) for track in other.tracks
    )
    assert mapped == expected


def test_repeated_invocation_is_deterministic():
    views, _world, _descriptors = _clean_session(seed=3)
    first = _tracks_of(views)
    second = _tracks_of(views)
    assert gt.tracks_to_records(first.tracks) == gt.tracks_to_records(second.tracks)


# --------------------------------------------------------------------------- #
# 4. Degenerate refusal.
# --------------------------------------------------------------------------- #
def test_session_with_fewer_than_three_views_yields_no_tracks():
    views, _world, _descriptors = _clean_session(names=("alpha", "bravo"))
    result = _tracks_of(views)
    assert result.tracks == ()
    assert result.candidates == 0


def test_descriptor_inconsistent_views_yield_no_tracks():
    """Geometry agrees, descriptors do not: no ratio-qualified cycle can close."""
    views, _world, _descriptors = _clean_session(names=("alpha", "bravo", "charlie"))
    rng = np.random.default_rng(99)
    broken = []
    for view in views:
        features = view.features
        descriptors = rng.normal(0.0, 60.0, size=features.descriptors.shape).astype(
            np.float32
        )
        broken.append(
            gt.SessionView(
                name=view.name,
                features=type(features)(
                    frame_id=features.frame_id,
                    points=features.points,
                    descriptors=descriptors,
                    width=features.width,
                    height=features.height,
                    usable_pixels=features.usable_pixels,
                    sizes=features.sizes,
                    boundary_dropped=features.boundary_dropped,
                ),
                camera=view.camera,
                intrinsics=view.intrinsics,
            )
        )
    result = _tracks_of(broken)
    assert result.tracks == ()


def test_pose_selector_refuses_a_zero_parallax_track():
    """A degenerate (no-parallax) track must be refused, never solved."""
    import blind_pose_policy as blind_policy
    from mc_photogrammetry.pipeline.contracts import (
        CameraIntrinsics,
        CameraPose,
    )

    views, _world, _descriptors = _clean_session(names=("alpha", "bravo", "charlie"))
    left = views[0]
    right = views[1]
    intrinsics = CameraIntrinsics(
        image_size=(left.width, left.height),
        focal_length_pixels=left.intrinsics[0],
        principal_point_pixels=left.intrinsics[1],
    )
    pose = CameraPose(
        position=None,
        yaw_degrees=left.camera.yaw_degrees,
        pitch_degrees=left.camera.pitch_degrees,
        roll_degrees=left.camera.roll_degrees,
    )
    right_pose = CameraPose(
        position=None,
        yaw_degrees=right.camera.yaw_degrees,
        pitch_degrees=right.camera.pitch_degrees,
        roll_degrees=right.camera.roll_degrees,
    )
    # Every correspondence collapses onto one pixel in both views: zero parallax.
    matches = np.tile(
        np.asarray([100.0, 120.0, 100.0, 120.0], dtype=np.float64), (24, 1)
    )
    blind = blind_policy.to_blind_input(
        left_intrinsics=intrinsics,
        left_pose=pose,
        right_intrinsics=intrinsics,
        right_pose=right_pose,
        matches=matches,
        image_size=(left.width, left.height),
    )
    decision = blind_policy.select_blind_pose(blind)
    assert not decision.solved
    assert decision.reason in {
        "pure_rotation",
        "rank_deficient",
        "insufficient_inliers",
        "low_inlier_fraction",
        "no_yaw_class_passes_gates",
    }


# --------------------------------------------------------------------------- #
# 5. Real frozen inputs: boundary + determinism.
# --------------------------------------------------------------------------- #
REAL_BLIND = ROOT / "exports" / "diagnostics" / "ds16" / "match_candidates.json"
REAL_ARRAYS = ROOT / "exports" / "diagnostics" / "ds16" / "match_candidates.npz"


def _real_inputs():
    if not (REAL_BLIND.exists() and REAL_ARRAYS.exists()):
        pytest.skip("DS-16 frozen artifacts not present in this checkout")
    frozen = json.loads(REAL_BLIND.read_text(encoding="utf-8"))
    arrays = dict(np.load(REAL_ARRAYS))
    return frozen, arrays


def test_real_frozen_t6_tracks_are_deterministic_and_supported():
    frozen, arrays = _real_inputs()
    session = {"set": "T6"}
    views = ds25.session_views(session, frozen, arrays)
    assert len(views) == 4
    first = gt.select_tracks(views, gt.build_directed_graphs(views))
    second = gt.select_tracks(views, gt.build_directed_graphs(views))
    assert gt.tracks_to_records(first.tracks) == gt.tracks_to_records(second.tracks)
    assert first.tracks, "T6 should close at least one graph-wide track"
    for track in first.tracks:
        assert track.cardinality >= 3
        assert track.epipolar_residual_degrees <= gt.DEFAULT_EPIPOLAR_DEGREES
        assert 0.0 < track.cycle_support_ratio <= 1.0
        assert np.isfinite(track.descriptor_cost)


def test_blind_phases_never_touch_reference_accessors(monkeypatch, tmp_path):
    """Poison every reference entry point, then run the real blind phases."""
    def poison(*_args, **_kwargs):
        raise AssertionError("reference accessor reached during a blind phase")

    monkeypatch.setattr("diagnose_match_attribution.load_frozen", poison)
    for name in ("reference_truth_fractions", "run_attribution", "extract_candidates"):
        if hasattr(ds25.attribution, name):
            monkeypatch.setattr(ds25.attribution, name, poison)
    monkeypatch.setattr("diagnose_ds24_image_derived_pose.reference_models", poison)
    monkeypatch.setattr("diagnose_ds24_image_derived_pose.reference_classes", poison)
    monkeypatch.setattr("diagnose_ds24_image_derived_pose.reference_truth_fractions", poison)
    monkeypatch.setattr("diagnose_ds25_graph_tracks.phase_c_reference", poison)
    monkeypatch.setattr("diagnose_ds25_graph_tracks.reference_truth", poison)

    frozen, arrays = _real_inputs()
    # Both real sessions are processed through the real phase A + phase B code
    # paths; only pixels/features and the image-derived M3 estimates are read.
    out_dir = tmp_path / "ds25"
    out_dir.mkdir()
    log_lines: list[str] = []
    phase_a = ds25.write_phase_a(out_dir, frozen, arrays, log=log_lines.append)
    assert phase_a["sessions"], "the real frozen sessions must produce tracks"
    assert any(entry["tracks"] for entry in phase_a["sessions"])
    track_payload, track_arrays, pair_payload, pair_arrays = ds25.read_frozen_phase_a(
        out_dir, log_lines.append
    )
    ds25.write_freeze(out_dir, phase_a, log_lines.append)
    decisions, read_union = ds25.select_pairs(pair_payload, pair_arrays)
    assert len(decisions) == 9
    assert not [name for name in read_union
                if name not in ds25.blind_policy.GEOMETRIC_INPUT_WHITELIST]
    # The frozen tape is content-addressed: re-reading it must reproduce the
    # digests that were written before the selection ran.
    assert phase_a["tracks"]["arrays_sha256"] == track_payload["arrays"]["arrays_sha256"]
    assert (
        phase_a["pair_match_sets"]["arrays_sha256"]
        == pair_payload["arrays"]["arrays_sha256"]
    )
    freeze = json.loads((out_dir / "freeze.json").read_text(encoding="utf-8"))
    assert freeze["tracks"]["arrays_sha256"] == phase_a["tracks"]["arrays_sha256"]
    assert len(track_arrays) == len(phase_a["track_records"])


def test_real_frozen_track_arrays_match_written_artifact():
    """The frozen DS-25 tape, when present, must reproduce its own digests."""
    base = ROOT / "exports" / "diagnostics" / "ds25"
    tracks_json = base / "tracks.json"
    if not tracks_json.exists():
        pytest.skip("DS-25 artifacts not present in this checkout")
    payload = json.loads(tracks_json.read_text(encoding="utf-8"))
    arrays = dict(np.load(base / "tracks.npz"))
    actual = {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in sorted(arrays.items())
    }
    assert actual == payload["arrays"]["sha256_by_array"]
    freeze = json.loads((base / "freeze.json").read_text(encoding="utf-8"))
    assert freeze["tracks"]["arrays_sha256"] == payload["arrays"]["arrays_sha256"]
    evaluation_path = base / "evaluation.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        assert evaluation["tracks"]["arrays_sha256"] == payload["arrays"]["arrays_sha256"]
        assert (
            evaluation["boundary"]["freeze"]["tracks"]["arrays_sha256"]
            == payload["arrays"]["arrays_sha256"]
        )
