"""Focused tests for the bounded M7 normal-mode raw-image entry seam.

Every test drives the real accepted chain (session -> connections -> pose graph)
with only the declared injectable seams substituted: ``image_only.solve_image_cameras``
so no SIFT calibration runs, and the connection matcher so no SIFT matching runs.
The entry's own jobs - ordinary-image discovery/decoding, explicit bounded budgets,
schema-2 persistence through the existing store, and the typed
``geometry_unavailable`` terminal - are what these controls exercise.

The controls are deliberately hostile about scope: the entry must accept only
ordinary image paths, must never construct V1/V2/V3/M5 geometry or call the M6
orchestration facade, and must carry a refused graph or pose graph unchanged
rather than fabricating a result.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_connections as connections_mod
from mc_photogrammetry.pipeline import image_derived_entry as entry
from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_graph_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
    CameraSolution,
)
from mc_photogrammetry.project.store import load_project


_IMAGE_SIZE = (60, 40)
_FORBIDDEN_PIXEL_KEYS = {"image", "pixels", "array", "pixel_data", "image_data"}


# ---------------------------------------------------------------------------
# Fixtures: ordinary PNGs, a fake solver, and a fake matcher
# ---------------------------------------------------------------------------

def test_persist_session_forwards_capture_limit_without_changing_default(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mc_photogrammetry.pipeline import image_derived_entry as entry
    session = SimpleNamespace(session_sha256="a" * 64)
    record = {"session_sha256": session.session_sha256}
    monkeypatch.setattr(entry.session_mod, "image_derived_session_to_record", lambda value: record)
    calls = []
    def save(project, destination, **kwargs):
        calls.append(kwargs)
    monkeypatch.setattr(entry, "save_project", save)
    monkeypatch.setattr(entry, "load_project", lambda path: SimpleNamespace(image_derived_session=record))
    for cap in (None, 4096):
        entry._persist_session(destination=tmp_path / "session.mcscan", session=session,
                               source_paths=(), name="synthetic", project_max_bytes=cap)
    assert calls == [{}, {"max_bytes": 4096}]
    assert not list(tmp_path.iterdir())


def _write_png(path: Path, colour: tuple[int, int, int], *, size=_IMAGE_SIZE) -> Path:
    from PIL import Image

    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = colour
    image[: height // 2, : width // 2] = tuple(value // 3 for value in colour)
    Image.fromarray(image).save(path)
    return path


def _estimate() -> CameraEstimate:
    return CameraEstimate(
        intrinsics=CameraIntrinsics(_IMAGE_SIZE, (100.0, 100.0), (30.0, 20.0)),
        pose=CameraPose(
            position=None, yaw_degrees=14.0, pitch_degrees=17.0, roll_degrees=3.5),
        residual_pixels=0.4,
        confidence=0.0,
    )


def _patch_solver(monkeypatch) -> None:
    def fake_solve(frames):
        unresolved: list[str] = []
        for frame_id in frames:
            unresolved.append(f"{frame_id}: position (vanishing points fix no translation)")
            unresolved.append(f"{frame_id}: scale (needs the lattice period)")
            unresolved.append(f"{frame_id}: yaw quadrant (recovered modulo 90 degrees)")
        return CameraSolution(
            cameras={frame_id: _estimate() for frame_id in frames},
            unresolved=tuple(unresolved),
            scale_known=False,
        )

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)


def _solved_pair(left_id: str, right_id: str, *, support: int = 6) -> SimpleNamespace:
    """A unit-baseline, identity-rotation pair that composes without a cycle error."""
    return SimpleNamespace(
        left_id=left_id, right_id=right_id, status="solved", reason="", solved=True,
        pose=SimpleNamespace(
            baseline_direction=(1.0, 0.0, 0.0),
            relative_rotation=np.eye(3, dtype=np.float64),
            solved=True,
        ),
        right_yaw_class=0,
        matches=np.arange(4 * support, dtype=np.float64).reshape(support, 4),
        support=support, support_floor=8, raw_keypoints=(support, support),
        candidate_edges=support * 2, mutual_survivors=support,
        median_selected_residual_degrees=0.4, parallax_degrees=5.0,
        cheirality_fraction=1.0, singular_ratio=0.01, ambiguity_margin=3,
        hypotheses=(),
    )


def _patch_matcher(monkeypatch) -> None:
    def matcher(left_id, left_image, right_id, right_image, *, left_camera, right_camera):
        return _solved_pair(left_id, right_id)

    monkeypatch.setattr(entry, "match_and_solve_pair", matcher)


def _partial_memory_entry(monkeypatch, tmp_path, admitted=("image-0000", "image-0002")):
    """No source file or screenshot bytes: inject synthetic ingress arrays."""
    ids = ("image-0000", "image-0001", "image-0002")
    sources = tuple(SimpleNamespace(frame_id=name, source_path=str(tmp_path / f"{name}.png")) for name in ids)
    def discover(paths, *, recursive, ingress_sink):
        for index, source in enumerate(sources):
            ingress_sink(source.source_path, source, bytes([index]))
        return SimpleNamespace(image_frames=sources, video_sources=())
    monkeypatch.setattr(entry, "discover_media", discover)
    monkeypatch.setattr(entry, "load_image_bytes", lambda data, **kwargs: np.full((40, 60, 3), data[0], dtype=np.uint8))
    def solve(frames):
        assert tuple(frames) == ids
        notes = tuple(note for name in admitted for note in (
            f"{name}: position (vanishing points fix no translation)",
            f"{name}: scale (needs the lattice period)",
            f"{name}: yaw quadrant (recovered modulo 90 degrees)"))
        return CameraSolution(cameras={name: _estimate() for name in admitted}, unresolved=notes, scale_known=False)
    monkeypatch.setattr(image_only, "solve_image_cameras", solve)
    calls = []
    def matcher(left_id, left_image, right_id, right_image, **kwargs):
        calls.append((left_id, right_id))
        return _solved_pair(left_id, right_id)
    result = entry.build_image_derived_entry([tmp_path], project_path=tmp_path / "partial.mcscan", matcher=matcher)
    return result, calls


def test_partial_entry_keeps_full_sources_and_matches_only_admitted_original_ids(monkeypatch, tmp_path):
    result, calls = _partial_memory_entry(monkeypatch, tmp_path)
    assert calls == [("image-0000", "image-0002")]
    assert result.graph.frame_ids == result.pose_graph.frame_ids == ("image-0000", "image-0002")
    assert len(result.source_paths) == 3 and result.source_paths == tuple(frame.source_label for frame in result.session.input_frames)
    saved = load_project(result.project_path)
    assert saved.schema_version == 2 and saved.image_derived_session["version"] == 2
    assert saved.image_derived_session == result.session_record
    assert saved.source_paths == list(result.source_paths)
    with pytest.raises(ValueError, match="source paths"):
        dataclasses.replace(result, source_paths=(result.source_paths[0], result.source_paths[2]), entry_sha256=None)


def test_zero_camera_terminal_skips_connections_pose_and_geometry(monkeypatch, tmp_path):
    from mc_photogrammetry.pipeline import image_derived_production as production
    monkeypatch.setattr(connections_mod, "build_image_derived_connections", lambda *a, **k: pytest.fail("zero connections"))
    monkeypatch.setattr(pose_graph_mod, "compose_image_derived_pose_graph", lambda *a, **k: pytest.fail("zero pose"))
    result, calls = _partial_memory_entry(monkeypatch, tmp_path, admitted=())
    assert result.terminal == result.reason == "camera_unavailable"
    assert result.graph is None and result.pose_graph is None and calls == []
    assert len(result.session_record["frames"]) == 3
    monkeypatch.setattr(production.entry_mod, "build_image_derived_entry", lambda *a, **k: result)
    monkeypatch.setattr(production.lattice_mod, "measure_image_derived_lattice", lambda **k: pytest.fail("zero lattice"))
    outcome = production.derive_image_output_from_paths([tmp_path], project_path=tmp_path / "zero.mcscan")
    assert outcome.status == "geometry_unavailable" and outcome.reason == "camera_unavailable"
    assert outcome.lattice is outcome.geometry is outcome.output is None
    assert load_project(tmp_path / "zero.mcscan").volume.blocks == {}


def test_one_camera_local_review_has_no_pair_or_geometry_invocation(monkeypatch, tmp_path):
    from mc_photogrammetry.pipeline import image_derived_production as production
    result, calls = _partial_memory_entry(monkeypatch, tmp_path, admitted=("image-0002",))
    assert calls == [] and result.graph.pair_count == 0 and result.graph.evidence == ()
    assert tuple(result.session.cameras) == ("image-0002",)
    assert result.session.cameras["image-0002"].pose.position is None
    monkeypatch.setattr(production.entry_mod, "build_image_derived_entry", lambda *a, **k: result)
    monkeypatch.setattr(production.lattice_mod, "measure_image_derived_lattice", lambda **k: pytest.fail("single lattice"))
    outcome = production.derive_image_output_from_paths([tmp_path], project_path=tmp_path / "one.mcscan")
    assert outcome.status == "geometry_unavailable" and outcome.reason == "single_camera_local_review"
    assert outcome.lattice is outcome.geometry is outcome.output is None


def test_entry_replays_excluded_pixels_in_full_inventory(monkeypatch, tmp_path):
    result, _ = _partial_memory_entry(monkeypatch, tmp_path)
    excluded = result.session.input_frames[1]
    bad = np.full(excluded.image_shape, 99, dtype=np.uint8)
    object.__setattr__(excluded, "image", np.frombuffer(bad.tobytes(), dtype=np.uint8).reshape(bad.shape))
    with pytest.raises(ValueError, match="frame"):
        dataclasses.replace(result, entry_sha256=None)


@pytest.mark.parametrize("tamper", ["writable", "reopenable", "cross-pixels", "detached-identity"])
def test_partial_entry_replays_active_objects_against_full_inventory(monkeypatch, tmp_path, tamper):
    result, _ = _partial_memory_entry(monkeypatch, tmp_path)
    left, right = result.session.frames
    if tamper in ("writable", "reopenable"):
        pixels = np.array(left.image, copy=True)
        if tamper == "reopenable":
            pixels.setflags(write=False)
        hostile = _clone(left, image=pixels)
    elif tamper == "cross-pixels":
        assert left.image_sha256 != right.image_sha256
        hostile = _clone(left, image=right.image)
    else:
        hostile = _clone(left)
    session = _clone(result.session, frames=(hostile, right))
    assert session.input_frames is result.session.input_frames
    assert session.session_sha256 == result.session.session_sha256
    with pytest.raises(ValueError, match="session frame does not replay"):
        _construct(result, session=session)


def _two_images(tmp_path: Path) -> None:
    _write_png(tmp_path / "a.png", (10, 20, 30))
    _write_png(tmp_path / "b.png", (40, 50, 60))


def _contains_pixel_payload(value) -> bool:
    if isinstance(value, dict):
        return any(
            key in _FORBIDDEN_PIXEL_KEYS or _contains_pixel_payload(item)
            for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_pixel_payload(item) for item in value)
    return isinstance(value, np.ndarray)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_the_entry_builds_the_chain_persists_it_and_terminates_geometry_unavailable(
        monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    project = tmp_path / "scan.mcscan"

    result = entry.build_image_derived_entry([tmp_path], project_path=project)

    assert result.terminal == entry.GEOMETRY_UNAVAILABLE
    assert result.reason == ""
    assert result.graph.status == "ok"
    assert result.pose_graph.status == "ok"
    assert len(result.session.frames) == 2
    assert result.graph.session_sha256 == result.session.session_sha256
    assert result.pose_graph.session_sha256 == result.session.session_sha256
    assert result.pose_graph.graph_sha256 == result.graph.graph_sha256
    assert project.exists()


def test_the_persisted_project_is_schema_two_with_the_canonical_session_and_empty_volume(
        monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    project = tmp_path / "scan.mcscan"

    result = entry.build_image_derived_entry([tmp_path], project_path=project)
    loaded = load_project(project)

    assert loaded.schema_version == 2
    assert loaded.image_derived_session == result.session_record
    assert loaded.image_derived_session["session_sha256"] == result.session.session_sha256
    assert loaded.volume.to_dict()["blocks"] == []
    assert loaded.source_paths == list(result.source_paths)
    assert not _contains_pixel_payload(loaded.image_derived_session)


def test_directory_inputs_expand_to_resolved_ordinary_image_paths(monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    result = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "scan.mcscan")

    assert result.source_paths == (
        str((tmp_path / "a.png").resolve()),
        str((tmp_path / "b.png").resolve()),
    )
    assert all(Path(path).suffix == ".png" for path in result.source_paths)


def test_explicit_duplicate_bytes_are_deduplicated_by_accepted_discovery(
        monkeypatch, tmp_path):
    _write_png(tmp_path / "a.png", (10, 20, 30))
    _write_png(tmp_path / "copy.png", (10, 20, 30))
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    result = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "scan.mcscan")

    assert len(result.session.frames) == 1
    assert result.source_paths == (str((tmp_path / "a.png").resolve()),)


def test_recursive_walk_finds_nested_images(monkeypatch, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_png(nested / "deep.png", (70, 80, 90))
    _write_png(tmp_path / "top.png", (1, 2, 3))
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    flat = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "flat.mcscan")
    deep = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "deep.mcscan", recursive=True)

    assert len(flat.session.frames) == 1
    assert len(deep.session.frames) == 2


def test_the_entry_digest_is_deterministic_and_replays(monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    project = tmp_path / "scan.mcscan"

    first = entry.build_image_derived_entry([tmp_path], project_path=project)
    second = entry.build_image_derived_entry([tmp_path], project_path=project)

    assert first.entry_sha256 == second.entry_sha256
    assert first.entry_sha256 == entry._digest(entry._entry_payload(first))
    assert first.to_record()["entry_sha256"] == first.entry_sha256
    json.dumps(first.to_record(), allow_nan=False)


# ---------------------------------------------------------------------------
# Input discipline
# ---------------------------------------------------------------------------

def test_video_sources_are_refused(monkeypatch, tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"not really a video")
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    with pytest.raises(entry.ImageDerivedEntryError) as raised:
        entry.build_image_derived_entry(
            [tmp_path], project_path=tmp_path / "scan.mcscan")

    assert raised.value.reason == entry.ENTRY_VIDEO_UNSUPPORTED
    assert not (tmp_path / "scan.mcscan").exists()


def test_an_empty_discovery_is_refused(monkeypatch, tmp_path):
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    with pytest.raises(entry.ImageDerivedEntryError) as raised:
        entry.build_image_derived_entry(
            [tmp_path], project_path=tmp_path / "scan.mcscan")

    assert raised.value.reason == entry.ENTRY_NO_IMAGES


@pytest.mark.parametrize("bad", [None, "images", b"images", 42, object()])
def test_non_path_input_collections_are_refused(monkeypatch, tmp_path, bad):
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    with pytest.raises(entry.ImageDerivedEntryError) as raised:
        entry.build_image_derived_entry(
            bad, project_path=tmp_path / "scan.mcscan")

    assert raised.value.reason == entry.ENTRY_INPUT_INVALID


def test_non_path_members_are_refused(monkeypatch, tmp_path):
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    with pytest.raises(entry.ImageDerivedEntryError) as raised:
        entry.build_image_derived_entry(
            [tmp_path, 7], project_path=tmp_path / "scan.mcscan")

    assert raised.value.reason == entry.ENTRY_INPUT_INVALID


def test_a_missing_input_path_is_refused(monkeypatch, tmp_path):
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    with pytest.raises(entry.ImageDerivedEntryError) as raised:
        entry.build_image_derived_entry(
            [tmp_path / "missing"], project_path=tmp_path / "scan.mcscan")

    assert raised.value.reason == entry.ENTRY_INPUT_INVALID


def test_the_declared_frame_cap_is_enforced(monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    with pytest.raises(entry.ImageDerivedEntryError) as raised:
        entry.build_image_derived_entry(
            [tmp_path], project_path=tmp_path / "scan.mcscan", max_frames=1)

    assert raised.value.reason == entry.ENTRY_FRAME_LIMIT


def test_the_normal_entry_accepts_nine_images(monkeypatch, tmp_path):
    for index in range(9):
        _write_png(tmp_path / f"image-{index:02}.png",
                   (10 + index, 30 + index, 60 + index))
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    result = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "nine-images.mcscan", max_frames=9)

    assert len(result.session.frames) == 9
    assert len(result.graph.frame_ids) == 9
    assert result.terminal == entry.GEOMETRY_UNAVAILABLE


def test_production_frame_budgets_cover_the_public_32_image_ceiling():
    from mc_photogrammetry.pipeline import image_derived_output_policy
    from mc_photogrammetry.pipeline import image_derived_phase_anchor_policy
    from mc_photogrammetry.pipeline import image_derived_relative_translation_policy

    cap = entry.MAX_FRAMES_CAP
    assert cap == 32
    assert image_derived_output_policy.IMAGE_DERIVED_OUTPUT_POLICY.face_budget.max_frames >= cap
    assert image_derived_phase_anchor_policy.IMAGE_DERIVED_PHASE_ANCHOR_POLICY.budget.max_frames >= cap
    assert image_derived_relative_translation_policy.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.budget.max_frames >= cap


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_frames": 0},
        {"max_frames": entry.MAX_FRAMES_CAP + 1},
        {"max_frames": 2.5},
        {"max_pairs": -1},
        {"max_pairs": entry.MAX_PAIRS_CAP + 1},
        {"max_total_support": entry.MAX_TOTAL_SUPPORT_CAP + 1},
        {"max_edges": entry.MAX_EDGES_CAP + 1},
        {"max_cycle_checks": entry.MAX_CYCLE_CHECKS_CAP + 1},
        {"rotation_tolerance_degrees": 181.0},
        {"direction_tolerance": 1.5},
        {"direction_tolerance": float("nan")},
    ],
)
def test_budget_caps_are_bounded(monkeypatch, tmp_path, kwargs):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    with pytest.raises(entry.ImageDerivedEntryError) as raised:
        entry.build_image_derived_entry(
            [tmp_path], project_path=tmp_path / "scan.mcscan", **kwargs)

    assert raised.value.reason == entry.ENTRY_BUDGET_INVALID


def test_the_public_entry_point_takes_only_paths_budgets_and_the_matcher_seam():
    signature = inspect.signature(entry.build_image_derived_entry)

    assert list(signature.parameters) == [
        "image_inputs", "project_path", "matcher", "recursive", "max_frames",
        "max_pairs", "max_total_support", "max_edges", "max_cycle_checks",
        "rotation_tolerance_degrees", "direction_tolerance", "capture_ingress", "project_max_bytes", "retain_project_lease",
    ]
    # The single-read ingress pin is an evaluation-only, default-off observer: it
    # cannot change the default result and it accepts no manual metadata.
    assert signature.parameters["capture_ingress"].default is False
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values())
    for forbidden in (
        "manifest", "camera_manifest", "cameras", "scene", "bounds", "coordinates",
        "directions", "f3", "scale", "translation", "horizon", "annotations",
    ):
        assert forbidden not in signature.parameters


# ---------------------------------------------------------------------------
# Typed refusals and the geometry boundary
# ---------------------------------------------------------------------------

def test_a_session_refusal_is_typed_and_writes_no_project(monkeypatch, tmp_path):
    _two_images(tmp_path)

    def refusing_solver(frames):
        raise image_only.ImageOnlyCameraError("too little lattice structure")

    monkeypatch.setattr(image_only, "solve_image_cameras", refusing_solver)
    _patch_matcher(monkeypatch)
    project = tmp_path / "scan.mcscan"

    with pytest.raises(entry.ImageDerivedEntryError) as raised:
        entry.build_image_derived_entry([tmp_path], project_path=project)

    assert raised.value.reason == entry.ENTRY_SESSION_FAILED
    assert not project.exists()


def test_a_refused_connection_graph_still_terminates_and_persists_the_session(
        monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    project = tmp_path / "scan.mcscan"

    result = entry.build_image_derived_entry(
        [tmp_path], project_path=project, max_total_support=0)

    assert result.terminal == entry.GEOMETRY_UNAVAILABLE
    assert result.reason == connections_mod.CONNECTIONS_BUDGET_EXHAUSTED
    assert result.graph.status == "refused"
    assert result.pose_graph.status == "refused"
    assert project.exists()
    assert load_project(project).image_derived_session["session_sha256"] \
        == result.session.session_sha256


def test_the_terminal_carries_no_geometry_and_no_metric_scale(monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)

    result = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "scan.mcscan")

    for geometry_attribute in (
        "volume", "placement", "universe", "supports", "face_observations",
        "proposals", "promotion", "delivery", "export_bounds",
    ):
        assert not hasattr(result, geometry_attribute)
    assert result.terminal == entry.GEOMETRY_UNAVAILABLE
    assert result.session.scale_known is False
    assert result.graph.scale_known is False
    assert result.pose_graph.metric_scale_known is False
    record = result.to_record()
    assert record["scale_known"] is False
    assert "universe" not in record
    assert "volume" not in record


def test_the_module_never_imports_geometry_or_orchestration_seams():
    source = Path(entry.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)

    forbidden_modules = {
        "mc_photogrammetry.pipeline.image_derived_orchestration",
        "mc_photogrammetry.reconstruction.plane_region_support",
        "mc_photogrammetry.reconstruction.plane_placement",
        "mc_photogrammetry.reconstruction.plane_volume",
        "mc_photogrammetry.reconstruction.plane_face_observation",
        "mc_photogrammetry.reconstruction.appearance_state_proposals",
        "mc_photogrammetry.reconstruction.state_promotion",
        "mc_photogrammetry.reconstruction.plane_graph",
    }
    forbidden_names = {
        "PlacementUniverse", "PlaneVolumeResult", "PlaneFaceObservationResult",
        "FaceStateProposalResult", "PromotionResult", "compose_image_derived_pipeline",
        "resolve_face_placement", "build_plane_volume", "propose_face_states",
    }
    assert not (imported & forbidden_modules)
    assert not (referenced & forbidden_names)


# ---------------------------------------------------------------------------
# Deep replay, exact bindings, and the full-record entry digest
# ---------------------------------------------------------------------------

def _valid_result(monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    return entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "scan.mcscan")


def _refused_result(monkeypatch, tmp_path):
    _two_images(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    return entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "scan.mcscan", max_total_support=0)


def _clone(record, **changes):
    """Copy a frozen record without re-running its validating constructor."""
    clone = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(clone, field.name, getattr(record, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


def _reseal_graph(graph, **changes):
    """Reseal a cloned graph's enclosing digest around a nested tamper."""
    clone = _clone(graph, **changes)
    return _clone(clone, graph_sha256=connections_mod._digest(
        connections_mod._graph_payload(clone)))


def _reseal_pose_graph(pose_graph, **changes):
    """Reseal a cloned pose graph's enclosing digest around a nested tamper."""
    clone = _clone(pose_graph, **changes)
    return _clone(clone, pose_graph_sha256=pose_graph_mod._digest(
        pose_graph_mod._pose_graph_payload(clone)))


def _reseal_session(session, **changes):
    """Reseal a cloned session's digest around a constructor-bypassed frame state."""
    clone = _clone(session, **changes)
    return _clone(clone, session_sha256=session_mod._digest(
        session_mod._session_payload(clone)))


def _entry_fields(result):
    return {field.name: getattr(result, field.name)
            for field in dataclasses.fields(result)
            if field.name != "entry_sha256"}


def _construct(result, **changes):
    return entry.ImageDerivedEntryResult(**{**_entry_fields(result), **changes})


def test_a_genuine_entry_replays_every_carried_record_and_rebuilds(
        monkeypatch, tmp_path):
    """Valid replay: every carried record passes its own canonical validator."""
    result = _valid_result(monkeypatch, tmp_path)

    entry._replay_session(result.session)
    entry._replay_connection_graph(result.graph, result.session)
    entry._replay_pose_graph(result.pose_graph, result.graph, result.session)

    rebuilt = entry.ImageDerivedEntryResult(**_entry_fields(result))
    assert rebuilt.entry_sha256 == result.entry_sha256
    assert rebuilt.to_record() == result.to_record()


def test_the_entry_digest_binds_the_full_canonical_records(monkeypatch, tmp_path):
    """``entry_sha256`` hashes the carried records, not a scalar summary."""
    result = _valid_result(monkeypatch, tmp_path)

    payload = entry._entry_payload(result)
    assert payload["session_record"] == result.session_record
    assert payload["graph_record"] == result.graph.to_record()
    assert payload["pose_graph_record"] == result.pose_graph.to_record()

    scalar_only = {key: value for key, value in payload.items()
                   if key not in {"session_record", "graph_record", "pose_graph_record"}}
    assert entry._digest(scalar_only) != result.entry_sha256

    tampered = dict(payload)
    tampered["graph_record"] = dict(tampered["graph_record"], detail="forged")
    assert entry._digest(tampered) != result.entry_sha256


def test_a_session_with_a_forged_digest_is_refused(monkeypatch, tmp_path):
    result = _valid_result(monkeypatch, tmp_path)
    forged = _clone(result.session, session_sha256="f" * 64)

    with pytest.raises(ValueError, match="session does not replay"):
        _construct(result, session=forged)


def test_a_valid_produced_session_replays_every_retained_frame(monkeypatch, tmp_path):
    """Valid replay: a produced session's frames are immutable bytes-backed uint8."""
    result = _valid_result(monkeypatch, tmp_path)

    for frame in result.session.frames:
        assert frame.image is not None
        assert frame.image.dtype == np.uint8
        assert frame.image.flags.writeable is False
        base = frame.image
        while isinstance(base, np.ndarray):
            assert base.flags.writeable is False
            base = base.base
        assert isinstance(base, bytes)

    entry._replay_session(result.session)
    rebuilt = entry.ImageDerivedEntryResult(**_entry_fields(result))
    assert rebuilt.entry_sha256 == result.entry_sha256
    assert rebuilt.to_record() == result.to_record()


def test_a_session_with_writable_frame_pixels_is_refused(monkeypatch, tmp_path):
    """Sol counterexample: writable pixels with identical bytes and digest."""
    result = _valid_result(monkeypatch, tmp_path)
    original = result.session.frames[0]
    writable = np.array(original.image, dtype=np.uint8, copy=True)
    assert writable.flags.writeable is True
    assert writable.shape == original.image.shape
    assert session_mod.image_sha256(writable) == original.image_sha256

    hostile_frame = _clone(original, image=writable)
    hostile_session = _clone(
        result.session, frames=(hostile_frame,) + result.session.frames[1:])
    assert hostile_session.session_sha256 == result.session.session_sha256

    with pytest.raises(ValueError, match="session frame does not replay"):
        _construct(result, session=hostile_session)


def test_a_session_with_reopenable_numpy_owned_frame_pixels_is_refused(
        monkeypatch, tmp_path):
    """Read-only is not enough: numpy-owned memory can be reopened writable."""
    result = _valid_result(monkeypatch, tmp_path)
    original = result.session.frames[0]
    reopenable = np.array(original.image, dtype=np.uint8, copy=True)
    reopenable.setflags(write=False)
    assert reopenable.flags.writeable is False
    assert not isinstance(reopenable.base, bytes)
    assert session_mod.image_sha256(reopenable) == original.image_sha256

    hostile_frame = _clone(original, image=reopenable)
    hostile_session = _clone(
        result.session, frames=(hostile_frame,) + result.session.frames[1:])

    with pytest.raises(ValueError, match="session frame does not replay"):
        _construct(result, session=hostile_session)


def test_a_constructor_bypassed_non_uint8_frame_is_refused(monkeypatch, tmp_path):
    """A resealed record digest cannot hide a uint16 retained frame."""
    result = _valid_result(monkeypatch, tmp_path)
    original = result.session.frames[0]
    non_uint8 = original.image.astype(np.uint16)
    assert non_uint8.dtype != np.uint8

    hostile_frame = _clone(
        original, image=non_uint8,
        image_sha256=session_mod.image_sha256(non_uint8))
    hostile_session = _reseal_session(
        result.session, frames=(hostile_frame,) + result.session.frames[1:])
    assert hostile_session.session_sha256 == session_mod._digest(
        session_mod._session_payload(hostile_session))

    with pytest.raises(ValueError, match="session frame does not replay"):
        _construct(result, session=hostile_session)


def test_a_constructor_bypassed_frame_shape_is_refused(monkeypatch, tmp_path):
    """A resealed record digest cannot hide a declared shape that lies about pixels."""
    result = _valid_result(monkeypatch, tmp_path)
    original = result.session.frames[0]
    height, width, channels = original.image.shape
    wrong_shape = (width, height, channels)
    assert wrong_shape != original.image.shape

    hostile_frame = _clone(original, image_shape=wrong_shape)
    hostile_session = _reseal_session(
        result.session, frames=(hostile_frame,) + result.session.frames[1:])
    assert hostile_session.session_sha256 == session_mod._digest(
        session_mod._session_payload(hostile_session))

    with pytest.raises(ValueError, match="session frame does not replay"):
        _construct(result, session=hostile_session)


def test_a_pixel_free_deserialized_session_is_refused(monkeypatch, tmp_path):
    """The record round-trip is pixel-free; an entry session must retain pixels."""
    result = _valid_result(monkeypatch, tmp_path)
    deserialized = session_mod.image_derived_session_from_record(result.session_record)
    assert deserialized.session_sha256 == result.session.session_sha256
    assert all(frame.image is None for frame in deserialized.frames)

    with pytest.raises(ValueError, match="has no retained pixels"):
        _construct(result, session=deserialized)


def test_frames_cannot_trade_pixels_across_identities(monkeypatch, tmp_path):
    """A frame's retained pixels must be the ones its declared identity digests."""
    result = _valid_result(monkeypatch, tmp_path)
    left, right = result.session.frames
    swapped = (
        _clone(left, image=right.image),
        _clone(right, image=left.image),
    )
    hostile_session = _clone(result.session, frames=swapped)

    with pytest.raises(ValueError, match="session frame does not replay"):
        _construct(result, session=hostile_session)


def test_a_graph_with_a_stale_digest_is_refused(monkeypatch, tmp_path):
    """Sol counterexample: a graph whose content changed without a reseal."""
    result = _valid_result(monkeypatch, tmp_path)
    stale = _clone(result.graph, detail="stale graph detail")
    assert stale.graph_sha256 == result.graph.graph_sha256

    with pytest.raises(ValueError, match="graph digest mismatch"):
        _construct(result, graph=stale)


def test_a_resealed_graph_with_a_constructor_bypassed_record_is_refused(
        monkeypatch, tmp_path):
    """A self-consistent outer digest cannot hide a nested evidence violation."""
    result = _valid_result(monkeypatch, tmp_path)
    solved = next(record for record in result.graph.evidence if record.status == "solved")
    hostile_record = _clone(solved, baseline_direction=(3.0, 0.0, 0.0))
    evidence = tuple(hostile_record if record is solved else record
                     for record in result.graph.evidence)
    hostile = _reseal_graph(result.graph, evidence=evidence)
    assert hostile.graph_sha256 == connections_mod._digest(
        connections_mod._graph_payload(hostile))

    with pytest.raises(ValueError, match="graph evidence does not replay"):
        _construct(result, graph=hostile)


def test_a_resealed_graph_with_a_tampered_nested_budget_is_refused(
        monkeypatch, tmp_path):
    """The outer graph digest does not carry the nested ``budget_sha256``."""
    result = _valid_result(monkeypatch, tmp_path)
    hostile_budget = _clone(result.graph.budgets,
                            max_pairs=result.graph.budgets.max_pairs + 1)
    hostile = _reseal_graph(result.graph, budgets=hostile_budget)
    assert hostile.graph_sha256 == connections_mod._digest(
        connections_mod._graph_payload(hostile))

    with pytest.raises(ValueError, match="connection budget does not replay"):
        _construct(result, graph=hostile)


def test_connection_budgets_must_be_exactly_the_graph_budgets(monkeypatch, tmp_path):
    """Sol counterexample: alternate budgets swapped in for the carried graph."""
    result = _valid_result(monkeypatch, tmp_path)
    alternate = connections_mod.ConnectionBudgets(max_pairs=0, max_total_support=0)
    assert alternate != result.graph.budgets

    with pytest.raises(ValueError, match="connection budgets are not the graph budgets"):
        _construct(result, connection_budgets=alternate)


def test_pose_budgets_must_be_exactly_the_pose_graph_budgets(monkeypatch, tmp_path):
    """Sol counterexample: alternate budgets swapped in for the carried pose graph."""
    result = _valid_result(monkeypatch, tmp_path)
    alternate = pose_graph_mod.PoseGraphBudgets(
        max_edges=0, max_cycle_checks=0,
        rotation_tolerance_degrees=result.pose_budgets.rotation_tolerance_degrees,
        direction_tolerance=result.pose_budgets.direction_tolerance)
    assert alternate != result.pose_graph.budgets

    with pytest.raises(ValueError, match="pose budgets are not the pose-graph budgets"):
        _construct(result, pose_budgets=alternate)


def test_a_pose_graph_with_a_stale_digest_is_refused(monkeypatch, tmp_path):
    result = _valid_result(monkeypatch, tmp_path)
    stale = _clone(result.pose_graph, detail="stale pose detail")
    assert stale.pose_graph_sha256 == result.pose_graph.pose_graph_sha256

    with pytest.raises(ValueError, match="pose graph digest mismatch"):
        _construct(result, pose_graph=stale)


def test_a_resealed_pose_graph_with_a_constructor_bypassed_pose_is_refused(
        monkeypatch, tmp_path):
    """A self-consistent pose-graph digest cannot hide a malformed nested pose."""
    result = _valid_result(monkeypatch, tmp_path)
    hostile_pose = _clone(
        result.pose_graph.poses[1],
        rotation=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    poses = result.pose_graph.poses[:1] + (hostile_pose,) + result.pose_graph.poses[2:]
    hostile = _reseal_pose_graph(result.pose_graph, poses=poses)
    assert hostile.pose_graph_sha256 == pose_graph_mod._digest(
        pose_graph_mod._pose_graph_payload(hostile))

    with pytest.raises(ValueError, match="pose-graph poses do not replay"):
        _construct(result, pose_graph=hostile)


def test_invented_source_paths_are_refused(monkeypatch, tmp_path):
    """Sol counterexample: source paths that are not the session frame labels."""
    result = _valid_result(monkeypatch, tmp_path)
    invented = ("C:/invented/not-a-frame.png",)
    assert invented != result.source_paths

    with pytest.raises(ValueError, match="source paths are not the session frame labels"):
        _construct(result, source_paths=invented)


def test_reordered_source_paths_are_refused(monkeypatch, tmp_path):
    """Canonical ordering: the session frame order is the only accepted order."""
    result = _valid_result(monkeypatch, tmp_path)
    assert len(result.source_paths) == 2
    reordered = tuple(reversed(result.source_paths))
    assert reordered != result.source_paths

    with pytest.raises(ValueError, match="source paths are not the session frame labels"):
        _construct(result, source_paths=reordered)


def test_a_refused_entry_with_a_mismatched_reason_is_refused(monkeypatch, tmp_path):
    """Sol counterexample: a typed but wrong refusal token."""
    result = _refused_result(monkeypatch, tmp_path)
    assert result.graph.status == "refused"
    alternate = pose_graph_mod.POSE_GRAPH_SOURCE_REFUSED
    assert alternate != result.reason

    with pytest.raises(ValueError, match="reason/detail is not the derived terminal state"):
        _construct(result, reason=alternate)


def test_a_refused_entry_with_a_stale_detail_is_refused(monkeypatch, tmp_path):
    """Sol counterexample: the right reason token with an invented detail."""
    result = _refused_result(monkeypatch, tmp_path)
    assert result.detail

    with pytest.raises(ValueError, match="reason/detail is not the derived terminal state"):
        _construct(result, detail="forged refusal detail")


def test_a_composed_entry_cannot_invent_a_refusal_reason(monkeypatch, tmp_path):
    result = _valid_result(monkeypatch, tmp_path)
    assert result.reason == ""

    with pytest.raises(ValueError, match="reason/detail is not the derived terminal state"):
        _construct(result, reason=connections_mod.CONNECTIONS_BUDGET_EXHAUSTED)


def test_a_swapped_pose_graph_session_binding_is_refused(monkeypatch, tmp_path):
    """The pose graph must bind the carried session, not merely its own graph."""
    result = _valid_result(monkeypatch, tmp_path)
    hostile = _reseal_pose_graph(result.pose_graph, session_sha256="a" * 64)
    assert hostile.pose_graph_sha256 == pose_graph_mod._digest(
        pose_graph_mod._pose_graph_payload(hostile))

    with pytest.raises(ValueError, match="does not bind the session"):
        _construct(result, pose_graph=hostile)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def _write_two_pngs(tmp_path: Path) -> None:
    _write_png(tmp_path / "a.png", (11, 21, 31))
    _write_png(tmp_path / "b.png", (41, 51, 61))


def test_the_cli_exposes_only_normal_mode_image_inputs():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args(["image-derived-session", "images"])

    assert not hasattr(args, "mode")
    assert not hasattr(args, "cameras")
    assert not hasattr(args, "scene")
    assert not hasattr(args, "horizon")
    assert not hasattr(args, "camera_manifest")


def test_the_cli_defaults_match_the_declared_module_controls():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args(["image-derived-session", "images"])

    assert args.max_frames == entry.DEFAULT_MAX_FRAMES
    assert args.max_pairs == entry.DEFAULT_MAX_PAIRS
    assert args.max_total_support == entry.DEFAULT_MAX_TOTAL_SUPPORT
    assert args.max_edges == entry.DEFAULT_MAX_EDGES
    assert args.max_cycle_checks == entry.DEFAULT_MAX_CYCLE_CHECKS
    assert args.rotation_tolerance_degrees == entry.DEFAULT_ROTATION_TOLERANCE_DEGREES
    assert args.direction_tolerance == entry.DEFAULT_DIRECTION_TOLERANCE
    assert args.recursive is False


def test_the_cli_refuses_developer_style_manual_metadata():
    from mc_photogrammetry.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["image-derived-session", "images", "--mode", "developer"])


def test_the_cli_builds_a_session_project_and_prints_the_typed_terminal(
        monkeypatch, tmp_path, capsys):
    from mc_photogrammetry import cli

    _write_two_pngs(tmp_path)
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    project = tmp_path / "out.mcscan"
    monkeypatch.setattr(
        "sys.argv",
        ["mc-photogrammetry", "image-derived-session", str(tmp_path),
         "--project", str(project)],
    )

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert entry.GEOMETRY_UNAVAILABLE in output
    assert "Geometry: unavailable" in output
    assert project.exists()
    loaded = load_project(project)
    assert loaded.schema_version == 2
    assert loaded.image_derived_session is not None
    assert loaded.volume.to_dict()["blocks"] == []


def test_the_cli_reports_a_typed_refusal_without_a_traceback(monkeypatch, tmp_path):
    from mc_photogrammetry import cli

    (tmp_path / "notes.txt").write_text("no image here", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["mc-photogrammetry", "image-derived-session", str(tmp_path),
         "--project", str(tmp_path / "out.mcscan")],
    )

    with pytest.raises(SystemExit, match=entry.ENTRY_NO_IMAGES):
        cli.main()


@pytest.mark.skipif(__import__("os").name != "nt", reason="native Windows lease required")
def test_persist_session_hands_off_same_live_writer_lease_without_path_reload(monkeypatch, tmp_path):
    from mc_photogrammetry.project import store
    session = SimpleNamespace(session_sha256="a" * 64)
    record = {"session_sha256": session.session_sha256}
    monkeypatch.setattr(entry.session_mod, "image_derived_session_to_record", lambda _: record)
    real_save = store.save_project
    captured = []
    def save(project, destination, **kwargs):
        # Schema-1 fixture uses the real capped writer; no session/solver inference.
        from mc_photogrammetry.domain.models import ScanProject
        lease = real_save(ScanProject(name="lease"), destination, **kwargs)
        captured.append(lease)
        return lease
    def load(path, *, snapshot_bytes):
        assert snapshot_bytes == captured[0].read_bytes()
        return SimpleNamespace(image_derived_session=record)
    monkeypatch.setattr(entry, "save_project", save)
    monkeypatch.setattr(entry, "load_project", load)
    lease = entry._persist_session(destination=tmp_path / "scan.mcscan", session=session,
        source_paths=(), name="lease", project_max_bytes=4096, retain_project_lease=True)
    try:
        assert lease is captured[0] and lease.active
        assert lease._held._identity() == lease.identity
        lease.validate_live(tmp_path / "scan.mcscan")
    finally:
        lease.close()


@pytest.mark.skipif(__import__("os").name != "nt", reason="native Windows lease required")
def test_persist_session_replay_failure_discards_only_its_writer_handle(monkeypatch, tmp_path):
    from mc_photogrammetry.project import store
    from mc_photogrammetry.domain.models import ScanProject
    session = SimpleNamespace(session_sha256="a" * 64)
    monkeypatch.setattr(entry.session_mod, "image_derived_session_to_record", lambda _: {})
    captured = []
    def save(project, destination, **kwargs):
        lease = store.save_project(ScanProject(name="lease"), destination, **kwargs)
        captured.append(lease)
        return lease
    monkeypatch.setattr(entry, "save_project", save)
    monkeypatch.setattr(entry, "load_project", lambda *args, **kwargs: SimpleNamespace(image_derived_session={}))
    with pytest.raises(entry.ImageDerivedEntryError) as error:
        entry._persist_session(destination=tmp_path / "scan.mcscan", session=session,
            source_paths=(), name="lease", project_max_bytes=4096, retain_project_lease=True)
    assert error.value.reason == entry.ENTRY_PERSIST_FAILED
    assert not captured[0].active and not (tmp_path / "scan.mcscan").exists()


@pytest.mark.skipif(__import__("os").name != "nt", reason="native Windows lease required")
def test_synthetic_entry_result_keeps_live_birth_lease_outside_canonical_payload(monkeypatch, tmp_path):
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    _two_images(tmp_path)
    result = entry.build_image_derived_entry([tmp_path], project_path=tmp_path / "scan.mcscan",
        project_max_bytes=4 * 1024 * 1024, retain_project_lease=True)
    lease = result.project_lease
    try:
        assert lease.active and lease._held._identity() == lease.identity
        project = load_project(result.project_path, snapshot_bytes=lease.read_bytes())
        assert project.image_derived_session["session_sha256"] == result.session.session_sha256
        payload = entry._entry_payload(result)
        assert "project_lease" not in payload
        without_lease = dataclasses.replace(result, project_lease=None)
        assert entry._entry_payload(without_lease) == payload
        assert without_lease.entry_sha256 == result.entry_sha256
    finally:
        lease.close()
