"""Focused controls for the DS-M9-C image-derived lattice measurement seam.

Every fixture drives the real accepted image chain: a ray-cast synthetic room is
rendered to real ``uint8`` pixels, an accepted :class:`ImageDerivedSession` is
built over them with the *true* camera for that render injected through the
session's own declared solver seam (so no SIFT calibration search runs, but the
camera the seam must bind really is the camera the pixels describe), and the
accepted connection graph plus pose graph are composed from that session with a
real one-frame graph.

The controls are deliberately hostile about measurement and scope: the seam must
measure its own readings and split sources from the pixels, retain them through
DS-M9-B, never fabricate V1/V2/V3 geometry, preserve frame immutability, and
return a typed abstention rather than a partial bridge when a prerequisite is
absent.
"""
from __future__ import annotations

import sys
import json
from dataclasses import FrozenInstanceError, fields as dc_fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_lattice_planes import (  # noqa: E402
    CAMERA,
    ROOM,
    calibration as scene_calibration,
    mixed_material,
    render_room,
)

from mc_photogrammetry.pipeline import candidate_support_provenance as csp
from mc_photogrammetry.pipeline import image_derived_connections as connections_mod
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as m9c
from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_graph_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.pipeline import frame_ingress as ingress_mod
from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.pipeline.contracts import FrameTransform
from mc_photogrammetry.reconstruction import calibration as calibration_mod
from mc_photogrammetry.reconstruction import lattice_planes as lp
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction.camera import CameraModel

WIDTH, HEIGHT = 360, 240
_SCENE: dict[str, np.ndarray] = {}


def test_lattice_replay_binds_excluded_pixels_but_active_ids_remain_original(monkeypatch):
    from test_image_derived_session import _partial_session
    session, _ = _partial_session(monkeypatch)
    assert m9c._replay_session(session) is session
    assert tuple(frame.frame_id for frame in session.frames) == ("image-0000", "image-0002")
    excluded = session.input_frames[1]
    object.__setattr__(excluded, "image", np.frombuffer(np.ones(excluded.image_shape, dtype=np.uint8).tobytes(), dtype=np.uint8).reshape(excluded.image_shape))
    with pytest.raises(m9c._InputRefusal):
        m9c._replay_session(session)


@pytest.mark.parametrize("tamper", ["writable", "reopenable", "changed-pixels", "detached-identity"])
def test_partial_lattice_replay_validates_active_objects_against_full_inventory(monkeypatch, tamper):
    from test_image_derived_session import _partial_session
    session, _ = _partial_session(monkeypatch)
    left, right = session.frames
    if tamper in ("writable", "reopenable"):
        pixels = np.array(left.image, copy=True)
        if tamper == "reopenable":
            pixels.setflags(write=False)
        hostile = _forge(left, image=pixels)
    elif tamper == "changed-pixels":
        altered = np.array(left.image, copy=True)
        altered.flat[0] ^= 1
        pixels = np.frombuffer(altered.tobytes(), dtype=np.uint8).reshape(altered.shape)
        hostile = _forge(left, image=pixels)
    else:
        hostile = _forge(left)
    forged = _forge(session, frames=(hostile, right))
    assert forged.input_frames is session.input_frames
    assert forged.session_sha256 == session.session_sha256
    with pytest.raises(m9c._InputRefusal) as refusal:
        m9c._replay_session(forged)
    assert refusal.value.reason == "provenance_mismatch"


def _scene_rgb() -> np.ndarray:
    """One cached render; real pixels with a real recoverable Manhattan camera."""
    if "rgb" not in _SCENE:
        position, yaw, pitch = CAMERA
        camera = CameraModel(position=position, yaw_degrees=yaw, pitch_degrees=pitch,
                             fov_degrees=70.0)
        grey = render_room(camera, WIDTH, HEIGHT, ROOM, materials=mixed_material)
        _SCENE["rgb"] = np.repeat(grey[..., None], 3, axis=2)
    return _SCENE["rgb"]


def _true_calibration() -> calibration_mod.CalibrationResult:
    position, yaw, pitch = CAMERA
    camera = CameraModel(position=position, yaw_degrees=yaw, pitch_degrees=pitch,
                         fov_degrees=70.0)
    return scene_calibration(camera, width=WIDTH, height=HEIGHT)


def _build_session(monkeypatch, *, image=None, frame_ids=("f0",), ingress=False):
    """Build an accepted session whose injected camera really matches the pixels."""
    pixels = _scene_rgb() if image is None else image
    solution = calibration_mod.to_camera_solution(
        {frame_id: _true_calibration() for frame_id in frame_ids})

    def fake_solve(frames):
        return solution

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    pins = ({frame_id: _ingress_pin(frame_id, pixels) for frame_id in frame_ids}
            if ingress else None)
    return session_mod.build_image_derived_session(
        {frame_id: pixels for frame_id in frame_ids},
        source_paths={frame_id: f"{frame_id}.png" for frame_id in frame_ids},
        ingress=pins)


def _ingress_pin(frame_id, pixels, *, label=None,
                 encoded=b"synthetic-encoded-frame") -> object:
    """One real ingress pin bound to ``pixels`` through the accepted builder."""
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    return ingress_mod.build_frame_ingress_provenance(
        frame_id=frame_id, source_label=label or f"{frame_id}.png",
        encoded_bytes=encoded, decoded_pixels=pixels,
        transform=FrameTransform(source_size=(width, height)),
        decoded_pixel_sha256=session_mod.image_sha256(pixels))


def _reading_signature(reading) -> tuple:
    """The measured content of one reading, independent of its canonical index."""
    return (reading.frame, int(reading.axis), float(reading.period_pixels),
            float(reading.structural_share), reading.phase_evidence,
            int(reading.samples), float(reading.extent))


def _pose_graph(session):
    budgets = connections_mod.make_connection_budgets(
        max_pairs=max(1, len(session.frames)), max_total_support=64)
    graph = connections_mod.build_image_derived_connections(session, budgets=budgets)
    pose_budgets = pose_graph_mod.make_pose_graph_budgets(
        max_edges=4, max_cycle_checks=100, rotation_tolerance_degrees=2.0,
        direction_tolerance=0.05)
    return pose_graph_mod.compose_image_derived_pose_graph(graph, budgets=pose_budgets)


def _solved_pair(left_id, right_id, *, support=6):
    """A unit-baseline, identity-rotation pair that composes without a cycle error."""
    return SimpleNamespace(
        left_id=left_id, right_id=right_id, status="solved", reason="", solved=True,
        pose=SimpleNamespace(
            baseline_direction=(1.0, 0.0, 0.0),
            relative_rotation=np.eye(3, dtype=np.float64), solved=True),
        right_yaw_class=0,
        matches=np.arange(4 * support, dtype=np.float64).reshape(support, 4),
        support=support, support_floor=8, raw_keypoints=(support, support),
        candidate_edges=support * 2, mutual_survivors=support,
        median_selected_residual_degrees=0.4, parallax_degrees=5.0,
        cheirality_fraction=1.0, singular_ratio=0.01, ambiguity_margin=3,
        hypotheses=())


def _build_two_frame_session(monkeypatch):
    """Two translated renders that share one orientation, each with its true camera."""
    cameras = ((3.6, 0.7, 3.6), (5.0, 0.7, 5.0))
    images, calibrations = {}, {}
    for index, position in enumerate(cameras):
        frame_id = f"f{index}"
        camera = CameraModel(position=position, yaw_degrees=10.0,
                             pitch_degrees=-25.0, fov_degrees=70.0)
        grey = render_room(camera, WIDTH, HEIGHT, ROOM, materials=mixed_material)
        images[frame_id] = np.repeat(grey[..., None], 3, axis=2)
        calibrations[frame_id] = scene_calibration(camera, width=WIDTH, height=HEIGHT)
    solution = calibration_mod.to_camera_solution(calibrations)

    def fake_solve(frames):
        return solution

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    return session_mod.build_image_derived_session(images)


def _pose_graph_for_two(session):
    budgets = connections_mod.make_connection_budgets(
        max_pairs=2, max_total_support=64)

    def matcher(left_id, _left_image, right_id, _right_image, **_kwargs):
        return _solved_pair(left_id, right_id)

    graph = connections_mod.build_image_derived_connections(
        session, budgets=budgets, matcher=matcher)
    assert graph.status == "ok", graph.reason
    pose_budgets = pose_graph_mod.make_pose_graph_budgets(
        max_edges=4, max_cycle_checks=100, rotation_tolerance_degrees=2.0,
        direction_tolerance=0.05)
    return pose_graph_mod.compose_image_derived_pose_graph(graph, budgets=pose_budgets)


def _forge(record, **changes):
    """Bypass ``__post_init__`` to model a tampered on-the-wire record."""
    forged = object.__new__(type(record))
    for field in dc_fields(record):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(record, field.name)))
    return forged


def _result(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    result = m9c.measure_image_derived_lattice(session=session, pose_graph=pose_graph)
    assert isinstance(result, m9c.ImageDerivedLatticeResult), result
    return session, pose_graph, result


def _producer1_record(result):
    historical = _forge(result, producer="m9c-image-derived-lattice-measure-1")
    return _forge(historical, result_sha256=placement._sha256(m9c._result_payload(historical)))


def _historical_bytes(result):
    return json.dumps(result.to_record(), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@pytest.mark.parametrize("historical_producer", [1, 2])
def test_measure_producer3_replays_and_preserves_historical_bytes(monkeypatch, historical_producer):
    _session, _pose, result = _result(monkeypatch)
    assert result.status == "assembled" and result.producer == "m9c-image-derived-lattice-measure-3"
    assert result.schema == m9c.IMAGE_DERIVED_LATTICE_MEASURE_SCHEMA
    assert replace(result) == result
    historical = _producer1_record(result)
    historical = _forge(historical, producer=f"m9c-image-derived-lattice-measure-{historical_producer}")
    historical = _forge(historical, result_sha256=placement._sha256(m9c._result_payload(historical)))
    before = _historical_bytes(historical)
    assert historical.result_sha256 != result.result_sha256
    assert historical.result_sha256 == placement._sha256(m9c._result_payload(historical))
    with pytest.raises(ValueError, match="unsupported image-derived lattice producer"):
        replace(historical)
    assert _historical_bytes(historical) == before


def test_cap_recovery_reaches_real_reader_without_manufacturing_the_absent_axis(monkeypatch):
    import test_region_growth as room
    from mc_photogrammetry.reconstruction import region_growth as rg
    image, usable, segments, _ = room._synthetic_room()
    monkeypatch.setattr(rg, "detect_segments", lambda _: segments)
    calibration = room._calibration()
    full = rg.propose_image_planes_v2(image, usable, calibration)
    capped = rg.propose_image_planes_v2(image, usable, calibration, max_candidates=1)
    diagnostics = {}
    full_readings = lp.frame_plane_readings_from_proposals(
        image, usable, calibration, full, frame="room", diagnostics=diagnostics)
    capped_readings = lp.frame_plane_readings_from_proposals(
        image, usable, calibration, capped, frame="room")
    assert diagnostics["abstentions"] == []
    assert {r.axis for r in full_readings} == {0, 2}
    assert [r.axis for r in capped_readings] == [2]
    recovered = capped_readings[0]
    existing = next(r for r in full_readings if r.axis == 2)
    assert recovered == existing
    assert recovered.period_pixels == pytest.approx(47.954643947924275)
    assert recovered.region_evidence is not None and recovered.phase_evidence is not None
    assert recovered.region_evidence.axis == 2
    assert recovered.region_evidence.mask_sha256 == existing.region_evidence.mask_sha256
    assert recovered.region_evidence.source_region_id == existing.region_evidence.source_region_id
    assert 1 not in {r.axis for r in full_readings + capped_readings}
    decoded = lp._decode_region_evidence(recovered.region_evidence, recovered.axis, recovered.polygon)
    np.testing.assert_array_equal(decoded, capped.proposals[0].mask & usable)
    with pytest.raises(ValueError):
        lp._decode_region_evidence(replace(recovered.region_evidence, mask_sha256="0" * 64),
                                  recovered.axis, recovered.polygon)
    with pytest.raises(ValueError):
        lp._decode_region_evidence(replace(recovered.region_evidence, axis=0),
                                  recovered.axis, recovered.polygon)


def test_pixel_proposals_reach_three_axis_signature_and_matcher_accounting(monkeypatch):
    import test_region_growth as room
    from mc_photogrammetry.reconstruction import region_growth as rg, plane_graph
    image, _usable, segments, _ = room._synthetic_room()
    monkeypatch.setattr(rg, "detect_segments", lambda _: segments)
    monkeypatch.setattr(sys.modules[__name__], "_true_calibration", room._calibration)
    session = _build_session(monkeypatch, image=image, frame_ids=("f0", "f1"))
    pose = _pose_graph_for_two(session)
    observer = rg._ScalarAttritionObserver()
    with rg._scalar_attrition_context(observer):
        lattice = m9c.measure_image_derived_lattice(session=session, pose_graph=pose)
    assert lattice.status == "assembled", lattice.reason
    assert lattice.producer == "m9c-image-derived-lattice-measure-3"
    assert len(lattice.readings) == 6
    assert {(r.frame, r.axis) for r in lattice.readings} == {
        (frame, axis) for frame in ("f0", "f1") for axis in range(3)}
    assert len(lattice.bridge.readings) == len(lattice.bridge.split_sources_for_assembly()) == 6
    record = observer.record()
    assert record["status"] == "complete"
    assert record["stages"]["reading"]["classified"] == len(lattice.readings)
    signatures = signatures_mod.derive_image_derived_signatures(session=session, lattice_result=lattice)
    assert signatures.status == "assembled", signatures.reason
    assert len(signatures.signatures) == sum(len(r.alternatives) for r in lattice.readings) == 10
    refused = []
    matches = plane_graph.match_plane_graph(lattice.readings,
        tuple(e.signature for e in signatures.signatures), refusals=refused)
    assert len(matches) == 13
    canonical = placement._readings_index(lattice.readings)
    by_endpoint = {(frame, index): r for (frame, index, _axis), r in canonical.items()}
    # Q=0 active factors retain every measured endpoint and all three axes.
    active = [m for m in matches if m.quadrant_of_right == 0]
    assert len(active) == 4
    endpoints = {(m.left, m.left_plane) for m in active} | {(m.right, m.right_plane) for m in active}
    assert endpoints == set(by_endpoint)
    assert {by_endpoint[(m.left, m.left_plane)].axis for m in active} == {0, 1, 2}
    assert all(r.region_evidence is not None and r.phase_evidence is not None
               for r in by_endpoint.values())
    assert all(e.signature.reading == lattice.readings[e.canonical_index]
               and e.frame_id == lattice.readings[e.canonical_index].frame
               and e.signature.region_footprint.source_region_id ==
                   lattice.readings[e.canonical_index].region_evidence.source_region_id
               for e in signatures.signatures)
    # This measures support, not a numerical geometry acceptance or a C5 fix.


# ---------------------------------------------------------------------------
# The sealed happy path
# ---------------------------------------------------------------------------

def test_the_seam_measures_readings_and_seals_one_provenance_bridge(monkeypatch):
    session, pose_graph, result = _result(monkeypatch)
    assert result.terminal == m9c.GEOMETRY_UNAVAILABLE
    assert result.status == "assembled" and result.reason == ""
    assert result.session_sha256 == session.session_sha256
    assert result.pose_graph_sha256 == pose_graph.pose_graph_sha256
    assert result.graph_sha256 == pose_graph.graph_sha256
    assert result.frame_ids == tuple(frame.frame_id for frame in session.frames)
    assert len(result.readings) >= 1
    for reading in result.readings:
        assert isinstance(reading, lp.PlaneReading)
        assert reading.region_evidence is not None
        assert reading.phase_evidence is not None
    assert result.bridge is not None
    assert len(result.bridge.readings) == len(result.readings)
    assert all(first is second for first, second in
               zip(result.readings, result.bridge.readings))
    assert len(result.bridge.split_sources_for_assembly()) == len(result.readings)


def test_multiple_frames_measure_and_bind_their_own_readings(monkeypatch):
    session = _build_two_frame_session(monkeypatch)
    pose_graph = _pose_graph_for_two(session)
    result = m9c.measure_image_derived_lattice(session=session, pose_graph=pose_graph)
    assert isinstance(result, m9c.ImageDerivedLatticeResult), result
    assert result.status == "assembled"
    assert result.frame_ids == ("f0", "f1")
    assert len(result.camera_calibrations) == 2
    assert {reading.frame for reading in result.readings} == {"f0", "f1"}
    assert len(result.bridge.readings) == len(result.readings)
    # Every sealed split source addresses a reading in the sealed universe.
    universe = {id(reading) for reading in result.readings}
    assert all(id(reading) in universe for reading in result.bridge.readings)


def test_the_reconstructed_calibrations_are_measured_from_the_pixels(monkeypatch):
    _session, _pose, result = _result(monkeypatch)
    assert len(result.camera_calibrations) == len(result.frame_ids)
    for calibration in result.camera_calibrations:
        # The injected camera is the render's true camera, so the accepted residual
        # rule really does find its structural inliers in the rendered pixels.
        assert calibration.inliers >= 12
        assert calibration.segments >= calibration.inliers
        assert calibration.focal_pixels > 0.0
    assert result.pose_graph_sha256 != ""
    assert result.graph_sha256 != ""


def test_the_result_digest_replays_and_record_round_trips(monkeypatch):
    _session, _pose, result = _result(monkeypatch)
    record = result.to_record()
    assert record["result_sha256"] == result.result_sha256
    assert record["status"] == "assembled"
    assert record["reading_universe_sha256"] == result.bridge.reading_universe_sha256
    assert record["bridge_sha256"] == result.bridge.bridge_sha256
    assert m9c._result_payload(result)["session_sha256"] == result.session_sha256


def test_the_terminal_token_is_the_ds_m7_geometry_unavailable_boundary():
    assert m9c.GEOMETRY_UNAVAILABLE == entry_mod.GEOMETRY_UNAVAILABLE
    assert m9c.GEOMETRY_UNAVAILABLE == "geometry_unavailable"


def test_the_seam_is_deterministic_for_the_same_session(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    first = m9c.measure_image_derived_lattice(session=session, pose_graph=pose_graph)
    second = m9c.measure_image_derived_lattice(session=session, pose_graph=pose_graph)
    assert isinstance(first, m9c.ImageDerivedLatticeResult)
    assert isinstance(second, m9c.ImageDerivedLatticeResult)
    assert first.result_sha256 == second.result_sha256
    assert len(first.readings) == len(second.readings)


def test_frame_immutability_and_pixel_digest_survive_the_measurement(monkeypatch):
    session, _pose, _result_value = _result(monkeypatch)
    for frame in session.frames:
        assert frame.image is not None
        assert frame.image.flags.writeable is False
        assert session_mod.image_sha256(frame.image) == frame.image_sha256


def test_the_result_carries_no_v1_v2_v3_or_placement_records(monkeypatch):
    _session, _pose, result = _result(monkeypatch)
    assert not hasattr(result, "universe")
    assert not hasattr(result, "volume")
    assert not hasattr(result, "placement")
    record = result.to_record()
    assert "universe" not in record and "placement" not in record
    assert set(record) == {
        "schema", "producer", "terminal", "status", "reason", "detail",
        "session_sha256", "pose_graph_sha256", "graph_sha256", "frame_ids",
        "camera_calibrations", "reading_universe_sha256", "bridge_sha256",
        "abstentions", "result_sha256"}


# ---------------------------------------------------------------------------
# Abstention controls
# ---------------------------------------------------------------------------

def test_a_pixel_free_session_is_typed_image_unavailable(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    stripped = session_mod.image_derived_session_from_record(
        session_mod.image_derived_session_to_record(session))
    result = m9c.measure_image_derived_lattice(session=stripped, pose_graph=pose_graph)
    assert isinstance(result, m9c.ImageDerivedLatticeResult)
    assert result.status == "unavailable"
    assert result.reason == "image_unavailable"
    assert result.terminal == m9c.GEOMETRY_UNAVAILABLE
    assert result.readings == () and result.bridge is None


def test_a_refused_pose_graph_is_typed_pose_unavailable(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    refused = replace(
        pose_graph, status="refused",
        reason=pose_graph_mod.POSE_GRAPH_SOURCE_REFUSED, detail="source refused",
        poses=(), components=(), pose_graph_sha256=None)
    result = m9c.measure_image_derived_lattice(session=session, pose_graph=refused)
    assert isinstance(result, m9c.ImageDerivedLatticeResult)
    assert result.status == "unavailable"
    assert result.reason == "pose_unavailable"
    assert result.readings == () and result.bridge is None


def test_an_image_without_lattice_structure_is_typed_no_lattice_reading(monkeypatch):
    blank = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    session = _build_session(monkeypatch, image=blank)
    pose_graph = _pose_graph(session)
    result = m9c.measure_image_derived_lattice(session=session, pose_graph=pose_graph)
    assert isinstance(result, m9c.ImageDerivedLatticeResult)
    assert result.status == "unavailable"
    assert result.reason == "no_lattice_reading"
    assert result.abstentions, "a blank frame must record why it produced no reading"
    assert result.bridge is None


def test_a_refused_split_producer_is_typed_provenance_incomplete(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)

    def refused_split(*_args, **_kwargs):
        return SimpleNamespace(status="refused")

    monkeypatch.setattr(lp, "produce_split_phase_evidence", refused_split)
    result = m9c.measure_image_derived_lattice(session=session, pose_graph=pose_graph)
    assert isinstance(result, m9c.ImageDerivedLatticeResult)
    assert result.status == "unavailable"
    assert result.reason == "provenance_incomplete"
    assert result.readings == () and result.bridge is None
    assert any(item.reason == "split_source_unavailable" for item in result.abstentions)


def test_the_unavailable_reason_vocabulary_is_closed(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    refused = replace(
        pose_graph, status="refused",
        reason=pose_graph_mod.POSE_GRAPH_SOURCE_REFUSED, detail="source refused",
        poses=(), components=(), pose_graph_sha256=None)
    result = m9c.measure_image_derived_lattice(session=session, pose_graph=refused)
    assert result.reason in m9c.LATTICE_UNAVAILABLE_REASONS


# ---------------------------------------------------------------------------
# Tamper and cross-binding controls
# ---------------------------------------------------------------------------

def test_a_stale_session_digest_is_refused(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    refusal = m9c.measure_image_derived_lattice(
        session=_forge(session, session_sha256="0" * 64), pose_graph=pose_graph)
    assert refusal.reason == "provenance_mismatch"
    assert refusal.detail["field"] == "session"


def test_a_writable_constructor_bypassed_frame_is_refused(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    writable = np.array(session.frames[0].image, copy=True)
    assert writable.flags.writeable is True
    forged_session = _forge(session, frames=(_forge(session.frames[0], image=writable),))
    refusal = m9c.measure_image_derived_lattice(
        session=forged_session, pose_graph=pose_graph)
    assert refusal.reason == "provenance_mismatch"
    assert refusal.detail["field"] == "session.frame.pixels"


def test_a_pose_graph_bound_to_a_foreign_session_is_refused(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    foreign = replace(pose_graph, session_sha256="0" * 64, pose_graph_sha256=None)
    refusal = m9c.measure_image_derived_lattice(session=session, pose_graph=foreign)
    assert refusal.reason == "provenance_mismatch"
    assert refusal.detail["field"] == "pose_graph.session_sha256"


def test_a_pose_graph_for_a_foreign_frame_set_is_refused(monkeypatch):
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    forged = _forge(pose_graph, frame_ids=("frame-other",))
    object.__setattr__(forged, "pose_graph_sha256", pose_graph_mod._digest(
        pose_graph_mod._pose_graph_payload(forged)))
    refusal = m9c.measure_image_derived_lattice(session=session, pose_graph=forged)
    assert refusal.reason == "provenance_mismatch"
    assert refusal.detail["field"] == "pose_graph"


def test_the_sealed_result_is_frozen_and_rejects_a_foreign_digest(monkeypatch):
    _session, _pose, result = _result(monkeypatch)
    with pytest.raises(FrozenInstanceError):
        result.session_sha256 = "0" * 64
    with pytest.raises(ValueError):
        replace(result, result_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(result, status="unavailable")


# ---------------------------------------------------------------------------
# Accepted R4 ingress replay controls (DS-R4-REPLAY-R1)
# ---------------------------------------------------------------------------

def test_an_ingress_bearing_session_replays_through_the_lattice_measurement(
        monkeypatch):
    """A capture-enabled session must not stop at lattice replay."""
    session = _build_session(monkeypatch, ingress=True)
    assert all(frame.ingress is not None for frame in session.frames)
    record = session_mod.image_derived_session_to_record(session)
    assert all("ingress" in item for item in record["frames"])

    pose_graph = _pose_graph(session)
    result = m9c.measure_image_derived_lattice(session=session, pose_graph=pose_graph)
    assert isinstance(result, m9c.ImageDerivedLatticeResult), result
    assert result.terminal == m9c.GEOMETRY_UNAVAILABLE
    assert result.status == "assembled" and result.reason == ""
    assert result.session_sha256 == session.session_sha256
    assert result.pose_graph_sha256 == pose_graph.pose_graph_sha256
    assert result.readings and result.bridge is not None


def test_a_mismatched_record_frame_ingress_pin_is_refused(monkeypatch):
    """The carried record pin is re-validated against the retained pixels."""
    session = _build_session(monkeypatch, ingress=True)
    rebuilt = session_mod.image_derived_session_from_record(
        session_mod.image_derived_session_to_record(session))
    frame = session.frames[0]
    wrong_pixels = np.array(frame.image, copy=True)
    flat = wrong_pixels.reshape(-1)
    flat[0] = (int(flat[0]) + 1) % 256
    assert session_mod.image_sha256(wrong_pixels) != frame.image_sha256
    mismatched = _ingress_pin(frame.frame_id, wrong_pixels)
    forged_record = _forge(rebuilt.frames[0], ingress=mismatched)
    with pytest.raises(m9c._InputRefusal) as caught:
        m9c._replay_frame(frame, forged_record)
    assert caught.value.reason == "provenance_mismatch"
    assert caught.value.detail["field"] == "session.frame.pixels"


def test_a_capture_enabled_session_binds_its_pin_to_the_signature_witness(
        monkeypatch):
    """The session's own ingress pin reaches the ordinary source-tap witness seam.

    Before the replay fix the lattice stage refused an ingress-bearing session, so
    nothing downstream existed.  Here the pin is threaded through the replayed
    lattice into ``derive_image_derived_signatures``: the observed sink never
    refuses as if the pin were absent, while an otherwise identical no-ingress
    control refuses at exactly that ingress gate.  (Whether the sampler witness
    itself then binds is a separate, pre-existing condition outside this task's
    owned paths; see the DS-R4-REPLAY-R1 report.)
    """
    observed = _build_session(monkeypatch, ingress=True)
    observed_result = m9c.measure_image_derived_lattice(
        session=observed, pose_graph=_pose_graph(observed))
    assert isinstance(observed_result, m9c.ImageDerivedLatticeResult)
    assert observed_result.status == "assembled"
    observed_sink: list = []
    observed_signatures = signatures_mod.derive_image_derived_signatures(
        session=observed, lattice_result=observed_result,
        signature_witness_sink=observed_sink)
    assert observed_signatures.status == "assembled"
    assert observed_sink
    frame_ids = {frame.frame_id for frame in observed.frames}
    assert all(item.frame_id in frame_ids for item in observed_sink)
    observed_reasons = {item.reason for item in observed_sink
                        if isinstance(item, csp.SourceTapWitnessRefusalV1)}
    assert ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE not in observed_reasons

    control = _build_session(monkeypatch)
    control_result = m9c.measure_image_derived_lattice(
        session=control, pose_graph=_pose_graph(control))
    assert isinstance(control_result, m9c.ImageDerivedLatticeResult)
    assert control_result.status == "assembled"
    control_sink: list = []
    control_signatures = signatures_mod.derive_image_derived_signatures(
        session=control, lattice_result=control_result,
        signature_witness_sink=control_sink)
    assert control_signatures.status == "assembled"
    assert control_sink
    control_reasons = {item.reason for item in control_sink
                       if isinstance(item, csp.SourceTapWitnessRefusalV1)}
    assert control_reasons == {ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE}


def test_the_default_no_ingress_path_replays_with_unchanged_parity(monkeypatch):
    """The default path carries no pin, needs none, and measures identically."""
    control = _build_session(monkeypatch)
    observed = _build_session(monkeypatch, ingress=True)
    assert all(frame.ingress is None for frame in control.frames)
    assert all(frame.ingress is not None for frame in observed.frames)

    control_payload = session_mod.image_derived_session_to_record(control)
    observed_payload = session_mod.image_derived_session_to_record(observed)
    assert all("ingress" not in item for item in control_payload["frames"])
    for left, right in zip(control_payload["frames"], observed_payload["frames"]):
        assert {key: value for key, value in right.items()
                if key != "ingress"} == left

    # The default replay reconstructs each frame without a pin and does not need
    # one for the canonical payload comparison.
    rebuilt = session_mod.image_derived_session_from_record(control_payload)
    for frame, record_frame in zip(control.frames, rebuilt.frames):
        assert record_frame.ingress is None
        assert m9c._replay_frame(frame, record_frame) is None

    control_result = m9c.measure_image_derived_lattice(
        session=control, pose_graph=_pose_graph(control))
    observed_result = m9c.measure_image_derived_lattice(
        session=observed, pose_graph=_pose_graph(observed))
    assert isinstance(control_result, m9c.ImageDerivedLatticeResult)
    assert isinstance(observed_result, m9c.ImageDerivedLatticeResult)
    assert control_result.status == "assembled"
    assert observed_result.status == "assembled"
    # The pin is additive metadata: calibration and measured reading content are
    # unchanged, and only the session-bound digests differ through the pin.
    assert [m9c._calibration_payload(item)
            for item in control_result.camera_calibrations] == [
        m9c._calibration_payload(item)
        for item in observed_result.camera_calibrations]
    assert [_reading_signature(item)
            for item in control_result.readings] == [
        _reading_signature(item) for item in observed_result.readings]
    assert control_result.session_sha256 != observed_result.session_sha256
