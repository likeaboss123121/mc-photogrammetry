"""Focused controls for the DS-M9-C validating lattice/provenance adapter.
Fixtures are the accepted in-memory records from the DS-M9-B and session suites:
a session built from real uint8 frames through the session producer, a
deliberately non-square lattice reading with retained region/phase evidence,
per-frame calibrations and proposals, and an accepted pose graph. The adapter
must replay and cross-bind them and never measure or synthesize a record.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_image_derived_session import _solution  # noqa: E402
from test_plane_geometry_provenance import (  # noqa: E402
    AXIS, CENTRE, EXTENT, FRAME, PERIOD, SAMPLES, _forge, _phase, _split_source,
    _structural_patch, _weights)

from mc_photogrammetry.pipeline import image_only  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_lattice as m9c  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_pose_graph as poses  # noqa: E402
from mc_photogrammetry.pipeline.image_derived_session import (  # noqa: E402
    build_image_derived_session)
from mc_photogrammetry.reconstruction import (  # noqa: E402
    boundary_candidates as bc, calibration as calibration_module, lattice_planes as lp)
from mc_photogrammetry.reconstruction.calibration import CalibrationResult  # noqa: E402

OTHER = "frame-1"
HEIGHT, WIDTH = 12, 8          # deliberately non-square: height != width
PRINCIPAL = (WIDTH / 2.0, HEIGHT / 2.0)
POLY = ((0.0, 0.0), (WIDTH - 1.0, 0.0), (WIDTH - 1.0, HEIGHT - 1.0), (0.0, HEIGHT - 1.0))
SOURCES = ("probe-region",)
_MISSING = object()

def _mask():
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[0, 0] = mask[-1, -1] = mask[2, 3] = True
    return mask


def _calibration(**overrides):
    payload = dict(focal_pixels=20.0, yaw_degrees=0.0, pitch_degrees=0.0,
                   width=WIDTH, height=HEIGHT, inliers=100, segments=200,
                   median_residual=0.5, roll_degrees=0.0, principal_point=PRINCIPAL)
    payload.update(overrides)
    return CalibrationResult(**payload)


def _region():
    return lp._make_region_evidence(_mask(), AXIS, POLY, SOURCES)

def _reading(cal, region):
    patch = _structural_patch()
    phase = _phase(patch, region)
    axes = calibration_module.axis_directions(
        cal.yaw_degrees, cal.pitch_degrees, cal.roll_degrees)
    principal = cal.principal_point or (cal.width / 2.0, cal.height / 2.0)
    rectification = lp.PlaneRectification(
        axis_matrix=tuple(tuple(float(value) for value in row) for row in axes),
        focal=float(cal.focal_pixels),
        principal_point=(float(principal[0]), float(principal[1])), centre=CENTRE)
    return lp.PlaneReading(
        frame=FRAME, axis=AXIS, side=1.0,
        alternatives=(lp.DistanceAlternative(1.0, 0.5, 1.0),),
        period_pixels=PERIOD,
        phase_pixels=tuple(float(value) for value in phase.phase_uv_pixels),
        extent=EXTENT, samples=SAMPLES, polygon=POLY, coverage_cells=int(_mask().sum()),
        structural_share=float(lp._structural_share(patch, PERIOD)), confidence=1.0,
        rectification=rectification, region_evidence=region, phase_evidence=phase)

def _proposals(*, axis=AXIS, status="ok"):
    proposal = bc.PlaneProposal(axis=axis, side=1, polygon=POLY, mask=_mask(),
                                strength=1.0, sources=SOURCES, coverage=1.0)
    return bc.PlaneProposalResult(status=status, reason="image_evidence",
                                  proposals=(proposal,) if status == "ok" else ())


def _session(monkeypatch, frame_ids=(FRAME,)):
    monkeypatch.setattr(image_only, "solve_image_cameras",
                        lambda frames: _solution(list(frames), image_size=(WIDTH, HEIGHT),
                                                 principal=PRINCIPAL))
    return build_image_derived_session(
        {frame_id: np.zeros((HEIGHT, WIDTH), dtype=np.uint8) for frame_id in frame_ids})


def _pose_graph(session, *, session_sha256=_MISSING, frame_id=None, unresolved=None):
    graph_sha = "a" * 64
    frame_ids = (frame_id,) if frame_id is not None else tuple(session.cameras)
    gauge_poses = tuple(poses.GaugeCameraPose(
        name, 0, frame_ids[0], ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (0.0, 0.0, 0.0), graph_sha) for name in frame_ids)
    return poses.ImageDerivedPoseGraph(
        schema=poses.IMAGE_DERIVED_POSE_GRAPH_SCHEMA,
        version=poses.IMAGE_DERIVED_POSE_GRAPH_VERSION, status="ok", reason="", detail="",
        graph_sha256=graph_sha,
        session_sha256=(session.session_sha256 if session_sha256 is _MISSING
                        else session_sha256),
        budgets=poses.make_pose_graph_budgets(
            max_edges=0, max_cycle_checks=0, rotation_tolerance_degrees=0.0,
            direction_tolerance=0.0),
        frame_ids=frame_ids, components=(frame_ids,), poses=gauge_poses,
        unresolved=(unresolved or session.unresolved), metric_scale_known=False)


def _records(monkeypatch, frame_ids=(FRAME,)):
    cal, region = _calibration(), _region()
    reading = _reading(cal, region)
    return _session(monkeypatch, frame_ids), cal, region, reading, _weights(reading)

def _bind(session, cal, region, reading, weights, *,
          calibrations=None, proposals=None, pose=_MISSING):
    return m9c.assemble_image_derived_lattice(
        session=session, pose_graph=_pose_graph(session) if pose is _MISSING else pose,
        calibrations={FRAME: cal} if calibrations is None else calibrations,
        proposals={FRAME: _proposals()} if proposals is None else proposals,
        readings=(reading,), structural_weights=(weights,),
        split_sources=(_split_source(region),))


def test_complete_records_seal_one_result_bound_to_the_pose_graph(monkeypatch):
    session, cal, region, reading, weights = _records(monkeypatch)
    result = _bind(session, cal, region, reading, weights)
    assert isinstance(result, m9c.ImageDerivedLatticeResult)
    assert result.session_sha256 == session.session_sha256
    assert result.frame_ids == (FRAME,)
    assert result.pose_graph_sha256 == _pose_graph(session).pose_graph_sha256
    assert result.bridge.readings[0] is reading
    assert m9c._hex64(result.calibrations_sha256) and m9c._hex64(result.proposals_sha256)


def test_the_pose_graph_is_required_and_must_bind_the_session(monkeypatch):
    session, cal, region, reading, weights = _records(monkeypatch)
    missing = _bind(session, cal, region, reading, weights, pose=None)
    assert isinstance(missing, m9c.ImageDerivedLatticeRefusal)
    assert (missing.reason, missing.field) == (m9c.GEOMETRY_UNAVAILABLE, "pose_graph")
    foreign = _bind(session, cal, region, reading, weights,
                    pose=_pose_graph(session, session_sha256="b" * 64))
    assert (foreign.reason, foreign.field) == (
        "provenance_mismatch", "pose_graph.session_sha256")


def test_a_nonsquare_calibration_binds_height_width_to_the_frame(monkeypatch):
    session, cal, region, reading, weights = _records(monkeypatch)
    frame_shape = session.frames[0].image_shape
    assert frame_shape[:2] == (HEIGHT, WIDTH) != (WIDTH, HEIGHT)
    assert (cal.height, cal.width) == frame_shape[:2]
    swapped = _bind(session, cal, region, reading, weights,
                    calibrations={FRAME: _calibration(width=HEIGHT, height=WIDTH)})
    assert (swapped.reason, swapped.field) == (
        "provenance_mismatch", f"calibrations[{FRAME}].dimensions")


@pytest.mark.parametrize("overrides", (
    {"focal_pixels": 21.0}, {"principal_point": (1.0, 1.0)}, {"yaw_degrees": 12.0}))
def test_a_foreign_same_size_calibration_is_refused(monkeypatch, overrides):
    session, cal, region, reading, weights = _records(monkeypatch)
    foreign = _bind(session, cal, region, reading, weights,
                    calibrations={FRAME: _calibration(**overrides)})
    assert (foreign.reason, foreign.field) == (
        "provenance_mismatch", f"calibrations[{FRAME}].rectification")


def test_every_session_frame_needs_a_calibration_and_a_proposal(monkeypatch):
    session, cal, region, reading, weights = _records(monkeypatch, (FRAME, OTHER))
    full_calibrations = {FRAME: cal, OTHER: cal}
    full_proposals = {FRAME: _proposals(), OTHER: _proposals()}
    no_cal = _bind(session, cal, region, reading, weights,
                   calibrations={FRAME: cal}, proposals=full_proposals)
    assert (no_cal.reason, no_cal.field) == (
        m9c.GEOMETRY_UNAVAILABLE, f"calibrations[{OTHER}]")
    no_prop = _bind(session, cal, region, reading, weights,
                    calibrations=full_calibrations, proposals={FRAME: _proposals()})
    assert (no_prop.reason, no_prop.field) == (m9c.GEOMETRY_UNAVAILABLE,
                                               f"proposals[{OTHER}]")


def test_absent_required_records_are_typed_geometry_unavailable(monkeypatch):
    session, cal, region, reading, weights = _records(monkeypatch)
    base = dict(session=session, pose_graph=_pose_graph(session), calibrations={FRAME: cal},
                proposals={FRAME: _proposals()}, readings=(reading,),
                structural_weights=(weights,), split_sources=(_split_source(region),))
    for name, absent in (("session", None), ("readings", ()),
                         ("structural_weights", ()), ("split_sources", ())):
        refusal = m9c.assemble_image_derived_lattice(**{**base, name: absent})
        assert isinstance(refusal, m9c.ImageDerivedLatticeRefusal)
        assert (refusal.reason, refusal.field) == (m9c.GEOMETRY_UNAVAILABLE, name)


def test_tampered_records_are_refused_without_resealing(monkeypatch):
    session, cal, region, reading, weights = _records(monkeypatch)
    share = float(reading.structural_share)
    tampered = _bind(session, cal, region, reading,
                     _weights(reading, structural_share=share + 0.5))
    assert (tampered.reason, tampered.field) == (
        "provenance_mismatch", "structural_weights.structural_share")
    stale_pose = _bind(session, cal, region, reading, weights,
                       pose=_forge(_pose_graph(session), session_sha256="c" * 64))
    assert (stale_pose.reason, stale_pose.field) == ("provenance_mismatch", "pose_graph")
    stale_session = m9c.assemble_image_derived_lattice(
        session=_forge(session, session_sha256="0" * 64), pose_graph=_pose_graph(session),
        calibrations={FRAME: cal}, proposals={FRAME: _proposals()},
        readings=(reading,), structural_weights=(weights,),
        split_sources=(_split_source(region),))
    assert (stale_session.reason, stale_session.field) == ("provenance_mismatch", "session")


def test_a_fresh_valid_pose_graph_with_foreign_frames_or_facts_is_refused(monkeypatch):
    session, cal, region, reading, weights = _records(monkeypatch)
    for graph in (_pose_graph(session, frame_id=OTHER), _pose_graph(session, unresolved=("x",))):
        refusal = _bind(session, cal, region, reading, weights, pose=graph)
        assert (refusal.reason, refusal.field) == ("provenance_mismatch", "pose_graph.frame_ids")
