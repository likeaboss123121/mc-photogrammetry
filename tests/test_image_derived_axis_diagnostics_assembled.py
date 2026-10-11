"""One assembled-lattice integration control for the DS-R21 axis diagnostic.

The DS-R21 helper controls cover the diagnostic's own helpers over hand-built
readings.  This control instead drives the public ``diagnose_production_result``
entry point through its *assembled-lattice* branch, so the wiring that DS-R21
left unexercised - signature derivation, ``_connection_constraints`` and the
matcher seam - runs for real:

* a genuine two-frame synthetic session is ray-cast in memory (real ``uint8``
  pixels, the true camera injected through the session's own declared solver
  seam), the accepted connection graph / pose graph are composed from it, and
  ``measure_image_derived_lattice`` measures the lattice from those pixels;
* a valid frozen entry and a valid production result carry that assembled
  lattice;
* only ``plane_graph.match_plane_graph`` is replaced, returning deterministic
  integer-indexed matches keyed to the actual measured readings;
* ``solve_component`` is spied on, so the baseline and the empty-constraint
  ablation are proven to reuse the identical ordered reading/match objects and
  the fixed thresholds, changing only ``quadrant_constraints``.

No real image, capture, fixture, reference or mask file is read, hashed or
processed: the whole scene is rendered from constants in memory.  The control
does not require the synthetic geometry to solve; a typed abstention is a valid
outcome, and the assertions stay independent of solver success.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_lattice_planes import (  # noqa: E402
    ROOM,
    calibration as scene_calibration,
    mixed_material,
    render_room,
)

from mc_photogrammetry.pipeline import image_derived_connections as connections_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_graph_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_production as production_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_session as session_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_only  # noqa: E402
from mc_photogrammetry.reconstruction import calibration as calibration_mod  # noqa: E402
from mc_photogrammetry.reconstruction import plane_graph  # noqa: E402
from mc_photogrammetry.reconstruction.camera import CameraModel  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "diagnose_image_derived_axis_coverage.py"
SPEC = importlib.util.spec_from_file_location("axis_diagnostic_assembled", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)

WIDTH, HEIGHT = 360, 240
_SOLVED_REQUIRED = {"left", "right", "status", "reason", "support", "support_floor",
                    "match_count", "candidate_edges", "mutual_survivors",
                    "right_yaw_class", "hypotheses"}


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


def _synthetic_two_frame_session(monkeypatch):
    """Two translated in-memory renders that share one orientation and true cameras."""
    cameras = ((3.6, 0.7, 3.6), (5.0, 0.7, 5.0))
    images, calibrations, labels = {}, {}, {}
    for index, position in enumerate(cameras):
        frame_id = f"f{index}"
        camera = CameraModel(position=position, yaw_degrees=10.0,
                             pitch_degrees=-25.0, fov_degrees=70.0)
        grey = render_room(camera, WIDTH, HEIGHT, ROOM, materials=mixed_material)
        images[frame_id] = np.repeat(grey[..., None], 3, axis=2)
        calibrations[frame_id] = scene_calibration(camera, width=WIDTH, height=HEIGHT)
        labels[frame_id] = f"{frame_id}.png"
    solution = calibration_mod.to_camera_solution(calibrations)

    def fake_solve(_frames):
        return solution

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    return session_mod.build_image_derived_session(images, source_paths=labels)


def _accepted_graphs(session):
    """The real connection graph and the pose graph composed from it."""
    budgets = connections_mod.make_connection_budgets(
        max_pairs=max(1, len(session.frames)), max_total_support=64)

    def matcher(left_id, _left_image, right_id, _right_image, **_kwargs):
        return _solved_pair(left_id, right_id)

    graph = connections_mod.build_image_derived_connections(
        session, budgets=budgets, matcher=matcher)
    assert graph.status == "ok", graph.reason
    pose_budgets = pose_graph_mod.make_pose_graph_budgets(
        max_edges=4, max_cycle_checks=100, rotation_tolerance_degrees=2.0,
        direction_tolerance=0.05)
    pose_graph = pose_graph_mod.compose_image_derived_pose_graph(
        graph, budgets=pose_budgets)
    assert pose_graph.status == "ok", pose_graph.reason
    return graph, pose_graph


def _indexed_matches(readings):
    """Deterministic matches indexed against the actual measured readings.

    One same-axis, relative-quadrant-zero match per axis that both frames really
    measured.  Endpoints are the canonical per-frame reading indices the solver
    itself uses, and each selected ``b`` is one of that reading's own retained
    alternatives, so the real solver sees feasible, self-consistent endpoints.
    """
    read_by_index, index_by_object, frames = plane_graph._canonical_reading_index(readings)
    del read_by_index
    by_frame: dict[str, dict[int, tuple]] = {}
    for reading in readings:
        frame, position = index_by_object[id(reading)]
        by_frame.setdefault(frame, {}).setdefault(int(reading.axis), []).append(
            (position, reading))
    assert len(frames) == 2, frames
    left_frame, right_frame = frames
    shared = sorted(set(by_frame.get(left_frame, {})) &
                    set(by_frame.get(right_frame, {})))
    matches = []
    for axis in shared:
        left_position, left = by_frame[left_frame][axis][0]
        right_position, right = by_frame[right_frame][axis][0]
        matches.append(plane_graph.PlaneMatch(
            left=left.frame, right=right.frame,
            left_axis=int(left.axis), right_axis=int(right.axis),
            left_b=float(left.alternatives[0].blocks_per_period),
            right_b=float(right.alternatives[0].blocks_per_period),
            quadrant_of_right=0, psr=4.0, peak=0.8, second_peak=0.2,
            overlap_cells=64, integer_residual=None, score=0.75,
            left_plane=int(left_position), right_plane=int(right_position),
            cell_shift_rc=(0, 0)))
    assert matches, "the synthetic session must share at least one measured axis"
    return tuple(matches)


def _entry(session, graph, pose_graph):
    labels = tuple(frame.source_label for frame in session.frames)
    reason, detail = entry_mod._terminal_reason(graph, pose_graph)
    return entry_mod.ImageDerivedEntryResult(
        schema=entry_mod.IMAGE_DERIVED_ENTRY_SCHEMA,
        producer=entry_mod.IMAGE_DERIVED_ENTRY_PRODUCER,
        terminal=entry_mod.GEOMETRY_UNAVAILABLE,
        reason=reason, detail=detail,
        project_path="synthetic-assembled-branch.mcscan",
        source_paths=labels,
        connection_budgets=graph.budgets,
        pose_budgets=pose_graph.budgets,
        session=session, graph=graph, pose_graph=pose_graph)


def _snapshot(result):
    return (result.stage, result.status, result.reason, result.detail,
            result.result_sha256, result.entry.graph.graph_sha256)


@pytest.fixture()
def assembled_result(monkeypatch):
    """A valid production result whose carried lattice is genuinely assembled."""
    session = _synthetic_two_frame_session(monkeypatch)
    graph, pose_graph = _accepted_graphs(session)
    lattice = measure_mod.measure_image_derived_lattice(
        session=session, pose_graph=pose_graph)
    assert isinstance(lattice, measure_mod.ImageDerivedLatticeResult), lattice
    assert lattice.status == "assembled", lattice.reason
    assert lattice.graph_sha256 == graph.graph_sha256
    entry = _entry(session, graph, pose_graph)
    return production_mod._make(
        entry, stage="lattice", status="geometry_unavailable",
        reason="lattice_geometry_deferred", detail="", lattice=lattice)


def test_assembled_branch_emits_evidence_and_reuses_ablation_identities(
        monkeypatch, assembled_result):
    captured: dict = {}
    calls: list = []

    def fake_match(readings, signatures, *, max_shift_cells=32, **_kwargs):
        del signatures
        captured["readings"] = readings
        captured["matches"] = _indexed_matches(readings)
        captured["max_shift_cells"] = max_shift_cells
        return captured["matches"]

    real_solve = plane_graph.solve_component

    def spy(readings, matches, **kwargs):
        calls.append((readings, matches, dict(kwargs)))
        return real_solve(readings, matches, **kwargs)

    monkeypatch.setattr(plane_graph, "match_plane_graph", fake_match)
    monkeypatch.setattr(plane_graph, "solve_component", spy)

    before = _snapshot(assembled_result)
    payload = subject.diagnose_production_result(assembled_result)
    after = _snapshot(assembled_result)

    # 1. The public entry point really reached the assembled-lattice branch.
    assert captured["max_shift_cells"] == subject.MAX_SHIFT_CELLS == 32
    lattice = assembled_result.lattice
    readings = tuple(lattice.readings)
    assert sum(payload["reading_counts"].values()) == len(readings)
    assert payload["reading_counts"] == subject._counts(
        reading.axis for reading in readings)

    # 2. Per-pair evidence came from the real carried connection graph.  The
    #    graph's own records are the oracle: every emitted ID/status/reason/
    #    support/support_floor must reproduce its record exactly, in the
    #    diagnostic's canonical (left_id, right_id) order, and the directly
    #    asserted counts must be grounded in those same records.
    graph = assembled_result.entry.graph
    evidence = payload["pair_evidence"]
    assert isinstance(evidence, list) and evidence
    assert all(_SOLVED_REQUIRED <= set(item) for item in evidence)
    graph_records = sorted(graph.evidence,
                           key=lambda record: (record.left_id, record.right_id))
    assert len(evidence) == len(graph_records)
    for item, record in zip(evidence, graph_records):
        assert item["left"] == record.left_id
        assert item["right"] == record.right_id
        assert item["status"] == record.status
        assert item["reason"] == record.reason
        assert item["support"] == record.support
        assert item["support_floor"] == record.support_floor
    assert payload["solved_edge_count"] == sum(
        1 for record in graph_records if record.status == "solved")
    assert payload["solved_edge_count"] >= 1

    # 3. The solved constraint rows are the graph's own solved yaw classes.
    expected_rows = [[record.left_id, record.right_id, record.right_yaw_class]
                     for record in graph.evidence if record.status == "solved"]
    assert payload["yaw_constraints"] == expected_rows
    assert payload["yaw_constraint_count"] == len(expected_rows) >= 1

    # 4. Non-null ablation and selected-match/solver fields from the real solver.
    ablation = payload["constraint_ablation"]
    assert set(ablation) == {"baseline", "ablated", "changed"}
    for half in ("baseline", "ablated"):
        assert set(ablation[half]) == {
            "fired_gate", "matched_axes", "resultants", "residuals", "quadrants"}
    assert payload["fired_gate"] == ablation["baseline"]["fired_gate"]
    selected = payload["selected_matches"]
    assert len(selected) == len(captured["matches"]) and selected
    for item in selected:
        assert {"index", "left", "right", "left_axis", "right_axis", "left_b",
                "right_b", "quadrant_of_right", "score", "psr", "peak",
                "second_peak", "overlap_cells", "cell_shift_rc",
                "circular_phase_residual", "competing_candidates",
                "leave_one_out"} <= set(item)
        assert {"frame", "plane", "source_region_id"} <= set(item["left"])
        assert {"fired_gate", "matched_axes", "resultants", "residuals",
                "quadrants"} == set(item["leave_one_out"])

    # 5. solve_component was spied on: baseline and ablation reuse the identical
    #    ordered readings/matches, fixed thresholds, and change only constraints.
    readings_arg, matches_arg = captured["readings"], captured["matches"]
    deep = [call for call in calls
            if call[0] is readings_arg and call[1] is matches_arg]
    assert len(deep) >= 3, deep
    constraints = tuple(tuple(row) for row in payload["yaw_constraints"])
    assert deep[0][2]["quadrant_constraints"] == constraints
    ablated_index = next(
        index for index, call in enumerate(deep)
        if call[2].get("quadrant_constraints") == ())
    baseline_call, ablated_call = deep[ablated_index - 1], deep[ablated_index]
    assert baseline_call[2]["quadrant_constraints"] == constraints
    assert baseline_call[0] is readings_arg and ablated_call[0] is readings_arg
    assert baseline_call[1] is matches_arg and ablated_call[1] is matches_arg
    assert all(first is second for first, second in
               zip(baseline_call[1], ablated_call[1]))
    baseline_kwargs = dict(baseline_call[2])
    ablated_kwargs = dict(ablated_call[2])
    baseline_kwargs.pop("quadrant_constraints")
    ablated_kwargs.pop("quadrant_constraints")
    assert baseline_kwargs == ablated_kwargs
    assert baseline_kwargs == {
        "integer_tolerance": subject.INTEGER_TOLERANCE,
        "min_resultant": subject.MIN_RESULTANT,
        "ambiguous_margin": subject.AMBIGUOUS_MARGIN}
    assert (subject.INTEGER_TOLERANCE, subject.MIN_RESULTANT,
            subject.AMBIGUOUS_MARGIN) == (0.15, 0.90, 0.02)
    leave_one_out = [call for call in calls
                     if call[0] is readings_arg and call[1] is not matches_arg]
    assert len(leave_one_out) == len(matches_arg)

    # 6. Deterministic JSON and an unchanged result/digest snapshot.
    second = subject.diagnose_production_result(assembled_result)
    assert payload == second
    assert subject.canonical_json(payload) == subject.canonical_json(second)
    assert after == before
    assert payload["normal_mode"]["unchanged"] is True
    assert payload["normal_mode"]["result_sha256"] == assembled_result.result_sha256
    assert payload["normal_mode"]["graph_sha256"] == graph.graph_sha256
    assert payload["normal_mode"]["reason"] == assembled_result.reason
