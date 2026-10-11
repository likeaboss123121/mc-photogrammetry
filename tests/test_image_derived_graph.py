"""Focused controls for the bound image-derived graph producer."""
from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_lattice_measure import _result, _producer1_record, _historical_bytes  # noqa: E402
from test_image_derived_connections import (  # noqa: E402
    _budgets, _refused_result, _simple_session, _solved_result)
from test_image_derived_graph_attribution import _local_alias_case  # noqa: E402
from test_lattice_planes import (  # noqa: E402
    ROOM, calibration as scene_calibration, mixed_material, render_room)
from mc_photogrammetry.pipeline import image_derived_graph as subject  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_signatures as f1  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_connections as connections  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_graph_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_session as session_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_only  # noqa: E402
from mc_photogrammetry.reconstruction import calibration as calibration_mod  # noqa: E402
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph, plane_placement  # noqa: E402
from mc_photogrammetry.reconstruction.camera import CameraModel  # noqa: E402


@pytest.fixture(scope="module")
def records():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = _result(patch)
        signatures = f1.derive_image_derived_signatures(
            session=session, lattice_result=lattice)
        assert signatures.status == "assembled"
        yield session, lattice, signatures
    finally:
        patch.undo()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def _readings_stand_in(graph, frames):
    """A digest-bound readings carrier for direct adapter controls.

    ``_connection_constraints`` only needs the reading universe's frame ids plus
    the graph's own digest; a lightweight carrier keeps these controls focused on
    the Z4 projection without building a full lattice result.
    """
    readings = tuple(SimpleNamespace(frame=frame, axis=0) for frame in frames)
    return SimpleNamespace(graph_sha256=graph.graph_sha256, readings=readings)


def _constraints_graph(monkeypatch, frame_ids, pairs, *, omitted=()):
    """A real connection graph over ``frame_ids`` with explicit pair outcomes.

    ``pairs`` maps a canonical ``(left, right)`` pair to either a yaw class or
    ``None`` for a typed matcher refusal. ``omitted`` names frames the adapter
    must treat as reading-less (the lattice stand-in reports no reading for
    them), so a solved edge through such a frame is only reachable by
    composition.
    """
    session = _simple_session(monkeypatch, frame_ids)

    def matcher(left, _li, right, _ri, **_kwargs):
        yaw = pairs[(left, right)]
        if yaw is None:
            return _refused_result(left, right, reason="cheirality_failed")
        return _solved_result(left, right, yaw_class=yaw)

    graph = connections.build_image_derived_connections(
        session, budgets=_budgets(), matcher=matcher)
    assert graph.status == "ok", graph.reason
    present = tuple(frame for frame in frame_ids if frame not in omitted)
    return session, graph, _readings_stand_in(graph, present)


def _yaw(refusal):
    return refusal.reason if isinstance(refusal, plane_placement.PlacementRefusal) else None


def test_genuine_fixture_calls_matcher_and_returns_typed_no_match(records):
    session,lattice,signatures=records
    result=subject.derive_image_derived_graph(session=session,lattice_result=lattice,signatures_result=signatures)
    assert isinstance(result,subject.ImageDerivedGraphResult)
    assert result.selection.status == "no" and result.geometry_evaluations == 0
    assert result.reason == result.selection.reason
    assert result.matches == () and result.component_solution is None
    assert (result.session_sha256,result.lattice_sha256,result.signatures_sha256) == (session.session_sha256,lattice.result_sha256,signatures.result_sha256)
    assert result.schema == subject.IMAGE_DERIVED_GRAPH_SCHEMA == "mc-photogrammetry-image-derived-graph-v3"
    assert result.producer == subject.IMAGE_DERIVED_GRAPH_PRODUCER == "m9-g-image-derived-graph-7"
    assert result.correlation_algorithm == plane_graph.PLANE_CORRELATION_ALGORITHM and result.max_shift_cells == 32


def test_producer5_current_replay_and_self_consistent_producer4_rejection(records):
    from mc_photogrammetry.pipeline import image_derived_joint_graph as current
    source=_current_sources();result=subject.derive_image_derived_graph(**source)
    assert result.schema == subject.IMAGE_DERIVED_GRAPH_SCHEMA == current.IMAGE_DERIVED_GRAPH_SCHEMA
    assert result.producer == subject.IMAGE_DERIVED_GRAPH_PRODUCER == current.IMAGE_DERIVED_GRAPH_PRODUCER
    for producer in ("m9-g-image-derived-graph-4","m9-g-image-derived-graph-5","m9-g-image-derived-graph-6"):
        stale=_forge(result,producer=producer)
        stale=_forge(stale,result_sha256=subject._sha(current._payload(stale)))
        assert isinstance(current.replay_image_derived_graph(graph_result=stale,**source),plane_placement.PlacementRefusal)
    assert dataclasses.replace(result) == result
    historical=subject.derive_historical_image_derived_graph(session=records[0],lattice_result=records[1],signatures_result=records[2])
    assert type(historical) is subject.HistoricalImageDerivedGraphResult
    assert historical.producer == "m9-g-image-derived-graph-5" and historical.schema.endswith("v2")


def test_producer1_measure_refuses_before_graph_numerics_and_producer3_binds(records, monkeypatch):
    session, lattice, signatures = records
    current = subject.derive_image_derived_graph(
        session=session, lattice_result=lattice, signatures_result=signatures)
    assert lattice.producer == "m9c-image-derived-lattice-measure-3"
    assert current.lattice_sha256 == lattice.result_sha256
    assert current.signatures_sha256 == signatures.result_sha256
    historical = _producer1_record(lattice)
    before = _historical_bytes(historical)
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: pytest.fail("historical measure must refuse before matching"))
    monkeypatch.setattr(plane_graph, "solve_component",
                        lambda *_a, **_k: pytest.fail("historical measure must refuse before solving"))
    refused = subject.derive_image_derived_graph(
        session=session, lattice_result=historical, signatures_result=signatures)
    assert isinstance(refused, plane_placement.PlacementRefusal) and refused.reason == "provenance_mismatch"
    assert _historical_bytes(historical) == before


def test_v2_psr_payload_is_canonical_for_finite_and_positive_infinity():
    assert subject._psr_payload(5.0) == {
        "value": 5.0, "positive_infinity": False}
    assert subject._psr_payload(float("inf")) == {
        "value": None, "positive_infinity": True}
    assert subject._psr_from_payload(subject._psr_payload(5.0)) == 5.0
    assert subject._psr_from_payload(
        subject._psr_payload(float("inf"))) == float("inf")
    for invalid in (
            {"value": None, "positive_infinity": False},
            {"value": 1.0, "positive_infinity": True},
            {"value": float("nan"), "positive_infinity": False},
            {"value": float("-inf"), "positive_infinity": False}):
        with pytest.raises(ValueError):
            subject._psr_from_payload(invalid)

    match = plane_graph.PlaneMatch(
        "a", "b", 0, 0, 1.0, 1.0, 0, float("inf"), 1.0, 0.0,
        9, None, 0.9, 0, 0, (0, -20))
    before = dataclasses.replace(match)
    encoded = subject._match_payload(match)
    assert encoded[7] == {"value": None, "positive_infinity": True}
    assert match == before


def test_exact_ordered_universe_reaches_matcher(records,monkeypatch):
    source=_current_sources(monkeypatch);seen=[];original=plane_graph.match_plane_graph
    def matcher(readings,signatures,**kwargs):
        seen.append((readings,signatures,kwargs));return original(readings,signatures,**kwargs)
    monkeypatch.setattr(plane_graph,"match_plane_graph",matcher)
    result=subject.derive_image_derived_graph(**source)
    assert result.status == "assembled" and len(seen)==1
    readings,signatures,options=seen[0]
    assert readings is source["lattice_result"].readings
    assert all(a is b.signature for a,b in zip(signatures,source["signatures_result"].signatures))
    assert options == {"max_shift_cells":32}


def test_solver_success_is_retained_with_explicit_parameters(records, monkeypatch):
    result,calls,source=_fixed_graph(monkeypatch)
    assert result.status == "assembled" and result.geometry_evaluations == len(calls) == 1
    assert result.component_solution.scale_known and not result.component_solution.reason
    assert result.geometry_settings == (.15,.90,.02)
    from mc_photogrammetry.pipeline import image_derived_joint_graph as current
    for settings in ({"integer_tolerance":.1},{"min_resultant":.8},{"ambiguous_margin":.01}):
        before=len(calls)
        refused=current.derive_image_derived_graph(**source,**settings)
        assert isinstance(refused,plane_placement.PlacementRefusal) and refused.reason == "invalid_input"
        assert len(calls)==before


def test_only_solved_connection_edges_become_immutable_constraints(monkeypatch):
    session = _simple_session(monkeypatch, ("f0", "f1", "f2"))

    def matcher(left, _li, right, _ri, **_kwargs):
        if (left, right) == ("f1", "f2"):
            return _refused_result(left, right, reason="cheirality_failed")
        return _solved_result(left, right, yaw_class=(1 if right == "f1" else 3))

    graph = connections.build_image_derived_connections(
        session, budgets=_budgets(), matcher=matcher)
    # Every solved edge is read; the adapter projects them onto the frames the
    # lattice actually measured (here all three, so the solved edges survive).
    constraints = subject._connection_constraints(
        graph, session, _readings_stand_in(graph, ("f0", "f1", "f2")))
    assert constraints == (("f0", "f1", 1), ("f0", "f2", 3))
    assert isinstance(constraints, tuple)


# --- DS-R25 Z4 reading-present projection ------------------------------------

def _connection_constraints(monkeypatch, frame_ids, pairs, *, omitted=()):
    session, graph, lattice = _constraints_graph(
        monkeypatch, frame_ids, pairs, omitted=omitted)
    return subject._connection_constraints(graph, session, lattice)

def test_all_observed_solved_edges_reproduce_exactly(monkeypatch):
    constraints = _connection_constraints(
        monkeypatch, ("f0", "f1", "f2"),
        {("f0", "f1"): 1, ("f0", "f2"): 3, ("f1", "f2"): None})
    assert constraints == (("f0", "f1", 1), ("f0", "f2", 3))


def test_omitted_component_root_star_is_anchored_at_first_observed_frame(monkeypatch):
    """DS-R25C: never emit the solved component's reading-less root.

    Solved edges ``f0->f1:1`` and ``f1->f2:1`` make ``f0`` the traversal root of
    the component, but only ``f1``/``f2`` have readings. The star must be rooted
    at the first reading-present frame (``f1``) and compose ``f1->f2`` as
    ``(potential[f2] - potential[f1]) mod 4 == 1``; emitting the full root would
    leak the foreign endpoint ``f0`` into the solver's reading universe.
    """
    session, graph, lattice = _constraints_graph(
        monkeypatch, ("f0", "f1", "f2"),
        {("f0", "f1"): 1, ("f1", "f2"): 1, ("f0", "f2"): None},
        omitted=("f0",))
    observed = {reading.frame for reading in lattice.readings}
    constraints = subject._connection_constraints(graph, session, lattice)
    assert constraints == (("f1", "f2", 1),)
    assert all(endpoint in observed for row in constraints for endpoint in row[:2])


def test_no_readings_and_missing_constraint_graph_preserve_no_constraints(monkeypatch):
    session = _simple_session(monkeypatch, ("f0", "f1"))
    solved = connections.build_image_derived_connections(
        session, budgets=_budgets(),
        matcher=lambda left, _li, right, _ri, **_kw:
        _solved_result(left, right, yaw_class=1))
    bound = _readings_stand_in(solved, ("f0", "f1"))
    # No connection graph at all: the adapter's documented no-graph contract.
    assert subject._connection_constraints(None, session, bound) == ()
    # A bound graph whose lattice measured no readings emits nothing.
    assert subject._connection_constraints(
        solved, session, _readings_stand_in(solved, ())) == ()


@pytest.mark.parametrize("pairs, expected", (
    ({("f0", "f1"): 1, ("f0", "f2"): None, ("f1", "f2"): 2},
     (("f0", "f2", 3),)),
    ({("f0", "f1"): 3, ("f0", "f2"): None, ("f1", "f2"): 3},
     (("f0", "f2", 2),)),
))
def test_omitted_middle_frame_is_composed_as_a_star(monkeypatch, pairs, expected):
    constraints = _connection_constraints(
        monkeypatch, ("f0", "f1", "f2"), pairs, omitted=("f1",))
    assert constraints == expected
    assert all("f1" not in row for row in constraints)


def test_reverse_traversal_negates_the_solved_yaw_modulo_four(monkeypatch):
    """A star member reachable only by traversing its solved edge backwards.

    ``f0`` solves to ``f2`` with yaw ``1`` and ``f1`` solves to ``f2`` with yaw
    ``3`` (``k_f2 - k_f1 == 3``), so ``f1`` has no edge to ``f0``. The canonical
    traversal reaches ``f1`` from ``f2`` over the *reverse* of its recorded
    relation: ``values[f1] = values[f2] + (-3) % 4 == 1 + 1 == 2``. The emitted
    star relation ``f0->f1`` is therefore ``values[f1] - values[f0] == 2``,
    i.e. ``1 - 3 == -2 == 2 (mod 4)``. Replaying the forward yaw instead of its
    negation would compose ``1 + 3 == 0`` and fail this regression.
    """
    constraints = _connection_constraints(
        monkeypatch, ("f0", "f1", "f2"),
        {("f0", "f1"): None, ("f0", "f2"): 1, ("f1", "f2"): 3})
    assert constraints == (("f0", "f1", 2), ("f0", "f2", 1))


def test_consistent_alternate_paths_compose_to_the_direct_relation(monkeypatch):
    constraints = _connection_constraints(
        monkeypatch, ("f0", "f1", "f2"),
        {("f0", "f1"): 1, ("f1", "f2"): 2, ("f0", "f2"): 3})
    assert constraints == (("f0", "f1", 1), ("f0", "f2", 3))
    assert (1 + 2) % 4 == 3


@pytest.mark.parametrize("omitted", ((), ("f1",)))
def test_direct_and_composed_path_contradiction_refuses_atomically(
        monkeypatch, omitted):
    result = _connection_constraints(
        monkeypatch, ("f0", "f1", "f2"),
        {("f0", "f1"): 1, ("f1", "f2"): 1, ("f0", "f2"): 3}, omitted=omitted)
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.reason == "quadrant_constraint_inconsistent"
    assert result.detail["field"] == "connection_graph.quadrant_constraints"


def test_inconsistent_cycle_refusal_uses_the_normal_validated_builder(monkeypatch):
    """The adapter's Z4 refusal is exactly the placement module's builder output.

    DS-R25's adapter previously constructed ``PlacementRefusal`` directly and
    handcrafted its detail/refusal digests because the solver token was outside
    the closed placement enum. The additive enum entry makes the ordinary
    validated builder usable, so the adapter refusal must replay as the exact
    record ``plane_placement._make_refusal`` emits -- not a direct constructor
    and not a handcrafted digest. ``_make_refusal`` silently normalizes an
    unknown token to ``invalid_input``, so this equality also proves the token is
    a genuine closed-enum member that survives validation.
    """
    result = _connection_constraints(
        monkeypatch, ("f0", "f1", "f2"),
        {("f0", "f1"): 1, ("f1", "f2"): 1, ("f0", "f2"): 3}, omitted=("f1",))
    assert isinstance(result, plane_placement.PlacementRefusal)
    expected = plane_placement._make_refusal(
        "quadrant_constraint_inconsistent",
        {"field": "connection_graph.quadrant_constraints"})
    assert result.to_record() == expected.to_record()
    assert result.refusal_sha256 == expected.refusal_sha256
    assert result.reason in plane_placement.V2_PLACEMENT_REFUSAL_REASONS
    assert result.reason != "invalid_input"


def test_refused_edges_and_their_omitted_endpoint_emit_nothing(monkeypatch):
    constraints = _connection_constraints(
        monkeypatch, ("f0", "f1", "f2"),
        {("f0", "f1"): None, ("f0", "f2"): 3, ("f1", "f2"): None},
        omitted=("f1",))
    assert constraints == (("f0", "f2", 3),)
    assert all(row[:2] != ("f0", "f1") and row[:2] != ("f1", "f2")
               for row in constraints)


def test_disconnected_components_and_singletons_keep_independent_gauges(monkeypatch):
    constraints = _connection_constraints(
        monkeypatch, ("f0", "f1", "f2", "f3"),
        {("f0", "f1"): 2, ("f0", "f2"): None, ("f0", "f3"): None,
         ("f1", "f2"): None, ("f1", "f3"): None, ("f2", "f3"): 1})
    assert constraints == (("f0", "f1", 2), ("f2", "f3", 1))
    # Each component's star is rooted at its own earliest reading-present frame,
    # so both gauges are anchored at zero and neither relation is recomputed
    # against the other component.
    assert [row[0] for row in constraints] == ["f0", "f2"]


def test_observed_singleton_with_no_solved_edge_emits_nothing(monkeypatch):
    """A reading-present frame with no solved incident edge is an isolated node.

    ``f0``/``f1`` solve while both of ``f2``'s pairs are refused, so ``f2`` has
    a reading but no solved incident edge: in the solved graph it is a singleton
    component of its own, yet a singleton carries no pair relation, so it emits
    no pair constraint. It must contribute no constraint row and must not disturb
    the ``f0``/``f1`` star; the adapter emits only relations the solved graph
    actually carries.
    """
    session, graph, lattice = _constraints_graph(
        monkeypatch, ("f0", "f1", "f2"),
        {("f0", "f1"): 2, ("f0", "f2"): None, ("f1", "f2"): None})
    observed = {reading.frame for reading in lattice.readings}
    assert observed == {"f0", "f1", "f2"}
    constraints = subject._connection_constraints(graph, session, lattice)
    assert constraints == (("f0", "f1", 2),)
    assert all("f2" not in row for row in constraints)


def test_yaw_and_constraint_order_are_deterministic(monkeypatch):
    session, graph, lattice = _constraints_graph(
        monkeypatch, ("f0", "f1", "f2", "f3"),
        {("f0", "f1"): 3, ("f0", "f2"): 1, ("f0", "f3"): 2,
         ("f1", "f2"): 2, ("f1", "f3"): 3, ("f2", "f3"): 1})
    first = subject._connection_constraints(graph, session, lattice)
    second = subject._connection_constraints(graph, session, lattice)
    assert first == second
    assert first == (("f0", "f1", 3), ("f0", "f2", 1), ("f0", "f3", 2))


@pytest.mark.parametrize("bad", (True, False, 1.0, 2.0, "1", -1, 4, 7, None))
def test_malformed_yaw_class_is_refused_not_coerced(monkeypatch, bad):
    session, graph, _lattice = _constraints_graph(
        monkeypatch, ("f0", "f1"), {("f0", "f1"): 1})
    forged = _forge(
        graph, evidence=(_forge(graph.evidence[0], right_yaw_class=bad),))
    result = subject._connection_constraints(
        forged, session, _readings_stand_in(graph, ("f0", "f1")))
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.reason == "invalid_input"
    assert result.detail["field"] == "connection_graph.yaw_class"


def test_strict_yaw_class_accepts_only_exact_ints_in_range():
    for good in (0, 1, 2, 3, np.int64(2), np.int32(3)):
        assert subject._strict_yaw_class(good) == int(good)
    for bad in (True, False, 1.0, 3.0, "2", -1, 4, None, (1,)):
        assert subject._strict_yaw_class(bad) is None


def test_packaged_r24_shape_reaches_the_real_solver_with_projected_constraints(
        monkeypatch):
    """The adapter-to-solver path over one genuinely measured R24-shaped lattice.

    Four fixed in-memory views reproduce the accepted DS-R24 layout; the fixed
    view ``f1`` measures no lattice reading, so the solved full graph crosses an
    omitted node. The camera solution is injected through the
    ``image_only.solve_image_cameras`` seam and the matcher is replaced with
    deterministic matches keyed to the actual readings; session construction,
    signature derivation, the adapter projection and the real solver run. The
    solver must receive the identical ordered readings and matches under the
    fixed gates, and the projected constraints must name only
    the reading-present frames. Success is deliberately not required: the
    synthetic geometry abstains when axis evidence is insufficient.
    """
    cameras = (
        ("f0", (3.6, 0.7, 3.6)), ("f1", (5.0, 0.7, 3.6)),
        ("f2", (3.6, 0.7, 5.0)), ("f3", (5.0, 0.7, 5.0)))
    width, height = 360, 240
    images, calibrations = {}, {}
    for frame_id, position in cameras:
        camera = CameraModel(position=position, yaw_degrees=10.0,
                             pitch_degrees=-25.0, fov_degrees=70.0)
        grey = render_room(camera, width, height, ROOM, materials=mixed_material)
        images[frame_id] = np.repeat(grey[..., None], 3, axis=2)
        calibrations[frame_id] = scene_calibration(
            camera, width=width, height=height)
    solution = calibration_mod.to_camera_solution(calibrations)

    def fake_solve(_frames):
        return solution

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    session = session_mod.build_image_derived_session(
        images, source_paths={frame_id: f"{frame_id}.png" for frame_id, _ in cameras})
    graph = connections.build_image_derived_connections(
        session, budgets=connections.make_connection_budgets(
            max_pairs=6, max_total_support=64),
        matcher=lambda left, _li, right, _ri, **_kw:
        _solved_result(left, right, yaw_class=0))
    assert graph.status == "ok", graph.reason
    assert len([r for r in graph.evidence if r.status == "solved"]) == 6
    pose_graph = pose_graph_mod.compose_image_derived_pose_graph(
        graph, budgets=pose_graph_mod.make_pose_graph_budgets(
            max_edges=6, max_cycle_checks=100, rotation_tolerance_degrees=2.0,
            direction_tolerance=0.05))
    assert pose_graph.status == "ok", pose_graph.reason
    lattice = measure.measure_image_derived_lattice(
        session=session, pose_graph=pose_graph)
    assert lattice.status == "assembled", lattice.reason
    measured = tuple(sorted({reading.frame for reading in lattice.readings}))
    assert measured == ("f0", "f2", "f3")
    signatures = f1.derive_image_derived_signatures(
        session=session, lattice_result=lattice)
    assert signatures.status == "assembled", signatures.reason

    captured: dict = {}

    def indexed_matches(readings, _signatures, *, max_shift_cells=32):
        captured["readings_arg"] = readings
        read_by_index, index_by_object, frames = plane_graph._canonical_reading_index(
            readings)
        del read_by_index
        by_frame: dict[str, dict[int, tuple]] = {}
        for reading in readings:
            frame, position = index_by_object[id(reading)]
            by_frame.setdefault(frame, {}).setdefault(int(reading.axis), []).append(
                (position, reading))
        matches = []
        for index, left_frame in enumerate(frames):
            for right_frame in frames[index + 1:]:
                for axis in sorted(set(by_frame.get(left_frame, {}))
                                   & set(by_frame.get(right_frame, {}))):
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
        assert matches
        captured["max_shift_cells"] = max_shift_cells
        captured["matches"] = tuple(matches)
        return captured["matches"]

    monkeypatch.setattr(plane_graph,"match_plane_graph",indexed_matches)
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",lambda *_a,**_k:pytest.fail("missing session frame reached geometry"))
    result=subject.derive_image_derived_graph(session=session,lattice_result=lattice,signatures_result=signatures,connection_graph=graph)
    assert captured["readings_arg"] is lattice.readings and captured["max_shift_cells"]==32
    assert subject._connection_constraints(graph,session,lattice)==(("f0","f2",0),("f0","f3",0))
    assert type(result) is subject.ImageDerivedGraphResult and result.status=="unavailable"
    assert result.selection.status=="no" and result.geometry_evaluations==0
    assert result.frame_ids==("f0","f1","f2","f3")
    assert result.matches==() and result.component_solution is None


def test_connection_graph_binding_rejects_foreign_lattice_digest(monkeypatch):
    session = _simple_session(monkeypatch, ("f0", "f1"))
    graph = connections.build_image_derived_connections(
        session, budgets=_budgets(),
        matcher=lambda left, _li, right, _ri, **_kw:
        _solved_result(left, right, yaw_class=0))
    result = subject._connection_constraints(
        graph, session, SimpleNamespace(graph_sha256="0" * 64))
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.reason == "provenance_mismatch"


def test_solver_abstention_is_atomic_unavailable(records, monkeypatch):
    result,calls,_source=_fixed_graph(monkeypatch,_low_resultant)
    assert result.status == "unavailable" and result.reason == "plane_integer_residual"
    assert result.matches == () and result.component_solution is None
    assert result.selection.status == "unique" and result.geometry_evaluations == len(calls) == 1


def _finite_solution(**changes):
    values = dict(
        frames=("f0", "f1"), quadrants={"f0": 0, "f1": 0},
        translations={"f0": (0., 0., 0.), "f1": (1., 0., 0.)},
        axis_residuals={0: 0.0}, axis_resultants={0: 1.0}, scale_known=True,
        blocks_per_period={("f0", 0): 1.0}, reason="",
        alternatives=(plane_graph.ComponentAlternative(
            (0, 0), 1.0, (1.0, 0.0, 0.0), (2, 0, 0)),))
    values.update(changes)
    return plane_graph.ComponentSolution(**values)


@pytest.mark.parametrize("changes,path", (
    ({"translations": {"f0": (0., 0., 0.), "f1": (1., float("nan"), 0.)}},
     "component_solution.translations['f1'][1]"),
    ({"axis_residuals": {0: float("inf")}},
     "component_solution.axis_residuals[0]"),
    ({"axis_resultants": {0: float("-inf")}},
     "component_solution.axis_resultants[0]"),
    ({"blocks_per_period": {("f0", 0): float("nan")}},
     "component_solution.blocks_per_period['f0'][0]"),
    ({"alternatives": (plane_graph.ComponentAlternative(
        (0, 0), float("nan"), (1.0, 0.0, 0.0), (2, 0, 0)),)},
     "component_solution.alternatives[0].cost"),
    ({"alternatives": (plane_graph.ComponentAlternative(
        (0, 0), 1.0, (1.0, float("inf"), 0.0), (2, 0, 0)),)},
     "component_solution.alternatives[0].resultants[1]"),
    ({"quadrants": {"f0": 0, "f1": float("nan")}},
     "component_solution.quadrants['f1']"),
    ({"quadrants": {"f0": float("inf"), "f1": 0}},
     "component_solution.quadrants['f0']"),
    ({"alternatives": (plane_graph.ComponentAlternative(
        (0, float("nan")), 1.0, (1.0, 0.0, 0.0), (2, 0, 0)),)},
     "component_solution.alternatives[0].quadrants[1]"),
    ({"alternatives": (plane_graph.ComponentAlternative(
        (0, 0), 1.0, (1.0, 0.0, 0.0), (2, float("inf"), 0)),)},
     "component_solution.alternatives[0].axis_observations[1]"),
    ({"axis_resultants": {float("-inf"): 1.0}},
     "component_solution.axis_resultants[<non-finite-key>]"),
))
def test_non_finite_solver_output_abstains_without_leaking(records, monkeypatch, changes, path):
    # Preserve the exhaustive field checker, then inject the corresponding value
    # at the real selected-Q evaluator. Quadrants themselves come from the cert.
    assert subject._non_finite_solution_field(_finite_solution(**changes)) == path
    def corrupt(evaluation):
        cost,alt,detail=evaluation; detail=dict(detail)
        for field,target in (("translations","translations"),("axis_residuals","residuals"),
                             ("axis_resultants","resultants"),("blocks_per_period","blocks_per_period")):
            if field in changes:
                current=dict(detail[target])
                for key,value in changes[field].items():
                    if isinstance(key,tuple): key=("a" if key[0]=="f0" else "b",key[1])
                    elif isinstance(key,str):key={"f0":"a","f1":"b"}.get(key,key)
                    current[key]=value
                detail[target]=current
        if "alternatives" in changes:alt=changes["alternatives"][0]
        if "quadrants" in changes:
            bad=next(value for value in changes["quadrants"].values() if not np.isfinite(value))
            alt=dataclasses.replace(alt,quadrants=(0,bad))
        return cost,alt,detail
    result,calls,_source=_fixed_graph(monkeypatch,corrupt)
    assert len(calls)==1
    if isinstance(result,plane_placement.PlacementRefusal):
        assert result.reason == "provenance_mismatch"
        encoded=json.dumps(result.to_record(),allow_nan=False)
    else:
        expected=("plane_integer_residual" if "axis_residuals" in changes or "axis_resultants" in changes
                  else subject.NON_FINITE_SOLVER_OUTPUT)
        assert result.status == "unavailable" and result.reason == expected
        assert result.matches == () and result.component_solution is None
        encoded=json.dumps({"detail":result.detail,"reason":result.reason},allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_finite_solver_output_is_the_positive_control(records, monkeypatch):
    result,calls,_source=_fixed_graph(monkeypatch)
    assert result.status == "assembled" and len(calls)==1
    assert subject._non_finite_solution_field(result.component_solution) is None
    assert result.matches and result.component_solution.reason == ""


def test_strict_json_allow_nan_false_is_preserved():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            subject._sha({"field": value})


@pytest.mark.parametrize("target", ("session", "lattice", "signatures"))
def test_tampered_upstream_digest_is_a_refusal(records, target):
    session, lattice, signatures = records
    values = {"session": session, "lattice": lattice, "signatures": signatures}
    field = {"session": "session_sha256", "lattice": "result_sha256",
             "signatures": "result_sha256"}[target]
    values[target] = _forge(values[target], **{field: "0" * 64})
    result = subject.derive_image_derived_graph(
        session=values["session"], lattice_result=values["lattice"],
        signatures_result=values["signatures"])
    assert isinstance(result, plane_placement.PlacementRefusal)
    assert result.reason == "provenance_mismatch"


def test_reordered_and_foreign_signature_evidence_are_refused(records):
    session, lattice, signatures = records
    reordered = dataclasses.replace(
        signatures, signatures=tuple(reversed(signatures.signatures)),
        signatures_sha256="", result_sha256="")
    result = subject.derive_image_derived_graph(
        session=session, lattice_result=lattice, signatures_result=reordered)
    assert isinstance(result, plane_placement.PlacementRefusal)
    foreign = _forge(signatures.signatures[0], frame_id="foreign")
    forged = _forge(signatures, signatures=(foreign,) + signatures.signatures[1:])
    result = subject.derive_image_derived_graph(
        session=session, lattice_result=lattice, signatures_result=forged)
    assert isinstance(result, plane_placement.PlacementRefusal)


def test_api_has_no_manual_or_filesystem_surface():
    import inspect
    signature = inspect.signature(subject.derive_image_derived_graph)
    assert tuple(signature.parameters) == (
        "session", "lattice_result", "signatures_result", "connection_graph", "max_shift_cells",
        "integer_tolerance", "min_resultant", "ambiguous_margin")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "subprocess",
                      "pathlib", "open(", "argparse", "sys.argv"):
        assert forbidden not in source


def _abstaining_solution(**changes):
    values = dict(
        reason="plane_integer_residual",
        axis_resultants={0: 0.5, 1: 0.95, 2: 0.95},
        axis_residuals={0: 0.1, 1: 0.1, 2: 0.1})
    values.update(changes)
    return _finite_solution(**values)


def _matched_pair():
    return plane_graph.PlaneMatch(
        "f0", "f1", 0, 0, 1., 1., 0, 5., 1., .1, 16, None, .9, 0, 0)


def test_solver_failure_sink_receives_one_bounded_immutable_snapshot(records, monkeypatch):
    captured=[]
    result,calls,source=_fixed_graph(monkeypatch,_low_resultant,solver_failure_sink=captured.append)
    assert result.reason == "plane_integer_residual" and len(calls)==len(captured)==1
    snapshot=captured[0]
    assert type(snapshot) is subject.SolverFailureSnapshot
    assert snapshot.frames == source["lattice_result"].frame_ids and snapshot.quadrants==(0,0)
    assert dict(snapshot.axis_resultants)=={0:.5,1:1.,2:1.}
    assert snapshot.invalid_field is None
    assert snapshot.active_edge_aliases.recorded and snapshot.active_edge_aliases.active_edge_count==3
    assert snapshot.phase_capture is not None and snapshot.solver_input_bindings is not None
    assert snapshot.boundary_saturation.refusal_reason == subject.BOUNDARY_SATURATION_UNAVAILABLE
    assert snapshot.boundary_saturation.field == "trace"
    assert {f.name for f in dataclasses.fields(snapshot)} == {"reason","frames","quadrants","axis_resultants",
        "axis_residuals","best_alternative_quadrants","best_alternative_axis_observations","invalid_field",
        "active_edge_aliases","boundary_saturation","phase_capture","solver_input_bindings","solve_context"}
    with pytest.raises(dataclasses.FrozenInstanceError):snapshot.reason="changed"


def test_solver_failure_sink_is_silent_on_success_pre_solver_and_guard(records, monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_joint_graph as current
    from test_image_derived_graph_v3 import _source_records
    def run(transform=None,source=None):
        with monkeypatch.context() as patch:
            if source is None:source=_current_sources(patch)
            original=plane_graph._evaluate_assignment;calls=[]
            def evaluate(*args,**kwargs):
                calls.append(1);value=original(*args,**kwargs)
                return value if transform is None else transform(value)
            patch.setattr(plane_graph,"_evaluate_assignment",evaluate)
            plain=current.derive_image_derived_graph(**source)
            calls.clear();captured=[]
            observed=current.derive_image_derived_graph(**source,solver_failure_sink=captured.append)
            assert observed==plain and observed.result_sha256==plain.result_sha256
            return observed,calls,captured
    success,calls,captured=run()
    assert success.status=="assembled" and len(calls)==1 and captured==[]
    with monkeypatch.context() as patch:multiple_source=_source_records(patch,alternatives=True)
    pre,calls,captured=run(source=multiple_source)
    assert pre.selection.status=="multiple" and calls==captured==[]
    numerical,calls,captured=run(_low_resultant)
    assert numerical.reason=="plane_integer_residual" and len(calls)==len(captured)==1
    def nonfinite(evaluation):
        cost,alternative,detail=evaluation;detail=dict(detail)
        t=dict(detail["translations"]);t["b"]=(float("nan"),0.,0.);detail["translations"]=t
        return cost,alternative,detail
    guarded,calls,captured=run(nonfinite)
    assert guarded.reason==subject.NON_FINITE_SOLVER_OUTPUT and len(calls)==1 and captured==[]
    def missing_frame(evaluation):
        cost,alternative,detail=evaluation;detail=dict(detail)
        detail["translations"]={"a":detail["translations"]["a"]}
        return cost,alternative,detail
    guarded,calls,captured=run(missing_frame)
    assert guarded.reason=="graph_frame_mismatch" and len(calls)==1 and captured==[]


def test_public_path_never_builds_a_solver_failure_snapshot(records, monkeypatch):
    def forbidden(*_args,**_kwargs):raise AssertionError("default path allocated failure observations")
    for name in ("_solver_failure_snapshot","_failed_active_edges","_boundary_saturation_attestation","failed_solve_context"):
        monkeypatch.setattr(subject,name,forbidden)
    result,calls,_source=_fixed_graph(monkeypatch,_low_resultant)
    assert result.status == "unavailable" and result.reason == "plane_integer_residual" and len(calls)==1


def test_solver_failure_sink_is_output_neutral_and_solves_once(records, monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_joint_graph as current
    plain,calls,source=_fixed_graph(monkeypatch,_low_resultant)
    calls.clear();captured=[]
    diagnosed=current.derive_image_derived_graph(**source,solver_failure_sink=captured.append)
    assert len(calls)==len(captured)==1
    assert diagnosed==plain and diagnosed.result_sha256==plain.result_sha256
    assert diagnosed.matches==() and diagnosed.component_solution is None


def test_v4_alias_snapshot_is_derived_from_local_inputs_once_and_output_neutral(records, monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_joint_graph as current
    plain,calls,source=_fixed_graph(monkeypatch,_low_resultant);calls.clear();captured=[]
    result=current.derive_image_derived_graph(**source,solver_failure_sink=captured.append)
    assert result==plain and len(calls)==len(captured)==1
    aliases=captured[0].active_edge_aliases
    assert aliases.recorded and aliases.active_edge_count==3 and aliases.conflicting_endpoints==()
    record=aliases.to_record();assert record["input_match_count"]==3
    subject.validate_failed_active_edge_aliases(record)
    json.dumps(record,allow_nan=False)


def _malformed_axis_map_solution():
    return plane_graph.ComponentSolution(
        frames=("f0", "f1"), quadrants={"f0": 0, "f1": 0}, translations={},
        axis_residuals={0: 0.1}, axis_resultants=[(0, 0.5)], scale_known=False,
        blocks_per_period={}, reason="insufficient_axis_coverage",
        alternatives=())


@pytest.mark.parametrize("make_solution,field", (
    (lambda: _finite_solution(
        reason="insufficient_axis_coverage",
        alternatives=(plane_graph.ComponentAlternative(
            (0, 0), 1.0, (0.0, 0.0, 0.0), None),)), None),
    (lambda: _finite_solution(
        reason="insufficient_axis_coverage",
        alternatives=(plane_graph.ComponentAlternative(
            (0, 0), 1.0, (0.0, 0.0, 0.0), 7),)),
     "alternatives[0].axis_observations"),
    (lambda: _malformed_axis_map_solution(), "axis_resultants"),
))
def test_solver_failure_snapshot_extraction_never_raises(records, monkeypatch, make_solution, field):
    plain,calls,_source=_fixed_graph(monkeypatch,_low_resultant)
    # Malformed detached projection is never fed into an accepted component.
    snapshot=subject._solver_failure_snapshot(make_solution())
    assert plain.status == "unavailable" and plain.matches==() and len(calls)==1
    assert snapshot.invalid_field==field
    if field is None:assert snapshot.best_alternative_axis_observations is None


def _production_guard_case(*, quadrants=(0, 0, 0), cycle=False):
    """Source-only constraints with known signed distances on all world axes."""
    frames = ("f0", "f1", "f2")
    translations = {frame: tuple(float(index * (axis + 1)) for axis in range(3))
                    for index, frame in enumerate(frames)}
    assignment = dict(zip(frames, quadrants))
    rectification = lattice_planes.PlaneRectification(
        ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), 100., (0., 0.), (0., 0.))
    readings = []
    by_world = {}
    for frame in frames:
        for local in range(3):
            world, sign = plane_graph.axis_transform(local, assignment[frame])
            face = -20. if world == 2 else 20.
            side = float(sign * (-1 if world == 2 else 1))
            distance = abs(face - translations[frame][world])
            reading = lattice_planes.PlaneReading(
                frame, local, side,
                (lattice_planes.DistanceAlternative(1., distance, 1.),
                 lattice_planes.DistanceAlternative(2., 2. * distance, .5)),
                8., (0., 0.), 1., 16, None, 16, 1., 1., rectification)
            readings.append(reading)
            by_world[(frame, world)] = reading
    # Reversed input order must still resolve original canonical frame indices.
    readings = tuple(reversed(readings))
    _table, indices, _frames = plane_graph._canonical_reading_index(readings)
    matches = []
    pairs = (("f0", "f1"), ("f1", "f2"))
    if cycle:
        pairs += (("f0", "f2"),)
    for left, right in pairs:
        for world in range(3):
            lread, rread = by_world[(left, world)], by_world[(right, world)]
            matches.append(plane_graph.PlaneMatch(
                left, right, lread.axis, rread.axis, 1., 1.,
                (assignment[right] - assignment[left]) % 4,
                5., 1., .1, 16, None, .9,
                indices[id(lread)][1], indices[id(rread)][1]))
    matches = tuple(matches)
    solution = plane_graph.ComponentSolution(
        frames, assignment, translations,
        {axis: 0. for axis in range(3)}, {axis: 1. for axis in range(3)},
        True, _selected_endpoint_map(matches), "", ())
    return frames, readings, matches, solution


def _selected_endpoint_map(matches):
    return {key: value for match in matches
            for key, value in (((match.left, match.left_plane), match.left_b),
                               ((match.right, match.right_plane), match.right_b))}


def _guard_reason(case):
    refusal = subject._production_solution_refusal(*case)
    return refusal[0] if refusal is not None else None


@pytest.mark.parametrize("quadrants", ((0, 0, 0), (0, 1, 2), (0, 3, 1)))
def test_production_guard_valid_full_graph_and_transformed_sides_are_immutable(quadrants):
    case = _production_guard_case(quadrants=quadrants, cycle=True)
    before = copy.deepcopy(case)
    assert subject._non_finite_solution_field(case[3]) is None
    assert _guard_reason(case) is None
    assert case == before
    assert case[3].reason == ""


@pytest.mark.parametrize("mismatch", (
    "graph_extra_frame", "readings_missing_frame", "solution_missing_frame",
    "solution_frame_order", "quadrants_missing", "quadrants_extra",
    "translations_missing", "translations_extra",
))
def test_production_guard_requires_fixed_frame_and_mapping_coverage(mismatch):
    frames, readings, matches, solution = _production_guard_case()
    if mismatch == "graph_extra_frame":
        frames += ("f3",)
    elif mismatch == "readings_missing_frame":
        readings = tuple(reading for reading in readings if reading.frame != "f2")
    elif mismatch == "solution_missing_frame":
        solution = dataclasses.replace(solution, frames=("f0", "f1"))
    elif mismatch == "solution_frame_order":
        solution = dataclasses.replace(solution, frames=tuple(reversed(frames)))
    else:
        name, change = mismatch.rsplit("_", 1)
        mapping = dict(getattr(solution, name))
        if change == "missing":
            del mapping["f2"]
        else:
            mapping["f3"] = 0 if name == "quadrants" else (0., 0., 0.)
        solution = dataclasses.replace(solution, **{name: mapping})
    assert _guard_reason((frames, readings, matches, solution)) == "graph_frame_mismatch"


def test_production_guard_disconnected_axis_with_all_three_global_axes():
    frames, readings, matches, solution = _production_guard_case()
    matches = tuple(match for match in matches
                    if not (match.left == "f1" and match.left_axis == 2))
    assert {match.left_axis for match in matches} == {0, 1, 2}
    solution = dataclasses.replace(solution, blocks_per_period=_selected_endpoint_map(matches))
    assert _guard_reason((frames, readings, matches, solution)) == "graph_translation_unobservable"


def test_production_guard_inactive_yaw_alternative_cannot_connect_an_axis():
    frames, readings, matches, solution = _production_guard_case()
    missing = next(match for match in matches if match.left == "f1" and match.left_axis == 2)
    active = tuple(match for match in matches if match is not missing)
    inactive = dataclasses.replace(missing, quadrant_of_right=1)
    solution = dataclasses.replace(
        solution, blocks_per_period=_selected_endpoint_map(active),
        alternatives=(plane_graph.ComponentAlternative(
            (0, 0, 1), 1., (1., 1., 1.), (4, 4, 4)),))
    assert _guard_reason((frames, readings, active + (inactive,), solution)) == "graph_translation_unobservable"


def test_production_guard_whole_block_cycle_error_is_unwrapped():
    frames, readings, matches, solution = _production_guard_case(cycle=True)
    cycle_edge = next(match for match in matches if match.left == "f0"
                      and match.right == "f2" and match.left_axis == 0)
    table, _indices, _frames = plane_graph._canonical_reading_index(readings)
    original = table[(cycle_edge.right, cycle_edge.right_plane)]
    changed = dataclasses.replace(
        original, alternatives=(lattice_planes.DistanceAlternative(1., 15., 1.),
                                lattice_planes.DistanceAlternative(2., 30., .5)))
    readings = tuple(changed if reading is original else reading for reading in readings)
    error = solution.translations["f2"][0] - solution.translations["f0"][0] - (20. - 15.)
    assert abs(error) == 3. and error % 1. == 0.
    # The forged successful solver metrics remain perfect despite the cycle error.
    assert solution.axis_residuals == {0: 0., 1: 0., 2: 0.}
    assert _guard_reason((frames, readings, matches, solution)) == "graph_equation_inconsistent"


@pytest.mark.parametrize("error,reason", (
    (.149999, None), (.150001, "graph_equation_inconsistent"),
    (-.149999, None), (-.150001, "graph_equation_inconsistent"),
))
def test_production_guard_explicit_edge_error_limit(error, reason):
    frames, readings, matches, solution = _production_guard_case()
    translations = dict(solution.translations)
    translations["f2"] = (2. + error, 4., 6.)
    solution = dataclasses.replace(solution, translations=translations)
    assert subject._PRODUCTION_EDGE_ERROR_LIMIT_BLOCKS == .15
    assert _guard_reason((frames, readings, matches, solution)) == reason


def test_production_guard_endpoint_scale_conflict_across_distinct_edges():
    frames, readings, matches, solution = _production_guard_case()
    changed = tuple(dataclasses.replace(match, left_b=2.)
                    if match.left == "f1" and match.left_axis == 0 else match
                    for match in matches)
    # Last-write-wins is the exact unsupported solver-map case being guarded.
    solution = dataclasses.replace(solution, blocks_per_period=_selected_endpoint_map(changed))
    before = copy.deepcopy(solution)
    assert _guard_reason((frames, readings, changed, solution)) == "graph_endpoint_scale_conflict"
    assert solution == before


@pytest.mark.parametrize("change", ("missing", "extra", "wrong"))
def test_production_guard_solution_scale_map_must_match_active_endpoints(change):
    frames, readings, matches, solution = _production_guard_case()
    mapping = dict(solution.blocks_per_period)
    if change == "missing":
        del mapping[("f1", 0)]
    elif change == "extra":
        mapping[("f2", 99)] = 1.
    else:
        mapping[("f1", 0)] = 2.
    solution = dataclasses.replace(solution, blocks_per_period=mapping)
    assert _guard_reason((frames, readings, matches, solution)) == "graph_endpoint_scale_conflict"


@pytest.mark.parametrize("change", ("index", "axis", "world_axis", "missing_b", "side", "yaw"))
def test_production_guard_rejects_malformed_or_incompatible_active_endpoints(change):
    frames, readings, matches, solution = _production_guard_case()
    replacement = {"index": {"left_plane": 99}, "axis": {"left_axis": 1},
                   "world_axis": {"right_axis": 1, "right_plane": 1},
                   "missing_b": {"left_b": 3.}, "yaw": {"quadrant_of_right": True}}
    if change == "side":
        readings = tuple(dataclasses.replace(reading, side=.5)
                         if reading.frame == "f0" and reading.axis == 0 else reading
                         for reading in readings)
    else:
        matches = (dataclasses.replace(matches[0], **replacement[change]),) + matches[1:]
    assert _guard_reason((frames, readings, matches, solution)) == "graph_active_constraint_invalid"


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_production_guard_nonfinite_equation_error(value):
    frames, readings, matches, solution = _production_guard_case()
    translations = dict(solution.translations)
    translations["f2"] = (value, 4., 6.)
    solution = dataclasses.replace(solution, translations=translations)
    assert _guard_reason((frames, readings, matches, solution)) == "graph_equation_inconsistent"


def test_production_guard_finite_endpoints_cannot_overflow_the_equation():
    frames, readings, matches, solution = _production_guard_case()
    readings = tuple(dataclasses.replace(
        reading, side=-1. if reading.frame == "f1" else 1.,
        alternatives=(lattice_planes.DistanceAlternative(1., 1e308, 1.),))
        if reading.axis == 0 else reading for reading in readings)
    assert _guard_reason((frames, readings, matches, solution)) == "graph_equation_inconsistent"


def _stub_verified_production_inputs(monkeypatch, case):
    """Isolate the production boundary after the unrelated upstream replay checks."""
    frames, readings, matches, solution = case
    session = SimpleNamespace(frames=tuple(SimpleNamespace(frame_id=frame) for frame in frames),
                              session_sha256="1" * 64)
    lattice = SimpleNamespace(readings=readings, frame_ids=frames, status="assembled",
                              session_sha256=session.session_sha256, result_sha256="2" * 64)
    signatures = SimpleNamespace(signatures=(), frame_ids=frames, status="assembled",
                                 session_sha256=session.session_sha256,
                                 lattice_sha256=lattice.result_sha256, result_sha256="3" * 64)
    monkeypatch.setattr(subject, "_replay_session", lambda value: value)
    monkeypatch.setattr(subject, "_replay", lambda value, *_args: value)
    monkeypatch.setattr(subject, "_ordered_universe", lambda *_args: (readings, ()))
    monkeypatch.setattr(plane_graph, "match_plane_graph", lambda *_args, **_kwargs: matches)
    calls = []

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return solution

    monkeypatch.setattr(plane_graph, "solve_component", solve)
    return dict(session=session, lattice_result=lattice, signatures_result=signatures), calls


@pytest.mark.parametrize("failure", (None, "frames", "scale", "disconnected", "equation"))
def test_production_boundary_assembles_or_refuses_atomically_after_one_solve(monkeypatch,failure):
    # Preserve the signed three-view guard controls independently, then exercise
    # the current production boundary on source-validated selected geometry.
    frames,readings,matches,solution=_production_guard_case(quadrants=(0,1,2))
    expected={"frames":"graph_frame_mismatch","scale":"graph_endpoint_scale_conflict",
              "disconnected":"graph_translation_unobservable","equation":"graph_equation_inconsistent"}
    if failure=="frames":frames+=("f3",)
    elif failure=="scale":solution=dataclasses.replace(solution,blocks_per_period={})
    elif failure=="disconnected":
        matches=matches[:3];solution=dataclasses.replace(solution,blocks_per_period=_selected_endpoint_map(matches))
    elif failure=="equation":
        translations=dict(solution.translations);translations["f2"]=(3.,4.,6.)
        solution=dataclasses.replace(solution,translations=translations)
    assert _guard_reason((frames,readings,matches,solution))==expected.get(failure)
    def corrupt(evaluation):
        cost,alternative,detail=evaluation;detail=dict(detail)
        if failure=="frames":detail["translations"]={"a":detail["translations"]["a"]}
        elif failure=="scale":detail["blocks_per_period"]={}
        elif failure=="equation":
            translations=dict(detail["translations"]);translations["b"]=(1.,0.,0.);detail["translations"]=translations
        return cost,alternative,detail
    captured=[]
    if failure=="disconnected":
        from mc_photogrammetry.pipeline import image_derived_joint_graph as current
        source=_current_sources(monkeypatch);evidence=[]
        for item in source["signatures_result"].signatures:
            sig=item.signature
            if item.frame_id=="b" and item.reading_axis==2:
                sig=dataclasses.replace(sig,coverage=f1._readonly_array(np.zeros_like(sig.coverage)))
            evidence.append(dataclasses.replace(item,signature=sig,signature_sha256=""))
        source["signatures_result"]=dataclasses.replace(source["signatures_result"],signatures=tuple(evidence),signatures_sha256="",result_sha256="")
        monkeypatch.setattr(plane_graph,"_evaluate_assignment",lambda *_a,**_k:pytest.fail("disconnected evidence reached geometry"))
        result=current.derive_image_derived_graph(**source,solver_failure_sink=captured.append)
        assert result.selection.status=="no" and result.geometry_evaluations==0
    else:
        result,calls,_source=_fixed_graph(monkeypatch,corrupt,solver_failure_sink=captured.append)
        assert result.geometry_evaluations==len(calls)==1
        if failure is None:assert result.status=="assembled" and result.matches and result.component_solution is not None
        else:assert result.reason==expected[failure]
    assert captured==[]
    if failure is not None:assert result.status=="unavailable" and result.matches==() and result.component_solution is None


def test_production_guard_keeps_solver_refusal_and_finite_validation_precedence(monkeypatch):
    def forbidden(*_a,**_k):raise AssertionError("guard ran before evaluator refusal or finite check")
    monkeypatch.setattr(subject,"_production_solution_refusal",forbidden)
    with monkeypatch.context() as patch:
        refused,calls,_source=_fixed_graph(patch,_low_resultant)
        assert refused.reason=="plane_integer_residual" and len(calls)==1
    def corrupt(evaluation):
        cost,alternative,detail=evaluation;detail=dict(detail)
        t=dict(detail["translations"]);t["b"]=(float("nan"),0.,0.);detail["translations"]=t
        return cost,alternative,detail
    result,calls,_source=_fixed_graph(monkeypatch,corrupt)
    assert result.reason==subject.NON_FINITE_SOLVER_OUTPUT and len(calls)==1


def _current_sources(monkeypatch=None, *, energy=True):
    from test_image_derived_graph_v3 import _source_records
    if monkeypatch is not None:
        cached=getattr(monkeypatch,"_current_fixture_sources",{})
        if energy in cached:return dict(cached[energy])
        source = _source_records(monkeypatch)
    else:
        patch=pytest.MonkeyPatch()
        try:
            source = _source_records(patch)
        finally:
            patch.undo()
    if energy:
        # Explicit synthetic energy evidence, identical for corresponding axes
        # across views; never signed Gaussian cells or a fabricated matcher row.
        evidence=[]
        for item in source["signatures_result"].signatures:
            cells=np.abs(item.signature.cells)
            if item.frame_id == "b":
                cells=np.abs(cells + .15*np.random.default_rng(500+item.reading_axis).normal(size=cells.shape))
            signature=dataclasses.replace(item.signature,
                cells=f1._readonly_array(cells.astype(np.float32)))
            evidence.append(dataclasses.replace(item,signature=signature,signature_sha256=""))
        source["signatures_result"]=dataclasses.replace(source["signatures_result"],
            signatures=tuple(evidence),signatures_sha256="",result_sha256="")
    if monkeypatch is not None:
        cached[energy]=source
        monkeypatch._current_fixture_sources=cached
    return dict(source)


def _current_graph(monkeypatch=None):
    return subject.derive_image_derived_graph(**_current_sources(monkeypatch))


def _low_resultant(evaluation):
    cost,alternative,detail=evaluation
    detail=dict(detail);detail["resultants"]={0:.5,1:1.,2:1.}
    return cost,alternative,detail


def _fixed_graph(monkeypatch, transform=None, **diagnostic_options):
    """Genuine sources/matcher/selection with only fixed-Q numerical injection."""
    from mc_photogrammetry.pipeline import image_derived_joint_graph as current
    source=_current_sources(monkeypatch)
    original=plane_graph._evaluate_assignment;calls=[]
    def evaluate(*args,**kwargs):
        calls.append((args,kwargs));value=original(*args,**kwargs)
        return value if transform is None else transform(value)
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",evaluate)
    result=current.derive_image_derived_graph(**source,**diagnostic_options)
    return result,calls,source


def _current_unavailable():
    """Independent valid source universe with no admitted support."""
    source=_current_sources();evidence=[]
    for item in source["signatures_result"].signatures:
        sig=dataclasses.replace(item.signature,coverage=f1._readonly_array(np.zeros_like(item.signature.coverage)))
        evidence.append(dataclasses.replace(item,signature=sig,signature_sha256=""))
    source["signatures_result"]=dataclasses.replace(source["signatures_result"],signatures=tuple(evidence),signatures_sha256="",result_sha256="")
    result=subject.derive_image_derived_graph(**source)
    assert result.status=="unavailable" and result.selection.status=="no" and result.geometry_evaluations==0
    return result
