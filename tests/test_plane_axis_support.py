"""Synthetic C1 discrete-axis mapping and per-axis support contract tests."""
from __future__ import annotations

from dataclasses import replace
import ast
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph, plane_residue
from mc_photogrammetry.reconstruction import plane_axis_support as c1


TRANSFORMS = (
    ((0, 1), (1, 1), (2, 1)),
    ((2, 1), (1, 1), (0, -1)),
    ((0, -1), (1, 1), (2, -1)),
    ((2, -1), (1, 1), (0, 1)),
)


@pytest.mark.parametrize("quadrant", range(4))
@pytest.mark.parametrize("local_axis", range(3))
def test_public_axis_transform_matches_literal_table(local_axis, quadrant):
    assert plane_graph.axis_transform(local_axis, quadrant) == TRANSFORMS[quadrant][local_axis]


@pytest.mark.parametrize("local_axis,quadrant", [
    (True, 0), (0, True), (1.0, 0), (0, 4), (-1, 0),
    (float("nan"), 0), (0, float("inf")), (None, 0),
])
def test_public_axis_transform_rejects_noncanonical_labels(local_axis, quadrant):
    with pytest.raises(ValueError):
        plane_graph.axis_transform(local_axis, quadrant)


def _reading(frame: str, axis: int, side: float = 1.0) -> lattice_planes.PlaneReading:
    rect = lattice_planes.PlaneRectification(
        axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        focal=100.0, principal_point=(32.0, 24.0), centre=(32.0, 24.0))
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=side,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=16.0, phase_pixels=(0.0, 0.0), extent=4.0, samples=32,
        polygon=None, coverage_cells=16, structural_share=0.8, confidence=0.9,
        rectification=rect)


def _signature(reading: lattice_planes.PlaneReading) -> lattice_planes.PlaneSignature:
    cells = np.arange(25, dtype=np.float32).reshape(5, 5)
    return lattice_planes.PlaneSignature(
        reading=reading, blocks_per_period=1.0, cells=cells,
        interior=np.zeros_like(cells), coverage=np.ones_like(cells, dtype=bool),
        origin_uv=(0.25, -0.5))


def _local_axis_for_world(world_axis: int, quadrant: int) -> int:
    return next(axis for axis in range(3)
                if plane_graph.axis_transform(axis, quadrant)[0] == world_axis)


def _fixture(left_q: int = 0, right_q: int = 1, *, shift=(-3, 4),
             left_side: float = 1.0, right_side: float = 1.0):
    frames = ("a", "b")
    readings = tuple(_reading(frame, axis, left_side if frame == "a" else right_side)
                     for frame in frames for axis in range(3))
    by_frame = {
        frame: tuple(sorted((r for r in readings if r.frame == frame),
                            key=lattice_planes.reading_order_key))
        for frame in frames
    }
    indices = {(r.frame, r.axis): index for frame in frames
               for index, r in enumerate(by_frame[frame])}
    matches = []
    for world_axis in range(3):
        left_axis = _local_axis_for_world(world_axis, left_q)
        right_axis = _local_axis_for_world(world_axis, right_q)
        matches.append(plane_graph.PlaneMatch(
            left="a", right="b", left_axis=left_axis, right_axis=right_axis,
            left_b=1.0, right_b=1.0,
            quadrant_of_right=(right_q - left_q) % 4,
            psr=3.0, peak=0.8, second_peak=0.1, overlap_cells=24,
            integer_residual=None, score=0.7,
            left_plane=indices[("a", left_axis)],
            right_plane=indices[("b", right_axis)], cell_shift_rc=shift))
    blocks = {(r.frame, index): 1.0 for frame in frames
              for index, r in enumerate(by_frame[frame])}
    solution = plane_graph.ComponentSolution(
        frames=frames, quadrants={"a": left_q, "b": right_q},
        translations={"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period=blocks, reason="", alternatives=())
    signatures = tuple(_signature(r) for r in readings)
    return solution, readings, tuple(matches), signatures


@pytest.mark.parametrize("left_q,right_q", [(0, 1), (1, 2), (2, 3), (3, 0)])
def test_builder_maps_signed_integer_shifts_with_right_frame_world_sign(left_q, right_q):
    solution, readings, matches, signatures = _fixture(left_q, right_q)
    result = c1.build_axis_support(solution, readings, matches, signatures)
    assert result.refusal is None
    assert result.frames == ("a", "b")
    assert len(result.normal_face_constraints) == 3
    assert len(result.mapped_cell_shifts) == 6
    assert not result.match_refusals
    for constraint in result.normal_face_constraints:
        assert constraint.delta_translation == (
            constraint.left_world_sign * constraint.left_side
            * constraint.left_distance_blocks
            - constraint.right_world_sign * constraint.right_side
            * constraint.right_distance_blocks)
    for item in result.mapped_cell_shifts:
        expected_axis, expected_sign = plane_graph.axis_transform(
            item.right_local_axis_label, right_q)
        expected_grid = 4 if item.right_local_axis == "u" else -3
        assert (item.world_axis, item.right_world_sign) == (expected_axis, expected_sign)
        assert item.right_grid_shift == expected_grid
        assert item.signed_world_integer == expected_sign * expected_grid
    assert all(frame.connected_to_global_anchor for frame in result.frame_support)
    assert {component.world_axis for component in result.axis_components
            if component.status == "supported"} == {0, 1, 2}


@pytest.mark.parametrize("left_side,right_side", [(1.0, 1.0), (-1.0, 1.0),
                                                    (1.0, -1.0), (-1.0, -1.0)])
def test_normal_metric_constraints_preserve_both_camera_sides(left_side, right_side):
    solution, readings, matches, signatures = _fixture(
        left_side=left_side, right_side=right_side)
    result = c1.build_axis_support(solution, readings, matches, signatures)
    assert result.refusal is None
    assert all(item.left_side == left_side and item.right_side == right_side
               for item in result.normal_face_constraints)


def test_competing_nonselected_quadrant_and_alternative_rows_are_filtered():
    solution, readings, matches, signatures = _fixture()
    losing_q = replace(matches[0], quadrant_of_right=3)
    losing_b = replace(matches[1], left_b=2.0, right_b=2.0)
    result = c1.build_axis_support(solution, readings,
                                   matches + (losing_q, losing_b), signatures)
    baseline = c1.build_axis_support(solution, readings, matches, signatures)
    assert result == baseline
    duplicated = c1.build_axis_support(solution, readings,
                                       matches + (matches[0],), signatures)
    assert duplicated == baseline


def test_quadrant_plus_two_preserves_axis_and_reverses_horizontal_signs():
    for quadrant in range(4):
        for local_axis in range(3):
            axis, sign = plane_graph.axis_transform(local_axis, quadrant)
            opposite_axis, opposite_sign = plane_graph.axis_transform(
                local_axis, (quadrant + 2) % 4)
            assert opposite_axis == axis
            assert opposite_sign == (sign if local_axis == 1 else -sign)


def test_signature_refusals_are_specific_and_foreign_objects_are_rejected():
    solution, readings, matches, signatures = _fixture()
    assert c1.build_axis_support(solution, readings, matches, ()).refusal == "signature_missing"
    duplicate = signatures + (signatures[0],)
    assert c1.build_axis_support(solution, readings, matches, duplicate).refusal == "signature_ambiguous"
    foreign = _reading("a", readings[0].axis)
    foreign_sig = _signature(foreign)
    assert c1.build_axis_support(solution, readings, matches,
                                 (foreign_sig,) + signatures[1:]).refusal \
        == "signature_identity_mismatch"
    assert c1.build_axis_support(solution, readings, matches,
                                 (object(),) + signatures[1:]).refusal \
        == "signature_identity_mismatch"
    invalid_geometry = replace(signatures[0], origin_uv=(float("nan"), 0.0))
    assert c1.build_axis_support(solution, readings, matches,
                                 (invalid_geometry,) + signatures[1:]).refusal \
        == "homography_invalid"


def test_lateral_refusal_retains_actual_p1b_support_and_normal_constraints():
    solution, readings, matches, signatures = _fixture(shift=None)
    result = c1.build_axis_support(solution, readings, matches, signatures)
    assert result.refusal is None
    assert len(result.normal_face_constraints) == 3
    assert not result.mapped_cell_shifts
    assert len(result.match_refusals) == 3
    assert all(value.support.refusal == "cell_shift_absent"
               and value.support.cell_shift_rc is None
               for value in result.match_refusals)
    assert all(component.normal_face_evidence_count == 1
               and component.lateral_shift_evidence_count == 0
               for component in result.axis_components if component.match_identities)
    refusal = result.match_refusals[0]
    other_identity = result.normal_face_constraints[1].match_identity
    with pytest.raises(ValueError):
        replace(refusal, support=replace(refusal.support,
                                        left_frame=other_identity.left_frame,
                                        right_frame=other_identity.right_frame,
                                        left_plane=other_identity.left_plane,
                                        right_plane=other_identity.right_plane,
                                        left_axis=other_identity.left_axis,
                                        right_axis=other_identity.right_axis,
                                        left_b=other_identity.left_b,
                                        right_b=other_identity.right_b))


def test_nonfinite_match_quality_refusal_retains_nonzero_p1b_shift():
    solution, readings, matches, signatures = _fixture()
    malformed_quality = replace(matches[0], psr=float("nan"))
    result = c1.build_axis_support(solution, readings,
                                   (malformed_quality,) + matches[1:], signatures)
    assert result.refusal is None
    refusal = next(value for value in result.match_refusals
                   if value.match_identity.left_plane == matches[0].left_plane)
    assert refusal.refusal == "nonfinite_support"
    assert refusal.support.refusal == "nonfinite_derived_value"
    assert refusal.support.cell_shift_rc == (-3, 4)
    assert refusal.support.nonfinite_fields == ("psr",)


def test_roll_sign_control_maps_admitted_minus_three_plus_four_shift():
    left = np.arange(48, dtype=np.int64).reshape(6, 8)
    right = np.roll(left, (3, -4), axis=(0, 1))
    rows, columns = np.indices(left.shape)
    assert np.array_equal(right, left[(rows - 3) % 6, (columns + 4) % 8])
    solution, readings, matches, signatures = _fixture(shift=(-3, 4))
    result = c1.build_axis_support(solution, readings, matches, signatures)
    assert {(item.right_local_axis, item.right_grid_shift)
            for item in result.mapped_cell_shifts} == {("u", 4), ("v", -3)}
    assert all(item.match_identity.quadrant_of_right == 1
               for item in result.mapped_cell_shifts)


def _multi_frame_fixture(matches_spec):
    frames = ("a", "b", "c")
    readings = tuple(_reading(frame, axis) for frame in frames for axis in range(3))
    by_frame = {frame: tuple(sorted((r for r in readings if r.frame == frame),
                                    key=lattice_planes.reading_order_key))
                for frame in frames}
    indices = {(r.frame, r.axis): index for frame in frames
               for index, r in enumerate(by_frame[frame])}
    matches = tuple(plane_graph.PlaneMatch(
        left=left, right=right, left_axis=axis, right_axis=axis,
        left_b=1.0, right_b=1.0, quadrant_of_right=0,
        psr=3.0, peak=0.8, second_peak=0.1, overlap_cells=24,
        integer_residual=None, score=0.7,
        left_plane=indices[(left, axis)], right_plane=indices[(right, axis)],
        cell_shift_rc=shift)
        for left, right, axis, shift in matches_spec)
    solution = plane_graph.ComponentSolution(
        frames=frames, quadrants={frame: 0 for frame in frames},
        translations={frame: (0.0, 0.0, 0.0) for frame in frames},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={(frame, index): 1.0 for frame in frames
                           for index in range(3)},
        reason="", alternatives=())
    return solution, readings, matches, tuple(_signature(r) for r in readings)


def test_three_frame_cycle_has_one_deterministic_component_per_supported_axis():
    fixture = _multi_frame_fixture((
        ("a", "b", 0, (-3, 4)),
        ("b", "c", 1, (2, 1)),
        ("a", "c", 2, (-1, 5)),
    ))
    result = c1.build_axis_support(*fixture)
    assert result.refusal is None
    for axis in range(3):
        component = next(value for value in result.axis_components
                         if value.world_axis == axis and value.contains_global_anchor)
        assert component.frames == ("a", "b", "c")
        assert len(component.match_identities) == 3
        assert component.status == "supported"
        assert all(value.connected_to_global_anchor
                   for value in result.frame_support if value.world_axis == axis)


def test_axis_edges_outside_global_component_remain_disconnected_and_zero_is_not_support():
    fixture = _multi_frame_fixture((("b", "c", 2, None),))
    result = c1.build_axis_support(*fixture)
    assert result.refusal is None
    support = {(value.frame, value.world_axis): value for value in result.frame_support}
    assert support[("a", 0)].refusal == "no_constraints"
    assert support[("a", 1)].refusal == "no_constraints"
    assert support[("a", 2)].refusal == "no_constraints"
    assert support[("b", 2)].refusal == "disconnected_from_global_gauge"
    assert support[("c", 2)].refusal == "disconnected_from_global_gauge"
    assert fixture[0].translations["b"] == (0.0, 0.0, 0.0)
    assert not support[("b", 0)].connected_to_global_anchor


def test_permutation_determinism_for_readings_matches_and_signatures():
    solution, readings, matches, signatures = _fixture()
    expected = c1.build_axis_support(solution, readings, matches, signatures).to_record()
    actual = c1.build_axis_support(solution, readings[::-1], matches[::-1],
                                   signatures[::-1]).to_record()
    assert actual == expected


def test_duplicate_reading_object_and_duplicate_order_key_are_refused():
    solution, readings, matches, signatures = _fixture()
    repeated_object = readings + (readings[0],)
    assert c1.build_axis_support(solution, repeated_object, matches, signatures).refusal \
        == "reading_missing"
    duplicate_key = readings + (replace(readings[0]),)
    assert c1.build_axis_support(solution, duplicate_key, matches, signatures).refusal \
        == "reading_missing"


def test_normal_face_record_binds_absolute_quadrants_axis_and_signs():
    solution, readings, matches, signatures = _fixture()
    item = c1.build_axis_support(solution, readings, matches, signatures).normal_face_constraints[0]
    with pytest.raises(ValueError):
        replace(item, left_frame_quadrant=1)
    with pytest.raises(ValueError):
        replace(item, right_frame_quadrant=2)
    with pytest.raises(ValueError):
        replace(item, left_world_sign=-item.left_world_sign)
    with pytest.raises(ValueError):
        replace(item, world_axis=(item.world_axis + 1) % 3)


def test_result_rejects_dropped_or_forged_lateral_pair_and_external_endpoint():
    solution, readings, matches, signatures = _fixture()
    result = c1.build_axis_support(solution, readings, matches, signatures)
    with pytest.raises(ValueError):
        replace(result, mapped_cell_shifts=result.mapped_cell_shifts[:-1])
    with pytest.raises(ValueError):
        replace(result, frame_support=result.frame_support[:-1])
    with pytest.raises(ValueError):
        replace(result.frame_support[0], connected_to_global_anchor=False)
    with pytest.raises(ValueError):
        replace(result.axis_components[0], contains_global_anchor=False)
    with pytest.raises(ValueError):
        replace(result, normal_face_constraints=())
    one = result.mapped_cell_shifts[0]
    changed_support = replace(one.support, psr=one.support.psr + 0.25)
    changed_one = replace(one, support=changed_support)
    with pytest.raises(ValueError):
        replace(result, mapped_cell_shifts=(changed_one,) + result.mapped_cell_shifts[1:])
    changed_digest = replace(one, cell_homography_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(result, mapped_cell_shifts=(changed_digest,) + result.mapped_cell_shifts[1:])
    with pytest.raises(ValueError):
        replace(one, right_local_axis_label=(one.right_local_axis_label + 1) % 3)
    with pytest.raises(ValueError):
        replace(one, signed_world_integer=one.signed_world_integer + 1)
    forged = replace(result.normal_face_constraints[0].match_identity,
                     left_frame="outside")
    forged_normal = replace(result.normal_face_constraints[0], match_identity=forged)
    with pytest.raises(ValueError):
        replace(result, normal_face_constraints=(forged_normal,) + result.normal_face_constraints[1:])


def test_match_identity_rejects_boolean_and_nonfinite_b_before_numeric_coercion():
    solution, readings, matches, signatures = _fixture()
    for bad in (True, float("nan"), float("inf")):
        malformed = replace(matches[0], left_b=bad)
        result = c1.build_axis_support(solution, readings,
                                       (malformed,) + matches[1:], signatures)
        assert result.refusal == "phase_shift_identity_mismatch"


def test_zero_selected_rows_and_missing_canonical_alternative_abstain_explicitly():
    solution, readings, matches, signatures = _fixture()
    wrong_q = replace(solution, quadrants={"a": 0, "b": 0})
    assert c1.build_axis_support(wrong_q, readings, matches, signatures).refusal \
        == "relative_quadrant_mismatch"
    wrong_b = replace(solution, blocks_per_period={key: 2.0 for key in solution.blocks_per_period})
    assert c1.build_axis_support(wrong_b, readings, matches, signatures).refusal \
        == "selected_provenance_missing"
    missing_b = replace(solution, blocks_per_period={})
    assert c1.build_axis_support(missing_b, readings, matches, signatures).refusal \
        == "selected_provenance_missing"
    losing_b = replace(matches[0], left_b=2.0, right_b=2.0)
    assert c1.build_axis_support(solution, readings, (losing_b,), signatures).refusal \
        == "alternative_not_selected"


def test_module_does_not_depend_on_evaluator_or_capture_metadata():
    path = Path(__file__).parents[1] / "src" / "mc_photogrammetry" / "reconstruction" / "plane_axis_support.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("evaluator", "dataset", "manifest", "reference", "capture",
                 "camera", "bounds", "cli", "renderer", "export", "placement",
                 "minecraft", "ground_truth")
    assert not any(token in name.lower() for name in imported for token in forbidden)
