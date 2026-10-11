from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from mc_photogrammetry.evaluation_static_block_pair import (
    MAX_CANONICAL_OUTPUT_BYTES,
    OPAQUE_FULL_CUBE_NAMES,
    StaticPairError,
    _canonical_bytes,
    _summarize_edges,
    build_report,
    trace_static_ray,
)


def _state(name: str, properties: tuple[tuple[str, str], ...] = ()):
    return (name, properties)


def _grid(xmax: int, ymax: int = 0, zmax: int = 0):
    return {
        (x, y, z): _state("minecraft:air")
        for x in range(xmax + 1)
        for y in range(ymax + 1)
        for z in range(zmax + 1)
    }


def test_nearer_certified_block_wins_before_farther_block():
    grid = _grid(4)
    grid[(1, 0, 0)] = _state("minecraft:stone")
    grid[(3, 0, 0)] = _state("minecraft:dirt")
    hit = trace_static_ray((0.5, 0.5, 0.5), (1.0, 0.0, 0.0), grid, ((0, 0, 0), (4, 0, 0)))
    assert hit["status"] == "hit"
    assert hit["cell"] == [1, 0, 0]
    assert hit["face"] == "west"
    assert hit["point"] == [1.0, 0.5, 0.5]


def test_depth_limit_is_checked_before_occupancy():
    grid = _grid(2)
    grid[(1, 0, 0)] = _state("minecraft:stone")
    hit = trace_static_ray((0.5, 0.5, 0.5), (1.0, 0.0, 0.0), grid, ((0, 0, 0), (2, 0, 0)), max_depth=0.49)
    assert hit == {"status": "unknown", "reason": "max_depth"}


def test_simultaneous_axis_tie_skips_zero_length_side_cells_and_marks_owner_ambiguous():
    grid = _grid(2, 2)
    grid[(1, 0, 0)] = _state("minecraft:glass")  # Side cell is not traversed at a corner tie.
    grid[(1, 1, 0)] = _state("minecraft:stone")
    hit = trace_static_ray((0.5, 0.5, 0.5), (1.0, 1.0, 0.0), grid, ((0, 0, 0), (2, 2, 0)))
    assert hit["status"] == "unknown"
    assert hit["reason"] == "tied_face_ownership"
    assert hit["cell"] == [1, 1, 0]


def test_near_tied_positive_crossings_visit_the_positive_length_side_cell():
    grid = _grid(2, 2)
    grid[(0, 1, 0)] = _state("minecraft:glass")
    grid[(1, 1, 0)] = _state("minecraft:stone")
    hit = trace_static_ray((0.5, 0.5, 0.5), (1.0, 1.0 + 1e-13, 0.0), grid, ((0, 0, 0), (2, 2, 0)))
    assert hit["status"] == "unknown"
    assert hit["reason"] == "unsupported_prefix"
    assert hit["cell"] == [0, 1, 0]
    assert hit["name"] == "minecraft:glass"


def test_negative_ray_on_grid_plane_starts_in_the_cell_entered_for_positive_depth():
    grid = _grid(1)
    grid[(1, 0, 0)] = _state("minecraft:glass")  # touched only at t=0; must not be inspected
    grid[(0, 0, 0)] = _state("minecraft:stone")
    hit = trace_static_ray((1.0, 0.5, 0.5), (-1.0, 0.0, 0.0), grid, ((0, 0, 0), (1, 0, 0)))
    assert hit["status"] == "hit"
    assert hit["cell"] == [0, 0, 0]
    assert hit["face"] == "east"
    assert hit["point"] == [1.0, 0.5, 0.5]


def test_context_exit_is_unknown_not_empty_space():
    grid = _grid(1)
    hit = trace_static_ray((0.5, 0.5, 0.5), (-1.0, 0.0, 0.0), grid, ((0, 0, 0), (1, 0, 0)))
    assert hit == {"status": "unknown", "reason": "context_exit"}


def test_missing_cell_is_a_grid_gap_not_air():
    grid = _grid(1)
    del grid[(0, 0, 0)]
    hit = trace_static_ray((0.5, 0.5, 0.5), (1.0, 0.0, 0.0), grid, ((0, 0, 0), (1, 0, 0)))
    assert hit["status"] == "unknown"
    assert hit["reason"] == "grid_gap"


@pytest.mark.parametrize("name", ["minecraft:iron_bars", "minecraft:white_stained_glass_pane", "minecraft:water", "minecraft:oak_slab"])
def test_unsupported_partial_or_transparent_prefix_refuses(name: str):
    grid = _grid(2)
    grid[(1, 0, 0)] = _state(name)
    grid[(2, 0, 0)] = _state("minecraft:stone")
    hit = trace_static_ray((0.5, 0.5, 0.5), (1.0, 0.0, 0.0), grid, ((0, 0, 0), (2, 0, 0)))
    assert hit["status"] == "unknown"
    assert hit["reason"] == "unsupported_prefix"
    assert hit["name"] == name


def test_empty_and_duplicate_denominators_are_null_and_pixel_pair_unique():
    points_l = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    points_r = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    def hit(cell, face, point):
        return {"status": "hit", "cell": cell, "face": face, "point": point}
    left = [hit([0, 0, 0], "west", [1.0, 0.2, 0.2]), hit([0, 0, 0], "west", [1.0, 0.2, 0.2]),
            hit([1, 0, 0], "west", [2.0, 0.2, 0.2])]
    right = [hit([0, 0, 0], "west", [1.0, 0.2, 0.2]), hit([0, 0, 0], "west", [1.0, 0.2, 0.2]),
             hit([1, 0, 0], "west", [2.0, 0.2, 0.2])]
    edges = [(0, 0), (1, 1), (2, 2), (0, 2), (1, 1)]
    # Use the same truth population construction contract as the production module.
    from mc_photogrammetry.evaluation_static_block_pair import _truth_pairs
    truth = _truth_pairs(points_l, points_r, left, right)
    summary, _ = _summarize_edges(list(range(len(edges))), edges, points_l, points_r, left, right, truth)
    assert summary["raw_edge_rows"] == 5
    assert summary["unique_pixel_pairs"] == 3
    assert summary["correct"] == 2
    assert summary["incorrect"] == 1
    assert summary["truth_pair_denominator"] == 2
    assert summary["conditional_static_precision"] == pytest.approx(2 / 3)
    assert summary["conditional_static_recall"] == 1.0

    unknown = [{"status": "unknown", "reason": "context_exit"}] * 3
    empty, _ = _summarize_edges([0], [(0, 0)], points_l, points_r, unknown, unknown, set())
    assert empty["coverage"] == 0.0
    assert empty["conditional_static_precision"] is None
    assert empty["conditional_static_recall"] is None


def test_context_visibility_and_target_scoring_domains_are_separate():
    from mc_photogrammetry.evaluation_static_block_pair import _truth_pairs
    points = [(5.0, 5.0)]
    target_hit = {"status": "hit", "target_hit": True, "cell": [5, 5, 5], "face": "west", "point": [6.0, 5.5, 5.5]}
    context_only = {"status": "hit", "target_hit": False, "cell": [5, 5, 5], "face": "west", "point": [6.0, 5.5, 5.5]}
    truth = _truth_pairs(points, points, [target_hit], [context_only])
    assert truth == set()
    summary, rows = _summarize_edges([0], [(0, 0)], points, points, [target_hit], [context_only], truth)
    assert rows[0][-1] == "outside_target"
    assert summary["outside_target"] == 1
    assert summary["scorable"] == 0
    assert summary["conditional_static_precision"] is None
    assert summary["truth_pair_denominator"] == 0
    assert summary["conditional_static_recall"] is None


def test_surface_tolerance_and_same_state_other_block_are_reported_separately():
    from mc_photogrammetry.evaluation_static_block_pair import _truth_pairs
    points = [(float(i), 0.0) for i in range(4)]
    def hit(cell, face, point, name):
        return {"status": "hit", "target_hit": True, "cell": cell, "face": face,
                "point": point, "name": name, "properties": {}}
    left = [
        hit([0, 0, 0], "west", [1.0, 0.2, 0.2], "minecraft:stone"),
        hit([1, 0, 0], "west", [2.0, 0.2, 0.2], "minecraft:stone"),
        hit([2, 0, 0], "west", [3.0, 0.2, 0.2], "minecraft:stone"),
        hit([4, 0, 0], "west", [5.0, 0.2, 0.2], "minecraft:dirt"),
    ]
    right = [
        hit([0, 0, 0], "west", [1.02, 0.2, 0.2], "minecraft:stone"),
        hit([1, 0, 0], "west", [2.2, 0.2, 0.2], "minecraft:stone"),
        hit([3, 0, 0], "west", [4.0, 0.2, 0.2], "minecraft:stone"),
        hit([4, 0, 0], "east", [4.0, 0.2, 0.2], "minecraft:stone"),
    ]
    truth = _truth_pairs(points, points, left, right)
    summary, _ = _summarize_edges(list(range(4)), [(i, i) for i in range(4)], points, points, left, right, truth)
    assert summary["correct"] == 1
    assert summary["same_face_out_of_tolerance"] == 1
    assert summary["same_state_same_slot_other_block"] == 1
    assert summary["wrong_surface"] == 1
    assert summary["scorable"] == 4
    assert summary["conditional_static_precision"] == 0.25
    assert summary["truth_pair_denominator"] == 1


def test_real_pair_static_block_diagnostic_replays_complete_pinned_populations():
    report = build_report()
    assert report["population"]["left_keypoints"] == 3562
    assert report["population"]["right_keypoints"] == 3737
    assert report["population"]["complete_reciprocal_candidate_edges"] == 7695
    assert report["population"]["candidate_graph_truncated"] is False
    assert report["population"]["native_modes"] == 12
    assert len(report["native_modes"]) == 12
    assert len(report["candidate_graph"]["edge_dispositions"]) == 7695
    assert sum(report["ray_status_counts"]["image-0000"].values()) == 3562
    assert sum(report["ray_status_counts"]["image-0001"].values()) == 3737
    for frame_rows in report["ray_outcomes"].values():
        assert all(row["context_hit"] is (row["status"] == "hit") for row in frame_rows)
        assert all((type(row["target_hit"]) is bool) if row["context_hit"] else row["target_hit"] is None
                   for row in frame_rows)
    assert all(mode["summary"]["raw_edge_rows"] == len(mode["edge_dispositions"])
               for mode in report["native_modes"])
    assert report["inputs"]["sha256"]["context"] == "180e49099e3e26ee957671dd94c3410a9fbe1124802d4f8976d6a4eac3bf2f8d"
    assert len(report["policy"]["opaque_full_cube_names_empty_properties_only"]) == len(OPAQUE_FULL_CUBE_NAMES)
    seal = report.pop("seal_sha256")
    assert seal == hashlib.sha256(_canonical_bytes(report)).hexdigest()
    assert len(_canonical_bytes(report)) <= MAX_CANONICAL_OUTPUT_BYTES
