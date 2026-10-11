"""Offline connected-support hypothesis; no production policy is enabled.

Truth is evaluator-only. A same-color corner can expose the same measured
graph fields as a planar patch, so rescuing the positive is insufficient.
"""
from dataclasses import fields, replace
import json

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import region_growth as rg
from mc_photogrammetry.reconstruction.calibration import CalibrationResult


def _patch(supports, *, disconnected=False, different_side=False, zero_side=False,
           color_barrier=False, edge_barrier=False):
    supports = np.asarray(supports, dtype=np.float32)
    count = len(supports)
    labels = np.repeat(np.arange(count, dtype=np.int32), 16)[None, :]
    labels = np.repeat(labels, 16, axis=0)
    if disconnected:
        labels = np.insert(labels, 16, -1, axis=1)
    lab = np.zeros((*labels.shape, 3), dtype=np.float32)
    if color_barrier:
        lab[labels == 1] = 100
    area, node_lab, texture, xy = rg._node_features(labels, lab)
    adjacency = rg._graph_adjacency(labels, lab, node_lab, texture)
    if edge_barrier:
        adjacency = tuple(tuple((row[0], 49., *row[2:]) for row in rows)
                          for rows in adjacency)
    height, width = labels.shape
    calibration = CalibrationResult(
        focal_pixels=300., yaw_degrees=30., pitch_degrees=-15.,
        width=width, height=height, inliers=120, segments=160,
        median_residual=.5)
    sides = rg._atom_sides(xy, calibration, width, height)
    assert np.all(sides[:, 0] == -1)
    if different_side:
        sides[1, 0] = 1
    if zero_side:
        sides[1, 0] = 0
    return rg._AtomGraph(
        labels=labels, lab=lab, usable=labels >= 0, node_area=area,
        node_lab=node_lab, node_texture=texture, node_xy=xy,
        supports=supports, plane_scores=rg._normal_scores(supports),
        sides=sides,
        axis_line_dirs=rg._axis_line_directions(xy, calibration, width, height),
        adjacency=adjacency)


def _connected_candidate(graph, axis=0, side=-1):
    """Test-only target+direct-neighbor balance pooling; no promotion authority.

    Target mass, normal suppression, reliability, opposing scores and side
    remain original. Donors use existing conservative continuation/boundary
    gates. There is no independent coplanarity cue in this graph, which the
    paired falsifier exposes. Never recursively pool neighbors' evidence.
    """
    config = rg.DEFAULT_REGION_GROWTH_CONFIG
    count = len(graph.node_area)
    assert count <= 128
    tangents = [other for other in range(3) if other != axis]
    total = graph.supports[:, tangents].sum(axis=1)
    dominance = total / np.maximum(total + graph.supports[:, axis], 1e-6)
    continuation = graph.supports[:, tangents].max(axis=1) * dominance
    other = np.max(np.delete(graph.plane_scores, axis, axis=1), axis=1)
    competing = ((other >= config.min_seed_score)
                 & (other >= graph.plane_scores[:, axis] + config.min_seed_margin))
    admissible = ((graph.sides[:, axis] == side) & ~competing
                  & (continuation >= config.min_continuation_support)
                  & (dominance >= config.min_tangent_dominance))
    result = np.zeros(count, dtype=bool)
    for start in np.flatnonzero(admissible):
        start = int(start)
        members = [start]
        for neighbor, gradient, color, texture, tx, ty in graph.adjacency[start]:
            if not admissible[neighbor]:
                continue
            boundary = np.asarray((tx, ty), dtype=np.float32)
            alignment = max(
                np.max(np.abs(graph.axis_line_dirs[start, tangents] @ boundary)),
                np.max(np.abs(graph.axis_line_dirs[neighbor, tangents] @ boundary)))
            if (gradient > config.bridge_edge_barrier
                    or color > config.max_bridge_lab_delta or texture > 18.
                    or alignment < config.min_boundary_axis_alignment):
                continue
            members.append(neighbor)
        if not 2 <= len(members) <= 8:
            continue
        members.sort()
        densities = graph.supports[members][:, tangents].astype(np.float64)
        densities /= graph.node_area[members, None]
        first, second = densities.mean(axis=0)
        balance = 2 * np.sqrt(first * second) / max(first + second, 1e-6)
        original = graph.supports[start]
        tangent = original[tangents].sum()
        suppression = tangent / max(tangent + original[axis], np.float32(1e-6))
        reliability = tangent / (tangent + np.float32(24))
        score = np.float32(balance) * suppression * reliability
        score = max(graph.plane_scores[start, axis], score)
        result[start] = (score >= config.min_seed_score
                         and np.float32(score - other[start]) >= config.min_seed_margin
                         and tangent >= config.min_seed_support)
    return result


@pytest.mark.parametrize("count", [2, 4])
def test_connected_aggregation_rescues_known_planar_alternating_tangents(count):
    graph = _patch([(0, 24, 0) if i % 2 == 0 else (0, 0, 24)
                    for i in range(count)])
    assert not np.any(graph.plane_scores[:, 0])
    assert rg._seed_and_grow(graph, 0, -1, rg.DEFAULT_REGION_GROWTH_CONFIG) == []
    assert _connected_candidate(graph).tolist() == [True] * count


@pytest.mark.parametrize("case", ["disconnected", "noise", "different_side", "zero_side",
                                 "normal_contamination", "contaminated_donor", "one_family",
                                 "color_barrier", "edge_barrier"])
def test_connected_aggregation_expected_negatives_reject(case):
    supports = [(0, 24, 0), (0, 0, 24)]
    if case == "noise":
        supports = [(0, .1, 0), (0, 0, .1)]
    if case == "normal_contamination":
        supports = [(96, 24, 0), (96, 0, 24)]
    if case == "contaminated_donor":
        supports = [(0, 24, 0), (96, 0, 24)]
    if case == "one_family":
        supports = [(0, 24, 0), (0, 24, 0)]
    graph = _patch(supports, **{case: True} if case in {
        "disconnected", "different_side", "zero_side", "color_barrier", "edge_barrier"} else {})
    assert rg._seed_and_grow(graph, 0, -1, rg.DEFAULT_REGION_GROWTH_CONFIG) == []
    assert not np.any(_connected_candidate(graph))


@pytest.mark.parametrize("permutation", [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
def test_connected_aggregation_paired_corner_falsifies_production_promotion(permutation):
    planar = _patch([(0, 24, 0), (0, 0, 24)])
    # Held-out physical labels are deliberately not an input: two distinct
    # perpendicular faces with only one tangent family each can expose the
    # same graph evidence. See frozen C10 paired falsifier in AGENT_COMMS.
    corner = _patch([(0, 24, 0), (0, 0, 24)])
    planar_truth, corner_truth = (0, 0), (2, 1)
    assert planar_truth != corner_truth
    for field in fields(planar):
        left, right = getattr(planar, field.name), getattr(corner, field.name)
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes()
        else:
            assert left == right
    def permute(graph):
        return replace(graph, supports=graph.supports[:, permutation].copy(),
                       plane_scores=graph.plane_scores[:, permutation].copy(),
                       sides=graph.sides[:, permutation].copy(),
                       axis_line_dirs=graph.axis_line_dirs[:, permutation].copy())
    axis = permutation.index(0)
    positive = _connected_candidate(permute(planar), axis)
    false_plane = _connected_candidate(permute(corner), axis)
    assert positive.tolist() == false_plane.tolist() == [True, True]
    print("CONNECTED_FALSIFIER=" + json.dumps({
        "rescued_planar_atoms": int(positive.sum()),
        "false_plane_atoms": int(false_plane.sum()),
        "same_visible_graph": True, "thresholds_unchanged": True,
        "production_promotion": False}, sort_keys=True))
