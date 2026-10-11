"""Synthetic controls for the image-only v2 plane-region graph."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from mc_photogrammetry.reconstruction import (
    RegionGrowthConfig,
    propose_image_planes_v2,
)
from mc_photogrammetry.reconstruction import region_growth as rg
from mc_photogrammetry.reconstruction.calibration import (
    CalibrationResult,
    axis_directions,
    vanishing_points,
)


WIDTH, HEIGHT = 320, 240
FOCAL = 300.0


def _calibration(*, yaw=0.0, pitch=0.0, roll=0.0, width=WIDTH, height=HEIGHT,
                 inliers=120):
    return CalibrationResult(
        focal_pixels=FOCAL,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        width=width,
        height=height,
        inliers=inliers,
        segments=160,
        median_residual=0.5,
        roll_degrees=roll,
    )


def _draw_projected_line(image, calibration, axis, point, span=160, value=64):
    points = vanishing_points(
        calibration.focal_pixels, calibration.yaw_degrees,
        calibration.pitch_degrees, calibration.width, calibration.height,
        calibration.roll_degrees, calibration.principal_point,
    )[axis]
    if abs(points[2]) > 1e-9:
        vp = points[:2] / points[2]
        direction = np.asarray(point, dtype=float) - vp
    else:
        direction = points[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    p0 = np.asarray(point, dtype=float) - direction * span / 2.0
    p1 = np.asarray(point, dtype=float) + direction * span / 2.0
    cv2.line(image, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)),
             (value, value, value), 2, cv2.LINE_8)
    return (*p0, *p1)


def _synthetic_room(*, occluder=False, low_contrast=False, repeated=False,
                    calibration=None):
    """Front wall normal=2 plus same-texture orthogonal wall normal=0."""
    cal = calibration or _calibration()
    image = np.full((HEIGHT, WIDTH, 3), 18, dtype=np.uint8)
    yy, xx = np.indices((HEIGHT, WIDTH))
    modulation = 3 if low_contrast else 12
    base = 110 + (((xx // 12 + yy // 12) % 2) * modulation)
    image[18:229, 0:190] = np.repeat(base[18:229, 0:190, None], 3, axis=2)
    image[18:229, 190:WIDTH] = np.repeat(base[18:229, 190:WIDTH, None], 3, axis=2)
    line_value = 92 if low_contrast else 65
    segments = []
    obstacle = np.zeros((HEIGHT, WIDTH), dtype=bool)
    if occluder:
        obstacle[88:169, 74:131] = True
        image[obstacle] = (25, 70, 175)
        cv2.rectangle(image, (74, 88), (130, 168), (230, 230, 230), 3)

    def add_seg(p0, p1, color=line_value):
        p0 = tuple(int(round(float(v))) for v in p0)
        p1 = tuple(int(round(float(v))) for v in p1)
        cv2.line(image, p0, p1, (color, color, color), 1, cv2.LINE_8)
        if not occluder:
            segments.append((*p0, *p1))
            return
        # Split line evidence at the foreground furniture; the wall never
        # receives evidence from the pixels hidden behind the obstacle.
        x0, y0 = p0
        x1, y1 = p1
        fractions = np.linspace(0.0, 1.0, max(2, int(np.hypot(x1-x0, y1-y0)) + 1))
        visible = []
        for fraction in fractions:
            x = int(round(x0 + fraction * (x1 - x0)))
            y = int(round(y0 + fraction * (y1 - y0)))
            inside = 0 <= x < WIDTH and 0 <= y < HEIGHT and obstacle[y, x]
            if not inside:
                visible.append((x, y))
            elif visible:
                if len(visible) > 2:
                    segments.append((visible[0][0], visible[0][1],
                                     visible[-1][0], visible[-1][1]))
                visible = []
        if len(visible) > 2:
            segments.append((visible[0][0], visible[0][1],
                             visible[-1][0], visible[-1][1]))

    if (cal.yaw_degrees == 0 and cal.pitch_degrees == 0
            and cal.roll_degrees == 0):
        for y in range(30, 220, 24):
            add_seg((4, y), (185, y))
        for x in range(12, 190, 24):
            add_seg((x, 20), (x, 226))
        # Orthogonal wall: one shared vertical axis and one radial axis.
        for x in range(198, WIDTH, 22):
            add_seg((x, 20), (x, 226))
        vp = vanishing_points(cal.focal_pixels, cal.yaw_degrees, cal.pitch_degrees,
                              WIDTH, HEIGHT, cal.roll_degrees)[2]
        vp_xy = vp[:2] / vp[2]
        for point in ((200, 28), (300, 30), (205, 210), (302, 205),
                      (225, 75), (285, 170), (240, 110), (310, 130)):
            direction = np.asarray(point, dtype=float) - vp_xy
            x_corner = 190.0
            fraction = (x_corner - vp_xy[0]) / direction[0]
            start = vp_xy + fraction * direction
            add_seg(start, point)
    else:
        # A rolled/projective front wall whose visible seams lie on the exact
        # image-derived Manhattan pencils for this calibration.
        for y in (48, 88, 128, 168, 208):
            for axis in (0, 1):
                point = (95, y) if axis == 0 else (42 + y // 5, y)
                segments.append(_draw_projected_line(image, cal, axis,
                                                     point, span=142))
    # The corner is an explicit high-gradient barrier despite identical wood.
    image[18:229, 188:193] = (238, 238, 238)
    usable = np.ones((HEIGHT, WIDTH), dtype=bool)
    return image, usable, np.asarray(segments, dtype=np.float64), obstacle


def _run(monkeypatch, image, usable, segments, calibration=None, **kwargs):
    monkeypatch.setattr(rg, "detect_segments",
                        lambda _grey: np.asarray(segments, dtype=np.float64))
    return propose_image_planes_v2(
        image, usable, calibration or _calibration(), **kwargs)


def _best_axis(result, axis):
    rows = [p for p in result.proposals if p.axis == axis]
    assert rows, f"no axis {axis} proposal in {[p.axis for p in result.proposals]}"
    return max(rows, key=lambda p: (p.strength, p.coverage))


def test_truncated_wall_grows_to_the_image_boundary_without_crossing_its_corner(monkeypatch):
    image, usable, segments, _ = _synthetic_room()
    result = _run(monkeypatch, image, usable, segments)
    assert result.status == "ok"
    front = _best_axis(result, 2)
    assert front.side == 1
    assert np.count_nonzero(front.mask[28:220, :175]) > 0.72 * (192 * 175)
    assert front.mask[100:150, 230:].mean() < 0.03
    assert front.mask[:, 0].any()  # The plane is genuinely frame-truncated.


def test_same_texture_orthogonal_walls_keep_distinct_axis_and_masks(monkeypatch):
    image, usable, segments, _ = _synthetic_room()
    result = _run(monkeypatch, image, usable, segments)
    front = _best_axis(result, 2)
    return_wall = _best_axis(result, 0)
    left = np.zeros((HEIGHT, WIDTH), dtype=bool)
    right = np.zeros((HEIGHT, WIDTH), dtype=bool)
    left[18:229, :188] = True
    right[18:229, 193:] = True
    assert front.mask[left].mean() > 0.70
    assert front.mask[right].mean() < 0.10
    assert return_wall.mask[right].mean() > 0.45
    assert return_wall.mask[left].mean() < 0.20
    overlap = np.count_nonzero(front.mask & return_wall.mask)
    assert overlap / max(min(int(front.mask.sum()), int(return_wall.mask.sum())), 1) < 0.10


def test_furniture_occlusion_preserves_a_hole_and_reconnects_visible_wall_islands(monkeypatch):
    image, usable, segments, obstacle = _synthetic_room(occluder=True)
    result = _run(monkeypatch, image, usable, segments)
    front = _best_axis(result, 2)
    assert front.mask[obstacle].mean() < 0.08
    assert front.mask[115, 55] and front.mask[115, 150]
    # The visible wall remains a single proposal with a literal furniture hole.
    assert len([p for p in result.proposals if p.axis == 2]) == 1


def test_low_contrast_wall_still_has_a_supported_region(monkeypatch):
    image, usable, segments, _ = _synthetic_room(low_contrast=True)
    result = _run(monkeypatch, image, usable, segments)
    assert result.status == "ok"
    assert _best_axis(result, 2).coverage > 0.15


def test_repeated_texture_does_not_union_orthogonal_plane_masks(monkeypatch):
    image, usable, segments, _ = _synthetic_room(repeated=True)
    result = _run(monkeypatch, image, usable, segments)
    assert result.status == "ok"
    assert len(result.proposals) <= 32
    assert {p.axis for p in result.proposals} >= {0, 2}
    for a in result.proposals:
        for b in result.proposals:
            if a is b:
                continue
            if a.axis != b.axis:
                assert not np.shares_memory(a.mask, b.mask)


def test_competing_manhattan_ownership_is_kept_as_continuous_axis_evidence(monkeypatch):
    image, usable, segments, _ = _synthetic_room()
    result = _run(monkeypatch, image, usable, segments)
    assert result.stats["segments"] == len(segments)
    assert result.stats["seed_atoms"] > 0
    assert _best_axis(result, 0).side in (-1, 1)
    assert _best_axis(result, 2).side in (-1, 1)


def test_nonzero_roll_is_used_for_line_ownership_and_side(monkeypatch):
    cal = _calibration(roll=7.0)
    image, usable, segments, _ = _synthetic_room(calibration=cal)
    result = _run(monkeypatch, image, usable, segments, calibration=cal)
    assert result.stats["segments_detected"] == len(segments)
    assert result.stats["segments_clipped"] > 0
    assert result.stats["segments_with_axis_assignment"] > 0
    assert result.stats["atoms_with_support"] > 0
    assert result.stats["seed_atoms"] > 0
    assert result.status == "ok"
    assert any(p.axis == 2 for p in result.proposals)
    directions = axis_directions(cal.yaw_degrees, cal.pitch_degrees, cal.roll_degrees)
    front = _best_axis(result, 2)
    ys, xs = np.nonzero(front.mask)
    ray = np.asarray([(xs.mean() - WIDTH / 2) / FOCAL,
                      (HEIGHT / 2 - ys.mean()) / FOCAL, 1.0])
    assert front.side == int(np.sign(np.dot(directions[2], ray)))


def test_segment_order_and_view_call_order_do_not_change_results(monkeypatch):
    image, usable, segments, _ = _synthetic_room()
    first = _run(monkeypatch, image, usable, segments)
    second = _run(monkeypatch, image, usable, segments[::-1].copy())
    assert [(p.axis, p.side, p.sources, p.polygon) for p in first.proposals] == [
        (p.axis, p.side, p.sources, p.polygon) for p in second.proposals]
    for a, b in zip(first.proposals, second.proposals):
        assert a.strength == pytest.approx(b.strength, abs=1e-8)
        np.testing.assert_array_equal(a.mask, b.mask)


def test_pixels_outside_usable_mask_cannot_change_masks(monkeypatch):
    image, usable, segments, _ = _synthetic_room()
    usable[:12] = False
    altered = image.copy()
    altered[~usable] = (250, 0, 250)
    first = _run(monkeypatch, image, usable, segments)
    second = _run(monkeypatch, altered, usable, segments)
    assert [(p.axis, p.side) for p in first.proposals] == [
        (p.axis, p.side) for p in second.proposals]
    for a, b in zip(first.proposals, second.proposals):
        np.testing.assert_array_equal(a.mask, b.mask)
    assert all(not p.mask[~usable].any() for p in first.proposals)


def test_typed_abstentions_and_candidate_cap(monkeypatch):
    image, usable, segments, _ = _synthetic_room()
    missing = propose_image_planes_v2(image, usable, None)
    assert (missing.status, missing.reason) == ("abstain", "missing_calibration")
    invalid = _calibration(inliers=2)
    low_support = propose_image_planes_v2(image, usable, invalid)
    assert low_support.reason == "insufficient_calibration_inliers"
    result = _run(monkeypatch, image, usable, segments, max_candidates=2)
    assert result.status == "ok"
    assert len(result.proposals) <= 2


def _historical_group_selection(proposals, width, height, cap):
    """Exact pre-repair selector, confined to the regression fixture."""
    proposals = sorted(proposals, key=lambda p: (
        -p.strength, p.axis, p.side, rg._region_key(p, width, height), p.sources))
    groups = {}
    for proposal in proposals:
        groups.setdefault(rg._region_key(proposal, width, height), []).append(proposal)
    selected = []
    for level in range(max((len(group) for group in groups.values()), default=0)):
        for key in sorted(groups):
            if level < len(groups[key]):
                selected.append(groups[key][level])
                if len(selected) == cap:
                    return tuple(selected)
    return tuple(selected)


def test_supported_later_axis_seed_and_cap_loss_are_observed_then_recovered(monkeypatch):
    image, usable, segments, _ = _synthetic_room()
    monkeypatch.setattr(rg, "detect_segments", lambda _: segments)
    real_selector = rg._select_diverse
    before_observer = rg._ScalarAttritionObserver()
    monkeypatch.setattr(rg, "_select_diverse", _historical_group_selection)
    with rg._scalar_attrition_context(before_observer, frame="room"):
        before = propose_image_planes_v2(image, usable, _calibration(), max_candidates=1)
    monkeypatch.setattr(rg, "_select_diverse", real_selector)
    after_observer = rg._ScalarAttritionObserver()
    with rg._scalar_attrition_context(after_observer, frame="room"):
        after = propose_image_planes_v2(image, usable, _calibration(), max_candidates=1)
    full = propose_image_planes_v2(image, usable, _calibration())
    assert before.status == after.status == full.status == "ok"
    assert [p.axis for p in before.proposals] == [0]
    assert [p.axis for p in after.proposals] == [2]
    assert after.proposals[0].strength > before.proposals[0].strength
    assert before.stats["proposals_before_cap"] == after.stats["proposals_before_cap"] == 2
    for observer, expected in ((before_observer, "diversity_cap"), (after_observer, "returned")):
        record = observer.record()
        assert record["status"] == "complete"
        assert any(row["axis"] == 2 and row["eligible_seed_count"] > 0
                   and row["disposition"] == "available"
                   for row in record["stages"]["seed"]["examples"])
        assert any(row["axis"] == 2 and row["disposition"] == expected
                   for row in record["stages"]["proposal"]["examples"])
        assert all(row["eligible_seed_count"] == 0 for row in
                   record["stages"]["seed"]["examples"] if row["axis"] == 1)
    # A genuinely absent normal remains absent even with unused capacity.
    assert {p.axis for p in full.proposals} == {0, 2}
    np.testing.assert_array_equal(after.proposals[0].mask, _best_axis(full, 2).mask)
    assert not after.proposals[0].mask.flags.writeable


def _seed_predicate_graph():
    scores = np.asarray([
        [0.10, 0.20, 0.05],  # side matches; normal score fails
        [0.49, 0.50, 0.10],  # score passes; margin fails
        [0.10, 0.50, 0.10],  # score and margin pass; support fails
        [0.10, 0.50, 0.10],  # all predicates pass
        [0.10, 0.60, 0.10],  # all except the requested side pass
    ], dtype=np.float32)
    supports = np.asarray([
        [10, 1, 10], [10, 1, 10], [2, 1, 3],
        [10, 1, 10], [15, 1, 15],
    ], dtype=np.float32)
    sides = np.ones((5, 3), dtype=np.int8)
    sides[4, 1] = -1
    return rg._AtomGraph(
        labels=np.arange(5, dtype=np.int32)[None, :],
        lab=np.zeros((1, 5, 3), dtype=np.float32),
        usable=np.ones((1, 5), dtype=bool),
        node_area=np.ones(5, dtype=np.int32),
        node_lab=np.zeros((5, 3), dtype=np.float32),
        node_texture=np.zeros(5, dtype=np.float32),
        node_xy=np.zeros((5, 2), dtype=np.float32),
        supports=supports, plane_scores=scores, sides=sides,
        axis_line_dirs=np.zeros((5, 3, 2), dtype=np.float32),
        adjacency=((),) * 5)


def test_seed_predicate_attrition_is_exact_bounded_and_output_neutral():
    from scripts.evaluation import diagnose_image_geometry_failure as diagnostic

    graph = _seed_predicate_graph()
    config = rg.DEFAULT_REGION_GROWTH_CONFIG
    without = [rg._seed_and_grow(graph, 1, side, config) for side in (1, -1, 0)]
    observer = rg._ScalarAttritionObserver()
    with rg._scalar_attrition_context(observer, frame="synthetic"):
        with_observer = [rg._seed_and_grow(graph, 1, side, config)
                         for side in (1, -1, 0)]
    assert without == with_observer == [[[3]], [[4]], []]
    record = observer.record()
    assert record["status"] == "complete"
    assert diagnostic._validate_attrition_record(record) is record
    stage = record["stages"]["seed"]
    assert (stage["expected"], stage["classified"], stage["unclassified"],
            stage["unexpected"], stage["multiply_classified"],
            stage["unexpected_events"], stage["omitted_examples"]) == (3, 3, 0, 0, 0, 0, 0)
    assert len(stage["examples"]) == 3
    by_side = {row["side"]: row for row in stage["examples"]}
    positive = by_side[1]
    assert positive["disposition"] == "available"
    assert {key: positive[key] for key in (
        "node_count", "side_match_count", "normal_score_pass_count",
        "margin_pass_count", "tangent_support_pass_count", "after_side_count",
        "after_score_count", "after_margin_count", "after_support_count",
        "eligible_seed_count")} == {
            "node_count": 5, "side_match_count": 4, "normal_score_pass_count": 4,
            "margin_pass_count": 4, "tangent_support_pass_count": 4,
            "after_side_count": 4, "after_score_count": 3,
            "after_margin_count": 2, "after_support_count": 1,
            "eligible_seed_count": 1}
    assert positive["max_side_normal_score"] == pytest.approx(0.5)
    assert positive["max_side_margin"] == pytest.approx(0.4)
    assert positive["max_side_tangent_support"] == pytest.approx(20)
    assert by_side[-1]["eligible_seed_count"] == 1
    assert by_side[-1]["after_support_count"] == 1
    assert by_side[-1]["max_side_normal_score"] == pytest.approx(0.6)
    assert by_side[0]["disposition"] == "absent"
    assert by_side[0]["side_match_count"] == 0
    assert by_side[0]["after_support_count"] == 0
    assert by_side[0]["max_side_normal_score"] is None
    assert by_side[0]["max_side_margin"] is None
    assert by_side[0]["max_side_tangent_support"] is None
    assert len(rg._ScalarAttritionObserver._encode(positive)) <= 2048


def test_seed_predicate_attrition_accepts_zero_nodes_without_nonfinite_extrema():
    from scripts.evaluation import diagnose_image_geometry_failure as diagnostic

    graph = _seed_predicate_graph()
    graph = replace(graph, labels=np.empty((0, 0), dtype=np.int32),
                    lab=np.empty((0, 0, 3), dtype=np.float32),
                    usable=np.empty((0, 0), dtype=bool),
                    node_area=graph.node_area[:0], node_lab=graph.node_lab[:0],
                    node_texture=graph.node_texture[:0], node_xy=graph.node_xy[:0],
                    supports=graph.supports[:0], plane_scores=graph.plane_scores[:0],
                    sides=graph.sides[:0], axis_line_dirs=graph.axis_line_dirs[:0],
                    adjacency=())
    assert rg._seed_and_grow(graph, 1, 1, rg.DEFAULT_REGION_GROWTH_CONFIG) == []
    observer = rg._ScalarAttritionObserver()
    with rg._scalar_attrition_context(observer, frame="empty"):
        assert rg._seed_and_grow(graph, 1, 1, rg.DEFAULT_REGION_GROWTH_CONFIG) == []
    record = observer.record()
    assert record["status"] == "complete"
    diagnostic._validate_attrition_record(record)
    row = record["stages"]["seed"]["examples"][0]
    assert row["disposition"] == "absent"
    assert all(row[key] == 0 for key in ("node_count", "side_match_count",
        "normal_score_pass_count", "margin_pass_count", "tangent_support_pass_count",
        "after_side_count", "after_score_count", "after_margin_count",
        "after_support_count", "eligible_seed_count"))
    assert all(row[key] is None for key in ("max_side_normal_score",
        "max_side_margin", "max_side_tangent_support"))


def _post_dedup_proposal(axis, side, row, column, strength):
    """Selection-only fixture; no claim of measured pixel support."""
    mask = np.zeros((90, 90), dtype=bool)
    mask[row * 30 + 8:row * 30 + 22, column * 30 + 8:column * 30 + 22] = True
    mask.setflags(write=False)
    return rg.PlaneProposal(axis=axis, side=side, polygon=((column * 30 + 8, row * 30 + 8),
        (column * 30 + 21, row * 30 + 8), (column * 30 + 21, row * 30 + 21),
        (column * 30 + 8, row * 30 + 21)), mask=mask, strength=strength,
        sources=("selection-only",), coverage=float(mask.mean()))


def test_fixed_cap32_prioritises_support_without_axis_quotas_or_copying_masks():
    proposals = [_post_dedup_proposal(axis, side, row, column, .2 + len_index / 1000)
                 for len_index, (axis, side, row, column) in enumerate(
                     (axis, side, row, column) for axis in (0, 1)
                     for side in (-1, 1) for row in range(3) for column in range(3))]
    supported = _post_dedup_proposal(2, 1, 1, 1, .9)
    proposals.append(supported)
    assert len(proposals) == 37 and rg.MAX_REGION_GROWTH_PROPOSALS == 32
    old = _historical_group_selection(proposals, 90, 90, 32)
    assert all(p.axis != 2 for p in old)
    selected = rg._select_diverse(list(proposals), 90, 90, 32)
    assert len(selected) == 32 and selected[0] is supported
    assert all(any(p is original and p.mask is original.mask for original in proposals)
               for p in selected)
    # Axis0 has only18 groups and could not be starved by the old fixed-cap pass.
    assert sum(p.axis == 0 for p in old) == 18
    assert rg._select_diverse([], 90, 90, 32) == ()
    assert rg._select_diverse([supported], 90, 90, 32) == (supported,)


def test_group_strength_selection_is_axis_permutation_equivariant_and_spatially_diverse():
    strong = [_post_dedup_proposal(axis, 1, axis, 0, .9 - axis / 10) for axis in range(3)]
    redundant = [_post_dedup_proposal(0, 1, 0, 0, .6),
                 _post_dedup_proposal(0, -1, 1, 2, .5)]
    proposals = strong + redundant
    selected = rg._select_diverse(list(proposals), 90, 90, 3)
    assert all(a is b for a, b in zip(selected, strong))
    assert all(a is b for a, b in zip(selected,
        rg._select_diverse(list(reversed(proposals)), 90, 90, 3)))
    for permutation in ((2, 0, 1), (1, 2, 0), (2, 1, 0)):
        relabelled = [replace(p, axis=permutation[p.axis]) for p in proposals]
        mapped = rg._select_diverse(relabelled, 90, 90, 3)
        assert [p.axis for p in mapped] == [permutation[p.axis] for p in selected]
        assert all(a.mask is b.mask for a, b in zip(mapped, selected))
    # Smaller caps cannot cover all directions and do not fabricate coverage.
    assert [p.axis for p in rg._select_diverse(list(proposals), 90, 90, 2)] == [0, 1]
    weak_third = replace(strong[2], strength=.1)
    assert 2 not in {p.axis for p in rg._select_diverse(strong[:2] + [weak_third] + redundant, 90, 90, 3)}
    # All spatial/side groups get a turn before a second member of any group.
    four = rg._select_diverse(list(proposals), 90, 90, 4)
    assert len({rg._region_key(p, 90, 90) for p in four}) == 4


def test_group_strength_ties_are_deterministic_by_spatial_cell_side_then_axis():
    proposals = [_post_dedup_proposal(2, 1, 1, 0, .8),
                 _post_dedup_proposal(0, 1, 0, 1, .8),
                 _post_dedup_proposal(1, -1, 0, 1, .8)]
    selected = rg._select_diverse(list(proposals), 90, 90, 2)
    assert [p.axis for p in selected] == [1, 0]
    assert all(a is b for a, b in zip(selected,
        rg._select_diverse(list(reversed(proposals)), 90, 90, 2)))


def test_working_resolution_atom_memory_and_runtime_are_bounded(monkeypatch):
    width, height = 960, 640
    image = np.full((height, width, 3), 110, dtype=np.uint8)
    for y in range(12, height, 20):
        cv2.line(image, (0, y), (width - 1, y), (70, 70, 70), 1)
    segments = np.asarray([(0, y, width - 1, y)
                           for y in range(12, height, 20)], dtype=float)
    cal = _calibration(width=width, height=height)
    started = time.perf_counter()
    result = _run(monkeypatch, image, np.ones((height, width), bool), segments,
                  calibration=cal)
    elapsed = time.perf_counter() - started
    assert result.stats["analysis_pixels"] <= 960 * 640
    assert result.stats["atoms"] <= RegionGrowthConfig().max_atoms
    assert len(result.proposals) <= 32
    assert elapsed < 8.0


def _atomize_rgb(image, usable, *, spacing=16, iterations=3, compactness=10.0,
                 minimum_fraction=0.25):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    return rg._local_lab_xy_atoms(
        lab, usable, spacing=spacing, iterations=iterations,
        compactness=compactness, min_component_area_fraction=minimum_fraction,
        return_stats=True)


def _connectivity_component_count(labels):
    split = rg._split_disconnected_atoms(labels)
    return int(split.max()) + 1 if split.size and split.max() >= 0 else 0


def test_checker_stripes_merge_small_fragments_preserve_gui_holes_and_connectivity():
    height, width = 128, 160
    yy, xx = np.indices((height, width))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    checker = ((xx // 3 + yy // 3) % 2).astype(bool)
    image[checker] = (235, 24, 35)
    image[~checker] = (25, 60, 230)
    usable = np.ones((height, width), dtype=bool)
    usable[:10, :] = False  # GUI-shaped top band.
    usable[46:83, 63:101] = False  # Interior mask hole.

    labels, stats = _atomize_rgb(image, usable)

    assert np.all(labels[~usable] == -1)
    assert np.all(labels[usable] >= 0)
    final_count = int(labels.max()) + 1
    assert stats["components_after_cleanup"] == final_count
    assert stats["components_before_cleanup"] > final_count
    assert stats["connectivity_cleanup_merges"] == (
        stats["components_before_cleanup"] - final_count)
    assert stats["connectivity_mask_preserved"] is True
    assert _connectivity_component_count(labels) == final_count


def test_cleanup_uses_greatest_shared_boundary_and_stable_id_ties():
    labels = np.asarray([
        [0, 0, 2, 1, 1],
        [0, 0, 2, 1, 1],
        [0, 0, 2, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=np.int32)
    cleaned, merges, isolated = rg._merge_small_components(labels, minimum_area=5)
    # Component 2 shares more edges with 0 than 1, so it joins 0.
    assert cleaned[:, :3].max() == 0
    assert set(np.unique(cleaned)) == {0, 1}
    assert merges == 1
    assert isolated == 0

    tied = np.asarray([
        [0, 0, 2, 1, 1],
        [0, 0, 2, 1, 1],
        [0, 0, 2, 1, 1],
    ], dtype=np.int32)
    tied_cleaned, tied_merges, _ = rg._merge_small_components(tied, minimum_area=4)
    assert np.all(tied_cleaned[:, 2] == 0)  # equal edge counts choose stable ID 0
    assert tied_merges == 1


def test_diagonal_and_tiny_isolated_usable_islands_are_retained_without_bridging():
    image = np.full((48, 48, 3), (130, 75, 40), dtype=np.uint8)
    usable = np.zeros((48, 48), dtype=bool)
    diagonal = [(2 + 4 * index, 2 + 4 * index) for index in range(9)]
    for y, x in diagonal:
        usable[y, x] = True
    usable[3:5, 30:32] = True  # A separate 4-connected 2x2 island.

    labels, stats = _atomize_rgb(image, usable, spacing=8)

    assert np.all(labels[usable] >= 0)
    assert np.all(labels[~usable] == -1)
    assert stats["components_before_cleanup"] == len(diagonal) + 1
    assert stats["components_after_cleanup"] == len(diagonal) + 1
    assert stats["connectivity_cleanup_merges"] == 0
    assert stats["isolated_small_components_retained"] == len(diagonal) + 1
    assert _connectivity_component_count(labels) == len(diagonal) + 1


def test_exact_seed_cap_boundary_passes_atomization_and_next_seed_fails_preflight(monkeypatch):
    width = height = 64
    image = np.full((height, width, 3), 100, dtype=np.uint8)
    usable = np.ones((height, width), dtype=bool)
    calibration = _calibration(width=width, height=height)
    monkeypatch.setattr(rg, "detect_segments", lambda _grey: np.empty((0, 4)))
    at_cap = propose_image_planes_v2(
        image, usable, calibration,
        config=RegionGrowthConfig(atom_spacing=16, max_atoms=16))
    assert at_cap.reason == "no_structural_segments"
    assert at_cap.stats["seed_count"] == 16
    assert at_cap.stats["atoms"] <= 16
    assert at_cap.stats["connectivity_mask_preserved"] is True

    over_cap = propose_image_planes_v2(
        image, usable, calibration,
        config=RegionGrowthConfig(atom_spacing=16, max_atoms=15))
    assert over_cap.reason == "atom_cap"
    assert over_cap.stats["seed_count"] == 16


def test_unmergeable_isolated_islands_use_specific_connectivity_cap_abstention(monkeypatch):
    width = height = 128
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    usable = np.zeros((height, width), dtype=bool)
    for row in range(8):
        for column in range(10):
            usable[1 + 3 * row, 1 + 3 * column] = True
    calibration = _calibration(width=width, height=height)
    result = propose_image_planes_v2(
        image, usable, calibration,
        config=RegionGrowthConfig(atom_spacing=16, max_atoms=64,
                                  min_usable_fraction=0.0))
    assert result.reason == "atom_connectivity_cap"
    assert result.stats["seed_count"] == 64
    assert result.stats["components_before_cleanup"] == 80
    assert result.stats["components_after_cleanup"] == 80
    assert result.stats["connectivity_mask_preserved"] is True
    assert result.stats["atoms"] > 64


def test_atom_labels_are_hash_stable_and_rgb_channel_order_is_explicit():
    height, width = 80, 96
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, ::2] = (245, 12, 40)
    image[:, 1::2] = (14, 70, 238)
    usable = np.ones((height, width), dtype=bool)
    lab_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_bgr_bytes = cv2.cvtColor(image[:, :, ::-1], cv2.COLOR_RGB2LAB)
    assert not np.array_equal(lab_rgb, lab_bgr_bytes)

    first, first_stats = _atomize_rgb(image, usable)
    second, second_stats = _atomize_rgb(image, usable)
    permuted_first, _ = _atomize_rgb(image[:, :, ::-1].copy(), usable)
    permuted_second, _ = _atomize_rgb(image[:, :, ::-1].copy(), usable)

    digest = lambda array: hashlib.sha256(array.astype("<i4").tobytes()).hexdigest()
    assert digest(first) == digest(second)
    assert digest(permuted_first) == digest(permuted_second)
    assert first_stats == second_stats


def test_global_overlap_assignment_uses_stable_lower_seed_for_exact_ties():
    image = np.full((16, 16, 3), (80, 80, 80), dtype=np.uint8)
    usable = np.ones((16, 16), dtype=bool)
    first, stats = _atomize_rgb(image, usable, spacing=8, iterations=1,
                                compactness=0.0, minimum_fraction=0.0)
    second, _ = _atomize_rgb(image, usable, spacing=8, iterations=1,
                             compactness=0.0, minimum_fraction=0.0)
    np.testing.assert_array_equal(first, second)
    assert first[4, 8] == 0  # seed 0 wins the equal-distance overlap with seed 1
    assert stats["components_after_cleanup"] == _connectivity_component_count(first)


def test_module_source_has_no_annotation_or_external_data_access():
    source_path = Path(rg.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not any(name.startswith(("skimage", "cv2.ximgproc")) for name in imported)
    assert not ({"open", "read_text", "read_bytes", "imread"} & calls)
    lowered = source.lower()
    assert "boundary-targets-v4" not in lowered
    assert "manifest.json" not in lowered
    assert "reference_camera" not in lowered


def _scale_threshold_working_frame(width, height):
    image = np.full((height, width, 3), 110, dtype=np.uint8)
    usable = np.ones((height, width), dtype=bool)
    config = RegionGrowthConfig(analysis_max_side=96)
    _rgb, grey, _lab, mask, scale_x, scale_y = rg._working_image(image, usable, config)
    calibration = replace(_calibration(width=width, height=height),
        focal_pixels=FOCAL / scale_x, width=grey.shape[1], height=grey.shape[0])
    labels = np.zeros(grey.shape, dtype=np.int32)
    return grey, mask, labels, calibration, config, (scale_x, scale_y)


def _install_scale_threshold_lsd(monkeypatch, analysis_segments, work_shape):
    """Use the real detect_segments length filter with an authored LSD result."""
    rows = np.asarray(analysis_segments, dtype=np.float64).reshape((-1, 1, 4))
    calls = []
    class Detector:
        def detect(self, image):
            calls.append(image.shape)
            # The second, half-resolution detector pass has no additional lines.
            return (rows.copy() if image.shape == work_shape else None, None, None, None)
    monkeypatch.setattr(cv2, "createLineSegmentDetector", lambda: Detector())
    return calls


@pytest.mark.parametrize("size", [(96, 64), (128, 72), (129, 73), (192, 128)],
    ids=["unresized", "downscaled", "rounded-anisotropic", "half-size"])
def test_original_40_pixel_admission_survives_analysis_resize_and_reaches_support(monkeypatch, size):
    grey, usable, labels, calibration, config, scales = _scale_threshold_working_frame(*size)
    scale_x, scale_y = scales
    # 52 original pixels also clear the unchanged 24-analysis-pixel clip run
    # at 2x reduction; the 39.5-original-pixel negative remains unchanged.
    original = np.asarray([(8, 12, 60, 12), (8, 22, 47.5, 22)], dtype=float)
    analysis = original / np.asarray((scale_x, scale_y, scale_x, scale_y))
    if scales != (1.0, 1.0):
        assert np.hypot(*(analysis[0, 2:] - analysis[0, :2])) < 40
    calls = _install_scale_threshold_lsd(monkeypatch, analysis, grey.shape)
    support, evidence = rg._segment_support(grey, usable, labels, calibration, config,
        original_pixel_scale=scales)
    assert evidence["segments_detected"] == evidence["segments_clipped"] == 1
    assert evidence["segments_with_axis_assignment"] == 1
    assert evidence["support_mass"] > 0 and np.any(support > 0)
    assert len(calls) == 2 and calls[0] == grey.shape


@pytest.mark.parametrize("size", [(96, 64), (192, 128)], ids=["unresized", "exact-half-size"])
@pytest.mark.parametrize("length,accepted", [(39.999, False), (40.0, True), (40.001, True)],
    ids=["below-40", "exact-40", "above-40"])
def test_original_space_40_boundary_is_strict_without_epsilon(monkeypatch, size, length, accepted):
    grey, usable, labels, calibration, config, scales = _scale_threshold_working_frame(*size)
    # Zero endpoint makes the exact 40 case representable at both tested scales.
    analysis = np.asarray([(0, 12, length / scales[0], 12)], dtype=float)
    calls = _install_scale_threshold_lsd(monkeypatch, analysis, grey.shape)
    support, evidence = rg._segment_support(grey, usable, labels, calibration, config,
        original_pixel_scale=scales)
    assert rg._clip_segments_to_usable_mask.__kwdefaults__["minimum_run"] == 24.0
    assert evidence["segments_detected"] == int(accepted)
    # Detector admission is in original pixels; clipping remains in analysis
    # pixels. The admitted 40/40.001 rows are only 20/20.0005 at 2x reduction.
    reaches_support = accepted and scales == (1.0, 1.0)
    assert evidence["segments_clipped"] == int(reaches_support)
    assert evidence["segments_with_axis_assignment"] == int(reaches_support)
    assert bool(np.any(support > 0)) is reaches_support
    assert (evidence["support_mass"] > 0) is reaches_support
    assert len(calls) == 2


@pytest.mark.parametrize("direction", ["horizontal", "vertical", "diagonal"])
def test_actual_rounded_xy_transform_controls_original_length_not_nominal_scale(monkeypatch, direction):
    grey, usable, labels, calibration, config, scales = _scale_threshold_working_frame(129, 73)
    scale_x, scale_y = scales
    assert grey.shape == (54, 96) and scale_x == 129 / 96 and scale_y == 73 / 54
    assert scale_x != scale_y
    if direction == "horizontal":
        # A max-axis scalar would incorrectly admit this 39.775-original-pixel line.
        analysis = np.asarray([(2, 12, 31.6, 12)], dtype=float)
        assert 29.6 * scale_x < 40 <= 29.6 * scale_y
        accepted = False
    elif direction == "vertical":
        # A nominal scalar would incorrectly discard this 40.0148... pixel line.
        analysis = np.asarray([(50, 3, 50, 32.6)], dtype=float)
        assert 29.6 * scale_x < 40 <= 29.6 * scale_y
        accepted = True
    else:
        original = np.asarray([(8, 10, 33.2, 43.6), (8, 10, 31.7, 41.6)], dtype=float)
        assert np.hypot(25.2, 33.6) > 40 > np.hypot(23.7, 31.6)
        analysis = original / np.asarray((scale_x, scale_y, scale_x, scale_y))
        accepted = True  # Exactly the first, 42-original-pixel diagonal is admitted.
    _install_scale_threshold_lsd(monkeypatch, analysis, grey.shape)
    _support, evidence = rg._segment_support(grey, usable, labels, calibration, config,
        original_pixel_scale=scales)
    assert evidence["segments_detected"] == int(accepted)


@pytest.mark.parametrize("mask_kind", ["all-unusable", "too-short-usable-run"])
def test_recovered_original_length_segment_still_obeys_existing_usable_mask_clipping(monkeypatch, mask_kind):
    grey, usable, labels, calibration, config, scales = _scale_threshold_working_frame(128, 72)
    analysis = np.asarray([(0, 12, 44 / scales[0], 12)], dtype=float)
    _install_scale_threshold_lsd(monkeypatch, analysis, grey.shape)
    usable[:] = False
    if mask_kind == "too-short-usable-run": usable[:, :3] = True
    support, evidence = rg._segment_support(grey, usable, labels, calibration, config,
        original_pixel_scale=scales)
    assert evidence["segments_detected"] == 1 and evidence["segments_clipped"] == 0
    assert evidence["support_mass"] == 0 and not np.any(support)


@pytest.mark.parametrize("length,expected", [(44.0, 1), (39.5, 0)], ids=["recover-native-line", "reject-short-native-line"])
def test_public_proposal_path_threads_actual_scales_without_loosening_seed_or_growth(monkeypatch, length, expected):
    width, height = 128, 72
    image = np.full((height, width, 3), 110, dtype=np.uint8)
    usable = np.ones((height, width), dtype=bool)
    config = RegionGrowthConfig(analysis_max_side=96)
    prepared = rg._working_image(image, usable, config)
    grey, scales = prepared[1], prepared[4:]
    original = np.asarray([(0, 16, length, 16)], dtype=float)
    analysis = original / np.asarray((scales[0], scales[1], scales[0], scales[1]))
    calls = _install_scale_threshold_lsd(monkeypatch, analysis, grey.shape)
    original_build = rg._build_graph
    forwarded = []
    def build(*args, **kwargs):
        forwarded.append(kwargs["original_pixel_scale"])
        assert args[-1] is config
        return original_build(*args, **kwargs)
    monkeypatch.setattr(rg, "_build_graph", build)
    result = propose_image_planes_v2(image, usable, _calibration(width=width, height=height), config=config)
    assert forwarded == [scales] and len(calls) == 2
    assert result.stats["segments_detected"] == result.stats["segments_clipped"] == expected
    if expected:
        assert result.stats["segments_with_axis_assignment"] == 1 and result.stats["support_mass"] > 0
        assert result.reason != "no_structural_segments"
    else:
        assert result.reason == "no_structural_segments" and result.proposals == ()
    assert result.stats["atoms"] <= config.max_atoms and len(result.proposals) <= config.max_candidates
    assert config.min_seed_support == RegionGrowthConfig().min_seed_support


def test_unfiltered_detector_volume_is_original_filtered_before_clipping_and_existing_segment_cap(monkeypatch):
    grey, usable, labels, calibration, config, scales = _scale_threshold_working_frame(128, 72)
    config = replace(config, max_segments=1)
    short = np.tile(np.asarray([(0, 12, 2, 12)], dtype=float), (4096, 1))
    valid = np.asarray([(0, 12, 44 / scales[0], 12), (0, 22, 45 / scales[0], 22)], dtype=float)
    calls = _install_scale_threshold_lsd(monkeypatch, np.vstack((short, valid)), grey.shape)
    clip = rg._clip_segments_to_usable_mask
    clipped_inputs = []
    def observe_clip(segments, mask):
        clipped_inputs.append(segments.copy())
        return clip(segments, mask)
    monkeypatch.setattr(rg, "_clip_segments_to_usable_mask", observe_clip)
    support, evidence = rg._segment_support(grey, usable, labels, calibration, config,
        original_pixel_scale=scales)
    assert len(calls) == 2 and len(clipped_inputs) == 1
    assert clipped_inputs[0].shape == (2, 4)
    assert np.array_equal(clipped_inputs[0], valid)
    assert evidence["segments_detected"] == evidence["segments_clipped"] == 2
    assert evidence["segments_with_axis_assignment"] == config.max_segments == 1
    assert evidence["support_mass"] > 0 and np.any(support)


@pytest.mark.parametrize("permutation", [
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
], ids=["012", "021", "102", "120", "201", "210"])
def test_normal_scores_authored_support_formula_and_all_axis_permutations(permutation):
    import math

    canonical = np.asarray([[0, 48, 48], [0, 96, 0], [300, 48, 48]], dtype=np.float64)
    expected = np.asarray([[4 / 5, 0, 0], [0, 0, 0],
                           [32 / 165, 580 / 1023, 580 / 1023]], dtype=np.float64)
    supports = canonical[:, permutation].copy()
    original = supports.copy()
    actual = rg._normal_scores(supports)

    # Scalar oracle uses each authored row, independent of the producer's
    # vectorized axis indexing and broadcasting. Rational values also bind
    # the intended balanced/missing/contaminated support interpretations.
    oracle = []
    for row in supports.tolist():
        axis_scores = []
        for axis in range(3):
            first, second = [row[column] for column in range(3) if column != axis]
            tangent = first + second
            balance = 2 * math.sqrt(first * second) / max(tangent, 1e-6)
            absence = tangent / max(sum(row), 1e-6)
            reliability = tangent / (tangent + 24)
            axis_scores.append(min(1.0, max(0.0, balance * absence * reliability)))
        oracle.append(axis_scores)
    assert actual.shape == (3, 3) and actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(supports, original)
    np.testing.assert_allclose(oracle, expected[:, permutation], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, oracle, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(actual, expected[:, permutation], rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("case", ["narrow-miss", "phase-aligned", "different-label-boundary"])
def test_segment_support_sparse_label_sampling_against_dense_raster_oracle(monkeypatch, case):
    # Authored atoms/segments isolate allocation; this is not image inference
    # or evidence that the same sampling loss occurred in historical inputs.
    labels = np.zeros((97, 97), dtype=np.int32)
    left = 23 if case == "phase-aligned" else 19
    labels[16:81, left:left + 3] = 1
    target_pixels = {(x, y) for y in range(97) for x in range(97) if labels[y, x] == 1}
    assert len(target_pixels) == 195 >= 64
    reached = {min(target_pixels)}
    frontier = list(reached)
    while frontier:
        x, y = frontier.pop()
        for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbour in target_pixels and neighbour not in reached:
                reached.add(neighbour)
                frontier.append(neighbour)
    assert reached == target_pixels  # One 4-connected atom, not a tiny fragment.

    horizontal_y = 15 if case == "different-label-boundary" else 32
    vertical_x = left + 1
    segments = np.asarray([(8, horizontal_y, 88, horizontal_y),
                           (vertical_x, 8, vertical_x, 88)], dtype=np.float64)
    np.testing.assert_array_equal(np.hypot(segments[:, 2] - segments[:, 0],
                                          segments[:, 3] - segments[:, 1]), [80, 80])
    assert 80 >= rg.MINIMUM_LENGTH
    grey = np.full(labels.shape, 110, dtype=np.uint8)
    usable = np.ones(labels.shape, dtype=bool)
    config = rg.DEFAULT_REGION_GROWTH_CONFIG
    assert config.atom_spacing == 16 and config.max_segment_samples == 128
    detector_calls, clipped_calls, residual_calls, weight_calls = [], [], [], []
    clip = rg._clip_segments_to_usable_mask
    residual = rg.segment_residuals

    def detect(received, **kwargs):
        assert received.shape == grey.shape and kwargs == {}
        detector_calls.append(received.shape)
        return segments.copy()

    def observe_clip(received, mask):
        assert mask is usable
        result = clip(received, mask)
        np.testing.assert_array_equal(result, segments)
        clipped_calls.append(result.copy())
        return result

    def observe_residual(received, points):
        result = residual(received, points)
        # Real calibrated horizontal/vertical axes, not authored associations.
        assert result.shape == (2, 3)
        assert result[0, 0] == result[1, 1] == 0
        assert np.all(result[0, [1, 2]] > 3 * rg.INLIER_PIXELS)
        assert np.all(result[1, [0, 2]] > 3 * rg.INLIER_PIXELS)
        residual_calls.append(result.copy())
        return result

    def unit_weights(received, lines):
        assert received is grey
        np.testing.assert_array_equal(lines, segments)
        weight_calls.append(lines.copy())
        return np.ones(len(lines), dtype=np.float64)

    monkeypatch.setattr(rg, "detect_segments", detect)
    monkeypatch.setattr(rg, "_clip_segments_to_usable_mask", observe_clip)
    monkeypatch.setattr(rg, "segment_residuals", observe_residual)
    monkeypatch.setattr(rg, "structural_weights", unit_weights)
    actual, diagnostics = rg._segment_support(grey, usable, labels,
        _calibration(width=97, height=97), config)
    assert len(detector_calls) == len(clipped_calls) == len(residual_calls) == len(weight_calls) == 1
    assert diagnostics["segments_detected"] == diagnostics["segments_clipped"] == diagnostics["segments_with_axis_assignment"] == 2

    # Independent dense axis-aligned raster oracle uses every integer pixel;
    # it does not call linspace/rint or copy the production sparse sampler.
    dense = np.zeros((2, 3), dtype=np.float64)
    raster_counts = np.zeros((2, 2), dtype=np.int32)
    for axis, pixels in enumerate(([(x, horizontal_y) for x in range(8, 89)],
                                   [(vertical_x, y) for y in range(8, 89)])):
        assert len(pixels) == 81
        for x, y in pixels:
            raster_counts[int(labels[y, x]), axis] += 1
        for label in (0, 1):
            dense[label, axis] = 80 * int(raster_counts[label, axis]) / 81
    dense_horizontal = 0 if case == "different-label-boundary" else 3
    np.testing.assert_array_equal(raster_counts[1], [dense_horizontal, 65])
    np.testing.assert_allclose(dense[1], [80 * dense_horizontal / 81, 80 * 65 / 81, 0])

    # For these 80px segments the unchanged default sampler has exactly
    # eleven coordinates 8,16,...,88. A 3px run between 16 and24 is missed.
    sparse_horizontal_hits = int(case == "phase-aligned")
    expected_target = np.asarray([80 * sparse_horizontal_hits / 11, 80 * 9 / 11, 0])
    expected = np.asarray([[80, 80, 0] - expected_target, expected_target])
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-6)
    assert actual.sum() == pytest.approx(160, abs=2e-5)
    assert diagnostics["support_mass"] == pytest.approx(160, abs=2e-5)
    assert dense.sum() == pytest.approx(160)
    np.testing.assert_array_equal(actual[:, 2], [0, 0])
    source_both = bool(actual[1, 0] > 0 and actual[1, 1] > 0)
    oracle_both = bool(dense[1, 0] > 0 and dense[1, 1] > 0)
    assert source_both is (case == "phase-aligned")
    assert oracle_both is (case != "different-label-boundary")
    if case == "different-label-boundary":
        assert labels[15, vertical_x] == 0 and labels[16, vertical_x] == 1
        assert actual[1, 0] == dense[1, 0] == 0  # No adjacent-label transfer.


# Private raster prototype controls: authored labels/lines, no image inference.
def _raster_controls(monkeypatch, segments, *, residuals=None, weights=None):
    segments = np.asarray(segments, dtype=np.float64).reshape((-1, 4))
    calls = dict(enhance=0, detect=0, clip=0, residual=0, weight=0)

    def enhance(grey):
        calls["enhance"] += 1
        return grey

    def detect(grey, **kwargs):
        calls["detect"] += 1
        return segments.copy()

    def clip(lines, mask):
        calls["clip"] += 1
        return lines.copy()

    def residual(lines, points):
        calls["residual"] += 1
        return (np.tile([0, 100, 100], (len(lines), 1)) if residuals is None
                else np.asarray(residuals, dtype=float).copy())

    def weight(grey, lines):
        calls["weight"] += 1
        return (np.ones(len(lines)) if weights is None else np.asarray(weights).copy())

    for name, function in (("enhance", enhance), ("detect_segments", detect),
                           ("_clip_segments_to_usable_mask", clip),
                           ("segment_residuals", residual), ("structural_weights", weight)):
        monkeypatch.setattr(rg, name, function)
    return calls


def _centerline_oracle(segment):
    # Fractions' independently implemented round() is an exact nearest-even
    # oracle; no production vector arithmetic or pixel helper is used.
    from fractions import Fraction
    x0, y0, x1, y1 = [round(float(value)) for value in segment]
    if (x0, y0) > (x1, y1):
        x0, y0, x1, y1 = x1, y1, x0, y0
    dx, dy = x1 - x0, y1 - y0
    n = max(abs(dx), abs(dy))
    if not n:
        return [(x0, y0)]
    return [(round(Fraction(x0 * n + dx * k, n)),
             round(Fraction(y0 * n + dy * k, n))) for k in range(n + 1)]


@pytest.mark.parametrize("segment,expected", [
    ((0, 0, 4, 0), [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]),
    ((0, 4, 0, 0), [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]),
    ((0, 0, 4, 1), [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1)]),
    ((0, 0, 1, 4), [(0, 0), (0, 1), (0, 2), (1, 3), (1, 4)]),
    ((0, 4, 1, 0), [(0, 4), (0, 3), (0, 2), (1, 1), (1, 0)]),
    ((0, 0, 3, 3), [(0, 0), (1, 1), (2, 2), (3, 3)]),
    ((0, 3, 3, 0), [(0, 3), (1, 2), (2, 1), (3, 0)]),
    ((0, 0, 2, 1), [(0, 0), (1, 0), (2, 1)]),
    ((0, 1, 2, 2), [(0, 1), (1, 2), (2, 2)]),
    ((0.5, 0.5, 2.5, 1.5), [(0, 0), (1, 1), (2, 2)]),
    ((2, 2, 2, 2), [(2, 2)]),
])
def test_raster_hand_enumerated_centerlines(segment, expected):
    workspace = rg._RasterSupportWorkspace()
    xs, ys = rg._thin_centerline(segment, 5, 5, workspace)
    actual = list(zip(xs.tolist(), ys.tolist()))
    assert actual == expected == _centerline_oracle(segment)
    assert len(actual) == len(set(actual))
    reverse = (*segment[2:], *segment[:2])
    xs, ys = rg._thin_centerline(reverse, 5, 5, workspace)
    assert list(zip(xs.tolist(), ys.tolist())) == expected


def test_raster_grid_properties_axis_swap_reversal_and_non_tie_reflection():
    from fractions import Fraction
    workspace = rg._RasterSupportWorkspace()
    for x0 in range(5):
        for y0 in range(5):
            for x1 in range(5):
                for y1 in range(5):
                    segment = (x0, y0, x1, y1)
                    expected = _centerline_oracle(segment)
                    xs, ys = rg._thin_centerline(segment, 5, 5, workspace)
                    assert list(zip(xs.tolist(), ys.tolist())) == expected
                    assert len(expected) == len(set(expected))
                    xs, ys = rg._thin_centerline((x1, y1, x0, y0), 5, 5, workspace)
                    assert list(zip(xs.tolist(), ys.tolist())) == expected
                    xs, ys = rg._thin_centerline((y0, x0, y1, x1), 5, 5, workspace)
                    assert set(zip(xs.tolist(), ys.tolist())) == {(y, x) for x, y in expected}
                    n = max(abs(x1 - x0), abs(y1 - y0))
                    ties = n and any(Fraction(delta * k, n).denominator == 2
                                     for delta in (x1 - x0, y1 - y0)
                                     for k in range(n + 1))
                    if not ties:
                        xs, ys = rg._thin_centerline((4-x0, 4-y0, 4-x1, 4-y1), 5, 5, workspace)
                        assert set(zip(xs.tolist(), ys.tolist())) == {(4-x, 4-y) for x, y in expected}
    # Reflection at half-pixel ties is deliberately not universally equivariant.
    assert _centerline_oracle((0, 0, 2, 1)) == [(0, 0), (1, 0), (2, 1)]
    assert _centerline_oracle((2, 1, 0, 0)) == [(0, 0), (1, 0), (2, 1)]
    assert {(2-x, 1-y) for x, y in _centerline_oracle((0, 0, 2, 1))} != set(_centerline_oracle((0, 0, 2, 1)))


@pytest.mark.parametrize("case", ["narrow-miss", "phase-aligned", "different-label-boundary"])
def test_raster_falsifier_counterparts_preserve_same_label_boundary(monkeypatch, case):
    labels = np.zeros((97, 97), dtype=np.int32)
    left = 23 if case == "phase-aligned" else 19
    labels[16:81, left:left+3] = 1
    y = 15 if case == "different-label-boundary" else 32
    lines = [(8, y, 88, y), (left+1, 8, left+1, 88)]
    calls = _raster_controls(monkeypatch, lines, residuals=[[0, 100, 100], [100, 0, 100]])
    grey = np.zeros(labels.shape, dtype=np.uint8)
    support, diagnostics = rg._segment_support(grey, np.ones(labels.shape, bool), labels,
        _calibration(width=97, height=97), rg.DEFAULT_REGION_GROWTH_CONFIG, _raster_support=True)
    h = 0 if case == "different-label-boundary" else 3
    target = np.asarray([80*h/81, 80*65/81, 0])
    np.testing.assert_allclose(support, [[80, 80, 0]-target, target], rtol=2e-7, atol=1e-6)
    assert diagnostics["support_mass"] == pytest.approx(160, abs=2e-5)
    assert calls == dict(enhance=1, detect=1, clip=1, residual=1, weight=1)
    assert bool(support[1, 0] > 0) is (case != "different-label-boundary")


@pytest.mark.parametrize("all_unusable", [False, True])
def test_raster_exact_atom_counts_invalid_mask_and_unsupported_mass(all_unusable):
    labels = np.asarray([[0, 1, -1, 2, 0, 1, 8, 1, 0]], dtype=np.int32)
    usable = np.ones(labels.shape, bool)
    usable[0, 5] = False
    if all_unusable:
        usable[:] = False
    support = np.zeros((3, 3), np.float32)
    association = np.asarray([.25, .75, 0])
    rg._raster_allocate_segment(support, labels, usable, (0, 0, 8, 0), 8, .5,
                                 association, rg._RasterSupportWorkspace())
    counts = [sum(int(usable[y, x] and labels[y, x] == node)
                  for x, y in [(i, 0) for i in range(9)]) for node in range(3)]
    expected = np.asarray([4*count/9*association for count in counts])
    np.testing.assert_allclose(support, expected, rtol=2e-7, atol=1e-7)
    unsupported = 4*(9-sum(counts))/9
    assert float(support.sum()) + unsupported == pytest.approx(4, abs=2e-7)
    assert counts == ([0, 0, 0] if all_unusable else [3, 2, 1])


@pytest.mark.parametrize("weight", [.5, 0])
def test_raster_mixed_association_weights_and_zero_valid_line(monkeypatch, weight):
    softness = max(rg.INLIER_PIXELS*1.35, 1)
    # exp(-r^2/2s^2) = 1/3 versus1 -> normalized [.25,.75,0].
    residuals = [[softness*np.sqrt(2*np.log(3)), 0, 100], [100, 100, 100]]
    calls = _raster_controls(monkeypatch, [(0, 0, 80, 0), (0, 1, 80, 1)],
                             residuals=residuals, weights=[weight, 1])
    labels = np.zeros((2, 81), np.int32)
    support, diagnostics = rg._segment_support(np.zeros(labels.shape, np.uint8),
        np.ones(labels.shape, bool), labels, _calibration(), rg.DEFAULT_REGION_GROWTH_CONFIG,
        _raster_support=True)
    np.testing.assert_allclose(support, [[80*weight*.25, 80*weight*.75, 0]], rtol=2e-7)
    assert diagnostics["segments_with_axis_assignment"] == 1
    assert calls == dict(enhance=1, detect=1, clip=1, residual=1, weight=1)


@pytest.mark.parametrize("flag", [False, None, 1, np.bool_(True)])
@pytest.mark.parametrize("case", ["ordinary", "large-side", "large-census", "float-labels", "many-nodes"])
def test_raster_literal_true_only_legacy_isolation(monkeypatch, flag, case):
    width = 961 if case == "large-side" else (960 if case == "large-census" else 81)
    labels = np.zeros((1, width), np.int32)
    if case == "float-labels":
        labels = labels.astype(float)
    if case == "many-nodes":
        labels[0, -1] = 20000
    config = replace(rg.DEFAULT_REGION_GROWTH_CONFIG, max_segments=6000, max_segment_samples=2)
    lines = [(0, 0, width-1, 0)] * (5001 if case == "large-census" else 1)
    calls = _raster_controls(monkeypatch, lines)
    def trap(*args, **kwargs):
        pytest.fail("nonliteral True entered private raster code")
    for name in ("_raster_validate_frame", "_raster_validate_segments", "_thin_centerline",
                 "_raster_allocate_segment", "_RasterSupportWorkspace"):
        monkeypatch.setattr(rg, name, trap)
    grey, usable = np.zeros(labels.shape, np.uint8), np.ones(labels.shape, bool)
    expected = rg._segment_support(grey, usable, labels, _calibration(), config)
    actual = rg._segment_support(grey, usable, labels, _calibration(), config, _raster_support=flag)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    assert calls == dict(enhance=2, detect=2, clip=2, residual=2, weight=2)


@pytest.mark.parametrize("flag", [False, None, 1])
@pytest.mark.parametrize("exception", [RuntimeError("original detector failure"), KeyboardInterrupt()])
@pytest.mark.parametrize("seam", ["detect_segments", "_clip_segments_to_usable_mask", "segment_residuals", "structural_weights"])
def test_raster_legacy_preserves_exact_exceptions_and_cancellation(monkeypatch, flag, exception, seam):
    calls = []
    _raster_controls(monkeypatch, [(0, 0, 80, 0)])
    def raise_original(*args, **kwargs):
        calls.append(1)
        raise exception
    monkeypatch.setattr(rg, seam, raise_original)
    monkeypatch.setattr(rg, "_raster_validate_frame", lambda *a: pytest.fail("raster entered"))
    for kwargs in ({}, {"_raster_support": flag}):
        with pytest.raises(type(exception)) as caught:
            rg._segment_support(np.zeros((1, 1000), np.uint8), np.ones((2, 1), bool),
                                np.zeros((1, 1), float), _calibration(),
                                replace(rg.DEFAULT_REGION_GROWTH_CONFIG, max_segments=10000), **kwargs)
        assert caught.value is exception
    assert len(calls) == 2


@pytest.mark.parametrize("case", ["segment-cap", "negative-cap", "float-cap", "large-side", "empty",
    "labels-float", "labels-object", "labels-bool", "label-shape", "mask-shape", "mask-dtype", "nodes"])
def test_raster_frame_refusal_precedes_detector(monkeypatch, case):
    grey = np.zeros((2, 81), np.uint8)
    labels = np.zeros(grey.shape, np.int32)
    usable = np.ones(grey.shape, bool)
    config = rg.DEFAULT_REGION_GROWTH_CONFIG
    if case == "segment-cap": config = replace(config, max_segments=5001)
    if case == "negative-cap": config = replace(config, max_segments=-1)
    if case == "float-cap": config = replace(config, max_segments=5000.0)
    if case == "large-side":
        grey, labels, usable = np.zeros((1, 961), np.uint8), np.zeros((1, 961), np.int32), np.ones((1, 961), bool)
    if case == "empty":
        grey, labels, usable = np.zeros((0, 81), np.uint8), np.zeros((0, 81), np.int32), np.ones((0, 81), bool)
    if case.startswith("labels-"): labels = labels.astype({"labels-float": float, "labels-object": object, "labels-bool": bool}[case])
    if case == "label-shape": labels = labels[:, :1]
    if case == "mask-shape": usable = usable[:, :1]
    if case == "mask-dtype": usable = usable.astype(np.uint8)
    if case == "nodes": labels[0, 0] = 20000
    monkeypatch.setattr(rg, "detect_segments", lambda *a, **k: pytest.fail("detected before refusal"))
    with pytest.raises(ValueError, match="raster_support_(invalid_input|budget_exceeded)"):
        rg._segment_support(grey, usable, labels, _calibration(), config, _raster_support=True)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("resized", [False, True])
def test_raster_nonfinite_detected_endpoint_refused_before_clip(monkeypatch, value, resized):
    _raster_controls(monkeypatch, [(value, 0, 80, 0)])
    monkeypatch.setattr(rg, "_clip_segments_to_usable_mask", lambda *a: pytest.fail("clipped invalid endpoint"))
    labels = np.zeros((1, 81), np.int32)
    with pytest.raises(ValueError, match=rg._RASTER_INVALID):
        rg._segment_support(np.zeros(labels.shape, np.uint8), np.ones(labels.shape, bool), labels,
            _calibration(), rg.DEFAULT_REGION_GROWTH_CONFIG, _raster_support=True,
            original_pixel_scale=(2, 2) if resized else (1, 1))


@pytest.mark.parametrize("case", ["negative-weight", "nan-weight", "inf-weight", "nan-residual", "inf-residual", "weight-shape", "residual-shape"])
def test_raster_bad_evidence_refuses_without_support_allocation(monkeypatch, case):
    weights, residuals = [1.], [[0., 100., 100.]]
    if case == "negative-weight": weights = [-.1]
    if case == "nan-weight": weights = [np.nan]
    if case == "inf-weight": weights = [np.inf]
    if case == "nan-residual": residuals[0][0] = np.nan
    if case == "inf-residual": residuals[0][0] = np.inf
    if case == "weight-shape": weights = [1., 2.]
    if case == "residual-shape": residuals = [[0., 100.]]
    _raster_controls(monkeypatch, [(0, 0, 80, 0)], weights=weights, residuals=residuals)
    monkeypatch.setattr(rg, "_RasterSupportWorkspace", lambda: pytest.fail("allocated before refusal"))
    labels = np.zeros((1, 81), np.int32)
    with pytest.raises(ValueError, match=rg._RASTER_INVALID):
        rg._segment_support(np.zeros(labels.shape, np.uint8), np.ones(labels.shape, bool), labels,
                            _calibration(), rg.DEFAULT_REGION_GROWTH_CONFIG, _raster_support=True)


def test_raster_visit_and_endpoint_limits_exact_and_plus_one():
    workspace = rg._RasterSupportWorkspace()
    xs, ys = rg._thin_centerline((0, 0, 959, 0), 960, 1, workspace)
    assert len(xs) == 960 and xs[0] == 0 and xs[-1] == 959 and not ys.any()
    with pytest.raises(ValueError, match=rg._RASTER_BUDGET):
        rg._thin_centerline((0, 0, 960, 0), 961, 1, workspace)
    lines = np.tile([0., 0., 959., 0.], (5000, 1))
    assert rg._raster_validate_segments(lines, 960, 1) == 4_800_000
    plus_one = np.vstack((lines, [0., 0., 0., 0.]))
    with pytest.raises(ValueError, match=rg._RASTER_BUDGET):
        rg._raster_validate_segments(plus_one, 960, 1)
    for segment in ((-1, 0, 8, 0), (0, 0, 960, 0), (0, -1, 9, 0)):
        with pytest.raises(ValueError, match=rg._RASTER_INVALID):
            rg._thin_centerline(segment, 960, 1, workspace)


@pytest.mark.parametrize("supports", [[0, 96, 0], [0, .01, 100], [300, 48, 48], [0, 1, 1]])
def test_raster_does_not_bypass_real_normal_score_seed_gate_negatives(supports):
    from types import SimpleNamespace
    supports = np.asarray([supports], dtype=np.float32)
    scores = rg._normal_scores(supports)
    graph = SimpleNamespace(node_area=np.asarray([195]), supports=supports,
                            plane_scores=scores, sides=np.ones((1, 3), np.int8))
    assert scores[0, 0] < rg.DEFAULT_REGION_GROWTH_CONFIG.min_seed_score
    assert rg._seed_and_grow(graph, 0, 1, rg.DEFAULT_REGION_GROWTH_CONFIG) == []


@pytest.mark.parametrize("case", ["side", "margin", "support"])
def test_raster_seed_joint_predicates_remain_required(case):
    from types import SimpleNamespace
    supports = np.asarray([[0, 48, 48]], np.float32)
    scores = rg._normal_scores(supports)
    sides = np.ones((1, 3), np.int8)
    config = rg.DEFAULT_REGION_GROWTH_CONFIG
    if case == "side": sides[0, 0] = -1
    if case == "margin": scores[0, 1] = scores[0, 0]
    if case == "support": config = replace(config, min_seed_support=97)
    graph = SimpleNamespace(node_area=np.asarray([195]), supports=supports,
                            plane_scores=scores, sides=sides)
    assert rg._seed_and_grow(graph, 0, 1, config) == []
    assert rg.DEFAULT_REGION_GROWTH_CONFIG == RegionGrowthConfig()


def test_raster_max_bound_benchmark_and_scratch_accounting(monkeypatch):
    import tracemalloc
    workspace = rg._RasterSupportWorkspace()
    assert workspace.numeric_bytes == 56_640
    # Conservative maximum gather, vector row arithmetic, headers and state.
    accounted_scratch = workspace.numeric_bytes + 960*9 + 96 + 8192
    assert accounted_scratch == 73_568 < rg._RASTER_SCRATCH_BYTES
    lines = np.tile([0., 0., 959., 0.], (5000, 1))
    # Worst visited-label diversity (960 unique nodes/line); returned output
    # remains <=20000*3*4. No max_atoms-sized histogram is constructed.
    labels = np.arange(960, dtype=np.int32).reshape(1, 960)
    grey, usable = np.zeros(labels.shape, np.uint8), np.ones(labels.shape, bool)
    calls = _raster_controls(monkeypatch, lines)
    started = time.perf_counter()
    support, diagnostics = rg._segment_support(grey, usable, labels, _calibration(),
                            rg.DEFAULT_REGION_GROWTH_CONFIG, _raster_support=True)
    elapsed = time.perf_counter()-started
    expected = np.full(960, np.float32(5000*959/960))
    np.testing.assert_allclose(support[:, 0], expected, rtol=8e-5)
    np.testing.assert_array_equal(support[:, 1:], np.zeros((960, 2)))
    assert calls == dict(enhance=1, detect=1, clip=1, residual=1, weight=1)
    assert diagnostics["segments_with_axis_assignment"] == 5000
    # Track the helper only, excluding retained frame inputs/output above.
    tracemalloc.start()
    traced_support = np.zeros((960, 3), np.float32)
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    rg._raster_allocate_segment(traced_support, labels, usable, lines[0], 959, 1,
                                 np.asarray([1., 0., 0.]), workspace)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    incremental_peak = peak-baseline
    assert workspace.numeric_bytes + incremental_peak + 8192 < rg._RASTER_SCRATCH_BYTES
    print(f"raster max-bound: visits=4800000 elapsed_seconds={elapsed:.6f} "
          f"workspace_bytes={workspace.numeric_bytes} accounted_scratch_bytes={accounted_scratch} "
          f"traced_incremental_peak_bytes={incremental_peak} output_bytes={support.nbytes}")


@pytest.mark.parametrize("case", ["no-detection", "no-clipped-segments", "all-negative-labels"])
def test_raster_empty_outputs_float32_and_legacy_dtype_preserved(monkeypatch, case):
    lines = [] if case == "no-detection" else [(0, 0, 80, 0)]
    _raster_controls(monkeypatch, lines)
    if case == "no-clipped-segments":
        monkeypatch.setattr(rg, "_clip_segments_to_usable_mask", lambda *a: np.empty((0, 4)))
    labels = np.full((1, 81), -1 if case == "all-negative-labels" else 0, np.int32)
    args = (np.zeros(labels.shape, np.uint8), np.ones(labels.shape, bool), labels,
            _calibration(), rg.DEFAULT_REGION_GROWTH_CONFIG)
    raster, diagnostics = rg._segment_support(*args, _raster_support=True)
    legacy, old_diagnostics = rg._segment_support(*args)
    assert raster.dtype == np.float32
    assert legacy.dtype == (np.float32 if case == "all-negative-labels" else np.float64)
    assert diagnostics == old_diagnostics
    np.testing.assert_array_equal(raster, legacy)


@pytest.mark.parametrize("association", [[np.nan, 0, 0], [np.inf, 0, 0], [-.1, 1.1, 0]])
def test_raster_invalid_accepted_associations_refused(association):
    with pytest.raises(ValueError, match=rg._RASTER_INVALID):
        rg._raster_validate_evidence(np.asarray([80.]), np.asarray([1.]),
                                     np.asarray([association]), 1)


def test_raster_opt_in_propagates_original_cancellation(monkeypatch):
    exception = KeyboardInterrupt()
    _raster_controls(monkeypatch, [(0, 0, 80, 0)])
    def cancel(*args, **kwargs):
        raise exception
    monkeypatch.setattr(rg, "_raster_allocate_segment", cancel)
    labels = np.zeros((1, 81), np.int32)
    with pytest.raises(KeyboardInterrupt) as caught:
        rg._segment_support(np.zeros(labels.shape, np.uint8), np.ones(labels.shape, bool),
            labels, _calibration(), rg.DEFAULT_REGION_GROWTH_CONFIG, _raster_support=True)
    assert caught.value is exception


@pytest.mark.parametrize("segments", [np.asarray([0, 0, 80, 0]), np.zeros((1, 5)),
                                     np.asarray([["bad", "0", "80", "0"]])])
def test_raster_malformed_detected_shapes_and_types_refused(segments):
    with pytest.raises(ValueError, match=rg._RASTER_INVALID):
        rg._raster_validate_detected(segments)


@pytest.mark.parametrize("mode", ["ordinary", "zero-association", "empty", "zero-cap"])
def test_paired_source_seam_census_actual_samples_and_return(monkeypatch, mode):
    lines=[] if mode=="empty" else [(0,0,80,0),(0,1,80,1)]
    residuals=None if mode!="zero-association" else [[0,100,100],[100,100,100]]
    calls=_raster_controls(monkeypatch,lines,residuals=residuals)
    labels=np.zeros((2,81),np.int32);labels[:,20]=1
    grey=np.zeros(labels.shape,np.uint8);usable=np.ones(labels.shape,bool)
    config=replace(rg.DEFAULT_REGION_GROWTH_CONFIG,max_segments=0) if mode=="zero-cap" else rg.DEFAULT_REGION_GROWTH_CONFIG
    events=[]
    class Sink:
        def begin(self,*args):
            events.append(("begin",args))
        def segment(self,*args): events.append(("segment",args))
        def finalize(self,support): events.append(("finalize",support))
        def discard(self,reason): pytest.fail(reason)
    baseline=rg._segment_support(grey,usable,labels,_calibration(),config)
    prior=calls.copy()
    actual=rg._segment_support(grey,usable,labels,_calibration(),config,_support_observer=Sink())
    np.testing.assert_array_equal(actual[0],baseline[0]);assert actual[1]==baseline[1]
    assert actual[0].dtype==baseline[0].dtype
    assert all(calls[k]==2*v for k,v in prior.items())
    assert events[0][0]=="begin" and events[-1][0]=="finalize" and events[-1][1] is actual[0]
    begin=events[0][1];assert begin[0] is grey and begin[1] is usable and begin[2] is labels and begin[4] is config
    expected=0 if mode in ("empty","zero-cap") else 2
    assert len(begin[5])==expected and len(events)==expected+2
    for ordinal,(_,args) in enumerate(events[1:-1]):
        assert args[0]==ordinal and all(args[i+1] is begin[i+5] for i in range(4))
        if mode=="zero-association" and ordinal==1: assert args[-2:]==(None,None)
        else: assert args[-1]==11 and len(args[-2])==11 and args[-2][0]==0


@pytest.mark.parametrize("method", ["begin","segment","finalize"])
def test_paired_source_observer_exception_is_telemetry_only(monkeypatch,method):
    _raster_controls(monkeypatch,[(0,0,80,0)])
    labels=np.zeros((1,81),np.int32);args=(np.zeros(labels.shape,np.uint8),np.ones(labels.shape,bool),labels,_calibration(),rg.DEFAULT_REGION_GROWTH_CONFIG)
    original=rg._segment_support(*args)
    discards=[]
    class Sink:
        def begin(self,*args): pass
        def segment(self,*args): pass
        def finalize(self,*args): pass
        def discard(self,reason): discards.append(reason)
    sink=Sink()
    def fail(*args): raise RuntimeError("private callback")
    setattr(sink,method,fail)
    actual=rg._segment_support(*args,_support_observer=sink)
    np.testing.assert_array_equal(original[0],actual[0]);assert original[1]==actual[1]
    assert discards==["observer_error"]


@pytest.mark.parametrize("flag", [None,False])
def test_paired_source_omitted_none_never_observes_or_allocates(monkeypatch,flag):
    _raster_controls(monkeypatch,[(0,0,80,0)])
    monkeypatch.setattr(rg,"_paired_support_observe",lambda *a,**k:pytest.fail("default observation"))
    labels=np.zeros((1,81),np.int32)
    kwargs={} if flag is False else {"_support_observer":None}
    support,_=rg._segment_support(np.zeros(labels.shape,np.uint8),np.ones(labels.shape,bool),labels,_calibration(),rg.DEFAULT_REGION_GROWTH_CONFIG,**kwargs)
    assert support[0,0]==80


def test_paired_source_raster_authority_never_records_sparse_census(monkeypatch):
    _raster_controls(monkeypatch,[(0,0,80,0)])
    labels=np.zeros((1,81),np.int32);args=(np.zeros(labels.shape,np.uint8),np.ones(labels.shape,bool),labels,_calibration(),rg.DEFAULT_REGION_GROWTH_CONFIG)
    baseline=rg._segment_support(*args,_raster_support=True)
    reasons=[]
    class Sink:
        def discard(self,reason): reasons.append(reason)
        def begin(self,*args): pytest.fail("raster authority observed as sparse")
        def segment(self,*args): pytest.fail("raster ordinal observed as sparse")
        def finalize(self,*args): pytest.fail("raster return observed as sparse")
    actual=rg._segment_support(*args,_raster_support=True,_support_observer=Sink())
    assert reasons==["raster_authority"] and actual[1]==baseline[1]
    np.testing.assert_array_equal(actual[0],baseline[0])


def _line_weight_policy_target_graph(monkeypatch, *, policy=None, weights=None, missing_tangent=None):
    # Authored post-clipping fragments reconstruct the sealed image-0000
    # same-atom scalar witness; no screenshot or source detector claim.
    lengths=np.asarray([1.885591983795166,3.130382537841797,15.87571907043457])
    weighted=np.asarray([1.0933902263641357,1.8152016401290894,3.9689297676086426])
    lines=[(20,20+10*i,20,20+10*i+length) for i,length in enumerate(lengths)]
    residuals=np.full((3,3),100.)
    np.fill_diagonal(residuals,0.)
    if missing_tangent is not None: residuals[missing_tangent,:]=100.
    calls=_raster_controls(monkeypatch,lines,residuals=residuals,
                           weights=weighted/lengths if weights is None else weights)
    # Detector evidence can be long; the controlled clip stage emits the
    # retained short fragments independently of production clipping behaviour.
    def detect(grey,**kwargs):
        calls['detect']+=1
        return np.asarray([(20,8,20,88)]*3,dtype=float)
    def clip(raw,mask):
        calls['clip']+=1
        return np.asarray(lines,dtype=float)
    monkeypatch.setattr(rg,'detect_segments',detect)
    monkeypatch.setattr(rg,'_clip_segments_to_usable_mask',clip)
    labels=np.zeros((97,97),np.int32);labels[16:81,19:22]=1
    grey=np.zeros(labels.shape,np.uint8);usable=np.ones(labels.shape,bool)
    lab=np.zeros((*labels.shape,3),np.float32)
    config=rg.DEFAULT_REGION_GROWTH_CONFIG
    options={} if policy is None else dict(_support_line_weight_policy=policy)
    graph,stats=rg._build_graph(labels,grey,lab,usable,
                              _calibration(yaw=180,width=97,height=97),config,**options)
    return graph,stats,config,calls


def test_support_line_weight_policy_actual_allocator_seed_growth_target_regression(monkeypatch):
    outcomes=[]
    for policy,expected_support,expected_score,expected_tangent,expected_regions in [
        ('structural',[1.0933902263641357,1.8152016401290894,3.9689297676086426],
         .15158295631408691,5.7841315269470215,[]),
        ('unit',[1.885591983795166,3.130382537841797,15.87571907043457],
         .2982529401779175,19.006101608276367,[[1]])]:
        graph,stats,config,calls=_line_weight_policy_target_graph(monkeypatch,policy=policy)
        np.testing.assert_array_equal(graph.supports[1],np.asarray(expected_support,np.float32))
        assert graph.plane_scores[1,0]==pytest.approx(expected_score,abs=1e-8)
        assert graph.supports[1,1:].sum()==pytest.approx(expected_tangent,abs=1e-7)
        assert graph.sides[1,0]==-1
        assert (config.min_seed_score,config.min_seed_margin,config.min_seed_support)==(.26,.035,12.)
        observer=rg._ScalarAttritionObserver()
        with rg._scalar_attrition_context(observer,frame='authored-target'):
            regions=rg._seed_and_grow(graph,0,-1,config)
        assert regions==expected_regions
        terminal=observer.record()['stages']['seed']['examples'][0]
        assert terminal['eligible_seed_count']==len(expected_regions)
        assert terminal['after_score_count']==len(expected_regions)
        assert terminal['after_support_count']==len(expected_regions)
        assert calls==dict(enhance=1,detect=1,clip=1,residual=1,weight=int(policy=='structural'))
        outcomes.append((stats,terminal))
    assert outcomes[0][0]['segments_with_axis_assignment']==outcomes[1][0]['segments_with_axis_assignment']==3


@pytest.mark.parametrize('missing_tangent',[1,2])
def test_support_line_weight_policy_missing_required_tangent_still_rejects(monkeypatch,missing_tangent):
    graph,stats,config,_=_line_weight_policy_target_graph(monkeypatch,policy='unit',missing_tangent=missing_tangent)
    assert graph.supports[1,missing_tangent]==0.
    assert graph.supports[1,3-missing_tangent]>0.
    assert graph.sides[1,0]==-1 and stats['segments_with_axis_assignment']==2
    assert graph.plane_scores[1,0]==0.
    assert rg._seed_and_grow(graph,0,-1,config)==[]


def test_support_line_weight_policy_all_unit_equivalence_and_default_isolation(monkeypatch):
    default=_line_weight_policy_target_graph(monkeypatch)
    structural=_line_weight_policy_target_graph(monkeypatch,policy='structural')
    assert default[2] is rg.DEFAULT_REGION_GROWTH_CONFIG
    assert rg.DEFAULT_REGION_GROWTH_CONFIG==RegionGrowthConfig()
    assert not hasattr(rg.DEFAULT_REGION_GROWTH_CONFIG,'_support_line_weight_policy')
    np.testing.assert_array_equal(default[0].supports,structural[0].supports)
    np.testing.assert_array_equal(default[0].plane_scores,structural[0].plane_scores)
    assert default[1]==structural[1] and default[3]==structural[3]
    old_unit=_line_weight_policy_target_graph(monkeypatch,policy='structural',weights=[1.,1.,1.])
    opt_unit=_line_weight_policy_target_graph(monkeypatch,policy='unit',weights=[.25,.5,.75])
    for field in ('supports','plane_scores','sides','axis_line_dirs'):
        np.testing.assert_array_equal(getattr(old_unit[0],field),getattr(opt_unit[0],field))
    assert old_unit[1]==opt_unit[1]
    assert rg._seed_and_grow(old_unit[0],0,-1,old_unit[2])==rg._seed_and_grow(opt_unit[0],0,-1,opt_unit[2])==[[1]]
    assert opt_unit[3]['weight']==0
    # A later omitted/default call remains structural after the opt-in call.
    later=_line_weight_policy_target_graph(monkeypatch)
    np.testing.assert_array_equal(default[0].supports,later[0].supports)
    assert later[2] is rg.DEFAULT_REGION_GROWTH_CONFIG and later[3]['weight']==1


@pytest.mark.parametrize('policy',[None,False,True,1,'raster','UNIT',''])
def test_support_line_weight_policy_invalid_opt_ins_refuse_before_detector(monkeypatch,policy):
    def forbidden(*args,**kwargs): raise AssertionError('detector must not run')
    monkeypatch.setattr(rg,'detect_segments',forbidden)
    with pytest.raises(ValueError,match='internal support line-weight policy'):
        rg._segment_support(np.zeros((2,2),np.uint8),np.ones((2,2),bool),
            np.zeros((2,2),np.int32),_calibration(),rg.DEFAULT_REGION_GROWTH_CONFIG,
            _support_line_weight_policy=policy)
    with pytest.raises(ValueError,match='internal support line-weight policy'):
        rg._build_graph(np.zeros((2,2),np.int32),np.zeros((2,2),np.uint8),
            np.zeros((2,2,3),np.float32),np.ones((2,2),bool),_calibration(),
            rg.DEFAULT_REGION_GROWTH_CONFIG,_support_line_weight_policy=policy)
    with pytest.raises(ValueError,match='internal support line-weight policy'):
        propose_image_planes_v2(np.zeros((2,2,3),np.uint8),np.ones((2,2),bool),
            _calibration(),_support_line_weight_policy=policy)


def test_support_line_weight_policy_is_not_serialized_in_legacy_config():
    from dataclasses import asdict
    expected=dict(analysis_max_side=960,max_input_pixels=6_000_000,atom_spacing=16,
        slic_iterations=3,slic_compactness=10.,connectivity_min_component_fraction=.25,
        max_atoms=20_000,max_segments=5_000,max_segment_samples=128,min_calibration_inliers=12,
        min_usable_fraction=.20,min_seed_score=.26,min_seed_margin=.035,min_growth_score=.055,
        min_seed_support=12.,min_bridge_support=18.,min_continuation_support=5.,
        min_tangent_dominance=.84,min_boundary_axis_alignment=.90,max_lab_delta=40.,
        max_bridge_lab_delta=22.,hard_edge_barrier=48.,bridge_edge_barrier=15.,
        min_region_fraction=.004,max_candidates=32,dedup_iou=.70)
    assert asdict(RegionGrowthConfig())==asdict(rg.DEFAULT_REGION_GROWTH_CONFIG)==expected


def test_support_line_weight_policy_private_keyword_threads_public_graph_allocator_and_seed(monkeypatch):
    authored,_,config,calls=_line_weight_policy_target_graph(monkeypatch)
    monkeypatch.setattr(rg,'_local_lab_xy_atoms',lambda *args,**kwargs:
        (authored.labels.copy(),dict(connectivity_mask_preserved=True)))
    original_build,original_support,original_seed=rg._build_graph,rg._segment_support,rg._seed_and_grow
    forwarding=[];target_regions=[];graphs=[]
    def build(*args,**kwargs):
        forwarding.append(('graph',kwargs['_support_line_weight_policy']))
        result=original_build(*args,**kwargs);graphs.append(result[0]);return result
    def support(*args,**kwargs):
        forwarding.append(('allocator',kwargs['_support_line_weight_policy']))
        return original_support(*args,**kwargs)
    def seed(graph,axis,side,*args,**kwargs):
        result=original_seed(graph,axis,side,*args,**kwargs)
        if axis==0 and side==-1: target_regions.append(result)
        return result
    monkeypatch.setattr(rg,'_build_graph',build)
    monkeypatch.setattr(rg,'_segment_support',support)
    monkeypatch.setattr(rg,'_seed_and_grow',seed)
    image=np.zeros((*authored.labels.shape,3),np.uint8)
    usable=np.ones(authored.labels.shape,bool)
    cal=_calibration(yaw=180,width=97,height=97)
    # Omitted and explicit structural, opt-in unit, then omitted again: no
    # serialization/global/context state can leak the internal experiment.
    for policy in (None,'structural','unit',None):
        before=len(forwarding);before_weight=calls['weight'];before_regions=len(target_regions)
        options={} if policy is None else dict(_support_line_weight_policy=policy)
        propose_image_planes_v2(image,usable,cal,config=config,**options)
        effective='structural' if policy is None else policy
        assert forwarding[before:]==[('graph',effective),('allocator',effective)]
        assert calls['weight']-before_weight==int(effective=='structural')
        if effective=='unit':
            assert target_regions[before_regions:]==[[[1]]]
        else:
            assert target_regions[before_regions:]==[]  # Public no-seed refusal.
    assert config is rg.DEFAULT_REGION_GROWTH_CONFIG
    np.testing.assert_array_equal(graphs[0].supports,graphs[1].supports)
    np.testing.assert_array_equal(graphs[0].supports,graphs[3].supports)
    assert graphs[2].plane_scores[1,0]==pytest.approx(.2982529401779175,abs=1e-8)


def test_authored_pixels_real_detector_to_support_normal_side_and_seed():
    """Six authored tangent planes; no injected detections/support/scores/gates.

    One known connected atom isolates wiring from superpixel fragmentation.
    Camera truth authors/evaluates these pixels only; this is not camera recovery.
    """
    config = rg.DEFAULT_REGION_GROWTH_CONFIG
    cases = [(0, -1, 35., 25.), (0, 1, -35., 25.),
             (1, -1, 35., 25.), (1, 1, 35., -25.),
             (2, 1, 35., 25.), (2, -1, 145., 25.)]
    for normal, side, yaw, pitch in cases:
        cal = _calibration(yaw=yaw, pitch=pitch)
        image, usable, _unused_authored_segments, _ = _synthetic_room(calibration=cal)
        image.fill(130)
        # Independent Minecraft world-to-camera basis, not axis_directions or
        # vanishing_points. Each authored plane has the other two world tangents.
        y, p = np.radians([yaw, pitch])
        basis = np.asarray([[-np.cos(y), 0., -np.sin(y)],
                            [-np.sin(p)*np.sin(y), np.cos(p), np.sin(p)*np.cos(y)],
                            [-np.cos(p)*np.sin(y), -np.sin(p), np.cos(p)*np.cos(y)]])
        centre = np.asarray([WIDTH/2., HEIGHT/2.])
        vps = centre + FOCAL * np.stack([basis[0]/basis[2], -basis[1]/basis[2]], axis=1)
        tangents = [axis for axis in range(3) if axis != normal]
        for tangent in tangents:
            direction = vps[tangent] - centre
            direction /= np.linalg.norm(direction)
            perpendicular = np.asarray([-direction[1], direction[0]])
            for offset in (-72., 0., 72.):
                point = centre + offset * perpendicular
                drawn = np.asarray(_draw_projected_line(
                    image, cal, tangent, point, span=300., value=55)).reshape(2, 2)
                delta = drawn[1] - drawn[0]
                toward = vps[tangent] - drawn.mean(axis=0)
                assert abs(delta[0]*toward[1] - delta[1]*toward[0]) < 1e-7

        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        labels = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
        # Real LSD runs twice on identical authored pixels, once independently
        # retained for a projective reference and once through the actual graph.
        raw = rg.detect_segments(rg.enhance(grey))
        assert len(raw) > 0, (normal, side, "no real detections")
        retained = rg._clip_segments_to_usable_mask(raw, usable)
        assert 0 < len(retained) < config.max_segments
        graph, evidence = rg._build_graph(labels, grey, lab, usable, cal, config)
        assert graph is not None and evidence["segments_detected"] == len(raw)
        assert evidence["segments_clipped"] == len(retained)

        # Independent endpoint-to-pencil displacement, using authored world
        # axes. A single fully usable atom receives each retained line's mass.
        delta = retained[:, 2:] - retained[:, :2]
        midpoint = (retained[:, 2:] + retained[:, :2])/2
        toward = vps[None, :, :] - midpoint[:, None, :]
        cross = np.abs(delta[:, None, 0]*toward[:, :, 1]
                       - delta[:, None, 1]*toward[:, :, 0])
        displacement = cross/(2*np.maximum(np.linalg.norm(toward, axis=2), 1e-9))
        association = np.exp(-.5*(displacement/(rg.INLIER_PIXELS*1.35))**2)
        association[displacement > 3*rg.INLIER_PIXELS] = 0.
        mass = association.sum(axis=1)
        association[mass > 1e-9] /= mass[mass > 1e-9, None]
        association[mass <= 1e-9] = 0.
        weighted_length = np.linalg.norm(delta, axis=1)*rg.structural_weights(grey, retained)
        expected_support = (weighted_length[:, None]*association).sum(axis=0)
        np.testing.assert_allclose(graph.supports[0], expected_support, rtol=2e-5, atol=1e-3)
        assert np.all(graph.supports[0, tangents] > config.min_seed_support)
        assert graph.supports[0, normal] < .2*min(graph.supports[0, tangents])

        support = expected_support.astype(float)
        expected_scores = []
        for axis in range(3):
            first, second = [i for i in range(3) if i != axis]
            tangent_mass = support[first] + support[second]
            expected_scores.append(
                2*np.sqrt(support[first]*support[second])/max(tangent_mass, 1e-6)
                * tangent_mass/max(support.sum(), 1e-6)
                * tangent_mass/(tangent_mass + 24.))
        np.testing.assert_allclose(graph.plane_scores[0], expected_scores, rtol=2e-5, atol=1e-6)
        assert int(np.argmax(graph.plane_scores[0])) == normal
        centroid = np.asarray([(WIDTH-1)/2., (HEIGHT-1)/2.])
        ray = np.asarray([(centroid[0]-centre[0])/FOCAL,
                          (centre[1]-centroid[1])/FOCAL, 1.])
        expected_sides = np.sign(basis.T @ ray).astype(np.int8)
        np.testing.assert_array_equal(graph.sides[0], expected_sides)
        assert expected_sides[normal] == side

        # All six axis/side gates use the authored geometry and independent
        # reference scores; wrong-side/other-normal hypotheses must abstain.
        for axis in range(3):
            for requested_side in (-1, 1):
                other = max(expected_scores[i] for i in range(3) if i != axis)
                eligible = (expected_sides[axis] == requested_side
                            and expected_scores[axis] >= config.min_seed_score
                            and expected_scores[axis]-other >= config.min_seed_margin
                            and support[[i for i in range(3) if i != axis]].sum()
                            >= config.min_seed_support)
                assert eligible == (axis == normal and requested_side == side)
                assert rg._seed_and_grow(graph, axis, requested_side, config) == (
                    [[0]] if eligible else [])
