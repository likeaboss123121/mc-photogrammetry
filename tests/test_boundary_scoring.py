"""Analytic controls for boundary_scoring.py, per the lead review's S2
assignment: a perfect-region control and occluded/clipped/unknown-pixel
controls, independent of any real image or detector."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mc_photogrammetry.reconstruction.boundary_scoring import (
    boundary_error, fill_polygon, region_metrics, target_masks, usable_domain,
)

WIDTH, HEIGHT = 200, 150


def test_a_perfect_candidate_scores_one_on_every_metric():
    target = {"positive_polygon": [[20, 20], [100, 20], [100, 100], [20, 100]],
             "uncertain_polygons": []}
    domain = usable_domain(WIDTH, HEIGHT, top_exclude=0, bottom_exclude=0)
    positive, unknown = target_masks(target, WIDTH, HEIGHT, domain)
    candidate = fill_polygon(target["positive_polygon"], WIDTH, HEIGHT)

    metrics = region_metrics(candidate, positive, domain, unknown)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)

    error = boundary_error(candidate, positive, domain, unknown, WIDTH, HEIGHT)
    assert error < 0.01


def test_a_disjoint_candidate_scores_zero_not_a_crash():
    target = {"positive_polygon": [[20, 20], [60, 20], [60, 60], [20, 60]],
             "uncertain_polygons": []}
    domain = usable_domain(WIDTH, HEIGHT, top_exclude=0, bottom_exclude=0)
    positive, unknown = target_masks(target, WIDTH, HEIGHT, domain)
    candidate = fill_polygon([[120, 90], [160, 90], [160, 130], [120, 130]], WIDTH, HEIGHT)

    metrics = region_metrics(candidate, positive, domain, unknown)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["iou"] == 0.0


def test_the_hud_and_hotbar_bands_are_excluded_from_both_sides():
    """A positive polygon that extends into the excluded top/bottom bands
    (an annotator's polygon does not have to stop exactly at the domain
    edge) must be clipped by the domain before scoring - the excluded
    bands never count for or against a candidate."""
    target = {"positive_polygon": [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]],
             "uncertain_polygons": []}
    domain = usable_domain(WIDTH, HEIGHT, top_exclude=20, bottom_exclude=15)
    positive, unknown = target_masks(target, WIDTH, HEIGHT, domain)
    assert positive[:20, :].sum() == 0
    assert positive[HEIGHT - 15:, :].sum() == 0

    candidate = fill_polygon([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], WIDTH, HEIGHT)
    metrics = region_metrics(candidate, positive, domain, unknown)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_unknown_pixels_are_excluded_from_both_the_candidate_and_the_reference():
    """A candidate that extends into a declared-uncertain region must not
    be penalized for it (not a false positive), and the uncertain area must
    not inflate recall either - it is simply removed from consideration,
    per the review's 'record uncertainty as unknown, not negative'."""
    target = {"positive_polygon": [[20, 20], [80, 20], [80, 80], [20, 80]],
             "uncertain_polygons": [[[80, 20], [120, 20], [120, 80], [80, 80]]]}
    domain = usable_domain(WIDTH, HEIGHT, top_exclude=0, bottom_exclude=0)
    positive, unknown = target_masks(target, WIDTH, HEIGHT, domain)
    # A candidate covering the positive region plus the whole uncertain band.
    candidate = fill_polygon([[20, 20], [120, 20], [120, 80], [20, 80]], WIDTH, HEIGHT)

    metrics = region_metrics(candidate, positive, domain, unknown)
    # Perfect on the positive region; the uncertain area contributes to
    # neither the candidate's area nor a penalty.
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["excluded_unknown_area"] == unknown.sum()

    # Without the exclusion, naively treating the uncertain band as
    # negative would have scored this candidate's precision well under 1 -
    # confirming the exclusion is actually doing something, not a no-op.
    naive_negative_count = int((candidate & domain & ~positive).sum())
    assert naive_negative_count > 0


def test_a_partially_occluded_positive_region_still_scores_a_covering_candidate_well():
    """The positive region itself may be a smaller, irregular shape (an
    occluder already carved out of it upstream, as the wall targets do) -
    a candidate that fully covers that irregular shape, and nothing else,
    still scores perfectly; one that covers its old, larger bounding
    rectangle instead loses precision for the extra area."""
    target = {"positive_polygon": [[20, 20], [100, 20], [100, 100], [20, 100]],
             "uncertain_polygons": []}
    domain = usable_domain(WIDTH, HEIGHT, top_exclude=0, bottom_exclude=0)
    positive, unknown = target_masks(target, WIDTH, HEIGHT, domain)
    # Simulate an occluder already excluded from the declared positive
    # shape by carving a notch out of the mask directly (irregular footprint).
    positive = positive.copy()
    positive[60:100, 60:100] = False

    exact_candidate = positive.copy()
    metrics_exact = region_metrics(exact_candidate, positive, domain, unknown)
    assert metrics_exact["precision"] == pytest.approx(1.0)
    assert metrics_exact["recall"] == pytest.approx(1.0)

    bounding_box_candidate = fill_polygon([[20, 20], [100, 20], [100, 100], [20, 100]], WIDTH, HEIGHT)
    metrics_box = region_metrics(bounding_box_candidate, positive, domain, unknown)
    assert metrics_box["precision"] < 1.0
    assert metrics_box["recall"] == pytest.approx(1.0)


def test_e1_a_negative_polygon_is_subtracted_from_the_broad_positive_extent():
    """A lead follow-up review's exact finding: target_masks ignored
    negative_polygons entirely, so a declared occluder/opening overlapping
    the broad positive extent still counted as positive. A specific
    negative claim must override the broad positive one where they overlap
    - the composition footprint_adapter's own region masks already use."""
    target = {"positive_polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
             "negative_polygons": {"foreground_occlusion": [[[0, 0], [50, 0], [50, 100], [0, 100]]]},
             "uncertain_polygons": []}
    domain = usable_domain(WIDTH, HEIGHT, top_exclude=0, bottom_exclude=0)
    positive, unknown = target_masks(target, WIDTH, HEIGHT, domain)

    assert positive[50, 25] == False  # inside the declared occluder
    assert positive[50, 75] == True   # inside the positive extent, clear of the occluder

    candidate = fill_polygon([[0, 0], [100, 0], [100, 100], [0, 100]], WIDTH, HEIGHT)
    metrics = region_metrics(candidate, positive, domain, unknown)
    # Covering the whole declared extent (including the occluder) must not
    # score perfect precision any more - half of it is a declared negative.
    assert metrics["precision"] == pytest.approx(0.5, abs=0.02)
    assert metrics["recall"] == pytest.approx(1.0)


def test_e1_negative_and_uncertain_overlap_becomes_unknown_not_negative():
    """A pixel both a negative polygon and an uncertain polygon claim is an
    unresolved disagreement between two specific annotations - excluded as
    unknown, not silently kept as a confident negative."""
    target = {"positive_polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
             "negative_polygons": {"opening": [[[40, 40], [80, 40], [80, 80], [40, 80]]]},
             "uncertain_polygons": [[[60, 40], [100, 40], [100, 80], [60, 80]]]}
    domain = usable_domain(WIDTH, HEIGHT, top_exclude=0, bottom_exclude=0)
    positive, unknown = target_masks(target, WIDTH, HEIGHT, domain)

    assert positive[60, 50] == False   # negative-only region
    assert unknown[60, 70] == True     # negative/uncertain overlap
    assert positive[60, 70] == False   # overlap must not read as positive either


def test_e1_an_absent_negative_polygons_key_behaves_like_an_empty_one():
    """A target with no negative_polygons key at all (an older-style
    annotation) must score identically to one with an explicit empty dict -
    absence is not an error, and not silently different behaviour."""
    target_without_key = {"positive_polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                          "uncertain_polygons": []}
    target_with_empty = {"positive_polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                         "negative_polygons": {}, "uncertain_polygons": []}
    domain = usable_domain(WIDTH, HEIGHT, top_exclude=0, bottom_exclude=0)
    positive_a, unknown_a = target_masks(target_without_key, WIDTH, HEIGHT, domain)
    positive_b, unknown_b = target_masks(target_with_empty, WIDTH, HEIGHT, domain)
    assert (positive_a == positive_b).all()
    assert (unknown_a == unknown_b).all()


def test_f3_v3_annotation_anchors_match_the_composed_masks():
    """The lead's fourth review's F3 finding: the v2 annotation's table
    polygon still removed real wall - source pixels (400,600) and (600,600)
    in frame 1 are wall but were recorded as a known negative. The
    replacement `M4-C2-boundary-targets-v3-2026-09-09.json` was built by
    tracing each frame's actual boundaries directly from a labelled
    source-coordinate grid (see `docs/reports/M4-F3-annotation-audit-2026-09-09/`
    for the crop-by-crop trace and contact sheets) - the anchors below were
    placed by reading the image first, not by adjusting vertices until this
    test passed. This test only guards against a future edit silently
    breaking what the trace already established: every `wall_interior_anchors`
    point (including the review's own (400,600) and (600,600)) must resolve
    positive, and every `negative_anchors` point must resolve negative, in
    all three frames.
    """
    import hashlib
    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1]
           / "docs/reports/M4-C2-boundary-targets-v3-2026-09-09.json")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "082ec648845be2354a5324821e6484291ca867563948e0396cecb496e5adf56a"
    ), "the frozen v3 annotation file changed - re-verify and re-freeze before editing this test"
    data = json.loads(raw)

    for target in data["targets"]:
        width, height = target["image_size"]
        domain = usable_domain(width, height, top_exclude=0, bottom_exclude=0)
        positive, unknown = target_masks(target, width, height, domain)
        for x, y in target["wall_interior_anchors"]:
            assert positive[y, x], (target["frame"], "wall_interior_anchor", x, y)
        for x, y in target["negative_anchors"]:
            assert not positive[y, x] and not unknown[y, x], (
                target["frame"], "negative_anchor", x, y)

    # The review's own two cited pixels, checked by name rather than only by
    # position in the list, so a reordering of the anchors above cannot
    # silently drop this specific check.
    frame1 = next(t for t in data["targets"] if t["frame"] == "2026-09-02_13.22.14.png")
    width, height = frame1["image_size"]
    domain = usable_domain(width, height, top_exclude=0, bottom_exclude=0)
    positive, _ = target_masks(frame1, width, height, domain)
    assert positive[600, 400] and positive[600, 600]
