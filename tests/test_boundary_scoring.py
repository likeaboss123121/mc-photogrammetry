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
