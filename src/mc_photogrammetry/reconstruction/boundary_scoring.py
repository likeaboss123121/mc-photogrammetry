"""Scoring a `boundary_candidates` proposal against a frozen evaluation target.

Kept separate from `boundary_candidates.py` on purpose: that module proposes,
this module grades - a lead review named exactly this boundary explicitly
("target geometry and reference cameras remain unavailable to the
proposer"). Nothing here is imported by the proposal path.

A target has three explicit regions, not two: `positive` (confidently this
plane's own visible material), `negative` (confidently not - a different
surface, an occluder, an opening) and `unknown` (a genuinely ambiguous
transition, excluded from scoring in both directions rather than guessed).
Precision/recall/IoU are computed only over the ``domain`` (the usable
image area) with `unknown` pixels removed from both the candidate and the
reference - an unknown pixel a candidate happens to cover is neither a hit
nor a miss, since the target itself does not know what is there.
"""
from __future__ import annotations

import math

import numpy as np


def usable_domain(width: int, height: int, *, top_exclude: int, bottom_exclude: int) -> np.ndarray:
    """The image area scoring considers at all - excludes the F3 HUD band
    at top and the hotbar (plus, on some frames, a screenshot-saved
    notice) at bottom. The same domain a proposer should crop to before
    feature extraction, so neither side of the comparison sees pixels the
    other is denied."""
    domain = np.ones((height, width), dtype=bool)
    domain[:top_exclude, :] = False
    if bottom_exclude > 0:
        domain[height - bottom_exclude:, :] = False
    return domain


def fill_polygon(polygon, width: int, height: int) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon:
        points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 255)
    return mask > 0


def target_masks(target: dict, width: int, height: int, domain: np.ndarray):
    """Returns ``(positive_mask, unknown_mask)``, both already intersected
    with `domain`.

    Composition is explicit, in this order: start from ``positive_polygon``
    (the broad claim - "this is the wall"), then remove every polygon in
    ``negative_polygons`` (any sub-category - an occluder or an opening is
    a specific, more-local claim that overrides the broad one where they
    overlap, the same "more specific wins" rule `footprint_adapter`'s own
    region composition uses), then remove ``uncertain_polygons`` (a
    genuinely ambiguous transition is excluded regardless of what the
    broad or specific claims said). A pixel a negative polygon and an
    uncertain polygon both cover is therefore excluded as unknown, not
    silently kept negative - an unresolved specific disagreement is
    exactly the kind of thing this category exists for. Everything in
    ``domain`` that ends up in neither ``positive`` nor ``unknown`` is a
    negative by construction; ``negative_polygons`` exist to make that
    explicit and to label overlays, not as a separate scoring category
    computed independently of ``positive_polygon``.
    """
    positive = fill_polygon(target.get("positive_polygon", []), width, height)
    negative = np.zeros((height, width), dtype=bool)
    for group in target.get("negative_polygons", {}).values():
        for polygon in group:
            negative |= fill_polygon(polygon, width, height)
    unknown = np.zeros((height, width), dtype=bool)
    for polygon in target.get("uncertain_polygons", []):
        unknown |= fill_polygon(polygon, width, height)
    unknown &= domain
    positive = positive & domain & ~negative & ~unknown
    return positive, unknown


def region_metrics(candidate_mask: np.ndarray, positive_mask: np.ndarray,
                   domain: np.ndarray, unknown_mask: np.ndarray) -> dict:
    """Precision/recall/IoU over ``domain`` with `unknown` pixels excluded
    from both sides. ``candidate_mask`` need not already be domain-clipped -
    this function does that."""
    scored_candidate = candidate_mask & domain & ~unknown_mask
    true_positive = int((scored_candidate & positive_mask).sum())
    candidate_area = int(scored_candidate.sum())
    positive_area = int(positive_mask.sum())
    union_area = int((scored_candidate | positive_mask).sum())
    precision = true_positive / candidate_area if candidate_area else 0.0
    recall = true_positive / positive_area if positive_area else 0.0
    iou = true_positive / union_area if union_area else 0.0
    return {
        "precision": precision, "recall": recall, "iou": iou,
        "candidate_area": candidate_area, "positive_area": positive_area,
        "true_positive_area": true_positive,
        "excluded_unknown_area": int(unknown_mask.sum()),
    }


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """A one-pixel-wide boundary of a boolean region mask, as a uint8 0/255
    image - derived from the mask actually scored (already domain-clipped
    and with unknown pixels removed), not from a declared polygon's raw
    vertices, which may extend outside the domain entirely."""
    import cv2

    mask_u8 = (mask.astype(np.uint8)) * 255
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8))
    return mask_u8 & ~eroded


def boundary_error(candidate_mask: np.ndarray, positive_mask: np.ndarray,
                   domain: np.ndarray, unknown_mask: np.ndarray,
                   width: int, height: int) -> float:
    """Symmetric mean boundary distance between the scored candidate region
    and the scored positive region, normalized by the image diagonal.

    Formula: for each boundary pixel of one region, its distance (in
    pixels, via `cv2.distanceTransform`, L2) to the nearest boundary pixel
    of the other region; average both directions; divide by
    ``hypot(width, height)``. Both masks are the actual scored regions
    (domain-clipped, unknown pixels removed) - not the original declared
    polygons, which the domain crop and unknown exclusion may have since
    cut into a different shape.
    """
    import cv2

    scored_candidate = candidate_mask & domain & ~unknown_mask
    scored_positive = positive_mask  # already domain/unknown-clipped by target_masks
    candidate_boundary = mask_boundary(scored_candidate)
    positive_boundary = mask_boundary(scored_positive)
    diagonal = float(np.hypot(width, height))
    if not candidate_boundary.any() or not positive_boundary.any():
        return float("nan")
    distance_to_positive = cv2.distanceTransform(255 - positive_boundary, cv2.DIST_L2, 5)
    distance_to_candidate = cv2.distanceTransform(255 - candidate_boundary, cv2.DIST_L2, 5)
    mean_a = float(distance_to_positive[candidate_boundary > 0].mean())
    mean_b = float(distance_to_candidate[positive_boundary > 0].mean())
    return (mean_a + mean_b) / 2.0 / diagonal
