"""Image-only SIFT correspondence extraction for connected-camera pairs.

DS-12's module: it turns two RGB frames into a clean set of mutual pixel
correspondences that :func:`mc_photogrammetry.pipeline.connected_camera.solve_pair`
can consume. It is deliberately blind to cameras, scenes, lattices, manifests and
references - extraction sees pixels and a usable-pixel mask and nothing else - so
a correspondence set can be frozen before any evaluation code looks at a fixture
manifest.

The usable-pixel domain follows the same authority DS-4 established for the
reconstructor (``raycast._pixel_authority``):

* ``usable_masks is None`` derives ``~create_gui_mask(h, w)`` per frame, so the
  default domain is the production HUD exclusion;
* a supplied mapping is authoritative - a frame it omits is unusable (all false);
* a mask whose shape is not the frame's ``(height, width)`` raises ``ValueError``
  before any feature is detected.

Repetition is the central failure mode for Minecraft texture: a wall of identical
blocks yields near-equal nearest and second-nearest descriptors, so the ratio
test has nothing to separate. Extraction therefore reports its stages honestly
(raw keypoints, ratio survivors in both directions, mutual survivors, spatial-bin
coverage, descriptor-distance statistics) and labels a pair ``insufficient_*`` or
``repetition_risk`` rather than padding the match list to look usable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from mc_photogrammetry.vision.gui_mask import create_gui_mask

__all__ = [
    "MATCH_STATUSES",
    "CorrespondenceSettings",
    "FrameFeatures",
    "MatchSelection",
    "PairMatches",
    "ReciprocalGraph",
    "boundary_distances",
    "descriptor_footprint_keep",
    "detect_features",
    "extract_pair_matches",
    "reciprocal_topk_graph",
    "select_mutual_matches",
    "usable_mask_for",
]

MATCH_STATUSES: tuple[str, ...] = (
    "ok",
    "unusable_frame",
    "insufficient_keypoints",
    "insufficient_matches",
    "repetition_risk",
)
"""Every status :class:`PairMatches` may report."""

_TINY = 1e-12
_DISTANCE_CHUNK = 256


@dataclass(frozen=True, slots=True)
class CorrespondenceSettings:
    """Bounds for one extraction; the defaults are declared, not tuned per pair.

    ``min_mutual_fraction`` is the repetition guard: a frame pair whose raw
    keypoints barely survive the ratio test and the mutual check is reporting
    ambiguity, and refusing it is more honest than feeding noisy matches to the
    pair solver.
    """

    max_features: int = 4000
    ratio_test: float = 0.75
    min_matches: int = 12
    min_keypoints_per_frame: int = 24
    spatial_grid: tuple[int, int] = (4, 4)
    min_spatial_coverage: float = 0.2
    min_mutual_fraction: float = 0.02
    mask_boundary_pad_px: float = 0.0
    """Extra margin, in pixels, before a keypoint's descriptor footprint is kept.

    ``0.0`` is the frozen DS-12 behaviour: no boundary filtering at all, so every
    existing caller and artifact is unchanged. A positive value rejects a
    keypoint whose SIFT footprint (radius ``0.5 * keypoint.size``) reaches within
    ``mask_boundary_pad_px`` of the excluded domain, removing descriptors SIFT
    placed on the synthetic mask contour. The filter only shrinks the usable
    domain (it never invents pixels or reads outside ``U``).
    """

    def __post_init__(self) -> None:
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if not 0.0 < self.ratio_test < 1.0:
            raise ValueError("ratio_test must lie in (0, 1)")
        if self.min_matches < 1:
            raise ValueError("min_matches must be positive")
        if self.min_keypoints_per_frame < 0:
            raise ValueError("min_keypoints_per_frame must be non-negative")
        grid_x, grid_y = self.spatial_grid
        if grid_x < 1 or grid_y < 1:
            raise ValueError("spatial_grid dimensions must be positive")
        if not 0.0 <= self.min_spatial_coverage <= 1.0:
            raise ValueError("min_spatial_coverage must lie in [0, 1]")
        if not 0.0 <= self.min_mutual_fraction <= 1.0:
            raise ValueError("min_mutual_fraction must lie in [0, 1]")
        if self.mask_boundary_pad_px < 0.0:
            raise ValueError("mask_boundary_pad_px must be non-negative")


@dataclass(frozen=True, slots=True)
class MatchSelection:
    """The pure match-filtering result, independent of any image or OpenCV call.

    ``matches`` has one row per surviving correspondence as
    ``(left_x, left_y, right_x, right_y)``. The counters describe the funnel
    before cleaning: ``ratio_left``/``ratio_right`` are the directed KNN ratio
    survivors and ``mutual`` is their reciprocal intersection.
    """

    matches: np.ndarray
    left_keypoints: int
    right_keypoints: int
    ratio_left: int
    ratio_right: int
    mutual: int
    discarded_non_finite: int
    discarded_duplicate: int
    distance_min: float | None
    distance_median: float | None
    distance_mean: float | None
    distance_max: float | None


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    """SIFT output for one frame inside its usable-pixel domain.

    ``sizes`` (the SIFT keypoint diameters) and ``boundary_dropped`` are additive
    DS-14 fields. Both default to the frozen DS-12 shape, so existing constructors
    and artifacts are unchanged: ``sizes is None`` means "not recorded" and
    ``boundary_dropped == 0`` means "no descriptor-footprint filtering ran".
    """

    frame_id: str
    points: np.ndarray
    descriptors: np.ndarray
    width: int
    height: int
    usable_pixels: int
    sizes: np.ndarray | None = None
    boundary_dropped: int = 0


@dataclass(frozen=True, slots=True)
class ReciprocalGraph:
    """Symmetric top-k descriptor candidate graph for one frame pair.

    ``edges`` rows are ``(left_index, right_index)`` of the *reciprocal* k-NN
    graph (an edge survives only when each endpoint is in the other's k nearest),
    sorted by ``(distance, left_index, right_index)`` and truncated to the
    declared cap. ``truncated`` counts edges dropped by that cap and is reported,
    never silent; ``total_edges`` is the untruncated reciprocal count.
    """

    edges: np.ndarray
    distances: np.ndarray
    left_keypoints: int
    right_keypoints: int
    total_edges: int
    truncated: int


def reciprocal_topk_graph(
    left_descriptors: np.ndarray,
    right_descriptors: np.ndarray,
    *,
    neighbors: int = 5,
    max_edges: int = 8192,
    distance_ceiling: float | None = None,
) -> ReciprocalGraph:
    """Build the reciprocal top-k descriptor graph, deterministically.

    Each left keypoint keeps its ``neighbors`` nearest right descriptors and vice
    versa; an edge survives when both directions agree, so the graph is symmetric
    and many-to-one by construction (one-to-one adjudication is the geometry
    stage's job). Edges are sorted by ``(distance, left, right)`` so a cap
    truncates the least trustworthy edges first; ties break on indices, never on
    memory order.
    """
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if max_edges < 1:
        raise ValueError("max_edges must be positive")
    if distance_ceiling is not None and distance_ceiling < 0.0:
        raise ValueError("distance_ceiling must be non-negative")
    left = _descriptors(left_descriptors, "left_descriptors")
    right = _descriptors(right_descriptors, "right_descriptors")
    left_count, right_count = len(left), len(right)
    if left_count == 0 or right_count == 0 or left.size == 0 or right.size == 0:
        return _empty_graph(left_count, right_count)
    if left.shape[1] != right.shape[1]:
        raise ValueError("descriptor dimensions must match")

    left_k = min(int(neighbors), right_count)
    right_k = min(int(neighbors), left_count)
    left_index, left_distance = _k_nearest(left, right, left_k)
    right_index, _right_distance = _k_nearest(right, left, right_k)
    right_sets = [set(int(value) for value in row if value >= 0) for row in right_index]

    edge_rows: list[tuple[int, int, float]] = []
    for origin in range(left_count):
        for slot in range(left_index.shape[1]):
            target = int(left_index[origin, slot])
            if target < 0 or origin not in right_sets[target]:
                continue
            distance = float(left_distance[origin, slot])
            if distance_ceiling is not None and distance > distance_ceiling:
                continue
            edge_rows.append((origin, target, distance))

    if not edge_rows:
        return _empty_graph(left_count, right_count)
    table = np.asarray(edge_rows, dtype=np.float64)
    order = np.lexsort((table[:, 1], table[:, 0], table[:, 2]))
    table = table[order]
    total = int(len(table))
    truncated = max(0, total - int(max_edges))
    table = table[: int(max_edges)]
    return ReciprocalGraph(
        edges=table[:, :2].astype(np.int64),
        distances=table[:, 2].astype(np.float64),
        left_keypoints=left_count,
        right_keypoints=right_count,
        total_edges=total,
        truncated=truncated,
    )


def boundary_distances(usable: np.ndarray) -> np.ndarray:
    """Per-pixel distance (px) to the nearest excluded pixel of ``usable``.

    The authoritative usable mask is the only input; excluded pixels score
    ``0.0``. The transform is a deterministic function of the mask, so the
    descriptor-footprint rule below introduces no second authority.
    """
    import cv2

    mask = np.asarray(usable, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("usable mask must be a 2-D array")
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def descriptor_footprint_keep(
    points: np.ndarray,
    sizes: np.ndarray | None,
    usable: np.ndarray,
    pad_px: float,
) -> np.ndarray:
    """Boolean keep-mask for keypoints whose descriptor footprint touches ``~U``.

    A keypoint with SIFT diameter ``size`` reaches ``0.5 * size`` pixels from its
    centre; it is rejected when that footprint comes within ``pad_px`` of the
    excluded domain, i.e. when ``dist(p, ~U) <= pad_px + 0.5 * size``. ``pad_px
    == 0`` short-circuits to "keep everything" so the DS-12 path is byte-for-byte
    unchanged. The rule only removes keypoints from ``U`` - it can never add one.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if pad_px <= 0.0 or len(coordinates) == 0:
        return np.ones(len(coordinates), dtype=bool)
    distance = boundary_distances(usable)
    height, width = distance.shape
    columns = np.clip(np.rint(coordinates[:, 0]).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(coordinates[:, 1]).astype(np.int64), 0, height - 1)
    radius = (
        np.zeros(len(coordinates), dtype=np.float64)
        if sizes is None
        else 0.5 * np.asarray(sizes, dtype=np.float64)
    )
    return distance[rows, columns] > (float(pad_px) + radius)


@dataclass(frozen=True, slots=True)
class PairMatches:
    """One pair's correspondence evidence plus the reason it is or is not usable.

    ``feedable`` is the single gate the diagnostic uses: only an ``ok`` pair is
    handed to the relative-pose solver. Every other status carries a typed
    ``reason`` and is reported as a refusal, never padded into a solve.
    """

    left_id: str
    right_id: str
    status: str
    reason: str
    matches: np.ndarray
    left: FrameFeatures
    right: FrameFeatures
    selection: MatchSelection
    spatial_bins_left: int
    spatial_bins_right: int
    spatial_bins_total: int
    spatial_coverage: float
    mutual_fraction: float
    ratio_test: float

    @property
    def feedable(self) -> bool:
        return self.status == "ok"


def usable_mask_for(
    frame_id: str,
    shape: tuple[int, ...],
    usable_masks: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    """Resolve one frame's usable-pixel mask with the DS-4 authority semantics.

    ``None`` derives the production GUI exclusion; a supplied mapping is
    authoritative and a missing frame is all-false; a wrong-shaped mask raises.
    The returned array is always a boolean array of ``shape[:2]``.
    """
    height, width = int(shape[0]), int(shape[1])
    if usable_masks is None:
        return ~create_gui_mask(height, width)
    supplied = usable_masks.get(frame_id)
    if supplied is None:
        return np.zeros((height, width), dtype=bool)
    mask = np.asarray(supplied, dtype=bool)
    if mask.shape != (height, width):
        raise ValueError("usable mask must match the image dimensions")
    return mask


def detect_features(
    frame_id: str,
    frame: np.ndarray,
    *,
    usable_mask: np.ndarray | None = None,
    settings: CorrespondenceSettings | None = None,
) -> FrameFeatures:
    """Detect SIFT features in one frame, restricted to its usable pixels.

    Excluded pixels are zeroed *before* detection as well as being excluded by
    the detector mask, so RGB changes outside the domain cannot reach a keypoint
    or a descriptor. ``None`` derives ``~create_gui_mask``.
    """
    import cv2

    bounds = CorrespondenceSettings() if settings is None else settings
    image = _validate_frame(frame)
    height, width = image.shape[:2]
    if usable_mask is None:
        usable = ~create_gui_mask(height, width)
    else:
        usable = np.asarray(usable_mask, dtype=bool)
        if usable.shape != (height, width):
            raise ValueError("usable mask must match the image dimensions")
    grey = _to_grey(image).copy()
    grey[~usable] = 0
    detector = cv2.SIFT_create(nfeatures=int(bounds.max_features))
    points, descriptors = detector.detectAndCompute(
        grey, (usable * 255).astype(np.uint8)
    )
    if points is None or len(points) == 0 or descriptors is None:
        return FrameFeatures(
            frame_id=frame_id,
            points=np.empty((0, 2), dtype=np.float64),
            descriptors=np.empty((0, 0), dtype=np.float32),
            width=width,
            height=height,
            usable_pixels=int(np.count_nonzero(usable)),
            sizes=np.empty(0, dtype=np.float64),
            boundary_dropped=0,
        )
    pixel_points = np.asarray([keypoint.pt for keypoint in points], dtype=np.float64)
    sizes = np.asarray([keypoint.size for keypoint in points], dtype=np.float64)
    descriptor_array = np.asarray(descriptors)
    keep = descriptor_footprint_keep(
        pixel_points, sizes, usable, bounds.mask_boundary_pad_px
    )
    boundary_dropped = int(np.count_nonzero(~keep))
    if boundary_dropped:
        pixel_points = pixel_points[keep]
        descriptor_array = descriptor_array[keep]
        sizes = sizes[keep]
    return FrameFeatures(
        frame_id=frame_id,
        points=pixel_points,
        descriptors=descriptor_array,
        width=width,
        height=height,
        usable_pixels=int(np.count_nonzero(usable)),
        sizes=sizes,
        boundary_dropped=boundary_dropped,
    )


def select_mutual_matches(
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_descriptors: np.ndarray,
    right_descriptors: np.ndarray,
    *,
    ratio_test: float = 0.75,
) -> MatchSelection:
    """Pure bidirectional KNN ratio filter over two descriptor tables.

    Both directions run the ratio test; a pair survives only when each side is
    the other's ratio-surviving nearest neighbour, which makes the result
    one-to-one by construction. Non-finite pixel rows and duplicate pixel pairs
    are then dropped. The function knows nothing about images, cameras or
    references, so it is directly testable with synthetic descriptor tables.
    """
    if not 0.0 < ratio_test < 1.0:
        raise ValueError("ratio_test must lie in (0, 1)")
    left_points = _points(left_points, "left_points")
    right_points = _points(right_points, "right_points")
    left_descriptors = _descriptors(left_descriptors, "left_descriptors")
    right_descriptors = _descriptors(right_descriptors, "right_descriptors")
    if len(left_points) != len(left_descriptors):
        raise ValueError("left_points and left_descriptors must have equal length")
    if len(right_points) != len(right_descriptors):
        raise ValueError("right_points and right_descriptors must have equal length")

    left_count, right_count = len(left_points), len(right_points)
    if (
        left_count == 0
        or right_count == 0
        or left_descriptors.size == 0
        or right_descriptors.size == 0
    ):
        return _empty_selection(left_count, right_count)
    if left_descriptors.shape[1] != right_descriptors.shape[1]:
        raise ValueError("descriptor dimensions must match")
    if min(left_count, right_count) < 2:
        return _empty_selection(left_count, right_count)

    left_index, left_distance = _two_nearest(left_descriptors, right_descriptors)
    right_index, right_distance = _two_nearest(right_descriptors, left_descriptors)

    ratio_left = _ratio_survivors(left_distance, ratio_test)
    ratio_right = _ratio_survivors(right_distance, ratio_test)
    left_best = {
        int(origin): int(target)
        for origin in np.flatnonzero(ratio_left)
        if (target := int(left_index[origin, 0])) >= 0
    }
    right_best = {
        int(origin): int(target)
        for origin in np.flatnonzero(ratio_right)
        if (target := int(right_index[origin, 0])) >= 0
    }
    mutual = [
        (origin, target)
        for origin, target in sorted(left_best.items())
        if right_best.get(target) == origin
    ]

    left_indices = np.asarray([pair[0] for pair in mutual], dtype=np.int64)
    right_indices = np.asarray([pair[1] for pair in mutual], dtype=np.int64)
    rows = np.column_stack(
        (
            left_points[left_indices],
            right_points[right_indices],
        )
    )
    finite = np.all(np.isfinite(rows), axis=1) if len(rows) else np.zeros(0, dtype=bool)
    discarded_non_finite = int(np.count_nonzero(~finite))
    rows = rows[finite]
    left_indices = left_indices[finite]
    right_indices = right_indices[finite]

    if len(rows):
        _unique, first = np.unique(rows, axis=0, return_index=True)
        keep = np.sort(first)
    else:
        keep = np.zeros(0, dtype=np.int64)
    discarded_duplicate = int(len(rows) - len(keep))
    rows = rows[keep]
    left_indices = left_indices[keep]
    right_indices = right_indices[keep]

    distances = (
        np.linalg.norm(
            left_descriptors[left_indices] - right_descriptors[right_indices], axis=1
        )
        if len(rows)
        else np.zeros(0, dtype=np.float64)
    )
    minimum, median, mean, maximum = _distance_statistics(distances)
    return MatchSelection(
        matches=rows.astype(np.float64, copy=False),
        left_keypoints=left_count,
        right_keypoints=right_count,
        ratio_left=int(np.count_nonzero(ratio_left)),
        ratio_right=int(np.count_nonzero(ratio_right)),
        mutual=len(mutual),
        discarded_non_finite=discarded_non_finite,
        discarded_duplicate=discarded_duplicate,
        distance_min=minimum,
        distance_median=median,
        distance_mean=mean,
        distance_max=maximum,
    )


def extract_pair_matches(
    left_id: str,
    left_frame: np.ndarray,
    right_id: str,
    right_frame: np.ndarray,
    *,
    usable_masks: Mapping[str, np.ndarray] | None = None,
    settings: CorrespondenceSettings | None = None,
) -> PairMatches:
    """Extract one pair's correspondences and classify its usability.

    Extraction reads only the two frames and, when supplied, the authoritative
    usable-pixel mapping: no camera, scene, lattice or reference data is
    consulted, so the result can be frozen before any evaluation runs.
    """
    bounds = CorrespondenceSettings() if settings is None else settings
    left_mask = usable_mask_for(left_id, left_frame.shape, usable_masks)
    right_mask = usable_mask_for(right_id, right_frame.shape, usable_masks)
    left = detect_features(left_id, left_frame, usable_mask=left_mask, settings=bounds)
    right = detect_features(right_id, right_frame, usable_mask=right_mask, settings=bounds)
    selection = select_mutual_matches(
        left.points,
        right.points,
        left.descriptors,
        right.descriptors,
        ratio_test=bounds.ratio_test,
    )
    matches = selection.matches
    bins_left = _bins_covered(matches[:, 0:2], left.width, left.height, bounds.spatial_grid)
    bins_right = _bins_covered(matches[:, 2:4], right.width, right.height, bounds.spatial_grid)
    bins_total = int(bounds.spatial_grid[0] * bounds.spatial_grid[1])
    coverage = min(bins_left, bins_right) / bins_total if bins_total else 0.0
    smaller = min(selection.left_keypoints, selection.right_keypoints)
    mutual_fraction = selection.mutual / smaller if smaller else 0.0

    status, reason = _classify(
        left,
        right,
        matches.shape[0],
        coverage,
        mutual_fraction,
        bounds,
    )
    return PairMatches(
        left_id=left_id,
        right_id=right_id,
        status=status,
        reason=reason,
        matches=matches,
        left=left,
        right=right,
        selection=selection,
        spatial_bins_left=bins_left,
        spatial_bins_right=bins_right,
        spatial_bins_total=bins_total,
        spatial_coverage=coverage,
        mutual_fraction=mutual_fraction,
        ratio_test=bounds.ratio_test,
    )


def _classify(
    left: FrameFeatures,
    right: FrameFeatures,
    match_count: int,
    coverage: float,
    mutual_fraction: float,
    bounds: CorrespondenceSettings,
) -> tuple[str, str]:
    """The first gate a pair fails, as a typed status and a human reason."""
    if left.usable_pixels == 0 or right.usable_pixels == 0:
        return "unusable_frame", "one frame has no usable pixels"
    raw_left, raw_right = len(left.points), len(right.points)
    if min(raw_left, raw_right) < bounds.min_keypoints_per_frame:
        return (
            "insufficient_keypoints",
            f"raw keypoints {raw_left}/{raw_right} below "
            f"{bounds.min_keypoints_per_frame}",
        )
    if match_count < bounds.min_matches:
        return (
            "insufficient_matches",
            f"{match_count} mutual matches below {bounds.min_matches}",
        )
    if coverage < bounds.min_spatial_coverage:
        return (
            "repetition_risk",
            f"spatial coverage {coverage:.2f} below {bounds.min_spatial_coverage:.2f}",
        )
    if mutual_fraction < bounds.min_mutual_fraction:
        return (
            "repetition_risk",
            f"mutual fraction {mutual_fraction:.4f} below {bounds.min_mutual_fraction:.4f}",
        )
    return "ok", ""


def _ratio_survivors(distance: np.ndarray, ratio_test: float) -> np.ndarray:
    """Ratio-test mask; a row without two candidates cannot survive."""
    if distance.size == 0:
        return np.zeros(0, dtype=bool)
    has_two = distance[:, 1] >= 0.0
    with np.errstate(invalid="ignore"):
        passed = distance[:, 0] < ratio_test * distance[:, 1]
    return has_two & passed


def _two_nearest(query: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two nearest L2 neighbours of every query row, in deterministic order.

    Distances are expanded through the Gram matrix in blocks, which keeps the
    full ``N x M`` distance matrix out of memory without changing the result.
    A train set with fewer than two rows reports ``-1``/``inf`` for the second
    slot so the ratio test refuses those rows.
    """
    query = np.asarray(query, dtype=np.float64)
    train = np.asarray(train, dtype=np.float64)
    rows = len(query)
    index = np.full((rows, 2), -1, dtype=np.int64)
    distance = np.full((rows, 2), np.inf, dtype=np.float64)
    if rows == 0 or len(train) == 0:
        return index, distance
    neighbours = min(2, len(train))
    train_squared = np.einsum("ij,ij->i", train, train)
    for start in range(0, rows, _DISTANCE_CHUNK):
        block = query[start : start + _DISTANCE_CHUNK]
        gram = block @ train.T
        squared = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + train_squared[None, :]
            - 2.0 * gram
        )
        np.maximum(squared, 0.0, out=squared)
        candidate = np.argpartition(squared, neighbours - 1, axis=1)[:, :neighbours]
        candidate_distance = np.take_along_axis(squared, candidate, axis=1)
        order = np.argsort(candidate_distance, axis=1, kind="stable")
        candidate = np.take_along_axis(candidate, order, axis=1)
        candidate_distance = np.take_along_axis(candidate_distance, order, axis=1)
        index[start : start + _DISTANCE_CHUNK, :neighbours] = candidate
        distance[start : start + _DISTANCE_CHUNK, :neighbours] = np.sqrt(
            candidate_distance
        )
    if neighbours < 2:
        index[:, 1] = -1
        distance[:, 1] = np.inf
    return index, distance


def _k_nearest(query: np.ndarray, train: np.ndarray, neighbours: int) -> tuple[np.ndarray, np.ndarray]:
    """``neighbours`` nearest L2 train rows for every query row.

    The DS-14 graph generalisation of :func:`_two_nearest`: the same blocked Gram
    expansion keeps the full distance matrix out of memory, and equal distances
    are broken by train index (``lexsort``) so the graph is deterministic across
    runs and memory orders. Rows with fewer than ``neighbours`` candidates report
    ``-1`` / ``inf`` in the unused slots.
    """
    query = np.asarray(query, dtype=np.float64)
    train = np.asarray(train, dtype=np.float64)
    rows = len(query)
    keep = max(1, min(int(neighbours), len(train)))
    index = np.full((rows, keep), -1, dtype=np.int64)
    distance = np.full((rows, keep), np.inf, dtype=np.float64)
    if rows == 0 or len(train) == 0:
        return index, distance
    train_squared = np.einsum("ij,ij->i", train, train)
    for start in range(0, rows, _DISTANCE_CHUNK):
        block = query[start : start + _DISTANCE_CHUNK]
        gram = block @ train.T
        squared = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + train_squared[None, :]
            - 2.0 * gram
        )
        np.maximum(squared, 0.0, out=squared)
        candidate = np.argpartition(squared, keep - 1, axis=1)[:, :keep]
        candidate_distance = np.take_along_axis(squared, candidate, axis=1)
        order = np.lexsort((candidate, candidate_distance), axis=1)
        candidate = np.take_along_axis(candidate, order, axis=1)
        candidate_distance = np.take_along_axis(candidate_distance, order, axis=1)
        index[start : start + _DISTANCE_CHUNK] = candidate
        distance[start : start + _DISTANCE_CHUNK] = np.sqrt(candidate_distance)
    if len(train) < keep:
        index[:, len(train) :] = -1
        distance[:, len(train) :] = np.inf
    return index, distance


def _bins_covered(
    points: np.ndarray, width: int, height: int, grid: tuple[int, int]
) -> int:
    """Distinct spatial bins holding at least one matched endpoint."""
    if points.size == 0:
        return 0
    grid_x, grid_y = grid
    column = np.clip((points[:, 0] / max(width, 1) * grid_x).astype(int), 0, grid_x - 1)
    row = np.clip((points[:, 1] / max(height, 1) * grid_y).astype(int), 0, grid_y - 1)
    return int(np.unique(column * grid_y + row).size)


def _distance_statistics(
    distances: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None]:
    if distances.size == 0:
        return None, None, None, None
    return (
        float(np.min(distances)),
        float(np.median(distances)),
        float(np.mean(distances)),
        float(np.max(distances)),
    )


def _empty_graph(left_count: int, right_count: int) -> ReciprocalGraph:
    return ReciprocalGraph(
        edges=np.empty((0, 2), dtype=np.int64),
        distances=np.empty(0, dtype=np.float64),
        left_keypoints=left_count,
        right_keypoints=right_count,
        total_edges=0,
        truncated=0,
    )


def _empty_selection(left_count: int, right_count: int) -> MatchSelection:
    return MatchSelection(
        matches=np.empty((0, 4), dtype=np.float64),
        left_keypoints=left_count,
        right_keypoints=right_count,
        ratio_left=0,
        ratio_right=0,
        mutual=0,
        discarded_non_finite=0,
        discarded_duplicate=0,
        distance_min=None,
        distance_median=None,
        distance_mean=None,
        distance_max=None,
    )


def _points(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be an N x 2 array")
    return array


def _descriptors(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 0)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return array


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype != np.uint8:
        raise ValueError("correspondence extraction expects uint8 frames")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(
        f"expected an RGB frame with shape (height, width, 3), got {image.shape}"
    )


def _to_grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
