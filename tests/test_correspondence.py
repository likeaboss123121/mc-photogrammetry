"""Controls for image-only SIFT correspondence extraction (DS-12).

The pure match filter is exercised with synthetic descriptor tables, so the
one-to-one, ratio and cleanup behaviour is pinned without any image or OpenCV
call. The remaining controls build bounded synthetic frames and read only the
public :class:`PairMatches` result: excluded-RGB invariance, the DS-4 usable-mask
authority, a real SIFT smoke pair made from unique geometric marks, and a
repeated tile grid that must be flagged instead of matched confidently.
"""
import cv2
import numpy as np
import pytest

from mc_photogrammetry.pipeline.correspondence import (
    CorrespondenceSettings,
    descriptor_footprint_keep,
    detect_features,
    extract_pair_matches,
    reciprocal_topk_graph,
    select_mutual_matches,
    usable_mask_for,
)
from mc_photogrammetry.vision.gui_mask import create_gui_mask


def _noise(width: int, height: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _marks(width: int = 380, height: int = 300, seed: int = 5) -> np.ndarray:
    """A bounded image of unique geometric marks: corners with distinct interiors."""
    generator = np.random.default_rng(seed)
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    for _ in range(30):
        mark_width = int(generator.integers(14, 34))
        mark_height = int(generator.integers(14, 34))
        x = int(generator.integers(4, width - mark_width - 4))
        y = int(generator.integers(4, height - mark_height - 4))
        colour = tuple(int(value) for value in generator.integers(70, 256, 3))
        cv2.rectangle(
            image, (x, y), (x + mark_width, y + mark_height), colour, -1
        )
        cv2.rectangle(
            image, (x, y), (x + mark_width, y + mark_height), (255, 255, 255), 1
        )
        cv2.line(
            image,
            (x + 2, y + 2),
            (x + mark_width - 3, y + mark_height - 3),
            tuple(int(value) for value in generator.integers(0, 256, 3)),
            2,
        )
        cv2.circle(
            image,
            (x + mark_width // 2, y + mark_height // 2),
            max(2, min(mark_width, mark_height) // 5),
            tuple(int(value) for value in generator.integers(0, 256, 3)),
            -1,
        )
    return image


def _shifted_view(image: np.ndarray, dx: float = 11.0, dy: float = -7.0) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 3.0, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _repeated_grid(width: int = 320, height: int = 240, tile: int = 40) -> np.ndarray:
    """One tile stamped over and over: the classic ambiguous Minecraft wall."""
    base = np.full((tile, tile, 3), 40, dtype=np.uint8)
    cv2.rectangle(base, (6, 6), (tile - 7, tile - 7), (200, 200, 200), 2)
    cv2.circle(base, (tile // 2, tile // 2), 7, (30, 30, 30), -1)
    cv2.line(base, (2, tile - 3), (tile - 3, 2), (120, 120, 120), 2)
    rows = height // tile + 2
    columns = width // tile + 2
    grid = np.tile(base, (rows, columns, 1))[:height, :width]
    return np.ascontiguousarray(grid)


def _one_to_one(matches: np.ndarray) -> bool:
    left = {(round(float(x), 6), round(float(y), 6)) for x, y in matches[:, 0:2]}
    right = {(round(float(x), 6), round(float(y), 6)) for x, y in matches[:, 2:4]}
    return len(left) == len(matches) and len(right) == len(matches)


# --- pure filter ------------------------------------------------------------


def _tables():
    left_descriptors = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.05, 0.0),
            (0.7, 0.7, 0.0),
        )
    )
    right_descriptors = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.9, 0.1, 0.0),
        )
    )
    left_points = np.asarray(
        ((10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (11.0, 21.0), (70.0, 80.0))
    )
    right_points = np.asarray(
        ((100.0, 200.0), (300.0, 400.0), (500.0, 600.0), (110.0, 210.0))
    )
    return left_points, right_points, left_descriptors, right_descriptors


def test_mutual_selection_keeps_reciprocal_ratio_survivors():
    left_points, right_points, left_descriptors, right_descriptors = _tables()
    selection = select_mutual_matches(
        left_points, right_points, left_descriptors, right_descriptors, ratio_test=0.75
    )
    # The near-duplicate left row 3 points at the same right feature as row 0, so
    # only row 0 is reciprocal; row 4 fails the ratio test in both directions.
    assert selection.ratio_left == 4
    assert selection.ratio_right == 3
    assert selection.mutual == 3
    expected = np.asarray(
        (
            (10.0, 20.0, 100.0, 200.0),
            (30.0, 40.0, 300.0, 400.0),
            (50.0, 60.0, 500.0, 600.0),
        )
    )
    np.testing.assert_array_equal(selection.matches, expected)
    assert selection.distance_min == pytest.approx(0.0)
    assert selection.distance_max == pytest.approx(0.0)


def test_mutual_selection_is_deterministic_and_one_to_one():
    left_points, right_points, left_descriptors, right_descriptors = _tables()
    first = select_mutual_matches(
        left_points, right_points, left_descriptors, right_descriptors, ratio_test=0.75
    )
    second = select_mutual_matches(
        left_points, right_points, left_descriptors, right_descriptors, ratio_test=0.75
    )
    np.testing.assert_array_equal(first.matches, second.matches)
    assert _one_to_one(first.matches)
    # A right feature with two equally close left rows is not reciprocal at all.
    ambiguous_left = np.asarray(((1.0, 0.0), (1.0, 0.0)))
    ambiguous_right = np.asarray(((1.0, 0.0),))
    selection = select_mutual_matches(
        np.asarray(((1.0, 1.0), (2.0, 2.0))),
        np.asarray(((9.0, 9.0),)),
        ambiguous_left,
        ambiguous_right,
        ratio_test=0.75,
    )
    assert selection.mutual == 0
    assert selection.matches.shape == (0, 4)


def test_non_finite_and_duplicate_pixel_pairs_are_removed():
    descriptors = np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    selection = select_mutual_matches(
        np.asarray(((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))),
        np.asarray(((10.0, 10.0), (np.nan, 5.0), (30.0, 30.0))),
        descriptors,
        descriptors,
        ratio_test=0.75,
    )
    assert selection.mutual == 3
    assert selection.discarded_non_finite == 1
    assert selection.matches.shape == (2, 4)
    assert np.all(np.isfinite(selection.matches))

    duplicates = select_mutual_matches(
        np.asarray(((1.0, 1.0), (1.0, 1.0))),
        np.asarray(((9.0, 9.0), (9.0, 9.0))),
        descriptors[:2],
        descriptors[:2],
        ratio_test=0.75,
    )
    assert duplicates.mutual == 2
    assert duplicates.discarded_duplicate == 1
    assert duplicates.matches.shape == (1, 4)


# --- usable-mask authority --------------------------------------------------


def test_none_derives_the_production_gui_mask_per_frame():
    derived = usable_mask_for("frame.png", (100, 200), None)
    np.testing.assert_array_equal(derived, ~create_gui_mask(100, 200))

    left = _noise(200, 160, 1)
    right = _noise(200, 160, 2)
    automatic = extract_pair_matches("left.png", left, "right.png", right)
    explicit = extract_pair_matches(
        "left.png",
        left,
        "right.png",
        right,
        usable_masks={
            "left.png": np.ones((160, 200), dtype=bool),
            "right.png": np.ones((160, 200), dtype=bool),
        },
    )
    assert automatic.left.usable_pixels == int(np.count_nonzero(~create_gui_mask(160, 200)))
    assert explicit.left.usable_pixels == 160 * 200
    assert automatic.left.usable_pixels < explicit.left.usable_pixels


def test_missing_frame_is_unusable_and_wrong_shape_rejects():
    left = _noise(200, 160, 3)
    right = _noise(200, 160, 4)
    mask = np.ones((160, 200), dtype=bool)
    missing = extract_pair_matches(
        "left.png", left, "right.png", right, usable_masks={"right.png": mask}
    )
    assert missing.status == "unusable_frame"
    assert missing.matches.shape == (0, 4)

    with pytest.raises(ValueError):
        extract_pair_matches(
            "left.png",
            left,
            "right.png",
            right,
            usable_masks={"left.png": np.ones((5, 5), dtype=bool), "right.png": mask},
        )


# --- image controls ---------------------------------------------------------


def test_excluded_rgb_cannot_change_extracted_matches():
    left = _noise(220, 180, 11)
    right = _noise(220, 180, 12)
    usable = np.zeros((180, 220), dtype=bool)
    usable[20:160, 20:200] = True
    masks = {"left.png": usable, "right.png": usable}
    baseline = extract_pair_matches(
        "left.png", left, "right.png", right, usable_masks=masks
    )
    assert baseline.matches.shape[0] > 0

    mutated_left = left.copy()
    mutated_left[~usable] = 255
    mutated_right = right.copy()
    mutated_right[~usable] = 0
    changed = extract_pair_matches(
        "left.png", mutated_left, "right.png", mutated_right, usable_masks=masks
    )
    np.testing.assert_array_equal(baseline.matches, changed.matches)
    assert baseline.status == changed.status
    assert baseline.selection.mutual == changed.selection.mutual


def test_unique_geometric_marks_yield_a_feedable_pair():
    image = _marks()
    view = _shifted_view(image)
    result = extract_pair_matches(
        "a.png",
        image,
        "b.png",
        view,
        usable_masks={
            "a.png": np.ones(image.shape[:2], dtype=bool),
            "b.png": np.ones(view.shape[:2], dtype=bool),
        },
    )
    assert result.status == "ok", result.reason
    assert result.matches.shape[0] >= CorrespondenceSettings().min_matches
    assert _one_to_one(result.matches)
    assert result.selection.distance_median is not None
    repeated = extract_pair_matches(
        "a.png",
        image,
        "b.png",
        view,
        usable_masks={
            "a.png": np.ones(image.shape[:2], dtype=bool),
            "b.png": np.ones(view.shape[:2], dtype=bool),
        },
    )
    np.testing.assert_array_equal(result.matches, repeated.matches)


def test_repeated_grid_is_flagged_instead_of_confidently_matched():
    image = _repeated_grid()
    height, width = image.shape[:2]
    matrix = np.float64([[1.0, 0.0, 13.0], [0.0, 1.0, 7.0]])
    view = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    result = extract_pair_matches(
        "grid.png",
        image,
        "grid-shift.png",
        view,
        usable_masks={
            "grid.png": np.ones((height, width), dtype=bool),
            "grid-shift.png": np.ones((height, width), dtype=bool),
        },
    )
    assert result.status in {"insufficient_matches", "repetition_risk"}, result.status
    assert result.status != "ok"


# --- DS-14 additive helpers: reciprocal top-k graph and boundary filtering ---


def test_reciprocal_topk_graph_is_exact_capped_and_deterministic():
    left = np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    right = np.asarray(((1.0, 0.0), (0.0, 1.0)))

    first = reciprocal_topk_graph(left, right, neighbors=2)
    second = reciprocal_topk_graph(left, right, neighbors=2)

    np.testing.assert_array_equal(first.edges, second.edges)
    np.testing.assert_array_equal(first.distances, second.distances)
    np.testing.assert_array_equal(
        first.edges,
        np.asarray(((0, 0), (1, 1), (2, 0), (2, 1))),
    )
    np.testing.assert_allclose(first.distances, (0.0, 0.0, 1.0, 1.0))
    assert first.total_edges == 4 and first.truncated == 0
    assert first.left_keypoints == 3 and first.right_keypoints == 2

    capped = reciprocal_topk_graph(left, right, neighbors=2, max_edges=2)
    np.testing.assert_array_equal(capped.edges, first.edges[:2])
    assert capped.total_edges == 4
    assert capped.truncated == 2

    ceiling = reciprocal_topk_graph(left, right, neighbors=2, distance_ceiling=0.5)
    np.testing.assert_array_equal(ceiling.edges, np.asarray(((0, 0), (1, 1))))
    assert ceiling.total_edges == 2 and ceiling.truncated == 0


def test_descriptor_footprint_keep_rejects_only_boundary_keypoints():
    usable = np.ones((60, 80), dtype=bool)
    usable[:, :10] = False
    points = np.asarray(((30.0, 30.0), (12.0, 30.0), (11.0, 30.0)))
    sizes = np.asarray((4.0, 4.0, 4.0))

    kept = descriptor_footprint_keep(points, sizes, usable, pad_px=4.0)
    assert kept.tolist() == [True, False, False]
    # Zero pad is the frozen DS-12 path: every keypoint is kept.
    assert descriptor_footprint_keep(points, sizes, usable, pad_px=0.0).all()


def test_detect_features_reports_boundary_dropped_only_when_asked():
    generator = np.random.default_rng(21)
    image = generator.integers(0, 256, size=(160, 200, 3), dtype=np.uint8)
    usable = np.ones((160, 200), dtype=bool)
    usable[:, :30] = False

    frozen = detect_features("a.png", image, usable_mask=usable)
    assert frozen.boundary_dropped == 0
    assert frozen.sizes is not None and len(frozen.sizes) == len(frozen.points)

    padded = detect_features(
        "a.png",
        image,
        usable_mask=usable,
        settings=CorrespondenceSettings(mask_boundary_pad_px=10.0),
    )
    assert padded.boundary_dropped > 0
    assert len(padded.points) == len(frozen.points) - padded.boundary_dropped
    assert len(padded.sizes) == len(padded.points)
    # The filter only shrinks the domain: every surviving footprint stays inside U.
    assert np.all(padded.points[:, 0] > 30.0)
