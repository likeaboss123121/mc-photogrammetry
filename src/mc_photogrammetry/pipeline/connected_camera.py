"""Bounded analytic two-view pair pose from image-only camera estimates.

DS-11's module: the smallest estimator that turns two single-view
:class:`~mc_photogrammetry.pipeline.contracts.CameraEstimate` values plus pixel
correspondences into a *relative* rotation and a *world* baseline direction. It
consumes DS-9's per-view estimates. It does not calibrate, match features, build
components, recover scale, or read a manifest, lattice, or any asserted
metadata.

Repository conventions used throughout (derived, not assumed):

* ``camera.camera_rotation_matrix`` returns the world-to-camera matrix ``R``
  whose rows are (right, up, forward), so a camera-space point is
  ``x = R (X - C)`` and the pixel ray is
  ``u = normalize(((px - cx)/fx, -(py - cy)/fy, 1))`` in that
  screen-right/screen-up/positive-depth frame (``camera.py:255-298``).
* Writing the pair as ``P_l = [I | 0]`` and ``P_r = [R | t]``:

      R = R_r R_l^T,      t = R_r (C_l - C_r) = -R_r (C_r - C_l).

  ``t`` is the baseline expressed in the **right** camera frame, pointing from
  the right eye towards the left eye. The world direction from the left eye to
  the right eye is ``b = C_r - C_l = -R_r^T t``.
* Calibration recovers yaw only modulo 90 degrees. Here
  ``R(yaw) = Rz(roll) D Rx(-pitch) Ry(yaw)`` with ``D = diag(-1, 1, 1)``, so a
  90-degree relabelling of the world is a **right** multiplication:

      R(yaw + 90 k) = R(yaw) @ Ry(90 k),      k in {0, 1, 2, 3}.

  ``Ry(90 k) @ R`` would be ``R(yaw - 90 k)`` and is a different, wrong
  candidate set.

Epipolar constraint. For a correspondence with rays ``u`` (left) and ``u'``
(right), ``x' = R x + t`` and coplanarity give ``u'^T [t]_x R u = 0``, which is
linear in ``t``:

    a(u, u')^T t = 0,      a = (R u) x u'.                              (1)

Two non-parallel rows already fix ``t`` up to sign; ``t`` is the unit
right-singular vector of the stacked ``A``. Robustness comes from deterministic
seeded RANSAC over minimal samples plus an SVD refit on the inliers. The per-row
residual ``|a^T t| / ||a||`` (radians) is reported as ``median_rho_degrees``.

Yaw quadrant. ``R`` depends on the right view's class ``k``, so the class cannot
be scored against a fixed rotation: all four candidates are solved and gated,
the left view stays anchored at ``k = 0`` (a gauge, not a measurement), and a
pair is accepted only when the winner beats the runner-up class by a residual
margin. Otherwise it is unresolved.

Sign. ``t`` is homogeneous, so only its sign is missing. The sign chosen is the
one that triangulates more inliers in front of *both* cameras under a unit
baseline; ``cheirality_fraction`` records the winner's positive-depth fraction.

Everything the module cannot establish is reported as ``solved=False`` with a
typed ``reason`` and no vector - never a fabricated direction. It does not round
block IDs, inspect a lattice, or consult metadata: before scale and frame
recovery those quantities do not exist, so repetition rejection belongs to a
later stage (DS-10 correction C3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraIntrinsics, CameraPose
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix

__all__ = [
    "PairGates",
    "PairPose",
    "YawClassScore",
    "REFUSAL_REASONS",
    "solve_pair",
]

_TINY = 1e-12
_ZERO_ROW = 1e-9
_YAW_CLASSES: tuple[int, ...] = (0, 1, 2, 3)

REFUSAL_REASONS: tuple[str, ...] = (
    "missing_pose",
    "missing_intrinsics",
    "invalid_intrinsics",
    "invalid_pose",
    "invalid_matches",
    "non_finite_input",
    "insufficient_correspondences",
    "rank_deficient",
    "pure_rotation",
    "insufficient_inliers",
    "low_inlier_fraction",
    "high_residual",
    "cheirality_failed",
    "no_yaw_class_passes_gates",
    "ambiguous_yaw_class",
)
"""Every typed reason :func:`solve_pair` may return for ``solved=False``."""


@dataclass(frozen=True, slots=True)
class PairGates:
    """Bounds a pair must clear before a direction is reported.

    Defaults are deliberately strict: a pair that cannot show at least eight
    correspondences, eight inliers, half the matches agreeing within one degree,
    three degrees of median parallax, and a clear residual win over the
    runner-up quadrant stays unresolved instead of returning a plausible vector.
    """

    min_correspondences: int = 8
    inlier_residual_degrees: float = 1.0
    min_inliers: int = 8
    min_inlier_fraction: float = 0.5
    max_median_residual_degrees: float = 1.0
    min_parallax_degrees: float = 3.0
    min_cheirality_fraction: float = 0.75
    min_singular_ratio: float = 1e-3
    residual_margin_degrees: float = 0.25
    ransac_seed: int = 0
    ransac_iterations: int = 512
    ransac_sample_size: int = 5


@dataclass(frozen=True, slots=True)
class YawClassScore:
    """Diagnostics for one yaw-quadrant hypothesis of the right view."""

    yaw_class: int
    passed: bool
    refusal: str
    inliers: int
    informative: int
    inlier_fraction: float
    median_rho_degrees: float | None
    parallax_degrees: float | None
    cheirality_fraction: float | None
    singular_ratio: float | None
    singular_values: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class PairPose:
    """Public pair result: solved with a rotation and direction, or unresolved.

    ``baseline_direction`` is the unit world vector ``b = C_r - C_l`` (left eye
    to right eye) in the anchor frame declared by the left estimate; the left
    view's own yaw quadrant stays a gauge. When ``solved`` is false, every
    vector field is ``None`` and ``reason`` is one of :data:`REFUSAL_REASONS`;
    ``classes`` carries the per-quadrant diagnostics once the four candidates
    have been solved, and is empty for an input fault that stops before that.
    """

    left_id: str
    right_id: str
    correspondences: int
    solved: bool
    reason: str
    right_yaw_class: int | None
    relative_rotation: np.ndarray | None
    baseline_direction: tuple[float, float, float] | None
    inliers: int
    inlier_fraction: float
    median_rho_degrees: float | None
    parallax_degrees: float | None
    cheirality_fraction: float | None
    classes: tuple[YawClassScore, ...]


@dataclass(frozen=True, slots=True)
class _Fit:
    """A robust homogeneous solution for one yaw class."""

    t: np.ndarray | None
    inliers: np.ndarray
    informative: np.ndarray
    median_rho_degrees: float | None
    singular_values: tuple[float, float, float] | None
    singular_ratio: float | None
    row_max: float
    informative_count: int
    inlier_count: int
    total: int


def solve_pair(
    left: CameraEstimate,
    right: CameraEstimate,
    matches: "np.ndarray | list | tuple",
    *,
    left_id: str = "left",
    right_id: str = "right",
    gates: PairGates | None = None,
) -> PairPose:
    """Resolve the right view's yaw quadrant and the world baseline direction.

    ``matches`` is ``N x 4`` with rows ``(left_x, left_y, right_x, right_y)``
    (``N x 2 x 2`` is accepted as the same data). Both estimates must carry a
    pose and explicit intrinsics; each view's own focal length and principal
    point are used, and the two image sizes may differ.
    """
    bounds = PairGates() if gates is None else gates
    parsed = _parse_matches(matches)
    correspondences = 0 if parsed is None else int(parsed.shape[0])
    if parsed is None:
        return _unresolved("invalid_matches", left_id, right_id, correspondences)
    if not np.all(np.isfinite(parsed)):
        return _unresolved("non_finite_input", left_id, right_id, correspondences)
    if left.pose is None or right.pose is None:
        return _unresolved("missing_pose", left_id, right_id, correspondences)
    if left.intrinsics is None or right.intrinsics is None:
        return _unresolved("missing_intrinsics", left_id, right_id, correspondences)
    if not _intrinsics_ok(left.intrinsics) or not _intrinsics_ok(right.intrinsics):
        return _unresolved("invalid_intrinsics", left_id, right_id, correspondences)
    if correspondences < bounds.min_correspondences:
        return _unresolved("insufficient_correspondences", left_id, right_id, correspondences)

    rotation_left = _world_to_camera(left.pose)
    rotation_right = _world_to_camera(right.pose)
    if rotation_left is None or rotation_right is None:
        return _unresolved("invalid_pose", left_id, right_id, correspondences)

    rays_left = _camera_rays(parsed[:, 0:2], left.intrinsics)
    rays_right = _camera_rays(parsed[:, 2:4], right.intrinsics)
    world_left = rays_left @ rotation_left

    scores: list[YawClassScore] = []
    models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for yaw_class in _YAW_CLASSES:
        rotation_right_class = rotation_right @ _ry90(yaw_class)
        relative = rotation_right_class @ rotation_left.T
        left_in_right = rays_left @ relative.T
        a_rows = np.cross(left_in_right, rays_right)
        fit = _robust_baseline(a_rows, bounds)
        world_right = rays_right @ rotation_right_class
        mask = fit.inliers

        parallax = (
            _median_angle_degrees(world_left[mask], world_right[mask])
            if np.any(mask)
            else None
        )
        signed_t: np.ndarray | None = None
        baseline: np.ndarray | None = None
        cheirality: float | None = None
        if fit.t is not None and np.any(mask):
            candidate = -(rotation_right_class.T @ fit.t)
            positive = _positive_depth_fraction(world_left[mask], world_right[mask], candidate)
            negative = _positive_depth_fraction(world_left[mask], world_right[mask], -candidate)
            if negative > positive:
                signed_t, baseline, cheirality = -fit.t, -candidate, negative
            else:
                signed_t, baseline, cheirality = fit.t, candidate, positive

        refusal = _class_refusal(fit, parallax, cheirality, bounds)
        scores.append(
            YawClassScore(
                yaw_class=yaw_class,
                passed=refusal == "",
                refusal=refusal,
                inliers=fit.inlier_count,
                informative=fit.informative_count,
                inlier_fraction=fit.inlier_count / correspondences,
                median_rho_degrees=fit.median_rho_degrees,
                parallax_degrees=parallax,
                cheirality_fraction=cheirality,
                singular_ratio=fit.singular_ratio,
                singular_values=fit.singular_values,
            )
        )
        if signed_t is not None and baseline is not None:
            models[yaw_class] = (signed_t, baseline)

    passing = [score for score in scores if score.passed]
    if any(score.informative == 0 for score in scores):
        return _unresolved("pure_rotation", left_id, right_id, correspondences, tuple(scores))
    if not passing:
        best_refused = min(scores, key=_refusal_order, default=None)
        reason = "no_yaw_class_passes_gates" if best_refused is None else best_refused.refusal
        return _unresolved(reason, left_id, right_id, correspondences, tuple(scores))

    best = min(passing, key=lambda score: (score.median_rho_degrees, score.yaw_class))
    competitors = [
        score
        for score in scores
        if score.yaw_class != best.yaw_class
        and score.inliers >= bounds.min_inliers
        and score.median_rho_degrees is not None
    ]
    if competitors:
        runner_up = min(competitors, key=lambda score: (score.median_rho_degrees, score.yaw_class))
        if (
            best.median_rho_degrees is not None
            and runner_up.median_rho_degrees is not None
            and runner_up.median_rho_degrees - best.median_rho_degrees
            < bounds.residual_margin_degrees
        ):
            return _unresolved(
                "ambiguous_yaw_class", left_id, right_id, correspondences, tuple(scores)
            )

    _signed_t, baseline = models[best.yaw_class]
    rotation_right_class = rotation_right @ _ry90(best.yaw_class)
    return PairPose(
        left_id=left_id,
        right_id=right_id,
        correspondences=correspondences,
        solved=True,
        reason="",
        right_yaw_class=best.yaw_class,
        relative_rotation=rotation_right_class @ rotation_left.T,
        baseline_direction=(float(baseline[0]), float(baseline[1]), float(baseline[2])),
        inliers=best.inliers,
        inlier_fraction=best.inlier_fraction,
        median_rho_degrees=best.median_rho_degrees,
        parallax_degrees=best.parallax_degrees,
        cheirality_fraction=best.cheirality_fraction,
        classes=tuple(scores),
    )


def _ry90(yaw_class: int) -> np.ndarray:
    """``Ry(90 * yaw_class)``: the world relabelling for one quadrant."""
    angle = math.radians(90.0 * (yaw_class % 4))
    cos, sin = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos)), dtype=np.float64
    )


def _world_to_camera(pose: CameraPose) -> np.ndarray | None:
    if not _finite(pose.yaw_degrees, pose.pitch_degrees, pose.roll_degrees):
        return None
    try:
        return camera_rotation_matrix(
            CameraModel(
                position=(0.0, 0.0, 0.0),
                yaw_degrees=float(pose.yaw_degrees),
                pitch_degrees=float(pose.pitch_degrees),
                roll_degrees=float(pose.roll_degrees),
            )
        )
    except ValueError:
        return None


def _camera_rays(pixels: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Unit camera-space rays for pixels, in the image-up convention."""
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    x = (pixels[:, 0] - centre_x) / focal_x
    y = -(pixels[:, 1] - centre_y) / focal_y
    rays = np.column_stack((x, y, np.ones(len(pixels), dtype=np.float64)))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _robust_baseline(a_rows: np.ndarray, bounds: PairGates) -> _Fit:
    """Deterministic seeded RANSAC over ``a^T t = 0`` rows, then an SVD refit."""
    norms = np.linalg.norm(a_rows, axis=1)
    informative = norms > _ZERO_ROW
    informative_count = int(np.count_nonzero(informative))
    row_max = float(norms.max()) if norms.size else 0.0
    total = int(len(a_rows))
    blank = _Fit(None, np.zeros(total, dtype=bool), informative, None, None, None,
                 row_max, informative_count, 0, total)
    if informative_count < 3:
        return blank

    rows = a_rows[informative]

    def evaluate(candidate: np.ndarray) -> tuple[int, float, np.ndarray]:
        residual = _residuals(rows, candidate)
        mask = residual <= bounds.inlier_residual_degrees
        count = int(np.count_nonzero(mask))
        median = float(np.median(residual[mask])) if count else float("inf")
        return count, median, mask

    candidates = [_null_vector(rows)]
    if bounds.ransac_iterations > 0:
        generator = np.random.default_rng(int(bounds.ransac_seed))
        sample_size = max(3, min(int(bounds.ransac_sample_size), informative_count))
        for _ in range(int(bounds.ransac_iterations)):
            indices = generator.choice(informative_count, size=sample_size, replace=False)
            candidates.append(_null_vector(rows[indices]))

    best_t: np.ndarray | None = None
    best_count, best_median, best_mask = -1, float("inf"), None
    for candidate in candidates:
        if candidate is None:
            continue
        count, median, mask = evaluate(candidate)
        if count > best_count or (count == best_count and median < best_median):
            best_t, best_count, best_median, best_mask = candidate, count, median, mask
    if best_t is None or best_mask is None or best_count == 0:
        return blank

    t, mask, median = best_t, best_mask, best_median
    for _ in range(10):
        candidate = _null_vector(rows[mask])
        if candidate is None:
            break
        count, candidate_median, candidate_mask = evaluate(candidate)
        if count == 0:
            break
        converged = np.array_equal(candidate_mask, mask)
        t, mask, median = candidate, candidate_mask, candidate_median
        if converged:
            break

    inliers = np.zeros(total, dtype=bool)
    inliers[informative] = mask
    support = rows[mask] if np.any(mask) else rows
    singular_values = _singular_values(support)
    ratio = singular_values[1] / singular_values[0] if singular_values[0] > _TINY else 0.0
    # Score on every informative correspondence, not just the model's own
    # support: a wrong quadrant can always shrink its median by dropping matches,
    # so class comparison and the ambiguity margin must use the same rows.
    full_median = float(np.median(_residuals(rows, t)))
    return _Fit(t, inliers, informative, full_median, singular_values, float(ratio),
                row_max, informative_count, int(np.count_nonzero(inliers)), total)


def _null_vector(rows: np.ndarray) -> np.ndarray | None:
    """Unit right-singular vector of the smallest singular value (the null space)."""
    if rows.size == 0:
        return None
    _u, _s, vt = np.linalg.svd(rows, full_matrices=True)
    vector = vt[-1]
    norm = float(np.linalg.norm(vector))
    if norm <= _TINY:
        return None
    return vector / norm


def _singular_values(rows: np.ndarray) -> tuple[float, float, float]:
    if rows.size == 0:
        return (0.0, 0.0, 0.0)
    _u, singular, _vt = np.linalg.svd(rows, full_matrices=True)
    padded = np.zeros(3, dtype=np.float64)
    padded[: min(3, len(singular))] = singular[:3]
    return (float(padded[0]), float(padded[1]), float(padded[2]))


def _residuals(rows: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``|a^T t| / ||a||`` in degrees; a degenerate zero row scores worst."""
    projections = np.abs(rows @ t)
    norms = np.linalg.norm(rows, axis=1)
    values = np.ones(len(rows), dtype=np.float64)
    safe = norms > _TINY
    values[safe] = np.clip(projections[safe] / norms[safe], 0.0, 1.0)
    return np.degrees(np.arcsin(values))


def _median_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    dots = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(dots))))


def _positive_depth_fraction(
    world_left: np.ndarray, world_right: np.ndarray, baseline: np.ndarray
) -> float:
    """Fraction of rays whose unit-baseline triangulation is in front of both views.

    With ``C_l = 0`` and ``C_r = baseline``, the point on the left ray is
    ``lambda_l * d_l = baseline + lambda_r * d_r``; the normal equations give a
    closed 2x2 solve per correspondence. A wrong baseline sign flips both
    depths, so the fraction separates the two signs.
    """
    if len(world_left) == 0:
        return 0.0
    cosine = np.clip(np.sum(world_left * world_right, axis=1), -1.0, 1.0)
    determinant = 1.0 - cosine * cosine
    safe = determinant > 1e-12
    if not np.any(safe):
        return 0.0
    first = world_left @ baseline
    second = -(world_right @ baseline)
    lambda_left = (first[safe] + cosine[safe] * second[safe]) / determinant[safe]
    lambda_right = (second[safe] + cosine[safe] * first[safe]) / determinant[safe]
    positive = int(np.count_nonzero((lambda_left > 0.0) & (lambda_right > 0.0)))
    return positive / float(len(world_left))


def _refusal_order(score: YawClassScore) -> tuple[bool, float, int]:
    """Rank classes for the unresolved reason: finite fits first, then class."""
    median = score.median_rho_degrees
    return (median is None, 0.0 if median is None else median, score.yaw_class)


def _class_refusal(
    fit: _Fit,
    parallax: float | None,
    cheirality: float | None,
    bounds: PairGates,
) -> str:
    """First gate this class fails, as a typed token; empty when it passes."""
    if fit.row_max <= _ZERO_ROW or fit.informative_count < 2:
        return "pure_rotation"
    if fit.t is None:
        return "rank_deficient"
    if fit.singular_ratio is None or fit.singular_ratio < bounds.min_singular_ratio:
        return "rank_deficient"
    if parallax is None or parallax < bounds.min_parallax_degrees:
        return "pure_rotation"
    if fit.inlier_count < bounds.min_inliers:
        return "insufficient_inliers"
    if fit.total == 0 or fit.inlier_count / fit.total < bounds.min_inlier_fraction:
        return "low_inlier_fraction"
    if fit.median_rho_degrees is None or fit.median_rho_degrees > bounds.max_median_residual_degrees:
        return "high_residual"
    if cheirality is None or cheirality < bounds.min_cheirality_fraction:
        return "cheirality_failed"
    return ""


def _parse_matches(raw: object) -> np.ndarray | None:
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim == 3 and array.shape[1:] == (2, 2):
        array = array.reshape(-1, 4)
    if array.ndim != 2 or array.shape[1] != 4 or array.shape[0] == 0:
        return None
    return array


def _intrinsics_ok(intrinsics: CameraIntrinsics) -> bool:
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    width, height = intrinsics.image_size
    if not _finite(focal_x, focal_y, centre_x, centre_y):
        return False
    return focal_x > 0.0 and focal_y > 0.0 and width > 0 and height > 0


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _unresolved(
    reason: str,
    left_id: str,
    right_id: str,
    correspondences: int,
    classes: tuple[YawClassScore, ...] = (),
) -> PairPose:
    """An unresolved pair result: a typed reason and diagnostics, never a vector."""
    return PairPose(
        left_id=left_id,
        right_id=right_id,
        correspondences=correspondences,
        solved=False,
        reason=reason,
        right_yaw_class=None,
        relative_rotation=None,
        baseline_direction=None,
        inliers=0,
        inlier_fraction=0.0,
        median_rho_degrees=None,
        parallax_degrees=None,
        cheirality_fraction=None,
        classes=classes,
    )
