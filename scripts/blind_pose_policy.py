"""DS-23 - label-blind yaw-class selection with the DS-19 support tie-break.

This module is the **refactor of the DS-19 support tie-break behind an API
boundary**. DS-19's decision rule lives in
``scripts/diagnose_oracle_pose_failures.py``:

* ``class_rows``  -> per-yaw-class ``a = d_l x d_r`` rows from camera intrinsics,
* ``fit_class``   -> bounded Gram/SVD consensus fit (no RANSAC, no subsample),
* ``choose_sign`` -> baseline sign by maximum depth positivity,
* ``class_hypothesis`` / ``_gate_check`` -> the unchanged DS-11 gate degrees,
* ``adjudicate``  -> consensus-count tie-break with an explicit margin.

Every one of those reads *only* production-available geometry: pixel
correspondences, camera intrinsics/orientation, the four yaw hypotheses, their
consensus counts and inlier masks, residual distributions, parallax,
cheirality, rank conditioning and spatial-stratum stability. The refactor here
re-expresses exactly that maths as a pure function of a single geometric input
so the boundary can be enforced and tested:

    select_blind_pose(BlindPairInput) -> BlindDecision

**Blindness boundary (enforced, not assumed).** Before a selection exists this
module may not touch, and :func:`select_blind_pose` cannot reach:

* fixture manifests / reference poses / reference yaw classes,
* litematics, scene transforms or rendered block labels,
* the frozen evaluator (``_evaluate_pose``) or any reference score,
* pair identity (frame names, fixture names, pair slots) or fixture constants,
* oracle/truth edge sets and their per-pair counts (evidence about which edges
  are correct is a reference label, so it must not enter the decision).

``BlindPairInput`` carries no identity field at all, and the runtime tracer
(:func:`start_trace` / :func:`stop_trace`) records every attribute the decision
reads, so a test can assert the read set is a subset of the geometric
whitelist. Reference evaluation may only run *after* the decisions are frozen
(this module never calls it; the DS-23 driver does).

Nothing in production is imported by :func:`select_blind_pose`: only ``numpy``
and module-level constants of this file. This is a *clean-room* reimplementation
verified numerically identical to DS-19's own functions by
``tests/test_blind_pose_policy.py`` on synthetic geometry and on the nine
frozen pairs; ``diagnose_oracle_pose_failures.py`` itself is not edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DecisionTrace",
    "BlindDecision",
    "BlindPairInput",
    "FROZEN_POLICY",
    "FORBIDDEN_DECISION_INPUTS",
    "GEOMETRIC_INPUT_WHITELIST",
    "FrozenPolicy",
    "select_blind_pose",
    "start_trace",
    "stop_trace",
    "to_blind_input",
]


# --------------------------------------------------------------------------- #
# Frozen policy constants (copied verbatim from DS-19's preregistered set).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FrozenPolicy:
    """The DS-19 adjudicator constants, frozen for DS-23.

    ``edge_*`` duplicate ``PairGates`` defaults exactly; the selection margins
    are DS-19's preregistered tie-break. No gate degree is changed here.
    """

    edge_consensus_degrees: float = 1.0
    edge_consensus_min: int = 8
    edge_consensus_fraction: float = 0.5
    edge_parallax_min_degrees: float = 3.0
    edge_cheirality_min: float = 0.75
    edge_rank_min_ratio: float = 1e-3
    class_margin_min: int = 3
    class_margin_fraction: float = 0.25
    stability_degrees: float = 5.0
    stratum_min_edges: int = 8
    spatial_grid: tuple[int, int] = (4, 4)


FROZEN_POLICY = FrozenPolicy()


# --------------------------------------------------------------------------- #
# The API boundary.
# --------------------------------------------------------------------------- #
class BlindPairInput:
    """Only production-available geometry. No identity, no labels, no scores.

    Parameters are stored shallowly; arrays are not copied and are treated as
    read-only. ``image_size`` is the pixel frame the correspondences live in
    (``(width, height)``), which image-plane binning needs; it is geometry, not
    a frame identity. ``width``/``height`` are exposed as properties.
    """

    __slots__ = (
        "left_intrinsics",
        "left_yaw_degrees",
        "left_pitch_degrees",
        "left_roll_degrees",
        "right_intrinsics",
        "right_yaw_degrees",
        "right_pitch_degrees",
        "right_roll_degrees",
        "matches",
        "image_size",
    )

    def __init__(
        self,
        *,
        left_intrinsics: Any,
        left_yaw_degrees: float,
        left_pitch_degrees: float,
        left_roll_degrees: float,
        right_intrinsics: Any,
        right_yaw_degrees: float,
        right_pitch_degrees: float,
        right_roll_degrees: float,
        matches: np.ndarray,
        image_size: tuple[int, int],
    ) -> None:
        self.left_intrinsics = left_intrinsics
        self.left_yaw_degrees = float(left_yaw_degrees)
        self.left_pitch_degrees = float(left_pitch_degrees)
        self.left_roll_degrees = float(left_roll_degrees)
        self.right_intrinsics = right_intrinsics
        self.right_yaw_degrees = float(right_yaw_degrees)
        self.right_pitch_degrees = float(right_pitch_degrees)
        self.right_roll_degrees = float(right_roll_degrees)
        self.matches = np.asarray(matches, dtype=np.float64).reshape(-1, 4)
        self.image_size = (int(image_size[0]), int(image_size[1]))

    def __getattribute__(self, name: str):
        # Audit hook: while a trace is open, record every geometric read so a
        # test can assert the decision never asked for a forbidden field (and
        # that forbidden fields do not exist in the first place).
        trace = _TRACE
        if trace is not None and trace._active:
            trace.record(self, name)
        return object.__getattribute__(self, name)

    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"BlindPairInput(correspondences={len(self.matches)}, "
            f"image_size={self.image_size})"
        )


#: Reads that must never be required by the selection function: reference
#: labels, identity, truth edges or evaluation output.
FORBIDDEN_DECISION_INPUTS = (
    "reference_class",
    "reference_class_floors",
    "expected_class",
    "manifest",
    "cameras",
    "litematic",
    "reference",
    "scene_transform",
    "truth",
    "oracle",
    "oracle_truths",
    "oracle_bestof_edges",
    "edges",
    "evaluation",
    "evaluator",
    "usable",
    "class_equals_expected",
    "within_5deg",
    "baseline_error_degrees_gauge_aligned",
    "baseline_error_degrees_signless",
    "gauge_index",
    "fixture",
    "pair",
    "pair_id",
    "left_id",
    "right_id",
    "left_frame",
    "right_frame",
    "frame",
    "frame_key",
    "left_points",
    "right_points",
    "candidate_stats",
    "sweep",
    "diagnosis",
)

#: The reads the decision is allowed to make on its geometric input.
GEOMETRIC_INPUT_WHITELIST = frozenset(
    {
        "left_intrinsics",
        "left_yaw_degrees",
        "left_pitch_degrees",
        "left_roll_degrees",
        "right_intrinsics",
        "right_yaw_degrees",
        "right_pitch_degrees",
        "right_roll_degrees",
        "matches",
        "image_size",
        "width",
        "height",
        "focal_length_pixels",
        "principal_point_pixels",
        "image_size_px",
    }
)


class DecisionTrace:
    """Records every attribute read during one selection (the audit hook)."""

    def __init__(self) -> None:
        self.reads: list[str] = []
        self._active = False
        self._targets: tuple[Any, ...] = ()

    def start(self, *targets: Any) -> None:
        self._targets = tuple(targets)
        self.reads = []
        self._active = True

    def stop(self) -> None:
        self._active = False

    def record(self, owner: Any, name: str) -> None:
        if not self._active:
            return
        if any(owner is target for target in self._targets):
            self.reads.append(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self.reads)


_TRACE: DecisionTrace | None = None


def start_trace(*targets: Any) -> DecisionTrace:
    """Begin recording attribute reads on ``targets``."""
    global _TRACE
    if _TRACE is None:
        _TRACE = DecisionTrace()
    _TRACE.start(*targets)
    return _TRACE


def stop_trace() -> DecisionTrace | None:
    """Stop recording and return the trace (or ``None`` if never started)."""
    if _TRACE is not None:
        _TRACE.stop()
    return _TRACE


class BlindDecision:
    """The frozen DS-19 decision, carried without any reference label.

    ``right_yaw_class`` is the geometric quadrant index the formula selected
    (``0..3``), not a correctness judgement. ``per_class`` holds the four
    hypothesis read-outs the report needs (consensus counts, residual
    distributions, parallax, cheirality, conditioning, spatial-stratum
    stability) - all geometry.
    """

    __slots__ = (
        "solved",
        "reason",
        "right_yaw_class",
        "baseline_direction",
        "inliers",
        "inlier_fraction",
        "median_rho_degrees",
        "parallax_degrees",
        "cheirality_fraction",
        "singular_ratio",
        "class_margin",
        "stability",
        "per_class",
    )

    def __init__(
        self,
        *,
        solved: bool,
        reason: str,
        right_yaw_class: int | None,
        baseline_direction: tuple[float, float, float] | None,
        inliers: int,
        inlier_fraction: float,
        median_rho_degrees: float | None,
        parallax_degrees: float | None,
        cheirality_fraction: float | None,
        singular_ratio: float | None,
        class_margin: int | None,
        stability: dict | None,
        per_class: dict[str, dict],
    ) -> None:
        self.solved = bool(solved)
        self.reason = str(reason)
        self.right_yaw_class = None if right_yaw_class is None else int(right_yaw_class)
        self.baseline_direction = baseline_direction
        self.inliers = int(inliers)
        self.inlier_fraction = float(inlier_fraction)
        self.median_rho_degrees = median_rho_degrees
        self.parallax_degrees = parallax_degrees
        self.cheirality_fraction = cheirality_fraction
        self.singular_ratio = singular_ratio
        self.class_margin = class_margin
        self.stability = stability
        self.per_class = per_class


# --------------------------------------------------------------------------- #
# Geometry primitives (numerically identical to DS-19's implementations).
# --------------------------------------------------------------------------- #
def _ry90(yaw_class: int) -> np.ndarray:
    angle = np.radians(90.0 * (yaw_class % 4))
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(
        ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos)), dtype=np.float64
    )


def _rays(pixels: np.ndarray, intrinsics) -> np.ndarray:
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    x = (pixels[:, 0] - centre_x) / focal_x
    y = -(pixels[:, 1] - centre_y) / focal_y
    rays = np.column_stack((x, y, np.ones(len(pixels), dtype=np.float64)))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _rotation_for_pose(camera_model) -> np.ndarray:
    """World->camera rotation from a position-free ``CameraModel``."""
    from mc_photogrammetry.reconstruction.camera import camera_rotation_matrix

    return camera_rotation_matrix(camera_model)


def _pose_model(yaw_degrees: float, pitch_degrees: float, roll_degrees: float):
    from mc_photogrammetry.reconstruction.camera import CameraModel

    return CameraModel(
        position=(0.0, 0.0, 0.0),
        yaw_degrees=float(yaw_degrees),
        pitch_degrees=float(pitch_degrees),
        roll_degrees=float(roll_degrees),
        eye_height=0.0,
    )


def _residual_degrees(rows: np.ndarray, direction: np.ndarray) -> np.ndarray:
    projections = np.abs(rows @ direction)
    norms = np.linalg.norm(rows, axis=1)
    values = np.ones(len(rows), dtype=np.float64)
    safe = norms > 1e-9
    values[safe] = np.clip(projections[safe] / norms[safe], 0.0, 1.0)
    return np.degrees(np.arcsin(values))


def _null_vector(rows: np.ndarray) -> np.ndarray | None:
    if rows.size == 0:
        return None
    _u, _s, vt = np.linalg.svd(rows, full_matrices=False)
    vector = vt[-1]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _singular_values(rows: np.ndarray) -> tuple[float, float, float]:
    if rows.size == 0:
        return (0.0, 0.0, 0.0)
    _u, singular, _vt = np.linalg.svd(rows, full_matrices=False)
    padded = np.zeros(3, dtype=np.float64)
    padded[: min(3, len(singular))] = singular[:3]
    return (float(padded[0]), float(padded[1]), float(padded[2]))


def _positive_depth_fraction(
    world_left: np.ndarray, world_right: np.ndarray, baseline: np.ndarray
) -> float:
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


def _median_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    dots = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(dots))))


def _class_rows(blind: BlindPairInput) -> dict:
    """Per-yaw-class ``a = d_l x d_r`` rows for the blind correspondence set."""
    from mc_photogrammetry.reconstruction.camera import camera_rotation_matrix

    rotation_left = camera_rotation_matrix(
        _pose_model(
            blind.left_yaw_degrees, blind.left_pitch_degrees, blind.left_roll_degrees
        )
    )
    rotation_right = camera_rotation_matrix(
        _pose_model(
            blind.right_yaw_degrees, blind.right_pitch_degrees, blind.right_roll_degrees
        )
    )
    rays_left = _rays(blind.matches[:, 0:2], blind.left_intrinsics)
    rays_right = _rays(blind.matches[:, 2:4], blind.right_intrinsics)
    world_left = rays_left @ rotation_left
    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for yaw_class in (0, 1, 2, 3):
        rotation_right_class = rotation_right @ _ry90(yaw_class)
        relative = rotation_right_class @ rotation_left.T
        left_in_right = rays_left @ relative.T
        rows[yaw_class] = (
            np.cross(left_in_right, rays_right),
            rays_right @ rotation_right_class,
        )
    return {"rays_left": rays_left, "world_left": world_left, "rows": rows}


def _fit_class(rows: np.ndarray, consensus_degrees: float) -> dict:
    """One yaw class: bounded Gram/SVD consensus fit, no RANSAC, no sampling."""
    norms = np.linalg.norm(rows, axis=1)
    informative = norms > 1e-9
    informative_count = int(np.count_nonzero(informative))
    if informative_count:
        direction = _null_vector(rows[informative])
    else:
        direction = None
    fit: dict = {
        "informative_count": informative_count,
        "row_max": float(norms.max()) if norms.size else 0.0,
        "direction": None,
        "consensus": np.zeros(0, dtype=np.int64),
        "consensus_count": 0,
        "consensus_fraction": 0.0,
        "median_rho_degrees": None,
        "median_all_residual_degrees": None,
        "singular_values": None,
        "singular_ratio": None,
        "row_scale_min": None,
        "row_scale_max": None,
    }
    if direction is None:
        return fit
    residual = _residual_degrees(rows, direction)
    consensus = np.flatnonzero(residual <= consensus_degrees)
    for _ in range(10):
        if len(consensus) == 0:
            break
        candidate = _null_vector(rows[consensus])
        if candidate is None:
            break
        refined = np.flatnonzero(_residual_degrees(rows, candidate) <= consensus_degrees)
        converged = len(refined) == len(consensus) and np.array_equal(refined, consensus)
        direction, consensus = candidate, refined
        if converged:
            break
    if len(consensus) == 0:
        return fit
    singular_values = _singular_values(rows[consensus])
    gram = rows[consensus].T @ rows[consensus]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if not np.all(np.isfinite(eigenvectors[:, 0])):
        return fit
    candidate = eigenvectors[:, 0]
    reference = _null_vector(rows[consensus])
    if reference is not None and float(np.dot(candidate, reference)) < 0.0:
        candidate = -candidate
    fit["direction"] = candidate
    fit["consensus"] = consensus
    fit["consensus_count"] = int(len(consensus))
    fit["consensus_fraction"] = len(consensus) / max(informative_count, 1)
    fit["median_rho_degrees"] = float(np.median(residual[consensus]))
    fit["median_all_residual_degrees"] = (
        float(np.median(residual[informative])) if informative_count else None
    )
    fit["singular_values"] = singular_values
    fit["singular_ratio"] = (
        singular_values[1] / singular_values[0] if singular_values[0] > 1e-12 else 0.0
    )
    scales = np.linalg.norm(rows[informative], axis=1)
    fit["row_scale_min"] = float(scales.min()) if len(scales) else None
    fit["row_scale_max"] = float(scales.max()) if len(scales) else None
    return fit


def _spatial_bins(
    points: np.ndarray, width: int, height: int, grid: tuple[int, int]
) -> np.ndarray:
    columns, rows = grid
    x = np.clip(
        (points[:, 0] / max(width, 1) * columns).astype(np.int64), 0, columns - 1
    )
    y = np.clip((points[:, 1] / max(height, 1) * rows).astype(np.int64), 0, rows - 1)
    return y * columns + x


def _stability_degrees(
    rows: np.ndarray,
    inliers: np.ndarray,
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
    width: int,
    height: int,
    grid: tuple[int, int],
    minimum: int,
    stability_bound: float,
) -> dict:
    """Largest direction disagreement between trustworthy spatial strata."""
    if len(inliers) == 0:
        return {
            "strata": 0,
            "usable_strata": 0,
            "pairwise_max_degrees": None,
            "available": False,
            "disagreeing_pairs": 0,
            "pairs": 0,
        }
    left_bins = _spatial_bins(left_pixels[inliers], width, height, grid)
    right_bins = _spatial_bins(right_pixels[inliers], width, height, grid)
    key = left_bins * (grid[0] * grid[1]) + right_bins
    strata: list[tuple[int, np.ndarray]] = []
    for value in np.unique(key):
        subset = inliers[key == value]
        if len(subset) >= 2:
            strata.append((int(value), subset))
    usable = [(value, subset) for value, subset in strata if len(subset) >= minimum]
    if len(usable) < 2:
        return {
            "strata": len(strata),
            "usable_strata": len(usable),
            "pairwise_max_degrees": None,
            "available": False,
            "disagreeing_pairs": 0,
            "pairs": 0,
        }
    directions: list[np.ndarray] = []
    for _value, subset in usable:
        vector = _null_vector(rows[subset])
        if vector is None:
            continue
        if directions and float(np.dot(vector, directions[0])) < 0.0:
            vector = -vector
        directions.append(vector)
    if len(directions) < 2:
        return {
            "strata": len(strata),
            "usable_strata": len(usable),
            "pairwise_max_degrees": None,
            "available": False,
            "disagreeing_pairs": 0,
            "pairs": 0,
        }
    worst = 0.0
    disagreeing = 0
    pairs = 0
    for first in range(len(directions)):
        for second in range(first + 1, len(directions)):
            value = float(
                np.degrees(
                    np.arccos(
                        float(
                            np.clip(
                                np.dot(directions[first], directions[second]), -1.0, 1.0
                            )
                        )
                    )
                )
            )
            worst = max(worst, value)
            pairs += 1
            if value > stability_bound:
                disagreeing += 1
    return {
        "strata": len(strata),
        "usable_strata": len(usable),
        "pairwise_max_degrees": worst,
        "available": True,
        "disagreeing_pairs": disagreeing,
        "pairs": pairs,
    }


def _choose_sign(
    rows_direction: np.ndarray,
    rotation_right_class: np.ndarray,
    world_left: np.ndarray,
    world_right: np.ndarray,
    inliers: np.ndarray,
) -> dict:
    """Pick the baseline sign by depth positivity; report both fractions."""
    candidate = -(rotation_right_class.T @ rows_direction)
    positive = _positive_depth_fraction(
        world_left[inliers], world_right[inliers], candidate
    )
    negative = _positive_depth_fraction(
        world_left[inliers], world_right[inliers], -candidate
    )
    if negative > positive:
        return {
            "baseline": -candidate,
            "cheirality_fraction": negative,
            "positive": positive,
            "negative": negative,
        }
    return {
        "baseline": candidate,
        "cheirality_fraction": positive,
        "positive": positive,
        "negative": negative,
    }


def _gate_check(hypothesis: dict, policy: FrozenPolicy) -> tuple[bool, str]:
    """The unchanged DS-11 class gates, applied to a blind hypothesis."""
    if hypothesis["row_max"] <= 1e-9 or hypothesis["informative_count"] < 2:
        return False, "pure_rotation"
    if (
        hypothesis["singular_ratio"] is None
        or hypothesis["singular_ratio"] < policy.edge_rank_min_ratio
    ):
        return False, "rank_deficient"
    if (
        hypothesis["parallax_degrees"] is None
        or hypothesis["parallax_degrees"] < policy.edge_parallax_min_degrees
    ):
        return False, "pure_rotation"
    if hypothesis["consensus_count"] < policy.edge_consensus_min:
        return False, "insufficient_inliers"
    if hypothesis["consensus_fraction"] < policy.edge_consensus_fraction:
        return False, "low_inlier_fraction"
    if (
        hypothesis["median_rho_degrees"] is None
        or hypothesis["median_rho_degrees"] > policy.edge_consensus_degrees
    ):
        return False, "high_residual"
    if (
        hypothesis["cheirality_fraction"] is None
        or hypothesis["cheirality_fraction"] < policy.edge_cheirality_min
    ):
        return False, "cheirality_failed"
    return True, ""


def _class_hypothesis(
    geometry: dict,
    yaw_class: int,
    blind: BlindPairInput,
    policy: FrozenPolicy,
) -> dict:
    """One complete blind yaw-class hypothesis, bounded and deterministic."""
    rows, world_right = geometry["rows"][yaw_class]
    fit = _fit_class(rows, policy.edge_consensus_degrees)
    record: dict = {
        "yaw_class": yaw_class,
        "informative_count": fit["informative_count"],
        "row_max": fit["row_max"],
        "row_scale_min": fit["row_scale_min"],
        "row_scale_max": fit["row_scale_max"],
        "consensus_count": fit["consensus_count"],
        "consensus_fraction": fit["consensus_fraction"],
        "median_rho_degrees": fit["median_rho_degrees"],
        "median_all_residual_degrees": fit["median_all_residual_degrees"],
        "singular_values": fit["singular_values"],
        "singular_ratio": fit["singular_ratio"],
        "consensus_singular_values": None,
        "rank_direction_ratio": None,
        "baseline": None,
        "baseline_direction": None,
        "parallax_degrees": None,
        "cheirality_fraction": None,
        "cheirality_positive": None,
        "cheirality_negative": None,
        "stability": None,
        "gate_pass": False,
        "gate_refusal": "rank_deficient",
    }
    if fit["direction"] is None or fit["consensus_count"] == 0:
        return record
    consensus = fit["consensus"]
    consensus_singular = _singular_values(rows[consensus])
    record["consensus_singular_values"] = consensus_singular
    record["rank_direction_ratio"] = (
        consensus_singular[2] / consensus_singular[0]
        if consensus_singular[0] > 1e-12
        else 0.0
    )
    rotation_right_class = _rotation_for_pose(
        _pose_model(
            blind.right_yaw_degrees,
            blind.right_pitch_degrees,
            blind.right_roll_degrees,
        )
    ) @ _ry90(yaw_class)
    sign = _choose_sign(
        fit["direction"],
        rotation_right_class,
        geometry["world_left"],
        world_right,
        consensus,
    )
    record["baseline"] = sign["baseline"]
    record["baseline_direction"] = sign["baseline"] / max(
        float(np.linalg.norm(sign["baseline"])), 1e-12
    )
    record["cheirality_fraction"] = sign["cheirality_fraction"]
    record["cheirality_positive"] = sign["positive"]
    record["cheirality_negative"] = sign["negative"]
    record["parallax_degrees"] = _median_angle_degrees(
        geometry["world_left"][consensus], world_right[consensus]
    )
    record["stability"] = _stability_degrees(
        rows,
        consensus,
        blind.matches[:, 0:2],
        blind.matches[:, 2:4],
        blind.width,
        blind.height,
        policy.spatial_grid,
        policy.stratum_min_edges,
        policy.stability_degrees,
    )
    record["gate_pass"], record["gate_refusal"] = _gate_check(record, policy)
    return record


def select_blind_pose(
    blind: BlindPairInput, policy: FrozenPolicy = FROZEN_POLICY
) -> BlindDecision:
    """Select a yaw class and baseline from geometry alone.

    This is DS-19's ``consensus_yaw_with_stratum_stability`` adjudicator behind
    the API boundary: same gates, same consensus fit, same support-margin
    tie-break. It reads only ``blind`` (whitelisted geometric attributes) and
    ``policy`` (frozen constants); the trace hook records the reads.
    """
    trace = _TRACE
    matches = blind.matches
    geometry = _class_rows(blind)
    hypotheses = {
        yaw_class: _class_hypothesis(geometry, yaw_class, blind, policy)
        for yaw_class in (0, 1, 2, 3)
    }
    solved = False
    reason = "no_yaw_class_passes_gates"
    right_yaw_class: int | None = None
    baseline_direction: tuple[float, float, float] | None = None
    inliers = 0
    inlier_fraction = 0.0
    median_rho_degrees = None
    parallax_degrees = None
    cheirality_fraction = None
    singular_ratio = None
    class_margin = None
    stability = None

    passing = [value for value in hypotheses.values() if value["gate_pass"]]
    if not passing:
        refused = sorted(
            hypotheses.values(),
            key=lambda item: (
                item["median_all_residual_degrees"] is None,
                0.0
                if item["median_all_residual_degrees"] is None
                else item["median_all_residual_degrees"],
                item["yaw_class"],
            ),
        )
        reason = refused[0]["gate_refusal"] or "no_yaw_class_passes_gates"
    else:
        winner = sorted(
            passing,
            key=lambda item: (
                -item["consensus_count"],
                item["median_rho_degrees"],
                item["yaw_class"],
            ),
        )[0]
        competitors = [
            value for value in passing if value["yaw_class"] != winner["yaw_class"]
        ]
        ambiguous = False
        if competitors:
            runner_up = sorted(
                competitors,
                key=lambda item: (
                    -item["consensus_count"],
                    item["median_rho_degrees"],
                    item["yaw_class"],
                ),
            )[0]
            margin = winner["consensus_count"] - runner_up["consensus_count"]
            required = max(
                policy.class_margin_min,
                policy.class_margin_fraction * winner["consensus_count"],
            )
            class_margin = int(margin)
            ambiguous = margin < required
        if ambiguous:
            reason = "ambiguous_yaw_class"
        else:
            solved = True
            reason = ""
            right_yaw_class = int(winner["yaw_class"])
            direction = winner["baseline_direction"]
            baseline_direction = tuple(float(value) for value in direction)
            inliers = int(winner["consensus_count"])
            inlier_fraction = float(winner["consensus_fraction"])
            median_rho_degrees = winner["median_rho_degrees"]
            parallax_degrees = winner["parallax_degrees"]
            cheirality_fraction = winner["cheirality_fraction"]
            singular_ratio = winner["singular_ratio"]
            stability = winner["stability"]
    return BlindDecision(
        solved=solved,
        reason=reason,
        right_yaw_class=right_yaw_class,
        baseline_direction=baseline_direction,
        inliers=inliers,
        inlier_fraction=inlier_fraction,
        median_rho_degrees=median_rho_degrees,
        parallax_degrees=parallax_degrees,
        cheirality_fraction=cheirality_fraction,
        singular_ratio=singular_ratio,
        class_margin=class_margin,
        stability=stability,
        per_class={str(key): value for key, value in hypotheses.items()},
    )


# --------------------------------------------------------------------------- #
# Adapter: assemble a BlindPairInput from geometric values only.
# --------------------------------------------------------------------------- #
def to_blind_input(
    *,
    left_intrinsics,
    left_pose,
    right_intrinsics,
    right_pose,
    matches: np.ndarray,
    image_size: tuple[int, int],
    width: int | None = None,
    height: int | None = None,
) -> BlindPairInput:
    """Build the boundary input from intrinsics, poses and pixel matches.

    Every parameter is production-available geometry. There is deliberately no
    parameter for a frame name, fixture, camera manifest, reference class,
    truth edge set or evaluator output: those cannot be passed through this
    adapter, so they cannot reach :func:`select_blind_pose`.
    """
    if left_pose is None or right_pose is None:
        raise ValueError("blind input requires both camera poses")
    if width is not None and height is not None:
        image_size = (int(width), int(height))
    return BlindPairInput(
        left_intrinsics=left_intrinsics,
        left_yaw_degrees=float(left_pose.yaw_degrees),
        left_pitch_degrees=float(left_pose.pitch_degrees),
        left_roll_degrees=float(left_pose.roll_degrees),
        right_intrinsics=right_intrinsics,
        right_yaw_degrees=float(right_pose.yaw_degrees),
        right_pitch_degrees=float(right_pose.pitch_degrees),
        right_roll_degrees=float(right_pose.roll_degrees),
        matches=matches,
        image_size=image_size,
    )
