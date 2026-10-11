"""Evaluation-only per-pair correspondence scorecard for finalized captures.

DS-R5I. This module lives outside reconstruction and the production CLI. It reads
finalized dataset-factory artifacts (``capture.json``, ``ground_truth.json`` and
the hash-pinned screenshots) plus the existing production feature/matcher
interfaces, and reports how often a mutual SIFT match links two pixels that
observe the *same world surface point*. It changes no production code, threshold,
matcher, solver, project or promotion decision.

The truth model is evaluation-only and deliberately strict:

* The registered ground truth (exact camera pose and exact region block states)
  is the world model. The camera is rebuilt from the recorded *actual* eye
  position and yaw/pitch in ``capture.json`` rather than the planned pose, so the
  projection is the pose the client reported, not the one it was asked for.
* Visibility is resolved per query with a front-to-back axis-aligned voxel
  traversal ("sparse z-buffer"). For a pixel, the first occupied cell along the
  ray owns the surface; the entry face gives the surface slot and the entry point
  gives the world-space surface point.
* A *strict positive* pair is two keypoints, one in each frame, whose owned
  surfaces are the same block face and whose world surface points agree within
  :data:`SURFACE_TOLERANCE_BLOCKS` (one vanilla texel, 1/16 block). Both are
  visible by construction, so a strict positive is a jointly visible
  same-surface correspondence.
* A *same-state/same-slot/other-block* pair is an **appearance proxy**, not a
  verified texture match: the two endpoints observe the same exact
  :class:`~mc_photogrammetry.domain.models.BlockState` on the same face slot but
  at two different block coordinates. No block texture is read, hashed or
  compared anywhere in this module, so the bucket makes no texture-identity
  claim. It is never counted as a positive.

Both the returned identity metric and the truth denominator are keyed by the
same *unique pixel-pair identity*: repeated descriptor keypoints at exactly the
same pixel are one identity, while distinct pixels stay distinct. Every returned
row's identity is read from the detected keypoint point arrays, including rows
with no truth surface, so unattributed rows are deduplicated by pixel pair too.

Accuracy is reported two ways: pair precision over returned match identities and
pair recall over the explicitly constructed truth-visible keypoint-pair
denominator. Undefined denominators are reported as ``None``/``null`` with a
typed reason, never as zero.

Renderer limitation: partial block models (slabs of type ``bottom``/``top``,
stairs, trapdoors, fences, walls, panes, plants, fluids, ...) are not available
without the vanilla model assets. They still occlude as full cubes but are
excluded from truth ownership, so surfaces behind them leave the denominator
instead of being mislabelled. ``_slab`` states with ``type=double`` are full
cubes and stay in. Every frame observation reports how many queried keypoints hit
such an excluded owner.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.dataset.blocks import parse_ground_truth, volume_from_ground_truth
from mc_photogrammetry.dataset.hashing import canonical_json, sha256_file
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.visibility import ray_directions
from mc_photogrammetry.pipeline.correspondence import (
    CorrespondenceSettings,
    FrameFeatures,
    detect_features,
    select_mutual_matches,
    usable_mask_for,
)
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_eye

__all__ = [
    "CORRESPONDENCE_MATCH_SOURCES",
    "MAX_SURFACE_DISTANCE_BLOCKS",
    "SCORECARD_SCHEMA",
    "SURFACE_TOLERANCE_BLOCKS",
    "DatasetSceneInput",
    "FinalizedScene",
    "FrameInput",
    "FrameObservation",
    "PairCorrespondenceEvaluationError",
    "PairGeometry",
    "PairScore",
    "RayOutcome",
    "ScoreTotals",
    "SceneEvaluation",
    "SurfaceHit",
    "build_group_manifest",
    "build_scene",
    "check_fold_leakage",
    "check_split_leakage",
    "classify_returned_pair",
    "construct_truth_pairs",
    "discover_finalized_entries",
    "evaluate_scene",
    "frame_record",
    "geometry_policy_record",
    "is_full_cube_truth_state",
    "macro_average",
    "match_indices",
    "metric_value",
    "observe_frame",
    "pair_record",
    "raycast_surface",
    "score_pair_matches",
    "seal_record",
    "summarise_scene",
    "totals_from_pairs",
]

SCORECARD_SCHEMA = "mc-photogrammetry-pair-correspondence-scorecard-v2"
"""v2 renames the invalid ``repeated_texture`` bucket to an appearance proxy and
keys the truth denominator and returned identity by unique pixel pair."""

SURFACE_TOLERANCE_BLOCKS = 0.0625
"""Preregistered same-surface tolerance: one vanilla texel (1/16 block)."""

MAX_SURFACE_DISTANCE_BLOCKS = 512.0
"""Camera-depth bound for one visibility query, in blocks."""

MAX_TRAVERSAL_STEPS = 1_000_000
"""Hard step cap so a pathological ray cannot loop."""

CORRESPONDENCE_MATCH_SOURCES = (
    "mc_photogrammetry.pipeline.correspondence.detect_features",
    "mc_photogrammetry.pipeline.correspondence.select_mutual_matches",
)
"""The production interfaces that produce the returned match identities.

PlaneGraphTrace hypothesis counts are never used as match counts.
"""

_PARTIAL_FAMILY_SUFFIXES = (
    "_slab", "_stairs", "_trapdoor", "_door", "_fence", "_fence_gate", "_wall",
    "_pane", "_carpet", "_button", "_lever", "_pressure_plate", "_sign",
    "_banner", "_chain", "_lantern", "_torch", "_flower_pot", "_cauldron",
    "_bed", "_cake", "_candle", "_campfire", "_anvil",
)
"""Conservative partial-model families; anything not listed is a full cube."""

_PARTIAL_EXACT_NAMES = frozenset({
    "minecraft:short_grass", "minecraft:tall_grass", "minecraft:fern",
    "minecraft:large_fern", "minecraft:dead_bush", "minecraft:seagrass",
    "minecraft:vine", "minecraft:lily_pad", "minecraft:snow",
    "minecraft:water", "minecraft:lava", "minecraft:rail",
    "minecraft:powered_rail", "minecraft:detector_rail",
    "minecraft:activator_rail", "minecraft:redstone_wire", "minecraft:tripwire",
})

_ENTRY_SLOTS = (("east", "west"), ("up", "down"), ("south", "north"))
"""Face entered when stepping negative/positive along x, y, z."""


class PairCorrespondenceEvaluationError(ValueError):
    """Invalid, unverifiable or inconsistent evaluation input."""


Pixel = tuple[float, float]
"""Exact keypoint pixel, the evaluation-only identity of one observed ray."""


@dataclass(frozen=True, slots=True)
class SurfaceHit:
    """One visible world surface observed by one camera ray.

    ``coordinate`` and ``slot`` together are the *surface identity*: two hits at
    the same block face are candidates for the same physical surface, and the
    ``world_point`` decides whether they are the same point. ``state`` is the
    full exported block state (name plus properties) and ``pixel`` is the exact
    detecting keypoint pixel, so hit pairs can be deduplicated at exactly the
    same pixel without collapsing distinct pixels.
    """

    frame_id: str
    coordinate: tuple[int, int, int]
    slot: str
    world_point: tuple[float, float, float]
    camera_depth: float
    state: BlockState
    pixel: Pixel

    @property
    def face(self) -> tuple[tuple[int, int, int], str]:
        return (self.coordinate, self.slot)

    @property
    def block_name(self) -> str:
        """Exported block name; the full ``state`` is the identity used here."""
        return self.state.name


@dataclass(frozen=True, slots=True)
class RayOutcome:
    """A visibility query plus the typed reason no truth surface was returned."""

    hit: SurfaceHit | None
    reason: str


def is_full_cube_truth_state(state: BlockState) -> bool:
    """Whether an exported state is modelled exactly as an axis-aligned cube.

    Documented, conservative and deterministic: a ``_slab`` is a full cube only
    as ``type=double``; the listed partial families and exact names are not. A
    state the classifier does not recognise is treated as a full cube, as the
    exported ground truth is dominated by full-cube terrain blocks.
    """
    name = state.name
    if name in _PARTIAL_EXACT_NAMES:
        return False
    if name.endswith("_slab"):
        return dict(state.properties).get("type") == "double"
    return not any(name.endswith(suffix) for suffix in _PARTIAL_FAMILY_SUFFIXES)


def _entry_slot(axis: int, positive: bool) -> str:
    return _ENTRY_SLOTS[axis][1 if positive else 0]


def _traverse(
    origin: np.ndarray,
    direction: np.ndarray,
    occupancy: Mapping[tuple[int, int, int], BlockState],
    frame_id: str,
    pixel: Pixel,
    *,
    max_distance: float,
) -> RayOutcome:
    """First occupied full-cube cell along one ray, or a typed refusal.

    A front-to-back voxel walk; the first occupied cell wins and the axis that
    was last crossed is the face the ray entered. Partial-model cells occlude but
    report ``excluded_partial_owner`` instead of a truth surface.
    """
    voxel = [math.floor(float(origin[axis])) for axis in range(3)]
    step = [0, 0, 0]
    t_delta = [math.inf, math.inf, math.inf]
    t_max = [math.inf, math.inf, math.inf]
    for axis in range(3):
        component = float(direction[axis])
        if component > 0.0:
            step[axis] = 1
            t_max[axis] = (voxel[axis] + 1 - float(origin[axis])) / component
            t_delta[axis] = 1.0 / component
        elif component < 0.0:
            step[axis] = -1
            t_max[axis] = (voxel[axis] - float(origin[axis])) / component
            t_delta[axis] = -1.0 / component

    travelled = 0.0
    entry_slot: str | None = None
    for _ in range(MAX_TRAVERSAL_STEPS):
        state = occupancy.get((voxel[0], voxel[1], voxel[2]))
        if state is not None:
            if not is_full_cube_truth_state(state):
                return RayOutcome(None, "excluded_partial_owner")
            if entry_slot is None:
                return RayOutcome(None, "origin_inside_block")
            point = origin + travelled * direction
            return RayOutcome(
                SurfaceHit(
                    frame_id=frame_id,
                    coordinate=(voxel[0], voxel[1], voxel[2]),
                    slot=entry_slot,
                    world_point=(float(point[0]), float(point[1]), float(point[2])),
                    camera_depth=float(travelled),
                    state=state,
                    pixel=pixel,
                ),
                "hit",
            )
        if travelled > max_distance:
            return RayOutcome(None, "exceeded_max_distance")
        axis = 0
        if t_max[1] < t_max[axis]:
            axis = 1
        if t_max[2] < t_max[axis]:
            axis = 2
        if not math.isfinite(t_max[axis]):
            return RayOutcome(None, "no_hit")
        travelled = t_max[axis]
        voxel[axis] += step[axis]
        entry_slot = _entry_slot(axis, step[axis] > 0)
        t_max[axis] += t_delta[axis]
    return RayOutcome(None, "traversal_limit")


def raycast_surface(
    frame_id: str,
    camera: CameraModel,
    width: int,
    height: int,
    occupancy: Mapping[tuple[int, int, int], BlockState],
    x: float,
    y: float,
    *,
    max_distance: float = MAX_SURFACE_DISTANCE_BLOCKS,
) -> RayOutcome:
    """Resolve the world surface a pixel observes, evaluation-only.

    The ray is built with the production camera projection (``ray_directions``);
    the traversal is the query form of the same front-to-back depth test the
    renderer uses.
    """
    if width <= 0 or height <= 0:
        raise PairCorrespondenceEvaluationError("frame dimensions must be positive")
    if not occupancy:
        return RayOutcome(None, "empty_ground_truth")
    pixel = (float(x), float(y))
    direction = ray_directions(
        np.asarray([pixel[0]]), np.asarray([pixel[1]]), camera, width, height
    )[0]
    return _traverse(camera_eye(camera), direction, occupancy, frame_id, pixel,
                     max_distance=max_distance)


def _squared_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return sum((float(first[index]) - float(second[index])) ** 2 for index in range(3))


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(_squared_distance(first, second))


def classify_returned_pair(
    left: SurfaceHit | None,
    right: SurfaceHit | None,
    *,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> str:
    """One returned match's geometry verdict.

    ``positive`` means the two endpoints observe the same block face within the
    preregistered tolerance. ``same_state_same_slot_other_block`` is the
    appearance *proxy* bucket and makes no verified-texture claim: it requires
    the same exact :class:`BlockState` on the same face slot at two *different*
    block coordinates. A same-coordinate/different-slot pair (and any pair that
    differs in either the exact state or the face slot) is ``wrong_surface``.
    """
    if tolerance <= 0.0:
        raise PairCorrespondenceEvaluationError("tolerance must be positive")
    if left is None or right is None:
        return "unattributed"
    if left.face == right.face:
        if _distance(left.world_point, right.world_point) <= tolerance:
            return "positive"
        return "same_face_out_of_tolerance"
    if (left.coordinate != right.coordinate and left.slot == right.slot
            and left.state == right.state):
        return "same_state_same_slot_other_block"
    return "wrong_surface"


def construct_truth_pairs(
    left_hits: Sequence[SurfaceHit | None],
    right_hits: Sequence[SurfaceHit | None],
    *,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> frozenset[tuple[Pixel, Pixel]]:
    """The truth-visible keypoint-pair denominator for one frame pair.

    A pair is in the denominator exactly when both keypoints own the same block
    face and their world surface points are within ``tolerance``; both are
    visible by construction, so the pair is jointly visible. The identity is the
    *unique pixel pair* ``(left pixel, right pixel)``: repeated descriptor
    keypoints at exactly the same pixel collapse to one identity, while distinct
    pixels stay distinct. A voxel-grid index keeps the same-face search linear in
    the number of keypoints.
    """
    if tolerance <= 0.0:
        raise PairCorrespondenceEvaluationError("tolerance must be positive")
    grids: dict[tuple[tuple[int, int, int], str], dict[tuple[int, int, int], list[int]]] = {}
    for index, hit in enumerate(right_hits):
        if hit is None:
            continue
        grid = grids.setdefault(hit.face, {})
        cell = tuple(int(math.floor(hit.world_point[axis] / tolerance)) for axis in range(3))
        grid.setdefault(cell, []).append(index)
    if not grids:
        return frozenset()

    tolerance_squared = tolerance * tolerance
    found: set[tuple[Pixel, Pixel]] = set()
    for hit in left_hits:
        if hit is None:
            continue
        grid = grids.get(hit.face)
        if not grid:
            continue
        base = [int(math.floor(hit.world_point[axis] / tolerance)) for axis in range(3)]
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    cell = (base[0] + delta_x, base[1] + delta_y, base[2] + delta_z)
                    for other in grid.get(cell, ()):
                        other_hit = right_hits[other]
                        assert other_hit is not None
                        if (_squared_distance(hit.world_point, other_hit.world_point)
                                <= tolerance_squared):
                            found.add((hit.pixel, other_hit.pixel))
    return frozenset(found)


@dataclass(frozen=True, slots=True)
class PairGeometry:
    """Returned-match counts and the truth denominator for one frame pair."""

    returned: int
    positive: int
    same_face_out_of_tolerance: int
    same_state_same_slot_other_block: int
    wrong_surface: int
    unattributed: int
    truth_pairs: int
    truth_pairs_returned: int

    @property
    def precision(self) -> float | None:
        if self.returned == 0:
            return None
        return self.positive / self.returned

    @property
    def recall(self) -> float | None:
        if self.truth_pairs == 0:
            return None
        return self.truth_pairs_returned / self.truth_pairs


def match_indices(
    matches: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> np.ndarray:
    """Map ``(x, y, x, y)`` match rows back to keypoint index pairs.

    The match rows are copied verbatim from the keypoint arrays by the production
    matcher, so exact coordinate equality is sound. Repeated keypoints at exactly
    the same pixel share one lookup entry (the first index), which is the unique
    pixel identity the truth denominator also uses; distinct pixels keep distinct
    entries. An unmapped row raises rather than silently dropping evidence.
    """
    rows = np.asarray(matches, dtype=np.float64)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    left_lookup: dict[Pixel, int] = {}
    for index, point in enumerate(np.asarray(left_points, dtype=np.float64)):
        left_lookup.setdefault((float(point[0]), float(point[1])), index)
    right_lookup: dict[Pixel, int] = {}
    for index, point in enumerate(np.asarray(right_points, dtype=np.float64)):
        right_lookup.setdefault((float(point[0]), float(point[1])), index)
    indices: list[tuple[int, int]] = []
    for row in rows:
        left = left_lookup.get((float(row[0]), float(row[1])))
        right = right_lookup.get((float(row[2]), float(row[3])))
        if left is None or right is None:
            raise PairCorrespondenceEvaluationError(
                "a returned match pixel does not map back to a detected keypoint")
        indices.append((left, right))
    return np.asarray(indices, dtype=np.int64)


def _pixel_table(
    points: Sequence[Sequence[float]],
    side: str,
) -> tuple[Pixel, ...]:
    """Detected keypoint pixels as an exact ``(x, y)`` identity table.

    The detected keypoint arrays are the *only* source of a returned row's pixel
    identity. A row whose truth surface is missing (``None``) therefore still
    receives the same identity space :func:`construct_truth_pairs` builds, so
    unattributed rows are deduplicated by pixel pair instead of by keypoint index.
    """
    table = np.asarray(points, dtype=np.float64)
    if table.size == 0:
        return ()
    if table.ndim != 2 or table.shape[1] < 2:
        raise PairCorrespondenceEvaluationError(
            f"{side} keypoint points must be an (n, >= 2) table")
    return tuple((float(row[0]), float(row[1])) for row in table)


def score_pair_matches(
    indices: np.ndarray,
    left_hits: Sequence[SurfaceHit | None],
    right_hits: Sequence[SurfaceHit | None],
    left_points: Sequence[Sequence[float]],
    right_points: Sequence[Sequence[float]],
    *,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> PairGeometry:
    """Precision/recall counts for one pair's returned match identities.

    Every returned row is identified by the *unique pixel pair* of its two
    endpoints - exactly the identity space :func:`construct_truth_pairs` builds -
    and the identity is read from the full detected keypoint point arrays
    (``left_points``/``right_points``), never from a keypoint index fallback. That
    covers rows whose truth surface is absent, so two distinct keypoint indices at
    exactly one pixel pair are one identity for the classification counts, the
    returned/precision denominator and recall reach alike. A repeated descriptor
    keypoint at the same pixel therefore cannot inflate any count, and distinct
    pixels are never collapsed. One identity whose rows disagree about the surface
    verdict raises, because the truth table is then internally inconsistent.
    """
    pairs = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
    left_pixels = _pixel_table(left_points, "left")
    right_pixels = _pixel_table(right_points, "right")
    counts = {
        "positive": 0,
        "same_face_out_of_tolerance": 0,
        "same_state_same_slot_other_block": 0,
        "wrong_surface": 0,
        "unattributed": 0,
    }
    returned: dict[tuple[Pixel, Pixel], str] = {}
    for left_index, right_index in pairs:
        left_index, right_index = int(left_index), int(right_index)
        if not 0 <= left_index < len(left_hits) or not 0 <= right_index < len(right_hits):
            raise PairCorrespondenceEvaluationError(
                "match index lies outside the keypoint truth table")
        if (not 0 <= left_index < len(left_pixels)
                or not 0 <= right_index < len(right_pixels)):
            raise PairCorrespondenceEvaluationError(
                "match index lies outside the keypoint pixel table")
        identity = (left_pixels[left_index], right_pixels[right_index])
        verdict = classify_returned_pair(
            left_hits[left_index], right_hits[right_index], tolerance=tolerance)
        previous = returned.get(identity)
        if previous is not None:
            if previous != verdict:
                raise PairCorrespondenceEvaluationError(
                    "one pixel-pair identity carries conflicting surface verdicts")
            continue
        returned[identity] = verdict
        counts[verdict] += 1
    truth = construct_truth_pairs(left_hits, right_hits, tolerance=tolerance)
    return PairGeometry(
        returned=len(returned),
        positive=counts["positive"],
        same_face_out_of_tolerance=counts["same_face_out_of_tolerance"],
        same_state_same_slot_other_block=counts["same_state_same_slot_other_block"],
        wrong_surface=counts["wrong_surface"],
        unattributed=counts["unattributed"],
        truth_pairs=len(truth),
        truth_pairs_returned=len(set(returned) & truth),
    )


@dataclass(frozen=True, slots=True)
class FrameInput:
    """One finalized frame: pose, resolution and hash-pinned screenshot."""

    frame_id: str
    filename: str
    camera: CameraModel
    width: int
    height: int
    screenshot_path: str
    screenshot_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetSceneInput:
    """A hash-verified entry from one finalized ``index.json``."""

    dataset_id: str
    index_path: str
    index_sha256: str
    scene_id: str
    scene_group: str
    split: str
    directory: str
    scene_root: str
    hashes: tuple[tuple[str, str], ...]
    screenshots: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FinalizedScene:
    """A scene ready to evaluate: frames, occupancy and region bounds."""

    input: DatasetSceneInput
    frames: tuple[FrameInput, ...]
    occupancy: Mapping[tuple[int, int, int], BlockState]
    region_min: tuple[int, int, int]
    region_max: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """SIFT keypoints, descriptors and the truth surface each keypoint observes."""

    frame_id: str
    width: int
    height: int
    usable_pixels: int
    points: np.ndarray
    descriptors: np.ndarray
    hits: tuple[SurfaceHit | None, ...]
    no_truth_reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PairScore:
    """One unordered frame pair's stage funnel, geometry verdicts and metrics."""

    left_id: str
    right_id: str
    status: str
    reason: str
    left_keypoints: int
    right_keypoints: int
    ratio_left: int
    ratio_right: int
    mutual: int
    match_count: int
    spatial_coverage: float
    mutual_fraction: float
    geometry: PairGeometry

    @property
    def precision(self) -> float | None:
        return self.geometry.precision

    @property
    def recall(self) -> float | None:
        return self.geometry.recall


@dataclass(frozen=True, slots=True)
class ScoreTotals:
    """Additive totals behind one pooled or per-group metric."""

    pairs: int = 0
    returned: int = 0
    positive: int = 0
    same_face_out_of_tolerance: int = 0
    same_state_same_slot_other_block: int = 0
    wrong_surface: int = 0
    unattributed: int = 0
    truth_pairs: int = 0
    truth_pairs_returned: int = 0

    @property
    def precision(self) -> float | None:
        if self.returned == 0:
            return None
        return self.positive / self.returned

    @property
    def recall(self) -> float | None:
        if self.truth_pairs == 0:
            return None
        return self.truth_pairs_returned / self.truth_pairs

    def __add__(self, other: "ScoreTotals") -> "ScoreTotals":
        if not isinstance(other, ScoreTotals):
            return NotImplemented
        return ScoreTotals(
            pairs=self.pairs + other.pairs,
            returned=self.returned + other.returned,
            positive=self.positive + other.positive,
            same_face_out_of_tolerance=(
                self.same_face_out_of_tolerance + other.same_face_out_of_tolerance),
            same_state_same_slot_other_block=(
                self.same_state_same_slot_other_block
                + other.same_state_same_slot_other_block),
            wrong_surface=self.wrong_surface + other.wrong_surface,
            unattributed=self.unattributed + other.unattributed,
            truth_pairs=self.truth_pairs + other.truth_pairs,
            truth_pairs_returned=self.truth_pairs_returned + other.truth_pairs_returned,
        )


def metric_value(numerator: int, denominator: int, *, undefined_reason: str) -> dict[str, Any]:
    """A metric that names its denominator and refuses to invent a value."""
    if denominator == 0:
        return {"value": None, "numerator": numerator, "denominator": denominator,
                "undefined_reason": undefined_reason}
    return {"value": numerator / denominator, "numerator": numerator,
            "denominator": denominator, "undefined_reason": None}


def macro_average(values: Sequence[float | None]) -> dict[str, Any]:
    """Mean over *defined* group metrics, reporting how many groups were defined.

    Undefined groups are excluded, not treated as zero, and the count is returned
    so a reader can see the macro rests on a partial set of groups.
    """
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return {"value": None, "defined_groups": 0, "total_groups": len(values),
                "undefined_reason": "no_group_with_a_defined_metric"}
    return {"value": sum(defined) / len(defined), "defined_groups": len(defined),
            "total_groups": len(values), "undefined_reason": None}


def totals_from_pairs(pairs: Sequence[PairScore]) -> ScoreTotals:
    total = ScoreTotals()
    for pair in pairs:
        total = total + ScoreTotals(
            pairs=1,
            returned=pair.geometry.returned,
            positive=pair.geometry.positive,
            same_face_out_of_tolerance=pair.geometry.same_face_out_of_tolerance,
            same_state_same_slot_other_block=(
                pair.geometry.same_state_same_slot_other_block),
            wrong_surface=pair.geometry.wrong_surface,
            unattributed=pair.geometry.unattributed,
            truth_pairs=pair.geometry.truth_pairs,
            truth_pairs_returned=pair.geometry.truth_pairs_returned,
        )
    return total


def _totals_record(totals: ScoreTotals) -> dict[str, int]:
    return {
        "pairs": totals.pairs,
        "returned": totals.returned,
        "positive": totals.positive,
        "same_face_out_of_tolerance": totals.same_face_out_of_tolerance,
        "same_state_same_slot_other_block": totals.same_state_same_slot_other_block,
        "wrong_surface": totals.wrong_surface,
        "unattributed": totals.unattributed,
        "truth_pairs": totals.truth_pairs,
        "truth_pairs_returned": totals.truth_pairs_returned,
        "positive_identity_mismatch": totals.positive - totals.truth_pairs_returned,
    }


def summarise_scene(pairs: Sequence[PairScore]) -> dict[str, Any]:
    """Per-scene totals, pooled metrics and the stage-funnel tally."""
    totals = totals_from_pairs(pairs)
    statuses: dict[str, int] = {}
    for pair in pairs:
        statuses[pair.status] = statuses.get(pair.status, 0) + 1
    return {
        "pair_count": len(pairs),
        "stage_status_counts": dict(sorted(statuses.items())),
        "totals": _totals_record(totals),
        "precision": metric_value(totals.positive, totals.returned,
                                  undefined_reason="no_returned_matches"),
        "recall": metric_value(totals.truth_pairs_returned, totals.truth_pairs,
                               undefined_reason="no_truth_keypoint_pairs"),
    }


def _actual_camera(frame: Mapping[str, Any], planned: CameraModel) -> CameraModel:
    """The recorded *actual* pose, falling back to the planned pose.

    The actual eye position and yaw/pitch are the pose the client reported, so
    the truth projection matches the screenshot rather than the request.
    """
    actual = frame.get("actual")
    actual = actual if isinstance(actual, Mapping) else {}
    eye = actual.get("eye_position")
    yaw = actual.get("yaw_degrees")
    pitch = actual.get("pitch_degrees")
    if (not isinstance(eye, Sequence) or isinstance(eye, (str, bytes)) or len(eye) != 3
            or not isinstance(yaw, (int, float)) or isinstance(yaw, bool)
            or not isinstance(pitch, (int, float)) or isinstance(pitch, bool)):
        return planned
    fov = actual.get("fov_degrees")
    if not isinstance(fov, (int, float)) or isinstance(fov, bool):
        fov = planned.fov_degrees
    try:
        return CameraModel(
            position=(float(eye[0]), float(eye[1]), float(eye[2])),
            yaw_degrees=float(yaw),
            pitch_degrees=float(pitch),
            fov_degrees=float(fov),
            eye_height=0.0,
            intrinsics=planned.intrinsics,
            roll_degrees=planned.roll_degrees,
        )
    except ValueError:
        return planned


def _frame_resolution(frame: Mapping[str, Any], fallback: Sequence[int]) -> tuple[int, int]:
    actual = frame.get("actual")
    actual = actual if isinstance(actual, Mapping) else {}
    width = actual.get("framebuffer_width")
    height = actual.get("framebuffer_height")
    if (isinstance(width, int) and not isinstance(width, bool)
            and isinstance(height, int) and not isinstance(height, bool)
            and width > 0 and height > 0):
        return int(width), int(height)
    return int(fallback[0]), int(fallback[1])


def build_scene(scene_input: DatasetSceneInput) -> FinalizedScene:
    """Read and validate one hash-pinned scene's capture and ground truth."""
    root = Path(scene_input.scene_root)
    capture = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    frames = capture.get("frames")
    cameras = capture.get("cameras")
    resolutions = capture.get("resolutions")
    if not (isinstance(frames, list) and isinstance(cameras, list)
            and isinstance(resolutions, list)):
        raise PairCorrespondenceEvaluationError(f"{scene_input.scene_id}: malformed capture arrays")
    if not (len(frames) == len(cameras) == len(resolutions)):
        raise PairCorrespondenceEvaluationError(
            f"{scene_input.scene_id}: frame/camera/resolution counts differ")
    recorded = {str(row.get("filename")): (str(row.get("frame_id")), str(row.get("sha256")))
                for row in capture.get("screenshots", [])
                if isinstance(row, Mapping)}
    frame_inputs: list[FrameInput] = []
    for index, payload in enumerate(frames):
        if not isinstance(payload, Mapping):
            raise PairCorrespondenceEvaluationError(
                f"{scene_input.scene_id}: malformed frame record")
        planned = CameraModel.from_dict(dict(cameras[index]))
        filename = str(payload.get("filename", ""))
        frame_id = str(payload.get("frame_id", ""))
        if not filename or not frame_id:
            raise PairCorrespondenceEvaluationError(
                f"{scene_input.scene_id}: frame is missing frame_id/filename")
        digest = recorded.get(filename, (frame_id, ""))[1]
        if not digest:
            raise PairCorrespondenceEvaluationError(
                f"{scene_input.scene_id}: no recorded screenshot hash for {filename}")
        width, height = _frame_resolution(payload, resolutions[index])
        frame_inputs.append(FrameInput(
            frame_id=frame_id,
            filename=filename,
            camera=_actual_camera(payload, planned),
            width=width,
            height=height,
            screenshot_path=str(root / "captures" / filename),
            screenshot_sha256=digest,
        ))
    export = parse_ground_truth(ground_truth)
    volume = volume_from_ground_truth(export)
    occupancy = {coordinate: observation.state for coordinate, observation in volume.items()}
    if not occupancy:
        raise PairCorrespondenceEvaluationError(
            f"{scene_input.scene_id}: ground truth contains no non-air blocks")
    frame_inputs.sort(key=lambda item: item.frame_id)
    return FinalizedScene(
        input=scene_input,
        frames=tuple(frame_inputs),
        occupancy=occupancy,
        region_min=tuple(int(value) for value in export.region.minimum),
        region_max=tuple(int(value) for value in export.region.maximum),
    )


def observe_frame(
    scene: FinalizedScene,
    frame: FrameInput,
    *,
    settings: CorrespondenceSettings | None = None,
) -> FrameObservation:
    """Detect keypoints and resolve the truth surface each one observes."""
    from PIL import Image

    with Image.open(frame.screenshot_path) as opened:
        opened.load()
        image = opened.convert("RGB")
    if image.size != (frame.width, frame.height):
        raise PairCorrespondenceEvaluationError(
            f"{frame.frame_id}: screenshot {image.size} does not match declared "
            f"{(frame.width, frame.height)}")
    pixels = np.asarray(image, dtype=np.uint8)
    usable = usable_mask_for(frame.frame_id, pixels.shape, None)
    features = detect_features(frame.frame_id, pixels, usable_mask=usable, settings=settings)
    hits: list[SurfaceHit | None] = []
    reasons: dict[str, int] = {}
    for point in features.points:
        outcome = raycast_surface(
            frame.frame_id, frame.camera, frame.width, frame.height, scene.occupancy,
            float(point[0]), float(point[1]),
        )
        hits.append(outcome.hit)
        reasons[outcome.reason] = reasons.get(outcome.reason, 0) + 1
    return FrameObservation(
        frame_id=frame.frame_id,
        width=frame.width,
        height=frame.height,
        usable_pixels=int(features.usable_pixels),
        points=np.asarray(features.points, dtype=np.float64),
        descriptors=np.asarray(features.descriptors, dtype=np.float64),
        hits=tuple(hits),
        no_truth_reasons=tuple(sorted(reasons.items())),
    )


def _bins_covered(points: np.ndarray, width: int, height: int,
                  grid: tuple[int, int]) -> int:
    if points.size == 0:
        return 0
    grid_x, grid_y = grid
    column = np.clip((points[:, 0] / max(width, 1) * grid_x).astype(int), 0, grid_x - 1)
    row = np.clip((points[:, 1] / max(height, 1) * grid_y).astype(int), 0, grid_y - 1)
    return int(np.unique(column * grid_y + row).size)


def _classify_stage(
    left: FrameFeatures,
    right: FrameFeatures,
    match_count: int,
    coverage: float,
    mutual_fraction: float,
    bounds: CorrespondenceSettings,
) -> tuple[str, str]:
    """Mirror of the production correspondence funnel, in the same order.

    Kept evaluation-only so the scorecard can use cached per-frame features while
    reporting the same typed refusal a production extraction would.
    """
    if left.usable_pixels == 0 or right.usable_pixels == 0:
        return "unusable_frame", "one frame has no usable pixels"
    raw_left, raw_right = len(left.points), len(right.points)
    if min(raw_left, raw_right) < bounds.min_keypoints_per_frame:
        return ("insufficient_keypoints",
                f"raw keypoints {raw_left}/{raw_right} below {bounds.min_keypoints_per_frame}")
    if match_count < bounds.min_matches:
        return ("insufficient_matches",
                f"{match_count} mutual matches below {bounds.min_matches}")
    if coverage < bounds.min_spatial_coverage:
        return ("repetition_risk",
                f"spatial coverage {coverage:.2f} below {bounds.min_spatial_coverage:.2f}")
    if mutual_fraction < bounds.min_mutual_fraction:
        return ("repetition_risk",
                f"mutual fraction {mutual_fraction:.4f} below {bounds.min_mutual_fraction:.4f}")
    return "ok", ""


def _features(observation: FrameObservation) -> FrameFeatures:
    return FrameFeatures(
        frame_id=observation.frame_id,
        points=observation.points,
        descriptors=np.empty((len(observation.points), 0), dtype=np.float32),
        width=observation.width,
        height=observation.height,
        usable_pixels=observation.usable_pixels,
    )


def _score_with_selection(
    left: FrameObservation,
    right: FrameObservation,
    selection: Any,
    bounds: CorrespondenceSettings,
    tolerance: float,
) -> PairScore:
    matches = selection.matches
    match_count = int(matches.shape[0])
    bins_left = _bins_covered(matches[:, 0:2], left.width, left.height, bounds.spatial_grid)
    bins_right = _bins_covered(matches[:, 2:4], right.width, right.height, bounds.spatial_grid)
    bins_total = int(bounds.spatial_grid[0] * bounds.spatial_grid[1])
    coverage = min(bins_left, bins_right) / bins_total if bins_total else 0.0
    smaller = min(int(selection.left_keypoints), int(selection.right_keypoints))
    mutual_fraction = selection.mutual / smaller if smaller else 0.0
    status, reason = _classify_stage(
        _features(left), _features(right), match_count, coverage, mutual_fraction, bounds)
    indices = match_indices(matches, left.points, right.points)
    geometry = score_pair_matches(
        indices, left.hits, right.hits, left.points, right.points, tolerance=tolerance)
    return PairScore(
        left_id=left.frame_id,
        right_id=right.frame_id,
        status=status,
        reason=reason,
        left_keypoints=int(selection.left_keypoints),
        right_keypoints=int(selection.right_keypoints),
        ratio_left=int(selection.ratio_left),
        ratio_right=int(selection.ratio_right),
        mutual=int(selection.mutual),
        match_count=match_count,
        spatial_coverage=coverage,
        mutual_fraction=mutual_fraction,
        geometry=geometry,
    )


@dataclass(frozen=True, slots=True)
class SceneEvaluation:
    """One scene's frame observations and every unordered pair score."""

    observations: tuple[FrameObservation, ...]
    pairs: tuple[PairScore, ...]


def evaluate_scene(
    scene: FinalizedScene,
    *,
    settings: CorrespondenceSettings | None = None,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> SceneEvaluation:
    """Evaluate every unordered frame pair in one scene.

    Features are detected once per frame and reused across pairs; the pairing and
    the returned match identities come from the production matcher
    (``select_mutual_matches``).
    """
    bounds = CorrespondenceSettings() if settings is None else settings
    observations = [observe_frame(scene, frame, settings=bounds) for frame in scene.frames]
    scores: list[PairScore] = []
    for left_index in range(len(observations)):
        for right_index in range(left_index + 1, len(observations)):
            selection = select_mutual_matches(
                observations[left_index].points,
                observations[right_index].points,
                observations[left_index].descriptors,
                observations[right_index].descriptors,
                ratio_test=bounds.ratio_test,
            )
            scores.append(_score_with_selection(
                observations[left_index], observations[right_index], selection, bounds, tolerance))
    return SceneEvaluation(observations=tuple(observations), pairs=tuple(scores))


def discover_finalized_entries(exports_root: Path | str) -> tuple[DatasetSceneInput, ...]:
    """Hash-pin every finalized ``index.json`` entry under ``exports_root``.

    Every listed scene-file hash and screenshot hash is recomputed before the
    entry is admitted, so the scorecard can only run against unmodified artifacts.
    """
    root = Path(exports_root)
    if not root.is_dir():
        raise PairCorrespondenceEvaluationError(f"no dataset export root at {root}")
    entries: list[DatasetSceneInput] = []
    for index_path in sorted(root.glob("**/index.json")):
        raw = index_path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PairCorrespondenceEvaluationError(f"{index_path}: unreadable index") from error
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise PairCorrespondenceEvaluationError(f"{index_path}: unsupported index schema")
        scenes = payload.get("scenes")
        if not isinstance(scenes, list):
            raise PairCorrespondenceEvaluationError(f"{index_path}: scenes must be a list")
        dataset_id = index_path.parent.name
        index_sha256 = hashlib.sha256(raw).hexdigest()
        for scene in scenes:
            if not isinstance(scene, Mapping):
                raise PairCorrespondenceEvaluationError(f"{index_path}: malformed scene entry")
            scene_root = index_path.parent / str(scene.get("directory", ""))
            hashes = scene.get("hashes")
            hashes = hashes if isinstance(hashes, Mapping) else {}
            verified: list[tuple[str, str]] = []
            for name, digest in sorted(hashes.items()):
                path = scene_root / str(name)
                actual = sha256_file(path) if path.is_file() else ""
                if actual != str(digest):
                    raise PairCorrespondenceEvaluationError(
                        f"{path}: hash {actual or 'missing'} != pinned {digest}")
                verified.append((str(name), str(digest)))
            screenshots: list[tuple[str, str]] = []
            rows = scene.get("screenshots")
            if not isinstance(rows, list):
                raise PairCorrespondenceEvaluationError(
                    f"{scene_root}: screenshots must be a list")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise PairCorrespondenceEvaluationError(
                        f"{scene_root}: malformed screenshot row")
                filename = str(row.get("filename", ""))
                digest = str(row.get("sha256", ""))
                path = scene_root / "captures" / filename
                actual = sha256_file(path) if path.is_file() else ""
                if actual != digest:
                    raise PairCorrespondenceEvaluationError(
                        f"{path}: hash {actual or 'missing'} != pinned {digest}")
                screenshots.append((filename, digest))
            group = str(scene.get("scene_group") or "")
            if not group:
                raise PairCorrespondenceEvaluationError(
                    f"{scene_root}: entry must carry a scene_group")
            entries.append(DatasetSceneInput(
                dataset_id=dataset_id,
                index_path=str(index_path),
                index_sha256=index_sha256,
                scene_id=str(scene.get("scene_id", "")),
                scene_group=group,
                split=str(scene.get("split", "")),
                directory=str(scene.get("directory", "")),
                scene_root=str(scene_root),
                hashes=tuple(verified),
                screenshots=tuple(sorted(screenshots)),
            ))
    if not entries:
        raise PairCorrespondenceEvaluationError(f"no finalized dataset entries under {root}")
    return tuple(sorted(entries, key=lambda item: (
        item.scene_group, item.scene_id, item.dataset_id, item.directory)))


def check_split_leakage(inputs: Sequence[DatasetSceneInput]) -> tuple[str, ...]:
    """Scene-id/group/split consistency issues; empty means reusable grouping."""
    issues: list[str] = []
    groups_by_scene: dict[str, set[str]] = {}
    splits_by_group: dict[str, set[str]] = {}
    for item in inputs:
        groups_by_scene.setdefault(item.scene_id, set()).add(item.scene_group)
        splits_by_group.setdefault(item.scene_group, set()).add(item.split)
    for scene_id, groups in sorted(groups_by_scene.items()):
        if len(groups) > 1:
            issues.append(f"scene {scene_id!r} appears in multiple groups {sorted(groups)}")
    for group, splits in sorted(splits_by_group.items()):
        if len(splits) > 1:
            issues.append(f"scene group {group!r} carries multiple index splits {sorted(splits)}")
    return tuple(issues)


def build_group_manifest(inputs: Sequence[DatasetSceneInput]) -> dict[str, Any]:
    """Deterministic topology-group manifest plus exploratory LOO folds.

    Groups are keyed strictly by ``scene_group`` (topology), so duplicate exports
    of one structure cannot be separated. Nothing is trained here; the LOO folds
    are descriptive and no generalization claim is made from three groups.
    """
    issues = check_split_leakage(inputs)
    if issues:
        raise PairCorrespondenceEvaluationError("split leakage: " + "; ".join(issues))
    grouped: dict[str, list[DatasetSceneInput]] = {}
    for item in inputs:
        grouped.setdefault(item.scene_group, []).append(item)
    group_records: list[dict[str, Any]] = []
    for group in sorted(grouped):
        members = sorted(grouped[group], key=lambda item: (item.scene_id, item.dataset_id,
                                                           item.directory))
        group_records.append({
            "scene_group": group,
            "scene_ids": sorted({member.scene_id for member in members}),
            "datasets": sorted({member.dataset_id for member in members}),
            "entry_count": len(members),
            "entries": [f"{member.dataset_id}:{member.directory}" for member in members],
        })
    folds: list[dict[str, Any]] = []
    for group in sorted(grouped):
        held = [item for item in inputs if item.scene_group == group]
        retained = [item for item in inputs if item.scene_group != group]
        folds.append({
            "fold": f"heldout-{group[:12]}",
            "held_out_scene_group": group,
            "held_out_entries": sorted(
                f"{item.dataset_id}:{item.directory}" for item in held),
            "retained_entries": sorted(
                f"{item.dataset_id}:{item.directory}" for item in retained),
            "retained_scene_groups": sorted({item.scene_group for item in retained}),
        })
    manifest = {
        "grouping_key": "scene_group",
        "entry_count": len(inputs),
        "group_count": len(grouped),
        "groups": group_records,
        "folds": folds,
    }
    leakage = check_fold_leakage(manifest)
    if leakage:
        raise PairCorrespondenceEvaluationError("fold leakage: " + "; ".join(leakage))
    manifest["leakage_issues"] = []
    return manifest


def check_fold_leakage(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """No group or duplicated scene may sit on both sides of any fold."""
    issues: list[str] = []
    groups = manifest.get("groups")
    groups = groups if isinstance(groups, list) else []
    group_of_entry: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        for entry in group.get("entries", []):
            group_of_entry[str(entry)] = str(group.get("scene_group"))
    for fold in manifest.get("folds", []):
        if not isinstance(fold, Mapping):
            continue
        held = {str(entry) for entry in fold.get("held_out_entries", [])}
        retained = {str(entry) for entry in fold.get("retained_entries", [])}
        for entry in sorted(held & retained):
            issues.append(f"fold {fold.get('fold')!r} has {entry!r} on both sides")
        held_groups = {group_of_entry.get(entry, "") for entry in held}
        retained_groups = {group_of_entry.get(entry, "") for entry in retained}
        for group in sorted(held_groups & retained_groups):
            issues.append(f"scene group {group!r} crosses fold {fold.get('fold')!r}")
    return tuple(issues)


def seal_record(payload: Mapping[str, Any]) -> str:
    """Canonical self-seal over a payload without its ``seal_sha256`` field."""
    body = {key: value for key, value in payload.items() if key != "seal_sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def geometry_policy_record() -> dict[str, Any]:
    """The preregistered truth/geometry policy, for the sealed report record."""
    return {
        "truth_model": (
            "registered ground-truth region blocks + recorded actual camera pose; "
            "per-query front-to-back voxel z-buffer/face ownership"
        ),
        "partial_model_policy": (
            "partial families occlude as full cubes but are excluded from truth "
            "ownership; only _slab type=double is treated as a full cube"
        ),
        "partial_family_suffixes": list(_PARTIAL_FAMILY_SUFFIXES),
        "partial_exact_names": sorted(_PARTIAL_EXACT_NAMES),
        "surface_tolerance_blocks": SURFACE_TOLERANCE_BLOCKS,
        "max_surface_distance_blocks": MAX_SURFACE_DISTANCE_BLOCKS,
        "appearance_proxy": (
            "same_state_same_slot_other_block counts only the same exact BlockState "
            "on the same face slot at a different block coordinate; no texture "
            "identity is read, hashed or compared, so it is not texture ground truth"
        ),
        "identity": (
            "returned identity and truth denominator both use the unique pixel pair "
            "(left pixel, right pixel); repeated keypoints at exactly the same pixel "
            "collapse, distinct pixels never collapse"
        ),
        "match_sources": list(CORRESPONDENCE_MATCH_SOURCES),
    }


def pair_record(pair: PairScore) -> dict[str, Any]:
    """One frame pair's stage funnel and geometry verdicts as JSON data."""
    geometry = pair.geometry
    return {
        "left_frame_id": pair.left_id,
        "right_frame_id": pair.right_id,
        "stage_status": pair.status,
        "stage_reason": pair.reason,
        "left_keypoints": pair.left_keypoints,
        "right_keypoints": pair.right_keypoints,
        "ratio_left": pair.ratio_left,
        "ratio_right": pair.ratio_right,
        "mutual": pair.mutual,
        "returned_matches": pair.match_count,
        "spatial_coverage": pair.spatial_coverage,
        "mutual_fraction": pair.mutual_fraction,
        "returned": geometry.returned,
        "positive": geometry.positive,
        "same_face_out_of_tolerance": geometry.same_face_out_of_tolerance,
        "same_state_same_slot_other_block": geometry.same_state_same_slot_other_block,
        "wrong_surface": geometry.wrong_surface,
        "unattributed": geometry.unattributed,
        "truth_pairs": geometry.truth_pairs,
        "truth_pairs_returned": geometry.truth_pairs_returned,
        "precision": pair.precision,
        "recall": pair.recall,
    }


def frame_record(observation: FrameObservation) -> dict[str, Any]:
    """One frame's keypoint count, truth coverage and no-truth reason tally."""
    with_truth = sum(1 for hit in observation.hits if hit is not None)
    return {
        "frame_id": observation.frame_id,
        "width": observation.width,
        "height": observation.height,
        "usable_pixels": observation.usable_pixels,
        "keypoints": len(observation.points),
        "keypoints_with_truth": with_truth,
        "keypoints_without_truth": len(observation.points) - with_truth,
        "no_truth_reasons": dict(observation.no_truth_reasons),
    }
