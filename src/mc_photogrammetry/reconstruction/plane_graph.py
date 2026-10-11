"""Label-blind plane identity, quadrant and translation from lattice readings.

:mod:`lattice_planes` measures, per frame and per candidate plane, a small
ordered set of block distances. This module decides which readings across frames
are the *same physical plane*, which of the four yaw quadrants each frame sits
in, and where the frames stand relative to one another — using arithmetic that
needs no appearance model and no reference:

* **Same-axis equality.** Two same-axis planes seen by one frame pair must imply
  the same relative eye displacement, because both name ``t_right - t_left`` on
  that axis. A ``(b, identity)`` combination that breaks the equality is a wrong
  combination.
* **The integer-face test.** A Minecraft face lies at an integer world
  coordinate, so ``t_v[a] + s_v d_v`` must be integer for every view, up to one
  declared gauge per axis. The resultant ``R_a = |mean(exp(2*pi*i*(t+s*d)))|``
  is 1 when they agree and drops when plane identity, ``b``, side or quadrant is
  wrong. It needs no correlation at all.

Phase correlation is an *optional scored cue*: it contributes to a match's
score and it is the tie-breaker of last resort, but it is never treated as
ground truth, and a uniform repeating plane is refused rather than matched.
Nothing here rounds a block id before the lattice frame exists (DS-10 C3) and
nothing reads a manifest, a reference litematic, an evaluation mask or an
annotation. There is no RNG anywhere in this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
from itertools import product
from typing import Iterator, Mapping, Sequence

import numpy as np

from mc_photogrammetry.reconstruction.lattice_planes import (
    MIN_STRUCTURAL_SHARE,
    PlaneReading,
    PlaneSignature,
    WarpSignatureWitnessV1,
    reading_order_key,
    warp_signature,
)
from mc_photogrammetry.reconstruction import lattice, region_growth

COMPONENT_ABSTENTION_REASONS: tuple[str, ...] = (
    "no_plane_reading",
    "no_matches",
    "insufficient_structural_share",
    "repetition_unresolved",
    "plane_identity_ambiguous",
    "same_axis_mismatch",
    "ambiguous_quadrant",
    "quadrant_constraint_inconsistent",
    "endpoint_period_conflict",
    "plane_integer_residual",
    "insufficient_axis_coverage",
    "b_unresolved",
    "component_too_large",
)
"""Every typed reason :func:`solve_component` may return for an incomplete solve."""

MATCH_REFUSAL_REASONS: tuple[str, ...] = (
    "axis_mismatch",
    "invalid_distance_alternative",
    "insufficient_overlap",
    "no_signal",
    "repetition_unresolved",
    "ambiguity_check_inconclusive",
    "same_axis_mismatch",
)
"""Every typed reason :func:`match_plane_pair` or the equality filter may record."""

SAME_AXIS_DISPOSITION_KEPT = "kept"
"""Disposition of a pre-filter candidate that survived :func:`_apply_same_axis_filter`."""

SAME_AXIS_DISPOSITION_DROPPED = "same_axis_mismatch"
"""Disposition of a pre-filter candidate the equality filter dropped."""

SAME_AXIS_TOLERANCE = 0.15
"""Blocks within which two same-axis constraints count as the same displacement."""

PHYSICAL_DISPLACEMENT_POLICY = "signed-world-axis-diameter-and-interval-cycles-v1"
MAX_ADMITTED_CANDIDATES = 16_384


@dataclass(frozen=True, slots=True)
class AdmittedCandidateUniverse:
    """Ordered immutable matcher admission, before scored legacy filtering."""
    candidates: tuple
    dispositions: tuple
    source_rows: tuple

    def __post_init__(self):
        if (type(self.candidates) is not tuple or len(self.candidates) > MAX_ADMITTED_CANDIDATES
                or type(self.dispositions) is not tuple or type(self.source_rows) is not tuple
                or self.source_rows != tuple(range(len(self.candidates)))
                or len(self.dispositions) != len(self.candidates)
                or any(type(row) is not PlaneMatch for row in self.candidates)
                or any(d not in (SAME_AXIS_DISPOSITION_KEPT, SAME_AXIS_DISPOSITION_DROPPED)
                       for d in self.dispositions)):
            raise ValueError("invalid bounded admitted candidate universe")


def signed_physical_relation(indexed, row, quadrants, selected, row_index, *, compact=False):
    """Bind exact selected alternatives; normalize frame order after world mapping."""
    left, right = _validate_indexed_match(row, indexed)
    u, v = (row.left, row.left_plane), (row.right, row.right_plane)
    if (row.left == row.right or set(quadrants) != {f for f, _ in indexed}
            or any(type(q) is not int or q not in range(4) for q in quadrants.values())
            or (quadrants[row.right] - quadrants[row.left]) % 4 != row.quadrant_of_right):
        raise ValueError("unresolvable physical quadrants")
    endpoints = []
    for key, reading, requested in ((u, left, row.left_b), (v, right, row.right_b)):
        a = selected.get(key)
        if type(a) is not int or not 0 <= a < len(reading.alternatives):
            raise ValueError("unresolvable physical alternative")
        alternative = reading.alternatives[a]
        if not any(alternative is item for item in reading.matching_alternatives(requested)):
            raise ValueError("physical alternative does not bind source row")
        axis, sign = axis_transform(reading.axis, quadrants[reading.frame])
        side, distance = float(reading.side), float(alternative.distance_blocks)
        if side not in (-1.0, 1.0) or not math.isfinite(distance):
            raise ValueError("unverified physical displacement")
        endpoints.append((key, a, reading.axis, axis, sign, side,
                          float(alternative.blocks_per_period), distance))
    if endpoints[0][3] != endpoints[1][3]:
        raise ValueError("physical world axis mismatch")
    delta = endpoints[0][4] * endpoints[0][5] * endpoints[0][7] - endpoints[1][4] * endpoints[1][5] * endpoints[1][7]
    if not math.isfinite(delta):
        raise ValueError("nonfinite physical displacement")
    if row.right < row.left:
        endpoints.reverse()
        delta = -delta
    f, g = endpoints[0][0][0], endpoints[1][0][0]
    if compact:
        # Source replay binds omitted fields to the complete original universe.
        return ((f, g, endpoints[0][3]), float(delta), row_index,
                endpoints[0][1], endpoints[1][1])
    payload = {name: getattr(row, name) for name in row.__dataclass_fields__}
    for name, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            payload[name] = str(value)
    row_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    return ((f, g, endpoints[0][3]), float(delta), row_index, row_digest,
            (quadrants[g] - quadrants[f]) % 4, tuple(endpoints))


def physical_displacement_refusal(indexed, matches, quadrants, selected, translations):
    """Independent real-output guard. No circular or integer-period residuals."""
    if len(matches) > MAX_ADMITTED_CANDIDATES:
        return "unverified_physical_displacement"
    extrema = {}
    try:
        for i, row in enumerate(matches):
            key, delta, *_ = signed_physical_relation(indexed, row, quadrants, selected, i)
            f, g, axis = key
            measured = float(translations[g][axis]) - float(translations[f][axis])
            if not math.isfinite(measured) or abs(measured - delta) > SAME_AXIS_TOLERANCE:
                return "plane_integer_residual"
            low, high = extrema.get(key, (delta, delta))
            low, high = min(low, delta), max(high, delta)
            if high - low > SAME_AXIS_TOLERANCE:
                return "plane_integer_residual"
            extrema[key] = (low, high)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return "unverified_physical_displacement"
    return None if extrema else "unverified_physical_displacement"

LAMBDA_UNRESOLVED = 0.1
"""Cost added per frame with no matched plane (preregistered)."""

MU_MATCH_SCORE = 0.1
"""Weight on ``1 - mean match score`` in the assignment cost (preregistered)."""

MAX_ASSIGNMENTS = 16_384
"""Hard cap on exact full assignments: ``4^(C-1)`` for C constraint components.

This permits eight free components, including isolated reading-present frames;
numerical work and peak RSS per assignment still depend on frames and matches.
"""

MAX_ALTERNATIVES = 8
"""Competing quadrant assignments retained on a solved component."""

MIN_OVERLAP_CELLS = 9
"""Cells two signatures must share before a correlation is even attempted."""


def _order_key_is_finite_and_orderable(value: object) -> bool:
    if type(value) is tuple:
        return all(_order_key_is_finite_and_orderable(item) for item in value)
    if type(value) is str:
        return True
    if type(value) is bool or isinstance(value, np.bool_):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return bool(np.isfinite(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _is_finite_positive_number(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(numeric) and numeric > 0.0)


def _canonical_reading_index(
    readings: Sequence[PlaneReading],
) -> tuple[dict[tuple[str, int], PlaneReading], dict[int, tuple[str, int]], tuple[str, ...]]:
    """Build the one complete frame/index map used by matcher and solver."""
    if not isinstance(readings, Sequence):
        raise ValueError("complete reading universe must be a sequence")
    by_frame: dict[str, list[tuple[tuple, PlaneReading]]] = {}
    seen_objects: set[int] = set()
    for reading in readings:
        if not isinstance(reading, PlaneReading):
            raise ValueError("complete reading universe contains a non-reading value")
        if type(reading.frame) is not str or not reading.frame:
            raise ValueError("every reading must have a non-empty string frame")
        if (type(reading.axis) is not int or reading.axis not in (0, 1, 2)):
            raise ValueError("every reading must have a canonical axis")
        identity = id(reading)
        if identity in seen_objects:
            raise ValueError("complete reading universe repeats a reading object")
        seen_objects.add(identity)
        try:
            order_key = reading_order_key(reading)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("reading order key is malformed") from error
        if (type(order_key) is not tuple
                or not _order_key_is_finite_and_orderable(order_key)):
            raise ValueError("reading order key must contain finite, orderable values")
        by_frame.setdefault(reading.frame, []).append((order_key, reading))

    read_by_index: dict[tuple[str, int], PlaneReading] = {}
    index_by_object: dict[int, tuple[str, int]] = {}
    frames = tuple(sorted(by_frame))
    for frame in frames:
        try:
            keyed_readings = sorted(by_frame[frame], key=lambda pair: pair[0])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("readings in one frame have unordered canonical keys") from error
        for previous, current in zip(keyed_readings, keyed_readings[1:]):
            if previous[0] == current[0]:
                raise ValueError("distinct readings have an ambiguous canonical order key")
        for index, (_order_key, reading) in enumerate(keyed_readings):
            pair = (frame, index)
            read_by_index[pair] = reading
            index_by_object[id(reading)] = pair
    return read_by_index, index_by_object, frames


def _validate_indexed_match(
    match: PlaneMatch,
    read_by_index: Mapping[tuple[str, int], PlaneReading],
    *,
    expected_left: PlaneReading | None = None,
    expected_right: PlaneReading | None = None,
    require_selected_b: bool = False,
) -> tuple[PlaneReading, PlaneReading]:
    """Resolve a match against the canonical index and check both endpoints."""
    if not isinstance(match, PlaneMatch):
        raise ValueError("graph contains an invalid match value")
    if (type(match.left_plane) is not int or match.left_plane < 0
            or type(match.right_plane) is not int or match.right_plane < 0):
        raise ValueError("match endpoints need canonical non-negative indices")
    left = read_by_index.get((match.left, match.left_plane))
    right = read_by_index.get((match.right, match.right_plane))
    if left is None or right is None:
        raise ValueError("match endpoint is outside the complete reading universe")
    if (left.frame != match.left or right.frame != match.right
            or left.axis != match.left_axis or right.axis != match.right_axis):
        raise ValueError("match endpoint index resolves to a different frame or axis")
    if ((expected_left is not None and left is not expected_left)
            or (expected_right is not None and right is not expected_right)):
        raise ValueError("match endpoint does not preserve source reading identity")
    selected_b = (match.left_b, match.right_b)
    if any(not _is_finite_positive_number(value) for value in selected_b):
        raise ValueError("match endpoint selected b must be finite and positive")
    if require_selected_b:
        if (left.distance_for(match.left_b) is None
                or right.distance_for(match.right_b) is None):
            raise ValueError("match endpoint has no unique selected canonical b")
    return left, right

PSR_SCALE = 6.0
"""``psr`` at which the peak term of the score saturates (preregistered)."""

PLANE_CORRELATION_ALGORITHM = "common-canvas-separate-mask-halo-v3"
"""Version of the support-aware plane-signature correlation algorithm."""

ROBUST_SAMPLES = 512
"""Deterministic endpoint-distinct pair samples for the class-constrained fit."""


@dataclass(frozen=True, slots=True)
class PlaneMatch:
    """One candidate identity between a left and a right plane reading.

    ``left``/``right`` are frame ids; ``left_plane``/``right_plane`` index the
    reading within that frame's ordered reading tuple, which is how the solver
    recovers the measured distance behind ``left_b``/``right_b``. ``psr`` is the
    phase-correlation peak-to-sidelobe ratio of the seam-band maps, retained as
    a scored cue and never as proof; ``integer_residual`` is filled by
    :func:`solve_component` (it is meaningless before the translations exist).
    """

    left: str
    right: str
    left_axis: int
    right_axis: int
    left_b: float
    right_b: float
    quadrant_of_right: int
    psr: float
    peak: float
    second_peak: float
    overlap_cells: int
    integer_residual: float | None
    score: float
    left_plane: int = -1
    right_plane: int = -1
    cell_shift_rc: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        shift = self.cell_shift_rc
        if shift is not None and (
                not isinstance(shift, tuple) or len(shift) != 2
                or any(type(value) is not int for value in shift)):
            raise ValueError("cell_shift_rc must be an integer (row, column) pair")


@dataclass(frozen=True, slots=True)
class PlaneMatchRefusal:
    """A candidate plane pair that was deliberately not matched."""

    left: str
    right: str
    left_axis: int
    right_axis: int
    quadrant_of_right: int
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PlanePairTrace:
    left: str
    right: str
    left_axis: int
    right_axis: int
    quadrant: int
    admitted: bool
    reason: str
    raw_max_shift_rc: tuple[int, int] | None
    raw_max_value: float | None
    raw_max_support: int | None
    supported_max_shift_rc: tuple[int, int] | None
    supported_max_value: float | None
    supported_max_support: int | None
    left_plane: int | None = None
    right_plane: int | None = None
    selected: bool = False
    psr: float | None = None
    score: float | None = None
    overlap_cells: int | None = None
    valid_cells: int | None = None
    overlap_fraction: float | None = None
    cell_shift_rc: tuple[int, int] | None = None
    left_structural_share: float | None = None
    right_structural_share: float | None = None


@dataclass(slots=True)
class PlaneGraphTrace:
    pairs: list[PlanePairTrace]
    pre_admission_candidates: int = 0
    post_admission_candidates: int = 0
    post_same_axis_candidates: int = 0
    admitted_ledger_sha256: str = ""


@dataclass(frozen=True, slots=True)
class SameAxisCandidateObservation:
    """Raw, output-neutral record of one complete same-axis filtering pass.

    ``candidates`` is the matcher's complete admitted set in canonical
    :func:`_match_order` -- every :class:`PlaneMatch` that passed correlation,
    including the ones the equality filter later dropped. ``dispositions`` is
    parallel to it and holds each candidate's typed same-axis verdict:
    :data:`SAME_AXIS_DISPOSITION_KEPT` when the candidate survived into the
    solver input, or :data:`SAME_AXIS_DISPOSITION_DROPPED` when the filter dropped
    it because another same-axis plane pair implied a different displacement.

    It is produced only when a caller explicitly attaches an observation sink to
    :func:`match_plane_graph`, so the default matcher path allocates and retains
    nothing and the returned match tuple is byte-for-byte unchanged. Evaluation
    code can therefore bind the *pre-filter* candidates of any same-axis group to
    the survivors the solver actually received.
    """

    candidates: tuple["PlaneMatch", ...]
    dispositions: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.dispositions):
            raise ValueError(
                "same-axis candidate observation dispositions must be parallel")
        for match, disposition in zip(self.candidates, self.dispositions):
            if not isinstance(match, PlaneMatch):
                raise ValueError(
                    "same-axis candidate observation holds a non-match value")
            if disposition not in (SAME_AXIS_DISPOSITION_KEPT,
                                   SAME_AXIS_DISPOSITION_DROPPED):
                raise ValueError(
                    "same-axis candidate observation has an unknown disposition")


@dataclass(frozen=True, slots=True)
class MatcherPairWitnessV1:
    schema: str
    algorithm: str
    max_shift_cells: int
    quadrant: int
    homography: np.ndarray
    warp_witness_sha256: str
    canvas_shape: tuple[int, int]
    selected_shift_rc: tuple[int, int]
    selected_support: int
    logical_pairs: np.ndarray
    witness_sha256: str = ""
    warp_witness: WarpSignatureWitnessV1 | None = field(
        default=None, compare=False, repr=False)

    right_grid_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if (self.schema != "mc-photogrammetry-matcher-pair-witness-v1"
                or self.algorithm != PLANE_CORRELATION_ALGORITHM
                or self.homography.shape != (3, 3) or self.homography.flags.writeable
                or self.logical_pairs.ndim != 2 or self.logical_pairs.shape[1] != 4
                or self.logical_pairs.dtype != np.int64
                or self.logical_pairs.flags.writeable
                or len(self.logical_pairs) != self.selected_support):
            raise ValueError("matcher pair witness is malformed or mutable")
        # C5: every selected logical pair must be a real source/target cell. The
        # ``-1`` border sentinel belongs to the warp sampling map only, never to a
        # selected pair, and ``0`` is a real cell index (not a sentinel).
        if self.logical_pairs.size and int(self.logical_pairs.min()) < 0:
            raise ValueError("matcher pair witness selects a sentinel logical pair")
        if (self.warp_witness is not None
                and not isinstance(self.warp_witness, WarpSignatureWitnessV1)):
            raise ValueError("matcher warp witness does not bind pair witness")
        if self.right_grid_shape is None:
            raise ValueError("matcher pair witness requires original right grid shape")
        shape, fft_shape = _common_canvas_geometry(self.right_grid_shape, self.max_shift_cells)
        if self.canvas_shape != fft_shape:
            raise ValueError("matcher pair witness canvas dimensions disagree")
        if self.logical_pairs.size and (
                int(self.logical_pairs[:, 2].max()) >= self.right_grid_shape[0]
                or int(self.logical_pairs[:, 3].max()) >= self.right_grid_shape[1]):
            raise ValueError("matcher pair witness logical pair is outside its grid")
        if self.warp_witness is not None:
            warp = WarpSignatureWitnessV1(**{
                name: getattr(self.warp_witness, name)
                for name in self.warp_witness.__dataclass_fields__})
            if (warp.witness_sha256 != self.warp_witness_sha256
                    or warp.target_shape != shape):
                raise ValueError("matcher warp witness does not bind pair witness")
            offset = np.eye(3, dtype=np.float64)
            offset[:2, 2] = -self.max_shift_cells
            if not np.array_equal(warp.homography, self.homography @ offset):
                raise ValueError("matcher warp witness common-canvas geometry differs")
            if self.logical_pairs.size:
                pairs = self.logical_pairs
                if (int(pairs[:, 0].max()) >= warp.source_shape[0]
                        or int(pairs[:, 1].max()) >= warp.source_shape[1]
                        or int(pairs[:, 2].max()) >= self.right_grid_shape[0]
                        or int(pairs[:, 3].max()) >= self.right_grid_shape[1]):
                    raise ValueError(
                        "matcher pair witness logical pair is outside its grid")
        digest = hashlib.sha256(json.dumps(
            self.to_record(include_digest=False), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if self.witness_sha256 not in ("", digest):
            raise ValueError("matcher pair witness digest mismatch")
        object.__setattr__(self, "witness_sha256", digest)

    def to_record(self, *, include_digest=True):
        def row(value):
            return {"dtype": value.dtype.str, "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        result = {"schema": self.schema, "algorithm": self.algorithm,
                  "max_shift_cells": self.max_shift_cells,
                  "quadrant": self.quadrant, "homography": row(self.homography),
                  "warp_witness_sha256": self.warp_witness_sha256,
                  "canvas_shape": list(self.canvas_shape),
                  "right_grid_shape": list(self.right_grid_shape),
                  "selected_shift_rc": list(self.selected_shift_rc),
                  "selected_support": self.selected_support,
                  "pair_order": "right-grid-row-major-v1",
                  "logical_pairs": row(self.logical_pairs)}
        if include_digest:
            result["witness_sha256"] = self.witness_sha256
        return result


@dataclass(frozen=True, slots=True)
class GraphMatcherWitnessEnvelopeV1:
    schema: str
    match_sha256: str
    left_signature_sha256: str
    right_signature_sha256: str
    left: str
    right: str
    left_plane: int
    right_plane: int
    left_axis: int
    right_axis: int
    quadrant: int
    shift_rc: tuple[int, int]
    overlap_cells: int
    psr: float | str
    score: float
    matcher_witness_sha256: str
    envelope_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != "mc-photogrammetry-graph-matcher-witness-envelope-v1"
                or any(type(value) is not str or len(value) != 64 for value in
                       (self.match_sha256, self.left_signature_sha256,
                        self.right_signature_sha256, self.matcher_witness_sha256))
                or self.overlap_cells < 0):
            raise ValueError("graph matcher witness envelope is malformed")
        digest = hashlib.sha256(json.dumps(
            self.to_record(include_digest=False), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if self.envelope_sha256 not in ("", digest):
            raise ValueError("graph matcher witness envelope digest mismatch")
        object.__setattr__(self, "envelope_sha256", digest)

    def to_record(self, *, include_digest=True):
        result = {name: getattr(self, name) for name in self.__dataclass_fields__
                  if name != "envelope_sha256"}
        result["shift_rc"] = list(self.shift_rc)
        if include_digest:
            result["envelope_sha256"] = self.envelope_sha256
        return result


@dataclass(frozen=True, slots=True)
class GraphMatcherWitnessPayloadV1:
    """Evaluation-only carrier for a final-match envelope and exact pair witness."""
    envelope: GraphMatcherWitnessEnvelopeV1
    witness: MatcherPairWitnessV1

    def __post_init__(self) -> None:
        if (not isinstance(self.envelope, GraphMatcherWitnessEnvelopeV1)
                or not isinstance(self.witness, MatcherPairWitnessV1)):
            raise ValueError("graph matcher witness payload is malformed")
        envelope = GraphMatcherWitnessEnvelopeV1(**{
            name: getattr(self.envelope, name)
            for name in self.envelope.__dataclass_fields__})
        replay = MatcherPairWitnessV1(**{
            name: getattr(self.witness, name)
            for name in self.witness.__dataclass_fields__})
        if (replay.witness_sha256 != envelope.matcher_witness_sha256
                or replay.selected_shift_rc != envelope.shift_rc
                or replay.selected_support != envelope.overlap_cells
                or replay.quadrant != envelope.quadrant):
            raise ValueError("graph matcher witness payload does not bind envelope")


PREFILTER_MATCHER_WITNESS_ROW_SCHEMA = (
    "mc-photogrammetry-pre-filter-matcher-witness-row-v1")

MATCHER_WITNESS_UNAVAILABLE = "matcher_witness_unavailable"


@dataclass(frozen=True, slots=True)
class PreFilterMatcherWitnessRowV1:
    """One sorted pre-filter admitted candidate's own publication.

    The row is emitted for every candidate admitted before
    :func:`_apply_same_axis_filter` -- kept or dropped -- so a dropped
    alternative's already-minted matcher witness (or its own typed refusal) is
    never silently absent and never aliases a survivor's witness. It binds the
    immutable ``admitted_row_index`` (the position after
    ``matches.sort(key=_match_order)``), the keep/drop ``disposition``, the
    post-filter ``post_match_index`` (``None`` when dropped), the exact
    ``match_sha256``, ``envelope_sha256`` and both endpoint signature digests.
    """

    admitted_row_index: int
    disposition: str
    post_match_index: int | None
    match_sha256: str
    envelope_sha256: str | None
    left_signature_sha256: str | None
    right_signature_sha256: str | None
    payload: GraphMatcherWitnessPayloadV1 | None
    refusal: lattice.WitnessRefusalV1 | None = None
    refusal_reason: str | None = None
    schema: str = PREFILTER_MATCHER_WITNESS_ROW_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != PREFILTER_MATCHER_WITNESS_ROW_SCHEMA
                or type(self.admitted_row_index) is not int
                or self.admitted_row_index < 0
                or self.disposition not in (SAME_AXIS_DISPOSITION_KEPT,
                                            SAME_AXIS_DISPOSITION_DROPPED)
                or not _sha256_hex(self.match_sha256)):
            raise ValueError("pre-filter matcher witness row is malformed")
        if self.disposition == SAME_AXIS_DISPOSITION_KEPT:
            if type(self.post_match_index) is not int or self.post_match_index < 0:
                raise ValueError("kept pre-filter row needs a post-filter match index")
        elif self.post_match_index is not None:
            raise ValueError("dropped pre-filter row must not claim a match index")
        if self.payload is not None:
            if (self.refusal is not None or self.refusal_reason is not None
                    or self.envelope_sha256 is None
                    or self.left_signature_sha256 is None
                    or self.right_signature_sha256 is None):
                raise ValueError("pre-filter payload row needs bound digests")
        else:
            if self.refusal is None and self.refusal_reason is None:
                raise ValueError("pre-filter row needs a witness or a typed refusal")
            if self.envelope_sha256 is not None:
                raise ValueError("refused pre-filter row must not claim an envelope")
        if self.refusal is not None:
            if self.refusal_reason != self.refusal.reason:
                raise ValueError("pre-filter refusal row must name its reason")
        elif self.refusal_reason is not None and not self.refusal_reason:
            raise ValueError("pre-filter refusal row must name a bounded reason")

    def to_record(self) -> dict:
        return {
            "schema": self.schema, "admitted_row_index": self.admitted_row_index,
            "disposition": self.disposition, "post_match_index": self.post_match_index,
            "match_sha256": self.match_sha256,
            "envelope_sha256": self.envelope_sha256,
            "left_signature_sha256": self.left_signature_sha256,
            "right_signature_sha256": self.right_signature_sha256,
            "matcher_witness_sha256": (None if self.payload is None
                                       else self.payload.witness.witness_sha256),
            "refusal_reason": self.refusal_reason,
        }


def _sha256_hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _pre_filter_witness_row(
    match: PlaneMatch,
    *,
    admitted_row_index: int,
    filtered_index: dict[int, int],
    candidate_witnesses: dict,
    signature_sha256_by_identity: Mapping[int, str] | None,
) -> PreFilterMatcherWitnessRowV1:
    """Publish one admitted candidate's own witness or its own typed refusal."""
    captured = candidate_witnesses.get(id(match))
    disposition = (SAME_AXIS_DISPOSITION_KEPT if id(match) in filtered_index
                   else SAME_AXIS_DISPOSITION_DROPPED)
    post_match_index = filtered_index.get(id(match))
    match_sha256 = hashlib.sha256(json.dumps(
        _indexed_match_payload(match), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    if captured is None:
        return PreFilterMatcherWitnessRowV1(
            admitted_row_index=admitted_row_index, disposition=disposition,
            post_match_index=post_match_index, match_sha256=match_sha256,
            envelope_sha256=None, left_signature_sha256=None,
            right_signature_sha256=None, payload=None,
            refusal_reason=MATCHER_WITNESS_UNAVAILABLE)
    # ``candidate_witnesses`` stores the minted ``(witness, left, right)`` triple.
    # Unpack it *before* testing for a refusal: the refusal is nested inside the
    # triple, so testing the container would miss it and then attribute the
    # refusal's (absent) fields to the survivor's witness.
    if isinstance(captured, tuple) and len(captured) == 3:
        captured_witness, left_signature, right_signature = captured
    else:
        captured_witness, left_signature, right_signature = captured, None, None
    if isinstance(captured_witness, lattice.WitnessRefusalV1):
        return PreFilterMatcherWitnessRowV1(
            admitted_row_index=admitted_row_index, disposition=disposition,
            post_match_index=post_match_index, match_sha256=match_sha256,
            envelope_sha256=None, left_signature_sha256=None,
            right_signature_sha256=None, payload=None, refusal=captured_witness,
            refusal_reason=captured_witness.reason)
    hashes = signature_sha256_by_identity or {}
    left_hash = hashes.get(id(left_signature))
    right_hash = hashes.get(id(right_signature))
    if (not _sha256_hex(left_hash) or not _sha256_hex(right_hash)
            or captured_witness.selected_support != match.overlap_cells
            or len(captured_witness.logical_pairs) != match.overlap_cells
            or not _logical_pairs_in_grid(captured_witness)):
        return PreFilterMatcherWitnessRowV1(
            admitted_row_index=admitted_row_index, disposition=disposition,
            post_match_index=post_match_index, match_sha256=match_sha256,
            envelope_sha256=None, left_signature_sha256=None,
            right_signature_sha256=None, payload=None,
            refusal_reason=MATCHER_WITNESS_UNAVAILABLE)
    envelope = GraphMatcherWitnessEnvelopeV1(
        "mc-photogrammetry-graph-matcher-witness-envelope-v1", match_sha256,
        left_hash, right_hash, match.left, match.right, match.left_plane,
        match.right_plane, match.left_axis, match.right_axis,
        match.quadrant_of_right, match.cell_shift_rc, match.overlap_cells,
        "positive_infinity" if match.psr == float("inf") else float(match.psr),
        float(match.score), captured_witness.witness_sha256)
    return PreFilterMatcherWitnessRowV1(
        admitted_row_index=admitted_row_index, disposition=disposition,
        post_match_index=post_match_index, match_sha256=match_sha256,
        envelope_sha256=envelope.envelope_sha256,
        left_signature_sha256=left_hash, right_signature_sha256=right_hash,
        payload=GraphMatcherWitnessPayloadV1(envelope, captured_witness))


def _logical_pairs_in_grid(witness: MatcherPairWitnessV1) -> bool:
    """C5: every selected logical pair is a real source/target cell index.

    ``-1`` is the warp sampling map's border sentinel and is *never* a selected
    logical pair; ``0`` is a real cell (not a sentinel). When the witness carries
    its warp grid the pair is also range-checked against that grid.
    """
    pairs = witness.logical_pairs
    if pairs.size == 0:
        return True
    if int(pairs.min()) < 0:
        return False
    warp = witness.warp_witness
    if warp is None:
        return True
    return (int(pairs[:, 0].max()) < warp.source_shape[0]
            and int(pairs[:, 1].max()) < warp.source_shape[1]
            and int(pairs[:, 2].max()) < witness.right_grid_shape[0]
            and int(pairs[:, 3].max()) < witness.right_grid_shape[1])


def _indexed_match_payload(match: PlaneMatch) -> dict:
    psr = "positive_infinity" if match.psr == float("inf") else float(match.psr)
    return {"left": match.left, "right": match.right,
            "left_axis": match.left_axis, "right_axis": match.right_axis,
            "left_b": float(match.left_b), "right_b": float(match.right_b),
            "quadrant": match.quadrant_of_right, "psr": psr,
            "peak": float(match.peak), "second_peak": float(match.second_peak),
            "overlap_cells": match.overlap_cells, "score": float(match.score),
            "left_plane": match.left_plane, "right_plane": match.right_plane,
            "cell_shift_rc": list(match.cell_shift_rc)}


@dataclass(frozen=True, slots=True)
class ComponentAlternative:
    """One quadrant assignment the solver ranked, with its cost."""

    quadrants: tuple[int, ...]
    cost: float
    resultants: tuple[float, ...]
    axis_observations: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ComponentSolution:
    """A component's quadrants and translations, or a typed abstention.

    ``translations`` is anchored at ``LOCAL_FRAME_ORIGIN`` — a declared gauge,
    never a recovered world position. ``reason`` is empty only when all three
    axes are matched and integer-consistent; every other outcome names one of
    :data:`COMPONENT_ABSTENTION_REASONS`. ``scale_known`` is set only by the
    image-measured lattice consistency, never by connectivity.
    """

    frames: tuple[str, ...]
    quadrants: Mapping[str, int]
    translations: Mapping[str, tuple[float, float, float]]
    axis_residuals: Mapping[int, float]
    axis_resultants: Mapping[int, float]
    scale_known: bool
    blocks_per_period: Mapping[tuple[str, int], float]
    reason: str
    alternatives: tuple[ComponentAlternative, ...]


@dataclass(frozen=True, slots=True)
class ClassConstrainedFeatures:
    """The geometry ``class_constrained_bestof`` needs, and nothing else.

    Keypoints are the pixel coordinates of the candidate graph's endpoints; rays
    are unit camera-space directions; the two rotations are the world-to-camera
    matrices of the image-derived estimates. No descriptor, label or frame
    identity is reachable from here.
    """

    left_keypoints: np.ndarray
    right_keypoints: np.ndarray
    left_rays: np.ndarray
    right_rays: np.ndarray
    rotation_left: np.ndarray
    rotation_right: np.ndarray


_AXIS_TRANSFORMS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (1, 1), (2, 1)),
    ((2, 1), (1, 1), (0, -1)),
    ((0, -1), (1, 1), (2, -1)),
    ((2, -1), (1, 1), (0, 1)),
)


def axis_transform(local_axis: int, quadrant: int) -> tuple[int, int]:
    """Map one strict local axis label to ``(world_axis, world_sign)``.

    This literal table is the shared discrete representation of
    ``Ry(-90 degrees * quadrant) * e_local``. Booleans, non-integer values and
    out-of-range labels are rejected; quadrants are not normalized modulo four.
    """
    if type(local_axis) is not int or local_axis not in (0, 1, 2):
        raise ValueError("local_axis must be exactly 0, 1 or 2")
    if type(quadrant) is not int or quadrant not in (0, 1, 2, 3):
        raise ValueError("quadrant must be exactly 0, 1, 2 or 3")
    return _AXIS_TRANSFORMS[quadrant][local_axis]


def cell_grid_homography(
    left_signature: PlaneSignature,
    right_signature: PlaneSignature,
    relative_quadrant: int,
) -> np.ndarray:
    """Map right block-grid ``(column,row)`` indices into the left grid.

    Signature grids are already sampled at one cell per block. Their relative
    coordinate map therefore comes only from the discrete quadrant and the
    declared grid origins; camera projection parameters remain provenance for
    how each signature was measured, not a scale to apply a second time.
    """
    if type(relative_quadrant) is not int or relative_quadrant not in (0, 1, 2, 3):
        raise ValueError("relative_quadrant must be exactly 0, 1, 2 or 3")
    if not isinstance(left_signature, PlaneSignature) or not isinstance(
            right_signature, PlaneSignature):
        raise ValueError("cell grid homography requires two plane signatures")
    left_reading = left_signature.reading
    right_reading = right_signature.reading
    left_axis, right_axis = left_reading.axis, right_reading.axis
    if type(left_axis) is not int or left_axis not in (0, 1, 2):
        raise ValueError("left reading axis must be exactly 0, 1 or 2")
    if type(right_axis) is not int or right_axis not in (0, 1, 2):
        raise ValueError("right reading axis must be exactly 0, 1 or 2")
    mapped_normal, _normal_sign = axis_transform(left_axis, relative_quadrant)
    if mapped_normal != right_axis:
        raise ValueError("right plane normal is incompatible with relative quadrant")

    def _grid_geometry(signature: PlaneSignature, label: str) -> tuple[int, int]:
        try:
            cells = np.asarray(signature.cells)
            interior = np.asarray(signature.interior)
            coverage = np.asarray(signature.coverage)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} signature arrays are malformed") from error
        if (cells.ndim != 2 or cells.size == 0 or interior.shape != cells.shape
                or coverage.shape != cells.shape or coverage.dtype != np.dtype(bool)
                or not np.issubdtype(cells.dtype, np.number)
                or not np.issubdtype(interior.dtype, np.number)
                or not np.isfinite(cells).all() or not np.isfinite(interior).all()):
            raise ValueError(f"{label} signature grids must be finite, non-empty and shape-matched")
        origin = signature.origin_uv
        if (type(origin) is not tuple or len(origin) != 2
                or any(type(value) not in (int, float) for value in origin)
                or not all(np.isfinite(float(value)) for value in origin)):
            raise ValueError(f"{label} signature origin must be a finite pair")
        return cells.shape

    _grid_geometry(left_signature, "left")
    _grid_geometry(right_signature, "right")
    left_tangent_labels = tuple(label for label in range(3) if label != left_axis)
    right_tangent_labels = tuple(label for label in range(3) if label != right_axis)
    permutation = np.zeros((2, 2), dtype=np.float64)
    # H is consumed right-grid -> left-grid, so each column is a right local
    # tangent label and the relative transform identifies its left label.
    for right_index, local_label in enumerate(right_tangent_labels):
        left_label, sign = axis_transform(local_label, relative_quadrant)
        if left_label not in left_tangent_labels:
            raise ValueError("relative quadrant does not bijectively map tangent axes")
        left_index = left_tangent_labels.index(left_label)
        if permutation[left_index, right_index] != 0.0 or sign not in (-1, 1):
            raise ValueError("relative quadrant does not form a signed permutation")
        permutation[left_index, right_index] = float(sign)
    if (not np.all(np.count_nonzero(permutation, axis=0) == 1)
            or not np.all(np.count_nonzero(permutation, axis=1) == 1)
            or not np.isin(permutation, (-1.0, 0.0, 1.0)).all()):
        raise ValueError("relative quadrant does not form a signed permutation")
    left_origin = np.asarray(left_signature.origin_uv, dtype=np.float64)
    right_origin = np.asarray(right_signature.origin_uv, dtype=np.float64)
    translation = permutation @ right_origin - left_origin
    result = np.eye(3, dtype=np.float64)
    result[:2, :2] = permutation
    result[:2, 2] = translation
    return result


@dataclass(frozen=True, slots=True)
class CorrelationSurface:
    algorithm: str
    values: np.ndarray
    support_cells: np.ndarray


def _correlation_surface_with_support(
    first: np.ndarray,
    second: np.ndarray,
    *,
    max_shift_cells: int,
    first_support: np.ndarray | None = None,
    second_support: np.ndarray | None = None,
) -> CorrelationSurface | None:
    """Whitened phase correlation of two real maps over integer cell shifts.

    The maps are mean-subtracted *inside the overlap* and zero elsewhere, padded
    so a shift up to ``max_shift_cells`` is linear rather than wrapped, and
    correlated with the unit-magnitude cross-spectrum. A periodic map produces a
    train of equal peaks and a low peak-to-sidelobe ratio, which is exactly the
    honest outcome for a uniform repeating wall; a unique match produces one.
    """
    if first.shape != second.shape or first.ndim != 2:
        return None
    first_mask = (np.ones(first.shape, dtype=bool) if first_support is None
                  else np.asarray(first_support, dtype=bool))
    second_mask = (np.ones(second.shape, dtype=bool) if second_support is None
                   else np.asarray(second_support, dtype=bool))
    if first_mask.shape != first.shape or second_mask.shape != second.shape:
        return None
    first_values = np.where(first_mask, first, 0.0).astype(np.float64)
    second_values = np.where(second_mask, second, 0.0).astype(np.float64)
    if np.any(first_mask):
        first_values[first_mask] -= float(first_values[first_mask].mean())
    if np.any(second_mask):
        second_values[second_mask] -= float(second_values[second_mask].mean())
    rows, columns = first.shape
    halo = 2 * max_shift_cells
    shape = (rows + 4 * max_shift_cells, columns + 4 * max_shift_cells)
    padded_first = np.zeros(shape, dtype=np.float64)
    padded_second = np.zeros(shape, dtype=np.float64)
    padded_first[halo:halo + rows, halo:halo + columns] = first_values
    padded_second[halo:halo + rows, halo:halo + columns] = second_values
    padded_first_mask = np.zeros(shape, dtype=bool)
    padded_second_mask = np.zeros(shape, dtype=bool)
    padded_first_mask[halo:halo + rows, halo:halo + columns] = first_mask
    padded_second_mask[halo:halo + rows, halo:halo + columns] = second_mask
    spectrum_first = np.fft.rfft2(padded_first)
    spectrum_second = np.fft.rfft2(padded_second)
    cross = spectrum_first * np.conj(spectrum_second)
    magnitude = np.abs(cross)
    safe = magnitude > 1e-12
    cross = np.where(safe, cross / np.where(safe, magnitude, 1.0), 0.0)
    correlation = np.fft.irfft2(cross, s=shape)
    span = 2 * max_shift_cells + 1
    surface = np.empty((span, span), dtype=np.float64)
    supports = np.zeros((span, span), dtype=np.int64)
    for row_shift in range(-max_shift_cells, max_shift_cells + 1):
        for column_shift in range(-max_shift_cells, max_shift_cells + 1):
            surface[row_shift + max_shift_cells, column_shift + max_shift_cells] = \
                correlation[row_shift % shape[0], column_shift % shape[1]]
            supports[row_shift + max_shift_cells,
                     column_shift + max_shift_cells] = np.count_nonzero(
                np.roll(padded_first_mask, (-row_shift, -column_shift), axis=(0, 1))
                & padded_second_mask)
    if not np.all(np.isfinite(surface)):
        return None
    surface.setflags(write=False)
    supports.setflags(write=False)
    return CorrelationSurface(PLANE_CORRELATION_ALGORITHM, surface, supports)


def _correlation_surface(first: np.ndarray, second: np.ndarray, *,
                         max_shift_cells: int) -> np.ndarray | None:
    result = _correlation_surface_with_support(
        first, second, max_shift_cells=max_shift_cells)
    return None if result is None else result.values


def _peak_statistics(surface: np.ndarray, *, max_shift_cells: int,
                     admissible: np.ndarray | None = None
                     ) -> tuple[float, float | None, float, tuple[int, int]]:
    """Peak, peak-to-sidelobe ratio and second peak of a correlation surface.

    ``psr`` is ``None`` when the surface has no admissible sidelobe cell at all.
    With no sample to measure spread against, the ratio is undefined and the
    caller must abstain rather than promote an unmeasured peak to perfect
    evidence. A non-empty sidelobe set whose deviation is numerically zero keeps
    the historical positive-infinity ratio.
    """
    span = surface.shape[0]
    allowed = (np.ones(surface.shape, dtype=bool) if admissible is None
               else np.asarray(admissible, dtype=bool))
    if allowed.shape != surface.shape or not np.any(allowed):
        raise ValueError("correlation surface has no admissible shift")
    ranked = np.where(allowed, surface, -np.inf)
    flat_index = int(np.argmax(ranked))
    peak_row, peak_column = divmod(flat_index, span)
    peak = float(surface[peak_row, peak_column])
    exclusion = 1
    mask = np.ones(surface.shape, dtype=bool)
    mask[max(0, peak_row - exclusion):peak_row + exclusion + 1,
         max(0, peak_column - exclusion):peak_column + exclusion + 1] = False
    mask &= allowed
    sidelobes = surface[mask]
    if sidelobes.size:
        deviation = float(sidelobes.std())
        psr: float | None = float(peak / deviation) if deviation > 1e-12 else float("inf")
        second = float(sidelobes.max())
    else:
        # Never call ``std`` on an empty array: it warns and returns NaN, which
        # the old code then converted into a positive-infinity "perfect" match.
        psr = None
        second = 0.0
    shift = (peak_row - max_shift_cells, peak_column - max_shift_cells)
    return peak, psr, second, shift


def _attrition_signature_identity(signature, plane_index=0):
    alternatives = signature.reading.matching_alternatives(signature.blocks_per_period)
    alternative_index = next((index for index, value in enumerate(signature.reading.alternatives)
                              if len(alternatives) == 1 and value is alternatives[0]), -1)
    return (signature.reading.frame, plane_index, alternative_index,
            float(signature.blocks_per_period))


def _attrition_pair_identity(observer, left, right, quadrant):
    first = observer.bindings.get(("signature", id(left)))
    second = observer.bindings.get(("signature", id(right)))
    if first is None:
        first = _attrition_signature_identity(left)
    if second is None:
        second = _attrition_signature_identity(right)
    return (*first, *second, quadrant)


# Per-pair limits bound the expanded warp, padded FFT, and support search.
MAX_COMMON_CANVAS_CELLS = 1_000_000
MAX_PADDED_FFT_CELLS = 2_000_000
MAX_CORRELATION_SHIFT_POSITIONS = 100_000


def _common_canvas_geometry(right_shape, radius):
    """Validate allocation dimensions before creating any canvas or warp."""
    if (type(radius) is not int or radius < 0
            or len(right_shape) != 2
            or any(type(side) is not int or side <= 0 for side in right_shape)):
        raise ValueError("invalid common-canvas dimensions or radius")
    rows, columns = right_shape
    shape = (rows + 2 * radius, columns + 2 * radius)
    fft_shape = (rows + 6 * radius, columns + 6 * radius)
    if (shape[0] * shape[1] > MAX_COMMON_CANVAS_CELLS
            or fft_shape[0] * fft_shape[1] > MAX_PADDED_FFT_CELLS
            or (2 * radius + 1) ** 2 > MAX_CORRELATION_SHIFT_POSITIONS):
        raise ValueError("common-canvas allocation budget exceeded")
    return shape, fft_shape


def _common_canvas_inputs(left, right, homography, radius, *, warp=None,
                          witness_sink=None, witness_budget=None):
    """Sample reachable left cells and embed real right cells with separate masks.

    Canvas (column,row) coordinates minus (radius,radius) are original right
    coordinates. The original homography remains the provenance geometry.
    """
    shape, _fft_shape = _common_canvas_geometry(tuple(right.cells.shape), radius)
    offset = np.eye(3, dtype=np.float64)
    offset[:2, 2] = -radius
    canvas_homography = homography @ offset
    warp = warp_signature if warp is None else warp
    if witness_sink is None:
        first_cells, first_coverage = warp(left, canvas_homography, shape)
    else:
        first_cells, first_coverage = warp(
            left, canvas_homography, shape, witness_sink=witness_sink,
            witness_budget=witness_budget)
    second_cells = np.zeros(shape, dtype=right.cells.dtype)
    second_coverage = np.zeros(shape, dtype=bool)
    rows, columns = right.cells.shape
    target = np.s_[radius:radius + rows, radius:radius + columns]
    second_cells[target] = right.cells
    second_coverage[target] = right.coverage
    return first_cells, first_coverage, second_cells, second_coverage


# Exact alias checks add no FFT, warp, or correlation pass. These caps bound
# all enumerated positions, mask scans/gathers/comparisons, value work and live
# temporary buffers. Large inconclusive checks abstain rather than admit.
MAX_ALIAS_SHIFT_POSITIONS = MAX_CORRELATION_SHIFT_POSITIONS
MAX_ALIAS_MASK_VISITS = 64_000_000
MAX_ALIAS_VALUE_VISITS = 64_000_000
MAX_ALIAS_TEMP_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _OverlapAliasBudget:
    max_shift_positions: int
    max_mask_visits: int
    max_value_visits: int
    max_temp_bytes: int


@dataclass(frozen=True, slots=True)
class _OverlapAliasCheck:
    status: str
    reason: str
    alias_shift_rc: tuple[int, int] | None
    shift_positions_charged: int
    mask_visits_charged: int
    value_visits_charged: int
    temp_peak_bytes: int


class _AliasBudgetExceeded(Exception):
    pass


def _exact_overlap_alias_check(cells, coverage, right_coverage, admissible,
                               winner, *, support_cells, max_shift_cells, budget=None):
    """Compare complete supported right-cell identities on the existing canvas.

    Right coordinates are enumerated once, in stable row/column order. A shift
    samples left at right+shift without wrap or a second quadrant transform.
    Equality of the full support mask precedes exact finite float32 comparison;
    equal cardinalities or an intersection cannot establish an alias. All
    admissible shifts are examined even after an alias is found. Reuse source
    support counts to skip different-cardinality masks without gathering cells.
    Equal counts still require full identity/value comparisons. Exhaustion
    invalidates completeness and never counts as absence of ambiguity.
    """
    if budget is None:
        budget = _OverlapAliasBudget(
            MAX_ALIAS_SHIFT_POSITIONS, MAX_ALIAS_MASK_VISITS,
            MAX_ALIAS_VALUE_VISITS, MAX_ALIAS_TEMP_BYTES)
    work = dict(shift_positions_charged=0, mask_visits_charged=0,
                value_visits_charged=0, temp_peak_bytes=0)
    alias = None

    def result(status, reason=""):
        return _OverlapAliasCheck(status, reason, alias, **work)

    def charge(name, amount, limit):
        if amount > limit - work[name]:
            raise _AliasBudgetExceeded(name)
        work[name] += amount

    if (type(budget) is not _OverlapAliasBudget or any(
            type(value) is not int or value < 0 for value in (
                budget.max_shift_positions, budget.max_mask_visits,
                budget.max_value_visits, budget.max_temp_bytes))):
        return result("inconclusive", "invalid_budget")
    radius = max_shift_cells
    if (type(radius) is not int or radius < 0
            or type(winner) is not tuple or len(winner) != 2
            or any(type(v) is not int or abs(v) > radius for v in winner)
            or any(type(a) is not np.ndarray or a.ndim != 2
                   for a in (cells, coverage, right_coverage, admissible, support_cells))
            or cells.dtype != np.dtype(np.float32)
            or support_cells.dtype != np.dtype(np.int64)
            or support_cells.shape != admissible.shape
            or any(a.dtype != np.dtype(bool) for a in (coverage, right_coverage, admissible))
            or cells.shape != coverage.shape or cells.shape != right_coverage.shape
            or admissible.shape != (2 * radius + 1, 2 * radius + 1)
            or not admissible[winner[0] + radius, winner[1] + radius]):
        return result("inconclusive", "invalid_alias_inputs")
    try:
        charge("shift_positions_charged", admissible.size, budget.max_shift_positions)
        # Count/nonzero scans are charged before allocating coordinate arrays.
        charge("mask_visits_charged", 2 * right_coverage.size, budget.max_mask_visits)
        count = int(np.count_nonzero(right_coverage))
        # Conservative live-buffer bound: coordinates, shifted indices, two
        # masks/value vectors and filtered equality/finite-check temporaries.
        required_bytes = 64 * count + 4096
        if required_bytes > budget.max_temp_bytes:
            raise _AliasBudgetExceeded("temp_peak_bytes")
        work["temp_peak_bytes"] = required_bytes
        if count == 0:
            return result("inconclusive", "empty_right_support")
        rr, cc = np.nonzero(right_coverage)
        charge("mask_visits_charged", 4 * count, budget.max_mask_visits)
        if (int(rr.min()) < radius or int(cc.min()) < radius
                or int(rr.max()) >= cells.shape[0] - radius
                or int(cc.max()) >= cells.shape[1] - radius):
            return result("inconclusive", "right_support_outside_halo")
        charge("mask_visits_charged", 2 * count, budget.max_mask_visits)
        winner_mask = coverage[rr + winner[0], cc + winner[1]]
        shared = int(np.count_nonzero(winner_mask))
        if shared == 0:
            return result("inconclusive", "empty_winner_support")
        if int(support_cells[winner[0] + radius, winner[1] + radius]) != shared:
            return result("inconclusive", "winner_support_count_mismatch")
        charge("value_visits_charged", count + 3 * shared, budget.max_value_visits)
        charge("mask_visits_charged", count, budget.max_mask_visits)
        winner_values = cells[rr + winner[0], cc + winner[1]]
        if not np.isfinite(winner_values[winner_mask]).all():
            return result("inconclusive", "nonfinite_supported_value")
        for row in range(admissible.shape[0]):
            for column in range(admissible.shape[1]):
                shift = (row - radius, column - radius)
                source_count = int(support_cells[row, column])
                if not 0 <= source_count <= count:
                    return result("inconclusive", "invalid_source_support_count")
                if (not admissible[row, column] or shift == winner
                        or source_count != shared):
                    continue
                # Cardinality is only a rejection prefilter, never alias proof.
                # Equal-count shifts still construct/compare their whole mask.
                charge("mask_visits_charged", 4 * count, budget.max_mask_visits)
                candidate_mask = coverage[rr + shift[0], cc + shift[1]]
                if not np.array_equal(candidate_mask, winner_mask):
                    continue
                charge("value_visits_charged", count + 8 * shared, budget.max_value_visits)
                charge("mask_visits_charged", 3 * count, budget.max_mask_visits)
                candidate_values = cells[rr + shift[0], cc + shift[1]]
                if not np.isfinite(candidate_values[winner_mask]).all():
                    return result("inconclusive", "nonfinite_supported_value")
                if (np.array_equal(candidate_values[winner_mask], winner_values[winner_mask])
                        and alias is None):
                    alias = shift
        return result("alias" if alias is not None else "complete")
    except _AliasBudgetExceeded as error:
        # No partial alias/absence conclusion escapes a budget-limited check.
        alias = None
        return result("inconclusive", str(error))


def match_plane_pair(
    left: PlaneSignature,
    right: PlaneSignature,
    *,
    quadrant: int,
    max_shift_cells: int = 32,
    refusals: list | None = None,
    diagnostic_sink: PlaneGraphTrace | None = None,
    witness_sink=None,
    witness_budget: lattice.WitnessBudgetV1 | None = None,
) -> PlaneMatch | None:
    """Correlate two signatures as the same plane under one quadrant hypothesis.

    Returns ``None`` — with a typed :class:`PlaneMatchRefusal` appended to
    ``refusals`` when given — for an axis that does not map under this quadrant,
    too little overlap, no signal, or the repetition gate
    (``min(structural_share) < 0.15`` and ``psr < 3.0``). The gate is the explicit,
    honest treatment of Minecraft's repetition: a uniform wall must abstain
    rather than emit a chicken-wire peak.
    """
    left_reading = left.reading
    right_reading = right.reading
    raw_stats = supported_stats = None
    admitted_metrics = None
    observer = region_growth._attrition_state.get()
    observation_id = None
    subcause = None
    signal_failure_operand = None
    left_covered_cells = right_covered_cells = None
    left_standard_deviation = right_standard_deviation = None
    maximum_overlap = maximum_overlap_count = None
    if observer is not None:
        with region_growth._scalar_observation_guard(observer):
            observation_id = _attrition_pair_identity(observer, left, right, quadrant)
            if ("defer", observation_id) not in observer.bindings:
                observer.expect("pair", observation_id)
    def trace(reason: str, admitted: bool = False) -> None:
        if observer is not None:
            with region_growth._scalar_observation_guard(observer):
                fields = dict(axis=int(left_reading.axis), right_axis=int(right_reading.axis),
                              subcause=subcause, maximum_searched_overlap=maximum_overlap,
                              maximum_overlap_shift_count=maximum_overlap_count,
                              signal_failure_operand=signal_failure_operand,
                              left_warped_covered_cells=(int(left_covered_cells)
                                                         if left_covered_cells is not None else None),
                              right_covered_cells=(int(right_covered_cells)
                                                   if right_covered_cells is not None else None),
                              left_warped_standard_deviation=left_standard_deviation,
                              right_standard_deviation=right_standard_deviation,
                              left_warped_covered_cells_state=("measured" if left_covered_cells is not None
                                                               else "not_evaluated"),
                              right_covered_cells_state=("measured" if right_covered_cells is not None
                                                         else "not_evaluated"),
                              left_warped_standard_deviation_state=("measured" if left_standard_deviation is not None
                                                                    else "not_evaluated"),
                              right_standard_deviation_state=("measured" if right_standard_deviation is not None
                                                              else "not_evaluated"))
                if admitted and ("defer", observation_id) in observer.bindings:
                    observer.bind(("pair_metrics", observation_id), fields)
                else:
                    observer.finish("pair", observation_id, "admitted" if admitted else reason, **fields)
        if diagnostic_sink is None:
            return
        def fields(stats):
            if stats is None:
                return (None, None, None)
            value, shift, support = stats
            return (shift, float(value), int(support))
        raw_shift, raw_value, raw_support = fields(raw_stats)
        supported_shift, supported_value, supported_support = fields(supported_stats)
        metrics = admitted_metrics or (None,) * 8
        diagnostic_sink.pairs.append(PlanePairTrace(
            left_reading.frame, right_reading.frame, left_reading.axis,
            right_reading.axis, quadrant, admitted, reason,
            raw_shift, raw_value, raw_support,
            supported_shift, supported_value, supported_support,
            psr=metrics[0], score=metrics[1], overlap_cells=metrics[2],
            valid_cells=metrics[3], overlap_fraction=metrics[4],
            cell_shift_rc=metrics[5], left_structural_share=metrics[6],
            right_structural_share=metrics[7]))
    expected_right_axis, _left_sign = axis_transform(left_reading.axis, quadrant)
    if right_reading.axis != expected_right_axis:
        _refuse(refusals, left_reading, right_reading, quadrant, "axis_mismatch",
                f"left axis {left_reading.axis} maps to {expected_right_axis} under "
                f"quadrant {quadrant}, right reading is axis {right_reading.axis}")
        trace("axis_mismatch")
        return None

    if (left_reading.distance_for(left.blocks_per_period) is None
            or right_reading.distance_for(right.blocks_per_period) is None):
        _refuse(refusals, left_reading, right_reading, quadrant,
                "invalid_distance_alternative",
                "signature b must identify exactly one valid retained alternative")
        trace("invalid_distance_alternative")
        return None
    try:
        homography = cell_grid_homography(left, right, quadrant)
    except (TypeError, ValueError, OverflowError) as error:
        subcause = "invalid_grid_homography"
        _refuse(refusals, left_reading, right_reading, quadrant, "no_signal",
                f"invalid block-grid geometry: {error}")
        trace("no_signal")
        return None
    warp_witnesses = [] if witness_sink is not None else None
    try:
        warped_cells, warped_coverage, right_cells, right_coverage = _common_canvas_inputs(
            left, right, homography, max_shift_cells,
            witness_sink=warp_witnesses, witness_budget=witness_budget)
    except (TypeError, ValueError, OverflowError) as error:
        subcause = "common_canvas_bounds"
        _refuse(refusals, left_reading, right_reading, quadrant, "no_signal",
                f"invalid or excessive common canvas: {error}")
        trace("no_signal")
        return None
    first = np.where(warped_coverage, warped_cells, 0.0).astype(np.float64)
    second = np.where(right_coverage, right_cells, 0.0).astype(np.float64)
    if ((left_covered_cells := np.count_nonzero(warped_coverage)) < 2
            or (right_covered_cells := np.count_nonzero(right_coverage)) < 2
            or (left_standard_deviation := float(np.std(first[warped_coverage]))) <= 1e-12
            or (right_standard_deviation := float(np.std(second[right_coverage]))) <= 1e-12):
        subcause = ("fewer_than_two_covered_cells" if left_covered_cells < 2
                    or right_covered_cells < 2 else "constant_covered_energy")
        if left_covered_cells < 2 or (subcause == "constant_covered_energy"
                                     and left_standard_deviation <= 1e-12):
            signal_failure_operand = "left_warped"
        else:
            signal_failure_operand = "right"
        _refuse(refusals, left_reading, right_reading, quadrant, "no_signal",
                "one of the seam-band maps is empty")
        trace("no_signal")
        return None
    correlation = _correlation_surface_with_support(
        first, second, max_shift_cells=int(max_shift_cells),
        first_support=warped_coverage, second_support=right_coverage)
    if correlation is None:
        subcause = "correlation_unavailable"
        _refuse(refusals, left_reading, right_reading, quadrant, "no_signal",
                "correlation surface was not finite")
        trace("no_signal")
        return None
    raw_index = int(np.argmax(correlation.values))
    raw_position = divmod(raw_index, correlation.values.shape[1])
    raw_shift = (raw_position[0] - int(max_shift_cells),
                 raw_position[1] - int(max_shift_cells))
    raw_stats = (correlation.values[raw_position], raw_shift,
                 correlation.support_cells[raw_position])
    admissible = correlation.support_cells >= MIN_OVERLAP_CELLS
    if not np.any(admissible):
        maximum = int(correlation.support_cells.max())
        if observer is not None:
            with region_growth._scalar_observation_guard(observer):
                maximum_overlap = maximum
                maximum_overlap_count = int(np.count_nonzero(correlation.support_cells == maximum))
        _refuse(refusals, left_reading, right_reading, quadrant, "insufficient_overlap",
                f"only {maximum} cells of overlap at the best-supported shift")
        trace("insufficient_overlap")
        return None
    peak, psr, second_peak, cell_shift_rc = _peak_statistics(
        correlation.values, max_shift_cells=int(max_shift_cells),
        admissible=admissible)
    overlap_cells = int(correlation.support_cells[
        cell_shift_rc[0] + int(max_shift_cells),
        cell_shift_rc[1] + int(max_shift_cells)])
    supported_stats = (peak, cell_shift_rc, overlap_cells)

    share = min(float(left_reading.structural_share), float(right_reading.structural_share))
    if psr is None:
        # No admissible cell outside the peak's exclusion window: there is no
        # sidelobe sample, so the peak-to-sidelobe ratio is genuinely undefined.
        # An unmeasured peak is not evidence of a unique match.
        _refuse(refusals, left_reading, right_reading, quadrant, "repetition_unresolved",
                "no admissible sidelobe cells outside the peak: peak-to-sidelobe "
                "ratio is undefined")
        trace("repetition_unresolved")
        return None
    if share < MIN_STRUCTURAL_SHARE:
        # No edge energy off the plane's own lattice: a uniform repeating wall is
        # genuinely ambiguous, and two views of it correlate *well* precisely
        # because they are the same pattern. Identity is refused on the absence
        # of non-repeating evidence, not on the peak.
        _refuse(refusals, left_reading, right_reading, quadrant, "repetition_unresolved",
                f"structural share {share:.3f} below {MIN_STRUCTURAL_SHARE}")
        trace("repetition_unresolved")
        return None
    if psr < 3.0:
        _refuse(refusals, left_reading, right_reading, quadrant, "repetition_unresolved",
                f"no unique correlation peak (psr {psr:.3f})")
        trace("repetition_unresolved")
        return None

    alias_check = _exact_overlap_alias_check(
        warped_cells, warped_coverage, right_coverage, admissible, cell_shift_rc,
        support_cells=correlation.support_cells, max_shift_cells=max_shift_cells)
    if alias_check.status == "inconclusive":
        subcause = "overlap_alias_" + alias_check.reason
        _refuse(refusals, left_reading, right_reading, quadrant,
                "ambiguity_check_inconclusive",
                "exact overlap alias check inconclusive: " + alias_check.reason)
        trace("ambiguity_check_inconclusive")
        return None
    if alias_check.status == "alias":
        subcause = "exact_supported_overlap_alias"
        _refuse(refusals, left_reading, right_reading, quadrant,
                "repetition_unresolved", "distinct shifts have identical supported cell values")
        trace("repetition_unresolved")
        return None

    valid_cells = min(int(np.count_nonzero(warped_coverage)), int(np.count_nonzero(right_coverage)))
    overlap_fraction = overlap_cells / valid_cells if valid_cells else 0.0
    margin = (peak - second_peak) / peak if peak > 1e-12 else 0.0
    score = (0.5 * min(psr / PSR_SCALE, 1.0)
             + 0.3 * overlap_fraction
             + 0.2 * float(np.clip(margin, 0.0, 1.0))
             - 0.5 * (1.0 - share))
    admitted_metrics = (
        float(psr), float(score), overlap_cells, valid_cells,
        float(overlap_fraction), cell_shift_rc,
        float(left_reading.structural_share), float(right_reading.structural_share))
    trace("", True)
    result = PlaneMatch(
        left=left_reading.frame,
        right=right_reading.frame,
        left_axis=int(left_reading.axis),
        right_axis=int(right_reading.axis),
        left_b=float(left.blocks_per_period),
        right_b=float(right.blocks_per_period),
        quadrant_of_right=int(quadrant),
        psr=float(psr),
        peak=float(peak),
        second_peak=float(second_peak),
        overlap_cells=overlap_cells,
        integer_residual=None,
        score=float(score),
        left_plane=-1,
        right_plane=-1,
        cell_shift_rc=cell_shift_rc,
    )
    if witness_sink is not None:
        if (len(warp_witnesses) != 1
                or isinstance(warp_witnesses[0], lattice.WitnessRefusalV1)):
            if warp_witnesses:
                witness_sink.append(warp_witnesses[0])
        else:
            warp_witness = warp_witnesses[0]
            budget = witness_budget or lattice.WitnessBudgetV1()
            row_shift, column_shift = cell_shift_rc
            pairs = []
            rows, columns = right.coverage.shape
            for right_row in range(rows):
                for right_column in range(columns):
                    left_row = right_row + max_shift_cells + row_shift
                    left_column = right_column + max_shift_cells + column_shift
                    if (0 <= left_row < warped_coverage.shape[0]
                            and 0 <= left_column < warped_coverage.shape[1]
                            and warped_coverage[left_row, left_column]
                            and right.coverage[right_row, right_column]):
                        pairs.append((
                            int(warp_witness.source_rows[left_row, left_column]),
                            int(warp_witness.source_columns[left_row, left_column]),
                            right_row, right_column))
            planned = len(pairs)
            payload = planned * 4 * 8 + 9 * 8
            exhaustion = (("matcher_pair_entries", budget.max_pair_entries, planned)
                          if planned > budget.max_pair_entries else
                          (("matcher_pair_payload_bytes", budget.max_payload_bytes,
                            payload) if payload > budget.max_payload_bytes else None))
            if exhaustion is not None:
                stage, limit, planned_value = exhaustion
                witness_sink.append(lattice.WitnessRefusalV1(
                    "refused", "budget_exhausted", stage, limit, planned_value, 0))
            elif planned == overlap_cells:
                pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 4)
                pair_array.flags.writeable = False
                frozen_homography = np.asarray(homography, dtype=np.float64).copy()
                frozen_homography.flags.writeable = False
                witness_sink.append(MatcherPairWitnessV1(
                    "mc-photogrammetry-matcher-pair-witness-v1",
                    PLANE_CORRELATION_ALGORITHM, int(max_shift_cells), int(quadrant),
                    frozen_homography, warp_witness.witness_sha256,
                    (rows + 6 * int(max_shift_cells),
                     columns + 6 * int(max_shift_cells)),
                    cell_shift_rc, overlap_cells, pair_array,
                    warp_witness=warp_witness, right_grid_shape=right.cells.shape))
    capture = lattice._period_support_state()
    if capture is not None and capture.context.get("match_target"):
        def observe_pair(state):
            warp = state.warp
            if warp is None:
                raise lattice._PeriodSupportIncomplete("missing_dependency", "matcher_warp")
            row_shift, column_shift = cell_shift_rc
            rows, columns = right.coverage.shape
            state.memory.reserve(512 * rows * columns + 1024**2,
                                 "matcher_pair_lists_and_buffers")
            pairs = []
            for right_row in range(rows):
                for right_column in range(columns):
                    left_row = right_row + max_shift_cells + row_shift
                    left_column = right_column + max_shift_cells + column_shift
                    if (0 <= left_row < warped_coverage.shape[0]
                            and 0 <= left_column < warped_coverage.shape[1]
                            and warped_coverage[left_row, left_column]
                            and right.coverage[right_row, right_column]):
                        pairs.append((int(warp.source_rows[left_row, left_column]),
                                      int(warp.source_columns[left_row, left_column]),
                                      right_row, right_column))
            if len(pairs) != overlap_cells:
                raise lattice._PeriodSupportIncomplete("identity_drift", "matcher_support")
            if len(pairs) > 1048576:
                raise lattice._PeriodSupportIncomplete("cap_exhausted", "matcher_pairs")
            pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 4)
            pair_array = np.frombuffer(pair_array.tobytes(), dtype=np.int64).reshape(-1, 4)
            matrix = np.frombuffer(np.asarray(homography, dtype=np.float64).tobytes(), dtype=np.float64).reshape(3, 3)
            witness = MatcherPairWitnessV1(
                "mc-photogrammetry-matcher-pair-witness-v1", PLANE_CORRELATION_ALGORITHM,
                int(max_shift_cells), int(quadrant), matrix, warp.witness_sha256,
                (rows + 6 * int(max_shift_cells), columns + 6 * int(max_shift_cells)),
                cell_shift_rc, overlap_cells, pair_array, warp_witness=warp, right_grid_shape=right.cells.shape)
            lattice._period_support_event("matcher_pair", match=result,
                metadata=witness.to_record(), homography=matrix, logical_pairs=pair_array,
                right_coverage=right.coverage, warped_coverage=warped_coverage,
                left_signature=left, right_signature=right)
        lattice._period_support_private(observe_pair)
    return result


def _refuse(refusals: list | None, left: PlaneReading, right: PlaneReading,
            quadrant: int, reason: str, detail: str) -> None:
    if refusals is None:
        return
    refusals.append(PlaneMatchRefusal(
        left=left.frame, right=right.frame, left_axis=int(left.axis),
        right_axis=int(right.axis), quadrant_of_right=int(quadrant),
        reason=reason, detail=detail))


def match_plane_graph(
    readings: Sequence[PlaneReading],
    signatures: Sequence[PlaneSignature],
    *,
    max_shift_cells: int = 32,
    max_structural_share: float | None = None,
    refusals: list | None = None,
    diagnostic_sink: PlaneGraphTrace | None = None,
    witness_sink=None,
    witness_payload_sink=None,
    witness_budget: lattice.WitnessBudgetV1 | None = None,
    signature_sha256_by_identity: Mapping[int, str] | None = None,
    candidate_observation_sink: list | None = None,
    pre_filter_witness_sink: list | None = None,
    admitted_candidate_sink: list | None = None,
) -> tuple[PlaneMatch, ...]:
    """Every candidate plane match over a complete reading universe.

    Signatures are grouped by frame and paired left/right by frame id order.
    ``left_plane``/``right_plane`` index a reading within its frame's ordered
    complete-reading tuple, so :func:`solve_component` resolves the exact same
    plane objects. Signatureless readings remain in this canonical index space.
    The same-axis equality is applied as a consistency filter after correlation.

    ``max_structural_share`` is accepted for interface symmetry with the reading
    stage and, when given, only matches whose two readings both clear it are
    considered; the default is the module constant.

    ``candidate_observation_sink`` is an optional evaluation-only list. When
    given, exactly one :class:`SameAxisCandidateObservation` is appended after
    the equality filter: the complete pre-filter admitted set with each
    candidate's typed kept/dropped verdict. It is never part of the returned
    match tuple, and the default path retains and allocates nothing.
    """
    if witness_payload_sink is not None and len(witness_payload_sink):
        raise ValueError("matcher witness payload sink must be empty")
    if admitted_candidate_sink is not None and len(admitted_candidate_sink):
        raise ValueError("admitted candidate sink must be empty")
    if pre_filter_witness_sink is not None and len(pre_filter_witness_sink):
        raise ValueError("pre-filter matcher witness sink must be empty")
    threshold = MIN_STRUCTURAL_SHARE if max_structural_share is None else float(max_structural_share)
    refusals = [] if refusals is None else refusals
    read_by_index, index_by_object, frames = _canonical_reading_index(readings)
    if not isinstance(signatures, Sequence):
        raise ValueError("signature inputs must be a sequence")
    by_frame: dict[str, list[PlaneSignature]] = {}
    for signature in signatures:
        if not isinstance(signature, PlaneSignature):
            raise ValueError("signature inputs contain a non-signature value")
        signature_reading = getattr(signature, "reading", None)
        if not isinstance(signature_reading, PlaneReading):
            raise ValueError("signature does not name a valid reading")
        if not _is_finite_positive_number(signature.blocks_per_period):
            raise ValueError("signature b must be a finite positive canonical number")
        index = index_by_object.get(id(signature_reading))
        if index is None or read_by_index.get(index) is not signature_reading:
            raise ValueError("signature reading is foreign to the complete universe")
        by_frame.setdefault(signature.reading.frame, []).append(signature)

    frames = sorted(by_frame)
    observer = region_growth._attrition_state.get()
    if observer is not None:
        with region_growth._scalar_observation_guard(observer):
            accepted_readings = {tuple(identity) for key, identity in observer.bindings.items()
                                 if key[0] == "reading"}
            observer.check("reading_universe", accepted_readings == set(read_by_index))
            universe = []
            for signature in signatures:
                identity = _attrition_signature_identity(
                    signature, index_by_object[id(signature.reading)][1])
                universe.append(identity)
                observer.bind(("signature", id(signature)), identity)
            if "signature" in observer.stages:
                built = {key for key, disposition in observer._stage("signature")["terminal"].items()
                         if disposition == "built"}
                observer.check("signature_universe", built == {observer._encode(value).decode() for value in universe}
                               and len(universe) == len(set(universe)))
            else:
                observer.check("signature_universe", False)
    matches: list[PlaneMatch] = []
    source_readings: dict[int, tuple[PlaneReading, PlaneReading]] = {}
    candidate_witnesses = {}
    for left_position, left_frame in enumerate(frames):
        for right_frame in frames[left_position + 1:]:
            for left_signature in by_frame[left_frame]:
                for right_signature in by_frame[right_frame]:
                    if observer is not None:
                        with region_growth._scalar_observation_guard(observer):
                            for quadrant in range(4):
                                identity = _attrition_pair_identity(observer, left_signature, right_signature, quadrant)
                                observer.expect("pair", identity)
                                observer.bind(("defer", identity), identity)
                    if (left_signature.reading.structural_share < threshold
                            and right_signature.reading.structural_share < threshold):
                        if observer is not None:
                            with region_growth._scalar_observation_guard(observer):
                                for quadrant in range(4):
                                    identity = _attrition_pair_identity(observer, left_signature, right_signature, quadrant)
                                    observer.finish("pair", identity, "both_low_share_pre_trace_skip",
                                                    axis=int(left_signature.reading.axis),
                                                    right_axis=int(right_signature.reading.axis),
                                                    left_structural_share=float(left_signature.reading.structural_share),
                                                    right_structural_share=float(right_signature.reading.structural_share))
                        continue
                    for quadrant in range(4):
                        if diagnostic_sink is not None:
                            diagnostic_sink.pre_admission_candidates += 1
                        pair_kwargs = {"quadrant": quadrant,
                                       "max_shift_cells": max_shift_cells,
                                       "refusals": refusals}
                        if diagnostic_sink is not None:
                            pair_kwargs["diagnostic_sink"] = diagnostic_sink
                        pair_witnesses = [] if (witness_sink is not None or witness_payload_sink is not None or pre_filter_witness_sink is not None) else None
                        if pair_witnesses is not None:
                            pair_kwargs["witness_sink"] = pair_witnesses
                            pair_kwargs["witness_budget"] = witness_budget
                        trace_index = (None if diagnostic_sink is None
                                       else len(diagnostic_sink.pairs))
                        capture_identity = (
                            left_frame, index_by_object[id(left_signature.reading)][1],
                            int(left_signature.reading.axis), float(left_signature.blocks_per_period),
                            right_frame, index_by_object[id(right_signature.reading)][1],
                            int(right_signature.reading.axis), float(right_signature.blocks_per_period), quadrant)
                        capture_target = lattice._period_support_match(capture_identity)
                        capture = lattice._period_support_state()
                        if capture_target:
                            capture.warp = None
                        try:
                            with lattice._period_support_context(match_target=capture_target, match_identity=capture_identity):
                                found = match_plane_pair(
                                    left_signature, right_signature, **pair_kwargs)
                        finally:
                            if capture_target:
                                capture.warp = None
                        if found is None:
                            continue
                        match = PlaneMatch(
                            left=found.left, right=found.right,
                            left_axis=found.left_axis, right_axis=found.right_axis,
                            left_b=found.left_b, right_b=found.right_b,
                            quadrant_of_right=found.quadrant_of_right, psr=found.psr,
                            peak=found.peak, second_peak=found.second_peak,
                            overlap_cells=found.overlap_cells,
                            integer_residual=None, score=found.score,
                            left_plane=index_by_object[id(left_signature.reading)][1],
                            right_plane=index_by_object[id(right_signature.reading)][1],
                            cell_shift_rc=found.cell_shift_rc,
                        )
                        _validate_indexed_match(
                            match, read_by_index,
                            expected_left=left_signature.reading,
                            expected_right=right_signature.reading,
                            require_selected_b=True)
                        if admitted_candidate_sink is not None and len(matches) >= MAX_ADMITTED_CANDIDATES:
                            raise ValueError("admitted candidate universe exceeds endpoint ceiling")
                        matches.append(match)
                        if observer is not None:
                            with region_growth._scalar_observation_guard(observer):
                                observer.bind(("match", id(match)), _attrition_pair_identity(
                                    observer, left_signature, right_signature, quadrant))
                        if diagnostic_sink is not None:
                            diagnostic_sink.post_admission_candidates += 1
                            row = diagnostic_sink.pairs[trace_index]
                            diagnostic_sink.pairs[trace_index] = replace(
                                row, left_plane=match.left_plane,
                                right_plane=match.right_plane)
                        source_readings[id(match)] = (
                            left_signature.reading, right_signature.reading)
                        if pair_witnesses is not None:
                            if len(pair_witnesses) != 1:
                                raise ValueError("matcher witness publication is incomplete")
                            candidate_witnesses[id(match)] = (
                                pair_witnesses[0], left_signature, right_signature)

    if admitted_candidate_sink is not None and len(matches) > MAX_ADMITTED_CANDIDATES:
        raise ValueError("admitted candidate universe exceeds endpoint ceiling")
    matches.sort(key=_match_order)
    filtered = _apply_same_axis_filter(matches, read_by_index, refusals)
    if admitted_candidate_sink is not None:
        kept_ids = {id(row) for row in filtered}
        admitted_candidate_sink.append(AdmittedCandidateUniverse(
            tuple(matches), tuple(SAME_AXIS_DISPOSITION_KEPT if id(row) in kept_ids
                                  else SAME_AXIS_DISPOSITION_DROPPED for row in matches),
            tuple(range(len(matches)))))
    if observer is not None:
        with region_growth._scalar_observation_guard(observer):
            kept_ids = {id(match) for match in filtered}
            for match in matches:
                identity = observer.bindings.get(("match", id(match)))
                metrics = observer.bindings.get(("pair_metrics", identity), {})
                observer.check("pair_admission_traces",
                               observer.checks.get("pair_admission_traces", True) and bool(metrics))
                observer.finish("pair", identity, "selected_admitted" if id(match) in kept_ids
                                else "same_axis_dropped", **metrics)
            expected_pairs = 4 * sum(len(by_frame[first]) * len(by_frame[second])
                for index, first in enumerate(frames) for second in frames[index + 1:])
            observer.check("pair_product", len(observer._stage("pair")["expected"]) == expected_pairs)
    if diagnostic_sink is not None:
        kept_positions = {id(match): index for index, match in enumerate(filtered)}
        diagnostic_sink.admitted_ledger_sha256 = admitted_candidate_ledger_sha256(
            (index, SAME_AXIS_DISPOSITION_KEPT if id(match) in kept_positions
             else SAME_AXIS_DISPOSITION_DROPPED, kept_positions.get(id(match)),
             hashlib.sha256(json.dumps(
                 _indexed_match_payload(match), sort_keys=True,
                 separators=(",", ":"), allow_nan=False).encode()).hexdigest())
            for index, match in enumerate(matches))
    if candidate_observation_sink is not None:
        # The filter only ever keeps or drops whole ``PlaneMatch`` objects taken
        # from ``matches``, so object identity is the exact kept/dropped
        # disposition. Deriving it here keeps ``_apply_same_axis_filter``'s
        # long-standing three-argument signature (and any monkeypatched
        # replacement) unchanged.
        kept_ids = {id(match) for match in filtered}
        dispositions = tuple(
            SAME_AXIS_DISPOSITION_KEPT if id(match) in kept_ids
            else SAME_AXIS_DISPOSITION_DROPPED for match in matches)
        candidate_observation_sink.append(
            SameAxisCandidateObservation(tuple(matches), dispositions))
    if diagnostic_sink is not None:
        diagnostic_sink.post_same_axis_candidates = len(filtered)
        selected = {(match.left, match.right, match.left_axis, match.right_axis,
                     match.left_plane, match.right_plane,
                     match.quadrant_of_right, match.score) for match in filtered}
        diagnostic_sink.pairs[:] = [
            replace(row, selected=True)
            if (row.admitted and row.left_plane is not None
                and (row.left, row.right, row.left_axis, row.right_axis,
                     row.left_plane, row.right_plane, row.quadrant,
                     row.score) in selected)
            else row for row in diagnostic_sink.pairs]
    for match in filtered:
        expected_left, expected_right = source_readings[id(match)]
        _validate_indexed_match(
            match, read_by_index, expected_left=expected_left,
            expected_right=expected_right, require_selected_b=True)
        if witness_sink is not None or witness_payload_sink is not None:
            captured, left_signature, right_signature = candidate_witnesses[id(match)]
            if isinstance(captured, lattice.WitnessRefusalV1):
                if witness_sink is not None:
                    witness_sink.append(captured)
                if witness_payload_sink is not None:
                    witness_payload_sink.append(captured)
                continue
            hashes = signature_sha256_by_identity or {}
            left_hash, right_hash = hashes.get(id(left_signature)), hashes.get(id(right_signature))
            if (type(left_hash) is not str or len(left_hash) != 64
                    or type(right_hash) is not str or len(right_hash) != 64
                    or captured.selected_support != match.overlap_cells
                    or len(captured.logical_pairs) != match.overlap_cells):
                raise ValueError("matcher witness does not bind final match")
            payload = _indexed_match_payload(match)
            envelope = GraphMatcherWitnessEnvelopeV1(
                "mc-photogrammetry-graph-matcher-witness-envelope-v1",
                hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest(),
                left_hash, right_hash, match.left, match.right,
                match.left_plane, match.right_plane, match.left_axis, match.right_axis,
                match.quadrant_of_right, match.cell_shift_rc, match.overlap_cells,
                "positive_infinity" if match.psr == float("inf") else float(match.psr),
                float(match.score), captured.witness_sha256)
            if witness_sink is not None:
                witness_sink.append(envelope)
            if witness_payload_sink is not None:
                witness_payload_sink.append(
                    GraphMatcherWitnessPayloadV1(envelope, captured))
    if pre_filter_witness_sink is not None:
        # DS-IMG-CANDIDATE-SUPPORT-PROVENANCE-IMPL-R4: full-universe publication
        # over the sorted pre-filter admitted rows. Every admitted candidate --
        # kept or dropped -- emits its own already-minted witness/refusal bound to
        # its immutable row index and disposition; the default call never
        # allocates, collects or appends anything here.
        filtered_index = {id(match): index for index, match in enumerate(filtered)}
        for admitted_row_index, match in enumerate(matches):
            pre_filter_witness_sink.append(_pre_filter_witness_row(
                match, admitted_row_index=admitted_row_index,
                filtered_index=filtered_index,
                candidate_witnesses=candidate_witnesses,
                signature_sha256_by_identity=signature_sha256_by_identity))
    lattice._period_support_event("matcher_final", readings=tuple(readings),
        indexed_readings=tuple((identity, reading) for identity, reading in read_by_index.items()),
        matches=tuple(filtered), refusals=tuple(refusals))
    return tuple(filtered)


def admitted_candidate_ledger_sha256(rows) -> str:
    """Canonical full admitted-row commitment minted at the matcher seam."""
    return hashlib.sha256(json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _match_order(match: PlaneMatch) -> tuple:
    return (match.left, match.right, match.left_axis, match.right_axis,
            match.left_plane, match.right_plane, match.left_b, match.right_b,
            match.quadrant_of_right, -match.score)


def _delta(match: PlaneMatch, readings: Mapping[tuple[str, int], PlaneReading]) -> float | None:
    left = readings.get((match.left, match.left_plane))
    right = readings.get((match.right, match.right_plane))
    if left is None or right is None:
        return None
    left_distance = left.distance_for(match.left_b)
    right_distance = right.distance_for(match.right_b)
    if left_distance is None or right_distance is None:
        return None
    mapped_axis, sigma = axis_transform(
        right.axis, match.quadrant_of_right)
    if mapped_axis != left.axis:
        return None
    return float(left.side * left_distance
                 - sigma * right.side * right_distance)


def annotate_integer_residuals(
    solution: ComponentSolution,
    readings: Sequence[PlaneReading],
    matches: Sequence[PlaneMatch],
) -> tuple[PlaneMatch, ...]:
    """Fill each match's ``integer_residual`` from a solved component.

    The residual is the circular distance, in blocks, between the two faces the
    match names after the translations are known: ``frac(t + s*d)`` on the left
    and on the right. Consistent identities sit near zero; a wrong ``b`` or
    identity does not. A match whose reading or axis the solution does not carry
    keeps ``None`` rather than a fabricated number.
    """
    index, _index_by_object, _frames = _canonical_reading_index(readings)
    for match in matches:
        _validate_indexed_match(match, index)

    annotated: list[PlaneMatch] = []
    for match in matches:
        left = index.get((match.left, match.left_plane))
        right = index.get((match.right, match.right_plane))
        if left is None or right is None:
            annotated.append(match)
            continue
        if match.left not in solution.quadrants or match.right not in solution.quadrants:
            annotated.append(match)
            continue
        axis, _left_sign = axis_transform(left.axis, solution.quadrants[match.left])
        right_axis, _right_sign = axis_transform(
            right.axis, solution.quadrants[match.right])
        if right_axis != axis:
            annotated.append(match)
            continue
        left_translation = solution.translations.get(match.left)
        right_translation = solution.translations.get(match.right)
        left_distance = left.distance_for(match.left_b)
        right_distance = right.distance_for(match.right_b)
        if (left_translation is None or right_translation is None
                or left_distance is None or right_distance is None):
            annotated.append(match)
            continue
        left_face = left_translation[axis] + _left_sign * left.side * left_distance
        right_face = right_translation[axis] + _right_sign * right.side * right_distance
        difference = abs((left_face - right_face + 0.5) % 1.0 - 0.5)
        annotated.append(replace(match, integer_residual=float(difference)))
    return tuple(annotated)


def _apply_same_axis_filter(
    matches: list[PlaneMatch],
    readings: Mapping[tuple[str, int], PlaneReading],
    refusals: list,
) -> list[PlaneMatch]:
    """Drop the lower-scoring member of any same-axis pair that disagrees.

    A group shares its frame pair, left local axis and relative-quadrant
    hypothesis. Its members all describe ``t_right - t_left`` in the left-frame
    gauge, so they must imply the same measured displacement. Mutually exclusive
    quadrant hypotheses are never compared here. A disagreement means at
    least one ``(b, identity)`` combination is wrong; the higher-scoring match
    is kept and the other recorded as ``same_axis_mismatch``. An exact score tie
    drops both, because nothing in the evidence chooses between them.

    This is a post-correlation realisation of the directive's cheap equality
    prefilter: the candidate identities are already enumerated here, and cell
    resolution correlation is milliseconds, so the arithmetic is applied to the
    candidates rather than to a separate pre-enumeration.

    The filter keeps or drops whole ``PlaneMatch`` objects from ``matches``, so a
    caller derives each candidate's keep/drop verdict by object identity against
    the returned list; no parallel disposition list is written here.
    """
    grouped: dict[tuple[str, str, int, int], list[PlaneMatch]] = {}
    for match in matches:
        grouped.setdefault((match.left, match.right, match.left_axis,
                            match.quadrant_of_right), []).append(match)
    drop: set[tuple] = set()
    for group in grouped.values():
        if len(group) < 2:
            continue
        deltas = [_delta(match, readings) for match in group]
        for first_position in range(len(group)):
            for second_position in range(first_position + 1, len(group)):
                first, second = group[first_position], group[second_position]
                first_delta, second_delta = deltas[first_position], deltas[second_position]
                if first_delta is None or second_delta is None:
                    continue
                if abs(first_delta - second_delta) <= SAME_AXIS_TOLERANCE:
                    continue
                if abs(first.score - second.score) <= 1e-12:
                    drop.add(_match_key(first))
                    drop.add(_match_key(second))
                elif first.score > second.score:
                    drop.add(_match_key(second))
                else:
                    drop.add(_match_key(first))
    kept: list[PlaneMatch] = []
    for match in matches:
        if _match_key(match) in drop:
            refusals.append(PlaneMatchRefusal(
                left=match.left, right=match.right, left_axis=match.left_axis,
                right_axis=match.right_axis, quadrant_of_right=match.quadrant_of_right,
                reason="same_axis_mismatch",
                detail="another same-axis plane pair implied a different displacement"))
        else:
            kept.append(match)
    return kept


def _match_key(match: PlaneMatch) -> tuple:
    return (match.left, match.right, match.left_plane, match.right_plane,
            match.left_b, match.right_b, match.quadrant_of_right)


def _component(
    reason: str,
    frames: Sequence[str],
    *,
    quadrants: Mapping[str, int] | None = None,
    translations: Mapping[str, tuple[float, float, float]] | None = None,
    residuals: Mapping[int, float] | None = None,
    resultants: Mapping[int, float] | None = None,
    blocks_per_period: Mapping[tuple[str, int], float] | None = None,
    alternatives: tuple[ComponentAlternative, ...] = (),
    scale_known: bool = False,
) -> ComponentSolution:
    frame_tuple = tuple(frames)
    empty_translations = {frame: (0.0, 0.0, 0.0) for frame in frame_tuple}
    return ComponentSolution(
        frames=frame_tuple,
        quadrants=dict(quadrants or {frame: 0 for frame in frame_tuple}),
        translations=dict(translations or empty_translations),
        axis_residuals=dict(residuals or {}),
        axis_resultants=dict(resultants or {}),
        scale_known=bool(scale_known),
        blocks_per_period=dict(blocks_per_period or {}),
        reason=reason,
        alternatives=alternatives,
    )


def _quadrant_assignment_plan(
    frames: Sequence[str], constraints: Mapping[tuple[str, str], int],
) -> tuple[str, int, Iterator[dict[str, int]]]:
    """Plan exact, lazy assignments from validated modular-4 constraints.

    Each component's first frame has offset zero. Ordering free component roots
    by their first frame therefore reproduces the filtered full-frame Cartesian
    order, even when other offsets wrap around. Preprocessing is O(N+E); the
    budget bounds full assignments, not numerical work or RSS per assignment.
    """
    adjacency: dict[str, list[tuple[str, int]]] = {frame: [] for frame in frames}
    for (left, right), relative in constraints.items():
        adjacency[left].append((right, relative))
        adjacency[right].append((left, (-relative) % 4))
    offsets: dict[str, int] = {}
    component_of: dict[str, int] = {}
    components = 0
    for root in frames:
        if root in offsets:
            continue
        offsets[root] = 0
        component_of[root] = components
        pending = [root]
        while pending:
            left = pending.pop()
            for right, relative in adjacency[left]:
                expected = (offsets[left] + relative) % 4
                if right in offsets:
                    if offsets[right] != expected:
                        return "quadrant_constraint_inconsistent", 0, iter(())
                else:
                    offsets[right] = expected
                    component_of[right] = components
                    pending.append(right)
        components += 1
    count = 1
    for _ in range(max(components - 1, 0)):
        if count > MAX_ASSIGNMENTS // 4:
            return "component_too_large", 0, iter(())
        count *= 4

    def assignments() -> Iterator[dict[str, int]]:
        for tail in product(range(4), repeat=max(components - 1, 0)):
            rotations = (0, *tail)
            yield {frame: (rotations[component_of[frame]] + offsets[frame]) % 4
                   for frame in frames}

    return "", count, assignments()


_ENDPOINT_CONFLICT_OBSERVER = ContextVar("endpoint_conflict_observer", default=None)


@contextmanager
def _endpoint_conflict_context(observer):
    """Evaluation-only same-pass observation; never part of a solver payload."""
    token = _ENDPOINT_CONFLICT_OBSERVER.set(observer)
    try:
        yield observer
    finally:
        _ENDPOINT_CONFLICT_OBSERVER.reset(token)


class _EndpointConflictObserver:
    """Bounded scalar receipt for the first conflict of each visited assignment.

    Input bindings and assignment events are streamed in their actual order.
    The byte budget bounds total JSON hashing work, not only retained examples.
    Exhaustion marks the receipt refused while the ordinary solve continues.
    """

    MAX_WITNESSES = 128
    MAX_BYTES = 16 * 1024 * 1024
    MAX_ROW_BYTES = 32 * 1024
    MAX_READINGS = 256
    MAX_FRAMES = 32

    def __init__(self):
        self.calls = self.planned = self.visited = self.conflicted = 0
        self.evaluated = self.feasible = self.hashed_assignments = 0
        self.closed = False
        self.refusal = ""
        self.frames = ()
        self.frame_count = 0
        self.bindings = {}
        self.witnesses = []
        self.classes = {"same_endpoint_pair_competing_hypotheses": 0,
                        "different_incident_edges": 0}
        self.work_bytes = 0
        self.digest = hashlib.sha256()
        self.reading_digest = hashlib.sha256()
        self.match_digest = hashlib.sha256()
        self.pending = None
        self.plan_reason = ""
        self.match_count = 0

    def _encode(self, value):
        if self.refusal:
            return None
        # Bound container traversal/string chunks before JSON allocates them.
        stack = [(value, 0)]
        lower_bytes = items = 0
        while stack:
            item, depth = stack.pop()
            items += 1
            if depth > 16 or items + lower_bytes > self.MAX_ROW_BYTES:
                self.refusal = "scalar_byte_budget"
                return None
            if isinstance(item, (tuple, list, dict)):
                if len(item) + len(stack) + items + lower_bytes > self.MAX_ROW_BYTES:
                    self.refusal = "scalar_byte_budget"
                    return None
                if isinstance(item, dict):
                    lower_bytes += sum(len(k) for k in item)
                    children = item.values()
                else:
                    children = item
                stack.extend((child, depth + 1) for child in children)
            elif isinstance(item, str):
                lower_bytes += len(item)
                if lower_bytes + items > self.MAX_ROW_BYTES:
                    self.refusal = "scalar_byte_budget"
                    return None
        data = bytearray()
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                   allow_nan=False, default=lambda v: v.item())
        try:
            for chunk in encoder.iterencode(value):
                raw = chunk.encode("utf-8")
                if (len(data) + len(raw) > self.MAX_ROW_BYTES
                        or self.work_bytes + len(data) + len(raw) > self.MAX_BYTES):
                    self.refusal = "scalar_byte_budget"
                    return None
                data.extend(raw)
        except (ValueError, TypeError, AttributeError, OverflowError):
            self.refusal = "invalid_scalar_evidence"
            return None
        self.work_bytes += len(data)
        return bytes(data)

    @staticmethod
    def _match_record(match):
        record = {name: getattr(match, name) for name in match.__dataclass_fields__}
        # PSR may legitimately be infinite in a refused matcher hypothesis.
        for name, value in record.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                record[name] = str(float(value))
        return record

    def begin(self, frames, readings, matches, planned, reason=""):
        self.calls += 1
        if self.calls != 1:
            self.refusal = "solver_call_multiplicity"
            return
        self.frame_count = len(frames)
        self.frames, self.planned, self.plan_reason = tuple(frames[:self.MAX_FRAMES]), planned, reason
        self.match_count = len(matches)
        if len(frames) > self.MAX_FRAMES or len(readings) > self.MAX_READINGS:
            self.refusal = "source_binding_budget"
            return
        for key, reading in readings.items():
            encoded = self._encode([key, reading_order_key(reading)])
            if encoded is None:
                return
            self.reading_digest.update(encoded + b"\n")
            self.bindings[key] = {
                "endpoint": list(key), "axis": reading.axis,
                "reading_order_key_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_region_id": (reading.region_evidence.source_region_id
                                     if reading.region_evidence is not None else None),
            }
        for index, match in enumerate(matches):
            encoded = self._encode([index, self._match_record(match)])
            if encoded is None:
                return
            self.match_digest.update(encoded + b"\n")

    def visit(self, quadrants):
        self.visited += 1
        self.pending = None

    def conflict(self, key, first, second, reading):
        if self.refusal:
            return
        def selection(value):
            index, match, requested_b, alternative = value
            return {"input_match_index": index,
                    "match_identity": list(_match_key(match)),
                    "match": self._match_record(match),
                    "requested_b": float(requested_b),
                    "canonical_alternative_index": next(
                        i for i, item in enumerate(reading.alternatives)
                        if item is alternative),
                    "blocks_per_period": float(alternative.blocks_per_period),
                    "plane_distance_blocks": float(alternative.distance_blocks)}
        endpoint_pair = lambda m: tuple(sorted(((m.left, m.left_plane),
                                                (m.right, m.right_plane))))
        category = ("same_endpoint_pair_competing_hypotheses"
                    if endpoint_pair(first[1]) == endpoint_pair(second[1])
                    else "different_incident_edges")
        self.classes[category] += 1
        self.pending = {"conflict_class": category,
                        "endpoint": list(key), "source_binding": self.bindings[key],
                        "first": selection(first), "second": selection(second)}

    def terminal(self, quadrants, conflict, feasible=False):
        self.conflicted += int(conflict)
        self.evaluated += int(not conflict)
        self.feasible += int(feasible)
        if conflict and self.pending is None and not self.refusal:
            self.refusal = "missing_conflict_hook"
        if self.refusal:
            self.pending = None
            return
        event = {"assignment_index": self.visited - 1,
                 "quadrants": [quadrants[f] for f in self.frames],
                 "disposition": ("endpoint_period_conflict" if conflict else
                                 "evaluated_feasible" if feasible else "evaluated_infeasible"),
                 "first_conflict": self.pending}
        encoded = self._encode(event)
        if encoded is not None:
            self.digest.update(encoded + b"\n")
            self.hashed_assignments += 1
            if conflict and len(self.witnesses) < self.MAX_WITNESSES:
                self.witnesses.append(event)
        self.pending = None

    def finish(self):
        self.closed = True

    def record(self):
        complete = (self.calls == 1 and self.closed and not self.refusal
                    and self.planned == self.visited == self.hashed_assignments
                    and self.visited == self.conflicted + self.evaluated
                    and self.feasible <= self.evaluated
                    and sum(self.classes.values()) == self.conflicted)
        return {"schema": "mc-photogrammetry-endpoint-conflict-observation-v1",
                "status": "refused" if self.refusal else "complete" if complete else "incomplete",
                "reason": self.refusal or ("" if complete else "accounting_incomplete"),
                "solver_calls": self.calls, "frames": list(self.frames),
                "frame_count": self.frame_count,
                "planned_assignments": self.planned, "visited_assignments": self.visited,
                "conflicted_assignments": self.conflicted, "evaluator_calls": self.evaluated,
                "feasible_evaluations": self.feasible, "hashed_assignments": self.hashed_assignments,
                "planner_reason": self.plan_reason, "conflict_classes": dict(self.classes),
                "assignment_sequence_sha256": self.digest.hexdigest(),
                "source_readings_sha256": self.reading_digest.hexdigest(),
                "input_matches_sha256": self.match_digest.hexdigest(),
                "input_match_count": self.match_count,
                "source_bindings": list(self.bindings.values()),
                "witnesses": list(self.witnesses),
                "retained_witness_count": len(self.witnesses),
                "omitted_witness_count": self.conflicted - len(self.witnesses),
                "example_order": "first conflict per assignment in exact planner order",
                "work_bytes": self.work_bytes,
                "limits": {"witnesses": self.MAX_WITNESSES, "work_bytes": self.MAX_BYTES,
                           "row_bytes": self.MAX_ROW_BYTES, "readings": self.MAX_READINGS,
                           "frames": self.MAX_FRAMES, "assignments": MAX_ASSIGNMENTS}}


def solve_component(
    readings: Sequence[PlaneReading],
    matches: Sequence[PlaneMatch],
    *,
    integer_tolerance: float = 0.15,
    min_resultant: float = 0.90,
    ambiguous_margin: float = 0.02,
    quadrant_constraints: tuple[tuple[str, str, int], ...] = (),
) -> ComponentSolution:
    """Solve quadrants and per-axis translations, or abstain with a typed reason.

    Quadrant assignments are enumerated with the anchor frame at ``k = 0`` (a
    gauge, never a measurement). Per assignment, matches whose relative quadrant
    agrees are kept, grouped by world axis, and the translations are solved by
    least squares in ``t_right[a] - t_left[a] = s_left*d_left - s_right*d_right``
    — the exact :mod:`registration` semantics, generalised to more than one plane
    per axis. The integer-face resultant then scores the assignment. Selection is
    by cost with the lexicographic assignment as tie-break; a runner-up within
    ``ambiguous_margin`` is a typed abstention, not a guess.
    """
    read_by_index, _index_by_object, frames = _canonical_reading_index(readings)
    observer = _ENDPOINT_CONFLICT_OBSERVER.get()
    if not readings:
        if observer is not None:
            observer.begin(frames, read_by_index, matches, 0, "no_plane_reading")
            observer.finish()
        return _component("no_plane_reading", frames)
    if not matches:
        if observer is not None:
            observer.begin(frames, read_by_index, matches, 0, "no_matches")
            observer.finish()
        return _component("no_plane_reading" if not frames else "no_matches", frames)
    for match in matches:
        _validate_indexed_match(match, read_by_index)
    if not isinstance(quadrant_constraints, tuple):
        raise ValueError("quadrant constraints must be an immutable tuple")
    constraints: dict[tuple[str, str], int] = {}
    frame_set = set(frames)
    for value in quadrant_constraints:
        if (not isinstance(value, tuple) or len(value) != 3
                or type(value[0]) is not str or type(value[1]) is not str
                or not value[0] or not value[1] or value[0] == value[1]
                or value[0] not in frame_set or value[1] not in frame_set
                or type(value[2]) is not int or value[2] not in (0, 1, 2, 3)):
            raise ValueError("quadrant constraint is malformed or names a foreign frame")
        left, right, relative = value
        if right < left:
            left, right, relative = right, left, (-relative) % 4
        key = (left, right)
        if key in constraints and constraints[key] != relative:
            if observer is not None:
                observer.begin(frames, read_by_index, matches, 0,
                               "quadrant_constraint_inconsistent")
                observer.finish()
            return _component("quadrant_constraint_inconsistent", frames)
        constraints[key] = relative
    plan_reason, _assignment_count, assignments = _quadrant_assignment_plan(
        frames, constraints)
    if observer is not None:
        observer.begin(frames, read_by_index, matches, _assignment_count, plan_reason)
    if plan_reason:
        if observer is not None:
            observer.finish()
        return _component(plan_reason, frames)
    ranked: list[tuple[float, ComponentAlternative, dict]] = []
    permitted_assignments = 0
    period_conflicts = 0
    for quadrants in assignments:
        permitted_assignments += 1
        if observer is not None:
            observer.visit(quadrants)
        if _assignment_endpoint_period_conflict(quadrants, matches, read_by_index):
            period_conflicts += 1
            if observer is not None:
                observer.terminal(quadrants, True)
            continue
        evaluation = _evaluate_assignment(frames, quadrants, matches, read_by_index)
        if observer is not None:
            observer.terminal(quadrants, False, evaluation is not None)
        if evaluation is None:
            continue
        cost, alternative, detail = evaluation
        ranked.append((cost, alternative, detail))

    if observer is not None:
        observer.finish()
    if not ranked:
        if permitted_assignments and period_conflicts == permitted_assignments:
            return _component("endpoint_period_conflict", frames)
        return _component(
            "quadrant_constraint_inconsistent" if constraints else "no_matches",
            frames)
    ranked.sort(key=lambda item: (item[0], item[1].quadrants))
    best_cost, best_alternative, best_detail = ranked[0]
    alternatives = tuple(item[1] for item in ranked[:MAX_ALTERNATIVES])

    ambiguous = (len(ranked) > 1
                 and (ranked[1][0] - best_cost) < ambiguous_margin
                 and ranked[1][1].quadrants != best_alternative.quadrants)
    quadrants = {frame: best_alternative.quadrants[index] for index, frame in enumerate(frames)}
    base = dict(
        quadrants=quadrants,
        translations=best_detail["translations"],
        residuals=best_detail["residuals"],
        resultants=best_detail["resultants"],
        blocks_per_period=best_detail["blocks_per_period"],
        alternatives=alternatives,
    )
    if ambiguous:
        return _component("ambiguous_quadrant", frames, **base)
    if best_detail["duplicate_b"]:
        return _component("b_unresolved", frames, **base)
    matched_axes = sorted(best_detail["resultants"])
    if len(matched_axes) < 3:
        return _component("insufficient_axis_coverage", frames, **base)
    if (any(best_detail["resultants"][axis] < min_resultant for axis in matched_axes)
            or any(best_detail["residuals"][axis] > integer_tolerance for axis in matched_axes)):
        return _component("plane_integer_residual", frames, **base)
    return _component("", frames, scale_known=len(matched_axes) >= 2, **base)


def _assignment_endpoint_period_conflict(
    quadrants: Mapping[str, int],
    matches: Sequence[PlaneMatch],
    read_by_index: Mapping[tuple[str, int], PlaneReading],
) -> bool:
    """Check canonical alternatives only on edges active in this assignment.

    ``matching_alternatives`` returns the reading's existing objects, so identity
    compares canonical alternatives rather than noisy requested b labels.
    Unresolved selections remain subject to the evaluator's existing refusal.
    No edge or frame is removed to reconcile contradictory selections.
    """
    selected: dict[tuple[str, int], object] = {}
    observer = _ENDPOINT_CONFLICT_OBSERVER.get()
    origins = {} if observer is not None and not observer.refusal else None
    for match_index, match in enumerate(matches):
        if ((quadrants[match.right] - quadrants[match.left]) % 4
                != match.quadrant_of_right):
            continue
        for key, b in (((match.left, match.left_plane), match.left_b),
                       ((match.right, match.right_plane), match.right_b)):
            alternatives = read_by_index[key].matching_alternatives(b)
            if len(alternatives) != 1:
                continue
            alternative = alternatives[0]
            if key in selected and selected[key] is not alternative:
                if origins is not None:
                    observer.conflict(key, origins[key],
                                      (match_index, match, b, alternative), read_by_index[key])
                return True
            selected[key] = alternative
            if origins is not None:
                # Keep the actual prior encounter used by the unchanged guard.
                origins[key] = (match_index, match, b, alternative)
    return False


def _evaluate_assignment(
    frames: Sequence[str],
    quadrants: Mapping[str, int],
    matches: Sequence[PlaneMatch],
    read_by_index: Mapping[tuple[str, int], PlaneReading],
) -> tuple[float, ComponentAlternative, dict] | None:
    """Solve one quadrant assignment and score it; ``None`` if infeasible."""
    active: list[PlaneMatch] = []
    for match in matches:
        if match.left not in quadrants or match.right not in quadrants:
            continue
        if (quadrants[match.right] - quadrants[match.left]) % 4 != match.quadrant_of_right:
            continue
        active.append(match)

    constraints: dict[int, list[tuple[str, str, float]]] = {0: [], 1: [], 2: []}
    blocks: dict[tuple[str, int], float] = {}
    duplicate_b = False
    seen_without_b: set[tuple] = set()
    for match in active:
        left = read_by_index.get((match.left, match.left_plane))
        right = read_by_index.get((match.right, match.right_plane))
        if left is None or right is None:
            return None
        world, left_sign = axis_transform(left.axis, quadrants[match.left])
        right_world, right_sign = axis_transform(right.axis, quadrants[match.right])
        if right_world != world:
            return None
        left_distance = left.distance_for(match.left_b)
        right_distance = right.distance_for(match.right_b)
        if left_distance is None or right_distance is None:
            return None
        left_face = left_sign * left.side * left_distance
        right_face = right_sign * right.side * right_distance
        delta = float(left_face - right_face)
        constraints[world].append((match.left, match.right, delta))
        key = (match.left, match.left_plane, match.right, match.right_plane, match.quadrant_of_right)
        if key in seen_without_b:
            duplicate_b = True
        seen_without_b.add(key)
        blocks[(match.left, match.left_plane)] = float(match.left_b)
        blocks[(match.right, match.right_plane)] = float(match.right_b)

    translations: dict[str, list[float]] = {frame: [0.0, 0.0, 0.0] for frame in frames}
    resultants: dict[int, float] = {}
    residuals: dict[int, float] = {}
    observations: dict[int, int] = {0: 0, 1: 0, 2: 0}
    anchor = frames[0]

    for axis in range(3):
        rows = constraints[axis]
        if not rows:
            continue
        unknown_frames = [frame for frame in frames if frame != anchor]
        column_of = {frame: index for index, frame in enumerate(unknown_frames)}
        matrix = np.zeros((len(rows), len(unknown_frames)), dtype=float)
        target = np.zeros(len(rows), dtype=float)
        for row_index, (left_frame, right_frame, delta) in enumerate(rows):
            if right_frame != anchor:
                matrix[row_index, column_of[right_frame]] += 1.0
            if left_frame != anchor:
                matrix[row_index, column_of[left_frame]] -= 1.0
            target[row_index] = delta
        if len(unknown_frames) == 0:
            solution = np.zeros(0, dtype=float)
        else:
            solution, _residues, _rank, _singular = np.linalg.lstsq(matrix, target, rcond=None)
        axis_translation = {frame: 0.0 for frame in frames}
        for frame in unknown_frames:
            axis_translation[frame] = float(solution[column_of[frame]])
        axis_translation[anchor] = 0.0
        if axis == 0:
            for frame in frames:
                translations[frame][0] = axis_translation[frame]
        elif axis == 1:
            for frame in frames:
                translations[frame][1] = axis_translation[frame]
        else:
            for frame in frames:
                translations[frame][2] = axis_translation[frame]

        face_values: list[float] = []
        for match in active:
            left, right = (read_by_index[(match.left, match.left_plane)],
                           read_by_index[(match.right, match.right_plane)])
            left_world, left_sign = axis_transform(
                left.axis, quadrants[match.left])
            if left_world != axis:
                continue
            right_world, right_sign = axis_transform(
                right.axis, quadrants[match.right])
            if right_world != axis:
                continue
            face_values.append(axis_translation[match.left]
                               + left_sign * left.side
                               * left.distance_for(match.left_b))
            face_values.append(axis_translation[match.right]
                               + right_sign * right.side
                               * right.distance_for(match.right_b))
        angles = 2.0 * np.pi * np.mod(np.asarray(face_values, dtype=float), 1.0)
        resultant = float(abs(np.mean(np.exp(1j * angles)))) if len(angles) else 0.0
        if len(angles):
            mean_angle = float(np.angle(np.mean(np.exp(1j * angles))))
            deviation = np.angle(np.exp(1j * (angles - mean_angle)))
            residual = float(np.mean(np.abs(deviation)) / (2.0 * np.pi))
        else:
            residual = 0.0
        resultants[axis] = resultant
        residuals[axis] = residual
        observations[axis] = len(face_values)

    if not resultants:
        return None
    unresolved = sum(1 for frame in frames
                     if not any(left == frame or right == frame
                                for axis in constraints for left, right, _d in constraints[axis]))
    mean_score = float(np.mean([match.score for match in active])) if active else 0.0
    cost = (sum((1.0 - resultants[axis]) * observations[axis] for axis in resultants)
            + LAMBDA_UNRESOLVED * unresolved
            + MU_MATCH_SCORE * (1.0 - mean_score))
    alternative = ComponentAlternative(
        quadrants=tuple(quadrants[frame] for frame in frames),
        cost=float(cost),
        resultants=tuple(resultants.get(axis, 0.0) for axis in range(3)),
        axis_observations=tuple(observations[axis] for axis in range(3)),
    )
    detail = {
        "translations": {frame: tuple(float(value) for value in translations[frame])
                         for frame in frames},
        "residuals": {axis: residuals[axis] for axis in resultants},
        "resultants": {axis: resultants[axis] for axis in resultants},
        "blocks_per_period": blocks,
        "duplicate_b": duplicate_b,
    }
    return float(cost), alternative, detail


def resolved_class(
    solution: ComponentSolution,
    frame: str,
    calibration_result: object,
) -> tuple[int, float]:
    """The frame's resolved yaw quadrant and the margin over its best rival.

    The margin is the cost gap to the best retained assignment that puts *this*
    frame in a different quadrant; ``inf`` means no retained rival differs for
    this frame. Callers treating a margin below ``0.25`` as
    ``ambiguous_quadrant`` get exactly the directive's semantics. ``frame`` must
    be present in both the solution and the calibration mapping; otherwise the
    honest answer is ``(0, 0.0)`` — unresolved.
    """
    if frame not in solution.quadrants:
        return (0, 0.0)
    if calibration_result is None or frame not in calibration_result:
        return (0, 0.0)
    index = solution.frames.index(frame)
    quadrant = int(solution.quadrants[frame])
    if not solution.alternatives:
        return (quadrant, float("inf"))
    best = solution.alternatives[0]
    competitor: float | None = None
    for alternative in solution.alternatives[1:]:
        if alternative.quadrants[index] == quadrant:
            continue
        if competitor is None or alternative.cost < competitor:
            competitor = float(alternative.cost)
    if competitor is None:
        return (quadrant, float("inf"))
    return (quadrant, competitor - float(best.cost))


def _ry90(yaw_class: int) -> np.ndarray:
    angle = np.radians(90.0 * (int(yaw_class) % 4))
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos)), dtype=float)


def _residuals(rows: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Per-edge ``|a^T t| / ||a||`` in degrees; a degenerate row scores worst."""
    projections = np.abs(rows @ direction)
    norms = np.linalg.norm(rows, axis=1)
    values = np.ones(len(rows), dtype=float)
    safe = norms > 1e-12
    values[safe] = np.clip(projections[safe] / norms[safe], 0.0, 1.0)
    return np.degrees(np.arcsin(values))


def _robust_direction(rows: np.ndarray, threshold_degrees: float) -> np.ndarray | None:
    """A deterministic robust baseline direction for the fixed yaw quadrant.

    One null vector over every candidate edge is corrupted when the graph is
    mostly impostors, so a bounded deterministic schedule of endpoint-distinct
    pairs is scored by residual-qualified support and refined to a fixed point.
    No random generator is used: the pair schedule is a fixed arithmetic
    sequence, which keeps the result bit-identical across runs.
    """
    if rows.size == 0:
        return None
    count = len(rows)
    candidates = [_null_vector(rows)]
    for sample in range(int(ROBUST_SAMPLES)):
        first = (sample * 37 + 7) % count
        second = (sample * 91 + 17) % count
        if first == second:
            continue
        candidate = _null_vector(rows[[first, second]])
        if candidate is not None:
            candidates.append(candidate)
    best_direction = None
    best_key: tuple | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        residual = _residuals(rows, candidate)
        mask = residual <= threshold_degrees
        support = int(np.count_nonzero(mask))
        median = float(np.median(residual[mask])) if support else float("inf")
        key = (-support, median)
        if best_key is None or key < best_key:
            best_direction, best_key = candidate, key
    if best_direction is None:
        return None
    direction = best_direction
    residual = _residuals(rows, direction)
    mask = residual <= threshold_degrees
    for _ in range(10):
        if not mask.any():
            break
        refined = _null_vector(rows[mask])
        if refined is None:
            break
        new_residual = _residuals(rows, refined)
        new_mask = new_residual <= threshold_degrees
        if np.array_equal(new_mask, mask):
            direction = refined
            break
        direction, mask = refined, new_mask
    return direction


def _null_vector(rows: np.ndarray) -> np.ndarray | None:
    if rows.size == 0:
        return None
    gram = rows.T @ rows
    if not np.all(np.isfinite(gram)):
        return None
    values, vectors = np.linalg.eigh(gram)
    vector = vectors[:, int(np.argmin(values))]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def class_constrained_bestof(
    features: ClassConstrainedFeatures,
    neighbors: object,
    quadrant: int,
    *,
    residual_degrees: float = 1.0,
) -> np.ndarray:
    """The best one-to-one match set under a *fixed* yaw quadrant.

    Borrows DS-18's rule exactly: every left endpoint keeps its single best
    residual-qualified edge, and only then are right collisions resolved in a
    deterministic ``(residual, distance, left, right)`` order into a one-to-one
    set. No left endpoint's evidence is dropped before geometry sees it, and no
    random sampling is involved. ``neighbors`` is any object exposing ``edges``
    (``M x 2`` left/right indices) and ``distances`` (``M``) — DS-18's
    ``DirectedCandidateGraph`` satisfies it unchanged.
    """
    edges = np.asarray(neighbors.edges)
    distances = np.asarray(neighbors.distances, dtype=float)
    if edges.size == 0:
        return np.zeros((0, 4), dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("neighbors.edges must be M x 2")
    rotation = np.asarray(features.rotation_right) @ _ry90(quadrant) @ np.asarray(features.rotation_left).T
    left_rays = np.asarray(features.left_rays, dtype=float)
    right_rays = np.asarray(features.right_rays, dtype=float)
    left_in_right = left_rays @ rotation.T
    rows = np.cross(left_in_right[edges[:, 0]], right_rays[edges[:, 1]])
    direction = _robust_direction(rows, float(residual_degrees))
    if direction is None:
        return np.zeros((0, 4), dtype=float)
    residual = _residuals(rows, direction)

    order = np.lexsort((edges[:, 1], edges[:, 0], distances, residual))
    best_left: dict[int, int] = {}
    for position in order:
        if residual[position] > float(residual_degrees):
            break
        left = int(edges[position, 0])
        if left not in best_left:
            best_left[left] = int(position)
    if not best_left:
        return np.zeros((0, 4), dtype=float)
    survivors = sorted(
        best_left.values(),
        key=lambda position: (float(residual[position]), float(distances[position]),
                              int(edges[position, 0]), int(edges[position, 1])),
    )
    used_right: set[int] = set()
    selected: list[tuple[float, float, float, float]] = []
    for position in survivors:
        right = int(edges[position, 1])
        if right in used_right:
            continue
        used_right.add(right)
        left = int(edges[position, 0])
        selected.append((
            float(features.left_keypoints[left, 0]), float(features.left_keypoints[left, 1]),
            float(features.right_keypoints[right, 0]), float(features.right_keypoints[right, 1]),
        ))
    return np.asarray(selected, dtype=float).reshape(-1, 4)
