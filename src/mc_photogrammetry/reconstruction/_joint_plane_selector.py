"""Internal exact joint q/b selector prototype for M4 G1C.

This module is deliberately not imported by the public reconstruction pipeline.
It enumerates one anchored quadrant per frame and one blocks-per-period value
per touched reading; every compatible match is induced by those labels. A
caller must provide all work limits. Exhaustion returns no partial candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from itertools import product
import json
import math
from typing import Callable, Mapping, Sequence, TypeVar

from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction.lattice_planes import (
    PlaneReading,
    reading_order_key,
)


ReadingKey = tuple[str, int]
PairKey = tuple[str, int, str, int]
_T = TypeVar("_T")
MAX_DIAGNOSTIC_SIGNATURES = 4_096


class _CompleteLabelGate(str, Enum):
    NO_COMPATIBLE_RECORDS = "no_compatible_records"
    OBJECTIVE_REJECTED = "objective_rejected"
    INSUFFICIENT_AXES = "insufficient_axes"
    MINIMUM_RESULTANT = "minimum_resultant"
    INTEGER_TOLERANCE = "integer_tolerance"
    ACCEPTED = "accepted"
    BUDGET_ABORTED = "budget_aborted"


@dataclass(frozen=True, slots=True)
class _PartialSemanticSignature:
    frame_order: tuple[str, ...]
    depth: int
    quadrant_labels: tuple[tuple[str, int | None], ...]
    blocks_per_period_labels: tuple[tuple[str, int, float | None], ...]
    feasible_row_mask: int
    active_row_indices: tuple[int, ...]
    possible_axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CompleteLeafSignature:
    frame_order: tuple[str, ...]
    quadrant_labels: tuple[tuple[str, int], ...]
    blocks_per_period_labels: tuple[tuple[str, int, float], ...]
    feasible_row_mask: int
    active_row_indices: tuple[int, ...]
    objective_rows: tuple[tuple[int, int, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Explicit caller-supplied deterministic work limits; no defaults exist."""

    max_search_nodes: int
    max_compatibility_checks: int
    max_objective_rows: int
    max_axis_solves: int
    max_pareto_comparisons: int
    max_candidate_certificates: int
    max_buffered_match_rows: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_search_nodes", self.max_search_nodes),
            ("max_compatibility_checks", self.max_compatibility_checks),
            ("max_objective_rows", self.max_objective_rows),
            ("max_axis_solves", self.max_axis_solves),
            ("max_pareto_comparisons", self.max_pareto_comparisons),
            ("max_candidate_certificates", self.max_candidate_certificates),
            ("max_buffered_match_rows", self.max_buffered_match_rows),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SearchCounters:
    search_nodes: int
    compatibility_checks: int
    objective_rows: int
    axis_solves: int
    axis_solve_requests: int
    pareto_comparisons: int
    complete_labels: int
    evaluated_candidates: int
    pruned_insufficient_axis: int
    shared_objective_evaluations: int
    duplicate_rows_deduplicated: int
    candidate_certificates_buffered: int
    buffered_match_rows: int
    objective_cache_hits: int
    objective_cache_misses: int
    objective_cache_unique_signatures: int
    physical_axis_cache_hits: int
    physical_axis_cache_misses: int
    diagnostic_gate_no_compatible_records: int
    diagnostic_gate_objective_rejected: int
    diagnostic_gate_insufficient_axes: int
    diagnostic_gate_minimum_resultant: int
    diagnostic_gate_integer_tolerance: int
    diagnostic_gate_accepted: int
    diagnostic_gate_budget_aborted: int
    diagnostic_post_accept_buffering_aborts: int
    diagnostic_partial_state_observations: int
    diagnostic_partial_state_unique: int
    diagnostic_partial_state_repeats: int
    diagnostic_partial_state_truncated: int
    diagnostic_leaf_signature_observations: int
    diagnostic_leaf_signature_unique: int
    diagnostic_leaf_signature_repeats: int
    diagnostic_leaf_signature_truncated: int


@dataclass(frozen=True, slots=True)
class JointCandidate:
    quadrants: tuple[tuple[str, int], ...]
    blocks_per_period: tuple[tuple[str, int, float], ...]
    pair_groups: frozenset[PairKey]
    covered_frames: frozenset[str]
    covered_readings: frozenset[ReadingKey]
    axis_pair_groups: tuple[tuple[int, frozenset[PairKey]], ...]
    resultants: tuple[tuple[int, float], ...]
    residuals: tuple[tuple[int, float], ...]
    translations: tuple[tuple[str, tuple[float, float, float]], ...]
    axis_observations: tuple[int, int, int]
    scale_known: bool
    mean_match_score: float
    cost: float
    _active_matches: tuple[plane_graph.PlaneMatch, ...]

    @property
    def quadrant_map(self) -> dict[str, int]:
        return dict(self.quadrants)

    @property
    def b_map(self) -> dict[ReadingKey, float]:
        return {(frame, plane): value for frame, plane, value in self.blocks_per_period}

    @property
    def axis_group_map(self) -> dict[int, frozenset[PairKey]]:
        return dict(self.axis_pair_groups)

    @property
    def resultant_map(self) -> dict[int, float]:
        return dict(self.resultants)

    @property
    def residual_map(self) -> dict[int, float]:
        return dict(self.residuals)


@dataclass(frozen=True, slots=True)
class SelectorResult:
    status: str
    reason: str
    candidate: JointCandidate | None
    pareto: tuple[JointCandidate, ...]
    counters: SearchCounters
    certificate_sha256: str | None


class _BudgetExceeded(Exception):
    def __init__(self, counter: str):
        super().__init__(counter)
        self.counter = counter


class _Work:
    def __init__(
        self,
        budget: SearchBudget | None,
        *,
        optimized: bool = True,
        axis_budget_mode: str = "physical",
        signature_limit: int = MAX_DIAGNOSTIC_SIGNATURES,
    ):
        if axis_budget_mode not in ("physical", "requests"):
            raise ValueError("axis_budget_mode must be 'physical' or 'requests'")
        if (isinstance(signature_limit, bool) or not isinstance(signature_limit, int)
                or signature_limit < 0):
            raise ValueError("signature_limit must be a non-negative integer")
        self.budget = budget
        self.optimized = optimized
        self.axis_budget_mode = axis_budget_mode
        self.signature_limit = signature_limit
        self.partial_state_signatures: set[_PartialSemanticSignature] = set()
        self.leaf_signatures: set[_CompleteLeafSignature] = set()
        self.objective_cache: dict[tuple, object] = {}
        self.values = {
            "search_nodes": 0,
            "compatibility_checks": 0,
            "objective_rows": 0,
            "axis_solves": 0,
            "axis_solve_requests": 0,
            "pareto_comparisons": 0,
            "complete_labels": 0,
            "evaluated_candidates": 0,
            "pruned_insufficient_axis": 0,
            "shared_objective_evaluations": 0,
            "duplicate_rows_deduplicated": 0,
            "candidate_certificates_buffered": 0,
            "buffered_match_rows": 0,
            "objective_cache_hits": 0,
            "objective_cache_misses": 0,
            "objective_cache_unique_signatures": 0,
            "physical_axis_cache_hits": 0,
            "physical_axis_cache_misses": 0,
            "diagnostic_gate_no_compatible_records": 0,
            "diagnostic_gate_objective_rejected": 0,
            "diagnostic_gate_insufficient_axes": 0,
            "diagnostic_gate_minimum_resultant": 0,
            "diagnostic_gate_integer_tolerance": 0,
            "diagnostic_gate_accepted": 0,
            "diagnostic_gate_budget_aborted": 0,
            "diagnostic_post_accept_buffering_aborts": 0,
            "diagnostic_partial_state_observations": 0,
            "diagnostic_partial_state_unique": 0,
            "diagnostic_partial_state_repeats": 0,
            "diagnostic_partial_state_truncated": 0,
            "diagnostic_leaf_signature_observations": 0,
            "diagnostic_leaf_signature_unique": 0,
            "diagnostic_leaf_signature_repeats": 0,
            "diagnostic_leaf_signature_truncated": 0,
        }
        self.limits = {
            "search_nodes": None if budget is None else budget.max_search_nodes,
            "compatibility_checks": None if budget is None else budget.max_compatibility_checks,
            "objective_rows": None if budget is None else budget.max_objective_rows,
            "axis_solves": (
                None if budget is None or axis_budget_mode == "requests"
                else budget.max_axis_solves
            ),
            "axis_solve_requests": (
                None if budget is None or axis_budget_mode == "physical"
                else budget.max_axis_solves
            ),
            "pareto_comparisons": None if budget is None else budget.max_pareto_comparisons,
            "candidate_certificates_buffered": (
                None if budget is None else budget.max_candidate_certificates
            ),
            "buffered_match_rows": None if budget is None else budget.max_buffered_match_rows,
        }

    def charge(self, counter: str, amount: int = 1) -> None:
        limit = self.limits[counter]
        if limit is not None and self.values[counter] + amount > limit:
            raise _BudgetExceeded(counter)
        self.values[counter] += amount

    def charge_logical_scan(self, counter: str, amount: int) -> None:
        """Charge a row-by-row logical scan, including its partial final count."""
        if amount <= 0:
            return
        limit = self.limits[counter]
        if limit is None:
            self.values[counter] += amount
            return
        available = limit - self.values[counter]
        if amount > available:
            # The pre-mask loop charged until the first row beyond the limit.
            self.values[counter] = limit
            raise _BudgetExceeded(counter)
        self.values[counter] += amount

    def charge_axis_solve_request(self, amount: int) -> None:
        """Record requested per-axis work; optionally enforce the old request cap."""
        if self.axis_budget_mode == "requests":
            self.charge("axis_solve_requests", amount)
        else:
            self.values["axis_solve_requests"] += amount

    def charge_physical_axis_solves(self, amount: int) -> None:
        """Charge actual cache-miss work under the active diagnostic budget mode."""
        if self.axis_budget_mode == "requests":
            self.values["axis_solves"] += amount
        else:
            self.charge("axis_solves", amount)

    def count_complete_label_gate(self, gate: _CompleteLabelGate) -> None:
        field = f"diagnostic_gate_{gate.value}"
        self.values[field] += 1

    def observe_partial_state(self, signature: _PartialSemanticSignature) -> None:
        self._observe_signature(
            signature, self.partial_state_signatures,
            "diagnostic_partial_state_observations",
            "diagnostic_partial_state_unique",
            "diagnostic_partial_state_repeats",
            "diagnostic_partial_state_truncated",
        )

    def observe_leaf_signature(self, signature: _CompleteLeafSignature) -> None:
        self._observe_signature(
            signature, self.leaf_signatures,
            "diagnostic_leaf_signature_observations",
            "diagnostic_leaf_signature_unique",
            "diagnostic_leaf_signature_repeats",
            "diagnostic_leaf_signature_truncated",
        )

    def _observe_signature(self, signature, stored, observations, unique,
                           repeats, truncated) -> None:
        self.values[observations] += 1
        if signature in stored:
            self.values[repeats] += 1
        elif len(stored) < self.signature_limit:
            stored.add(signature)
            self.values[unique] += 1
        else:
            self.values[truncated] += 1

    def snapshot(self) -> SearchCounters:
        return SearchCounters(**self.values)


@dataclass(frozen=True, slots=True)
class _Prepared:
    frames: tuple[str, ...]
    readings: tuple[PlaneReading, ...]
    read_by_index: Mapping[ReadingKey, PlaneReading]
    matches: tuple[plane_graph.PlaneMatch, ...]
    b_domains: tuple[tuple[ReadingKey, tuple[float, ...]], ...]
    q_support: tuple[tuple[tuple[int, int], ...], ...]
    match_readings: tuple[tuple[ReadingKey, ReadingKey], ...]
    match_pair_keys: tuple[PairKey, ...]
    match_possible_axes: tuple[tuple[tuple[int, int, int], ...], ...]
    all_row_mask: int
    pair_row_masks: Mapping[tuple[str, str], int]
    pair_q_row_masks: Mapping[tuple[str, str, int | None, int | None], int]
    pair_q_axis_row_masks: Mapping[tuple[str, str, int | None, int | None, int], int]
    b_value_row_masks: Mapping[tuple[ReadingKey, float], int]
    reading_row_masks: Mapping[ReadingKey, int]
    candidate_b_endpoint_mask: int
    match_index_by_identity: Mapping[int, int]
    duplicate_rows_deduplicated: int


@dataclass(frozen=True, slots=True)
class _RowMasks:
    all_row_mask: int
    pair_row_masks: Mapping[tuple[str, str], int]
    pair_q_row_masks: Mapping[tuple[str, str, int | None, int | None], int]
    pair_q_axis_row_masks: Mapping[tuple[str, str, int | None, int | None, int], int]
    b_value_row_masks: Mapping[tuple[ReadingKey, float], int]
    reading_row_masks: Mapping[ReadingKey, int]
    candidate_b_endpoint_mask: int
    match_index_by_identity: Mapping[int, int]


def _optional_float_key(value: float | None) -> tuple:
    """Sort optional diagnostics without manufacturing a numeric null value."""
    return (0,) if value is None else (1, float(value))


def _match_sort_key(match: plane_graph.PlaneMatch) -> tuple:
    return (
        match.left, int(match.left_plane), match.right, int(match.right_plane),
        int(match.quadrant_of_right), int(match.left_axis), int(match.right_axis),
        float(match.left_b), float(match.right_b), float(match.score),
        _optional_float_key(match.psr), float(match.peak), float(match.second_peak),
        int(match.overlap_cells),
        (0, 0.0) if match.integer_residual is None
        else (1, float(match.integer_residual)),
    )


def _pair_key(match: plane_graph.PlaneMatch) -> PairKey:
    left = (match.left, int(match.left_plane))
    right = (match.right, int(match.right_plane))
    first, second = sorted((left, right))
    return (first[0], first[1], second[0], second[1])


def _compile_row_masks(
    matches: tuple[plane_graph.PlaneMatch, ...],
    reading_endpoints: Sequence[tuple[ReadingKey, ReadingKey]],
    possible_axes: Sequence[tuple[tuple[int, int, int], ...]],
    domains: Sequence[tuple[ReadingKey, tuple[float, ...]]],
) -> _RowMasks:
    """Compile exact canonical row compatibility into Python-int bit masks."""
    all_row_mask = (1 << len(matches)) - 1
    pair_row_masks: dict[tuple[str, str], int] = {}
    exact_q_rows: dict[tuple[str, str, int, int], int] = {}
    exact_q_axis_rows: dict[tuple[str, str, int, int, int], int] = {}
    for row_index, (match, axis_options) in enumerate(zip(matches, possible_axes)):
        bit = 1 << row_index
        pair = (match.left, match.right)
        pair_row_masks[pair] = pair_row_masks.get(pair, 0) | bit
        for left_q, right_q, world_axis in axis_options:
            q_key = (match.left, match.right, left_q, right_q)
            exact_q_rows[q_key] = exact_q_rows.get(q_key, 0) | bit
            axis_key = (match.left, match.right, left_q, right_q, world_axis)
            exact_q_axis_rows[axis_key] = exact_q_axis_rows.get(axis_key, 0) | bit

    pair_q_row_masks: dict[tuple[str, str, int | None, int | None], int] = {}
    pair_q_axis_row_masks: dict[
        tuple[str, str, int | None, int | None, int], int
    ] = {}
    selectors: tuple[int | None, ...] = (None, 0, 1, 2, 3)
    for left_frame, right_frame in sorted(pair_row_masks):
        for wanted_left in selectors:
            for wanted_right in selectors:
                row_mask = 0
                axis_masks = [0, 0, 0]
                for left_q in range(4):
                    if wanted_left is not None and left_q != wanted_left:
                        continue
                    for right_q in range(4):
                        if wanted_right is not None and right_q != wanted_right:
                            continue
                        key = (left_frame, right_frame, left_q, right_q)
                        row_mask |= exact_q_rows.get(key, 0)
                        for axis in range(3):
                            axis_masks[axis] |= exact_q_axis_rows.get(
                                (*key, axis), 0)
                pair_q_row_masks[(left_frame, right_frame,
                                  wanted_left, wanted_right)] = row_mask
                for axis, axis_mask in enumerate(axis_masks):
                    pair_q_axis_row_masks[(left_frame, right_frame,
                                           wanted_left, wanted_right, axis)] = axis_mask

    b_value_row_masks: dict[tuple[ReadingKey, float], int] = {}
    reading_row_masks: dict[ReadingKey, int] = {}
    b_domain_keys = {key for key, _values in domains}
    candidate_b_incompatible = 0
    for row_index, (left_key, right_key) in enumerate(reading_endpoints):
        bit = 1 << row_index
        reading_row_masks[left_key] = reading_row_masks.get(left_key, 0) | bit
        reading_row_masks[right_key] = reading_row_masks.get(right_key, 0) | bit
        if left_key not in b_domain_keys or right_key not in b_domain_keys:
            candidate_b_incompatible |= bit
    for reading_key, values in domains:
        for value in values:
            incompatible = 0
            for row_index, (match, (left_key, right_key)) in enumerate(
                zip(matches, reading_endpoints)
            ):
                if ((left_key == reading_key and float(match.left_b) != value)
                        or (right_key == reading_key and float(match.right_b) != value)):
                    incompatible |= 1 << row_index
            b_value_row_masks[(reading_key, value)] = all_row_mask & ~incompatible
    return _RowMasks(
        all_row_mask=all_row_mask,
        pair_row_masks=pair_row_masks,
        pair_q_row_masks=pair_q_row_masks,
        pair_q_axis_row_masks=pair_q_axis_row_masks,
        b_value_row_masks=b_value_row_masks,
        reading_row_masks=reading_row_masks,
        candidate_b_endpoint_mask=all_row_mask & ~candidate_b_incompatible,
        match_index_by_identity={id(match): index for index, match in enumerate(matches)},
    )


def _empty_row_masks(matches: tuple[plane_graph.PlaneMatch, ...]) -> _RowMasks:
    return _RowMasks(
        all_row_mask=(1 << len(matches)) - 1,
        pair_row_masks={}, pair_q_row_masks={}, pair_q_axis_row_masks={},
        b_value_row_masks={}, reading_row_masks={},
        candidate_b_endpoint_mask=(1 << len(matches)) - 1,
        match_index_by_identity={id(match): index for index, match in enumerate(matches)},
    )


def _prepare(
    readings: Sequence[PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    *,
    compile_masks: bool = True,
) -> _Prepared | str:
    frames = tuple(sorted({reading.frame for reading in readings}))
    by_frame: dict[str, list[PlaneReading]] = {frame: [] for frame in frames}
    for reading in readings:
        by_frame[reading.frame].append(reading)

    read_by_index: dict[ReadingKey, PlaneReading] = {}
    ordered_readings: list[PlaneReading] = []
    for frame in frames:
        ordered = sorted(by_frame[frame], key=reading_order_key)
        for index, reading in enumerate(ordered):
            read_by_index[(frame, index)] = reading
            ordered_readings.append(reading)

    for match in matches:
        if (not isinstance(getattr(match, "left", None), str)
                or not isinstance(getattr(match, "right", None), str)
                or not match.left or not match.right
                or isinstance(getattr(match, "left_plane", None), bool)
                or not isinstance(getattr(match, "left_plane", None), int)
                or isinstance(getattr(match, "right_plane", None), bool)
                or not isinstance(getattr(match, "right_plane", None), int)
                or isinstance(getattr(match, "overlap_cells", None), bool)
                or not isinstance(getattr(match, "overlap_cells", None), int)
                or match.overlap_cells < 0):
            return "match has malformed identity/count fields"
        try:
            numeric = (
                float(match.left_b), float(match.right_b), float(match.score),
                None if match.psr is None else float(match.psr),
                float(match.peak), float(match.second_peak),
                None if match.integer_residual is None else float(match.integer_residual),
            )
            left_plane, right_plane = int(match.left_plane), int(match.right_plane)
            left_axis, right_axis = int(match.left_axis), int(match.right_axis)
            relative_q = int(match.quadrant_of_right)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return "match has malformed fields"
        if (any(isinstance(getattr(match, name, None), bool) for name in (
                    "left_b", "right_b", "score", "peak", "second_peak",
                    "integer_residual"))
                or (match.psr is not None and isinstance(match.psr, bool))
                or not all(math.isfinite(value) for value in numeric if value is not None)
                or numeric[0] <= 0.0 or numeric[1] <= 0.0
                or not 0 <= left_plane or not 0 <= right_plane
                or left_axis not in (0, 1, 2) or right_axis not in (0, 1, 2)
                or relative_q not in (0, 1, 2, 3)
                or isinstance(match.left_axis, bool) or isinstance(match.right_axis, bool)
                or isinstance(match.quadrant_of_right, bool)):
            return "match has invalid numeric/axis fields"
        if match.left >= match.right:
            return "match endpoints must be in canonical frame order"
    sorted_matches = tuple(sorted(matches, key=_match_sort_key))
    unique_matches: list[plane_graph.PlaneMatch] = []
    variant_fields: dict[tuple, tuple] = {}
    duplicate_rows_deduplicated = 0
    for match in sorted_matches:
        left_key = (match.left, int(match.left_plane))
        right_key = (match.right, int(match.right_plane))
        fields = (
            float(match.left_b), float(match.right_b), float(match.score),
            _optional_float_key(match.psr), float(match.peak), float(match.second_peak),
            int(match.overlap_cells),
            None if match.integer_residual is None else float(match.integer_residual),
            int(match.left_axis), int(match.right_axis),
        )
        variant_key = (
            _pair_key(match), int(match.quadrant_of_right),
            float(match.left_b), float(match.right_b),
        )
        previous = variant_fields.get(variant_key)
        if previous is None:
            variant_fields[variant_key] = fields
            unique_matches.append(match)
        elif previous == fields:
            duplicate_rows_deduplicated += 1
        else:
            return "conflicting duplicate pair/quadrant/b variant"
    canonical_matches = tuple(unique_matches)
    reading_endpoints: list[tuple[ReadingKey, ReadingKey]] = []
    pair_keys: list[PairKey] = []
    q_supports: list[tuple[tuple[int, int], ...]] = []
    possible_axes: list[tuple[tuple[int, int, int], ...]] = []
    b_values: dict[ReadingKey, set[float]] = {}

    for match in canonical_matches:
        left_key = (match.left, int(match.left_plane))
        right_key = (match.right, int(match.right_plane))
        if left_key not in read_by_index or right_key not in read_by_index:
            return "match references a missing frame/plane index"
        values = (
            float(match.left_b), float(match.right_b), float(match.score),
            None if match.psr is None else float(match.psr),
            float(match.peak), float(match.second_peak),
        )
        if not all(math.isfinite(value) for value in values if value is not None):
            return "match contains a non-finite numeric field"
        if match.left_axis not in (0, 1, 2) or match.right_axis not in (0, 1, 2):
            return "match axis is outside 0..2"
        if match.quadrant_of_right not in (0, 1, 2, 3):
            return "relative quadrant is outside 0..3"
        left_reading = read_by_index[left_key]
        right_reading = read_by_index[right_key]
        if (int(match.left_axis) != int(left_reading.axis)
                or int(match.right_axis) != int(right_reading.axis)):
            return "match axes must match referenced plane readings"
        left_intrinsic = {float(alternative.blocks_per_period) for alternative in left_reading.alternatives}
        right_intrinsic = {float(alternative.blocks_per_period) for alternative in right_reading.alternatives}
        if match.left_b in left_intrinsic:
            b_values.setdefault(left_key, set()).add(float(match.left_b))
        if match.right_b in right_intrinsic:
            b_values.setdefault(right_key, set()).add(float(match.right_b))
        reading_endpoints.append((left_key, right_key))
        pair_keys.append(_pair_key(match))
        q_pairs: list[tuple[int, int]] = []
        axis_rows: list[tuple[int, int, int]] = []
        for left_q in range(4):
            for right_q in range(4):
                if (right_q - left_q) % 4 != int(match.quadrant_of_right):
                    continue
                left_world = plane_graph.axis_transform(int(match.left_axis), left_q)[0]
                right_world = plane_graph.axis_transform(int(match.right_axis), right_q)[0]
                if left_world != right_world:
                    continue
                q_pairs.append((left_q, right_q))
                axis_rows.append((left_q, right_q, left_world))
        q_supports.append(tuple(q_pairs))
        possible_axes.append(tuple(axis_rows))

    domains = tuple(
        (key, tuple(sorted(values)))
        for key, values in sorted(b_values.items())
        if values
    )
    row_masks = (
        _compile_row_masks(canonical_matches, reading_endpoints, possible_axes, domains)
        if compile_masks else _empty_row_masks(canonical_matches)
    )

    return _Prepared(
        frames=frames,
        readings=tuple(ordered_readings),
        read_by_index=read_by_index,
        matches=canonical_matches,
        b_domains=domains,
        q_support=tuple(q_supports),
        match_readings=tuple(reading_endpoints),
        match_pair_keys=tuple(pair_keys),
        match_possible_axes=tuple(possible_axes),
        all_row_mask=row_masks.all_row_mask,
        pair_row_masks=row_masks.pair_row_masks,
        pair_q_row_masks=row_masks.pair_q_row_masks,
        pair_q_axis_row_masks=row_masks.pair_q_axis_row_masks,
        b_value_row_masks=row_masks.b_value_row_masks,
        reading_row_masks=row_masks.reading_row_masks,
        candidate_b_endpoint_mask=row_masks.candidate_b_endpoint_mask,
        match_index_by_identity=row_masks.match_index_by_identity,
        duplicate_rows_deduplicated=duplicate_rows_deduplicated,
    )


def _candidate_records(
    prepared: _Prepared,
    quadrants: Mapping[str, int],
    b_values: Mapping[ReadingKey, float],
    work: _Work,
) -> tuple[plane_graph.PlaneMatch, ...]:
    if not work.optimized:
        return _candidate_records_legacy(prepared, quadrants, b_values, work)

    # Masks encode the same per-row predicates, but logical charges remain the
    # number and order of checks the original canonical scan would perform.
    work.charge_logical_scan("compatibility_checks", len(prepared.matches))
    mask = prepared.candidate_b_endpoint_mask
    for left_frame, right_frame in sorted(prepared.pair_row_masks):
        left_q, right_q = quadrants[left_frame], quadrants[right_frame]
        compatible = prepared.pair_q_row_masks[
            (left_frame, right_frame, left_q, right_q)]
        pair_rows = prepared.pair_row_masks[(left_frame, right_frame)]
        incompatible = pair_rows & ~compatible
        mask &= prepared.all_row_mask ^ incompatible
    for reading_key, _values in prepared.b_domains:
        if reading_key not in b_values:
            mask &= prepared.all_row_mask ^ prepared.reading_row_masks.get(reading_key, 0)
        else:
            mask &= prepared.b_value_row_masks[
                (reading_key, float(b_values[reading_key]))]
    return tuple(prepared.matches[index] for index in _row_indices(mask))


def _candidate_records_legacy(
    prepared: _Prepared,
    quadrants: Mapping[str, int],
    b_values: Mapping[ReadingKey, float],
    work: _Work,
) -> tuple[plane_graph.PlaneMatch, ...]:
    active: list[plane_graph.PlaneMatch] = []
    for match, (left_key, right_key) in zip(prepared.matches, prepared.match_readings):
        work.charge("compatibility_checks")
        if (quadrants[match.right] - quadrants[match.left]) % 4 != match.quadrant_of_right:
            continue
        if (plane_graph.axis_transform(match.left_axis, quadrants[match.left])[0]
                != plane_graph.axis_transform(match.right_axis, quadrants[match.right])[0]):
            continue
        if b_values.get(left_key) != match.left_b or b_values.get(right_key) != match.right_b:
            continue
        active.append(match)
    return tuple(active)


def _constraint_first_variable_order(
    prepared: _Prepared,
) -> tuple[tuple[str, object], ...]:
    """Return a label-blind static order: small domains, then high row degree.

    The anchored first-frame quadrant and singleton b domains are constants,
    not search variables. Degree is the count of canonical match rows incident
    on a frame (q) or reading (b); canonical names/indices break all ties.
    """
    variables: list[tuple[tuple, tuple[str, object]]] = []
    frame_degrees = {frame: 0 for frame in prepared.frames}
    for match in prepared.matches:
        frame_degrees[match.left] += 1
        frame_degrees[match.right] += 1
    for frame in prepared.frames[1:]:
        variable = ("q", frame)
        variables.append(((4, -frame_degrees[frame], 0, frame, -1), variable))
    for key, values in prepared.b_domains:
        if len(values) <= 1:
            continue
        degree = prepared.reading_row_masks.get(key, 0).bit_count()
        variable = ("b", key)
        variables.append(((len(values), -degree, 1, key[0], key[1]), variable))
    variables.sort(key=lambda item: item[0])
    return tuple(variable for _sort_key, variable in variables)


def _refine_quadrant_rows(
    prepared: _Prepared,
    row_mask: int,
    quadrants: Mapping[str, int | None],
    changed_frame: str,
) -> int:
    """Intersect rows with exact q compatibility on pairs incident to a frame."""
    for left_frame, right_frame in sorted(prepared.pair_row_masks):
        if changed_frame not in (left_frame, right_frame):
            continue
        left_q = quadrants[left_frame]
        right_q = quadrants[right_frame]
        allowed = prepared.pair_q_row_masks[
            (left_frame, right_frame, left_q, right_q)]
        pair_rows = prepared.pair_row_masks[(left_frame, right_frame)]
        row_mask &= prepared.all_row_mask ^ (pair_rows & ~allowed)
    return row_mask


def _refine_blocks_per_period_row_mask(
    prepared: _Prepared,
    row_mask: int,
    reading: ReadingKey,
    value: float,
) -> int:
    return row_mask & prepared.b_value_row_masks[(reading, float(value))]


def _constraint_first_potential_axes(
    prepared: _Prepared,
    quadrants: Mapping[str, int | None],
    feasible_rows: int,
    work: _Work,
) -> frozenset[int]:
    """Match the reference partial-axis bound using incrementally feasible rows."""
    # Preserve one reference-equivalent row-pass charge per search node. The
    # actual update and lookup work below is performed with canonical bitmasks.
    work.charge_logical_scan("compatibility_checks", len(prepared.matches))
    axes: set[int] = set()
    for left_frame, right_frame in sorted(prepared.pair_row_masks):
        left_q = quadrants[left_frame]
        right_q = quadrants[right_frame]
        for axis in range(3):
            possible = prepared.pair_q_axis_row_masks[
                (left_frame, right_frame, left_q, right_q, axis)]
            if possible & feasible_rows:
                axes.add(axis)
    return frozenset(axes)


def _constraint_first_row_mask(
    prepared: _Prepared,
    quadrants: Mapping[str, int | None],
    b_values: Mapping[ReadingKey, float | None],
) -> int:
    """Build an exact partial feasible-row mask for tests and root setup."""
    row_mask = prepared.all_row_mask
    for frame in prepared.frames:
        value = quadrants.get(frame)
        if value is not None:
            row_mask = _refine_quadrant_rows(prepared, row_mask, quadrants, frame)
    for reading, value in sorted(b_values.items()):
        if value is not None:
            row_mask = _refine_blocks_per_period_row_mask(
                prepared, row_mask, reading, value)
    return row_mask


def _row_indices(mask: int):
    """Yield set bits in ascending canonical row order."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _potential_axes(
    prepared: _Prepared,
    quadrants: Mapping[str, int | None],
    b_values: Mapping[ReadingKey, float | None],
    work: _Work,
) -> frozenset[int]:
    if work.optimized:
        work.charge_logical_scan("compatibility_checks", len(prepared.matches))
        b_compatible = prepared.all_row_mask
        for reading_key, value in sorted(b_values.items()):
            if value is not None:
                b_compatible &= prepared.b_value_row_masks[(reading_key, float(value))]
        axes: set[int] = set()
        for left_frame, right_frame in sorted(prepared.pair_row_masks):
            left_q = quadrants[left_frame]
            right_q = quadrants[right_frame]
            for axis in range(3):
                rows = prepared.pair_q_axis_row_masks[
                    (left_frame, right_frame, left_q, right_q, axis)]
                if rows & b_compatible:
                    axes.add(axis)
        return frozenset(axes)

    axes: set[int] = set()
    for index, (match, (left_key, right_key)) in enumerate(
        zip(prepared.matches, prepared.match_readings)
    ):
        work.charge("compatibility_checks")
        left_b, right_b = b_values.get(left_key), b_values.get(right_key)
        if (left_b is not None and left_b != match.left_b) or (
            right_b is not None and right_b != match.right_b
        ):
            continue
        left_q = quadrants[match.left]
        right_q = quadrants[match.right]
        for q_left, q_right, world_axis in prepared.match_possible_axes[index]:
            if left_q is not None and left_q != q_left:
                continue
            if right_q is not None and right_q != q_right:
                continue
            axes.add(world_axis)
    return frozenset(axes)


def _evaluate_objective(
    prepared: _Prepared,
    quadrants: Mapping[str, int],
    matches: Sequence[plane_graph.PlaneMatch],
    work: _Work,
):
    """Reuse exact per-run objectives and charge actual axis solves on misses."""
    if not work.optimized:
        axis_count = _objective_axis_count_for(prepared, quadrants, matches)
        work.charge_axis_solve_request(axis_count)
        work.charge_physical_axis_solves(axis_count)
        result = plane_graph._evaluate_assignment(
            prepared.frames, quadrants, matches, prepared.read_by_index)
        work.values["physical_axis_cache_misses"] += _objective_axis_count(result)
        return result
    signature = (
        tuple((frame, int(quadrants[frame])) for frame in prepared.frames),
        tuple(prepared.match_index_by_identity[id(match)] for match in matches),
    )
    if signature in work.objective_cache:
        result = work.objective_cache[signature]
        axis_count = _objective_axis_count(result)
        work.charge_axis_solve_request(axis_count)
        work.values["objective_cache_hits"] += 1
        work.values["physical_axis_cache_hits"] += axis_count
        return result
    axis_count = _objective_axis_count_for(prepared, quadrants, matches)
    work.charge_axis_solve_request(axis_count)
    # Check the exact miss cost before entering the least-squares objective.
    # Cache hits request the same results but consume no physical solve budget.
    work.charge_physical_axis_solves(axis_count)
    work.values["objective_cache_misses"] += 1
    work.values["objective_cache_unique_signatures"] += 1
    result = plane_graph._evaluate_assignment(
        prepared.frames, quadrants, matches, prepared.read_by_index)
    work.values["physical_axis_cache_misses"] += _objective_axis_count(result)
    work.objective_cache[signature] = result
    return result


def _objective_axis_count_for(
    prepared: _Prepared,
    quadrants: Mapping[str, int],
    matches: Sequence[plane_graph.PlaneMatch],
) -> int:
    """Predict exact distinct axes solved by a valid objective cache miss.

    `_prepare` validates identities, axes, and scale alternatives. This helper
    mirrors `_evaluate_assignment`'s remaining active-row, world-axis, and
    distance checks, so a cache-miss budget is preflighted before any LSQ.
    Invalid objectives perform no axis solves and are left for the evaluator
    to reject.
    """
    active: list[plane_graph.PlaneMatch] = []
    for match in matches:
        if (match.left not in quadrants or match.right not in quadrants):
            continue
        if ((quadrants[match.right] - quadrants[match.left]) % 4
                != match.quadrant_of_right):
            continue
        active.append(match)
    axes: set[int] = set()
    for match in active:
        left = prepared.read_by_index.get((match.left, match.left_plane))
        right = prepared.read_by_index.get((match.right, match.right_plane))
        if left is None or right is None:
            return 0
        world, _left_sign = plane_graph.axis_transform(left.axis, quadrants[match.left])
        right_world, _right_sign = plane_graph.axis_transform(right.axis, quadrants[match.right])
        if right_world != world:
            return 0
        if (left.distance_for(match.left_b) is None
                or right.distance_for(match.right_b) is None):
            return 0
        axes.add(world)
    return len(axes)


def _objective_axis_count(result: object) -> int:
    """Count axis result sets physically computed or reused with an objective."""
    if not isinstance(result, tuple) or len(result) != 3:
        return 0
    detail = result[2]
    if not isinstance(detail, dict):
        return 0
    resultants = detail.get("resultants")
    return len(resultants) if isinstance(resultants, dict) else 0


def _semantic_feasible_row_mask(
    prepared: _Prepared,
    quadrants: Mapping[str, int | None],
    b_values: Mapping[ReadingKey, float | None],
) -> int:
    """Return the exact canonical row set still feasible under assigned labels.

    This is diagnostic-only and does not charge search budgets. A row remains
    feasible when its referenced q/b labels admit at least one of its prepared
    world-axis options. The bit positions are the immutable canonical order in
    ``prepared.matches``.
    """
    mask = 0
    for index, (match, (left_key, right_key)) in enumerate(
        zip(prepared.matches, prepared.match_readings)
    ):
        left_b, right_b = b_values.get(left_key), b_values.get(right_key)
        if ((left_b is not None and float(match.left_b) != float(left_b))
                or (right_b is not None and float(match.right_b) != float(right_b))):
            continue
        left_q, right_q = quadrants.get(match.left), quadrants.get(match.right)
        possible = any(
            (left_q is None or int(left_q) == option_left_q)
            and (right_q is None or int(right_q) == option_right_q)
            for option_left_q, option_right_q, _axis
            in prepared.match_possible_axes[index]
        )
        if possible:
            mask |= 1 << index
    return mask


def _semantic_possible_axes(
    prepared: _Prepared,
    quadrants: Mapping[str, int | None],
    feasible_row_mask: int,
) -> tuple[int, ...]:
    axes: set[int] = set()
    for index in _row_indices(feasible_row_mask):
        left_q, right_q = quadrants[prepared.matches[index].left], quadrants[
            prepared.matches[index].right]
        for option_left_q, option_right_q, axis in prepared.match_possible_axes[index]:
            if ((left_q is None or int(left_q) == option_left_q)
                    and (right_q is None or int(right_q) == option_right_q)):
                axes.add(axis)
    return tuple(sorted(axes))


def _partial_semantic_signature(
    prepared: _Prepared,
    quadrants: Mapping[str, int | None],
    b_values: Mapping[ReadingKey, float | None],
    depth: int,
) -> _PartialSemanticSignature:
    feasible_mask = _semantic_feasible_row_mask(prepared, quadrants, b_values)
    active_mask = feasible_mask & prepared.candidate_b_endpoint_mask
    active_rows = tuple(_row_indices(active_mask))
    return _PartialSemanticSignature(
        frame_order=prepared.frames,
        depth=depth,
        quadrant_labels=tuple((frame, quadrants[frame]) for frame in prepared.frames),
        blocks_per_period_labels=tuple(
            (key[0], key[1], b_values.get(key))
            for key, _values in prepared.b_domains
        ),
        feasible_row_mask=feasible_mask,
        active_row_indices=active_rows,
        possible_axes=_semantic_possible_axes(prepared, quadrants, feasible_mask),
    )


def _complete_leaf_signature(
    prepared: _Prepared,
    quadrants: Mapping[str, int],
    b_values: Mapping[ReadingKey, float],
) -> _CompleteLeafSignature:
    feasible_mask = _semantic_feasible_row_mask(prepared, quadrants, b_values)
    active_mask = feasible_mask & prepared.candidate_b_endpoint_mask
    active_indices = tuple(_row_indices(active_mask))
    objective_rows = []
    for index in active_indices:
        match = prepared.matches[index]
        left = prepared.read_by_index[(match.left, int(match.left_plane))]
        right = prepared.read_by_index[(match.right, int(match.right_plane))]
        world_axis, left_sign = plane_graph.axis_transform(
            left.axis, quadrants[match.left])
        right_world, right_sign = plane_graph.axis_transform(
            right.axis, quadrants[match.right])
        if right_world != world_axis:
            raise ValueError("prepared match axes disagree under the assigned quadrants")
        left_face = left_sign * left.side * left.distance_for(match.left_b)
        right_face = right_sign * right.side * right.distance_for(match.right_b)
        objective_rows.append((
            index, world_axis, float(left_face), float(right_face), float(match.score)))
    return _CompleteLeafSignature(
        frame_order=prepared.frames,
        quadrant_labels=tuple((frame, int(quadrants[frame])) for frame in prepared.frames),
        blocks_per_period_labels=tuple(
            (key[0], key[1], float(b_values[key]))
            for key, _values in prepared.b_domains
        ),
        feasible_row_mask=feasible_mask,
        active_row_indices=active_indices,
        objective_rows=tuple(objective_rows),
    )


def _metrics_for(
    candidate: JointCandidate,
    group_subset: frozenset[PairKey],
    prepared: _Prepared,
    work: _Work,
) -> tuple[dict[int, float], dict[int, float], float] | None:
    qmap = candidate.quadrant_map
    bmap = candidate.b_map
    matches = tuple(
        match for match in candidate._active_matches
        if _pair_key(match) in group_subset
    )
    if not matches:
        return None
    work.charge("objective_rows", len(matches))
    objective, alternative, detail = _evaluate_objective(
        prepared, qmap, matches, work)
    work.values["shared_objective_evaluations"] += 1
    if objective is None or detail is None:
        return None
    axes = tuple(sorted(detail["resultants"]))
    return (
        {int(axis): float(detail["resultants"][axis]) for axis in axes},
        {int(axis): float(detail["residuals"][axis]) for axis in axes},
        float(sum(match.score for match in matches) / len(matches)),
    )


def _dominates(
    left: JointCandidate,
    right: JointCandidate,
    prepared: _Prepared,
    work: _Work,
) -> bool:
    if not left.pair_groups.issuperset(right.pair_groups):
        return False
    if not left.covered_frames.issuperset(right.covered_frames):
        return False
    left_axes, right_axes = left.axis_group_map, right.axis_group_map
    if any(not left_axes.get(axis, frozenset()).issuperset(groups)
           for axis, groups in right_axes.items()):
        return False
    common = left.pair_groups & right.pair_groups
    left_metrics = _metrics_for(left, common, prepared, work)
    right_metrics = _metrics_for(right, common, prepared, work)
    if left_metrics is None or right_metrics is None:
        return False
    left_resultants, left_residuals, left_score = left_metrics
    right_resultants, right_residuals, right_score = right_metrics
    for axis, value in right_resultants.items():
        if axis not in left_resultants or left_resultants[axis] + 1e-12 < value:
            return False
    for axis, value in right_residuals.items():
        if axis not in left_residuals or left_residuals[axis] > value + 1e-12:
            return False
    if left_score + 1e-12 < right_score:
        return False

    strict_objective = (
        any(left_resultants[axis] > value + 1e-12 for axis, value in right_resultants.items())
        or any(left_residuals[axis] + 1e-12 < value for axis, value in right_residuals.items())
        or left_score > right_score + 1e-12
    )
    # Coverage inclusion is necessary to prevent sparse evidence from winning,
    # but extra-only groups cannot certify truth by themselves. Strict
    # improvement must occur on an objective recomputed over shared evidence.
    return strict_objective


def _make_candidate(
    prepared: _Prepared,
    quadrants: Mapping[str, int],
    b_values: Mapping[ReadingKey, float],
    active: tuple[plane_graph.PlaneMatch, ...],
    integer_tolerance: float,
    min_resultant: float,
    work: _Work,
) -> tuple[JointCandidate | None, _CompleteLabelGate]:
    if not active:
        return None, _CompleteLabelGate.NO_COMPATIBLE_RECORDS
    work.charge("objective_rows", len(active))
    evaluated = _evaluate_objective(prepared, quadrants, active, work)
    if evaluated is None:
        return None, _CompleteLabelGate.OBJECTIVE_REJECTED
    cost, alternative, detail = evaluated
    axes = tuple(sorted(detail["resultants"]))
    # A one-b-per-reading label has already removed cross-row b ambiguity.
    # Repeated records for the same pair/q are therefore retained as evidence;
    # the legacy duplicate_b sentinel cannot be reused here.
    if len(axes) < 3:
        return None, _CompleteLabelGate.INSUFFICIENT_AXES
    if any(float(detail["resultants"][axis]) < min_resultant for axis in axes):
        return None, _CompleteLabelGate.MINIMUM_RESULTANT
    if any(float(detail["residuals"][axis]) > integer_tolerance for axis in axes):
        return None, _CompleteLabelGate.INTEGER_TOLERANCE

    group_axes: dict[int, set[PairKey]] = {0: set(), 1: set(), 2: set()}
    pair_groups: set[PairKey] = set()
    covered_frames: set[str] = set()
    covered_readings: set[ReadingKey] = set()
    for match in active:
        left_key = (match.left, int(match.left_plane))
        right_key = (match.right, int(match.right_plane))
        axis = plane_graph.axis_transform(match.left_axis, quadrants[match.left])[0]
        pair_key = _pair_key(match)
        pair_groups.add(pair_key)
        group_axes[axis].add(pair_key)
        covered_frames.update((match.left, match.right))
        covered_readings.update((left_key, right_key))

    mean_score = float(sum(match.score for match in active) / len(active))
    return JointCandidate(
        quadrants=tuple((frame, int(quadrants[frame])) for frame in prepared.frames),
        blocks_per_period=tuple(
            (frame, plane, float(value))
            for (frame, plane), value in sorted(b_values.items())
            if (frame, plane) in covered_readings
        ),
        pair_groups=frozenset(pair_groups),
        covered_frames=frozenset(covered_frames),
        covered_readings=frozenset(covered_readings),
        axis_pair_groups=tuple((axis, frozenset(group_axes[axis])) for axis in range(3)),
        resultants=tuple((axis, float(detail["resultants"][axis])) for axis in axes),
        residuals=tuple((axis, float(detail["residuals"][axis])) for axis in axes),
        translations=tuple(
            (frame, tuple(float(value) for value in detail["translations"][frame]))
            for frame in prepared.frames
        ),
        axis_observations=tuple(int(value) for value in alternative.axis_observations),
        scale_known=len(axes) >= 2,
        mean_match_score=mean_score,
        cost=float(cost),
        _active_matches=active,
    ), _CompleteLabelGate.ACCEPTED


def _same_evidence_objective(first: JointCandidate, second: JointCandidate) -> bool:
    if (first.pair_groups != second.pair_groups
            or first.covered_frames != second.covered_frames
            or first.axis_pair_groups != second.axis_pair_groups):
        return False
    fr, sr = first.resultant_map, second.resultant_map
    fd, sd = first.residual_map, second.residual_map
    return (
        fr.keys() == sr.keys() and fd.keys() == sd.keys()
        and all(abs(fr[axis] - sr[axis]) <= 1e-12 for axis in fr)
        and all(abs(fd[axis] - sd[axis]) <= 1e-12 for axis in fd)
        and abs(first.mean_match_score - second.mean_match_score) <= 1e-12
    )


def _quadrant_only_half_turn_ambiguity(candidates: Sequence[JointCandidate]) -> bool:
    if len(candidates) < 2:
        return False
    first = candidates[0]
    if any(candidate.blocks_per_period != first.blocks_per_period
           or not _same_evidence_objective(first, candidate)
           for candidate in candidates[1:]):
        return False
    changed = False
    base = first.quadrant_map
    for candidate in candidates[1:]:
        other = candidate.quadrant_map
        differences = {(other[frame] - base[frame]) % 4 for frame in base}
        if not differences.issubset({0, 2}) or 2 not in differences:
            return False
        changed = True
    return changed


def _candidate_payload(candidate: JointCandidate) -> dict:
    return {
        "q": list(candidate.quadrants),
        "b": list(candidate.blocks_per_period),
        "groups": sorted(candidate.pair_groups),
        "frames": sorted(candidate.covered_frames),
        "readings": sorted(candidate.covered_readings),
        "axis_groups": [(axis, sorted(groups)) for axis, groups in candidate.axis_pair_groups],
        "resultants": list(candidate.resultants),
        "residuals": list(candidate.residuals),
        "translations": list(candidate.translations),
        "axis_observations": list(candidate.axis_observations),
        "scale_known": candidate.scale_known,
        "mean_match_score": candidate.mean_match_score,
        "cost": candidate.cost,
        "matches": [_match_sort_key(match) for match in candidate._active_matches],
    }


def _result(
    status: str,
    reason: str,
    candidates: Sequence[JointCandidate],
    work: _Work,
) -> SelectorResult:
    ordered = tuple(sorted(
        candidates,
        key=lambda candidate: (
            candidate.quadrants,
            candidate.blocks_per_period,
            tuple(sorted(candidate.pair_groups)),
        ),
    ))
    payload = json.dumps(
        [_candidate_payload(candidate) for candidate in ordered],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest() if status != "budget_exhausted" else None
    selected = ordered[0] if status == "selected" and ordered else None
    return SelectorResult(status, reason, selected, ordered, work.snapshot(), sha)


def _finish(
    frontier: Sequence[JointCandidate],
    work: _Work,
    failure_reason: str | None,
) -> SelectorResult:
    if not frontier:
        return _result("no_candidate", failure_reason or "no_full_label_passed_existing_gates", (), work)
    if len(frontier) == 1:
        return _result("selected", "", frontier, work)
    if _quadrant_only_half_turn_ambiguity(frontier):
        return _result("ambiguous_quadrant", "equivalent_half_turn_labels", frontier, work)
    return _result("b_unresolved", "multiple_pareto_maxima", frontier, work)


def _accept_complete_label(
    prepared: _Prepared,
    quadrants: Mapping[str, int],
    b_values: Mapping[ReadingKey, float],
    integer_tolerance: float,
    min_resultant: float,
    work: _Work,
    candidates: list[JointCandidate],
    *,
    active_matches: tuple[plane_graph.PlaneMatch, ...] | None = None,
) -> str:
    work.values["complete_labels"] += 1
    work.observe_leaf_signature(
        _complete_leaf_signature(prepared, quadrants, b_values))
    try:
        if active_matches is None:
            active = _candidate_records(prepared, quadrants, b_values, work)
        else:
            # Retain the same logical charge as the reference complete-row scan;
            # the tuple itself comes from the accumulated mask in canonical order.
            work.charge_logical_scan("compatibility_checks", len(prepared.matches))
            active = active_matches
        candidate, gate = _make_candidate(
            prepared, quadrants, b_values, active,
            integer_tolerance, min_resultant, work)
    except _BudgetExceeded:
        work.count_complete_label_gate(_CompleteLabelGate.BUDGET_ABORTED)
        raise
    work.count_complete_label_gate(gate)
    if candidate is None:
        return ("no_compatible_records" if gate == _CompleteLabelGate.NO_COMPATIBLE_RECORDS
                else "existing_geometry_or_axis_gate")
    work.values["evaluated_candidates"] += 1
    try:
        work.charge("candidate_certificates_buffered")
        work.charge("buffered_match_rows", len(candidate._active_matches))
    except _BudgetExceeded:
        # The complete label passed geometry/numeric gates, but a later
        # result-buffering limit refused to append it. Keep this separate from
        # the mutually exclusive terminal gate histogram.
        work.values["diagnostic_post_accept_buffering_aborts"] += 1
        raise
    candidates.append(candidate)
    return ""


def _all_pairs_frontier(
    candidates: Sequence[JointCandidate],
    prepared: _Prepared,
    work: _Work,
) -> tuple[JointCandidate, ...]:
    """Compare every ordered pair; no online deletion assumes transitivity."""
    def dominates(left: JointCandidate, right: JointCandidate) -> bool:
        work.charge("pareto_comparisons")
        return _dominates(left, right, prepared, work)

    indices = _nondominated_indices(candidates, dominates)
    return tuple(candidates[index] for index in indices)


def _nondominated_indices(
    candidates: Sequence[_T],
    dominates: Callable[[_T, _T], bool],
) -> tuple[int, ...]:
    """Order-independent all-pairs reduction, including non-transitive relations."""
    dominated = [False] * len(candidates)
    for left_index, left in enumerate(candidates):
        for right_index, right in enumerate(candidates):
            if left_index != right_index and dominates(left, right):
                dominated[right_index] = True
    return tuple(index for index, is_dominated in enumerate(dominated) if not is_dominated)


def _prepare_search(
    readings: Sequence[PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    budget: SearchBudget | None,
    *,
    optimized: bool = True,
    axis_budget_mode: str = "physical",
) -> tuple[_Prepared | None, _Work]:
    work = _Work(
        budget, optimized=optimized, axis_budget_mode=axis_budget_mode)
    prepared_result = _prepare(readings, matches, compile_masks=optimized)
    if isinstance(prepared_result, str):
        return None, work
    work.values["duplicate_rows_deduplicated"] = prepared_result.duplicate_rows_deduplicated
    return prepared_result, work


def _invalid_or_empty(
    prepared: _Prepared,
    work: _Work,
) -> SelectorResult | None:
    if not prepared.frames:
        return _result("no_candidate", "no_plane_reading", (), work)
    if not prepared.matches or not prepared.b_domains:
        return _result("no_candidate", "no_matches", (), work)
    return None


def _search_bounded(
    readings: Sequence[PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    budget: SearchBudget,
    *,
    min_resultant: float,
    integer_tolerance: float,
    optimized: bool = True,
    variable_order: str = "natural",
    axis_budget_mode: str = "physical",
) -> SelectorResult:
    prepared, work = _prepare_search(
        readings, matches, budget, optimized=optimized,
        axis_budget_mode=axis_budget_mode)
    if prepared is None:
        prepared_result = _prepare(readings, matches, compile_masks=optimized)
        reason = prepared_result if isinstance(prepared_result, str) else "invalid input"
        return _result("invalid_input", reason, (), work)
    empty = _invalid_or_empty(prepared, work)
    if empty is not None:
        return empty

    domains = dict(prepared.b_domains)
    q_values: dict[str, int | None] = {frame: None for frame in prepared.frames}
    q_values[prepared.frames[0]] = 0
    b_values: dict[ReadingKey, float | None] = {
        key: values[0] if len(values) == 1 else None
        for key, values in prepared.b_domains
    }
    if variable_order not in ("natural", "constraint_first"):
        return _result("invalid_input", "unknown variable ordering", (), work)
    variables = (
        tuple(("q", frame) for frame in prepared.frames[1:])
        + tuple(("b", key) for key, values in prepared.b_domains if len(values) > 1)
        if variable_order == "natural" else _constraint_first_variable_order(prepared)
    )
    candidates: list[JointCandidate] = []
    failure_reason: str | None = None

    if variable_order == "constraint_first":
        # The anchored q and singleton b values are fixed constraints. Carry
        # their compatible rows once, then refine only as each variable is set.
        feasible_rows = prepared.all_row_mask
        feasible_rows = _refine_quadrant_rows(
            prepared, feasible_rows, q_values, prepared.frames[0])
        for reading_key, values in prepared.b_domains:
            if len(values) == 1:
                feasible_rows = _refine_blocks_per_period_row_mask(
                    prepared, feasible_rows, reading_key, values[0])

        def visit_constraint_first(
            depth: int,
            row_mask: int,
        ) -> None:
            nonlocal failure_reason
            work.charge("search_nodes")
            work.observe_partial_state(_partial_semantic_signature(
                prepared, q_values, b_values, depth))
            possible_axes = _constraint_first_potential_axes(
                prepared, q_values, row_mask, work)
            if len(possible_axes) < 3:
                work.values["pruned_insufficient_axis"] += 1
                failure_reason = "insufficient_axis_coverage"
                return
            if depth == len(variables):
                concrete_q = {frame: int(value) for frame, value in q_values.items()}
                concrete_b = {key: float(value) for key, value in b_values.items()}
                active_mask = row_mask & prepared.candidate_b_endpoint_mask
                active_matches = tuple(
                    prepared.matches[index] for index in _row_indices(active_mask))
                failure_reason = _accept_complete_label(
                    prepared, concrete_q, concrete_b, integer_tolerance,
                    min_resultant, work, candidates,
                    active_matches=active_matches) or failure_reason
                return

            kind, key = variables[depth]
            domain = range(4) if kind == "q" else domains[key]  # type: ignore[index]
            for value in domain:
                if kind == "q":
                    frame = str(key)
                    q_values[frame] = int(value)
                    child_mask = _refine_quadrant_rows(
                        prepared, row_mask, q_values, frame)
                    visit_constraint_first(depth + 1, child_mask)
                    q_values[frame] = None
                else:
                    reading_key = key  # type: ignore[assignment]
                    b_values[reading_key] = float(value)
                    child_mask = _refine_blocks_per_period_row_mask(
                        prepared, row_mask, reading_key, float(value))
                    visit_constraint_first(depth + 1, child_mask)
                    b_values[reading_key] = None

        try:
            visit_constraint_first(0, feasible_rows)
            frontier = _all_pairs_frontier(candidates, prepared, work)
        except _BudgetExceeded as error:
            return _result("budget_exhausted", error.counter, (), work)
        return _finish(frontier, work, failure_reason)

    def visit(depth: int) -> None:
        nonlocal failure_reason
        work.charge("search_nodes")
        work.observe_partial_state(_partial_semantic_signature(
            prepared, q_values, b_values, depth))
        possible_axes = _potential_axes(prepared, q_values, b_values, work)
        if len(possible_axes) < 3:
            work.values["pruned_insufficient_axis"] += 1
            failure_reason = "insufficient_axis_coverage"
            return
        if depth == len(variables):
            concrete_q = {frame: int(value) for frame, value in q_values.items()}
            concrete_b = {key: float(value) for key, value in b_values.items()}
            failure_reason = _accept_complete_label(
                prepared, concrete_q, concrete_b, integer_tolerance,
                min_resultant, work, candidates) or failure_reason
            return

        kind, key = variables[depth]
        domain = range(4) if kind == "q" else domains[key]  # type: ignore[index]
        for value in domain:
            if kind == "q":
                q_values[str(key)] = int(value)
            else:
                b_values[key] = float(value)  # type: ignore[index]
            visit(depth + 1)
        if kind == "q":
            q_values[str(key)] = None
        else:
            b_values[key] = None  # type: ignore[index]

    try:
        visit(0)
        frontier = _all_pairs_frontier(candidates, prepared, work)
    except _BudgetExceeded as error:
        return _result("budget_exhausted", error.counter, (), work)
    return _finish(frontier, work, failure_reason)


def _search_oracle(
    readings: Sequence[PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    *,
    min_resultant: float,
    integer_tolerance: float,
) -> SelectorResult:
    """Independent small-case oracle: plain Cartesian products, no pruning."""
    prepared_result = _prepare(readings, matches, compile_masks=False)
    # The Cartesian implementation intentionally retains the uncached row and
    # objective path as an independent small-case reference oracle.
    work = _Work(None, optimized=False)
    if isinstance(prepared_result, str):
        return _result("invalid_input", prepared_result, (), work)
    prepared = prepared_result
    work.values["duplicate_rows_deduplicated"] = prepared.duplicate_rows_deduplicated
    empty = _invalid_or_empty(prepared, work)
    if empty is not None:
        return empty

    quadrants_for = product(range(4), repeat=max(0, len(prepared.frames) - 1))
    candidates: list[JointCandidate] = []
    failure_reason: str | None = None
    b_domains = tuple((key, values) for key, values in prepared.b_domains)
    for q_tail in quadrants_for:
        q_values = {prepared.frames[0]: 0}
        q_values.update(zip(prepared.frames[1:], (int(value) for value in q_tail)))
        value_products = product(*(values for _key, values in b_domains))
        for b_tuple in value_products:
            b_values = {key: float(value) for (key, _values), value in zip(b_domains, b_tuple)}
            work.values["search_nodes"] += 1
            failure_reason = _accept_complete_label(
                prepared, q_values, b_values, integer_tolerance,
                min_resultant, work, candidates) or failure_reason
    frontier = _all_pairs_frontier(candidates, prepared, work)
    return _finish(frontier, work, failure_reason)


def select_joint_labels(
    readings: Sequence[PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    *,
    budget: SearchBudget,
    min_resultant: float = 0.90,
    integer_tolerance: float = 0.15,
) -> SelectorResult:
    """Exact bounded search. No default work budget and no production wiring."""
    zero_counters = SearchCounters(**{
        name: 0 for name in SearchCounters.__dataclass_fields__
    })
    if not math.isfinite(min_resultant) or not 0.0 <= min_resultant <= 1.0:
        return SelectorResult("invalid_input", "min_resultant must be finite in [0,1]",
                              None, (), zero_counters, None)
    if not math.isfinite(integer_tolerance) or integer_tolerance < 0.0:
        return SelectorResult("invalid_input", "integer_tolerance must be finite and non-negative",
                              None, (), zero_counters, None)
    return _search_bounded(readings, matches, budget, min_resultant=min_resultant,
                           integer_tolerance=integer_tolerance)


def _select_joint_labels_constraint_first(
    readings: Sequence[PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    *,
    budget: SearchBudget,
    min_resultant: float = 0.90,
    integer_tolerance: float = 0.15,
) -> SelectorResult:
    """Isolated G1E-B exact ordering/mask path; not used by production callers."""
    zero_counters = SearchCounters(**{
        name: 0 for name in SearchCounters.__dataclass_fields__
    })
    if not math.isfinite(min_resultant) or not 0.0 <= min_resultant <= 1.0:
        return SelectorResult("invalid_input", "min_resultant must be finite in [0,1]",
                              None, (), zero_counters, None)
    if not math.isfinite(integer_tolerance) or integer_tolerance < 0.0:
        return SelectorResult("invalid_input", "integer_tolerance must be finite and non-negative",
                              None, (), zero_counters, None)
    return _search_bounded(
        readings, matches, budget, min_resultant=min_resultant,
        integer_tolerance=integer_tolerance, variable_order="constraint_first")


def exhaustive_oracle(
    readings: Sequence[PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    *,
    min_resultant: float = 0.90,
    integer_tolerance: float = 0.15,
) -> SelectorResult:
    """Unbounded cartesian oracle for small synthetic cases only."""
    return _search_oracle(readings, matches, min_resultant=min_resultant,
                          integer_tolerance=integer_tolerance)
