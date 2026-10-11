"""Bound image-derived plane matching and component solving."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
from mc_photogrammetry.pipeline import image_derived_connections as connections_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.pipeline import candidate_support_provenance as candidate_support_mod
from mc_photogrammetry.pipeline import candidate_support_provenance_v2 as complete_support_mod
from mc_photogrammetry.reconstruction import lattice as support_observation
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph
from mc_photogrammetry.reconstruction import plane_placement as placement

failed_candidate_support_provenance = (
    candidate_support_mod.failed_candidate_support_provenance)

__all__ = ["HISTORICAL_IMAGE_DERIVED_GRAPH_SCHEMA", "HISTORICAL_IMAGE_DERIVED_GRAPH_PRODUCER",
           "NON_FINITE_SOLVER_OUTPUT", "HistoricalImageDerivedGraphResult",
           "SolverFailureSnapshot", "FailedActiveEdgeAliases",
           "BoundarySaturationAttestation",
           "FAILED_ACTIVE_EDGE_ALIASES_SCHEMA",
           "FAILED_ACTIVE_EDGE_ALIASES_BASIS",
           "FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN",
           "failed_edge_alias_record_digest",
           "validate_failed_active_edge_aliases",
           "BOUNDARY_SATURATION_SCHEMA",
           "BOUNDARY_SATURATION_DIGEST_DOMAIN",
           "BOUNDARY_SATURATION_INVALID",
           "BOUNDARY_SATURATION_UNAVAILABLE",
           "BOUNDARY_SATURATION_MAX_TRACE_ROWS",
           "boundary_saturation_record_digest",
           "validate_boundary_saturation_attestation",
           "boundary_saturation_attestation",
           "ACTIVE_EDGE_ALIASES_MAX_READINGS",
           "ACTIVE_EDGE_ALIASES_MAX_MATCHES",
           "ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES",
           "ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES",
           "ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES",
           "ACTIVE_EDGE_ALIASES_MAX_FIELD_BYTES",
           "ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES",
           "SolverPhaseCapture", "PHASE_CAPTURE_SCHEMA", "PHASE_CAPTURE_BASIS",
           "PHASE_CAPTURE_DIGEST_DOMAIN", "PHASE_CAPTURE_MAX_FACES",
           "PHASE_CAPTURE_MAX_FRAME_ID_BYTES", "PHASE_CAPTURE_MAX_RECORD_BYTES",
           "PHASE_CAPTURE_COMPATIBILITY_DECIMALS",
           "solver_phase_capture_record_digest", "validate_solver_phase_capture",
           "failed_phase_capture",
           "SolverInputBindings", "SOLVER_INPUT_BINDINGS_SCHEMA",
           "SOLVER_INPUT_BINDINGS_BASIS", "SOLVER_INPUT_BINDINGS_DIGEST_DOMAIN",
           "SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN",
           "SOLVER_INPUT_BINDINGS_MAX_READINGS",
           "SOLVER_INPUT_BINDINGS_MAX_MATCHES",
           "SOLVER_INPUT_BINDINGS_MAX_ALTERNATIVES",
           "SOLVER_INPUT_BINDINGS_MAX_FRAME_ID_BYTES",
           "SOLVER_INPUT_BINDINGS_MAX_RECORD_BYTES",
           "solver_input_bindings_record_digest",
           "validate_solver_input_bindings", "failed_solver_input_bindings",
           "SameAxisCandidateCapture", "SAME_AXIS_CANDIDATES_SCHEMA",
           "SAME_AXIS_CANDIDATES_BASIS", "SAME_AXIS_CANDIDATES_DIGEST_DOMAIN",
           "SAME_AXIS_CANDIDATES_AXIS", "SAME_AXIS_CANDIDATES_MAX_GROUPS",
           "SAME_AXIS_CANDIDATES_MAX_ROWS",
           "SAME_AXIS_CANDIDATES_MAX_RECORD_BYTES",
           "SAME_AXIS_CANDIDATES_MAX_READINGS",
           "SAME_AXIS_CANDIDATES_MAX_ALTERNATIVES",
           "SAME_AXIS_CANDIDATES_MAX_FRAME_ID_BYTES",
           "SAME_AXIS_CANDIDATES_AXES_V1", "SAME_AXIS_CANDIDATES_AXES_ALL",
           "SAME_AXIS_CANDIDATES_SCHEMA_V2", "SAME_AXIS_CANDIDATES_BASIS_V2",
           "SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V2",
           "SAME_AXIS_CANDIDATES_SCHEMA_V3", "failed_joint_same_axis_candidates",
           "same_axis_candidate_record_digest",
           "validate_same_axis_candidates", "failed_same_axis_candidates",
           "SolverSolveContext", "SOLVER_SOLVE_CONTEXT_SCHEMA",
           "SOLVER_SOLVE_CONTEXT_BASIS", "SOLVER_SOLVE_CONTEXT_DIGEST_DOMAIN",
           "SOLVER_SOLVE_CONTEXT_INVALID",
           "SOLVER_SOLVE_CONTEXT_BUILD_FAILURE",
           "SOLVER_SOLVE_CONTEXT_MAX_CONSTRAINTS",
           "SOLVER_SOLVE_CONTEXT_MAX_FRAME_ID_BYTES",
           "SOLVER_SOLVE_CONTEXT_MAX_RECORD_BYTES",
           "solver_solve_context_record_digest", "validate_solver_solve_context",
           "failed_solve_context", "refused_solve_context",
           "derive_image_derived_graph"]

HISTORICAL_IMAGE_DERIVED_GRAPH_SCHEMA = "mc-photogrammetry-image-derived-graph-v2"
# Producer 5 searches exact constraint-component rotations within the same budget.
# The serialized shape remains v2; earlier producers require fresh derivation.
HISTORICAL_IMAGE_DERIVED_GRAPH_PRODUCER = "m9-g-image-derived-graph-5"
GRAPH_TERMINAL = "geometry_unavailable"
NON_FINITE_SOLVER_OUTPUT = "non_finite_solver_output"
# Production-only unwrapped edge error in blocks; independent of circular phase.
_PRODUCTION_EDGE_ERROR_LIMIT_BLOCKS = 0.15

FAILED_ACTIVE_EDGE_ALIASES_SCHEMA = (
    "mc-photogrammetry-failed-active-edge-aliases-v1")
FAILED_ACTIVE_EDGE_ALIASES_BASIS = "derived_from_returned_best_assignment"
FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN = "failed_active_edge_aliases"
# DS-IMG-GEOM-PHASE-CAPTURE-R1: one shared failed-best-assignment digest domain
# and schema, so the active-edge alias record and the per-face phase capture
# bind the exact same assignment identity instead of comparing two unlike hash
# domains (which could never be equal).
FAILED_ASSIGNMENT_DIGEST_DOMAIN = "failed_best_assignment"
FAILED_ASSIGNMENT_DIGEST_SCHEMA = FAILED_ACTIVE_EDGE_ALIASES_SCHEMA

ACTIVE_EDGE_ALIASES_MAX_READINGS = 256
ACTIVE_EDGE_ALIASES_MAX_MATCHES = 256
ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES = 64
ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES = 64
ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES = 256
ACTIVE_EDGE_ALIASES_MAX_FIELD_BYTES = 256
ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES = 65_536

_ACTIVE_EDGE_ALIASES_MALFORMED = "invalid_active_edge_diagnostic"
_ACTIVE_EDGE_ALIASES_UNAVAILABLE = "best_assignment_unavailable"
_ACTIVE_EDGE_ALIASES_EXHAUSTED = "budget_exhausted"

# DS-R5: optional, failure-only boundary-saturation attestation. A shift is
# "saturated" when a row/column component pins the exact search-domain bound
# ``+/- max_shift_cells``; the record counts only, never modifies, the matcher's
# admitted set and the same-axis-kept set the solver received.
BOUNDARY_SATURATION_SCHEMA = "mc-photogrammetry-boundary-saturation-v1"
BOUNDARY_SATURATION_ALGORITHM = plane_graph.PLANE_CORRELATION_ALGORITHM
BOUNDARY_SATURATION_DIGEST_DOMAIN = "boundary_saturation"
BOUNDARY_SATURATION_INPUT_MATCH_IDENTITY_DOMAIN = (
    "boundary_saturation_input_match_identity")
BOUNDARY_SATURATION_BASIS = "derived_from_matcher_diagnostic_trace"
BOUNDARY_SATURATION_MAX_SHIFT_CELLS_CAP = 256
BOUNDARY_SATURATION_MAX_RECORD_BYTES = 65_536
# DS-R5B: the matcher appends exactly one ``PlanePairTrace`` row per candidate
# attempt (``plane_graph.match_plane_pair`` calls its local ``trace`` on every
# return path) and counts those same attempts in ``pre_admission_candidates``,
# so the diagnostic trace can never hold more rows than candidate attempts.
# ``plane_graph`` places no separate cap on that list, and the frozen v3b
# fixture records 7,476 pre-admission candidates with 1,495 admitted matches, so
# this explicit cap admits that real trace with headroom while refusing an
# unbounded one atomically before it is iterated.
BOUNDARY_SATURATION_MAX_TRACE_ROWS = 8_192

BOUNDARY_SATURATION_INVALID = "invalid_boundary_saturation_attestation"
BOUNDARY_SATURATION_UNAVAILABLE = "boundary_trace_unavailable"

# DS-IMG-GEOM-PHASE-CAPTURE: optional, failure-only per-face phase capture. It
# re-derives the failed best assignment's own integer-face constraints and the
# per-axis least-squares translation from the SAME local readings and matches the
# sink already holds, then recomposes each axis's resultant/residual from the
# captured faces so the raw snapshot can be independently reconciled. It is
# evaluation-only: the default production path never collects, allocates or
# serializes it, and the returned graph result and its digest stay unchanged.
PHASE_CAPTURE_SCHEMA = "mc-photogrammetry-solver-phase-capture-v1"
PHASE_CAPTURE_BASIS = "derived_from_returned_best_assignment"
PHASE_CAPTURE_DIGEST_DOMAIN = "solver_phase_capture"
PHASE_CAPTURE_MAX_FACES = 2 * ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES
PHASE_CAPTURE_MAX_FRAME_ID_BYTES = ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES
PHASE_CAPTURE_MAX_RECORD_BYTES = 65_536
# Exact float32-exact compatibility: the captured faces are re-expressed at this
# many decimal places before the independent recomposition is compared with the
# solver's own returned metrics, so a mismatch is a typed refusal and never a
# silent substitution.
PHASE_CAPTURE_COMPATIBILITY_DECIMALS = 6

_PHASE_CAPTURE_MALFORMED = "invalid_phase_capture_diagnostic"
_PHASE_CAPTURE_UNAVAILABLE = "best_assignment_unavailable"
_PHASE_CAPTURE_EXHAUSTED = "budget_exhausted"
_PHASE_CAPTURE_MISMATCH = "phase_recomposition_mismatch"

# DS-R7: optional, failure-only binding of *every* post-same-axis solver input.
# Where the active-edge alias record keeps only the best assignment's active
# edges and the phase capture keeps only their integer faces, this record keeps
# all input matches -- active or not -- together with each endpoint's retained
# alternative index and the exact measured alternative table that index resolves
# to. It is evaluation-only and output-neutral: the default production path
# never collects, allocates or serializes it, the returned graph result and its
# digest stay byte-for-byte unchanged, and no matcher/solver/gate decision is
# read or re-run.
SOLVER_INPUT_BINDINGS_SCHEMA = "mc-photogrammetry-solver-input-bindings-v1"
SOLVER_INPUT_BINDINGS_BASIS = "derived_from_post_same_axis_solver_inputs"
SOLVER_INPUT_BINDINGS_DIGEST_DOMAIN = "solver_input_bindings"
# One shared input-match-identity domain/schema with the active-edge alias
# record, so the two records can be independently shown to bind the exact same
# ordered 38-match input identity sequence rather than only the same count.
SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN = "input_match_identity"
SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA = (
    FAILED_ACTIVE_EDGE_ALIASES_SCHEMA)
SOLVER_INPUT_BINDINGS_MAX_READINGS = ACTIVE_EDGE_ALIASES_MAX_READINGS
SOLVER_INPUT_BINDINGS_MAX_MATCHES = ACTIVE_EDGE_ALIASES_MAX_MATCHES
SOLVER_INPUT_BINDINGS_MAX_ALTERNATIVES = (
    ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES)
SOLVER_INPUT_BINDINGS_MAX_FRAME_ID_BYTES = ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES
# Fixed byte budget. The structural caps above (readings/matches 256, endpoint
# alternatives 64, bounded frame IDs) bound collection work: they limit how many
# rows and fields are gathered, not the encoded size of the result. They do not
# prove the largest structurally valid record is below this 262,144-byte (256
# KiB) ceiling, so the ceiling is enforced independently of them: any record
# whose canonical encoding exceeds 262,144 bytes refuses atomically
# (`budget_exhausted`/`record_bytes`) even when it is fully within the
# structural caps, rather than growing the artifact. For the observed failure
# universe of 38 solver inputs over a small reading set with a handful of
# retained alternatives each, the canonical record was measured at 3,441 bytes
# on the synthetic 38-input / 12-reading / 19-active case, and the retained R4
# universe (16 canonical readings) is estimated well under 16 KiB; both are
# observations of those cases, not bounds on the largest structurally admitted
# record.
SOLVER_INPUT_BINDINGS_MAX_RECORD_BYTES = 262_144

_SOLVER_INPUT_BINDINGS_MALFORMED = "invalid_solver_input_bindings_diagnostic"
_SOLVER_INPUT_BINDINGS_UNAVAILABLE = "best_assignment_unavailable"
_SOLVER_INPUT_BINDINGS_EXHAUSTED = "budget_exhausted"

_SOLVER_INPUT_BINDINGS_COMPLETE_FIELDS = frozenset({
    "schema", "status", "basis", "failed_assignment_sha256",
    "input_match_identity_sha256", "input_match_count", "active_edge_count",
    "frames", "assignments", "readings", "edges"})
_SOLVER_INPUT_BINDINGS_REFUSAL_FIELDS = frozenset({
    "schema", "status", "reason", "field"})
# The one bounded set of field codes this producer can emit, so a replayed
# refusal cannot name an invented or unbounded field.
_SOLVER_INPUT_BINDINGS_FIELD_CODES = frozenset({
    "record", "readings", "matches", "solution", "frames", "frame_id",
    "quadrants", "best_assignment", "alternatives[0].quadrants",
    "input_match_identity", "match_endpoint", "endpoint_alternatives",
    "endpoint_alias", "endpoint_side", "readings_b", "reading_table",
    "match_index", "active_edge_count", "record_bytes"})

# DS-IMG-SAMEAXIS-PREFILTER-CAPTURE-R9: optional, failure-only capture of the
# complete pre-same-axis admitted candidate set for the same-axis groups the
# returned best assignment activates on world axis 1. Where the R7 record keeps
# only the post-filter solver inputs, this record keeps every admitted
# ``PlaneMatch`` the equality filter saw in those groups -- including the ones it
# dropped -- so a gate-passing ``b`` alternative can be shown to have existed as
# a real matcher candidate (or not) before ``_apply_same_axis_filter``. It is
# evaluation-only and output-neutral: the default production path never collects,
# allocates or serializes it, the returned graph result and its digest stay
# byte-for-byte unchanged, and no matcher/filter/solver choice is read or re-run.
SAME_AXIS_CANDIDATES_SCHEMA = "mc-photogrammetry-same-axis-candidate-capture-v1"
SAME_AXIS_CANDIDATES_BASIS = "derived_from_axis1_active_pre_same_axis_candidates"
SAME_AXIS_CANDIDATES_DIGEST_DOMAIN = "same_axis_candidate_capture"
SAME_AXIS_CANDIDATES_AXIS = 1
# DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: the v1 record is frozen exactly as above.
# V2 is an additive, evaluation-only all-world-axis sibling that keeps the same
# raw `SameAxisCandidateObservation` source and every original row field, changes
# only the declared axis set (a sorted world-axis list) and the group-to-survivor
# binding (an ordered list of retained post-filter match indices plus their exact
# input identities, because one same-axis group can legitimately retain more than
# one compatible match). The three global caps and the strict digest/refusal
# contract are shared verbatim, and the v1 validator path stays byte-for-byte.
SAME_AXIS_CANDIDATES_AXES_V1 = (SAME_AXIS_CANDIDATES_AXIS,)
SAME_AXIS_CANDIDATES_AXES_ALL = (0, 1, 2)
SAME_AXIS_CANDIDATES_SCHEMA_V2 = (
    "mc-photogrammetry-same-axis-candidate-capture-v2")
SAME_AXIS_CANDIDATES_BASIS_V2 = "derived_from_active_pre_same_axis_candidates"
SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V2 = "same_axis_candidate_capture_v2"
SAME_AXIS_CANDIDATES_SCHEMA_V3 = "mc-photogrammetry-same-axis-candidate-capture-v3"
SAME_AXIS_CANDIDATES_BASIS_V3 = "joint_selected_source_rows_and_raw_matcher_capture"
SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V3 = "same_axis_candidate_capture_v3"
SAME_AXIS_CANDIDATES_MAX_GROUPS = 16
SAME_AXIS_CANDIDATES_MAX_ROWS = 1_024
# Fixed byte budget, enforced independently of the structural caps below: any
# record whose canonical encoding exceeds 262,144 bytes refuses atomically
# (`budget_exhausted`/`record_bytes`) rather than growing the artifact. The
# structural caps bound collection work, not the encoded size.
SAME_AXIS_CANDIDATES_MAX_RECORD_BYTES = 262_144
SAME_AXIS_CANDIDATES_MAX_READINGS = ACTIVE_EDGE_ALIASES_MAX_READINGS
SAME_AXIS_CANDIDATES_MAX_ALTERNATIVES = (
    ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES)
SAME_AXIS_CANDIDATES_MAX_FRAME_ID_BYTES = ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES
SAME_AXIS_CANDIDATES_MAX_INPUT_MATCHES = ACTIVE_EDGE_ALIASES_MAX_MATCHES

_SAME_AXIS_CANDIDATES_MALFORMED = "invalid_same_axis_candidate_capture"
_SAME_AXIS_CANDIDATES_UNAVAILABLE = "best_assignment_unavailable"
_SAME_AXIS_CANDIDATES_EXHAUSTED = "budget_exhausted"

_SAME_AXIS_CANDIDATES_COMPLETE_FIELDS = frozenset({
    "schema", "status", "basis", "axis", "failed_assignment_sha256",
    "input_match_identity_sha256", "input_match_count", "group_count",
    "row_count", "frames", "assignments", "readings", "groups", "candidates"})
# V2 replaces the single world axis with the declared axis set; every other
# complete-record field is shared, so a v2 record differs in exactly two keys.
_SAME_AXIS_CANDIDATES_COMPLETE_FIELDS_V2 = (
    _SAME_AXIS_CANDIDATES_COMPLETE_FIELDS - {"axis"}) | {
        "axes", "input_identities"}
_SAME_AXIS_CANDIDATES_REFUSAL_FIELDS = frozenset({
    "schema", "status", "reason", "field"})
_SAME_AXIS_CANDIDATES_FIELD_CODES = frozenset({
    "record", "readings", "matches", "solution", "observation", "frames",
    "frame_id", "quadrants", "best_assignment", "alternatives[0].quadrants",
    "input_match_identity", "match_endpoint", "endpoint_alternatives",
    "endpoint_alias", "endpoint_side", "candidate_groups", "candidate_rows",
    "reading_table", "groups", "rows", "record_bytes", "axis", "axes",
    "input_identities"})

_PHASE_CAPTURE_COMPLETE_FIELDS = frozenset({
    "schema", "status", "basis", "source_sha256", "frames", "assignments",
    "compatibility_decimals", "translations", "axes", "faces",
    "mismatch_reason"})
_PHASE_CAPTURE_REFUSAL_FIELDS = frozenset({
    "schema", "status", "reason", "field"})
_PHASE_CAPTURE_FIELD_CODES = frozenset({
    "record", "readings", "matches", "solution", "frames", "frame_quadrants",
    "best_assignment", "quadrants", "alternatives[0].quadrants", "matches_b",
    "readings_b", "duplicate_b", "assignments", "translations", "axes",
    "axis_resultants", "axis_residuals", "axis_observations", "face_rows",
    "face_count", "recomposition", "frame_id", "record_bytes",
    "endpoint_alternatives", "endpoint_side", "axis_metric_coverage",
    "axis_observation_count", "axis_recomposition"})

_BOUNDARY_SATURATION_FIELD_CODES = frozenset({
    "trace", "pairs", "admitted", "selected", "matches", "max_shift_cells",
    "shift_components", "record_bytes", "input_match_identity"})

# DS-IMG-SOLVER-CONTEXT-CAPTURE-DIAGNOSTIC-20261004: one versioned, evaluation-only
# record of the exact effective arguments handed to ``plane_graph.solve_component``
# at the failing call. It is derived only inside the optional failure-observer
# block, so the default Route A/B path never allocates or serializes it.
SOLVER_SOLVE_CONTEXT_SCHEMA = "mc-photogrammetry-solver-solve-context-v1"
SOLVER_SOLVE_CONTEXT_BASIS = "derived_from_effective_solve_component_call_arguments"
SOLVER_SOLVE_CONTEXT_DIGEST_DOMAIN = "solver_solve_context"
SOLVER_SOLVE_CONTEXT_INVALID = "invalid_solver_solve_context"
# A bounded field code for a capture that could not be built at all. The failure
# observer must never degrade such a failure to an absent ``solve_context`` key,
# because absent is the historical no-context artifact shape and would be
# indistinguishable from a capture that was never attempted.
SOLVER_SOLVE_CONTEXT_BUILD_FAILURE = "context_build"
SOLVER_SOLVE_CONTEXT_MAX_CONSTRAINTS = 4096
SOLVER_SOLVE_CONTEXT_MAX_FRAME_ID_BYTES = ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES
SOLVER_SOLVE_CONTEXT_MAX_RECORD_BYTES = 262_144

_SOLVER_SOLVE_CONTEXT_NUMERIC_FIELDS = frozenset({
    "integer_tolerance", "min_resultant", "ambiguous_margin"})
_SOLVER_SOLVE_CONTEXT_ALL_FIELDS = frozenset({
    "schema", "status", "basis", "integer_tolerance", "min_resultant",
    "ambiguous_margin", "constraints", "frames", "invalid_fields"})
_SOLVER_SOLVE_CONTEXT_FIELD_CODES = frozenset({
    "integer_tolerance", "min_resultant", "ambiguous_margin", "constraints",
    "frames", "record_bytes", SOLVER_SOLVE_CONTEXT_BUILD_FAILURE})
_SOLVER_SOLVE_CONTEXT_CONSTRAINT_FIELDS = frozenset({"left", "right", "yaw"})

_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _number(value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not np.isfinite(number):
        raise ValueError("graph result contains a non-finite number")
    return number


def _finite_number(value: object) -> bool:
    """A plain finite ``int``/``float``; bools and non-numbers are rejected.

    ``math.isfinite`` raises ``OverflowError`` for an integer too large to
    convert to a float, so that conversion is guarded: an unrepresentable
    integer is refused rather than clamped or substituted.
    """
    if type(value) is int:
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return type(value) is float and math.isfinite(value)


def _psr_payload(value: object) -> dict[str, object]:
    number = float(value)  # type: ignore[arg-type]
    if np.isnan(number) or number == float("-inf"):
        raise ValueError("graph result contains an invalid psr")
    return {"value": None if number == float("inf") else number,
            "positive_infinity": number == float("inf")}


def _psr_from_payload(value: object) -> float:
    if (type(value) is not dict or set(value) != {"value", "positive_infinity"}
            or type(value["positive_infinity"]) is not bool):
        raise ValueError("graph psr tag is malformed")
    if value["positive_infinity"]:
        if value["value"] is not None:
            raise ValueError("positive-infinity psr must have a null value")
        return float("inf")
    if value["value"] is None:
        raise ValueError("finite psr must carry a value")
    return _number(value["value"])


def _match_payload(match: plane_graph.PlaneMatch) -> list:
    return [match.left, match.right, match.left_axis, match.right_axis,
            _number(match.left_b), _number(match.right_b),
            match.quadrant_of_right, _psr_payload(match.psr), _number(match.peak),
            _number(match.second_peak), match.overlap_cells,
            None if match.integer_residual is None else _number(match.integer_residual),
            _number(match.score), match.left_plane, match.right_plane,
            None if match.cell_shift_rc is None else list(match.cell_shift_rc)]


def _solution_payload(solution: plane_graph.ComponentSolution | None) -> object:
    if solution is None:
        return None
    return {
        "frames": list(solution.frames),
        "quadrants": sorted(solution.quadrants.items()),
        "translations": [[key, list(value)] for key, value in
                         sorted(solution.translations.items())],
        "axis_residuals": sorted(solution.axis_residuals.items()),
        "axis_resultants": sorted(solution.axis_resultants.items()),
        "scale_known": solution.scale_known,
        "blocks_per_period": [[frame, index, value]
                              for (frame, index), value in
                              sorted(solution.blocks_per_period.items())],
        "reason": solution.reason,
        "alternatives": [[list(item.quadrants), item.cost,
                          list(item.resultants), list(item.axis_observations)]
                         for item in solution.alternatives],
    }


@dataclass(frozen=True, slots=True)
class SolverFailureSnapshot:
    """Bounded immutable copy of one abstaining solver evaluation.

    Only the public result's refusal reason plus the per-axis metrics the
    solver already computed and its best alternative's observation counts are
    retained. Translations, blocks-per-period, scale and the whole alternatives
    tuple are deliberately dropped, and no second solve or ranking happens.

    ``invalid_field`` is the bounded carrier for diagnostic data that could not
    be copied at all (missing or malformed observations, metric maps or frame
    controls). It names only the offending field, never the raw value, so the
    default evaluation path can still return its original unavailable result.

    ``active_edge_aliases`` is the optional v4 evaluation-only alias view of the
    failed best assignment. It is requested only by the attribution path and is
    ``None`` for the default production call, so nothing new is allocated or
    serialized unless the caller explicitly asks for the alias diagnostic.

    ``boundary_saturation`` is the optional DS-R5 evaluation-only count of how
    many admitted and same-axis-kept selected shifts pin the exact search bound.
    It is derived by the same failure sink from the matcher's own diagnostic
    trace and the post-filter matches, so it is also ``None`` -- and nothing is
    counted -- unless the caller attached the evaluation sink.

    ``phase_capture`` is the optional DS-IMG-GEOM-PHASE-CAPTURE evaluation-only
    per-face phase view of the failed best assignment. It is likewise ``None``
    for the default production call and is derived only by the attached failure
    sink from the same local readings and matches.

    ``solver_input_bindings`` is the optional DS-R7 evaluation-only binding of
    the complete post-same-axis solver input set (active and inactive) to each
    endpoint's retained alternative index and measured alternative table. It is
    likewise ``None`` for the default production call and is derived only by the
    attached failure sink from the same local readings and matches.

    ``solve_context`` is the optional DS-IMG-SOLVER-CONTEXT-CAPTURE diagnostic
    record of the exact effective arguments the failing call passed to
    ``plane_graph.solve_component`` -- the three numeric thresholds plus the
    exact immutable ``quadrant_constraints`` tuple. It is likewise ``None`` for
    the default production call and is built only inside the failure-observer
    block from the same values bound into the call, so the default path never
    allocates or serializes it and the returned result and digests are unchanged.
    """

    reason: str
    frames: tuple[str, ...]
    quadrants: tuple[object, ...]
    axis_resultants: tuple[tuple[object, object], ...]
    axis_residuals: tuple[tuple[object, object], ...]
    best_alternative_quadrants: tuple[object, ...] | None
    best_alternative_axis_observations: tuple[object, ...] | None
    invalid_field: str | None = None
    active_edge_aliases: "FailedActiveEdgeAliases | None" = None
    boundary_saturation: "BoundarySaturationAttestation | None" = None
    phase_capture: "SolverPhaseCapture | None" = None
    solver_input_bindings: "SolverInputBindings | None" = None
    solve_context: "SolverSolveContext | None" = None


def _snapshot_tuple(
    value: object, field: str, *, optional: bool = False,
) -> tuple[tuple[object, ...], str | None]:
    """Copy a diagnostic collection; ``(items, invalid_field)`` never raises.

    A missing optional collection is ``None``-equivalent (represented by an
    empty tuple plus ``None`` from the caller); anything else that is not a
    usable collection is reported as the bounded ``field`` code.
    """
    if value is None:
        return (), (None if optional else field)
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        return (), field
    try:
        return tuple(value), None
    except _ERRORS:
        return (), field


def _snapshot_items(
    value: object, field: str,
) -> tuple[tuple[tuple[object, object], ...], str | None]:
    """Copy a diagnostic mapping's items; malformed maps yield a field code."""
    items = getattr(value, "items", None)
    if not callable(items):
        return (), field
    try:
        return tuple(items()), None
    except _ERRORS:
        return (), field


def _malformed_snapshot(reason: object, field: str) -> SolverFailureSnapshot:
    return SolverFailureSnapshot(
        reason, (), (), (), (), None, None, field)  # type: ignore[arg-type]


def _solver_failure_snapshot(
        solution: plane_graph.ComponentSolution,
        *,
        active_edge_aliases: "FailedActiveEdgeAliases | None" = None,
        boundary_saturation: "BoundarySaturationAttestation | None" = None,
        phase_capture: "SolverPhaseCapture | None" = None,
        solver_input_bindings: "SolverInputBindings | None" = None,
        solve_context: "SolverSolveContext | None" = None,
) -> SolverFailureSnapshot:
    """Copy one abstaining solution into the private, bounded sink payload.

    Every raw conversion is guarded so that missing or malformed diagnostic
    data becomes a bounded ``invalid_field`` carrier instead of an exception;
    the caller then still returns the original unavailable graph result
    unchanged. Missing best-alternative counts stay unavailable rather than
    becoming a refusal or an invented zero.
    """
    frames, field = _snapshot_tuple(getattr(solution, "frames", None), "frames")
    if field is not None:
        return _malformed_snapshot(solution.reason, field)
    getter = getattr(getattr(solution, "quadrants", None), "get", None)
    if not callable(getter):
        return _malformed_snapshot(solution.reason, "quadrants")
    try:
        quadrants = tuple(getter(frame) for frame in frames)
    except _ERRORS:
        return _malformed_snapshot(solution.reason, "quadrants")
    resultants, field = _snapshot_items(
        getattr(solution, "axis_resultants", None), "axis_resultants")
    if field is not None:
        return _malformed_snapshot(solution.reason, field)
    residuals, field = _snapshot_items(
        getattr(solution, "axis_residuals", None), "axis_residuals")
    if field is not None:
        return _malformed_snapshot(solution.reason, field)
    best = None
    alternatives = getattr(solution, "alternatives", None)
    if alternatives:
        try:
            best = alternatives[0]
        except _ERRORS:
            best = None
    best_quadrants: tuple[object, ...] | None = None
    best_observations: tuple[object, ...] | None = None
    if best is not None:
        if getattr(best, "quadrants", None) is not None:
            best_quadrants, field = _snapshot_tuple(
                best.quadrants, "alternatives[0].quadrants")
            if field is not None:
                return _malformed_snapshot(solution.reason, field)
        if getattr(best, "axis_observations", None) is not None:
            best_observations, field = _snapshot_tuple(
                best.axis_observations, "alternatives[0].axis_observations")
            if field is not None:
                return _malformed_snapshot(solution.reason, field)
    return SolverFailureSnapshot(
        solution.reason, frames, quadrants, resultants, residuals,
        best_quadrants, best_observations, active_edge_aliases=active_edge_aliases,
        boundary_saturation=boundary_saturation, phase_capture=phase_capture,
        solver_input_bindings=solver_input_bindings, solve_context=solve_context)


def _solve_context_value(value: object, field: str) -> object:
    """Keep one exact finite threshold, or a bounded unavailable mapping.

    The exact ``int``/``float`` the call passed is retained verbatim; a NaN,
    infinity, bool, non-number or unrepresentable integer becomes a fixed-shape
    mapping naming only its own field and concrete Python type, never a clamped
    or substituted default.
    """
    if _finite_number(value):
        return value
    return {"unavailable": SOLVER_SOLVE_CONTEXT_INVALID, "field": field,
            "type": type(value).__name__}


def _solve_context_constraints(
    value: object,
) -> tuple[list[dict] | None, str | None]:
    """Serialize the exact constraint tuple, or report one bounded field code.

    ``None`` means the value was not an immutable tuple. An empty tuple is a
    legitimate capture and serializes to an empty list, never to ``None``. A
    malformed, oversized, foreign or contradictory tuple returns ``[]`` plus the
    bounded ``constraints`` code so the record stays structural rather than
    crashing the failure observer.
    """
    if type(value) is not tuple:
        return None, "constraints"
    if len(value) > SOLVER_SOLVE_CONTEXT_MAX_CONSTRAINTS:
        return [], "constraints"
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if (type(item) is not tuple or len(item) != 3
                or type(item[0]) is not str or type(item[1]) is not str
                or not item[0] or not item[1] or item[0] == item[1]
                or type(item[2]) is not int or item[2] not in (0, 1, 2, 3)):
            return [], "constraints"
        key = (item[0], item[1])
        if key in seen:
            return [], "constraints"
        seen.add(key)
        rows.append({"left": item[0], "right": item[1], "yaw": item[2]})
    return rows, None


def _solve_context_row(row: object) -> dict:
    """One serialized constraint row from either a tuple pair or a mapping."""
    if type(row) is dict:
        return {"left": row.get("left"), "right": row.get("right"),
                "yaw": row.get("yaw")}
    if type(row) is tuple and len(row) == 3:
        return {"left": row[0], "right": row[1], "yaw": row[2]}
    return {"left": None, "right": None, "yaw": None}


def _solve_context_payload(value: "SolverSolveContext") -> dict:
    """Canonical serialized payload of one capture, excluding its own digest.

    ``invalid_fields`` is omitted exactly when the capture recorded every field,
    so both the in-memory digest and the serialized record bind the identical
    strict map and a re-hashed record cannot silently disagree with ``to_record``.
    """
    payload = {
        "schema": value.schema,
        "status": "complete" if value.recorded else "refused",
        "basis": SOLVER_SOLVE_CONTEXT_BASIS,
        "integer_tolerance": _solve_context_value(
            value.integer_tolerance, "integer_tolerance"),
        "min_resultant": _solve_context_value(
            value.min_resultant, "min_resultant"),
        "ambiguous_margin": _solve_context_value(
            value.ambiguous_margin, "ambiguous_margin"),
        "constraints": [_solve_context_row(row) for row in value.constraints],
        "frames": list(value.frames),
    }
    if value.invalid_fields:
        payload["invalid_fields"] = list(value.invalid_fields)
    return payload


def solver_solve_context_record_digest(value: "SolverSolveContext") -> str:
    """Recompute one capture's strict digest from its canonical payload."""
    return _sha({"domain": SOLVER_SOLVE_CONTEXT_DIGEST_DOMAIN,
                 "schema": value.schema, "payload": _solve_context_payload(value)})


@dataclass(frozen=True, slots=True)
class SolverSolveContext:
    """Evaluation-only capture of the exact ``solve_component`` call arguments.

    A capture retains, verbatim, the three effective numeric thresholds
    (``integer_tolerance``, ``min_resultant``, ``ambiguous_margin``) and the
    exact immutable ``quadrant_constraints`` tuple the failing call passed --
    including an empty tuple, which is a real capture and never a missing value.
    ``frames`` is the solver's own reading-universe order, so a replay can tell
    whether a constraint names a foreign frame. ``invalid_fields`` lists, in
    fixed order, any field whose value could not be captured exactly; it is
    omitted from the serialized payload when every field was captured. The
    separate ``SOLVER_SOLVE_CONTEXT_BUILD_FAILURE`` (``context_build``) code is
    declared by :func:`refused_solve_context` when the capture could not be built
    at all, so a failed build stays an explicit refused record rather than
    becoming an absent key that would mimic a historical no-context artifact.

    ``record_sha256`` binds the strict UTF-8 canonical JSON payload with explicit
    domain/schema separation, excluding only the field itself. The record is
    never allocated unless the optional failure observer is attached, and it
    never changes the returned graph result, its digest or the solver output.
    """

    schema: str
    recorded: bool
    integer_tolerance: object
    min_resultant: object
    ambiguous_margin: object
    constraints: tuple[tuple[str, str, int], ...]
    frames: tuple[str, ...]
    invalid_fields: tuple[str, ...] = ()
    record_sha256: str = ""

    def __post_init__(self) -> None:
        digest = solver_solve_context_record_digest(self)
        if self.record_sha256 not in ("", digest):
            raise ValueError("solver solve context digest mismatch")
        object.__setattr__(self, "record_sha256", digest)

    def to_record(self) -> dict:
        payload = _solve_context_payload(self)
        payload["record_sha256"] = self.record_sha256
        return payload


def _solve_context_universe(frames: object) -> tuple | None:
    """Copy one bounded frame universe, or ``None`` when it is not usable.

    ``None`` means the value cannot be read as a bounded sequence -- a string,
    a non-sequence, an over-cap sequence, or one whose ``len``/iteration raises;
    the caller maps that to the bounded ``frames`` code. ``tuple`` is returned
    without inspecting item types, exactly as the capture recorded before, so a
    malformed identifier stays visible rather than being silently dropped.
    """
    if (isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence)
            or len(frames) > ACTIVE_EDGE_ALIASES_MAX_READINGS):
        return None
    return tuple(frames)  # type: ignore[arg-type]


def failed_solve_context(
    integer_tolerance: object,
    min_resultant: object,
    ambiguous_margin: object,
    quadrant_constraints: object,
    frames: object,
) -> SolverSolveContext:
    """Build the exact failure-only capture of one solver call's arguments.

    The builder is total: a malformed value becomes a bounded unavailable marker
    or one bounded ``invalid_fields`` code rather than an exception, so deriving
    the default graph result can never be disturbed by the diagnostic. It is
    called only inside the optional failure-observer block.
    """
    invalid: list[str] = []
    rows, constraint_field = _solve_context_constraints(quadrant_constraints)
    if constraint_field is not None:
        invalid.append(constraint_field)
        rows = []
    frames_tuple: tuple[str, ...] = ()
    try:
        captured_universe = _solve_context_universe(frames)
    except _ERRORS:
        captured_universe = None
    if captured_universe is None:
        invalid.append("frames")
    else:
        frames_tuple = captured_universe
    values = {
        "integer_tolerance": integer_tolerance,
        "min_resultant": min_resultant,
        "ambiguous_margin": ambiguous_margin,
    }
    for name in sorted(values):
        if not _finite_number(values[name]):
            invalid.append(name)
    return SolverSolveContext(
        schema=SOLVER_SOLVE_CONTEXT_SCHEMA, recorded=not invalid,
        integer_tolerance=integer_tolerance, min_resultant=min_resultant,
        ambiguous_margin=ambiguous_margin,
        constraints=() if rows is None else tuple(
            (row["left"], row["right"], row["yaw"]) for row in rows),
        frames=frames_tuple, invalid_fields=tuple(sorted(set(invalid))))


def refused_solve_context(
    integer_tolerance: object,
    min_resultant: object,
    ambiguous_margin: object,
    frames: object,
) -> SolverSolveContext:
    """Explicit bounded refusal when the exact capture cannot be built at all.

    ``failed_solve_context`` is total for its documented inputs, but the failure
    observer must never let an unexpected build failure degrade to an absent
    ``solve_context`` key: absent is exactly the historical no-context artifact
    shape, so a default or pre-change record would be indistinguishable from a
    capture whose construction failed. This fallback always returns a refused,
    digest-bound record that declares the bounded ``context_build`` code and
    still carries the same call thresholds and solver frame universe, so the
    failed build stays an explicit diagnostic bound to the same failing snapshot.

    It is total by construction: the numeric thresholds go through the same
    serializer that already handles any value, and the frame copy is guarded, so
    this fallback cannot itself raise a second time and leave the observer with
    nothing to serialize.
    """
    try:
        universe = _solve_context_universe(frames)
    except _ERRORS:
        universe = None
    return SolverSolveContext(
        schema=SOLVER_SOLVE_CONTEXT_SCHEMA, recorded=False,
        integer_tolerance=integer_tolerance, min_resultant=min_resultant,
        ambiguous_margin=ambiguous_margin, constraints=(),
        frames=() if universe is None else universe,
        invalid_fields=(SOLVER_SOLVE_CONTEXT_BUILD_FAILURE,))


def _solve_context_unavailable(value: object) -> bool:
    """One exact bounded unavailable marker, or ``False`` for anything else."""
    if type(value) is not dict or set(value) != {"unavailable", "field", "type"}:
        return False
    return (value["unavailable"] == SOLVER_SOLVE_CONTEXT_INVALID
            and isinstance(value["field"], str)
            and value["field"] in _SOLVER_SOLVE_CONTEXT_NUMERIC_FIELDS
            and isinstance(value["type"], str))


def _validate_solve_context_number(value: object) -> None:
    if _finite_number(value):
        return
    if not _solve_context_unavailable(value):
        raise ValueError("solver solve context numeric value is malformed")


def _validate_solve_context_constraint_row(row: object) -> tuple[str, str, int]:
    if type(row) is not dict or set(row) != _SOLVER_SOLVE_CONTEXT_CONSTRAINT_FIELDS:
        raise ValueError("solver solve context constraint row is malformed")
    left, right, yaw = row["left"], row["right"], row["yaw"]
    if (type(left) is not str or type(right) is not str or not left or not right
            or left == right or type(yaw) is not int or yaw not in (0, 1, 2, 3)):
        raise ValueError("solver solve context constraint row is malformed")
    return left, right, yaw


def _validate_solve_context_constraints(value: object) -> tuple[tuple[str, str, int], ...]:
    if type(value) is not list or len(value) > SOLVER_SOLVE_CONTEXT_MAX_CONSTRAINTS:
        raise ValueError("solver solve context constraints must be a list")
    rows = tuple(_validate_solve_context_constraint_row(item) for item in value)
    keys = [(left, right) for left, right, _yaw in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("solver solve context repeats a constraint endpoint pair")
    return rows


def _validate_solve_context_frames(value: object) -> tuple[str, ...]:
    """Validate one ordered frame-universe list.

    An empty list is legitimate -- a solver call with no reading universe is a
    real capture -- so it serializes and replays as an empty list rather than
    being refused as missing.
    """
    if type(value) is not list:
        raise ValueError("solver solve context frames must be a list")
    if any(type(frame) is not str or not frame for frame in value):
        raise ValueError("solver solve context frames must be identifiers")
    if len(set(value)) != len(value):
        raise ValueError("solver solve context frames must be unique")
    if any(len(frame.encode("utf-8")) > SOLVER_SOLVE_CONTEXT_MAX_FRAME_ID_BYTES
           for frame in value):
        raise ValueError("solver solve context frame identifier is over the cap")
    return tuple(value)


def _solve_context_expected_frames(value: object) -> tuple[str, ...]:
    """Normalize one expected frame universe without requiring a non-empty list."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("expected solver solve context frames must be a sequence")
    return _validate_solve_context_frames(list(value))


def _validate_solve_context_invalid_fields(value: object) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError("solver solve context invalid fields must be a list")
    if len(set(value)) != len(value):
        raise ValueError("solver solve context invalid fields repeat")
    if any(type(field) is not str
           or field not in _SOLVER_SOLVE_CONTEXT_FIELD_CODES for field in value):
        raise ValueError("solver solve context invalid field is not a known code")
    return tuple(value)


def _validate_solve_context_payload(payload: object) -> None:
    """Re-derive every structural and cross-field relation from a payload."""
    if type(payload) is not dict or set(payload) - _SOLVER_SOLVE_CONTEXT_ALL_FIELDS:
        raise ValueError("solver solve context has unexpected fields")
    if payload.get("schema") != SOLVER_SOLVE_CONTEXT_SCHEMA:
        raise ValueError("solver solve context schema is not v1")
    if payload.get("basis") != SOLVER_SOLVE_CONTEXT_BASIS:
        raise ValueError("solver solve context basis is unknown")
    status = payload.get("status")
    if status not in ("complete", "refused"):
        raise ValueError("solver solve context status is unknown")
    for name in _SOLVER_SOLVE_CONTEXT_NUMERIC_FIELDS:
        if name not in payload:
            raise ValueError("solver solve context is missing a numeric field")
        _validate_solve_context_number(payload[name])
    rows = _validate_solve_context_constraints(payload.get("constraints"))
    frames = _validate_solve_context_frames(payload.get("frames"))
    invalid = _validate_solve_context_invalid_fields(
        payload.get("invalid_fields", []))
    # Every constraint endpoint must name a frame inside the solver's own
    # reading universe; a foreign frame is not a repair case.
    if any(left not in frames or right not in frames for left, right, _yaw in rows):
        raise ValueError("solver solve context constraint names a foreign frame")
    # The bounded unavailable markers and the declared status must agree: each
    # numeric field's shape implies its own invalid code, and any structural
    # refusal implies exactly the "constraints" or "frames" code.
    declared = set(invalid)
    implied: set[str] = set()
    for name in _SOLVER_SOLVE_CONTEXT_NUMERIC_FIELDS:
        if _solve_context_unavailable(payload[name]):
            implied.add(name)
    if status == "complete":
        if declared or implied:
            raise ValueError("complete solve context declares invalid fields")
    elif not declared and not implied:
        raise ValueError("refused solve context declares no invalid field")
    if not (implied | declared) <= set(_SOLVER_SOLVE_CONTEXT_FIELD_CODES):
        raise ValueError("solver solve context invalid field is not a known code")


def validate_solver_solve_context(
    record: object, expected_frames: object = None,
) -> None:
    """Fail-closed replay for one serialized solve-context capture.

    A matching digest is necessary but not sufficient. The schema, basis,
    status, exact numeric value shape, ordered constraint rows, frame universe
    and declared invalid-field set are all re-derived, and when ``expected_frames``
    is supplied every context frame must agree with it -- so a resealed context
    cannot borrow another stage's frame universe. Reordered or substituted
    constraints change the digest, and a constraint naming a frame outside the
    context's own universe is refused outright.
    """
    if type(record) is not dict:
        raise ValueError("solver solve context must be a mapping")
    body = dict(record)
    claimed = body.pop("record_sha256", None)
    if not _hex(claimed):
        raise ValueError("solver solve context digest must be lowercase hex")
    _validate_solve_context_payload(body)
    if claimed != _sha({"domain": SOLVER_SOLVE_CONTEXT_DIGEST_DOMAIN,
                        "schema": body.get("schema"), "payload": body}):
        raise ValueError("solver solve context digest mismatch")
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > SOLVER_SOLVE_CONTEXT_MAX_RECORD_BYTES:
        raise ValueError("solver solve context exceeds the fixed byte budget")
    if expected_frames is not None:
        expected = _solve_context_expected_frames(expected_frames)
        if tuple(body.get("frames", ())) != expected:
            raise ValueError("solver solve context frames are foreign to the snapshot")


def _endpoint_canonical_b(
    reading: lattice_planes.PlaneReading, requested: object,
    max_alternatives: int,
) -> tuple[float | None, int | None]:
    """Resolve one raw ``b`` through the existing unique-alternative contract.

    Ambiguous and missing aliases both resolve to ``(None, None)``, never the
    first of several candidates. The count of alternatives actually inspected is
    returned so the caller can bound the alias inspection itself: a reading
    carrying more retained alternatives than the cap is refused *before* the
    resolver is invoked at all, so an over-cap endpoint never reaches alias
    matching.
    """
    if (isinstance(requested, (bool, np.bool_))
            or not isinstance(requested, (int, float, np.integer, np.floating))):
        return None, None
    alternatives = getattr(reading, "alternatives", None)
    if (isinstance(alternatives, (str, bytes))
            or not isinstance(alternatives, Sequence)):
        return None, None
    inspected = len(alternatives)
    if inspected > max_alternatives:
        return None, inspected
    try:
        matches = reading.matching_alternatives(requested)
    except _ERRORS:
        return None, inspected
    if len(matches) != 1:
        return None, inspected
    try:
        canonical = float(matches[0].blocks_per_period)
    except _ERRORS:
        return None, inspected
    if not np.isfinite(canonical) or canonical <= 0.0:
        return None, inspected
    return canonical, inspected


def _retained_alternative_count(reading: object) -> int | None:
    """Number of retained alternatives on one reading, or ``None`` if malformed.

    Only the retained collection's length is read; no alternative is inspected,
    resolved or copied, so this preflight stays linear in the reading count and
    never triggers alias work. A malformed collection stays ``None`` here and is
    refused later by the canonical index or the alias resolver.
    """
    alternatives = getattr(reading, "alternatives", None)
    if (isinstance(alternatives, (str, bytes))
            or not isinstance(alternatives, Sequence)):
        return None
    try:
        return len(alternatives)
    except _ERRORS:
        return None


def _raw_endpoint_b(value: object) -> float | None:
    """Return one source ``b`` label exactly, as a finite JSON scalar.

    The raw-b column has to stay the caller's own label so each edge row replays
    against the supplied match, independently of the resolver's canonical
    alternative. A missing, non-numeric or non-finite label returns ``None`` and
    the caller refuses atomically; it never reaches a complete record, and no
    rounding or re-derivation of the source value happens here.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except _ERRORS:
        return None
    return number if np.isfinite(number) else None


def _alias_domain_digest(domain: str, schema: str, payload: object) -> str:
    """Hash one diagnostic projection with explicit domain/schema separation."""
    return _sha({"domain": domain, "schema": schema, "payload": payload})


def _failed_assignment_digest(
    frames: Sequence[str], frame_quadrants: dict[str, int],
) -> str:
    """Hash one failed best assignment identically for every observer seam.

    Both the active-edge alias record and the per-face phase capture bind the
    same ``(frame, quadrant)`` identity through this single helper, so the
    runner's equality check compares like domains by construction.
    """
    return _alias_domain_digest(
        FAILED_ASSIGNMENT_DIGEST_DOMAIN, FAILED_ASSIGNMENT_DIGEST_SCHEMA,
        [[frame, frame_quadrants[frame]] for frame in frames])


def failed_edge_alias_record_digest(value: "FailedActiveEdgeAliases") -> str:
    """Recompute one record's strict digest from its canonical payload."""
    return _sha({
        "domain": FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN,
        "schema": value.schema,
        "payload": _failed_edge_aliases_payload(value)})


def _failed_edge_aliases_payload(value: "FailedActiveEdgeAliases") -> dict:
    """Canonical payload of one record, without its own digest field."""
    if not value.recorded:
        return {"schema": value.schema, "status": "refused",
                "reason": value.refusal_reason, "field": value.field}
    return {
        "schema": value.schema,
        "status": "complete",
        "basis": value.basis,
        "failed_assignment_sha256": value.failed_assignment_sha256,
        "input_match_identity_sha256": value.input_match_identity_sha256,
        "input_match_count": value.input_match_count,
        "active_edge_count": value.active_edge_count,
        "axis_edge_counts": [value.axis_edge_counts[axis] for axis in range(3)],
        "edges": [list(edge) for edge in value.edges],
        "conflicting_endpoints": [list(endpoint)
                                  for endpoint in value.conflicting_endpoints],
    }


_FAILED_ALIAS_COMPLETE_FIELDS = frozenset({
    "schema", "status", "basis", "failed_assignment_sha256",
    "input_match_identity_sha256", "input_match_count", "active_edge_count",
    "axis_edge_counts", "edges", "conflicting_endpoints"})
_FAILED_ALIAS_REFUSAL_FIELDS = frozenset({
    "schema", "status", "reason", "field"})
_ACTIVE_EDGE_ALIASES_REFUSAL_REASONS = frozenset({
    _ACTIVE_EDGE_ALIASES_MALFORMED, _ACTIVE_EDGE_ALIASES_UNAVAILABLE,
    _ACTIVE_EDGE_ALIASES_EXHAUSTED})
# The one bounded set of field codes this producer can emit. A replayed refusal
# must name one of them (or nothing), so an invented or unbounded field cannot
# be sealed into a record the producer would never write.
_ACTIVE_EDGE_ALIASES_FIELD_CODES = frozenset({
    "record", "matches", "input_match_count", "readings", "reading_count",
    "frames", "frame_id", "input_match_identity", "best_assignment",
    "quadrants", "alternatives[0].quadrants", "best_assignment_identity",
    "axis_observations", "edge_endpoint", "active_edge_count",
    "endpoint_alternatives", "endpoint_alias", "observation_conservation",
    "record_bytes"})


def _is_finite_number(value: object) -> bool:
    """A finite, non-bool JSON numeric scalar (``int``/``float``).

    A serialized edge row's ``b`` columns must already be JSON numbers, so a
    numeric string is refused instead of being coerced: replay binds the value
    the record actually carries. An integer too large to represent as a float is
    refused rather than defaulted.
    """
    if type(value) is int:
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return type(value) is float and math.isfinite(value)


def _boundary_shift(shift: object) -> tuple[int, int] | None:
    """Read one exact integer ``(row, column)`` cell shift, or ``None``.

    A missing shift, a non-tuple, a wrong arity or a bool/float component is
    refused by returning ``None`` rather than being coerced, so a malformed
    diagnostic can never be counted or silently dropped.
    """
    if type(shift) is not tuple or len(shift) != 2:
        return None
    if any(type(component) is not int for component in shift):
        return None
    return shift  # type: ignore[return-value]


def _boundary_components(
    shift: object, max_shift_cells: int,
) -> tuple[int, int, int] | None:
    """One 3-bit mask over ``(row, column, reserved)``.

    Bit 0 is the cell-shift row component and bit 1 the column component; bit 2
    is always clear, so every count triple is a fixed length-3 axis tuple. Only
    the *magnitude* is compared, so both ``+max_shift_cells`` and
    ``-max_shift_cells`` are boundary readings: the shift domain is symmetric.
    """
    value = _boundary_shift(shift)
    if value is None:
        return None
    row, column = value
    return (1 if abs(row) == max_shift_cells else 0,
            1 if abs(column) == max_shift_cells else 0, 0)


def _validate_edge_row(
    row: object, previous_index: int, input_match_count: int,
) -> tuple[int, tuple[str, int], tuple[str, int]]:
    """Validate one immutable edge row and return its match/endpoint identity.

    Arity, scalar types, canonical index bounds and ordering, finite source
    labels, positive canonical aliases and canonical axes/quadrants are all
    re-derived here rather than trusted from the serialized digest.
    """
    if type(row) not in (tuple, list) or len(row) != 13:
        raise ValueError("alias edge row must be a 13-element sequence")
    (match_index, left_frame, left_plane, left_axis, left_b, left_canonical_b,
     right_frame, right_plane, right_axis, right_b, right_canonical_b,
     quadrant, world_axis) = row
    if (type(match_index) is not int or match_index < 0
            or match_index >= input_match_count or match_index <= previous_index):
        raise ValueError("alias edge match index is out of bounds or order")
    for frame in (left_frame, right_frame):
        if (type(frame) is not str or not frame
                or _utf8_length(frame) is None
                or _utf8_length(frame) > ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES):
            raise ValueError("alias edge frame identifier is malformed or over cap")
    for plane in (left_plane, right_plane):
        if type(plane) is not int or plane < 0:
            raise ValueError("alias edge plane index must be non-negative")
    for axis in (left_axis, right_axis):
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("alias edge local axis must be canonical")
    for number in (left_b, right_b, left_canonical_b, right_canonical_b):
        if not _is_finite_number(number):
            raise ValueError("alias edge b value must be finite")
    if left_canonical_b <= 0.0 or right_canonical_b <= 0.0:
        raise ValueError("alias edge canonical b must be positive")
    if type(quadrant) is not int or quadrant not in (0, 1, 2, 3):
        raise ValueError("alias edge relative quadrant must be canonical")
    if type(world_axis) is not int or world_axis not in (0, 1, 2):
        raise ValueError("alias edge world axis must be canonical")
    return match_index, (left_frame, left_plane), (right_frame, right_plane)


def _validate_complete_alias_payload(payload: dict) -> None:
    """Re-derive every count and conflict from the edge rows themselves."""
    if set(payload) != _FAILED_ALIAS_COMPLETE_FIELDS:
        raise ValueError("complete alias record has unexpected fields")
    if payload["basis"] != FAILED_ACTIVE_EDGE_ALIASES_BASIS:
        raise ValueError("complete alias record has an unknown basis")
    if (not _hex(payload["failed_assignment_sha256"])
            or not _hex(payload["input_match_identity_sha256"])):
        raise ValueError("alias record source digests must be lowercase hex")
    input_match_count = payload["input_match_count"]
    if (type(input_match_count) is not int or input_match_count < 0
            or input_match_count > ACTIVE_EDGE_ALIASES_MAX_MATCHES):
        raise ValueError("alias record input match count is out of bounds")
    active_edge_count = payload["active_edge_count"]
    if (type(active_edge_count) is not int or active_edge_count < 0
            or active_edge_count > ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES):
        raise ValueError("alias record active edge count is out of bounds")
    axis_edge_counts = payload["axis_edge_counts"]
    if (type(axis_edge_counts) is not list or len(axis_edge_counts) != 3
            or any(type(value) is not int or value < 0
                   for value in axis_edge_counts)):
        raise ValueError("alias record axis counts must be three non-negative ints")
    edges = payload["edges"]
    if type(edges) is not list or len(edges) != active_edge_count:
        raise ValueError("alias record edge rows must match the active edge count")

    pooled: dict[tuple[str, int], set[float]] = {}
    recomputed_axis = [0, 0, 0]
    previous_index = -1
    for row in edges:
        match_index, left, right = _validate_edge_row(
            row, previous_index, input_match_count)
        previous_index = match_index
        pooled.setdefault(left, set()).add(float(row[5]))
        pooled.setdefault(right, set()).add(float(row[10]))
        recomputed_axis[int(row[12])] += 1
    if axis_edge_counts != recomputed_axis:
        raise ValueError("alias record axis counts contradict its edge rows")

    declared = payload["conflicting_endpoints"]
    if type(declared) is not list:
        raise ValueError("alias record conflicts must be a list")
    normalized: list[tuple[str, int]] = []
    for endpoint in declared:
        if (type(endpoint) not in (list, tuple) or len(endpoint) != 2
                or type(endpoint[0]) is not str or type(endpoint[1]) is not int):
            raise ValueError("alias conflict endpoint must be (frame, plane)")
        normalized.append((endpoint[0], endpoint[1]))
    if normalized != sorted(set(normalized)):
        raise ValueError("alias record conflicts must be sorted and unique")
    recomputed_conflicts = sorted(
        endpoint for endpoint, values in pooled.items() if len(values) > 1)
    if normalized != recomputed_conflicts:
        raise ValueError("alias record conflicts contradict its edge rows")


def _validate_refused_alias_payload(payload: dict) -> None:
    if set(payload) != _FAILED_ALIAS_REFUSAL_FIELDS:
        raise ValueError("refused alias record has unexpected fields")
    if payload["reason"] not in _ACTIVE_EDGE_ALIASES_REFUSAL_REASONS:
        raise ValueError("refused alias record has an unknown reason")
    field = payload["field"]
    if field is None:
        return
    if type(field) is not str:
        raise ValueError("refused alias record field must be a bounded string")
    length = _utf8_length(field)
    if length is None or length > ACTIVE_EDGE_ALIASES_MAX_FIELD_BYTES:
        raise ValueError("refused alias record field exceeds the byte cap")
    if field not in _ACTIVE_EDGE_ALIASES_FIELD_CODES:
        raise ValueError("refused alias record field is not a known code")


def _validate_failed_alias_payload(payload: object) -> None:
    """Semantically validate one record payload, excluding its own digest."""
    if type(payload) is not dict:
        raise ValueError("alias record must be a mapping")
    if payload.get("schema") != FAILED_ACTIVE_EDGE_ALIASES_SCHEMA:
        raise ValueError("alias record schema is not v1")
    status = payload.get("status")
    if status == "complete":
        _validate_complete_alias_payload(payload)
    elif status == "refused":
        _validate_refused_alias_payload(payload)
    else:
        raise ValueError("alias record status must be complete or refused")


def validate_failed_active_edge_aliases(record: object) -> None:
    """Raise ``ValueError`` unless one serialized record replays semantically.

    A matching digest is necessary but not sufficient: the edge rows, axis
    counts and pooled conflicts are independently re-derived from the rows, so a
    contradictory record cannot be blessed merely because someone recomputed
    its digest. The complete strict JSON encoding, including the record's own
    ``record_sha256`` field, must also stay inside the fixed byte budget.
    """
    if type(record) is not dict:
        raise ValueError("alias record must be a mapping")
    payload = dict(record)
    digest = payload.pop("record_sha256", None)
    if not _hex(digest):
        raise ValueError("alias record digest must be a lowercase hex digest")
    if digest != _alias_domain_digest(
            FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN,
            FAILED_ACTIVE_EDGE_ALIASES_SCHEMA, payload):
        raise ValueError("alias record digest does not replay")
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES:
        raise ValueError("alias record exceeds the fixed byte budget")
    _validate_failed_alias_payload(payload)


_BOUNDARY_SATURATION_COMPLETE_FIELDS = frozenset({
    "schema", "status", "basis", "algorithm_id", "max_shift_cells",
    "input_match_count", "input_match_identity_sha256",
    "admitted_match_count", "same_axis_kept_match_count",
    "admitted_boundary_counts", "kept_boundary_counts"})
_BOUNDARY_SATURATION_REFUSAL_FIELDS = frozenset({
    "schema", "status", "reason", "field"})


def _boundary_saturation_payload(value: "BoundarySaturationAttestation") -> dict:
    """Canonical payload of one attestation, excluding its own digest field."""
    if not value.recorded:
        return {"schema": value.schema, "status": "refused",
                "reason": value.refusal_reason, "field": value.field}
    return {
        "schema": value.schema,
        "status": "complete",
        "basis": value.basis,
        "algorithm_id": value.algorithm_id,
        "max_shift_cells": value.max_shift_cells,
        "input_match_count": value.input_match_count,
        "input_match_identity_sha256": value.input_match_identity_sha256,
        "admitted_match_count": value.admitted_match_count,
        "same_axis_kept_match_count": value.same_axis_kept_match_count,
        "admitted_boundary_counts": [value.admitted_boundary_counts[axis]
                                     for axis in range(3)],
        "kept_boundary_counts": [value.kept_boundary_counts[axis]
                                 for axis in range(3)],
    }


def boundary_saturation_record_digest(value: "BoundarySaturationAttestation") -> str:
    """Recompute one record's strict digest from its canonical payload."""
    return _sha({
        "domain": BOUNDARY_SATURATION_DIGEST_DOMAIN,
        "schema": value.schema,
        "payload": _boundary_saturation_payload(value)})


def _boundary_count_tuple(value: object) -> tuple[int, int, int]:
    """Validate one per-axis boundary count triple inside the fixed int bounds.

    The serialized form is a JSON list; the in-memory form is a tuple, so both
    are accepted and normalized rather than one being silently refused. This is
    only the structural item bound; the count-specific population bound (each
    admitted axis count against ``admitted``, each kept axis count against
    ``kept``) is re-derived separately in the complete-payload validator.
    """
    if (type(value) not in (tuple, list) or len(value) != 3
            or any(type(item) is not int or item < 0
                   or item > BOUNDARY_SATURATION_MAX_TRACE_ROWS
                   for item in value)):
        raise ValueError("boundary count triple must be three bounded ints")
    return (value[0], value[1], value[2])


def _validate_complete_boundary_payload(payload: dict) -> None:
    """Re-derive every internal bound and count relation from the payload."""
    if set(payload) != _BOUNDARY_SATURATION_COMPLETE_FIELDS:
        raise ValueError("complete boundary attestation has unexpected fields")
    if payload["basis"] != BOUNDARY_SATURATION_BASIS:
        raise ValueError("complete boundary attestation has an unknown basis")
    if payload["algorithm_id"] != BOUNDARY_SATURATION_ALGORITHM:
        raise ValueError("boundary attestation names another algorithm")
    max_shift = payload["max_shift_cells"]
    if (type(max_shift) is not int or max_shift <= 0
            or max_shift > BOUNDARY_SATURATION_MAX_SHIFT_CELLS_CAP):
        raise ValueError("boundary attestation shift radius is out of bounds")
    if not _hex(payload["input_match_identity_sha256"]):
        raise ValueError("boundary attestation identity digest is not lowercase hex")
    input_count = payload["input_match_count"]
    admitted = payload["admitted_match_count"]
    kept = payload["same_axis_kept_match_count"]
    for name, value, cap in (
            ("input_match_count", input_count, ACTIVE_EDGE_ALIASES_MAX_MATCHES),
            ("same_axis_kept_match_count", kept, ACTIVE_EDGE_ALIASES_MAX_MATCHES),
            ("admitted_match_count", admitted,
             BOUNDARY_SATURATION_MAX_TRACE_ROWS)):
        if type(value) is not int or value < 0 or value > cap:
            raise ValueError(f"boundary attestation {name} is out of bounds")
    # DS-R5B: the admitted candidate population and the solver-kept match
    # population differ by construction, so the admitted count is bounded only
    # by the trace-row cap above -- never by the solver-input identity count. A
    # record is still contradictory unless the kept set nests inside both the
    # admitted population and the solver-input identity count that produced it.
    if kept > admitted or kept > input_count:
        raise ValueError("boundary attestation counts contradict their nesting")
    admitted_counts = _boundary_count_tuple(payload["admitted_boundary_counts"])
    kept_counts = _boundary_count_tuple(payload["kept_boundary_counts"])
    # Each match contributes at most one component per row/column axis, so a
    # single axis count can never exceed its population. A resealed record that
    # claims more than that is internally contradictory.
    if any(value > admitted for value in admitted_counts):
        raise ValueError("admitted boundary counts exceed the admitted match count")
    if any(value > kept for value in kept_counts):
        raise ValueError("kept boundary counts exceed the kept match count")


def _validate_refused_boundary_payload(payload: dict) -> None:
    if set(payload) != _BOUNDARY_SATURATION_REFUSAL_FIELDS:
        raise ValueError("refused boundary attestation has unexpected fields")
    if payload["reason"] not in (BOUNDARY_SATURATION_INVALID,
                                 BOUNDARY_SATURATION_UNAVAILABLE,
                                 _ACTIVE_EDGE_ALIASES_EXHAUSTED):
        raise ValueError("refused boundary attestation has an unknown reason")
    field = payload["field"]
    if field is None:
        return
    if type(field) is not str or field not in _BOUNDARY_SATURATION_FIELD_CODES:
        raise ValueError("refused boundary attestation field is not a known code")


def _validate_boundary_payload(payload: object) -> None:
    """Semantically validate one record payload, excluding its own digest."""
    if type(payload) is not dict:
        raise ValueError("boundary attestation must be a mapping")
    if payload.get("schema") != BOUNDARY_SATURATION_SCHEMA:
        raise ValueError("boundary attestation schema is not v1")
    status = payload.get("status")
    if status == "complete":
        _validate_complete_boundary_payload(payload)
    elif status == "refused":
        _validate_refused_boundary_payload(payload)
    else:
        raise ValueError("boundary attestation status must be complete or refused")


def validate_boundary_saturation_attestation(record: object) -> None:
    """Raise ``ValueError`` unless one serialized attestation replays.

    A matching digest is necessary but not sufficient: the count nesting and
    the per-axis component bounds are independently re-derived from the record,
    so a resealed contradiction cannot be blessed. The complete strict JSON
    encoding, including the record's own ``record_sha256`` field, must also stay
    inside the fixed byte budget.
    """
    if type(record) is not dict:
        raise ValueError("boundary attestation must be a mapping")
    payload = dict(record)
    digest = payload.pop("record_sha256", None)
    if not _hex(digest):
        raise ValueError("boundary attestation digest must be a lowercase hex digest")
    if digest != _sha({"domain": BOUNDARY_SATURATION_DIGEST_DOMAIN,
                       "schema": BOUNDARY_SATURATION_SCHEMA,
                       "payload": payload}):
        raise ValueError("boundary attestation digest does not replay")
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > BOUNDARY_SATURATION_MAX_RECORD_BYTES:
        raise ValueError("boundary attestation exceeds the fixed byte budget")
    _validate_boundary_payload(payload)


def _boundary_saturation_refusal(
    reason: str, field: str | None = None,
) -> "BoundarySaturationAttestation":
    return BoundarySaturationAttestation(
        schema=BOUNDARY_SATURATION_SCHEMA, recorded=False, basis=None,
        refusal_reason=reason, field=field, algorithm_id="", max_shift_cells=0,
        input_match_count=0, input_match_identity_sha256="",
        admitted_match_count=0, same_axis_kept_match_count=0,
        admitted_boundary_counts=(0, 0, 0), kept_boundary_counts=(0, 0, 0))


@dataclass(frozen=True, slots=True)
class BoundarySaturationAttestation:
    """Failure-only, output-neutral boundary-saturation count record.

    A complete record counts, per row/column axis, how many of the matcher's
    *admitted* candidates and how many of the *same-axis-kept* matches the solver
    received carry a cell-shift component exactly on the declared search bound
    ``+/- max_shift_cells``. It retains only counts plus a digest of the exact
    selected-match identity sequence -- no shift vector, no match payload and no
    image data. A refusal is atomic: it carries one bounded reason/field and
    never a partial count. ``record_sha256`` binds the strict UTF-8 canonical
    JSON payload with explicit domain/schema separation, excluding only the
    field itself.

    ``admitted`` is read from the matcher's own ``PlaneGraphTrace`` diagnostic
    rows (every row with ``admitted`` set is one admitted candidate, so one
    match is counted once); ``same_axis_kept`` is read from the exact post-filter
    match tuple handed to ``solve_component``. The two populations are
    deliberately independent sizes: ``admitted`` is bounded by the explicit
    :data:`BOUNDARY_SATURATION_MAX_TRACE_ROWS` trace-row cap and is never
    required to fit inside the smaller solver-kept (``input_match_count``)
    population, while ``kept <= admitted`` must still hold. ``input_match_count``
    stays the solver-input identity count. Nothing here re-runs the matcher, the
    solver, the alias resolver or signature generation.
    """

    schema: str
    recorded: bool
    basis: str | None
    refusal_reason: str | None
    algorithm_id: str
    max_shift_cells: int
    input_match_count: int
    input_match_identity_sha256: str
    admitted_match_count: int
    same_axis_kept_match_count: int
    admitted_boundary_counts: tuple[int, int, int]
    kept_boundary_counts: tuple[int, int, int]
    field: str | None = None
    record_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_boundary_payload(_boundary_saturation_payload(self))
        digest = boundary_saturation_record_digest(self)
        if self.record_sha256 not in ("", digest):
            raise ValueError("boundary saturation record digest mismatch")
        object.__setattr__(self, "record_sha256", digest)

    def to_record(self) -> dict:
        payload = _boundary_saturation_payload(self)
        payload["record_sha256"] = self.record_sha256
        return payload


def _boundary_saturation_counts(
    shifts: object, max_shift_cells: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Count boundary components and cardinality for one ordered shift row set.

    ``(component_counts, cardinality)``: each entry contributes at most once per
    row/column axis, so an aliased or repeated shift can never be double-counted
    inside one row. A malformed shift raises ``ValueError``; the caller turns
    that into one atomic refusal.
    """
    counts = [0, 0, 0]
    cardinality = 0
    for shift in shifts:
        mask = _boundary_components(shift, max_shift_cells)
        if mask is None:
            raise ValueError("boundary attestation shift is malformed")
        for axis in range(3):
            counts[axis] += mask[axis]
        cardinality += sum(mask)
    return (counts[0], counts[1], counts[2]), cardinality


def _collect_boundary_saturation(
    matches: object,
    trace: object,
    max_shift_cells: object,
) -> BoundarySaturationAttestation:
    """Derive the complete attestation from the local matches and trace, or refuse.

    Both the selected-match identity digest and the admitted boundary counts are
    projected from the exact objects the graph seam already holds. The trace is
    the matcher's own diagnostic output, so it is read only, never rebuilt or
    extended; a missing or malformed trace is an atomic typed refusal rather than
    a partial or invented count. The explicit trace-row/admitted cap is checked
    by row count before the trace is iterated, so an unbounded diagnostic refuses
    atomically without any per-row work.
    """
    if type(max_shift_cells) is not int or max_shift_cells <= 0 \
            or max_shift_cells > BOUNDARY_SATURATION_MAX_SHIFT_CELLS_CAP:
        return _boundary_saturation_refusal(
            BOUNDARY_SATURATION_INVALID, "max_shift_cells")
    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        return _boundary_saturation_refusal(BOUNDARY_SATURATION_INVALID, "matches")
    if len(matches) > ACTIVE_EDGE_ALIASES_MAX_MATCHES:
        return _boundary_saturation_refusal(_ACTIVE_EDGE_ALIASES_EXHAUSTED,
                                            "matches")
    if trace is None:
        return _boundary_saturation_refusal(BOUNDARY_SATURATION_UNAVAILABLE,
                                            "trace")
    pairs = getattr(trace, "pairs", None)
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        return _boundary_saturation_refusal(BOUNDARY_SATURATION_UNAVAILABLE,
                                            "pairs")
    # DS-R5B: refuse on the raw trace-row count before any row is inspected, so
    # an over-cap or unbounded admitted population can never be partially
    # counted. The cap is above the 7,476-candidate frozen fixture.
    if len(pairs) > BOUNDARY_SATURATION_MAX_TRACE_ROWS:
        return _boundary_saturation_refusal(_ACTIVE_EDGE_ALIASES_EXHAUSTED,
                                            "admitted")

    identity_rows: list[list] = []
    kept_shifts: list[tuple[int, int]] = []
    for match in matches:
        try:
            identity_rows.append(_input_match_identity(match))
        except _ERRORS:
            return _boundary_saturation_refusal(
                BOUNDARY_SATURATION_INVALID, "input_match_identity")
        shift = _boundary_shift(getattr(match, "cell_shift_rc", None))
        if shift is None:
            return _boundary_saturation_refusal(
                BOUNDARY_SATURATION_INVALID, "shift_components")
        kept_shifts.append(shift)

    admitted_shifts: list[tuple[int, int]] = []
    for row in pairs:
        if not isinstance(row, plane_graph.PlanePairTrace):
            return _boundary_saturation_refusal(BOUNDARY_SATURATION_INVALID,
                                                "admitted")
        if row.admitted is not True:
            continue
        shift = _boundary_shift(getattr(row, "cell_shift_rc", None))
        if shift is None:
            return _boundary_saturation_refusal(
                BOUNDARY_SATURATION_INVALID, "admitted")
        admitted_shifts.append(shift)

    try:
        admitted_counts, _admitted_cardinality = _boundary_saturation_counts(
            admitted_shifts, max_shift_cells)
        kept_counts, _kept_cardinality = _boundary_saturation_counts(
            kept_shifts, max_shift_cells)
    except _ERRORS:
        return _boundary_saturation_refusal(BOUNDARY_SATURATION_INVALID,
                                            "shift_components")

    return BoundarySaturationAttestation(
        schema=BOUNDARY_SATURATION_SCHEMA, recorded=True,
        basis=BOUNDARY_SATURATION_BASIS, refusal_reason=None,
        algorithm_id=BOUNDARY_SATURATION_ALGORITHM,
        max_shift_cells=max_shift_cells,
        input_match_count=len(identity_rows),
        input_match_identity_sha256=_alias_domain_digest(
            BOUNDARY_SATURATION_INPUT_MATCH_IDENTITY_DOMAIN,
            BOUNDARY_SATURATION_SCHEMA, identity_rows),
        admitted_match_count=len(admitted_shifts),
        same_axis_kept_match_count=len(kept_shifts),
        admitted_boundary_counts=admitted_counts,
        kept_boundary_counts=kept_counts)


def _boundary_saturation_attestation(
    matches: object,
    trace: object,
    max_shift_cells: object,
) -> BoundarySaturationAttestation:
    """Fail-closed wrapper: any unexpected malformed structure refuses atomically."""
    try:
        return _collect_boundary_saturation(matches, trace, max_shift_cells)
    except _ERRORS:
        return _boundary_saturation_refusal(BOUNDARY_SATURATION_INVALID, "trace")


def boundary_saturation_attestation(
    matches: object,
    trace: object,
    max_shift_cells: object,
) -> BoundarySaturationAttestation:
    """Public bounded entry point for the failure-only boundary attestation."""
    return _boundary_saturation_attestation(matches, trace, max_shift_cells)


def _failed_alias_refusal(
    reason: str, field: str | None = None,
) -> "FailedActiveEdgeAliases":
    return FailedActiveEdgeAliases(
        schema=FAILED_ACTIVE_EDGE_ALIASES_SCHEMA, recorded=False, basis=None,
        refusal_reason=reason, failed_assignment_sha256="",
        input_match_identity_sha256="", input_match_count=0,
        active_edge_count=0, edges=(), axis_edge_counts=(0, 0, 0),
        conflicting_endpoints=(), field=field)


@dataclass(frozen=True, slots=True)
class FailedActiveEdgeAliases:
    """Bounded evaluation-only alias view of one failed best assignment.

    A complete record lists every edge the solver's own predicate activated in
    the returned best assignment, the exact source raw ``b`` label each endpoint
    was requested with and the resolver's canonical ``b`` for it, plus the
    derived endpoints that were used at more than one canonical ``b``. A refusal
    is atomic: it carries one bounded reason/field and never a partial edge row
    or a conflict conclusion. ``record_sha256`` binds the strict UTF-8 canonical
    JSON payload with explicit domain/schema separation, excluding only the
    field itself.
    """

    schema: str
    recorded: bool
    basis: str | None
    refusal_reason: str | None
    failed_assignment_sha256: str
    input_match_identity_sha256: str
    input_match_count: int
    active_edge_count: int
    edges: tuple[tuple, ...]
    axis_edge_counts: tuple[int, int, int]
    conflicting_endpoints: tuple[tuple[str, int], ...]
    field: str | None = None
    record_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_failed_alias_payload(_failed_edge_aliases_payload(self))
        digest = failed_edge_alias_record_digest(self)
        if self.record_sha256 not in ("", digest):
            raise ValueError("failed active-edge alias record digest mismatch")
        object.__setattr__(self, "record_sha256", digest)

    def to_record(self) -> dict:
        payload = _failed_edge_aliases_payload(self)
        payload["record_sha256"] = self.record_sha256
        return payload


def _failed_active_edges(
    readings: object,
    matches: object,
    solution: plane_graph.ComponentSolution,
    *,
    max_readings: int,
    max_matches: int,
    max_active_edges: int,
    max_endpoint_alternatives: int,
    max_frame_id_bytes: int,
    max_record_bytes: int,
) -> FailedActiveEdgeAliases:
    """Fail-closed wrapper around the active-edge alias collection.

    Any unexpected malformed structure becomes one atomic typed refusal instead
    of an exception escaping after the graph solver has already abstained.
    """
    try:
        return _collect_failed_active_edges(
            readings, matches, solution, max_readings=max_readings,
            max_matches=max_matches, max_active_edges=max_active_edges,
            max_endpoint_alternatives=max_endpoint_alternatives,
            max_frame_id_bytes=max_frame_id_bytes,
            max_record_bytes=max_record_bytes)
    except _ERRORS:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED, "record")


def _collect_failed_active_edges(
    readings: object,
    matches: object,
    solution: plane_graph.ComponentSolution,
    *,
    max_readings: int,
    max_matches: int,
    max_active_edges: int,
    max_endpoint_alternatives: int,
    max_frame_id_bytes: int,
    max_record_bytes: int,
) -> FailedActiveEdgeAliases:
    """Collect the failed best assignment's active edge aliases, or refuse.

    Each emitted row carries the source match's own ``b`` label in the raw
    position and the resolver's unique canonical ``b`` in the canonical
    position, so the row replays from the supplied match even when several raw
    labels collapse onto one alternative. Only identities the solver already
    returned are read: no matching, solving, least squares or signature
    generation is repeated.

    Every fixed structural cap -- the input counts and the UTF-8 frame
    identifier bound for the readings, the returned solution and *all* input
    matches, including matches the assignment never activates -- is checked
    before the canonical reading index is built or an identity digest is
    copied. The per-endpoint alternative budget is preflighted over every input
    reading before the canonical index can call ``reading_order_key`` and walk a
    reading's retained alternatives, so an over-cap reading is refused without
    any alternative traversal or alias resolution. The active-edge budget is
    counted as the traversal proceeds, so an exhausted budget refuses rather
    than materializing a partial record.
    """
    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED, "matches")
    if len(matches) > max_matches:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_EXHAUSTED,
                                     "input_match_count")
    if (isinstance(readings, (str, bytes))
            or not isinstance(readings, Sequence)):
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED, "readings")
    if len(readings) > max_readings:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_EXHAUSTED,
                                     "reading_count")

    # Conservative retained-alternative preflight. ``_canonical_reading_index``
    # calls ``reading_order_key``, which walks every reading's alternatives, so
    # an over-cap reading has to be refused before that traversal. Every input
    # reading is inspected -- not only the endpoints an active edge later uses --
    # so the bounded refusal does not depend on which edges the failed
    # assignment activates. An unknown retained count (a malformed or unsized
    # alternative collection) fails closed here as well: letting it fall through
    # would hand ``reading_order_key`` an unbounded collection to walk, so it is
    # refused before any canonical ordering, index build or alias resolution.
    for reading in readings:
        count = _retained_alternative_count(reading)
        if count is None:
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                         "endpoint_alternatives")
        if count > max_endpoint_alternatives:
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_EXHAUSTED,
                                         "endpoint_alternatives")

    # Structural frame-identifier caps, before canonical-index or digest work.
    frames, field = _solution_frame_ids(solution)
    if frames is None:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                     field or "frames")
    for frame in frames:
        verdict = _check_frame_id(frame, max_frame_id_bytes)
        if verdict is not None:
            return _failed_alias_refusal(*verdict)
    for reading in readings:
        verdict = _check_frame_id(getattr(reading, "frame", None),
                                  max_frame_id_bytes)
        if verdict is not None:
            return _failed_alias_refusal(*verdict)
    for match in matches:
        if not isinstance(match, plane_graph.PlaneMatch):
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                         "input_match_identity")
        for endpoint_frame in (getattr(match, "left", None),
                               getattr(match, "right", None)):
            verdict = _check_frame_id(endpoint_frame, max_frame_id_bytes)
            if verdict is not None:
                return _failed_alias_refusal(*verdict)

    # Bounded identity projection for every post-filter match, including the
    # ones this assignment never activates: a malformed inactive identity must
    # refuse instead of being silently dropped from the bound digest.
    identity_rows: list[list] = []
    for match in matches:
        try:
            identity_rows.append(_input_match_identity(match))
        except _ERRORS:
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                         "input_match_identity")

    try:
        read_by_index, _index_by_object, _frames = \
            plane_graph._canonical_reading_index(readings)
    except _ERRORS:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED, "readings")

    alternatives = getattr(solution, "alternatives", None)
    if (isinstance(alternatives, (str, bytes))
            or not isinstance(alternatives, Sequence) or not alternatives):
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_UNAVAILABLE,
                                     "best_assignment")
    best = alternatives[0]
    best_quadrants = getattr(best, "quadrants", None)
    observations = getattr(best, "axis_observations", None)
    if best_quadrants is None:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_UNAVAILABLE,
                                     "best_assignment")
    getter = getattr(getattr(solution, "quadrants", None), "get", None)
    if not callable(getter):
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED, "quadrants")
    try:
        solution_quadrants = tuple(getter(frame) for frame in frames)
    except _ERRORS:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED, "quadrants")
    if (isinstance(best_quadrants, (str, bytes))
            or not isinstance(best_quadrants, Sequence)):
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                     "alternatives[0].quadrants")
    if tuple(best_quadrants) != solution_quadrants:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_UNAVAILABLE,
                                     "best_assignment_identity")
    frame_quadrants = dict(zip(frames, solution_quadrants))

    if (isinstance(observations, (str, bytes))
            or not isinstance(observations, Sequence)):
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                     "axis_observations")
    if (len(observations) != 3
            or any(type(value) is not int or value < 0
                   for value in observations)):
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                     "axis_observations")

    edges: list[tuple] = []
    endpoint_canonical: dict[tuple[str, int], set[float]] = {}
    for match_index, match in enumerate(matches):
        try:
            left_frame = match.left
            right_frame = match.right
            if left_frame not in frame_quadrants or right_frame not in frame_quadrants:
                continue
            if ((frame_quadrants[right_frame] - frame_quadrants[left_frame]) % 4
                    != match.quadrant_of_right):
                continue
            left, right = plane_graph._validate_indexed_match(match, read_by_index)
            world, _left_sign = plane_graph.axis_transform(
                left.axis, frame_quadrants[left_frame])
            right_world, _right_sign = plane_graph.axis_transform(
                right.axis, frame_quadrants[right_frame])
        except _ERRORS:
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                         "edge_endpoint")
        if right_world != world:
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                         "edge_endpoint")
        if len(edges) >= max_active_edges:
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_EXHAUSTED,
                                         "active_edge_count")
        left_canonical_b, left_count = _endpoint_canonical_b(
            left, match.left_b, max_endpoint_alternatives)
        right_canonical_b, right_count = _endpoint_canonical_b(
            right, match.right_b, max_endpoint_alternatives)
        if (left_count is None or left_count > max_endpoint_alternatives
                or right_count is None
                or right_count > max_endpoint_alternatives):
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_EXHAUSTED,
                                         "endpoint_alternatives")
        if left_canonical_b is None or right_canonical_b is None:
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                         "endpoint_alias")
        left_raw_b = _raw_endpoint_b(match.left_b)
        right_raw_b = _raw_endpoint_b(match.right_b)
        if left_raw_b is None or right_raw_b is None:
            return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                         "endpoint_alias")
        endpoint_canonical.setdefault(
            (left_frame, match.left_plane), set()).add(left_canonical_b)
        endpoint_canonical.setdefault(
            (right_frame, match.right_plane), set()).add(right_canonical_b)
        edges.append((
            match_index, left_frame, match.left_plane, left.axis,
            left_raw_b, left_canonical_b,
            right_frame, match.right_plane, right.axis,
            right_raw_b, right_canonical_b,
            match.quadrant_of_right, world))

    axis_edge_counts = tuple(
        sum(1 for edge in edges if edge[12] == axis) for axis in range(3))
    if (len(edges) != sum(axis_edge_counts)
            or 2 * axis_edge_counts[0] != observations[0]
            or 2 * axis_edge_counts[1] != observations[1]
            or 2 * axis_edge_counts[2] != observations[2]):
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_MALFORMED,
                                     "observation_conservation")
    conflicting = tuple(sorted(
        endpoint for endpoint, values in endpoint_canonical.items()
        if len(values) > 1))

    value = FailedActiveEdgeAliases(
        schema=FAILED_ACTIVE_EDGE_ALIASES_SCHEMA, recorded=True,
        basis=FAILED_ACTIVE_EDGE_ALIASES_BASIS, refusal_reason=None,
        failed_assignment_sha256=_failed_assignment_digest(
            frames, frame_quadrants),
        input_match_identity_sha256=_alias_domain_digest(
            "input_match_identity", FAILED_ACTIVE_EDGE_ALIASES_SCHEMA,
            identity_rows),
        input_match_count=len(matches), active_edge_count=len(edges),
        edges=tuple(edges), axis_edge_counts=axis_edge_counts,
        conflicting_endpoints=conflicting, field=None)
    # The fixed byte budget covers the COMPLETE serialized record, including its
    # own record_sha256 field, not just the payload the digest is taken over.
    encoded = json.dumps(value.to_record(), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    if len(encoded) > max_record_bytes:
        return _failed_alias_refusal(_ACTIVE_EDGE_ALIASES_EXHAUSTED, "record_bytes")
    return value


@dataclass(frozen=True, slots=True)
class SolverPhaseCapture:
    """Bounded evaluation-only per-face phase view of one failed best assignment.

    A complete record lists every integer-face row the failed assignment's own
    predicate observed (both endpoints of every active edge, any axis, bounded to
    ``PHASE_CAPTURE_MAX_FACES``). Each row carries the match index, the endpoint's
    frame/plane/local-axis and selected raw ``b``, the exact measured
    ``distance_blocks``, that endpoint's measured ``side`` (exactly ``+/-1``), the
    signed world contribution ``axis_sign * side * distance``, its relative
    quadrant, the unwrapped face scalar, and that scalar's modulo-1 phase and unit
    phasor. The record adds the fitted per-axis translation (the declared gauge,
    anchored at the first frame), the independently recomposed per-axis
    resultant/residual, and the source digest binding it to the failed
    assignment.

    ``mismatch_reason`` is ``None`` for a complete capture whose independent
    recomposition agreed with the solver's own returned
    ``axis_resultants``/``axis_residuals`` and observation counts. A refusal is
    atomic: it carries one bounded reason/field and never a partial face row, and
    every value stays typed unavailable rather than substituted with zero.
    ``record_sha256`` binds the strict UTF-8 canonical JSON payload with explicit
    domain/schema separation, excluding only itself.
    """

    schema: str
    recorded: bool
    basis: str | None
    refusal_reason: str | None
    source_sha256: str
    frames: tuple[str, ...]
    assignments: tuple[int, ...]
    compatibility_decimals: int
    translations: tuple[tuple[str, tuple[float, float, float]], ...]
    axes: tuple[tuple, ...]
    faces: tuple[tuple, ...]
    mismatch_reason: str | None = None
    field: str | None = None
    record_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_phase_capture_payload(_phase_capture_payload(self))
        digest = solver_phase_capture_record_digest(self)
        if self.record_sha256 not in ("", digest):
            raise ValueError("phase capture record digest mismatch")
        object.__setattr__(self, "record_sha256", digest)

    def to_record(self) -> dict:
        payload = _phase_capture_payload(self)
        payload["record_sha256"] = self.record_sha256
        return payload


def _phase_capture_refusal(
    reason: str, field: str | None = None,
) -> "SolverPhaseCapture":
    return SolverPhaseCapture(
        schema=PHASE_CAPTURE_SCHEMA, recorded=False, basis=None,
        refusal_reason=reason, source_sha256="", frames=(), assignments=(),
        compatibility_decimals=0, translations=(), axes=(), faces=(), field=field)


def _phase_capture_payload(value: "SolverPhaseCapture") -> dict:
    if not value.recorded:
        return {"schema": value.schema, "status": "refused",
                "reason": value.refusal_reason, "field": value.field}
    return {
        "schema": value.schema,
        "status": "complete",
        "basis": value.basis,
        "source_sha256": value.source_sha256,
        "frames": list(value.frames),
        "assignments": list(value.assignments),
        "compatibility_decimals": value.compatibility_decimals,
        "translations": [[frame, list(vector)]
                         for frame, vector in value.translations],
        "axes": [list(row) for row in value.axes],
        "faces": [list(row) for row in value.faces],
        "mismatch_reason": value.mismatch_reason,
    }


def solver_phase_capture_record_digest(value: "SolverPhaseCapture") -> str:
    """Recompute one record's strict digest from its canonical payload."""
    return _sha({
        "domain": PHASE_CAPTURE_DIGEST_DOMAIN,
        "schema": value.schema,
        "payload": _phase_capture_payload(value)})


def _validate_phase_capture_face_row(row: object) -> int:
    """Validate one immutable 22-column face row and return its match index.

    Canonical layout: 0 match_index, 1 own_frame, 2 own_plane, 3 own_axis,
    4 own_b, 5 own_distance, 6 own_side, 7 own_sign, 8 other_frame,
    9 other_plane, 10 other_axis, 11 other_b, 12 other_distance,
    13 other_side, 14 other_sign, 15 relative_quadrant, 16 face_scalar,
    17 own_quadrant, 18 other_quadrant, 19 face_phase, 20 phasor_real,
    21 phasor_imag.

    The row serializes each endpoint's measured ``side`` next to its distance so
    the solver's own signed face contribution -- ``axis_sign * side * distance``
    -- is fully determined by the row rather than inferred from the reading.
    """
    if type(row) not in (tuple, list) or len(row) != 22:
        raise ValueError("phase face row must be a 22-element sequence")
    (match_index, left_frame, left_plane, left_axis, left_b, left_distance,
     left_side, left_sign, right_frame, right_plane, right_axis, right_b,
     right_distance, right_side, right_sign, quadrant, face_scalar,
     own_quadrant, other_quadrant, face_phase, phasor_real, phasor_imag) = row
    if type(match_index) is not int or match_index < 0:
        raise ValueError("phase face match index is out of bounds")
    for frame in (left_frame, right_frame):
        if type(frame) is not str or not frame:
            raise ValueError("phase face frame identifier is malformed")
    for plane in (left_plane, right_plane):
        if type(plane) is not int or plane < 0:
            raise ValueError("phase face plane index must be non-negative")
    for axis in (left_axis, right_axis):
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("phase face local axis must be canonical")
    for value in (left_b, right_b):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("phase face selected b must be a finite float")
    for distance in (left_distance, right_distance):
        if type(distance) is not float or not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("phase face distance must be a finite positive float")
    for side in (left_side, right_side):
        if type(side) is not int or side not in (-1, 1):
            raise ValueError("phase face measured side must be exactly +/-1")
    for sign in (left_sign, right_sign):
        if type(sign) is not int or sign not in (-1, 1):
            raise ValueError("phase face sign must be +/-1")
    for value in (quadrant, own_quadrant, other_quadrant):
        if type(value) is not int or value not in (0, 1, 2, 3):
            raise ValueError("phase face quadrant must be canonical")
    if type(face_scalar) is not float or not math.isfinite(face_scalar):
        raise ValueError("phase face scalar must be a finite float")
    if (type(face_phase) is not float or not math.isfinite(face_phase)
            or not 0.0 <= face_phase < 1.0):
        raise ValueError("phase face phase must be a finite modulo-1 float")
    for value in (phasor_real, phasor_imag):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("phase face phasor component must be finite")
    magnitude = math.hypot(phasor_real, phasor_imag)
    if abs(magnitude - 1.0) > 1e-9:
        raise ValueError("phase face phasor must be a unit phasor")
    return match_index


def _validate_phase_capture_axis_row(row: object) -> int:
    """Validate one per-axis recomposition row and return its world axis."""
    if type(row) not in (tuple, list) or len(row) != 7:
        raise ValueError("phase axis row must be a 7-element sequence")
    (axis, resultants, residuals, reconstructed_resultant,
     reconstructed_residual, observation_count, face_count) = row
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ValueError("phase axis must be canonical")
    for value in (resultants, residuals, reconstructed_resultant,
                  reconstructed_residual):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("phase axis metric must be a finite float")
    for count in (observation_count, face_count):
        if type(count) is not int or count < 0:
            raise ValueError("phase axis count must be a non-negative int")
    return axis


def _validate_complete_phase_payload(payload: dict) -> None:
    """Re-derive every phase formula and recomposition from the face rows."""
    if set(payload) != _PHASE_CAPTURE_COMPLETE_FIELDS:
        raise ValueError("complete phase record has unexpected fields")
    if payload["basis"] != PHASE_CAPTURE_BASIS:
        raise ValueError("complete phase record has an unknown basis")
    if not _hex(payload["source_sha256"]):
        raise ValueError("phase record source digest must be lowercase hex")
    frames = payload["frames"]
    if (type(frames) is not list or not frames
            or any(type(frame) is not str or not frame for frame in frames)):
        raise ValueError("phase record frames must be non-empty identifiers")
    assignments = payload["assignments"]
    if (type(assignments) is not list or len(assignments) != len(frames)
            or any(type(value) is not int or value not in (0, 1, 2, 3)
                   for value in assignments)):
        raise ValueError("phase record assignments must be canonical quadrants")
    decimals = payload["compatibility_decimals"]
    if type(decimals) is not int or decimals != PHASE_CAPTURE_COMPATIBILITY_DECIMALS:
        raise ValueError("phase record compatibility decimals are unknown")
    translations = payload["translations"]
    if type(translations) is not list or len(translations) != len(frames):
        raise ValueError("phase record translations must cover every frame")
    seen_frames: list[str] = []
    translation_by_frame: dict[str, tuple] = {}
    for row in translations:
        if (type(row) not in (tuple, list) or len(row) != 2
                or type(row[0]) is not str or row[0] not in frames
                or type(row[1]) not in (tuple, list) or len(row[1]) != 3):
            raise ValueError("phase record translation row is malformed")
        if any(type(value) is not float or not math.isfinite(value)
               for value in row[1]):
            raise ValueError("phase record translation must be finite floats")
        if row[0] in seen_frames:
            raise ValueError("phase record translations must be unique")
        seen_frames.append(row[0])
        translation_by_frame[row[0]] = row[1]
    if seen_frames != list(frames):
        raise ValueError("phase record translations must follow frame order")
    faces = payload["faces"]
    mismatch = payload["mismatch_reason"]
    if mismatch is not None and (type(mismatch) is not str or not mismatch):
        raise ValueError("phase record mismatch reason must be a non-empty string")
    if type(faces) is not list or len(faces) > PHASE_CAPTURE_MAX_FACES:
        raise ValueError("phase record face rows are over the fixed cap")
    face_frame_quadrants: dict[str, int] = {}
    face_scalars: dict[int, list[float]] = {0: [], 1: [], 2: []}
    face_rows_by_match: dict[int, list[tuple]] = {}
    for row in faces:
        match_index = _validate_phase_capture_face_row(row)
        face_rows_by_match.setdefault(match_index, []).append(tuple(row))
        left_world, left_sign = plane_graph.axis_transform(
            int(row[3]), int(row[17]))
        right_world, right_sign = plane_graph.axis_transform(
            int(row[10]), int(row[18]))
        if left_sign != int(row[7]) or right_sign != int(row[14]):
            raise ValueError("phase face sign contradicts its quadrant")
        if right_world != left_world:
            raise ValueError("phase face endpoints do not share one world axis")
        for frame, quadrant in ((row[1], int(row[17])), (row[8], int(row[18]))):
            prior = face_frame_quadrants.get(frame)
            if prior is not None and prior != quadrant:
                raise ValueError("phase face frame quadrant contradicts its rows")
            face_frame_quadrants[frame] = quadrant
        scalar = float(row[16])
        if not math.isfinite(scalar):
            raise ValueError("phase face scalar must be a finite float")
        # The solver's exact face equation: each captured scalar must equal the
        # fitted translation plus ``axis_sign * side * distance`` for that
        # endpoint. Recomputed from the row's own serialized side/sign/distance,
        # so an omitted or flipped side can never replay as the same scalar.
        translation = translation_by_frame[row[1]]
        expected_scalar = translation[left_world] + (
            float(row[7]) * float(row[6]) * float(row[5]))
        if scalar != expected_scalar:
            raise ValueError("phase face scalar contradicts its side equation")
        phase = _face_modulo_phase(scalar)
        phasor = _unit_phasor(phase)
        if (float(row[19]) != phase
                or float(row[20]) != phasor[0] or float(row[21]) != phasor[1]):
            raise ValueError("phase face phase/phasor contradicts its scalar")
        face_scalars[left_world].append(scalar)
    for match_index, endpoints in face_rows_by_match.items():
        if len(endpoints) != 2:
            raise ValueError("phase face rows must carry exactly two endpoints per match")
        first, second = endpoints
        # The two rows are the match's two endpoints, own-first. Row A's own
        # endpoint tuple (columns 1-7) must be row B's partner tuple (columns
        # 8-14) and vice versa, so a re-sealed record cannot change one copy of a
        # captured endpoint side without contradicting the other.
        if first[1:8] != second[8:15] or first[8:15] != second[1:8]:
            raise ValueError("phase face partner endpoints do not mirror")
        if int(first[15]) != int(second[15]):
            raise ValueError("phase face partner relative quadrants disagree")
        # The own endpoint is the left endpoint of one row and the right endpoint
        # of the other, so the relative quadrant flips sign between them.
        if ((int(first[18]) - int(first[17])) % 4 != int(first[15])
                or (int(second[18]) - int(second[17])) % 4
                != (-int(second[15])) % 4):
            raise ValueError("phase face partner relative quadrant relation is inconsistent")
    if set(face_frame_quadrants) != set(frames):
        raise ValueError("phase record face rows do not cover every frame")
    if [face_frame_quadrants[frame] for frame in frames] != list(assignments):
        raise ValueError("phase record assignments contradict its face rows")
    axes = payload["axes"]
    if type(axes) is not list or len(axes) != 3:
        raise ValueError("phase record must carry exactly three axis rows")
    for row in axes:
        axis = _validate_phase_capture_axis_row(row)
        scalars = face_scalars[axis]
        if int(row[6]) != len(scalars):
            raise ValueError("phase axis face count contradicts its face rows")
        if int(row[5]) != len(scalars):
            raise ValueError("phase axis observation count contradicts its face rows")
        if mismatch is not None:
            continue
        reconstructed = _recompose_phase_rows(scalars)
        if (_quantize_pair(reconstructed[0], reconstructed[1], decimals)
                != _quantize_pair(float(row[3]), float(row[4]), decimals)):
            raise ValueError("phase axis recomposition is not reproducible")
        if (_quantize_pair(float(row[1]), float(row[2]), decimals)
                != _quantize_pair(float(row[3]), float(row[4]), decimals)):
            raise ValueError("phase axis solver metrics contradict its recomposition")


def _validate_refused_phase_payload(payload: dict) -> None:
    if set(payload) != _PHASE_CAPTURE_REFUSAL_FIELDS:
        raise ValueError("refused phase record has unexpected fields")
    if payload["reason"] not in (
            _PHASE_CAPTURE_MALFORMED, _PHASE_CAPTURE_UNAVAILABLE,
            _PHASE_CAPTURE_EXHAUSTED, _PHASE_CAPTURE_MISMATCH):
        raise ValueError("refused phase record has an unknown reason")
    field = payload["field"]
    if field is not None and field not in _PHASE_CAPTURE_FIELD_CODES:
        raise ValueError("refused phase record has an unknown field")


def _validate_phase_capture_payload(payload: dict) -> None:
    if type(payload) is not dict or payload.get("schema") != PHASE_CAPTURE_SCHEMA:
        raise ValueError("phase capture record schema is malformed")
    if payload.get("status") == "refused":
        _validate_refused_phase_payload(payload)
    elif payload.get("status") == "complete":
        _validate_complete_phase_payload(payload)
    else:
        raise ValueError("phase capture record status is unknown")


def validate_solver_phase_capture(record: object) -> None:
    """Public fail-closed replay for one serialized phase-capture record."""
    if type(record) is not dict:
        raise ValueError("phase capture record must be a mapping")
    body = dict(record)
    claimed = body.pop("record_sha256", None)
    if not _hex(claimed):
        raise ValueError("phase capture record digest is missing")
    _validate_phase_capture_payload(body)
    computed = _sha({"domain": PHASE_CAPTURE_DIGEST_DOMAIN,
                     "schema": body.get("schema"), "payload": body})
    if computed != claimed:
        raise ValueError("phase capture record digest mismatch")


def _quantize_pair(first: object, second: object, decimals: int) -> tuple:
    """Re-express two values at a fixed decimal scale for comparison."""
    return (round(float(first), decimals), round(float(second), decimals))


def _face_modulo_phase(scalar: float) -> float:
    """One face scalar's modulo-1 phase, in ``[0, 1)``."""
    return float(scalar % 1.0)


def _strict_face_side(value: object) -> int | None:
    """One reading's measured side as an exact ``+/-1`` int, or ``None``.

    The solver's face equation multiplies by this side, so a missing, boolean,
    non-integral or out-of-range value is a typed malformed refusal rather than a
    silently coerced ``+1``.
    """
    if isinstance(value, bool):
        return None
    if type(value) is int:
        return value if value in (-1, 1) else None
    if type(value) is float:
        if not math.isfinite(value):
            return None
        return int(value) if value in (-1.0, 1.0) else None
    return None


def _unit_phasor(phase: float) -> tuple[float, float]:
    """The unit phasor ``(cos, sin)`` of one modulo-1 phase."""
    angle = 2.0 * math.pi * phase
    return (float(math.cos(angle)), float(math.sin(angle)))


def _recompose_phase_rows(scalars: Sequence[float]) -> tuple[float, float]:
    """Recompose ``(resultant, residual)`` from face scalars alone.

    This is the solver's own integer-face expression, re-applied to the captured
    faces rather than copied from the returned solution: the resultant is the
    magnitude of the mean unit phasor and the residual the mean absolute angular
    deviation over ``2*pi``. An empty axis returns the solver's own ``0.0``
    convention for a skipped axis.
    """
    values = [float(value) for value in scalars]
    if not values:
        return 0.0, 0.0
    angles = 2.0 * np.pi * np.mod(np.asarray(values, dtype=float), 1.0)
    phasors = np.exp(1j * angles)
    resultant = float(abs(np.mean(phasors)))
    mean_angle = float(np.angle(np.mean(phasors)))
    deviation = np.angle(np.exp(1j * (angles - mean_angle)))
    residual = float(np.mean(np.abs(deviation)) / (2.0 * np.pi))
    return resultant, residual


def _phase_metric_map(value: object) -> dict[int, float] | None:
    """Read one finite ``{axis: float}`` diagnostic mapping, or ``None``."""
    items = getattr(value, "items", None)
    if not callable(items):
        return None
    try:
        pairs = tuple(items())
    except _ERRORS:
        return None
    metrics: dict[int, float] = {}
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            return None
        key, item = pair
        if type(key) is not int or key not in (0, 1, 2) or key in metrics:
            return None
        try:
            number = float(item)
        except _ERRORS:
            return None
        if not np.isfinite(number):
            return None
        metrics[key] = number
    return metrics


def _phase_capture_control(
    observations: object,
    axis_scalars: dict[int, list[float]],
    axis_resultants: dict[int, float],
    axis_residuals: dict[int, float],
) -> str | None:
    """Reconcile the recomposed axes with the solver's own returned metrics.

    Returns a bounded mismatch code, or ``None`` when every axis the solver
    reported agrees with the independent recomposition and observation count. An
    axis the solver reported but this capture did not reach, or a reconstruction
    whose quantized metrics differ, is a typed mismatch rather than a silently
    dropped or substituted axis.
    """
    for axis in range(3):
        scalars = axis_scalars[axis]
        has_resultant = axis in axis_resultants
        has_residual = axis in axis_residuals
        if not has_resultant or not has_residual:
            if has_resultant or has_residual or scalars:
                return "axis_metric_coverage"
            continue
        if int(observations[axis]) != len(scalars):
            return "axis_observation_count"
        reconstructed = _recompose_phase_rows(scalars)
        if (_quantize_pair(axis_resultants[axis], axis_residuals[axis],
                           PHASE_CAPTURE_COMPATIBILITY_DECIMALS)
                != _quantize_pair(reconstructed[0], reconstructed[1],
                                  PHASE_CAPTURE_COMPATIBILITY_DECIMALS)):
            return "axis_recomposition"
    return None


def _collect_failed_phase_capture(
        readings: object,
        matches: object,
        solution: plane_graph.ComponentSolution,
) -> "SolverPhaseCapture":
    """Capture the failed best assignment's per-face phase evidence, or refuse.

    Every structural precondition -- the fixed input caps, the solution's own
    frames and best alternative, the canonical reading index and the endpoint
    aliases -- is re-derived from the same local inputs the solver consumed. The
    fixed per-axis least-squares translation and the face scalars are recomputed
    with the solver's own expression, and the recomposed resultants/residuals are
    compared with the solver's returned metrics at a fixed decimal scale. A
    disagreement is the typed ``phase_recomposition_mismatch`` refusal, never a
    substituted zero. Nothing here mutates an input or the returned solution.
    """
    if isinstance(readings, (str, bytes)) or not isinstance(readings, Sequence):
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "readings")
    if len(readings) > ACTIVE_EDGE_ALIASES_MAX_READINGS:
        return _phase_capture_refusal(_PHASE_CAPTURE_EXHAUSTED, "readings")
    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "matches")
    if len(matches) > ACTIVE_EDGE_ALIASES_MAX_MATCHES:
        return _phase_capture_refusal(_PHASE_CAPTURE_EXHAUSTED, "matches")
    if not isinstance(solution, plane_graph.ComponentSolution):
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "solution")

    frames, field = _solution_frame_ids(solution)
    if frames is None or not frames:
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, field or "frames")
    for frame in frames:
        verdict = _check_frame_id(frame, PHASE_CAPTURE_MAX_FRAME_ID_BYTES)
        if verdict is not None:
            return _phase_capture_refusal(*verdict)
    for reading in readings:
        verdict = _check_frame_id(getattr(reading, "frame", None),
                                  PHASE_CAPTURE_MAX_FRAME_ID_BYTES)
        if verdict is not None:
            return _phase_capture_refusal(*verdict)
        if _retained_alternative_count(reading) is None:
            return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED,
                                          "endpoint_alternatives")
    for match in matches:
        if not isinstance(match, plane_graph.PlaneMatch):
            return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "matches")

    getter = getattr(getattr(solution, "quadrants", None), "get", None)
    if not callable(getter):
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "quadrants")
    try:
        frame_quadrants = {frame: getter(frame) for frame in frames}
    except _ERRORS:
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "quadrants")
    for quadrant in frame_quadrants.values():
        if type(quadrant) is not int or quadrant not in (0, 1, 2, 3):
            return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "quadrants")

    alternatives = getattr(solution, "alternatives", None)
    if (isinstance(alternatives, (str, bytes))
            or not isinstance(alternatives, Sequence) or not alternatives):
        return _phase_capture_refusal(_PHASE_CAPTURE_UNAVAILABLE,
                                      "best_assignment")
    best_quadrants = getattr(alternatives[0], "quadrants", None)
    if (isinstance(best_quadrants, (str, bytes))
            or not isinstance(best_quadrants, Sequence)):
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED,
                                      "alternatives[0].quadrants")
    if tuple(best_quadrants) != tuple(frame_quadrants[frame] for frame in frames):
        return _phase_capture_refusal(_PHASE_CAPTURE_UNAVAILABLE,
                                      "best_assignment")
    observations = getattr(alternatives[0], "axis_observations", None)
    if (isinstance(observations, (str, bytes))
            or not isinstance(observations, Sequence) or len(observations) != 3
            or any(type(value) is not int or value < 0 for value in observations)):
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED,
                                      "axis_observations")

    try:
        read_by_index, _index_by_object, _index_frames = \
            plane_graph._canonical_reading_index(readings)
    except _ERRORS:
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "readings")

    constraints: dict[int, list[tuple[str, str, float]]] = {0: [], 1: [], 2: []}
    # One row per active edge *endpoint* (left then right), in match order. The
    # solver counts one observation per endpoint, so this list is the exact face
    # universe and duplicates are preserved rather than collapsed: the same
    # reading may legitimately back several active edges.
    endpoint_rows: dict[int, list[tuple]] = {0: [], 1: [], 2: []}
    seen_without_b: set[tuple] = set()
    for match_index, match in enumerate(matches):
        try:
            left = read_by_index.get((match.left, match.left_plane))
            right = read_by_index.get((match.right, match.right_plane))
            if left is None or right is None:
                return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED,
                                              "readings_b")
            left_quadrant = frame_quadrants[match.left]
            right_quadrant = frame_quadrants[match.right]
            if (right_quadrant - left_quadrant) % 4 != match.quadrant_of_right:
                continue
            world, left_sign = plane_graph.axis_transform(left.axis, left_quadrant)
            right_world, right_sign = plane_graph.axis_transform(
                right.axis, right_quadrant)
            if right_world != world:
                return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "matches")
            left_distance = left.distance_for(match.left_b)
            right_distance = right.distance_for(match.right_b)
        except _ERRORS:
            return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "matches_b")
        if left_distance is None or right_distance is None:
            return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "readings_b")
        left_side = _strict_face_side(getattr(left, "side", None))
        right_side = _strict_face_side(getattr(right, "side", None))
        if left_side is None or right_side is None:
            return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED,
                                          "endpoint_side")
        left_col = left_sign * left_side
        right_col = right_sign * right_side
        constraints[world].append(
            (match.left, match.right,
             float(left_col * left_distance - right_col * right_distance)))
        key = (match.left, match.left_plane, match.right, match.right_plane,
               match.quadrant_of_right)
        if key in seen_without_b:
            return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "duplicate_b")
        seen_without_b.add(key)
        left_meta = (match.left, match.left_plane, int(left.axis),
                     float(match.left_b), float(left_distance), left_side,
                     int(left_sign), int(left_quadrant))
        right_meta = (match.right, match.right_plane, int(right.axis),
                      float(match.right_b), float(right_distance), right_side,
                      int(right_sign), int(right_quadrant))
        base = (match_index, match.quadrant_of_right)
        endpoint_rows[world].append((base, left_meta, right_meta))
        endpoint_rows[world].append((base, right_meta, left_meta))

    face_total = sum(len(endpoint_rows[axis]) for axis in range(3))
    if face_total > PHASE_CAPTURE_MAX_FACES:
        return _phase_capture_refusal(_PHASE_CAPTURE_EXHAUSTED, "face_count")

    anchor = frames[0]
    unknown_frames = [frame for frame in frames if frame != anchor]
    column_of = {frame: index for index, frame in enumerate(unknown_frames)}
    axis_translation: dict[str, list[float]] = {
        frame: [0.0, 0.0, 0.0] for frame in frames}
    for axis in range(3):
        rows = constraints[axis]
        if not rows:
            continue
        matrix = np.zeros((len(rows), len(unknown_frames)), dtype=float)
        target = np.zeros(len(rows), dtype=float)
        for row_index, (left_frame, right_frame, delta) in enumerate(rows):
            if right_frame != anchor:
                matrix[row_index, column_of[right_frame]] += 1.0
            if left_frame != anchor:
                matrix[row_index, column_of[left_frame]] -= 1.0
            target[row_index] = delta
        if not unknown_frames:
            fitted = np.zeros(0, dtype=float)
        else:
            fitted, _residues, _rank, _singular = np.linalg.lstsq(
                matrix, target, rcond=None)
        for frame in unknown_frames:
            axis_translation[frame][axis] = float(fitted[column_of[frame]])
    translations: list[tuple[str, tuple[float, float, float]]] = []
    for frame in frames:
        vector = tuple(float(value) for value in axis_translation[frame])
        if any(not math.isfinite(value) for value in vector):
            return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "translations")
        translations.append((frame, vector))

    faces: list[list] = []
    axes: list[list] = []
    axis_scalars: dict[int, list[float]] = {0: [], 1: [], 2: []}
    recomposed: dict[int, tuple[float, float]] = {}
    for axis in range(3):
        scalars: list[float] = []
        for (base, own, other) in endpoint_rows[axis]:
            match_index, quadrant = base
            (frame, plane, local_axis, selected_b, distance, side, sign,
             own_quadrant) = own
            (other_frame, other_plane, other_axis, other_b, other_distance,
             other_side, other_sign, other_quadrant) = other
            # The solver's exact face equation (``plane_graph._evaluate_assignment``):
            # the signed world contribution keeps the reading's own measured side.
            scalar = axis_translation[frame][axis] + sign * side * distance
            if not math.isfinite(float(scalar)):
                return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "axes")
            scalars.append(float(scalar))
            phase = _face_modulo_phase(float(scalar))
            phasor_real, phasor_imag = _unit_phasor(phase)
            # The canonical 22-column face row (one per active edge endpoint, the
            # own endpoint first): 0 match_index, 1 own_frame, 2 own_plane,
            # 3 own_axis, 4 own_b, 5 own_distance, 6 own_side, 7 own_sign,
            # 8 other_frame, 9 other_plane, 10 other_axis, 11 other_b,
            # 12 other_distance, 13 other_side, 14 other_sign,
            # 15 relative_quadrant, 16 face_scalar, 17 own_quadrant,
            # 18 other_quadrant, 19 face_phase, 20 phasor_real, 21 phasor_imag.
            faces.append([
                match_index, frame, plane, local_axis, selected_b, distance,
                side, sign, other_frame, other_plane, other_axis, other_b,
                other_distance, other_side, other_sign, quadrant, float(scalar),
                own_quadrant, other_quadrant, phase, phasor_real, phasor_imag])
        axis_scalars[axis] = scalars
        recomposed[axis] = _recompose_phase_rows(scalars)

    axis_resultants = _phase_metric_map(getattr(solution, "axis_resultants", None))
    if axis_resultants is None:
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "axis_resultants")
    axis_residuals = _phase_metric_map(getattr(solution, "axis_residuals", None))
    if axis_residuals is None:
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "axis_residuals")
    mismatch = _phase_capture_control(observations, axis_scalars,
                                      axis_resultants, axis_residuals)
    if mismatch is not None:
        return _phase_capture_refusal(_PHASE_CAPTURE_MISMATCH, mismatch)
    # The axis row carries the solver's returned metrics in the first metric
    # pair and the independent recomposition in the second, so the two are never
    # conflated. ``observation_count`` is the solver's own per-axis face
    # observation count (one per active endpoint), not twice that.
    for axis in range(3):
        resultant, residual = recomposed[axis]
        axes.append([axis, float(axis_resultants[axis]),
                     float(axis_residuals[axis]), resultant, residual,
                     len(axis_scalars[axis]), len(axis_scalars[axis])])

    source_sha256 = _failed_assignment_digest(frames, frame_quadrants)
    value = SolverPhaseCapture(
        schema=PHASE_CAPTURE_SCHEMA, recorded=True, basis=PHASE_CAPTURE_BASIS,
        refusal_reason=None, source_sha256=source_sha256, frames=tuple(frames),
        assignments=tuple(frame_quadrants[frame] for frame in frames),
        compatibility_decimals=PHASE_CAPTURE_COMPATIBILITY_DECIMALS,
        translations=tuple(translations),
        axes=tuple(tuple(row) for row in axes),
        faces=tuple(tuple(row) for row in faces), mismatch_reason=None)
    encoded = json.dumps(value.to_record(), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    if len(encoded) > PHASE_CAPTURE_MAX_RECORD_BYTES:
        return _phase_capture_refusal(_PHASE_CAPTURE_EXHAUSTED, "record_bytes")
    return value


def failed_phase_capture(
        readings: object,
        matches: object,
        solution: plane_graph.ComponentSolution,
) -> "SolverPhaseCapture":
    """Fail-closed wrapper around the bounded phase-capture collection."""
    try:
        return _collect_failed_phase_capture(readings, matches, solution)
    except _ERRORS:
        return _phase_capture_refusal(_PHASE_CAPTURE_MALFORMED, "record")


@dataclass(frozen=True, slots=True)
class SolverInputBindings:
    """Bounded evaluation-only binding of the complete solver input set.

    A complete record covers *every* post-same-axis match handed to
    ``solve_component`` -- not only the edges the best assignment activated. Each
    edge row names both endpoints by their retained-alternative index in a shared
    canonical reading table, carries that alternative's measured ``b``,
    ``distance_blocks`` and ``support``, the reading's measured ``side``, the
    world-axis ``sign`` implied by the captured best assignment, the endpoint's
    local ``axis``, the relative quadrant and whether the edge was active under
    that assignment. Each reading table row names the canonical reading (its
    global sequence index, frame and per-frame plane index) and serializes its
    full measured alternative table, so an alternative index is bound to the
    exact table it came from rather than to a re-derived lookalike.

    ``failed_assignment_sha256`` is the shared failed-best-assignment digest the
    active-edge alias and phase-capture records also carry, and it is recomputed
    from ``frames``/``assignments`` by the replay validator.
    ``input_match_identity_sha256`` binds the ordered scalar identity of all
    input matches. A refusal is atomic: it carries one bounded reason/field and
    never a partial edge or reading row. ``record_sha256`` binds the strict UTF-8
    canonical JSON payload with explicit domain/schema separation, excluding
    only itself.
    """

    schema: str
    recorded: bool
    basis: str | None
    refusal_reason: str | None
    failed_assignment_sha256: str
    input_match_identity_sha256: str
    input_match_count: int
    active_edge_count: int
    frames: tuple[str, ...]
    assignments: tuple[int, ...]
    readings: tuple[tuple, ...]
    edges: tuple[tuple, ...]
    field: str | None = None
    record_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_solver_input_bindings_payload(
            _solver_input_bindings_payload(self))
        digest = solver_input_bindings_record_digest(self)
        if self.record_sha256 not in ("", digest):
            raise ValueError("solver input bindings record digest mismatch")
        object.__setattr__(self, "record_sha256", digest)

    def to_record(self) -> dict:
        payload = _solver_input_bindings_payload(self)
        payload["record_sha256"] = self.record_sha256
        return payload


def _solver_input_bindings_refusal(
    reason: str, field: str | None = None,
) -> "SolverInputBindings":
    return SolverInputBindings(
        schema=SOLVER_INPUT_BINDINGS_SCHEMA, recorded=False, basis=None,
        refusal_reason=reason, failed_assignment_sha256="",
        input_match_identity_sha256="", input_match_count=0,
        active_edge_count=0, frames=(), assignments=(), readings=(), edges=(),
        field=field)


def _solver_input_bindings_payload(value: "SolverInputBindings") -> dict:
    """Canonical payload of one record, without its own digest field."""
    if not value.recorded:
        return {"schema": value.schema, "status": "refused",
                "reason": value.refusal_reason, "field": value.field}
    return {
        "schema": value.schema,
        "status": "complete",
        "basis": value.basis,
        "failed_assignment_sha256": value.failed_assignment_sha256,
        "input_match_identity_sha256": value.input_match_identity_sha256,
        "input_match_count": value.input_match_count,
        "active_edge_count": value.active_edge_count,
        "frames": list(value.frames),
        "assignments": list(value.assignments),
        "readings": [list(row) for row in value.readings],
        "edges": [list(row) for row in value.edges],
    }


def solver_input_bindings_record_digest(value: "SolverInputBindings") -> str:
    """Recompute one record's strict digest from its canonical payload."""
    return _sha({
        "domain": SOLVER_INPUT_BINDINGS_DIGEST_DOMAIN,
        "schema": value.schema,
        "payload": _solver_input_bindings_payload(value)})


def _validate_solver_input_reading_row(row: object) -> int:
    """Validate one immutable reading-table row and return its reading index.

    Canonical layout: ``[reading_index, frame, plane, local_axis, side,
    alternatives]`` where ``alternatives`` is the reading's measured table in
    canonical order, each entry ``[b, distance_blocks, support]``. The measured
    alternative order is re-derived (non-increasing support, ``b`` descending
    within one support) rather than trusted from the serialized digest.
    """
    if type(row) not in (tuple, list) or len(row) != 6:
        raise ValueError("solver input reading row must be a 6-element sequence")
    reading_index, frame, plane, axis, side, alternatives = row
    if type(reading_index) is not int or reading_index < 0:
        raise ValueError("solver input reading index must be non-negative")
    if type(frame) is not str or not frame:
        raise ValueError("solver input reading frame must be a non-empty string")
    if type(plane) is not int or plane < 0:
        raise ValueError("solver input reading plane must be non-negative")
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ValueError("solver input reading axis must be canonical")
    if type(side) is not int or side not in (-1, 1):
        raise ValueError("solver input reading side must be exactly +/-1")
    if type(alternatives) is not list or not alternatives:
        raise ValueError("solver input reading needs a non-empty alternative table")
    previous: tuple[float, float] | None = None
    for entry in alternatives:
        if type(entry) not in (tuple, list) or len(entry) != 3:
            raise ValueError("solver input alternative must be a triple")
        b, distance, support = entry
        for value in (b, distance, support):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("solver input alternative must be finite floats")
        if b <= 0.0 or distance <= 0.0 or support < 0.0:
            raise ValueError("solver input alternative must be positive")
        if previous is not None:
            if support > previous[0]:
                raise ValueError("solver input alternatives must not gain support")
            if support == previous[0] and b > previous[1]:
                raise ValueError("solver input alternatives must descend in b on ties")
        previous = (float(support), float(b))
    return reading_index


def _validate_solver_input_endpoint(row: object) -> tuple[int, int]:
    """Validate one 8-column edge endpoint and return ``(reading, alternative)``.

    Canonical layout: ``[reading_ref, alternative_index, b, distance_blocks,
    support, side, sign, local_axis]``. The ``sign`` is the world-axis sign the
    captured best assignment implies for the endpoint's frame; the payload
    validator re-derives it from ``axis_transform`` rather than trusting it.
    """
    if type(row) not in (tuple, list) or len(row) != 8:
        raise ValueError("solver input endpoint must be an 8-element sequence")
    (reading_ref, alternative_index, b, distance, support, side, sign,
     axis) = row
    if type(reading_ref) is not int or reading_ref < 0:
        raise ValueError("solver input endpoint reading index is malformed")
    if type(alternative_index) is not int or alternative_index < 0:
        raise ValueError("solver input endpoint alternative index is malformed")
    for value in (b, distance, support):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("solver input endpoint alternative must be finite floats")
    if b <= 0.0 or distance <= 0.0 or support < 0.0:
        raise ValueError("solver input endpoint alternative must be positive")
    if type(side) is not int or side not in (-1, 1):
        raise ValueError("solver input endpoint side must be exactly +/-1")
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("solver input endpoint sign must be +/-1")
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ValueError("solver input endpoint axis must be canonical")
    return reading_ref, alternative_index


def _validate_solver_input_edge_row(row: object) -> tuple:
    """Validate one edge row and return its declared index and endpoint refs."""
    if type(row) not in (tuple, list) or len(row) != 5:
        raise ValueError("solver input edge row must be a 5-element sequence")
    match_index, left, right, quadrant, active = row
    if type(match_index) is not int or match_index < 0:
        raise ValueError("solver input match index must be non-negative")
    if type(quadrant) is not int or quadrant not in (0, 1, 2, 3):
        raise ValueError("solver input relative quadrant must be canonical")
    if type(active) is not bool:
        raise ValueError("solver input active flag must be a boolean")
    left_ref, left_alt = _validate_solver_input_endpoint(left)
    right_ref, right_alt = _validate_solver_input_endpoint(right)
    return match_index, left_ref, left_alt, right_ref, right_alt


def _validate_solver_input_endpoint_binding(
    reading_row: tuple, alternative_index: int, endpoint: tuple,
    frame_quadrant: dict[str, int],
) -> int:
    """Re-derive one endpoint from its reading table and best assignment.

    Returns the endpoint's world axis. Every serialized endpoint field is
    re-derived from the bound measured alternative table and the captured
    assignment, so a resealed row cannot point an alternative index at a
    different table entry or change a measured side without detection.
    """
    _reading_index, frame, _plane, axis, side, alternatives = reading_row
    quadrant = frame_quadrant.get(frame)
    if quadrant is None:
        raise ValueError("solver input endpoint frame is not in the assignment")
    if type(alternative_index) is not int or alternative_index < 0:
        raise ValueError("solver input endpoint alternative index is malformed")
    if alternative_index >= len(alternatives):
        raise ValueError("solver input alternative index is out of the table")
    entry = alternatives[alternative_index]
    if (endpoint[2], endpoint[3], endpoint[4]) != (entry[0], entry[1], entry[2]):
        raise ValueError("solver input endpoint contradicts its alternative table")
    if int(endpoint[7]) != int(axis):
        raise ValueError("solver input endpoint axis contradicts its reading")
    if int(endpoint[5]) != int(side):
        raise ValueError("solver input endpoint side contradicts its reading")
    world, expected_sign = plane_graph.axis_transform(int(axis), quadrant)
    if int(endpoint[6]) != int(expected_sign):
        raise ValueError("solver input endpoint sign contradicts its assignment")
    return int(world)


def _validate_complete_solver_input_bindings_payload(payload: dict) -> None:
    """Re-derive every binding, count and active flag from the record rows."""
    if set(payload) != _SOLVER_INPUT_BINDINGS_COMPLETE_FIELDS:
        raise ValueError("complete solver input bindings have unexpected fields")
    if payload["basis"] != SOLVER_INPUT_BINDINGS_BASIS:
        raise ValueError("complete solver input bindings have an unknown basis")
    if not _hex(payload["failed_assignment_sha256"]):
        raise ValueError("solver input bindings assignment digest must be hex")
    if not _hex(payload["input_match_identity_sha256"]):
        raise ValueError("solver input bindings identity digest must be hex")
    frames = payload["frames"]
    if (type(frames) is not list or not frames
            or any(type(frame) is not str or not frame for frame in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError("solver input bindings frames must be unique identifiers")
    assignments = payload["assignments"]
    if (type(assignments) is not list or len(assignments) != len(frames)
            or any(type(value) is not int or value not in (0, 1, 2, 3)
                   for value in assignments)):
        raise ValueError("solver input bindings assignments must be canonical")
    frame_quadrant = dict(zip(frames, assignments))
    if (payload["failed_assignment_sha256"]
            != _failed_assignment_digest(frames, frame_quadrant)):
        raise ValueError("solver input bindings assignment digest does not replay")
    input_match_count = payload["input_match_count"]
    if (type(input_match_count) is not int or input_match_count < 0
            or input_match_count > SOLVER_INPUT_BINDINGS_MAX_MATCHES):
        raise ValueError("solver input bindings match count is out of bounds")
    active_edge_count = payload["active_edge_count"]
    if (type(active_edge_count) is not int or active_edge_count < 0
            or active_edge_count > input_match_count):
        raise ValueError("solver input bindings active count is out of bounds")

    readings = payload["readings"]
    if type(readings) is not list or len(readings) > SOLVER_INPUT_BINDINGS_MAX_READINGS:
        raise ValueError("solver input bindings reading table is over the cap")
    table: dict[int, tuple] = {}
    seen_canonical: set[tuple[str, int]] = set()
    previous_reading = -1
    for row in readings:
        reading_index = _validate_solver_input_reading_row(row)
        if reading_index <= previous_reading:
            raise ValueError("solver input bindings readings must be ordered and unique")
        previous_reading = reading_index
        canonical = (row[1], int(row[2]))
        if canonical in seen_canonical:
            raise ValueError("solver input bindings readings repeat a canonical index")
        seen_canonical.add(canonical)
        table[reading_index] = tuple(row)

    edges = payload["edges"]
    if type(edges) is not list or len(edges) != input_match_count:
        raise ValueError("solver input bindings edges must match the input count")
    used_readings: set[int] = set()
    recomputed_active = 0
    previous_match = -1
    for row in edges:
        match_index, left_ref, left_alt, right_ref, right_alt = \
            _validate_solver_input_edge_row(row)
        if match_index != previous_match + 1:
            raise ValueError("solver input bindings match indices must be contiguous")
        previous_match = match_index
        left_row, right_row = table.get(left_ref), table.get(right_ref)
        if left_row is None or right_row is None:
            raise ValueError("solver input endpoint names a missing reading")
        left_world = _validate_solver_input_endpoint_binding(
            left_row, left_alt, tuple(row[1]), frame_quadrant)
        right_world = _validate_solver_input_endpoint_binding(
            right_row, right_alt, tuple(row[2]), frame_quadrant)
        used_readings.add(left_ref)
        used_readings.add(right_ref)
        relative = int(row[3])
        quadrants_align = (
            (frame_quadrant[right_row[1]] - frame_quadrant[left_row[1]]) % 4
            == relative)
        expected_active = bool(quadrants_align and left_world == right_world)
        if row[4] is not expected_active:
            raise ValueError("solver input edge active flag contradicts the assignment")
        if expected_active:
            recomputed_active += 1
    if set(table) != used_readings:
        raise ValueError("solver input bindings reading table is not the exact set")
    if recomputed_active != active_edge_count:
        raise ValueError("solver input bindings active count contradicts its edges")


def _validate_refused_solver_input_bindings_payload(payload: dict) -> None:
    if set(payload) != _SOLVER_INPUT_BINDINGS_REFUSAL_FIELDS:
        raise ValueError("refused solver input bindings have unexpected fields")
    if payload["reason"] not in (
            _SOLVER_INPUT_BINDINGS_MALFORMED, _SOLVER_INPUT_BINDINGS_UNAVAILABLE,
            _SOLVER_INPUT_BINDINGS_EXHAUSTED):
        raise ValueError("refused solver input bindings have an unknown reason")
    field = payload["field"]
    if field is None:
        return
    if type(field) is not str or field not in _SOLVER_INPUT_BINDINGS_FIELD_CODES:
        raise ValueError("refused solver input bindings field is not a known code")


def _validate_solver_input_bindings_payload(payload: object) -> None:
    if (type(payload) is not dict
            or payload.get("schema") != SOLVER_INPUT_BINDINGS_SCHEMA):
        raise ValueError("solver input bindings schema is malformed")
    if payload.get("status") == "refused":
        _validate_refused_solver_input_bindings_payload(payload)
    elif payload.get("status") == "complete":
        _validate_complete_solver_input_bindings_payload(payload)
    else:
        raise ValueError("solver input bindings status is unknown")


def validate_solver_input_bindings(record: object) -> None:
    """Public fail-closed replay for one serialized solver-input binding record.

    A matching digest is necessary but not sufficient: every reading index,
    alternative index, measured value, endpoint side/sign/axis, relative
    quadrant, active flag and both counts is independently re-derived from the
    rows and the shared assignment digest, and the complete strict JSON encoding
    must stay inside the fixed byte budget.
    """
    if type(record) is not dict:
        raise ValueError("solver input bindings must be a mapping")
    body = dict(record)
    claimed = body.pop("record_sha256", None)
    if not _hex(claimed):
        raise ValueError("solver input bindings digest is missing")
    _validate_solver_input_bindings_payload(body)
    computed = _sha({"domain": SOLVER_INPUT_BINDINGS_DIGEST_DOMAIN,
                     "schema": body.get("schema"), "payload": body})
    if computed != claimed:
        raise ValueError("solver input bindings digest mismatch")
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > SOLVER_INPUT_BINDINGS_MAX_RECORD_BYTES:
        raise ValueError("solver input bindings exceed the fixed byte budget")


def _endpoint_alternative(
    reading: lattice_planes.PlaneReading, requested: object,
    max_alternatives: int,
) -> tuple[int | None, tuple | None, int | None]:
    """Resolve one raw ``b`` to ``(alternative_index, table_row, inspected)``.

    The unique-alternative contract is the reading's own
    :meth:`~mc_photogrammetry.reconstruction.lattice_planes.PlaneReading.matching_alternatives`;
    an ambiguous or missing label yields no index and is refused by the caller.
    The returned table row is ``(b, distance_blocks, support)`` exactly as the
    reading measured it. The inspected count bounds the traversal: a reading
    whose retained table exceeds the cap is refused before it is walked.
    """
    if (isinstance(requested, (bool, np.bool_))
            or not isinstance(requested, (int, float, np.integer, np.floating))):
        return None, None, None
    alternatives = getattr(reading, "alternatives", None)
    if (isinstance(alternatives, (str, bytes))
            or not isinstance(alternatives, Sequence)):
        return None, None, None
    inspected = len(alternatives)
    if inspected > max_alternatives:
        return None, None, inspected
    try:
        matches = reading.matching_alternatives(requested)
    except _ERRORS:
        return None, None, inspected
    if len(matches) != 1:
        return None, None, inspected
    matched = matches[0]
    index = None
    for position, candidate in enumerate(alternatives):
        if candidate is matched:
            index = position
            break
    if index is None:
        return None, None, inspected
    try:
        table_row = (float(matched.blocks_per_period),
                     float(matched.distance_blocks), float(matched.support))
    except _ERRORS:
        return None, None, inspected
    if (not all(math.isfinite(value) for value in table_row)
            or table_row[0] <= 0.0 or table_row[1] <= 0.0
            or table_row[2] < 0.0):
        return None, None, inspected
    return index, table_row, inspected


def _reading_alternative_table(reading: object) -> list[list[float]] | None:
    """Project one reading's full measured alternative table, or ``None``."""
    alternatives = getattr(reading, "alternatives", None)
    if (isinstance(alternatives, (str, bytes))
            or not isinstance(alternatives, Sequence)):
        return None
    rows: list[list[float]] = []
    for alternative in alternatives:
        try:
            b = float(getattr(alternative, "blocks_per_period"))
            distance = float(getattr(alternative, "distance_blocks"))
            support = float(getattr(alternative, "support"))
        except _ERRORS:
            return None
        if (not all(math.isfinite(value) for value in (b, distance, support))
                or b <= 0.0 or distance <= 0.0 or support < 0.0):
            return None
        rows.append([b, distance, support])
    return rows or None


def _collect_failed_solver_input_bindings(
    readings: object,
    matches: object,
    solution: plane_graph.ComponentSolution,
) -> "SolverInputBindings":
    """Bind every post-same-axis solver input, or refuse atomically.

    The record is derived from the same local readings/matches/solution the
    graph seam already holds: no matcher, solver, least-squares or signature
    pass is repeated. All fixed structural caps (input counts, per-reading
    retained alternatives, frame-identifier bytes and the record byte budget)
    are preflighted or enforced before/while materializing the record, and any
    malformed value becomes one atomic typed refusal instead of a partial row.
    """
    max_readings = SOLVER_INPUT_BINDINGS_MAX_READINGS
    max_matches = SOLVER_INPUT_BINDINGS_MAX_MATCHES
    max_alternatives = SOLVER_INPUT_BINDINGS_MAX_ALTERNATIVES
    max_frame_id_bytes = SOLVER_INPUT_BINDINGS_MAX_FRAME_ID_BYTES
    max_record_bytes = SOLVER_INPUT_BINDINGS_MAX_RECORD_BYTES
    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        return _solver_input_bindings_refusal(_SOLVER_INPUT_BINDINGS_MALFORMED,
                                              "matches")
    if len(matches) > max_matches:
        return _solver_input_bindings_refusal(_SOLVER_INPUT_BINDINGS_EXHAUSTED,
                                              "matches")
    if (isinstance(readings, (str, bytes))
            or not isinstance(readings, Sequence)):
        return _solver_input_bindings_refusal(_SOLVER_INPUT_BINDINGS_MALFORMED,
                                              "readings")
    if len(readings) > max_readings:
        return _solver_input_bindings_refusal(_SOLVER_INPUT_BINDINGS_EXHAUSTED,
                                              "readings")
    if not isinstance(solution, plane_graph.ComponentSolution):
        return _solver_input_bindings_refusal(_SOLVER_INPUT_BINDINGS_MALFORMED,
                                              "solution")

    # Conservative retained-alternative preflight over every input reading,
    # before the canonical index walks a reading's table.
    for reading in readings:
        count = _retained_alternative_count(reading)
        if count is None:
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_MALFORMED, "endpoint_alternatives")
        if count > max_alternatives:
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_EXHAUSTED, "endpoint_alternatives")

    frames, field = _solution_frame_ids(solution)
    if frames is None or not frames:
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_MALFORMED, field or "frames")
    for frame in frames:
        verdict = _check_frame_id(frame, max_frame_id_bytes)
        if verdict is not None:
            return _solver_input_bindings_refusal(*verdict)
    for reading in readings:
        verdict = _check_frame_id(getattr(reading, "frame", None),
                                  max_frame_id_bytes)
        if verdict is not None:
            return _solver_input_bindings_refusal(*verdict)
    for match in matches:
        if not isinstance(match, plane_graph.PlaneMatch):
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_MALFORMED, "input_match_identity")
        for endpoint_frame in (getattr(match, "left", None),
                               getattr(match, "right", None)):
            verdict = _check_frame_id(endpoint_frame, max_frame_id_bytes)
            if verdict is not None:
                return _solver_input_bindings_refusal(*verdict)

    identity_rows: list[list] = []
    for match in matches:
        try:
            identity_rows.append(_input_match_identity(match))
        except _ERRORS:
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_MALFORMED, "input_match_identity")

    getter = getattr(getattr(solution, "quadrants", None), "get", None)
    if not callable(getter):
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_MALFORMED, "quadrants")
    try:
        frame_quadrants = {frame: getter(frame) for frame in frames}
    except _ERRORS:
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_MALFORMED, "quadrants")
    for quadrant in frame_quadrants.values():
        if type(quadrant) is not int or quadrant not in (0, 1, 2, 3):
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_MALFORMED, "quadrants")

    alternatives = getattr(solution, "alternatives", None)
    if (isinstance(alternatives, (str, bytes))
            or not isinstance(alternatives, Sequence) or not alternatives):
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_UNAVAILABLE, "best_assignment")
    best_quadrants = getattr(alternatives[0], "quadrants", None)
    if (isinstance(best_quadrants, (str, bytes))
            or not isinstance(best_quadrants, Sequence)):
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_MALFORMED, "alternatives[0].quadrants")
    if tuple(best_quadrants) != tuple(frame_quadrants[frame] for frame in frames):
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_UNAVAILABLE, "best_assignment")

    try:
        read_by_index, _index_by_object, _index_frames = \
            plane_graph._canonical_reading_index(readings)
    except _ERRORS:
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_MALFORMED, "readings")
    global_index = {id(reading): position
                    for position, reading in enumerate(readings)}

    table: dict[int, tuple] = {}
    edges: list[tuple] = []
    active_total = 0
    for match_index, match in enumerate(matches):
        try:
            left, right = plane_graph._validate_indexed_match(
                match, read_by_index, require_selected_b=True)
            left_frame, right_frame = match.left, match.right
            if (left_frame not in frame_quadrants
                    or right_frame not in frame_quadrants):
                return _solver_input_bindings_refusal(
                    _SOLVER_INPUT_BINDINGS_MALFORMED, "match_endpoint")
            left_quadrant = frame_quadrants[left_frame]
            right_quadrant = frame_quadrants[right_frame]
            left_world, left_sign = plane_graph.axis_transform(
                left.axis, left_quadrant)
            right_world, right_sign = plane_graph.axis_transform(
                right.axis, right_quadrant)
        except _ERRORS:
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_MALFORMED, "match_endpoint")
        left_index, left_alternative, left_count = _endpoint_alternative(
            left, match.left_b, max_alternatives)
        right_index, right_alternative, right_count = _endpoint_alternative(
            right, match.right_b, max_alternatives)
        if (left_count is None or left_count > max_alternatives
                or right_count is None or right_count > max_alternatives):
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_EXHAUSTED, "endpoint_alternatives")
        if left_alternative is None or right_alternative is None:
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_MALFORMED, "endpoint_alias")
        left_side = _strict_face_side(getattr(left, "side", None))
        right_side = _strict_face_side(getattr(right, "side", None))
        if left_side is None or right_side is None:
            return _solver_input_bindings_refusal(
                _SOLVER_INPUT_BINDINGS_MALFORMED, "endpoint_side")
        relative = match.quadrant_of_right
        active = bool(((right_quadrant - left_quadrant) % 4 == relative)
                      and right_world == left_world)
        if active:
            active_total += 1
        for reading, plane, local_axis, side, index, alternative in (
                (left, match.left_plane, left.axis, left_side, left_index,
                 left_alternative),
                (right, match.right_plane, right.axis, right_side, right_index,
                 right_alternative)):
            position = global_index.get(id(reading))
            if position is None:
                return _solver_input_bindings_refusal(
                    _SOLVER_INPUT_BINDINGS_MALFORMED, "readings")
            if position not in table:
                rows = _reading_alternative_table(reading)
                if rows is None:
                    return _solver_input_bindings_refusal(
                        _SOLVER_INPUT_BINDINGS_MALFORMED, "reading_table")
                if len(rows) > max_alternatives:
                    return _solver_input_bindings_refusal(
                        _SOLVER_INPUT_BINDINGS_EXHAUSTED,
                        "endpoint_alternatives")
                table[position] = (position, reading.frame, int(plane),
                                   int(local_axis), int(side), rows)
        edges.append((
            match_index,
            (global_index[id(left)], left_index, left_alternative[0],
             left_alternative[1], left_alternative[2], int(left_side),
             int(left_sign), int(left.axis)),
            (global_index[id(right)], right_index, right_alternative[0],
             right_alternative[1], right_alternative[2], int(right_side),
             int(right_sign), int(right.axis)),
            int(relative), active))

    reading_rows = tuple(table[position] for position in sorted(table))
    value = SolverInputBindings(
        schema=SOLVER_INPUT_BINDINGS_SCHEMA, recorded=True,
        basis=SOLVER_INPUT_BINDINGS_BASIS, refusal_reason=None,
        failed_assignment_sha256=_failed_assignment_digest(
            frames, frame_quadrants),
        input_match_identity_sha256=_alias_domain_digest(
            SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
            SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA, identity_rows),
        input_match_count=len(matches), active_edge_count=active_total,
        frames=tuple(frames),
        assignments=tuple(frame_quadrants[frame] for frame in frames),
        readings=tuple(tuple(row) for row in reading_rows),
        edges=tuple(tuple(row) for row in edges))
    encoded = json.dumps(value.to_record(), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    if len(encoded) > max_record_bytes:
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_EXHAUSTED, "record_bytes")
    return value


def failed_solver_input_bindings(
        readings: object,
        matches: object,
        solution: plane_graph.ComponentSolution,
) -> "SolverInputBindings":
    """Fail-closed wrapper around the bounded solver-input binding collector."""
    try:
        return _collect_failed_solver_input_bindings(readings, matches, solution)
    except _ERRORS:
        return _solver_input_bindings_refusal(
            _SOLVER_INPUT_BINDINGS_MALFORMED, "record")


@dataclass(frozen=True, slots=True)
class SameAxisCandidateCapture:
    """Bounded evaluation-only capture of the best assignment's pre-filter groups.

    A complete v1 record covers every same-axis group named by a world-axis-1
    active post-filter solver input. For each such group it keeps the *complete*
    pre-filter admitted candidate set -- not only the survivor the solver
    received -- so a dropped ``b`` alternative can be shown to have existed (or
    not) before ``_apply_same_axis_filter``. Each group row names its exact
    ``(left, right, left_axis, relative_quadrant)`` key and binds the group to
    the retained survivor's post-filter identity (its match index plus the shared
    scalar input-match identity projection). Each candidate row binds both
    endpoints to a shared canonical reading table and one retained-alternative
    index, carries the measured ``b``/``distance_blocks``/``support`` that index
    resolves to, the score, PSR/peak/overlap/cell shift, the recomputed
    same-axis ``delta``, and the typed kept/dropped disposition.

    A complete v2 record (``axes=(0, 1, 2)``) generalizes the same capture to
    every world axis: each group row instead binds the complete ordered list of
    its retained post-filter match indices and their exact nine-scalar input
    identities, because one same-axis group can retain more than one compatible
    match. V2 keeps two deliberately different domains: ``input_identities``
    carries the full ordered post-filter identity for *every* input index (so the
    shared digest replays even for inputs no captured group retains), while the
    captured groups and candidate rows stay a bounded subset of the
    solver-active groups. The v2 validator independently re-derives the expected
    active groups from those same identities, frames and assignments (production
    ``axis_transform`` and quadrant relation) and requires the payload's group keys
    and each group's claimed survivor indices to equal the derived active set
    exactly, so an omitted or extra group cannot reseal; it still never requires
    the active subset to cover the whole input. V1's schema, payload and validator
    path are unchanged.

    ``failed_assignment_sha256`` is the same shared failed-best-assignment digest
    the R7/phase/alias records carry and is recomputed by replay.
    ``input_match_identity_sha256`` is the same shared post-filter input-match
    identity digest the R7 record carries, so the two records can be shown to
    describe the same solver input sequence. ``record_sha256`` binds the strict
    UTF-8 canonical JSON payload with explicit domain/schema separation,
    excluding only itself. A refusal is atomic: one bounded reason/field and no
    partial group, candidate or reading row.
    """

    schema: str
    recorded: bool
    basis: str | None
    refusal_reason: str | None
    failed_assignment_sha256: str
    input_match_identity_sha256: str
    input_match_count: int
    group_count: int
    row_count: int
    frames: tuple[str, ...]
    assignments: tuple[int, ...]
    readings: tuple[tuple, ...]
    groups: tuple[tuple, ...]
    candidates: tuple[tuple, ...]
    field: str | None = None
    record_sha256: str = ""
    # DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: the declared ordered world-axis set.
    # Defaulting to the v1 single axis keeps every existing constructor call a
    # byte-identical v1 record, while the explicit all-axis set selects v2.
    axes: tuple[int, ...] = SAME_AXIS_CANDIDATES_AXES_V1
    # DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2-CORRECTION: the full ordered
    # post-filter solver-input identity list. V2 binds *every* input index here,
    # independently of which subset the captured groups retain, so the shared
    # full-input identity digest replays even though only the solver-active
    # groups carry candidate rows. V1 never emits it.
    input_identities: tuple[tuple, ...] = ()
    # V3 keeps the matcher-owned full admitted ledger separate from the selected
    # solver inputs. Historical v1/v2 serialization ignores these defaults.
    matcher_capture: dict | None = None
    admitted_rows: tuple[tuple, ...] = ()
    selected_source_rows: tuple[int, ...] = ()
    selection_sha256: str = ""
    selection_payload: dict | None = None
    admitted_ledger_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_same_axis_candidates_payload(
            _same_axis_candidates_payload(self))
        digest = same_axis_candidate_record_digest(self)
        if self.record_sha256 not in ("", digest):
            raise ValueError("same-axis candidate capture record digest mismatch")
        object.__setattr__(self, "record_sha256", digest)

    def to_record(self) -> dict:
        payload = _same_axis_candidates_payload(self)
        payload["record_sha256"] = self.record_sha256
        return payload


def _same_axis_candidates_refusal(
    reason: str, field: str | None = None,
) -> "SameAxisCandidateCapture":
    return SameAxisCandidateCapture(
        schema=SAME_AXIS_CANDIDATES_SCHEMA, recorded=False, basis=None,
        refusal_reason=reason, failed_assignment_sha256="",
        input_match_identity_sha256="", input_match_count=0, group_count=0,
        row_count=0, frames=(), assignments=(), readings=(), groups=(),
        candidates=(), field=field)


def _same_axis_candidates_payload(value: "SameAxisCandidateCapture") -> dict:
    """Canonical payload of one record, without its own digest field."""
    if not value.recorded:
        return {"schema": value.schema, "status": "refused",
                "reason": value.refusal_reason, "field": value.field}
    if value.schema == SAME_AXIS_CANDIDATES_SCHEMA_V3:
        return {
            "schema": value.schema, "status": "complete", "basis": value.basis,
            "failed_assignment_sha256": value.failed_assignment_sha256,
            "input_match_identity_sha256": value.input_match_identity_sha256,
            "input_match_count": value.input_match_count,
            "frames": list(value.frames), "assignments": list(value.assignments),
            "axes": list(value.axes), "group_count": value.group_count,
            "row_count": value.row_count,
            "input_identities": [list(row) for row in value.input_identities],
            "selected_source_rows": list(value.selected_source_rows),
            "selection_sha256": value.selection_sha256,
            "selection_payload": value.selection_payload,
            "admitted_rows": [list(row) for row in value.admitted_rows],
            "admitted_ledger_sha256": value.admitted_ledger_sha256,
            "matcher_capture": value.matcher_capture,
        }
    axes = tuple(value.axes)
    v2 = axes != SAME_AXIS_CANDIDATES_AXES_V1
    payload = {
        "schema": value.schema,
        "status": "complete",
        "basis": value.basis,
        "failed_assignment_sha256": value.failed_assignment_sha256,
        "input_match_identity_sha256": value.input_match_identity_sha256,
        "input_match_count": value.input_match_count,
        "group_count": value.group_count,
        "row_count": value.row_count,
        "frames": list(value.frames),
        "assignments": list(value.assignments),
        "readings": [list(row) for row in value.readings],
        "groups": [list(row) for row in value.groups],
        "candidates": [list(row) for row in value.candidates],
    }
    if v2:
        payload["axes"] = list(axes)
        payload["input_identities"] = [
            list(identity) for identity in value.input_identities]
    else:
        payload["axis"] = SAME_AXIS_CANDIDATES_AXIS
    return payload


def same_axis_candidate_record_digest(value: "SameAxisCandidateCapture") -> str:
    """Recompute one record's strict digest from its canonical payload."""
    if value.schema == SAME_AXIS_CANDIDATES_SCHEMA_V3:
        domain = SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V3
    else:
        domain = (SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V2
                  if tuple(value.axes) != SAME_AXIS_CANDIDATES_AXES_V1
                  else SAME_AXIS_CANDIDATES_DIGEST_DOMAIN)
    return _sha({
        "domain": domain,
        "schema": value.schema,
        "payload": _same_axis_candidates_payload(value)})


def _same_axis_psr_columns(value: object) -> tuple[float | None, bool]:
    """One admitted match's PSR as ``(finite_value_or_None, is_infinite)``."""
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number) or number == float("-inf"):
        raise ValueError("same-axis candidate psr is not canonical")
    if number == float("inf"):
        return None, True
    return number, False


def _same_axis_frame_id_verdict(
    value: object, max_frame_id_bytes: int,
) -> tuple[str, str] | None:
    """Map the shared frame-id check onto this producer's bounded reason set."""
    verdict = _check_frame_id(value, max_frame_id_bytes)
    if verdict is None:
        return None
    reason, field = verdict
    if reason == _ACTIVE_EDGE_ALIASES_MALFORMED:
        return _SAME_AXIS_CANDIDATES_MALFORMED, field
    return _SAME_AXIS_CANDIDATES_EXHAUSTED, field


def _validate_same_axis_group_row_v1(row: object) -> tuple:
    """Validate one v1 group header and return its canonical fields.

    Layout: ``[group_index, left, right, left_axis, relative_quadrant,
    survivor_match_index, survivor_identity]`` where ``survivor_identity`` is the
    nine-scalar post-filter input identity
    ``[left, left_plane, left_axis, right, right_plane, right_axis, quadrant,
    left_b, right_b]``. The payload validator re-derives the identity from the
    bound reading table and the candidate rows rather than trusting it.
    """
    if type(row) not in (tuple, list) or len(row) != 7:
        raise ValueError("same-axis group row must be a 7-element sequence")
    group_index, left, right, left_axis, relative, survivor_index, identity = row
    if type(group_index) is not int or group_index < 0:
        raise ValueError("same-axis group index must be non-negative")
    if type(left) is not str or not left or type(right) is not str or not right:
        raise ValueError("same-axis group frames must be non-empty strings")
    if type(left_axis) is not int or left_axis not in (0, 1, 2):
        raise ValueError("same-axis group left axis must be canonical")
    if type(relative) is not int or relative not in (0, 1, 2, 3):
        raise ValueError("same-axis group relative quadrant must be canonical")
    if type(survivor_index) is not int or survivor_index < 0:
        raise ValueError("same-axis group survivor index must be non-negative")
    if type(identity) is not list or len(identity) != 9:
        raise ValueError("same-axis group survivor identity must be a list")
    _validate_same_axis_survivor_identity(identity)
    return (group_index, left, right, left_axis, relative, survivor_index,
            tuple(identity))


def _validate_same_axis_group_row_v2(row: object) -> tuple:
    """Validate one v2 group header and return its canonical fields.

    Layout: ``[group_index, left, right, left_axis, relative_quadrant,
    survivor_match_indices, survivor_identities]``. The two trailing lists are
    positionally parallel and non-empty: every retained post-filter match the
    group holds, in match-index order, with its exact nine-scalar input identity.
    One same-axis group can legitimately retain more than one compatible match, so
    the ordered list replaces the v1 scalar survivor.
    """
    if type(row) not in (tuple, list) or len(row) != 7:
        raise ValueError("same-axis group row must be a 7-element sequence")
    group_index, left, right, left_axis, relative, survivors, identities = row
    if type(group_index) is not int or group_index < 0:
        raise ValueError("same-axis group index must be non-negative")
    if type(left) is not str or not left or type(right) is not str or not right:
        raise ValueError("same-axis group frames must be non-empty strings")
    if type(left_axis) is not int or left_axis not in (0, 1, 2):
        raise ValueError("same-axis group left axis must be canonical")
    if type(relative) is not int or relative not in (0, 1, 2, 3):
        raise ValueError("same-axis group relative quadrant must be canonical")
    if (type(survivors) is not list or not survivors
            or any(type(value) is not int or value < 0 for value in survivors)):
        raise ValueError(
            "same-axis group survivor indices must be a non-empty integer list")
    if any(survivors[position] >= survivors[position + 1]
           for position in range(len(survivors) - 1)):
        raise ValueError(
            "same-axis group survivor indices must be strictly increasing")
    if type(identities) is not list or len(identities) != len(survivors):
        raise ValueError(
            "same-axis group survivor identities must pair with the indices")
    canonical = []
    for identity in identities:
        if type(identity) is not list or len(identity) != 9:
            raise ValueError("same-axis group survivor identity must be a list")
        _validate_same_axis_survivor_identity(identity)
        canonical.append(tuple(identity))
    return (group_index, left, right, left_axis, relative, tuple(survivors),
            tuple(canonical))


def _validate_same_axis_survivor_identity(identity: list) -> None:
    """Shared nine-scalar retained-input identity shape check."""
    if (type(identity[0]) is not str or not identity[0]
            or type(identity[1]) is not int or identity[1] < 0
            or type(identity[2]) is not int or identity[2] not in (0, 1, 2)
            or type(identity[3]) is not str or not identity[3]
            or type(identity[4]) is not int or identity[4] < 0
            or type(identity[5]) is not int or identity[5] not in (0, 1, 2)
            or type(identity[6]) is not int or identity[6] not in (0, 1, 2, 3)
            or not _is_finite_number(identity[7]) or identity[7] <= 0.0
            or not _is_finite_number(identity[8]) or identity[8] <= 0.0):
        raise ValueError("same-axis group survivor identity is malformed")


def _validate_same_axis_candidate_row(row: object) -> tuple:
    """Validate one candidate row and return its canonical fields.

    Layout: ``[group_index, match_index, row_index, left_ref, left_alt,
    right_ref, right_alt, relative_quadrant, score, psr_value,
    psr_positive_infinity, peak, second_peak, overlap_cells, shift_row,
    shift_col, delta, disposition, reason]``. ``match_index`` is the candidate's
    post-filter solver-input index, or ``-1`` when the filter dropped it. The
    payload validator re-derives every endpoint, delta and disposition from the
    bound reading table and the group rows.
    """
    if type(row) not in (tuple, list) or len(row) != 19:
        raise ValueError("same-axis candidate row must be a 19-element sequence")
    (group_index, match_index, row_index, left_ref, left_alt, right_ref,
     right_alt, relative, score, psr_value, psr_infinite, peak, second_peak,
     overlap_cells, shift_row, shift_col, delta, disposition, reason) = row
    for name, value in (("group index", group_index), ("row index", row_index),
                        ("left reading", left_ref), ("left alternative", left_alt),
                        ("right reading", right_ref),
                        ("right alternative", right_alt)):
        if type(value) is not int or value < 0:
            raise ValueError(f"same-axis candidate {name} must be non-negative")
    if type(match_index) is not int or match_index < -1:
        raise ValueError("same-axis candidate match index is malformed")
    if type(relative) is not int or relative not in (0, 1, 2, 3):
        raise ValueError("same-axis candidate relative quadrant must be canonical")
    for name, value in (("score", score), ("peak", peak),
                        ("second_peak", second_peak)):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"same-axis candidate {name} must be a finite float")
    if type(psr_infinite) is not bool:
        raise ValueError("same-axis candidate psr tag must be a boolean")
    if psr_infinite:
        if psr_value is not None:
            raise ValueError(
                "same-axis positive-infinity psr must have a null value")
    elif type(psr_value) is not float or not math.isfinite(psr_value):
        raise ValueError("same-axis finite psr must be a finite float")
    if type(overlap_cells) is not int or overlap_cells < 0:
        raise ValueError("same-axis candidate overlap must be non-negative")
    if type(shift_row) is not int or type(shift_col) is not int:
        raise ValueError("same-axis candidate shift must be an integer pair")
    if type(delta) is not float or not math.isfinite(delta):
        raise ValueError("same-axis candidate delta must be a finite float")
    if disposition not in ("kept", "dropped"):
        raise ValueError("same-axis candidate disposition is unknown")
    if type(reason) is not str:
        raise ValueError("same-axis candidate reason must be a string")
    return (group_index, match_index, row_index, left_ref, left_alt, right_ref,
            right_alt, relative, score, psr_value, psr_infinite, peak,
            second_peak, overlap_cells, shift_row, shift_col, delta,
            disposition, reason)


def _validate_complete_same_axis_candidates_payload(payload: dict) -> None:
    """Re-derive a v1 record's every group, endpoint, delta and disposition."""
    frame_quadrant, input_match_count, group_count, row_count, table = \
        _validate_same_axis_capture_header(payload)
    if payload["axis"] != SAME_AXIS_CANDIDATES_AXIS:
        raise ValueError("same-axis candidate capture must bind world axis 1")
    group_state = _validate_same_axis_capture_groups(
        payload, input_match_count, group_count, v2=False)
    _validate_same_axis_capture_candidate_rows(
        payload, table, frame_quadrant, input_match_count, row_count,
        group_state, SAME_AXIS_CANDIDATES_AXES_V1)

    kept_indices: set[int] = set()
    for group in group_state.values():
        rows = group["rows"]
        dropped: set[int] = set()
        for first in range(len(rows)):
            for second in range(first + 1, len(rows)):
                first_delta = rows[first]["delta"]
                second_delta = rows[second]["delta"]
                if abs(first_delta - second_delta) <= plane_graph.SAME_AXIS_TOLERANCE:
                    continue
                if abs(rows[first]["score"] - rows[second]["score"]) <= 1e-12:
                    dropped.add(first)
                    dropped.add(second)
                elif rows[first]["score"] > rows[second]["score"]:
                    dropped.add(second)
                else:
                    dropped.add(first)
        for position, row in enumerate(rows):
            expected = "dropped" if position in dropped else "kept"
            if row["disposition"] != expected:
                raise ValueError(
                    "same-axis candidate capture disposition does not replay")
            if expected == "kept":
                if row["match_index"] in kept_indices:
                    raise ValueError(
                        "same-axis candidate capture repeats a kept match index")
                kept_indices.add(row["match_index"])
        survivors = [row for row in rows
                     if row["match_index"] == group["survivor"]]
        if len(survivors) != 1 or survivors[0]["disposition"] != "kept":
            raise ValueError(
                "same-axis candidate capture group has no unique survivor")
        survivor = survivors[0]
        left_row = table[survivor["left_ref"]]
        right_row = table[survivor["right_ref"]]
        expected_identity = (
            left_row[1], int(left_row[2]), int(left_row[3]),
            right_row[1], int(right_row[2]), int(right_row[3]),
            int(group["key"][3]),
            float(left_row[5][survivor["left_alt"]][0]),
            float(right_row[5][survivor["right_alt"]][0]))
        if group["identity"] != expected_identity:
            raise ValueError(
                "same-axis candidate capture survivor identity does not replay")
        if (group["key"][0], group["key"][1], group["key"][2]) != (
                left_row[1], right_row[1], int(left_row[3])):
            raise ValueError(
                "same-axis candidate capture group key contradicts its survivor")


def _validate_same_axis_capture_header(
        payload: dict,
) -> tuple[dict[str, int], int, int, int, dict[int, tuple]]:
    """Shared complete-record header: identity digests, counts and reading table."""
    if not _hex(payload["failed_assignment_sha256"]):
        raise ValueError("same-axis candidate capture assignment digest must be hex")
    if not _hex(payload["input_match_identity_sha256"]):
        raise ValueError("same-axis candidate capture identity digest must be hex")
    frames = payload["frames"]
    if (type(frames) is not list or not frames
            or any(type(frame) is not str or not frame for frame in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError("same-axis candidate capture frames must be unique")
    assignments = payload["assignments"]
    if (type(assignments) is not list or len(assignments) != len(frames)
            or any(type(value) is not int or value not in (0, 1, 2, 3)
                   for value in assignments)):
        raise ValueError("same-axis candidate capture assignments must be canonical")
    frame_quadrant = dict(zip(frames, assignments))
    if payload["failed_assignment_sha256"] != _failed_assignment_digest(
            frames, frame_quadrant):
        raise ValueError(
            "same-axis candidate capture assignment digest does not replay")
    input_match_count = payload["input_match_count"]
    if (type(input_match_count) is not int or input_match_count < 0
            or input_match_count > SAME_AXIS_CANDIDATES_MAX_INPUT_MATCHES):
        raise ValueError("same-axis candidate capture input count is out of bounds")
    group_count = payload["group_count"]
    if (type(group_count) is not int or group_count < 0
            or group_count > SAME_AXIS_CANDIDATES_MAX_GROUPS):
        raise ValueError("same-axis candidate capture group count is out of bounds")
    row_count = payload["row_count"]
    if (type(row_count) is not int or row_count < 0
            or row_count > SAME_AXIS_CANDIDATES_MAX_ROWS):
        raise ValueError("same-axis candidate capture row count is out of bounds")

    readings = payload["readings"]
    if (type(readings) is not list
            or len(readings) > SAME_AXIS_CANDIDATES_MAX_READINGS):
        raise ValueError("same-axis candidate capture reading table is over the cap")
    table: dict[int, tuple] = {}
    seen_canonical: set[tuple[str, int]] = set()
    previous_reading = -1
    for row in readings:
        reading_index = _validate_solver_input_reading_row(row)
        if reading_index <= previous_reading:
            raise ValueError("same-axis candidate capture readings must be ordered")
        previous_reading = reading_index
        canonical = (row[1], int(row[2]))
        if canonical in seen_canonical:
            raise ValueError("same-axis candidate capture readings repeat an index")
        seen_canonical.add(canonical)
        table[reading_index] = tuple(row)
    return frame_quadrant, input_match_count, group_count, row_count, table


def _validate_same_axis_capture_groups(
    payload: dict, input_match_count: int, group_count: int, *, v2: bool,
) -> dict[int, dict]:
    """Shared group-header walk; v2 binds an ordered list of retained matches."""
    groups = payload["groups"]
    if type(groups) is not list or len(groups) != group_count:
        raise ValueError("same-axis candidate capture groups do not match the count")
    group_keys: set[tuple] = set()
    group_state: dict[int, dict] = {}
    for row in groups:
        if v2:
            (group_index, left, right, left_axis, relative, survivors,
             identities) = _validate_same_axis_group_row_v2(row)
        else:
            (group_index, left, right, left_axis, relative, survivor,
             identity) = _validate_same_axis_group_row_v1(row)
        if group_index != len(group_state):
            raise ValueError(
                "same-axis candidate capture group indices must be contiguous")
        key = (left, right, left_axis, relative)
        if key in group_keys:
            raise ValueError("same-axis candidate capture repeats a group key")
        group_keys.add(key)
        if v2:
            if survivors[-1] >= input_match_count:
                raise ValueError(
                    "same-axis candidate capture survivor is out of range")
            group_state[group_index] = {"key": key, "claim": tuple(survivors),
                                        "identity": tuple(identities), "rows": []}
        else:
            if survivor >= input_match_count:
                raise ValueError(
                    "same-axis candidate capture survivor is out of range")
            group_state[group_index] = {"key": key, "survivor": survivor,
                                        "identity": tuple(identity), "rows": []}
    return group_state


def _validate_same_axis_capture_candidate_rows(
    payload: dict, table: dict[int, tuple], frame_quadrant: dict[str, int],
    input_match_count: int, row_count: int, group_state: dict[int, dict],
    axes: tuple[int, ...],
) -> dict:
    """Shared candidate-row re-derivation: endpoints, axes, delta, disposition."""
    candidates = payload["candidates"]
    if type(candidates) is not list or len(candidates) != row_count:
        raise ValueError(
            "same-axis candidate capture candidates do not match the count")
    identity_by_index: dict[int, tuple[float, ...]] = {}
    for row in candidates:
        (group_index, match_index, row_index, left_ref, left_alt, right_ref,
         right_alt, relative, score, psr_value, psr_infinite, peak, second_peak,
         overlap_cells, shift_row, shift_col, delta, disposition,
         reason) = _validate_same_axis_candidate_row(row)
        state = group_state.get(group_index)
        if state is None:
            raise ValueError("same-axis candidate capture row names a missing group")
        if row_index != len(state["rows"]):
            raise ValueError(
                "same-axis candidate capture row indices must be contiguous")
        if relative != state["key"][3]:
            raise ValueError(
                "same-axis candidate capture row contradicts its group")
        left_row = table.get(left_ref)
        right_row = table.get(right_ref)
        if left_row is None or right_row is None:
            raise ValueError(
                "same-axis candidate capture endpoint names a missing reading")
        if left_alt >= len(left_row[5]) or right_alt >= len(right_row[5]):
            raise ValueError(
                "same-axis candidate capture alternative index is out of range")
        left_axis_local = int(left_row[3])
        right_axis_local = int(right_row[3])
        mapped_axis, sigma = plane_graph.axis_transform(right_axis_local, relative)
        if mapped_axis != left_axis_local:
            raise ValueError("same-axis candidate capture endpoint axes disagree")
        left_quadrant = frame_quadrant.get(left_row[1])
        right_quadrant = frame_quadrant.get(right_row[1])
        if left_quadrant is None or right_quadrant is None:
            raise ValueError(
                "same-axis candidate capture reading frame is not in the assignment")
        left_world, _left_sign = plane_graph.axis_transform(
            left_axis_local, left_quadrant)
        right_world, _right_sign = plane_graph.axis_transform(
            right_axis_local, right_quadrant)
        if left_world not in axes or right_world != left_world:
            raise ValueError("same-axis candidate capture row is not active")
        if (right_quadrant - left_quadrant) % 4 != relative:
            raise ValueError(
                "same-axis candidate capture row contradicts the assignment")
        left_distance = float(left_row[5][left_alt][1])
        right_distance = float(right_row[5][right_alt][1])
        expected_delta = float(
            float(left_row[4]) * left_distance
            - float(sigma) * float(right_row[4]) * right_distance)
        if delta != expected_delta:
            raise ValueError("same-axis candidate capture delta does not replay")
        if disposition == "kept":
            if match_index < 0 or match_index >= input_match_count:
                raise ValueError(
                    "same-axis candidate capture kept row needs a match index")
            if reason != "":
                raise ValueError(
                    "same-axis candidate capture kept row has a reason")
        else:
            if match_index != -1:
                raise ValueError(
                    "same-axis candidate capture dropped row carries an index")
            if reason != plane_graph.SAME_AXIS_DISPOSITION_DROPPED:
                raise ValueError(
                    "same-axis candidate capture dropped row has no reason")
        state["rows"].append({
            "match_index": match_index, "score": score, "delta": delta,
            "disposition": disposition, "reason": reason,
            "left_ref": left_ref, "left_alt": left_alt,
            "right_ref": right_ref, "right_alt": right_alt})
        if disposition == "kept":
            identity_by_index[match_index] = (
                left_row[1], int(left_row[2]), int(left_row[3]),
                right_row[1], int(right_row[2]), int(right_row[3]),
                int(relative),
                float(left_row[5][left_alt][0]),
                float(right_row[5][right_alt][0]))
    bound_readings = {ref for state in group_state.values()
                      for row in state["rows"]
                      for ref in (row["left_ref"], row["right_ref"])}
    if set(table) != bound_readings:
        raise ValueError(
            "same-axis candidate capture reading table is not the exact set")
    return {"identity_by_index": identity_by_index}


def _validate_same_axis_active_group_coverage(
    input_identities: list[tuple], frame_quadrant: dict[str, int],
    group_state: dict[int, dict],
) -> None:
    """V2-only: prove the captured groups are exactly the solver-active ones.

    V2 always captures every world axis, so the payload's ``axes`` is required to
    be exactly ``(0, 1, 2)``. The expected active groups are re-derived
    *independently* from the record's own full ordered ``input_identities`` list
    plus its frames and assignments, using the production relation
    ``(q_right - q_left) % 4 == relative`` and
    ``axis_transform(left_axis, q_left) == axis_transform(right_axis, q_right)``.
    A group is expected exactly when both endpoints map to the same world axis;
    that world axis is necessarily inside the required axis set.

    The payload's group keys must equal that derived key set exactly, and each
    group's claimed ordered survivor indices must equal the derived active input
    indices for its key exactly. Indices outside the active set are intentionally
    never required to have candidate rows; the 28 non-captured inputs of a
    38-input capture stay a silent complement. An identity row that names a frame
    absent from the record's assignments fails closed as a foreign frame, so an
    omitted active group cannot be hidden by rewriting its identities.
    """
    active_groups: dict[tuple, list[int]] = {}
    for index, identity in enumerate(input_identities):
        left_frame, left_axis, right_frame, right_axis, relative = (
            identity[0], int(identity[2]), identity[3], int(identity[5]),
            int(identity[6]))
        left_quadrant = frame_quadrant.get(left_frame)
        right_quadrant = frame_quadrant.get(right_frame)
        if left_quadrant is None or right_quadrant is None:
            raise ValueError(
                "same-axis candidate capture input identity names a foreign frame")
        left_world, _left_sign = plane_graph.axis_transform(
            left_axis, left_quadrant)
        right_world, _right_sign = plane_graph.axis_transform(
            right_axis, right_quadrant)
        if ((right_quadrant - left_quadrant) % 4 == relative
                and left_world == right_world):
            active_groups.setdefault(
                (left_frame, right_frame, left_axis, relative), []).append(index)
    payload_groups = {group["key"]: group for group in group_state.values()}
    if set(payload_groups) != set(active_groups):
        raise ValueError(
            "same-axis candidate capture does not cover the active group set")
    for key, group in payload_groups.items():
        if tuple(group["claim"]) != tuple(active_groups[key]):
            raise ValueError(
                "same-axis candidate capture active survivors do not replay")


def _validate_complete_same_axis_candidates_payload_v2(payload: dict) -> None:
    """Re-derive a v2 record's every group, retained index, axis and identity."""
    frame_quadrant, input_match_count, group_count, row_count, table = \
        _validate_same_axis_capture_header(payload)
    if payload["basis"] != SAME_AXIS_CANDIDATES_BASIS_V2:
        raise ValueError("same-axis candidate capture has an unknown basis")
    axes = payload["axes"]
    if (type(axes) is not list
            or any(type(value) is not int or value not in (0, 1, 2)
                   for value in axes)
            or axes != list(SAME_AXIS_CANDIDATES_AXES_ALL)):
        raise ValueError(
            "same-axis candidate capture axes must bind every world axis")
    axis_set = tuple(axes)
    input_identities = _validate_same_axis_input_identities(
        payload, input_match_count)
    group_state = _validate_same_axis_capture_groups(
        payload, input_match_count, group_count, v2=True)
    _validate_same_axis_capture_candidate_rows(
        payload, table, frame_quadrant, input_match_count, row_count,
        group_state, axis_set)
    _validate_same_axis_active_group_coverage(
        input_identities, frame_quadrant, group_state)

    covered: set[int] = set()
    for group_index, group in group_state.items():
        rows = group["rows"]
        dropped: set[int] = set()
        for first in range(len(rows)):
            for second in range(first + 1, len(rows)):
                if abs(rows[first]["delta"] - rows[second]["delta"]) <= (
                        plane_graph.SAME_AXIS_TOLERANCE):
                    continue
                if abs(rows[first]["score"] - rows[second]["score"]) <= 1e-12:
                    dropped.add(first)
                    dropped.add(second)
                elif rows[first]["score"] > rows[second]["score"]:
                    dropped.add(second)
                else:
                    dropped.add(first)
        for position, row in enumerate(rows):
            expected = "dropped" if position in dropped else "kept"
            if row["disposition"] != expected:
                raise ValueError(
                    "same-axis candidate capture disposition does not replay")
        kept_rows = [row for row in rows if row["disposition"] == "kept"]
        kept = [row["match_index"] for row in kept_rows]
        if kept != list(group["claim"]):
            raise ValueError(
                "same-axis candidate capture group retention does not replay")
        for match_index in kept:
            if match_index in covered:
                raise ValueError(
                    "same-axis candidate capture repeats a retained match index")
            covered.add(match_index)
        for match_index, identity, row in zip(
                group["claim"], group["identity"], kept_rows):
            left_row = table[row["left_ref"]]
            right_row = table[row["right_ref"]]
            expected_identity = (
                left_row[1], int(left_row[2]), int(left_row[3]),
                right_row[1], int(right_row[2]), int(right_row[3]),
                int(group["key"][3]),
                float(left_row[5][row["left_alt"]][0]),
                float(right_row[5][row["right_alt"]][0]))
            if tuple(identity) != expected_identity:
                raise ValueError(
                    "same-axis candidate capture survivor identity does not replay")
            if tuple(identity) != tuple(input_identities[match_index]):
                raise ValueError(
                    "same-axis candidate capture survivor identity does not "
                    "match the input identity")
            if (group["key"][0], group["key"][1], group["key"][2]) != (
                    left_row[1], right_row[1], int(left_row[3])):
                raise ValueError(
                    "same-axis candidate capture group key contradicts its survivor")
    _validate_same_axis_capture_input_identities(
        payload, input_match_count, input_identities)


def _validate_same_axis_input_identities(
    payload: dict, input_match_count: int,
) -> list[tuple]:
    """v2-only strict shape/order check of every post-filter input identity.

    The canonical list is positional: row ``index`` is the exact nine-scalar
    identity of solver input ``index``. Its length must equal the declared input
    count, so an abridged or reordered list is rejected here, and the shared
    full-input identity digest is replayed from this list alone -- never
    reconstructed from the captured survivors, which are deliberately a proper
    subset of the input indices.
    """
    rows = payload["input_identities"]
    if type(rows) is not list or len(rows) != input_match_count:
        raise ValueError(
            "same-axis candidate capture input identities do not match the "
            "input count")
    canonical: list[tuple] = []
    for identity in rows:
        if type(identity) is not list or len(identity) != 9:
            raise ValueError(
                "same-axis candidate capture input identity must be a list")
        _validate_same_axis_survivor_identity(identity)
        canonical.append(tuple(identity))
    return canonical


def _validate_same_axis_capture_input_identities(
    payload: dict, input_match_count: int, input_identities: list[tuple],
) -> None:
    """v2-only full-input identity digest replay from the canonical list.

    This replaces the rejected v2 rule that reconstructed the full-input digest
    from the retained group survivors: those survivors are only the
    solver-active subset, so the digest must replay from the complete declared
    identity list instead.
    """
    if len(input_identities) != input_match_count:
        raise ValueError(
            "same-axis candidate capture input identities are incomplete")
    identity_rows = [list(identity) for identity in input_identities]
    if payload["input_match_identity_sha256"] != _alias_domain_digest(
            SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
            SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA, identity_rows):
        raise ValueError(
            "same-axis candidate capture identity digest does not replay")


def _validate_refused_same_axis_candidates_payload(payload: dict) -> None:
    if set(payload) != _SAME_AXIS_CANDIDATES_REFUSAL_FIELDS:
        raise ValueError("refused same-axis candidate capture has unexpected fields")
    if payload["reason"] not in (
            _SAME_AXIS_CANDIDATES_MALFORMED, _SAME_AXIS_CANDIDATES_UNAVAILABLE,
            _SAME_AXIS_CANDIDATES_EXHAUSTED):
        raise ValueError("refused same-axis candidate capture has an unknown reason")
    field = payload["field"]
    if field is None:
        return
    if type(field) is not str or field not in _SAME_AXIS_CANDIDATES_FIELD_CODES:
        raise ValueError(
            "refused same-axis candidate capture field is not a known code")


def _validate_same_axis_candidates_payload(payload: object) -> None:
    if type(payload) is not dict:
        raise ValueError("same-axis candidate capture schema is malformed")
    schema = payload.get("schema")
    if schema == SAME_AXIS_CANDIDATES_SCHEMA:
        if payload.get("status") == "refused":
            _validate_refused_same_axis_candidates_payload(payload)
        elif payload.get("status") == "complete":
            if set(payload) != _SAME_AXIS_CANDIDATES_COMPLETE_FIELDS:
                raise ValueError(
                    "complete same-axis candidate capture has unexpected fields")
            if payload["basis"] != SAME_AXIS_CANDIDATES_BASIS:
                raise ValueError(
                    "same-axis candidate capture has an unknown basis")
            _validate_complete_same_axis_candidates_payload(payload)
        else:
            raise ValueError("same-axis candidate capture status is unknown")
        return
    if schema == SAME_AXIS_CANDIDATES_SCHEMA_V2:
        if payload.get("status") == "refused":
            _validate_refused_same_axis_candidates_payload(payload)
        elif payload.get("status") == "complete":
            if set(payload) != _SAME_AXIS_CANDIDATES_COMPLETE_FIELDS_V2:
                raise ValueError(
                    "complete same-axis candidate capture has unexpected fields")
            _validate_complete_same_axis_candidates_payload_v2(payload)
        else:
            raise ValueError("same-axis candidate capture status is unknown")
        return
    if schema == SAME_AXIS_CANDIDATES_SCHEMA_V3:
        if payload.get("status") == "refused":
            _validate_refused_same_axis_candidates_payload(payload)
        elif payload.get("status") == "complete":
            _validate_joint_same_axis_payload(payload)
        else:
            raise ValueError("same-axis candidate capture status is unknown")
        return
    raise ValueError("same-axis candidate capture schema is malformed")


def _matcher_payload_match(payload: object) -> plane_graph.PlaneMatch:
    """Strictly replay a serialized matcher row, including its exact hash shape."""
    if type(payload) is not dict or set(payload) != {
            "left", "right", "left_axis", "right_axis", "left_b", "right_b",
            "quadrant", "psr", "peak", "second_peak", "overlap_cells",
            "score", "left_plane", "right_plane", "cell_shift_rc"}:
        raise ValueError("joint matcher row shape is malformed")
    psr = payload["psr"]
    if psr == "positive_infinity":
        psr = float("inf")
    elif type(psr) is not float or not math.isfinite(psr):
        raise ValueError("joint matcher psr is malformed")
    shift = payload["cell_shift_rc"]
    if type(shift) is not list or len(shift) != 2:
        raise ValueError("joint matcher shift is malformed")
    match = plane_graph.PlaneMatch(
        payload["left"], payload["right"], payload["left_axis"],
        payload["right_axis"], payload["left_b"], payload["right_b"],
        payload["quadrant"], psr, payload["peak"], payload["second_peak"],
        payload["overlap_cells"], None, payload["score"],
        payload["left_plane"], payload["right_plane"], tuple(shift))
    if plane_graph._indexed_match_payload(match) != payload:
        raise ValueError("joint matcher row is not canonical")
    return match


def _joint_candidate_payload(row: list, table: dict[int, list],
                             group: list) -> dict:
    left = table[row[3]]
    right = table[row[5]]
    return {
        "left": left[1], "right": right[1],
        "left_axis": left[3], "right_axis": right[3],
        "left_b": left[5][row[4]][0], "right_b": right[5][row[6]][0],
        "quadrant": row[7],
        "psr": "positive_infinity" if row[10] else row[9],
        "peak": row[11], "second_peak": row[12], "overlap_cells": row[13],
        "score": row[8], "left_plane": left[2], "right_plane": right[2],
        "cell_shift_rc": [row[14], row[15]],
    }


def _joint_representative_match(row: object) -> plane_graph.PlaneMatch:
    """Rebuild the full 16-field certificate match, without losing score/shift."""
    if type(row) not in (list, tuple) or len(row) != 16:
        raise ValueError("joint representative match shape differs")
    shift = row[15]
    if shift is not None and (type(shift) not in (list, tuple) or len(shift) != 2):
        raise ValueError("joint representative shift differs")
    match = plane_graph.PlaneMatch(
        row[0], row[1], row[2], row[3], row[4], row[5], row[6],
        _psr_from_payload(row[7]), row[8], row[9], row[10], row[11],
        row[12], row[13], row[14], None if shift is None else tuple(shift))
    if _match_payload(match) != list(row):
        raise ValueError("joint representative match is not canonical")
    return match


def _joint_certificate_from_payload(data: object):
    """Replay the complete typed selector proof from serialized certificate data."""
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection_mod
    from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
    if (type(data) is not dict
            or set(data) != {field.name for field in dataclasses.fields(
                selection_mod.EndpointSelectionCertificate)
                if field.name != "result_sha256"}):
        raise ValueError("joint certificate fields differ")
    raw_bindings = data["reading_bindings"]
    if type(raw_bindings) not in (list, tuple):
        raise ValueError("joint certificate reading bindings differ")
    bindings = tuple((tuple(row[0]), row[1], row[2], row[3],
                      tuple(tuple(alt) for alt in row[4]), tuple(row[5]))
                     for row in raw_bindings)
    representatives = tuple(selection_mod.RepresentativeConstraint(
        row[0], row[1], _joint_representative_match(row[2]))
        for row in data["representatives"])
    values = dict(data)
    values["frame_ids"] = tuple(data["frame_ids"])
    values["quadrant_constraints"] = tuple(tuple(row) for row in data["quadrant_constraints"])
    values["budget"] = csp.EndpointBudget(**data["budget"])
    values["reading_bindings"] = bindings
    values["representatives"] = representatives
    cert = selection_mod.EndpointSelectionCertificate(**values)
    if _sha(selection_mod._payload(cert)) != _sha(data):
        raise ValueError("joint certificate canonical replay differs")
    return cert


def _validate_joint_same_axis_payload(payload: dict) -> None:
    fields = {"schema", "status", "basis", "failed_assignment_sha256",
              "input_match_identity_sha256", "input_match_count", "frames",
              "assignments", "axes", "group_count", "row_count",
              "input_identities", "selected_source_rows", "selection_sha256",
              "selection_payload",
              "admitted_rows", "admitted_ledger_sha256", "matcher_capture"}
    if set(payload) != fields or payload["basis"] != SAME_AXIS_CANDIDATES_BASIS_V3:
        raise ValueError("joint same-axis capture has unexpected fields")
    if payload["axes"] != list(SAME_AXIS_CANDIDATES_AXES_ALL):
        raise ValueError("joint same-axis capture must cover all axes")
    if not _hex(payload["selection_sha256"]):
        raise ValueError("joint selection digest is malformed")
    selection_payload = payload["selection_payload"]
    if (type(selection_payload) is not dict
            or _sha(selection_payload) != payload["selection_sha256"]
            or selection_payload.get("status") != "unique"):
        raise ValueError("joint selection certificate digest differs")
    matcher = payload["matcher_capture"]
    if (type(matcher) is not dict
            or matcher.get("schema") != SAME_AXIS_CANDIDATES_SCHEMA_V2
            or matcher.get("status") != "complete"):
        raise ValueError("joint matcher capture must be complete v2")
    validate_same_axis_candidates(matcher)
    cert = _joint_certificate_from_payload(selection_payload)
    if cert.result_sha256 != payload["selection_sha256"]:
        raise ValueError("joint typed certificate digest differs")
    if (list(cert.frame_ids) != payload["frames"]
            or cert.proof_record()["choices"][0]["quadrants"] !=
            [[frame, quadrant] for frame, quadrant in zip(
                payload["frames"], payload["assignments"])]):
        raise ValueError("joint certificate assignment differs from failure")
    for name in ("frames", "assignments", "axes", "group_count", "row_count",
                 "failed_assignment_sha256"):
        if payload[name] != matcher[name]:
            raise ValueError("joint matcher capture assignment/group binding differs")
    admitted = payload["admitted_rows"]
    if type(admitted) is not list or len(admitted) > SAME_AXIS_CANDIDATES_MAX_ROWS:
        raise ValueError("joint admitted ledger exceeds the row cap")
    if (not _hex(payload["admitted_ledger_sha256"])
            or payload["admitted_ledger_sha256"] !=
            plane_graph.admitted_candidate_ledger_sha256(
                (row[0], row[1], row[2], row[4]) for row in admitted)):
        raise ValueError("joint admitted ledger commitment differs")
    kept: list[tuple[int, plane_graph.PlaneMatch]] = []
    candidates: list[plane_graph.PlaneMatch] = []
    for index, row in enumerate(admitted):
        if type(row) is not list or len(row) != 5 or row[0] != index:
            raise ValueError("joint admitted rows must be contiguous")
        _, disposition, post_index, match_payload, match_sha = row
        match = _matcher_payload_match(match_payload)
        if type(match_sha) is not str or _sha(match_payload) != match_sha:
            raise ValueError("joint admitted match digest differs")
        if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT:
            if type(post_index) is not int or post_index != len(kept):
                raise ValueError("joint kept rows must bind the matcher order")
            kept.append((index, match))
        elif disposition == plane_graph.SAME_AXIS_DISPOSITION_DROPPED:
            if post_index is not None:
                raise ValueError("joint dropped row claims a matcher index")
        else:
            raise ValueError("joint admitted disposition is malformed")
        candidates.append(match)
    if candidates != sorted(candidates, key=plane_graph._match_order):
        raise ValueError("joint admitted rows are not matcher ordered")
    if len(kept) != matcher["input_match_count"]:
        raise ValueError("joint kept count differs from matcher capture")
    if [_input_match_identity(match) for _, match in kept] != matcher["input_identities"]:
        raise ValueError("joint kept identities differ from matcher capture")
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection_mod
    if (selection_payload.get("candidate_count") != len(kept)
            or selection_payload.get("candidate_matches_sha256") !=
            selection_mod.candidate_digest(tuple(match for _, match in kept))):
        raise ValueError("joint selection candidate digest differs")
    # Every candidate in every active matcher group is materialized once in the
    # v2 capture; the independent ledger supplies the expected source rows.
    table = {row[0]: row for row in matcher["readings"]}
    groups = {tuple(row[1:5]): row for row in matcher["groups"]}
    expected: dict[tuple, list[dict]] = {key: [] for key in groups}
    for match in candidates:
        key = (match.left, match.right, match.left_axis,
               match.quadrant_of_right)
        if key in expected:
            expected[key].append(plane_graph._indexed_match_payload(match))
    actual: dict[tuple, list[dict]] = {key: [] for key in groups}
    for row in matcher["candidates"]:
        key = tuple(matcher["groups"][row[0]][1:5])
        actual[key].append(_joint_candidate_payload(row, table, groups[key]))
    if actual != expected:
        raise ValueError("joint active candidate rows omit or substitute matcher rows")
    selected_rows = payload["selected_source_rows"]
    identities = payload["input_identities"]
    count = payload["input_match_count"]
    if (type(count) is not int or count < 0 or count > SAME_AXIS_CANDIDATES_MAX_INPUT_MATCHES
            or type(selected_rows) is not list or len(selected_rows) != count
            or len(set(selected_rows)) != count
            or type(identities) is not list or len(identities) != count):
        raise ValueError("joint selected source rows are incomplete or repeated")
    for source_row, identity in zip(selected_rows, identities):
        if type(source_row) is not int or source_row < 0 or source_row >= len(kept):
            raise ValueError("joint selected source row is foreign")
        if type(identity) is not list or len(identity) != 9:
            raise ValueError("joint selected identity is malformed")
        _validate_same_axis_survivor_identity(identity)
        if identity[:7] != _input_match_identity(kept[source_row][1])[:7]:
            raise ValueError("joint selected identity changed its source endpoint")
    representatives = cert.representatives
    if len(representatives) != count:
        raise ValueError("joint certificate representative count differs")
    for index, (representative, source_row, identity) in enumerate(zip(
            representatives, selected_rows, identities)):
        if (representative.factor_index != index
                or representative.source_row != source_row):
            raise ValueError("joint certificate source row/order differs")
        selected = representative.match
        if _input_match_identity(selected) != identity:
            raise ValueError("joint certificate selected identity differs")
        original = kept[source_row][1]
        if dataclasses.replace(original, left_b=selected.left_b,
                               right_b=selected.right_b) != selected:
            raise ValueError("joint certificate representative changed source evidence")
    _validate_same_axis_capture_input_identities(payload, count,
                                                [tuple(row) for row in identities])


def validate_same_axis_candidates(record: object) -> None:
    """Public fail-closed replay for one serialized candidate-capture record.

    A matching digest is necessary but not sufficient: every group key, survivor
    identity, endpoint alternative, measured distance, delta, disposition and
    count is independently re-derived from the rows and the shared assignment
    digest, and the complete strict JSON encoding must stay inside the fixed byte
    budget. The v1 single-axis record and the v2 all-axis record dispatch on the
    declared schema; the v2 path additionally requires the declared axes to be
    exactly ``(0, 1, 2)``, re-derives the expected active group keys and their
    active input indices from the record's own ``input_identities`` list, and
    requires the captured groups and their claimed survivor indices to equal that
    derived active set exactly before cross-binding the identities and replaying
    the full ordered input identity digest from the complete list rather than from
    the bounded captured groups.
    """
    if type(record) is not dict:
        raise ValueError("same-axis candidate capture must be a mapping")
    body = dict(record)
    claimed = body.pop("record_sha256", None)
    if not _hex(claimed):
        raise ValueError("same-axis candidate capture digest is missing")
    _validate_same_axis_candidates_payload(body)
    domain = (SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V3
              if body.get("schema") == SAME_AXIS_CANDIDATES_SCHEMA_V3
              else SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V2
              if body.get("schema") == SAME_AXIS_CANDIDATES_SCHEMA_V2
              else SAME_AXIS_CANDIDATES_DIGEST_DOMAIN)
    computed = _sha({"domain": domain,
                     "schema": body.get("schema"), "payload": body})
    if computed != claimed:
        raise ValueError("same-axis candidate capture digest mismatch")
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > SAME_AXIS_CANDIDATES_MAX_RECORD_BYTES:
        raise ValueError("same-axis candidate capture exceeds the fixed byte budget")


def _collect_same_axis_candidates(
    readings: object,
    matches: object,
    solution: plane_graph.ComponentSolution,
    observation: object,
    axes: object = SAME_AXIS_CANDIDATES_AXES_V1,
) -> "SameAxisCandidateCapture":
    """Capture the bound world-axis active groups' complete pre-filter set.

    ``axes`` selects the frozen v1 single-axis record (the default, world axis 1)
    or the evaluation-only v2 all-axis record (``(0, 1, 2)``). Both read the same
    raw matcher observation plus the same local readings, post-filter matches and
    returned solution the graph seam already holds: no matcher, filter, solver,
    least-squares or signature pass is repeated. Each captured group is bound to
    the complete ordered list of its retained post-filter match indices; the v1
    record keeps its single-survivor binding. The three named caps (16 groups,
    1,024 rows, 262,144 encoded bytes) and the structural reading/alternative/frame
    caps are enforced before or while materializing the record, and any malformed
    value becomes one atomic typed refusal instead of a partial row.
    """
    max_groups = SAME_AXIS_CANDIDATES_MAX_GROUPS
    max_rows = SAME_AXIS_CANDIDATES_MAX_ROWS
    max_readings = SAME_AXIS_CANDIDATES_MAX_READINGS
    max_alternatives = SAME_AXIS_CANDIDATES_MAX_ALTERNATIVES
    max_frame_id_bytes = SAME_AXIS_CANDIDATES_MAX_FRAME_ID_BYTES
    max_input_matches = SAME_AXIS_CANDIDATES_MAX_INPUT_MATCHES
    max_record_bytes = SAME_AXIS_CANDIDATES_MAX_RECORD_BYTES

    def refuse(reason: str, field: str | None):
        return _same_axis_candidates_refusal(reason, field)

    if not isinstance(observation, plane_graph.SameAxisCandidateObservation):
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "observation")
    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "matches")
    if len(matches) > max_input_matches:
        return refuse(_SAME_AXIS_CANDIDATES_EXHAUSTED, "matches")
    if isinstance(readings, (str, bytes)) or not isinstance(readings, Sequence):
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "readings")
    if len(readings) > max_readings:
        return refuse(_SAME_AXIS_CANDIDATES_EXHAUSTED, "readings")
    if not isinstance(solution, plane_graph.ComponentSolution):
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "solution")

    for reading in readings:
        count = _retained_alternative_count(reading)
        if count is None:
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "endpoint_alternatives")
        if count > max_alternatives:
            return refuse(_SAME_AXIS_CANDIDATES_EXHAUSTED, "endpoint_alternatives")

    frames, field = _solution_frame_ids(solution)
    if frames is None or not frames:
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, field or "frames")
    for frame in frames:
        verdict = _same_axis_frame_id_verdict(frame, max_frame_id_bytes)
        if verdict is not None:
            return refuse(*verdict)
    for reading in readings:
        verdict = _same_axis_frame_id_verdict(
            getattr(reading, "frame", None), max_frame_id_bytes)
        if verdict is not None:
            return refuse(*verdict)
    for match in matches:
        if not isinstance(match, plane_graph.PlaneMatch):
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "input_match_identity")
        for endpoint_frame in (getattr(match, "left", None),
                               getattr(match, "right", None)):
            verdict = _same_axis_frame_id_verdict(
                endpoint_frame, max_frame_id_bytes)
            if verdict is not None:
                return refuse(*verdict)

    identity_rows: list[list] = []
    for match in matches:
        try:
            identity_rows.append(_input_match_identity(match))
        except _ERRORS:
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "input_match_identity")

    if axes == SAME_AXIS_CANDIDATES_AXES_V1:
        capture_axes, schema, basis, declared_axes = (
            SAME_AXIS_CANDIDATES_AXES_V1, SAME_AXIS_CANDIDATES_SCHEMA,
            SAME_AXIS_CANDIDATES_BASIS, None)
    else:
        if (type(axes) is not tuple or not axes
                or any(type(value) is not int or value not in (0, 1, 2)
                       for value in axes)
                or axes != SAME_AXIS_CANDIDATES_AXES_ALL):
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "axes")
        capture_axes, schema, basis, declared_axes = (
            axes, SAME_AXIS_CANDIDATES_SCHEMA_V2, SAME_AXIS_CANDIDATES_BASIS_V2,
            list(axes))

    getter = getattr(getattr(solution, "quadrants", None), "get", None)
    if not callable(getter):
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "quadrants")
    try:
        frame_quadrants = {frame: getter(frame) for frame in frames}
    except _ERRORS:
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "quadrants")
    for quadrant in frame_quadrants.values():
        if type(quadrant) is not int or quadrant not in (0, 1, 2, 3):
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "quadrants")

    alternatives = getattr(solution, "alternatives", None)
    if (isinstance(alternatives, (str, bytes))
            or not isinstance(alternatives, Sequence) or not alternatives):
        return refuse(_SAME_AXIS_CANDIDATES_UNAVAILABLE, "best_assignment")
    best_quadrants = getattr(alternatives[0], "quadrants", None)
    if (isinstance(best_quadrants, (str, bytes))
            or not isinstance(best_quadrants, Sequence)):
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED,
                      "alternatives[0].quadrants")
    if tuple(best_quadrants) != tuple(frame_quadrants[frame] for frame in frames):
        return refuse(_SAME_AXIS_CANDIDATES_UNAVAILABLE, "best_assignment")

    try:
        read_by_index, _index_by_object, _index_frames = \
            plane_graph._canonical_reading_index(readings)
    except _ERRORS:
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "readings")
    global_index = {id(reading): position
                    for position, reading in enumerate(readings)}

    # DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: every post-filter solver-active match is
    # resolved to its *world* axis with the production ``axis_transform`` under the
    # captured assignment, and each captured group binds the complete ordered list
    # of its retained match indices. A same-axis group can legitimately retain more
    # than one compatible match, so the binding is a list, not the v1 scalar.
    bound_groups: dict[tuple, list[int]] = {}
    for match_index, match in enumerate(matches):
        try:
            left, right = plane_graph._validate_indexed_match(match, read_by_index)
            left_quadrant = frame_quadrants[match.left]
            right_quadrant = frame_quadrants[match.right]
            left_world, _left_sign = plane_graph.axis_transform(
                left.axis, left_quadrant)
            right_world, _right_sign = plane_graph.axis_transform(
                right.axis, right_quadrant)
        except _ERRORS:
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "match_endpoint")
        active = bool(((right_quadrant - left_quadrant) % 4
                       == match.quadrant_of_right) and left_world == right_world)
        if active and left_world in capture_axes:
            key = (match.left, match.right, match.left_axis,
                   match.quadrant_of_right)
            bound_groups.setdefault(key, []).append(match_index)
    if len(bound_groups) > max_groups:
        return refuse(_SAME_AXIS_CANDIDATES_EXHAUSTED, "groups")
    ordered_keys = sorted(bound_groups)
    group_index_of = {key: index for index, key in enumerate(ordered_keys)}

    selected: list = []
    for candidate in observation.candidates:
        key = (candidate.left, candidate.right, candidate.left_axis,
               candidate.quadrant_of_right)
        if key in group_index_of:
            selected.append(candidate)
            if len(selected) > max_rows:
                return refuse(_SAME_AXIS_CANDIDATES_EXHAUSTED, "rows")
    disposition_by_id = {
        id(candidate): disposition
        for candidate, disposition in zip(observation.candidates,
                                          observation.dispositions)}
    post_index_by_id = {id(match): index for index, match in enumerate(matches)}

    table: dict[int, tuple] = {}
    rows_by_group: dict[int, list[tuple]] = {
        index: [] for index in range(len(ordered_keys))}
    for candidate in selected:
        key = (candidate.left, candidate.right, candidate.left_axis,
               candidate.quadrant_of_right)
        group_index = group_index_of[key]
        disposition = disposition_by_id.get(id(candidate))
        if disposition is None:
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "observation")
        match_index = post_index_by_id.get(id(candidate), -1)
        kept = disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT
        if kept != (match_index >= 0):
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "candidate_groups")
        try:
            left, right = plane_graph._validate_indexed_match(
                candidate, read_by_index, require_selected_b=True)
        except _ERRORS:
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "match_endpoint")
        left_index, left_alternative, left_count = _endpoint_alternative(
            left, candidate.left_b, max_alternatives)
        right_index, right_alternative, right_count = _endpoint_alternative(
            right, candidate.right_b, max_alternatives)
        if (left_count is None or left_count > max_alternatives
                or right_count is None or right_count > max_alternatives):
            return refuse(_SAME_AXIS_CANDIDATES_EXHAUSTED, "endpoint_alternatives")
        if left_alternative is None or right_alternative is None:
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "endpoint_alias")
        left_side = _strict_face_side(getattr(left, "side", None))
        right_side = _strict_face_side(getattr(right, "side", None))
        if left_side is None or right_side is None:
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "endpoint_side")
        try:
            delta = plane_graph._delta(candidate, read_by_index)
            if delta is None:
                raise ValueError("candidate delta is undefined")
            delta = float(delta)
            psr_value, psr_infinite = _same_axis_psr_columns(candidate.psr)
            score = _number(candidate.score)
            peak = _number(candidate.peak)
            second_peak = _number(candidate.second_peak)
            overlap_cells = int(candidate.overlap_cells)
            shift = candidate.cell_shift_rc
            if (type(shift) is not tuple or len(shift) != 2
                    or any(type(value) is not int for value in shift)):
                raise ValueError("candidate cell shift is malformed")
            if overlap_cells < 0:
                raise ValueError("candidate overlap is negative")
        except _ERRORS:
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "candidate_rows")
        relative = int(candidate.quadrant_of_right)
        for reading, plane, local_axis, side, index, alternative in (
                (left, candidate.left_plane, left.axis, left_side, left_index,
                 left_alternative),
                (right, candidate.right_plane, right.axis, right_side,
                 right_index, right_alternative)):
            position = global_index.get(id(reading))
            if position is None:
                return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "readings")
            if position not in table:
                rows = _reading_alternative_table(reading)
                if rows is None:
                    return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "reading_table")
                if len(rows) > max_alternatives:
                    return refuse(_SAME_AXIS_CANDIDATES_EXHAUSTED,
                                  "endpoint_alternatives")
                table[position] = (position, reading.frame, int(plane),
                                   int(local_axis), int(side), rows)
            else:
                stored = table[position]
                if (stored[2] != int(plane) or stored[3] != int(local_axis)
                        or stored[4] != int(side)):
                    return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "reading_table")
        rows_by_group[group_index].append((
            group_index, match_index, len(rows_by_group[group_index]),
            global_index[id(left)], left_index, global_index[id(right)],
            right_index, relative, score, psr_value, psr_infinite, peak,
            second_peak, overlap_cells, int(shift[0]), int(shift[1]), delta,
            "kept" if kept else "dropped",
            "" if kept else plane_graph.SAME_AXIS_DISPOSITION_DROPPED))

    groups: list[tuple] = []
    for group_index, key in enumerate(ordered_keys):
        claims = bound_groups[key]
        survivor_rows = [row for row in rows_by_group[group_index]
                         if row[1] in claims]
        if len(survivor_rows) != len(claims):
            return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "candidate_groups")
        if declared_axes is None:
            survivor_index = claims[0]
            survivor = next(row for row in survivor_rows
                            if row[1] == survivor_index)
            groups.append((
                group_index, key[0], key[1], key[2], key[3], survivor_index,
                list(identity_rows[survivor_index])))
        else:
            groups.append((
                group_index, key[0], key[1], key[2], key[3], list(claims),
                [list(identity_rows[index]) for index in claims]))

    candidates = tuple(row for group_index in range(len(ordered_keys))
                       for row in rows_by_group[group_index])
    if len(candidates) != len(selected):
        return refuse(_SAME_AXIS_CANDIDATES_MALFORMED, "candidate_rows")
    reading_rows = tuple(table[position] for position in sorted(table))
    value = SameAxisCandidateCapture(
        schema=schema, recorded=True,
        basis=basis, refusal_reason=None,
        failed_assignment_sha256=_failed_assignment_digest(
            frames, frame_quadrants),
        input_match_identity_sha256=_alias_domain_digest(
            SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
            SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA, identity_rows),
        input_match_count=len(matches), group_count=len(groups),
        row_count=len(candidates), frames=tuple(frames),
        assignments=tuple(frame_quadrants[frame] for frame in frames),
        readings=reading_rows, groups=tuple(groups), candidates=candidates,
        axes=tuple(capture_axes),
        input_identities=tuple(tuple(row) for row in identity_rows))
    encoded = json.dumps(value.to_record(), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    if len(encoded) > max_record_bytes:
        return refuse(_SAME_AXIS_CANDIDATES_EXHAUSTED, "record_bytes")
    return value


def failed_same_axis_candidates(
        readings: object,
        matches: object,
        solution: plane_graph.ComponentSolution,
        observation: object,
        axes: object = SAME_AXIS_CANDIDATES_AXES_V1,
) -> "SameAxisCandidateCapture":
    """Fail-closed wrapper around the bounded pre-filter candidate collector.

    ``axes`` defaults to the frozen v1 world-axis-1 record; passing
    :data:`SAME_AXIS_CANDIDATES_AXES_ALL` selects the evaluation-only v2 all-axis
    aggregate. Any other value is one atomic typed refusal, never a partial record.
    """
    try:
        return _collect_same_axis_candidates(
            readings, matches, solution, observation, axes)
    except _ERRORS:
        return _same_axis_candidates_refusal(
            _SAME_AXIS_CANDIDATES_MALFORMED, "record")


def failed_joint_same_axis_candidates(
    readings: object, selected: object, solution: plane_graph.ComponentSolution,
    observation: object, matcher_matches: object, admitted_witness_rows: object,
    source_rows: object, selection_sha256: object,
    selection_payload: object,
) -> "SameAxisCandidateCapture":
    """Bind copied joint representatives to matcher-owned full candidate rows.

    The independent pre-filter witness publication must cover every admitted
    observation row, including dropped ones. A partial or inconsistent ledger
    refuses before any success-shaped candidate capture is constructed.
    """
    def refuse(field: str):
        return _same_axis_candidates_refusal(_SAME_AXIS_CANDIDATES_MALFORMED,
                                             field)
    try:
        if (not isinstance(observation, plane_graph.SameAxisCandidateObservation)
                or not isinstance(matcher_matches, tuple)
                or not isinstance(selected, tuple)
                or not isinstance(admitted_witness_rows, list)
                or not isinstance(source_rows, tuple)
                or not _hex(selection_sha256)
                or type(selection_payload) is not dict
                or _sha(selection_payload) != selection_sha256):
            return refuse("observation")
        if len(observation.candidates) > SAME_AXIS_CANDIDATES_MAX_ROWS:
            return _same_axis_candidates_refusal(
                _SAME_AXIS_CANDIDATES_EXHAUSTED, "rows")
        if (len(observation.candidates) != len(admitted_witness_rows)
                or len(selected) != len(source_rows)
                or len(set(source_rows)) != len(source_rows)):
            return refuse("candidate_rows")
        kept = []
        ledger = []
        for index, (candidate, disposition, witness) in enumerate(zip(
                observation.candidates, observation.dispositions,
                admitted_witness_rows)):
            if (type(candidate) is not plane_graph.PlaneMatch
                    or type(witness) is not plane_graph.PreFilterMatcherWitnessRowV1
                    or witness.admitted_row_index != index
                    or witness.disposition != disposition):
                return refuse("candidate_rows")
            payload = plane_graph._indexed_match_payload(candidate)
            if witness.match_sha256 != _sha(payload):
                return refuse("candidate_rows")
            if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT:
                if witness.post_match_index != len(kept):
                    return refuse("candidate_rows")
                kept.append(candidate)
            elif (disposition != plane_graph.SAME_AXIS_DISPOSITION_DROPPED
                  or witness.post_match_index is not None):
                return refuse("candidate_rows")
            ledger.append((index, disposition, witness.post_match_index,
                           payload, witness.match_sha256))
        if (len(kept) != len(matcher_matches)
                or any(a is not b for a, b in zip(kept, matcher_matches))):
            return refuse("candidate_groups")
        for index, (match, source_row) in enumerate(zip(selected, source_rows)):
            if (type(source_row) is not int or source_row < 0
                    or source_row >= len(matcher_matches)
                    or type(match) is not plane_graph.PlaneMatch
                    or dataclasses.replace(matcher_matches[source_row],
                        left_b=match.left_b, right_b=match.right_b) != match):
                return refuse("candidate_groups")
        matcher_capture = _collect_same_axis_candidates(
            readings, matcher_matches, solution, observation,
            SAME_AXIS_CANDIDATES_AXES_ALL)
        if not matcher_capture.recorded:
            return matcher_capture
        identities = [_input_match_identity(match) for match in selected]
        value = SameAxisCandidateCapture(
            schema=SAME_AXIS_CANDIDATES_SCHEMA_V3, recorded=True,
            basis=SAME_AXIS_CANDIDATES_BASIS_V3, refusal_reason=None,
            failed_assignment_sha256=matcher_capture.failed_assignment_sha256,
            input_match_identity_sha256=_alias_domain_digest(
                SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
                SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA, identities),
            input_match_count=len(selected),
            group_count=matcher_capture.group_count,
            row_count=matcher_capture.row_count,
            frames=matcher_capture.frames, assignments=matcher_capture.assignments,
            readings=(), groups=(), candidates=(),
            axes=SAME_AXIS_CANDIDATES_AXES_ALL,
            input_identities=tuple(tuple(row) for row in identities),
            matcher_capture=matcher_capture.to_record(),
            admitted_rows=tuple(ledger),
            admitted_ledger_sha256=plane_graph.admitted_candidate_ledger_sha256(
                (row[0], row[1], row[2], row[4]) for row in ledger),
            selected_source_rows=source_rows,
            selection_sha256=selection_sha256,
            selection_payload=selection_payload)
        encoded = json.dumps(value.to_record(), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True,
                             allow_nan=False).encode("utf-8")
        if len(encoded) > SAME_AXIS_CANDIDATES_MAX_RECORD_BYTES:
            return _same_axis_candidates_refusal(
                _SAME_AXIS_CANDIDATES_EXHAUSTED, "record_bytes")
        return value
    except _ERRORS:
        return refuse("record")


def _check_frame_id(
    value: object, max_frame_id_bytes: int,
) -> tuple[str, str] | None:
    """Bounded structural check for one frame identifier.

    A missing, empty or non-string identifier is a malformed join; an identifier
    the fixed UTF-8 cap cannot hold, including one that cannot be encoded at all,
    is a budget refusal. ``None`` means the identifier is a usable bounded
    string. Only the bounded field name leaves this helper -- the offending
    value itself is never copied into a refusal record.
    """
    if type(value) is not str or not value:
        return _ACTIVE_EDGE_ALIASES_MALFORMED, "frame_id"
    length = _utf8_length(value)
    if length is None or length > max_frame_id_bytes:
        return _ACTIVE_EDGE_ALIASES_EXHAUSTED, "frame_id"
    return None


def _solution_frame_ids(
    solution: object,
) -> tuple[tuple[object, ...] | None, str | None]:
    """Copy the returned solution's frame sequence without ever raising."""
    frames = getattr(solution, "frames", None)
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        return None, "frames"
    try:
        return tuple(frames), None
    except _ERRORS:
        return None, "frames"


def _identity_number(value: object) -> float:
    """One finite non-bool identity scalar, or a bounded refusal.

    The identity digest is only meaningful if it binds real numbers, so a bool,
    a non-numeric label or a non-finite value raises instead of being coerced.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("input match identity label must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except _ERRORS as error:
        raise ValueError(
            "input match identity label must be a finite number") from error
    if not np.isfinite(number):
        raise ValueError("input match identity label must be finite")
    return number


def _input_match_identity(match: plane_graph.PlaneMatch) -> list:
    """Ordered scalar identity projection of one post-filter input match.

    Every scalar is copied exactly and the two ``b`` labels must be finite
    numbers, so a malformed or non-finite label refuses instead of being
    serialized into the identity digest.
    """
    return [match.left, match.left_plane, match.left_axis, match.right,
            match.right_plane, match.right_axis, match.quadrant_of_right,
            _identity_number(match.left_b), _identity_number(match.right_b)]


def _utf8_length(value: object) -> int | None:
    if type(value) is not str:
        return None
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _non_finite_solution_field(
        solution: plane_graph.ComponentSolution) -> str | None:
    """Exact path of the first non-finite number ``_solution_payload`` emits.

    ``_solution_payload`` forwards solver numbers verbatim, so a non-finite
    value would otherwise reach ``json.dumps(allow_nan=False)`` and abort the
    producer instead of abstaining. Every numeric scalar the payload emits is
    inspected -- quadrant values, translation components, axis residuals and
    resultants, blocks-per-period entries, and each alternative's quadrants,
    cost, resultants and axis observations -- together with numeric mapping
    keys the payload re-emits as values. Only the offending field path is
    returned; the non-finite value itself is never echoed into that path.
    """
    def offender(value: object, path: str) -> str | None:
        try:
            _number(value)
        except (TypeError, ValueError, OverflowError):
            return path
        return None

    def key_offender(key: object, container: str) -> str | None:
        """Numeric mapping keys are re-emitted as payload values, so validate."""
        if isinstance(key, str) or not isinstance(
                key, (int, float, np.integer, np.floating)):
            return None
        try:
            _number(key)
        except (TypeError, ValueError, OverflowError):
            return f"{container}[<non-finite-key>]"
        return None

    def label(key: object) -> str:
        if isinstance(key, str):
            return repr(key)
        if isinstance(key, (int, np.integer)):
            return repr(int(key))
        try:
            _number(key)
        except (TypeError, ValueError, OverflowError):
            return "<non-finite-key>"
        return repr(float(key))

    for frame, quadrant in solution.quadrants.items():
        container = "component_solution.quadrants"
        path = key_offender(frame, container)
        if path is not None:
            return path
        path = offender(quadrant, f"{container}[{label(frame)}]")
        if path is not None:
            return path
    for frame, translation in solution.translations.items():
        container = "component_solution.translations"
        path = key_offender(frame, container)
        if path is not None:
            return path
        base = f"{container}[{label(frame)}]"
        for index, item in enumerate(translation):
            path = offender(item, f"{base}[{index}]")
            if path is not None:
                return path
    for axis, residual in solution.axis_residuals.items():
        container = "component_solution.axis_residuals"
        path = key_offender(axis, container)
        if path is not None:
            return path
        path = offender(residual, f"{container}[{label(axis)}]")
        if path is not None:
            return path
    for axis, resultant in solution.axis_resultants.items():
        container = "component_solution.axis_resultants"
        path = key_offender(axis, container)
        if path is not None:
            return path
        path = offender(resultant, f"{container}[{label(axis)}]")
        if path is not None:
            return path
    for key, value in solution.blocks_per_period.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        frame, index = key
        container = "component_solution.blocks_per_period"
        path = key_offender(frame, container)
        if path is not None:
            return path
        base = f"{container}[{label(frame)}]"
        path = key_offender(index, base)
        if path is not None:
            return path
        path = offender(value, f"{base}[{label(index)}]")
        if path is not None:
            return path
    for position, alternative in enumerate(solution.alternatives):
        base = f"component_solution.alternatives[{position}]"
        for index, quadrant in enumerate(alternative.quadrants):
            path = offender(quadrant, f"{base}.quadrants[{index}]")
            if path is not None:
                return path
        path = offender(alternative.cost, f"{base}.cost")
        if path is not None:
            return path
        for index, resultant in enumerate(alternative.resultants):
            path = offender(resultant, f"{base}.resultants[{index}]")
            if path is not None:
                return path
        for index, observation in enumerate(alternative.axis_observations):
            path = offender(observation, f"{base}.axis_observations[{index}]")
            if path is not None:
                return path
    return None


def _payload(result: "HistoricalImageDerivedGraphResult") -> dict:
    return {
        "schema": result.schema, "producer": result.producer,
        "terminal": result.terminal, "status": result.status,
        "reason": result.reason, "detail": result.detail,
        "session_sha256": result.session_sha256,
        "lattice_sha256": result.lattice_sha256,
        "signatures_sha256": result.signatures_sha256,
        "correlation_algorithm": result.correlation_algorithm,
        "max_shift_cells": result.max_shift_cells,
        "frame_ids": list(result.frame_ids),
        "reading_indices": list(range(len(result.readings))),
        "signature_digests": [item.signature_sha256 for item in result.signatures],
        "matches": [_match_payload(item) for item in result.matches],
        "component_solution": _solution_payload(result.component_solution),
    }


@dataclass(frozen=True, slots=True)
class HistoricalImageDerivedGraphResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    session_sha256: str
    lattice_sha256: str
    signatures_sha256: str
    frame_ids: tuple[str, ...]
    readings: tuple[lattice_planes.PlaneReading, ...]
    signatures: tuple[signatures_mod.SignatureEvidence, ...]
    matches: tuple[plane_graph.PlaneMatch, ...]
    component_solution: plane_graph.ComponentSolution | None
    correlation_algorithm: str = plane_graph.PLANE_CORRELATION_ALGORITHM
    max_shift_cells: int = 32
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != HISTORICAL_IMAGE_DERIVED_GRAPH_SCHEMA
                or self.producer != HISTORICAL_IMAGE_DERIVED_GRAPH_PRODUCER
                or self.terminal != GRAPH_TERMINAL
                or self.status not in ("assembled", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(value) for value in (
                    self.session_sha256, self.lattice_sha256,
                    self.signatures_sha256))
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(value) is not str or not value for value in self.frame_ids)
                or not isinstance(self.readings, tuple)
                or any(not isinstance(value, lattice_planes.PlaneReading)
                       for value in self.readings)
                or not isinstance(self.signatures, tuple)
                or any(not isinstance(value, signatures_mod.SignatureEvidence)
                       for value in self.signatures)
                or not isinstance(self.matches, tuple)
                or any(not isinstance(value, plane_graph.PlaneMatch)
                       for value in self.matches)
                or self.correlation_algorithm != plane_graph.PLANE_CORRELATION_ALGORITHM
                or type(self.max_shift_cells) is not int
                or self.max_shift_cells <= 0):
            raise ValueError("image-derived graph result is malformed")
        for match in self.matches:
            if _psr_from_payload(_psr_payload(match.psr)) != match.psr:
                raise ValueError("image-derived graph psr does not replay")
        for evidence in self.signatures:
            signatures_mod.SignatureEvidence(**{
                field.name: getattr(evidence, field.name)
                for field in dataclasses.fields(signatures_mod.SignatureEvidence)})
        if self.status == "assembled":
            if (self.reason or self.detail or not self.matches
                    or not isinstance(self.component_solution,
                                      plane_graph.ComponentSolution)
                    or self.component_solution.reason):
                raise ValueError("assembled image-derived graph is inconsistent")
        elif (not self.reason or self.matches or self.component_solution is not None):
            raise ValueError("unavailable image-derived graph must be atomic")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived graph result digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = "") -> placement.PlacementRefusal:
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _fields(value: object) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}


def _replay_session(value: object) -> session_mod.ImageDerivedSession | placement.PlacementRefusal:
    if not isinstance(value, session_mod.ImageDerivedSession):
        return _refusal("invalid_input", "session")
    try:
        rebuilt = session_mod.image_derived_session_from_record(
            session_mod.image_derived_session_to_record(value))
        for frame in value.frames:
            dataclasses.replace(frame)
    except (session_mod.ImageDerivedSessionError, *_ERRORS) as error:
        return _refusal("provenance_mismatch", "session", type(error).__name__)
    if rebuilt.session_sha256 != value.session_sha256:
        return _refusal("provenance_mismatch", "session.session_sha256")
    return value


def _replay(value: object, expected: type, field: str) -> object:
    if not isinstance(value, expected):
        return _refusal("invalid_input", field)
    try:
        return expected(**_fields(value))
    except _ERRORS as error:
        return _refusal("provenance_mismatch", field, type(error).__name__)


def _strict_yaw_class(value: object) -> int | None:
    """One exact integer yaw class in ``0..3``, or ``None`` when malformed.

    ``bool`` (and ``numpy.bool_``) is never an integer, a float/string is never
    coerced through ``int()``, and an out-of-range value is refused; the caller
    turns every refusal into one atomic typed graph refusal rather than silently
    dropping or rounding the evidence.
    """
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= 3):
        return None
    return int(value)


def _reading_present_frames(readings: object) -> tuple[str, ...]:
    """The reading-present frame ids, in their deterministic lattice order.

    The lattice reading order is canonical session order, so the first reading
    seen for a frame id fixes that frame's star anchor. A malformed reading
    without a usable ``frame`` contributes nothing: it can never introduce a
    fabricated frame into the solver's universe, and the solver retains its own
    reading-universe validation.
    """
    frames: list[str] = []
    seen: set[str] = set()
    if isinstance(readings, (str, bytes)) or not isinstance(readings, Sequence):
        return ()
    for reading in readings:
        frame = getattr(reading, "frame", None)
        if type(frame) is str and frame and frame not in seen:
            seen.add(frame)
            frames.append(frame)
    return tuple(frames)


def _z4_potentials(
    edges: tuple[tuple[str, str, int], ...],
) -> tuple[dict[str, int], dict[str, str], dict[str, int]] | placement.PlacementRefusal:
    """Canonical Z4 potentials for every solved edge, or an atomic refusal.

    ``right_yaw_class`` is the relative rotation ``k_right - k_left`` in Z4, so
    a solved component is consistent exactly when such potentials exist. Every
    edge -- including edges whose endpoints have no plane reading -- is checked
    against the potentials derived from the component's other edges, and a
    contradictory cycle refuses the whole graph atomically. Returns the
    potentials plus each frame's component root, so the caller can group
    reading-present frames by an explicit component identity rather than by an
    accidental potential value.
    """
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for left, right, relative in edges:
        adjacency.setdefault(left, [])
        if right != left:
            adjacency.setdefault(right, [])
            adjacency[left].append((right, relative))
            adjacency[right].append((left, (-relative) % 4))
        elif relative % 4 != 0:
            # A self-edge in Z4 requires 0; any other value is already a
            # contradiction, so it refuses before any emission. The solver's
            # own token is a closed-enum member, so the ordinary validated
            # builder carries it instead of normalizing it to invalid_input.
            return _refusal("quadrant_constraint_inconsistent",
                            "connection_graph.quadrant_constraints")
    potentials: dict[str, int] = {}
    component_root: dict[str, str] = {}
    component_rank: dict[str, int] = {}
    for root in adjacency:
        if root in potentials:
            continue
        rank = len(component_rank)
        potentials[root] = 0
        component_root[root] = root
        component_rank[root] = rank
        queue: deque[str] = deque([root])
        while queue:
            current = queue.popleft()
            for neighbour, relative in adjacency[current]:
                expected = (potentials[current] + relative) % 4
                known = potentials.get(neighbour)
                if known is None:
                    potentials[neighbour] = expected
                    component_root[neighbour] = root
                    component_rank[neighbour] = rank
                    queue.append(neighbour)
                elif known != expected:
                    return _refusal("quadrant_constraint_inconsistent",
                                    "connection_graph.quadrant_constraints")
    return potentials, component_root, component_rank


def _reading_present_constraint_projection(
    edges: tuple[tuple[str, str, int], ...],
    frame_ids: tuple[str, ...],
    readings: object,
) -> tuple[tuple[str, str, int], ...] | placement.PlacementRefusal:
    """Project validated solved edges onto the reading-present frames.

    The full solved component is replayed and its canonical Z4 potentials are
    checked over *every* edge, so a contradiction through an omitted frame still
    refuses atomically. When the component is consistent, the adapter emits a
    deterministic spanning star -- the earliest reading-present frame of each
    component to each of that component's other reading-present frames -- so the
    solver only ever sees endpoints inside its own reading universe. Frames with
    no reading are omitted: they contribute no planes, translation, scale or
    axis coverage, and no relation is invented for a refused edge. Each
    connected component keeps its own gauge, because every emitted star is
    rooted at that component's own first reading-present frame -- not at the
    solved component's arbitrary root, which may itself have no reading.

    The projection is a consistent spanning-star basis over the observed frames,
    not a verbatim replay of the solved edges: it is derived only after the full
    graph -- including edges through omitted frames -- has passed cycle
    validation, and each observed frame is related to its component's observed
    root by the composed yaw class. An all-observed component therefore keeps
    the same relations as its solved edges, but it is emitted as this star basis
    rather than as the original direct edges.
    """
    potentials = _z4_potentials(edges)
    if isinstance(potentials, placement.PlacementRefusal):
        return potentials
    values, component_root, component_rank = potentials
    present = tuple(frame for frame in _reading_present_frames(readings)
                    if frame in values)
    if len(present) < 2:
        return ()
    index_of = {frame: index for index, frame in enumerate(frame_ids)}
    grouped: dict[str, list[str]] = {}
    for frame in present:
        grouped.setdefault(component_root[frame], []).append(frame)
    # Anchor each star at the component's earliest reading-present frame. The
    # solved component root is only an arbitrary traversal seed and may have no
    # reading, so emitting from it would leak an endpoint outside the solver's
    # reading universe. ``present`` is ordered by the reading universe, so
    # ``grouped[component][0]`` is that first retained frame.
    observed_root = {component: members[0] for component, members in grouped.items()}
    emitted = [
        (observed_root[component], member,
         (values[member] - values[observed_root[component]]) % 4)
        for component in sorted(grouped, key=component_rank.__getitem__)
        for member in grouped[component][1:]]
    emitted.sort(key=lambda item: (index_of.get(item[0], len(frame_ids)),
                                   index_of.get(item[1], len(frame_ids))))
    return tuple(emitted)


def _connection_constraints(
    value: object,
    session: session_mod.ImageDerivedSession,
    lattice: lattice_mod.ImageDerivedLatticeResult,
) -> tuple[tuple[str, str, int], ...] | placement.PlacementRefusal:
    """Replay and bind accepted pair evidence, retaining solved yaw classes only."""
    if value is None:
        return ()
    if not isinstance(value, connections_mod.ImageDerivedConnectionGraph):
        return _refusal("invalid_input", "connection_graph")
    # Validate the raw solved yaw classes BEFORE any dataclass replay. A
    # malformed bool/float/string/out-of-range value must surface as the typed
    # ``invalid_input`` field ``connection_graph.yaw_class`` rather than being
    # masked by a replay/digest failure or silently coerced. Only solved records
    # carry a relation; a refused record's ``None`` yaw class is legitimate.
    raw_evidence = getattr(value, "evidence", ())
    if isinstance(raw_evidence, tuple):
        for record in raw_evidence:
            if getattr(record, "status", None) != "solved":
                continue
            if _strict_yaw_class(getattr(record, "right_yaw_class", None)) is None:
                return _refusal("invalid_input", "connection_graph.yaw_class")
    try:
        graph = connections_mod.ImageDerivedConnectionGraph(**_fields(value))
        connections_mod.ConnectionBudgets(**_fields(graph.budgets))
        for record in graph.evidence:
            for diagnostic in record.hypotheses:
                connections_mod.YawClassDiagnostic(**_fields(diagnostic))
            connections_mod.RelativeCameraEvidence(**_fields(record))
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "connection_graph",
                        type(error).__name__)
    frame_ids = tuple(frame.frame_id for frame in session.frames)
    if (graph.status != "ok" or graph.session_sha256 != session.session_sha256
            or graph.frame_ids != frame_ids
            or graph.graph_sha256 != lattice.graph_sha256):
        return _refusal("provenance_mismatch", "connection_graph.binding")
    if any(record.session_sha256 != graph.session_sha256
           for record in graph.evidence):
        return _refusal("provenance_mismatch", "connection_graph.evidence")
    edges: list[tuple[str, str, int]] = []
    for record in graph.evidence:
        if record.status != "solved":
            continue
        relative = _strict_yaw_class(record.right_yaw_class)
        if relative is None:
            return _refusal("invalid_input", "connection_graph.yaw_class")
        edges.append((record.left_id, record.right_id, relative))
    return _reading_present_constraint_projection(
        tuple(edges), frame_ids, getattr(lattice, "readings", None))


def _unavailable(session: session_mod.ImageDerivedSession,
                 lattice: lattice_mod.ImageDerivedLatticeResult,
                 signatures: signatures_mod.ImageDerivedSignaturesResult,
                 reason: str, detail: str, max_shift_cells: int = 32
                 ) -> HistoricalImageDerivedGraphResult:
    return HistoricalImageDerivedGraphResult(
        HISTORICAL_IMAGE_DERIVED_GRAPH_SCHEMA, HISTORICAL_IMAGE_DERIVED_GRAPH_PRODUCER, GRAPH_TERMINAL,
        "unavailable", reason, detail, session.session_sha256,
        lattice.result_sha256, signatures.result_sha256,
        tuple(frame.frame_id for frame in session.frames),
        tuple(lattice.readings), tuple(signatures.signatures), (), None,
        plane_graph.PLANE_CORRELATION_ALGORITHM, max_shift_cells)


def _ordered_universe(
    lattice: lattice_mod.ImageDerivedLatticeResult,
    signatures: signatures_mod.ImageDerivedSignaturesResult,
) -> tuple[tuple[lattice_planes.PlaneReading, ...], tuple[signatures_mod.SignatureEvidence, ...]]:
    readings = tuple(lattice.readings)
    expected = [(index, alternative_index)
                for index, reading in enumerate(readings)
                for alternative_index, _ in enumerate(reading.alternatives)]
    actual = [(item.canonical_index, item.alternative_index)
              for item in signatures.signatures]
    if actual != expected:
        raise ValueError("signature evidence order does not cover the lattice universe")
    for item in signatures.signatures:
        reading = readings[item.canonical_index]
        alternative = reading.alternatives[item.alternative_index]
        if (item.signature.reading is not reading or item.frame_id != reading.frame
                or item.reading_axis != reading.axis
                or item.blocks_per_period != alternative.blocks_per_period):
            raise ValueError("signature evidence is foreign to its lattice reading")
    return readings, tuple(signatures.signatures)


def _production_solution_refusal(
    frame_ids: tuple[str, ...],
    readings: Sequence[lattice_planes.PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    solution: plane_graph.ComponentSolution,
) -> tuple[str, str] | None:
    """Check observability and equations conditional on the selected assignment.

    This stricter production boundary neither changes the solver's decisions nor
    proves the selected correspondences globally correct. Connectivity uses the
    solver's one existing anchor on each world axis; inactive matches contribute
    no translation evidence.
    """
    try:
        read_by_index, _index_by_object, reading_frames = (
            plane_graph._canonical_reading_index(readings))
        required = set(frame_ids)
        if (not required or len(required) != len(frame_ids)
                or reading_frames != tuple(sorted(required))
                or solution.frames != reading_frames
                or not isinstance(solution.quadrants, Mapping)
                or not isinstance(solution.translations, Mapping)
                or set(solution.quadrants) != required
                or set(solution.translations) != required):
            return "graph_frame_mismatch", "production frame coverage differs"
        for frame in solution.frames:
            quadrant = solution.quadrants[frame]
            translation = solution.translations[frame]
            if (type(quadrant) is not int or quadrant not in (0, 1, 2, 3)
                    or not isinstance(translation, Sequence)
                    or len(translation) != 3):
                return "graph_active_constraint_invalid", "selected assignment is malformed"
        constraints: dict[int, list[tuple[str, str, float]]] = {
            axis: [] for axis in range(3)}
        selected_b: dict[tuple[str, int], float] = {}
        for match in matches:
            if (not isinstance(match, plane_graph.PlaneMatch)
                    or match.left not in required or match.right not in required
                    or type(match.quadrant_of_right) is not int
                    or match.quadrant_of_right not in (0, 1, 2, 3)):
                return "graph_active_constraint_invalid", "match assignment is malformed"
            if ((solution.quadrants[match.right] - solution.quadrants[match.left])
                    % 4 != match.quadrant_of_right):
                continue
            left, right = plane_graph._validate_indexed_match(
                match, read_by_index, require_selected_b=True)
            world, left_sign = plane_graph.axis_transform(
                left.axis, solution.quadrants[match.left])
            right_world, right_sign = plane_graph.axis_transform(
                right.axis, solution.quadrants[match.right])
            if (world != right_world or not _finite_number(left.side)
                    or not _finite_number(right.side)
                    or left.side not in (-1.0, 1.0)
                    or right.side not in (-1.0, 1.0)):
                return "graph_active_constraint_invalid", "active endpoint axes or sides differ"
            for key, value in (((match.left, match.left_plane), match.left_b),
                               ((match.right, match.right_plane), match.right_b)):
                if key in selected_b and selected_b[key] != value:
                    return "graph_endpoint_scale_conflict", "active endpoint scales differ"
                selected_b[key] = float(value)
            delta = (left_sign * left.side * left.distance_for(match.left_b)
                     - right_sign * right.side * right.distance_for(match.right_b))
            constraints[world].append((match.left, match.right, float(delta)))
        if (not isinstance(solution.blocks_per_period, Mapping)
                or any(not isinstance(key, tuple) or len(key) != 2
                       or type(key[0]) is not str or key[0] not in required
                       or type(key[1]) is not int or key[1] < 0
                       or not _finite_number(value) or value <= 0.
                       for key, value in solution.blocks_per_period.items())
                or dict(solution.blocks_per_period) != selected_b):
            return "graph_endpoint_scale_conflict", "solution endpoint scale map differs"
        anchor = solution.frames[0]
        for axis, rows in constraints.items():
            adjacency = {frame: set() for frame in solution.frames}
            for left, right, _delta in rows:
                adjacency[left].add(right)
                adjacency[right].add(left)
            visited = {anchor}
            queue = deque([anchor])
            while queue:
                for neighbour in adjacency[queue.popleft()] - visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
            if not rows or visited != required:
                return "graph_translation_unobservable", f"world axis {axis} is disconnected"
        for axis, rows in constraints.items():
            for left, right, delta in rows:
                error = (solution.translations[right][axis]
                         - solution.translations[left][axis] - delta)
                if (not _finite_number(error)
                        or abs(error) > _PRODUCTION_EDGE_ERROR_LIMIT_BLOCKS):
                    return "graph_equation_inconsistent", f"world axis {axis} edge error exceeds limit"
    except _ERRORS:
        return "graph_active_constraint_invalid", "active constraints are malformed"
    return None


def derive_historical_image_derived_graph(
    *, session: object, lattice_result: object, signatures_result: object,
    connection_graph: object = None,
    max_shift_cells: int = 32, integer_tolerance: float = 0.15,
    min_resultant: float = 0.90, ambiguous_margin: float = 0.02,
    diagnostic_sink: plane_graph.PlaneGraphTrace | None = None,
    matcher_witness_sink=None,
    matcher_witness_payload_sink=None,
    witness_budget=None,
    solver_failure_sink=None,
    same_axis_candidate_sink=None,
    candidate_support_sink=None,
    complete_support_sink=None,
    signature_witnesses=None,
    same_axis_capture_axes=SAME_AXIS_CANDIDATES_AXES_V1,
    alias_max_readings: int = ACTIVE_EDGE_ALIASES_MAX_READINGS,
    alias_max_matches: int = ACTIVE_EDGE_ALIASES_MAX_MATCHES,
    alias_max_active_edges: int = ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES,
    alias_max_endpoint_alternatives: int =
        ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES,
    alias_max_frame_id_bytes: int = ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES,
    alias_max_record_bytes: int = ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES,
) -> HistoricalImageDerivedGraphResult | placement.PlacementRefusal:
    """Match the exact F1 universe and solve its component without fallbacks.

    ``same_axis_candidate_sink`` is an optional evaluation-only callback. When
    supplied and the component solver abstains, it is called once with the
    bounded :class:`SameAxisCandidateCapture` derived from the same local
    readings, post-filter matches, returned solution, and the raw pre-filter
    matcher observation. ``same_axis_capture_axes`` selects that record's frozen
    v1 world-axis-1 shape (the default) or the evaluation-only v2 all-axis
    aggregate. Neither is consulted by the production result and both are
    ``None`` on the default call, so the returned result and digest are unchanged
    whether or not the sink is attached.

    ``complete_support_sink`` is the additive, default-off V2 diagnostic seam
    (DS-R4-V2-ROUTEB-DIAGNOSTIC-WIRING). When supplied and the component solver
    abstains, it is called once with the complete
    :class:`candidate_support_provenance_v2.CompleteSupportCaptureV2` collected
    from the same accepted session ingress pins, pre-filter matcher witness rows,
    signature witnesses and raw pre-filter observation, under exactly the frozen
    Route B diagnostic profile. It re-matches, re-signatures and re-solves
    nothing, is never consulted by the production result, and defaults to
    ``None``, so the default call allocates no V2 material and the returned result
    and digest stay byte-for-byte unchanged.
    """
    accepted_session = _replay_session(session)
    if isinstance(accepted_session, placement.PlacementRefusal):
        return accepted_session
    lattice = _replay(lattice_result, lattice_mod.ImageDerivedLatticeResult,
                      "lattice_result")
    if isinstance(lattice, placement.PlacementRefusal):
        return lattice
    signatures = _replay(signatures_result,
                         signatures_mod.ImageDerivedSignaturesResult,
                         "signatures_result")
    if isinstance(signatures, placement.PlacementRefusal):
        return signatures
    frame_ids = tuple(frame.frame_id for frame in accepted_session.frames)
    if (lattice.session_sha256 != accepted_session.session_sha256
            or signatures.session_sha256 != accepted_session.session_sha256
            or signatures.lattice_sha256 != lattice.result_sha256
            or lattice.frame_ids != frame_ids or signatures.frame_ids != frame_ids):
        return _refusal("provenance_mismatch", "upstream.binding")
    constraints = _connection_constraints(
        connection_graph, accepted_session, lattice)
    if isinstance(constraints, placement.PlacementRefusal):
        return constraints
    if lattice.status != "assembled" or signatures.status != "assembled":
        reason = lattice.reason if lattice.status != "assembled" else signatures.reason
        return _unavailable(accepted_session, lattice, signatures,
                            "upstream_unavailable", reason, max_shift_cells)
    try:
        readings, evidence = _ordered_universe(lattice, signatures)
        raw_observations = ([] if (same_axis_candidate_sink is not None
                                   or candidate_support_sink is not None
                                   or complete_support_sink is not None) else None)
        pre_filter_rows = ([] if (candidate_support_sink is not None
                                  or complete_support_sink is not None) else None)
        match_kwargs = {"max_shift_cells": max_shift_cells}
        if diagnostic_sink is not None:
            match_kwargs["diagnostic_sink"] = diagnostic_sink
        if raw_observations is not None:
            match_kwargs["candidate_observation_sink"] = raw_observations
        if (matcher_witness_sink is not None or matcher_witness_payload_sink is not None
                or candidate_support_sink is not None
                or complete_support_sink is not None):
            if matcher_witness_sink is not None:
                match_kwargs["witness_sink"] = matcher_witness_sink
            if matcher_witness_payload_sink is not None:
                match_kwargs["witness_payload_sink"] = matcher_witness_payload_sink
            if pre_filter_rows is not None:
                match_kwargs["pre_filter_witness_sink"] = pre_filter_rows
            match_kwargs["witness_budget"] = witness_budget
            match_kwargs["signature_sha256_by_identity"] = {
                id(item.signature): item.signature_sha256 for item in evidence}
        matches = plane_graph.match_plane_graph(
            readings, tuple(item.signature for item in evidence), **match_kwargs)
        if not matches:
            return _unavailable(accepted_session, lattice, signatures,
                                "no_matches", "plane matcher found no matches",
                                max_shift_cells)
        # DS-IMG-SOLVER-CONTEXT-CAPTURE-DIAGNOSTIC-20261004: bind the exact
        # effective arguments immediately before the single solve call, so the
        # failure observer can later capture precisely what ``solve_component``
        # received rather than re-reading a caller-visible variable. These names
        # only alias the values already passed; the call itself is unchanged.
        effective_integer_tolerance = integer_tolerance
        effective_min_resultant = min_resultant
        effective_ambiguous_margin = ambiguous_margin
        effective_constraints = constraints
        support_observation._period_support_private(lambda _capture: support_observation._period_support_event(
            "solver_call", readings=tuple(readings), matches=tuple(matches),
            integer_tolerance=effective_integer_tolerance, min_resultant=effective_min_resultant,
            ambiguous_margin=effective_ambiguous_margin,
            quadrant_constraints=(tuple(effective_constraints.items())
                if isinstance(effective_constraints, dict) else effective_constraints)))
        solution = plane_graph.solve_component(
            readings, matches, integer_tolerance=effective_integer_tolerance,
            min_resultant=effective_min_resultant,
            ambiguous_margin=effective_ambiguous_margin,
            quadrant_constraints=effective_constraints)
        support_observation._period_support_private(lambda _capture:
            support_observation._period_support_event("solver_result", solution=solution))
    except _ERRORS as error:
        return _refusal("invalid_input", "graph.derivation", type(error).__name__)
    if solution.reason:
        if solver_failure_sink is not None:
            # The alias diagnostic is derived from the SAME local readings and
            # post-filter matches that were handed to ``solve_component`` above,
            # plus its already-returned solution. No caller-supplied reading or
            # match list can substitute for them, and no second matcher, solver
            # or signature pass is made here.
            active_edge_aliases = _failed_active_edges(
                readings, matches, solution,
                max_readings=alias_max_readings,
                max_matches=alias_max_matches,
                max_active_edges=alias_max_active_edges,
                max_endpoint_alternatives=alias_max_endpoint_alternatives,
                max_frame_id_bytes=alias_max_frame_id_bytes,
                max_record_bytes=alias_max_record_bytes)
            # DS-R5: the boundary attestation is counted from the SAME local
            # post-filter matches and the matcher's own diagnostic trace. It is
            # evaluation-only and lives inside this sink block, so the default
            # production path never counts, allocates or serializes it, and the
            # returned graph result and digest stay byte-for-byte unchanged.
            boundary_saturation = _boundary_saturation_attestation(
                matches, diagnostic_sink, max_shift_cells)
            # DS-IMG-GEOM-PHASE-CAPTURE: the per-face phase capture is derived
            # from the same local readings/matches and the same returned
            # solution, inside this failure-only sink block, so the default
            # production path never allocates or serializes it and the returned
            # graph result and digest stay byte-for-byte unchanged.
            phase_capture = failed_phase_capture(readings, matches, solution)
            # DS-R7: the complete solver-input binding record is derived from the
            # same local readings/matches and the same returned solution, inside
            # this failure-only sink block, so the default production path never
            # allocates or serializes it and the returned graph result and digest
            # stay byte-for-byte unchanged.
            solver_input_bindings = failed_solver_input_bindings(
                readings, matches, solution)
            # DS-IMG-SOLVER-CONTEXT-CAPTURE-DIAGNOSTIC-20261004: capture the
            # exact effective solve-call arguments -- the three numeric thresholds
            # and the exact immutable ``quadrant_constraints`` tuple (including an
            # empty tuple) -- as a versioned, digest-bound record. It is built
            # only here, inside the failure-observer block, from the same bound
            # values the call above received, so the default path never allocates
            # or serializes it and the returned result and digests are unchanged.
            # A malformed value yields a bounded refused capture rather than
            # disturbing default result derivation.
            #
            # Lead correction 2026-10-04: a build failure must never become
            # ``solve_context=None``, because an absent key is indistinguishable
            # from the historical no-context artifact and would silently hide a
            # failed capture. Fail closed with an explicit bounded refusal bound
            # to the same call arguments and frame universe; the observer run is
            # the only path here, so the default result stays output-neutral.
            try:
                solve_context = failed_solve_context(
                    effective_integer_tolerance, effective_min_resultant,
                    effective_ambiguous_margin, effective_constraints,
                    solution.frames)
            except _ERRORS:
                solve_context = refused_solve_context(
                    effective_integer_tolerance, effective_min_resultant,
                    effective_ambiguous_margin, solution.frames)
            solver_failure_sink(_solver_failure_snapshot(
                solution, active_edge_aliases=active_edge_aliases,
                boundary_saturation=boundary_saturation,
                phase_capture=phase_capture,
                solver_input_bindings=solver_input_bindings,
                solve_context=solve_context))
        if same_axis_candidate_sink is not None:
            # DS-IMG-SAMEAXIS-PREFILTER-CAPTURE-R9: the pre-filter candidate
            # capture is derived from the same local readings/matches, the same
            # returned solution, and the raw matcher observation collected only
            # when this sink is attached. It is evaluation-only, so the default
            # production path never collects, allocates or serializes it and the
            # returned graph result and digest stay byte-for-byte unchanged. No
            # second matcher, filter, solver or signature pass is made here.
            same_axis_candidate_sink(failed_same_axis_candidates(
                readings, matches, solution,
                raw_observations[0] if raw_observations else None,
                (SAME_AXIS_CANDIDATES_AXES_V1
                 if same_axis_capture_axes is None else same_axis_capture_axes)))
        if candidate_support_sink is not None or complete_support_sink is not None:
            # DS-IMG-CANDIDATE-SUPPORT-PROVENANCE-IMPL-R4 (D): the strict,
            # fail-closed candidate-support record over the complete pre-filter
            # admitted universe, built only in this failure-observer block from
            # the already-collected pre-filter witness rows, signature-tap
            # witnesses and same-axis observation. It re-matches, re-signatures
            # and re-solves nothing, and the default path never collects it.
            # The per-frame ingress pins come from the accepted session binding
            # itself, so the record can only ever bind the pins the session
            # digest carries.
            ingress_records = tuple(
                frame.ingress for frame in accepted_session.frames
                if getattr(frame, "ingress", None) is not None)
            if candidate_support_sink is not None:
                candidate_support_sink(
                    candidate_support_mod.failed_candidate_support_provenance(
                        readings, matches, signature_witnesses or (),
                        pre_filter_rows or (),
                        raw_observations[0] if raw_observations else None,
                        ingress=ingress_records,
                        failed_assignment_sha256=_failed_assignment_digest(
                            tuple(solution.frames), dict(solution.quadrants))))
            if complete_support_sink is not None:
                # DS-R4-V2-ROUTEB-DIAGNOSTIC-WIRING: the complete V2 record over
                # the same accepted session ingress pins, full pre-filter witness
                # rows, signature witnesses and raw observation, under exactly the
                # frozen Route B diagnostic profile. Collected only here, inside
                # the failure-observer block, from material the ordinary path
                # already produced (no second match, filter, signature or solve),
                # and passed to the caller's evaluation-only sink.
                complete_support_sink(
                    complete_support_mod.collect_route_b_diagnostic_support(
                        readings, matches, signature_witnesses or (),
                        pre_filter_rows or (),
                        raw_observations[0] if raw_observations else None,
                        ingress=ingress_records,
                        failed_assignment_sha256=_failed_assignment_digest(
                            tuple(solution.frames), dict(solution.quadrants))))
        return _unavailable(accepted_session, lattice, signatures,
                            solution.reason, "component solver abstained",
                            max_shift_cells)
    offender = _non_finite_solution_field(solution)
    if offender is not None:
        return _unavailable(
            accepted_session, lattice, signatures, NON_FINITE_SOLVER_OUTPUT,
            f"component solution field is not finite: {offender}",
            max_shift_cells)
    production_refusal = _production_solution_refusal(
        frame_ids, readings, matches, solution)
    if production_refusal is not None:
        reason, detail = production_refusal
        return _unavailable(accepted_session, lattice, signatures,
                            reason, detail, max_shift_cells)
    return HistoricalImageDerivedGraphResult(
        HISTORICAL_IMAGE_DERIVED_GRAPH_SCHEMA, HISTORICAL_IMAGE_DERIVED_GRAPH_PRODUCER, GRAPH_TERMINAL,
        "assembled", "", "", accepted_session.session_sha256,
        lattice.result_sha256, signatures.result_sha256, frame_ids,
        readings, evidence, tuple(matches), solution,
        plane_graph.PLANE_CORRELATION_ALGORITHM, max_shift_cells)


def derive_image_derived_graph(
    *, session: object, lattice_result: object, signatures_result: object,
    connection_graph: object = None,
    max_shift_cells: int = 32, integer_tolerance: float = 0.15,
    min_resultant: float = 0.90, ambiguous_margin: float = 0.02,
) -> ImageDerivedGraphResult | placement.PlacementRefusal:
    """Production G: joint endpoint selection before one fixed geometry pass."""
    from mc_photogrammetry.pipeline import image_derived_joint_graph
    return image_derived_joint_graph.derive_image_derived_graph(
        session=session, lattice_result=lattice_result,
        signatures_result=signatures_result, connection_graph=connection_graph,
        max_shift_cells=max_shift_cells, integer_tolerance=integer_tolerance,
        min_resultant=min_resultant, ambiguous_margin=ambiguous_margin)

# Public producer7 identity/type are resolved lazily to avoid an import cycle:
# the joint implementation imports these historical diagnostic helpers.
IMAGE_DERIVED_GRAPH_SCHEMA = "mc-photogrammetry-image-derived-graph-v3"
IMAGE_DERIVED_GRAPH_PRODUCER = "m9-g-image-derived-graph-7"
__all__ += ["IMAGE_DERIVED_GRAPH_SCHEMA", "IMAGE_DERIVED_GRAPH_PRODUCER",
            "ImageDerivedGraphResult", "derive_historical_image_derived_graph"]


def __getattr__(name):
    if name == "ImageDerivedGraphResult":
        from mc_photogrammetry.pipeline.image_derived_joint_graph import ImageDerivedGraphResult
        return ImageDerivedGraphResult
    raise AttributeError(name)


def _derive_image_derived_graph(*args, **kwargs):
    raise ValueError("Historical graph derivation requires derive_historical_image_derived_graph")
