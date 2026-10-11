"""Output-neutral attribution captured by the existing G-stage execution."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as policy_mod
from mc_photogrammetry.reconstruction import plane_graph

HISTORICAL_IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA = (
    "mc-photogrammetry-image-derived-graph-attribution-v4")
IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA = (
    "mc-photogrammetry-image-derived-graph-attribution-v5")
SOLVER_ID = "plane_graph._evaluate_assignment"
PREDICATE_BASIS = "derived_from_returned_solution"
INVALID_DIAGNOSTIC = "invalid_solver_failure_diagnostic"
INVALID_EFFECTIVE_SETTING = "invalid_effective_setting"
SOLVER_FAILURE_OBSERVATION_SCHEMA = (
    "mc-photogrammetry-evaluation-solver-failure-observation-v1")


def _pair_record(pair: plane_graph.PlanePairTrace) -> dict:
    """Serialize one pair trace, tagging the positive-infinity PSR sentinel.

    Only the existing ``positive_infinity`` token is substituted; finite values
    and ``None`` stay untouched, and NaN/negative infinity remain rejected by
    the strict ``allow_nan=False`` digest encoding.
    """
    record = asdict(pair)
    if record["psr"] == float("inf"):
        record["psr"] = "positive_infinity"
    return record


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


def _effective_settings(
    values: dict[str, object],
) -> tuple[tuple[tuple[str, float | int | None], ...],
           tuple[tuple[str, str], ...]]:
    """Sanitize serialized settings and report bounded invalid fields.

    A finite ``int``/``float`` is retained exactly. Any other value -- NaN,
    either infinity, an integer too large to represent as a float, a bool or a
    non-number -- becomes ``null`` under its own key, never a substituted
    default or an infinity sentinel, and its field name plus concrete Python
    type is recorded so the bounded reason stays bound into the digest.
    """
    settings: list[tuple[str, float | int | None]] = []
    invalid: list[tuple[str, str]] = []
    for name in sorted(values):
        value = values[name]
        if _finite_number(value):
            settings.append((name, value))  # type: ignore[arg-type]
        else:
            settings.append((name, None))
            invalid.append((name, type(value).__name__))
    return tuple(settings), tuple(invalid)


@dataclass(frozen=True, slots=True)
class SolverFailureDiagnostic:
    """One bounded evaluation-only view of a solver abstention.

    A recorded diagnostic reproduces the plane-integer gate the active solver
    expression applies. A refused diagnostic means the returned values could not
    support that derivation; it carries only a bounded reason/field and never an
    invalid numeric payload, and it never alters the graph result.

    ``active_edge_aliases`` is the optional v4 nested record of the failed best
    assignment's active edges. The graph seam derives it whenever the evaluation
    failure sink is attached -- which is the ordinary attribution/contained path
    -- and it is either a complete bounded record or an atomic typed refusal of
    its own.

    ``boundary_saturation`` is the optional DS-R5 nested failure-only count of
    admitted and same-axis-kept shifts that pin the exact search bound. It is
    derived by the same failure sink from the matcher's own diagnostic trace and
    the local post-filter matches, and it is likewise either complete or one
    atomic bounded refusal.
    """

    solver_id: str
    recorded: bool
    reason: str
    matched_axes: tuple[int, ...] = ()
    axes: tuple[tuple[int, float | None, float | None, int | None], ...] = ()
    predicate_basis: str | None = None
    first_failed_predicate: str | None = None
    resultant_below_min_axes: tuple[int, ...] = ()
    residual_above_tolerance_axes: tuple[int, ...] = ()
    residual_predicate_evaluated: bool | None = None
    field: str | None = None
    active_edge_aliases: graph_mod.FailedActiveEdgeAliases | None = None
    boundary_saturation: graph_mod.BoundarySaturationAttestation | None = None

    def to_record(self) -> dict:
        if not self.recorded:
            return {"solver_id": self.solver_id, "refused": True,
                    "reason": self.reason, "field": self.field}
        record = {
            "solver_id": self.solver_id,
            "reason": self.reason,
            "matched_axes": list(self.matched_axes),
            "axes": [list(row) for row in self.axes],
            "predicate_basis": self.predicate_basis,
            "first_failed_predicate": self.first_failed_predicate,
            "resultant_below_min_axes": list(self.resultant_below_min_axes),
            "residual_above_tolerance_axes": list(
                self.residual_above_tolerance_axes),
            "residual_predicate_evaluated": self.residual_predicate_evaluated,
        }
        if self.active_edge_aliases is not None:
            record["active_edge_aliases"] = self.active_edge_aliases.to_record()
        if self.boundary_saturation is not None:
            record["boundary_saturation"] = self.boundary_saturation.to_record()
        return record


def _refused_diagnostic(field: str) -> SolverFailureDiagnostic:
    return SolverFailureDiagnostic(
        SOLVER_ID, False, INVALID_DIAGNOSTIC, field=field)


def _raw_snapshot_scalar(value: object) -> object:
    """Keep one exact JSON scalar; tag anything else with its Python type.

    ``None``, ``bool``, ``str`` and ``int`` are JSON-native and are retained
    verbatim. A finite Python ``float`` is retained exactly. NaN, either
    infinity, a numpy scalar and every other object become a bounded
    ``unavailable`` token instead of being clamped, dropped or serialized as a
    non-finite value.
    """
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return {"unavailable": INVALID_DIAGNOSTIC, "type": type(value).__name__}


def _raw_snapshot_sequence(value: object) -> list | None:
    """Serialize one raw tuple without dropping or substituting an entry."""
    if value is None:
        return None
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return [{"unavailable": INVALID_DIAGNOSTIC,
                 "type": type(value).__name__}]
    return [_raw_snapshot_scalar(item) for item in items]


def _raw_snapshot_axis_pairs(items: object) -> list:
    """Serialize one raw axis mapping, preserving partial or malformed keys.

    Unlike the plane-integer projection, this never refuses a partial map and
    never borrows the solver alternative's invented ``0.0``: every pair the
    snapshot actually carried is retained once, in order.
    """
    try:
        pairs = tuple(items)  # type: ignore[arg-type]
    except TypeError:
        return [{"axis": None, "value": {"unavailable": INVALID_DIAGNOSTIC,
                                         "type": type(items).__name__}}]
    rows: list = []
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            rows.append({"axis": None,
                         "value": {"unavailable": INVALID_DIAGNOSTIC,
                                   "type": type(pair).__name__}})
            continue
        key, value = pair
        axis = key if type(key) is int else None
        row: dict = {"axis": axis, "value": _raw_snapshot_scalar(value)}
        if axis is None:
            row["axis_type"] = type(key).__name__
        rows.append(row)
    return rows


def solver_failure_snapshot_record(
    snapshot: graph_mod.SolverFailureSnapshot,
) -> dict:
    """Serialize the exact raw failure snapshot for an evaluation observer.

    This is the unprojected companion of :func:`solver_failure_diagnostic`. It
    retains the solver's actual per-axis resultants, residuals and observation
    counts even where the plane-integer projection must refuse a partial axis
    map, and it never substitutes a zero. Values that are not finite JSON
    numbers are tagged with their Python type rather than dropped, clamped or
    made non-finite. It is evaluation-only, is never reached by the default
    path, and does not alter the returned graph result or its digests.

    When the failure snapshot carries the DS-IMG-SOLVER-CONTEXT-CAPTURE
    ``solve_context`` capture, its own digest-bound record is mirrored under
    ``solve_context``; the key is omitted entirely for a snapshot without it, so
    historical artifacts and the default path keep the exact previous shape.
    """
    reason: object = snapshot.reason
    if type(reason) is not str:
        reason = {"unavailable": INVALID_DIAGNOSTIC,
                  "type": type(reason).__name__}
    invalid_field = (snapshot.invalid_field
                     if type(snapshot.invalid_field) is str else None)
    record = {
        "schema": SOLVER_FAILURE_OBSERVATION_SCHEMA,
        "solver_id": SOLVER_ID,
        "reason": reason,
        "frames": _raw_snapshot_sequence(snapshot.frames),
        "quadrants": _raw_snapshot_sequence(snapshot.quadrants),
        "axis_resultants": _raw_snapshot_axis_pairs(snapshot.axis_resultants),
        "axis_residuals": _raw_snapshot_axis_pairs(snapshot.axis_residuals),
        "best_alternative_quadrants": _raw_snapshot_sequence(
            snapshot.best_alternative_quadrants),
        "best_alternative_axis_observations": _raw_snapshot_sequence(
            snapshot.best_alternative_axis_observations),
        "invalid_field": invalid_field,
    }
    # DS-IMG-SOLVER-CONTEXT-CAPTURE-DIAGNOSTIC-20261004: mirror the optional
    # exact effective solve-call capture the same failure snapshot already
    # carries. It is present only when the graph seam built it (the failure
    # observer was attached) and is the record's own digest-bound serialization,
    # never a re-derivation from the projected diagnostic; it stays absent for
    # historical artifacts and for the default production path.
    solve_context = getattr(snapshot, "solve_context", None)
    if solve_context is not None:
        record["solve_context"] = solve_context.to_record()
    return record


def _axis_metric_map(items: object, field: str
                     ) -> tuple[dict[int, float] | None, str | None]:
    """Validate one raw axis mapping into finite ``int -> float`` entries."""
    try:
        pairs = tuple(items)  # type: ignore[arg-type]
    except TypeError:
        return None, f"{field}.key"
    metrics: dict[int, float] = {}
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            return None, f"{field}.key"
        key, value = pair
        if type(key) is not int or key not in (0, 1, 2) or key in metrics:
            return None, f"{field}.key"
        if not _finite_number(value):
            return None, f"{field}[{key}]"
        metrics[key] = float(value)
    return metrics, None


def solver_failure_diagnostic(
    snapshot: graph_mod.SolverFailureSnapshot,
    min_resultant: object,
    integer_tolerance: object,
    ambiguous_margin: object = 0.02,
) -> SolverFailureDiagnostic:
    """Derive the bounded ``solver_failure`` record, refusing bad values.

    Axis rows always cover the fixed order ``0, 1, 2``; an axis the returned
    maps do not carry stays ``null`` rather than borrowing the solver
    alternative's invented ``0.0``. Observations come only from a best
    alternative whose identity agrees with the returned quadrants. Every
    numeric threshold this record compares -- ``min_resultant``,
    ``integer_tolerance`` and ``ambiguous_margin`` -- is validated first, so an
    invalid effective setting cannot reach strict serialization. The v4
    ``active_edge_aliases`` record and the DS-R5 ``boundary_saturation``
    attestation carried by the failure snapshot are forwarded verbatim,
    including their own atomic refusals.
    """
    diagnostic = _solver_failure_metrics(
        snapshot, min_resultant, integer_tolerance, ambiguous_margin)
    if diagnostic.active_edge_aliases is None and snapshot.active_edge_aliases is not None:
        object.__setattr__(diagnostic, "active_edge_aliases",
                           snapshot.active_edge_aliases)
    if (diagnostic.boundary_saturation is None
            and getattr(snapshot, "boundary_saturation", None) is not None):
        object.__setattr__(diagnostic, "boundary_saturation",
                           snapshot.boundary_saturation)
    return diagnostic


def _solver_failure_metrics(
    snapshot: graph_mod.SolverFailureSnapshot,
    min_resultant: object,
    integer_tolerance: object,
    ambiguous_margin: object = 0.02,
) -> SolverFailureDiagnostic:
    """The v3 metric/predicate projection of one failure snapshot."""
    if type(snapshot.reason) is not str:
        return _refused_diagnostic("reason")
    if snapshot.invalid_field is not None:
        return _refused_diagnostic(snapshot.invalid_field)
    if not _finite_number(min_resultant):
        return _refused_diagnostic("effective_settings.min_resultant")
    if not _finite_number(integer_tolerance):
        return _refused_diagnostic("effective_settings.integer_tolerance")
    if not _finite_number(ambiguous_margin):
        return _refused_diagnostic("effective_settings.ambiguous_margin")
    resultants, field = _axis_metric_map(snapshot.axis_resultants, "axis_resultants")
    if resultants is None:
        return _refused_diagnostic(field or "axis_resultants")
    residuals, field = _axis_metric_map(snapshot.axis_residuals, "axis_residuals")
    if residuals is None:
        return _refused_diagnostic(field or "axis_residuals")
    if set(resultants) != set(residuals):
        return _refused_diagnostic("axis_coverage")
    matched_axes = tuple(sorted(resultants))

    observations = None
    best_quadrants = snapshot.best_alternative_quadrants
    best_counts = snapshot.best_alternative_axis_observations
    if (best_quadrants is not None and best_counts is not None
            and tuple(best_quadrants) == tuple(snapshot.quadrants)):
        observations = tuple(best_counts)
        if (len(observations) != 3
                or any(type(value) is not int or value < 0
                       for value in observations)):
            return _refused_diagnostic("alternatives[0].axis_observations")

    rows = tuple(
        (axis, resultants.get(axis), residuals.get(axis),
         None if observations is None or axis not in resultants
         else observations[axis])
        for axis in range(3))

    if snapshot.reason != "plane_integer_residual":
        return SolverFailureDiagnostic(
            SOLVER_ID, True, snapshot.reason, matched_axes, rows,
            residual_predicate_evaluated=False)
    if matched_axes != (0, 1, 2):
        return _refused_diagnostic("plane_integer_residual")
    below = tuple(axis for axis in matched_axes
                  if resultants[axis] < min_resultant)
    above = tuple(axis for axis in matched_axes
                  if residuals[axis] > integer_tolerance)
    if not below and not above:
        return _refused_diagnostic("plane_integer_residual")
    return SolverFailureDiagnostic(
        SOLVER_ID, True, snapshot.reason, matched_axes, rows, PREDICATE_BASIS,
        "minimum_resultant" if below else "integer_tolerance", below, above,
        not below)


@dataclass(frozen=True, slots=True)
class ImageDerivedGraphAttribution:
    """Strict, digest-bound evaluation record for one graph derivation.

    ``effective_settings`` retains each finite numeric setting exactly and
    stores ``null`` for any invalid value instead of the raw NaN/infinity or an
    unrepresentable integer. ``invalid_settings`` carries the bounded
    ``(field, type)`` reasons for those nulls; it is omitted entirely when every
    setting is valid. ``solver_failure`` is the optional failure-only diagnostic
    described above, and it carries the v4 ``active_edge_aliases`` record when
    the failure snapshot provided one.
    """

    schema: str
    algorithm_id: str
    graph_result_sha256: str
    session_sha256: str
    lattice_sha256: str
    signatures_sha256: str
    policy_sha256: str
    effective_settings: tuple[tuple[str, float | int | None], ...]
    first_failing_substage: str
    reason: str
    detail: str
    pre_admission_candidates: int
    post_admission_candidates: int
    post_same_axis_candidates: int
    pair_attribution: tuple[plane_graph.PlanePairTrace, ...]
    selected_quadrants: tuple[tuple[str, int], ...]
    selected_scale: tuple[tuple[str, int, float], ...]
    axis_diagnostics: tuple[tuple[int, float | None, float | None], ...]
    solver_failure: SolverFailureDiagnostic | None = None
    invalid_settings: tuple[tuple[str, str], ...] = ()
    attribution_sha256: str = ""
    matcher_admitted_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema not in (HISTORICAL_IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA,
                               IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA):
            raise ValueError("graph attribution schema is unsupported")
        if self.schema == IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA and (
                type(self.matcher_admitted_sha256) is not str
                or len(self.matcher_admitted_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.matcher_admitted_sha256)):
            raise ValueError("matcher admitted ledger digest is malformed")
        payload = self.to_record(include_digest=False)
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        if self.attribution_sha256 not in ("", digest):
            raise ValueError("graph attribution digest mismatch")
        object.__setattr__(self, "attribution_sha256", digest)

    def to_record(self, *, include_digest: bool = True) -> dict:
        record = {
            "schema": self.schema,
            "algorithm_id": self.algorithm_id,
            "graph_result_sha256": self.graph_result_sha256,
            "source_digests": {
                "session": self.session_sha256, "lattice": self.lattice_sha256,
                "signatures": self.signatures_sha256, "policy": self.policy_sha256},
            "effective_settings": dict(self.effective_settings),
            "first_failing_substage": self.first_failing_substage,
            "reason": self.reason, "detail": self.detail,
            "candidate_counts": {
                "pre_admission": self.pre_admission_candidates,
                "post_admission": self.post_admission_candidates,
                "post_same_axis": self.post_same_axis_candidates},
            "pairs": [_pair_record(value) for value in self.pair_attribution],
            "selected_quadrants": [list(value) for value in self.selected_quadrants],
            "selected_scale": [list(value) for value in self.selected_scale],
            "axis_diagnostics": [list(value) for value in self.axis_diagnostics],
        }
        if self.invalid_settings:
            record["invalid_settings"] = [
                {"field": field, "reason": INVALID_EFFECTIVE_SETTING,
                 "type": type_name}
                for field, type_name in self.invalid_settings]
        if self.solver_failure is not None:
            record["solver_failure"] = self.solver_failure.to_record()
        if self.schema == IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA:
            record["matcher_admitted_sha256"] = self.matcher_admitted_sha256
        if include_digest:
            record["attribution_sha256"] = self.attribution_sha256
        return record


def derive_image_derived_graph_attribution(*, matcher_witness_sink=None,
                                           matcher_witness_payload_sink=None,
                                           witness_budget=None,
                                           solver_failure_observer=None,
                                           **kwargs):
    """Run G once and return its unchanged result plus a nonserialized trace.

    The failure-only alias diagnostic is derived by the graph seam itself from
    the exact local readings and post-filter matches it hands to the solver, so
    this adapter must never accept a caller-supplied reading or match list: a
    foreign override attempt is rejected outright rather than being able to
    describe the graph under another graph's provenance.

    ``solver_failure_observer`` is an optional evaluation-only callback. When
    supplied and the component solver abstains, it is called once with the exact
    in-memory :class:`graph_mod.SolverFailureSnapshot` that the same graph seam
    produced, after the ordinary projection. It is never forwarded to the graph
    derivation, so the returned result, the attribution record and every digest
    stay unchanged whether or not an observer is attached; the callback is
    responsible for serializing (see :func:`solver_failure_snapshot_record`).
    """
    for name in ("readings", "matches", "alias_readings", "alias_matches"):
        if name in kwargs:
            raise TypeError(
                f"graph attribution does not accept caller-supplied {name!r}")
    trace = plane_graph.PlaneGraphTrace([])
    raw_settings = {
        "max_shift_cells": kwargs.get("max_shift_cells", 32),
        "integer_tolerance": kwargs.get("integer_tolerance", 0.15),
        "min_resultant": kwargs.get("min_resultant", 0.90),
        "ambiguous_margin": kwargs.get("ambiguous_margin", 0.02),
        "min_overlap_cells": plane_graph.MIN_OVERLAP_CELLS,
        "min_structural_share": plane_graph.MIN_STRUCTURAL_SHARE,
    }
    settings, invalid_settings = _effective_settings(raw_settings)
    captured: list[graph_mod.SolverFailureSnapshot] = []
    from mc_photogrammetry.pipeline import image_derived_joint_graph as joint_mod
    result = joint_mod.derive_image_derived_graph(
        **kwargs, diagnostic_sink=trace,
        matcher_witness_sink=matcher_witness_sink,
        matcher_witness_payload_sink=matcher_witness_payload_sink,
        witness_budget=witness_budget,
        solver_failure_sink=captured.append)
    if not isinstance(result, joint_mod.ImageDerivedGraphResult):
        return result, None
    solution = result.component_solution
    assembled = result.status == "assembled" and solution is not None
    stage = "complete" if assembled else (
        "joint_selection" if result.geometry_evaluations == 0 else
        "admission" if trace.post_admission_candidates == 0 else
        "same_axis_filter" if trace.post_same_axis_candidates == 0 else "solver")
    solver_failure = None
    if captured:
        solver_failure = solver_failure_diagnostic(
            captured[0], raw_settings["min_resultant"],
            raw_settings["integer_tolerance"],
            raw_settings["ambiguous_margin"])
        if solver_failure_observer is not None:
            solver_failure_observer(captured[0])
    attribution = ImageDerivedGraphAttribution(
        (IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA
         if trace.admitted_ledger_sha256 else
         HISTORICAL_IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA),
        plane_graph.PLANE_CORRELATION_ALGORITHM, result.result_sha256,
        result.session_sha256, result.lattice_sha256, result.signatures_sha256,
        policy_mod.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.policy_sha256,
        settings, stage, result.reason, result.detail,
        trace.pre_admission_candidates, trace.post_admission_candidates,
        trace.post_same_axis_candidates, tuple(trace.pairs),
        () if solution is None else tuple(sorted(solution.quadrants.items())),
        () if solution is None else tuple(
            (frame, index, value) for (frame, index), value in
            sorted(solution.blocks_per_period.items())),
        () if solution is None else tuple(
            (axis, solution.axis_resultants.get(axis),
             solution.axis_residuals.get(axis)) for axis in range(3)),
        solver_failure, invalid_settings,
        matcher_admitted_sha256=trace.admitted_ledger_sha256)
    return result, attribution
