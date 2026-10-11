"""Production graph: joint endpoint selection followed by one fixed-Q evaluation.

Complete matcher candidates and the selection certificate remain distinct from
the representative numerical constraints. Explicit source replay is the sole
way to re-establish source-backed uniqueness; record hashes alone cannot do so.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import weakref

from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.pipeline import image_derived_graph_v3 as phase_b
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import plane_graph, plane_placement as placement

IMAGE_DERIVED_GRAPH_SCHEMA = "mc-photogrammetry-image-derived-graph-v6"
IMAGE_DERIVED_GRAPH_PRODUCER = "m9-g-image-derived-graph-10"
GRAPH_TERMINAL = "geometry_unavailable"
_ERRORS = phase_b._ERRORS
DEFAULT_GEOMETRY_SETTINGS = (0.15, 0.90, 0.02)
_MAX_LIVE_RECEIPTS = 64
_LIVE_RECEIPT_SECONDS = 900.0
_LIVE_RECEIPTS = {}
_LIVE_LOCK = threading.RLock()


def _expire_receipts(now):
    for key, (reference, _digest, expires) in tuple(_LIVE_RECEIPTS.items()):
        if reference() is None or now >= expires:
            _LIVE_RECEIPTS.pop(key, None)


def _register_live(graph):
    """Private bounded capability; no receipt is serialized in a graph."""
    with _LIVE_LOCK:
        now = time.monotonic()
        _expire_receipts(now)
        key = id(graph)
        if key not in _LIVE_RECEIPTS and len(_LIVE_RECEIPTS) >= _MAX_LIVE_RECEIPTS:
            raise ValueError("live graph receipt capacity exhausted")
        def released(reference):
            with _LIVE_LOCK:
                current = _LIVE_RECEIPTS.get(key)
                if current is not None and current[0] is reference:
                    _LIVE_RECEIPTS.pop(key, None)
        _LIVE_RECEIPTS[key] = (weakref.ref(graph, released), graph.result_sha256,
                               now + _LIVE_RECEIPT_SECONDS)
    return graph


def _require_live(graph):
    with _LIVE_LOCK:
        now = time.monotonic()
        _expire_receipts(now)
        receipt = _LIVE_RECEIPTS.get(id(graph))
        if (receipt is None or receipt[0]() is not graph
                or receipt[1] != graph.result_sha256):
            raise ValueError("graph requires validated source derivation or explicit replay")
    # Digest-bound constructors check nested mutations; they never register a copy.
    ImageDerivedGraphResult(**legacy._fields(graph))
    return graph


def _payload(value):
    return {"schema": value.schema, "producer": value.producer,
            "terminal": value.terminal, "status": value.status,
            "reason": value.reason, "detail": value.detail,
            "session_sha256": value.session_sha256, "lattice_sha256": value.lattice_sha256,
            "signatures_sha256": value.signatures_sha256, "connection_sha256": value.connection_sha256,
            "frame_ids": value.frame_ids, "selection_sha256": value.selection.result_sha256,
            "signature_digests": [s.signature_sha256 for s in value.signatures],
            "candidate_matches": [legacy._match_payload(m) for m in value.candidate_matches],
            "matches": [legacy._match_payload(m) for m in value.matches],
            "component_solution": legacy._solution_payload(value.component_solution),
            "correlation_algorithm": value.correlation_algorithm, "max_shift_cells": value.max_shift_cells,
            "geometry_settings": value.geometry_settings, "geometry_evaluations": value.geometry_evaluations}


class _WeakGraph:
    __slots__ = ("__weakref__",)


@dataclass(frozen=True, slots=True)
class ImageDerivedGraphResult(_WeakGraph):
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    session_sha256: str
    lattice_sha256: str
    signatures_sha256: str
    connection_sha256: str
    frame_ids: tuple
    readings: tuple
    signatures: tuple
    candidate_matches: tuple
    selection: selection.EndpointSelectionCertificate
    matches: tuple = ()
    component_solution: plane_graph.ComponentSolution | None = None
    correlation_algorithm: str = plane_graph.PLANE_CORRELATION_ALGORITHM
    max_shift_cells: int = 32
    geometry_settings: tuple = DEFAULT_GEOMETRY_SETTINGS
    geometry_evaluations: int = 0
    result_sha256: str = ""

    def __post_init__(self):
        if (self.schema != IMAGE_DERIVED_GRAPH_SCHEMA or self.producer != IMAGE_DERIVED_GRAPH_PRODUCER
                or self.terminal != GRAPH_TERMINAL or self.status not in ("assembled", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or type(self.matches) is not tuple
                or self.correlation_algorithm != plane_graph.PLANE_CORRELATION_ALGORITHM
                or type(self.max_shift_cells) is not int or self.max_shift_cells != 32
                or type(self.selection) is not selection.EndpointSelectionCertificate
                or type(self.geometry_settings) is not tuple or self.geometry_settings != DEFAULT_GEOMETRY_SETTINGS
                or any(type(v) is not float for v in self.geometry_settings)
                or type(self.geometry_evaluations) is not int or self.geometry_evaluations not in (0, 1)):
            raise ValueError("production joint graph identity/settings malformed")
        selection._frames(self.frame_ids)
        selection.validate_candidate_certificate(self.selection, self.readings, self.signatures, self.candidate_matches)
        cert = self.selection
        if (self.session_sha256 != cert.session_sha256 or self.lattice_sha256 != cert.lattice_sha256
                or self.signatures_sha256 != cert.signatures_sha256 or self.connection_sha256 != cert.connection_sha256
                or self.frame_ids != cert.frame_ids):
            raise ValueError("production joint graph source bindings differ")
        selected = tuple(r.match for r in cert.representatives)
        if cert.status != "unique":
            if self.geometry_evaluations or self.reason != cert.reason or self.detail != cert.status:
                raise ValueError("ineligible selection reached geometry")
        elif self.geometry_evaluations != 1:
            raise ValueError("unique selection must be evaluated exactly once")
        if self.status == "assembled":
            solution = self.component_solution
            if (cert.status != "unique" or self.reason or self.detail or not selected
                    or self.matches != selected or type(solution) is not plane_graph.ComponentSolution
                    or solution.reason or not solution.scale_known):
                raise ValueError("assembled joint graph is inconsistent")
            proof = cert.proof_record()
            if dict(proof["choices"][0]["quadrants"]) != solution.quadrants:
                raise ValueError("solution quadrants differ from unique certificate")
            if legacy._non_finite_solution_field(solution) is not None:
                raise ValueError("component is nonfinite")
            indexed = plane_graph._canonical_reading_index(self.readings)[0]
            selected_indices = {(f, i): a for f, i, a in proof["choices"][0]["alternatives"]}
            if plane_graph.physical_displacement_refusal(indexed, self.matches, solution.quadrants,
                                                         selected_indices, solution.translations):
                raise ValueError("component physical displacement does not replay")
            if legacy._production_solution_refusal(self.frame_ids, self.readings, self.matches, solution) is not None:
                raise ValueError("component production constraints do not replay")
            if (set(solution.axis_resultants) != {0, 1, 2}
                    or any(solution.axis_resultants[a] < 0.90 or solution.axis_residuals.get(a, float("inf")) > 0.15
                           for a in range(3))):
                raise ValueError("component numerical gates fail")
        elif not self.reason or self.matches or self.component_solution is not None:
            raise ValueError("unavailable joint graph must be atomic")
        digest = selection._sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("production joint graph digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _evaluate_selected(readings, matches, cert):
    """No quadrant search: reuse the existing numerical kernel and exact gates."""
    selection.EndpointSelectionCertificate(**legacy._fields(cert))
    if cert.status != "unique" or matches != tuple(r.match for r in cert.representatives):
        raise ValueError("geometry requires exhaustive unique eligible selection")
    proof = cert.proof_record()
    if not proof["exhaustive"] or len(proof["choices"]) != 1 or proof["evidence_ambiguous"]:
        raise ValueError("geometry requires exhaustive unambiguous certificate")
    indexed, _, frames = plane_graph._canonical_reading_index(readings)
    q = dict(proof["choices"][0]["quadrants"])
    if plane_graph._assignment_endpoint_period_conflict(q, matches, indexed):
        raise ValueError("representatives contradict unique endpoints")
    # This is the only numerical evaluator call. No fallback or reconsideration.
    evaluation = plane_graph._evaluate_assignment(frames, q, matches, indexed)
    if evaluation is None:
        return plane_graph._component("no_matches", frames)
    _, alternative, detail = evaluation
    base = dict(quadrants=q, translations=detail["translations"], residuals=detail["residuals"],
                resultants=detail["resultants"], blocks_per_period=detail["blocks_per_period"],
                alternatives=(alternative,))
    selected_indices = {(f, i): a for f, i, a in proof["choices"][0]["alternatives"]}
    refusal = plane_graph.physical_displacement_refusal(indexed, matches, q, selected_indices,
                                                        detail["translations"])
    if refusal:
        return plane_graph._component(refusal, frames, **base)
    if detail["duplicate_b"]:
        return plane_graph._component("b_unresolved", frames, **base)
    axes = sorted(detail["resultants"])
    if len(axes) < 3:
        return plane_graph._component("insufficient_axis_coverage", frames, **base)
    if (any(detail["resultants"][a] < 0.90 for a in axes)
            or any(detail["residuals"][a] > 0.15 for a in axes)):
        return plane_graph._component("plane_integer_residual", frames, **base)
    return plane_graph._component("", frames, scale_known=len(axes) >= 2, **base)


@dataclass(frozen=True, slots=True)
class _SelectionEvaluation:
    reason: str
    detail: str
    selected: tuple
    evaluated: plane_graph.ComponentSolution | None
    matches: tuple
    solution: plane_graph.ComponentSolution | None
    calls: int


def _evaluate_selection(*, readings, signatures, candidate_matches, frame_ids, certificate):
    """Reuse the production post-selection gates without matching or publication."""
    cert = certificate
    reason, detail, matches, solution, calls = cert.reason, cert.status, (), None, 0
    selected, evaluated = (), None
    if cert.status == "unique":
        selection.validate_candidate_certificate(cert, readings, signatures, candidate_matches)
        selected = tuple(r.match for r in cert.representatives)
        calls = 1
        evaluated = _evaluate_selected(readings, selected, cert)
        if evaluated.reason:
            reason, detail = evaluated.reason, "selected geometry abstained; no reconsideration"
        elif legacy._non_finite_solution_field(evaluated) is not None:
            reason, detail = legacy.NON_FINITE_SOLVER_OUTPUT, "selected geometry is nonfinite"
        else:
            refused = legacy._production_solution_refusal(frame_ids, readings, selected, evaluated)
            if refused is None:
                reason, detail, matches, solution = "", "", selected, evaluated
            else:
                reason, detail = refused
    return _SelectionEvaluation(reason, detail, selected, evaluated, matches, solution, calls)


def derive_image_derived_graph(*, session, lattice_result, signatures_result, connection_graph=None,
                             max_shift_cells=32, integer_tolerance=0.15, min_resultant=0.90,
                             ambiguous_margin=0.02, budget=csp.EndpointBudget(), diagnostic_sink=None,
                             matcher_witness_sink=None, matcher_witness_payload_sink=None,
                             witness_budget=None, solver_failure_sink=None,
                             same_axis_candidate_sink=None, candidate_support_sink=None,
                             complete_support_sink=None, signature_witnesses=None,
                             same_axis_capture_axes=legacy.SAME_AXIS_CANDIDATES_AXES_V1,
                             alias_max_readings=legacy.ACTIVE_EDGE_ALIASES_MAX_READINGS,
                             alias_max_matches=legacy.ACTIVE_EDGE_ALIASES_MAX_MATCHES,
                             alias_max_active_edges=legacy.ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES,
                             alias_max_endpoint_alternatives=legacy.ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES,
                             alias_max_frame_id_bytes=legacy.ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES,
                             alias_max_record_bytes=legacy.ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES,
                             endpoint_incidence_sink=None):
    """One matcher, one joint selector and at most one numerical evaluation."""
    try:
        if (type(max_shift_cells) is not int or max_shift_cells != 32
                or (integer_tolerance, min_resultant, ambiguous_margin) != DEFAULT_GEOMETRY_SETTINGS
                or any(type(v) is not float for v in (integer_tolerance, min_resultant, ambiguous_margin))):
            return legacy._refusal("invalid_input", "joint_graph.fixed_settings")
        source = phase_b._sources(session=session, lattice_result=lattice_result,
                         signatures_result=signatures_result, connection_graph=connection_graph)
        if isinstance(source, placement.PlacementRefusal):
            return source
        accepted, lattice, signatures, frames, readings, evidence, constraints = source
        selection._frames(frames)
        selection.reading_bindings(readings, evidence)
        if type(budget) is not csp.EndpointBudget:
            raise ValueError("joint graph requires exact EndpointBudget")
        csp.EndpointBudget(**legacy._fields(budget))
        options = {"max_shift_cells": 32}
        admitted_universes = []
        options["admitted_candidate_sink"] = admitted_universes
        raw_observations = ([] if (same_axis_candidate_sink is not None or candidate_support_sink is not None
                                   or complete_support_sink is not None) else None)
        pre_filter_rows = ([] if ((same_axis_candidate_sink is not None
                                   and same_axis_capture_axes == legacy.SAME_AXIS_CANDIDATES_AXES_ALL)
                                  or candidate_support_sink is not None
                                  or complete_support_sink is not None) else None)
        if raw_observations is not None: options["candidate_observation_sink"] = raw_observations
        if pre_filter_rows is not None: options["pre_filter_witness_sink"] = pre_filter_rows
        if diagnostic_sink is not None: options["diagnostic_sink"] = diagnostic_sink
        if matcher_witness_sink is not None: options["witness_sink"] = matcher_witness_sink
        if matcher_witness_payload_sink is not None: options["witness_payload_sink"] = matcher_witness_payload_sink
        if (matcher_witness_sink is not None or matcher_witness_payload_sink is not None or pre_filter_rows is not None):
            options["witness_budget"] = witness_budget
            options["signature_sha256_by_identity"] = {id(e.signature): e.signature_sha256 for e in evidence}
        plane_graph.match_plane_graph(readings, tuple(e.signature for e in evidence), **options)
        if len(admitted_universes) != 1 or type(admitted_universes[0]) is not plane_graph.AdmittedCandidateUniverse:
            raise ValueError("matcher did not publish complete admitted universe")
        universe = admitted_universes[0]
        candidates = universe.candidates
        observer_options = ({} if endpoint_incidence_sink is None else
                            {"endpoint_incidence_sink": endpoint_incidence_sink})
        cert = selection.select_endpoint_hypotheses(readings=readings, signatures=evidence,
                    candidate_matches=candidates, frame_ids=frames, quadrant_constraints=constraints,
                    budget=budget, session_sha256=accepted.session_sha256, lattice_sha256=lattice.result_sha256,
                    signatures_sha256=signatures.result_sha256,
                    connection_sha256="" if connection_graph is None else connection_graph.graph_sha256,
                    candidate_dispositions=universe.dispositions, **observer_options)
        outcome = _evaluate_selection(readings=readings, signatures=evidence,
            candidate_matches=candidates, frame_ids=frames, certificate=cert)
        reason, detail, matches, solution, calls = (
            outcome.reason, outcome.detail, outcome.matches, outcome.solution, outcome.calls)
        if cert.status == "unique":
            selected, evaluated = outcome.selected, outcome.evaluated
            if evaluated.reason and solver_failure_sink is not None:
                alias_limits = dict(max_readings=alias_max_readings, max_matches=alias_max_matches,
                    max_active_edges=alias_max_active_edges, max_endpoint_alternatives=alias_max_endpoint_alternatives,
                    max_frame_id_bytes=alias_max_frame_id_bytes, max_record_bytes=alias_max_record_bytes)
                invalid_limit = next((name for name, value in alias_limits.items()
                                      if type(value) is not int or value < 0), None)
                aliases = (legacy._failed_alias_refusal(legacy._ACTIVE_EDGE_ALIASES_MALFORMED, "record")
                           if invalid_limit is not None else
                           legacy._failed_active_edges(readings, selected, evaluated, **alias_limits))
                try:
                    context = legacy.failed_solve_context(integer_tolerance, min_resultant, ambiguous_margin,
                                                         constraints, evaluated.frames)
                except _ERRORS:
                    context = legacy.refused_solve_context(integer_tolerance, min_resultant,
                                                          ambiguous_margin, evaluated.frames)
                solver_failure_sink(legacy._solver_failure_snapshot(evaluated, active_edge_aliases=aliases,
                    boundary_saturation=legacy._boundary_saturation_attestation(selected, diagnostic_sink, 32),
                    phase_capture=legacy.failed_phase_capture(readings, selected, evaluated),
                    solver_input_bindings=legacy.failed_solver_input_bindings(readings, selected, evaluated),
                    solve_context=context))
            if reason and same_axis_candidate_sink is not None:
                if same_axis_capture_axes in (None, legacy.SAME_AXIS_CANDIDATES_AXES_V1):
                    same_axis_candidate_sink(legacy.failed_same_axis_candidates(
                        readings, selected, evaluated,
                        raw_observations[0] if raw_observations and len(raw_observations) == 1 else None,
                        legacy.SAME_AXIS_CANDIDATES_AXES_V1))
                elif same_axis_capture_axes != legacy.SAME_AXIS_CANDIDATES_AXES_ALL:
                    same_axis_candidate_sink(legacy._same_axis_candidates_refusal(
                        legacy._SAME_AXIS_CANDIDATES_MALFORMED, "axes"))
                else:
                    same_axis_candidate_sink(legacy.failed_joint_same_axis_candidates(
                        readings, selected, evaluated,
                        raw_observations[0] if raw_observations and len(raw_observations) == 1 else None,
                        candidates, pre_filter_rows,
                        tuple(rep.source_row for rep in cert.representatives),
                        cert.result_sha256, selection._payload(cert)))
            if reason and (candidate_support_sink is not None or complete_support_sink is not None):
                ingress = tuple(f.ingress for f in accepted.frames if getattr(f,"ingress",None) is not None)
                binding = dict(ingress=ingress, failed_assignment_sha256=legacy._failed_assignment_digest(
                              tuple(evaluated.frames),dict(evaluated.quadrants)))
                args = (readings,selected,signature_witnesses or (),pre_filter_rows or (),
                        raw_observations[0] if raw_observations else None)
                if candidate_support_sink is not None:
                    candidate_support_sink(legacy.candidate_support_mod.failed_candidate_support_provenance(*args,**binding))
                if complete_support_sink is not None:
                    complete_support_sink(legacy.complete_support_mod.collect_route_b_diagnostic_support(*args,**binding))
        return _register_live(ImageDerivedGraphResult(IMAGE_DERIVED_GRAPH_SCHEMA, IMAGE_DERIVED_GRAPH_PRODUCER,
                    GRAPH_TERMINAL, "assembled" if solution is not None else "unavailable", reason, detail,
                    cert.session_sha256, cert.lattice_sha256, cert.signatures_sha256, cert.connection_sha256,
                    frames, readings, evidence, candidates, cert, matches, solution, geometry_evaluations=calls))
    except _ERRORS as error:
        return legacy._refusal("provenance_mismatch", "joint_graph.derivation", type(error).__name__)


def replay_image_derived_graph(*, graph_result, session, lattice_result, signatures_result,
                             connection_graph=None, budget=csp.EndpointBudget()):
    """Source replay rebuilds full candidates once; selected rows are not candidates."""
    if type(graph_result) is not ImageDerivedGraphResult:
        return legacy._refusal("provenance_mismatch", "joint_graph.identity")
    try:
        ImageDerivedGraphResult(**legacy._fields(graph_result))
        if type(budget) is not csp.EndpointBudget or graph_result.selection.budget != budget:
            return legacy._refusal("provenance_mismatch", "joint_graph.expected_budget")
    except _ERRORS as error:
        return legacy._refusal("provenance_mismatch", "joint_graph.structure", type(error).__name__)
    rebuilt = derive_image_derived_graph(session=session, lattice_result=lattice_result,
                       signatures_result=signatures_result, connection_graph=connection_graph, budget=budget)
    if isinstance(rebuilt, placement.PlacementRefusal):
        return rebuilt
    if rebuilt.result_sha256 != graph_result.result_sha256:
        return legacy._refusal("provenance_mismatch", "joint_graph.source_replay")
    try:
        return _register_live(graph_result)
    except _ERRORS as error:
        return legacy._refusal("provenance_mismatch", "joint_graph.live_receipt", type(error).__name__)
