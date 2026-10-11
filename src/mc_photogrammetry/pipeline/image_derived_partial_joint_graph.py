"""Explicit partial geometry; complete evidence remains selection authority.

Only the three-axis observable core receives 3D coordinates. Per-axis offsets
outside its anchor components are None, never least-squares world-zero values.
Ordinary graph derivation and its schemas remain untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time

import numpy as np

from . import image_derived_endpoint_selection as selection
from . import image_derived_graph as legacy
from . import image_derived_graph_v3 as source_mod
from . import image_derived_joint_graph as full_graph
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import plane_graph, plane_placement as placement

SCHEMA = "mc-photogrammetry-partial-joint-graph-v1"
PRODUCER = "m9-g-partial-joint-graph-1"


def _numerical_budget(search_budget):
    """Independent legacy-sized numerical allowance, even for extended search."""
    csp.validate_budget(search_budget, partial=True)
    values = asdict(search_budget)
    values["relation_visits"] = min(values["relation_visits"], csp.EndpointBudget().relation_visits)
    return csp.EndpointBudget(**values)


def _selected(cert):
    return tuple(rep.match for rep in cert.representatives)


def _core_inputs(readings, signatures, matches, cert):
    core = cert.observability[0].core
    projected = tuple(reading for reading in readings if reading.frame in core)
    raw = tuple(item.signature for item in signatures if item.frame_id in core)
    rows = tuple(row for row in matches if row.left in core and row.right in core)
    return projected, raw, rows


def _axis_metrics(face_values):
    angles = 2.0 * np.pi * np.mod(np.asarray(face_values, dtype=float), 1.0)
    resultant = float(abs(np.mean(np.exp(1j * angles))))
    mean = float(np.angle(np.mean(np.exp(1j * angles))))
    deviations = np.angle(np.exp(1j * (angles - mean)))
    residual = float(np.mean(np.abs(deviations)) / (2.0 * np.pi))
    return resultant, residual


def _constraints(indexed, matches, cert, work):
    choice = cert.proof_record()["choices"][0]
    q = dict(choice["quadrants"])
    selected = {(f, i): a for f, i, a in choice["alternatives"]}
    rows = {axis: [] for axis in range(3)}
    for index, match in enumerate(matches):
        work.charge("relation_visits", 1)
        key, delta, *_ = plane_graph.signed_physical_relation(
            indexed, match, q, selected, index, compact=True)
        rows[key[2]].append((match, delta if key[0] == match.left else -delta))
    return q, selected, rows


def _evaluate_core(readings, matches, cert):
    """One fixed-Q evaluation, with rank checked before coordinates are retained."""
    observation, = cert.observability
    indexed, _, frames = plane_graph._canonical_reading_index(readings)
    work = csp._Work(_numerical_budget(cert.budget))
    q, selected, rows = _constraints(indexed, matches, cert, work)
    offsets = {frame: [None, None, None] for frame in frames}
    resultants, residuals, ranks, counts = {}, {}, [], []
    for axis, components in observation.partitions:
        members = next(members for members, _gauge, known in components if known)
        active = [(row, delta) for row, delta in rows[axis] if row.left in members and row.right in members]
        variables = tuple(frame for frame in members if frame != observation.anchor)
        if not active or not variables:
            raise ValueError("partial core axis has insufficient equations")
        # Charge matrix cells and row traversal before allocating numerical buffers.
        work.charge("relation_visits", len(active) * (len(variables) + 1))
        columns = {frame: index for index, frame in enumerate(variables)}
        matrix = np.zeros((len(active), len(variables)), dtype=float)
        target = np.empty(len(active), dtype=float)
        for index, (row, delta) in enumerate(active):
            if row.right != observation.anchor:
                matrix[index, columns[row.right]] += 1.0
            if row.left != observation.anchor:
                matrix[index, columns[row.left]] -= 1.0
            target[index] = delta
        solved, _error, rank, _singular = np.linalg.lstsq(matrix, target, rcond=None)
        if rank != len(variables) or not np.all(np.isfinite(solved)):
            raise ValueError("partial core translation rank/nonfinite refusal")
        offsets[observation.anchor][axis] = 0.0  # explicit one global gauge only
        for frame in variables:
            offsets[frame][axis] = float(solved[columns[frame]])
        faces = []
        for row, delta in active:
            work.charge("relation_visits", 1)
            error = offsets[row.right][axis] - offsets[row.left][axis] - delta
            if not math.isfinite(error) or abs(error) > legacy._PRODUCTION_EDGE_ERROR_LIMIT_BLOCKS:
                raise ValueError("partial core equation residual refusal")
            for frame, plane, b in ((row.left, row.left_plane, row.left_b),
                                    (row.right, row.right_plane, row.right_b)):
                reading = indexed[(frame, plane)]
                _, sign = plane_graph.axis_transform(reading.axis, q[frame])
                faces.append(offsets[frame][axis] + sign * reading.side * reading.distance_for(b))
        resultants[axis], residuals[axis] = _axis_metrics(faces)
        if resultants[axis] < 0.90 or residuals[axis] > 0.15:
            raise ValueError("partial core integer residual refusal")
        ranks.append((axis, int(rank), len(variables), len(active)))
        counts.append(len(faces))
    core = observation.core
    blocks = {key: float(indexed[key].alternatives[alt].blocks_per_period)
              for key, alt in selected.items() if key[0] in core}
    mean_score = float(np.mean([row.score for row in matches]))
    alternative = plane_graph.ComponentAlternative(tuple(q[frame] for frame in core),
        float(sum((1.0 - resultants[axis]) * counts[axis] for axis in range(3))
              + plane_graph.MU_MATCH_SCORE * (1.0 - mean_score)),
        tuple(resultants[axis] for axis in range(3)), tuple(counts))
    component = plane_graph._component("", core, quadrants={frame: q[frame] for frame in core},
        translations={frame: tuple(offsets[frame]) for frame in core},
        residuals=residuals, resultants=resultants, blocks_per_period=blocks,
        alternatives=(alternative,), scale_known=True)
    encoded = tuple((frame, tuple(offsets[frame])) for frame in frames)
    work.encode([encoded, ranks], category="support")
    return component, encoded, tuple(ranks), work.counts["relation_visits"]


def _validate_geometry(graph):
    cert = graph.selection
    observation, = cert.observability
    solution = graph.core_solution
    if (type(solution) is not plane_graph.ComponentSolution or solution.reason or not solution.scale_known
            or solution.frames != observation.core or legacy._non_finite_solution_field(solution) is not None):
        raise ValueError("partial core solution differs")
    offsets = dict(graph.axis_offsets)
    if (type(graph.axis_offsets) is not tuple or [row[0] for row in graph.axis_offsets] != sorted(graph.frame_ids)
            or any(type(values) is not tuple or len(values) != 3 for values in offsets.values())):
        raise ValueError("partial offset census differs")
    for frame, values in offsets.items():
        resolved = observation.resolved_axes(frame)
        for axis, value in enumerate(values):
            if (axis not in resolved and value is not None
                    or axis in resolved and (type(value) is not float or not math.isfinite(value))):
                raise ValueError("partial unknown offset was fabricated")
    if offsets[observation.anchor] != (0., 0., 0.):
        raise ValueError("partial global gauge differs")
    choice = cert.proof_record()["choices"][0]
    q = dict(choice["quadrants"])
    if (solution.translations != {frame: offsets[frame] for frame in observation.core}
            or solution.quadrants != {frame: q[frame] for frame in observation.core}):
        raise ValueError("partial core projection differs")
    if (type(graph.axis_ranks) is not tuple or len(graph.axis_ranks) != 3
            or any(type(row) is not tuple or len(row) != 4
                   or any(type(value) is not int for value in row) for row in graph.axis_ranks)
            or type(graph.geometry_work) is not int
            or not 0 <= graph.geometry_work <= _numerical_budget(cert.budget).relation_visits):
        raise ValueError("partial numerical work differs")
    indexed, _, _ = plane_graph._canonical_reading_index(graph.readings)
    work = csp._Work(_numerical_budget(cert.budget))
    _q, selected, rows = _constraints(indexed, graph.matches, cert, work)
    expected_blocks = {key: float(indexed[key].alternatives[alt].blocks_per_period)
                       for key, alt in selected.items() if key[0] in observation.core}
    if solution.blocks_per_period != expected_blocks:
        raise ValueError("partial core selected scale differs")
    # Check every original supporting row, including locally dropped/peripheral
    # evidence. Floating constraints were physically checked by the certificate.
    all_source_rows = sorted({i for factor in choice["factors"] for i in factor["source_rows"]})
    for i in all_source_rows:
        work.charge("relation_visits", 1)
        (left, right, axis), delta, *_ = plane_graph.signed_physical_relation(
            indexed, graph.candidate_matches[i], q, selected, i, compact=True)
        if offsets[left][axis] is not None:
            if (offsets[right][axis] is None
                    or abs(offsets[right][axis] - offsets[left][axis] - delta) > 0.15):
                raise ValueError("partial original-source equation differs")
    expected_work = len(graph.matches)
    for axis, components in observation.partitions:
        members = next(members for members, _gauge, known in components if known)
        active = [(row, delta) for row, delta in rows[axis] if row.left in members and row.right in members]
        expected_rank = (axis, len(members) - 1, len(members) - 1, len(active))
        expected_work += len(active) * (len(members) + 1)
        if graph.axis_ranks[axis] != expected_rank:
            raise ValueError("partial rank/source dimensions differ")
        faces = []
        for row, _delta in active:
            for frame, plane, b in ((row.left, row.left_plane, row.left_b), (row.right, row.right_plane, row.right_b)):
                reading = indexed[(frame, plane)]
                _, sign = plane_graph.axis_transform(reading.axis, q[frame])
                faces.append(offsets[frame][axis] + sign * reading.side * reading.distance_for(b))
        resultant, residual = _axis_metrics(faces)
        if (solution.axis_resultants.get(axis) != resultant or solution.axis_residuals.get(axis) != residual
                or resultant < 0.90 or residual > 0.15):
            raise ValueError("partial numerical gates/source metrics differ")
    if graph.geometry_work != expected_work:
        raise ValueError("partial numerical work census differs")


def _payload(graph):
    return dict(schema=graph.schema, producer=graph.producer, status=graph.status,
        reason=graph.reason, frame_ids=graph.frame_ids, selection_sha256=graph.selection.result_sha256,
        candidates=selection.candidate_digest(graph.candidate_matches),
        core_solution=legacy._solution_payload(graph.core_solution),
        axis_offsets=graph.axis_offsets, axis_ranks=graph.axis_ranks,
        geometry_work=graph.geometry_work, geometry_evaluations=graph.geometry_evaluations)


@dataclass(frozen=True, slots=True)
class PartialJointGraphResult(full_graph._WeakGraph):
    schema: str
    producer: str
    status: str
    reason: str
    frame_ids: tuple
    readings: tuple
    signatures: tuple
    candidate_matches: tuple
    selection: selection.PartialEndpointSelectionCertificate
    matches: tuple = ()
    core_solution: plane_graph.ComponentSolution | None = None
    axis_offsets: tuple = ()
    axis_ranks: tuple = ()
    geometry_work: int = 0
    geometry_evaluations: int = 0
    result_sha256: str = ""

    def __post_init__(self):
        if (self.schema != SCHEMA or self.producer != PRODUCER
                or self.status not in ("assembled", "unavailable") or type(self.reason) is not str
                or type(self.selection) is not selection.PartialEndpointSelectionCertificate
                or self.frame_ids != self.selection.frame_ids
                or type(self.geometry_evaluations) is not int or self.geometry_evaluations not in (0, 1)):
            raise ValueError("partial joint graph identity differs")
        selection.validate_candidate_certificate(self.selection, self.readings, self.signatures, self.candidate_matches)
        if self.selection.status != "unique" and self.geometry_evaluations != 0:
            raise ValueError("ambiguous partial selection reached geometry")
        if self.selection.status == "unique" and self.geometry_evaluations != 1:
            raise ValueError("unique partial selection requires one core evaluation")
        if self.status == "assembled":
            if self.reason or self.matches != _selected(self.selection):
                raise ValueError("partial factor universe differs")
            _validate_geometry(self)
        elif (not self.reason or self.matches or self.core_solution is not None
                or self.axis_offsets or self.axis_ranks or self.geometry_work):
            raise ValueError("partial joint refusal must be atomic")
        digest = selection._sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("partial graph digest differs")
        object.__setattr__(self, "result_sha256", digest)


def _require_live(graph):
    if type(graph) is not PartialJointGraphResult:
        raise ValueError("foreign partial graph")
    with full_graph._LIVE_LOCK:
        full_graph._expire_receipts(time.monotonic())
        receipt = full_graph._LIVE_RECEIPTS.get(id(graph))
        if receipt is None or receipt[0]() is not graph or receipt[1] != graph.result_sha256:
            raise ValueError("partial graph needs validated derivation or source replay")
    PartialJointGraphResult(**legacy._fields(graph))
    return graph


def _assemble(readings, signatures, candidates, cert):
    options = {}
    reason, calls = cert.reason, 0
    if cert.status == "unique":
        calls = 1
        try:
            component, offsets, ranks, used = _evaluate_core(readings, _selected(cert), cert)
            options = dict(matches=_selected(cert), core_solution=component,
                axis_offsets=offsets, axis_ranks=ranks, geometry_work=used)
            reason = ""
        except csp._Exhausted:
            reason = "partial_geometry_budget_refused"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            reason = "partial_geometry_unavailable"
    return PartialJointGraphResult(SCHEMA, PRODUCER, "assembled" if options else "unavailable",
        reason, cert.frame_ids, readings, signatures, candidates, cert,
        geometry_evaluations=calls, **options)


def derive_partial_image_derived_graph(*, session, lattice_result, signatures_result,
                                     connection_graph=None, budget=csp.EndpointBudget()):
    """Explicit opt-in; one complete matcher pass, one selector, one core solve."""
    try:
        csp.validate_budget(budget, partial=True)
        source = source_mod._sources(session=session, lattice_result=lattice_result,
            signatures_result=signatures_result, connection_graph=connection_graph)
        if isinstance(source, placement.PlacementRefusal):
            return source
        accepted, lattice, signatures, frames, readings, evidence, constraints = source
        selection._frames(frames)
        selection.reading_bindings(readings, evidence)
        universes = []
        plane_graph.match_plane_graph(readings, tuple(item.signature for item in evidence),
            max_shift_cells=32, admitted_candidate_sink=universes)
        if len(universes) != 1 or type(universes[0]) is not plane_graph.AdmittedCandidateUniverse:
            raise ValueError("partial matcher did not publish complete candidates")
        universe = universes[0]
        cert = selection.select_partial_endpoint_hypotheses(readings=readings, signatures=evidence,
            candidate_matches=universe.candidates, candidate_dispositions=universe.dispositions,
            frame_ids=frames, quadrant_constraints=constraints, budget=budget,
            session_sha256=accepted.session_sha256, lattice_sha256=lattice.result_sha256,
            signatures_sha256=signatures.result_sha256,
            connection_sha256="" if connection_graph is None else connection_graph.graph_sha256)
        return full_graph._register_live(_assemble(readings, evidence, universe.candidates, cert))
    except source_mod._ERRORS as error:
        return legacy._refusal("provenance_mismatch", "partial_graph.derivation", type(error).__name__)


def replay_partial_image_derived_graph(*, graph_result, session, lattice_result,
        signatures_result, connection_graph=None, budget=csp.EndpointBudget()):
    if type(graph_result) is not PartialJointGraphResult or graph_result.selection.budget != budget:
        return legacy._refusal("provenance_mismatch", "partial_graph.identity_or_budget")
    rebuilt = derive_partial_image_derived_graph(session=session, lattice_result=lattice_result,
        signatures_result=signatures_result, connection_graph=connection_graph, budget=budget)
    if isinstance(rebuilt, placement.PlacementRefusal):
        return rebuilt
    if rebuilt.result_sha256 != graph_result.result_sha256:
        return legacy._refusal("provenance_mismatch", "partial_graph.source_replay")
    try:
        PartialJointGraphResult(**legacy._fields(graph_result))
        return full_graph._register_live(graph_result)
    except source_mod._ERRORS as error:
        return legacy._refusal("provenance_mismatch", "partial_graph.structure", type(error).__name__)
