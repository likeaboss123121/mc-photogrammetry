"""Developer-only, one-pass image geometry evidence; no evaluator inputs."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

# Direct execution places scripts/evaluation on sys.path, not the source tree.
# Anchor imports to this file so the runner needs neither installation nor an
# ambient PYTHONPATH or a particular working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
import mc_photogrammetry  # noqa: E402,F401 - verify imports before image processing
from mc_photogrammetry.pipeline.image_derived_entry import MAX_FRAMES_CAP as MAX_IMAGES  # noqa: E402

SCHEMA = "mc-photogrammetry-image-geometry-failure-diagnostic-v1"
MASK_DIAGNOSTIC_SCHEMA = "mc-photogrammetry-image-geometry-failure-diagnostic-v2"
MASK_BUNDLE_SCHEMA = "mc-photogrammetry-source-region-mask-bundle-v1"
MASK_BUNDLE_PRODUCER = "native-plane-region-mask-export-1"
MAX_MASK_PIXELS = 4_000_000
MAX_TOTAL_MASK_PIXELS = 12_000_000
MAX_PACKED_MASK_BYTES = 524_288
MAX_MASK_BUNDLE_BYTES = 1024 * 1024
MASK_ENDPOINTS = (("image-0000", 2), ("image-0000", 3), ("image-0001", 1))
MASK_ROWS = (176, 197)
DEFAULT_SETTINGS = {"max_shift_cells": 32, "integer_tolerance": 0.15,
                    "min_resultant": 0.90, "ambiguous_margin": 0.02}
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_PROJECT_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_JSON_ITEMS = 262144
MAX_JSON_DEPTH = 24
MAX_PAIR_EXAMPLES = 128
JOINT_ENDPOINT_SCHEMA = "mc-photogrammetry-joint-endpoint-observation-v3"
JOINT_ENDPOINT_PRODUCER = "first-retained-witness-provenance-1"
MAX_JOINT_ENDPOINT_BYTES = 4 * 1024 * 1024
MAX_INPUT_CONTEXT_TEXT = 1024


class DiagnosticError(RuntimeError):
    """Atomic diagnostic refusal with a bounded reason and field."""

    def __init__(self, reason, field="", context=None):
        self.reason, self.field, self.context = reason, field, context
        super().__init__(f"{reason}: {field}")


class _SingleCapture:
    def __init__(self, field):
        self.field, self.value, self.count = field, None, 0

    def append(self, value):
        self.count += 1
        if self.count != 1:
            raise DiagnosticError("callback_multiplicity", self.field)
        self.value = value


@dataclasses.dataclass
class ChainCapture:
    entry: object
    lattice: object
    signatures: object
    graph: object
    attribution: object
    signature_witnesses: tuple
    failure: _SingleCapture
    same_axis: _SingleCapture
    vertical_attrition: dict | None = None
    endpoint_conflicts: dict | None = None


def _bounded_bytes(value, *, limit=None):
    """Bound strict JSON before serialization, including aggregate traversal."""
    cap = MAX_RECORD_BYTES if limit is None else min(limit, MAX_RECORD_BYTES)
    reserve = 1 if limit is None else 0  # outer publication adds a newline
    stack = [(value, 0)]
    items = lower_bytes = 0
    while stack:
        item, depth = stack.pop()
        items += 1
        if items > MAX_JSON_ITEMS or depth > MAX_JSON_DEPTH:
            raise DiagnosticError("evidence_cap", "json_structure")
        if type(item) is dict:
            if len(item) > MAX_JSON_ITEMS - items - len(stack):
                raise DiagnosticError("evidence_cap", "json_structure")
            if any(type(key) is not str for key in item):
                raise DiagnosticError("invalid_evidence", "json_key")
            lower_bytes += sum(len(key) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) in (tuple, list):
            if len(item) > MAX_JSON_ITEMS - items - len(stack):
                raise DiagnosticError("evidence_cap", "json_structure")
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            lower_bytes += len(item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise DiagnosticError("non_finite_evidence", "json_number")
        elif type(item) is int:
            if item.bit_length() > 64:
                raise DiagnosticError("invalid_evidence", "json_integer")
        elif item is not None and type(item) is not bool:
            raise DiagnosticError("invalid_evidence", "json_type")
        if lower_bytes + items > cap:
            raise DiagnosticError("evidence_cap", "json_bytes")
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False)
    data = bytearray()
    for chunk in encoder.iterencode(value):
        encoded = chunk.encode("utf-8")
        if len(data) + len(encoded) + reserve > cap:
            raise DiagnosticError("evidence_cap", "json_bytes")
        data.extend(encoded)
    return bytes(data)


def _digest(value, *, limit=None):
    return hashlib.sha256(_bounded_bytes(value, limit=limit)).hexdigest()


def _validate_attrition_record(record):
    """Fail closed on false closure claims; incomplete accounting is reportable."""
    if (type(record) is not dict
            or record.get("schema") != "mc-photogrammetry-scalar-attrition-v1"
            or record.get("status") not in ("complete", "incomplete")
            or type(record.get("stages")) is not dict
            or type(record.get("checks")) is not dict):
        raise DiagnosticError("invalid_evidence", "vertical_attrition")
    _bounded_bytes(record)
    closed = bool(record["stages"])
    for name, stage in record["stages"].items():
        if name not in ("proposal", "reading", "signature", "pair", "seed", "frame",
                        "period_measurement", "period_round", "period_profile",
                        "period_estimate", "period_reconciliation",
                        "local_period_parent", "local_period_child", "local_period_measurement",
                        "local_period_round", "local_period_profile", "local_period_estimate",
                        "local_period_reconciliation"):
            raise DiagnosticError("invalid_evidence", "vertical_attrition_stage")
        for key in ("expected", "classified", "unclassified", "unexpected",
                    "multiply_classified", "unexpected_events", "omitted_examples"):
            if type(stage.get(key)) is not int or stage[key] < 0:
                raise DiagnosticError("invalid_evidence", "vertical_attrition_counts")
        _sha(stage.get("terminal_sha256"), "vertical_attrition_digest")
        if (type(stage.get("counts")) is not list or type(stage.get("examples")) is not list
                or any(type(group.get("count")) is not int or group["count"] < 1
                       for group in stage["counts"])):
            raise DiagnosticError("invalid_evidence", "vertical_attrition_counts")
        closed = closed and stage["expected"] == stage["classified"] and all(
            stage[key] == 0 for key in ("unclassified", "unexpected", "multiply_classified", "unexpected_events"))
        closed = closed and sum(group["count"] for group in stage["counts"]) == stage["classified"]
        closed = closed and len(stage["examples"]) + stage["omitted_examples"] == stage["classified"]
    if (any(type(value) is not bool for value in record["checks"].values())
            or type(record.get("omitted_accounting_events")) is not int
            or record["omitted_accounting_events"] < 0):
        raise DiagnosticError("invalid_evidence", "vertical_attrition_closure")
    if record["status"] == "complete" and (
            not closed or not all(record["checks"].values()) or record["omitted_accounting_events"]):
        raise DiagnosticError("invalid_evidence", "vertical_attrition_closure")
    return record


def _sha(value, field):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise DiagnosticError("invalid_evidence", field)
    return value


def _validate_endpoint_conflicts(record):
    from mc_photogrammetry.reconstruction import plane_graph

    if (type(record) is not dict
            or record.get("schema") != "mc-photogrammetry-endpoint-conflict-observation-v1"
            or record.get("status") not in ("complete", "incomplete", "refused")):
        raise DiagnosticError("invalid_evidence", "endpoint_period_conflicts")
    _bounded_bytes(record)
    counts = ("solver_calls", "planned_assignments", "visited_assignments",
              "conflicted_assignments", "evaluator_calls", "feasible_evaluations",
              "hashed_assignments", "retained_witness_count", "omitted_witness_count")
    if any(type(record.get(k)) is not int or record[k] < 0 for k in counts):
        raise DiagnosticError("invalid_evidence", "endpoint_conflict_counts")
    if record["status"] == "refused":
        raise DiagnosticError("evidence_cap", "endpoint_period_conflicts")
    for k in ("assignment_sequence_sha256", "source_readings_sha256", "input_matches_sha256"):
        _sha(record.get(k), "endpoint_conflict_digest")
    witnesses, classes = record.get("witnesses"), record.get("conflict_classes")
    if (type(witnesses) is not list or len(witnesses) > plane_graph._EndpointConflictObserver.MAX_WITNESSES
            or len(witnesses) != record["retained_witness_count"] or type(classes) is not dict
            or any(type(v) is not int or v < 0 for v in classes.values())
            or record["retained_witness_count"] + record["omitted_witness_count"] != record["conflicted_assignments"]):
        raise DiagnosticError("invalid_evidence", "endpoint_conflict_witnesses")
    if record["status"] == "complete" and not (
            record["solver_calls"] == 1
            and record["planned_assignments"] == record["visited_assignments"] == record["hashed_assignments"]
            and record["visited_assignments"] == record["conflicted_assignments"] + record["evaluator_calls"]
            and record["feasible_evaluations"] <= record["evaluator_calls"]
            and sum(classes.values()) == record["conflicted_assignments"]):
        raise DiagnosticError("invalid_evidence", "endpoint_conflict_closure")
    return record


def _joint_witness_quadrants(cert, proof, requested):
    """One bounded planner traversal; projection never searches alternatives."""
    from mc_photogrammetry.reconstruction import plane_graph

    if not requested:
        return {}
    planned = proof["counters"]["planned_assignments"]
    if (len(requested) > 128 or type(planned) is not int or not 0 < planned <= 16384
            or any(type(i) is not int or not 0 <= i < planned for i in requested)):
        raise DiagnosticError("invalid_evidence", "joint_endpoint_q_indices")
    constraints = {}
    for left, right, relative in cert.quadrant_constraints:
        if right < left: left, right, relative = right, left, (-relative) % 4
        if (left, right) in constraints and constraints[(left, right)] != relative:
            raise DiagnosticError("invalid_evidence", "joint_endpoint_q_constraints")
        constraints[(left, right)] = relative
    reason, count, plans = plane_graph._quadrant_assignment_plan(tuple(sorted(cert.frame_ids)), constraints)
    if reason or count != planned:
        raise DiagnosticError("binding_mismatch", "joint_endpoint_q_plan")
    maps, identities, last = {}, set(), max(requested)
    for index, quadrants in enumerate(plans):
        if index >= planned or index >= 16384:
            raise DiagnosticError("evidence_cap", "joint_endpoint_q_plan")
        if index in requested:
            if (type(quadrants) is not dict or list(quadrants) != sorted(cert.frame_ids)
                    or any(type(q) is not int or q not in range(4) for q in quadrants.values())
                    or quadrants[sorted(cert.frame_ids)[0]] != 0):
                raise DiagnosticError("invalid_evidence", "joint_endpoint_q_map")
            identity = tuple(quadrants.items())
            if identity in identities:
                raise DiagnosticError("binding_mismatch", "joint_endpoint_q_map")
            identities.add(identity)
            maps[index] = dict(quadrants)
        if index == last: break
    if set(maps) != requested:
        raise DiagnosticError("binding_mismatch", "joint_endpoint_q_map")
    return maps


def _joint_resolved_extrema(graph, proof):
    """Resolve only retained partial extrema, preserving original row identity."""
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    from mc_photogrammetry.reconstruction import plane_graph

    cert, counts = graph.selection, proof["counters"]
    partial, leaves = proof["partial_exclusions"], proof["physical_exclusions"]
    if (type(partial) is not list or type(leaves) is not list
            or len(partial) != counts["retained_partial_exclusions"]
            or len(leaves) != counts["retained_physical_exclusions"]
            or len(partial) + len(leaves) > 128
            or len(partial) + counts["omitted_partial_exclusions"] != counts["partial_prunes"]
            or len(leaves) + counts["omitted_physical_exclusions"] != counts["physical_rejected"]):
        raise DiagnosticError("invalid_evidence", "joint_endpoint_shared_exclusions")
    requested = set()
    for event in partial:
        if type(event) is not list or len(event) != 6 or event[0] != "partial":
            raise DiagnosticError("invalid_evidence", "joint_endpoint_partial_event")
        requested.add(event[1])
    maps = _joint_witness_quadrants(cert, proof, requested)
    indexed, _, _ = plane_graph._canonical_reading_index(graph.readings)
    bindings = {tuple(row[0]): row for row in cert.reading_bindings}
    rows, alternatives, resolved = {}, {}, []
    for event_index, event in enumerate(partial):
        _, assignment_index, _, _, _, group = event
        key, low, high, lo, hi = group
        quadrants = maps[assignment_index]
        extrema = {"event_kind": "partial", "event_index": event_index,
                   "assignment_index": assignment_index,
                   "quadrants": [[frame, absolute] for frame, absolute in quadrants.items()],
                   "frame_pair": key[:2], "world_axis": key[2],
                   "tolerance": proof["physical_tolerance"]}
        for label, delta, ref in (("low", low, lo), ("high", high, hi)):
            row_index, canonical_left, canonical_right, world_axis, signed_delta = ref
            if type(row_index) is not int or not 0 <= row_index < len(graph.candidate_matches):
                raise DiagnosticError("invalid_evidence", "joint_endpoint_witness_row")
            row = graph.candidate_matches[row_index]
            u, v = (row.left, row.left_plane), (row.right, row.right_plane)
            a, b = ((canonical_left, canonical_right) if row.left < row.right
                    else (canonical_right, canonical_left))
            relation = plane_graph.signed_physical_relation(
                indexed, row, quadrants, {u: a, v: b}, row_index, compact=True)
            if (list(relation[0]) != key or relation[1] != delta or delta != signed_delta
                    or relation[2:] != (row_index, canonical_left, canonical_right)
                    or world_axis != key[2]):
                raise DiagnosticError("binding_mismatch", "joint_endpoint_witness_relation")
            if row_index not in rows:
                if len(rows) >= 256:
                    raise DiagnosticError("evidence_cap", "joint_endpoint_witness_rows")
                raw = selection._encode(selection.legacy._match_payload(row),
                    limit=cert.budget.scalar_row_bytes)
                rows[row_index] = {"source_row": row_index, "row_sha256": hashlib.sha256(raw).hexdigest(),
                    "left": row.left, "right": row.right, "left_plane": row.left_plane,
                    "right_plane": row.right_plane, "left_axis": row.left_axis,
                    "right_axis": row.right_axis, "relative_quadrant": row.quadrant_of_right,
                    "left_b": row.left_b, "right_b": row.right_b}
            for endpoint, alternative_index in ((u, a), (v, b)):
                identity = (endpoint, alternative_index)
                if identity in alternatives: continue
                if len(alternatives) >= 512:
                    raise DiagnosticError("evidence_cap", "joint_endpoint_witness_alternatives")
                binding, reading = bindings[endpoint], indexed[endpoint]
                if graph.readings[binding[1]] is not reading:
                    raise DiagnosticError("binding_mismatch", "joint_endpoint_witness_reading")
                alternative = reading.alternatives[alternative_index]
                alternatives[identity] = {"endpoint": list(endpoint), "reading_index": binding[1],
                    "alternative_index": alternative_index, "local_axis": reading.axis,
                    "side": float(reading.side), "blocks_per_period": float(alternative.blocks_per_period),
                    "distance_blocks": float(alternative.distance_blocks),
                    "reading_binding_sha256": binding[3], "signature_sha256": binding[5][alternative_index]}
            extrema[label] = {"reference": list(ref), "source_row": row_index, "delta": delta,
                            "original_left_alternative_index": a, "original_right_alternative_index": b}
        resolved.append(extrema)
        # Bound each accumulated projection before adding more presentation data.
        _bounded_bytes({"resolved_extrema": resolved, "witness_rows": list(rows.values()),
                        "witness_alternatives": list(alternatives.values())}, limit=MAX_JOINT_ENDPOINT_BYTES)
    return resolved, [rows[i] for i in sorted(rows)], [alternatives[i] for i in sorted(alternatives)]


@dataclasses.dataclass
class _ProjectionSequence:
    """Lazy scalar records: do not allocate a roster before its byte charge."""
    items: object


def _precharged_projection(value):
    """Charge each JSON fragment before retention, then allocate the projection.

    Source strings and lazy rosters are traversed without copying source masks
    or signature arrays. The final JSON object is allocated only after all byte,
    item and depth charges fit the unchanged nested cap.
    """
    data, items = bytearray(), 0
    def fragments(item, depth=0):
        nonlocal items
        items += 1
        if items > MAX_JSON_ITEMS or depth > MAX_JSON_DEPTH:
            raise DiagnosticError("evidence_cap", "joint_endpoint_provenance")
        if type(item) is dict:
            yield b"{"
            for index, (key, child) in enumerate(item.items()):
                if index: yield b","
                yield _bounded_bytes(key, limit=MAX_JOINT_ENDPOINT_BYTES - len(data))
                yield b":"
                yield from fragments(child, depth + 1)
            yield b"}"
        elif type(item) in (tuple, list, _ProjectionSequence):
            yield b"["
            for index, child in enumerate(item.items if type(item) is _ProjectionSequence else item):
                if index: yield b","
                yield from fragments(child, depth + 1)
            yield b"]"
        else:
            yield _bounded_bytes(item, limit=MAX_JOINT_ENDPOINT_BYTES - len(data))
    for fragment in fragments(value):
        if len(data) + len(fragment) > MAX_JOINT_ENDPOINT_BYTES:
            raise DiagnosticError("evidence_cap", "joint_endpoint_provenance")
        data.extend(fragment)
    return json.loads(data)


def _provenance_metric(value):
    if value is None:
        return {"type": "unavailable", "reason": "not_present_in_source_record"}
    number = float(value)
    if math.isfinite(number): return number
    return {"type": "nonfinite", "value": ("nan" if math.isnan(number) else
            "positive_infinity" if number > 0 else "negative_infinity")}


def _provenance_unavailable(reason="not_retained_in_current_joinable_stream"):
    return {"type": "unavailable", "reason": reason}


def _joint_first_witness_provenance(graph, resolved):
    """Join only the first retained partial witness; no producer work or solve."""
    from mc_photogrammetry.reconstruction import lattice, plane_graph

    if not resolved:
        return _provenance_unavailable("no_retained_partial_witness")
    first, cert = resolved[0], graph.selection
    row_indices = sorted({first[label]["source_row"] for label in ("low", "high")})
    endpoints = set()
    for index in row_indices:
        row = graph.candidate_matches[index]
        endpoints.update(((row.left, row.left_plane), (row.right, row.right_plane)))
    # Count preflight precedes projection dictionaries and alternative rosters.
    if len(row_indices) > 2 or len(endpoints) > 3:
        raise DiagnosticError("evidence_cap", "joint_endpoint_provenance_joins")
    bindings = {}
    for binding in cert.reading_bindings:
        endpoint = tuple(binding[0])
        if endpoint in bindings:
            raise DiagnosticError("binding_mismatch", "joint_endpoint_provenance_reading")
        bindings[endpoint] = binding
    indexed, _, _ = plane_graph._canonical_reading_index(graph.readings)
    signature_index = {}
    for evidence in graph.signatures:
        identity = (evidence.canonical_index, evidence.alternative_index)
        if identity in signature_index:
            raise DiagnosticError("binding_mismatch", "joint_endpoint_provenance_signature")
        signature_index[identity] = evidence
    if sum(len(indexed[e].alternatives) for e in endpoints) > 512:
        raise DiagnosticError("evidence_cap", "joint_endpoint_provenance_alternatives")

    def alternative_records(reading, binding, world_period):
        for index, alternative in enumerate(reading.alternatives):
            evidence = signature_index[(binding[1], index)]
            if (evidence.signature.reading is not reading
                    or evidence.signature_sha256 != binding[5][index]
                    or (float(alternative.blocks_per_period), float(alternative.distance_blocks),
                        float(alternative.support)) != tuple(binding[4][index])):
                raise DiagnosticError("binding_mismatch", "joint_endpoint_provenance_alternative")
            signature, footprint = evidence.signature, evidence.signature.region_footprint
            sampling = _provenance_unavailable("not_present_in_source_record")
            if footprint is not None:
                if (footprint.blocks_per_period != alternative.blocks_per_period
                        or footprint.origin_uv != signature.origin_uv
                        or footprint.grid_shape != signature.cells.shape
                        or (reading.region_evidence is not None and footprint.source_region_id !=
                            reading.region_evidence.source_region_id)):
                    raise DiagnosticError("binding_mismatch", "joint_endpoint_provenance_region")
                sampling = {"type": "available", "source_region_id": footprint.source_region_id,
                    "subsample_factor": footprint.subsample_factor,
                    "total_samples_per_cell": footprint.total_samples_per_cell,
                    "inside_region_counts_sha256": hashlib.sha256(footprint.inside_region_sample_counts).hexdigest(),
                    "valid_counts_sha256": hashlib.sha256(footprint.valid_sample_counts).hexdigest()}
            formula = float(alternative.blocks_per_period) / world_period
            yield {"alternative_index": index, "blocks_per_period": float(alternative.blocks_per_period),
                "distance_blocks": float(alternative.distance_blocks), "support": _provenance_metric(alternative.support),
                "formula_distance_blocks": _provenance_metric(formula),
                "distance_minus_formula_blocks": _provenance_metric(float(alternative.distance_blocks) - formula),
                "signature": {"sha256": evidence.signature_sha256, "reading_index": evidence.canonical_index,
                    "alternative_index": evidence.alternative_index, "frame": evidence.frame_id,
                    "reading_axis": evidence.reading_axis, "blocks_per_period": float(evidence.blocks_per_period),
                    "origin_uv": signature.origin_uv, "grid_shape": signature.cells.shape,
                    "sampling": sampling}}

    def reading_records():
        for endpoint in sorted(endpoints):
            reading, binding = indexed[endpoint], bindings[endpoint]
            if graph.readings[binding[1]] is not reading:
                raise DiagnosticError("binding_mismatch", "joint_endpoint_provenance_reading")
            step = lattice.rectified_pixel_step(reading.extent, reading.samples)
            world_period = float(reading.period_pixels) * step
            if not math.isfinite(world_period) or world_period <= 0:
                raise DiagnosticError("invalid_evidence", "joint_endpoint_provenance_measurement")
            region = reading.region_evidence
            region_record = (_provenance_unavailable("not_present_in_source_record") if region is None else
                {"type": "available", "source_region_id": region.source_region_id,
                 "mask_sha256": region.mask_sha256, "axis": region.axis, "sources": region.sources,
                 "image_shape": region.image_shape, "bbox_xyxy": region.bbox_xyxy,
                 "crop_shape": region.crop_shape, "pixel_count": region.pixel_count,
                 "encoding": region.encoding, "bit_order": region.bit_order})
            yield {"endpoint": endpoint, "reading_index": binding[1], "reading_binding_sha256": binding[3],
                "quadrant": dict(first["quadrants"])[endpoint[0]], "local_axis": reading.axis,
                "side": float(reading.side), "period_pixels": float(reading.period_pixels),
                "phase_pixels": reading.phase_pixels, "extent": float(reading.extent),
                "measurement_samples": reading.samples, "rectified_pixel_step": step,
                "world_units_per_period": world_period, "coverage_cells": reading.coverage_cells,
                "structural_share": _provenance_metric(reading.structural_share),
                "confidence": _provenance_metric(reading.confidence), "polygon": reading.polygon,
                "rectification": dataclasses.asdict(reading.rectification), "source_region": region_record,
                "phase_evidence": (_provenance_unavailable("not_present_in_source_record")
                    if reading.phase_evidence is None else dataclasses.asdict(reading.phase_evidence)),
                "alternatives": _ProjectionSequence(alternative_records(reading, binding, world_period))}

    def row_records():
        from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
        for index in row_indices:
            row = graph.candidate_matches[index]
            row_hash = hashlib.sha256(selection._encode(selection.legacy._match_payload(row),
                limit=cert.budget.scalar_row_bytes)).hexdigest()
            joins = []
            for label in ("low", "high"):
                ref = first[label]
                if ref["source_row"] != index: continue
                left, right = (row.left, row.left_plane), (row.right, row.right_plane)
                a, b = ref["original_left_alternative_index"], ref["original_right_alternative_index"]
                joins.append({"extremum": label, "original_alternative_indices": [a, b],
                    "canonical_alternative_indices": ref["reference"][1:3],
                    "original_signature_sha256": [bindings[left][5][a], bindings[right][5][b]],
                    "world_axis": first["world_axis"], "canonical_signed_delta": ref["delta"],
                    "original_signed_delta": (-ref["delta"] if row.right < row.left else ref["delta"])})
            yield {"source_universe_sha256": cert.candidate_matches_sha256, "source_row": index,
                "row_sha256": row_hash, "original_direction": [[row.left, row.left_plane], [row.right, row.right_plane]],
                "canonical_direction": sorted([[row.left, row.left_plane], [row.right, row.right_plane]]),
                "local_axes": [row.left_axis, row.right_axis], "selected_b": [float(row.left_b), float(row.right_b)],
                "relative_quadrant": row.quadrant_of_right, "joins": joins,
                "matcher_metrics": {name: _provenance_metric(getattr(row, name)) for name in
                    ("psr", "peak", "second_peak", "integer_residual", "score")},
                "overlap_cells": row.overlap_cells, "cell_shift_rc": row.cell_shift_rc}

    return _precharged_projection({"type": "available", "partial_exclusion_index": 0,
        "assignment_index": first["assignment_index"], "quadrants": first["quadrants"],
        "source_universe_sha256": cert.candidate_matches_sha256,
        "proof_sha256": hashlib.sha256(cert.proof_json.encode("utf-8")).hexdigest(),
        "rows": _ProjectionSequence(row_records()), "readings": _ProjectionSequence(reading_records()),
        "optional_groups": {name: _provenance_unavailable() for name in
            ("winning_round", "ranked_support_floor_candidates", "raw_supported_correlation_maxima")},
        "unresolved_causal_questions": ["why_estimator_selected_period", "which_rejected_alternative_lost_at_support_floor",
            "whether_competing_peak_had_sufficient_support", "which_physical_correspondence_is_wrong"],
        "formula_residual_policy": "observation_only", "physical_identity": "not_established"})


def _joint_endpoint_observation(graph):
    """Project the original selector result after its existing authenticity replay.

    This is distinct from solve_component's legacy conflict observer. In
    particular, raw matcher rows are never classified as solver conflicts.
    Certificate validation recomputes the CSP; collecting witnesses adds no solve.
    """
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    from mc_photogrammetry.pipeline import image_derived_joint_graph as joint

    if type(graph) is not joint.ImageDerivedGraphResult:
        raise DiagnosticError("invalid_evidence", "joint_endpoint_graph")
    if (type(graph.readings) is not tuple or len(graph.readings) > 128
            or type(graph.signatures) is not tuple or len(graph.signatures) > 2048
            or type(graph.candidate_matches) is not tuple or len(graph.candidate_matches) > 16384):
        raise DiagnosticError("evidence_cap", "joint_endpoint_sources")
    cert = graph.selection
    try:
        selection.validate_candidate_certificate(
            cert, graph.readings, graph.signatures, graph.candidate_matches)
        proof = cert.proof_record()
        resolved, witness_rows, witness_alternatives = _joint_resolved_extrema(graph, proof)
        provenance = _joint_first_witness_provenance(graph, resolved)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        raise DiagnosticError("invalid_evidence", "joint_endpoint_certificate") from error
    terminal = {"no": "complete", "unique": "complete",
                "multiple": "incomplete", "unresolved_evidence": "incomplete",
                "budget_refused": "refused"}[cert.status]
    body = {
        "schema": JOINT_ENDPOINT_SCHEMA, "producer": JOINT_ENDPOINT_PRODUCER, "status": terminal,
        "first_witness_provenance": provenance,
        "observer_work": {"csp_solves": 0, "measurement_calls": 0, "signature_regenerations": 0,
                          "matcher_calls": 0, "authenticity_certificate_replays": 1},
        "reason": cert.reason, "selection_status": cert.status,
        "selection_sha256": cert.result_sha256,
        "source_bindings": {
            "session_sha256": cert.session_sha256,
            "lattice_sha256": cert.lattice_sha256,
            "signatures_sha256": cert.signatures_sha256,
            "connection_sha256": cert.connection_sha256,
            "graph_sha256": graph.result_sha256,
        },
        "frames": list(cert.frame_ids),
        "candidate_count": cert.candidate_count,
        "candidate_matches_sha256": cert.candidate_matches_sha256,
        "reading_bindings_sha256": selection._sha(cert.reading_bindings),
        "proof_sha256": hashlib.sha256(cert.proof_json.encode("utf-8")).hexdigest(),
        "proof_exhaustive": proof["exhaustive"],
        "proof_counters": proof["counters"],
        "eligibility_digest": proof["eligibility_digest"],
        "eligibility_reasons": proof["eligibility_reasons"],
        "excluded_assignments": proof["excluded_assignments"],
        "source_digest": proof["source_digest"],
        "choices": [{"assignment_index": choice["assignment_index"],
                     "factor_source_rows": [factor["source_rows"] for factor in choice["factors"]]}
                    for choice in proof["choices"]],
        "representative_source_rows": [rep.source_row for rep in cert.representatives],
        "transcript_policy": proof["transcript_policy"],
        "physical_policy": proof["physical_policy"], "physical_tolerance": proof["physical_tolerance"],
        "physical_digest": proof["physical_digest"],
        "partial_exclusions": proof["partial_exclusions"], "physical_exclusions": proof["physical_exclusions"],
        "partial_exclusion_counts": {"total": proof["counters"]["partial_prunes"],
            "retained": proof["counters"]["retained_partial_exclusions"],
            "omitted": proof["counters"]["omitted_partial_exclusions"]},
        "physical_exclusion_counts": {"total": proof["counters"]["physical_rejected"],
            "retained": proof["counters"]["retained_physical_exclusions"],
            "omitted": proof["counters"]["omitted_physical_exclusions"]},
        "resolved_extrema": resolved, "witness_rows": witness_rows, "witness_alternatives": witness_alternatives,
        "retained_example_coverage": sorted({event[1] for event in proof["partial_exclusions"]}),
        "representative_counterexample": ({"status": "representative_counterexample_only", "partial_exclusion_index": 0}
                                           if proof["partial_exclusions"] else None),
        "witness_availability": "retained_partial_counterexample" if proof["partial_exclusions"] else "unavailable",
    }
    try:
        record = dict(body, record_sha256=_digest(body, limit=MAX_JOINT_ENDPOINT_BYTES))
        _bounded_bytes(record, limit=MAX_JOINT_ENDPOINT_BYTES)
    except DiagnosticError as error:
        if error.reason == "evidence_cap":
            raise DiagnosticError("evidence_cap", "joint_endpoint_observation") from error
        raise
    return record


def _validate_joint_endpoint_observation(record, graph):
    if type(record) is not dict or record.get("schema") != JOINT_ENDPOINT_SCHEMA:
        raise DiagnosticError("invalid_evidence", "joint_endpoint_observation")
    try:
        _bounded_bytes(record, limit=MAX_JOINT_ENDPOINT_BYTES)
    except DiagnosticError as error:
        if error.reason == "evidence_cap":
            raise DiagnosticError("evidence_cap", "joint_endpoint_observation") from error
        raise
    body = dict(record)
    digest = body.pop("record_sha256", None)
    if digest != _digest(body):
        raise DiagnosticError("binding_mismatch", "joint_endpoint_record")
    expected = _joint_endpoint_observation(graph)
    if record != expected:
        raise DiagnosticError("binding_mismatch", "joint_endpoint_certificate")
    return record


def _context_repr(value):
    """Bound path metadata rendered in developer-only refusal JSON."""
    rendered = repr(value).encode("ascii", "backslashreplace").decode("ascii")
    return rendered[:MAX_INPUT_CONTEXT_TEXT], len(rendered) > MAX_INPUT_CONTEXT_TEXT


def _context_code(value):
    return value if type(value) is int and 0 <= value < 2**32 else None


def _input_failure_context(index, value, path, suffix, result=None, error=None):
    raw, raw_truncated = _context_repr(value)
    try:
        cwd, cwd_truncated = _context_repr(os.getcwd())
    except OSError:
        cwd, cwd_truncated = "<unavailable>", False
    resolved, resolved_truncated = _context_repr(str(path))
    suffix_text, suffix_truncated = _context_repr(suffix)
    if error is not None:
        outcome = {"status": "error", "errno": _context_code(getattr(error, "errno", None)),
                   "winerror": _context_code(getattr(error, "winerror", None))}
    else:
        mode = result.st_mode
        outcome = {"status": "ok", "mode": mode,
                   "type": "regular_file" if stat.S_ISREG(mode) else
                           "directory" if stat.S_ISDIR(mode) else "other"}
    return {"schema": "mc-photogrammetry-image-input-refusal-context-v1",
            "arg_index": index, "raw_argument_repr": raw, "cwd_repr": cwd,
            "resolved_path_repr": resolved, "suffix_repr": suffix_text,
            "truncated": {"raw_argument": raw_truncated, "cwd": cwd_truncated,
                          "resolved_path": resolved_truncated, "suffix": suffix_truncated},
            "stat": outcome}


def _input_records(images):
    if type(images) not in (tuple, list) or not 2 <= len(images) <= MAX_IMAGES:
        raise DiagnosticError("invalid_inputs", "image_count")
    # Resolve and stat the complete path list before opening any image bytes.
    # This keeps even a later missing/wrong-suffix argument pre-open.
    admitted = []
    for index, value in enumerate(images):
        if not isinstance(value, (str, Path)) or isinstance(value, bool):
            raise DiagnosticError("invalid_inputs", "image_path")
        path = Path(value).absolute()
        suffix = path.suffix.lower()
        try:
            path_stat = os.stat(path)
        except (OSError, ValueError) as error:
            raise DiagnosticError("invalid_inputs", "ordinary_image",
                                  _input_failure_context(index, value, path, suffix, error=error)) from None
        if not stat.S_ISREG(path_stat.st_mode) or suffix not in (".png", ".jpg", ".jpeg"):
            raise DiagnosticError("invalid_inputs", "ordinary_image",
                                  _input_failure_context(index, value, path, suffix, result=path_stat))
        admitted.append((path, path_stat.st_size))
    records, paths, identities, digests = [], [], set(), set()
    total = 0
    for path, size in admitted:
        identity = os.path.normcase(str(path))
        if identity in identities:
            raise DiagnosticError("invalid_inputs", "duplicate_image")
        identities.add(identity)
        total += size
        if size <= 0 or size > MAX_IMAGE_BYTES or total > MAX_INPUT_BYTES:
            raise DiagnosticError("input_cap", "image_bytes")
        digest = hashlib.sha256()
        consumed = 0
        with path.open("rb") as stream:
            header = stream.read(16)
            if not (header.startswith(b"\x89PNG\r\n\x1a\n")
                    or header.startswith(b"\xff\xd8\xff")):
                raise DiagnosticError("invalid_inputs", "image_header")
            digest.update(header)
            consumed += len(header)
            while chunk := stream.read(1024 * 1024):
                consumed += len(chunk)
                if consumed > size:
                    raise DiagnosticError("input_changed", "image_bytes")
                digest.update(chunk)
        if consumed != size:
            raise DiagnosticError("input_changed", "image_bytes")
        pin = digest.hexdigest()
        if pin in digests:
            raise DiagnosticError("invalid_inputs", "duplicate_image_bytes")
        digests.add(pin)
        records.append({"path": str(path), "sha256": pin, "bytes": size})
        paths.append(path)
    return tuple(paths), records


def _run_image_chain(paths, project_path, entry_ready, *, joint_endpoint_observer=False):
    """The accepted R4 seams, with one G call and bounded failure callbacks."""
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline.image_derived_entry import build_image_derived_entry
    from mc_photogrammetry.pipeline.image_derived_lattice_measure import measure_image_derived_lattice
    from mc_photogrammetry.pipeline.image_derived_signatures import derive_image_derived_signatures
    from mc_photogrammetry.pipeline.image_derived_graph_attribution import derive_image_derived_graph_attribution
    from mc_photogrammetry.reconstruction.plane_placement import PlacementRefusal
    from mc_photogrammetry.reconstruction import region_growth

    observer = region_growth._attrition_state.get()

    # Observe the declared lattice-through-matcher chain, excluding entry's
    # independent camera bootstrap. No inference stage is repeated.
    with region_growth._scalar_attrition_context(None):
        entry = build_image_derived_entry(
            paths, project_path=project_path, capture_ingress=True, max_frames=MAX_IMAGES,
            project_max_bytes=MAX_PROJECT_BYTES, retain_project_lease=True)
    entry_ready(entry)
    if entry.graph.status != "ok" or entry.pose_graph.status != "ok":
        raise DiagnosticError("upstream_unavailable", "entry")
    lattice = measure_image_derived_lattice(session=entry.session, pose_graph=entry.pose_graph)
    if isinstance(lattice, PlacementRefusal) or lattice.status != "assembled":
        raise DiagnosticError("upstream_unavailable", "lattice")
    witnesses = []
    signatures = derive_image_derived_signatures(
        session=entry.session, lattice_result=lattice,
        signature_witness_sink=witnesses)
    if isinstance(signatures, PlacementRefusal) or signatures.status != "assembled":
        raise DiagnosticError("upstream_unavailable", "signatures")
    failure, same_axis = _SingleCapture("failure"), _SingleCapture("same_axis")
    graph, attribution = derive_image_derived_graph_attribution(
        session=entry.session, lattice_result=lattice, signatures_result=signatures,
        connection_graph=entry.graph, solver_failure_observer=failure.append,
        same_axis_candidate_sink=same_axis.append,
        same_axis_capture_axes=graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL,
        **DEFAULT_SETTINGS)
    if isinstance(graph, PlacementRefusal) or attribution is None:
        raise DiagnosticError("upstream_unavailable", "graph")
    if observer is not None:
        observer.check("chain_stages", all(stage in observer.stages for stage in
                       ("frame", "seed", "proposal", "reading", "signature", "pair")))
    attrition = observer.record() if observer is not None else None
    from mc_photogrammetry.reconstruction import plane_graph
    endpoint_observer = plane_graph._ENDPOINT_CONFLICT_OBSERVER.get()
    endpoint_record = (_joint_endpoint_observation(graph) if joint_endpoint_observer else
                       endpoint_observer.record() if endpoint_observer is not None else None)
    return ChainCapture(entry, lattice, signatures, graph, attribution,
                        tuple(witnesses), failure, same_axis, attrition, endpoint_record)


def _capture_record(value, expected, validator, field, issues):
    if value is None:
        issues.append(field + ":missing")
        return {"status": "missing"}
    if type(value) is not expected:
        raise DiagnosticError("invalid_evidence", field)
    record = value.to_record()
    _bounded_bytes(record)
    validator(record)
    if record.get("status") != "complete":
        issues.append(field + ":" + str(record.get("status")))
    return record


def _summarize_attribution(trace, frames):
    """Stream-verify the full native digest; aggregate every pair without loss.

    Only examples are capped. The full canonical pair sequence is hashed in
    order, while exact counters and numeric ranges include all rows. No matcher
    or solver is invoked and no full pair-record array is allocated.
    """
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as trace_mod
    from mc_photogrammetry.reconstruction import plane_graph

    if type(trace.pair_attribution) is not tuple:
        raise DiagnosticError("invalid_evidence", "pair_rows")
    base = dataclasses.replace(trace, pair_attribution=(), attribution_sha256="").to_record(include_digest=False)
    metrics = ("psr", "score", "overlap_cells", "valid_cells", "overlap_fraction",
               "raw_max_support", "supported_max_support", "left_structural_share",
               "right_structural_share")
    groups, examples = {}, []
    replay, pairs_digest = hashlib.sha256(), hashlib.sha256()
    pairs_digest.update(b"[")
    replay.update(b"{")
    for key_index, key in enumerate(sorted(base)):
        if key_index:
            replay.update(b",")
        replay.update(_bounded_bytes(key) + b":")
        if key != "pairs":
            replay.update(_bounded_bytes(base[key]))
            continue
        replay.update(b"[")
        for row_index, row in enumerate(trace.pair_attribution):
            if (type(row) is not plane_graph.PlanePairTrace
                    or row.left not in frames or row.right not in frames or row.left == row.right
                    or type(row.admitted) is not bool or type(row.selected) is not bool
                    or type(row.reason) is not str
                    or row.reason not in ("", *plane_graph.MATCH_REFUSAL_REASONS)
                    or any(type(a) is not int or a not in (0, 1, 2) for a in (row.left_axis, row.right_axis))
                    or type(row.quadrant) is not int or row.quadrant not in (0, 1, 2, 3)):
                raise DiagnosticError("invalid_evidence", "pair_row")
            for name in ("raw_max_shift_rc", "supported_max_shift_rc", "cell_shift_rc"):
                shift = getattr(row, name)
                if shift is not None and (type(shift) is not tuple or len(shift) != 2
                                         or any(type(v) is not int or v.bit_length() > 64 for v in shift)):
                    raise DiagnosticError("invalid_evidence", "pair_shift")
            for name in ("raw_max_value", "supported_max_value", "left_plane", "right_plane", *metrics):
                value = getattr(row, name)
                if (value is not None and not (name == "psr" and value == float("inf"))
                        and (type(value) not in (int, float) or not math.isfinite(value))):
                    raise DiagnosticError("non_finite_evidence", "pair_" + name)
            pair_record = trace_mod._pair_record(row)
            encoded = _bounded_bytes(pair_record)
            if row_index:
                replay.update(b",")
                pairs_digest.update(b",")
            replay.update(encoded)
            pairs_digest.update(encoded)
            if len(examples) < MAX_PAIR_EXAMPLES:
                examples.append(pair_record)
            group_key = (row.left, row.right, row.left_axis, row.right_axis, row.quadrant)
            if group_key not in groups:
                groups[group_key] = {"rows": 0, "admitted": 0, "selected": 0,
                                     "refused_by_reason": {}, "metrics": {}}
            group = groups[group_key]
            group["rows"] += 1
            group["admitted"] += int(row.admitted)
            group["selected"] += int(row.selected)
            if not row.admitted:
                reasons = group["refused_by_reason"]
                reasons[row.reason] = reasons.get(row.reason, 0) + 1
            for name in metrics:
                value = getattr(row, name)
                bounds = group["metrics"].setdefault(name, {"finite_count": 0, "missing_count": 0,
                                                            "positive_infinity_count": 0, "min": None, "max": None})
                if value is None:
                    bounds["missing_count"] += 1
                elif value == float("inf"):
                    bounds["positive_infinity_count"] += 1
                else:
                    bounds["finite_count"] += 1
                    bounds["min"] = value if bounds["min"] is None else min(bounds["min"], value)
                    bounds["max"] = value if bounds["max"] is None else max(bounds["max"], value)
        replay.update(b"]")
    replay.update(b"}")
    pairs_digest.update(b"]")
    if replay.hexdigest() != trace.attribution_sha256:
        raise DiagnosticError("binding_mismatch", "attribution_digest")
    grouped_rows = [{"left": k[0], "right": k[1], "left_local_axis": k[2],
                     "right_local_axis": k[3], "relative_quadrant": k[4], **v}
                    for k, v in sorted(groups.items())]
    count = len(trace.pair_attribution)
    summary = {
        "status": "complete_aggregation", "source_row_count": count,
        "admitted_row_count": sum(g["admitted"] for g in groups.values()),
        "selected_row_count": sum(g["selected"] for g in groups.values()),
        "refused_row_count": sum(g["rows"] - g["admitted"] for g in groups.values()),
        "groups": grouped_rows, "pair_sequence_sha256": pairs_digest.hexdigest(),
        "verified_native_attribution_sha256": replay.hexdigest(),
        "examples_status": "truncated" if len(examples) < count else "complete",
        "retained_example_count": len(examples), "omitted_example_count": count - len(examples),
        "example_order": "first rows in the native deterministic sequence",
    }
    base["pairs"] = examples
    base["attribution_sha256"] = trace.attribution_sha256
    if len(examples) < count:
        base["pairs_capture"] = {"status": "truncated", "omitted_count": count - len(examples),
                                 "digest_basis": "complete native record verified by streaming replay"}
    return base, summary


def _verified_joint_graph_payload(graph):
    """Recompute the digest with the exact serializer used by the producer."""
    from mc_photogrammetry.pipeline import image_derived_joint_graph as joint_graph_mod

    if type(graph) is not joint_graph_mod.ImageDerivedGraphResult:
        raise DiagnosticError("invalid_evidence", "graph_type")
    payload = joint_graph_mod._payload(graph)
    _bounded_bytes(payload)
    if _digest(payload) != graph.result_sha256:
        raise DiagnosticError("binding_mismatch", "graph_digest")
    return payload


def _evidence(capture, inputs, *, source_region_masks=False):
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as trace_mod
    from mc_photogrammetry.pipeline.candidate_support_provenance import SourceTapWitnessBindingV1

    entry, lattice, signatures = capture.entry, capture.lattice, capture.signatures
    graph, trace = capture.graph, capture.attribution
    if type(graph) is not graph_mod.ImageDerivedGraphResult or type(trace) is not trace_mod.ImageDerivedGraphAttribution:
        raise DiagnosticError("invalid_evidence", "graph_type")
    graph_payload = _verified_joint_graph_payload(graph)
    trace_record, pair_summary = _summarize_attribution(trace, graph.frame_ids)
    _bounded_bytes(trace_record)
    stage = {
        "entry": _sha(entry.entry_sha256, "entry_digest"),
        "session": _sha(entry.session.session_sha256, "session_digest"),
        "lattice": _sha(lattice.result_sha256, "lattice_digest"),
        "signatures": _sha(signatures.result_sha256, "signatures_digest"),
        "connections": _sha(entry.graph.graph_sha256, "connections_digest"),
        "graph": _sha(graph.result_sha256, "graph_digest"),
        "attribution": _sha(trace.attribution_sha256, "attribution_digest"),
    }
    if (graph.session_sha256 != stage["session"] or graph.lattice_sha256 != stage["lattice"]
            or graph.signatures_sha256 != stage["signatures"]
            or trace.graph_result_sha256 != stage["graph"]
            or trace.session_sha256 != stage["session"] or trace.lattice_sha256 != stage["lattice"]
            or trace.signatures_sha256 != stage["signatures"]
            or lattice.session_sha256 != stage["session"]
            or signatures.session_sha256 != stage["session"]
            or signatures.lattice_sha256 != stage["lattice"]
            or entry.graph.session_sha256 != stage["session"]):
        raise DiagnosticError("binding_mismatch", "stage_digests")
    settings = dict(trace.effective_settings)
    if any(settings.get(key) != value for key, value in DEFAULT_SETTINGS.items()):
        raise DiagnosticError("binding_mismatch", "default_settings")
    frames = entry.session.frames
    if len(frames) != len(inputs) or len(frames) > MAX_IMAGES:
        raise DiagnosticError("binding_mismatch", "input_frames")
    ingress = []
    for frame, input_record in zip(frames, inputs):
        pin = frame.ingress
        if (pin is None or pin.byte_sha256 != input_record["sha256"]
                or pin.byte_length != input_record["bytes"]):
            raise DiagnosticError("binding_mismatch", "input_ingress")
        ingress.append(pin.to_record())
    issues = []
    attrition = capture.vertical_attrition
    if attrition is not None:
        _validate_attrition_record(attrition)
        if attrition["status"] != "complete":
            issues.append("vertical_attrition:incomplete")
    endpoint_record = capture.endpoint_conflicts
    if endpoint_record is not None:
        if endpoint_record.get("schema") == JOINT_ENDPOINT_SCHEMA:
            _validate_joint_endpoint_observation(endpoint_record, graph)
            if endpoint_record["status"] != "complete":
                issues.append("joint_endpoint_selection:" + endpoint_record["status"])
        else:
            _validate_endpoint_conflicts(endpoint_record)
            if endpoint_record["status"] != "complete":
                issues.append("endpoint_period_conflicts:incomplete")
            elif endpoint_record["frames"] != list(graph.frame_ids):
                raise DiagnosticError("binding_mismatch", "endpoint_conflict_frames")
            elif (endpoint_record["visited_assignments"] > 0 and
                  (endpoint_record["conflicted_assignments"] == endpoint_record["visited_assignments"])
                  != (graph.reason == "endpoint_period_conflict")):
                raise DiagnosticError("binding_mismatch", "endpoint_conflict_reason")
    if pair_summary["examples_status"] == "truncated":
        issues.append("pair_attribution_examples:truncated")
    witnesses = []
    if len(capture.signature_witnesses) > MAX_JSON_ITEMS:
        raise DiagnosticError("evidence_cap", "signature_witnesses")
    for witness in capture.signature_witnesses:
        if isinstance(witness, SourceTapWitnessBindingV1):
            record = witness.to_record()
            record["status"] = "bound"
        else:
            record = {"status": "refused", "reason": getattr(witness, "reason", "missing")}
            issues.append("signature_witness:refused")
        witnesses.append(record)
    if len(witnesses) != len(signatures.signatures):
        issues.append("signature_witness:missing")
    for sink in (capture.failure, capture.same_axis):
        if sink.count not in (0, 1) or (sink.count == 0) != (sink.value is None):
            raise DiagnosticError("callback_multiplicity", sink.field)
    snapshot = capture.failure.value
    if snapshot is not None and type(snapshot) is not graph_mod.SolverFailureSnapshot:
        raise DiagnosticError("invalid_evidence", "snapshot_type")
    raw, context, bindings, candidate = None, None, None, None
    metrics = [{"axis": axis, "resultant": None, "residual": None, "observations": None}
               for axis in range(3)]
    if snapshot is None:
        issues.append("solver_failure:missing")
        if capture.same_axis.value is not None:
            raise DiagnosticError("binding_mismatch", "orphan_candidate_capture")
    else:
        raw = trace_mod.solver_failure_snapshot_record(snapshot)
        diagnostic = trace_mod.solver_failure_diagnostic(
            snapshot, DEFAULT_SETTINGS["min_resultant"], DEFAULT_SETTINGS["integer_tolerance"])
        if snapshot.reason != graph.reason or trace.solver_failure != diagnostic:
            raise DiagnosticError("binding_mismatch", "failure_attribution")
        if not diagnostic.recorded or snapshot.invalid_field is not None:
            issues.append("solver_failure:invalid_or_partial")
        else:
            metrics = [{"axis": a, "resultant": r, "residual": e, "observations": n}
                       for a, r, e, n in diagnostic.axes]
            if any(row["resultant"] is None or row["residual"] is None
                   or row["observations"] is None for row in metrics):
                issues.append("axis_metrics:missing")
        context = _capture_record(
            snapshot.solve_context, graph_mod.SolverSolveContext,
            lambda record: graph_mod.validate_solver_solve_context(record, snapshot.frames),
            "solve_context", issues)
        bindings = _capture_record(snapshot.solver_input_bindings, graph_mod.SolverInputBindings,
                                   graph_mod.validate_solver_input_bindings, "solver_input_bindings", issues)
        candidate = _capture_record(capture.same_axis.value, graph_mod.SameAxisCandidateCapture,
                                    graph_mod.validate_same_axis_candidates, "same_axis_candidates", issues)
        if (candidate.get("status") == "complete"
                and candidate.get("schema") == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V3):
            if candidate["selection_sha256"] != graph.selection.result_sha256:
                raise DiagnosticError("binding_mismatch", "candidate_selection")
            if (trace.schema != trace_mod.IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA
                    or candidate["admitted_ledger_sha256"] != trace.matcher_admitted_sha256
                    or len(candidate["admitted_rows"]) != trace.post_admission_candidates):
                raise DiagnosticError("binding_mismatch", "candidate_admitted_ledger")
        if candidate.get("status") == "complete" and candidate.get("axes") != [0, 1, 2]:
            raise DiagnosticError("binding_mismatch", "candidate_axes")
        for record in (bindings, candidate):
            if record.get("status") == "complete" and (
                    record["frames"] != list(snapshot.frames)
                    or record["assignments"] != list(snapshot.quadrants)):
                raise DiagnosticError("binding_mismatch", "failure_assignment")
        if bindings.get("status") == candidate.get("status") == "complete":
            for key in ("failed_assignment_sha256", "input_match_identity_sha256", "input_match_count"):
                if bindings[key] != candidate[key]:
                    raise DiagnosticError("binding_mismatch", key)
        if context.get("status") == "complete":
            for key in ("integer_tolerance", "min_resultant", "ambiguous_margin"):
                if context[key] != DEFAULT_SETTINGS[key]:
                    raise DiagnosticError("binding_mismatch", "solve_context_settings")
    record = {
        "schema": SCHEMA, "status": "incomplete" if issues else "complete",
        "issues": sorted(set(issues)), "developer_diagnostic_only": True,
        "reconstruction_authorized": False, "inputs": inputs, "stage_digests": stage,
        "default_solver_settings": dict(DEFAULT_SETTINGS), "ingress": ingress,
        "signature_witnesses": witnesses,
        "graph": {"result_sha256": graph.result_sha256, "payload": graph_payload},
        "attribution": trace_record, "pair_attribution_summary": pair_summary,
        "vertical_attrition": attrition,
        "solver_failure_snapshot": raw,
        "solve_context": context, "solver_input_bindings": bindings,
        "same_axis_candidates": candidate, "axis_metrics": metrics,
        "callback_counts": {"solver_failure": capture.failure.count, "same_axis_candidates": capture.same_axis.count},
        "limits": {"json_bytes": MAX_RECORD_BYTES, "json_items": MAX_JSON_ITEMS,
                   "pair_examples": MAX_PAIR_EXAMPLES, "image_frames": MAX_IMAGES},
        "alternatives": {"status": "not_captured_by_existing_failure_snapshot",
                         "supported_complete_alternative_established": False},
    }
    if endpoint_record is not None:
        endpoint_record = dict(endpoint_record)
        if endpoint_record["schema"] == JOINT_ENDPOINT_SCHEMA:
            record["joint_endpoint_selection"] = endpoint_record
        else:
            endpoint_record["stage_bindings"] = {
                "session_sha256": stage["session"], "lattice_sha256": stage["lattice"],
                "signatures_sha256": stage["signatures"], "graph_sha256": stage["graph"],
                "solve_context_sha256": (context.get("record_sha256") if context else None),
            }
            endpoint_record["record_sha256"] = _digest(endpoint_record)
            record["endpoint_period_conflicts"] = endpoint_record
    if source_region_masks:
        record["schema"] = MASK_DIAGNOSTIC_SCHEMA
        record["source_region_masks"] = _source_region_mask_bundle(capture, record)
    record["record_sha256"] = _digest(record)
    _bounded_bytes(record)
    return record


def _mask_code_pins():
    paths = ("src/mc_photogrammetry/reconstruction/lattice_planes.py",
             "src/mc_photogrammetry/pipeline/image_derived_signatures.py",
             "src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py",
             "src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py",
             "src/mc_photogrammetry/pipeline/image_derived_joint_graph.py",
             "scripts/evaluation/image_point_candidate_roster.py",
             "scripts/evaluation/diagnose_image_geometry_failure.py")
    return {path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() for path in paths}


def _mask_source_context(record):
    """Read source joins already authenticated by the existing diagnostic path."""
    from mc_photogrammetry.reconstruction.lattice_planes import _canonical_region_polygon
    from mc_photogrammetry.pipeline import image_derived_joint_graph as graph_source
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection_source
    _bounded_bytes(record)
    try:
        joint = record["joint_endpoint_selection"]
        witness = joint["first_witness_provenance"]
        ingress = record["ingress"]
        if (joint["schema"] != JOINT_ENDPOINT_SCHEMA or joint["producer"] != JOINT_ENDPOINT_PRODUCER
                or witness["type"] != "available" or witness["partial_exclusion_index"] != 0
                or len(ingress) != 2 or len(witness["readings"]) != 3 or len(witness["rows"]) != 2):
            raise DiagnosticError("invalid_evidence", "mask_source_join")
        if ([row["source_row"] for row in witness["rows"]] != list(MASK_ROWS)
                or [tuple(row["endpoint"]) for row in witness["readings"]] != list(MASK_ENDPOINTS)
                or [pin["frame_id"] for pin in ingress] != ["image-0000", "image-0001"]):
            raise DiagnosticError("binding_mismatch", "mask_source_endpoints")
        expected_directions = [[list(MASK_ENDPOINTS[0]), list(MASK_ENDPOINTS[2])],
                               [list(MASK_ENDPOINTS[1]), list(MASK_ENDPOINTS[2])]]
        if [row["original_direction"] for row in witness["rows"]] != expected_directions:
            raise DiagnosticError("binding_mismatch", "mask_source_rows")
        for pin in ingress:
            if set(pin) != {"frame_id", "source_label", "byte_sha256", "byte_length", "decoded_shape",
                            "decoded_dtype", "decoded_pixel_sha256", "transform", "schema", "decoder"}:
                raise DiagnosticError("invalid_evidence", "mask_ingress_fields")
            if pin["schema"] != "mc-photogrammetry-frame-ingress-provenance-v1":
                raise DiagnosticError("invalid_evidence", "mask_ingress_schema")
            shape = pin["decoded_shape"]
            if (type(shape) is not list or len(shape) != 3 or shape[2] != 3
                    or any(type(v) is not int or v <= 0 for v in shape)
                    or pin["decoded_dtype"] != "|u1"
                    or pin["decoder"] != "input.image.load_image:exif_transpose+convert('RGB')"):
                raise DiagnosticError("invalid_evidence", "mask_ingress")
            transform = pin["transform"]
            size = [shape[1], shape[0]]
            if (type(transform) is not dict or set(transform) != {"source_size", "crop_origin", "crop_size", "output_size"}
                    or transform["source_size"] != size or transform["crop_origin"] != [0, 0]
                    or transform["crop_size"] not in (None, size) or transform["output_size"] not in (None, size)):
                raise DiagnosticError("binding_mismatch", "mask_identity_transform")
            if any(value is not None and (type(value) is not list or len(value) != 2
                   or any(type(v) is not int for v in value)) for value in transform.values()):
                raise DiagnosticError("binding_mismatch", "mask_identity_transform")
            for name in ("byte_sha256", "decoded_pixel_sha256"):
                _sha(pin[name], "mask_ingress_digest")
            if type(pin["byte_length"]) is not int or pin["byte_length"] <= 0:
                raise DiagnosticError("invalid_evidence", "mask_ingress_length")
        for key in ("proof_sha256", "candidate_matches_sha256", "selection_sha256", "reading_bindings_sha256", "record_sha256"):
            _sha(joint[key], "mask_certificate_digest")
        joint_body = dict(joint); joint_body.pop("record_sha256")
        if _digest(joint_body, limit=MAX_JOINT_ENDPOINT_BYTES) != joint["record_sha256"]:
            raise DiagnosticError("binding_mismatch", "mask_joint_record_digest")
        if (witness["proof_sha256"] != joint["proof_sha256"]
                or witness["source_universe_sha256"] != joint["candidate_matches_sha256"]):
            raise DiagnosticError("binding_mismatch", "mask_certificate_join")
        for key, value in record["stage_digests"].items():
            _sha(value, "mask_stage_digest")
        for key in ("session", "lattice", "signatures", "graph"):
            if joint["source_bindings"][key + "_sha256"] != record["stage_digests"][key]:
                raise DiagnosticError("binding_mismatch", "mask_stage_join")
        rows = [{"source_row": row["source_row"], "row_sha256": _sha(row["row_sha256"], "mask_row_digest"),
                 "original_direction": row["original_direction"]} for row in witness["rows"]]
        readings = []
        for reading in witness["readings"]:
            region = reading["source_region"]
            if region["type"] != "available":
                raise DiagnosticError("invalid_evidence", "missing_source_mask")
            polygon = _canonical_region_polygon(reading["polygon"])
            readings.append({"endpoint": reading["endpoint"], "reading_index": reading["reading_index"],
                             "reading_binding_sha256": _sha(reading["reading_binding_sha256"], "mask_reading_digest"),
                             "canonical_polygon": [list(p) for p in polygon], "region": region})
        return {"stage_digests": record["stage_digests"], "ingress": ingress,
                "source_producers": {"graph_schema": graph_source.IMAGE_DERIVED_GRAPH_SCHEMA,
                    "graph_producer": graph_source.IMAGE_DERIVED_GRAPH_PRODUCER,
                    "selection_schema": selection_source.ENDPOINT_SELECTION_SCHEMA,
                    "selection_producer": selection_source.ENDPOINT_SELECTION_PRODUCER,
                    "joint_schema": JOINT_ENDPOINT_SCHEMA, "joint_producer": JOINT_ENDPOINT_PRODUCER},
                "proof_sha256": joint["proof_sha256"], "source_universe_sha256": joint["candidate_matches_sha256"],
                "selection_sha256": joint["selection_sha256"], "reading_bindings_sha256": joint["reading_bindings_sha256"],
                "joint_record_sha256": joint["record_sha256"], "rows": rows, "readings": readings}
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise DiagnosticError("invalid_evidence", "mask_source_context") from error


def _preflight_mask_entries(entries):
    """Complete preallocation pass; never decode a prefix before later caps."""
    mask_keys = {"source_region_id", "axis", "sources", "image_shape", "bbox_xyxy", "crop_shape",
                 "encoding", "bit_order", "packed_mask_base64", "mask_sha256", "pixel_count"}
    entry_keys = {"endpoint", "reading_index", "reading_binding_sha256", "canonical_polygon",
                  "ingress_sha256", "row_refs", "mask", "full_native_bool_c_sha256"}
    if type(entries) is not list or len(entries) != 3:
        raise DiagnosticError("invalid_evidence", "mask_count")
    total_pixels = total_packed = 0
    for entry in entries:
        if type(entry) is not dict or set(entry) != entry_keys:
            raise DiagnosticError("invalid_evidence", "mask_entry_fields")
        endpoint = entry["endpoint"]
        if (type(endpoint) is not list or len(endpoint) != 2 or type(endpoint[0]) is not str
                or type(endpoint[1]) is not int or endpoint[1] < 0
                or type(entry["reading_index"]) is not int or entry["reading_index"] < 0
                or type(entry["row_refs"]) is not list):
            raise DiagnosticError("invalid_evidence", "mask_entry_identity")
        mask = entry["mask"]
        if type(mask) is not dict or set(mask) != mask_keys:
            raise DiagnosticError("invalid_evidence", "mask_fields")
        if mask["encoding"] != "packbits-base64" or mask["bit_order"] != "little":
            raise DiagnosticError("invalid_evidence", "mask_encoding")
        for key, size in (("image_shape", 2), ("crop_shape", 2), ("bbox_xyxy", 4)):
            if (type(mask[key]) is not list or len(mask[key]) != size
                    or any(type(v) is not int for v in mask[key])):
                raise DiagnosticError("invalid_evidence", "mask_dimensions")
        height, width = mask["image_shape"]
        ch, cw = mask["crop_shape"]
        x0, y0, x1, y1 = mask["bbox_xyxy"]
        count = mask["pixel_count"]
        if (min(height, width, ch, cw) <= 0 or x0 < 0 or y0 < 0 or x1 > width or y1 > height
                or (ch, cw) != (y1 - y0, x1 - x0) or type(count) is not int or not 0 < count <= ch * cw):
            raise DiagnosticError("invalid_evidence", "mask_bounds")
        full = height * width
        packed = (ch * cw + 7) // 8
        total_pixels += full
        total_packed += packed
        if full > MAX_MASK_PIXELS or total_pixels > MAX_TOTAL_MASK_PIXELS:
            raise DiagnosticError("evidence_cap", "mask_pixels")
        if packed > MAX_PACKED_MASK_BYTES or total_packed > MAX_PACKED_MASK_BYTES:
            raise DiagnosticError("evidence_cap", "mask_packed_bytes")
        payload = mask["packed_mask_base64"]
        if type(payload) is not str or len(payload) != 4 * ((packed + 2) // 3):
            raise DiagnosticError("invalid_evidence", "mask_base64_length")
        if type(mask["axis"]) is not int or mask["axis"] not in (0, 1, 2):
            raise DiagnosticError("invalid_evidence", "mask_axis")
        if (type(mask["sources"]) is not list or not mask["sources"]
                or any(type(v) is not str or not v for v in mask["sources"])
                or mask["sources"] != sorted(mask["sources"])):
            raise DiagnosticError("invalid_evidence", "mask_sources")
        polygon = entry["canonical_polygon"]
        if (type(polygon) is not list or not 3 <= len(polygon) <= MAX_JSON_ITEMS
                or any(type(p) is not list or len(p) != 2 or any(type(v) not in (int, float)
                       or not math.isfinite(v) for v in p) for p in polygon)):
            raise DiagnosticError("invalid_evidence", "mask_polygon")
        for key in ("source_region_id", "mask_sha256"):
            _sha(mask[key], "mask_digest")
        for key in ("reading_binding_sha256", "ingress_sha256", "full_native_bool_c_sha256"):
            _sha(entry[key], "mask_entry_digest")
    return {"full_pixels": total_pixels, "packed_bytes": total_packed}


def _decode_source_mask(entry):
    from mc_photogrammetry.reconstruction.lattice_planes import PlaneRegionEvidence, _decode_region_evidence
    values = dict(entry["mask"])
    for key in ("sources", "image_shape", "bbox_xyxy", "crop_shape"):
        values[key] = tuple(values[key])
    try:
        array = _decode_region_evidence(PlaneRegionEvidence(**values), values["axis"],
                                       tuple(tuple(p) for p in entry["canonical_polygon"]))
    except (ValueError, TypeError, OverflowError) as error:
        raise DiagnosticError("invalid_evidence", "mask_payload") from error
    if hashlib.sha256(array.tobytes(order="C")).hexdigest() != entry["full_native_bool_c_sha256"]:
        raise DiagnosticError("binding_mismatch", "mask_full_native_digest")
    array.setflags(write=False)
    return array


def _validate_source_region_masks(bundle, record):
    if (type(bundle) is not dict or set(bundle) != {"schema", "producer", "status", "source_context",
            "code_pins", "limits", "entries", "record_sha256"}
            or bundle.get("schema") != MASK_BUNDLE_SCHEMA or bundle.get("producer") != MASK_BUNDLE_PRODUCER
            or bundle.get("status") != "complete"):
        raise DiagnosticError("invalid_evidence", "mask_bundle_schema")
    _bounded_bytes(bundle, limit=MAX_MASK_BUNDLE_BYTES)
    _preflight_mask_entries(bundle["entries"])
    context = _mask_source_context(record)
    if (_bounded_bytes(bundle["source_context"]) != _bounded_bytes(context)
            or bundle["code_pins"] != _mask_code_pins() or bundle["limits"] != _mask_limits()):
        raise DiagnosticError("binding_mismatch", "mask_bundle_source")
    body = dict(bundle); checksum = body.pop("record_sha256")
    if checksum != _digest(body, limit=MAX_MASK_BUNDLE_BYTES):
        raise DiagnosticError("binding_mismatch", "mask_bundle_digest")
    entries = bundle["entries"]
    _validate_mask_entry_joins(entries, context)
    # All entries, joins and caps close before the first unpack/full allocation.
    for entry in entries:
        array = _decode_source_mask(entry)
        del array
    return bundle


def _validate_mask_entry_joins(entries, context):
    if ([tuple(e["endpoint"]) for e in entries] != list(MASK_ENDPOINTS)
            or len({e["mask"]["source_region_id"] for e in entries}) != 3):
        raise DiagnosticError("binding_mismatch", "mask_bundle_endpoints")
    pins = {pin["frame_id"]: pin for pin in context["ingress"]}
    for entry, reading in zip(entries, context["readings"]):
        endpoint = entry["endpoint"]
        region = dict(reading["region"]); region.pop("type")
        actual = dict(entry["mask"]); actual.pop("packed_mask_base64")
        refs = [row for row in context["rows"] if endpoint in row["original_direction"]]
        if (actual != region or entry["canonical_polygon"] != reading["canonical_polygon"]
                or entry["reading_index"] != reading["reading_index"]
                or entry["reading_binding_sha256"] != reading["reading_binding_sha256"]
                or entry["row_refs"] != refs or entry["ingress_sha256"] != _digest(pins[endpoint[0]])
                or entry["mask"]["image_shape"] != pins[endpoint[0]]["decoded_shape"][:2]):
            raise DiagnosticError("binding_mismatch", "mask_entry_join")


def _mask_limits():
    return {"mask_pixels": MAX_MASK_PIXELS, "aggregate_pixels": MAX_TOTAL_MASK_PIXELS,
            "aggregate_packed_bytes": MAX_PACKED_MASK_BYTES, "bundle_bytes": MAX_MASK_BUNDLE_BYTES}


def _source_region_mask_bundle(capture, record):
    from mc_photogrammetry.reconstruction.lattice_planes import _decode_region_evidence
    context = _mask_source_context(record)
    bindings = {}
    for binding in capture.graph.selection.reading_bindings:
        endpoint = tuple(binding[0])
        if endpoint in bindings:
            raise DiagnosticError("binding_mismatch", "mask_duplicate_live_binding")
        bindings[endpoint] = binding
    pins = {pin["frame_id"]: pin for pin in context["ingress"]}
    entries = []
    for source in context["readings"]:
        endpoint = source["endpoint"]
        binding = bindings.get(tuple(endpoint))
        if (binding is None or binding[1] != source["reading_index"]
                or binding[3] != source["reading_binding_sha256"]):
            raise DiagnosticError("binding_mismatch", "mask_live_reading_join")
        reading = capture.graph.readings[binding[1]]
        if reading.frame != endpoint[0] or reading.axis != binding[2]:
            raise DiagnosticError("binding_mismatch", "mask_live_reading_join")
        if reading.region_evidence is None:
            raise DiagnosticError("invalid_evidence", "missing_source_mask")
        mask = reading.region_evidence.to_record()
        entries.append({"endpoint": endpoint, "reading_index": binding[1],
                        "reading_binding_sha256": binding[3], "canonical_polygon": source["canonical_polygon"],
                        "ingress_sha256": _digest(pins[endpoint[0]]),
                        "row_refs": [row for row in context["rows"] if endpoint in row["original_direction"]],
                        "mask": mask, "full_native_bool_c_sha256": "0" * 64})
    _preflight_mask_entries(entries)
    _validate_mask_entry_joins(entries, context)
    # Bound the complete potential bundle, including fixed-width hashes, BEFORE decode.
    bundle = {"schema": MASK_BUNDLE_SCHEMA, "producer": MASK_BUNDLE_PRODUCER, "status": "complete",
              "source_context": context, "code_pins": _mask_code_pins(), "limits": _mask_limits(),
              "entries": entries, "record_sha256": "0" * 64}
    _bounded_bytes(bundle, limit=MAX_MASK_BUNDLE_BYTES)
    for entry, source in zip(entries, context["readings"]):
        reading = capture.graph.readings[entry["reading_index"]]
        from mc_photogrammetry.reconstruction.lattice_planes import _canonical_region_polygon
        if [list(p) for p in _canonical_region_polygon(reading.polygon)] != source["canonical_polygon"]:
            raise DiagnosticError("binding_mismatch", "mask_live_region_join")
        try:
            array = _decode_region_evidence(reading.region_evidence, reading.axis, reading.polygon)
        except (ValueError, TypeError) as error:
            raise DiagnosticError("invalid_evidence", "mask_payload") from error
        entry["full_native_bool_c_sha256"] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        del array
    body = dict(bundle); body.pop("record_sha256")
    bundle["record_sha256"] = _digest(body, limit=MAX_MASK_BUNDLE_BYTES)
    return _validate_source_region_masks(bundle, record)


def load_native_source_region_mask(bundle, record, endpoint):
    """Validate all bindings first, then return one owned readonly bool raster."""
    _validate_source_region_masks(bundle, record)
    for entry in bundle["entries"]:
        if tuple(entry["endpoint"]) == tuple(endpoint):
            return _decode_source_mask(entry)
    raise DiagnosticError("binding_mismatch", "mask_consumer_endpoint")


def _strict_readback(path, encoded, validator=None):
    with path.open("rb") as stream:
        actual = stream.read(MAX_RECORD_BYTES + 1)
    if len(actual) > MAX_RECORD_BYTES or actual != encoded:
        raise DiagnosticError("binding_mismatch", "publication_readback")
    if validator is not None:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise DiagnosticError("invalid_evidence", "duplicate_json_key")
                result[key] = value
            return result
        try:
            record = json.loads(actual, object_pairs_hook=unique,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        except (ValueError, TypeError) as error:
            raise DiagnosticError("invalid_evidence", "publication_json") from error
        if _bounded_bytes(record) + b"\n" != actual:
            raise DiagnosticError("binding_mismatch", "publication_canonical")
        validator(record)
    return hashlib.sha256(actual).hexdigest()


def _validate_mask_diagnostic(record):
    if record.get("schema") != MASK_DIAGNOSTIC_SCHEMA:
        raise DiagnosticError("invalid_evidence", "mask_diagnostic_schema")
    body = dict(record); checksum = body.pop("record_sha256", None)
    if checksum != _digest(body):
        raise DiagnosticError("binding_mismatch", "mask_diagnostic_digest")
    _validate_source_region_masks(record.get("source_region_masks"), record)


def _publish_new(destination, encoded, *, validator=None):
    """Publish fully flushed bytes with an atomic, no-overwrite hard link.

    The single fixed temp name is exclusively created in the final directory.
    An unsupported link or a final-path collision raises without publishing our
    bytes; there is no overwrite or copy fallback.
    """
    temporary = destination.with_name(".diagnostic.json.partial")
    created_temp = False
    try:
        with temporary.open("xb") as stream:
            created_temp = True
            written = stream.write(encoded)
            if written != len(encoded):
                raise OSError("short diagnostic write")
            stream.flush()
            os.fsync(stream.fileno())
        _strict_readback(temporary, encoded, validator)
        os.link(temporary, destination)
        _strict_readback(destination, encoded, validator)
    finally:
        if created_temp:
            temporary.unlink()


def run(*, images, output, local_period_support=False):
    return _run(images=images, output=output, local_period_support=local_period_support)


def _run(*, images, output, local_period_support=False, source_region_masks=False):
    """Publish one create-new diagnostic; roll back only our staging artifacts."""
    if type(local_period_support) is not bool:
        raise DiagnosticError("invalid_inputs", "local_period_support")
    if type(source_region_masks) is not bool:
        raise DiagnosticError("invalid_inputs", "source_region_masks")
    root = Path(output).absolute()
    if root.exists() or not root.parent.is_dir():
        raise DiagnosticError("output_collision_or_missing_parent", "output")
    paths, inputs = _input_records(images)
    root.mkdir(exist_ok=False)
    held = []
    published = False
    destination = root / "diagnostic.json"
    try:
        from mc_photogrammetry.reconstruction import region_growth, lattice_planes
        observer = region_growth._ScalarAttritionObserver()
        with (region_growth._scalar_attrition_context(observer),
              lattice_planes._local_period_diagnostic_context(observer, enabled=local_period_support)):
            capture = _run_image_chain(paths, root / "entry.mcscan", held.append,
                                       joint_endpoint_observer=True)
        record = (_evidence(capture, inputs, source_region_masks=True) if source_region_masks
                  else _evidence(capture, inputs))
        # Bind hashes to the bytes actually consumed through the accepted ingress
        # and detect post-discovery replacement before publication.
        _, after = _input_records(paths)
        if after != inputs:
            raise DiagnosticError("input_changed", "image_bytes")
        encoded = _bounded_bytes(record) + b"\n"
        for entry in held:
            lease = getattr(entry, "project_lease", None)
            if lease is not None:
                lease.discard()
        if source_region_masks:
            _publish_new(destination, encoded, validator=_validate_mask_diagnostic)
        else:
            _publish_new(destination, encoded)
        published = True
        return {"status": record["status"], "path": str(destination),
                "file_sha256": hashlib.sha256(encoded).hexdigest(),
                "record_sha256": record["record_sha256"],
                "graph_reason": capture.graph.reason,
                "stage_digests": record["stage_digests"],
                "issues": record["issues"], "axis_metrics": record["axis_metrics"]}
    finally:
        for entry in held:
            lease = getattr(entry, "project_lease", None)
            if lease is not None:
                lease.discard()
        if not published:
            # Never remove a colliding path; only remove our empty directory.
            try:
                root.rmdir()
            except OSError:
                pass


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True,
                        help="ordinary PNG/JPEG file; repeat for each view")
    parser.add_argument("--output", required=True, help="new directory in an existing parent")
    parser.add_argument("--local-period-support", action="store_true",
                        help="diagnose five fixed local windows of failed parent periods")
    parser.add_argument("--source-region-masks", action="store_true", default=argparse.SUPPRESS,
                        help="developer-only exact three native source-mask inline export")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if getattr(args, "source_region_masks", False):
            result = _run(images=args.image, output=args.output,
                          local_period_support=args.local_period_support, source_region_masks=True)
        else:
            result = run(images=args.image, output=args.output,
                         local_period_support=args.local_period_support)
    except (DiagnosticError, OSError, ValueError) as error:
        refusal = {"status": "refused", "reason": getattr(error, "reason", type(error).__name__),
                   "field": getattr(error, "field", "")}
        if isinstance(error, DiagnosticError) and error.context is not None:
            refusal["context"] = error.context
        print(json.dumps(refusal), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
