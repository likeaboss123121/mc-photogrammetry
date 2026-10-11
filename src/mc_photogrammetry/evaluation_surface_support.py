"""Evaluation-only gate for exact surface-local matcher support.

Current signature records retain aggregate cell values, not the source-pixel
tap witness required by DS-R3-AR.  This module binds and audits the available
inputs, then refuses atomically rather than substituting mask intersections or
whole-reading metrics for surface-local evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from typing import Any

import numpy as np

from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry import evaluation_signature_sampling as sampling_mod
from mc_photogrammetry import synthetic_targeted_capture_preflight as preflight_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signature_mod
from mc_photogrammetry.reconstruction import plane_graph

SCHEMA = "mc-photogrammetry-surface-local-support-v1"
SAMPLING_CONTRACT = "rectify-bilinear-gradient-cell-pair-v1"
DEFAULT_MAX_SOURCE_PIXELS = 20_000_000
DEFAULT_MAX_PAIR_CELLS = 10_000_000
_MISSING = (
    "rectification_nonzero_source_taps",
    "gradient_neighbor_source_taps",
    "cell_aggregation_source_taps",
    "renderer_subface_plane_identity",
)


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _match_record(value: plane_graph.PlaneMatch) -> dict[str, Any]:
    psr = float(value.psr)
    if np.isnan(psr) or psr == float("-inf"):
        raise ValueError("match PSR is invalid")
    return {
        "left": value.left, "right": value.right,
        "left_axis": value.left_axis, "right_axis": value.right_axis,
        "left_b": float(value.left_b), "right_b": float(value.right_b),
        "quadrant": value.quadrant_of_right,
        "psr": {"value": None if psr == float("inf") else psr,
                "positive_infinity": psr == float("inf")},
        "peak": float(value.peak), "second_peak": float(value.second_peak),
        "overlap_cells": value.overlap_cells, "score": float(value.score),
        "left_plane": value.left_plane, "right_plane": value.right_plane,
        "cell_shift_rc": None if value.cell_shift_rc is None
        else list(value.cell_shift_rc),
    }


@dataclass(frozen=True, slots=True)
class SurfaceLocalSupportRefusal:
    schema: str
    status: str
    reason: str
    detail: str
    sampling_contract: str
    correlation_algorithm_id: str
    max_shift_cells: int
    left_truth_sha256: str
    right_truth_sha256: str
    left_evidence_sha256: str
    right_evidence_sha256: str
    left_signature_sha256: str
    right_signature_sha256: str
    match_sha256: str
    homography_sha256: str
    left_ownership_sha256: str
    right_ownership_sha256: str
    missing_dependencies: tuple[str, ...]
    whole_reading_overlap_cells: int | None
    whole_reading_psr: float | str | None
    report_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != SCHEMA or self.status != "refused"
                or self.reason not in ("provenance_mismatch", "budget_exhausted",
                                       "unavailable_exact_dependencies")):
            raise ValueError("surface-support refusal is malformed")
        digest = _sha(self.to_record(include_digest=False))
        if self.report_sha256 not in ("", digest):
            raise ValueError("surface-support refusal digest mismatch")
        object.__setattr__(self, "report_sha256", digest)

    def to_record(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)
                  if field.name != "report_sha256"}
        result["missing_dependencies"] = list(self.missing_dependencies)
        if include_digest:
            result["report_sha256"] = self.report_sha256
        return result


def _refusal(reason: str, detail: str, **bindings: Any) -> SurfaceLocalSupportRefusal:
    defaults = dict(
        sampling_contract=SAMPLING_CONTRACT, correlation_algorithm_id="",
        max_shift_cells=0, left_truth_sha256="", right_truth_sha256="",
        left_evidence_sha256="", right_evidence_sha256="",
        left_signature_sha256="", right_signature_sha256="", match_sha256="",
        homography_sha256="", left_ownership_sha256="",
        right_ownership_sha256="", missing_dependencies=(),
        whole_reading_overlap_cells=None, whole_reading_psr=None)
    defaults.update(bindings)
    return SurfaceLocalSupportRefusal(
        SCHEMA, "refused", reason, detail, **defaults)


def evaluate_surface_support(
    *, left_truth: truth_mod.RenderedTruthEvidence,
    right_truth: truth_mod.RenderedTruthEvidence,
    left_signature: signature_mod.SignatureEvidence,
    right_signature: signature_mod.SignatureEvidence,
    match: plane_graph.PlaneMatch, homography: np.ndarray,
    correlation_algorithm_id: str, max_shift_cells: int,
    max_source_pixels: int = DEFAULT_MAX_SOURCE_PIXELS,
) -> SurfaceLocalSupportRefusal:
    """Audit DS-R3-AR inputs and refuse until exact tap witnesses exist."""
    try:
        if (not isinstance(left_truth, truth_mod.RenderedTruthEvidence)
                or not isinstance(right_truth, truth_mod.RenderedTruthEvidence)
                or not isinstance(left_signature, signature_mod.SignatureEvidence)
                or not isinstance(right_signature, signature_mod.SignatureEvidence)
                or not isinstance(match, plane_graph.PlaneMatch)):
            raise ValueError("surface-support inputs have unsupported types")
        # Constructor replay verifies canonical evidence/signature digests and
        # immutable arrays without modifying the caller's records.
        left_truth = truth_mod.RenderedTruthEvidence(**{
            field.name: getattr(left_truth, field.name) for field in fields(left_truth)})
        right_truth = truth_mod.RenderedTruthEvidence(**{
            field.name: getattr(right_truth, field.name) for field in fields(right_truth)})
        left_signature = signature_mod.SignatureEvidence(**{
            field.name: getattr(left_signature, field.name)
            for field in fields(left_signature)})
        right_signature = signature_mod.SignatureEvidence(**{
            field.name: getattr(right_signature, field.name)
            for field in fields(right_signature)})
        if (correlation_algorithm_id != plane_graph.PLANE_CORRELATION_ALGORITHM
                or type(max_shift_cells) is not int or max_shift_cells < 0
                or type(max_source_pixels) is not int or max_source_pixels <= 0):
            raise ValueError("algorithm, radius, or budget is invalid")
        matrix = np.asarray(homography)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("homography is malformed")
        if (match.left != left_signature.frame_id
                or match.right != right_signature.frame_id
                or match.left_plane != left_signature.canonical_index
                or match.right_plane != right_signature.canonical_index
                or match.left_axis != left_signature.reading_axis
                or match.right_axis != right_signature.reading_axis
                or match.cell_shift_rc is None):
            raise ValueError("match endpoints do not bind signatures")
        match_record = _match_record(match)
        bindings = dict(
            correlation_algorithm_id=correlation_algorithm_id,
            max_shift_cells=max_shift_cells,
            left_truth_sha256=left_truth.truth_sha256,
            right_truth_sha256=right_truth.truth_sha256,
            left_evidence_sha256=left_truth.evidence_sha256,
            right_evidence_sha256=right_truth.evidence_sha256,
            left_signature_sha256=left_signature.signature_sha256,
            right_signature_sha256=right_signature.signature_sha256,
            match_sha256=_sha(match_record),
            homography_sha256=_array_sha(np.ascontiguousarray(matrix)),
            left_ownership_sha256=_array_sha(left_truth.ownership),
            right_ownership_sha256=_array_sha(right_truth.ownership),
            whole_reading_overlap_cells=match.overlap_cells,
            whole_reading_psr="positive_infinity" if match.psr == float("inf")
            else float(match.psr))
        pixels = int(left_truth.rgb.shape[0] * left_truth.rgb.shape[1]
                     + right_truth.rgb.shape[0] * right_truth.rgb.shape[1])
        if pixels > max_source_pixels:
            return _refusal("budget_exhausted", "source pixel cap exceeded", **bindings)
        return _refusal(
            "unavailable_exact_dependencies",
            "PlaneSignature has aggregate cells but no exact rectification/gradient "
            "source-tap witness; renderer ownership lacks resolvable subface planes",
            missing_dependencies=_MISSING, **bindings)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return _refusal("provenance_mismatch", str(error))


@dataclass(frozen=True, slots=True)
class SurfaceSupportRowV1:
    physical_plane_key: tuple[int, int, int, int]
    surface_support_cells: int
    left_endpoint_cells: int
    right_endpoint_cells: int
    left_surface_canvas_cells: int
    right_surface_canvas_cells: int
    surface_share_of_pair_support: float
    surface_overlap_fraction: float | None


@dataclass(frozen=True, slots=True)
class SurfaceLocalSupportReportV1:
    schema: str
    status: str
    sampling_contract: str
    correlation_algorithm_id: str
    bundle_sha256: str
    left_truth_sha256: str
    right_truth_sha256: str
    left_signature_sha256: str
    right_signature_sha256: str
    matcher_envelope_sha256: str
    sampling_witness_sha256: tuple[str, str]
    ownership_witness_sha256: tuple[str, str]
    matcher_witness_sha256: str
    whole_reading_overlap_cells: int
    whole_reading_psr: float | str
    pair_count: int
    same_surface_pairs: int
    different_surface_pairs: int
    mixed_pairs: int
    unknown_pairs: int
    surfaces: tuple[SurfaceSupportRowV1, ...]
    report_sha256: str = ""

    def __post_init__(self):
        if (self.schema != SCHEMA or self.status != "complete"
                or self.pair_count != self.whole_reading_overlap_cells
                or sum((self.same_surface_pairs, self.different_surface_pairs,
                        self.mixed_pairs, self.unknown_pairs)) != self.pair_count):
            raise ValueError("surface-local support report is malformed")
        digest = _sha(self.to_record(include_digest=False))
        if self.report_sha256 not in ("", digest):
            raise ValueError("surface-local support report digest mismatch")
        object.__setattr__(self, "report_sha256", digest)

    def to_record(self, *, include_digest=True):
        result = {field.name: getattr(self, field.name) for field in fields(self)
                  if field.name not in ("surfaces", "report_sha256")}
        result["sampling_witness_sha256"] = list(self.sampling_witness_sha256)
        result["ownership_witness_sha256"] = list(self.ownership_witness_sha256)
        result["surfaces"] = [{name: getattr(row, name) for name in row.__dataclass_fields__}
                              for row in self.surfaces]
        for row in result["surfaces"]:
            row["physical_plane_key"] = list(row["physical_plane_key"])
        if include_digest:
            result["report_sha256"] = self.report_sha256
        return result


@dataclass(frozen=True, slots=True)
class CellOwnershipDiagnosticV1:
    schema: str
    total_cells: int
    selected_cells: int
    full_reason_counts: tuple[tuple[str, int], ...]
    selected_reason_counts: tuple[tuple[str, int], ...]
    examples: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    owner_absence_examples: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = ()
    full_dependency_counts: tuple[tuple[str, int], ...] = ()
    selected_dependency_counts: tuple[tuple[str, int], ...] = ()
    full_background_fraction: float = 0.0
    selected_background_fraction: float = 0.0
    full_unusable_fraction: float = 0.0
    selected_unusable_fraction: float = 0.0
    full_plane_memberships: tuple[tuple[tuple[int, int, int, int], int], ...] = ()
    selected_plane_memberships: tuple[tuple[tuple[int, int, int, int], int], ...] = ()
    coordinate_examples: tuple[tuple[str, tuple[tuple[int, int, int, int], ...]], ...] = ()

    def __post_init__(self):
        reasons=("absent_owner_records","empty_dependencies","out_of_range_indices",
                 "unresolved_physical_plane_identity","unusable_source_pixels")
        if (self.schema != "mc-photogrammetry-cell-ownership-diagnostic-v1"
                or tuple(k for k,_ in self.full_reason_counts) != reasons
                or tuple(k for k,_ in self.selected_reason_counts) != reasons
                or any(type(v) is not int or v < 0 for _,v in
                       self.full_reason_counts+self.selected_reason_counts)
                or tuple(k for k,_ in self.owner_absence_examples)
                   not in ((),("background", "missing_owner_table"))
                or any(not np.isfinite(value) or not 0 <= value <= 1 for value in
                       (self.full_background_fraction,self.selected_background_fraction,
                        self.full_unusable_fraction,self.selected_unusable_fraction))):
            raise ValueError("cell ownership diagnostic is malformed")


def _cell_owners(payload, owner_payload, evidence, *, diagnostic_sink=None,
                 selected_cells=(), max_examples=8,
                 max_diagnostic_bytes=256 * 1024 * 1024):
    if diagnostic_sink is not None and len(diagnostic_sink):
        raise ValueError("ownership diagnostic sink must be empty")
    if (type(max_examples) is not int or max_examples < 0
            or type(max_diagnostic_bytes) is not int or max_diagnostic_bytes < 0):
        raise ValueError("ownership diagnostic example cap is invalid")
    witness = payload.witness
    owner_rows = {row.owner_id: row for row in owner_payload.witness.owners}
    flat_owners = evidence.ownership.reshape(-1)
    flat_usable = evidence.usable.reshape(-1)
    states = []
    reasons=("absent_owner_records","empty_dependencies","out_of_range_indices",
             "unresolved_physical_plane_identity","unusable_source_pixels")
    selected=frozenset(int(v) for v in selected_cells)
    if any(v < 0 or v >= len(witness.cell_rectangles) for v in selected):
        raise ValueError("selected diagnostic cell is out of range")
    counts={key:0 for key in reasons}; selected_counts={key:0 for key in reasons}
    examples={key:[] for key in reasons}
    absence_examples={"background":[],"missing_owner_table":[]}
    dependency_names=("background_source_pixels","border_sentinel_entries",
        "certified_plane_source_pixels","invalid_index_entries",
        "missing_owner_table_source_pixels","unique_source_dependencies",
        "unresolved_plane_source_pixels","unusable_source_pixels")
    pixel_names=tuple(k for k in dependency_names if not k.endswith("entries"))
    audit=diagnostic_sink is not None
    planned_bitmap_bytes=2*len(pixel_names)*len(flat_owners)
    if audit and planned_bitmap_bytes > max_diagnostic_bytes:
        raise MemoryError("ownership diagnostic bitmap cap exceeded")
    full_maps=({key:np.zeros(len(flat_owners),dtype=np.bool_) for key in pixel_names}
               if audit else {})
    selected_maps=({key:np.zeros(len(flat_owners),dtype=np.bool_) for key in pixel_names}
                   if audit else {})
    entry_counts={"border_sentinel_entries":0,"invalid_index_entries":0}
    selected_entry_counts={"border_sentinel_entries":0,"invalid_index_entries":0}
    coordinate={key:[] for key in ("background","border_sentinel","invalid_index",
        "missing_owner_table","unresolved_plane","unusable")}
    width=evidence.ownership.shape[1]
    def mark(maps,key,position):
        if audit: maps[key][position]=True
    for cell in range(len(witness.cell_rectangles)):
        indices = witness.cell_source_indices[
            witness.cell_source_ptr[cell]:witness.cell_source_ptr[cell + 1]]
        causes=set(); offending={key:[] for key in reasons}
        if len(indices) == 0: causes.add("empty_dependencies")
        valid=[]
        for raw in indices:
            position=int(raw)
            if position < 0 or position >= len(flat_owners):
                causes.add("out_of_range_indices")
                offending["out_of_range_indices"].append(position)
                subtype="border_sentinel_entries" if position == -1 else "invalid_index_entries"
                if audit:
                    entry_counts[subtype]+=1
                    if cell in selected: selected_entry_counts[subtype]+=1
                label="border_sentinel" if position == -1 else "invalid_index"
                if len(coordinate[label]) < max_examples:
                    coordinate[label].append((cell,position,-1,-1))
            else: valid.append(position)
        keys = set()
        for position in valid:
            mark(full_maps,"unique_source_dependencies",position)
            if cell in selected: mark(selected_maps,"unique_source_dependencies",position)
            owner_id = int(flat_owners[position])
            row = owner_rows.get(owner_id)
            if not bool(flat_usable[position]):
                causes.add("unusable_source_pixels")
                offending["unusable_source_pixels"].append(position)
                mark(full_maps,"unusable_source_pixels",position)
                if cell in selected: mark(selected_maps,"unusable_source_pixels",position)
                if len(coordinate["unusable"]) < max_examples:
                    coordinate["unusable"].append((cell,position,position//width,position%width))
            if row is None:
                causes.add("absent_owner_records")
                offending["absent_owner_records"].append(position)
                subtype="background" if owner_id < 0 else "missing_owner_table"
                set_name=("background_source_pixels" if owner_id < 0
                          else "missing_owner_table_source_pixels")
                mark(full_maps,set_name,position)
                if cell in selected: mark(selected_maps,set_name,position)
                if len(coordinate[subtype]) < max_examples:
                    coordinate[subtype].append((cell,position,position//width,position%width))
                if len(absence_examples[subtype]) < max_examples:
                    absence_examples[subtype].append((cell,position))
            elif row.physical_plane_key is None:
                causes.add("unresolved_physical_plane_identity")
                offending["unresolved_physical_plane_identity"].append(position)
                mark(full_maps,"unresolved_plane_source_pixels",position)
                if cell in selected: mark(selected_maps,"unresolved_plane_source_pixels",position)
                if len(coordinate["unresolved_plane"]) < max_examples:
                    coordinate["unresolved_plane"].append((cell,position,position//width,position%width))
            else:
                keys.add(row.physical_plane_key)
                mark(full_maps,"certified_plane_source_pixels",position)
                if cell in selected:
                    mark(selected_maps,"certified_plane_source_pixels",position)
        for reason in sorted(causes):
            counts[reason]+=1
            if cell in selected: selected_counts[reason]+=1
            if len(examples[reason]) < max_examples:
                source=(-1 if reason=="empty_dependencies" else
                        offending[reason][0])
                examples[reason].append((cell,source))
        if causes:
            states.append(("unknown", None))
        elif len(keys) == 1:
            states.append(("pure", (evidence.truth_sha256,) + next(iter(keys))))
        else:
            states.append(("mixed", None))
    if diagnostic_sink is not None:
        def count(maps,name):
            return entry_counts[name] if name in entry_counts else int(np.count_nonzero(maps[name]))
        def selected_count(maps,name):
            return selected_entry_counts[name] if name in selected_entry_counts else int(np.count_nonzero(maps[name]))
        def fraction(maps,name):
            denominator=int(np.count_nonzero(maps["unique_source_dependencies"]))
            return 0.0 if denominator==0 else int(np.count_nonzero(maps[name]))/denominator
        def planes(maps):
            result={}
            for position in np.flatnonzero(maps["certified_plane_source_pixels"]):
                row=owner_rows[int(flat_owners[position])]
                result[row.physical_plane_key]=result.get(row.physical_plane_key,0)+1
            return tuple(sorted(result.items()))
        diagnostic_sink.append(CellOwnershipDiagnosticV1(
            "mc-photogrammetry-cell-ownership-diagnostic-v1",len(states),len(selected),
            tuple((k,counts[k]) for k in reasons),
            tuple((k,selected_counts[k]) for k in reasons),
            tuple((k,tuple(examples[k])) for k in reasons),
            tuple((k,tuple(absence_examples[k]))
                  for k in ("background","missing_owner_table")),
            tuple((k,count(full_maps,k)) for k in dependency_names),
            tuple((k,selected_count(selected_maps,k)) for k in dependency_names),
            fraction(full_maps,"background_source_pixels"),
            fraction(selected_maps,"background_source_pixels"),
            fraction(full_maps,"unusable_source_pixels"),
            fraction(selected_maps,"unusable_source_pixels"),
            planes(full_maps), planes(selected_maps),
            tuple((k,tuple(coordinate[k])) for k in sorted(coordinate))))
    return tuple(states)


def evaluate_surface_support_from_payload(
    *, payload: preflight_mod.SyntheticPreflightWitnessPayloadV1,
    left_truth: truth_mod.RenderedTruthEvidence,
    right_truth: truth_mod.RenderedTruthEvidence,
    left_signature: signature_mod.SignatureEvidence,
    right_signature: signature_mod.SignatureEvidence,
    max_pair_cells: int = DEFAULT_MAX_PAIR_CELLS,
) -> SurfaceLocalSupportReportV1 | SurfaceLocalSupportRefusal:
    """Attribute exact selected matcher pairs without whole-reading binding."""
    bindings = {}
    try:
        if (not isinstance(payload, preflight_mod.SyntheticPreflightWitnessPayloadV1)
                or type(max_pair_cells) is not int or max_pair_cells < 0):
            raise ValueError("payload or pair budget is invalid")
        # Reconstructing the carrier revalidates nested hashes and ordering.
        payload = preflight_mod.SyntheticPreflightWitnessPayloadV1(
            payload.public_bundle, payload.ownership, payload.signatures,
            payload.matches)
        left_truth = truth_mod.RenderedTruthEvidence(**{
            name: getattr(left_truth, name) for name in left_truth.__dataclass_fields__})
        right_truth = truth_mod.RenderedTruthEvidence(**{
            name: getattr(right_truth, name) for name in right_truth.__dataclass_fields__})
        left_signature = signature_mod.SignatureEvidence(**{
            name: getattr(left_signature, name)
            for name in left_signature.__dataclass_fields__})
        right_signature = signature_mod.SignatureEvidence(**{
            name: getattr(right_signature, name)
            for name in right_signature.__dataclass_fields__})
        truth_by_hash = {row.evidence_sha256: row for row in (left_truth, right_truth)}
        owner_by_hash = {row.binding.evidence_sha256: row for row in payload.ownership}
        signature_by_hash = {row.binding.signature_sha256: row for row in payload.signatures}
        if len(truth_by_hash) != 2 or len(owner_by_hash) != len(payload.ownership):
            raise ValueError("truth or ownership payload is duplicated")
        left_sample = signature_by_hash.get(left_signature.signature_sha256)
        right_sample = signature_by_hash.get(right_signature.signature_sha256)
        if left_sample is None or right_sample is None:
            raise ValueError("signature payload join is missing")
        if (left_sample.binding.truth_evidence_sha256 != left_truth.evidence_sha256
                or right_sample.binding.truth_evidence_sha256 != right_truth.evidence_sha256):
            raise ValueError("signature sampling payload does not bind truth evidence")
        matches = [row for row in payload.matches
                   if row.envelope.left_signature_sha256 == left_signature.signature_sha256
                   and row.envelope.right_signature_sha256 == right_signature.signature_sha256]
        if len(matches) != 1:
            raise ValueError("selected matcher payload join is missing or ambiguous")
        match_payload = matches[0]
        left_owner = owner_by_hash.get(left_truth.evidence_sha256)
        right_owner = owner_by_hash.get(right_truth.evidence_sha256)
        if left_owner is None or right_owner is None:
            raise ValueError("renderer ownership payload join is missing")
        if (truth_mod.bind_renderer_ownership_witness(left_truth, left_owner.witness)
                != left_owner.binding
                or truth_mod.bind_renderer_ownership_witness(right_truth, right_owner.witness)
                != right_owner.binding):
            raise ValueError("renderer ownership payload does not replay")
        pairs = match_payload.witness.logical_pairs
        warp = match_payload.witness.warp_witness
        if warp is None:
            raise ValueError("matcher warp payload is unavailable")
        if (warp.source_shape != left_signature.signature.cells.shape
                or warp.target_shape != right_signature.signature.cells.shape
                or not np.array_equal(warp.homography,
                                      match_payload.witness.homography)
                or match_payload.witness.canvas_shape != (
                    warp.target_shape[0] + 4 * match_payload.witness.max_shift_cells,
                    warp.target_shape[1] + 4 * match_payload.witness.max_shift_cells)):
            raise ValueError("matcher warp dimensions or homography mismatch")
        if len(pairs) > max_pair_cells:
            bindings.update(whole_reading_overlap_cells=match_payload.envelope.overlap_cells,
                            whole_reading_psr=match_payload.envelope.psr)
            return _refusal("budget_exhausted", "pair cell cap exceeded", **bindings)
        if (len(pairs) != match_payload.witness.selected_support
                or len(pairs) != match_payload.envelope.overlap_cells):
            raise ValueError("matcher pair conservation failed")
        left_states = _cell_owners(left_sample, left_owner, left_truth)
        right_states = _cell_owners(right_sample, right_owner, right_truth)
        left_width = left_signature.signature.cells.shape[1]
        right_width = right_signature.signature.cells.shape[1]
        if (len(left_states) != left_signature.signature.cells.size
                or len(right_states) != right_signature.signature.cells.size):
            raise ValueError("sampling witness cell universe differs from signature")
        counts = {"same": 0, "different": 0, "mixed": 0, "unknown": 0}
        support = {}; left_endpoints = {}; right_endpoints = {}
        for lr, lc, rr, rc in pairs:
            if (int(lr) < 0 or int(lr) >= left_signature.signature.cells.shape[0]
                    or int(lc) < 0 or int(lc) >= left_width
                    or int(rr) < 0 or int(rr) >= right_signature.signature.cells.shape[0]
                    or int(rc) < 0 or int(rc) >= right_width):
                raise ValueError("matcher pair indexes outside signature dimensions")
            li = int(lr) * left_width + int(lc)
            ri = int(rr) * right_width + int(rc)
            if li < 0 or ri < 0 or li >= len(left_states) or ri >= len(right_states):
                raise ValueError("matcher pair indexes outside signature cells")
            ls, lk = left_states[li]; rs, rk = right_states[ri]
            if "unknown" in (ls, rs): category = "unknown"
            elif "mixed" in (ls, rs): category = "mixed"
            elif lk != rk: category = "different"
            else:
                category = "same"
                support[lk] = support.get(lk, 0) + 1
                left_endpoints.setdefault(lk, set()).add((int(lr), int(lc)))
                right_endpoints.setdefault(lk, set()).add((int(rr), int(rc)))
            counts[category] += 1
        total = len(pairs)
        surfaces = []
        target_rows, target_columns = warp.target_shape
        left_canvas = []
        for row in range(target_rows):
            for column in range(target_columns):
                if not bool(warp.geometric_valid[row, column]):
                    left_canvas.append(("unknown", None)); continue
                sr = int(warp.source_rows[row, column])
                sc = int(warp.source_columns[row, column])
                index = sr * left_width + sc
                left_canvas.append(left_states[index] if 0 <= index < len(left_states)
                                   and bool(left_signature.signature.coverage[sr, sc])
                                   else ("unknown", None))
        for key in sorted(support):
            right_covered = right_signature.signature.coverage.reshape(-1)
            a = sum(state == "pure" and plane == key
                    for state, plane in left_canvas)
            b = sum(bool(covered) and state == "pure" and plane == key
                    for covered, (state, plane) in zip(right_covered, right_states))
            denominator = min(a, b)
            surfaces.append(SurfaceSupportRowV1(
                key[1:], support[key], len(left_endpoints[key]),
                len(right_endpoints[key]), a, b,
                support[key] / total if total else 0.0,
                None if denominator == 0 else support[key] / denominator))
        return SurfaceLocalSupportReportV1(
            SCHEMA, "complete", SAMPLING_CONTRACT,
            plane_graph.PLANE_CORRELATION_ALGORITHM,
            payload.public_bundle.bundle_sha256, left_truth.truth_sha256,
            right_truth.truth_sha256, left_signature.signature_sha256,
            right_signature.signature_sha256, match_payload.envelope.envelope_sha256,
            (left_sample.witness.witness_sha256, right_sample.witness.witness_sha256),
            (left_owner.witness.witness_sha256, right_owner.witness.witness_sha256),
            match_payload.witness.witness_sha256,
            match_payload.envelope.overlap_cells, match_payload.envelope.psr,
            total, counts["same"], counts["different"], counts["mixed"],
            counts["unknown"], tuple(surfaces))
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return _refusal("provenance_mismatch", str(error), **bindings)


__all__ = ["SCHEMA", "SAMPLING_CONTRACT", "SurfaceLocalSupportRefusal",
           "SurfaceLocalSupportReportV1", "SurfaceSupportRowV1",
           "evaluate_surface_support", "evaluate_surface_support_from_payload"]
