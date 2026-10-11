"""Evaluation-only synthetic preflight from immutable rendered truth evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

from PIL import Image
import numpy as np

from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry import evaluation_reading_region_mask as region_mod
from mc_photogrammetry import evaluation_signature_sampling as sampling_mod
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
from mc_photogrammetry.pipeline import image_derived_graph_attribution as trace_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import plane_graph, plane_placement

SCHEMA = "mc-photogrammetry-synthetic-targeted-preflight-v2"
MAX_SYNTHETIC_PNG_BYTES = 32 * 1024 * 1024
MAX_WITNESS_PAYLOAD_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SyntheticPreflightEvidenceBundleV1:
    schema: str
    frame_ids: tuple[str, ...]
    truth_evidence_sha256: tuple[str, ...]
    ownership_binding_sha256: tuple[str, ...]
    signature_binding_sha256: tuple[str, ...]
    matcher_envelope_sha256: tuple[str, ...]
    trace_sha256: str
    bundle_sha256: str = ""

    def __post_init__(self):
        if (self.schema != "mc-photogrammetry-synthetic-preflight-evidence-bundle-v1"
                or len(self.frame_ids) != len(self.truth_evidence_sha256)
                or len(self.frame_ids) != len(self.ownership_binding_sha256)):
            raise ValueError("synthetic preflight evidence bundle is malformed")
        digest = _sha(self.to_record(include_digest=False))
        if self.bundle_sha256 not in ("", digest):
            raise ValueError("synthetic preflight evidence bundle digest mismatch")
        object.__setattr__(self, "bundle_sha256", digest)

    def to_record(self, *, include_digest=True):
        result = {"schema": self.schema, "frame_ids": list(self.frame_ids),
                  "truth_evidence_sha256": list(self.truth_evidence_sha256),
                  "ownership_binding_sha256": list(self.ownership_binding_sha256),
                  "signature_binding_sha256": list(self.signature_binding_sha256),
                  "matcher_envelope_sha256": list(self.matcher_envelope_sha256),
                  "trace_sha256": self.trace_sha256}
        if include_digest:
            result["bundle_sha256"] = self.bundle_sha256
        return result


@dataclass(frozen=True, slots=True)
class SyntheticPreflightEvidenceRefusalV1:
    status: str
    reason: str
    stage: str
    detail: str


@dataclass(frozen=True, slots=True)
class SignatureReplayDiagnosticRowV1:
    """Evaluation-only identity plus the exact result of one signature replay."""
    frame_id: str
    canonical_index: int
    alternative_index: int
    signature_sha256: str
    result: (sampling_mod.SignatureSamplingBindingV1
             | sampling_mod.SignatureSamplingReplayRefusalV1)

    def __post_init__(self):
        if (type(self.frame_id) is not str or not self.frame_id
                or type(self.canonical_index) is not int or self.canonical_index < 0
                or type(self.alternative_index) is not int or self.alternative_index < 0
                or type(self.signature_sha256) is not str
                or not isinstance(self.result,
                                  (sampling_mod.SignatureSamplingBindingV1,
                                   sampling_mod.SignatureSamplingReplayRefusalV1))
                or self.result.signature_sha256 != self.signature_sha256):
            raise ValueError("signature replay diagnostic row is malformed")
        if (isinstance(self.result, sampling_mod.SignatureSamplingBindingV1)
                and (self.result.frame_id != self.frame_id
                     or self.result.canonical_reading_index != self.canonical_index
                     or self.result.alternative_index != self.alternative_index)):
            raise ValueError("signature replay diagnostic identity mismatch")


@dataclass(frozen=True, slots=True)
class SyntheticPreflightWitnessPayloadV1:
    """Private in-memory witness payload; never part of the preflight report."""
    public_bundle: SyntheticPreflightEvidenceBundleV1
    ownership: tuple[truth_mod.RenderedTruthOwnershipWitnessPayloadV1, ...]
    signatures: tuple[sampling_mod.SignatureSamplingWitnessPayloadV1, ...]
    matches: tuple[plane_graph.GraphMatcherWitnessPayloadV1, ...]

    def __post_init__(self):
        if not isinstance(self.public_bundle, SyntheticPreflightEvidenceBundleV1):
            raise ValueError("preflight witness payload has no public binding")
        ownership = tuple(truth_mod.RenderedTruthOwnershipWitnessPayloadV1(
            row.binding, row.witness) for row in self.ownership)
        signatures = tuple(sampling_mod.SignatureSamplingWitnessPayloadV1(
            row.binding, row.witness) for row in self.signatures)
        matches = tuple(plane_graph.GraphMatcherWitnessPayloadV1(
            row.envelope, row.witness) for row in self.matches)
        object.__setattr__(self, "ownership", ownership)
        object.__setattr__(self, "signatures", signatures)
        object.__setattr__(self, "matches", matches)
        if (tuple(row.binding.binding_sha256 for row in self.ownership)
                != self.public_bundle.ownership_binding_sha256
                or tuple(row.binding.report_sha256 for row in self.signatures)
                != self.public_bundle.signature_binding_sha256
                or tuple(row.envelope.envelope_sha256 for row in self.matches)
                != self.public_bundle.matcher_envelope_sha256):
            raise ValueError("preflight witness payload order or digest mismatch")
        signature_hashes = {row.binding.signature_sha256 for row in self.signatures}
        if any(row.envelope.left_signature_sha256 not in signature_hashes
               or row.envelope.right_signature_sha256 not in signature_hashes
               for row in self.matches):
            raise ValueError("matcher endpoints do not join signature payloads")


def _payload_nbytes(value, seen=None):
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        if value.flags.writeable:
            raise ValueError("witness payload contains mutable array")
        return value.nbytes
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if value is None or isinstance(value, (bool, int, float)):
        return len(json.dumps(value, allow_nan=False).encode("ascii"))
    if hasattr(value, "__dataclass_fields__"):
        return sum(_payload_nbytes(getattr(value, name), seen)
                   for name in value.__dataclass_fields__)
    if isinstance(value, (tuple, list)):
        return sum(_payload_nbytes(row, seen) for row in value)
    return 0


def _assemble_witness_payload(public_bundle, evidences, ownership_payloads,
                              signature_payloads, matcher_payloads, max_bytes):
    try:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("witness payload cap is invalid")
        if (len(ownership_payloads) != len(evidences)
                or any(not isinstance(row, truth_mod.RenderedTruthOwnershipWitnessPayloadV1)
                       for row in ownership_payloads)):
            raise ValueError("renderer witness payloads are missing or reordered")
        for evidence, row in zip(evidences, ownership_payloads):
            rebound = truth_mod.bind_renderer_ownership_witness(evidence, row.witness)
            if rebound != row.binding:
                raise ValueError("renderer witness payload does not bind evidence")
        if any(not isinstance(row, sampling_mod.SignatureSamplingWitnessPayloadV1)
               for row in signature_payloads):
            raise ValueError("signature witness payload is missing")
        if any(not isinstance(row, plane_graph.GraphMatcherWitnessPayloadV1)
               for row in matcher_payloads):
            raise ValueError("matcher witness payload is missing")
        result = SyntheticPreflightWitnessPayloadV1(
            public_bundle, tuple(ownership_payloads), tuple(signature_payloads),
            tuple(matcher_payloads))
        planned = _payload_nbytes(result)
        if planned > max_bytes:
            return SyntheticPreflightEvidenceRefusalV1(
                "refused", "budget_exhausted", "witness_payload",
                f"limit={max_bytes};planned={planned};actual=0")
        return result
    except (AttributeError, TypeError, ValueError) as error:
        return SyntheticPreflightEvidenceRefusalV1(
            "refused", "provenance_mismatch", "witness_payload", str(error))


def _evaluate_surface_support_payload(payload, evidences, signatures, max_pairs):
    from mc_photogrammetry import evaluation_surface_support as support_mod
    try:
        if not isinstance(payload, SyntheticPreflightWitnessPayloadV1):
            raise ValueError("complete private witness payload is required")
        evidence_by_hash = {row.evidence_sha256: row for row in evidences}
        signature_by_hash = {row.signature_sha256: row for row in signatures}
        binding_by_signature = {row.binding.signature_sha256: row.binding
                                for row in payload.signatures}
        if len(signature_by_hash) != len(signatures):
            raise ValueError("signature hashes are duplicated")
        pending = []
        remaining = max_pairs
        for matched in payload.matches:
            planned = len(matched.witness.logical_pairs)
            if planned > remaining:
                return support_mod._refusal(
                    "budget_exhausted", "aggregate pair cell cap exceeded",
                    whole_reading_overlap_cells=matched.envelope.overlap_cells,
                    whole_reading_psr=matched.envelope.psr)
            left = signature_by_hash.get(matched.envelope.left_signature_sha256)
            right = signature_by_hash.get(matched.envelope.right_signature_sha256)
            left_binding = binding_by_signature.get(matched.envelope.left_signature_sha256)
            right_binding = binding_by_signature.get(matched.envelope.right_signature_sha256)
            left_truth = None if left_binding is None else evidence_by_hash.get(
                left_binding.truth_evidence_sha256)
            right_truth = None if right_binding is None else evidence_by_hash.get(
                right_binding.truth_evidence_sha256)
            if any(value is None for value in (left, right, left_truth, right_truth)):
                raise ValueError("matcher endpoint signature crosswalk is missing")
            result = support_mod.evaluate_surface_support_from_payload(
                payload=payload, left_truth=left_truth, right_truth=right_truth,
                left_signature=left, right_signature=right,
                max_pair_cells=remaining)
            if isinstance(result, support_mod.SurfaceLocalSupportRefusal):
                return result
            pending.append(result)
            remaining -= planned
        if not pending:
            raise ValueError("no selected matcher envelopes are available")
        return tuple(pending)
    except (AttributeError, TypeError, ValueError) as error:
        return SyntheticPreflightEvidenceRefusalV1(
            "refused", "provenance_mismatch", "surface_support", str(error))


def _assemble_evidence_bundle(evidences, frame_ids, ownership_bindings,
                              signature_bindings, matcher_envelopes, trace):
    try:
        if (len(ownership_bindings) != len(frame_ids)
                or any(not isinstance(row, truth_mod.RenderedTruthOwnershipBindingV1)
                       for row in ownership_bindings)
                or any(row.evidence_sha256 != evidence.evidence_sha256
                       for row, evidence in zip(ownership_bindings, evidences))):
            raise ValueError("ownership bindings are missing or reordered")
        if any(not isinstance(row, sampling_mod.SignatureSamplingBindingV1)
               for row in signature_bindings):
            raise ValueError("signature replay contains a refusal")
        selected = [row for row in trace.pair_attribution if row.selected]
        if len(selected) != len(matcher_envelopes):
            raise ValueError("selected trace and matcher witness counts differ")
        for traced, envelope in zip(selected, matcher_envelopes):
            if not isinstance(envelope, plane_graph.GraphMatcherWitnessEnvelopeV1):
                raise ValueError("matcher witness contains a refusal")
            actual = (envelope.left, envelope.right, envelope.left_axis,
                      envelope.right_axis, envelope.left_plane,
                      envelope.right_plane, envelope.quadrant, envelope.shift_rc,
                      envelope.overlap_cells, envelope.psr, envelope.score)
            expected = (traced.left, traced.right, traced.left_axis,
                        traced.right_axis, traced.left_plane, traced.right_plane,
                        traced.quadrant, traced.cell_shift_rc, traced.overlap_cells,
                        "positive_infinity" if traced.psr == float("inf") else traced.psr,
                        traced.score)
            if actual != expected:
                raise ValueError("selected trace does not bind matcher witness")
        return SyntheticPreflightEvidenceBundleV1(
            "mc-photogrammetry-synthetic-preflight-evidence-bundle-v1",
            tuple(frame_ids), tuple(row.evidence_sha256 for row in evidences),
            tuple(row.binding_sha256 for row in ownership_bindings),
            tuple(row.report_sha256 for row in signature_bindings),
            tuple(row.envelope_sha256 for row in matcher_envelopes),
            trace.attribution_sha256)
    except (AttributeError, TypeError, ValueError) as error:
        return SyntheticPreflightEvidenceRefusalV1(
            "refused", "provenance_mismatch", "evidence_bundle", str(error))


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PredictedPair:
    left: str
    right: str
    surface_id: str
    left_reading_index: int
    right_reading_index: int
    support_cells: int
    support_fraction: float
    displacement_rc: tuple[int, int]
    structural_share: float
    psr: float | str


@dataclass(frozen=True, slots=True)
class SurfaceIntersection:
    surface_id: str
    intersection_pixels: int
    surface_pixels: int
    reading_purity: float
    truth_coverage: float


@dataclass(frozen=True, slots=True)
class ReadingBindingDiagnostic:
    frame_id: str
    reading_index: int
    reading_axis: int
    decoded_mask_pixels: int
    intersections: tuple[SurfaceIntersection, ...]
    exact_candidates: tuple[str, ...]
    rejection_reason: str


@dataclass(frozen=True, slots=True)
class SyntheticTargetedPreflightReport:
    schema: str
    status: str
    reason: str
    detail: str
    frame_ids: tuple[str, ...]
    evidence_sha256: tuple[str, ...]
    camera_sha256: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str, str], ...]
    selected_surface_id: str
    entry_sha256: str
    lattice_sha256: str
    signatures_sha256: str
    graph_sha256: str
    trace_sha256: str
    solver_reason: str
    reading_bindings: tuple[ReadingBindingDiagnostic, ...]
    bound_surface_sets: tuple[tuple[str, tuple[str, ...]], ...]
    pairs: tuple[PredictedPair, ...]
    report_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.status not in (
                "predicted", "unresolved_surface", "insufficient_evidence"):
            raise ValueError("synthetic preflight report is malformed")
        digest = _sha(self.to_record(include_digest=False))
        if self.report_sha256 not in ("", digest):
            raise ValueError("synthetic preflight report digest mismatch")
        object.__setattr__(self, "report_sha256", digest)

    def to_record(self, *, include_digest: bool = True) -> dict[str, Any]:
        value = {"schema": self.schema, "status": self.status,
                 "reason": self.reason, "detail": self.detail,
                 "frame_ids": list(self.frame_ids),
                 "evidence_sha256": list(self.evidence_sha256),
                 "camera_sha256": list(self.camera_sha256),
                 "artifact_hashes": [list(row) for row in self.artifact_hashes],
                 "selected_surface_id": self.selected_surface_id,
                 "entry_sha256": self.entry_sha256,
                 "lattice_sha256": self.lattice_sha256,
                 "signatures_sha256": self.signatures_sha256,
                 "graph_sha256": self.graph_sha256,
                 "trace_sha256": self.trace_sha256,
                 "solver_reason": self.solver_reason,
                 "reading_bindings": [asdict(row) for row in self.reading_bindings],
                 "bound_surface_sets": [[frame, list(surfaces)]
                                        for frame, surfaces in self.bound_surface_sets],
                 "pairs": [asdict(row) for row in self.pairs]}
        if include_digest:
            value["report_sha256"] = self.report_sha256
        return value


def _base(evidences, frame_ids, *, status, reason, detail="", **kwargs):
    return SyntheticTargetedPreflightReport(
        SCHEMA, status, reason, detail, frame_ids,
        tuple(item.evidence_sha256 for item in evidences),
        tuple(_sha(item.camera.to_dict()) for item in evidences),
        tuple((item.truth_sha256, item.vanilla_jar_sha256, item.palette_sha256)
              for item in evidences), kwargs.get("selected_surface_id", ""),
        kwargs.get("entry_sha256", ""), kwargs.get("lattice_sha256", ""),
        kwargs.get("signatures_sha256", ""), kwargs.get("graph_sha256", ""),
        kwargs.get("trace_sha256", ""), kwargs.get("solver_reason", ""),
        kwargs.get("reading_bindings", ()), kwargs.get("bound_surface_sets", ()),
        kwargs.get("pairs", ()))


def _surface_binding_diagnostics(evidences, frame_ids, readings):
    by_frame = {frame: (index, evidence) for index, (frame, evidence)
                in enumerate(zip(frame_ids, evidences))}
    bindings = {}
    diagnostics = []
    per_frame_index = {}
    for reading in readings:
        canonical_index = per_frame_index.get(reading.frame, 0)
        per_frame_index[reading.frame] = canonical_index + 1
        if reading.frame not in by_frame:
            continue
        _position, evidence = by_frame[reading.frame]
        if reading.region_evidence is None:
            intersections = tuple(SurfaceIntersection(
                f"east:{surface.coordinate:g}", 0,
                int(np.count_nonzero(surface.mask)), 0.0, 0.0)
                for surface in sorted(evidence.surfaces,
                                      key=lambda row: (row.direction, row.coordinate))
                if surface.direction == "east")
            diagnostics.append(ReadingBindingDiagnostic(
                reading.frame, canonical_index, reading.axis, 0,
                intersections, (), "zero"))
            continue
        decoded = region_mod.reading_region_mask(
            reading, canonical_reading_index=canonical_index,
            expected_frame_id=reading.frame, usable_mask=evidence.usable)
        mask = decoded.mask
        mask_pixels = int(np.count_nonzero(mask))
        candidates = []
        intersections = []
        for surface in sorted(evidence.surfaces,
                              key=lambda row: (row.direction, row.coordinate)):
            if surface.direction != "east":
                continue
            surface_pixels = int(np.count_nonzero(surface.mask))
            intersection = int(np.count_nonzero(mask & surface.mask))
            surface_id = f"east:{surface.coordinate:g}"
            coverage = intersection / surface_pixels if surface_pixels else 0.0
            intersections.append(SurfaceIntersection(
                surface_id, intersection, surface_pixels,
                intersection / mask_pixels if mask_pixels else 0.0,
                coverage))
            # Binding remains deliberately strict: all decoded reading pixels
            # must belong to exactly one truth surface.  Reverse containment is
            # diagnostic only because a tiny surface inside a broad reading
            # cannot justify attributing whole-reading match metrics to it.
            if intersection and intersection == mask_pixels:
                candidates.append(surface_id)
        candidates = sorted(candidates)
        if len(candidates) == 1:
            bindings[(reading.frame, canonical_index)] = candidates[0]
            rejection = ""
        elif len(candidates) > 1:
            rejection = "multiple_exact"
        elif any(row.intersection_pixels for row in intersections):
            rejection = "partial"
        else:
            rejection = "zero"
        diagnostics.append(ReadingBindingDiagnostic(
            reading.frame, canonical_index, reading.axis, mask_pixels,
            tuple(intersections), tuple(candidates), rejection))
    diagnostics.sort(key=lambda row: (row.frame_id, row.reading_index))
    return bindings, tuple(diagnostics)


def _surface_bindings(evidences, frame_ids, readings):
    return _surface_binding_diagnostics(evidences, frame_ids, readings)[0]


def _collect_signature_replays(items, evidence_by_frame, witness_budget,
                               *, collect_payload):
    """Build a complete replay batch before any observer can see it."""
    bindings, payloads, diagnostics = [], [], []
    for item in items:
        item_payload = []
        replay = sampling_mod.replay_signature_sampling_witness(
            signature_evidence=item,
            truth_evidence=evidence_by_frame[item.frame_id],
            witness_budget=witness_budget,
            witness_payload_sink=(item_payload if collect_payload else None))
        bindings.append(replay)
        payloads.extend(item_payload)
        diagnostics.append(SignatureReplayDiagnosticRowV1(
            item.frame_id, item.canonical_index, item.alternative_index,
            item.signature_sha256, replay))
    return bindings, payloads, diagnostics


def preflight_synthetic_targeted_capture(
    *, evidences: Sequence[truth_mod.RenderedTruthEvidence],
    frame_ids: Sequence[str], project_path: str | Path,
    max_png_bytes: int = MAX_SYNTHETIC_PNG_BYTES,
    ownership_bindings: Sequence[truth_mod.RenderedTruthOwnershipBindingV1] | None = None,
    evidence_bundle_sink=None,
    ownership_witness_payloads: Sequence[truth_mod.RenderedTruthOwnershipWitnessPayloadV1] | None = None,
    evidence_payload_sink=None,
    surface_support_sink=None,
    signature_replay_diagnostic_sink=None,
    max_witness_payload_bytes: int = MAX_WITNESS_PAYLOAD_BYTES,
    max_surface_support_pairs: int = 10_000_000,
    witness_budget=None,
) -> SyntheticTargetedPreflightReport:
    if (signature_replay_diagnostic_sink is not None
            and (not hasattr(signature_replay_diagnostic_sink, "extend")
                 or len(signature_replay_diagnostic_sink))):
        raise ValueError("signature replay diagnostic sink must be empty")
    if evidence_payload_sink is not None and len(evidence_payload_sink):
        raise ValueError("preflight witness payload sink must be empty")
    if surface_support_sink is not None and len(surface_support_sink):
        raise ValueError("surface support sink must be empty")
    if (not isinstance(evidences, Sequence) or len(evidences) != 4
            or any(not isinstance(value, truth_mod.RenderedTruthEvidence)
                   for value in evidences)):
        raise ValueError("preflight requires four rendered truth evidence records")
    evidences = tuple(truth_mod.RenderedTruthEvidence(**{
        name: getattr(value, name) for name in value.__dataclass_fields__})
        for value in evidences)
    frame_ids = tuple(frame_ids)
    if (len(frame_ids) != 4 or len(set(frame_ids)) != 4
            or any(not isinstance(value, str) or not value for value in frame_ids)
            or type(max_png_bytes) is not int or max_png_bytes <= 0):
        raise ValueError("preflight frame IDs or PNG cap are invalid")
    wants_payload = evidence_payload_sink is not None or surface_support_sink is not None
    wants_bundle = evidence_bundle_sink is not None or wants_payload
    wants_signature_diagnostics = signature_replay_diagnostic_sink is not None
    private_payload_output = evidence_payload_sink if evidence_payload_sink is not None else []
    def refuse_payload(stage, detail):
        refusal = SyntheticPreflightEvidenceRefusalV1(
            "refused", "upstream_refusal", stage, detail)
        for target in (evidence_bundle_sink,
                       private_payload_output if wants_payload else None,
                       evidence_payload_sink):
            if target is not None and not target:
                target.append(refusal)
        if surface_support_sink is not None and not surface_support_sink:
            surface_support_sink.append(refusal)
    supplied_payloads = tuple(ownership_witness_payloads or ())
    supplied = tuple(ownership_bindings or ())
    if wants_payload:
        if ownership_bindings is None:
            if all(isinstance(row, truth_mod.RenderedTruthOwnershipWitnessPayloadV1)
                   for row in supplied_payloads):
                supplied = tuple(row.binding for row in supplied_payloads)
            else:
                private_payload_output.append(SyntheticPreflightEvidenceRefusalV1(
                    "refused", "provenance_mismatch", "ownership_payloads",
                    "renderer payloads contain a foreign value"))
        elif (not all(isinstance(row, truth_mod.RenderedTruthOwnershipWitnessPayloadV1)
                      for row in supplied_payloads)
              or tuple(row.binding for row in supplied_payloads) != supplied):
            private_payload_output.append(SyntheticPreflightEvidenceRefusalV1(
                "refused", "provenance_mismatch", "ownership_payloads",
                "renderer payloads do not bind supplied ownership bindings"))
    if wants_bundle:
        if (len(supplied) != len(evidences)
                or any(not isinstance(row, truth_mod.RenderedTruthOwnershipBindingV1)
                       for row in supplied)
                or any(row.evidence_sha256 != evidence.evidence_sha256
                       for row, evidence in zip(supplied, evidences))):
            target = private_payload_output if wants_payload else evidence_bundle_sink
            if target is not None and not target:
                target.append(SyntheticPreflightEvidenceRefusalV1(
                "refused", "provenance_mismatch", "ownership_bindings",
                "one exact renderer ownership binding is required per frame"))
    entry = lattice = signatures = graph = trace = None
    with tempfile.TemporaryDirectory(prefix="mc-photo-truth-preflight-") as temporary:
        paths = []
        total = 0
        for frame, evidence in zip(frame_ids, evidences):
            path = Path(temporary) / f"{frame}.png"
            Image.fromarray(evidence.rgb, "RGB").save(path, format="PNG")
            total += path.stat().st_size
            if total > max_png_bytes:
                refuse_payload("png", "png byte cap exceeded")
                return _base(evidences, frame_ids, status="insufficient_evidence",
                             reason="png_byte_cap")
            paths.append(path)
        entry = entry_mod.build_image_derived_entry(paths, project_path=project_path)
        if entry.graph.status != "ok" or entry.pose_graph.status != "ok":
            refuse_payload("entry", entry.reason or "entry_unavailable")
            return _base(evidences, frame_ids, status="insufficient_evidence",
                         reason=entry.reason or "entry_unavailable",
                         entry_sha256=entry.entry_sha256)
        lattice = lattice_mod.measure_image_derived_lattice(
            session=entry.session, pose_graph=entry.pose_graph)
        if isinstance(lattice, plane_placement.PlacementRefusal) or lattice.status != "assembled":
            refuse_payload("lattice", getattr(lattice, "reason", "lattice_refused"))
            return _base(evidences, frame_ids, status="insufficient_evidence",
                         reason=getattr(lattice, "reason", "lattice_refused"),
                         entry_sha256=entry.entry_sha256,
                         lattice_sha256=getattr(lattice, "result_sha256", ""))
        signatures = signatures_mod.derive_image_derived_signatures(
            session=entry.session, lattice_result=lattice)
        if (isinstance(signatures, plane_placement.PlacementRefusal)
                or signatures.status != "assembled"):
            refuse_payload("signatures", getattr(signatures, "reason", "signatures_refused"))
            return _base(evidences, frame_ids, status="insufficient_evidence",
                         reason=getattr(signatures, "reason", "signatures_refused"),
                         entry_sha256=entry.entry_sha256,
                         lattice_sha256=lattice.result_sha256,
                         signatures_sha256=getattr(signatures, "result_sha256", ""))
        signature_bindings = []
        signature_payloads = []
        if wants_bundle or wants_signature_diagnostics:
            evidence_by_frame = dict(zip(frame_ids, evidences))
            signature_bindings, signature_payloads, diagnostic_rows = (
                _collect_signature_replays(
                    signatures.signatures, evidence_by_frame, witness_budget,
                    collect_payload=wants_bundle))
            if signature_replay_diagnostic_sink is not None:
                signature_replay_diagnostic_sink.extend(diagnostic_rows)
            if (wants_bundle
                    and any(not isinstance(row, sampling_mod.SignatureSamplingBindingV1)
                            for row in signature_bindings)):
                target = private_payload_output if wants_payload else evidence_bundle_sink
                if target is not None and not target:
                    target.append(SyntheticPreflightEvidenceRefusalV1(
                    "refused", "upstream_refusal", "signature_replay",
                    "one or more signature witnesses refused"))
        matcher_envelopes = []
        matcher_payloads = []
        graph, trace = trace_mod.derive_image_derived_graph_attribution(
            session=entry.session, lattice_result=lattice,
            signatures_result=signatures, connection_graph=entry.graph,
            matcher_witness_sink=(matcher_envelopes if wants_bundle else None),
            matcher_witness_payload_sink=(matcher_payloads if wants_payload else None),
            witness_budget=witness_budget)
    if isinstance(graph, plane_placement.PlacementRefusal) or trace is None:
        refuse_payload("graph", getattr(graph, "reason", "graph_refused"))
        return _base(evidences, frame_ids, status="insufficient_evidence",
                     reason=getattr(graph, "reason", "graph_refused"),
                     entry_sha256=entry.entry_sha256,
                     lattice_sha256=lattice.result_sha256,
                     signatures_sha256=signatures.result_sha256)
    if wants_bundle and not ((evidence_bundle_sink or []) or private_payload_output):
        bundle = _assemble_evidence_bundle(
            evidences, frame_ids, supplied,
            tuple(signature_bindings), tuple(matcher_envelopes), trace)
        if evidence_bundle_sink is not None:
            evidence_bundle_sink.append(bundle)
        if wants_payload:
            if isinstance(bundle, SyntheticPreflightEvidenceRefusalV1):
                private_payload_output.append(bundle)
            else:
                private_payload_output.append(_assemble_witness_payload(
                    bundle, evidences, supplied_payloads, tuple(signature_payloads),
                    tuple(matcher_payloads), max_witness_payload_bytes))
    # A refusal is one atomic outcome shared by every requested observer.  An
    # earlier private validation must not suppress the public hash observer.
    refusal = next((row for row in private_payload_output
                    if isinstance(row, SyntheticPreflightEvidenceRefusalV1)), None)
    if refusal is not None:
        for target in (evidence_bundle_sink, evidence_payload_sink,
                       surface_support_sink):
            if target is not None and not target:
                target.append(refusal)
    if surface_support_sink is not None and not surface_support_sink:
        if (len(private_payload_output) != 1
                or not isinstance(private_payload_output[0],
                                  SyntheticPreflightWitnessPayloadV1)):
            surface_support_sink.append(SyntheticPreflightEvidenceRefusalV1(
                "refused", "upstream_refusal", "surface_support",
                "complete private witness payload is required"))
        else:
            witness_payload = private_payload_output[0]
            evaluated = _evaluate_surface_support_payload(
                witness_payload, evidences, signatures.signatures,
                max_surface_support_pairs)
            if isinstance(evaluated, tuple):
                surface_support_sink.extend(evaluated)
            else:
                surface_support_sink.append(evaluated)
    bindings, binding_diagnostics = _surface_binding_diagnostics(
        evidences, frame_ids, graph.readings)
    bound_sets = tuple((frame, tuple(sorted({
        surface for (bound_frame, _index), surface in bindings.items()
        if bound_frame == frame}))) for frame in frame_ids)
    target_frames = frame_ids[:3]
    common = set.intersection(*(
        {surface for (frame, _index), surface in bindings.items() if frame == target}
        for target in target_frames))
    if len(common) != 1:
        return _base(evidences, frame_ids, status="unresolved_surface",
                     reason="unresolved_surface", entry_sha256=entry.entry_sha256,
                     lattice_sha256=lattice.result_sha256,
                     signatures_sha256=signatures.result_sha256,
                     graph_sha256=graph.result_sha256,
                     trace_sha256=trace.attribution_sha256,
                     solver_reason=graph.reason,
                     reading_bindings=binding_diagnostics,
                     bound_surface_sets=bound_sets)
    surface_id = next(iter(common))
    rows = []
    for item in trace.pair_attribution:
        if (not item.selected or item.left not in target_frames
                or item.right not in target_frames or item.left_plane is None
                or item.right_plane is None
                or bindings.get((item.left, item.left_plane)) != surface_id
                or bindings.get((item.right, item.right_plane)) != surface_id):
            continue
        required = (item.overlap_cells, item.overlap_fraction, item.cell_shift_rc,
                    item.left_structural_share, item.right_structural_share, item.psr)
        if any(value is None for value in required):
            continue
        rows.append(PredictedPair(
            item.left, item.right, surface_id, item.left_plane, item.right_plane,
            item.overlap_cells, item.overlap_fraction, item.cell_shift_rc,
            min(item.left_structural_share, item.right_structural_share),
            item.psr if np.isfinite(item.psr) else "positive_infinity"))
    rows.sort(key=lambda row: (row.left, row.right, row.left_reading_index,
                               row.right_reading_index))
    required_pairs = {tuple(sorted((target_frames[a], target_frames[b])))
                      for a, b in ((0, 1), (0, 2), (1, 2))}
    observed = {tuple(sorted((row.left, row.right))) for row in rows}
    status = "predicted" if required_pairs.issubset(observed) else "insufficient_evidence"
    reason = "" if status == "predicted" else "missing_required_pair"
    return _base(
        evidences, frame_ids, status=status, reason=reason,
        selected_surface_id=surface_id, entry_sha256=entry.entry_sha256,
        lattice_sha256=lattice.result_sha256,
        signatures_sha256=signatures.result_sha256,
        graph_sha256=graph.result_sha256, trace_sha256=trace.attribution_sha256,
        solver_reason=graph.reason, reading_bindings=binding_diagnostics,
        bound_surface_sets=bound_sets, pairs=tuple(rows))
