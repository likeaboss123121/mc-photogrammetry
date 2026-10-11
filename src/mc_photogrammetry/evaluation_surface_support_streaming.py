"""Evaluation-only staged surface support with one sampling witness at a time."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
import numpy as np

from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry import evaluation_compact_truth as compact_mod
from mc_photogrammetry import evaluation_signature_sampling as sampling_mod
from mc_photogrammetry import evaluation_surface_support as support
from mc_photogrammetry.pipeline import image_derived_signatures as signature_mod
from mc_photogrammetry.reconstruction import plane_graph

SCHEMA = "mc-photogrammetry-signature-cell-ownership-summary-v1"
DEFAULT_MAX_CELLS = 1_000_000
DEFAULT_MAX_RETAINED_BYTES = 64 * 1024 * 1024

def _freeze_truth(value):
    cls = (compact_mod.CompactRenderedTruthSourceV1
           if isinstance(value, compact_mod.CompactRenderedTruthSourceV1)
           else truth_mod.RenderedTruthEvidence)
    return cls(**{name:getattr(value,name) for name in value.__dataclass_fields__})

def _owner_binding(truth, witness):
    if isinstance(truth, truth_mod.RenderedTruthEvidence):
        return truth_mod.bind_renderer_ownership_witness(truth,witness)
    if len(witness.owners) != len(truth.owner_index):
        raise ValueError("renderer owner index does not bind compact evidence")
    for row,(position,slot) in zip(witness.owners,truth.owner_index):
        if row.owner_id < 0 or row.block_coordinate != position or row.slot != slot:
            raise ValueError("renderer owner index does not bind compact evidence")
    ownership_hash=hashlib.sha256(truth.ownership.tobytes(order="C")).hexdigest()
    owner_index_hash=_sha([[list(position),slot]
                           for position,slot in truth.owner_index])
    return truth_mod.RenderedTruthOwnershipBindingV1(
        "mc-photogrammetry-rendered-truth-owner-binding-v1",
        truth.evidence_sha256,ownership_hash,owner_index_hash,witness.witness_sha256)


@dataclass(frozen=True, slots=True)
class StreamingSurfaceSupportBundleV1:
    schema: str
    bi_merge_sha256: str
    left_summary_sha256: str
    right_summary_sha256: str
    matcher_envelope_sha256: str
    bundle_sha256: str = ""

    def __post_init__(self):
        if (self.schema != "mc-photogrammetry-streaming-surface-support-bundle-v1"
                or any(len(value) != 64 for value in (
                    self.bi_merge_sha256, self.left_summary_sha256,
                    self.right_summary_sha256, self.matcher_envelope_sha256))):
            raise ValueError("streaming bundle is malformed")
        digest = _sha(self.to_record(False))
        if self.bundle_sha256 not in ("", digest):
            raise ValueError("streaming bundle digest mismatch")
        object.__setattr__(self, "bundle_sha256", digest)

    def to_record(self, include_digest=True):
        result = {"schema": self.schema, "bi_merge_sha256": self.bi_merge_sha256,
                  "left_summary_sha256": self.left_summary_sha256,
                  "right_summary_sha256": self.right_summary_sha256,
                  "matcher_envelope_sha256": self.matcher_envelope_sha256}
        if include_digest: result["bundle_sha256"] = self.bundle_sha256
        return result


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SignatureCellOwnershipSummaryV1:
    schema: str
    bi_merge_sha256: str
    frame_id: str
    canonical_index: int
    alternative_index: int
    signature_sha256: str
    sampling_binding_sha256: str
    sampling_witness_sha256: str
    truth_evidence_sha256: str
    truth_sha256: str
    ownership_binding_sha256: str
    ownership_witness_sha256: str
    rows: int
    columns: int
    coverage_sha256: str
    physical_planes: tuple[tuple[str, int, int, int, int], ...]
    cell_states: tuple[int, ...]  # -1 unknown, -2 mixed, otherwise plane index
    summary_sha256: str = ""

    def __post_init__(self):
        if (self.schema != SCHEMA or len(self.bi_merge_sha256) != 64
                or len(self.signature_sha256) != 64 or self.rows <= 0
                or self.columns <= 0 or len(self.cell_states) != self.rows*self.columns
                or any(value < -2 or value >= len(self.physical_planes)
                       for value in self.cell_states)
                or any(len(value) != 5 or len(value[0]) != 64
                       or any(type(item) is not int for item in value[1:])
                       for value in self.physical_planes)
                or tuple(sorted(set(self.physical_planes))) != self.physical_planes):
            raise ValueError("cell ownership summary is malformed")
        digest = _sha(self.to_record(False))
        if self.summary_sha256 not in ("", digest):
            raise ValueError("cell ownership summary digest mismatch")
        object.__setattr__(self, "summary_sha256", digest)

    def to_record(self, include_digest=True):
        result = {"schema": self.schema, "bi_merge_sha256": self.bi_merge_sha256,
                  "identity": [self.frame_id, self.canonical_index,
                               self.alternative_index, self.signature_sha256],
                  "sampling_binding_sha256": self.sampling_binding_sha256,
                  "sampling_witness_sha256": self.sampling_witness_sha256,
                  "truth_evidence_sha256": self.truth_evidence_sha256,
                  "truth_sha256": self.truth_sha256,
                  "ownership_binding_sha256": self.ownership_binding_sha256,
                  "ownership_witness_sha256": self.ownership_witness_sha256,
                  "shape": [self.rows, self.columns],
                  "coverage_sha256": self.coverage_sha256,
                  "physical_planes": [list(value) for value in self.physical_planes],
                  "cell_states": list(self.cell_states)}
        if include_digest:
            result["summary_sha256"] = self.summary_sha256
        return result


def summarize_signature_ownership(*, bi_merge_sha256, signature_evidence,
                                  sampling_payload, truth_evidence,
                                  ownership_payload,
                                  max_cells=DEFAULT_MAX_CELLS,
                                  max_retained_bytes=DEFAULT_MAX_RETAINED_BYTES,
                                  ownership_diagnostic_sink=None,
                                  selected_cells=(), max_diagnostic_examples=8,
                                  max_diagnostic_bytes=256 * 1024 * 1024):
    try:
        if ownership_diagnostic_sink is not None and len(ownership_diagnostic_sink):
            raise ValueError("ownership diagnostic sink must be empty")
        if type(max_diagnostic_bytes) is not int or max_diagnostic_bytes < 0:
            raise ValueError("ownership diagnostic byte cap is invalid")
        signature_evidence = signature_mod.SignatureEvidence(**{
            name: getattr(signature_evidence, name)
            for name in signature_evidence.__dataclass_fields__})
        truth_evidence = _freeze_truth(truth_evidence)
        sampling_payload = sampling_mod.SignatureSamplingWitnessPayloadV1(
            sampling_payload.binding, sampling_payload.witness)
        ownership_payload = truth_mod.RenderedTruthOwnershipWitnessPayloadV1(
            ownership_payload.binding, ownership_payload.witness)
        if (type(max_cells) is not int or max_cells < 0
                or type(max_retained_bytes) is not int or max_retained_bytes < 0
                or len(bi_merge_sha256) != 64):
            raise ValueError("summary budget or campaign binding is invalid")
        shape = signature_evidence.signature.cells.shape
        if (ownership_diagnostic_sink is not None
                and truth_evidence.ownership.size * 12 > max_diagnostic_bytes):
            return support._refusal("budget_exhausted",
                                    "ownership diagnostic bitmap cap exceeded")
        if int(np.prod(shape)) > max_cells:
            return support._refusal("budget_exhausted", "summary cell cap exceeded")
        binding = sampling_payload.binding
        if ((binding.frame_id, binding.canonical_reading_index,
             binding.alternative_index, binding.signature_sha256) !=
                (signature_evidence.frame_id, signature_evidence.canonical_index,
                 signature_evidence.alternative_index,
                 signature_evidence.signature_sha256)
                or binding.truth_evidence_sha256 != truth_evidence.evidence_sha256
                or ownership_payload.binding.evidence_sha256 != truth_evidence.evidence_sha256
                or _owner_binding(
                    truth_evidence, ownership_payload.witness) != ownership_payload.binding):
            raise ValueError("summary inputs do not join")
        diagnostic=[] if ownership_diagnostic_sink is not None else None
        states = support._cell_owners(
            sampling_payload, ownership_payload, truth_evidence,
            diagnostic_sink=diagnostic,
            selected_cells=selected_cells,max_examples=max_diagnostic_examples,
            max_diagnostic_bytes=max_diagnostic_bytes)
        if len(states) != int(np.prod(shape)):
            raise ValueError("cell ownership shape mismatch")
        planes = tuple(sorted({key for state, key in states if state == "pure"}))
        if any(key[0] != truth_evidence.truth_sha256 for key in planes):
            raise ValueError("physical plane belongs to a foreign truth namespace")
        lookup = {key: index for index, key in enumerate(planes)}
        encoded = tuple(-1 if state == "unknown" else -2 if state == "mixed"
                        else lookup[key] for state, key in states)
        coverage_sha = hashlib.sha256(
            signature_evidence.signature.coverage.tobytes(order="C")).hexdigest()
        result = SignatureCellOwnershipSummaryV1(
            SCHEMA, bi_merge_sha256, signature_evidence.frame_id,
            signature_evidence.canonical_index, signature_evidence.alternative_index,
            signature_evidence.signature_sha256, binding.report_sha256,
            sampling_payload.witness.witness_sha256, truth_evidence.evidence_sha256,
            truth_evidence.truth_sha256, ownership_payload.binding.binding_sha256,
            ownership_payload.witness.witness_sha256, shape[0], shape[1],
            coverage_sha, planes, encoded)
        retained = len(json.dumps(result.to_record(), sort_keys=True,
                                  separators=(",", ":")).encode())
        if retained > max_retained_bytes:
            return support._refusal("budget_exhausted", "summary byte cap exceeded")
        if ownership_diagnostic_sink is not None:
            ownership_diagnostic_sink.extend(diagnostic)
        return result
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return support._refusal("provenance_mismatch", str(error))


def replay_and_summarize_endpoint(*, bi_merge_sha256, signature_evidence,
                                  truth_evidence, ownership_payload,
                                  witness_budget=None,
                                  max_cells=DEFAULT_MAX_CELLS,
                                  max_retained_bytes=DEFAULT_MAX_RETAINED_BYTES,
                                  ownership_diagnostic_sink=None,
                                  selected_cells=(), max_diagnostic_examples=8,
                                  max_diagnostic_bytes=256 * 1024 * 1024,
                                  replay=sampling_mod.replay_signature_sampling_witness):
    """Replay one witness, reduce it immediately, and retain no raw payload."""
    if ownership_diagnostic_sink is not None:
        try:
            planned=int(truth_evidence.ownership.size)*12
        except (AttributeError,TypeError,ValueError,OverflowError) as error:
            return support._refusal("provenance_mismatch",str(error))
        if (type(max_diagnostic_bytes) is not int or max_diagnostic_bytes < 0):
            return support._refusal("provenance_mismatch",
                                    "ownership diagnostic byte cap is invalid")
        if planned > max_diagnostic_bytes:
            return support._refusal("budget_exhausted",
                                    "ownership diagnostic bitmap cap exceeded")
    payload = []
    replay_result = replay(
        signature_evidence=signature_evidence, truth_evidence=truth_evidence,
        witness_budget=witness_budget, witness_payload_sink=payload)
    if not isinstance(replay_result, sampling_mod.SignatureSamplingBindingV1):
        reason = ("budget_exhausted" if replay_result.reason == "budget_exhausted"
                  else "provenance_mismatch")
        return support._refusal(reason,
                                f"sampling replay refused: {replay_result.reason}/{replay_result.stage}")
    if len(payload) != 1:
        return support._refusal("provenance_mismatch",
                                "sampling payload publication is incomplete")
    return summarize_signature_ownership(
        bi_merge_sha256=bi_merge_sha256,
        signature_evidence=signature_evidence, sampling_payload=payload.pop(),
        truth_evidence=truth_evidence, ownership_payload=ownership_payload,
        max_cells=max_cells, max_retained_bytes=max_retained_bytes,
        ownership_diagnostic_sink=ownership_diagnostic_sink,
        selected_cells=selected_cells,
        max_diagnostic_examples=max_diagnostic_examples,
        max_diagnostic_bytes=max_diagnostic_bytes)


def replay_and_summarize_compact_endpoint(**kwargs):
    """Streaming adapter selecting the explicit compact replay implementation."""
    kwargs.setdefault("replay", compact_mod.replay_compact_signature)
    return replay_and_summarize_endpoint(**kwargs)


def _retained_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray): return value.nbytes
    if isinstance(value, (str, bytes)): return len(value)
    if isinstance(value, (int, float, bool)) or value is None: return 8
    if hasattr(value, "__dataclass_fields__"):
        return sum(_retained_bytes(getattr(value, name), seen)
                   for name in value.__dataclass_fields__)
    if isinstance(value, dict):
        return sum(_retained_bytes(k, seen)+_retained_bytes(v, seen)
                   for k, v in value.items())
    if isinstance(value, (tuple, list)): return sum(_retained_bytes(v, seen) for v in value)
    return 0


def _descriptor_reservation(row):
    signature = row["signature_evidence"]
    cells = int(np.prod(signature.signature.cells.shape))
    owner = row["ownership_payload"]
    planes = {entry.physical_plane_key for entry in owner.witness.owners
              if entry.physical_plane_key is not None}
    # Worst canonical state token is the largest possible dictionary index.
    state_bytes = cells * max(3, len(str(max(len(planes)-1, 0))) + 1)
    metadata = _retained_bytes(row) + sum(
        len(json.dumps(("f"*64,) + tuple(plane), separators=(",", ":")))
        for plane in planes)
    return state_bytes + metadata + 4096


def _incremental_summary_reservation(row):
    """Upper bound for a summary added beside an already-retained descriptor."""
    return max(0, _descriptor_reservation(row) - _retained_bytes(row))


def summarize_endpoints_atomically(requests, sink, *,
                                   max_aggregate_summary_bytes=256*1024*1024,
                                   planned_pair_work=0, max_pair_work=10_000_000,
                                   matcher_payloads=()):
    if sink is None or len(sink):
        raise ValueError("summary sink must be empty")
    if (type(max_aggregate_summary_bytes) is not int
            or max_aggregate_summary_bytes < 0 or type(planned_pair_work) is not int
            or planned_pair_work < 0 or type(max_pair_work) is not int
            or max_pair_work < 0):
        raise ValueError("streaming aggregate budget is invalid")
    requests = tuple(requests)  # Freeze generators exactly once.
    matcher_payloads = tuple(matcher_payloads)
    try:
        frozen_matchers = tuple(plane_graph.GraphMatcherWitnessPayloadV1(
            value.envelope, value.witness) for value in matcher_payloads)
        actual_pair_work = sum(len(value.witness.logical_pairs)
                               for value in frozen_matchers)
        if any(len(value.witness.logical_pairs) != value.witness.selected_support
               or len(value.witness.logical_pairs) != value.envelope.overlap_cells
               for value in frozen_matchers):
            raise ValueError("matcher pair-work conservation failed")
        if planned_pair_work != actual_pair_work:
            raise ValueError("declared pair work differs from matcher payloads")
    except (AttributeError, TypeError, ValueError) as error:
        refusal = support._refusal("provenance_mismatch", str(error))
        sink.append(refusal); return refusal
    if actual_pair_work > max_pair_work:
        refusal = support._refusal("budget_exhausted", "pair work cap exceeded")
        sink.append(refusal); return refusal
    # Count shared descriptor, evidence, ownership, matcher, warp, and pair objects
    # once. logical_pairs is already retained by frozen_matchers; pair work remains
    # an independent operation cap and is not charged a second time as bytes.
    context_bytes = _retained_bytes((requests, frozen_matchers))
    summary_bound = sum(_incremental_summary_reservation(row)
                        for row in requests)
    scratch_bound = 0  # replay allocation is governed by its WitnessBudgetV1.
    planned = context_bytes + summary_bound + scratch_bound
    if planned > max_aggregate_summary_bytes:
        refusal = support._refusal("budget_exhausted",
                                   "streaming aggregate reservation exceeded")
        sink.append(refusal); return refusal
    pending = []
    retained = 0
    for request in requests:
        value = replay_and_summarize_endpoint(**request)
        if not isinstance(value, SignatureCellOwnershipSummaryV1):
            sink.append(value); return value
        retained += len(json.dumps(value.to_record(), sort_keys=True,
                                   separators=(",", ":")).encode())
        if context_bytes + retained > max_aggregate_summary_bytes:
            refusal = support._refusal("budget_exhausted",
                                       "exact cumulative summary bytes exceeded")
            sink.append(refusal); return refusal
        pending.append(value)
    sink.extend(pending)
    return tuple(pending)


def _decoded(summary):
    return tuple(("unknown", None) if value == -1 else
                 ("mixed", None) if value == -2 else
                 ("pure", summary.physical_planes[value])
                 for value in summary.cell_states)


def _validate_sources(summary, signature, sampling_binding, truth,
                      ownership_binding):
    sampling_binding = sampling_mod.SignatureSamplingBindingV1(**{
        name: getattr(sampling_binding, name)
        for name in sampling_binding.__dataclass_fields__})
    truth = _freeze_truth(truth)
    ownership_binding = truth_mod.RenderedTruthOwnershipBindingV1(**{
        name: getattr(ownership_binding, name)
        for name in ownership_binding.__dataclass_fields__})
    identity = (signature.frame_id, signature.canonical_index,
                signature.alternative_index, signature.signature_sha256)
    if (identity != (summary.frame_id, summary.canonical_index,
                     summary.alternative_index, summary.signature_sha256)
            or identity != (sampling_binding.frame_id,
                            sampling_binding.canonical_reading_index,
                            sampling_binding.alternative_index,
                            sampling_binding.signature_sha256)
            or summary.sampling_binding_sha256 != sampling_binding.report_sha256
            or summary.sampling_witness_sha256 != sampling_binding.witness_sha256
            or sampling_binding.truth_evidence_sha256 != truth.evidence_sha256
            or summary.truth_evidence_sha256 != truth.evidence_sha256
            or summary.truth_sha256 != truth.truth_sha256
            or summary.ownership_binding_sha256 != ownership_binding.binding_sha256
            or summary.ownership_witness_sha256 != ownership_binding.renderer_witness_sha256
            or ownership_binding.evidence_sha256 != truth.evidence_sha256
            or ownership_binding.ownership_sha256 != hashlib.sha256(
                truth.ownership.tobytes(order="C")).hexdigest()
            or ownership_binding.owner_index_sha256 != _sha([
                [list(position), slot] for position, slot in truth.owner_index])
            or any(plane[0] != truth.truth_sha256 for plane in summary.physical_planes)):
        raise ValueError("summary does not bind accepted source records")


def evaluate_streaming_surface_support(*, bi_merge_sha256, left_summary,
                                       right_summary, left_signature,
                                       right_signature, matcher_payload,
                                       left_sampling_binding=None,
                                       right_sampling_binding=None,
                                       left_truth=None, right_truth=None,
                                       left_ownership_binding=None,
                                       right_ownership_binding=None,
                                       max_pair_cells=support.DEFAULT_MAX_PAIR_CELLS):
    try:
        left_summary = SignatureCellOwnershipSummaryV1(**{
            name: getattr(left_summary, name) for name in left_summary.__dataclass_fields__})
        right_summary = SignatureCellOwnershipSummaryV1(**{
            name: getattr(right_summary, name) for name in right_summary.__dataclass_fields__})
        left_signature = signature_mod.SignatureEvidence(**{
            name: getattr(left_signature, name) for name in left_signature.__dataclass_fields__})
        right_signature = signature_mod.SignatureEvidence(**{
            name: getattr(right_signature, name) for name in right_signature.__dataclass_fields__})
        matcher_payload = plane_graph.GraphMatcherWitnessPayloadV1(
            matcher_payload.envelope, matcher_payload.witness)
        _validate_sources(left_summary, left_signature, left_sampling_binding,
                          left_truth, left_ownership_binding)
        _validate_sources(right_summary, right_signature, right_sampling_binding,
                          right_truth, right_ownership_binding)
        envelope, witness = matcher_payload.envelope, matcher_payload.witness
        if (type(max_pair_cells) is not int or max_pair_cells < 0
                or left_summary.bi_merge_sha256 != bi_merge_sha256
                or right_summary.bi_merge_sha256 != bi_merge_sha256
                or envelope.left_signature_sha256 != left_summary.signature_sha256
                or envelope.right_signature_sha256 != right_summary.signature_sha256
                or left_signature.signature_sha256 != left_summary.signature_sha256
                or right_signature.signature_sha256 != right_summary.signature_sha256
                or left_signature.signature.cells.shape != (left_summary.rows,left_summary.columns)
                or right_signature.signature.cells.shape != (right_summary.rows,right_summary.columns)
                or hashlib.sha256(left_signature.signature.coverage.tobytes(order="C")).hexdigest() != left_summary.coverage_sha256
                or hashlib.sha256(right_signature.signature.coverage.tobytes(order="C")).hexdigest() != right_summary.coverage_sha256):
            raise ValueError("streaming endpoint join mismatch")
        pairs = witness.logical_pairs
        if len(pairs) > max_pair_cells:
            return support._refusal("budget_exhausted", "pair cell cap exceeded")
        if len(pairs) != witness.selected_support or len(pairs) != envelope.overlap_cells:
            raise ValueError("matcher pair conservation failed")
        left_states, right_states = _decoded(left_summary), _decoded(right_summary)
        lw, rw = left_summary.columns, right_summary.columns
        counts = dict(same=0, different=0, mixed=0, unknown=0)
        supported, le, re = {}, {}, {}
        for lr, lc, rr, rc in pairs:
            if not (0 <= lr < left_summary.rows and 0 <= lc < lw
                    and 0 <= rr < right_summary.rows and 0 <= rc < rw):
                raise ValueError("matcher pair indexes outside signature dimensions")
            ls, lk = left_states[lr*lw+lc]; rs, rk = right_states[rr*rw+rc]
            category = "unknown" if "unknown" in (ls,rs) else "mixed" if "mixed" in (ls,rs) else "different" if lk != rk else "same"
            counts[category] += 1
            if category == "same":
                supported[lk] = supported.get(lk,0)+1
                le.setdefault(lk,set()).add((lr,lc)); re.setdefault(lk,set()).add((rr,rc))
        warp = witness.warp_witness
        if (warp is None or warp.source_shape != (left_summary.rows,left_summary.columns)
                or warp.target_shape != (right_summary.rows,right_summary.columns)
                or witness.canvas_shape != (right_summary.rows + 4*witness.max_shift_cells,
                                             right_summary.columns + 4*witness.max_shift_cells)
                or witness.algorithm != plane_graph.PLANE_CORRELATION_ALGORITHM
                or not np.array_equal(warp.homography, witness.homography)):
            raise ValueError("matcher warp dimensions mismatch")
        left_canvas=[]
        for row in range(right_summary.rows):
            for column in range(right_summary.columns):
                if not warp.geometric_valid[row,column]: left_canvas.append(("unknown",None)); continue
                sr,sc=int(warp.source_rows[row,column]),int(warp.source_columns[row,column])
                left_canvas.append(left_states[sr*lw+sc] if 0<=sr<left_summary.rows and 0<=sc<lw and left_signature.signature.coverage[sr,sc] else ("unknown",None))
        right_coverage=right_signature.signature.coverage.reshape(-1)
        surfaces=[]
        for key in sorted(supported):
            a=sum(state=="pure" and plane==key for state,plane in left_canvas)
            b=sum(bool(c) and state=="pure" and plane==key for c,(state,plane) in zip(right_coverage,right_states))
            denominator=min(a,b); total=len(pairs)
            surfaces.append(support.SurfaceSupportRowV1(
                key[1:],supported[key],len(le[key]),len(re[key]),a,b,
                supported[key]/total if total else 0.0,
                None if denominator==0 else supported[key]/denominator))
        bundle = StreamingSurfaceSupportBundleV1(
            "mc-photogrammetry-streaming-surface-support-bundle-v1",
            bi_merge_sha256, left_summary.summary_sha256,
            right_summary.summary_sha256, envelope.envelope_sha256)
        return support.SurfaceLocalSupportReportV1(
            support.SCHEMA,"complete",support.SAMPLING_CONTRACT,
            plane_graph.PLANE_CORRELATION_ALGORITHM,bundle.bundle_sha256,
            left_summary.truth_sha256,right_summary.truth_sha256,
            left_summary.signature_sha256,right_summary.signature_sha256,
            envelope.envelope_sha256,
            (left_summary.sampling_witness_sha256,right_summary.sampling_witness_sha256),
            (left_summary.ownership_witness_sha256,right_summary.ownership_witness_sha256),
            witness.witness_sha256,envelope.overlap_cells,envelope.psr,len(pairs),
            counts["same"],counts["different"],counts["mixed"],counts["unknown"],
            tuple(surfaces))
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return support._refusal("provenance_mismatch", str(error))
