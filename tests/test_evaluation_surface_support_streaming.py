from dataclasses import replace
import gc
from types import SimpleNamespace
import numpy as np

import pytest

from mc_photogrammetry import evaluation_surface_support as legacy
from mc_photogrammetry import evaluation_surface_support_streaming as subject
from mc_photogrammetry import evaluation_signature_sampling as sampling
from test_evaluation_surface_support import _payload_fixture


MERGE = "9" * 64


def _summaries():
    payload, left, right, left_signature, right_signature = _payload_fixture()
    left_summary = subject.summarize_signature_ownership(
        bi_merge_sha256=MERGE, signature_evidence=left_signature,
        sampling_payload=payload.signatures[0], truth_evidence=left,
        ownership_payload=payload.ownership[0])
    right_summary = subject.summarize_signature_ownership(
        bi_merge_sha256=MERGE, signature_evidence=right_signature,
        sampling_payload=payload.signatures[1], truth_evidence=right,
        ownership_payload=payload.ownership[1])
    return payload, left, right, left_signature, right_signature, left_summary, right_summary


def _evaluate(payload, left, right, ls, rs, lsum, rsum, **kwargs):
    left_sampling = kwargs.pop("left_sampling_binding", payload.signatures[0].binding)
    left_owner = kwargs.pop("left_ownership_binding", payload.ownership[0].binding)
    return subject.evaluate_streaming_surface_support(
        bi_merge_sha256=MERGE, left_summary=lsum, right_summary=rsum,
        left_signature=ls, right_signature=rs, matcher_payload=payload.matches[0],
        left_sampling_binding=left_sampling,
        right_sampling_binding=payload.signatures[1].binding,
        left_truth=left, right_truth=right,
        left_ownership_binding=left_owner,
        right_ownership_binding=payload.ownership[1].binding, **kwargs)


def test_streaming_report_matches_aggregate_reducer_arithmetic():
    payload, left, right, ls, rs, lsum, rsum = _summaries()
    expected = legacy.evaluate_surface_support_from_payload(
        payload=payload, left_truth=left, right_truth=right,
        left_signature=ls, right_signature=rs)
    actual = _evaluate(payload, left, right, ls, rs, lsum, rsum)
    assert actual.status == expected.status == "complete"
    assert (actual.pair_count, actual.same_surface_pairs,
            actual.different_surface_pairs, actual.mixed_pairs,
            actual.unknown_pairs, actual.surfaces) == (
                expected.pair_count, expected.same_surface_pairs,
                expected.different_surface_pairs, expected.mixed_pairs,
                expected.unknown_pairs, expected.surfaces)


def test_summary_is_digest_bound_row_major_and_aliases_planes():
    *_prefix, summary, _right = _summaries()
    assert len(summary.cell_states) == summary.rows * summary.columns
    assert len(summary.physical_planes) == len(set(summary.physical_planes))
    assert subject.SignatureCellOwnershipSummaryV1(**{
        name: getattr(summary, name) for name in summary.__dataclass_fields__}) == summary
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(summary, summary_sha256="0" * 64)
    with pytest.raises(ValueError, match="malformed"):
        replace(summary, cell_states=(len(summary.physical_planes),)
                + summary.cell_states[1:], summary_sha256="")


def test_mixed_unknown_border_and_pair_bounds_remain_conservative():
    payload, left, right, ls, rs, lsum, rsum = _summaries()
    mixed = replace(lsum, cell_states=(-2,) + lsum.cell_states[1:],
                    summary_sha256="")
    report = _evaluate(payload, left, right, ls, rs, mixed, rsum)
    assert report.pair_count == payload.matches[0].envelope.overlap_cells
    assert report.mixed_pairs + report.unknown_pairs >= 1
    pairs = payload.matches[0].witness.logical_pairs.copy()
    pairs[0, 0] = -1; pairs.flags.writeable = False
    with pytest.raises(ValueError, match="sentinel logical pair"):
        replace(payload.matches[0].witness, logical_pairs=pairs,
                witness_sha256="")


def test_caps_and_late_failure_publish_atomically():
    payload, left, right, ls, rs, *_ = _summaries()
    capped = subject.summarize_signature_ownership(
        bi_merge_sha256=MERGE, signature_evidence=ls,
        sampling_payload=payload.signatures[0], truth_evidence=left,
        ownership_payload=payload.ownership[0], max_cells=0)
    assert capped.reason == "budget_exhausted"
    sink = []
    def replay_for(value):
        return lambda **kwargs: (kwargs["witness_payload_sink"].append(value)
                                 or value.binding)
    requests = [dict(bi_merge_sha256=MERGE, signature_evidence=ls,
                     truth_evidence=left, ownership_payload=payload.ownership[0],
                     replay=replay_for(payload.signatures[0])),
                dict(bi_merge_sha256=MERGE, signature_evidence=rs,
                     truth_evidence=right, ownership_payload=payload.ownership[0],
                     replay=replay_for(payload.signatures[1]))]
    result = subject.summarize_endpoints_atomically(requests, sink)
    assert result.status == "refused" and len(sink) == 1
    with pytest.raises(ValueError, match="must be empty"):
        subject.summarize_endpoints_atomically(requests, sink)


def test_aggregate_reservation_and_typed_replay_budget_refusal():
    payload, left, _right, ls, _rs, *_ = _summaries()
    request = dict(bi_merge_sha256=MERGE, signature_evidence=ls,
                   truth_evidence=left, ownership_payload=payload.ownership[0])
    sink = []
    refused = subject.summarize_endpoints_atomically(
        (request,), sink, max_aggregate_summary_bytes=0)
    assert refused.reason == "budget_exhausted" and len(sink) == 1
    typed = sampling.SignatureSamplingReplayRefusalV1(
        sampling.SCHEMA, "refused", "budget_exhausted", "samples",
        "limit=1;planned=2;actual=0", ls.signature_sha256,
        left.evidence_sha256)
    replayed = subject.replay_and_summarize_endpoint(
        **request, replay=lambda **_kwargs: typed)
    assert replayed.reason == "budget_exhausted"


def test_coverage_pair_budget_and_truth_namespace_tamper_refuse():
    payload, left, right, ls, rs, lsum, rsum = _summaries()
    changed = ls.signature.coverage.copy()
    changed.reshape(-1)[0] = ~changed.reshape(-1)[0]
    changed.flags.writeable = False
    changed_signature = replace(ls, signature=replace(ls.signature, coverage=changed),
                                signature_sha256="")
    refusal = _evaluate(payload, left, right, changed_signature, rs, lsum, rsum)
    assert refusal.reason == "provenance_mismatch"
    for invalid in (True, 1.0, -1):
        refusal = _evaluate(payload, left, right, ls, rs, lsum, rsum,
                            max_pair_cells=invalid)
        assert refusal.reason == "provenance_mismatch"
    plane = lsum.physical_planes[0]
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(lsum, physical_planes=(("0" * 64,) + plane[1:],),
                summary_sha256=lsum.summary_sha256)

    foreign_frame = replace(lsum, frame_id="foreign", summary_sha256="")
    assert _evaluate(payload, left, right, ls, rs, foreign_frame, rsum).status == "refused"
    foreign_sampling = replace(lsum, sampling_binding_sha256="0" * 64,
                               summary_sha256="")
    assert _evaluate(payload, left, right, ls, rs, foreign_sampling, rsum).status == "refused"
    foreign_truth = "0" * 64
    planes = tuple((foreign_truth,) + value[1:] for value in lsum.physical_planes)
    rehashed_truth = replace(lsum, truth_sha256=foreign_truth,
                             physical_planes=planes, summary_sha256="")
    assert _evaluate(payload, left, right, ls, rs, rehashed_truth, rsum).status == "refused"


def test_generator_descriptors_freeze_once_and_publish_all_rows():
    payload, left, right, ls, rs, *_ = _summaries()
    calls = []
    def request(value, signature, truth, ownership):
        def replay(**kwargs):
            calls.append(signature.signature_sha256)
            kwargs["witness_payload_sink"].append(value)
            return value.binding
        return dict(bi_merge_sha256=MERGE, signature_evidence=signature,
                    truth_evidence=truth, ownership_payload=ownership,
                    replay=replay)
    source = (row for row in (
        request(payload.signatures[0], ls, left, payload.ownership[0]),
        request(payload.signatures[1], rs, right, payload.ownership[1])))
    sink = []
    result = subject.summarize_endpoints_atomically(source, sink)
    assert len(result) == len(sink) == 2
    assert calls == [ls.signature_sha256, rs.signature_sha256]


def test_exact_cumulative_serialization_cap_cannot_be_underestimated():
    payload, left, _right, ls, _rs, *_ = _summaries()
    def replay(**kwargs):
        kwargs["witness_payload_sink"].append(payload.signatures[0])
        return payload.signatures[0].binding
    request = dict(bi_merge_sha256=MERGE, signature_evidence=ls,
                   truth_evidence=left, ownership_payload=payload.ownership[0],
                   replay=replay)
    exact = subject.replay_and_summarize_endpoint(**request)
    import json
    size = len(json.dumps(exact.to_record(), sort_keys=True,
                          separators=(",", ":")).encode())
    sink = []
    refused = subject.summarize_endpoints_atomically(
        (request,), sink, max_aggregate_summary_bytes=size - 1)
    assert refused.reason == "budget_exhausted" and len(sink) == 1


def test_source_binding_cross_hashes_are_recomputed_not_trusted():
    payload, left, right, ls, rs, lsum, rsum = _summaries()
    sampling_binding = replace(payload.signatures[0].binding,
                               truth_evidence_sha256="0" * 64,
                               report_sha256="")
    assert _evaluate(payload, left, right, ls, rs, lsum, rsum,
                     left_sampling_binding=sampling_binding).status == "refused"
    owner_binding = replace(payload.ownership[0].binding,
                            renderer_witness_sha256="0" * 64,
                            binding_sha256="")
    assert _evaluate(payload, left, right, ls, rs, lsum, rsum,
                     left_ownership_binding=owner_binding).status == "refused"
    raster_binding = replace(payload.ownership[0].binding,
                             ownership_sha256="0" * 64,
                             owner_index_sha256="1" * 64,
                             binding_sha256="")
    assert _evaluate(payload, left, right, ls, rs, lsum, rsum,
                     left_ownership_binding=raster_binding).status == "refused"


def test_combined_coexisting_context_plus_summary_cap_is_enforced():
    payload, left, _right, ls, _rs, *_ = _summaries()
    def replay(**kwargs):
        kwargs["witness_payload_sink"].append(payload.signatures[0])
        return payload.signatures[0].binding
    request = dict(bi_merge_sha256=MERGE, signature_evidence=ls,
                   truth_evidence=left, ownership_payload=payload.ownership[0],
                   replay=replay)
    summary = subject.replay_and_summarize_endpoint(**request)
    import json
    context = subject._retained_bytes(request)
    summary_bytes = len(json.dumps(summary.to_record(), sort_keys=True,
                                   separators=(",", ":")).encode())
    assert context > 0 and summary_bytes > 0
    sink = []
    result = subject.summarize_endpoints_atomically(
        (request,), sink,
        max_aggregate_summary_bytes=context + summary_bytes - 1)
    assert result.reason == "budget_exhausted" and len(sink) == 1


def test_pair_work_is_derived_before_replay_with_exact_cap_boundaries():
    payload, left, _right, ls, _rs, *_ = _summaries()
    calls = []
    def replay(**kwargs):
        calls.append(ls.signature_sha256)
        kwargs["witness_payload_sink"].append(payload.signatures[0])
        return payload.signatures[0].binding
    request = dict(bi_merge_sha256=MERGE, signature_evidence=ls,
                   truth_evidence=left, ownership_payload=payload.ownership[0],
                   replay=replay)
    matcher = payload.matches[0]
    actual = len(matcher.witness.logical_pairs)

    sink = []
    refused = subject.summarize_endpoints_atomically(
        (request,), sink, planned_pair_work=actual,
        max_pair_work=actual - 1, matcher_payloads=(matcher,))
    assert refused.reason == "budget_exhausted" and calls == []

    sink = []
    complete = subject.summarize_endpoints_atomically(
        (request,), sink, planned_pair_work=actual,
        max_pair_work=actual, matcher_payloads=(matcher,))
    assert len(complete) == len(sink) == 1 and len(calls) == 1

    calls.clear(); sink = []
    refused = subject.summarize_endpoints_atomically(
        (request,), sink, planned_pair_work=actual + 1,
        max_pair_work=actual + 1, matcher_payloads=(matcher,))
    assert refused.reason == "provenance_mismatch" and calls == []

    calls.clear(); sink = []
    complete = subject.summarize_endpoints_atomically(
        (request,), sink, planned_pair_work=0, max_pair_work=0,
        matcher_payloads=())
    assert len(complete) == 1 and len(calls) == 1


def test_reservation_counts_shared_context_once_and_summary_incrementally():
    payload, left, _right, ls, _rs, *_ = _summaries()
    calls = []
    def replay(**kwargs):
        calls.append(1)
        kwargs["witness_payload_sink"].append(payload.signatures[0])
        return payload.signatures[0].binding
    request = dict(bi_merge_sha256=MERGE, signature_evidence=ls,
                   truth_evidence=left, ownership_payload=payload.ownership[0],
                   replay=replay)
    shared = (request, request)
    unique_context = subject._retained_bytes((shared, ()))
    incremental = 2 * subject._incremental_summary_reservation(request)
    sink = []
    refused = subject.summarize_endpoints_atomically(
        shared, sink, max_aggregate_summary_bytes=unique_context+incremental-1)
    assert refused.reason == "budget_exhausted" and calls == []

    sink = []
    result = subject.summarize_endpoints_atomically(
        shared, sink, max_aggregate_summary_bytes=unique_context+incremental)
    assert len(result) == 2 and calls == [1, 1]
    distinct = (request, dict(request, unique_marker="x" * 100))
    assert subject._retained_bytes((distinct, ())) > unique_context


def test_exact_cumulative_context_plus_summary_boundary():
    payload, left, _right, ls, _rs, *_ = _summaries()
    def replay(**kwargs):
        kwargs["witness_payload_sink"].append(payload.signatures[0])
        return payload.signatures[0].binding
    request = dict(bi_merge_sha256=MERGE, signature_evidence=ls,
                   truth_evidence=left, ownership_payload=payload.ownership[0],
                   replay=replay)
    summary = subject.replay_and_summarize_endpoint(**request)
    import json
    context = subject._retained_bytes(((request,), ()))
    exact = len(json.dumps(summary.to_record(), sort_keys=True,
                           separators=(",", ":")).encode())
    sink = []
    refused = subject.summarize_endpoints_atomically(
        (request,), sink, max_aggregate_summary_bytes=context+exact-1)
    assert refused.reason == "budget_exhausted"
    sink = []
    result = subject.summarize_endpoints_atomically(
        (request,), sink,
        max_aggregate_summary_bytes=max(
            context+exact,
            context+subject._incremental_summary_reservation(request)))
    assert len(result) == 1


def test_summary_does_not_retain_witness_or_truth_objects():
    payload, left, _right, ls, _rs, *_ = _summaries()
    summary = subject.summarize_signature_ownership(
        bi_merge_sha256=MERGE, signature_evidence=ls,
        sampling_payload=payload.signatures[0], truth_evidence=left,
        ownership_payload=payload.ownership[0])
    assert not any(hasattr(summary, name) for name in
                   ("witness", "truth_evidence", "sampling_payload"))
    gc.collect()
    assert summary.summary_sha256


def test_ownership_diagnostic_is_output_neutral_bounded_and_atomic():
    payload, left, _right, ls, _rs, baseline, _ = _summaries()
    diagnostics=[]
    result=subject.summarize_signature_ownership(
        bi_merge_sha256=MERGE,signature_evidence=ls,
        sampling_payload=payload.signatures[0],truth_evidence=left,
        ownership_payload=payload.ownership[0],
        ownership_diagnostic_sink=diagnostics,selected_cells=(0,),
        max_diagnostic_examples=1)
    assert result==baseline and result.summary_sha256==baseline.summary_sha256
    assert len(diagnostics)==1 and diagnostics[0].selected_cells==1
    assert all(len(rows)<=1 for _reason,rows in diagnostics[0].examples)
    occupied=[object()]
    refused=subject.summarize_signature_ownership(
        bi_merge_sha256=MERGE,signature_evidence=ls,
        sampling_payload=payload.signatures[0],truth_evidence=left,
        ownership_payload=payload.ownership[0],
        ownership_diagnostic_sink=occupied)
    assert refused.reason=="provenance_mismatch" and occupied==occupied[:1]
    late=[]
    refused=subject.summarize_signature_ownership(
        bi_merge_sha256=MERGE,signature_evidence=ls,
        sampling_payload=payload.signatures[0],truth_evidence=left,
        ownership_payload=payload.ownership[0],
        ownership_diagnostic_sink=late,max_retained_bytes=0)
    assert refused.reason=="budget_exhausted" and late==[]


def test_ownership_diagnostic_counts_multiple_causes_deterministically():
    ownership=np.asarray([[7,0]],dtype=np.int64); ownership.flags.writeable=False
    usable=np.asarray([[False,True]],dtype=bool); usable.flags.writeable=False
    evidence=SimpleNamespace(ownership=ownership,usable=usable,truth_sha256="a"*64)
    witness=SimpleNamespace(cell_rectangles=(0,1,2),
        cell_source_ptr=np.asarray([0,0,2,3]),
        cell_source_indices=np.asarray([-1,0,1]))
    payload=SimpleNamespace(witness=witness)
    row=SimpleNamespace(owner_id=0,physical_plane_key=None)
    owners=SimpleNamespace(witness=SimpleNamespace(owners=(row,)))
    sink=[]
    states=legacy._cell_owners(payload,owners,evidence,diagnostic_sink=sink,
                               selected_cells=(1,2),max_examples=2)
    assert states==(("unknown",None),("unknown",None),("unknown",None))
    full=dict(sink[0].full_reason_counts); selected=dict(sink[0].selected_reason_counts)
    assert full=={"absent_owner_records":1,"empty_dependencies":1,
                  "out_of_range_indices":1,
                  "unresolved_physical_plane_identity":1,
                  "unusable_source_pixels":1}
    assert selected["empty_dependencies"]==0 and sum(selected.values())==4
    examples=dict(sink[0].examples)
    assert examples["out_of_range_indices"]==((1,-1),)
    absence=dict(sink[0].owner_absence_examples)
    assert absence["missing_owner_table"]==((1,0),)
    assert absence["background"]==()
    deps=dict(sink[0].full_dependency_counts)
    assert deps["unique_source_dependencies"]==2
    assert deps["border_sentinel_entries"]==1
    assert deps["invalid_index_entries"]==0
    assert deps["missing_owner_table_source_pixels"]==1
    assert deps["unresolved_plane_source_pixels"]==1
    assert deps["unusable_source_pixels"]==1
    coordinates=dict(sink[0].coordinate_examples)
    assert coordinates["border_sentinel"]==((1,-1,-1,-1),)
    assert coordinates["missing_owner_table"]==((1,0,0,0),)


def test_5329_cell_bitmap_accounting_is_bounded_and_exact():
    shape=(1080,1920); count=5329
    ownership=np.full(shape,-1,dtype=np.int32); ownership.flags.writeable=False
    usable=np.ones(shape,dtype=bool); usable.flags.writeable=False
    evidence=SimpleNamespace(ownership=ownership,usable=usable,truth_sha256="a"*64)
    witness=SimpleNamespace(cell_rectangles=tuple(range(count)),
        cell_source_ptr=np.zeros(count+1,dtype=np.int64),
        cell_source_indices=np.asarray([],dtype=np.int64))
    sink=[]
    states=legacy._cell_owners(SimpleNamespace(witness=witness),
        SimpleNamespace(witness=SimpleNamespace(owners=())),evidence,
        diagnostic_sink=sink,max_diagnostic_bytes=256*1024*1024)
    assert len(states)==count and dict(sink[0].full_reason_counts)["empty_dependencies"]==count
    assert ownership.size*12 < 256*1024*1024


def test_diagnostic_budget_refuses_before_sampling_replay():
    payload,left,_right,ls,_rs,*_= _summaries(); calls=[]; sink=[]
    result=subject.replay_and_summarize_endpoint(
        bi_merge_sha256=MERGE,signature_evidence=ls,truth_evidence=left,
        ownership_payload=payload.ownership[0],ownership_diagnostic_sink=sink,
        max_diagnostic_bytes=0,
        replay=lambda **kwargs: calls.append(kwargs))
    assert result.reason=="budget_exhausted" and calls==[] and sink==[]


def test_one_endpoint_replay_reduces_payload_and_reordered_join_refuses():
    payload, left, right, ls, rs, lsum, rsum = _summaries()
    replayed = subject.replay_and_summarize_endpoint(
        bi_merge_sha256=MERGE, signature_evidence=ls, truth_evidence=left,
        ownership_payload=payload.ownership[0],
        replay=lambda **kwargs: (kwargs["witness_payload_sink"].append(
            payload.signatures[0]) or payload.signatures[0].binding))
    assert isinstance(replayed, subject.SignatureCellOwnershipSummaryV1)
    assert replayed.cell_states == lsum.cell_states
    refusal = _evaluate(payload, left, right, ls, rs, rsum, lsum)
    assert refusal.status == "refused" and refusal.reason == "provenance_mismatch"
