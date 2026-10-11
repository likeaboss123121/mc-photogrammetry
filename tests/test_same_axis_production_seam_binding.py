"""Synthetic admitted-universe seam controls; no images, solver or inference."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice_planes as lp, plane_graph as pg
from mc_photogrammetry.reconstruction import _same_axis_production_seam_binding as seam


def reading(frame, *, axis=1, alternatives=((1., 3., 1.),), share=.6, side=1.):
    return lp.PlaneReading(frame, axis, side,
        tuple(lp.DistanceAlternative(*values) for values in alternatives),
        60., (0., 0.), 1., 512, None, 8, share, .9,
        lp.PlaneRectification(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                             500., (120., 90.), (0., 0.)))


def signature(source, *, b=1., cells=None, shape=(12, 12), coverage=None):
    if cells is None:
        cells = np.random.default_rng(81).random(shape).astype(np.float32)
    cells = np.asarray(cells, dtype=np.float32)
    return lp.PlaneSignature(source, b, cells, np.zeros_like(cells),
        np.ones(cells.shape, dtype=bool) if coverage is None else coverage, (0., 0.))


def fixture(*, share=.6, alternatives=((1., 3., 1.),), right_axis=1):
    left = reading("a", alternatives=alternatives, share=share)
    right = reading("b", axis=right_axis, share=share)
    return (left, right), (signature(left), signature(right))


def controlled_pair(left, right, *, quadrant, max_shift_cells, refusals=None,
                    diagnostic_sink=None, **kwargs):
    l, r = left.reading, right.reading
    score = .9 if left.blocks_per_period == 1. else .5
    if pg.axis_transform(l.axis, quadrant)[0] != r.axis:
        if refusals is not None:
            refusals.append(pg.PlaneMatchRefusal(l.frame, r.frame, l.axis, r.axis,
                                                quadrant, "axis_mismatch", "synthetic"))
        if diagnostic_sink is not None:
            diagnostic_sink.pairs.append(pg.PlanePairTrace(l.frame, r.frame, l.axis, r.axis,
                quadrant, False, "axis_mismatch", None, None, None, None, None, None))
        return None
    if diagnostic_sink is not None:
        diagnostic_sink.pairs.append(pg.PlanePairTrace(l.frame, r.frame, l.axis, r.axis,
            quadrant, True, "", None, None, None, (0, 0), 1., 64,
            psr=5., score=score, overlap_cells=64, cell_shift_rc=(0, 0)))
    return pg.PlaneMatch(l.frame, r.frame, l.axis, r.axis,
        left.blocks_per_period, right.blocks_per_period, quadrant,
        5., 1., .1, 64, None, score, cell_shift_rc=(0, 0))


def complete(result):
    assert result.status == "complete", result.reason
    assert result.receipt is not None
    assert not result.reconstruction_authorized and not result.receipt.reconstruction_authorized
    replay = seam.validate_receipt(result.receipt)
    assert replay.status == "complete", replay.reason
    assert replay == result
    return result.receipt


def atomic(result, reason=None):
    assert result.status == "refused"
    if reason is not None:
        assert result.reason == reason
    assert result.receipt is None and result.matches == result.refusals == ()
    assert not result.reconstruction_authorized


def collect(readings, signatures, **kwargs):
    return seam.collect_matcher_call(readings, signatures, max_shift_cells=4, **kwargs)


def test_all_yaws_exact_single_call_fresh_sinks(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    inputs = fixture()
    matcher = pg.match_plane_graph
    calls, sinks = [], []
    def observe(*args, **kwargs):
        assert kwargs["candidate_observation_sink"] == []
        assert kwargs["refusals"] == [] and kwargs["diagnostic_sink"].pairs == []
        sinks.extend((kwargs["candidate_observation_sink"], kwargs["refusals"],
                      kwargs["diagnostic_sink"]))
        calls.append(1)
        return matcher(*args, **kwargs)
    monkeypatch.setattr(pg, "match_plane_graph", observe)
    first = complete(collect(*inputs))
    second = complete(collect(*inputs))
    assert len(calls) == 2 and all(sinks[i] is not sinks[i + 3] for i in range(3))
    assert tuple(m.quadrant_of_right for m in first.candidates) == (0, 1, 2, 3)
    assert (first.ledger.hypotheses, first.ledger.structural_skips,
        first.ledger.attempts, first.ledger.admitted, first.ledger.kept, first.ledger.dropped
        ) == (4, 0, 4, 4, 4, 0)
    assert first.receipt_sha256 == second.receipt_sha256


def test_nine_frame_admitted_matcher_call_exceeds_legacy_eight_frame_limit(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    readings = tuple(reading(f"frame-{i}") for i in range(9))
    signatures = tuple(signature(row) for row in readings)
    result = complete(collect(readings, signatures))
    assert {frame for match in result.candidates for frame in (match.left, match.right)} == {
        f"frame-{i}" for i in range(9)}
    assert seam.HARD_CAPS.frames == 32


@pytest.mark.parametrize("kind", ["empty", "signatureless", "one_frame", "structural_skip"])
def test_normal_zero_admitted_completion(kind):
    readings, signatures = fixture(share=0. if kind == "structural_skip" else .6)
    if kind == "empty":
        readings, signatures = (), ()
    elif kind == "signatureless":
        signatures = ()
    elif kind == "one_frame":
        readings, signatures = readings[:1], signatures[:1]
    receipt = complete(collect(readings, signatures))
    assert receipt.ledger.admitted == receipt.ledger.kept == receipt.ledger.dropped == 0
    assert receipt.ledger.attempts == 0
    assert receipt.reservation.correlation_workspace_bytes == 0
    assert receipt.ledger.hypotheses == receipt.ledger.structural_skips


def test_exact_both_below_predicate():
    readings, signatures = fixture()
    low = replace(readings[0], structural_share=0.)
    readings = (low, readings[1])
    signatures = (signature(low), signatures[1])
    receipt = complete(collect(readings, signatures))
    assert receipt.ledger.hypotheses == receipt.ledger.attempts == 4
    assert receipt.ledger.structural_skips == 0 and receipt.ledger.admitted == 0
    assert "repetition_unresolved" in {row.reason for row in receipt.refusals}


@pytest.mark.parametrize("reason", ["axis_mismatch", "invalid_distance_alternative",
    "no_signal", "insufficient_overlap", "repetition_unresolved"])
def test_each_real_pair_refusal(reason):
    readings, signatures = fixture(right_axis=0 if reason == "axis_mismatch" else 1)
    if reason == "axis_mismatch":
        readings = (replace(readings[0], axis=1), readings[1])
        signatures = (signature(readings[0]), signature(readings[1]))
    elif reason == "invalid_distance_alternative":
        signatures = (signature(readings[0], b=2.), signatures[1])
    elif reason == "no_signal":
        signatures = tuple(signature(row, cells=np.zeros((12, 12))) for row in readings)
    elif reason == "insufficient_overlap":
        signatures = tuple(signature(row, shape=(2, 2)) for row in readings)
    elif reason == "repetition_unresolved":
        readings = (replace(readings[0], structural_share=0.), readings[1])
        signatures = tuple(signature(row) for row in readings)
    receipt = complete(collect(readings, signatures))
    assert receipt.ledger.admitted == 0
    assert reason in {refusal.reason for refusal in receipt.refusals}
    assert receipt.ledger.attempts == sum(n for _reason, n in receipt.ledger.pair_refusal_counts)


def test_real_matcher_with_without_collection_equivalence():
    readings, signatures = fixture()
    refusals = []
    baseline = pg.match_plane_graph(readings, signatures, max_shift_cells=4, refusals=refusals)
    result = collect(readings, signatures)
    receipt = complete(result)
    assert len(baseline) > 0
    assert result.matches == baseline and result.refusals == tuple(refusals)
    trace, observations, with_refusals = pg.PlaneGraphTrace([]), [], []
    with_collection = pg.match_plane_graph(readings, signatures, max_shift_cells=4,
        refusals=with_refusals, diagnostic_sink=trace, candidate_observation_sink=observations)
    assert with_collection == baseline and with_refusals == refusals
    assert observations[0].candidates == receipt.candidates
    assert observations[0].dispositions == receipt.dispositions


@pytest.mark.parametrize("all_dropped", [False, True])
def test_kept_dropped_and_all_dropped(monkeypatch, all_dropped):
    readings, _signatures = fixture(alternatives=((1., 3., 1.), (.5, 1.5, .8)))
    signatures = (signature(readings[0]), signature(readings[0], b=.5), signature(readings[1]))
    def pair(*args, **kwargs):
        found = controlled_pair(*args, **kwargs)
        if all_dropped and found is not None:
            found = replace(found, score=.9)
            trace = kwargs.get("diagnostic_sink")
            if trace is not None:
                trace.pairs[-1] = replace(trace.pairs[-1], score=.9)
        return found
    monkeypatch.setattr(pg, "match_plane_pair", pair)
    receipt = complete(collect(readings, signatures))
    assert receipt.ledger.admitted == 8
    assert receipt.ledger.kept == (0 if all_dropped else 4)
    assert receipt.ledger.dropped == (8 if all_dropped else 4)
    assert len(receipt.refusals) == receipt.ledger.dropped


def shared_trace_key_fixture(monkeypatch):
    # Deltas 0, .10, .20: only the first conflicts with the higher-score
    # third alternative. The first and second share a score/trace key.
    readings, _ = fixture(alternatives=((1., 3., 1.), (2., 3.1, .9), (3., 3.2, .8)))
    signatures = tuple(signature(readings[0], b=b) for b in (1., 2., 3.)) + (signature(readings[1]),)
    def pair(left, right, **kwargs):
        found = controlled_pair(left, right, **kwargs)
        score = 1. if left.blocks_per_period == 3. else .9
        trace = kwargs.get("diagnostic_sink")
        if found is not None:
            found = replace(found, score=score)
            if trace is not None:
                trace.pairs[-1] = replace(trace.pairs[-1], score=score)
        return found
    monkeypatch.setattr(pg, "match_plane_pair", pair)
    return readings, signatures


def rebind_receipt_records(receipt, *, payload=None, evidence=None):
    # Recompute outer integrity so regression controls exercise semantic replay
    # checks even when the changed record is internally digest-consistent.
    input_record = receipt.input_record if payload is None else seam._json(payload)
    evidence_record = receipt.evidence_record if evidence is None else seam._json(evidence)
    input_pin = hashlib.sha256(input_record).hexdigest()
    return replace(receipt, input_record=input_record, evidence_record=evidence_record,
        input_sha256=input_pin, receipt_sha256=seam._digest([
            seam.SCHEMA, seam.DOMAIN, seam.CAP_VERSION, input_pin,
            hashlib.sha256(evidence_record).hexdigest(), receipt.admitted_sha256, receipt.kept_sha256]))


def test_shared_trace_key_mixed_dispositions_seals_and_replays(monkeypatch):
    inputs = shared_trace_key_fixture(monkeypatch)
    first = complete(collect(*inputs))
    second = complete(collect(*inputs))
    assert first.receipt_sha256 == second.receipt_sha256
    assert (first.ledger.admitted, first.ledger.kept, first.ledger.dropped) == (12, 8, 4)
    assert tuple((row.left_b, disposition) for row, disposition in
                 zip(first.candidates, first.dispositions)) == (
        ((1., pg.SAME_AXIS_DISPOSITION_DROPPED),) * 4
        + ((2., pg.SAME_AXIS_DISPOSITION_KEPT),) * 4
        + ((3., pg.SAME_AXIS_DISPOSITION_KEPT),) * 4)
    rows = json.loads(first.evidence_record)["trace"]["pairs"]
    assert all(row["selected"] is True for row in rows)
    assert [(row["quadrant"], row["score"]) for row in rows[:4]] == [
        (row["quadrant"], row["score"]) for row in rows[4:8]]
    assert tuple(row.left_b for row in first.returned_matches) == (2.,) * 4 + (3.,) * 4


@pytest.mark.parametrize("stage", ["collection", "replay"])
def test_shared_trace_key_mutated_selected_bit_refuses(monkeypatch, stage):
    inputs = shared_trace_key_fixture(monkeypatch)
    if stage == "replay":
        receipt = complete(collect(*inputs))
        evidence = json.loads(receipt.evidence_record)
        evidence["trace"]["pairs"][0]["selected"] = False
        atomic(seam.validate_receipt(rebind_receipt_records(receipt, evidence=evidence)),
               "admitted_trace_mismatch")
    else:
        matcher = pg.match_plane_graph
        def mutate(*args, **kwargs):
            returned = matcher(*args, **kwargs)
            trace = kwargs["diagnostic_sink"]
            trace.pairs[0] = replace(trace.pairs[0], selected=False)
            return returned
        monkeypatch.setattr(pg, "match_plane_graph", mutate)
        atomic(collect(*inputs), "admitted_trace_mismatch")


def mutate_call(monkeypatch, mutation):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    matcher = pg.match_plane_graph
    def wrap(*args, **kwargs):
        returned = matcher(*args, **kwargs)
        return mutation(args, kwargs, returned)
    monkeypatch.setattr(pg, "match_plane_graph", wrap)


@pytest.mark.parametrize("fault", ["missing_observation", "duplicate_observation", "missing_return",
    "extra_return", "return_permutation", "cloned_return", "missing_candidate", "duplicate_identity",
    "extra_disposition", "altered_disposition", "missing_trace", "extra_trace", "altered_trace",
    "altered_count", "extra_refusal", "exception_after_append", "cancel_after_append"])
def test_lifecycle_faults_atomically_refuse(monkeypatch, fault):
    def mutation(args, kwargs, returned):
        sink, trace = kwargs["candidate_observation_sink"], kwargs["diagnostic_sink"]
        observation = sink[0]
        if fault == "missing_observation":
            sink.clear()
        elif fault == "duplicate_observation":
            sink.append(observation)
        elif fault == "missing_return":
            return returned[:-1]
        elif fault == "extra_return":
            return returned + returned[:1]
        elif fault == "return_permutation":
            return tuple(reversed(returned))
        elif fault == "cloned_return":
            return tuple(replace(row) for row in returned)
        elif fault == "missing_candidate":
            sink[0] = pg.SameAxisCandidateObservation(observation.candidates[:-1], observation.dispositions[:-1])
        elif fault == "duplicate_identity":
            duplicated = observation.candidates[:1] + observation.candidates[:-1]
            sink[0] = pg.SameAxisCandidateObservation(duplicated, observation.dispositions)
            return duplicated
        elif fault == "extra_disposition":
            object.__setattr__(observation, "dispositions", observation.dispositions + ("kept",))
        elif fault == "altered_disposition":
            sink[0] = pg.SameAxisCandidateObservation(observation.candidates, ("same_axis_mismatch",) * 4)
        elif fault == "missing_trace":
            trace.pairs.pop()
        elif fault == "extra_trace":
            trace.pairs.append(trace.pairs[0])
        elif fault == "altered_trace":
            trace.pairs[0] = replace(trace.pairs[0], score=.1)
        elif fault == "altered_count":
            trace.post_admission_candidates -= 1
        elif fault == "extra_refusal":
            kwargs["refusals"].append(pg.PlaneMatchRefusal("a", "b", 1, 1, 0, "no_signal"))
        elif fault == "exception_after_append":
            raise RuntimeError("after observation publication")
        elif fault == "cancel_after_append":
            raise KeyboardInterrupt()
        return returned
    mutate_call(monkeypatch, mutation)
    atomic(collect(*fixture()))


def test_duplicate_signature_occurrences_refuse_typed_identity(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    readings, signatures = fixture()
    result = collect(readings, (signatures[0], signatures[0], signatures[1]))
    atomic(result, "duplicate_candidate_identity")


def test_sink_replay_mixed_call_rows_refuse(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    inputs = fixture()
    old = complete(collect(*inputs))
    def replay(args, kwargs, returned):
        kwargs["candidate_observation_sink"][0] = pg.SameAxisCandidateObservation(old.candidates, old.dispositions)
        return returned
    mutate_call(monkeypatch, replay)
    atomic(collect(*inputs), "ordered_return_mismatch")


@pytest.mark.parametrize("field", ["side", "confidence", "alternatives", "signature", "order", "settings"])
def test_complete_input_pins_include_unused_readings_and_order(monkeypatch, field):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    readings, signatures = fixture()
    unused = reading("a", axis=0)
    readings = readings + (unused,)
    pin = complete(collect(readings, signatures)).input_sha256
    if field == "side":
        readings = readings[:2] + (replace(unused, side=-1.),)
    elif field == "confidence":
        readings = readings[:2] + (replace(unused, confidence=.4),)
    elif field == "alternatives":
        readings = readings[:2] + (replace(unused, alternatives=(lp.DistanceAlternative(2., 6., .8),)),)
    elif field == "signature":
        signatures[0].interior[0, 0] = .5
    elif field == "order":
        readings = tuple(reversed(readings))
    kwargs = {"min_resultant": .91} if field == "settings" else {}
    calls = []
    monkeypatch.setattr(pg, "match_plane_graph", lambda *a, **k: calls.append(1))
    atomic(collect(readings, signatures, expected_input_sha256=pin, **kwargs), "stale_input_pin")
    assert calls == []


def test_input_mutation_during_call_refuses(monkeypatch):
    readings, signatures = fixture()
    def mutation(args, kwargs, returned):
        signatures[0].cells[0, 0] += 1.
        return returned
    mutate_call(monkeypatch, mutation)
    atomic(collect(readings, signatures), "input_changed_during_call")


@pytest.mark.parametrize("fault", ["candidate", "disposition", "ledger", "input", "evidence",
    "digest", "authority", "reservation", "missing_rows", "extra_rows"])
def test_receipt_validator_rejects_altered_or_missing_evidence(monkeypatch, fault):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    receipt = complete(collect(*fixture()))
    if fault == "candidate":
        receipt = replace(receipt, candidates=(replace(receipt.candidates[0], peak=.1),) + receipt.candidates[1:])
    elif fault == "disposition":
        receipt = replace(receipt, dispositions=("same_axis_mismatch",) * 4)
    elif fault == "ledger":
        receipt = replace(receipt, ledger=replace(receipt.ledger, admitted=3))
    elif fault == "input":
        record = json.loads(receipt.input_record)
        record["readings"][0]["reading"]["confidence"] = .2
        receipt = replace(receipt, input_record=seam._json(record))
    elif fault == "evidence":
        record = json.loads(receipt.evidence_record)
        record["trace"]["pairs"][0]["score"] = .2
        receipt = replace(receipt, evidence_record=seam._json(record))
    elif fault == "digest":
        receipt = replace(receipt, receipt_sha256="0" * 64)
    elif fault == "authority":
        receipt = replace(receipt, reconstruction_authorized=True)
    elif fault == "reservation":
        receipt = replace(receipt, reservation=replace(receipt.reservation, attempts=3))
    elif fault == "missing_rows":
        receipt = replace(receipt, returned_matches=receipt.returned_matches[:-1])
    elif fault == "extra_rows":
        receipt = replace(receipt, returned_matches=receipt.returned_matches + receipt.returned_matches[:1])
    atomic(seam.validate_receipt(receipt))


def forbid_allocation_and_matcher(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must refuse before matcher/trace/payload construction")
    monkeypatch.setattr(pg, "PlaneGraphTrace", forbidden)
    monkeypatch.setattr(pg, "match_plane_graph", forbidden)
    monkeypatch.setattr(seam, "_input_payload", forbidden)


@pytest.mark.parametrize("cap,limit,reason", [
    ("readings", 1, "readings_cap"), ("frames", 1, "frames_cap"),
    ("alternatives_per_reading", 1, "alternatives_per_reading_cap"),
    ("hypotheses", 3, "hypotheses_cap"), ("attempts", 3, "attempts_cap"),
    ("candidate_identities", 3, "potential_candidate_identities_cap"),
    ("input_evidence_bytes", 4096, "input_evidence_bytes_cap")])
def test_atomic_cap_refusal_before_allocations(monkeypatch, cap, limit, reason):
    inputs = fixture(alternatives=((1., 3., 1.), (.5, 1.5, .8))) if cap == "alternatives_per_reading" else fixture()
    forbid_allocation_and_matcher(monkeypatch)
    atomic(collect(*inputs, caps=replace(seam.HARD_CAPS, **{cap: limit})), reason)


@pytest.mark.parametrize("count,reason", [(33, "potential_candidate_identities_cap"),
                                         (65, "hypotheses_cap")])
def test_hard_candidate_and_hypothesis_caps_preflight(monkeypatch, count, reason):
    readings, signatures = fixture()
    signatures = (signatures[0],) * count + (signatures[1],) * count
    forbid_allocation_and_matcher(monkeypatch)
    atomic(collect(readings, signatures), reason)


def test_full_reservation_exact_boundary_and_minus_one(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    inputs = fixture()
    planned = complete(collect(*inputs)).reservation.input_evidence_bytes
    exact = complete(collect(*inputs, caps=replace(seam.HARD_CAPS, input_evidence_bytes=planned)))
    assert exact.reservation.input_evidence_bytes == planned
    forbid_allocation_and_matcher(monkeypatch)
    atomic(collect(*inputs, caps=replace(seam.HARD_CAPS, input_evidence_bytes=planned - 1)),
           "input_evidence_bytes_cap")


def test_workspace_exact_boundary_and_independent_caps(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    inputs = fixture()
    planned = complete(collect(*inputs)).reservation
    caps = replace(seam.HARD_CAPS,
        correlation_workspace_bytes=planned.correlation_workspace_bytes,
        input_evidence_bytes=planned.input_evidence_bytes)
    exact = complete(collect(*inputs, caps=caps))
    assert exact.reservation == planned
    forbid_allocation_and_matcher(monkeypatch)
    atomic(collect(*inputs, caps=replace(caps,
        correlation_workspace_bytes=planned.correlation_workspace_bytes - 1)),
        "correlation_workspace_bytes_cap")
    atomic(collect(*inputs, caps=replace(caps,
        input_evidence_bytes=planned.input_evidence_bytes - 1)), "input_evidence_bytes_cap")


@pytest.mark.parametrize("shape,shift", [((3, 7), 0), ((7, 3), 2), ((9, 5), 1)])
def test_rectangular_odd_workspace_dimensions(shape, shift):
    rows, columns = shape
    n = rows * columns
    p = (rows + 4 * shift) * (columns + 4 * shift)
    f = (rows + 4 * shift) * ((columns + 4 * shift) // 2 + 1)
    t = (2 * shift + 1) ** 2
    assert seam._pair_workspace(shape, 8, shift) == 300 * n + 29 * p + 145 * f + 43 * t + 576


def test_small_shift_32_calls_have_separate_workspace(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    readings, _ = fixture()
    signatures = tuple(signature(row, shape=(64, 64)) for row in readings)
    receipt = complete(seam.collect_matcher_call(readings, signatures, max_shift_cells=32))
    assert 192 * 192 * 512 > seam.INPUT_EVIDENCE_BYTES
    assert 0 < receipt.reservation.input_evidence_bytes < seam.INPUT_EVIDENCE_BYTES
    assert receipt.reservation.correlation_workspace_bytes == seam._pair_workspace((64, 64), 8, 32)
    assert receipt.reservation.correlation_workspace_bytes < seam.CORRELATION_WORKSPACE_BYTES


def test_retained_256_signature_input_evidence_refusal(monkeypatch):
    readings, _ = fixture()
    signatures = tuple(signature(row, shape=(256, 256)) for row in readings)
    forbid_allocation_and_matcher(monkeypatch)
    atomic(seam.collect_matcher_call(readings, signatures), "input_evidence_bytes_cap")


@pytest.mark.parametrize("shift,reason", [(seam.sys.maxsize, "correlation_workspace_overflow"),
    (10 ** 100, "correlation_workspace_overflow"), (1000, "correlation_workspace_bytes_cap")])
def test_shift_overflow_or_workspace_cap_before_allocation(monkeypatch, shift, reason):
    inputs = fixture()
    forbid_allocation_and_matcher(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("dimension validation must not convert NumPy inputs")
    monkeypatch.setattr(seam.np, "asarray", forbidden)
    atomic(seam.collect_matcher_call(*inputs, max_shift_cells=shift), reason)


@pytest.mark.parametrize("shape,shift", [((seam.sys.maxsize, 2), 0),
    ((seam.sys.maxsize, 1), 1), ((1, seam.sys.maxsize), 0), ((1, 1), seam.sys.maxsize // 4)])
def test_dimension_and_reservation_arithmetic_overflow_without_arrays(shape, shift):
    with pytest.raises(seam._Refusal, match="^correlation_workspace_overflow$"):
        seam._pair_workspace(shape, 8, shift)


def test_shift_zero_reservation_and_replay(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    receipt = complete(seam.collect_matcher_call(*fixture(), max_shift_cells=0))
    assert receipt.reservation.correlation_workspace_bytes == seam._pair_workspace((12, 12), 8, 0)


@pytest.mark.parametrize("kind", ["one_frame", "fully_skipped", "axis_mismatch", "invalid_distance"])
def test_proven_no_correlation_reserves_zero_even_with_huge_shift(kind):
    readings, signatures = fixture(share=0. if kind == "fully_skipped" else .6,
                                   right_axis=0 if kind == "axis_mismatch" else 1)
    if kind == "one_frame":
        readings, signatures = readings[:1], signatures[:1]
    elif kind == "invalid_distance":
        signatures = (signature(readings[0], b=2.), signatures[1])
    receipt = complete(seam.collect_matcher_call(readings, signatures,
        max_shift_cells=seam.sys.maxsize, caps=replace(seam.HARD_CAPS, correlation_workspace_bytes=1)))
    assert receipt.reservation.correlation_workspace_bytes == 0


def test_below_admission_share_still_reserves_correlation_workspace():
    readings, signatures = fixture()
    readings = (replace(readings[0], structural_share=0.), readings[1])
    signatures = (signature(readings[0]), signatures[1])
    receipt = complete(collect(readings, signatures))
    assert receipt.reservation.potential_candidates == 0
    assert receipt.reservation.correlation_workspace_bytes > 0


def test_workspace_uses_sequential_target_peak_and_source_dtype(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    readings, _ = fixture()
    signatures = (signature(readings[0], shape=(40, 40)),
                  signature(readings[1], shape=(9, 5)))
    # A larger source grid is indexed into the right target grid, never padded.
    receipt = complete(collect(readings, signatures))
    assert receipt.reservation.correlation_workspace_bytes == seam._pair_workspace((9, 5), 8, 4)
    wide = replace(signatures[0], cells=signatures[0].cells.astype(np.longdouble))
    receipt = complete(collect(readings, (wide, signatures[1])))
    assert receipt.reservation.correlation_workspace_bytes == seam._pair_workspace(
        (9, 5), max(8, np.dtype(np.longdouble).itemsize), 4)


def test_active_capture_refuses_before_preflight_or_allocation(monkeypatch):
    inputs = fixture()
    forbid_allocation_and_matcher(monkeypatch)
    monkeypatch.setattr(lp.lattice, "_period_support_state", lambda: SimpleNamespace(context={}))
    def forbidden(*args, **kwargs):
        raise AssertionError("capture refusal must precede preflight")
    monkeypatch.setattr(seam, "_preflight", forbidden)
    atomic(collect(*inputs), "active_period_support_capture")


@pytest.mark.parametrize("fault", ["cap", "reservation", "rebound_cap", "rebound_reservation", "shift"])
def test_workspace_cap_and_reservation_receipt_tampering(monkeypatch, fault):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    receipt = complete(collect(*fixture()))
    payload = json.loads(receipt.input_record)
    if fault in ("cap", "rebound_cap"):
        payload["caps"]["correlation_workspace_bytes"] = receipt.reservation.correlation_workspace_bytes - 1
    elif fault in ("reservation", "rebound_reservation"):
        receipt = replace(receipt, reservation=replace(receipt.reservation,
            correlation_workspace_bytes=receipt.reservation.correlation_workspace_bytes - 1))
        if fault == "rebound_reservation":
            payload["reservation"]["correlation_workspace_bytes"] -= 1
    else:
        payload["settings"]["max_shift_cells"] += 1
    if fault.startswith("rebound") or fault == "shift":
        receipt = rebind_receipt_records(receipt, payload=payload)
    elif fault == "cap":
        receipt = replace(receipt, input_record=seam._json(payload))
    atomic(seam.validate_receipt(receipt))


def test_workspace_policy_ceiling_and_old_receipt_version(monkeypatch):
    assert seam.HARD_CAPS.correlation_workspace_bytes == 32 * 1024 * 1024
    forbid_allocation_and_matcher(monkeypatch)
    atomic(collect(*fixture(), caps=replace(seam.HARD_CAPS,
        correlation_workspace_bytes=seam.CORRELATION_WORKSPACE_BYTES + 1)), "invalid_caps")
    monkeypatch.undo()
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    receipt = complete(collect(*fixture()))
    atomic(seam.validate_receipt(replace(receipt,
        cap_version="dscale-04-conservative-production-seam-v1")), "invalid_receipt")


def test_caps_match_existing_dscale_definition():
    from mc_photogrammetry.reconstruction._same_axis_constraint_search import ConstraintCaps
    assert seam.HARD_CAPS.candidate_identities == ConstraintCaps().candidate_identities
    assert seam.INPUT_EVIDENCE_BYTES == ConstraintCaps().canonical_input_bytes


def test_invalid_source_pin_refuses_before_matcher(monkeypatch):
    monkeypatch.setattr(seam, "SOURCE_PINS", (("src/mc_photogrammetry/reconstruction/plane_graph.py", "0" * 64),))
    calls = []
    monkeypatch.setattr(pg, "match_plane_graph", lambda *a, **k: calls.append(1))
    atomic(collect(*fixture()), "source_pin_mismatch")
    assert calls == []


@pytest.mark.parametrize("relative,pin", seam.SOURCE_PINS)
@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_receipt_replay_requires_complete_exact_dependency_manifest(monkeypatch, relative, pin, fault):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    receipt = complete(collect(*fixture()))
    payload = json.loads(receipt.input_record)
    if fault == "missing":
        payload["sources"] = [row for row in payload["sources"] if row[0] != relative]
    else:
        for row in payload["sources"]:
            if row[0] == relative:
                row[1] = "0" * 64
    atomic(seam.validate_receipt(rebind_receipt_records(receipt, payload=payload)),
           "source_manifest_mismatch")


@pytest.mark.parametrize("fault", ["missing_manifest", "adapter_revision"])
def test_receipt_replay_rejects_missing_or_incompatible_producer_manifest(monkeypatch, fault):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    receipt = complete(collect(*fixture()))
    if fault == "missing_manifest":
        payload = json.loads(receipt.input_record)
        del payload["sources"]
        receipt = rebind_receipt_records(receipt, payload=payload)
    else:
        current = seam._source_manifest()
        monkeypatch.setattr(seam, "_source_manifest", lambda: current[:-1] + (("adapter", "0" * 64),))
    atomic(seam.validate_receipt(receipt), "source_manifest_mismatch")


@pytest.mark.parametrize("relative,pin", seam.SOURCE_PINS)
def test_receipt_replay_rejects_checked_dependency_revision(monkeypatch, relative, pin):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    receipt = complete(collect(*fixture()))
    target = (seam.Path(seam.__file__).resolve().parents[3] / relative).resolve()
    read_bytes = seam.Path.read_bytes
    changed = read_bytes(target) + b"\n# synthetic source revision\n"
    monkeypatch.setattr(seam.Path, "read_bytes",
        lambda path: changed if path.resolve() == target else read_bytes(path))
    # A real file revision first invalidates its frozen pin.
    atomic(seam.validate_receipt(receipt), "source_pin_mismatch")
    # Even an updated, internally checked producer manifest cannot replay an
    # old receipt under new semantics.
    revised_pin = hashlib.sha256(changed).hexdigest()
    monkeypatch.setattr(seam, "SOURCE_PINS", tuple(
        (name, revised_pin if name == relative else value) for name, value in seam.SOURCE_PINS))
    atomic(seam.validate_receipt(receipt), "source_manifest_mismatch")


def test_missing_dependency_file_atomically_refuses_replay(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    receipt = complete(collect(*fixture()))
    target = (seam.Path(seam.__file__).resolve().parents[3] / seam.SOURCE_PINS[4][0]).resolve()
    read_bytes = seam.Path.read_bytes
    def missing(path):
        if path.resolve() == target:
            raise FileNotFoundError("synthetic missing source dependency")
        return read_bytes(path)
    monkeypatch.setattr(seam.Path, "read_bytes", missing)
    atomic(seam.validate_receipt(receipt), "invalid_or_incomplete_evidence")


def test_upstream_unavailable_refuses_before_trace(monkeypatch):
    # A typed already-validated context may never turn an upstream refusal into
    # a completed producer call. No upstream producer or payload is evaluated.
    def raw(kind, **values):
        result = object.__new__(kind)
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result
    context = seam.AcceptedCallContext(raw(seam.session_module.ImageDerivedSession),
        raw(seam.lattice_module.ImageDerivedLatticeResult, status="unavailable"),
        raw(seam.signature_module.ImageDerivedSignaturesResult, status="assembled"),
        raw(seam.connections.ImageDerivedConnectionGraph, status="ok"),
        *("0" * 64,) * 4)
    forbid_allocation_and_matcher(monkeypatch)
    atomic(collect(*fixture(), context=context), "upstream_refusal")


def synthetic_context(readings, signatures, *, relative=None, solved_pairs=None):
    """Pinned context metadata fixture, never real accepted geometry evidence.

    Session/graph/signature records use their normal constructors without any
    pixels. Lattice metadata is explicitly fabricated for this local integrity
    control; its numerical provenance producer is outside this test's scope.
    """
    from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraIntrinsics, CameraPose
    from test_same_axis_constraint_search import graph as graph_fixture
    sm, lm, sigm = seam.session_module, seam.lattice_module, seam.signature_module
    frame_ids = ("a", "b", "c")
    estimates = {name: CameraEstimate(CameraIntrinsics((80, 60), (100., 100.), (40., 30.)),
        CameraPose(None, 14., 3., 0.), .4, 0.) for name in frame_ids}
    unresolved = tuple(f"{name}: {fact}" for name in frame_ids for fact in (
        "position (vanishing points fix no translation)", "scale (needs the lattice period)",
        "yaw quadrant (recovered modulo 90 degrees)"))
    session = sm.ImageDerivedSession(sm.IMAGE_DERIVED_SESSION_SCHEMA, 1,
        tuple(sm.ImageDerivedFrame(name, None, None, (60, 80), "1" * 64) for name in frame_ids),
        estimates, (frame_ids,), unresolved, False,
        {name: sm._adapt(estimate, name) for name, estimate in estimates.items()})
    graph = graph_fixture(frame_ids, relative=relative, solved_pairs=solved_pairs)
    graph = replace(graph, session_sha256=session.session_sha256,
        evidence=tuple(replace(row, session_sha256=session.session_sha256, evidence_sha256=None)
                       for row in graph.evidence), graph_sha256=None)
    lattice = object.__new__(lm.ImageDerivedLatticeResult)
    metadata = dict(schema=lm.IMAGE_DERIVED_LATTICE_MEASURE_SCHEMA,
        producer=lm.IMAGE_DERIVED_LATTICE_MEASURE_PRODUCER, terminal=lm.GEOMETRY_UNAVAILABLE,
        status="assembled", reason="", detail="", session_sha256=session.session_sha256,
        pose_graph_sha256="3" * 64, graph_sha256=graph.graph_sha256, frame_ids=frame_ids,
        camera_calibrations=(), readings=readings,
        bridge=SimpleNamespace(reading_universe_sha256="4" * 64, bridge_sha256="5" * 64),
        abstentions=())
    for name, value in metadata.items():
        object.__setattr__(lattice, name, value)
    object.__setattr__(lattice, "result_sha256", seam._digest(lm._result_payload(lattice)))
    evidence = []
    for index, (source, row) in enumerate(zip(readings, signatures)):
        for array in (row.cells, row.interior, row.coverage):
            array.flags.writeable = False
        evidence.append(sigm.SignatureEvidence(source.frame, index, source.axis, 0, 1., row))
    signature_result = sigm.ImageDerivedSignaturesResult(sigm.IMAGE_DERIVED_SIGNATURES_SCHEMA,
        sigm.IMAGE_DERIVED_SIGNATURES_PRODUCER, sigm.SIGNATURES_TERMINAL, "assembled", "", "",
        session.session_sha256, lattice.result_sha256, frame_ids,
        tuple(sigm.UsableMask(name, 1, 1, b"\x01") for name in frame_ids), tuple(evidence), ())
    return seam.AcceptedCallContext(session, lattice, signature_result, graph,
        session.session_sha256, lattice.result_sha256, signature_result.result_sha256, graph.graph_sha256)


def test_full_context_pins_graph_order_and_omitted_frame_projection(monkeypatch):
    monkeypatch.setattr(pg, "match_plane_pair", controlled_pair)
    readings, signatures = fixture()
    context = synthetic_context(readings, signatures)
    receipt = complete(collect(readings, signatures, context=context))
    record = json.loads(receipt.input_record)["accepted_context"]
    assert record["pins"] == [context.session_sha256, context.lattice_sha256,
                              context.signatures_sha256, context.graph_sha256]
    assert record["omitted_frame_ids"] == ["c"]
    assert record["projected_relations"] == [["a", "b", 0]]
    assert record["connection_graph"]["frame_ids"] == ["a", "b", "c"]


@pytest.mark.parametrize("pin", ["session_sha256", "lattice_sha256", "signatures_sha256", "graph_sha256"])
def test_stale_upstream_pins_refuse_before_matcher(monkeypatch, pin):
    readings, signatures = fixture()
    context = replace(synthetic_context(readings, signatures), **{pin: "0" * 64})
    calls = []
    monkeypatch.setattr(pg, "match_plane_graph", lambda *a, **k: calls.append(1))
    atomic(collect(readings, signatures, context=context))
    assert calls == []


def test_omitted_frame_contradictory_cycle_refuses_before_matcher(monkeypatch):
    readings, signatures = fixture()
    context = synthetic_context(readings, signatures, relative={("b", "c"): 1})
    calls = []
    monkeypatch.setattr(pg, "match_plane_graph", lambda *a, **k: calls.append(1))
    atomic(collect(readings, signatures, context=context), "invalid_graph_relations")
    assert calls == []
