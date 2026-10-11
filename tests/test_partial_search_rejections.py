"""Offline source-bound controls; observer never repeats selector/physical work."""
from dataclasses import replace
import json

import pytest

from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp, plane_graph
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from test_partial_core_image_diagnostic import worker
from test_partial_axis_observability import certificate
from test_plane_graph import make_reading, make_match


def source_case(kind, alternatives=2):
    frames = ("a", "b", "c")
    readings, stable, main, extra = [], {}, {}, {}
    for frame in frames:
        for axis in (0, 2):
            reading = make_reading(frame, axis, 1, [(1., 8., 1.)])
            stable[frame, axis] = reading
            readings.append(reading)
        if kind == "cycle":
            distances = {"a": (8., 30.), "b": (8., 20.), "c": (20., 29.)}[frame]
            main[frame] = make_reading(frame, 1, 1, [(1., distances[0], 1.)])
            extra[frame] = make_reading(frame, 1, 1, [(99., distances[1], 1.)])
        else:
            main[frame] = make_reading(frame, 1, 1,
                [(float(i+1), float(8*(i+1) + (2 if kind == "diameter" and frame == "c" else 0)), 1.)
                 for i in range(alternatives)])
            extra[frame] = make_reading(frame, 1, 1, [(99., 21. if frame == "c" else 20., 1.)])
        readings.extend((main[frame], extra[frame]))
    readings = tuple(readings)
    _, ids, _ = plane_graph._canonical_reading_index(readings)
    def edge(u, v, axis, left, right, a=1., b=1.):
        return make_match(u, v, axis, axis, 0, ids[id(left)][1], ids[id(right)][1], left_b=a, right_b=b)
    rows = [edge("a", f, axis, stable["a", axis], stable[f, axis])
            for f in frames[1:] for axis in (0, 2)]
    if kind == "cycle":
        rows += [edge("a", "b", 1, main["a"], main["b"]),
                 edge("b", "c", 1, extra["b"], main["c"], 99., 1.),
                 edge("a", "c", 1, extra["a"], extra["c"], 99., 99.)]
    else:
        for u, v in (("a", "b"), ("b", "c"), ("a", "c")):
            pairs = ((0, 1), (1, 0)) if kind == "relational" and (u,v) == ("a","c") else [(i,i) for i in range(alternatives)]
            rows += [edge(u, v, 1, main[u], main[v], float(a+1), float(b+1)) for a,b in pairs]
            rows.append(edge(u, v, 1, extra[u], extra[v], 99., 99.))
    return readings, tuple(rows), frames, (("a", "b", 0), ("a", "c", 0))


@pytest.mark.parametrize("kind", ["relational", "diameter", "cycle"])
def test_source_conflict_census_distinguishes_failure_without_repeated_work(kind, monkeypatch):
    data = source_case(kind)
    cert, evidence = certificate(data)
    assert cert.status == "no"
    selection.validate_candidate_certificate(cert, data[0], evidence, data[1])
    before, digest = cert.proof_json, cert.result_sha256
    calls = []
    original = type(cert).proof_record
    monkeypatch.setattr(type(cert), "proof_record", lambda self: calls.append(self) or original(self))
    for owner, name in ((csp, "solve_endpoint_hypotheses"), (csp, "_physical_feasibility"),
                        (plane_graph, "signed_physical_relation"), (selection, "validate_candidate_certificate")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("observer repeated source search/evaluation"))
    result = worker._endpoint_search(cert, cert.budget)
    assert calls == [cert] and (cert.proof_json, cert.result_sha256) == (before, digest)
    rejection = result["rejections"]
    assert result["status"] == rejection["status"] == "complete"
    assert rejection["inconsistent_nodes"] > 0 if kind != "cycle" else rejection["physical_rejected"] == 1
    if kind == "relational":
        assert rejection["partial_prunes"] == rejection["physical_rejected"] == 0
        assert rejection["forced_witnesses"]["rows"] == []
    elif kind == "diameter":
        assert rejection["partial_prunes"] == 2 and rejection["physical_rejected"] == 0
        rows = rejection["forced_witnesses"]["rows"]
        assert len(rows) == 2
        for row in rows:
            assert row["extrema"][1] - row["extrema"][0] > .15
            assert set(row["extrema_source_rows"]) == {f["source_row"] for f in row["factors"]}
    else:
        assert rejection["partial_prunes"] == 0
        assert rejection["physical_reasons"] == {"physical_frame_cycle": 1}
    assert worker._encoded_size(rejection) <= worker.REJECTION_BYTES
    assert "masks" not in json.dumps(rejection) and "choices" not in json.dumps(rejection)


def test_actual_source_witness_capping_and_work_limits():
    cert, _ = certificate(source_case("diameter", alternatives=8))
    result = worker._endpoint_search(cert, cert.budget)["rejections"]
    assert result["status"] == "complete" and result["partial_prunes"] == 8
    witness = result["forced_witnesses"]
    assert witness["status"] == "capped"
    assert witness["source_retained"] == 8 and witness["source_omitted"] == 0
    assert witness["observer_retained"] == witness["observer_omitted"] == 4
    assert witness["certificate_sha256"] == cert.result_sha256
    assert witness["source_sha256"] == cert.proof_record()["source_digest"]
    assert witness["factor_visits"] <= worker.WITNESS_FACTOR_VISITS
    assert witness["source_row_visits"] <= worker.WITNESS_SOURCE_VISITS
    assert all(worker._encoded_size(row) <= worker.WITNESS_ROW_BYTES for row in witness["rows"])


@pytest.mark.parametrize("limit", ["WITNESS_FACTOR_VISITS", "WITNESS_SOURCE_VISITS", "WITNESS_ROW_BYTES"])
def test_witness_resource_refusal_is_explicit_omission_and_does_not_change_certificate(limit, monkeypatch):
    cert, _ = certificate(source_case("diameter"))
    before = cert.proof_json
    monkeypatch.setattr(worker, limit, 0 if limit != "WITNESS_ROW_BYTES" else 1)
    result = worker._endpoint_search(cert, cert.budget)["rejections"]
    assert result["status"] == "complete"
    assert result["forced_witnesses"]["observer_retained"] == 0
    assert result["forced_witnesses"]["observer_omitted"] == 2
    assert result["forced_witnesses"]["status"] == "capped" and cert.proof_json == before


@pytest.mark.parametrize("change", ["bool", "omission", "pruning", "reasons", "join", "row_bool", "axis_bool",
                                   "source_digest", "physical_digest", "policy", "tolerance", "physical_census"])
def test_malformed_rejection_evidence_fails_closed(change):
    cert, _ = certificate(source_case("diameter"))
    proof = cert.proof_record()
    if change == "bool": proof["counters"]["partial_prunes"] = True
    elif change == "omission": proof["counters"]["omitted_partial_exclusions"] += 1
    elif change == "pruning": proof["pruning"]["inconsistent_nodes"] = True
    elif change == "reasons": proof["physical_reasons"] = {"private payload": 1}
    elif change == "join": proof["partial_exclusions"][0][-1][-1][0] = 0
    elif change == "row_bool": proof["partial_exclusions"][0][-1][-1][0] = True
    elif change == "axis_bool": proof["partial_exclusions"][0][-1][0][2] = True
    elif change == "source_digest": proof["source_digest"] = "private payload"
    elif change == "physical_digest": proof["physical_digest"] = "private payload"
    elif change == "policy": proof["transcript_policy"] = "private payload"
    elif change == "tolerance": proof["physical_tolerance"] = True
    else: proof["counters"]["retained_physical_exclusions"] = 1
    result = worker._endpoint_rejections(cert, proof, cert.budget)
    assert result == {"status": "malformed", "reason": "invalid_rejection_evidence"}
    assert "private" not in json.dumps(result)


def test_bad_rejection_census_marks_whole_observation_malformed(monkeypatch):
    cert, _ = certificate(source_case("diameter"))
    proof = cert.proof_record()
    proof["counters"]["omitted_partial_exclusions"] += 1
    monkeypatch.setattr(type(cert), "proof_record", lambda self: proof)
    result = worker._endpoint_search(cert, cert.budget)
    assert result["status"] == "malformed" and result["rejections"]["status"] == "malformed"


def test_resealed_source_witness_tamper_still_fails_existing_certificate_replay():
    data = source_case("diameter")
    cert, evidence = certificate(data)
    proof = cert.proof_record()
    proof["partial_exclusions"][0][-1][1] -= .01
    proof["partial_exclusions"][0][-1][3][4] -= .01
    tampered = replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
    with pytest.raises(ValueError):
        selection.validate_candidate_certificate(tampered, data[0], evidence, data[1])
