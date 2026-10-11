"""Synthetic Phase B certificates; hashes alone never prove uniqueness."""
import dataclasses
import json

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_signatures as f1
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph
from test_endpoint_hypothesis import connected_fixture


def _forge(value, **changes):
    result = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(result, field.name, changes.get(field.name, getattr(value, field.name)))
    return result


def _evidence(readings, *, alternative_specific=False):
    evidence = []
    for index, reading in enumerate(readings):
        for a_index, alternative in enumerate(reading.alternatives):
            seed = 100 + reading.axis + (10_000 * a_index if alternative_specific else 0)
            cells = np.random.default_rng(seed).normal(size=(11, 9)).astype(np.float32)
            signature = lattice_planes.PlaneSignature(reading, alternative.blocks_per_period,
                f1._readonly_array(cells), f1._readonly_array(np.zeros_like(cells)),
                f1._readonly_array(np.ones(cells.shape, dtype=bool)), (0., 0.))
            evidence.append(f1.SignatureEvidence(reading.frame, index, reading.axis, a_index,
                                                alternative.blocks_per_period, signature))
    return tuple(evidence)


def _fixture(rows=None, budget=csp.EndpointBudget()):
    readings, original = connected_fixture("a", "b")
    readings = tuple(readings)
    rows = tuple(original if rows is None else rows)
    evidence = _evidence(readings)
    cert = selection.select_endpoint_hypotheses(readings=readings, signatures=evidence,
                candidate_matches=rows, frame_ids=("a", "b"), quadrant_constraints=(), budget=budget,
                session_sha256="a" * 64, lattice_sha256="b" * 64, signatures_sha256="c" * 64)
    return cert, readings, evidence, rows


def _proof_change(cert, callback):
    proof = cert.proof_record()
    callback(proof)
    return dataclasses.replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")


def test_unique_certificate_is_structural_and_has_only_proposed_constraints(monkeypatch):
    monkeypatch.setattr(plane_graph, "solve_component", lambda *a, **k: pytest.fail("geometry solver"))
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", lambda *a, **k: pytest.fail("geometry evaluator"))
    cert, readings, evidence, rows = _fixture()
    assert cert.schema == "mc-photogrammetry-endpoint-selection-v4"
    assert cert.producer == "m9-g-endpoint-selection-4"
    assert cert.status == "unique" and cert.reason == "joint_geometry_not_evaluated"
    assert len(cert.representatives) == 3
    assert tuple(rep.match for rep in cert.representatives) == rows
    selection.validate_candidate_certificate(cert, readings, evidence, rows)
    assert dataclasses.replace(cert) == cert
    copy = cert.proof_record(); copy["status"] = "no"
    assert cert.proof_record()["status"] == "unique"


@pytest.mark.parametrize("state", ["no", "multiple", "unresolved_evidence", "budget_refused"])
def test_all_abstaining_states_are_atomic(state):
    _, rows = connected_fixture("a", "b")
    budget = csp.EndpointBudget()
    if state == "no":
        rows = []
    elif state == "multiple":
        rows += [dataclasses.replace(row, quadrant_of_right=2) for row in rows]
    elif state == "unresolved_evidence":
        rows.append(dataclasses.replace(rows[0], cell_shift_rc=(1, 2)))
    else:
        budget = dataclasses.replace(budget, branch_nodes=0)
    cert, readings, evidence, matches = _fixture(rows, budget)
    assert cert.status == state and cert.representatives == ()
    selection.validate_candidate_certificate(cert, readings, evidence, matches)
    if state == "unresolved_evidence":
        assert cert.proof_record()["choices"] == []
        assert cert.proof_record()["counters"]["solutions_found"] == 1


def test_noisy_b_canonicalized_with_exact_source_row_evidence():
    _, rows = connected_fixture("a", "b")
    rows = [dataclasses.replace(row, left_b=1.+5e-10, right_b=1.-5e-10) for row in rows]
    cert, readings, evidence, rows = _fixture(rows)
    assert cert.status == "unique"
    assert all(r.match.left_b == r.match.right_b == 1. for r in cert.representatives)
    assert all(rows[r.source_row].left_b != r.match.left_b for r in cert.representatives)
    selection.validate_candidate_certificate(cert, readings, evidence, rows)


def test_duplicate_rows_preserve_support_without_multiplying_disposition():
    _, rows = connected_fixture("a", "b")
    cert, readings, evidence, rows = _fixture(rows + rows)
    proof = cert.proof_record()
    assert cert.status == "unique" and proof["counters"]["solutions_found"] == 1
    assert all(len(f["source_rows"]) == 2 and len(f["distinguishable_support_groups"]) == 1
               for f in proof["choices"][0]["factors"])
    selection.validate_candidate_certificate(cert, readings, evidence, rows)


@pytest.mark.parametrize("field,value", [("schema", "mc-photogrammetry-endpoint-selection-v0"),
                                         ("producer", "m9-g-endpoint-selection-0")])
def test_resealed_historical_selection_identity_is_rejected(field, value):
    cert, _, _, _ = _fixture()
    forged = _forge(cert, **{field: value})
    forged = _forge(forged, result_sha256=selection._sha(selection._payload(forged)))
    with pytest.raises(ValueError):
        dataclasses.replace(forged)


@pytest.mark.parametrize("change", ["factor", "alternative", "index", "exhaustive", "duplicate_weight"])
def test_proof_internal_consistency_rejects_resealed_changes(change):
    cert, _, _, _ = _fixture()
    def changed(proof):
        if change == "factor": proof["choices"][0]["factors"].pop()
        elif change == "alternative": proof["choices"][0]["alternatives"][0][2] = 99
        elif change == "index": proof["choices"][0]["alternatives"][0][1] = 99
        elif change == "exhaustive": proof["exhaustive"] = False
        else: proof["counters"]["solutions_found"] = 2
    with pytest.raises(ValueError):
        _proof_change(cert, changed)


@pytest.mark.parametrize("change", ["wrong_row", "wrong_match", "order", "missing"])
def test_wrong_representative_or_order_refuses(change):
    cert, readings, evidence, rows = _fixture()
    reps = cert.representatives
    if change == "wrong_row": reps = (dataclasses.replace(reps[0], source_row=1), *reps[1:])
    elif change == "wrong_match": reps = (dataclasses.replace(reps[0], match=dataclasses.replace(reps[0].match, score=.123)), *reps[1:])
    elif change == "order": reps = tuple(reversed(reps))
    else: reps = reps[:-1]
    with pytest.raises(ValueError):
        altered = dataclasses.replace(cert, representatives=tuple(reps), result_sha256="")
        selection.validate_candidate_certificate(altered, readings, evidence, rows)


def test_missing_signature_or_candidate_cannot_reseal_source_binding():
    cert, readings, evidence, rows = _fixture()
    with pytest.raises(ValueError):
        selection.validate_candidate_certificate(cert, readings, evidence[:-1], rows)
    with pytest.raises(ValueError):
        selection.validate_candidate_certificate(cert, readings, evidence, rows[:-1])


@pytest.mark.parametrize("resource", [f.name for f in dataclasses.fields(csp.EndpointBudget)])
def test_each_exact_phase_a_budget_can_be_preserved_as_refusal(resource):
    cert, _, _, _ = _fixture(budget=dataclasses.replace(csp.EndpointBudget(), **{resource: 0}))
    assert cert.status == "budget_refused" and cert.representatives == ()
    assert cert.budget == dataclasses.replace(csp.EndpointBudget(), **{resource: 0})


def test_cap_plus_one_alternative_refuses_before_canonical_or_expected_traversal(monkeypatch):
    cert, readings, evidence, rows = _fixture()
    excessive = dataclasses.replace(readings[0], alternatives=readings[0].alternatives * 17)
    monkeypatch.setattr(plane_graph,"_canonical_reading_index",
                        lambda *a:pytest.fail("cap+1 reached canonical/key traversal"))
    with pytest.raises(ValueError,match="alternative count/type"):
        selection.reading_bindings((excessive,),())


@pytest.mark.parametrize("resource", ["scalar_payload_bytes", "relation_visits"])
def test_one_stored_budget_validation_meter_exhausts_cumulatively(resource,monkeypatch):
    cert, readings, evidence, rows = _fixture()
    cues=[]
    for row in rows:
        cues.append({name:value for name,value in csp._row_payload(row).items()
                     if name not in ("left","right","left_axis","right_axis","left_plane",
                                     "right_plane","left_b","right_b","quadrant_of_right")})
    if resource == "scalar_payload_bytes":
        cap=max(len(csp._Work(csp.EndpointBudget()).encode(cue)) for cue in cues)
        assert sum(len(csp._Work(csp.EndpointBudget()).encode(cue)) for cue in cues)>cap
    else:
        cap=16  # selected-row visits+resolution=9; axis/connectivity visits total18
    proof=cert.proof_record()
    proof["counters"][resource]=0  # hostile self-sealed accounting, not genuine search evidence
    if resource == "scalar_payload_bytes":
        for category in csp.BYTE_CATEGORIES: proof["counters"][category + "_bytes"] = 0
    else:
        for name in ("preflight", "incidence", "partial_support", "partial_update", "partial_queue",
                     "partial_journal", "partial_undo", "partial_event"):
            proof["counters"][name + "_visits"] = 0
    counterfeit=dataclasses.replace(cert,budget=dataclasses.replace(cert.budget,**{resource:cap}),
                    proof_json=selection._encode(proof).decode(),result_sha256="")
    actual=selection._ValidationWork; meters=[]
    class ObservedWork(actual):
        def __init__(self,budget):
            super().__init__(budget);meters.append(self)
    monkeypatch.setattr(selection,"_ValidationWork",ObservedWork)
    with pytest.raises(ValueError,match="validation budget refused: "+resource):
        selection.validate_candidate_certificate(counterfeit,readings,evidence,rows)
    assert len(meters)==1 and meters[0].budget==counterfeit.budget
    assert meters[0].counts[resource]<=cap and meters[0].denial[0]==resource
