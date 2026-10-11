"""Offline source-bound partial selection; no pixels or geometry solver."""
from dataclasses import fields, replace
import json

import pytest

from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import partial_axis_observability as partial
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from test_plane_graph import make_reading, make_match
from test_image_derived_endpoint_selection import _evidence


def fixture(*, floating=False, complete=False, alternatives=1):
    axes = {"a": (0, 1), "b": (0, 1, 2), "c": (0, 1, 2),
            "d": (0, 1, 2), "e": (1, 2)}
    if complete:
        axes = {frame: (0, 1, 2) for frame in axes}
    if floating:
        axes.update(f=(2,), g=(2,))
    readings = tuple(make_reading(frame, axis, 1,
        [(float(i + 1), float(8 * (i + 1)), 1.) for i in range(alternatives)])
        for frame, normals in axes.items() for axis in normals)
    indexed, _, _ = plane_graph._canonical_reading_index(readings)
    keys = {(frame, reading.axis): index for (frame, index), reading in indexed.items()}
    def row(left, right, axis, alt=0):
        return make_match(left, right, axis, axis, 0, keys[(left, axis)], keys[(right, axis)],
                          left_b=float(alt + 1), right_b=float(alt + 1))
    rows = tuple(row("b", frame, axis, alt) for frame, normals in axes.items()
                 if frame not in ("b", "f", "g") for axis in normals
                 for alt in range(alternatives))
    if floating:
        rows += (row("f", "g", 2),)
    frames = tuple(axes)
    constraints = tuple(("b", frame, 0) for frame in frames if frame != "b")
    return readings, rows, frames, constraints


def solve(data, budget=csp.EndpointBudget(), mode="partial"):
    readings, rows, frames, constraints = data
    return csp.solve_endpoint_hypotheses(readings, rows, session_frames=frames,
        quadrant_constraints=constraints, budget=budget, _observability_mode=mode)


def certificate(data, budget=csp.EndpointBudget(), mode="partial"):
    readings, rows, frames, constraints = data
    source = selection.select_partial_endpoint_hypotheses if mode == "partial" else selection.select_endpoint_hypotheses
    evidence = _evidence(readings)
    cert = source(readings=readings, signatures=evidence, candidate_matches=rows,
        frame_ids=frames, quadrant_constraints=constraints, budget=budget,
        session_sha256="a" * 64, lattice_sha256="b" * 64, signatures_sha256="c" * 64)
    return cert, evidence


def test_partial_three_frame_core_incomplete_peripherals_and_source_replay(monkeypatch):
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", lambda *a: pytest.fail("geometry evaluation"))
    data = fixture()
    full = solve(data, mode="full")
    result = solve(data)
    assert full.status == "no" and result.status == "unique" and result.exhaustive
    assert result.transcript_policy == partial.TRANSCRIPT_POLICY
    assert result.choices[0].quadrants == tuple((frame, 0) for frame in data[2])
    assert len(result.choices[0].alternatives) == len(data[0])
    assert sorted(i for factor in result.choices[0].factors for i in factor.source_rows) == list(range(len(data[1])))
    cert, evidence = certificate(data)
    assert type(cert) is selection.PartialEndpointSelectionCertificate
    assert cert.reason == "partial_geometry_not_evaluated"
    observation, = cert.observability
    assert observation.anchor == "b" and observation.core == ("b", "c", "d")
    assert observation.resolved_axes("a") == (0, 1)
    assert observation.resolved_axes("e") == (1, 2)
    assert not hasattr(cert, "translations")
    assert partial.from_payload(json.loads(selection._encode(observation.payload()))) == observation
    selection.validate_candidate_certificate(cert, data[0], evidence, data[1])
    assert replace(cert) == cert


def test_partial_floating_component_offsets_remain_unknown_and_factors_mandatory():
    data = fixture(floating=True)
    cert, evidence = certificate(data)
    assert cert.status == "unique"
    observation, = cert.observability
    assert observation.core == ("b", "c", "d")
    assert observation.resolved_axes("f") == observation.resolved_axes("g") == ()
    assert (("f", "g"), "f", False) in dict(observation.partitions)[2]
    assert cert.representatives[-1].source_row == len(data[1]) - 1
    assert len(cert.representatives) == len(data[1])
    selection.validate_candidate_certificate(cert, data[0], evidence, data[1])


def test_partial_quadrant_gauge_uses_measured_anchor_not_lexical_peripheral():
    readings, rows, frames, constraints = fixture()
    rows = tuple(replace(row, quadrant_of_right=2) if row.right == "a" else row for row in rows)
    constraints = tuple((left, right, 2 if right == "a" else q) for left, right, q in constraints)
    data = (readings, rows, frames, constraints)
    cert, evidence = certificate(data)
    assert cert.status == "unique" and cert.observability[0].anchor == "b"
    quadrants = dict(cert.proof_record()["choices"][0]["quadrants"])
    assert quadrants["a"] == 2 and quadrants["b"] == 0
    selection.validate_candidate_certificate(cert, readings, evidence, rows)


def test_partial_contradictory_peripheral_evidence_is_not_dropped():
    readings, rows, frames, constraints = fixture()
    readings += (make_reading("e", 2, 1, [(1., 20., 1.)]),)
    indexed, _, _ = plane_graph._canonical_reading_index(readings)
    b = next(i for (f, i), r in indexed.items() if f == "b" and r.axis == 2)
    e = next(i for (f, i), r in indexed.items() if f == "e" and r.axis == 2 and r.alternatives[0].distance_blocks == 20.)
    # Rebuild original row indices after the additional reading's canonical sort.
    old, _, _ = plane_graph._canonical_reading_index(fixture()[0])
    rebound = []
    for row in rows:
        right_reading = old[(row.right, row.right_plane)]
        index = next(i for (f, i), r in indexed.items() if f == row.right and r == right_reading)
        rebound.append(replace(row, right_plane=index))
    rows = (*rebound, make_match("b", "e", 2, 2, 0, b, e))
    result = solve((readings, rows, frames, constraints))
    assert result.status == "no" and result.exhaustive and not result.choices
    assert result.counters["partial_prunes"] + result.counters["physical_rejected"] > 0


@pytest.mark.parametrize("ambiguity", ["scale", "quadrant", "association"])
def test_partial_ambiguity_never_selects_a_preferred_core(ambiguity):
    if ambiguity == "scale":
        data = fixture(alternatives=2)
    elif ambiguity == "association":
        readings, rows, frames, constraints = fixture()
        data = (readings, (*rows, replace(rows[0], cell_shift_rc=(1, 2))), frames, constraints)
    else:
        readings, rows, frames, constraints = fixture(floating=True)
        # Floating peripheral pair has four equally supported orientations.
        constraints = tuple(row for row in constraints if row[1] not in ("f", "g"))
        data = (readings, rows, frames, constraints)
    result = solve(data)
    assert result.status == ("unresolved_evidence" if ambiguity == "association" else "multiple")
    cert, evidence = certificate(data)
    assert cert.status == result.status and cert.representatives == ()
    selection.validate_candidate_certificate(cert, data[0], evidence, data[1])


def test_partial_no_two_frame_core_refuses_without_zero_filling():
    readings, rows, frames, constraints = fixture()
    rows = tuple(row for row in rows if not (row.right in ("c", "d") and row.left_axis == 0))
    rows += (make_match("c", "d", 0, 0, 0, 0, 0),)
    result = solve((readings, rows, frames, constraints))
    assert result.status == "no" and result.eligibility_reasons == {"no_two_frame_observable_core": 1}
    assert not result.choices and result.counters["branch_nodes"] == 0


def test_partial_no_fully_measured_anchor_is_source_bound_refusal():
    readings = tuple(make_reading(frame, 1, 1, [(1., 8., 1.)]) for frame in ("a", "b"))
    data = (readings, (make_match("a", "b", 1, 1, 0, 0, 0),), ("a", "b"), ())
    result = solve(data)
    assert result.status == "no" and result.reason == "no_fully_observed_anchor" and result.exhaustive
    assert result.source_digest and result.counters["planned_assignments"] == 0
    cert, evidence = certificate(data)
    selection.validate_candidate_certificate(cert, readings, evidence, data[1])


def test_partial_fully_observable_semantic_parity_and_full_default_bytes():
    data = fixture(complete=True)
    full, candidate = solve(data, mode="full"), solve(data)
    assert full.status == candidate.status == "unique"
    a, b = full.choices[0], candidate.choices[0]
    assert (a.quadrants, a.alternatives, a.factors, a.physical_groups) == (b.quadrants, b.alternatives, b.factors, b.physical_groups)
    full_cert, evidence = certificate(data, mode="full")
    omitted = csp.solve_endpoint_hypotheses(data[0], data[1], session_frames=data[2], quadrant_constraints=data[3])
    assert omitted == full
    assert full.transcript_policy == "source-bound-physical-transcript-v3"
    assert "observability" not in selection._payload(full_cert)
    assert full_cert.schema == "mc-photogrammetry-endpoint-selection-v4"
    assert full_cert.reason == "joint_geometry_not_evaluated"
    candidate_cert, _ = certificate(data)
    assert candidate_cert.observability[0].core == data[2]
    selection.validate_candidate_certificate(full_cert, data[0], evidence, data[1])


def test_partial_invalid_inactive_original_row_still_fails_global_preflight():
    readings, rows, frames, constraints = fixture()
    invalid = replace(rows[0], quadrant_of_right=2, left_b=99.)
    with pytest.raises(ValueError, match="unresolvable"):
        solve((readings, (*rows, invalid), frames, constraints))


def test_partial_exact_relation_cap_and_cap_minus_one_are_atomic():
    data = fixture()
    result = solve(data)
    used = result.counters["relation_visits"]
    budget = replace(csp.EndpointBudget(), relation_visits=used)
    exact = solve(data, budget)
    assert exact.status == "unique" and exact.counters["relation_visits"] == used
    refused = solve(data, replace(budget, relation_visits=used - 1))
    assert refused.status == "budget_refused" and not refused.choices and not refused.exhaustive
    assert refused.denial[0] == "relation_visits"
    cert, evidence = certificate(data, budget)
    selection.validate_candidate_certificate(cert, data[0], evidence, data[1])


@pytest.mark.parametrize("resource", [field.name for field in fields(csp.EndpointBudget)])
def test_partial_each_budget_refusal_has_no_resolved_projection(resource):
    data = fixture()
    cert, evidence = certificate(data, replace(csp.EndpointBudget(), **{resource: 0}))
    assert cert.status == "budget_refused" and cert.observability == cert.representatives == ()
    selection.validate_candidate_certificate(cert, data[0], evidence, data[1])


@pytest.mark.parametrize("change", ["core", "gauge", "unknown", "policy"])
def test_partial_resealed_topology_and_transcript_tampering_rejected(change):
    cert, evidence = certificate(fixture(floating=True))
    observation, = cert.observability
    with pytest.raises(ValueError):
        if change == "core":
            replace(cert, observability=(replace(observation, core=("a", "b", "c", "d")),), result_sha256="")
        elif change in ("gauge", "unknown"):
            partitions = list(observation.partitions)
            axis, components = partitions[2]
            components = list(components)
            members, gauge, known = components[-1]
            components[-1] = (members, "b" if change == "gauge" else gauge, True if change == "unknown" else known)
            partitions[2] = (axis, tuple(components))
            replace(cert, observability=(replace(observation, partitions=tuple(partitions)),), result_sha256="")
        else:
            proof = cert.proof_record(); proof["transcript_policy"] = csp.TRANSCRIPT_POLICY
            replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")


def test_partial_resealed_physical_proof_requires_exact_source_replay():
    data = fixture()
    cert, evidence = certificate(data)
    proof = cert.proof_record()
    proof["physical_digest"] = "f" * 64
    altered = replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
    with pytest.raises(ValueError, match="does not recompute"):
        selection.validate_candidate_certificate(altered, data[0], evidence, data[1])
    with pytest.raises(ValueError, match="bindings differ"):
        selection.validate_candidate_certificate(cert, data[0], evidence, data[1][:-1])


@pytest.mark.parametrize("mode", [None, False, True, 1, "Partial", ""])
def test_partial_invalid_opt_in_is_rejected_before_source_traversal(mode):
    with pytest.raises(ValueError, match="observability mode"):
        csp.solve_endpoint_hypotheses((), (), session_frames=(), _observability_mode=mode)


@pytest.mark.parametrize("fault", ["bool_axis", "oversized_component", "oversized_frame", "duplicate", "wrong_core"])
def test_partial_partition_readback_caps_and_exact_scalar_types(fault):
    cert, _ = certificate(fixture())
    value = json.loads(selection._encode(cert.observability[0].payload()))
    if fault == "bool_axis":
        value[2][0][0] = False
    elif fault == "oversized_component":
        value[2][0][1][0][0] = [str(i) for i in range(33)]
    elif fault == "oversized_frame":
        value[0] = "x" * 4097
    elif fault == "duplicate":
        value[2][0][1].append(value[2][0][1][0])
    else:
        value[1] = ["a", "b", "c", "d"]
    with pytest.raises(ValueError):
        partial.from_payload(value)
