"""Explicit partial-only search profile; no image or pipeline execution."""
from dataclasses import asdict, fields, replace

import pytest

from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_partial_joint_graph as graph_mod
from test_partial_axis_observability import fixture, solve, certificate
from test_plane_graph import make_reading, make_match


def busy_case():
    """Many genuine same-cue witnesses and 16 physical distance alternatives.

    A separate fixed-distance parallel-plane factor forces exactly alternative
    7 of b's first X reading. Duplicate source rows remain distinct mandatory
    witnesses; no counter mocking/scaling is used to cross the old hard cap.
    """
    original, rows, frames, constraints = fixture()
    replaced = tuple(make_reading("b", 0, 1, [(1., float(i + 1), 1.) for i in range(16)])
                     if r.frame == "b" and r.axis == 0 else r for r in original)
    readings = (*replaced, make_reading("a", 0, 1, [(2., 20., 1.)]),
                make_reading("b", 0, 1, [(2., 20., 1.)]))
    old, old_ids, _ = plane_graph._canonical_reading_index(original)
    indexed, new_ids, _ = plane_graph._canonical_reading_index(readings)
    mapping = {old_ids[id(r)]: new_ids[id(replaced[i])] for i, r in enumerate(original)}
    rebound = tuple(replace(row, left_plane=mapping[(row.left, row.left_plane)][1],
                            right_plane=mapping[(row.right, row.right_plane)][1]) for row in rows)
    duplicate = next(row for row in rebound if row.left == "b" and row.right == "a" and row.left_axis == 0)
    bfix = next(i for (f, i), r in indexed.items() if f == "b" and r.axis == 0 and r.alternatives[0].blocks_per_period == 2.)
    afix = next(i for (f, i), r in indexed.items() if f == "a" and r.axis == 0 and r.alternatives[0].blocks_per_period == 2.)
    return readings, (*rebound, make_match("b", "a", 0, 0, 0, bfix, afix, left_b=2., right_b=2.),
                      *((duplicate,) * 2500)), frames, constraints


@pytest.fixture(scope="module")
def busy():
    data = busy_case()
    extended = solve(data, csp.PartialEndpointBudget())
    return data, extended


def test_natural_old_hard_cap_refuses_and_explicit_extended_is_exhaustive_unique(busy):
    data, extended = busy
    assert extended.status == "unique" and extended.exhaustive, (extended.status, extended.denial)
    assert 262144 < extended.counters["relation_visits"] <= 524288
    old = solve(data)
    assert old.status == "budget_refused" and not old.exhaustive and not old.choices
    assert old.denial[0] == "relation_visits" and old.denial[3] == 262144
    assert extended.choices[0].physical_groups
    print("natural synthetic relation_visits=", extended.counters["relation_visits"],
          "old_denial=", old.denial, "new_exhaustive=", extended.exhaustive)


def test_extended_exact_cap_and_found_one_before_cap_minus_one_still_refuses(busy):
    data, extended = busy
    used = extended.counters["relation_visits"]
    exact = solve(data, csp.PartialEndpointBudget(relation_visits=used))
    assert exact.status == "unique" and exact.exhaustive
    refused = solve(data, csp.PartialEndpointBudget(relation_visits=used - 1))
    assert refused.status == "budget_refused" and not refused.choices and not refused.exhaustive
    assert refused.counters["solutions_found"] == 1
    assert refused.denial[0] == "relation_visits" and refused.denial[3] == used - 1


def test_legacy_default_and_full_mode_budget_payload_unchanged():
    default = csp.EndpointBudget()
    assert default.relation_visits == 262144
    assert asdict(default) == {field.name: field.default for field in fields(csp.EndpointBudget)}
    data = fixture(complete=True)
    assert solve(data, mode="full") == solve(data, default, mode="full")
    cert, _ = certificate(data, mode="full")
    assert selection._payload(cert)["budget"] == asdict(default)
    assert "budget_profile" not in selection._payload(cert)
    ordinary, _ = certificate(fixture())
    assert "budget_profile" not in selection._payload(ordinary)


@pytest.mark.parametrize("name", [field.name for field in fields(csp.EndpointBudget)])
def test_profile_bool_and_other_ceilings_are_unchanged(name):
    ceiling = csp.PartialEndpointBudget().relation_visits if name == "relation_visits" else getattr(csp.EndpointBudget(), name)
    with pytest.raises(ValueError):
        csp.PartialEndpointBudget(**{name: True})
    with pytest.raises(ValueError):
        csp.PartialEndpointBudget(**{name: ceiling + 1})
    with pytest.raises(ValueError):
        csp.EndpointBudget(**{name: getattr(csp.EndpointBudget(), name) + 1})


def test_arbitrary_subclass_and_full_mode_extended_profile_rejected_at_boundaries():
    class Foreign(csp.EndpointBudget):
        pass
    with pytest.raises(ValueError):
        Foreign()
    foreign = object.__new__(Foreign)
    for field in fields(csp.EndpointBudget):
        object.__setattr__(foreign, field.name, field.default)
    for budget in (foreign, csp.PartialEndpointBudget(), csp.PartialEndpointBudget(relation_visits=1)):
        with pytest.raises(ValueError):
            solve(fixture(complete=True), budget, mode="full")
        with pytest.raises(ValueError):
            certificate(fixture(complete=True), budget, mode="full")
    with pytest.raises(ValueError):
        solve(fixture(), foreign)
    with pytest.raises(ValueError):
        csp._Work(foreign)


def test_extended_certificate_binds_profile_cap_and_replays_tamper():
    data = fixture()
    budget = csp.PartialEndpointBudget()
    cert, evidence = certificate(data, budget)
    assert type(cert.budget) is csp.PartialEndpointBudget
    assert selection._payload(cert)["budget_profile"] == "partial-relations-524288-v1"
    selection.validate_candidate_certificate(cert, data[0], evidence, data[1])
    with pytest.raises(ValueError):
        replace(cert, budget=csp.EndpointBudget())  # old digest cannot bind a changed profile
    altered = replace(cert, budget=replace(budget, relation_visits=524287), result_sha256="")
    assert altered.result_sha256 != cert.result_sha256
    selection.validate_candidate_certificate(altered, data[0], evidence, data[1])
    same_cap_legacy, _ = certificate(data)
    same_cap_profile, _ = certificate(data, csp.PartialEndpointBudget(relation_visits=262144))
    assert same_cap_profile.proof_json == same_cap_legacy.proof_json
    assert same_cap_profile.result_sha256 != same_cap_legacy.result_sha256
    proof = cert.proof_record()
    proof["counters"]["relation_visits"] += 1
    altered = replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
    with pytest.raises(ValueError):
        selection.validate_candidate_certificate(altered, data[0], evidence, data[1])


def test_second_solution_remains_ambiguity_and_contradictory_peripheral_remains_no():
    ambiguous = solve(fixture(alternatives=2), csp.PartialEndpointBudget())
    assert ambiguous.status == "multiple" and len(ambiguous.choices) == 2
    readings, rows, frames, constraints = fixture()
    # Conflicting relative-Q constraints remain decisive for the complete
    # universe, including the partially observed peripheral e.
    inconsistent = (readings, rows, frames, (*constraints, ("b", "e", 1)))
    assert solve(inconsistent, csp.PartialEndpointBudget()).status == "no"
    physically_bad = list(readings)
    index = next(i for i, r in enumerate(readings) if r.frame == "e" and r.axis == 2)
    physically_bad[index] = make_reading("e", 2, 1, [(1., float("inf"), 1.)])
    with pytest.raises(ValueError):
        solve((tuple(physically_bad), rows, frames, constraints), csp.PartialEndpointBudget())


@pytest.mark.parametrize("resource", ["scalar_payload_bytes", "scalar_row_bytes", "solution_rows", "branch_nodes"])
def test_extended_scalar_proof_solution_and_branch_limits_do_not_increase(resource):
    result = solve(fixture(), replace(csp.PartialEndpointBudget(), **{resource: 0}))
    assert result.status == "budget_refused" and not result.choices
    assert result.denial[0] == resource
    assert getattr(csp.PartialEndpointBudget(), resource) == getattr(csp.EndpointBudget(), resource)


def test_partial_numerical_allowance_never_inherits_extended_search_limit(monkeypatch):
    search = csp.PartialEndpointBudget()
    assert graph_mod._numerical_budget(search) == csp.EndpointBudget()
    assert graph_mod._numerical_budget(replace(search, relation_visits=17)).relation_visits == 17
    data = fixture()
    cert, evidence = certificate(data, search)
    original = graph_mod._constraints
    def at_numeric_boundary(indexed, matches, certificate, work):
        assert type(work.budget) is csp.EndpointBudget and work.budget.relation_visits == 262144
        work.counts["relation_visits"] = 262144
        return original(indexed, matches, certificate, work)
    monkeypatch.setattr(graph_mod, "_constraints", at_numeric_boundary)
    monkeypatch.setattr(graph_mod.np.linalg, "lstsq", lambda *a, **k: pytest.fail("allocation/solve after refusal"))
    graph = graph_mod._assemble(data[0], evidence, data[1], cert)
    assert graph.selection.status == "unique" and graph.geometry_evaluations == 1
    assert graph.status == "unavailable" and graph.reason == "partial_geometry_budget_refused"
    assert graph.core_solution is None and graph.axis_offsets == ()


def test_extended_profile_genuine_source_graph_and_exact_budget_replay(monkeypatch):
    from test_image_derived_graph_v3 import _source_records
    from mc_photogrammetry.reconstruction import plane_placement as placement
    from mc_photogrammetry.pipeline import image_derived_joint_graph as full_graph
    sources = _source_records(monkeypatch)
    before = full_graph.derive_image_derived_graph(**sources)
    budget = csp.PartialEndpointBudget()
    graph = graph_mod.derive_partial_image_derived_graph(**sources, budget=budget)
    assert type(graph) is graph_mod.PartialJointGraphResult and graph.status == "assembled"
    assert graph.selection.budget is budget and graph.geometry_work <= 262144
    assert isinstance(graph_mod.replay_partial_image_derived_graph(graph_result=graph, **sources), placement.PlacementRefusal)
    assert graph_mod.replay_partial_image_derived_graph(graph_result=graph, **sources, budget=budget) is graph
    after = full_graph.derive_image_derived_graph(**sources)
    assert before.result_sha256 == after.result_sha256
    class Foreign(csp.PartialEndpointBudget):
        pass
    foreign = object.__new__(Foreign)
    monkeypatch.setattr(graph_mod.source_mod, "_sources", lambda **k: pytest.fail("invalid budget reached source"))
    assert isinstance(graph_mod.derive_partial_image_derived_graph(**sources, budget=foreign), placement.PlacementRefusal)


def test_extended_profile_rejects_finite_contradictory_peripheral_physics():
    readings, rows, frames, constraints = fixture()
    readings += (make_reading("e", 2, 1, [(1., 20., 1.)]),)
    indexed, _, _ = plane_graph._canonical_reading_index(readings)
    old, _, _ = plane_graph._canonical_reading_index(fixture()[0])
    rebound = []
    for row in rows:
        i = next(i for (f, i), r in indexed.items() if f == row.right and r == old[(row.right, row.right_plane)])
        rebound.append(replace(row, right_plane=i))
    b = next(i for (f, i), r in indexed.items() if f == "b" and r.axis == 2)
    e = next(i for (f, i), r in indexed.items() if f == "e" and r.axis == 2 and r.alternatives[0].distance_blocks == 20.)
    result = solve((readings, (*rebound, make_match("b", "e", 2, 2, 0, b, e)), frames, constraints),
                   csp.PartialEndpointBudget())
    assert result.status == "no" and result.exhaustive and not result.choices
    assert result.counters["partial_prunes"] + result.counters["physical_rejected"] > 0


def test_profile_cannot_bypass_fixed_proof_or_record_limits():
    assert selection.MAX_PROOF_BYTES == 4 * 1024 * 1024
    assert selection.MAX_RECORD_BYTES == 16 * 1024 * 1024
    old, _ = certificate(fixture())
    extended, _ = certificate(fixture(), csp.PartialEndpointBudget())
    assert len(extended.proof_json.encode()) == len(old.proof_json.encode())
    # Encoder proof limits are independent of the selected budget class.
    with pytest.raises(selection._RecordExhausted, match="proof_bytes"):
        selection._encode(extended.proof_record(), limit=1, name="proof_bytes")
