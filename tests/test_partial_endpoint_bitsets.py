"""Rollback controls: accepted pair AC3 is authoritative; bitsets remain archived."""
from dataclasses import replace
from heapq import heappush, heappop
import gc
import random
import weakref

import pytest

from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from test_partial_endpoint_ac3 import meter, incidence_for, random_source, semantic, parallel_source
from test_partial_axis_observability import fixture, solve, certificate


def accepted_ac3(work, factors, incidence, domains, dirty, singleton):
    """Accepted v2 AC3 kernel: immutable original pair scans, same dirty queue."""
    if any(not values for values in domains.values()):
        work.pruning["inconsistent_nodes"] += 1
        return False
    queue, queued = [], set()
    def enqueue(fid):
        if fid not in queued:
            work.relation_charge("arc_queue")
            queued.add(fid)
            heappush(queue, fid)
    if dirty is None:
        for fid in range(len(factors)):
            work.relation_charge("arc_incidence")
            enqueue(fid)
    else:
        for key in dirty:
            for fid in incidence[key]:
                work.relation_charge("arc_incidence")
                enqueue(fid)
    while queue:
        work.relation_charge("arc_queue")
        fid = heappop(queue)
        queued.remove(fid)
        work.counts["arc_factor_scans"] += 1
        u, v, relation = factors[fid]
        su, sv = set(), set()
        for a, b in sorted(relation):
            work.relation_charge("arc_relation")
            if a in domains[u] and b in domains[v]:
                su.add(a)
                sv.add(b)
        changed = []
        for key, supported in ((u, su), (v, sv)):
            removed = len(domains[key] - supported)
            if removed:
                work.pruning["values_removed"] += removed
                domains[key].intersection_update(supported)
                if len(domains[key]) == 1:
                    singleton(key)
                changed.append(key)
            if not domains[key]:
                work.pruning["inconsistent_nodes"] += 1
                return False
        for key in changed:
            for neighbor in incidence[key]:
                work.relation_charge("arc_incidence")
                if neighbor != fid:
                    enqueue(neighbor)
    return True


@pytest.mark.parametrize("seed", range(80))
def test_sixteen_alternative_random_root_child_and_sibling_match_accepted_ac3(seed):
    rng = random.Random(seed)
    domains = {i: set(range(16)) for i in range(4)}
    factors = tuple((u, v, {(a, b): (0,) for a in range(16) for b in range(16)
                           if rng.random() < (.2 if seed % 2 else .8)})
                    for u, v in ((0, 1), (1, 2), (2, 3), (0, 3)))
    incidence = incidence_for(factors, domains)
    work = meter()
    reference_work = meter()
    new = {k: v.copy() for k, v in domains.items()}
    old = {k: v.copy() for k, v in domains.items()}
    assert csp._propagate_partial_arcs(work, factors, incidence, new, None, lambda k: None)
    assert accepted_ac3(reference_work, factors, incidence, old, None, lambda k: None)
    assert old == new
    root = {k: v.copy() for k, v in new.items()}
    for value in (0, 7, 15):
        new_child, old_child = {k: v.copy() for k, v in new.items()}, {k: v.copy() for k, v in old.items()}
        new_child[3] = old_child[3] = {value}
        new_ok = csp._propagate_partial_arcs(work, factors, incidence, new_child, (3,), lambda k: None)
        old_ok = accepted_ac3(reference_work, factors, incidence, old_child, (3,), lambda k: None)
        assert old_ok == new_ok
        if old_ok:
            assert new_child == old_child
    assert new == old == root  # siblings never contaminate propagated parent
    assert work.counts == reference_work.counts and work.pruning == reference_work.pruning
    assert not hasattr(work, "arc_support_indexes")


@pytest.mark.parametrize("seed", range(40))
def test_real_partial_search_choices_physical_and_eligibility_digests_match_ac3(seed, monkeypatch):
    data = random_source(seed)
    new = solve(data)
    monkeypatch.setattr(csp, "_propagate_partial_arcs", accepted_ac3)
    old = solve(data)
    assert semantic(old) == semantic(new)
    assert dict(new.counters) == dict(old.counters)
    assert dict(new.pruning) == dict(old.pruning)
    assert old.counters["branch_nodes"] == new.counters["branch_nodes"]
    assert old.counters["physical_assignments"] == new.counters["physical_assignments"]


@pytest.mark.parametrize("kind", ["unique", "no", "multiple"])
def test_physical_exclusion_and_original_support_rows_match_ac3(kind, monkeypatch):
    data = parallel_source(kind)
    new = solve(data)
    monkeypatch.setattr(csp, "_propagate_partial_arcs", accepted_ac3)
    old = solve(data)
    assert semantic(new) == semantic(old)
    assert dict(new.counters) == dict(old.counters)
    assert dict(new.pruning) == dict(old.pruning)


def test_actual_multi_quadrant_search_reuses_source_factors_not_active_ids(monkeypatch):
    readings, rows, frames, constraints = fixture(floating=True)
    data = readings, rows, frames, tuple(q for q in constraints if q[1] not in ("f", "g"))
    new = solve(data)
    assert new.status == "multiple" and new.counters["visited_assignments"] >= 2
    monkeypatch.setattr(csp, "_propagate_partial_arcs", accepted_ac3)
    old = solve(data)
    assert semantic(new) == semantic(old)
    assert dict(new.counters) == dict(old.counters)
    assert dict(new.pruning) == dict(old.pruning)


def repeated_branches(kernel, relation):
    factors = ((0, 1, relation),)
    parent = {0: set(range(16)), 1: set(range(16))}
    incidence = incidence_for(factors, parent)
    work = meter()
    assert kernel(work, factors, incidence, parent, None, lambda k: None)
    for value in range(16):
        child = {k: v.copy() for k, v in parent.items()}
        child[0] = {value}
        assert kernel(work, factors, incidence, child, (0,), lambda k: None)
    return work


@pytest.mark.parametrize("dense", [False, True])
def test_restored_dense_and_sparse_costs_exactly_match_accepted_ac3(dense):
    relation = {(a, b): (0,) for a in range(16) for b in range(16) if dense or a == b}
    old, new = repeated_branches(accepted_ac3, relation), repeated_branches(csp._propagate_partial_arcs, relation)
    assert new.counts == old.counts
    assert new.counts["relation_visits"] == (4403 if dense else 339)
    assert new.counts["arc_relation_visits"] == 17 * len(relation)
    assert new.counts["relation_visits"] == sum(new.counts[k] for k in csp.PARTIAL_ARC_COUNTERS
                                              if k != "arc_factor_scans")
    print("dense=" + str(dense), "accepted_total=" + str(old.counts["relation_visits"]),
          "restored_total=" + str(new.counts["relation_visits"]))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_reversed_asymmetric_endpoints_and_empty_or_relation(reverse, empty):
    relation = {} if empty else {(0, 15): (0,), (1, 2): (1,), (8, 3): (2,)}
    domains = {0: {0, 1, 8, 12}, 1: {2, 15}}
    if reverse:
        relation = {(b, a): rows for (a, b), rows in relation.items()}
    factors = ((1, 0, relation),) if reverse else ((0, 1, relation),)
    incidence = incidence_for(factors, domains)
    old, new = {k: v.copy() for k, v in domains.items()}, {k: v.copy() for k, v in domains.items()}
    assert accepted_ac3(meter(), factors, incidence, old, None, lambda k: None) == (
        csp._propagate_partial_arcs(meter(), factors, incidence, new, None, lambda k: None))
    assert old == new


def test_reused_active_factor_ids_scan_original_relations_and_retain_no_domain_cache():
    work = meter()
    relations = ({(0, 1): (0,)}, {(1, 0): (1,)})
    for relation, expected in zip(relations, ((0, 1), (1, 0))):
        domains = {0: {0, 1}, 1: {0, 1}}
        factors = ((0, 1, relation),)  # same active-list local factor ID0
        incidence = incidence_for(factors, domains)
        assert csp._propagate_partial_arcs(work, factors, incidence, domains, None, lambda k: None)
        assert domains == {0: {expected[0]}, 1: {expected[1]}}
    assert work.counts["arc_relation_visits"] == 2
    assert not hasattr(work, "arc_support_indexes")
    assert not hasattr(csp, "_partial_support_index")
    ref = weakref.ref(domains[0])
    del domains
    gc.collect()
    assert ref() is None


def test_restored_exact_cap_and_cap_minus_one_match_accepted_pair_accounting():
    relation = {(a, b): (0,) for a in range(16) for b in range(16)}
    factors = ((0, 1, relation),)
    domains = {0: set(range(16)), 1: set(range(16))}
    incidence = incidence_for(factors, domains)
    work = meter()
    assert csp._propagate_partial_arcs(work, factors, incidence, domains, None, lambda k: None)
    used = work.counts["relation_visits"]
    assert used == 259  # root incidence + push + pop + 256 original pairs
    for cap in (used, used - 1):
        current = {0: set(range(16)), 1: set(range(16))}
        limited = meter(cap)
        if cap == used:
            assert csp._propagate_partial_arcs(limited, factors, incidence, current, None, lambda k: None)
        else:
            with pytest.raises(csp._Exhausted):
                csp._propagate_partial_arcs(limited, factors, incidence, current, None, lambda k: None)
            category, charged, amount, ceiling = limited.denial
            assert category == "relation_visits" and charged <= cap < charged + amount and ceiling == cap


@pytest.mark.parametrize("counter", ["arc_relation_visits", "arc_queue_visits",
    "arc_incidence_visits", "arc_factor_scans"])
def test_resealed_tampered_restored_pair_proof_is_rejected(counter):
    data = fixture()
    cert, evidence = certificate(data)
    proof = cert.proof_record()
    assert proof["counters"][counter] > 0
    proof["counters"][counter] -= 1
    tampered = replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
    with pytest.raises(ValueError):
        selection.validate_candidate_certificate(tampered, data[0], evidence, data[1])


def test_restored_pair_charge_precedes_domain_membership_and_has_no_bitset_state():
    class NoRead(set):
        def __contains__(self, item):
            pytest.fail("refused pair scan inspected domain")
    relation = {(0, 0): (0,)}
    work = meter(3)  # seed + push + pop; pair charge must refuse before read
    factors = ((0, 1, relation),)
    domains = {0: NoRead((0,)), 1: NoRead((0,))}
    with pytest.raises(csp._Exhausted):
        csp._propagate_partial_arcs(work, factors, incidence_for(factors, domains), domains, None, lambda k: None)
    assert work.counts["arc_relation_visits"] == 0 and domains == {0: {0}, 1: {0}}
    assert work.denial == ("relation_visits", 3, 1, 3)
    assert not hasattr(work, "arc_support_indexes")


def test_bitset_v3_proof_and_extra_counters_cannot_obtain_restored_v2_authority():
    from mc_photogrammetry.reconstruction import partial_axis_observability as partial
    assert partial.TRANSCRIPT_POLICY == "source-bound-partial-axis-transcript-v2-ac3"
    cert, _ = certificate(fixture())
    proof = cert.proof_record()
    proof["transcript_policy"] = "source-bound-partial-axis-transcript-v3-bitsets"
    proof["counters"]["arc_index_build_visits"] = 0
    with pytest.raises(ValueError):
        replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
