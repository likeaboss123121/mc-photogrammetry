"""Offline differential controls against the replaced full-sweep algorithm."""
from dataclasses import replace
import random

import pytest

from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from test_partial_axis_observability import fixture, solve, certificate
from test_plane_graph import make_reading, make_match


def reference_sweep(work, factors, incidence, domains, dirty, singleton):
    """Verbatim propagation semantics from the pre-AC3 partial visit kernel."""
    changed = True
    while changed:
        changed = False
        if any(not values for values in domains.values()):
            work.pruning["inconsistent_nodes"] += 1
            return False
        for u, v, relation in factors:
            su, sv = set(), set()
            for a, b in sorted(relation):
                work.charge("relation_visits", 1)
                if a in domains[u] and b in domains[v]:
                    su.add(a)
                    sv.add(b)
            for key, supported in ((u, su), (v, sv)):
                removed = len(domains[key] - supported)
                if removed:
                    work.pruning["values_removed"] += removed
                    domains[key].intersection_update(supported)
                    changed = True
                    if len(domains[key]) == 1:
                        singleton(key)
                if not domains[key]:
                    work.pruning["inconsistent_nodes"] += 1
                    return False
    return True


def meter(cap=262144):
    work = csp._Work(csp.EndpointBudget(relation_visits=cap))
    work.enable_partial_arcs()
    return work


def incidence_for(factors, domains):
    return {key: tuple(i for i, (u, v, _) in enumerate(factors) if key in (u, v)) for key in domains}


@pytest.mark.parametrize("seed", range(120))
def test_random_root_and_incremental_child_closures_match_reference(seed):
    rng = random.Random(seed)
    count, alternatives = rng.randrange(2, 7), rng.randrange(2, 5)
    domains = {key: set(range(alternatives)) for key in range(count)}
    factors = tuple((u, v, {(a, b): (0,) for a in range(alternatives) for b in range(alternatives)
                           if rng.random() < .65})
                    for u in range(count) for v in range(u + 1, count) if rng.random() < .6)
    incidence = incidence_for(factors, domains)
    old, new = {k: v.copy() for k, v in domains.items()}, {k: v.copy() for k, v in domains.items()}
    old_ok = reference_sweep(meter(), factors, incidence, old, None, lambda key: None)
    new_ok = csp._propagate_partial_arcs(meter(), factors, incidence, new, None, lambda key: None)
    assert old_ok == new_ok
    if not old_ok:
        return  # inconsistent prefixes need not have identical empty-domain order
    assert old == new
    for branch in new:
        if len(new[branch]) < 2:
            continue
        old_child, new_child = {k: v.copy() for k, v in new.items()}, {k: v.copy() for k, v in new.items()}
        old_child[branch] = new_child[branch] = {min(new[branch])}
        old_ok = reference_sweep(meter(), factors, incidence, old_child, (branch,), lambda key: None)
        new_ok = csp._propagate_partial_arcs(meter(), factors, incidence, new_child, (branch,), lambda key: None)
        assert old_ok == new_ok
        if old_ok:
            assert old_child == new_child


def chain_case(length=24, alternatives=4):
    domains = {i: set(range(alternatives)) for i in range(length)}
    factors = tuple((i, i + 1, {(a, a): (0,) for a in range(alternatives)}) for i in range(length - 1))
    return domains, factors, incidence_for(factors, domains)


def test_reverse_chain_revisits_dirty_neighbors_and_reduces_actual_scans():
    domains, factors, incidence = chain_case()
    old = {k: v.copy() for k, v in domains.items()}
    new = {k: v.copy() for k, v in domains.items()}
    old[23] = new[23] = {2}
    old_work, new_work = meter(), meter()
    assert reference_sweep(old_work, factors, incidence, old, (23,), lambda key: None)
    assert csp._propagate_partial_arcs(new_work, factors, incidence, new, (23,), lambda key: None)
    assert old == new == {key: {2} for key in domains}
    assert new_work.counts["arc_relation_visits"] == 23 * 4
    assert new_work.counts["arc_factor_scans"] == 23
    assert old_work.counts["relation_visits"] == 24 * 23 * 4
    assert new_work.counts["relation_visits"] < old_work.counts["relation_visits"] / 4
    print("reverse-chain legacy_scans=", old_work.counts["relation_visits"],
          "AC3_pair_scans=", new_work.counts["arc_relation_visits"],
          "AC3_total_metered=", new_work.counts["relation_visits"])


def test_child_only_touches_incident_connected_component():
    domains, factors, incidence = chain_case(length=8)
    factors = (*factors, (100, 101, {(a, a): (0,) for a in range(4)}))
    domains.update({100: set(range(4)), 101: set(range(4))})
    incidence = incidence_for(factors, domains)
    domains[7] = {1}
    work = meter()
    assert csp._propagate_partial_arcs(work, factors, incidence, domains, (7,), lambda key: None)
    assert domains[100] == domains[101] == set(range(4))
    assert work.counts["arc_factor_scans"] == 7 and work.counts["arc_relation_visits"] == 28


def test_dirty_factor_order_is_canonical_and_root_seed_has_no_duplicates():
    domains = {0: {0, 1}, 1: {0, 1}, 2: {0, 1}}
    scans = []
    class Tracked(dict):
        def __init__(self, fid):
            super().__init__({(0, 0): (0,), (1, 1): (1,)})
            self.fid = fid
        def __iter__(self):
            scans.append(self.fid)
            return super().__iter__()
    factors = ((0, 1, Tracked(0)), (1, 2, Tracked(1)), (0, 2, Tracked(2)))
    incidence = incidence_for(factors, domains)
    assert csp._propagate_partial_arcs(meter(), factors, incidence, domains, None, lambda key: None)
    assert scans == [0, 1, 2]
    scans.clear()
    domains[2] = {0}
    assert csp._propagate_partial_arcs(meter(), factors, incidence, domains, (2,), lambda key: None)
    assert scans == [1, 0, 2]  # newly dirty smaller factor precedes already queued2
    assert domains == {0: {0}, 1: {0}, 2: {0}}


def test_actual_pair_queue_incidence_accounting_and_exact_cap_refusal():
    domains, factors, incidence = chain_case(length=6)
    domains[5] = {3}
    work = meter()
    assert csp._propagate_partial_arcs(work, factors, incidence, domains, (5,), lambda key: None)
    assert work.counts["relation_visits"] == sum(work.counts[name] for name in csp.PARTIAL_ARC_COUNTERS
                                               if name != "arc_factor_scans")
    used = work.counts["relation_visits"]
    for cap in (used, used - 1):
        child = {i: set(range(4)) for i in range(6)}
        child[5] = {3}
        capped = meter(cap)
        if cap == used:
            assert csp._propagate_partial_arcs(capped, factors, incidence, child, (5,), lambda key: None)
            assert capped.counts["relation_visits"] == used
        else:
            with pytest.raises(csp._Exhausted):
                csp._propagate_partial_arcs(capped, factors, incidence, child, (5,), lambda key: None)
            assert capped.denial == ("relation_visits", used - 1, 1, used - 1)


def test_refused_pair_charge_precedes_domain_membership_evaluation():
    membership = []
    class Domain(set):
        def __contains__(self, value):
            membership.append(value)
            return super().__contains__(value)
    domains = {0: Domain((0, 1)), 1: Domain((0, 1))}
    factors = ((0, 1, {(0, 0): (0,), (1, 1): (1,)}),)
    work = meter(3)  # root incidence + queue insertion + queue removal
    with pytest.raises(csp._Exhausted):
        csp._propagate_partial_arcs(work, factors, incidence_for(factors, domains), domains,
                                    None, lambda key: pytest.fail("refused pair mutated singleton"))
    assert membership == [] and domains == {0: {0, 1}, 1: {0, 1}}
    assert work.counts["arc_relation_visits"] == 0
    assert work.denial == ("relation_visits", 3, 1, 3)


def semantic(result):
    return (result.status, result.reason, result.exhaustive, result.evidence_ambiguous,
            result.source_digest, result.choices, result.session_frames,
            result.eligibility_digest, dict(result.eligibility_reasons), result.excluded_assignments,
            result.physical_policy, result.physical_tolerance, result.physical_digest,
            result.physical_exclusions, result.partial_exclusions, dict(result.physical_reasons))


def random_source(seed):
    rng = random.Random(seed)
    frames = ("a", "b", "c")
    readings = tuple(make_reading(f, axis, 1,
        [(float(a + 1), 8. * (a + 1), 1.) for a in range(3)] if axis == 1 else [(1., 8., 1.)])
        for f in frames for axis in range(3))
    rows = tuple(make_match("a", f, axis, axis, 0, axis, axis)
                 for f in frames[1:] for axis in (0, 2))
    for u, v in (("a", "b"), ("b", "c"), ("a", "c")):
        pairs = [(a, b) for a in range(3) for b in range(3) if rng.random() < .4]
        if not pairs:
            pairs = [(rng.randrange(3), rng.randrange(3))]
        rows += tuple(make_match(u, v, 1, 1, 0, 1, 1, left_b=float(a + 1), right_b=float(b + 1))
                      for a, b in pairs)
    return readings, rows, frames, (("a", "b", 0), ("a", "c", 0))


@pytest.mark.parametrize("seed", range(40))
def test_actual_partial_solver_search_and_physical_transcripts_match_sweep(seed, monkeypatch):
    data = random_source(seed)
    new = solve(data)
    monkeypatch.setattr(csp, "_propagate_partial_arcs", reference_sweep)
    old = solve(data)
    assert semantic(old) == semantic(new)
    assert old.counters["branch_nodes"] == new.counters["branch_nodes"]
    assert old.counters["physical_assignments"] == new.counters["physical_assignments"]


@pytest.mark.parametrize("options", [{}, {"floating": True}, {"alternatives": 2}, {"complete": True}])
def test_mandatory_source_factors_physical_prunes_and_ambiguity_match_reference(options, monkeypatch):
    data = fixture(**options)
    new = solve(data)
    monkeypatch.setattr(csp, "_propagate_partial_arcs", reference_sweep)
    old = solve(data)
    assert semantic(new) == semantic(old)


def parallel_source(kind):
    """Independent same-axis evidence fixes depth and rejects the wrong alias."""
    from mc_photogrammetry.reconstruction import plane_graph
    frames = ("a", "b", "c")
    readings = []
    main, parallel, stable = {}, {}, {}
    for frame in frames:
        stable[frame] = {}
        for axis in (0, 2):
            reading = make_reading(frame, axis, 1, [(1., 8., 1.)])
            readings.append(reading)
            stable[frame][axis] = reading
        if frame == "c":
            d1, d2 = {"unique": (9., 16.), "no": (10., 16.), "multiple": (9., 17.)}[kind]
        else:
            d1, d2 = 8., 16.
        main[frame] = make_reading(frame, 1, 1, [(1., d1, 1.), (2., d2, 1.)])
        parallel[frame] = make_reading(frame, 1, 1, [(3., 21. if frame == "c" else 20., 1.)])
        readings.extend((main[frame], parallel[frame]))
    readings = tuple(readings)
    _, keys, _ = plane_graph._canonical_reading_index(readings)
    rows = []
    for right in frames[1:]:
        for axis in (0, 2):
            rows.append(make_match("a", right, axis, axis, 0,
                keys[id(stable["a"][axis])][1], keys[id(stable[right][axis])][1]))
        for b in (1., 2.):
            rows.append(make_match("a", right, 1, 1, 0, keys[id(main["a"])][1],
                keys[id(main[right])][1], left_b=b, right_b=b))
        rows.append(make_match("a", right, 1, 1, 0, keys[id(parallel["a"])][1],
            keys[id(parallel[right])][1], left_b=3., right_b=3.))
    return readings, tuple(rows), frames, (("a", "b", 0), ("a", "c", 0))


@pytest.mark.parametrize("kind", ["unique", "no", "multiple"])
def test_forced_parallel_physical_pruning_and_complete_source_proof_match_reference(kind, monkeypatch):
    data = parallel_source(kind)
    new = solve(data)
    assert new.status == kind
    if kind != "multiple":
        assert new.counters["partial_prunes"] > 0
    monkeypatch.setattr(csp, "_propagate_partial_arcs", reference_sweep)
    old = solve(data)
    assert semantic(new) == semantic(old)
    assert new.counters["partial_prunes"] == old.counters["partial_prunes"]


def test_full_mode_never_calls_partial_queue_and_keeps_legacy_counter_shape(monkeypatch):
    data = fixture(complete=True)
    before, evidence = certificate(data, mode="full")
    monkeypatch.setattr(csp, "_propagate_partial_arcs", lambda *a: pytest.fail("full mode entered queue"))
    after, _ = certificate(data, mode="full")
    assert before.result_sha256 == after.result_sha256
    assert selection._payload(before) == selection._payload(after)
    assert not set(csp.PARTIAL_ARC_COUNTERS).intersection(after.proof_record()["counters"])


def test_partial_proof_binds_scan_counters_and_rejects_resealed_false_accounting():
    data = fixture()
    cert, evidence = certificate(data)
    counts = cert.proof_record()["counters"]
    assert counts["arc_relation_visits"] > 0 and counts["arc_factor_scans"] > 0
    selection.validate_candidate_certificate(cert, data[0], evidence, data[1])
    proof = cert.proof_record()
    proof["counters"]["arc_relation_visits"] = True
    with pytest.raises(ValueError):
        replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
    proof = cert.proof_record()
    proof["counters"]["arc_relation_visits"] -= 1
    tampered = replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
    with pytest.raises(ValueError):
        selection.validate_candidate_certificate(tampered, data[0], evidence, data[1])
