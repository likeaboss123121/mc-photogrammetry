"""Relation-model controls only; no image producer, source pixels or truth input."""
from dataclasses import replace
import hashlib
import itertools
import json
import math

import numpy as np
import pytest

from scripts.evaluation import endpoint_association_counterfactual as subject
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph


def _reading(frame, axis, distance, period, alternatives=None):
    options = ((1.0, distance),) if alternatives is None else alternatives
    return lattice_planes.PlaneReading(frame, axis, 1.0,
        tuple(lattice_planes.DistanceAlternative(float(b), float(d), 1.0) for b, d in options),
        float(period), (0.0, 0.0), 1.0, 32, None, 20, 0.6, 0.9,
        lattice_planes.PlaneRectification(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                                         500.0, (120.0, 90.0), (0.0, 0.0)))


def _evidence(readings):
    evidence = []
    for index, r in enumerate(readings):
        for a, alternative in enumerate(r.alternatives):
            cells = np.arange(64, dtype=np.float32).reshape(8, 8)
            sig = lattice_planes.PlaneSignature(r, alternative.blocks_per_period,
                signatures_mod._readonly_array(cells), signatures_mod._readonly_array(cells * 0),
                signatures_mod._readonly_array(np.ones_like(cells, dtype=bool)), (0.0, 0.0))
            evidence.append(signatures_mod.SignatureEvidence(r.frame, index, r.axis, a,
                                                            alternative.blocks_per_period, sig))
    return tuple(evidence)


def _row(readings, left, right, lb=1.0, rb=1.0, q=0):
    _, identities, _ = plane_graph._canonical_reading_index(readings)
    return plane_graph.PlaneMatch(left.frame, right.frame, left.axis, right.axis,
        float(lb), float(rb), q, 5.0, 1.0, 0.1, 64, None, 0.7,
        identities[id(left)][1], identities[id(right)][1], (0, 0))


def _packet(readings, rows, *, constraints=None, signatures=None):
    readings, rows = tuple(readings), tuple(rows)
    frames = tuple(sorted({r.frame for r in readings}))
    return dict(readings=readings, signatures=_evidence(readings) if signatures is None else signatures,
        universe=plane_graph.AdmittedCandidateUniverse(rows, ("kept",) * len(rows), tuple(range(len(rows)))),
        frame_ids=frames, quadrant_constraints=(tuple((frames[0], f, 0) for f in frames[1:])
            if constraints is None else constraints), session_sha256="a" * 64,
        lattice_sha256="b" * 64, signatures_sha256="c" * 64)


def _fixture(*, parallel=False, coplanar=False, alternatives=False):
    readings = []
    for frame, base in (("a", 10.0), ("b", 8.0)):
        for axis in range(3):
            alts = ((1.0, base), (2.0, base * 2)) if alternatives else None
            readings.append(_reading(frame, axis, base, 10 + axis, alts))
        if parallel or coplanar:
            readings.append(_reading(frame, 1, base if coplanar else base + 4, 21))
    readings = tuple(readings)
    rows = []
    for left in readings:
        for right in readings:
            if left.frame != "a" or right.frame != "b" or left.axis != right.axis: continue
            for b in ((1, 2) if alternatives else (1,)):
                rows.append(_row(readings, left, right, b, b))
    return _packet(readings, rows)


def _semantic(result):
    """Ignore original source-row numbering, never factor/alternative identity."""
    return (result["status"], result["optimal_unknown_factor_count"],
        tuple(sorted((tuple(map(tuple, m["quadrants"])), tuple(map(tuple, m["alternatives"])),
                      tuple(m["associated_factors"])) for m in result["modes"])), result["membership"])


def _oracle(packet):
    """Independent tiny exhaustive enumeration; no helper search or CSP calls.

    Fixed-gauge Q, complete alternative products and every factor activation
    mask are enumerated. Coverage uses independent traversal; cycles use a
    Floyd-Warshall interval closure instead of production Bellman relaxation.
    """
    readings, rows = packet["readings"], packet["universe"].candidates
    indexed, _, frames = plane_graph._canonical_reading_index(readings)
    endpoints = tuple(sorted(indexed))
    pairs = tuple(sorted({tuple(sorted(((r.left, r.left_plane), (r.right, r.right_plane)))) for r in rows}))
    transforms = (((0, 1), (1, 1), (2, 1)), ((2, 1), (1, 1), (0, -1)),
                  ((0, -1), (1, 1), (2, -1)), ((2, -1), (1, 1), (0, 1)))
    best, modes = None, set()
    for qvalues in itertools.product(range(4), repeat=len(frames) - 1):
        q = dict(zip(frames, (0,) + qvalues))
        if any((q[r] - q[l]) % 4 != v for l, r, v in packet["quadrant_constraints"]): continue
        for values in itertools.product(*(range(len(indexed[e].alternatives)) for e in endpoints)):
            selected = dict(zip(endpoints, values))
            available = {}
            for fid, (u, v) in enumerate(pairs):
                relations = []
                for row in rows:
                    left, right = (row.left, row.left_plane), (row.right, row.right_plane)
                    if tuple(sorted((left, right))) != (u, v) or (q[row.right] - q[row.left]) % 4 != row.quadrant_of_right: continue
                    lr, rr = indexed[left], indexed[right]
                    la, ra = lr.alternatives[selected[left]], rr.alternatives[selected[right]]
                    if abs(la.blocks_per_period-row.left_b) > 1e-9 or abs(ra.blocks_per_period-row.right_b) > 1e-9: continue
                    axis, sign = transforms[q[row.left]][lr.axis]
                    other, right_sign = transforms[q[row.right]][rr.axis]
                    if axis != other: continue
                    delta = sign*lr.side*la.distance_blocks - right_sign*rr.side*ra.distance_blocks
                    if row.right < row.left: delta = -delta
                    relations.append(((u[0], v[0], axis), delta))
                if relations: available[fid] = relations
            for mask in range(1 << len(pairs)):
                active = tuple(i for i in range(len(pairs)) if mask & (1 << i))
                if any(i not in available for i in active): continue
                supported = {e for i in active for e in pairs[i]}
                if supported != set(endpoints): continue
                groups, adjacency = {}, {axis: {f: set() for f in frames} for axis in range(3)}
                for i in active:
                    for key, delta in available[i]:
                        groups.setdefault(key, []).append(delta)
                        f, g, axis = key
                        adjacency[axis][f].add(g); adjacency[axis][g].add(f)
                eligible = True
                for axis in range(3):
                    reached, pending = {frames[0]}, [frames[0]]
                    while pending:
                        f = pending.pop()
                        for g in adjacency[axis][f]-reached: reached.add(g); pending.append(g)
                    if reached != set(frames): eligible = False
                if not eligible or any(max(v)-min(v) > 0.15 for v in groups.values()): continue
                feasible = True
                for axis in range(3):
                    distance = {(f,g): (0.0 if f == g else math.inf) for f in frames for g in frames}
                    for (f,g,world), deltas in groups.items():
                        if world == axis:
                            distance[(f,g)] = min(distance[(f,g)], min(deltas)+0.15)
                            distance[(g,f)] = min(distance[(g,f)], -max(deltas)+0.15)
                    for k in frames:
                        for f in frames:
                            for g in frames: distance[(f,g)] = min(distance[(f,g)], distance[(f,k)]+distance[(k,g)])
                    if any(distance[(f,f)] < 0 for f in frames): feasible = False
                if not feasible: continue
                unknown = len(pairs)-len(active)
                mode = (tuple(q.items()), tuple((f,i,selected[(f,i)]) for f,i in endpoints), active)
                if best is None or unknown < best: best, modes = unknown, {mode}
                elif unknown == best: modes.add(mode)
    return best, modes


def test_consistent_full_xyz_reproduces_existing_no_removal_unique_assignment():
    packet = _fixture()
    original = csp.solve_endpoint_hypotheses(packet["readings"], packet["universe"].candidates,
        session_frames=packet["frame_ids"], quadrant_constraints=packet["quadrant_constraints"])
    result = subject.run_counterfactual(**packet)
    assert original.status == "unique"
    assert result["status"] == "robust_hypothesis_only" and result["optimal_unknown_factor_count"] == 0
    assert result["modes"][0]["alternatives"] == [list(x) for x in original.choices[0].alternatives]
    assert result["membership"] == {"mandatory": [0,1,2], "possible": [0,1,2], "never_selected": []}
    assert result["production_certificate"] is False and result["counters"]["replay_modes"] == 1


def test_distinct_parallel_planes_diagonal_only_unique_full_coverage():
    packet = _fixture(parallel=True)
    original = csp.solve_endpoint_hypotheses(packet["readings"], packet["universe"].candidates,
        session_frames=packet["frame_ids"], quadrant_constraints=packet["quadrant_constraints"])
    result = subject.run_counterfactual(**packet)
    assert original.status == "no" and original.reason == "canonical_constraints_inconsistent"
    assert result["status"] == "robust_hypothesis_only" and result["optimal_unknown_factor_count"] == 2
    chosen = [result["factors"][fid] for fid in result["hypothesis"]["associated_factors"]]
    assert [f["endpoints"] for f in chosen] == [
        [["a",0],["b",0]], [["a",1],["b",1]], [["a",2],["b",2]], [["a",3],["b",3]]]
    assert result["hypothesis"]["coverage"]["unknown_endpoints"] == []
    assert len(result["factors"]) == 6 and result["source_bindings"]["candidate_count"] == 6


def test_coplanar_patches_keep_all_many_to_many_associations():
    result = subject.run_counterfactual(**_fixture(coplanar=True))
    assert result["status"] == "robust_hypothesis_only" and result["optimal_unknown_factor_count"] == 0
    assert result["membership"]["mandatory"] == list(range(6))
    assert len(result["hypothesis"]["support_rows"]) == 6


def test_symmetric_equal_alternative_optima_abstain_and_retain_every_mode():
    packet = _fixture(alternatives=True)
    result = subject.run_counterfactual(**packet)
    optimum, modes = _oracle(packet)
    assert result["status"] == "multiple_hypotheses" and result["hypothesis"] is None
    assert result["optimal_unknown_factor_count"] == optimum == 0
    assert len(result["modes"]) == len(modes) == 8


def test_one_of_two_conflicting_left_planes_supported_cannot_drop_endpoint():
    packet = _fixture(parallel=True)
    readings = tuple(r for r in packet["readings"] if not (r.frame == 'b' and r.period_pixels == 21))
    candidates = tuple(_row(readings,left,right) for left in readings for right in readings
        if left.frame == 'a' and right.frame == 'b' and left.axis == right.axis)
    packet = _packet(readings, candidates)
    result = subject.run_counterfactual(**packet)
    assert result["status"] == "coverage_or_physical_refused"
    assert result["hypothesis"] is None and result["modes"] == []
    assert result["counters"]["coverage_rejections"] > 0


def _cycle_fixture():
    readings = tuple(_reading(f, axis, 10 - 2*i, 10+axis) for i,f in enumerate(("a","b","c")) for axis in (0,2))
    y = tuple(_reading(f, 1, d, 30+j) for j,(f,d) in enumerate((('a',10),('a',11),('b',8),('b',12),('c',6),('c',20))))
    readings += y
    rows = []
    for axis in (0,2):
        by_frame = {r.frame:r for r in readings if r.axis == axis}
        rows.extend((_row(readings,by_frame['a'],by_frame['b']),_row(readings,by_frame['b'],by_frame['c'])))
    rows.extend((_row(readings,y[0],y[2]),_row(readings,y[3],y[4]),_row(readings,y[1],y[5])))
    return _packet(readings, rows)


def test_inconsistent_cycle_cannot_pass_by_losing_axis_or_endpoint_connectivity():
    packet = _cycle_fixture()
    result = subject.run_counterfactual(**packet)
    assert result["hypothesis"] is None and result["modes"] == []
    assert result["counters"]["cycle_rejections"] > 0
    assert _oracle(packet) == (None,set())


@pytest.mark.parametrize("order", ("duplicate", "reverse", "rotate"))
def test_duplicate_rows_and_permutations_do_not_change_objective_or_support(order):
    packet = _fixture(parallel=True)
    baseline = subject.run_counterfactual(**packet)
    rows = packet['universe'].candidates
    if order == 'duplicate': rows = rows + rows + rows[:1]
    elif order == 'reverse': rows = tuple(reversed(rows))
    else: rows = rows[2:]+rows[:2]
    result = subject.run_counterfactual(**_packet(packet['readings'],rows))
    assert _semantic(result) == _semantic(baseline)
    assert result['source_bindings']['candidate_count'] == len(rows)
    assert result['source_bindings']['candidate_matches_sha256'] != baseline['source_bindings']['candidate_matches_sha256']


def test_coherent_oracle_labelled_wrong_subset_remains_hypothesis_only():
    packet = _fixture(parallel=True)
    oracle_true_factor_ids = {0,2,3,5}  # frozen external label; never a selector input
    result = subject.run_counterfactual(**packet)
    assert set(result['hypothesis']['associated_factors']) != oracle_true_factor_ids
    assert result['status'] == 'robust_hypothesis_only' and result['production_certificate'] is False
    assert 'coherent_spurious_unique_subset_possible' in result['limitations']
    assert not isinstance(result,csp.EndpointResult)


@pytest.mark.parametrize("resource", ('branches','relation_visits','proof_bytes'))
def test_exact_cumulative_budget_boundary_one_below_discards_best_so_far(resource):
    packet = _fixture(parallel=True)
    result = subject.run_counterfactual(**packet)
    exact = result['counters'][resource]
    boundary = subject.run_counterfactual(**packet,budget=replace(subject.Budget(),**{resource:exact}))
    assert _semantic(boundary) == _semantic(result)
    refused = subject.run_counterfactual(**packet,budget=replace(subject.Budget(),**{resource:exact-1}))
    assert refused['status'] == 'budget_refused'
    assert refused['hypothesis'] is None and refused['optimal_unknown_factor_count'] is None
    assert refused['modes'] == [] and refused['membership'] is None and refused['exhaustive'] is False
    assert refused['denial']['resource'] == resource


@pytest.mark.parametrize('seed',range(16))
def test_independent_exhaustive_tiny_oracle_agrees_generated_cases(seed):
    rng = np.random.default_rng(seed)
    readings=[]
    for frame in ('a','b'):
        for axis in range(3):
            d=float(rng.integers(4,12))
            readings.append(_reading(frame,axis,d,10+axis,((1.0,d),(2.0,d*2))))
    readings=tuple(readings)
    rows=[]
    for axis in range(3):
        left,right=readings[axis],readings[axis+3]
        for lb,rb in itertools.product((1,2),repeat=2):
            if rng.integers(0,3): rows.append(_row(readings,left,right,lb,rb))
    packet=_packet(readings,rows)
    result=subject.run_counterfactual(**packet)
    optimum, expected=_oracle(packet)
    actual=set((tuple(map(tuple,m['quadrants'])),tuple(map(tuple,m['alternatives'])),tuple(m['associated_factors'])) for m in result['modes'])
    assert result['exhaustive'] and result['optimal_unknown_factor_count']==optimum
    assert actual==expected
    assert (result['hypothesis'] is not None)==(len(expected)==1)


def test_q_modes_reversed_rows_and_signed_sides_match_tiny_oracle():
    packet=_fixture()
    readings=tuple(replace(r,side=-1.0) for r in packet['readings'])
    rows=tuple(replace(r,left=r.right,right=r.left,left_plane=r.right_plane,right_plane=r.left_plane,
        left_axis=r.right_axis,right_axis=r.left_axis,left_b=r.right_b,right_b=r.left_b)
        for r in packet['universe'].candidates)
    packet=_packet(readings,rows)
    result=subject.run_counterfactual(**packet)
    assert result['status']=='robust_hypothesis_only'
    assert result['optimal_unknown_factor_count']==_oracle(packet)[0]


def test_nonzero_latent_q_optima_remain_equal_and_uncertain():
    packet = _fixture()
    readings = tuple(replace(r,side=-1.0) if r.frame == 'b' and r.axis != 1 else r
                     for r in packet['readings'])
    rows = packet['universe'].candidates
    rows += tuple(replace(r,quadrant_of_right=2) for r in rows)
    packet = _packet(readings,rows,constraints=())
    result = subject.run_counterfactual(**packet)
    best,modes = _oracle(packet)
    assert best == result['optimal_unknown_factor_count'] == 0
    assert result['status'] == 'multiple_hypotheses' and result['hypothesis'] is None
    assert {m['quadrants'][1][1] for m in result['modes']} == {0,2}
    assert len(result['modes']) == len(modes) == 2


@pytest.mark.parametrize('seed',range(12))
def test_generated_association_choices_match_independent_exhaustive_oracle(seed):
    rng = np.random.default_rng(seed)
    packet = _fixture(parallel=True)
    mode = seed % 3
    values = (10.0,14.0,8.0,12.0) if mode == 0 else ((10.0,10.0,8.0,8.0) if mode == 1 else
        tuple(float(x) for x in rng.integers(5,15,size=4)))
    counter = iter(values)
    readings = tuple(replace(r,alternatives=(replace(r.alternatives[0],distance_blocks=next(counter)),))
                     if r.axis == 1 else r for r in packet['readings'])
    rows = []
    for left in readings:
        for right in readings:
            if left.frame == 'a' and right.frame == 'b' and left.axis == right.axis:
                if seed < 6 or left.axis != 1 or rng.integers(0,4): rows.append(_row(readings,left,right))
    packet = _packet(readings,rows)
    result = subject.run_counterfactual(**packet)
    best, expected = _oracle(packet)
    actual = set((tuple(map(tuple,m['quadrants'])),tuple(map(tuple,m['alternatives'])),
        tuple(m['associated_factors'])) for m in result['modes'])
    assert result['exhaustive'] and result['optimal_unknown_factor_count'] == best
    assert actual == expected


@pytest.mark.parametrize('resource',subject.LIMITS)
def test_hard_global_maxima_cannot_be_increased(resource):
    with pytest.raises(ValueError,match='only decrease'):
        replace(subject.Budget(),**{resource:subject.LIMITS[resource]+1})


def test_endpoint_count_preflight_refuses_before_source_expansion():
    packet = _fixture()
    packet['readings'] = packet['readings'] * 22
    packet['signatures'] = ()
    result = subject.run_counterfactual(**packet)
    assert result['status'] == 'budget_refused' and result['denial']['resource'] == 'endpoints'
    assert result['counters']['relation_visits'] == result['counters']['proof_bytes'] == 0


def test_scores_overlap_and_duplicate_support_never_weight_factor_objective():
    packet = _fixture(parallel=True)
    baseline = subject.run_counterfactual(**packet)
    rows = tuple(replace(r,score=0.001,psr=100.0,overlap_cells=1000) for r in packet['universe'].candidates)
    changed = subject.run_counterfactual(**_packet(packet['readings'],rows+rows))
    assert _semantic(changed) == _semantic(baseline)
    assert changed['source_bindings']['retained_alternative_count'] == len(packet['signatures'])


def test_full_size_factor_universe_uses_cumulative_refusal_without_promoting_subset():
    readings = tuple(_reading(frame,1,base+i,10+i) for frame,base in (('a',10.0),('b',8.0))
                     for i in range(64))
    # The input is exactly 128 endpoints/4096 canonical factors. Its absence of
    # X/Z support cannot justify returning a partial best-so-far association.
    rows = tuple(_row(readings,left,right) for left in readings[:64] for right in readings[64:])
    result = subject.run_counterfactual(**_packet(readings,rows))
    assert result['status'] == 'budget_refused' and result['hypothesis'] is None
    assert result['modes'] == [] and result['optimal_unknown_factor_count'] is None
    assert result['counters']['registry_factors'] == 4096
    assert result['counters']['registry_rows'] == 4096
    assert all(result['counters'][k] <= subject.LIMITS[k] for k in ('branches','relation_visits','proof_bytes'))


@pytest.mark.parametrize('damage',('missing_signature','foreign_row','wrong_digest','ambiguous_reading','foreign_q'))
def test_incomplete_or_tampered_source_packet_refuses(damage):
    packet=_fixture()
    if damage=='missing_signature': packet['signatures']=packet['signatures'][1:]
    elif damage=='foreign_row':
        rows=(replace(packet['universe'].candidates[0],left_plane=99),)+packet['universe'].candidates[1:]
        packet=_packet(packet['readings'],rows)
    elif damage=='wrong_digest': packet['expected_source_digest']='0'*64
    elif damage=='ambiguous_reading':
        packet['readings']=packet['readings']+packet['readings'][:1]
    else: packet['quadrant_constraints']=(('a','b',9),)
    result=subject.run_counterfactual(**packet)
    assert result['status']=='input_refused' and result['hypothesis'] is None


def test_single_ledger_replay_never_calls_new_csp_and_has_bounded_seal(monkeypatch):
    packet=_fixture(parallel=True)
    monkeypatch.setattr(csp,'solve_endpoint_hypotheses',lambda *_a,**_k:pytest.fail('subset minted a fresh CSP budget'))
    result=subject.run_counterfactual(**packet)
    assert result['status']=='robust_hypothesis_only' and result['counters']['replay_modes']==1
    assert result['counters']['rollback_visits']>0
    record=dict(result); seal=record.pop('record_sha256')
    raw=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest()==seal
    assert len(json.dumps(result).encode())<4*1024*1024
    assert all(result['counters'][name]<=subject.LIMITS[name] for name in ('branches','relation_visits','proof_bytes','quadrant_plans'))
