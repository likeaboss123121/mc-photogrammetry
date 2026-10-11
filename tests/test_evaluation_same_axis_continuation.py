from __future__ import annotations

import numpy as np
from dataclasses import replace
import hashlib
import json
import time

import pytest

from mc_photogrammetry import evaluation_same_axis_continuation as continuation
from mc_photogrammetry.reconstruction import lattice_planes


def test_source_digest_uses_authentic_distance_alternative_support():
    reading = lattice_planes.PlaneReading(
        frame="a", axis=1, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 3.0, 0.75),),
        period_pixels=8.0, phase_pixels=(0.0, 0.0), extent=1.0,
        samples=17, polygon=None, coverage_cells=4,
        structural_share=0.6, confidence=0.8,
        rectification=lattice_planes.PlaneRectification(
            tuple(tuple(float(value) for value in row) for row in np.eye(3)),
            12.0, (16.0, 16.0), (0.0, 0.0)))
    cells = np.arange(16, dtype=np.float64).reshape(4, 4)
    cells.setflags(write=False)
    coverage = np.ones((4, 4), dtype=bool)
    coverage.setflags(write=False)
    signature = lattice_planes.PlaneSignature(
        reading=reading, blocks_per_period=1.0, cells=cells,
        coverage=coverage, interior=coverage.copy(), origin_uv=(0.0, 0.0))
    digest = continuation._source_sha((reading,), (signature,))
    assert len(digest) == 64
    assert digest == continuation._source_sha((reading,), (signature,))


def _b0prime_raw_case(*, include_prime=True):
    """Fresh P0 and raw signatures; final canonical identities precede P0."""
    import test_plane_placement as fixture

    points = {"A": (0, 0, 0), "B": (1, 3, 4)}
    specs = (("A", 0, .5, 5.), ("A", 0, 1., 10.),
             ("A", 1, 1., 10.), ("A", 2, 1., 10.),
             ("B", 0, .5, 4.), ("B", 0, 1., 8.),
             ("B", 0, 1., 4.), ("B", 1, 1., 7.),
             ("B", 2, 1., 6.))
    if not include_prime:
        specs = tuple(s for s in specs if s != ("B", 0, 1., 4.))
    bases = tuple(fixture._reading(
        frame, axis, b, distance, 1., (0, 0), (0., 0.),
        samples=128, extent=127. * b / (8. * distance), focal=30.)
        for frame, axis, b, distance in specs)
    ordered = tuple(sorted(bases, key=lambda r: (r.frame,
                            lattice_planes.reading_order_key(r))))
    patch = np.zeros((16, 16), dtype=np.float64)
    patch[0, 0] = 1.
    patch[8, 8] = patch[12, 12] = 4.
    p0_budget = fixture._xyz_p0_budget()
    readings, signatures, sources, image_hashes = [], [], [], []
    for base in ordered:
        _, index = fixture._canonical_identity(ordered, base)
        geometry = lattice_planes.SplitPhaseGeometry(
            centre=base.rectification.centre, extent=base.extent,
            samples=base.samples,
            rectified_pixel_step=lattice_planes.lattice.rectified_pixel_step(
                base.extent, base.samples), period_pixels=base.period_pixels)
        generated = lattice_planes.produce_split_phase_evidence(
            patch, geometry=geometry, frame=base.frame, plane_index=index,
            reading_axis=base.axis,
            source_region_id=base.region_evidence.source_region_id,
            partition_config=lattice_planes.SPLIT_PHASE_PARTITION_CONFIG,
            budget=p0_budget)
        assert generated.status == "complete", generated.refusals
        assert lattice_planes.validate_split_phase_evidence(
            generated, patch, geometry=geometry,
            partition_config=lattice_planes.SPLIT_PHASE_PARTITION_CONFIG,
            budget=p0_budget)
        phase = fixture._g_evidence(generated)
        assert phase.phase_uv_pixels == (0., 0.)
        reading = replace(base, phase_evidence=phase,
                          phase_pixels=phase.phase_uv_pixels)
        translation = np.asarray(tuple(points[base.frame][axis]
            for axis in (0, 1, 2) if axis != base.axis), dtype=np.float64)
        image = fixture._raw_fixture_image(reading, translation)
        signature = lattice_planes.plane_signature(
            image, reading, reading.alternatives[0].blocks_per_period,
            usable=fixture._mask_for(reading.axis), max_cells=32)
        readings.append(reading)
        signatures.append(signature)
        sources.append((generated, patch.copy(), p0_budget))
        image_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
    readings = tuple(readings)
    assert tuple((r.frame, fixture._canonical_identity(readings, r)[1])
                 for r in readings) == tuple(
        (r.frame, fixture._canonical_identity(ordered, r)[1]) for r in ordered)
    return readings, tuple(signatures), tuple(sources), tuple(image_hashes)


def _b0prime_work_plan(*, max_labels=21):
    """Reject the declared finite experiment before any producers or output."""
    planned = {"labels": 21, "primary_solver_calls": 23,
               "quadrant_assignments": 92, "induced_rows": 231,
               "downstream_chains": 21, "nested_producer_calls": 1000,
               "encoded_bytes": 2_000_000, "elapsed_seconds": 60}
    if max_labels < planned["labels"]:
        return continuation.EvalArmRefusal("budget_exhausted", (
            ("stage", "global_plan"), ("labels", planned["labels"]),
            ("max_labels", max_labels)))
    return planned


def _b0prime_source_join(label_sources, claimed):
    keys = tuple((s[0].reading_identity["frame"],
                  s[0].reading_identity["plane_index"])
                 for s in label_sources)
    if len(set(keys)) != len(keys) or set(keys) != set(claimed):
        return continuation.EvalArmRefusal("p1b_source_mismatch", (
            ("stage", "exact_active_source_join"),))
    return keys


def _b0prime_labels(index):
    from itertools import combinations
    fixed = {key for key, reading in index.items() if reading.axis != 0}
    def subsets(frame):
        values = tuple(sorted(key for key, reading in index.items()
                              if reading.frame == frame and reading.axis == 0))
        return tuple(s for size in range(1, len(values) + 1)
                     for s in combinations(values, size))
    return tuple((left, right, {
        key: index[key].alternatives[0].blocks_per_period
        for key in sorted(fixed | set(left) | set(right))})
        for left in subsets("A") for right in subsets("B"))


def test_b0prime_global_cap_is_atomic_before_producers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a capped plan must not invoke a raw producer")
    monkeypatch.setattr(lattice_planes, "produce_split_phase_evidence", forbidden)
    refusal = _b0prime_work_plan(max_labels=20)
    assert isinstance(refusal, continuation.EvalArmRefusal)
    assert refusal.reason == "budget_exhausted"
    assert dict(refusal.detail)["stage"] == "global_plan"


def _b0prime_chain(readings, signatures, sources, solution, induced,
                  *, source_check=None):
    """Use complete canonical readings and precisely the label's P0 sources.

    V2-B binds every supplied P0 source to P1-B incident evidence. Inactive
    readings remain in the canonical index universe, but are not P0 sources
    for this label (plane_phase_anchor._build_bindings).
    """
    import test_plane_placement as fixture
    matches = tuple(m for m in induced
        if (solution.quadrants[m.right] - solution.quadrants[m.left]) % 4
        == m.quadrant_of_right)
    claimed = {}
    for m in matches:
        for key, b in (((m.left, m.left_plane), m.left_b),
                       ((m.right, m.right_plane), m.right_b)):
            assert key not in claimed or claimed[key] == b
            claimed[key] = b
    assert claimed == solution.blocks_per_period
    label_sources = tuple(s for s in sources if (
        s[0].reading_identity["frame"], s[0].reading_identity["plane_index"])
        in claimed)
    assert len(label_sources) == len(claimed)
    source_keys = _b0prime_source_join(label_sources, claimed)
    assert isinstance(source_keys, tuple), source_keys
    record = {"active_rows_sha256": continuation._sha([
        fixture.plane_graph._indexed_match_payload(m) for m in matches]),
        "source_count": len(label_sources), "source_keys": source_keys,
        "b_map": [(frame, index, b) for (frame, index), b in sorted(claimed.items())]}
    support = fixture.plane_axis_support.build_axis_support(
        solution, readings, matches, signatures)
    assert support.refusal is None, support.refusal
    record["c1_sha256"] = continuation._sha(support.to_record())
    p1d = fixture.plane_shift_domain.build_shift_domains(
        support, matches, signatures, max_shift_cells=4,
        budget=fixture._shift_budget(max_domains=16), readings=readings)
    assert p1d.status == "complete", p1d.refusals
    record["p1d_sha256"] = continuation._sha(p1d.to_record())
    phases = tuple(fixture._phase_for(m, readings) for m in matches)
    assert all(p.refusal is None for p in phases), phases
    record["p1b_sha256"] = continuation._sha([p.to_record() for p in phases])
    vectors = fixture.plane_vector_domain.build_lateral_vector_domains(
        support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        budget=fixture._large_budget())
    assert vectors.status == "complete", vectors.refusals
    record["vector_sha256"] = vectors.result_sha256
    relative = fixture.relative.solve_relative_translation_graph(
        vectors, support, p1d, matches, phases, readings=readings,
        signatures=signatures, comparison_tolerance=1e-12,
        closure_tolerance=0., budget=fixture._runtime_budget(
            max_components=10, max_frames=10, max_edges=16,
            max_tree_edges=10, max_chords=16, max_cycle_rows=48))
    record["c2rb"] = relative.to_record()
    endpoints = fixture.v2a.analyze_endpoint_provenance(
        relative, vectors, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.,
        budget=fixture._v2a_budget())
    record["v2a"] = endpoints.to_record()
    anchor = fixture.v2b.analyze_phase_anchor(
        relative, vectors, support, p1d, matches, phases,
        readings=readings, signatures=signatures,
        comparison_tolerance=1e-12, closure_tolerance=0.,
        v2a_result=endpoints, split_sources=label_sources,
        budget=fixture._anchor_budget())
    record["v2b"] = anchor.to_record()
    if source_check is not None:
        source_check(fixture, relative, vectors, support, p1d, matches,
                     phases, endpoints, label_sources)
    return record


def test_b0prime_raw_endpoint_proof(tmp_path, monkeypatch):
    """Diagnostic endpoint proof; every eligible identity label stays separate."""
    from collections import Counter
    from mc_photogrammetry.reconstruction import plane_graph
    import test_plane_placement as fixture
    started = time.monotonic()
    planned = _b0prime_work_plan()
    assert isinstance(planned, dict)
    nested_calls = Counter()
    # Charge actual producer/replay calls across the whole experiment, including
    # the reconstruction calls made inside validators. Nothing is published
    # until all caps and all labels finish.
    for module, name in (
        (lattice_planes, "produce_split_phase_evidence"),
        (lattice_planes, "validate_split_phase_evidence"),
        (lattice_planes, "plane_signature"),
        (plane_graph, "match_plane_graph"),
        (plane_graph, "solve_component"),
        (fixture.plane_axis_support, "build_axis_support"),
        (fixture.plane_shift_domain, "build_shift_domains"),
        (fixture.plane_vector_domain, "build_lateral_vector_domains"),
        (fixture.relative, "solve_relative_translation_graph"),
        (fixture.v2a, "analyze_endpoint_provenance"),
        (fixture.v2b, "analyze_phase_anchor"),
    ):
        original_function = getattr(module, name)
        def charged(*args, _function=original_function, _name=name, **kwargs):
            nested_calls[_name] += 1
            assert sum(nested_calls.values()) <= planned["nested_producer_calls"]
            assert time.monotonic() - started < planned["elapsed_seconds"]
            return _function(*args, **kwargs)
        monkeypatch.setattr(module, name, charged)

    # Reproduce the original dropped row using a fresh eight-reading producer
    # universe, then add B0' and regenerate all records in its final order.
    original_readings, original_signatures, _, _ = _b0prime_raw_case(
        include_prime=False)
    old_sink = []
    plane_graph.match_plane_graph(
        original_readings, original_signatures, max_shift_cells=4,
        candidate_observation_sink=old_sink)
    original = next((m, d) for m, d in zip(
        old_sink[0].candidates, old_sink[0].dispositions)
        if (m.left_plane, m.right_plane, m.left_b, m.right_b,
            m.quadrant_of_right) == (0, 0, .5, .5, 0))
    assert original[1] == plane_graph.SAME_AXIS_DISPOSITION_DROPPED

    readings, signatures, sources, image_hashes = _b0prime_raw_case()
    assert len(readings) == len(signatures) == len(sources) == 9
    observations = []
    canonical = plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=4,
        candidate_observation_sink=observations)
    assert len(observations) == 1
    pool = observations[0]
    assert len(pool.candidates) == 11
    assert {m.quadrant_of_right for m in pool.candidates} == {0, 1, 2, 3}
    kept = tuple(m for m, d in zip(pool.candidates, pool.dispositions)
                 if d == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
    assert canonical == kept
    assert canonical == plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=4)
    reversed_sink = []
    assert canonical == plane_graph.match_plane_graph(
        tuple(reversed(readings)), tuple(reversed(signatures)),
        max_shift_cells=4, candidate_observation_sink=reversed_sink)
    assert reversed_sink == observations
    assert pool.candidates[0] == original[0]
    assert pool.dispositions[0] == plane_graph.SAME_AXIS_DISPOSITION_DROPPED
    bridge = next(m for m in pool.candidates
                  if (m.left_plane, m.right_plane, m.left_b, m.right_b,
                      m.quadrant_of_right) == (0, 2, .5, 1., 0))
    assert bridge.cell_shift_rc == original[0].cell_shift_rc == (4, 3)
    index = {(r.frame, fixture._canonical_identity(readings, r)[1]): r
             for r in readings}
    assert index[("B", 2)].axis == 0
    assert index[("B", 2)].alternatives[0].blocks_per_period == 1.
    assert index[("B", 2)].phase_pixels == (0., 0.)
    assert plane_graph._delta(bridge, index) == 1.
    assert plane_graph._delta(original[0], index) == 1.

    # No constraints come from a candidate's quadrant. All rival quadrants
    # reach the unchanged .02 ambiguity check for every identity/b label.
    gates = dict(integer_tolerance=.15, min_resultant=.90,
                 ambiguous_margin=.02, quadrant_constraints=())
    whole = plane_graph.solve_component(readings, pool.candidates, **gates)
    baseline = plane_graph.solve_component(readings, canonical, **gates)
    assert whole.reason == "ambiguous_quadrant"
    assert {a.quadrants for a in whole.alternatives} == {
        (0, 0), (0, 1), (0, 2), (0, 3)}

    # Finite model: all 3 nonempty A-axis-0 subsets by all 7 B-axis-0
    # subsets, each with all four axis-1/2 readings. Singleton b alternatives
    # give exactly one complete b map per identity subset. Every pool row
    # induced by that map is retained; no arbitrary row subsets are tested.
    declared_labels = _b0prime_labels(index)
    assert declared_labels == _b0prime_labels(dict(reversed(tuple(index.items()))))
    assert len({(left, right) for left, right, _ in declared_labels}) == 21
    labels = []
    joined_sources_checked = []
    def check_sources(fixture, relative, vectors, support, p1d, matches,
                      phases, endpoints, label_sources):
        # Authentic extra producer source is rejected without rebinding.
        extra = tuple(s for s in sources
                      if all(s is not value for value in label_sources))
        assert extra
        claimed = {(m.left, m.left_plane) for m in matches} | {
            (m.right, m.right_plane) for m in matches}
        for invalid in (label_sources[:-1], label_sources + (extra[0],),
                        label_sources + (label_sources[0],)):
            joined = _b0prime_source_join(invalid, claimed)
            assert isinstance(joined, continuation.EvalArmRefusal)
            assert joined.reason == "p1b_source_mismatch"
        common = dict(readings=readings, signatures=signatures,
                      comparison_tolerance=1e-12, closure_tolerance=0.,
                      v2a_result=endpoints, budget=fixture._anchor_budget())
        refused = fixture.v2b.analyze_phase_anchor(
            relative, vectors, support, p1d, matches, phases,
            split_sources=label_sources + (extra[0],), **common)
        assert refused.status == "refused"
        assert refused.refusals[0].reason == "p1b_source_mismatch"
        joined_sources_checked.append(refused.to_record())
    for left, right, b_map in declared_labels:
            induced = tuple(m for m in pool.candidates
                if b_map.get((m.left, m.left_plane)) == m.left_b
                and b_map.get((m.right, m.right_plane)) == m.right_b)
            result = plane_graph.solve_component(readings, induced, **gates)
            label = {"a_axis0": left, "b_axis0": right,
                     "b_map": [(f, i, b) for (f, i), b in b_map.items()],
                     "induced_pool_rows": [i for i, m in enumerate(pool.candidates)
                                           if m in induced],
                     "solver_reason": result.reason,
                     "solver_alternatives": [
                         {"quadrants": a.quadrants, "cost": a.cost}
                         for a in result.alternatives]}
            if not result.reason:
                assert time.monotonic() - started < 60.
                label["chain"] = _b0prime_chain(
                    readings, signatures, sources, result, induced,
                    source_check=(check_sources if
                        left == (("A", 0),) and
                        right == (("B", 0), ("B", 2)) else None))
            labels.append(label)
    assert len(labels) == 21  # one solve per label, plus full pool and baseline
    assert nested_calls["solve_component"] == planned["primary_solver_calls"]
    assert sum(len(l["induced_pool_rows"]) for l in labels) <= planned["induced_rows"]
    assert Counter(l["solver_reason"] for l in labels) == {
        "": 12, "ambiguous_quadrant": 6, "insufficient_axis_coverage": 3}
    by_label = {(l["a_axis0"], l["b_axis0"]): l for l in labels}
    assert len(by_label) == len(labels)
    target = by_label[(("A", 0),), (("B", 0), ("B", 2))]
    control = by_label[(("A", 0),), (("B", 0),)]
    assert control["chain"]["v2a"]["classification"] == "endpoint_multiple"
    assert target["chain"]["c2rb"]["placement_readiness"] == "ready"
    assert target["chain"]["v2a"]["classification"] == "endpoint_unique"
    assert target["chain"]["v2a"]["exhaustive"] is True
    assert target["chain"]["v2a"]["endpoint_assignment_count"] == 1
    assert target["chain"]["v2b"]["status"] == "partial"
    assert [(a["status"], a["reason"]) for a in
            target["chain"]["v2b"]["prediction"]["axes"]] == [
        ("determined", None), ("abstained", "no_seam_evidence"),
        ("abstained", "no_seam_evidence")]
    supported = [l for l in labels if l.get("chain", {}).get(
        "v2b", {}).get("status") == "determined"]
    print("OUTCOMES", [(l["a_axis0"], l["b_axis0"],
        l.get("chain", {}).get("c2rb", {}).get("placement_readiness"),
        l.get("chain", {}).get("v2a", {}).get("classification"),
        l.get("chain", {}).get("v2b", {}).get("status")) for l in labels])
    assert [(l["a_axis0"], l["b_axis0"]) for l in supported] == [
        ((("A", 1),), (("B", 1),))]
    assert Counter(l["chain"]["v2b"]["status"] for l in labels
                   if "chain" in l) == {"partial": 3, "abstained": 8,
                                        "determined": 1}
    assert sum(l["chain"]["c2rb"]["placement_readiness"] == "review_required"
               for l in labels if "chain" in l) == 7
    assert len(joined_sources_checked) == 1
    endpoint_frontier = [l for l in labels if l.get("chain", {}).get(
        "v2a", {}).get("classification") == "endpoint_unique"]
    assert len(endpoint_frontier) == 4
    # Two authentic endpoint successes at the same translation remain distinct
    # evidence labels. Their strict-subset relation cannot select a winner.
    endpoint_pair = (
        by_label[(("A", 0),), (("B", 2),)], target)
    assert all(l in endpoint_frontier for l in endpoint_pair)
    assert endpoint_pair[0]["b_map"] != endpoint_pair[1]["b_map"]
    assert len({(l["a_axis0"], l["b_axis0"]) for l in endpoint_pair}) == 2
    # Half-b D evidence is balanced: the true endpoint has zero contrast.
    observations = target["chain"]["v2b"]["prediction"]["observations"]
    half_true = [o for o in observations
                 if o["selected_b"] == [1, 2]
                 and o["hypothesis_sha256"] in {
                     h["hypothesis_sha256"] for h in
                     target["chain"]["v2b"]["prediction"]["hypotheses"]
                     if (h["numerator"], h["denominator"]) == (0, 1)}]
    assert half_true and all(o["margin"] == [0, 1] for o in half_true)
    record = {"source_sha256": continuation._source_sha(readings, signatures),
              "raw_image_sha256s": image_hashes,
              "p0_sha256s": [s[0].result_sha256 for s in sources],
              "pool": [dict(plane_graph._indexed_match_payload(m), disposition=d)
                       for m, d in zip(pool.candidates, pool.dispositions)],
              "canonical_solver_reason": baseline.reason,
              "whole_pool_solver_reason": whole.reason, "labels": labels,
              "supported_label_count": len(supported),
              "extra_source_control": joined_sources_checked,
              "work_plan": planned, "producer_replay_calls": dict(nested_calls),
              "endpoint_frontier_count": len(endpoint_frontier),
              "scope": "test-only synthetic raw-producer endpoint diagnostic",
              "source_contract": "all nine canonical readings/signatures; exact active P0 source join",
              "stop_reason": "target V2-B partial; X determined; Y/Z no_seam_evidence; half-b true-hypothesis D margin zero",
              "geometry": "unavailable; placement and volume were not called",
              "limits": "finite declared labels only; no geometry truth or production promotion",
              "decision": "stop_at_target_v2b_partial"}
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    assert len(encoded) <= 2_000_000
    assert time.monotonic() - started < 60.
    artifact = tmp_path / "b0prime-proof.json"
    artifact.write_bytes(encoded)
    print("B0PRIME_ARTIFACT", str(artifact), len(encoded))
    print("B0PRIME_PROOF_SHA256", continuation._sha(record))
    print("B0PRIME_TARGET", json.dumps({name: {
        key: value for key, value in target["chain"][name].items()
        if key in ("status", "placement_readiness", "classification",
                   "exhaustive", "endpoint_assignment_count", "result_sha256")}
        for name in ("c2rb", "v2a", "v2b")}, sort_keys=True))
