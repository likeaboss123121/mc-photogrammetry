"""Synthetic D-only intervention; an exhaustive diagnostic, never a selector."""
from __future__ import annotations

import base64
from collections import Counter
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import test_plane_placement as fixture
from test_evaluation_same_axis_continuation import (
    _b0prime_chain, _b0prime_labels, _b0prime_raw_case, _b0prime_source_join,
)
from mc_photogrammetry import evaluation_same_axis_continuation as continuation
from mc_photogrammetry.reconstruction import lattice_planes as lp, plane_graph


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "tests/test_evaluation_same_axis_continuation.py":
        "68ed8036081b2ac931f7df856da0bbd7e54bc07815dbe40b1a7dafce4ffa622b",
    "src/mc_photogrammetry/evaluation_same_axis_continuation.py":
        "69483d14c4efbb5d528780541fa735881a9535f71071d0b62b9b7b73ba60feeb",
    "docs/reports/DS-IMG-B0PRIME-ENDPOINT-PROOF-20261005.json":
        "341aa61ca4d9bd1d8101361fd3ba2b17573f043f62ba28ce4072acf555df20a9",
    "docs/reports/DS-IMG-B0PRIME-ENDPOINT-PROOF-20261005.txt":
        "3ddccf70f1983493118b6b89f14667d2c61717bc44fcc1dd4091744aa7ba9184",
}
TARGET = ((("A", 0),), (("B", 0), ("B", 2)))
GATES = dict(integer_tolerance=.15, min_resultant=.90,
             ambiguous_margin=.02, quadrant_constraints=())


def _file_sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _ratio(value):
    return [value.numerator, value.denominator]


def _decode(seam):
    samples = np.frombuffer(base64.b64decode(
        seam.profile_samples_float64_le_base64), dtype="<f8")
    weights = np.frombuffer(base64.b64decode(
        seam.profile_weights_float64_le_base64), dtype="<f8")
    indices = np.frombuffer(base64.b64decode(
        seam.axis_bin_indices_int64_le_base64), dtype="<i8")
    assert len(samples) == len(weights) == len(indices) == 16
    assert np.array_equal(indices, np.arange(16))
    assert np.array_equal(weights, np.full(16, 8.))
    return samples, weights


def _regenerate_intervention(readings, signatures, sources, image_hashes):
    changed_sources, fresh_readings, fresh_signatures, fresh_hashes = [], [], [], []
    source_records = []
    points = {"A": (0, 0, 0), "B": (1, 3, 4)}
    changed = []
    for reading, signature, (old, baseline_patch, budget), image_sha in zip(
            readings, signatures, sources, image_hashes):
        patch = baseline_patch.copy()
        identity = old.reading_identity
        key = (identity["frame"], identity["plane_index"])
        half = reading.alternatives[0].blocks_per_period == .5
        assert half == (key in {("A", 0), ("B", 0)})
        if half:
            patch[8, 8], patch[12, 12] = 8., 0.
            changed.append(key)
        assert float(patch.sum()) == float(baseline_patch.sum()) == 9.
        assert np.array_equal(patch[:8, :], baseline_patch[:8, :])
        assert np.array_equal(patch[:, :8], baseline_patch[:, :8])
        assert np.array_equal(patch != baseline_patch,
            np.eye(16, dtype=bool) & np.isin(np.arange(16)[:, None], [8, 12])
            if half else np.zeros((16, 16), dtype=bool))
        assert [(s.local_axis, s.role, s.lane_index_start, s.lane_index_stop)
                for s in old.lane_sets] == [
            ("u", "G", 0, 8), ("u", "D", 8, 16),
            ("v", "G", 0, 8), ("v", "D", 8, 16)]
        fresh = lp.produce_split_phase_evidence(
            patch, geometry=old.geometry, **identity,
            source_region_id=reading.region_evidence.source_region_id,
            partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
        assert fresh.status == "complete", fresh.refusals
        assert lp.validate_split_phase_evidence(
            fresh, patch, geometry=old.geometry,
            partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
        assert fresh.reading_identity == old.reading_identity
        assert fresh.geometry == old.geometry and fresh.lane_sets == old.lane_sets
        assert fresh.generation_evidence == old.generation_evidence
        phase = fixture._g_evidence(fresh)
        assert phase == reading.phase_evidence and phase.phase_uv_pixels == (0., 0.)
        new_reading = replace(reading, phase_evidence=phase,
                              phase_pixels=phase.phase_uv_pixels)
        translation = np.asarray([points[reading.frame][axis]
            for axis in range(3) if axis != reading.axis], dtype=np.float64)
        image = fixture._raw_fixture_image(new_reading, translation)
        new_image_sha = hashlib.sha256(image.tobytes()).hexdigest()
        assert new_image_sha == image_sha
        new_signature = lp.plane_signature(
            image, new_reading, reading.alternatives[0].blocks_per_period,
            usable=fixture._mask_for(reading.axis), max_cells=32)
        assert continuation._source_sha((reading,), (signature,)) == \
            continuation._source_sha((new_reading,), (new_signature,))
        assert (fresh.result_sha256 != old.result_sha256) == half
        assert (fresh.patch_ref != old.patch_ref) == half
        profiles = {}
        for a, b in zip(old.seam_profile_evidence, fresh.seam_profile_evidence):
            before, weights = _decode(a)
            after, after_weights = _decode(b)
            assert np.array_equal(weights, after_weights)
            assert before.sum() == after.sum() == 8.
            assert before.tolist() == [4. if i in (8, 12) else 0. for i in range(16)]
            if half:
                assert after.tolist() == [8. if i == 8 else 0. for i in range(16)]
            else:
                assert np.array_equal(before, after)
            assert (a.d_profile_sha256 != b.d_profile_sha256) == half
            profiles[a.local_axis] = {"samples_baseline": before.tolist(),
                "samples_intervention": after.tolist(), "weights": weights.tolist(),
                "period_hex": a.period_pixels.hex(),
                "period_ratio": _ratio(Fraction(a.period_pixels)),
                "baseline": a.to_record(), "intervention": b.to_record()}
        source_records.append({"reading_identity": identity, "changed": half,
            "image_sha256": image_sha,
            "signature_source_sha256": continuation._source_sha((reading,), (signature,)),
            "baseline_p0_sha256": old.result_sha256,
            "intervention_p0_sha256": fresh.result_sha256,
            "baseline_patch_sha256": hashlib.sha256(baseline_patch.tobytes()).hexdigest(),
            "intervention_patch_sha256": hashlib.sha256(patch.tobytes()).hexdigest(),
            "generation": fresh.generation_evidence[0].to_record(),
            "baseline_p0": old.to_record(), "intervention_p0": fresh.to_record(),
            "profiles": profiles})
        changed_sources.append((fresh, patch, budget))
        fresh_readings.append(new_reading)
        fresh_signatures.append(new_signature)
        fresh_hashes.append(new_image_sha)
    assert changed == [("A", 0), ("B", 0)]
    return (tuple(fresh_readings), tuple(fresh_signatures), tuple(changed_sources),
            tuple(fresh_hashes), source_records)


def _energy_oracle(chain, context, intervention):
    """Independent Fraction masks from decoded D arrays, not classifier helpers."""
    phases, sources = context
    phase_by_key = {}
    for phase in phases:
        for record in phase.phase_candidates:
            key = (record.frame, record.plane_index, record.local_axis)
            if key in phase_by_key:
                assert phase_by_key[key].to_record() == record.to_record()
            phase_by_key[key] = record
    source_by_key = {(s[0].reading_identity["frame"],
                      s[0].reading_identity["plane_index"]): s[0] for s in sources}
    prediction = chain["v2b"]["prediction"]
    hypotheses = {h["hypothesis_sha256"]: Fraction(h["numerator"], h["denominator"])
                  for h in prediction["hypotheses"]}
    rows = []
    half_count = 0
    for observation in prediction["observations"]:
        identity = observation["reading_identity"]
        key = (identity["frame"], identity["plane_index"])
        axis = observation["local_axis"]
        residue = phase_by_key[key + (axis,)]
        seam = next(s for s in source_by_key[key].seam_profile_evidence
                    if s.local_axis == axis)
        samples, weights = _decode(seam)
        b = Fraction(residue.blocks_per_period)
        leading = Fraction(residue.leading_cell_coordinate)
        assert observation["selected_b"] == _ratio(b)
        alpha = hypotheses[observation["hypothesis_sha256"]]
        sign = observation["world_sign"]
        # The serialized propagation origin is sign*L+t modulo one; integer
        # translation is irrelevant to the cyclic band-energy calculation.
        origin = Fraction(*observation["propagation_origin"])
        translation_mod1 = (origin - sign * leading) % 1
        anchor = (sign * (alpha - translation_mod1)) % 1
        coordinates = [(leading + b * i / Fraction(seam.period_pixels)) % 1
                       for i in range(len(samples))]
        def distance(a, c):
            separation = abs(a - c)
            return min(separation, 1 - separation)
        boundary = [i for i, x in enumerate(coordinates)
                    if distance(x, anchor) <= Fraction(1, 8)]
        interior = [i for i, x in enumerate(coordinates)
                    if distance(x, (anchor + Fraction(1, 2)) % 1) <= Fraction(1, 8)]
        assert boundary and interior and not set(boundary) & set(interior)
        bn = sum((Fraction(float(samples[i])) for i in boundary), Fraction())
        dn = sum((Fraction(float(samples[i])) for i in interior), Fraction())
        bw = sum(int(weights[i]) for i in boundary)
        dw = sum(int(weights[i]) for i in interior)
        be, ie = bn / bw, dn / dw
        assert observation["boundary_energy"] == _ratio(be)
        assert observation["interior_energy"] == _ratio(ie)
        assert observation["margin"] == _ratio(be - ie)
        if b == Fraction(1, 2):
            half_count += 1
            assert leading == 0 and origin == 0 and alpha in (0, Fraction(1, 2))
            true_boundary = ([0, 1, 8, 9] if key[0] == "A"
                             else [0, 1, 7, 8, 9, 15])
            true_interior = ([4, 5, 12, 13] if key[0] == "A"
                             else [3, 4, 5, 11, 12, 13])
            assert (boundary, interior) == ((true_boundary, true_interior)
                if alpha == 0 else (true_interior, true_boundary))
            expected = Fraction(1, 4) if key[0] == "A" else Fraction(1, 6)
            if intervention:
                assert be - ie == (expected if alpha == 0 else -expected)
            else:
                assert be == ie == expected / 2
        rows.append({"observation": observation,
            "leading": _ratio(leading), "anchor": _ratio(anchor),
            "translation_mod1": _ratio(translation_mod1),
            "coordinates": [_ratio(x) for x in coordinates],
            "boundary_indices": boundary, "interior_indices": interior,
            "boundary_sample_sum": _ratio(bn), "boundary_weight_sum": bw,
            "interior_sample_sum": _ratio(dn), "interior_weight_sum": dw,
            "boundary_energy": _ratio(be), "interior_energy": _ratio(ie),
            "margin": _ratio(be - ie)})
    assert len(rows) == 24 and half_count == 8
    return rows


def test_constant_total_d_redistribution(tmp_path):
    started = time.monotonic()
    assert {name: _file_sha(name) for name in PINS} == PINS
    source_files = ["src/mc_photogrammetry/reconstruction/" + name + ".py"
        for name in ("lattice_planes", "plane_phase_anchor", "split_phase_calibration",
                     "plane_graph", "plane_residue", "plane_vector_domain",
                     "plane_relative_translation", "plane_endpoint_provenance")]
    source_pins = {name: _file_sha(name) for name in source_files}
    readings, signatures, sources, hashes = _b0prime_raw_case()
    fresh_readings, fresh_signatures, fresh_sources, fresh_hashes, source_records = \
        _regenerate_intervention(readings, signatures, sources, hashes)
    assert hashes == fresh_hashes and len(hashes) == 9
    source_sha = continuation._source_sha(readings, signatures)
    assert source_sha == continuation._source_sha(fresh_readings, fresh_signatures)
    sink, fresh_sink = [], []
    canonical = plane_graph.match_plane_graph(readings, signatures,
        max_shift_cells=4, candidate_observation_sink=sink)
    assert canonical == plane_graph.match_plane_graph(fresh_readings, fresh_signatures,
        max_shift_cells=4, candidate_observation_sink=fresh_sink)
    assert sink == fresh_sink and len(sink) == 1
    pool = sink[0]
    assert len(pool.candidates) == 11
    assert {m.quadrant_of_right for m in pool.candidates} == {0, 1, 2, 3}
    index = {(r.frame, fixture._canonical_identity(readings, r)[1]): r for r in readings}
    fresh_index = {(r.frame, fixture._canonical_identity(fresh_readings, r)[1]): r
                   for r in fresh_readings}
    declared = _b0prime_labels(index)
    assert declared == _b0prime_labels(fresh_index)
    assert declared == _b0prime_labels(dict(reversed(tuple(index.items()))))
    assert len(declared) == len({(a, b) for a, b, _ in declared}) == 21
    whole = plane_graph.solve_component(readings, pool.candidates, **GATES)
    canonical_result = plane_graph.solve_component(readings, canonical, **GATES)
    assert whole.reason == "ambiguous_quadrant"
    labels, contexts = [], {}
    upstream_changes = []
    for left, right, b_map in declared:
        induced = tuple(m for m in pool.candidates
            if b_map.get((m.left, m.left_plane)) == m.left_b
            and b_map.get((m.right, m.right_plane)) == m.right_b)
        solved = plane_graph.solve_component(readings, induced, **GATES)
        record = {"a_axis0": left, "b_axis0": right,
            "b_map": [(f, i, b) for (f, i), b in b_map.items()],
            "induced_pool_rows": [i for i, m in enumerate(pool.candidates) if m in induced],
            "solver_reason": solved.reason,
            "solver_alternatives": [{"quadrants": a.quadrants, "cost": a.cost}
                                    for a in solved.alternatives]}
        if not solved.reason:
            def capture(_fixture, relative, vectors, support, domains, matches,
                        phases, endpoints, active_sources):
                claimed = {(m.left, m.left_plane) for m in matches} | {
                    (m.right, m.right_plane) for m in matches}
                assert set(solved.blocks_per_period) == set(b_map) == claimed
                assert isinstance(_b0prime_source_join(active_sources, claimed), tuple)
                extra = tuple(s for s in sources if (
                    s[0].reading_identity["frame"], s[0].reading_identity["plane_index"])
                    not in claimed)
                invalids = [active_sources[:-1], active_sources + (active_sources[0],)]
                if extra:
                    invalids.append(active_sources + (extra[0],))
                assert all(isinstance(_b0prime_source_join(s, claimed),
                                      continuation.EvalArmRefusal) for s in invalids)
                contexts[(left, right, capture.variant)] = (phases, active_sources)
            capture.variant = "baseline"
            record["baseline"] = _b0prime_chain(readings, signatures, sources,
                solved, induced, source_check=capture)
            capture.variant = "intervention"
            record["intervention"] = _b0prime_chain(fresh_readings, fresh_signatures,
                fresh_sources, solved, induced, source_check=capture)
            for name in ("active_rows_sha256", "source_count", "source_keys", "b_map",
                         "c1_sha256", "p1d_sha256", "p1b_sha256", "vector_sha256",
                         "c2rb", "v2a"):
                if record["baseline"][name] != record["intervention"][name]:
                    upstream_changes.append((left, right, name))
        labels.append(record)
    assert not upstream_changes
    assert Counter(l["solver_reason"] for l in labels) == {
        "": 12, "ambiguous_quadrant": 6, "insufficient_axis_coverage": 3}
    by_label = {(l["a_axis0"], l["b_axis0"]): l for l in labels}
    target = by_label[TARGET]
    assert target["baseline"]["v2b"]["status"] == "partial"
    assert target["intervention"]["v2b"]["status"] == "determined"
    assert [(a["status"], a["reason"]) for a in
            target["baseline"]["v2b"]["prediction"]["axes"]] == [
        ("determined", None), ("abstained", "no_seam_evidence"),
        ("abstained", "no_seam_evidence")]
    assert all(a["status"] == "determined" and a["alpha_dyadic"] == [0, 1]
        for a in target["intervention"]["v2b"]["prediction"]["axes"])
    mask_oracles = {variant: _energy_oracle(target[variant],
        contexts[TARGET + (variant,)], variant == "intervention")
        for variant in ("baseline", "intervention")}
    frontier = {}
    for variant in ("baseline", "intervention"):
        supported = [key for key, l in by_label.items()
                     if l.get(variant, {}).get("v2b", {}).get("status") == "determined"]
        endpoint = [key for key, l in by_label.items()
                    if l.get(variant, {}).get("v2a", {}).get("classification") == "endpoint_unique"]
        frontier[variant] = {"determined_labels": supported,
            "endpoint_unique_labels": endpoint,
            "v2b_status_counts": dict(Counter(l[variant]["v2b"]["status"]
                                              for l in labels if variant in l))}
    assert frontier["baseline"]["v2b_status_counts"] == {
        "partial": 3, "abstained": 8, "determined": 1}
    assert frontier["intervention"]["v2b_status_counts"] == {
        "abstained": 8, "determined": 4}
    assert frontier["baseline"]["endpoint_unique_labels"] == \
        frontier["intervention"]["endpoint_unique_labels"]
    assert set(frontier["intervention"]["determined_labels"]) == {
        TARGET, ((("A", 0),), (("B", 1),)),
        ((("A", 0),), (("B", 2),)), ((("A", 1),), (("B", 1),))}
    control = by_label[((("A", 0),), (("B", 0),))]
    assert control["intervention"]["v2a"]["classification"] == "endpoint_multiple"
    assert len(frontier["intervention"]["determined_labels"]) > 1
    assert {name: _file_sha(name) for name in PINS} == PINS
    assert {name: _file_sha(name) for name in source_pins} == source_pins
    record = {"scope": "synthetic test-only constant-total D redistribution",
        "decision": "abstain", "geometry": "unavailable; no placement or volume called",
        "reason": "four distinct full-gate labels; other competing labels unresolved",
        "original_pins": PINS, "production_source_pins": source_pins,
        "gates": GATES, "source_sha256": source_sha,
        "raw_image_sha256s": hashes, "sources": source_records,
        "pool": [dict(plane_graph._indexed_match_payload(m), disposition=d)
                 for m, d in zip(pool.candidates, pool.dispositions)],
        "whole_pool_solver_reason": whole.reason,
        "canonical_solver_reason": canonical_result.reason,
        "labels": labels, "frontier": frontier, "upstream_changes": upstream_changes,
        "target_masks_and_energies": mask_oracles,
        "measurement": {"primary_solver_calls": 23, "labels": 21,
                        "downstream_chains": 24,
                        "induced_rows": sum(len(l["induced_pool_rows"]) for l in labels)},
        "limits": "finite synthetic labels; cooperative elapsed/publication assertions only; no hard containment"}
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    # Publication checks are cooperative measurements, not runtime containment.
    assert len(encoded) < 4_000_000 and time.monotonic() - started < 120.
    artifact = tmp_path / "b0prime-d-redistribution.json"
    artifact.write_bytes(encoded)
    print("D_REDISTRIBUTION_ARTIFACT", str(artifact), len(encoded))
    print("D_REDISTRIBUTION_SHA256", hashlib.sha256(encoded).hexdigest())
    print("D_REDISTRIBUTION_FRONTIER", json.dumps(frontier, sort_keys=True))
