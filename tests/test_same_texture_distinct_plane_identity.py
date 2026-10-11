"""Frozen genuine-matcher falsifier; surface identities are a test-only oracle.

No image input, threshold override, candidate construction or fixture search.
All three arms are constructed before the first matcher call. A metadata period
of 63 identifies the duplicate reading independently; its cell grid, origin,
coverage and alternative b are identical to the original Y signature.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import sys

import numpy as np

from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline.image_derived_signatures import SignatureEvidence
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph


# Fixed once, before admission: these are construction constants, not a search.
SEEDS = (744, 745, 746)
SHAPE = (32, 32)
FRAMES = ("a", "b")
QUADRANT_CONSTRAINTS = (("a", "b", 0),)
RECTIFICATION = lattice_planes.PlaneRectification(
    axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    focal=500.0, principal_point=(120.0, 90.0), centre=(0.0, 0.0))


def _readonly(array):
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def _patterns():
    result = []
    for seed in SEEDS:
        raw = np.random.default_rng(seed).normal(size=SHAPE)
        cells = _readonly(((raw - raw.mean()) / raw.std()).astype(np.float32))
        assert np.unique(cells).size == cells.size  # no repeated cells or tiles
        result.append(cells)
    return tuple(result)


def _reading(frame, axis, distance, period):
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=-1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, distance, 1.0),),
        period_pixels=period, phase_pixels=(0.0, 0.0), extent=1.0,
        samples=1024, polygon=None, coverage_cells=1024,
        structural_share=0.6, confidence=0.9, rectification=RECTIFICATION)


def _freeze_arm(name, patterns):
    readings = [_reading("a", axis, distance, 60.0 + axis)
                for axis, distance in enumerate((20.0, 10.0, 30.0))]
    surfaces = ["surface-x", "surface-y-main", "surface-z"]
    if name != "A":
        readings.append(_reading("a", 1, 14.0 if name == "B" else 10.0, 63.0))
        surfaces.append("surface-y-impostor" if name == "B" else "surface-y-main")
    readings.extend(_reading("b", axis, distance, 60.0 + axis)
                    for axis, distance in enumerate((18.0, 8.0, 28.0)))
    surfaces.extend(("surface-x", "surface-y-main", "surface-z"))
    readings = tuple(readings)
    coverage = _readonly(np.ones(SHAPE, dtype=bool))
    interior = _readonly(np.zeros(SHAPE, dtype=np.float32))
    signatures = tuple(lattice_planes.PlaneSignature(
        reading=reading, blocks_per_period=1.0, cells=patterns[reading.axis],
        interior=interior, coverage=coverage, origin_uv=(0.0, 0.0))
        for reading in readings)
    indexed, identities, _ = plane_graph._canonical_reading_index(readings)
    # The oracle never appears in a reading, signature, match or selector input.
    oracle = {identities[id(reading)]: surface
              for reading, surface in zip(readings, surfaces)}
    evidence = tuple(SignatureEvidence(
        reading.frame, i, reading.axis, 0, 1.0, signature)
        for i, (reading, signature) in enumerate(zip(readings, signatures)))
    return name, readings, signatures, indexed, identities, oracle, evidence


def _grid_record(signature):
    return {
        "cells_sha256": hashlib.sha256(signature.cells.tobytes()).hexdigest(),
        "coverage_sha256": hashlib.sha256(signature.coverage.tobytes()).hexdigest(),
        "interior_sha256": hashlib.sha256(signature.interior.tobytes()).hexdigest(),
        "shape": signature.cells.shape, "origin_uv": signature.origin_uv,
        "blocks_per_period": signature.blocks_per_period,
    }


def _json_evidence(value):
    """Preserve measured IEEE special values explicitly, never emit invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_evidence(item) for item in value]
    return value


def _profiled(function, action):
    codes = {getattr(plane_graph, name).__code__: name
             for name in ("match_plane_graph", "match_plane_pair")}
    codes[selection.select_endpoint_hypotheses.__code__] = "select_endpoint_hypotheses"
    codes[csp.solve_endpoint_hypotheses.__code__] = "solve_endpoint_hypotheses"
    counts = Counter()
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    assert function.__code__ in codes
    try:
        sys.setprofile(profile)
        value = action()
    finally:
        sys.setprofile(previous)
    return value, dict(counts)


def _run_arm(frozen, report, save):
    name, readings, signatures, indexed, identities, oracle, evidence = frozen
    trace = plane_graph.PlaneGraphTrace([])
    universes, refusals = [], []
    filtered, calls = _profiled(plane_graph.match_plane_graph, lambda:
        plane_graph.match_plane_graph(readings, signatures, refusals=refusals,
            diagnostic_sink=trace, admitted_candidate_sink=universes))
    assert len(universes) == 1, f"{name}: complete admission universe not published"
    universe = universes[0]
    candidates = universe.candidates  # exact published tuple; no row deletion
    attempts = [(left, right, q) for left in signatures if left.reading.frame == "a"
                for right in signatures if right.reading.frame == "b" for q in range(4)]
    assert calls == {"match_plane_graph": 1, "match_plane_pair": len(attempts)}
    assert len(trace.pairs) == len(attempts) == trace.pre_admission_candidates
    assert sum(row.admitted for row in trace.pairs) == len(candidates)
    assert len(candidates) == trace.post_admission_candidates
    assert len(filtered) == trace.post_same_axis_candidates
    record = {"arm": name, "genuine_code_calls": calls, "attempts": [],
              "candidate_count": len(candidates), "legacy_filtered_count": len(filtered),
              "matcher_refusals": [asdict(row) for row in refusals],
              "readings": [], "candidates": [], "mandatory_q0_factors": []}
    report["arms"].append(record)
    for reading, signature in zip(readings, signatures):
        endpoint = identities[id(reading)]
        record["readings"].append({"endpoint": endpoint, "surface_id": oracle[endpoint],
            "axis": reading.axis, "side": reading.side,
            "period_pixels": reading.period_pixels,
            "structural_share": reading.structural_share,
            "alternatives": [{"alternative_index": i, **asdict(a)}
                             for i, a in enumerate(reading.alternatives)],
            "grid": _grid_record(signature)})
    for (left, right, q), row in zip(attempts, trace.pairs):
        u, v = identities[id(left.reading)], identities[id(right.reading)]
        assert (row.left, row.right, row.left_axis, row.right_axis, row.quadrant) == (
            u[0], v[0], left.reading.axis, right.reading.axis, q)
        if row.admitted:
            assert (row.left_plane, row.right_plane) == (u[1], v[1])
        record["attempts"].append({"endpoints": (u, v), "surface_ids": (oracle[u], oracle[v]),
            "oracle_same_surface": oracle[u] == oracle[v], "alternative_indices": (0, 0),
            "structural_shares": (left.reading.structural_share, right.reading.structural_share),
            "matcher_trace": asdict(row)})
    factors = defaultdict(list)
    for i, (row, disposition) in enumerate(zip(candidates, universe.dispositions)):
        u, v = (row.left, row.left_plane), (row.right, row.right_plane)
        # Every source row binds to the real indexed readings and exact b/alt.
        assert indexed[u].matching_alternatives(row.left_b) == (indexed[u].alternatives[0],)
        assert indexed[v].matching_alternatives(row.right_b) == (indexed[v].alternatives[0],)
        relation = plane_graph.signed_physical_relation(indexed, row,
            {"a": 0, "b": row.quadrant_of_right}, {u: 0, v: 0}, i, compact=True)
        record["candidates"].append({"source_row": i, "endpoints": (u, v),
            "surface_ids": (oracle[u], oracle[v]), "oracle_same_surface": oracle[u] == oracle[v],
            "alternative_indices": (0, 0), "admitted": True,
            "legacy_disposition": disposition, "active_at_frozen_q0": row.quadrant_of_right == 0,
            "signed_physical_relation": relation, "match": asdict(row)})
        if row.quadrant_of_right == 0:
            factors[(u, v)].append(i)
    for endpoints, rows in sorted(factors.items()):
        record["mandatory_q0_factors"].append({"endpoints": endpoints,
            "surface_ids": tuple(oracle[u] for u in endpoints), "source_rows": rows,
            "alternatives": (0, 0), "mandatory": True})
    save()
    y_rows = [r for r in record["candidates"]
              if r["active_at_frozen_q0"] and r["match"]["left_axis"] == 1]
    expected_y = 1 if name == "A" else 2
    assert len(y_rows) == expected_y, (
        f"{name}: frozen matcher admission failed; expected {expected_y} Q0 Y rows, "
        f"got {len(y_rows)}; see exact rejected attempts in {report['artifact']}")
    assert len(factors) == (3 if name == "A" else 4)
    assert all(r["match"]["overlap_cells"] >= plane_graph.MIN_OVERLAP_CELLS
               and r["match"]["psr"] >= 3.0 for r in y_rows)
    assert sum(r["oracle_same_surface"] for r in y_rows) == (2 if name == "C" else 1)
    cert, solver_calls = _profiled(selection.select_endpoint_hypotheses, lambda:
        selection.select_endpoint_hypotheses(readings=readings, signatures=evidence,
            candidate_matches=candidates, frame_ids=FRAMES,
            quadrant_constraints=QUADRANT_CONSTRAINTS, budget=csp.EndpointBudget(),
            session_sha256="0" * 64, lattice_sha256="1" * 64, signatures_sha256="2" * 64,
            candidate_dispositions=universe.dispositions))
    assert solver_calls == {"select_endpoint_hypotheses": 1, "solve_endpoint_hypotheses": 1}
    assert cert.candidate_count == len(candidates)
    assert cert.candidate_matches_sha256 == selection.candidate_digest(candidates)
    proof = json.loads(cert.proof_json)
    record.update(selector_status=cert.status, selector_reason=cert.reason,
                  selector_genuine_code_calls=solver_calls, selector_proof=proof,
                  selector_candidate_sha256=cert.candidate_matches_sha256)
    save()
    assert proof["exhaustive"] and proof["denial"] is None
    assert proof["physical_tolerance"] == plane_graph.SAME_AXIS_TOLERANCE == 0.15
    assert proof["counters"]["visited_assignments"] == proof["counters"]["planned_assignments"] == 1
    counts = proof["counters"]
    assert counts["scalar_payload_bytes"] == sum(counts[k + "_bytes"] for k in csp.BYTE_CATEGORIES)
    for prefix in ("scalar_records", "search_events"):
        assert counts[prefix + "_attempted"] == counts[prefix + "_committed"] + counts[prefix + "_uncommitted"]
        assert counts[prefix + "_uncommitted"] == 0
    assert counts["partial_prunes"] == counts["retained_partial_exclusions"] + counts["omitted_partial_exclusions"]
    if name == "B":
        assert cert.status == "no", f"B: expected exhaustive no, got {cert.status}: {cert.reason}"
        assert len(proof["partial_exclusions"]) == 1
        event = proof["partial_exclusions"][0]
        assert event[0] == "partial"
        group, low, high, low_ref, high_ref = event[5]
        assert group == ["a", "b", 1] and (low, high) == (-6.0, -2.0)
        assert high - low > proof["physical_tolerance"]
        for ref in (low_ref, high_ref):
            assert ref[0] in {r["source_row"] for r in y_rows}
        # Resolve every partial-proof forced factor back to oracle-bound source rows.
        for factor in event[4]:
            rows = factor[5]
            matching = [f for f in record["mandatory_q0_factors"] if f["source_rows"] == rows]
            assert len(matching) == 1
        assert proof["counters"]["solutions_found"] == 0
    else:
        assert cert.status == "unique", f"{name}: expected unique, got {cert.status}: {cert.reason}"
        assert proof["counters"]["solutions_found"] == 1
        assert not proof["partial_exclusions"] and not proof["physical_exclusions"]
        actual = proof["choices"][0]["factors"]
        assert len(actual) == len(factors)
        for factor in actual:
            endpoints = tuple(map(tuple, factor["endpoints"]))
            assert factor["source_rows"] == factors[endpoints]
            assert factor["alternatives"] == [0, 0]
            assert oracle[endpoints[0]] == oracle[endpoints[1]]
    return record


def _local(row):
    # Physical distance, surface IDs and canonical indices are outside local admission.
    return {key: value for key, value in row["match"].items()
            if key not in ("left_plane", "right_plane")}


def test_frozen_same_texture_distinct_plane_identity(tmp_path):
    patterns = _patterns()
    frozen = tuple(_freeze_arm(name, patterns) for name in ("A", "B", "C"))
    artifact = tmp_path / "same-texture-falsifier.json"
    report = {"task_id": "SOL-SAME-TEXTURE-DISTINCT-PLANE-FALSIFIER-01",
              "artifact": str(artifact), "fixed_seeds": SEEDS, "fixed_shape": SHAPE,
              "nonfinite_number_encoding": "IEEE nonfinite numbers use explicit inf/-inf/nan strings",
              "quadrant_constraints": QUADRANT_CONSTRAINTS,
              "selector_budget": asdict(csp.EndpointBudget()), "arms": []}

    def save():
        artifact.write_text(json.dumps(_json_evidence(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")

    # Prove frozen identical connectivity and duplicate grids before any matching.
    for axis in (0, 2):
        controls = [[(_grid_record(s), asdict(s.reading)) for s in arm[2]
                     if s.reading.axis == axis] for arm in frozen]
        assert controls[0] == controls[1] == controls[2]
    for arm in frozen[1:]:
        y_left = [s for s in arm[2] if s.reading.frame == "a" and s.reading.axis == 1]
        assert _grid_record(y_left[0]) == _grid_record(y_left[1])
    before = [[_grid_record(s) for s in arm[2]] for arm in frozen]
    save()
    print(f"FROZEN_EVIDENCE_ARTIFACT={artifact}")
    records = []
    for arm in frozen:
        record = _run_arm(arm, report, save)
        records.append(record)
        print(json.dumps({"arm": record["arm"], "candidate_count": record["candidate_count"],
              "legacy_filtered_count": record["legacy_filtered_count"],
              "status": record["selector_status"],
              "counters": record["selector_proof"]["counters"]}, sort_keys=True))
    y = [[r for r in record["candidates"] if r["active_at_frozen_q0"]
          and r["match"]["left_axis"] == 1] for record in records]
    assert _local(y[0][0]) == _local(y[1][0]) == _local(y[1][1]) == _local(y[2][0]) == _local(y[2][1])
    assert [_local(r) for r in records[1]["candidates"]] == [_local(r) for r in records[2]["candidates"]]
    local_traces = [[{k: v for k, v in attempt["matcher_trace"].items()
                      if k not in ("left_plane", "right_plane", "selected")}
                     for attempt in record["attempts"]] for record in records[1:]]
    assert local_traces[0] == local_traces[1]  # all 48 admitted/rejected B/C attempts
    assert {r["signed_physical_relation"][1] for r in y[1]} == {-6.0, -2.0}
    assert {r["signed_physical_relation"][1] for r in y[2]} == {-2.0}
    assert y[0][0]["oracle_same_surface"] and any(not r["oracle_same_surface"] for r in y[1])
    assert before == [[_grid_record(s) for s in arm[2]] for arm in frozen]
    report["conclusion"] = {
        "same_local_admission_evidence_all_y_pairs": True,
        "frozen_q0_statuses": [r["selector_status"] for r in records],
        "B_y_diameter": 4.0, "synthetic_hypothesis_activated": True,
        "real_image_hypothesis_established": False,
        "candidate_deletion_or_gate_override": False}
    save()
