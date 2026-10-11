"""Output-neutral, test-local G capture for one fixed default two-view call."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from test_image_only_multiview_gate import PINS, VIEWS, ROOT, _record
from test_image_derived_p0_pixel_bridge import _profiles
from test_lattice_planes import ROOM, mixed_material, render_room
from mc_photogrammetry import evaluation_same_axis_continuation as continuation
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_production as production
from mc_photogrammetry.pipeline import image_derived_session as sessions
from mc_photogrammetry.project.store import load_project
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction.camera import CameraModel


BASELINE = "docs/reports/SOL-IMG-MULTIVIEW-GATE-20261005.json"
PRIOR = dict(PINS, **{
    "tests/test_image_only_multiview_gate.py":
        "3fe03e23a8cbfbbc11ebf8526c8628e72ce3c3a4c1b5e068404df5000971420f",
    BASELINE: "a369b97da7e07787a0c84e663391c8cdcb7f1646002d4d18bdc55cb68a7dd09f",
    "docs/reports/SOL-IMG-MULTIVIEW-GATE-20261005.txt":
        "2e8ff7f989aa04129986a48598fb443759cab84857ff12d8c608c2c06b73fbaa",
})


def _file_sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _semantic(value):
    """Exclude declared path/digest identity; preserve all numeric/status evidence."""
    if isinstance(value, (list, tuple)):
        return [_semantic(v) for v in value]
    if not isinstance(value, dict):
        return value
    record = {}
    for key, v in value.items():
        if key.endswith("_sha256") or key == "sha256":
            continue
        if key == "stage_digests":
            record[key] = [row[0] for row in v]
        elif key == "signature_digests":
            record[key] = len(v)
        elif key == "source_region_id":
            record[key] = None if v is None else "<region-identity>"
        elif key in ("project_path", "source_label"):
            record[key] = None if v is None else Path(v).name
        elif key == "source_paths":
            record[key] = [Path(path).name for path in v]
        else:
            record[key] = _semantic(v)
    return record


def test_fixed_pair_g_prefilter_capture(tmp_path, monkeypatch):
    assert {path: _file_sha(path) for path in PRIOR} == PRIOR
    baseline = json.loads((ROOT / BASELINE).read_text(encoding="utf-8"))
    source_pins = {path: _file_sha(path) for path in baseline["source_pins"]}
    assert source_pins == baseline["source_pins"]
    inputs, input_records = [], []
    for index, (position, yaw, pitch) in enumerate(VIEWS):
        camera = CameraModel(position=position, yaw_degrees=yaw,
                             pitch_degrees=pitch, fov_degrees=70.)
        rgb = np.repeat(render_room(camera, 360, 240, ROOM,
            materials=mixed_material)[..., None], 3, axis=2)
        path = tmp_path / f"view{index}.png"
        Image.fromarray(rgb).save(path)
        inputs.append(path)
        input_records.append({"filename": path.name,
            "rgb_sha256": sessions.image_sha256(rgb),
            "encoded_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(rgb.shape)})
    assert input_records == baseline["input_files"]
    original_matcher = plane_graph.match_plane_graph
    original_solver = plane_graph.solve_component
    calls = Counter()
    capture = {}

    def matcher(readings, signatures, **kwargs):
        calls["matcher_wrapper"] += 1
        assert calls["matcher_wrapper"] == 1
        assert kwargs == {"max_shift_cells": 32}
        source_sha = continuation._source_sha(readings, signatures)
        observations = []
        calls["matcher_delegate"] += 1
        returned = original_matcher(readings, signatures,
            **kwargs, candidate_observation_sink=observations)
        assert len(observations) == 1
        pool = observations[0]
        assert len(pool.candidates) <= 256
        kept = tuple(m for m, d in zip(pool.candidates, pool.dispositions)
                     if d == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
        assert kept == returned
        assert all(a is b for a, b in zip(kept, returned))
        assert continuation._source_sha(readings, signatures) == source_sha
        capture.update(readings=readings, signatures=signatures,
            source_sha256=source_sha, matcher_kwargs=dict(kwargs),
            pool=pool, returned=returned)
        return returned  # original object, never a reconstructed graph tuple

    def solver(readings, matches, **kwargs):
        calls["solver_wrapper"] += 1
        assert calls["solver_wrapper"] == 1
        assert readings is capture["readings"]
        assert matches is capture["returned"]
        assert kwargs == dict(integer_tolerance=.15, min_resultant=.9,
                             ambiguous_margin=.02, quadrant_constraints=())
        assert continuation._source_sha(readings, capture["signatures"]) == capture["source_sha256"]
        input_payload = [plane_graph._indexed_match_payload(m) for m in matches]
        calls["solver_delegate"] += 1
        solution = original_solver(readings, matches, **kwargs)
        assert input_payload == [plane_graph._indexed_match_payload(m) for m in matches]
        assert continuation._source_sha(readings, capture["signatures"]) == capture["source_sha256"]
        capture.update(solver_kwargs=dict(kwargs), solution=solution,
                       solver_inputs=input_payload)
        return solution  # original failure and every original rival alternative

    target = tmp_path / "multiview.mcscan"
    with monkeypatch.context() as local:
        local.setattr(plane_graph, "match_plane_graph", matcher)
        local.setattr(plane_graph, "solve_component", solver)
        outcome = production.derive_image_output_from_paths(inputs, project_path=target)
    assert plane_graph.match_plane_graph is original_matcher
    assert plane_graph.solve_component is original_solver
    assert calls == {"matcher_wrapper": 1, "matcher_delegate": 1,
                     "solver_wrapper": 1, "solver_delegate": 1}
    assert isinstance(outcome, production.ImageDerivedProductionResult)
    assert (outcome.stage, outcome.status, outcome.reason, outcome.detail) == (
        "geometry", "geometry_unavailable", "g_unavailable", "ambiguous_quadrant")
    persisted = load_project(target)
    stages = [{"stage": stage, "record": _record(value)}
              for stage, value in outcome.geometry.records]
    current = {"production": _record(outcome), "entry": _record(outcome.entry),
        "session": outcome.entry.session_record,
        "connections": outcome.entry.graph.to_record(),
        "pose": outcome.entry.pose_graph.to_record(),
        "lattice": outcome.lattice.to_record(),
        "p0_profiles": _profiles(outcome.lattice, outcome.entry.session),
        "geometry": _record(outcome.geometry), "geometry_stages": stages,
        "output": None,
        "project": {"schema_version": persisted.schema_version,
            "image_derived_production": persisted.image_derived_production,
            "blocks": len(persisted.volume.blocks),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}}
    for name, value in current.items():
        assert _semantic(value) == _semantic(baseline[name]), name
    assert not persisted.volume.blocks and outcome.output is None
    assert [stage["stage"] for stage in stages] == ["f1", "g"]
    assert stages[1]["record"]["matches"] == []
    assert stages[1]["record"]["component_solution"] is None
    assert all(e["status"] == "refused" and e["right_yaw_class"] is None
               for e in current["connections"]["evidence"])
    pool, solution = capture["pool"], capture["solution"]
    assert solution.reason == "ambiguous_quadrant"
    raw_rows = [dict(plane_graph._indexed_match_payload(m), disposition=d)
                for m, d in zip(pool.candidates, pool.dispositions)]
    returned_ids = {id(m) for m in capture["returned"]}
    kept_indices = [i for i, m in enumerate(pool.candidates) if id(m) in returned_ids]
    assert capture["solver_inputs"] == [
        plane_graph._indexed_match_payload(pool.candidates[i]) for i in kept_indices]
    index, _, _ = plane_graph._canonical_reading_index(capture["readings"])
    alternatives = []
    for alternative in solution.alternatives:
        quadrants = dict(zip(solution.frames, alternative.quadrants))
        active_indices = [i for i in kept_indices
            if (quadrants[pool.candidates[i].right] - quadrants[pool.candidates[i].left]) % 4
               == pool.candidates[i].quadrant_of_right]
        active = [pool.candidates[i] for i in active_indices]
        observed = [0, 0, 0]
        for m in active:
            axis, _ = plane_graph.axis_transform(index[(m.left, m.left_plane)].axis,
                                                quadrants[m.left])
            observed[axis] += 2
        assert tuple(observed) == alternative.axis_observations
        unresolved = sum(not any(f in (m.left, m.right) for m in active)
                         for f in solution.frames)
        mean_score = float(np.mean([m.score for m in active])) if active else 0.
        phase_cost = sum((1.-r)*n for r, n in zip(
            alternative.resultants, alternative.axis_observations))
        cost = (phase_cost + plane_graph.LAMBDA_UNRESOLVED*unresolved
                + plane_graph.MU_MATCH_SCORE*(1.-mean_score))
        assert abs(cost - alternative.cost) < 1e-12
        alternatives.append(dict(asdict(alternative),
            active_prefilter_indices=active_indices, unresolved_frames=unresolved,
            mean_match_score=mean_score, weighted_resultant_cost=phase_cost,
            recomposed_cost=cost))
    assert len(alternatives) <= 4
    assert len(alternatives) >= 2
    assert alternatives[0]["quadrants"] != alternatives[1]["quadrants"]
    gap = alternatives[1]["cost"] - alternatives[0]["cost"]
    assert 0. <= gap < capture["solver_kwargs"]["ambiguous_margin"]
    reading_rows = [{"frame": f, "canonical_index": i, "axis": r.axis,
        "side": r.side, "alternatives": [asdict(a) for a in r.alternatives],
        "period_pixels": r.period_pixels, "phase_pixels": r.phase_pixels,
        "phase_evidence": asdict(r.phase_evidence), "extent": r.extent,
        "samples": r.samples, "structural_share": r.structural_share,
        "rectification": asdict(r.rectification),
        "source_region_id": r.region_evidence.source_region_id}
        for (f, i), r in index.items()]
    proof = {"scope": "test-local output-neutral G capture; one fixed default pair",
        "decision": "abstain", "prior_pins": PRIOR, "source_pins": source_pins,
        "inputs": input_records, "public_result": current,
        "semantic_baseline": {"path": BASELINE, "sha256": PRIOR[BASELINE],
                              "all_sections_equal": list(current)},
        "capture": {"calls": dict(calls), "source_sha256": capture["source_sha256"],
            "matcher_kwargs": capture["matcher_kwargs"], "solver_kwargs": capture["solver_kwargs"],
            "reading_rows": reading_rows, "prefilter_rows": raw_rows,
            "kept_prefilter_indices": kept_indices,
            "postfilter_solver_inputs": capture["solver_inputs"],
            "solution": graph_mod._solution_payload(solution),
            "alternatives": alternatives, "best_runner_up_cost_gap": gap},
        "limits": "256 admitted rows; one original matcher + one original solver; 2MB cooperative publication guard; no runtime containment or new pose/assignment run",
        "geometry": "unavailable; no placement/output/export called"}
    assert {path: _file_sha(path) for path in PRIOR} == PRIOR
    assert {path: _file_sha(path) for path in source_pins} == source_pins
    encoded = json.dumps(proof, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    assert len(encoded) < 2_000_000
    artifact = tmp_path / "g-prefilter-capture.json"
    artifact.write_bytes(encoded)
    print("G_CAPTURE_RESULT", json.dumps({"raw_rows": len(raw_rows),
        "kept_rows": len(kept_indices), "calls": dict(calls),
        "quadrant_counts": dict(Counter(r["quadrant"] for r in raw_rows)),
        "alternatives": alternatives, "cost_gap": gap}, sort_keys=True))
    print("G_CAPTURE_ARTIFACT", str(artifact), len(encoded),
          hashlib.sha256(encoded).hexdigest())
