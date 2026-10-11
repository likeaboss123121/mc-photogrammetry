"""One fixed four-view public image-only call with output-neutral G capture."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json

import numpy as np
from PIL import Image

from test_image_only_multiview_gate import ROOT, _record
from test_image_only_g_prefilter_capture import PRIOR, _file_sha
from test_image_derived_p0_pixel_bridge import _profiles
from test_lattice_planes import ROOM, mixed_material, render_room
from mc_photogrammetry import evaluation_same_axis_continuation as continuation
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_production as production
from mc_photogrammetry.pipeline import image_derived_session as sessions
from mc_photogrammetry.project.store import load_project
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction.camera import CameraModel


# Preregistered before the first run; cameras belong only to the RGB generator.
VIEWS = (((3.6, .7, 3.6), 10., -25.), ((5., .7, 5.), 10., -25.),
         ((8.4, .7, 3.6), -80., -25.), ((8.4, .7, 8.4), -170., -25.))
PINS = dict(PRIOR, **{
    "tests/test_image_only_g_prefilter_capture.py":
        "dabf9aeede618348bdc28fa7bdab998148a8055747692024b0ffc8e0e0a03218",
    "docs/reports/SOL-IMG-G-PREFILTER-CAPTURE-20261005.json":
        "ffe9160bc0eb84dd97e27dfadded169d8b57f0fc853f935be910045b93c851c9",
    "docs/reports/SOL-IMG-G-PREFILTER-CAPTURE-20261005.txt":
        "8e2f821a0baa5560bc11d833d265484e9d5f3fdbacbf7dcd19d16ff487c48433",
})


def test_default_image_only_fourview_gate(tmp_path, monkeypatch):
    assert {path: _file_sha(path) for path in PINS} == PINS
    baseline = json.loads((ROOT / "docs/reports/SOL-IMG-MULTIVIEW-GATE-20261005.json").read_bytes())
    source_pins = dict(baseline["source_pins"])
    assert {path: _file_sha(path) for path in source_pins} == source_pins
    inputs, rendered = [], []
    for i, (position, yaw, pitch) in enumerate(VIEWS):
        camera = CameraModel(position=position, yaw_degrees=yaw,
                             pitch_degrees=pitch, fov_degrees=70.)
        rgb = np.repeat(render_room(camera, 360, 240, ROOM,
                        materials=mixed_material)[..., None], 3, axis=2)
        path = tmp_path / f"view{i}.png"
        Image.fromarray(rgb).save(path)
        inputs.append(path)
        rendered.append({"filename": path.name, "shape": list(rgb.shape),
            "rgb_sha256": sessions.image_sha256(rgb),
            "encoded_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert rendered[:2] == baseline["input_files"]
    assert len({r["rgb_sha256"] for r in rendered}) == 4
    original_matcher, original_solver = plane_graph.match_plane_graph, plane_graph.solve_component
    calls, capture = Counter(), {}

    def matcher(readings, signatures, **kwargs):
        calls["matcher_wrapper"] += 1
        # Observe the first G call; later replay calls, if reached, delegate verbatim.
        if "pool" in capture:
            calls["matcher_delegate"] += 1
            return original_matcher(readings, signatures, **kwargs)
        assert kwargs == {"max_shift_cells": 32}
        before = continuation._source_sha(readings, signatures)
        observations = []
        calls["matcher_delegate"] += 1
        returned = original_matcher(readings, signatures, **kwargs,
                                    candidate_observation_sink=observations)
        assert len(observations) == 1
        pool = observations[0]
        kept = tuple(m for m, d in zip(pool.candidates, pool.dispositions)
                     if d == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
        assert kept == returned and all(a is b for a, b in zip(kept, returned))
        assert continuation._source_sha(readings, signatures) == before
        capture.update(readings=readings, signatures=signatures, pool=pool,
            returned=returned, source_sha256=before, matcher_kwargs=dict(kwargs))
        return returned

    def solver(readings, matches, **kwargs):
        calls["solver_wrapper"] += 1
        if "solution" in capture:
            calls["solver_delegate"] += 1
            return original_solver(readings, matches, **kwargs)
        assert readings is capture["readings"] and matches is capture["returned"]
        before = [plane_graph._indexed_match_payload(m) for m in matches]
        assert continuation._source_sha(readings, capture["signatures"]) == capture["source_sha256"]
        calls["solver_delegate"] += 1
        returned = original_solver(readings, matches, **kwargs)
        assert before == [plane_graph._indexed_match_payload(m) for m in matches]
        assert continuation._source_sha(readings, capture["signatures"]) == capture["source_sha256"]
        capture.update(solution=returned, solver_inputs=before, solver_kwargs=dict(kwargs))
        return returned

    target = tmp_path / "fourview.mcscan"
    with monkeypatch.context() as local:
        local.setattr(plane_graph, "match_plane_graph", matcher)
        local.setattr(plane_graph, "solve_component", solver)
        # The sole production invocation receives no generator pose/camera metadata.
        outcome = production.derive_image_output_from_paths(inputs, project_path=target)
    assert plane_graph.match_plane_graph is original_matcher
    assert plane_graph.solve_component is original_solver
    assert calls["matcher_wrapper"] == calls["matcher_delegate"]
    assert calls["solver_wrapper"] == calls["solver_delegate"]
    assert isinstance(outcome, production.ImageDerivedProductionResult), outcome
    persisted = load_project(target)
    assert persisted.schema_version == 3
    assert persisted.image_derived_session == outcome.entry.session_record
    assert persisted.image_derived_production["status"] == outcome.status
    assert len(outcome.entry.session.frames) == 4
    assert outcome.entry.graph.pair_count == len(outcome.entry.graph.evidence) == 6
    stages = [] if outcome.geometry is None else [
        {"stage": stage, "record": _record(value)} for stage, value in outcome.geometry.records]
    g_capture = None
    if "pool" in capture:
        pool = capture["pool"]
        rows = [dict(plane_graph._indexed_match_payload(m), disposition=d)
                for m, d in zip(pool.candidates, pool.dispositions)]
        ids = {id(m) for m in capture["returned"]}
        kept_indices = [i for i, m in enumerate(pool.candidates) if id(m) in ids]
        g_capture = {"source_sha256": capture["source_sha256"],
            "matcher_kwargs": capture["matcher_kwargs"], "prefilter_rows": rows,
            "kept_prefilter_indices": kept_indices,
            "signature_count": len(capture["signatures"])}
        index, _, _ = plane_graph._canonical_reading_index(capture["readings"])
        g_capture["reading_rows"] = [{"frame": f, "canonical_index": i,
            "axis": r.axis, "side": r.side, "alternatives": [asdict(a) for a in r.alternatives],
            "period_pixels": r.period_pixels, "phase_pixels": r.phase_pixels,
            "phase_evidence": asdict(r.phase_evidence), "extent": r.extent,
            "samples": r.samples, "structural_share": r.structural_share,
            "rectification": asdict(r.rectification),
            "source_region_id": r.region_evidence.source_region_id} for (f, i), r in index.items()]
        if "solution" in capture:
            solution = capture["solution"]
            assert capture["solver_inputs"] == [plane_graph._indexed_match_payload(
                pool.candidates[i]) for i in kept_indices]
            alternatives = []
            for alternative in solution.alternatives:
                q = dict(zip(solution.frames, alternative.quadrants))
                active_indices = [i for i in kept_indices if (
                    q[pool.candidates[i].right] - q[pool.candidates[i].left]) % 4
                    == pool.candidates[i].quadrant_of_right]
                active = [pool.candidates[i] for i in active_indices]
                observed = [0, 0, 0]
                for m in active:
                    axis, _ = plane_graph.axis_transform(index[(m.left, m.left_plane)].axis, q[m.left])
                    observed[axis] += 2
                assert tuple(observed) == alternative.axis_observations
                unresolved = sum(not any(f in (m.left, m.right) for m in active) for f in solution.frames)
                mean_score = float(np.mean([m.score for m in active])) if active else 0.
                phase_cost = sum((1.-r)*n for r, n in zip(alternative.resultants, alternative.axis_observations))
                cost = phase_cost + plane_graph.LAMBDA_UNRESOLVED*unresolved + plane_graph.MU_MATCH_SCORE*(1.-mean_score)
                assert abs(cost - alternative.cost) < 1e-12
                alternatives.append(dict(asdict(alternative), active_prefilter_indices=active_indices,
                    unresolved_frames=unresolved, mean_match_score=mean_score,
                    weighted_resultant_cost=phase_cost, recomposed_cost=cost))
            gap = alternatives[1]["cost"] - alternatives[0]["cost"] if len(alternatives) > 1 else None
            g_capture.update(solver_kwargs=capture["solver_kwargs"],
                postfilter_solver_inputs=capture["solver_inputs"],
                solution=graph_mod._solution_payload(solution), alternatives=alternatives,
                best_runner_up_cost_gap=gap,
                unconstrained_assignment_upper_bound=4**(len(solution.frames)-1))
    export_record = None
    metrics = {"geometry_accuracy": None, "exact_state_accuracy": None,
               "reason": "no promoted geometry/state; no post-hoc alignment or world-state ground truth claimed"}
    promotion = None if outcome.output is None else getattr(outcome.output, "promotion", None)
    if promotion is not None and getattr(promotion, "export_ready", False):
        from mc_photogrammetry.pipeline.image_derived_delivery import deliver_image_derived_litematic
        delivered = deliver_image_derived_litematic(promotion, tmp_path / "fourview.litematic",
            name="fourview", description="Fixed generated-room default pipeline experiment")
        export_record = _record(delivered)
        # Renderer material intensities do not supply exact Minecraft block-state truth.
        metrics["reason"] = "export boundary exercised; absolute coordinates/scale and Minecraft state truth are not established by the grayscale renderer"
    proof = {"scope": "one preregistered four-view default public image-only call",
        "decision": "abstain" if outcome.status == "geometry_unavailable" else outcome.status,
        "prior_pins": PINS, "source_pins": source_pins, "input_files": rendered,
        "generator_truth_posthoc_only": {"room": ROOM, "views": VIEWS, "fov_degrees": 70.,
            "material_rule": "(floor(u)*3 + floor(v)*5) mod 3", "truth_was_production_input": False},
        "production": _record(outcome), "entry": _record(outcome.entry),
        "session": outcome.entry.session_record, "connections": outcome.entry.graph.to_record(),
        "pose": outcome.entry.pose_graph.to_record(),
        "lattice": None if outcome.lattice is None else outcome.lattice.to_record(),
        "p0_profiles": [] if outcome.lattice is None else _profiles(outcome.lattice, outcome.entry.session),
        "geometry": None if outcome.geometry is None else _record(outcome.geometry),
        "geometry_stages": stages, "g_capture": g_capture, "observer_calls": dict(calls),
        "output": None if outcome.output is None else _record(outcome.output),
        "project": {"schema_version": persisted.schema_version, "blocks": len(persisted.volume.blocks),
            "image_derived_production": persisted.image_derived_production,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "metrics": metrics, "export": export_record,
        "limits": "one four-view call/six camera pairs; G original-return delegates; <=2048 admitted rows/128 observed delegates/4MB cooperative publication assertions; no hard runtime containment or universal typed overflow"}
    assert {path: _file_sha(path) for path in PINS} == PINS
    assert {path: _file_sha(path) for path in source_pins} == source_pins
    encoded = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    artifact = tmp_path / "fourview-gate.json"
    artifact.write_bytes(encoded)
    print("FOURVIEW_ARTIFACT", str(artifact), len(encoded), hashlib.sha256(encoded).hexdigest())
    print("FOURVIEW_RESULT", json.dumps({"stage": outcome.stage, "status": outcome.status,
        "reason": outcome.reason, "detail": outcome.detail,
        "pair_outcomes": [(e.status, e.reason) for e in outcome.entry.graph.evidence],
        "stages": [s["stage"] for s in stages], "calls": dict(calls),
        "raw_rows": None if g_capture is None else len(g_capture["prefilter_rows"]),
        "kept_rows": None if g_capture is None else len(g_capture["kept_prefilter_indices"]),
        "solver_reason": None if "solution" not in capture else capture["solution"].reason}, sort_keys=True))
    assert len(encoded) < 4_000_000
    assert sum(calls.values()) <= 128
    assert g_capture is None or len(g_capture["prefilter_rows"]) <= 2048
