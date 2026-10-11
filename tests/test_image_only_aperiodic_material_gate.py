"""One material-layout-only intervention; renderer truth is evaluated post hoc."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json

import numpy as np
from PIL import Image

from test_image_only_fourview_gate import PINS as OLD_PINS, VIEWS
from test_image_only_g_prefilter_capture import _file_sha
from test_image_only_multiview_gate import ROOT, _record
from test_image_derived_p0_pixel_bridge import _profiles
from test_lattice_planes import ROOM, render_room
from mc_photogrammetry import evaluation_same_axis_continuation as continuation
from mc_photogrammetry.pipeline import image_derived_entry as entry
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_production as production
from mc_photogrammetry.pipeline import image_derived_session as sessions
from mc_photogrammetry.pipeline import joint_matching as joint
from mc_photogrammetry.project.store import load_project
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction.camera import (
    CameraModel, camera_eye, camera_rotation_matrix, intrinsics_for, project_points)


BASELINE = "docs/reports/SOL-IMG-FOURVIEW-GATE-20261005.json"
PINS = dict(OLD_PINS, **{
    "tests/test_image_only_fourview_gate.py":
        "419ba735bad9651a181ef959cf9ab210a0b942989a2845261f76c32aeeb2b6dc",
    BASELINE: "94fa47a7f7c12ac4442833c20907229edf63f8b7c695e6e7098daf928c75297e",
    "docs/reports/SOL-IMG-FOURVIEW-GATE-20261005.txt":
        "2d34a209c1aa611d0a56c3efebe5fc59e4d3f3444b11d6913bf237ed58ef890e",
})
TOLERANCE_PX = 2.0  # Post-hoc metric, never a producer/matcher argument.
VISIBILITY_WORLD_TOLERANCE = 1e-6


def _material(bx, by, axis):
    return ((bx.astype(np.int64)*73856093) ^ (by.astype(np.int64)*19349663)
            ^ (axis*83492791)) % 3


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _array_sha(value):
    array = np.asarray(value)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode()
                          + array.tobytes(order="C")).hexdigest()


def _hits(camera, pixels):
    """Continuous-pixel counterpart of render_room's first in-room plane hit."""
    pixels = np.asarray(pixels, dtype=float).reshape((-1, 2))
    k = intrinsics_for(camera, 360, 240)
    rays = np.column_stack(((pixels[:, 0]-180)/k.focal_length_pixels[0],
        -(pixels[:, 1]-120)/k.focal_length_pixels[1], np.ones(len(pixels))))
    world = rays @ camera_rotation_matrix(camera)
    eye = camera_eye(camera)
    best, faces = np.full(len(pixels), np.inf), np.full(len(pixels), -1, dtype=int)
    xyz = np.zeros((len(pixels), 3))
    for face, (axis, value) in enumerate(((0, ROOM[0]), (0, ROOM[1]),
            (1, ROOM[2]), (1, ROOM[3]), (2, ROOM[4]), (2, ROOM[5]))):
        t = np.full(len(pixels), np.inf)
        np.divide(value-eye[axis], world[:, axis], out=t, where=np.abs(world[:, axis]) > 1e-15)
        point = eye + np.where(np.isfinite(t), t, 0.)[:, None]*world
        valid = np.isfinite(t) & (t > 1e-6) & (t < best)
        for other in range(3):
            if other != axis:
                valid &= (point[:, other] >= ROOM[2*other]-1e-6) & (point[:, other] <= ROOM[2*other+1]+1e-6)
        xyz[valid], faces[valid], best[valid] = point[valid], face, t[valid]
    return xyz, faces


def _feature_truth(camera, pixels):
    xyz, faces = _hits(camera, pixels)
    stable = (faces >= 0) & np.all((pixels >= (0., 0.)) & (pixels < (360., 240.)), axis=1)
    for offset in ((2., 0.), (-2., 0.), (0., 2.), (0., -2.)):
        shifted = pixels + offset
        other_xyz, other_faces = _hits(camera, shifted)
        stable &= (other_faces == faces) & np.all((shifted >= (0., 0.)) & (shifted < (360., 240.)), axis=1)
        for face in range(6):
            mask = faces == face
            tangents = [axis for axis in range(3) if axis != face//2]
            stable[mask] &= np.all(np.floor(xyz[mask][:, tangents]) ==
                np.floor(other_xyz[mask][:, tangents]), axis=1)
    # Bind the analytic continuous ray to the same renderer camera projection.
    projected = project_points(xyz, camera, width=360, height=240)
    for i, pixel in enumerate(projected):
        if faces[i] >= 0:
            assert pixel is not None and np.linalg.norm(np.asarray(pixel[:2])-pixels[i]) < 1e-8
    return xyz, faces, stable


def _edge_truth(left, right, graph, cameras):
    lp, rp = left.points, right.points
    xyz_l, faces_l, stable_l = _feature_truth(cameras[left.frame_id], lp)
    _, _, stable_r = _feature_truth(cameras[right.frame_id], rp)
    projected = project_points(xyz_l, cameras[right.frame_id], width=360, height=240)
    predicted = np.full((len(lp), 2), np.nan)
    visible = np.zeros(len(lp), dtype=bool)
    for i, pixel in enumerate(projected):
        if faces_l[i] >= 0 and pixel is not None:
            predicted[i] = pixel[:2]
            visible[i] = 0 <= pixel[0] < 360 and 0 <= pixel[1] < 240
    in_frame = np.flatnonzero(visible)
    right_hits, right_faces = _hits(cameras[right.frame_id], predicted[in_frame])
    visible[in_frame] &= ((right_faces == faces_l[in_frame]) &
        (np.linalg.norm(right_hits-xyz_l[in_frame], axis=1) <= VISIBILITY_WORLD_TOLERANCE))
    indices_l, indices_r = graph.edges.T
    scorable = stable_l[indices_l] & stable_r[indices_r]
    errors = np.linalg.norm(predicted[indices_l]-rp[indices_r], axis=1)
    correct = scorable & visible[indices_l] & (errors <= TOLERANCE_PX)
    return [{"scorable": bool(scorable[i]), "boundary_excluded": not bool(scorable[i]),
        "truth_visible_in_right": bool(visible[a]),
        "error_pixels": None if not np.isfinite(errors[i]) else float(errors[i]),
        "correct_within_2px": bool(correct[i])} for i, a in enumerate(indices_l)]


def _truth_counts(rows):
    scorable = sum(row["scorable"] for row in rows)
    correct = sum(row["correct_within_2px"] for row in rows)
    return {"total": len(rows), "scorable": scorable, "correct": correct,
        "incorrect": scorable-correct, "boundary_excluded": len(rows)-scorable,
        "precision_on_scorable": None if not scorable else correct/scorable}


def test_default_image_only_aperiodic_material_gate(tmp_path, monkeypatch):
    assert {path: _file_sha(path) for path in PINS} == PINS
    baseline = json.loads((ROOT/BASELINE).read_bytes())
    source_pins = dict(baseline["source_pins"])
    source_pins.update({"src/mc_photogrammetry/"+path+".py": _file_sha(
        "src/mc_photogrammetry/"+path+".py") for path in (
        "pipeline/joint_matching", "pipeline/correspondence", "pipeline/connected_camera", "reconstruction/camera")})
    assert {path: _file_sha(path) for path in source_pins} == source_pins
    cameras, inputs, rendered = {}, [], []
    for i, (position, yaw, pitch) in enumerate(VIEWS):
        camera = CameraModel(position=position, yaw_degrees=yaw, pitch_degrees=pitch, fov_degrees=70.)
        cameras[f"image-{i:04d}"] = camera  # generator/post-hoc only
        rgb = np.repeat(render_room(camera, 360, 240, ROOM, materials=_material)[..., None], 3, axis=2)
        path = tmp_path/f"view{i}.png"
        Image.fromarray(rgb).save(path)
        inputs.append(path)
        rendered.append({"filename": path.name, "shape": list(rgb.shape),
            "rgb_sha256": sessions.image_sha256(rgb), "encoded_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert all(a["rgb_sha256"] != b["rgb_sha256"] for a, b in zip(rendered, baseline["input_files"]))
    original_pair, original_detect = entry.match_and_solve_pair, joint.detect_features
    original_graph, original_scores = joint.reciprocal_topk_graph, joint._score_modes
    original_matcher, original_solver = plane_graph.match_plane_graph, plane_graph.solve_component
    calls, pairs, features, active, g = Counter(), [], {}, {}, {}

    def detect(frame, image, **kwargs):
        calls["detect_delegate"] += 1
        before = sessions.image_sha256(image)
        result = original_detect(frame, image, **kwargs)
        assert before == sessions.image_sha256(image)
        active["features"].append(result)
        fingerprint = (_array_sha(result.points), _array_sha(result.descriptors), _array_sha(result.sizes))
        if frame in features:
            assert features[frame][1] == fingerprint
        else:
            features[frame] = (result, fingerprint, before)
        return result

    def candidate_graph(left, right, **kwargs):
        calls["candidate_graph_delegate"] += 1
        before = (_array_sha(left), _array_sha(right))
        result = original_graph(left, right, **kwargs)
        assert before == (_array_sha(left), _array_sha(right))
        assert "graph" not in active
        active["graph"] = result
        active["graph_kwargs"] = kwargs
        return result

    def score_modes(modes, best, refusal, support_floor, bounds):
        calls["score_modes_delegate"] += 1
        before = [(_array_sha(m.matches), _array_sha(m.selection)) for m in modes]
        result = original_scores(modes, best, refusal, support_floor, bounds)
        assert before == [(_array_sha(m.matches), _array_sha(m.selection)) for m in modes]
        assert len(modes) == len(result)
        active.update(modes=tuple(modes), scores=result, best=best, bounds=asdict(bounds))
        return result

    def pair(left_id, left_image, right_id, right_image, **kwargs):
        calls["pair_delegate"] += 1
        assert not active and set(kwargs) == {"left_camera", "right_camera"}
        before = (sessions.image_sha256(left_image), sessions.image_sha256(right_image))
        active.update(left_id=left_id, right_id=right_id, features=[], camera_kwargs=kwargs)
        result = original_pair(left_id, left_image, right_id, right_image, **kwargs)
        assert before == (sessions.image_sha256(left_image), sessions.image_sha256(right_image))
        assert result.left_id == left_id and result.right_id == right_id
        if "best" in active:
            assert np.array_equal(result.matches, active["best"].matches)
        pairs.append(dict(active, result=result, image_sha256=before))
        active.clear()
        return result

    def matcher(readings, signatures, **kwargs):
        calls["g_matcher_delegate"] += 1
        if "pool" in g:
            return original_matcher(readings, signatures, **kwargs)
        assert kwargs == {"max_shift_cells": 32}
        before = continuation._source_sha(readings, signatures)
        observation = []
        result = original_matcher(readings, signatures, **kwargs, candidate_observation_sink=observation)
        assert len(observation) == 1
        pool = observation[0]
        kept = tuple(m for m, d in zip(pool.candidates, pool.dispositions) if d == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
        assert kept == result and all(a is b for a, b in zip(kept, result))
        assert continuation._source_sha(readings, signatures) == before
        g.update(readings=readings, signatures=signatures, pool=pool, matches=result,
                 source_sha256=before, matcher_kwargs=kwargs)
        return result

    def solver(readings, matches, **kwargs):
        calls["g_solver_delegate"] += 1
        if "solution" in g:
            return original_solver(readings, matches, **kwargs)
        assert readings is g["readings"] and matches is g["matches"]
        before = [plane_graph._indexed_match_payload(m) for m in matches]
        result = original_solver(readings, matches, **kwargs)
        assert before == [plane_graph._indexed_match_payload(m) for m in matches]
        assert continuation._source_sha(readings, g["signatures"]) == g["source_sha256"]
        g.update(solution=result, solver_kwargs=kwargs, solver_inputs=before)
        return result

    target = tmp_path/"aperiodic.mcscan"
    with monkeypatch.context() as local:
        for module, name, fn in ((entry, "match_and_solve_pair", pair),
            (joint, "detect_features", detect), (joint, "reciprocal_topk_graph", candidate_graph),
            (joint, "_score_modes", score_modes), (plane_graph, "match_plane_graph", matcher),
            (plane_graph, "solve_component", solver)):
            local.setattr(module, name, fn)
        outcome = production.derive_image_output_from_paths(inputs, project_path=target)
    assert (entry.match_and_solve_pair, joint.detect_features, joint.reciprocal_topk_graph,
        joint._score_modes, plane_graph.match_plane_graph, plane_graph.solve_component) == (
        original_pair, original_detect, original_graph, original_scores, original_matcher, original_solver)
    assert isinstance(outcome, production.ImageDerivedProductionResult), outcome
    assert calls["pair_delegate"] == len(pairs) == 6 and not active
    assert calls["detect_delegate"] == 12 and calls["candidate_graph_delegate"] == 6
    persisted = load_project(target)
    assert persisted.schema_version == 3 and persisted.image_derived_session == outcome.entry.session_record
    assert persisted.image_derived_production["status"] == outcome.status
    # Truth enters only now, after the sole production call and restored observers.
    pair_records = []
    for packet in pairs:
        left, right = packet["features"]
        assert (left.frame_id, right.frame_id) == (packet["left_id"], packet["right_id"])
        graph = packet["graph"]
        truth = _edge_truth(left, right, graph, cameras)
        native = packet["result"]
        estimates = packet["camera_kwargs"]
        shifts = [int(round((cameras[f].yaw_degrees-estimates[k].pose.yaw_degrees)/90)) % 4
            for f, k in ((left.frame_id, "left_camera"), (right.frame_id, "right_camera"))]
        expected_q = (shifts[1]-shifts[0]) % 4
        estimate_l = estimates["left_camera"]
        native_rotation_l = camera_rotation_matrix(CameraModel(position=(0., 0., 0.),
            yaw_degrees=estimate_l.pose.yaw_degrees, pitch_degrees=estimate_l.pose.pitch_degrees,
            roll_degrees=estimate_l.pose.roll_degrees))
        true_delta = camera_eye(cameras[right.frame_id])-camera_eye(cameras[left.frame_id])
        aligned_delta = native_rotation_l.T @ camera_rotation_matrix(cameras[left.frame_id]) @ true_delta
        aligned_delta /= np.linalg.norm(aligned_delta)
        modes = []
        for mode, score in zip(packet.get("modes", ()), packet.get("scores", ())):
            chosen = graph.edges[mode.selection]
            exact_matches = np.column_stack((left.points[chosen[:, 0]], right.points[chosen[:, 1]]))
            assert np.array_equal(mode.matches, exact_matches)
            assert mode.support == len(mode.selection)
            selected_truth = [truth[int(i)] for i in mode.selection]
            direction = np.asarray(mode.baseline_direction)
            angle = float(np.degrees(np.arccos(np.clip(direction @ aligned_delta, -1., 1.))))
            modes.append({"native_mode": _plain(asdict(mode)), "native_score": _plain(asdict(score)),
                "ranked_winner": mode is packet["best"], "truth_relative_yaw_class": expected_q,
                "yaw_class_correct": mode.yaw_class == expected_q,
                "signed_baseline_error_degrees_in_left_gauge": angle,
                "baseline_comparison_requires_correct_yaw_class": True,
                "truth_support": _truth_counts(selected_truth)})
        pair_records.append({"left": left.frame_id, "right": right.frame_id,
            "input_image_sha256": packet["image_sha256"], "camera_kwargs": _plain({k: asdict(v) for k, v in estimates.items()}),
            "settings": packet.get("bounds"), "graph_kwargs": packet["graph_kwargs"],
            "native_graph": _plain(asdict(graph)), "native_result": _plain(asdict(native)),
            "edge_truth": truth, "candidate_truth": _truth_counts(truth), "modes": modes,
            "truth_relative_yaw_class": expected_q, "truth_baseline_in_left_gauge": aligned_delta.tolist()})
    stages = [] if outcome.geometry is None else [{"stage": s, "record": _record(v)} for s, v in outcome.geometry.records]
    g_record = None
    if "pool" in g:
        pool = g["pool"]
        kept_ids = {id(m) for m in g["matches"]}
        g_record = {"source_sha256": g["source_sha256"], "matcher_kwargs": g["matcher_kwargs"],
            "reading_count": len(g["readings"]), "signature_count": len(g["signatures"]),
            "prefilter_rows": [dict(plane_graph._indexed_match_payload(m), disposition=d) for m, d in zip(pool.candidates, pool.dispositions)],
            "kept_prefilter_indices": [i for i, m in enumerate(pool.candidates) if id(m) in kept_ids]}
        if "solution" in g:
            g_record.update(solver_kwargs=g["solver_kwargs"], postfilter_solver_inputs=g["solver_inputs"],
                solution=graph_mod._solution_payload(g["solution"]),
                alternatives=[asdict(a) for a in g["solution"].alternatives])
            assert g["solver_inputs"] == [plane_graph._indexed_match_payload(pool.candidates[i]) for i in g_record["kept_prefilter_indices"]]
    calibration = []
    for current, old in zip(outcome.entry.session_record["cameras"], baseline["session"]["cameras"]):
        p, q = current["pose"], old["pose"]
        focal, old_focal = current["intrinsics"]["focal_length_pixels"][0], old["intrinsics"]["focal_length_pixels"][0]
        calibration.append({"frame": current["frame_id"], "focal_pixels": focal,
            "focal_shift_pixels": focal-old_focal, "focal_relative_shift": focal/old_focal-1,
            "yaw_mod90_shift_degrees": (p["yaw_degrees"]-q["yaw_degrees"]+45)%90-45,
            "pitch_shift_degrees": p["pitch_degrees"]-q["pitch_degrees"],
            "roll_shift_degrees": p["roll_degrees"]-q["roll_degrees"]})
    feature_records = {f: {"image_sha256": image_sha, "points": value.points.tolist(),
        "sizes": None if value.sizes is None else value.sizes.tolist(),
        "array_sha256": hashes, "width": value.width, "height": value.height,
        "usable_pixels": value.usable_pixels, "boundary_dropped": value.boundary_dropped}
        for f, (value, hashes, image_sha) in features.items()}
    export = None
    promotion = None if outcome.output is None else outcome.output.promotion
    if promotion is not None and getattr(promotion, "export_ready", False):
        from mc_photogrammetry.pipeline.image_derived_delivery import deliver_image_derived_litematic
        export = _record(deliver_image_derived_litematic(promotion, tmp_path/"aperiodic.litematic",
            name="aperiodic", description="Fixed material-layout default pipeline experiment"))
    proof = {"scope": "one fixed material-layout intervention; default public four-view pipeline",
        "decision": "abstain" if outcome.status == "geometry_unavailable" else outcome.status,
        "baseline": {"path": BASELINE, "sha256": PINS[BASELINE], "not_rerun": True,
            "correspondence_truth_not_measured": "pinned baseline lacks native feature/match coordinates"},
        "prior_pins": PINS, "source_pins": source_pins, "input_files": rendered,
        "generator_truth_posthoc_only": {"room": ROOM, "views": VIEWS, "fov_degrees": 70.,
            "material_rule": "((bx*73856093)^(by*19349663)^(axis*83492791))%3", "truth_was_production_input": False},
        "metric": {"tolerance_pixels": TOLERANCE_PX,
            "visibility_world_tolerance": VISIBILITY_WORLD_TOLERANCE,
            "rule": "continuous left feature ray first room face -> true right projection; both endpoints' +/-2px footprint must preserve face and integer material tile; forward reprojection <=2px and in-frame positive-depth; right projected-pixel first-hit face and point must equal left hit within1e-6 world units; scorable denominator includes stable endpoints, invisible truth is incorrect", 
            "limitations": "geometry-based approximate feature-centre consistency; conservative boundary exclusions; not exact SIFT footprint/material-state truth; no baseline truth comparison"},
        "production": _record(outcome), "entry": _record(outcome.entry), "session": outcome.entry.session_record,
        "connections": outcome.entry.graph.to_record(), "pose": outcome.entry.pose_graph.to_record(),
        "features": feature_records, "pair_observations": pair_records, "observer_calls": dict(calls),
        "calibration_shifts": calibration,
        "lattice": None if outcome.lattice is None else outcome.lattice.to_record(),
        "p0_profiles": [] if outcome.lattice is None else _profiles(outcome.lattice, outcome.entry.session),
        "geometry": None if outcome.geometry is None else _record(outcome.geometry), "geometry_stages": stages,
        "g_capture": g_record, "output": None if outcome.output is None else _record(outcome.output),
        "project": {"schema_version": persisted.schema_version, "blocks": len(persisted.volume.blocks),
            "image_derived_production": persisted.image_derived_production, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "export": export, "accuracy": "correspondence-centre metric only; no geometry/state accuracy claim",
        "limits": "one intervention/four frames/six pairs; <=256 delegates/8192 candidate edges per pair/8MB JSON cooperative assertions; no hard runtime containment or universal typed overflow"}
    assert {path: _file_sha(path) for path in PINS} == PINS
    assert {path: _file_sha(path) for path in source_pins} == source_pins
    encoded = json.dumps(_plain(proof), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    artifact = tmp_path/"aperiodic-material-gate.json"
    artifact.write_bytes(encoded)
    print("APERIODIC_ARTIFACT", str(artifact), len(encoded), hashlib.sha256(encoded).hexdigest())
    print("APERIODIC_RESULT", json.dumps({"stage": outcome.stage, "status": outcome.status,
        "reason": outcome.reason, "detail": outcome.detail, "calls": dict(calls),
        "pairs": [{"left": p["left"], "right": p["right"], "status": p["native_result"]["status"],
            "reason": p["native_result"]["reason"], "candidate_truth": p["candidate_truth"],
            "winner_truth": p["modes"][0]["truth_support"] if p["modes"] else None,
            "winner_class_correct": p["modes"][0]["yaw_class_correct"] if p["modes"] else None} for p in pair_records]}, sort_keys=True))
    assert len(encoded) < 8_000_000 and sum(calls.values()) <= 256
    assert all(len(p["native_graph"]["edges"]) <= 8192 for p in pair_records)
