"""One fixed real factory pair; recorded capture cameras are posthoc only."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json

import numpy as np
from PIL import Image

from test_image_only_multiview_track_quality import PINS as OLD_PINS
from test_image_only_aperiodic_material_gate import _plain, _array_sha
from test_image_only_g_prefilter_capture import _file_sha
from test_image_only_multiview_gate import ROOT, _record
from mc_photogrammetry import evaluation_same_axis_continuation as continuation
from mc_photogrammetry.pipeline import image_derived_entry as entry
from mc_photogrammetry.pipeline import image_derived_production as production
from mc_photogrammetry.pipeline import image_derived_session as sessions
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import joint_matching as joint
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix

BASE = "exports/datasets/fresh-c4-canal-town-framed-2026-09-12"
SCENE = BASE + "/scenes/canal-town-architectural-component-bbox"
INPUTS = [SCENE + "/captures/view-canal-town-architectural-component-bbox-" + n + ".png" for n in ("05", "06")]
ASSETS = {
    BASE + "/index.json": "0637caba1a523bbdf7351dedd7c4f93bfdb611f0b549fc54b3a781f3d25ee46b",
    SCENE + "/capture.json": "c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6",
    INPUTS[0]: "6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b",
    INPUTS[1]: "53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923",
}
PINS = dict(OLD_PINS, **{
    "tests/test_image_only_multiview_track_quality.py": "24e2165aabcd7fd916ea94a84d17e6b12f158f885004eb1db1b35fe0d11b394b",
    "docs/reports/SOL-IMG-MULTIVIEW-TRACK-QUALITY-20261005.json": "c830218dece213c2271f41cba184fdbf2998a1139c1607d1addb336ae2c6d53f",
    "docs/reports/SOL-IMG-MULTIVIEW-TRACK-QUALITY-20261005.txt": "d6a8f4f46689d24e6dedbec29025c125381894206fd5b326f0816e41434a30c8",
})


def _angle(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1., 1.))))


def _rotation_error(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T)-1.)/2., -1., 1.))))


def test_default_image_only_factory_camera_pair(tmp_path, monkeypatch):
    assert {p: _file_sha(p) for p in PINS} == PINS
    assert {p: _file_sha(p) for p in ASSETS} == ASSETS
    previous = json.loads((ROOT/"docs/reports/SOL-IMG-APERIODIC-MATERIAL-GATE-20261005.json").read_bytes())
    source_pins = previous["source_pins"]
    assert {p: _file_sha(p) for p in source_pins} == source_pins
    index = json.loads((ROOT/BASE/"index.json").read_bytes())
    capture = json.loads((ROOT/SCENE/"capture.json").read_bytes())
    scene = index["scenes"][0]
    assert scene["directory"] == "scenes/canal-town-architectural-component-bbox"
    assert scene["hashes"]["capture.json"] == ASSETS[SCENE+"/capture.json"]
    bindings = []
    for slot, ordinal in enumerate((5, 6)):
        path = ROOT/INPUTS[slot]
        frame = capture["frames"][ordinal]
        item = scene["screenshots"][ordinal]
        actual = frame["actual"]
        assert path.name == frame["filename"] == item["filename"]
        assert frame["frame_id"] == item["frame_id"]
        assert ASSETS[INPUTS[slot]] == item["sha256"] == item["source_sha256"] == actual["screenshot_sha256"]
        assert path.stat().st_size == item["bytes"] == actual["screenshot_bytes"]
        assert frame["captured"] and not frame["errors"] and actual["pose_converged"]
        assert actual["hud_hidden"] and actual["active_resource_packs"] == ["vanilla"]
        assert not actual["resource_pack_applied"] and not actual["shader_pack_applied"]
        with Image.open(path) as image:
            assert image.size == (actual["framebuffer_width"], actual["framebuffer_height"]) == (1600, 900)
            rgb = np.asarray(image.convert("RGB"))
        bindings.append({"path": INPUTS[slot], "pipeline_frame": f"image-{slot:04d}",
            "capture_frame": frame, "recorded_camera_intrinsics": capture["cameras"][ordinal]["intrinsics"],
            "recorded_roll_degrees": capture["cameras"][ordinal]["roll_degrees"],
            "rgb_sha256": sessions.image_sha256(rgb), "encoded_sha256": ASSETS[INPUTS[slot]]})
    originals = (entry.match_and_solve_pair, joint.detect_features, joint.reciprocal_topk_graph,
                 joint._score_modes, plane_graph.match_plane_graph, plane_graph.solve_component)
    calls, active, packets, features, g = Counter(), {}, [], {}, {}

    def detect(frame, image, **kwargs):
        calls["detect"] += 1
        before = sessions.image_sha256(image)
        result = originals[1](frame, image, **kwargs)
        assert before == sessions.image_sha256(image)
        features[frame] = result
        active.setdefault("features", []).append(result)
        return result

    def candidates(left, right, **kwargs):
        calls["candidate_graph"] += 1
        before = (_array_sha(left), _array_sha(right))
        result = originals[2](left, right, **kwargs)
        assert before == (_array_sha(left), _array_sha(right))
        active.update(graph=result, graph_kwargs=kwargs)
        return result

    def scores(modes, best, refusal, floor, bounds):
        calls["score_modes"] += 1
        before = [(_array_sha(m.matches), _array_sha(m.selection)) for m in modes]
        result = originals[3](modes, best, refusal, floor, bounds)
        assert before == [(_array_sha(m.matches), _array_sha(m.selection)) for m in modes]
        active.update(modes=tuple(modes), scores=result, best=best, settings=asdict(bounds))
        return result

    def pair(left_id, left_image, right_id, right_image, **kwargs):
        calls["pair"] += 1
        assert calls["pair"] == 1 and not active
        assert (left_id, right_id) == ("image-0000", "image-0001")
        assert set(kwargs) == {"left_camera", "right_camera"}
        before = (sessions.image_sha256(left_image), sessions.image_sha256(right_image))
        assert before == tuple(b["rgb_sha256"] for b in bindings)
        active.update(camera_kwargs=kwargs, image_sha256=before)
        result = originals[0](left_id, left_image, right_id, right_image, **kwargs)
        assert before == (sessions.image_sha256(left_image), sessions.image_sha256(right_image))
        if "best" in active:
            assert np.array_equal(result.matches, active["best"].matches)
        packets.append(dict(active, result=result))
        active.clear()
        return result

    def matcher(readings, signatures, **kwargs):
        calls["g_matcher"] += 1
        assert calls["g_matcher"] == 1 and kwargs == {"max_shift_cells": 32}
        before = continuation._source_sha(readings, signatures)
        sink = []
        result = originals[4](readings, signatures, **kwargs, candidate_observation_sink=sink)
        assert len(sink) == 1
        pool = sink[0]
        kept = tuple(m for m, d in zip(pool.candidates, pool.dispositions) if d == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
        assert kept == result and all(a is b for a, b in zip(kept, result))
        assert before == continuation._source_sha(readings, signatures)
        g.update(readings=readings, signatures=signatures, source_sha256=before, pool=pool, matches=result)
        return result

    def solver(readings, matches, **kwargs):
        calls["g_solver"] += 1
        assert readings is g["readings"] and matches is g["matches"]
        before = [plane_graph._indexed_match_payload(m) for m in matches]
        result = originals[5](readings, matches, **kwargs)
        assert before == [plane_graph._indexed_match_payload(m) for m in matches]
        g.update(solution=result, solver_kwargs=kwargs, inputs=before)
        return result

    # The public call receives only the two untouched PNG paths and fresh target.
    # No metadata, cameras, FOV, coordinates, scale or mask can enter this call.
    target = tmp_path/"factory-pair.mcscan"
    raised = None
    with monkeypatch.context() as local:
        for module, name, fn in ((entry, "match_and_solve_pair", pair), (joint, "detect_features", detect),
                (joint, "reciprocal_topk_graph", candidates), (joint, "_score_modes", scores),
                (plane_graph, "match_plane_graph", matcher), (plane_graph, "solve_component", solver)):
            local.setattr(module, name, fn)
        try:
            outcome = production.derive_image_output_from_paths([ROOT/p for p in INPUTS], project_path=target)
        except entry.ImageDerivedEntryError as error:
            raised = {"type": type(error).__name__, "reason": error.reason, "detail": error.detail}
            outcome = None
    assert (entry.match_and_solve_pair, joint.detect_features, joint.reciprocal_topk_graph,
        joint._score_modes, plane_graph.match_plane_graph, plane_graph.solve_component) == originals
    # Freeze native records before constructing any oracle camera/rotation.
    native_records = [{"result": _plain(asdict(p["result"])), "graph": _plain(asdict(p["graph"])),
        "cameras": _plain({k: asdict(v) for k, v in p["camera_kwargs"].items()}),
        "settings": p.get("settings"), "graph_kwargs": p["graph_kwargs"],
        "modes": [{"mode": _plain(asdict(m)), "score": _plain(asdict(s)), "winner": m is p["best"]}
            for m, s in zip(p.get("modes", ()), p.get("scores", ()))]} for p in packets]
    frozen = json.dumps(native_records, sort_keys=True, allow_nan=False).encode()
    oracle = [CameraModel(position=tuple(b["capture_frame"]["actual"]["eye_position"]), eye_height=0.,
        yaw_degrees=b["capture_frame"]["actual"]["yaw_degrees"],
        pitch_degrees=b["capture_frame"]["actual"]["pitch_degrees"],
        roll_degrees=b["recorded_roll_degrees"], fov_degrees=b["capture_frame"]["actual"]["fov_degrees"])
        for b in bindings]
    truth_delta = np.asarray(oracle[1].position)-oracle[0].position
    calibration, comparisons = [], []
    for packet, record in zip(packets, native_records):
        estimates = [packet["camera_kwargs"][k] for k in ("left_camera", "right_camera")]
        rotations = [joint._world_to_camera(e) for e in estimates]
        shifts = [int(round((t.yaw_degrees-e.pose.yaw_degrees)/90)) % 4 for t, e in zip(oracle, estimates)]
        expected_q = (shifts[1]-shifts[0]) % 4
        truth_relative = camera_rotation_matrix(oracle[1]) @ camera_rotation_matrix(oracle[0]).T
        truth_aligned = rotations[0].T @ camera_rotation_matrix(oracle[0]) @ truth_delta
        for ordinal, (e, t) in enumerate(zip(estimates, oracle)):
            k = bindings[ordinal]["recorded_camera_intrinsics"]
            assert np.allclose(k["focal_length_pixels"], 450/np.tan(np.radians(t.fov_degrees/2)), rtol=0, atol=1e-9)
            calibration.append({"frame": f"image-{ordinal:04d}", "native": _plain(asdict(e)),
                "focal_relative_errors": (np.asarray(e.intrinsics.focal_length_pixels)/k["focal_length_pixels"]-1).tolist(),
                "principal_point_errors_pixels": (np.asarray(e.intrinsics.principal_point_pixels)-k["principal_point_pixels"]).tolist(),
                "yaw_error_mod90_degrees": (e.pose.yaw_degrees-t.yaw_degrees+45)%90-45,
                "pitch_error_degrees": e.pose.pitch_degrees-t.pitch_degrees,
                "roll_error_degrees": e.pose.roll_degrees-t.roll_degrees})
        for mode in packet.get("modes", ()):
            chosen = packet["graph"].edges[mode.selection]
            left, right = packet["features"]
            assert np.array_equal(mode.matches, np.column_stack((left.points[chosen[:, 0]], right.points[chosen[:, 1]])))
            relative = (rotations[1] @ joint._ry90(mode.yaw_class)) @ rotations[0].T
            comparisons.append({"yaw_class": mode.yaw_class, "mode_index": mode.mode_index,
                "winner": mode is packet["best"], "nearest_truth_relative_quarter": expected_q,
                "quarter_correct": mode.yaw_class == expected_q,
                "relative_rotation_error_degrees": _rotation_error(relative, truth_relative),
                "signed_baseline_direction_error_degrees_in_left_gauge": _angle(mode.baseline_direction, truth_aligned),
                "baseline_interpretation_requires_correct_quarter_and_calibration": True})
    assert frozen == json.dumps(native_records, sort_keys=True, allow_nan=False).encode()
    g_record = None
    if g:
        g_record = {"source_sha256": g["source_sha256"], "reading_count": len(g["readings"]),
            "signature_count": len(g["signatures"]), "prefilter_rows": [dict(plane_graph._indexed_match_payload(m), disposition=d)
                for m, d in zip(g["pool"].candidates, g["pool"].dispositions)],
            "kept_rows": [plane_graph._indexed_match_payload(m) for m in g["matches"]]}
        if "solution" in g:
            g_record.update(solution=graph_mod._solution_payload(g["solution"]), solver_kwargs=g["solver_kwargs"], solver_inputs=g["inputs"])
    export = None
    if outcome is not None and outcome.output is not None and getattr(outcome.output.promotion, "export_ready", False):
        from mc_photogrammetry.pipeline.image_derived_delivery import deliver_image_derived_litematic
        export = _record(deliver_image_derived_litematic(outcome.output.promotion, tmp_path/"factory-pair.litematic",
            name="factory-pair", description="Fixed factory pair default pipeline"))
    proof = {"task": "SOL-IMG-FACTORY-CAMERA-PAIR-20261005", "prior_pins": PINS, "source_pins": source_pins,
        "asset_pins": ASSETS, "bindings": bindings, "scene_group": scene["scene_group"], "provenance": scene["provenance"],
        "public_inputs": {"paths": INPUTS, "kwargs": ["project_path"], "metadata_supplied": False},
        "observer_calls": dict(calls), "native_records_sha256_before_oracle": hashlib.sha256(frozen).hexdigest(),
        "pairs": native_records, "features": {f: {"points": v.points.tolist(), "sizes": None if v.sizes is None else v.sizes.tolist(),
            "descriptors_sha256": _array_sha(v.descriptors), "native": {"width": v.width, "height": v.height,
                "usable_pixels": v.usable_pixels, "boundary_dropped": v.boundary_dropped}} for f, v in features.items()},
        "calibration": calibration, "mode_posthoc_comparisons": comparisons,
        "oracle_baseline_blocks": float(np.linalg.norm(truth_delta)),
        "public_accepted_pose": None if not packets or not packets[0]["result"].solved else _plain(asdict(packets[0]["result"].pose)),
        "production": None if outcome is None else _record(outcome), "raised_entry_error": raised,
        "session": None if outcome is None else outcome.entry.session_record,
        "connections": None if outcome is None else outcome.entry.graph.to_record(),
        "pose_graph": None if outcome is None else outcome.entry.pose_graph.to_record(),
        "g_capture": g_record, "export": export,
        "limits": "single fixed actual-image pair; 16MB/call<=256 cooperative publication assertions, not runtime containment; descriptors hashes only; no pre-admission G trials",
        "caveats": "Recorded camera oracle only; capture intrinsics imply ideal pinhole FOV, not independent optical calibration. Native confidence is deliberately zero. Nearest quarter conditional on calibration. Refused modes are proposals, not accepted poses. No world/reference read; occlusion/correspondence/geometry/state accuracy unmeasured."}
    assert {p: _file_sha(p) for p in PINS} == PINS
    assert {p: _file_sha(p) for p in ASSETS} == ASSETS
    assert {p: _file_sha(p) for p in source_pins} == source_pins
    encoded = json.dumps(_plain(proof), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    artifact = tmp_path/"factory-camera-pair.json"
    artifact.write_bytes(encoded)
    print("FACTORY_ARTIFACT", artifact, len(encoded), hashlib.sha256(encoded).hexdigest())
    print("FACTORY_RESULT", json.dumps({"terminal": proof["production"], "error": raised, "calls": dict(calls),
        "pairs": [{"status": p["result"]["status"], "reason": p["result"]["reason"], "support": p["result"]["support"],
            "candidate_edges": p["result"]["candidate_edges"]} for p in native_records]}, sort_keys=True))
    assert len(encoded) < 16_000_000 and sum(calls.values()) <= 256
    assert calls["pair"] <= 1 and calls["detect"] <= 2
