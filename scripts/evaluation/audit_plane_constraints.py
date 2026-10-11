"""Sealed R8 arithmetic only: evaluate captured constraints, never solve/refit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation import attribute_plane_residuals as replay

REPLAY_SOURCE_SHA256 = "30ea9a6f50b667b445195b92ee9f0c37a4e325f3d7e49ee556ca97ef53a489f6"
STAGE_SHA256 = "9d6beca406eafaf7310d7964932effbb1d3d357eeeeae9dde3608a34ce70a7b8"
MAX_STAGE_BYTES = 1_048_576
ARITHMETIC_IDENTITY_TOLERANCE = 1e-12


def _require(value, reason):
    if not value:
        raise replay.ArtifactError(reason)


def verify_sources():
    replay.verify_source_pins()
    _require(hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest()
             == REPLAY_SOURCE_SHA256, "replay_source_sha256_mismatch")


def _endpoint(endpoint, readings, quadrants, translations, plane_graph):
    reading = readings[endpoint[0]]
    alternative = reading[5][endpoint[1]]
    _require(endpoint[2:5] == alternative and endpoint[5] == reading[4]
             and endpoint[7] == reading[3], "endpoint_reading_binding")
    frame, plane, local_axis = reading[1:4]
    world_axis, sign = plane_graph.axis_transform(local_axis, quadrants[frame])
    _require(sign == endpoint[6], "endpoint_sign_binding")
    b, distance, confidence = alternative
    offset = sign * reading[4] * distance
    translation = translations[frame][world_axis]
    return {"reading_index": endpoint[0], "alternative_index": endpoint[1],
        "frame": frame, "plane": plane, "local_axis": local_axis,
        "quadrant": quadrants[frame], "world_axis": world_axis, "sign": sign,
        "side": reading[4], "b": b, "distance": distance, "confidence": confidence,
        "offset": offset, "translation": translation, "face_scalar": translation+offset,
        "distance_per_b": distance/b, "implied_rectified_units_per_period": b/distance}


def _constraint_row(edge, readings, quadrants, translations, faces, plane_graph):
    """Pure substitution into an existing equation, no candidate modification."""
    _require(edge[4] is True, "inactive_constraint")
    left = _endpoint(edge[1], readings, quadrants, translations, plane_graph)
    right = _endpoint(edge[2], readings, quadrants, translations, plane_graph)
    _require(left["world_axis"] == right["world_axis"] and
             (right["quadrant"]-left["quadrant"]) % 4 == edge[3], "active_equation_binding")
    rows = faces[edge[0]]
    _require(len(rows) == 2, "face_pair_count")
    for endpoint in (left, right):
        identity = [endpoint[key] for key in ("frame", "plane", "local_axis", "b",
                                              "distance", "side", "sign")]
        captured = [row for row in rows if row[1:8] == identity]
        _require(len(captured) == 1, "face_endpoint_identity")
        _require(captured[0][16] == endpoint["face_scalar"], "captured_translation_face_equation")
    target = left["offset"]-right["offset"]
    observed = right["translation"]-left["translation"]
    residual = observed-target
    face_gap = right["face_scalar"]-left["face_scalar"]
    _require(math.isclose(residual, face_gap, rel_tol=0., abs_tol=ARITHMETIC_IDENTITY_TOLERANCE),
             "linear_face_gap_identity")
    return {"match_index": edge[0], "world_axis": left["world_axis"], "left": left, "right": right,
        "relative_quadrant": edge[3], "target_delta": target, "captured_translation_delta": observed,
        "signed_linear_residual": residual, "absolute_linear_residual": abs(residual),
        "captured_right_minus_left_face": face_gap,
        "wrapped_face_pair_gap": abs((face_gap+.5) % 1.-.5),
        "paired_phasor_resultant": abs(math.cos(math.pi*face_gap))}


def _stage_evidence(stage, artifact):
    _require(stage.stat().st_size <= MAX_STAGE_BYTES, "stage_byte_cap")
    raw = stage.read_bytes()
    _require(len(raw) <= MAX_STAGE_BYTES, "stage_byte_cap")
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == STAGE_SHA256 == artifact["project_sha256"], "stage_project_digest")
    value = json.loads(raw)
    session = value["image_derived_session"]
    return {"sha256": actual, "bytes": len(raw), "schema_version": value["schema_version"],
        "session_field_names": sorted(session), "persisted_component_count": len(session["components"]),
        "raw_pixel_period_or_patch_evidence_retained": False,
        "camera_and_frame_values_used": False, "source_images_opened": False}


def _audit_validated(artifact):
    observation = artifact["solver_failure_observation"]
    inputs, phase = observation["solver_input_bindings"], observation["phase_capture"]
    _, plane_graph = replay._load_production_validators(ROOT / "src")
    readings = {row[0]: row for row in inputs["readings"]}
    quadrants = dict(zip(inputs["frames"], inputs["assignments"]))
    translations = dict(phase["translations"])
    _require(phase["frames"] == inputs["frames"] and phase["assignments"] == inputs["assignments"],
             "captured_frame_assignment_identity")
    faces = {}
    for face in phase["faces"]:
        faces.setdefault(face[0], []).append(face)
    active = [edge for edge in inputs["edges"] if edge[4] is True]
    _require(len(active) == inputs["active_edge_count"] == 10 and len(faces) == 10,
             "active_constraint_count")
    rows = [_constraint_row(edge, readings, quadrants, translations, faces, plane_graph)
            for edge in active]
    _require([row["match_index"] for row in rows] == sorted(faces), "active_constraint_order")
    axes = []
    recorded_axes = {row[0]: row for row in phase["axes"]}
    for axis in range(3):
        same = [row for row in rows if row["world_axis"] == axis]
        residuals = [row["signed_linear_residual"] for row in same]
        bound = math.fsum(row["paired_phasor_resultant"] for row in same)/len(same)
        _require(recorded_axes[axis][3] <= bound+ARITHMETIC_IDENTITY_TOLERANCE,
                 "paired_phasor_triangle_bound")
        axes.append({"axis": axis, "constraints": len(same), "signed_residuals": residuals,
            "rms_linear_residual": math.sqrt(math.fsum(r*r for r in residuals)/len(residuals)),
            "max_absolute_linear_residual": max(map(abs, residuals)),
            "paired_phasor_resultant_upper_bound": bound,
            "captured_global_resultant": recorded_axes[axis][3],
            "captured_global_phase_residual": recorded_axes[axis][4]})
    axis1 = [row for row in rows if row["world_axis"] == 1]
    f0, f1, f2 = inputs["frames"]
    pairs = {(r["left"]["frame"], r["right"]["frame"]): r for r in axis1}
    _require(len(axis1) == len(pairs) == 3 and set(pairs) == {(f0, f1), (f0, f2), (f1, f2)},
             "axis1_captured_triangle_identity")
    triangle = [(pairs[(f0, f1)], 1), (pairs[(f0, f2)], -1), (pairs[(f1, f2)], 1)]
    target_closure = math.fsum(sign*row["target_delta"] for row, sign in triangle)
    translation_closure = math.fsum(sign*row["captured_translation_delta"] for row, sign in triangle)
    residual_closure = math.fsum(sign*row["signed_linear_residual"] for row, sign in triangle)
    _require(abs(translation_closure) <= ARITHMETIC_IDENTITY_TOLERANCE and
             math.isclose(residual_closure, -target_closure, rel_tol=0., abs_tol=ARITHMETIC_IDENTITY_TOLERANCE),
             "captured_triangle_closure_identity")
    plane_use = {}
    for row in rows:
        for endpoint in (row["left"], row["right"]):
            key = (endpoint["frame"], endpoint["plane"])
            if key not in plane_use:
                reading = readings[endpoint["reading_index"]]
                ratios = [alternative[1]/alternative[0] for alternative in reading[5]]
                plane_use[key] = {"frame": key[0], "plane": key[1], "local_axis": endpoint["local_axis"],
                    "side": endpoint["side"], "selected_b": [], "active_match_indices": [],
                    "captured_alternative_distance_per_b_range": [min(ratios), max(ratios)]}
            use = plane_use[key]
            use["selected_b"].append(endpoint["b"])
            use["active_match_indices"].append(row["match_index"])
    for use in plane_use.values():
        use["selected_b"] = sorted(set(use["selected_b"]))
        use["multiple_selected_b"] = len(use["selected_b"]) > 1
    return {"constraints": rows, "axes": axes, "plane_uses": [plane_use[k] for k in sorted(plane_use)],
        "axis1_triangle": {"match_indices_and_signs": [[r["match_index"], s] for r, s in triangle],
            "target_closure_blocks": target_closure, "captured_translation_closure_blocks": translation_closure,
            "signed_residual_closure_blocks": residual_closure,
            "absolute_residual_range_blocks": [min(r["absolute_linear_residual"] for r in axis1),
                                                max(r["absolute_linear_residual"] for r in axis1)]}}


def audit(path=ROOT / replay.ARTIFACT_RELATIVE):
    verify_sources()
    artifact = replay.validate_pinned_artifact(path)
    stage_evidence = _stage_evidence(path.parent / "stage.mcscan", artifact)
    result = _audit_validated(artifact)
    result.update({"schema": "mc-photogrammetry-fixed-plane-constraint-audit-v1", "status": "complete",
        "artifact_sha256": replay.ARTIFACT_SHA256, "phase_capture_sha256": replay.PHASE_CAPTURE_SHA256,
        "solver_input_bindings_sha256": replay.SOLVER_INPUT_BINDINGS_SHA256,
        "input_match_identity_sha256": replay.INPUT_MATCH_IDENTITY_SHA256,
        "fixed_captured_translations": artifact["solver_failure_observation"]["phase_capture"]["translations"],
        "stage_evidence": stage_evidence,
        "arithmetic_identity_tolerance": ARITHMETIC_IDENTITY_TOLERANCE,
        "solver_fit": False, "edge_deletion": False, "b_changed": False, "reweighted": False,
        "images_or_truth_read": False, "missing_period_reconstructed": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = audit(args.repo_root / replay.ARTIFACT_RELATIVE)
    except (replay.ArtifactError, OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        result = {"schema": "mc-photogrammetry-fixed-plane-constraint-audit-v1", "status": "refused",
                  "reason": str(exc)}
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
