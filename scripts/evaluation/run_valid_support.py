"""Supplied-plan CAM-VALID driver; no cohort builder or automatic run entry.

Calling requires a separately approved frozen plan and experiment GO. Imports
never load the native harness; truth callbacks wait for 24 durable readbacks.
"""
from __future__ import annotations

import ast
import base64
from dataclasses import asdict, dataclass
import hashlib
import inspect
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
from types import CodeType

import numpy as np
from scripts.evaluation import valid_camera_scoring as scoring

from mc_photogrammetry.reconstruction import evaluation_gui_masked_segments as support
from mc_photogrammetry.reconstruction import evaluation_valid_stencil_segments as stencil
from mc_photogrammetry.reconstruction.evaluation_valid_stencil_segments import (
    AcquisitionTerminalError, AcquisitionIntegrityError, AcquisitionCapError,
    AllocationBudget, FROZEN_CONFIG,
)

ROOT = Path(__file__).resolve().parents[2]
STUDY = "CAM-VALID-SUPPORT-05"
SCHEMA = "CAM-VALID-plan-v1"
ARMS = ("baseline", "candidate")
ATTESTATION_RESERVE = 16*1024*1024
SOURCES = tuple(sorted((*support._EXPECTED_DEPENDENCY_SHA256,
    "src/mc_photogrammetry/reconstruction/evaluation_valid_stencil_segments.py",
    "src/mc_photogrammetry/reconstruction/evaluation_gui_masked_segments.py",
    "scripts/evaluation/run_valid_support.py",
    "scripts/evaluation/valid_camera_trace.py",
    "scripts/evaluation/valid_camera_scoring.py")))
SOLVER = {"minimum_segments": 12, "minimum_length": 40.0,
    "sample_spacing": 0.5, "boundary_radius": 3,
    "trace_policy": {"max_score_events": 120000, "max_stage_events": 32,
        "max_array_bytes": 2000000, "max_trace_bytes": 64000000, "max_segments": 10000}}
POLICY = {"camera_gates": {"focal_relative_error": .05,
    "orientation_degrees_mod90": 1., "principal_error_over_diagonal": .01},
    "success": {"candidate_passes_min": 9, "net_added_passes_min": 3,
    "baseline_passes_lost_max": 0, "newly_wrong_stress_cases_max": 0,
    "candidate_stress_wrong_count_le_baseline": True}}
MUTABLE = {"execution_state", "sealed_predictions", "seal_validation", "evaluations", "summary", "failure"}
CORE = {"schema", "study", "config", "config_sha256", "solver", "policy",
    "source_sha256", "runtime", "paths", "retired_paths", "retired_rgb_sha256",
    "registry", "truth_registry", "score_sites", "execution_contract"}
_RUN_LOCK = threading.Lock()


def require(ok, reason):
    if not ok:
        raise AcquisitionIntegrityError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def runtime_pins():
    # Distribution metadata avoids importing OpenCV during plan inspection.
    distribution = metadata.distribution("opencv-python")
    binaries = [distribution.locate_file(p) for p in distribution.files
        if str(p).startswith("cv2/") and str(p).endswith((".pyd", ".so"))]
    require(len(binaries) == 1, "opencv_binary_identity")
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "opencv_binary_sha256": support._file_sha256(binaries[0]),
        "distributions": {name: metadata.version(name) for name in ("numpy", "opencv-python", "Pillow")}}


def source_pins():
    return {name: support._file_sha256(ROOT / name) for name in SOURCES}


def score_sites():
    tree = ast.parse((ROOT / "src/mc_photogrammetry/reconstruction/calibration.py").read_text(encoding="utf-8"))
    result = []
    for function, tags in (("estimate_camera", ("coarse", "coarse-full-rescore", "fine")),
        ("refine_optics", ("refine-base", "refine-roll", "refine-principal", "refine-orientation"))):
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
        require(len(matches) == 1, "score_function_drift")
        calls = sorted((n for n in ast.walk(matches[0]) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "_score"), key=lambda n: n.lineno)
        require(len(calls) == len(tags), "score_call_drift")
        for node, tag in zip(calls, tags):
            result.append({"function": function, "start": node.lineno, "end": node.end_lineno,
                "tag": tag, "ast_sha256": sha(ast.dump(node, include_attributes=False))})
    return result


def preregistration_digest(plan):
    """Digest supplied immutable inputs; creates no plan or case."""
    require(CORE <= plan.keys(), "plan_core_missing")
    return sha(canonical({k: plan[k] for k in sorted(CORE)}))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _path(value):
    require(isinstance(value, str) and str(Path(value).resolve()) == value, "noncanonical_path")
    return Path(value)


def _rgb(row):
    raw = base64.b64decode(row["rgb_base64"], validate=True)
    require(len(raw) == math.prod(row["rgb_shape"]) and sha(raw) == row["rgb_sha256"], "rgb_hash")
    return np.frombuffer(raw, dtype=np.uint8).reshape(row["rgb_shape"])


def validate_plan(plan, plan_path, *, fresh=True):
    require(isinstance(plan, dict) and set(plan) == CORE | MUTABLE | {"preregistration_sha256"}, "plan_schema")
    require(plan["schema"] == SCHEMA and plan["study"] == STUDY, "study_schema_pin")
    require(plan["config"] == asdict(FROZEN_CONFIG) and plan["config_sha256"] == FROZEN_CONFIG.sha256, "config_pin")
    require(plan["solver"] == SOLVER and plan["policy"] == POLICY, "solver_policy_pin")
    require(plan["source_sha256"] == source_pins(), "source_drift")
    require(plan["runtime"] == runtime_pins(), "runtime_drift")
    require(plan["score_sites"] == score_sites(), "score_site_pin")
    require(plan["execution_contract"] == execution_contract(), "execution_contract_pin")
    require(_digest(plan["preregistration_sha256"]) and plan["preregistration_sha256"] == preregistration_digest(plan), "preregistration_pin")
    require(set(plan["paths"]) == {"plan", "sidecar", "claim"}, "path_schema")
    paths = {k: _path(v) for k, v in plan["paths"].items()}
    require(paths["plan"] == Path(plan_path).resolve() and len(set(paths.values())) == 3, "path_pin")
    require(paths["sidecar"].parent == paths["plan"].parent == paths["claim"].parent, "path_parent")
    retired = plan["retired_paths"]
    require(isinstance(retired, list) and len(retired) == len(set(retired)), "retired_path_schema")
    retired = [_path(v) for v in retired]
    for path in paths.values():
        require(not any(path == old or old in path.parents for old in retired), "retired_path")
        require(not any(tag in path.name.upper() for tag in
            ("CAM-SUPPORT", "CAM-BOUNDARY", "CAM-MD-", "R0", "R1", "R2", "R3")), "prior_study_path")
        require(not path.is_symlink(), "symlink_path")
    if fresh:
        require(plan["execution_state"] == "NOT_STARTED" and plan["sealed_predictions"] == []
            and all(plan[k] is None for k in ("seal_validation", "evaluations", "summary", "failure")), "attempt_consumed")
        require(not paths["claim"].exists(), "claim_consumed")
        require(not paths["sidecar"].exists() or paths["sidecar"].stat().st_size == 0, "nonempty_sidecar")
    retired_rgb = plan["retired_rgb_sha256"]
    require(isinstance(retired_rgb, list) and all(_digest(v) for v in retired_rgb), "retired_rgb_schema")
    rows = plan["registry"]
    require(isinstance(rows, list) and len(rows) == 12, "registry_count")
    row_keys = {"identity", "family", "base_identity", "stratum", "rgb_shape", "rgb_sha256",
        "rgb_base64", "mask_sha256", "expanded_mask_sha256"}
    identities = set(); hashes = set(); families = set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == row_keys, "registry_schema")
        require(all(isinstance(row[k], str) and 0 < len(row[k]) <= 256 for k in ("identity", "family")), "registry_identity")
        require(row["identity"] not in identities and row["rgb_sha256"] not in hashes
            and row["rgb_sha256"] not in retired_rgb, "duplicate_or_retired_rgb_identity")
        identities.add(row["identity"]); hashes.add(row["rgb_sha256"])
        require(row["stratum"] == ("ordinary" if index < 6 else "stress"), "stratum_order")
        if index < 6:
            require(row["base_identity"] is None and row["family"] not in families, "ordinary_family")
            families.add(row["family"])
        else:
            bases = [b for b in rows[:6] if b["identity"] == row["base_identity"]]
            require(len(bases) == 1 and bases[0]["family"] == row["family"], "stress_family")
        shape = row["rgb_shape"]
        require(isinstance(shape, list) and len(shape) == 3 and shape[2] == 3
            and all(type(v) is int and v > 0 for v in shape) and max(shape[:2]) <= FROZEN_CONFIG.max_dimension
            and shape[0]*shape[1] <= FROZEN_CONFIG.max_pixels, "rgb_shape_cap")
        require(all(_digest(row[k]) for k in ("rgb_sha256", "mask_sha256", "expanded_mask_sha256")), "registry_hash")
        require(isinstance(row["rgb_base64"], str) and len(row["rgb_base64"]) == 4*((math.prod(shape)+2)//3), "rgb_encoding_length")
        rgb = _rgb(row)
        del rgb
    truth = plan["truth_registry"]
    require(isinstance(truth, list) and len(truth) == 12, "truth_registry_count")
    for row, ref in zip(rows, truth):
        require(isinstance(ref, dict) and set(ref) == {"identity", "reference", "sha256"}
            and ref["identity"] == row["identity"] and isinstance(ref["reference"], str)
            and 0 < len(ref["reference"]) <= 4096 and _digest(ref["sha256"]), "truth_reference_schema")
    return paths


@dataclass(frozen=True)
class Dependencies:
    trace_rgb: object
    trace_module: object
    calibration: object
    gui_mask: object


def _bindings(plan, row, arm):
    return {k: row[k] for k in ("identity", "family", "base_identity", "stratum", "rgb_sha256", "rgb_shape")} | {
        "arm": arm, "preregistration_sha256": plan["preregistration_sha256"],
        "source_sha256": sha(canonical(plan["source_sha256"])), "runtime_sha256": sha(canonical(plan["runtime"]))}


def _array(meta, budget):
    size = meta["byte_count"]
    budget.acquire(3*size + 2048)
    dtype = np.dtype([tuple(v) for v in meta["dtype_descr"]]) if meta["dtype_descr"] else np.dtype(meta["dtype"])
    raw = base64.b64decode(meta["base64"], validate=True)
    require(len(raw) == size and sha(raw) == meta["sha256"], "replay_array_hash")
    return np.frombuffer(raw, dtype=dtype).reshape(meta["shape"])


def _finite(value):
    if value is None or isinstance(value, (str, bool)):
        return True
    if type(value) in (int, float):
        return math.isfinite(value)
    if isinstance(value, (list, dict)):
        return all(_finite(v) for v in (value.values() if isinstance(value, dict) else value))
    return False


def _prediction(prediction, shape, emitted):
    keys = {"focal_pixels", "yaw_degrees", "pitch_degrees", "width", "height", "inliers", "segments",
        "median_residual", "per_axis_inliers", "roll_degrees", "principal_point", "notes"}
    require(isinstance(prediction, dict) and set(prediction) == keys and _finite(prediction), "prediction_schema_finite")
    require(type(prediction["width"]) is int and type(prediction["height"]) is int
        and (prediction["height"], prediction["width"]) == tuple(shape[:2]), "prediction_frame")
    require(type(prediction["segments"]) is int and prediction["segments"] == emitted >= 12
        and type(prediction["inliers"]) is int and 0 <= prediction["inliers"] <= emitted, "prediction_counts")
    require(type(prediction["focal_pixels"]) in (int, float) and prediction["focal_pixels"] > 0, "prediction_physical")
    axis = prediction["per_axis_inliers"]
    require(isinstance(axis, list) and len(axis) == 3 and all(type(v) is int and 0 <= v <= emitted for v in axis)
        and sum(axis) == prediction["inliers"], "prediction_axis_counts")
    residual = prediction["median_residual"]
    require((prediction["inliers"] == 0 and axis == [0, 0, 0] and residual is None)
        or (prediction["inliers"] > 0 and type(residual) in (int, float)
            and math.isfinite(residual) and residual >= 0), "prediction_residual_consistency")
    require(all(type(prediction[k]) in (int, float) for k in ("yaw_degrees", "pitch_degrees", "roll_degrees")), "prediction_angles")
    point = prediction["principal_point"]
    require(point is None or isinstance(point, list) and len(point) == 2
        and all(type(v) in (int, float) for v in point), "prediction_principal")
    require(isinstance(prediction["notes"], list) and all(isinstance(v, str) for v in prediction["notes"]), "prediction_notes")


def _candidate_replay(arrays, counts, shape, budget):
    keys = {"footprint_frame", "footprint_original", "footprint_band", "footprint_valid",
        "below_threshold", "nms_suppressed", "edgels", "neighbor_probes", "graph_edges", "paths",
        "singletons", "cycle_closures", "fit_nodes", "fit_points", "degenerate_groups", "pre_length",
        "duplicate", "frame", "original", "band", "short", "admitted", "projection_samples"}
    require(isinstance(counts, dict) and set(counts) == keys
        and all(type(v) is int and v >= 0 for v in counts.values()), "acquisition_counts_schema")
    specs = {"edgels": stencil.EDGEL, "graph_edges": stencil.EDGE, "paths": stencil.PATH,
        "path_membership": np.dtype("<u4"), "cycle_closures": stencil.EDGE, "fit_groups": stencil.GROUP,
        "support_membership": np.dtype("<u4"), "line_ledger": stencil.LINE,
        "projection_occupancy": np.dtype("u1"), "admitted_candidate_ids": np.dtype("<u4")}
    require(set(arrays) == set(specs) | {"original_mask", "boundary_band", "expanded_mask", "detector_segments",
        "emitted_segments", "run_ledger", "sample_occupancy", "candidate_pre_length"}, "candidate_array_schema")
    for name, dtype in specs.items():
        require(arrays[name].dtype == dtype and arrays[name].ndim == 1, "candidate_array_dtype")
    require(sum(counts[k] for k in ("footprint_frame", "footprint_original", "footprint_band", "footprint_valid")) == math.prod(shape[:2])
        and counts["below_threshold"] + counts["nms_suppressed"] + counts["edgels"] == counts["footprint_valid"], "footprint_counts")
    for name in ("edgels", "graph_edges", "paths", "cycle_closures"):
        require(counts[name] == len(arrays[name]), "candidate_count_binding")
    require(counts["fit_nodes"] == len(arrays["fit_groups"])
        and counts["pre_length"] == len(arrays["candidate_pre_length"]) == len(arrays["line_ledger"])
        and counts["admitted"] == len(arrays["detector_segments"]) == len(arrays["admitted_candidate_ids"]), "line_count_binding")
    for name, cap in (("edgels", FROZEN_CONFIG.max_edgels), ("graph_edges", FROZEN_CONFIG.max_edges),
        ("paths", FROZEN_CONFIG.max_groups), ("neighbor_probes", FROZEN_CONFIG.max_neighbor_probes),
        ("fit_nodes", FROZEN_CONFIG.max_fit_nodes), ("fit_points", FROZEN_CONFIG.max_fit_points),
        ("pre_length", FROZEN_CONFIG.max_candidates), ("projection_samples", FROZEN_CONFIG.max_projection_samples)):
        if counts[name] > cap:
            raise AcquisitionCapError("candidate_replay_cap:" + name)
    ids = arrays["admitted_candidate_ids"]
    require(len(set(int(v) for v in ids)) == len(ids) and all(int(v) < counts["pre_length"] for v in ids), "admitted_ids")
    for index, candidate_id in enumerate(ids):
        require(arrays["line_ledger"][candidate_id]["reason"] == 0
            and arrays["candidate_pre_length"][candidate_id].tobytes() == arrays["detector_segments"][index].tobytes(), "admitted_coordinates")
    reasons = [0]*6; sample_offset = 0
    for line in arrays["line_ledger"]:
        reason = int(line["reason"])
        require(reason < 6 and int(line["count"]) >= 2
            and int(line["offset"]) + int(line["count"]) <= len(arrays["support_membership"])
            and int(line["node"]) < counts["fit_nodes"] and math.isfinite(float(line["residual"]))
            and int(line["samples"]) >= 2, "line_support_contract")
        reasons[reason] += 1
        # Lines are ranked before sampling, so row offsets need not be sorted.
        sample_offset += int(line["samples"])
    require(reasons == [counts[k] for k in ("admitted", "duplicate", "frame", "original", "band", "short")]
        and sample_offset == counts["projection_samples"]
        and len(arrays["projection_occupancy"]) == (sample_offset+3)//4, "line_ledger_counts")
    require(all(int(v) < counts["edgels"] for v in arrays["support_membership"])
        and all(int(v) < counts["edgels"] for v in arrays["path_membership"]), "support_reference")
    budget.acquire(1024*len(arrays["line_ledger"]) + arrays["support_membership"].nbytes + 65536)
    ledger = arrays["line_ledger"]; coordinates = arrays["candidate_pre_length"]
    order = sorted(range(len(ledger)), key=lambda i: (-int(ledger[i]["count"]), -int(ledger[i]["score"]),
        float(ledger[i]["residual"]), tuple(float(v) for v in coordinates[i]), int(ledger[i]["node"])))
    signatures = set(); duplicate = set()
    for index in order:
        line = ledger[index]; offset = int(line["offset"]); count = int(line["count"])
        ids = arrays["support_membership"][offset:offset+count]
        require(all(int(ids[k]) < int(ids[k+1]) for k in range(len(ids)-1)), "support_sorted_unique")
        signature = ids.tobytes()
        if signature in signatures:
            duplicate.add(index)
        signatures.add(signature)
        require(int(line["score"]) == sum(int(arrays["edgels"][v]["m"]) for v in ids), "support_score_sum")
    position = 0
    for index in sorted(range(len(ledger)), key=lambda i: int(ledger[i]["sample_offset"])):
        line = ledger[index]
        require(int(line["sample_offset"]) == position, "projection_offset")
        categories, outside, length = stencil.sample_occupancy(coordinates[index], arrays["original_mask"],
            arrays["expanded_mask"], budget, FROZEN_CONFIG.max_projection_samples-position, endpoint_domain=True)
        try:
            require(len(categories) == int(line["samples"]), "projection_sample_count")
            for j, category in enumerate(categories):
                packed = int(arrays["projection_occupancy"][(position+j)//4])
                require((packed >> (2*((position+j)%4))) & 3 == int(category), "projection_occupancy_replay")
            reason = 1 if index in duplicate else (2 if outside or any(v == 0 for v in categories)
                else (3 if any(v == 1 for v in categories) else (4 if any(v == 2 for v in categories)
                else (5 if length < 40. else 0))))
            require(int(line["reason"]) == reason, "candidate_reason_replay")
            position += len(categories)
        finally:
            category_bytes = categories.nbytes + 256
            categories = None
            budget.release(category_bytes)
    require(position == counts["projection_samples"], "projection_total")
    for edge in arrays["edgels"]:
        x, y = int(edge["x"]), int(edge["y"])
        require(2 <= x < shape[1]-2 and 2 <= y < shape[0]-2
            and not arrays["expanded_mask"][y-2:y+3, x-2:x+3].any(), "edgel_valid_stencil")


def validate_receipt(encoded, plan, row, arm, budget):
    """Strict persisted receipt replay; detailed score records stay streamed."""
    start = budget.live
    packed_start = budget.packed_live
    payload = arrays = replay = split = trace = stages = event = None
    original = band = expanded = detector = emitted = expected_band = seen = traced = record = seeds = None
    try:
        support._validate_valid_member_blob(encoded, budget)
        budget.acquire(support._json_allocation_bound(encoded))
        payload = json.loads(encoded)
        required = set(_bindings(plan, row, arm)) | {"schema", "config_sha256", "trace_result", "coordinate_transform",
            "control_pre_length", "arrays", "split_counts", "acquisition_counts"}
        require(set(payload) == required and payload["schema"] == "VALID-NATIVE-CHAIN-01-member-v1", "member_schema")
        require(all(payload[k] == v for k, v in _bindings(plan, row, arm).items())
            and payload["config_sha256"] == plan["config_sha256"], "member_binding")
        require(payload["coordinate_transform"] == {"scale": 1., "offset": [0., 0.], "units": "native_center_pixels"}
            and payload["control_pre_length"] == "unobservable", "member_transform")
        arrays = {k: _array(v, budget) for k, v in payload["arrays"].items()}
        original, band, expanded = (arrays[k] for k in ("original_mask", "boundary_band", "expanded_mask"))
        budget.acquire(32*original.size + 16384)
        seeds, expected_band = support.mask_transition_band(original, max_mask_pixels=FROZEN_CONFIG.max_pixels)
        require(np.array_equal(band, expected_band) and np.array_equal(expanded, original | band), "mask_derivation")
        require(sha(original.tobytes()) == row["mask_sha256"] and sha(expanded.tobytes()) == row["expanded_mask_sha256"], "mask_binding")
        detector, emitted = arrays["detector_segments"], arrays["emitted_segments"]
        samples = sum(max(2, math.ceil(math.hypot(float(r[2])-float(r[0]), float(r[3])-float(r[1]))/.5)+1)
            for r in detector)
        if samples > FROZEN_CONFIG.max_projection_samples:
            raise AcquisitionCapError("replay_sample_cap")
        budget.acquire(128*min(samples, support.MAX_SAMPLES_PER_SEGMENT)+2048*(samples//80+len(detector))+65536)
        split = support.split_at_excluded_mask(detector, expanded)
        require(split.segments.dtype == emitted.dtype and split.segments.shape == emitted.shape
            and split.segments.tobytes() == emitted.tobytes(), "split_replay")
        replay = support.audit_valid_support(detector, original, expanded, split, budget=budget)
        require(replay["counts"] == payload["split_counts"], "ledger_counts")
        require(all(replay[k].dtype == arrays[name].dtype and replay[k].shape == arrays[name].shape
            and replay[k].tobytes() == arrays[name].tobytes() for k, name in
            (("runs", "run_ledger"), ("occupancy", "sample_occupancy"))), "ledger_replay")
        require((arm == "baseline") == (payload["acquisition_counts"] is None), "acquisition_arm")
        if arm == "candidate":
            _candidate_replay(arrays, payload["acquisition_counts"], row["rgb_shape"], budget)
        else:
            require(set(arrays) == {"original_mask", "boundary_band", "expanded_mask", "detector_segments",
                "emitted_segments", "run_ledger", "sample_occupancy"}, "baseline_array_schema")
        traced = payload["trace_result"]
        require(set(traced) == {"status", "reason", "prediction", "trace", "seconds", "rgb_sha256", "rgb_shape"}
            and traced["rgb_sha256"] == row["rgb_sha256"] and traced["rgb_shape"] == row["rgb_shape"], "trace_binding")
        require(traced["status"] in ("returned", "ordinary_refusal") and isinstance(traced["reason"], str)
            and type(traced["seconds"]) in (int, float) and math.isfinite(traced["seconds"])
            and traced["seconds"] >= 0, "trace_status")
        trace = traced["trace"]
        require(set(trace) == {"scores", "stages", "arrays", "trace_complete", "score_events", "array_bytes",
            "trace_event_bytes", "trace_seconds", "score_delegate_seconds"} and trace["trace_complete"] is True, "trace_complete_schema")
        require(type(trace["score_events"]) is int and 0 <= trace["score_events"] <= 120000, "trace_count")
        require(all(type(trace[k]) in (int, float) and math.isfinite(trace[k]) and trace[k] >= 0
            for k in ("trace_seconds", "score_delegate_seconds")), "trace_timing")
        stages = trace["stages"]
        require(isinstance(stages, list) and 3 <= len(stages) <= 32, "stage_count")
        seen = {}
        for event in stages:
            require(isinstance(event, dict) and set(event) == {"stage", "input_sha256", "output", "delegate_seconds"}, "stage_schema")
            name = event["stage"]
            require(name in ("create_gui_mask", "enhance", "detect_segments", "estimate_camera", "refine_optics")
                and name not in seen and type(event["delegate_seconds"]) in (int, float)
                and math.isfinite(event["delegate_seconds"]) and event["delegate_seconds"] >= 0, "stage_contract")
            if event["input_sha256"] is not None:
                require(event["input_sha256"] in trace["arrays"], "stage_input_reference")
            seen[name] = event
        require({"create_gui_mask", "enhance", "detect_segments"} <= seen.keys(), "missing_stage")
        require([v["stage"] for v in stages[:3]] == ["create_gui_mask", "enhance", "detect_segments"]
            and ("estimate_camera" not in seen or stages[-1]["stage"] == "estimate_camera"), "stage_order")
        for stage, wanted in (("create_gui_mask", original), ("detect_segments", emitted)):
            meta = trace["arrays"].get(seen[stage]["output"])
            require(meta is not None and meta["dtype"] == wanted.dtype.str and meta["shape"] == list(wanted.shape)
                and meta["bytes_hex"] == wanted.tobytes().hex(), "stage_output_binding")
        enhanced = trace["arrays"].get(seen["enhance"]["output"])
        require(enhanced is not None and enhanced["dtype"] == np.dtype(np.uint8).str
            and enhanced["shape"] == row["rgb_shape"][:2], "enhance_output")
        require(seen["create_gui_mask"]["input_sha256"] is None
            and seen["detect_segments"]["input_sha256"] == seen["enhance"]["output"], "stage_input_binding")
        if traced["status"] == "returned":
            require(traced["reason"] == "" and trace["score_events"] > 0 and "estimate_camera" in seen, "return_receipt")
            _prediction(traced["prediction"], row["rgb_shape"], len(emitted))
            require(seen["estimate_camera"]["output"] == traced["prediction"], "solver_prediction_binding")
        else:
            require(traced["prediction"] is None and traced["reason"] == "image_calibration_failed"
                and ("estimate_camera" not in seen or seen["estimate_camera"]["output"] is None), "refusal_consistency")
        if len(emitted) < 12:
            require(traced["status"] == "ordinary_refusal" and trace["score_events"] == 0
                and "estimate_camera" not in seen and "refine_optics" not in seen, "minimum_segment_gate")
        event_bytes = sum(len(canonical(e).encode()) for e in stages)
        sites = {v["tag"]: [v["start"], v["end"]] for v in plan["score_sites"]}
        validated_pairs = set()
        def record(index, event):
            nonlocal event_bytes
            require(set(event) == {"site", "span", "segments_sha256", "weights_sha256", "camera", "winner", "score"}, "score_schema")
            require(event["site"] in sites and event["span"] == sites[event["site"]], "score_site")
            require(type(event["score"]) in (int, float) and math.isfinite(event["score"]), "score_finite")
            seg = trace["arrays"].get(event["segments_sha256"]); weight = trace["arrays"].get(event["weights_sha256"])
            require(seg is not None and weight is not None and len(seg["shape"]) == 2 and seg["shape"][1] == 4
                and weight["shape"] == [seg["shape"][0]] and np.dtype(seg["dtype"]).kind == "f"
                and np.dtype(weight["dtype"]).kind == "f", "score_array_reference")
            pair = (event["segments_sha256"], event["weights_sha256"], event["site"] == "coarse")
            if pair not in validated_pairs:
                scratch = budget.acquire(16*(len(seg["bytes_hex"])+len(weight["bytes_hex"]))+65536)
                score_segments = score_weights = expected_segments = expected_weights = None
                try:
                    score_segments = np.frombuffer(bytes.fromhex(seg["bytes_hex"]), dtype=np.dtype(seg["dtype"])).reshape(seg["shape"])
                    score_weights = np.frombuffer(bytes.fromhex(weight["bytes_hex"]), dtype=np.dtype(weight["dtype"])).reshape(weight["shape"])
                    expected_segments = emitted
                    expected_weights = np.hypot(emitted[:, 2]-emitted[:, 0], emitted[:, 3]-emitted[:, 1])
                    if pair[2]:
                        order = np.argsort(-expected_weights)[:300]
                        expected_segments = emitted[order]
                        expected_weights = expected_weights[order]
                    require(score_segments.dtype == expected_segments.dtype and score_segments.shape == expected_segments.shape
                        and score_segments.tobytes() == expected_segments.tobytes()
                        and score_weights.dtype == expected_weights.dtype and score_weights.tobytes() == expected_weights.tobytes(),
                        "score_solver_input_binding")
                finally:
                    score_segments = score_weights = expected_segments = expected_weights = order = None
                    budget.release(scratch)
                budget.acquire(512)
                validated_pairs.add(pair)
            camera = event["camera"]
            require(isinstance(camera, dict) and set(camera) == {"focal", "yaw", "pitch", "width", "height", "threshold", "roll", "principal_point"}
                and _finite(camera) and all(type(camera[k]) in (int, float) for k in
                ("focal", "yaw", "pitch", "width", "height", "threshold", "roll"))
                and camera["focal"] > 0 and camera["threshold"] == 3.
                and camera["width"] == row["rgb_shape"][1] and camera["height"] == row["rgb_shape"][0], "score_camera")
            point = camera["principal_point"]
            require(point is None or isinstance(point, list) and len(point) == 2
                and all(type(v) in (int, float) for v in point), "score_principal")
            winner = event["winner"]
            require(isinstance(winner, dict) and _finite(winner), "score_winner")
            if event["site"] in ("coarse", "coarse-full-rescore", "fine"):
                require(set(winner) == {"best_before"} and (winner["best_before"] is None
                    or isinstance(winner["best_before"], list) and len(winner["best_before"]) == 4
                    and all(type(v) in (int, float) for v in winner["best_before"])), "score_best")
            else:
                require(set(winner) == {"winner_before"} and isinstance(winner["winner_before"], dict)
                    and set(winner["winner_before"]) <= {"value", "focal", "yaw", "pitch", "roll", "point"}, "score_refine_winner")
            event_bytes += len(canonical(event).encode())
        support.validate_score_stream(trace["scores"], budget, expected_records=trace["score_events"], on_record=record)
        require(type(trace["trace_event_bytes"]) is int and trace["trace_event_bytes"] == event_bytes <= 64000000, "trace_byte_count")
        # Copy the small prediction out of the parser arena before releasing it.
        prediction = None if traced["prediction"] is None else json.loads(canonical(traced["prediction"]))
        return {"identity": row["identity"], "arm": arm, "status": traced["status"], "prediction": prediction,
            "mask_sha256": row["mask_sha256"], "expanded_mask_sha256": row["expanded_mask_sha256"], "score_events": trace["score_events"]}
    except AcquisitionTerminalError:
        raise
    except Exception as exc:
        raise AcquisitionIntegrityError("receipt_decode", {"exception_type": type(exc).__name__}) from exc
    finally:
        payload = arrays = replay = split = trace = stages = event = None
        original = band = expanded = detector = emitted = expected_band = seen = traced = record = seeds = None
        meta = enhanced = wanted = validated_pairs = None
        budget.packed_live = packed_start
        budget.release(budget.live - start)


def _persist(path, plan, sidecar, *, budget=None):
    supplied = budget is not None
    budget = budget or AllocationBudget(FROZEN_CONFIG.max_live_bytes)
    if not supplied:
        budget.acquire(support._managed_tree_size(plan, ceiling=8*1024*1024) + 65536)
    size = support._exact_json_size(plan) + 1
    lease = budget.acquire(3*size + 65536)
    encoded = None
    try:
        encoded = (canonical(plan) + "\n").encode()
        if len(encoded) + (sidecar.stat().st_size if sidecar.exists() else 0) + support.STUDY_RESERVE_BYTES > support.MAX_STUDY_TOTAL_BYTES:
            raise AcquisitionCapError("aggregate_cap")
        temporary = path.with_name(path.name + ".writing")
        with temporary.open("xb") as stream:
            require(stream.write(encoded) == len(encoded), "short_plan_write")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        encoded = None
        budget.release(lease)


def _driver_recorder(base, retained):
    """Charge retained driver inputs to the adapter's actual active budget.

    GuardedRecorder installs its LosslessScoreStream before any image stage.
    This narrow data observer also accounts for the plan and shared mask owners
    throughout acquisition, encoding, append and per-arm plan persistence.
    """
    class DriverRecorder(base):
        def __setattr__(self, name, value):
            if name == "scores" and isinstance(value, support.LosslessScoreStream):
                value.budget.acquire(retained, "driver_retained_input_cap")
            super().__setattr__(name, value)
    return DriverRecorder


TRACE_DEFINITIONS = ('CameraStudyError', '_require', '_json', '_sha', 'TracePolicy',
    'score_call_sites', 'classify_score_site', '_clean', 'winner_snapshot', 'TraceRecorder',
    '_install', '_call_once', 'trace_rgb')


def execution_contract():
    return {"schema": "CAM-VALID-execution-contract-v1", "truth_schema": scoring.TRUTH_SCHEMA,
        "truth_max_bytes": 1024*1024, "attestation_reserve_bytes": ATTESTATION_RESERVE, "loader": "scripts.evaluation.valid_camera_scoring:load_truth",
        "evaluator": "scripts.evaluation.valid_camera_scoring:evaluate_prediction",
        "trace": "scripts.evaluation.valid_camera_trace:trace_rgb",
        "copied_trace_definitions": list(TRACE_DEFINITIONS)}


def _code_signature(code):
    return (code.co_code, tuple(_code_signature(c) if isinstance(c, CodeType) else c
        for c in code.co_consts), code.co_names, code.co_varnames, code.co_freevars,
        code.co_cellvars, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_flags)


def _attest_module(module, relative, names):
    """Independently compile approved source without executing or trusting helpers."""
    path = ROOT/relative
    canonical_name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
    require(module.__name__ == canonical_name and sys.modules.get(canonical_name) is module
        and module.__spec__ is not None and module.__spec__.name == canonical_name
        and Path(module.__spec__.origin).resolve() == path
        and Path(module.__file__).resolve() == path, "dependency_path")
    with path.open("rb") as stream:
        source_bytes = stream.read(128*1024+1)
    require(len(source_bytes) <= 128*1024, "source_metadata_cap")
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source)
    compiled = compile(source, str(path), "exec", dont_inherit=True)
    bindings = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)}
    def default_value(node):
        if isinstance(node, ast.Name) and node.id in bindings:
            return ast.literal_eval(bindings[node.id])
        return ast.literal_eval(node)
    def function_node(name):
        body = tree.body
        node = None
        for part in name.split('.'):
            matches = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
            require(len(matches) == 1, "dependency_source_definition")
            node = matches[0]; body = node.body
        return node
    def lookup(code, name):
        for part in name.split('.'):
            matches = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == part]
            if len(matches) != 1:
                return None
            code = matches[0]
        return code
    for name in names:
        value = module
        for part in name.split('.'):
            value = getattr(value, part, None)
        expected = lookup(compiled, name)
        require(inspect.isfunction(value) and value.__globals__ is module.__dict__
            and Path(value.__code__.co_filename).resolve() == path and expected is not None
            and _code_signature(value.__code__) == _code_signature(expected), "dependency_function_identity")
        # Two exact trace defaults are verified explicitly below: the immutable
        # TracePolicy instance and the source-derived PINNED_SITES tuple.
        if name not in ('trace_rgb', 'classify_score_site'):
            node = function_node(name)
            positional = tuple(default_value(n) for n in node.args.defaults) or None
            keywords = {arg.arg: default_value(n) for arg, n in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if n is not None} or None
            require(value.__defaults__ == positional and value.__kwdefaults__ == keywords, "dependency_default_identity")


def validate_dependencies(dependencies):
    """Attest actual safe trace origin, copied ASTs and independent solver source."""
    require(isinstance(dependencies, Dependencies), "driver_dependencies")
    module = dependencies.trace_module
    names = tuple(n for n in TRACE_DEFINITIONS if n not in ('CameraStudyError', 'TracePolicy', 'TraceRecorder'))
    _attest_module(module, "scripts/evaluation/valid_camera_trace.py", (*names,
        'CameraStudyError.__init__', 'TraceRecorder.__init__', 'TraceRecorder.add', 'TraceRecorder.array',
        'TraceRecorder.payload', '_verify_preregistration', 'bind_trace_contract', 'clear_trace_contract',
        '_runtime_pins', 'attest_module', '_code_signature', '_file_sha256', '_source_text'))
    _attest_module(dependencies.calibration, "src/mc_photogrammetry/reconstruction/calibration.py",
        ('_score', 'enhance', 'detect_segments', 'estimate_camera', 'refine_optics', 'estimate_image_calibrations'))
    _attest_module(dependencies.gui_mask, "src/mc_photogrammetry/vision/gui_mask.py", ('create_gui_mask',))
    require(dependencies.trace_rgb is module.trace_rgb, "trace_function_binding")
    def definitions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in TRACE_DEFINITIONS}
    require(definitions(ROOT/"scripts/evaluation/run_camera_mixed_depth.py") ==
        definitions(ROOT/"scripts/evaluation/valid_camera_trace.py"), "trace_copied_ast_pin")
    require(module.calibration is dependencies.calibration and module.gui_mask is dependencies.gui_mask,
        "dependency_module_pairing")
    require(module._PINNED_CAMERA_CODE is dependencies.calibration.estimate_camera.__code__
        and module._PINNED_REFINE_CODE is dependencies.calibration.refine_optics.__code__
        and module._SCORE_SIGNATURE == inspect.signature(dependencies.calibration._score), "solver_code_binding")
    require(asdict(module.TracePolicy()) == SOLVER["trace_policy"]
        and list(module.PINNED_SITES) == score_sites()
        and module.classify_score_site.__defaults__ == (module.PINNED_SITES,)
        and module.trace_rgb.__kwdefaults__ == {"policy": module.TracePolicy()}, "trace_policy_pin")


def validate_scoring(truth_loader, evaluator):
    _attest_module(scoring, "scripts/evaluation/valid_camera_scoring.py",
        ('load_truth', 'evaluate_prediction', 'validate_truth', 'literal_rotation', 'camera_metrics', '_require', '_canonical'))
    require(truth_loader is scoring.load_truth and evaluator is scoring.evaluate_prediction, "scoring_callback_identity")
    require(scoring.GATES == POLICY["camera_gates"] and scoring.TRUTH_MAX_BYTES == 1024*1024
        and scoring.TRUTH_SCHEMA == "CAM-VALID-camera-truth-v1", "scoring_constant_pin")


def _trace_contract(plan, expected_digest):
    return {"schema": "CAM-VALID-trace-contract-v1", "external_preregistration_sha256": expected_digest,
        "source_sha256": plan["source_sha256"], "runtime": plan["runtime"],
        "score_sites": plan["score_sites"], "policy": plan["solver"]["trace_policy"]}


def _barrier(plan, store):
    expected = [(row, arm) for row in plan["registry"] for arm in ARMS]
    require(len(plan["sealed_predictions"]) == 24, "seal_count")
    offset = 0; receipts = []; digests = []
    for index, ((row, arm), descriptor) in enumerate(zip(expected, plan["sealed_predictions"])):
        require(set(descriptor) == {"identity", "arm", "member_index", "offset", "compressed_bytes", "decoded_bytes",
            "payload_sha256", "compressed_sha256"}, "descriptor_schema")
        require(descriptor["identity"] == row["identity"] and descriptor["arm"] == arm
            and type(descriptor["member_index"]) is int and descriptor["member_index"] == index
            and type(descriptor["offset"]) is int and descriptor["offset"] == offset, "seal_order")
        require(all(type(descriptor[k]) is int for k in ("compressed_bytes", "decoded_bytes"))
            and _digest(descriptor["payload_sha256"]) and _digest(descriptor["compressed_sha256"]), "descriptor_types")
        encoded = store.read_member(descriptor)  # Independent of append's active budget.
        budget = AllocationBudget(FROZEN_CONFIG.max_live_bytes)
        budget.acquire(len(encoded) + support._managed_tree_size(plan, ceiling=8*1024*1024) + 256*1024)
        receipts.append(validate_receipt(encoded, plan, row, arm, budget))
        offset += descriptor["compressed_bytes"]; digests.append(descriptor["payload_sha256"])
        del encoded, budget
    require(store.sidecar_path.stat().st_size == offset, "sidecar_tail")
    return receipts, {"validated": True, "prediction_count": 24, "sidecar_bytes": offset,
        "ordered_payload_sha256": sha(canonical(digests))}


def _summary(rows, evaluations):
    before = [v for v in evaluations if v["arm"] == "baseline"]
    after = [v for v in evaluations if v["arm"] == "candidate"]
    added = sum(not a["passed"] and b["passed"] for a, b in zip(before, after))
    lost = sum(a["passed"] and not b["passed"] for a, b in zip(before, after))
    stress = [(a, b) for row, a, b in zip(rows, before, after) if row["stratum"] == "stress"]
    wrong_before = sum(a["status"] == "returned" and not a["passed"] for a, _ in stress)
    wrong_after = sum(b["status"] == "returned" and not b["passed"] for _, b in stress)
    newly_wrong = sum(b["status"] == "returned" and not b["passed"]
        and not (a["status"] == "returned" and not a["passed"]) for a, b in stress)
    gates = {"candidate_passes": sum(v["passed"] for v in after) >= 9,
        "net_added": added - lost >= 3, "no_lost": lost == 0,
        "no_newly_wrong_stress": newly_wrong == 0, "stress_wrong": wrong_after <= wrong_before}
    return {"accepted": all(gates.values()), "gates": gates, "added": added, "lost": lost,
        "family_outcomes": [{"family": row["family"], "identity": row["identity"], "stratum": row["stratum"],
            "baseline": a["status"] if not a["passed"] else "pass",
            "candidate": b["status"] if not b["passed"] else "pass"}
            for row, a, b in zip(rows, before, after)]}


def run_plan(plan_path, *, expected_preregistration_sha256, dependencies, truth_loader, evaluator):
    """Consume one supplied plan; requires separate experiment GO.

    The mandatory expected digest comes from the caller's external approval
    record, never from this plan. Both embedded and recomputed digests must match
    it before the durable one-shot claim is created.
    Image adapters receive no registry/truth callback. Truth loader returns
    bounded canonical JSON bytes whose hash matches the supplied reference.
    All truth receipts validate after the barrier and before the evaluator.
    Evaluator consumes finite predictions, decoded truth and pinned camera gates.
    """
    require(_RUN_LOCK.acquire(blocking=False), "runner_concurrency")
    plan = paths = None; claimed = False; truth_accessed = False
    active_case = active_arm = None
    contract_bound = False
    try:
        require(_digest(expected_preregistration_sha256), "expected_preregistration_digest_syntax")
        path = Path(plan_path).resolve()
        if path.stat().st_size > 8*1024*1024:
            raise AcquisitionCapError("plan_byte_cap")
        raw = path.read_bytes()
        initial_budget = AllocationBudget(FROZEN_CONFIG.max_live_bytes)
        initial_budget.acquire(len(raw) + support._json_allocation_bound(raw) + 16*1024*1024)
        plan = json.loads(raw)
        require(raw == (canonical(plan)+"\n").encode(), "plan_not_canonical")
        paths = validate_plan(plan, path)
        require(expected_preregistration_sha256 == plan["preregistration_sha256"]
            == preregistration_digest(plan), "external_preregistration_anchor")
        validate_scoring(truth_loader, evaluator)
        validate_dependencies(dependencies)
        trace_contract = _trace_contract(plan, expected_preregistration_sha256)
        dependencies.trace_module.bind_trace_contract(trace_contract, expected_contract_sha256=sha(canonical(trace_contract)))
        contract_bound = True
        del trace_contract
        claim = {"study": STUDY, "preregistration_sha256": plan["preregistration_sha256"], "plan_sha256": sha(raw),
            "expected_preregistration_sha256": expected_preregistration_sha256,
            "paths": plan["paths"], "retry_performed": False}
        with paths["claim"].open("xb") as stream:
            claimed = True
            blob = (canonical(claim)+"\n").encode()
            require(stream.write(blob) == len(blob), "claim_write")
            stream.flush()
            os.fsync(stream.fileno())
        plan["execution_state"] = "RUNNING"
        _persist(path, plan, paths["sidecar"])
        del raw, blob, claim, initial_budget
        store = support.ValidAcquisitionMemberStore(paths["sidecar"])
        for row in plan["registry"]:
            active_case = row["identity"]
            rgb = _rgb(row)
            pair_budget = AllocationBudget(FROZEN_CONFIG.max_live_bytes)
            pair_budget.acquire(support._managed_tree_size(plan, ceiling=8*1024*1024)
                + rgb.nbytes + 32*math.prod(rgb.shape[:2]) + 65536)
            original = dependencies.gui_mask.create_gui_mask(*rgb.shape[:2])
            require(isinstance(original, np.ndarray) and original.dtype == np.bool_ and original.shape == rgb.shape[:2]
                and original.flags.c_contiguous, "pair_mask_contract")
            original = original.copy(); original.setflags(write=False)
            _, band = support.mask_transition_band(original, max_mask_pixels=FROZEN_CONFIG.max_pixels)
            require(sha(original.tobytes()) == row["mask_sha256"] and sha((original | band).tobytes()) == row["expanded_mask_sha256"], "pair_mask_pin")
            del pair_budget
            mask_delegate = dependencies.gui_mask.create_gui_mask
            recorder_delegate = dependencies.trace_module.TraceRecorder
            retained = support._managed_tree_size(plan, ceiling=8*1024*1024) + 6*original.size + ATTESTATION_RESERVE
            # Shared namespace is the one observed by the source-pinned harness.
            dependencies.gui_mask.create_gui_mask = lambda h, w: original.copy()
            dependencies.trace_module.TraceRecorder = _driver_recorder(recorder_delegate, retained)
            try:
                for arm in ARMS:
                    active_arm = arm
                    require(source_pins() == plan["source_sha256"] and runtime_pins() == plan["runtime"], "active_pin_drift")
                    traced, captured = support.trace_valid_acquisition(rgb, row["identity"], candidate=arm == "candidate",
                        trace_rgb=dependencies.trace_rgb, trace_module=dependencies.trace_module,
                        calibration_module=dependencies.calibration, gui_mask_module=dependencies.gui_mask)
                    encoded = support.valid_acquisition_payload(_bindings(plan, row, arm), traced, captured)
                    encoded = support.valid_sealed_bytes(encoded)
                    receipt = validate_receipt(encoded, plan, row, arm, captured["budget"])
                    descriptor = store.append(encoded, plan, row["identity"], arm, budget=captured["budget"])
                    plan["sealed_predictions"].append(descriptor)
                    _persist(path, plan, paths["sidecar"], budget=captured["budget"])
                    del encoded, captured, traced, receipt
            finally:
                dependencies.gui_mask.create_gui_mask = mask_delegate
                dependencies.trace_module.TraceRecorder = recorder_delegate
            del rgb, original, band
        plan["execution_state"] = "PREDICTIONS_SEALED"
        _persist(path, plan, paths["sidecar"])
        require(source_pins() == plan["source_sha256"] and runtime_pins() == plan["runtime"], "barrier_pin_drift")
        receipts, validation = _barrier(plan, store)
        plan["seal_validation"] = validation
        plan["execution_state"] = "PREDICTIONS_VALIDATED"
        _persist(path, plan, paths["sidecar"])
        truth_budget = AllocationBudget(FROZEN_CONFIG.max_live_bytes)
        truth_budget.acquire(support._managed_tree_size(plan, ceiling=8*1024*1024) + ATTESTATION_RESERVE)
        validate_scoring(truth_loader, evaluator)
        truths = []
        for row, reference in zip(plan["registry"], plan["truth_registry"]):
            active_case = reference["identity"]; active_arm = "truth_loading"
            truth_accessed = True
            truth_raw = truth_loader(dict(reference))
            if not isinstance(truth_raw, bytes) or len(truth_raw) > 1024*1024:
                raise AcquisitionCapError("truth_receipt_byte_cap")
            truth_budget.acquire(len(truth_raw) + support._json_allocation_bound(truth_raw))
            require(sha(truth_raw) == reference["sha256"], "truth_reference_hash")
            truth = json.loads(truth_raw)
            require(_finite(truth) and canonical(truth).encode() == truth_raw, "truth_receipt_canonical_finite")
            truths.append(scoring.validate_truth(truth, row))
            del truth_raw, truth
        evaluations = []
        for index, (row, truth) in enumerate(zip(plan["registry"], truths)):
            active_case = row["identity"]; active_arm = "truth_scoring"
            for receipt in receipts[2*index:2*index+2]:
                metrics = None; passed = False
                if receipt["status"] == "returned":
                    validate_scoring(truth_loader, evaluator)
                    copy_lease = 4*(support._managed_tree_size(truth, ceiling=16*1024*1024)
                        + support._managed_tree_size(receipt["prediction"], ceiling=65536)) + 65536
                    truth_budget.acquire(copy_lease, "scoring_copy_cap")
                    prediction_copy = truth_copy = None
                    try:
                        prediction_copy = json.loads(canonical(receipt["prediction"]))
                        truth_copy = json.loads(canonical(truth))
                        metrics = evaluator(prediction_copy, truth_copy, json.loads(canonical(POLICY["camera_gates"])))
                        require(prediction_copy == receipt["prediction"] and truth_copy == truth, "scoring_input_mutation")
                    finally:
                        prediction_copy = truth_copy = None
                        truth_budget.release(copy_lease)
                    require(isinstance(metrics, dict) and _finite(metrics) and type(metrics.get("diagnostic_pass")) is bool, "evaluator_receipt")
                    gates = POLICY["camera_gates"]
                    require(all(type(metrics.get(k)) in (int, float) and metrics[k] >= 0 for k in gates), "camera_metric_schema")
                    require(metrics["diagnostic_pass"] == all(metrics[k] <= limit for k, limit in gates.items()), "camera_gate_receipt")
                    truth_budget.acquire(support._managed_tree_size(metrics, ceiling=65536)+2048)
                    passed = metrics["diagnostic_pass"]
                evaluations.append({"identity": row["identity"], "arm": receipt["arm"],
                    "status": receipt["status"], "passed": passed, "metrics": metrics})
            del truth
        del truths, truth_budget
        plan["evaluations"] = evaluations
        plan["summary"] = _summary(plan["registry"], evaluations)
        plan["execution_state"] = "COMPLETE"
        _persist(path, plan, paths["sidecar"])
        return plan
    except Exception as exc:
        if claimed:
            plan["execution_state"] = "FAILED"
            plan["evaluations"] = plan["summary"] = None
            plan["failure"] = {"reason": str(exc)[:512], "terminal_kind": type(exc).__name__,
                "case": active_case, "stage": active_arm, "sealed_prediction_count": len(plan["sealed_predictions"]),
                "post_hoc_truth_accessed": truth_accessed, "retry_performed": False}
            _persist(paths["plan"], plan, paths["sidecar"])
        if isinstance(exc, AcquisitionTerminalError):
            raise
        raise AcquisitionIntegrityError("runner_failure", {"reason": str(exc)[:240]}) from exc
    finally:
        try:
            if contract_bound:
                dependencies.trace_module.clear_trace_contract()
        finally:
            _RUN_LOCK.release()
