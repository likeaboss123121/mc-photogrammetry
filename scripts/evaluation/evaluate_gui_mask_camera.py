"""One-shot camera-only GUI-mask measurement; execution needs separate authorization.

The exact production adapter runs once on six ordered ordinary views. Evaluation
truth is parsed only after exclusive scalar prediction publication, close, and
size/SHA readback. No camera selection or fitting uses evaluation metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix

MANIFEST_PATH = Path("agent swarm instantiation template/runs/"
                     "SOL-G7-UNIT-WEIGHT-FULL-DELIVERY-01.packet-prep01.input-manifest.json")
MANIFEST_SHA = "90F1AF81157225066E4494047BA67CEF0C7CC646A36E4427FED782810E2BB12D"
SCENE = Path("exports/datasets/pilot-batch-train-station-2026-09-12/scenes/"
             "train-station-spawn-208-96-orbit-a")
CAPTURE_PATH = SCENE / "capture.json"
CAPTURE_BYTES = 21_027
CAPTURE_SHA = "36B1CB23F0666ADC406BE816D41F226E8AE2347F8E32F999C5052DCD64FA7219"
BASELINE_PATH = Path("docs/reports/spatial-sift-matcher-ab-05-2026-10-11/pair-ab.scalar.json")
BASELINE_BYTES = 218_772
BASELINE_SHA = "41F8673CB30F155374D36F586844B78D330FA297F4A45292096D4F2EACF84399"
IDS = tuple(f"image-{i:04d}" for i in range(6))
MAX_JSON_BYTES = 1_048_576
MAX_IMAGE_BYTES = 16_777_216
MAX_PIXELS = 6_000_000
SCHEMA = "mc-gui-mask-camera-predictions-v1"
RULE = {"producer": "pipeline.image_only.solve_image_cameras", "adapter_calls": 1,
        "gui_mask": "production_default_zero_fill", "selection": "production_only",
        "rotation_gauge": "min_world_Y_quarter_turn_SO3", "rotation_bound_degrees": 5.,
        "relative_vertical_focal_bound": .20, "primary_frame": "image-0003"}


def number(value):
    """Detach finite native/NumPy numeric scalars without accepting booleans."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError("finite numeric scalar required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("finite numeric scalar required")
    return result


def dimensions(values):
    if (type(values) not in (list, tuple) or len(values) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or v <= 0 for v in values)):
        raise ValueError("camera dimensions")
    result = [int(v) for v in values]
    if result[0] * result[1] > MAX_PIXELS:
        raise ValueError("native pixel cap")
    return result


def pair(values):
    if type(values) not in (list, tuple) or len(values) != 2:
        raise ValueError("numeric pair required")
    return [number(v) for v in values]


def digest(raw):
    return hashlib.sha256(raw).hexdigest().upper()


def closure_pin(value):
    """Validate a source/runtime closure digest verified by the launch owner.

    Closure construction and verification belong to the separately reviewed
    execution packet. This harness binds their supplied identities to the
    persisted same-pass predictions; it does not construct a closure.
    """
    if (type(value) is not str or len(value) != 64
            or any(ch not in "0123456789abcdefABCDEF" for ch in value)):
        raise ValueError("verified source/runtime closure SHA256 required")
    return value.upper()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def bounded_bytes(path, cap):
    path = Path(path)
    if path.stat().st_size > cap:
        raise ValueError("snapshot byte cap")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("snapshot byte cap")
    return raw


def pinned_bytes(path, size, sha, cap=MAX_JSON_BYTES):
    raw = bounded_bytes(path, cap)
    if (size is not None and len(raw) != size) or digest(raw) != sha:
        raise ValueError("pinned byte size/SHA mismatch")
    return raw


def validate_seals(seals):
    if type(seals) is not list or len(seals) != 6:
        raise ValueError("six input seals required")
    paths = set()
    for index, seal in enumerate(seals):
        if (type(seal) is not dict or set(seal) != {"frame", "ordinal", "path", "bytes", "sha256"}
                or seal["frame"] != IDS[index] or type(seal["ordinal"]) is not int
                or seal["ordinal"] != index or type(seal["path"]) is not str
                or not seal["path"] or seal["path"] in paths
                or Path(seal["path"]).is_absolute() or ".." in Path(seal["path"]).parts
                or type(seal["bytes"]) is not int or not 0 < seal["bytes"] <= MAX_IMAGE_BYTES
                or type(seal["sha256"]) is not str or len(seal["sha256"]) != 64
                or any(ch not in "0123456789ABCDEF" for ch in seal["sha256"])):
            raise ValueError("input seal identity/order/size/SHA")
        paths.add(seal["path"])


def sealed_input_preflight(manifest_path, image_paths):
    """Check the full inventory and every byte seal before decoding any view."""
    raw = pinned_bytes(manifest_path, None, MANIFEST_SHA)
    manifest = json.loads(raw)
    rows = manifest.get("inputs")
    if type(rows) is not list or len(rows) != 6 or len(image_paths) != 6:
        raise ValueError("six ordered input frames required")
    seals = []
    for index, (row, provided) in enumerate(zip(rows, image_paths, strict=True)):
        if type(row) is not dict:
            raise ValueError("input inventory row")
        seal = {k: row.get(k) for k in ("frame", "ordinal", "path", "bytes", "sha256")}
        seals.append(seal)
        # Manifest absolutes are historical inventory, not a replacement for
        # the repository-relative identity or caller-provided ordered mapping.
        expected = (ROOT / str(row.get("path"))).resolve()
        if (not expected.is_relative_to(ROOT.resolve())
                or Path(provided).resolve() != expected
                or Path(str(row.get("absolute_path"))).resolve() != expected):
            raise ValueError("accepted input path/order")
    validate_seals(seals)
    for seal, path in zip(seals, image_paths, strict=True):
        if Path(path).stat().st_size != seal["bytes"]:
            raise ValueError("accepted input size")
    for seal, path in zip(seals, image_paths, strict=True):
        pinned_bytes(path, seal["bytes"], seal["sha256"], MAX_IMAGE_BYTES)
    return seals


def camera_scalars(estimate, image_size):
    if estimate is None or estimate.pose is None or estimate.intrinsics is None:
        return None
    pose, optics = estimate.pose, estimate.intrinsics
    if pose.position is not None:
        raise ValueError("unexpected resolved camera translation")
    size = dimensions(optics.image_size)
    if size != image_size:
        raise ValueError("camera decoded dimensions mismatch")
    result = {"yaw_degrees": number(pose.yaw_degrees),
              "pitch_degrees": number(pose.pitch_degrees),
              "roll_degrees": number(pose.roll_degrees), "image_size": size,
              "focal_length_pixels": pair(optics.focal_length_pixels),
              "principal_point_pixels": pair(optics.principal_point_pixels),
              "translation": None, "scale": None}
    if min(result["focal_length_pixels"]) <= 0 or not -90 <= result["pitch_degrees"] <= 90:
        raise ValueError("camera focal/pitch range")
    return result


def predict_cameras(frames, seals, *, source_closure_sha256=None, runtime_closure_sha256=None):
    """One delegation to the exact production seam; never refit or fill abstentions."""
    source_closure_sha256 = closure_pin(source_closure_sha256)
    runtime_closure_sha256 = closure_pin(runtime_closure_sha256)
    validate_seals(seals)
    if tuple(frames) != IDS:
        raise ValueError("exact six ordered frame IDs required")
    sizes = {}
    for identity, image in frames.items():
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3)):
            raise ValueError("RGB/grey uint8 arrays required")
        sizes[identity] = dimensions((image.shape[1], image.shape[0]))
    solution = image_only.solve_image_cameras(frames)
    if solution.scale_known is not False or not set(solution.cameras).issubset(IDS):
        raise ValueError("unexpected camera solution scale/identity")
    rows = []
    for index, identity in enumerate(IDS):
        camera = camera_scalars(solution.cameras.get(identity), sizes[identity])
        rows.append({"frame_id": identity, "ordinal": index, "image_size": sizes[identity],
                     "status": "abstained" if camera is None else "available",
                     "reason": "production_camera_unavailable" if camera is None else "",
                     "camera": camera, "translation_resolved": False, "scale_resolved": False})
    result = {"schema": SCHEMA, "rule": dict(RULE), "input_manifest_sha256": MANIFEST_SHA,
              "source_closure_sha256": source_closure_sha256,
              "runtime_closure_sha256": runtime_closure_sha256,
              "input_seals": seals, "frames": rows}
    validate_predictions(result)
    return result


def validate_camera(camera, size):
    keys = {"yaw_degrees", "pitch_degrees", "roll_degrees", "image_size",
            "focal_length_pixels", "principal_point_pixels", "translation", "scale"}
    if (type(camera) is not dict or set(camera) != keys
            or camera["translation"] is not None or camera["scale"] is not None
            or dimensions(camera["image_size"]) != size):
        raise ValueError("camera scalar schema/dimensions")
    for key in ("yaw_degrees", "pitch_degrees", "roll_degrees"):
        number(camera[key])
    if min(pair(camera["focal_length_pixels"])) <= 0 or not -90 <= camera["pitch_degrees"] <= 90:
        raise ValueError("camera focal/pitch range")
    pair(camera["principal_point_pixels"])


def validate_predictions(table):
    if (type(table) is not dict or set(table) != {"schema", "rule", "input_manifest_sha256",
                                               "source_closure_sha256", "runtime_closure_sha256",
                                               "input_seals", "frames"}
            or table["schema"] != SCHEMA or encoded(table["rule"]) != encoded(RULE)
            or table["input_manifest_sha256"] != MANIFEST_SHA
            or type(table["frames"]) is not list or len(table["frames"]) != 6):
        raise ValueError("prediction census/provenance")
    for key in ("source_closure_sha256", "runtime_closure_sha256"):
        if closure_pin(table[key]) != table[key]:
            raise ValueError("frozen closure SHA256 must be normalized")
    validate_seals(table["input_seals"])
    keys = {"frame_id", "ordinal", "image_size", "status", "reason", "camera",
            "translation_resolved", "scale_resolved"}
    for index, row in enumerate(table["frames"]):
        if (type(row) is not dict or set(row) != keys or row["frame_id"] != IDS[index]
                or type(row["ordinal"]) is not int or row["ordinal"] != index
                or row["translation_resolved"] is not False or row["scale_resolved"] is not False):
            raise ValueError("prediction frame census")
        size = dimensions(row["image_size"])
        if row["status"] == "abstained":
            if row["camera"] is not None or row["reason"] != "production_camera_unavailable":
                raise ValueError("prediction abstention")
        elif row["status"] == "available" and row["reason"] == "":
            validate_camera(row["camera"], size)
        else:
            raise ValueError("prediction status")


def freeze_json(path, value):
    """Exclusive create, close, then verify byte identity, size and SHA."""
    payload = encoded(value)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("scalar output byte cap")
    path = Path(path)
    with path.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise ValueError("incomplete scalar freeze")
    raw = pinned_bytes(path, len(payload), digest(payload))
    if raw != payload:
        raise ValueError("scalar freeze readback mismatch")
    return {"bytes": len(raw), "sha256": digest(raw)}


def freeze_predictions(path, table):
    validate_predictions(table)
    return freeze_json(path, table)


def rotation(camera):
    return camera_rotation_matrix(CameraModel((0., 0., 0.), camera["yaw_degrees"],
        camera["pitch_degrees"], roll_degrees=camera["roll_degrees"], eye_height=0.))


def angular_distance(first, second):
    errors = []
    for k in range(4):
        angle = math.radians(90 * k)
        co, si = math.cos(angle), math.sin(angle)
        gauge = np.asarray(((co, 0., si), (0., 1., 0.), (-si, 0., co)))
        errors.append(float(np.degrees(np.arccos(np.clip(
            (np.trace(first.T @ second @ gauge) - 1.) / 2., -1., 1.)))))
    return min(errors)


def metrics(camera, truth_rotation, truth_pitch, truth_focal):
    if camera is None:
        return {"status": "abstained", "rotation_error_degrees": None,
                "pitch_error_degrees": None, "vertical_focal_error_relative": None,
                "qualifies": False}
    angle = angular_distance(truth_rotation, rotation(camera))
    focal = (camera["focal_length_pixels"][1] - truth_focal) / truth_focal
    return {"status": "available", "rotation_error_degrees": angle,
            "pitch_error_degrees": camera["pitch_degrees"] - truth_pitch,
            "vertical_focal_error_relative": focal,
            "qualifies": angle <= 5. and abs(focal) <= .20}


def baseline_camera(row, identity, size):
    old = row.get("image_camera")
    if type(old) is not dict or old.get("frame_id") != identity:
        raise ValueError("baseline camera frame join")
    if old.get("status") == "unavailable":
        return None
    if old.get("status") != "available":
        raise ValueError("baseline camera availability")
    camera = {k: old.get(k) for k in ("yaw_degrees", "pitch_degrees", "roll_degrees",
                                      "image_size", "focal_length_pixels", "principal_point_pixels")}
    camera.update(translation=None, scale=None)
    validate_camera(camera, size)
    return camera


def evaluate_frozen(prediction_path, freeze, capture_path, baseline_path):
    """Refuse missing, changed or malformed prediction freezes before truth parsing."""
    if (type(freeze) is not dict or set(freeze) != {"bytes", "sha256"}
            or type(freeze["bytes"]) is not int or not 0 < freeze["bytes"] <= MAX_JSON_BYTES
            or type(freeze["sha256"]) is not str or len(freeze["sha256"]) != 64):
        raise ValueError("prediction freeze seal required")
    raw = pinned_bytes(prediction_path, freeze["bytes"], freeze["sha256"])
    table = json.loads(raw)
    validate_predictions(table)
    # First truth parsing point, after persisted prediction size/SHA/readback.
    capture = json.loads(pinned_bytes(capture_path, CAPTURE_BYTES, CAPTURE_SHA))
    baseline = json.loads(pinned_bytes(baseline_path, BASELINE_BYTES, BASELINE_SHA))
    from scripts.evaluation.evaluate_spatial_matcher_ab import _actual_truth_camera
    if (type(capture) is not dict or type(baseline) is not dict
            or type(capture.get("frames")) is not list or len(capture["frames"]) != 6
            or type(capture.get("screenshots")) is not list or len(capture["screenshots"]) != 6
            or type(baseline.get("frames")) is not list or len(baseline["frames"]) != 6
            or type(baseline.get("input_seals")) is not list or len(baseline["input_seals"]) != 6):
        raise ValueError("evaluation metadata inventory")
    screenshots = {}
    for shot in capture["screenshots"]:
        if (type(shot) is not dict or type(shot.get("filename")) is not str
                or shot["filename"] in screenshots):
            raise ValueError("evaluation screenshot join")
        screenshots[shot["filename"]] = shot
    rows = []
    for index, (current, seal, actual_row, prior, prior_seal) in enumerate(zip(
            table["frames"], table["input_seals"], capture["frames"], baseline["frames"],
            baseline["input_seals"], strict=True)):
        identity = IDS[index]
        filename = Path(seal["path"]).name
        shot = screenshots.get(filename, {})
        actual = actual_row.get("actual", {}) if type(actual_row) is dict else {}
        size = current["image_size"]
        if (type(actual_row) is not dict or type(actual) is not dict
                or actual_row.get("filename") != filename or actual_row.get("captured") is not True
                or actual_row.get("frame_id") != Path(filename).stem
                or shot.get("frame_id") != actual_row.get("frame_id")
                or type(shot.get("sha256")) is not str or shot["sha256"].upper() != seal["sha256"]
                or type(shot.get("bytes")) is not int or shot["bytes"] != seal["bytes"]
                or type(actual.get("screenshot_sha256")) is not str
                or actual["screenshot_sha256"].upper() != seal["sha256"]
                or type(actual.get("screenshot_bytes")) is not int
                or actual["screenshot_bytes"] != seal["bytes"]
                or type(prior) is not dict or prior.get("frame_id") != identity
                or prior.get("shape") != [size[1], size[0], 3]
                or type(prior_seal) is not dict or prior_seal.get("path") != seal["path"]
                or type(prior_seal.get("bytes")) is not int or prior_seal["bytes"] != seal["bytes"]
                or type(prior_seal.get("sha256")) is not str
                or prior_seal["sha256"].upper() != seal["sha256"]):
            raise ValueError("evaluation filename/ordinal/screenshot size/SHA join")
        truth, roll_source = _actual_truth_camera(actual_row, [size[1], size[0], 3])
        truth_focal = size[1] / 2. / math.tan(math.radians(truth.fov_degrees) / 2.)
        old = baseline_camera(prior, identity, size)
        current_error = metrics(current["camera"], camera_rotation_matrix(truth), truth.pitch_degrees, truth_focal)
        old_error = metrics(old, camera_rotation_matrix(truth), truth.pitch_degrees, truth_focal)
        comparable = current["camera"] is not None and old is not None
        deltas = {key: current_error[key] - old_error[key] if comparable else None for key in
                  ("rotation_error_degrees", "pitch_error_degrees", "vertical_focal_error_relative")}
        principal = ([a - b for a, b in zip(current["camera"]["principal_point_pixels"],
                                          old["principal_point_pixels"], strict=True)] if comparable else None)
        rows.append({"frame_id": identity, "ordinal": index, "filename": filename,
                     "current": current_error, "AB05": old_error, "deltas": deltas,
                     "principal_point_change_pixels": principal,
                     "principal_point_change_norm_pixels": math.hypot(*principal) if principal is not None else None,
                     "roll_truth_source": roll_source})
    primary = rows[3]["current"]["qualifies"]
    guard = all(row["current"]["qualifies"] for i, row in enumerate(rows) if i != 3)
    decision = ("stop_mask_camera_recovery_hypothesis" if not primary else
                "mixed_no_general_recovery_claim" if not guard else "one_scene_camera_evidence_only")
    return {"schema": "mc-gui-mask-camera-evaluation-v1", "frames": rows,
            "prediction_freeze": freeze, "evaluation_capture_sha256": CAPTURE_SHA,
            "baseline_report_sha256": BASELINE_SHA, "input_manifest_sha256": MANIFEST_SHA,
            "source_closure_sha256": table["source_closure_sha256"],
            "runtime_closure_sha256": table["runtime_closure_sha256"],
            "primary_success": primary, "regression_guard": guard, "joint_camera_success": primary and guard,
            "decision": decision, "truth_read_after_prediction_freeze": True,
            "evidence_scope": "one_scene_cameras_only_no_matcher_geometry_state_or_export_claim",
            "default_promoted": False, "camera_selected_by_truth": False}


def decode_frames(paths, seals):
    from io import BytesIO
    from PIL import Image
    frames = {}
    for identity, path, seal in zip(IDS, paths, seals, strict=True):
        raw = pinned_bytes(path, seal["bytes"], seal["sha256"], MAX_IMAGE_BYTES)
        with Image.open(BytesIO(raw)) as source:
            if source.format != "PNG":
                raise ValueError("pinned image must be PNG")
            size = dimensions((source.width, source.height))
            image = np.asarray(source.convert("RGB")).copy()
        if image.shape != (size[1], size[0], 3) or image.dtype != np.uint8:
            raise ValueError("decoded image dimensions/type")
        frames[identity] = image
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--evaluation-capture", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--source-closure-sha256", required=True,
                        help="source closure SHA256 verified by the execution packet owner")
    parser.add_argument("--runtime-closure-sha256", required=True,
                        help="runtime closure SHA256 verified by the execution packet owner")
    parser.add_argument("--prediction-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args(argv)
    source_closure_sha256 = closure_pin(args.source_closure_sha256)
    runtime_closure_sha256 = closure_pin(args.runtime_closure_sha256)
    inputs = (args.input_manifest, args.evaluation_capture, args.baseline_report)
    for supplied, expected in zip(inputs, (MANIFEST_PATH, CAPTURE_PATH, BASELINE_PATH), strict=True):
        if Path(supplied).resolve() != (ROOT / expected).resolve():
            raise ValueError("exact pinned metadata paths required")
    outputs = [Path(p).resolve() for p in (args.prediction_output, args.report_output)]
    source_paths = {Path(p).resolve() for p in (*inputs, *args.image)}
    if (len(args.image) != 6 or len({Path(p).resolve() for p in args.image}) != 6
            or outputs[0] == outputs[1] or any(p in source_paths or p.exists() for p in outputs)
            or outputs[0].parent != outputs[1].parent or not outputs[0].parent.is_dir()):
        raise ValueError("six distinct inputs, existing output parent, and distinct absent output files required")
    seals = sealed_input_preflight(args.input_manifest, args.image)
    # Verify the evaluation pins now WITHOUT parsing them or retaining truth.
    pinned_bytes(args.evaluation_capture, CAPTURE_BYTES, CAPTURE_SHA)
    pinned_bytes(args.baseline_report, BASELINE_BYTES, BASELINE_SHA)
    frames = decode_frames(args.image, seals)
    predictions = predict_cameras(frames, seals, source_closure_sha256=source_closure_sha256,
                                  runtime_closure_sha256=runtime_closure_sha256)
    del frames
    freeze = freeze_predictions(outputs[0], predictions)
    report = evaluate_frozen(outputs[0], freeze, args.evaluation_capture, args.baseline_report)
    for path, seal in zip(args.image, seals, strict=True):
        pinned_bytes(path, seal["bytes"], seal["sha256"], MAX_IMAGE_BYTES)
    freeze_json(outputs[1], report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
