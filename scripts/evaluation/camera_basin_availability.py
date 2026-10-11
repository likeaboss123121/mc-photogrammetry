"""Bounded camera-basin recall diagnostic; never selects a production camera.

Ordinary pixels drive one detector/coarse pass and at most four existing local
refinements per frame. A complete scalar candidate table is persisted and hashed
before evaluation capture or historical baseline reports are read.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from mc_photogrammetry.reconstruction import calibration as c
from mc_photogrammetry.pipeline.image_only import _to_grayscale
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix

SOURCE_SHA = "D6A233BA4EB88FCF36A3E1E2D44D8AE821D536BCE5B0E13A5872645EA6F825F3"
MAX_BASINS = 4
BASIN_DEGREES = 10.0
MAX_ORIENTATIONS = 45 * 89
COARSE_CALLS = MAX_ORIENTATIONS * 19
MAX_JSON_BYTES = 1_048_576
INPUT_MANIFEST_SHA = "90F1AF81157225066E4494047BA67CEF0C7CC646A36E4427FED782810E2BB12D"


def sealed_input_preflight(manifest_path, image_paths):
    """Truth-free inventory first, then all six byte seals before ANY decoding."""
    manifest, digest = snapshot(manifest_path)
    if digest != INPUT_MANIFEST_SHA or len(image_paths) != 6:
        raise ValueError("accepted input manifest seal/count")
    rows = manifest.get("inputs")
    if type(rows) is not list or len(rows) != 6:
        raise ValueError("accepted input inventory")
    seals = []
    for index, (row, provided) in enumerate(zip(rows, image_paths, strict=True)):
        relative = Path(row["path"])
        expected = (ROOT / relative).resolve()
        actual = Path(provided).resolve()
        if (relative.is_absolute() or not expected.is_relative_to(ROOT)
                or row["frame"] != f"image-{index:04d}" or row["ordinal"] != index
                or actual != expected or Path(row["absolute_path"]).resolve() != expected
                or type(row["bytes"]) is not int or not 0 < row["bytes"] <= 16_777_216
                or type(row["sha256"]) is not str or len(row["sha256"]) != 64
                or any(ch not in "0123456789ABCDEF" for ch in row["sha256"])
                or actual.stat().st_size != row["bytes"]):
            raise ValueError("accepted input path/order/size")
        seals.append({key: row[key] for key in ("path", "bytes", "sha256")})
    for seal, provided in zip(seals, image_paths, strict=True):
        verified_input_bytes(provided, seal)
    return seals, digest


def verified_input_bytes(path, seal):
    raw = Path(path).read_bytes()
    if len(raw) != seal["bytes"] or hashlib.sha256(raw).hexdigest().upper() != seal["sha256"]:
        raise ValueError("source image seal mismatch before decoding/work")
    return raw


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def snapshot(path, cap=MAX_JSON_BYTES):
    path = Path(path)
    if path.stat().st_size > cap:
        raise ValueError("snapshot byte cap")
    raw = path.read_bytes()
    if len(raw) > cap:
        raise ValueError("snapshot byte cap")
    return json.loads(raw), hashlib.sha256(raw).hexdigest().upper()


def rotation(yaw, pitch, roll=0.):
    return camera_rotation_matrix(CameraModel((0., 0., 0.), yaw, pitch,
                                              roll_degrees=roll, eye_height=0.))


def angular_distance(first, second):
    errors = []
    for k in range(4):
        angle = math.radians(90 * k)
        co, si = math.cos(angle), math.sin(angle)
        gauge = np.asarray(((co, 0., si), (0., 1., 0.), (-si, 0., co)))
        errors.append(float(np.degrees(np.arccos(np.clip(
            (np.trace(first.T @ second @ gauge) - 1) / 2, -1., 1.)))))
    return min(errors)


class CoarseCensus:
    """One best focal per orientation; first enumeration wins exact score ties."""
    def __init__(self):
        self.calls = 0
        self.rows = {}

    def remember(self, value, focal, yaw, pitch):
        if self.calls >= COARSE_CALLS:
            raise ValueError("coarse call cap")
        key = (float(yaw), float(pitch))
        if key not in self.rows and len(self.rows) >= MAX_ORIENTATIONS:
            raise ValueError("coarse orientation cap")
        row = (float(value), float(focal), *key, self.calls)
        if not all(math.isfinite(v) for v in row[:-1]):
            raise ValueError("nonfinite coarse candidate")
        if key not in self.rows or value > self.rows[key][0]:
            self.rows[key] = row
        self.calls += 1


def select_basins(rows):
    if len(rows) > MAX_ORIENTATIONS:
        raise ValueError("coarse orientation cap")
    for row in rows:
        if (len(row) != 5 or type(row[4]) is not int or row[4] < 0
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in row[:4])
                or row[0] < 0 or row[1] <= 0 or not 0 <= row[2] < 90 or not -88 <= row[3] <= 88):
            raise ValueError("invalid coarse candidate")
    if len({row[4] for row in rows}) != len(rows):
        raise ValueError("duplicate coarse enumeration")
    ranked = sorted(rows, key=lambda row: (-row[0], row[4]))
    selected, matrices = [], []
    for row in ranked:
        matrix = rotation(row[2], row[3])
        if all(angular_distance(old, matrix) > BASIN_DEGREES for old in matrices):
            selected.append(row)
            matrices.append(matrix)
            if len(selected) == MAX_BASINS:
                break
    return selected


def validate_table(table):
    """Fail closed on incomplete producer census before any truth is read."""
    if (type(table) is not dict or table.get("schema") != "mc-camera-basin-candidates-v1"
            or table.get("calibration_source_sha256") != SOURCE_SHA
            or type(table.get("frames")) is not list or len(table["frames"]) != 6
            or type(table.get("input_seals")) is not list or len(table["input_seals"]) != 6
            or table.get("rule") != CANDIDATE_RULE):
        raise ValueError("candidate table inventory/provenance")
    paths = set()
    for index, (frame, seal) in enumerate(zip(table["frames"], table["input_seals"], strict=True)):
        if (type(seal) is not dict or type(seal.get("path")) is not str or not seal["path"]
                or seal["path"] in paths or type(seal.get("bytes")) is not int or seal["bytes"] <= 0
                or type(seal.get("sha256")) is not str or len(seal["sha256"]) != 64
                or any(ch not in "0123456789ABCDEF" for ch in seal["sha256"])):
            raise ValueError("candidate input seal")
        paths.add(seal["path"])
        if (type(frame) is not dict or frame.get("frame_id") != f"image-{index:04d}"
                or type(frame.get("shape")) is not list or len(frame["shape"]) != 3
                or any(type(n) is not int or n <= 0 for n in frame["shape"])
                or frame["shape"][2] != 3 or frame["shape"][0]*frame["shape"][1] > 6_000_000
                or type(frame.get("detector_passes")) is not int or frame["detector_passes"] != 1
                or type(frame.get("candidates")) is not list
                or type(frame.get("seconds")) not in (int, float)
                or not math.isfinite(frame["seconds"]) or frame["seconds"] < 0):
            raise ValueError("candidate frame census")
        candidates = frame["candidates"]
        if frame.get("status") == "unavailable":
            if candidates or any(type(frame.get(k)) is not int or frame[k] != 0 for k in
                                 ("coarse_passes", "coarse_calls", "refinements")):
                raise ValueError("abstention census")
            continue
        if (frame.get("status") != "complete" or not 1 <= len(candidates) <= MAX_BASINS
                or any(type(frame.get(k)) is not int or frame[k] != v for k, v in
                       (("coarse_passes", 1), ("coarse_calls", COARSE_CALLS),
                        ("orientation_rows", MAX_ORIENTATIONS), ("refinements", len(candidates))))):
            raise ValueError("candidate coarse census")
        seeds = []
        for rank, candidate in enumerate(candidates):
            if (type(candidate) is not dict or type(candidate.get("rank")) is not int
                    or candidate["rank"] != rank or candidate.get("incumbent") is not (rank == 0)
                    or candidate.get("width") != frame["shape"][1]
                    or candidate.get("height") != frame["shape"][0]):
                raise ValueError("candidate rank/dimension")
            for key in ("focal_pixels", "yaw_degrees", "pitch_degrees", "roll_degrees", "objective_score"):
                value = candidate.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("nonfinite candidate")
            if candidate["focal_pixels"] <= 0 or candidate["objective_score"] < 0:
                raise ValueError("candidate focal/objective")
            counts = candidate.get("per_axis_inliers")
            if (type(counts) not in (list, tuple) or len(counts) != 3
                    or any(type(n) is not int or n < 0 for n in counts)
                    or type(candidate.get("segments")) is not int or candidate["segments"] < 12
                    or type(candidate.get("inliers")) is not int
                    or candidate["inliers"] != sum(counts) or candidate["inliers"] > candidate["segments"]):
                raise ValueError("candidate inlier census")
            point = candidate.get("principal_point")
            if point is not None and (type(point) not in (list, tuple) or len(point) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
                raise ValueError("candidate principal point")
            seed = candidate.get("coarse_seed", {})
            seeds.append(tuple(seed.get(k) for k in ("score", "focal", "yaw", "pitch", "enumeration")))
        if select_basins(seeds) != seeds:
            raise ValueError("candidate coarse rank/distinctness")


CANDIDATE_RULE = {"K": 4, "orientation_distance_strictly_greater_degrees": 10.,
                  "gauge": "min_world_Y_quarter_turn_SO3", "tie": "first_coarse_enumeration",
                  "selection": "existing_objective_only_no_truth"}


def source_functions(census, detect):
    """Source-bound clones: observer insertion and coarse-loop substitution only.

    Production globals/functions are never patched. The baseline clone is the
    actual estimate_camera AST with one read-only callback inside its coarse loop.
    Other clones replace that same loop with a previously scored seed; all input,
    rescore, fine search, optics and output code remains the original source.
    """
    source = Path(c.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest().upper() != SOURCE_SHA:
        raise ValueError("accepted calibration source pin changed")
    original = ast.parse(inspect.getsource(c.estimate_camera))
    baseline = ast.parse(inspect.getsource(c.estimate_camera))
    fn = baseline.body[0]
    coarse = [n for n in fn.body if isinstance(n, ast.For)
              and isinstance(n.target, ast.Name) and n.target.id == "yaw"]
    if len(coarse) != 1:
        raise ValueError("coarse source seam changed")
    inner = coarse[0].body[0].body[0]
    if (not isinstance(inner, ast.For) or not isinstance(inner.target, ast.Name)
            or inner.target.id != "candidate" or not isinstance(inner.body[0], ast.Assign)):
        raise ValueError("coarse source seam changed")
    inner.body.insert(1, ast.parse("_remember(value, candidate, yaw, pitch)").body[0])
    ast.fix_missing_locations(baseline)
    baseline_ns = dict(c.__dict__, _remember=census.remember, detect_segments=detect)
    exec(compile(baseline, "<source-bound-coarse-observer>", "exec"), baseline_ns)

    local_fn = original.body[0]
    loops = [n for n in local_fn.body if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "yaw"]
    index = local_fn.body.index(loops[0])
    local_fn.body[index:index + 1] = ast.parse("best = _seed").body
    ast.fix_missing_locations(original)
    local_ns = dict(c.__dict__)
    exec(compile(original, "<source-bound-local-refinement>", "exec"), local_ns)

    def refine(grey, segments, seed):
        local_ns["_seed"] = tuple(seed[:4])
        return local_ns["estimate_camera"](grey, segments=segments)
    return baseline_ns["estimate_camera"], refine


def candidate_row(found, seed, segments, rank):
    data = asdict(found)
    # Accepted optics refinement returns NumPy coordinates. Detach to native,
    # finite scalars before validation/JSON publication; never alter the fit.
    point = data["principal_point"]
    if point is not None:
        if (type(point) is not tuple or len(point) != 2
                or any(isinstance(value, (bool, np.bool_))
                       or not isinstance(value, (int, float, np.integer, np.floating))
                       for value in point)):
            raise ValueError("candidate principal point")
        point = tuple(float(value) for value in point)
        if not all(math.isfinite(value) for value in point):
            raise ValueError("candidate principal point")
        data["principal_point"] = point
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    score = c._score(segments, lengths, found.focal_pixels, found.yaw_degrees,
                     found.pitch_degrees, found.width, found.height, c.INLIER_PIXELS,
                     found.roll_degrees, found.principal_point)
    data["median_residual"] = data["median_residual"] if math.isfinite(data["median_residual"]) else None
    data.update(rank=rank, incumbent=rank == 0, objective_score=score,
                coarse_seed={"score": seed[0], "focal": seed[1], "yaw": seed[2],
                             "pitch": seed[3], "enumeration": seed[4]})
    return data


def frame_candidates(frame_id, image):
    grey = _to_grayscale(image)
    if grey.shape[0] * grey.shape[1] > 6_000_000:
        raise ValueError("native pixel cap")
    census = CoarseCensus()
    detected = []

    def detect(*args, **kwargs):
        if detected:
            raise ValueError("second detector pass refused")
        rows = c.detect_segments(*args, **kwargs)
        detected.append(rows)
        return rows

    baseline, refine = source_functions(census, detect)
    start = time.perf_counter()
    incumbent = baseline(grey)
    if len(detected) != 1:
        raise ValueError("detector pass census")
    if incumbent is None:
        return {"frame_id": frame_id, "shape": list(image.shape), "status": "unavailable",
                "reason": "estimator_abstention", "detector_passes": 1, "coarse_passes": 0,
                "coarse_calls": census.calls, "refinements": 0, "candidates": [],
                "seconds": time.perf_counter() - start}
    if census.calls != COARSE_CALLS or len(census.rows) != MAX_ORIENTATIONS:
        raise ValueError("coarse census incomplete")
    seeds = select_basins(list(census.rows.values()))
    candidates = [candidate_row(incumbent, seeds[0], detected[0], 0)]
    for rank, seed in enumerate(seeds[1:], start=1):
        found = refine(grey, detected[0], seed)
        if found is None:
            raise ValueError("retained seed refinement unexpectedly abstained")
        candidates.append(candidate_row(found, seed, detected[0], rank))
    return {"frame_id": frame_id, "shape": list(image.shape), "status": "complete",
            "reason": "", "detector_passes": 1, "coarse_passes": 1,
            "coarse_calls": census.calls, "orientation_rows": len(census.rows),
            "refinements": len(candidates), "candidates": candidates,
            "seconds": time.perf_counter() - start}


def freeze_candidates(path, table):
    payload = encoded(table)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("candidate table byte cap")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(payload)
    raw = path.read_bytes()
    if raw != payload:
        raise ValueError("candidate freeze readback mismatch")
    return hashlib.sha256(raw).hexdigest().upper()


def evaluate_frozen(table_path, table_hash, capture_path, baseline_path):
    # Importing evaluator helpers does not read truth; all file reads below are
    # strictly after successful persisted candidate freeze and digest readback.
    table, digest = snapshot(table_path)
    if digest != table_hash:
        raise ValueError("frozen candidate digest mismatch")
    validate_table(table)
    from scripts.evaluation.evaluate_spatial_matcher_ab import _actual_truth_camera
    capture, capture_hash = snapshot(capture_path, 2 * MAX_JSON_BYTES)
    baseline, baseline_hash = snapshot(baseline_path)
    if len(capture.get("frames", [])) != 6 or len(baseline.get("frames", [])) != 6:
        raise ValueError("evaluation frame inventory")
    screenshots = {row["filename"]: row["sha256"].upper() for row in capture["screenshots"]}
    if len(capture["screenshots"]) != 6 or len(screenshots) != 6:
        raise ValueError("evaluation screenshot inventory")
    rows = []
    for i, (frame, truth, prior) in enumerate(zip(table["frames"], capture["frames"], baseline["frames"], strict=True)):
        identity = f"image-{i:04d}"
        seal = table["input_seals"][i]
        if (frame["frame_id"] != identity or prior["frame_id"] != identity
                or Path(seal["path"]).name != truth["filename"]
                or screenshots.get(truth["filename"]) != seal["sha256"]
                or truth["actual"]["screenshot_sha256"].upper() != seal["sha256"]
                or truth["actual"]["screenshot_bytes"] != seal["bytes"]):
            raise ValueError("evaluation frame pixel join")
        actual, roll_source = _actual_truth_camera(truth, frame["shape"])
        actual_focal = frame["shape"][0]/2/math.tan(math.radians(actual.fov_degrees)/2)
        errors = []
        for candidate in frame["candidates"]:
            angle = angular_distance(camera_rotation_matrix(actual), rotation(
                candidate["yaw_degrees"], candidate["pitch_degrees"], candidate["roll_degrees"]))
            focal_error = (candidate["focal_pixels"]-actual_focal)/actual_focal
            errors.append({"rank": candidate["rank"], "rotation_error_degrees": angle,
                           "pitch_error_degrees": candidate["pitch_degrees"]-actual.pitch_degrees,
                           "focal_error_relative": focal_error,
                           "qualifies": angle <= 5. and abs(focal_error) <= .20})
        preserved = False
        if frame["candidates"]:
            current = frame["candidates"][0]
            old = prior["image_camera"]
            point = current["principal_point"] or [frame["shape"][1]/2, frame["shape"][0]/2]
            preserved = (old["status"] == "available" and all(current[key] == old[key] for key in
                ("yaw_degrees", "pitch_degrees", "roll_degrees"))
                and old["focal_length_pixels"] == [current["focal_pixels"]]*2
                and old["principal_point_pixels"] == list(point))
        rows.append({"frame_id": identity, "candidate_errors": errors,
                     "incumbent_matches_AB05": preserved, "roll_truth_source": roll_source})
    available = any(row["qualifies"] for row in rows[3]["candidate_errors"])
    return {"schema": "mc-camera-basin-availability-v1", "status": "complete",
            "candidate_table_sha256": table_hash, "frames": rows,
            "evaluation_capture_sha256": capture_hash, "baseline_report_sha256": baseline_hash,
            "image0003_candidate_available": available,
            "other_five_incumbents_preserved": all(row["incumbent_matches_AB05"] for i, row in enumerate(rows) if i != 3),
            "decision": "availability_only_requires_image_only_selector" if available else "stop_K4_branch",
            "truth_read_after_persisted_candidate_freeze": True,
            "default_promoted": False, "camera_selected_by_truth": False,
            "correct_pair_edges": None, "geometry_F1": None, "state_match": None, "export": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--evaluation-capture", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args(argv)
    if len(args.image) != 6 or len(set(args.image)) != 6:
        raise ValueError("exact six distinct image paths required")
    if args.candidate_output == args.report_output or any(Path(p).exists() for p in (args.candidate_output, args.report_output)):
        raise ValueError("fresh distinct outputs required")
    seals, manifest_hash = sealed_input_preflight(args.input_manifest, args.image)
    from PIL import Image
    from io import BytesIO
    start = time.perf_counter()
    table = {"schema": "mc-camera-basin-candidates-v1", "calibration_source_sha256": SOURCE_SHA,
             "rule": CANDIDATE_RULE, "input_manifest_sha256": manifest_hash,
             "frames": [], "input_seals": []}
    for index, path in enumerate(args.image):
        seal = seals[index]
        raw = verified_input_bytes(path, seal)
        with Image.open(BytesIO(raw)) as source:
            if source.width * source.height > 6_000_000:
                raise ValueError("native pixel cap")
            image = np.asarray(source.convert("RGB")).copy()
        table["frames"].append(frame_candidates(f"image-{index:04d}", image))
        table["input_seals"].append(seal)
        del image, raw
    validate_table(table)
    digest = freeze_candidates(args.candidate_output, table)
    report = evaluate_frozen(args.candidate_output, digest, args.evaluation_capture, args.baseline_report)
    report.update(wall_seconds=time.perf_counter()-start, input_seals=table["input_seals"],
                  input_manifest_sha256=manifest_hash)
    # Source inputs must remain unchanged during the one-shot diagnostic.
    for seal, path in zip(table["input_seals"], args.image, strict=True):
        verified_input_bytes(path, seal)
    freeze_candidates(args.report_output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
