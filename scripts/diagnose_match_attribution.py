"""DS-16 reference-assisted attribution of DS-14 joint-match failures.

This is an **evaluation-only** diagnostic for the negative DS-15 result (8/37
solved, 4 correct classes, 4 false solves, every solved graph a forest). It does
not change any production module, threshold, test or frozen DS-14/DS-15 artifact.
It answers one question: *where* do DS-14's 37 real pairs lose the geometry?

The run is split exactly like DS-12/DS-14/DS-15 so reference data cannot leak
into extraction:

1. **Blind candidate freeze** (``--freeze-only``, default first phase). Loads
   only the 15 frozen T2/T6/t3-sparse-8 screenshots, the derived GUI usable
   masks and the image-derived M3 calibration artifacts. For every unordered
   pair (37) it detects SIFT features with the *exact* DS-14 settings, builds the
   reciprocal top-k candidate graph, and runs the unchanged DS-14
   ``match_feature_frames`` adjudication. Candidate edges, selections, feature
   tables and predictions are written to ``match_candidates.json`` +
   ``match_candidates.npz`` and digested. The run aborts before opening any
   manifest if the predictions do not reproduce the frozen DS-15 pair records.

2. **Reference attribution** (``--attribute-only``, default second phase). Only
   now are ``cameras.json``, ``scene.json`` and the reference ``.litematic``
   opened. Each frame's reference volume is rendered from its *manifest* camera
   with the shared CPU renderer (``export.attribution.render_result``, exact
   model geometry, ownership + camera depth per pixel). Every keypoint is mapped
   to the visible reference block, face and world hit point. Ownership
   coordinates are verified with a renderer round-trip (re-projecting every hit
   point through the same manifest camera). Whether a manifest pose actually
   explains the *image* matches is measured separately per pair: the ground-truth
   epipolar + cheirality count is compared with a shuffled-pair chance control.

Preregistered oracle definitions (fixed before the first run, never tuned):

* ``WORLD_HIT_DISTANCE_BLOCKS = 0.5`` - two endpoints are the same surface point
  only when their world hit points are within half a block.
* ``FACE_NORMAL_COSINE_MIN = cos(30 deg)`` - their face normals must agree.
* ``EPIPOLAR_NECESSARY_DEGREES = 1.0`` - the DS-14 inlier residual, computed with
  the *manifest* relative rotation and baseline, is a necessary condition.
* ``REFERENCE_RENDER_SCALE = 0.35`` - the attribution module's own render scale.

Labels per candidate edge: ``geometric`` (both hit, in the usable mask, distance
and normal gates), ``strict`` (same block coordinate **and** face, within
distance) and ``coordinate_only`` (same block, within distance). The oracle
truth set is *every* keypoint pair - not only candidate edges - that is
geometrically consistent, found with a tolerance-sized spatial hash, so
``top-k recall`` is measurable.

Four fixed comparisons, all through the public DS-11 ``solve_pair`` adjudicator
(the unchanged DS-14 prediction is reported separately):

===========================  ==================  ====================
                            current cameras     reference cameras
===========================  ==================  ====================
DS-14 candidates            end-to-end          candidate recall
oracle-consistent edges     calibration         upper bound
===========================  ==================  ====================

Run with the workspace interpreter::

    .venv\\Scripts\\python.exe scripts/diagnose_match_attribution.py --freeze-only
    .venv\\Scripts\\python.exe scripts/diagnose_match_attribution.py --attribute-only
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS  # noqa: E402
from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider  # noqa: E402
from mc_photogrammetry.export.attribution import (  # noqa: E402
    face_colours,
    face_model_faces,
    face_occluding_slots,
    face_shapes,
    render_result,
)
from mc_photogrammetry.export.compare import load_litematic  # noqa: E402
from mc_photogrammetry.pipeline.connected_camera import PairGates, solve_pair  # noqa: E402
from mc_photogrammetry.pipeline.contracts import (  # noqa: E402
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
)
from mc_photogrammetry.pipeline.correspondence import (  # noqa: E402
    CorrespondenceSettings,
    FrameFeatures,
    detect_features,
    reciprocal_topk_graph,
    usable_mask_for,
)
from mc_photogrammetry.pipeline.joint_matching import (  # noqa: E402
    JointMatchSettings,
    match_feature_frames,
)
from mc_photogrammetry.reconstruction.camera import (  # noqa: E402
    camera_eye,
    camera_rotation_matrix,
    intrinsics_for,
    load_camera_manifest,
    project_points,
)
from mc_photogrammetry.reconstruction.scene import load_scene_transform  # noqa: E402
from mc_photogrammetry.vision.gui_mask import create_gui_mask  # noqa: E402

from diagnose_connected_pairs import (  # noqa: E402
    _GAUGE_ROTATIONS,
    _reference_class,
    angle_between,
    digest,
    load_rgb,
    plain,
    source_revision,
    to_camera_estimate,
)
from diagnose_joint_all_pairs import frozen_frame_sets  # noqa: E402
from diagnose_joint_pairs import _estimate_from_record, _result_record, canonical_digest  # noqa: E402

OUTPUT_DIR = ROOT / "exports" / "diagnostics" / "ds16"
DS15_ESTIMATES = ROOT / "exports" / "diagnostics" / "ds15" / "joint_estimates.json"

# Preregistered constants. Fixed here before the first run; the report never
# re-thresholds them after reading an outcome.
REFERENCE_RENDER_SCALE = 0.35
WORLD_HIT_DISTANCE_BLOCKS = 0.5
FACE_NORMAL_COSINE_MIN = float(np.cos(np.radians(30.0)))
EPIPOLAR_NECESSARY_DEGREES = 1.0
ALIGNMENT_MEDIAN_PX_MAX = 10.0
DESCRIPTOR_CHUNK = 256
SPATIAL_GRID = (4, 4)
FACE_SLOTS_INDEX = {slot: position for position, slot in enumerate(FACE_BASIS)}


# --------------------------------------------------------------------------- #
# Phase 1: blind candidate freeze (images, derived masks, M3 artifacts only).
# --------------------------------------------------------------------------- #
def _directed_topk(query: np.ndarray, train: np.ndarray, neighbours: int) -> np.ndarray:
    """Directed top-k L2 indices, ``(index, distance)`` tie-break, DS-14 order.

    Re-derived locally only to measure *directed* top-k recall; the reciprocal
    graph itself is always the module's public ``reciprocal_topk_graph``.
    """
    query = np.asarray(query, dtype=np.float64)
    train = np.asarray(train, dtype=np.float64)
    rows = len(query)
    keep = max(1, min(int(neighbours), len(train)))
    index = np.full((rows, keep), -1, dtype=np.int64)
    if rows == 0 or len(train) == 0:
        return index
    train_squared = np.einsum("ij,ij->i", train, train)
    for start in range(0, rows, DESCRIPTOR_CHUNK):
        block = query[start : start + DESCRIPTOR_CHUNK]
        gram = block @ train.T
        squared = (
            np.einsum("ij,ij->i", block, block)[:, None] + train_squared[None, :] - 2.0 * gram
        )
        np.maximum(squared, 0.0, out=squared)
        candidate = np.argpartition(squared, keep - 1, axis=1)[:, :keep]
        candidate_distance = np.take_along_axis(squared, candidate, axis=1)
        order = np.lexsort((candidate, candidate_distance), axis=1)
        index[start : start + DESCRIPTOR_CHUNK] = np.take_along_axis(candidate, order, axis=1)
    if len(train) < keep:
        index[:, len(train) :] = -1
    return index


def extract_candidates(out_dir: Path) -> dict:
    """Phase 1: extract and freeze DS-14 candidates/predictions, manifest-free."""
    matcher = JointMatchSettings()
    detection = CorrespondenceSettings(
        max_features=matcher.max_features,
        mask_boundary_pad_px=matcher.mask_boundary_pad_px,
    )
    gates = PairGates()
    sets = frozen_frame_sets()
    out_dir.mkdir(parents=True, exist_ok=True)

    features: dict[str, FrameFeatures] = {}
    frame_records: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    pairs: list[dict] = []
    inputs: dict[str, dict] = {}
    started_all = time.perf_counter()

    for entry in sets:
        fixture = entry["fixture"]
        estimates = entry["estimates"]
        for name in entry["frames"]:
            if name in features:
                continue
            path = ROOT / fixture / name
            rgb = load_rgb(path)
            mask = usable_mask_for(name, rgb.shape, None)
            feature = detect_features(name, rgb, usable_mask=mask, settings=detection)
            features[name] = feature
            key = len(frame_records)
            frame_records.append(
                {
                    "key": key,
                    "frame": name,
                    "fixture": fixture,
                    "calibration_artifact": entry["calibration_artifact"],
                    "width": int(feature.width),
                    "height": int(feature.height),
                    "usable_pixels": int(feature.usable_pixels),
                    "keypoints": int(len(feature.points)),
                    "boundary_dropped": int(feature.boundary_dropped),
                }
            )
            arrays[f"feat_points_{key}"] = np.asarray(feature.points, dtype=np.float32)
            descriptors = np.asarray(feature.descriptors, dtype=np.float32)
            if (
                descriptors.size
                and float(descriptors.min()) >= 0.0
                and float(descriptors.max()) <= 255.0
                and np.array_equal(descriptors, np.rint(descriptors))
            ):
                arrays[f"feat_desc_{key}"] = descriptors.astype(np.uint8)
            else:
                arrays[f"feat_desc_{key}"] = descriptors
            arrays[f"feat_sizes_{key}"] = np.asarray(feature.sizes, dtype=np.float32)
            inputs[str(path.relative_to(ROOT))] = {
                "sha256": digest(path),
                "size": [int(feature.width), int(feature.height)],
                "keypoints": int(len(feature.points)),
            }

        for left_name, right_name in itertools.combinations(list(entry["frames"]), 2):
            left = features[left_name]
            right = features[right_name]
            graph = reciprocal_topk_graph(
                left.descriptors,
                right.descriptors,
                neighbors=matcher.neighbors,
                max_edges=matcher.max_candidate_edges,
                distance_ceiling=matcher.graph_distance_ceiling,
            )
            result = match_feature_frames(
                left,
                right,
                to_camera_estimate(estimates[left_name]),
                to_camera_estimate(estimates[right_name]),
                matcher=matcher,
                gates=gates,
                ratio_test=detection.ratio_test,
            )
            index = len(pairs)
            edges = np.asarray(graph.edges, dtype=np.float64).reshape(-1, 2)
            arrays[f"cand_{index}"] = (
                np.column_stack((edges, np.asarray(graph.distances, dtype=np.float64)))
                if len(edges)
                else np.empty((0, 3), dtype=np.float64)
            )
            arrays[f"sel_{index}"] = np.asarray(result.matches, dtype=np.float64).reshape(-1, 4)
            key_by_frame = {record["frame"]: record["key"] for record in frame_records}
            pairs.append(
                {
                    "index": index,
                    "set": entry["set"],
                    "fixture": fixture,
                    "calibration_artifact": entry["calibration_artifact"],
                    "left": left_name,
                    "right": right_name,
                    "left_frame_key": key_by_frame[left_name],
                    "right_frame_key": key_by_frame[right_name],
                    "left_estimate": asdict(estimates[left_name]),
                    "right_estimate": asdict(estimates[right_name]),
                    "graph": {
                        "candidate_edges": int(len(graph.edges)),
                        "total_edges": int(graph.total_edges),
                        "truncated": int(graph.truncated),
                    },
                    "prediction": _result_record(result),
                }
            )

    npz_path = out_dir / "match_candidates.npz"
    np.savez_compressed(npz_path, **arrays)

    payload = {
        "schema": "ds16-freeze-1",
        "phase": "blind_candidate_extraction",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
        },
        "settings": {
            "joint_matcher": asdict(matcher),
            "correspondence": asdict(detection),
            "pair_gates": asdict(gates),
            "usable_masks": "None (derived ~create_gui_mask per frame)",
            "pair_rule": "all unordered pairs C(n,2) of each frozen calibrated frame set",
        },
        "frame_sets": [
            {
                "set": entry["set"],
                "fixture": entry["fixture"],
                "calibration_artifact": entry["calibration_artifact"],
                "frames": list(entry["frames"]),
                "pairs": len(entry["frames"]) * (len(entry["frames"]) - 1) // 2,
            }
            for entry in sets
        ],
        "frames": frame_records,
        "inputs": inputs,
        "calibration_artifacts": {
            entry["calibration_artifact"]: digest(ROOT / entry["calibration_artifact"]) for entry in sets
        },
        "pairs": pairs,
        "canonical_predictions_digest": canonical_digest(
            [
                {"fixture": p["fixture"], "left": p["left"], "right": p["right"], "joint": p["prediction"]}
                for p in pairs
            ]
        ),
        "candidates_npz": {
            "path": str(npz_path.relative_to(ROOT)),
            "sha256": digest(npz_path),
            "arrays": {key: list(value.shape) for key, value in sorted(arrays.items())},
        },
        "ds15_cross_check": compare_with_ds15(pairs),
        "extraction_seconds_total": time.perf_counter() - started_all,
        "recipe": " ".join(sys.argv),
        "note": (
            "No cameras.json, scene.json, reference litematic, lattice, coordinate or "
            "asserted geometry was opened while producing this file. Every candidate "
            "and prediction is a deterministic function of pixels, the derived GUI "
            "mask and the image-derived M3 calibration artifact."
        ),
    }
    json_path = out_dir / "match_candidates.json"
    payload["_path"] = str(json_path.relative_to(ROOT))
    json_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    payload["_digest"] = digest(json_path)
    return payload


def compare_with_ds15(pairs: list[dict]) -> dict:
    """Hard reproduction check: this run must match the frozen DS-15 predictions."""
    if not DS15_ESTIMATES.exists():
        return {"available": False}
    frozen = json.loads(DS15_ESTIMATES.read_text(encoding="utf-8"))
    known = {(r["fixture"], r["left"], r["right"]): r["joint"] for r in frozen["pairs"]}
    fields = ("status", "reason", "support", "candidate_edges", "right_yaw_class")
    mismatches: list[dict] = []
    for pair in pairs:
        reference = known.get((pair["fixture"], pair["left"], pair["right"]))
        if reference is None:
            mismatches.append({"pair": [pair["fixture"], pair["left"], pair["right"]], "missing": True})
            continue
        for field in fields:
            if pair["prediction"][field] != reference[field]:
                mismatches.append(
                    {
                        "pair": [pair["fixture"], pair["left"], pair["right"]],
                        "field": field,
                        "frozen_ds15": reference[field],
                        "this_run": pair["prediction"][field],
                    }
                )
    return {
        "available": True,
        "path": str(DS15_ESTIMATES.relative_to(ROOT)),
        "sha256": digest(DS15_ESTIMATES),
        "pairs_compared": len(pairs),
        "fields": list(fields),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def load_frozen(out_dir: Path) -> dict:
    path = out_dir / "match_candidates.json"
    if not path.exists():
        raise SystemExit(f"no frozen candidates to attribute: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    recorded = frozen["candidates_npz"]["sha256"]
    actual = digest(ROOT / frozen["candidates_npz"]["path"])
    if actual != recorded:
        raise SystemExit(
            f"frozen candidate arrays were modified: {frozen['candidates_npz']['path']} "
            f"sha256 {actual} != recorded {recorded}"
        )
    cross_check = frozen.get("ds15_cross_check", {})
    if cross_check.get("available") and cross_check.get("mismatch_count"):
        raise SystemExit(
            "blind extraction did not reproduce the frozen DS-15 predictions; "
            "refusing to open any reference manifest"
        )
    frozen["_path"] = str(path.relative_to(ROOT))
    frozen["_digest"] = digest(path)
    return frozen


# --------------------------------------------------------------------------- #
# Phase 2: reference render, keypoint ownership, oracle labels.
# --------------------------------------------------------------------------- #
def _scaled_unproject(col: int, row: int, depth: float, camera, scale: float, width: int, height: int):
    """World point of a scaled render pixel at its camera depth (uniform scale)."""
    calibration = intrinsics_for(camera, width, height)
    focal_x, focal_y = calibration.focal_length_pixels
    centre_x, centre_y = calibration.principal_point_pixels
    view = np.asarray(
        (
            (col - centre_x * scale) / (focal_x * scale) * depth,
            -(row - centre_y * scale) / (focal_y * scale) * depth,
            depth,
        ),
        dtype=np.float64,
    )
    return camera_rotation_matrix(camera).T @ view + camera_eye(camera)


def label_keypoints(points, result, index, camera, scale, usable_full, width, height) -> dict:
    """Map every keypoint to reference block/face/world point under one camera."""
    rows, columns = result.ownership.shape
    count = len(points)
    hit = np.zeros(count, dtype=bool)
    block_code = np.full(count, -1, dtype=np.int64)
    slot = np.full(count, -1, dtype=np.int64)
    normal = np.zeros((count, 3), dtype=np.float64)
    world = np.full((count, 3), np.nan, dtype=np.float64)
    if count == 0:
        return {
            "hit": hit,
            "block_code": block_code,
            "slot": slot,
            "normal": normal,
            "world": world,
            "alignment_px": np.empty(0, dtype=np.float64),
        }
    columns_index = np.clip(np.rint(points[:, 0] * scale).astype(np.int64), 0, columns - 1)
    rows_index = np.clip(np.rint(points[:, 1] * scale).astype(np.int64), 0, rows - 1)
    owner = result.ownership[rows_index, columns_index]
    depth = np.asarray(result.depth[rows_index, columns_index], dtype=np.float64)
    pixel_x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    pixel_y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    hit = (owner >= 0) & np.isfinite(depth) & usable_full[pixel_y, pixel_x]
    blocks: dict[tuple[int, int, int], int] = {}
    for position in np.flatnonzero(hit):
        coordinate, face_slot = index[int(owner[position])]
        code = blocks.get(coordinate)
        if code is None:
            code = len(blocks)
            blocks[coordinate] = code
        block_code[position] = code
        slot[position] = FACE_SLOTS_INDEX[face_slot]
        normal[position] = FACE_BASIS[face_slot][0]
        world[position] = _scaled_unproject(
            int(columns_index[position]),
            int(rows_index[position]),
            float(depth[position]),
            camera,
            scale,
            width,
            height,
        )
    alignment_px = _alignment_errors(points[hit], world[hit], camera, width, height)
    return {
        "hit": hit,
        "block_code": block_code,
        "slot": slot,
        "normal": normal,
        "world": world,
        "alignment_px": alignment_px,
    }


def _alignment_errors(points, world, camera, width, height) -> np.ndarray:
    """Re-projection error of each world hit point through the same camera."""
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    projected = project_points([tuple(value) for value in world], camera, width=width, height=height)
    errors = []
    for (pixel_x, pixel_y), point in zip(points, projected):
        if point is None:
            errors.append(np.inf)
        else:
            errors.append(float(np.hypot(point[0] - pixel_x, point[1] - pixel_y)))
    return np.asarray(errors, dtype=np.float64)


def _truth_edges(left: dict, right: dict, tolerance: float, normal_cosine: float) -> np.ndarray:
    """All geometrically consistent keypoint pairs, via a tolerance-sized hash."""
    left_index = np.flatnonzero(left["hit"])
    right_index = np.flatnonzero(right["hit"])
    if len(left_index) == 0 or len(right_index) == 0:
        return np.empty((0, 2), dtype=np.int64)
    left_world, right_world = left["world"][left_index], right["world"][right_index]
    left_normal, right_normal = left["normal"][left_index], right["normal"][right_index]
    cell = max(float(tolerance), 1e-6)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    right_cells = np.floor(right_world / cell).astype(np.int64)
    for position in range(len(right_cells)):
        key = (
            int(right_cells[position, 0]),
            int(right_cells[position, 1]),
            int(right_cells[position, 2]),
        )
        buckets.setdefault(key, []).append(position)
    found: set[tuple[int, int]] = set()
    for position in range(len(left_world)):
        base = np.floor(left_world[position] / cell).astype(np.int64)
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    for other in buckets.get(
                        (int(base[0]) + delta_x, int(base[1]) + delta_y, int(base[2]) + delta_z), ()
                    ):
                        if float(np.dot(left_normal[position], right_normal[other])) < normal_cosine:
                            continue
                        if float(np.linalg.norm(left_world[position] - right_world[other])) <= tolerance:
                            found.add((int(left_index[position]), int(right_index[other])))
    return np.asarray(sorted(found), dtype=np.int64).reshape(-1, 2)


def _edge_labels(left: dict, right: dict, edges: np.ndarray) -> dict:
    """Strict/coordinate/geometric labels for the given keypoint edge set."""
    count = len(edges)
    if count == 0:
        return {
            "hit": np.zeros(0, dtype=bool),
            "distance": np.empty(0, dtype=np.float64),
            "normal_dot": np.empty(0, dtype=np.float64),
            "geometric": np.zeros(0, dtype=bool),
            "strict": np.zeros(0, dtype=bool),
            "coordinate": np.zeros(0, dtype=bool),
        }
    left_i, right_j = edges[:, 0], edges[:, 1]
    hit = left["hit"][left_i] & right["hit"][right_j]
    distance = np.linalg.norm(left["world"][left_i] - right["world"][right_j], axis=1)
    normal_dot = np.einsum("ij,ij->i", left["normal"][left_i], right["normal"][right_j])
    same_block = left["block_code"][left_i] == right["block_code"][right_j]
    same_slot = left["slot"][left_i] == right["slot"][right_j]
    near = hit & (distance <= WORLD_HIT_DISTANCE_BLOCKS)
    return {
        "hit": hit,
        "distance": distance,
        "normal_dot": normal_dot,
        "geometric": near & (normal_dot >= FACE_NORMAL_COSINE_MIN),
        "strict": near & same_block & same_slot,
        "coordinate": near & same_block,
    }


def _camera_estimate(model, width: int, height: int) -> CameraEstimate:
    return CameraEstimate(
        intrinsics=intrinsics_for(model, width, height),
        pose=CameraPose(
            position=model.position,
            yaw_degrees=model.yaw_degrees,
            pitch_degrees=model.pitch_degrees,
            roll_degrees=model.roll_degrees,
        ),
    )


def _pixel_rays(points: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    rays = np.column_stack(
        (
            (points[:, 0] - centre_x) / focal_x,
            -(points[:, 1] - centre_y) / focal_y,
            np.ones(len(points), dtype=np.float64),
        )
    )
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _gt_epipolar_and_cheirality(points_left, points_right, truth_left, truth_right, left_model, right_model):
    """Necessary conditions under the manifest geometry, per candidate edge."""
    ray_left_camera = _pixel_rays(points_left, truth_left.intrinsics)
    ray_right_camera = _pixel_rays(points_right, truth_right.intrinsics)
    rotation_left = camera_rotation_matrix(left_model)
    rotation_right = camera_rotation_matrix(right_model)
    true_rotation = rotation_right @ rotation_left.T
    world_left = ray_left_camera @ rotation_left
    world_right = ray_right_camera @ rotation_right
    baseline = np.asarray(right_model.position, dtype=np.float64) - np.asarray(
        left_model.position, dtype=np.float64
    )
    length = float(np.linalg.norm(baseline))
    baseline_unit = baseline / length if length > 1e-9 else baseline
    t_true = rotation_right @ (-baseline_unit)
    rows = np.cross(world_left @ true_rotation.T, world_right)
    norms = np.linalg.norm(rows, axis=1)
    projections = np.abs(rows @ t_true)
    ratios = np.ones(len(rows), dtype=np.float64)
    safe = norms > 1e-12
    ratios[safe] = np.clip(projections[safe] / norms[safe], 0.0, 1.0)
    epipolar_degrees = np.degrees(np.arcsin(ratios))
    offset = camera_eye(left_model) - camera_eye(right_model)
    cosine = np.clip(np.sum(world_left * world_right, axis=1), -1.0, 1.0)
    denominator = 1.0 - cosine * cosine
    safe = denominator > 1e-12
    first = world_left @ offset
    second = world_right @ offset
    lambda_left = np.zeros(len(rows), dtype=np.float64)
    mu_right = np.zeros(len(rows), dtype=np.float64)
    lambda_left[safe] = (cosine[safe] * second[safe] - first[safe]) / denominator[safe]
    mu_right[safe] = (second[safe] - cosine[safe] * first[safe]) / denominator[safe]
    return epipolar_degrees, (lambda_left > 0.0) & (mu_right > 0.0)


def _bins(points: np.ndarray, width: int, height: int, grid=SPATIAL_GRID) -> set:
    if len(points) == 0:
        return set()
    grid_x, grid_y = grid
    column = np.clip((points[:, 0] / max(width, 1) * grid_x).astype(int), 0, grid_x - 1)
    row = np.clip((points[:, 1] / max(height, 1) * grid_y).astype(int), 0, grid_y - 1)
    return set(zip(column.tolist(), row.tolist()))


def _descriptor_stats(distances: np.ndarray) -> dict:
    if len(distances) == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(len(distances)),
        "min": float(np.min(distances)),
        "median": float(np.median(distances)),
        "mean": float(np.mean(distances)),
        "max": float(np.max(distances)),
    }


def _feature_view(record: dict, points: np.ndarray, descriptors: np.ndarray) -> FrameFeatures:
    return FrameFeatures(
        frame_id=record["frame"],
        points=np.asarray(points, dtype=np.float64),
        descriptors=np.asarray(descriptors, dtype=np.float32),
        width=int(record["width"]),
        height=int(record["height"]),
        usable_pixels=int(record["usable_pixels"]),
        sizes=None,
        boundary_dropped=int(record["boundary_dropped"]),
    )


def _matches_from_edges(left_points, right_points, edges: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.empty((0, 4), dtype=np.float64)
    left_i, right_j = edges[:, 0], edges[:, 1]
    return np.column_stack(
        (
            left_points[left_i, 0],
            left_points[left_i, 1],
            right_points[right_j, 0],
            right_points[right_j, 1],
        )
    )


def _evaluate_pose(pose, left_model, right_model, expected_class) -> dict:
    record = {
        "solved": bool(pose.solved),
        "reason": pose.reason,
        "right_yaw_class": pose.right_yaw_class,
        "class_equals_expected": bool(pose.solved and pose.right_yaw_class == expected_class),
        "inliers": int(pose.inliers),
        "inlier_fraction": float(pose.inlier_fraction),
        "median_rho_degrees": pose.median_rho_degrees,
        "parallax_degrees": pose.parallax_degrees,
        "cheirality_fraction": pose.cheirality_fraction,
        "rotation_error_degrees": None,
        "within_5deg": False,
        "baseline_error_degrees_gauge_aligned": None,
        "baseline_error_degrees_signless": None,
        "gauge_index": None,
    }
    if pose.solved and pose.relative_rotation is not None:
        true_rotation = camera_rotation_matrix(right_model) @ camera_rotation_matrix(left_model).T
        record["rotation_error_degrees"] = angle_between(np.asarray(pose.relative_rotation), true_rotation)
        record["within_5deg"] = bool(record["rotation_error_degrees"] <= 5.0)
        if pose.baseline_direction is not None:
            baseline_true = np.asarray(right_model.position, dtype=np.float64) - np.asarray(
                left_model.position, dtype=np.float64
            )
            length = float(np.linalg.norm(baseline_true))
            if length > 1e-9:
                baseline_true = baseline_true / length
                baseline_got = np.asarray(pose.baseline_direction, dtype=np.float64)
                baseline_got = baseline_got / max(float(np.linalg.norm(baseline_got)), 1e-12)
                errors = {}
                for step in _GAUGE_ROTATIONS:
                    angle = np.radians(float(step))
                    rotation = np.asarray(
                        (
                            (np.cos(angle), 0.0, np.sin(angle)),
                            (0.0, 1.0, 0.0),
                            (-np.sin(angle), 0.0, np.cos(angle)),
                        )
                    )
                    candidate = rotation @ baseline_true
                    errors[step] = float(
                        np.degrees(np.arccos(float(np.clip(np.dot(candidate, baseline_got), -1.0, 1.0))))
                    )
                best = min(errors, key=lambda step: (errors[step], step))
                record["gauge_index"] = best // 90
                record["baseline_error_degrees_gauge_aligned"] = errors[best]
                cosine = float(np.clip(np.dot(baseline_true, baseline_got), -1.0, 1.0))
                record["baseline_error_degrees_signless"] = float(np.degrees(np.arccos(abs(cosine))))
    return record


def _comparison_cells(
    left_points,
    right_points,
    candidate_edges,
    truth_edges,
    current_left,
    current_right,
    reference_left,
    reference_right,
    left_model,
    right_model,
    reference_class,
    left_name,
    right_name,
) -> dict:
    cells: dict[str, dict] = {}
    minimum = PairGates().min_correspondences
    for edge_name, edges in (("candidates", candidate_edges), ("oracle", truth_edges)):
        matches = _matches_from_edges(left_points, right_points, edges)
        for camera_name, (left_camera, right_camera, expected) in (
            ("current", (current_left, current_right, reference_class)),
            ("reference", (reference_left, reference_right, 0)),
        ):
            key = f"{edge_name}_{camera_name}"
            if len(matches) < minimum:
                cells[key] = {
                    "matches": int(len(matches)),
                    "solved": False,
                    "reason": "insufficient_correspondences",
                    "right_yaw_class": None,
                    "class_equals_expected": False,
                    "inliers": 0,
                    "inlier_fraction": 0.0,
                    "median_rho_degrees": None,
                    "parallax_degrees": None,
                    "cheirality_fraction": None,
                    "rotation_error_degrees": None,
                    "within_5deg": False,
                    "baseline_error_degrees_gauge_aligned": None,
                    "baseline_error_degrees_signless": None,
                    "gauge_index": None,
                }
                continue
            pose = solve_pair(
                left_camera,
                right_camera,
                matches,
                left_id=left_name,
                right_id=right_name,
                gates=PairGates(),
            )
            record = _evaluate_pose(pose, left_model, right_model, expected)
            record["matches"] = int(len(matches))
            cells[key] = record
    return cells


def _selected_edge_set(left_points, right_points, selected: np.ndarray) -> set[tuple[int, int]]:
    """Recover candidate edge indices from frozen DS-14 selected pixel matches."""
    if len(selected) == 0:
        return set()
    left_lookup = {(float(point[0]), float(point[1])): index for index, point in enumerate(left_points)}
    right_lookup = {(float(point[0]), float(point[1])): index for index, point in enumerate(right_points)}
    found: set[tuple[int, int]] = set()
    for row in selected:
        left_index = left_lookup.get((float(row[0]), float(row[1])))
        right_index = right_lookup.get((float(row[2]), float(row[3])))
        if left_index is not None and right_index is not None:
            found.add((left_index, right_index))
    return found


def _yaw_support(prediction, reference_class, selected_true: int) -> dict:
    """DS-14 hypothesis support beside the true class."""
    return {
        "reference_class": reference_class,
        "ds14_solved_class": prediction.get("right_yaw_class"),
        "ds14_winner_support": prediction.get("support"),
        "ds14_ambiguity_margin": prediction.get("ambiguity_margin"),
        "ds14_support_floor": prediction.get("support_floor"),
        "selected_true": int(selected_true),
        "hypotheses_by_class": [
            {
                "yaw_class": score["yaw_class"],
                "support": score["support"],
                "passed": score["passed"],
                "refusal": score["refusal"],
                "median_selected_residual_degrees": score["median_selected_residual_degrees"],
                "descriptor_cost": score["descriptor_cost"],
                "parallax_degrees": score["parallax_degrees"],
                "cheirality_fraction": score["cheirality_fraction"],
            }
            for score in (prediction.get("hypotheses") or [])
        ],
    }


def _attribute(candidate_stats: dict, cells: dict, reference_class: int) -> dict:
    """Preregistered single-label attribution, priority order as declared."""
    oracle_reference = cells["oracle_reference"]
    oracle_current = cells["oracle_current"]
    candidate_reference = cells["candidates_reference"]
    candidate_current = cells["candidates_current"]
    if candidate_stats["oracle_truths"] < 8:
        label = "insufficient_overlap"
    elif oracle_reference["within_5deg"] and not oracle_current["within_5deg"]:
        label = "camera_calibration"
    elif oracle_reference["within_5deg"] and not candidate_reference["within_5deg"]:
        label = "descriptor_candidate_recall"
    elif (
        candidate_reference["within_5deg"]
        and candidate_stats["reciprocal_true_lost_to_one_to_one"] > 0
        and not candidate_current["within_5deg"]
    ):
        label = "one_to_one_selection"
    elif oracle_reference["within_5deg"] and not candidate_current["within_5deg"]:
        label = "yaw_baseline_scoring"
    elif candidate_stats["oracle_truths"] > 0:
        label = "yaw_baseline_scoring"
    else:
        label = "insufficient_overlap"
    return {"label": label, "reference_class": reference_class}


def run_attribution(out_dir: Path, frozen: dict) -> dict:
    """Phase 2: open manifests, label candidates, run the four comparisons."""
    import os

    provider = VanillaJarTextureProvider()
    provider.load_jar(Path(os.environ["APPDATA"]) / ".minecraft/versions/1.21.5/1.21.5.jar")
    palette = provider.complete_palette()
    colours = face_colours(palette)
    shapes = face_shapes(palette)
    occluding_slots = face_occluding_slots(palette)
    model_faces = face_model_faces(palette)

    arrays = np.load(ROOT / frozen["candidates_npz"]["path"])
    frames = {record["frame"]: record for record in frozen["frames"]}

    # Manifest / reference loading happens strictly after the freeze is digested.
    fixtures: dict[str, dict] = {}
    for pair in frozen["pairs"]:
        fixture = pair["fixture"]
        if fixture in fixtures:
            continue
        folder = ROOT / fixture
        litematic_path = next(folder.glob("*.litematic"))
        manifest_path = folder / "cameras.json"
        scene_path = folder / "scene.json"
        fixtures[fixture] = {
            "folder": fixture,
            "litematic": str(litematic_path.relative_to(ROOT)),
            "litematic_sha256": digest(litematic_path),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": digest(manifest_path),
            "scene": str(scene_path.relative_to(ROOT)),
            "scene_sha256": digest(scene_path),
            "reference": load_litematic(litematic_path),
            "scene_transform": load_scene_transform(scene_path),
            "cameras": load_camera_manifest(manifest_path),
        }

    labels: dict[str, dict] = {}
    alignment: list[dict] = []
    for name, record in frames.items():
        fixture = fixtures[record["fixture"]]
        camera = fixture["cameras"][name]
        width, height = int(record["width"]), int(record["height"])
        started = time.perf_counter()
        result, index = render_result(
            fixture["reference"].blocks,
            fixture["scene_transform"],
            camera,
            width,
            height,
            colours,
            shapes=shapes,
            scale=REFERENCE_RENDER_SCALE,
            occluding_slots=occluding_slots,
            model_faces=model_faces,
        )
        points = np.asarray(arrays[f"feat_points_{record['key']}"], dtype=np.float64)
        usable = ~create_gui_mask(height, width)
        labels[name] = label_keypoints(
            points, result, index, camera, REFERENCE_RENDER_SCALE, usable, width, height
        )
        errors = labels[name]["alignment_px"]
        alignment.append(
            {
                "frame": name,
                "fixture": record["fixture"],
                "render_seconds": time.perf_counter() - started,
                "keypoints": int(len(points)),
                "hits": int(np.count_nonzero(labels[name]["hit"])),
                "hit_fraction": float(np.mean(labels[name]["hit"])) if len(points) else None,
                "alignment_median_px": float(np.median(errors)) if len(errors) else None,
                "alignment_p90_px": float(np.percentile(errors, 90)) if len(errors) else None,
                "alignment_within_2px_fraction": float(np.mean(errors <= 2.0)) if len(errors) else None,
            }
        )

    medians = [row["alignment_median_px"] for row in alignment if row["alignment_median_px"] is not None]
    if not medians or float(np.median(medians)) > ALIGNMENT_MEDIAN_PX_MAX:
        raise SystemExit(
            "blocker: reference renderer ownership does not align with the manifest cameras "
            f"(median re-projection error {None if not medians else float(np.median(medians))} px); "
            "refusing to trust oracle labels"
        )

    rows: list[dict] = []
    for pair in frozen["pairs"]:
        fixture = fixtures[pair["fixture"]]
        left_name, right_name = pair["left"], pair["right"]
        left_record, right_record = frames[left_name], frames[right_name]
        left_points = np.asarray(arrays[f"feat_points_{left_record['key']}"], dtype=np.float64)
        right_points = np.asarray(arrays[f"feat_points_{right_record['key']}"], dtype=np.float64)
        left_descriptors = np.asarray(arrays[f"feat_desc_{left_record['key']}"], dtype=np.float64)
        right_descriptors = np.asarray(arrays[f"feat_desc_{right_record['key']}"], dtype=np.float64)
        candidate_edges = np.asarray(arrays[f"cand_{pair['index']}"], dtype=np.float64).reshape(-1, 3)
        edges = candidate_edges[:, :2].astype(np.int64)
        distances = candidate_edges[:, 2]
        selected = np.asarray(arrays[f"sel_{pair['index']}"], dtype=np.float64).reshape(-1, 4)

        left_est = _estimate_from_record(pair["left_estimate"])
        right_est = _estimate_from_record(pair["right_estimate"])
        left_model, right_model = fixture["cameras"][left_name], fixture["cameras"][right_name]
        width, height = left_record["width"], left_record["height"]
        current_left = to_camera_estimate(left_est)
        current_right = to_camera_estimate(right_est)
        reference_left = _camera_estimate(left_model, width, height)
        reference_right = _camera_estimate(right_model, width, height)
        reference_class, reference_floors = _reference_class(left_est, right_est, left_model, right_model)

        left_label, right_label = labels[left_name], labels[right_name]
        edge_labels = _edge_labels(left_label, right_label, edges)
        truth_edges = _truth_edges(left_label, right_label, WORLD_HIT_DISTANCE_BLOCKS, FACE_NORMAL_COSINE_MIN)

        directed = _directed_topk(
            left_descriptors, right_descriptors, frozen["settings"]["joint_matcher"]["neighbors"]
        )
        directed_set = {
            (int(origin), int(target))
            for origin in range(directed.shape[0])
            for target in directed[origin]
            if target >= 0
        }
        reciprocal_set = {(int(left), int(right)) for left, right in edges}
        selected_set = _selected_edge_set(left_points, right_points, selected)
        truth_set = {(int(left), int(right)) for left, right in truth_edges}

        true_candidate = edge_labels["geometric"]
        descriptor_stats = {
            "true": _descriptor_stats(distances[true_candidate]),
            "false": _descriptor_stats(distances[~true_candidate & edge_labels["hit"]]),
            "unscored": int(np.count_nonzero(~edge_labels["hit"])),
        }
        spatial = {
            "true_bins_left": len(_bins(left_points[edges[true_candidate, 0]], width, height)),
            "true_bins_right": len(_bins(right_points[edges[true_candidate, 1]], width, height)),
            "false_bins_left": len(_bins(left_points[edges[~true_candidate, 0]], width, height)),
            "false_bins_right": len(_bins(right_points[edges[~true_candidate, 1]], width, height)),
        }
        if len(edges):
            epipolar, cheirality = _gt_epipolar_and_cheirality(
                left_points[edges[:, 0]],
                right_points[edges[:, 1]],
                reference_left,
                reference_right,
                left_model,
                right_model,
            )
            permutation = np.random.default_rng(0).permutation(len(edges))
            epipolar_chance, cheirality_chance = _gt_epipolar_and_cheirality(
                left_points[edges[:, 0]],
                right_points[edges[permutation, 1]],
                reference_left,
                reference_right,
                left_model,
                right_model,
            )
        else:
            epipolar = np.empty(0)
            cheirality = np.zeros(0, dtype=bool)
            epipolar_chance = np.empty(0)
            cheirality_chance = np.zeros(0, dtype=bool)
        gt_necessary = (epipolar <= EPIPOLAR_NECESSARY_DEGREES) & cheirality
        gt_necessary_chance = (epipolar_chance <= EPIPOLAR_NECESSARY_DEGREES) & cheirality_chance

        candidate_stats = {
            "candidate_edges": int(len(edges)),
            "hits": int(np.count_nonzero(edge_labels["hit"])),
            "geometric_true": int(np.count_nonzero(edge_labels["geometric"])),
            "strict_true": int(np.count_nonzero(edge_labels["strict"])),
            "coordinate_true": int(np.count_nonzero(edge_labels["coordinate"])),
            "gt_epipolar_and_cheirality_true": int(np.count_nonzero(edge_labels["hit"] & gt_necessary)),
            "gt_epipolar_and_cheirality_chance": int(np.count_nonzero(gt_necessary_chance)),
            "gt_necessary_excess": int(
                np.count_nonzero(edge_labels["hit"] & gt_necessary) - np.count_nonzero(gt_necessary_chance)
            ),
            "geometric_prevalence": (
                float(np.mean(edge_labels["geometric"][edge_labels["hit"]]))
                if np.any(edge_labels["hit"])
                else None
            ),
            "oracle_truths": int(len(truth_edges)),
            "directed_topk_edges": int(len(directed_set)),
            "reciprocal_edges": int(len(reciprocal_set)),
            "selected_matches": int(len(selected)),
            "selected_true": int(len(selected_set & truth_set)),
            "recall_directed_topk": (
                len(directed_set & truth_set) / len(truth_set) if truth_set else None
            ),
            "recall_reciprocal": (len(reciprocal_set & truth_set) / len(truth_set) if truth_set else None),
            "recall_selected": (len(selected_set & truth_set) / len(truth_set) if truth_set else None),
            "selected_true_precision": (
                len(selected_set & truth_set) / len(selected_set) if selected_set else None
            ),
            "reciprocal_true_lost_to_one_to_one": int(len((reciprocal_set & truth_set) - selected_set)),
            "descriptor_distance": descriptor_stats,
            "spatial_bins": spatial,
            "median_gt_epipolar_degrees_true": (
                float(np.median(epipolar[edge_labels["geometric"]]))
                if np.any(edge_labels["geometric"])
                else None
            ),
            "gt_cheirality_fraction_true": (
                float(np.mean(cheirality[edge_labels["geometric"]]))
                if np.any(edge_labels["geometric"])
                else None
            ),
        }

        cells = _comparison_cells(
            left_points,
            right_points,
            edges,
            truth_edges,
            current_left,
            current_right,
            reference_left,
            reference_right,
            left_model,
            right_model,
            reference_class,
            left_name,
            right_name,
        )
        ds14_reference = match_feature_frames(
            _feature_view(left_record, left_points, left_descriptors),
            _feature_view(right_record, right_points, right_descriptors),
            reference_left,
            reference_right,
            matcher=JointMatchSettings(),
            gates=PairGates(),
            ratio_test=frozen["settings"]["correspondence"]["ratio_test"],
        )
        prediction = pair["prediction"]
        rows.append(
            {
                "index": pair["index"],
                "set": pair["set"],
                "fixture": pair["fixture"],
                "left": left_name,
                "right": right_name,
                "reference_yaw_class": reference_class,
                "reference_yaw_class_floors": reference_floors,
                "manifest_left_yaw_degrees": float(left_model.yaw_degrees),
                "manifest_right_yaw_degrees": float(right_model.yaw_degrees),
                "left_estimate": pair["left_estimate"],
                "right_estimate": pair["right_estimate"],
                "ds14_prediction": {
                    "status": prediction["status"],
                    "reason": prediction["reason"],
                    "solved": prediction["solved"],
                    "right_yaw_class": prediction["right_yaw_class"],
                    "support": prediction["support"],
                    "support_floor": prediction["support_floor"],
                    "candidate_edges": prediction["candidate_edges"],
                },
                "ds14_prediction_reference_cameras": {
                    "status": ds14_reference.status,
                    "reason": ds14_reference.reason,
                    "solved": ds14_reference.solved,
                    "right_yaw_class": ds14_reference.right_yaw_class,
                    "support": ds14_reference.support,
                    "support_floor": ds14_reference.support_floor,
                },
                "correct_vs_wrong_yaw_support": _yaw_support(
                    prediction, reference_class, candidate_stats["selected_true"]
                ),
                "candidate_stats": candidate_stats,
                "cells": cells,
                "attribution": _attribute(candidate_stats, cells, reference_class),
            }
        )

    summary = _summarise(rows)
    payload = {
        "schema": "ds16-attribution-1",
        "phase": "reference_assisted_attribution",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source_revision(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
        },
        "frozen_candidates": {
            "json": frozen["_path"],
            "json_sha256": frozen["_digest"],
            "npz": frozen["candidates_npz"]["path"],
            "npz_sha256": frozen["candidates_npz"]["sha256"],
            "canonical_predictions_digest": frozen["canonical_predictions_digest"],
            "ds15_cross_check": frozen["ds15_cross_check"],
        },
        "reference_inputs": {
            fixture: {
                key: entry[key]
                for key in ("litematic", "litematic_sha256", "manifest", "manifest_sha256", "scene", "scene_sha256")
            }
            for fixture, entry in fixtures.items()
        },
        "constants": {
            "reference_render_scale": REFERENCE_RENDER_SCALE,
            "world_hit_distance_blocks": WORLD_HIT_DISTANCE_BLOCKS,
            "face_normal_cosine_min": FACE_NORMAL_COSINE_MIN,
            "face_normal_degrees": 30.0,
            "epipolar_necessary_degrees": EPIPOLAR_NECESSARY_DEGREES,
            "alignment_median_px_max": ALIGNMENT_MEDIAN_PX_MAX,
            "spatial_grid": list(SPATIAL_GRID),
        },
        "alignment": alignment,
        "alignment_check_note": (
            "The alignment block is a renderer round-trip check (scaled unprojection "
            "then full-resolution reprojection through the same manifest camera); it "
            "verifies the ownership/depth convention but is tautological with respect "
            "to camera-vs-photo registration. The non-tautological check is "
            "gt_epipolar_and_cheirality_true against gt_epipolar_and_cheirality_chance "
            "(shuffled right endpoints): a set whose true count does not exceed chance "
            "has no demonstrated manifest-pose agreement with the candidate edges."
        ),
        "summary": summary,
        "pairs": rows,
        "note": (
            "Evaluation only. Reference manifests and litematics were opened only after "
            "match_candidates.json + .npz were frozen and digested, and no extraction or "
            "solver parameter was changed after this comparison. All labels here are "
            "reference-assisted and must never enter a production path."
        ),
        "recipe": " ".join(sys.argv),
    }
    json_path = out_dir / "match_attribution.json"
    payload["_path"] = str(json_path.relative_to(ROOT))
    json_path.write_text(json.dumps(payload, indent=2, default=plain), encoding="utf-8")
    _write_csv(out_dir / "match_attribution.csv", rows)
    payload["_digest"] = digest(json_path)
    return payload


def _aggregate_recall(rows: list[dict], field: str) -> float | None:
    total = 0.0
    denominator = 0
    for row in rows:
        truths = row["candidate_stats"]["oracle_truths"]
        total += (row["candidate_stats"][field] or 0.0) * truths
        denominator += truths
    return total / denominator if denominator else None


def _summarise(rows: list[dict]) -> dict:
    cells: dict[str, dict] = {}
    for key in ("candidates_current", "candidates_reference", "oracle_current", "oracle_reference"):
        solved = [row["cells"][key] for row in rows if row["cells"][key]["solved"]]
        correct = [row for row in solved if row["within_5deg"]]
        rotation = [row["rotation_error_degrees"] for row in solved if row["rotation_error_degrees"] is not None]
        baseline = [
            row["baseline_error_degrees_gauge_aligned"]
            for row in solved
            if row["baseline_error_degrees_gauge_aligned"] is not None
        ]
        cells[key] = {
            "pairs": len(rows),
            "evaluated": sum(1 for row in rows if row["cells"][key]["matches"] >= 8),
            "solved": len(solved),
            "within_5deg": len(correct),
            "precision_within_5deg": len(correct) / len(solved) if solved else None,
            "median_rotation_error_degrees": float(np.median(rotation)) if rotation else None,
            "median_baseline_error_degrees_gauge_aligned": float(np.median(baseline)) if baseline else None,
            "reason_counts": dict(
                Counter(row["cells"][key]["reason"] for row in rows if not row["cells"][key]["solved"])
            ),
        }
    candidate_stats = {
        key: sum(row["candidate_stats"][key] for row in rows)
        for key in (
            "candidate_edges",
            "hits",
            "geometric_true",
            "strict_true",
            "coordinate_true",
            "gt_epipolar_and_cheirality_true",
            "gt_epipolar_and_cheirality_chance",
            "gt_necessary_excess",
            "oracle_truths",
            "selected_matches",
            "selected_true",
            "reciprocal_true_lost_to_one_to_one",
        )
    }
    candidate_stats["geometric_prevalence"] = (
        candidate_stats["geometric_true"] / candidate_stats["hits"] if candidate_stats["hits"] else None
    )
    candidate_stats["recall_directed_topk"] = _aggregate_recall(rows, "recall_directed_topk")
    candidate_stats["recall_reciprocal"] = _aggregate_recall(rows, "recall_reciprocal")
    candidate_stats["recall_selected"] = _aggregate_recall(rows, "recall_selected")
    candidate_stats["selected_true_precision"] = (
        candidate_stats["selected_true"] / candidate_stats["selected_matches"]
        if candidate_stats["selected_matches"]
        else None
    )
    per_set: dict[str, dict] = {}
    for label in sorted({row["set"] for row in rows}):
        set_rows = [row for row in rows if row["set"] == label]
        per_set[label] = {
            "pairs": len(set_rows),
            "candidate_edges": sum(row["candidate_stats"]["candidate_edges"] for row in set_rows),
            "geometric_true": sum(row["candidate_stats"]["geometric_true"] for row in set_rows),
            "oracle_truths": sum(row["candidate_stats"]["oracle_truths"] for row in set_rows),
            "ds14_solved": sum(1 for row in set_rows if row["ds14_prediction"]["solved"]),
            "ds14_class_correct": sum(
                1
                for row in set_rows
                if row["ds14_prediction"]["solved"]
                and row["ds14_prediction"]["right_yaw_class"] == row["reference_yaw_class"]
            ),
            "candidates_current_within_5deg": sum(
                1 for row in set_rows if row["cells"]["candidates_current"]["within_5deg"]
            ),
            "oracle_reference_within_5deg": sum(
                1 for row in set_rows if row["cells"]["oracle_reference"]["within_5deg"]
            ),
            "attribution": dict(Counter(row["attribution"]["label"] for row in set_rows)),
        }
    return {
        "pairs": len(rows),
        "cells": cells,
        "candidate_stats": candidate_stats,
        "per_set": per_set,
        "attribution_labels": dict(Counter(row["attribution"]["label"] for row in rows)),
        "ds14_end_to_end": {
            "solved": sum(1 for row in rows if row["ds14_prediction"]["solved"]),
            "class_correct": sum(
                1
                for row in rows
                if row["ds14_prediction"]["solved"]
                and row["ds14_prediction"]["right_yaw_class"] == row["reference_yaw_class"]
            ),
            "reference_camera_solved": sum(
                1 for row in rows if row["ds14_prediction_reference_cameras"]["solved"]
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "set",
        "fixture",
        "left",
        "right",
        "reference_yaw_class",
        "ds14_status",
        "ds14_solved",
        "ds14_class",
        "ds14_support",
        "candidate_edges",
        "candidate_hits",
        "geometric_true",
        "strict_true",
        "coordinate_true",
        "gt_necessary_true",
        "gt_necessary_chance",
        "gt_necessary_excess",
        "oracle_truths",
        "recall_directed_topk",
        "recall_reciprocal",
        "recall_selected",
        "selected_matches",
        "selected_true",
        "selected_true_precision",
        "one_to_one_lost_true",
        "candidates_current_solved",
        "candidates_current_within_5deg",
        "candidates_current_rotation_error_degrees",
        "candidates_reference_within_5deg",
        "oracle_current_within_5deg",
        "oracle_reference_solved",
        "oracle_reference_within_5deg",
        "oracle_reference_rotation_error_degrees",
        "oracle_reference_baseline_error_degrees_gauge_aligned",
        "attribution",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            stats = row["candidate_stats"]
            cells = row["cells"]
            writer.writerow(
                {
                    "set": row["set"],
                    "fixture": row["fixture"],
                    "left": row["left"],
                    "right": row["right"],
                    "reference_yaw_class": row["reference_yaw_class"],
                    "ds14_status": row["ds14_prediction"]["status"],
                    "ds14_solved": row["ds14_prediction"]["solved"],
                    "ds14_class": row["ds14_prediction"]["right_yaw_class"],
                    "ds14_support": row["ds14_prediction"]["support"],
                    "candidate_edges": stats["candidate_edges"],
                    "candidate_hits": stats["hits"],
                    "geometric_true": stats["geometric_true"],
                    "strict_true": stats["strict_true"],
                    "coordinate_true": stats["coordinate_true"],
                    "gt_necessary_true": stats["gt_epipolar_and_cheirality_true"],
                    "gt_necessary_chance": stats["gt_epipolar_and_cheirality_chance"],
                    "gt_necessary_excess": stats["gt_necessary_excess"],
                    "oracle_truths": stats["oracle_truths"],
                    "recall_directed_topk": stats["recall_directed_topk"],
                    "recall_reciprocal": stats["recall_reciprocal"],
                    "recall_selected": stats["recall_selected"],
                    "selected_matches": stats["selected_matches"],
                    "selected_true": stats["selected_true"],
                    "selected_true_precision": stats["selected_true_precision"],
                    "one_to_one_lost_true": stats["reciprocal_true_lost_to_one_to_one"],
                    "candidates_current_solved": cells["candidates_current"]["solved"],
                    "candidates_current_within_5deg": cells["candidates_current"]["within_5deg"],
                    "candidates_current_rotation_error_degrees": cells["candidates_current"][
                        "rotation_error_degrees"
                    ],
                    "candidates_reference_within_5deg": cells["candidates_reference"]["within_5deg"],
                    "oracle_current_within_5deg": cells["oracle_current"]["within_5deg"],
                    "oracle_reference_solved": cells["oracle_reference"]["solved"],
                    "oracle_reference_within_5deg": cells["oracle_reference"]["within_5deg"],
                    "oracle_reference_rotation_error_degrees": cells["oracle_reference"][
                        "rotation_error_degrees"
                    ],
                    "oracle_reference_baseline_error_degrees_gauge_aligned": cells["oracle_reference"][
                        "baseline_error_degrees_gauge_aligned"
                    ],
                    "attribution": row["attribution"]["label"],
                }
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--freeze-only", action="store_true", help="stop after the blind phase")
    parser.add_argument(
        "--attribute-only",
        action="store_true",
        help="attribute the existing frozen candidates without re-extracting",
    )
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    if args.freeze_only and args.attribute_only:
        parser.error("--freeze-only and --attribute-only are mutually exclusive")

    if args.attribute_only:
        frozen = load_frozen(args.out_dir)
    else:
        frozen = extract_candidates(args.out_dir)
        print(
            f"[freeze] {len(frozen['pairs'])} pairs; candidates "
            f"{sum(pair['graph']['candidate_edges'] for pair in frozen['pairs'])}; "
            f"ds14 solved {sum(1 for pair in frozen['pairs'] if pair['prediction']['solved'])}; "
            f"canonical digest {frozen['canonical_predictions_digest']}; "
            f"frozen {frozen['_path']} sha256 {frozen['_digest']}"
        )
        print(f"[freeze] ds15 cross-check mismatches {frozen['ds15_cross_check'].get('mismatch_count')}")
        if args.freeze_only:
            return 0

    attribution = run_attribution(args.out_dir, frozen)
    summary = attribution["summary"]
    print(
        f"[attribute] pairs {summary['pairs']}; attribution {summary['attribution_labels']}; "
        f"cells within_5deg "
        f"{ {key: value['within_5deg'] for key, value in summary['cells'].items()} }"
    )
    print(f"[attribute] wrote {attribution['_path']} sha256 {attribution['_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
