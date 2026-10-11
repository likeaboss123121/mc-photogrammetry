"""Bounded posthoc static-block scoring for the pinned Canal Town 05/06 pair.

This module consumes only already-frozen feature/mode rows and a hash-pinned
context export. It never decodes screenshots or calls production reconstruction.
The result is conditional on the explicit opaque full-cube allowlist below; it
is not a screenshot-accuracy or Minecraft render-equivalence claim.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.reconstruction.camera import CameraIntrinsics, CameraModel, camera_rotation_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROOT = Path("exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox")
CONTEXT_PATH = Path("exports/evaluation/context-gt-f2b-canal-town-2026-09-12-attempt2/context-ground-truth.json")
PAIR_PATH = Path("docs/reports/SOL-IMG-FACTORY-CAMERA-PAIR-20261005.json")
R2_PATH = Path("exports/evaluation/frozen-image-run-r2-atom-a1-canal-town-2026-09-12/freeze.json")
R2_MASKS_PATH = Path("exports/evaluation/frozen-image-run-r2-atom-a1-canal-town-2026-09-12/freeze.json.68277b112b4b7ad755878501c8c04617119477f5ceff76be9dff2c0403bc74db.masks.bin")
TARGET_PATH = SCENE_ROOT / "ground_truth.json"
CAPTURE_PATH = SCENE_ROOT / "capture.json"

PINNED_SHA256 = {
    "context": "180e49099e3e26ee957671dd94c3410a9fbe1124802d4f8976d6a4eac3bf2f8d",
    "pair": "4e0c810a1a4f1deffad79beddbe951391d70e7b6db308d0b3be23e76fcd1133b",
    "capture": "c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6",
    "target": "36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a",
    "r2": "3c4e8e1b63f73748bde1f072d97dbfb4276ebc90c957889123760edcc6a1c90f",
    "r2_masks": "68277b112b4b7ad755878501c8c04617119477f5ceff76be9dff2c0403bc74db",
    "png05": "6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b",
    "png06": "53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923",
    "camera_source": "0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680",
}
PINNED_BYTES = {
    "context": 28_685_500,
    "pair": 1_123_826,
    "capture": 28_051,
    "target": 962_450,
    "r2": 25_083_593,
    "r2_masks": 127_087_308,
    "png05": 3_896_541,
    "png06": 3_258_563,
}

MAX_CONTEXT_VOLUME = 250_000
MAX_FEATURES_PER_FRAME = 4_000
MAX_GRAPH_EDGES = 8_192
MAX_MODES = 12
MAX_DDA_STEPS = 512
MAX_DEPTH = 512.0
SURFACE_TOLERANCE = 1.0 / 16.0
MAX_CANONICAL_OUTPUT_BYTES = 16 * 1024 * 1024

# Deliberately narrow, name-and-empty-properties policy for static opaque unit
# cubes. Any other state is unsupported, including slabs, leaves, panes, fluids,
# stairs, fences, block entities, and unknown models. This is a declared proxy
# policy, not a claim that the live client renderer was independently verified.
OPAQUE_FULL_CUBE_NAMES = frozenset({
    "minecraft:stone", "minecraft:dirt", "minecraft:bricks",
    "minecraft:white_concrete", "minecraft:granite", "minecraft:polished_granite",
    "minecraft:gravel", "minecraft:cobbled_deepslate", "minecraft:stone_bricks",
    "minecraft:terracotta", "minecraft:coal_ore", "minecraft:smooth_quartz",
    "minecraft:sandstone", "minecraft:polished_andesite", "minecraft:dark_prismarine",
    "minecraft:copper_ore", "minecraft:dripstone_block", "minecraft:diamond_block",
    "minecraft:lapis_ore", "minecraft:polished_blackstone_bricks",
    "minecraft:cracked_polished_blackstone_bricks", "minecraft:cobblestone",
})


class StaticPairError(ValueError):
    """Pinned input or bounded static-pair contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        return raw, json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticPairError(f"invalid JSON input: {path.as_posix()}") from exc


def _verify_pin(name: str, path: Path, expected: str) -> str:
    if not path.is_file():
        raise StaticPairError(f"missing pinned input {name}")
    expected_size = PINNED_BYTES.get(name)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise StaticPairError(f"pinned input byte size mismatch: {name}")
    actual = _sha256(path)
    if actual != expected:
        raise StaticPairError(f"pinned input hash mismatch: {name}")
    return actual


def _xyz3(value: Any, label: str) -> tuple[float, float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(part) not in (int, float) or not math.isfinite(float(part)) for part in value)):
        raise StaticPairError(f"invalid {label}")
    return tuple(float(part) for part in value)  # type: ignore[return-value]


def _region(document: Mapping[str, Any]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    region = document.get("region")
    if not isinstance(region, Mapping) or set(region) != {"min", "max"}:
        raise StaticPairError("invalid context region")
    low, high = region["min"], region["max"]
    if (not isinstance(low, list) or not isinstance(high, list) or len(low) != 3 or len(high) != 3
            or any(type(v) is not int for v in low + high)):
        raise StaticPairError("invalid context region coordinates")
    lo, hi = tuple(low), tuple(high)
    if any(hi[a] < lo[a] for a in range(3)):
        raise StaticPairError("inverted context region")
    volume = math.prod(hi[a] - lo[a] + 1 for a in range(3))
    if volume <= 0 or volume > MAX_CONTEXT_VOLUME:
        raise StaticPairError("context volume cap exceeded")
    return lo, hi


def _load_inputs(root: Path = REPO_ROOT) -> dict[str, Any]:
    paths = {
        "context": root / CONTEXT_PATH,
        "pair": root / PAIR_PATH,
        "capture": root / CAPTURE_PATH,
        "target": root / TARGET_PATH,
        "r2": root / R2_PATH,
        "r2_masks": root / R2_MASKS_PATH,
        "camera_source": root / "src/mc_photogrammetry/reconstruction/camera.py",
    }
    input_hashes = {name: _verify_pin(name, path, PINNED_SHA256[name]) for name, path in paths.items()}
    png_paths = {
        "image-0000": root / SCENE_ROOT / "captures/view-canal-town-architectural-component-bbox-05.png",
        "image-0001": root / SCENE_ROOT / "captures/view-canal-town-architectural-component-bbox-06.png",
    }
    for frame, path in png_paths.items():
        name = "png05" if frame == "image-0000" else "png06"
        if path.stat().st_size != PINNED_BYTES[name]:
            raise StaticPairError(f"pinned input byte size mismatch: {name}")
    png_hashes = {frame: _sha256(path) for frame, path in png_paths.items()}
    if png_hashes != {"image-0000": PINNED_SHA256["png05"], "image-0001": PINNED_SHA256["png06"]}:
        raise StaticPairError("selected PNG byte hash differs")

    context = json.loads((root / CONTEXT_PATH).read_bytes())
    pair = json.loads((root / PAIR_PATH).read_bytes())
    capture = json.loads((root / CAPTURE_PATH).read_bytes())
    target = json.loads((root / TARGET_PATH).read_bytes())
    preserved_pin_checks: dict[str, dict[str, str]] = {}
    for pin_group in ("asset_pins", "source_pins", "prior_pins"):
        pin_rows = pair.get(pin_group)
        if not isinstance(pin_rows, Mapping):
            raise StaticPairError(f"native diagnostic lacks {pin_group}")
        verified: dict[str, str] = {}
        for relative, expected in sorted(pin_rows.items()):
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise StaticPairError(f"malformed {pin_group} entry")
            pinned_path = root / Path(relative)
            if not pinned_path.is_file() or _sha256(pinned_path) != expected:
                raise StaticPairError(f"previous {pin_group} pin changed: {relative}")
            verified[relative] = expected
        preserved_pin_checks[pin_group] = verified
    if (context.get("schema_version") != 1
            or context.get("artifact_type") != "mc-photogrammetry-context-ground-truth"
            or context.get("capture_metadata_sha256") != PINNED_SHA256["capture"]
            or context.get("target_ground_truth_sha256") != PINNED_SHA256["target"]
            or context.get("scene_id") != capture.get("scene_id")
            or context.get("scene_id") != target.get("scene_id")
            or context.get("world_identity") != capture.get("world_identity")
            or context.get("world_identity") != target.get("world_identity")):
        raise StaticPairError("context/capture/target binding mismatch")

    low, high = _region(context)
    rows = context.get("blocks")
    if not isinstance(rows, list) or len(rows) != math.prod(high[a] - low[a] + 1 for a in range(3)):
        raise StaticPairError("context grid count is incomplete")
    occupancy: dict[tuple[int, int, int], tuple[str, tuple[tuple[str, str], ...]]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"pos", "name", "properties"}:
            raise StaticPairError("malformed context block row")
        pos = row["pos"]
        props = row["properties"]
        if (not isinstance(pos, list) or len(pos) != 3 or any(type(c) is not int for c in pos)
                or not isinstance(row["name"], str) or not isinstance(props, Mapping)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in props.items())):
            raise StaticPairError("invalid context block row fields")
        xyz = tuple(pos)
        if any(xyz[a] < low[a] or xyz[a] > high[a] for a in range(3)) or xyz in occupancy:
            raise StaticPairError("context row outside region or duplicated")
        occupancy[xyz] = (row["name"], tuple(sorted(props.items())))
    if len(occupancy) != len(rows):
        raise StaticPairError("context grid uniqueness failure")
    # Dense grid check, not merely row count: gaps must never become air.
    if len(occupancy) != math.prod(high[a] - low[a] + 1 for a in range(3)):
        raise StaticPairError("context contains a grid gap")
    for record in target.get("blocks", []):
        if not isinstance(record, Mapping):
            raise StaticPairError("malformed target row")
        pos = tuple(record.get("pos", ()))
        expected = (record.get("name"), tuple(sorted((record.get("properties") or {}).items())))
        if occupancy.get(pos) != expected:
            raise StaticPairError("context does not preserve target state")

    images = context.get("ordered_images")
    screenshots = capture.get("screenshots")
    if not isinstance(images, list) or not isinstance(screenshots, list) or len(images) != len(screenshots) or len(images) != 8:
        raise StaticPairError("ordered image inventory mismatch")
    if images != [
        {"basename": item["filename"], "width": capture["resolutions"][i][0],
         "height": capture["resolutions"][i][1], "sha256": item["sha256"]}
        for i, item in enumerate(screenshots)
    ]:
        raise StaticPairError("context ordered images differ from capture")

    pair_bindings = pair.get("bindings")
    if not isinstance(pair_bindings, list) or len(pair_bindings) != 2:
        raise StaticPairError("pinned pair binding inventory is not exactly two frames")
    cap_by_name = {row["filename"]: (i, row) for i, row in enumerate(capture["frames"])}
    pair_by_pipeline: dict[str, Mapping[str, Any]] = {}
    for binding in pair_bindings:
        frame_key = binding.get("pipeline_frame")
        if frame_key not in {"image-0000", "image-0001"} or frame_key in pair_by_pipeline:
            raise StaticPairError("invalid native frame binding")
        pair_by_pipeline[frame_key] = binding
        record = binding["capture_frame"]
        if record.get("filename") not in cap_by_name:
            raise StaticPairError("native frame absent from capture metadata")
        index, cap_record = cap_by_name[record["filename"]]
        if record != cap_record:
            raise StaticPairError("pinned frame record differs from current capture")
        if capture["resolutions"][index] != [1600, 900]:
            raise StaticPairError("pinned frame dimensions differ")
        if binding.get("encoded_sha256") != png_hashes[frame_key]:
            raise StaticPairError("native encoded image hash differs")
        if record["actual"].get("screenshot_sha256") != png_hashes[frame_key]:
            raise StaticPairError("capture screenshot hash differs")
        if binding.get("path") != str(png_paths[frame_key].relative_to(root)).replace("\\", "/"):
            raise StaticPairError("native path does not bind selected PNG")
    if set(pair_by_pipeline) != {"image-0000", "image-0001"}:
        raise StaticPairError("native frame order incomplete")

    scene_region = target.get("region")
    if (scene_region != context.get("target_region")
            or context.get("region") != {"min": [low[0], low[1], low[2]], "max": [high[0], high[1], high[2]]}):
        raise StaticPairError("context/target region metadata differs")

    camera_rows = capture.get("cameras")
    frames = capture.get("frames")
    if not isinstance(camera_rows, list) or not isinstance(frames, list) or len(camera_rows) != len(frames):
        raise StaticPairError("capture camera/frame arrays malformed")
    cameras: dict[str, CameraModel] = {}
    camera_comparison: dict[str, dict[str, Any]] = {}
    features = pair.get("features")
    if not isinstance(features, Mapping):
        raise StaticPairError("native feature rows absent")
    for frame_key, binding in pair_by_pipeline.items():
        basename = binding["capture_frame"]["filename"]
        index, capture_frame = cap_by_name[basename]
        actual = capture_frame.get("actual")
        planned = camera_rows[index]
        if not isinstance(actual, Mapping) or actual.get("pose_converged") is not True:
            raise StaticPairError("actual camera pose is absent or unconverged")
        if actual.get("active_resource_packs") != ["vanilla"] or actual.get("shader_pack_applied") is not False:
            raise StaticPairError("selected frame pack/shader conditions changed")
        eye = _xyz3(actual.get("eye_position"), f"{frame_key} eye")
        if any(eye[a] < low[a] or eye[a] >= high[a] + 1 for a in range(3)):
            raise StaticPairError("actual eye lies outside complete context")
        intrinsics = planned.get("intrinsics")
        if not isinstance(intrinsics, Mapping):
            raise StaticPairError("recorded camera intrinsics absent")
        size = tuple(intrinsics.get("image_size", ()))
        focals = tuple(intrinsics.get("focal_length_pixels", ()))
        pp = tuple(intrinsics.get("principal_point_pixels", ()))
        if size != (1600, 900) or len(focals) != 2 or len(pp) != 2:
            raise StaticPairError("recorded intrinsics do not match native dimensions")
        if binding.get("recorded_camera_intrinsics") != {
            "image_size": list(size), "focal_length_pixels": list(focals), "principal_point_pixels": list(pp)
        }:
            raise StaticPairError("native recorded intrinsics differ from capture")
        camera_record = pair["pairs"][0]["cameras"]["left_camera" if frame_key == "image-0000" else "right_camera"]
        native_intrinsics = camera_record.get("intrinsics")
        native_pose = camera_record.get("pose")
        if (not isinstance(native_intrinsics, Mapping) or not isinstance(native_pose, Mapping)
                or native_intrinsics.get("image_size") != [1600, 900]):
            raise StaticPairError("native recovered camera record malformed")
        camera_comparison[frame_key] = {
            "posthoc_actual_eye": list(eye),
            "posthoc_actual_yaw_degrees": float(actual["yaw_degrees"]),
            "posthoc_actual_pitch_degrees": float(actual["pitch_degrees"]),
            "posthoc_intrinsics": dict(binding["recorded_camera_intrinsics"]),
            "native_recovered_camera": {
                "pose": dict(native_pose), "intrinsics": dict(native_intrinsics),
            },
            "interpretation": "native recovered camera and posthoc capture camera are separate; only capture camera casts truth rays",
        }
        cameras[frame_key] = CameraModel(
            position=eye,
            yaw_degrees=float(actual["yaw_degrees"]),
            pitch_degrees=float(actual["pitch_degrees"]),
            fov_degrees=float(actual["fov_degrees"]),
            eye_height=0.0,
            intrinsics=CameraIntrinsics(size, focals, pp),
            roll_degrees=float(planned.get("roll_degrees", 0.0)),
        )
        feature = features.get(frame_key)
        if (not isinstance(feature, Mapping) or len(feature.get("points", ())) > MAX_FEATURES_PER_FRAME
                or feature.get("native", {}).get("width") != 1600
                or feature.get("native", {}).get("height") != 900):
            raise StaticPairError("native feature count/dimensions invalid")
        points = feature.get("points")
        if not isinstance(points, list) or not points:
            raise StaticPairError("native keypoint points absent")
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(float(v)) for v in point)
                    or not (0.0 <= point[0] < 1600 and 0.0 <= point[1] < 900)):
                raise StaticPairError("native keypoint lies outside image")

    pair_record = pair.get("pairs")
    if not isinstance(pair_record, list) or len(pair_record) != 1:
        raise StaticPairError("native diagnostic is not a single fixed pair")
    pair_record = pair_record[0]
    graph = pair_record.get("graph")
    truncated = graph.get("truncated") if isinstance(graph, Mapping) else None
    if not isinstance(graph, Mapping) or not (truncated is False or (type(truncated) is int and truncated == 0)):
        raise StaticPairError("native candidate graph is not complete")
    edges = graph.get("edges")
    if not isinstance(edges, list) or len(edges) > MAX_GRAPH_EDGES or len(edges) != graph.get("total_edges"):
        raise StaticPairError("native candidate graph count mismatch/cap")
    if len(pair_record.get("modes", ())) > MAX_MODES or not pair_record.get("modes"):
        raise StaticPairError("native mode count invalid/cap")
    left_points = features["image-0000"]["points"]
    right_points = features["image-0001"]["points"]
    for edge in edges:
        if (not isinstance(edge, list) or len(edge) != 2 or any(type(i) is not int for i in edge)
                or not (0 <= edge[0] < len(left_points) and 0 <= edge[1] < len(right_points))):
            raise StaticPairError("native graph edge index invalid")
    if graph.get("left_keypoints") != len(left_points) or graph.get("right_keypoints") != len(right_points):
        raise StaticPairError("native graph/keypoint population mismatch")
    for mode_row in pair_record["modes"]:
        mode = mode_row.get("mode")
        selection = mode.get("selection") if isinstance(mode, Mapping) else None
        matches = mode.get("matches") if isinstance(mode, Mapping) else None
        if not isinstance(selection, list) or not isinstance(matches, list) or len(selection) != len(matches):
            raise StaticPairError("native mode selection/match rows incomplete")
        for edge_index, coordinates in zip(selection, matches):
            if type(edge_index) is not int or not (0 <= edge_index < len(edges)):
                raise StaticPairError("native mode edge reference outside graph")
            li, ri = edges[edge_index]
            expected = [*left_points[li], *right_points[ri]]
            if (not isinstance(coordinates, list) or len(coordinates) != 4
                    or any(abs(float(a) - float(b)) > 1e-4 for a, b in zip(expected, coordinates))):
                raise StaticPairError("native mode row differs from bound graph endpoints")

    return {
        "root": root, "input_hashes": input_hashes, "context": context, "occupancy": occupancy,
        "bounds": (low, high), "pair": pair, "pair_record": pair_record,
        "capture": capture, "target": target, "cameras": cameras,
        "camera_comparison": camera_comparison, "preserved_pin_checks": preserved_pin_checks,
        "features": features,
        "edges": edges, "png_hashes": png_hashes,
    }


def _classify_state(state: tuple[str, tuple[tuple[str, str], ...]]) -> str:
    name, properties = state
    if name == "minecraft:air" and not properties:
        return "air"
    if name in OPAQUE_FULL_CUBE_NAMES and not properties:
        return "opaque_full_cube"
    return "unsupported_geometry"


def trace_static_ray(
    origin: Sequence[float], direction: Sequence[float],
    occupancy: Mapping[tuple[int, int, int], tuple[str, tuple[tuple[str, str], ...]]],
    bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
    *, max_depth: float = MAX_DEPTH,
) -> dict[str, Any]:
    """Trace one bounded camera-depth ray; no missing-cell-as-air behavior."""
    if not math.isfinite(max_depth) or max_depth <= 0.0:
        raise StaticPairError("max depth must be finite and positive")
    o = _xyz3(origin, "ray origin")
    d = _xyz3(direction, "ray direction")
    if not any(component != 0.0 for component in d):
        return {"status": "unknown", "reason": "zero_direction"}
    low, high = bounds
    voxel = [math.floor(v) for v in o]
    initial_entry_axes: list[tuple[int, int]] = []
    step = [0, 0, 0]
    t_delta = [math.inf, math.inf, math.inf]
    t_max = [math.inf, math.inf, math.inf]
    for axis in range(3):
        on_plane = o[axis] == math.floor(o[axis])
        if on_plane and d[axis] == 0.0:
            return {"status": "unknown", "reason": "origin_parallel_to_grid_plane"}
        if d[axis] > 0.0:
            step[axis] = 1
            if on_plane:
                initial_entry_axes.append((axis, 1))
            t_max[axis] = (voxel[axis] + 1 - o[axis]) / d[axis]
            t_delta[axis] = 1.0 / d[axis]
        elif d[axis] < 0.0:
            step[axis] = -1
            if on_plane:
                # The floor-selected cell is touched only at t=0. Start in the
                # neighboring cell occupied for every sufficiently small t>0.
                voxel[axis] -= 1
                initial_entry_axes.append((axis, -1))
            t_max[axis] = (voxel[axis] - o[axis]) / d[axis]
            t_delta[axis] = -1.0 / d[axis]
    if any(voxel[a] < low[a] or voxel[a] > high[a] for a in range(3)):
        return {"status": "unknown", "reason": "context_exit"}
    entry_axes: tuple[tuple[int, int], ...] = tuple(initial_entry_axes)
    travelled = 0.0
    for _ in range(MAX_DDA_STEPS):
        # Depth is checked before occupancy, including the origin cell.
        if travelled > max_depth:
            return {"status": "unknown", "reason": "max_depth"}
        xyz = tuple(voxel)
        if any(xyz[a] < low[a] or xyz[a] > high[a] for a in range(3)):
            return {"status": "unknown", "reason": "context_exit"}
        state = occupancy.get(xyz)
        if state is None:
            return {"status": "unknown", "reason": "grid_gap", "cell": list(xyz)}
        kind = _classify_state(state)
        if kind == "unsupported_geometry":
            return {"status": "unknown", "reason": "unsupported_prefix", "cell": list(xyz), "name": state[0]}
        if kind == "opaque_full_cube":
            if len(entry_axes) == 0:
                return {"status": "unknown", "reason": "origin_inside_block", "cell": list(xyz), "name": state[0]}
            if len(entry_axes) > 1:
                return {"status": "unknown", "reason": "tied_face_ownership", "cell": list(xyz), "name": state[0]}
            axis, sign = entry_axes[0]
            face = {
                (0, 1): "west", (0, -1): "east",
                (1, 1): "down", (1, -1): "up",
                (2, 1): "north", (2, -1): "south",
            }[(axis, sign)]
            point = [o[a] + travelled * d[a] for a in range(3)]
            return {"status": "hit", "cell": list(xyz), "face": face, "point": point,
                    "depth": travelled, "name": state[0], "properties": dict(state[1])}
        if not any(math.isfinite(t) for t in t_max):
            return {"status": "unknown", "reason": "zero_direction"}
        next_t = min(t_max)
        if next_t > max_depth:
            return {"status": "unknown", "reason": "max_depth"}
        # Exact equality is the deterministic represented-number tie rule. A
        # merely near-equal positive crossing is stepped separately so its
        # positive-length side voxel is inspected rather than silently skipped.
        axes = tuple(axis for axis in range(3) if t_max[axis] == next_t)
        if not axes:
            return {"status": "unknown", "reason": "numerical_step_failure"}
        # Step all simultaneous axes. Zero-length side cells are never inspected.
        for axis in axes:
            voxel[axis] += step[axis]
            t_max[axis] += t_delta[axis]
        travelled = next_t
        if any(voxel[a] < low[a] or voxel[a] > high[a] for a in range(3)):
            return {"status": "unknown", "reason": "context_exit"}
        entry_axes = tuple((axis, step[axis]) for axis in axes)
    return {"status": "unknown", "reason": "dda_step_cap"}


def _ray_direction(pixel: Sequence[float], camera: CameraModel) -> tuple[float, float, float]:
    intrinsics = camera.intrinsics
    if intrinsics is None:
        raise StaticPairError("pinned actual camera needs explicit intrinsics")
    x, y = float(pixel[0]), float(pixel[1])
    fx, fy = intrinsics.focal_length_pixels
    cx, cy = intrinsics.principal_point_pixels
    view = np.asarray(((x - cx) / fx, -(y - cy) / fy, 1.0), dtype=np.float64)
    direction = view @ camera_rotation_matrix(camera)
    return tuple(float(v) for v in direction)


def _cast_features(inputs: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    low, high = inputs["bounds"]
    occupancy = inputs["occupancy"]
    target_region = inputs["context"]["target_region"]
    for frame_key in ("image-0000", "image-0001"):
        camera = inputs["cameras"][frame_key]
        origin = camera.position  # eye position; eye_height is explicitly zero
        outcomes = []
        for index, point in enumerate(inputs["features"][frame_key]["points"]):
            outcome = trace_static_ray(origin, _ray_direction(point, camera), occupancy, (low, high))
            outcome["context_hit"] = outcome.get("status") == "hit"
            if outcome.get("status") == "hit":
                outcome["target_hit"] = _target_contains(outcome["cell"], target_region)
            else:
                outcome["target_hit"] = None
            outcome["feature_index"] = index
            outcome["pixel"] = [float(point[0]), float(point[1])]
            outcomes.append(outcome)
        result[frame_key] = outcomes
    return result


def _pixel_identity(point: Sequence[float]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _target_contains(cell: Sequence[int], target_region: Mapping[str, Any]) -> bool:
    lo = target_region.get("min")
    hi = target_region.get("max")
    return (isinstance(lo, list) and isinstance(hi, list) and len(lo) == 3 and len(hi) == 3
            and all(type(lo[a]) is int and type(hi[a]) is int and lo[a] <= int(cell[a]) <= hi[a]
                    for a in range(3)))


def _truth_pairs(
    left_points: Sequence[Sequence[float]], right_points: Sequence[Sequence[float]],
    left_hits: Sequence[Mapping[str, Any]], right_hits: Sequence[Mapping[str, Any]],
) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    buckets: dict[tuple[tuple[int, int, int], str], dict[tuple[int, int, int], list[int]]] = defaultdict(lambda: defaultdict(list))
    tolerance = SURFACE_TOLERANCE
    for index, hit in enumerate(right_hits):
        if hit.get("status") != "hit" or hit.get("target_hit", True) is not True:
            continue
        surface = (tuple(hit["cell"]), str(hit["face"]))
        cell = tuple(math.floor(float(hit["point"][axis]) / tolerance) for axis in range(3))
        buckets[surface][cell].append(index)
    result: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    tolerance_sq = tolerance * tolerance
    for left_index, left in enumerate(left_hits):
        if left.get("status") != "hit" or left.get("target_hit", True) is not True:
            continue
        surface = (tuple(left["cell"]), str(left["face"]))
        grid = buckets.get(surface)
        if not grid:
            continue
        base = tuple(math.floor(float(left["point"][axis]) / tolerance) for axis in range(3))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for right_index in grid.get((base[0] + dx, base[1] + dy, base[2] + dz), ()):
                        right = right_hits[right_index]
                        delta = [float(left["point"][a]) - float(right["point"][a]) for a in range(3)]
                        if sum(v * v for v in delta) <= tolerance_sq:
                            result.add((_pixel_identity(left_points[left_index]),
                                        _pixel_identity(right_points[right_index])))
    return result


def _edge_disposition(
    left_index: int, right_index: int, left_hits: Sequence[Mapping[str, Any]],
    right_hits: Sequence[Mapping[str, Any]], truth: set[tuple[tuple[float, float], tuple[float, float]]],
    left_points: Sequence[Sequence[float]], right_points: Sequence[Sequence[float]],
) -> str:
    left, right = left_hits[left_index], right_hits[right_index]
    if left.get("status") != "hit" or right.get("status") != "hit":
        return "unknown"
    if left.get("target_hit", True) is not True or right.get("target_hit", True) is not True:
        return "outside_target"
    identity = (_pixel_identity(left_points[left_index]), _pixel_identity(right_points[right_index]))
    if identity in truth:
        return "correct"
    if tuple(left["cell"]) == tuple(right["cell"]) and left["face"] == right["face"]:
        return "same_face_out_of_tolerance"
    if (left.get("name") == right.get("name")
            and left.get("properties", {}) == right.get("properties", {})
            and left.get("face") == right.get("face")):
        return "same_state_same_slot_other_block"
    return "wrong_surface"


def _summarize_edges(
    edge_indices: Sequence[int], edges: Sequence[Sequence[int]], left_points: Sequence[Sequence[float]],
    right_points: Sequence[Sequence[float]], left_hits: Sequence[Mapping[str, Any]],
    right_hits: Sequence[Mapping[str, Any]], truth: set[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[dict[str, Any], list[list[Any]]]:
    verdict_by_identity: dict[tuple[tuple[float, float], tuple[float, float]], str] = {}
    rows: list[list[Any]] = []
    raw_unknown_reasons: Counter[str] = Counter()
    for edge_index in edge_indices:
        left_index, right_index = edges[edge_index]
        verdict = _edge_disposition(left_index, right_index, left_hits, right_hits, truth, left_points, right_points)
        identity = (_pixel_identity(left_points[left_index]), _pixel_identity(right_points[right_index]))
        previous = verdict_by_identity.get(identity)
        if previous is not None and previous != verdict:
            raise StaticPairError("duplicate native pixel-pair identity has conflicting verdicts")
        verdict_by_identity[identity] = verdict
        if verdict == "unknown":
            lreason = left_hits[left_index].get("reason", "") if left_hits[left_index].get("status") != "hit" else ""
            rreason = right_hits[right_index].get("reason", "") if right_hits[right_index].get("status") != "hit" else ""
            raw_unknown_reasons[str(lreason or rreason)] += 1
        rows.append([int(edge_index), int(left_index), int(right_index), verdict])
    counts = Counter(verdict_by_identity.values())
    total = len(verdict_by_identity)
    scorable = (counts["correct"] + counts["same_face_out_of_tolerance"]
                + counts["same_state_same_slot_other_block"] + counts["wrong_surface"])
    recall_denominator = len(truth)
    summary = {
        "raw_edge_rows": len(edge_indices),
        "unique_pixel_pairs": total,
        "correct": counts["correct"],
        "incorrect": scorable - counts["correct"],
        "same_face_out_of_tolerance": counts["same_face_out_of_tolerance"],
        "same_state_same_slot_other_block": counts["same_state_same_slot_other_block"],
        "wrong_surface": counts["wrong_surface"],
        "unknown": counts["unknown"],
        "outside_target": counts["outside_target"],
        "unknown_reason_rows_raw": dict(sorted(raw_unknown_reasons.items())),
        "scorable": scorable,
        "coverage": None if total == 0 else scorable / total,
        "conditional_static_precision": None if scorable == 0 else counts["correct"] / scorable,
        "truth_pair_denominator": recall_denominator,
        "conditional_static_recall": None if recall_denominator == 0 else counts["correct"] / recall_denominator,
    }
    return summary, rows


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def build_report(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build the bounded diagnostic from exact pinned artifacts; no image decode."""
    inputs = _load_inputs(root)
    pair_record = inputs["pair_record"]
    features = inputs["features"]
    points_left = features["image-0000"]["points"]
    points_right = features["image-0001"]["points"]
    edges = inputs["edges"]
    rays = _cast_features(inputs)
    left_hits, right_hits = rays["image-0000"], rays["image-0001"]
    truth = _truth_pairs(points_left, points_right, left_hits, right_hits)
    all_edge_ids = list(range(len(edges)))
    graph_summary, graph_rows = _summarize_edges(
        all_edge_ids, edges, points_left, points_right, left_hits, right_hits, truth)
    mode_outputs = []
    for mode_index, mode_row in enumerate(pair_record["modes"]):
        selection = mode_row["mode"]["selection"]
        summary, rows = _summarize_edges(
            selection, edges, points_left, points_right, left_hits, right_hits, truth)
        mode_outputs.append({
            "mode_index": mode_index,
            "winner": bool(mode_row.get("winner")),
            "native_refusal": mode_row.get("score", {}).get("refusal"),
            "summary": summary,
            "edge_dispositions": rows,
        })
    status_counts: dict[str, dict[str, int]] = {}
    target_counts: dict[str, dict[str, int]] = {}
    for frame, outcomes in rays.items():
        counts = Counter(hit["status"] if hit["status"] == "hit" else hit.get("reason", "unknown") for hit in outcomes)
        status_counts[frame] = dict(sorted(counts.items()))
        target_counts[frame] = {
            "context_hits": sum(hit.get("status") == "hit" for hit in outcomes),
            "target_hits": sum(hit.get("status") == "hit" and hit.get("target_hit") is True for hit in outcomes),
            "known_outside_target_hits": sum(hit.get("status") == "hit" and hit.get("target_hit") is False for hit in outcomes),
        }
    document: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "mc-photogrammetry-static-block-pair-diagnostic",
        "task": "SOL-IMG-STATIC-BLOCK-PAIR-DIAGNOSTIC-20261005",
        "inputs": {
            "sha256": dict(sorted(PINNED_SHA256.items())),
            "bytes": dict(sorted(PINNED_BYTES.items())),
            "context_region": inputs["context"]["region"],
            "context_rows": len(inputs["occupancy"]),
            "target_region": inputs["context"]["target_region"],
            "selected_pipeline_frames": ["image-0000", "image-0001"],
            "selected_png_sha256": inputs["png_hashes"],
        },
        "policy": {
            "oracle": "conditional static-block first-hit oracle over the complete pinned context grid",
            "opaque_full_cube_names_empty_properties_only": sorted(OPAQUE_FULL_CUBE_NAMES),
            "unsupported_nonair": "unknown at first unsupported voxel; no opacity/transparency assumption",
            "unknown_handling": "never counted as a negative; excluded from conditional precision and truth denominator",
            "target_scope": "context supplies first-hit visibility; only first hits inside target_region enter target correspondence denominator; known context hits outside target are separately excluded",
            "max_depth_camera_units": MAX_DEPTH,
            "max_dda_steps": MAX_DDA_STEPS,
            "surface_tolerance_blocks": SURFACE_TOLERANCE,
            "tie_rule": "only exactly equal represented crossing parameters advance together; near-equal positive crossings are stepped separately; a first occupied voxel entered through a tie is unknown; a grid-plane origin is initialized to the cell entered for t>0 (parallel-to-plane origin is unknown)",
            "entity_dynamic_and_renderer_limits": "not represented; results are not screenshot truth",
        },
        "source_pins": {
            "reconstruction_camera_py": PINNED_SHA256["camera_source"],
            "diagnostic_module_sha256": _sha256(Path(__file__)),
            "diagnostic_test_sha256": _sha256(root / "tests/test_evaluation_static_block_pair.py"),
            "pair_diagnostic_task": inputs["pair"].get("task"),
            "preserved_pin_checks": inputs["preserved_pin_checks"],
        },
        "camera_models": inputs["camera_comparison"],
        "population": {
            "left_keypoints": len(points_left),
            "right_keypoints": len(points_right),
            "complete_reciprocal_candidate_edges": len(edges),
            "candidate_graph_truncated": False,
            "native_modes": len(pair_record["modes"]),
            "truth_pair_unique_pixel_identities": len(truth),
        },
        "ray_outcomes": rays,
        "ray_status_counts": status_counts,
        "ray_target_counts": target_counts,
        "candidate_graph": {"summary": graph_summary, "edge_dispositions": graph_rows},
        "native_modes": mode_outputs,
        "interpretation": (
            "Conditional static-block correspondence only. Camera poses and intrinsics are posthoc capture metadata; "
            "the native feature/candidate/mode rows are reused without rerunning production. The voxel oracle does not "
            "model entities, dynamic changes, transparent/cutout/fluid rendering, or unsupported block models."
        ),
    }
    encoded = _canonical_bytes(document)
    if len(encoded) > MAX_CANONICAL_OUTPUT_BYTES:
        raise StaticPairError("canonical output byte cap exceeded")
    document["seal_sha256"] = hashlib.sha256(encoded).hexdigest()
    final = _canonical_bytes(document)
    if len(final) > MAX_CANONICAL_OUTPUT_BYTES:
        raise StaticPairError("sealed output byte cap exceeded")
    return document


def write_report(path: Path, *, root: Path = REPO_ROOT) -> str:
    """Create one sealed report without overwriting an existing file."""
    if path.exists():
        raise StaticPairError(f"refusing to overwrite {path}")
    document = build_report(root)
    raw = json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    if len(raw) > MAX_CANONICAL_OUTPUT_BYTES:
        raise StaticPairError("formatted output byte cap exceeded")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(write_report(arguments.output))
