"""Fresh generated visible-roof control; never invokes production reconstruction.

Scene/cameras are renderer/test oracles only. A later packet passes PNG paths to
ordinary production and does not import this generator or its oracle receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mc_photogrammetry.reconstruction.camera import CameraModel, camera_eye, camera_rotation_matrix, intrinsics_for
from mc_photogrammetry.reconstruction import calibration as cal, region_growth as rg
from mc_photogrammetry.vision.gui_mask import create_gui_mask

WIDTH, HEIGHT = 960, 720
SEED = 202610091416
BOX = ((0., 8.), (0., 4.), (0., 8.))
EYES = ((-3., 11., -4.), (-4., 11., -2.), (-2., 12., -4.5))
TARGET = (4., 1.5, 4.)
PATCH = (2.15, 5.85)
SEAM_HALF_WIDTH = .045


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def render_camera(index):
    eye = EYES[index]
    dx, dz = TARGET[0] - eye[0], TARGET[2] - eye[2]
    return CameraModel(position=eye, eye_height=0., yaw_degrees=-math.degrees(math.atan2(dx, dz)),
        pitch_degrees=math.degrees(math.atan2(eye[1] - TARGET[1], math.hypot(dx, dz))), fov_degrees=60.)


def render_view(index):
    camera = render_camera(index)
    intrinsic = intrinsics_for(camera, WIDTH, HEIGHT)
    fx, fy = intrinsic.focal_length_pixels; cx, cy = intrinsic.principal_point_pixels
    xx, yy = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    local = np.stack(((xx - cx) / fx, (cy - yy) / fy, np.ones_like(xx)), axis=-1)
    rays = local @ camera_rotation_matrix(camera)
    eye = camera_eye(camera)
    distance = np.full((HEIGHT, WIDTH), np.inf)
    points = np.zeros((HEIGHT, WIDTH, 3)); faces = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
    for axis in range(3):
        for side in range(2):
            with np.errstate(divide="ignore", invalid="ignore"):
                length = (BOX[axis][side] - eye[axis]) / rays[..., axis]
            valid = np.isfinite(length) & (length > 0) & (length < distance)
            hit = eye + np.where(valid, length, 0)[..., None] * rays
            for other in range(3):
                if other != axis:
                    valid &= (hit[..., other] >= BOX[other][0]) & (hit[..., other] <= BOX[other][1])
            distance[valid] = length[valid]; points[valid] = hit[valid]; faces[valid] = 2 * axis + side
    rgb = np.full((HEIGHT, WIDTH, 3), (65, 80, 95), dtype=np.uint8)
    for face in range(6):
        mask = faces == face
        if not mask.any(): continue
        tangent = [axis for axis in range(3) if axis != face // 2]
        u, v = points[..., tangent[0]], points[..., tangent[1]]
        a, b = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        base = 178 + ((a * 17 + b * 23 + face * 31 + SEED) % 17 - 8)
        for channel in range(3): rgb[..., channel][mask] = np.clip(base[mask] + channel * 4, 0, 255)
        # Both families on every visible plane; one-world-unit cell pitch.
        seam = (np.abs(u - np.rint(u)) <= SEAM_HALF_WIDTH) | (np.abs(v - np.rint(v)) <= SEAM_HALF_WIDTH)
        rgb[mask & seam] = 24
        # Deterministic small cell marks; they do not replace the continuous seams.
        marks = (np.mod(u, 1.) > .40) & (np.mod(u, 1.) < .50) & (np.mod(v, 1.) > .35) & (np.mod(v, 1.) < .45)
        rgb[mask & marks] = 130
    return rgb, faces


def oracle_project(index, points):
    """Independent analytic forward projection, not renderer ray-face reuse."""
    camera = render_camera(index)
    yaw, pitch = math.radians(camera.yaw_degrees), math.radians(camera.pitch_degrees)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rotation = np.array(((-cy, 0., -sy), (-sp * sy, cp, sp * cy), (-cp * sy, -sp, cp * cy)))
    camera_points = (np.asarray(points) - np.asarray(EYES[index])) @ rotation.T
    focal = HEIGHT / (2 * math.tan(math.radians(60.) / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = np.stack((WIDTH / 2 + focal * camera_points[:, 0] / camera_points[:, 2],
                              HEIGHT / 2 - focal * camera_points[:, 1] / camera_points[:, 2]), axis=1)
    return projected, camera_points[:, 2]


def inside_polygon(points, polygon):
    values = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        delta = end - start
        values.append(delta[0] * (points[..., 1] - start[1]) - delta[1] * (points[..., 0] - start[0]))
    cross = np.stack(values, axis=-1)
    return np.all(cross >= -1e-8, axis=-1) | np.all(cross <= 1e-8, axis=-1)


def roof_oracle(index, faces, usable):
    lo, hi = PATCH
    polygon, depth = oracle_project(index, [(lo, 4., lo), (hi, 4., lo), (hi, 4., hi), (lo, 4., hi)])
    yy, xx = np.indices((HEIGHT, WIDTH))
    patch = inside_polygon(np.stack((xx, yy), axis=-1), polygon)
    # Convex box occlusion proof: before reaching roof y4, each segment from an
    # eye above it has y>4, outside every box face's bounded extent.
    unoccluded = EYES[index][1] > BOX[1][1] and np.all(depth > 0)
    bound = bool(np.all(faces[patch] == 3)) and bool(patch.any())
    families = []
    for varying in (0, 2):
        extents = []
        for fixed in (3., 4., 5.):
            t = np.linspace(lo, hi, 1025)
            points = np.stack((t if varying == 0 else np.full_like(t, fixed),
                               np.full_like(t, 4.), t if varying == 2 else np.full_like(t, fixed)), axis=1)
            projected, z = oracle_project(index, points)
            pixels = np.rint(projected).astype(int)
            valid = (z > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < WIDTH) & (pixels[:, 1] >= 0) & (pixels[:, 1] < HEIGHT)
            ids = np.flatnonzero(valid)
            valid[ids] &= usable[pixels[ids, 1], pixels[ids, 0]] & patch[pixels[ids, 1], pixels[ids, 0]] & (faces[pixels[ids, 1], pixels[ids, 0]] == 3)
            edges = np.diff(np.r_[False, valid, False].astype(np.int8))
            runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
            extents.append(max((float(np.linalg.norm(projected[b] - projected[a])) for a, b in runs), default=0.))
        families.append({"tangent_axis": varying, "seam_count": 3, "contiguous_usable_extents_pixels": extents})
    pitch = []; widths = []
    for varying in (0, 2):
        a = np.array((4., 4., 4.)); b = a.copy(); b[varying] += 1.
        projected, _ = oracle_project(index, [a, b]); pitch.append(float(np.linalg.norm(projected[1] - projected[0])))
        a[varying] -= SEAM_HALF_WIDTH; b = a.copy(); b[varying] += 2 * SEAM_HALF_WIDTH
        projected, _ = oracle_project(index, [a, b]); widths.append(float(np.linalg.norm(projected[1] - projected[0])))
    record = {"unoccluded_by_convex_box": bool(unoccluded), "renderer_face_binding": bound,
        "oracle_patch_pixels": int(patch.sum()), "usable_oracle_patch_pixels": int((patch & usable).sum()),
        "seam_families": families, "one_unit_cell_pitch_pixels": pitch, "stripe_width_pixels": widths,
        "sampling_justification": "One-unit cells project to measured pitches; seam width is0.09unit. Both families have continuous GUI-usable roof runs.40px is the unchanged detector floor, not proof of plane visibility.",
        "oracle_qualifies": bool(unoccluded and bound and (patch & usable).sum() >= math.ceil(WIDTH * HEIGHT * rg.DEFAULT_REGION_GROWTH_CONFIG.min_region_fraction)
            and min(pitch) > 2 * max(widths) and min(widths) >= 2
            and all(min(f["contiguous_usable_extents_pixels"]) >= cal.MINIMUM_LENGTH for f in families))}
    return record, polygon


def default_proposal_control(rgb, polygon, frame_id):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fit = cal.estimate_camera(grey)  # pixels only, every production default retained
    if fit is None:
        return {"stage": "camera", "status": "camera_estimate_unavailable", "qualifies": False}
    usable = ~create_gui_mask(HEIGHT, WIDTH)
    observer = rg._ScalarAttritionObserver(max_items=16384, max_bytes=4 * 1024 * 1024)
    original = rg._build_graph; captured = []
    def forwarding(*args, **kwargs):
        result = original(*args, **kwargs)
        graph, evidence = result
        if graph is not None:
            sx, sy = kwargs.get("original_pixel_scale", (1., 1.))
            roi = inside_polygon(graph.node_xy * np.array((sx, sy)), polygon)
            support, score = graph.supports[roi], graph.plane_scores[roi]
            captured.append({"roof_patch_atoms": int(roi.sum()),
                "tangent_support_mass": [float(support[:, a].sum()) for a in (0, 2)],
                "both_tangents_nonzero_atoms": int(np.count_nonzero((support[:, 0] > 0) & (support[:, 2] > 0))),
                "normal1_score_max": float(score[:, 1].max()) if len(score) else None,
                "normal1_score_pass_atoms": int(np.count_nonzero(score[:, 1] >= rg.DEFAULT_REGION_GROWTH_CONFIG.min_seed_score))})
        return result
    rg._build_graph = forwarding
    try:
        with rg._scalar_attrition_context(observer, frame=frame_id):
            result = rg.propose_image_planes_v2(rgb, usable, fit)
    finally:
        rg._build_graph = original
    raw = observer.record(); seed = raw["stages"].get("seed", {})
    seeds = [{key: row.get(key) for key in ("axis", "side", "disposition", "node_count", "eligible_seed_count",
        "after_side_count", "after_score_count", "after_margin_count", "after_support_count",
        "max_side_normal_score", "max_side_tangent_support")} for row in seed.get("examples", ())]
    proposal_counts = [sum(p.axis == a for p in result.proposals) for a in range(3)]
    complete = raw["status"] == "complete" and len(captured) == 1 and len(seeds) == 6 and seed.get("omitted_examples", 0) == 0
    normal1_seeds = sum(r["eligible_seed_count"] or 0 for r in seeds if r["axis"] == 1)
    return {"stage": "proposal", "status": result.status, "reason": result.reason,
        "camera_segments": fit.segments, "camera_inliers": fit.inliers, "camera_axis_inliers": list(fit.per_axis_inliers),
        "same_pass_build_graph_calls": len(captured), "support": captured[0] if len(captured) == 1 else None,
        "seed_predicates": seeds, "normal1_eligible_seeds": normal1_seeds,
        "proposal_normal_counts": proposal_counts, "observer_complete": complete,
        "qualifies": bool(complete and normal1_seeds > 0 and proposal_counts[1] > 0)}


def run_controls(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / "inputs"; inputs.mkdir()
    rows = []
    for index in range(3):
        rgb, faces = render_view(index)
        path = inputs / f"view-{index:02d}.png"; Image.fromarray(rgb).save(path, format="PNG")
        oracle, polygon = roof_oracle(index, faces, ~create_gui_mask(HEIGHT, WIDTH))
        before = hashlib.sha256(rgb.tobytes()).hexdigest()
        control = default_proposal_control(rgb, polygon, f"image-{index:04d}")
        unchanged = before == hashlib.sha256(rgb.tobytes()).hexdigest()
        rows.append({"ordinal": index, "frame_id": f"image-{index:04d}", "label": path.name,
            "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "oracle": oracle, "estimated_default_control": control, "pixels_unchanged": unchanged,
            "qualifies": bool(oracle["oracle_qualifies"] and control["qualifies"] and unchanged)})
    record = {"schema": "synthetic-visible-normal1-control-v1", "generated_only": True,
        "seed": SEED, "width": WIDTH, "height": HEIGHT, "views": rows,
        "qualifying_view_count": sum(r["qualifies"] for r in rows),
        "packet_eligible": sum(r["qualifies"] for r in rows) >= 2 and all(r["estimated_default_control"]["stage"] == "proposal" for r in rows),
        "production_invocations": 0, "oracle_passed_to_estimator_or_proposal": False,
        "production_defaults_unchanged": True, "accuracy_claim": False}
    record["result_sha256"] = digest(record)
    with (destination / "control.json").open("x", encoding="ascii", newline="\n") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
    return record
