"""Synthetic-first, image-only projective plane-track diagnostic.

Inputs are RGB pixels and opaque, lexicographically ordered frame IDs. A result
is sparse observed support, not a filled surface, camera pose, or 3-D geometry.
No production reconstruction module is imported or called.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Sequence

import cv2
import numpy as np


MAX_FRAMES = 32
LEGACY_MAX_FRAMES = 8
MAX_PIXELS_PER_FRAME = 1_000_000
MAX_TOTAL_RGB_BYTES = 24_000_000
MAX_PAIRS = MAX_FRAMES * (MAX_FRAMES - 1) // 2
EXTENDED_MAX_TOTAL_RGB_BYTES = 96_000_000
EXTENDED_MAX_DESCRIPTOR_COMPARISONS = 486_080_000  # 2 * MAX_PAIRS * MAX_FEATURES**2
EXTENDED_MAX_OUTPUT_BYTES = 128_000_000
PAIR_SCHEDULE = "complete-unordered-pairs-v2"
LEGACY_PAIR_SCHEDULE = "id-order-cyclic-distance-2-v1"
MAX_FEATURES = 700
MAX_MODELS_PER_PAIR = 3
MAX_MATCHES_PER_PAIR = 700
MAX_OUTPUT_BYTES = 1_000_000
MAX_DESCRIPTOR_COMPARISONS = 12_000_000
MAX_CYCLE_COMBINATIONS = 1_000
RATIO = 0.72
TRANSFER_PX = 3.0
MIN_HOLDOUT = 8
MIN_CYCLE_TRIPLES = 8
UNRESOLVED = ("global_scale", "orientation_and_axis_assignment", "plane_offset", "occlusion", "block_lattice")


@dataclass(frozen=True, slots=True)
class PairModel:
    pair: tuple[str, str]
    transform: tuple[tuple[float, float, float], ...]
    fit_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    fit_keys: tuple[tuple[int, int], ...]
    support_keys: tuple[tuple[int, int], ...]
    holdout_rows: tuple[tuple[int, int, float, float, float, float, float, float], ...]


def _residual_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int | str]:
    data = np.sort(np.asarray(values, dtype=np.float64))
    def quantile(q: float) -> float | str:
        position = (len(data) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            value = float(data[low])
        elif math.isinf(data[low]) or math.isinf(data[high]):
            value = math.inf
        else:
            value = float(data[low] + (data[high] - data[low]) * (position - low))
        return "inf" if math.isinf(value) else value
    return {"count": len(data), "p50_px": quantile(.5),
            "p95_px": quantile(.95), "max_px": quantile(1.)}


def _pair_holdout_report(model: PairModel) -> dict[str, object]:
    rows = model.holdout_rows
    forward = np.array([row[6] for row in rows])
    reverse = np.array([row[7] for row in rows])
    forward_good = forward <= TRANSFER_PX
    reverse_good = reverse <= TRANSFER_PX
    return {
        "fit_keys": [list(key) for key in model.fit_keys],
        "held_out_rows": [[*row[:6], *("inf" if math.isinf(v) else v for v in row[6:])]
                          for row in rows],
        "count": len(rows), "joint_accepted_count": int(np.sum(forward_good & reverse_good)),
        "forward_rejected_count": int(np.sum(~forward_good)),
        "reverse_rejected_count": int(np.sum(~reverse_good)),
        "joint_rejected_count": int(np.sum(~(forward_good & reverse_good))),
        "forward": _residual_summary(forward), "reverse": _residual_summary(reverse),
    }


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _pair_key(left: str, right: str) -> str:
    """A canonical JSON pair keeps opaque frame IDs unambiguous."""
    return _canonical([left, right]).decode("ascii")


def _pair_schedule(names: Sequence[str]) -> list[tuple[str, str]]:
    """Evaluate every unordered image pair, through the 32-frame hard limit."""
    return list(itertools.combinations(names, 2))


def _legacy_cyclic_pair_schedule(names: Sequence[str]) -> list[tuple[str, str]]:
    """Replay v5 reports, whose extended runs sampled a distance-two ring."""
    pairs = {tuple(sorted((name, names[(index + step) % len(names)])))
             for index, name in enumerate(names) for step in (1, 2)}
    return sorted(pairs)


def _scheduled_triangles(names: Sequence[str], pairs: Sequence[tuple[str, str]]
                         ) -> list[tuple[str, str, str]]:
    edges = set(pairs)
    return [(a, b, c) for a, b, c in itertools.combinations(names, 3)
            if (a, b) in edges and (a, c) in edges and (b, c) in edges]


def _bounded_cycle_triangles(names: Sequence[str]) -> list[tuple[str, str, str]]:
    """Retain a fixed cycle-validation budget while matching all image pairs."""
    ring = _legacy_cyclic_pair_schedule(names)
    return _scheduled_triangles(names, ring)


def _output_limit(schema: str) -> int:
    return EXTENDED_MAX_OUTPUT_BYTES if schema in (
        "projective-plane-tracks-v5", "projective-plane-tracks-v6") else MAX_OUTPUT_BYTES


def _frame_input(frames: Sequence[tuple[str, np.ndarray]]) -> tuple[list[tuple[str, np.ndarray, str]], int]:
    if not isinstance(frames, (tuple, list)) or not 3 <= len(frames) <= MAX_FRAMES:
        raise ValueError("frame count must be in [3, 32]")
    if len(_pair_schedule([str(i) for i in range(len(frames))])) > MAX_PAIRS:
        raise ValueError("pair budget exceeded")
    result = []
    total = 0
    names = []
    for entry in frames:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("each frame must be an (id, RGB array) pair")
        name, raw = entry
        if not isinstance(name, str) or not 1 <= len(name) <= 100 or not name.isprintable():
            raise ValueError("invalid frame ID")
        if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError("RGB frames must be uint8 HxWx3 arrays")
        h, w = raw.shape[:2]
        if h < 64 or w < 64 or h * w > MAX_PIXELS_PER_FRAME:
            raise ValueError("frame pixel budget exceeded or image too small")
        total += int(raw.nbytes)
        if total > (EXTENDED_MAX_TOTAL_RGB_BYTES if len(frames) > LEGACY_MAX_FRAMES else MAX_TOTAL_RGB_BYTES):
            raise ValueError("RGB byte budget exceeded")
        image = np.ascontiguousarray(raw).copy()
        digest = _digest_bytes(h.to_bytes(4, "big") + w.to_bytes(4, "big") + image.tobytes())
        result.append((name, image, digest))
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError("frame IDs must be unique and lexicographically ordered")
    return result, total


def _seed(*values: object) -> int:
    return int.from_bytes(hashlib.sha256(_canonical(values)).digest()[:4], "big") & 0x7fffffff


def _features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # OpenCV's requested nfeatures can return a few extra tied responses.
    detector = cv2.SIFT_create(nfeatures=MAX_FEATURES - 32)
    cv2.setRNGSeed(0)
    points, desc = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    if desc is None or len(points) < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 128), dtype=np.float32)
    if len(points) > MAX_FEATURES:
        raise ValueError("feature budget exceeded; refusing truncation")
    return np.array([point.pt for point in points], dtype=np.float64), np.array(desc, dtype=np.float32, copy=True)


def _matches(left: np.ndarray, right: np.ndarray,
             telemetry: dict[str, object] | None = None) -> tuple[tuple[int, int], ...]:
    if len(left) < 2 or len(right) < 2:
        if telemetry is not None:
            telemetry.update(matching_status="insufficient_descriptors",
                             forward_ratio_pairs=[], reverse_ratio_pairs=[], reciprocal_pairs=[],
                             forward_ratio_pass_count=0, reverse_ratio_pass_count=0,
                             reciprocal_count=0)
        return ()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(left, right, k=2)
    reverse = matcher.knnMatch(right, left, k=2)
    def good(rows: list[list[cv2.DMatch]]) -> dict[int, int]:
        return {row[0].queryIdx: row[0].trainIdx for row in rows
                if len(row) == 2 and row[0].distance < RATIO * row[1].distance}
    a, b = good(forward), good(reverse)
    pairs = sorted((i, j) for i, j in a.items() if b.get(j) == i)
    if len(pairs) > MAX_MATCHES_PER_PAIR:
        raise ValueError("match budget exceeded; refusing truncation")
    if telemetry is not None:
        telemetry.update(matching_status="evaluated",
                         forward_ratio_pairs=[list(row) for row in sorted(a.items())],
                         reverse_ratio_pairs=[list(row) for row in sorted(b.items())],
                         reciprocal_pairs=[list(row) for row in pairs],
                         forward_ratio_pass_count=len(a), reverse_ratio_pass_count=len(b),
                         reciprocal_count=len(pairs))
    return tuple(pairs)


def _transfer(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    homo = np.column_stack((xy, np.ones(len(xy)))) @ h.T
    out = np.full((len(xy), 2), np.inf)
    valid = np.abs(homo[:, 2]) > 1e-10
    out[valid] = homo[valid, :2] / homo[valid, 2, None]
    return out


def _normalized(h: np.ndarray, telemetry: dict[str, object] | None = None) -> np.ndarray:
    if h.shape != (3, 3) or not np.all(np.isfinite(h)) or abs(np.linalg.det(h)) < 1e-9:
        raise ValueError("malformed projective transform")
    scale = h[2, 2]
    if abs(scale) < 1e-9:
        raise ValueError("malformed projective scale")
    result = h / scale
    condition = float(np.linalg.cond(result))
    if telemetry is not None:
        telemetry["condition"] = condition if math.isfinite(condition) else "inf"
    if condition > 1e8 or not np.all(np.isfinite(result)):
        raise ValueError("ill-conditioned projective transform")
    return result


def _coverage(points: np.ndarray, shape: tuple[int, int],
              telemetry: dict[str, object] | None = None) -> bool:
    if telemetry is not None:
        telemetry.update(count=len(points), extent_xy=None, extent_area=None,
                         grid_cell_count=None, connected_count=None, connected_required=None,
                         first_failed_gate=None, passed=None)
    if len(points) < 12:
        if telemetry is not None:
            telemetry.update(first_failed_gate="too_few_points", passed=False)
        return False
    h, w = shape
    extent = np.ptp(points, axis=0)
    if telemetry is not None:
        telemetry.update(extent_xy=[float(v) for v in extent],
                         extent_area=float(extent[0] * extent[1]))
    if extent[0] < w * .18 or extent[1] < h * .18 or extent[0] * extent[1] < w * h * .055:
        if telemetry is not None:
            telemetry.update(first_failed_gate="extent", passed=False)
        return False
    cells = {(min(3, int(x * 4 / w)), min(3, int(y * 4 / h))) for x, y in points}
    if telemetry is not None:
        telemetry["grid_cell_count"] = len(cells)
    if len(cells) < 4:
        if telemetry is not None:
            telemetry.update(first_failed_gate="grid", passed=False)
        return False
    # Sparse support must be one locally continuous feature component.
    radius2 = (.14 * math.hypot(w, h)) ** 2
    visited = {0}
    queue = [0]
    while queue:
        i = queue.pop()
        neighbors = np.flatnonzero(np.sum((points - points[i]) ** 2, axis=1) <= radius2)
        for j in neighbors:
            k = int(j)
            if k not in visited:
                visited.add(k)
                queue.append(k)
    required = math.ceil(.9 * len(points))
    passed = len(visited) >= required
    if telemetry is not None:
        telemetry.update(connected_count=len(visited), connected_required=required,
                         first_failed_gate=None if passed else "component", passed=passed)
    return passed


def _model(pair: tuple[str, str], matches: tuple[tuple[int, int], ...],
           a: np.ndarray, b: np.ndarray, shape_a: tuple[int, int], shape_b: tuple[int, int],
           remaining: tuple[int, ...], seed: int,
           telemetry: dict[str, object] | None = None) -> PairModel | None:
    if telemetry is not None:
        telemetry.update(remaining_indices=list(remaining), remaining_count=len(remaining),
                         fit_indices=None, fit_count=None, held_indices=None, held_count=None,
                         homography_available=None, mask_available=None,
                         forward_normalized=None, inversion_available=None,
                         inverse_normalized=None,
                         forward_condition=None, inverse_condition=None,
                         fit_inlier_indices=None, fit_inlier_count=None,
                         held_rows=None, held_forward_pass_count=None,
                         held_reverse_pass_count=None, held_joint_pass_count=None,
                         held_forward_rejected_count=None, held_reverse_rejected_count=None,
                         held_joint_rejected_count=None, held_forward=None, held_reverse=None,
                         source_coverage=None, destination_coverage=None, first_rejection=None)
    def reject(reason: str) -> None:
        if telemetry is not None:
            telemetry["first_rejection"] = reason
        return None
    # The holdout assignment depends on identities/feature indices, never residuals.
    fit = tuple(i for i in remaining if _seed(pair[0], matches[i][0]) % 5 != 0)
    hold = tuple(i for i in remaining if _seed(pair[0], matches[i][0]) % 5 == 0)
    if telemetry is not None:
        telemetry.update(fit_indices=list(fit), fit_count=len(fit),
                         held_indices=list(hold), held_count=len(hold))
    if len(fit) < 16:
        return reject("fit_starved")
    if len(hold) < MIN_HOLDOUT:
        return reject("held_starved")
    src = np.array([a[matches[i][0]] for i in fit], dtype=np.float64)
    dst = np.array([b[matches[i][1]] for i in fit], dtype=np.float64)
    cv2.setRNGSeed(seed)
    raw, mask = cv2.findHomography(src, dst, cv2.RANSAC, TRANSFER_PX, maxIters=500, confidence=.995)
    if telemetry is not None:
        telemetry.update(homography_available=raw is not None,
                         mask_available=None if raw is None else mask is not None)
    if raw is None:
        return reject("homography_unavailable")
    if mask is None:
        return reject("mask_unavailable")
    forward_probe = {} if telemetry is not None else None
    try:
        h = _normalized(raw, forward_probe) if telemetry is not None else _normalized(raw)
    except (ValueError, np.linalg.LinAlgError):
        if telemetry is not None:
            telemetry.update(forward_normalized=False,
                             forward_condition=forward_probe.get("condition"))
        return reject("forward_normalization")
    if telemetry is not None:
        telemetry.update(forward_normalized=True,
                         forward_condition=forward_probe.get("condition"))
    try:
        inverse_raw = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        if telemetry is not None:
            telemetry["inversion_available"] = False
        return reject("inversion_unavailable")
    if telemetry is not None:
        telemetry["inversion_available"] = True
    inverse_probe = {} if telemetry is not None else None
    try:
        inv = _normalized(inverse_raw, inverse_probe) if telemetry is not None else _normalized(inverse_raw)
    except (ValueError, np.linalg.LinAlgError):
        if telemetry is not None:
            telemetry.update(inverse_normalized=False,
                             inverse_condition=inverse_probe.get("condition"))
        return reject("inverse_normalization")
    if telemetry is not None:
        telemetry.update(inverse_normalized=True,
                         inverse_condition=inverse_probe.get("condition"))
    fit_indices = tuple(i for i, flag in zip(fit, mask.ravel()) if flag)
    if telemetry is not None:
        telemetry.update(fit_inlier_indices=list(fit_indices), fit_inlier_count=len(fit_indices))
    if len(fit_indices) < 16:
        return reject("fit_inlier_starved")
    hold_src = np.array([a[matches[i][0]] for i in hold])
    hold_dst = np.array([b[matches[i][1]] for i in hold])
    ef = np.linalg.norm(_transfer(h, hold_src) - hold_dst, axis=1)
    er = np.linalg.norm(_transfer(inv, hold_dst) - hold_src, axis=1)
    good_hold = tuple(i for i, x, y in zip(hold, ef, er) if x <= TRANSFER_PX and y <= TRANSFER_PX)
    if telemetry is not None:
        telemetry.update(
            held_rows=[[i, "inf" if math.isinf(x) else float(x),
                        "inf" if math.isinf(y) else float(y)]
                       for i, x, y in zip(hold, ef, er)],
            held_forward_pass_count=int(np.sum(ef <= TRANSFER_PX)),
            held_reverse_pass_count=int(np.sum(er <= TRANSFER_PX)),
            held_joint_pass_count=len(good_hold),
            held_forward_rejected_count=int(np.sum(ef > TRANSFER_PX)),
            held_reverse_rejected_count=int(np.sum(er > TRANSFER_PX)),
            held_joint_rejected_count=len(hold) - len(good_hold),
            held_forward=_residual_summary(ef), held_reverse=_residual_summary(er))
    if len(good_hold) < MIN_HOLDOUT:
        return reject("joint_holdout_starved")
    if len(good_hold) / len(hold) < .38:
        return reject("holdout_fraction")
    all_good = fit_indices + good_hold
    source_coverage = {} if telemetry is not None else None
    if not _coverage(np.array([a[matches[i][0]] for i in all_good]), shape_a, source_coverage):
        if telemetry is not None:
            telemetry["source_coverage"] = source_coverage
        return reject("source_coverage")
    if telemetry is not None:
        telemetry["source_coverage"] = source_coverage
    destination_coverage = {} if telemetry is not None else None
    if not _coverage(np.array([b[matches[i][1]] for i in all_good]), shape_b, destination_coverage):
        if telemetry is not None:
            telemetry["destination_coverage"] = destination_coverage
        return reject("destination_coverage")
    if telemetry is not None:
        telemetry.update(destination_coverage=destination_coverage, first_rejection="accepted")
    return PairModel(pair, tuple(tuple(float(v) for v in row) for row in h), fit_indices,
                     good_hold, tuple(matches[i] for i in fit_indices),
                     tuple(matches[i] for i in all_good),
                     tuple((matches[i][0], matches[i][1],
                            float(hold_src[n, 0]), float(hold_src[n, 1]),
                            float(hold_dst[n, 0]), float(hold_dst[n, 1]), float(x), float(y))
                           for n, (i, x, y) in enumerate(zip(hold, ef, er))))


def _pair_models(pair: tuple[str, str], matches: tuple[tuple[int, int], ...],
                 a: np.ndarray, b: np.ndarray, shapes: tuple[tuple[int, int], tuple[int, int]],
                 telemetry: dict[str, object] | None = None) -> tuple[PairModel, ...]:
    remaining = tuple(range(len(matches)))
    models = []
    for number in range(MAX_MODELS_PER_PAIR):
        attempt = {"index": number} if telemetry is not None else None
        candidate = _model(pair, matches, a, b, *shapes, remaining, _seed(pair, number), attempt)
        if telemetry is not None:
            telemetry["attempts"].append(attempt)
        if candidate is None:
            break
        models.append(candidate)
        used = set(candidate.fit_indices + candidate.holdout_indices)
        remaining = tuple(i for i in remaining if i not in used)
    if telemetry is not None:
        telemetry.update(accepted_model_count=len(models),
                         termination=("model_limit_reached" if len(models) == MAX_MODELS_PER_PAIR else
                                      "first_attempt_rejected" if not models else "later_attempt_rejected"))
    return tuple(models)


def _track(a: str, b: str, c: str, ab: PairModel, bc: PairModel, ac: PairModel,
           points: dict[str, np.ndarray], shapes: dict[str, tuple[int, int]],
           hashes: dict[str, str]) -> dict[str, object] | None:
    h_ab, h_bc, h_ac = (np.array(model.transform) for model in (ab, bc, ac))
    # Common, same-identity feature triangles prevent combining unrelated planes.
    ab_map = dict(ab.support_keys)
    bc_set = set(bc.support_keys)
    ac_map = dict(ac.support_keys)
    triangles = sorted((ia, ib, ic) for ia, ib in ab_map.items()
                       for ic in (ac_map.get(ia),)
                       if ic is not None and (ib, ic) in bc_set)
    if len(triangles) < MIN_CYCLE_TRIPLES:
        return None
    # Composition is tested by transfer at observed points; projective matrices
    # can differ by a global scale and must not be compared entrywise.
    xyz = np.array([points[a][i] for i, _, _ in triangles])
    direct = _transfer(h_ac, xyz)
    composed = _transfer(h_bc, _transfer(h_ab, xyz))
    closure = np.linalg.norm(direct - composed, axis=1)
    if not np.all(np.isfinite(closure)) or np.percentile(closure, 95) > TRANSFER_PX:
        return None
    # Held-out triangles contain an edge not used to fit that edge's transform.
    fit_ab, fit_bc, fit_ac = (set(model.fit_keys) for model in (ab, bc, ac))
    held = [(i, j, k) for i, j, k in triangles
            if (i, j) not in fit_ab or (j, k) not in fit_bc or (i, k) not in fit_ac]
    if len(held) < MIN_CYCLE_TRIPLES:
        return None
    held_set = set(held)
    held_closure = closure[[row in held_set for row in triangles]]
    if np.percentile(held_closure, 95) > TRANSFER_PX:
        return None
    for frame, indices in ((a, [i for i, _, _ in triangles]),
                           (b, [j for _, j, _ in triangles]),
                           (c, [k for _, _, k in triangles])):
        if not _coverage(points[frame][indices], shapes[frame]):
            return None
    pair_models = (ab, bc, ac)
    feature_ids = {frame: set() for frame in (a, b, c)}
    for model in pair_models:
        left, right = model.pair
        for i, j in model.fit_keys:
            feature_ids[left].add(i)
            feature_ids[right].add(j)
        for i, j, *_ in model.holdout_rows:
            feature_ids[left].add(i)
            feature_ids[right].add(j)
    return {
        "frame_ids": [a, b, c],
        "frame_sha256": {x: hashes[x] for x in (a, b, c)},
        "transforms": {_pair_key(*model.pair): model.transform for model in pair_models},
        "feature_points": {frame: [[index, *(float(v) for v in points[frame][index])]
                                   for index in sorted(feature_ids[frame])]
                           for frame in (a, b, c)},
        "support": {a: [[float(v) for v in points[a][i]] for i, _, _ in triangles],
                    b: [[float(v) for v in points[b][j]] for _, j, _ in triangles],
                    c: [[float(v) for v in points[c][k]] for _, _, k in triangles]},
        "triangles": [list(row) for row in triangles],
        "triangle_is_held_out": [row in held_set for row in triangles],
        "triangle_count": len(triangles), "held_out_triangle_count": len(held),
        "cycle_all": _residual_summary(closure),
        "cycle_held_out": _residual_summary(held_closure),
        "pair_held_out": {_pair_key(*model.pair): _pair_holdout_report(model)
                          for model in pair_models},
    }


def diagnose(frames: Sequence[tuple[str, np.ndarray]]) -> dict[str, object]:
    """Return copied JSON-compatible report; invalid inputs raise ValueError.

    Frame IDs must be sorted. No API argument accepts camera or truth metadata.
    A missing coherent three-view plane returns typed abstention.
    """
    data, total_bytes = _frame_input(frames)
    names = [name for name, _, _ in data]
    hashes = {name: digest for name, _, digest in data}
    pairs = _pair_schedule(names)
    triangles = _bounded_cycle_triangles(names) if len(names) > LEGACY_MAX_FRAMES else _scheduled_triangles(names, pairs)
    extended = len(names) > LEGACY_MAX_FRAMES
    schema = "projective-plane-tracks-v6" if extended else "projective-plane-tracks-v4"
    points, descriptors, shapes = {}, {}, {}
    for name, image, _ in data:
        shapes[name] = image.shape[:2]
        points[name], descriptors[name] = _features(image)
    comparisons = 2 * sum(len(descriptors[a]) * len(descriptors[b])
                      for a, b in pairs)
    if comparisons > (EXTENDED_MAX_DESCRIPTOR_COMPARISONS if extended else MAX_DESCRIPTOR_COMPARISONS):
        raise ValueError("descriptor work budget exceeded")
    models = {}
    pair_telemetry = []
    for a, b in pairs:
        record = {"frame_ids": [a, b], "pair_key": _pair_key(a, b),
                  "left_feature_count": len(points[a]), "right_feature_count": len(points[b]),
                  "attempts": []}
        matches = _matches(descriptors[a], descriptors[b], record)
        models[a, b] = _pair_models((a, b), matches, points[a], points[b],
                                    (shapes[a], shapes[b]), record)
        pair_telemetry.append(record)
    tracks = []
    combinations = sum(len(models[a, b]) * len(models[b, c]) * len(models[a, c])
                       for a, b, c in triangles)
    if combinations > MAX_CYCLE_COMBINATIONS:
        raise ValueError("cycle work budget exceeded")
    for a, b, c in triangles:
        for ab, bc, ac in itertools.product(models[a, b], models[b, c], models[a, c]):
            candidate = _track(a, b, c, ab, bc, ac, points, shapes, hashes)
            if candidate is not None:
                tracks.append(candidate)
    # Preserve separate hypotheses, but deduplicate identical support.
    unique = {_digest_bytes(_canonical(x["support"])): x for x in tracks}
    result = {
        "schema": schema, "status": "supported" if unique else "abstained",
        "abstention": None if unique else "no_coherent_three_view_plane",
        "frame_ids": names, "frame_sha256": hashes,
        "frame_dimensions": {name: list(shapes[name]) for name in names},
        "frame_hash_scheme": "sha256(u32be(height)||u32be(width)||row-major-rgb-uint8)",
        "unresolved": list(UNRESOLVED),
        "work": {"input_rgb_bytes": total_bytes, "pair_count": len(models),
                 "descriptor_comparisons": comparisons,
                 "cycle_model_combinations": combinations,
                 "track_attempt_count": combinations,
                 "feature_counts": {name: len(points[name]) for name in names}},
        "pair_telemetry": pair_telemetry,
        "tracks": [unique[key] for key in sorted(unique)],
    }
    if extended:
        result["work"].update(pair_schedule=PAIR_SCHEDULE, omitted_pair_count=0)
    raw = _canonical(result)
    if len(raw) > _output_limit(result["schema"]):
        raise ValueError("output budget exceeded; refusing truncation")
    result["canonical_sha256"] = _digest_bytes(raw)
    if len(_canonical(result)) > _output_limit(result["schema"]):
        raise ValueError("output budget exceeded; refusing truncation")
    return result


def _validate_pair_telemetry(payload: dict[str, object], validate_summary: object
                             ) -> tuple[dict[tuple[str, str], int],
                                        dict[tuple[str, str], list[tuple[tuple[tuple[int, int], ...],
                                                                        tuple[tuple[object, ...], ...]]]]]:
    """Replay the bounded pair ledger, not SIFT or RANSAC provenance."""
    names = payload["frame_ids"]
    counts = payload["work"]["feature_counts"]
    records = payload["pair_telemetry"]
    pairs = (_legacy_cyclic_pair_schedule(names)
             if payload["schema"] == "projective-plane-tracks-v5" else _pair_schedule(names))
    if not isinstance(records, list) or len(records) != len(pairs):
        raise ValueError("malformed pair telemetry")
    accepted_counts = {}
    accepted_evidence = {}
    pair_fields = {"frame_ids", "pair_key", "left_feature_count", "right_feature_count",
                   "matching_status",
                   "forward_ratio_pairs", "reverse_ratio_pairs", "reciprocal_pairs",
                   "forward_ratio_pass_count", "reverse_ratio_pass_count", "reciprocal_count",
                   "attempts", "accepted_model_count", "termination"}
    attempt_fields = {"index", "remaining_indices", "remaining_count", "fit_indices", "fit_count",
                      "held_indices", "held_count", "homography_available", "mask_available",
                      "forward_normalized", "inversion_available", "inverse_normalized",
                      "forward_condition", "inverse_condition",
                      "fit_inlier_indices", "fit_inlier_count", "held_rows",
                      "held_forward_pass_count", "held_reverse_pass_count", "held_joint_pass_count",
                      "held_forward_rejected_count", "held_reverse_rejected_count",
                      "held_joint_rejected_count", "held_forward", "held_reverse",
                      "source_coverage", "destination_coverage", "first_rejection"}
    after_partition = attempt_fields - {"index", "remaining_indices", "remaining_count",
                                        "fit_indices", "fit_count", "held_indices", "held_count",
                                        "first_rejection"}
    after_homography = after_partition - {"homography_available", "mask_available"}
    after_forward = after_homography - {"forward_normalized", "forward_condition"}
    after_inversion = after_forward - {"inversion_available"}
    after_inverse = after_inversion - {"inverse_normalized", "inverse_condition"}
    after_inliers = after_inverse - {"fit_inlier_indices", "fit_inlier_count"}
    held_fields = {"held_rows", "held_forward_pass_count", "held_reverse_pass_count",
                   "held_joint_pass_count", "held_forward_rejected_count",
                   "held_reverse_rejected_count", "held_joint_rejected_count",
                   "held_forward", "held_reverse"}

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def null_fields(row: dict[str, object], fields: set[str]) -> None:
        require(all(row[field] is None for field in fields), "malformed nullable attempt stages")

    def index_pairs(rows: object, left_count: int, right_count: int) -> list[tuple[int, int]]:
        require(isinstance(rows, list), "malformed ratio identities")
        require(all(isinstance(row, list) and len(row) == 2 and
                    all(type(v) is int for v in row) and
                    0 <= row[0] < left_count and 0 <= row[1] < right_count for row in rows),
                "malformed ratio identities")
        values = [tuple(row) for row in rows]
        require(values == sorted(values) and len({i for i, _ in values}) == len(values),
                "malformed ratio identities")
        return values

    def coverage(row: object, frame: str, count: int) -> bool:
        fields = {"count", "extent_xy", "extent_area", "grid_cell_count", "connected_count",
                  "connected_required", "first_failed_gate", "passed"}
        require(isinstance(row, dict) and set(row) == fields and
                type(row["count"]) is int and row["count"] == count and
                type(row["passed"]) is bool, "malformed coverage telemetry")
        h, w = payload["frame_dimensions"][frame]
        later = ("extent_xy", "extent_area", "grid_cell_count", "connected_count", "connected_required")
        if count < 12:
            require(all(row[field] is None for field in later) and
                    row["first_failed_gate"] == "too_few_points" and not row["passed"],
                    "malformed coverage precedence")
            return False
        xy = row["extent_xy"]
        require(isinstance(xy, list) and len(xy) == 2 and
                all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in xy) and
                xy[0] < w and xy[1] < h and
                all(type(row[field]) in (int, float) and math.isfinite(row[field])
                    for field in ("extent_area",)) and
                math.isclose(row["extent_area"], xy[0] * xy[1], rel_tol=1e-9, abs_tol=1e-8),
                "malformed coverage extent")
        extent_good = xy[0] >= w * .18 and xy[1] >= h * .18 and xy[0] * xy[1] >= w * h * .055
        if not extent_good:
            require(row["grid_cell_count"] is None and row["connected_count"] is None and
                    row["connected_required"] is None and row["first_failed_gate"] == "extent" and
                    not row["passed"], "malformed coverage precedence")
            return False
        cells = row["grid_cell_count"]
        require(type(cells) is int and 0 <= cells <= min(16, count), "malformed coverage cells")
        if cells < 4:
            require(row["connected_count"] is None and row["connected_required"] is None and
                    row["first_failed_gate"] == "grid" and not row["passed"],
                    "malformed coverage precedence")
            return False
        connected = row["connected_count"]
        required = math.ceil(.9 * count)
        require(type(connected) is int and 1 <= connected <= count and
                type(row["connected_required"]) is int and row["connected_required"] == required and
                row["passed"] == (connected >= required) and
                row["first_failed_gate"] == (None if row["passed"] else "component"),
                "malformed coverage component")
        return row["passed"]

    for pair, record in zip(pairs, records):
        left, right = pair
        require(isinstance(record, dict) and set(record) == pair_fields and
                record["frame_ids"] == [left, right] and record["pair_key"] == _pair_key(left, right) and
                record["left_feature_count"] == counts[left] and
                record["right_feature_count"] == counts[right], "malformed pair identity")
        forward = index_pairs(record["forward_ratio_pairs"], counts[left], counts[right])
        reverse = index_pairs(record["reverse_ratio_pairs"], counts[right], counts[left])
        reciprocal = index_pairs(record["reciprocal_pairs"], counts[left], counts[right])
        reverse_map = dict(reverse)
        require(reciprocal == sorted((i, j) for i, j in forward if reverse_map.get(j) == i) and
                len({j for _, j in reciprocal}) == len(reciprocal) and
                type(record["forward_ratio_pass_count"]) is int and
                record["forward_ratio_pass_count"] == len(forward) and
                type(record["reverse_ratio_pass_count"]) is int and
                record["reverse_ratio_pass_count"] == len(reverse) and
                type(record["reciprocal_count"]) is int and
                record["reciprocal_count"] == len(reciprocal) <= MAX_MATCHES_PER_PAIR and
                (min(counts[left], counts[right]) >= 2 or not (forward or reverse or reciprocal)),
                "malformed reciprocal accounting")
        require(record["matching_status"] == (
            "evaluated" if min(counts[left], counts[right]) >= 2 else "insufficient_descriptors"),
            "malformed matcher reach")
        attempts = record["attempts"]
        require(isinstance(attempts, list) and 1 <= len(attempts) <= MAX_MODELS_PER_PAIR,
                "malformed attempt count")
        remaining = list(range(len(reciprocal)))
        accepted = 0
        evidence = []
        for number, attempt in enumerate(attempts):
            require(isinstance(attempt, dict) and set(attempt) == attempt_fields and
                    type(attempt["index"]) is int and attempt["index"] == number and
                    attempt["remaining_indices"] == remaining and
                    type(attempt["remaining_count"]) is int and
                    attempt["remaining_count"] == len(remaining), "malformed remaining identities")
            fit = [i for i in remaining if _seed(left, reciprocal[i][0]) % 5 != 0]
            held = [i for i in remaining if _seed(left, reciprocal[i][0]) % 5 == 0]
            require(attempt["fit_indices"] == fit and attempt["held_indices"] == held and
                    type(attempt["fit_count"]) is int and attempt["fit_count"] == len(fit) and
                    type(attempt["held_count"]) is int and attempt["held_count"] == len(held),
                    "malformed deterministic attempt partition")
            reason = attempt["first_rejection"]
            if len(fit) < 16 or len(held) < MIN_HOLDOUT:
                require(reason == ("fit_starved" if len(fit) < 16 else "held_starved"),
                        "malformed partition first exit")
                null_fields(attempt, after_partition)
            else:
                raw = attempt["homography_available"]
                mask = attempt["mask_available"]
                require(type(raw) is bool and
                        (mask is None if not raw else type(mask) is bool),
                        "malformed homography reach")
                if not raw or not mask:
                    require(reason == ("homography_unavailable" if not raw else "mask_unavailable"),
                            "malformed homography first exit")
                    null_fields(attempt, after_homography)
                else:
                    forward_ok = attempt["forward_normalized"]
                    require(type(forward_ok) is bool, "malformed normalization reach")
                    for status, condition in ((forward_ok, attempt["forward_condition"]),
                                              (attempt["inverse_normalized"], attempt["inverse_condition"])):
                        if status is not None:
                            require(condition is None or condition == "inf" or
                                    (type(condition) in (int, float) and math.isfinite(condition) and condition >= 1),
                                    "malformed transform condition")
                            if status:
                                require(condition is not None and condition != "inf" and condition <= 1e8,
                                        "malformed accepted transform condition")
                    if not forward_ok:
                        require(reason == "forward_normalization", "malformed forward first exit")
                        null_fields(attempt, after_forward)
                    else:
                        inversion_ok = attempt["inversion_available"]
                        require(type(inversion_ok) is bool, "malformed inversion reach")
                        if not inversion_ok:
                            require(reason == "inversion_unavailable", "malformed inversion first exit")
                            null_fields(attempt, after_inversion)
                            require(number == len(attempts) - 1, "attempt after first rejection")
                            continue
                        inverse_ok = attempt["inverse_normalized"]
                        require(type(inverse_ok) is bool, "malformed inverse reach")
                        if not inverse_ok:
                            require(reason == "inverse_normalization", "malformed inverse first exit")
                            null_fields(attempt, after_inverse)
                        else:
                            inliers = attempt["fit_inlier_indices"]
                            require(isinstance(inliers, list) and
                                    all(type(i) is int for i in inliers) and
                                    inliers == sorted(set(inliers)) and set(inliers) <= set(fit) and
                                    type(attempt["fit_inlier_count"]) is int and
                                    attempt["fit_inlier_count"] == len(inliers),
                                    "malformed fit inlier subset")
                            if len(inliers) < 16:
                                require(reason == "fit_inlier_starved", "malformed inlier first exit")
                                null_fields(attempt, after_inliers)
                            else:
                                rows = attempt["held_rows"]
                                require(isinstance(rows, list) and len(rows) == len(held) and
                                        all(isinstance(row, list) and len(row) == 3 and
                                            type(row[0]) is int and row[0] == held[i] and
                                            all(value == "inf" or
                                                (type(value) in (int, float) and math.isfinite(value) and value >= 0)
                                                for value in row[1:]) for i, row in enumerate(rows)),
                                        "malformed held residual rows")
                                ef = np.asarray([math.inf if row[1] == "inf" else row[1] for row in rows])
                                er = np.asarray([math.inf if row[2] == "inf" else row[2] for row in rows])
                                fpass = ef <= TRANSFER_PX
                                rpass = er <= TRANSFER_PX
                                both = fpass & rpass
                                expected = {"held_forward_pass_count": int(sum(fpass)),
                                            "held_reverse_pass_count": int(sum(rpass)),
                                            "held_joint_pass_count": int(sum(both)),
                                            "held_forward_rejected_count": int(sum(~fpass)),
                                            "held_reverse_rejected_count": int(sum(~rpass)),
                                            "held_joint_rejected_count": int(sum(~both))}
                                require(all(type(attempt[key]) is int and attempt[key] == value
                                            for key, value in expected.items()),
                                        "malformed held counts")
                                validate_summary(attempt["held_forward"], ef)
                                validate_summary(attempt["held_reverse"], er)
                                joint = expected["held_joint_pass_count"]
                                if joint < MIN_HOLDOUT or joint / len(held) < .38:
                                    require(reason == ("joint_holdout_starved" if joint < MIN_HOLDOUT else
                                                       "holdout_fraction"), "malformed holdout first exit")
                                    null_fields(attempt, {"source_coverage", "destination_coverage"})
                                else:
                                    supported_count = len(inliers) + joint
                                    src_ok = coverage(attempt["source_coverage"], left, supported_count)
                                    if not src_ok:
                                        require(reason == "source_coverage", "malformed source coverage exit")
                                        null_fields(attempt, {"destination_coverage"})
                                    else:
                                        dst_ok = coverage(attempt["destination_coverage"], right, supported_count)
                                        require(reason == ("accepted" if dst_ok else "destination_coverage"),
                                                "malformed destination coverage exit")
                                    if reason == "accepted":
                                        accepted += 1
                                        evidence.append((
                                            tuple(reciprocal[i] for i in inliers),
                                            tuple((*reciprocal[row[0]], row[1], row[2]) for row in rows)))
                                        used = set(inliers) | {held[i] for i, good in enumerate(both) if good}
                                        remaining = [i for i in remaining if i not in used]
            if reason != "accepted":
                require(number == len(attempts) - 1, "attempt after first rejection")
            else:
                require(number < MAX_MODELS_PER_PAIR - 1 or len(attempts) == MAX_MODELS_PER_PAIR,
                        "malformed model attempt bound")
        termination = ("model_limit_reached" if accepted == MAX_MODELS_PER_PAIR else
                       "first_attempt_rejected" if accepted == 0 else "later_attempt_rejected")
        require(type(record["accepted_model_count"]) is int and
                record["accepted_model_count"] == accepted and record["termination"] == termination and
                (len(attempts) == MAX_MODELS_PER_PAIR or attempts[-1]["first_rejection"] != "accepted"),
                "malformed model termination")
        accepted_counts[pair] = accepted
        accepted_evidence[pair] = evidence
    return accepted_counts, accepted_evidence


def canonical_report_bytes(report: dict[str, object]) -> bytes:
    """Validate the report digest and return canonical bytes for independent replay."""
    if not isinstance(report, dict) or not isinstance(report.get("canonical_sha256"), str):
        raise ValueError("malformed report")
    payload = dict(report)
    digest = payload.pop("canonical_sha256")
    if _digest_bytes(_canonical(payload)) != digest:
        raise ValueError("report digest mismatch")
    if set(payload) != {"schema", "status", "abstention", "frame_ids", "frame_sha256", "frame_dimensions", "frame_hash_scheme",
                        "unresolved", "work", "pair_telemetry", "tracks"}:
        raise ValueError("malformed report fields")
    names = payload["frame_ids"]
    tracks = payload["tracks"]
    extended = payload["schema"] in ("projective-plane-tracks-v5", "projective-plane-tracks-v6")
    if (payload["schema"] not in ("projective-plane-tracks-v4", "projective-plane-tracks-v5", "projective-plane-tracks-v6") or
            payload["frame_hash_scheme"] != "sha256(u32be(height)||u32be(width)||row-major-rgb-uint8)" or
            not isinstance(names, list) or not 3 <= len(names) <= MAX_FRAMES or
            extended != (len(names) > LEGACY_MAX_FRAMES) or
            not all(isinstance(x, str) and 1 <= len(x) <= 100 and x.isprintable() for x in names) or
            names != sorted(set(names)) or
            not isinstance(payload["frame_sha256"], dict) or
            set(payload["frame_sha256"]) != set(names) or
            not all(isinstance(x, str) and re.fullmatch("[0-9a-f]{64}", x)
                    for x in payload["frame_sha256"].values()) or
            not isinstance(payload["frame_dimensions"], dict) or
            set(payload["frame_dimensions"]) != set(names) or
            any(not isinstance(dims, list) or len(dims) != 2 or
                any(type(v) is not int or v < 64 for v in dims) or
                dims[0] * dims[1] > MAX_PIXELS_PER_FRAME
                for dims in payload["frame_dimensions"].values()) or
            payload["unresolved"] != list(UNRESOLVED) or not isinstance(tracks, list)):
        raise ValueError("malformed report binding")
    if (payload["status"], payload["abstention"]) != (
            ("supported", None) if tracks else ("abstained", "no_coherent_three_view_plane")):
        raise ValueError("malformed report status")
    pairs = (_legacy_cyclic_pair_schedule(names) if payload["schema"] == "projective-plane-tracks-v5"
             else _pair_schedule(names))
    triangles = (_bounded_cycle_triangles(names) if extended
                 else _scheduled_triangles(names, pairs))
    work = payload["work"]
    schedule_fields = {"pair_schedule", "omitted_pair_count"} if extended else set()
    if (not isinstance(work, dict) or set(work) != {"input_rgb_bytes", "pair_count", "feature_counts",
                                                  "descriptor_comparisons", "cycle_model_combinations",
                                                  "track_attempt_count"} | schedule_fields or
            not isinstance(work["input_rgb_bytes"], int) or
            not 0 < work["input_rgb_bytes"] <= (EXTENDED_MAX_TOTAL_RGB_BYTES if extended else MAX_TOTAL_RGB_BYTES) or
            type(work["pair_count"]) is not int or work["pair_count"] != len(pairs) or len(pairs) > MAX_PAIRS or
            not isinstance(work["descriptor_comparisons"], int) or
            not 0 <= work["descriptor_comparisons"] <= (EXTENDED_MAX_DESCRIPTOR_COMPARISONS if extended else MAX_DESCRIPTOR_COMPARISONS) or
            not isinstance(work["cycle_model_combinations"], int) or
            not 0 <= work["cycle_model_combinations"] <= MAX_CYCLE_COMBINATIONS or
            type(work["track_attempt_count"]) is not int or
            work["track_attempt_count"] != work["cycle_model_combinations"] or
            not isinstance(work["feature_counts"], dict) or
            set(work["feature_counts"]) != set(names) or
            any(not isinstance(v, int) or not 0 <= v <= MAX_FEATURES
                for v in work["feature_counts"].values())):
        raise ValueError("malformed work accounting")
    expected_schedule = LEGACY_PAIR_SCHEDULE if payload["schema"] == "projective-plane-tracks-v5" else PAIR_SCHEDULE
    expected_omitted = len(names) * (len(names) - 1) // 2 - len(pairs)
    if extended and (work["pair_schedule"] != expected_schedule or
                     type(work["omitted_pair_count"]) is not int or
                     work["omitted_pair_count"] != expected_omitted):
        raise ValueError("malformed pair schedule accounting")
    expected_comparisons = 2 * sum(work["feature_counts"][a] * work["feature_counts"][b]
                                   for a, b in pairs)
    if work["descriptor_comparisons"] != expected_comparisons:
        raise ValueError("malformed descriptor work accounting")
    if work["input_rgb_bytes"] != 3 * sum(h * w for h, w in payload["frame_dimensions"].values()):
        raise ValueError("malformed input byte accounting")

    def valid_point(frame: str, xy: Sequence[object]) -> bool:
        h, w = payload["frame_dimensions"][frame]
        return (len(xy) == 2 and all(type(v) in (float, int) and math.isfinite(v) for v in xy)
                and 0 <= xy[0] < w and 0 <= xy[1] < h)

    def same_point(left: Sequence[float], right: Sequence[float]) -> bool:
        # JSON decimal roundtrips the emitted floats; this tolerance only absorbs
        # the existing residual/serialization rounding allowance.
        return len(left) == len(right) == 2 and all(math.isclose(x, y, rel_tol=0, abs_tol=1e-8)
                   for x, y in zip(left, right))

    def validate_summary(row: object, values: np.ndarray) -> None:
        expected = _residual_summary(values)
        if (not isinstance(row, dict) or set(row) != set(expected) or
                type(row["count"]) is not int or row["count"] != len(values) or
                any((row[key] != "inf" if expected[key] == "inf" else
                     type(row[key]) not in (float, int) or not math.isfinite(row[key]) or
                     not math.isclose(row[key], expected[key], rel_tol=1e-9, abs_tol=1e-8))
                    for key in ("p50_px", "p95_px", "max_px"))):
            raise ValueError("malformed residual summary")

    accepted_counts, accepted_evidence = _validate_pair_telemetry(payload, validate_summary)
    expected_combinations = sum(accepted_counts[a, b] * accepted_counts[b, c] * accepted_counts[a, c]
                                for a, b, c in triangles)
    if work["cycle_model_combinations"] != expected_combinations:
        raise ValueError("malformed accepted pair graph")

    for track in tracks:
        if not isinstance(track, dict) or set(track) != {
                "frame_ids", "frame_sha256", "transforms", "feature_points", "support", "triangle_count",
                "held_out_triangle_count", "triangles", "triangle_is_held_out",
                "cycle_all", "cycle_held_out", "pair_held_out"}:
            raise ValueError("malformed track")
        ids = track["frame_ids"]
        if (not isinstance(ids, list) or len(ids) != 3 or ids != sorted(set(ids)) or
                not set(ids) <= set(names) or
                track["frame_sha256"] != {x: payload["frame_sha256"][x] for x in ids} or
                not isinstance(track["triangle_count"], int) or
                track["triangle_count"] < MIN_CYCLE_TRIPLES or
                not isinstance(track["held_out_triangle_count"], int) or
                not MIN_CYCLE_TRIPLES <= track["held_out_triangle_count"] <= track["triangle_count"]):
            raise ValueError("malformed track support")
        a, b, c = ids
        if any(accepted_counts.get(pair, 0) == 0 for pair in ((a, b), (b, c), (a, c))):
            raise ValueError("track lacks an accepted pair model")
        pairs = ((a, b), (b, c), (a, c))
        pair_keys = {_pair_key(*pair): pair for pair in pairs}
        keys = set(pair_keys)
        if (not isinstance(track["transforms"], dict) or set(track["transforms"]) != keys or
                not isinstance(track["feature_points"], dict) or set(track["feature_points"]) != set(ids) or
                not isinstance(track["support"], dict) or set(track["support"]) != set(ids) or
                not isinstance(track["pair_held_out"], dict) or set(track["pair_held_out"]) != keys):
            raise ValueError("malformed track fields")
        feature_points = {}
        for frame in ids:
            rows = track["feature_points"][frame]
            if (not isinstance(rows, list) or any(
                    not isinstance(row, list) or len(row) != 3 or type(row[0]) is not int or
                    not 0 <= row[0] < work["feature_counts"][frame] or
                    not valid_point(frame, row[1:]) for row in rows) or
                    [row[0] for row in rows] != sorted(set(row[0] for row in rows))):
                raise ValueError("malformed feature points")
            feature_points[frame] = {row[0]: row[1:] for row in rows}
        for matrix in track["transforms"].values():
            try:
                _normalized(np.asarray(matrix, dtype=np.float64))
            except (TypeError, ValueError, np.linalg.LinAlgError):
                raise ValueError("malformed track transform") from None
        for frame, support in track["support"].items():
            if (not isinstance(support, list) or len(support) != track["triangle_count"] or
                    any(not isinstance(p, list) or not valid_point(frame, p)
                        for p in support)):
                raise ValueError("malformed track points")
        pair_fit = {}
        pair_supported = {}
        observed_feature_ids = {frame: set() for frame in ids}
        for key, row in track["pair_held_out"].items():
            if (not isinstance(row, dict) or set(row) != {
                    "fit_keys", "held_out_rows", "count", "joint_accepted_count",
                    "forward_rejected_count", "reverse_rejected_count", "joint_rejected_count",
                    "forward", "reverse"} or
                    not isinstance(row["fit_keys"], list) or
                    not isinstance(row["held_out_rows"], list)):
                raise ValueError("malformed held-out residual")
            left, right = pair_keys[key]
            def valid_key(item: object) -> bool:
                return (isinstance(item, list) and len(item) == 2 and
                        all(type(v) is int for v in item) and
                        0 <= item[0] < work["feature_counts"][left] and
                        0 <= item[1] < work["feature_counts"][right])
            fit = [tuple(item) for item in row["fit_keys"] if valid_key(item)]
            held_rows = row["held_out_rows"]
            if (len(fit) != len(row["fit_keys"]) or len(fit) < 16 or
                    fit != sorted(set(fit)) or
                    any(not isinstance(item, list) or len(item) != 8 or
                        not valid_key(item[:2]) or
                        any(type(v) not in (float, int) or not math.isfinite(v)
                            for v in item[2:6]) or
                        any(v != "inf" and (type(v) not in (float, int) or
                            not math.isfinite(v) or v < 0)
                            for v in item[6:]) for item in held_rows)):
                raise ValueError("malformed held-out identities")
            held_keys = [tuple(item[:2]) for item in held_rows]
            if held_keys != sorted(set(held_keys)) or set(fit) & set(held_keys):
                raise ValueError("malformed held-out identities")
            if (any(_seed(left, i) % 5 == 0 for i, _ in fit) or
                    any(_seed(left, i) % 5 != 0 for i, _ in held_keys)):
                raise ValueError("malformed deterministic fit/holdout partition")
            all_keys = fit + held_keys
            if (len({i for i, _ in all_keys}) != len(all_keys) or
                    len({j for _, j in all_keys}) != len(all_keys)):
                raise ValueError("malformed reciprocal pair identities")
            for i, j in all_keys:
                observed_feature_ids[left].add(i)
                observed_feature_ids[right].add(j)
            for item in held_rows:
                i, j = item[:2]
                if (not valid_point(left, item[2:4]) or
                        not valid_point(right, item[4:6]) or
                        not same_point(item[2:4], feature_points[left].get(i, ())) or
                        not same_point(item[4:6], feature_points[right].get(j, ()))):
                    raise ValueError("malformed held-out feature coordinates")
            ef = np.array([math.inf if item[6] == "inf" else item[6] for item in held_rows])
            er = np.array([math.inf if item[7] == "inf" else item[7] for item in held_rows])
            if held_rows:
                src = np.asarray([item[2:4] for item in held_rows], dtype=np.float64)
                dst = np.asarray([item[4:6] for item in held_rows], dtype=np.float64)
                h = np.asarray(track["transforms"][key], dtype=np.float64)
                try:
                    inverse = np.linalg.inv(h)
                except np.linalg.LinAlgError:
                    raise ValueError("malformed held-out transform") from None
                replay_f = np.linalg.norm(_transfer(h, src) - dst, axis=1)
                replay_r = np.linalg.norm(_transfer(inverse, dst) - src, axis=1)
                if (not np.allclose(ef, replay_f, rtol=1e-9, atol=1e-8) or
                        not np.allclose(er, replay_r, rtol=1e-9, atol=1e-8)):
                    raise ValueError("malformed held-out residual replay")
            both = (ef <= TRANSFER_PX) & (er <= TRANSFER_PX)
            if (type(row["count"]) is not int or row["count"] != len(held_rows) or
                    row["count"] < MIN_HOLDOUT or
                    type(row["joint_accepted_count"]) is not int or
                    row["joint_accepted_count"] != int(np.sum(both)) or
                    row["joint_accepted_count"] < MIN_HOLDOUT or
                    row["joint_accepted_count"] / row["count"] < .38 or
                    type(row["forward_rejected_count"]) is not int or
                    row["forward_rejected_count"] != int(np.sum(ef > TRANSFER_PX)) or
                    type(row["reverse_rejected_count"]) is not int or
                    row["reverse_rejected_count"] != int(np.sum(er > TRANSFER_PX)) or
                    type(row["joint_rejected_count"]) is not int or
                    row["joint_rejected_count"] != int(np.sum(~both))):
                raise ValueError("malformed held-out counts")
            validate_summary(row["forward"], ef)
            validate_summary(row["reverse"], er)
            def same_residual(left_value: object, right_value: object) -> bool:
                return (left_value == right_value == "inf" or
                        type(left_value) in (int, float) and type(right_value) in (int, float) and
                        math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-8))
            actual_held = tuple((*item[:2], *item[6:8]) for item in held_rows)
            matching_attempts = [
                model for model in accepted_evidence[left, right]
                if model[0] == tuple(fit) and len(model[1]) == len(actual_held) and
                all(expected[:2] == actual[:2] and
                    same_residual(expected[2], actual[2]) and
                    same_residual(expected[3], actual[3])
                    for expected, actual in zip(model[1], actual_held))]
            if len(matching_attempts) != 1:
                raise ValueError("track pair contradicts accepted attempt telemetry")
            pair_fit[key] = set(fit)
            pair_supported[key] = pair_fit[key] | {held_keys[i] for i, good in enumerate(both) if good}
        if any(set(feature_points[frame]) != observed_feature_ids[frame] for frame in ids):
            raise ValueError("malformed feature point identities")
        triangles = track["triangles"]
        flags = track["triangle_is_held_out"]
        if (not isinstance(triangles, list) or not isinstance(flags, list) or
                len(triangles) != track["triangle_count"] or len(flags) != len(triangles) or
                any(not isinstance(row, list) or len(row) != 3 or
                    any(type(v) is not int or not 0 <= v < work["feature_counts"][ids[i]]
                        for i, v in enumerate(row)) for row in triangles) or
                [tuple(row) for row in triangles] != sorted(set(tuple(row) for row in triangles)) or
                any(type(flag) is not bool for flag in flags) or
                sum(flags) != track["held_out_triangle_count"]):
            raise ValueError("malformed triangle partition")
        ab, bc, ac = (_pair_key(*pair) for pair in pairs)
        bc_edges = pair_supported[bc]
        ac_map = dict(pair_supported[ac])
        expected_triangles = sorted((i, j, ac_map[i]) for i, j in pair_supported[ab]
                                    if i in ac_map and (j, ac_map[i]) in bc_edges)
        if [tuple(row) for row in triangles] != expected_triangles:
            raise ValueError("incomplete triangle replay")
        for n, ((i, j, k), flag) in enumerate(zip(expected_triangles, flags)):
            edges = {ab: (i, j), bc: (j, k), ac: (i, k)}
            if flag != any(edge not in pair_fit[key] for key, edge in edges.items()):
                raise ValueError("malformed triangle partition")
            for frame, index in ((a, i), (b, j), (c, k)):
                if not same_point(track["support"][frame][n], feature_points[frame][index]):
                    raise ValueError("malformed triangle feature coordinates")
        xyz = np.asarray(track["support"][a], dtype=np.float64)
        h_ab, h_bc, h_ac = (np.asarray(track["transforms"][key], dtype=np.float64)
                             for key in (ab, bc, ac))
        closure = np.linalg.norm(_transfer(h_ac, xyz) -
                                 _transfer(h_bc, _transfer(h_ab, xyz)), axis=1)
        if not np.all(np.isfinite(closure)):
            raise ValueError("malformed cycle residual")
        validate_summary(track["cycle_all"], closure)
        validate_summary(track["cycle_held_out"], closure[flags])
        if (track["cycle_all"]["p95_px"] > TRANSFER_PX or
                track["cycle_held_out"]["p95_px"] > TRANSFER_PX):
            raise ValueError("malformed cycle residual")
    raw = _canonical(report)
    if len(raw) > _output_limit(payload["schema"]):
        raise ValueError("output budget exceeded")
    return raw


def verify_frame_binding(report: dict[str, object], frames: Sequence[tuple[str, np.ndarray]]) -> None:
    """Independently rehash the RGB input without running feature extraction."""
    canonical_report_bytes(report)
    data, _ = _frame_input(frames)
    if (report.get("frame_ids") != [name for name, _, _ in data] or report.get("frame_sha256") != {
        name: digest for name, _, digest in data
    } or report.get("frame_dimensions") != {
        name: list(image.shape[:2]) for name, image, _ in data
    }):
        raise ValueError("frame source hash or identity drift")


def persist_report(path: str | Path, report: dict[str, object]) -> None:
    """Atomically publish a validated report without clobbering an existing path."""
    raw = canonical_report_bytes(report)
    target = Path(path)
    if not target.parent.is_dir():
        raise ValueError("output parent does not exist")
    fd, temporary = tempfile.mkstemp(prefix=".plane-tracks-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            if out.write(raw) != len(raw):
                raise OSError("partial report write")
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, target)  # atomic create; FileExistsError preserves target
    finally:
        os.unlink(temporary)
