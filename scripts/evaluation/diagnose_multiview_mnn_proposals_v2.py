"""Synthetic-only paired SIFT proposal policies, v2 development repair.

Additive to the frozen v1 module: the policy semantics, geometry calls, the
700-feature hard refusal, per-image/pair/work/artifact limits, truth isolation,
deterministic order and atomic publication are carried over unchanged. Only the
report schema/pins, the independent scene-group seeds and bounded-complexity
fixture generator, and the development-only calibration entry point differ.
No image loader, no held-out generation and no truth enters diagnosis.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import cv2
import numpy as np

from scripts.evaluation import diagnose_multiview_matcher_ablation as ablation
from scripts.evaluation import diagnose_multiview_plane_tracks as geometry

ABLATION_SHA256 = "8907fa39bbdf7e2b04960fa8c5565757dc94227502c6272038823cc838d8b02e"
GEOMETRY_SHA256 = "f69be2ef78e8ae90a2bad09f973ed1ce3ec37d1a3a99f77101ecb68f9b2291b0"
V1_SOURCE_SHA256 = "6b4437e60a0739673ddb6b70574b03d63866af9cbae6607b8a639b020423ca40"
V1_TEST_SHA256 = "7765d1d99bb61c57a2f14b30726cb11dddff142db10e10bec009f85c17ca76f0"
SCHEMA = "opencv-mnn-proposals-v2"
CALIBRATION_SCHEMA = "opencv-mnn-synthetic-calibration-v2"
POLICIES = ("reciprocal-lowe-085", "strict-unique-mutual-nearest")
TIE_RULE = "first_distance_strictly_less_than_second; exact_ties_abstain"
ADAPTER = "geometry ratio fields encode policy directional survivors; outer nearest witness defines their meaning"
FEATURE_CAP = geometry.MAX_FEATURES
MAX_CASES = 64
MAX_DEV_CASES = 32
MAX_SHARED_COMPARISONS = 188_160_000
MAX_ARTIFACT_BYTES = 96_000_000
AUTHORIZED_CALIBRATION_SPLITS = ("dev",)
FIXTURE_FAMILY = "v2-bounded-texture-192"
VARIANT = ablation.VARIANTS[3]
# Independent v2 scene-group seeds; no dev or held-out seed is reused from v1.
SEEDS = (("dev", "weak-contrast", 83011), ("dev", "repeated-brick-tile", 83021),
         ("dev", "mixed-scale-pixel", 83031), ("dev", "sparse-corners", 83041),
         ("heldout", "weak-contrast", 93011), ("heldout", "repeated-brick-tile", 93021),
         ("heldout", "mixed-scale-pixel", 93031), ("heldout", "sparse-corners", 93041))
CASE_KINDS = ("viewpoint", "brightness", "noise", "occlusion", "palette-nonoverlap",
              "incompatible-tiles", "multi-plane-corner", "disconnected-fragments")
GATES = {"positives_per_family_min": 3, "accepted_precision_min": .95,
         "pure_negative_accepted_pairs_max": 0, "invalid_structural_support_max": 0,
         "positive_support_losses_max": 0, "pooled_true_recall_gain": "strict",
         "partition_eligible_pair_gain": "strict", "correct_geometry_pair_min": 1}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def configuration() -> dict:
    return dict(schema=SCHEMA, policies=list(POLICIES), detector="existing-low-contrast-sift-0.01-700",
                ratio=.85, tie_rule=TIE_RULE, geometry_adapter=ADAPTER,
                ablation_source_sha256=ABLATION_SHA256, geometry_source_sha256=GEOMETRY_SHA256,
                v1_source_sha256=V1_SOURCE_SHA256, v1_test_sha256=V1_TEST_SHA256,
                seeds=[list(row) for row in SEEDS], case_kinds=list(CASE_KINDS), gates=dict(GATES),
                feature_cap=FEATURE_CAP, fixture_family=FIXTURE_FAMILY,
                authorized_calibration_splits=list(AUTHORIZED_CALIBRATION_SPLITS),
                max_cases=MAX_CASES, max_dev_cases=MAX_DEV_CASES,
                max_shared_comparisons=MAX_SHARED_COMPARISONS,
                max_artifact_bytes=MAX_ARTIFACT_BYTES)


CONFIG_SHA256 = _sha(_canonical(configuration()))


def _check_pins() -> None:
    if (_sha(Path(ablation.__file__).read_bytes()) != ABLATION_SHA256 or
            _sha(Path(geometry.__file__).read_bytes()) != GEOMETRY_SHA256 or
            _sha(_canonical(configuration())) != CONFIG_SHA256):
        raise ValueError("source or configuration pin drift")


def _enforce_feature_cap(retained: object, detected: object) -> None:
    """Hard 700-feature refusal; never truncate over-budget SIFT rows."""
    if type(detected) is not int or type(retained) is not int:
        raise ValueError("malformed feature counts")
    if detected > FEATURE_CAP or retained > FEATURE_CAP:
        raise ValueError("feature budget exceeded before scoring; refusing truncation")
    if retained > detected:
        raise ValueError("retained feature count exceeds detected count")


def _extract(image: np.ndarray, variant: tuple) -> tuple[np.ndarray, np.ndarray, int]:
    points, descriptors, detected = ablation._extract(image, variant)
    _enforce_feature_cap(len(points), detected)
    return points, descriptors, detected


def _nearest(left: np.ndarray, right: np.ndarray) -> tuple[list[list], list[list]]:
    if min(len(left), len(right)) < 2:
        return [], []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def rows(a, b):
        out = matcher.knnMatch(a, b, k=2)
        if len(out) != len(a) or any(len(row) != 2 for row in out):
            raise ValueError("incomplete two-neighbor query")
        return [[int(row[0].trainIdx), float(row[0].distance),
                 int(row[1].trainIdx), float(row[1].distance)] for row in out]
    return rows(left, right), rows(right, left)


def _proposal_rows(forward: list[list], reverse: list[list], policy: str) -> tuple[list[list], list[list], list[list]]:
    def survivors(rows):
        return [[i, row[0]] for i, row in enumerate(rows)
                if row[1] < ((.85 * row[3]) if policy == POLICIES[0] else row[3])]
    f, r = survivors(forward), survivors(reverse)
    back = dict(r)
    reciprocal = [[i, j] for i, j in f if back.get(j) == i]
    return f, r, reciprocal


def _adapter(policy: str) -> dict:
    meaning = ("strict-lowe-0.85-directional-survivors" if policy == POLICIES[0] else
               "strict-unique-first-neighbor-directional-survivors-no-lowe-margin")
    return dict(geometry_fields=["forward_ratio_pairs", "reverse_ratio_pairs",
                                 "forward_ratio_pass_count", "reverse_ratio_pass_count"],
                meaning=meaning, policy_id=policy,
                tie_rule="first_distance_less_than_0.85_times_second" if policy == POLICIES[0] else TIE_RULE)


def _witness_valid(rows: object, source_count: int, target_count: int) -> bool:
    if not isinstance(rows, list) or len(rows) != source_count:
        return False
    for row in rows:
        if (not isinstance(row, list) or len(row) != 4 or
                type(row[0]) is not int or type(row[2]) is not int or
                not 0 <= row[0] < target_count or not 0 <= row[2] < target_count or
                row[0] == row[2] or
                any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (row[1], row[3])) or
                row[1] > row[3]):
            return False
    return True


def _geometry_from_proposals(data: list, points: dict, proposals: dict,
                             ledgers: dict, comparisons: int) -> dict:
    names = [name for name, _, _ in data]
    hashes = {name: digest for name, _, digest in data}
    shapes = {name: image.shape[:2] for name, image, _ in data}
    pairs = geometry._pair_schedule(names)
    triangles = geometry._scheduled_triangles(names, pairs)
    models, telemetry = {}, []
    for a, b in pairs:
        row = dict(frame_ids=[a, b], pair_key=geometry._pair_key(a, b),
                   left_feature_count=len(points[a]), right_feature_count=len(points[b]),
                   attempts=[], **ledgers[a, b])
        models[a, b] = geometry._pair_models((a, b), tuple(map(tuple, proposals[a, b])),
                                             points[a], points[b], (shapes[a], shapes[b]), row)
        telemetry.append(row)
    combinations = sum(len(models[a, b]) * len(models[b, c]) * len(models[a, c])
                       for a, b, c in triangles)
    if combinations > geometry.MAX_CYCLE_COMBINATIONS:
        raise ValueError("cycle work budget exceeded")
    tracks = []
    for a, b, c in triangles:
        for ab, bc, ac in itertools.product(models[a, b], models[b, c], models[a, c]):
            track = geometry._track(a, b, c, ab, bc, ac, points, shapes, hashes)
            if track is not None:
                tracks.append(track)
    unique = {geometry._digest_bytes(geometry._canonical(t["support"])): t for t in tracks}
    report = dict(schema="projective-plane-tracks-v4", status="supported" if unique else "abstained",
                  abstention=None if unique else "no_coherent_three_view_plane", frame_ids=names,
                  frame_sha256=hashes, frame_dimensions={n: list(shapes[n]) for n in names},
                  frame_hash_scheme="sha256(u32be(height)||u32be(width)||row-major-rgb-uint8)",
                  unresolved=list(geometry.UNRESOLVED), pair_telemetry=telemetry,
                  tracks=[unique[key] for key in sorted(unique)],
                  work=dict(input_rgb_bytes=sum(image.nbytes for _, image, _ in data),
                            pair_count=len(pairs), descriptor_comparisons=comparisons,
                            cycle_model_combinations=combinations, track_attempt_count=combinations,
                            feature_counts={n: len(points[n]) for n in names}))
    report["canonical_sha256"] = geometry._digest_bytes(geometry._canonical(report))
    geometry.canonical_report_bytes(report)
    return report


def _diagnose(frames: Sequence[tuple[str, np.ndarray]], remaining_shared_budget: int) -> dict:
    _check_pins()
    if len(frames) != 3:
        raise ValueError("paired diagnostic requires one three-view case")
    data, total = geometry._frame_input(frames)
    names = [name for name, _, _ in data]
    pairs = geometry._pair_schedule(names)
    points, descriptors, detected = {}, {}, {}
    for name, image, _ in data:
        points[name], descriptors[name], detected[name] = _extract(image, VARIANT)
    comparisons = 2 * sum(len(points[a]) * len(points[b]) for a, b in pairs)
    if comparisons > geometry.MAX_DESCRIPTOR_COMPARISONS:
        raise ValueError("per-policy descriptor work budget exceeded")
    if comparisons > remaining_shared_budget:
        raise ValueError("task shared descriptor work budget exceeded")
    witnesses = []
    for a, b in pairs:
        forward, reverse = _nearest(descriptors[a], descriptors[b])
        witnesses.append(dict(frame_ids=[a, b], forward=forward, reverse=reverse,
                              forward_exact_ties=sum(x[1] == x[3] for x in forward),
                              reverse_exact_ties=sum(x[1] == x[3] for x in reverse)))
    policy_reports = []
    for policy in POLICIES:
        proposals, ledgers, policy_rows = {}, {}, []
        for witness in witnesses:
            a, b = witness["frame_ids"]
            f, r, reciprocal = _proposal_rows(witness["forward"], witness["reverse"], policy)
            proposals[a, b] = reciprocal
            ledgers[a, b] = dict(matching_status="evaluated" if min(len(points[a]), len(points[b])) >= 2 else "insufficient_descriptors",
                                 forward_ratio_pairs=f, reverse_ratio_pairs=r, reciprocal_pairs=reciprocal,
                                 forward_ratio_pass_count=len(f), reverse_ratio_pass_count=len(r),
                                 reciprocal_count=len(reciprocal))
            policy_rows.append(dict(frame_ids=[a, b], forward_survivors=f, reverse_survivors=r,
                                    reciprocal_pairs=reciprocal, forward_survivor_count=len(f),
                                    reverse_survivor_count=len(r), reciprocal_count=len(reciprocal)))
        geo = _geometry_from_proposals(data, points, proposals, ledgers, comparisons)
        policy_reports.append(dict(id=policy, tie_rule=_adapter(policy)["tie_rule"],
                                   geometry_compatibility_adapter=_adapter(policy),
                                   proposal_telemetry=policy_rows, geometry=geo))
    result = dict(schema=SCHEMA, configuration=configuration(), config_sha256=CONFIG_SHA256,
                  frame_ids=names, frame_sha256={name: digest for name, _, digest in data},
                  frame_dimensions={name: list(image.shape[:2]) for name, image, _ in data},
                  input_rgb_bytes=total, detected_feature_counts=detected,
                  feature_points={name: points[name].tolist() for name in names},
                  shared_descriptor_comparisons=comparisons, nearest_witness=witnesses,
                  policies=policy_reports)
    result["canonical_sha256"] = _sha(_canonical(result))
    canonical_report_bytes(result)
    return result


def diagnose(frames: Sequence[tuple[str, np.ndarray]]) -> dict:
    """Only RGB arrays and opaque IDs enter either proposal or geometry policy."""
    return _diagnose(frames, geometry.MAX_DESCRIPTOR_COMPARISONS)


def canonical_report_bytes(report: dict) -> bytes:
    """Replay policy identities and pinned geometry evidence, not SIFT provenance."""
    _check_pins()
    fields = {"schema", "configuration", "config_sha256", "frame_ids", "frame_sha256",
              "frame_dimensions", "input_rgb_bytes", "detected_feature_counts", "feature_points",
              "shared_descriptor_comparisons", "nearest_witness", "policies", "canonical_sha256"}
    if not isinstance(report, dict) or set(report) != fields:
        raise ValueError("malformed paired report")
    payload = dict(report)
    digest = payload.pop("canonical_sha256")
    if (digest != _sha(_canonical(payload)) or report["schema"] != SCHEMA or
            report["configuration"] != configuration() or report["config_sha256"] != CONFIG_SHA256):
        raise ValueError("paired digest or configuration drift")
    names, dims, hashes = report["frame_ids"], report["frame_dimensions"], report["frame_sha256"]
    if (not isinstance(names, list) or len(names) != 3 or names != sorted(names) or len(set(names)) != 3 or
            not isinstance(dims, dict) or set(dims) != set(names) or
            not isinstance(hashes, dict) or set(hashes) != set(names) or
            any(not isinstance(v, list) or len(v) != 2 or any(type(x) is not int or x < 64 for x in v)
                or v[0] * v[1] > geometry.MAX_PIXELS_PER_FRAME for v in dims.values()) or
            any(not isinstance(v, str) or len(v) != 64 for v in hashes.values()) or
            type(report["input_rgb_bytes"]) is not int or
            report["input_rgb_bytes"] != 3 * sum(h * w for h, w in dims.values())):
        raise ValueError("paired frame binding drift")
    points, detected = report["feature_points"], report["detected_feature_counts"]
    if (not isinstance(points, dict) or not isinstance(detected, dict) or
            set(points) != set(names) or set(detected) != set(names)):
        raise ValueError("malformed shared features")
    for name in names:
        h, w = dims[name]
        xy = points[name]
        if (not isinstance(xy, list) or len(xy) > FEATURE_CAP or
                type(detected[name]) is not int or not len(xy) <= detected[name] <= h * w * 8 or
                any(not isinstance(p, list) or len(p) != 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in p) or
                    not 0 <= p[0] < w or not 0 <= p[1] < h for p in xy)):
            raise ValueError("malformed shared feature coordinates")
        _enforce_feature_cap(len(xy), detected[name])
    pairs = geometry._pair_schedule(names)
    comparisons = 2 * sum(len(points[a]) * len(points[b]) for a, b in pairs)
    if (type(report["shared_descriptor_comparisons"]) is not int or
            report["shared_descriptor_comparisons"] != comparisons or
            comparisons > geometry.MAX_DESCRIPTOR_COMPARISONS):
        raise ValueError("shared work budget drift")
    witnesses = report["nearest_witness"]
    if not isinstance(witnesses, list) or len(witnesses) != len(pairs):
        raise ValueError("missing nearest witness")
    for pair, row in zip(pairs, witnesses):
        a, b = pair
        if (not isinstance(row, dict) or set(row) != {"frame_ids", "forward", "reverse",
                                                 "forward_exact_ties", "reverse_exact_ties"} or
                row["frame_ids"] != [a, b] or
                not _witness_valid(row["forward"], len(points[a]) if len(points[b]) >= 2 else 0, len(points[b])) or
                not _witness_valid(row["reverse"], len(points[b]) if len(points[a]) >= 2 else 0, len(points[a])) or
                type(row["forward_exact_ties"]) is not int or
                row["forward_exact_ties"] != sum(x[1] == x[3] for x in row["forward"]) or
                type(row["reverse_exact_ties"]) is not int or
                row["reverse_exact_ties"] != sum(x[1] == x[3] for x in row["reverse"])):
            raise ValueError("malformed nearest witness or tie count")
    rows = report["policies"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("policy count drift")
    for policy, entry in zip(POLICIES, rows):
        if (not isinstance(entry, dict) or set(entry) != {"id", "tie_rule", "geometry_compatibility_adapter",
                                                 "proposal_telemetry", "geometry"} or
                entry["id"] != policy or entry["tie_rule"] != _adapter(policy)["tie_rule"] or
                entry["geometry_compatibility_adapter"] != _adapter(policy) or
                not isinstance(entry["proposal_telemetry"], list) or
                len(entry["proposal_telemetry"]) != len(pairs)):
            raise ValueError("policy identity drift")
        geo = entry["geometry"]
        geometry.canonical_report_bytes(geo)
        if (geo["frame_ids"] != names or geo["frame_sha256"] != hashes or
                geo["frame_dimensions"] != dims or geo["work"]["input_rgb_bytes"] != report["input_rgb_bytes"] or
                geo["work"]["descriptor_comparisons"] != comparisons or
                geo["work"]["feature_counts"] != {n: len(points[n]) for n in names}):
            raise ValueError("policy geometry frame or work binding drift")
        for pair, witness, policy_row, geo_row in zip(pairs, witnesses, entry["proposal_telemetry"], geo["pair_telemetry"]):
            f, r, reciprocal = _proposal_rows(witness["forward"], witness["reverse"], policy)
            expected = dict(frame_ids=list(pair), forward_survivors=f, reverse_survivors=r,
                            reciprocal_pairs=reciprocal, forward_survivor_count=len(f),
                            reverse_survivor_count=len(r), reciprocal_count=len(reciprocal))
            if policy_row != expected or any((geo_row[key] != value for key, value in (
                    ("forward_ratio_pairs", f), ("reverse_ratio_pairs", r),
                    ("reciprocal_pairs", reciprocal), ("forward_ratio_pass_count", len(f)),
                    ("reverse_ratio_pass_count", len(r)), ("reciprocal_count", len(reciprocal))))):
                raise ValueError("policy proposal identity or count drift")
        for track in geo["tracks"]:
            for name in track["frame_ids"]:
                for feature in track["feature_points"][name]:
                    if feature[1:] != points[name][feature[0]]:
                        raise ValueError("track contradicts shared extracted features")
    raw = _canonical(report)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("paired output budget exceeded")
    return raw


def verify_frame_binding(report: dict, frames: Sequence[tuple[str, np.ndarray]]) -> None:
    canonical_report_bytes(report)
    data, total = geometry._frame_input(frames)
    if (report["frame_ids"] != [n for n, _, _ in data] or
            report["frame_sha256"] != {n: digest for n, _, digest in data} or
            report["frame_dimensions"] != {n: list(image.shape[:2]) for n, image, _ in data} or
            report["input_rgb_bytes"] != total):
        raise ValueError("paired frame source drift")


def persist_report(path: str | Path, report: dict,
                   frames: Sequence[tuple[str, np.ndarray]]) -> None:
    verify_frame_binding(report, frames)
    raw = canonical_report_bytes(report)
    target = Path(path)
    if not target.parent.is_dir():
        raise ValueError("output parent missing")
    fd, temporary = tempfile.mkstemp(prefix=".mnn-proposals-v2-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            if out.write(raw) != len(raw):
                raise OSError("partial report write")
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)


# Everything below is generated-fixture evaluation. None of its truth arguments
# enter diagnose, matching, partitioning, RANSAC, or geometric acceptance.
SIZE = 192


def _asset(rng: np.random.Generator, family: str) -> np.ndarray:
    """v2 bounded-complexity texture: fewer/smoother responses per frame."""
    image = np.full((SIZE, SIZE, 3), 112, np.uint8)
    if family == "weak-contrast":
        for _ in range(120):
            x, y = rng.integers(8, 184, 2)
            shade = int(rng.choice((-17, -12, 12, 17)) + 112)
            cv2.rectangle(image, (int(x), int(y)), (int(min(x + rng.integers(4, 18), 190)),
                          int(min(y + rng.integers(4, 18), 190))), (shade, shade, shade), 1)
    elif family == "repeated-brick-tile":
        tile = rng.integers(88, 142, (48, 48, 3), dtype=np.uint8)
        tile = cv2.GaussianBlur(tile, (15, 15), 0)
        for y in range(4):
            for x in range(4):
                patch = np.roll(tile, ((y % 2) * 7, 0), axis=(0, 1))
                image[y*48:(y+1)*48, x*48:(x+1)*48] = patch
                cv2.rectangle(image, (x*48, y*48), (x*48+47, y*48+47), (60, 60, 60), 2)
    elif family == "mixed-scale-pixel":
        small = rng.integers(70, 180, (24, 24, 3), dtype=np.uint8)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        image = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        for _ in range(26):
            x, y = rng.integers(6, 186, 2)
            cv2.circle(image, (int(x), int(y)), int(rng.integers(6, 15)),
                       tuple(int(v) for v in rng.integers(25, 225, 3)), -1)
    else:
        for _ in range(34):
            x, y = rng.integers(12, 166, 2)
            side = int(rng.integers(9, 24))
            color = tuple(int(v) for v in rng.integers(45, 210, 3))
            cv2.rectangle(image, (int(x), int(y)), (int(x + side), int(y + side)), color, -1)
        for _ in range(12):
            a, b = rng.integers(0, SIZE, (2, 2))
            cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (55, 55, 55), 1)
    return image


def _tile_asset(rng: np.random.Generator, family: str) -> np.ndarray:
    """One complexity-bounded 48px tile, built per family, for hard negatives."""
    tile = np.full((48, 48, 3), 112, np.uint8)
    if family == "weak-contrast":
        for _ in range(14):
            x, y = rng.integers(4, 42, 2)
            shade = int(rng.choice((-17, -12, 12, 17)) + 112)
            cv2.rectangle(tile, (int(x), int(y)), (int(min(x + rng.integers(4, 9), 46)),
                          int(min(y + rng.integers(4, 9), 46))), (shade, shade, shade), 1)
    elif family == "repeated-brick-tile":
        tile = cv2.GaussianBlur(rng.integers(88, 142, (48, 48, 3), dtype=np.uint8), (15, 15), 0)
    elif family == "mixed-scale-pixel":
        small = cv2.GaussianBlur(rng.integers(70, 180, (12, 12, 3), dtype=np.uint8), (3, 3), 0)
        tile = cv2.resize(small, (48, 48), interpolation=cv2.INTER_LINEAR)
    else:
        for _ in range(8):
            x, y = rng.integers(2, 40, 2)
            side = int(rng.integers(6, 14))
            cv2.rectangle(tile, (int(x), int(y)), (int(x + side), int(y + side)),
                          tuple(int(v) for v in rng.integers(45, 210, 3)), -1)
    return tile


def _tiled_negative(rng: np.random.Generator, family: str) -> np.ndarray:
    """v2 incompatible-tile negative from an independently permuted bounded tile."""
    tile = _tile_asset(rng, family)
    image = np.full((SIZE, SIZE, 3), 112, np.uint8)
    for y in range(4):
        for x in range(4):
            image[y*48:(y+1)*48, x*48:(x+1)*48] = np.rot90(tile, int(rng.integers(0, 4)))
            cv2.rectangle(image, (x*48, y*48), (x*48+47, y*48+47), (60, 60, 60), 2)
    return image


def _transforms(rng: np.random.Generator) -> list[np.ndarray]:
    skew = float(rng.uniform(-.012, .012))
    return [np.eye(3),
            np.array([[.985, .045 + skew, 5.], [-.04, .99, 8.], [.00007, -.00005, 1.]]),
            np.array([[1.015, -.04 + skew, -5.], [.035, .99, 4.], [-.00006, .00008, 1.]])]


def _warp(asset: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(asset, transform, (SIZE, SIZE), borderValue=(112, 112, 112))


def synthetic_cases(split: str = "dev"):
    """Yield one triple at a time; scene assets never cross a split or group."""
    if split not in AUTHORIZED_CALIBRATION_SPLITS:
        raise ValueError("synthetic split not authorized by this task")
    for group_split, family, seed in SEEDS:
        if group_split != split:
            continue
        rng = np.random.default_rng(seed)
        base, other = _asset(rng, family), _asset(rng, family)
        group = f"{group_split}-{family}-{seed}"
        transforms = _transforms(rng)
        for kind in CASE_KINDS:
            if kind in CASE_KINDS[:4]:
                images = [_warp(base, h) for h in transforms]
                if kind == "brightness":
                    images[1] = np.clip(images[1].astype(float) * .72, 0, 255).astype(np.uint8)
                elif kind == "noise":
                    for i in (1, 2):
                        images[i] = np.clip(images[i].astype(np.int16) +
                                            rng.integers(-4, 5, images[i].shape), 0, 255).astype(np.uint8)
                elif kind == "occlusion":
                    images[1][70:113, 65:118] = 112
                oracle = dict(kind="positive", transforms={"scene": transforms}, mode="positive")
            elif kind == "palette-nonoverlap":
                images = [_asset(rng, family) for _ in range(3)]
                oracle = dict(kind="pure-negative", transforms={}, mode="unshared")
            elif kind == "incompatible-tiles":
                images = [_tiled_negative(rng, family) for _ in range(3)]
                oracle = dict(kind="pure-negative", transforms={}, mode="unshared")
            elif kind == "multi-plane-corner":
                second = [h @ np.array([[1., .07, 3.], [-.04, 1., -3.], [0., 0., 1.]])
                          for h in transforms]
                images = []
                for h1, h2 in zip(transforms, second):
                    left, right = _warp(base, h1), _warp(other, h2)
                    image = left.copy()
                    image[:, 100:] = right[:, 100:]
                    image[:, 94:100] = 112
                    images.append(image)
                oracle = dict(kind="structural", transforms={"face-a": transforms,
                                                              "face-b": second}, mode="corner")
            else:
                source = np.full((SIZE, SIZE, 3), 112, np.uint8)
                source[20:172, :78] = base[20:172, :78]
                source[20:172, 114:] = other[20:172, 114:]
                images = [_warp(source, h) for h in transforms]
                oracle = dict(kind="structural", transforms={"fragment-a": transforms,
                                                              "fragment-b": transforms}, mode="fragments")
            frames = [(f"view-{i}", np.ascontiguousarray(image)) for i, image in enumerate(images)]
            yield dict(group=group, split=group_split, family=family, kind=kind,
                       frames=frames, oracle=oracle)


def _label(oracle: dict, frame: str, xy: list[float]) -> tuple[str, np.ndarray] | None:
    index = int(frame[-1])
    mode = oracle["mode"]
    if mode == "unshared":
        return f"unshared-{index}", np.asarray(xy, float)
    if mode == "corner":
        surface = "face-a" if xy[0] < 94 else "face-b" if xy[0] >= 100 else None
    elif mode == "fragments":
        base_uv = geometry._transfer(np.linalg.inv(oracle["transforms"]["fragment-a"][index]),
                                     np.asarray([xy], float))[0]
        surface = "fragment-a" if 20 <= base_uv[1] < 172 and base_uv[0] < 78 else (
            "fragment-b" if 20 <= base_uv[1] < 172 and base_uv[0] >= 114 else None)
    else:
        surface = "scene"
    if surface is None:
        return None
    uv = geometry._transfer(np.linalg.inv(oracle["transforms"][surface][index]),
                            np.asarray([xy], float))[0]
    if not np.all(np.isfinite(uv)) or not (0 <= uv[0] < SIZE and 0 <= uv[1] < SIZE):
        return None
    if mode == "corner" and ((surface == "face-a" and xy[0] >= 94) or
                             (surface == "face-b" and xy[0] < 100)):
        return None
    return surface, uv


def _oracle_pair(oracle: dict, points: dict, pair: list[str], proposal: tuple[int, int]) -> bool:
    a, b = pair
    x = _label(oracle, a, points[a][proposal[0]])
    y = _label(oracle, b, points[b][proposal[1]])
    return bool(x is not None and y is not None and x[0] == y[0] and
                np.linalg.norm(x[1] - y[1]) <= 3)


def _eligible(oracle: dict, points: dict, pair: list[str]) -> int:
    a, b = pair
    right = [_label(oracle, b, point) for point in points[b]]
    count = 0
    for point in points[a]:
        x = _label(oracle, a, point)
        if x is not None and any(y is not None and x[0] == y[0] and
                                 np.linalg.norm(x[1] - y[1]) <= 3 for y in right):
            count += 1
    return count


def _track_valid(oracle: dict, points: dict, track: dict) -> bool:
    names = track["frame_ids"]
    surfaces = set()
    for triangle in track["triangles"]:
        labels = [_label(oracle, name, points[name][i]) for name, i in zip(names, triangle)]
        if any(label is None for label in labels) or len({label[0] for label in labels}) != 1:
            return False
        surfaces.add(labels[0][0])
    return len(surfaces) == 1


def _frame_feature_counts(report: dict) -> dict:
    """Per-frame detected/retained SIFT counts recorded for every case."""
    counts = {}
    for name in report["frame_ids"]:
        detected = report["detected_feature_counts"][name]
        retained = len(report["feature_points"][name])
        _enforce_feature_cap(retained, detected)
        counts[name] = dict(detected=detected, retained=retained)
    return counts


def _case_metrics(case: dict, report: dict) -> dict:
    oracle, points = case["oracle"], report["feature_points"]
    results = []
    eligibility = {_pair_key: _eligible(oracle, points, list(_pair_key))
                   for _pair_key in geometry._pair_schedule(report["frame_ids"])}
    for policy in report["policies"]:
        pair_rows = []
        for witness, proposal_row, geo_row in zip(report["nearest_witness"],
                                                   policy["proposal_telemetry"],
                                                   policy["geometry"]["pair_telemetry"]):
            pair = witness["frame_ids"]
            proposals = proposal_row["reciprocal_pairs"]
            true_proposals = sum(_oracle_pair(oracle, points, pair, tuple(p)) for p in proposals)
            accepted, true_accepted, invalid = 0, 0, False
            for attempt in geo_row["attempts"]:
                if attempt["first_rejection"] != "accepted":
                    continue
                support = [proposals[i] for i in attempt["fit_inlier_indices"]]
                support += [proposals[row[0]] for row in attempt["held_rows"]
                            if row[1] != "inf" and row[2] != "inf" and row[1] <= 3 and row[2] <= 3]
                labels = set()
                for candidate in support:
                    good = _oracle_pair(oracle, points, pair, tuple(candidate))
                    true_accepted += good
                    accepted += 1
                    if not good:
                        invalid = True
                    else:
                        label = _label(oracle, pair[0], points[pair[0]][candidate[0]])
                        labels.add(label[0])
                if oracle["mode"] in ("corner", "fragments") and len(labels) > 1:
                    invalid = True
            first = geo_row["attempts"][0]
            eligible = eligibility[tuple(pair)]
            pair_rows.append(dict(frame_ids=pair, detected_feature_counts=[report["detected_feature_counts"][n] for n in pair],
                                  retained_feature_counts=[len(points[n]) for n in pair],
                                  forward_exact_ties=witness["forward_exact_ties"],
                                  reverse_exact_ties=witness["reverse_exact_ties"],
                                  forward_first_neighbor_count=len(witness["forward"]),
                                  reverse_first_neighbor_count=len(witness["reverse"]),
                                  forward_ratio_survivor_count=sum(x[1] < .85*x[3] for x in witness["forward"]),
                                  reverse_ratio_survivor_count=sum(x[1] < .85*x[3] for x in witness["reverse"]),
                                  forward_policy_survivor_count=proposal_row["forward_survivor_count"],
                                  reverse_policy_survivor_count=proposal_row["reverse_survivor_count"],
                                  reciprocal_proposal_identities=proposals, true_proposals=true_proposals,
                                  false_proposals=len(proposals)-true_proposals,
                                  eligible_sources=eligible, true_proposal_recall=true_proposals/eligible if eligible else 0.,
                                  fit_count=first["fit_count"], held_count=first["held_count"],
                                  partition_eligible=first["fit_count"] >= 16 and first["held_count"] >= 8,
                                  ransac_reached=first["homography_available"] is not None,
                                  first_geometry_rejection=[row["first_rejection"] for row in geo_row["attempts"]],
                                  accepted_model_count=geo_row["accepted_model_count"],
                                  accepted_support=accepted, true_accepted_support=true_accepted,
                                  accepted_support_precision=true_accepted/accepted if accepted else 0.,
                                  invalid_structural_support=invalid if oracle["kind"] == "structural" else False))
        valid_tracks = sum(_track_valid(oracle, points, t) for t in policy["geometry"]["tracks"])
        results.append(dict(id=policy["id"], pairs=pair_rows, valid_tracks=valid_tracks,
                            invalid_tracks=len(policy["geometry"]["tracks"])-valid_tracks,
                            positive_supported=oracle["kind"] == "positive" and valid_tracks > 0,
                            accepted_pair_count=sum(row["accepted_model_count"] > 0 for row in pair_rows),
                            invalid_structural_support=any(row["invalid_structural_support"] for row in pair_rows),
                            resource_refusal=None))
    return dict(group=case["group"], split=case["split"], family=case["family"],
                kind=case["kind"], label_kind=oracle["kind"],
                input_sha256=report["frame_sha256"], report_sha256=_sha(canonical_report_bytes(report)),
                frame_feature_counts=_frame_feature_counts(report),
                shared_descriptor_comparisons=report["shared_descriptor_comparisons"], policies=results)


def _aggregate(cases: list[dict], policy_index: int) -> dict:
    selected = [case["policies"][policy_index] for case in cases]
    pairs = [row for policy in selected for row in policy["pairs"]]
    true_proposals = sum(p["true_proposals"] for p in pairs)
    eligible = sum(p["eligible_sources"] for p in pairs)
    accepted = sum(p["accepted_support"] for p in pairs)
    true_accepted = sum(p["true_accepted_support"] for p in pairs)
    positives = [p for case, p in zip(cases, selected) if case["label_kind"] == "positive"]
    negatives = [p for case, p in zip(cases, selected) if case["label_kind"] == "pure-negative"]
    structural = [p for case, p in zip(cases, selected) if case["label_kind"] == "structural"]
    return dict(positive_supported=sum(p["positive_supported"] for p in positives),
                positive_cases=len(positives), negative_accepted_cases=sum(p["accepted_pair_count"] > 0 for p in negatives),
                structural_invalid_cases=sum(p["invalid_structural_support"] or p["invalid_tracks"] > 0 for p in structural),
                true_proposals=true_proposals, eligible_sources=eligible,
                true_proposal_recall=true_proposals/eligible if eligible else 0.,
                partition_eligible_positive_pairs=sum(row["partition_eligible"] for case in cases
                    if case["label_kind"] == "positive" for row in case["policies"][policy_index]["pairs"]),
                accepted_support=accepted, true_accepted_support=true_accepted,
                accepted_support_precision=true_accepted/accepted if accepted else 0.,
                correct_geometry_pairs=sum(row["accepted_model_count"] > 0 and row["true_accepted_support"] > 0
                    for case in cases if case["label_kind"] == "positive"
                    for row in case["policies"][policy_index]["pairs"]))


def _families(split: str) -> list[str]:
    return list(dict.fromkeys(seed[1] for seed in SEEDS if seed[0] == split))


def _dev_gates(records: list[dict], aggregates: dict) -> dict:
    dev = aggregates["dev"]
    a, b = (dev["pooled"][policy] for policy in POLICIES)
    family_pass = all(dev["families"][family][POLICIES[1]]["positive_supported"] >= GATES["positives_per_family_min"] and
                      dev["families"][family][POLICIES[1]]["accepted_support_precision"] >= GATES["accepted_precision_min"]
                      for family in dev["families"])
    a_supported = {c["group"] + "/" + c["kind"] for c in records if
                   c["label_kind"] == "positive" and c["policies"][0]["positive_supported"]}
    b_supported = {c["group"] + "/" + c["kind"] for c in records if
                   c["label_kind"] == "positive" and c["policies"][1]["positive_supported"]}
    return dict(family_positive_and_precision=family_pass,
                pooled_precision=b["accepted_support_precision"] >= GATES["accepted_precision_min"],
                pure_negative_zero=b["negative_accepted_cases"] == 0,
                structural_invalid_zero=b["structural_invalid_cases"] == 0,
                no_a_positive_loss=a_supported <= b_supported,
                strict_recall_gain=b["true_proposal_recall"] > a["true_proposal_recall"],
                strict_partition_gain=b["partition_eligible_positive_pairs"] > a["partition_eligible_positive_pairs"],
                correct_geometry_pair=b["correct_geometry_pairs"] >= GATES["correct_geometry_pair_min"],
                a_not_saturated=a["partition_eligible_positive_pairs"] < 16*3)


def calibrate_synthetic_dev() -> dict:
    """Authorized run: the four-family development split only (32 cases)."""
    _check_pins()
    records, actual_work = [], 0
    for case in synthetic_cases("dev"):
        report = _diagnose(case["frames"], MAX_SHARED_COMPARISONS - actual_work)
        # A development case above 700 fails here, before any match-policy metric
        # is computed or published.
        _frame_feature_counts(report)
        actual_work += report["shared_descriptor_comparisons"]
        if actual_work > MAX_SHARED_COMPARISONS:
            raise ValueError("task shared descriptor work budget exceeded")
        records.append(_case_metrics(case, report))
        del report, case
    if len(records) != MAX_DEV_CASES:
        raise ValueError("development case count drift")
    if any(row["split"] != "dev" for row in records):
        raise ValueError("development split drift")
    aggregates = dict(dev=dict(pooled={policy: _aggregate(records, i) for i, policy in enumerate(POLICIES)},
                               families={family: {policy: _aggregate([c for c in records if c["family"] == family], i)
                                                  for i, policy in enumerate(POLICIES)}
                                         for family in _families("dev")}))
    gates = _dev_gates(records, aggregates)
    result = dict(schema=CALIBRATION_SCHEMA, configuration=configuration(), config_sha256=CONFIG_SHA256,
                  evaluation_split="dev", heldout_run=False, cases=records, aggregates=aggregates,
                  actual_shared_descriptor_comparisons=actual_work, gates=gates,
                  decision="go" if all(gates.values()) else "inconclusive-no-go" if not gates["a_not_saturated"] else "no-go",
                  selection_basis="generated development scene groups only; held-out groups declared but unevaluated; no ordinary images or scene metadata")
    result["canonical_sha256"] = _sha(_canonical(result))
    canonical_calibration_bytes(result)
    return result


def canonical_calibration_bytes(result: dict) -> bytes:
    """Validate a development calibration artifact before publication."""
    _check_pins()
    fields = {"schema", "configuration", "config_sha256", "evaluation_split", "heldout_run",
              "cases", "aggregates", "actual_shared_descriptor_comparisons", "gates", "decision",
              "selection_basis", "canonical_sha256"}
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError("malformed calibration result")
    payload = dict(result)
    digest = payload.pop("canonical_sha256")
    if (digest != _sha(_canonical(payload)) or result["schema"] != CALIBRATION_SCHEMA or
            result["configuration"] != configuration() or result["config_sha256"] != CONFIG_SHA256):
        raise ValueError("calibration digest or configuration drift")
    if result["evaluation_split"] != "dev" or result["heldout_run"] is not False:
        raise ValueError("calibration split drift")
    cases = result["cases"]
    if not isinstance(cases, list) or len(cases) != MAX_DEV_CASES:
        raise ValueError("calibration case count drift")
    if any(not isinstance(case, dict) or case.get("split") != "dev" for case in cases):
        raise ValueError("calibration development split drift")
    for case in cases:
        counts = case["frame_feature_counts"]
        if not isinstance(counts, dict) or set(counts) != set(case["input_sha256"]):
            raise ValueError("malformed per-frame feature counts")
        for entry in counts.values():
            if not isinstance(entry, dict) or set(entry) != {"detected", "retained"}:
                raise ValueError("malformed per-frame feature counts")
            _enforce_feature_cap(entry["retained"], entry["detected"])
    raw = _canonical(result)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("calibration artifact output budget exceeded")
    return raw


def persist_calibration(path: str | Path, result: dict) -> None:
    raw = canonical_calibration_bytes(result)
    target = Path(path)
    if not target.parent.is_dir():
        raise ValueError("output parent missing")
    fd, temporary = tempfile.mkstemp(prefix=".mnn-proposals-v2-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            if out.write(raw) != len(raw):
                raise OSError("partial calibration write")
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)
