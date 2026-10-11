"""Bounded, image-only OpenCV matcher comparison; synthetic calibration is separate.

No production import, image-file loader, camera input, or ground-truth API exists.
The image entry point compares frozen front ends. Known transforms occur only in
make_synthetic_suite/calibrate_synthetic and never enter geometric estimation.
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

from scripts.evaluation import diagnose_multiview_plane_tracks as geometry

BASELINE_SHA256 = "97eaac2bdcc09649a2a7e364ff7589f9cd09916ff62922fc8fe52f24bdff8db7"
SCHEMA = "opencv-matcher-ablation-v2"
MAX_VARIANTS = 6
MAX_TOTAL_DESCRIPTOR_COMPARISONS = 376_320_000
MAX_OUTPUT_BYTES = 96_000_000
# Tuple representation prevents callers from altering the preregistered sweep.
VARIANTS = (
    ("sift-baseline", "sift", .04, .72),
    ("sift-low-contrast", "sift", .01, .72),
    ("sift-wide-ratio", "sift", .04, .85),
    ("sift-low-wide", "sift", .01, .85),
    ("akaze", "akaze", None, .85),
    ("orb", "orb", None, .85),
)
GATES = {"positive_cases_supported_min": 3, "accepted_precision_min": .95,
         "negative_cases_with_accepted_pair_max": 0, "positive_cases_total": 4,
         "negative_cases_total": 2}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def configuration() -> dict[str, object]:
    return {"variants": [dict(id=name, detector=detector, contrast=contrast, ratio=ratio,
                              norm="L2" if detector == "sift" else "HAMMING",
                              max_features=700)
                         for name, detector, contrast, ratio in VARIANTS],
            "geometry_source_sha256": BASELINE_SHA256,
            "selection": "opencv-response-descending-with-stable-keypoint-tiebreak-for-akaze-orb",
            "synthetic_gates": dict(GATES),
            "ranking": "passing-only:positive-support-desc,proposal-recall-desc,precision-desc,config-order",
            "truth_distance_px": 3.0}


CONFIG_SHA256 = _sha(_canonical(configuration()))


def _check_baseline() -> None:
    if _sha(Path(geometry.__file__).read_bytes()) != BASELINE_SHA256:
        raise ValueError("geometric diagnostic source pin drift")
    if _sha(_canonical(configuration())) != CONFIG_SHA256:
        raise ValueError("preregistered configuration pin drift")


def _extract(image: np.ndarray, variant: tuple) -> tuple[np.ndarray, np.ndarray, int]:
    _, kind, contrast, _ = variant
    if kind == "sift" and contrast == .04:
        points, descriptors = geometry._features(image)
        return points, descriptors, len(points)
    detector = (cv2.SIFT_create(nfeatures=668, contrastThreshold=contrast) if kind == "sift" else
                cv2.AKAZE_create() if kind == "akaze" else cv2.ORB_create(nfeatures=700))
    cv2.setRNGSeed(0)
    keys, descriptors = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    detected = len(keys)
    if descriptors is None:
        return (np.empty((0, 2), np.float64),
                np.empty((0, 128 if kind == "sift" else 61 if kind == "akaze" else 32),
                         np.float32 if kind == "sift" else np.uint8), detected)
    if kind == "sift":
        if detected > 700:
            raise ValueError("SIFT feature budget exceeded; refusing truncation")
        order = list(range(detected))
    else:
        order = sorted(range(detected), key=lambda i: (-keys[i].response, keys[i].pt,
                        keys[i].size, keys[i].angle, keys[i].octave, keys[i].class_id))[:700]
    return (np.array([keys[i].pt for i in order], dtype=np.float64).reshape(-1, 2),
            np.ascontiguousarray(descriptors[order]), detected)


def _proposals(left: np.ndarray, right: np.ndarray, variant: tuple, record: dict) -> tuple:
    if len(left) < 2 or len(right) < 2:
        record.update(matching_status="insufficient_descriptors", forward_ratio_pairs=[],
                      reverse_ratio_pairs=[], reciprocal_pairs=[], forward_ratio_pass_count=0,
                      reverse_ratio_pass_count=0, reciprocal_count=0)
        return ()
    matcher = cv2.BFMatcher(cv2.NORM_L2 if variant[1] == "sift" else cv2.NORM_HAMMING)
    def good(a, b):
        return {row[0].queryIdx: row[0].trainIdx for row in matcher.knnMatch(a, b, k=2)
                if len(row) == 2 and row[0].distance < variant[3] * row[1].distance}
    forward, reverse = good(left, right), good(right, left)
    pairs = tuple(sorted((i, j) for i, j in forward.items() if reverse.get(j) == i))
    if len(pairs) > 700:
        raise ValueError("proposal budget exceeded")
    record.update(matching_status="evaluated", forward_ratio_pairs=[list(x) for x in sorted(forward.items())],
                  reverse_ratio_pairs=[list(x) for x in sorted(reverse.items())],
                  reciprocal_pairs=[list(x) for x in pairs], forward_ratio_pass_count=len(forward),
                  reverse_ratio_pass_count=len(reverse), reciprocal_count=len(pairs))
    return pairs


def _geometry_report(data: list, variant: tuple, descriptor_budget: int) -> tuple[dict, dict, dict]:
    names = [x[0] for x in data]
    hashes = {name: digest for name, _, digest in data}
    shapes = {name: image.shape[:2] for name, image, _ in data}
    points, descriptors, detected = {}, {}, {}
    for name, image, _ in data:
        points[name], descriptors[name], detected[name] = _extract(image, variant)
    pairs = geometry._pair_schedule(names)
    triangles = geometry._scheduled_triangles(names, pairs)
    comparisons = 2 * sum(len(points[a]) * len(points[b]) for a, b in pairs)
    extended = len(names) > 8
    limit = geometry.EXTENDED_MAX_DESCRIPTOR_COMPARISONS if extended else geometry.MAX_DESCRIPTOR_COMPARISONS
    if comparisons > min(limit, descriptor_budget):
        raise ValueError("descriptor work budget exceeded")
    models, ledger = {}, []
    for a, b in pairs:
        record = dict(frame_ids=[a, b], pair_key=geometry._pair_key(a, b),
                      left_feature_count=len(points[a]), right_feature_count=len(points[b]), attempts=[])
        proposals = _proposals(descriptors[a], descriptors[b], variant, record)
        models[a, b] = geometry._pair_models((a, b), proposals, points[a], points[b],
                                             (shapes[a], shapes[b]), record)
        ledger.append(record)
    combinations = sum(len(models[a, b]) * len(models[a, c]) * len(models[b, c])
                       for a, b, c in triangles)
    if combinations > geometry.MAX_CYCLE_COMBINATIONS:
        raise ValueError("cycle work budget exceeded")
    tracks = []
    for a, b, c in triangles:
        for ab, bc, ac in itertools.product(models[a, b], models[b, c], models[a, c]):
            candidate = geometry._track(a, b, c, ab, bc, ac, points, shapes, hashes)
            if candidate is not None:
                tracks.append(candidate)
    unique = {geometry._digest_bytes(geometry._canonical(t["support"])): t for t in tracks}
    result = dict(schema="projective-plane-tracks-v5" if extended else "projective-plane-tracks-v4",
                  status="supported" if unique else "abstained",
                  abstention=None if unique else "no_coherent_three_view_plane", frame_ids=names,
                  frame_sha256=hashes, frame_dimensions={n: list(shapes[n]) for n in names},
                  frame_hash_scheme="sha256(u32be(height)||u32be(width)||row-major-rgb-uint8)",
                  unresolved=list(geometry.UNRESOLVED), pair_telemetry=ledger,
                  tracks=[unique[k] for k in sorted(unique)],
                  work=dict(input_rgb_bytes=sum(image.nbytes for _, image, _ in data),
                            pair_count=len(pairs), descriptor_comparisons=comparisons,
                            cycle_model_combinations=combinations, track_attempt_count=combinations,
                            feature_counts={n: len(points[n]) for n in names}))
    if extended:
        result["work"].update(pair_schedule=geometry.PAIR_SCHEDULE,
                              omitted_pair_count=len(names) * (len(names) - 1) // 2 - len(pairs))
    result["canonical_sha256"] = geometry._digest_bytes(geometry._canonical(result))
    geometry.canonical_report_bytes(result)
    return result, {n: points[n].tolist() for n in names}, detected


def _stage_summary(report: dict) -> dict:
    records = report["pair_telemetry"]
    rejections = {}
    for row in records:
        for attempt in row["attempts"]:
            gate = attempt["first_rejection"]
            rejections[gate] = rejections.get(gate, 0) + 1
    return dict(proposal_count=sum(row["reciprocal_count"] for row in records),
                accepted_pair_count=sum(row["accepted_model_count"] > 0 for row in records),
                accepted_model_count=sum(row["accepted_model_count"] for row in records),
                held_joint_support=sum(attempt["held_joint_pass_count"] or 0 for row in records
                                       for attempt in row["attempts"] if attempt["first_rejection"] == "accepted"),
                track_count=len(report["tracks"]), first_rejections=rejections)


def diagnose(frames: Sequence[tuple[str, np.ndarray]]) -> dict:
    """Compare all frozen variants on RGB pixels, never select using scene truth."""
    _check_baseline()
    if len(VARIANTS) != MAX_VARIANTS:
        raise ValueError("variant budget or config drift")
    data, total = geometry._frame_input(frames)
    results, comparisons = [], 0
    for variant in VARIANTS:
        if variant[1] == "akaze" and not hasattr(cv2, "AKAZE_create"):
            results.append(dict(id=variant[0], availability="unavailable",
                                reason="OpenCV API AKAZE_create absent", feature_points=None,
                                detected_feature_counts=None, stage_summary=None, geometry=None))
            continue
        report, points, detected = _geometry_report(data, variant, MAX_TOTAL_DESCRIPTOR_COMPARISONS - comparisons)
        comparisons += report["work"]["descriptor_comparisons"]
        if comparisons > MAX_TOTAL_DESCRIPTOR_COMPARISONS:
            raise ValueError("total descriptor work budget exceeded")
        results.append(dict(id=variant[0], availability="available", reason=None, feature_points=points, detected_feature_counts=detected,
                            stage_summary=_stage_summary(report), geometry=report))
        if len(_canonical(results)) > MAX_OUTPUT_BYTES:
            raise ValueError("ablation output budget exceeded")
    result = dict(schema=SCHEMA, configuration=configuration(), config_sha256=CONFIG_SHA256,
                  frame_ids=[n for n, _, _ in data], frame_sha256={n: h for n, _, h in data},
                  frame_dimensions={n: list(image.shape[:2]) for n, image, _ in data},
                  input_rgb_bytes=total, total_descriptor_comparisons=comparisons, variants=results)
    result["canonical_sha256"] = _sha(_canonical(result))
    canonical_report_bytes(result)
    return result


def canonical_report_bytes(report: dict) -> bytes:
    """Replay all internal bindings without feature extraction or RANSAC."""
    _check_baseline()
    if not isinstance(report, dict) or set(report) != {"schema", "configuration", "config_sha256", "frame_ids",
            "frame_sha256", "frame_dimensions", "input_rgb_bytes", "total_descriptor_comparisons", "variants", "canonical_sha256"}:
        raise ValueError("malformed ablation fields")
    payload = dict(report)
    digest = payload.pop("canonical_sha256")
    if digest != _sha(_canonical(payload)):
        raise ValueError("ablation digest mismatch")
    if report["schema"] != SCHEMA or report["configuration"] != configuration() or report["config_sha256"] != CONFIG_SHA256 or _sha(_canonical(report["configuration"])) != CONFIG_SHA256:
        raise ValueError("ablation configuration drift")
    results = report["variants"]
    if not isinstance(results, list) or len(results) != MAX_VARIANTS:
        raise ValueError("variant budget exceeded or ledger missing")
    dimensions = report["frame_dimensions"]
    if (not isinstance(dimensions, dict) or set(dimensions) != set(report["frame_ids"]) or
            any(not isinstance(shape, list) or len(shape) != 2 or
                any(type(value) is not int or value < 64 for value in shape) or
                shape[0] * shape[1] > geometry.MAX_PIXELS_PER_FRAME
                for shape in dimensions.values()) or
            type(report["input_rgb_bytes"]) is not int or
            report["input_rgb_bytes"] != 3 * sum(h * w for h, w in dimensions.values())):
        raise ValueError("ablation frame dimensions drift")
    comparisons = 0
    for variant, row in zip(VARIANTS, results):
        if not isinstance(row, dict) or set(row) != {"id", "availability", "reason", "feature_points", "detected_feature_counts", "stage_summary", "geometry"} or row["id"] != variant[0]:
            raise ValueError("malformed variant identity")
        unavailable = variant[1] == "akaze" and not hasattr(cv2, "AKAZE_create")
        if unavailable:
            if row != dict(id=variant[0], availability="unavailable", reason="OpenCV API AKAZE_create absent",
                           feature_points=None, detected_feature_counts=None, stage_summary=None, geometry=None):
                raise ValueError("malformed unavailable variant")
            continue
        if row["availability"] != "available" or row["reason"] is not None:
            raise ValueError("malformed available variant")
        geo = row["geometry"]
        geometry.canonical_report_bytes(geo)
        if (report["frame_ids"] != geo["frame_ids"] or report["frame_sha256"] != geo["frame_sha256"] or
                dimensions != geo["frame_dimensions"] or
                type(report["input_rgb_bytes"]) is not int or report["input_rgb_bytes"] != geo["work"]["input_rgb_bytes"]):
            raise ValueError("variant frame binding drift")
        comparisons += geo["work"]["descriptor_comparisons"]
        points, detected = row["feature_points"], row["detected_feature_counts"]
        if not isinstance(points, dict) or not isinstance(detected, dict) or set(points) != set(geo["frame_ids"]) or set(detected) != set(points):
            raise ValueError("malformed feature binding")
        for name in geo["frame_ids"]:
            h, w = geo["frame_dimensions"][name]
            xy = points[name]
            if (not isinstance(xy, list) or len(xy) != geo["work"]["feature_counts"][name] or
                    type(detected[name]) is not int or not len(xy) <= detected[name] <= h * w * 8 or
                    any(not isinstance(p, list) or len(p) != 2 or
                        any(type(v) not in (int, float) or not math.isfinite(v) for v in p) or
                        not 0 <= p[0] < w or not 0 <= p[1] < h for p in xy)):
                raise ValueError("malformed feature coordinates or counts")
        for track in geo["tracks"]:
            for name in track["frame_ids"]:
                for feature in track["feature_points"][name]:
                    if feature[1:] != points[name][feature[0]]:
                        raise ValueError("track coordinates contradict extracted features")
        if _canonical(row["stage_summary"]) != _canonical(_stage_summary(geo)):
            raise ValueError("malformed stage summary")
    if (type(report["total_descriptor_comparisons"]) is not int or
            report["total_descriptor_comparisons"] != comparisons or comparisons > MAX_TOTAL_DESCRIPTOR_COMPARISONS):
        raise ValueError("ablation descriptor accounting drift")
    raw = _canonical(report)
    if len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError("ablation output budget exceeded")
    return raw


def verify_frame_binding(report: dict, frames: Sequence[tuple[str, np.ndarray]]) -> None:
    canonical_report_bytes(report)
    data, total = geometry._frame_input(frames)
    actual_dimensions = {name: list(image.shape[:2]) for name, image, _ in data}
    if ([n for n, _, _ in data] != report["frame_ids"] or
            {n: h for n, _, h in data} != report["frame_sha256"] or
            actual_dimensions != report["frame_dimensions"] or
            any(row["geometry"]["frame_dimensions"] != actual_dimensions
                for row in report["variants"] if row["availability"] == "available") or
            total != report["input_rgb_bytes"]):
        raise ValueError("ablation frame source drift")


def persist_report(path: str | Path, report: dict,
                   frames: Sequence[tuple[str, np.ndarray]]) -> None:
    verify_frame_binding(report, frames)
    raw = canonical_report_bytes(report)
    target = Path(path)
    if not target.parent.is_dir():
        raise ValueError("output parent does not exist")
    fd, temporary = tempfile.mkstemp(prefix=".matcher-ablation-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            if out.write(raw) != len(raw):
                raise OSError("partial report write")
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)


# Evaluation-only generated fixtures. These transforms are not passed to diagnose.
def make_synthetic_suite() -> list[dict]:
    rng = np.random.default_rng(260108)
    base = np.full((320, 320, 3), 35, np.uint8)
    for _ in range(280):
        x, y = rng.integers(12, 308, 2)
        cv2.circle(base, (int(x), int(y)), int(rng.integers(2, 6)),
                   tuple(int(v) for v in rng.integers(65, 240, 3)), -1)
    transforms = [np.eye(3), np.array([[.96, .10, 12], [-.10, .96, 22], [.00015, -.00010, 1]]),
                  np.array([[1.03, -.06, -10], [.05, 1.00, 3], [-.00012, .00008, 1]])]
    cases = []
    for name, brightness, noise, occlusion in [("projective", 1., 0, False),
             ("brightness", .70, 0, False), ("noise", 1., 8, False), ("occlusion", 1., 3, True)]:
        frames = []
        for i, h in enumerate(transforms):
            image = cv2.warpPerspective(base, h, (320, 320), borderValue=(35, 35, 35))
            if i:
                image = np.clip(image.astype(float) * brightness + rng.integers(-noise, noise + 1, image.shape), 0, 255).astype(np.uint8)
            if occlusion and i == 1:
                image[125:190, 125:190] = 35
            frames.append((f"view-{i}", image))
        cases.append(dict(id=name, kind="positive", frames=frames, transforms=transforms))
    # Identical texels with incompatible per-tile permutations and independent
    # distractor blocks: no global scene correspondence is supplied by the repeat.
    for case_index, name in enumerate(("repeated-tiles", "independent-block-textures")):
        frames = []
        tile = rng.integers(20, 235, (32, 32, 3), dtype=np.uint8)
        for i in range(3):
            image = np.zeros((320, 320, 3), np.uint8)
            for y in range(10):
                for x in range(10):
                    patch = np.rot90(tile, int(rng.integers(0, 4))).copy() if not case_index else rng.integers(20, 235, (32, 32, 3), dtype=np.uint8)
                    image[y*32:(y+1)*32, x*32:(x+1)*32] = patch
            frames.append((f"view-{i}", image))
        cases.append(dict(id=name, kind="negative", frames=frames, transforms=None))
    return cases


def _oracle_pair(points: dict, names: list, transforms: list, proposals: list) -> tuple[int, int]:
    a, b = names
    left, right = np.asarray(points[a], float).reshape(-1, 2), np.asarray(points[b], float).reshape(-1, 2)
    h = transforms[int(b[-1])] @ np.linalg.inv(transforms[int(a[-1])])
    inverse = np.linalg.inv(h)
    forward, reverse = geometry._transfer(h, left), geometry._transfer(inverse, right)
    forward_distance = np.linalg.norm(forward[:, None, :] - right[None, :, :], axis=2)
    reverse_distance = np.linalg.norm(left[:, None, :] - reverse[None, :, :], axis=2)
    correct = (forward_distance <= 3) & (reverse_distance <= 3)
    eligible = int(np.any(correct, axis=1).sum())
    true = sum(bool(correct[i, j]) for i, j in proposals)
    return int(true), int(eligible)


def calibrate_synthetic() -> dict:
    """Separate evaluation entry point: freeze a candidate using generated truth."""
    cases = make_synthetic_suite()
    records = []
    for case in cases:
        report = diagnose(case["frames"])
        rows = []
        for variant in report["variants"]:
            if variant["availability"] == "unavailable":
                rows.append(dict(id=variant["id"], available=False, supported=False, proposal_count=0,
                                 true_proposals=0, eligible_sources=0, accepted_count=0,
                                 true_accepted=0, stage_summary=None))
                continue
            geo, points = variant["geometry"], variant["feature_points"]
            proposals = true_proposals = eligible = accepted = true_accepted = 0
            for pair in geo["pair_telemetry"]:
                proposals += pair["reciprocal_count"]
                if case["kind"] == "positive":
                    truth, denominator = _oracle_pair(points, pair["frame_ids"], case["transforms"], pair["reciprocal_pairs"])
                    true_proposals += truth
                    eligible += denominator
                for attempt in pair["attempts"]:
                    if attempt["first_rejection"] != "accepted":
                        continue
                    support = [pair["reciprocal_pairs"][index] for index in attempt["fit_inlier_indices"]]
                    support += [pair["reciprocal_pairs"][row[0]] for row in attempt["held_rows"] if row[1] != "inf" and row[2] != "inf" and row[1] <= 3 and row[2] <= 3]
                    accepted += len(support)
                    if case["kind"] == "positive":
                        true_accepted += _oracle_pair(points, pair["frame_ids"], case["transforms"], support)[0]
            rows.append(dict(id=variant["id"], available=True, supported=(geo["status"] == "supported" if case["kind"] == "positive" else accepted > 0), proposal_count=proposals,
                             true_proposals=true_proposals, eligible_sources=eligible, accepted_count=accepted,
                             true_accepted=true_accepted, stage_summary=variant["stage_summary"]))
        records.append(dict(id=case["id"], kind=case["kind"], input_sha256=report["frame_sha256"],
                            report_sha256=_sha(canonical_report_bytes(report)), metrics=rows))
    rankings = []
    for index, variant in enumerate(VARIANTS):
        positive = [case["metrics"][index] for case in records if case["kind"] == "positive"]
        negative = [case["metrics"][index] for case in records if case["kind"] == "negative"]
        supported = sum(row["supported"] for row in positive)
        false_support = sum(row["supported"] for row in negative)
        accepted = sum(row["accepted_count"] for row in positive)
        true_accepted = sum(row["true_accepted"] for row in positive)
        eligible = sum(row["eligible_sources"] for row in positive)
        true_proposals = sum(row["true_proposals"] for row in positive)
        precision = true_accepted / accepted if accepted else 0.
        recall = true_proposals / eligible if eligible else 0.
        passed = supported >= GATES["positive_cases_supported_min"] and precision >= GATES["accepted_precision_min"] and false_support <= GATES["negative_cases_with_accepted_pair_max"]
        rankings.append(dict(id=variant[0], available=all(row["available"] for row in positive+negative), positive_supported=supported, negative_supported=false_support,
                             accepted_precision=precision, proposal_recall=recall, passing=passed,
                             accepted_count=accepted, true_accepted=true_accepted, true_proposals=true_proposals,
                             eligible_sources=eligible))
    passing = [i for i, row in enumerate(rankings) if row["passing"]]
    chosen = min(passing, key=lambda i: (-rankings[i]["positive_supported"], -rankings[i]["proposal_recall"],
                                      -rankings[i]["accepted_precision"], i)) if passing else None
    result = dict(schema="opencv-matcher-synthetic-calibration-v1", config_sha256=CONFIG_SHA256,
                  configuration=configuration(), cases=records, rankings=rankings,
                  chosen_variant=rankings[chosen]["id"] if chosen is not None else None,
                  selection_basis="generated-pixels-only; no ordinary images or scene truth")
    result["canonical_sha256"] = _sha(_canonical(result))
    return result
