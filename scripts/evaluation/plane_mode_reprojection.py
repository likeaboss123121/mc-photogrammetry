"""Synthetic/evaluation-only projective evidence; no production decision policy.

Conventions: X_right = R X_left + t and n_left.T X_left = d.
H = K_right (R + t n_left.T / d) inverse(K_left).
For camera-centre displacement delta expressed in LEFT camera coordinates,
t = -R delta. World-coordinate displacement first requires R_left delta_world.
Uniformly scaling all d and t leaves H unchanged; metric scale stays unresolved.
Source seals bind supplied observations, not their acquisition or truthfulness.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np

MAX_MODES, MAX_PLANES, MAX_PAIRS = 2, 3, 32
MAX_CHECKS, MAX_OUTPUT_BYTES = 192, 64 * 1024
NUMERIC_EPS = 1e-12
ROTATION_EPS = 1e-10
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Pair:
    pair_id: str
    left_feature_id: str
    right_feature_id: str
    left_xy: tuple
    right_xy: tuple


@dataclass(frozen=True)
class Roster:
    explanation_id: str
    left_source_sha256: str
    right_source_sha256: str
    pairs: tuple


@dataclass(frozen=True)
class Observations:
    plane_id: str
    explanations: tuple


@dataclass(frozen=True)
class Plane:
    plane_id: str
    normal_left: tuple
    distance_left: float


@dataclass(frozen=True)
class Mode:
    mode_id: str
    rotation_right_from_left: tuple
    translation_right_from_left: tuple
    planes: tuple


class _Refusal(ValueError):
    pass


def _require(ok, reason):
    if not ok:
        raise _Refusal(reason)


def _identifier(value):
    _require(type(value) is str and _ID.fullmatch(value) is not None, "malformed_identity")


def _vector(value, size):
    _require(type(value) is tuple and len(value) == size, "malformed_shape")
    return tuple(_scalar(v) for v in value)


def _scalar(value):
    """Declared real scalars only; normalize without losing integer source IDs."""
    _require(not isinstance(value, (bool, np.bool_)), "malformed_numeric")
    _require(type(value) in (int, float) or isinstance(value, (np.integer, np.floating)),
             "malformed_numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise _Refusal("nonfinite_input") from None
    _require(math.isfinite(result), "nonfinite_input")
    # The numerical kernels use float64, but the source seal must retain every
    # supplied integer bit instead of collapsing adjacent values above 2**53.
    return int(value) if type(value) is int or isinstance(value, np.integer) else result


def _matrix(value):
    _require(type(value) is tuple and len(value) == 3, "malformed_shape")
    return tuple(_vector(row, 3) for row in value)


def _observations(observations):
    """Bound counts and validate immutable scalar inputs before numerical arrays."""
    _require(type(observations) is tuple, "malformed_input")
    _require(0 < len(observations) <= MAX_PLANES, "input_limit")
    plane_ids, total = set(), 0
    for plane in observations:
        _require(type(plane) is Observations, "malformed_input")
        _identifier(plane.plane_id)
        _require(plane.plane_id not in plane_ids, "duplicate_identity")
        plane_ids.add(plane.plane_id)
        _require(type(plane.explanations) is tuple, "malformed_input")
        _require(0 < len(plane.explanations) <= MAX_PAIRS, "input_limit")
        count = 0
        for roster in plane.explanations:
            _require(type(roster) is Roster and type(roster.pairs) is tuple, "malformed_input")
            _require(0 < len(roster.pairs) <= MAX_PAIRS, "input_limit")
            count += len(roster.pairs)
        _require(count <= MAX_PAIRS, "input_limit")
        total += count
        explanations = set()
        for roster in plane.explanations:
            _identifier(roster.explanation_id)
            _require(roster.explanation_id not in explanations, "duplicate_identity")
            explanations.add(roster.explanation_id)
            for source in (roster.left_source_sha256, roster.right_source_sha256):
                _require(type(source) is str and _HEX.fullmatch(source) is not None, "malformed_source")
            pair_ids, features = set(), set()
            for pair in roster.pairs:
                _require(type(pair) is Pair, "malformed_input")
                for identity in (pair.pair_id, pair.left_feature_id, pair.right_feature_id):
                    _identifier(identity)
                feature_pair = pair.left_feature_id, pair.right_feature_id
                _require(pair.pair_id not in pair_ids and feature_pair not in features, "duplicate_identity")
                pair_ids.add(pair.pair_id)
                features.add(feature_pair)
                _vector(pair.left_xy, 2)
                _vector(pair.right_xy, 2)
    return total


def _source_payload(observations):
    return [[p.plane_id, [[r.explanation_id, r.left_source_sha256, r.right_source_sha256,
        [[v.pair_id, v.left_feature_id, v.right_feature_id, list(_vector(v.left_xy, 2)), list(_vector(v.right_xy, 2))]
         for v in sorted(r.pairs, key=lambda v: v.pair_id)]]
        for r in sorted(p.explanations, key=lambda r: r.explanation_id)]]
        for p in sorted(observations, key=lambda p: p.plane_id)]


def _encode(value):
    def real_scalar(item):
        if isinstance(item, (np.integer, np.floating)):
            return _scalar(item)
        raise TypeError("not a declared JSON scalar")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=real_scalar).encode()


def source_digest(observations):
    """Order-independent identity seal; caller must retain the independent anchor."""
    _observations(observations)
    return hashlib.sha256(_encode(_source_payload(observations))).hexdigest()


def _rotation(matrix):
    value = np.asarray(matrix, dtype=np.float64)
    _require(np.max(np.abs(value.T @ value - np.eye(3))) <= ROTATION_EPS
             and abs(float(np.linalg.det(value)) - 1.0) <= ROTATION_EPS, "degenerate_pose")
    return value


def camera_center_to_translation(rotation_right_from_left, delta_in_left_coordinates):
    """Candidate camera-centre delta is converted as t=-R delta, never t=delta."""
    _matrix(rotation_right_from_left)
    _vector(delta_in_left_coordinates, 3)
    rotation = _rotation(rotation_right_from_left)
    return tuple(float(v) for v in -rotation @ np.asarray(delta_in_left_coordinates))


def _calibration(matrix):
    value = np.asarray(matrix, dtype=np.float64)
    _require(value[0, 0] > 0 and value[1, 1] > 0 and value[1, 0] == 0
             and np.array_equal(value[2], (0.0, 0.0, 1.0)), "degenerate_calibration")
    inverse = np.linalg.inv(value)
    _require(np.isfinite(inverse).all(), "degenerate_calibration")
    return value, inverse


def check_modes(*, calibration_left, calibration_right, modes, observations, expected_source_digest):
    """One atomic bounded request over every mode/plane/roster/point.

    Return raw residuals and source seals only. No mode ranking, point deletion,
    numerical-residual acceptance flag or metric-scale claim. Multiple rosters
    are retained even when one explains the pixels less accurately than another.
    """
    checks = 0
    try:
        _require(type(modes) is tuple and 0 < len(modes) <= MAX_MODES, "input_limit")
        total = _observations(observations)
        _require(len(modes) * total <= MAX_CHECKS, "input_limit")
        _matrix(calibration_left)
        _matrix(calibration_right)
        plane_ids = {p.plane_id for p in observations}
        mode_ids = set()
        for mode in modes:
            _require(type(mode) is Mode and type(mode.planes) is tuple, "malformed_input")
            _identifier(mode.mode_id)
            _require(mode.mode_id not in mode_ids, "duplicate_identity")
            mode_ids.add(mode.mode_id)
            _require(0 < len(mode.planes) <= MAX_PLANES, "input_limit")
            _matrix(mode.rotation_right_from_left)
            _vector(mode.translation_right_from_left, 3)
            ids = set()
            for plane in mode.planes:
                _require(type(plane) is Plane, "malformed_input")
                _identifier(plane.plane_id)
                _require(plane.plane_id not in ids, "duplicate_identity")
                ids.add(plane.plane_id)
                _vector(plane.normal_left, 3)
                _require(_scalar(plane.distance_left) != 0.0, "degenerate_plane")
            _require(ids == plane_ids, "plane_roster_mismatch")
        actual = hashlib.sha256(_encode(_source_payload(observations))).hexdigest()
        _require(type(expected_source_digest) is str and _HEX.fullmatch(expected_source_digest) is not None,
                 "malformed_source")
        _require(actual == expected_source_digest, "source_binding_mismatch")
        kl, il = _calibration(calibration_left)
        kr, ir = _calibration(calibration_right)
        output = []
        for mode in sorted(modes, key=lambda m: m.mode_id):
            rotation = _rotation(mode.rotation_right_from_left)
            translation = np.asarray(mode.translation_right_from_left, dtype=np.float64)
            planes = {p.plane_id: p for p in mode.planes}
            measured = []
            for observation in sorted(observations, key=lambda p: p.plane_id):
                plane = planes[observation.plane_id]
                normal = np.asarray(plane.normal_left, dtype=np.float64)
                distance_left = _scalar(plane.distance_left)
                _require(abs(float(np.linalg.norm(normal)) - 1.0) <= ROTATION_EPS, "degenerate_plane")
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    homography = kr @ (rotation + np.outer(translation, normal) / distance_left) @ il
                _require(np.isfinite(homography).all(), "nonfinite_geometry")
                _require(abs(float(np.linalg.det(homography))) > NUMERIC_EPS, "degenerate_homography")
                inverse = np.linalg.inv(homography)
                _require(np.isfinite(inverse).all(), "nonfinite_geometry")
                normal_right = rotation @ normal
                distance_right = distance_left + float(normal_right @ translation)
                explanations = []
                for roster in sorted(observation.explanations, key=lambda r: r.explanation_id):
                    residuals = []
                    for pair in sorted(roster.pairs, key=lambda p: p.pair_id):
                        _require(checks < MAX_CHECKS, "input_limit")
                        checks += 1
                        left = np.asarray((*pair.left_xy, 1.0))
                        right = np.asarray((*pair.right_xy, 1.0))
                        ray_left, ray_right = il @ left, ir @ right
                        denominator_left = float(normal @ ray_left)
                        denominator_right = float(normal_right @ ray_right)
                        _require(abs(denominator_left) > NUMERIC_EPS
                                 and abs(denominator_right) > NUMERIC_EPS, "degenerate_ray")
                        depth_left = distance_left / denominator_left
                        depth_right = distance_right / denominator_right
                        point_left = depth_left * ray_left
                        point_right = rotation @ point_left + translation
                        reverse_left = rotation.T @ (depth_right * ray_right - translation)
                        depths = (depth_left, depth_right, float(point_right[2]), float(reverse_left[2]))
                        _require(all(math.isfinite(d) for d in depths), "nonfinite_geometry")
                        _require(all(d > NUMERIC_EPS for d in depths), "behind_camera")
                        forward, backward = homography @ left, inverse @ right
                        _require(np.isfinite(forward).all() and np.isfinite(backward).all()
                                 and abs(float(forward[2])) > NUMERIC_EPS
                                 and abs(float(backward[2])) > NUMERIC_EPS, "degenerate_projection")
                        fwd = float(np.linalg.norm(forward[:2] / forward[2] - right[:2]))
                        back = float(np.linalg.norm(backward[:2] / backward[2] - left[:2]))
                        symmetric = math.hypot(fwd, back) / math.sqrt(2.0)
                        _require(all(math.isfinite(v) for v in (fwd, back, symmetric)), "nonfinite_geometry")
                        residuals.append(dict(pair_id=pair.pair_id, left_feature_id=pair.left_feature_id,
                            right_feature_id=pair.right_feature_id, forward_pixels=fwd, backward_pixels=back,
                            symmetric_pixels=symmetric, left_depth=depth_left, right_depth=depth_right,
                            finite_and_positive_depth=True))
                    explanations.append(dict(explanation_id=roster.explanation_id, residuals=residuals))
                measured.append(dict(plane_id=plane.plane_id, homography=homography.tolist(),
                    correspondence_state="multiple_explanations" if len(explanations) > 1 else "one_explanation",
                    explanations=explanations))
            candidate_payload = [mode.mode_id, _matrix(calibration_left), _matrix(calibration_right),
                _matrix(mode.rotation_right_from_left), _vector(mode.translation_right_from_left, 3),
                [[p.plane_id, _vector(p.normal_left, 3), _scalar(p.distance_left)]
                 for p in sorted(mode.planes, key=lambda p: p.plane_id)]]
            output.append(dict(mode_id=mode.mode_id, planes=measured,
                candidate_sha256=hashlib.sha256(_encode(candidate_payload)).hexdigest()))
        body = dict(status="raw_evidence", source_sha256=actual, modes=output,
                    point_checks=checks, metric_scale="unresolved_uniform_global_scale",
                    semantics="raw_residuals_no_ranking_or_production_acceptance")
        _require(len(_encode(body)) <= MAX_OUTPUT_BYTES, "output_limit")
        return body
    except (_Refusal, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        # No partial residuals, winning mode, or retained point subset survives.
        reason = str(error) if isinstance(error, _Refusal) else "nonfinite_or_singular_geometry"
        return dict(status="abstained", reason=reason, modes=[], point_checks=checks,
                    metric_scale="unresolved_uniform_global_scale")
