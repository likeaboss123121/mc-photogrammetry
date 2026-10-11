"""Unpromoted bounded cross-view union diagnostic; no matcher or seed authority.

Only already-selected same-pass correspondences are consumed. A consistent
projective map is necessary evidence, not proof of physical coplanarity or
correct texture identities. Missing pose/ambiguity evidence causes abstention.
"""
from __future__ import annotations

from itertools import combinations, islice, product
import hashlib
import math

import numpy as np
import cv2

from mc_photogrammetry.pipeline.image_derived_connections import (
    _PairCorrespondenceObservation, _array_sha256, _MATCH_DIGEST_DOMAIN)
from mc_photogrammetry.reconstruction import region_growth as rg


def _project(matrix, points):
    mapped = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(mapped[:, 2]) < 1e-9):
        return None
    return mapped[:, :2] / mapped[:, 2, None]


def _fit(source, target):
    """Four-point normalized DLT, rejecting collinear/ill-conditioned sets."""
    transforms, normalized = [], []
    for points in (source, target):
        centered = points - points.mean(axis=0)
        if np.linalg.matrix_rank(centered, tol=1e-8) < 2:
            return None
        scale = math.sqrt(2) / max(float(np.linalg.norm(centered, axis=1).mean()), 1e-9)
        transform = np.array(((scale, 0, -scale * points[:, 0].mean()),
                              (0, scale, -scale * points[:, 1].mean()), (0, 0, 1)))
        transforms.append(transform)
        normalized.append(_project(transform, points))
    rows = []
    for (x, y), (u, v) in zip(*normalized):
        rows.extend(([-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                     [0, 0, 0, -x, -y, -1, v*x, v*y, v]))
    _, singular, vectors = np.linalg.svd(np.asarray(rows), full_matrices=True)
    if singular[-1] <= singular[0] * 1e-10:
        return None
    matrix = np.linalg.inv(transforms[1]) @ vectors[-1].reshape(3, 3) @ transforms[0]
    if abs(matrix[2, 2]) < 1e-9 or abs(np.linalg.det(matrix)) < 1e-12:
        return None
    return matrix / matrix[2, 2]


def diagnose_plane_union(observation, snapshot, union):
    """Return scalar-only necessary-consistency evidence for two adjacent atoms.

    Membership and adjacency are derived internally from source-bound bytes.
    Fixed experimental tolerance is 1 pixel, not a production threshold change.
    Training and holdout are separated within EACH constituent region; every
    held-out match must pass, with noncollinear coverage in each region.
    """
    report = dict(status="abstain", reason="", correspondences=0,
                  hypotheses=0, fitting_hypotheses=0, heldout_counts=(0, 0),
                  max_heldout_error=None, diagnostic_only=True)
    report.update(source_join=False, seed_authority=False,
                  appearance_identity_verified=False, physical_coplanarity_proven=False)
    def refuse(reason):
        report["reason"] = reason
        return report
    if type(observation) is not _PairCorrespondenceObservation:
        raise ValueError("source-bound pair observation required")
    if (type(union) is not tuple or len(union) != 2
            or any(type(node) is not int or not 0 <= node < 20_000 for node in union)
            or union[0] == union[1]):
        raise ValueError("two distinct atom ids required")
    if (observation.complete is not True or type(observation.source_count) is not int
            or observation.source_count != len(observation.matches)
            or len(observation.matches) > 64):
        return refuse("incomplete_correspondences")
    if type(snapshot) is not rg._DiagnosticAtomSnapshot:
        return refuse("missing_source_atom_snapshot")
    if (snapshot.frame_id != observation.left_id
            or snapshot.session_sha256 != observation.session_sha256
            or snapshot.pixel_sha256 != observation.left_pixel_sha256
            or snapshot.source_shape != observation.left_shape):
        return refuse("source_frame_binding_mismatch")
    shape = snapshot.working_shape
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(v) is not int or not 0 < v <= 960 for v in shape)
            or type(snapshot.labels_bytes) is not bytes or type(snapshot.usable_bytes) is not bytes
            or len(snapshot.labels_bytes) != math.prod(shape)*4
            or len(snapshot.usable_bytes) != math.prod(shape)):
        return refuse("invalid_atom_snapshot_bound")
    if (type(snapshot.source_shape) is not tuple or len(snapshot.source_shape) not in (2, 3)
            or any(type(v) is not int or v <= 0 for v in snapshot.source_shape)
            or math.prod(snapshot.source_shape[:2]) > 6_000_000
            or type(snapshot.native_usable_bytes) is not bytes
            or len(snapshot.native_usable_bytes) != math.prod(snapshot.source_shape[:2])):
        return refuse("invalid_native_mask_bound")
    if (hashlib.sha256(snapshot.labels_bytes).hexdigest() != snapshot.labels_sha256
            or hashlib.sha256(snapshot.usable_bytes).hexdigest() != snapshot.usable_sha256
            or rg._diagnostic_snapshot_digest(snapshot) != snapshot.snapshot_sha256):
        return refuse("atom_snapshot_digest_mismatch")
    if (rg._diagnostic_native_mask_digest(snapshot.frame_id, snapshot.source_shape,
            snapshot.native_usable_bytes) != snapshot.native_usable_sha256):
        return refuse("native_gui_mask_digest_mismatch")
    expected_scale = (snapshot.source_shape[1]/shape[1], snapshot.source_shape[0]/shape[0])
    if (snapshot.native_to_working_scale != expected_scale or snapshot.coordinate_rule !=
            "native_pixel_centers_plus_half_divided_by_axis_scale_minus_half_numpy_rint_abstain_edges"):
        return refuse("source_coordinate_mapping_mismatch")
    labels = np.frombuffer(snapshot.labels_bytes, dtype="<i4").reshape(shape)
    usable = np.frombuffer(snapshot.usable_bytes, dtype=bool).reshape(shape)
    native_usable = np.frombuffer(snapshot.native_usable_bytes, dtype=bool).reshape(snapshot.source_shape[:2])
    resized_mask = (native_usable if native_usable.shape == shape else
        cv2.resize(native_usable.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0)
    if not np.array_equal(resized_mask, usable):
        return refuse("native_working_mask_mismatch")
    if (not np.array_equal(labels >= 0, usable) or np.any(labels < -1)
            or np.any(labels >= 20_000)):
        return refuse("atom_mask_label_mismatch")
    points = np.asarray(observation.matches, dtype=np.float64).reshape(-1, 4)
    if (not np.isfinite(points).all() or _array_sha256(points, _MATCH_DIGEST_DOMAIN)
            != observation.matches_sha256):
        return refuse("pair_coordinate_digest_mismatch")
    native = points[:, :2]
    if np.any(native < 0) or np.any(native > (snapshot.source_shape[1]-1,
                                             snapshot.source_shape[0]-1)):
        return refuse("pair_coordinates_outside_native_frame")
    native_indices = np.rint(native).astype(int)
    if np.any(~native_usable[native_indices[:, 1], native_indices[:, 0]]):
        return refuse("correspondence_in_unusable_atom")
    memberships = rg._diagnostic_atom_membership(labels,
        rg._diagnostic_native_to_working(native, expected_scale))
    if any(node < 0 for node in memberships):
        return refuse("correspondence_in_unusable_atom")
    adjacent = any(np.any(((first == union[0]) & (second == union[1]))
                         | ((first == union[1]) & (second == union[0])))
                   for first, second in ((labels[:, :-1], labels[:, 1:]),
                                         (labels[:-1, :], labels[1:, :])))
    if not adjacent:
        return refuse("nonneighboring_atoms")
    report["source_join"] = True
    if observation.status != "solved":
        return refuse("pair_not_solved")
    if (observation.parallax_degrees is None or not math.isfinite(observation.parallax_degrees)
            or observation.parallax_degrees < 3.):
        return refuse("rotation_or_unmeasured_parallax")
    if observation.ambiguity_margin is None or observation.ambiguity_margin < 2:
        return refuse("pair_ambiguity_unresolved")
    selected = [i for i, node in enumerate(memberships) if node in union]
    report["correspondences"] = len(selected)
    groups = [[i for i in selected if memberships[i] == node] for node in union]
    if min(map(len, groups)) < 6:
        return refuse("insufficient_bilateral_support")
    if any(len(np.unique(points[selected, part], axis=0)) != len(selected)
           for part in (slice(0, 2), slice(2, 4))):
        return refuse("repeated_texture_or_duplicate_coordinates")
    training = [group[::2] for group in groups]
    holdout = [group[1::2] for group in groups]
    report["heldout_counts"] = tuple(map(len, holdout))
    accepted = []
    test_ids = [i for group in holdout for i in group]
    train_ids = [i for group in training for i in group]
    # Streaming product avoids materializing O(n^4) model combinations.
    hypotheses = product(combinations(training[0], 2), combinations(training[1], 2))
    model_count = math.comb(len(training[0]), 2) * math.comb(len(training[1]), 2)
    for first, second in islice(hypotheses, 128):
        report["hypotheses"] += 1
        ids = list(first + second)
        try:
            model = _fit(points[ids, :2], points[ids, 2:])
        except np.linalg.LinAlgError:
            return refuse("numerical_refusal")
        if model is None:
            continue
        predicted = _project(model, points[train_ids, :2])
        if predicted is None:
            continue
        errors = np.linalg.norm(predicted - points[train_ids, 2:], axis=1)
        if not np.isfinite(errors).all() or np.max(errors) > 1.:
            continue
        report["fitting_hypotheses"] += 1
        if accepted:
            probes = np.array(((points[train_ids, 0].min(), points[train_ids, 1].min()),
                               (points[train_ids, 0].max(), points[train_ids, 1].max()),
                               (points[train_ids, 0].min(), points[train_ids, 1].max()),
                               (points[train_ids, 0].max(), points[train_ids, 1].min())))
            old, new = _project(accepted[0], probes), _project(model, probes)
            if old is None or new is None or np.max(np.linalg.norm(old-new, axis=1)) > 1.:
                return refuse("multiple_projective_models")
        else:
            accepted.append(model)
    if model_count > 128:
        return refuse("hypothesis_budget_exhausted")
    if not accepted:
        return refuse("no_training_projective_model")
    # Freeze exactly one training-only choice. Holdout never resolves ambiguity,
    # selects a candidate, changes tolerance, refits or retries another model.
    frozen = accepted[0].copy()
    frozen.setflags(write=False)
    for group in holdout:
        for part in (slice(0, 2), slice(2, 4)):
            values = points[group, part]
            if np.linalg.matrix_rank(values-values.mean(axis=0), tol=1e-8) < 2:
                return refuse("collinear_heldout_support")
    prediction = _project(frozen, points[test_ids, :2])
    if prediction is None:
        return refuse("heldout_projection_invalid")
    maximum = float(np.linalg.norm(prediction-points[test_ids, 2:], axis=1).max())
    report["max_heldout_error"] = maximum
    if not math.isfinite(maximum) or maximum > 1.:
        return refuse("bilateral_heldout_rejected")
    # A coherent wrong one-to-one texture shift can fit exactly the same model.
    # Pair-level yaw/baseline ambiguity margins do not certify feature identity.
    report.update(status="appearance_only", reason="training_frozen_bilateral_holdout_passed")
    return report
