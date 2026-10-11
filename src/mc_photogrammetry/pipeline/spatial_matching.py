"""Opt-in pixel/mask-only spatial SIFT candidate; ordinary matcher unchanged.

The candidate allocates a bounded feature budget round-robin across spatial
cells instead of allowing one high-response patch to consume it. It never
changes descriptor matching, geometry, acceptance thresholds or camera inputs.
"""
from __future__ import annotations

import numpy as np

from .correspondence import (
    CorrespondenceSettings, FrameFeatures, _to_grey, _validate_frame,
    descriptor_footprint_keep, usable_mask_for,
)
from .joint_matching import JointMatchSettings, match_feature_frames


def _balanced_indices(points, responses, *, width, height, limit, grid=(4, 4)):
    """Stable strongest-first queues; descriptor orientation behavior unchanged."""
    columns = np.clip((points[:, 0] * grid[0] / width).astype(int), 0, grid[0] - 1)
    rows = np.clip((points[:, 1] * grid[1] / height).astype(int), 0, grid[1] - 1)
    cells = columns + grid[0] * rows
    order = np.lexsort((np.arange(len(points)), points[:, 1], points[:, 0], -responses))
    queues = [[] for _ in range(grid[0] * grid[1])]
    for index in order:
        queues[int(cells[index])].append(int(index))
    selected = []
    rank = 0
    quota = min(250, (limit + len(queues) - 1) // len(queues))
    while len(selected) < limit and rank < quota:
        added = False
        for queue in queues:
            if rank < len(queue):
                selected.append(queue[rank])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        rank += 1
    return np.asarray(selected, dtype=np.int64)


def _stage_census(points, width, height):
    columns = np.clip((points[:, 0] * 4 / width).astype(int), 0, 3)
    rows = np.clip((points[:, 1] * 4 / height).astype(int), 0, 3)
    return {'count': len(points), 'cells': np.bincount(columns + 4 * rows, minlength=16).tolist()}


def detect_spatial_features(frame_id, frame, *, usable_mask=None, settings=None, _census=None):
    """Detect on full masked grayscale, balance before descriptor computation.

    Native frame size is bounded at six million pixels; final features <=4000,
    250/cell and 4000 total. More than 64000 detector proposals refuses before
    descriptor computation. No excluded RGB reaches SIFT. Detecting unlimited
    response candidates is necessary: SIFT applies its strongest-feature cap
    before its detector mask, so repeated per-cell masked capped calls do not
    establish independent cell quotas.
    """
    import cv2

    if _census is not None and type(_census) is not dict:
        raise ValueError('census must be a private diagnostic dict')
    bounds = CorrespondenceSettings() if settings is None else settings
    if type(bounds.max_features) is not int or not 1 <= bounds.max_features <= 4000:
        raise ValueError("spatial candidate max_features must be an integer in 1..4000")
    image = _validate_frame(frame)
    height, width = image.shape[:2]
    if height * width > 6_000_000:
        raise ValueError("spatial candidate native pixel budget exceeded")
    usable = usable_mask_for(frame_id, image.shape, None) if usable_mask is None else np.asarray(usable_mask, dtype=bool)
    if usable.shape != (height, width):
        raise ValueError("usable mask must match the image dimensions")
    grey = _to_grey(image).copy()
    grey[~usable] = 0
    detector = cv2.SIFT_create(nfeatures=0)
    keypoints = detector.detect(grey, (usable * 255).astype(np.uint8))
    if len(keypoints) > 64000:
        raise ValueError("spatial candidate detector proposal budget exceeded")
    if not keypoints:
        if _census is not None:
            empty = np.empty((0, 2))
            _census.update({stage: _stage_census(empty, width, height) for stage in
                ('raw_sift_proposals', 'footprint_retained', 'quota_selected', 'descriptor_returned')})
        return FrameFeatures(frame_id, np.empty((0, 2)), np.empty((0, 0), dtype=np.float32),
                             width, height, int(usable.sum()), sizes=np.empty(0))
    points = np.asarray([key.pt for key in keypoints], dtype=np.float64)
    sizes = np.asarray([key.size for key in keypoints], dtype=np.float64)
    responses = np.asarray([key.response for key in keypoints], dtype=np.float64)
    if _census is not None:
        _census['raw_sift_proposals'] = _stage_census(points, width, height)
    keep = descriptor_footprint_keep(points, sizes, usable, bounds.mask_boundary_pad_px)
    dropped = int(np.count_nonzero(~keep))
    valid = np.flatnonzero(keep)
    points, sizes, responses = points[keep], sizes[keep], responses[keep]
    if _census is not None:
        _census['footprint_retained'] = _stage_census(points, width, height)
    chosen = _balanced_indices(points, responses, width=width, height=height,
                               limit=bounds.max_features)
    if _census is not None:
        _census['quota_selected'] = _stage_census(points[chosen], width, height)
    computed, descriptors = detector.compute(grey, [keypoints[int(valid[index])] for index in chosen])
    if descriptors is None:
        if _census is not None:
            _census['descriptor_returned'] = _stage_census(np.empty((0, 2)), width, height)
        return FrameFeatures(frame_id, np.empty((0, 2)), np.empty((0, 0), dtype=np.float32),
                             width, height, int(usable.sum()), sizes=np.empty(0), boundary_dropped=dropped)
    computed_points = np.asarray([key.pt for key in computed], dtype=np.float64).reshape(-1, 2)
    if len(computed_points) != len(descriptors) or len(computed_points) > len(chosen):
        raise ValueError('descriptor return violates selected feature census')
    if _census is not None:
        _census['descriptor_returned'] = _stage_census(computed_points, width, height)
    return FrameFeatures(frame_id, computed_points,
                         descriptors, width, height, int(usable.sum()),
                         sizes=np.asarray([key.size for key in computed]), boundary_dropped=dropped)


def match_and_solve_spatial_pair(left_id, left_rgb, right_id, right_rgb, *,
                                left_camera, right_camera, usable_masks=None,
                                settings=None, matcher=None, gates=None):
    """Explicit alternative; production's match_and_solve_pair is untouched."""
    bounds = JointMatchSettings() if matcher is None else matcher
    detection = CorrespondenceSettings(max_features=bounds.max_features,
                                       mask_boundary_pad_px=bounds.mask_boundary_pad_px) if settings is None else settings
    left = detect_spatial_features(left_id, left_rgb,
        usable_mask=usable_mask_for(left_id, left_rgb.shape, usable_masks), settings=detection)
    right = detect_spatial_features(right_id, right_rgb,
        usable_mask=usable_mask_for(right_id, right_rgb.shape, usable_masks), settings=detection)
    return match_feature_frames(left, right, left_camera, right_camera,
                               matcher=bounds, gates=gates, ratio_test=detection.ratio_test)
