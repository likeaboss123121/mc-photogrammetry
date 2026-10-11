"""Stateless post-seal camera scoring; no import-time I/O or renderer.

The loader receives one approved opaque reference only after the durable barrier.
This is an audited in-process boundary, not protection from Python introspection.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from mc_photogrammetry.reconstruction.evaluation_valid_stencil_segments import (
    AcquisitionIntegrityError, AcquisitionCapError,
)
TRUTH_SCHEMA = 'CAM-VALID-camera-truth-v1'
TRUTH_MAX_BYTES = 1024*1024
GATES = {'focal_relative_error': .05, 'orientation_degrees_mod90': 1.,
    'principal_error_over_diagonal': .01}


def _require(condition, reason):
    if not condition:
        raise AcquisitionIntegrityError(reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load_truth(reference):
    """Read one bounded approved canonical absolute path once; no fallback."""
    _require(isinstance(reference, dict) and set(reference) == {'identity', 'reference', 'sha256'},
        'truth_reference_schema')
    path = Path(reference['reference'])
    _require(path.is_absolute() and str(path.resolve()) == reference['reference']
        and not any(p.is_symlink() for p in (path, *path.parents)), 'truth_path_authority')
    with path.open('rb') as stream:
        raw = stream.read(TRUTH_MAX_BYTES+1)
    if len(raw) > TRUTH_MAX_BYTES:
        raise AcquisitionCapError('truth_receipt_byte_cap')
    _require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'truth_reference_hash')
    return raw


def validate_truth(truth, row):
    keys = {'schema', 'identity', 'rgb_sha256', 'shape', 'focal', 'yaw', 'pitch', 'roll', 'principal'}
    _require(isinstance(truth, dict) and set(truth) == keys, 'truth_camera_schema')
    _require(isinstance(truth['identity'], str) and 0 < len(truth['identity']) <= 256
        and isinstance(truth['rgb_sha256'], str) and len(truth['rgb_sha256']) == 64
        and all(c in '0123456789abcdef' for c in truth['rgb_sha256']), 'truth_identity_schema')
    _require(truth['schema'] == TRUTH_SCHEMA and truth['identity'] == row['identity']
        and truth['rgb_sha256'] == row['rgb_sha256'] and truth['shape'] == row['rgb_shape'][:2],
        'truth_camera_binding')
    _require(isinstance(truth['shape'], list) and len(truth['shape']) == 2
        and all(type(v) is int and v > 0 for v in truth['shape']), 'truth_shape')
    _require(all(type(truth[k]) in (int, float) and math.isfinite(truth[k])
        for k in ('focal', 'yaw', 'pitch', 'roll')) and truth['focal'] > 0, 'truth_camera_finite')
    _require(isinstance(truth['principal'], list) and len(truth['principal']) == 2
        and all(type(v) in (int, float) and math.isfinite(v) for v in truth['principal']), 'truth_principal')
    return json.loads(_canonical(truth))

def literal_rotation(yaw, pitch, roll=0.):
    """Independent world-to-camera convention, never estimator camera input."""
    yaw, pitch, roll = map(math.radians, (yaw, pitch, roll))
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    basis = np.array(((-cy, 0., -sy), (-sp*sy, cp, sp*cy), (-cp*sy, -sp, cp*cy)))
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(((cr, -sr, 0.), (sr, cr, 0.), (0., 0., 1.))) @ basis

def camera_metrics(prediction, truth):
    """Diagnostic optics, modulo horizontal yaw gauge; no M3 gate alteration."""
    h, w = truth['shape']
    point = prediction['principal_point'] or [w/2., h/2.]
    rotation_true = literal_rotation(truth['yaw'], truth['pitch'], truth['roll'])
    candidates = []
    for quarter in range(4):
        yaw = prediction['yaw_degrees']+90*quarter
        rotation = literal_rotation(yaw, prediction['pitch_degrees'], prediction['roll_degrees'])
        cosine = np.clip((np.trace(rotation @ rotation_true.T)-1)/2, -1., 1.)
        candidates.append((math.degrees(math.acos(float(cosine))), quarter, rotation))
    angle, quarter, rotation = min(candidates, key=lambda x: (x[0], x[1]))
    focal_error = abs(prediction['focal_pixels']/truth['focal']-1.)
    principal_error = float(np.linalg.norm(np.asarray(point)-truth['principal']))/math.hypot(w, h)
    # Fixed angular probes condition on renderer eye only for projection error;
    # no translation or reconstructed-world accuracy is claimed.
    fractions = np.array([.1, .3, .5, .7, .9])
    xx, yy = np.meshgrid(fractions*w, fractions*h)
    rays = np.stack(((xx-truth['principal'][0])/truth['focal'],
                     -(yy-truth['principal'][1])/truth['focal'], np.ones_like(xx)), axis=-1)
    predicted_rays = rays @ rotation_true @ rotation.T
    forward = predicted_rays[..., 2] > 1e-7
    errors = None
    if forward.all():
        projected = np.stack((point[0]+prediction['focal_pixels']*predicted_rays[..., 0]/predicted_rays[..., 2],
                              point[1]-prediction['focal_pixels']*predicted_rays[..., 1]/predicted_rays[..., 2]), axis=-1)
        displacement = np.linalg.norm(projected-np.stack((xx, yy), axis=-1), axis=-1)
        errors = {'median': float(np.median(displacement)), 'p95': float(np.quantile(displacement, .95)),
                  'max': float(displacement.max())}
    return {'focal_relative_error': focal_error, 'orientation_degrees_mod90': angle,
        'principal_error_over_diagonal': principal_error, 'horizontal_gauge_quarter_turn': quarter,
        'projection_direction_error_pixels': errors, 'projection_behind_count': int((~forward).sum()),
        'diagnostic_pass': bool(focal_error <= .05 and angle <= 1. and principal_error <= .01)}


def evaluate_prediction(prediction, truth, gates):
    """Fixed stateless identity; independent input copies prevent cross-arm aliases."""
    _require(gates == GATES, 'scoring_gate_pin')
    _require(isinstance(prediction, dict) and isinstance(truth, dict), 'scoring_input_schema')
    shape = truth.get('shape')
    _require(isinstance(shape, list) and len(shape) == 2
        and all(type(v) is int and v > 0 for v in shape), 'truth_shape')
    validate_truth(truth, {'identity': truth.get('identity'), 'rgb_sha256': truth.get('rgb_sha256'),
        'rgb_shape': [*shape, 3]})
    _require(prediction.get('width') == shape[1] and prediction.get('height') == shape[0], 'scoring_prediction_shape')
    _require(all(type(prediction.get(k)) in (int, float) and math.isfinite(prediction[k])
        for k in ('focal_pixels', 'yaw_degrees', 'pitch_degrees', 'roll_degrees'))
        and prediction['focal_pixels'] > 0, 'scoring_prediction_finite')
    point = prediction.get('principal_point')
    _require(point is None or isinstance(point, list) and len(point) == 2
        and all(type(v) in (int, float) and math.isfinite(v) for v in point), 'scoring_prediction_principal')
    return camera_metrics(json.loads(_canonical(prediction)), json.loads(_canonical(truth)))
