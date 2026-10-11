"""GUI-BORDER-01 evaluation-only preregistration; no production integration.

The only treatment is an ordered detector-return subset derived from the one
automatic mask used by the public RGB entry. Study execution needs lead review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import gzip
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import zlib

import cv2
import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from scripts.evaluation import run_camera_mixed_depth as cam


class BorderStudyError(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def require(condition, reason):
    if not condition:
        raise BorderStudyError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT/'docs/reports/GUI-BORDER-01-2026-10-05.md'
EVIDENCE_PATH = ROOT/'docs/reports/GUI-BORDER-01-evidence-2026-10-05.jsonl.gz'
FILTER_RULE = {'boundary': 'both pixels of adjacent in-image mask-value transitions; no perimeter',
    'dilation_chebyshev_pixels': 3, 'samples': 'ceil(Euclidean length)+1; endpoints included',
    'rounding': 'floor(coordinate+0.5); out-of-image samples count as non-overlap',
    'exclude_fraction_at_least': .5, 'whole_segments': True,
    'minimum_support': 'unchanged estimator refusal below 12; no axis veto'}
CAPS = {'max_segments': 10_000, 'max_samples_per_segment': 20_001,
        'max_total_samples': 50_000_000, 'max_record_bytes': 200_000_000,
        'max_compressed_member_bytes': 16*1024*1024}
PILOT_GATE = {'minimum_primary_B_passes': 9, 'minimum_B_minus_A_passes': 3,
              'maximum_lost_A_pass_families': 0, 'maximum_added_stress_wrong_emissions': 0,
              'gui_only_prediction_stable_within_each_arm': True}
_ARM_LOCK = threading.Lock()
_WRITER_LOCK = threading.Lock()
_STUDY_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Scene:
    identity: str
    family: str
    stratum: str
    renderer: str
    recipe_json: str

    @property
    def sha256(self):
        return sha(canonical(asdict(self)))


def scenes():
    result = [Scene('dev-'+s.identity, s.identity, 'development', 'cam-exposed', s.recipe_json)
              for s in cam.SCENES if s.physical]
    result.append(Scene('dev-input-identifiable', 'input-identifiable', 'development',
                        'input-exposed', cam.legacy.RECIPE_JSON))
    settings = ((11., -21., 52.), (38., -9., 67.), (62., -27., 79.), (81., -16., 61.))
    for kind in ('planar', 'stepped', 'occluding'):
        for number, (yaw, pitch, fov) in enumerate(settings):
            seed = 610_500+100*('planar', 'stepped', 'occluding').index(kind)+number
            rng = np.random.default_rng(seed)
            eye = [-.65+.21*number, 2.65+.17*number, -1.4-.13*number]
            forward = np.array([-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))])
            recipe = {'version': 'GUI-BORDER-01-world-native-v1', 'rng': 'numpy-PCG64',
                'seed': seed, 'texture_seed': seed+70_000, 'shape': [360, 480],
                'yaw': yaw, 'pitch': pitch, 'vertical_fov': fov,
                'focal': 180./math.tan(math.radians(fov)/2), 'roll': 0.,
                'principal': [240., 180.], 'eye': eye, 'native_side': 16,
                'back_plane': {'axis': 2, 'side': -1,
                    'value': eye[2]+forward[1]*(14.4+.7*number),
                    'bounds': [[-55., 20.], [0., 15.]]},
                'floor': kind != 'planar', 'boxes': [], 'crop': None,
                'low_texture': False, 'gui_contamination': False,
                'seam_width_blocks': .014+.002*number}
            if kind != 'planar':
                for depth in (4.1, 7.4, 10.3):
                    centre = np.array([eye[0], eye[2]])+forward*depth
                    centre += rng.uniform(-.7, .7, 2)
                    size = rng.uniform([1.5, 1.2], [2.7, 2.4])
                    recipe['boxes'].append({'lo': [float(centre[0]-size[0]/2), 0.,
                                                   float(centre[1]-size[1]/2)],
                        'hi': [float(centre[0]+size[0]/2), float(rng.uniform(1.4, 3.6)),
                               float(centre[1]+size[1]/2)]})
            if kind == 'occluding':
                centre = np.array([eye[0], eye[2]])+forward*2.6
                recipe['boxes'].append({'lo': [float(centre[0]-1.1), 0., float(centre[1]-.7)],
                    'hi': [float(centre[0]+.8), 4.15+.1*number, float(centre[1]+.75)]})
            family = f'heldout-{kind}-{number+1}'
            for variant in ('primary', 'crop', 'low-texture', 'gui-only'):
                changed = json.loads(canonical(recipe))
                if variant == 'crop':
                    changed['crop'] = [37, 19, 418, 314]
                changed['low_texture'] = variant == 'low-texture'
                changed['gui_contamination'] = variant == 'gui-only'
                result.append(Scene(f'{family}-{variant}', family, variant, 'new-world', canonical(changed)))
    base = json.loads(next(s.recipe_json for s in result if s.identity == 'heldout-planar-1-primary'))
    for name, seam in (('hud-free', .011), ('structure-crosses-mask', .038)):
        changed = dict(base, seed=930_100, texture_seed=930_200, seam_width_blocks=seam,
                       yaw=19., pitch=-18., eye=[-.4, 2.9, -1.8])
        result.append(Scene('control-'+name, name, 'control', 'new-world', canonical(changed)))
    return tuple(result)


SCENES = scenes()
MAX_EVIDENCE_BYTES = 512*1024*1024


def native_rgb(points, axis, side, recipe):
    """A frozen independent native texture per signed plane; no inferred camera."""
    tangents = [i for i in range(3) if i != axis]
    fractions = np.asarray(points)[..., tangents] % 1.
    if recipe['low_texture']:
        return np.full(fractions.shape[:-1]+(3,), 117.+axis*9.+side*3.), np.zeros(fractions.shape[:-1], bool)
    native = recipe['native_side']
    rng = np.random.default_rng(recipe['texture_seed']+axis*101+(side+1)*17)
    phases = rng.uniform(-math.pi, math.pi, (3, 3))
    yy, xx = np.meshgrid((np.arange(native)+.5)/native, (np.arange(native)+.5)/native, indexing='ij')
    tex = np.stack([121+axis*5+side*3+24*np.sin(2*np.pi*xx+phases[c, 0])
                    +19*np.cos(2*np.pi*yy+phases[c, 1])
                    +13*np.sin(2*np.pi*(2*xx-yy)+phases[c, 2]) for c in range(3)], axis=-1)
    tex = np.rint(tex).astype(np.uint8)
    xy = fractions*native-.5
    base = np.floor(xy).astype(int)
    mix = xy-base
    color = np.zeros(fractions.shape[:-1]+(3,))
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        weight = (mix[..., 0] if dx else 1-mix[..., 0])*(mix[..., 1] if dy else 1-mix[..., 1])
        color += weight[..., None]*tex[(base[..., 1]+dy) % native, (base[..., 0]+dx) % native]
    seams = (np.minimum(fractions, 1-fractions) <= recipe['seam_width_blocks']).any(axis=-1)
    color[seams] = 31.
    return color, seams


def render_new(recipe):
    h, w = recipe['shape']
    yy, xx = np.indices((h, w))
    rays = np.stack(((xx-recipe['principal'][0])/recipe['focal'],
                     -(yy-recipe['principal'][1])/recipe['focal'], np.ones((h, w))), axis=-1)
    faces = new_faces(recipe)
    directions = rays @ cam.literal_rotation(recipe['yaw'], recipe['pitch'], recipe['roll'])
    owners, depth, points = cam.intersect_rays(recipe['eye'], directions, faces)
    rgb = np.full((h, w, 3), 96, np.uint8)
    seams = np.zeros((h, w), bool)
    for index, face in enumerate(faces):
        selected = owners == index
        colors, edges = native_rgb(points[selected], face['axis'], face['side'], recipe)
        rgb[selected] = np.rint(colors).astype(np.uint8)
        seams[selected] = edges
    if recipe['crop'] is not None:
        x, y, width, height = recipe['crop']
        rgb, owners, depth, seams = [a[y:y+height, x:x+width].copy() for a in (rgb, owners, depth, seams)]
    if recipe['gui_contamination']:
        mask = cam.create_gui_mask(*rgb.shape[:2])
        yy, xx = np.indices(mask.shape)
        tint = np.where(((xx//7+yy//11) % 2)[..., None], [241, 34, 187], [26, 225, 211])
        rgb[mask] = tint[mask]
    return rgb, owners, depth, seams


def new_faces(recipe):
    faces = list(cam.face_planes(dict(recipe, floor=False)))
    if recipe['floor']:
        faces.append({'axis': 1, 'value': 0., 'side': 1, 'bounds': [[-55., 20.], [-3., 25.]]})
    return faces


def render_rgb(scene):
    if scene.renderer == 'input-exposed':
        return cam.legacy.render_rgb('identifiable')[0]
    if scene.renderer == 'cam-exposed':
        return cam.physical_render(json.loads(scene.recipe_json))[0]
    require(scene.renderer == 'new-world', 'unknown_renderer')
    return render_new(json.loads(scene.recipe_json))[0]


def truth_camera(scene):
    if scene.renderer == 'input-exposed':
        recipe = json.loads(scene.recipe_json)
        return {'shape': recipe['shape'], 'focal': recipe['renderer_focal'],
                'yaw': recipe['renderer_yaw'], 'pitch': recipe['renderer_pitch'], 'roll': 0.,
                'principal': [recipe['shape'][1]/2., recipe['shape'][0]/2.]}
    recipe = json.loads(scene.recipe_json)
    h, w = recipe['shape']
    point = list(recipe['principal'])
    if recipe['crop'] is not None:
        x, y, w, h = recipe['crop']
        point = [point[0]-x, point[1]-y]
    return {key: recipe[key] for key in ('focal', 'yaw', 'pitch', 'roll')} | {'shape': [h, w], 'principal': point}


def mask_boundary(mask):
    require(isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.ndim == 2
            and min(mask.shape) > 0, 'invalid_automatic_mask')
    border = np.zeros(mask.shape, np.uint8)
    horizontal, vertical = mask[:, 1:] != mask[:, :-1], mask[1:] != mask[:-1]
    border[:, 1:] |= horizontal
    border[:, :-1] |= horizontal
    border[1:] |= vertical
    border[:-1] |= vertical
    return cv2.dilate(border, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT,
                      borderValue=0).astype(bool)


def select_segments(raw, mask):
    require(isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[1] == 4
            and raw.dtype.kind in 'fiu' and np.isfinite(raw).all(), 'invalid_raw_segments')
    require(len(raw) <= CAPS['max_segments'], 'segment_cap')
    boundary = mask_boundary(mask)
    lengths = np.hypot(raw[:, 2].astype(float)-raw[:, 0], raw[:, 3].astype(float)-raw[:, 1])
    require(np.isfinite(lengths).all(), 'nonfinite_segment_length')
    sample_counts = [math.ceil(float(length))+1 for length in lengths]
    require(all(n <= CAPS['max_samples_per_segment'] for n in sample_counts), 'segment_sample_cap')
    require(sum(sample_counts) <= CAPS['max_total_samples'], 'total_sample_cap')
    h, w = mask.shape
    overlapping = []
    for row, count in zip(raw, sample_counts):
        alpha = np.linspace(0., 1., count)
        xs = np.floor(float(row[0])+(float(row[2])-float(row[0]))*alpha+.5)
        ys = np.floor(float(row[1])+(float(row[3])-float(row[1]))*alpha+.5)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        overlap = np.zeros(count, bool)
        overlap[inside] = boundary[ys[inside].astype(int), xs[inside].astype(int)]
        overlapping.append(int(overlap.sum()))
    kept = np.array([i for i, (hits, count) in enumerate(zip(overlapping, sample_counts))
                     if 2*hits < count], dtype=np.int64)
    return kept, {'sample_counts': sample_counts, 'overlap_counts': overlapping,
                  'total_samples': sum(sample_counts), 'boundary_sha256': sha(boundary.tobytes())}


def snapshot(array):
    require(isinstance(array, np.ndarray) and array.dtype.kind in 'biuf' and np.isfinite(array).all(),
            'invalid_snapshot')
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'bytes_hex': array.tobytes().hex()}


def decode(record):
    dtype = np.dtype(record['dtype'])
    shape = record['shape']
    require(dtype.kind in 'biuf' and all(type(n) is int and n >= 0 for n in shape), 'invalid_snapshot_type')
    raw = bytes.fromhex(record['bytes_hex'])
    require(len(raw) == math.prod(shape)*dtype.itemsize, 'snapshot_size_mismatch')
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    require(np.isfinite(array).all(), 'snapshot_nonfinite')
    return array


def trace_arm(rgb, identity, arm):
    """One public entry call; the treatment receives its actual delegated mask."""
    verify_preregistration()
    require(arm in ('A', 'B'), 'invalid_arm')
    require(_ARM_LOCK.acquire(blocking=False), 'arm_reentrancy_or_concurrency')
    original_detector = cam.calibration.detect_segments
    original_mask = cam.gui_mask.create_gui_mask
    owner = threading.get_ident()
    captures = {}
    started = time.perf_counter()

    def capture_mask(*args, **kwargs):
        require(threading.get_ident() == owner and 'mask' not in captures, 'mask_repeated_or_cross_thread')
        found = original_mask(*args, **kwargs)
        captures['mask'] = snapshot(found)
        return found

    def detector(*args, **kwargs):
        require(threading.get_ident() == owner and 'raw' not in captures, 'detector_repeated_or_cross_thread')
        require('mask' in captures, 'detector_before_mask')
        found = original_detector(*args, **kwargs)
        before = snapshot(found)
        selected, sampling = select_segments(found, decode(captures['mask']))
        indices = np.arange(len(found), dtype=np.int64) if arm == 'A' else selected
        returned = found if arm == 'A' else found[indices].copy()
        require(snapshot(found) == before, 'raw_detector_mutated')
        captures.update(raw=before, retained=snapshot(returned), indices=indices.tolist(),
                        filter_indices=selected.tolist(), sampling=sampling)
        return returned

    try:
        cam.gui_mask.create_gui_mask = capture_mask
        cam.calibration.detect_segments = detector
        output = cam.trace_rgb(rgb, identity)
        require({'mask', 'raw', 'retained', 'indices', 'filter_indices', 'sampling'} <= captures.keys(),
                'incomplete_arm_capture')
        require(output['status'] != 'trace_refusal', 'incomplete_camera_trace:'+output['reason'])
        require(captures['mask'] == cam.replay_trace(output)['arrays'][next(
            event['output'] for event in output['trace']['stages'] if event['stage'] == 'create_gui_mask')],
            'automatic_mask_capture_mismatch')
        return {'arm': arm, 'output': output, 'support': captures, 'arm_seconds': time.perf_counter()-started}
    finally:
        cam.calibration.detect_segments = original_detector
        cam.gui_mask.create_gui_mask = original_mask
        _ARM_LOCK.release()


def stage_array(result, name):
    events = [event for event in result['output']['trace']['stages'] if event['stage'] == name]
    require(len(events) == 1, 'stage_parity_capture_missing')
    return result['output']['trace']['arrays'][events[0]['output']]


def verify_arm(record):
    require(record['arm'] in ('A', 'B'), 'invalid_arm')
    cam.replay_trace(record['output'])
    require(record['output']['status'] != 'trace_refusal', 'incomplete_camera_trace')
    support = record['support']
    raw, mask, retained = (decode(support[key]) for key in ('raw', 'mask', 'retained'))
    selected, sampling = select_segments(raw, mask)
    require(sampling == support['sampling'] and selected.tolist() == support['filter_indices'], 'filter_replay_mismatch')
    expected = np.arange(len(raw), dtype=np.int64) if record['arm'] == 'A' else selected
    require(support['indices'] == expected.tolist() and retained.dtype == raw.dtype
            and np.array_equal(retained, raw[expected]), 'ordered_raw_subset_mismatch')
    require(stage_array(record, 'create_gui_mask') == support['mask']
            and stage_array(record, 'detect_segments') == support['retained'], 'support_trace_mismatch')
    require(record['output']['prediction'] is None or len(retained) >= 12, 'prediction_below_support_minimum')
    if record['output']['prediction'] is not None:
        require(record['output']['prediction']['segments'] == len(retained), 'prediction_support_count_mismatch')
    detection_sha = next(e['output'] for e in record['output']['trace']['stages'] if e['stage'] == 'detect_segments')
    lengths = np.hypot(retained[:, 2]-retained[:, 0], retained[:, 3]-retained[:, 1])
    full_events = [event for event in record['output']['trace']['scores'] if event['site'] != 'coarse']
    require({event['segments_sha256'] for event in full_events} <= {detection_sha}, 'full_objective_support_mismatch')
    for digest in {event['weights_sha256'] for event in full_events}:
        require(np.array_equal(cam.decode_array(record['output']['trace'], digest), lengths),
                'full_objective_support_mismatch')
    return record


@dataclass(frozen=True, slots=True)
class SealedPair:
    payload_json: str
    sha256: str


def seal_pair(scene, A, B):
    verify_arm(A)
    verify_arm(B)
    require(A['arm'] == 'A' and B['arm'] == 'B', 'pair_arm_order_mismatch')
    require(A['output']['rgb_shape'] == B['output']['rgb_shape']
            and A['output']['rgb_sha256'] == B['output']['rgb_sha256'] ==
            json.loads(PREREGISTRATION_JSON)['rgb_sha256'][scene.identity], 'pair_rgb_mismatch')
    require(A['support']['mask'] == B['support']['mask'] and A['support']['raw'] == B['support']['raw']
            and stage_array(A, 'enhance') == stage_array(B, 'enhance'), 'pair_upstream_parity_mismatch')
    payload = canonical({'scene': asdict(scene), 'scene_sha256': scene.sha256,
        'preregistration_sha256': PREREGISTRATION_SHA256, 'arms': {'A': A, 'B': B},
        'support_sha256': {arm: {key: sha(canonical(value['support'][key])) for key in ('raw', 'retained', 'mask')}
                           for arm, value in (('A', A), ('B', B))}, 'production_gates_changed': False})
    return SealedPair(payload, sha(payload))


def verify_pair(scene, sealed):
    require(sha(sealed.payload_json) == sealed.sha256, 'pair_seal_mismatch')
    record = json.loads(sealed.payload_json)
    require(record['scene'] == asdict(scene) and record['scene_sha256'] == scene.sha256
            and record['preregistration_sha256'] == PREREGISTRATION_SHA256, 'pair_recipe_binding_mismatch')
    regenerated = seal_pair(scene, record['arms']['A'], record['arms']['B'])
    require(regenerated == sealed, 'pair_payload_replay_mismatch')
    return record


def normalized_score(lines, camera):
    weights = np.hypot(lines[:, 2]-lines[:, 0], lines[:, 3]-lines[:, 1])
    report = cam.full_segment_score(lines, weights, camera)
    total = float(weights.sum())
    report.update(total_length=total, score_over_total_length=report['score']/total if total > 0 else None,
                  weak_axis_minimum=min(report['per_axis_inliers']),
                  axis_veto_applied=False, calibrated_confidence=None)
    return report


def evaluate_pair(scene, sealed):
    record = verify_pair(scene, sealed)
    exclusions = json.loads(PREREGISTRATION_JSON)['camera_accuracy_collision_exclusions']
    eligible = scene.identity not in exclusions
    truth = truth_camera(scene) if eligible else None  # Only after pair replay by registered runner.
    A = record['arms']['A']
    raw = decode(A['support']['raw'])
    filtered = raw[np.asarray(A['support']['filter_indices'], dtype=int)]
    scopes = {'original': raw, 'filtered': filtered}
    result = {}
    renderer_scores = {scope: normalized_score(lines, truth) for scope, lines in scopes.items()} if eligible else None
    for arm, value in record['arms'].items():
        prediction = value['output']['prediction']
        support = decode(value['support']['retained'])
        base = {'status': 'refused' if prediction is None else 'emitted', 'metrics': None,
            'reason': value['output']['reason'], 'raw_segments': len(raw), 'retained_segments': len(support),
            'retained_total_length': float(np.hypot(support[:, 2]-support[:, 0], support[:, 3]-support[:, 1]).sum()),
            'renderer_scores': renderer_scores, 'emitted_scores': None, 'stages': None,
            'arm_seconds': value['arm_seconds'], 'trace_seconds': value['output']['trace']['trace_seconds'],
            'score_delegate_seconds': value['output']['trace']['score_delegate_seconds'],
            'emitted_wrong': False, 'diagnostic_pass': False, 'axis_veto_applied': False}
        if prediction is not None:
            h, w = value['output']['rgb_shape'][:2]
            camera = {'focal': prediction['focal_pixels'], 'yaw': prediction['yaw_degrees'],
                'pitch': prediction['pitch_degrees'], 'roll': prediction['roll_degrees'], 'shape': [h, w],
                'principal': prediction['principal_point'] or [w/2., h/2.]}
            if eligible:
                base['metrics'] = cam.camera_metrics(prediction, truth)
                base['diagnostic_pass'] = base['metrics']['diagnostic_pass']
                base['emitted_wrong'] = not base['diagnostic_pass']
            base['emitted_scores'] = {scope: normalized_score(lines, camera) for scope, lines in scopes.items()}
            base['stages'] = cam.stage_camera_evidence(value['output'])
        result[arm] = base
    paired_errors = {key: result['B']['metrics'][key]-result['A']['metrics'][key]
                     if result['A']['metrics'] is not None and result['B']['metrics'] is not None else None
                     for key in ('focal_relative_error', 'orientation_degrees_mod90', 'principal_error_over_diagonal')}
    return {'scene': scene.identity, 'family': scene.family, 'stratum': scene.stratum, 'arms': result,
            'camera_accuracy_eligible': eligible,
            'camera_truth_scope': 'registered_camera' if eligible else 'equal_RGB_conflicting_camera_truth_unscored',
            'paired_B_minus_A': {'diagnostic_pass': int(result['B']['diagnostic_pass'])-int(result['A']['diagnostic_pass']),
                'wrong_emission': int(result['B']['emitted_wrong'])-int(result['A']['emitted_wrong']),
                'refusal': int(result['B']['status'] == 'refused')-int(result['A']['status'] == 'refused'),
                'seconds': result['B']['arm_seconds']-result['A']['arm_seconds'],
                'retained_segments': result['B']['retained_segments']-result['A']['retained_segments'],
                'retained_total_length': result['B']['retained_total_length']-result['A']['retained_total_length'],
                'camera_errors': paired_errors}}


def aggregate(evaluations, prediction_records):
    require([e['scene'] for e in evaluations] == [s.identity for s in SCENES], 'evaluation_registry_mismatch')
    primary = [e for e in evaluations if e['stratum'] == 'primary' and e['camera_accuracy_eligible']]
    require(len(primary) == 12 and len({e['family'] for e in primary}) == 12, 'primary_denominator_mismatch')
    stress = [e for e in evaluations if e['stratum'] in ('crop', 'low-texture', 'gui-only') and e['camera_accuracy_eligible']]
    passes = {arm: sum(e['arms'][arm]['diagnostic_pass'] for e in primary) for arm in ('A', 'B')}
    lost = [e['family'] for e in primary if e['arms']['A']['diagnostic_pass'] and not e['arms']['B']['diagnostic_pass']]
    added_wrong = [e['scene'] for e in stress if e['arms']['B']['emitted_wrong'] and not e['arms']['A']['emitted_wrong']]
    stable = {}
    for scene in SCENES:
        if scene.stratum == 'gui-only':
            base = scene.family+'-primary'
            stable[scene.family] = {arm: prediction_records[scene.identity][arm] == prediction_records[base][arm]
                                    for arm in ('A', 'B')}
    gate = passes['B'] >= 9 and passes['B']-passes['A'] >= 3 and not lost and not added_wrong
    gate = gate and all(all(value.values()) for value in stable.values())
    return {'primary_families': 12, 'primary_passes': passes, 'B_minus_A_passes': passes['B']-passes['A'],
        'lost_A_pass_families': lost, 'added_stress_wrong_emissions': added_wrong,
        'gui_only_prediction_stability': stable, 'pilot_gate_pass': bool(gate),
        'primary_counts': {arm: {'wrong': sum(e['arms'][arm]['emitted_wrong'] for e in primary),
                               'refused': sum(e['arms'][arm]['status'] == 'refused' for e in primary)} for arm in ('A', 'B')},
        'stress_counts': {arm: {'n_correlated_variants': len(stress),
            'pass': sum(e['arms'][arm]['diagnostic_pass'] for e in stress),
            'wrong': sum(e['arms'][arm]['emitted_wrong'] for e in stress),
            'refused': sum(e['arms'][arm]['status'] == 'refused' for e in stress)} for arm in ('A', 'B')},
        'camera_accuracy_excluded_scenes': [e['scene'] for e in evaluations if not e['camera_accuracy_eligible']],
        'stimulus_collisions': json.loads(PREREGISTRATION_JSON)['stimulus_collisions'],
        'refusal_is_accuracy_recovery': False, 'production_integration_authorized': False,
        'interpretation': 'pilot gate only; requires bounded integration review and independent real-screenshot evaluation'}


def evidence_header():
    return {'kind': 'GUI-BORDER-01-full-sealed-pairs-gzip-v2', 'preregistration_sha256': PREREGISTRATION_SHA256}


class CompressedBuffer(io.BytesIO):
    def write(self, value):
        require(self.tell()+len(value) <= CAPS['max_compressed_member_bytes'], 'compressed_member_cap')
        return super().write(value)


def compress_member(decoded):
    require(len(decoded) <= CAPS['max_record_bytes'], 'decompressed_record_cap')
    sink = CompressedBuffer()
    with gzip.GzipFile(fileobj=sink, mode='wb', compresslevel=6, mtime=0, filename='') as member:
        for start in range(0, len(decoded), 65_536):
            member.write(decoded[start:start+65_536])
    return sink.getvalue()


def gzip_members(artifact, digest=None):
    """Bound output allocation, member input and total input while decoding.

    zlib verifies each gzip CRC/footer. Unused bytes belong to the next member;
    only consumed bytes count towards the current member cap and digest.
    """
    pending, read_bytes, offset = b'', 0, 0
    def read_chunk():
        nonlocal read_bytes
        found = artifact.read(min(65_536, MAX_EVIDENCE_BYTES-read_bytes+1))
        read_bytes += len(found)
        require(read_bytes <= MAX_EVIDENCE_BYTES, 'compressed_total_cap')
        if digest is not None:
            digest.update(found)
        return found
    while True:
        if not pending:
            pending = read_chunk()
        if not pending:
            return
        decoder = zlib.decompressobj(16+zlib.MAX_WBITS)
        chunks, decoded_size, compressed_size, member_digest = [], 0, 0, hashlib.sha256()
        while True:
            if not pending:
                pending = read_chunk()
            require(bool(pending), 'gzip_truncated_member')
            try:
                found = decoder.decompress(pending, CAPS['max_record_bytes']-decoded_size+1)
            except zlib.error as error:
                raise BorderStudyError('gzip_integrity_error') from error
            decoded_size += len(found)
            require(decoded_size <= CAPS['max_record_bytes'] and not decoder.unconsumed_tail,
                    'decompressed_record_cap')
            consumed = len(pending)-len(decoder.unused_data)
            compressed_size += consumed
            require(compressed_size <= CAPS['max_compressed_member_bytes'], 'compressed_member_cap')
            member_digest.update(pending[:consumed])
            chunks.append(found)
            pending = decoder.unused_data
            if decoder.eof:
                break
        decoded = b''.join(chunks)
        descriptor = {'offset': offset, 'compressed_bytes': compressed_size,
                      'compressed_sha256': member_digest.hexdigest(), 'decoded_sha256': sha(decoded)}
        offset += compressed_size
        yield decoded, descriptor


def decode_member(raw):
    members = gzip_members(io.BytesIO(raw))
    first = next(members, None)
    require(first is not None and next(members, None) is None, 'gzip_member_count_mismatch')
    return first


def json_line(decoded):
    require(decoded.endswith(b'\n') and decoded.count(b'\n') == 1, 'member_jsonl_line_mismatch')
    try:
        return json.loads(decoded)
    except (ValueError, UnicodeError) as error:
        raise BorderStudyError('member_json_mismatch') from error


def pair_member(decoded):
    value = json_line(decoded)
    require(type(value) is dict and set(value) == {'payload_json', 'sha256'}
            and type(value['payload_json']) is str and type(value['sha256']) is str,
            'member_pair_shape_mismatch')
    return SealedPair(**value)


class EvidenceWriter:
    """Exclusive creation, one open owner, append only; never repair/truncate."""
    def __init__(self, path):
        self.path, self.artifact = path, None
        self.size, self.seals, self.members, self.digest = 0, [], [], hashlib.sha256()
    def __enter__(self):
        require(_WRITER_LOCK.acquire(blocking=False), 'evidence_writer_already_owned')
        try:
            self.owner = threading.get_ident()
            self.artifact = self.path.open('x+b')
            self.append_record((canonical(evidence_header())+'\n').encode())
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise
    def __exit__(self, *args):
        try:
            if self.artifact is not None:
                self.artifact.__exit__(*args)
                self.artifact = None
        finally:
            _WRITER_LOCK.release()
    def append_record(self, decoded):
        require(self.artifact is not None and _WRITER_LOCK.locked() and self.owner == threading.get_ident(),
                'evidence_writer_not_owned')
        compressed = compress_member(decoded)  # All caps precede artifact mutation.
        require(self.size+len(compressed) <= MAX_EVIDENCE_BYTES, 'compressed_total_cap')
        self.artifact.seek(0, os.SEEK_END)
        require(self.artifact.tell() == self.size, 'evidence_owner_size_mismatch')
        require(self.artifact.write(compressed) == len(compressed), 'evidence_short_write')
        self.artifact.flush()
        os.fsync(self.artifact.fileno())
        self.artifact.seek(self.size)
        replay = self.artifact.read(len(compressed))
        require(replay == compressed, 'appended_member_digest_mismatch')
        restored, descriptor = decode_member(replay)
        require(restored == decoded, 'appended_member_payload_mismatch')
        descriptor['offset'] = self.size
        self.members.append(descriptor)
        self.size += len(compressed)
        self.digest.update(compressed)
        return descriptor


def append_pair(scene, sealed, previous, *, writer):
    require(tuple(writer.seals) == tuple(previous) and len(previous) < len(SCENES)
            and scene == SCENES[len(previous)], 'evidence_scene_order_mismatch')
    verify_pair(scene, sealed)
    descriptor = writer.append_record((canonical(asdict(sealed))+'\n').encode())
    # Restore the actual newly fsynced member and validate its full sealed pair.
    require(descriptor['decoded_sha256'] == sha((canonical(asdict(sealed))+'\n').encode()),
            'appended_member_payload_mismatch')
    writer.seals.append(sealed.sha256)
    return descriptor


def replay_evidence(expected_seals, expected_digest=None):
    require(EVIDENCE_PATH.is_file(), 'evidence_missing')
    require(EVIDENCE_PATH.stat().st_size <= MAX_EVIDENCE_BYTES, 'compressed_total_cap')
    require(len(expected_seals) <= len(SCENES), 'evidence_prefix_mismatch')
    digest = hashlib.sha256()
    with EVIDENCE_PATH.open('rb') as artifact:
        members = gzip_members(artifact, digest)
        header = next(members, None)
        require(header is not None and json_line(header[0]) == evidence_header(), 'evidence_header_mismatch')
        for scene, expected in zip(SCENES, expected_seals):
            found = next(members, None)
            require(found is not None, 'evidence_prefix_mismatch')
            sealed = pair_member(found[0])
            require(sealed.sha256 == expected, 'evidence_seal_order_mismatch')
            verify_pair(scene, sealed)
        require(next(members, None) is None, 'evidence_prefix_mismatch')
    result = digest.hexdigest()
    require(expected_digest is None or result == expected_digest, 'artifact_digest_mismatch')
    return result


def read_pair_member(descriptor):
    require(EVIDENCE_PATH.is_file(), 'evidence_missing')
    require(0 <= descriptor['offset'] and 0 < descriptor['compressed_bytes'] <= CAPS['max_compressed_member_bytes'],
            'member_descriptor_mismatch')
    with EVIDENCE_PATH.open('rb') as artifact:
        artifact.seek(descriptor['offset'])
        raw = artifact.read(descriptor['compressed_bytes'])
    require(sha(raw) == descriptor['compressed_sha256'], 'appended_member_digest_mismatch')
    decoded, found = decode_member(raw)
    require(found['decoded_sha256'] == descriptor['decoded_sha256'], 'appended_member_payload_mismatch')
    return pair_member(decoded)


def artifact_digest():
    require(EVIDENCE_PATH.is_file(), 'evidence_missing')
    require(EVIDENCE_PATH.stat().st_size <= MAX_EVIDENCE_BYTES, 'compressed_total_cap')
    digest, size = hashlib.sha256(), 0
    with EVIDENCE_PATH.open('rb') as artifact:
        while True:
            chunk = artifact.read(65_536)
            if not chunk:
                break
            size += len(chunk)
            require(size <= MAX_EVIDENCE_BYTES, 'compressed_total_cap')
            digest.update(chunk)
    return digest.hexdigest()


def claim_report():
    require(REPORT_PATH.is_file(), 'preregistration_report_missing')
    with REPORT_PATH.open('r+', encoding='utf-8', newline='') as report:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(report.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: (report.seek(0), msvcrt.locking(report.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            fcntl.flock(report.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(report.fileno(), fcntl.LOCK_UN)
        try:
            content = report.read()
            require(PREREGISTRATION_SHA256 in content and content.count('GUI_BORDER_EXECUTION_STATE: NOT_STARTED') == 1,
                    'study_already_claimed_or_pin_mismatch')
            begin, end = 'PREREGISTRATION_BEGIN\n```json\n', '\n```\nPREREGISTRATION_END'
            require(content.count(begin) == content.count(end) == 1, 'report_preregistration_missing_or_ambiguous')
            registered = json.loads(content.split(begin)[1].split(end)[0])
            require(registered == json.loads(PREREGISTRATION_JSON), 'report_preregistration_payload_mismatch')
            report.seek(0)
            report.write(content.replace('GUI_BORDER_EXECUTION_STATE: NOT_STARTED', 'GUI_BORDER_EXECUTION_STATE: RUNNING'))
            report.truncate()
            report.flush()
            os.fsync(report.fileno())
        finally:
            unlock()


def run_study():
    require(_STUDY_LOCK.acquire(blocking=False), 'study_reentrancy_or_concurrency')
    try:
        return run_study_owned()
    finally:
        _STUDY_LOCK.release()


def run_study_owned():
    verify_preregistration()
    require(not EVIDENCE_PATH.exists(), 'evidence_already_exists')
    claim_report()  # Durable consumption BEFORE rendering/inference. No retry after a crash.
    with EvidenceWriter(EVIDENCE_PATH) as writer:
        for scene in SCENES:
            rgb = render_rgb(scene)
            A = trace_arm(rgb, scene.identity, 'A')
            B = trace_arm(rgb, scene.identity, 'B')
            sealed = seal_pair(scene, A, B)
            append_pair(scene, sealed, tuple(writer.seals), writer=writer)
        seals, member_descriptors = tuple(writer.seals), tuple(writer.members[1:])
        expected_digest = writer.digest.hexdigest()
    # Do not keep all bounded traces simultaneously in memory. Replay the whole
    # durable artifact before any truth, then evaluate one already sealed pair.
    artifact_sha = replay_evidence(seals, expected_digest)
    evaluations, predictions = [], {}
    for scene, expected, descriptor in zip(SCENES, seals, member_descriptors):
        sealed = read_pair_member(descriptor)
        require(sealed.sha256 == expected, 'evidence_seal_order_mismatch')
        record = verify_pair(scene, sealed)
        predictions[scene.identity] = {arm: value['output']['prediction'] for arm, value in record['arms'].items()}
        evaluations.append(evaluate_pair(scene, sealed))
    require(artifact_digest() == artifact_sha, 'evidence_mutated_during_evaluation')
    result = {'preregistration_sha256': PREREGISTRATION_SHA256, 'evidence_sha256': artifact_sha,
        'sealed_pairs': seals, 'gzip_pair_members': member_descriptors,
        'scenes': evaluations, 'aggregate': aggregate(evaluations, predictions),
        'production_gates_changed': False, 'real_screenshot_claim': False}
    with REPORT_PATH.open('r+', encoding='utf-8', newline='') as report:
        content = report.read()
        require(content.count('GUI_BORDER_EXECUTION_STATE: RUNNING') == 1, 'run_marker_drift')
        report.seek(0)
        report.write(content.replace('GUI_BORDER_EXECUTION_STATE: RUNNING', 'GUI_BORDER_EXECUTION_STATE: COMPLETE')+
                     '\n\nDurable registered result:\n\n```json\n'+canonical(result)+'\n```\n')
        report.truncate()
        report.flush()
        os.fsync(report.fileno())
    return result


def dependency_pins():
    result = cam.dependency_pins()
    result['source_sha256']['scripts/evaluation/run_camera_mixed_depth.py'] = sha(Path(cam.__file__).read_bytes())
    result['source_sha256']['scripts/evaluation/run_gui_border.py'] = sha(Path(__file__).read_bytes())
    return result


def stimulus_collisions(rgb_hashes):
    groups = {}
    for scene in SCENES:
        groups.setdefault(rgb_hashes[scene.identity], []).append(scene)
    collisions = []
    for digest, group in sorted(groups.items()):
        if len(group) > 1:
            camera_hashes = {s.identity: sha(canonical(truth_camera(s))) for s in group}
            geometry_hashes = {s.identity: sha(canonical(new_faces(json.loads(s.recipe_json))))
                               if s.renderer == 'new-world' else sha(s.recipe_json) for s in group}
            camera_conflict = len(set(camera_hashes.values())) > 1
            collisions.append({'rgb_sha256': digest, 'scenes': [s.identity for s in group],
                'camera_truth_sha256': camera_hashes, 'recipe_sha256': {s.identity: s.sha256 for s in group},
                'geometry_truth_sha256': geometry_hashes,
                'kind': 'image_identifiability_camera_collision' if camera_conflict else
                        'image_identifiability_geometry_collision_same_camera_truth',
                'camera_accuracy_eligible': not camera_conflict,
                'independent_observation_claim': False})
    return collisions


def freeze_preregistration():
    rgb_hashes = {s.identity: sha(render_rgb(s).tobytes()) for s in SCENES}
    primary_hashes = [rgb_hashes[s.identity] for s in SCENES if s.stratum == 'primary']
    require(len(primary_hashes) == len(set(primary_hashes)) == 12, 'primary_RGB_not_unique')
    collisions = stimulus_collisions(rgb_hashes)
    exclusions = sorted(identity for group in collisions if not group['camera_accuracy_eligible'] for identity in group['scenes'])
    return canonical({'version': 'GUI-BORDER-01-prereg-v1', 'scenes': [asdict(s) for s in SCENES],
        'dependencies': dependency_pins(), 'rgb_sha256': rgb_hashes,
        'stimulus_collisions': collisions, 'camera_accuracy_collision_exclusions': exclusions,
        'collision_policy': 'same-RGB/different-camera-truth groups wholly excluded from camera accuracy; retained/report all; primary RGBs unique',
        'filter': FILTER_RULE, 'caps': CAPS, 'trace_policy': asdict(cam.TracePolicy()), 'pilot_gate': PILOT_GATE,
        'max_evidence_bytes': MAX_EVIDENCE_BYTES, 'evidence_path': 'docs/reports/'+EVIDENCE_PATH.name,
        'evidence_format': 'gzip deterministic level6/mtime0 header member plus independent JSONL pair members; fsync/member',
        'replay': 'each appended member verified; one bounded full replay before truth; final compressed-byte digest',
        'truth_order': 'all-57-full-pairs-sealed-fsynced-replayed-before-any-truth-score',
        'one_shot': 'locked-fsynced-report-NOT_STARTED-to-RUNNING-before-exclusive-artifact-no-retry',
        'primary_denominator': 12, 'development_count': 7, 'correlated_stress_count': 36, 'control_count': 2,
        'diagnostic_tolerances': {'focal_relative': .05, 'orientation_degrees_mod90': 1., 'principal_diagonal': .01},
        'score_scopes': 'each emitted and renderer camera on original/filtered; normalize by own total length; no cross-set raw comparison',
        'stage_semantics': 'accepted CAM candidate_max is highest tested; actual selected transitions captured separately',
        'cost': 'each arm wall time; trace overhead and objective delegate separately; A also computes diagnostic filter',
        'inference_inputs': 'RGB-and-frame-identity-only', 'axis_veto': False, 'production_gates_changed': False,
        'refusal_is_accuracy_recovery': False, 'integration': 'pilot gate plus review plus independent real screenshots'})


PREREGISTRATION_JSON = None
PREREGISTRATION_SHA256 = None


def initialize_preregistration(payload=None):
    """Explicit freeze for setup; execution loads reviewed report, no rendering."""
    global PREREGISTRATION_JSON, PREREGISTRATION_SHA256
    if payload is None:
        require(REPORT_PATH.is_file(), 'preregistration_report_missing')
        content = REPORT_PATH.read_text(encoding='utf-8')
        begin, end = 'PREREGISTRATION_BEGIN\n```json\n', '\n```\nPREREGISTRATION_END'
        require(content.count(begin) == content.count(end) == 1, 'report_preregistration_missing_or_ambiguous')
        payload = content.split(begin)[1].split(end)[0]
        require('Preregistration SHA256: `'+sha(canonical(json.loads(payload)))+'`.' in content,
                'report_preregistration_digest_mismatch')
    PREREGISTRATION_JSON = canonical(json.loads(payload))
    PREREGISTRATION_SHA256 = sha(PREREGISTRATION_JSON)


def verify_preregistration():
    if PREREGISTRATION_JSON is None:
        initialize_preregistration()
    pinned = json.loads(PREREGISTRATION_JSON)
    require(dependency_pins() == pinned['dependencies'], 'dependency_or_runtime_drift')
    require([asdict(s) for s in SCENES] == pinned['scenes'] and FILTER_RULE == pinned['filter']
            and CAPS == pinned['caps'] and PILOT_GATE == pinned['pilot_gate']
            and MAX_EVIDENCE_BYTES == pinned['max_evidence_bytes']
            and asdict(cam.TracePolicy()) == pinned['trace_policy'], 'registry_or_policy_drift')


if __name__ == '__main__':
    print(canonical(run_study()))
