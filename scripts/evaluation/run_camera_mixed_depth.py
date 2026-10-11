"""CAM-MD-01 evaluation only; experiment execution awaits independent review.

Renderer camera/geometry are stimulus and post-seal evaluation metadata only.
Inference accepts RGB and an identity; no camera, surface, region or label.
No production calibration/default/gate modification is made here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import PIL

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from mc_photogrammetry.vision.gui_mask import create_gui_mask
from mc_photogrammetry.vision import gui_mask
from mc_photogrammetry.reconstruction import calibration
from scripts.evaluation import run_unit_face_input as legacy


class CameraStudyError(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _require(condition, reason):
    if not condition:
        raise CameraStudyError(reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True, slots=True)
class CameraRecipe:
    identity: str
    physical: bool
    recipe_json: str

    @property
    def sha256(self):
        return _sha(_json(asdict(self)))


def _base_recipe():
    return {'version': 'CAM-MD-01-ray-world-v1', 'shape': [480, 640],
        'focal': 410., 'yaw': 27., 'pitch': -13., 'roll': 0.,
        'principal': [320., 240.], 'eye': [1.25, 3.15, -.5],
        'back_plane': {'axis': 2, 'value': 9., 'side': -1,
                       'bounds': [[-11., 8.], [0., 10.]]},
        'boxes': [], 'floor': False, 'crop': None, 'low_texture': False,
        'gui_contamination': False, 'native_side': 16, 'seam_width_blocks': .018}


def _scene_recipes():
    result = []
    for name in ('planar-positive', 'stepped', 'occluding', 'crop', 'low-texture', 'gui-contamination'):
        recipe = _base_recipe()
        if name != 'planar-positive':
            recipe['floor'] = True
            recipe['boxes'] = [
                {'lo': [-3., 0., 3.], 'hi': [-1., 2., 5.]},
                {'lo': [-6., 0., 6.], 'hi': [-3., 4., 8.]},
                {'lo': [0., 0., 5.], 'hi': [2., 3., 7.]},
            ]
        if name in ('occluding', 'crop', 'gui-contamination'):
            recipe['boxes'].append({'lo': [-1., 0., 1.5], 'hi': [.5, 3.5, 2.5]})
        if name == 'crop':
            recipe['crop'] = [38, 26, 560, 416]
        recipe['low_texture'] = name == 'low-texture'
        recipe['gui_contamination'] = name == 'gui-contamination'
        result.append(CameraRecipe('physical-'+name, True, _json(recipe)))
    result.append(CameraRecipe('legacy-composited-mixed-depth', False, legacy.RECIPE_JSON))
    return tuple(result)


SCENES = _scene_recipes()


def literal_rotation(yaw, pitch, roll=0.):
    """Independent world-to-camera convention, never estimator camera input."""
    yaw, pitch, roll = map(math.radians, (yaw, pitch, roll))
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    basis = np.array(((-cy, 0., -sy), (-sp*sy, cp, sp*cy), (-cp*sy, -sp, cp*cy)))
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(((cr, -sr, 0.), (sr, cr, 0.), (0., 0., 1.))) @ basis


def face_planes(recipe):
    """Finite axis-aligned rectangles; every box supplies all six faces."""
    faces = [dict(recipe['back_plane'])]
    if recipe['floor']:
        faces.append({'axis': 1, 'value': 0., 'side': 1,
                      'bounds': [[-11., 8.], [0., 12.]]})
    for box in recipe['boxes']:
        for axis in range(3):
            tangents = [i for i in range(3) if i != axis]
            bounds = [[box['lo'][i], box['hi'][i]] for i in tangents]
            for side, key in ((-1, 'lo'), (1, 'hi')):
                faces.append({'axis': axis, 'value': box[key][axis], 'side': side,
                              'bounds': bounds})
    return tuple(faces)


def intersect_rays(origin, directions, faces):
    """Nearest positive finite hit. Ray t is camera depth for z=1 rays."""
    directions = np.asarray(directions, dtype=float)
    origin = np.asarray(origin, dtype=float)
    _require(directions.shape[-1] == 3 and origin.shape == (3,), 'invalid_ray_shape')
    _require(np.isfinite(directions).all() and np.isfinite(origin).all(), 'nonfinite_ray')
    distance = np.full(directions.shape[:-1], np.inf)
    owners = np.full(directions.shape[:-1], -1, dtype=np.int32)
    for identity, face in enumerate(faces):
        axis = face['axis']
        denominator = directions[..., axis]
        t = np.zeros_like(distance)
        np.divide(face['value']-origin[axis], denominator, out=t, where=np.abs(denominator) > 1e-12)
        inside = (np.abs(denominator) > 1e-12) & (t > 1e-7) & (t < distance)
        for tangent, (low, high) in zip((i for i in range(3) if i != axis), face['bounds']):
            coordinate = origin[tangent]+t*directions[..., tangent]
            inside &= (coordinate >= low) & (coordinate <= high)
        distance[inside] = t[inside]
        owners[inside] = identity
    safe = np.where(owners >= 0, distance, 0.)
    points = origin + safe[..., None]*directions
    return owners, distance, points


def native_surface_rgb(points, axis, side, *, native_side=16, low_texture=False, seam_width=.018):
    """Absolute world tangents give one consistent block/UV gauge per plane."""
    tangents = [i for i in range(3) if i != axis]
    uv = np.asarray(points)[..., tangents]
    fractions = uv % 1.
    seams = (np.minimum(fractions, 1-fractions) <= seam_width).any(axis=-1)
    if low_texture:
        return np.full(uv.shape[:-1]+(3,), 126., dtype=float), np.zeros(uv.shape[:-1], bool)
    yy, xx = np.meshgrid((np.arange(native_side)+.5)/native_side,
                         (np.arange(native_side)+.5)/native_side, indexing='ij')
    tex = np.stack((132+28*np.sin(2*np.pi*xx)+12*np.cos(4*np.pi*yy),
                    141+22*np.cos(2*np.pi*yy+.4)+10*np.sin(2*np.pi*(xx+yy)),
                    118+25*np.sin(2*np.pi*(xx-yy)+.7)), axis=-1)
    tex += np.array((axis*7., side*5., -axis*5.))
    texture = np.rint(tex).astype(np.uint8)
    xy = fractions*native_side-.5
    base = np.floor(xy).astype(int)
    mix = xy-base
    color = np.zeros(uv.shape[:-1]+(3,))
    for dx, dy, weight in ((0, 0, (1-mix[..., 0])*(1-mix[..., 1])),
                          (1, 0, mix[..., 0]*(1-mix[..., 1])),
                          (0, 1, (1-mix[..., 0])*mix[..., 1]),
                          (1, 1, mix[..., 0]*mix[..., 1])):
        color += weight[..., None]*texture[(base[..., 1]+dy) % native_side,
                                          (base[..., 0]+dx) % native_side]
    color[seams] = 27.
    return color, seams


def physical_render(recipe):
    """Nearest-hit ray render; labels are kept out of the RGB-only entry."""
    h, w = recipe['shape']
    yy, xx = np.indices((h, w))
    cx, cy = recipe['principal']
    rays = np.stack(((xx-cx)/recipe['focal'], -(yy-cy)/recipe['focal'], np.ones((h, w))), axis=-1)
    directions = rays @ literal_rotation(recipe['yaw'], recipe['pitch'], recipe['roll'])
    faces = face_planes(recipe)
    owners, depth, points = intersect_rays(recipe['eye'], directions, faces)
    rgb = np.full((h, w, 3), 102, dtype=np.uint8)
    lattice = np.zeros((h, w), bool)
    for index, face in enumerate(faces):
        selected = owners == index
        color, seams = native_surface_rgb(points[selected], face['axis'], face['side'],
            native_side=recipe['native_side'], low_texture=recipe['low_texture'],
            seam_width=recipe['seam_width_blocks'])
        rgb[selected] = np.rint(color).astype(np.uint8)
        lattice[selected] = seams
    crop = recipe['crop']
    if crop is not None:
        x, y, width, height = crop
        rgb, owners, depth, lattice = (array[y:y+height, x:x+width].copy()
                                     for array in (rgb, owners, depth, lattice))
    if recipe['gui_contamination']:
        mask = create_gui_mask(*rgb.shape[:2])
        yy, xx = np.indices(mask.shape)
        tint = np.where(((xx//8+yy//8) % 2)[..., None], [248, 22, 196], [22, 238, 202])
        rgb[mask] = tint[mask]
    return rgb, owners, depth, lattice


def render_rgb(scene):
    if not scene.physical:
        return legacy.render_rgb('mixed_depth')[0]
    return physical_render(json.loads(scene.recipe_json))[0]


def truth_camera(scene):
    _require(scene.physical, 'legacy_compositor_outside_physical_camera_aggregate')
    recipe = json.loads(scene.recipe_json)
    cx, cy = recipe['principal']
    h, w = recipe['shape']
    if recipe['crop'] is not None:
        x, y, w, h = recipe['crop']
        cx, cy = cx-x, cy-y
    return {'focal': recipe['focal'], 'yaw': recipe['yaw'], 'pitch': recipe['pitch'],
            'roll': recipe['roll'], 'principal': [cx, cy], 'shape': [h, w]}


def project_world(points, recipe):
    camera = (np.asarray(points)-np.asarray(recipe['eye'])) @ literal_rotation(
        recipe['yaw'], recipe['pitch'], recipe['roll']).T
    cx, cy = recipe['principal']
    _require((camera[..., 2] > 0).all(), 'projection_behind_camera')
    xy = np.stack((cx+recipe['focal']*camera[..., 0]/camera[..., 2],
                   cy-recipe['focal']*camera[..., 1]/camera[..., 2]), axis=-1)
    if recipe['crop'] is not None:
        xy -= np.asarray(recipe['crop'][:2])
    return xy


@dataclass(frozen=True, slots=True)
class TracePolicy:
    max_score_events: int = 120_000
    max_stage_events: int = 32
    max_array_bytes: int = 2_000_000
    max_trace_bytes: int = 64_000_000
    max_segments: int = 10_000


SOURCE_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = SOURCE_ROOT/'docs/reports/CAM-MD-01-2026-10-05.md'
EVIDENCE_PATH = SOURCE_ROOT/'docs/reports/CAM-MD-01-evidence-2026-10-05.jsonl'
MAX_EVIDENCE_BYTES = 600_000_000
CALIBRATION_PATH = SOURCE_ROOT/'src/mc_photogrammetry/reconstruction/calibration.py'
_PINNED_CAMERA_CODE = calibration.estimate_camera.__code__
_PINNED_REFINE_CODE = calibration.refine_optics.__code__
_SCORE_SIGNATURE = inspect.signature(calibration._score)
_TRACE_LOCK = threading.Lock()


def dependency_pins():
    paths = ('src/mc_photogrammetry/reconstruction/calibration.py',
             'src/mc_photogrammetry/reconstruction/camera.py',
             'src/mc_photogrammetry/pipeline/contracts.py',
             'src/mc_photogrammetry/domain/models.py',
             'src/mc_photogrammetry/vision/gui_mask.py',
             'scripts/evaluation/run_unit_face_input.py',
             'src/mc_photogrammetry/blocks/unit_face_catalogue.py')
    return {'source_sha256': {path: _sha((SOURCE_ROOT/path).read_bytes()) for path in paths},
            'runtime': {'python': sys.version, 'numpy': np.__version__,
                        'opencv': cv2.__version__, 'pillow': PIL.__version__}}


def score_call_sites(source):
    """AST spans, not guessed loop contents or a copy of optimizer control flow."""
    tree = ast.parse(source)
    result = []
    expected = {'estimate_camera': ('coarse', 'coarse-full-rescore', 'fine'),
                'refine_optics': ('refine-base', 'refine-roll', 'refine-principal', 'refine-orientation')}
    for function, tags in expected.items():
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
        _require(len(matches) == 1, 'trace_ast_function_drift')
        calls = sorted((node for node in ast.walk(matches[0]) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == '_score'), key=lambda n: n.lineno)
        _require(len(calls) == len(tags), 'trace_ast_score_site_drift')
        for call, tag in zip(calls, tags):
            wanted = 'coarse' if tag == 'coarse' else 'segments'
            _require(len(call.args) >= 8 and isinstance(call.args[0], ast.Name)
                     and call.args[0].id == wanted, 'trace_ast_argument_drift')
            result.append({'function': function, 'start': call.lineno, 'end': call.end_lineno,
                           'tag': tag, 'ast_sha256': _sha(ast.dump(call, include_attributes=False))})
    _require(all(a['end'] < b['start'] or b['end'] < a['start']
                 for i, a in enumerate(result) for b in result[i+1:]), 'trace_ambiguous_source_spans')
    return tuple(result)


PINNED_SITES = score_call_sites(CALIBRATION_PATH.read_text(encoding='utf-8'))


def classify_score_site(code, line, sites=PINNED_SITES):
    function = ('estimate_camera' if code is _PINNED_CAMERA_CODE else
                'refine_optics' if code is _PINNED_REFINE_CODE else None)
    _require(function is not None, 'trace_unexpected_score_caller')
    found = [site for site in sites if site['function'] == function and site['start'] <= line <= site['end']]
    _require(len(found) == 1, 'trace_unknown_or_ambiguous_score_site')
    return found[0]


def _clean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


def winner_snapshot(frame, site):
    """Only whitelisted optimizer winner locals; candidate params come from args."""
    if site['function'] == 'estimate_camera':
        best = frame.f_locals.get('best')
        _require(best is None or isinstance(best, tuple) and len(best) == 4, 'trace_winner_shape_drift')
        return {'best_before': _clean(best)}
    values = frame.f_locals
    return {'winner_before': {key: _clean(values[key]) for key in
                             ('value', 'focal', 'yaw', 'pitch', 'roll', 'point') if key in values}}


class TraceRecorder:
    def __init__(self, policy):
        _require(type(policy) is TracePolicy and all(type(v) is int and v >= 0
                 for v in asdict(policy).values()), 'invalid_trace_policy')
        _require(all(asdict(policy)[key] <= limit for key, limit in asdict(TracePolicy()).items()),
                 'trace_policy_may_only_lower_caps')
        self.policy = policy
        self.arrays, self.array_objects = {}, {}
        self.scores, self.stages = [], []
        self.array_bytes = self.trace_bytes = 0
        self.trace_seconds = self.score_delegate_seconds = 0.

    def add(self, collection, event, cap):
        _require(len(collection) < cap, 'trace_event_cap')
        size = len(_json(event).encode())
        _require(self.trace_bytes+size <= self.policy.max_trace_bytes, 'trace_payload_cap')
        collection.append(event)
        self.trace_bytes += size

    def array(self, value, *, segments=False):
        _require(isinstance(value, np.ndarray), 'trace_array_contract_drift')
        _require(np.isfinite(value).all(), 'trace_nonfinite_array')
        if segments:
            _require(value.ndim == 2 and value.shape[1] == 4, 'trace_segment_shape_drift')
            _require(len(value) <= self.policy.max_segments, 'trace_segment_cap')
        raw = value.tobytes(order='C')
        metadata = {'dtype': value.dtype.str, 'shape': list(value.shape), 'bytes_hex': raw.hex()}
        digest = _sha(_json(metadata))
        old = self.array_objects.get(id(value))
        _require(old is None or old[0] == digest, 'trace_array_mutated')
        # Retain the object during tracing so allocator ID reuse cannot appear
        # as mutation. The exported snapshot consists only of independent bytes.
        self.array_objects[id(value)] = (digest, value)
        if digest not in self.arrays:
            _require(self.array_bytes+len(raw) <= self.policy.max_array_bytes, 'trace_array_cap')
            self.array_bytes += len(raw)
            self.arrays[digest] = metadata
        return digest

    def payload(self):
        return {'scores': self.scores, 'stages': self.stages, 'arrays': self.arrays,
                'trace_complete': True, 'score_events': len(self.scores),
                'array_bytes': self.array_bytes, 'trace_event_bytes': self.trace_bytes,
                'trace_seconds': self.trace_seconds, 'score_delegate_seconds': self.score_delegate_seconds}


def _install(module, name, wrapper):
    setattr(module, name, wrapper)


def _call_once(function, args, kwargs):
    """The delegation seam is identity preserving and deliberately trivial."""
    return function(*args, **kwargs)


def trace_rgb(image, frame_id, *, policy=TracePolicy()):
    """RGB-only accepted entry, exact call-through instrumentation, single thread."""
    _verify_preregistration()
    _require(isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3
             and image.dtype == np.uint8, 'trace_rgb_contract')
    recorder = TraceRecorder(policy)
    _require(_TRACE_LOCK.acquire(blocking=False), 'trace_reentrancy_or_concurrency')
    names = ((calibration, '_score'), (calibration, 'enhance'), (calibration, 'detect_segments'),
             (calibration, 'estimate_camera'), (calibration, 'refine_optics'), (gui_mask, 'create_gui_mask'))
    originals = [(module, name, getattr(module, name)) for module, name in names]
    source_bytes = image.tobytes()
    source_shape, source_dtype = image.shape, image.dtype
    result = None
    started = time.perf_counter()
    owner_thread = threading.get_ident()
    thread_violation = [False]
    stage_started = set()

    def check_thread():
        if threading.get_ident() != owner_thread:
            thread_violation[0] = True
        _require(not thread_violation[0], 'trace_cross_thread_call')

    def scoring(*args, **kwargs):
        before = time.perf_counter()
        check_thread()
        _require(len(recorder.scores) < policy.max_score_events, 'trace_event_cap')
        frame = inspect.currentframe().f_back
        try:
            site = classify_score_site(frame.f_code, frame.f_lineno)
            winner = winner_snapshot(frame, site)
        finally:
            del frame
        try:
            bound = _SCORE_SIGNATURE.bind(*args, **kwargs)
        except TypeError as error:
            raise CameraStudyError('trace_score_argument_drift') from error
        bound.apply_defaults()
        value = bound.arguments
        segments, weights = value['segments'], value['lengths']
        segment_sha = recorder.array(segments, segments=True)
        weight_sha = recorder.array(weights)
        _require(weights.shape == (len(segments),), 'trace_weight_shape_drift')
        event = {'site': site['tag'], 'span': [site['start'], site['end']],
                 'segments_sha256': segment_sha, 'weights_sha256': weight_sha,
                 'camera': {key: _clean(value[key]) for key in
                            ('focal', 'yaw', 'pitch', 'width', 'height', 'threshold', 'roll', 'principal_point')},
                 'winner': winner}
        # Reserve exact bounded event before delegation; no silent truncation.
        recorder.add(recorder.scores, dict(event, score=0.), policy.max_score_events)
        delegated = time.perf_counter()
        found = _call_once(originals[0][2], args, kwargs)
        delegate_seconds = time.perf_counter()-delegated
        check_thread()
        _require(recorder.array(segments, segments=True) == segment_sha and recorder.array(weights) == weight_sha,
                 'trace_array_mutated')
        try:
            event['score'] = float(found)
        except (TypeError, ValueError) as error:
            raise CameraStudyError('trace_score_result_contract_drift') from error
        _require(math.isfinite(event['score']), 'trace_nonfinite_score')
        extra = len(_json(event).encode())-len(_json(recorder.scores[-1]).encode())
        _require(recorder.trace_bytes+extra <= policy.max_trace_bytes, 'trace_payload_cap')
        recorder.trace_bytes += extra
        recorder.scores[-1] = event
        recorder.score_delegate_seconds += delegate_seconds
        recorder.trace_seconds += time.perf_counter()-before-delegate_seconds
        return found

    def stage_wrapper(name, original):
        def wrapped(*args, **kwargs):
            before = time.perf_counter()
            check_thread()
            _require(name not in stage_started, 'trace_reentrant_or_repeated_stage')
            stage_started.add(name)
            _require(len(recorder.stages) < policy.max_stage_events, 'trace_stage_cap')
            input_sha = recorder.array(args[0]) if args and isinstance(args[0], np.ndarray) else None
            delegated = time.perf_counter()
            found = _call_once(original, args, kwargs)
            delegate_seconds = time.perf_counter()-delegated
            check_thread()
            if input_sha is not None:
                _require(recorder.array(args[0]) == input_sha, 'trace_array_mutated')
            output = recorder.array(found, segments=name == 'detect_segments') if isinstance(found, np.ndarray) else (
                _clean(asdict(found)) if name == 'estimate_camera' and found is not None else _clean(found))
            recorder.add(recorder.stages, {'stage': name, 'input_sha256': input_sha,
                         'output': output, 'delegate_seconds': delegate_seconds}, policy.max_stage_events)
            recorder.trace_seconds += time.perf_counter()-before-delegate_seconds
            return found
        return wrapped

    try:
        _require(calibration.estimate_camera.__code__ is _PINNED_CAMERA_CODE
                 and calibration.refine_optics.__code__ is _PINNED_REFINE_CODE,
                 'trace_caller_code_drift')
        for module, name, original in originals:
            _install(module, name, scoring if name == '_score' else stage_wrapper(name, original))
        result = calibration.estimate_image_calibrations({frame_id: image})[frame_id]
        _require(image.shape == source_shape and image.dtype == source_dtype and image.tobytes() == source_bytes,
                 'trace_source_mutated')
        return {'status': 'returned' if result is not None else 'ordinary_refusal',
                'reason': '' if result is not None else 'image_calibration_failed',
                'prediction': None if result is None else _clean(asdict(result)),
                'trace': recorder.payload(), 'seconds': time.perf_counter()-started,
                'rgb_sha256': _sha(source_bytes), 'rgb_shape': list(source_shape)}
    except CameraStudyError as error:
        partial = recorder.payload()
        partial['trace_complete'] = False
        return {'status': 'trace_refusal', 'reason': error.reason, 'prediction': None,
                'trace': partial,
                'seconds': time.perf_counter()-started, 'rgb_sha256': _sha(source_bytes),
                'rgb_shape': list(source_shape)}
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
        _TRACE_LOCK.release()


@dataclass(frozen=True, slots=True)
class SealedPrediction:
    payload_json: str
    sha256: str


def decode_array(trace, digest):
    metadata = trace['arrays'][digest]
    _require(_sha(_json(metadata)) == digest, 'trace_snapshot_digest_mismatch')
    raw = bytes.fromhex(metadata['bytes_hex'])
    dtype = np.dtype(metadata['dtype'])
    _require(dtype.kind in 'biuf' and all(type(n) is int and n >= 0 for n in metadata['shape']),
             'trace_snapshot_dtype_shape_mismatch')
    _require(len(raw) == math.prod(metadata['shape'])*dtype.itemsize, 'trace_snapshot_byte_mismatch')
    result = np.frombuffer(raw, dtype=dtype).reshape(metadata['shape'])
    _require(np.isfinite(result).all(), 'trace_snapshot_nonfinite')
    return result


def replay_trace(result):
    trace = result['trace']
    _require(trace['trace_complete'] == (result['status'] != 'trace_refusal'), 'trace_status_mismatch')
    _require(trace['score_events'] == len(trace['scores']) <= TracePolicy().max_score_events,
             'trace_score_count_mismatch')
    _require(len(trace['stages']) <= TracePolicy().max_stage_events, 'trace_stage_count_mismatch')
    total = 0
    decoded = {}
    for digest in trace['arrays']:
        decoded[digest] = decode_array(trace, digest)
        total += decoded[digest].nbytes
    _require(total == trace['array_bytes'] <= TracePolicy().max_array_bytes, 'trace_array_count_mismatch')
    for event in trace['scores']:
        segments = decoded[event['segments_sha256']]
        weights = decoded[event['weights_sha256']]
        _require(segments.ndim == 2 and segments.shape[1] == 4 and weights.shape == (len(segments),),
                 'trace_replay_weight_segment_mismatch')
        _require(any(site['tag'] == event['site'] and [site['start'], site['end']] == event['span']
                     for site in PINNED_SITES), 'trace_replay_unknown_site')
    computed_bytes = sum(len(_json(event).encode()) for event in trace['scores']+trace['stages'])
    _require(computed_bytes == trace['trace_event_bytes'] <= TracePolicy().max_trace_bytes,
             'trace_event_byte_mismatch')
    _require(result['status'] != 'trace_refusal' or result['prediction'] is None, 'trace_refusal_with_prediction')
    return trace


def seal_prediction(scene, result):
    replay_trace(result)
    payload = _json({'scene_identity': scene.identity, 'scene_recipe_sha256': scene.sha256,
        'physical_scene': scene.physical, 'preregistration_sha256': PREREGISTRATION_SHA256,
        'prediction_and_trace': result, 'production_gate_changed': False})
    return SealedPrediction(payload, _sha(payload))


def verify_prediction(scene, sealed):
    _require(_sha(sealed.payload_json) == sealed.sha256, 'camera_prediction_seal_mismatch')
    record = json.loads(sealed.payload_json)
    _require(record['scene_identity'] == scene.identity and record['scene_recipe_sha256'] == scene.sha256
             and record['physical_scene'] == scene.physical
             and record['preregistration_sha256'] == PREREGISTRATION_SHA256,
             'camera_prediction_recipe_binding_mismatch')
    replay_trace(record['prediction_and_trace'])
    return record['prediction_and_trace']


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


def full_segment_score(segments, weights, camera):
    h, w = camera['shape']
    points = calibration.vanishing_points(camera['focal'], camera['yaw'], camera['pitch'], w, h,
        camera['roll'], tuple(camera['principal']))
    residuals = calibration.segment_residuals(segments, points)
    closest = residuals.min(axis=1)
    owner = residuals.argmin(axis=1)
    inside = closest <= calibration.INLIER_PIXELS
    score = calibration._score(segments, weights, camera['focal'], camera['yaw'], camera['pitch'], w, h,
                              calibration.INLIER_PIXELS, camera['roll'], tuple(camera['principal']))
    return {'score': score, 'median_residual_all': float(np.median(closest)) if len(closest) else None,
        'median_residual_inliers': float(np.median(closest[inside])) if inside.any() else None,
        'inliers': int(inside.sum()), 'segment_count': len(segments),
        'per_axis_inliers': [int(((owner == axis) & inside).sum()) for axis in range(3)],
        'residuals': closest.tolist(), 'axis_owners': owner.tolist()}


def attribution_masks(scene):
    """Post-seal proximity labels; categories overlap, never causal proof."""
    if scene.physical:
        rgb, _, _, pencil = physical_render(json.loads(scene.recipe_json))
        shape = rgb.shape[:2]
        strip = np.zeros(shape, bool)
    else:
        recipe = json.loads(scene.recipe_json)
        shape = tuple(recipe['shape'])
        strip = np.zeros(shape, np.uint8)
        x0, y0, x1, y1 = recipe['face_rectangle']
        # The legacy renderer uses GLOBAL (xx//48)%2, not rectangle-relative
        # strips. Mark both adjacent pixels only when both belong to its face;
        # the rectangle/image perimeter is not an internal compositor join.
        depth_class = (np.arange(shape[1])//48) % 2
        joins = np.flatnonzero(depth_class[1:] != depth_class[:-1])+1
        top, bottom = max(0, y0), min(shape[0], y1)
        for x in joins:
            if x0 <= x-1 and x < x1 and top < bottom:
                strip[top:bottom, x-1:x+1] = 1
        pencil = np.zeros(shape, np.uint8)
        rotation = literal_rotation(recipe['renderer_yaw'], recipe['renderer_pitch'])
        for direction in rotation.T:
            vp = np.array((recipe['renderer_focal']*direction[0]+shape[1]/2*direction[2],
                           -recipe['renderer_focal']*direction[1]+shape[0]/2*direction[2], direction[2]))
            for row in recipe['line_rows']:
                for col in recipe['line_anchors']:
                    anchor = np.array([col, row], dtype=float)
                    delta = anchor-vp[:2]/vp[2] if abs(vp[2]) > 1e-10 else vp[:2]
                    delta /= np.linalg.norm(delta)
                    cv2.line(pencil, tuple(np.rint(anchor-90*delta).astype(int)),
                             tuple(np.rint(anchor+90*delta).astype(int)), 1, 2)
        pencil[:, recipe['structure_band_edges'][0]:recipe['structure_band_edges'][1]] = 0
    gui = create_gui_mask(*shape).astype(np.uint8)
    border = cv2.morphologyEx(gui, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    kernel = np.ones((7, 7), np.uint8)
    return {key: cv2.dilate(value.astype(np.uint8), kernel).astype(bool) for key, value in
            (('gui_border', border), ('compositor_strip', strip), ('projected_pencil', pencil))}


def segment_attribution(segments, masks):
    fractions = np.linspace(0., 1., 16)
    coordinates = segments[:, None, :2]*(1-fractions[None, :, None])+segments[:, None, 2:]*fractions[None, :, None]
    xy = np.rint(coordinates).astype(int)
    h, w = next(iter(masks.values())).shape
    valid = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    xy[..., 0] = np.clip(xy[..., 0], 0, w-1)
    xy[..., 1] = np.clip(xy[..., 1], 0, h-1)
    labels, combinations = {}, np.zeros(len(segments), int)
    for bit, (name, mask) in enumerate(masks.items()):
        proximity = (mask[xy[..., 1], xy[..., 0]] & valid).sum(axis=1)/16.
        active = proximity >= .5
        labels[name] = {'sample_fraction': proximity.tolist(), 'segment_count': int(active.sum())}
        combinations += active.astype(int)*(1 << bit)
    return {'overlapping': labels, 'joint_bitmask_counts': [int((combinations == i).sum()) for i in range(8)],
            'exclusive_partition_claim': False, 'causal_attribution_claim': False}


def stage_camera_evidence(result):
    """Tested maxima and optimizer-selected transitions are separate evidence."""
    trace = result['trace']
    _require(result['status'] == 'returned' and trace['trace_complete'], 'camera_stage_evidence_incomplete')
    candidate_max = {}
    for tag in sorted({event['site'] for event in trace['scores']}):
        event = max((item for item in trace['scores'] if item['site'] == tag), key=lambda item: item['score'])
        candidate_max[tag] = {'camera': event['camera'], 'native_score': event['score'],
            'native_score_scope': 'coarse_subset' if tag == 'coarse' else 'full_segments',
            'optimizer_retention_claim': False}
    rescored = [event for event in trace['scores'] if event['site'] == 'coarse-full-rescore']
    refine_base = [event for event in trace['scores'] if event['site'] == 'refine-base']
    refine_return = [event for event in trace['stages'] if event['stage'] == 'refine_optics']
    _require(len(rescored) == len(refine_base) == len(refine_return) == 1,
             'camera_selected_transition_trace_mismatch')
    coarse = rescored[0]
    winner = coarse['winner']['best_before']
    _require(isinstance(winner, list) and len(winner) == 4 and winner[1:] ==
             [coarse['camera'][key] for key in ('focal', 'yaw', 'pitch')], 'camera_coarse_transition_mismatch')
    refined = refine_return[0]['output']
    _require(isinstance(refined, list) and len(refined) == 6 and type(refined[4]) is bool,
             'camera_refinement_return_mismatch')
    return {'candidate_max': candidate_max, 'selected_transitions': {
        'coarse_winner_to_full_rescore': {'selected_camera': coarse['camera'],
            'coarse_native_score': winner[0], 'full_rescore': coarse['score'],
            'source': 'captured full-rescore arguments and best-before'},
        'fine_winner_to_refinement': {'selected_fine_camera': refine_base[0]['camera'],
            'refinement_return': refined, 'refinement_accepted': refined[4],
            'source': 'captured refine-base arguments and actual refine return'},
        'final_prediction': result['prediction']}}


def evaluate_prediction(scene, sealed):
    result = verify_prediction(scene, sealed)
    _require(result['rgb_sha256'] == json.loads(PREREGISTRATION_JSON)['stimulus_sha256'][scene.identity],
             'camera_evaluator_source_binding_mismatch')
    if result['status'] != 'returned':
        return {'status': 'refused', 'reason': result['reason'], 'metrics': None,
                'physical_aggregate_eligible': scene.physical}
    trace = result['trace']
    detection = [stage for stage in trace['stages'] if stage['stage'] == 'detect_segments']
    _require(len(detection) == 1, 'camera_evaluator_segment_source_mismatch')
    segments = decode_array(trace, detection[0]['output'])
    # The same complete segment array and unmodified length weights serve all
    # returned/stage/renderer scores. Coarse scores are never compared to full.
    weights = np.hypot(segments[:, 2]-segments[:, 0], segments[:, 3]-segments[:, 1])
    full_events = [event for event in trace['scores'] if event['site'] != 'coarse']
    _require({event['segments_sha256'] for event in full_events} == {detection[0]['output']},
             'camera_evaluator_full_segment_weight_mismatch')
    for digest in {event['weights_sha256'] for event in full_events}:
        _require(np.array_equal(decode_array(trace, digest), weights), 'camera_evaluator_full_segment_weight_mismatch')
    prediction = result['prediction']
    h, w = result['rgb_shape'][:2]
    returned = {'focal': prediction['focal_pixels'], 'yaw': prediction['yaw_degrees'],
                'pitch': prediction['pitch_degrees'], 'roll': prediction['roll_degrees'],
                'principal': prediction['principal_point'] or [w/2., h/2.], 'shape': [h, w]}
    stage_evidence = stage_camera_evidence(result)
    scored = {'returned': full_segment_score(segments, weights, returned), 'candidate_max': {},
              'selected_transitions': {}}
    for tag, maximum in stage_evidence['candidate_max'].items():
        camera = maximum['camera']
        scored['candidate_max'][tag] = full_segment_score(segments, weights,
            {'focal': camera['focal'], 'yaw': camera['yaw'], 'pitch': camera['pitch'],
             'roll': camera['roll'], 'principal': camera['principal_point'] or [w/2., h/2.], 'shape': [h, w]})
    transitions = stage_evidence['selected_transitions']
    for name, camera in (('coarse_winner_to_full_rescore', transitions['coarse_winner_to_full_rescore']['selected_camera']),
                         ('fine_winner_to_refinement', transitions['fine_winner_to_refinement']['selected_fine_camera'])):
        scored['selected_transitions'][name] = full_segment_score(segments, weights,
            {'focal': camera['focal'], 'yaw': camera['yaw'], 'pitch': camera['pitch'],
             'roll': camera['roll'], 'principal': camera['principal_point'] or [w/2., h/2.], 'shape': [h, w]})
    masks = attribution_masks(scene)
    attribution = segment_attribution(segments, masks)
    if not scene.physical:
        recipe = json.loads(scene.recipe_json)
        scored['declared_compositor_renderer'] = full_segment_score(segments, weights,
            {'focal': recipe['renderer_focal'], 'yaw': recipe['renderer_yaw'],
             'pitch': recipe['renderer_pitch'], 'roll': 0., 'principal': [w/2., h/2.], 'shape': [h, w]})
        return {'status': 'adversarial_stress_only', 'reason': 'legacy_compositor_not_physical_camera_benchmark',
                'metrics': None, 'full_segment_scores': scored, 'attribution': attribution,
                'stage_camera_evidence': stage_evidence,
                'physical_aggregate_eligible': False}
    truth = truth_camera(scene)
    scored['renderer'] = full_segment_score(segments, weights, truth)
    metrics = camera_metrics(prediction, truth)
    fraction = prediction['inliers']/prediction['segments'] if prediction['segments'] else 0.
    return {'status': 'evaluated', 'reason': '', 'metrics': metrics, 'full_segment_scores': scored,
        'attribution': attribution, 'physical_aggregate_eligible': True,
        'stage_camera_evidence': stage_evidence,
        'emitted_wrong_camera': not metrics['diagnostic_pass'],
        'calibrated_confidence': None, 'high_support_wrong_proxy': bool(fraction >= .8 and not metrics['diagnostic_pass']),
        'confidence_proxy_threshold': .8}


def evidence_header():
    return {'kind': 'CAM-MD-01-prediction-trace-v1', 'preregistration_sha256': PREREGISTRATION_SHA256}


def replay_evidence(raw, expected_seals):
    _require(len(raw) <= MAX_EVIDENCE_BYTES, 'camera_evidence_byte_cap')
    _require(raw.endswith(b'\n'), 'camera_evidence_truncated')
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    _require(records and records[0] == evidence_header(), 'camera_evidence_header_mismatch')
    _require(len(records)-1 == len(expected_seals), 'camera_evidence_prefix_mismatch')
    for scene, record, digest in zip(SCENES, records[1:], expected_seals):
        _require(record['sha256'] == digest, 'camera_evidence_seal_mismatch')
        verify_prediction(scene, SealedPrediction(record['payload_json'], record['sha256']))
    return _sha(raw)


def append_evidence(scene, sealed, previous_seals):
    _require(EVIDENCE_PATH.is_file(), 'camera_evidence_missing')
    replay_evidence(EVIDENCE_PATH.read_bytes(), previous_seals)
    _require(scene == SCENES[len(previous_seals)], 'camera_evidence_order_mismatch')
    verify_prediction(scene, sealed)
    encoded = (_json(asdict(sealed))+'\n').encode()
    _require(len(EVIDENCE_PATH.read_bytes())+len(encoded) <= MAX_EVIDENCE_BYTES, 'camera_evidence_byte_cap')
    with EVIDENCE_PATH.open('r+b') as artifact:
        artifact.seek(0, os.SEEK_END)
        artifact.write(encoded)
        artifact.flush()
        os.fsync(artifact.fileno())
    return replay_evidence(EVIDENCE_PATH.read_bytes(), tuple(previous_seals)+(sealed.sha256,))


def _claim_report():
    _require(REPORT_PATH.is_file(), 'camera_preregistration_report_missing')
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
            _require(PREREGISTRATION_SHA256 in content and content.count('CAMERA_EXECUTION_STATE: NOT_STARTED') == 1,
                     'camera_study_already_claimed_or_pin_mismatch')
            report.seek(0)
            report.write(content.replace('CAMERA_EXECUTION_STATE: NOT_STARTED', 'CAMERA_EXECUTION_STATE: RUNNING'))
            report.truncate()
            report.flush()
            os.fsync(report.fileno())
        finally:
            unlock()


def run_study():
    _verify_preregistration()
    _require(not EVIDENCE_PATH.exists(), 'camera_evidence_already_exists')
    _claim_report()
    seals = []
    # Any error leaves the durable RUNNING claim and retained prefix; no retry.
    with EVIDENCE_PATH.open('xb') as artifact:
        artifact.write((_json(evidence_header())+'\n').encode())
        artifact.flush()
        os.fsync(artifact.fileno())
    outputs = []
    for scene in SCENES:
        result = trace_rgb(render_rgb(scene), scene.identity)
        sealed = seal_prediction(scene, result)
        append_evidence(scene, sealed, tuple(seals))
        seals.append(sealed.sha256)
        outputs.append(sealed)
    # Strong ordering: ALL fixed outputs and trace bytes are sealed/durable
    # before the first post-seal renderer-camera/label access.
    evaluated = [evaluate_prediction(scene, sealed) for scene, sealed in zip(SCENES, outputs)]
    predictions = {scene.identity: verify_prediction(scene, sealed) for scene, sealed in zip(SCENES, outputs)}
    physical = [value for scene, value in zip(SCENES, evaluated) if scene.physical]
    counts = {'physical_scenes': len(physical), 'diagnostic_pass': sum(value.get('metrics') is not None
              and value['metrics']['diagnostic_pass'] for value in physical),
              'emitted_wrong': sum(value.get('emitted_wrong_camera', False) for value in physical),
              'refused': sum(value['status'] == 'refused' for value in physical),
              'high_support_wrong_proxy': sum(value.get('high_support_wrong_proxy', False) for value in physical),
              'confidently_wrong': None, 'calibrated_confidence_available': False}
    stability = {'gui_vs_occluding_prediction_identical':
        predictions['physical-gui-contamination']['prediction'] == predictions['physical-occluding']['prediction'],
        'interpretation': 'fixed synthetic perturbation only, not independent generalization'}
    result = {'preregistration_sha256': PREREGISTRATION_SHA256, 'sealed_predictions': seals,
            'evidence_sha256': replay_evidence(EVIDENCE_PATH.read_bytes(), tuple(seals)),
            'scenes': [{'identity': scene.identity, 'evaluation': value,
                       'cost': {key: predictions[scene.identity][key] for key in ('seconds',)},
                       'trace_cost': {key: predictions[scene.identity]['trace'][key] for key in
                                      ('trace_seconds', 'score_delegate_seconds', 'score_events',
                                       'array_bytes', 'trace_event_bytes')}} for scene, value in zip(SCENES, evaluated)],
            'physical_counts': counts, 'stability': stability,
            'production_gate_changed': False, 'generalization_claim': False}
    with REPORT_PATH.open('r+', encoding='utf-8', newline='') as report:
        content = report.read()
        _require(content.count('CAMERA_EXECUTION_STATE: RUNNING') == 1, 'camera_run_marker_drift')
        report.seek(0)
        report.write(content.replace('CAMERA_EXECUTION_STATE: RUNNING', 'CAMERA_EXECUTION_STATE: COMPLETE')+
                     '\n\nDurable registered result:\n\n```json\n'+_json(result)+'\n```\n')
        report.truncate()
        report.flush()
        os.fsync(report.fileno())
    return result


def _verify_preregistration():
    pinned = json.loads(PREREGISTRATION_JSON)
    _require(dependency_pins() == pinned['dependencies'], 'camera_dependency_or_runtime_drift')
    _require(score_call_sites(CALIBRATION_PATH.read_text(encoding='utf-8')) == PINNED_SITES,
             'camera_ast_source_drift')
    _require([asdict(scene) for scene in SCENES] == pinned['scenes'], 'camera_scene_registry_drift')


def _freeze_preregistration():
    functions = (CameraStudyError, _require, _json, _sha, _base_recipe, _scene_recipes,
        literal_rotation, face_planes, intersect_rays, native_surface_rgb, physical_render, render_rgb,
        truth_camera, project_world, dependency_pins, score_call_sites, classify_score_site, _clean,
        winner_snapshot, TraceRecorder, _install, _call_once, trace_rgb, decode_array, replay_trace,
        seal_prediction, verify_prediction, camera_metrics, full_segment_score, attribution_masks,
        segment_attribution, stage_camera_evidence, evaluate_prediction, evidence_header, replay_evidence, append_evidence,
        _claim_report, run_study, _verify_preregistration)
    return _json({'version': 'CAM-MD-01-prereg-v1', 'scenes': [asdict(scene) for scene in SCENES],
        'dependencies': dependency_pins(), 'score_sites': PINNED_SITES, 'trace_policy': asdict(TracePolicy()),
        'stimulus_sha256': {scene.identity: _sha(render_rgb(scene).tobytes()) for scene in SCENES},
        'algorithm_sha256': _sha('\n'.join(inspect.getsource(function) for function in functions)),
        'diagnostic_tolerances': {'focal_relative': .05, 'orientation_deg_mod90': 1., 'principal_diagonal': .01},
        'production_m3_gates_changed': False, 'projection_probe_fractions': [.1, .3, .5, .7, .9],
        'attribution': {'samples_per_segment': 16, 'proximity_pixels': 3, 'fraction_threshold': .5,
                        'overlap_preserved': True, 'causal_claim': False},
        'confidence': {'calibrated_confidence': None, 'high_support_wrong_proxy_inlier_fraction': .8},
        'stage_camera_semantics': 'candidate_max-is-highest-tested; selected-transitions-from-rescore/refine/final-captures',
        'legacy_excluded_from_physical_aggregate': True,
        'one_shot': 'locked-fsynced-report-claim-before-exclusive-evidence-creation-no-retry',
        'evidence_path': 'docs/reports/'+EVIDENCE_PATH.name,
        'max_evidence_bytes': MAX_EVIDENCE_BYTES,
        'truth_order': 'all-fixed-predictions-and-traces-sealed-and-durable-before-any-evaluation',
        'stability': 'gui-contamination-vs-occluding-prediction-equality-no-new-arm',
        'cost': 'wall-seconds-plus-local-trace-overhead-and-score-delegate-time'})


PREREGISTRATION_JSON = _freeze_preregistration()
PREREGISTRATION_SHA256 = _sha(PREREGISTRATION_JSON)


if __name__ == '__main__':
    print('PREREGISTRATION '+PREREGISTRATION_SHA256, flush=True)
    print(_json(run_study()), flush=True)
