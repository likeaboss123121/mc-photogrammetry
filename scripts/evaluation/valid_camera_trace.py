"""Static RGB-only tracing. Import binds metadata, never renders or infers.

An unbound trace refuses. The runner supplies only an externally anchored,
sanitary trace contract; this module has no truth or scoring reference.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import ast
import hashlib
import inspect
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import threading
import time
from types import CodeType
import numpy as np
from mc_photogrammetry.reconstruction import calibration
from mc_photogrammetry.vision import gui_mask


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
class TracePolicy:
    max_score_events: int = 120_000
    max_stage_events: int = 32
    max_array_bytes: int = 2_000_000
    max_trace_bytes: int = 64_000_000
    max_segments: int = 10_000

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

SOURCE_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = SOURCE_ROOT/'src/mc_photogrammetry/reconstruction/calibration.py'
_PINNED_CAMERA_CODE = calibration.estimate_camera.__code__
_PINNED_REFINE_CODE = calibration.refine_optics.__code__
_SCORE_SIGNATURE = inspect.signature(calibration._score)
_TRACE_LOCK = threading.Lock()
PINNED_SITES = score_call_sites(CALIBRATION_PATH.read_text(encoding='utf-8'))
_CONTRACT_JSON = None
_CONTRACT_SHA256 = None


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


def _file_sha256(path):
    """Bound hashing scratch independently of native binary/source size."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_text(path):
    with path.open('rb') as stream:
        raw = stream.read(128*1024+1)
    _require(len(raw) <= 128*1024, 'trace_source_metadata_cap')
    return raw.decode('utf-8')


def _code_signature(code):
    """Independent compiled-source identity, including nested code/constants."""
    return (code.co_code, tuple(_code_signature(c) if isinstance(c, CodeType) else c
        for c in code.co_consts), code.co_names, code.co_varnames, code.co_freevars,
        code.co_cellvars, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_flags)


def attest_module(module, relative, names):
    """Independently compile approved source without executing or trusting helpers."""
    path = SOURCE_ROOT/relative
    canonical_name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
    _require(module.__name__ == canonical_name and sys.modules.get(canonical_name) is module
        and module.__spec__ is not None and module.__spec__.name == canonical_name
        and Path(module.__spec__.origin).resolve() == path
        and Path(module.__file__).resolve() == path, "dependency_path")
    source = _source_text(path)
    tree = ast.parse(source)
    compiled = compile(source, str(path), "exec", dont_inherit=True)
    bindings = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)}
    def default_value(node):
        if isinstance(node, ast.Name) and node.id in bindings:
            return ast.literal_eval(bindings[node.id])
        return ast.literal_eval(node)
    def function_node(name):
        body = tree.body
        node = None
        for part in name.split('.'):
            matches = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
            _require(len(matches) == 1, "dependency_source_definition")
            node = matches[0]; body = node.body
        return node
    def lookup(code, name):
        for part in name.split('.'):
            matches = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == part]
            if len(matches) != 1:
                return None
            code = matches[0]
        return code
    for name in names:
        value = module
        for part in name.split('.'):
            value = getattr(value, part, None)
        expected = lookup(compiled, name)
        _require(inspect.isfunction(value) and value.__globals__ is module.__dict__
            and Path(value.__code__.co_filename).resolve() == path and expected is not None
            and _code_signature(value.__code__) == _code_signature(expected), "dependency_function_identity")
        # Two exact trace defaults are verified explicitly below: the immutable
        # TracePolicy instance and the source-derived PINNED_SITES tuple.
        if name not in ('trace_rgb', 'classify_score_site'):
            node = function_node(name)
            positional = tuple(default_value(n) for n in node.args.defaults) or None
            keywords = {arg.arg: default_value(n) for arg, n in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if n is not None} or None
            _require(value.__defaults__ == positional and value.__kwdefaults__ == keywords, "dependency_default_identity")


def _runtime_pins():
    distribution = metadata.distribution('opencv-python')
    binaries = [distribution.locate_file(p) for p in distribution.files
        if str(p).startswith('cv2/') and str(p).endswith(('.pyd', '.so'))]
    _require(len(binaries) == 1, 'trace_opencv_binary_identity')
    return {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
        'executable': str(Path(sys.executable).resolve()),
        'opencv_binary_sha256': _file_sha256(binaries[0]),
        'distributions': {name: metadata.version(name) for name in ('numpy', 'opencv-python', 'Pillow')}}


def bind_trace_contract(contract, *, expected_contract_sha256):
    """Called only after runner verifies the caller's external approval anchor."""
    global _CONTRACT_JSON, _CONTRACT_SHA256
    _require(not _TRACE_LOCK.locked(), 'trace_contract_busy')
    raw = _json(contract)
    _require(_sha(raw) == expected_contract_sha256, 'trace_contract_anchor')
    _require(isinstance(contract, dict) and set(contract) == {'schema',
        'external_preregistration_sha256', 'source_sha256', 'runtime', 'score_sites', 'policy'},
        'trace_contract_schema')
    digest = contract['external_preregistration_sha256']
    _require(contract['schema'] == 'CAM-VALID-trace-contract-v1' and isinstance(digest, str)
        and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'trace_contract_schema')
    _CONTRACT_JSON, _CONTRACT_SHA256 = raw, expected_contract_sha256
    try:
        _verify_preregistration()
    except Exception:
        _CONTRACT_JSON = _CONTRACT_SHA256 = None
        raise


def clear_trace_contract():
    global _CONTRACT_JSON, _CONTRACT_SHA256
    _require(not _TRACE_LOCK.locked(), 'trace_contract_busy')
    _CONTRACT_JSON = _CONTRACT_SHA256 = None


def _verify_preregistration():
    _require(_CONTRACT_JSON is not None and _sha(_CONTRACT_JSON) == _CONTRACT_SHA256,
        'trace_contract_unbound_or_mutated')
    contract = json.loads(_CONTRACT_JSON)
    _require(contract['policy'] == asdict(TracePolicy()), 'trace_policy_pin')
    _require(contract['runtime'] == _runtime_pins(), 'trace_runtime_drift')
    sources = contract['source_sha256']
    _require(isinstance(sources, dict) and {'scripts/evaluation/valid_camera_trace.py',
        'src/mc_photogrammetry/reconstruction/calibration.py',
        'src/mc_photogrammetry/vision/gui_mask.py'} <= sources.keys(), 'trace_source_contract')
    for relative, digest in sources.items():
        path = SOURCE_ROOT/relative
        _require(not Path(relative).is_absolute() and '..' not in Path(relative).parts
            and path.resolve() == path and not path.is_symlink()
            and isinstance(digest, str) and _file_sha256(path) == digest, 'trace_source_drift')
    _require(list(PINNED_SITES) == contract['score_sites']
        and score_call_sites(CALIBRATION_PATH.read_text(encoding='utf-8')) == PINNED_SITES
        and classify_score_site.__defaults__ == (PINNED_SITES,)
        and trace_rgb.__kwdefaults__ == {'policy': TracePolicy()}, 'trace_score_sites_drift')
    _require(calibration.estimate_camera.__code__ is _PINNED_CAMERA_CODE
        and calibration.refine_optics.__code__ is _PINNED_REFINE_CODE
        and inspect.signature(calibration._score) == _SCORE_SIGNATURE, 'trace_solver_identity')
    attest_module(calibration, 'src/mc_photogrammetry/reconstruction/calibration.py',
        ('estimate_camera', 'refine_optics', '_score', 'estimate_image_calibrations'))
    attest_module(sys.modules[__name__], 'scripts/evaluation/valid_camera_trace.py',
        ('trace_rgb', '_verify_preregistration', 'bind_trace_contract', '_runtime_pins',
         'attest_module', '_code_signature', '_file_sha256', '_source_text', '_require', '_json', '_sha', 'classify_score_site',
         'winner_snapshot', '_clean', '_install', '_call_once', 'score_call_sites'))
