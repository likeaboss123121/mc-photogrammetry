"""Private paired density ablation; frozen upstream context and validation.

Only this harness instruments preparation. It restores the accepted callables
before fitting and never changes their files/defaults/thresholds. Recipe v1 is
development; the preregistered disjoint v2 heldout pair is executed once by the
study entry. Unit tests do not execute its arms. No real inputs or placements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import PIL

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from scripts.evaluation import run_unit_face_input as input_v1
from mc_photogrammetry.pipeline import image_derived_unit_face as adapter
from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction.unit_face_estimator import SearchPolicy
from mc_photogrammetry.reconstruction.unit_face_lattice import (
    ContractLimits, DependencyGroup, PartitionedSupport, SourceTapPartition,
    WitnessProof, partition_tap_support,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


class DensityIntegrityError(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _require(condition, reason):
    if not condition:
        raise DensityIntegrityError(reason)


def _replay_support(support):
    try:
        partition = SourceTapPartition(**asdict(support.partition))
        proofs = tuple(WitnessProof(**asdict(w)) for w in support.witnesses)
        rebuilt = PartitionedSupport('complete', '', partition, support.samples, support.groups, proofs)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as error:
        raise DensityIntegrityError('density_source_replay_failed') from error
    _require(rebuilt == support, 'density_source_replay_mismatch')
    return support


@dataclass(frozen=True, slots=True)
class SceneRecipe:
    identity: str
    purpose: str
    case: str
    recipe_json: str

    @property
    def sha256(self):
        return _sha(_json(asdict(self)))


HELDOUT_JSON = _json({
    'version': 'UNIT-FACE-DENSITY-01-heldout-v2', 'shape': [600, 800],
    'renderer_focal': 730., 'renderer_yaw': 31., 'renderer_pitch': -17.,
    'distance': 3.61, 'phase': [.337, .619], 'native_side': 16,
    'face_rectangle': [270, 90, 650, 480], 'structure_band_edges': [260, 660],
    'line_anchors': [25, 85, 160, 220, 695, 775],
    'line_rows': [100, 155, 210, 265, 320, 375, 430, 485],
    'fourier': [[54., .91, 31., -.4, 14., .28],
                [47., -.8, 37., .51, 18., .61], [44., -.35, 40., .2, 19., .49]],
})
SCENES = tuple(SceneRecipe('development-v1-' + case, 'development', case, input_v1.RECIPE_JSON)
               for case in ('identifiable', 'ambiguous', 'mixed_depth')) + (
    SceneRecipe('heldout-v2-identifiable', 'heldout', 'identifiable', HELDOUT_JSON),)


def _literal_axes(yaw_degrees, pitch_degrees):
    yaw, pitch = map(math.radians, (yaw_degrees, pitch_degrees))
    return np.array(((-math.cos(yaw), -math.sin(pitch)*math.sin(yaw), -math.cos(pitch)*math.sin(yaw)),
        (0., math.cos(pitch), -math.sin(pitch)),
        (-math.sin(yaw), math.sin(pitch)*math.cos(yaw), math.cos(pitch)*math.cos(yaw))))


def render_scene(scene):
    """Stimulus generation only; optics/labels never enter the RGB adapter."""
    if scene.purpose == 'development':
        return input_v1.render_rgb(scene.case)
    recipe = json.loads(scene.recipe_json)
    side = recipe['native_side']
    h, w = recipe['shape']
    axes = _literal_axes(recipe['renderer_yaw'], recipe['renderer_pitch'])
    yy, xx = np.indices((h, w))
    rays = np.stack(((xx-w/2)/recipe['renderer_focal'],
                    -(yy-h/2)/recipe['renderer_focal'], np.ones((h, w))), axis=-1)
    unit = rays / np.abs(rays @ axes[2])[..., None]
    bx = recipe['distance']*(unit @ axes[0]) + recipe['phase'][0]
    by = recipe['distance']*(unit @ axes[1]) + recipe['phase'][1]
    y, x = np.meshgrid((np.arange(side)+.5)*2*np.pi/side,
                       (np.arange(side)+.5)*2*np.pi/side, indexing='ij')
    a, b, c = recipe['fourier']
    texture = np.rint(np.stack((128+a[0]*np.sin(x+a[1])+a[2]*np.cos(y+a[3])+a[4]*np.sin(2*x+y+a[5]),
        128+b[0]*np.cos(x+b[1])+b[2]*np.sin(y+b[3])+b[4]*np.sin(x-2*y+b[5]),
        128+c[0]*np.sin(x+y+c[1])+c[2]*np.cos(2*x+c[3])+c[4]*np.sin(2*y+c[5])), axis=-1)).astype(np.uint8)
    tx, ty = ((-bx+.5) % 1)*side-.5, ((-by+.5) % 1)*side-.5
    ix, iy = np.floor(tx).astype(int), np.floor(ty).astype(int)
    dx, dy = tx-ix, ty-iy
    color = np.zeros((h, w, 3))
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        color += weight[..., None]*texture[(iy+oy) % side, (ix+ox) % side]
    image = np.full((h, w, 3), 112, np.uint8)
    x0, y0, x1, y1 = recipe['face_rectangle']
    image[y0:y1, x0:x1] = np.rint(color[y0:y1, x0:x1]).astype(np.uint8)
    ink = np.zeros_like(image)
    for axis, direction in enumerate(axes):
        vp = np.array((recipe['renderer_focal']*direction[0]+w/2*direction[2],
                       -recipe['renderer_focal']*direction[1]+h/2*direction[2], direction[2]))
        for row in recipe['line_rows']:
            for col in recipe['line_anchors']:
                anchor = np.array((col, row), dtype=float)
                delta = anchor-vp[:2]/vp[2] if abs(vp[2]) > 1e-10 else vp[:2]
                delta /= np.linalg.norm(delta)
                p0, p1 = anchor-90*delta, anchor+90*delta
                value = (220, 30, 200)[axis]
                cv2.line(ink, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)),
                         (value, value, value), 2, cv2.LINE_AA)
    left, right = recipe['structure_band_edges']
    marks = ink.any(axis=-1) & ((xx < left) | (xx >= right))
    image[marks] = ink[marks]
    return image, input_v1.generated_catalogue(texture)


@dataclass(frozen=True, slots=True)
class CapturedPreparation:
    prepared: object | None
    ordinary_refusal: object | None
    args: tuple
    fixed_policy: SearchPolicy | None
    restored_fitter: object
    restored_prepare: object
    seconds: float


class _PreparationOnly(BaseException):
    pass


def capture_rgb_preparation(image, catalogue, frame_id):
    """Astra-approved SINGLE-THREADED evaluation instrumentation, always restored."""
    _verify_dependencies()
    original_prepare = adapter._fit_prepared
    original_fitter = adapter.estimate_unit_face
    sentinel = _PreparationOnly()
    prepared_calls, fitter_calls = [], []
    started = time.perf_counter()

    def preparing(prepared, actual_catalogue):
        _require(not prepared_calls, 'more_than_one_preparation_interception')
        prepared_calls.append(prepared)
        return original_prepare(prepared, actual_catalogue)

    def capturing(*args, **kwargs):
        _require(not fitter_calls and len(prepared_calls) == 1, 'invalid_fitter_interception_count')
        _require(len(args) == 4 and set(kwargs) == {'fixed_policy'}, 'changed_adapter_fitter_contract')
        p = prepared_calls[0]
        _require(args[1] is p.context and args[2] is p.support and args[3] is catalogue,
                 'changed_adapter_context_support_catalogue')
        _require(args[0].tobytes() == p.crop_rgb and not args[0].flags.writeable, 'changed_adapter_source')
        _require(type(kwargs['fixed_policy']) is SearchPolicy and kwargs['fixed_policy'] == SearchPolicy(),
                 'nondefault_density_fit_policy')
        fitter_calls.append((args, kwargs['fixed_policy']))
        raise sentinel

    ordinary = None
    try:
        adapter._fit_prepared = preparing
        adapter.estimate_unit_face = capturing
        try:
            ordinary = adapter.estimate_image_unit_face(image, catalogue, frame_id=frame_id)
        except _PreparationOnly as caught:
            if caught is not sentinel:
                raise
    finally:
        adapter._fit_prepared = original_prepare
        adapter.estimate_unit_face = original_fitter
    if ordinary is not None:
        _require(not fitter_calls and not prepared_calls and ordinary.prepared is None,
                 'unexpected_preparation_refusal_after_interception')
        return CapturedPreparation(None, ordinary, (), None, original_fitter, original_prepare,
                                   time.perf_counter()-started)
    _require(len(prepared_calls) == len(fitter_calls) == 1, 'invalid_preparation_capture_count')
    args, policy = fitter_calls[0]
    return CapturedPreparation(prepared_calls[0], None, args, policy, original_fitter, original_prepare,
                               time.perf_counter()-started)


@dataclass(frozen=True, slots=True)
class DenseSupport:
    status: str
    reason: str
    support: object | None
    window: tuple | None
    eligible_generation_stencils: int = 0
    added_generation_pixels: int = 0
    planned_generation_pixels: int = 0
    rejected_depth: int = 0
    rejected_border: int = 0
    rejected_excluded: int = 0
    rejected_cross_role: int = 0
    omitted_validation: int = 0
    baseline_witness_samples: int = 0
    baseline_worst_case_taps: int = 0
    remaining_witness_samples: int = 0
    remaining_witness_taps: int = 0
    remaining_witness_bytes: int = 0


def extend_generation_support(prepared, *, limits=ContractLimits()):
    """Add ALL complete generation40 stencils; never top-K or subsample."""
    baseline = _replay_support(prepared.support)
    policy = SearchPolicy()
    counts = {}
    window = None
    def refuse(reason):
        return DenseSupport('refused', reason, None, window, **counts)
    source_shape = prepared.context.source_shape
    baseline_samples = sum(w.map_shape[0]*w.map_shape[1] for w in baseline.witnesses)
    total_samples = baseline_samples+1600
    previous_bytes = sum(w.payload_bytes for w in baseline.witnesses)
    counts = dict(baseline_witness_samples=baseline_samples, baseline_worst_case_taps=baseline_samples*4,
        remaining_witness_samples=max(0, limits.max_rectified_samples-baseline_samples),
        remaining_witness_taps=max(0, limits.max_bilinear_taps-baseline_samples*4),
        remaining_witness_bytes=max(0, limits.max_witness_payload_bytes-previous_bytes))
    if source_shape[0]*source_shape[1] > limits.max_source_pixels:
        return refuse('density_source_pixel_cap')
    if total_samples > limits.max_rectified_samples:
        return refuse('density_witness_sample_cap')
    if total_samples*4 > limits.max_bilinear_taps:
        return refuse('density_witness_tap_cap')
    if previous_bytes >= limits.max_witness_payload_bytes:
        return refuse('density_witness_payload_cap')
    partition = baseline.partition
    region = np.frombuffer(partition.region, np.uint8).reshape(source_shape).astype(bool)
    usable = np.frombuffer(partition.usable, np.uint8).reshape(source_shape).astype(bool)
    ctx = prepared.context
    window = lattice.plane_window(region & usable, ctx.axis, np.asarray(ctx.axis_matrix),
                                 ctx.focal_pixels, principal_point=ctx.principal_point)
    if window is None or window[3] != prepared.proposal.side:
        return refuse('density_plane_window_unavailable')
    # Recreate every complete original witness, retaining its original map,
    # depth and quantized taps. Source RGB is never rectified as an observation.
    witnesses = [lattice.rectification_witness_v1(
        np.frombuffer(w.map_x, np.float32).reshape(w.map_shape),
        np.frombuffer(w.map_y, np.float32).reshape(w.map_shape),
        np.frombuffer(w.depth_valid, np.bool_).reshape(w.map_shape),
        source_shape=w.source_shape) for w in baseline.witnesses]
    _require(all(w.witness_sha256 == original.witness_sha256 for w, original in zip(witnesses, baseline.witnesses)),
             'baseline_witness_replay_mismatch')
    sink = []
    lattice.rectify_plane(np.zeros(source_shape+(3,), np.uint8), ctx.axis, np.asarray(ctx.axis_matrix),
        ctx.focal_pixels, extent=window[2], centre=window[:2], side=window[3], samples=40,
        principal_point=ctx.principal_point, witness_sink=sink,
        witness_budget=lattice.WitnessBudgetV1(max_rectified_samples=counts['remaining_witness_samples'],
            max_bilinear_taps=counts['remaining_witness_taps'],
            max_payload_bytes=limits.max_witness_payload_bytes-previous_bytes))
    if len(sink) != 1 or type(sink[0]) is not lattice.RectificationWitnessV1:
        return refuse('density_witness_budget_exhausted')
    witness = sink[0]
    witness_index = len(witnesses)
    groups = list(baseline.groups)
    generation = set(baseline.generation_pixels)
    rejected = dict(depth=0, border=0, excluded=0, cross_role=0, validation=0)
    eligible = 0
    h, w = source_shape
    for index in range(1600):
        first, last = witness.tap_ptr[index:index+2]
        pixels = witness.tap_source_indices[first:last]
        if np.any(pixels >= h*w):
            return refuse('density_source_tap_out_of_range')
        if not witness.depth_valid.flat[index]:
            rejected['depth'] += 1
        elif np.any(pixels < 0) or any(int(p) % w == w-1 or int(p)//w == h-1 for p in pixels):
            rejected['border'] += 1
        elif any(not partition.usable[int(p)] or not partition.region[int(p)] for p in pixels):
            rejected['excluded'] += 1
        else:
            roles = {partition.roles[int(p)] for p in pixels}
            if roles == {2}:
                rejected['validation'] += 1
            elif roles != {1}:
                rejected['cross_role'] += 1
            else:
                eligible += 1
                groups.append(DependencyGroup(f'generation40-{index}', 'generation', ((witness_index, index),)))
                generation.update(int(p) for p in pixels)
    counts.update(eligible_generation_stencils=eligible,
        added_generation_pixels=len(generation)-len(baseline.generation_pixels),
        planned_generation_pixels=len(generation), rejected_depth=rejected['depth'],
        rejected_border=rejected['border'], rejected_excluded=rejected['excluded'],
        rejected_cross_role=rejected['cross_role'], omitted_validation=rejected['validation'])
    if len(generation) > policy.max_pixels_per_role:
        return refuse('density_role_source_pixel_cap')
    result = partition_tap_support(partition, witnesses+[witness], groups, limits=limits)
    if result.status != 'complete':
        return refuse(result.reason)
    _require(result.validation_pixels == baseline.validation_pixels and bool(result.validation_pixels),
             'changed_or_empty_density_validation')
    _require(set(baseline.generation_pixels) <= set(result.generation_pixels), 'lost_baseline_generation')
    _require(result.groups[:len(baseline.groups)] == baseline.groups, 'changed_baseline_groups')
    _require(result.witnesses[:len(baseline.witnesses)] == baseline.witnesses, 'changed_baseline_witnesses')
    return DenseSupport('complete', '', result, window, **counts)


@dataclass(frozen=True, slots=True)
class SupportPair:
    capture: CapturedPreparation
    dense: DenseSupport | None
    manifest_json: str
    manifest_sha256: str


def _support_manifest(capture, dense):
    p = capture.prepared
    return {'version': 'same-validation-generation-density-v1', 'dense_side': 40,
        'base_binding_sha256': None if p is None else p.binding_sha256,
        'partition_sha256': None if p is None else p.support.partition.identity_sha256,
        'policy_sha256': None if capture.fixed_policy is None else capture.fixed_policy.sha256,
        'original_support_sha256': None if p is None else p.support.identity_sha256,
        'dense_support_sha256': None if dense is None or dense.support is None else dense.support.identity_sha256,
        'validation_pixels': None if p is None else p.support.validation_pixels,
        'dense_status': None if dense is None else dense.status,
        'dense_reason': None if dense is None else dense.reason,
        'window': None if dense is None else dense.window}


def build_support_pair(capture, *, limits=ContractLimits()):
    p = capture.prepared
    dense = None if p is None else extend_generation_support(p, limits=limits)
    payload = _json(_support_manifest(capture, dense))
    return SupportPair(capture, dense, payload, _sha(payload))


def _arm_record(pair, support, fit, reason, seconds, case):
    p = pair.capture.prepared
    record = {'case': case, 'reason': reason, 'fit': None if fit is None else asdict(fit), 'seconds': seconds,
              'preparation': None, 'counts': None, 'pair_manifest_sha256': pair.manifest_sha256}
    if p is not None and support is not None:
        # Reuse accepted sealing schema, but clearly bind the arm's actual
        # support separately from the unchanged base preparation binding.
        counts = replace(p.counts, retained_samples=len(support.samples),
            generation_pixels=len(support.generation_pixels), validation_pixels=len(support.validation_pixels),
            witness_samples=sum(w.map_shape[0]*w.map_shape[1] for w in support.witnesses),
            witness_bytes=sum(w.payload_bytes for w in support.witnesses))
        declaration = replace(p, support=support, counts=counts)
        snapshot = adapter.ImageUnitFaceResult('abstained', 'density_diagnostic',
                                               prepared=declaration, counts=counts)
        original = json.loads(input_v1.seal_diagnostic(case, snapshot).payload_json)
        preparation = original['preparation']
        preparation['base_preparation_binding_sha256'] = preparation.pop('binding_sha256')
        record.update(preparation=preparation, counts=asdict(counts))
        if fit is not None:
            _require(fit.support_sha256 == support.identity_sha256, 'fit_support_binding_mismatch')
            _require(fit.context_sha256 == p.context.sha256, 'fit_context_binding_mismatch')
            _require(fit.policy_sha256 == pair.capture.fixed_policy.sha256, 'fit_policy_binding_mismatch')
    return record


def fit_pair(pair, *, case='identifiable'):
    """Both supports/manifest exist before EXACTLY one restored fit per arm."""
    _verify_dependencies()
    _require(_sha(pair.manifest_json) == pair.manifest_sha256, 'stale_support_pair_manifest')
    _require(pair.manifest_json == _json(_support_manifest(pair.capture, pair.dense)),
             'support_pair_manifest_binding_mismatch')
    capture = pair.capture
    _require(adapter.estimate_unit_face is capture.restored_fitter and adapter._fit_prepared is capture.restored_prepare,
             'density_callables_not_restored')
    if capture.prepared is None:
        reason = capture.ordinary_refusal.reason
        return {name: _arm_record(pair, None, None, reason, 0., case) for name in ('A', 'B')}
    p = capture.prepared
    adapter._verify_prepared(p)
    _require(capture.args[1] is p.context and capture.args[2] is p.support,
             'density_capture_context_support_mismatch')
    _require(capture.args[0].tobytes() == p.crop_rgb and not capture.args[0].flags.writeable,
             'density_capture_source_mismatch')
    _require(capture.fixed_policy == SearchPolicy(), 'density_capture_policy_mismatch')
    _replay_support(p.support)
    if pair.dense.support is not None:
        b = _replay_support(pair.dense.support)
        _require(b.partition == p.support.partition, 'density_partition_changed')
        _require(b.validation_pixels == p.support.validation_pixels and bool(b.validation_pixels),
                 'changed_or_empty_density_validation')
        _require(b.groups[:len(p.support.groups)] == p.support.groups and
                 b.witnesses[:len(p.support.witnesses)] == p.support.witnesses,
                 'density_baseline_proofs_changed')
        _require(set(p.support.generation_pixels) <= set(b.generation_pixels), 'lost_baseline_generation')
    records = {}
    for name, support, refusal in (
        ('A', capture.prepared.support, ''),
        ('B', pair.dense.support, pair.dense.reason)):
        if support is None:
            records[name] = _arm_record(pair, None, None, refusal, 0., case)
            continue
        args = capture.args
        started = time.perf_counter()
        fit = capture.restored_fitter(args[0], args[1], support, args[3], fixed_policy=capture.fixed_policy)
        records[name] = _arm_record(pair, support, fit, fit.reason, time.perf_counter()-started, case)
    return records


@dataclass(frozen=True, slots=True)
class SealedPair:
    payload_json: str
    sha256: str


def seal_pair(scene, pair, arms):
    _require(set(arms) == {'A', 'B'} and all(arm['case'] == scene.case for arm in arms.values()),
             'missing_or_mismatched_pair_arms')
    dense = None if pair.dense is None else {k: v for k, v in asdict(pair.dense).items() if k != 'support'}
    payload = _json({'scene_identity': scene.identity, 'scene_recipe_sha256': scene.sha256,
        'preregistration_sha256': PREREGISTRATION_SHA256, 'purpose': scene.purpose, 'case': scene.case,
        'support_manifest': json.loads(pair.manifest_json), 'support_manifest_sha256': pair.manifest_sha256,
        'preparation_seconds': pair.capture.seconds, 'ordinary_preparation_reason':
            None if pair.capture.ordinary_refusal is None else pair.capture.ordinary_refusal.reason,
        'dense_counts': dense, 'arms': arms, 'placement_supported': False})
    return SealedPair(payload, _sha(payload))


def _score_arm(recipe, arm):
    evaluated = {'status': 'unavailable', 'reason': '', 'ownership_verified': False,
        'joint_covered': None, 'distance_covered': None, 'phase_covered': None,
        'relative_distance_width': None, 'phase_arcs': None, 'primary_success': False}
    p, fit = arm['preparation'], arm['fit']
    if p is None or fit is None:
        evaluated['reason'] = 'prepared_fit_unavailable'
        return evaluated
    gauge = p['phase_gauge']
    axes = np.asarray(gauge['axis_matrix'])
    correspondence = axes @ _literal_axes(recipe['renderer_yaw'], recipe['renderer_pitch']).T
    if (p['axis'] != 2 or p['side'] != 1 or gauge['in_plane_axes'] != [0, 1]
            or gauge['convention'] != 'world-axis-block-centre-mod-one-v1'
            or not np.array_equal(np.argmax(np.abs(correspondence), axis=1), np.arange(3))
            or np.any(np.diag(correspondence) <= 0)):
        evaluated['reason'] = 'target_phase_gauge_mismatch'
        return evaluated
    evidence = p['source_tap_evidence']
    h, w = recipe['shape']
    labels = np.zeros((h, w), np.bool_)
    x0, y0, x1, y1 = recipe['face_rectangle']
    labels[y0:y1, x0:x1] = True
    crop_x, crop_y, crop_w, crop_h = p['crop']
    for role, code in (('generation', 1), ('validation', 2)):
        pixels = evidence[role+'_crop_pixels']
        parents = evidence[role+'_parent_pixels']
        if (not pixels or len(parents) != len(pixels)
                or any(type(i) is not int or not 0 <= i < crop_w*crop_h for i in pixels)
                or any(type(i) is not int or not 0 <= i < h*w for i in parents)
                or evidence[role+'_roles'] != [code]*len(pixels)
                or parents != [(i//crop_w+crop_y)*w+i % crop_w+crop_x for i in pixels]
                or evidence['source_shape'] != [crop_h, crop_w] or evidence['parent_shape'] != [h, w]
                or any(not labels.flat[index] for index in parents)):
            evaluated['reason'] = 'source_support_outside_rendered_target'
            return evaluated
    evaluated['ownership_verified'] = True
    if not fit['search_complete'] or not fit['domain_search_complete'] or not fit['cover']:
        evaluated['reason'] = 'completed_nonempty_cover_unavailable'
        return evaluated
    distance, (u, v) = recipe['distance'], recipe['phase']
    evaluated['joint_covered'] = any(box['distance'][0] <= distance <= box['distance'][1]
        and box['phase_u'][0] <= u <= box['phase_u'][1]
        and box['phase_v'][0] <= v <= box['phase_v'][1] for box in fit['cover'])
    lower, upper = fit['distance_interval']
    evaluated['distance_covered'] = lower <= distance <= upper
    evaluated['relative_distance_width'] = (upper-lower)/lower
    evaluated['phase_arcs'] = fit['phase_intervals']
    evaluated['phase_covered'] = [((truth-start) % 1) <= width
                                for truth, (start, width) in zip((u, v), fit['phase_intervals'])]
    evaluated['status'] = 'evaluated'
    evaluated['primary_success'] = (fit['status'] == 'supported' and evaluated['joint_covered']
        and upper-lower < .10*lower and all(width < .2 for _, width in fit['phase_intervals'])
        and fit['representative'] is not None and bool(fit['source_validation_sha256']))
    return evaluated


def _same_validation(p_a, p_b):
    if not p_a or not p_b:
        return False
    a, b = p_a['source_tap_evidence'], p_b['source_tap_evidence']
    return bool(a['validation_parent_pixels'] and b['validation_parent_pixels']
        and a['validation_parent_pixels'] == b['validation_parent_pixels']
        and a['validation_crop_pixels'] == b['validation_crop_pixels']
        and a['validation_roles'] == b['validation_roles'] == [2]*len(a['validation_parent_pixels'])
        and p_a['crop'] == p_b['crop'] and a['source_shape'] == b['source_shape']
        and a['parent_shape'] == b['parent_shape'])


def evaluate_pair(scene, sealed):
    """Only after BOTH fit records sealed: label, gauge, truth and success."""
    if _sha(sealed.payload_json) != sealed.sha256:
        raise ValueError('stale_pair_seal')
    record = json.loads(sealed.payload_json)
    if (record['scene_identity'] != scene.identity or record['scene_recipe_sha256'] != scene.sha256
            or record['preregistration_sha256'] != PREREGISTRATION_SHA256):
        raise ValueError('scene_recipe_binding_mismatch')
    if scene.case != 'identifiable':
        reason = 'mixed_depth_has_no_single_plane_truth' if scene.case == 'mixed_depth' else 'constant_texture_control'
        return {'status': 'not_applicable', 'reason': reason, 'arms': None, 'primary_success': False}
    recipe = json.loads(scene.recipe_json)
    arms = {name: _score_arm(recipe, arm) for name, arm in record['arms'].items()}
    p_a, p_b = record['arms']['A']['preparation'], record['arms']['B']['preparation']
    same_validation = _same_validation(p_a, p_b)
    return {'status': 'evaluated', 'reason': '', 'arms': arms, 'same_validation': same_validation,
            'primary_success': bool(same_validation and arms['B']['primary_success'])}


def summarize_pair(scene, sealed, evaluated):
    record = json.loads(sealed.payload_json)
    arms = {}
    for name, arm in record['arms'].items():
        fit = arm['fit']
        arms[name] = {'reason': arm['reason'], 'seconds': arm['seconds'], 'counts': arm['counts'],
            'support_sha256': None if arm['preparation'] is None else arm['preparation']['support_sha256'],
            'fit': None if fit is None else {k: fit[k] for k in (
                'status', 'search_complete', 'domain_search_complete', 'work_used', 'work_budget',
                'actual_pixel_work', 'estimated_pixel_work', 'pixel_work_budget', 'retained_boxes',
                'distance_interval', 'phase_intervals', 'compatible_faces', 'unique_templates',
                'source_generation_sha256', 'source_validation_sha256', 'policy_sha256')},
            'candidate_count': 0 if fit is None else len(fit['candidates']),
            'alias_count': 0 if fit is None else sum(len(c['state_aliases']) for c in fit['candidates']),
            'cover_geometries': [] if fit is None else sorted({b['geometry_sha256'] for b in fit['cover']})}
    return {'scene': asdict(scene), 'scene_recipe_sha256': scene.sha256,
        'sealed_pair_sha256': sealed.sha256, 'manifest_sha256': record['support_manifest_sha256'],
        'preparation_seconds': record['preparation_seconds'], 'dense_counts': record['dense_counts'],
        'arms': arms, 'evaluation': evaluated}


REPORT_PATH = Path(__file__).resolve().parents[2] / 'docs/reports/UNIT-FACE-DENSITY-01-2026-10-05.md'
EVIDENCE_PATH = REPORT_PATH.with_name('UNIT-FACE-DENSITY-01-evidence-2026-10-05.jsonl')
RUN_MARKER = 'HELDOUT_EXECUTION_STATE: NOT_STARTED'


def _report_transition(text, state, payload):
    """Pure transition checked by tests; actual writes use an exclusive lock."""
    _require(PREREGISTRATION_SHA256 in text, 'report_preregistration_pin_missing')
    if state == 'RUNNING':
        _require(text.count(RUN_MARKER) == 1, 'heldout_execution_already_claimed')
        return text.replace(RUN_MARKER, 'HELDOUT_EXECUTION_STATE: RUNNING')
    _require(state in ('COMPLETE', 'FAILED') and text.count('HELDOUT_EXECUTION_STATE: RUNNING') == 1,
             'invalid_heldout_result_transition')
    return text.replace('HELDOUT_EXECUTION_STATE: RUNNING', 'HELDOUT_EXECUTION_STATE: '+state) + (
        '\n\nDurable one-shot result:\n\n```json\n'+_json(payload)+'\n```\n')


def _update_owned_report(state, payload=None):
    """Lock/fsync the single authorized report; no additional marker artifact."""
    _require(REPORT_PATH.is_file(), 'owned_preregistration_report_missing')
    with REPORT_PATH.open('r+', encoding='utf-8', newline='') as report:
        if os.name == 'nt':
            import msvcrt
            report.seek(0)
            msvcrt.locking(report.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: (report.seek(0), msvcrt.locking(report.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            fcntl.flock(report.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(report.fileno(), fcntl.LOCK_UN)
        try:
            report.seek(0)
            replacement = _report_transition(report.read(), state, payload)
            report.seek(0)
            report.write(replacement)
            report.truncate()
            report.flush()
            os.fsync(report.fileno())
        finally:
            unlock()


def _dependency_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ('src/mc_photogrammetry/pipeline/image_derived_unit_face.py',
             'src/mc_photogrammetry/reconstruction/unit_face_estimator.py',
             'src/mc_photogrammetry/reconstruction/unit_face_lattice.py',
             'src/mc_photogrammetry/reconstruction/lattice.py',
             'src/mc_photogrammetry/reconstruction/calibration.py',
             'src/mc_photogrammetry/reconstruction/camera.py',
             'src/mc_photogrammetry/reconstruction/region_growth.py',
             'src/mc_photogrammetry/reconstruction/boundary_candidates.py',
             'src/mc_photogrammetry/vision/gui_mask.py',
             'src/mc_photogrammetry/blocks/unit_face_catalogue.py',
             'src/mc_photogrammetry/pipeline/contracts.py',
             'src/mc_photogrammetry/domain/models.py',
             'scripts/evaluation/run_unit_face_input.py')
    return {'source_sha256': {path: _sha((root/path).read_bytes()) for path in paths},
            'runtime_versions': {'python': sys.version, 'numpy': np.__version__,
                                 'opencv': cv2.__version__, 'pillow': PIL.__version__}}


def _verify_dependencies():
    _require(_dependency_hashes() == json.loads(PREREGISTRATION_JSON)['dependencies'],
             'changed_preregistered_dependency')


def _evidence_header():
    return {'kind': 'unit-face-density-sealed-pairs-v1',
            'preregistration_sha256': PREREGISTRATION_SHA256}


def _initialize_evidence():
    """Exclusive creation after durable study claim; never truncate/reuse."""
    try:
        with EVIDENCE_PATH.open('xb') as artifact:
            artifact.write((_json(_evidence_header())+'\n').encode())
            artifact.flush()
            os.fsync(artifact.fileno())
    except FileExistsError as error:
        raise DensityIntegrityError('density_evidence_already_exists') from error
    _verify_evidence(())


def _verify_evidence(expected_pairs):
    """Replay the complete retained prefix, including full payload hashes."""
    _require(EVIDENCE_PATH.is_file(), 'density_evidence_missing')
    raw = EVIDENCE_PATH.read_bytes()
    try:
        _require(raw.endswith(b'\n'), 'density_evidence_truncated')
        lines = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        _require(lines and lines[0] == _evidence_header(), 'density_evidence_header_mismatch')
        _require(len(lines)-1 == len(expected_pairs), 'density_evidence_prefix_mismatch')
        for index, (record, expected_sha) in enumerate(zip(lines[1:], expected_pairs)):
            scene = SCENES[index]
            _require(set(record) == {'preregistration_sha256', 'scene_identity',
                                     'sealed_pair_sha256', 'sealed_pair_payload_json'},
                     'density_evidence_record_shape_mismatch')
            _require(record['preregistration_sha256'] == PREREGISTRATION_SHA256
                     and record['scene_identity'] == scene.identity
                     and record['sealed_pair_sha256'] == expected_sha
                     and _sha(record['sealed_pair_payload_json']) == expected_sha,
                     'density_evidence_pair_mismatch')
            payload = json.loads(record['sealed_pair_payload_json'])
            _require(payload['preregistration_sha256'] == PREREGISTRATION_SHA256
                     and payload['scene_identity'] == scene.identity
                     and payload['scene_recipe_sha256'] == scene.sha256,
                     'density_evidence_scene_binding_mismatch')
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError) as error:
        raise DensityIntegrityError('density_evidence_malformed') from error
    return _sha(raw)


def _append_sealed_evidence(scene, sealed, previous_pairs):
    """Durably append and replay BEFORE consulting any evaluator truth."""
    _verify_evidence(previous_pairs)
    _require(len(previous_pairs) < len(SCENES) and scene == SCENES[len(previous_pairs)],
             'density_evidence_scene_order_mismatch')
    _require(_sha(sealed.payload_json) == sealed.sha256, 'density_evidence_stale_pair_seal')
    payload = json.loads(sealed.payload_json)
    _require(payload['preregistration_sha256'] == PREREGISTRATION_SHA256
             and payload['scene_identity'] == scene.identity
             and payload['scene_recipe_sha256'] == scene.sha256,
             'density_evidence_scene_binding_mismatch')
    record = {'preregistration_sha256': PREREGISTRATION_SHA256,
              'scene_identity': scene.identity, 'sealed_pair_sha256': sealed.sha256,
              'sealed_pair_payload_json': sealed.payload_json}
    # r+b forbids recreation if removed between replay and append. Always seek
    # to the end; prior sealed evidence is never overwritten or truncated.
    with EVIDENCE_PATH.open('r+b') as artifact:
        artifact.seek(0, os.SEEK_END)
        artifact.write((_json(record)+'\n').encode())
        artifact.flush()
        os.fsync(artifact.fileno())
    return _verify_evidence(tuple(previous_pairs)+(sealed.sha256,))


def run_study():
    """Run fixed development/control pairs and ONE disjoint heldout pair once."""
    _verify_dependencies()
    _require([asdict(scene) for scene in SCENES] == json.loads(PREREGISTRATION_JSON)['scenes'],
             'changed_preregistered_scene_registry')
    _require(not EVIDENCE_PATH.exists(), 'density_evidence_already_exists')
    # Claim DURABLY before ANY scene/fit. A crash/failure keeps the claim and
    # cannot silently rerun the heldout pair. Tests never call this real entry.
    _update_owned_report('RUNNING')
    summaries = []
    retained_pairs = []
    try:
        _initialize_evidence()
        for scene in SCENES:
            image, catalogue = render_scene(scene)
            capture = capture_rgb_preparation(image, catalogue, scene.identity)
            pair = build_support_pair(capture)
            arms = fit_pair(pair, case=scene.case)
            sealed = seal_pair(scene, pair, arms)
            _append_sealed_evidence(scene, sealed, tuple(retained_pairs))
            retained_pairs.append(sealed.sha256)
            summaries.append(summarize_pair(scene, sealed, evaluate_pair(scene, sealed)))
        result = {'preregistration_sha256': PREREGISTRATION_SHA256, 'scenes': summaries,
                  'evidence_path': EVIDENCE_PATH.name, 'sealed_pairs': retained_pairs,
                  'evidence_sha256': _verify_evidence(tuple(retained_pairs)),
                  'placement_supported': False, 'generalization_claim': False}
        _update_owned_report('COMPLETE', result)
        return result
    except BaseException as error:
        try:
            _update_owned_report('FAILED', {'error_type': type(error).__name__,
                'completed_summaries': summaries, 'retained_sealed_pairs': retained_pairs,
                'evidence_path': EVIDENCE_PATH.name,
                'evidence_sha256': _sha(EVIDENCE_PATH.read_bytes()) if EVIDENCE_PATH.is_file() else None})
        except BaseException:
            # Initial claim is already durable and disallows reruns. Preserve
            # the original experiment exception if reporting itself fails.
            pass
        raise


def _freeze_preregistration():
    functions = (DensityIntegrityError, _require, _replay_support, _literal_axes,
        render_scene, capture_rgb_preparation,
        extend_generation_support, _support_manifest, build_support_pair, _arm_record, fit_pair,
        seal_pair, _score_arm, _same_validation, evaluate_pair, _report_transition, _update_owned_report,
        _dependency_hashes, _verify_dependencies, _evidence_header, _initialize_evidence,
        _verify_evidence, _append_sealed_evidence, run_study)
    return _json({'scenes': [asdict(scene) for scene in SCENES],
        'arms': ['original20', 'original20-plus-generation40'], 'dense_side': 40,
        'support_algorithm_sha256': _sha('\n'.join(inspect.getsource(f) for f in functions)),
        'contract_caps': asdict(ContractLimits()), 'search_policy': asdict(SearchPolicy()),
        'policy_sha256': SearchPolicy().sha256, 'dependencies': _dependency_hashes(),
        'success_rules': {'strict_distance_width_over_lower': .10, 'strict_phase_arc_width': .2,
            'joint_renderer_truth_required': True, 'supported_frozen_validation_required': True,
            'identical_nonempty_validation_required': True, 'all_source_taps_target_owned': True,
            'axis_side_phase_gauge_required': True, 'controls_no_single_plane_score': True},
        'one_shot': 'exclusive-locked-owned-report-marker-before-any-fit-and-durable-result',
        'evidence': 'exclusive-new-jsonl-full-sealed-pair+sha-fsync+replay-before-evaluation',
        'evidence_path': EVIDENCE_PATH.name})


PREREGISTRATION_JSON = _freeze_preregistration()
PREREGISTRATION_SHA256 = _sha(PREREGISTRATION_JSON)


if __name__ == '__main__':
    print('PREREGISTRATION ' + PREREGISTRATION_SHA256, flush=True)
    print(_json(run_study()), flush=True)
