"""Preregistered generated RGB input experiment; no external inputs or exports.

Recipe v1 is fixed before its first run: 800x600, focal 700, yaw 23,
pitch -14, normal-Z face at 3.23, phases .173/.287, native 16 RGB Fourier
texels; independent Manhattan line pencils occupy the side bands. Controls
replace alternating face strips with depth 4.6835, or constant RGB 128.
Truth is consulted only after the immutable diagnostic JSON has been sealed.
This script never supplies its renderer optics/regions/geometry to the adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from mc_photogrammetry.blocks.unit_face_catalogue import load_unit_face_catalogue
from mc_photogrammetry.pipeline.image_derived_unit_face import estimate_image_unit_face


RECIPE = {
    'version': 'UNIT-FACE-INPUT-01-generated-v1', 'shape': [600, 800],
    'renderer_focal': 700., 'renderer_yaw': 23., 'renderer_pitch': -14.,
    'distance': 3.23, 'phase': [.173, .287], 'native_side': 16,
    'face_rectangle': [270, 90, 650, 480], 'structure_band_edges': [260, 660],
    'line_anchors': [30, 100, 180, 230, 700, 760],
    'line_rows': [95, 145, 195, 245, 295, 345, 395, 445, 495],
    'controls': ['identifiable', 'mixed_depth', 'ambiguous'],
}
RECIPE_JSON = json.dumps(RECIPE, sort_keys=True, separators=(',', ':'), allow_nan=False)
RECIPE_SHA256 = hashlib.sha256(RECIPE_JSON.encode()).hexdigest()


def native_texture(ambiguous=False):
    if ambiguous:
        return np.full((16, 16, 3), 128, np.uint8)
    y, x = np.meshgrid((np.arange(16) + .5) * 2 * np.pi / 16,
                       (np.arange(16) + .5) * 2 * np.pi / 16, indexing='ij')
    return np.rint(np.stack((128 + 60*np.sin(x) + 35*np.cos(y) + 15*np.sin(x+2*y+.3),
        128 + 55*np.cos(x+.6) + 40*np.sin(y+.8) + 17*np.sin(2*x-y+.2),
        128 + 45*np.sin(x+y+.17) + 45*np.cos(2*x+.9) + 20*np.sin(2*y+.8)), axis=-1)).astype(np.uint8)


def generated_catalogue(texture):
    """Generated deterministic archive, not a JAR scan or reference palette."""
    rgba = np.concatenate((texture, np.full(texture.shape[:2] + (1,), 255, np.uint8)), axis=-1)
    stream = io.BytesIO()
    Image.fromarray(rgba).save(stream, format='PNG')
    assets = {'assets/minecraft/textures/block/input.png': stream.getvalue(),
        'assets/minecraft/models/block/input.json': {
            'textures': {'all': 'block/input'}, 'elements': [{'from': [0, 0, 0], 'to': [16, 16, 16],
                'faces': {slot: {'texture': '#all'} for slot in
                          ('down', 'up', 'north', 'south', 'west', 'east')}}]},
        'assets/minecraft/blockstates/input.json': {'variants': {'': {'model': 'block/input'}}}}
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, 'w') as archive:
        for name, value in sorted(assets.items()):
            archive.writestr(zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0)),
                             value if isinstance(value, bytes) else json.dumps(value, sort_keys=True))
    return load_unit_face_catalogue(archive_bytes.getvalue())


def render_rgb(case):
    """Independent literal pinhole/periodic native bilinear renderer."""
    if case not in RECIPE['controls']:
        raise ValueError('unknown_generated_control')
    h, w = RECIPE['shape']
    yaw, pitch = map(math.radians, (RECIPE['renderer_yaw'], RECIPE['renderer_pitch']))
    # Rows are world axes in camera coordinates; this is stated independently
    # of camera/calibration/lattice/fitter transformation helpers.
    axes = np.array(((-math.cos(yaw), -math.sin(pitch)*math.sin(yaw), -math.cos(pitch)*math.sin(yaw)),
        (0., math.cos(pitch), -math.sin(pitch)),
        (-math.sin(yaw), math.sin(pitch)*math.cos(yaw), math.cos(pitch)*math.cos(yaw))))
    yy, xx = np.indices((h, w))
    rays = np.stack(((xx-w/2)/RECIPE['renderer_focal'],
                     -(yy-h/2)/RECIPE['renderer_focal'], np.ones((h, w))), axis=-1)
    unit = rays / np.abs(rays @ axes[2])[..., None]
    distance = np.full((h, w), RECIPE['distance'])
    if case == 'mixed_depth':
        distance[(xx // 48) % 2 == 0] *= 1.45
    bx = distance * (unit @ axes[0]) + RECIPE['phase'][0]
    by = distance * (unit @ axes[1]) + RECIPE['phase'][1]
    texture = native_texture(case == 'ambiguous')
    tx, ty = ((-bx+.5) % 1)*16-.5, ((-by+.5) % 1)*16-.5
    ix, iy = np.floor(tx).astype(int), np.floor(ty).astype(int)
    dx, dy = tx-ix, ty-iy
    color = np.zeros((h, w, 3))
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        color += weight[..., None] * texture[(iy+oy) % 16, (ix+ox) % 16]
    image = np.full((h, w, 3), 112, np.uint8)
    x0, y0, x1, y1 = RECIPE['face_rectangle']
    image[y0:y1, x0:x1] = np.rint(color[y0:y1, x0:x1]).astype(np.uint8)
    # Independently rendered directions through image anchors; line endpoints
    # are projected from homogeneous Manhattan vanishing points, not supplied
    # to calibration. Bands keep calibration marks out of the native face.
    ink = np.zeros_like(image)
    for axis, direction in enumerate(axes):
        homogeneous = np.array((RECIPE['renderer_focal']*direction[0] + w/2*direction[2],
                                -RECIPE['renderer_focal']*direction[1] + h/2*direction[2],
                                direction[2]))
        for row in RECIPE['line_rows']:
            for col in RECIPE['line_anchors']:
                anchor = np.array((col, row), dtype=float)
                delta = (anchor-homogeneous[:2]/homogeneous[2]) if abs(homogeneous[2]) > 1e-10 else homogeneous[:2]
                delta /= np.linalg.norm(delta)
                a, b = anchor-90*delta, anchor+90*delta
                value = (220, 30, 200)[axis]
                cv2.line(ink, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                         (value, value, value), 2, cv2.LINE_AA)
    marks = ink.any(axis=-1) & ((xx < 260) | (xx >= 660))
    image[marks] = ink[marks]
    return image, generated_catalogue(texture)


@dataclass(frozen=True, slots=True)
class SealedDiagnostic:
    payload_json: str
    sha256: str


def seal_diagnostic(case, result):
    record = {'case': case, 'recipe_sha256': RECIPE_SHA256,
        'status': result.status, 'reason': result.reason,
        'calibration_status': result.calibration_status, 'proposal_status': result.proposal_status,
        'counts': asdict(result.counts), 'placement_supported': result.placement_supported,
        'validation_interpretation': result.validation_interpretation,
        'full_calibration': None if result.full_calibration is None else asdict(result.full_calibration),
        'preparation': None, 'fit': None}
    if result.prepared is not None:
        p = result.prepared
        record['preparation'] = {'binding_sha256': p.binding_sha256, 'frame_sha256': p.frame.sha256,
            'full_calibration': asdict(p.full_calibration), 'cropped_calibration': asdict(p.cropped_calibration),
            'proposal_sha256': p.proposal.sha256, 'proposal_order': p.proposal_order,
            'axis': p.proposal.axis, 'side': p.proposal.side, 'crop': p.crop,
            'context_sha256': p.context.sha256, 'support_sha256': p.support.identity_sha256,
            'policy_sha256': p.policy.sha256,
            'phase_gauge': {'convention': 'world-axis-block-centre-mod-one-v1',
                'normal_axis': p.context.axis, 'side': p.proposal.side,
                'in_plane_axes': [i for i in range(3) if i != p.context.axis],
                'axis_matrix': p.context.axis_matrix},
            'source_tap_evidence': {
                'source_shape': p.context.source_shape, 'parent_shape': p.frame.shape,
                'generation_crop_pixels': p.support.generation_pixels,
                'validation_crop_pixels': p.support.validation_pixels,
                'generation_parent_pixels': p.parent_pixels('generation'),
                'validation_parent_pixels': p.parent_pixels('validation'),
                'generation_roles': [p.support.partition.roles[i] for i in p.support.generation_pixels],
                'validation_roles': [p.support.partition.roles[i] for i in p.support.validation_pixels],
                'samples': [asdict(sample) for sample in p.support.samples],
                'groups': [asdict(group) for group in p.support.groups]}}
    if result.fit is not None:
        record['fit'] = asdict(result.fit)
    payload = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return SealedDiagnostic(payload, hashlib.sha256(payload.encode()).hexdigest())


def _rendered_target_labels():
    """Evaluator-only renderer labels, never supplied to image preparation."""
    labels = np.zeros(tuple(RECIPE['shape']), np.bool_)
    x0, y0, x1, y1 = RECIPE['face_rectangle']
    labels[y0:y1, x0:x1] = True
    return labels


def _verify_target_support(diagnostic):
    """Replay sealed source ownership/roles, then check independent labels."""
    p = diagnostic['preparation']
    if p is None:
        return 'missing_prepared_source_support', None
    if p['axis'] != 2 or p['side'] != 1:
        return 'target_plane_axis_or_side_mismatch', None
    gauge = p.get('phase_gauge', {})
    if (gauge.get('convention') != 'world-axis-block-centre-mod-one-v1'
            or gauge.get('normal_axis') != 2 or gauge.get('side') != 1
            or gauge.get('in_plane_axes') != [0, 1]):
        return 'target_phase_gauge_mismatch', None
    # Gauge matching names signed world axes, not exact optics. Unknown optical
    # error is still outside the fitter's conditional interval certificate.
    yaw, pitch = map(math.radians, (RECIPE['renderer_yaw'], RECIPE['renderer_pitch']))
    renderer_axes = np.array(((-math.cos(yaw), -math.sin(pitch)*math.sin(yaw), -math.cos(pitch)*math.sin(yaw)),
        (0., math.cos(pitch), -math.sin(pitch)),
        (-math.sin(yaw), math.sin(pitch)*math.cos(yaw), math.cos(pitch)*math.cos(yaw))))
    try:
        axes = np.asarray(gauge.get('axis_matrix'), dtype=float)
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            return 'target_phase_gauge_mismatch', None
        correspondence = axes @ renderer_axes.T
        if (not np.array_equal(np.argmax(np.abs(correspondence), axis=1), np.arange(3))
                or np.any(np.diag(correspondence) <= 0)):
            return 'target_phase_gauge_mismatch', None
        e = p['source_tap_evidence']
        parent_h, parent_w = e['parent_shape']
        x0, y0, width, height = p['crop']
        if (e['parent_shape'] != RECIPE['shape'] or e['source_shape'] != [height, width]
                or any(type(v) is not int for v in p['crop'])
                or min(x0, y0) < 0 or not 0 < width <= 256 or not 0 < height <= 256
                or x0+width > parent_w or y0+height > parent_h):
            return 'invalid_sealed_source_evidence', None
        ownership = {}
        parent_pixels = {}
        for role, code in (('generation', 1), ('validation', 2)):
            pixels = e[role+'_crop_pixels']
            parents = e[role+'_parent_pixels']
            if (not pixels or any(type(i) is not int or not 0 <= i < width*height for i in pixels)
                    or pixels != sorted(set(pixels)) or len(parents) != len(pixels)
                    or e[role+'_roles'] != [code]*len(pixels)
                    or diagnostic['counts'][role+'_pixels'] != len(pixels)):
                return 'invalid_sealed_source_evidence', None
            if parents != [(i//width+y0)*parent_w + i % width+x0 for i in pixels]:
                return 'source_crop_parent_binding_mismatch', None
            if any(i in ownership for i in pixels):
                return 'overlapping_sealed_source_roles', None
            ownership.update((i, code) for i in pixels)
            parent_pixels[role] = parents
        samples = {(s['witness_index'], s['sample_index']): s for s in e['samples']}
        if len(samples) != len(e['samples']) or not samples:
            return 'invalid_sealed_tap_closure', None
        observed = {'generation': set(), 'validation': set()}
        selected = set()
        for group in e['groups']:
            role = group['role']
            if role not in observed or not group['sample_refs']:
                return 'invalid_sealed_tap_closure', None
            for ref in group['sample_refs']:
                key = tuple(ref)
                sample = samples[key]
                taps = sample['taps']
                if (not sample['depth_valid'] or not 0 < len(taps) <= 4
                        or any(type(i) is not int or type(w) is not int or w <= 0 for i, w in taps)
                        or sum(w for _, w in taps) != 1024):
                    return 'invalid_sealed_tap_closure', None
                code = 1 if role == 'generation' else 2
                if any(ownership.get(i) != code for i, _ in taps):
                    return 'cross_role_sealed_source_tap', None
                observed[role].update(i for i, _ in taps)
                selected.add(key)
        if (selected != set(samples) or any(observed[role] != set(e[role+'_crop_pixels'])
                                           for role in observed)):
            return 'invalid_sealed_tap_closure', None
        labels = _rendered_target_labels()
        if any(not labels.flat[index] for pixels in parent_pixels.values() for index in pixels):
            return 'source_support_outside_rendered_target', None
        return '', {role: len(pixels) for role, pixels in parent_pixels.items()}
    except (KeyError, TypeError, ValueError, IndexError):
        return 'invalid_sealed_source_evidence', None


def _score_single_plane(diagnostic, evaluated):
    """Called only after scope, recipe, gauge and ALL tap ownership checks."""
    fit = diagnostic['fit']
    d, (u, v) = RECIPE['distance'], RECIPE['phase']
    evaluated['joint_covered'] = any(box['distance'][0] <= d <= box['distance'][1]
        and box['phase_u'][0] <= u <= box['phase_u'][1]
        and box['phase_v'][0] <= v <= box['phase_v'][1] for box in fit['cover'])
    interval, arcs = fit['distance_interval'], fit['phase_intervals']
    if interval is not None:
        evaluated['distance_covered'] = interval[0] <= d <= interval[1]
    if arcs is not None:
        evaluated['phase_covered'] = [((p-start) % 1) <= width for p, (start, width) in zip((u, v), arcs)]
    candidate = fit['representative']
    if candidate is not None:
        evaluated['distance_relative_error'] = abs(candidate['distance']/d-1)
        evaluated['phase_circular_errors'] = [abs((p-t+.5) % 1-.5) for p, t in zip(candidate['phase'], (u, v))]
    return evaluated


def evaluate_sealed(sealed):
    """Post-seal scoring is typed by scene scope; controls have no scalar score."""
    if hashlib.sha256(sealed.payload_json.encode()).hexdigest() != sealed.sha256:
        raise ValueError('stale_sealed_diagnostic')
    diagnostic = json.loads(sealed.payload_json)
    evaluated = {'sealed_sha256': sealed.sha256, 'accuracy_claim': False,
                 'truth_scope': 'unavailable', 'truth_status': 'unavailable', 'truth_reason': '',
                 'source_ownership_verified': False, 'source_ownership_counts': None,
                 'distance_covered': None, 'phase_covered': None, 'joint_covered': None,
                 'distance_relative_error': None, 'phase_circular_errors': None}
    current_recipe = json.dumps(RECIPE, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (diagnostic['recipe_sha256'] != RECIPE_SHA256
            or hashlib.sha256(current_recipe.encode()).hexdigest() != RECIPE_SHA256):
        evaluated['truth_reason'] = 'recipe_binding_mismatch'
        return evaluated
    case = diagnostic['case']
    if case in ('mixed_depth', 'ambiguous'):
        evaluated.update(truth_scope=case, truth_status='not_applicable',
            truth_reason=('mixed_depth_has_no_single_plane_truth' if case == 'mixed_depth'
                          else 'ambiguous_scene_has_no_identifiable_unit_anchor'))
        return evaluated
    if case != 'identifiable':
        evaluated['truth_reason'] = 'unknown_scene_truth_scope'
        return evaluated
    evaluated['truth_scope'] = 'identifiable_single_plane_target'
    reason, counts = _verify_target_support(diagnostic)
    if reason:
        evaluated['truth_reason'] = reason
        return evaluated
    evaluated.update(source_ownership_verified=True, source_ownership_counts=counts)
    fit = diagnostic['fit']
    if fit is None or not fit['domain_search_complete'] or not fit['search_complete'] or not fit['cover']:
        evaluated['truth_reason'] = 'completed_nonempty_fit_unavailable'
        return evaluated
    evaluated['truth_status'] = 'evaluated'
    return _score_single_plane(diagnostic, evaluated)


def run_controlled_scene():
    sealed = []
    for case in RECIPE['controls']:
        image, catalogue = render_rgb(case)
        result = estimate_image_unit_face(image, catalogue, frame_id=case)
        sealed.append(seal_diagnostic(case, result))
    # Seal ALL case outputs before evaluator truth is inspected.
    return {'recipe': RECIPE, 'recipe_sha256': RECIPE_SHA256,
            'diagnostics': [json.loads(s.payload_json) for s in sealed],
            'evaluation': [evaluate_sealed(s) for s in sealed]}


if __name__ == '__main__':
    print(json.dumps(run_controlled_scene(), sort_keys=True, indent=2, allow_nan=False))
