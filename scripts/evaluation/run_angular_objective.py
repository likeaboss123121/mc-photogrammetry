"""ANGULAR-OBJECTIVE-01 evaluation only; explicit setup, no import-time study.

Both arms use accepted GUI-border treatment. Only B's delegated scoring function
changes; geometry helpers, solver control flow and pixel-labeled result fields
stay untouched. Study execution awaits exact-package review and lead clearance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))

from scripts.evaluation import run_gui_border as border


class AngularStudyError(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def require(condition, reason):
    if not condition:
        raise AngularStudyError(reason)


canonical, sha = border.canonical, border.sha
cam, calibration = border.cam, border.cam.calibration
Scene = border.Scene
ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT/'docs/reports/ANGULAR-OBJECTIVE-01-2026-10-05.md'
EVIDENCE_PATH = ROOT/'docs/reports/ANGULAR-OBJECTIVE-01-evidence-2026-10-05.jsonl.gz'
TAU = math.asin(6./120.)
LOSS = {'A': 'unchanged pixel/length score', 'B': 'sum L*max(1-asin(clip(2*r/L,0,1))/tau,0)',
        'tau_radians': TAU, 'tau_degrees': math.degrees(TAU), 'reference_length_pixels': 120.,
        'pixel_threshold': 3., 'zero_length_contribution': 0., 'threshold_sweep': False}
CAPS = dict(border.CAPS)
MAX_EVIDENCE_BYTES = 512*1024*1024
GATE = dict(border.PILOT_GATE, minimum_corrected_wrong_families=3)
_PIXEL_SCORE = calibration._score
_OBJECTIVE_LOCK, _WRITER_LOCK, _STUDY_LOCK = threading.Lock(), threading.Lock(), threading.Lock()
PREREGISTRATION_JSON = PREREGISTRATION_SHA256 = None


def scenes():
    settings = ((17., 12., 58., 320, 448), (34., -17., 73., 400, 544),
                (57., 7., 86., 352, 512), (76., -24., 64., 416, 576))
    result = []
    for kind_index, kind in enumerate(('planar', 'stepped', 'occluding')):
        for index, (yaw, pitch, fov, h, w) in enumerate(settings):
            seed = 8_104_000+kind_index*101+index*13
            rng = np.random.default_rng(seed)
            eye = [1.1-.24*index, 4.45+.16*index, -2.2-.21*index]
            forward = np.array([-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))])
            recipe = {'version': 'ANGULAR-OBJECTIVE-01-world-native-v1', 'rng': 'numpy-PCG64',
                'seed': seed, 'texture_seed': seed+333_001, 'shape': [h, w], 'yaw': yaw, 'pitch': pitch,
                'vertical_fov': fov, 'focal': h/2./math.tan(math.radians(fov)/2), 'roll': 0.,
                'principal': [w/2., h/2.], 'eye': eye, 'native_side': 16,
                'back_plane': {'axis': 2, 'side': -1, 'value': eye[2]+forward[1]*(15.7+index*.6),
                               'bounds': [[-65., 18.], [0., 16.]]},
                'floor': kind != 'planar', 'boxes': [], 'crop': None,
                'low_texture': False, 'gui_contamination': False, 'seam_width_blocks': .016+.001*index}
            if kind != 'planar':
                for distance, offset in ((4.8, -1.5), (8.1, 1.4), (11.7, -.7)):
                    centre = np.array([eye[0], eye[2]])+forward*distance
                    centre += np.array([forward[1], -forward[0]])*offset+rng.uniform(-.4, .4, 2)
                    size = rng.uniform([1.7, 1.3], [3.1, 2.7])
                    recipe['boxes'].append({'lo': [float(centre[0]-size[0]/2), 0., float(centre[1]-size[1]/2)],
                        'hi': [float(centre[0]+size[0]/2), float(rng.uniform(1.8, 4.2)), float(centre[1]+size[1]/2)]})
            if kind == 'occluding':
                centre = np.array([eye[0], eye[2]])+forward*3.8
                recipe['boxes'].append({'lo': [float(centre[0]-1.3), 0., float(centre[1]-.85)],
                    'hi': [float(centre[0]+1.05), 5.2+.17*index, float(centre[1]+.9)]})
            family = f'angular-{kind}-{index+1}'
            for variant in ('primary', 'crop', 'gui-only'):
                changed = json.loads(canonical(recipe))
                if variant == 'crop':
                    changed['crop'] = [27, 17, w-53, h-49]
                changed['gui_contamination'] = variant == 'gui-only'
                result.append(Scene(f'{family}-{variant}', family, variant, 'new-world', canonical(changed)))
    return tuple(result)


SCENES = scenes()


def render_rgb(scene):
    return border.render_new(json.loads(scene.recipe_json))[0]


def truth_camera(scene):
    return border.truth_camera(scene)


def angular_terms(residuals, lengths):
    """Zero-weight degenerate lines contribute zero; clip roundoff at +/-1."""
    r, L = np.asarray(residuals), np.asarray(lengths)
    require(r.shape == L.shape and np.isfinite(r).all() and np.isfinite(L).all()
            and (L >= 0).all(), 'invalid_angular_arrays')
    valid = L > 0
    sine = np.zeros(L.shape, float)
    np.divide(2*r, L, out=sine, where=valid)
    theta = np.arcsin(np.clip(sine, 0., 1.))
    contributions = L*np.maximum(1-theta/TAU, 0.)
    contributions[~valid] = 0.
    return theta, contributions, valid


def angular_score(segments, lengths, focal, yaw, pitch, width, height, threshold,
                  roll=0., principal_point=None):
    require(threshold == 3. and all(math.isfinite(float(v)) for v in (focal, yaw, pitch, width, height, roll)),
            'angular_camera_or_threshold_drift')
    require(isinstance(segments, np.ndarray) and segments.ndim == 2 and segments.shape[1] == 4
            and np.isfinite(segments).all() and isinstance(lengths, np.ndarray)
            and lengths.shape == (len(segments),), 'angular_segment_contract')
    geometric = np.hypot(segments[:, 2]-segments[:, 0], segments[:, 3]-segments[:, 1])
    require(np.array_equal(lengths, geometric), 'angular_weights_are_not_original_lengths')
    if not len(segments):
        return 0.
    points = calibration.vanishing_points(focal, yaw, pitch, width, height, roll, principal_point)
    residuals = calibration.segment_residuals(segments, points).min(axis=1)
    _, contributions, _ = angular_terms(residuals, lengths)
    result = float(contributions.sum())
    require(math.isfinite(result), 'nonfinite_angular_score')
    return result


def trace_arm(image, identity, arm):
    verify_preregistration()
    require(arm in ('A', 'B'), 'invalid_arm')
    require(_OBJECTIVE_LOCK.acquire(blocking=False), 'objective_reentrancy_or_concurrency')
    original = calibration._score
    owner = threading.get_ident()
    started = time.perf_counter()
    def delegated_angular(*args, **kwargs):
        require(threading.get_ident() == owner, 'objective_cross_thread_call')
        return angular_score(*args, **kwargs)
    try:
        require(original is _PIXEL_SCORE and not cam._TRACE_LOCK.locked() and not border._ARM_LOCK.locked(),
                'objective_original_or_trace_owner_drift')
        if arm == 'B':
            calibration._score = delegated_angular
        expected = original if arm == 'A' else delegated_angular
        # BOTH arms run the identical accepted border treatment. The accepted
        # trace captures this scorer as its delegated original; no loop copy.
        support = border.trace_arm(image, identity, 'B')
        require(calibration._score is expected, 'objective_restoration_drift')
        border.verify_arm(support)
        return {'arm': arm, 'native_objective': 'pixel' if arm == 'A' else 'angular',
                'fixed_border_treatment': support, 'seconds': time.perf_counter()-started}
    finally:
        calibration._score = original
        _OBJECTIVE_LOCK.release()


def verify_arm(record):
    require(record['arm'] in ('A', 'B') and record['native_objective'] == ('pixel' if record['arm'] == 'A' else 'angular'),
            'objective_arm_binding_mismatch')
    support = record['fixed_border_treatment']
    require(support['arm'] == 'B', 'both_arms_must_use_fixed_filter')
    border.verify_arm(support)
    return support


@dataclass(frozen=True, slots=True)
class SealedPair:
    payload_json: str
    sha256: str


def seal_pair(scene, A, B):
    a, b = verify_arm(A), verify_arm(B)
    require(A['arm'] == 'A' and B['arm'] == 'B', 'pair_arm_order_mismatch')
    pinned = json.loads(PREREGISTRATION_JSON)
    require(a['output']['rgb_sha256'] == b['output']['rgb_sha256'] == pinned['rgb_sha256'][scene.identity]
            and a['output']['rgb_shape'] == b['output']['rgb_shape'], 'pair_rgb_mismatch')
    require(a['support'] == b['support'] and border.stage_array(a, 'enhance') == border.stage_array(b, 'enhance'),
            'pair_filter_or_upstream_mismatch')
    coarse = lambda s: [e['camera'] for e in s['output']['trace']['scores'] if e['site'] == 'coarse']
    require(coarse(a) == coarse(b), 'coarse_candidate_universe_changed')
    payload = canonical({'scene': asdict(scene), 'scene_sha256': scene.sha256,
        'preregistration_sha256': PREREGISTRATION_SHA256, 'arms': {'A': A, 'B': B}, 'loss': LOSS,
        'support_sha256': sha(canonical(a['support'])), 'production_changed': False})
    return SealedPair(payload, sha(payload))


def verify_pair(scene, sealed):
    require(sha(sealed.payload_json) == sealed.sha256, 'pair_seal_mismatch')
    value = json.loads(sealed.payload_json)
    require(value['scene'] == asdict(scene) and value['scene_sha256'] == scene.sha256
            and value['preregistration_sha256'] == PREREGISTRATION_SHA256 and value['loss'] == LOSS,
            'pair_scene_or_loss_binding_mismatch')
    require(seal_pair(scene, value['arms']['A'], value['arms']['B']) == sealed, 'pair_replay_mismatch')
    return value


def cross_score(lines, camera):
    h, w = camera['shape']
    lengths = np.hypot(lines[:, 2]-lines[:, 0], lines[:, 3]-lines[:, 1])
    points = calibration.vanishing_points(camera['focal'], camera['yaw'], camera['pitch'], w, h,
                                         camera['roll'], tuple(camera['principal']))
    all_residuals = calibration.segment_residuals(lines, points)
    residuals, owners = all_residuals.min(axis=1), all_residuals.argmin(axis=1)
    theta, contributions, valid = angular_terms(residuals, lengths)
    total = float(lengths.sum())
    pixel = float(_PIXEL_SCORE(lines, lengths, camera['focal'], camera['yaw'], camera['pitch'], w, h,
                              3., camera['roll'], tuple(camera['principal'])))
    angular = float(contributions.sum())
    pin, ain = residuals <= 3., valid & (theta <= TAU)
    return {'segment_scope': 'same_full_retained_lines', 'total_length': total, 'segments': len(lines),
        'pixel_score': pixel, 'angular_score': angular,
        'pixel_score_over_length': pixel/total if total else None,
        'angular_score_over_length': angular/total if total else None,
        'pixel_inliers': int(pin.sum()), 'angular_inliers': int(ain.sum()),
        'pixel_per_axis': [int(((owners == k) & pin).sum()) for k in range(3)],
        'angular_per_axis': [int(((owners == k) & ain).sum()) for k in range(3)],
        'pixel_residuals': residuals.tolist(), 'theta_radians': theta.tolist(),
        'pixel_result_field_semantics_unchanged': True, 'axis_veto': False}


def camera_from_prediction(prediction):
    h, w = prediction['height'], prediction['width']
    return {'shape': [h, w], 'focal': prediction['focal_pixels'], 'yaw': prediction['yaw_degrees'],
        'pitch': prediction['pitch_degrees'], 'roll': prediction['roll_degrees'],
        'principal': prediction['principal_point'] or [w/2., h/2.]}


def camera_from_event(event):
    c = event['camera']
    return {'shape': [c['height'], c['width']], 'focal': c['focal'], 'yaw': c['yaw'],
            'pitch': c['pitch'], 'roll': c['roll'],
            'principal': c['principal_point'] or [c['width']/2., c['height']/2.]}


def coarse_diagnostic(result, truth, retained):
    events = [e for e in result['trace']['scores'] if e['site'] == 'coarse']
    if not events:
        return {'scored': False, 'reason': 'support_refusal_before_objective'}
    distances = {'focal': lambda c: abs(c['focal']-truth['focal']),
                 'yaw': lambda c: abs((c['yaw']-truth['yaw']+45.) % 90.-45.),
                 'pitch': lambda c: abs(c['pitch']-truth['pitch'])}
    nearest = {key: min(distance(e['camera']) for e in events) for key, distance in distances.items()}
    tied = [e for e in events if all(abs(distance(e['camera'])-nearest[key]) <= 1e-8
                                   for key, distance in distances.items())]
    require(bool(tied), 'nearest_coarse_grid_empty')
    best = max(tied, key=lambda e: e['score'])
    rank = 1+sum(e['score'] > best['score'] for e in events)
    winner = max(events, key=lambda e: e['score'])
    return {'scored': True, 'candidate_count': len(events), 'nearest_truth_grid_ties': len(tied),
        'nearest_coordinate_distances': nearest, 'nearest_grid_rank_competition': rank,
        'nearest_grid_native_score': best['score'], 'coarse_winner_native_score': winner['score'],
        'native_score_scope': 'actual_coarse_subset', 'nearest_grid_camera': best['camera'],
        'nearest_grid_full_cross_scores': cross_score(retained, camera_from_event(best)),
        'coarse_winner_full_cross_scores': cross_score(retained, camera_from_event(winner)),
        'truth_seeded_inference': False, 'continuous_coverage_claim': False}


def evaluate_pair(scene, sealed):
    value = verify_pair(scene, sealed)
    truth = truth_camera(scene)  # No truth enters either trace; runner seals ALL pairs first.
    a = value['arms']['A']['fixed_border_treatment']
    retained = border.decode(a['support']['retained'])
    renderer = cross_score(retained, truth)
    arms = {}
    for arm, record in value['arms'].items():
        support = record['fixed_border_treatment']
        result, prediction = support['output'], support['output']['prediction']
        metrics = cam.camera_metrics(prediction, truth) if prediction is not None else None
        emitted = cross_score(retained, camera_from_prediction(prediction)) if prediction is not None else None
        native = 'pixel_score' if arm == 'A' else 'angular_score'
        arms[arm] = {'status': 'refused' if prediction is None else 'emitted', 'reason': result['reason'],
            'metrics': metrics, 'diagnostic_pass': bool(metrics is not None and metrics['diagnostic_pass']),
            'wrong': bool(metrics is not None and not metrics['diagnostic_pass']),
            'emitted_cross_scores': emitted, 'renderer_cross_scores': renderer,
            'renderer_minus_emitted_native_score': renderer[native]-emitted[native] if emitted else None,
            'raw_segments': len(border.decode(support['support']['raw'])), 'retained_segments': len(retained),
            'coarse': coarse_diagnostic(result, truth, retained),
            'stages': cam.stage_camera_evidence(result) if prediction is not None else None,
            'seconds': record['seconds'], 'trace_seconds': result['trace']['trace_seconds'],
            'score_delegate_seconds': result['trace']['score_delegate_seconds'],
            'result_inlier_fields': 'pixel threshold3; unchanged even for angular arm',
            'support_refusal_outside_loss_scope': prediction is None and len(retained) < 12,
            'calibrated_confidence': None}
    return {'scene': scene.identity, 'family': scene.family, 'stratum': scene.stratum, 'arms': arms,
        'B_minus_A': {'pass': int(arms['B']['diagnostic_pass'])-int(arms['A']['diagnostic_pass']),
            'wrong': int(arms['B']['wrong'])-int(arms['A']['wrong']),
            'seconds': arms['B']['seconds']-arms['A']['seconds']}}


def aggregate(values, predictions):
    require([v['scene'] for v in values] == [s.identity for s in SCENES], 'result_registry_mismatch')
    primary = [v for v in values if v['stratum'] == 'primary']
    stress = [v for v in values if v['stratum'] != 'primary']
    require(len(primary) == 12 and len({v['family'] for v in primary}) == 12, 'primary_denominator_mismatch')
    passes = {arm: sum(v['arms'][arm]['diagnostic_pass'] for v in primary) for arm in ('A', 'B')}
    corrected = [v['family'] for v in primary if v['arms']['A']['wrong'] and v['arms']['B']['diagnostic_pass']]
    lost = [v['family'] for v in primary if v['arms']['A']['diagnostic_pass'] and not v['arms']['B']['diagnostic_pass']]
    added = [v['scene'] for v in stress if v['arms']['B']['wrong'] and not v['arms']['A']['wrong']]
    stable = {s.family: {arm: predictions[s.identity][arm] == predictions[s.family+'-primary'][arm]
                          for arm in ('A', 'B')} for s in SCENES if s.stratum == 'gui-only'}
    gate = passes['B'] >= 9 and passes['B']-passes['A'] >= 3 and len(corrected) >= 3
    gate = gate and not lost and not added and all(all(x.values()) for x in stable.values())
    return {'primary_families': 12, 'primary_passes': passes, 'net_gain': passes['B']-passes['A'],
        'corrected_wrong_families': corrected, 'lost_A_pass': lost, 'added_stress_wrong': added,
        'gui_stability': stable, 'pilot_gate_pass': bool(gate),
        'counts': {stratum: {arm: {'n': len(rows), 'pass': sum(v['arms'][arm]['diagnostic_pass'] for v in rows),
            'wrong': sum(v['arms'][arm]['wrong'] for v in rows),
            'refused': sum(v['arms'][arm]['status'] == 'refused' for v in rows)} for arm in ('A', 'B')}
            for stratum, rows in (('primary', primary), ('correlated_stress', stress))},
        'refusal_is_accuracy_gain': False, 'integration_authorized': False}


def evidence_header():
    return {'kind': 'ANGULAR-OBJECTIVE-01-full-pairs-gzip-v1', 'preregistration_sha256': PREREGISTRATION_SHA256}


class EvidenceWriter:
    def __init__(self, path):
        self.path, self.artifact = path, None
        self.size, self.members, self.seals, self.digest = 0, [], [], hashlib.sha256()
    def __enter__(self):
        require(_WRITER_LOCK.acquire(blocking=False), 'writer_already_owned')
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
        require(self.artifact is not None and self.owner == threading.get_ident(), 'writer_not_owned')
        compressed = border.compress_member(decoded)
        require(self.size+len(compressed) <= MAX_EVIDENCE_BYTES, 'compressed_total_cap')
        self.artifact.seek(0, os.SEEK_END)
        require(self.artifact.tell() == self.size, 'writer_offset_mismatch')
        require(self.artifact.write(compressed) == len(compressed), 'short_write')
        self.artifact.flush()
        os.fsync(self.artifact.fileno())
        self.artifact.seek(self.size)
        restored = self.artifact.read(len(compressed))
        require(restored == compressed, 'member_readback_mismatch')
        plain, descriptor = border.decode_member(restored)
        require(plain == decoded, 'member_payload_mismatch')
        descriptor['offset'] = self.size
        self.members.append(descriptor)
        self.size += len(compressed)
        self.digest.update(compressed)
        return descriptor


def append_pair(scene, sealed, writer):
    require(len(writer.seals) < len(SCENES) and scene == SCENES[len(writer.seals)], 'member_scene_order')
    verify_pair(scene, sealed)
    descriptor = writer.append_record((canonical(asdict(sealed))+'\n').encode())
    writer.seals.append(sealed.sha256)
    return descriptor


def replay_evidence(expected_seals, expected_digest):
    require(EVIDENCE_PATH.is_file(), 'evidence_missing')
    require(EVIDENCE_PATH.stat().st_size <= MAX_EVIDENCE_BYTES and len(expected_seals) <= len(SCENES), 'evidence_cap_or_count')
    digest = hashlib.sha256()
    with EVIDENCE_PATH.open('rb') as artifact:
        members = border.gzip_members(artifact, digest)
        header = next(members, None)
        require(header is not None and border.json_line(header[0]) == evidence_header(), 'header_mismatch')
        for scene, expected in zip(SCENES, expected_seals):
            member = next(members, None)
            require(member is not None, 'member_missing')
            sealed = SealedPair(**asdict(border.pair_member(member[0])))
            require(sealed.sha256 == expected, 'member_seal_order')
            verify_pair(scene, sealed)
        require(next(members, None) is None, 'extra_member')
    require(digest.hexdigest() == expected_digest, 'artifact_digest_mismatch')
    return digest.hexdigest()


def read_pair(descriptor):
    require(EVIDENCE_PATH.is_file() and 0 <= descriptor['offset']
            and 0 < descriptor['compressed_bytes'] <= CAPS['max_compressed_member_bytes'], 'member_location_mismatch')
    with EVIDENCE_PATH.open('rb') as artifact:
        artifact.seek(descriptor['offset'])
        raw = artifact.read(descriptor['compressed_bytes'])
    require(sha(raw) == descriptor['compressed_sha256'], 'member_compressed_digest')
    plain, found = border.decode_member(raw)
    require(found['decoded_sha256'] == descriptor['decoded_sha256'], 'member_decoded_digest')
    return SealedPair(**asdict(border.pair_member(plain)))


def artifact_digest():
    require(EVIDENCE_PATH.is_file() and EVIDENCE_PATH.stat().st_size <= MAX_EVIDENCE_BYTES, 'evidence_missing_or_cap')
    digest, size = hashlib.sha256(), 0
    with EVIDENCE_PATH.open('rb') as artifact:
        for chunk in iter(lambda: artifact.read(65_536), b''):
            size += len(chunk)
            require(size <= MAX_EVIDENCE_BYTES, 'compressed_total_cap')
            digest.update(chunk)
    return digest.hexdigest()


def claim_report():
    require(REPORT_PATH.is_file(), 'report_missing')
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
            require(content.count('ANGULAR_EXECUTION_STATE: NOT_STARTED') == 1 and PREREGISTRATION_SHA256 in content,
                    'already_claimed_or_pin_mismatch')
            require(read_report_payload(content) == json.loads(PREREGISTRATION_JSON), 'report_payload_mismatch')
            report.seek(0)
            report.write(content.replace('ANGULAR_EXECUTION_STATE: NOT_STARTED', 'ANGULAR_EXECUTION_STATE: RUNNING'))
            report.truncate()
            report.flush()
            os.fsync(report.fileno())
        finally:
            unlock()


def run_study():
    require(_STUDY_LOCK.acquire(blocking=False), 'study_already_owned')
    try:
        verify_preregistration()
        require(not EVIDENCE_PATH.exists(), 'evidence_exists')
        claim_report()
        with EvidenceWriter(EVIDENCE_PATH) as writer:
            for scene in SCENES:
                rgb = render_rgb(scene)
                A, B = trace_arm(rgb, scene.identity, 'A'), trace_arm(rgb, scene.identity, 'B')
                append_pair(scene, seal_pair(scene, A, B), writer)
            seals, descriptors, expected_digest = tuple(writer.seals), tuple(writer.members[1:]), writer.digest.hexdigest()
        digest = replay_evidence(seals, expected_digest)  # ALL outputs durable BEFORE any evaluator.
        values, predictions = [], {}
        for scene, expected, descriptor in zip(SCENES, seals, descriptors):
            sealed = read_pair(descriptor)
            require(sealed.sha256 == expected, 'member_seal_order')
            record = verify_pair(scene, sealed)
            predictions[scene.identity] = {arm: v['fixed_border_treatment']['output']['prediction'] for arm, v in record['arms'].items()}
            values.append(evaluate_pair(scene, sealed))
        require(artifact_digest() == digest, 'artifact_changed_during_evaluation')
        result = {'preregistration_sha256': PREREGISTRATION_SHA256, 'evidence_sha256': digest,
            'sealed_pairs': seals, 'members': descriptors, 'scenes': values, 'aggregate': aggregate(values, predictions),
            'production_changed': False, 'consumed_prior_holdout_used': False}
        with REPORT_PATH.open('r+', encoding='utf-8', newline='') as report:
            content = report.read()
            require(content.count('ANGULAR_EXECUTION_STATE: RUNNING') == 1, 'run_marker_drift')
            report.seek(0)
            report.write(content.replace('ANGULAR_EXECUTION_STATE: RUNNING', 'ANGULAR_EXECUTION_STATE: COMPLETE')+
                         '\n\nDurable registered result:\n\n```json\n'+canonical(result)+'\n```\n')
            report.truncate()
            report.flush()
            os.fsync(report.fileno())
        return result
    finally:
        _STUDY_LOCK.release()


def dependency_pins():
    result = border.dependency_pins()
    result['source_sha256']['scripts/evaluation/run_angular_objective.py'] = sha(Path(__file__).read_bytes())
    return result


def freeze_preregistration():
    border.verify_preregistration()
    old = json.loads(border.PREREGISTRATION_JSON)['rgb_sha256'].values()
    consumed = set(old) | set(json.loads(cam.PREREGISTRATION_JSON)['stimulus_sha256'].values())
    hashes = {s.identity: sha(render_rgb(s).tobytes()) for s in SCENES}
    require(len(SCENES) == 36 and len(set(hashes.values())) == 36 and not (set(hashes.values()) & consumed),
            'new_RGB_collision_or_consumed_reuse')
    truths = {s.identity: sha(canonical(truth_camera(s))) for s in SCENES}
    geometry = {s.identity: sha(canonical(border.new_faces(json.loads(s.recipe_json)))) for s in SCENES}
    return canonical({'version': 'ANGULAR-OBJECTIVE-01-prereg-v1', 'scenes': [asdict(s) for s in SCENES],
        'dependencies': dependency_pins(), 'rgb_sha256': hashes, 'camera_truth_sha256': truths,
        'geometry_truth_sha256': geometry, 'consumed_RGB_set_sha256': sha(canonical(sorted(consumed))),
        'filter': border.FILTER_RULE, 'loss': LOSS, 'caps': CAPS, 'trace_policy': asdict(cam.TracePolicy()),
        'max_evidence_bytes': MAX_EVIDENCE_BYTES, 'gate': GATE, 'primary_n': 12, 'correlated_stress_n': 24,
        'evidence_path': 'docs/reports/'+EVIDENCE_PATH.name,
        'evidence': 'independent gzip JSONL members;512MiB total/16MiB member/200MB decoded;fsync/readback;one-full-replay-before-truth;final-digest',
        'one_shot': 'locked-fsynced-report-NOT_STARTED-to-RUNNING-before-inference/exclusive-create;no-retry',
        'diagnostic_tolerances': {'focal_relative': .05, 'orientation_degrees_mod90': 1., 'principal_diagonal': .01},
        'truth_order': 'all36pairs-and-traces-durable-before-any-cross-score/metric/nearest-truth-grid-rank',
        'coarse_rank': 'minimum coordinate distance per f/yawmod90/pitch;tied nearest max native score;competition rank;no truth-seeded fit',
        'result_inliers': 'unchanged pixel threshold3 fields;angular support separately post-seal',
        'support_refusals': 'same retained set;<12 is outside loss intervention;no relaxation',
        'solver_changes': False, 'axis_or_confidence_gate': False, 'production_changed': False,
        'prior_holdout': 'CAM-MD/GUI-BORDER consumed;no new inference/evaluation/tuning of prior images'})


def read_report_payload(content):
    begin, end = 'PREREGISTRATION_BEGIN\n```json\n', '\n```\nPREREGISTRATION_END'
    require(content.count(begin) == content.count(end) == 1, 'report_payload_missing_or_ambiguous')
    return json.loads(content.split(begin)[1].split(end)[0])


def initialize_preregistration(payload=None):
    global PREREGISTRATION_JSON, PREREGISTRATION_SHA256
    if payload is None:
        require(REPORT_PATH.is_file(), 'report_missing')
        content = REPORT_PATH.read_text(encoding='utf-8')
        value = read_report_payload(content)
        require('Preregistration SHA256: `'+sha(canonical(value))+'`.' in content, 'report_digest_mismatch')
    else:
        value = json.loads(payload)
    PREREGISTRATION_JSON, PREREGISTRATION_SHA256 = canonical(value), sha(canonical(value))


def verify_preregistration():
    if PREREGISTRATION_JSON is None:
        initialize_preregistration()
    pinned = json.loads(PREREGISTRATION_JSON)
    require(dependency_pins() == pinned['dependencies'], 'dependency_or_runtime_drift')
    require([asdict(s) for s in SCENES] == pinned['scenes'] and LOSS == pinned['loss'] and TAU == pinned['loss']['tau_radians']
            and GATE == pinned['gate'] and CAPS == pinned['caps'] == border.CAPS
            and MAX_EVIDENCE_BYTES == pinned['max_evidence_bytes'] == border.MAX_EVIDENCE_BYTES
            and border.FILTER_RULE == pinned['filter'] and asdict(cam.TracePolicy()) == pinned['trace_policy'],
            'registry_loss_filter_or_cap_drift')


if __name__ == '__main__':
    print(canonical(run_study()))
