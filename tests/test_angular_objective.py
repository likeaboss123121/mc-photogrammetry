"""No-fit objective algebra, stimulus, stub call-through and memory-only QA."""
from copy import deepcopy
from dataclasses import asdict, replace
import ast
import gzip
import io
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import run_angular_objective as study


@pytest.fixture(scope='session', autouse=True)
def explicit_pure_stimulus_setup():
    study.initialize_preregistration(study.freeze_preregistration())


def test_fixed_theory_tau_and_no_sweep():
    assert study.TAU == math.asin(6/120)
    assert math.degrees(study.TAU) == pytest.approx(2.8659839825988622)
    assert study.LOSS['threshold_sweep'] is False
    assert study.LOSS['pixel_threshold'] == 3.


@pytest.mark.parametrize('length', [1., 40., 120., 240., 10000.])
@pytest.mark.parametrize('fraction', [0., .25, .75, 1., 1.01])
def test_same_angle_normalized_score_independent_of_length(length, fraction):
    angle = fraction*study.TAU
    theta, weighted, valid = study.angular_terms(np.array([length/2*math.sin(angle)]), np.array([length]))
    assert theta[0] == pytest.approx(angle)
    assert weighted[0]/length == pytest.approx(max(1-fraction, 0), abs=1e-14)
    assert valid.tolist() == [True]


def test_clipping_zero_length_and_reference_threshold():
    theta, score, valid = study.angular_terms(np.array([-1., 1., 0., 3.]), np.array([2., 1., 0., 120.]))
    assert theta.tolist() == [0., math.pi/2, 0., study.TAU]
    assert score.tolist() == [2., 0., 0., 0.]
    assert valid.tolist() == [True, True, False, True]
    assert study.angular_terms(np.array([]), np.array([]))[1].sum() == 0


@pytest.mark.parametrize('r,L', [([0], [-1]), ([float('nan')], [1]), ([1], [float('inf')]), ([1, 2], [1])])
def test_invalid_angular_inputs_fail_closed(r, L):
    with pytest.raises(study.AngularStudyError, match='invalid_angular_arrays'):
        study.angular_terms(np.array(r), np.array(L))


@pytest.mark.parametrize('infinite', [True, False])
def test_score_uses_existing_finite_and_infinite_geometry_without_optimization(monkeypatch, infinite):
    lines = np.array([[10., 20., 110., 20.], [10., 30., 110., 40.]], np.float64)
    lengths = np.hypot(lines[:, 2]-lines[:, 0], lines[:, 3]-lines[:, 1])
    points = np.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]) if infinite else np.array([[210., 20., 1.], [60., 200., 1.], [-400., -400., 1.]])
    monkeypatch.setattr(study.calibration, 'vanishing_points', lambda *args: points)
    residuals = study.calibration.segment_residuals(lines, points).min(axis=1)
    expected = study.angular_terms(residuals, lengths)[1].sum()
    assert study.angular_score(lines, lengths, 100., 20., 10., 200, 160, 3.) == expected
    assert expected >= lengths[0]
    assert study.angular_score(lines[:0], lengths[:0], 100., 20., 10., 200, 160, 3.) == 0


@pytest.mark.parametrize('change', ['threshold', 'weight', 'nan', 'shape'])
def test_scorer_rejects_contract_drift(change):
    lines = np.array([[0., 0., 120., 0.]], np.float32)
    lengths = np.array([120.], np.float32)
    threshold = 3.
    if change == 'threshold': threshold = 4.
    if change == 'weight': lengths[0] = 119.
    if change == 'nan': lines[0, 0] = np.nan
    if change == 'shape': lines = lines.reshape(2, 2)
    with pytest.raises(study.AngularStudyError):
        study.angular_score(lines, lengths, 100., 0., 0., 200, 160, threshold)


def test_singular_midpoint_vp_and_zero_length_match_existing_geometry(monkeypatch):
    lines = np.array([[0., 0., 120., 0.], [60., 0., 60., 0.]], np.float64)
    lengths = np.array([120., 0.])
    points = np.array([[60., 0., 1.], [0., 1., 0.], [1., 1., 0.]])
    monkeypatch.setattr(study.calibration, 'vanishing_points', lambda *args: points)
    residuals = study.calibration.segment_residuals(lines, points)
    assert residuals.min(axis=1).tolist() == [0., 0.]
    assert study.angular_score(lines, lengths, 100., 0., 0., 200, 160, 3.) == 120.
    # The accepted helper defines the singular midpoint direction through its
    # epsilon denominator. The experiment keeps that behavior and adds no veto.
    assert study._PIXEL_SCORE(lines, lengths, 100., 0., 0., 200, 160, 3.) == 120.


def test_stimulus_freeze_reproducible_fresh_and_recipe_bound():
    pin = json.loads(study.freeze_preregistration())
    assert study.canonical(pin) == study.PREREGISTRATION_JSON
    assert len(study.SCENES) == len(set(pin['rgb_sha256'].values())) == 36
    primary = [json.loads(s.recipe_json) for s in study.SCENES if s.stratum == 'primary']
    assert len(primary) == 12 and any(r['pitch'] > 0 for r in primary) and any(r['pitch'] < 0 for r in primary)
    assert len({tuple(r['shape']) for r in primary}) == 4
    assert len({r['vertical_fov'] for r in primary}) == 4
    assert all(r['native_side'] == 16 for r in primary)
    old = set(json.loads(study.border.PREREGISTRATION_JSON)['rgb_sha256'].values())
    old |= set(json.loads(study.cam.PREREGISTRATION_JSON)['stimulus_sha256'].values())
    assert not old.intersection(pin['rgb_sha256'].values())
    for scene in study.SCENES:
        assert pin['camera_truth_sha256'][scene.identity] == study.sha(study.canonical(study.truth_camera(scene)))
        assert pin['geometry_truth_sha256'][scene.identity] == study.sha(study.canonical(study.border.new_faces(json.loads(scene.recipe_json))))


def test_pure_render_crop_and_gui_only_unmasked_bytes():
    primary, crop, gui = study.SCENES[:3]
    first, cropped, contaminated = map(study.render_rgb, (primary, crop, gui))
    recipe = json.loads(crop.recipe_json)
    x, y, w, h = recipe['crop']
    assert np.array_equal(cropped, first[y:y+h, x:x+w])
    mask = study.cam.gui_mask.create_gui_mask(*first.shape[:2])
    assert np.array_equal(first[~mask], contaminated[~mask])
    assert not np.array_equal(first[mask], contaminated[mask])
    assert study.truth_camera(crop)['principal'] == [study.truth_camera(primary)['principal'][0]-x, study.truth_camera(primary)['principal'][1]-y]


def test_collision_rejected_before_any_fit(monkeypatch):
    monkeypatch.setattr(study, 'render_rgb', lambda *_: np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr(study, 'trace_arm', lambda *_: pytest.fail('fit forbidden'))
    with pytest.raises(study.AngularStudyError, match='collision'):
        study.freeze_preregistration()


@pytest.fixture
def stub_entry(monkeypatch):
    """Accepted RGB/mask/enhance/filter flow; detector and camera never optimize."""
    image = np.full((80, 100, 3), 123, np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[:, 40:] = True
    raw = np.array([[5., 10.+i, 30., 10.+i] for i in range(12)]+
                   [[39., 15., 39., 55.], [40., 20., 40., 60.]], np.float32)
    calls = {'mask': [], 'detector': [], 'selected': []}
    def mask_original(*args, **kwargs):
        calls['mask'].append((args, kwargs))
        return mask
    def detector_original(*args, **kwargs):
        calls['detector'].append((args, kwargs))
        return raw
    def camera_original(grey, **kwargs):
        enhanced = study.calibration.enhance(grey)
        selected = study.calibration.detect_segments(enhanced)
        calls['selected'].append(selected)
        if len(selected) < 12:
            return None
        return study.calibration.CalibrationResult(100., 0., 0., 100, 80, len(selected), len(selected),
            0., (len(selected), 0, 0), 0., None, ())
    monkeypatch.setattr(study.cam.gui_mask, 'create_gui_mask', mask_original)
    monkeypatch.setattr(study.calibration, 'detect_segments', detector_original)
    monkeypatch.setattr(study.calibration, 'estimate_camera', camera_original)
    monkeypatch.setattr(study.cam, '_PINNED_CAMERA_CODE', camera_original.__code__)
    monkeypatch.setattr(study, 'verify_preregistration', lambda: None)
    scene = replace(study.SCENES[0], identity='stub-angular', renderer='stub', recipe_json='{}')
    pinned = json.loads(study.PREREGISTRATION_JSON)
    pinned['rgb_sha256'][scene.identity] = study.sha(image.tobytes())
    monkeypatch.setattr(study, 'PREREGISTRATION_JSON', study.canonical(pinned))
    return SimpleNamespace(image=image, mask=mask, raw=raw, calls=calls, scene=scene,
        detector=detector_original, mask_function=mask_original, camera=camera_original)


def stub_pair(f):
    A = study.trace_arm(f.image, f.scene.identity, 'A')
    B = study.trace_arm(f.image, f.scene.identity, 'B')
    return A, B, study.seal_pair(f.scene, A, B)


def test_identical_filter_native_order_original_input_and_pixel_result_fields(stub_entry):
    f = stub_entry
    before = [x.tobytes() for x in (f.image, f.raw, f.mask)]
    A, B, sealed = stub_pair(f)
    assert len(f.calls['mask']) == len(f.calls['detector']) == 2
    assert all(np.array_equal(s, f.raw[:12]) and s is not f.raw for s in f.calls['selected'])
    a, b = (r['fixed_border_treatment'] for r in (A, B))
    assert a['support'] == b['support'] and a['support']['indices'] == list(range(12))
    assert study.border.stage_array(a, 'enhance') == study.border.stage_array(b, 'enhance')
    assert a['output']['prediction'] == b['output']['prediction']
    assert b['output']['prediction']['per_axis_inliers'] == [12, 0, 0]
    assert before == [x.tobytes() for x in (f.image, f.raw, f.mask)]
    assert A['native_objective'] == 'pixel' and B['native_objective'] == 'angular'
    assert study.verify_pair(f.scene, sealed)['production_changed'] is False
    assert study.calibration._score is study._PIXEL_SCORE


def test_original_pixel_and_angular_delegate_only_intervention(monkeypatch, stub_entry):
    f = stub_entry
    base = study.trace_arm(f.image, f.scene.identity, 'A')['fixed_border_treatment']
    lines = np.array([[0., 0., 120., 0.]], np.float32)
    lengths = np.array([120.], np.float32)
    args = (lines, lengths, 100., 20., 10., 200, 160, 3.)
    delegated = []
    def trace(image, identity, treatment):
        assert treatment == 'B'
        delegate = study.calibration._score
        delegated.append((delegate, delegate(*args)))
        return deepcopy(base)
    monkeypatch.setattr(study.border, 'trace_arm', trace)
    study.trace_arm(f.image, f.scene.identity, 'A')
    study.trace_arm(f.image, f.scene.identity, 'B')
    assert delegated[0][0] is study._PIXEL_SCORE
    assert delegated[0][1] == study._PIXEL_SCORE(*args)
    assert delegated[1][0] is not study._PIXEL_SCORE
    assert delegated[1][1] == study.angular_score(*args)
    assert study.calibration._score is study._PIXEL_SCORE


def test_support_refusal_both_arms_before_any_score(stub_entry):
    stub_entry.raw[:, [0, 2]] = 39.
    for arm in ('A', 'B'):
        record = study.trace_arm(stub_entry.image, 'qa', arm)['fixed_border_treatment']
        assert record['output']['prediction'] is None
        assert record['output']['trace']['scores'] == []
        assert record['support']['indices'] == []
    assert study.calibration._score is study._PIXEL_SCORE


def test_gui_only_mutation_is_prediction_stable_in_each_arm(stub_entry):
    f = stub_entry
    changed = f.image.copy()
    changed[f.mask] = [9, 199, 71]
    assert not np.array_equal(changed, f.image)
    for arm in ('A', 'B'):
        old = study.trace_arm(f.image, 'qa', arm)['fixed_border_treatment']
        new = study.trace_arm(changed, 'qa', arm)['fixed_border_treatment']
        assert old['output']['prediction'] == new['output']['prediction']
        assert old['support'] == new['support']
        assert study.border.stage_array(old, 'enhance') == study.border.stage_array(new, 'enhance')


@pytest.mark.parametrize('failure', [RuntimeError('unexpected'), KeyboardInterrupt(), study.AngularStudyError('cap')])
def test_objective_restores_on_all_exception_classes(monkeypatch, stub_entry, failure):
    def broken(*args):
        assert study.calibration._score is not study._PIXEL_SCORE
        raise failure
    monkeypatch.setattr(study.border, 'trace_arm', broken)
    with pytest.raises(type(failure)):
        study.trace_arm(stub_entry.image, 'qa', 'B')
    assert study.calibration._score is study._PIXEL_SCORE and not study._OBJECTIVE_LOCK.locked()


def test_nested_and_cross_thread_calls_refuse_and_restore(monkeypatch, stub_entry):
    base = study.trace_arm(stub_entry.image, 'qa', 'A')['fixed_border_treatment']
    failures = []
    def trace(*args):
        with pytest.raises(study.AngularStudyError, match='reentrancy'):
            study.trace_arm(stub_entry.image, 'qa', 'B')
        delegate = study.calibration._score
        def concurrent():
            try: delegate()
            except study.AngularStudyError as exc: failures.append(exc.reason)
        thread = threading.Thread(target=concurrent)
        thread.start(); thread.join()
        return base
    monkeypatch.setattr(study.border, 'trace_arm', trace)
    study.trace_arm(stub_entry.image, 'qa', 'B')
    assert failures == ['objective_cross_thread_call']
    assert study.calibration._score is study._PIXEL_SCORE


@pytest.mark.parametrize('field', ['rgb', 'support', 'coarse', 'seal'])
def test_pair_tamper_fails_before_truth(monkeypatch, stub_entry, field):
    A, B, sealed = stub_pair(stub_entry)
    monkeypatch.setattr(study, 'truth_camera', lambda *_: pytest.fail('truth accessed'))
    if field == 'seal':
        with pytest.raises(study.AngularStudyError, match='seal'):
            study.verify_pair(stub_entry.scene, replace(sealed, payload_json=sealed.payload_json+' '))
        return
    if field == 'rgb': B['fixed_border_treatment']['output']['rgb_sha256'] = '0'*64
    if field == 'support': B['fixed_border_treatment']['support']['indices'].reverse()
    if field == 'coarse':
        monkeypatch.setattr(study.border, 'verify_arm', lambda *_: None)
        B['fixed_border_treatment']['output']['trace']['scores'].append({'site': 'coarse', 'camera': {}})
    with pytest.raises((study.AngularStudyError, study.border.BorderStudyError)):
        study.seal_pair(stub_entry.scene, A, B)


def event(focal, yaw, pitch, score):
    return {'site': 'coarse', 'score': score, 'camera': {'focal': focal, 'yaw': yaw, 'pitch': pitch,
        'width': 200, 'height': 160, 'threshold': 3., 'roll': 0., 'principal_point': None}}


def test_actual_coarse_native_rank_ties_mod90_and_full_cross_score(monkeypatch):
    lines = np.array([[0., 0., 120., 0.]], np.float32)
    calls = []
    def score(support, camera):
        assert support is lines
        calls.append(camera)
        return {'stub_full_score': True}
    monkeypatch.setattr(study, 'cross_score', score)
    events = [event(100, 0, -10, 7), event(100, 45, -10, 9), event(200, 0, -10, 8),
              event(200, 45, -10, 6)]
    truth = {'focal': 150., 'yaw': 90., 'pitch': -10.}
    value = study.coarse_diagnostic({'trace': {'scores': events}}, truth, lines)
    assert value['nearest_truth_grid_ties'] == 2 and value['nearest_grid_rank_competition'] == 2
    assert value['nearest_grid_native_score'] == 8 and value['coarse_winner_native_score'] == 9
    assert value['native_score_scope'] == 'actual_coarse_subset'
    assert len(calls) == 2 and value['truth_seeded_inference'] is False
    assert study.coarse_diagnostic({'trace': {'scores': []}}, truth, lines)['scored'] is False


def test_candidate_max_not_retained_and_actual_selected_transition_semantics():
    rescore = event(100, 20, -10, 20)
    rescore.update(site='coarse-full-rescore', winner={'best_before': [9., 100, 20, -10]})
    base = event(110, 21, -11, 23); base['site'] = 'refine-base'
    rejected = event(111, 22, -12, 23.1); rejected['site'] = 'refine-principal'
    pred = {'focal_pixels': 110., 'yaw_degrees': 21., 'pitch_degrees': -11.}
    result = {'status': 'returned', 'prediction': pred, 'trace': {'trace_complete': True,
        'scores': [event(100, 20, -10, 9), rescore, base, rejected],
        'stages': [{'stage': 'refine_optics', 'output': [110., 0., None, 23., False, []]}]}}
    values = study.cam.stage_camera_evidence(result)
    assert values['candidate_max']['refine-principal']['camera']['focal'] == 111
    assert values['candidate_max']['refine-principal']['optimizer_retention_claim'] is False
    transitions = values['selected_transitions']
    assert transitions['coarse_winner_to_full_rescore']['coarse_native_score'] == 9
    assert transitions['coarse_winner_to_full_rescore']['full_rescore'] == 20
    assert transitions['fine_winner_to_refinement']['selected_fine_camera']['focal'] == 110
    assert transitions['fine_winner_to_refinement']['refinement_accepted'] is False
    assert transitions['final_prediction'] is pred


def test_analytic_cross_scores_same_full_lines_labels_no_axis_veto(monkeypatch):
    lines = np.array([[0., 0., 120., 0.], [0., 0., 240., 12.]], np.float64)
    camera = {'focal': 100., 'yaw': 0., 'pitch': 0., 'shape': [160, 200], 'roll': 0., 'principal': [100., 80.]}
    monkeypatch.setattr(study.calibration, 'vanishing_points', lambda *args: np.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]))
    value = study.cross_score(lines, camera)
    assert value['segment_scope'] == 'same_full_retained_lines' and value['segments'] == 2
    assert value['pixel_inliers'] == 1 and value['angular_inliers'] == 2
    assert value['pixel_result_field_semantics_unchanged'] and value['axis_veto'] is False
    assert value['angular_score'] > value['pixel_score']


class MemoryArtifact(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def fileno(self): return 777


class MemoryPath:
    def __init__(self, raw=None):
        self.exists_flag = raw is not None
        self.artifact = MemoryArtifact(raw or b'')
    def exists(self): return self.exists_flag
    def is_file(self): return self.exists_flag
    def stat(self): return SimpleNamespace(st_size=len(self.artifact.getvalue()))
    def open(self, mode):
        if mode == 'x+b':
            if self.exists_flag: raise FileExistsError()
            self.exists_flag = True
        elif not self.exists_flag: raise FileNotFoundError()
        self.artifact.seek(0)
        return self.artifact


class TextArtifact(io.StringIO):
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def fileno(self): return 111


class MemoryReport:
    def __init__(self, content): self.artifact = TextArtifact(content)
    def is_file(self): return True
    def read_text(self, **kwargs): return self.artifact.getvalue()
    def open(self, *args, **kwargs):
        self.artifact.seek(0)
        return self.artifact


def report_content(state='NOT_STARTED'):
    return ('ANGULAR_EXECUTION_STATE: '+state+'\nPreregistration SHA256: `'+study.PREREGISTRATION_SHA256+'`.\n'+
        '\nPREREGISTRATION_BEGIN\n```json\n'+study.PREREGISTRATION_JSON+'\n```\nPREREGISTRATION_END\n')


def memory_pair(monkeypatch, f, count=1):
    A, B, pair = stub_pair(f)
    scenes, pairs = [f.scene], [pair]
    if count == 2:
        second = replace(f.scene, identity='stub-second')
        pin = json.loads(study.PREREGISTRATION_JSON)
        pin['rgb_sha256'][second.identity] = pin['rgb_sha256'][f.scene.identity]
        monkeypatch.setattr(study, 'PREREGISTRATION_JSON', study.canonical(pin))
        scenes.append(second)
        pairs.append(study.seal_pair(second, A, B))
    monkeypatch.setattr(study, 'SCENES', tuple(scenes))
    path = MemoryPath()
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    with study.EvidenceWriter(path) as writer:
        for scene, sealed in zip(scenes, pairs): study.append_pair(scene, sealed, writer)
        descriptors = deepcopy(writer.members[1:])
        digest = writer.digest.hexdigest()
    return path, pairs, descriptors, digest


def test_independent_multi_member_gzip_durable_replay_and_random_read(monkeypatch, stub_entry):
    path, pairs, descriptors, digest = memory_pair(monkeypatch, stub_entry, 2)
    assert len(gzip.decompress(path.artifact.getvalue()).splitlines()) == 3
    assert study.replay_evidence(tuple(p.sha256 for p in pairs), digest) == digest
    assert [study.read_pair(d) for d in descriptors] == pairs
    assert study.artifact_digest() == digest


@pytest.mark.parametrize('fault', ['trailer', 'crc', 'junk', 'empty', 'extra', 'order', 'digest', 'prefix'])
def test_gzip_faults_fail_closed_prefix_preserved(monkeypatch, stub_entry, fault):
    path, pairs, descriptors, digest = memory_pair(monkeypatch, stub_entry, 2)
    raw = path.artifact.getvalue()
    expected = digest
    if fault == 'trailer': changed = raw[:-1]
    if fault == 'crc':
        changed = bytearray(raw); changed[-8] ^= 1; changed = bytes(changed)
    if fault == 'junk': changed = raw+b'junk'
    if fault == 'empty': changed = raw+study.border.compress_member(b'')
    if fault == 'extra': changed = raw+raw[descriptors[1]['offset']:]
    if fault == 'order':
        changed = raw[:descriptors[0]['offset']]+raw[descriptors[1]['offset']:]+raw[descriptors[0]['offset']:descriptors[1]['offset']]
    if fault == 'digest': changed, expected = raw, '0'*64
    if fault == 'prefix':
        changed = bytearray(raw); changed[20] ^= 1; changed = bytes(changed)
    path.artifact = MemoryArtifact(changed)
    with pytest.raises((study.AngularStudyError, study.border.BorderStudyError)):
        study.replay_evidence(tuple(p.sha256 for p in pairs), expected)
    assert path.artifact.getvalue() == changed
    if fault == 'trailer': assert changed[:descriptors[1]['offset']] == raw[:descriptors[1]['offset']]


@pytest.mark.parametrize('scope', ['total', 'member', 'decoded'])
def test_evidence_cap_boundary_exact_allowed_one_less_refuses(monkeypatch, stub_entry, scope):
    path, pairs, descriptors, digest = memory_pair(monkeypatch, stub_entry)
    raw = path.artifact.getvalue()
    decoded = gzip.decompress(raw).splitlines(keepends=True)
    caps = dict(study.border.CAPS)
    if scope == 'total':
        limit = len(raw)
        monkeypatch.setattr(study, 'MAX_EVIDENCE_BYTES', limit)
        monkeypatch.setattr(study.border, 'MAX_EVIDENCE_BYTES', limit)
    else:
        key = 'max_compressed_member_bytes' if scope == 'member' else 'max_record_bytes'
        limit = max(d['compressed_bytes'] for d in [*descriptors]) if scope == 'member' else max(map(len, decoded))
        caps[key] = limit
        monkeypatch.setattr(study.border, 'CAPS', caps)
    assert study.replay_evidence((pairs[0].sha256,), digest) == digest
    if scope == 'total':
        monkeypatch.setattr(study, 'MAX_EVIDENCE_BYTES', limit-1)
        monkeypatch.setattr(study.border, 'MAX_EVIDENCE_BYTES', limit-1)
    else: caps[key] = limit-1
    with pytest.raises((study.AngularStudyError, study.border.BorderStudyError)):
        study.replay_evidence((pairs[0].sha256,), digest)


def test_append_cap_preserves_prefix_and_writer_ownership(monkeypatch, stub_entry):
    _, _, pair = stub_pair(stub_entry)
    monkeypatch.setattr(study, 'SCENES', (stub_entry.scene,))
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    path = MemoryPath()
    with study.EvidenceWriter(path) as writer:
        prefix = path.artifact.getvalue()
        monkeypatch.setattr(study, 'MAX_EVIDENCE_BYTES', len(prefix))
        with pytest.raises(study.AngularStudyError, match='total_cap'):
            study.append_pair(stub_entry.scene, pair, writer)
        assert path.artifact.getvalue() == prefix and not writer.seals
        with pytest.raises(study.AngularStudyError, match='already_owned'):
            with study.EvidenceWriter(MemoryPath()): pass
    assert not study._WRITER_LOCK.locked()


def test_partial_write_preserves_prior_members_without_repair(monkeypatch, stub_entry):
    _, _, pair = stub_pair(stub_entry)
    monkeypatch.setattr(study, 'SCENES', (stub_entry.scene,))
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    path = MemoryPath()
    with study.EvidenceWriter(path) as writer:
        prefix = path.artifact.getvalue()
        original = path.artifact.write
        monkeypatch.setattr(path.artifact, 'write', lambda value: original(value[:5]))
        with pytest.raises(study.AngularStudyError, match='short_write'):
            study.append_pair(stub_entry.scene, pair, writer)
        assert path.artifact.getvalue().startswith(prefix)
        assert len(path.artifact.getvalue()) == len(prefix)+5 and writer.seals == []
    assert not study._WRITER_LOCK.locked()


def test_writer_offset_and_readback_fail_closed(monkeypatch):
    path = MemoryPath()
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    with study.EvidenceWriter(path) as writer:
        path.artifact.seek(0, 2)
        path.artifact.write(b'foreign writer')
        with pytest.raises(study.AngularStudyError, match='offset_mismatch'):
            writer.append_record(b'{}\n')
    assert not study._WRITER_LOCK.locked()


def configure_memory_run(monkeypatch, f):
    A, B, sealed = stub_pair(f)
    monkeypatch.setattr(study, 'SCENES', (f.scene,))
    path, report = MemoryPath(), MemoryReport(report_content())
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    monkeypatch.setattr(study, 'REPORT_PATH', report)
    if study.os.name == 'nt':
        import msvcrt
        monkeypatch.setattr(msvcrt, 'locking', lambda *_: None)
    monkeypatch.setattr(study, 'render_rgb', lambda *_: f.image)
    monkeypatch.setattr(study, 'trace_arm', lambda image, identity, arm: A if arm == 'A' else B)
    return path, report, sealed


def test_complete_only_after_all_seals_fsync_replay_and_final_digest(monkeypatch, stub_entry):
    path, report, sealed = configure_memory_run(monkeypatch, stub_entry)
    syncs, replays, evaluations = [], [], []
    monkeypatch.setattr(study.os, 'fsync', lambda fd: syncs.append(fd))
    original = study.replay_evidence
    def replay(*args):
        replays.append('bounded-full-replay')
        return original(*args)
    monkeypatch.setattr(study, 'replay_evidence', replay)
    def evaluate(scene, pair):
        assert replays == ['bounded-full-replay'] and syncs == [111, 777, 777]
        assert 'ANGULAR_EXECUTION_STATE: RUNNING' in report.artifact.getvalue()
        assert pair == sealed
        evaluations.append('stub-no-truth-or-metric')
        return {'stub': True}
    monkeypatch.setattr(study, 'evaluate_pair', evaluate)
    monkeypatch.setattr(study, 'aggregate', lambda *_: {'stub': True})
    result = study.run_study()
    assert evaluations == ['stub-no-truth-or-metric'] and len(replays) == 1
    assert syncs == [111, 777, 777, 111]
    assert 'ANGULAR_EXECUTION_STATE: COMPLETE' in report.artifact.getvalue()
    assert result['evidence_sha256'] == study.sha(path.artifact.getvalue())
    with pytest.raises(study.AngularStudyError, match='evidence_exists'): study.run_study()


def test_later_failure_retains_first_pair_no_evaluator_and_no_retry(monkeypatch, stub_entry):
    path, report, sealed = configure_memory_run(monkeypatch, stub_entry)
    second = replace(stub_entry.scene, identity='never-sealed')
    monkeypatch.setattr(study, 'SCENES', (stub_entry.scene, second))
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    def render(scene):
        if scene == second: raise KeyboardInterrupt()
        return stub_entry.image
    monkeypatch.setattr(study, 'render_rgb', render)
    monkeypatch.setattr(study, 'evaluate_pair', lambda *_: pytest.fail('evaluator before all sealed'))
    with pytest.raises(KeyboardInterrupt): study.run_study()
    assert len(gzip.decompress(path.artifact.getvalue()).splitlines()) == 2
    assert study.replay_evidence((sealed.sha256,), study.sha(path.artifact.getvalue()))
    assert 'ANGULAR_EXECUTION_STATE: RUNNING' in report.artifact.getvalue()
    with pytest.raises(study.AngularStudyError, match='evidence_exists'): study.run_study()
    with pytest.raises(study.AngularStudyError, match='already_claimed'): study.claim_report()


def test_final_digest_veto_does_not_append_result(monkeypatch, stub_entry):
    _, report, _ = configure_memory_run(monkeypatch, stub_entry)
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    monkeypatch.setattr(study, 'evaluate_pair', lambda *_: {'stub': True})
    monkeypatch.setattr(study, 'artifact_digest', lambda: '0'*64)
    monkeypatch.setattr(study, 'aggregate', lambda *_: pytest.fail('aggregate after tamper'))
    with pytest.raises(study.AngularStudyError, match='changed_during_evaluation'): study.run_study()
    assert 'ANGULAR_EXECUTION_STATE: RUNNING' in report.artifact.getvalue()
    assert 'Durable registered result' not in report.artifact.getvalue()


def aggregate_fixture():
    values, predictions = [], {}
    for i, scene in enumerate(study.SCENES):
        A_pass, B_pass = i//3 < 3, i//3 < 9
        arm = lambda p: {'diagnostic_pass': p, 'wrong': not p, 'status': 'emitted'}
        values.append({'scene': scene.identity, 'family': scene.family, 'stratum': scene.stratum,
            'arms': {'A': arm(A_pass), 'B': arm(B_pass)}})
        predictions[scene.identity] = {'A': {'stub': 'A'}, 'B': {'stub': 'B'}}
    return values, predictions


@pytest.mark.parametrize('failure', ['none', 'pass_count', 'gain', 'corrected', 'lost', 'stress_wrong', 'gui'])
def test_pilot_gate_each_criterion_and_refusal_not_recovery(failure):
    values, predictions = aggregate_fixture()
    if failure == 'pass_count': values[24]['arms']['B']['diagnostic_pass'] = False
    if failure == 'gain':
        for i in (9, 12, 15, 18): values[i]['arms']['A']['diagnostic_pass'] = True
    if failure == 'corrected':
        for v in values:
            if v['arms']['B']['diagnostic_pass'] and not v['arms']['A']['diagnostic_pass']:
                v['arms']['A'].update(wrong=False, status='refused')
    if failure == 'lost': values[0]['arms']['A']['diagnostic_pass'] = True; values[0]['arms']['B']['diagnostic_pass'] = False
    if failure == 'stress_wrong': values[1]['arms']['B'].update(wrong=True, diagnostic_pass=False)
    if failure == 'gui': predictions[study.SCENES[2].identity]['B'] = {'stub': 'changed'}
    result = study.aggregate(values, predictions)
    assert result['pilot_gate_pass'] == (failure == 'none')
    assert result['refusal_is_accuracy_gain'] is False and result['integration_authorized'] is False


def test_dependency_runtime_and_loss_drift_refuse_before_render(monkeypatch):
    original = study.dependency_pins()
    monkeypatch.setattr(study, 'dependency_pins', lambda: dict(original, runtime={'drift': True}))
    monkeypatch.setattr(study, 'render_rgb', lambda *_: pytest.fail('render before pin check'))
    with pytest.raises(study.AngularStudyError, match='dependency_or_runtime_drift'): study.verify_preregistration()


def test_each_source_and_version_is_bound_before_inference(monkeypatch):
    pins = study.dependency_pins()
    monkeypatch.setattr(study, 'render_rgb', lambda *_: pytest.fail('render on drift'))
    monkeypatch.setattr(study, 'trace_arm', lambda *_: pytest.fail('fit on drift'))
    for section, mapping in pins.items():
        assert isinstance(mapping, dict)
        for key in mapping:
            altered = deepcopy(pins)
            altered[section][key] = 'changed upstream source or runtime'
            monkeypatch.setattr(study, 'dependency_pins', lambda value=altered: value)
            with pytest.raises(study.AngularStudyError, match='dependency_or_runtime_drift'):
                study.run_study()
    assert not study._STUDY_LOCK.locked()


def test_report_load_explicit_no_render_and_loss_drift(monkeypatch):
    monkeypatch.setattr(study, 'REPORT_PATH', MemoryReport(report_content()))
    monkeypatch.setattr(study, 'PREREGISTRATION_JSON', None)
    monkeypatch.setattr(study, 'render_rgb', lambda *_: pytest.fail('load rendered'))
    study.initialize_preregistration()
    study.verify_preregistration()
    monkeypatch.setattr(study, 'TAU', study.TAU+.001)
    with pytest.raises(study.AngularStudyError, match='loss_filter_or_cap_drift'): study.verify_preregistration()


def test_source_ast_fail_closed_checks_whitespace_and_calibration_pin():
    for filename in ('scripts/evaluation/run_angular_objective.py', 'tests/test_angular_objective.py'):
        raw = Path(filename).read_bytes()
        assert raw.endswith(b'\n') and b'\t' not in raw
        assert all(line.rstrip(b' \t') == line for line in raw.splitlines())
        tree = ast.parse(raw)
        if filename.startswith('scripts/'):
            assert not any(isinstance(n, ast.Assert) for n in ast.walk(tree))
            assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'freeze_preregistration'
                           for n in tree.body)
    assert study.sha(Path('src/mc_photogrammetry/reconstruction/calibration.py').read_bytes()) == '2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc'
