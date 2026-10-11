"""GUI-BORDER pure renderer/filter and stub call-through QA; no study/metrics."""
from dataclasses import asdict, replace
import copy
import io
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import run_gui_border as study


@pytest.fixture(scope='session', autouse=True)
def explicitly_freeze_test_preregistration():
    # Explicit stimulus setup; importing the new harness renders no fixtures.
    study.initialize_preregistration(study.freeze_preregistration())


def vertical_mask(height=30, width=40, start=20):
    mask = np.zeros((height, width), bool)
    mask[:, start:] = True
    return mask


@pytest.mark.parametrize('value', [False, True])
def test_constant_masks_have_no_image_perimeter_boundary(value):
    mask = np.full((13, 17), value, bool)
    assert not study.mask_boundary(mask).any()
    raw = np.array([[0., 0., 16., 0.], [0., 0., 0., 12.], [16., 12., 0., 12.]])
    kept, info = study.select_segments(raw, mask)
    assert kept.tolist() == [0, 1, 2] and info['overlap_counts'] == [0, 0, 0]


def test_both_transition_pixels_and_chebyshev_corner_dilation_literal_oracle():
    mask = np.zeros((17, 19), bool)
    mask[7:10, 8:12] = True
    expected = np.zeros_like(mask)
    for y in range(17):
        for x in range(19):
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = x+dx, y+dy
                if nx < 19 and ny < 17 and mask[y, x] != mask[ny, nx]:
                    for sx, sy in ((x, y), (nx, ny)):
                        for yy in range(max(0, sy-3), min(17, sy+4)):
                            for xx in range(max(0, sx-3), min(19, sx+4)):
                                expected[yy, xx] = True
    assert np.array_equal(study.mask_boundary(mask), expected)
    assert expected[4, 4] and not expected[3, 4]


@pytest.mark.parametrize('start,end,hits,count,retained', [
    (0., 16., 1, 17, True),  # Endpoint-only contact is insufficient.
    (0., 39., 8, 40, True),  # A long crossing stays intact.
    (9., 24., 8, 16, False),  # Exact half excludes the whole line.
    (8., 24., 8, 17, True),
    (16., 23., 8, 8, False),
])
def test_endpoint_crossing_and_exact_half_threshold(start, end, hits, count, retained):
    kept, info = study.select_segments(np.array([[start, 10., end, 10.]]), vertical_mask())
    assert info['sample_counts'] == [count] and info['overlap_counts'] == [hits]
    assert kept.tolist() == ([0] if retained else [])


@pytest.mark.parametrize('x,excluded', [(15.49, False), (15.5, True), (15.51, True), (23.49, True), (23.5, False)])
def test_nearest_pixel_rounding_half_goes_up_and_zero_length(x, excluded):
    kept, info = study.select_segments(np.array([[x, 10., x, 10.]]), vertical_mask())
    assert info['sample_counts'] == [1]
    assert kept.tolist() == ([] if excluded else [0])


def test_outside_samples_are_not_clipped_onto_a_boundary():
    mask = np.zeros((10, 10), bool)
    mask[:, 1:] = True
    kept, info = study.select_segments(np.array([[-20., 5., -1., 5.]]), mask)
    assert kept.tolist() == [0] and info['overlap_counts'] == [0]


def test_order_dtype_coordinates_bytes_immutable_and_deterministic():
    raw = np.array([[2, 3, 5, 3], [20, 2, 20, 20], [30, 25, 35, 26]], dtype=np.float32)
    mask = vertical_mask()
    source, before = raw.tobytes(), mask.tobytes()
    first, info = study.select_segments(raw, mask)
    second, info2 = study.select_segments(raw, mask)
    assert first.tolist() == second.tolist() == [0, 2] and info == info2
    assert raw.tobytes() == source and mask.tobytes() == before
    subset = raw[first].copy()
    assert subset.dtype == raw.dtype and subset.tobytes() == np.vstack((raw[0], raw[2])).tobytes()


@pytest.mark.parametrize('raw,mask,reason', [
    (np.zeros((1, 3)), vertical_mask(), 'invalid_raw_segments'),
    (np.array([[0., 0., float('nan'), 2.]]), vertical_mask(), 'invalid_raw_segments'),
    (np.zeros((1, 4)), np.zeros((3, 4), np.uint8), 'invalid_automatic_mask'),
    (np.zeros((1, 4)), np.zeros((0, 4), bool), 'invalid_automatic_mask'),
])
def test_invalid_filter_inputs_fail_closed(raw, mask, reason):
    with pytest.raises(study.BorderStudyError, match=reason):
        study.select_segments(raw, mask)


@pytest.mark.parametrize('cap,exact,reason', [('max_segments', 2, 'segment_cap'),
    ('max_samples_per_segment', 6, 'segment_sample_cap'), ('max_total_samples', 12, 'total_sample_cap')])
def test_filter_cap_exact_boundary_then_one_less_refuses(monkeypatch, cap, exact, reason):
    raw = np.array([[1., 1., 6., 1.], [1., 2., 6., 2.]])
    caps = dict(study.CAPS, **{cap: exact})
    monkeypatch.setattr(study, 'CAPS', caps)
    study.select_segments(raw, vertical_mask())
    caps[cap] = exact-1
    with pytest.raises(study.BorderStudyError, match=reason):
        study.select_segments(raw, vertical_mask())


def test_preregistry_disjoint_primary_and_correlated_variant_denominators():
    pinned = json.loads(study.PREREGISTRATION_JSON)
    primary = [s for s in study.SCENES if s.stratum == 'primary']
    assert len(study.SCENES) == 57 and len(primary) == len({s.family for s in primary}) == 12
    assert [sum(s.stratum == name for s in study.SCENES) for name in
            ('development', 'primary', 'crop', 'low-texture', 'gui-only', 'control')] == [7, 12, 12, 12, 12, 2]
    assert len({pinned['rgb_sha256'][s.identity] for s in primary}) == 12
    # Two low-texture stress images are observationally identical despite
    # different ray depths: the foreground and back wall have the same flat
    # signed-plane colour. Stress is correlated, never the primary denominator.
    assert pinned['rgb_sha256']['heldout-planar-1-low-texture'] == pinned['rgb_sha256']['heldout-occluding-1-low-texture']
    assert len(set(pinned['rgb_sha256'].values())) == 56
    assert pinned['camera_accuracy_collision_exclusions'] == []
    collision, = pinned['stimulus_collisions']
    assert collision['kind'] == 'image_identifiability_geometry_collision_same_camera_truth'
    assert len(set(collision['camera_truth_sha256'].values())) == 1
    assert len(set(collision['recipe_sha256'].values())) == 2
    assert len(set(collision['geometry_truth_sha256'].values())) == 2
    assert set(s.family.split('-')[1] for s in primary) == {'planar', 'stepped', 'occluding'}
    assert len({tuple(json.loads(s.recipe_json)[k] for k in ('yaw', 'pitch', 'vertical_fov')) for s in primary}) == 4
    old = {s.recipe_json for s in study.cam.SCENES}
    assert all(s.recipe_json not in old for s in primary)
    assert pinned['pilot_gate'] == study.PILOT_GATE and pinned['primary_denominator'] == 12
    assert pinned['production_gates_changed'] is False and pinned['axis_veto'] is False


def test_new_renderer_camera_geometry_native_seed_and_periodicity():
    recipe = json.loads(next(s.recipe_json for s in study.SCENES if s.identity == 'heldout-stepped-2-primary'))
    rgb, owners, depths, seams = study.render_new(recipe)
    assert rgb.dtype == np.uint8 and rgb.shape == (360, 480, 3)
    assert len(np.unique(owners[owners >= 0])) > 2 and np.ptp(depths[np.isfinite(depths)]) > 1.
    assert seams.any() and recipe['native_side'] == 16
    points = np.array([[.127, .381, 3.], [2.127, -1.619, 3.], [1., .381, 3.]])
    colors, lines = study.native_rgb(points, 2, -1, recipe)
    assert np.allclose(colors[0], colors[1], atol=1e-12) and lines.tolist() == [False, False, True]
    assert np.array_equal(colors[2], [31., 31., 31.])
    changed, _ = study.native_rgb(points, 2, -1, dict(recipe, texture_seed=recipe['texture_seed']+1))
    assert not np.array_equal(colors[0], changed[0])


def test_crop_native_subarray_gui_only_and_low_texture_renderer_no_fit():
    primary = next(s for s in study.SCENES if s.identity == 'heldout-occluding-3-primary')
    rgb = study.render_rgb(primary)
    crop = next(s for s in study.SCENES if s.identity == 'heldout-occluding-3-crop')
    assert np.array_equal(study.render_rgb(crop), rgb[19:333, 37:455])
    gui = next(s for s in study.SCENES if s.identity == 'heldout-occluding-3-gui-only')
    contaminated = study.render_rgb(gui)
    mask = study.cam.create_gui_mask(360, 480)
    assert np.array_equal(contaminated[~mask], rgb[~mask]) and np.any(contaminated[mask] != rgb[mask])
    low = next(s for s in study.SCENES if s.identity == 'heldout-occluding-3-low-texture')
    low_rgb, owners, depths, seams = study.render_new(json.loads(low.recipe_json))
    original = study.render_new(json.loads(primary.recipe_json))
    assert np.array_equal(owners, original[1]) and np.array_equal(depths, original[2]) and not seams.any()
    assert len(np.unique(low_rgb.reshape(-1, 3), axis=0)) <= 7


def test_hud_free_structure_crossing_control_contains_world_seams_at_mask_boundary():
    scene = next(s for s in study.SCENES if s.identity == 'control-structure-crosses-mask')
    rgb, owners, _, seams = study.render_new(json.loads(scene.recipe_json))
    mask = study.cam.create_gui_mask(*rgb.shape[:2])
    assert (seams & study.mask_boundary(mask) & (owners >= 0)).any()
    assert (seams & ~mask & (owners >= 0)).any()
    assert json.loads(scene.recipe_json)['gui_contamination'] is False


@pytest.fixture
def stub_public_entry(monkeypatch):
    """Real RGB/mask/enhance wrappers; camera/detector stubs do no optimization."""
    cam = study.cam
    image = np.full((80, 100, 3), 123, np.uint8)
    mask = vertical_mask(80, 100, 40)
    raw = np.array([[5., 10.+i, 30., 10.+i] for i in range(12)]+
                   [[39., 15., 39., 55.], [40., 20., 40., 60.]], np.float32)
    calls = {'mask': [], 'detector': [], 'camera_segments': []}
    def mask_original(*args, **kwargs):
        calls['mask'].append((args, kwargs))
        return mask
    def detector_original(*args, **kwargs):
        calls['detector'].append((args, kwargs))
        return raw
    def camera_original(grey, **kwargs):
        enhanced = cam.calibration.enhance(grey)
        selected = cam.calibration.detect_segments(enhanced)
        calls['camera_segments'].append(selected)
        if len(selected) < 12:
            return None
        return cam.calibration.CalibrationResult(100., 0., 0., 100, 80, len(selected), len(selected),
            0., (len(selected), 0, 0), 0., None, ())
    monkeypatch.setattr(cam.gui_mask, 'create_gui_mask', mask_original)
    monkeypatch.setattr(cam.calibration, 'detect_segments', detector_original)
    monkeypatch.setattr(cam.calibration, 'estimate_camera', camera_original)
    monkeypatch.setattr(cam, '_PINNED_CAMERA_CODE', camera_original.__code__)
    monkeypatch.setattr(study, 'verify_preregistration', lambda: None)
    scene = replace(study.SCENES[0], identity='stub-qa', renderer='stub', recipe_json='{}')
    pinned = json.loads(study.PREREGISTRATION_JSON)
    pinned['rgb_sha256'][scene.identity] = study.sha(image.tobytes())
    monkeypatch.setattr(study, 'PREREGISTRATION_JSON', study.canonical(pinned))
    return SimpleNamespace(image=image, raw=raw, mask=mask, calls=calls, scene=scene,
                           detector=detector_original, mask_function=mask_original, camera=camera_original)


def run_stub_pair(fixture):
    A = study.trace_arm(fixture.image, fixture.scene.identity, 'A')
    B = study.trace_arm(fixture.image, fixture.scene.identity, 'B')
    sealed = study.seal_pair(fixture.scene, A, B)
    return A, B, sealed


def test_A_identity_B_only_ordered_subset_delegates_once_and_full_upstream_parity(stub_public_entry):
    fixture = stub_public_entry
    image_bytes, raw_bytes, mask_bytes = fixture.image.tobytes(), fixture.raw.tobytes(), fixture.mask.tobytes()
    A, B, sealed = run_stub_pair(fixture)
    assert len(fixture.calls['mask']) == len(fixture.calls['detector']) == 2
    assert fixture.calls['camera_segments'][0] is fixture.raw
    assert fixture.calls['camera_segments'][1] is not fixture.raw
    assert np.array_equal(fixture.calls['camera_segments'][1], fixture.raw[:12])
    assert B['support']['indices'] == list(range(12)) and A['support']['indices'] == list(range(14))
    assert study.stage_array(A, 'enhance') == study.stage_array(B, 'enhance')
    assert A['output']['status'] == B['output']['status'] == 'returned'
    assert fixture.image.tobytes() == image_bytes and fixture.raw.tobytes() == raw_bytes and fixture.mask.tobytes() == mask_bytes
    assert study.cam.calibration.detect_segments is fixture.detector and study.cam.gui_mask.create_gui_mask is fixture.mask_function
    record = study.verify_pair(fixture.scene, sealed)
    assert record['arms']['B']['output']['prediction']['per_axis_inliers'] == [12, 0, 0]  # No added axis veto.


def test_excluded_pixel_mutation_preserves_generation_pipeline_and_prediction(stub_public_entry):
    fixture = stub_public_entry
    A1 = study.trace_arm(fixture.image, fixture.scene.identity, 'A')
    B1 = study.trace_arm(fixture.image, fixture.scene.identity, 'B')
    changed = fixture.image.copy()
    changed[fixture.mask] = [11, 217, 59]
    A2 = study.trace_arm(changed, fixture.scene.identity, 'A')
    B2 = study.trace_arm(changed, fixture.scene.identity, 'B')
    assert changed.tobytes() != fixture.image.tobytes()
    for first, second in ((A1, A2), (B1, B2)):
        assert first['support'] == second['support']
        assert first['output']['prediction'] == second['output']['prediction']
        assert study.stage_array(first, 'enhance') == study.stage_array(second, 'enhance')
        assert first['output']['rgb_sha256'] != second['output']['rgb_sha256']


def test_existing_below_12_support_refuses_without_score_or_axis_gate(stub_public_entry):
    fixture = stub_public_entry
    fixture.raw[:3, [0, 2]] = 39.
    B = study.trace_arm(fixture.image, fixture.scene.identity, 'B')
    assert B['output']['status'] == 'ordinary_refusal' and B['output']['prediction'] is None
    assert len(B['support']['indices']) == 9 and B['output']['trace']['score_events'] == 0
    study.verify_arm(B)


@pytest.mark.parametrize('kind', ['ordinary_refusal', 'exception', 'interrupt', 'trace_refusal'])
def test_outer_wrappers_restore_after_every_exit(monkeypatch, stub_public_entry, kind):
    fixture = stub_public_entry
    original = study.cam.trace_rgb
    if kind == 'ordinary_refusal':
        fixture.raw[:, [0, 2]] = 39.
        result = study.trace_arm(fixture.image, fixture.scene.identity, 'B')
        assert result['output']['prediction'] is None
    else:
        def fail(*args, **kwargs):
            if kind == 'interrupt':
                raise KeyboardInterrupt()
            if kind == 'trace_refusal':
                # Exercise actual entry up to captures, then explicit incomplete trace.
                found = original(*args, **kwargs)
                return dict(found, status='trace_refusal', reason='test-cap')
            raise LookupError('unexpected')
        monkeypatch.setattr(study.cam, 'trace_rgb', fail)
        with pytest.raises(KeyboardInterrupt if kind == 'interrupt' else
                           study.BorderStudyError if kind == 'trace_refusal' else LookupError):
            study.trace_arm(fixture.image, fixture.scene.identity, 'B')
    assert study.cam.calibration.detect_segments is fixture.detector
    assert study.cam.gui_mask.create_gui_mask is fixture.mask_function
    assert not study._ARM_LOCK.locked()


def test_nested_install_refuses_before_public_entry(stub_public_entry):
    assert study._ARM_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(study.BorderStudyError, match='arm_reentrancy_or_concurrency'):
            study.trace_arm(stub_public_entry.image, 'qa', 'B')
        assert not stub_public_entry.calls['detector']
    finally:
        study._ARM_LOCK.release()


@pytest.mark.parametrize('field', ['mask', 'raw', 'enhance'])
def test_pair_rejects_upstream_mismatch_before_truth(monkeypatch, stub_public_entry, field):
    A = study.trace_arm(stub_public_entry.image, 'qa', 'A')
    B = study.trace_arm(stub_public_entry.image, 'qa', 'B')
    monkeypatch.setattr(study, 'verify_arm', lambda x: x)
    if field == 'enhance':
        event = next(e for e in B['output']['trace']['stages'] if e['stage'] == 'enhance')
        B['output']['trace']['arrays'][event['output']] = study.snapshot(np.ones((2, 2)))
    else:
        B['support'][field] = study.snapshot(np.ones((2, 2)))
    with pytest.raises(study.BorderStudyError, match='pair_upstream_parity_mismatch'):
        study.seal_pair(stub_public_entry.scene, A, B)


def test_replay_rejects_indices_order_dtype_tamper_and_payload_hash(stub_public_entry):
    _, B, sealed = run_stub_pair(stub_public_entry)
    bad = copy.deepcopy(B)
    bad['support']['indices'] = list(reversed(bad['support']['indices']))
    with pytest.raises(study.BorderStudyError, match='ordered_raw_subset_mismatch'):
        study.verify_arm(bad)
    bad = copy.deepcopy(B)
    bad['support']['retained'] = study.snapshot(study.decode(bad['support']['retained']).astype(np.float64))
    with pytest.raises(study.BorderStudyError, match='ordered_raw_subset_mismatch'):
        study.verify_arm(bad)
    with pytest.raises(study.BorderStudyError, match='pair_seal_mismatch'):
        study.verify_pair(stub_public_entry.scene, replace(sealed, payload_json=sealed.payload_json+' '))


@pytest.mark.parametrize('change', ['dependency', 'version'])
def test_upstream_dependency_or_runtime_drift_refuses_before_entry(monkeypatch, change):
    pins = copy.deepcopy(study.dependency_pins())
    if change == 'dependency':
        pins['source_sha256']['src/mc_photogrammetry/vision/gui_mask.py'] = '0'*64
    else:
        pins['runtime']['numpy'] += '-changed'
    monkeypatch.setattr(study, 'dependency_pins', lambda: pins)
    monkeypatch.setattr(study, 'render_rgb', lambda *_: pytest.fail('render before pin check'))
    monkeypatch.setattr(study.cam, 'trace_rgb', lambda *_: pytest.fail('inference before pin check'))
    with pytest.raises(study.BorderStudyError, match='dependency_or_runtime_drift'):
        study.run_study()


def test_normalized_objectives_keep_set_denominators_separate_without_camera_scoring(monkeypatch):
    def scored(lines, weights, camera):
        return {'score': float(weights.sum())*.25, 'per_axis_inliers': [2, 1, 0]}
    monkeypatch.setattr(study.cam, 'full_segment_score', scored)
    raw = np.array([[0., 0., 3., 4.], [0., 0., 6., 8.]])
    original = study.normalized_score(raw, {})
    filtered = study.normalized_score(raw[:1], {})
    assert original['score'] == 3.75 and filtered['score'] == 1.25
    assert original['score_over_total_length'] == filtered['score_over_total_length'] == .25
    assert original['weak_axis_minimum'] == 0 and not original['axis_veto_applied']


def aggregate_inputs(A_passes=6, B_passes=9):
    values, predictions, i = [], {}, 0
    for scene in study.SCENES:
        arm = {}
        for name, total in (('A', A_passes), ('B', B_passes)):
            passed = scene.stratum == 'primary' and i < total
            arm[name] = {'diagnostic_pass': passed, 'emitted_wrong': not passed, 'status': 'emitted'}
        if scene.stratum == 'primary':
            i += 1
        values.append({'scene': scene.identity, 'family': scene.family, 'stratum': scene.stratum, 'arms': arm,
                       'camera_accuracy_eligible': True})
        predictions[scene.identity] = {'A': {'stub': 'A'}, 'B': {'stub': 'B'}}
    return values, predictions


def test_pilot_gate_exact_primary_counts_no_correlated_n_and_no_refusal_recovery():
    values, predictions = aggregate_inputs()
    report = study.aggregate(values, predictions)
    assert report['pilot_gate_pass'] and report['primary_families'] == 12
    assert report['primary_passes'] == {'A': 6, 'B': 9} and report['stress_counts']['B']['n_correlated_variants'] == 36
    assert not report['refusal_is_accuracy_recovery'] and not report['production_integration_authorized']
    first = next(v for v in values if v['stratum'] == 'primary')
    first['arms']['B'] = {'diagnostic_pass': False, 'emitted_wrong': False, 'status': 'refused'}
    report = study.aggregate(values, predictions)
    assert not report['pilot_gate_pass'] and report['primary_passes']['B'] == 8
    assert report['primary_counts']['B']['refused'] == 1 and report['lost_A_pass_families']


@pytest.mark.parametrize('failure', ['gain', 'B_minimum', 'lost', 'stress', 'gui_stability'])
def test_every_pilot_requirement_is_binding(failure):
    values, predictions = aggregate_inputs(A_passes=6, B_passes=9)
    if failure == 'gain':
        values, predictions = aggregate_inputs(A_passes=7, B_passes=9)
    elif failure == 'B_minimum':
        values, predictions = aggregate_inputs(A_passes=5, B_passes=8)
    elif failure == 'lost':
        primary = [v for v in values if v['stratum'] == 'primary']
        primary[0]['arms']['B']['diagnostic_pass'] = False
        primary[9]['arms']['B']['diagnostic_pass'] = True
    elif failure == 'stress':
        entry = next(v for v in values if v['stratum'] == 'crop')
        entry['arms']['A']['emitted_wrong'] = False
    else:
        gui = next(s.identity for s in study.SCENES if s.stratum == 'gui-only')
        predictions[gui]['B'] = {'stub': 'changed'}
    assert not study.aggregate(values, predictions)['pilot_gate_pass']


class MemoryArtifact(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def fileno(self):
        return 777


class MemoryPath:
    def __init__(self, content=None):
        self.exists_flag = content is not None
        self.artifact = MemoryArtifact(content or b'')
    def exists(self):
        return self.exists_flag
    def is_file(self):
        return self.exists_flag
    def stat(self):
        return SimpleNamespace(st_size=len(self.artifact.getvalue()))
    def open(self, mode):
        if mode in ('xb', 'x+b'):
            if self.exists_flag:
                raise FileExistsError()
            self.exists_flag = True
            self.artifact.seek(0)
            return self.artifact
        if not self.exists_flag:
            raise FileNotFoundError()
        self.artifact.seek(0)
        return self.artifact


def test_durable_full_pair_append_replay_prefix_tamper_without_real_artifact(monkeypatch, stub_public_entry):
    _, _, sealed = run_stub_pair(stub_public_entry)
    monkeypatch.setattr(study, 'SCENES', (stub_public_entry.scene,))
    path = MemoryPath()
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    flushes = []
    monkeypatch.setattr(study.os, 'fsync', lambda fd: flushes.append(fd))
    with study.EvidenceWriter(path) as writer:
        descriptor = study.append_pair(stub_public_entry.scene, sealed, (), writer=writer)
        result = writer.digest.hexdigest()
    assert flushes == [777, 777] and result == study.sha(path.artifact.getvalue())
    assert study.read_pair_member(descriptor) == sealed
    assert study.replay_evidence((sealed.sha256,)) == result
    records = gzip.decompress(path.artifact.getvalue()).splitlines()
    assert json.loads(records[1]) == asdict(sealed)
    damaged = json.loads(records[1])
    damaged['payload_json'] += ' '
    path.artifact = MemoryArtifact(study.compress_member(records[0]+b'\n')+
                                   study.compress_member(study.canonical(damaged).encode()+b'\n'))
    with pytest.raises(study.BorderStudyError, match='pair_seal_mismatch'):
        study.replay_evidence((sealed.sha256,))


def test_registered_order_all_pairs_durable_before_evaluator_and_prefix_survives_later_failure(monkeypatch, stub_public_entry):
    _, _, sealed = run_stub_pair(stub_public_entry)
    first = stub_public_entry.scene
    second = replace(first, identity='later-failing-scene')
    monkeypatch.setattr(study, 'SCENES', (first, second))
    path = MemoryPath()
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    monkeypatch.setattr(study, 'claim_report', lambda: None)
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    monkeypatch.setattr(study, 'render_rgb', lambda *_: stub_public_entry.image)
    monkeypatch.setattr(study, 'trace_arm', lambda *a: {} if a[1] == first.identity else
                        (_ for _ in ()).throw(LookupError('later failure')))
    monkeypatch.setattr(study, 'seal_pair', lambda *_: sealed)
    monkeypatch.setattr(study, 'evaluate_pair', lambda *_: pytest.fail('truth before every pair durably sealed'))
    with pytest.raises(LookupError, match='later failure'):
        study.run_study()
    assert len(gzip.decompress(path.artifact.getvalue()).splitlines()) == 2
    assert study.replay_evidence((sealed.sha256,)) == study.sha(path.artifact.getvalue())
    with pytest.raises(study.BorderStudyError, match='evidence_already_exists'):
        study.run_study()


def test_missing_evidence_rejected_and_real_artifact_remains_absent():
    assert not study.EVIDENCE_PATH.exists()
    with pytest.raises(study.BorderStudyError, match='evidence_missing'):
        study.replay_evidence(())


def test_empty_detector_output_is_ordinary_refusal_and_replayable(monkeypatch, stub_public_entry):
    monkeypatch.setattr(study.cam.calibration, 'detect_segments', lambda *a, **k: np.empty((0, 4), np.float32))
    for arm in ('A', 'B'):
        result = study.trace_arm(stub_public_entry.image, 'qa', arm)
        assert result['output']['status'] == 'ordinary_refusal' and result['support']['indices'] == []
        study.verify_arm(result)


def test_filter_cap_in_actual_wrapper_aborts_without_prediction_and_restores(monkeypatch, stub_public_entry):
    monkeypatch.setattr(study, 'CAPS', dict(study.CAPS, max_segments=13))
    with pytest.raises(study.BorderStudyError, match='segment_cap'):
        study.trace_arm(stub_public_entry.image, 'qa', 'B')
    assert study.cam.calibration.detect_segments is stub_public_entry.detector
    assert study.cam.gui_mask.create_gui_mask is stub_public_entry.mask_function
    assert not study._ARM_LOCK.locked()


def test_full_objective_cannot_claim_different_retained_lines(stub_public_entry):
    B = study.trace_arm(stub_public_entry.image, 'qa', 'B')
    trace = B['output']['trace']
    sites = next(site for site in study.cam.PINNED_SITES if site['tag'] == 'fine')
    wrong_lines = np.zeros((12, 4), np.float32)
    weights = np.zeros(12, np.float32)
    line_meta, weight_meta = study.snapshot(wrong_lines), study.snapshot(weights)
    line_sha, weight_sha = study.sha(study.canonical(line_meta)), study.sha(study.canonical(weight_meta))
    trace['arrays'][line_sha], trace['arrays'][weight_sha] = line_meta, weight_meta
    trace['array_bytes'] += wrong_lines.nbytes+weights.nbytes
    event = {'site': 'fine', 'span': [sites['start'], sites['end']], 'segments_sha256': line_sha,
             'weights_sha256': weight_sha, 'score': 0.}
    trace['scores'].append(event)
    trace['score_events'] += 1
    trace['trace_event_bytes'] += len(study.canonical(event).encode())
    with pytest.raises(study.BorderStudyError, match='full_objective_support_mismatch'):
        study.verify_arm(B)


def test_duplicate_RGB_conflicting_camera_metadata_is_frozen_exclusion_not_accuracy(monkeypatch):
    original = next(s for s in study.SCENES if s.identity == 'heldout-planar-1-low-texture')
    recipe = json.loads(original.recipe_json)
    changed = replace(original, identity='camera-conflict', recipe_json=study.canonical(dict(recipe, yaw=recipe['yaw']+1)))
    monkeypatch.setattr(study, 'SCENES', (original, changed))
    collisions = study.stimulus_collisions({s.identity: 'same-rendered-rgb' for s in study.SCENES})
    collision, = collisions
    assert collision['kind'] == 'image_identifiability_camera_collision'
    assert not collision['camera_accuracy_eligible'] and len(set(collision['camera_truth_sha256'].values())) == 2


def test_conflicting_truth_group_is_not_scored_or_primary_denominator(monkeypatch, stub_public_entry):
    _, _, sealed = run_stub_pair(stub_public_entry)
    pinned = json.loads(study.PREREGISTRATION_JSON)
    pinned['camera_accuracy_collision_exclusions'] = [stub_public_entry.scene.identity]
    monkeypatch.setattr(study, 'PREREGISTRATION_JSON', study.canonical(pinned))
    monkeypatch.setattr(study, 'truth_camera', lambda *_: pytest.fail('conflicting truth must not be selected'))
    monkeypatch.setattr(study.cam, 'camera_metrics', lambda *_: pytest.fail('camera accuracy must not be scored'))
    monkeypatch.setattr(study.cam, 'stage_camera_evidence', lambda *_: {'stub_stages': True})
    monkeypatch.setattr(study, 'normalized_score', lambda *_: {'stub_emitted_objective': True})
    evaluated = study.evaluate_pair(stub_public_entry.scene, sealed)
    assert not evaluated['camera_accuracy_eligible'] and evaluated['arms']['A']['metrics'] is None
    assert evaluated['arms']['A']['renderer_scores'] is None
    values, predictions = aggregate_inputs()
    excluded = next(v for v in values if v['stratum'] == 'crop')
    excluded['camera_accuracy_eligible'] = False
    report = study.aggregate(values, predictions)
    assert report['primary_families'] == 12 and report['stress_counts']['B']['n_correlated_variants'] == 35
    assert report['camera_accuracy_excluded_scenes'] == [excluded['scene']]


class TextArtifact(io.StringIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def fileno(self):
        return 111


class MemoryReport:
    def __init__(self, content):
        self.artifact = TextArtifact(content)
    def is_file(self):
        return True
    def open(self, *args, **kwargs):
        self.artifact.seek(0)
        return self.artifact
    def read_text(self, **kwargs):
        return self.artifact.getvalue()


def report_content(state='NOT_STARTED'):
    return ('GUI_BORDER_EXECUTION_STATE: '+state+'\nPreregistration SHA256: `'+study.PREREGISTRATION_SHA256+'`.\n'+
            '\nPREREGISTRATION_BEGIN\n```json\n'+study.PREREGISTRATION_JSON+'\n```\nPREREGISTRATION_END\n')


def test_one_shot_claim_is_fsynced_before_inference_and_crash_forbids_retry(monkeypatch):
    report = MemoryReport(report_content())
    evidence = MemoryPath()
    monkeypatch.setattr(study, 'REPORT_PATH', report)
    monkeypatch.setattr(study, 'EVIDENCE_PATH', evidence)
    monkeypatch.setattr(study, 'verify_preregistration', lambda: None)
    locks, durable = [], []
    if study.os.name == 'nt':
        import msvcrt
        monkeypatch.setattr(msvcrt, 'locking', lambda *a: locks.append(a))
    def fsync(fd):
        if fd == 111:
            durable.append(report.artifact.getvalue())
    monkeypatch.setattr(study.os, 'fsync', fsync)
    def crash(scene):
        assert durable and 'GUI_BORDER_EXECUTION_STATE: RUNNING' in durable[0]
        raise KeyboardInterrupt()
    monkeypatch.setattr(study, 'render_rgb', crash)
    monkeypatch.setattr(study, 'trace_arm', lambda *_: pytest.fail('unexpected inference'))
    with pytest.raises(KeyboardInterrupt):
        study.run_study()
    assert evidence.exists() and len(gzip.decompress(evidence.artifact.getvalue()).splitlines()) == 1
    assert 'GUI_BORDER_EXECUTION_STATE: RUNNING' in report.artifact.getvalue()
    with pytest.raises(study.BorderStudyError, match='evidence_already_exists'):
        study.run_study()
    with pytest.raises(study.BorderStudyError, match='study_already_claimed_or_pin_mismatch'):
        study.claim_report()


def test_report_registered_payload_tamper_refuses_before_durable_claim(monkeypatch):
    content = report_content().replace('"primary_denominator":12', '"primary_denominator":13')
    report = MemoryReport(content)
    monkeypatch.setattr(study, 'REPORT_PATH', report)
    monkeypatch.setattr(study.os, 'fsync', lambda *_: pytest.fail('tamper must not consume marker'))
    if study.os.name == 'nt':
        import msvcrt
        monkeypatch.setattr(msvcrt, 'locking', lambda *_: None)
    with pytest.raises(study.BorderStudyError, match='report_preregistration_payload_mismatch'):
        study.claim_report()
    assert report.artifact.getvalue() == content


def test_evaluator_spy_runs_only_after_all_full_pairs_append_flush_and_replay(monkeypatch, stub_public_entry):
    A, B, sealed1 = run_stub_pair(stub_public_entry)
    first, second = stub_public_entry.scene, replace(stub_public_entry.scene, identity='second-stub-qa')
    pinned = json.loads(study.PREREGISTRATION_JSON)
    pinned['rgb_sha256'][second.identity] = pinned['rgb_sha256'][first.identity]
    monkeypatch.setattr(study, 'PREREGISTRATION_JSON', study.canonical(pinned))
    sealed2 = study.seal_pair(second, A, B)
    monkeypatch.setattr(study, 'SCENES', (first, second))
    path, events = MemoryPath(), []
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    monkeypatch.setattr(study, 'claim_report', lambda: events.append('durable-claim'))
    monkeypatch.setattr(study.os, 'fsync', lambda *_: events.append('fsync'))
    monkeypatch.setattr(study, 'render_rgb', lambda *_: stub_public_entry.image)
    monkeypatch.setattr(study, 'trace_arm', lambda *a: A if a[2] == 'A' else B)
    monkeypatch.setattr(study, 'seal_pair', lambda scene, *a: sealed1 if scene == first else sealed2)
    class StopAfterOrderProof(BaseException):
        pass
    def evaluator(scene, sealed):
        assert events == ['durable-claim', 'fsync', 'fsync', 'fsync']
        assert len(gzip.decompress(path.artifact.getvalue()).splitlines()) == 3
        decoded_records = gzip.decompress(path.artifact.getvalue()).splitlines()[1:]
        assert [json.loads(line)['sha256'] for line in decoded_records] == [sealed1.sha256, sealed2.sha256]
        raise StopAfterOrderProof()
    monkeypatch.setattr(study, 'evaluate_pair', evaluator)
    with pytest.raises(StopAfterOrderProof):
        study.run_study()


def test_algorithm_contains_no_optimization_disabled_bare_assert_gate():
    import ast
    tree = ast.parse(study.Path(study.__file__).read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_partial_outer_install_restores_prior_mask_and_releases_lock(monkeypatch, stub_public_entry):
    original_calibration = study.cam.calibration
    class PartialInstall:
        def __init__(self):
            object.__setattr__(self, 'detect_segments', stub_public_entry.detector)
            object.__setattr__(self, 'failed', False)
        def __setattr__(self, name, value):
            if name == 'detect_segments' and value is not stub_public_entry.detector and not self.failed:
                object.__setattr__(self, 'failed', True)
                raise LookupError('partial-install')
            object.__setattr__(self, name, value)
    proxy = PartialInstall()
    monkeypatch.setattr(study.cam, 'calibration', proxy)
    with pytest.raises(LookupError, match='partial-install'):
        study.trace_arm(stub_public_entry.image, 'qa', 'B')
    assert proxy.detect_segments is stub_public_entry.detector
    assert study.cam.gui_mask.create_gui_mask is stub_public_entry.mask_function
    assert not study._ARM_LOCK.locked() and original_calibration.detect_segments is stub_public_entry.detector


def test_raw_input_mutation_detected_by_accepted_trace_and_outer_restores(monkeypatch, stub_public_entry):
    def bad_detector(enhanced, *args, **kwargs):
        enhanced[:] = 0
        return stub_public_entry.raw
    monkeypatch.setattr(study.cam.calibration, 'detect_segments', bad_detector)
    with pytest.raises(study.BorderStudyError, match='incomplete_camera_trace:trace_array_mutated'):
        study.trace_arm(stub_public_entry.image, 'qa', 'B')
    assert study.cam.calibration.detect_segments is bad_detector
    assert study.cam.gui_mask.create_gui_mask is stub_public_entry.mask_function and not study._ARM_LOCK.locked()


@pytest.mark.parametrize('short', [False, True])
def test_durable_record_byte_cap_exact_then_one_less_before_append(monkeypatch, stub_public_entry, short):
    _, _, sealed = run_stub_pair(stub_public_entry)
    encoded = (study.canonical(asdict(sealed))+'\n').encode()
    monkeypatch.setattr(study, 'SCENES', (stub_public_entry.scene,))
    monkeypatch.setattr(study, 'CAPS', dict(study.CAPS, max_record_bytes=len(encoded)-int(short)))
    header = study.compress_member((study.canonical(study.evidence_header())+'\n').encode())
    path = MemoryPath()
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    flushes = []
    monkeypatch.setattr(study.os, 'fsync', lambda *_: flushes.append(True))
    with study.EvidenceWriter(path) as writer:
        if short:
            with pytest.raises(study.BorderStudyError, match='decompressed_record_cap'):
                study.append_pair(stub_public_entry.scene, sealed, (), writer=writer)
            assert path.artifact.getvalue() == header and flushes == [True]
        else:
            study.append_pair(stub_public_entry.scene, sealed, (), writer=writer)
            assert flushes == [True, True]
            assert len(gzip.decompress(path.artifact.getvalue()).splitlines()) == 2


def gzip_pair_fixture(monkeypatch, fixture, count=1):
    A, B, first = run_stub_pair(fixture)
    registered = [fixture.scene]
    sealed = [first]
    if count == 2:
        second = replace(fixture.scene, identity='second-stub-pair')
        pinned = json.loads(study.PREREGISTRATION_JSON)
        pinned['rgb_sha256'][second.identity] = pinned['rgb_sha256'][fixture.scene.identity]
        monkeypatch.setattr(study, 'PREREGISTRATION_JSON', study.canonical(pinned))
        registered.append(second)
        sealed.append(study.seal_pair(second, A, B))
    monkeypatch.setattr(study, 'SCENES', tuple(registered))
    path = MemoryPath()
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    with study.EvidenceWriter(path) as writer:
        for scene, pair in zip(registered, sealed):
            study.append_pair(scene, pair, tuple(writer.seals), writer=writer)
        descriptors, expected_digest = list(writer.members), writer.digest.hexdigest()
    return path, sealed, descriptors, expected_digest


def test_multiple_independent_gzip_members_standard_reader_and_single_full_replay(monkeypatch, stub_public_entry):
    path, pairs, descriptors, expected_digest = gzip_pair_fixture(monkeypatch, stub_public_entry, 2)
    assert len(descriptors) == 3 and descriptors[0]['offset'] == 0
    raw = path.artifact.getvalue()
    assert len(gzip.decompress(raw).splitlines()) == 3
    for before, after in zip(descriptors, descriptors[1:]):
        assert before['offset']+before['compressed_bytes'] == after['offset']
    for descriptor, pair in zip(descriptors[1:], pairs):
        assert study.read_pair_member(descriptor) == pair
    assert study.replay_evidence(tuple(pair.sha256 for pair in pairs), expected_digest) == study.sha(raw)


@pytest.mark.parametrize('cut', [1, 4, 8, 25])
def test_gzip_member_truncation_rejects_without_discarding_earlier_members(monkeypatch, stub_public_entry, cut):
    path, pairs, descriptors, _ = gzip_pair_fixture(monkeypatch, stub_public_entry, 2)
    original = path.artifact.getvalue()
    path.artifact = MemoryArtifact(original[:-cut])
    assert path.artifact.getvalue()[:descriptors[2]['offset']] == original[:descriptors[2]['offset']]
    with pytest.raises(study.BorderStudyError, match='gzip_truncated_member|gzip_integrity_error'):
        study.replay_evidence(tuple(pair.sha256 for pair in pairs))
    assert path.artifact.getvalue() == original[:-cut]


def test_gzip_crc_and_artifact_digest_mismatch_rejected(monkeypatch, stub_public_entry):
    path, pairs, _, expected = gzip_pair_fixture(monkeypatch, stub_public_entry)
    with pytest.raises(study.BorderStudyError, match='artifact_digest_mismatch'):
        study.replay_evidence((pairs[0].sha256,), '0'*64)
    changed = bytearray(path.artifact.getvalue())
    changed[-8] ^= 1
    path.artifact = MemoryArtifact(bytes(changed))
    with pytest.raises(study.BorderStudyError, match='gzip_integrity_error'):
        study.replay_evidence((pairs[0].sha256,), expected)


@pytest.mark.parametrize('scope', ['member', 'total', 'decoded'])
def test_compressed_and_decompressed_replay_caps_exact_and_one_less(monkeypatch, stub_public_entry, scope):
    path, pairs, descriptors, expected = gzip_pair_fixture(monkeypatch, stub_public_entry)
    limits = {'member': max(d['compressed_bytes'] for d in descriptors),
              'total': len(path.artifact.getvalue()),
              'decoded': len((study.canonical(asdict(pairs[0]))+'\n').encode())}
    def set_limit(value):
        if scope == 'total':
            monkeypatch.setattr(study, 'MAX_EVIDENCE_BYTES', value)
        else:
            key = 'max_compressed_member_bytes' if scope == 'member' else 'max_record_bytes'
            monkeypatch.setattr(study, 'CAPS', dict(study.CAPS, **{key: value}))
    set_limit(limits[scope])
    assert study.replay_evidence((pairs[0].sha256,), expected) == expected
    set_limit(limits[scope]-1)
    reason = {'member': 'compressed_member_cap', 'total': 'compressed_total_cap', 'decoded': 'decompressed_record_cap'}[scope]
    with pytest.raises(study.BorderStudyError, match=reason):
        study.replay_evidence((pairs[0].sha256,), expected)


@pytest.mark.parametrize('scope', ['member', 'total'])
def test_compression_append_caps_preserve_header_and_prior_pair_members(monkeypatch, stub_public_entry, scope):
    _, _, pair = run_stub_pair(stub_public_entry)
    monkeypatch.setattr(study, 'SCENES', (stub_public_entry.scene,))
    encoded = (study.canonical(asdict(pair))+'\n').encode()
    compressed = study.compress_member(encoded)
    path = MemoryPath()
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    with study.EvidenceWriter(path) as writer:
        before = path.artifact.getvalue()
        if scope == 'member':
            monkeypatch.setattr(study, 'CAPS', dict(study.CAPS, max_compressed_member_bytes=len(compressed)-1))
        else:
            monkeypatch.setattr(study, 'MAX_EVIDENCE_BYTES', len(before)+len(compressed)-1)
        with pytest.raises(study.BorderStudyError, match='compressed_member_cap|compressed_total_cap'):
            study.append_pair(stub_public_entry.scene, pair, (), writer=writer)
        assert path.artifact.getvalue() == before and writer.seals == []


def test_short_write_leaves_existing_member_prefix_and_never_truncates(monkeypatch, stub_public_entry):
    _, _, pair = run_stub_pair(stub_public_entry)
    monkeypatch.setattr(study, 'SCENES', (stub_public_entry.scene,))
    path = MemoryPath()
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    with study.EvidenceWriter(path) as writer:
        before, original = path.artifact.getvalue(), path.artifact.write
        monkeypatch.setattr(path.artifact, 'write', lambda value: original(value[:-3]))
        with pytest.raises(study.BorderStudyError, match='evidence_short_write'):
            study.append_pair(stub_public_entry.scene, pair, (), writer=writer)
        after = path.artifact.getvalue()
        assert after.startswith(before) and len(after) > len(before) and writer.seals == []
    members = study.gzip_members(io.BytesIO(after))
    assert study.json_line(next(members)[0]) == study.evidence_header()
    with pytest.raises(study.BorderStudyError, match='gzip_truncated_member'):
        next(members)


def test_writer_exclusive_single_owner_and_size_drift_refusal(monkeypatch):
    path, other = MemoryPath(), MemoryPath()
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    with study.EvidenceWriter(path) as writer:
        with pytest.raises(study.BorderStudyError, match='evidence_writer_already_owned'):
            with study.EvidenceWriter(other):
                pytest.fail('second owner')
        assert not other.exists()
        path.artifact.seek(0, 2)
        path.artifact.write(b'unknown')
        with pytest.raises(study.BorderStudyError, match='evidence_owner_size_mismatch'):
            writer.append_record(b'{}\n')
    assert not study._WRITER_LOCK.locked()
    with pytest.raises(FileExistsError):
        with study.EvidenceWriter(path):
            pytest.fail('existing artifact reopened')
    assert not study._WRITER_LOCK.locked()


def test_new_fixture_freeze_is_explicit_reproducible_and_import_inert():
    import ast
    import subprocess
    import sys
    tree = ast.parse(study.Path(study.__file__).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assert not any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                           and call.func.id == 'freeze_preregistration' for call in ast.walk(node))
    result = subprocess.run([sys.executable, '-B', '-c',
        'from scripts.evaluation import run_gui_border as s; '
        'assert s.PREREGISTRATION_JSON is None and s.PREREGISTRATION_SHA256 is None; print("new-prereg-inert")'],
        cwd=str(study.ROOT), capture_output=True, text=True)
    assert result.returncode == 0 and 'new-prereg-inert' in result.stdout
    assert study.freeze_preregistration() == study.PREREGISTRATION_JSON


def test_execution_loads_reviewed_report_without_rendering_fixtures(monkeypatch):
    expected = study.PREREGISTRATION_JSON
    monkeypatch.setattr(study, 'REPORT_PATH', MemoryReport(report_content()))
    monkeypatch.setattr(study, 'PREREGISTRATION_JSON', None)
    monkeypatch.setattr(study, 'PREREGISTRATION_SHA256', None)
    monkeypatch.setattr(study, 'render_rgb', lambda *_: pytest.fail('execution-time import/preflight rendered fixture'))
    study.verify_preregistration()
    assert study.PREREGISTRATION_JSON == expected


def test_owned_files_actual_byte_whitespace_and_AST_checks():
    import ast
    for name in ('scripts/evaluation/run_gui_border.py', 'tests/test_gui_border.py',
                 'docs/reports/GUI-BORDER-01-2026-10-05.md'):
        data = (study.ROOT/name).read_bytes()
        assert data.endswith(b'\n') and b'\t' not in data
        assert all(line.rstrip() == line for line in data.splitlines())
        if name.endswith('.py'):
            ast.parse(data)


def test_complete_transition_appends_result_only_after_final_digest_and_one_full_replay(monkeypatch, stub_public_entry):
    A, B, sealed = run_stub_pair(stub_public_entry)
    monkeypatch.setattr(study, 'SCENES', (stub_public_entry.scene,))
    path, report = MemoryPath(), MemoryReport(report_content())
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    monkeypatch.setattr(study, 'REPORT_PATH', report)
    if study.os.name == 'nt':
        import msvcrt
        monkeypatch.setattr(msvcrt, 'locking', lambda *_: None)
    fsyncs, replays, evaluated = [], [], []
    monkeypatch.setattr(study.os, 'fsync', lambda fd: fsyncs.append(fd))
    original_replay = study.replay_evidence
    def replay(*args, **kwargs):
        replays.append('full-replay')
        return original_replay(*args, **kwargs)
    monkeypatch.setattr(study, 'replay_evidence', replay)
    monkeypatch.setattr(study, 'render_rgb', lambda *_: stub_public_entry.image)
    monkeypatch.setattr(study, 'trace_arm', lambda *a: A if a[2] == 'A' else B)
    monkeypatch.setattr(study, 'seal_pair', lambda *_: sealed)
    def evaluate(*args):
        assert replays == ['full-replay'] and fsyncs == [111, 777, 777]
        assert 'GUI_BORDER_EXECUTION_STATE: RUNNING' in report.artifact.getvalue()
        evaluated.append('stub-only-no-metric')
        return {'stub_evaluation': True}
    monkeypatch.setattr(study, 'evaluate_pair', evaluate)
    monkeypatch.setattr(study, 'aggregate', lambda *_: {'stub_aggregate': True})
    result = study.run_study()
    assert evaluated == ['stub-only-no-metric'] and replays == ['full-replay']
    assert fsyncs == [111, 777, 777, 111] and 'GUI_BORDER_EXECUTION_STATE: COMPLETE' in report.artifact.getvalue()
    assert 'Durable registered result:' in report.artifact.getvalue()
    assert result['evidence_sha256'] == study.sha(path.artifact.getvalue())
    assert json.loads(report.artifact.getvalue().split('Durable registered result:')[1].split('```json\n')[1].split('\n```')[0]) == json.loads(study.canonical(result))
    with pytest.raises(study.BorderStudyError, match='evidence_already_exists'):
        study.run_study()


@pytest.mark.parametrize('tail', ['junk', 'empty_member', 'additional_member'])
def test_trailing_junk_empty_and_additional_members_refuse(monkeypatch, stub_public_entry, tail):
    path, pairs, _, expected = gzip_pair_fixture(monkeypatch, stub_public_entry)
    suffix = b'junk' if tail == 'junk' else study.compress_member(b'' if tail == 'empty_member' else b'{}\n')
    path.artifact = MemoryArtifact(path.artifact.getvalue()+suffix)
    with pytest.raises(study.BorderStudyError, match='gzip_integrity_error|evidence_prefix_mismatch'):
        study.replay_evidence((pairs[0].sha256,), expected)


def test_wrong_order_and_same_length_earlier_member_mutation_detected(monkeypatch, stub_public_entry):
    path, pairs, descriptors, expected = gzip_pair_fixture(monkeypatch, stub_public_entry, 2)
    original = path.artifact.getvalue()
    raw_members = [original[d['offset']:d['offset']+d['compressed_bytes']] for d in descriptors]
    path.artifact = MemoryArtifact(raw_members[0]+raw_members[2]+raw_members[1])
    with pytest.raises(study.BorderStudyError, match='evidence_seal_order_mismatch'):
        study.replay_evidence(tuple(p.sha256 for p in pairs), expected)
    altered = bytearray(original)
    altered[4] ^= 1  # Legal changed gzip MTIME: all plain payload seals still match.
    path.artifact = MemoryArtifact(bytes(altered))
    with pytest.raises(study.BorderStudyError, match='artifact_digest_mismatch'):
        study.replay_evidence(tuple(p.sha256 for p in pairs), expected)


def test_header_trailer_count_in_compressed_cap_and_malformed_pair_shape(monkeypatch):
    raw = b'{}\n'
    compressed = study.compress_member(raw)
    monkeypatch.setattr(study, 'CAPS', dict(study.CAPS, max_compressed_member_bytes=len(compressed)))
    assert study.compress_member(raw) == compressed
    monkeypatch.setattr(study, 'CAPS', dict(study.CAPS, max_compressed_member_bytes=len(compressed)-1))
    with pytest.raises(study.BorderStudyError, match='compressed_member_cap'):
        study.compress_member(raw)
    with pytest.raises(study.BorderStudyError, match='member_pair_shape_mismatch'):
        study.pair_member(b'{"payload_json":null,"sha256":"x"}\n')


def test_final_compressed_digest_veto_preserves_RUNNING_and_forbids_result(monkeypatch, stub_public_entry):
    A, B, sealed = run_stub_pair(stub_public_entry)
    monkeypatch.setattr(study, 'SCENES', (stub_public_entry.scene,))
    path, report = MemoryPath(), MemoryReport(report_content())
    monkeypatch.setattr(study, 'EVIDENCE_PATH', path)
    monkeypatch.setattr(study, 'REPORT_PATH', report)
    if study.os.name == 'nt':
        import msvcrt
        monkeypatch.setattr(msvcrt, 'locking', lambda *_: None)
    monkeypatch.setattr(study.os, 'fsync', lambda *_: None)
    monkeypatch.setattr(study, 'render_rgb', lambda *_: stub_public_entry.image)
    monkeypatch.setattr(study, 'trace_arm', lambda *a: A if a[2] == 'A' else B)
    monkeypatch.setattr(study, 'seal_pair', lambda *_: sealed)
    monkeypatch.setattr(study, 'evaluate_pair', lambda *_: {'stub_only': True})
    monkeypatch.setattr(study, 'artifact_digest', lambda: '0'*64)
    with pytest.raises(study.BorderStudyError, match='evidence_mutated_during_evaluation'):
        study.run_study()
    assert 'GUI_BORDER_EXECUTION_STATE: RUNNING' in report.artifact.getvalue()
    assert 'Durable registered result:' not in report.artifact.getvalue()
    assert len(gzip.decompress(path.artifact.getvalue()).splitlines()) == 2
    assert not study._STUDY_LOCK.locked() and not study._WRITER_LOCK.locked()


def test_total_compressed_ceiling_is_frozen_before_registered_rendering(monkeypatch):
    monkeypatch.setattr(study, 'MAX_EVIDENCE_BYTES', study.MAX_EVIDENCE_BYTES+1)
    monkeypatch.setattr(study, 'render_rgb', lambda *_: pytest.fail('render before frozen-cap check'))
    with pytest.raises(study.BorderStudyError, match='registry_or_policy_drift'):
        study.run_study()
