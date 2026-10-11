"""Pure CAM-MD renderer/contracts; no actual camera fitting or study run."""
from dataclasses import asdict, replace
import copy
import io
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from scripts.evaluation import run_camera_mixed_depth as camera_study


def test_nearest_finite_ray_hit_occlusion_and_parallel_miss():
    faces = ({'axis': 2, 'value': 5., 'side': -1, 'bounds': [[-2., 2.], [-2., 2.]]},
             {'axis': 2, 'value': 2., 'side': -1, 'bounds': [[-.5, .5], [-.5, .5]]})
    rays = np.array(((0., 0., 1.), (.3, 0., 1.), (0., 1., 0.), (0., 0., -1.)))
    owners, depth, points = camera_study.intersect_rays((0., 0., 0.), rays, faces)
    assert owners.tolist() == [1, 0, -1, -1]
    assert depth[:2].tolist() == [2., 5.] and np.isinf(depth[2:]).all()
    assert np.array_equal(points[0], [0., 0., 2.])
    assert np.array_equal(points[1], [1.5, 0., 5.])


def test_box_six_axis_orthogonal_faces_and_nearest_boundary():
    recipe = json.loads(camera_study.SCENES[1].recipe_json)
    faces = camera_study.face_planes(recipe)
    assert len(faces) == 2+6*len(recipe['boxes'])
    first_box = faces[2:8]
    assert {(face['axis'], face['side']) for face in first_box} == {(a, s) for a in range(3) for s in (-1, 1)}
    owners, depth, point = camera_study.intersect_rays((-2., 1., 0.), np.array([[0., 0., 1.]]), first_box)
    assert owners.tolist() == [4] and depth.tolist() == [3.]
    assert np.array_equal(point[0], [-2., 1., 3.])


def test_world_uv_translation_period_and_integer_lattice_edges():
    points = np.array(((.125, .375, 9.), (1.125, -1.625, 9.), (1., .375, 9.), (.4, 2., 9.)))
    rgb, seams = camera_study.native_surface_rgb(points, 2, -1)
    assert np.array_equal(rgb[0], rgb[1])
    assert seams.tolist() == [False, False, True, True]
    assert np.array_equal(rgb[2], [27., 27., 27.])


def test_independent_projection_line_directions_match_ray_basis():
    recipe = json.loads(camera_study.SCENES[1].recipe_json)
    rotation = camera_study.literal_rotation(recipe['yaw'], recipe['pitch'], recipe['roll'])
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    start = np.array([-2., 2., 6.])
    for axis in range(3):
        end = start+np.eye(3)[axis]*.3
        a, b = camera_study.project_world([start, end], recipe)
        direction = rotation[:, axis]
        vp = np.array([recipe['focal']*direction[0]+recipe['principal'][0]*direction[2],
                       -recipe['focal']*direction[1]+recipe['principal'][1]*direction[2], direction[2]])
        line = np.cross(np.r_[a, 1.], np.r_[b, 1.])
        assert abs(np.dot(line, vp)) < 1e-7


def test_crop_is_native_pixel_subarray_with_translated_principal_point():
    scene = camera_study.SCENES[3]
    recipe = json.loads(scene.recipe_json)
    full_recipe = dict(recipe, crop=None)
    full = camera_study.physical_render(full_recipe)[0]
    cropped = camera_study.render_rgb(scene)
    x, y, w, h = recipe['crop']
    assert np.array_equal(cropped, full[y:y+h, x:x+w])
    truth = camera_study.truth_camera(scene)
    assert truth['principal'] == [282., 214.] and truth['shape'] == [416, 560]


def test_ray_rendered_family_has_real_multiple_depths_and_visible_occlusion():
    recipe = json.loads(camera_study.SCENES[2].recipe_json)
    rgb, owners, depth, lattice = camera_study.physical_render(recipe)
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    assert len(set(owners[owners >= 0].tolist())) >= 3
    assert np.ptp(depth[np.isfinite(depth)]) > 1.
    assert lattice.any()
    no_foreground = dict(recipe, boxes=recipe['boxes'][:-1])
    _, old_owners, old_depth, _ = camera_study.physical_render(no_foreground)
    newly_near = depth < old_depth-1e-7
    assert newly_near.sum() > 100
    assert np.isfinite(depth[newly_near]).all()


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Real optimizer control flow, stub objective/detector/reduced ranges."""
    calibration = camera_study.calibration
    image = np.full((80, 100, 3), 120, np.uint8)
    segments = np.array([[10., 10.+i, 80., 14.+i] for i in range(12)])
    calls = []
    genuine_np = calibration.np
    class ReducedRanges:
        def __getattr__(self, name):
            return getattr(genuine_np, name)
        def arange(self, start, *args, **kwargs):
            return np.array([float(start)])
    monkeypatch.setattr(calibration, 'np', ReducedRanges())
    monkeypatch.setattr(calibration, 'detect_segments', lambda *a, **k: segments.copy())
    def objective(*args, **kwargs):
        bound = camera_study._SCORE_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        point = bound.arguments['principal_point']
        calls.append((args, kwargs))
        return 100.+(0. if point is None else point[0]-bound.arguments['width']/2.)
    monkeypatch.setattr(calibration, '_score', objective)
    return SimpleNamespace(image=image, segments=segments, calls=calls)


def _wrapped_functions():
    calibration = camera_study.calibration
    return (calibration._score, calibration.enhance, calibration.detect_segments,
            calibration.estimate_camera, calibration.refine_optics, camera_study.gui_mask.create_gui_mask)


def test_stub_parity_all_stage_sites_exact_calls_and_input_arrays(stub_pipeline):
    image = stub_pipeline.image
    original = image.copy()
    before = _wrapped_functions()
    untraced = camera_study.calibration.estimate_image_calibrations({'unit': image})['unit']
    stub_pipeline.calls.clear()
    result = camera_study.trace_rgb(image, 'unit')
    assert result['status'] == 'returned', result['reason']
    assert result['prediction'] == camera_study._clean(asdict(untraced))
    assert np.array_equal(original, image) and _wrapped_functions() == before
    assert len(stub_pipeline.calls) == result['trace']['score_events'] > 0
    assert {event['site'] for event in result['trace']['scores']} == {site['tag'] for site in camera_study.PINNED_SITES}
    assert {'coarse', 'coarse-full-rescore', 'fine'} <= {event['site'] for event in result['trace']['scores']}
    for (args, kwargs), event in zip(stub_pipeline.calls, result['trace']['scores']):
        bound = camera_study._SCORE_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        assert np.array_equal(camera_study.decode_array(result['trace'], event['segments_sha256']), args[0])
        assert np.array_equal(camera_study.decode_array(result['trace'], event['weights_sha256']), args[1])
        assert event['camera'] == {key: camera_study._clean(bound.arguments[key]) for key in
            ('focal', 'yaw', 'pitch', 'width', 'height', 'threshold', 'roll', 'principal_point')}
    assert result['trace']['trace_seconds'] >= 0 and result['trace']['score_delegate_seconds'] >= 0


def test_exact_delegate_arg_and_return_identity_once():
    arg, kwarg, returned = object(), object(), object()
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return returned
    assert camera_study._call_once(original, (arg,), {'option': kwarg}) is returned
    assert len(calls) == 1 and calls[0][0][0] is arg and calls[0][1]['option'] is kwarg


def test_all_pinned_multiline_spans_classify_and_ambiguous_unknown_refuse():
    codes = {'estimate_camera': camera_study._PINNED_CAMERA_CODE, 'refine_optics': camera_study._PINNED_REFINE_CODE}
    assert any(site['end'] > site['start'] for site in camera_study.PINNED_SITES)
    for site in camera_study.PINNED_SITES:
        for line in range(site['start'], site['end']+1):
            assert camera_study.classify_score_site(codes[site['function']], line) == site
    first = camera_study.PINNED_SITES[0]
    with pytest.raises(camera_study.CameraStudyError, match='unknown_or_ambiguous'):
        camera_study.classify_score_site(codes[first['function']], first['start'], (first, first))
    with pytest.raises(camera_study.CameraStudyError, match='unknown_or_ambiguous'):
        camera_study.classify_score_site(codes[first['function']], -1)
    with pytest.raises(camera_study.CameraStudyError, match='unexpected_score_caller'):
        camera_study.classify_score_site(test_all_pinned_multiline_spans_classify_and_ambiguous_unknown_refuse.__code__, 1)
    source = camera_study.CALIBRATION_PATH.read_text(encoding='utf-8')
    with pytest.raises(camera_study.CameraStudyError, match='ast_score_site_drift'):
        camera_study.score_call_sites(source.replace('value = _score(coarse,', 'value = not_score(coarse,'))


def test_source_and_runtime_drift_refuse_before_install(monkeypatch):
    monkeypatch.setattr(camera_study.cv2, '__version__', 'unit-mutation')
    monkeypatch.setattr(camera_study, '_install', lambda *a: pytest.fail('stale version installed trace'))
    with pytest.raises(camera_study.CameraStudyError, match='dependency_or_runtime_drift'):
        camera_study.trace_rgb(np.zeros((8, 8, 3), np.uint8), 'unit')


@pytest.mark.parametrize('array_name', ['segments', 'weights'])
def test_segment_or_weight_mutation_detected_and_snapshots_do_not_alias(stub_pipeline, monkeypatch, array_name):
    calibration = camera_study.calibration
    def mutate(*args, **kwargs):
        value = args[0 if array_name == 'segments' else 1]
        value.flat[0] += 1
        return 100.
    monkeypatch.setattr(calibration, '_score', mutate)
    before = _wrapped_functions()
    result = camera_study.trace_rgb(stub_pipeline.image, 'unit')
    assert result['status'] == 'trace_refusal' and result['reason'] == 'trace_array_mutated'
    assert result['prediction'] is None and not result['trace']['trace_complete']
    assert _wrapped_functions() == before
    event = result['trace']['scores'][0]
    snapshot = camera_study.decode_array(result['trace'], event['segments_sha256' if array_name == 'segments' else 'weights_sha256'])
    assert not snapshot.flags.writeable


def test_array_snapshot_mismatch_and_independent_bytes():
    recorder = camera_study.TraceRecorder(camera_study.TracePolicy())
    array = np.arange(12.).reshape(3, 4)
    digest = recorder.array(array, segments=True)
    saved = recorder.arrays[digest]['bytes_hex']
    array[0, 0] = 99.
    assert recorder.arrays[digest]['bytes_hex'] == saved
    with pytest.raises(camera_study.CameraStudyError, match='array_mutated'):
        recorder.array(array, segments=True)
    result = {'arrays': copy.deepcopy(recorder.arrays)}
    result['arrays'][digest]['bytes_hex'] = '00'+saved[2:]
    result['arrays'][digest]['shape'] = [12]
    with pytest.raises(camera_study.CameraStudyError, match='snapshot_digest_mismatch'):
        camera_study.decode_array(result, digest)


def test_score_cap_boundary_exact_and_next_event_refuses(stub_pipeline):
    default = camera_study.trace_rgb(stub_pipeline.image, 'unit')
    count = default['trace']['score_events']
    assert count > 0 and default['status'] == 'returned'
    exact = camera_study.trace_rgb(stub_pipeline.image, 'unit', policy=replace(camera_study.TracePolicy(), max_score_events=count))
    assert exact['status'] == 'returned' and exact['trace']['score_events'] == count
    over = camera_study.trace_rgb(stub_pipeline.image, 'unit', policy=replace(camera_study.TracePolicy(), max_score_events=count-1))
    assert over['status'] == 'trace_refusal' and over['reason'] == 'trace_event_cap'
    assert over['prediction'] is None and not over['trace']['trace_complete']
    assert over['trace']['score_events'] == count-1


@pytest.mark.parametrize('field,reason', [('max_stage_events', 'trace_stage_cap'),
    ('max_array_bytes', 'trace_array_cap'), ('max_trace_bytes', 'trace_payload_cap'),
    ('max_segments', 'trace_segment_cap')])
def test_other_trace_caps_fail_closed(stub_pipeline, field, reason):
    before = _wrapped_functions()
    result = camera_study.trace_rgb(stub_pipeline.image, 'unit', policy=replace(camera_study.TracePolicy(), **{field: 0}))
    assert result['status'] == 'trace_refusal' and result['reason'] == reason
    assert result['prediction'] is None and not result['trace']['trace_complete']
    assert _wrapped_functions() == before


@pytest.mark.parametrize('exception', [RuntimeError('unit-failure'), KeyboardInterrupt()])
def test_every_wrapper_restored_on_unexpected_exception_or_interruption(stub_pipeline, monkeypatch, exception):
    def failure(*a, **k):
        raise exception
    monkeypatch.setattr(camera_study.calibration, 'detect_segments', failure)
    before = _wrapped_functions()
    with pytest.raises(type(exception)) as caught:
        camera_study.trace_rgb(stub_pipeline.image, 'unit')
    assert caught.value is exception and _wrapped_functions() == before
    assert camera_study._TRACE_LOCK.acquire(blocking=False)
    camera_study._TRACE_LOCK.release()


def test_partial_install_restores_every_original(stub_pipeline, monkeypatch):
    before = _wrapped_functions()
    original = camera_study._install
    calls = []
    def partial(module, name, wrapper):
        original(module, name, wrapper)
        calls.append(name)
        if len(calls) == 2:
            raise RuntimeError('unit-partial-install')
    monkeypatch.setattr(camera_study, '_install', partial)
    with pytest.raises(RuntimeError, match='partial-install'):
        camera_study.trace_rgb(stub_pipeline.image, 'unit')
    assert _wrapped_functions() == before and len(calls) == 2


def test_nested_reentrant_installation_rejected(stub_pipeline, monkeypatch):
    def nested(*a, **k):
        camera_study.trace_rgb(stub_pipeline.image, 'nested')
        pytest.fail('nested trace returned')
    monkeypatch.setattr(camera_study.calibration, '_score', nested)
    before = _wrapped_functions()
    result = camera_study.trace_rgb(stub_pipeline.image, 'unit')
    assert result['status'] == 'trace_refusal' and result['reason'] == 'trace_reentrancy_or_concurrency'
    assert _wrapped_functions() == before


def test_concurrent_installation_rejected_before_wrapping(stub_pipeline, monkeypatch):
    entered, released = threading.Event(), threading.Event()
    def owner():
        camera_study._TRACE_LOCK.acquire()
        entered.set()
        released.wait(10)
        camera_study._TRACE_LOCK.release()
    thread = threading.Thread(target=owner)
    thread.start()
    assert entered.wait(2)
    monkeypatch.setattr(camera_study, '_install', lambda *a: pytest.fail('concurrent install occurred'))
    try:
        with pytest.raises(camera_study.CameraStudyError, match='reentrancy_or_concurrency'):
            camera_study.trace_rgb(stub_pipeline.image, 'unit')
    finally:
        released.set()
        thread.join(2)
    assert not thread.is_alive()


def test_cross_thread_wrapper_use_invalidates_owner_trace(stub_pipeline, monkeypatch):
    failures = []
    def foreign():
        try:
            camera_study.calibration._score(stub_pipeline.segments, np.ones(12), 40, 0, 0, 100, 80, 3)
        except camera_study.CameraStudyError as error:
            failures.append(error.reason)
    def detection(*a, **k):
        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
        return stub_pipeline.segments.copy()
    monkeypatch.setattr(camera_study.calibration, 'detect_segments', detection)
    before = _wrapped_functions()
    result = camera_study.trace_rgb(stub_pipeline.image, 'unit')
    assert failures == ['trace_cross_thread_call']
    assert result['status'] == 'trace_refusal' and result['reason'] == 'trace_cross_thread_call'
    assert _wrapped_functions() == before


def test_genuine_rgb_pipeline_parity_on_fixed_development_rgb_no_truth(monkeypatch):
    # Lead explicitly authorizes this real calibration QA pair, not CAM-MD arms.
    image = camera_study.legacy.render_rgb('identifiable')[0]
    original = image.copy()
    before = _wrapped_functions()
    monkeypatch.setattr(camera_study, 'truth_camera', lambda *a: pytest.fail('parity accessed truth'))
    monkeypatch.setattr(camera_study, 'evaluate_prediction', lambda *a: pytest.fail('parity evaluated'))
    untraced = camera_study.calibration.estimate_image_calibrations({'dev': image})['dev']
    assert untraced is not None
    traced = camera_study.trace_rgb(image, 'dev')
    assert traced['status'] == 'returned', traced['reason']
    assert traced['prediction'] == camera_study._clean(asdict(untraced))
    assert np.array_equal(image, original) and _wrapped_functions() == before
    assert traced['trace']['score_events'] > 70000
    assert traced['trace']['trace_complete'] and traced['trace']['scores']
    camera_study.replay_trace(traced)


def test_rgb_only_trace_signature_and_ordinary_refusal_preserve_path(monkeypatch):
    import inspect
    assert list(inspect.signature(camera_study.trace_rgb).parameters) == ['image', 'frame_id', 'policy']
    before = _wrapped_functions()
    image = np.zeros((16, 16, 3), np.uint8)
    original = image.copy()
    result = camera_study.trace_rgb(image, 'flat-unit')
    assert result['status'] == 'ordinary_refusal' and result['prediction'] is None
    assert result['trace']['trace_complete'] and not result['trace']['scores']
    assert _wrapped_functions() == before and np.array_equal(image, original)


def test_unexpected_score_call_site_refuses_before_delegate(stub_pipeline, monkeypatch):
    calls = []
    monkeypatch.setattr(camera_study.calibration, '_score', lambda *a, **k: calls.append(1))
    def outside(*a, **k):
        camera_study.calibration._score(stub_pipeline.segments, np.ones(12), 40, 0, 0, 100, 80, 3)
        return stub_pipeline.segments
    monkeypatch.setattr(camera_study.calibration, 'detect_segments', outside)
    result = camera_study.trace_rgb(stub_pipeline.image, 'unit')
    assert result['status'] == 'trace_refusal' and result['reason'] == 'trace_unexpected_score_caller'
    assert not calls and result['prediction'] is None


@pytest.mark.parametrize('relative', json.loads(camera_study.PREREGISTRATION_JSON)['dependencies']['source_sha256'])
def test_upstream_hash_drift_refuses_before_install(relative, monkeypatch):
    target = camera_study.SOURCE_ROOT/relative
    original = camera_study.Path.read_bytes
    def altered(path):
        return original(path)+(b'\n# unit-source-drift\n' if path == target else b'')
    monkeypatch.setattr(camera_study.Path, 'read_bytes', altered)
    monkeypatch.setattr(camera_study, '_install', lambda *a: pytest.fail('drift installed wrappers'))
    with pytest.raises(camera_study.CameraStudyError, match='dependency_or_runtime_drift'):
        camera_study.trace_rgb(np.zeros((8, 8, 3), np.uint8), 'unit')


def test_complete_prediction_seal_replay_and_tamper_precede_evaluator(stub_pipeline, monkeypatch):
    result = camera_study.trace_rgb(stub_pipeline.image, 'unit')
    scene = camera_study.SCENES[0]
    sealed = camera_study.seal_prediction(scene, result)
    assert camera_study.verify_prediction(scene, sealed) == result
    monkeypatch.setattr(camera_study, 'truth_camera', lambda *a: pytest.fail('unsealed prediction read truth'))
    monkeypatch.setattr(camera_study, 'attribution_masks', lambda *a: pytest.fail('unsealed prediction read labels'))
    with pytest.raises(camera_study.CameraStudyError, match='prediction_seal_mismatch'):
        camera_study.evaluate_prediction(scene, replace(sealed, sha256='0'*64))
    with pytest.raises(camera_study.CameraStudyError, match='recipe_binding_mismatch'):
        camera_study.evaluate_prediction(camera_study.SCENES[1], sealed)
    damaged = copy.deepcopy(result)
    damaged['trace']['scores'][0]['weights_sha256'] = damaged['trace']['scores'][0]['segments_sha256']
    with pytest.raises(camera_study.CameraStudyError, match='weight_segment_mismatch'):
        camera_study.seal_prediction(scene, damaged)


def test_overlapping_attribution_preserves_joint_categories_without_exclusive_claim():
    mask = np.zeros((20, 20), bool)
    mask[10, 2:18] = True
    segments = np.array([[2., 10., 17., 10.], [2., 3., 17., 3.]])
    result = camera_study.segment_attribution(segments, {'gui_border': mask, 'compositor_strip': mask,
                                                        'projected_pencil': np.zeros_like(mask)})
    assert result['overlapping']['gui_border']['segment_count'] == 1
    assert result['overlapping']['compositor_strip']['segment_count'] == 1
    assert result['joint_bitmask_counts'][3] == 1 and result['joint_bitmask_counts'][0] == 1
    assert not result['exclusive_partition_claim'] and not result['causal_attribution_claim']


def test_legacy_true_global_joins_selected_and_shifted_rectangle_joins_rejected():
    scene = camera_study.SCENES[-1]
    masks = camera_study.attribution_masks(scene)
    # Independent literal expected positions from the renderer's global strips.
    true_joins = [288, 336, 384, 432, 480, 528, 576, 624]
    false_shifted = [318, 366, 414, 462, 510, 558, 606]
    segments = np.array([[float(x), 150., float(x), 400.] for x in true_joins+false_shifted])
    measured = camera_study.segment_attribution(segments, masks)['overlapping']['compositor_strip']
    assert measured['sample_fraction'] == [1.]*len(true_joins)+[0.]*len(false_shifted)
    assert measured['segment_count'] == len(true_joins)
    strip = masks['compositor_strip']
    assert strip[200, 287] and strip[200, 288]
    assert not strip[200, 270] and not strip[200, 650]
    assert not strip[80, 288] and not strip[490, 288]


@pytest.mark.parametrize('rectangle,expected', [
    ([0, 10, 160, 54], [48, 96, 144]),
    ([48, 10, 96, 54], []),  # neither perimeter jump is internal
    ([47, 10, 97, 54], [48, 96]),
    ([49, 10, 95, 54], []),
    ([-7, 10, 50, 54], [48]),  # global x=0 has no previous image pixel
    ([143, 10, 170, 54], [144]),
    ([160, 10, 200, 54], []),
    ([-50, 10, -1, 54], []),
    ([0, -4, 160, 2], [48, 96, 144]),
    ([0, 70, 160, 90], []),
])
def test_compositor_joins_require_both_in_rectangle_image_pixels(rectangle, expected):
    recipe = json.loads(camera_study.SCENES[-1].recipe_json)
    recipe['shape'] = [64, 160]
    recipe['face_rectangle'] = rectangle
    scene = replace(camera_study.SCENES[-1], identity='unit-strip-edges', recipe_json=json.dumps(recipe))
    mask = camera_study.attribution_masks(scene)['compositor_strip']
    # Independently construct the two-pixel true join footprint plus the existing
    # 3-pixel proximity dilation; no production join-finding algorithm is reused.
    oracle = np.zeros((64, 160), bool)
    for x in expected:
        y0, y1 = max(0, rectangle[1]), min(64, rectangle[3])
        oracle[max(0, y0-3):min(64, y1+3), max(0, x-4):min(160, x+4)] = True
    assert np.array_equal(mask, oracle)


def test_candidate_max_is_not_retained_refinement_camera_and_selected_transitions_are_captured(monkeypatch):
    monkeypatch.setattr(camera_study.calibration, '_score', lambda *a, **k: pytest.fail('semantics test calibrated'))
    def camera(focal):
        return {'focal': focal, 'yaw': 27., 'pitch': -13., 'width': 640, 'height': 480,
                'threshold': 3., 'roll': 0., 'principal_point': None}
    def event(tag, focal, score, winner=None):
        return {'site': tag, 'camera': camera(focal), 'score': score,
                'winner': {'best_before': winner}}
    final = {'focal_pixels': 411., 'yaw_degrees': 27., 'pitch_degrees': -13.,
             'roll_degrees': 0., 'principal_point': None}
    result = {'status': 'returned', 'prediction': final, 'trace': {'trace_complete': True,
        'scores': [event('coarse', 410., 100.), event('coarse-full-rescore', 410., 80., [100., 410., 27., -13.]),
                   event('fine', 411., 81.), event('refine-base', 411., 81.),
                   event('refine-principal', 450., 82.)],
        # 82 < 81*1.02: actual refine API rejects the highest tested optics;
        # its returned focal can differ, while estimate_camera retains fine411.
        'stages': [{'stage': 'refine_optics', 'output': [0., None, 27., -13., False, 450.]}]}}
    evidence = camera_study.stage_camera_evidence(result)
    maximum = evidence['candidate_max']['refine-principal']
    assert maximum['camera']['focal'] == 450. and not maximum['optimizer_retention_claim']
    transitions = evidence['selected_transitions']
    assert transitions['coarse_winner_to_full_rescore']['selected_camera']['focal'] == 410.
    assert transitions['coarse_winner_to_full_rescore']['coarse_native_score'] == 100.
    assert transitions['coarse_winner_to_full_rescore']['full_rescore'] == 80.
    assert transitions['fine_winner_to_refinement']['selected_fine_camera']['focal'] == 411.
    assert not transitions['fine_winner_to_refinement']['refinement_accepted']
    assert transitions['final_prediction'] == final
    accepted = copy.deepcopy(result)
    accepted['trace']['stages'][0]['output'][4] = True
    accepted['prediction']['focal_pixels'] = 450.
    evidence = camera_study.stage_camera_evidence(accepted)
    assert evidence['selected_transitions']['fine_winner_to_refinement']['refinement_accepted']
    assert evidence['selected_transitions']['final_prediction']['focal_pixels'] == 450.


def test_metric_literal_zero_and_simple_focal_principal_arithmetic_no_scene_evaluation():
    truth = {'focal': 100., 'yaw': 0., 'pitch': 0., 'roll': 0., 'principal': [40., 30.], 'shape': [60, 80]}
    prediction = {'focal_pixels': 100., 'yaw_degrees': 0., 'pitch_degrees': 0.,
                  'roll_degrees': 0., 'principal_point': [40., 30.]}
    zero = camera_study.camera_metrics(prediction, truth)
    assert zero['focal_relative_error'] == zero['orientation_degrees_mod90'] == zero['principal_error_over_diagonal'] == 0.
    assert zero['projection_direction_error_pixels']['max'] < 1e-12 and zero['diagnostic_pass']
    shifted = camera_study.camera_metrics(dict(prediction, principal_point=[41., 30.]), truth)
    assert shifted['principal_error_over_diagonal'] == .01  # diagonal=100
    assert shifted['projection_direction_error_pixels']['median'] == pytest.approx(1.)
    scaled = camera_study.camera_metrics(dict(prediction, focal_pixels=102.), truth)
    assert scaled['focal_relative_error'] == pytest.approx(.02)
    assert scaled['projection_direction_error_pixels']['max'] == pytest.approx(.8)
    assert scaled['diagnostic_pass']
    assert not camera_study.camera_metrics(dict(prediction, focal_pixels=106.), truth)['diagnostic_pass']


def test_gui_contamination_masked_pixels_only_and_low_texture_has_no_texture_grid():
    occluding = camera_study.render_rgb(camera_study.SCENES[2])
    contaminated = camera_study.render_rgb(camera_study.SCENES[5])
    mask = camera_study.create_gui_mask(*occluding.shape[:2])
    assert np.array_equal(occluding[~mask], contaminated[~mask])
    assert not np.array_equal(occluding[mask], contaminated[mask])
    rgb, owners, _, seams = camera_study.physical_render(json.loads(camera_study.SCENES[4].recipe_json))
    assert not seams.any() and (rgb[owners >= 0] == 126).all()


def test_legacy_is_not_physical_truth_or_aggregate_and_source_hashes_are_frozen():
    prereg = json.loads(camera_study.PREREGISTRATION_JSON)
    assert prereg['legacy_excluded_from_physical_aggregate']
    assert not camera_study.SCENES[-1].physical
    with pytest.raises(camera_study.CameraStudyError, match='outside_physical_camera_aggregate'):
        camera_study.truth_camera(camera_study.SCENES[-1])
    for scene in camera_study.SCENES:
        assert camera_study._sha(camera_study.render_rgb(scene).tobytes()) == prereg['stimulus_sha256'][scene.identity]
    assert prereg['diagnostic_tolerances'] == {'focal_relative': .05, 'orientation_deg_mod90': 1., 'principal_diagonal': .01}
    assert not prereg['production_m3_gates_changed']


class MemoryCameraEvidence:
    name = 'CAM-MD-01-evidence-2026-10-05.jsonl'
    def __init__(self, content=None):
        self.content = content
    def exists(self):
        return self.content is not None
    is_file = exists
    def read_bytes(self):
        if self.content is None:
            raise FileNotFoundError(self.name)
        return self.content
    def open(self, mode):
        if mode == 'xb':
            if self.exists():
                raise FileExistsError(self.name)
            self.content = b''
        elif mode != 'r+b' or not self.exists():
            raise FileNotFoundError(self.name)
        owner = self
        class Handle(io.BytesIO):
            def fileno(self):
                return 54322
            def flush(self):
                owner.content = self.getvalue()
            def close(self):
                if not self.closed:
                    self.flush()
                super().close()
        return Handle(self.content)


def test_evidence_full_trace_payload_fsync_replay_tamper_and_prefix_retention(stub_pipeline, monkeypatch):
    result = camera_study.trace_rgb(stub_pipeline.image, 'unit')
    sealed = camera_study.seal_prediction(camera_study.SCENES[0], result)
    artifact = MemoryCameraEvidence((camera_study._json(camera_study.evidence_header())+'\n').encode())
    flushed = []
    monkeypatch.setattr(camera_study, 'EVIDENCE_PATH', artifact)
    monkeypatch.setattr(camera_study.os, 'fsync', lambda fd: flushed.append(fd))
    digest = camera_study.append_evidence(camera_study.SCENES[0], sealed, ())
    assert flushed == [54322]
    assert camera_study.replay_evidence(artifact.content, (sealed.sha256,)) == digest
    retained = artifact.content
    assert json.loads(retained.splitlines()[1])['payload_json'] == sealed.payload_json
    with pytest.raises(camera_study.CameraStudyError, match='order_mismatch'):
        camera_study.append_evidence(camera_study.SCENES[2], sealed, (sealed.sha256,))
    assert artifact.content == retained
    with pytest.raises(camera_study.CameraStudyError, match='truncated'):
        camera_study.replay_evidence(retained[:-1], (sealed.sha256,))
    records = [json.loads(line) for line in retained.splitlines()]
    records[1]['payload_json'] += ' '
    with pytest.raises(camera_study.CameraStudyError, match='prediction_seal_mismatch'):
        camera_study.replay_evidence(('\n'.join(camera_study._json(r) for r in records)+'\n').encode(), (sealed.sha256,))


def test_all_predictions_and_trace_artifact_precede_first_evaluator(stub_pipeline, monkeypatch):
    unit = camera_study.trace_rgb(stub_pipeline.image, 'unit')
    artifact = MemoryCameraEvidence()
    stages = []
    monkeypatch.setattr(camera_study, 'EVIDENCE_PATH', artifact)
    monkeypatch.setattr(camera_study, '_claim_report', lambda: stages.append('claimed'))
    monkeypatch.setattr(camera_study.os, 'fsync', lambda *a: stages.append('fsync'))
    monkeypatch.setattr(camera_study, 'render_rgb', lambda *a: stub_pipeline.image)
    monkeypatch.setattr(camera_study, 'trace_rgb', lambda *a: copy.deepcopy(unit))
    class EvaluationReached(BaseException):
        pass
    def no_truth(scene, sealed):
        records = [json.loads(line) for line in artifact.content.splitlines()]
        assert len(records) == len(camera_study.SCENES)+1
        seals = tuple(record['sha256'] for record in records[1:])
        assert camera_study.replay_evidence(artifact.content, seals)
        assert stages[0] == 'claimed' and stages.count('fsync') == len(camera_study.SCENES)+1
        raise EvaluationReached()
    monkeypatch.setattr(camera_study, 'evaluate_prediction', no_truth)
    with pytest.raises(EvaluationReached):
        camera_study.run_study()
    # A new invocation refuses before report claim/render; the entire prefix remains.
    retained = artifact.content
    with pytest.raises(camera_study.CameraStudyError, match='evidence_already_exists'):
        camera_study.run_study()
    assert artifact.content == retained
