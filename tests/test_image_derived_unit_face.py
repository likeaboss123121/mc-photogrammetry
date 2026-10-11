"""Generated RGB and isolated upstream seams; no external evaluator fixtures."""
from dataclasses import FrozenInstanceError, asdict, replace
import ast
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_unit_face as adapter
from mc_photogrammetry.reconstruction import unit_face_estimator as estimator
from mc_photogrammetry.reconstruction.boundary_candidates import PlaneProposal, PlaneProposalResult
from mc_photogrammetry.reconstruction.calibration import CalibrationResult
from mc_photogrammetry.reconstruction.unit_face_lattice import WitnessProof
from mc_photogrammetry.vision.gui_mask import create_gui_mask


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/evaluation/run_unit_face_input.py'
spec = importlib.util.spec_from_file_location('unit_face_input_experiment', SCRIPT)
experiment = importlib.util.module_from_spec(spec)
# dataclasses resolves its module namespace through sys.modules.
import sys
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


@pytest.fixture(scope='module')
def generated():
    return experiment.render_rgb('identifiable')


def synthetic_upstream(image, *, pp=None, hole=True):
    """Unit-test-only seam; these optics never enter the controlled RGB run."""
    h, w = image.shape[:2]
    cal = CalibrationResult(700., 23., -14., w, h, 90, 120, .1, (30, 30, 30), 0., pp)
    mask = np.zeros((h, w), np.bool_)
    mask[90:480, 270:650] = True
    if hole:
        mask[200:225, 400:430] = False
    proposal = PlaneProposal(2, 1, ((270., 90.), (650., 90.), (650., 480.), (270., 480.)),
                             mask, .8, ('synthetic-unit-seam',), .5)
    return cal, proposal


def prepare(image, policy=adapter.InputPolicy(), *, pp=None, hole=True):
    cal, proposal = synthetic_upstream(image, pp=pp, hole=hole)
    frame = adapter.RgbFrame.snapshot(image, 'generated-unit-seam', policy)
    snapshot, order = adapter._select_proposal((proposal,), frame.shape, policy)
    return adapter._prepare(frame, cal, snapshot, order, policy)


@pytest.fixture(scope='module')
def prepared(generated):
    return prepare(generated[0])


def fake_upstream(monkeypatch, image, *, fail=None, duplicate=False):
    cal, proposal = synthetic_upstream(image)
    calls = []
    def calibration(frames):
        calls.append(('calibration', tuple(frames)))
        assert len(frames) == 1 and not next(iter(frames.values())).flags.writeable
        return {key: None if fail == 'calibration' else cal for key in frames}
    def proposals(rgb, usable, actual_cal, **kwargs):
        calls.append(('proposal', actual_cal))
        assert actual_cal is cal
        assert np.array_equal(usable, ~create_gui_mask(*image.shape[:2]))
        assert kwargs == {'max_candidates': 32}
        if fail == 'proposal':
            return PlaneProposalResult('abstain', 'no_structural_segments')
        return PlaneProposalResult('ok', 'synthetic-unit-seam', (proposal, proposal) if duplicate else (proposal,))
    monkeypatch.setattr(adapter.calibration, 'estimate_image_calibrations', calibration)
    monkeypatch.setattr(adapter.region_growth, 'propose_image_planes_v2', proposals)
    return calls


def test_native_crop_and_full_frame_ray_equivalence(prepared):
    p = prepared
    x0, y0, width, height = p.crop
    assert width * height == 65_536
    assert p.crop_rgb == p.frame.image()[y0:y0+height, x0:x0+width].tobytes()
    full_pp = p.full_calibration.principal_point or (p.frame.shape[1]/2, p.frame.shape[0]/2)
    assert p.cropped_calibration.principal_point == (full_pp[0]-x0, full_pp[1]-y0)
    assert p.cropped_calibration.focal_pixels == p.full_calibration.focal_pixels
    assert p.cropped_calibration.inliers == p.full_calibration.inliers
    assert p.cropped_calibration.per_axis_inliers == p.full_calibration.per_axis_inliers
    for index, parent in zip(p.support.generation_pixels, p.parent_pixels('generation')):
        cx, cy = index % width, index // width
        fx, fy = parent % p.frame.shape[1], parent // p.frame.shape[1]
        a = ((cx-p.context.principal_point[0])/p.context.focal_pixels,
             -(cy-p.context.principal_point[1])/p.context.focal_pixels, 1)
        b = ((fx-full_pp[0])/p.full_calibration.focal_pixels,
             -(fy-full_pp[1])/p.full_calibration.focal_pixels, 1)
        assert a == b
    assert not p.image().flags.writeable and not p.frame.image().flags.writeable
    with pytest.raises(FrozenInstanceError):
        p.crop = (0, 0, 1, 1)


def test_offcentre_principal_point_is_translated_not_recentered(generated):
    p = prepare(generated[0], pp=(389.25, 310.5))
    assert p.context.principal_point == (389.25-p.crop[0], 310.5-p.crop[1])
    assert p.context.focal_pixels == 700.


def test_holes_gui_roles_complete_stencils_and_original_replay(prepared):
    p = prepared
    shape = p.context.source_shape
    x0, y0, width, height = p.crop
    region = np.frombuffer(p.support.partition.region, np.uint8).reshape(shape)
    expected = np.frombuffer(p.proposal.mask, np.bool_).reshape(p.frame.shape)[y0:y0+height, x0:x0+width]
    assert np.array_equal(region, expected)
    assert not region[200-y0:225-y0, 400-x0:430-x0].any()
    gui = create_gui_mask(*p.frame.shape)
    assert not set(p.support.generation_pixels) & set(p.support.validation_pixels)
    assert not set(p.parent_pixels('generation')) & set(p.parent_pixels('validation'))
    assert p.counts.rejected_border and p.counts.rejected_excluded and p.counts.rejected_cross_role
    assert (p.counts.retained_samples + p.counts.rejected_border + p.counts.rejected_depth
            + p.counts.rejected_excluded + p.counts.rejected_cross_role == p.counts.witness_samples)
    for witness in p.support.witnesses:
        assert WitnessProof(**asdict(witness)) == witness
    for role in ('generation', 'validation'):
        for pixel, parent in zip(getattr(p.support, role+'_pixels'), p.parent_pixels(role)):
            assert not gui.flat[parent] and region.flat[pixel]
            assert pixel % width < width-1 and pixel // width < height-1
            assert np.array_equal(p.image().reshape(-1, 3)[pixel], p.frame.image().reshape(-1, 3)[parent])
    adapter._verify_prepared(p)


def test_source_snapshot_not_changed_by_later_caller_mutation(generated):
    image = generated[0].copy()
    frame = adapter.RgbFrame.snapshot(image, 'immutable', adapter.InputPolicy())
    original = frame.rgb
    image[:] = 0
    assert frame.rgb == original and frame.image().any()


def test_deterministic_selection_uses_no_rgb_or_fit_residual(generated):
    image = generated[0]
    _, p = synthetic_upstream(image)
    a = replace(p, strength=.7)
    b = replace(p, mask=np.roll(p.mask, 1, axis=1), strength=.8)
    first = adapter._select_proposal((a, b, p), image.shape[:2], adapter.InputPolicy())
    reverse = adapter._select_proposal((p, b, a), image.shape[:2], adapter.InputPolicy())
    assert first == reverse
    source = inspect.getsource(adapter._select_proposal)
    assert 'image' not in inspect.signature(adapter._select_proposal).parameters
    assert 'residual' not in source and 'validation' not in source


@pytest.mark.parametrize('field', ['source', 'full_calibration', 'proposal', 'crop', 'crop_rgb', 'context', 'support', 'policy', 'order'])
def test_forged_or_stale_bindings_never_reach_fitter(prepared, generated, monkeypatch, field):
    p = prepared
    if field == 'source':
        forged = replace(p, frame=replace(p.frame, rgb=b'\0'+p.frame.rgb[1:]))
    elif field == 'full_calibration':
        forged = replace(p, full_calibration=replace(p.full_calibration, focal_pixels=701.))
    elif field == 'proposal':
        forged = replace(p, proposal=replace(p.proposal, mask=b'\1'+p.proposal.mask[1:]))
    elif field == 'crop':
        forged = replace(p, crop=(p.crop[0]+1,) + p.crop[1:])
    elif field == 'crop_rgb':
        forged = replace(p, crop_rgb=b'\0'+p.crop_rgb[1:])
    elif field == 'context':
        forged = replace(p, context=replace(p.context, focal_pixels=701.))
    elif field == 'support':
        witness = object.__new__(WitnessProof)
        for key in witness.__slots__:
            object.__setattr__(witness, key, getattr(p.support.witnesses[0], key))
        object.__setattr__(witness, 'witness_sha256', '0'*64)
        # Immutable contract rejects ordinary construction; emulate a corrupted
        # deserialized object without invoking its constructor.
        forged_support = object.__new__(type(p.support))
        for key in p.support.__slots__:
            object.__setattr__(forged_support, key, getattr(p.support, key))
        object.__setattr__(forged_support, 'witnesses', (witness,))
        forged = replace(p, support=forged_support)
    elif field == 'policy':
        forged = replace(p, policy=replace(p.policy, max_fit_work=99999))
    else:
        forged = replace(p, proposal_order=('0'*64,))
    def forbidden(*args, **kwargs):
        pytest.fail('forged preparation called fitter')
    monkeypatch.setattr(adapter, 'estimate_unit_face', forbidden)
    with pytest.raises(adapter.PreparationError):
        adapter._fit_prepared(forged, generated[1])


@pytest.mark.parametrize('field', ['source', 'calibration', 'proposal', 'crop_rgb'])
def test_rehashing_one_forged_component_cannot_bypass_replay(prepared, field):
    if field == 'source':
        forged = replace(prepared, frame=replace(prepared.frame, sha256='0'*64))
    elif field == 'calibration':
        forged = replace(prepared, full_calibration=replace(prepared.full_calibration, focal_pixels=701.))
    elif field == 'proposal':
        forged = replace(prepared, proposal=replace(prepared.proposal, sha256='0'*64))
    else:
        forged = replace(prepared, crop_rgb=b'\0'+prepared.crop_rgb[1:])
    forged = replace(forged, binding_sha256=adapter._binding(forged))
    with pytest.raises(adapter.PreparationError):
        adapter._verify_prepared(forged)


@pytest.mark.parametrize('cap,reason', [
    ('max_frame_pixels', 'frame_pixel_cap'), ('max_crop_pixels', 'crop_pixel_cap'),
    ('max_proposals', 'proposal_cap'), ('max_source_pixels_per_role', 'role_source_pixel_cap'),
    ('max_witness_samples', 'witness_sample_cap'), ('max_witness_taps', 'witness_tap_cap'),
    ('max_dependency_memberships', 'dependency_cap_exceeded'),
    ('max_witness_bytes', 'witness_payload_cap_exceeded'),
])
def test_preparation_caps_atomic_no_fitter(generated, monkeypatch, cap, reason):
    fake_upstream(monkeypatch, generated[0], duplicate=cap == 'max_proposals')
    monkeypatch.setattr(adapter, 'estimate_unit_face', lambda *a, **k: pytest.fail('failed preparation called fitter'))
    policy = replace(adapter.InputPolicy(), **{cap: 1})
    result = adapter.estimate_image_unit_face(generated[0], generated[1], policy=policy)
    assert result.status == 'abstained' and result.reason == reason
    assert result.fit is None and result.prepared is None and not result.placement_supported


@pytest.mark.parametrize('failure,reason', [('calibration', 'image_calibration_failed'),
                                           ('proposal', 'image_proposal_no_structural_segments')])
def test_upstream_failure_never_calls_fitter(generated, monkeypatch, failure, reason):
    calls = fake_upstream(monkeypatch, generated[0], fail=failure)
    monkeypatch.setattr(adapter, 'estimate_unit_face', lambda *a, **k: pytest.fail('upstream failure called fitter'))
    result = adapter.estimate_image_unit_face(generated[0], generated[1])
    assert result.reason == reason and result.fit is None
    assert len(calls) == (1 if failure == 'calibration' else 2)


@pytest.mark.parametrize('policy,reason', [
    (adapter.InputPolicy(max_fit_work=1), 'root_work_preflight'),
    (adapter.InputPolicy(max_fit_work=2), 'search_work_exhausted'),
    (adapter.InputPolicy(max_fit_pixel_work=1), 'estimated_pixel_work_cap'),
])
def test_unchanged_fitter_work_caps_no_certificate(generated, monkeypatch, policy, reason):
    fake_upstream(monkeypatch, generated[0])
    result = adapter.estimate_image_unit_face(generated[0], generated[1], policy=policy)
    assert result.prepared is not None and result.fit is not None
    assert result.reason == reason and not result.fit.search_complete
    assert result.fit.representative is None and result.fit.cover == ()
    assert result.fit.distance_interval is None and result.fit.phase_intervals is None


def test_validation_mutation_preserves_generation_under_frozen_context(prepared, generated):
    p = prepared
    baseline = estimator.estimate_unit_face(p.image(), p.context, p.support, generated[1])
    # Non-vacuous: the full bounded search completed and retained actual
    # numerical candidates/cover. This small GUI-masked crop is ambiguous;
    # that gate prevents validation, and no successful veto is claimed here.
    assert baseline.search_complete and baseline.domain_search_complete
    assert baseline.candidates and baseline.cover and baseline.work_used > 0
    assert baseline.source_generation_sha256
    assert baseline.reason == 'scale_or_phase_ambiguous'
    changed = p.image().copy()
    changed.reshape(-1, 3)[list(p.support.validation_pixels)] = 255-changed.reshape(-1, 3)[list(p.support.validation_pixels)]
    assert np.array_equal(changed.reshape(-1, 3)[list(p.support.generation_pixels)],
                          p.image().reshape(-1, 3)[list(p.support.generation_pixels)])
    assert not np.array_equal(changed.reshape(-1, 3)[list(p.support.validation_pixels)],
                              p.image().reshape(-1, 3)[list(p.support.validation_pixels)])
    mutated = estimator.estimate_unit_face(changed, p.context, p.support, generated[1])
    assert mutated.status == 'abstained'
    assert mutated.search_complete and mutated.domain_search_complete
    assert len(mutated.candidates) == len(baseline.candidates) > 0
    assert mutated.source_generation_sha256 == baseline.source_generation_sha256
    assert mutated.source_validation_sha256 == baseline.source_validation_sha256 == ''
    assert mutated.cover == baseline.cover and mutated.distance_interval == baseline.distance_interval
    assert mutated.phase_intervals == baseline.phase_intervals and mutated.work_used == baseline.work_used
    assert mutated.actual_pixel_work == baseline.actual_pixel_work > 0
    for a, b in zip(baseline.candidates, mutated.candidates):
        assert (a.face_sha256, a.distance, a.phase, a.gain, a.bias, a.generation_rmse) == (
            b.face_sha256, b.distance, b.phase, b.gain, b.bias, b.generation_rmse)
    # RGB changed AFTER freezing upstream, so this is conditional independence.
    # The adapter itself must refuse changing its bound source snapshot.
    with pytest.raises(adapter.PreparationError):
        adapter._verify_prepared(replace(p, crop_rgb=changed.tobytes()))


@pytest.mark.parametrize('name', ['manifest', 'evaluator_mask', 'reference_palette', 'f3', 'length',
                                  'axis', 'distance', 'bounds', 'calibration', 'proposal', 'context'])
def test_rgb_entry_rejects_manual_and_evaluator_arguments(generated, name):
    with pytest.raises(TypeError):
        adapter.estimate_image_unit_face(*generated, **{name: object()})


def test_crop_evidence_has_no_evaluator_reads_and_all_outputs_sealed_before_truth(monkeypatch, prepared):
    tree = ast.parse(inspect.getsource(adapter))
    assert not any(isinstance(n, ast.Name) and n.id in ('RECIPE', 'evaluate_sealed', 'truth') for n in ast.walk(tree))
    events = []
    monkeypatch.setattr(experiment, 'render_rgb', lambda case: (np.zeros((2, 2, 3), np.uint8), None))
    monkeypatch.setattr(experiment, 'estimate_image_unit_face', lambda *a, **k: adapter.ImageUnitFaceResult(
        'abstained', 'test', prepared=prepared, counts=prepared.counts))
    original_seal = experiment.seal_diagnostic
    original_evaluate = experiment.evaluate_sealed
    original_labels = experiment._rendered_target_labels
    def seal(case, result):
        events.append(('seal', case))
        return original_seal(case, result)
    def evaluate(sealed):
        assert len([event for event in events if event[0] == 'seal']) == 3
        events.append(('evaluate', sealed.sha256))
        return original_evaluate(sealed)
    def labels():
        assert len([event for event in events if event[0] == 'seal']) == 3
        events.append(('labels', 'independent-renderer'))
        return original_labels()
    monkeypatch.setattr(experiment, 'seal_diagnostic', seal)
    monkeypatch.setattr(experiment, 'evaluate_sealed', evaluate)
    monkeypatch.setattr(experiment, '_rendered_target_labels', labels)
    result = experiment.run_controlled_scene()
    assert len(result['diagnostics']) == 3
    assert [e[0] for e in events if e[0] != 'labels'] == ['seal']*3+['evaluate']*3
    assert [e[0] for e in events].count('labels') == 1
    sealed = original_seal('identifiable', adapter.ImageUnitFaceResult('abstained', 'test'))
    assert json.loads(sealed.payload_json)['fit'] is None
    with pytest.raises(ValueError, match='stale_sealed'):
        original_evaluate(replace(sealed, sha256='0'*64))


def reseal_record(record):
    import hashlib
    payload = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return experiment.SealedDiagnostic(payload, hashlib.sha256(payload.encode()).hexdigest())


def support_record(prepared):
    return json.loads(experiment.seal_diagnostic('identifiable', adapter.ImageUnitFaceResult(
        'abstained', 'test', prepared=prepared, counts=prepared.counts)).payload_json)


def test_sealed_original_taps_have_actual_roles_and_post_seal_target_ownership(prepared):
    record = support_record(prepared)
    e = record['preparation']['source_tap_evidence']
    for role, code in (('generation', 1), ('validation', 2)):
        assert e[role+'_crop_pixels'] == list(getattr(prepared.support, role+'_pixels'))
        assert e[role+'_parent_pixels'] == list(prepared.parent_pixels(role))
        assert e[role+'_roles'] == [prepared.support.partition.roles[i] for i in e[role+'_crop_pixels']]
        assert set(e[role+'_roles']) == {code}
    assert e['samples'] and e['groups']
    result = experiment.evaluate_sealed(reseal_record(record))
    assert result['source_ownership_verified']
    assert result['source_ownership_counts'] == {
        'generation': prepared.counts.generation_pixels, 'validation': prepared.counts.validation_pixels}
    assert result['truth_status'] == 'unavailable'
    assert result['truth_reason'] == 'completed_nonempty_fit_unavailable'


@pytest.mark.parametrize('case,reason', [('mixed_depth', 'mixed_depth_has_no_single_plane_truth'),
                                       ('ambiguous', 'ambiguous_scene_has_no_identifiable_unit_anchor')])
def test_controls_cannot_receive_single_plane_scoring_even_with_forged_cover(prepared, monkeypatch, case, reason):
    record = support_record(prepared)
    record['case'] = case
    record['fit'] = {'search_complete': True, 'domain_search_complete': True,
        'cover': [{'distance': [1., 8.], 'phase_u': [0., 1.], 'phase_v': [0., 1.]}],
        'representative': {'distance': 3.23, 'phase': [.173, .287]}}
    monkeypatch.setattr(experiment, '_score_single_plane', lambda *a: pytest.fail('control scored single-plane truth'))
    monkeypatch.setattr(experiment, '_rendered_target_labels', lambda: pytest.fail('control read single-plane labels'))
    result = experiment.evaluate_sealed(reseal_record(record))
    assert result['truth_status'] == 'not_applicable' and result['truth_reason'] == reason
    assert not result['source_ownership_verified']
    assert all(result[key] is None for key in ('distance_covered', 'phase_covered', 'joint_covered',
                                              'distance_relative_error', 'phase_circular_errors'))


@pytest.mark.parametrize('wrong,reason', [
    ('non_target_crop', 'source_support_outside_rendered_target'),
    ('parent_index', 'source_crop_parent_binding_mismatch'),
    ('axis', 'target_plane_axis_or_side_mismatch'), ('side', 'target_plane_axis_or_side_mismatch'),
    ('phase_axes', 'target_phase_gauge_mismatch'), ('phase_origin', 'target_phase_gauge_mismatch'),
    ('signed_axes', 'target_phase_gauge_mismatch'), ('roles', 'invalid_sealed_source_evidence'),
    ('tap_closure', 'cross_role_sealed_source_tap'), ('recipe', 'recipe_binding_mismatch'),
])
def test_wrong_selected_support_or_gauge_refuses_evaluator_scoring(prepared, monkeypatch, wrong, reason):
    record = support_record(prepared)
    p = record['preparation']
    e = p['source_tap_evidence']
    if wrong == 'non_target_crop':
        # A coherent integer native crop/parent-index transform selecting
        # calibration/background pixels WITH target bbox overlap. The complete
        # parent tap labels, rather than that overlap, must reject scoring.
        p['crop'][:2] = [40, 157]
        assert p['crop'][0] < 270 < p['crop'][0]+p['crop'][2]
        width = p['crop'][2]
        x0, y0 = p['crop'][:2]
        for role in ('generation', 'validation'):
            e[role+'_parent_pixels'] = [(i//width+y0)*800+i % width+x0 for i in e[role+'_crop_pixels']]
    elif wrong == 'parent_index':
        e['generation_parent_pixels'][0] += 1
    elif wrong in ('axis', 'side'):
        p[wrong] = 0 if wrong == 'axis' else -1
    elif wrong == 'phase_axes':
        p['phase_gauge']['in_plane_axes'] = [1, 0]
    elif wrong == 'phase_origin':
        p['phase_gauge']['convention'] = 'texel-origin-instead-of-block-centre'
    elif wrong == 'signed_axes':
        p['phase_gauge']['axis_matrix'][0] = [-v for v in p['phase_gauge']['axis_matrix'][0]]
    elif wrong == 'roles':
        e['generation_roles'][0] = 2
    elif wrong == 'tap_closure':
        first = e['groups'][0]
        first['role'] = 'generation' if first['role'] == 'validation' else 'validation'
    else:
        record['recipe_sha256'] = '0'*64
    monkeypatch.setattr(experiment, '_score_single_plane', lambda *a: pytest.fail('incompatible support scored truth'))
    result = experiment.evaluate_sealed(reseal_record(record))
    assert result['truth_status'] == 'unavailable' and result['truth_reason'] == reason
    assert not result['source_ownership_verified']
    assert result['joint_covered'] is None and result['distance_covered'] is None


def test_preregistered_actual_rgb_calibration_proposal_path_no_fallback():
    result = experiment.run_controlled_scene()
    assert result['recipe_sha256'] == '98227b8b880a1f8588841907acf5efc81a0d101e13f8e47d87b069cc6ea33cf0'
    assert [d['case'] for d in result['diagnostics']] == ['identifiable', 'mixed_depth', 'ambiguous']
    for diagnostic, evaluation in zip(result['diagnostics'], result['evaluation']):
        assert diagnostic['calibration_status'] == 'complete'
        assert diagnostic['proposal_status'] == 'ok'
        assert not diagnostic['placement_supported'] and not evaluation['accuracy_claim']
        # Refusal is preserved; this test does not demand numerical success or
        # substitute evaluator optics if the actual path cannot identify scale.
        if diagnostic['fit'] is not None:
            assert diagnostic['fit']['work_used'] <= diagnostic['fit']['work_budget']
            assert diagnostic['fit']['actual_pixel_work'] <= diagnostic['fit']['pixel_work_budget']
        if diagnostic['status'] == 'abstained':
            assert diagnostic['reason']
    assert result['diagnostics'][1]['status'] == 'abstained'
    assert result['diagnostics'][2]['status'] == 'abstained'
    native, scored = result['diagnostics'][0], result['evaluation'][0]
    assert native['preparation'] is not None and native['fit'] is not None
    fit = native['fit']
    assert fit['domain_search_complete'] and fit['search_complete']
    assert fit['candidates'] and fit['cover'] and fit['retained_boxes'] == len(fit['cover']) > 0
    assert fit['work_used'] > 0 and fit['actual_pixel_work'] > 0
    assert native['status'] == 'abstained' and native['reason'] == 'scale_or_phase_ambiguous'
    lower, upper = fit['distance_interval']
    assert 1 < lower < upper < 8
    assert upper-lower >= .10*lower
    assert fit['phase_intervals'][0][1] >= .2
    assert fit['representative'] is None and fit['source_validation_sha256'] == ''
    assert all(candidate['validation_rmse'] is None for candidate in fit['candidates'])
    assert scored['truth_status'] == 'evaluated' and scored['truth_scope'] == 'identifiable_single_plane_target'
    assert scored['source_ownership_verified'] and scored['joint_covered']
    assert scored['distance_covered'] and scored['phase_covered'] == [True, True]
    assert native['preparation']['frame_sha256'] == 'c66b632232dab88e0c688615056d2e66471ff264cbd5197522d1f5ad4cd49870'
    assert result['evaluation'][1]['truth_status'] == 'not_applicable'
    assert result['evaluation'][1]['truth_reason'] == 'mixed_depth_has_no_single_plane_truth'
    assert result['evaluation'][2]['truth_status'] == 'not_applicable'
