"""Density contracts/development only; NEVER execute a heldout density arm."""
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.evaluation import run_unit_face_density as density
from mc_photogrammetry.reconstruction.unit_face_lattice import (
    ContractLimits, PartitionedSupport, SourceTapPartition, WitnessProof,
)


@pytest.fixture(scope='module')
def capture():
    scene = density.SCENES[0]
    assert scene.purpose == 'development'
    image, catalogue = density.render_scene(scene)
    result = density.capture_rgb_preparation(image, catalogue, scene.identity)
    assert result.prepared is not None and result.ordinary_refusal is None
    return result


@pytest.fixture(scope='module')
def pair(capture):
    result = density.build_support_pair(capture)
    assert result.dense.status == 'complete', result.dense.reason
    return result


def test_preregistered_disjoint_identity_pins_no_heldout_execution():
    assert density.PREREGISTRATION_SHA256 == '87f0e54c8cd721c625c9ca59bc86e4cb3a24bddbd19083bf4449c9df27f1b0d3'
    prereg = json.loads(density.PREREGISTRATION_JSON)
    assert prereg['contract_caps'] == asdict(ContractLimits())
    assert prereg['search_policy'] == json.loads(density._json(asdict(density.SearchPolicy())))
    assert prereg['support_algorithm_sha256'] and prereg['dependencies']
    assert prereg['success_rules']['strict_distance_width_over_lower'] == .10
    assert prereg['success_rules']['strict_phase_arc_width'] == .2
    heldout = [scene for scene in density.SCENES if scene.purpose == 'heldout']
    assert len(heldout) == 1 and heldout[0].identity == 'heldout-v2-identifiable'
    assert heldout[0].sha256 == '2e7d8d9f57c9ffcd8837977ee3c4b35e50cdc21a3530d9df78f73aafed79559c'
    assert heldout[0].recipe_json != density.SCENES[0].recipe_json


def test_capture_actual_adapter_once_no_numerical_fit_and_restored(capture):
    p = capture.prepared
    assert density.adapter.estimate_unit_face is capture.restored_fitter
    assert capture.args[1] is p.context and capture.args[2] is p.support
    assert capture.args[0].tobytes() == p.crop_rgb and not capture.args[0].flags.writeable
    assert capture.fixed_policy == density.SearchPolicy()
    density.adapter._verify_prepared(p)


@pytest.mark.parametrize('failure', ['missing_calibration', 'runtime', 'interrupt', 'other_sentinel'])
def test_capture_restores_both_functions_for_refusal_and_every_exception(monkeypatch, failure):
    adapter = density.adapter
    before_prepare, before_fit = adapter._fit_prepared, adapter.estimate_unit_face
    image = np.zeros((4, 4, 3), np.uint8)
    if failure == 'missing_calibration':
        monkeypatch.setattr(adapter.calibration, 'estimate_image_calibrations', lambda frames: {key: None for key in frames})
        result = density.capture_rgb_preparation(image, None, 'ordinary-refusal')
        assert result.prepared is None and result.ordinary_refusal.reason == 'image_calibration_failed'
    else:
        error = {'runtime': RuntimeError('do-not-swallow'), 'interrupt': KeyboardInterrupt(),
                 'other_sentinel': density._PreparationOnly()}[failure]
        def fail(frames):
            raise error
        monkeypatch.setattr(adapter.calibration, 'estimate_image_calibrations', fail)
        with pytest.raises(type(error)) as caught:
            density.capture_rgb_preparation(image, None, 'propagation')
        assert caught.value is error
    assert adapter._fit_prepared is before_prepare and adapter.estimate_unit_face is before_fit


def test_density_retains_every_baseline_record_and_exact_validation(pair):
    p = pair.capture.prepared
    baseline, dense = p.support, pair.dense.support
    assert dense.partition is baseline.partition
    assert dense.groups[:len(baseline.groups)] == baseline.groups
    assert dense.witnesses[:len(baseline.witnesses)] == baseline.witnesses
    original_samples = {(s.witness_index, s.sample_index): s for s in baseline.samples}
    for sample in dense.samples:
        key = (sample.witness_index, sample.sample_index)
        if key in original_samples:
            assert sample == original_samples[key]
    assert dense.validation_pixels == baseline.validation_pixels
    assert set(baseline.generation_pixels) < set(dense.generation_pixels)
    assert not set(dense.generation_pixels) & set(dense.validation_pixels)
    assert pair.dense.added_generation_pixels == len(dense.generation_pixels)-len(baseline.generation_pixels) > 0
    assert len(dense.generation_pixels) <= 2048
    assert dense.partition.identity_sha256 == baseline.partition.identity_sha256
    assert sum(w.map_shape[0]*w.map_shape[1] for w in dense.witnesses) == 2000
    for witness in dense.witnesses:
        assert WitnessProof(**asdict(witness)) == witness


def test_all_eligible_generation40_stencils_exhaustively_retained_no_validation(pair):
    support = pair.dense.support
    partition = support.partition
    witness_index = len(support.witnesses)-1
    witness = support.witnesses[witness_index]
    h, w = partition.source_shape
    eligible = set()
    for index in range(1600):
        _, depth, taps = witness.sample_payload(index)
        pixels = [i for i, _ in taps]
        if (depth and all(i >= 0 and i % w < w-1 and i//w < h-1
                          and partition.usable[i] and partition.region[i]
                          and partition.roles[i] == 1 for i in pixels)):
            eligible.add(index)
    extra = support.groups[len(pair.capture.prepared.support.groups):]
    assert extra and all(g.role == 'generation' for g in extra)
    assert {g.sample_refs[0][1] for g in extra} == eligible
    assert all(g.sample_refs[0][0] == witness_index and len(g.sample_refs) == 1 for g in extra)
    assert len(extra) == pair.dense.eligible_generation_stencils
    assert (len(extra)+pair.dense.rejected_depth+pair.dense.rejected_border+pair.dense.rejected_excluded
            +pair.dense.rejected_cross_role+pair.dense.omitted_validation == 1600)
    expected = set(pair.capture.prepared.support.generation_pixels)
    for index in eligible:
        expected.update(i for i, _ in witness.sample_payload(index)[2])
    assert support.generation_pixels == tuple(sorted(expected))


def test_same_plane_window_native_source_and_density_rgb_noninterference(pair):
    p = pair.capture.prepared
    partition = p.support.partition
    shape = partition.source_shape
    mask = (np.frombuffer(partition.region, np.uint8).reshape(shape).astype(bool)
            & np.frombuffer(partition.usable, np.uint8).reshape(shape).astype(bool))
    ctx = p.context
    expected = density.lattice.plane_window(mask, ctx.axis, np.asarray(ctx.axis_matrix),
        ctx.focal_pixels, principal_point=ctx.principal_point)
    assert pair.dense.window == expected
    changed = replace(p, crop_rgb=bytes(len(p.crop_rgb)))
    rebuilt = density.extend_generation_support(changed)
    assert rebuilt == pair.dense
    assert hashlib.sha256(pair.manifest_json.encode()).hexdigest() == pair.manifest_sha256


@pytest.mark.parametrize('cap,value,reason', [
    ('max_source_pixels', 65535, 'density_source_pixel_cap'),
    ('max_rectified_samples', 1999, 'density_witness_sample_cap'),
    ('max_bilinear_taps', 7999, 'density_witness_tap_cap'),
    ('max_witness_payload_bytes', 1, 'density_witness_payload_cap'),
    ('max_dependency_memberships', 1, 'dependency_cap_exceeded'),
])
def test_all_contract_caps_atomic_refusal_no_model(capture, cap, value, reason):
    result = density.extend_generation_support(capture.prepared, limits=replace(ContractLimits(), **{cap: value}))
    assert result.status == 'refused' and result.reason == reason and result.support is None


def test_remaining_witness_payload_budget_refuses(capture):
    previous = sum(w.payload_bytes for w in capture.prepared.support.witnesses)
    result = density.extend_generation_support(capture.prepared,
        limits=replace(ContractLimits(), max_witness_payload_bytes=previous+1))
    assert result.status == 'refused' and result.reason == 'density_witness_budget_exhausted'


def test_more_than_role_cap_refuses_all_dense_evidence_instead_of_subsampling(capture):
    p = capture.prepared
    old = p.support
    partition = old.partition
    shape = partition.source_shape
    usable = np.frombuffer(partition.usable, np.uint8).reshape(shape).astype(bool)
    region = np.frombuffer(partition.region, np.uint8).reshape(shape).astype(bool)
    validation = np.zeros(shape, np.bool_)
    validation.flat[list(old.validation_pixels)] = True
    generation = usable & region & ~validation
    enlarged_partition = SourceTapPartition.from_masks(generation, validation, usable, region,
                                                       conditioning_sha256=partition.conditioning_sha256)
    # Synthetic cap control only. Actual study uses the genuine untouched
    # checker partition captured from RGB and never changes its roles.
    baseline = PartitionedSupport('complete', '', enlarged_partition, old.samples, old.groups, old.witnesses)
    result = density.extend_generation_support(replace(p, support=baseline))
    assert result.status == 'refused' and result.reason == 'density_role_source_pixel_cap'
    assert result.planned_generation_pixels > 2048 and result.support is None


@dataclass(frozen=True)
class StubFit:
    support_sha256: str
    context_sha256: str
    policy_sha256: str
    reason: str = 'stub-no-numerical-work'


def test_both_supports_manifest_exist_before_exactly_one_restored_call_per_arm(pair, monkeypatch):
    calls = []
    def fitter(image, context, support, catalogue, *, fixed_policy):
        assert pair.dense.support is not None and pair.manifest_sha256
        calls.append(support.identity_sha256)
        assert context is pair.capture.prepared.context and image is pair.capture.args[0]
        assert catalogue is pair.capture.args[3] and fixed_policy is pair.capture.fixed_policy
        return StubFit(support.identity_sha256, context.sha256, fixed_policy.sha256)
    monkeypatch.setattr(density.adapter, 'estimate_unit_face', fitter)
    capture = replace(pair.capture, restored_fitter=fitter)
    arms = density.fit_pair(replace(pair, capture=capture))
    assert calls == [pair.capture.prepared.support.identity_sha256, pair.dense.support.identity_sha256]
    assert arms['A']['preparation']['source_tap_evidence']['validation_parent_pixels'] == (
        arms['B']['preparation']['source_tap_evidence']['validation_parent_pixels'])


def test_stale_pair_manifest_refuses_before_any_fit(pair):
    with pytest.raises(density.DensityIntegrityError, match='stale_support_pair_manifest'):
        density.fit_pair(replace(pair, manifest_sha256='0'*64))


def test_pair_seal_requires_both_outputs_and_precedes_truth(pair, monkeypatch):
    scene = density.SCENES[0]
    dummy = {name: {'case': scene.case, 'fit': None, 'preparation': None} for name in ('A', 'B')}
    sealed = density.seal_pair(scene, pair, dummy)
    calls = []
    def score(recipe, arm):
        assert json.loads(sealed.payload_json)['arms'].keys() == {'A', 'B'}
        calls.append(arm)
        return {'primary_success': False}
    monkeypatch.setattr(density, '_score_arm', score)
    result = density.evaluate_pair(scene, sealed)
    assert len(calls) == 2 and not result['primary_success']
    with pytest.raises(ValueError, match='stale_pair_seal'):
        density.evaluate_pair(scene, replace(sealed, sha256='0'*64))
    with pytest.raises(ValueError, match='scene_recipe_binding_mismatch'):
        density.evaluate_pair(density.SCENES[-1], sealed)


@pytest.mark.parametrize('scene', density.SCENES[1:3])
def test_constant_and_composited_mixed_controls_never_score_single_plane(scene, pair, monkeypatch):
    sealed = density.seal_pair(scene, pair, {name: {'case': scene.case, 'fit': 'attempted-complete-cover'} for name in ('A', 'B')})
    monkeypatch.setattr(density, '_score_arm', lambda *a: pytest.fail('control scored a single plane'))
    result = density.evaluate_pair(scene, sealed)
    assert result['status'] == 'not_applicable' and result['arms'] is None and not result['primary_success']


def test_heldout_renderer_only_native_resolution_recipe_driven_no_fit_or_score(monkeypatch):
    monkeypatch.setattr(density, 'capture_rgb_preparation', lambda *a: pytest.fail('renderer fitted'))
    monkeypatch.setattr(density, 'evaluate_pair', lambda *a: pytest.fail('renderer scored truth'))
    image, catalogue = density.render_scene(density.SCENES[-1])
    assert image.shape == (600, 800, 3) and image.dtype == np.uint8
    assert catalogue.complete_within_subset
    assert all(face.texture_size == (16, 16) for face in catalogue.faces)
    recipe = json.loads(density.SCENES[-1].recipe_json)
    recipe['native_side'] = 8
    unit_scene = replace(density.SCENES[-1], identity='unit-renderer-native8', recipe_json=json.dumps(recipe))
    changed, smaller = density.render_scene(unit_scene)
    assert all(face.texture_size == (8, 8) for face in smaller.faces)
    assert not np.array_equal(image, changed)


def test_same_validation_true_path_real_prepared_records_requires_nonempty(pair, monkeypatch):
    scene = density.SCENES[0]
    records = {name: density._arm_record(pair, support, None, 'unit-no-fit', 0., scene.case)
        for name, support in (('A', pair.capture.prepared.support), ('B', pair.dense.support))}
    monkeypatch.setattr(density, '_score_arm', lambda *a: {'primary_success': True})
    scored = density.evaluate_pair(scene, density.seal_pair(scene, pair, records))
    assert scored['same_validation'] and scored['primary_success']
    for arm in records.values():
        evidence = arm['preparation']['source_tap_evidence']
        evidence['validation_parent_pixels'] = []
        evidence['validation_crop_pixels'] = []
        evidence['validation_roles'] = []
    empty = density.evaluate_pair(scene, density.seal_pair(scene, pair, records))
    assert not empty['same_validation'] and not empty['primary_success']


def test_refused_dense_arm_never_invokes_restored_fitter(capture, monkeypatch):
    refused_pair = density.build_support_pair(capture, limits=replace(ContractLimits(), max_rectified_samples=1999))
    calls = []
    def stub(image, context, support, catalogue, *, fixed_policy):
        calls.append(support)
        return StubFit(support.identity_sha256, context.sha256, fixed_policy.sha256)
    monkeypatch.setattr(density.adapter, 'estimate_unit_face', stub)
    refused_pair = replace(refused_pair, capture=replace(capture, restored_fitter=stub))
    records = density.fit_pair(refused_pair)
    assert calls == [capture.prepared.support]
    assert records['B']['fit'] is None and records['B']['reason'] == 'density_witness_sample_cap'


def test_remaining_sample_tap_and_payload_budgets_deduct_baseline(capture, monkeypatch):
    seen = []
    original = density.lattice.rectify_plane
    def instrument(*args, **kwargs):
        seen.append(kwargs['witness_budget'])
        return original(*args, **kwargs)
    monkeypatch.setattr(density.lattice, 'rectify_plane', instrument)
    limits = replace(ContractLimits(), max_rectified_samples=2000, max_bilinear_taps=8000)
    result = density.extend_generation_support(capture.prepared, limits=limits)
    assert result.status == 'complete'
    assert len(seen) == 1 and seen[0].max_rectified_samples == 1600 and seen[0].max_bilinear_taps == 6400
    used_bytes = sum(w.payload_bytes for w in capture.prepared.support.witnesses)
    assert seen[0].max_payload_bytes == limits.max_witness_payload_bytes-used_bytes
    assert result.baseline_witness_samples == 400 and result.baseline_worst_case_taps == 1600
    assert sum(w.map_shape[0]*w.map_shape[1]*4 for w in result.support.witnesses) == 8000


def test_tap_upper_bound_atomic_refusal_before_partition_index(capture, monkeypatch):
    original = density.lattice.rectify_plane
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        witness = kwargs['witness_sink'][0]
        forged = object.__new__(type(witness))
        for field in witness.__dataclass_fields__:
            object.__setattr__(forged, field, getattr(witness, field))
        indices = witness.tap_source_indices.copy()
        indices[0] = witness.source_shape[0]*witness.source_shape[1]
        object.__setattr__(forged, 'tap_source_indices', indices)
        kwargs['witness_sink'][0] = forged
        return result
    monkeypatch.setattr(density.lattice, 'rectify_plane', corrupt)
    result = density.extend_generation_support(capture.prepared)
    assert result.status == 'refused' and result.reason == 'density_source_tap_out_of_range'


def test_pair_rehashed_but_changed_support_binding_refuses(pair):
    forged = replace(pair, dense=replace(pair.dense, support=pair.capture.prepared.support))
    with pytest.raises(density.DensityIntegrityError, match='support_pair_manifest_binding_mismatch'):
        density.fit_pair(forged)


def test_integrity_gates_survive_python_optimized_mode():
    code = """from types import SimpleNamespace
from scripts.evaluation import run_unit_face_density as d
try:
    d.fit_pair(SimpleNamespace(manifest_json='{}',manifest_sha256='0'*64))
except d.DensityIntegrityError as error:
    print(error.reason)
else:
    raise RuntimeError('optimized mode removed gate')
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-O', '-c', code], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and result.stdout.strip() == 'stale_support_pair_manifest'


def test_durable_one_shot_marker_transitions_and_rerun_before_renderer(monkeypatch):
    fresh = density.PREREGISTRATION_SHA256+'\n'+density.RUN_MARKER+'\n'
    claimed = density._report_transition(fresh, 'RUNNING', None)
    assert 'HELDOUT_EXECUTION_STATE: RUNNING' in claimed
    with pytest.raises(density.DensityIntegrityError, match='heldout_execution_already_claimed'):
        density._report_transition(claimed, 'RUNNING', None)
    for state in ('COMPLETE', 'FAILED'):
        ended = density._report_transition(claimed, state, {'sealed_pair_sha256': 'unit-result'})
        assert state in ended and 'unit-result' in ended
        with pytest.raises(density.DensityIntegrityError, match='heldout_execution_already_claimed'):
            density._report_transition(ended, 'RUNNING', None)
    monkeypatch.setattr(density, '_update_owned_report', lambda state: density._report_transition(claimed, state, None))
    monkeypatch.setattr(density, 'render_scene', lambda *a: pytest.fail('rerun reached renderer'))
    with pytest.raises(density.DensityIntegrityError, match='heldout_execution_already_claimed'):
        density.run_study()


@pytest.mark.parametrize('relative', [
    'src/mc_photogrammetry/reconstruction/calibration.py',
    'src/mc_photogrammetry/reconstruction/camera.py',
    'src/mc_photogrammetry/reconstruction/region_growth.py',
    'src/mc_photogrammetry/reconstruction/boundary_candidates.py',
    'src/mc_photogrammetry/vision/gui_mask.py',
    'src/mc_photogrammetry/blocks/unit_face_catalogue.py',
    'src/mc_photogrammetry/pipeline/contracts.py',
    'src/mc_photogrammetry/domain/models.py',
])
def test_upstream_source_change_invalidates_prereg_before_renderer(relative, monkeypatch):
    target = density.REPORT_PATH.parents[1].parent / relative
    original = density.Path.read_bytes
    def altered(path):
        raw = original(path)
        return raw+b'\n# unit mutation\n' if path == target else raw
    monkeypatch.setattr(density.Path, 'read_bytes', altered)
    monkeypatch.setattr(density, 'render_scene', lambda *a: pytest.fail('changed source reached renderer'))
    monkeypatch.setattr(density, '_update_owned_report', lambda *a: pytest.fail('changed source claimed run'))
    with pytest.raises(density.DensityIntegrityError, match='changed_preregistered_dependency'):
        density.run_study()


@pytest.mark.parametrize('module,attribute', [(density.sys, 'version'), (density.np, '__version__'),
                                             (density.cv2, '__version__'), (density.PIL, '__version__')])
def test_runtime_version_change_invalidates_prereg_before_renderer(module, attribute, monkeypatch):
    monkeypatch.setattr(module, attribute, 'unit-mutated-version')
    monkeypatch.setattr(density, 'render_scene', lambda *a: pytest.fail('changed version reached renderer'))
    monkeypatch.setattr(density, '_update_owned_report', lambda *a: pytest.fail('changed version claimed run'))
    with pytest.raises(density.DensityIntegrityError, match='changed_preregistered_dependency'):
        density.run_study()


class MemoryEvidence:
    """No filesystem artifacts: exercise exact artifact codec/write routines."""
    name = 'UNIT-FACE-DENSITY-01-evidence-2026-10-05.jsonl'
    def __init__(self):
        self.content = None
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
                return 54321
            def flush(self):
                owner.content = self.getvalue()
            def close(self):
                if not self.closed:
                    self.flush()
                super().close()
        return Handle(self.content)


@pytest.fixture
def memory_evidence(monkeypatch):
    artifact = MemoryEvidence()
    flushed = []
    monkeypatch.setattr(density, 'EVIDENCE_PATH', artifact)
    monkeypatch.setattr(density.os, 'fsync', lambda fd: flushed.append(fd))
    return artifact, flushed


def _unit_sealed_pair(scene, pair):
    records = {name: density._arm_record(pair, support, None, 'unit-no-fit', 0., scene.case)
        for name, support in (('A', pair.capture.prepared.support), ('B', pair.dense.support))}
    return density.seal_pair(scene, pair, records)


def test_evidence_full_payload_durably_retained_and_replayed(pair, memory_evidence):
    artifact, flushed = memory_evidence
    density._initialize_evidence()
    scene = density.SCENES[0]
    sealed = _unit_sealed_pair(scene, pair)
    sha = density._append_sealed_evidence(scene, sealed, ())
    record = json.loads(artifact.content.splitlines()[1])
    assert record['sealed_pair_payload_json'] == sealed.payload_json
    assert record['sealed_pair_sha256'] == sealed.sha256
    assert record['preregistration_sha256'] == density.PREREGISTRATION_SHA256
    assert sha == hashlib.sha256(artifact.content).hexdigest() and flushed == [54321, 54321]
    assert density._verify_evidence((sealed.sha256,)) == sha


@pytest.mark.parametrize('mutation,reason', [
    ('missing', 'density_evidence_missing'), ('truncated', 'density_evidence_truncated'),
    ('header', 'density_evidence_header_mismatch'), ('payload', 'density_evidence_pair_mismatch'),
    ('sha', 'density_evidence_pair_mismatch'), ('prefix', 'density_evidence_prefix_mismatch'),
])
def test_evidence_missing_or_tampered_fails_closed(pair, memory_evidence, mutation, reason):
    artifact, _ = memory_evidence
    density._initialize_evidence()
    sealed = _unit_sealed_pair(density.SCENES[0], pair)
    density._append_sealed_evidence(density.SCENES[0], sealed, ())
    expected = (sealed.sha256,)
    if mutation == 'missing':
        artifact.content = None
    elif mutation == 'truncated':
        artifact.content = artifact.content[:-1]
    else:
        lines = [json.loads(line) for line in artifact.content.splitlines()]
        if mutation == 'header':
            lines[0]['preregistration_sha256'] = '0'*64
        elif mutation == 'payload':
            lines[1]['sealed_pair_payload_json'] += ' '
        elif mutation == 'sha':
            lines[1]['sealed_pair_sha256'] = '0'*64
        else:
            expected = ()
        artifact.content = ('\n'.join(density._json(line) for line in lines)+'\n').encode()
    with pytest.raises(density.DensityIntegrityError, match=reason):
        density._verify_evidence(expected)


def test_existing_evidence_refuses_before_run_claim_or_renderer(memory_evidence, monkeypatch):
    artifact, _ = memory_evidence
    artifact.content = b'existing artifact'
    monkeypatch.setattr(density, '_update_owned_report', lambda *a: pytest.fail('existing evidence claimed run'))
    monkeypatch.setattr(density, 'render_scene', lambda *a: pytest.fail('existing evidence rendered'))
    with pytest.raises(density.DensityIntegrityError, match='density_evidence_already_exists'):
        density.run_study()


def test_missing_evidence_after_seal_refuses_before_evaluator(pair, memory_evidence, monkeypatch):
    artifact, _ = memory_evidence
    report_calls = []
    monkeypatch.setattr(density, '_update_owned_report', lambda *args: report_calls.append(args))
    monkeypatch.setattr(density, 'render_scene', lambda *a: (None, None))
    monkeypatch.setattr(density, 'capture_rgb_preparation', lambda *a: pair.capture)
    monkeypatch.setattr(density, 'build_support_pair', lambda *a: pair)
    monkeypatch.setattr(density, 'fit_pair', lambda *a, **k: {})
    genuine_seal = density.seal_pair
    def deleted_after_seal(scene, actual_pair, arms):
        records = {name: density._arm_record(pair, support, None, 'unit-no-fit', 0., scene.case)
            for name, support in (('A', pair.capture.prepared.support), ('B', pair.dense.support))}
        result = genuine_seal(scene, actual_pair, records)
        artifact.content = None
        return result
    monkeypatch.setattr(density, 'seal_pair', deleted_after_seal)
    monkeypatch.setattr(density, 'evaluate_pair', lambda *a: pytest.fail('missing evidence reached evaluator'))
    with pytest.raises(density.DensityIntegrityError, match='density_evidence_missing'):
        density.run_study()
    assert report_calls[-1][0] == 'FAILED' and artifact.content is None


def test_study_append_precedes_evaluator_and_later_failure_preserves_prefix(pair, memory_evidence, monkeypatch):
    artifact, flushed = memory_evidence
    report_calls, scored = [], []
    monkeypatch.setattr(density, '_update_owned_report', lambda *args: report_calls.append(args))
    monkeypatch.setattr(density, 'render_scene', lambda *a: (None, None))
    monkeypatch.setattr(density, 'capture_rgb_preparation', lambda *a: pair.capture)
    monkeypatch.setattr(density, 'build_support_pair', lambda *a: pair)
    def no_fit(actual_pair, *, case):
        assert actual_pair is pair
        return {name: density._arm_record(pair, support, None, 'unit-no-fit', 0., case)
            for name, support in (('A', pair.capture.prepared.support), ('B', pair.dense.support))}
    monkeypatch.setattr(density, 'fit_pair', no_fit)
    def no_truth(scene, sealed):
        # The exact full payload was appended AND fsync called before this spy.
        records = [json.loads(line) for line in artifact.content.splitlines()[1:]]
        assert records[-1]['sealed_pair_payload_json'] == sealed.payload_json
        assert len(flushed) == len(records)+1
        scored.append(sealed.sha256)
        assert density._verify_evidence(tuple(scored)) == hashlib.sha256(artifact.content).hexdigest()
        if len(scored) == 2:
            raise RuntimeError('unit-later-evaluator-failure')
        return {'unit_no_truth': True}
    monkeypatch.setattr(density, 'evaluate_pair', no_truth)
    monkeypatch.setattr(density, 'summarize_pair', lambda scene, sealed, value: {'scene_identity': scene.identity})
    with pytest.raises(RuntimeError, match='unit-later-evaluator-failure'):
        density.run_study()
    assert len(scored) == 2 and len(artifact.content.splitlines()) == 3
    assert density._verify_evidence(tuple(scored)) == hashlib.sha256(artifact.content).hexdigest()
    assert report_calls[0] == ('RUNNING',) and report_calls[-1][0] == 'FAILED'
    assert report_calls[-1][1]['retained_sealed_pairs'] == scored
    assert len(report_calls[-1][1]['completed_summaries']) == 1
    with pytest.raises(density.DensityIntegrityError, match='density_evidence_already_exists'):
        density.run_study()


def test_numerical_validation_rgb_mutation_noninterference_per_arm(pair):
    # DEVELOPMENT ONLY. No heldout arm is executed by this test. Both real
    # development fit outputs are sealed before any evaluator access; mutation
    # tests inspect generation evidence only and perform no truth scoring.
    arms = density.fit_pair(pair)
    density.seal_pair(density.SCENES[0], pair, arms)
    p, capture = pair.capture.prepared, pair.capture
    for name, support in (('A', p.support), ('B', pair.dense.support)):
        baseline = arms[name]['fit']
        assert baseline['search_complete'] and baseline['domain_search_complete']
        assert baseline['candidates'] and baseline['cover'] and baseline['work_used'] > 0
        changed = p.image().copy()
        changed.reshape(-1, 3)[list(support.validation_pixels)] = 255-changed.reshape(-1, 3)[list(support.validation_pixels)]
        assert np.array_equal(changed.reshape(-1, 3)[list(support.generation_pixels)],
                              p.image().reshape(-1, 3)[list(support.generation_pixels)])
        mutated = asdict(capture.restored_fitter(changed, capture.args[1], support, capture.args[3],
                                                fixed_policy=capture.fixed_policy))
        assert mutated['search_complete'] and mutated['domain_search_complete']
        assert mutated['cover'] == baseline['cover'] and mutated['distance_interval'] == baseline['distance_interval']
        assert mutated['phase_intervals'] == baseline['phase_intervals']
        assert mutated['work_used'] == baseline['work_used'] and mutated['actual_pixel_work'] == baseline['actual_pixel_work']
        assert mutated['source_generation_sha256'] == baseline['source_generation_sha256']
        assert len(mutated['candidates']) == len(baseline['candidates']) > 0
        for before, after in zip(baseline['candidates'], mutated['candidates']):
            assert all(before[k] == after[k] for k in ('face_sha256', 'geometry_sha256', 'distance',
                                                      'phase', 'gain', 'bias', 'generation_rmse'))
