"""Fixed-input substitution only; no solve, fit, deletion or candidate creation."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation import audit_plane_constraints as study

ARTIFACT = study.ROOT / study.replay.ARTIFACT_RELATIVE
STAGE = ARTIFACT.parent / 'stage.mcscan'


@pytest.fixture
def validated():
    return study.replay.validate_pinned_artifact(ARTIFACT)


def test_exact_captured_equation_substitution_all_axes(validated):
    result = study._audit_validated(validated)
    assert [r['match_index'] for r in result['constraints']] == [4, 5, 9, 16, 17, 19, 23, 26, 33, 37]
    assert [a['constraints'] for a in result['axes']] == [4, 3, 3]
    for row in result['constraints']:
        L, R = row['left'], row['right']
        expected_target = L['sign']*L['side']*L['distance']-R['sign']*R['side']*R['distance']
        assert row['target_delta'] == expected_target
        assert row['captured_translation_delta'] == R['translation']-L['translation']
        assert row['signed_linear_residual'] == row['captured_translation_delta']-expected_target
        assert row['signed_linear_residual'] == pytest.approx(row['captured_right_minus_left_face'], abs=1e-12)
        assert 0 <= row['wrapped_face_pair_gap'] <= .5


def test_deepseek_magnitude_is_linear_residual_not_period_or_global_phase(validated):
    result = study._audit_validated(validated)
    axis1 = [r for r in result['constraints'] if r['world_axis'] == 1]
    assert [r['match_index'] for r in axis1] == [5, 19, 33]
    assert [r['signed_linear_residual'] for r in axis1] == pytest.approx([-1.494204530979042, 1.494204530979042, -1.494204530979042], abs=1e-12)
    assert [r['wrapped_face_pair_gap'] for r in axis1] == pytest.approx([.494204530979042]*3, abs=1e-12)
    triangle = result['axis1_triangle']
    assert triangle['target_closure_blocks'] == pytest.approx(4.482613592937126, abs=1e-12)
    assert abs(triangle['captured_translation_closure_blocks']) < 1e-12
    assert triangle['signed_residual_closure_blocks'] == pytest.approx(-triangle['target_closure_blocks'], abs=1e-12)
    axis = result['axes'][1]
    assert axis['captured_global_phase_residual'] == .24710226548952122
    assert axis['captured_global_resultant'] == .013942164981374504
    assert axis['paired_phasor_resultant_upper_bound'] == pytest.approx(.018205996995421646, abs=1e-12)
    assert axis['captured_global_resultant'] <= axis['paired_phasor_resultant_upper_bound']


def test_plane_identity_b_conflicts_are_retained_without_choice_change(validated):
    before = deepcopy(validated)
    result = study._audit_validated(validated)
    uses = {(u['frame'], u['plane']): u for u in result['plane_uses']}
    assert uses['image-0000', 2]['selected_b'] == [.25, .5]
    assert uses['image-0001', 3]['selected_b'] == [.125, 3.]
    assert uses['image-0002', 2]['selected_b'] == [.25]
    assert uses['image-0002', 3]['selected_b'] == [.125]
    assert uses['image-0000', 2]['captured_alternative_distance_per_b_range'] == pytest.approx([3.0746233309304403]*2)
    assert validated == before


@pytest.mark.parametrize('mutation', ['b', 'distance', 'side', 'sign', 'alternative', 'translation', 'scalar', 'assignment', 'relative'])
def test_changed_constraint_identity_refuses_before_reporting(validated, mutation):
    value = deepcopy(validated)
    phase = value['solver_failure_observation']['phase_capture']
    inputs = value['solver_failure_observation']['solver_input_bindings']
    edge = next(e for e in inputs['edges'] if e[0] == 5)
    if mutation == 'b': edge[1][2] += .01
    if mutation == 'distance': edge[1][3] += .01
    if mutation == 'side': edge[1][5] *= -1
    if mutation == 'sign': edge[1][6] *= -1
    if mutation == 'alternative': edge[1][1] = (edge[1][1]+1) % 3
    if mutation == 'translation': phase['translations'][0][1][1] += .01
    if mutation == 'scalar': next(f for f in phase['faces'] if f[0] == 5)[16] += .01
    if mutation == 'assignment': inputs['assignments'][0] = (inputs['assignments'][0]+1) % 4
    if mutation == 'relative': edge[3] = (edge[3]+1) % 4
    with pytest.raises(study.replay.ArtifactError): study._audit_validated(value)


def test_public_audit_never_calls_any_fit_or_loo_and_preserves_bytes(monkeypatch):
    before = ARTIFACT.read_bytes(), STAGE.read_bytes()
    _, plane_graph = study.replay._load_production_validators(study.ROOT / 'src')
    def forbidden(*args, **kwargs): pytest.fail('no fit/solve/deletion/LOO permitted')
    monkeypatch.setattr(np.linalg, 'lstsq', forbidden)
    monkeypatch.setattr(np.linalg, 'solve', forbidden)
    monkeypatch.setattr(plane_graph, 'solve_component', forbidden)
    monkeypatch.setattr(plane_graph, '_evaluate_assignment', forbidden)
    monkeypatch.setattr(study.replay, 'leave_one_edge_out', forbidden)
    monkeypatch.setattr(study.replay, 'analyze', forbidden)
    result = study.audit()
    assert result['status'] == 'complete'
    for key in ('solver_fit', 'edge_deletion', 'b_changed', 'reweighted', 'images_or_truth_read', 'missing_period_reconstructed'):
        assert result[key] is False
    assert (ARTIFACT.read_bytes(), STAGE.read_bytes()) == before
    assert result['fixed_captured_translations'] == study.replay.validate_pinned_artifact(ARTIFACT)['solver_failure_observation']['phase_capture']['translations']


def test_available_metadata_contains_no_retained_period_patch_or_lattice_component():
    result = study.audit()['stage_evidence']
    assert result['sha256'] == study.STAGE_SHA256
    assert result['persisted_component_count'] == 0
    assert result['raw_pixel_period_or_patch_evidence_retained'] is False
    assert result['camera_and_frame_values_used'] is False


@pytest.mark.parametrize('changed', ['artifact', 'stage', 'replay_source'])
def test_digest_tamper_cannot_reach_arithmetic(monkeypatch, changed):
    target = {'artifact': ARTIFACT, 'stage': STAGE, 'replay_source': Path(study.replay.__file__)}[changed]
    original = Path.read_bytes
    def read(path):
        raw = original(path)
        return raw+b' ' if path == target else raw
    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(study, '_audit_validated', lambda *_: pytest.fail('unverified arithmetic'))
    with pytest.raises(study.replay.ArtifactError, match='sha256|digest'): study.audit()


def test_stage_cap_refuses_and_arithmetic_is_deterministic(monkeypatch, validated):
    first = study._audit_validated(validated)
    assert study._audit_validated(validated) == first
    monkeypatch.setattr(study, 'MAX_STAGE_BYTES', 1)
    with pytest.raises(study.replay.ArtifactError, match='stage_byte_cap'): study.audit()


def test_only_sealed_metadata_and_pinned_sources_are_opened(monkeypatch):
    original = Path.read_bytes
    allowed = {ARTIFACT.resolve(), STAGE.resolve(), Path(study.replay.__file__).resolve()}
    allowed.update((study.ROOT / name).resolve() for name in study.replay.SOURCE_PINS)
    opened = []
    def read(path):
        assert path.resolve() in allowed, 'image/truth/unregistered artifact read'
        opened.append(path.resolve())
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    assert study.audit()['status'] == 'complete'
    assert ARTIFACT.resolve() in opened and STAGE.resolve() in opened
