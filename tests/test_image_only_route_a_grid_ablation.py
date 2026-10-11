"""Synthetic seams only: no fixture media, truth or camera manifests are read."""
from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.facade import reconstruct_facade

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/evaluation' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def replay_module():
    return load('_grid_replay_test', 'run_image_only_route_a.py')


@pytest.fixture
def report_module():
    return load('_grid_report_test', 'report_image_only_attribution.py')


def seams(module, monkeypatch, tmp_path, *, fail=False):
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[:8] = True
    image[48:56] = 0
    camera = CameraModel(position=(0, 0, 0), yaw_degrees=35, pitch_degrees=5,
                         fov_degrees=70)
    calls = []
    source = tmp_path / 'synthetic.png'
    source.write_bytes(b'synthetic-input-only')
    monkeypatch.setattr(module, 'load_image', lambda _: image)
    monkeypatch.setattr(module, 'create_gui_mask', lambda *_: mask)
    def solve(_):
        calls.append('solve')
        if fail:
            raise module.ImageOnlyCameraError('forced failure')
        return SimpleNamespace(cameras={source.name: object()})
    monkeypatch.setattr(module, 'solve_image_cameras', solve)
    monkeypatch.setattr(module, 'camera_model_from_estimate', lambda *_a, **_k: camera)
    def facade(*args, **kwargs):
        calls.append(('facade', kwargs['camera']))
        return reconstruct_facade(*args, **kwargs)
    monkeypatch.setattr(module, 'reconstruct_facade', facade)
    monkeypatch.setattr(module, 'save_project', lambda project, path: path.write_text(
        module.canonical_json([[list(c), o.state.to_dict()] for c, o in sorted(project.volume.blocks.items())])))
    def export(_self, volume, path, **kwargs):
        path.write_bytes(b'synthetic-native-export')
    monkeypatch.setattr(module.LitematicExporter, 'export', export)
    monkeypatch.setattr(module, '_decoded_volume', lambda _: (None, (0, 0, 0)))
    return source, image, mask, camera, calls


def test_default_matches_explicit_control_native_bytes_and_semantics(replay_module, monkeypatch, tmp_path):
    m = replay_module
    source, image, mask, camera, calls = seams(m, monkeypatch, tmp_path)
    default = m.replay(source, workdir=tmp_path / 'default')
    explicit = m.replay(source, workdir=tmp_path / 'explicit', arm='camera-projected')
    legacy_facade = reconstruct_facade(
        image, mask, columns=32, rows=32, depth_layers=8, horizon=None,
        camera=camera, scene_transform=None, palette=m.DEFAULT_PALETTE)
    assert default.facade.blocks == legacy_facade.blocks
    assert default.facade.blocks == explicit.facade.blocks
    assert default.project_path.read_bytes() == explicit.project_path.read_bytes()
    assert default.litematic_path.read_bytes() == explicit.litematic_path.read_bytes()
    assert default.route_a.frontier.to_record() == explicit.route_a.frontier.to_record()
    assert calls == ['solve', ('facade', camera), 'solve', ('facade', camera)]
    assert default.settings == m.RouteASettings()
    assert json.loads(default.candidate_seal)['normal_cli_parity_eligible'] is True


def test_pixel_grid_matches_existing_branch_and_keeps_camera_admission(replay_module, monkeypatch, tmp_path):
    m = replay_module
    source, image, mask, camera, calls = seams(m, monkeypatch, tmp_path)
    result = m.replay(source, workdir=tmp_path / 'grid', arm='pixel-grid')
    existing = reconstruct_facade(image, mask, columns=32, rows=32, depth_layers=8,
                                  horizon=None, camera=None, scene_transform=None, palette=m.DEFAULT_PALETTE)
    assert result.facade.blocks == existing.blocks
    assert calls == ['solve', ('facade', None)]
    assert result.camera is camera
    record = json.loads(result.candidate_seal)
    assert record['camera_admission'] == 'solved'
    assert record['settings'] == dict(arm='pixel-grid', columns=32, rows=32,
                                    depth_layers=8, horizon=None, scene_transform=None, scale_known=False)
    assert record['admitted_pixel_samples'] == len(existing.blocks)
    assert record['pixel_sample_collisions'] == 0
    assignments = record['pixel_sample_assignments']
    assert len(assignments) == len(existing.blocks)
    assert len({(c, r) for c, r, cell in assignments}) == len(assignments)
    assert {tuple(cell) for c, r, cell in assignments} == set(existing.blocks)
    assert record['normal_cli_parity_eligible'] is False
    assert all(0 <= c[2] < 8 for c in result.facade.blocks)
    assert any(c[2] == 0 for c in result.facade.blocks)
    with pytest.raises(FrozenInstanceError):
        result.settings.arm = 'camera-projected'
    before = result.candidate_seal
    result.facade.blocks.clear()
    assert result.candidate_seal == before


@pytest.mark.parametrize('arm', ['camera-projected', 'pixel-grid'])
def test_camera_failure_blocks_both_arms_identically(replay_module, monkeypatch, tmp_path, arm):
    m = replay_module
    source, _, _, _, calls = seams(m, monkeypatch, tmp_path, fail=True)
    with pytest.raises(m.FrontierRefusal, match='needs a solved camera'):
        m.replay(source, workdir=tmp_path / arm, arm=arm)
    assert calls == ['solve']
    assert not list((tmp_path / arm).iterdir())


def test_ablation_settings_cannot_be_tuned(replay_module, tmp_path):
    for change in ({'columns': 16}, {'rows': 16}, {'depth_layers': 1}, {'palette_jar': 'jar'}):
        with pytest.raises(replay_module.FrontierRefusal, match='fixed 32x32x8'):
            replay_module.replay('unused', workdir=tmp_path, arm='pixel-grid', **change)


def test_native_seal_is_written_before_pins_and_truth(report_module, replay_module, monkeypatch, tmp_path):
    r, m = report_module, replay_module
    source, _, _, _, calls = seams(m, monkeypatch, tmp_path)
    monkeypatch.setattr(r, '_route_a_replay', lambda: m)
    work = tmp_path / 'work'
    def pins(*_):
        record = json.loads((work / 'candidate-frozen.json').read_text())
        assert record['settings']['arm'] == 'pixel-grid'
        assert record['camera_admission'] == 'solved'
        assert record['native_artifacts']
        calls.append('pins')
        return {}
    def truth(**_):
        assert (work / 'candidate-frozen.json').is_file()
        calls.append('truth')
        raise r.AttributionRefusal('truth sentinel')
    monkeypatch.setattr(r, 'load_pins', pins)
    monkeypatch.setattr(r, 'load_frozen_truth', truth)
    assert r.main(['--route', 'route-a', '--route-a-arm', 'pixel-grid', '--image', str(source),
                   '--workdir', str(work), '--reference', 'unused', '--scene', 'unused',
                   '--cameras', 'unused', '--pins', 'unused', '--fixture-role', 'template-2']) == 2
    assert calls == ['solve', ('facade', None), 'pins', 'truth']


def test_cli_parity_is_refused_before_any_ablation_run(report_module, monkeypatch):
    def forbidden(**_):
        pytest.fail('a parity-labelled ablation must not run')
    monkeypatch.setattr(report_module, 'run_route_a', forbidden)
    assert report_module.main(['--route', 'route-a', '--route-a-arm', 'pixel-grid',
                               '--cli-parity', '--image', 'unused', '--reference', 'unused',
                               '--scene', 'unused', '--cameras', 'unused']) == 2
    outcome = SimpleNamespace(settings=SimpleNamespace(arm='pixel-grid'))
    with pytest.raises(report_module.AttributionRefusal, match='not normal CLI parity'):
        report_module._route_a_cli_parity(Path('unused'), outcome, Path('unused'))
    with pytest.raises(report_module.AttributionRefusal, match='not normal CLI parity'):
        report_module._score_route_a(image=Path('unused'), outcome=outcome,
                                     directory=Path('unused'), truth={}, cli_parity=True)


@pytest.mark.parametrize('flag, match', [('--expected-visible-cells', 'visible-cell'),
                                        ('--expected-reference-cells', 'reference-cell')])
def test_denominator_guard_preserves_fixed_universes(report_module, monkeypatch, flag, match, capsys):
    m = report_module
    monkeypatch.setattr(m, 'run_route_a', lambda **_: (object(), None))
    monkeypatch.setattr(m, 'load_frozen_truth', lambda **_: {
        'reference': {0: object()}, 'visible_reference': SimpleNamespace(visible={0})})
    monkeypatch.setattr(m, '_score_route_a', lambda **_: pytest.fail('mismatch must refuse'))
    assert m.main(['--route', 'route-a', '--route-a-arm', 'pixel-grid', '--image', 'unused',
                   '--reference', 'unused', '--scene', 'unused', '--cameras', 'unused',
                   flag, '398' if 'visible' in flag else '852']) == 2
    assert match in capsys.readouterr().out
