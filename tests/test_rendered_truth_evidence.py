from dataclasses import replace
import json

import numpy as np
import pytest

from mc_photogrammetry import evaluation_frozen_planes as evaluator
from mc_photogrammetry.blocks.textures import PaletteColor
from mc_photogrammetry.domain.models import BlockState
from test_frozen_plane_evaluation import _camera, _single_block_truth


def _artifacts(tmp_path, truth):
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth.to_dict()), encoding="utf-8")
    jar = tmp_path / "client.jar"
    cache = tmp_path / "palette.json"
    jar.write_bytes(b"jar")
    cache.write_bytes(b"palette")
    return truth_path, jar, cache


def _install_palette(monkeypatch):
    state = BlockState("minecraft:stone")
    palette = (PaletteColor(state, (200, 200, 200)),)
    monkeypatch.setattr(evaluator, "_load_palette",
                        lambda *_args: (palette, "a" * 64, "b" * 64, []))


def test_evidence_preserves_default_renderer_masks(tmp_path, monkeypatch):
    truth = _single_block_truth(BlockState("minecraft:stone"))
    paths = _artifacts(tmp_path, truth)
    _install_palette(monkeypatch)
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    legacy = evaluator.render_truth_planes(
        truth, _camera(), image, vanilla_jar_path=paths[1],
        palette_cache_path=paths[2])
    evidence = evaluator.render_truth_evidence(
        truth, _camera(), width=96, height=72,
        truth_artifact_path=paths[0], vanilla_jar_path=paths[1],
        palette_cache_path=paths[2])
    actual = {(row.direction, row.coordinate): row.mask for row in evidence.surfaces}
    assert actual.keys() == legacy.keys()
    assert all(np.array_equal(actual[key], legacy[key]) for key in legacy)
    assert not evidence.rgb.flags.writeable
    assert not evidence.ownership.flags.writeable
    assert not evidence.usable.flags.writeable


def test_digest_replay_and_tampering(tmp_path, monkeypatch):
    truth = _single_block_truth(BlockState("minecraft:stone"))
    paths = _artifacts(tmp_path, truth)
    _install_palette(monkeypatch)
    evidence = evaluator.render_truth_evidence(
        truth, _camera(), width=96, height=72,
        truth_artifact_path=paths[0], vanilla_jar_path=paths[1],
        palette_cache_path=paths[2])
    replay = evaluator.RenderedTruthEvidence(**{
        name: getattr(evidence, name) for name in evidence.__dataclass_fields__})
    assert replay == evidence and replay.evidence_sha256 == evidence.evidence_sha256
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="digest"):
        replace(evidence, truth_sha256="0" * 64)
    mutable = evidence.ownership.copy()
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="mutable"):
        replace(evidence, ownership=mutable, evidence_sha256="")
    changed = evidence.ownership.copy()
    changed.reshape(-1)[0] = changed.reshape(-1)[0] + 1
    changed.flags.writeable = False
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="digest"):
        replace(evidence, ownership=changed)


@pytest.mark.parametrize("kwargs,match", [
    ({"max_pixels": 1}, "pixel cap"),
    ({"max_blocks": 0}, "block cap"),
    ({"max_artifact_bytes": 1}, "byte cap"),
    ({"max_evidence_bytes": 1}, "retained-byte cap"),
])
def test_all_resource_caps_fail_closed(tmp_path, monkeypatch, kwargs, match):
    truth = _single_block_truth(BlockState("minecraft:stone"))
    paths = _artifacts(tmp_path, truth)
    _install_palette(monkeypatch)
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match=match):
        evaluator.render_truth_evidence(
            truth, _camera(), width=96, height=72,
            truth_artifact_path=paths[0], vanilla_jar_path=paths[1],
            palette_cache_path=paths[2], **kwargs)


def test_truth_artifact_hash_and_content_are_bound(tmp_path, monkeypatch):
    truth = _single_block_truth(BlockState("minecraft:stone"))
    paths = _artifacts(tmp_path, truth)
    _install_palette(monkeypatch)
    payload = json.loads(paths[0].read_text())
    payload["scene_id"] = "foreign"
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="does not bind"):
        evaluator.render_truth_evidence(
            truth, _camera(), width=96, height=72,
            truth_artifact_path=paths[0], vanilla_jar_path=paths[1],
            palette_cache_path=paths[2])


def test_evidence_module_has_no_production_or_launcher_coupling():
    source = open(evaluator.__file__, encoding="utf-8").read()
    assert "image_derived_production" not in source
    assert "PrismCaptureLauncher" not in source
