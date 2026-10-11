from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from mc_photogrammetry import evaluation_frozen_planes as evaluator
from mc_photogrammetry.export import attribution
from mc_photogrammetry.reconstruction import lattice
from test_frozen_plane_evaluation import _camera, _single_block_truth
from test_rendered_truth_evidence import _artifacts, _install_palette
from mc_photogrammetry.domain.models import BlockState


def _render(tmp_path, monkeypatch, *, sink=None, payload_sink=None, budget=None):
    truth = _single_block_truth(BlockState("minecraft:stone"))
    paths = _artifacts(tmp_path, truth)
    _install_palette(monkeypatch)
    kwargs = {}
    if sink is not None:
        kwargs.update(ownership_witness_sink=sink, witness_budget=budget)
    if payload_sink is not None:
        kwargs.update(ownership_witness_payload_sink=payload_sink,
                      witness_budget=budget)
    return evaluator.render_truth_evidence(
        truth, _camera(), width=96, height=72,
        truth_artifact_path=paths[0], vanilla_jar_path=paths[1],
        palette_cache_path=paths[2], **kwargs)


def test_optional_binding_preserves_evidence_and_binds_exact_hashes(tmp_path, monkeypatch):
    plain = _render(tmp_path, monkeypatch)
    sink = []
    observed = _render(tmp_path, monkeypatch, sink=sink)
    assert observed.evidence_sha256 == plain.evidence_sha256 and len(sink) == 1
    np.testing.assert_array_equal(observed.ownership, plain.ownership)
    binding = sink[0]
    assert isinstance(binding, evaluator.RenderedTruthOwnershipBindingV1)
    assert binding.evidence_sha256 == observed.evidence_sha256
    assert binding.ownership_sha256 == hashlib.sha256(
        observed.ownership.tobytes(order="C")).hexdigest()
    payload = [[list(position), slot] for position, slot in observed.owner_index]
    assert binding.owner_index_sha256 == hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def test_binding_is_deterministic_and_tamper_replay_fails(tmp_path, monkeypatch):
    first, second = [], []
    _render(tmp_path, monkeypatch, sink=first)
    _render(tmp_path, monkeypatch, sink=second)
    assert first == second
    replay = evaluator.RenderedTruthOwnershipBindingV1(**{
        name: getattr(first[0], name) for name in first[0].__dataclass_fields__})
    assert replay == first[0]
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="digest mismatch"):
        replace(first[0], ownership_sha256="0" * 64)


def test_owner_budget_refusal_is_typed_atomic_and_evidence_unchanged(tmp_path, monkeypatch):
    plain = _render(tmp_path, monkeypatch)
    sink = []
    observed = _render(
        tmp_path, monkeypatch, sink=sink,
        budget=lattice.WitnessBudgetV1(max_owner_faces=0))
    assert observed.evidence_sha256 == plain.evidence_sha256
    np.testing.assert_array_equal(observed.ownership, plain.ownership)
    assert len(sink) == 1 and isinstance(sink[0], lattice.WitnessRefusalV1)
    assert sink[0].stage == "owner_faces" and sink[0].actual == 0


def test_owner_index_mismatch_is_rejected(tmp_path, monkeypatch):
    captured = []
    original = attribution.render_result
    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("witness_sink"):
            captured.append(kwargs["witness_sink"][0])
        return result
    monkeypatch.setattr(attribution, "render_result", wrapper)
    sink = []
    evidence = _render(tmp_path, monkeypatch, sink=sink)
    witness = captured[0]
    row = replace(witness.owners[0], block_coordinate=(99, 99, 99))
    forged = replace(witness, owners=(row,) + witness.owners[1:], witness_sha256="")
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="does not bind"):
        evaluator.bind_renderer_ownership_witness(evidence, forged)


def test_private_owner_payload_binds_witness_and_rejects_prepopulated_sink(
        tmp_path, monkeypatch):
    sink = []
    evidence = _render(tmp_path, monkeypatch, payload_sink=sink)
    assert len(sink) == 1
    payload = sink[0]
    assert isinstance(payload, evaluator.RenderedTruthOwnershipWitnessPayloadV1)
    assert payload.binding == evaluator.bind_renderer_ownership_witness(
        evidence, payload.witness)
    assert evaluator.RenderedTruthOwnershipWitnessPayloadV1(
        payload.binding, payload.witness) == payload
    with pytest.raises(evaluator.FrozenPlaneEvaluationError, match="must be empty"):
        _render(tmp_path, monkeypatch, payload_sink=[payload])
