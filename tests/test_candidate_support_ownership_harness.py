"""Focused synthetic tests for the DS-IMG-SUPPORT-OWNERSHIP-R1 harness.

These tests are self-contained: they run no reconstruction, no production CLI,
no evaluator/solver path and no image/reference/camera/mask/JAR read. They pin
the bounded harness contract the Lead entry requires:

* Phase A uses only the accepted R4 session-bound capture path
  (``capture_ingress=True``) and the ordinary witness/support sinks; the removed
  side-channel hooks never reappear;
* Phase B is a typed unavailable when no *real* pinned evaluator ownership
  payload and camera/occlusion alignment exist, and synthetic ownership is used
  only inside these tests, never reported as a score;
* the evaluator payload is opened only after the Phase A freeze is recorded;
* malformed/tampered pinned payloads refuse typed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.pipeline.contracts import FrameTransform
from test_candidate_support_provenance import (
    _HEX,
    _binding,
    _pixels,
    _png_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "evaluation" / "run_candidate_support_ownership.py"


def _harness():
    name = "_r1_support_ownership_under_test"
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _payload_record(module, binding, *, camera="c" * 64, convention="z-buffer-v1",
                    tamper_sha=False):
    shape = binding.ingress.source_shape
    ownership = np.full(shape, -1, dtype=np.int32)
    ownership[1:3, 1:4] = 0
    usable = np.ones(shape, dtype=bool)
    record = module._array_record(ownership)
    if tamper_sha:
        record["sha256"] = "0" * 64
    return {
        "schema": module.OWNERSHIP_PAYLOAD_SCHEMA,
        "frames": [{
            "frame_id": binding.frame_id,
            "frame_size": [shape[1], shape[0]],
            "camera_sha256": camera,
            "occlusion_convention": convention,
            "transform": {
                "source_size": list(binding.ingress.transform.source_size),
                "crop_origin": [0, 0],
                "crop_size": None,
                "output_size": None,
            },
            "ownership": record,
            "usable": module._array_record(usable),
            "owner_plane_keys": {"-1": None, "0": ["x", 1, 1, 1]},
        }],
    }


def test_harness_uses_only_the_accepted_r4_capture_path():
    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert "capture_ingress=True" in source
    assert "signature_witness_sink=signature_witnesses" in source
    assert "candidate_support_sink=candidate_support.append" in source
    for hook in ("ingress_observer", "ingress_by_frame",
                 "candidate_support_ingress"):
        assert hook not in source


def test_phase_b_is_typed_unavailable_without_a_real_ownership_payload():
    module = _harness()
    phase_a = {"phase_a_seal_sha256": "a" * 64,
               "candidate_support": {"status": "assembled", "rows": []}}
    result = module.execute_phase_b(
        phase_a=phase_a, ownership_payload=None, witness_by_signature={})
    assert result["status"] == "unavailable"
    assert result["reason"] == "ownership_payload_unavailable"
    assert result["field"] == "ownership_payload"
    assert "projection" not in result


def test_phase_b_refuses_before_alignment_when_candidate_support_refused(tmp_path):
    module = _harness()
    payload = tmp_path / "payload.json"
    payload.write_text("{}", encoding="utf-8")

    def exploding_loader(_path):
        raise AssertionError("evaluator payload must not open for a refused capture")

    result = module.execute_phase_b(
        phase_a={"phase_a_seal_sha256": "a" * 64,
                 "candidate_support": {"status": "refused",
                                       "reason": "budget_exhausted",
                                       "field": "row_count"}},
        ownership_payload=payload, witness_by_signature={},
        inference_camera_sha256="c" * 64, occlusion_convention="z-buffer-v1",
        loader=exploding_loader)
    assert result["status"] == "unavailable"
    assert result["reason"] == "candidate_support_unavailable"
    assert result["field"] == "row_count"


def test_phase_b_refuses_unpinned_camera_or_occlusion_without_reading(tmp_path):
    module = _harness()
    payload = tmp_path / "payload.json"
    payload.write_text("{}", encoding="utf-8")
    phase_a = {"phase_a_seal_sha256": "a" * 64,
               "candidate_support": {"status": "assembled", "rows": []}}

    def exploding_loader(_path):
        raise AssertionError("alignment must be proven before the payload opens")

    camera = module.execute_phase_b(
        phase_a=phase_a, ownership_payload=payload, witness_by_signature={},
        loader=exploding_loader)
    assert camera["reason"] == "evaluator_camera_misalignment"
    occlusion = module.execute_phase_b(
        phase_a=phase_a, ownership_payload=payload, witness_by_signature={},
        inference_camera_sha256="c" * 64, loader=exploding_loader)
    assert occlusion["reason"] == "occlusion_unavailable"


def test_load_projected_ownership_refuses_malformed_and_tampered_payloads(tmp_path):
    module = _harness()
    binding = _binding("f", _pixels(31), _png_bytes((1, 2, 3)))

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(module.OwnershipPayloadError) as error:
        module.load_projected_ownership(empty)
    assert error.value.reason == "ownership_payload_unavailable"

    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(
        _payload_record(module, binding, tamper_sha=True)), encoding="utf-8")
    with pytest.raises(module.OwnershipPayloadError) as error:
        module.load_projected_ownership(tampered)
    assert error.value.reason == "byte_digest_mismatch"

    no_camera = _payload_record(module, binding, camera="")
    no_camera_path = tmp_path / "no_camera.json"
    no_camera_path.write_text(json.dumps(no_camera), encoding="utf-8")
    with pytest.raises(module.OwnershipPayloadError) as error:
        module.load_projected_ownership(no_camera_path)
    assert error.value.reason == "evaluator_camera_misalignment"


def test_phase_b_projects_a_pinned_synthetic_payload_only_inside_tests(tmp_path):
    module = _harness()
    binding = _binding("f", _pixels(32), _png_bytes((4, 5, 6)))
    payload = tmp_path / "real_shaped.json"
    payload.write_text(json.dumps(
        _payload_record(module, binding, camera=_HEX)), encoding="utf-8")
    evaluator = module.load_projected_ownership(payload)
    support_rows = [{
        "disposition": "kept", "left_signature_sha256": binding.signature_sha256,
        "right_signature_sha256": binding.signature_sha256}]
    projection = module._project_support_rows(
        support_rows=support_rows, witness_by_signature={
            binding.signature_sha256: binding},
        evaluator_by_frame=evaluator, inference_camera_sha256=_HEX,
        occlusion_convention="z-buffer-v1")
    assert projection["totals"]["rows"] == 1
    assert projection["totals"]["kept"] == 1
    assert projection["totals"]["projected"] == 2
    assert projection["abstentions"] == {}
    assert set(projection["cell_states"]) <= {"pure", "mixed", "unknown"}


def test_run_freezes_phase_a_before_opening_evaluator_material(
        monkeypatch, tmp_path):
    module = _harness()
    root = tmp_path / "run"
    payload = tmp_path / "payload.json"
    payload.write_text("{}", encoding="utf-8")
    opened: list[bool] = []

    monkeypatch.setattr(
        module, "_resolve_images",
        lambda images: tuple((REPO_ROOT / "fake.png", digest)
                             for _name, digest in images))
    monkeypatch.setattr(
        module, "_phase_a_payload",
        lambda run, **kwargs: {
            "stage_digests": {}, "signature_witnesses": {"coverage": {}},
            "candidate_support": {"status": "assembled", "rows": []},
            "comparison": {}})

    def sentinel_loader(_path):
        opened.append((root / "phase_a.json").is_file())
        return {}

    report = module.run(
        root=root, images=(("x", "a" * 64),), ownership_payload=payload,
        inference_camera_sha256="c" * 64, occlusion_convention="z-buffer-v1",
        chain=lambda **kwargs: SimpleNamespace(witness_by_signature={}),
        ownership_loader=sentinel_loader)
    # The evaluator payload loader ran exactly once, and only after the sealed
    # Phase A file existed on disk.
    assert opened == [True]
    assert report["phase_b"]["status"] == "assembled"
    phase_a = json.loads((root / "phase_a.json").read_text(encoding="utf-8"))
    seal = phase_a.pop("phase_a_seal_sha256")
    assert seal == module._digest(phase_a)
    assert report["phase_a"]["file_sha256"] == module._file_sha256(
        root / "phase_a.json")


def test_run_refuses_a_non_fresh_output_root(tmp_path):
    module = _harness()
    root = tmp_path / "run"
    root.mkdir()
    (root / "leftover.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(module.HarnessError):
        module.run(root=root)
