"""Synthetic-only guards for the headless blinded annotation workflow."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/evaluation/landmark_annotation_bundle.py"
_SPEC = importlib.util.spec_from_file_location("landmark_annotation_bundle_tests", _PATH)
bundle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bundle
_SPEC.loader.exec_module(bundle)
qa = bundle.qa


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def plan_fixture():
    doc = {
        "schema": qa.PLAN_SCHEMA, "evaluation_only": True,
        "policy_sha256": qa.POLICY_SHA256,
        "world_sha256": "1" * 64, "model_sha256": "2" * 64,
        "context_sha256": "3" * 64,
        "landmarks": [{
            "landmark_id": "corner-a", "world_xyz": [1.0, 2.0, 3.0],
            "surface_id": "wall-east", "structural_identity": "stone-to-wood corner",
            "identity_cue": "upper left natural corner above the doorway",
            "support_class": "natural_corner", "required": True,
        }],
    }
    raw = canon(doc)
    return raw, hashlib.sha256(raw).hexdigest()


def png_fixture(path):
    # Minimal bounded PNG header fixture; the workflow intentionally reads IHDR
    # and bytes only, and never invokes an image decoder.
    ihdr = b"IHDR" + struct.pack(">II", 640, 360) + bytes([8, 2, 0, 0, 0])
    header = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + ihdr
              + struct.pack(">I", zlib.crc32(ihdr) & 0xffffffff))
    path.write_bytes(header + b"synthetic-test-tail")
    return header + b"synthetic-test-tail"


def binding_for(raw):
    return qa.FrameBinding("fresh-frame-a", hashlib.sha256(raw).hexdigest(), 640, 360,
                           "fresh-capture-9", "1" * 64, "2" * 64, "3" * 64)


def packet(plan_raw, plan_sha, binding, observer, uv=(100.25, 80.5), disposition="observed"):
    row = {"landmark_id": "corner-a", "disposition": disposition}
    if disposition == "observed":
        row.update({"uv": list(uv), "uncertainty_pixels": 0.5,
                    "uncertainty_method": "manual_localization_bound",
                    "support_segments": [[[99.0, 80.0], [101.0, 81.0]]],
                    "second_annotation": None})
    else:
        row["reason"] = "cannot identify the corner in this raster"
    doc = {
        "schema": qa.PACKET_SCHEMA, "evaluation_only": True,
        "method": "manual_png_observation", "observer_id": observer,
        "tool_version": "vision-annotator-test-1", "protocol": qa.PROTOCOL,
        "pixel_convention": qa.PIXEL_CONVENTION,
        "projected_coordinates_exposed": False, "frozen_before_projection": True,
        "landmark_plan_sha256": plan_sha, "policy_sha256": qa.POLICY_SHA256,
        "frame_id": binding.frame_id, "png_sha256": binding.png_sha256,
        "dimensions": [binding.width, binding.height], "capture_id": binding.capture_id,
        "world_sha256": binding.world_sha256, "model_sha256": binding.model_sha256,
        "context_sha256": binding.context_sha256, "observations": [row],
    }
    return canon(doc)


def test_creates_two_camera_free_exact_png_inputs(tmp_path):
    png = tmp_path / "fresh.png"
    raw_png = png_fixture(png)
    binding = binding_for(raw_png)
    plan_raw, plan_sha = plan_fixture()
    out = tmp_path / "inputs"
    result = bundle.create_annotation_inputs(
        png_path=png, binding=binding, plan_raw=plan_raw, plan_sha256=plan_sha,
        annotators=(("reviewer-a", "tool-a"), ("reviewer-b", "tool-b")),
        output_root=out)
    assert set(result["annotators"][i]["annotator_id"] for i in range(2)) == {
        "reviewer-a", "reviewer-b"}
    for reviewer in ("reviewer-a", "reviewer-b"):
        assert (out / reviewer / "frame.png").read_bytes() == raw_png
        doc = json.loads((out / reviewer / "annotation-input.json").read_bytes())
        assert doc["candidate_camera_exposed"] is False
        assert doc["mod_actual_exposed"] is False
        assert doc["projected_coordinates_exposed"] is False
        assert doc["packet_template"]["projected_coordinates_exposed"] is None
        assert doc["packet_template"]["frozen_before_projection"] is None
        assert "Annotator must author" in doc["attestation_instructions"][
            "projected_coordinates_exposed"]
        assert doc["packet_template"]["projected_coordinates_exposed"] is None
        assert doc["packet_template"]["frozen_before_projection"] is None
        assert "Annotator must author" in doc["attestation_instructions"][
            "projected_coordinates_exposed"]
        assert doc["landmarks"][0]["landmark_id"] == "corner-a"
        assert "uv" not in doc["landmarks"][0]
        assert doc["packet_template"]["observations"] == [
            {"landmark_id": "corner-a", "disposition": None}]
    with pytest.raises(bundle.BundleError, match="output_exists"):
        bundle.create_annotation_inputs(
            png_path=png, binding=binding, plan_raw=plan_raw, plan_sha256=plan_sha,
            annotators=(("reviewer-a", "tool-a"), ("reviewer-b", "tool-b")),
            output_root=out)


def test_input_creation_refuses_wrong_hash_dimensions_and_duplicate_annotators(tmp_path):
    png = tmp_path / "fresh.png"
    raw_png = png_fixture(png)
    plan_raw, plan_sha = plan_fixture()
    binding = binding_for(raw_png)
    wrong_hash = qa.FrameBinding(binding.frame_id, "f" * 64, 640, 360,
                                 binding.capture_id, binding.world_sha256,
                                 binding.model_sha256, binding.context_sha256)
    with pytest.raises(bundle.BundleError, match="png_binding_mismatch"):
        bundle.create_annotation_inputs(
            png_path=png, binding=wrong_hash, plan_raw=plan_raw, plan_sha256=plan_sha,
            annotators=(("a", "t"), ("b", "t")), output_root=tmp_path / "bad-hash")
    wrong_dimensions = qa.FrameBinding(binding.frame_id, binding.png_sha256, 639, 360,
                                       binding.capture_id, binding.world_sha256,
                                       binding.model_sha256, binding.context_sha256)
    with pytest.raises(bundle.BundleError, match="png_binding_mismatch"):
        bundle.create_annotation_inputs(
            png_path=png, binding=wrong_dimensions, plan_raw=plan_raw, plan_sha256=plan_sha,
            annotators=(("a", "t"), ("b", "t")), output_root=tmp_path / "bad-size")
    with pytest.raises(bundle.BundleError, match="duplicate_annotator_identity"):
        bundle.create_annotation_inputs(
            png_path=png, binding=binding, plan_raw=plan_raw, plan_sha256=plan_sha,
            annotators=(("same", "t1"), ("same", "t2")), output_root=tmp_path / "same")


@pytest.mark.parametrize("ids", [
    (("reviewer", "a"), ("REVIEWER", "b")),
    (("annotator", "a"), ("annotator.", "b")),
    (("CON", "a"), ("second", "b")),
    (("..\\escape", "a"), ("second", "b")),
])
def test_annotator_ids_reject_windows_collisions_reserved_and_paths(tmp_path, ids):
    png = tmp_path / "fresh.png"
    raw_png = png_fixture(png)
    plan_raw, plan_sha = plan_fixture()
    with pytest.raises(bundle.BundleError):
        bundle.create_annotation_inputs(
            png_path=png, binding=binding_for(raw_png), plan_raw=plan_raw,
            plan_sha256=plan_sha, annotators=ids, output_root=tmp_path / "bad-id")


def test_invalid_binding_is_normalized_to_bundle_error(tmp_path):
    png = tmp_path / "fresh.png"
    raw_png = png_fixture(png)
    plan_raw, plan_sha = plan_fixture()
    binding = binding_for(raw_png)
    bad = qa.FrameBinding(binding.frame_id, "not-a-digest", binding.width, binding.height,
                          binding.capture_id, binding.world_sha256,
                          binding.model_sha256, binding.context_sha256)
    with pytest.raises(bundle.BundleError, match="invalid_binding_invalid_sha256"):
        bundle.create_annotation_inputs(
            png_path=png, binding=bad, plan_raw=plan_raw, plan_sha256=plan_sha,
            annotators=(("a", "tool"), ("b", "tool")), output_root=tmp_path / "bad-binding")
    with pytest.raises(bundle.BundleError, match="invalid_binding_invalid_frame_binding"):
        bundle.create_annotation_inputs(
            png_path=png, binding=object(), plan_raw=plan_raw, plan_sha256=plan_sha,
            annotators=(("a", "tool"), ("b", "tool")), output_root=tmp_path / "foreign-binding")


def test_cli_binding_cap_is_checked_before_open(tmp_path, monkeypatch):
    binding_path = tmp_path / "too-large-binding.json"
    binding_path.write_bytes(b" " * (bundle.MAX_BINDING_BYTES + 1))
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == binding_path:
            raise AssertionError("oversized binding must be rejected before opening/reading")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    argv = ["create-inputs", "--png", str(tmp_path / "unused.png"),
            "--binding-json", str(binding_path), "--plan", str(tmp_path / "unused-plan"),
            "--plan-sha256", "0" * 64, "--annotator", "a", "tool-a",
            "--annotator", "b", "tool-b", "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="binding_byte_cap"):
        bundle.main(argv)


@pytest.mark.parametrize("kind,cap", [
    ("plan", bundle.MAX_PLAN_BYTES),
    ("first_observation", bundle.MAX_OBSERVATION_BYTES),
    ("second_observation", bundle.MAX_OBSERVATION_BYTES),
])
def test_cli_plan_and_packet_caps_precede_read(tmp_path, monkeypatch, kind, cap):
    plan_raw, plan_sha = plan_fixture()
    binding = binding_for(b"unused png bytes")
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(canon({
        "frame_id": binding.frame_id, "png_sha256": binding.png_sha256,
        "dimensions": [binding.width, binding.height], "capture_id": binding.capture_id,
        "world_sha256": binding.world_sha256, "model_sha256": binding.model_sha256,
        "context_sha256": binding.context_sha256,
    }))
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b" " * (cap + 1) if kind == "plan" else plan_raw)
    for name in ("unused-a", "unused-b", "small-a", "oversized"):
        path = tmp_path / name
        path.write_bytes(b" " * (cap + 1) if name == "oversized" else b"{}")
    original_open = Path.open
    forbidden = tmp_path / ("plan.json" if kind == "plan" else "oversized")

    def guarded_open(path, *args, **kwargs):
        if path == forbidden:
            raise AssertionError("oversized CLI input must be rejected before opening/reading")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    argv = ["merge", "--binding-json", str(binding_path), "--plan", str(plan_path),
            "--plan-sha256", plan_sha, "--annotator", "a", "b", "--out", str(tmp_path / "merged")]
    first_name = "oversized" if kind == "first_observation" else "small-a"
    second_name = "oversized" if kind == "second_observation" else "unused-b"
    argv.extend(["--first", str(tmp_path / first_name), "--first-sha256", "0" * 64,
                 "--second", str(tmp_path / second_name), "--second-sha256", "0" * 64])
    expected_code = "plan_byte_cap" if kind == "plan" else kind + "_byte_cap"
    with pytest.raises(SystemExit, match=expected_code):
        bundle.main(argv)


def test_merge_preserves_two_annotations_and_is_deterministic(tmp_path):
    raw_png = png_fixture(tmp_path / "fresh.png")
    binding = binding_for(raw_png)
    plan_raw, plan_sha = plan_fixture()
    first = packet(plan_raw, plan_sha, binding, "reviewer-a")
    second = packet(plan_raw, plan_sha, binding, "reviewer-b", uv=(101.0, 80.0))
    args = dict(plan_raw=plan_raw, plan_sha256=plan_sha,
                first_raw=first, first_sha256=hashlib.sha256(first).hexdigest(),
                second_raw=second, second_sha256=hashlib.sha256(second).hexdigest(),
                binding=binding, expected_annotators=("reviewer-a", "reviewer-b"))
    merged, audit = bundle.merge_observations(**args)
    merged_again, audit_again = bundle.merge_observations(**args)
    assert merged == merged_again and audit == audit_again
    row = json.loads(merged)["observations"][0]
    assert row["uv"] == [100.25, 80.5]
    assert row["second_annotation"] == {
        "observer_id": "reviewer-b", "uv": [101.0, 80.0],
        "uncertainty_pixels": 0.5}
    audit_doc = json.loads(audit)
    assert audit_doc["annotations"][0]["packet_sha256"] == hashlib.sha256(first).hexdigest()
    assert audit_doc["annotations"][1]["packet_sha256"] == hashlib.sha256(second).hexdigest()


def test_packet_blinding_declarations_are_annotator_authored_and_validated(tmp_path):
    raw_png = png_fixture(tmp_path / "fresh.png")
    binding = binding_for(raw_png)
    plan_raw, plan_sha = plan_fixture()
    authored = json.loads(packet(plan_raw, plan_sha, binding, "reviewer-a"))
    authored["projected_coordinates_exposed"] = True
    raw = canon(authored)
    plan = qa.load_landmark_plan(plan_raw, expected_sha256=plan_sha)
    with pytest.raises(qa.QualificationError, match="observation_not_blinded"):
        qa.seal_observations(raw, plan, expected_packet_sha256=hashlib.sha256(raw).hexdigest(),
                             expected_binding=binding)


def test_merge_fails_closed_for_abstention_mismatch_or_binding_drift(tmp_path):
    raw_png = png_fixture(tmp_path / "fresh.png")
    binding = binding_for(raw_png)
    plan_raw, plan_sha = plan_fixture()
    first = packet(plan_raw, plan_sha, binding, "reviewer-a")
    second = packet(plan_raw, plan_sha, binding, "reviewer-b", disposition="occluded")
    merged, _audit = bundle.merge_observations(
        plan_raw=plan_raw, plan_sha256=plan_sha,
        first_raw=first, first_sha256=hashlib.sha256(first).hexdigest(),
        second_raw=second, second_sha256=hashlib.sha256(second).hexdigest(),
        binding=binding, expected_annotators=("reviewer-a", "reviewer-b"))
    row = json.loads(merged)["observations"][0]
    assert row["disposition"] == "ambiguous"
    assert "observed; occluded" in row["reason"]
    with pytest.raises(bundle.BundleError, match="annotator_assignment_mismatch"):
        bundle.merge_observations(
            plan_raw=plan_raw, plan_sha256=plan_sha,
            first_raw=first, first_sha256=hashlib.sha256(first).hexdigest(),
            second_raw=packet(plan_raw, plan_sha, binding, "reviewer-a"),
            second_sha256=hashlib.sha256(packet(plan_raw, plan_sha, binding, "reviewer-a")).hexdigest(),
            binding=binding, expected_annotators=("reviewer-a", "reviewer-b"))
    bad_binding = qa.FrameBinding(binding.frame_id, "f" * 64, binding.width, binding.height,
                                  binding.capture_id, binding.world_sha256,
                                  binding.model_sha256, binding.context_sha256)
    with pytest.raises(bundle.BundleError, match="observation_packet_frame_binding_mismatch"):
        bundle.merge_observations(
            plan_raw=plan_raw, plan_sha256=plan_sha,
            first_raw=first, first_sha256=hashlib.sha256(first).hexdigest(),
            second_raw=packet(plan_raw, plan_sha, bad_binding, "reviewer-b"),
            second_sha256=hashlib.sha256(packet(plan_raw, plan_sha, bad_binding, "reviewer-b")).hexdigest(),
            binding=binding, expected_annotators=("reviewer-a", "reviewer-b"))
