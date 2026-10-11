from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from mc_photogrammetry.dataset.runtime_render_provenance import (
    MAX_PACK_FILE_BYTES,
    ProvenanceError,
    directory_manifest,
    canonical_manifest_json,
    seal_sidecar,
    validate_sidecar,
    verify_seal,
    zip_manifest,
    canonical_json,
)


def _complete_sidecar(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    artifact_paths = {}
    for name, payload in (("minecraft", b"minecraft runtime"),
                          ("fabricloader", b"fabric loader"),
                          ("mc_photo_capture", b"capture mod")):
        path = artifact_root / f"{name}.jar"
        path.write_bytes(payload)
        artifact_paths[name] = path
    capture_mod_sha = hashlib.sha256(artifact_paths["mc_photo_capture"].read_bytes()).hexdigest()
    pack_root = tmp_path / "vanilla-pack"
    (pack_root / "assets" / "test").mkdir(parents=True, exist_ok=True)
    (pack_root / "assets" / "test" / "sample.bin").write_bytes(b"verified pack")
    pack_manifest = directory_manifest(pack_root)
    pack_order = ["vanilla"]
    pack_order_sha = hashlib.sha256(canonical_json(pack_order)).hexdigest()
    combined_manifest_sha = hashlib.sha256(
        canonical_json([pack_manifest["sha256"]])).hexdigest()
    program_sha = "d" * 64
    minecraft_sha = hashlib.sha256(artifact_paths["minecraft"].read_bytes()).hexdigest()
    render_policy = {"status": "verified", "pipelines": [
        {"id": "solid", "render_layer": "opaque", "artifact_sha256": minecraft_sha,
         "resource_manifest_sha256": pack_manifest["sha256"],
         "program_sha256": program_sha}]}
    render_policy_sha = hashlib.sha256(canonical_json(render_policy)).hexdigest()
    render_policy["sha256"] = render_policy_sha
    expected = {
        "attempt_nonce": "a" * 32,
        "plan_id": "plan-1",
        "plan_sha256": "1" * 64,
        "dataset_plan_sha256": "6" * 64,
        "staging_token": "2" * 64,
        "capture_mod_sha256": capture_mod_sha,
        "capture_mod_source_sha256": capture_mod_sha,
        "capture_mod_staged_sha256": capture_mod_sha,
        "scene_id": "scene-1", "world_identity": "world-1",
        "mode": "strict", "expectation_sha256": "7" * 64,
    }
    frames = [{"frame_id": "f0", "filename": "f0.png",
               "screenshot_sha256": "4" * 64, "screenshot_bytes": 123}]
    snapshot = {"frame_id": "f0", "capture_epoch_status": "bound",
                "resource_epoch_status": "bound", "resource_order_stable": True,
                "ordered_resource_packs": pack_order,
                "resource_pack_order_sha256": pack_order_sha,
                "resource_epoch_id": "resource-1", "program_epoch_id": "program-1",
                "program_sha256": program_sha,
                "acquisition_epoch": {"status": "readback_bound", "id": "readback-0",
                                      "resource_epoch_id": "resource-1",
                                      "program_epoch_id": "program-1",
                                      "screenshot_sha256": frames[0]["screenshot_sha256"],
                                      "screenshot_bytes": frames[0]["screenshot_bytes"]},
                "gl": {"vendor": "test", "renderer": "test", "version": "test",
                       "glsl_version": "test"}}
    manifest = {"kind": "directory", "entries": [
        {"path": "assets/test.bin", "bytes": 1, "sha256": "a" * 64}],
        "entry_count": 1, "total_bytes": 1}
    manifest["sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    snapshot_digest = hashlib.sha256(canonical_json(snapshot)).hexdigest()
    doc = {
        "schema_version": 1,
        **{key: expected[key] for key in (
            "attempt_nonce", "plan_id", "plan_sha256", "dataset_plan_sha256",
            "staging_token", "scene_id", "world_identity", "mode", "expectation_sha256")},
        "runtime": {
            "minecraft_version": "1.21.5", "fabric_loader_version": "0.16.10",
            "java_version": "21", "capture_mod_sha256": expected["capture_mod_sha256"],
            "capture_mod_source_sha256": expected["capture_mod_source_sha256"],
            "capture_mod_staged_sha256": expected["capture_mod_staged_sha256"],
            "capture_mod_loaded_sha256": capture_mod_sha,
            "client_artifacts": [
                {"id": name, "origins": [str(artifact_paths[name])],
                 "sha256": hashlib.sha256(artifact_paths[name].read_bytes()).hexdigest()}
                for name in ("minecraft", "fabricloader", "mc_photo_capture")
            ], "identity_status": "complete",
        },
        "resource_status": "complete", "resource_pack_order_sha256": pack_order_sha,
        "resource_epochs": [{"id": "resource-1", "pack_order_sha256": pack_order_sha,
                             "manifest_sha256": combined_manifest_sha}],
        "program_epochs": [{"id": "program-1", "sha256": program_sha,
                             "shader_identity_sha256": minecraft_sha,
                             "render_policy_sha256": render_policy_sha}],
        "certification": {"status": "verified", "unavailable_reasons": []},
        "shader_identity": {"status": "verified", "artifact_id": "minecraft",
                             "sha256": minecraft_sha},
        "resource_packs": [
            {"id": "vanilla", "origin": str(pack_root), "manifest": pack_manifest}
        ], "shader_status": "vanilla_builtin", "render_policy": render_policy,
        "gl_status": "complete",
        "frames": [{**frames[0], "snapshot_sha256": snapshot_digest,
                     "capture_epoch_status": "bound", "snapshot": snapshot}],
    }
    return expected, frames, seal_sidecar(doc)


def test_canonical_seal_and_strict_identity_frame_join(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    seal = verify_seal(doc)
    assert seal == doc["seal_sha256"]
    assert validate_sidecar(doc, expected=expected, expected_frames=frames,
                            require_certifiable=False) == seal
    with pytest.raises(ProvenanceError, match="strict render certification unavailable"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)

    altered = dict(doc)
    altered["attempt_nonce"] = "b" * 32
    with pytest.raises(ProvenanceError, match="seal"):
        validate_sidecar(altered, expected=expected, expected_frames=frames)


def test_shared_python_java_golden_bytes_and_digest_match():
    golden_path = (Path(__file__).resolve().parents[1] / "tools" / "minecraft-capture-mod" /
                   "src" / "test" / "resources" /
                   "runtime_render_provenance_canonical.golden")
    raw = golden_path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    assert canonical_json(document) == raw
    assert hashlib.sha256(raw).hexdigest() == (
        "63e6f86bf176989980e2f19a9dffb4b08cf5f67a996c7fb0f74d89e992abc2f8")


def test_canonical_domain_rejects_float_numbers_for_cross_language_stability():
    with pytest.raises(ProvenanceError, match="floating-point"):
        canonical_json({"value": 2.5})
    with pytest.raises(ProvenanceError, match="signed 64-bit"):
        canonical_json({"value": 2**63})
    with pytest.raises(ProvenanceError, match="signed 64-bit"):
        canonical_json({"value": -(2**63) - 1})


def test_canonical_unicode_key_order_matches_unicode_codepoint_order():
    assert canonical_json({"😀": 2, "\ue000": 1}) == (
        '{"\ue000":1,"\\ud83d\\ude00":2}'.encode("utf-8"))
    assert canonical_json({"max": 2**63 - 1, "min": -(2**63)}) == (
        b'{"max":9223372036854775807,"min":-9223372036854775808}')
    assert canonical_json(json.loads('{"value":-0}')) == b'{"value":0}'


def test_canonical_size_is_refused_before_json_dumps_allocation(monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    monkeypatch.setattr(provenance.json, "dumps",
                        lambda *_args, **_kwargs: pytest.fail("serialization allocated before cap"))
    with pytest.raises(ProvenanceError, match="byte cap"):
        canonical_json({"value": "x" * (provenance.MAX_SIDECAR_BYTES + 1)})


def test_bounded_file_hash_detects_path_replacement_before_open(tmp_path, monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    target = tmp_path / "pack.bin"
    backup = tmp_path / "original.bin"
    target.write_bytes(b"original")
    real_open = Path.open
    swapped = False

    def replace_then_open(path, *args, **kwargs):
        nonlocal swapped
        if path == target and args and args[0] == "rb" and not swapped:
            swapped = True
            path.rename(backup)
            path.write_bytes(b"replacement")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", replace_then_open)
    with pytest.raises(ProvenanceError, match="changed before hashing"):
        provenance.sha256_file_bounded(target)


def test_directory_manifest_detects_parent_replacement_during_scan(tmp_path, monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    root = tmp_path / "pack"
    backup = tmp_path / "pack-old"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    real_scandir = provenance.os.scandir
    replaced = False

    def replace_then_scan(path):
        nonlocal replaced
        if Path(path) == root and not replaced:
            replaced = True
            root.rename(backup)
            root.mkdir()
            (root / "b.bin").write_bytes(b"b")
        return real_scandir(path)

    monkeypatch.setattr(provenance.os, "scandir", replace_then_scan)
    with pytest.raises(ProvenanceError, match="directory changed during enumeration"):
        provenance.directory_manifest(root)


def test_bounded_snapshot_refuses_linklike_input(tmp_path, monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    target = tmp_path / "sidecar.json"
    target.write_bytes(b"{}")
    monkeypatch.setattr(provenance, "_linklike", lambda _path: True)
    with pytest.raises(ProvenanceError, match="symlink"):
        provenance.read_bounded_file_snapshot(target)


@pytest.mark.parametrize("field,value,match", [
    ("filename", "other.png", "filename mismatch"),
    ("screenshot_sha256", "0" * 64, "screenshot_sha256 mismatch"),
    ("capture_epoch_status", "unbound", "contradicts snapshot"),
])
def test_frame_identity_mismatch_fails_closed(tmp_path, field, value, match):
    expected, frames, doc = _complete_sidecar(tmp_path)
    doc["frames"][0][field] = value
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match=match):
        validate_sidecar(doc, expected=expected, expected_frames=frames)


def test_unknown_runtime_resource_shader_or_gl_blocks_certification(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    doc["runtime"]["identity_status"] = "unavailable"
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="runtime identity"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)
    assert validate_sidecar(doc, expected=expected, expected_frames=frames,
                            require_certifiable=False) == doc["seal_sha256"]


def test_empty_artifact_inventory_cannot_claim_complete_identity(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    doc["runtime"]["client_artifacts"] = []
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="inventory is empty"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)


def test_two_frame_sidecar_uses_per_frame_snapshot_joins_not_cumulative_hashes(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    second = {"frame_id": "f1", "filename": "f1.png",
              "screenshot_sha256": "a" * 64, "screenshot_bytes": 321}
    second_snapshot = {"frame_id": "f1", "capture_epoch_status": "bound",
                       "resource_epoch_status": "bound", "resource_order_stable": True,
                       "ordered_resource_packs": ["vanilla"],
                       "resource_pack_order_sha256": doc["resource_pack_order_sha256"],
                       "resource_epoch_id": "resource-1", "program_epoch_id": "program-1",
                       "program_sha256": doc["program_epochs"][0]["sha256"],
                       "acquisition_epoch": {"status": "readback_bound", "id": "readback-1",
                                             "resource_epoch_id": "resource-1",
                                             "program_epoch_id": "program-1",
                                             "screenshot_sha256": second["screenshot_sha256"],
                                             "screenshot_bytes": second["screenshot_bytes"]},
                       "gl": {"vendor": "test", "renderer": "test", "version": "test",
                              "glsl_version": "test"}}
    doc["frames"].append({**second,
                          "snapshot_sha256": hashlib.sha256(
                              canonical_json(second_snapshot)).hexdigest(),
                          "capture_epoch_status": "bound",
                          "snapshot": second_snapshot})
    frames.append(second)
    doc = seal_sidecar(doc)
    assert validate_sidecar(doc, expected=expected, expected_frames=frames,
                            require_certifiable=False) == doc["seal_sha256"]


def test_unbound_epoch_is_preserved_in_report_only_but_refused_in_strict(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    snapshot = dict(doc["frames"][0]["snapshot"])
    snapshot["capture_epoch_status"] = "capture_epoch_unbound"
    doc["frames"][0]["snapshot"] = snapshot
    doc["frames"][0]["snapshot_sha256"] = hashlib.sha256(canonical_json(snapshot)).hexdigest()
    doc["frames"][0]["capture_epoch_status"] = "capture_epoch_unbound"
    doc = seal_sidecar(doc)
    assert validate_sidecar(doc, expected=expected, expected_frames=frames,
                            require_certifiable=False) == doc["seal_sha256"]
    with pytest.raises(ProvenanceError, match="epoch is unbound"):
        validate_sidecar(doc, expected=expected, expected_frames=frames,
                          require_certifiable=True)


def test_resealed_fake_complete_manifest_and_capture_mod_conflict_are_rejected(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    doc["resource_packs"][0]["manifest"]["sha256"] = "f" * 64
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="manifest digest mismatch"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)

    expected, frames, doc = _complete_sidecar(tmp_path)
    doc["runtime"]["capture_mod_loaded_sha256"] = "b" * 64
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="loaded capture mod digest"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)

    expected, frames, doc = _complete_sidecar(tmp_path)
    doc["runtime"]["client_artifacts"][2]["sha256"] = "b" * 64
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="origin digest mismatch"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)


def test_strict_rejects_resealed_null_or_empty_render_pipeline(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    for pipelines in ([None], [{}], []):
        altered = json.loads(json.dumps(doc))
        altered["render_policy"]["pipelines"] = pipelines
        policy_unsigned = {key: value for key, value in altered["render_policy"].items()
                           if key != "sha256"}
        altered["render_policy"]["sha256"] = hashlib.sha256(
            canonical_json(policy_unsigned)).hexdigest()
        altered = seal_sidecar(altered)
        with pytest.raises(ProvenanceError, match="render pipeline|render-policy evidence|render-program epoch"):
            validate_sidecar(altered, expected=expected, expected_frames=frames)


def test_strict_rejects_external_shader_without_identity(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    doc["shader_status"] = "external_active"
    doc.pop("shader_identity")
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="shader/program identity"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)


@pytest.mark.parametrize("field,value", [
    ("vendor", True), ("renderer", 4), ("version", {"claimed": "yes"}),
    ("glsl_version", ["unknown"]),
])
def test_strict_rejects_truthy_non_string_gl_values(tmp_path, field, value):
    expected, frames, doc = _complete_sidecar(tmp_path)
    doc["frames"][0]["snapshot"]["gl"][field] = value
    snapshot = doc["frames"][0]["snapshot"]
    doc["frames"][0]["snapshot_sha256"] = hashlib.sha256(canonical_json(snapshot)).hexdigest()
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="resource/render epoch"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)


def test_strict_rejects_status_only_epoch_and_reversed_pack_order(tmp_path):
    expected, frames, doc = _complete_sidecar(tmp_path)
    snapshot = doc["frames"][0]["snapshot"]
    del snapshot["acquisition_epoch"]
    doc["frames"][0]["snapshot_sha256"] = hashlib.sha256(canonical_json(snapshot)).hexdigest()
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="acquisition epoch"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)

    expected, frames, doc = _complete_sidecar(tmp_path)
    snapshot = doc["frames"][0]["snapshot"]
    snapshot["ordered_resource_packs"] = ["other", "vanilla"]
    doc["frames"][0]["snapshot_sha256"] = hashlib.sha256(canonical_json(snapshot)).hexdigest()
    doc = seal_sidecar(doc)
    with pytest.raises(ProvenanceError, match="resource/render epoch"):
        validate_sidecar(doc, expected=expected, expected_frames=frames)


def test_directory_manifest_is_ordered_content_bound_and_rejects_links(tmp_path):
    root = tmp_path / "pack"
    (root / "assets" / "demo").mkdir(parents=True)
    (root / "assets" / "demo" / "z.txt").write_bytes(b"z")
    (root / "assets" / "demo" / "a.txt").write_bytes(b"a")
    first = directory_manifest(root)
    assert [row["path"] for row in first["entries"]] == [
        "assets/demo/a.txt", "assets/demo/z.txt"]
    (root / "assets" / "demo" / "a.txt").write_bytes(b"changed")
    assert directory_manifest(root)["sha256"] != first["sha256"]
    link = root / "escape"
    try:
        link.symlink_to(tmp_path / "outside")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this filesystem")
    with pytest.raises(ProvenanceError, match="symlink"):
        directory_manifest(root)


@pytest.mark.parametrize("name", ["a//b", "a/./b", "C:/pack/file", "/absolute", "a/../b"])
def test_pack_paths_reject_aliases_and_windows_drive_forms(name):
    from mc_photogrammetry.dataset.runtime_render_provenance import _relative_posix
    with pytest.raises(ProvenanceError):
        _relative_posix(name)


def test_directory_empty_entries_and_depth_are_bounded_before_sort(tmp_path, monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    root = tmp_path / "many-empty"
    root.mkdir()
    for index in range(8):
        (root / f"empty-{index}").mkdir()
    monkeypatch.setattr(provenance, "MAX_PACK_ENTRIES", 4)
    with pytest.raises(ProvenanceError, match="total path count exceeds cap"):
        directory_manifest(root)

    monkeypatch.setattr(provenance, "MAX_PACK_ENTRIES", 100)
    monkeypatch.setattr(provenance, "MAX_PACK_DEPTH", 1)
    deep = root / "deep" / "deeper"
    deep.mkdir(parents=True)
    with pytest.raises(ProvenanceError, match="depth exceeds cap"):
        directory_manifest(root)


def test_directory_scan_deadline_is_enforced(tmp_path, monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    root = tmp_path / "slow-pack"
    root.mkdir()
    (root / "item").write_bytes(b"x")
    monkeypatch.setattr(provenance, "MAX_PACK_SCAN_SECONDS", -1.0)
    with pytest.raises(ProvenanceError, match="timed out"):
        directory_manifest(root)


def test_zip_manifest_rejects_path_escape_and_duplicate_entries(tmp_path):
    escaped = tmp_path / "escaped.zip"
    with zipfile.ZipFile(escaped, "w") as archive:
        archive.writestr("../outside", b"x")
    with pytest.raises(ProvenanceError, match="escapes"):
        zip_manifest(escaped)

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("asset.txt", b"a")
        archive.writestr("asset.txt", b"b")
    with pytest.raises(ProvenanceError, match="duplicate"):
        zip_manifest(duplicate)


def test_zip_forged_count_and_directory_only_entries_fail_closed(tmp_path, monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    path = tmp_path / "count.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("file.bin", b"x")
    raw = bytearray(path.read_bytes())
    eocd = raw.rfind(b"PK\x05\x06")
    raw[eocd + 10:eocd + 12] = (0).to_bytes(2, "little")
    path.write_bytes(raw)
    with pytest.raises(ProvenanceError, match="central-record count exceeds cap or is inconsistent"):
        zip_manifest(path)

    dirs = tmp_path / "dirs.zip"
    with zipfile.ZipFile(dirs, "w") as archive:
        for index in range(5):
            archive.writestr(f"empty-{index}/", b"")
    monkeypatch.setattr(provenance, "MAX_PACK_ENTRIES", 3)
    with pytest.raises(ProvenanceError, match="central-record count exceeds cap"):
        zip_manifest(dirs)


def test_zip_uses_one_bounded_snapshot_for_hash_and_entries(tmp_path, monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    path = tmp_path / "mutable.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("asset.bin", b"original")
    original_raw = path.read_bytes()
    expected_digest = hashlib.sha256(original_raw).hexdigest()
    original_snapshot_reader = provenance._read_bounded_snapshot
    calls = 0

    def read_then_replace(target, *, max_bytes):
        nonlocal calls
        calls += 1
        snapshot = original_snapshot_reader(target, max_bytes=max_bytes)
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("asset.bin", b"replacement")
        return snapshot

    monkeypatch.setattr(provenance, "_read_bounded_snapshot", read_then_replace)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: (_ for _ in ()).throw(
        AssertionError("ZIP input was reopened by path")))
    result = zip_manifest(path)
    assert calls == 1
    assert result["archive_sha256"] == expected_digest
    assert result["entries"][0]["sha256"] == hashlib.sha256(b"original").hexdigest()


def test_zip_archive_cap_is_checked_before_snapshot_read(tmp_path, monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    path = tmp_path / "too-large.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("asset", b"content")
    monkeypatch.setattr(provenance, "MAX_PACK_ARCHIVE_BYTES", path.stat().st_size - 1)
    monkeypatch.setattr(provenance, "_zip_central_records", lambda _raw: pytest.fail(
        "central parser must not run after a cap failure"))
    with pytest.raises(ProvenanceError, match="byte cap"):
        zip_manifest(path)


def test_manifest_serialization_has_independent_manifest_cap(monkeypatch):
    import mc_photogrammetry.dataset.runtime_render_provenance as provenance
    monkeypatch.setattr(provenance, "MAX_SIDECAR_BYTES", 8)
    assert len(canonical_manifest_json({"payload": "x" * 100})) > 8
    with pytest.raises(ProvenanceError, match="sidecar exceeds byte cap"):
        canonical_json({"payload": "x" * 100})


def test_zip_manifest_binds_uncompressed_contents(tmp_path):
    path = tmp_path / "resources.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("assets/a.bin", b"abc")
    result = zip_manifest(path)
    assert result["entry_count"] == 1
    assert result["entries"][0]["sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_sidecar_size_and_field_limits_are_explicit(tmp_path):
    oversized = tmp_path / "large.bin"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_PACK_FILE_BYTES + 1)
    from mc_photogrammetry.dataset.runtime_render_provenance import sha256_file_bounded
    with pytest.raises(ProvenanceError, match="byte cap"):
        sha256_file_bounded(oversized)
