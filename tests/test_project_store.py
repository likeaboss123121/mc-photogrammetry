"""Focused tests for scan-project persistence at the schema-2 seam.

Schema 2 adds exactly one optional field: the canonical image-derived session
record produced by
``pipeline.image_derived_session.image_derived_session_to_record``. The record is
validated by ``image_derived_session_from_record`` at the store boundary, so a
tampered digest, a forbidden scale/translation claim, an unknown key, or an
unsupported schema is refused with a typed :class:`ProjectStoreError` rather than
persisted or loaded.

The session records here are produced by the real ``build_image_derived_session``
with a fake solver, mirroring ``tests/test_image_derived_session.py``; only camera
recovery is substituted, never the record integrity contract.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from mc_photogrammetry.domain.models import (
    SCAN_PROJECT_LEGACY_SCHEMA_VERSION,
    SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
    SCAN_PROJECT_SCHEMA_VERSION,
    BlockObservation,
    BlockState,
    BlockVolume,
    ObservationKind,
    ScanProject,
)
from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
    CameraSolution,
)
from mc_photogrammetry.pipeline.image_derived_session import (
    IMAGE_DERIVED_SESSION_SCHEMA,
    IMAGE_DERIVED_SESSION_VERSION,
    build_image_derived_session,
    image_derived_session_to_record,
)
from mc_photogrammetry.project.store import ProjectStoreError, load_project, save_project


def test_bounded_project_exact_bytes_and_unicode_parity(tmp_path):
    from mc_photogrammetry.project import store
    project = ScanProject(name="café \U0001f642\n", source_paths=[])
    legacy = tmp_path / "legacy.mcscan"
    save_project(project, legacy)
    payload = legacy.read_bytes()
    bounded = tmp_path / "bounded.mcscan"
    save_project(project, bounded, max_bytes=len(payload))
    assert bounded.read_bytes() == payload
    assert load_project(bounded).name == project.name
    refused = tmp_path / "absent" / "refused.mcscan"
    with pytest.raises(ProjectStoreError, match="project_over_cap"):
        save_project(project, refused, max_bytes=len(payload) - 1)
    assert not refused.parent.exists()
    data = {"a": [[], {}, True, False, None, -23, 1.25], "é": "🙂\n\t"}
    assert store._indented_json_size(data, 4096) == len(json.dumps(data, indent=2))


def test_v2_nested_session_round_trip_keeps_all_inputs_exclusions_and_outer_schema(monkeypatch, tmp_path):
    from test_image_derived_session import _partial_session
    session, _ = _partial_session(monkeypatch)
    record = image_derived_session_to_record(session)
    paths = [frame.source_label for frame in session.input_frames]
    project = ScanProject(name="partial synthetic", source_paths=paths, image_derived_session=record)
    target = tmp_path / "partial.mcscan"
    save_project(project, target)
    loaded = load_project(target)
    assert loaded.schema_version == project.schema_version == 2
    assert loaded.image_derived_session == record and loaded.source_paths == paths
    assert loaded.volume.blocks == {}
    malformed = ScanProject(name="missing excluded input", source_paths=[paths[0], paths[2]], image_derived_session=record)
    with pytest.raises(ProjectStoreError, match="full ordered input inventory"):
        save_project(malformed, tmp_path / "must-not-exist.mcscan")
    assert not (tmp_path / "must-not-exist.mcscan").exists()
    data = json.loads(target.read_text())
    data["source_paths"].reverse()
    target.write_text(json.dumps(data))
    with pytest.raises(ProjectStoreError, match="full ordered input inventory"):
        load_project(target)


def test_bounded_project_preserves_colliding_destination_and_temporary(tmp_path):
    project = ScanProject(name="bounded", source_paths=[])
    target = tmp_path / "scan.mcscan"
    temporary = tmp_path / "scan.mcscan.tmp"
    for collision in (target, temporary):
        collision.write_bytes(b"foreign")
        with pytest.raises(ProjectStoreError, match="already exists"):
            save_project(project, target, max_bytes=4096)
        assert collision.read_bytes() == b"foreign"
        collision.unlink()


def test_bounded_writer_handles_short_writes_and_zero_write():
    from mc_photogrammetry.project import store
    class Short:
        def __init__(self):
            self.data = bytearray()
        def write(self, data):
            count = min(3, len(data))
            self.data.extend(data[:count])
            return count
    stream = Short()
    store._write_bounded(stream, b"1234567890", 10)
    assert stream.data == b"1234567890"
    with pytest.raises(ProjectStoreError, match="project_over_cap"):
        store._write_bounded(stream, b"too long", 1)
    class Zero:
        def write(self, data):
            return 0
    with pytest.raises(ProjectStoreError, match="write result"):
        store._write_bounded(Zero(), b"x", 1)


_FORBIDDEN_PIXEL_KEYS = {"image", "pixels", "array", "pixel_data", "image_data"}


def _contains_pixel_payload(value) -> bool:
    if isinstance(value, dict):
        return any(
            key in _FORBIDDEN_PIXEL_KEYS or _contains_pixel_payload(item)
            for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_pixel_payload(item) for item in value)
    return isinstance(value, np.ndarray)


def _estimate():
    return CameraEstimate(
        intrinsics=CameraIntrinsics((60, 40), (100.0, 100.0), (30.0, 20.0)),
        pose=CameraPose(
            position=None, yaw_degrees=14.0, pitch_degrees=17.0, roll_degrees=3.5),
        residual_pixels=0.4,
        confidence=0.0,
    )


def _solution(frame_ids):
    cameras = {frame_id: _estimate() for frame_id in frame_ids}
    unresolved = []
    for frame_id in frame_ids:
        unresolved.append(f"{frame_id}: position (vanishing points fix no translation)")
        unresolved.append(f"{frame_id}: scale (needs the lattice period)")
        unresolved.append(f"{frame_id}: yaw quadrant (recovered modulo 90 degrees)")
    return CameraSolution(cameras=cameras, unresolved=tuple(unresolved), scale_known=False)


def _rgb(shape=(40, 60)):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[:, : shape[1] // 2] = (255, 0, 0)
    image[:, shape[1] // 2 :] = (0, 0, 255)
    return image


def _patched_solution(monkeypatch, frame_ids=("frame.png",)):
    def fake_solve(frames):
        return _solution(list(frames) if frame_ids is None else list(frame_ids))

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)


def _session_record(monkeypatch, frame_ids=("frame.png",)) -> dict:
    _patched_solution(monkeypatch, frame_ids)
    session = build_image_derived_session({frame_id: _rgb() for frame_id in frame_ids})
    return image_derived_session_to_record(session)


def _volume() -> BlockVolume:
    return BlockVolume(blocks={
        (0, 64, 0): BlockObservation(BlockState("minecraft:stone")),
        (1, 64, 0): BlockObservation(
            BlockState("minecraft:oak_log", (("axis", "y"),)),
            confidence=0.42,
            kind=ObservationKind.INFERRED,
            frame_ids=["frame-001"],
        ),
    })


def _valid_schema2_payload(monkeypatch) -> dict:
    return ScanProject(
        name="capture",
        source_paths=["captures/frame.png"],
        volume=_volume(),
        image_derived_session=_session_record(monkeypatch),
    ).to_dict()


# --- BlockVolume compatibility ----------------------------------------------

def test_project_round_trip_preserves_sparse_volume(tmp_path):
    project = ScanProject(name="Test build", source_paths=["capture.png"])
    project.volume = BlockVolume(
        blocks={
            (0, 64, 0): BlockObservation(BlockState("minecraft:stone")),
            (1, 64, 0): BlockObservation(
                BlockState("minecraft:oak_log", (("axis", "y"),)),
                confidence=0.42,
                kind=ObservationKind.INFERRED,
                frame_ids=["frame-001"],
            ),
        }
    )

    path = tmp_path / "scan.mcscan"
    save_project(project, path)
    restored = load_project(path)

    assert restored.to_dict() == project.to_dict()
    assert restored.volume.to_dict() == project.volume.to_dict()


# --- Schema-2 default and explicit payload -----------------------------------

def test_new_projects_default_to_explicit_schema_two(tmp_path):
    project = ScanProject(name="Untitled")

    assert project.schema_version == SCAN_PROJECT_SCHEMA_VERSION == 2
    assert project.to_dict()["schema_version"] == 2
    assert project.to_dict()["image_derived_session"] is None

    path = tmp_path / "scan.mcscan"
    save_project(project, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert "image_derived_session" in payload
    assert payload["image_derived_session"] is None


def test_a_schema_two_project_without_a_session_round_trips(tmp_path):
    project = ScanProject(name="no session", source_paths=["a.png"], volume=_volume())
    path = tmp_path / "scan.mcscan"

    save_project(project, path)
    first_bytes = path.read_bytes()
    restored = load_project(path)

    assert restored.schema_version == 2
    assert restored.image_derived_session is None
    assert restored.to_dict() == project.to_dict()
    save_project(restored, path)
    assert path.read_bytes() == first_bytes


# --- Accepted session round trip ---------------------------------------------

def test_a_valid_session_record_round_trips_byte_for_byte(monkeypatch, tmp_path):
    record = _session_record(monkeypatch)
    project = ScanProject(
        name="capture", source_paths=["captures/frame.png"], volume=_volume(),
        image_derived_session=record)
    path = tmp_path / "scan.mcscan"

    save_project(project, path)
    first_bytes = path.read_bytes()
    payload = json.loads(first_bytes)
    assert payload["image_derived_session"] == record

    restored = load_project(path)

    assert restored.to_dict() == project.to_dict()
    assert restored.image_derived_session == record
    assert restored.volume.to_dict() == project.volume.to_dict()

    save_project(restored, path)
    assert path.read_bytes() == first_bytes


def test_the_accepted_record_carries_no_pixel_payload(monkeypatch, tmp_path):
    project = ScanProject(image_derived_session=_session_record(monkeypatch))
    path = tmp_path / "scan.mcscan"

    save_project(project, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    session = payload["image_derived_session"]
    assert session["schema"] == IMAGE_DERIVED_SESSION_SCHEMA
    assert session["version"] == IMAGE_DERIVED_SESSION_VERSION
    assert session["scale_known"] is False
    assert session["cameras"][0]["pose"]["position"] is None
    json.dumps(payload, allow_nan=False)
    assert not _contains_pixel_payload(payload)


# --- Schema-1 legacy compatibility -------------------------------------------

def _legacy_payload() -> dict:
    return {
        "schema_version": SCAN_PROJECT_LEGACY_SCHEMA_VERSION,
        "name": "legacy",
        "minecraft_version": "1.20.1",
        "source_paths": ["old.png"],
        "volume": _volume().to_dict(),
    }


def test_a_legacy_schema_one_payload_still_loads(tmp_path):
    path = tmp_path / "legacy.mcscan"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    restored = load_project(path)

    assert restored.schema_version == SCAN_PROJECT_LEGACY_SCHEMA_VERSION
    assert restored.image_derived_session is None
    assert restored.name == "legacy"
    assert restored.minecraft_version == "1.20.1"
    assert restored.source_paths == ["old.png"]
    assert restored.volume.to_dict() == _volume().to_dict()


def test_re_saving_a_legacy_project_keeps_schema_one_without_a_session(tmp_path):
    source = tmp_path / "legacy.mcscan"
    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    destination = tmp_path / "copy.mcscan"

    save_project(load_project(source), destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["schema_version"] == SCAN_PROJECT_LEGACY_SCHEMA_VERSION
    assert "image_derived_session" not in payload
    save_project(load_project(destination), destination)
    assert destination.read_text(encoding="utf-8") \
        == json.dumps(_legacy_payload(), indent=2) + "\n"


def test_a_schema_two_payload_cannot_carry_a_production_record(monkeypatch, tmp_path):
    payload = _valid_schema2_payload(monkeypatch)
    payload["image_derived_production"] = {"schema": "forged"}
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


def test_a_schema_one_payload_cannot_carry_a_production_record(tmp_path):
    payload = _legacy_payload()
    payload["image_derived_production"] = {"schema": "forged"}
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


def test_a_schema_one_payload_cannot_smuggle_a_session(monkeypatch, tmp_path):
    payload = _legacy_payload()
    payload["image_derived_session"] = _session_record(monkeypatch)
    path = tmp_path / "legacy.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


def test_a_schema_one_project_cannot_be_saved_with_a_session(monkeypatch, tmp_path):
    project = ScanProject(
        schema_version=SCAN_PROJECT_LEGACY_SCHEMA_VERSION,
        image_derived_session=_session_record(monkeypatch))
    path = tmp_path / "legacy.mcscan"

    with pytest.raises(ProjectStoreError):
        save_project(project, path)

    assert not path.exists()


# --- Forbidden metadata and unknown keys -------------------------------------

@pytest.mark.parametrize(
    "key, value",
    [
        ("pixels", []),
        ("cameras", [{"frame_id": "frame.png"}]),
        ("camera_manifest", {"frame.png": {}}),
        ("f3", [0, 64, 0]),
        ("coordinates", [[0, 64, 0]]),
        ("directions", ["north"]),
        ("bounds", {"min": [0, 0, 0], "max": [1, 1, 1]}),
        ("scene", {}),
        ("scene_transform", {"origin": [0, 0, 0]}),
        ("scale", 1.0),
        ("translation", [0, 0, 0]),
    ],
)
def test_a_schema_two_payload_with_forbidden_metadata_is_refused(
        monkeypatch, tmp_path, key, value):
    payload = _valid_schema2_payload(monkeypatch)
    payload[key] = value
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError) as raised:
        load_project(path)

    assert "cannot carry" in str(raised.value)


def test_a_schema_two_payload_with_unknown_keys_is_refused(monkeypatch, tmp_path):
    payload = _valid_schema2_payload(monkeypatch)
    payload["mystery"] = 1
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


def test_a_non_mapping_session_field_is_refused(monkeypatch, tmp_path):
    payload = _valid_schema2_payload(monkeypatch)
    payload["image_derived_session"] = ["not", "a", "record"]
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


# --- Unsupported schema ------------------------------------------------------

def test_an_unsupported_project_schema_is_refused(tmp_path):
    payload = _legacy_payload()
    payload["schema_version"] = 4
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError, match="unsupported scan project schema"):
        load_project(path)


def test_a_schema_three_payload_without_its_records_is_refused(tmp_path):
    payload = _legacy_payload()
    payload["schema_version"] = 3
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


def test_a_schema_two_project_cannot_carry_a_production_record(tmp_path):
    project = ScanProject(image_derived_production={"schema": "forged"})

    with pytest.raises(ProjectStoreError):
        save_project(project, tmp_path / "scan.mcscan")


def test_save_project_refuses_a_schema_three_production_payload(tmp_path):
    production = {
        "schema": "mc-photogrammetry-image-derived-production",
        "version": 1,
    }
    project = ScanProject(
        schema_version=SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
        image_derived_production=production)
    target = tmp_path / "durable.mcscan"
    target.write_text("predecessor bytes", encoding="utf-8")

    with pytest.raises(ProjectStoreError, match="publish_project_no_clobber"):
        save_project(project, target)

    # The durable terminal is preserved and no replace temp is left behind.
    assert target.read_text(encoding="utf-8") == "predecessor bytes"
    assert not (tmp_path / "durable.mcscan.tmp").exists()


def test_an_unsupported_session_version_is_refused(monkeypatch, tmp_path):
    payload = _valid_schema2_payload(monkeypatch)
    payload["image_derived_session"]["version"] = 2
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


# --- Tampered or forged session records --------------------------------------

def test_a_tampered_session_digest_is_refused_on_load(monkeypatch, tmp_path):
    payload = _valid_schema2_payload(monkeypatch)
    payload["image_derived_session"]["session_sha256"] = "0" * 64
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


def test_a_tampered_session_digest_is_refused_on_save(monkeypatch, tmp_path):
    record = _session_record(monkeypatch)
    record["session_sha256"] = "0" * 64
    project = ScanProject(image_derived_session=record)
    path = tmp_path / "scan.mcscan"

    with pytest.raises(ProjectStoreError):
        save_project(project, path)

    assert not path.exists()


def test_a_session_claiming_scale_is_refused(monkeypatch, tmp_path):
    payload = _valid_schema2_payload(monkeypatch)
    payload["image_derived_session"]["scale_known"] = True
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


def test_a_session_claiming_translation_is_refused(monkeypatch, tmp_path):
    payload = _valid_schema2_payload(monkeypatch)
    payload["image_derived_session"]["cameras"][0]["pose"]["position"] = [1.0, 2.0, 3.0]
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


def test_a_session_record_with_unknown_keys_is_refused(monkeypatch, tmp_path):
    payload = _valid_schema2_payload(monkeypatch)
    payload["image_derived_session"]["bounds"] = [0, 0, 0]
    path = tmp_path / "scan.mcscan"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectStoreError):
        load_project(path)


# --- Typed error surface -----------------------------------------------------

def test_store_refusals_are_typed_project_store_errors(tmp_path):
    broken = tmp_path / "broken.mcscan"
    broken.write_text("not json", encoding="utf-8")

    with pytest.raises(ProjectStoreError) as raised:
        load_project(broken)

    assert isinstance(raised.value, ValueError)

    root = tmp_path / "root.mcscan"
    root.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ProjectStoreError, match="JSON object"):
        load_project(root)


@pytest.mark.skipif(__import__("os").name != "nt", reason="native Windows file identity required")
def test_native_project_lease_retains_birth_handle_and_blocks_replacement(tmp_path, monkeypatch):
    from mc_photogrammetry.project import store
    from test_route_b_diagnostic_runner import _native_barrier_pair, _assert_native_replacement_blocked
    source, target, proposed = tmp_path / "scan.mcscan", tmp_path / "published.mcscan", tmp_path / "foreign.mcscan"
    lease = store.save_project(ScanProject(name="leased"), source, max_bytes=4096, retain_lease=True)
    data = lease.read_bytes()
    proposed.write_bytes(data)
    identity = lease.identity
    original = store._WindowsRetainedFile.rename_no_replace
    held = None
    try:
        def operation(barrier):
            def rename(file, destination):
                if file.path == source:
                    assert file._identity() == identity and len(identity[1]) == 16
                    barrier()
                return original(file, destination)
            monkeypatch.setattr(store._WindowsRetainedFile, "rename_no_replace", rename)
            return lease.take_file(target)
        held = _native_barrier_pair(operation, lambda: _assert_native_replacement_blocked(
            source, proposed, tmp_path / "saved.mcscan"))
        assert held.identity == identity
        assert held.read_bounded(4096) == data
        assert not lease.active
        with pytest.raises(store.ProjectStoreError):
            lease.read_bytes()
    finally:
        if held is not None:
            held.close()
        lease.close()
    assert target.read_bytes() == data and proposed.read_bytes() == data


@pytest.mark.skipif(__import__("os").name != "nt", reason="native Windows file identity required")
def test_native_store_temp_cleanup_uses_birth_handle_under_replacement_race(tmp_path, monkeypatch):
    from mc_photogrammetry.project import store
    from test_route_b_diagnostic_runner import _native_barrier_pair, _assert_native_replacement_blocked
    target = tmp_path / "scan.mcscan"
    temporary = tmp_path / "scan.mcscan.tmp"
    proposed = tmp_path / "foreign.mcscan"
    proposed.write_bytes(b"foreign")
    write, delete = store._write_bounded, store._WindowsRetainedFile.delete_owned
    def operation(barrier):
        def fail(stream, data, cap):
            write(stream, data, cap)
            raise OSError("injected bounded-write failure")
        def cleanup(file):
            if file.path == temporary:
                file._verify()
                barrier()
            return delete(file)
        monkeypatch.setattr(store, "_write_bounded", fail)
        monkeypatch.setattr(store._WindowsRetainedFile, "delete_owned", cleanup)
        with pytest.raises(store.ProjectStoreError):
            store.save_project(ScanProject(name="bounded"), target, max_bytes=4096)
    _native_barrier_pair(operation, lambda: _assert_native_replacement_blocked(
        temporary, proposed, tmp_path / "saved.mcscan"))
    assert proposed.read_bytes() == b"foreign"
    assert not temporary.exists() and not target.exists()


@pytest.mark.skipif(__import__("os").name != "nt", reason="native Windows lease required")
def test_native_project_lease_kernel_handle_has_no_write_rights(tmp_path):
    import ctypes
    from ctypes import wintypes as w
    from mc_photogrammetry.project import store
    lease = store.save_project(ScanProject(name="read-only"), tmp_path / "scan.mcscan",
                               max_bytes=4096, retain_lease=True)
    try:
        before = lease.read_bytes()
        buffer, written = ctypes.create_string_buffer(b"foreign"), w.DWORD()
        assert not lease._held._api.WriteFile(lease._held.handle, buffer, 7,
                                              ctypes.byref(written), None)
        assert ctypes.get_last_error() == 5 and written.value == 0
        assert lease.read_bytes() == before
        assert not lease._held.writable
    finally:
        lease.close()


@pytest.mark.skipif(__import__("os").name != "nt", reason="native Windows directory identity required")
def test_native_root_anchor_blocks_empty_root_rename_but_namespace_and_write_remain_hold(tmp_path):
    from mc_photogrammetry.project import store
    import ctypes
    root = tmp_path / "empty-root"
    root.mkdir()
    with store._WindowsRootAnchor(root) as anchor:
        # The root is EMPTY: open child files cannot supply accidental protection.
        with pytest.raises(OSError) as error:
            root.rename(tmp_path / "renamed-root")
        assert error.value.winerror in (5, 32)
        file = anchor.files[-1]
        handle = file._api.CreateFileW(str(root), 0x40000000, 7, None, 3,
                                      0x02000000 | 0x00200000, None)
        assert handle != ctypes.c_void_p(-1).value
        assert file._api.CloseHandle(handle)  # WRITE is admitted: explicit hard HOLD
        # No reparse point is created; WRITE exclusion is not falsely claimed.
        child = root / "foreign.bin"
        child.write_bytes(b"foreign")
        anchor.verify()
        assert anchor.namespace_exclusive is False
    assert child.read_bytes() == b"foreign"
