"""Frozen synthetic driver controls; never read the two ordinary screenshots."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from mc_photogrammetry.input.image import load_image_bytes
from mc_photogrammetry.pipeline.correspondence import FrameFeatures, usable_mask_for
from mc_photogrammetry.pipeline.image_derived_session import image_sha256
from mc_photogrammetry.reconstruction import lattice_planes as regions
from scripts.evaluation import acquire_endpoint_conflict_native_roster as subject
from scripts.evaluation import diagnose_image_geometry_failure as diagnostic
from scripts.evaluation import image_point_candidate_roster as roster


POINTS = [(8.125, 12.375), (24.125, 12.375), (8.125, 28.375), (24.125, 28.375)]
SERVICES = (np, load_image_bytes, usable_mask_for, roster, diagnostic)


def synthetic_source():
    """Fresh 64x64 PNGs and genuine source-region producer, not ordinary data."""
    buffers, ingress = [], []
    for index, frame_id in enumerate(("image-0000", "image-0001")):
        pixels = np.zeros((64, 64, 3), np.uint8)
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        data = stream.getvalue()
        buffers.append(data)
        ingress.append({"frame_id": frame_id, "source_label": "synthetic-native-driver-control",
                        "byte_sha256": subject.sha(data), "byte_length": len(data),
                        "decoded_shape": [64, 64, 3], "decoded_dtype": "|u1",
                        "decoded_pixel_sha256": image_sha256(pixels),
                        "decoder": "input.image.load_image:exif_transpose+convert('RGB')",
                        "schema": "mc-photogrammetry-frame-ingress-provenance-v1",
                        "transform": {"source_size": [64, 64], "crop_origin": [0, 0],
                                      "crop_size": None, "output_size": None}})
    readings, records, bindings, originals = [], [], [], []
    polygon = ((4., 8.), (30., 8.), (30., 32.), (4., 32.))
    for index, endpoint in enumerate(diagnostic.MASK_ENDPOINTS):
        mask = np.zeros((64, 64), bool)
        mask[8:32, 4:30] = True
        mask[18:21, 15:18] = False  # Hole; no rectangular reconstruction.
        mask[30, 29] = False
        region = regions._make_region_evidence(mask, 1, polygon, (f"synthetic-driver-{index}",))
        originals.append(mask)
        readings.append(SimpleNamespace(region_evidence=region, polygon=polygon, axis=1, frame=endpoint[0]))
        binding_sha = str(index + 1) * 64
        bindings.append((endpoint, index, 1, binding_sha, (), ()))
        summary = region.to_record()
        summary.pop("packed_mask_base64")
        summary["type"] = "available"
        records.append({"endpoint": list(endpoint), "reading_index": index,
                        "reading_binding_sha256": binding_sha,
                        "polygon": [list(p) for p in polygon], "source_region": summary})
    stage = {k: "b" * 64 for k in ("entry", "session", "lattice", "signatures", "connections", "graph", "attribution")}
    rows = [{"source_row": n, "row_sha256": str(i + 6) * 64,
             "original_direction": [list(diagnostic.MASK_ENDPOINTS[i]), list(diagnostic.MASK_ENDPOINTS[2])]}
            for i, n in enumerate(diagnostic.MASK_ROWS)]
    joint = {"schema": diagnostic.JOINT_ENDPOINT_SCHEMA, "producer": diagnostic.JOINT_ENDPOINT_PRODUCER,
             "first_witness_provenance": {"type": "available", "partial_exclusion_index": 0,
                "proof_sha256": "c" * 64, "source_universe_sha256": "d" * 64,
                "rows": rows, "readings": records},
             "proof_sha256": "c" * 64, "candidate_matches_sha256": "d" * 64,
             "selection_sha256": "e" * 64, "reading_bindings_sha256": "f" * 64,
             "source_bindings": {k + "_sha256": stage[k] for k in ("session", "lattice", "signatures", "graph")}}
    joint["record_sha256"] = diagnostic._digest(joint)
    record = {"schema": diagnostic.MASK_DIAGNOSTIC_SCHEMA, "ingress": ingress,
              "stage_digests": stage, "joint_endpoint_selection": joint,
              "status": "incomplete", "issues": ["synthetic-unrelated:missing"]}
    capture = SimpleNamespace(graph=SimpleNamespace(readings=tuple(readings),
                              selection=SimpleNamespace(reading_bindings=tuple(bindings))))
    record["source_region_masks"] = diagnostic._source_region_mask_bundle(capture, record)
    record["record_sha256"] = diagnostic._digest(record)
    return tuple(buffers), record, originals


def injected_packet(monkeypatch):
    buffers, record, originals = synthetic_source()
    frames = subject.bind_frames(buffers, record["ingress"], services=SERVICES)
    groups = subject.bind_groups(record, frames, services=SERVICES)
    calls = []
    def features(frame_id, pixels, *, usable_mask, settings):
        calls.append(frame_id)
        points = POINTS + ([(40.125, 12.375)] if frame_id == "image-0001" else [])
        count = len(points)
        return FrameFeatures(frame_id, np.array(points, np.float64),
                             (np.arange(count, dtype=np.float32) * 10).reshape(count, 1),
                             64, 64, int(usable_mask.sum()), np.full(count, 2.125, np.float64), 0)
    monkeypatch.setattr(roster, "detect_features", features)
    packet = roster.acquire_roster(frames, groups)
    return record, packet, frames, groups, calls


def test_import_and_missing_review_do_not_touch_images(tmp_path, monkeypatch):
    seen = []
    real = subject.read_bounded
    def read(path, limit):
        seen.append(str(path))
        return real(path, limit)
    monkeypatch.setattr(subject, "read_bounded", read)
    with pytest.raises(FileNotFoundError):
        subject.authorize(tmp_path)
    assert seen == [str(tmp_path / subject.REVIEW)]
    assert not (tmp_path / subject.OUTPUT).exists()


def test_source_gate_checks_driver_test_and_input_authority_before_pixels(tmp_path, monkeypatch):
    _, record, _ = synthetic_source()
    diagnostic_bytes = subject.canonical(record) + b"\n"
    monkeypatch.setattr(subject, "SOURCE_PINS", {})
    monkeypatch.setattr(subject, "DIAGNOSTIC_SHA", subject.sha(diagnostic_bytes))
    monkeypatch.setattr(subject, "verify_mask_pins", lambda value: assert_equal(value, record))
    code_pins = {}
    for name in (subject.DRIVER, subject.TEST):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-frozen-code")
        code_pins[name] = subject.sha(path.read_bytes())
    path = tmp_path / subject.DIAGNOSTIC
    path.parent.mkdir(parents=True)
    path.write_bytes(diagnostic_bytes)
    review = {"disposition": "ACCEPT_DRIVER; AUTHORIZE_ONE_ACQUISITION", "output": subject.OUTPUT,
              "single_acquisition": True, "code_pins": code_pins,
              "input_pins": dict(subject.INPUTS), "diagnostic_sha256": subject.DIAGNOSTIC_SHA}
    review_path = tmp_path / subject.REVIEW
    review_path.parent.mkdir(parents=True)
    review_path.write_bytes(subject.canonical(review))
    assert subject.authorize(tmp_path)[0] == record
    (tmp_path / subject.DRIVER).write_bytes(b"changed")
    with pytest.raises(subject.Stop, match="source_pin_mismatch"):
        subject.authorize(tmp_path)
    # No ordinary input paths exist in this synthetic root; none were accessed.


def assert_equal(left, right):
    assert left == right


def test_exact_masks_digest_semantics_and_shared_right():
    buffers, record, originals = synthetic_source()
    frames = subject.bind_frames(buffers, record["ingress"], services=SERVICES)
    groups = subject.bind_groups(record, frames, services=SERVICES)
    assert len(frames) == len(groups) == 2
    assert groups[0].right is groups[1].right
    for mask, original in zip((groups[0].left, groups[1].left, groups[0].right), originals):
        assert np.array_equal(mask.pixels, original)
        assert mask.pixels.flags.owndata and mask.pixels.flags.c_contiguous
        assert not mask.pixels.flags.writeable
        provenance = subject.strict_json(mask.derivation)
        assert mask.mask_sha256 == provenance["full_native_bool_c_sha256"]
        assert mask.mask_sha256 != provenance["header_packed_sha256"]
    for frame in frames:
        assert np.array_equal(frame.usable_mask, usable_mask_for(frame.frame_id, frame.pixels.shape, None))
        assert not np.all(frame.usable_mask) and not frame.usable_mask.flags.writeable
        assert not np.array_equal(frame.usable_mask, groups[0].left.pixels)


@pytest.mark.parametrize("damage", ["bytes", "length", "identity", "shape", "frame"])
def test_ingress_mismatch_refuses_before_any_decode(damage):
    buffers, record, _ = synthetic_source()
    ingress = deepcopy(record["ingress"])
    pin = ingress[1]
    if damage == "bytes": pin["byte_sha256"] = "0" * 64
    elif damage == "length": pin["byte_length"] += 1
    elif damage == "identity": pin["transform"]["crop_origin"] = [1, 0]
    elif damage == "shape": pin["decoded_shape"] = [4000001, 1, 3]
    else: pin["frame_id"] = "foreign"
    def forbidden(*args, **kwargs):
        pytest.fail("decode preceded complete ingress preflight")
    with pytest.raises((subject.Stop, roster.Refusal)):
        subject.bind_frames(buffers, ingress, services=(np, forbidden, usable_mask_for, roster, diagnostic))


def test_decoded_digest_substitution_refuses_with_one_decode():
    buffers, record, _ = synthetic_source()
    calls = []
    def substituted(data, **kwargs):
        calls.append(data)
        pixels = np.zeros((64, 64, 3), np.uint8)
        pixels[0, 0, 0] = 1  # One byte off the all-zero frozen source: mismatch is deterministic.
        return pixels
    with pytest.raises(subject.Stop, match="decoded_ingress_mismatch"):
        subject.bind_frames(buffers, record["ingress"], services=(np, substituted, usable_mask_for, roster, diagnostic))
    assert len(calls) == 1


def test_actual_production_ingress_accepts_framed_digest_and_preserves_raw_roster_identity():
    from mc_photogrammetry.pipeline.contracts import FrameTransform
    from mc_photogrammetry.pipeline.frame_ingress import build_frame_ingress_provenance
    buffers, record, _ = synthetic_source()
    ingress = []
    for data, expected in zip(buffers, record["ingress"]):
        pixels = load_image_bytes(data, suffix=".png")
        pin = build_frame_ingress_provenance(
            frame_id=expected["frame_id"], source_label=expected["source_label"],
            encoded_bytes=data, decoded_pixels=pixels,
            transform=FrameTransform(source_size=(64, 64)),
            decoded_pixel_sha256=image_sha256(pixels))
        assert pin.to_record() == expected
        ingress.append(pin.to_record())
    frames = subject.bind_frames(buffers, ingress, services=SERVICES)
    for frame, expected in zip(frames, ingress):
        assert image_sha256(frame.pixels) == expected["decoded_pixel_sha256"]
        assert frame.source["decoded_sha256"] == roster.array_digest(frame.pixels)
        assert frame.source["decoded_sha256"] != expected["decoded_pixel_sha256"]
        roster._frame_source(frame)


@pytest.mark.parametrize("damage", ["dtype", "shape"])
def test_actual_decoded_dtype_and_shape_mismatch_refuse_before_further_decode(damage):
    buffers, record, _ = synthetic_source()
    calls = []
    def substituted(data, **kwargs):
        calls.append(data)
        return (np.zeros((64, 64, 3), np.float32) if damage == "dtype"
                else np.zeros((63, 64, 3), np.uint8))
    with pytest.raises(subject.Stop, match="decoded_ingress_mismatch"):
        subject.bind_frames(buffers, record["ingress"], services=(np, substituted, usable_mask_for, roster, diagnostic))
    assert len(calls) == 1


def test_raw_pixel_hash_cannot_substitute_for_production_ingress_digest():
    buffers, record, _ = synthetic_source()
    record["ingress"][0]["decoded_pixel_sha256"] = subject.sha(np.zeros((64, 64, 3), np.uint8).tobytes())
    with pytest.raises(subject.Stop, match="decoded_ingress_mismatch"):
        subject.bind_frames(buffers, record["ingress"], services=SERVICES)


@pytest.mark.parametrize("damage", ["missing", "extra", "order", "region", "packed", "raw", "row", "payload"])
def test_foreign_or_corrupt_masks_refuse(damage):
    buffers, record, _ = synthetic_source()
    entries = record["source_region_masks"]["entries"]
    pins = tuple((tuple(e["endpoint"]), e["mask"]["source_region_id"],
                  e["mask"]["mask_sha256"], e["full_native_bool_c_sha256"]) for e in entries)
    changed = deepcopy(record)
    entries = changed["source_region_masks"]["entries"]
    if damage == "missing": entries.pop()
    elif damage == "extra": entries.append(deepcopy(entries[0]))
    elif damage == "order": entries.reverse()
    elif damage == "region": entries[0]["mask"]["source_region_id"] = "0" * 64
    elif damage == "packed": entries[0]["mask"]["mask_sha256"] = "0" * 64
    elif damage == "raw": entries[0]["full_native_bool_c_sha256"] = "0" * 64
    elif damage == "row": entries[0]["row_refs"][0]["source_row"] += 1
    else: entries[0]["mask"]["packed_mask_base64"] = "!"
    if damage in ("missing", "extra", "order", "region", "packed", "raw"):
        with pytest.raises(subject.Stop): subject.verify_mask_pins(changed, pins)
    else:
        frames = subject.bind_frames(buffers, record["ingress"], services=SERVICES)
        with pytest.raises(diagnostic.DiagnosticError):
            subject.bind_groups(changed, frames, services=SERVICES)


def test_complete_inventory_one_detection_per_frame_full_right_and_no_selection(monkeypatch):
    record, packet, frames, groups, calls = injected_packet(monkeypatch)
    assert calls == ["image-0000", "image-0001"]
    subject.validate_packet(packet)
    assert packet["counters"]["comparisons_charged"] == 40
    assert packet["counters"]["alternatives_exposed"] == 40
    for group in packet["groups"]:
        assert len(group["queries"]) == 4
        assert len(group["alternatives"]) == 20
        assert all(q["candidate_domain_classes"] == 5 and q["all_cutoff_ties"] for q in group["queries"])
        assert any(not p["memberships"][group["group_id"]]["right"] for p in group["alternatives"])
    sealed = subject.seal_acquisition(record, {"synthetic": True}, packet)
    subject.validate_acquisition(sealed)
    assert sealed["status"] == "ready"
    assert all(c["four_distinct_general_position_pairs_exist"] for c in sealed["structural_support"].values())
    assert "selected_pairs" not in subject.canonical(sealed).decode()


def test_genuine_native_sift_on_frozen_synthetic_pngs_is_inconclusive():
    buffers, record, _ = synthetic_source()
    frames = subject.bind_frames(buffers, record["ingress"], services=SERVICES)
    groups = subject.bind_groups(record, frames, services=SERVICES)
    packet = roster.acquire_roster(frames, groups)
    subject.validate_packet(packet)
    assert all(f["retained_count"] == 0 for f in packet["frames"])
    sealed = subject.seal_acquisition(record, {"synthetic": True}, packet)
    assert sealed["status"] == "inconclusive"
    assert sealed["roster"]["groups"] == packet["groups"]  # Inventory remains complete.


@pytest.mark.parametrize("damage", ["three_queries", "same_target", "coincident", "collinear_left", "collinear_right", "outside"])
def test_structural_shortage_is_inconclusive_without_deleting_alternatives(monkeypatch, damage):
    _, packet, _, _, _ = injected_packet(monkeypatch)
    group = deepcopy(packet["groups"][0])
    original_count = len(group["alternatives"])
    for pair in group["alternatives"]:
        if damage == "three_queries": pair["left_id"] = "q" + str(POINTS.index(tuple(pair["left_xy"])) % 3)
        elif damage == "same_target": pair["right_id"] = "one"
        elif damage == "coincident": pair["right_xy"] = [1., 1.]
        elif damage == "collinear_left": pair["left_xy"][1] = pair["left_xy"][0]
        elif damage == "collinear_right": pair["right_xy"][1] = pair["right_xy"][0]
        else: pair["memberships"][group["group_id"]]["right"] = False
    assert not subject.support_condition(group)["four_distinct_general_position_pairs_exist"]
    assert len(group["alternatives"]) == original_count


@pytest.mark.parametrize("damage", ["refused", "groups", "ties", "comparisons", "count", "query_domain"])
def test_refusal_or_incomplete_inventory_cannot_seal(monkeypatch, damage):
    record, packet, _, _, _ = injected_packet(monkeypatch)
    if damage == "refused": packet = {"status": "refused", "reason": "group_pair_limit"}
    elif damage == "groups": packet["groups"].pop()
    elif damage == "ties": packet["groups"][0]["queries"][0]["all_cutoff_ties"] = False
    elif damage == "comparisons": packet["counters"]["comparisons_charged"] -= 1
    elif damage == "count": packet["counters"]["alternatives_exposed"] -= 1
    else: packet["groups"][0]["queries"][0]["candidate_domain_classes"] -= 1
    with pytest.raises(subject.Stop): subject.seal_acquisition(record, {}, packet)


def test_canonical_publish_roundtrip_no_clobber_and_seal_corruption(tmp_path, monkeypatch):
    record, packet, _, _, _ = injected_packet(monkeypatch)
    sealed = subject.seal_acquisition(record, {"synthetic": True}, packet)
    target = tmp_path / "roster.json"
    receipt = subject.publish(target, sealed, subject.validate_acquisition)
    assert receipt["sha256"] == subject.sha(target.read_bytes())
    assert subject.strict_json(target.read_bytes()) == sealed
    assert list(tmp_path.iterdir()) == [target]
    with pytest.raises(FileExistsError): subject.publish(target, sealed, subject.validate_acquisition)
    assert subject.strict_json(target.read_bytes()) == sealed
    corrupted = deepcopy(sealed)
    corrupted["record_sha256"] = "0" * 64
    with pytest.raises(subject.Stop):
        subject.publish(tmp_path / "bad.json", corrupted, subject.validate_acquisition)
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("failure", ["fsync", "link", "staged_readback", "final_readback"])
def test_publication_failure_has_no_successful_completion(tmp_path, monkeypatch, failure):
    record = {"synthetic": True}
    target = tmp_path / "completion.json"
    original_readback = subject._readback
    def broken(*args, **kwargs):
        raise OSError("frozen synthetic failure")
    if failure == "fsync": monkeypatch.setattr(subject.os, "fsync", broken)
    elif failure == "link": monkeypatch.setattr(subject.os, "link", broken)
    else:
        def readback(path, *args):
            if (failure == "staged_readback" and path.name.startswith(".")) or (failure == "final_readback" and path == target):
                broken()
            return original_readback(path, *args)
        monkeypatch.setattr(subject, "_readback", readback)
    with pytest.raises(OSError):
        subject.publish(target, record, lambda _: None, remove_failed_completion=True)
    assert not target.exists() and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,limit", [("wall", subject.WALL_SECONDS), ("rss", subject.RSS_BYTES), ("bytes", subject.ARTIFACT_BYTES)])
def test_exact_and_one_over_supervisor_bounds(field, limit):
    values = {"wall": 0, "rss": 0, "bytes": 0}
    values[field] = limit
    subject.budget_check(values["wall"], values["rss"], values["bytes"])
    values[field] += 1
    with pytest.raises(subject.Stop): subject.budget_check(values["wall"], values["rss"], values["bytes"])


def test_artifact_cap_precedes_any_stage_creation(tmp_path):
    with pytest.raises(subject.Stop, match="artifact_byte_limit"):
        subject.publish(tmp_path / "too-large.json", {"value": "x" * (subject.ARTIFACT_BYTES // 2)}, lambda _: None)
    assert list(tmp_path.iterdir()) == []


def test_short_write_and_existing_completion_are_terminal(tmp_path, monkeypatch):
    original_open = Path.open
    class ShortWriter:
        def __init__(self, handle): self.handle = handle
        def __enter__(self): self.handle.__enter__(); return self
        def __exit__(self, *args): return self.handle.__exit__(*args)
        def write(self, data): self.handle.write(data[:-1]); return len(data) - 1
    def altered_open(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        return ShortWriter(handle) if mode == "xb" and path.name.endswith(".partial") else handle
    monkeypatch.setattr(Path, "open", altered_open)
    with pytest.raises(subject.Stop, match="short_write"):
        subject.publish(tmp_path / "completion.json", {"x": 1}, lambda _: None, remove_failed_completion=True)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(Path, "open", original_open)
    target = tmp_path / "completion.json"
    target.write_bytes(b"foreign-existing-receipt")
    with pytest.raises(FileExistsError):
        subject.publish(target, {"x": 1}, lambda _: None, remove_failed_completion=True)
    assert target.read_bytes() == b"foreign-existing-receipt"


@pytest.mark.parametrize("failure", [None, "worker", "rss"])
def test_supervisor_completion_requires_worker_success_and_budget(tmp_path, monkeypatch, failure):
    record, packet, _, _, _ = injected_packet(monkeypatch)
    bindings = {"synthetic": True}
    sealed = subject.seal_acquisition(record, bindings, packet)
    buffers, _, _ = synthetic_source()
    inputs = []
    for i, data in enumerate(buffers):
        name = f"synthetic-{i}.png"
        (tmp_path / name).write_bytes(data)
        inputs.append((name, subject.sha(data)))
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "INPUTS", tuple(inputs))
    monkeypatch.setattr(subject, "authorize", lambda: (record, bindings))
    events = []
    class Process:
        returncode = 1 if failure == "worker" else 0
        def poll(self): return self.returncode
        def wait(self, timeout): return self.returncode
    def spawn(*args, **kwargs):
        events.append("spawn")
        subject.publish(tmp_path / subject.OUTPUT / "roster.json", sealed, subject.validate_acquisition)
        return Process()
    class Job:
        def attach_and_resume(self, process): events.append("attach_resume")
        def snapshot(self): return (subject.RSS_BYTES + 1 if failure == "rss" else 100, 200)
        def close(self): events.append("close")
    monkeypatch.setattr(subject, "_WindowsJob", Job)
    monkeypatch.setattr(subject.subprocess, "Popen", spawn)
    if failure:
        with pytest.raises(subject.Stop): subject.supervise()
        assert not (tmp_path / subject.OUTPUT / "completion.json").exists()
    else:
        assert subject.supervise() == 0
        completion = subject.strict_json((tmp_path / subject.OUTPUT / "completion.json").read_bytes())
        assert completion["status"] == "ready"
        assert completion["roster_sha256"] == subject.sha((tmp_path / subject.OUTPUT / "roster.json").read_bytes())
        with pytest.raises(subject.Stop, match="output_already_exists"): subject.supervise()
    assert events == ["spawn", "attach_resume", "close"]


def test_duplicate_nonfinite_json_and_narrow_cli_refuse(monkeypatch):
    for data in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(subject.Stop): subject.strict_json(data)
    monkeypatch.setattr(subject.sys, "argv", [subject.DRIVER, "--image", "forbidden"])
    with pytest.raises(subject.Stop, match="no_acquisition_overrides"):
        subject.main()


@pytest.mark.skipif(subject.os.name != "nt", reason="Windows kernel job test")
def test_real_windows_supervisor_suspends_attaches_and_tracks_synthetic_child():
    job = subject._WindowsJob()
    process = None
    try:
        process = subject.subprocess.Popen([subject.sys.executable, "-B", "-c", "import time; time.sleep(0.15)"],
                                           creationflags=0x4, stdout=subject.subprocess.DEVNULL,
                                           stderr=subject.subprocess.DEVNULL)
        assert process.poll() is None
        job.attach_and_resume(process)
        rss, committed = job.snapshot()
        assert rss > 0 and committed < subject.RSS_BYTES
        assert process.wait(timeout=5) == 0
    finally:
        job.close()
        if process is not None and process.poll() is None:
            process.wait(timeout=5)


def test_worker_single_synthetic_acquisition_stages_without_completion(tmp_path, monkeypatch):
    buffers, record, _ = synthetic_source()
    inputs = []
    for index, data in enumerate(buffers):
        name = f"synthetic-{index}.png"
        (tmp_path / name).write_bytes(data)
        inputs.append((name, subject.sha(data)))
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "INPUTS", tuple(inputs))
    monkeypatch.setattr(subject, "authorize", lambda: (record, {"synthetic": True}))
    monkeypatch.setattr(subject, "_services", lambda: SERVICES)
    (tmp_path / subject.OUTPUT).mkdir(parents=True)
    calls = []
    original_acquire = roster.acquire_roster
    def counted(*args, **kwargs):
        calls.append(1)
        return original_acquire(*args, **kwargs)
    monkeypatch.setattr(roster, "acquire_roster", counted)
    subject._worker()
    assert calls == [1]
    directory = tmp_path / subject.OUTPUT
    assert [p.name for p in directory.iterdir()] == ["roster.json"]
    sealed = subject.strict_json((directory / "roster.json").read_bytes())
    subject.validate_acquisition(sealed)
    assert sealed["status"] == "inconclusive"
