"""Synthetic exact-mask persistence controls. No screenshot or inference calls."""
import base64
import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import diagnose_image_geometry_failure as subject
from scripts.evaluation import image_point_candidate_roster as roster
from mc_photogrammetry.reconstruction import lattice_planes as regions


def fixture_source():
    masks, readings, records, bindings = [], [], [], []
    polygon = ((1., 1.), (8., 1.), (8., 6.), (1., 6.))
    for index, endpoint in enumerate(subject.MASK_ENDPOINTS):
        mask = np.zeros((9, 11), bool)
        mask[1:6, 1:8] = True
        mask[2:4, 3:5] = False
        mask[3, 1:3] = False
        mask[3, 6] = False
        mask[4, 6] = False
        mask[3, 7] = False
        mask[2, 6] = False
        # Exact hole plus a disconnected corner; 35 bits, not byte-aligned.
        mask[1:6, 6:8] = False
        mask[5, 7] = True
        if index == 1:
            mask[4, 2] = False
        region = regions._make_region_evidence(mask, 1, polygon, (f"synthetic-native-{index}",))
        canonical = regions._canonical_region_polygon(polygon)
        readings.append(SimpleNamespace(region_evidence=region, polygon=canonical, axis=1, frame=endpoint[0]))
        binding_sha = str(index + 1) * 64
        bindings.append((endpoint, index, 1, binding_sha, (), ()))
        summary = region.to_record(); summary.pop("packed_mask_base64"); summary["type"] = "available"
        records.append({"endpoint": list(endpoint), "reading_index": index,
                        "reading_binding_sha256": binding_sha, "polygon": [list(p) for p in polygon],
                        "source_region": summary})
        masks.append(mask)
    ingress = []
    for index, name in enumerate(("image-0000", "image-0001")):
        ingress.append({"frame_id": name, "source_label": "synthetic-only",
                        "byte_sha256": "a" * 64, "byte_length": 12,
                        "decoded_shape": [9, 11, 3], "decoded_dtype": "|u1",
                        "decoded_pixel_sha256": str(index + 4) * 64,
                        "decoder": "input.image.load_image:exif_transpose+convert('RGB')",
                        "schema": "mc-photogrammetry-frame-ingress-provenance-v1",
                        "transform": {"source_size": [11, 9], "crop_origin": [0, 0],
                                      "crop_size": None, "output_size": None}})
    stage = {name: "b" * 64 for name in ("entry", "session", "lattice", "signatures", "connections", "graph", "attribution")}
    rows = [{"source_row": n, "row_sha256": str(i + 6) * 64,
             "original_direction": [list(subject.MASK_ENDPOINTS[i]), list(subject.MASK_ENDPOINTS[2])]}
            for i, n in enumerate(subject.MASK_ROWS)]
    joint = {"schema": subject.JOINT_ENDPOINT_SCHEMA, "producer": subject.JOINT_ENDPOINT_PRODUCER,
             "first_witness_provenance": {"type": "available", "partial_exclusion_index": 0,
                "proof_sha256": "c" * 64, "source_universe_sha256": "d" * 64,
                "rows": rows, "readings": records},
             "proof_sha256": "c" * 64, "candidate_matches_sha256": "d" * 64,
             "selection_sha256": "e" * 64, "reading_bindings_sha256": "f" * 64,
             "record_sha256": "1" * 64,
             "source_bindings": {k + "_sha256": stage[k] for k in ("session", "lattice", "signatures", "graph")}}
    joint.pop("record_sha256")
    joint["record_sha256"] = subject._digest(joint)
    record = {"schema": subject.MASK_DIAGNOSTIC_SCHEMA, "ingress": ingress,
              "stage_digests": stage, "joint_endpoint_selection": joint,
              "status": "incomplete", "issues": ["unrelated:missing"]}
    capture = SimpleNamespace(graph=SimpleNamespace(readings=tuple(readings),
                              selection=SimpleNamespace(reading_bindings=tuple(bindings))))
    return capture, record, masks


def bundle_fixture():
    capture, record, masks = fixture_source()
    bundle = subject._source_region_mask_bundle(capture, record)
    record["source_region_masks"] = bundle
    record["record_sha256"] = subject._digest(record)
    return capture, record, bundle, masks


def reseal(bundle):
    body = dict(bundle); body.pop("record_sha256")
    bundle["record_sha256"] = subject._digest(body, limit=subject.MAX_MASK_BUNDLE_BYTES)


def test_exact_holes_disconnected_non_byte_crop_and_native_consumer():
    _, record, bundle, originals = bundle_fixture()
    assert bundle["status"] == "complete" and len(bundle["entries"]) == 3
    assert len(bundle["entries"][2]["row_refs"]) == 2
    assert len(subject._bounded_bytes(bundle)) < 1048576
    for entry, original in zip(bundle["entries"], originals):
        native = subject.load_native_source_region_mask(bundle, record, entry["endpoint"])
        assert np.array_equal(native, original)
        assert native.dtype == np.bool_ and native.flags.owndata and native.flags.c_contiguous
        assert not native.flags.writeable and not np.shares_memory(native, original)
        assert entry["mask"]["crop_shape"] == [5, 7]
        assert entry["full_native_bool_c_sha256"] == hashlib.sha256(native.tobytes()).hexdigest()
        assert entry["full_native_bool_c_sha256"] != entry["mask"]["mask_sha256"]
        consumer = roster.RegionMask(entry["endpoint"][0], record["ingress"][0 if entry["endpoint"][0] == "image-0000" else 1]["decoded_pixel_sha256"],
                                     native, entry["full_native_bool_c_sha256"], "bound exact native source")
        # Use the actual accepted consumer validation, with no feature/image work.
        source_pixels = np.zeros((9, 11, 3), np.uint8); source_pixels.setflags(write=False)
        usable = np.ones((9, 11), bool); usable.setflags(write=False)
        frame = roster.NativeFrame(consumer.frame_id, b"synthetic", source_pixels, usable,
                                   {"decoded_sha256": consumer.source_pixel_sha256})
        assert roster._region_record(consumer, frame)["mask_sha256"] == entry["full_native_bool_c_sha256"]
    restored = json.loads(subject._bounded_bytes(record))
    subject._validate_mask_diagnostic(restored)
    assert restored["issues"] == ["unrelated:missing"]


@pytest.mark.parametrize("damage", ["encoding", "padding", "bitflip", "base64", "count", "polygon", "source_id", "full_digest",
                                    "producer", "ingress", "identity", "reading_join", "row_join", "unknown_mask_field",
                                    "unknown_bundle_field", "unknown_entry_field", "unknown_ingress_field", "bbox", "axis"])
def test_corrupt_payload_or_join_atomic_refusal(damage):
    _, record, bundle, _ = bundle_fixture()
    changed = copy.deepcopy(bundle); parent = copy.deepcopy(record); entry = changed["entries"][0]; mask = entry["mask"]
    if damage == "encoding": mask["encoding"] = "bbox"
    elif damage in ("padding", "bitflip"):
        raw = bytearray(base64.b64decode(mask["packed_mask_base64"]))
        if damage == "padding": raw[-1] |= 128
        else: raw[0] ^= 2
        mask["packed_mask_base64"] = base64.b64encode(raw).decode()
    elif damage == "base64": mask["packed_mask_base64"] = "!" * len(mask["packed_mask_base64"])
    elif damage == "count": mask["pixel_count"] += 1
    elif damage == "polygon": entry["canonical_polygon"][0][0] += 0.25
    elif damage == "source_id": mask["source_region_id"] = "0" * 64
    elif damage == "full_digest": entry["full_native_bool_c_sha256"] = "0" * 64
    elif damage == "producer": changed["producer"] = "foreign"
    elif damage == "ingress": parent["ingress"][0]["decoded_pixel_sha256"] = "0" * 64
    elif damage == "identity": parent["ingress"][0]["transform"]["crop_origin"] = [1, 0]
    elif damage == "reading_join": entry["reading_index"] += 1
    elif damage == "row_join": entry["row_refs"][0]["source_row"] += 1
    elif damage == "unknown_mask_field": mask["extra"] = 1
    elif damage == "unknown_bundle_field": changed["extra"] = 1
    elif damage == "unknown_entry_field": entry["extra"] = 1
    elif damage == "unknown_ingress_field": parent["ingress"][0]["extra"] = 1
    elif damage == "axis": mask["axis"] = 0
    else: mask["bbox_xyxy"][0] += 1
    reseal(changed)
    with pytest.raises(subject.DiagnosticError): subject._validate_source_region_masks(changed, parent)


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "foreign", "foreign_region", "order"])
def test_mask_set_is_exact_and_shared_right_stored_once(damage, monkeypatch):
    _, record, bundle, _ = bundle_fixture(); changed = copy.deepcopy(bundle)
    if damage == "missing": changed["entries"].pop()
    elif damage == "extra": changed["entries"].append(copy.deepcopy(changed["entries"][0]))
    elif damage == "duplicate": changed["entries"][1] = copy.deepcopy(changed["entries"][0])
    elif damage == "foreign": changed["entries"][0]["endpoint"] = ["foreign", 2]
    elif damage == "foreign_region": changed["entries"][0]["mask"]["source_region_id"] = "0" * 64
    else: changed["entries"].reverse()
    reseal(changed)
    monkeypatch.setattr(subject, "_decode_source_mask", lambda *_: pytest.fail("bad inventory allocated"))
    with pytest.raises(subject.DiagnosticError): subject._validate_source_region_masks(changed, record)


def test_missing_live_mask_or_changed_live_reading_join_refuses():
    capture, record, _ = fixture_source()
    capture.graph.readings[0].region_evidence = None
    with pytest.raises(subject.DiagnosticError, match="missing_source_mask"):
        subject._source_region_mask_bundle(capture, record)
    capture, record, _ = fixture_source()
    bindings = list(capture.graph.selection.reading_bindings)
    bindings[0] = (bindings[0][0], 1, 1, "1" * 64, (), ())
    capture.graph.selection.reading_bindings = tuple(bindings)
    with pytest.raises(subject.DiagnosticError, match="mask_live_reading_join"):
        subject._source_region_mask_bundle(capture, record)


def test_duplicate_live_binding_refused_before_decode(monkeypatch):
    capture, record, _ = fixture_source()
    capture.graph.selection.reading_bindings += (capture.graph.selection.reading_bindings[0],)
    monkeypatch.setattr(regions, "_decode_region_evidence", lambda *_: pytest.fail("duplicate source allocated"))
    with pytest.raises(subject.DiagnosticError): subject._source_region_mask_bundle(capture, record)


def preflight_entries(full_shapes, crop_shapes):
    _, _, bundle, _ = bundle_fixture(); entries = copy.deepcopy(bundle["entries"])
    for entry, full, crop in zip(entries, full_shapes, crop_shapes):
        mask = entry["mask"]; mask["image_shape"] = full; mask["crop_shape"] = crop
        mask["bbox_xyxy"] = [0, 0, crop[1], crop[0]]; mask["pixel_count"] = 1
        packed = (crop[0] * crop[1] + 7) // 8
        mask["packed_mask_base64"] = "A" * (4 * ((packed + 2) // 3))
    return entries


@pytest.mark.parametrize("one_over", [False, True])
def test_pixel_bound_before_decode_unpack_or_allocation(monkeypatch, one_over):
    entries = preflight_entries([[1, 4000000 + int(one_over)]] * 3, [[1, 1]] * 3)
    monkeypatch.setattr(regions.base64, "b64decode", lambda *_a, **_k: pytest.fail("base64 before bounds"))
    monkeypatch.setattr(np, "unpackbits", lambda *_a, **_k: pytest.fail("unpack before bounds"))
    monkeypatch.setattr(np, "zeros", lambda *_a, **_k: pytest.fail("allocation before bounds"))
    if one_over:
        with pytest.raises(subject.DiagnosticError, match="mask_pixels"): subject._preflight_mask_entries(entries)
    else:
        assert subject._preflight_mask_entries(entries) == {"full_pixels": 12000000, "packed_bytes": 3}


@pytest.mark.parametrize("one_over", [False, True])
def test_aggregate_packed_bound_before_decode(monkeypatch, one_over):
    # 500000 + 24287 + 1 = 524288; adding 8 crop pixels adds one packed byte.
    width = 194296 + 8 * int(one_over)
    entries = preflight_entries([[2000, 2000], [1, width], [1, 1]], [[2000, 2000], [1, width], [1, 1]])
    monkeypatch.setattr(regions.base64, "b64decode", lambda *_a, **_k: pytest.fail("decode before cap"))
    if one_over:
        with pytest.raises(subject.DiagnosticError, match="mask_packed_bytes"): subject._preflight_mask_entries(entries)
    else:
        assert subject._preflight_mask_entries(entries)["packed_bytes"] == 524288


@pytest.mark.parametrize("extra", [-1, 1])
def test_exact_encoded_length_preflight(extra, monkeypatch):
    _, record, bundle, _ = bundle_fixture(); changed = copy.deepcopy(bundle)
    payload = changed["entries"][0]["mask"]["packed_mask_base64"]
    changed["entries"][0]["mask"]["packed_mask_base64"] = payload[:-1] if extra < 0 else payload + "A"
    reseal(changed)
    monkeypatch.setattr(subject, "_decode_source_mask", lambda *_: pytest.fail("bad length allocated"))
    with pytest.raises(subject.DiagnosticError, match="mask_base64_length"):
        subject._validate_source_region_masks(changed, record)


def test_exact_bundle_byte_cap_and_one_over_before_decode(monkeypatch):
    _, record, bundle, _ = bundle_fixture(); changed = copy.deepcopy(bundle)
    # Valid presentation source label belongs to the source context, not an invented bundle field.
    source = changed["source_context"]["ingress"][0]
    size = len(subject._bounded_bytes(changed)); source["source_label"] += "x" * (1048576 - size)
    reseal(changed)
    assert len(subject._bounded_bytes(changed, limit=1048576)) == 1048576
    monkeypatch.setattr(subject, "_decode_source_mask", lambda *_: pytest.fail("unbound presentation allocated"))
    with pytest.raises(subject.DiagnosticError, match="mask_bundle_source"):
        subject._validate_source_region_masks(changed, record)
    source["source_label"] += "x"
    with pytest.raises(subject.DiagnosticError) as error:
        subject._validate_source_region_masks(changed, record)
    assert error.value.reason == "evidence_cap"


def test_default_interface_unchanged_optin_cli_only():
    assert set(inspect.signature(subject.run).parameters) == {"images", "output", "local_period_support"}
    off = subject._parse_args(["--image", "synthetic.png", "--output", "new"])
    on = subject._parse_args(["--image", "synthetic.png", "--output", "new", "--source-region-masks"])
    assert vars(off) == {"image": ["synthetic.png"], "output": "new", "local_period_support": False}
    assert vars(on) == {**vars(off), "source_region_masks": True}
    assert subject.SCHEMA.endswith("-v1") and subject.JOINT_ENDPOINT_SCHEMA.endswith("-v3")


def test_optin_default_calls_and_stage_result_parity(monkeypatch, tmp_path):
    from test_diagnose_image_geometry_failure import harness as original_harness
    harness = original_harness.__wrapped__(tmp_path, monkeypatch)
    output_off = subject.run(images=harness.paths, output=tmp_path / "off")
    before = dict(harness.counts)
    original = subject._source_region_mask_bundle
    marker = {"synthetic": "export seam only"}
    monkeypatch.setattr(subject, "_source_region_mask_bundle", lambda capture, record: marker)
    monkeypatch.setattr(subject, "_validate_mask_diagnostic", lambda record: None)
    output_on = subject._run(images=harness.paths, output=tmp_path / "on", source_region_masks=True)
    off, on = (json.loads(Path(value["path"]).read_text()) for value in (output_off, output_on))
    assert harness.counts == {k: 2 * v for k, v in before.items()}
    assert off["stage_digests"] == on["stage_digests"] and off["graph"] == on["graph"]
    assert off["joint_endpoint_selection"] == on["joint_endpoint_selection"]
    assert "source_region_masks" not in off and off["schema"] == subject.SCHEMA
    on.pop("record_sha256"); on.pop("source_region_masks"); on["schema"] = subject.SCHEMA
    off.pop("record_sha256")
    assert subject._bounded_bytes(off) == subject._bounded_bytes(on)
    monkeypatch.setattr(subject, "_source_region_mask_bundle", original)


def test_staged_canonical_readback_seal_then_final_readback(tmp_path):
    _, record, _, _ = bundle_fixture(); destination = tmp_path / "diagnostic.json"
    encoded = subject._bounded_bytes(record) + b"\n"
    subject._publish_new(destination, encoded, validator=subject._validate_mask_diagnostic)
    assert destination.read_bytes() == encoded and not (tmp_path / ".diagnostic.json.partial").exists()
    assert subject._strict_readback(destination, encoded, subject._validate_mask_diagnostic) == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("failure", ["shortwrite", "fsync", "staged_readback", "link", "final_readback", "collision", "temp_collision"])
def test_publication_failure_has_no_success_and_preserves_collision(tmp_path, monkeypatch, failure):
    _, record, _, _ = bundle_fixture(); destination = tmp_path / "diagnostic.json"
    temporary = tmp_path / ".diagnostic.json.partial"; encoded = subject._bounded_bytes(record) + b"\n"
    if failure == "collision": destination.write_bytes(b"foreign-final")
    if failure == "temp_collision": temporary.write_bytes(b"foreign-temp")
    if failure == "shortwrite":
        original_open = Path.open
        class Writer:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): self.stream.__enter__(); return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def write(self, data): return self.stream.write(data[:10])
        def open_path(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return Writer(stream) if path == temporary and args == ("xb",) else stream
        monkeypatch.setattr(Path, "open", open_path)
    elif failure == "fsync":
        monkeypatch.setattr(subject.os, "fsync", lambda *_: (_ for _ in ()).throw(OSError("fsync")))
    elif failure == "link":
        monkeypatch.setattr(subject.os, "link", lambda *_: (_ for _ in ()).throw(OSError("unsupported link")))
    elif failure in ("staged_readback", "final_readback"):
        original = subject._strict_readback
        def readback(path, data, validator=None):
            if path == (temporary if failure == "staged_readback" else destination):
                raise subject.DiagnosticError("binding_mismatch", "publication_readback")
            return original(path, data, validator)
        monkeypatch.setattr(subject, "_strict_readback", readback)
    with pytest.raises((OSError, subject.DiagnosticError)):
        subject._publish_new(destination, encoded, validator=subject._validate_mask_diagnostic)
    if failure == "collision": assert destination.read_bytes() == b"foreign-final"
    elif failure == "final_readback": assert destination.read_bytes() == encoded  # orphan is not successful evidence
    else: assert not destination.exists()
    if failure == "temp_collision": assert temporary.read_bytes() == b"foreign-temp"
    else: assert not temporary.exists()


def test_strict_duplicate_or_noncanonical_json_never_links(tmp_path):
    for data in (b'{"schema":"x","schema":"y"}\n', b'{ "schema": "x" }\n', b'{"schema":NaN}\n'):
        path = tmp_path / "diagnostic.json"
        with pytest.raises(subject.DiagnosticError):
            subject._publish_new(path, data, validator=subject._validate_mask_diagnostic)
        assert not path.exists()


def test_all_task_budgets_fixed():
    assert subject._mask_limits() == {"mask_pixels": 4000000, "aggregate_pixels": 12000000,
                                    "aggregate_packed_bytes": 524288, "bundle_bytes": 1048576}
    assert (subject.MAX_RECORD_BYTES, subject.MAX_JSON_ITEMS, subject.MAX_JSON_DEPTH, subject.MAX_JOINT_ENDPOINT_BYTES) == (16777216, 262144, 24, 4194304)
