"""Focused tests for the DS-IMG-V2-SUPPORT-PREFLIGHT evaluation preflight.

These tests read the ONE accepted frozen V2 Route B record when it is present in
this repository. They never open an image, camera, truth, mask, scene, world,
Litematic, JAR or evaluator payload and never run an inference, matcher, solver
or signature replay.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry import evaluation_same_axis_ownership_preflight as pf

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_RECORD = (REPO_ROOT / ".tmp-routeb-v2-frozen-r4-20261004"
                 / "record.json")

_FROZEN_BYTES: bytes | None = None


def _frozen_bytes() -> bytes:
    global _FROZEN_BYTES
    if _FROZEN_BYTES is None:
        _FROZEN_BYTES = FROZEN_RECORD.read_bytes()
    return _FROZEN_BYTES


def _fresh_document() -> dict:
    return json.loads(_frozen_bytes().decode("utf-8"))


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()


def _resign(document: dict) -> None:
    document["artifact_sha256"] = _digest(
        {key: value for key, value in document.items()
         if key != "artifact_sha256"})


def _write(tmp_path: Path, document: dict) -> Path:
    target = tmp_path / "record.json"
    target.write_text(json.dumps(document, separators=(",", ":")),
                      encoding="utf-8")
    return target


requires_frozen = pytest.mark.skipif(
    not FROZEN_RECORD.is_file(),
    reason="frozen V2 Route B record is not present in this checkout")


# --- Positive frozen-record preflight ----------------------------------------

@requires_frozen
def test_frozen_record_reproduces_six_cross_tabs_and_unknown_rows():
    result = pf.run_same_axis_ownership_preflight()

    assert result.status == pf.STATUS_COMPLETE
    assert result.reason == ""
    assert result.input_bytes == 16_896_513
    assert result.input_sha256 == (
        "006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee")
    assert result.row_count == 1_928
    assert result.present_row_count == 971
    assert result.unknown_row_count == 957
    assert result.kept_row_count == 38
    assert result.dropped_row_count == 1_890
    assert result.index_page_count == 5
    assert result.digest_page_count == 6
    assert result.sidecar_page_count == 67
    assert result.sidecar_entry_count == 971
    assert result.payload_element_count == 4_576_118
    assert result.payload_elements_allocated == 4_576_118
    assert result.signature_witness_count == 85
    assert result.bound_witness_count == 66
    assert result.refused_witness_count == 19
    assert tuple(result.cross_tabs) == pf.FROZEN_ROUTE_B_EXPECTATIONS.cross_tabs

    endpoint = result.endpoint_join
    assert endpoint is not None
    assert endpoint.referenced_signature_count == 78
    assert endpoint.referenced_bound_count == 59
    assert endpoint.referenced_unbound_count == 19
    assert endpoint.bound_unreferenced_count == 7
    assert endpoint.missing_witness_row_count == 525
    assert endpoint.missing_witness_over_cap_count == 412
    assert endpoint.missing_witness_refusal_count == 113
    assert endpoint.raw_cap_provable_count == 102
    # 412 + 113 closes the 525 rows with at least one unavailable endpoint.
    assert (endpoint.missing_witness_over_cap_count
            + endpoint.missing_witness_refusal_count
            == endpoint.missing_witness_row_count)

    # All six cross-tab rows close the exact 1,928-row universe.
    assert sum(row[3] for row in result.cross_tabs) == 1_928
    # The 957 UNKNOWN rows are exactly the refused rows; no present row is
    # counted unknown and no refused row carries a payload digest.
    assert result.unknown_row_count == sum(
        row[3] for row in result.cross_tabs if row[1] == pf.v2.SUPPORT_STATE_REFUSED)

    assert result.ingress_frame_ids == ("image-0000", "image-0001", "image-0002")
    for inventory in result.frame_pins:
        states = {pin.pin: pin.state for pin in inventory.pins}
        assert set(states) == set(pf.REQUIRED_ALIGNMENT_PINS)
        assert states[pf.PIN_SOURCE_TRANSFORM] == pf.PIN_STATE_AVAILABLE
        for name in (pf.PIN_CAMERA, pf.PIN_INTRINSICS, pf.PIN_POSE,
                     pf.PIN_OCCLUSION, pf.PIN_EVALUATOR_DOMAIN):
            assert states[name] == pf.PIN_STATE_UNAVAILABLE

    assert result.tap_surface_state == pf.PIN_STATE_AVAILABLE
    assert result.sampling_witness_state == pf.PIN_STATE_UNAVAILABLE
    assert result.sampling_witness_seam == pf.SAMPLING_WITNESS_TAP_SEAM
    assert result.ownership_authorization == pf.OWNERSHIP_WITHHELD
    assert "sampling_witness_tap_tables_not_persisted" in result.ownership_blockers
    assert result.gate_sealed is True
    assert result.evaluator_payload_read is False

    # No ownership/correctness percentage is emitted anywhere.
    serialized = json.dumps(result.to_record()).lower()
    for forbidden in ("accuracy", "percent", "recall", "correctness", "ownership_ratio"):
        assert forbidden not in serialized


@requires_frozen
def test_frozen_gate_stays_sealed_after_evaluator_inventory():
    gate = pf.InferenceDigestGate()
    result = pf.run_same_axis_ownership_preflight(gate=gate)
    assert result.status == pf.STATUS_COMPLETE
    assert gate.sealed is True
    seal = gate.require_evaluator_access("frame_pins")
    assert seal.record_sha256 == (
        "6eddf7f2bb60c1bdeac053d86e592dcd3c5c1b4bdd91a056c960c6f8db8ba797")


# --- Asymmetric source/target orientation ------------------------------------

def test_asymmetric_left_source_right_target_orientation_is_refused():
    digest_left = "a" * 64
    digest_right = "b" * 64
    row = SimpleNamespace(left="image-0000", right="image-0001",
                          left_alternative_index=3, right_alternative_index=2,
                          support_refusal_reason=None, overlap_cells=10)
    sections = {"left_signature_sha256": (digest_left,),
                "right_signature_sha256": (digest_right,)}
    correct = {
        digest_left: {"frame_id": "image-0000", "alternative_index": 3},
        digest_right: {"frame_id": "image-0001", "alternative_index": 2},
    }
    facts = pf.verify_endpoint_alternative_join([row], sections, correct)
    assert facts.endpoint_rows == 1
    assert facts.missing_witness_row_count == 0
    assert facts.referenced_signature_count == 2

    # A swapped source/target binding is refused rather than silently accepted.
    swapped = {
        digest_left: {"frame_id": "image-0001", "alternative_index": 2},
        digest_right: {"frame_id": "image-0000", "alternative_index": 3},
    }
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf.verify_endpoint_alternative_join([row], sections, swapped)
    assert error.value.reason == pf.REASON_ENDPOINT_ALTERNATIVE_MISMATCH

    # Swapping the endpoint alternative index is also refused.
    wrong_alternative = {
        digest_left: {"frame_id": "image-0000", "alternative_index": 2},
        digest_right: {"frame_id": "image-0001", "alternative_index": 3},
    }
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf.verify_endpoint_alternative_join([row], sections, wrong_alternative)
    assert error.value.reason == pf.REASON_ENDPOINT_ALTERNATIVE_MISMATCH


@requires_frozen
def test_orientation_field_tamper_is_refused(tmp_path):
    document = _fresh_document()
    document["capture"]["index_pages"][0]["rows"][0]["orientation"] = (
        "right-source-left-target")
    _resign(document)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_ORIENTATION_MISMATCH


# --- Tampered index / signature / ingress ------------------------------------

@requires_frozen
def test_tampered_index_row_is_refused(tmp_path):
    document = _fresh_document()
    document["capture"]["index_pages"][0]["rows"][0]["left_b"] = 0.9
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason in (pf.REASON_INDEX_REPLAY_MISMATCH,
                             pf.REASON_CAPTURE_VALIDATION_FAILED)


@requires_frozen
def test_tampered_signature_section_is_refused(tmp_path):
    document = _fresh_document()
    section = next(page for page in document["capture"]["digest_pages"]
                   if page["section"] == "left_signature_sha256")
    section["digests"][5] = "0" * 64
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason in (pf.REASON_DIGEST_REPLAY_MISMATCH,
                             pf.REASON_CAPTURE_VALIDATION_FAILED)


@requires_frozen
def test_tampered_ingress_is_refused(tmp_path):
    document = _fresh_document()
    document["ingress"][0]["source_label"] = "C:\\foreign\\frame.png"
    _resign(document)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_INGRESS_JOIN_MISMATCH


# --- Duplicate row / missing payload / sentinel ------------------------------

@requires_frozen
def test_duplicate_admitted_row_is_refused(tmp_path):
    document = _fresh_document()
    rows = document["capture"]["index_pages"][0]["rows"]
    rows[1] = copy.deepcopy(rows[0])
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_DUPLICATE_ROW_IDENTITY


@requires_frozen
def test_missing_sidecar_payload_is_refused_at_exact_seam(tmp_path):
    document = _fresh_document()
    entries = document["capture"]["sidecar_pages"][0]["entries"]
    removed = entries.pop(0)
    document["capture"]["sidecar_pages"][0]["entry_count"] = len(entries)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_SOURCE_TAP_RECONSTRUCTION_UNAVAILABLE
    assert result.field == f"sidecar.entry.{removed['admitted_row_index']}"


def test_source_tap_index_sentinel_semantics():
    assert pf.verify_source_tap_index_semantics([0, -1, 5], 10) == 1
    assert pf.verify_source_tap_index_semantics([0, 1, 2], 10) == 0
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf.verify_source_tap_index_semantics([-2, 0], 10)
    assert error.value.reason == pf.REASON_SENTINEL_INVALID
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf.verify_source_tap_index_semantics([10], 10)
    assert error.value.reason == pf.REASON_TAP_INDEX_INVALID
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf.verify_source_tap_index_semantics([-1], 10, allow_border_sentinel=False)
    assert error.value.reason == pf.REASON_TAP_INDEX_INVALID


@requires_frozen
def test_negative_sentinel_value_is_refused(tmp_path):
    document = _fresh_document()
    mutated = False
    for page in document["capture"]["sidecar_pages"]:
        for entry in page["entries"]:
            values = entry["payload"]["warp_source_rows"]["values"]
            for row_index, row in enumerate(values):
                for column_index, value in enumerate(row):
                    if value == -1:
                        row[column_index] = -2
                        mutated = True
                        break
                if mutated:
                    break
            if mutated:
                break
        if mutated:
            break
    assert mutated
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_SENTINEL_INVALID


# --- Caps before allocation ---------------------------------------------------

def test_byte_cap_is_checked_before_the_input_is_read(tmp_path):
    target = tmp_path / "record.json"
    target.write_bytes(b"{}")
    called = []

    def _reader(_path):
        called.append(True)
        raise AssertionError("the reader must not run for an over-cap input")

    result = pf.run_same_axis_ownership_preflight(
        target, limits=pf.PreflightLimitsV1(max_input_bytes=1),
        expectations=None, reader=_reader)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_INPUT_OVER_CAP
    assert called == []


def test_json_depth_and_token_scans_run_before_parsing():
    assert pf.scan_json_bounds(b'{"a":[1,2,3]}') == (2, 5)
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf.scan_json_bounds(b"[[")
    assert error.value.reason == pf.REASON_RECORD_MALFORMED  # unbalanced input


@requires_frozen
def test_depth_work_and_row_caps_refuse_before_typed_allocation():
    depth = pf.run_same_axis_ownership_preflight(
        FROZEN_RECORD, limits=pf.PreflightLimitsV1(max_json_depth=1),
        expectations=None)
    assert depth.status == pf.STATUS_REFUSED
    assert depth.reason == pf.REASON_DEPTH_OVER_CAP

    work = pf.run_same_axis_ownership_preflight(
        FROZEN_RECORD, limits=pf.PreflightLimitsV1(max_json_tokens=10),
        expectations=None)
    assert work.status == pf.STATUS_REFUSED
    assert work.reason == pf.REASON_WORK_OVER_CAP

    row_cap = pf.run_same_axis_ownership_preflight(
        FROZEN_RECORD, limits=pf.PreflightLimitsV1(max_universe_rows=10),
        expectations=None)
    assert row_cap.status == pf.STATUS_REFUSED
    assert row_cap.reason == pf.REASON_ROW_CAP_EXCEEDED
    # The row cap fired in the plain-document prescan: no sidecar payload arrays
    # were ever allocated.
    assert row_cap.payload_elements_allocated == 0
    assert row_cap.sidecar_entry_count == 0


@requires_frozen
def test_payload_element_cap_refuses_before_array_allocation():
    result = pf.run_same_axis_ownership_preflight(
        FROZEN_RECORD, limits=pf.PreflightLimitsV1(max_payload_elements=100),
        expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_PAYLOAD_OVER_CAP
    assert result.payload_elements_allocated == 0


# --- Evaluator access gate ----------------------------------------------------

def test_evaluator_access_gate_blocks_before_frozen_inference_digest():
    gate = pf.InferenceDigestGate()
    assert gate.sealed is False
    with pytest.raises(pf.EvaluatorAccessRefused) as error:
        gate.require_evaluator_access("camera")
    assert error.value.reason == pf.REASON_EVALUATOR_ACCESS_BEFORE_INFERENCE_DIGEST
    assert error.value.field == "camera"
    assert gate.seal is None

    gate.seal_inference_digest(
        artifact_sha256="1" * 64, record_sha256="2" * 64,
        whole_universe_identity_sha256="3" * 64,
        input_match_identity_sha256="4" * 64,
        failed_assignment_sha256="5" * 64)
    assert gate.sealed is True
    seal = gate.require_evaluator_access("evaluator_domain")
    assert seal.artifact_sha256 == "1" * 64


def test_gate_refuses_a_non_hex_inference_digest():
    gate = pf.InferenceDigestGate()
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        gate.seal_inference_digest(
            artifact_sha256="not-a-digest", record_sha256="2" * 64,
            whole_universe_identity_sha256="3" * 64,
            input_match_identity_sha256="4" * 64,
            failed_assignment_sha256="5" * 64)
    assert error.value.reason == pf.REASON_RECORD_PIN_MISMATCH
    assert gate.sealed is False


# --- Bounded stable opened-file reader (growth / replacement) -----------------

def test_bounded_stream_read_is_capped_at_max_bytes_plus_one():
    payload = b"a" * 4096
    assert pf._read_stream_bounded(io.BytesIO(payload), 64) == payload[:65]
    assert pf._read_stream_bounded(io.BytesIO(payload), 8192) == payload


def test_default_reader_never_uses_unbounded_read_bytes(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_bytes(b'{"a":[1,2,3]}')

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("unbounded Path.read_bytes must not be used")

    monkeypatch.setattr(Path, "read_bytes", _forbidden)
    bounded = pf.read_bounded_record_bytes(target)
    assert bounded.data == b'{"a":[1,2,3]}'
    assert bounded.byte_length == len(b'{"a":[1,2,3]}')
    assert bounded.max_json_depth == 2


def test_linklike_check_is_fail_closed(tmp_path):
    target = tmp_path / "record.json"
    target.write_bytes(b"{}")
    assert pf._linklike_path(target) is False
    # A path that cannot be lstat'ed is refused rather than followed.
    assert pf._linklike_path(tmp_path / "missing.json") is True


def test_record_reader_refuses_simulated_growth_without_retaining_bytes(
        tmp_path, monkeypatch):
    cap = 64
    target = tmp_path / "record.json"
    target.write_bytes(b'{"a":1}')
    real_read = pf.os.read
    pulled = {"bytes": 0}
    grown = {"done": False}

    def _growing_read(descriptor, count):
        chunk = real_read(descriptor, count)
        if chunk and not grown["done"]:
            grown["done"] = True
            with open(target, "ab") as handle:
                handle.write(b"z" * 4096)
            chunk += real_read(descriptor, count - len(chunk))
        pulled["bytes"] += len(chunk)
        return chunk

    monkeypatch.setattr(pf.os, "read", _growing_read)
    result = pf.run_same_axis_ownership_preflight(
        target, limits=pf.PreflightLimitsV1(max_input_bytes=cap),
        expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_INPUT_OVER_CAP
    # The refused result retains no over-cap bytes...
    assert result.input_bytes == 0
    assert result.input_sha256 == ""
    # ...and the bounded loop never pulled more than cap + 1 bytes.
    assert pulled["bytes"] <= cap + 1


def test_record_reader_refuses_simulated_replacement(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_bytes(b'{"a":1}')
    real_fstat = pf.os.fstat
    before = os.stat(target)
    calls = {"count": 0}

    def _replacing_fstat(descriptor):
        info = real_fstat(descriptor)
        if (info.st_dev, info.st_ino) == (before.st_dev, before.st_ino):
            calls["count"] += 1
            if calls["count"] >= 2:
                return SimpleNamespace(
                    st_dev=info.st_dev, st_ino=info.st_ino + 1,
                    st_size=info.st_size, st_mtime_ns=info.st_mtime_ns)
        return info

    monkeypatch.setattr(pf.os, "fstat", _replacing_fstat)
    result = pf.run_same_axis_ownership_preflight(target, expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_INPUT_SIZE_DRIFT
    assert result.input_sha256 == ""


# --- Nested values validated before NumPy allocation --------------------------

def _first_payload_array(document: dict, name: str = "logical_pairs") -> dict:
    entry = document["capture"]["sidecar_pages"][0]["entries"][0]
    return entry["payload"][name]


@requires_frozen
def test_declared_shape_smaller_than_values_is_refused_before_np_asarray(
        tmp_path, monkeypatch):
    document = _fresh_document()
    array = _first_payload_array(document)
    assert array["shape"] == [147, 4]
    # Declares only 4 elements while ``values`` still nests the full 588.
    array["shape"] = [1, 4]
    calls = []

    def _trap(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("np.asarray must not run on mismatched values")

    monkeypatch.setattr(np, "asarray", _trap)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_SIDECAR_REPLAY_MISMATCH
    assert "logical_pairs" in result.field
    assert calls == []
    assert result.payload_elements_allocated == 0


@requires_frozen
def test_values_row_count_mismatch_is_refused_before_np_asarray(
        tmp_path, monkeypatch):
    document = _fresh_document()
    array = _first_payload_array(document)
    array["values"].pop()  # 146 rows against declared shape [147, 4]
    calls = []
    monkeypatch.setattr(np, "asarray", lambda *_a, **_k: calls.append(True))
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_SIDECAR_REPLAY_MISMATCH
    assert "logical_pairs" in result.field
    assert calls == []
    assert result.payload_elements_allocated == 0


@requires_frozen
def test_nested_list_leaf_is_refused_before_np_asarray(tmp_path, monkeypatch):
    document = _fresh_document()
    array = _first_payload_array(document)
    array["values"][0][0] = [1]  # a list where the declared 2-D leaf must be scalar
    calls = []
    monkeypatch.setattr(np, "asarray", lambda *_a, **_k: calls.append(True))
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_SIDECAR_REPLAY_MISMATCH
    assert "logical_pairs" in result.field
    assert calls == []
    assert result.payload_elements_allocated == 0


@requires_frozen
def test_over_cap_nested_values_are_refused_before_np_asarray(
        tmp_path, monkeypatch):
    document = _fresh_document()
    calls = []
    monkeypatch.setattr(np, "asarray", lambda *_a, **_k: calls.append(True))
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document),
        limits=pf.PreflightLimitsV1(max_payload_elements=16),
        expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_PAYLOAD_OVER_CAP
    assert calls == []
    assert result.payload_elements_allocated == 0


# --- Exact contract dtype / payload-byte caps before NumPy --------------------

def _trap_numpy(monkeypatch) -> dict:
    """Trap both NumPy entry points a tampered dtype could reach."""
    calls = {"asarray": 0, "dtype": 0}

    def _asarray(*_args, **_kwargs):
        calls["asarray"] += 1
        raise AssertionError("np.asarray must not run before validation")

    def _dtype(*_args, **_kwargs):
        calls["dtype"] += 1
        raise AssertionError("np.dtype must not run on an unvalidated dtype")

    monkeypatch.setattr(np, "asarray", _asarray)
    monkeypatch.setattr(np, "dtype", _dtype)
    return calls


def test_contract_dtype_and_payload_byte_caps_are_exact():
    assert pf.PAYLOAD_CONTRACT_DTYPE == "<i8"
    assert pf.PAYLOAD_CONTRACT_ITEM_BYTES == 8
    assert pf.MAX_PAYLOAD_BYTES == pf.MAX_PAYLOAD_ELEMENTS * 8 == 48_000_000
    limits = pf.PreflightLimitsV1()
    assert limits.max_payload_bytes == pf.MAX_PAYLOAD_BYTES
    assert limits.max_payload_array_bytes == pf.MAX_PAYLOAD_BYTES


@requires_frozen
def test_oversized_string_dtype_is_refused_before_np_dtype_or_asarray(
        tmp_path, monkeypatch):
    document = _fresh_document()
    array = _first_payload_array(document)
    # On this valid [147, 4] array a ``U1048576`` dtype would request ~2.47 GB.
    array["dtype"] = "U1048576"
    calls = _trap_numpy(monkeypatch)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_PAYLOAD_DTYPE_INVALID
    assert "logical_pairs" in result.field
    assert calls == {"asarray": 0, "dtype": 0}
    assert result.payload_elements_allocated == 0


# Every one of these is a NumPy-accepted spelling or a different concrete dtype;
# only the exact serialized contract string ``<i8`` may pass.
NON_CONTRACT_DTYPES = (
    "int64", "i8", "=i8", ">i8", "<i4", "int32", "<u8", "uint8", "float64",
    "|u1", "<U8", "U8", "S8", "|b1", "bool", None, 8, ["<i8"],
)


@pytest.mark.parametrize("dtype", NON_CONTRACT_DTYPES)
@requires_frozen
def test_any_non_contract_dtype_is_refused_before_numpy(
        dtype, tmp_path, monkeypatch):
    document = _fresh_document()
    _first_payload_array(document)["dtype"] = dtype
    calls = _trap_numpy(monkeypatch)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_PAYLOAD_DTYPE_INVALID
    assert calls == {"asarray": 0, "dtype": 0}
    assert result.payload_elements_allocated == 0


@requires_frozen
def test_per_array_payload_byte_cap_refuses_before_numpy(tmp_path, monkeypatch):
    document = _fresh_document()
    # The first payload array is [147, 4] = 588 contract elements = 4,704 bytes.
    calls = _trap_numpy(monkeypatch)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document),
        limits=pf.PreflightLimitsV1(max_payload_array_bytes=4_703),
        expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_PAYLOAD_OVER_CAP
    assert result.field == "payload.logical_pairs.bytes"
    assert calls == {"asarray": 0, "dtype": 0}
    assert result.payload_elements_allocated == 0


@requires_frozen
def test_aggregate_payload_byte_cap_refuses_across_arrays_before_numpy(
        tmp_path, monkeypatch):
    document = _fresh_document()
    # Exactly the first array's contract bytes: the second array (4,608 bytes)
    # can no longer fit the aggregate budget even though each array is under its
    # own per-array cap.
    calls = _trap_numpy(monkeypatch)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document),
        limits=pf.PreflightLimitsV1(max_payload_bytes=4_704),
        expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_PAYLOAD_OVER_CAP
    assert result.field == "payload_bytes"
    assert calls == {"asarray": 0, "dtype": 0}
    assert result.payload_elements_allocated == 0


# --- Exact JSON-integer leaves and contract range before NumPy ----------------

NON_INTEGER_LEAVES = (1.0, 1.5, True, False, "1", None)


@pytest.mark.parametrize("leaf", NON_INTEGER_LEAVES)
@requires_frozen
def test_non_integer_leaf_is_refused_before_numpy(leaf, tmp_path, monkeypatch):
    document = _fresh_document()
    _first_payload_array(document)["values"][0][0] = leaf
    calls = _trap_numpy(monkeypatch)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_PAYLOAD_VALUE_INVALID
    assert "logical_pairs" in result.field
    assert calls == {"asarray": 0, "dtype": 0}
    assert result.payload_elements_allocated == 0


@requires_frozen
def test_out_of_range_integer_leaf_is_refused_before_numpy(
        tmp_path, monkeypatch):
    document = _fresh_document()
    _first_payload_array(document)["values"][0][0] = (
        pf.v2.PAYLOAD_MAX_ABS_ELEMENT + 1)
    calls = _trap_numpy(monkeypatch)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_TAP_INDEX_INVALID
    assert calls == {"asarray": 0, "dtype": 0}
    assert result.payload_elements_allocated == 0


@requires_frozen
def test_logical_pairs_negative_sentinel_is_refused_before_numpy(
        tmp_path, monkeypatch):
    document = _fresh_document()
    # ``-1`` is valid only in the two warp maps, never in ``logical_pairs``.
    _first_payload_array(document, "logical_pairs")["values"][0][0] = -1
    calls = _trap_numpy(monkeypatch)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_TAP_INDEX_INVALID
    assert calls == {"asarray": 0, "dtype": 0}
    assert result.payload_elements_allocated == 0


def test_minus_one_is_preserved_only_in_border_sentinel_arrays():
    limits = pf.PreflightLimitsV1()
    values = [[-1], [0], [5]]
    array = np.asarray(values, dtype="<i8")
    record = {"dtype": "<i8", "shape": [3, 1], "values": values,
              "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}

    accepted = pf._array_from_record(
        dict(record), limits, allow_border_sentinel=True)
    assert accepted.tolist() == [[-1], [0], [5]]
    assert accepted.dtype.str == pf.PAYLOAD_CONTRACT_DTYPE
    assert accepted.flags.writeable is False

    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf._array_from_record(
            dict(record), limits, allow_border_sentinel=False)
    assert error.value.reason == pf.REASON_TAP_INDEX_INVALID


@requires_frozen
def test_conversion_overflow_is_mapped_to_a_typed_refusal(tmp_path, monkeypatch):
    document = _fresh_document()

    def _overflow(*_args, **_kwargs):
        raise OverflowError("simulated conversion overflow")

    monkeypatch.setattr(np, "asarray", _overflow)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_SIDECAR_REPLAY_MISMATCH
    assert result.detail == "OverflowError"
    assert result.payload_elements_allocated == 0


# ---------------------------------------------------------------------------
# Gate 6 - frozen JSON unchanged; V2 sampling availability; V1 legacy
# ---------------------------------------------------------------------------

PINNED_RESULT = (REPO_ROOT / "docs" / "reports"
                 / "SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json")


@requires_frozen
def test_frozen_result_record_is_byte_identical_to_the_pinned_json():
    result = pf.run_same_axis_ownership_preflight()
    regenerated = (json.dumps(pf.result_record(result), indent=1, sort_keys=True)
                   + "\n").encode("utf-8")
    assert regenerated == PINNED_RESULT.read_bytes()
    assert hashlib.sha256(regenerated).hexdigest() == (
        "448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d")
    # A V1 record never carries the V2 sampling key.
    assert "sampling_persistence" not in pf.result_record(result)


@requires_frozen
def test_frozen_v1_record_reports_the_legacy_sampling_seam_unchanged():
    result = pf.run_same_axis_ownership_preflight()
    assert result.record_schema == pf.RECORD_SCHEMA_V1
    assert result.sampling_witness_state == pf.PIN_STATE_UNAVAILABLE
    assert result.sampling_witness_seam == pf.SAMPLING_WITNESS_TAP_SEAM
    blockers = list(result.ownership_blockers)
    assert blockers[0] == "sampling_witness_tap_tables_not_persisted"
    assert result.ownership_authorization == pf.OWNERSHIP_WITHHELD


def _frozen_capture():
    document = _fresh_document()
    capture, _ = pf._capture_from_record(document["capture"],
                                         pf.PreflightLimitsV1())
    return document, capture


def _frozen_pin_bindings(pins):
    """Real ordered bindings over the record's own capture-root ingress pins.

    Each binding names a root pin as its ingress and carries a witness produced
    by the ordinary sampler over a synthetic frame shaped exactly like that pin,
    so the persisted section is fully root-pinned and decodes as available.
    """
    from mc_photogrammetry.pipeline import candidate_support_provenance as csp
    from mc_photogrammetry.reconstruction import lattice_planes
    from test_candidate_support_provenance import (
        _HEX as hex_a, _HEX_B as hex_b, _pixels, _profile_reading)

    pins = tuple(pins)
    assert pins
    pixels = _pixels(300, shape=tuple(pins[0].source_shape) + (3,))
    sink: list = []
    lattice_planes.plane_signature(
        pixels, _profile_reading(pins[0].frame_id), 1.0, witness_sink=sink)
    witness = sink[0]
    bindings = []
    for index, pin in enumerate(pins):
        bindings.append(csp.SourceTapWitnessBindingV1(
            frame_id=pin.frame_id, canonical_index=0, alternative_index=index,
            signature_sha256=(hex_a if index == 0 else hex_b), ingress=pin,
            mask_sha256=hex_b, masks_sha256=hex_a,
            max_cells=lattice_planes.MAX_SIGNATURE_CELLS,
            max_rectified_side=lattice_planes.MAX_RECTIFIED_SIDE,
            pixels_per_cell=lattice_planes.SIGNATURE_PIXELS_PER_CELL,
            witness=witness))
    return tuple(bindings)


def _synthetic_sampling_section(pins=None):
    """A real, ordered sampling section over synthetic bindings.

    With ``pins`` (the record's own capture-root ingress pins) the section is
    fully root-pinned; without them the runner's standalone synthetic bindings
    are used, whose frames are not capture-root pins.
    """
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    if pins is None:
        from test_route_b_diagnostic_runner import _route_b_witness_bindings

        bindings = _route_b_witness_bindings()
    else:
        bindings = _frozen_pin_bindings(pins)
    section = persistence.persist_signature_witnesses(bindings)
    summaries = [dict(binding.to_record(), outcome="bound")
                 for binding in bindings]
    return bindings, section, summaries


@requires_frozen
def test_v2_sampling_section_decodes_available_against_typed_ingress():
    document, capture = _frozen_capture()
    bindings, section, summaries = _synthetic_sampling_section(
        capture.root.ingress[:2])
    decoded, record = pf._decode_sampling_section(
        {"sampling_persistence": section, "signature_witnesses": summaries},
        capture)
    assert record["state"] == (
        pf.persistence.SAMPLING_PERSISTENCE_STATE_AVAILABLE)
    assert record["entry_count"] == 2
    assert record["bound_count"] == 2
    assert record["refused_count"] == 0
    assert record["persistence_refused_count"] == 0
    assert record["payload_bytes"] > 0 and record["metadata_bytes"] > 0
    # Every bound witness is pinned to one of the record's own root ingress pins.
    assert record["ingress_bound_count"] == 2
    assert record["ingress_unbound_count"] == 0
    assert decoded.bound_count == 2
    assert decoded.entries[0].witness.witness_sha256 == (
        bindings[0].witness.witness_sha256)


@requires_frozen
def test_v2_missing_root_pin_is_a_typed_refusal():
    document, capture = _frozen_capture()
    # A real pin, but not one of the record's own root ingress pins.
    from test_candidate_support_provenance import (
        _make_ingress, _pixels, _png_bytes)

    image = np.full(tuple(capture.root.ingress[0].source_shape) + (3,), 1,
                    dtype=np.uint8)
    foreign_pin = _make_ingress("foreign-frame", image, _png_bytes((3, 3, 3)))
    _bindings, section, summaries = _synthetic_sampling_section((foreign_pin,))
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf._decode_sampling_section(
            {"sampling_persistence": section, "signature_witnesses": summaries},
            capture)
    assert error.value.reason == pf.REASON_SAMPLING_PERSISTENCE_INVALID


@requires_frozen
def test_v2_foreign_root_pin_digest_is_a_typed_refusal():
    document, capture = _frozen_capture()
    from test_candidate_support_provenance import (
        _make_ingress, _pixels, _png_bytes)

    # Same frame id and shape as the record's pin, different bytes: the
    # persisted ingress digest can never match the record's root pin.
    root_pin = capture.root.ingress[0]
    foreign = _make_ingress(root_pin.frame_id,
                            _pixels(301, shape=tuple(root_pin.source_shape) + (3,)),
                            _png_bytes((4, 4, 4)))
    _bindings, section, summaries = _synthetic_sampling_section((foreign,))
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf._decode_sampling_section(
            {"sampling_persistence": section, "signature_witnesses": summaries},
            capture)
    assert error.value.reason == pf.REASON_SAMPLING_PERSISTENCE_INVALID


@requires_frozen
def test_v2_root_pinned_sampling_section_full_preflight_is_available(tmp_path):
    """Positive availability with the real decoder and the record's own pins."""
    document, capture = _frozen_capture()
    bindings, section, summaries = _synthetic_sampling_section(
        capture.root.ingress[:2])
    document["schema"] = pf.RECORD_SCHEMA_V2
    document["sampling_persistence"] = section
    document["signature_witnesses"] = summaries
    document["signature_witness_count"] = len(summaries)
    _resign(document)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_COMPLETE
    assert result.record_schema == pf.RECORD_SCHEMA_V2
    assert result.sampling_witness_state == pf.PIN_STATE_AVAILABLE
    assert result.sampling_witness_seam == ""
    assert result.sampling_persistence["state"] == "available"
    assert result.sampling_persistence["ingress_bound_count"] == 2
    assert result.sampling_persistence["ingress_unbound_count"] == 0
    assert result.sampling_persistence["section_sha256"] == (
        section["section_sha256"])
    blockers = list(result.ownership_blockers)
    assert "sampling_witness_tap_tables_not_persisted" not in blockers
    for name in ("camera_pin_absent", "intrinsics_pin_absent", "pose_pin_absent",
                 "occlusion_convention_absent", "evaluator_domain_absent"):
        assert name in blockers
    assert result.ownership_authorization == pf.OWNERSHIP_WITHHELD
    serialized = json.dumps(result.to_record()).lower()
    for forbidden in ("accuracy", "percent", "recall", "correctness",
                      "ownership_ratio"):
        assert forbidden not in serialized


@requires_frozen
def test_v2_resigned_sampling_tamper_is_refused_by_decode():
    document, capture = _frozen_capture()
    bindings, section, summaries = _synthetic_sampling_section(
        capture.root.ingress[:2])
    array = section["entries"][0]["witness"]["rgb_rectification"]["map_x"]
    raw = bytearray(base64.b64decode(array["b64"]))
    raw[0] ^= 0x01
    array["b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    array["sha256"] = hashlib.sha256(bytes(raw)).hexdigest()
    section["section_sha256"] = pf.persistence.section_digest(section)
    with pytest.raises(pf.SameAxisOwnershipPreflightError) as error:
        pf._decode_sampling_section(
            {"sampling_persistence": section, "signature_witnesses": summaries},
            capture)
    assert error.value.reason == pf.REASON_SAMPLING_PERSISTENCE_INVALID


@requires_frozen
def test_v2_refused_sampling_section_full_preflight_is_typed_unavailable(
        tmp_path):
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    document = _fresh_document()
    document["schema"] = pf.RECORD_SCHEMA_V2
    document["sampling_persistence"] = persistence.refused_section_record(
        persistence.REASON_PERSISTENCE_SERIALIZED_OVER_CAP, field="record")
    _resign(document)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_COMPLETE
    assert result.record_schema == pf.RECORD_SCHEMA_V2
    assert result.sampling_witness_state == pf.PIN_STATE_UNAVAILABLE
    assert result.sampling_witness_seam.startswith(
        pf.SAMPLING_WITNESS_SECTION_INVALID)
    assert result.sampling_persistence is not None
    assert result.sampling_persistence["status"] == "refused"
    assert "sampling_witness_tap_tables_not_persisted" in (
        result.ownership_blockers)
    assert result.ownership_authorization == pf.OWNERSHIP_WITHHELD


@requires_frozen
def test_v2_available_wiring_closes_the_tap_blocker_but_withholds_ownership(
        tmp_path, monkeypatch):
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    document = _fresh_document()
    document["schema"] = pf.RECORD_SCHEMA_V2
    document["sampling_persistence"] = persistence.refused_section_record(
        persistence.REASON_PERSISTENCE_SERIALIZED_OVER_CAP, field="record")
    _resign(document)
    fake = SimpleNamespace(
        entry_count=0,
        state=persistence.SAMPLING_PERSISTENCE_STATE_AVAILABLE)
    monkeypatch.setattr(
        pf, "_decode_sampling_section",
        lambda doc, capture: (fake, {"state": "available", "entry_count": 0}))
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_COMPLETE
    assert result.sampling_witness_state == pf.PIN_STATE_AVAILABLE
    assert result.sampling_witness_seam == ""
    blockers = list(result.ownership_blockers)
    assert "sampling_witness_tap_tables_not_persisted" not in blockers
    for name in ("camera_pin_absent", "intrinsics_pin_absent", "pose_pin_absent",
                 "occlusion_convention_absent", "evaluator_domain_absent"):
        assert name in blockers
    assert result.ownership_authorization == pf.OWNERSHIP_WITHHELD
    serialized = json.dumps(result.to_record()).lower()
    for forbidden in ("accuracy", "percent", "recall", "correctness",
                      "ownership_ratio"):
        assert forbidden not in serialized


@requires_frozen
def test_v2_missing_sampling_section_is_refused(tmp_path):
    document = _fresh_document()
    document["schema"] = pf.RECORD_SCHEMA_V2
    _resign(document)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_SAMPLING_PERSISTENCE_INVALID


@requires_frozen
def test_v1_schema_with_a_sampling_section_is_refused(tmp_path):
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    document = _fresh_document()
    document["schema"] = pf.RECORD_SCHEMA_V1
    document["sampling_persistence"] = persistence.refused_section_record(
        persistence.REASON_PERSISTENCE_SERIALIZED_OVER_CAP, field="record")
    _resign(document)
    result = pf.run_same_axis_ownership_preflight(
        _write(tmp_path, document), expectations=None)
    assert result.status == pf.STATUS_REFUSED
    assert result.reason == pf.REASON_SAMPLING_PERSISTENCE_INVALID


