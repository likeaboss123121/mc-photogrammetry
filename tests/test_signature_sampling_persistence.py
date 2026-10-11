"""Focused synthetic tests for DS-IMG-V2-SAMPLING-WITNESS-PERSISTENCE.

Only synthetic, in-memory material is used: tiny locally generated PNG bytes,
synthetic ``uint8`` arrays, synthetic readings and signatures, and real
``SignatureSamplingWitnessV1`` values produced by the ordinary sampler over
those synthetic pixels. No source screenshot, truth, camera manifest, mask,
scene, world, Litematic, JAR, evaluator payload or network input is opened, and
no inference, matcher, solver or screenshot replay is run.

Acceptance gates covered here:

1. ordinary sampler roundtrip reproduces every array and the original nested and
   witness digests, including optional region data and valid border sentinels;
3. allocation traps cover oversized/aliased dtype, false shape, malformed and
   oversized base64, byte/work caps, invalid pointers/weights/indices,
   out-of-range integers and metadata expansion;
4. re-signed tampering rejects foreign/swapped identities, missing/duplicate
   entries, modified arrays and nested backend/hash mismatches;
5. exact cap boundaries, persistence refusals, immutable decoded arrays, stable
   read growth/replacement and cap-configuration validation.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.pipeline import candidate_support_provenance as csp
from mc_photogrammetry.pipeline import signature_sampling_persistence as sp
from mc_photogrammetry.reconstruction import lattice, lattice_planes
from test_candidate_support_provenance import (
    _HEX,
    _HEX_B,
    _binding,
    _border_rectification,
    _make_ingress,
    _pixels,
    _png_bytes,
    _profile_reading,
)

_RECT_ARRAYS = ("map_x", "map_y", "depth_valid", "tap_ptr",
                "tap_source_indices", "tap_weight_numerators")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persist(binding, *, limits=sp.SAMPLING_PERSISTENCE_LIMITS):
    return sp.persist_signature_witnesses([binding], limits=limits)


def _decode(section, binding, *, limits=sp.SAMPLING_PERSISTENCE_LIMITS,
            require_ingress=False):
    return sp.decode_section(
        section, limits=limits,
        ingress_by_frame={binding.frame_id: binding.ingress},
        require_ingress=require_ingress)


def _resign(section):
    section["section_sha256"] = sp.section_digest(section)
    return section


def _entry_array(section, name, *, entry=0, rectification="rgb_rectification",
                 witness_level=False):
    witness = section["entries"][entry]["witness"]
    if witness_level:
        return witness[name]
    return witness[rectification][name]


def _array_values(record):
    raw = base64.b64decode(record["b64"])
    return np.frombuffer(raw, dtype=np.dtype(record["dtype"])).copy()


def _rewrite_array(section, path, values):
    record = section
    for key in path:
        record = record[key]
    array = np.ascontiguousarray(np.asarray(values))
    raw = array.tobytes(order="C")
    record["dtype"] = array.dtype.str
    record["shape"] = list(array.shape)
    record["bytes"] = len(raw)
    record["sha256"] = hashlib.sha256(raw).hexdigest()
    record["b64"] = base64.b64encode(raw).decode("ascii")
    return record


def _trap_allocation(monkeypatch):
    """Trap every NumPy entry point the decoder could allocate through."""
    calls = {"dtype": 0, "frombuffer": 0, "asarray": 0}

    def trap(name):
        def _forbidden(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"numpy.{name} must not be called")
        return _forbidden

    for name in calls:
        monkeypatch.setattr(np, name, trap(name))
    return calls


def _region_binding(frame_id="f"):
    pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    polygon = ((4.0, 4.0), (28.0, 4.0), (28.0, 28.0), (4.0, 28.0))
    canonical = lattice_planes._canonical_region_polygon(polygon)
    evidence = lattice_planes._make_region_evidence(mask, 2, polygon, ("s1",))
    reading = lattice_planes.PlaneReading(
        frame=frame_id, axis=2, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=8.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=17,
        polygon=canonical, coverage_cells=4, structural_share=0.5,
        confidence=0.8,
        rectification=lattice_planes.PlaneRectification(
            tuple(tuple(float(value) for value in row) for row in np.eye(3)),
            12.0, (16.0, 16.0), (0.0, 0.0)),
        region_evidence=evidence)
    sink: list = []
    lattice_planes.plane_signature(
        pixels, reading, 1.0, usable=np.ones((32, 32), dtype=bool),
        witness_sink=sink)
    witness = sink[0]
    assert witness.region_rectification is not None
    return csp.SourceTapWitnessBindingV1(
        frame_id=frame_id, canonical_index=0, alternative_index=0,
        signature_sha256=_HEX, ingress=_make_ingress(
            frame_id, pixels, _png_bytes((1, 2, 3))),
        mask_sha256=_HEX_B, masks_sha256=_HEX,
        max_cells=lattice_planes.MAX_SIGNATURE_CELLS,
        max_rectified_side=lattice_planes.MAX_RECTIFIED_SIDE,
        pixels_per_cell=lattice_planes.SIGNATURE_PIXELS_PER_CELL, witness=witness)


# ---------------------------------------------------------------------------
# Gate 1 - ordinary sampler roundtrip
# ---------------------------------------------------------------------------

def test_ordinary_sampler_roundtrip_reproduces_every_array_and_digest():
    binding = _binding("f", _pixels(7), _png_bytes((1, 2, 3)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    assert section["status"] == "complete"
    assert section["entry_count"] == 1
    assert section["entries"][0]["outcome"] == "bound"
    decoded = _decode(section, binding)

    assert decoded.state == sp.SAMPLING_PERSISTENCE_STATE_AVAILABLE
    assert decoded.bound_count == 1 and decoded.persistence_refused_count == 0
    entry = decoded.entries[0]
    original = binding.witness
    rebuilt = entry.witness
    assert rebuilt is not None
    # Every nested digest is the sampler's own, recomputed from the persisted
    # metadata and raw bytes.
    assert rebuilt.witness_sha256 == original.witness_sha256
    assert rebuilt.cell_rectangles == original.cell_rectangles
    assert rebuilt.interior_rectangles == original.interior_rectangles
    assert rebuilt.grayscale_sha256 == original.grayscale_sha256
    assert rebuilt.gradient_x_sha256 == original.gradient_x_sha256
    assert rebuilt.gradient_y_sha256 == original.gradient_y_sha256
    assert rebuilt.energy_sha256 == original.energy_sha256
    assert rebuilt.coverage_patch_sha256 == original.coverage_patch_sha256
    np.testing.assert_array_equal(rebuilt.cell_source_ptr,
                                  original.cell_source_ptr)
    np.testing.assert_array_equal(rebuilt.cell_source_indices,
                                  original.cell_source_indices)
    assert rebuilt.region_rectification is None
    for name in ("rgb_rectification", "coverage_rectification"):
        left = getattr(rebuilt, name)
        right = getattr(original, name)
        assert left.witness_sha256 == right.witness_sha256
        assert left.opencv_version == right.opencv_version
        assert left.opencv_build_sha256 == right.opencv_build_sha256
        assert left.source_shape == right.source_shape
        for field in _RECT_ARRAYS:
            np.testing.assert_array_equal(getattr(left, field),
                                          getattr(right, field))
    # The full binding is rebuilt and its canonical record is unchanged.
    assert entry.binding is not None
    assert entry.binding.to_record() == binding.to_record()


def test_valid_border_sentinel_tap_roundtrips_exactly():
    pixels = _pixels(11)
    binding = _binding("f", pixels, _png_bytes((2, 3, 4)),
                       signature_sha256=_HEX)
    sentinel = _border_rectification(pixels, -1)
    assert int(sentinel.tap_source_indices.min()) == -1
    witness = replace(binding.witness, rgb_rectification=sentinel,
                      witness_sha256="")
    rebound = replace(binding, witness=witness, witness_sha256="")
    section = _persist(rebound)
    decoded = _decode(section, rebound)
    rebuilt = decoded.entries[0].witness
    assert int(rebuilt.rgb_rectification.tap_source_indices.min()) == -1
    np.testing.assert_array_equal(rebuilt.rgb_rectification.tap_source_indices,
                                  sentinel.tap_source_indices)
    assert rebuilt.witness_sha256 == rebound.witness.witness_sha256


def test_optional_region_rectification_roundtrips():
    binding = _region_binding()
    original = binding.witness.region_rectification
    assert original is not None
    section = _persist(binding)
    decoded = _decode(section, binding)
    rebuilt = decoded.entries[0].witness
    assert rebuilt.region_rectification is not None
    assert rebuilt.region_rectification.witness_sha256 == original.witness_sha256
    for field in _RECT_ARRAYS:
        np.testing.assert_array_equal(
            getattr(rebuilt.region_rectification, field),
            getattr(original, field))
    assert rebuilt.witness_sha256 == binding.witness.witness_sha256


def test_ordered_refusal_outcome_is_preserved_and_cross_binds():
    binding = _binding("f", _pixels(13), _png_bytes((1, 1, 1)),
                       signature_sha256=_HEX)
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason="source_tap_witness_unavailable", detail="witness_absent")
    section = sp.persist_signature_witnesses([binding, refusal])
    assert [entry["outcome"] for entry in section["entries"]] == [
        "bound", "refused"]
    summaries = [dict(binding.to_record(), outcome="bound"),
                 {"outcome": "refused", "frame_id": "g", "canonical_index": 1,
                  "alternative_index": 2,
                  "reason": "source_tap_witness_unavailable",
                  "detail": "witness_absent"}]
    sp.cross_bind_witness_summaries(section, summaries)
    decoded = _decode(section, binding)
    assert decoded.bound_count == 1 and decoded.refused_count == 1
    assert decoded.entries[1].reason == "source_tap_witness_unavailable"


# ---------------------------------------------------------------------------
# Gate 3 - allocation traps (plain validation refuses before NumPy)
# ---------------------------------------------------------------------------

def _adversarial_section(binding):
    section = _persist(binding)
    return section


def test_oversized_string_dtype_is_refused_before_numpy(tmp_path, monkeypatch):
    binding = _binding("f", _pixels(21), _png_bytes((4, 4, 4)),
                       signature_sha256=_HEX)
    section = _adversarial_section(binding)
    _entry_array(section, "map_x")["dtype"] = "U1048576"
    _resign(section)
    calls = _trap_allocation(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_DTYPE_INVALID
    assert calls == {"dtype": 0, "frombuffer": 0, "asarray": 0}


@pytest.mark.parametrize("dtype", ["i8", "int64", "=i8", ">i8", "<f8", "float32",
                                   "|u1", "<u4", "bool"])
def test_aliased_or_foreign_dtype_is_refused_before_numpy(dtype, monkeypatch):
    binding = _binding("f", _pixels(22), _png_bytes((4, 4, 5)),
                       signature_sha256=_HEX)
    section = _adversarial_section(binding)
    _entry_array(section, "map_x")["dtype"] = dtype
    _resign(section)
    calls = _trap_allocation(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_DTYPE_INVALID
    assert calls == {"dtype": 0, "frombuffer": 0, "asarray": 0}


def test_false_shape_is_refused_before_numpy(monkeypatch):
    binding = _binding("f", _pixels(23), _png_bytes((4, 4, 6)),
                       signature_sha256=_HEX)
    section = _adversarial_section(binding)
    array = _entry_array(section, "map_x")
    array["shape"] = [10 ** 9, 4]
    _resign(section)
    calls = _trap_allocation(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_ARRAY_OVER_CAP
    assert calls == {"dtype": 0, "frombuffer": 0, "asarray": 0}


def test_declared_byte_length_mismatch_is_refused_before_numpy(monkeypatch):
    binding = _binding("f", _pixels(24), _png_bytes((4, 4, 7)),
                       signature_sha256=_HEX)
    section = _adversarial_section(binding)
    array = _entry_array(section, "map_x")
    array["bytes"] = int(array["bytes"]) + 1
    _resign(section)
    calls = _trap_allocation(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_ARRAY_OVER_CAP
    assert calls["frombuffer"] == 0


def test_malformed_and_oversized_base64_are_refused_before_numpy(monkeypatch):
    binding = _binding("f", _pixels(25), _png_bytes((4, 4, 8)),
                       signature_sha256=_HEX)
    # Only the canonical base64 length is accepted; a truncated or padded
    # encoding is refused on the plain declared bytes, before any decode.
    for transform in (
            lambda value: value[:-4],
            lambda value: value + "AAAA",
            lambda value: value[2:]):
        section = _adversarial_section(binding)
        array = _entry_array(section, "map_x")
        array["b64"] = transform(array["b64"])
        _resign(section)
        calls = _trap_allocation(monkeypatch)
        with pytest.raises(sp.SamplingPersistenceError) as error:
            sp.validate_section_plain(section)
        assert error.value.reason == sp.REASON_PERSISTENCE_ARRAY_OVER_CAP
        assert calls == {"dtype": 0, "frombuffer": 0, "asarray": 0}
        monkeypatch.undo()


def test_decode_refuses_a_corrupt_base64_body_before_allocation(monkeypatch):
    binding = _binding("f", _pixels(26), _png_bytes((4, 4, 9)),
                       signature_sha256=_HEX)
    section = _adversarial_section(binding)
    array = _entry_array(section, "map_x")
    # Canonical length, valid alphabet, but the raw body no longer matches the
    # declared SHA-256: the plain pass accepts the length, the decode pass
    # rejects the hash.
    array["b64"] = "A" * len(array["b64"])
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(section)
    assert error.value.reason in (sp.REASON_PERSISTENCE_ARRAY_OVER_CAP,
                                  sp.REASON_PERSISTENCE_HASH_MISMATCH)


def test_invalid_tap_pointer_index_and_weight_are_refused():
    binding = _binding("f", _pixels(27), _png_bytes((4, 4, 10)),
                       signature_sha256=_HEX)
    original = _entry_array(_persist(binding), "tap_ptr")
    values = _array_values(original)

    def _section_with(name, rewritten):
        section = _persist(binding)
        _rewrite_array(section,
                       ["entries", 0, "witness", "rgb_rectification", name],
                       rewritten)
        return _resign(section)

    # A pointer table that no longer starts at zero.
    broken = values.copy()
    broken[0] = 1
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(_section_with("tap_ptr", broken))
    assert error.value.reason == sp.REASON_PERSISTENCE_INDEX_INVALID

    # A non-monotonic pointer table.
    broken = values.copy()
    if broken.size > 2:
        broken[1] = int(broken[2]) + 1
        with pytest.raises(sp.SamplingPersistenceError) as error:
            sp.decode_section(_section_with("tap_ptr", broken))
        assert error.value.reason == sp.REASON_PERSISTENCE_INDEX_INVALID

    # A source index beyond the decoded frame.
    indices = _array_values(_entry_array(_persist(binding),
                                         "tap_source_indices"))
    indices[0] = 2 ** 40
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(_section_with("tap_source_indices", indices))
    assert error.value.reason == sp.REASON_PERSISTENCE_INDEX_INVALID

    # A zero and an over-denominator bilinear weight.
    weights = _array_values(_entry_array(_persist(binding),
                                         "tap_weight_numerators"))
    for bad in (0, 1025):
        weights[0] = bad
        with pytest.raises(sp.SamplingPersistenceError) as error:
            sp.decode_section(_section_with("tap_weight_numerators", weights))
        assert error.value.reason == sp.REASON_PERSISTENCE_WEIGHT_INVALID


def test_out_of_range_rectangle_integer_is_refused():
    binding = _binding("f", _pixels(28), _png_bytes((4, 4, 11)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    section["entries"][0]["witness"]["cell_rectangles"][0] = [
        0, 2 ** 40, 0, 1]
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_RECTANGLE_INVALID


def test_rectangle_count_expansion_is_refused():
    binding = _binding("f", _pixels(29), _png_bytes((4, 4, 12)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    witness = section["entries"][0]["witness"]
    huge = sp.SAMPLING_PERSISTENCE_LIMITS.max_cells_per_side ** 2 + 1
    witness["cell_rectangles"] = [[0, 1, 0, 1]] * huge
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_RECTANGLE_INVALID


def test_metadata_expansion_is_a_typed_persistence_refusal():
    binding = _binding("f", _pixels(30), _png_bytes((4, 4, 13)),
                       signature_sha256=_HEX)
    witness = binding.witness
    count = 20_000
    rectangles = tuple((0, 1, 0, 1) for _ in range(count))
    interiors = tuple((0, 1, 0, 1) for _ in range(count))
    pointer = np.asarray([0] + [len(witness.cell_source_indices)] * count,
                         dtype=np.int64)
    pointer.flags.writeable = False
    bloated = replace(
        witness, cell_rectangles=rectangles, interior_rectangles=interiors,
        cell_source_ptr=pointer, witness_sha256="")
    rebound = replace(binding, witness=bloated, witness_sha256="")
    section = sp.persist_signature_witnesses([rebound])
    entry = section["entries"][0]
    assert entry["outcome"] == "persistence_refused"
    assert entry["reason"] == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    # The original bound witness is untouched: the persisted identity still
    # carries its own nested digest.
    assert entry["witness_sha256"] == bloated.witness_sha256
    assert binding.witness.witness_sha256 != bloated.witness_sha256
    decoded = sp.decode_section(section)
    assert decoded.persistence_refused_count == 1
    assert decoded.state == sp.SAMPLING_PERSISTENCE_STATE_UNAVAILABLE


def test_byte_and_work_caps_are_typed_persistence_refusals():
    binding = _binding("f", _pixels(31), _png_bytes((4, 4, 14)),
                       signature_sha256=_HEX)
    account = sp._account_witness(binding.witness,
                                  limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    capped = replace(sp.SAMPLING_PERSISTENCE_LIMITS,
                     max_entry_payload_bytes=1, max_total_payload_bytes=1)
    section = sp.persist_signature_witnesses([binding], limits=capped)
    entry = section["entries"][0]
    assert entry["outcome"] == "persistence_refused"
    assert entry["reason"] == sp.REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP
    assert entry["witness_sha256"] == binding.witness.witness_sha256

    work_capped = replace(sp.SAMPLING_PERSISTENCE_LIMITS, max_work_units=1)
    assert account.work_units > 1
    work_section = sp.persist_signature_witnesses([binding], limits=work_capped)
    assert work_section["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert work_section["reason"] == sp.REASON_PERSISTENCE_WORK_OVER_CAP
    assert work_section["entries"] == []


# ---------------------------------------------------------------------------
# Gate 4 - re-signed tampering
# ---------------------------------------------------------------------------

def test_modified_array_is_refused_even_when_resigned():
    binding = _binding("f", _pixels(41), _png_bytes((5, 5, 5)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    path = ["entries", 0, "witness", "rgb_rectification", "map_x"]
    values = _array_values(_entry_array(section, "map_x"))
    values[0] = np.float32(values[0] + np.float32(1.0))
    _rewrite_array(section, path, values)
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError):
        sp.decode_section(section)


@pytest.mark.parametrize("field", ["signature_sha256", "ingress_sha256",
                                   "mask_sha256", "masks_sha256",
                                   "witness_sha256"])
def test_foreign_or_swapped_identity_is_refused_by_cross_bind(field):
    binding = _binding("f", _pixels(42), _png_bytes((5, 5, 6)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    summaries = [dict(binding.to_record(), outcome="bound")]
    sp.cross_bind_witness_summaries(section, summaries)
    section["entries"][0][field] = "c" * 64
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.cross_bind_witness_summaries(section, summaries)
    assert error.value.reason == sp.REASON_PERSISTENCE_SUMMARY_MISMATCH


def test_swapped_witness_between_entries_is_refused():
    first = _binding("f", _pixels(43), _png_bytes((5, 5, 7)),
                     signature_sha256=_HEX)
    second = _binding("g", _pixels(44), _png_bytes((5, 5, 8)),
                      signature_sha256=_HEX_B)
    section = sp.persist_signature_witnesses([first, second])
    section["entries"][0]["witness"], section["entries"][1]["witness"] = (
        section["entries"][1]["witness"], section["entries"][0]["witness"])
    _resign(section)
    # The ordered summary cross-bind still matches (only the payloads moved), so
    # the typed decode is what refuses: the rebuilt witness digest no longer
    # matches the entry's own persisted ``witness_sha256``.
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_DIGEST_MISMATCH


def test_swapped_identity_between_entries_is_refused_by_cross_bind():
    first = _binding("f", _pixels(143), _png_bytes((15, 5, 7)),
                     signature_sha256=_HEX)
    second = _binding("g", _pixels(144), _png_bytes((15, 5, 8)),
                      signature_sha256=_HEX_B)
    section = sp.persist_signature_witnesses([first, second])
    for name in ("frame_id", "canonical_index", "alternative_index",
                 "signature_sha256", "witness_sha256"):
        section["entries"][0][name], section["entries"][1][name] = (
            section["entries"][1][name], section["entries"][0][name])
    _resign(section)
    summaries = [dict(first.to_record(), outcome="bound"),
                 dict(second.to_record(), outcome="bound")]
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.cross_bind_witness_summaries(section, summaries)
    assert error.value.reason == sp.REASON_PERSISTENCE_SUMMARY_MISMATCH


def test_missing_and_duplicate_entries_are_refused():
    first = _binding("f", _pixels(45), _png_bytes((5, 5, 9)),
                     signature_sha256=_HEX)
    second = _binding("g", _pixels(46), _png_bytes((5, 5, 10)),
                      signature_sha256=_HEX_B)
    section = sp.persist_signature_witnesses([first, second])
    summaries = [dict(first.to_record(), outcome="bound"),
                 dict(second.to_record(), outcome="bound")]

    # A valid one-entry section against a two-entry summary: the missing entry
    # is refused rather than silently ignored.
    missing = sp.persist_signature_witnesses([first])
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.cross_bind_witness_summaries(missing, summaries)
    assert error.value.reason == sp.REASON_PERSISTENCE_SUMMARY_MISMATCH

    duplicate = copy.deepcopy(section)
    duplicate["entries"][1] = copy.deepcopy(duplicate["entries"][0])
    duplicate["entries"][1]["ordinal"] = 1
    _resign(duplicate)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(duplicate)
    assert error.value.reason == sp.REASON_PERSISTENCE_DUPLICATE_ENTRY


def test_nested_backend_and_hash_mismatch_are_refused():
    binding = _binding("f", _pixels(47), _png_bytes((5, 5, 11)),
                       signature_sha256=_HEX)

    section = _persist(binding)
    section["entries"][0]["witness"]["rgb_rectification"][
        "opencv_build_sha256"] = "c" * 64
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_SCHEMA_INVALID

    section = _persist(binding)
    section["entries"][0]["witness"]["grayscale_sha256"] = "c" * 64
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_SCHEMA_INVALID


def test_tampered_section_digest_and_policy_are_refused():
    binding = _binding("f", _pixels(48), _png_bytes((5, 5, 12)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    stale = copy.deepcopy(section)
    stale["entries"][0]["signature_sha256"] = "c" * 64
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(stale)
    assert error.value.reason == sp.REASON_PERSISTENCE_DIGEST_MISMATCH

    foreign = copy.deepcopy(section)
    foreign["policy"]["max_entries"] = 2
    _resign(foreign)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(foreign)
    assert error.value.reason == sp.REASON_PERSISTENCE_POLICY_MISMATCH


def test_ingress_pin_mismatch_is_refused_when_a_pin_exists():
    binding = _binding("f", _pixels(49), _png_bytes((5, 5, 13)),
                       signature_sha256=_HEX)
    other = _binding("f", _pixels(50), _png_bytes((6, 6, 6)),
                     signature_sha256=_HEX)
    section = _persist(binding)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(section,
                          ingress_by_frame={"f": other.ingress},
                          require_ingress=True)
    assert error.value.reason == sp.REASON_PERSISTENCE_INGRESS_MISMATCH


# ---------------------------------------------------------------------------
# Gate 5 - cap boundaries, immutability, stable read, configuration
# ---------------------------------------------------------------------------

def test_exact_entry_payload_cap_boundary_binds_then_refuses():
    binding = _binding("f", _pixels(51), _png_bytes((7, 7, 7)),
                       signature_sha256=_HEX)
    account = sp._account_witness(binding.witness,
                                  limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    # The entry-level metadata budget is the identity fields plus the whole
    # witness metadata (the refusal/identity fields are no longer exempt).
    entry_metadata = sp._entry_metadata_bytes({
        "outcome": sp.SAMPLING_PERSISTENCE_OUTCOME_BOUND,
        "ordinal": 0,
        "schema": sp.SAMPLING_PERSISTENCE_ENTRY_SCHEMA,
        **sp._binding_identity(binding),
    }, witness_metadata=sp._witness_metadata(binding.witness, digest=False))
    assert entry_metadata >= account.metadata_bytes
    exact = replace(
        sp.SAMPLING_PERSISTENCE_LIMITS,
        max_entry_payload_bytes=account.payload_bytes,
        max_total_payload_bytes=account.payload_bytes,
        max_entry_metadata_bytes=entry_metadata,
        max_total_metadata_bytes=entry_metadata)
    section = sp.persist_signature_witnesses([binding], limits=exact)
    assert section["entries"][0]["outcome"] == "bound"
    assert section["payload_bytes"] == account.payload_bytes
    assert section["metadata_bytes"] == entry_metadata
    assert sp.validate_section_plain(section, limits=exact).metadata_bytes == (
        entry_metadata)

    under = replace(exact,
                    max_entry_payload_bytes=account.payload_bytes - 1,
                    max_total_payload_bytes=account.payload_bytes - 1)
    refused = sp.persist_signature_witnesses([binding], limits=under)
    assert refused["entries"][0]["outcome"] == "persistence_refused"
    assert refused["entries"][0]["reason"] == (
        sp.REASON_PERSISTENCE_ENTRY_PAYLOAD_OVER_CAP)

    # An identity-inclusive metadata boundary: one byte below the exact entry
    # metadata refuses the entry's metadata, one byte below the witness metadata
    # alone still binds because the identity is inside the same budget.
    under_metadata = replace(
        exact, max_entry_metadata_bytes=entry_metadata - 1,
        max_total_metadata_bytes=entry_metadata - 1)
    refused_meta = sp.persist_signature_witnesses([binding], limits=under_metadata)
    assert refused_meta["entries"][0]["outcome"] == "persistence_refused"
    assert refused_meta["entries"][0]["reason"] == (
        sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP)
    assert refused_meta["metadata_bytes"] > 0


def test_decoded_arrays_are_immutable():
    binding = _binding("f", _pixels(52), _png_bytes((7, 7, 8)),
                       signature_sha256=_HEX)
    decoded = _decode(_persist(binding), binding)
    witness = decoded.entries[0].witness
    assert witness.cell_source_ptr.flags.writeable is False
    assert witness.cell_source_indices.flags.writeable is False
    assert witness.rgb_rectification.map_x.flags.writeable is False
    for array in (witness.cell_source_ptr, witness.cell_source_indices,
                  witness.rgb_rectification.map_x,
                  witness.rgb_rectification.tap_weight_numerators):
        with pytest.raises(ValueError):
            array.reshape(-1)[0] = array.reshape(-1)[0]


def test_cap_configuration_validation():
    with pytest.raises(ValueError):
        sp.SamplingPersistenceLimitsV1(max_entries=0)
    with pytest.raises(ValueError):
        sp.SamplingPersistenceLimitsV1(max_entry_payload_bytes=100,
                                       max_total_payload_bytes=10)
    with pytest.raises(ValueError):
        sp.SamplingPersistenceLimitsV1(max_entry_metadata_bytes=100,
                                       max_total_metadata_bytes=10)
    with pytest.raises(ValueError):
        sp.SamplingPersistenceLimitsV1(max_cells_per_side=4096)
    with pytest.raises(ValueError):
        sp.SamplingPersistenceLimitsV1(max_depth=1)
    assert sp.SamplingPersistenceLimitsV1.from_record(
        sp.SAMPLING_PERSISTENCE_LIMITS.to_record()) == (
            sp.SAMPLING_PERSISTENCE_LIMITS)


def test_default_reader_never_uses_unbounded_read_bytes(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_bytes(b'{"a":[1,2,3]}')

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("unbounded Path.read_bytes must not be used")

    monkeypatch.setattr(Path, "read_bytes", _forbidden)
    snapshot = sp.read_bounded_record_bytes(target)
    assert snapshot.data == b'{"a":[1,2,3]}'
    assert snapshot.byte_length == len(b'{"a":[1,2,3]}')
    assert snapshot.max_json_depth == 2


def test_bounded_stream_read_is_capped_at_max_bytes_plus_one():
    payload = b"a" * 4096
    assert sp._read_stream_bounded(io.BytesIO(payload), 64) == payload[:65]
    assert sp._read_stream_bounded(io.BytesIO(payload), 8192) == payload


def test_reader_refuses_simulated_growth_without_retaining_bytes(tmp_path,
                                                                 monkeypatch):
    cap = 64
    target = tmp_path / "record.json"
    target.write_bytes(b'{"a":1}')
    limits = replace(
        sp.SAMPLING_PERSISTENCE_LIMITS, max_serialized_bytes=cap,
        max_entry_payload_bytes=16, max_total_payload_bytes=16,
        max_entry_metadata_bytes=16, max_total_metadata_bytes=16)
    real_read = sp.os.read
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

    monkeypatch.setattr(sp.os, "read", _growing_read)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.read_bounded_record_bytes(target, limits)
    assert error.value.reason == sp.REASON_PERSISTENCE_INPUT_OVER_CAP
    assert pulled["bytes"] <= cap + 1


def test_reader_refuses_simulated_replacement(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_bytes(b'{"a":1}')
    real_fstat = sp.os.fstat
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

    monkeypatch.setattr(sp.os, "fstat", _replacing_fstat)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.read_bounded_record_bytes(target)
    assert error.value.reason == sp.REASON_PERSISTENCE_INPUT_SIZE_DRIFT


def test_reader_depth_and_token_caps_refuse_before_json_loads(tmp_path):
    target = tmp_path / "record.json"
    target.write_bytes(json.dumps({"a": [[[[[1]]]]]}).encode("utf-8"))
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS, max_depth=4)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.read_bounded_record_bytes(target, limits)
    assert error.value.reason == sp.REASON_PERSISTENCE_DEPTH_OVER_CAP

    tokens = replace(sp.SAMPLING_PERSISTENCE_LIMITS, max_tokens=1)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.read_bounded_record_bytes(target, tokens)
    assert error.value.reason == sp.REASON_PERSISTENCE_TOKEN_OVER_CAP


def test_refused_section_record_is_small_typed_and_decodable():
    section = sp.refused_section_record(
        sp.REASON_PERSISTENCE_SERIALIZED_OVER_CAP, field="record")
    accounting = sp.validate_section_plain(section)
    assert accounting.entry_count == 0
    decoded = sp.decode_section(section)
    assert decoded.status == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert decoded.state == sp.SAMPLING_PERSISTENCE_STATE_UNAVAILABLE
    # The section stays tiny and never carries a payload.
    assert len(sp._canonical_bytes(section)) < 2048


# ---------------------------------------------------------------------------
# P1#1 - the source-image shape/pixel cap is separate from the rectified cap
# ---------------------------------------------------------------------------

def test_known_t2_source_shape_roundtrips_with_a_tiny_rectification():
    """The known 2560x1369 T2 ingress width exceeds the rectified 2048 cap."""
    limits = sp.SAMPLING_PERSISTENCE_LIMITS
    assert 2560 > limits.max_rectified_side
    assert 2560 <= limits.max_source_side
    pixels = np.full((1369, 2560, 3), 97, dtype=np.uint8)
    binding = _binding("f", pixels, _png_bytes((1, 2, 3)),
                       signature_sha256=_HEX)
    assert binding.ingress.source_shape == (1369, 2560)
    assert binding.witness.rgb_rectification.source_shape == (1369, 2560)
    section = _persist(binding)
    assert section["status"] == "complete"
    assert section["entries"][0]["outcome"] == "bound"
    decoded = _decode(section, binding)
    assert decoded.state == sp.SAMPLING_PERSISTENCE_STATE_AVAILABLE
    rebuilt = decoded.entries[0].witness
    assert rebuilt.rgb_rectification.source_shape == (1369, 2560)
    assert rebuilt.coverage_rectification.source_shape == (1369, 2560)
    # The rectified patch itself is tiny, far below the rectified-side cap.
    assert rebuilt.rgb_rectification.map_x.shape[0] < limits.max_rectified_side


def test_source_shape_cap_boundaries_are_independent_of_the_rectified_side():
    binding = _binding("f", _pixels(53), _png_bytes((7, 7, 9)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    record = copy.deepcopy(
        section["entries"][0]["witness"]["rgb_rectification"])
    limits = sp.SAMPLING_PERSISTENCE_LIMITS
    for shape, accepted in (((1369, 2560), True), ((4096, 4096), True),
                            ((4097, 1), False), ((1, 4097), False),
                            ((4096, 4097), False)):
        candidate = copy.deepcopy(record)
        candidate["source_shape"] = list(shape)
        if accepted:
            total, arrays = sp._validate_rectification_plain(
                candidate, field="rgb_rectification", limits=limits,
                remaining=limits.max_entry_payload_bytes)
            assert arrays == 1 + len(_RECT_ARRAYS) and total > 0
        else:
            with pytest.raises(sp.SamplingPersistenceError) as error:
                sp._validate_rectification_plain(
                    candidate, field="rgb_rectification", limits=limits,
                    remaining=limits.max_entry_payload_bytes)
            assert error.value.reason == sp.REASON_PERSISTENCE_RECTANGLE_INVALID


def _plain_rectification_digest(record):
    body = {key: value for key, value in record.items() if key != "witness_sha256"}
    for name in _RECT_ARRAYS:
        array = body[name]
        body[name] = {"dtype": array["dtype"], "shape": array["shape"],
                      "sha256": array["sha256"]}
    return lattice._witness_sha(body)


def _plain_witness_digest(witness):
    def row(record):
        return {"dtype": record["dtype"], "shape": record["shape"],
                "sha256": record["sha256"]}
    body = {
        "schema": witness["schema"],
        "rgb_rectification_sha256":
            witness["rgb_rectification"]["witness_sha256"],
        "coverage_rectification_sha256":
            witness["coverage_rectification"]["witness_sha256"],
        "region_rectification_sha256": (
            None if witness["region_rectification"] is None
            else witness["region_rectification"]["witness_sha256"]),
        "grayscale_sha256": witness["grayscale_sha256"],
        "gradient_y_sha256": witness["gradient_y_sha256"],
        "gradient_x_sha256": witness["gradient_x_sha256"],
        "energy_sha256": witness["energy_sha256"],
        "coverage_patch_sha256": witness["coverage_patch_sha256"],
        "gradient_edge_order": witness["gradient_edge_order"],
        "subsample_factor": witness["subsample_factor"],
        "seam_band": witness["seam_band"],
        "cell_rectangles": witness["cell_rectangles"],
        "interior_rectangles": witness["interior_rectangles"],
        "cell_source_ptr": row(witness["cell_source_ptr"]),
        "cell_source_indices": row(witness["cell_source_indices"]),
    }
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def test_witness_source_shape_is_bound_to_the_root_pin_on_decode():
    binding = _binding("f", _pixels(54), _png_bytes((7, 7, 10)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    # A fully re-signed witness whose rectification source shape is not the
    # pinned frame shape (but still within the standalone source caps, and large
    # enough that every stored tap index remains in range) must refuse at the
    # binding, because the binding re-checks the rectification source shape
    # against the record's own root ingress pin.
    candidate = copy.deepcopy(section)
    witness = candidate["entries"][0]["witness"]
    for field in ("rgb_rectification", "coverage_rectification"):
        witness[field]["source_shape"] = [64, 64]
        witness[field]["witness_sha256"] = _plain_rectification_digest(
            witness[field])
    witness["witness_sha256"] = _plain_witness_digest(witness)
    candidate["entries"][0]["witness_sha256"] = witness["witness_sha256"]
    _resign(candidate)
    sp.validate_section_plain(candidate)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(candidate,
                          ingress_by_frame={"f": binding.ingress})
    assert error.value.reason == sp.REASON_PERSISTENCE_INGRESS_MISMATCH
    assert error.value.field.endswith(".source_shape")


# ---------------------------------------------------------------------------
# P1#2 / P2#7 - declared caps and precharged work precede every traversal
# ---------------------------------------------------------------------------

def _oversized_noncontiguous_witness(binding):
    """A witness whose declared index array is oversized but never materialized."""
    witness = binding.witness
    count = 10_000_000
    pointer = np.asarray([0] + [count] * len(witness.cell_rectangles),
                         dtype=np.int64)
    pointer.flags.writeable = False
    indices = np.broadcast_to(np.zeros(1, dtype=np.int64), (count,))
    assert indices.flags.writeable is False
    assert indices.nbytes == count * 8
    # ``replace`` would re-run the binding's own value scans over 10M elements;
    # the frozen object is mutated directly so this test measures persistence.
    object.__setattr__(binding, "witness", replace(
        witness, cell_source_ptr=pointer, cell_source_indices=indices,
        witness_sha256=""))
    return binding


def test_oversized_noncontiguous_array_is_refused_before_any_copy(monkeypatch):
    binding = _oversized_noncontiguous_witness(
        _binding("f", _pixels(55), _png_bytes((7, 7, 11)),
                 signature_sha256=_HEX))
    real_contiguous = np.ascontiguousarray
    copied = {"oversized": 0}

    def guarded(array, *args, **kwargs):
        if getattr(array, "nbytes", 0) > sp.SAMPLING_PERSISTENCE_LIMITS.max_entry_payload_bytes:
            copied["oversized"] += 1
            raise AssertionError("an oversized array must not be made contiguous")
        return real_contiguous(array, *args, **kwargs)

    def forbidden_diff(*_args, **_kwargs):
        raise AssertionError("no pointer traversal may run before the caps")

    monkeypatch.setattr(sp.np, "ascontiguousarray", guarded)
    monkeypatch.setattr(sp.np, "diff", forbidden_diff)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp._account_witness(binding.witness, limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    assert error.value.reason == sp.REASON_PERSISTENCE_ARRAY_OVER_CAP
    assert copied["oversized"] == 0


def test_work_cap_is_refused_before_any_value_traversal(monkeypatch):
    binding = _binding("f", _pixels(56), _png_bytes((7, 7, 12)),
                       signature_sha256=_HEX)
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS, max_work_units=1)
    account = sp._account_witness(binding.witness,
                                  limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    # Work is charged on declared payload and metadata bytes before any walk.
    assert account.work_units >= account.payload_bytes

    def forbidden(*_args, **_kwargs):
        raise AssertionError("value traversal must not precede the work gate")

    for name in ("_validate_pointer_array", "_validate_index_array",
                 "_validate_weight_array", "_validate_rectification_values",
                 "_validate_bilinear_tap_semantics", "_encode_witness"):
        monkeypatch.setattr(sp, name, forbidden)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp._account_witness(binding.witness, limits=limits)
    assert error.value.reason == sp.REASON_PERSISTENCE_WORK_OVER_CAP
    section = sp.persist_signature_witnesses([binding], limits=limits)
    assert section["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert section["reason"] == sp.REASON_PERSISTENCE_WORK_OVER_CAP


def test_rectangle_expansion_refuses_before_metadata_materialization(monkeypatch):
    binding = _binding("f", _pixels(57), _png_bytes((7, 7, 13)),
                       signature_sha256=_HEX)
    witness = binding.witness
    count = 20_000
    rectangles = tuple((0, 1, 0, 1) for _ in range(count))
    interiors = tuple((0, 1, 0, 1) for _ in range(count))
    pointer = np.asarray([0] + [len(witness.cell_source_indices)] * count,
                         dtype=np.int64)
    pointer.flags.writeable = False
    bloated = replace(witness, cell_rectangles=rectangles,
                      interior_rectangles=interiors, cell_source_ptr=pointer,
                      witness_sha256="")
    calls = {"serialized": 0}

    def forbidden(payload):
        calls["serialized"] += 1
        raise AssertionError("metadata must not be serialized before its cap")

    monkeypatch.setattr(sp, "_canonical_bytes", forbidden)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp._account_witness(bloated, limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    assert error.value.reason == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    assert calls["serialized"] == 0


# ---------------------------------------------------------------------------
# P1#3 - caps and scalar checks precede the section digest
# ---------------------------------------------------------------------------

def _trap_section_serialization(monkeypatch):
    calls = {"digest": 0, "serialized": 0}

    def forbidden_digest(section):
        calls["digest"] += 1
        raise AssertionError("section_digest must not run before the caps")

    def forbidden_bytes(payload):
        calls["serialized"] += 1
        raise AssertionError("canonical serialization must not precede the caps")

    monkeypatch.setattr(sp, "section_digest", forbidden_digest)
    monkeypatch.setattr(sp, "_canonical_bytes", forbidden_bytes)
    return calls


def test_over_cap_refusal_detail_is_refused_before_any_serialization(monkeypatch):
    binding = _binding("f", _pixels(58), _png_bytes((7, 7, 14)),
                       signature_sha256=_HEX)
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE,
        detail="x" * (1024 * 1024))
    section = sp.persist_signature_witnesses([binding, refusal])
    section["entries"][1]["detail"] = "x" * (1024 * 1024)
    calls = _trap_section_serialization(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    assert calls == {"digest": 0, "serialized": 0}


def test_over_cap_rectangle_count_is_refused_before_any_serialization(monkeypatch):
    binding = _binding("f", _pixels(59), _png_bytes((7, 7, 15)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    witness = section["entries"][0]["witness"]
    count = 20_000
    indices = _array_values(witness["cell_source_indices"])
    _rewrite_array(section,
                   ["entries", 0, "witness", "cell_source_ptr"],
                   np.asarray([0] + [len(indices)] * count, dtype=np.int64))
    witness["cell_rectangles"] = [[0, 1, 0, 1]] * count
    witness["interior_rectangles"] = [[0, 1, 0, 1]] * count
    calls = _trap_section_serialization(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    assert calls == {"digest": 0, "serialized": 0}


# ---------------------------------------------------------------------------
# P1#4 - refusal identity/detail metadata is inside the metadata budget
# ---------------------------------------------------------------------------

def test_over_cap_refusal_detail_degrades_to_a_small_typed_record():
    binding = _binding("f", _pixels(60), _png_bytes((7, 7, 16)),
                       signature_sha256=_HEX)
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE,
        detail="x" * (1024 * 1024))
    section = sp.persist_signature_witnesses([binding, refusal])
    assert [entry["outcome"] for entry in section["entries"]] == [
        "bound", "persistence_refused"]
    degraded = section["entries"][1]
    assert degraded["reason"] == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    assert degraded["frame_id"] == "g" and degraded["canonical_index"] == 1
    assert degraded["detail"] == ""
    assert len(sp._canonical_bytes(section)) < 64 * 1024
    assert section["metadata_bytes"] < 16 * 1024
    accounting = sp.validate_section_plain(section)
    assert accounting.refused_count == 0
    assert accounting.persistence_refused_count == 1
    # The original refusal is still authoritative in the record's ordered
    # summary, and the degraded section still cross-binds to it.
    summaries = [
        dict(binding.to_record(), outcome="bound"),
        {"outcome": "refused", "frame_id": "g", "canonical_index": 1,
         "alternative_index": 2, "reason": csp.SOURCE_TAP_WITNESS_UNAVAILABLE,
         "detail": "x" * 32}]
    sp.cross_bind_witness_summaries(section, summaries)
    decoded = _decode(section, binding)
    assert decoded.state == sp.SAMPLING_PERSISTENCE_STATE_UNAVAILABLE
    assert decoded.persistence_refused_count == 1


def test_aggregate_metadata_exhaustion_stays_a_small_typed_refusal():
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="small")
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS,
                     max_entry_metadata_bytes=64, max_total_metadata_bytes=64)
    section = sp.persist_signature_witnesses([refusal], limits=limits)
    # Even the small typed record cannot fit the aggregate metadata backstop, so
    # the whole section is the typed refusal rather than an over-cap section.
    assert section["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert section["reason"] == sp.REASON_PERSISTENCE_TOTAL_METADATA_OVER_CAP
    sp.validate_section_plain(section, limits=limits)


def test_refused_identity_metadata_is_counted_in_the_section_total():
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="d" * 4096)
    section = sp.persist_signature_witnesses([refusal])
    assert section["entries"][0]["outcome"] == "refused"
    assert section["payload_bytes"] == 0
    assert section["metadata_bytes"] > 4096
    assert section["metadata_bytes"] == sp._non_bound_entry_metadata_bytes(
        section["entries"][0], limit=sp.SAMPLING_PERSISTENCE_LIMITS.max_entry_metadata_bytes)
    accounting = sp.validate_section_plain(section)
    assert accounting.metadata_bytes == section["metadata_bytes"]


# ---------------------------------------------------------------------------
# P1#5 - root-pinned ingress is required before a section is available
# ---------------------------------------------------------------------------

def test_availability_requires_root_pinned_ingress_for_every_bound_entry():
    binding = _binding("f", _pixels(61), _png_bytes((7, 7, 17)),
                       signature_sha256=_HEX)
    other = _binding("f", _pixels(62), _png_bytes((7, 7, 18)),
                     signature_sha256=_HEX)
    section = _persist(binding)

    # No pins at all: the bound witness decodes but never binds.
    unbound = sp.decode_section(section, require_ingress=False)
    assert unbound.bound_count == 1
    assert unbound.entries[0].binding is None
    assert unbound.state == sp.SAMPLING_PERSISTENCE_STATE_UNAVAILABLE
    assert sp.section_state(section) == sp.SAMPLING_PERSISTENCE_STATE_UNAVAILABLE

    # The strict default refuses a bound entry whose frame has no supplied pin.
    for kwargs in ({}, {"ingress_by_frame": {}},
                   {"ingress_by_frame": {"other": binding.ingress}},
                   {"ingress_by_frame": {"f": other.ingress}}):
        with pytest.raises(sp.SamplingPersistenceError) as error:
            sp.decode_section(section, **kwargs)
        assert error.value.reason == sp.REASON_PERSISTENCE_INGRESS_MISMATCH

    pinned = sp.decode_section(section, ingress_by_frame={"f": binding.ingress})
    assert pinned.state == sp.SAMPLING_PERSISTENCE_STATE_AVAILABLE
    assert sp.section_state(
        section, ingress_by_frame={"f": binding.ingress}) == (
        sp.SAMPLING_PERSISTENCE_STATE_AVAILABLE)


# ---------------------------------------------------------------------------
# P1#6 - malformed identity fields are typed refusals before the dedup set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tamper", [
    ("bound", "frame_id", []),
    ("bound", "frame_id", None),
    ("bound", "canonical_index", "0"),
    ("bound", "alternative_index", -1),
    ("refused", "frame_id", []),
    ("refused", "alternative_index", None),
])
def test_malformed_identity_is_a_typed_refusal_not_a_type_error(tamper):
    binding = _binding("f", _pixels(63), _png_bytes((7, 7, 19)),
                       signature_sha256=_HEX)
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="witness_absent")
    section = sp.persist_signature_witnesses([binding, refusal])
    outcome, name, value = tamper
    entry = section["entries"][0] if outcome == "bound" else section["entries"][1]
    assert entry["outcome"] == outcome
    entry[name] = value
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_RECORD_MALFORMED


def test_malformed_persistence_refused_identity_is_typed():
    binding = _binding("f", _pixels(64), _png_bytes((7, 7, 20)),
                       signature_sha256=_HEX)
    witness = binding.witness
    count = 20_000
    pointer = np.asarray([0] + [len(witness.cell_source_indices)] * count,
                         dtype=np.int64)
    pointer.flags.writeable = False
    bloated = replace(
        witness, cell_rectangles=tuple((0, 1, 0, 1) for _ in range(count)),
        interior_rectangles=tuple((0, 1, 0, 1) for _ in range(count)),
        cell_source_ptr=pointer, witness_sha256="")
    section = sp.persist_signature_witnesses(
        [replace(binding, witness=bloated, witness_sha256="")])
    assert section["entries"][0]["outcome"] == "persistence_refused"
    section["entries"][0]["frame_id"] = []
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_RECORD_MALFORMED


# ---------------------------------------------------------------------------
# P2#8 - per-sample bilinear producer semantics
# ---------------------------------------------------------------------------

def test_individually_valid_but_wrong_per_sample_weight_total_is_refused():
    binding = _binding("f", _pixels(65), _png_bytes((7, 7, 21)),
                       signature_sha256=_HEX)
    rectification = binding.witness.rgb_rectification
    pointer = np.asarray(rectification.tap_ptr, dtype=np.int64)
    weights = np.asarray(rectification.tap_weight_numerators,
                         dtype=np.uint16).copy()
    valid_sample = int(np.argmax(np.asarray(rectification.depth_valid).reshape(-1)))
    start = int(pointer[valid_sample])
    original = int(weights[start])
    weights[start] = np.uint16(2 if original == 1 else 1)
    assert 1 <= int(weights[start]) <= 1024
    section = _persist(binding)
    _rewrite_array(section,
                   ["entries", 0, "witness", "rgb_rectification",
                    "tap_weight_numerators"], weights)
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_WEIGHT_INVALID


def _small_frame_binding(seed):
    """A real witness over a 4x4 frame, so depth-invalid samples exist.

    The axis-1 synthetic reading places most rectified samples outside a 4x4
    source frame, which exercises the documented producer contract for a
    depth-invalid sample (exactly one tap, weight 1024, index ``-1``).
    """
    frame_id = "f"
    reading = lattice_planes.PlaneReading(
        frame=frame_id, axis=1, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=8.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=17,
        polygon=None, coverage_cells=4, structural_share=0.5, confidence=0.8,
        rectification=lattice_planes.PlaneRectification(
            tuple(tuple(float(value) for value in row) for row in np.eye(3)),
            12.0, (16.0, 16.0), (0.0, 0.0)))
    pixels = _pixels(seed, shape=(4, 4, 3))
    sink: list = []
    lattice_planes.plane_signature(pixels, reading, 1.0, witness_sink=sink)
    assert len(sink) == 1
    return csp.SourceTapWitnessBindingV1(
        frame_id=frame_id, canonical_index=0, alternative_index=0,
        signature_sha256=_HEX,
        ingress=_make_ingress(frame_id, pixels, _png_bytes((7, 7, 22))),
        mask_sha256=_HEX_B, masks_sha256=_HEX,
        max_cells=lattice_planes.MAX_SIGNATURE_CELLS,
        max_rectified_side=lattice_planes.MAX_RECTIFIED_SIDE,
        pixels_per_cell=lattice_planes.SIGNATURE_PIXELS_PER_CELL,
        witness=sink[0])


def test_invalid_sample_tap_span_semantics_are_enforced():
    binding = _small_frame_binding(66)
    rectification = binding.witness.rgb_rectification
    depth = np.asarray(rectification.depth_valid).reshape(-1)
    pointer = np.asarray(rectification.tap_ptr, dtype=np.int64)
    weights = np.asarray(rectification.tap_weight_numerators,
                         dtype=np.uint16).copy()
    indices = np.asarray(rectification.tap_source_indices,
                         dtype=np.int64).copy()
    invalid = np.flatnonzero(np.logical_not(depth))
    assert invalid.size > 0
    sample = int(invalid[0])
    assert int(pointer[sample + 1]) - int(pointer[sample]) == 1
    assert int(weights[pointer[sample]]) == 1024
    assert int(indices[pointer[sample]]) == -1

    # A wrong (but individually in-range) weight on the invalid sample.
    section = _persist(binding)
    weights[pointer[sample]] = np.uint16(512)
    _rewrite_array(section,
                   ["entries", 0, "witness", "rgb_rectification",
                    "tap_weight_numerators"], weights)
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_WEIGHT_INVALID

    # A wrong (but in-range) index on the invalid sample.
    section = _persist(binding)
    indices[pointer[sample]] = 0
    _rewrite_array(section,
                   ["entries", 0, "witness", "rgb_rectification",
                    "tap_source_indices"], indices)
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.decode_section(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_INDEX_INVALID


def test_cap_configuration_requires_consistent_source_caps():
    with pytest.raises(ValueError):
        sp.SamplingPersistenceLimitsV1(max_source_side=8, max_source_pixels=1024)
    limits = sp.SamplingPersistenceLimitsV1(max_source_side=4096,
                                            max_source_pixels=4096 * 4096)
    assert limits.max_source_pixels == 4096 * 4096
    assert sp.SamplingPersistenceLimitsV1.from_record(
        limits.to_record()) == limits


def test_canonical_json_size_matches_serialization_exactly():
    payload = {
        "strings": ['a"b\\c', "line\nbreak", "\u00e9\u2603", "\U0001F600"],
        "numbers": [0, -7, 2 ** 63, 1.5, -0.0],
        "nested": {"empty": [], "none": None, "bool": [True, False]},
    }
    assert sp.canonical_json_size(payload) == len(sp._canonical_bytes(payload))
    for value in (None, True, False, 0, "", [], {}, {"a": [1, {"b": "c"}]}):
        assert sp.canonical_json_size(value) == len(sp._canonical_bytes(value))
    with pytest.raises(sp.SamplingPersistenceError):
        sp.canonical_json_size({1: "non-string-key"})


# ---------------------------------------------------------------------------
# REVISE-2 #1 - exact escaped metadata cap for refused/persistence-refused
# ---------------------------------------------------------------------------

_ESCAPED_DETAIL = "\u00e9" * 20_000


def test_escaped_refusal_detail_is_refused_before_the_section_digest(monkeypatch):
    """20,000 U+00E9 are <64 KiB raw but >64 KiB escaped, and must be refused."""
    binding = _binding("f", _pixels(67), _png_bytes((7, 7, 23)),
                       signature_sha256=_HEX)
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="witness_absent")
    section = sp.persist_signature_witnesses([binding, refusal])
    assert section["entries"][1]["outcome"] == "refused"
    cap = sp.SAMPLING_PERSISTENCE_LIMITS.max_entry_metadata_bytes
    # The raw length passes a raw-length gate while the escaped JSON size does
    # not, so only the exact escaped canonical size can enforce this cap.
    assert len(_ESCAPED_DETAIL) < cap < sp._escaped_text_size(_ESCAPED_DETAIL)
    section["entries"][1]["detail"] = _ESCAPED_DETAIL
    _resign(section)
    calls = _trap_section_serialization(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    assert calls == {"digest": 0, "serialized": 0}


def test_escaped_refusal_detail_degrades_to_a_small_typed_record():
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail=_ESCAPED_DETAIL)
    section = sp.persist_signature_witnesses([refusal])
    degraded = section["entries"][0]
    assert degraded["outcome"] == "persistence_refused"
    assert degraded["reason"] == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    assert degraded["detail"] == ""
    assert section["metadata_bytes"] < 16 * 1024
    assert sp.validate_section_plain(section).persistence_refused_count == 1


def test_escaped_persistence_refused_detail_is_refused_before_digest(monkeypatch):
    binding = _binding("f", _pixels(68), _png_bytes((7, 7, 24)),
                       signature_sha256=_HEX)
    witness = binding.witness
    count = 20_000
    pointer = np.asarray([0] + [len(witness.cell_source_indices)] * count,
                         dtype=np.int64)
    pointer.flags.writeable = False
    bloated = replace(
        witness, cell_rectangles=tuple((0, 1, 0, 1) for _ in range(count)),
        interior_rectangles=tuple((0, 1, 0, 1) for _ in range(count)),
        cell_source_ptr=pointer, witness_sha256="")
    section = sp.persist_signature_witnesses(
        [replace(binding, witness=bloated, witness_sha256="")])
    assert section["entries"][0]["outcome"] == "persistence_refused"
    section["entries"][0]["detail"] = _ESCAPED_DETAIL
    _resign(section)
    calls = _trap_section_serialization(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    assert calls == {"digest": 0, "serialized": 0}


# ---------------------------------------------------------------------------
# REVISE-2 #2 - the complete section's serialized cap precedes its digest
# ---------------------------------------------------------------------------

def test_over_cap_complete_section_is_refused_before_any_serialization(monkeypatch):
    binding = _binding("f", _pixels(69), _png_bytes((7, 7, 25)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    # Re-policy the section to a serialized cap one byte below its own declared
    # size, keeping the per-entry/aggregate metadata caps large enough for the
    # entry to pass, so only the section-level serialized cap can refuse it.
    for _ in range(8):
        cap = sp.canonical_json_size(section)
        if cap > section["policy"]["max_serialized_bytes"]:
            break
        small = replace(sp.SAMPLING_PERSISTENCE_LIMITS,
                        max_serialized_bytes=cap - 1,
                        max_entry_metadata_bytes=cap - 1,
                        max_total_metadata_bytes=cap - 1)
        section["policy"] = small.to_record()
        _resign(section)
    limits = sp.SamplingPersistenceLimitsV1.from_record(section["policy"])
    assert sp.canonical_json_size(section) > limits.max_serialized_bytes
    calls = _trap_section_serialization(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section, limits=limits)
    assert error.value.reason == sp.REASON_PERSISTENCE_SERIALIZED_OVER_CAP
    assert calls == {"digest": 0, "serialized": 0}


def test_section_serialized_cap_rejects_a_producer_oversized_record():
    """The producer's own output can never exceed the section serialized cap."""
    binding = _binding("f", _pixels(70), _png_bytes((7, 7, 26)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    assert section["status"] == sp.SAMPLING_PERSISTENCE_STATUS_COMPLETE
    assert sp.canonical_json_size(section) <= (
        sp.SAMPLING_PERSISTENCE_LIMITS.max_serialized_bytes)
    sp.validate_section_plain(section)


# ---------------------------------------------------------------------------
# REVISE-4 - the pending section_sha256 reservation is exact, not an 80-byte guess
# ---------------------------------------------------------------------------

def test_sealed_section_size_probe_is_the_exact_digest_member():
    """The seal member contributes exactly 84 canonical bytes, not 80."""
    section = {"status": "complete", "entries": []}
    assert sp.canonical_json_size("section_sha256") == 16
    assert sp.canonical_json_size(sp._SECTION_DIGEST_PLACEHOLDER) == 66
    assert sp._sealed_section_size(section) == (
        sp.canonical_json_size(section) + 84)
    # The placeholder digest has the same canonical length as the real one.
    assert len(sp._canonical_bytes(sp._seal_section(section))) == (
        sp._sealed_section_size(section))


def test_over_cap_complete_section_is_refused_before_the_canonical_walk(
        monkeypatch):
    """Sol's exact boundary: a 1,085-byte complete section under a 1,084 cap.

    The refused-entry fixture's unsealed payload is exactly 1,001 canonical bytes;
    sealing appends the 84-byte ``section_sha256`` member for a complete section
    of exactly 1,085 bytes, one byte above the cap. The former ``+80`` reservation
    (1,081) passed the declared gate, so the over-cap complete section was sealed
    and canonicalized before the second check degraded it. With the exact probe
    the declared walk refuses first and no would-be complete section ever reaches
    ``_canonical_bytes``.
    """
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS,
                     max_serialized_bytes=1084,
                     max_entry_metadata_bytes=1084,
                     max_total_metadata_bytes=1084)
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="x" * 300)
    # Fixture proof, built under a same-width serialized cap so the embedded
    # policy keeps the identical canonical size: exactly the reported boundary.
    wide = replace(limits, max_serialized_bytes=9999)
    complete = sp.persist_signature_witnesses([refusal], limits=wide)
    assert complete["status"] == sp.SAMPLING_PERSISTENCE_STATUS_COMPLETE
    unsealed = sp.canonical_json_size(sp._section_payload(complete))
    assert unsealed == 1001
    assert len(sp._canonical_bytes(complete)) == 1085
    assert sp._sealed_section_size(sp._section_payload(complete)) == 1085
    # The former 80-byte reservation passed the declared gate with room to spare.
    assert unsealed + 80 <= limits.max_serialized_bytes

    calls = {"serialized": 0, "complete": 0}
    real_bytes = sp._canonical_bytes

    def _is_complete(payload):
        if type(payload) is not dict:
            return False
        if payload.get("status") == sp.SAMPLING_PERSISTENCE_STATUS_COMPLETE:
            return True
        inner = payload.get("payload")
        return (type(inner) is dict and inner.get("status")
                == sp.SAMPLING_PERSISTENCE_STATUS_COMPLETE)

    def guarded(payload):
        calls["serialized"] += 1
        if _is_complete(payload):
            calls["complete"] += 1
            raise AssertionError(
                "an over-cap complete section must not be canonicalized")
        return real_bytes(payload)

    monkeypatch.setattr(sp, "_canonical_bytes", guarded)
    section = sp.persist_signature_witnesses([refusal], limits=limits)
    assert section["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert section["reason"] == sp.REASON_PERSISTENCE_SERIALIZED_OVER_CAP
    assert section["field"] == "section"
    assert section["entries"] == []
    # No would-be complete section reached the canonical serializer; only the
    # small typed refusal was sealed, walked and digest-verified.
    assert calls["complete"] == 0
    sp.validate_section_plain(section, limits=limits)
    assert len(sp._canonical_bytes(section)) < 1024


# ---------------------------------------------------------------------------
# REVISE-2 #3 - declared budgets precede copies, scans, encoding and hashing
# ---------------------------------------------------------------------------

def test_work_refusal_makes_no_array_copy_or_value_scan(monkeypatch):
    binding = _binding("f", _pixels(71), _png_bytes((7, 7, 27)),
                       signature_sha256=_HEX)
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS, max_work_units=1)
    real_contiguous = np.ascontiguousarray
    copies = {"count": 0}

    def spy(array, *args, **kwargs):
        copies["count"] += 1
        return real_contiguous(array, *args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no value traversal may precede the work gate")

    monkeypatch.setattr(sp.np, "ascontiguousarray", spy)
    for name in ("_validate_pointer_array", "_validate_index_array",
                 "_validate_weight_array", "_validate_rectification_values",
                 "_validate_bilinear_tap_semantics", "_validate_witness_values",
                 "_encode_witness"):
        monkeypatch.setattr(sp, name, forbidden)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp._account_witness(binding.witness, limits=limits)
    assert error.value.reason == sp.REASON_PERSISTENCE_WORK_OVER_CAP
    # No array was ever made contiguous and no value scanner ran.
    assert copies["count"] == 0


def test_two_refusals_cannot_produce_a_complete_section_over_the_work_cap():
    refusal_a = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="a")
    refusal_b = csp.SourceTapWitnessRefusalV1(
        frame_id="h", canonical_index=3, alternative_index=4,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="b")
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS, max_work_units=1)
    section = sp.persist_signature_witnesses([refusal_a, refusal_b],
                                             limits=limits)
    # Every non-bound outcome charges one declared work unit *plus* its exact
    # metadata bytes, so the first refusal already exceeds a one-unit work cap and
    # the whole section is the typed work refusal instead of a complete section
    # its own validator would reject.
    assert section["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert section["reason"] == sp.REASON_PERSISTENCE_WORK_OVER_CAP
    assert section["entries"] == []
    sp.validate_section_plain(section, limits=limits)
    single = sp.persist_signature_witnesses([refusal_a], limits=limits)
    assert single["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert single["reason"] == sp.REASON_PERSISTENCE_WORK_OVER_CAP
    assert single["entries"] == []


def test_aggregate_payload_refusal_precedes_value_scan_and_encoding(monkeypatch):
    first = _binding("f", _pixels(72), _png_bytes((7, 7, 28)),
                     signature_sha256=_HEX)
    second = _binding("g", _pixels(73), _png_bytes((7, 7, 29)),
                      signature_sha256=_HEX_B)
    first_account = sp._account_witness(first.witness,
                                        limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    second_account = sp._account_witness(second.witness,
                                         limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    cap = max(first_account.payload_bytes, second_account.payload_bytes)
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS,
                     max_entry_payload_bytes=cap, max_total_payload_bytes=cap)
    encoded = {"count": 0}
    real_encode = sp._encode_witness

    def spy_encode(witness):
        encoded["count"] += 1
        return real_encode(witness)

    monkeypatch.setattr(sp, "_encode_witness", spy_encode)
    section = sp.persist_signature_witnesses([first, second], limits=limits)
    assert [entry["outcome"] for entry in section["entries"]] == [
        "bound", "persistence_refused"]
    assert section["entries"][1]["reason"] == (
        sp.REASON_PERSISTENCE_TOTAL_PAYLOAD_OVER_CAP)
    # Only the first (in-budget) witness was ever encoded; the aggregate-payload
    # refusal happened before the second witness was scanned or copied.
    assert encoded["count"] == 1


def test_refusal_entries_never_exceed_the_per_entry_metadata_cap():
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="witness_absent")
    # A per-entry cap below even the minimal degraded refusal record, with a
    # generous aggregate cap: the producer must refuse the whole section (which
    # its own validator accepts) rather than emit a complete section whose
    # degraded entry is over the per-entry metadata cap.
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS,
                     max_entry_metadata_bytes=64,
                     max_total_metadata_bytes=1024 * 1024)
    section = sp.persist_signature_witnesses([refusal], limits=limits)
    assert section["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert section["reason"] == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP
    assert section["entries"] == []
    assert sp.validate_section_plain(section, limits=limits).entry_count == 0


# ---------------------------------------------------------------------------
# REVISE-2 #4 - exact primitive types before equality/dedup/digest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ordinal_value", [0.0, False])
def test_non_integer_ordinal_never_satisfies_an_integer_ordinal(ordinal_value):
    binding = _binding("f", _pixels(74), _png_bytes((7, 7, 30)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    assert section["entries"][0]["ordinal"] == 0
    # ``0.0 == 0`` and ``False == 0`` in Python, so only an exact type check can
    # refuse these.
    assert ordinal_value == 0
    section["entries"][0]["ordinal"] = ordinal_value
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_RECORD_MALFORMED


@pytest.mark.parametrize("field,value", [
    ("entry_count", 1.0),
    ("payload_bytes", 0.0),
    ("metadata_bytes", 0.0),
    ("work_units", 0.0),
])
def test_accounting_scalars_require_exact_integers(field, value):
    binding = _binding("f", _pixels(75), _png_bytes((7, 7, 31)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    assert type(section[field]) is int
    section[field] = value
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_RECORD_MALFORMED


@pytest.mark.parametrize("name", ["cell_rectangles", "interior_rectangles"])
def test_null_rectangle_sequence_is_a_typed_refusal_not_a_type_error(name):
    binding = _binding("f", _pixels(76), _png_bytes((7, 7, 32)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    section["entries"][0]["witness"][name] = None
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_RECTANGLE_INVALID


@pytest.mark.parametrize("path,value", [
    (("gradient_edge_order",), 1.0),
    (("rgb_rectification", "inter_bits"), 5.0),
    (("rgb_rectification", "weight_denominator"), 1024.0),
])
def test_float_backend_scalars_are_refused_as_schema_invalid(path, value):
    binding = _binding("f", _pixels(77), _png_bytes((7, 7, 33)),
                       signature_sha256=_HEX)
    section = _persist(binding)
    record = section["entries"][0]["witness"]
    for key in path[:-1]:
        record = record[key]
    record[path[-1]] = value
    _resign(section)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(section)
    assert error.value.reason == sp.REASON_PERSISTENCE_SCHEMA_INVALID


# ---------------------------------------------------------------------------
# REVISE-3 #2 - declared witness work is reserved before the value scans and is
# never refunded on a semantic failure
# ---------------------------------------------------------------------------

def _weight_tampered_binding(binding):
    """Rebuild one binding whose last bilinear tap weight is changed.

    The change keeps every declared cap and structural invariant valid (the
    ``__post_init__`` digest is recomputed by passing ``witness_sha256=""``), so
    only the per-sample weight-sum contract fails.
    """
    rect = binding.witness.rgb_rectification
    weights = np.array(rect.tap_weight_numerators, copy=True)
    weights[-1] = 2 if int(weights[-1]) == 1 else 1
    weights.flags.writeable = False
    tampered_rect = replace(rect, tap_weight_numerators=weights,
                            witness_sha256="")
    tampered_witness = replace(binding.witness, rgb_rectification=tampered_rect,
                               witness_sha256="")
    return replace(binding, witness=tampered_witness, witness_sha256="")


def test_declared_witness_work_is_reserved_before_scans_and_never_refunded(
        monkeypatch):
    first = _weight_tampered_binding(
        _binding("f", _pixels(78), _png_bytes((7, 7, 34)),
                 signature_sha256=_HEX))
    second = _weight_tampered_binding(
        _binding("g", _pixels(79), _png_bytes((7, 7, 35)),
                 signature_sha256=_HEX_B))
    # The tamper passes every declared cap and fails only the per-sample bilinear
    # weight contract, so it exercises the semantic-refusal path *after* the
    # declared work was reserved and the value scan ran.
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp._validate_witness_values(first.witness,
                                    limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    assert error.value.reason == sp.REASON_PERSISTENCE_WEIGHT_INVALID
    first_account = sp._account_witness(
        first.witness, limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    second_account = sp._account_witness(
        second.witness, limits=sp.SAMPLING_PERSISTENCE_LIMITS)
    assert first_account.work_units == second_account.work_units
    # The cap admits exactly one witness's declared work; the two witnesses
    # together declare twice that.
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS,
                     max_work_units=first_account.work_units + 1)
    scans = {"count": 0}
    real_scan = sp._validate_witness_values

    def spy(witness, *, limits):
        scans["count"] += 1
        return real_scan(witness, limits=limits)

    monkeypatch.setattr(sp, "_validate_witness_values", spy)
    section = sp.persist_signature_witnesses([first, second], limits=limits)
    # The first witness was scanned and semantically refused; its full declared
    # work stayed charged, so the second witness trips the aggregate work cap
    # before it is ever scanned or encoded.
    assert scans["count"] == 1
    assert section["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert section["reason"] == sp.REASON_PERSISTENCE_WORK_OVER_CAP
    assert section["entries"] == []


# ---------------------------------------------------------------------------
# REVISE-3 #3 - non-bound refusal metadata traversal/encoding is charged as work
# ---------------------------------------------------------------------------

def test_non_bound_refusal_metadata_work_is_charged(monkeypatch):
    refusal = csp.SourceTapWitnessRefusalV1(
        frame_id="g", canonical_index=1, alternative_index=2,
        reason=csp.SOURCE_TAP_WITNESS_UNAVAILABLE, detail="x" * 60_000)
    section = sp.persist_signature_witnesses([refusal])
    entry = section["entries"][0]
    entry_metadata = sp._non_bound_entry_metadata_bytes(
        entry,
        limit=sp.SAMPLING_PERSISTENCE_LIMITS.max_entry_metadata_bytes)
    assert entry_metadata > 60_000
    assert section["metadata_bytes"] == entry_metadata
    assert section["work_units"] == 1 + entry_metadata
    # Producer and plain validator agree on the identical charge.
    accounting = sp.validate_section_plain(section)
    assert accounting.work_units == section["work_units"] == 1 + entry_metadata
    # Under a one-unit work cap the same refusal can no longer be a complete
    # section: the producer refuses the whole section and the plain validator
    # accepts that typed refusal.
    tight = replace(sp.SAMPLING_PERSISTENCE_LIMITS, max_work_units=1)
    capped = sp.persist_signature_witnesses([refusal], limits=tight)
    assert capped["status"] == sp.SAMPLING_PERSISTENCE_STATUS_REFUSED
    assert capped["reason"] == sp.REASON_PERSISTENCE_WORK_OVER_CAP
    assert capped["entries"] == []
    sp.validate_section_plain(capped, limits=tight)
    # Validator and digest order: the equivalent complete section that claims the
    # old one-unit work total is refused before the section digest is computed.
    forged = copy.deepcopy(section)
    forged["policy"] = tight.to_record()
    forged["work_units"] = 1
    _resign(forged)
    calls = _trap_section_serialization(monkeypatch)
    with pytest.raises(sp.SamplingPersistenceError) as error:
        sp.validate_section_plain(forged, limits=tight)
    assert error.value.reason == sp.REASON_PERSISTENCE_WORK_OVER_CAP
    assert calls == {"digest": 0, "serialized": 0}


# ---------------------------------------------------------------------------
# REVISE-3 #4 - rectangle lists are materialized only after every budget passes
# ---------------------------------------------------------------------------

def test_rectangle_lists_are_not_materialized_before_the_metadata_cap(monkeypatch):
    binding = _binding("f", _pixels(80), _png_bytes((7, 7, 36)),
                       signature_sha256=_HEX)
    witness_metadata = sp._declared_witness_metadata_bytes(binding.witness)
    entry_metadata = sp._declared_entry_metadata_bytes({
        "outcome": sp.SAMPLING_PERSISTENCE_OUTCOME_BOUND, "ordinal": 0,
        "schema": sp.SAMPLING_PERSISTENCE_ENTRY_SCHEMA,
        **sp._binding_identity(binding)}, binding.witness)
    # A cap that admits the witness metadata but not the slightly larger bound
    # entry metadata: only the entry metadata check can refuse this witness.
    assert entry_metadata > witness_metadata
    limits = replace(sp.SAMPLING_PERSISTENCE_LIMITS,
                     max_entry_metadata_bytes=witness_metadata)
    calls = {"materialized": 0}
    real = sp._witness_metadata

    def trap(witness, *, digest, materialize_rectangles=True):
        if materialize_rectangles:
            calls["materialized"] += 1
            raise AssertionError(
                "per-cell rectangle lists must not be built before the caps")
        return real(witness, digest=digest, materialize_rectangles=False)

    monkeypatch.setattr(sp, "_witness_metadata", trap)
    section = sp.persist_signature_witnesses([binding], limits=limits)
    # The over-cap witness refused at the declared entry-metadata size, so no
    # rectangle list was ever built and the encoder never ran.
    assert calls["materialized"] == 0
    entry = section["entries"][0]
    assert entry["outcome"] == "persistence_refused"
    assert entry["reason"] == sp.REASON_PERSISTENCE_ENTRY_METADATA_OVER_CAP

