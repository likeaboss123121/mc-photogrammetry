"""Focused controls for the atomic image-derived signature producer."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_lattice_measure import _result, _producer1_record, _historical_bytes  # noqa: E402

from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_session as sessions  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_signatures as subject  # noqa: E402
from mc_photogrammetry.reconstruction import lattice_planes  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement as placement  # noqa: E402
from mc_photogrammetry.vision.gui_mask import create_gui_mask  # noqa: E402


@pytest.fixture(scope="module")
def records():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = _result(patch)
        yield session, lattice
    finally:
        patch.undo()


def _forge(record, **changes):
    forged = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(record, field.name)))
    return forged


def test_assembled_result_covers_every_alternative_and_seals_digests(records):
    session, lattice = records
    result = subject.derive_image_derived_signatures(
        session=session, lattice_result=lattice)
    assert isinstance(result, subject.ImageDerivedSignaturesResult)
    assert result.status == "assembled" and result.reason == ""
    assert result.session_sha256 == session.session_sha256
    assert result.lattice_sha256 == lattice.result_sha256
    assert len(result.signatures) == sum(
        len(reading.alternatives) for reading in lattice.readings)
    assert subject._hex64(result.masks_sha256)
    assert subject._hex64(result.signatures_sha256)
    assert subject._hex64(result.result_sha256)


def test_producer1_measure_refuses_before_signature_numerics_and_producer3_binds(records, monkeypatch):
    session, lattice = records
    current = subject.derive_image_derived_signatures(session=session, lattice_result=lattice)
    assert current.status == "assembled" and lattice.producer == "m9c-image-derived-lattice-measure-3"
    assert current.lattice_sha256 == lattice.result_sha256
    historical = _producer1_record(lattice)
    before = _historical_bytes(historical)
    monkeypatch.setattr(lattice_planes, "plane_signature",
                        lambda *_a, **_k: pytest.fail("historical measure must refuse before sampling"))
    refused = subject.derive_image_derived_signatures(session=session, lattice_result=historical)
    assert isinstance(refused, placement.PlacementRefusal) and refused.reason == "provenance_mismatch"
    assert _historical_bytes(historical) == before


def test_masks_are_exact_normal_mode_usable_masks_and_are_read_only(records):
    session, lattice = records
    result = subject.derive_image_derived_signatures(
        session=session, lattice_result=lattice)
    assert tuple(mask.frame_id for mask in result.usable_masks) == result.frame_ids
    for frame, record in zip(session.frames, result.usable_masks):
        expected = ~create_gui_mask(*frame.image_shape[:2])
        assert np.array_equal(record.mask, expected)
        assert record.mask.dtype == np.bool_ and not record.mask.flags.writeable
        with pytest.raises(ValueError):
            record.mask[0, 0] = not record.mask[0, 0]


def test_signature_arrays_are_read_only_and_bound_to_readings(records):
    session, lattice = records
    result = subject.derive_image_derived_signatures(
        session=session, lattice_result=lattice)
    for evidence in result.signatures:
        reading = lattice.readings[evidence.canonical_index]
        assert evidence.signature.reading is reading
        assert evidence.frame_id == reading.frame
        assert evidence.blocks_per_period == reading.alternatives[
            evidence.alternative_index].blocks_per_period
        for array in (evidence.signature.cells, evidence.signature.interior,
                      evidence.signature.coverage):
            assert not array.flags.writeable


def test_a_deserialized_session_without_pixels_is_atomically_unavailable(records):
    session, lattice = records
    stripped = sessions.image_derived_session_from_record(
        sessions.image_derived_session_to_record(session))
    result = subject.derive_image_derived_signatures(
        session=stripped, lattice_result=lattice)
    assert isinstance(result, subject.ImageDerivedSignaturesResult)
    assert result.status == "unavailable" and result.reason == "image_unavailable"
    assert result.usable_masks == () and result.signatures == ()
    assert result.abstentions


def test_any_signature_failure_discards_all_partial_outputs(records, monkeypatch):
    session, lattice = records

    def fail(*_args, **_kwargs):
        raise ValueError("failed")

    monkeypatch.setattr(lattice_planes, "plane_signature", fail)
    result = subject.derive_image_derived_signatures(
        session=session, lattice_result=lattice)
    assert result.status == "unavailable"
    assert result.reason == "signature_unavailable"
    assert result.usable_masks == () and result.signatures == ()
    assert result.abstentions[0].reason == "signature_failure"


def test_foreign_or_tampered_upstream_records_are_typed_refusals(records):
    session, lattice = records
    foreign = dataclasses.replace(
        lattice, session_sha256="f" * 64, result_sha256=None)
    mismatch = subject.derive_image_derived_signatures(
        session=session, lattice_result=foreign)
    assert isinstance(mismatch, placement.PlacementRefusal)
    assert mismatch.reason == "provenance_mismatch"
    stale = _forge(lattice, result_sha256="0" * 64)
    tampered = subject.derive_image_derived_signatures(
        session=session, lattice_result=stale)
    assert isinstance(tampered, placement.PlacementRefusal)
    assert tampered.reason == "provenance_mismatch"


def test_result_replays_nested_helper_digests(records):
    session, lattice = records
    result = subject.derive_image_derived_signatures(
        session=session, lattice_result=lattice)
    mask = result.usable_masks[0]
    changed = bytes([1 - mask.usable_bool_bytes[0]]) + mask.usable_bool_bytes[1:]
    forged = _forge(mask, usable_bool_bytes=changed)
    with pytest.raises(ValueError, match="digest"):
        dataclasses.replace(result, usable_masks=(forged,) + result.usable_masks[1:],
                            result_sha256="")


def test_public_api_has_no_manual_geometry_or_solver_inputs():
    import inspect

    signature = inspect.signature(subject.derive_image_derived_signatures)
    # ``signature_witness_sink`` and the sampler caps are the evaluation-only
    # source-tap witness seam; the sink defaults off and no geometry/manual input
    # is accepted.
    assert tuple(signature.parameters) == (
        "session", "lattice_result", "signature_witness_sink", "max_cells",
        "max_rectified_side", "pixels_per_cell")
    assert signature.parameters["signature_witness_sink"].default is None
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "subprocess"):
        assert forbidden not in source
