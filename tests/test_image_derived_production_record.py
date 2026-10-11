"""Focused controls for the durable schema-3 image-derived production record.

The records here are built entirely in memory from the existing accepted
state-promotion fixtures (a synthetic ``PlaneVolumeResult`` and the real
proposal producer) plus the existing accepted image-derived session fixture
(a fake camera solver only).  No image file, camera workload, exporter workload
or v5 package is read; the exporter is exercised once over the already-accepted
promotion volume to prove the resume-export path.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_project_store import _session_record  # noqa: E402
from test_state_promotion import _budget, _proposals  # noqa: E402
from test_state_promotion import _volume as _geometry_volume  # noqa: E402

from mc_photogrammetry.domain.models import (  # noqa: E402
    SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
    BlockObservation,
    BlockState,
    BlockVolume,
    ScanProject,
)
from mc_photogrammetry.export.litematic import (  # noqa: E402
    LitematicExportError,
    LitematicExporter,
)
from mc_photogrammetry.pipeline import image_derived_delivery  # noqa: E402
from mc_photogrammetry.pipeline import (  # noqa: E402
    image_derived_production_record as record_mod,
)
from mc_photogrammetry.project import store as store_mod  # noqa: E402
from mc_photogrammetry.reconstruction import state_promotion  # noqa: E402

_SESSION_DIGEST = "a" * 64
_JOINT_LINEAGE = ("mc-photogrammetry-image-derived-graph-v3", "m9-g-image-derived-graph-7", "1" * 64,
                  "mc-photogrammetry-endpoint-selection-v1", "m9-g-endpoint-selection-1", "2" * 64)
_COMPACT_JOINT_LINEAGE = ("mc-photogrammetry-image-derived-graph-v5", "m9-g-image-derived-graph-9", "1" * 64,
                          "mc-photogrammetry-endpoint-selection-v3", "m9-g-endpoint-selection-3", "2" * 64)
_FORCED_JOINT_LINEAGE = ("mc-photogrammetry-image-derived-graph-v6", "m9-g-image-derived-graph-10", "1" * 64,
                         "mc-photogrammetry-endpoint-selection-v4", "m9-g-endpoint-selection-4", "2" * 64)
_CURRENT_JOINT_LINEAGE = ("mc-photogrammetry-image-derived-graph-v4", "m9-g-image-derived-graph-8", "1" * 64,
                          "mc-photogrammetry-endpoint-selection-v2", "m9-g-endpoint-selection-2", "2" * 64)
_FORBIDDEN_PIXEL_KEYS = {"image", "pixels", "array", "pixel_data", "image_data"}


def _contains_pixel_payload(value) -> bool:
    import numpy as np

    if isinstance(value, dict):
        return any(
            key in _FORBIDDEN_PIXEL_KEYS or _contains_pixel_payload(item)
            for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_pixel_payload(item) for item in value)
    return isinstance(value, np.ndarray)


def _promotion(coordinates=((0, 0, 0), (1, 0, 0))) -> state_promotion.PromotionResult:
    outcome = state_promotion.promote_plane_states(
        _geometry_volume(coordinates), _proposals(coordinates), budget=_budget())
    assert isinstance(outcome, state_promotion.PromotionResult), outcome
    return outcome


def _omitted_promotion() -> state_promotion.PromotionResult:
    volume = _geometry_volume(((0, 0, 0),), omitted=1)
    proposals = _proposals(((0, 0, 0),))
    outcome = state_promotion.promote_plane_states(
        volume, proposals, budget=_budget(), allow_omitted_unknown=True)
    assert isinstance(outcome, state_promotion.PromotionResult), outcome
    assert outcome.exact_state_eligible is False
    return outcome


def _promoted_envelope(session, promotion):
    return record_mod.build_production_envelope(
        stage="output", status="promoted", reason="", detail="",
        entry_sha256="b" * 64, lattice_sha256="c" * 64,
        geometry_sha256="d" * 64, output_sha256="e" * 64,
        session_sha256=session["session_sha256"], promotion=promotion, joint_lineage=_JOINT_LINEAGE)


def _refusal_envelope(session):
    return record_mod.build_production_envelope(
        stage="entry", status="geometry_unavailable", reason="entry_unavailable",
        detail="V1/V2/V3/M5 geometry is unavailable from images alone",
        entry_sha256="b" * 64, session_sha256=session["session_sha256"])


def _promoted_project(session, promotion, volume=None) -> ScanProject:
    return ScanProject(
        name="durable",
        source_paths=["a.png"],
        volume=promotion.to_block_volume() if volume is None else volume,
        schema_version=SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
        image_derived_session=session,
        image_derived_production=_promoted_envelope(session, promotion).to_record(),
    )


def _refusal_project(session) -> ScanProject:
    return ScanProject(
        name="refused",
        source_paths=["a.png"],
        volume=BlockVolume(),
        schema_version=SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
        image_derived_session=session,
        image_derived_production=_refusal_envelope(session).to_record(),
    )


def _resealed(record: dict) -> dict:
    """Copy a raw record and reseal ``record_sha256`` over the mutated payload.

    Negative tests use this so the size/digest checks cannot mask a semantic
    stage-prefix violation: the returned record has a *valid* enclosing digest.
    """
    payload = {key: value for key, value in record.items()
               if key != "record_sha256"}
    resealed = dict(record)
    resealed["record_sha256"] = hashlib.sha256(
        record_mod._canonical_json(payload).encode("utf-8")).hexdigest()
    return resealed


# ---------------------------------------------------------------------------
# Envelope codec
# ---------------------------------------------------------------------------

def test_promoted_envelope_round_trips_and_replays(monkeypatch):
    session = _session_record(monkeypatch)
    promotion = _promotion()

    envelope = _promoted_envelope(session, promotion)
    replayed = record_mod.production_envelope_from_record(envelope.to_record())

    assert replayed.to_record() == envelope.to_record()
    assert replayed.record_sha256 == envelope.record_sha256
    assert replayed.promotion.to_record() == promotion.to_record()
    assert record_mod.production_volume(replayed).to_dict() \
        == promotion.to_block_volume().to_dict()


@pytest.mark.parametrize("lineage", [_JOINT_LINEAGE, _CURRENT_JOINT_LINEAGE, _COMPACT_JOINT_LINEAGE, _FORCED_JOINT_LINEAGE], ids=["old", "current", "compact", "forced"])
def test_supported_joint_lineage_preserves_schema3_record_bytes(lineage):
    # Independently sealed schema-3/producer-2 bytes must replay without migration.
    expected = _resealed({
        "schema": "mc-photogrammetry-image-derived-production-record-3",
        "producer": "ds-r9-durable-production-record-2",
        "stage": "geometry", "status": "geometry_unavailable",
        "reason": "downstream", "detail": "",
        "entry_sha256": "b" * 64, "lattice_sha256": "c" * 64,
        "geometry_sha256": "d" * 64, "output_sha256": "",
        "session_sha256": _SESSION_DIGEST,
        "volume_sha256": hashlib.sha256(record_mod._canonical_json(BlockVolume().to_dict()).encode()).hexdigest(),
        "promotion": None, "refusal_sha256": "", "joint_lineage": list(lineage),
    })
    encoded = record_mod._canonical_json(expected).encode()
    replayed = record_mod.production_envelope_from_record(json.loads(encoded))
    assert replayed.joint_lineage == lineage
    assert record_mod._canonical_json(replayed.to_record()).encode() == encoded
    built = record_mod.build_production_envelope(
        stage="geometry", status="geometry_unavailable", reason="downstream",
        entry_sha256="b" * 64, lattice_sha256="c" * 64, geometry_sha256="d" * 64,
        session_sha256=_SESSION_DIGEST, joint_lineage=lineage)
    assert built.to_record() == expected


@pytest.mark.parametrize("lineage", [
    _JOINT_LINEAGE[:3] + _CURRENT_JOINT_LINEAGE[3:],
    _CURRENT_JOINT_LINEAGE[:3] + _JOINT_LINEAGE[3:],
    (_CURRENT_JOINT_LINEAGE[0], _JOINT_LINEAGE[1]) + _CURRENT_JOINT_LINEAGE[2:],
    (_JOINT_LINEAGE[0], _CURRENT_JOINT_LINEAGE[1]) + _JOINT_LINEAGE[2:],
    _CURRENT_JOINT_LINEAGE[:4] + (_JOINT_LINEAGE[4], _CURRENT_JOINT_LINEAGE[5]),
    _JOINT_LINEAGE[:4] + (_CURRENT_JOINT_LINEAGE[4], _JOINT_LINEAGE[5]),
    _CURRENT_JOINT_LINEAGE[:3] + _COMPACT_JOINT_LINEAGE[3:],
    _COMPACT_JOINT_LINEAGE[:3] + _CURRENT_JOINT_LINEAGE[3:],
    ("mc-photogrammetry-image-derived-graph-v7", "m9-g-image-derived-graph-11", "1" * 64,
     "mc-photogrammetry-endpoint-selection-v5", "m9-g-endpoint-selection-5", "2" * 64),
    _FORCED_JOINT_LINEAGE[:3] + _COMPACT_JOINT_LINEAGE[3:],
    _COMPACT_JOINT_LINEAGE[:3] + _FORCED_JOINT_LINEAGE[3:],
], ids=["old-graph-current-selection", "current-graph-old-selection",
        "current-graph-old-producer", "old-graph-current-producer",
        "current-selection-old-producer", "old-selection-current-producer",
        "current-graph-compact-selection", "compact-graph-current-selection", "unknown",
        "forced-graph-compact-selection", "compact-graph-forced-selection"])
def test_mixed_or_unknown_joint_lineage_fails_construction_and_resealed_replay(lineage):
    with pytest.raises(record_mod.ProductionEnvelopeError, match="joint lineage"):
        record_mod.build_production_envelope(
            stage="geometry", status="geometry_unavailable", reason="downstream",
            entry_sha256="b" * 64, lattice_sha256="c" * 64, geometry_sha256="d" * 64,
            session_sha256=_SESSION_DIGEST, joint_lineage=lineage)
    envelope = record_mod.build_production_envelope(
        stage="geometry", status="geometry_unavailable", reason="downstream",
        entry_sha256="b" * 64, lattice_sha256="c" * 64, geometry_sha256="d" * 64,
        session_sha256=_SESSION_DIGEST, joint_lineage=_JOINT_LINEAGE)
    raw = dict(envelope.to_record(), joint_lineage=list(lineage))
    with pytest.raises(record_mod.ProductionEnvelopeError, match="joint lineage"):
        record_mod.production_envelope_from_record(_resealed(raw))


@pytest.mark.parametrize("lineage", [_JOINT_LINEAGE, _CURRENT_JOINT_LINEAGE, _COMPACT_JOINT_LINEAGE, _FORCED_JOINT_LINEAGE], ids=["old", "current", "compact", "forced"])
@pytest.mark.parametrize("field", [2, 5], ids=["graph-digest", "selection-digest"])
@pytest.mark.parametrize("invalid", ["", "not-a-hash", "A" * 64, None])
def test_supported_joint_identities_still_require_both_valid_digests(lineage, field, invalid):
    envelope = record_mod.build_production_envelope(
        stage="geometry", status="geometry_unavailable", reason="downstream",
        entry_sha256="b" * 64, lattice_sha256="c" * 64, geometry_sha256="d" * 64,
        session_sha256=_SESSION_DIGEST, joint_lineage=lineage)
    raw = envelope.to_record()
    raw["joint_lineage"][field] = invalid
    with pytest.raises(record_mod.ProductionEnvelopeError, match="joint lineage"):
        record_mod.production_envelope_from_record(_resealed(raw))


def test_promoted_envelope_is_pixel_free_and_json_deterministic(monkeypatch):
    session = _session_record(monkeypatch)
    envelope = _promoted_envelope(session, _promotion())

    first = record_mod.production_envelope_from_record(
        envelope.to_record()).to_record()
    second = record_mod.production_envelope_from_record(
        envelope.to_record()).to_record()

    assert first == second
    assert json.dumps(first, allow_nan=False)
    assert not _contains_pixel_payload(first)


def test_a_non_promoted_envelope_carries_zero_volume(monkeypatch):
    session = _session_record(monkeypatch)
    envelope = _refusal_envelope(session)

    assert envelope.promotion is None
    assert record_mod.production_volume(envelope).blocks == {}
    replayed = record_mod.production_envelope_from_record(envelope.to_record())
    assert replayed.status == "geometry_unavailable"
    assert replayed.reason == "entry_unavailable"


def test_a_promotion_with_omissions_cannot_be_promoted(monkeypatch):
    session = _session_record(monkeypatch)
    omitted = _omitted_promotion()

    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.build_production_envelope(
            stage="output", status="promoted", reason="", detail="",
            entry_sha256="b" * 64, session_sha256=session["session_sha256"],
            promotion=omitted)


def test_a_promoted_envelope_needs_at_least_one_block(monkeypatch):
    session = _session_record(monkeypatch)
    promotion = _promotion()
    fields = {field.name: getattr(promotion, field.name)
              for field in dataclasses.fields(promotion)}
    empty = state_promotion.PromotionResult.build(**{**fields, "cells": ()})
    assert empty.cells == ()
    assert empty.export_ready is True and empty.exact_state_eligible is True

    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.build_production_envelope(
            stage="output", status="promoted", reason="", detail="",
            entry_sha256="b" * 64, session_sha256=session["session_sha256"],
            promotion=empty)


def test_a_promoted_envelope_needs_the_exact_promoted_stage(monkeypatch):
    session = _session_record(monkeypatch)
    promotion = _promotion()

    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.build_production_envelope(
            stage="geometry", status="promoted", reason="", detail="",
            entry_sha256="b" * 64, session_sha256=session["session_sha256"],
            promotion=promotion)


def test_a_non_promoted_envelope_cannot_claim_blocks(monkeypatch):
    session = _session_record(monkeypatch)
    volume = BlockVolume()
    volume.set((0, 0, 0), BlockObservation(BlockState("minecraft:stone")))

    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.build_production_envelope(
            stage="entry", status="geometry_unavailable", reason="entry_unavailable",
            entry_sha256="b" * 64, session_sha256=session["session_sha256"],
            volume=volume)


def test_envelope_rejects_a_tampered_record(monkeypatch):
    session = _session_record(monkeypatch)
    envelope = _promoted_envelope(session, _promotion())

    stale = envelope.to_record()
    stale["record_sha256"] = "0" * 64
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(stale)

    swapped = envelope.to_record()
    swapped["session_sha256"] = "0" * 64
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(swapped)

    forged = envelope.to_record()
    forged["promotion"]["cells"][0]["state"]["name"] = "minecraft:air"
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(forged)


def test_envelope_rejects_unknown_or_missing_keys(monkeypatch):
    session = _session_record(monkeypatch)
    envelope = _refusal_envelope(session)

    extra = envelope.to_record()
    extra["mystery"] = 1
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(extra)

    missing = envelope.to_record()
    missing.pop("volume_sha256")
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(missing)


def test_envelope_rejects_an_oversized_record(monkeypatch):
    session = _session_record(monkeypatch)
    monkeypatch.setattr(record_mod, "PRODUCTION_RECORD_MAX_BYTES", 16)

    with pytest.raises(record_mod.ProductionEnvelopeError):
        _promoted_envelope(session, _promotion())


def test_the_size_bound_counts_the_record_digest_member(monkeypatch):
    session = _session_record(monkeypatch)
    monkeypatch.setattr(record_mod, "PRODUCTION_RECORD_MAX_BYTES", 1 << 30)
    envelope = _refusal_envelope(session)

    full = record_mod._canonical_json(envelope.to_record())
    payload = record_mod._canonical_json(record_mod._record_payload(envelope))
    full_bytes = len(full.encode("utf-8"))
    payload_bytes = len(payload.encode("utf-8"))
    assert full_bytes > payload_bytes

    # At the pre-digest payload size the old bound would have accepted; the
    # complete record, which includes ``record_sha256``, must be refused.
    monkeypatch.setattr(record_mod, "PRODUCTION_RECORD_MAX_BYTES", payload_bytes)
    with pytest.raises(record_mod.ProductionEnvelopeError):
        _refusal_envelope(session)

    # The exact complete-record size is the inclusive boundary.
    monkeypatch.setattr(record_mod, "PRODUCTION_RECORD_MAX_BYTES", full_bytes)
    assert record_mod._canonical_json(
        _refusal_envelope(session).to_record()) == full

    monkeypatch.setattr(record_mod, "PRODUCTION_RECORD_MAX_BYTES", full_bytes - 1)
    with pytest.raises(record_mod.ProductionEnvelopeError):
        _refusal_envelope(session)


# ---------------------------------------------------------------------------
# Canonical stage-prefix semantics
# ---------------------------------------------------------------------------

def _lattice_envelope(session):
    return record_mod.build_production_envelope(
        stage="lattice", status="geometry_unavailable",
        reason="lattice_unavailable", entry_sha256="b" * 64,
        lattice_sha256="c" * 64, session_sha256=session["session_sha256"])


def _output_envelope(session):
    return record_mod.build_production_envelope(
        stage="output", status="geometry_unavailable",
        reason="output_unavailable", entry_sha256="b" * 64,
        lattice_sha256="c" * 64, geometry_sha256="d" * 64,
        output_sha256="e" * 64, session_sha256=session["session_sha256"], joint_lineage=_JOINT_LINEAGE)


def test_a_normal_stage_carries_its_stage_digest_and_no_refusal(monkeypatch):
    session = _session_record(monkeypatch)
    envelope = _lattice_envelope(session)
    assert envelope.lattice_sha256 == "c" * 64
    assert envelope.refusal_sha256 == ""

    missing_current = _resealed({**envelope.to_record(), "lattice_sha256": ""})
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(missing_current)

    stray_refusal = _resealed({**envelope.to_record(), "refusal_sha256": "0" * 64})
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(stray_refusal)


def test_a_later_stage_digest_cannot_appear_before_its_stage(monkeypatch):
    session = _session_record(monkeypatch)
    entry_stage = _refusal_envelope(session)
    entry_leak = _resealed({**entry_stage.to_record(), "lattice_sha256": "c" * 64})
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(entry_leak)

    lattice_leak = _resealed(
        {**_lattice_envelope(session).to_record(), "geometry_sha256": "d" * 64})
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(lattice_leak)


def test_an_output_terminal_needs_every_prior_and_current_digest(monkeypatch):
    session = _session_record(monkeypatch)
    envelope = _output_envelope(session)
    for field in ("lattice_sha256", "geometry_sha256", "output_sha256"):
        missing = _resealed({**envelope.to_record(), field: ""})
        with pytest.raises(record_mod.ProductionEnvelopeError):
            record_mod.production_envelope_from_record(missing)

    promoted = _promoted_envelope(session, _promotion())
    promoted_missing = _resealed({**promoted.to_record(), "geometry_sha256": ""})
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(promoted_missing)


def test_direct_refusals_omit_their_current_stage_digest(monkeypatch):
    session = _session_record(monkeypatch)
    refusal = "f" * 64
    lattice_refusal = record_mod.build_production_envelope(
        stage="lattice", status="refused", reason="placement_refused",
        entry_sha256="b" * 64, refusal_sha256=refusal,
        session_sha256=session["session_sha256"])
    geometry_refusal = record_mod.build_production_envelope(
        stage="geometry", status="refused", reason="placement_refused",
        entry_sha256="b" * 64, lattice_sha256="c" * 64,
        refusal_sha256=refusal, session_sha256=session["session_sha256"])
    output_refusal = record_mod.build_production_envelope(
        stage="output", status="refused", reason="placement_refused",
        entry_sha256="b" * 64, lattice_sha256="c" * 64,
        geometry_sha256="d" * 64, refusal_sha256=refusal,
        session_sha256=session["session_sha256"], joint_lineage=_JOINT_LINEAGE)

    assert lattice_refusal.lattice_sha256 == ""
    assert geometry_refusal.geometry_sha256 == ""
    assert output_refusal.output_sha256 == ""
    for envelope in (lattice_refusal, geometry_refusal, output_refusal):
        replayed = record_mod.production_envelope_from_record(
            envelope.to_record())
        assert replayed.to_record() == envelope.to_record()
        assert replayed.status == "refused"
        assert replayed.refusal_sha256 == refusal

    carried_current = _resealed(
        {**output_refusal.to_record(), "output_sha256": "e" * 64})
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(carried_current)

    missing_refusal = _resealed(
        {**lattice_refusal.to_record(), "refusal_sha256": ""})
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.production_envelope_from_record(missing_refusal)


def test_a_refusal_digest_not_a_status_string_marks_a_direct_refusal(
        monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    # An ordinary output-stage result may itself carry status "refused" while
    # still holding its completed stage digest and no refusal digest.
    envelope = record_mod.build_production_envelope(
        stage="output", status="refused", reason="placement_refused",
        entry_sha256="b" * 64, lattice_sha256="c" * 64,
        geometry_sha256="d" * 64, output_sha256="e" * 64,
        session_sha256=session["session_sha256"], joint_lineage=_JOINT_LINEAGE)
    assert envelope.refusal_sha256 == ""
    assert envelope.output_sha256 == "e" * 64
    replayed = record_mod.production_envelope_from_record(envelope.to_record())
    assert replayed.to_record() == envelope.to_record()
    assert replayed.status == "refused"

    # Its zero-volume terminal publishes and reloads through the durable store.
    project = ScanProject(
        name="ordinary-refused",
        source_paths=["a.png"],
        volume=BlockVolume(),
        schema_version=SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
        image_derived_session=session,
        image_derived_production=envelope.to_record())
    path = tmp_path / "ordinary-refused.mcscan"
    store_mod.publish_project_no_clobber(project, path)
    loaded = store_mod.load_project(path)
    assert loaded.volume.to_dict()["blocks"] == []
    assert loaded.image_derived_production["status"] == "refused"
    assert loaded.image_derived_production["promotion"] is None

    # A refusal digest on a non-refused status is not a direct refusal.
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.build_production_envelope(
            stage="lattice", status="geometry_unavailable",
            reason="lattice_unavailable", entry_sha256="b" * 64,
            refusal_sha256="f" * 64, session_sha256=session["session_sha256"])

    # The entry stage is always bound and can never be a direct refusal.
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.build_production_envelope(
            stage="entry", status="refused", reason="entry_refused",
            entry_sha256="b" * 64, refusal_sha256="f" * 64,
            session_sha256=session["session_sha256"])


def test_canonicalise_binds_the_session_and_stored_volume(monkeypatch):
    session = _session_record(monkeypatch)
    promotion = _promotion()
    record = _promoted_envelope(session, promotion).to_record()
    volume = promotion.to_block_volume()

    canonical = record_mod.canonicalise_production_record(
        record, session_sha256=session["session_sha256"], volume=volume)
    assert canonical == record

    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.canonicalise_production_record(
            record, session_sha256="0" * 64, volume=volume)

    missing = BlockVolume()
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.canonicalise_production_record(
            record, session_sha256=session["session_sha256"], volume=missing)

    altered = promotion.to_block_volume()
    observation = altered.blocks[(0, 0, 0)]
    altered.set((0, 0, 0), BlockObservation(
        state=observation.state, confidence=0.0, kind=observation.kind,
        frame_ids=list(observation.frame_ids)))
    with pytest.raises(record_mod.ProductionEnvelopeError):
        record_mod.canonicalise_production_record(
            record, session_sha256=session["session_sha256"], volume=altered)


# ---------------------------------------------------------------------------
# Schema-3 project persistence and resume-export
# ---------------------------------------------------------------------------

def test_promoted_schema3_project_round_trips_and_resume_exports(monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    promotion = _promotion()
    project = _promoted_project(session, promotion)
    path = tmp_path / "durable.mcscan"

    store_mod.publish_project_no_clobber(project, path)
    first_bytes = path.read_bytes()
    loaded = store_mod.load_project(path)

    assert loaded.schema_version == SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION
    assert loaded.image_derived_session == session
    assert loaded.image_derived_production \
        == project.image_derived_production
    expected = promotion.to_block_volume()
    assert loaded.volume.to_dict() == expected.to_dict()
    for coordinate, observation in expected.items():
        stored = loaded.volume.blocks[coordinate]
        assert stored.state == observation.state
        assert stored.confidence == observation.confidence
        assert tuple(stored.frame_ids) == tuple(observation.frame_ids)

    # The replaced schema-3 file is the durable terminal: neither the
    # replace-based writer nor a no-clobber publish may overwrite it.
    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.save_project(loaded, path)
    assert path.read_bytes() == first_bytes
    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.publish_project_no_clobber(loaded, path)
    assert path.read_bytes() == first_bytes

    artifact = tmp_path / "durable.litematic"
    LitematicExporter().export(loaded.volume, artifact, name="durable")
    replayed = record_mod.production_envelope_from_record(
        loaded.image_derived_production)
    verified = image_derived_delivery.verify_image_derived_litematic(
        replayed.promotion, artifact)
    assert not isinstance(verified, image_derived_delivery.DeliveryRefusal)
    artifact_sha256, block_count = verified
    assert block_count == len(promotion.cells)
    assert len(artifact_sha256) == 64


def test_a_non_promoted_schema3_project_stays_zero_blocks_and_cannot_export(
        monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    project = _refusal_project(session)
    path = tmp_path / "review.mcscan"

    store_mod.publish_project_no_clobber(project, path)
    loaded = store_mod.load_project(path)

    assert loaded.volume.to_dict()["blocks"] == []
    with pytest.raises(LitematicExportError):
        LitematicExporter().export(
            loaded.volume, tmp_path / "review.litematic", name="review")


def test_the_export_cli_refuses_a_zero_block_schema3_project(
        monkeypatch, tmp_path):
    from mc_photogrammetry import cli

    session = _session_record(monkeypatch)
    path = tmp_path / "review.mcscan"
    artifact = tmp_path / "review.litematic"
    store_mod.publish_project_no_clobber(_refusal_project(session), path)
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "export", str(path), str(artifact)])

    with pytest.raises(SystemExit, match="export refused"):
        cli.main()

    assert not artifact.exists()


def test_the_export_cli_writes_a_promoted_schema3_volume(monkeypatch, tmp_path):
    from mc_photogrammetry import cli

    session = _session_record(monkeypatch)
    promotion = _promotion()
    path = tmp_path / "durable.mcscan"
    artifact = tmp_path / "durable.litematic"
    store_mod.publish_project_no_clobber(
        _promoted_project(session, promotion), path)
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "export", str(path), str(artifact)])

    assert cli.main() == 0
    assert artifact.exists()


def test_the_export_cli_refuses_a_schema3_world_format(monkeypatch, tmp_path):
    from mc_photogrammetry import cli

    session = _session_record(monkeypatch)
    path = tmp_path / "durable.mcscan"
    world = tmp_path / "durable.zip"
    store_mod.publish_project_no_clobber(
        _promoted_project(session, _promotion()), path)
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "export", str(path), str(world),
        "--format", "world"])

    with pytest.raises(SystemExit, match="export refused"):
        cli.main()

    assert not world.exists()


def test_the_export_cli_refuses_a_sparse_oversized_schema3_promotion(
        monkeypatch, tmp_path):
    from mc_photogrammetry import cli

    session = _session_record(monkeypatch)
    coordinates = ((0, 0, 0), (0, 0, 200), (0, 200, 0), (200, 0, 0))
    promotion = _promotion(coordinates)
    path = tmp_path / "sparse.mcscan"
    artifact = tmp_path / "sparse.litematic"
    store_mod.publish_project_no_clobber(
        _promoted_project(session, promotion), path)
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "export", str(path), str(artifact)])

    with pytest.raises(SystemExit, match="export refused"):
        cli.main()

    assert not artifact.exists()


def test_the_export_cli_refuses_an_occupied_schema3_target(monkeypatch, tmp_path):
    from mc_photogrammetry import cli

    session = _session_record(monkeypatch)
    path = tmp_path / "durable.mcscan"
    artifact = tmp_path / "durable.litematic"
    store_mod.publish_project_no_clobber(
        _promoted_project(session, _promotion()), path)
    artifact.write_text("prior bytes", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "export", str(path), str(artifact)])

    with pytest.raises(SystemExit, match="export refused"):
        cli.main()

    assert artifact.read_text(encoding="utf-8") == "prior bytes"


def test_the_export_cli_refuses_a_malformed_exporter_readback(
        monkeypatch, tmp_path):
    from mc_photogrammetry import cli
    from mc_photogrammetry.export import litematic as litematic_mod

    session = _session_record(monkeypatch)
    path = tmp_path / "durable.mcscan"
    artifact = tmp_path / "durable.litematic"
    store_mod.publish_project_no_clobber(
        _promoted_project(session, _promotion()), path)

    def malformed_export(self, volume, destination, **kwargs):
        Path(destination).write_bytes(b"not a litematic")

    monkeypatch.setattr(
        litematic_mod.LitematicExporter, "export", malformed_export)
    monkeypatch.setattr(sys, "argv", [
        "mc-photogrammetry", "export", str(path), str(artifact)])

    with pytest.raises(SystemExit, match="export refused"):
        cli.main()

    assert not artifact.exists()
    assert [item for item in tmp_path.iterdir()
            if item.suffix == ".tmp"] == []


def test_a_schema3_project_refuses_a_mismatched_volume(monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    promotion = _promotion()
    altered = promotion.to_block_volume()
    altered.set((9, 9, 9), BlockObservation(BlockState("minecraft:stone")))
    project = _promoted_project(session, promotion, volume=altered)
    path = tmp_path / "mismatch.mcscan"

    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.publish_project_no_clobber(project, path)
    assert not path.exists()


def test_a_schema3_project_refuses_a_mismatched_session_binding(
        monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    promotion = _promotion()
    project = _promoted_project(session, promotion)
    project.image_derived_production["session_sha256"] = "0" * 64
    path = tmp_path / "binding.mcscan"

    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.publish_project_no_clobber(project, path)
    assert not path.exists()


def test_a_tampered_schema3_file_is_refused_on_load(monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    promotion = _promotion()
    path = tmp_path / "tampered.mcscan"
    store_mod.publish_project_no_clobber(
        _promoted_project(session, promotion), path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    payload["image_derived_session"]["session_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.load_project(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_derived_session"]["session_sha256"] = session["session_sha256"]
    payload["volume"]["blocks"].append({
        "coordinate": [9, 9, 9],
        "observation": {"state": {"name": "minecraft:stone", "properties": {}},
                        "confidence": 1.0, "kind": "observed",
                        "frame_ids": ["f0"]},
    })
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.load_project(path)


# ---------------------------------------------------------------------------
# No-clobber publication
# ---------------------------------------------------------------------------

def test_publish_refuses_an_existing_target_and_preserves_bytes(
        monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    project = _promoted_project(session, _promotion())
    target = tmp_path / "existing.mcscan"
    target.write_text("prior bytes", encoding="utf-8")

    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.publish_project_no_clobber(project, target)

    assert target.read_text(encoding="utf-8") == "prior bytes"


def test_the_replace_writer_refuses_a_schema3_payload_and_preserves_bytes(
        monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    project = _promoted_project(session, _promotion())
    target = tmp_path / "durable.mcscan"
    target.write_text("predecessor bytes", encoding="utf-8")

    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.save_project(project, target)

    assert target.read_text(encoding="utf-8") == "predecessor bytes"
    assert not (tmp_path / "durable.mcscan.tmp").exists()


def test_publish_no_clobber_writes_a_loadable_project_exactly_once(
        monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    promotion = _promotion()
    project = _promoted_project(session, promotion)
    target = tmp_path / "published.mcscan"

    store_mod.publish_project_no_clobber(project, target)

    loaded = store_mod.load_project(target)
    assert loaded.volume.to_dict() == promotion.to_block_volume().to_dict()
    first_bytes = target.read_bytes()

    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.publish_project_no_clobber(project, target)
    assert target.read_bytes() == first_bytes
    assert list(tmp_path.glob("*.staging")) == []


def test_publish_race_preserves_the_racer_bytes(monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    project = _promoted_project(session, _promotion())
    target = tmp_path / "race.mcscan"
    real_link = store_mod.os.link

    def racing_link(source, destination):
        Path(destination).write_text("racer bytes", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(store_mod.os, "link", racing_link)

    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.publish_project_no_clobber(project, target)

    assert target.read_text(encoding="utf-8") == "racer bytes"


def test_publish_failure_leaves_no_target_and_no_staging(monkeypatch, tmp_path):
    session = _session_record(monkeypatch)
    project = _promoted_project(session, _promotion())
    target = tmp_path / "failure.mcscan"

    def failing_link(source, destination):
        raise OSError("link unavailable")

    monkeypatch.setattr(store_mod.os, "link", failing_link)

    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.publish_project_no_clobber(project, target)

    assert not target.exists()
    assert list(tmp_path.glob("*.staging")) == []


def test_target_and_checkpoint_helpers(monkeypatch, tmp_path):
    target = tmp_path / "target.mcscan"
    assert record_mod.target_refusal(target) is None

    target.write_text("x", encoding="utf-8")
    refusal = record_mod.target_refusal(target)
    assert refusal is not None
    assert refusal.reason == "invalid_input"
    assert refusal.detail["reason"] == "target_exists"

    checkpoint = record_mod.checkpoint_path(tmp_path / "fresh.mcscan")
    assert checkpoint.parent.is_dir()
    checkpoint.write_text("staged", encoding="utf-8")
    record_mod.cleanup_checkpoint(checkpoint)
    assert not checkpoint.parent.exists()


def test_target_refusal_and_publish_use_an_occupied_entry_check(
        monkeypatch, tmp_path):
    """A dangling link is an occupied entry even though ``Path.exists`` is False.

    This is the deterministic guard for the fix; a real dangling symlink is also
    exercised in ``test_a_dangling_symlink_target_is_refused_before_entry_calls``,
    which skips when the runner lacks the symlink privilege.
    """
    target = tmp_path / "occupied.mcscan"
    session = _session_record(monkeypatch)
    project = _promoted_project(session, _promotion())
    real_lexists = store_mod.os.path.lexists

    # Model a dangling symlink: no stat target, but the entry is present.
    monkeypatch.setattr(
        target.__class__, "exists", lambda self: False, raising=False)
    monkeypatch.setattr(
        store_mod.os.path, "lexists",
        lambda path: True if Path(path) == target else real_lexists(path))

    refusal = record_mod.target_refusal(target)
    assert refusal is not None
    assert refusal.detail["reason"] == "target_exists"

    with pytest.raises(store_mod.ProjectStoreError):
        store_mod.publish_project_no_clobber(project, target)

    assert list(tmp_path.iterdir()) == []


def test_promotion_result_from_record_replays_and_refuses_tampering(monkeypatch):
    promotion = _promotion()
    record = promotion.to_record()

    replayed = state_promotion.promotion_result_from_record(record)
    assert replayed.to_record() == record
    assert replayed.promotion_sha256 == promotion.promotion_sha256

    stale = json.loads(json.dumps(record))
    stale["cells"][0]["confidence"] = 0.123456
    with pytest.raises(ValueError):
        state_promotion.promotion_result_from_record(stale)

    extra = json.loads(json.dumps(record))
    extra["mystery"] = 1
    with pytest.raises(ValueError):
        state_promotion.promotion_result_from_record(extra)
