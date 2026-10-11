"""Synthetic controls for the bounded M5 image-derived litematic delivery adapter.

Every promotion here is built in memory: a self-consistent synthetic
``PlaneVolumeResult`` is promoted through the real M5 ``promote_plane_states``
adapter over a synthetic ``FaceStateProposalResult`` produced by the real
``propose_face_states`` producer.  No image, manifest, camera, F3 reference,
project store or CLI/UI surface is read, and no fixture litematic is consumed.
The only real filesystem objects are pytest's own ``tmp_path`` outputs, which the
suite writes and then loads back through the accepted ``nbtlib``/``validation``
readers to prove the round trip.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export import litematic as litematic_module
from mc_photogrammetry.export.compare import compare_litematics, load_litematic
from mc_photogrammetry.export.litematic import LitematicExportError, LitematicExporter
from mc_photogrammetry.export.validation import validate_litematic
from mc_photogrammetry.pipeline import image_derived_delivery as delivery
from mc_photogrammetry.reconstruction import appearance_state_proposals as ap
from mc_photogrammetry.reconstruction import plane_volume as pv
from mc_photogrammetry.reconstruction import state_promotion as promotion
from mc_photogrammetry.reconstruction.face_observation import NORMAL_OF_SLOT, FaceObservation

SOURCE = Path(delivery.__file__)

STONE = PaletteColor(
    BlockState("minecraft:stone"),
    (125, 125, 125),
    face_rgbs=((125, 125, 125),),
    family="cube",
    face_map=tuple((slot, (125, 125, 125)) for slot in FACE_SLOTS),
    full_cube=True,
)
DIRT = PaletteColor(
    BlockState("minecraft:dirt"),
    (134, 96, 67),
    face_rgbs=((134, 96, 67),),
    family="cube",
    face_map=tuple((slot, (134, 96, 67)) for slot in FACE_SLOTS),
    full_cube=True,
)
OAK_LOG = PaletteColor(
    BlockState("minecraft:oak_log", (("axis", "y"),)),
    (162, 130, 78),
    face_rgbs=((162, 130, 78),),
    family="cube",
    face_map=tuple((slot, (162, 130, 78)) for slot in FACE_SLOTS),
    full_cube=True,
)


# ---------------------------------------------------------------------------
# Synthetic promotion construction (in memory only)
# ---------------------------------------------------------------------------

def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _observation(cell, *, frame_id: str = "f0", value: float = 125.0,
                 provenance: tuple[str, ...] = ("cam/a",)) -> FaceObservation:
    return FaceObservation(
        cell=cell,
        slot="north",
        frame_id=frame_id,
        normal=NORMAL_OF_SLOT["north"],
        patch=np.full((4, 4, 3), value, dtype=np.float32),
        usable_coverage=1.0,
        support_frames=(frame_id,),
        provenance=provenance,
    )


def _claim(coordinate, index: int) -> pv.CellClaim:
    return pv.CellClaim.build(
        frame="f0", plane_index=0, reading_axis=0, absolute_quadrant=0,
        source_region_id=_h(f"region{index}"),
        reading_identity_sha256=_h(f"reading{index}"),
        support_sha256=_h(f"support{index}"),
        face_sha256=_h(f"face{index}"),
        row=0, column=index,
        world_normal_axis=0, world_normal_sign=-1, outward_sign=-1,
        face=coordinate[0], normal_block=coordinate[0],
        tangent_u_axis=1, lateral_u=coordinate[1],
        tangent_v_axis=2, lateral_v=coordinate[2],
        coordinate=coordinate,
    )


def _binding(claim: pv.CellClaim, *, host: bool = False, omitted: int = 0) -> dict:
    return {
        "frame": claim.frame,
        "plane_index": claim.plane_index,
        "reading_axis": claim.reading_axis,
        "reading_identity_sha256": claim.reading_identity_sha256,
        "support_sha256": claim.support_sha256,
        "face_sha256": claim.face_sha256,
        "source_region_id": claim.source_region_id,
        "alternative_distance": [0, 1],
        "phase_anchor_sha256": _h("anchor"),
        "support_cells": 1 + (omitted if host else 0),
        "supported_cells": 1,
        "unknown_cells": omitted if host else 0,
    }


def _volume(coordinates, *, omitted: int = 0) -> pv.PlaneVolumeResult:
    cells = []
    bindings = []
    for index, coordinate in enumerate(coordinates):
        claim = _claim(coordinate, index)
        cells.append(pv.SupportedCellObservation.build(
            coordinate=coordinate, claims=(claim,), frame_ids=(claim.frame,),
            conflict_kind=None))
        bindings.append(_binding(claim, host=(index == 0), omitted=omitted))
    omitted_records = tuple(
        pv.OmittedCell.build(
            frame="f0", plane_index=0, reading_axis=0, row=0, column=100 + index,
            state="unknown", evidence="outside_frame",
            source_region_id=_h(f"oregion{index}"),
            reading_identity_sha256=_h(f"oreading{index}"),
            support_sha256=_h(f"osupport{index}"),
            face_sha256=_h(f"oface{index}"))
        for index in range(omitted))
    if cells:
        low = tuple(min(cell.coordinate[axis] for cell in cells) for axis in range(3))
        high = tuple(max(cell.coordinate[axis] for cell in cells) for axis in range(3))
        dense = 1
        for axis in range(3):
            dense *= high[axis] - low[axis] + 1
        bounds = (low, high)
        offset = low
    else:
        bounds = None
        dense = 0
        offset = None
    work = pv._make_work({
        "faces": len(bindings),
        "support_cells": sum(binding["support_cells"] for binding in bindings),
        "contributions": sum(len(cell.claims) for cell in cells),
        "coordinates": len(cells),
        "conflicts": 0,
        "omitted_unknown": len(omitted_records),
        "dense_bounds_cells": dense,
    })
    return pv.PlaneVolumeResult.build(
        schema_version=pv.V3_VOLUME_SCHEMA,
        producer_version=pv.V3_VOLUME_PRODUCER_VERSION,
        gauge_state=pv.V3_VOLUME_GAUGE_STATE,
        gauge_axes=(0, 1, 2),
        alpha=((0, 1), (0, 1), (0, 1)),
        placement_sha256=_h("placement"),
        support_bindings=tuple(sorted(bindings, key=pv._binding_sort_key)),
        cells=tuple(cells),
        conflicts=(),
        omitted=omitted_records,
        cell_bounds=bounds,
        dense_bounds_cells=dense,
        normalization_offset=offset,
        export_bounds=None,
        complete=False,
        work=work,
    )


def _proposals(coordinates, palette=(STONE, DIRT)) -> ap.FaceStateProposalResult:
    observations = tuple(_observation(cell) for cell in coordinates)
    return ap.propose_face_states(
        observations, palette,
        max_cost=1e9, min_margin=0.0, max_observations=1000,
        max_candidate_evaluations=100000, max_payload_bytes=10000000)


def _budget(**overrides: int) -> promotion.PromotionBudget:
    caps = dict(max_cells=16, max_proposals=16, max_payload_bytes=1_000_000)
    caps.update(overrides)
    return promotion.make_promotion_budget(**caps)


def _promote(coordinates=((0, 0, 0), (1, 0, 0)), *, palette=(STONE, DIRT),
             omitted: int = 0, allow_omitted_unknown: bool = False,
             ) -> promotion.PromotionResult:
    result = promotion.promote_plane_states(
        _volume(coordinates, omitted=omitted), _proposals(coordinates, palette),
        budget=_budget(), allow_omitted_unknown=allow_omitted_unknown)
    assert isinstance(result, promotion.PromotionResult), result
    return result


def _clone(record: object, **changes: object) -> object:
    clone = object.__new__(type(record))
    for field in dataclasses.fields(record):  # type: ignore[arg-type]
        object.__setattr__(clone, field.name, getattr(record, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


def _reseal_promotion(result: promotion.PromotionResult,
                      **changes: object) -> promotion.PromotionResult:
    """Clone a promotion past its constructor and recompute its own digest.

    The clone is self-consistent at the record level (its ``promotion_sha256``
    replays from its own payload) but carries a field the accepted constructor or
    a nested block digest rejects.  This models the hostile record the delivery
    seam must refuse before it touches the filesystem.
    """
    clone = _clone(result, **changes)
    payload = promotion._result_payload(clone)
    encoded = promotion._canonical_json(payload)
    object.__setattr__(clone, "payload_bytes", len(encoded.encode("utf-8")))
    object.__setattr__(clone, "payload_sha256",
                       hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    object.__setattr__(clone, "promotion_sha256", promotion._sha256({
        **payload,
        "payload_sha256": clone.payload_sha256,
        "max_payload_bytes": clone.max_payload_bytes,
    }))
    return clone  # type: ignore[return-value]


def _reforge_block(block: promotion.PromotedBlock, **changes: object):
    """Clone one promoted block past its constructor with the given changes."""
    return _clone(block, **changes)


def _destination(tmp_path: Path, name: str = "build.litematic") -> Path:
    return tmp_path / name


def _assert_refusal(result: object, reason: str) -> delivery.DeliveryRefusal:
    assert isinstance(result, delivery.DeliveryRefusal), result
    assert result.reason == reason, result
    assert result.detail_sha256 == hashlib.sha256(
        delivery._canonical_json(result.detail).encode("utf-8")).hexdigest()
    assert result.refusal_sha256 == delivery._sha256(
        delivery._refusal_payload(result))
    assert "cells" not in result.detail and "blocks" not in result.detail
    return result


def _assert_no_output(tmp_path: Path) -> None:
    assert list(tmp_path.iterdir()) == [], list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def happy_promotion() -> promotion.PromotionResult:
    return _promote()


@pytest.fixture(scope="module")
def omitted_promotion() -> promotion.PromotionResult:
    return _promote(omitted=1, allow_omitted_unknown=True)


# ---------------------------------------------------------------------------
# Happy path: real export and real load round trip
# ---------------------------------------------------------------------------


def test_happy_path_exports_a_loadable_litematic(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="Image derived build",
        description="Reconstructed from images")

    assert isinstance(result, delivery.DeliveryResult), result
    assert result.promotion_sha256 == happy_promotion.promotion_sha256
    assert result.block_count == len(happy_promotion.cells) == 2
    assert result.bytes_written > 0
    assert result.bytes_written == destination.stat().st_size
    assert result.output_path == destination
    assert destination.exists() and destination.stat().st_size > 0

    validation = validate_litematic(destination)
    assert validation["dimensions"] == (2, 1, 1)
    assert validation["position"] == (0, 0, 0)

    loaded = load_litematic(destination)
    assert loaded.size == (2, 1, 1)
    assert loaded.blocks == {
        (0, 0, 0): BlockState("minecraft:stone"),
        (1, 0, 0): BlockState("minecraft:stone"),
    }
    comparison = compare_litematics(destination, destination)
    assert comparison.exact_coordinate_matches == 2
    assert comparison.exact_state_matches == 2


def test_written_artifact_carries_the_explicit_name_and_description(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    import nbtlib

    destination = _destination(tmp_path)
    delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="Explicit name",
        description="Explicit description")
    metadata = nbtlib.load(destination, gzipped=True)["Metadata"]
    assert str(metadata["Name"]) == "Explicit name"
    assert str(metadata["Description"]) == "Explicit description"
    assert str(metadata["Author"]) == "mc-photogrammetry"


def test_delivered_blocks_match_the_promotion_to_block_volume(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="parity", description="parity")
    loaded = load_litematic(destination)
    volume = happy_promotion.to_block_volume()
    for coordinate, observation in volume.items():
        assert loaded.blocks[coordinate] == observation.state
    assert set(loaded.blocks) == set(volume.blocks)


def test_state_properties_round_trip_through_the_artifact(tmp_path: Path) -> None:
    promotion_result = _promote(((0, 0, 0),), palette=(OAK_LOG, STONE))
    expected = {block.coordinate: (block.state_name, block.state_properties)
                for block in promotion_result.cells}
    destination = _destination(tmp_path, "log.litematic")
    result = delivery.deliver_image_derived_litematic(
        promotion_result, destination, name="log", description="log")
    assert isinstance(result, delivery.DeliveryResult), result
    loaded = load_litematic(destination)
    assert {coordinate: (state.name, state.properties)
            for coordinate, state in loaded.blocks.items()} == expected


def test_component_zero_survives_the_trip(tmp_path: Path) -> None:
    """A region whose tight lower bound is not the world origin.

    ``load_litematic`` reports region-local coordinates because the exporter pins
    every region at ``Position=(0,0,0)``; the delivery seam resolves a block by
    adding the tight lower bound back, so the artifact must land at the original
    negative/positive coordinate.
    """
    promotion_result = _promote(((-4, 61, 7),), palette=(DIRT, STONE))
    expected = {block.coordinate: (block.state_name, block.state_properties)
                for block in promotion_result.cells}
    destination = _destination(tmp_path, "offset.litematic")
    result = delivery.deliver_image_derived_litematic(
        promotion_result, destination, name="offset", description="offset")
    assert isinstance(result, delivery.DeliveryResult), result
    loaded = load_litematic(destination)
    assert loaded.size == (1, 1, 1)
    assert set(loaded.blocks) == {(0, 0, 0)}
    assert (loaded.blocks[(0, 0, 0)].name, loaded.blocks[(0, 0, 0)].properties) \
        == expected[(-4, 61, 7)]
    verified = delivery.verify_image_derived_litematic(promotion_result, destination)
    assert not isinstance(verified, delivery.DeliveryRefusal)


# ---------------------------------------------------------------------------
# Result record contract
# ---------------------------------------------------------------------------


def test_result_record_is_immutable_and_destination_independent(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="record", description="record")
    assert isinstance(result, delivery.DeliveryResult), result

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.block_count = 99  # type: ignore[misc]

    assert {field.name for field in dataclasses.fields(result)} == {
        "promotion_sha256", "artifact_sha256", "block_count", "bytes_written",
        "output_path"}
    record = result.to_record()
    assert set(record) == {"promotion_sha256", "artifact_sha256", "block_count",
                           "bytes_written"}
    assert record["promotion_sha256"] == happy_promotion.promotion_sha256
    assert record["artifact_sha256"] == result.artifact_sha256
    assert json.loads(json.dumps(record)) == record
    assert "output_path" not in json.dumps(record)
    assert str(destination) not in json.dumps(record)


def test_result_retains_no_source_reference(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="retention", description="retention")
    assert isinstance(result, delivery.DeliveryResult), result

    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        assert value is not happy_promotion
        assert not isinstance(value, promotion.PromotionResult)
        assert not isinstance(value, promotion.PromotedBlock)
        assert not isinstance(value, LitematicExporter)
        if field.name != "output_path":
            assert not isinstance(value, Path)
    assert not hasattr(result, "promotion")
    assert not hasattr(result, "exporter")
    assert not hasattr(result, "volume")


# ---------------------------------------------------------------------------
# Overwrite policy
# ---------------------------------------------------------------------------


def test_existing_destination_is_refused_without_overwrite(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    destination.write_bytes(b"pre-existing destination sentinel")

    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="refuse", description="refuse")

    _assert_refusal(result, "destination_exists")
    assert destination.read_bytes() == b"pre-existing destination sentinel"
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]


def test_overwrite_opt_in_replaces_the_destination(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    volume = happy_promotion.to_block_volume()
    LitematicExporter().export(volume, destination, name="stale", description="stale")
    stale = destination.read_bytes()

    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="fresh", description="fresh",
        overwrite=True)

    assert isinstance(result, delivery.DeliveryResult), result
    assert destination.read_bytes() != stale
    import nbtlib

    metadata = nbtlib.load(destination, gzipped=True)["Metadata"]
    assert str(metadata["Name"]) == "fresh"
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]


def test_a_directory_destination_is_refused(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = tmp_path / "adir.litematic"
    destination.mkdir()
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="dir", description="dir", overwrite=True)
    _assert_refusal(result, "invalid_destination")
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


class _RacingDestinationExporter(LitematicExporter):
    """Write the real artifact, then create the destination inside the race window.

    The delivery seam validates the destination *before* it stages the temporary
    and exports, so creating the competing entry here lands exactly between that
    validation and the final no-clobber publication.  The exporter itself is the
    real accepted exporter; only the competing write is injected.
    """

    def __init__(self, competing: Path, payload: bytes) -> None:
        super().__init__()
        self.competing = competing
        self.payload = payload
        self.calls = 0

    def export(self, volume, destination, **options):  # type: ignore[no-untyped-def]
        self.calls += 1
        super().export(volume, destination, **options)
        self.competing.write_bytes(self.payload)


def test_a_destination_racing_into_place_is_refused_and_its_bytes_preserved(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    """A destination created after validation must never be replaced.

    The competing writer runs during the export, after the destination was
    validated as absent and before the verified temporary is published, so only an
    atomic no-clobber placement can keep the competing bytes.  The refusal must be
    typed and the seam must still clean up its own temporary.
    """
    destination = _destination(tmp_path)
    competing = b"competing writer bytes that must survive the race"
    exporter = _RacingDestinationExporter(destination, competing)

    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="race", description="race",
        exporter=exporter)

    assert exporter.calls == 1
    _assert_refusal(result, "destination_exists")
    assert result.detail["field"] == "destination"
    assert destination.read_bytes() == competing
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]


def test_no_overwrite_publication_leaves_no_temporary(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    """The no-clobber link leaves a second name for the staged file; it is cleaned."""
    destination = _destination(tmp_path)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="linked", description="linked")

    assert isinstance(result, delivery.DeliveryResult), result
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]
    verified = delivery.verify_image_derived_litematic(happy_promotion, destination)
    assert not isinstance(verified, delivery.DeliveryRefusal)
    assert verified == (result.artifact_sha256, result.block_count)


# ---------------------------------------------------------------------------
# Eligibility gates
# ---------------------------------------------------------------------------


def test_review_ready_false_is_refused_before_filesystem(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    tampered = _reseal_promotion(happy_promotion, export_ready=False)
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "promotion_not_export_ready")
    _assert_no_output(tmp_path)


def test_ineligible_exact_state_is_refused_before_filesystem(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    tampered = _reseal_promotion(happy_promotion, exact_state_eligible=False)
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "promotion_not_exact_state_eligible")
    _assert_no_output(tmp_path)


def test_omitted_unknown_is_refused_even_when_exact_state_claims_true(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    tampered = _reseal_promotion(happy_promotion, omitted_unknown=2)
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "promotion_omitted_unknown")
    _assert_no_output(tmp_path)


def test_an_opt_in_omitted_promotion_is_not_exact_state_eligible(
    omitted_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    """The upstream opt-in produces a real ineligible record for this gate."""
    assert omitted_promotion.omitted_unknown == 1
    assert omitted_promotion.export_ready is True
    assert omitted_promotion.exact_state_eligible is False
    result = delivery.deliver_image_derived_litematic(
        omitted_promotion, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "promotion_not_exact_state_eligible")
    _assert_no_output(tmp_path)


def test_an_empty_promotion_is_refused(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    tampered = _reseal_promotion(happy_promotion, cells=())
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "promotion_empty")
    _assert_no_output(tmp_path)


# ---------------------------------------------------------------------------
# Preserved open metadata and tampered records
# ---------------------------------------------------------------------------


def test_present_export_bounds_are_refused(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    tampered = _reseal_promotion(happy_promotion, export_bounds=((0, 0, 0), (9, 9, 9)))
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    assert result.detail["field"] == "export_bounds"
    _assert_no_output(tmp_path)


def test_resolved_scale_state_is_refused(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    tampered = _reseal_promotion(happy_promotion, scale_state="resolved")
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    assert result.detail["field"] == "scale_state"
    _assert_no_output(tmp_path)


def test_resolved_normalization_state_is_refused(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    tampered = _reseal_promotion(happy_promotion, normalization_state="metric")
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    assert result.detail["field"] == "normalization_state"
    _assert_no_output(tmp_path)


def test_tampered_block_digest_is_refused_before_filesystem(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    block = happy_promotion.cells[0]
    forged_block = _reforge_block(block, block_sha256=_h("forged"))
    tampered = _reseal_promotion(happy_promotion, cells=(forged_block,)
                                 + happy_promotion.cells[1:])
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    assert result.detail["field"] == "cells"
    _assert_no_output(tmp_path)


def test_tampered_nested_block_evidence_is_refused_before_filesystem(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    """A forged nested field resealed at every enclosing level is still refused."""
    block = promotion.PromotedBlock.build(
        coordinate=happy_promotion.cells[0].coordinate,
        state_name=happy_promotion.cells[0].state_name,
        state_properties=(("facing", "north"), ("waterlogged", "false")),
        confidence=happy_promotion.cells[0].confidence,
        frame_ids=happy_promotion.cells[0].frame_ids,
        provenance=happy_promotion.cells[0].provenance,
        observation_digests=happy_promotion.cells[0].observation_digests,
        proposal_digest=happy_promotion.cells[0].proposal_digest,
        evidence_digest=happy_promotion.cells[0].evidence_digest)
    # Reorder the properties past the constructor: the block digest is then
    # recomputed so the record is self-consistent but the accepted constructor
    # rejects the non-canonical order.
    object.__setattr__(block, "state_properties",
                       (("waterlogged", "false"), ("facing", "north")))
    object.__setattr__(block, "block_sha256",
                       promotion._sha256(promotion._block_payload(block)))
    tampered = _reseal_promotion(happy_promotion, cells=(block,)
                                 + happy_promotion.cells[1:])
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    _assert_no_output(tmp_path)


def test_tampered_promotion_digest_is_refused_before_filesystem(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    forged_block = promotion.PromotedBlock.build(
        coordinate=(0, 0, 0), state_name="minecraft:diamond_block",
        state_properties=(), confidence=happy_promotion.cells[0].confidence,
        frame_ids=happy_promotion.cells[0].frame_ids,
        provenance=happy_promotion.cells[0].provenance,
        observation_digests=happy_promotion.cells[0].observation_digests,
        proposal_digest=happy_promotion.cells[0].proposal_digest,
        evidence_digest=happy_promotion.cells[0].evidence_digest)
    # Swap the whole block but keep the original record digest: the nested block
    # digest replays, so only the enclosing promotion digest can catch this.
    tampered = _clone(happy_promotion, cells=(forged_block,)
                      + happy_promotion.cells[1:])
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    assert result.detail["field"] == "promotion_digest"
    _assert_no_output(tmp_path)


def test_non_canonical_cell_order_is_refused_before_filesystem(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    tampered = _reseal_promotion(
        happy_promotion, cells=tuple(reversed(happy_promotion.cells)))
    result = delivery.deliver_image_derived_litematic(
        tampered, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    assert result.detail["field"] == "cell_order"
    _assert_no_output(tmp_path)


def test_astronomically_sparse_promotion_is_refused_before_export(
    tmp_path: Path, monkeypatch,
) -> None:
    promotion_result = _promote(((0, 0, 0), (1_000_000, 0, 0)))
    monkeypatch.setattr(delivery, "DELIVERY_MAX_DENSE_CELLS", 1_000)
    result = delivery.deliver_image_derived_litematic(
        promotion_result, _destination(tmp_path), name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    assert result.detail["field"] == "dense_bounds"
    assert result.detail["limit"] == 1_000
    assert result.detail["observed"] == 1_000_001
    _assert_no_output(tmp_path)


def test_a_dense_span_at_the_cap_is_still_delivered(
    happy_promotion: promotion.PromotionResult, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(delivery, "DELIVERY_MAX_DENSE_CELLS", 2)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="cap", description="cap")
    assert isinstance(result, delivery.DeliveryResult), result


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "promotion", 7, object(), ()])
def test_a_non_promotion_is_refused(value: object, tmp_path: Path) -> None:
    result = delivery.deliver_image_derived_litematic(
        value, _destination(tmp_path), name="n", description="d")  # type: ignore[arg-type]
    _assert_refusal(result, "invalid_input")
    assert result.detail["field"] == "promotion"
    _assert_no_output(tmp_path)


def test_a_review_refusal_is_not_a_deliverable_promotion(tmp_path: Path) -> None:
    """The upstream review/refusal outcome carries no blocks and cannot deliver."""
    review = promotion.promote_plane_states(
        _volume(((0, 0, 0),), omitted=1), _proposals(((0, 0, 0),)),
        budget=_budget())
    assert isinstance(review, promotion.PromotionRefusal), review
    result = delivery.deliver_image_derived_litematic(
        review, _destination(tmp_path), name="n", description="d")  # type: ignore[arg-type]
    _assert_refusal(result, "invalid_input")
    assert result.detail["field"] == "promotion"
    _assert_no_output(tmp_path)


def test_promotion_validation_precedes_the_destination_check(tmp_path: Path) -> None:
    """A tampered promotion is named even when the destination also conflicts."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    tampered = _reseal_promotion(_promote(), export_bounds=((0, 0, 0), (9, 9, 9)))
    result = delivery.deliver_image_derived_litematic(
        tampered, blocker / "build.litematic", name="n", description="d")
    _assert_refusal(result, "invalid_promotion")
    assert result.detail["field"] == "export_bounds"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["blocker"]


@pytest.mark.parametrize("value", [None, "path.litematic", 7, object()])
def test_a_non_path_destination_is_refused(
    happy_promotion: promotion.PromotionResult, value: object, tmp_path: Path,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, value, name="n", description="d")  # type: ignore[arg-type]
    _assert_refusal(result, "invalid_destination")
    _assert_no_output(tmp_path)


def test_a_destination_under_a_file_parent_is_refused(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, blocker / "build.litematic", name="n", description="d")
    _assert_refusal(result, "invalid_destination")
    assert result.detail["field"] == "destination_parent"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["blocker"]


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_an_empty_name_is_refused(
    happy_promotion: promotion.PromotionResult, value: str, tmp_path: Path,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name=value, description="d")
    _assert_refusal(result, "invalid_name")
    _assert_no_output(tmp_path)


@pytest.mark.parametrize("value", [None, 7, b"name", "x" * 129, "line\nbreak"])
def test_a_malformed_name_is_refused(
    happy_promotion: promotion.PromotionResult, value: object, tmp_path: Path,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name=value, description="d")  # type: ignore[arg-type]
    _assert_refusal(result, "invalid_name")
    _assert_no_output(tmp_path)


@pytest.mark.parametrize("value", ["", "   ", None, 7, b"desc", "x" * 513])
def test_a_malformed_description_is_refused(
    happy_promotion: promotion.PromotionResult, value: object, tmp_path: Path,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description=value)  # type: ignore[arg-type]
    _assert_refusal(result, "invalid_description")
    _assert_no_output(tmp_path)


def test_a_non_boolean_overwrite_is_refused(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description="d",
        overwrite="yes")  # type: ignore[arg-type]
    _assert_refusal(result, "invalid_input")
    assert result.detail["field"] == "overwrite"
    _assert_no_output(tmp_path)


@pytest.mark.parametrize("value", ["exporter", object(), 7])
def test_a_non_exporter_is_refused(
    happy_promotion: promotion.PromotionResult, value: object, tmp_path: Path,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description="d",
        exporter=value)  # type: ignore[arg-type]
    _assert_refusal(result, "invalid_input")
    assert result.detail["field"] == "exporter"
    _assert_no_output(tmp_path)


def test_the_default_exporter_is_the_accepted_litematic_exporter(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="n", description="d", exporter=None)
    assert isinstance(result, delivery.DeliveryResult), result
    assert destination.exists()


def test_a_destination_without_a_name_is_refused(
    happy_promotion: promotion.PromotionResult,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, Path(""), name="n", description="d")
    _assert_refusal(result, "invalid_destination")
    assert result.detail["field"] == "destination_name"


# ---------------------------------------------------------------------------
# Exporter failure cleanup
# ---------------------------------------------------------------------------


class _RaisingExporter(LitematicExporter):
    """An exporter that fails midway, after its destination directory exists."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error
        self.calls = 0

    def export(self, volume, destination, **options):  # type: ignore[no-untyped-def]
        self.calls += 1
        Path(destination).write_bytes(b"")
        raise self.error


class _MalformedExporter(LitematicExporter):
    """An exporter that reports success but writes a non-litematic file."""

    def export(self, volume, destination, **options):  # type: ignore[no-untyped-def]
        Path(destination).write_bytes(b"not a litematic at all")


class _EmptyExporter(LitematicExporter):
    def export(self, volume, destination, **options):  # type: ignore[no-untyped-def]
        Path(destination).write_bytes(b"")


def test_an_exporter_error_refuses_and_removes_its_temporary(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    exporter = _RaisingExporter(LitematicExportError("no palette"))
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description="d",
        exporter=exporter)
    _assert_refusal(result, "export_failed")
    assert exporter.calls == 1
    assert list(tmp_path.iterdir()) == []


def test_an_unexpected_exporter_error_is_a_typed_refusal(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    exporter = _RaisingExporter(RuntimeError("exploded"))
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description="d",
        exporter=exporter)
    _assert_refusal(result, "export_failed")
    assert result.detail["detail"] == "RuntimeError"
    assert list(tmp_path.iterdir()) == []


def test_a_malformed_written_artifact_is_refused_with_no_output(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description="d",
        exporter=_MalformedExporter())
    _assert_refusal(result, "verify_failed")
    assert list(tmp_path.iterdir()) == []


def test_an_empty_written_artifact_is_refused_with_no_output(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description="d",
        exporter=_EmptyExporter())
    _assert_refusal(result, "verify_failed")
    assert list(tmp_path.iterdir()) == []


def test_a_refusal_does_not_disturb_an_overwritten_destination(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    destination.write_bytes(b"original bytes that must survive")
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="n", description="d",
        exporter=_MalformedExporter(), overwrite=True)
    _assert_refusal(result, "verify_failed")
    assert destination.read_bytes() == b"original bytes that must survive"
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]


# ---------------------------------------------------------------------------
# Filesystem placement failure cleanup (mkdir / mkstemp / os.link / os.replace)
# ---------------------------------------------------------------------------


def test_a_parent_mkdir_failure_is_a_typed_refusal_with_no_temporary(
    happy_promotion: promotion.PromotionResult, tmp_path: Path, monkeypatch,
) -> None:
    """An ``OSError`` from creating the destination parent must not escape.

    The destination parent is deliberately deeper than ``tmp_path`` so its
    ``mkdir(parents=True)`` is the call that fails; a leaked temporary or an
    escaped ``PermissionError`` would both be defects here.
    """
    target_parent = tmp_path / "missing" / "nested"
    destination = target_parent / "build.litematic"
    real_mkdir = Path.mkdir

    def failing_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if self == target_parent:
            raise PermissionError("parent creation denied")
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(delivery.Path, "mkdir", failing_mkdir)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="n", description="d")

    _assert_refusal(result, "export_failed")
    assert result.detail["field"] == "destination_parent"
    assert result.detail["detail"] == "PermissionError"
    assert list(tmp_path.iterdir()) == []


def test_a_temporary_creation_failure_is_a_typed_refusal(
    happy_promotion: promotion.PromotionResult, tmp_path: Path, monkeypatch,
) -> None:
    real_mkstemp = delivery.tempfile.mkstemp

    def failing_mkstemp(*args: object, **kwargs: object):
        raise PermissionError("temporary creation denied")

    monkeypatch.setattr(delivery.tempfile, "mkstemp", failing_mkstemp)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description="d")

    _assert_refusal(result, "export_failed")
    assert result.detail["field"] == "temporary"
    assert result.detail["detail"] == "PermissionError"
    assert list(tmp_path.iterdir()) == []
    assert real_mkstemp is not failing_mkstemp


def test_a_temporary_close_failure_removes_the_created_temporary(
    happy_promotion: promotion.PromotionResult, tmp_path: Path, monkeypatch,
) -> None:
    """``mkstemp`` creates the file before ``os.close``; a close failure must clean it."""
    real_close = os.close
    state = {"raised": False}

    def failing_close(handle: int) -> None:
        real_close(handle)
        if not state["raised"]:
            state["raised"] = True
            raise OSError("descriptor close denied")

    monkeypatch.setattr(delivery.os, "close", failing_close)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, _destination(tmp_path), name="n", description="d")

    _assert_refusal(result, "export_failed")
    assert result.detail["field"] == "temporary"
    assert list(tmp_path.iterdir()) == []


def test_a_replace_failure_is_a_typed_refusal_that_removes_the_temporary(
    happy_promotion: promotion.PromotionResult, tmp_path: Path, monkeypatch,
) -> None:
    """An ``OSError`` from the explicit overwrite placement must be a typed refusal."""
    destination = _destination(tmp_path)

    def failing_replace(source: object, target: object) -> None:
        raise PermissionError("destination is locked")

    monkeypatch.setattr(delivery.os, "replace", failing_replace)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="n", description="d", overwrite=True)

    _assert_refusal(result, "export_failed")
    assert result.detail["field"] == "placement"
    assert result.detail["detail"] == "PermissionError"
    assert list(tmp_path.iterdir()) == []


def test_a_replace_failure_preserves_an_overwritten_destination(
    happy_promotion: promotion.PromotionResult, tmp_path: Path, monkeypatch,
) -> None:
    """A failed overwrite placement must leave the original bytes exactly intact."""
    destination = _destination(tmp_path)
    original = b"overwrite target that must survive a failed replace"
    destination.write_bytes(original)

    def failing_replace(source: object, target: object) -> None:
        raise PermissionError("destination is locked")

    monkeypatch.setattr(delivery.os, "replace", failing_replace)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="n", description="d", overwrite=True)

    _assert_refusal(result, "export_failed")
    assert result.detail["field"] == "placement"
    assert destination.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]


def test_a_no_clobber_link_failure_is_a_typed_refusal_that_removes_the_temporary(
    happy_promotion: promotion.PromotionResult, tmp_path: Path, monkeypatch,
) -> None:
    """An ``OSError`` from the no-clobber publication must become a typed refusal."""
    destination = _destination(tmp_path)

    def failing_link(source: object, target: object) -> None:
        raise PermissionError("hard links are unavailable here")

    monkeypatch.setattr(delivery.os, "link", failing_link)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="n", description="d")

    _assert_refusal(result, "export_failed")
    assert result.detail["field"] == "placement"
    assert result.detail["detail"] == "PermissionError"
    assert list(tmp_path.iterdir()) == []
    assert not destination.exists()


# ---------------------------------------------------------------------------
# Determinism: logical digest independent of timestamps and destinations
# ---------------------------------------------------------------------------


def test_logical_digest_is_stable_while_the_timestamped_bytes_change(
    happy_promotion: promotion.PromotionResult, tmp_path: Path, monkeypatch,
) -> None:
    first = _destination(tmp_path, "first.litematic")
    second = _destination(tmp_path, "second.litematic")

    monkeypatch.setattr(litematic_module.time, "time", lambda: 1_700_000_000.0)
    first_result = delivery.deliver_image_derived_litematic(
        happy_promotion, first, name="one", description="one")
    monkeypatch.setattr(litematic_module.time, "time", lambda: 1_900_000_000.0)
    second_result = delivery.deliver_image_derived_litematic(
        happy_promotion, second, name="two", description="two")

    assert isinstance(first_result, delivery.DeliveryResult), first_result
    assert isinstance(second_result, delivery.DeliveryResult), second_result
    assert first.read_bytes() != second.read_bytes(), "timestamps must change the bytes"
    assert first_result.artifact_sha256 == second_result.artifact_sha256
    assert first_result.promotion_sha256 == second_result.promotion_sha256
    first_record = first_result.to_record()
    second_record = second_result.to_record()
    # ``bytes_written`` is the only destination/timestamp-sensitive field: it is
    # the gzip container size, not part of the logical delivery identity.
    assert first_record["bytes_written"] == first.stat().st_size
    assert second_record["bytes_written"] == second.stat().st_size
    assert {key: value for key, value in first_record.items()
            if key != "bytes_written"} \
        == {key: value for key, value in second_record.items()
            if key != "bytes_written"}


def test_logical_digest_is_destination_independent(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    first = _destination(tmp_path, "here.litematic")
    second = _destination(nested, "elsewhere.litematic")

    first_result = delivery.deliver_image_derived_litematic(
        happy_promotion, first, name="here", description="here")
    second_result = delivery.deliver_image_derived_litematic(
        happy_promotion, second, name="elsewhere", description="elsewhere")

    assert isinstance(first_result, delivery.DeliveryResult), first_result
    assert isinstance(second_result, delivery.DeliveryResult), second_result
    assert first_result.artifact_sha256 == second_result.artifact_sha256
    assert second.exists() and second.parent.is_dir()


def test_verification_is_repeatable_and_matches_the_delivery(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    destination = _destination(tmp_path)
    result = delivery.deliver_image_derived_litematic(
        happy_promotion, destination, name="verify", description="verify")
    assert isinstance(result, delivery.DeliveryResult), result
    first = delivery.verify_image_derived_litematic(happy_promotion, destination)
    second = delivery.verify_image_derived_litematic(happy_promotion, str(destination))
    assert not isinstance(first, delivery.DeliveryRefusal)
    assert first == second == (result.artifact_sha256, result.block_count)


def test_verification_refuses_a_promotion_that_does_not_match(
    tmp_path: Path,
) -> None:
    promotion_result = _promote(((0, 0, 0), (1, 0, 0)))
    other = _promote(((0, 0, 0),), palette=(DIRT, STONE))
    destination = _destination(tmp_path)
    result = delivery.deliver_image_derived_litematic(
        promotion_result, destination, name="mismatch", description="mismatch")
    assert isinstance(result, delivery.DeliveryResult), result

    verification = delivery.verify_image_derived_litematic(other, destination)
    _assert_refusal(verification, "verify_failed")
    assert verification.detail["field"] in ("artifact_bounds", "artifact_block_count")


def test_verification_refuses_a_missing_artifact(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    result = delivery.verify_image_derived_litematic(
        happy_promotion, tmp_path / "absent.litematic")
    _assert_refusal(result, "verify_failed")
    assert result.detail["field"] == "artifact_missing"


def test_verification_refuses_a_non_path_artifact(
    happy_promotion: promotion.PromotionResult,
) -> None:
    result = delivery.verify_image_derived_litematic(happy_promotion, 7)  # type: ignore[arg-type]
    _assert_refusal(result, "verify_failed")


def test_verification_refuses_a_tampered_artifact_state(
    happy_promotion: promotion.PromotionResult, tmp_path: Path,
) -> None:
    """A hand-written artifact with the right shape but the wrong state."""
    import nbtlib
    from nbtlib import tag

    destination = _destination(tmp_path, "handwritten.litematic")
    root = {
        "Version": tag.Int(7),
        "SubVersion": tag.Int(1),
        "MinecraftDataVersion": tag.Int(4440),
        "Metadata": tag.Compound({
            "Name": tag.String("handwritten"), "Description": tag.String("handwritten"),
        }),
        "Regions": tag.Compound({
            "scan": tag.Compound({
                "Position": tag.Compound({"x": tag.Int(0), "y": tag.Int(0),
                                          "z": tag.Int(0)}),
                "Size": tag.Compound({"x": tag.Int(2), "y": tag.Int(1),
                                      "z": tag.Int(1)}),
                "BlockStatePalette": tag.List[tag.Compound]([
                    tag.Compound({"Name": tag.String("minecraft:air")}),
                    tag.Compound({"Name": tag.String("minecraft:diamond_block")}),
                ]),
                "BlockStates": tag.LongArray([1 | (1 << 2)]),
            })
        }),
    }
    nbtlib.File(tag.Compound(root), gzipped=True).save(destination)

    result = delivery.verify_image_derived_litematic(happy_promotion, destination)
    _assert_refusal(result, "verify_failed")
    assert result.detail["field"] == "artifact_state"


def _write_gap_artifact(path: Path, palette_zero: str) -> None:
    """Write a 3x1x1 artifact with an implicit air gap at local (1,0,0).

    ``palette_zero`` names palette index 0, the index the reader treats as the
    implicit air state.  With ``minecraft:air`` the artifact is the genuine
    delivery of ``_promote(((0,0,0), (2,0,0)))``; with a non-air name the same
    packed ``BlockStates`` hides a real block in the gap cell.
    """
    import nbtlib
    from nbtlib import tag

    root = {
        "Version": tag.Int(7),
        "SubVersion": tag.Int(1),
        "MinecraftDataVersion": tag.Int(4440),
        "Metadata": tag.Compound({
            "Name": tag.String("gap"), "Description": tag.String("gap")}),
        "Regions": tag.Compound({
            "scan": tag.Compound({
                "Position": tag.Compound({"x": tag.Int(0), "y": tag.Int(0),
                                          "z": tag.Int(0)}),
                "Size": tag.Compound({"x": tag.Int(3), "y": tag.Int(1),
                                      "z": tag.Int(1)}),
                "BlockStatePalette": tag.List[tag.Compound]([
                    tag.Compound({"Name": tag.String(palette_zero)}),
                    tag.Compound({"Name": tag.String("minecraft:stone")}),
                ]),
                # x=0 -> index 1 (stone), x=1 -> index 0 (implicit air), x=2 -> index 1
                "BlockStates": tag.LongArray([1 | (1 << 4)]),
            })
        }),
    }
    nbtlib.File(tag.Compound(root), gzipped=True).save(path)


def test_a_non_air_palette_index_zero_cannot_hide_a_gap_cell(
    tmp_path: Path,
) -> None:
    """Palette index 0 is the implicit air state and must be ``minecraft:air``.

    Both artifacts carry byte-identical packed ``BlockStates``; only the palette
    index-zero name differs.  The ``minecraft:air`` artifact is the genuine
    delivery and verifies, while a reader that skipped index zero would silently
    drop the hidden ``minecraft:diamond_block`` gap cell and accept the corrupted
    artifact too.  Index zero is therefore checked before any exact block
    comparison.
    """
    promotion_result = _promote(((0, 0, 0), (2, 0, 0)))
    genuine = _destination(tmp_path, "genuine.litematic")
    corrupted = _destination(tmp_path, "corrupted.litematic")
    _write_gap_artifact(genuine, "minecraft:air")
    _write_gap_artifact(corrupted, "minecraft:diamond_block")

    accepted = delivery.verify_image_derived_litematic(promotion_result, genuine)
    assert not isinstance(accepted, delivery.DeliveryRefusal), accepted
    assert accepted[1] == len(promotion_result.cells) == 2

    refused = delivery.verify_image_derived_litematic(promotion_result, corrupted)
    _assert_refusal(refused, "verify_failed")
    assert refused.detail["field"] == "artifact"
    with pytest.raises(ValueError):
        delivery._read_artifact(corrupted)


# ---------------------------------------------------------------------------
# Deterministic digest contract and record validation
# ---------------------------------------------------------------------------


def test_logical_digest_binds_only_the_promotion_and_content(
    happy_promotion: promotion.PromotionResult,
) -> None:
    blocks = tuple(
        (block.coordinate, (block.state_name, block.state_properties))
        for block in happy_promotion.cells)
    digest = delivery._logical_digest(happy_promotion.promotion_sha256, blocks)
    assert delivery._logical_digest(happy_promotion.promotion_sha256, blocks) == digest
    assert delivery._logical_digest(_h("other"), blocks) != digest
    assert delivery._logical_digest(happy_promotion.promotion_sha256, blocks[:1]) != digest
    payload = json.loads(delivery._canonical_json({
        "promotion_sha256": happy_promotion.promotion_sha256,
        "blocks": [
            {"coordinate": list(coordinate),
             "state": {"name": state[0],
                       "properties": [list(item) for item in state[1]]}}
            for coordinate, state in blocks],
        "block_count": len(blocks),
    }))
    assert set(payload) == {"promotion_sha256", "blocks", "block_count"}


def test_delivery_result_fields_are_validated() -> None:
    good = Path("scan.litematic")
    with pytest.raises(ValueError):
        delivery.DeliveryResult(promotion_sha256="not-a-digest",
                                artifact_sha256=_h("a"), block_count=1,
                                bytes_written=1, output_path=good)
    with pytest.raises(ValueError):
        delivery.DeliveryResult(promotion_sha256=_h("p"), artifact_sha256="short",
                                block_count=1, bytes_written=1, output_path=good)
    with pytest.raises(ValueError):
        delivery.DeliveryResult(promotion_sha256=_h("p"), artifact_sha256=_h("a"),
                                block_count=0, bytes_written=1, output_path=good)
    with pytest.raises(ValueError):
        delivery.DeliveryResult(promotion_sha256=_h("p"), artifact_sha256=_h("a"),
                                block_count=1, bytes_written=0, output_path=good)
    with pytest.raises(ValueError):
        delivery.DeliveryResult(promotion_sha256=_h("p"), artifact_sha256=_h("a"),
                                block_count=1, bytes_written=1,
                                output_path="scan.litematic")  # type: ignore[arg-type]


def test_delivery_refusal_record_is_frozen_and_rejects_partial_artifacts() -> None:
    refusal = delivery._refuse("export_failed", {"field": "export"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        refusal.reason = "invalid_input"  # type: ignore[misc]
    with pytest.raises(ValueError):
        delivery.DeliveryRefusal.build(
            reason="export_failed", detail={"blocks": []},
            detail_sha256=hashlib.sha256(b"{}").hexdigest())
    coerced = delivery._refuse("not_a_reason")
    assert coerced.reason == "invalid_input"
    assert coerced.refusal_sha256 == delivery._sha256(
        delivery._refusal_payload(coerced))


# ---------------------------------------------------------------------------
# Isolation boundary
# ---------------------------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


def test_module_imports_stay_inside_the_delivery_boundary() -> None:
    modules = _imported_modules(SOURCE)
    roots = {module.split(".")[0] for module in modules}
    assert roots <= {"__future__", "dataclasses", "hashlib", "json", "math", "os",
                     "re", "tempfile", "pathlib", "typing", "mc_photogrammetry",
                     "nbtlib"}
    allowed = {
        "mc_photogrammetry.export.litematic",
        "mc_photogrammetry.reconstruction",
        "mc_photogrammetry.reconstruction.state_promotion",
        "mc_photogrammetry.domain.models",
    }
    for module in modules:
        if module.startswith("mc_photogrammetry"):
            assert module in allowed, module
        for forbidden in ("cli", "ui", "store", "project", "camera", "manifest",
                          "f3", "anvil", "attribution"):
            assert forbidden not in module, (module, forbidden)


def test_module_exposes_no_cli_ui_or_project_store_symbols() -> None:
    assert hasattr(delivery, "deliver_image_derived_litematic")
    assert hasattr(delivery, "DeliveryResult")
    assert hasattr(delivery, "DeliveryRefusal")
    assert hasattr(delivery, "verify_image_derived_litematic")
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    for name in names:
        lowered = name.lower()
        for forbidden in ("argparse", "click", "console", "qtwidget", "pyside",
                          "projectstore", "scanproject", "camera_manifest",
                          "f3reference", "manual"):
            assert forbidden not in lowered, (name, forbidden)


def test_module_never_rewrites_the_destination_exporter_or_existing_suites() -> None:
    assert delivery.DELIVERY_EXPORT_BOUNDS is None
    assert delivery.DELIVERY_SCHEMA == "m5-image-derived-delivery-1"
    assert delivery.DELIVERY_PRODUCER_VERSION == "m5-image-derived-delivery-adapter-1"
    assert delivery.DELIVERY_MAX_DENSE_CELLS > 0
    assert "destination_exists" in delivery.DELIVERY_REFUSAL_REASONS
    assert "verify_failed" in delivery.DELIVERY_REFUSAL_REASONS
    text = SOURCE.read_text(encoding="utf-8")
    for forbidden in ("AGENT_COMMS", "AGENT_HANDOFF", "argparse", "sys.argv"):
        assert forbidden not in text
    assert "LitematicExporter" in text, "the seam must reuse the accepted exporter"


def test_entry_point_signature_is_the_declared_seam() -> None:
    signature = inspect.signature(delivery.deliver_image_derived_litematic)
    assert list(signature.parameters) == [
        "promotion_result", "destination", "name", "description", "overwrite",
        "exporter"]
    for keyword in ("name", "description", "overwrite", "exporter"):
        parameter = signature.parameters[keyword]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, keyword
    assert signature.parameters["overwrite"].default is False
    assert signature.parameters["exporter"].default is None
    verify = inspect.signature(delivery.verify_image_derived_litematic)
    assert list(verify.parameters) == ["promotion_result", "artifact"]
