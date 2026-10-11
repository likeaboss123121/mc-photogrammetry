"""Synthetic controls for the bounded M5 state-promotion adapter.

Every record here is built in memory.  The geometry volume is a self-consistent
synthetic ``PlaneVolumeResult`` and the proposal result is produced by the real
``propose_face_states`` producer over synthetic ``FaceObservation`` patches; no
image, manifest, camera, bound, mask fixture or F3 reference is read.  The
adapter's whole contract is that it consumes only an accepted
``PlaneVolumeResult``, an accepted ``FaceStateProposalResult`` and an explicit
promotion budget.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS
from mc_photogrammetry.domain.models import BlockState, BlockVolume, ObservationKind
from mc_photogrammetry.reconstruction import appearance_state_proposals as ap
from mc_photogrammetry.reconstruction import plane_volume as pv
from mc_photogrammetry.reconstruction import state_promotion as promotion
from mc_photogrammetry.reconstruction.face_observation import NORMAL_OF_SLOT, FaceObservation

SOURCE = Path(promotion.__file__)

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
TIE_A = PaletteColor(
    BlockState("minecraft:aaa"),
    (125, 125, 125),
    face_rgbs=((125, 125, 125),),
    family="cube",
    face_map=tuple((slot, (125, 125, 125)) for slot in FACE_SLOTS),
)
TIE_B = PaletteColor(
    BlockState("minecraft:bbb"),
    (125, 125, 125),
    face_rgbs=((125, 125, 125),),
    family="cube",
    face_map=tuple((slot, (125, 125, 125)) for slot in FACE_SLOTS),
)


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _observation(
    cell: tuple[int, int, int],
    *,
    frame_id: str = "f0",
    value: float = 125.0,
    provenance: tuple[str, ...] = ("cam/a",),
) -> FaceObservation:
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


def _claim(coordinate: tuple[int, int, int], index: int) -> pv.CellClaim:
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
    supported = 1
    unknown = omitted if host else 0
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
        "support_cells": supported + unknown,
        "supported_cells": supported,
        "unknown_cells": unknown,
    }


def _volume(
    coordinates: tuple[tuple[int, int, int], ...],
    *,
    omitted: int = 0,
    placement_sha256: str | None = None,
) -> pv.PlaneVolumeResult:
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
        placement_sha256=placement_sha256 or _h("placement"),
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


def _proposals(
    coordinates: tuple[tuple[int, int, int], ...],
    palette: tuple[PaletteColor, ...] = (STONE, DIRT),
    **overrides: object,
) -> ap.FaceStateProposalResult:
    options: dict[str, object] = {
        "max_cost": 1e9,
        "min_margin": 0.0,
        "max_observations": 1000,
        "max_candidate_evaluations": 100000,
        "max_payload_bytes": 10000000,
    }
    options.update(overrides)
    observations = tuple(_observation(cell) for cell in coordinates)
    return ap.propose_face_states(observations, palette, **options)  # type: ignore[arg-type]


def _budget(**overrides: int) -> promotion.PromotionBudget:
    caps = dict(max_cells=16, max_proposals=16, max_payload_bytes=1_000_000)
    caps.update(overrides)
    return promotion.make_promotion_budget(**caps)


def _clone(record: object, **changes: object) -> object:
    clone = object.__new__(type(record))
    for field in dataclasses.fields(record):  # type: ignore[arg-type]
        object.__setattr__(clone, field.name, getattr(record, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


def _reseal_volume(volume: pv.PlaneVolumeResult, **changes: object) -> pv.PlaneVolumeResult:
    clone = _clone(volume, **changes)
    object.__setattr__(clone, "volume_sha256", pv._sha256(pv._volume_payload(clone)))
    return clone  # type: ignore[return-value]


def _reseal_proposals(
    result: ap.FaceStateProposalResult,
    proposals: tuple[ap.FaceStateProposal, ...],
    *,
    status: str | None = None,
    reason: str | None = None,
) -> ap.FaceStateProposalResult:
    return ap._result(
        status=result.status if status is None else status,
        reason=result.reason if reason is None else reason,
        budget=None,
        proposals=tuple(proposals),
        observation_count=result.observation_count,
        palette_size=result.palette_size,
        candidate_evaluations=result.candidate_evaluations,
        max_cost=result.max_cost,
        min_margin=result.min_margin,
        max_observations=result.max_observations,
        max_candidate_evaluations=result.max_candidate_evaluations,
        max_payload_bytes=result.max_payload_bytes,
    )


def _reforge_proposal(
    proposal: ap.FaceStateProposal, **changes: object
) -> ap.FaceStateProposal:
    fields = {field.name: getattr(proposal, field.name)
              for field in dataclasses.fields(proposal)
              if field.name != "proposal_digest"}
    fields.update(changes)
    return ap._make_proposal(**fields)


def _reforge_claim(claim: pv.CellClaim, **changes: object) -> pv.CellClaim:
    """Clone a claim past its constructor and reseal its stored digest.

    This models a hostile record that is digest-self-consistent but carries a
    field the accepted constructor rejects.
    """
    clone = object.__new__(pv.CellClaim)
    for field in dataclasses.fields(claim):
        object.__setattr__(clone, field.name, getattr(claim, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    object.__setattr__(clone, "claim_sha256",
                       pv._sha256(pv._claim_payload(clone)))
    return clone  # type: ignore[return-value]


def _reseal_cell(
    cell: pv.SupportedCellObservation, **changes: object
) -> pv.SupportedCellObservation:
    clone = _clone(cell, **changes)
    object.__setattr__(clone, "observation_sha256",
                       pv._sha256(pv._observation_payload(clone)))
    return clone  # type: ignore[return-value]


def _forged_budget(**changes: object) -> promotion.PromotionBudget:
    """Clone a valid budget past ``__post_init__`` with the given changes."""
    budget = _budget()
    clone = object.__new__(promotion.PromotionBudget)
    for field in dataclasses.fields(budget):
        object.__setattr__(clone, field.name, getattr(budget, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone  # type: ignore[return-value]


def _assert_refusal_sha_replays(refusal: promotion.PromotionRefusal) -> None:
    assert refusal.refusal_sha256 == promotion._sha256(
        promotion._refusal_payload(refusal))
    assert refusal.detail_sha256 == hashlib.sha256(
        promotion._canonical_json(refusal.detail).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def happy_volume() -> pv.PlaneVolumeResult:
    return _volume(((0, 0, 0), (1, 0, 0)))


@pytest.fixture(scope="module")
def happy_proposals() -> ap.FaceStateProposalResult:
    return _proposals(((0, 0, 0), (1, 0, 0)))


@pytest.fixture(scope="module")
def happy_result(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> promotion.PromotionResult:
    result = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=_budget())
    assert isinstance(result, promotion.PromotionResult), result
    return result


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_two_cells_promote_exact_states(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
    happy_result: promotion.PromotionResult,
) -> None:
    assert happy_proposals.status == ap.PROPOSED
    assert [proposal.status for proposal in happy_proposals.proposals] == [
        ap.PROPOSED, ap.PROPOSED]
    assert [block.coordinate for block in happy_result.cells] == [
        (0, 0, 0), (1, 0, 0)]
    assert len(happy_result.cells) == 2
    for block, proposal in zip(happy_result.cells, happy_proposals.proposals):
        assert block.coordinate == proposal.cell
        assert block.state_name == "minecraft:stone"
        assert block.state_properties == ()
        assert block.frame_ids == proposal.support_frame_ids == ("f0",)
        assert block.provenance == proposal.provenance == ("cam/a",)
        assert block.observation_digests == proposal.observation_digests
        assert block.proposal_digest == proposal.proposal_digest
        assert block.evidence_digest == proposal.evidence_digest
        assert 0.0 <= block.confidence <= 1.0
        assert math.isfinite(block.confidence)
        assert block.confidence == promotion.promotion_confidence(
            proposal.best_cost, proposal.margin)
    assert happy_result.volume_sha256 == happy_volume.volume_sha256
    assert happy_result.proposal_result_digest == happy_proposals.result_digest
    assert happy_result.omitted_unknown == 0
    assert happy_result.export_ready is True
    assert happy_result.exact_state_eligible is True
    assert happy_result.export_bounds is None
    assert happy_result.normalization_offset == (0, 0, 0)
    assert happy_result.scale_state == "unresolved"
    assert happy_result.normalization_state == "unresolved"
    assert happy_result.gauge_state == pv.V3_VOLUME_GAUGE_STATE
    assert happy_result.gauge_axes == (0, 1, 2)


def test_result_record_is_immutable_and_digest_bound(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
    happy_result: promotion.PromotionResult,
) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        happy_result.export_ready = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        happy_result.cells[0].confidence = 0.0  # type: ignore[misc]
    assert "volume" not in {field.name for field in dataclasses.fields(happy_result)}
    assert not hasattr(happy_result, "volume")
    for block in happy_result.cells:
        assert block.block_sha256 == promotion._sha256(
            promotion._block_payload(block))
    payload = promotion._result_payload(happy_result)
    assert happy_result.payload_bytes == len(
        promotion._canonical_json(payload).encode("utf-8"))
    assert happy_result.payload_sha256 == promotion._sha256(payload)
    assert happy_result.promotion_sha256 == promotion._sha256({
        **payload,
        "payload_sha256": happy_result.payload_sha256,
        "max_payload_bytes": happy_result.max_payload_bytes,
    })
    record = happy_result.to_record()
    assert json.loads(json.dumps(record, sort_keys=True))["promotion_sha256"] \
        == happy_result.promotion_sha256
    assert record["export_bounds"] is None


def test_promotion_record_rebuilds_and_replays_exactly(happy_result) -> None:
    record = happy_result.to_record()

    replayed = promotion.promotion_result_from_record(record)

    assert replayed.to_record() == record
    assert replayed.promotion_sha256 == happy_result.promotion_sha256
    assert replayed.to_block_volume().to_dict() \
        == happy_result.to_block_volume().to_dict()


def test_promotion_record_rejects_a_stale_nested_digest(happy_result) -> None:
    record = json.loads(json.dumps(happy_result.to_record()))
    record["cells"][0]["block_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        promotion.promotion_result_from_record(record)


def test_promotion_record_rejects_unknown_or_missing_keys(happy_result) -> None:
    record = json.loads(json.dumps(happy_result.to_record()))

    with pytest.raises(ValueError):
        promotion.promotion_result_from_record({**record, "mystery": 1})

    missing = dict(record)
    missing.pop("promotion_sha256")
    with pytest.raises(ValueError):
        promotion.promotion_result_from_record(missing)


def test_promotion_record_rejects_a_forged_field_the_record_digest_cannot_see(
    happy_result,
) -> None:
    record = json.loads(json.dumps(happy_result.to_record()))
    record["cells"][0]["state"]["name"] = "minecraft:air"

    with pytest.raises(ValueError):
        promotion.promotion_result_from_record(record)


def test_promotion_does_not_mutate_its_inputs(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    before_volume = happy_volume.to_record()
    before_proposals = ap._result_payload(happy_proposals)
    outcome = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionResult)
    assert happy_volume.to_record() == before_volume
    assert ap._result_payload(happy_proposals) == before_proposals


# ---------------------------------------------------------------------------
# Determinism and digest binding
# ---------------------------------------------------------------------------


def test_deterministic_order_and_digests(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
    happy_result: promotion.PromotionResult,
) -> None:
    again = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=_budget())
    assert isinstance(again, promotion.PromotionResult)
    assert again.to_record() == happy_result.to_record()
    assert again.promotion_sha256 == happy_result.promotion_sha256
    coordinates = [block.coordinate for block in happy_result.cells]
    assert coordinates == sorted(coordinates)


def test_promotion_digest_binds_the_volume_source(
    happy_proposals: ap.FaceStateProposalResult,
    happy_result: promotion.PromotionResult,
) -> None:
    other = _volume(((0, 0, 0), (1, 0, 0)), placement_sha256=_h("other-placement"))
    assert other.volume_sha256 != happy_result.volume_sha256
    outcome = promotion.promote_plane_states(
        other, happy_proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionResult), outcome
    assert outcome.volume_sha256 == other.volume_sha256
    assert outcome.promotion_sha256 != happy_result.promotion_sha256
    assert [block.coordinate for block in outcome.cells] == [
        block.coordinate for block in happy_result.cells]


def test_proposal_tuple_order_is_canonicalized_but_still_bound(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
    happy_result: promotion.PromotionResult,
) -> None:
    reversed_result = _reseal_proposals(
        happy_proposals, tuple(reversed(happy_proposals.proposals)))
    assert reversed_result.result_digest != happy_proposals.result_digest
    outcome = promotion.promote_plane_states(
        happy_volume, reversed_result, budget=_budget())
    assert isinstance(outcome, promotion.PromotionResult), outcome
    assert [block.coordinate for block in outcome.cells] == [
        (0, 0, 0), (1, 0, 0)]
    assert outcome.proposal_result_digest == reversed_result.result_digest
    assert outcome.promotion_sha256 != happy_result.promotion_sha256


# ---------------------------------------------------------------------------
# Missing / extraneous proposals
# ---------------------------------------------------------------------------


def test_missing_proposal_is_a_typed_refusal(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    partial = _reseal_proposals(happy_proposals, (happy_proposals.proposals[0],))
    outcome = promotion.promote_plane_states(
        happy_volume, partial, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "refusal"
    assert outcome.reason == "missing_proposal"
    assert outcome.detail == {"missing": 1}
    _assert_refusal_sha_replays(outcome)


def test_extraneous_proposal_is_a_typed_refusal(
    happy_volume: pv.PlaneVolumeResult,
) -> None:
    wide = _proposals(((0, 0, 0), (1, 0, 0), (2, 0, 0)))
    outcome = promotion.promote_plane_states(happy_volume, wide, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "refusal"
    assert outcome.reason == "extraneous_proposal"
    assert outcome.detail == {"extraneous": 1}


def test_duplicate_proposal_cell_is_a_typed_refusal(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    duplicated = _reseal_proposals(
        happy_proposals, (happy_proposals.proposals[0], happy_proposals.proposals[0]))
    outcome = promotion.promote_plane_states(
        happy_volume, duplicated, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "extraneous_proposal"
    assert outcome.detail == {"field": "duplicate_cell"}


def test_non_canonical_geometry_cells_are_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    reversed_volume = _reseal_volume(
        happy_volume, cells=tuple(reversed(happy_volume.cells)))
    outcome = promotion.promote_plane_states(
        reversed_volume, happy_proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "digest_mismatch"


# ---------------------------------------------------------------------------
# Review / abstention and invalid state
# ---------------------------------------------------------------------------


def test_review_abstention_is_a_typed_review_refusal(
    happy_volume: pv.PlaneVolumeResult,
) -> None:
    review = _proposals(((0, 0, 0),), (TIE_A, TIE_B), min_margin=1e-6)
    assert review.status == ap.REVIEW
    assert review.proposals[0].reason == ap.WEAK_MARGIN
    outcome = promotion.promote_plane_states(
        _volume(((0, 0, 0),)), review, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "review"
    assert outcome.reason == "review_abstention"
    assert outcome.detail == {"cell": [0, 0, 0], "reason": "weak_margin"}
    _assert_refusal_sha_replays(outcome)


def test_result_level_review_is_a_typed_review_refusal(
    happy_volume: pv.PlaneVolumeResult,
) -> None:
    empty = ap.propose_face_states(
        (), (STONE,), max_cost=1e9, min_margin=0.0, max_observations=10,
        max_candidate_evaluations=10, max_payload_bytes=1000)
    assert empty.status == ap.REVIEW and empty.reason == ap.NO_OBSERVATION
    outcome = promotion.promote_plane_states(
        happy_volume, empty, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "review"
    assert outcome.reason == "review_abstention"
    assert outcome.detail == {"field": "result", "reason": "no_observation"}


def test_result_level_review_with_proposed_children_never_promotes(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    assert happy_proposals.status == ap.PROPOSED
    assert all(proposal.status == ap.PROPOSED
               for proposal in happy_proposals.proposals)
    # A self-consistent result-level review whose children are all individually
    # proposed must not ride in on those children.
    reviewed = _reseal_proposals(
        happy_proposals, happy_proposals.proposals,
        status=ap.REVIEW, reason=ap.WEAK_MARGIN)
    assert reviewed.status == ap.REVIEW
    assert len(reviewed.proposals) == 2
    assert all(proposal.status == ap.PROPOSED for proposal in reviewed.proposals)

    outcome = promotion.promote_plane_states(
        happy_volume, reviewed, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "review"
    assert outcome.reason == "review_abstention"
    assert outcome.detail == {
        "field": "result_status", "status": "review", "reason": "weak_margin"}
    assert "cells" not in outcome.to_record()
    _assert_refusal_sha_replays(outcome)

    # The same hostile result is rejected before an all-zero budget can stop it.
    zero = _budget(max_cells=0, max_proposals=0, max_payload_bytes=0)
    assert promotion.promote_plane_states(
        happy_volume, reviewed, budget=zero).reason == "review_abstention"

    # A result-level review with no reason at all is the same typed refusal.
    unreasoned = _reseal_proposals(
        happy_proposals, happy_proposals.proposals,
        status=ap.REVIEW, reason=None)
    assert unreasoned.status == ap.REVIEW and unreasoned.reason is None
    assert promotion.promote_plane_states(
        happy_volume, unreasoned, budget=_budget()).detail == {
        "field": "result_status", "status": "review", "reason": None}


def test_proposal_source_budget_refusal_is_a_source_mismatch(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    refused = ap.propose_face_states(
        (_observation((0, 0, 0)),), (STONE, DIRT), max_cost=1e9, min_margin=0.0,
        max_observations=0, max_candidate_evaluations=0, max_payload_bytes=0)
    assert refused.status == ap.REFUSED
    outcome = promotion.promote_plane_states(
        happy_volume, refused, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "refusal"
    assert outcome.reason == "source_mismatch"
    assert outcome.detail == {"field": "proposal_status", "budget": "max_observations"}


def test_air_state_is_a_typed_invalid_state_refusal(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    air = _reforge_proposal(
        happy_proposals.proposals[0], state=BlockState("minecraft:air"))
    reforged = _reseal_proposals(
        happy_proposals, (air, happy_proposals.proposals[1]))
    outcome = promotion.promote_plane_states(
        happy_volume, reforged, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "refusal"
    assert outcome.reason == "invalid_state"
    assert outcome.detail == {"field": "state_air"}


def test_malformed_proposed_state_is_a_typed_invalid_state_refusal(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    broken = _clone(happy_proposals.proposals[0], state=None)
    forged = _reseal_proposals(
        happy_proposals, (broken, happy_proposals.proposals[1]))
    outcome = promotion.promote_plane_states(
        happy_volume, forged, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "invalid_state"
    assert outcome.detail == {"field": "state"}


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


def test_geometry_conflicts_are_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    forged = _clone(happy_volume, conflicts=(object(),))
    outcome = promotion.promote_plane_states(
        forged, happy_proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "refusal"
    assert outcome.reason == "conflict"
    assert outcome.detail == {"geometry_conflicts": 1}
    _assert_refusal_sha_replays(outcome)


def test_flagged_conflict_cells_are_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    cells = list(happy_volume.cells)
    cells[0] = _clone(cells[0], conflict_kind="opposed_sides")
    forged = _clone(happy_volume, cells=tuple(cells))
    outcome = promotion.promote_plane_states(
        forged, happy_proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "conflict"
    assert outcome.detail == {"flagged_cells": 1}


# ---------------------------------------------------------------------------
# Omitted unknown gate
# ---------------------------------------------------------------------------


def test_omitted_unknown_is_refused_by_default() -> None:
    volume = _volume(((0, 0, 0),), omitted=1)
    proposals = _proposals(((0, 0, 0),))
    outcome = promotion.promote_plane_states(volume, proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "review"
    assert outcome.reason == "omitted_unknown"
    assert outcome.detail == {"omitted_unknown": 1}
    assert "cells" not in outcome.to_record()
    assert "blocks" not in outcome.detail
    _assert_refusal_sha_replays(outcome)


def test_omitted_unknown_opt_in_gates_exact_state_but_not_export() -> None:
    volume = _volume(((0, 0, 0),), omitted=1)
    proposals = _proposals(((0, 0, 0),))
    outcome = promotion.promote_plane_states(
        volume, proposals, budget=_budget(), allow_omitted_unknown=True)
    assert isinstance(outcome, promotion.PromotionResult), outcome
    assert outcome.omitted_unknown == 1
    assert outcome.export_ready is True
    assert outcome.exact_state_eligible is False
    assert [block.coordinate for block in outcome.cells] == [(0, 0, 0)]
    exported = outcome.to_block_volume()
    assert set(exported.blocks) == {(0, 0, 0)}
    assert (0, 0, 100) not in exported.blocks  # the omitted cell is never invented


def test_zero_omissions_keep_exact_state_eligible(
    happy_result: promotion.PromotionResult,
) -> None:
    assert happy_result.omitted_unknown == 0
    assert happy_result.exact_state_eligible is True


def test_forged_eligibility_flags_are_rejected(
    happy_result: promotion.PromotionResult,
) -> None:
    fields = {field.name: getattr(happy_result, field.name)
              for field in dataclasses.fields(happy_result)}
    for name in ("payload_bytes", "payload_sha256", "promotion_sha256"):
        fields.pop(name)
    with pytest.raises(ValueError):
        promotion.PromotionResult.build(**{**fields, "export_ready": False})
    with pytest.raises(ValueError):
        promotion.PromotionResult.build(**{
            **fields, "exact_state_eligible": True, "omitted_unknown": 1})
    with pytest.raises(ValueError):
        promotion.PromotionResult.build(**{**fields, "export_bounds": (0, 0, 0)})


# ---------------------------------------------------------------------------
# Tampered nested digests
# ---------------------------------------------------------------------------


def test_tampered_geometry_cell_digest_is_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    cells = list(happy_volume.cells)
    cells[0] = _clone(cells[0], observation_sha256="0" * 64)
    forged = _reseal_volume(happy_volume, cells=tuple(cells))
    assert forged.volume_sha256 != happy_volume.volume_sha256
    outcome = promotion.promote_plane_states(
        forged, happy_proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "refusal"
    assert outcome.reason == "digest_mismatch"
    assert outcome.detail == {"field": "volume"}


def test_resealed_forged_claim_field_is_refused_before_budget(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    original = happy_volume.cells[0]
    forged_claim = _reforge_claim(original.claims[0], frame="")
    forged_cell = _reseal_cell(
        original, claims=(forged_claim,), frame_ids=("",))
    forged_volume = _reseal_volume(
        happy_volume, cells=(forged_cell, *happy_volume.cells[1:]))

    # The forged cell and volume are digest-self-consistent: a record-only
    # rebuild of the enclosing cell accepts the forged claim ...
    record_only = pv.SupportedCellObservation.build(**{
        field.name: getattr(forged_cell, field.name)
        for field in dataclasses.fields(forged_cell)})
    assert record_only == forged_cell
    # ... while the accepted claim constructor rejects the forged field.
    with pytest.raises(ValueError):
        pv.CellClaim.build(**{
            field.name: getattr(forged_claim, field.name)
            for field in dataclasses.fields(forged_claim)})

    zero = _budget(max_cells=0, max_proposals=0, max_payload_bytes=0)
    outcome = promotion.promote_plane_states(
        forged_volume, happy_proposals, budget=zero)
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "refusal"
    assert outcome.reason == "digest_mismatch"
    assert outcome.detail["field"] == "volume"
    _assert_refusal_sha_replays(outcome)


def test_resealed_forged_claim_digest_is_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    original = happy_volume.cells[0]
    stale_claim = _clone(original.claims[0], claim_sha256="0" * 64)
    forged_cell = _reseal_cell(
        original, claims=(stale_claim,), frame_ids=("f0",))
    forged_volume = _reseal_volume(
        happy_volume, cells=(forged_cell, *happy_volume.cells[1:]))
    outcome = promotion.promote_plane_states(
        forged_volume, happy_proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "digest_mismatch"
    assert outcome.detail == {"field": "volume"}


def test_tampered_work_digest_is_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    work = _clone(happy_volume.work, work_sha256="0" * 64)
    forged = _reseal_volume(happy_volume, work=work)
    outcome = promotion.promote_plane_states(
        forged, happy_proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "digest_mismatch"


def test_tampered_proposal_digest_is_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    stale = _clone(happy_proposals.proposals[0], proposal_digest="0" * 64)
    forged = _reseal_proposals(
        happy_proposals, (stale, happy_proposals.proposals[1]))
    # The result-level digest deliberately excludes the nested proposal digest,
    # so this record is self-consistent at the result level and only the nested
    # rebuild can reject it.
    assert isinstance(forged, ap.FaceStateProposalResult)
    outcome = promotion.promote_plane_states(
        happy_volume, forged, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "digest_mismatch"
    assert outcome.detail["field"] == "proposals"


def test_tampered_evidence_digest_is_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    stale = _clone(happy_proposals.proposals[0], evidence_digest="0" * 64)
    forged = _reseal_proposals(
        happy_proposals, (stale, happy_proposals.proposals[1]))
    outcome = promotion.promote_plane_states(
        happy_volume, forged, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "digest_mismatch"


def test_tampered_result_payload_digest_is_refused(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    forged = _clone(happy_proposals, payload_sha256="0" * 64)
    outcome = promotion.promote_plane_states(
        happy_volume, forged, budget=_budget())
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.reason == "digest_mismatch"
    assert outcome.detail["field"] == "proposals"


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_budgets_are_atomic_and_exact(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
    happy_result: promotion.PromotionResult,
) -> None:
    exact_cells = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=_budget(max_cells=2))
    assert isinstance(exact_cells, promotion.PromotionResult)

    too_few_cells = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=_budget(max_cells=1))
    assert isinstance(too_few_cells, promotion.PromotionRefusal), too_few_cells
    assert too_few_cells.kind == "refusal"
    assert too_few_cells.reason == "budget_exhausted"
    assert too_few_cells.detail == {"cap": "max_cells", "limit": 1, "observed": 2}
    assert "cells" not in too_few_cells.to_record()
    assert "blocks" not in too_few_cells.detail
    _assert_refusal_sha_replays(too_few_cells)

    exact_proposals = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=_budget(max_proposals=2))
    assert isinstance(exact_proposals, promotion.PromotionResult)

    too_few_proposals = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=_budget(max_proposals=1))
    assert isinstance(too_few_proposals, promotion.PromotionRefusal)
    assert too_few_proposals.reason == "budget_exhausted"
    assert too_few_proposals.detail == {
        "cap": "max_proposals", "limit": 1, "observed": 2}

    exact_bytes = promotion.promote_plane_states(
        happy_volume, happy_proposals,
        budget=_budget(max_payload_bytes=happy_result.payload_bytes))
    assert isinstance(exact_bytes, promotion.PromotionResult)
    assert exact_bytes.payload_bytes == happy_result.payload_bytes

    too_few_bytes = promotion.promote_plane_states(
        happy_volume, happy_proposals,
        budget=_budget(max_payload_bytes=happy_result.payload_bytes - 1))
    assert isinstance(too_few_bytes, promotion.PromotionRefusal), too_few_bytes
    assert too_few_bytes.reason == "budget_exhausted"
    assert too_few_bytes.detail == {
        "cap": "max_payload_bytes",
        "limit": happy_result.payload_bytes - 1,
        "observed": happy_result.payload_bytes,
    }
    assert all(block.block_sha256 != too_few_bytes.refusal_sha256
               for block in happy_result.cells)


def test_validation_precedes_zero_budgets(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    zero = _budget(max_cells=0, max_proposals=0, max_payload_bytes=0)
    valid = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=zero)
    assert isinstance(valid, promotion.PromotionRefusal)
    assert valid.reason == "budget_exhausted"

    assert promotion.promote_plane_states(
        None, happy_proposals, budget=zero).detail == {"field": "volume"}
    assert promotion.promote_plane_states(
        happy_volume, None, budget=zero).detail == {"field": "proposals"}
    assert promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=None).detail == {"field": "budget"}
    assert promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=zero,
        allow_omitted_unknown=1).detail == {"field": "allow_omitted_unknown"}

    tampered_cells = list(happy_volume.cells)
    tampered_cells[0] = _clone(tampered_cells[0], observation_sha256="0" * 64)
    tampered = _reseal_volume(happy_volume, cells=tuple(tampered_cells))
    assert promotion.promote_plane_states(
        tampered, happy_proposals, budget=zero).reason == "digest_mismatch"

    conflicted = _clone(happy_volume, conflicts=(object(),))
    assert promotion.promote_plane_states(
        conflicted, happy_proposals, budget=zero).reason == "conflict"

    review = _proposals(((0, 0, 0),), (TIE_A, TIE_B), min_margin=1e-6)
    assert promotion.promote_plane_states(
        _volume(((0, 0, 0),)), review, budget=zero).reason == "review_abstention"

    partial = _reseal_proposals(happy_proposals, (happy_proposals.proposals[0],))
    assert promotion.promote_plane_states(
        happy_volume, partial, budget=zero).reason == "missing_proposal"

    air = _reforge_proposal(
        happy_proposals.proposals[0], state=BlockState("minecraft:air"))
    reforged = _reseal_proposals(
        happy_proposals, (air, happy_proposals.proposals[1]))
    assert promotion.promote_plane_states(
        happy_volume, reforged, budget=zero).reason == "invalid_state"


def test_budget_record_is_self_validating() -> None:
    with pytest.raises(ValueError):
        promotion.make_promotion_budget(max_cells=1)
    with pytest.raises(ValueError):
        promotion.make_promotion_budget(
            max_cells=-1, max_proposals=1, max_payload_bytes=1)
    with pytest.raises(ValueError):
        promotion.make_promotion_budget(
            max_cells=1.0, max_proposals=1, max_payload_bytes=1)
    with pytest.raises(ValueError):
        promotion.PromotionBudget(
            max_cells=1, max_proposals=1, max_payload_bytes=1,
            budget_sha256="0" * 64)
    budget = promotion.make_promotion_budget(
        max_cells=0, max_proposals=0, max_payload_bytes=0)
    assert budget.budget_sha256 == promotion._sha256({
        "max_cells": 0, "max_proposals": 0, "max_payload_bytes": 0})
    assert budget.to_record()["max_cells"] == 0


def test_forged_budget_digest_is_refused_before_input_or_budget_work(
    happy_volume: pv.PlaneVolumeResult,
    happy_proposals: ap.FaceStateProposalResult,
) -> None:
    forged = _forged_budget(budget_sha256="0" * 64)
    outcome = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=forged)
    assert isinstance(outcome, promotion.PromotionRefusal), outcome
    assert outcome.kind == "refusal"
    assert outcome.reason == "digest_mismatch"
    assert outcome.detail == {"field": "budget"}
    _assert_refusal_sha_replays(outcome)

    # The budget replay precedes input inspection: a hostile budget is named
    # even when the volume itself is unusable.
    assert promotion.promote_plane_states(
        None, happy_proposals, budget=forged).detail == {"field": "budget"}

    # All-zero caps do not make a forged digest a budget stop.
    forged_zero = _forged_budget(
        max_cells=0, max_proposals=0, max_payload_bytes=0,
        budget_sha256="0" * 64)
    zero_outcome = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=forged_zero)
    assert zero_outcome.reason == "digest_mismatch"
    assert zero_outcome.detail == {"field": "budget"}

    # A constructor-bypassed malformed cap is a typed invalid_input even when
    # the stored digest is recomputed to match the forged caps.
    malformed = _forged_budget(max_cells=-1)
    object.__setattr__(malformed, "budget_sha256",
                       promotion._sha256(malformed._payload()))
    malformed_outcome = promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=malformed)
    assert malformed_outcome.reason == "invalid_input"
    assert malformed_outcome.detail == {"field": "budget"}

    noninteger = _forged_budget(max_payload_bytes=1.0)
    object.__setattr__(noninteger, "budget_sha256",
                       promotion._sha256(noninteger._payload()))
    assert promotion.promote_plane_states(
        happy_volume, happy_proposals, budget=noninteger).reason == "invalid_input"


# ---------------------------------------------------------------------------
# Fresh BlockVolume and state-snapshot isolation
# ---------------------------------------------------------------------------


def test_to_block_volume_returns_fresh_observed_observations(
    happy_result: promotion.PromotionResult,
) -> None:
    first = happy_result.to_block_volume()
    assert isinstance(first, BlockVolume)
    assert set(first.blocks) == {block.coordinate for block in happy_result.cells}
    for block in happy_result.cells:
        observation = first.blocks[block.coordinate]
        assert observation.kind is ObservationKind.OBSERVED
        assert observation.state == BlockState(block.state_name, block.state_properties)
        assert observation.confidence == block.confidence
        assert 0.0 <= observation.confidence <= 1.0
        assert tuple(observation.frame_ids) == block.frame_ids


def test_to_block_volume_is_mutation_isolated(
    happy_result: promotion.PromotionResult,
) -> None:
    before = happy_result.to_record()
    first = happy_result.to_block_volume()
    first.set((99, 99, 99), next(iter(first.blocks.values())))
    observation = first.blocks[(0, 0, 0)]
    observation.frame_ids.append("foreign")
    observation.confidence = 0.0
    second = happy_result.to_block_volume()
    assert (99, 99, 99) not in second.blocks
    assert "foreign" not in second.blocks[(0, 0, 0)].frame_ids
    assert second.blocks[(0, 0, 0)].confidence == happy_result.cells[0].confidence
    assert happy_result.to_record() == before


def test_list_backed_proposal_state_is_snapshotted_immutably(
    happy_volume: pv.PlaneVolumeResult,
) -> None:
    state = BlockState("minecraft:stone", [["facing", "north"]])
    assert type(state.properties) is list
    palette = PaletteColor(
        state,
        (125, 125, 125),
        face_rgbs=((125, 125, 125),),
        family="cube",
        face_map=tuple((slot, (125, 125, 125)) for slot in FACE_SLOTS),
        full_cube=True,
    )
    proposals = _proposals(((0, 0, 0),), (palette,))
    outcome = promotion.promote_plane_states(
        _volume(((0, 0, 0),)), proposals, budget=_budget())
    assert isinstance(outcome, promotion.PromotionResult), outcome
    block = outcome.cells[0]
    assert type(block.state_properties) is tuple
    assert block.state_properties == (("facing", "north"),)
    before = outcome.to_record()
    state.properties[0][1] = "south"
    state.properties.append(["waterlogged", "true"])
    assert outcome.to_record() == before
    exported = outcome.to_block_volume()
    assert exported.blocks[(0, 0, 0)].state.properties == (("facing", "north"),)


# ---------------------------------------------------------------------------
# Refusal record contract
# ---------------------------------------------------------------------------


def test_refusal_record_is_frozen_and_rejects_partial_blocks() -> None:
    refusal = promotion._refusal("conflict", {"geometry_conflicts": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        refusal.reason = "invalid_input"  # type: ignore[misc]
    assert "cells" not in refusal.to_record()
    assert "blocks" not in refusal.to_record()
    with pytest.raises(ValueError):
        promotion.PromotionRefusal.build(
            kind="refusal", reason="conflict", detail={"blocks": []},
            detail_sha256=hashlib.sha256(b"{}").hexdigest())
    with pytest.raises(ValueError):
        promotion.PromotionRefusal.build(
            kind="review", reason="conflict",
            detail={}, detail_sha256=hashlib.sha256(b"{}").hexdigest())


def test_budget_refusal_detail_shape_is_enforced() -> None:
    with pytest.raises(ValueError):
        promotion.PromotionRefusal.build(
            kind="refusal", reason="budget_exhausted", detail={"cap": "max_cells"},
            detail_sha256=hashlib.sha256(b"{}").hexdigest())


# ---------------------------------------------------------------------------
# Confidence derivation
# ---------------------------------------------------------------------------


def test_confidence_is_bounded_finite_and_deterministic() -> None:
    assert promotion.promotion_confidence(0.0, math.inf) == 1.0
    assert promotion.promotion_confidence(3.0, math.inf) == pytest.approx(0.25)
    assert promotion.promotion_confidence(0.0, 0.0) == 0.0
    assert promotion.promotion_confidence(1.0, 1.0) == pytest.approx(0.25)
    for best, margin in ((0.0, 0.0), (0.5, 2.0), (10.0, 1.0), (0.0, math.inf)):
        first = promotion.promotion_confidence(best, margin)
        second = promotion.promotion_confidence(best, margin)
        assert first == second
        assert 0.0 <= first <= 1.0 and math.isfinite(first)
    with pytest.raises(promotion._Stop):
        promotion.promotion_confidence(-1.0, 1.0)
    with pytest.raises(promotion._Stop):
        promotion.promotion_confidence(float("nan"), 1.0)
    with pytest.raises(promotion._Stop):
        promotion.promotion_confidence(1.0, float("nan"))
    assert promotion.PROMOTION_CONFIDENCE_POLICY == "margin-over-cost-separation-v1"


def test_promoted_block_rejects_air_state_and_malformed_properties() -> None:
    with pytest.raises(ValueError):
        promotion.PromotedBlock.build(
            coordinate=(0, 0, 0), state_name="minecraft:air",
            state_properties=(), confidence=1.0, frame_ids=("f0",),
            provenance=(), observation_digests=(_h("obs"),),
            proposal_digest=_h("proposal"), evidence_digest=_h("evidence"))
    with pytest.raises(ValueError):
        promotion.PromotedBlock.build(
            coordinate=(0, 0, 0), state_name="minecraft:stone",
            state_properties=(("b", "1"), ("a", "2")), confidence=1.0,
            frame_ids=("f0",), provenance=(), observation_digests=(_h("obs"),),
            proposal_digest=_h("proposal"), evidence_digest=_h("evidence"))


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


def test_module_imports_stay_inside_the_boundary() -> None:
    modules = _imported_modules(SOURCE)
    roots = {module.split(".")[0] for module in modules}
    assert roots <= {"__future__", "hashlib", "json", "math", "dataclasses",
                     "typing", "mc_photogrammetry"}
    for module in modules:
        for forbidden in ("export", "litematic", "cli", "ui", "store", "project",
                          "camera", "manifest", "f3"):
            assert forbidden not in module, (module, forbidden)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not (calls & {"open", "exec", "eval", "compile", "__import__"})


def _code_names(path: Path) -> set[str]:
    """Every identifier, attribute, definition and keyword name in the module.

    Docstrings and comments are deliberately excluded: the module documents the
    forbidden surfaces it refuses, so a raw text scan would flag that
    documentation rather than any real reference.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def test_module_exposes_no_export_manifest_or_manual_symbols() -> None:
    assert not hasattr(promotion, "LitematicExporter")
    assert not hasattr(promotion, "Path")
    assert hasattr(promotion, "promote_plane_states")
    assert hasattr(promotion, "PromotionResult")
    assert hasattr(promotion, "PromotedBlock")
    assert hasattr(promotion, "PromotionRefusal")
    names = _code_names(SOURCE)
    for name in names:
        lowered = name.lower()
        for forbidden in ("litematic", "exporter", "manifest", "camera", "f3",
                          "manual"):
            assert forbidden not in lowered, (name, forbidden)
    # No direct exporter call and no exporter symbol anywhere in the source text.
    text = SOURCE.read_text(encoding="utf-8")
    assert ".export(" not in text
    assert "LitematicExporter" not in text
    assert promotion.PROMOTION_EXPORT_BOUNDS is None
    assert promotion.PROMOTION_SCALE_STATE == "unresolved"
    assert promotion.PROMOTION_NORMALIZATION_STATE == "unresolved"


def test_entry_point_takes_only_the_two_sources_and_explicit_budgets() -> None:
    signature = inspect.signature(promotion.promote_plane_states)
    assert list(signature.parameters) == [
        "volume", "proposals", "budget", "allow_omitted_unknown"]
    assert signature.parameters["budget"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["budget"].default is inspect.Parameter.empty
    assert signature.parameters["allow_omitted_unknown"].kind \
        is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["allow_omitted_unknown"].default is False
