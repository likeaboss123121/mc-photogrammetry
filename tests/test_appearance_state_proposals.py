"""Synthetic controls for the bounded M5 appearance/state proposal adapter.

These tests build every observation and palette in memory. They never read a
manifest, camera, bound, mask fixture or image: the adapter's whole contract is
that it consumes only accepted immutable ``FaceObservation`` tuples, an explicit
``PaletteColor`` tuple, and explicit thresholds and budgets.
"""

from __future__ import annotations

import ast
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction import appearance_state_proposals as proposals
from mc_photogrammetry.reconstruction.appearance_scoring import FaceScore
from mc_photogrammetry.reconstruction.appearance_state_proposals import (
    ABSTENTION_REASONS,
    AppearanceStateProposalError,
    FaceStateProposal,
    FaceStateProposalResult,
    propose_face_states,
)
from mc_photogrammetry.reconstruction.face_observation import NORMAL_OF_SLOT, FaceObservation

ROOT = Path(__file__).resolve().parents[1]
FRAME = "frame-0000"
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


def _patch(value: float, size: int = 4) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.float32)


def _rgb_patch(rgb: tuple[int, int, int], size: int = 4) -> np.ndarray:
    return np.full((size, size, 3), np.asarray(rgb, dtype=np.float32), dtype=np.float32)


def _observation(
    cell: tuple[int, int, int] = (0, 0, 0),
    slot: str = "north",
    *,
    patch: np.ndarray | None = None,
    frame_id: str = FRAME,
    support_frames: tuple[str, ...] | None = None,
    provenance: tuple[str, ...] = (),
    usable_coverage: float = 1.0,
    weight: float = 1.0,
    valid: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
) -> FaceObservation:
    return FaceObservation(
        cell=cell,
        slot=slot,
        frame_id=frame_id,
        normal=NORMAL_OF_SLOT[slot],
        patch=_patch(125.0) if patch is None else patch,
        usable_coverage=usable_coverage,
        support_frames=(frame_id,) if support_frames is None else support_frames,
        provenance=provenance,
        weight=weight,
        valid=valid,
        coverage=coverage,
    )


def _propose(
    observations: tuple[FaceObservation, ...],
    palette: tuple[PaletteColor, ...],
    **overrides: object,
) -> FaceStateProposalResult:
    options: dict[str, object] = {
        "max_cost": 1e9,
        "min_margin": 0.0,
        "max_observations": 1000,
        "max_candidate_evaluations": 100_000,
        "max_payload_bytes": 10_000_000,
    }
    options.update(overrides)
    return propose_face_states(observations, palette, **options)  # type: ignore[arg-type]


def _tampered(observation: FaceObservation, **overrides: object) -> FaceObservation:
    """A record that bypassed validation, as a hostile caller could build it."""
    clone = object.__new__(FaceObservation)
    for name in (
        "cell",
        "slot",
        "frame_id",
        "normal",
        "patch",
        "usable_coverage",
        "support_frames",
        "valid",
        "coverage",
        "provenance",
        "weight",
    ):
        object.__setattr__(clone, name, getattr(observation, name))
    for name, value in overrides.items():
        object.__setattr__(clone, name, value)
    return clone


def _frozen(array: np.ndarray) -> np.ndarray:
    """A read-only, ``bytes``-backed copy, as the accepted bridge retains."""
    contiguous = np.ascontiguousarray(array)
    frozen = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    ).reshape(contiguous.shape)
    frozen.setflags(write=False)
    return frozen


def _zero_budget_propose(observation: FaceObservation):
    """Propose with every budget exhausted, to prove validation runs first."""
    return propose_face_states(
        (observation,),
        (STONE,),
        max_cost=1.0,
        min_margin=0.0,
        max_observations=0,
        max_candidate_evaluations=0,
        max_payload_bytes=0,
    )


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


# --- acceptance and per-cell evidence ------------------------------------------------------


def test_exact_winner_is_proposed_with_full_provenance():
    first = _observation(
        patch=_patch(125.0), frame_id="f0", support_frames=("f0", "f-extra"), provenance=("cam/a",)
    )
    second = _observation(
        patch=_patch(125.0), frame_id="f1", support_frames=("f1",), provenance=("cam/b",)
    )
    result = _propose((first, second), (STONE, DIRT))

    assert result.status == "proposed"
    assert result.reason is None
    assert result.budget is None
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert isinstance(proposal, FaceStateProposal)
    assert proposal.proposed
    assert proposal.reason is None
    assert proposal.state == STONE.state
    assert proposal.cell == (0, 0, 0)
    assert proposal.observation_count == 2
    assert proposal.usable_observation_count == 2
    assert proposal.support_frame_ids == ("f-extra", "f0", "f1")
    assert proposal.provenance == ("cam/a", "cam/b")
    assert proposal.slots == ("north",)
    assert len(proposal.observation_digests) == 2
    assert len(set(proposal.observation_digests)) == 2
    assert proposal.cost_for(STONE.state) < proposal.cost_for(DIRT.state)
    assert [state for state, _cost in proposal.ranked()][0] == STONE.state
    assert len(proposal.candidate_costs) == 2
    assert proposal.max_cost == 1e9 and proposal.min_margin == 0.0
    assert result.candidate_evaluations == 4
    assert result.payload_bytes > 0
    assert result.payload_bytes <= 10_000_000


def test_single_candidate_palette_has_an_infinite_margin():
    result = _propose((_observation(),), (STONE,))
    proposal = result.proposals[0]
    assert proposal.proposed
    assert proposal.state == STONE.state
    assert proposal.runner_up_state is None
    assert proposal.margin == math.inf
    assert proposal.runner_up_cost == math.inf


def test_repeated_canonical_state_spellings_aggregate_to_one_candidate():
    bare = PaletteColor(
        BlockState("minecraft:oak_log"),
        (162, 130, 79),
        face_rgbs=((162, 130, 79),),
    )
    defaulted = PaletteColor(
        BlockState("minecraft:oak_log", (("axis", "y"),)),
        (162, 130, 79),
        face_rgbs=((162, 130, 79),),
    )
    result = _propose((_observation(patch=_patch(140.0)),), (bare, defaulted))
    proposal = result.proposals[0]
    assert len(proposal.candidate_costs) == 1
    assert result.palette_size == 2
    assert result.candidate_evaluations == 2


def test_palette_order_does_not_change_the_proposal():
    observation = _observation(patch=_patch(125.0))
    forward = _propose((observation,), (STONE, DIRT))
    reverse = _propose((observation,), (DIRT, STONE))
    assert forward.proposals[0].state == reverse.proposals[0].state == STONE.state
    assert forward.proposals[0].proposal_digest == reverse.proposals[0].proposal_digest
    assert forward.proposals[0].evidence_digest == reverse.proposals[0].evidence_digest
    assert forward.result_digest == reverse.result_digest


# --- typed review/abstention reasons -------------------------------------------------------


def test_weak_margin_abstains_typed():
    first = PaletteColor(
        BlockState("minecraft:aaa"),
        (125, 125, 125),
        face_rgbs=((125, 125, 125),),
        face_map=tuple((slot, (125, 125, 125)) for slot in FACE_SLOTS),
    )
    second = PaletteColor(
        BlockState("minecraft:bbb"),
        (125, 125, 125),
        face_rgbs=((125, 125, 125),),
        face_map=tuple((slot, (125, 125, 125)) for slot in FACE_SLOTS),
    )
    result = _propose((_observation(),), (first, second), min_margin=1e-6)
    proposal = result.proposals[0]
    assert result.status == "review"
    assert proposal.status == "review"
    assert proposal.reason == "weak_margin"
    assert proposal.state is None
    assert proposal.margin == pytest.approx(0.0, abs=1e-9)
    assert len(proposal.candidate_costs) == 2


def test_high_cost_abstains_typed():
    result = _propose((_observation(patch=_patch(255.0)),), (STONE, DIRT), max_cost=0.0)
    proposal = result.proposals[0]
    assert result.status == "review"
    assert proposal.reason == "high_cost"
    assert proposal.state is None
    assert math.isfinite(proposal.best_cost)
    assert proposal.best_cost > 0.0


def test_conflicting_same_slot_evidence_abstains_typed():
    grey_a = _observation(patch=_patch(125.0), frame_id="f0")
    grey_b = _observation(patch=_patch(125.0), frame_id="f1")
    brown = _observation(patch=_rgb_patch((134, 96, 67)), frame_id="f2")
    result = _propose((grey_a, grey_b, brown), (STONE, DIRT), max_cost=100.0, min_margin=1.0)
    proposal = result.proposals[0]
    assert proposal.reason == "conflicting_faces"
    assert proposal.state is None
    # The abstention is a conflict, not a threshold failure: both gates passed.
    assert proposal.best_cost <= 100.0
    assert proposal.margin >= 1.0
    assert proposal.usable_observation_count == 3


def test_different_slot_disagreement_is_a_margin_problem_not_a_conflict():
    grey = _observation(slot="north", patch=_patch(125.0), frame_id="f0")
    brown = _observation(slot="east", patch=_rgb_patch((134, 96, 67)), frame_id="f1")
    result = _propose((grey, brown), (STONE, DIRT), min_margin=1e-6)
    proposal = result.proposals[0]
    assert proposal.reason == "weak_margin"
    assert proposal.margin == pytest.approx(0.0, abs=1e-9)


def test_no_observation_and_empty_palette_are_result_level_abstentions():
    empty = _propose((), (STONE,))
    assert empty.status == "review" and empty.reason == "no_observation"
    assert empty.proposals == () and empty.observation_count == 0

    without_palette = _propose((_observation(),), ())
    assert without_palette.status == "review" and without_palette.reason == "empty_palette"
    assert without_palette.proposals == () and without_palette.palette_size == 0


def test_cell_with_only_unusable_observations_abstains_but_keeps_provenance():
    unusable = _observation(
        patch=_patch(125.0), usable_coverage=0.0, provenance=("cam/dropped",)
    )
    result = _propose((unusable,), (STONE, DIRT))
    proposal = result.proposals[0]
    assert result.status == "review"
    assert proposal.reason == "no_observation"
    assert proposal.state is None
    assert proposal.observation_count == 1
    assert proposal.usable_observation_count == 0
    assert proposal.candidate_costs == ()
    assert proposal.provenance == ("cam/dropped",)
    assert proposal.slots == ("north",)
    assert len(proposal.observation_digests) == 1


def test_unusable_observation_cannot_change_the_winner_but_is_retained():
    usable = _observation(patch=_patch(125.0), frame_id="f0")
    dropped = _observation(
        patch=_patch(134.0),
        frame_id="f1",
        usable_coverage=0.0,
        provenance=("cam/dropped",),
    )
    only = _propose((usable,), (STONE, DIRT))
    both = _propose((usable, dropped), (STONE, DIRT))
    assert only.proposals[0].state == both.proposals[0].state == STONE.state
    assert both.proposals[0].observation_count == 2
    assert both.proposals[0].usable_observation_count == 1
    assert both.proposals[0].support_frame_ids == ("f0", "f1")
    assert both.proposals[0].provenance == ("cam/dropped",)
    assert both.proposals[0].proposal_digest != only.proposals[0].proposal_digest


def test_every_review_reason_is_typed():
    tie = (
        PaletteColor(
            BlockState("minecraft:aaa"),
            (125, 125, 125),
            face_rgbs=((125, 125, 125),),
            face_map=tuple((slot, (125, 125, 125)) for slot in FACE_SLOTS),
        ),
        PaletteColor(
            BlockState("minecraft:bbb"),
            (125, 125, 125),
            face_rgbs=((125, 125, 125),),
            face_map=tuple((slot, (125, 125, 125)) for slot in FACE_SLOTS),
        ),
    )
    reasons = set()
    reasons.add(_propose((), (STONE,)).reason)
    reasons.add(_propose((_observation(),), ()).reason)
    reasons.add(_propose((_observation(),), tie, min_margin=1e-6).proposals[0].reason)
    high = _propose((_observation(patch=_patch(255.0)),), (STONE, DIRT), max_cost=0.0)
    reasons.add(high.proposals[0].reason)
    reasons.add(
        _propose(
            (
                _observation(patch=_patch(125.0), frame_id="f0"),
                _observation(patch=_patch(125.0), frame_id="f1"),
                _observation(patch=_rgb_patch((134, 96, 67)), frame_id="f2"),
            ),
            (STONE, DIRT),
            max_cost=100.0,
            min_margin=1.0,
        ).proposals[0].reason
    )
    reasons.add(_propose((_observation(),), (STONE,), max_observations=0).reason)
    assert reasons <= set(ABSTENTION_REASONS)
    assert {"no_observation", "empty_palette", "weak_margin", "high_cost",
            "conflicting_faces", "budget"} <= reasons


# --- deterministic ordering and digests ----------------------------------------------------


def test_input_order_does_not_change_proposals_or_digests():
    first = _observation(slot="north", patch=_patch(125.0), frame_id="f0")
    second = _observation(slot="east", patch=_rgb_patch((134, 96, 67)), frame_id="f1")
    forward = _propose((first, second), (STONE, DIRT))
    reverse = _propose((second, first), (STONE, DIRT))
    assert forward.status == reverse.status
    assert [p.proposal_digest for p in forward.proposals] == [
        p.proposal_digest for p in reverse.proposals
    ]
    assert forward.result_digest == reverse.result_digest
    assert forward.payload_sha256 == reverse.payload_sha256


def test_proposals_are_emitted_in_canonical_cell_order():
    far = _observation(cell=(2, 0, -1), patch=_patch(125.0), frame_id="f0")
    near = _observation(cell=(0, 0, 0), patch=_patch(125.0), frame_id="f0")
    result = _propose((far, near), (STONE, DIRT))
    assert [p.cell for p in result.proposals] == [(0, 0, 0), (2, 0, -1)]


def test_same_cell_only_evidence_is_isolated_between_cells():
    cell_a = _observation(cell=(0, 0, 0), patch=_patch(125.0), frame_id="a0")
    cell_b = _observation(cell=(1, 0, 0), patch=_rgb_patch((134, 96, 67)), frame_id="b0")
    both = _propose((cell_a, cell_b), (STONE, DIRT))
    alone = _propose((cell_a,), (STONE, DIRT))
    by_cell = {p.cell: p for p in both.proposals}
    assert by_cell[(0, 0, 0)].state == STONE.state
    assert by_cell[(1, 0, 0)].state == DIRT.state
    # A neighbour's evidence never enters this cell's digest or frames.
    assert by_cell[(0, 0, 0)].proposal_digest == alone.proposals[0].proposal_digest
    assert by_cell[(0, 0, 0)].support_frame_ids == ("a0",)
    assert by_cell[(1, 0, 0)].support_frame_ids == ("b0",)
    assert by_cell[(0, 0, 0)].evidence_digest != by_cell[(1, 0, 0)].evidence_digest


def test_digests_are_stable_and_bind_values_evidence_and_thresholds():
    observation = _observation(provenance=("cam/only",))
    first = _propose((observation,), (STONE, DIRT))
    second = _propose((observation,), (STONE, DIRT))
    assert first.result_digest == second.result_digest
    assert first.proposals[0].proposal_digest == second.proposals[0].proposal_digest

    other_palette = _propose((observation,), (DIRT,))
    assert other_palette.proposals[0].proposal_digest != first.proposals[0].proposal_digest

    other_threshold = _propose((observation,), (STONE, DIRT), max_cost=5.0)
    assert other_threshold.proposals[0].proposal_digest != first.proposals[0].proposal_digest

    other_evidence = _propose((_observation(patch=_patch(126.0), provenance=("cam/only",)),),
                              (STONE, DIRT))
    assert other_evidence.proposals[0].proposal_digest != first.proposals[0].proposal_digest


def test_stored_digests_match_their_payloads():
    result = _propose((_observation(),), (STONE, DIRT))
    proposal = result.proposals[0]
    assert proposal.proposal_digest == proposals._sha256(proposals._proposal_payload(proposal))
    encoded = proposals._canonical_json(proposals._result_payload(result))
    assert result.payload_sha256 == proposals._sha256(encoded)
    assert result.payload_bytes == len(encoded.encode("utf-8"))


def test_observation_digest_binds_each_array_shape():
    short = _observation(
        patch=np.arange(6, dtype=np.float32).reshape(2, 1, 3),
        valid=np.ones((2, 1), dtype=bool),
        coverage=np.ones((2, 1), dtype=np.float64),
    )
    wide = _observation(
        patch=np.arange(6, dtype=np.float32).reshape(1, 2, 3),
        valid=np.ones((1, 2), dtype=bool),
        coverage=np.ones((1, 2), dtype=np.float64),
    )
    # Identity fields match and every retained array is byte-identical; only the
    # shapes differ, so a bytes-only digest would collide.
    assert short.identity == wide.identity
    for first, second in (
        (short.patch, wide.patch),
        (short.valid, wide.valid),
        (short.coverage, wide.coverage),
    ):
        assert second is not None
        assert first.shape != second.shape
        assert first.tobytes(order="C") == second.tobytes(order="C")
    assert proposals._observation_digest(short) != proposals._observation_digest(wide)


def test_observation_digest_binds_each_array_dtype():
    base = _observation(
        patch=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        valid=np.ones((2, 2), dtype=bool),
        coverage=np.ones((2, 2), dtype=np.float64),
    )
    patch_bytes = base.patch.tobytes(order="C")
    valid_bytes = base.valid.tobytes(order="C") if base.valid is not None else b""
    coverage_bytes = base.coverage.tobytes(order="C") if base.coverage is not None else b""
    retyped_patch = _tampered(
        base,
        patch=np.frombuffer(patch_bytes, dtype=np.int32).reshape(base.patch.shape),
    )
    retyped_valid = _tampered(
        base,
        valid=np.frombuffer(valid_bytes, dtype=np.uint8).reshape(base.valid.shape),  # type: ignore[union-attr]
    )
    retyped_coverage = _tampered(
        base,
        coverage=np.frombuffer(coverage_bytes, dtype=np.int64).reshape(base.coverage.shape),  # type: ignore[union-attr]
    )
    assert base.patch.dtype == np.float32 and retyped_patch.patch.dtype == np.int32
    assert base.valid.dtype == np.bool_ and retyped_valid.valid.dtype == np.uint8
    assert base.coverage.dtype == np.float64 and retyped_coverage.coverage.dtype == np.int64
    for variant in (retyped_patch, retyped_valid, retyped_coverage):
        assert variant.patch.shape == base.patch.shape
        assert proposals._observation_digest(variant) != proposals._observation_digest(base)


# --- budgets are explicit and atomic -------------------------------------------------------


class _RecordingScorer:
    calls = 0
    constructions = 0

    def __init__(self, palette: tuple[PaletteColor, ...]) -> None:
        type(self).constructions += 1
        self.palette = palette

    def score(self, observation: FaceObservation) -> FaceScore:
        type(self).calls += 1
        from mc_photogrammetry.reconstruction.appearance_scoring import AppearanceScorer

        return AppearanceScorer(self.palette).score(observation)


@pytest.fixture()
def recording_scorer(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingScorer]:
    _RecordingScorer.calls = 0
    _RecordingScorer.constructions = 0
    monkeypatch.setattr(proposals, "AppearanceScorer", _RecordingScorer)
    return _RecordingScorer


def test_max_observations_refuses_atomically_without_scoring(recording_scorer):
    observations = (
        _observation(frame_id="f0"),
        _observation(frame_id="f1"),
    )
    result = _propose(observations, (STONE, DIRT), max_observations=1)
    assert result.status == "budget"
    assert result.reason == "budget"
    assert result.budget == "max_observations"
    assert result.proposals == ()
    assert result.candidate_evaluations == 0
    assert recording_scorer.calls == 0
    assert recording_scorer.constructions == 0

    exact = _propose(observations, (STONE, DIRT), max_observations=2)
    assert exact.status == "proposed"


def test_max_candidate_evaluations_refuses_atomically_without_scoring(recording_scorer):
    result = _propose(
        (_observation(),), (STONE, DIRT), max_candidate_evaluations=1
    )
    assert result.status == "budget"
    assert result.budget == "max_candidate_evaluations"
    assert result.proposals == ()
    assert result.candidate_evaluations == 0
    assert recording_scorer.calls == 0
    assert recording_scorer.constructions == 0

    exact = _propose((_observation(),), (STONE, DIRT), max_candidate_evaluations=2)
    assert exact.status == "proposed"


def test_max_payload_bytes_refuses_atomically_after_scoring(recording_scorer):
    result = _propose((_observation(),), (STONE, DIRT), max_payload_bytes=0)
    assert result.status == "budget"
    assert result.budget == "max_payload_bytes"
    assert result.proposals == ()
    assert result.candidate_evaluations == 2
    assert recording_scorer.calls == 1


def test_payload_budget_boundary_accepts_the_exact_size():
    generous = _propose((_observation(),), (STONE, DIRT))
    exact = _propose(
        (_observation(),), (STONE, DIRT), max_payload_bytes=generous.payload_bytes
    )
    assert exact.status == "proposed"
    assert exact.payload_bytes == generous.payload_bytes


def test_validation_precedes_budget():
    with pytest.raises(AppearanceStateProposalError):
        propose_face_states(
            [_observation()],  # type: ignore[arg-type]
            (STONE,),
            max_cost=1.0,
            min_margin=0.0,
            max_observations=0,
            max_candidate_evaluations=0,
            max_payload_bytes=0,
        )
    with pytest.raises(AppearanceStateProposalError):
        propose_face_states(
            (object(),),  # type: ignore[arg-type]
            (STONE,),
            max_cost=1.0,
            min_margin=0.0,
            max_observations=0,
            max_candidate_evaluations=0,
            max_payload_bytes=0,
        )


def test_request_budgets_and_thresholds_are_validated():
    good = dict(
        max_cost=1.0,
        min_margin=0.0,
        max_observations=10,
        max_candidate_evaluations=10,
        max_payload_bytes=10,
    )
    for name, bad in (
        ("max_cost", -1.0),
        ("max_cost", float("nan")),
        ("max_cost", float("inf")),
        ("min_margin", -0.5),
        ("min_margin", float("nan")),
        ("max_observations", -1),
        ("max_observations", True),
        ("max_candidate_evaluations", 1.5),
        ("max_payload_bytes", -3),
    ):
        options = dict(good)
        options[name] = bad
        with pytest.raises(AppearanceStateProposalError):
            propose_face_states((_observation(),), (STONE,), **options)  # type: ignore[arg-type]


# --- malformed inputs and malformed scorer output ------------------------------------------


def test_malformed_observation_is_a_typed_error():
    valid = _observation()
    cases = [
        (object(),),
        (_tampered(valid, cell=(0, 0)),),
        (_tampered(valid, slot="sideways"),),
        (_tampered(valid, usable_coverage=2.0),),
        (_tampered(valid, support_frames=("other",)),),
        (_tampered(valid, weight=float("nan")),),
        (_tampered(valid, patch=np.zeros((4, 4), dtype=np.float32)),),
        (_tampered(valid, provenance=("dup", "dup")),),
    ]
    for observations in cases:
        with pytest.raises(AppearanceStateProposalError):
            _propose(observations, (STONE,))  # type: ignore[arg-type]


def test_writable_or_reopenable_retained_arrays_are_refused_even_with_zero_budgets():
    base = _observation(
        valid=np.ones((4, 4), dtype=bool),
        coverage=np.full((4, 4), 0.5, dtype=np.float64),
    )
    writable_patch = np.full((4, 4, 3), 125.0, dtype=np.float32)
    reopenable_patch = np.full((4, 4, 3), 125.0, dtype=np.float32)
    reopenable_patch.setflags(write=False)
    writable_valid = np.ones((4, 4), dtype=bool)
    reopenable_valid = np.ones((4, 4), dtype=bool)
    reopenable_valid.setflags(write=False)
    writable_coverage = np.full((4, 4), 0.5, dtype=np.float64)
    reopenable_coverage = np.full((4, 4), 0.5, dtype=np.float64)
    reopenable_coverage.setflags(write=False)
    cases = [
        _tampered(base, patch=writable_patch),
        _tampered(base, patch=reopenable_patch),
        _tampered(base, valid=writable_valid),
        _tampered(base, valid=reopenable_valid),
        _tampered(base, coverage=writable_coverage),
        _tampered(base, coverage=reopenable_coverage),
    ]
    for observation in cases:
        with pytest.raises(AppearanceStateProposalError):
            _zero_budget_propose(observation)


def test_retained_arrays_must_keep_their_accepted_dtypes_with_zero_budgets():
    base = _observation(
        valid=np.ones((4, 4), dtype=bool),
        coverage=np.full((4, 4), 0.5, dtype=np.float64),
    )
    cases = [
        _tampered(base, patch=_frozen(np.zeros((4, 4, 3), dtype=np.float64))),
        _tampered(base, coverage=_frozen(np.full((4, 4), 0.5, dtype=np.float32))),
        _tampered(base, valid=_frozen(np.ones((4, 4), dtype=np.int8))),
    ]
    for observation in cases:
        with pytest.raises(AppearanceStateProposalError):
            _zero_budget_propose(observation)


def test_accepted_retained_arrays_still_validate_and_propose():
    observation = _observation(
        valid=np.ones((4, 4), dtype=bool),
        coverage=np.full((4, 4), 0.5, dtype=np.float64),
    )
    for array in (observation.patch, observation.valid, observation.coverage):
        assert array is not None
        assert not array.flags.writeable
        assert proposals._has_immutable_bytes_base(array)
    with pytest.raises(ValueError):
        observation.patch.setflags(write=True)
    result = _propose((observation,), (STONE, DIRT))
    assert result.status == "proposed"
    assert result.proposals[0].state == STONE.state


def test_palette_must_be_an_immutable_palette_tuple():
    with pytest.raises(AppearanceStateProposalError):
        _propose((_observation(),), [STONE])  # type: ignore[arg-type]
    with pytest.raises(AppearanceStateProposalError):
        _propose((_observation(),), (object(),))  # type: ignore[arg-type]


def _fake_scorer(payload: object) -> type:
    class _Fake:
        def __init__(self, palette: tuple[PaletteColor, ...]) -> None:
            self.palette = palette

        def score(self, observation: FaceObservation) -> object:
            if callable(payload):
                return payload(observation)
            return payload

    return _Fake


def test_malformed_scorer_output_is_a_typed_error(monkeypatch: pytest.MonkeyPatch):
    valid = _observation()
    cases = [
        object(),
        FaceScore((), float("inf"), abstained=True, reason="zero_coverage"),
        FaceScore(((STONE.state, 0.0),), 1.0),
        FaceScore(((BlockState("minecraft:bogus"), 0.0), (DIRT.state, 1.0)), 1.0),
        FaceScore(((STONE.state, 0.0), (DIRT.state, float("inf"))), 1.0),
        FaceScore(((STONE.state, 0.0), (DIRT.state, -1.0)), 1.0),
        FaceScore(((STONE.state, 0.0), (DIRT.state, 1.0)), float("nan")),
    ]
    for payload in cases:
        monkeypatch.setattr(proposals, "AppearanceScorer", _fake_scorer(payload))
        with pytest.raises(AppearanceStateProposalError):
            _propose((valid,), (STONE, DIRT))


def test_scorer_failure_is_wrapped_typed(monkeypatch: pytest.MonkeyPatch):
    class _RaisingScorer:
        def __init__(self, palette: tuple[PaletteColor, ...]) -> None:
            self.palette = palette

        def score(self, observation: FaceObservation) -> FaceScore:
            raise ValueError("scorer exploded")

    monkeypatch.setattr(proposals, "AppearanceScorer", _RaisingScorer)
    with pytest.raises(AppearanceStateProposalError):
        _propose((_observation(),), (STONE, DIRT))

    class _BrokenConstruction:
        def __init__(self, palette: tuple[PaletteColor, ...]) -> None:
            raise ValueError("palette exploded")

    monkeypatch.setattr(proposals, "AppearanceScorer", _BrokenConstruction)
    with pytest.raises(AppearanceStateProposalError):
        _propose((_observation(),), (STONE, DIRT))


# --- boundaries ----------------------------------------------------------------------------


def test_no_export_or_volume_path_exists():
    module_path = ROOT / "src/mc_photogrammetry/reconstruction/appearance_state_proposals.py"
    modules = _imported_modules(module_path)
    for module in modules:
        for forbidden in (
            "facade",
            "raycast",
            "occupancy",
            "litematic",
            "export",
            "cli",
            "ui",
            "store",
            "shape_fit",
            "volume",
        ):
            assert forbidden not in module, (module, forbidden)
    assert "BlockVolume" not in module_path.read_text(encoding="utf-8")
    assert not hasattr(proposals, "BlockVolume")
    assert not hasattr(proposals, "BlockObservation")
    for name in dir(proposals):
        assert not name.lower().startswith("export")
        assert "volume" not in name.lower()


def test_entry_point_takes_only_observations_palette_and_explicit_limits():
    import inspect

    signature = inspect.signature(propose_face_states)
    assert list(signature.parameters) == [
        "observations",
        "palette",
        "max_cost",
        "min_margin",
        "max_observations",
        "max_candidate_evaluations",
        "max_payload_bytes",
    ]
    for name in ("max_cost", "min_margin", "max_observations",
                 "max_candidate_evaluations", "max_payload_bytes"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is inspect.Parameter.empty


def test_records_are_immutable():
    result = _propose((_observation(),), (STONE, DIRT))
    with pytest.raises(FrozenInstanceError):
        result.status = "review"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.proposals[0].reason = "weak_margin"  # type: ignore[misc]


def test_stored_cost_records_are_consistent():
    result = _propose((_observation(),), (STONE, DIRT))
    proposal = result.proposals[0]
    assert proposal.best_cost == pytest.approx(proposal.candidate_costs[0][1])
    assert proposal.runner_up_cost == pytest.approx(proposal.candidate_costs[1][1])
    assert proposal.margin == pytest.approx(
        proposal.runner_up_cost - proposal.best_cost
    )
    assert [cost for _state, cost in proposal.candidate_costs] == sorted(
        cost for _state, cost in proposal.candidate_costs
    )
