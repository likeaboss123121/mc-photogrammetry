"""Adversarial controls for the bounded M6 image-derived orchestration facade.

Every test drives the real accepted chain, not a stand-in: the genuine V3 mixed
normal recipe from ``tests/test_plane_placement.py`` supplies the universe,
supports, placement and volume; the accepted M5 bridge supplies the observations;
the accepted proposal and promotion adapters supply the proposal and promotion;
and the accepted connection and pose-graph seams supply the graph.  The only
injected pieces are the declared test seams the accepted adapters themselves
expose (``image_only.solve_image_cameras`` and the connection ``matcher``), so no
SIFT runs and no pixels are read from disk.

The controls are deliberately hostile.  They re-forge a record past its
constructor, swap one accepted record for another coherent one built from a
different source, force a review/abstention, force an ineligible promotion,
force a delivery refusal, and try to smuggle manual metadata - and every one of
them must produce a typed, atomic :class:`PipelineRefusal` or the exact typed
refusal of the accepted seam, never a partial outcome or an escaped exception.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import test_plane_placement as recipe_tests  # noqa: E402

from mc_photogrammetry.blocks.appearance import PaletteColor  # noqa: E402
from mc_photogrammetry.domain.models import BlockState  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_connections as connections  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_delivery as delivery  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_orchestration as m6  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_session as session_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_only  # noqa: E402
from mc_photogrammetry.pipeline.contracts import (  # noqa: E402
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
    CameraSolution,
)
from mc_photogrammetry.reconstruction import appearance_state_proposals as ap  # noqa: E402
from mc_photogrammetry.reconstruction import plane_face_observation as m5  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402
from mc_photogrammetry.reconstruction import plane_volume as pv  # noqa: E402
from mc_photogrammetry.reconstruction import plane_region_support as v1  # noqa: E402
from mc_photogrammetry.reconstruction import state_promotion as sp  # noqa: E402
from mc_photogrammetry.reconstruction.face_observation import (  # noqa: E402
    NORMAL_OF_SLOT,
    FaceObservation,
)

SOURCE = Path(m6.__file__)
SESSION_FRAMES = ("R0", "R1", "R2")
PROPOSAL_OPTIONS = dict(
    max_cost=1e9,
    min_margin=0.0,
    max_observations=1000,
    max_candidate_evaluations=100_000,
    max_payload_bytes=10_000_000,
)
PROMOTION_CAPS = dict(max_cells=1000, max_proposals=1000, max_payload_bytes=10_000_000)
PATCH_PIXELS = int(m5.face_sampling.PATCH_SIZE) ** 2
OBSERVATION_BYTES = PATCH_PIXELS * 3 * 4 + PATCH_PIXELS + PATCH_PIXELS * 8
SLOTS = ("east", "west", "up", "down", "south", "north")


# ---------------------------------------------------------------------------
# Palette, session and graph fixtures
# ---------------------------------------------------------------------------

def _palette_color(name: str, rgb: tuple[int, int, int]) -> PaletteColor:
    return PaletteColor(
        BlockState(name), rgb, face_rgbs=(rgb,), family="cube",
        face_map=tuple((slot, rgb) for slot in SLOTS), full_cube=True)


#: A permissive, deterministic material palette for the real recipe observations.
PALETTE = tuple(
    _palette_color(name, rgb)
    for name, rgb in (
        ("minecraft:stone", (125, 125, 125)),
        ("minecraft:dirt", (134, 96, 67)),
        ("minecraft:sand", (219, 207, 163)),
        ("minecraft:white_wool", (233, 236, 236)),
        ("minecraft:black_wool", (29, 29, 33)),
        ("minecraft:oak_planks", (162, 130, 78)),
        ("minecraft:cobblestone", (127, 127, 127)),
    ))


def _canonical_unresolved(frame_ids) -> tuple[str, ...]:
    notes: list[str] = []
    for frame_id in frame_ids:
        notes.append(f"{frame_id}: position (vanishing points fix no translation)")
        notes.append(f"{frame_id}: scale (needs the lattice period)")
        notes.append(f"{frame_id}: yaw quadrant (recovered modulo 90 degrees)")
    return tuple(notes)


def _fake_estimate(yaw: float = 14.0) -> CameraEstimate:
    return CameraEstimate(
        intrinsics=CameraIntrinsics((80, 60), (100.0, 100.0), (40.0, 30.0)),
        pose=CameraPose(position=None, yaw_degrees=yaw, pitch_degrees=3.0,
                        roll_degrees=0.0),
        residual_pixels=0.4,
        confidence=0.0,
    )


def _session(monkeypatch, frame_ids, *, seed_offset: int = 0):
    estimates = {frame_id: _fake_estimate() for frame_id in frame_ids}

    def fake_solve(frames):
        return CameraSolution(
            cameras={frame_id: estimates[frame_id] for frame_id in frames},
            unresolved=_canonical_unresolved(list(frames)),
            scale_known=False,
        )

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)
    return session_mod.build_image_derived_session({
        frame_id: np.random.default_rng(seed_offset + index).integers(
            0, 256, size=(60, 80, 3), dtype=np.uint8)
        for index, frame_id in enumerate(frame_ids)})


def _refused_pair(left_id: str, right_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        left_id=left_id, right_id=right_id, status="refused",
        reason="candidate_graph_empty", solved=False, pose=None,
        right_yaw_class=None, matches=np.empty((0, 4), dtype=np.float64),
        support=0, support_floor=8, raw_keypoints=(0, 0), candidate_edges=0,
        mutual_survivors=0, median_selected_residual_degrees=None,
        parallax_degrees=None, cheirality_fraction=None, singular_ratio=None,
        ambiguity_margin=None, hypotheses=(),
    )


def _unit(vector) -> tuple[float, float, float]:
    values = np.asarray(vector, dtype=np.float64)
    values = values / float(np.linalg.norm(values))
    return (float(values[0]), float(values[1]), float(values[2]))


def _solved_pair(left_id, right_id, baseline, *, support: int = 6) -> SimpleNamespace:
    return SimpleNamespace(
        left_id=left_id, right_id=right_id, status="solved", reason="", solved=True,
        pose=SimpleNamespace(
            baseline_direction=tuple(float(value) for value in baseline),
            relative_rotation=np.eye(3, dtype=np.float64), solved=True),
        right_yaw_class=0,
        matches=np.arange(4 * support, dtype=np.float64).reshape(support, 4),
        support=support, support_floor=8, raw_keypoints=(support, support),
        candidate_edges=support * 2, mutual_survivors=support,
        median_selected_residual_degrees=0.4, parallax_degrees=5.0,
        cheirality_fraction=1.0, singular_ratio=0.01, ambiguity_margin=3,
        hypotheses=(),
    )


#: A rotation/direction-consistent unit baseline table for (R0, R1, R2).
BASELINES = {
    ("R0", "R1"): (1.0, 0.0, 0.0),
    ("R0", "R2"): _unit((1.0, 1.0, 0.0)),
    ("R1", "R2"): (0.0, 1.0, 0.0),
}


def _graph(monkeypatch, frame_ids, *, seed_offset: int = 0):
    session = _session(monkeypatch, frame_ids, seed_offset=seed_offset)

    def matcher(left_id, left_image, right_id, right_image, *, left_camera,
                right_camera):
        baseline = BASELINES.get((left_id, right_id))
        if baseline is None:
            return _refused_pair(left_id, right_id)
        return _solved_pair(left_id, right_id, baseline)

    return connections.build_image_derived_connections(
        session,
        budgets=connections.make_connection_budgets(
            max_pairs=64, max_total_support=10 ** 9),
        matcher=matcher)


def _pose_graph(graph):
    return pose_mod.compose_image_derived_pose_graph(
        graph,
        budgets=pose_mod.make_pose_graph_budgets(
            max_edges=64, max_cycle_checks=64, rotation_tolerance_degrees=1.0,
            direction_tolerance=1e-9))


# ---------------------------------------------------------------------------
# The genuine V3/M5 fixture
# ---------------------------------------------------------------------------

def _volume_budget(**overrides):
    caps = dict(
        max_faces=9, max_support_cells=81, max_contributions=66,
        max_coordinates=58, max_conflicts=1, max_omitted_unknown=15,
        max_dense_bounds_cells=224)
    caps.update(overrides)
    return pv.make_volume_budget(**caps)


def _face_budget(**overrides):
    caps = dict(
        max_claims=66, max_frames=3, max_rectified_pixels=66 * PATCH_PIXELS,
        max_payload_bytes=66 * OBSERVATION_BYTES, max_replay_checks=21)
    caps.update(overrides)
    return m5.make_face_observation_budget(**caps)


@pytest.fixture(scope="module")
def recipe():
    """The accepted V3/M5 recipe: universe, supports, placement, volume, frames."""
    fixture = recipe_tests._build_recipe_fixture()
    universe = recipe_tests._recipe_universe(fixture)
    supports = recipe_tests._recipe_supports(fixture)
    placement_budget = recipe_tests._budget()
    placement = plane_placement.resolve_face_placement(
        universe=universe, supports=supports, budget=placement_budget)
    assert isinstance(placement, plane_placement.ResolvedFacePlacement)
    volume_budget = _volume_budget()
    volume = pv.build_plane_volume(
        universe=universe, supports=supports, placement=placement,
        placement_budget=placement_budget, budget=volume_budget)
    assert isinstance(volume, pv.PlaneVolumeResult)
    image = np.random.default_rng(992).random((1024, 1024, 3), dtype=np.float32) + 0.1
    frames = []
    for frame in recipe_tests.RECIPE_FRAMES:
        readings = tuple(reading for reading in universe.readings
                         if reading.frame == frame)
        masks = {reading.axis: recipe_tests._mask_for(reading.axis)
                 for reading in readings}
        frames.append(m5.frame_context_from_readings(
            frame_id=frame, image=image, image_sha256=recipe_tests.IMAGE_SHA,
            usable_mask=masks[readings[0].axis], readings=readings))
    return {
        "universe": universe, "supports": supports, "placement": placement,
        "placement_budget": placement_budget, "volume_budget": volume_budget,
        "volume": volume, "frames": tuple(frames), "face_budget": _face_budget(),
        "image": image,
    }


@pytest.fixture(scope="module")
def m5_result(recipe):
    built = m5.build_plane_face_observations(
        recipe["volume"], universe=recipe["universe"], supports=recipe["supports"],
        placement=recipe["placement"], placement_budget=recipe["placement_budget"],
        volume_budget=recipe["volume_budget"], frames=recipe["frames"],
        budget=recipe["face_budget"])
    assert isinstance(built, m5.PlaneFaceObservationResult), built
    return built


@pytest.fixture
def pipeline(recipe, m5_result, monkeypatch):
    """Every accepted record the facade consumes, plus its explicit parameters."""
    session = _session(monkeypatch, SESSION_FRAMES)
    graph = _graph(monkeypatch, SESSION_FRAMES)
    pose_graph = _pose_graph(graph)
    assert graph.status == "ok" and pose_graph.status == "ok"
    return {
        "session": session,
        "graph": graph,
        "pose_graph": pose_graph,
        "volume": recipe["volume"],
        "universe": recipe["universe"],
        "supports": recipe["supports"],
        "placement": recipe["placement"],
        "placement_budget": recipe["placement_budget"],
        "volume_budget": recipe["volume_budget"],
        "frames": recipe["frames"],
        "face_observations": m5_result,
        "face_budget": recipe["face_budget"],
        "palette": PALETTE,
        "promotion_budget": sp.make_promotion_budget(**PROMOTION_CAPS),
        **PROPOSAL_OPTIONS,
    }


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _synthetic_claim(coordinate, index: int):
    return pv.CellClaim.build(
        frame="f0", plane_index=0, reading_axis=0, absolute_quadrant=0,
        source_region_id=_h(f"region{index}"),
        reading_identity_sha256=_h(f"reading{index}"),
        support_sha256=_h(f"support{index}"), face_sha256=_h(f"face{index}"),
        row=0, column=index,
        world_normal_axis=0, world_normal_sign=-1, outward_sign=-1,
        face=coordinate[0], normal_block=coordinate[0], tangent_u_axis=1,
        lateral_u=coordinate[1], tangent_v_axis=2, lateral_v=coordinate[2],
        coordinate=coordinate)


def _synthetic_promotion(coordinates=((0, 0, 0), (1, 0, 0))):
    """One genuinely eligible promotion, built from an independent synthetic volume.

    It is internally valid and exact-state eligible, but it binds a different
    volume, so it exists only to prove the facade's ``expected_promotion``
    cross-check (never to be delivered).
    """
    claims = [_synthetic_claim(coordinate, index)
              for index, coordinate in enumerate(coordinates)]
    cells = tuple(pv.SupportedCellObservation.build(
        coordinate=coordinate, claims=(claim,), frame_ids=(claim.frame,),
        conflict_kind=None) for coordinate, claim in zip(coordinates, claims))
    binding_records = [
        {
            "frame": claim.frame, "plane_index": 0, "reading_axis": 0,
            "reading_identity_sha256": claim.reading_identity_sha256,
            "support_sha256": claim.support_sha256, "face_sha256": claim.face_sha256,
            "source_region_id": claim.source_region_id,
            "alternative_distance": [0, 1], "phase_anchor_sha256": _h("anchor"),
            "support_cells": 1, "supported_cells": 1, "unknown_cells": 0,
        }
        for claim in claims]
    bindings = tuple(sorted(binding_records, key=pv._binding_sort_key))
    low = tuple(min(coordinate[axis] for coordinate in coordinates) for axis in range(3))
    high = tuple(max(coordinate[axis] for coordinate in coordinates) for axis in range(3))
    dense = 1
    for axis in range(3):
        dense *= high[axis] - low[axis] + 1
    work = pv._make_work({
        "faces": len(bindings), "support_cells": len(bindings),
        "contributions": len(cells), "coordinates": len(cells), "conflicts": 0,
        "omitted_unknown": 0, "dense_bounds_cells": dense})
    volume = pv.PlaneVolumeResult.build(
        schema_version=pv.V3_VOLUME_SCHEMA,
        producer_version=pv.V3_VOLUME_PRODUCER_VERSION,
        gauge_state=pv.V3_VOLUME_GAUGE_STATE, gauge_axes=(0, 1, 2),
        alpha=((0, 1), (0, 1), (0, 1)), placement_sha256=_h("synthetic-placement"),
        support_bindings=bindings, cells=cells,
        conflicts=(), omitted=(), cell_bounds=(low, high),
        dense_bounds_cells=dense, normalization_offset=low, export_bounds=None,
        complete=False, work=work)
    observations = tuple(
        FaceObservation(
            cell=coordinate, slot="north", frame_id="f0",
            normal=NORMAL_OF_SLOT["north"],
            patch=np.full((4, 4, 3), 125.0, dtype=np.float32),
            usable_coverage=1.0, support_frames=("f0",), provenance=("cam/a",))
        for coordinate in coordinates)
    proposals = ap.propose_face_states(
        observations, PALETTE, max_cost=1e9, min_margin=0.0, max_observations=1000,
        max_candidate_evaluations=100_000, max_payload_bytes=10_000_000)
    assert proposals.status == ap.PROPOSED
    promotion = sp.promote_plane_states(
        volume, proposals, budget=sp.make_promotion_budget(max_cells=16,
                                                           max_proposals=16,
                                                           max_payload_bytes=1_000_000))
    assert isinstance(promotion, sp.PromotionResult), promotion
    assert promotion.exact_state_eligible is True
    return promotion


def _compose(pipeline, **overrides):
    return m6.compose_image_derived_pipeline(**{**pipeline, **overrides})


def _assert_refusal(outcome, reason: str) -> m6.PipelineRefusal:
    assert isinstance(outcome, m6.PipelineRefusal), outcome
    assert outcome.reason == reason, outcome.to_record()
    assert "cells" not in outcome.detail and "artifact" not in outcome.detail
    return outcome


def _frozen_uint8(array) -> np.ndarray:
    """A read-only, ``bytes``-backed ``uint8`` copy, as the accepted producer retains."""
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.uint8)
    frozen = frozen.reshape(contiguous.shape)
    frozen.setflags(write=False)
    return frozen


def _clone(record, **changes):
    """Copy a frozen record without re-running its validating constructor."""
    clone = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(clone, field.name, getattr(record, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


def _reseal_graph(graph, **changes):
    """Reseal a graph clone's *enclosing* digest without running its constructor.

    This is the hostile shape the nested replay must reject: the outer digest is
    self-consistent with the tampered payload, but a nested record never passed
    its own accepted constructor.
    """
    clone = _clone(graph, **changes)
    return _clone(clone, graph_sha256=connections._digest(
        connections._graph_payload(clone)))


def _reseal_pose_graph(pose_graph, **changes):
    clone = _clone(pose_graph, **changes)
    return _clone(clone, pose_graph_sha256=pose_mod._digest(
        pose_mod._pose_graph_payload(clone)))


def _reseal_evidence(record, **changes):
    """Reseal one evidence clone past its constructor so only binding can refuse."""
    clone = _clone(record, **changes)
    return _clone(clone, evidence_sha256=connections._digest(
        connections._evidence_payload(clone)))


def _reseal_promotion(record):
    """Reseal a promotion clone's payload digests around a nested tamper."""
    payload = sp._result_payload(record)
    encoded = sp._canonical_json(payload)
    payload_sha = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    promotion_sha = sp._sha256({**payload, "payload_sha256": payload_sha,
                                "max_payload_bytes": record.max_payload_bytes})
    return _clone(record, payload_bytes=len(encoded.encode("utf-8")),
                  payload_sha256=payload_sha, promotion_sha256=promotion_sha)


def _outcome_fields(outcome):
    """Every carried outcome field except the enclosing digest."""
    return {field.name: getattr(outcome, field.name)
            for field in dataclasses.fields(outcome)
            if field.name != "outcome_sha256"}


# ---------------------------------------------------------------------------
# Composed outcome
# ---------------------------------------------------------------------------

def test_permissive_promotion_reviews_the_unknown_cells(pipeline):
    """The genuine recipe has 15 unknown cells, so the default promotion reviews."""
    outcome = _compose(pipeline)
    _assert_refusal(outcome, "promotion_refused")
    assert outcome.detail["reason"] == "omitted_unknown"
    assert outcome.detail["kind"] == "review"


def test_opt_in_promotion_composes_an_ineligible_outcome(pipeline):
    """An opted-in omission promotes material states but is not exact-state eligible."""
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome), outcome
    assert outcome.exact_state_eligible is False
    assert outcome.delivered is False
    assert outcome.delivery is None
    assert outcome.omitted_unknown == 15
    assert outcome.export_ready is True
    assert outcome.scale_state == "unresolved"
    assert outcome.normalization_state == "unresolved"
    assert outcome.session is pipeline["session"]
    assert outcome.volume is pipeline["volume"]
    assert outcome.promotion.volume_sha256 == pipeline["volume"].volume_sha256
    assert outcome.proposal_result_digest == outcome.proposals.result_digest
    assert outcome.promotion.proposal_result_digest == outcome.proposals.result_digest
    assert outcome.unresolved == pipeline["pose_graph"].unresolved


def test_outcome_record_is_immutable_and_digest_bound(pipeline):
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    for field in dataclasses.fields(outcome):
        if field.name == "outcome_sha256":
            continue
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(outcome, field.name, None)  # type: ignore[misc]
    record = outcome.to_record()
    assert record["outcome_sha256"] == outcome.outcome_sha256
    recomputed = m6._sha256(m6._outcome_payload(outcome))
    assert recomputed == outcome.outcome_sha256
    forged = _clone(outcome, promotion_sha256="0" * 64)
    with pytest.raises(ValueError):
        m6.PipelineOutcome.__post_init__(forged)


def test_outcome_is_deterministic_across_identical_calls(pipeline):
    first = _compose(pipeline, allow_omitted_unknown=True)
    second = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(first, m6.PipelineOutcome)
    assert isinstance(second, m6.PipelineOutcome)
    assert first.to_record() == second.to_record()


def test_expected_promotion_must_bind_the_composed_records(pipeline):
    good = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(good, m6.PipelineOutcome)
    matched = _compose(pipeline, allow_omitted_unknown=True,
                       expected_promotion=good.promotion)
    assert isinstance(matched, m6.PipelineOutcome)
    assert matched.promotion_sha256 == good.promotion_sha256

    # A genuinely eligible promotion that binds a different volume must be a
    # typed source mismatch, not a silently accepted expectation.
    _assert_refusal(_compose(pipeline, allow_omitted_unknown=True,
                             expected_promotion=_synthetic_promotion()),
                    "source_mismatch")


def test_facade_does_not_mutate_any_accepted_input(pipeline):
    """Composition replays the records; it never edits, reseals or reorders them."""
    session_record = session_mod.image_derived_session_to_record
    snapshot = {
        "session": session_record(pipeline["session"]),
        "graph": pipeline["graph"].to_record(),
        "pose_graph": pipeline["pose_graph"].to_record(),
        "volume": pipeline["volume"].to_record(),
        "face_observations": pipeline["face_observations"].to_record(),
        "placement": pipeline["placement"].to_record(),
    }
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    assert session_record(pipeline["session"]) == snapshot["session"]
    assert pipeline["graph"].to_record() == snapshot["graph"]
    assert pipeline["pose_graph"].to_record() == snapshot["pose_graph"]
    assert pipeline["volume"].to_record() == snapshot["volume"]
    assert pipeline["face_observations"].to_record() == snapshot["face_observations"]
    assert pipeline["placement"].to_record() == snapshot["placement"]


# ---------------------------------------------------------------------------
# Hostile tamper
# ---------------------------------------------------------------------------

def test_tampered_session_pixels_are_refused(pipeline):
    """A forged frame past the constructor cannot smuggle different pixels."""
    session = pipeline["session"]
    frame = session.frames[0]
    hostile_image = np.array(frame.image, copy=True)
    hostile_image[0, 0, 0] = (int(hostile_image[0, 0, 0]) + 7) % 256
    hostile_frame = _clone(frame, image=_frozen_uint8(hostile_image))
    assert hostile_frame.image_sha256 == frame.image_sha256
    hostile_session = _clone(session, frames=(hostile_frame,) + session.frames[1:])
    _assert_refusal(_compose(pipeline, session=hostile_session), "digest_mismatch")


def test_resealed_session_record_is_refused(pipeline):
    """A session whose own stored digest no longer replays is a typed refusal."""
    session = pipeline["session"]
    forged = _clone(session, session_sha256="f" * 64)
    _assert_refusal(_compose(pipeline, session=forged), "invalid_record")


def test_tampered_graph_payload_is_refused(pipeline):
    graph = pipeline["graph"]
    hostile = _clone(graph, detail="smuggled")
    _assert_refusal(_compose(pipeline, graph=hostile), "digest_mismatch")


def test_tampered_pose_graph_payload_is_refused(pipeline):
    pose_graph = pipeline["pose_graph"]
    hostile = _clone(pose_graph, detail="smuggled")
    _assert_refusal(_compose(pipeline, pose_graph=hostile), "digest_mismatch")


def test_tampered_volume_payload_is_refused(pipeline):
    volume = pipeline["volume"]
    hostile = _clone(volume, complete=True)
    _assert_refusal(_compose(pipeline, volume=hostile), "digest_mismatch")


def test_tampered_face_observation_result_is_refused(pipeline):
    result = pipeline["face_observations"]
    forged = _clone(result, result_sha256="b" * 64)
    _assert_refusal(_compose(pipeline, face_observations=forged), "digest_mismatch")


def test_resealed_face_observation_result_is_refused(pipeline, recipe):
    """A fresh, self-consistent M5 result over a resealed volume must not ride along.

    The forged result recomputes its own digest from a volume whose placement
    binding was changed, so only the cross-record replay catches it.
    """
    result = pipeline["face_observations"]
    forged_volume = _clone(recipe["volume"], placement_sha256="c" * 64)
    hostile = _clone(result, volume_sha256=forged_volume.volume_sha256)
    _assert_refusal(_compose(pipeline, volume=forged_volume,
                             face_observations=hostile), "digest_mismatch")


def test_tampered_frame_context_is_refused(pipeline):
    frames = pipeline["frames"]
    hostile_frame = _clone(frames[0], context_sha256="d" * 64)
    _assert_refusal(_compose(pipeline, frames=(hostile_frame,) + frames[1:]),
                    "digest_mismatch")


def test_tampered_budget_digest_is_refused(pipeline):
    budget = pipeline["promotion_budget"]
    hostile = _clone(budget, max_cells=budget.max_cells + 1)
    _assert_refusal(_compose(pipeline, promotion_budget=hostile),
                    "promotion_refused")


def test_forged_placement_budget_digest_is_refused(pipeline):
    budget = pipeline["placement_budget"]
    hostile = _clone(budget, max_faces=budget.max_faces + 1)
    _assert_refusal(_compose(pipeline, placement_budget=hostile), "digest_mismatch")


def test_forged_volume_and_face_budget_digests_are_refused(pipeline):
    volume_budget = pipeline["volume_budget"]
    face_budget = pipeline["face_budget"]
    _assert_refusal(
        _compose(pipeline, volume_budget=_clone(
            volume_budget, max_support_cells=volume_budget.max_support_cells + 1)),
        "digest_mismatch")
    _assert_refusal(
        _compose(pipeline, face_budget=_clone(
            face_budget, max_claims=face_budget.max_claims + 1)),
        "digest_mismatch")


# ---------------------------------------------------------------------------
# Nested graph / pose-graph replay before the enclosing reseal is trusted
# ---------------------------------------------------------------------------

def test_resealed_graph_with_constructor_bypassed_evidence_is_refused(pipeline):
    """A resealed enclosing digest cannot hide an evidence invariant violation."""
    graph = pipeline["graph"]
    solved = next(record for record in graph.evidence if record.status == "solved")
    hostile_record = _clone(solved, baseline_direction=(3.0, 0.0, 0.0))
    evidence = tuple(hostile_record if record is solved else record
                     for record in graph.evidence)
    hostile = _reseal_graph(graph, evidence=evidence)
    assert hostile.graph_sha256 == connections._digest(
        connections._graph_payload(hostile))  # outer digest is self-consistent
    _assert_refusal(_compose(pipeline, graph=hostile), "invalid_record")


def test_resealed_graph_with_tampered_nested_budget_is_refused(pipeline):
    """The outer graph digest does not carry the nested ``budget_sha256``."""
    graph = pipeline["graph"]
    hostile_budget = _clone(graph.budgets,
                            max_pairs=graph.budgets.max_pairs + 1)
    hostile = _reseal_graph(graph, budgets=hostile_budget)
    _assert_refusal(_compose(pipeline, graph=hostile), "digest_mismatch")


def test_resealed_graph_beyond_declared_budget_caps_is_refused(pipeline):
    """A self-consistent budget whose caps the graph itself exceeds must refuse."""
    graph = pipeline["graph"]
    hostile_budget = connections.ConnectionBudgets(max_pairs=0,
                                                  max_total_support=0)
    hostile = _reseal_graph(graph, budgets=hostile_budget)
    _assert_refusal(_compose(pipeline, graph=hostile), "invalid_record")


def test_resealed_graph_with_stale_evidence_session_binding_is_refused(
        pipeline, monkeypatch):
    """A stale nested evidence session binding must refuse even when resealed."""
    graph = pipeline["graph"]
    stale_session = _session(monkeypatch, SESSION_FRAMES, seed_offset=97)
    hostile_record = _reseal_evidence(graph.evidence[0],
                                      session_sha256=stale_session.session_sha256)
    evidence = (hostile_record,) + graph.evidence[1:]
    hostile = _reseal_graph(graph, evidence=evidence)
    _assert_refusal(_compose(pipeline, graph=hostile), "source_mismatch")


def test_resealed_pose_graph_with_constructor_bypassed_pose_is_refused(pipeline):
    """A resealed pose-graph digest cannot hide a malformed nested pose."""
    pose_graph = pipeline["pose_graph"]
    hostile_pose = _clone(
        pose_graph.poses[1],
        rotation=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    poses = pose_graph.poses[:1] + (hostile_pose,) + pose_graph.poses[2:]
    hostile = _reseal_pose_graph(pose_graph, poses=poses)
    _assert_refusal(_compose(pipeline, pose_graph=hostile), "invalid_record")


def test_resealed_pose_graph_with_tampered_nested_budget_is_refused(pipeline):
    pose_graph = pipeline["pose_graph"]
    hostile_budget = _clone(pose_graph.budgets,
                            max_edges=pose_graph.budgets.max_edges + 1)
    hostile = _reseal_pose_graph(pose_graph, budgets=hostile_budget)
    _assert_refusal(_compose(pipeline, pose_graph=hostile), "digest_mismatch")


def test_resealed_pose_graph_beyond_declared_budget_caps_is_refused(pipeline):
    """A self-consistent pose budget the source graph exceeds must refuse."""
    pose_graph = pipeline["pose_graph"]
    hostile_budget = pose_mod.PoseGraphBudgets(
        max_edges=0, max_cycle_checks=0,
        rotation_tolerance_degrees=pose_graph.budgets.rotation_tolerance_degrees,
        direction_tolerance=pose_graph.budgets.direction_tolerance)
    hostile = _reseal_pose_graph(pose_graph, budgets=hostile_budget)
    _assert_refusal(_compose(pipeline, pose_graph=hostile), "invalid_record")


def test_resealed_pose_graph_with_inconsistent_components_is_refused(pipeline):
    """Components that do not replay from the nested poses must refuse."""
    pose_graph = pipeline["pose_graph"]
    hostile = _reseal_pose_graph(
        pose_graph, components=(("R1", "R0", "R2"),))
    _assert_refusal(_compose(pipeline, pose_graph=hostile), "invalid_record")


# ---------------------------------------------------------------------------
# Swapped inputs
# ---------------------------------------------------------------------------

def test_swapped_session_is_refused(pipeline, monkeypatch):
    """A coherent session built from different pixels cannot pair with the graph."""
    hostile = _session(monkeypatch, SESSION_FRAMES, seed_offset=97)
    assert hostile.session_sha256 != pipeline["session"].session_sha256
    _assert_refusal(_compose(pipeline, session=hostile), "source_mismatch")


def test_swapped_graph_from_another_session_is_refused(pipeline, monkeypatch):
    hostile = _graph(monkeypatch, SESSION_FRAMES, seed_offset=97)
    assert hostile.graph_sha256 != pipeline["graph"].graph_sha256
    _assert_refusal(_compose(pipeline, graph=hostile), "source_mismatch")


def test_swapped_pose_graph_from_another_graph_is_refused(pipeline, monkeypatch):
    hostile_graph = _graph(monkeypatch, SESSION_FRAMES, seed_offset=97)
    hostile_pose = _pose_graph(hostile_graph)
    _assert_refusal(_compose(pipeline, pose_graph=hostile_pose), "source_mismatch")


def test_tampered_volume_binding_is_refused(pipeline):
    """A volume whose placement binding was changed without resealing is refused."""
    forged = _clone(pipeline["volume"], placement_sha256="e" * 64)
    _assert_refusal(_compose(pipeline, volume=forged), "digest_mismatch")


def test_swapped_volume_from_another_placement_is_refused(pipeline):
    """A fully resealed volume that binds a different placement must be refused.

    The forged volume replays its own digest exactly (it is rebuilt through the
    accepted constructor path), so only the cross-record placement binding can
    reject it.
    """
    original = pipeline["volume"]
    swapped = _clone(original, placement_sha256="e" * 64)

    def _reseal(record):
        payload = pv._volume_payload(record)
        candidate = _clone(record, volume_sha256=pv._sha256(payload))
        pv.PlaneVolumeResult.__post_init__(candidate)
        return candidate

    hostile = _reseal(swapped)
    assert hostile.volume_sha256 != original.volume_sha256
    _assert_refusal(_compose(pipeline, volume=hostile), "source_mismatch")


def test_swapped_frame_contexts_are_refused(pipeline):
    frames = pipeline["frames"]
    reordered = (frames[1], frames[0], frames[2])
    _assert_refusal(_compose(pipeline, frames=reordered), "source_mismatch")


def test_swapped_m5_result_is_refused(pipeline):
    result = pipeline["face_observations"]
    forged = _clone(result, volume_sha256="1" * 64)
    _assert_refusal(_compose(pipeline, face_observations=forged), "source_mismatch")


# ---------------------------------------------------------------------------
# PipelineOutcome carries and binds every record
# ---------------------------------------------------------------------------

def test_outcome_record_binds_every_carried_record(pipeline):
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    record = outcome.to_record()
    assert set(record["record_bindings"]) == {
        "session", "graph", "pose_graph", "volume", "face_observations",
        "proposals", "promotion", "delivery"}
    assert record["record_bindings"]["delivery"] is None
    assert record["record_bindings"]["proposals"] == outcome.proposal_result_digest


def test_outcome_build_rejects_missing_carried_records(pipeline):
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    fields = _outcome_fields(outcome)
    for missing in ("session", "graph", "pose_graph", "volume",
                    "face_observations", "proposals", "promotion"):
        with pytest.raises(ValueError):
            m6.PipelineOutcome.build(**{**fields, missing: None})


def test_outcome_build_rejects_swapped_records_with_valid_scalar_hashes(
        pipeline, monkeypatch):
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    fields = _outcome_fields(outcome)
    other_session = _session(monkeypatch, SESSION_FRAMES, seed_offset=97)
    with pytest.raises(ValueError):
        m6.PipelineOutcome.build(**{**fields, "session": other_session})
    with pytest.raises(ValueError):
        m6.PipelineOutcome.build(**{
            **fields, "promotion": _synthetic_promotion(),
            "promotion_sha256": outcome.promotion_sha256})


def test_outcome_digest_binds_carried_record_payload_beyond_scalar(pipeline):
    """A constructor-bypassed record that keeps its declared digest still fails."""
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    hostile_volume = _clone(outcome.volume, complete=True)
    assert hostile_volume.volume_sha256 == outcome.volume_sha256
    forged = _clone(outcome, volume=hostile_volume)
    with pytest.raises(ValueError):
        m6.PipelineOutcome.__post_init__(forged)


# ---------------------------------------------------------------------------
# Deep nested yaw-hypothesis replay
# ---------------------------------------------------------------------------

def _hypothesis(**overrides):
    fields = dict(
        yaw_class=0, support=4, spatial_bins=4,
        median_selected_residual_degrees=0.5, parallax_degrees=2.0,
        cheirality_fraction=1.0, singular_ratio=0.01,
        passed=True, refusal="")
    fields.update(overrides)
    return connections.YawClassDiagnostic(**fields)


def _graph_with_hypotheses(graph, hypotheses, *, index=0):
    """Inject nested diagnostics into one evidence record and reseal outward.

    Both the evidence digest and the enclosing graph digest are recomputed around
    the injected hypotheses, so the enclosing record is self-consistent and only
    the deep nested replay can refuse a malformed diagnostic.
    """
    record = graph.evidence[index]
    forged = _reseal_evidence(record, hypotheses=tuple(hypotheses))
    evidence = tuple(forged if item is record else item for item in graph.evidence)
    return _reseal_graph(graph, evidence=evidence)


def test_wellformed_nested_hypothesis_replays_and_composes(pipeline):
    """A genuine nested diagnostic survives the deep reconstruction (positive control)."""
    graph = _graph_with_hypotheses(pipeline["graph"], (_hypothesis(),))
    pose_graph = _pose_graph(graph)
    outcome = _compose(pipeline, graph=graph, pose_graph=pose_graph,
                       allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome), outcome
    assert outcome.graph.evidence[0].hypotheses[0].support == 4


@pytest.mark.parametrize("mutation", [
    {"support": -1},
    {"support": 4.5},
    {"spatial_bins": -2},
    {"yaw_class": 4},
    {"yaw_class": True},
    {"passed": "yes"},
    {"refusal": 7},
    {"median_selected_residual_degrees": "nan"},
    {"parallax_degrees": True},
    {"cheirality_fraction": "high"},
])
def test_resealed_graph_with_forged_nested_hypothesis_is_refused(pipeline, mutation):
    """A malformed nested diagnostic refuses even with both digests resealed."""
    forged = _clone(_hypothesis(), **mutation)
    graph = _graph_with_hypotheses(pipeline["graph"], (forged,))
    # The outer digest is self-consistent around the forged diagnostic; only the
    # deep hypothesis reconstruction can refuse it.
    assert graph.graph_sha256 == connections._digest(
        connections._graph_payload(graph))
    _assert_refusal(_compose(pipeline, graph=graph, allow_omitted_unknown=True),
                    "invalid_record")


# ---------------------------------------------------------------------------
# PipelineOutcome summary/provenance cross-binding
# ---------------------------------------------------------------------------

_SUMMARY_MISMATCHES = (
    ("exact_state_eligible", True, "exact-state flag disagrees"),
    ("export_ready", False, "export flag disagrees"),
    ("omitted_unknown", 0, "omission count disagrees"),
    ("scale_state", "metric", "scale/normalization state must stay unresolved"),
    ("normalization_state", "metric",
     "scale/normalization state must stay unresolved"),
    ("unresolved", (), "unresolved notes disagree"),
    ("unresolved", ("invented: scale",), "unresolved notes disagree"),
)


@pytest.mark.parametrize("field,bad,match", _SUMMARY_MISMATCHES)
def test_outcome_refuses_a_summary_that_disagrees_with_its_records(
        pipeline, field, bad, match):
    """Every summary/provenance field is cross-bound to the record that owns it."""
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    fields = _outcome_fields(outcome)
    assert getattr(outcome, field) != bad
    with pytest.raises(ValueError, match=match):
        m6.PipelineOutcome.build(**{**fields, field: bad})


@pytest.mark.parametrize("field,bad,match", _SUMMARY_MISMATCHES)
def test_resealed_summary_mismatch_still_refuses(pipeline, field, bad, match):
    """A mismatched summary refuses even when ``outcome_sha256`` is resealed."""
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    forged = _clone(outcome, **{field: bad})
    resealed = _clone(forged, outcome_sha256=m6._sha256(m6._outcome_payload(forged)))
    with pytest.raises(ValueError, match=match):
        m6.PipelineOutcome.__post_init__(resealed)


def _delivered_outcome(pipeline, *, promotion_sha256=None, block_count=None):
    """A coherent delivered outcome built around the opted-in recipe promotion."""
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    assert outcome.promotion.cells
    delivered = delivery.DeliveryResult(
        promotion_sha256=(outcome.promotion.promotion_sha256
                          if promotion_sha256 is None else promotion_sha256),
        artifact_sha256=_h("delivered-artifact"),
        block_count=(len(outcome.promotion.cells)
                     if block_count is None else block_count),
        bytes_written=4096,
        output_path=Path("delivered.litematic"))
    return m6.PipelineOutcome.build(
        **{**_outcome_fields(outcome), "delivered": True, "delivery": delivered,
           "delivery_artifact_sha256": delivered.artifact_sha256})


def test_delivery_cross_bindings_accept_a_coherent_delivery(pipeline):
    outcome = _delivered_outcome(pipeline)
    assert outcome.delivered is True
    assert outcome.delivery.promotion_sha256 == outcome.promotion.promotion_sha256
    assert outcome.delivery.block_count == len(outcome.promotion.cells)


def test_outcome_refuses_a_delivery_that_binds_another_promotion(pipeline):
    with pytest.raises(ValueError, match="delivery does not bind its carried promotion"):
        _delivered_outcome(pipeline, promotion_sha256=_h("another-promotion"))


def test_outcome_refuses_a_delivery_with_a_forged_block_count(pipeline):
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    with pytest.raises(ValueError, match="delivery block count disagrees"):
        _delivered_outcome(pipeline,
                           block_count=len(outcome.promotion.cells) + 1)


# ---------------------------------------------------------------------------
# Manual metadata rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", m6.M6_MANUAL_METADATA_KEYS)
def test_manual_metadata_is_forbidden(pipeline, key):
    outcome = _compose(pipeline, manual_metadata={key: ("manual", "value")})
    refusal = _assert_refusal(outcome, "manual_metadata_forbidden")
    # The detail is stored as a deep immutable structure: sequences are tuples.
    assert list(refusal.detail["keys"]) == [key]


def test_empty_manual_metadata_is_allowed(pipeline):
    outcome = _compose(pipeline, manual_metadata={})
    assert isinstance(outcome, m6.PipelineRefusal)  # refused for the review, not metadata
    assert outcome.reason == "promotion_refused"


def test_manual_metadata_precedes_every_record(pipeline):
    """The metadata guard fires before a deliberately broken record is inspected."""
    outcome = _compose(pipeline, session=None,
                       manual_metadata={"coordinates": [[0, 0, 0]]})
    _assert_refusal(outcome, "manual_metadata_forbidden")


def test_module_declares_forbidden_keys_and_no_manual_input():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert set(m6.M6_MANUAL_METADATA_KEYS) <= literals
    # the declared forbidden concepts never appear as a consumed parameter name
    signature = inspect.signature(m6.compose_image_derived_pipeline)
    assert set(signature.parameters) & set(m6.M6_MANUAL_METADATA_KEYS) == set()
    assert set(signature.parameters) == {
        "session", "graph", "pose_graph", "volume", "universe", "supports",
        "placement", "placement_budget", "volume_budget", "frames",
        "face_observations", "face_budget", "palette", "promotion_budget",
        "max_cost", "min_margin", "max_observations", "max_candidate_evaluations",
        "max_payload_bytes", "allow_omitted_unknown", "destination", "name",
        "description", "overwrite", "exporter", "expected_promotion",
        "expected_delivery", "manual_metadata",
    }


# ---------------------------------------------------------------------------
# Review / abstention and eligibility gates
# ---------------------------------------------------------------------------

def test_proposal_abstention_is_a_typed_review_refusal(pipeline):
    outcome = _compose(pipeline, max_cost=0.0, min_margin=0.0)
    refusal = _assert_refusal(outcome, "promotion_refused")
    assert refusal.detail["kind"] == "review"


def test_result_level_review_never_composes_an_outcome(pipeline):
    """A result-level review is refused by the promotion seam, never exported."""
    outcome = _compose(pipeline, palette=PALETTE[:1], min_margin=1e9)
    _assert_refusal(outcome, "promotion_refused")


def test_ineligible_promotion_is_never_delivered(pipeline, tmp_path):
    destination = tmp_path / "ineligible.litematic"
    outcome = _compose(pipeline, allow_omitted_unknown=True,
                       destination=destination, name="facade",
                       description="facade delivery")
    refusal = _assert_refusal(outcome, "promotion_refused")
    assert refusal.detail["reason"] == "not_exact_state_eligible"
    assert refusal.detail["kind"] == "review"
    assert not destination.exists()


def test_delivery_seam_refuses_the_ineligible_promotion_it_is_given(pipeline, tmp_path):
    """The accepted delivery seam owns the same eligibility gate independently."""
    composed = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(composed, m6.PipelineOutcome)
    refused = delivery.deliver_image_derived_litematic(
        composed.promotion, tmp_path / "seam.litematic", name="facade",
        description="facade delivery")
    assert isinstance(refused, delivery.DeliveryRefusal)
    assert refused.reason == "promotion_not_exact_state_eligible"


def test_unresolved_scale_semantics_are_preserved(pipeline):
    outcome = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(outcome, m6.PipelineOutcome)
    promotion = outcome.promotion
    assert promotion.scale_state == "unresolved"
    assert promotion.normalization_state == "unresolved"
    assert promotion.export_bounds is None
    assert promotion.gauge_axes == (0, 1, 2)
    pose_graph = pipeline["pose_graph"]
    assert pose_graph.metric_scale_known is False
    assert pipeline["session"].scale_known is False
    # a tampered metric-scale claim cannot be laundered through the facade
    hostile_pose = _clone(pose_graph, metric_scale_known=True)
    _assert_refusal(_compose(pipeline, pose_graph=hostile_pose), "digest_mismatch")


# ---------------------------------------------------------------------------
# Delivery refusal
# ---------------------------------------------------------------------------

def test_delivery_refusal_from_the_seam_is_typed(pipeline, tmp_path):
    """An accepted promotion that the seam refuses is a typed, atomic refusal.

    The recipe composes an ineligible promotion, so the facade's own gate (and
    the seam's independent gate) refuse it while a pre-existing destination is
    left byte-for-byte intact.
    """
    destination = tmp_path / "taken.litematic"
    destination.write_bytes(b"pre-existing")
    outcome = _compose(pipeline, allow_omitted_unknown=True,
                       destination=destination, name="facade",
                       description="facade delivery")
    refusal = _assert_refusal(outcome, "promotion_refused")
    assert refusal.detail["reason"] == "not_exact_state_eligible"
    assert destination.read_bytes() == b"pre-existing"
    assert list(tmp_path.iterdir()) == [destination]


def test_invalid_delivery_metadata_is_refused_before_promotion(pipeline, tmp_path):
    _assert_refusal(
        _compose(pipeline, destination=tmp_path / "x.litematic", name="",
                 description="facade delivery", overwrite=False),
        "invalid_input")
    _assert_refusal(
        _compose(pipeline, destination=tmp_path / "x.litematic", name="facade",
                 description="facade delivery", overwrite="yes"),
        "invalid_input")
    _assert_refusal(
        _compose(pipeline, destination=tmp_path / "x.litematic", name="facade\t",
                 description="facade delivery", overwrite=False),
        "invalid_input")


def test_expected_delivery_without_a_destination_is_refused(pipeline):
    outcome = _compose(pipeline, allow_omitted_unknown=True,
                       expected_delivery="not-a-delivery")
    _assert_refusal(outcome, "invalid_input")


def test_delivery_is_not_attempted_for_an_ineligible_promotion(
        pipeline, tmp_path, monkeypatch):
    """The accepted delivery seam must never be called for an ineligible promotion."""
    calls: list[tuple] = []

    def spy(promotion, destination, **kwargs):
        calls.append((promotion, destination, kwargs))
        return delivery.DeliveryRefusal.build(
            reason="invalid_promotion", detail={"field": "spy"})

    monkeypatch.setattr(delivery, "deliver_image_derived_litematic", spy)
    destination = tmp_path / "spy.litematic"
    outcome = _compose(pipeline, allow_omitted_unknown=True,
                       destination=destination, name="facade",
                       description="facade delivery")
    _assert_refusal(outcome, "promotion_refused")
    assert calls == []
    assert not destination.exists()


def test_entry_point_requires_every_accepted_record():
    signature = inspect.signature(m6.compose_image_derived_pipeline)
    required = {
        "session", "graph", "pose_graph", "volume", "universe", "supports",
        "placement", "placement_budget", "volume_budget", "frames",
        "face_observations", "face_budget", "palette", "promotion_budget",
        "max_cost", "min_margin", "max_observations", "max_candidate_evaluations",
        "max_payload_bytes",
    }
    parameters = signature.parameters
    assert required <= set(parameters)
    assert all(parameters[name].default is inspect.Parameter.empty
               for name in required)
    for name in ("session", "graph", "pose_graph", "volume", "face_observations"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert set(parameters) <= required | {
        "allow_omitted_unknown", "destination", "name", "description",
        "overwrite", "exporter", "expected_promotion", "expected_delivery",
        "manual_metadata",
    }


def test_module_boundary_has_no_cli_ui_store_or_export_surface():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in (".cli", ".ui", ".desktop", ".project", ".dataset"):
        assert not any(name.endswith(forbidden) for name in imported), imported
    source = SOURCE.read_text(encoding="utf-8")
    assert "argparse" not in source
    assert "input(" not in source
    assert "subprocess" not in source


def test_refusal_record_is_frozen_and_digest_bound():
    refusal = m6.PipelineRefusal.build(reason="digest_mismatch",
                                       detail={"field": "session"})
    for field in dataclasses.fields(refusal):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(refusal, field.name, None)  # type: ignore[misc]
    assert refusal.refusal_sha256 == m6._sha256({
        "producer": refusal.producer, "reason": refusal.reason,
        "detail": refusal.detail})
    forged = _clone(refusal, reason="invalid_record")
    with pytest.raises(ValueError):
        m6.PipelineRefusal.__post_init__(forged)


def test_refusal_rejects_embedded_partial_records():
    refusal = m6.PipelineRefusal.build(reason="digest_mismatch", detail={"field": "x"})
    forged = _clone(refusal, detail={"cells": [1, 2, 3]})
    with pytest.raises(ValueError):
        m6.PipelineRefusal.__post_init__(forged)


def test_refusal_detail_is_deeply_immutable_and_replayable():
    detail = {"field": "graph", "nested": {"keys": ["a", "b"],
                                           "depth": [{"x": 1}]}}
    refusal = m6.PipelineRefusal.build(reason="digest_mismatch", detail=detail)
    # Nested mappings are read-only proxies and nested sequences are tuples.
    assert refusal.detail["field"] == "graph"
    assert refusal.detail["nested"]["keys"] == ("a", "b")
    assert refusal.detail["nested"]["depth"][0] == {"x": 1}
    with pytest.raises(TypeError):
        refusal.detail["field"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        refusal.detail["nested"]["keys"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        refusal.detail["nested"]["depth"][0]["x"] = 2  # type: ignore[index]
    with pytest.raises(AttributeError):
        refusal.detail["nested"]["keys"].append("c")
    # to_record is a deep plain copy and the digest replays from it.
    record = refusal.to_record()
    assert record["detail"] == detail
    rebuilt = m6.PipelineRefusal.build(reason=record["reason"],
                                       detail=record["detail"])
    assert rebuilt.refusal_sha256 == refusal.refusal_sha256
    assert rebuilt.to_record() == record
    assert record["refusal_sha256"] == m6._sha256({
        "producer": refusal.producer, "reason": refusal.reason,
        "detail": refusal.detail})


def test_promotion_replay_uses_promotion_module_not_delivery_private_helper():
    source = SOURCE.read_text(encoding="utf-8")
    assert "delivery_mod._record_fields" not in source
    assert "promotion_mod._record_fields" in source


def test_constructor_bypassed_promotion_expectation_is_refused(pipeline):
    """A promotion resealed past its constructor is refused through the promotion module."""
    composed = _compose(pipeline, allow_omitted_unknown=True)
    assert isinstance(composed, m6.PipelineOutcome)
    hostile_cells = tuple(_clone(cell, block_sha256="0" * 64)
                          for cell in composed.promotion.cells)
    hostile = _reseal_promotion(_clone(composed.promotion, cells=hostile_cells))
    _assert_refusal(
        _compose(pipeline, allow_omitted_unknown=True,
                 expected_promotion=hostile),
        "invalid_record")
