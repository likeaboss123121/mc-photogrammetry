"""Independent controls for the isolated M5 plane-face observation bridge.

The genuine end-to-end input is the accepted mixed-normal P0 recipe from
``tests/test_plane_placement.py``: its solver-produced nine-face
``ResolvedFacePlacement``, its original ``PlacementUniverse``, its nine accepted
V1 supports and its accepted V3 ``PlaneVolumeResult`` are reused here through
those public/test APIs only.  Every expected cell, slot, count and bound below is
declared from the recipe's own literal construction (``_literal_recipe_normal`` /
``_literal_recipe_tangents`` / ``_literal_recipe_cells`` and the declared
supported/unknown grids), never read back from the bridge under test.

The recipe retains no screenshot: ``test_plane_placement._signature`` builds its
frame buffer from ``np.random.default_rng(992)`` and never returns it, and the
supports carry only its shape plus the opaque literal ``IMAGE_SHA``.  The test
fixture therefore reconstructs **exactly that frame buffer** and **exactly the
recipe's own usable mask** (``_mask_for``), and a dedicated control proves the
reconstruction by reproducing the recipe's retained
``PlaneSignature.interior``/``coverage`` from it.  No camera, screenshot,
occupancy model or occlusion is invented anywhere.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import test_plane_placement as recipe_tests  # noqa: E402
from test_plane_placement import (  # noqa: E402
    IMAGE_SHA,
    RECIPE_AXES,
    RECIPE_B,
    RECIPE_FRAMES,
    _literal_recipe_cells,
    _literal_recipe_normal,
    _literal_recipe_tangents,
)

from mc_photogrammetry.blocks.appearance import DEFAULT_PALETTE  # noqa: E402
from mc_photogrammetry.reconstruction import appearance_scoring  # noqa: E402
from mc_photogrammetry.reconstruction import face_sampling  # noqa: E402
from mc_photogrammetry.reconstruction import lattice_planes as lp  # noqa: E402
from mc_photogrammetry.reconstruction import plane_face_observation as m5  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402
from mc_photogrammetry.reconstruction import plane_region_support as v1  # noqa: E402
from mc_photogrammetry.reconstruction import plane_volume as pv  # noqa: E402
from mc_photogrammetry.reconstruction.face_observation import (  # noqa: E402
    NORMAL_OF_SLOT,
)

SOURCE = Path(m5.__file__)
PATCH_PIXELS = int(face_sampling.PATCH_SIZE) ** 2
#: Retained bytes of one emitted observation: float32 (16, 16, 3) patch + bool
#: valid mask + float64 coverage mask (``FaceObservation`` retains coverage as
#: float64).
OBSERVATION_BYTES = PATCH_PIXELS * 3 * 4 + PATCH_PIXELS + PATCH_PIXELS * 8

# Literal, independent slot table declared from the canonical face normals: the
# outward world normal of a claimed face is ``outward_sign * e_{axis}``.
LITERAL_SLOT_OF_NORMAL = {
    (0, 1): "east", (0, -1): "west",
    (1, 1): "up", (1, -1): "down",
    (2, 1): "south", (2, -1): "north",
}
RECIPE_WORK = {
    "frames": 3,
    "claims": 66,
    "observations": 66,
    "reviews": 297,
    "rectified_pixels": 66 * PATCH_PIXELS,
    "payload_bytes": 66 * OBSERVATION_BYTES,
    "replay_checks": 21,
}
RECIPE_UNKNOWN_CELLS = 15
RECIPE_UNSUPPORTED_SLOTS = 58 * 6 - 66


def _literal_observations():
    """The declared ``(cell, slot, frame)`` identity of every recipe claim."""
    expected = []
    for frame in RECIPE_FRAMES:
        for axis in RECIPE_AXES:
            normal = _literal_recipe_normal(frame, axis)
            slot = LITERAL_SLOT_OF_NORMAL[(normal["world_axis"], normal["outward_sign"])]
            tangents = _literal_recipe_tangents(frame, axis)
            for row, column, lateral_u, lateral_v in _literal_recipe_cells(frame, axis):
                coordinate = [0, 0, 0]
                coordinate[normal["world_axis"]] = normal["normal_block"]
                coordinate[tangents[0][2]] = lateral_u
                coordinate[tangents[1][2]] = lateral_v
                expected.append(((tuple(coordinate), slot, frame),
                                 (frame, axis, row, column)))
    return tuple(expected)


def _fixture_image() -> np.ndarray:
    """The genuine recipe frame buffer, rebuilt from its declared construction.

    ``test_plane_placement._signature`` creates this exact array
    (``default_rng(992)``, 1024x1024x3 float32, ``+ 0.1``) for every reading and
    never retains it; ``test_fixture_image_reproduces_the_recipe_signature``
    proves the reconstruction against the recipe's own retained signature.
    """
    rng = np.random.default_rng(992)
    return rng.random((1024, 1024, 3), dtype=np.float32) + 0.1


def _clone(record, **changes):
    """Copy a frozen record without re-running its validating constructor."""
    clone = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(clone, field.name, getattr(record, field.name))
    for name, value in changes.items():
        object.__setattr__(clone, name, value)
    return clone


def _source_kwargs(context):
    return dict(
        universe=context["universe"], supports=context["supports"],
        placement=context["placement"], placement_budget=context["placement_budget"],
        volume_budget=context["volume_budget"])


def _bridge_kwargs(context):
    kwargs = _source_kwargs(context)
    kwargs.update(frames=context["frames"], budget=context["budget"])
    return kwargs


def _budget(**overrides):
    caps = dict(
        max_claims=66, max_frames=3, max_rectified_pixels=66 * PATCH_PIXELS,
        max_payload_bytes=66 * OBSERVATION_BYTES, max_replay_checks=21)
    caps.update(overrides)
    return m5.make_face_observation_budget(**caps)


def _volume_budget(**overrides):
    """The accepted V3 recipe volume caps (declared from its literal construction)."""
    caps = dict(
        max_faces=9, max_support_cells=81, max_contributions=66,
        max_coordinates=58, max_conflicts=1, max_omitted_unknown=15,
        max_dense_bounds_cells=224)
    caps.update(overrides)
    return pv.make_volume_budget(**caps)


@pytest.fixture(scope="module")
def recipe_context():
    fixture = recipe_tests._build_recipe_fixture()
    universe = recipe_tests._recipe_universe(fixture)
    supports = recipe_tests._recipe_supports(fixture)
    placement_budget = recipe_tests._budget()
    placement = plane_placement.resolve_face_placement(
        universe=universe, supports=supports, budget=placement_budget)
    assert isinstance(placement, plane_placement.ResolvedFacePlacement), placement
    volume_budget = _volume_budget()
    result = pv.build_plane_volume(
        universe=universe, supports=supports, placement=placement,
        placement_budget=placement_budget, budget=volume_budget)
    assert isinstance(result, pv.PlaneVolumeResult), result
    image = _fixture_image()
    frames = []
    for frame in RECIPE_FRAMES:
        readings = tuple(reading for reading in universe.readings
                         if reading.frame == frame)
        masks = {reading.axis: recipe_tests._mask_for(reading.axis)
                 for reading in readings}
        assert len({v1.usable_mask_sha256(mask) for mask in masks.values()}) == 1
        frames.append(m5.frame_context_from_readings(
            frame_id=frame, image=image, image_sha256=IMAGE_SHA,
            usable_mask=masks[readings[0].axis], readings=readings))
    return {
        "fixture": fixture, "universe": universe, "supports": supports,
        "placement": placement, "placement_budget": placement_budget,
        "volume_budget": volume_budget, "result": result, "image": image,
        "frames": tuple(frames), "budget": _budget(),
    }


@pytest.fixture(scope="module")
def bridge_result(recipe_context):
    built = m5.build_plane_face_observations(
        recipe_context["result"], **_bridge_kwargs(recipe_context))
    assert isinstance(built, m5.PlaneFaceObservationResult), built
    return built


def _support_by_digest(context):
    return {support.support_sha256: support for support in context["supports"]}


def _reading(context, frame_id, axis):
    return next(reading for reading in context["universe"].readings
                if reading.frame == frame_id and reading.axis == axis)


# ---------------------------------------------------------------------------
# Fixture provenance: the image and mask come from the original recipe only
# ---------------------------------------------------------------------------

def test_fixture_image_reproduces_the_recipe_signature(recipe_context):
    """The rebuilt frame buffer is exactly the recipe's own signature input."""
    image = recipe_context["image"]
    uncovered_seen = False
    for reading in recipe_context["universe"].readings[::3]:
        signature = next(
            item for item in recipe_context["fixture"]["signatures"]
            if item.reading.frame == reading.frame and item.reading.axis == reading.axis)
        measured = lp.plane_signature(
            image, reading, RECIPE_B,
            usable=recipe_tests._mask_for(reading.axis), max_cells=32)
        assert measured.cells.shape == signature.cells.shape
        assert measured.coverage.shape == signature.coverage.shape
        assert np.array_equal(measured.interior, signature.interior)
        # ``test_plane_placement._signature`` overwrites only the covered cells
        # with world texture, so the uncovered cells must still be identical.
        valid = np.asarray(measured.coverage, dtype=bool)
        if not valid.all():
            uncovered_seen = True
            assert np.array_equal(signature.cells[~valid], measured.cells[~valid])
    assert uncovered_seen


def test_frame_contexts_are_derived_from_original_readings_and_immutable(recipe_context):
    frames = {context.frame_id: context for context in recipe_context["frames"]}
    assert sorted(frames) == list(RECIPE_FRAMES)
    universe = recipe_context["universe"]
    for frame_id, context in frames.items():
        assert context.image.flags.writeable is False
        assert context.usable_mask.flags.writeable is False
        with pytest.raises(ValueError):
            context.image[0, 0, 0] = 1.0
        assert context.image_shape == (1024, 1024)
        assert context.image_sha256 == IMAGE_SHA
        readings = tuple(reading for reading in universe.readings
                         if reading.frame == frame_id)
        rebuilt = m5.frame_context_from_readings(
            frame_id=frame_id, image=recipe_context["image"], image_sha256=IMAGE_SHA,
            usable_mask=recipe_tests._mask_for(0), readings=readings)
        assert rebuilt == context
        assert rebuilt.context_sha256 == context.context_sha256
        for reading in readings:
            item = context.rectification_for(reading.axis)
            assert item is not None
            assert item.rectification is reading.rectification \
                or item.rectification == reading.rectification


# ---------------------------------------------------------------------------
# Exact geometry: one observation per literal claim, all six physical slots
# ---------------------------------------------------------------------------

def test_recipe_bridge_emits_one_observation_per_literal_claim(bridge_result):
    expected = {identity for identity, _ in _literal_observations()}
    observed = {(item.cell, item.slot, item.frame_id)
                for item in bridge_result.observations}
    assert observed == expected
    assert len(bridge_result.observations) == len(expected) == 66
    assert {(item.cell, item.slot, item.frame_id) for item in bridge_result.bindings} \
        == expected


def test_recipe_bridge_preserves_v3_invariants_and_provenance(recipe_context,
                                                              bridge_result):
    volume = recipe_context["result"]
    assert bridge_result.volume_sha256 == volume.volume_sha256
    assert bridge_result.placement_sha256 == volume.placement_sha256
    assert len(volume.cells) == 58
    assert sum(len(cell.claims) for cell in volume.cells) == 66
    assert len(volume.omitted) == 15
    assert len(volume.conflicts) == 0
    assert len({item.cell for item in bridge_result.observations}) == 58
    assert {name: getattr(bridge_result.work, name) for name in RECIPE_WORK} \
        == RECIPE_WORK
    unknown = [review for review in bridge_result.reviews
               if review.reason == "unknown_cell"]
    assert len(unknown) == RECIPE_UNKNOWN_CELLS
    assert {tuple(sorted(review.frame_ids)) for review in unknown} == {("R1",)}
    assert all(review.coordinate is None and review.scope == "cell"
               for review in unknown)
    supports = _support_by_digest(recipe_context)
    faces = {face.face_sha256: face for face in recipe_context["placement"].faces}
    for binding in bridge_result.bindings:
        face = faces[binding.face_sha256]
        support = supports[binding.support_sha256]
        assert binding.frame_id == face.frame
        assert binding.reading_axis == face.reading_axis
        assert binding.reading_identity_sha256 == face.reading_identity_sha256
        assert support.reading_identity.identity_sha256 \
            == binding.reading_identity_sha256
        assert support.support_sha256 == binding.support_sha256


def test_recipe_bridge_support_frames_are_only_same_cell_and_slot_claims(bridge_result):
    grouped = {}
    for binding in bridge_result.bindings:
        grouped.setdefault((binding.cell, binding.slot), set()).add(binding.frame_id)
    for observation in bridge_result.observations:
        frames = grouped[(observation.cell, observation.slot)]
        assert observation.support_frames == tuple(sorted(frames))
        assert observation.frame_id in observation.support_frames
    assert len(grouped) == 66
    assert {frame for frames in grouped.values() for frame in frames} \
        == set(RECIPE_FRAMES)
    assert all(len(frames) == 1 for frames in grouped.values())


def test_recipe_bridge_keeps_signed_reflected_and_negative_geometry(bridge_result):
    bindings = bridge_result.bindings
    expected_reflected = sum(
        len(_literal_recipe_cells(frame, axis))
        for frame in RECIPE_FRAMES for axis in RECIPE_AXES
        if _literal_recipe_normal(frame, axis)["world_sign"] < 0)
    expected_negative = sum(
        1 for identity, _ in _literal_observations() if min(identity[0]) < 0)
    assert {binding.slot for binding in bindings} == set(m5.FACE_SLOTS)
    assert {binding.outward_sign for binding in bindings} == {-1, 1}
    assert {binding.world_normal_axis for binding in bindings} == {0, 1, 2}
    assert sum(1 for binding in bindings if binding.world_normal_sign < 0) \
        == expected_reflected == 22
    assert sum(1 for binding in bindings if min(binding.cell) < 0) \
        == expected_negative > 0
    for binding in bindings:
        assert binding.slot == LITERAL_SLOT_OF_NORMAL[
            (binding.world_normal_axis, binding.outward_sign)]
        assert binding.slot == m5.slot_of_normal(binding.world_normal_axis,
                                                  binding.outward_sign)


def test_recipe_bridge_coverage_is_the_rectified_source_intersection(recipe_context,
                                                                     bridge_result):
    """Coverage is region & usable & in-frame, rectified by the accepted helper."""
    supports = _support_by_digest(recipe_context)
    contexts = {context.frame_id: context for context in recipe_context["frames"]}
    for binding, observation in zip(bridge_result.bindings, bridge_result.observations):
        support = supports[binding.support_sha256]
        reading = _reading(recipe_context, binding.frame_id, binding.reading_axis)
        context = contexts[binding.frame_id]
        region = lp._decode_region_evidence(
            reading.region_evidence, reading.axis, reading.polygon)
        homography = v1._exact_homography(
            binding.reading_axis, support.calibration, support.alternative)
        u_start = Fraction(*support.origin_uv[0]) + binding.column
        v_start = Fraction(*support.origin_uv[1]) + binding.row
        quad = np.asarray([
            v1._project(homography, u, v) for u, v in (
                (u_start, v_start), (u_start + 1, v_start),
                (u_start + 1, v_start + 1), (u_start, v_start + 1))],
            dtype=np.float32)
        patch, valid, coverage = face_sampling.rectify_face_valid(
            context.image, quad, region & context.usable_mask,
            size=face_sampling.PATCH_SIZE)
        assert np.array_equal(patch, observation.patch)
        assert np.array_equal(valid, observation.valid)
        assert np.array_equal(np.clip(coverage, 0.0, 1.0), observation.coverage)
        assert observation.usable_coverage == pytest.approx(
            float(np.clip(coverage, 0.0, 1.0).mean()), abs=1e-12)
        assert observation.usable_coverage > 0.9
        assert observation.is_usable
        assert observation.normal == NORMAL_OF_SLOT[observation.slot]


# ---------------------------------------------------------------------------
# Immutability and appearance-scoring compatibility
# ---------------------------------------------------------------------------

def test_recipe_bridge_observations_are_immutable_snapshots(bridge_result):
    assert isinstance(bridge_result.observations, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bridge_result.work = None
    for observation in bridge_result.observations[:8]:
        assert observation.patch.flags.writeable is False
        assert observation.valid.flags.writeable is False
        assert observation.coverage.flags.writeable is False
        with pytest.raises(ValueError):
            observation.patch[0, 0, 0] = 0.0
        with pytest.raises(ValueError):
            observation.valid[0, 0] = False
        with pytest.raises(ValueError):
            observation.coverage[0, 0] = 0.0
        assert np.isfinite(observation.patch).all()
        assert 0.0 <= float(observation.coverage.min())
        assert float(observation.coverage.max()) <= 1.0


def test_recipe_bridge_observations_score_through_appearance_scoring(bridge_result):
    scorer = appearance_scoring.AppearanceScorer(DEFAULT_PALETTE)
    for observation in bridge_result.observations[:3]:
        score = scorer.score(observation)
        assert not score.abstained
        assert len(score.costs) == len(DEFAULT_PALETTE)
        assert np.isfinite(score.best_cost)


# ---------------------------------------------------------------------------
# Digest binding, determinism and canonical ordering
# ---------------------------------------------------------------------------

def test_recipe_bridge_is_deterministic_and_canonically_ordered(recipe_context,
                                                                bridge_result):
    assert m5.canonical_face_observation_order(
        tuple(reversed(bridge_result.observations))) == bridge_result.observations
    assert m5.canonical_face_observation_order(
        tuple(reversed(bridge_result.reviews))) == bridge_result.reviews
    for binding, observation in zip(bridge_result.bindings, bridge_result.observations):
        assert binding.observation_sha256 == m5.face_observation_sha256(observation)
        fields = {field.name: getattr(binding, field.name)
                  for field in dataclasses.fields(binding)
                  if field.name != "binding_sha256"}
        rebuilt = m5.FaceObservationBinding.build(**fields)
        assert rebuilt.binding_sha256 == binding.binding_sha256
        assert rebuilt.to_record() == binding.to_record()
    again = m5.build_plane_face_observations(
        recipe_context["result"], **_bridge_kwargs(recipe_context))
    assert isinstance(again, m5.PlaneFaceObservationResult)
    assert again.result_sha256 == bridge_result.result_sha256
    assert again.to_record() == bridge_result.to_record()
    shuffled = dict(recipe_context)
    shuffled["frames"] = tuple(reversed(recipe_context["frames"]))
    shuffled["supports"] = tuple(reversed(recipe_context["supports"]))
    permuted = m5.build_plane_face_observations(
        recipe_context["result"], **_bridge_kwargs(shuffled))
    assert isinstance(permuted, m5.PlaneFaceObservationResult)
    assert permuted.result_sha256 == bridge_result.result_sha256
    assert permuted.work.to_record() == bridge_result.work.to_record()


def test_recipe_bridge_review_records_are_typed_and_replay(bridge_result):
    reasons = {review.reason for review in bridge_result.reviews}
    assert reasons <= set(m5.M5_FACE_OBSERVATION_REVIEW_REASONS)
    for review in bridge_result.reviews:
        assert review.scope == m5.M5_FACE_OBSERVATION_REVIEW_SCOPES[review.reason]
        rebuilt = m5.FaceObservationReview.build(
            reason=review.reason, scope=review.scope, coordinate=review.coordinate,
            slot=review.slot, frame_ids=review.frame_ids,
            claim_sha256s=review.claim_sha256s,
            support_sha256s=review.support_sha256s, detail=review.detail)
        assert rebuilt.review_sha256 == review.review_sha256
        assert rebuilt.to_record() == review.to_record()
    unsupported = [review for review in bridge_result.reviews
                   if review.reason == "unsupported_slot"]
    assert len(unsupported) == RECIPE_UNSUPPORTED_SLOTS
    assert all(review.scope == "slot" and review.slot is not None
               for review in unsupported)
    assert not [review for review in bridge_result.reviews
                if review.reason in ("conflict_cell", "degenerate_projection",
                                     "no_source_pixels", "off_frame",
                                     "empty_coverage")]


# ---------------------------------------------------------------------------
# Typed refusals: binding, replay and slot validation before any budget
# ---------------------------------------------------------------------------

def _refusal(built):
    assert isinstance(built, m5.PlaneFaceObservationRefusal), built
    return built


def test_slot_mismatch_refusal_is_removed_and_slot_geometry_stays_total():
    """The declared-but-unreachable ``slot_mismatch`` reason is gone.

    The bridge derives every slot from the accepted V3 claim's outward geometry
    (``world_normal_axis``/``outward_sign``), which the source-bound V3 replay has
    already validated, so no reachable bridge request can produce a slot/normal
    disagreement.  ``slot_of_normal`` remains total and injective over the six
    canonical normals, and a hand-built binding that contradicts its own normal is
    still rejected by the record constructor rather than by a phantom refusal.
    """
    assert "slot_mismatch" not in m5.M5_FACE_OBSERVATION_REFUSAL_REASONS
    table = {}
    for slot, basis in m5.FACE_BASIS.items():
        normal = basis[0]
        axis = next(index for index, value in enumerate(normal) if float(value) != 0.0)
        sign = 1 if float(normal[axis]) > 0.0 else -1
        table[(axis, sign)] = slot
    assert table == LITERAL_SLOT_OF_NORMAL
    assert set(table) == {(axis, sign) for axis in (0, 1, 2) for sign in (-1, 1)}
    for (axis, sign), slot in table.items():
        assert m5.slot_of_normal(axis, sign) == slot

    fields = dict(
        cell=(0, 0, 0), slot="east", frame_id="R0", reading_axis=0,
        world_normal_axis=0, world_normal_sign=1, outward_sign=1, row=0, column=0,
        claim_sha256="0" * 64, reading_identity_sha256="0" * 64,
        support_sha256="0" * 64, face_sha256="0" * 64,
        frame_context_sha256="0" * 64, rectification_sha256="0" * 64,
        image_pixels_sha256="0" * 64, usable_mask_sha256="0" * 64,
        observation_sha256="0" * 64)
    assert m5.FaceObservationBinding.build(**fields).slot == "east"
    # ``(world_normal_axis, outward_sign) == (0, 1)`` is canonically east, so a
    # hand-built binding claiming west must fail its own constructor.
    with pytest.raises(ValueError):
        m5.FaceObservationBinding.build(**{**fields, "slot": "west"})


def test_missing_frame_context_is_refused(recipe_context):
    frames = tuple(context for context in recipe_context["frames"]
                   if context.frame_id != "R1")
    built = m5.build_plane_face_observations(
        recipe_context["result"], **{**_bridge_kwargs(recipe_context), "frames": frames})
    refusal = _refusal(built)
    assert refusal.reason == "missing_frame_context"
    assert refusal.work is None
    assert "observations" not in refusal.to_record()


def test_unreferenced_frame_is_refused(recipe_context):
    readings = tuple(reading for reading in recipe_context["universe"].readings
                     if reading.frame == "R0")
    rectifications = tuple(
        m5.ReadingRectification.build(reading_axis=int(reading.axis),
                                      rectification=reading.rectification)
        for reading in readings)
    extra = m5.FrameRectificationContext.build(
        frame_id="R9", image=recipe_context["image"], image_sha256=IMAGE_SHA,
        usable_mask=recipe_tests._mask_for(0), rectifications=rectifications)
    built = m5.build_plane_face_observations(
        recipe_context["result"],
        **{**_bridge_kwargs(recipe_context),
           "frames": recipe_context["frames"] + (extra,)})
    assert _refusal(built).reason == "unreferenced_frame"


def test_missing_rectification_is_refused(recipe_context):
    original = recipe_context["frames"][0]
    trimmed = m5.FrameRectificationContext.build(
        frame_id=original.frame_id, image=original.image,
        image_sha256=original.image_sha256, usable_mask=original.usable_mask,
        rectifications=original.rectifications[:2])
    frames = (trimmed,) + tuple(context for context in recipe_context["frames"]
                                if context.frame_id != original.frame_id)
    built = m5.build_plane_face_observations(
        recipe_context["result"],
        **{**_bridge_kwargs(recipe_context), "frames": frames})
    assert _refusal(built).reason == "missing_rectification"


def test_rectification_and_mask_mismatches_are_refused(recipe_context):
    original = recipe_context["frames"][0]
    reading = _reading(recipe_context, original.frame_id,
                       original.rectifications[0].reading_axis)
    bad_rectification = _clone(
        reading.rectification, focal=float(reading.rectification.focal) + 1.0)
    tampered = m5.FrameRectificationContext.build(
        frame_id=original.frame_id, image=original.image,
        image_sha256=original.image_sha256,
        usable_mask=recipe_tests._mask_for(0),
        rectifications=tuple(
            m5.ReadingRectification.build(reading_axis=item.reading_axis,
                                          rectification=bad_rectification)
            if item.reading_axis == reading.axis else item
            for item in original.rectifications))
    frames = (tampered,) + tuple(context for context in recipe_context["frames"]
                                 if context.frame_id != original.frame_id)
    built = m5.build_plane_face_observations(
        recipe_context["result"],
        **{**_bridge_kwargs(recipe_context), "frames": frames})
    assert _refusal(built).reason == "frame_context_mismatch"

    other_mask = np.zeros_like(original.usable_mask)
    other_mask[0:16, 0:16] = True
    mismatch_mask = m5.FrameRectificationContext.build(
        frame_id=original.frame_id, image=original.image,
        image_sha256=original.image_sha256, usable_mask=other_mask,
        rectifications=original.rectifications)
    frames = (mismatch_mask,) + tuple(context for context in recipe_context["frames"]
                                      if context.frame_id != original.frame_id)
    built = m5.build_plane_face_observations(
        recipe_context["result"],
        **{**_bridge_kwargs(recipe_context), "frames": frames})
    assert _refusal(built).reason == "frame_context_mismatch"

    wrong_digest = m5.FrameRectificationContext.build(
        frame_id=original.frame_id, image=original.image,
        image_sha256=hashlib.sha256(b"not-the-recipe-image").hexdigest(),
        usable_mask=original.usable_mask, rectifications=original.rectifications)
    frames = (wrong_digest,) + tuple(context for context in recipe_context["frames"]
                                     if context.frame_id != original.frame_id)
    built = m5.build_plane_face_observations(
        recipe_context["result"],
        **{**_bridge_kwargs(recipe_context), "frames": frames})
    assert _refusal(built).reason == "frame_context_mismatch"


def test_forged_context_with_a_stale_digest_is_refused(recipe_context):
    original = recipe_context["frames"][0]
    forged_image = np.array(original.image, copy=True)
    forged_image[0, 0, 0] = float(forged_image[0, 0, 0]) + 0.25
    forged = _clone(original, image=forged_image)
    assert forged.pixels_sha256 == original.pixels_sha256
    frames = (forged,) + tuple(context for context in recipe_context["frames"]
                               if context.frame_id != original.frame_id)
    built = m5.build_plane_face_observations(
        recipe_context["result"],
        **{**_bridge_kwargs(recipe_context), "frames": frames})
    assert _refusal(built).reason == "frame_context_mismatch"


def test_non_volume_and_modified_results_are_refused_before_budget(recipe_context):
    tiny = _budget(max_claims=1, max_frames=1, max_rectified_pixels=1,
                   max_payload_bytes=1, max_replay_checks=1)
    built = m5.build_plane_face_observations(
        {"not": "a volume"}, **{**_bridge_kwargs(recipe_context), "budget": tiny})
    assert _refusal(built).reason == "non_volume_input"

    stale = _clone(recipe_context["result"], volume_sha256="0" * 64)
    built = m5.build_plane_face_observations(
        stale, **{**_bridge_kwargs(recipe_context), "budget": tiny})
    assert _refusal(built).reason == "replay_mismatch"


def test_tampered_universe_usable_masks_are_refused(recipe_context):
    """The original PlacementUniverse usable masks are load-bearing."""
    tampered = np.zeros((1024, 1024), dtype=bool)
    tampered[0:240, 0:240] = True
    universe = _clone(
        recipe_context["universe"],
        usable_masks=tuple((key, tampered)
                           for key, _ in recipe_context["universe"].usable_masks))
    tiny = _budget(max_claims=1, max_frames=1, max_rectified_pixels=1,
                   max_payload_bytes=1, max_replay_checks=1)
    refused = _refusal(m5.build_plane_face_observations(
        recipe_context["result"],
        **{**_bridge_kwargs(recipe_context), "universe": universe, "budget": tiny}))
    assert refused.reason == "volume_replay_refused"
    assert refused.work is None


def test_budget_caps_are_exact_atomic_and_after_validation(recipe_context):
    exact_result = m5.build_plane_face_observations(
        recipe_context["result"], **_bridge_kwargs(recipe_context))
    assert isinstance(exact_result, m5.PlaneFaceObservationResult)
    cases = (
        ("max_claims", 66), ("max_frames", 3),
        ("max_rectified_pixels", 66 * PATCH_PIXELS),
        ("max_payload_bytes", 66 * OBSERVATION_BYTES),
        ("max_replay_checks", 21),
    )
    for cap, exact in cases:
        under = _budget(**{cap: exact - 1})
        refused = _refusal(m5.build_plane_face_observations(
            recipe_context["result"],
            **{**_bridge_kwargs(recipe_context), "budget": under}))
        assert refused.reason == "budget_exhausted"
        assert set(refused.detail) == {"cap", "limit", "observed"}
        assert refused.detail["cap"] == cap
        assert refused.detail["limit"] == exact - 1
        assert refused.detail["observed"] == exact
        assert refused.work is not None
        assert refused.work.to_record()[m5._CAP_TO_WORK[cap]] == exact
        record = refused.to_record()
        assert "observations" not in record and "cells" not in record


def test_budget_errors_are_typed_and_self_validating():
    with pytest.raises(ValueError):
        m5.make_face_observation_budget()
    with pytest.raises(ValueError):
        m5.make_face_observation_budget(
            max_claims=1, max_frames=1, max_rectified_pixels=1,
            max_payload_bytes=1, max_replay_checks=0)
    budget = _budget()
    assert m5.PlaneFaceObservationBudget(**budget.to_record()) == budget


# ---------------------------------------------------------------------------
# Module boundary
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = {
    "__future__", "hashlib", "dataclasses", "fractions", "typing", "numpy",
    "mc_photogrammetry.blocks", "mc_photogrammetry.blocks.model_atlas",
    "mc_photogrammetry.reconstruction",
    "mc_photogrammetry.reconstruction.face_sampling",
    "mc_photogrammetry.reconstruction.lattice_planes",
    "mc_photogrammetry.reconstruction.plane_placement",
    "mc_photogrammetry.reconstruction.plane_region_support",
    "mc_photogrammetry.reconstruction.plane_volume",
    "mc_photogrammetry.reconstruction.face_observation",
}
_FORBIDDEN_NAMES = {
    "BlockVolume", "BlockState", "ObservationKind", "sample_visible_faces",
    "occupancy", "LitematicExporter", "project_store", "manual_metadata",
}


def test_module_imports_and_names_stay_inside_the_boundary():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    assert imports <= _ALLOWED_IMPORTS, sorted(imports - _ALLOWED_IMPORTS)

    referenced = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                calls.add(target.id)
            elif isinstance(target, ast.Attribute):
                calls.add(target.attr)
    assert not (referenced & _FORBIDDEN_NAMES), sorted(referenced & _FORBIDDEN_NAMES)
    assert not (calls & {"open", "exec", "eval", "compile", "__import__"}), sorted(calls)


def test_module_does_not_reference_export_cli_or_ui_paths():
    text = SOURCE.read_text(encoding="utf-8")
    for forbidden in ("mc_photogrammetry.export", "mc_photogrammetry.cli",
                      "mc_photogrammetry.ui", "mc_photogrammetry.evaluation_context_export"):
        assert forbidden not in text, forbidden


def test_result_record_is_frozen_and_digest_bound(bridge_result):
    with pytest.raises(dataclasses.FrozenInstanceError):
        bridge_result.result_sha256 = "0" * 64
    rebuilt = m5.PlaneFaceObservationResult.build(
        schema_version=bridge_result.schema_version,
        producer_version=bridge_result.producer_version,
        coverage_policy=bridge_result.coverage_policy,
        volume_sha256=bridge_result.volume_sha256,
        placement_sha256=bridge_result.placement_sha256,
        frame_bindings=bridge_result.frame_bindings,
        observations=bridge_result.observations,
        bindings=bridge_result.bindings,
        reviews=bridge_result.reviews,
        work=bridge_result.work)
    assert rebuilt.result_sha256 == bridge_result.result_sha256
    assert rebuilt.to_record() == bridge_result.to_record()
    for binding in bridge_result.bindings:
        assert binding.binding_sha256 == m5._sha256(m5._binding_payload(binding))
