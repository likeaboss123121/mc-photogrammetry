"""Synthetic exhaustive producer, admission ordering and logical envelope tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from functools import lru_cache

import numpy as np
import pytest

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.blocks.candidate_universe import (
    AppearanceSupport,
    CandidateUniverse,
    InventoryCompleteness,
    LegalStateManifest,
    record_sha256,
)
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.pipeline.image_derived_output_policy_v2 import ImageDerivedOutputPolicyV2
from mc_photogrammetry.reconstruction import appearance_scoring_v2 as scoring
from mc_photogrammetry.reconstruction import appearance_state_proposals_v2 as subject
from mc_photogrammetry.reconstruction.face_observation import NORMAL_OF_SLOT, FaceObservation


@lru_cache(maxsize=8)
def universe(count=3, descriptors=0, component=0.):
    states = tuple(BlockState(f"minecraft:state{index:05}") for index in range(count))
    support = tuple(AppearanceSupport(
        state.name, "supported", appearance=PaletteColor(
            state, (80, 90, 100), descriptors=((component,) * 7,) * descriptors),
        model_record_sha256="3" * 64, texture_record_sha256="4" * 64) for state in states)
    return CandidateUniverse(
        "synthetic", "1" * 64, LegalStateManifest(
            "synthetic", "2" * 64, "synthetic", "synthetic", "1", ("synthetic",),
            InventoryCompleteness("exhaustive")), "1", "1", (), states, support)


def policy(count=3, descriptors=0, component=0., **caps):
    return ImageDerivedOutputPolicyV2(
        universe(count, descriptors, component), scoring.scorer_identity(),
        subject.proposal_identity(), **caps)


def observation(index=0, *, cell=(0, 0, 0), slot="up", shape=(2, 1), rgb=(80, 90, 100),
                weight=1., usable=1., coverage=None, valid=None, provenance=("synthetic",)):
    frame = f"frame{index:04}"
    patch = np.broadcast_to(np.array(rgb, dtype=np.float32), (*shape, 3))
    return FaceObservation(cell, slot, frame, NORMAL_OF_SLOT[slot], patch, usable,
                           (frame,), valid, coverage, provenance, weight)


def forge(record, **changes):
    clone = object.__new__(type(record))
    for item in fields(record):
        object.__setattr__(clone, item.name, changes.get(item.name, getattr(record, item.name)))
    return clone


def frozen_array(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def forbidden(*args, **kwargs):
    raise AssertionError("forbidden later preflight stage executed")


def assert_early(monkeypatch, p, observations, reason, *, replay=True):
    monkeypatch.setattr(subject, "_hash_array", forbidden)
    monkeypatch.setattr(subject, "AppearanceScorerV2", forbidden)
    # A fitting minimal refusal may be serialized; request serialization cannot occur.
    original = subject._serialize
    monkeypatch.setattr(subject, "_serialize", lambda record: (
        original(record) if record.get("status") == "refused" else forbidden()))
    if replay:
        monkeypatch.setattr(subject, "_policy_preflight", forbidden)
    result = subject.produce_diagnostics(p, observations)
    assert result.status == "refused" and result.reason == reason
    assert result.observations == () and result.cells == () and result.state_ids == ()
    assert result.executed_work == subject.Work()
    return result


def test_complete_full_indexed_diagnostics_multiple_cells_order_and_weighted_reference():
    p = policy()
    observations = (observation(2, cell=(1, 0, 0)),
                    observation(1, rgb=(10, 20, 30), weight=3., usable=.5),
                    observation(0, weight=.25))
    result = subject.produce_diagnostics(p, observations)
    assert result.status == "complete", result.reason
    assert result.state_ids == p.universe.state_ids
    assert [row.metadata.frame_id for row in result.observations] == [
        "frame0000", "frame0001", "frame0002"]
    assert [row.cell for row in result.cells] == [(0, 0, 0), (1, 0, 0)]
    for index in range(3):
        expected = (.25 * result.observations[0].scores.absolute[index]
                    + 1.5 * result.observations[1].scores.absolute[index]) / 1.75
        assert result.cells[0].scores.absolute[index] == expected
    assert result.cells[0].scores.ties == ((0, 1, 2),)
    assert subject.verify_payload(result.payload) == result.to_record()
    record = result.to_record()
    assert len(result.payload) <= result.reserved_payload_bytes <= p.max_proposal_payload_bytes
    assert result.executed_work.candidate_evaluations == 9
    assert result.reserved_work.aggregate_components == 6
    assert not {"state", "winner", "accepted", "proposed", "confidence", "threshold"} & set(record)
    assert record["state_ids"] == list(p.universe.state_ids)
    assert all(len(value) == 16 for row in record["observations"] for value in row["absolute"])


def test_permutation_canonical_output_and_complete_metadata_array_digest_sensitivity():
    p, a, b = policy(), observation(0), observation(1)
    first = subject.produce_diagnostics(p, (b, a))
    assert first.payload == subject.produce_diagnostics(p, (a, b)).payload
    changes = (observation(0, rgb=(81, 90, 100)), observation(0, shape=(2, 2)),
               observation(0, provenance=("changed",)), observation(0, weight=.5),
               observation(0, usable=.5), observation(0, valid=np.ones((2, 1), dtype=bool)),
               observation(0, coverage=np.ones((2, 1), dtype=np.float64)),
               forge(a, support_frames=("extra", "frame0000")),
               forge(a, patch=frozen_array(a.patch.astype(">f4"))))
    for changed in changes:
        result = subject.produce_diagnostics(p, (changed, b))
        assert result.status == "complete", result.reason
        assert result.input_sha256 != first.input_sha256
        assert result.payload_sha256 != first.payload_sha256


@pytest.mark.parametrize("mode", (
    "zero_weight", "zero_usable", "zero_mass", "epsilon_mass", "invalid_mask"))
def test_no_evidence_never_fabricates_scores_and_work_is_explicit(mode):
    valid = np.zeros((2, 1), dtype=bool) if mode == "invalid_mask" else None
    coverage = np.full((2, 1), 5e-7 if mode == "epsilon_mass" else 0., dtype=np.float64)
    a = observation(weight=0 if mode == "zero_weight" else 1,
                    usable=0 if mode == "zero_usable" else 1, valid=valid,
                    coverage=coverage if "mass" in mode else None)
    result = subject.produce_diagnostics(policy(), (a,))
    assert result.status == "complete", result.reason
    assert result.observations[0].status == "no_evidence"
    assert result.cells[0].status == "no_evidence"
    assert result.observations[0].scores is None and result.cells[0].scores is None
    assert result.reserved_work.candidate_evaluations == 3
    assert result.executed_work.candidate_evaluations == 0
    subject.verify_payload(result.payload)


def test_no_observations_has_complete_table_without_cells_or_scores():
    result = subject.produce_diagnostics(policy(), ())
    assert result.status == "complete", result.reason
    assert result.state_ids == universe().state_ids
    assert result.observations == () and result.cells == ()
    subject.verify_payload(result.payload)


@pytest.mark.parametrize("completeness", ("unestablished", "certified_under_named_model"))
def test_incomplete_inventory_atomic_refusal_before_scorer(monkeypatch, completeness):
    u = universe()
    complete = InventoryCompleteness(
        completeness, "synthetic" if "certified" in completeness else None)
    p = replace(policy(), universe=replace(u, manifest=replace(u.manifest, completeness=complete)))
    assert_early(monkeypatch, p, (observation(),), "incomplete_universe", replay=False)


@pytest.mark.parametrize("disposition", ("unsupported", "unknown"))
def test_incomplete_support_refusal_omits_exclusion_inventory(monkeypatch, disposition):
    u = universe()
    entries = (AppearanceSupport(u.state_ids[0], disposition, ("large-exclusion-" + "x" * 1000,)),
               *u.appearance_support[1:])
    p = replace(policy(), universe=replace(u, appearance_support=entries))
    result = assert_early(monkeypatch, p, (observation(),), "incomplete_universe", replay=False)
    assert b"large-exclusion" not in result.payload


@pytest.mark.parametrize("kind,field,value", [
    ("scorer", "scorer_id", "other"), ("scorer", "version", "other"),
    ("scorer", "named_model", "other"), ("scorer", "definition_sha256", "9" * 64),
    ("scorer", "population_semantics", "other"), ("scorer", "normalization_semantics", "other"),
    ("scorer", "clipping_semantics", "other"), ("proposal", "proposal_id", "other"),
    ("proposal", "version", "other"), ("proposal", "definition_sha256", "9" * 64),
    ("proposal", "observation_schema", "other"), ("proposal", "observation_version", "other"),
    ("proposal", "tie_semantics", "other"),
])
def test_every_builtin_identity_field_rejects_arbitrary_policy(monkeypatch, kind, field, value):
    p = policy()
    p = replace(p, **{kind: replace(getattr(p, kind), **{field: value})})
    assert_early(monkeypatch, p, (observation(),), "builtin_identity", replay=False)


def test_forged_definition_digest_and_policy_or_universe_tamper_refuse(monkeypatch):
    p = policy()
    changed = scoring.scorer_definition()
    changed["absolute"] = "forged"
    p = replace(p, scorer=replace(p.scorer, definition_sha256=record_sha256(changed)))
    assert_early(monkeypatch, p, (observation(),), "builtin_identity", replay=False)


@pytest.mark.parametrize("location", ("policy", "universe", "support"))
def test_stale_nested_digest_replay_refuses_atomically(monkeypatch, location):
    p = policy()
    if location == "policy":
        p = forge(p, policy_sha256="0" * 64)
    elif location == "universe":
        p = forge(p, universe=forge(p.universe, universe_sha256="0" * 64))
    else:
        entries = (forge(p.universe.appearance_support[0], appearance_sha256="0" * 64),
                   *p.universe.appearance_support[1:])
        p = forge(p, universe=forge(p.universe, appearance_support=entries))
    assert_early(monkeypatch, p, (observation(),), "invalid_contract", replay=False)


def test_duplicate_identity_refuses_even_when_pixels_differ(monkeypatch):
    assert_early(monkeypatch, policy(), (observation(), observation(rgb=(0, 0, 0))),
                 "duplicate_observation", replay=False)


@pytest.mark.parametrize("refs", (("z", "frame0000"), ("frame0000", "frame0000")))
def test_unique_canonical_reference_order_required(monkeypatch, refs):
    obs = forge(observation(), support_frames=refs)
    assert_early(monkeypatch, policy(), (obs,), "reference_order", replay=False)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1., 256.))
def test_nonfinite_or_out_of_domain_pixels_refuse_before_hashing(monkeypatch, value):
    obs = observation()
    patch = obs.patch.copy()
    patch[0, 0, 0] = value
    obs = forge(obs, patch=frozen_array(patch))
    assert_early(monkeypatch, policy(), (observation(1), obs), "array_values", replay=False)


def test_nonfinite_scoring_after_one_row_is_atomic(monkeypatch):
    original = subject.AppearanceScorerV2.score
    calls = []

    def score(self, features, slot):
        calls.append(slot)
        if len(calls) == 2:
            raise ValueError("nonfinite cost")
        return original(self, features, slot)

    monkeypatch.setattr(subject.AppearanceScorerV2, "score", score)
    result = subject.produce_diagnostics(policy(), (observation(), observation(1)))
    assert len(calls) == 2
    assert result.status == "refused" and not result.observations and not result.cells
    # A late refusal keeps truthful typed counters while staying atomic and
    # serializing only the frozen five-key refusal record.
    assert result.reason == "invalid_contract"
    assert set(result.to_record()) == {"schema", "version", "status", "reason", "completeness"}
    assert result.to_record()["completeness"] == "refused"
    # Row 0 committed a complete hash and a complete candidate vector; row 1
    # hashed before its scoring unit aborted. Only completed units are counted,
    # so hashing advances for both rows while observation/scoring work reports one.
    assert result.executed_work != subject.Work()
    assert result.executed_work.hash_bytes == 48
    assert result.executed_work.observations == 1
    assert result.executed_work.patch_pixels == 2
    assert result.executed_work.retained_array_bytes == 24
    assert result.executed_work.metadata_bytes > 0
    assert result.executed_work.candidates == 3
    assert result.executed_work.candidate_evaluations == 3
    assert result.executed_work.candidate_pixel_work == 6
    assert result.executed_work.descriptor_components == 0
    assert result.executed_work.aggregate_components == 0
    assert result.reserved_work.candidate_evaluations == 6
    assert result.reserved_work.retained_array_bytes == 48


def test_built_in_work_semantics_are_explicit_and_immutable():
    # The built-in proposal definition distinguishes the four completed unit
    # classes and the counter behaviour on refusal.
    text = subject.proposal_definition()["work"]
    for term in ("Reserved", "observation unit", "hash unit", "scoring unit",
                 "aggregate unit", "pre-observation refusals", "refusal payload"):
        assert term in text
    assert (
        "Executed observation unit: only after feature extraction and a complete "
        "validated scored/no-evidence row, add one observation with its patch "
        "pixels, retained and metadata bytes."
    ) in text
    assert (
        "Executed hash unit: add only the exact array bytes already streamed "
        "for a completed observation hash."
    ) in text
    identity = subject.proposal_identity()
    definition = subject.proposal_definition()
    definition["work"] = "forged"
    assert subject.proposal_identity() == identity


def test_minimum_refusal_too_large_returns_typed_result_without_serialization(monkeypatch):
    p = policy(max_proposal_payload_bytes=1)
    monkeypatch.setattr(subject, "_serialize", forbidden)
    result = subject.produce_diagnostics(p, (observation(),))
    assert result.reason == "payload_bytes"
    assert result.payload is None and result.to_record() is None
    assert result.payload_sha256 is None


@pytest.mark.parametrize("cap,maximum", (("max_observations", 1000),
                                       ("max_candidate_evaluations", 100_000),
                                       ("max_proposal_payload_bytes", 10_000_000)))
def test_policy_cap_wrong_type_or_one_over_refuses_before_replay(monkeypatch, cap, maximum):
    assert_early(monkeypatch, forge(policy(), **{cap: maximum + 1}), (), "policy_cap")


@pytest.mark.parametrize("kind", ("list", "iterator", "policy_dict"))
def test_top_level_exact_types_refuse_without_iteration(monkeypatch, kind):
    p = policy()
    observations = [] if kind == "list" else iter((observation(),)) if kind == "iterator" else ()
    if kind == "policy_dict":
        p = {}
    assert_early(monkeypatch, p, observations, "top_level_type")


def test_count_products_exact_and_one_over_before_replay(monkeypatch):
    p = policy(1, max_observations=2, max_candidate_evaluations=2)
    assert subject.produce_diagnostics(p, (observation(), observation(1))).status == "complete"
    assert_early(monkeypatch, p, (observation(),) * 3, "observations")


def test_evaluation_policy_limit_exact_one_over(monkeypatch):
    p = policy(3, max_candidate_evaluations=6)
    assert subject.produce_diagnostics(p, (observation(), observation(1))).status == "complete"
    assert_early(monkeypatch, p, tuple(observation(i) for i in range(3)), "candidate_evaluations")


def test_8192_times_12_admitted_and_times_13_atomic_early_refusal(monkeypatch):
    p = policy(8192)
    observations = tuple(observation(i) for i in range(13))
    result = subject.produce_diagnostics(p, observations[:12])
    assert result.status == "complete", result.reason
    assert result.executed_work.candidate_evaluations == 98_304
    assert len(result.state_ids) == 8192
    assert all(len(row.scores.absolute) == 8192 for row in result.observations)
    assert_early(monkeypatch, p, observations, "candidate_evaluations")


def test_candidate_count_exact_and_one_over(monkeypatch):
    p = policy(16384)
    assert subject.produce_diagnostics(p, ()).status == "complete"
    u = forge(p.universe, legal_states=p.universe.legal_states + (p.universe.legal_states[0],),
              appearance_support=(p.universe.appearance_support
                                  + (p.universe.appearance_support[0],)))
    assert_early(monkeypatch, forge(p, universe=u), (), "candidates")


def test_observation_ceiling_exact_and_one_over(monkeypatch):
    p = policy(1)
    observations = tuple(observation(i, weight=0) for i in range(1000))
    assert subject.produce_diagnostics(p, observations).status == "complete"
    assert_early(monkeypatch, p, observations + (observations[0],), "observations")


@pytest.mark.parametrize("shape", ((1, 1), (257, 1), (2, 257), (2, 0)))
def test_patch_dimension_one_over_or_under_before_replay(monkeypatch, shape):
    obs = observation()
    obs = forge(obs, patch=frozen_array(np.zeros((*shape, 3), dtype=np.float32)))
    assert_early(monkeypatch, policy(1), (obs,), "patch_dimensions")


@pytest.mark.parametrize("shape", ((2, 1), (256, 1), (2, 256), (16, 256), (256, 16)))
def test_patch_dimension_and_pixel_exact_bounds(shape):
    result = subject.produce_diagnostics(policy(1), (observation(shape=shape),))
    assert result.status == "complete", result.reason


def test_patch_pixel_one_over_before_replay(monkeypatch):
    assert_early(monkeypatch, policy(1), (observation(shape=(17, 241)),), "pixels_per_patch")


@pytest.mark.parametrize("mode", ("writeable", "owner_readonly", "dtype", "noncontiguous",
                                 "large_backing", "wrong_valid", "wrong_coverage"))
def test_retained_array_contract_refuses_before_replay(monkeypatch, mode):
    obs = observation(shape=(2, 2))
    if mode == "wrong_valid":
        obs = forge(obs, valid=frozen_array(np.ones((2, 2), dtype=np.uint8)))
    elif mode == "wrong_coverage":
        obs = forge(obs, coverage=frozen_array(np.ones((2, 2), dtype=np.float32)))
    else:
        array = obs.patch.copy()
        if mode == "owner_readonly":
            array.setflags(write=False)
        elif mode == "dtype":
            array = frozen_array(array.astype(np.float64))
        elif mode == "noncontiguous":
            array = obs.patch[:, ::-1]
        elif mode == "large_backing":
            array = np.frombuffer(bytes(1000), dtype=np.float32, count=12).reshape(2, 2, 3)
        obs = forge(obs, patch=array)
    result = assert_early(monkeypatch, policy(1), (obs,),
                          "array_header" if mode in ("writeable", "dtype", "noncontiguous",
                                                     "wrong_valid", "wrong_coverage") else
                          "array_backing" if mode == "owner_readonly" else "array_backing_size")
    assert result.status == "refused"


def test_retained_per_observation_exact_total_bytes_exact_alias_count_and_one_over(monkeypatch):
    # 4096*21=86016 is dominated by the patch ceiling; exceeding it is already refused there.
    obs = observation(shape=(16, 256), valid=np.ones((16, 256), dtype=bool),
                      coverage=np.ones((16, 256), dtype=np.float64), weight=0)
    result = subject.produce_diagnostics(policy(1), (obs,))
    assert result.status == "complete", result.reason
    assert result.reserved_work.retained_array_bytes == 86_016
    # 250 admitted occurrences of 4000 pixels with all arrays: exactly both total ceilings.
    base = observation(shape=(16, 250), valid=np.ones((16, 250), dtype=bool),
                       coverage=np.ones((16, 250), dtype=np.float64), weight=0)
    observations = tuple(forge(base, frame_id=f"frame{i:04}", support_frames=(f"frame{i:04}",))
                         for i in range(250))
    result = subject.produce_diagnostics(policy(1), observations)
    assert result.status == "complete", result.reason
    assert result.reserved_work.patch_pixels == 1_000_000
    assert result.reserved_work.retained_array_bytes == 21_000_000
    assert_early(monkeypatch, policy(1), observations + (observation(300),), "patch_pixels")


def test_retained_bytes_cap_is_dominated_by_total_pixels_and_per_occurrence_accounted():
    assert subject.MAX_TOTAL_PIXELS * 21 == subject.MAX_TOTAL_ARRAY_BYTES
    assert subject.MAX_PATCH_PIXELS * 21 == subject.MAX_OBSERVATION_ARRAY_BYTES


@pytest.mark.parametrize("maximum", (
    16_384, 1000, 100_000, 4096, 1_000_000, 86_016, 21_000_000, 25_600_000,
    4_200_000, 6, 7, 1024, 32, 16_384, 1_000_000, 8_000_000, 10_000_000))
def test_logical_integer_guard_exact_and_one_over(maximum):
    # Exercise even dominated/granular bounds without altering production caps.
    subject._limit(maximum, maximum, "guard")
    with pytest.raises(subject._Refusal, match="guard"):
        subject._limit(maximum + 1, maximum, "guard")


def test_candidate_pixel_work_exact_and_one_over(monkeypatch):
    p = policy(100)
    observations = tuple(observation(i, shape=(16, 250), weight=0) for i in range(64))
    result = subject.produce_diagnostics(p, observations)
    assert result.status == "complete", result.reason
    assert result.reserved_work.candidate_pixel_work == 25_600_000
    assert_early(monkeypatch, p, observations + (observation(100, weight=0),),
                 "candidate_pixel_work")


def test_descriptor_work_exact_and_one_over(monkeypatch):
    p = policy(1000, descriptors=6)
    observations = tuple(observation(i, weight=0) for i in range(100))
    result = subject.produce_diagnostics(p, observations)
    assert result.status == "complete", result.reason
    assert result.reserved_work.descriptor_components == 4_200_000
    # At the profile maximum descriptor work is exactly 42 * candidate_evaluations,
    # so the 100000 evaluation ceiling dominates it: 999 candidates can admit only
    # 100 rows, and the would-be 101st row is refused by the earlier evaluation
    # guard before the descriptor counter is ever consulted.
    p = policy(999, descriptors=6)
    assert_early(monkeypatch, p, observations + (observation(100),), "candidate_evaluations")


def test_descriptor_work_cap_dominance_invariant(monkeypatch):
    # Each candidate contributes at most descriptors_per_candidate * descriptor_width
    # = 6 * 7 = 42 scalar components, so for every admitted request
    #   descriptor_work = observations * sum(components_c)
    #                   <= observations * 42 * candidates
    #                   == 42 * candidate_evaluations
    #                   <= 42 * MAX_EVALUATIONS == MAX_DESCRIPTOR_WORK.
    # The 4,200,000 descriptor ceiling is exactly reachable and can never be
    # exceeded: any request that would exceed it must already exceed 100000
    # candidate evaluations and is refused there first.
    profile = subject.producer_resource_profile()
    assert profile["descriptors_per_candidate"] == 6
    assert profile["descriptor_width"] == 7
    assert 6 * 7 * subject.MAX_EVALUATIONS == subject.MAX_DESCRIPTOR_WORK
    for candidates in (1, 3, 999, 1000, 8192, subject.MAX_CANDIDATES):
        admitted = subject.MAX_EVALUATIONS // candidates
        assert admitted * candidates <= subject.MAX_EVALUATIONS
        assert admitted * candidates * 42 <= subject.MAX_DESCRIPTOR_WORK
    # The attempted descriptor overflow needs an evaluation count above its cap,
    # so the candidate_evaluations refusal is the only reachable outcome.
    assert 999 * 101 > subject.MAX_EVALUATIONS
    assert 999 * 101 * 42 > subject.MAX_DESCRIPTOR_WORK
    assert_early(monkeypatch, policy(999, descriptors=6),
                 tuple(observation(i, weight=0) for i in range(101)),
                 "candidate_evaluations")


@pytest.mark.parametrize("descriptors,reason", (
    (((0.,) * 7,) * 7, "descriptor_count"), (((0.,) * 6,), "descriptor_shape"),
    (((1e6 + 1,) * 7,), "numeric_range"), (((float("inf"),) * 7,), "invalid_contract"),
))
def test_descriptor_incompatibilities_refuse_without_omission(monkeypatch, descriptors, reason):
    p = policy()
    first = p.universe.appearance_support[0]
    appearance = forge(first.appearance, descriptors=descriptors)
    support = (forge(first, appearance=appearance), *p.universe.appearance_support[1:])
    p = forge(p, universe=forge(p.universe, appearance_support=support))
    assert_early(monkeypatch, p, (observation(),), reason)


def test_descriptor_magnitude_and_count_exact_ceiling():
    result = subject.produce_diagnostics(policy(1, 6, 1e6), (observation(),))
    assert result.status == "complete", result.reason
    assert result.observations[0].scores.absolute[0] > 1e6


@pytest.mark.parametrize("mapping,reason", (
    ((("bad", (0, 0, 0)),), "face_map_slot"),
    ((("up", (0, 0, 0)),) * 2, "face_map_duplicate"),
    ((("up", (0, 0, 0)),) * 7, "face_map_count"),
    ((("up", (256, 0, 0)),), "face_map_rgb"),
))
def test_face_map_incompatibilities_refuse(monkeypatch, mapping, reason):
    p = policy()
    first = p.universe.appearance_support[0]
    support = (forge(first, appearance=forge(first.appearance, face_map=mapping)),
               *p.universe.appearance_support[1:])
    assert_early(monkeypatch, forge(p, universe=forge(p.universe, appearance_support=support)),
                 (observation(),), reason)


def test_six_unique_canonical_face_rgb_slots_admitted():
    p = policy(1)
    first = p.universe.appearance_support[0]
    appearance = replace(first.appearance, face_map=tuple((slot, (80, 90, 100))
                         for slot in sorted(NORMAL_OF_SLOT)))
    u = replace(p.universe, appearance_support=(replace(first, appearance=appearance),))
    result = subject.produce_diagnostics(replace(p, universe=u), (observation(),))
    assert result.status == "complete", result.reason


@pytest.mark.parametrize("weight,reason", ((1e6 + 1, "numeric_range"),
                                          (-1., "numeric_range"), (True, "numeric_type")))
def test_observation_weight_one_over_and_type_refuse(monkeypatch, weight, reason):
    assert_early(monkeypatch, policy(), (forge(observation(), weight=weight),), reason)


def test_observation_weight_exact_ceiling_and_zero():
    assert subject.produce_diagnostics(policy(), (observation(weight=1e6),)).status == "complete"
    assert subject.produce_diagnostics(policy(), (observation(weight=0),)).status == "complete"


def test_string_utf8_exact_and_one_over(monkeypatch):
    obs = observation(provenance=("é" * 512,))
    assert subject.produce_diagnostics(policy(), (obs,)).status == "complete"
    assert_early(monkeypatch, policy(), (observation(provenance=("é" * 512 + "a",)),),
                 "observation_string")


def test_reference_count_exact_and_one_over(monkeypatch):
    refs = tuple(f"ref{i:02}" for i in range(32))
    result = subject.produce_diagnostics(policy(), (observation(provenance=refs),))
    assert result.status == "complete"
    assert_early(monkeypatch, policy(), (observation(provenance=refs + ("ref32",)),),
                 "reference_count")


def test_state_id_utf8_exact_and_one_over(monkeypatch):
    p = policy(1)
    state = BlockState("minecraft:" + "a" * (1024 - len("minecraft:")))
    entry = p.universe.appearance_support[0]
    entry = replace(entry, state_id=state.name, appearance=replace(entry.appearance, state=state))
    u = replace(p.universe, legal_states=(state,), appearance_support=(entry,))
    assert subject.produce_diagnostics(replace(p, universe=u), ()).status == "complete"
    state = BlockState(state.name + "a")
    u = forge(u, legal_states=(state,))
    assert_early(monkeypatch, forge(p, universe=u), (), "observation_string")


def metadata_of_size(target):
    base = observation(provenance=())
    size = subject._measure(subject.asdict(subject._metadata(base)))
    # ASCII strings permit exact canonical-byte construction, including list separators.
    remainder = target - size
    refs = []
    while remainder > 1026:
        token = f"{len(refs):02}:" + "x" * 1021
        refs.append(token)
        remainder -= 1026 + (1 if len(refs) > 1 else 0)
    length = remainder - 2 - (1 if refs else 0)
    refs.append("z" * length)
    obs = forge(base, provenance=tuple(refs))
    assert subject._measure(subject.asdict(subject._metadata(obs)), 32_000_000) == target
    return obs


def test_observation_metadata_exact_and_one_over(monkeypatch):
    obs = metadata_of_size(16_384)
    assert subject.produce_diagnostics(policy(1), (obs,)).status == "complete"
    obs = metadata_of_size(16_385)
    assert_early(monkeypatch, policy(1), (obs,), "observation_metadata_bytes")


def test_total_metadata_exact_and_one_over(monkeypatch):
    # Keep frame string lengths fixed so each metadata occurrence has equal size.
    base = metadata_of_size(10_000)
    observations = tuple(forge(base, frame_id=f"frame{i:04}", support_frames=(f"frame{i:04}",))
                         for i in range(100))
    result = subject.produce_diagnostics(policy(1), observations)
    assert result.status == "complete", result.reason
    assert result.reserved_work.metadata_bytes == 1_000_000
    assert_early(monkeypatch, policy(1), observations + (observation(100),), "metadata_bytes")


def test_output_reservation_exact_and_one_under_before_hashing(monkeypatch):
    observations = (observation(), observation(1))
    result = subject.produce_diagnostics(policy(), observations)
    reserved = result.reserved_payload_bytes
    result = subject.produce_diagnostics(
        policy(max_proposal_payload_bytes=reserved), observations)
    assert result.status == "complete"
    assert_early(monkeypatch, policy(max_proposal_payload_bytes=reserved - 1), observations,
                 "payload_bytes", replay=False)


def test_cell_payload_exact_and_one_over_pure_arithmetic_guard():
    assert subject._cell_reservation(7_999_998, ()) == 8_000_000
    with pytest.raises(subject._Refusal, match="cell_payload_bytes"):
        subject._cell_reservation(7_999_999, ())
    # Upstream evaluation/metadata caps dominate this branch for an admitted cell:
    # <=100000 observation candidates + <=16384 aggregate candidates, each two
    # 19-byte costs and a <=8-byte singleton tie; 1MB metadata + bounded row headers.
    upper = 46 * (100_000 + 16_384) + 1_000_000 + 1000 * 1000
    assert upper < subject.MAX_CELL_PAYLOAD_BYTES


def test_whole_output_limit_many_cells_before_hashing(monkeypatch):
    p = policy(1000)
    # Shared IDs at their exact maximum plus 1MB admitted metadata, and two
    # complete candidate vectors per cell, exceed the final 10MB envelope.
    states = tuple(BlockState("minecraft:" + "a" * 1009 + f"{i:05}") for i in range(1000))
    entries = tuple(replace(entry, state_id=state.name,
                            appearance=replace(entry.appearance, state=state))
                    for entry, state in zip(p.universe.appearance_support, states))
    p = replace(p, universe=replace(p.universe, legal_states=states, appearance_support=entries))
    base = metadata_of_size(10_000)
    observations = tuple(forge(base, cell=(i, 0, 0), frame_id=f"frame{i:04}",
                               support_frames=(f"frame{i:04}",)) for i in range(100))
    # Coordinate decimal lengths would increase metadata: shorten one reference
    # by the exact delta so every admitted metadata record remains 10000 bytes.
    observations = tuple(forge(obs, provenance=(*obs.provenance[:-1],
                                obs.provenance[-1][len(str(i)) - 1:]))
                         for i, obs in enumerate(observations))
    assert_early(monkeypatch, p, observations, "payload_bytes", replay=False)


def test_payload_digest_tamper_and_self_consistent_incomplete_vector_refuse():
    result = subject.produce_diagnostics(policy(), (observation(),))
    record = result.to_record()
    record["observations"][0]["absolute"][0] = "3ff0000000000000"
    with pytest.raises(ValueError, match="digest"):
        subject.verify_payload(subject._serialize(record))
    record = result.to_record()
    record["observations"][0]["absolute"].pop()
    body = {k: v for k, v in record.items() if k != "payload_sha256"}
    record["payload_sha256"] = hashlib.sha256(subject._serialize(body)).hexdigest()
    with pytest.raises(ValueError, match="incomplete"):
        subject.verify_payload(subject._serialize(record))


def test_payload_exact_cost_bits_and_recomputed_digest():
    result = subject.produce_diagnostics(policy(), (observation(rgb=(20, 21, 22)),))
    record = json.loads(result.payload)
    for typed, encoded in zip(result.observations, record["observations"]):
        absolute = tuple(scoring.cost_from_bits(value) for value in encoded["absolute"])
        relative = tuple(scoring.cost_from_bits(value) for value in encoded["relative"])
        assert absolute == typed.scores.absolute
        assert relative == typed.scores.relative
    body = {key: value for key, value in record.items() if key != "payload_sha256"}
    assert hashlib.sha256(subject._serialize(body)).hexdigest() == result.payload_sha256
