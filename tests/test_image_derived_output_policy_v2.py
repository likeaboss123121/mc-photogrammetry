"""Synthetic v2 policy identity and frozen v1 isolation checks."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from mc_photogrammetry.blocks.appearance import DEFAULT_PALETTE, PaletteColor
from mc_photogrammetry.blocks import candidate_universe as universe_subject
from mc_photogrammetry.blocks.candidate_universe import (
    AppearanceSupport,
    CandidateUniverse,
    InventoryCompleteness,
    LegalStateManifest,
)
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.pipeline import image_derived_output_policy as v1
from mc_photogrammetry.pipeline import image_derived_output_policy_v2 as subject


def make_policy():
    state = BlockState("minecraft:birch_planks")
    universe = CandidateUniverse(
        minecraft_version="synthetic-1", source_archive_sha256="1" * 64,
        manifest=LegalStateManifest("synthetic-1", "2" * 64, "synthetic-states",
                                    "synthetic-enumerator", "1", ("synthetic:test",),
                                    InventoryCompleteness("exhaustive")),
        decoder_version="synthetic-decoder-1", support_version="synthetic-support-1",
        decoder_settings=(), legal_states=(state,),
        appearance_support=(AppearanceSupport(
            "minecraft:birch_planks", "supported", appearance=PaletteColor(state, (80, 80, 80)),
            model_record_sha256="3" * 64, texture_record_sha256="4" * 64),))
    scorer = subject.ScorerIdentity(
        scorer_id="synthetic-scorer", version="1", named_model="synthetic-model-1",
        definition_sha256="5" * 64, population_semantics="synthetic-population-definition-1",
        normalization_semantics="synthetic-normalization-definition-1",
        clipping_semantics="synthetic-clipping-definition-1")
    proposal = subject.ProposalIdentity(
        proposal_id="synthetic-proposal", version="1", definition_sha256="6" * 64,
        observation_schema="synthetic-observation", observation_version="1",
        tie_semantics="synthetic-preserve-tied-alternatives-definition-1")
    return subject.ImageDerivedOutputPolicyV2(universe, scorer, proposal)


def test_policy_canonical_digest_replay_and_initial_exact_ceilings():
    policy = make_policy()
    assert (policy.max_observations, policy.max_candidate_evaluations,
            policy.max_proposal_payload_bytes) == (1000, 100_000, 10_000_000)
    assert policy.opt_in is True
    assert policy.completeness == "exhaustive"
    assert policy.physical_truth_recall == "unestablished"
    assert policy.proposal.strategy == "exhaustive_or_abstain"
    record = policy.to_record()
    payload = {key: value for key, value in record.items() if key != "policy_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    assert policy.policy_sha256 == hashlib.sha256(encoded).hexdigest()
    assert record["universe_sha256"] == policy.universe.universe_sha256
    assert subject.ImageDerivedOutputPolicyV2.from_record(json.loads(json.dumps(record))) == policy


@pytest.mark.parametrize("cap,maximum", [
    ("max_observations", 1000), ("max_candidate_evaluations", 100_000),
    ("max_proposal_payload_bytes", 10_000_000),
])
def test_exact_cap_limits_accept_one_and_ceiling_reject_over_budget_configuration(cap, maximum):
    policy = make_policy()
    assert getattr(dataclasses.replace(policy, **{cap: 1}), cap) == 1
    assert getattr(dataclasses.replace(policy, **{cap: maximum}), cap) == maximum
    for invalid in (0, -1, maximum + 1, True, False, 1.0, "1", None):
        with pytest.raises(ValueError, match="must be an integer"):
            dataclasses.replace(policy, **{cap: invalid})
    changed = dataclasses.replace(policy, **{cap: maximum - 1})
    assert changed.policy_sha256 != policy.policy_sha256
    assert subject.ImageDerivedOutputPolicyV2.from_record(changed.to_record()) == changed


@pytest.mark.parametrize("field,value", [
    ("scorer_id", "other"), ("version", "2"), ("named_model", "other-model"),
    ("definition_sha256", "7" * 64), ("population_semantics", "other-population"),
    ("normalization_semantics", "other-normalization"), ("clipping_semantics", "other-clipping"),
])
def test_every_scorer_identity_and_semantics_field_binds_policy(field, value):
    policy = make_policy()
    changed = dataclasses.replace(
        policy, scorer=dataclasses.replace(policy.scorer, **{field: value}))
    assert changed.policy_sha256 != policy.policy_sha256


@pytest.mark.parametrize("field,value", [
    ("proposal_id", "other"), ("version", "2"), ("definition_sha256", "7" * 64),
    ("observation_schema", "other"), ("observation_version", "2"), ("tie_semantics", "other-ties"),
])
def test_every_proposal_and_observation_identity_field_binds_policy(field, value):
    policy = make_policy()
    changed = dataclasses.replace(
        policy, proposal=dataclasses.replace(policy.proposal, **{field: value}))
    assert changed.policy_sha256 != policy.policy_sha256


def test_archive_and_manifest_universe_identity_changes_policy():
    policy = make_policy()
    for universe in (
        dataclasses.replace(policy.universe, source_archive_sha256="7" * 64),
        dataclasses.replace(policy.universe, manifest=dataclasses.replace(
            policy.universe.manifest, source_manifest_sha256="7" * 64)),
    ):
        assert dataclasses.replace(policy, universe=universe).policy_sha256 != policy.policy_sha256


def test_incomplete_support_contracts_remain_representable_and_require_downstream_abstention():
    policy = make_policy()
    for disposition in ("unsupported", "unknown"):
        universe = dataclasses.replace(policy.universe, appearance_support=(
            AppearanceSupport(policy.universe.state_ids[0], disposition, ("not-implemented",)),))
        incomplete = dataclasses.replace(policy, universe=universe)
        assert incomplete.universe.requires_abstention
        assert incomplete.proposal.incomplete_support_action == "abstain"
        assert incomplete.proposal.over_budget_action == "abstain"
        assert subject.ImageDerivedOutputPolicyV2.from_record(incomplete.to_record()) == incomplete
    # Budget arithmetic is a contract example, not a throughput measurement.
    assert 12 * 8192 == 98_304 <= policy.max_candidate_evaluations
    assert 13 * 8192 == 106_496 > policy.max_candidate_evaluations


@pytest.mark.parametrize("mutation", [
    "policy_digest", "universe_identity", "scorer", "cap", "nested_archive", "extra", "missing",
])
def test_tampered_and_stale_policy_records_are_rejected(mutation):
    record = copy.deepcopy(make_policy().to_record())
    if mutation == "policy_digest":
        record["policy_sha256"] = "0" * 64
    elif mutation == "universe_identity":
        record["universe_sha256"] = "0" * 64
    elif mutation == "scorer":
        record["scorer"]["version"] = "2"
    elif mutation == "cap":
        record["max_observations"] -= 1
    elif mutation == "nested_archive":
        record["universe"]["source_archive_sha256"] = "0" * 64
    elif mutation == "extra":
        record["extra"] = "ignored?"
    else:
        del record["proposal"]["tie_semantics"]
    with pytest.raises(ValueError):
        subject.ImageDerivedOutputPolicyV2.from_record(record)


@pytest.mark.parametrize("changes", [
    {"schema": "unknown"}, {"version": 1}, {"version": 3}, {"version": True},
    {"opt_in": False}, {"opt_in": 1}, {"completeness": "certified_under_named_model"},
    {"physical_truth_recall": "exhaustive"}, {"universe": {}}, {"scorer": {}}, {"proposal": {}},
])
def test_unknown_schema_version_and_non_opt_in_or_incorrect_claims_are_rejected(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(make_policy(), **changes)


@pytest.mark.parametrize("changes", [
    {"strategy": "top_k"}, {"incomplete_support_action": "guess"},
    {"over_budget_action": "truncate"}, {"tie_semantics": ""},
    {"definition_sha256": "bad"}, {"observation_version": True},
])
def test_proposal_semantics_cannot_enable_truncation_or_unknown_support_winners(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(make_policy().proposal, **changes)


def test_frozen_detached_records_and_forged_nested_identity_rejection():
    policy = make_policy()
    record = policy.to_record()
    record["scorer"]["version"] = "new"
    record["universe"]["appearance_support"][0]["appearance"]["rgb"][0] = 1
    assert policy.to_record() == make_policy().to_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_observations = 1
    forged = copy.copy(policy.universe)
    object.__setattr__(forged, "universe_sha256", "0" * 64)
    with pytest.raises(ValueError):
        dataclasses.replace(policy, universe=forged)


def test_v1_policy_import_visible_constants_and_source_pins_remain_frozen():
    root = Path(__file__).resolve().parents[1]
    pins = {
        "src/mc_photogrammetry/blocks/appearance.py":
            "2380f1676e713d340432dfa28f2146b924206157bbb41f79f8c39ed897c621b8",
        "src/mc_photogrammetry/domain/models.py":
            "6a87d543661cc62ada9a154bcdaa089ca844f41691ff3b53207ad89e3f05a478",
        "src/mc_photogrammetry/pipeline/image_derived_output_policy.py":
            "eeef921e8f47925f1611354a7048947c3dc79c215750884d0900c5582803bd4d",
        "src/mc_photogrammetry/reconstruction/appearance_state_proposals.py":
            "39ecfc70415472a7770a76fe492cc251d42233a43a557ecbf60305ae8846876a",
    }
    for path, expected in pins.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    singleton = v1.IMAGE_DERIVED_OUTPUT_POLICY
    record = singleton.to_record()
    make_policy()
    assert v1.IMAGE_DERIVED_OUTPUT_POLICY is singleton
    assert singleton.to_record() == record
    assert singleton.palette is DEFAULT_PALETTE
    assert len(DEFAULT_PALETTE) == 8
    assert v1.IMAGE_DERIVED_OUTPUT_POLICY_VERSION == 1
    assert v1.IMAGE_DERIVED_OUTPUT_POLICY_PRODUCER == "r2-p-image-derived-output-policy-1"
    assert singleton.policy_sha256 == (
        "9d9c94c1c8a6be84927d2d6eeb0308098b66fc093935dde9770c0ae285aa4cf1")
    assert singleton.palette_sha256 == (
        "42bdddb9cdb2ce0861541022ec0e5733a49252d1915cdfb354b9865f283af6ab")
    assert (singleton.max_observations, singleton.max_candidate_evaluations,
            singleton.max_proposal_payload_bytes) == (1000, 100_000, 10_000_000)
    assert singleton.max_cost == 1e9 and singleton.min_margin == 0.0
    assert singleton.allow_omitted_unknown is False
    source = Path(subject.__file__).read_text(encoding="utf-8")
    imports = [node for node in ast.walk(ast.parse(source))
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("image_derived_output_policy" not in (getattr(node, "module", "") or "")
               for node in imports)
    assert not hasattr(subject, "IMAGE_DERIVED_OUTPUT_POLICY")


def forbid_work(*args, **kwargs):
    pytest.fail("policy refused too late: nested replay/copy/hash already ran")


def test_policy_resource_envelope_is_distinct_from_proposal_budget_and_bound_to_identity():
    policy = make_policy()
    record = policy.to_record()
    assert record["resource_contract"] == {
        **universe_subject.resource_contract(), "max_policy_record_bytes": 32000000}
    assert subject.MAX_POLICY_RECORD_BYTES != policy.max_proposal_payload_bytes
    record["resource_contract"]["max_policy_record_bytes"] -= 1
    with pytest.raises(ValueError, match="policy resources"):
        subject.ImageDerivedOutputPolicyV2.from_record(record)


def test_exact_full_policy_bytes_and_one_over_before_nested_replay_copy_hash(monkeypatch):
    policy = make_policy()
    record = policy.to_record()
    measured = len(universe_subject.canonical_json(record).encode("utf-8"))
    assert subject._policy_preflight(policy)[1] == measured
    assert subject._policy_preflight(record)[1] == measured
    original = subject._policy_preflight

    def reduced(value):
        # A smaller test envelope uses the production traversal and preserves
        # the fixed resource declaration and independent proposal budget.
        return subject._preflight(value, max_bytes=measured, project=subject._policy_view)

    monkeypatch.setattr(subject, "_policy_preflight", reduced)
    assert dataclasses.replace(policy).to_record() == record
    assert subject.ImageDerivedOutputPolicyV2.from_record(record) == policy
    longer_scorer = dataclasses.replace(policy.scorer, version=policy.scorer.version + "x")
    bad_record = {**record, "scorer": {**record["scorer"], "version": longer_scorer.version}}
    bad_object = copy.copy(policy)
    object.__setattr__(bad_object, "scorer", longer_scorer)
    assert original(bad_record)[1] == measured + 1
    monkeypatch.setattr(subject.CandidateUniverse, "from_record", forbid_work)
    monkeypatch.setattr(subject.CandidateUniverse, "to_record", forbid_work)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    for invoke in (lambda: dataclasses.replace(policy, scorer=longer_scorer),
                   bad_object.to_record,
                   lambda: subject.ImageDerivedOutputPolicyV2.from_record(bad_record)):
        with pytest.raises(ValueError, match="byte/metadata"):
            invoke()
    assert bad_record["max_proposal_payload_bytes"] == subject.MAX_PROPOSAL_PAYLOAD_BYTES


@pytest.mark.parametrize("cap,maximum", [
    ("max_observations", 1000), ("max_candidate_evaluations", 100000),
    ("max_proposal_payload_bytes", 10000000),
])
def test_policy_cap_invalidity_precedes_nested_universe_replay(monkeypatch, cap, maximum):
    policy = make_policy()
    record = policy.to_record()
    record[cap] = maximum + 1
    # Even an oversized/cyclic nested universe cannot change which cap error
    # is returned or cause universe serialization/replay to begin.
    record["universe"] = record
    monkeypatch.setattr(subject.CandidateUniverse, "from_record", forbid_work)
    monkeypatch.setattr(subject.CandidateUniverse, "to_record", forbid_work)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    for invoke in (lambda: dataclasses.replace(policy, **{cap: maximum + 1}),
                   lambda: subject.ImageDerivedOutputPolicyV2.from_record(record)):
        with pytest.raises(ValueError, match=f"{cap} must be an integer"):
            invoke()


@pytest.mark.parametrize("entry", ["scorer", "proposal"])
def test_policy_standalone_metadata_boundary_and_complete_overage_order(monkeypatch, entry):
    policy = make_policy()
    result = getattr(policy, entry)
    label = "population_semantics" if entry == "scorer" else "tie_semantics"
    exact = dataclasses.replace(result, **{
        label: "x" * universe_subject.MAX_STRING_CHARACTERS})
    assert type(exact).from_record(exact.to_record()) == exact
    embedded = dataclasses.replace(policy, **{entry: exact})
    assert subject.ImageDerivedOutputPolicyV2.from_record(embedded.to_record()) == embedded
    record = policy.to_record()
    record[entry][label] = "x" * (universe_subject.MAX_STRING_CHARACTERS + 1)
    bad_object = copy.copy(policy)
    bad_identity = copy.copy(exact)
    object.__setattr__(bad_identity, label, record[entry][label])
    object.__setattr__(bad_object, entry, bad_identity)
    monkeypatch.setattr(subject.CandidateUniverse, "from_record", forbid_work)
    monkeypatch.setattr(subject.CandidateUniverse, "to_record", forbid_work)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    for invoke in (lambda: dataclasses.replace(exact, **{label: record[entry][label]}),
                   lambda: type(exact).from_record(record[entry]),
                   lambda: dataclasses.replace(bad_object), bad_object.to_record,
                   lambda: subject.ImageDerivedOutputPolicyV2.from_record(record)):
        with pytest.raises(ValueError, match="string resource"):
            invoke()


@pytest.mark.parametrize("kind", ["depth", "cycle", "nodes"])
def test_full_policy_preflight_precedes_universe_and_identity_decoders(monkeypatch, kind):
    record = make_policy().to_record()
    if kind == "depth":
        value = None
        for _ in range(universe_subject.MAX_RECORD_DEPTH):
            value = [value]
        record["extra"] = value
    elif kind == "cycle":
        record["extra"] = record
    else:
        nodes = subject._policy_preflight(record)[0]
        monkeypatch.setattr(universe_subject, "MAX_RECORD_NODES", nodes - 1)
    monkeypatch.setattr(subject.CandidateUniverse, "from_record", forbid_work)
    monkeypatch.setattr(subject.ScorerIdentity, "from_record", forbid_work)
    monkeypatch.setattr(subject.ProposalIdentity, "from_record", forbid_work)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    with pytest.raises(ValueError, match="node/depth|cycles|node resource"):
        subject.ImageDerivedOutputPolicyV2.from_record(record)
