"""Opt-in v2 policy identity for future exhaustive-or-abstain consumers.

This contract does not dispatch scoring or select retrieval/threshold behavior.
Scorer and proposal definitions are supplied, versioned identities, independent
of v1. Exhaustive means evaluating every canonical state in the bound inventory;
it never establishes physical-truth candidate recall or exact-render fidelity.
Incomplete/unknown support and over-budget work require downstream abstention.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from mc_photogrammetry.blocks.candidate_universe import (
    CandidateUniverse,
    _digest,
    _preflight,
    _record,
    _replay,
    _text,
    record_sha256,
    resource_contract,
)

POLICY_SCHEMA = "mc-photogrammetry-image-derived-output-policy-v2"
POLICY_VERSION = 2
MAX_OBSERVATIONS = 1000
MAX_CANDIDATE_EVALUATIONS = 100_000
MAX_PROPOSAL_PAYLOAD_BYTES = 10_000_000
MAX_POLICY_RECORD_BYTES = 32_000_000


def policy_resource_contract() -> dict:
    return {**resource_contract(), "max_policy_record_bytes": MAX_POLICY_RECORD_BYTES}


def _policy_view(value: Any) -> Any:
    if type(value) in (ScorerIdentity, ProposalIdentity, ImageDerivedOutputPolicyV2):
        result = {item.name: getattr(value, item.name, "0" * 64)
                  for item in fields(type(value))}
        if type(value) is ImageDerivedOutputPolicyV2:
            result.update(universe_sha256=getattr(value.universe, "universe_sha256", "0" * 64),
                          resource_contract=policy_resource_contract())
        return result
    return value


def _policy_preflight(value: Any) -> tuple[int, int]:
    return _preflight(value, max_bytes=MAX_POLICY_RECORD_BYTES, project=_policy_view)


def _validate_caps(value: Any) -> None:
    for label, maximum in (("max_observations", MAX_OBSERVATIONS),
                           ("max_candidate_evaluations", MAX_CANDIDATE_EVALUATIONS),
                           ("max_proposal_payload_bytes", MAX_PROPOSAL_PAYLOAD_BYTES)):
        entry = value[label] if type(value) is dict else getattr(value, label)
        _positive_cap(entry, maximum, label)


@dataclass(frozen=True, slots=True)
class ScorerIdentity:
    """Bind a named scoring model and explicit population-dependent semantics.

    Strings name separately defined semantics, not an implementation here. The
    definition digest must bind the caller's complete scoring specification.
    """

    scorer_id: str
    version: str
    named_model: str
    definition_sha256: str
    population_semantics: str
    normalization_semantics: str
    clipping_semantics: str

    def __post_init__(self) -> None:
        _policy_preflight(self)
        for item in fields(self):
            if item.name == "definition_sha256":
                _digest(getattr(self, item.name), item.name)
            else:
                _text(getattr(self, item.name), item.name)

    def to_record(self) -> dict:
        _policy_preflight(self)
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_record(cls, record: Any) -> ScorerIdentity:
        _policy_preflight(record)
        return cls(**_record(record, {item.name for item in fields(cls)}, "scorer"))


@dataclass(frozen=True, slots=True)
class ProposalIdentity:
    """Definition identity; tied alternatives remain explicit, never truncated."""

    proposal_id: str
    version: str
    definition_sha256: str
    observation_schema: str
    observation_version: str
    tie_semantics: str
    strategy: str = "exhaustive_or_abstain"
    incomplete_support_action: str = "abstain"
    over_budget_action: str = "abstain"

    def __post_init__(self) -> None:
        _policy_preflight(self)
        for item in fields(self):
            if item.name == "definition_sha256":
                _digest(getattr(self, item.name), item.name)
            else:
                _text(getattr(self, item.name), item.name)
        if (self.strategy != "exhaustive_or_abstain"
                or self.incomplete_support_action != "abstain"
                or self.over_budget_action != "abstain"):
            raise ValueError("v2 proposal semantics require exhaustive-or-abstain")

    def to_record(self) -> dict:
        _policy_preflight(self)
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_record(cls, record: Any) -> ProposalIdentity:
        _policy_preflight(record)
        return cls(**_record(record, {item.name for item in fields(cls)}, "proposal"))


def _positive_cap(value: Any, maximum: int, label: str) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1, {maximum}]")


@dataclass(frozen=True, slots=True)
class ImageDerivedOutputPolicyV2:
    """Immutable opt-in identity; future consumers must replay it before use.

    There is no singleton or normal-mode dispatch. Lower caps are permitted;
    initial ceilings cannot be raised by this contract. Work equal to a cap is
    allowed; work exceeding it must abstain atomically downstream. For 8192
    candidates, 12 observations need 98304 evaluations and 13 need 106496.
    """

    universe: CandidateUniverse
    scorer: ScorerIdentity
    proposal: ProposalIdentity
    max_observations: int = MAX_OBSERVATIONS
    max_candidate_evaluations: int = MAX_CANDIDATE_EVALUATIONS
    max_proposal_payload_bytes: int = MAX_PROPOSAL_PAYLOAD_BYTES
    schema: str = POLICY_SCHEMA
    version: int = POLICY_VERSION
    opt_in: bool = True
    completeness: str = "exhaustive"
    physical_truth_recall: str = "unestablished"
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_caps(self)
        _policy_preflight(self)
        if type(self.schema) is not str or self.schema != POLICY_SCHEMA:
            raise ValueError("unsupported v2 output policy schema")
        if type(self.version) is not int or self.version != POLICY_VERSION:
            raise ValueError("unsupported v2 output policy version")
        if (self.opt_in is not True or type(self.completeness) is not str
                or self.completeness != "exhaustive"
                or type(self.physical_truth_recall) is not str
                or self.physical_truth_recall != "unestablished"):
            raise ValueError("v2 must be opt-in, exhaustive, and claim no truth recall")
        if (type(self.universe) is not CandidateUniverse
                or type(self.scorer) is not ScorerIdentity
                or type(self.proposal) is not ProposalIdentity):
            raise ValueError("v2 requires immutable universe/scorer/proposal identities")
        # Validate nested records again, including computed digest replay.
        CandidateUniverse.from_record(self.universe.to_record())
        ScorerIdentity.from_record(self.scorer.to_record())
        ProposalIdentity.from_record(self.proposal.to_record())
        object.__setattr__(self, "policy_sha256", record_sha256(self._payload()))

    def _payload(self) -> dict:
        _validate_caps(self)
        _policy_preflight(self)
        return {"schema": self.schema, "version": self.version, "opt_in": self.opt_in,
                "resource_contract": policy_resource_contract(),
                "completeness": self.completeness,
                "physical_truth_recall": self.physical_truth_recall,
                "universe_sha256": self.universe.universe_sha256,
                "universe": self.universe.to_record(), "scorer": self.scorer.to_record(),
                "proposal": self.proposal.to_record(), "max_observations": self.max_observations,
                "max_candidate_evaluations": self.max_candidate_evaluations,
                "max_proposal_payload_bytes": self.max_proposal_payload_bytes}

    def to_record(self) -> dict:
        _validate_caps(self)
        _policy_preflight(self)
        return {**self._payload(), "policy_sha256": self.policy_sha256}

    @classmethod
    def from_record(cls, record: Any) -> ImageDerivedOutputPolicyV2:
        # Read only the three operational caps before any nested replay. The
        # iterative preflight must precede even the complete field-set copy.
        if type(record) is not dict or any(label not in record for label in (
                "max_observations", "max_candidate_evaluations", "max_proposal_payload_bytes")):
            raise ValueError("v2 policy record has missing operational caps")
        _validate_caps(record)
        _policy_preflight(record)
        keys = {item.name for item in fields(cls)} | {"universe_sha256", "resource_contract"}
        record = _record(record, keys, "v2 policy")
        _replay(record["resource_contract"], policy_resource_contract(), "policy resources")
        values = {item.name: record[item.name] for item in fields(cls) if item.init}
        values.update(universe=CandidateUniverse.from_record(record["universe"]),
                      scorer=ScorerIdentity.from_record(record["scorer"]),
                      proposal=ProposalIdentity.from_record(record["proposal"]))
        result = cls(**values)
        _replay(result.to_record(), record, "v2 policy")
        return result
