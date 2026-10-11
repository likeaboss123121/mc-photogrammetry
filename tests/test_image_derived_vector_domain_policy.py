"""Controls for the immutable image-derived C2-R production policy."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.pipeline import image_derived_vector_domain_policy as subject  # noqa: E402
from mc_photogrammetry.reconstruction.plane_vector_domain import C2RVectorBudget  # noqa: E402


def _expected_hash(record: dict) -> str:
    payload = dict(record)
    payload.pop("policy_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replace_budget(policy, **changes):
    values = policy.budget.to_record()
    values.update(changes)
    budget = C2RVectorBudget(**values)
    fields = {
        "schema": policy.schema,
        "producer": policy.producer,
        "version": policy.version,
        "comparison_tolerance": policy.comparison_tolerance,
        "budget": budget,
    }
    return subject.ImageDerivedVectorDomainPolicy(
        **fields, policy_sha256=subject._digest(subject._payload(**fields)))


def test_builtin_policy_has_exact_values_and_digest():
    policy = subject.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY
    assert policy.to_record() == {
        "schema": "mc-photogrammetry-image-derived-vector-domain-policy",
        "producer": "m9-m1-image-derived-vector-domain-policy-1",
        "version": 1,
        "comparison_tolerance": 1e-12,
        "budget": {
            "max_requested_match_identities": 10,
            "max_joint_entries": 2_000_000,
            "max_axis_residue_pairs": 10_000_000,
            "max_vector_candidates": 10_000_000,
            "max_provenance_references": 100_000_000,
        },
        "policy_sha256": policy.policy_sha256,
    }
    assert policy.policy_sha256 == _expected_hash(policy.to_record())
    assert dataclasses.replace(policy) == policy


@pytest.mark.parametrize("field", [
    "max_requested_match_identities", "max_joint_entries",
    "max_axis_residue_pairs", "max_vector_candidates",
    "max_provenance_references",
])
def test_each_cap_changes_policy_digest(field):
    policy = subject.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY
    changed = _replace_budget(policy, **{field: getattr(policy.budget, field) + 1})
    assert changed.policy_sha256 != policy.policy_sha256


@pytest.mark.parametrize("field,value", [
    ("max_requested_match_identities", 0),
    ("max_joint_entries", True),
    ("max_axis_residue_pairs", 0),
    ("max_vector_candidates", 0),
    ("max_provenance_references", 0),
])
def test_invalid_budget_caps_are_rejected(field, value):
    values = subject.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY.budget.to_record()
    values[field] = value
    with pytest.raises(ValueError):
        C2RVectorBudget(**values)


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("producer", "foreign"), ("version", 2),
    ("version", True), ("policy_sha256", "0" * 64),
])
def test_metadata_and_digest_mutations_are_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(subject.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY,
                            **{field: value})


@pytest.mark.parametrize("value", [0, True, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_tolerance_is_rejected(value):
    with pytest.raises(ValueError):
        dataclasses.replace(subject.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY,
                            comparison_tolerance=value)


def test_policy_is_frozen_and_has_no_caller_configuration_surface():
    policy = subject.IMAGE_DERIVED_VECTOR_DOMAIN_POLICY
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.version = 2
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("argparse", "sys.argv", "environ", "getenv", "config",
                      "pathlib", "open(", "subprocess"):
        assert forbidden not in source
