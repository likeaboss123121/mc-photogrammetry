"""Controls for the immutable image-derived V2-A production policy."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.pipeline import image_derived_endpoint_provenance_policy as subject  # noqa: E402
from mc_photogrammetry.reconstruction.plane_endpoint_provenance import V2ABudget  # noqa: E402


def _expected_hash(record: dict) -> str:
    payload = dict(record)
    payload.pop("policy_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replace_budget(policy, **changes):
    values = policy.budget.to_record()
    values.update(changes)
    budget = V2ABudget(**values)
    fields = {"schema": policy.schema, "producer": policy.producer,
              "version": policy.version, "budget": budget}
    return subject.ImageDerivedEndpointProvenancePolicy(
        **fields, policy_sha256=subject._digest(subject._payload(**fields)))


def test_builtin_policy_has_exact_values_and_digest():
    policy = subject.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY
    assert tuple(policy.budget.to_record().values()) == (
        20, 10, 10_000_000, 40_000_000, 10_000_000, 10_000_000,
        10_000_000, 20_000_000, 2, 2)
    assert policy.policy_sha256 == _expected_hash(policy.to_record())
    assert dataclasses.replace(policy) == policy


@pytest.mark.parametrize("field", list(V2ABudget.__dataclass_fields__))
def test_each_cap_changes_policy_digest(field):
    policy = subject.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY
    changed = _replace_budget(policy, **{field: getattr(policy.budget, field) + 1})
    assert changed.policy_sha256 != policy.policy_sha256


@pytest.mark.parametrize("field", list(V2ABudget.__dataclass_fields__))
def test_each_cap_rejects_bool(field):
    values = subject.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY.budget.to_record()
    values[field] = True
    with pytest.raises(ValueError):
        V2ABudget(**values)


@pytest.mark.parametrize("field", list(V2ABudget.__dataclass_fields__))
def test_each_cap_rejects_zero(field):
    values = subject.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY.budget.to_record()
    values[field] = 0
    with pytest.raises(ValueError):
        V2ABudget(**values)


def test_distinct_assignment_cap_requires_two():
    values = subject.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY.budget.to_record()
    values["max_distinct_assignments"] = 1
    with pytest.raises(ValueError):
        V2ABudget(**values)


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("producer", "foreign"), ("version", 2),
    ("version", True), ("policy_sha256", "0" * 64),
])
def test_metadata_and_digest_mutations_are_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(
            subject.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY,
            **{field: value})


def test_policy_is_frozen_and_has_no_caller_configuration_surface():
    policy = subject.IMAGE_DERIVED_ENDPOINT_PROVENANCE_POLICY
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.version = 2
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("argparse", "sys.argv", "environ", "getenv", "config",
                      "pathlib", "open(", "subprocess"):
        assert forbidden not in source
