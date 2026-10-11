"""Controls for the immutable image-derived C2-R-B production policy."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.pipeline import image_derived_relative_translation_policy as subject  # noqa: E402
from mc_photogrammetry.reconstruction.plane_relative_translation import C2RBudget  # noqa: E402


def _expected_hash(record: dict) -> str:
    payload = dict(record)
    payload.pop("policy_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replace_budget(policy, **changes):
    values = policy.budget.to_record()
    values.update(changes)
    budget = C2RBudget(**values)
    fields = {
        "schema": policy.schema, "producer": policy.producer,
        "version": policy.version,
        "comparison_tolerance": policy.comparison_tolerance,
        "closure_tolerance": policy.closure_tolerance, "budget": budget,
    }
    return subject.ImageDerivedRelativeTranslationPolicy(
        **fields, policy_sha256=subject._digest(subject._payload(**fields)))


def test_builtin_policy_has_exact_values_and_digest():
    policy = subject.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY
    assert policy.comparison_tolerance == 1e-12
    assert policy.closure_tolerance == 0.0
    assert tuple(policy.budget.to_record().values()) == (
        10, 32, 10, 10, 10, 30, 2_000_000, 10_000_000,
        10_000_000, 10_000_000, 10_000_000, 10_000_000,
        10_000_000, 10_000_000, 100_000)
    assert policy.policy_sha256 == _expected_hash(policy.to_record())
    assert dataclasses.replace(policy) == policy


@pytest.mark.parametrize("field", list(C2RBudget.__dataclass_fields__))
def test_each_cap_changes_policy_digest(field):
    policy = subject.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY
    changed = _replace_budget(policy, **{field: getattr(policy.budget, field) + 1})
    assert changed.policy_sha256 != policy.policy_sha256


@pytest.mark.parametrize("field", list(C2RBudget.__dataclass_fields__))
def test_each_cap_rejects_bool(field):
    values = subject.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.budget.to_record()
    values[field] = True
    with pytest.raises(ValueError):
        C2RBudget(**values)


@pytest.mark.parametrize("field", list(C2RBudget.__dataclass_fields__))
def test_each_cap_rejects_zero(field):
    values = subject.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.budget.to_record()
    values[field] = 0
    with pytest.raises(ValueError):
        C2RBudget(**values)


@pytest.mark.parametrize("field", ["max_provenance_witnesses", "max_certificates"])
def test_witness_and_certificate_caps_require_two(field):
    values = subject.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY.budget.to_record()
    values[field] = 1
    with pytest.raises(ValueError):
        C2RBudget(**values)


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("producer", "foreign"), ("version", 2),
    ("version", True), ("policy_sha256", "0" * 64),
])
def test_metadata_and_digest_mutations_are_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(
            subject.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY,
            **{field: value})


@pytest.mark.parametrize("field", ["comparison_tolerance", "closure_tolerance"])
@pytest.mark.parametrize("value", [0, True, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_tolerances_are_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(
            subject.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY,
            **{field: value})


def test_policy_is_frozen_and_has_no_caller_configuration_surface():
    policy = subject.IMAGE_DERIVED_RELATIVE_TRANSLATION_POLICY
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.version = 2
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("argparse", "sys.argv", "environ", "getenv", "config",
                      "pathlib", "open(", "subprocess"):
        assert forbidden not in source
