"""Controls for the immutable image-derived V1 support policy."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.pipeline import image_derived_region_support_policy as subject  # noqa: E402
from mc_photogrammetry.reconstruction import plane_region_support  # noqa: E402


CAPS = ("max_cells", "max_region_pixel_tests", "max_usable_pixel_tests",
        "max_polygon_vertices", "max_supported_cells")


def _expected_hash(record):
    payload = dict(record)
    payload.pop("policy_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _changed(policy, field):
    values = policy.budget.to_record()
    values.pop("budget_sha256")
    values[field] += 1
    budget = plane_region_support.make_support_budget(**values)
    fields = {"schema": policy.schema, "producer": policy.producer,
              "version": policy.version, "budget": budget}
    return subject.ImageDerivedRegionSupportPolicy(
        **fields, policy_sha256=subject._digest(subject._payload(**fields)))


def test_builtin_exact_values_and_digest():
    policy = subject.IMAGE_DERIVED_REGION_SUPPORT_POLICY
    record = policy.budget.to_record()
    record.pop("budget_sha256")
    assert tuple(record.values()) == (
        10_000_000, 100_000_000, 100_000_000, 100_000_000, 10_000_000)
    assert policy.policy_sha256 == _expected_hash(policy.to_record())
    assert dataclasses.replace(policy) == policy


@pytest.mark.parametrize("field", CAPS)
def test_each_cap_changes_policy_digest(field):
    policy = subject.IMAGE_DERIVED_REGION_SUPPORT_POLICY
    assert _changed(policy, field).policy_sha256 != policy.policy_sha256


@pytest.mark.parametrize("field", CAPS)
@pytest.mark.parametrize("value", [0, True])
def test_each_cap_rejects_zero_and_bool(field, value):
    values = subject.IMAGE_DERIVED_REGION_SUPPORT_POLICY.budget.to_record()
    values.pop("budget_sha256")
    values[field] = value
    with pytest.raises(ValueError):
        plane_region_support.make_support_budget(**values)


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("producer", "foreign"), ("version", 2),
    ("version", True), ("policy_sha256", "0" * 64)])
def test_metadata_and_digest_mutations_are_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(subject.IMAGE_DERIVED_REGION_SUPPORT_POLICY,
                            **{field: value})


def test_frozen_and_no_caller_configuration_surface():
    policy = subject.IMAGE_DERIVED_REGION_SUPPORT_POLICY
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.version = 2
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("argparse", "sys.argv", "environ", "getenv", "config",
                      "pathlib", "open(", "subprocess"):
        assert forbidden not in source
