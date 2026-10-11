"""Controls for the immutable image-derived V2-B production policy."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.pipeline import image_derived_phase_anchor_policy as subject  # noqa: E402
from mc_photogrammetry.reconstruction import plane_phase_anchor  # noqa: E402


def _expected_hash(record: dict) -> str:
    payload = dict(record)
    payload.pop("policy_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replace_budget(policy, **changes):
    values = policy.budget.to_record()
    values.pop("budget_sha256")
    values.update(changes)
    budget = plane_phase_anchor.make_anchor_budget(**values)
    fields = {
        "schema": policy.schema, "producer": policy.producer,
        "version": policy.version, "budget": budget,
        "calibration_sha256": policy.calibration_sha256,
    }
    return subject.ImageDerivedPhaseAnchorPolicy(
        **fields, policy_sha256=subject._digest(subject._payload(**fields)))


def test_builtin_policy_has_exact_values_and_digests():
    policy = subject.IMAGE_DERIVED_PHASE_ANCHOR_POLICY
    record = policy.budget.to_record()
    record.pop("budget_sha256")
    assert tuple(record.values()) == (
        20, 10_000_000, 32, 10_000_000, 30_000_000, 20,
        200_000_000, 20, 20_000_000, 10_000_000, 100_000_000, 40,
        10_000_000, 1_000_000, 20, 10_000_000)
    assert policy.calibration_sha256 == \
        plane_phase_anchor.V2B_BUILTIN_CALIBRATION.calibration_sha256
    assert policy.policy_sha256 == _expected_hash(policy.to_record())
    assert dataclasses.replace(policy) == policy


@pytest.mark.parametrize("field", [
    name for name in plane_phase_anchor.V2BAnchorBudget.__dataclass_fields__
    if name != "budget_sha256"
])
def test_each_cap_changes_policy_digest(field):
    policy = subject.IMAGE_DERIVED_PHASE_ANCHOR_POLICY
    changed = _replace_budget(policy, **{field: getattr(policy.budget, field) + 1})
    assert changed.policy_sha256 != policy.policy_sha256


@pytest.mark.parametrize("field", [
    name for name in plane_phase_anchor.V2BAnchorBudget.__dataclass_fields__
    if name != "budget_sha256"
])
@pytest.mark.parametrize("value", [0, True])
def test_each_cap_rejects_zero_and_bool(field, value):
    values = subject.IMAGE_DERIVED_PHASE_ANCHOR_POLICY.budget.to_record()
    values.pop("budget_sha256")
    values[field] = value
    with pytest.raises(ValueError):
        plane_phase_anchor.make_anchor_budget(**values)


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("producer", "foreign"), ("version", 2),
    ("version", True), ("calibration_sha256", "0" * 64),
    ("policy_sha256", "0" * 64),
])
def test_metadata_calibration_and_digest_mutations_are_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(subject.IMAGE_DERIVED_PHASE_ANCHOR_POLICY,
                            **{field: value})


def test_alias_caps_cover_declared_maxima():
    budget = subject.IMAGE_DERIVED_PHASE_ANCHOR_POLICY.budget
    assert budget.max_boundary_interior_terms == 2 * budget.max_observations
    assert budget.max_replay_crosslinks == 2 * budget.max_source_records
    assert budget.max_hypotheses_total == 3 * budget.max_hypotheses_per_axis
    assert budget.max_contributor_refs_total == \
        budget.max_hypotheses_per_axis * budget.max_contributor_refs_per_hypothesis


def test_policy_is_frozen_and_has_no_caller_configuration_surface():
    policy = subject.IMAGE_DERIVED_PHASE_ANCHOR_POLICY
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.version = 2
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("argparse", "sys.argv", "environ", "getenv", "config",
                      "pathlib", "open(", "subprocess"):
        assert forbidden not in source
