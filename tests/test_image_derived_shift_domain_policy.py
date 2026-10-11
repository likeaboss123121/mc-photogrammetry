"""Focused controls for the frozen image-derived P1-D policy."""
from __future__ import annotations

import dataclasses
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.pipeline import image_derived_shift_domain_policy as subject  # noqa: E402
from mc_photogrammetry.reconstruction import plane_graph  # noqa: E402
from mc_photogrammetry.reconstruction.plane_shift_domain import ShiftDomainBudget  # noqa: E402


EXPECTED_POLICY_SHA256 = \
    "7b8135026e7e75cbce687e957c043b216ded14ea4a1ca669bd65d99125f467cd"


def _policy(**changes):
    original = subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    values = {field.name: getattr(original, field.name)
              for field in dataclasses.fields(original)}
    values.update(changes)
    return subject.ImageDerivedShiftDomainPolicy(**values)


def _forge_budget(budget, **changes):
    forged = object.__new__(ShiftDomainBudget)
    for field in dataclasses.fields(budget):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(budget, field.name)))
    return forged


def test_builtin_exact_values_record_and_digest():
    policy = subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    assert policy.schema == subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_SCHEMA
    assert policy.producer == subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_PRODUCER
    assert policy.version == subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY_VERSION == 3
    assert policy.correlation_algorithm_id == plane_graph.PLANE_CORRELATION_ALGORITHM
    assert policy.max_shift_cells == 32
    assert policy.budget.to_record() == {
        "max_domains": 10,
        "max_padded_fft_cells": 2_000_000,
        "max_shift_positions": 100_000,
        "max_overlap_cell_visits": 100_000_000,
        "max_buffered_entries": 100_000,
    }
    assert policy.policy_sha256 == EXPECTED_POLICY_SHA256
    assert json.loads(json.dumps(policy.to_record())) == policy.to_record()


def test_exact_record_replays_through_public_constructor():
    record = subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.to_record()
    record["budget"] = ShiftDomainBudget(**record["budget"])
    assert subject.ImageDerivedShiftDomainPolicy(**record).to_record() \
        == subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.to_record()


@pytest.mark.parametrize("field,value", [
    ("schema", "mc-photogrammetry-image-derived-shift-domain-policy"),
    ("schema", "foreign"),
    ("schema", "mc-photogrammetry-image-derived-shift-domain-policy-v2"),
    ("producer", "m9-i-image-derived-shift-domain-policy-1"),
    ("producer", "foreign"),
    ("producer", "m9-i-image-derived-shift-domain-policy-2"),
    ("version", 1), ("version", 2), ("version", 4), ("version", True),
    ("correlation_algorithm_id", "fixed-mask-v1"),
    ("correlation_algorithm_id", "separate-mask-halo-v2"),
    ("max_shift_cells", 31), ("max_shift_cells", 33),
    ("max_shift_cells", 0), ("max_shift_cells", -1),
    ("max_shift_cells", True), ("max_shift_cells", 32.0),
    ("policy_sha256", "0" * 64), ("policy_sha256", "invalid"),
])
def test_mutated_policy_fields_are_rejected(field, value):
    with pytest.raises(ValueError):
        _policy(**{field: value})


def test_wrong_budget_type_is_rejected():
    with pytest.raises(ValueError, match="ShiftDomainBudget"):
        _policy(budget=subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.budget.to_record())


@pytest.mark.parametrize("field", (
    "max_domains", "max_padded_fft_cells", "max_shift_positions",
    "max_overlap_cell_visits", "max_buffered_entries",
))
@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_every_forged_budget_cap_is_replayed_and_rejected(field, value):
    budget = subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.budget
    forged = _forge_budget(budget, **{field: value})
    with pytest.raises(ValueError, match="budget"):
        _policy(budget=forged)


@pytest.mark.parametrize("field", (
    "max_domains", "max_padded_fft_cells", "max_shift_positions",
    "max_overlap_cell_visits", "max_buffered_entries",
))
def test_valid_cap_mutation_with_stale_policy_digest_is_rejected(field):
    values = subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY.budget.to_record()
    values[field] += 1
    with pytest.raises(ValueError, match="digest"):
        _policy(budget=ShiftDomainBudget(**values))


def test_policy_and_budget_are_frozen_and_slotted():
    policy = subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_shift_cells = 8
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.budget.max_domains = 11
    assert not hasattr(policy, "__dict__") and not hasattr(policy.budget, "__dict__")


def test_record_and_digest_are_deterministic_and_input_immutable():
    policy = subject.IMAGE_DERIVED_SHIFT_DOMAIN_POLICY
    first = policy.to_record()
    second = policy.to_record()
    assert first == second and first is not second
    first["budget"]["max_domains"] = 999
    assert policy.budget.max_domains == 10
    assert policy.to_record()["policy_sha256"] == EXPECTED_POLICY_SHA256


def test_record_has_no_defaults_or_external_configuration_surface():
    parameters = inspect.signature(subject.ImageDerivedShiftDomainPolicy).parameters
    assert tuple(parameters) == (
        "schema", "producer", "version", "correlation_algorithm_id",
        "max_shift_cells", "budget", "policy_sha256")
    assert all(value.default is inspect.Parameter.empty
               for value in parameters.values())
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "config"):
        assert forbidden not in source
