"""Controls for the frozen image-derived output policy."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mc_photogrammetry.blocks.appearance import DEFAULT_PALETTE  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_output_policy as subject  # noqa: E402
from mc_photogrammetry.reconstruction import plane_face_observation as face  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement as placement  # noqa: E402
from mc_photogrammetry.reconstruction import plane_volume as volume  # noqa: E402
from mc_photogrammetry.reconstruction import state_promotion as promotion  # noqa: E402


POLICY_SHA256 = "9d9c94c1c8a6be84927d2d6eeb0308098b66fc093935dde9770c0ae285aa4cf1"
PALETTE_SHA256 = "42bdddb9cdb2ce0861541022ec0e5733a49252d1915cdfb354b9865f283af6ab"


def _fields(policy):
    return {field.name: getattr(policy, field.name)
            for field in dataclasses.fields(type(policy))}


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_exact_builtin_policy_and_hashes():
    policy = subject.IMAGE_DERIVED_OUTPUT_POLICY
    assert policy.policy_sha256 == POLICY_SHA256
    assert policy.palette_sha256 == PALETTE_SHA256
    assert policy.palette is DEFAULT_PALETTE
    assert policy.placement_budget.to_record() == placement.make_placement_budget(
        max_faces=256, max_supports_checked=64, max_support_cells=4096,
        max_frame_axis_lookups=4096, max_integer_checks=4096,
        max_anchors=65536).to_record()
    assert policy.volume_budget.to_record() == volume.make_volume_budget(
        max_faces=10_000, max_support_cells=10_000_000,
        max_contributions=10_000_000, max_coordinates=10_000_000,
        max_conflicts=100_000, max_omitted_unknown=1,
        max_dense_bounds_cells=10_000_000).to_record()
    assert policy.face_budget.to_record() == face.make_face_observation_budget(
        max_claims=10_000_000, max_frames=32,
        max_rectified_pixels=100_000_000, max_payload_bytes=100_000_000,
        max_replay_checks=1_000_000).to_record()
    assert policy.promotion_budget.to_record() == promotion.make_promotion_budget(
        max_cells=10_000_000, max_proposals=10_000_000,
        max_payload_bytes=100_000_000).to_record()
    assert (policy.max_cost, policy.min_margin, policy.max_observations,
            policy.max_candidate_evaluations,
            policy.max_proposal_payload_bytes,
            policy.allow_omitted_unknown) == (
                1e9, 0.0, 1000, 100_000, 10_000_000, False)


def test_record_replays_exactly_and_is_json_shaped():
    policy = subject.IMAGE_DERIVED_OUTPUT_POLICY
    assert subject.ImageDerivedOutputPolicy(**_fields(policy)) == policy
    record = policy.to_record()
    assert record["policy_sha256"] == POLICY_SHA256
    assert [item["state"]["name"] for item in record["palette"]] == [
        item.state.name for item in DEFAULT_PALETTE]


@pytest.mark.parametrize("field", [
    "max_cost", "min_margin", "max_observations",
    "max_candidate_evaluations", "max_proposal_payload_bytes",
    "allow_omitted_unknown", "palette_sha256", "policy_sha256",
])
def test_scalar_mutations_are_rejected(field):
    policy = subject.IMAGE_DERIVED_OUTPUT_POLICY
    changes = {
        "max_cost": 1e9 + 1.0, "min_margin": 0.1,
        "max_observations": 1001, "max_candidate_evaluations": 100_001,
        "max_proposal_payload_bytes": 10_000_001,
        "allow_omitted_unknown": True, "palette_sha256": "0" * 64,
        "policy_sha256": "0" * 64,
    }
    fields = _fields(policy)
    fields[field] = changes[field]
    with pytest.raises(ValueError):
        subject.ImageDerivedOutputPolicy(**fields)


def test_bool_is_not_accepted_as_an_integer_threshold():
    fields = _fields(subject.IMAGE_DERIVED_OUTPUT_POLICY)
    fields["max_observations"] = True
    with pytest.raises(ValueError):
        subject.ImageDerivedOutputPolicy(**fields)


def test_palette_order_and_content_are_frozen():
    fields = _fields(subject.IMAGE_DERIVED_OUTPUT_POLICY)
    fields["palette"] = tuple(reversed(DEFAULT_PALETTE))
    with pytest.raises(ValueError, match="exact default palette"):
        subject.ImageDerivedOutputPolicy(**fields)
    fields = _fields(subject.IMAGE_DERIVED_OUTPUT_POLICY)
    fields["palette"] = DEFAULT_PALETTE[:-1]
    with pytest.raises(ValueError, match="exact default palette"):
        subject.ImageDerivedOutputPolicy(**fields)


@pytest.mark.parametrize("budget_field,maker", [
    ("placement_budget", placement.make_placement_budget),
    ("volume_budget", volume.make_volume_budget),
    ("face_budget", face.make_face_observation_budget),
    ("promotion_budget", promotion.make_promotion_budget),
])
def test_every_budget_cap_mutation_is_rejected(budget_field, maker):
    policy = subject.IMAGE_DERIVED_OUTPUT_POLICY
    original = getattr(policy, budget_field).to_record()
    for cap in tuple(key for key in original if key != "budget_sha256"):
        caps = {key: value for key, value in original.items()
                if key != "budget_sha256"}
        caps[cap] += 1
        fields = _fields(policy)
        fields[budget_field] = maker(**caps)
        with pytest.raises(ValueError, match="policy digest"):
            subject.ImageDerivedOutputPolicy(**fields)


def test_forged_nested_budget_is_rejected_before_policy_digest():
    policy = subject.IMAGE_DERIVED_OUTPUT_POLICY
    forged = _forge(policy.volume_budget, max_faces=True)
    fields = _fields(policy)
    fields["volume_budget"] = forged
    with pytest.raises(ValueError, match="budgets do not replay"):
        subject.ImageDerivedOutputPolicy(**fields)


def test_no_caller_configuration_or_io_surface():
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("environ", "getenv", "argparse", "sys.argv", "open(",
                      "pathlib", "subprocess", "config", "**kwargs"):
        assert forbidden not in source
    assert len(source.splitlines()) < 500
