"""Focused tests for the evaluation-only E4 cross-view label-consistency audit.

Every test runs on synthetic, in-memory E3-shaped artifacts written under
``tmp_path``. No test loads the real factory artifacts, a screenshot, a
reference, a camera manifest, a scene definition, a JAR or a T2/T6 fixture, and
nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "audit_factory_texture_patch_consistency.py"
)
_spec = importlib.util.spec_from_file_location(
    "audit_factory_texture_patch_consistency_under_test", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
e4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e4  # dataclasses(slots=True) resolves the module by name
_spec.loader.exec_module(e4)

PATCH = e4.PATCH_SIZE
GROUP_A = "1" * 64
GROUP_B = "2" * 64
STAGING = "b" * 64
SHOT_1 = "3" * 64
SHOT_2 = "4" * 64
SHOT_3 = "5" * 64
STATE_STONE = {"name": "minecraft:stone", "properties": []}
STATE_DIRT = {"name": "minecraft:dirt", "properties": []}


# --------------------------------------------------------------------------- #
# Synthetic E3 artifact builders
# --------------------------------------------------------------------------- #


def _array_record(array: np.ndarray) -> dict:
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": [int(size) for size in value.shape],
        "base64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _flat(value: float) -> np.ndarray:
    return np.full((PATCH, PATCH, 3), float(value), dtype=np.float32)


def _example(
    *,
    scene_group: str = GROUP_A,
    world_identity: str = "Build A",
    shot: str = SHOT_1,
    frame_id: str = "frame-0",
    frame_key: str | None = None,
    world_coordinate: tuple[int, int, int] = (1, 2, 3),
    slot: str = "up",
    state: dict | None = None,
    patch: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
    fold: str = "train",
    dataset_id: str = "ds",
    scene_id: str = "scene",
) -> dict:
    valid_array = np.ones((PATCH, PATCH), dtype=bool) if valid is None else np.asarray(valid, bool)
    coverage_array = (
        np.ones((PATCH, PATCH), dtype=np.float32) if coverage is None else np.asarray(coverage, np.float32)
    )
    patch_array = _flat(100.0) if patch is None else np.asarray(patch, np.float32)
    record = {
        "scene_id": scene_id,
        "dataset_id": dataset_id,
        "scene_group": scene_group,
        "fold": fold,
        "frame_id": frame_id,
        "filename": f"{frame_id}.png",
        "coordinate": [0, 0, 0],
        "world_coordinate": [int(value) for value in world_coordinate],
        "slot": slot,
        "label": copy.deepcopy(STATE_STONE if state is None else state),
        "weight": 1.0,
        "area": 1.0,
        "depth": 1.0,
        "visible": 1.0,
        "facing": 1.0,
        "valid_fraction": float(valid_array.mean()),
        "coverage_mean": 1.0,
        "partial": False,
        "patch": _array_record(patch_array),
        "valid": _array_record(valid_array),
        "coverage": _array_record(coverage_array),
        "build": {"staging_token": STAGING, "world_identity": world_identity},
        "provenance": {
            "screenshot_sha256": shot,
            "frame_key": frame_key or f"{dataset_id}/{scene_id}/{frame_id}",
        },
    }
    return e4.seal_example(record)


def _artifact(examples: list[dict], *, groups: list[dict] | None = None) -> dict:
    if groups is None:
        counts: dict[str, int] = {}
        for example in examples:
            counts[example["scene_group"]] = counts.get(example["scene_group"], 0) + 1
        groups = [
            {
                "scene_group": scene_group,
                "fold": "train",
                "scene_ids": sorted(
                    {example["scene_id"] for example in examples
                     if example["scene_group"] == scene_group}),
                "example_count": count,
            }
            for scene_group, count in sorted(counts.items())
        ]
    fold_assignment = {group["scene_group"]: group["fold"] for group in groups}
    record = {
        "schema": e4.E3_SCHEMA,
        "engine_version": e4.E3_ENGINE_VERSION,
        "artifact_id": e4.E3_ARTIFACT_ID,
        "mode": "synthetic",
        "evaluation_only": True,
        "training_permitted": False,
        "t2_t6_excluded": True,
        "inputs": {},
        "admission": {},
        "sampler": {},
        "grouping": {
            "grouping_key": e4.GROUPING_KEY,
            "fold_key": e4.FOLD_KEY,
            "fold_assignment": fold_assignment,
            "groups": groups,
            "leakage_issues": [],
            "manifest": {},
        },
        "coverage": {"examples": len(examples)},
        "examples": examples,
    }
    return e4.seal_artifact(record)


def _write(tmp_path: Path, name: str, record: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return path


def _reseal(record: dict) -> dict:
    """Re-seal a modified record without double-counting its previous seal."""
    content = dict(record)
    content.pop("artifact_sha256", None)
    return e4.seal_artifact(content)


def _run(tmp_path: Path, *records: dict, minimum: int = e4.MINIMUM_SHARED_VALID_PIXELS) -> dict:
    paths = [_write(tmp_path, f"artifact-{index}.json", record)
             for index, record in enumerate(records)]
    return e4.audit_paths(paths, minimum_shared_valid_pixels=minimum)


# --------------------------------------------------------------------------- #
# Pair arithmetic
# --------------------------------------------------------------------------- #


def test_pair_metrics_raw_and_normalized_arithmetic():
    valid = np.zeros((PATCH, PATCH), dtype=bool)
    valid[0, 0] = True
    valid[0, 1] = True
    patch_a = np.zeros((PATCH, PATCH, 3), dtype=np.float32)
    patch_b = np.zeros((PATCH, PATCH, 3), dtype=np.float32)
    patch_a[0, 0] = 100.0
    patch_a[0, 1] = 200.0
    patch_b[0, 0] = 120.0
    patch_b[0, 1] = 120.0
    metrics = e4.pair_metrics(patch_a, patch_b, valid, valid, minimum_shared_valid_pixels=2)
    assert metrics.shared_valid_pixels == 2
    assert metrics.excluded_reason is None
    assert metrics.raw_mae == pytest.approx(50.0)
    # brightness_a=150, brightness_b=120 -> target 135; normalized -> [90,180] vs [135,135].
    assert metrics.normalized_mae == pytest.approx(45.0)


def test_pair_metrics_supports_negative_and_clipped_values():
    patch_a = _flat(-10.0)
    patch_b = _flat(300.0)
    valid = np.ones((PATCH, PATCH), dtype=bool)
    metrics = e4.pair_metrics(patch_a, patch_b, valid, valid)
    assert metrics.shared_valid_pixels == PATCH * PATCH
    assert metrics.raw_mae == pytest.approx(310.0)


def test_pair_metrics_refuses_insufficient_shared_pixels():
    valid_a = np.zeros((PATCH, PATCH), dtype=bool)
    valid_b = np.zeros((PATCH, PATCH), dtype=bool)
    valid_a[0, 0] = True
    valid_b[1, 1] = True
    metrics = e4.pair_metrics(_flat(10.0), _flat(20.0), valid_a, valid_b)
    assert metrics.shared_valid_pixels == 0
    assert metrics.raw_mae is None
    assert metrics.normalized_mae is None
    assert metrics.excluded_reason == "insufficient_shared_valid_pixels"


def test_pair_metrics_boundary_at_the_coverage_floor():
    valid = np.zeros((PATCH, PATCH), dtype=bool)
    valid.flat[:16] = True
    below = e4.pair_metrics(_flat(5.0), _flat(15.0), valid, valid,
                            minimum_shared_valid_pixels=17)
    at_floor = e4.pair_metrics(_flat(5.0), _flat(15.0), valid, valid,
                               minimum_shared_valid_pixels=16)
    assert below.excluded_reason == "insufficient_shared_valid_pixels"
    assert at_floor.excluded_reason is None
    assert at_floor.raw_mae == pytest.approx(10.0)


def test_pair_metrics_zero_brightness_keeps_raw_and_refuses_normalized():
    valid = np.ones((PATCH, PATCH), dtype=bool)
    metrics = e4.pair_metrics(_flat(0.0), _flat(0.0), valid, valid)
    assert metrics.raw_mae == pytest.approx(0.0)
    assert metrics.normalized_mae is None
    assert metrics.excluded_reason == "zero_brightness"


# --------------------------------------------------------------------------- #
# Duplicate image and same-frame suppression
# --------------------------------------------------------------------------- #


def test_duplicate_screenshot_is_suppressed_before_metrics(tmp_path):
    same = _flat(50.0)
    examples = [
        _example(frame_id="frame-0", shot=SHOT_1, patch=same),
        _example(frame_id="frame-1", shot=SHOT_1, patch=same),
        _example(frame_id="frame-2", shot=SHOT_2, patch=_flat(90.0)),
    ]
    payload = _run(tmp_path, _artifact(examples))
    accounting = payload["pairs"]
    assert accounting["raw_unordered_pairs"] == 3
    assert accounting["duplicate_image_pairs"] == 1
    assert accounting["same_frame_pairs"] == 0
    assert accounting["distinct_capture_pairs_raw"] == 2
    assert accounting["analyzed_distinct_capture_pairs"] == 1
    assert accounting["suppressed_duplicate_pair_equivalents"] == 1
    assert len(payload["pair_records"]) == 1
    # The surviving pair is the distinct capture versus the first representative.
    pair = payload["pair_records"][0]
    assert {pair["frame_a"], pair["frame_b"]} == {"ds/scene/frame-0", "ds/scene/frame-2"}
    assert payload["coverage"]["keys_with_distinct_captures"] == 1


def test_same_frame_duplicate_is_suppressed_and_stratified(tmp_path):
    examples = [
        _example(frame_id="frame-0", shot=SHOT_1, patch=_flat(10.0)),
        _example(frame_id="frame-0", shot=SHOT_3, patch=_flat(20.0),
                 world_coordinate=(1, 2, 3)),
    ]
    # Force the second observation onto the same frame key as the first.
    examples[1]["provenance"]["frame_key"] = examples[0]["provenance"]["frame_key"]
    examples[1] = e4.seal_example(examples[1])
    payload = _run(tmp_path, _artifact(examples))
    accounting = payload["pairs"]
    assert accounting["same_frame_pairs"] == 1
    assert accounting["duplicate_image_pairs"] == 0
    assert accounting["distinct_capture_pairs_raw"] == 0
    assert accounting["analyzed_distinct_capture_pairs"] == 0
    assert payload["pair_records"] == []
    # It is still a repeated key, just with no distinct-capture pair.
    assert payload["coverage"]["repeated_keys"] == 1
    assert payload["coverage"]["keys_with_distinct_captures"] == 0


def test_distinct_captures_are_measured_once_per_key(tmp_path):
    examples = [
        _example(frame_id="frame-0", shot=SHOT_1, patch=_flat(0.0)),
        _example(frame_id="frame-1", shot=SHOT_2, patch=_flat(16.0)),
    ]
    payload = _run(tmp_path, _artifact(examples))
    assert payload["pairs"]["analyzed_distinct_capture_pairs"] == 1
    assert payload["raw_rgb_mae"]["count"] == 1
    assert payload["raw_rgb_mae"]["median"] == pytest.approx(16.0)


# --------------------------------------------------------------------------- #
# Exact key and group isolation
# --------------------------------------------------------------------------- #


def test_exact_face_key_and_group_isolation(tmp_path):
    examples = [
        # The only repeated exact key, inside group A.
        _example(scene_group=GROUP_A, world_identity="Build A", frame_id="a0", shot=SHOT_1),
        _example(scene_group=GROUP_A, world_identity="Build A", frame_id="a1", shot=SHOT_2),
        # Same coordinate/slot/state but a different build group: never paired.
        _example(scene_group=GROUP_B, world_identity="Build B", frame_id="b0", shot=SHOT_1),
        _example(scene_group=GROUP_B, world_identity="Build B", frame_id="b1", shot=SHOT_2),
        # Same group as the repeated key but a different coordinate/slot/state.
        _example(scene_group=GROUP_A, world_identity="Build A", frame_id="a2", shot=SHOT_3,
                 world_coordinate=(9, 9, 9)),
        _example(scene_group=GROUP_A, world_identity="Build A", frame_id="a3", shot=SHOT_3,
                 slot="north"),
        _example(scene_group=GROUP_A, world_identity="Build A", frame_id="a4", shot=SHOT_3,
                 state=STATE_DIRT),
    ]
    payload = _run(tmp_path, _artifact(examples))
    assert payload["coverage"]["keys_total"] == 5
    assert payload["coverage"]["repeated_keys"] == 2
    assert payload["pairs"]["analyzed_distinct_capture_pairs"] == 2
    assert {pair["scene_group"] for pair in payload["pair_records"]} == {GROUP_A, GROUP_B}
    # No pair ever mixes two different exact keys: every paired face is the same
    # coordinate, slot and state inside one scene_group.
    for pair in payload["pair_records"]:
        assert pair["world_coordinate"] == [1, 2, 3]
        assert pair["slot"] == "up"
        assert pair["state"] == "minecraft:stone"
    assert {group["scene_group"] for group in payload["groups"]} == {GROUP_A, GROUP_B}


def test_state_property_order_normalizes_to_one_key(tmp_path):
    ordered = {"name": "minecraft:oak_stairs", "properties": [["facing", "north"], ["half", "top"]]}
    shuffled = {"name": "minecraft:oak_stairs", "properties": [["half", "top"], ["facing", "north"]]}
    examples = [
        _example(frame_id="frame-0", shot=SHOT_1, state=ordered),
        _example(frame_id="frame-1", shot=SHOT_2, state=shuffled),
    ]
    payload = _run(tmp_path, _artifact(examples))
    assert payload["coverage"]["repeated_keys"] == 1
    assert payload["pairs"]["analyzed_distinct_capture_pairs"] == 1
    assert payload["pair_records"][0]["state"] == "minecraft:oak_stairs"


# --------------------------------------------------------------------------- #
# Coverage floor and masks
# --------------------------------------------------------------------------- #


def test_empty_overlap_is_excluded_and_reported(tmp_path):
    valid_a = np.zeros((PATCH, PATCH), dtype=bool)
    valid_b = np.zeros((PATCH, PATCH), dtype=bool)
    valid_a[0, 0] = True
    valid_b[1, 1] = True
    examples = [
        _example(frame_id="frame-0", shot=SHOT_1, valid=valid_a),
        _example(frame_id="frame-1", shot=SHOT_2, valid=valid_b),
    ]
    payload = _run(tmp_path, _artifact(examples))
    assert payload["pairs"]["analyzed_distinct_capture_pairs"] == 1
    assert payload["pairs"]["excluded_insufficient_shared_valid_pixels"] == 1
    assert payload["raw_rgb_mae"]["count"] == 0
    assert payload["pair_records"][0]["excluded_reason"] == "insufficient_shared_valid_pixels"
    assert payload["pair_records"][0]["shared_valid_pixels"] == 0


def test_custom_coverage_floor_is_recorded_and_applied(tmp_path):
    valid = np.zeros((PATCH, PATCH), dtype=bool)
    valid.flat[:20] = True
    examples = [
        _example(frame_id="frame-0", shot=SHOT_1, valid=valid),
        _example(frame_id="frame-1", shot=SHOT_2, valid=valid),
    ]
    payload = _run(tmp_path, _artifact(examples), minimum=64)
    assert payload["parameters"]["minimum_shared_valid_pixels"] == 64
    assert payload["pairs"]["excluded_insufficient_shared_valid_pixels"] == 1
    assert payload["pairs"]["analyzed_distinct_capture_pairs"] == 1


# --------------------------------------------------------------------------- #
# Seal refusal
# --------------------------------------------------------------------------- #


def test_refuses_tampered_artifact_seal(tmp_path):
    record = _artifact([_example()])
    record["artifact_sha256"] = "0" * 64
    path = _write(tmp_path, "bad.json", record)
    with pytest.raises(e4.TexturePatchConsistencyError, match="artifact seal mismatch"):
        e4.load_artifact(path)


def test_refuses_tampered_example_seal(tmp_path):
    example = _example()
    example["weight"] = 2.0
    record = _artifact([example])
    path = _write(tmp_path, "bad.json", record)
    with pytest.raises(e4.TexturePatchConsistencyError, match="example seal mismatch"):
        e4.load_artifact(path)


def test_refuses_tampered_array_with_valid_example_seal(tmp_path):
    example = _example()
    replacement = _flat(7.0)
    example["patch"] = _array_record(replacement)
    example["patch"]["sha256"] = "9" * 64  # stale array digest, example seal recomputed below
    example = e4.seal_example(example)
    record = _artifact([example])
    path = _write(tmp_path, "bad.json", record)
    with pytest.raises(e4.TexturePatchConsistencyError, match="array digest mismatch"):
        e4.load_artifact(path)


def test_refuses_wrong_array_length(tmp_path):
    example = _example()
    raw = base64.b64decode(example["patch"]["base64"])[:-4]
    example["patch"]["base64"] = base64.b64encode(raw).decode("ascii")
    example = e4.seal_example(example)
    record = _artifact([example])
    path = _write(tmp_path, "bad.json", record)
    with pytest.raises(e4.TexturePatchConsistencyError, match="do not match the frozen domain"):
        e4.load_artifact(path)


def test_admission_verifies_every_seal_before_decoding_any_array(tmp_path, monkeypatch):
    """A tampered later array must refuse before the first patch is decoded.

    The first example is completely valid; the second has a stale patch digest
    under a recomputed example seal and a valid enclosing artifact seal. A
    one-pass loader would decode example 0 before discovering example 1's array
    mismatch, so asserting that ``_decode_array`` is never called proves the
    two-pass admission contract.
    """
    calls: list[str] = []
    real_decode = e4._decode_array

    def _spy(record, name):
        calls.append(name)
        return real_decode(record, name)

    monkeypatch.setattr(e4, "_decode_array", _spy)

    good = _example(frame_id="frame-0", shot=SHOT_1, patch=_flat(1.0))
    bad = _example(frame_id="frame-1", shot=SHOT_2, patch=_flat(2.0))
    bad["patch"]["sha256"] = "9" * 64  # stale array digest, resealed below
    bad = e4.seal_example(bad)
    record = _artifact([good, bad])
    path = _write(tmp_path, "bad-late-array.json", record)

    with pytest.raises(e4.TexturePatchConsistencyError, match="array digest mismatch"):
        e4.load_artifact(path)
    assert calls == []


def test_refuses_wrong_schema_and_missing_example_digests(tmp_path):
    record = _artifact([_example()])
    record["schema"] = "not-the-e3-schema"
    path = _write(tmp_path, "bad-schema.json", _reseal(record))
    with pytest.raises(e4.TexturePatchConsistencyError, match="unsupported artifact schema"):
        e4.load_artifact(path)

    record = _artifact([_example()])
    del record["examples"][0]["digests"]["example_sha256"]
    path = _write(tmp_path, "bad-digests.json", _reseal(record))
    with pytest.raises(e4.TexturePatchConsistencyError, match="unexpected digest keys"):
        e4.load_artifact(path)


def test_refuses_grouping_that_disagrees_with_examples(tmp_path):
    record = _artifact([_example()])
    record["grouping"]["groups"][0]["example_count"] = 5
    path = _write(tmp_path, "bad-grouping.json", _reseal(record))
    with pytest.raises(e4.TexturePatchConsistencyError, match="declares"):
        e4.load_artifact(path)


def test_refuses_missing_and_empty_artifacts(tmp_path):
    with pytest.raises(e4.TexturePatchConsistencyError, match="artifact is missing"):
        e4.load_artifact(tmp_path / "absent.json")
    record = _artifact([])
    path = _write(tmp_path, "empty.json", record)
    with pytest.raises(e4.TexturePatchConsistencyError, match="no examples"):
        e4.load_artifact(path)


# --------------------------------------------------------------------------- #
# Determinism and payload contract
# --------------------------------------------------------------------------- #


def test_audit_output_is_deterministic(tmp_path):
    examples = [
        _example(frame_id="frame-0", shot=SHOT_1, patch=_flat(0.0)),
        _example(frame_id="frame-1", shot=SHOT_2, patch=_flat(30.0)),
        _example(scene_group=GROUP_B, world_identity="Build B",
                 frame_id="frame-0", shot=SHOT_1, patch=_flat(12.0)),
        _example(scene_group=GROUP_B, world_identity="Build B",
                 frame_id="frame-1", shot=SHOT_2, patch=_flat(8.0)),
    ]
    record = _artifact(examples)
    first = _run(tmp_path, record)
    second = _run(tmp_path, record)
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False)


def test_payload_contract_and_strict_json(tmp_path):
    examples = [
        _example(frame_id="frame-0", shot=SHOT_1, patch=_flat(0.0)),
        _example(frame_id="frame-1", shot=SHOT_2, patch=_flat(10.0)),
    ]
    payload = _run(tmp_path, _artifact(examples))
    assert payload["schema"] == e4.SCHEMA
    assert payload["engine_version"] == e4.ENGINE_VERSION
    assert payload["evaluation_only"] is True
    assert payload["training_permitted"] is False
    assert payload["quality_claim"] == "none"
    assert payload["consistency_is_not_correctness"] is True
    assert payload["captures_split_independently"] is False
    assert payload["seal_verification"] == {
        "artifacts": 1,
        "artifact_seals_verified": 1,
        "example_seals_verified": 2,
        "array_seals_verified": 6,
        "refused": 0,
    }
    assert payload["limitations"]
    # Strict JSON: no NaN/Infinity can survive the serialization.
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_audit_requires_at_least_one_observation(tmp_path):
    with pytest.raises(e4.TexturePatchConsistencyError, match="no E3 artifacts"):
        e4.audit_paths([])


def test_repository_default_paths_are_the_four_named_artifacts():
    assert e4.DEFAULT_ARTIFACT_PATHS == (
        "exports/texture-poc-e3/fresh-c4-canal-town-2026-09-12.json",
        "exports/texture-poc-e3/fresh-c4-canal-town-framed-2026-09-12.json",
        "exports/texture-poc-e3/pilot-batch-train-station-2026-09-12.json",
        "exports/texture-poc-e3/smoke-train-station-ds20c-2026-09-12.json",
    )
