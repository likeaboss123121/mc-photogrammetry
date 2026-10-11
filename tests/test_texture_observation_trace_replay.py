"""Focused tests for the evaluator-only E1 <-> production trace replay reader.

Every test runs on small synthetic, self-sealed E1 objects and synthetic
schema-2 traces in ``tmp_path``. No test loads a Template 2 / Template 6 corpus
frame, reads a production trace artifact, runs a benchmark or corpus evaluator,
or writes anything outside ``tmp_path``.
"""

from __future__ import annotations

import base64
import copy
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
    / "replay_texture_observation_trace.py"
)
_spec = importlib.util.spec_from_file_location(
    "replay_texture_observation_trace_under_test", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
reader = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = reader
_spec.loader.exec_module(reader)

ROLE = "template-2"
STONE = {"name": "minecraft:stone", "properties": []}
DIRT = {"name": "minecraft:dirt", "properties": []}
SAND = {"name": "minecraft:sand", "properties": []}
GRAVEL = {"name": "minecraft:gravel", "properties": []}
EMERALD = {"name": "minecraft:emerald_block", "properties": []}
STATES = [STONE, DIRT, SAND, GRAVEL]


# --------------------------------------------------------------------------- #
# Synthetic serialization helpers
# --------------------------------------------------------------------------- #


def _array_record(array: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": [int(size) for size in value.shape],
        "base64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
    }


def _patch_arrays(seed: int = 0, *, partial: bool = False):
    patch = (np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3) + seed) % 251.0
    valid = np.ones((16, 16), dtype=bool)
    coverage = np.full((16, 16), 1.0, dtype=np.float32)
    if partial:
        valid[0, 0] = False
        valid[5, 7] = False
        coverage[0, 0] = 0.0
        coverage[5, 7] = 0.25
    return patch, valid, coverage


def _digests(patch: np.ndarray, valid: np.ndarray, coverage: np.ndarray) -> dict[str, str]:
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    valid = np.ascontiguousarray(valid, dtype=bool)
    coverage = np.ascontiguousarray(coverage, dtype=np.float32)
    observed = reader.production_observed_patch(patch, valid, coverage)
    return {
        "observed_patch": reader.array_digest(observed),
        "valid": reader.array_digest(valid),
        "coverage": reader.array_digest(coverage),
    }


def _patch_record(frame_id, coordinate, slot, truth, arrays, *, partial=None, valid_fraction=None):
    patch, valid, coverage = arrays
    if partial is None:
        partial = not bool(np.asarray(valid).all())
    if valid_fraction is None:
        valid_fraction = float(np.mean(valid))
    return {
        "frame_id": frame_id,
        "coordinate": list(coordinate),
        "slot": slot,
        "truth_state": truth,
        "weight": 1.0,
        "area": 1.0,
        "depth": 1.0,
        "visible": 1.0,
        "facing": 1.0,
        "valid_fraction": valid_fraction,
        "coverage_mean": 1.0,
        "partial": partial,
        "patch": _array_record(np.asarray(patch, dtype=np.float32)),
        "valid": _array_record(np.asarray(valid, dtype=bool)),
        "coverage": _array_record(np.asarray(coverage, dtype=np.float32)),
    }


def _e1_document(patches, *, role: str = ROLE, schema: str = reader.E1_SCHEMA, evaluation_only: bool = True):
    record = {
        "schema": schema,
        "artifact_id": "texture-poc-e1-patch-labels",
        "role": role,
        "mode": reader.MODE,
        "evaluation_only": evaluation_only,
        "pin_verification": {},
        "inputs": {},
        "sampler": {},
        "denominators": {
            "all_visible_cells": 10,
            "geometry_labelable_cells": 8,
            "model_vocabulary_cells": 6,
            "scorer_usable_cells": 5,
        },
        "patches": patches,
        "scorer": {},
        "reconciliation": {},
    }
    record["artifact_sha256"] = reader.seal_record(record)
    return record


def _write(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_e1(tmp_path: Path, patches, **kwargs) -> Path:
    return _write(tmp_path / "e1.json", _e1_document(patches, **kwargs))


def _resolved(frame_id, coordinate, slot, costs, digests, *, weight=1.0, view_weight=1.0):
    return {
        "frame_id": frame_id,
        "coordinate": tuple(coordinate),
        "slot": slot,
        "source": "resolved_face",
        "costs": [float(value) for value in costs],
        "weight": float(weight),
        "view_weight": float(view_weight),
        "observed_patch": digests["observed_patch"],
        "valid": digests["valid"],
        "coverage": digests["coverage"],
        "pattern": "c" * 64,
    }


def _fallback(frame_id, coordinate, slot, costs, *, weight=1.0, view_weight=1.0):
    return {
        "frame_id": frame_id,
        "coordinate": tuple(coordinate),
        "slot": slot,
        "source": "centre_patch_fallback",
        "costs": [float(value) for value in costs],
        "weight": float(weight),
        "view_weight": float(view_weight),
        "observed_patch": None,
        "valid": None,
        "coverage": None,
        "pattern": None,
    }


def _coordinate_record(coordinate, states, *, raw_winner=None, smoothing=None, selected=None, final=None):
    top = [{"state": state, "cost": float(position)} for position, state in enumerate(states)]
    return {
        "coordinate": list(coordinate),
        "raw_winner": raw_winner,
        "top_candidates": top,
        "ranked_candidates": top,
        "candidate_count": len(top),
        "top_k_truncated": False,
        "post_smoothing_label": smoothing,
        "selected_state": selected,
        "final_state": final,
    }


def _trace_document(states, specs, coordinates, *, candidate_limit: int = 8):
    observations = []
    cost_records = []
    totals: dict[tuple[int, int, int], np.ndarray] = {}
    weights: dict[tuple[int, int, int], float] = {}
    for ordinal, spec in enumerate(specs):
        observations.append(
            {
                "ordinal": ordinal,
                "frame_id": spec["frame_id"],
                "coordinate": list(spec["coordinate"]),
                "slot": spec["slot"],
                "source": spec["source"],
                "observed_patch": spec["observed_patch"],
                "valid": spec["valid"],
                "coverage": spec["coverage"],
                "rgb": "a" * 64,
                "descriptor": "b" * 64,
                "pattern": spec["pattern"],
            }
        )
        effective = float(spec["weight"]) * float(spec["view_weight"])
        cost_records.append(
            {
                "ordinal": ordinal,
                "frame_id": spec["frame_id"],
                "coordinate": list(spec["coordinate"]),
                "slot": spec["slot"],
                "source": spec["source"],
                "palette_order": states,
                "costs": [float(value) for value in spec["costs"]],
                "best_cost": 1.0,
                "weight": float(spec["weight"]),
                "view_weight": float(spec["view_weight"]),
                "effective_weight": effective,
            }
        )
        if effective > 0.0:
            coordinate = spec["coordinate"]
            contribution = np.asarray(spec["costs"], dtype=np.float64) * effective
            if coordinate in totals:
                totals[coordinate] = totals[coordinate] + contribution
            else:
                totals[coordinate] = contribution
            weights[coordinate] = weights.get(coordinate, 0.0) + effective
    aggregation = [
        {
            "coordinate": list(coordinate),
            "total": [float(value) for value in totals[coordinate]],
            "weight": float(weights[coordinate]),
        }
        for coordinate in sorted(totals)
    ]
    return {
        "schema": reader.TRACE_SCHEMA,
        "candidate_limit": candidate_limit,
        "palette": {"size": len(states), "states": states, "families": {state["name"]: 0 for state in states}},
        "support_rounds": [],
        "final_palette": {"size": len(states), "states": states},
        "coordinates": coordinates,
        "coordinate_evidence": [],
        "observations": observations,
        "observation_costs": cost_records,
        "aggregation": aggregation,
    }


def _golden(tmp_path: Path):
    """One fully joined fixture with a partial row, an OOV cell and both-only keys."""
    full = _patch_arrays(0)
    partial = _patch_arrays(0, partial=True)
    second = _patch_arrays(3)
    third = _patch_arrays(7)
    e1_patches = [
        _patch_record("a.png", (0, 0, 0), "north", STONE, full),
        _patch_record("a.png", (1, 0, 0), "north", DIRT, partial),
        _patch_record("b.png", (2, 0, 0), "up", EMERALD, second),
        _patch_record("c.png", (4, 0, 0), "south", SAND, third),
    ]
    specs = [
        _resolved("a.png", (0, 0, 0), "north", [0.0, 5.0, 3.0, 7.0], _digests(*full)),
        _resolved("a.png", (1, 0, 0), "north", [0.0, 0.1, 0.5, 0.3], _digests(*partial)),
        _resolved("b.png", (2, 0, 0), "up", [0.0, 1.0, 2.0, 3.0], _digests(*second)),
        _resolved("a.png", (3, 0, 0), "north", [0.0, 0.5, 0.5, 0.5], _digests(*third)),
        _fallback("a.png", (5, 0, 0), "north", [0.0, 0.0, 0.0, 0.0]),
    ]
    coordinates = [
        _coordinate_record((0, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE),
        _coordinate_record((1, 0, 0), STATES, raw_winner=STONE, smoothing=DIRT, selected=DIRT, final=DIRT),
        _coordinate_record((2, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE),
        _coordinate_record((3, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE),
    ]
    e1_path = _write_e1(tmp_path, e1_patches)
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, coordinates))
    return e1_path, trace_path


# --------------------------------------------------------------------------- #
# Golden join
# --------------------------------------------------------------------------- #


def test_golden_join_reconciles_identity_denominators_and_ranks(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    summary = reader.replay(e1_path, trace_path, role=ROLE)

    join = summary["join"]
    assert join["e1_patch_keys"] == 4
    assert join["production_observation_keys"] == 5
    assert join["resolved_face_keys"] == 4
    assert join["fallback_observation_keys"] == 1
    assert join["intersection_keys"] == 3
    assert join["e1_only_keys"] == 1
    assert join["production_only_keys"] == 1
    assert join["fallback_only_keys"] == 1
    assert join["fallback_key_collisions_with_e1"] == 0
    assert join["face_patch_join_source"] == "resolved_face"
    assert join["per_frame"]["a.png"] == {
        "e1": 2, "production": 4, "resolved_face": 3, "fallback": 1,
        "intersection": 2, "e1_only": 0, "production_only": 1,
    }
    assert join["per_slot"]["north"]["e1"] == 2
    assert join["per_slot"]["north"]["production"] == 4
    assert join["per_slot"]["north"]["resolved_face"] == 3
    assert join["per_slot"]["north"]["fallback"] == 1
    assert join["per_slot"]["north"]["intersection"] == 2
    assert join["per_slot"]["up"] == {
        "e1": 1, "production": 1, "resolved_face": 1, "fallback": 0,
        "intersection": 1, "e1_only": 0, "production_only": 0,
    }
    assert join["per_slot"]["south"]["e1_only"] == 1

    digest = summary["digest_equality"]
    assert digest["all_three_observed_contract"] == {"numerator": 3, "denominator": 3, "rate": 1.0}
    assert digest["patch_observed_reconstruction"] == {"numerator": 3, "denominator": 3, "rate": 1.0}
    assert digest["valid"]["rate"] == 1.0
    assert digest["coverage"]["rate"] == 1.0
    # The partial row's raw patch cannot equal production's observed-patch digest.
    assert digest["patch_raw_vs_observed"] == {"numerator": 2, "denominator": 3, "rate": pytest.approx(2 / 3)}
    assert digest["partial_rows_reconstruction"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert digest["fully_valid_rows_reconstruction"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert digest["mismatches"]["count"] == 0

    rank = summary["truth_rank"]
    assert rank["joined_key_denominator"] == 3
    assert rank["truth_in_final_palette"] == 2
    assert rank["truth_out_of_vocabulary"] == 1
    assert rank["top1"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert rank["top8"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    rows = {tuple(row["key"]["coordinate"]): row for row in rank["examples"]["examples"]}
    assert rows[(0, 0, 0)]["truth_rank"] == 1
    assert rows[(1, 0, 0)]["truth_rank"] == 2

    cell = summary["per_cell"]
    assert cell["e1_labeled_coordinates"] == 4
    assert cell["coordinates_with_trace_record"] == 3
    assert cell["coordinates_missing_trace_record"] == 1
    assert cell["fallback_observation_rows_in_labeled_coordinates"] == 0
    assert cell["fallback_labeled_coordinates_with_evidence"] == 0
    assert cell["truth_available_in_final_palette"] == 2
    assert cell["truth_out_of_vocabulary"] == 1
    assert cell["candidate_abstentions"] == 0
    assert cell["placement_abstentions"] == 0
    assert cell["top8_membership"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert cell["raw_winner_match"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert cell["post_smoothing_match"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert cell["selected_state_match"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert cell["final_state_match"] == {"numerator": 2, "denominator": 2, "rate": 1.0}

    aggregation = summary["aggregation_replay"]
    assert aggregation["coordinates"] == 5
    assert aggregation["weights_equal"]["rate"] == 1.0
    assert aggregation["totals_equal"]["rate"] == 1.0
    assert summary["production_top_k_check"]["checked"] == {"numerator": 4, "denominator": 4, "rate": 1.0}
    assert summary["e1"]["denominators"]["all_visible_cells"] == 10


def test_summary_is_json_safe_and_round_trips(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    summary = reader.replay(e1_path, trace_path, role=ROLE)
    decoded = json.loads(json.dumps(summary, sort_keys=True, allow_nan=False))
    assert decoded == summary
    assert decoded["schema"] == reader.REPLAY_SCHEMA
    assert decoded["evaluation_only"] is True
    assert decoded["role"] == ROLE


# --------------------------------------------------------------------------- #
# Fail-closed input validation
# --------------------------------------------------------------------------- #


def test_duplicate_e1_patch_keys_are_refused(tmp_path):
    arrays = _patch_arrays(0)
    patches = [
        _patch_record("a.png", (0, 0, 0), "north", STONE, arrays),
        _patch_record("a.png", (0, 0, 0), "north", STONE, arrays),
    ]
    e1_path = _write_e1(tmp_path, patches)
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, [], []))
    with pytest.raises(reader.TraceReplayError, match="duplicate E1 patch identity"):
        reader.replay(e1_path, trace_path, role=ROLE)


def test_e1_truth_state_conflict_across_patches_is_refused(tmp_path):
    patches = [
        _patch_record("a.png", (0, 0, 0), "north", STONE, _patch_arrays(0)),
        _patch_record("a.png", (0, 0, 0), "up", DIRT, _patch_arrays(1)),
    ]
    e1_path = _write_e1(tmp_path, patches)
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, [], []))
    with pytest.raises(reader.TraceReplayError, match="disagree on the truth state"):
        reader.replay(e1_path, trace_path, role=ROLE)


def test_duplicate_production_observation_identities_are_refused(tmp_path):
    e1_path, _ = _golden(tmp_path)
    arrays = _patch_arrays(0)
    specs = [
        _resolved("a.png", (0, 0, 0), "north", [0.0, 1.0, 2.0, 3.0], _digests(*arrays)),
        _resolved("a.png", (0, 0, 0), "north", [0.0, 2.0, 2.0, 3.0], _digests(*arrays)),
    ]
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, []))
    with pytest.raises(reader.TraceReplayError, match="duplicate production observation identity"):
        reader.replay(e1_path, trace_path, role=ROLE)


def test_bad_base64_shape_dtype_and_non_finite_arrays_are_refused(tmp_path):
    arrays = _patch_arrays(0)

    bad_base64 = _patch_record("a.png", (0, 0, 0), "north", STONE, arrays)
    bad_base64["patch"]["base64"] = "!!!not-base64!!!"
    with pytest.raises(reader.TraceReplayError, match="not valid base64"):
        reader.load_e1_artifact(_write_e1(tmp_path, [bad_base64]), role=ROLE)

    bad_shape = _patch_record("a.png", (0, 0, 0), "north", STONE, arrays)
    bad_shape["patch"]["shape"] = [8, 8, 3]
    with pytest.raises(reader.TraceReplayError, match="shape must be exactly"):
        reader.load_e1_artifact(_write(tmp_path / "shape.json", _e1_document([bad_shape])), role=ROLE)

    bad_dtype = _patch_record("a.png", (0, 0, 0), "north", STONE, arrays)
    bad_dtype["coverage"]["dtype"] = "<f8"
    with pytest.raises(reader.TraceReplayError, match="dtype must be exactly"):
        reader.load_e1_artifact(_write(tmp_path / "dtype.json", _e1_document([bad_dtype])), role=ROLE)

    non_finite = _patch_record("a.png", (0, 0, 0), "north", STONE, arrays)
    poisoned = np.asarray(arrays[0], dtype=np.float32).copy()
    poisoned[0, 0, 0] = np.nan
    non_finite["patch"] = _array_record(poisoned)
    with pytest.raises(reader.TraceReplayError, match="non-finite"):
        reader.load_e1_artifact(_write(tmp_path / "nan.json", _e1_document([non_finite])), role=ROLE)


def test_tampered_e1_seal_unknown_schema_role_mismatch_and_marker_are_refused(tmp_path):
    arrays = _patch_arrays(0)
    record = _e1_document([_patch_record("a.png", (0, 0, 0), "north", STONE, arrays)])
    tampered = copy.deepcopy(record)
    tampered["patches"][0]["weight"] = 99.0
    with pytest.raises(reader.TraceReplayError, match="seal mismatch"):
        reader.load_e1_artifact(_write(tmp_path / "seal.json", tampered), role=ROLE)

    unknown = _e1_document(
        [_patch_record("a.png", (0, 0, 0), "north", STONE, arrays)], schema="something-else"
    )
    with pytest.raises(reader.TraceReplayError, match="unknown E1 schema"):
        reader.load_e1_artifact(_write(tmp_path / "schema.json", unknown), role=ROLE)

    mismatched_role = _e1_document(
        [_patch_record("a.png", (0, 0, 0), "north", STONE, arrays)], role="template-6"
    )
    with pytest.raises(reader.TraceReplayError, match="does not match requested role"):
        reader.load_e1_artifact(_write(tmp_path / "role.json", mismatched_role), role=ROLE)

    not_evaluation = _e1_document(
        [_patch_record("a.png", (0, 0, 0), "north", STONE, arrays)], evaluation_only=False
    )
    with pytest.raises(reader.TraceReplayError, match="evaluation_only"):
        reader.load_e1_artifact(_write(tmp_path / "marker.json", not_evaluation), role=ROLE)


def test_unknown_trace_schema_and_missing_keys_are_refused(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["schema"] = 1
    with pytest.raises(reader.TraceReplayError, match="unknown trace schema"):
        reader.replay(e1_path, _write(tmp_path / "v1.json", document), role=ROLE)

    document = json.loads(trace_path.read_text(encoding="utf-8"))
    del document["aggregation"]
    with pytest.raises(reader.TraceReplayError, match="missing required keys"):
        reader.load_trace(_write(tmp_path / "missing.json", document))


def test_cost_length_and_palette_order_mismatches_are_refused(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["observation_costs"][0]["costs"] = [0.0, 1.0, 2.0]
    with pytest.raises(reader.TraceReplayError, match="costs has length"):
        reader.load_trace(_write(tmp_path / "short.json", document))

    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["observation_costs"][0]["palette_order"] = [STATES[1], STATES[0], STATES[2], STATES[3]]
    with pytest.raises(reader.TraceReplayError, match="palette_order does not equal"):
        reader.load_trace(_write(tmp_path / "order.json", document))


def test_observation_and_cost_ordinal_alignment_is_enforced(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["observation_costs"].pop()
    with pytest.raises(reader.TraceReplayError, match="different lengths"):
        reader.load_trace(_write(tmp_path / "short-costs.json", document))

    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["observation_costs"][1]["coordinate"] = [9, 9, 9]
    with pytest.raises(reader.TraceReplayError, match="coordinate does not match"):
        reader.load_trace(_write(tmp_path / "misaligned.json", document))


def test_duplicate_aggregation_coordinate_is_refused(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["aggregation"].append(copy.deepcopy(document["aggregation"][0]))
    with pytest.raises(reader.TraceReplayError, match="is duplicated"):
        reader.load_trace(_write(tmp_path / "dup-agg.json", document))


def test_aggregation_tampering_is_refused(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["aggregation"][0]["total"][1] += 1.0
    with pytest.raises(reader.TraceReplayError, match="inconsistent aggregate replay"):
        reader.replay(e1_path, _write(tmp_path / "tampered.json", document), role=ROLE)


def test_aggregation_weight_tampering_is_refused(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["aggregation"][0]["weight"] += 0.5
    with pytest.raises(reader.TraceReplayError, match="inconsistent aggregate replay"):
        reader.replay(e1_path, _write(tmp_path / "weight.json", document), role=ROLE)


def test_cost_vector_without_a_zero_is_refused(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["observation_costs"][0]["costs"] = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(reader.TraceReplayError, match="has no zero entry"):
        reader.load_trace(_write(tmp_path / "nozero.json", document))


def test_non_finite_cost_is_refused(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    document["observation_costs"][0]["costs"][1] = float("nan")
    with pytest.raises(reader.TraceReplayError, match="must be finite"):
        reader.load_trace(_write(tmp_path / "nan-cost.json", document))


# --------------------------------------------------------------------------- #
# Denominator separation, OOV and ties
# --------------------------------------------------------------------------- #


def test_oov_truth_is_counted_separately_from_missing_records(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    summary = reader.replay(e1_path, trace_path, role=ROLE)
    assert summary["per_cell"]["truth_out_of_vocabulary"] == 1
    assert summary["per_cell"]["coordinates_missing_trace_record"] == 1
    assert summary["truth_rank"]["truth_out_of_vocabulary"] == 1
    # The visible-cell denominator is the E1 artifact's own, never the patch rows.
    assert summary["e1"]["denominators"]["all_visible_cells"] == 10
    assert summary["e1"]["patch_count"] == 4


def test_oov_truth_is_never_counted_as_a_top_k_or_winner_match(tmp_path):
    e1_path, trace_path = _golden(tmp_path)
    summary = reader.replay(e1_path, trace_path, role=ROLE)
    # Emerald is OOV for the (2,0,0) cell, so no per-cell metric is attributable there.
    assert summary["per_cell"]["top8_membership"]["denominator"] == 2
    assert summary["per_cell"]["raw_winner_match"]["denominator"] == 2


def test_fallback_collision_is_production_only_and_not_a_face_join(tmp_path):
    """E1 face key colliding with a fallback observation is not a joined patch.

    The fallback shares ``(frame, coordinate, slot)`` with an E1 face key and no
    resolved observation exists for it, so it must not count as a joined patch or
    a truth-rank row; it is counted separately as fallback evidence, and it still
    contributes to the production per-cell result.
    """
    arrays = _patch_arrays(0)
    e1_path = _write_e1(tmp_path, [_patch_record("a.png", (0, 0, 0), "north", STONE, arrays)])
    specs = [_fallback("a.png", (0, 0, 0), "north", [0.0, 1.0, 2.0, 3.0])]
    coordinates = [
        _coordinate_record((0, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE)
    ]
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, coordinates))
    summary = reader.replay(e1_path, trace_path, role=ROLE)

    join = summary["join"]
    assert join["e1_patch_keys"] == 1
    assert join["production_observation_keys"] == 1
    assert join["resolved_face_keys"] == 0
    assert join["fallback_observation_keys"] == 1
    assert join["intersection_keys"] == 0
    assert join["e1_only_keys"] == 1
    assert join["production_only_keys"] == 0
    assert join["fallback_only_keys"] == 0
    assert join["fallback_key_collisions_with_e1"] == 1
    assert join["per_slot"]["north"]["fallback"] == 1
    assert join["per_slot"]["north"]["intersection"] == 0

    digest = summary["digest_equality"]
    assert digest["joined_key_denominator"] == 0
    assert digest["all_three_observed_contract"] == {"numerator": 0, "denominator": 0, "rate": None}
    assert digest["mismatches"]["count"] == 0

    rank = summary["truth_rank"]
    assert rank["joined_key_denominator"] == 0
    assert rank["truth_in_final_palette"] == 0
    assert rank["top1"] == {"numerator": 0, "denominator": 0, "rate": None}
    assert rank["examples"]["count"] == 0

    # Fallback evidence remains part of the actual production cell result.
    cell = summary["per_cell"]
    assert cell["e1_labeled_coordinates"] == 1
    assert cell["coordinates_with_trace_record"] == 1
    assert cell["fallback_observation_rows_in_labeled_coordinates"] == 1
    assert cell["fallback_labeled_coordinates_with_evidence"] == 1
    assert cell["raw_winner_match"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert summary["production_top_k_check"]["checked"] == {"numerator": 1, "denominator": 1, "rate": 1.0}


def test_fallback_observation_and_coordinate_counts_are_separate(tmp_path):
    """Two fallback observations at one E1-labeled coordinate split the counts.

    The renamed field reports fallback observation rows; a second field reports
    distinct E1-labeled coordinates carrying that evidence. A single number
    cannot honestly be both, so the pair is asserted together.
    """
    arrays = _patch_arrays(0)
    e1_path = _write_e1(tmp_path, [_patch_record("a.png", (0, 0, 0), "north", STONE, arrays)])
    specs = [
        _fallback("a.png", (0, 0, 0), "north", [0.0, 1.0, 2.0, 3.0]),
        _fallback("a.png", (0, 0, 0), "up", [0.0, 2.0, 1.0, 3.0]),
    ]
    coordinates = [
        _coordinate_record((0, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE)
    ]
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, coordinates))
    summary = reader.replay(e1_path, trace_path, role=ROLE)

    join = summary["join"]
    assert join["fallback_observation_keys"] == 2
    assert join["resolved_face_keys"] == 0
    assert join["intersection_keys"] == 0
    assert join["fallback_key_collisions_with_e1"] == 1

    cell = summary["per_cell"]
    assert cell["e1_labeled_coordinates"] == 1
    assert cell["fallback_observation_rows_in_labeled_coordinates"] == 2
    assert cell["fallback_labeled_coordinates_with_evidence"] == 1


@pytest.mark.parametrize("field", ["observed_patch", "valid", "coverage", "pattern"])
def test_fallback_observation_digest_fields_must_be_null(tmp_path, field):
    """A centre_patch_fallback carrying any patch digest is refused, not scored."""
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    fallback = next(
        record for record in document["observations"] if record["source"] == "centre_patch_fallback"
    )
    assert fallback[field] is None
    fallback[field] = "d" * 64
    with pytest.raises(reader.TraceReplayError, match="must be null"):
        reader.load_trace(_write(tmp_path / f"fallback-{field}.json", document))


def test_resolved_face_observation_requires_patch_digests(tmp_path):
    """The null requirement is one-sided: a resolved face must still carry them."""
    e1_path, trace_path = _golden(tmp_path)
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    resolved = next(
        record for record in document["observations"] if record["source"] == "resolved_face"
    )
    resolved["observed_patch"] = None
    with pytest.raises(reader.TraceReplayError, match="is required for a resolved face"):
        reader.load_trace(_write(tmp_path / "resolved-null.json", document))


def test_clipped_cost_ties_rank_by_palette_position(tmp_path):
    arrays = _patch_arrays(0)
    e1_path = _write_e1(tmp_path, [_patch_record("a.png", (0, 0, 0), "north", SAND, arrays)])
    specs = [_resolved("a.png", (0, 0, 0), "north", [0.0, 0.0, 0.0, 0.0], _digests(*arrays))]
    coordinates = [_coordinate_record((0, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE)]
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, coordinates))
    summary = reader.replay(e1_path, trace_path, role=ROLE)
    assert summary["truth_rank"]["top1"]["numerator"] == 0
    assert summary["truth_rank"]["top8"]["numerator"] == 1
    row = summary["truth_rank"]["examples"]["examples"][0]
    assert row["truth_rank"] == 3  # sand is palette index 2 with every cost tied at zero


def test_production_top_k_check_matches_an_explicit_cheapest_eight(tmp_path):
    states = [{"name": f"minecraft:state_{index}", "properties": []} for index in range(10)]
    arrays = _patch_arrays(0)
    e1_path = _write_e1(tmp_path, [_patch_record("a.png", (0, 0, 0), "north", states[0], arrays)])
    costs = [float(index) for index in range(8)] + [50.0, 60.0]
    specs = [_resolved("a.png", (0, 0, 0), "north", costs, _digests(*arrays))]
    top = [{"state": state, "cost": float(index)} for index, state in enumerate(states[:8])]
    coordinates = [
        {
            "coordinate": [0, 0, 0],
            "raw_winner": states[0],
            "top_candidates": top,
            "ranked_candidates": top,
            "candidate_count": 8,
            "top_k_truncated": True,
            "post_smoothing_label": states[0],
            "selected_state": states[0],
            "final_state": states[0],
        }
    ]
    trace_path = _write(tmp_path / "trace.json", _trace_document(states, specs, coordinates))
    summary = reader.replay(e1_path, trace_path, role=ROLE)
    assert summary["production_top_k_check"]["checked"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert summary["production_top_k_check"]["disagreements"]["count"] == 0


def test_top_k_check_reports_a_disagreement_without_refusing(tmp_path):
    states = [{"name": f"minecraft:state_{index}", "properties": []} for index in range(10)]
    arrays = _patch_arrays(0)
    e1_path = _write_e1(tmp_path, [_patch_record("a.png", (0, 0, 0), "north", states[0], arrays)])
    costs = [float(index) for index in range(8)] + [50.0, 60.0]
    specs = [_resolved("a.png", (0, 0, 0), "north", costs, _digests(*arrays))]
    wrong = [{"state": state, "cost": float(index)} for index, state in enumerate(states[1:9])]
    coordinates = [
        {
            "coordinate": [0, 0, 0],
            "raw_winner": states[0],
            "top_candidates": wrong,
            "ranked_candidates": wrong,
            "candidate_count": 8,
            "top_k_truncated": True,
            "post_smoothing_label": states[0],
            "selected_state": states[0],
            "final_state": states[0],
        }
    ]
    trace_path = _write(tmp_path / "trace.json", _trace_document(states, specs, coordinates))
    summary = reader.replay(e1_path, trace_path, role=ROLE)
    assert summary["production_top_k_check"]["checked"] == {"numerator": 0, "denominator": 1, "rate": 0.0}
    assert summary["production_top_k_check"]["disagreements"]["count"] == 1


def test_placement_and_candidate_abstentions_are_counted(tmp_path):
    arrays = _patch_arrays(0)
    e1_path = _write_e1(tmp_path, [_patch_record("a.png", (0, 0, 0), "north", STONE, arrays)])
    specs = [_resolved("a.png", (0, 0, 0), "north", [0.0, 1.0, 2.0, 3.0], _digests(*arrays))]
    coordinates = [
        {
            "coordinate": [0, 0, 0],
            "raw_winner": None,
            "top_candidates": [],
            "ranked_candidates": [],
            "candidate_count": 0,
            "top_k_truncated": False,
            "final_state": None,
        }
    ]
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, coordinates))
    summary = reader.replay(e1_path, trace_path, role=ROLE)
    assert summary["per_cell"]["candidate_abstentions"] == 1
    assert summary["per_cell"]["placement_abstentions"] == 1
    assert summary["per_cell"]["final_state_match"]["denominator"] == 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_main_prints_a_json_summary_and_refuses_on_bad_input(tmp_path, capsys):
    e1_path, trace_path = _golden(tmp_path)
    assert reader.main([str(e1_path), str(trace_path), "--role", ROLE]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == reader.REPLAY_SCHEMA

    bad_trace = _write(tmp_path / "bad.json", {"schema": 1})
    assert reader.main([str(e1_path), str(bad_trace), "--role", ROLE]) == 2
    assert "refused:" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Optional E2C production patch sidecar
# --------------------------------------------------------------------------- #


def _decode_record_array(record: dict) -> np.ndarray:
    """Decode a sidecar/E1 array record from its own declared dtype/shape bytes."""
    value = np.frombuffer(
        base64.b64decode(record["base64"]), dtype=np.dtype(record["dtype"])
    )
    return value.reshape(tuple(record["shape"])).copy()


def _sidecar_row(ordinal, frame_id, coordinate, slot, arrays) -> dict:
    """One synthetic E2C resolved-face payload row for the given sampler arrays."""
    patch = np.ascontiguousarray(arrays[0], dtype=np.float32)
    valid = np.ascontiguousarray(arrays[1], dtype=bool)
    coverage = np.ascontiguousarray(arrays[2], dtype=np.float32)
    observed = reader.production_observed_patch(patch, valid, coverage)
    return {
        "ordinal": ordinal,
        "frame_id": frame_id,
        "coordinate": list(coordinate),
        "slot": slot,
        "source": "resolved_face",
        "area": 1.0,
        "depth": 1.0,
        "visible": 1.0,
        "facing": 1.0,
        "weight": 1.0,
        "raw_patch": _array_record(patch),
        "observed_patch": _array_record(observed),
        "valid": _array_record(valid),
        "coverage": _array_record(coverage),
        "digests": {
            "raw_patch": reader.array_digest(patch),
            "observed_patch": reader.array_digest(observed),
            "valid": reader.array_digest(valid),
            "coverage": reader.array_digest(coverage),
        },
    }


def _write_sidecar(
    tmp_path: Path,
    trace_path: Path,
    records,
    *,
    observation_count: int,
    fallback_count: int,
    name: str = "sidecar.json",
    **overrides,
) -> Path:
    document = {
        "schema": reader.SIDECAR_SCHEMA,
        "trace_schema": reader.TRACE_SCHEMA,
        "observation_count": observation_count,
        "resolved_face_records": len(records),
        "excluded_fallback_observations": fallback_count,
        "records": records,
        "trace_sha256": reader.sha256_file(trace_path),
        "trace_bytes": trace_path.stat().st_size,
    }
    document.update(overrides)
    path = tmp_path / name
    _rewrite_sidecar(path, document)
    return path


def _rewrite_sidecar(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sidecar_fixture(tmp_path: Path):
    """Mirror of the E2B golden join plus a fully parity-matching E2C sidecar."""
    full = _patch_arrays(0)
    partial = _patch_arrays(0, partial=True)
    second = _patch_arrays(3)
    third = _patch_arrays(7)
    e1_patches = [
        _patch_record("a.png", (0, 0, 0), "north", STONE, full),
        _patch_record("a.png", (1, 0, 0), "north", DIRT, partial),
        _patch_record("b.png", (2, 0, 0), "up", EMERALD, second),
        _patch_record("c.png", (4, 0, 0), "south", SAND, third),
    ]
    specs = [
        _resolved("a.png", (0, 0, 0), "north", [0.0, 5.0, 3.0, 7.0], _digests(*full)),
        _resolved("a.png", (1, 0, 0), "north", [0.0, 0.1, 0.5, 0.3], _digests(*partial)),
        _resolved("b.png", (2, 0, 0), "up", [0.0, 1.0, 2.0, 3.0], _digests(*second)),
        _resolved("a.png", (3, 0, 0), "north", [0.0, 0.5, 0.5, 0.5], _digests(*third)),
        _fallback("a.png", (5, 0, 0), "north", [0.0, 0.0, 0.0, 0.0]),
    ]
    coordinates = [
        _coordinate_record((0, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE),
        _coordinate_record((1, 0, 0), STATES, raw_winner=STONE, smoothing=DIRT, selected=DIRT, final=DIRT),
        _coordinate_record((2, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE),
        _coordinate_record((3, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE),
    ]
    e1_path = _write_e1(tmp_path, e1_patches)
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, coordinates))
    records = [
        _sidecar_row(0, "a.png", (0, 0, 0), "north", full),
        _sidecar_row(1, "a.png", (1, 0, 0), "north", partial),
        _sidecar_row(2, "b.png", (2, 0, 0), "up", second),
        _sidecar_row(3, "a.png", (3, 0, 0), "north", third),
    ]
    sidecar_path = _write_sidecar(
        tmp_path, trace_path, records, observation_count=5, fallback_count=1
    )
    return e1_path, trace_path, sidecar_path


def test_sidecar_golden_parity_and_denominators(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    summary = reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    assert "sidecar" in summary
    json.loads(json.dumps(summary, sort_keys=True, allow_nan=False))
    sidecar = summary["sidecar"]
    assert sidecar["schema"] == reader.SIDECAR_SCHEMA
    assert sidecar["record_count"] == 4
    assert sidecar["trace_observation_count"] == 5
    assert sidecar["trace_resolved_face_observations"] == 4
    assert sidecar["fallback_observations_excluded"] == 1
    assert sidecar["trace_sha256_bound"] == reader.sha256_file(trace_path)
    assert sidecar["trace_bytes_bound"] == trace_path.stat().st_size

    recon = sidecar["ordinal_identity_reconciliation"]
    assert recon["reconciled_rows"] == 4
    assert recon["resolved_face_trace_observations"] == 4
    assert recon["fallback_observations_excluded"] == 1

    counts = sidecar["counts"]
    assert counts["sidecar_resolved_rows"] == 4
    assert counts["trace_resolved_face_rows"] == 4
    assert counts["e1_patch_keys"] == 4
    assert counts["identity_joins"] == 3
    assert counts["e1_only_keys"] == 1
    assert counts["sidecar_only_keys"] == 1

    parity = sidecar["parity"]
    assert parity["joined_key_denominator"] == 3
    for field in ("raw_patch", "observed_patch", "valid", "coverage", "all_four"):
        assert parity[field] == {"numerator": 3, "denominator": 3, "rate": 1.0}
    assert parity["mismatches"]["count"] == 0

    # The E2B sections remain intact and are not fed by the sidecar parity.
    assert summary["join"]["intersection_keys"] == 3
    assert summary["digest_equality"]["patch_observed_reconstruction"]["rate"] == 1.0
    assert summary["e1"]["denominators"]["all_visible_cells"] == 10
    assert summary["production_top_k_check"]["checked"]["rate"] == 1.0


def test_summary_without_sidecar_is_unchanged(tmp_path):
    e1_path, trace_path, _ = _sidecar_fixture(tmp_path)
    summary = reader.replay(e1_path, trace_path, role=ROLE)
    assert "sidecar" not in summary
    assert not any("sidecar" in limitation.lower() for limitation in summary["limitations"])


@pytest.mark.parametrize("field", ["raw_patch", "observed_patch", "valid", "coverage"])
def test_sidecar_one_field_mutant_reports_per_field_pixel_parity(tmp_path, field):
    """A joined identity whose pixels differ is a reported parity failure, not a refusal."""
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    record = document["records"][0]
    mutated = _decode_record_array(record[field])
    if field == "valid":
        mutated[0, 0] = not bool(mutated[0, 0])
    elif field == "coverage":
        mutated[0, 0] = float(mutated[0, 0]) + 1.0
    else:
        mutated[0, 0, 0] = float(mutated[0, 0, 0]) + 5.0
    record[field] = _array_record(mutated)
    record["digests"][field] = reader.array_digest(mutated)
    if field in ("observed_patch", "valid", "coverage"):
        trace_document = json.loads(trace_path.read_text(encoding="utf-8"))
        trace_document["observations"][0][field] = reader.array_digest(mutated)
        _write(trace_path, trace_document)
        document["trace_sha256"] = reader.sha256_file(trace_path)
        document["trace_bytes"] = trace_path.stat().st_size
    _rewrite_sidecar(sidecar_path, document)

    summary = reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)
    parity = summary["sidecar"]["parity"]
    assert parity[field] == {"numerator": 2, "denominator": 3, "rate": pytest.approx(2 / 3)}
    for other in ("raw_patch", "observed_patch", "valid", "coverage"):
        if other != field:
            assert parity[other] == {"numerator": 3, "denominator": 3, "rate": 1.0}
    assert parity["all_four"] == {"numerator": 2, "denominator": 3, "rate": pytest.approx(2 / 3)}
    assert parity["mismatches"]["count"] == 1
    assert parity["mismatches"]["examples"][0]["matched"][field] is False
    # Identity still joins even though the pixels differ.
    assert summary["sidecar"]["counts"]["identity_joins"] == 3


def test_sidecar_excludes_fallback_and_fallback_key_collision(tmp_path):
    arrays = _patch_arrays(0)
    e1_path = _write_e1(tmp_path, [_patch_record("a.png", (0, 0, 0), "north", STONE, arrays)])
    specs = [_fallback("a.png", (0, 0, 0), "north", [0.0, 1.0, 2.0, 3.0])]
    coordinates = [
        _coordinate_record((0, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE)
    ]
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, coordinates))
    sidecar_path = _write_sidecar(
        tmp_path, trace_path, [], observation_count=1, fallback_count=1
    )

    summary = reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)
    sidecar = summary["sidecar"]
    assert sidecar["record_count"] == 0
    assert sidecar["fallback_observations_excluded"] == 1
    assert sidecar["ordinal_identity_reconciliation"]["reconciled_rows"] == 0
    counts = sidecar["counts"]
    assert counts["sidecar_resolved_rows"] == 0
    assert counts["identity_joins"] == 0
    assert counts["e1_only_keys"] == 1
    assert counts["sidecar_only_keys"] == 0
    assert sidecar["parity"]["joined_key_denominator"] == 0
    assert sidecar["parity"]["all_four"] == {"numerator": 0, "denominator": 0, "rate": None}

    # The E2B fallback-collision semantics are untouched by the sidecar.
    assert summary["join"]["fallback_key_collisions_with_e1"] == 1
    assert summary["join"]["intersection_keys"] == 0
    assert summary["per_cell"]["fallback_observation_rows_in_labeled_coordinates"] == 1


def test_sidecar_missing_resolved_row_is_refused(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"].pop(1)
    document["resolved_face_records"] = len(document["records"])
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="exactly one row per resolved face"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_sidecar_extra_row_at_a_fallback_ordinal_is_refused(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][1] = _sidecar_row(4, "a.png", (5, 0, 0), "north", _patch_arrays(9))
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="is not a resolved_face trace observation"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_sidecar_duplicate_ordinal_is_refused(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][1] = copy.deepcopy(document["records"][0])
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="duplicated in the sidecar"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_sidecar_duplicate_identity_is_refused(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    duplicate = document["records"][1]
    duplicate["frame_id"] = document["records"][0]["frame_id"]
    duplicate["coordinate"] = list(document["records"][0]["coordinate"])
    duplicate["slot"] = document["records"][0]["slot"]
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="duplicate sidecar identity"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_sidecar_identity_source_and_trace_digest_disagreements_are_refused(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)

    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][1]["frame_id"] = "z.png"
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="disagrees with the trace"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][1]["source"] = "centre_patch_fallback"
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="must be 'resolved_face'"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    # A sidecar payload whose recomputed digest disagrees with the trace at the
    # same ordinal is refused; nothing is joined.
    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    mutated = _decode_record_array(document["records"][0]["observed_patch"])
    mutated[0, 0, 0] = float(mutated[0, 0, 0]) + 7.0
    document["records"][0]["observed_patch"] = _array_record(mutated)
    document["records"][0]["digests"]["observed_patch"] = reader.array_digest(mutated)
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="disagrees with trace observation"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_sidecar_bad_digest_base64_shape_dtype_length_and_nonfinite_are_refused(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)

    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][0]["digests"]["raw_patch"] = "0" * 64
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="does not match its decoded bytes"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][0]["raw_patch"]["base64"] = "!!!not-base64!!!"
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="not valid base64"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][0]["coverage"]["shape"] = [8, 8]
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="shape must be exactly"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][0]["valid"]["dtype"] = "<f4"
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="dtype must be exactly"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["records"][0]["raw_patch"]["base64"] = base64.b64encode(b"\x00" * 8).decode("ascii")
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="byte length does not match"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    nonfinite = np.zeros((16, 16, 3), dtype=np.float32)
    nonfinite[0, 0, 0] = np.nan
    document["records"][0]["raw_patch"] = _array_record(nonfinite)
    document["records"][0]["digests"]["raw_patch"] = reader.array_digest(nonfinite)
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="non-finite"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_sidecar_bad_headers_counts_and_trace_binding_are_refused(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    cases = [
        ("observation_count", 4, "observation_count"),
        ("resolved_face_records", 5, "resolved_face_records"),
        ("excluded_fallback_observations", 0, "excluded_fallback_observations"),
        ("trace_schema", 1, "does not declare trace schema"),
        ("trace_sha256", "0" * 64, "trace digest"),
        ("trace_bytes", trace_path.stat().st_size + 1, "trace byte count"),
    ]
    for field, value, match in cases:
        _, _, sidecar_path = _sidecar_fixture(tmp_path)
        document = json.loads(sidecar_path.read_text(encoding="utf-8"))
        document[field] = value
        _rewrite_sidecar(sidecar_path, document)
        with pytest.raises(reader.TraceReplayError, match=match):
            reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_sidecar_record_and_byte_bounds_are_enforced(tmp_path, monkeypatch):
    assert reader.SIDECAR_MAX_RECORDS == 10_000
    assert reader.SIDECAR_MAX_BYTES == 64 * 1024 * 1024
    e1_path, trace_path, _ = _sidecar_fixture(tmp_path)
    oversized = _write_sidecar(
        tmp_path,
        trace_path,
        [{} for _ in range(reader.SIDECAR_MAX_RECORDS + 1)],
        observation_count=5,
        fallback_count=1,
        name="oversized.json",
    )
    with pytest.raises(reader.TraceReplayError, match="record bound"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=oversized)

    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    monkeypatch.setattr(reader, "SIDECAR_MAX_BYTES", 10)
    with pytest.raises(reader.TraceReplayError, match="byte bound"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_sidecar_schema_and_e1_seal_role_safeguards(tmp_path):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    document["schema"] = "something-else"
    _rewrite_sidecar(sidecar_path, document)
    with pytest.raises(reader.TraceReplayError, match="unknown sidecar schema"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)

    _, _, sidecar_path = _sidecar_fixture(tmp_path)
    with pytest.raises(reader.TraceReplayError, match="does not match requested role"):
        reader.replay(e1_path, trace_path, role="template-6", sidecar_path=sidecar_path)

    e1_document = json.loads(e1_path.read_text(encoding="utf-8"))
    e1_document["patches"][0]["weight"] = 99.0
    _write(e1_path, e1_document)
    with pytest.raises(reader.TraceReplayError, match="seal mismatch"):
        reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)


def test_cli_sidecar_flag_reports_and_refuses(tmp_path, capsys):
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    assert reader.main([str(e1_path), str(trace_path), "--role", ROLE]) == 0
    assert "sidecar" not in json.loads(capsys.readouterr().out)

    assert (
        reader.main(
            [str(e1_path), str(trace_path), "--role", ROLE, "--sidecar", str(sidecar_path)]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["sidecar"]["counts"]["identity_joins"] == 3
    assert printed["sidecar"]["parity"]["all_four"]["rate"] == 1.0

    bad_sidecar = _write(tmp_path / "bad-sidecar.json", {"schema": "nope"})
    assert (
        reader.main(
            [str(e1_path), str(trace_path), "--role", ROLE, "--sidecar", str(bad_sidecar)]
        )
        == 2
    )
    assert "refused:" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Optional E2F bounded numeric patch similarity
# --------------------------------------------------------------------------- #


def _numeric_arrays(patch, observed, valid, coverage) -> dict:
    return {
        "patch": np.asarray(patch, dtype=np.float32),
        "observed_patch": np.asarray(observed, dtype=np.float32),
        "valid": np.asarray(valid, dtype=bool),
        "coverage": np.asarray(coverage, dtype=np.float32),
    }


def test_numeric_patch_similarity_arithmetic_all_pixels_and_intersection():
    """Hand-computed MAE, coverage error and mask agreement on tiny arrays."""
    e1 = _numeric_arrays(
        np.zeros((2, 2, 3)),
        np.zeros((2, 2, 3)),
        [[True, True], [False, False]],
        np.zeros((2, 2)),
    )
    sidecar = _numeric_arrays(
        [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]],
        [[[2, 2, 2], [4, 4, 4]], [[6, 6, 6], [8, 8, 8]]],
        [[True, False], [True, False]],
        np.ones((2, 2)),
    )
    metrics = reader.numeric_patch_similarity(e1, sidecar, label="row")

    assert set(metrics) == {
        "raw_rgb_mae_all_pixels",
        "raw_rgb_mae_both_valid",
        "observed_rgb_mae_all_pixels",
        "observed_rgb_mae_both_valid",
        "coverage_mae",
        "valid_mask_agreement",
        "patch_pixel_count",
        "both_valid_pixel_count",
    }
    assert metrics["patch_pixel_count"] == 4
    # The valid intersection is the single (0, 0) pixel.
    assert metrics["both_valid_pixel_count"] == 1
    assert metrics["raw_rgb_mae_all_pixels"] == pytest.approx((1 + 2 + 3 + 4) / 4)
    assert metrics["raw_rgb_mae_both_valid"] == pytest.approx(1.0)
    assert metrics["observed_rgb_mae_all_pixels"] == pytest.approx((2 + 4 + 6 + 8) / 4)
    assert metrics["observed_rgb_mae_both_valid"] == pytest.approx(2.0)
    assert metrics["coverage_mae"] == pytest.approx(1.0)
    assert metrics["valid_mask_agreement"] == pytest.approx(0.5)


def test_numeric_patch_similarity_empty_intersection_is_counted_and_null():
    """Disjoint valid masks yield a null intersection MAE, not a fabricated zero."""
    e1 = _numeric_arrays(
        np.zeros((2, 2, 3)),
        np.ones((2, 2, 3)),
        [[True, True], [True, True]],
        np.zeros((2, 2)),
    )
    disjoint = _numeric_arrays(
        np.full((2, 2, 3), 5.0),
        np.full((2, 2, 3), 5.0),
        [[False, False], [False, False]],
        np.full((2, 2), 2.0),
    )
    metrics = reader.numeric_patch_similarity(e1, disjoint, label="row")
    assert metrics["both_valid_pixel_count"] == 0
    assert metrics["raw_rgb_mae_both_valid"] is None
    assert metrics["observed_rgb_mae_both_valid"] is None
    assert metrics["raw_rgb_mae_all_pixels"] == pytest.approx(5.0)
    assert metrics["observed_rgb_mae_all_pixels"] == pytest.approx(4.0)
    assert metrics["valid_mask_agreement"] == pytest.approx(0.0)

    row = dict(metrics, key={"frame_id": "a.png", "coordinate": [0, 0, 0], "slot": "north"})
    summary = reader.summarize_numeric_rows([row])
    assert summary["joined_key_denominator"] == 1
    assert summary["rows_with_empty_both_valid_intersection"] == 1
    assert summary["rows_with_shape_mismatch"] == 0
    # The reported per-row size is the measured 2x2 count, never the 16x16 default.
    assert summary["pixel_denominators"]["patch_pixels_per_row"] == 4
    assert summary["pixel_denominators"]["patch_pixels_per_row_uniform"] is True
    assert summary["pixel_denominators"]["distinct_patch_pixel_counts"] == [4]
    assert summary["pixel_denominators"]["patch_pixel_comparisons"] == 4
    assert summary["pixel_denominators"]["both_valid_pixel_comparisons"] == 0
    both = summary["per_row_metrics"]["raw_rgb_mae_both_valid"]
    assert both == {
        "row_denominator": 0,
        "min": None,
        "median": None,
        "p90": None,
        "p99": None,
        "max": None,
    }


def test_numeric_patch_similarity_refuses_shape_mismatch_missing_and_nonfinite():
    """A shape mismatch or non-finite input is refused, never averaged over."""
    base = _numeric_arrays(
        np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.ones((2, 2)), np.ones((2, 2))
    )
    with pytest.raises(reader.TraceReplayError, match="shape mismatch"):
        reader.numeric_patch_similarity(
            base, dict(base, patch=np.zeros((2, 2, 2), dtype=np.float32)), label="row"
        )
    with pytest.raises(reader.TraceReplayError, match="shape mismatch"):
        reader.numeric_patch_similarity(
            base,
            dict(base, observed_patch=np.zeros((2, 2, 1), dtype=np.float32)),
            label="row",
        )
    with pytest.raises(reader.TraceReplayError, match="shape mismatch"):
        reader.numeric_patch_similarity(
            base, dict(base, valid=np.zeros((2, 3), dtype=bool)), label="row"
        )
    with pytest.raises(reader.TraceReplayError, match="missing the 'coverage' array"):
        reader.numeric_patch_similarity(
            base, {key: value for key, value in base.items() if key != "coverage"}, label="row"
        )
    with pytest.raises(reader.TraceReplayError, match="non-finite"):
        reader.numeric_patch_similarity(
            base, dict(base, patch=np.full((2, 2, 3), np.nan, dtype=np.float32)), label="row"
        )


def test_numeric_summary_handles_mixed_row_shapes_explicitly():
    """Mixed 2x2 and 16x16 rows report a null uniform size with both counts listed."""
    small = reader.numeric_patch_similarity(
        _numeric_arrays(
            np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.ones((2, 2)), np.ones((2, 2))
        ),
        _numeric_arrays(
            np.full((2, 2, 3), 1.0),
            np.full((2, 2, 3), 2.0),
            np.ones((2, 2)),
            np.full((2, 2), 3.0),
        ),
        label="small row",
    )
    small["key"] = {"frame_id": "a.png", "coordinate": [0, 0, 0], "slot": "north"}
    large = reader.numeric_patch_similarity(
        _numeric_arrays(
            np.zeros((16, 16, 3)), np.zeros((16, 16, 3)), np.ones((16, 16)), np.ones((16, 16))
        ),
        _numeric_arrays(
            np.full((16, 16, 3), 1.0),
            np.full((16, 16, 3), 2.0),
            np.ones((16, 16)),
            np.full((16, 16), 3.0),
        ),
        label="large row",
    )
    large["key"] = {"frame_id": "b.png", "coordinate": [0, 0, 0], "slot": "north"}

    summary = reader.summarize_numeric_rows([small, large])
    denominators = summary["pixel_denominators"]
    assert summary["joined_key_denominator"] == 2
    assert summary["rows_with_shape_mismatch"] == 0
    # The rows genuinely mix shapes, so the summary refuses to invent one size.
    assert denominators["patch_pixels_per_row"] is None
    assert denominators["patch_pixels_per_row_uniform"] is False
    assert denominators["distinct_patch_pixel_counts"] == [4, 256]
    assert denominators["patch_pixel_comparisons"] == 4 + 256
    assert denominators["coverage_pixel_comparisons"] == 4 + 256
    assert denominators["valid_mask_pixel_comparisons"] == 4 + 256
    assert denominators["both_valid_pixel_comparisons"] == 4 + 256
    assert "distinct_patch_pixel_counts" in summary["shape_contract"]
    assert "16x16" in summary["shape_contract"]

    # No measured rows means no measured per-row size either.
    empty = reader.summarize_numeric_rows([])
    assert empty["pixel_denominators"]["patch_pixels_per_row"] is None
    assert empty["pixel_denominators"]["patch_pixels_per_row_uniform"] is False
    assert empty["pixel_denominators"]["distinct_patch_pixel_counts"] == []
    assert empty["pixel_denominators"]["patch_pixel_comparisons"] == 0


def test_sidecar_numeric_similarity_reports_every_row_and_quantiles(tmp_path):
    """Per-row MAE, the aggregates and the untouched digest parity section."""
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for row_index, delta in ((0, 2.0), (1, 4.0)):
        record = document["records"][row_index]
        mutated = np.ascontiguousarray(
            _decode_record_array(record["raw_patch"]) + np.float32(delta), dtype=np.float32
        )
        record["raw_patch"] = _array_record(mutated)
        record["digests"]["raw_patch"] = reader.array_digest(mutated)
    _rewrite_sidecar(sidecar_path, document)

    summary = reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)
    numeric = summary["sidecar"]["numeric_similarity"]

    assert numeric["joined_key_denominator"] == 3
    assert numeric["rows_with_shape_mismatch"] == 0
    assert numeric["rows_with_empty_both_valid_intersection"] == 0
    assert len(numeric["rows"]) == 3
    rows = {tuple(row["key"]["coordinate"]): row for row in numeric["rows"]}
    assert set(rows) == {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
    assert rows[(0, 0, 0)]["raw_rgb_mae_all_pixels"] == pytest.approx(2.0)
    assert rows[(1, 0, 0)]["raw_rgb_mae_all_pixels"] == pytest.approx(4.0)
    assert rows[(2, 0, 0)]["raw_rgb_mae_all_pixels"] == pytest.approx(0.0)
    for row in rows.values():
        assert row["observed_rgb_mae_all_pixels"] == pytest.approx(0.0)
        assert row["coverage_mae"] == pytest.approx(0.0)
        assert row["valid_mask_agreement"] == pytest.approx(1.0)
    # The partial row keeps its two invalid pixels out of the intersection.
    assert rows[(0, 0, 0)]["both_valid_pixel_count"] == 256
    assert rows[(1, 0, 0)]["both_valid_pixel_count"] == 254
    assert rows[(1, 0, 0)]["raw_rgb_mae_both_valid"] == pytest.approx(4.0)

    raw = numeric["per_row_metrics"]["raw_rgb_mae_all_pixels"]
    assert raw["row_denominator"] == 3
    assert raw["min"] == pytest.approx(0.0)
    assert raw["median"] == pytest.approx(2.0)
    assert raw["p90"] == pytest.approx(3.6)
    assert raw["p99"] == pytest.approx(3.96)
    assert raw["max"] == pytest.approx(4.0)
    assert numeric["pixel_denominators"]["patch_pixels_per_row"] == 256
    assert numeric["pixel_denominators"]["patch_pixels_per_row_uniform"] is True
    assert numeric["pixel_denominators"]["distinct_patch_pixel_counts"] == [256]
    assert numeric["pixel_denominators"]["patch_pixel_comparisons"] == 768
    assert numeric["pixel_denominators"]["coverage_pixel_comparisons"] == 768
    assert numeric["pixel_denominators"]["valid_mask_pixel_comparisons"] == 768
    assert numeric["pixel_denominators"]["both_valid_pixel_comparisons"] == 766

    # Exact digest parity is unchanged and stays the separate authority: two of
    # the three joined raw patches are now byte-different, the third still matches.
    parity = summary["sidecar"]["parity"]
    assert parity["raw_patch"] == {"numerator": 1, "denominator": 3, "rate": pytest.approx(1 / 3)}
    assert parity["observed_patch"] == {"numerator": 3, "denominator": 3, "rate": 1.0}
    assert parity["valid"] == {"numerator": 3, "denominator": 3, "rate": 1.0}
    assert parity["coverage"] == {"numerator": 3, "denominator": 3, "rate": 1.0}
    assert parity["all_four"] == {"numerator": 1, "denominator": 3, "rate": pytest.approx(1 / 3)}


def test_sidecar_numeric_similarity_counts_empty_both_valid_intersection(tmp_path):
    """An all-invalid production mask is reported as an empty intersection."""
    e1_arrays = _patch_arrays(0)
    e1_path = _write_e1(
        tmp_path, [_patch_record("a.png", (0, 0, 0), "north", STONE, e1_arrays)]
    )
    sidecar_arrays = (
        np.ascontiguousarray(e1_arrays[0], dtype=np.float32),
        np.zeros((16, 16), dtype=bool),
        np.zeros((16, 16), dtype=np.float32),
    )
    specs = [
        _resolved("a.png", (0, 0, 0), "north", [0.0, 1.0, 2.0, 3.0], _digests(*sidecar_arrays))
    ]
    coordinates = [
        _coordinate_record(
            (0, 0, 0), STATES, raw_winner=STONE, smoothing=STONE, selected=STONE, final=STONE
        )
    ]
    trace_path = _write(tmp_path / "trace.json", _trace_document(STATES, specs, coordinates))
    sidecar_path = _write_sidecar(
        tmp_path,
        trace_path,
        [_sidecar_row(0, "a.png", (0, 0, 0), "north", sidecar_arrays)],
        observation_count=1,
        fallback_count=0,
    )

    summary = reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)
    numeric = summary["sidecar"]["numeric_similarity"]
    assert numeric["joined_key_denominator"] == 1
    assert numeric["rows_with_empty_both_valid_intersection"] == 1
    row = numeric["rows"][0]
    assert row["both_valid_pixel_count"] == 0
    assert row["raw_rgb_mae_both_valid"] is None
    assert row["observed_rgb_mae_both_valid"] is None
    assert row["raw_rgb_mae_all_pixels"] == pytest.approx(0.0)
    assert row["valid_mask_agreement"] == pytest.approx(0.0)
    assert numeric["per_row_metrics"]["raw_rgb_mae_both_valid"]["row_denominator"] == 0
    assert numeric["per_row_metrics"]["raw_rgb_mae_both_valid"]["max"] is None
    assert numeric["per_row_metrics"]["raw_rgb_mae_all_pixels"]["row_denominator"] == 1
    assert numeric["pixel_denominators"]["both_valid_pixel_comparisons"] == 0
    assert numeric["pixel_denominators"]["patch_pixel_comparisons"] == 256


def test_sidecar_numeric_similarity_is_finite_json_and_nested(tmp_path):
    """The numeric section serializes finitely and never leaves the sidecar block."""
    e1_path, trace_path, sidecar_path = _sidecar_fixture(tmp_path)
    summary = reader.replay(e1_path, trace_path, role=ROLE, sidecar_path=sidecar_path)
    encoded = json.dumps(summary, sort_keys=True, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    decoded = json.loads(encoded)

    assert "numeric_similarity" not in decoded
    numeric = decoded["sidecar"]["numeric_similarity"]
    assert numeric["joined_key_denominator"] == 3
    for row in numeric["rows"]:
        for metric in reader.NUMERIC_ROW_METRICS:
            value = row[metric]
            assert value is None or np.isfinite(value)
        assert np.isfinite(row["patch_pixel_count"])
        assert np.isfinite(row["both_valid_pixel_count"])
    for metric in reader.NUMERIC_ROW_METRICS:
        distribution = numeric["per_row_metrics"][metric]
        assert distribution["row_denominator"] == 3
        for name in ("min", "median", "p90", "p99", "max"):
            assert distribution[name] is not None
            assert np.isfinite(distribution[name])
    # The numeric addition does not disturb the E2D identity join counts.
    assert decoded["sidecar"]["counts"]["identity_joins"] == 3
    assert decoded["sidecar"]["parity"]["all_four"]["rate"] == 1.0

