"""Post-hoc evaluator-only audit of the sealed R11 exhaustive arm A020-308.

Task DS-IMG-R11-TRUTH-ALIGNMENT-20261004.

This module is evaluation-only and read-only over the frozen artifacts and the
pinned Template 2 evaluator truth.  It never runs the solver, never edits
production, and never feeds evaluator data back into inference.

Order of operations (the audit contract):

1. verify the sealed artifact bytes, its internal seal, and the five consumed
   input pins recorded inside it;
2. rebuild the frozen A020-308 prediction (substituted plane identities,
   active match identities, arm translations, solver metrics) *before* any
   evaluator truth is opened, and cross-check it against the sealed artifact's
   own recorded outcome and active-match indices;
3. only then load the pinned Template 2 truth (reference litematic, scene,
   cameras, frozen masks) and decide whether the captured prediction carries
   enough cell-level geometry to support the one permitted evaluator-only
   alignment and a visible geometry/state score.

If it does not, the geometry and exact-state metrics are reported as
typed-unavailable with the concrete missing serialization bridge named, and no
voxel, no alignment and no score is fabricated.  Plane-level correspondence to
visible truth is measured only where it is gauge-invariant, and the number of
integer offsets that would fit the same observed planes is reported explicitly
so a non-identifiable correspondence cannot be read as a measured one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# --- frozen artifact and pin constants (Lead-recorded values) ---------------

SEALED_ARTIFACT = (
    REPO_ROOT
    / ".tmp-geom-counterfactual-r11-axes02-exhaustive-20261004"
    / "counterfactual_replay.json"
)
SEALED_ARTIFACT_FILE_SHA256 = (
    "2fe78b7abf5ac25831f84624c91a31291b59ea09023dfb06bc1d314b4555b27a"
)
SEALED_ARTIFACT_SEAL_SHA256 = (
    "c274d77ecd0475809423523e7d76c17c24d2c0319e2e089e3718781ea8c9c2ac"
)
SEALED_ARTIFACT_TASK_ID = (
    "DS-IMG-SOLVER-CONTEXT-R11-AXES02-EXHAUSTIVE-COUNTERFACTUAL-20261004"
)
FINAL_ARM_ID = "A020-308"

R10_ARTIFACT = (
    REPO_ROOT / ".tmp-geom-prefilter-capture-r10-20261004" / "observation.json"
)
R11_ARTIFACT = (
    REPO_ROOT / ".tmp-geom-solver-context-r11r2-20261004" / "observation.json"
)
BASELINE_ARTIFACT = (
    REPO_ROOT / ".tmp-geom-counterfactual-r11-20261004" / "counterfactual_replay.json"
)
C1_C5_ARTIFACT = (
    REPO_ROOT
    / ".tmp-geom-counterfactual-r11-arms-20261004"
    / "counterfactual_replay.json"
)
AXES02_ARTIFACT = (
    REPO_ROOT
    / ".tmp-geom-counterfactual-r11-axes02-20261004"
    / "counterfactual_replay.json"
)

INPUT_PINS: tuple[tuple[str, Path, str], ...] = (
    ("r10", R10_ARTIFACT,
     "BC957B684152AEEBB5C055EB12816643F91A506AB01543761D6461FEE4C624B4"),
    ("r11", R11_ARTIFACT,
     "924D388D5ED41287EB88A6B0B709B8228C2EAD8422F9957946EFC6A734F75A92"),
    ("r11_baseline", BASELINE_ARTIFACT,
     "E0E7F9C8F46FF4B97E94E5CA8F2E612C46D926D773B006AD605E5BED27A7DD5C"),
    ("r11_c1_c5", C1_C5_ARTIFACT,
     "CFC12BF5A6CBD36374175F2B0A28EE254519D119A094C15A9347A2D7583E87DC"),
    ("r11_axes02", AXES02_ARTIFACT,
     "0ED1FF3328F8059FA041F0EDB3D8ABC539E67FDE630E347539F8103D95547149"),
)

# Frozen pinned Template 2 evaluator truth (reviewed, docs/reports/evaluator-fixture-pins.json).
FIXTURE = REPO_ROOT / "templates 2"
REFERENCE = FIXTURE / "Template 2.litematic"
SCENE = FIXTURE / "scene.json"
CAMERAS = FIXTURE / "cameras.json"
MASK_DIR = REPO_ROOT / "docs" / "evaluation_masks" / "templates-2"
PINS = REPO_ROOT / "docs" / "reports" / "evaluator-fixture-pins.json"
FIXTURE_ROLE = "template-2"
REFERENCE_PIN_SHA256 = (
    "519034a6a439ab98145aa820792d5afc588b5d697ecad961c14c36cbef2f3416"
)
FRAME_PINS = {
    "2026-09-02_13.22.14.png":
        "eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46",
    "2026-09-02_13.24.24.png":
        "e2fd9fd6c83bf9005fe0973987da9e46d213a4351ec91393ce578f8e6d3961ae",
    "2026-09-02_13.24.33.png":
        "35d9debb206c2505f65d3be29183f4a4f3d2bf3f9c05644ce75570f48381d1fc",
}

SCHEMA = "mc-photogrammetry-sealed-arm-truth-alignment-audit-v1"
AUDIT_TASK_ID = "DS-IMG-R11-TRUTH-ALIGNMENT-20261004"
ALIGNMENT_TOLERANCE = 1e-6
PLANE_OFFSET_SEARCH_BOUND = 96


class AlignmentAuditError(ValueError):
    """A fail-closed refusal inside the post-hoc audit."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise AlignmentAuditError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AlignmentAuditError(message)


# ---------------------------------------------------------------------------
# 1. Artifact / pin verification (before any evaluator truth)
# ---------------------------------------------------------------------------

def verify_artifact(replay: ModuleType) -> dict[str, Any]:
    """Recompute the sealed file hash, internal seal and every input pin."""
    _check(SEALED_ARTIFACT.is_file(), f"missing sealed artifact {SEALED_ARTIFACT}")
    file_sha = sha256_file(SEALED_ARTIFACT)
    _check(
        file_sha == SEALED_ARTIFACT_FILE_SHA256,
        f"sealed artifact file hash {file_sha} != recorded "
        f"{SEALED_ARTIFACT_FILE_SHA256}",
    )
    payload = json.loads(SEALED_ARTIFACT.read_text(encoding="utf-8"))
    _check(payload.get("schema") == replay.AXES02_EXHAUSTIVE_SCHEMA,
           "sealed artifact schema does not match the harness")
    _check(payload.get("task_id") == SEALED_ARTIFACT_TASK_ID,
           "sealed artifact task id does not match")
    _check(payload.get("status") == "complete",
           f"sealed artifact status is {payload.get('status')!r}")
    recorded_seal = payload.get("seal_sha256")
    _check(recorded_seal == SEALED_ARTIFACT_SEAL_SHA256,
           f"sealed artifact records seal {recorded_seal}")
    recomputed = replay.seal_payload(
        {key: value for key, value in payload.items() if key != "seal_sha256"})
    _check(recomputed == recorded_seal,
           "the sealed artifact's internal seal does not replay")

    pins: dict[str, Any] = {}
    for name, path, expected in INPUT_PINS:
        expected = expected.upper()
        _check(path.is_file(), f"missing pinned input {name}: {path}")
        observed = sha256_file(path)
        _check(observed.upper() == expected,
               f"pinned input {name} hash {observed} != {expected}")
        record = payload.get("inputs", {}).get(name, {})
        recorded_file = str(record.get("file_sha256", "")).upper()
        _check(recorded_file == expected,
               f"sealed artifact records {name} file hash {recorded_file}")
        pins[name] = {"path": str(path.relative_to(REPO_ROOT)),
                      "file_sha256": observed}

    return {
        "artifact": str(SEALED_ARTIFACT.relative_to(REPO_ROOT)),
        "file_sha256": file_sha,
        "seal_sha256": recorded_seal,
        "seal_replays": True,
        "task_id": payload.get("task_id"),
        "status": payload.get("status"),
        "solver_calls_authorized": payload.get("solver_calls_authorized"),
        "solver_calls_executed": payload.get("solver_calls_executed"),
        "baseline_rerun": payload.get("baseline_rerun"),
        "c3_rerun": payload.get("c3_rerun"),
        "axes02_rerun": payload.get("axes02_rerun"),
        "c1_c5_rerun": payload.get("c1_c5_rerun"),
        "input_pins": pins,
    }


# ---------------------------------------------------------------------------
# 2. Rebuild the frozen A020-308 prediction from the captured record
# ---------------------------------------------------------------------------

def _active_indices(
    identities: Sequence[Sequence[Any]], assignment: Sequence[int],
) -> list[int]:
    """Repository-identical active-set rule over the 38 captured inputs."""
    quadrants = {frame: int(value) for frame, value in zip(
        ("image-0000", "image-0001", "image-0002"), assignment)}
    active: list[int] = []
    for index, identity in enumerate(identities):
        relative = (quadrants[identity[3]] - quadrants[identity[0]]) % 4
        if relative == int(identity[6]):
            active.append(index)
    return active


def extract_prediction(replay: ModuleType, sealed: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the final arm's identities from captured data alone.

    Nothing here opens a reference, camera, mask or truth file.  The recorded
    plane indices come from the R10 reading table; the canonical remap that the
    solver used is intentionally not used as an identity.
    """
    r10 = json.loads(R10_ARTIFACT.read_text(encoding="utf-8"))
    record = r10["same_axis_candidates"]
    reconstruction = replay.load_r10_reconstruction(R10_ARTIFACT)
    universe = reconstruction.universe
    readings_by_index = {row[0]: row for row in record["readings"]}
    readings_by_pair = {(row[1], row[2]): row for row in record["readings"]}

    c1_c5 = json.loads(C1_C5_ARTIFACT.read_text(encoding="utf-8"))
    c3_arm = c1_c5["arms"]["C3"]
    c3_outcome = c3_arm["outcome"]
    _check(list(c3_outcome["assignment"]) == [0, 3, 1],
           "the sealed C3 outcome assignment changed")

    c3_replacements: dict[int, Any] = {}
    for row_index, target_match, _group, _axis in replay.C3_SUBSTITUTIONS:
        match, row = replay.candidate_match(record, universe, row_index)
        c3_replacements[int(target_match)] = match
    base_c3 = replay.substitute(reconstruction.matches, c3_replacements)

    # Measured-best winners of the sealed exhaustive run (recorded in the artifact).
    selection = sealed["selection"]
    best0 = selection["best_axis0"]
    best2 = selection["best_axis2"]
    replacements = dict(c3_replacements)
    for entry, axis in ((best0, 0), (best2, 2)):
        flat_row = int(entry["candidate_flat_row"])
        target_match = int(entry["candidate_target_match"])
        _check(int(entry["world_axis"]) == axis,
               f"recorded measured-best axis-{axis} row changed world axis")
        _check(target_match in c3_replacements or target_match in range(len(base_c3)),
               "measured-best target match is out of range")
        match, _row = replay.candidate_match(record, universe, flat_row)
        replacements[target_match] = match

    treatment = replay.substitute(reconstruction.matches, replacements)
    final_arm = selection["final_arm"]
    final_outcome = final_arm["outcome"]
    _check(final_arm["arm_id"] == FINAL_ARM_ID,
           f"sealed final arm is {final_arm['arm_id']!r}")
    _check(final_arm["diagnostics"]["active_match_indices"]
           == final_outcome["active_match_indices"],
           "the sealed final arm's diagnostics disagree with its outcome")
    assignment = [int(value) for value in final_outcome["assignment"]]
    frames = [str(frame) for frame in final_outcome["exact_context"]["frames"]]
    quadrants = {frame: assignment[position]
                 for position, frame in enumerate(frames)}

    # Solver-free active-set cross-check against the sealed recorded indices.
    recomputed_active = _active_indices(record["input_identities"], assignment)
    recorded_active = [int(value)
                       for value in final_outcome["active_match_indices"]]
    _check(recomputed_active == recorded_active,
           "the captured active-set rule does not reproduce the sealed "
           f"active indices: {recomputed_active} != {recorded_active}")

    active_matches: list[dict[str, Any]] = []
    for index in recomputed_active:
        identity = record["input_identities"][index]
        left_row = readings_by_pair[(identity[0], int(identity[1]))]
        right_row = readings_by_pair[(identity[3], int(identity[4]))]
        match = treatment[index]
        for row in (left_row, right_row):
            _check(row[3] in (0, 1, 2), "an active reading has a bad axis")
        # Distances are selected by the captured b, not by an index.
        left_distance = _distance_for_b(left_row, float(identity[7]))
        right_distance = _distance_for_b(right_row, float(identity[8]))
        left_world, left_world_sign = replay.plane_graph.axis_transform(
            int(identity[2]), int(quadrants[identity[0]]))
        right_world, right_world_sign = replay.plane_graph.axis_transform(
            int(identity[5]), int(quadrants[identity[3]]))
        _check(left_world == right_world,
               "an active match's endpoints disagree on the world axis")
        active_matches.append({
            "match_index": int(index),
            "treatment_index": int(index),
            "world_axis": int(left_world),
            "world_signs": [int(left_world_sign), int(right_world_sign)],
            "left": {"frame": identity[0], "plane": int(identity[1]),
                     "axis": int(identity[2]), "side": int(left_row[4]),
                     "b": float(identity[7]), "distance_blocks": left_distance,
                     "signed_distance": float(left_world_sign)
                     * float(left_row[4]) * left_distance},
            "right": {"frame": identity[3], "plane": int(identity[4]),
                      "axis": int(identity[5]), "side": int(right_row[4]),
                      "b": float(identity[8]), "distance_blocks": right_distance,
                      "signed_distance": float(right_world_sign)
                      * float(right_row[4]) * right_distance},
            "quadrant_of_right": int(identity[6]),
            "score": float(match.score),
        })

    substitutions: list[dict[str, Any]] = []
    for entry, axis in ((best0, 0), (best2, 2)):
        flat_row = int(entry["candidate_flat_row"])
        row = record["candidates"][flat_row]
        left_row = readings_by_index[row[3]]
        right_row = readings_by_index[row[5]]
        substitutions.append({
            "arm_id": entry["arm_id"],
            "world_axis": axis,
            "flat_row": flat_row,
            "target_match": int(entry["candidate_target_match"]),
            "group": int(row[0]),
            "left": {"frame": left_row[1], "plane": int(left_row[2]),
                     "axis": int(left_row[3]), "side": int(left_row[4]),
                     "b": float(left_row[5][row[4]][0]),
                     "distance_blocks": float(left_row[5][row[4]][1])},
            "right": {"frame": right_row[1], "plane": int(right_row[2]),
                      "axis": int(right_row[3]), "side": int(right_row[4]),
                      "b": float(right_row[5][row[6]][0]),
                      "distance_blocks": float(right_row[5][row[6]][1])},
            "candidate_score": float(row[8]),
            "candidate_delta": float(row[16]),
        })

    return {
        "arm_id": FINAL_ARM_ID,
        "included_c3_rows": [int(entry[0]) for entry in replay.C3_SUBSTITUTIONS],
        "substitutions": substitutions,
        "assignment": assignment,
        "pinned": bool(final_outcome["pinned"]),
        "mode": final_outcome["mode"],
        "reason": final_outcome["reason"],
        "accepted": bool(final_outcome["accepted"]),
        "resultants": {str(k): float(v) for k, v in final_outcome["resultants"].items()},
        "residuals": {str(k): float(v) for k, v in final_outcome["residuals"].items()},
        "observations": [int(value) for value in final_outcome["observations"]],
        "translations": {frame: [float(value) for value in row]
                         for frame, row in final_outcome["translations"].items()},
        "active_match_indices": recorded_active,
        "active_match_count": int(final_outcome["active_match_count"]),
        "active_matches": active_matches,
        "exact_context": final_outcome.get("exact_context"),
    }


def _distance_for_b(row: Sequence[Any], b: float) -> float:
    """The captured distance of the alternative selected by ``b``."""
    for entry in row[5]:
        if abs(float(entry[0]) - float(b)) <= 1e-12:
            return float(entry[1])
    raise AlignmentAuditError(
        f"reading {row[0]} carries no alternative at b={b}")


# ---------------------------------------------------------------------------
# 3. Gauge-invariant plane observables (arithmetic only)
# ---------------------------------------------------------------------------

def plane_observables(
    replay: ModuleType, prediction: Mapping[str, Any],
    treatment: Sequence[Any],
) -> dict[str, Any]:
    """Gauge-invariant per-axis plane structure.

    The plane graph constrains the active faces with
    ``delta = left_face - right_face`` where
    ``face = world_sign * side * distance_blocks``, and the per-frame translation
    must reproduce that delta for every active match.  Calling the unchanged
    production ``_delta`` helper on the reconstructed matches is the authoritative
    check, and it uses only the capture: no evaluator file and no gauge anchor.
    A residual here is a capture-integrity statement, never a world-position one.
    """
    translations = {frame: [float(value) for value in row]
                    for frame, row in prediction["translations"].items()}
    readings = list(replay.build_reading_universe(
        json.loads(R10_ARTIFACT.read_text(encoding="utf-8"))[
            "same_axis_candidates"]).readings)
    read_by_index = {}
    for reading in readings:
        read_by_index[(reading.frame, reading.plane)] = reading
    rows: list[dict[str, Any]] = []
    for match, plane_match in zip(prediction["active_matches"], treatment):
        left = match["left"]
        right = match["right"]
        axis = int(match["world_axis"])
        delta = replay.plane_graph._delta(plane_match, read_by_index)
        _check(delta is not None,
               f"active match {match['match_index']} has no captured delta")
        translation_difference = (
            translations[right["frame"]][axis]
            - translations[left["frame"]][axis])
        difference = float(delta) - translation_difference
        rows.append({
            "match_index": match["match_index"],
            "world_axis": axis,
            "captured_delta_blocks": float(delta),
            "arm_translation_difference_blocks": translation_difference,
            "difference_mod_1_block": abs(difference - round(difference)),
        })
    return {
        "equation": (
            "delta = world_sign * side * distance_blocks; the arm translation "
            "must reproduce delta for every active match"),
        "translation_origin_frame": sorted(translations)[0],
        "match_invariants": rows,
        "max_difference_mod_1_block": max(
            (row["difference_mod_1_block"] for row in rows), default=0.0),
        "all_matches_consistent": all(
            row["difference_mod_1_block"] <= ALIGNMENT_TOLERANCE for row in rows),
    }


def distinct_plane_groups(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Group the active endpoint planes by world axis and lattice congruence.

    Two endpoint planes on the same world axis and the same side are the *same*
    lattice plane when their signed block distances differ by a whole number of
    blocks; two planes on opposite sides coincide only when the signed values are
    exactly equal (the world cell boundary then carries both faces).  Both tests
    are independent of the unknown per-axis gauge offset.
    """
    entries: list[dict[str, Any]] = []
    for match in prediction["active_matches"]:
        for side_name in ("left", "right"):
            endpoint = match[side_name]
            entries.append({
                "match_index": int(match["match_index"]),
                "frame": endpoint["frame"],
                "plane": int(endpoint["plane"]),
                "world_axis": int(match["world_axis"]),
                "side": int(endpoint["side"]),
                "signed_distance": float(endpoint["signed_distance"]),
            })
    groups: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for entry in entries:
        axis = entry["world_axis"]
        placed = False
        for group in groups[axis]:
            head = group["members"][0]
            if head["side"] == entry["side"]:
                difference = entry["signed_distance"] - head["signed_distance"]
                same = abs(difference - round(difference)) <= ALIGNMENT_TOLERANCE
            else:
                same = abs(entry["signed_distance"] - head["signed_distance"]) \
                    <= ALIGNMENT_TOLERANCE
            if same:
                group["members"].append(entry)
                placed = True
                break
        if not placed:
            groups[axis].append({"axis": axis, "members": [entry]})
    result: list[dict[str, Any]] = []
    for axis in sorted(groups):
        for index, group in enumerate(groups[axis]):
            result.append({
                "key": f"axis{axis}:g{index}",
                "world_axis": axis,
                "canonical_signed_distance": float(
                    group["members"][0]["signed_distance"]),
                "distinct_frames": sorted({member["frame"]
                                           for member in group["members"]}),
                "endpoints": [
                    {"frame": member["frame"], "plane": member["plane"],
                     "side": member["side"],
                     "signed_distance": member["signed_distance"],
                     "match_index": member["match_index"]}
                    for member in group["members"]],
            })
    return result


# ---------------------------------------------------------------------------
# 4. Truth side: load the pinned Template 2 evaluator truth
# ---------------------------------------------------------------------------

def load_truth(runner: ModuleType) -> dict[str, Any]:
    pins = runner.load_pins(PINS, FIXTURE_ROLE)
    _check(pins["reference_sha256"] == REFERENCE_PIN_SHA256,
           "the reviewed reference pin changed")
    _check(pins["reviewed"] is True, "the template-2 fixture role is not reviewed")
    truth = runner.load_frozen_truth(
        fixture=FIXTURE, reference=REFERENCE, scene_path=SCENE,
        cameras_path=CAMERAS, mask_dir=MASK_DIR, filter_frames=set(FRAME_PINS),
        pins=pins, require_pins=True)
    frame_digests = {row["name"]: row["sha256"] for row in truth["frames"]}
    _check(frame_digests == FRAME_PINS,
           "the frozen frame digests do not match the reviewed pins")
    _check(truth["reference_pin_matches"] is True,
           "the reference litematic does not match its reviewed pin")
    _check(truth["mask_pins_match"] is True,
           "the frozen mask domain does not match its reviewed pin")
    _check(truth["mask_pinned_domains_checked"] == 3,
           "not all three frozen mask domains were checked")
    return truth


def reference_plane_index(truth: Mapping[str, Any]) -> dict[str, Any]:
    """Visible reference planes per world axis, keyed by exact cell coordinate.

    A reference plane here is only the set of *visible* reference cells sharing
    one coordinate on one axis, i.e. exactly the surfaces the evaluator could see
    in the frozen frames.  The reference litematic is a theoretical maximum, not
    an observed world, and this is recorded as a limitation.
    """
    visible: Mapping[Any, Any] = truth["visible_reference"].visible
    reference: Mapping[Any, Any] = truth["reference"]
    per_axis: dict[int, dict[int, int]] = {0: {}, 1: {}, 2: {}}
    for coordinate in visible:
        for axis in (0, 1, 2):
            per_axis[axis][int(coordinate[axis])] = \
                per_axis[axis].get(int(coordinate[axis]), 0) + 1
    full_per_axis: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
    for coordinate in reference:
        for axis in (0, 1, 2):
            full_per_axis[axis].add(int(coordinate[axis]))
    return {
        "visible_cell_plane_counts": {
            str(axis): dict(sorted(per_axis[axis].items())) for axis in (0, 1, 2)},
        "all_reference_plane_coordinates": {
            str(axis): sorted(full_per_axis[axis]) for axis in (0, 1, 2)},
        "visible_cells": len(visible),
        "reference_cells": len(reference),
    }


def gauge_identifiability(
    groups: Sequence[Mapping[str, Any]],
    reference_planes: Mapping[str, Any],
    *,
    search_bound: int = PLANE_OFFSET_SEARCH_BOUND,
) -> dict[str, Any]:
    """Count the integer per-axis offsets that fit the observed planes.

    The captured plane identities contain no anchor, so an offset is only
    meaningful relative to the reference lattice.  For every world axis this
    counts, inside ``[-search_bound, search_bound]``, the integer offsets under
    which every distinct observed plane lands on a reference cell boundary that
    actually carries visible cells.  A large count means the correspondence is
    not identifiable; zero means no such alignment exists at all.
    """
    observed_by_axis: dict[int, set[float]] = {0: set(), 1: set(), 2: set()}
    for group in groups:
        observed_by_axis[int(group["world_axis"])].add(
            float(group["canonical_signed_distance"]))
    reference_planes_by_axis = _boundary_lookup(reference_planes)
    report: dict[str, Any] = {}
    for axis in (0, 1, 2):
        plane_keys = reference_planes_by_axis[axis]
        observed = sorted(observed_by_axis[axis])
        per_plane: list[dict[str, Any]] = []
        joint_valid: set[int] | None = None
        for value in observed:
            offsets = [offset for offset in range(-search_bound, search_bound + 1)
                       if round(value + offset) in plane_keys]
            per_plane.append({
                "signed_distance_blocks": value,
                "fractional_part": abs(value - round(value)),
                "valid_offset_count": len(offsets),
                "valid_offset_sample": offsets[:6],
            })
            candidates = set(offsets)
            joint_valid = candidates if joint_valid is None \
                else joint_valid & candidates
        valid = sorted(joint_valid or set())
        report[str(axis)] = {
            "distinct_observed_planes": len(observed),
            "reference_boundary_classes": len(plane_keys),
            "valid_offsets_in_search": len(valid),
            "search_bound": search_bound,
            "valid_offset_sample": valid[:8],
            "identifiable": len(valid) == 1,
            "per_plane": per_plane,
            "all_observed_planes_are_near_integer": all(
                row["fractional_part"] <= 0.15 for row in per_plane),
        }
    return report


def _boundary_lookup(
    reference_planes: Mapping[str, Any],
) -> dict[int, set[int]]:
    """The integer boundary classes on each axis that carry visible cells.

    A visible cell at integer coordinate ``c`` bounds the boundary class ``c``
    on its minimum side and ``c + 1`` on its maximum side.  Any observed plane
    whose gauge-shifted signed distance lands on one of those classes is
    coincident with a boundary of a visible reference cell.
    """
    visible_counts = reference_planes["visible_cell_plane_counts"]
    lookup: dict[int, set[int]] = {}
    for axis in (0, 1, 2):
        keys: set[int] = set()
        for coordinate in visible_counts[str(axis)]:
            value = int(coordinate)
            keys.add(value)
            keys.add(value + 1)
        lookup[axis] = keys
    return lookup


# ---------------------------------------------------------------------------
# 5. The evaluator bridge attempt
# ---------------------------------------------------------------------------

def evaluator_input_audit(
    prediction: Mapping[str, Any], truth: Mapping[str, Any],
) -> dict[str, Any]:
    """State exactly what the accepted evaluator needs and what is captured.

    ``best_visible_alignment`` / ``visible_metrics`` /
    ``score_aligned_geometry_coordinates`` all consume a *candidate cell map*
    keyed by integer block coordinates.  The captured arm carries plane
    identities, one signed block distance per plane and inter-frame
    translations; it carries no cell coordinates, no plane extent/polygon, no
    occupancy mask and no promoted volume.  The bridge is therefore not
    constructible from the capture, and no voxel is fabricated here.
    """
    return {
        "candidate_cell_map_present": False,
        "captured_fields": [
            "plane identities (frame, recorded plane index, local axis, side)",
            "one selected alternative per plane (b, distance_blocks, support)",
            "inter-frame translations in block units (gauge-anchored to "
            "image-0000 = [0,0,0])",
            "per-axis resultants/residuals/observations and the solver reason",
        ],
        "evaluator_inputs_required": [
            "candidate mapping[BlockCoordinate, BlockState] for "
            "best_visible_alignment/visible_metrics",
            "candidate geometry coordinates for "
            "score_aligned_geometry_coordinates",
            "a palette and ranked candidates for any exact-state row",
        ],
        "captured_cell_level_geometry": [],
        "absent_bridge_fields": [
            "per-reading plane footprint: extent/polygon/coverage_cells are not "
            "serialized (the replay harness must synthesize polygon=None, "
            "extent=0.0, samples=0, coverage_cells=0)",
            "world lattice period in blocks: the arm's frame is a unit-edge "
            "gauge and the session records scale_known=False",
            "an absolute gauge anchor tying the plane lattice to world "
            "coordinates",
            "the observed/inferred cell occupancy or promoted volume derived "
            "from the selected planes",
            "a run palette with ranked per-cell candidates (E/C spaces)",
        ],
        "visible_denominator": {
            "universe": "frozen_reference_cells",
            "minimum_pixels": truth["visible_reference"].minimum_pixels,
            "visible_cells": len(truth["visible_reference"].visible),
            "reference_cells": len(truth["reference"]),
            "source": truth["visible_reference"].source,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_audit() -> dict[str, Any]:
    replay = load_module("replay_same_axis_counterfactual",
                         SCRIPT_DIR / "replay_same_axis_counterfactual.py")
    runner = load_module("report_image_only_attribution",
                         SCRIPT_DIR / "report_image_only_attribution.py")

    artifact = verify_artifact(replay)
    frozen = extract_prediction(replay, json.loads(
        SEALED_ARTIFACT.read_text(encoding="utf-8")))

    # The prediction is now frozen; only now is evaluator truth opened.
    truth = load_truth(runner)
    planes = reference_plane_index(truth)
    observables = plane_observables(frozen)
    groups = distinct_plane_groups(frozen)
    identifiability = gauge_identifiability(groups, planes)
    bridge = evaluator_input_audit(frozen, truth)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": AUDIT_TASK_ID,
        "status": "complete",
        "artifact_verification": artifact,
        "frozen_prediction": frozen,
        "truth": {
            "fixture": "templates 2",
            "fixture_role": FIXTURE_ROLE,
            "reference_sha256": truth["reference_sha256"],
            "reference_pin_matches": truth["reference_pin_matches"],
            "frame_pins_match": True,
            "mask_pins_match": truth["mask_pins_match"],
            "mask_origin": truth["mask_origin"],
            "minimum_pixels": truth["visible_reference"].minimum_pixels,
            "visible_cells": len(truth["visible_reference"].visible),
            "reference_cells": len(truth["reference"]),
            "visible_pixel_count_min": min(
                truth["visible_reference"].visible_pixels.values()),
            "visible_pixel_count_max": max(
                truth["visible_reference"].visible_pixels.values()),
        },
        "plane_observables": observables,
        "distinct_plane_groups": groups,
        "gauge_identifiability": identifiability,
        "evaluator_bridge": bridge,
        "metrics": {
            "visible_geometry": {
                "value": None,
                "kind": "typed-unavailable",
                "reason": "no_captured_cell_geometry",
                "denominator": "all frozen visible V (>= 1 usable pixel)",
                "denominator_count": len(truth["visible_reference"].visible),
            },
            "visible_exact_state": {
                "value": None,
                "kind": "typed-unavailable",
                "reason": "no_captured_cell_geometry",
                "denominator": "all frozen visible V (>= 1 usable pixel)",
                "denominator_count": len(truth["visible_reference"].visible),
            },
            "solver_acceptance_is_accuracy": False,
        },
        "plane_axis_class_correspondence": _axis_class_summary(frozen),
        "limitations": [
            "Evaluation-only post hoc audit of one sealed counterfactual arm on "
            "one fixture; the solver was not rerun and the sealed artifact was "
            "not modified.",
            "The reference litematic is a theoretical maximum, not an observed "
            "world; visible cells are the evaluator raycast of that reference "
            "through the pinned evaluator cameras and frozen masks.",
            "Plane-level correspondence is reported only where the plane graph's "
            "integer gauge freedom does not change the answer; an absolute "
            "plane-to-world mapping is not identifiable from the capture.",
            "No candidate voxel, alignment, geometry score or exact-state score "
            "is fabricated, and solver acceptance is never treated as accuracy.",
        ],
    }
    result["evidence_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def _axis_class_summary(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Coarse, gauge-free statement: one world axis class per active match."""
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for match in prediction["active_matches"]:
        counts[int(match["world_axis"])] += 1
    return {
        "description": (
            "every active match's two endpoints share one world-axis class; the "
            "axis label itself is invariant under the plane graph's per-axis "
            "integer gauge"),
        "active_matches_per_world_axis": {str(k): v for k, v in counts.items()},
        "axis_coverage_complete": all(counts[axis] > 0 for axis in (0, 1, 2)),
        "all_endpoints_axis_consistent": all(
            int(match["world_axis"]) in (0, 1, 2)
            for match in prediction["active_matches"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None,
                        help="write the sealed JSON evidence to this path")
    args = parser.parse_args(argv)
    try:
        result = run_audit()
    except (AlignmentAuditError, ValueError, OSError) as error:
        print(json.dumps({"status": "refused", "reason": type(error).__name__,
                          "detail": str(error)}, indent=2))
        return 2
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        if output.exists():
            print(json.dumps({"status": "refused",
                              "reason": "output_exists",
                              "detail": str(output)}, indent=2))
            return 2
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
