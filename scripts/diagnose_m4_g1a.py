"""Label-blind exact-solver diagnostic for the immutable M4 R4 freeze.

This adapter strict-loads the frozen graph inputs and reconstructs only the
serialized plane readings and matches. It does not open the listed PNG paths or
any evaluation, capture, manifest, reference, or camera metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from mc_photogrammetry.reconstruction import frozen_image_runner, lattice_planes, plane_graph


R4_ARTIFACT_ID = "7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0"
R4_INPUTS = (
    ("view-canal-town-architectural-component-bbox-00.png", 1280, 720,
     "48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c"),
    ("view-canal-town-architectural-component-bbox-01.png", 1600, 900,
     "8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696"),
    ("view-canal-town-architectural-component-bbox-02.png", 1920, 1080,
     "2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95"),
    ("view-canal-town-architectural-component-bbox-03.png", 1920, 1080,
     "e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9"),
    ("view-canal-town-architectural-component-bbox-04.png", 1280, 720,
     "8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f"),
    ("view-canal-town-architectural-component-bbox-05.png", 1600, 900,
     "6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b"),
    ("view-canal-town-architectural-component-bbox-06.png", 1600, 900,
     "53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923"),
    ("view-canal-town-architectural-component-bbox-07.png", 1280, 720,
     "fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556"),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plane_reading(record: dict[str, Any]) -> lattice_planes.PlaneReading:
    rect = record["rectification"]
    return lattice_planes.PlaneReading(
        frame=str(record["frame"]),
        axis=int(record["axis"]),
        side=float(record["side"]),
        alternatives=tuple(lattice_planes.DistanceAlternative(
            blocks_per_period=float(item["blocks_per_period"]),
            distance_blocks=float(item["distance_blocks"]),
            support=float(item["support"])) for item in record["alternatives"]),
        period_pixels=float(record["period_pixels"]),
        phase_pixels=tuple(float(value) for value in record["phase_pixels"]),
        extent=float(record["extent"]),
        samples=int(record["samples"]),
        polygon=(None if record["polygon"] is None else
                 tuple(tuple(float(value) for value in point)
                       for point in record["polygon"])),
        coverage_cells=int(record["coverage_cells"]),
        structural_share=float(record["structural_share"]),
        confidence=float(record["confidence"]),
        rectification=lattice_planes.PlaneRectification(
            axis_matrix=tuple(tuple(float(value) for value in row)
                              for row in rect["axis_matrix"]),
            focal=float(rect["focal"]),
            principal_point=tuple(float(value) for value in rect["principal_point"]),
            centre=tuple(float(value) for value in rect["centre"]),
        ),
    )


def _alternative_record(value: plane_graph.ComponentAlternative) -> dict[str, Any]:
    return {
        "quadrants": list(value.quadrants),
        "cost": float(value.cost),
        "resultants": list(value.resultants),
        "axis_observations": list(value.axis_observations),
    }


def _solution_record(solution: plane_graph.ComponentSolution) -> dict[str, Any]:
    ranked = solution.alternatives
    best = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    return {
        "reason": solution.reason,
        "frames": list(solution.frames),
        "quadrants": {key: int(value) for key, value in sorted(solution.quadrants.items())},
        "translations": {key: list(value) for key, value in sorted(solution.translations.items())},
        "axis_resultants": {str(key): float(value)
                             for key, value in sorted(solution.axis_resultants.items())},
        "axis_residuals": {str(key): float(value)
                            for key, value in sorted(solution.axis_residuals.items())},
        "selected_b": [
            {"frame": str(frame), "plane_index": int(index), "blocks_per_period": float(value)}
            for (frame, index), value in sorted(solution.blocks_per_period.items())
        ],
        "best": None if best is None else _alternative_record(best),
        "runner_up": None if runner_up is None else _alternative_record(runner_up),
        "best_runner_up_margin": (None if best is None or runner_up is None
                                  else float(runner_up.cost - best.cost)),
        "alternatives": [_alternative_record(value) for value in ranked],
        "scale_known": bool(solution.scale_known),
    }


def diagnose(freeze_path: Path) -> dict[str, Any]:
    freeze_bytes = freeze_path.read_bytes()
    document = frozen_image_runner.load_and_verify_frozen_image(freeze_path)
    if document["artifact_id"] != R4_ARTIFACT_ID or document["labels_present"] is not False:
        raise ValueError("input freeze is not the preregistered label-free R4 artifact")
    expected_inputs = [
        {"basename": name, "width": width, "height": height, "sha256": digest}
        for name, width, height, digest in R4_INPUTS
    ]
    if document["inputs"] != expected_inputs:
        raise ValueError("ordered R4 input identity differs from preregistration")

    frames = [str(frame["frame_id"]) for frame in document["frames"]]
    readings = [_plane_reading(reading)
                for frame in document["frames"] for reading in frame["readings"]]
    matches = tuple(plane_graph.PlaneMatch(**match)
                    for match in document["graph"]["matches"])
    calls = 0
    feasible = 0
    original_evaluate = plane_graph._evaluate_assignment

    def counted_evaluate(*args, **kwargs):
        nonlocal calls, feasible
        calls += 1
        result = original_evaluate(*args, **kwargs)
        feasible += result is not None
        return result

    plane_graph._evaluate_assignment = counted_evaluate
    started = time.perf_counter()
    try:
        solution = plane_graph.solve_component(
            readings, matches, integer_tolerance=0.15,
            min_resultant=0.90, ambiguous_margin=0.02)
    finally:
        elapsed = time.perf_counter() - started
        plane_graph._evaluate_assignment = original_evaluate

    assignment_count = 4 ** max(len({reading.frame for reading in readings}) - 1, 0)
    return {
        "schema_version": 1,
        "diagnostic": "M4-G1A exact solver only",
        "labels_present": False,
        "freeze_sha256": _sha256(freeze_bytes),
        "artifact_id": document["artifact_id"],
        "inputs": expected_inputs,
        "graph_input_counts": {
            "frames": len(frames), "readings": len(readings),
            "matches": len(matches), "match_refusals": len(document["graph"]["refusals"]),
        },
        "enumeration": {
            "anchor_gauge": "first sorted frame fixed at quadrant zero",
            "assignment_count": assignment_count,
            "assignment_cap": int(plane_graph.MAX_ASSIGNMENTS),
            "attempted_evaluations": calls,
            "feasible_evaluations": feasible,
        },
        "objective": {
            "integer_tolerance": 0.15,
            "min_resultant": 0.90,
            "ambiguous_margin": 0.02,
            "unresolved_weight": float(plane_graph.LAMBDA_UNRESOLVED),
            "match_score_weight": float(plane_graph.MU_MATCH_SCORE),
            "retained_alternatives": int(plane_graph.MAX_ALTERNATIVES),
        },
        "runtime_seconds": elapsed,
        "solution": _solution_record(solution),
        "saved_r4_solution_reason": document["graph"]["solution"]["reason"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("diagnostic output already exists; choose a fresh path")
    result = diagnose(args.freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "assignment_count": result["enumeration"]["assignment_count"],
        "assignment_cap": result["enumeration"]["assignment_cap"],
        "attempted_evaluations": result["enumeration"]["attempted_evaluations"],
        "feasible_evaluations": result["enumeration"]["feasible_evaluations"],
        "runtime_seconds": result["runtime_seconds"],
        "terminal_reason": result["solution"]["reason"],
        "best_runner_up_margin": result["solution"]["best_runner_up_margin"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
