"""Synthetic-only full-stage profiling for the isolated M4 G1C selector.

No capture, manifests, labels, or R4 artifacts are loaded by this script.
Run with the workspace interpreter: python scripts/benchmark_m4_g1c_selector.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from itertools import combinations
from pathlib import Path

from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import _joint_plane_selector as selector
from mc_photogrammetry.reconstruction.lattice_planes import (
    DistanceAlternative,
    PlaneReading,
    PlaneRectification,
)


RECTIFICATION = PlaneRectification(
    axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    focal=500.0,
    principal_point=(120.0, 90.0),
    centre=(0.0, 0.0),
)


def _reading(frame: str, axis: int) -> PlaneReading:
    return PlaneReading(
        frame=frame,
        axis=axis,
        side=-1.0,
        alternatives=(DistanceAlternative(1.0, float(4 + axis), 1.0),),
        period_pixels=60.0,
        phase_pixels=(0.0, 0.0),
        extent=1.0,
        samples=512,
        polygon=None,
        coverage_cells=8,
        structural_share=0.6,
        confidence=0.9,
        rectification=RECTIFICATION,
    )


def _match(left: str, right: str, axis: int, relative_q: int) -> plane_graph.PlaneMatch:
    right_axis = axis if relative_q % 2 == 0 else {0: 2, 1: 1, 2: 0}[axis]
    return plane_graph.PlaneMatch(
        left=left,
        right=right,
        left_axis=axis,
        right_axis=right_axis,
        left_b=1.0,
        right_b=1.0,
        quadrant_of_right=relative_q,
        psr=5.0,
        peak=1.0,
        second_peak=0.1,
        overlap_cells=64,
        integer_residual=None,
        score=1.0,
        left_plane=axis,
        right_plane=right_axis,
    )


def _frames(count: int):
    names = tuple(f"f{index:02d}" for index in range(count))
    readings = [_reading(frame, axis) for frame in names for axis in range(3)]
    return names, readings


def _strong_prune_8():
    names, readings = _frames(8)
    # Many locally compatible rows, but all induce only world axis 1; the
    # safe three-axis feasibility bound rejects at the root before LSQ.
    matches = [_match(left, right, 1, 0) for left, right in combinations(names, 2)]
    return readings, matches


def _worst_tie(count: int):
    names, readings = _frames(count)
    # Every frame pair has q and q+2 variants on each local axis. The anchored
    # labels preserve tied geometry/appearance while inducing incomparable
    # parity-partition pair sets.
    matches = [
        _match(left, right, axis, relative_q)
        for left, right in combinations(names, 2)
        for axis in range(3)
        for relative_q in (0, 2)
    ]
    return readings, matches


def _run(name, readings, matches, *, max_certificates=100_000):
    budget = selector.SearchBudget(
        max_search_nodes=1_000_000,
        max_compatibility_checks=50_000_000,
        max_objective_rows=50_000_000,
        max_axis_solves=5_000_000,
        max_pareto_comparisons=1_000_000,
        max_candidate_certificates=max_certificates,
        max_buffered_match_rows=50_000_000,
    )
    tracemalloc.start()
    started = time.perf_counter()
    result = selector.select_joint_labels(readings, matches, budget=budget)
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "name": name,
        "frames": len({reading.frame for reading in readings}),
        "readings": len(readings),
        "match_rows": len(matches),
        "status": result.status,
        "reason": result.reason,
        "candidate_count": result.counters.evaluated_candidates,
        "pareto_count": len(result.pareto),
        "certificate_sha256": result.certificate_sha256,
        "elapsed_seconds": round(elapsed, 6),
        "peak_tracemalloc_bytes": peak,
        "counters": {
            field: getattr(result.counters, field)
            for field in result.counters.__dataclass_fields__
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = {
        "schema": "m4-g1c-synthetic-profile-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {
            "selector": hashlib.sha256(
                (Path(__file__).resolve().parents[1]
                 / "src" / "mc_photogrammetry" / "reconstruction"
                 / "_joint_plane_selector.py").read_bytes()
            ).hexdigest(),
            "benchmark": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "selector_budget": "explicit; no default/production cap",
        "workloads": [
            _run("strong-prune-8", *_strong_prune_8()),
            _run("worst-tie-4", *_worst_tie(4)),
            _run("tie-budget-probe-8", *_worst_tie(8), max_certificates=64),
        ],
    }
    rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
