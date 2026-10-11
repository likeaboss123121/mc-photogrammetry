"""Paired synthetic profile for isolated G1E-B q/b ordering and row masks.

No capture, image, manifest, label, or evaluator input is read here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
import time
import tracemalloc
from itertools import combinations
from pathlib import Path

from benchmark_m4_g1c_selector import _match, _strong_prune_8, _worst_tie
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import _joint_plane_selector as selector
from mc_photogrammetry.reconstruction.lattice_planes import DistanceAlternative


ROOT = Path(__file__).resolve().parents[1]
LOGICAL_COUNTERS = tuple(
    name for name in selector.SearchCounters.__dataclass_fields__
    if not name.startswith(("objective_cache_", "physical_axis_cache_"))
)


def _budget(max_certificates: int = 100_000) -> selector.SearchBudget:
    return selector.SearchBudget(
        max_search_nodes=1_000_000,
        max_compatibility_checks=50_000_000,
        max_objective_rows=50_000_000,
        max_axis_solves=5_000_000,
        max_pareto_comparisons=1_000_000,
        max_candidate_certificates=max_certificates,
        max_buffered_match_rows=50_000_000,
    )


def _binary_b_tree_4():
    names = tuple(f"f{index:02d}" for index in range(4))
    base_readings, _ = _strong_prune_8()
    lookup = {(reading.frame, reading.axis): reading for reading in base_readings}
    readings = []
    for frame in names:
        for axis in range(3):
            reading = lookup[(frame, axis)]
            if axis == 0:
                alt = reading.alternatives[0]
                reading = replace(reading, alternatives=(
                    alt, DistanceAlternative(0.5, alt.distance_blocks, alt.support)))
            readings.append(reading)
    matches = []
    for left, right in (("f00", "f01"), ("f01", "f02"), ("f02", "f03")):
        for axis in range(3):
            if axis == 0:
                for left_b in (0.5, 1.0):
                    for right_b in (0.5, 1.0):
                        row = _match(left, right, axis, 0)
                        matches.append(replace(row, left_b=left_b, right_b=right_b))
            else:
                matches.append(_match(left, right, axis, 0))
    return readings, matches


def _candidate_key(candidate: selector.JointCandidate) -> tuple:
    return (
        candidate.quadrants, candidate.blocks_per_period,
        tuple(sorted(candidate.pair_groups)), candidate.axis_pair_groups,
        candidate.resultants, candidate.residuals, candidate.translations,
        candidate.axis_observations, candidate.mean_match_score, candidate.cost,
    )


def _measure(readings, matches, budget, *, constraint_first: bool):
    tracemalloc.start()
    started = time.perf_counter()
    if constraint_first:
        result = selector._select_joint_labels_constraint_first(
            readings, matches, budget=budget)
        path = "constraint_first_incremental_rows"
    else:
        result = selector.select_joint_labels(readings, matches, budget=budget)
        path = "g1e_a_natural_order_masks_cache"
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ({
        "path": path,
        "status": result.status,
        "reason": result.reason,
        "evaluated_candidates": result.counters.evaluated_candidates,
        "complete_labels": result.counters.complete_labels,
        "pruned_insufficient_axis": result.counters.pruned_insufficient_axis,
        "pareto_count": len(result.pareto),
        "certificate_sha256": result.certificate_sha256,
        "elapsed_seconds": round(elapsed, 6),
        "peak_tracemalloc_bytes": peak,
        "logical_counters": {
            name: getattr(result.counters, name) for name in LOGICAL_COUNTERS
        },
        "physical_cache_counters": {
            name: getattr(result.counters, name)
            for name in selector.SearchCounters.__dataclass_fields__
            if name.startswith(("objective_cache_", "physical_axis_cache_"))
        },
    }, result)


def _profile_case(name, readings, matches, budget):
    natural_stats, natural = _measure(readings, matches, budget, constraint_first=False)
    ordered_stats, ordered = _measure(readings, matches, budget, constraint_first=True)
    natural_candidates = tuple(map(_candidate_key, natural.pareto))
    ordered_candidates = tuple(map(_candidate_key, ordered.pareto))
    output_parity = (
        natural.status == ordered.status
        and natural.reason == ordered.reason
        and natural.certificate_sha256 == ordered.certificate_sha256
        and natural_candidates == ordered_candidates
    )
    if natural.status != "budget_exhausted" and not output_parity:
        raise SystemExit(f"complete output parity failure in {name}")
    if natural.status == "budget_exhausted":
        if (ordered.status != "budget_exhausted" or ordered.pareto
                or ordered.candidate is not None or ordered.certificate_sha256 is not None):
            raise SystemExit(f"whole-search exhaustion parity failure in {name}")

    prepared = selector._prepare(readings, matches)
    if isinstance(prepared, str):
        raise SystemExit(f"invalid synthetic fixture {name}: {prepared}")
    variable_order = [
        {"kind": kind, "key": list(key) if isinstance(key, tuple) else key}
        for kind, key in selector._constraint_first_variable_order(prepared)
    ]
    domains = dict(prepared.b_domains)
    variable_degrees = {
        f"{frame}:q": sum(frame in (row.left, row.right) for row in prepared.matches)
        for frame in prepared.frames[1:]
    }
    variable_degrees.update({
        f"{key[0]}:{key[1]}:b": prepared.reading_row_masks.get(key, 0).bit_count()
        for key, values in prepared.b_domains if len(values) > 1
    })
    return {
        "name": name,
        "frames": len({reading.frame for reading in readings}),
        "readings": len(readings),
        "canonical_match_rows": len(prepared.matches),
        "constraint_first_variable_order": variable_order,
        "constraint_first_variable_degrees": variable_degrees,
        "b_domain_sizes": {f"{key[0]}:{key[1]}": len(value)
                            for key, value in domains.items()},
        "full_output_parity": output_parity,
        "natural_order": natural_stats,
        "constraint_first": ordered_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing profile: {output_path}")
    selector_path = ROOT / "src/mc_photogrammetry/reconstruction/_joint_plane_selector.py"
    fixture_path = Path(__file__).with_name("benchmark_m4_g1c_selector.py")
    output = {
        "schema": "m4-g1e-b-synthetic-paired-profile-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {
            "selector": hashlib.sha256(selector_path.read_bytes()).hexdigest(),
            "profile_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "synthetic_fixture_source": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        },
        "static_order_rule": "(domain_size, -incident_canonical_row_count, kind_rank, canonical_key); q domain=4; singleton b and anchor q are constants",
        "mask_rule": "anchor q=0 and singleton b are applied at root; each q/b assignment only intersects its incident/current canonical row bitmask; partial possible-axis set is exact via pair-q-axis masks",
        "logical_budget_unit": "reference-equivalent full canonical row pass per partial-axis node and complete active-row extraction; mask propagation itself does not charge compatibility rows",
        "workloads": [
            _profile_case("strong-prune-8", *_strong_prune_8(), _budget()),
            _profile_case("shared-subset-tie-4", *_worst_tie(4), _budget()),
            _profile_case("binary-b-tree-4", *_binary_b_tree_4(), _budget()),
            _profile_case("tie-budget-probe-8", *_worst_tie(8), _budget(64)),
        ],
    }
    rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps({
        "output": str(output_path),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "schema": output["schema"],
        "parity": all(item["full_output_parity"] for item in output["workloads"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
