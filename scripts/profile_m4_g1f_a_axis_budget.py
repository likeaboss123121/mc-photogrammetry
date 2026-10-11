"""Profile old request versus new physical axis budgets on synthetic cases only."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
import time
import tracemalloc
from pathlib import Path

from benchmark_m4_g1c_selector import _frames, _match, _strong_prune_8, _worst_tie
from mc_photogrammetry.reconstruction import _joint_plane_selector as selector
from mc_photogrammetry.reconstruction.lattice_planes import DistanceAlternative


ROOT = Path(__file__).resolve().parents[1]


def _budget(axis_limit: int, *, max_certificates: int = 100_000):
    return selector.SearchBudget(
        max_search_nodes=1_000_000,
        max_compatibility_checks=50_000_000,
        max_objective_rows=50_000_000,
        max_axis_solves=axis_limit,
        max_pareto_comparisons=1_000_000,
        max_candidate_certificates=max_certificates,
        max_buffered_match_rows=50_000_000,
    )


def _cache_repeat_case():
    names, readings = _frames(4)
    lookup = {(reading.frame, reading.axis): reading for reading in readings}
    for frame, axis in (("f02", 0), ("f03", 2)):
        reading = lookup[(frame, axis)]
        alternative = reading.alternatives[0]
        lookup[(frame, axis)] = replace(reading, alternatives=(
            alternative,
            DistanceAlternative(0.5, alternative.distance_blocks, alternative.support),
        ))
    readings = [lookup[(frame, axis)] for frame in names for axis in range(3)]
    matches = [
        _match(left, right, axis, 0)
        for left, right in (("f00", "f01"), ("f01", "f02"))
        for axis in range(3)
    ]
    matches.extend((
        replace(_match("f02", "f03", 0, 1), left_b=0.5, right_b=0.5),
        replace(_match("f02", "f03", 0, 3), left_b=1.0, right_b=1.0),
    ))
    return readings, matches


def _shared_multi_pair_b_case(domain_size: int):
    names = ("a", "b", "c")
    values = (1.0, 0.5, 0.25, 0.125, 2.0, 4.0, 8.0)[:domain_size]
    _unused_names, base_readings = _frames(3)
    lookup = {(reading.frame, reading.axis): reading for reading in base_readings}
    readings = []
    for frame in names:
        for axis in range(3):
            source = lookup[(f"f{names.index(frame):02d}", axis)]
            reading = replace(source, frame=frame)
            if frame == "b" and axis == 0:
                base = reading.alternatives[0]
                reading = replace(reading, alternatives=tuple(
                    [base] + [DistanceAlternative(
                        value, base.distance_blocks, base.support)
                        for value in values[1:]]))
            readings.append(reading)
    matches = []
    for left, right in (("a", "b"), ("b", "c")):
        for axis in range(3):
            if axis == 0:
                for value in values:
                    left_b, right_b = ((1.0, value) if left == "a"
                                       else (value, 1.0))
                    matches.append(replace(
                        _match(left, right, axis, 0),
                        left_b=left_b, right_b=right_b,
                    ))
            else:
                matches.append(_match(left, right, axis, 0))
    return readings, matches


def _candidate_key(candidate):
    return (
        candidate.quadrants, candidate.blocks_per_period,
        tuple(sorted(candidate.pair_groups)),
        tuple((axis, tuple(sorted(groups)))
              for axis, groups in candidate.axis_pair_groups),
        candidate.resultants, candidate.residuals, candidate.translations,
        candidate.axis_observations, candidate.mean_match_score, candidate.cost,
    )


def _measure(readings, matches, budget, *, variable_order, axis_mode):
    tracemalloc.start()
    started = time.perf_counter()
    result = selector._search_bounded(
        readings, matches, budget, min_resultant=0.90, integer_tolerance=0.15,
        optimized=True, variable_order=variable_order,
        axis_budget_mode=axis_mode)
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "variable_order": variable_order,
        "axis_budget_mode": axis_mode,
        "status": result.status,
        "reason": result.reason,
        "candidate_count": len(result.pareto),
        "certificate_sha256": result.certificate_sha256,
        "elapsed_seconds": round(elapsed, 6),
        "peak_tracemalloc_bytes": peak,
        "counters": {
            name: getattr(result.counters, name)
            for name in result.counters.__dataclass_fields__
        },
        "candidate_keys": [_candidate_key(item) for item in result.pareto],
    }


def _case(name, readings, matches, budget):
    measurements = []
    for axis_mode in ("requests", "physical"):
        for variable_order in ("natural", "constraint_first"):
            measurements.append(_measure(
                readings, matches, budget,
                variable_order=variable_order, axis_mode=axis_mode))
    grouped = {(item["axis_budget_mode"], item["variable_order"]): item
               for item in measurements}
    comparisons = {}
    for mode in ("requests", "physical"):
        natural = grouped[(mode, "natural")]
        ordered = grouped[(mode, "constraint_first")]
        if natural["status"] == ordered["status"] == "budget_exhausted":
            parity = not natural["candidate_keys"] and not ordered["candidate_keys"]
        else:
            parity = (
                natural["status"] == ordered["status"]
                and natural["reason"] == ordered["reason"]
                and natural["certificate_sha256"] == ordered["certificate_sha256"]
                and natural["candidate_keys"] == ordered["candidate_keys"]
            )
        comparisons[f"{mode}_natural_vs_constraint_first"] = parity
    return {
        "name": name,
        "frames": len({reading.frame for reading in readings}),
        "readings": len(readings),
        "match_rows": len(matches),
        "numeric_axis_limit": budget.max_axis_solves,
        "b_domain_sizes": {
            f"{key[0]}:{key[1]}": len(values)
            for key, values in (selector._prepare(readings, matches).b_domains)
        },
        "natural_vs_ordered_output_parity_by_budget_mode": comparisons,
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing profile: {output_path}")
    selector_path = ROOT / "src/mc_photogrammetry/reconstruction/_joint_plane_selector.py"
    fixture_path = Path(__file__).with_name("benchmark_m4_g1c_selector.py")
    workloads = [
        _case("cache-repeat-4", *_cache_repeat_case(), _budget(189)),
        _case("shared-subset-tie-4", *_worst_tie(4), _budget(512)),
        _case("binary-b-tree-4", *_shared_multi_pair_b_case(4), _budget(10_000)),
        _case("shared-reading-b-domain-5", *_shared_multi_pair_b_case(5), _budget(10_000)),
        _case("shared-reading-b-domain-6", *_shared_multi_pair_b_case(6), _budget(10_000)),
        _case("shared-reading-b-domain-7", *_shared_multi_pair_b_case(7), _budget(10_000)),
        _case("tie-budget-probe-8", *_worst_tie(8), _budget(512, max_certificates=64)),
        _case("strong-prune-8", *_strong_prune_8(), _budget(512)),
    ]
    output = {
        "schema": "m4-g1f-a-axis-accounting-profile-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {
            "selector": hashlib.sha256(selector_path.read_bytes()).hexdigest(),
            "profile_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "synthetic_fixture_source": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        },
        "accounting": {
            "requests_mode": "same numeric max_axis_solves limit applies to axis_solve_requests; retained only to compare the former request-budget interpretation",
            "physical_mode": "same numeric max_axis_solves limit applies to unique physical cache-miss axis solves; hits add requests/reuse counters only",
            "axis_solve_requests": "per-axis objective work requested, including cached hits",
            "axis_solves": "per-axis LSQ work actually admitted/computed on cache misses",
        },
        "workloads": workloads,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps({
        "output": str(output_path),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "schema": output["schema"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
