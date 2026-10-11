"""Compare isolated G1E-A selector caches with the retained G1C bounded path.

Synthetic fixtures only; this profile has no capture, GT, image, or evaluator input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from pathlib import Path

from benchmark_m4_g1c_selector import _strong_prune_8, _worst_tie
from mc_photogrammetry.reconstruction import _joint_plane_selector as selector


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


def _candidate_key(candidate: selector.JointCandidate) -> tuple:
    return (
        candidate.quadrants, candidate.blocks_per_period,
        tuple(sorted(candidate.pair_groups)), candidate.axis_pair_groups,
        candidate.resultants, candidate.residuals, candidate.translations,
        candidate.axis_observations, candidate.mean_match_score, candidate.cost,
    )


def _measure(readings, matches, budget, *, optimized: bool) -> tuple[dict, selector.SelectorResult]:
    tracemalloc.start()
    started = time.perf_counter()
    if optimized:
        result = selector.select_joint_labels(readings, matches, budget=budget)
    else:
        result = selector._search_bounded(
            readings, matches, budget, min_resultant=0.90,
            integer_tolerance=0.15, optimized=False)
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ({
        "path": "row_masks_plus_exact_cache" if optimized else "legacy_canonical_scan_uncached",
        "status": result.status,
        "reason": result.reason,
        "evaluated_candidates": result.counters.evaluated_candidates,
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


def _profile_case(name, readings, matches, budget) -> dict:
    optimized_stats, optimized = _measure(readings, matches, budget, optimized=True)
    legacy_stats, legacy = _measure(readings, matches, budget, optimized=False)
    candidate_equal = tuple(map(_candidate_key, optimized.pareto)) == tuple(
        map(_candidate_key, legacy.pareto))
    logical_equal = optimized_stats["logical_counters"] == legacy_stats["logical_counters"]
    parity = (
        optimized.status == legacy.status
        and optimized.reason == legacy.reason
        and optimized.certificate_sha256 == legacy.certificate_sha256
        and candidate_equal and logical_equal
    )
    if not parity:
        raise SystemExit(f"optimized/legacy parity failure in {name}")
    return {
        "name": name,
        "frames": len({reading.frame for reading in readings}),
        "readings": len(readings),
        "canonical_match_rows": len(matches),
        "status_certificate_logical_parity": True,
        "optimized": optimized_stats,
        "legacy": legacy_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing profile: {output_path}")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_path = Path(__file__).with_name("benchmark_m4_g1c_selector.py")
    output = {
        "schema": "m4-g1e-a-synthetic-profile-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {
            "selector": hashlib.sha256(
                (ROOT / "src/mc_photogrammetry/reconstruction/_joint_plane_selector.py").read_bytes()
            ).hexdigest(),
            "profile_script": script_hash,
            "synthetic_fixture_source": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        },
        "cache_scope": "per call; exact anchored q tuple and ordered canonical row IDs",
        "logical_budget_semantics": "unchanged; physical cache counters do not consume limits",
        "workloads": [
            _profile_case("strong-prune-8", *_strong_prune_8(), _budget()),
            _profile_case("shared-subset-tie-4", *_worst_tie(4), _budget()),
            _profile_case("tie-budget-probe-8", *_worst_tie(8), _budget(64)),
        ],
    }
    rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output_path),
                      "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                      "schema": output["schema"],
                      "parity": all(item["status_certificate_logical_parity"]
                                    for item in output["workloads"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()
