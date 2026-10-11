"""One preregistered selector-only G1F-B diagnostic call on frozen R4 records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import _joint_plane_selector as selector
from mc_photogrammetry.reconstruction.lattice_planes import reading_order_key
from probe_m4_g1d_r4_selector import (
    _finite_records,
    _reading,
    canonical_artifact_id,
    sha256_file,
    strict_json,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "exports/evaluation/m4-g1f-b-r4-diagnostics-probe.json"
PREREG = ROOT / "docs/reports/M4-G1F-B-selector-diagnostics-prereg-2026-09-13.md"
SELECTOR_PATH = ROOT / "src/mc_photogrammetry/reconstruction/_joint_plane_selector.py"
TEST_PATH = ROOT / "tests/test_joint_plane_selector.py"
PARSER_PATH = ROOT / "scripts/probe_m4_g1d_r4_selector.py"
G1FA_OUTPUT = ROOT / "exports/evaluation/m4-g1f-a-r4-axis-accounting-probe.json"
G1D_OUTPUT = ROOT / "exports/evaluation/m4-g1d-r4-selector-probe.json"
G1EB_OUTPUT = ROOT / "exports/evaluation/m4-g1e-b-r4-selector-probe.json"
FREEZE = ROOT / "exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/freeze.json"
SIDECAR = ROOT / (
    "exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/"
    "freeze.json.f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac.masks.bin"
)
EXPECTED = {
    "selector": "ff33bf73cea5efa690b4e2a171df3f08c7dc9a82036542893bd9baaf0e420d3b",
    "tests": "a09215ef188edf156ade78e090a783041411dcda79fad0360489e572308aa627",
    "parser": "917f5d51bffab1dec7fa65a120ab7769ba67c68ea70cf0e761fc8a7c02ec9dee",
    "g1fa": "8aea1e7cccb2e979744a611f0c7c6685960cc9fd236b741c1c3066f2b543b48d",
    "g1d": "2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c",
    "g1eb": "a50885aa64277054ba05fe618eeb1d27eadedd150bd403abdd87e58e925bc9e3",
    "freeze": "96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d",
    "sidecar": "f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac",
}
ARTIFACT_ID = "7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0"
COUNTS = {"frames": 8, "readings": 33, "matches": 200}
BUDGET = selector.SearchBudget(
    max_search_nodes=4_096,
    max_compatibility_checks=1_000_000,
    max_objective_rows=20_000,
    max_axis_solves=512,
    max_pareto_comparisons=64,
    max_candidate_certificates=16,
    max_buffered_match_rows=1_024,
)


def _check(path: Path, expected: str, name: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{name} hash mismatch: expected {expected}, got {actual}")


def _load_records(document: dict):
    readings = []
    by_frame = {}
    for frame in document["frames"]:
        frame_id = frame["frame_id"]
        frame_readings = []
        for item in frame["readings"]:
            if item["frame"] != frame_id:
                raise SystemExit("serialized reading/frame identity mismatch")
            frame_readings.append(_reading(item))
        readings.extend(frame_readings)
        by_frame[frame_id] = frame_readings
    matches = [plane_graph.PlaneMatch(**item) for item in document["graph"]["matches"]]
    canonical = {
        (frame_id, index): reading
        for frame_id, frame_readings in by_frame.items()
        for index, reading in enumerate(sorted(frame_readings, key=reading_order_key))
    }
    for match in matches:
        try:
            left = canonical[(match.left, match.left_plane)]
            right = canonical[(match.right, match.right_plane)]
        except KeyError as exc:
            raise SystemExit("match references missing canonical reading index") from exc
        if match.left_axis != left.axis or match.right_axis != right.axis:
            raise SystemExit("match axis conflicts with canonical reading index")
    _finite_records(readings, matches)
    counts = {"frames": len(by_frame), "readings": len(readings), "matches": len(matches)}
    if counts != COUNTS:
        raise SystemExit(f"R4 selector input counts differ: {counts}")
    return readings, matches, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    args = parser.parse_args()
    expected_prereg = args.expected_prereg_sha256
    expected_probe = args.expected_probe_sha256

    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if OUTPUT.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite probe output/temp: {OUTPUT}")

    pinned = (
        (PREREG, expected_prereg, "preregistration"),
        (SELECTOR_PATH, EXPECTED["selector"], "selector source"),
        (TEST_PATH, EXPECTED["tests"], "selector tests"),
        (PARSER_PATH, EXPECTED["parser"], "frozen-record parser"),
        (G1FA_OUTPUT, EXPECTED["g1fa"], "G1F-A same-budget baseline"),
        (G1D_OUTPUT, EXPECTED["g1d"], "G1D baseline"),
        (G1EB_OUTPUT, EXPECTED["g1eb"], "G1E-B baseline"),
        (FREEZE, EXPECTED["freeze"], "R4 freeze"),
        (SIDECAR, EXPECTED["sidecar"], "R4 mask-sidecar digest"),
    )
    for path, expected, name in pinned:
        _check(path, expected, name)
    _check(Path(__file__).resolve(), expected_probe, "probe script")

    frozen = strict_json(FREEZE.read_bytes())
    if (frozen.get("labels_present") is not False
            or frozen.get("schema_version") != 1
            or frozen.get("artifact_id") != ARTIFACT_ID
            or frozen.get("artifact_id") != canonical_artifact_id(frozen)):
        raise SystemExit("R4 freeze identity/schema mismatch")
    sidecar = frozen.get("mask_sidecar", {})
    if (sidecar.get("sha256") != EXPECTED["sidecar"]
            or sidecar.get("file") != SIDECAR.name
            or sidecar.get("size_bytes") != SIDECAR.stat().st_size):
        raise SystemExit("R4 sidecar descriptor mismatch")

    readings, matches, counts = _load_records(frozen)
    g1fa = strict_json(G1FA_OUTPUT.read_bytes())
    g1d = strict_json(G1D_OUTPUT.read_bytes())
    g1eb = strict_json(G1EB_OUTPUT.read_bytes())
    budget = {name: getattr(BUDGET, name) for name in BUDGET.__dataclass_fields__}
    for baseline, schema in (
        (g1fa, "m4-g1f-a-r4-axis-accounting-probe-v1"),
        (g1d, "m4-g1d-r4-selector-probe-v1"),
        (g1eb, "m4-g1e-b-r4-selector-probe-v1"),
    ):
        if (baseline.get("schema") != schema
                or baseline.get("budget") != budget
                or baseline.get("r4_freeze_sha256") != EXPECTED["freeze"]
                or baseline.get("sidecar_sha256") != EXPECTED["sidecar"]):
            raise SystemExit("prior result is not a pinned same-budget R4 selector call")

    # Exactly one authorized selector invocation, on serialized records only.
    started = time.perf_counter()
    result = selector.select_joint_labels(readings, matches, budget=BUDGET)
    elapsed = time.perf_counter() - started
    counters = {name: getattr(result.counters, name)
                for name in result.counters.__dataclass_fields__}
    legacy_counters = {name: value for name, value in counters.items()
                       if not name.startswith("diagnostic_")}
    if legacy_counters != g1fa["counters"]:
        raise SystemExit("output-neutrality failed: pre-G1F-B counters differ from G1F-A")
    if (result.status != g1fa["status"] or result.reason != g1fa["reason"]
            or len(result.pareto) != g1fa["pareto_count"]
            or result.certificate_sha256 != g1fa["certificate_sha256"]):
        raise SystemExit("output-neutrality failed: selector result differs from G1F-A")
    if (result.status == "budget_exhausted"
            and (result.candidate is not None or result.pareto
                 or result.certificate_sha256 is not None)):
        raise SystemExit("whole-search exhaustion returned partial candidate state")

    gate_names = (
        "diagnostic_gate_no_compatible_records",
        "diagnostic_gate_objective_rejected",
        "diagnostic_gate_insufficient_axes",
        "diagnostic_gate_minimum_resultant",
        "diagnostic_gate_integer_tolerance",
        "diagnostic_gate_accepted",
        "diagnostic_gate_budget_aborted",
    )
    gate_histogram = {name.removeprefix("diagnostic_gate_"): counters[name]
                      for name in gate_names}
    if sum(gate_histogram.values()) != counters["complete_labels"]:
        raise SystemExit("typed terminal gates do not account for complete-label attempts")
    signatures = {}
    for group in ("partial_state", "leaf_signature"):
        observations = counters[f"diagnostic_{group}_observations"]
        unique = counters[f"diagnostic_{group}_unique"]
        repeats = counters[f"diagnostic_{group}_repeats"]
        truncated = counters[f"diagnostic_{group}_truncated"]
        if observations != unique + repeats + truncated:
            raise SystemExit(f"{group} census counters do not reconcile")
        signatures[group] = {
            "observations": observations,
            "unique": unique,
            "repeats": repeats,
            "repeat_rate_of_observations": repeats / observations if observations else 0.0,
            "truncated": truncated,
            "storage_cap": selector.MAX_DIAGNOSTIC_SIGNATURES,
        }

    for path, expected, name in pinned:
        _check(path, expected, f"post-call {name}")
    output = {
        "schema": "m4-g1f-b-r4-selector-diagnostics-probe-v1",
        "prereg_sha256": expected_prereg,
        "selector_sha256": EXPECTED["selector"],
        "test_sha256": EXPECTED["tests"],
        "probe_script_sha256": expected_probe,
        "record_parser_sha256": EXPECTED["parser"],
        "r4_freeze_sha256": EXPECTED["freeze"],
        "r4_artifact_id": ARTIFACT_ID,
        "sidecar_sha256": EXPECTED["sidecar"],
        "g1f_a_baseline_sha256": EXPECTED["g1fa"],
        "g1d_baseline_sha256": EXPECTED["g1d"],
        "g1e_b_baseline_sha256": EXPECTED["g1eb"],
        "counts": counts,
        "budget": budget,
        "selector_call_count": 1,
        "elapsed_seconds": elapsed,
        "status": result.status,
        "reason": result.reason,
        "gate_histogram": gate_histogram,
        "complete_labels": counters["complete_labels"],
        "signature_census": signatures,
        "logical_counters": {
            name: counters[name] for name in (
                "search_nodes", "compatibility_checks", "objective_rows",
                "axis_solve_requests", "axis_solves", "physical_axis_cache_hits",
                "physical_axis_cache_misses", "objective_cache_hits",
                "objective_cache_misses", "objective_cache_unique_signatures",
                "pareto_comparisons", "candidate_certificates_buffered",
                "buffered_match_rows", "evaluated_candidates",
            )
        },
        "all_counters": counters,
        "candidate_count": int(result.candidate is not None),
        "pareto_count": len(result.pareto),
        "certificate_sha256": result.certificate_sha256,
        "no_partial_result_on_exhaustion": result.status != "budget_exhausted"
            or (result.candidate is None and not result.pareto
                and result.certificate_sha256 is None),
        "output_neutral_vs_g1f_a": True,
        "labels_present": False,
        "images_opened": False,
        "ground_truth_opened": False,
        "evaluator_called": False,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT),
        "sha256": sha256_file(OUTPUT),
        "status": result.status,
        "reason": result.reason,
        "elapsed_seconds": elapsed,
        "gate_histogram": gate_histogram,
        "signature_census": signatures,
        "logical_counters": output["logical_counters"],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
