"""One preregistered same-numeric-budget G1F-A selector-only R4 call.

Only frozen serialized PlaneReading/PlaneMatch records are deserialized. The
R4 mask sidecar is hash-checked but never opened as a payload; no image or
ground-truth/evaluator data is consumed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import _joint_plane_selector as selector
from mc_photogrammetry.reconstruction.lattice_planes import reading_order_key
from probe_m4_g1d_r4_selector import (
    FREEZE,
    SIDECAR,
    _finite_records,
    _reading,
    canonical_artifact_id,
    sha256_file,
    strict_json,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "exports/evaluation/m4-g1f-a-r4-axis-accounting-probe.json"
PREREG = ROOT / "docs/reports/M4-G1F-A-r4-selector-prereg-2026-09-13.md"
SELECTOR_PATH = ROOT / "src/mc_photogrammetry/reconstruction/_joint_plane_selector.py"
TEST_PATH = ROOT / "tests/test_joint_plane_selector.py"
PROFILE_SCRIPT = ROOT / "scripts/profile_m4_g1f_a_axis_budget.py"
PROFILE_OUTPUT = ROOT / "exports/evaluation/m4-g1f-a-axis-accounting-profile-2026-09-13.json"
FIXTURE_SCRIPT = ROOT / "scripts/benchmark_m4_g1c_selector.py"
G1D_OUTPUT = ROOT / "exports/evaluation/m4-g1d-r4-selector-probe.json"
G1E_B_OUTPUT = ROOT / "exports/evaluation/m4-g1e-b-r4-selector-probe.json"
G1D_SCRIPT = ROOT / "scripts/probe_m4_g1d_r4_selector.py"

EXPECTED_PREREG_SHA256 = "44b7ab24e33809073a830d71257a10bfdcc8ef5d1c4d84b72be78bc18779618b"
EXPECTED_SELECTOR_SHA256 = "04a0114a184d622d4a8e6bb669377a672c3c35c35e1d7aff8416d572400bf485"
EXPECTED_TEST_SHA256 = "ee877c3d4a61e5bffb79a171c4620c6a35563636ce7ecc6f388aa0256692dff4"
EXPECTED_PROFILE_SCRIPT_SHA256 = "74ee89ad7a6c068e07414f500c6be60fe8b95eb9dc1ce3b23712a9217f0d86a2"
EXPECTED_PROFILE_SHA256 = "0604bba5f193a1ad1e9ff9bf5c7b0e8bbd2e0844e2c4ca3a15054e93c4d0f123"
EXPECTED_FIXTURE_SHA256 = "65170810a4786a17f31117461bf4690c38968476bd07f56e37a69e4a3a0be933"
EXPECTED_G1D_OUTPUT_SHA256 = "2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c"
EXPECTED_G1E_B_OUTPUT_SHA256 = "a50885aa64277054ba05fe618eeb1d27eadedd150bd403abdd87e58e925bc9e3"
EXPECTED_G1D_SCRIPT_SHA256 = "917f5d51bffab1dec7fa65a120ab7769ba67c68ea70cf0e761fc8a7c02ec9dee"
EXPECTED_FREEZE_SHA256 = "96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d"
EXPECTED_SIDECAR_SHA256 = "f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac"
EXPECTED_ARTIFACT_ID = "7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0"
EXPECTED_COUNTS = {"frames": 8, "readings": 33, "matches": 200}
BUDGET = selector.SearchBudget(
    max_search_nodes=4_096,
    max_compatibility_checks=1_000_000,
    max_objective_rows=20_000,
    max_axis_solves=512,
    max_pareto_comparisons=64,
    max_candidate_certificates=16,
    max_buffered_match_rows=1_024,
)


def _check(path: Path, expected: str, description: str) -> None:
    if sha256_file(path) != expected:
        raise SystemExit(f"{description} hash mismatch: {path}")


def _load_records(document: dict):
    readings = []
    readings_by_frame = {}
    for frame in document["frames"]:
        frame_id = frame["frame_id"]
        frame_readings = []
        for item in frame["readings"]:
            if item["frame"] != frame_id:
                raise SystemExit("serialized reading/frame identity mismatch")
            frame_readings.append(_reading(item))
        readings.extend(frame_readings)
        readings_by_frame[frame_id] = frame_readings
    matches = [plane_graph.PlaneMatch(**item) for item in document["graph"]["matches"]]
    canonical_by_index = {
        (frame_id, index): reading
        for frame_id, frame_readings in readings_by_frame.items()
        for index, reading in enumerate(sorted(frame_readings, key=reading_order_key))
    }
    for match in matches:
        try:
            left = canonical_by_index[(match.left, match.left_plane)]
            right = canonical_by_index[(match.right, match.right_plane)]
        except KeyError as exc:
            raise SystemExit("match references missing canonical reading index") from exc
        if match.left_axis != left.axis or match.right_axis != right.axis:
            raise SystemExit("match axis conflicts with canonical reading index")
    _finite_records(readings, matches)
    counts = {"frames": len(document["frames"]), "readings": len(readings),
              "matches": len(matches)}
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"R4 selector input counts differ: {counts}")
    return readings, matches, counts


def main() -> None:
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if OUTPUT.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite existing probe output/temp: {OUTPUT}")

    pinned = (
        (PREREG, EXPECTED_PREREG_SHA256, "preregistration"),
        (SELECTOR_PATH, EXPECTED_SELECTOR_SHA256, "selector source"),
        (TEST_PATH, EXPECTED_TEST_SHA256, "selector tests"),
        (PROFILE_SCRIPT, EXPECTED_PROFILE_SCRIPT_SHA256, "synthetic profile script"),
        (PROFILE_OUTPUT, EXPECTED_PROFILE_SHA256, "synthetic profile"),
        (FIXTURE_SCRIPT, EXPECTED_FIXTURE_SHA256, "synthetic fixture source"),
        (G1D_OUTPUT, EXPECTED_G1D_OUTPUT_SHA256, "G1D request-budget baseline"),
        (G1E_B_OUTPUT, EXPECTED_G1E_B_OUTPUT_SHA256, "G1E-B request-budget baseline"),
        (G1D_SCRIPT, EXPECTED_G1D_SCRIPT_SHA256, "R4 record parser"),
        (FREEZE, EXPECTED_FREEZE_SHA256, "R4 freeze"),
        (SIDECAR, EXPECTED_SIDECAR_SHA256, "R4 mask-sidecar digest"),
    )
    for path, expected, description in pinned:
        _check(path, expected, description)

    document = strict_json(FREEZE.read_bytes())
    if (document.get("labels_present") is not False
            or document.get("schema_version") != 1
            or document.get("artifact_id") != EXPECTED_ARTIFACT_ID
            or document.get("artifact_id") != canonical_artifact_id(document)):
        raise SystemExit("R4 freeze identity/schema mismatch")
    sidecar = document.get("mask_sidecar", {})
    if (sidecar.get("sha256") != EXPECTED_SIDECAR_SHA256
            or sidecar.get("file") != SIDECAR.name
            or sidecar.get("size_bytes") != SIDECAR.stat().st_size):
        raise SystemExit("R4 sidecar descriptor mismatch")

    readings, matches, counts = _load_records(document)
    old_g1d = strict_json(G1D_OUTPUT.read_bytes())
    old_g1e_b = strict_json(G1E_B_OUTPUT.read_bytes())
    expected_budget = {name: getattr(BUDGET, name) for name in BUDGET.__dataclass_fields__}
    for baseline, schema in ((old_g1d, "m4-g1d-r4-selector-probe-v1"),
                             (old_g1e_b, "m4-g1e-b-r4-selector-probe-v1")):
        if (baseline.get("schema") != schema
                or baseline.get("budget") != expected_budget
                or baseline.get("r4_freeze_sha256") != EXPECTED_FREEZE_SHA256
                or baseline.get("sidecar_sha256") != EXPECTED_SIDECAR_SHA256):
            raise SystemExit("prior selector output is not the pinned same-numeric-budget R4 call")

    # The single authorized R4 invocation: public natural-order default.
    started = time.perf_counter()
    result = selector.select_joint_labels(readings, matches, budget=BUDGET)
    elapsed = time.perf_counter() - started
    no_partial = (
        result.status != "budget_exhausted"
        or (result.candidate is None and not result.pareto
            and result.certificate_sha256 is None)
    )
    if result.status == "budget_exhausted" and not no_partial:
        raise SystemExit("selector violated whole-search exhaustion contract")

    for path, expected, description in pinned:
        _check(path, expected, f"post-call {description}")
    counters = {name: getattr(result.counters, name)
                for name in result.counters.__dataclass_fields__}
    output = {
        "schema": "m4-g1f-a-r4-axis-accounting-probe-v1",
        "prereg_sha256": EXPECTED_PREREG_SHA256,
        "r4_freeze_sha256": EXPECTED_FREEZE_SHA256,
        "r4_artifact_id": EXPECTED_ARTIFACT_ID,
        "sidecar_sha256": EXPECTED_SIDECAR_SHA256,
        "selector_sha256": EXPECTED_SELECTOR_SHA256,
        "test_sha256": EXPECTED_TEST_SHA256,
        "profile_script_sha256": EXPECTED_PROFILE_SCRIPT_SHA256,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "fixture_script_sha256": EXPECTED_FIXTURE_SHA256,
        "probe_script_sha256": sha256_file(Path(__file__).resolve()),
        "parser_script_sha256": EXPECTED_G1D_SCRIPT_SHA256,
        "g1d_baseline_sha256": EXPECTED_G1D_OUTPUT_SHA256,
        "g1e_b_baseline_sha256": EXPECTED_G1E_B_OUTPUT_SHA256,
        "counts": counts,
        "budget": expected_budget,
        "elapsed_seconds": elapsed,
        "status": result.status,
        "reason": result.reason,
        "counters": counters,
        "pareto_count": len(result.pareto),
        "certificate_sha256": result.certificate_sha256,
        "no_partial_result_on_exhaustion": no_partial,
        "prior_request_budget_runs": {
            "g1d": {
                "status": old_g1d["status"], "reason": old_g1d["reason"],
                "elapsed_seconds": old_g1d["elapsed_seconds"],
                "counters": old_g1d["counters"],
                "candidate_count": old_g1d["candidate_count"],
                "certificate_sha256": old_g1d["certificate_sha256"],
            },
            "g1e_b": {
                "status": old_g1e_b["status"], "reason": old_g1e_b["reason"],
                "elapsed_seconds": old_g1e_b["elapsed_seconds"],
                "counters": old_g1e_b["counters"],
                "pareto_count": old_g1e_b["pareto_count"],
                "certificate_sha256": old_g1e_b["certificate_sha256"],
            },
        },
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
        "counters": counters,
        "pareto_count": len(result.pareto),
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
