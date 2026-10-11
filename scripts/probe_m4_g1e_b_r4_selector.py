"""One preregistered same-budget R4 selector-only G1E-B call.

Only the frozen R4 JSON PlaneReading/PlaneMatch records are deserialized.
The mask sidecar is hash-checked but never opened; no images or labels are read.
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
OUTPUT = ROOT / "exports/evaluation/m4-g1e-b-r4-selector-probe.json"
PREREG = ROOT / "docs/reports/M4-G1E-B-r4-selector-prereg-2026-09-13.md"
SELECTOR_PATH = ROOT / "src/mc_photogrammetry/reconstruction/_joint_plane_selector.py"
TEST_PATH = ROOT / "tests/test_joint_plane_selector.py"
PROFILE_SCRIPT = ROOT / "scripts/profile_m4_g1e_b_selector.py"
PROFILE_OUTPUT = ROOT / "exports/evaluation/m4-g1e-b-paired-profile-2026-09-13.json"
G1D_OUTPUT = ROOT / "exports/evaluation/m4-g1d-r4-selector-probe.json"
G1D_SCRIPT = ROOT / "scripts/probe_m4_g1d_r4_selector.py"

EXPECTED_PREREG_SHA256 = "74c1f0760258f2843da5b5f96c74242ff1299ac67533bc301e44316f4d80b31e"
EXPECTED_SELECTOR_SHA256 = "c562fe2324562294be6b68cd4bb0d95234038965d2315f13dcc7a86b5183757e"
EXPECTED_TEST_SHA256 = "7823f0a519b04b227405e9f915f8198ccc66b0e35d7a480dcd564efea3b0d4fd"
EXPECTED_PROFILE_SCRIPT_SHA256 = "e168cc0002883ca292943c10525986ac85fb06ebc0a4bd397a89ea5201e5d194"
EXPECTED_PROFILE_SHA256 = "2a9376605c64930f7cc3d816709d6efa6b77ff8ea62a36f277138b470573bfd5"
EXPECTED_G1D_OUTPUT_SHA256 = "2f232056601fcbdfc90ee1c9576269b2df2eb387e10bf53c068b88fa8c919e6c"
EXPECTED_G1D_SCRIPT_SHA256 = "917f5d51bffab1dec7fa65a120ab7769ba67c68ea70cf0e761fc8a7c02ec9dee"
EXPECTED_FREEZE_SHA256 = "96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d"
EXPECTED_SIDECAR_SHA256 = "f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac"
EXPECTED_ARTIFACT_ID = "7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0"
EXPECTED_G1D_SELECTOR_SHA256 = "367329861d3838ed3df360bde9b2735242f611ce0e25bfecb68cf26ac2f0298b"
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


def _load_r4_records(document: dict):
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
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite probe output: {OUTPUT}")
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if temporary.exists():
        raise SystemExit(f"refusing to overwrite temporary output: {temporary}")

    _check(PREREG, EXPECTED_PREREG_SHA256, "preregistration")
    _check(SELECTOR_PATH, EXPECTED_SELECTOR_SHA256, "selector source")
    _check(TEST_PATH, EXPECTED_TEST_SHA256, "selector tests")
    _check(PROFILE_SCRIPT, EXPECTED_PROFILE_SCRIPT_SHA256, "synthetic profile script")
    _check(PROFILE_OUTPUT, EXPECTED_PROFILE_SHA256, "synthetic profile")
    _check(G1D_OUTPUT, EXPECTED_G1D_OUTPUT_SHA256, "same-budget G1D baseline")
    _check(G1D_SCRIPT, EXPECTED_G1D_SCRIPT_SHA256, "R4 parsing helper")
    _check(FREEZE, EXPECTED_FREEZE_SHA256, "R4 freeze")
    _check(SIDECAR, EXPECTED_SIDECAR_SHA256, "R4 sidecar")

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

    readings, matches, counts = _load_r4_records(document)
    g1d = strict_json(G1D_OUTPUT.read_bytes())
    if (g1d.get("schema") != "m4-g1d-r4-selector-probe-v1"
            or g1d.get("selector_sha256") != EXPECTED_G1D_SELECTOR_SHA256
            or g1d.get("budget") != {
                name: getattr(BUDGET, name) for name in BUDGET.__dataclass_fields__
            }
            or g1d.get("r4_freeze_sha256") != EXPECTED_FREEZE_SHA256
            or g1d.get("sidecar_sha256") != EXPECTED_SIDECAR_SHA256):
        raise SystemExit("G1D comparison is not the pinned same-budget R4 call")

    started = time.perf_counter()
    result = selector._select_joint_labels_constraint_first(
        readings, matches, budget=BUDGET)
    elapsed = time.perf_counter() - started
    no_partial = (
        result.status != "budget_exhausted"
        or (result.candidate is None and not result.pareto
            and result.certificate_sha256 is None)
    )
    if result.status == "budget_exhausted" and not no_partial:
        raise SystemExit("selector violated whole-search exhaustion contract")

    pinned_paths = (
        (PREREG, EXPECTED_PREREG_SHA256),
        (SELECTOR_PATH, EXPECTED_SELECTOR_SHA256),
        (TEST_PATH, EXPECTED_TEST_SHA256),
        (PROFILE_SCRIPT, EXPECTED_PROFILE_SCRIPT_SHA256),
        (PROFILE_OUTPUT, EXPECTED_PROFILE_SHA256),
        (G1D_OUTPUT, EXPECTED_G1D_OUTPUT_SHA256),
        (G1D_SCRIPT, EXPECTED_G1D_SCRIPT_SHA256),
        (FREEZE, EXPECTED_FREEZE_SHA256),
        (SIDECAR, EXPECTED_SIDECAR_SHA256),
    )
    for path, expected in pinned_paths:
        _check(path, expected, "post-call pinned input")

    g1d_counters = g1d["counters"]
    counters = {name: getattr(result.counters, name)
                for name in result.counters.__dataclass_fields__}
    output = {
        "schema": "m4-g1e-b-r4-selector-probe-v1",
        "prereg_sha256": EXPECTED_PREREG_SHA256,
        "r4_freeze_sha256": EXPECTED_FREEZE_SHA256,
        "r4_artifact_id": EXPECTED_ARTIFACT_ID,
        "sidecar_sha256": EXPECTED_SIDECAR_SHA256,
        "selector_sha256": EXPECTED_SELECTOR_SHA256,
        "test_sha256": EXPECTED_TEST_SHA256,
        "profile_script_sha256": EXPECTED_PROFILE_SCRIPT_SHA256,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "probe_script_sha256": sha256_file(Path(__file__).resolve()),
        "r4_reading_source_sha256": EXPECTED_G1D_SCRIPT_SHA256,
        "g1d_baseline_sha256": EXPECTED_G1D_OUTPUT_SHA256,
        "counts": counts,
        "budget": {name: getattr(BUDGET, name) for name in BUDGET.__dataclass_fields__},
        "elapsed_seconds": elapsed,
        "status": result.status,
        "reason": result.reason,
        "counters": counters,
        "pareto_count": len(result.pareto),
        "certificate_sha256": result.certificate_sha256,
        "no_partial_result_on_exhaustion": no_partial,
        "same_budget_g1d_comparison": {
            "status": g1d["status"],
            "reason": g1d["reason"],
            "elapsed_seconds": g1d["elapsed_seconds"],
            "counters": g1d_counters,
            "pareto_count": g1d["candidate_count"],
            "certificate_sha256": g1d["certificate_sha256"],
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
        "counters": counters,
        "elapsed_seconds": elapsed,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
