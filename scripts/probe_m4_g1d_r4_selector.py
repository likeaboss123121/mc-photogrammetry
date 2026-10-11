"""One preregistered low-budget R4 probe through isolated G1C only.

This script consumes serialized plane readings and pair matches from the pinned
R4 freeze. It never opens a capture image, mask payload, evaluator, or label.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import _joint_plane_selector as selector
from mc_photogrammetry.reconstruction.lattice_planes import (
    DistanceAlternative,
    PlaneReading,
    PlaneRectification,
    reading_order_key,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/freeze.json"
SIDECAR = ROOT / (
    "exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/"
    "freeze.json.f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac.masks.bin"
)
OUTPUT = ROOT / "exports/evaluation/m4-g1d-r4-selector-probe.json"

EXPECTED_FREEZE_SHA256 = "96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d"
EXPECTED_SIDECAR_SHA256 = "f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac"
EXPECTED_SELECTOR_SHA256 = "367329861d3838ed3df360bde9b2735242f611ce0e25bfecb68cf26ac2f0298b"
EXPECTED_PREREG_SHA256 = "f470d4be43b478a07a840a5c850afc24cd7408eafe0aaed2162edf6a56d8087b"
PREREG = ROOT / "docs/reports/M4-G1D-r4-selector-probe-prereg-2026-09-13.md"
SELECTOR_PATH = ROOT / "src/mc_photogrammetry/reconstruction/_joint_plane_selector.py"
SIDE_INFO = {
    "file": SIDECAR.name,
    "sha256": EXPECTED_SIDECAR_SHA256,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(payload: bytes) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    document = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=reject_constant)
    if not isinstance(document, dict):
        raise ValueError("freeze root must be an object")
    return document


def canonical_artifact_id(document: dict) -> str:
    content = json.loads(json.dumps(document))
    content.pop("artifact_id", None)
    sidecar = content.get("mask_sidecar")
    if isinstance(sidecar, dict):
        sidecar.pop("file", None)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tuple2(values):
    return tuple(float(value) for value in values)


def _reading(record: dict) -> PlaneReading:
    rect_record = record["rectification"]
    rectification = PlaneRectification(
        axis_matrix=tuple(tuple(float(value) for value in row)
                          for row in rect_record["axis_matrix"]),
        focal=float(rect_record["focal"]),
        principal_point=_tuple2(rect_record["principal_point"]),
        centre=_tuple2(rect_record["centre"]),
    )
    alternatives = tuple(
        DistanceAlternative(
            blocks_per_period=float(item["blocks_per_period"]),
            distance_blocks=float(item["distance_blocks"]),
            support=float(item["support"]),
        )
        for item in record["alternatives"]
    )
    polygon_record = record["polygon"]
    polygon = None if polygon_record is None else tuple(
        tuple(float(value) for value in point) for point in polygon_record
    )
    return PlaneReading(
        frame=str(record["frame"]),
        axis=int(record["axis"]),
        side=float(record["side"]),
        alternatives=alternatives,
        period_pixels=float(record["period_pixels"]),
        phase_pixels=_tuple2(record["phase_pixels"]),
        extent=float(record["extent"]),
        samples=int(record["samples"]),
        polygon=polygon,
        coverage_cells=int(record["coverage_cells"]),
        structural_share=float(record["structural_share"]),
        confidence=float(record["confidence"]),
        rectification=rectification,
    )


def _finite_records(readings, matches):
    for reading in readings:
        values = [reading.side, reading.period_pixels, reading.extent,
                  reading.structural_share, reading.confidence,
                  reading.rectification.focal, *reading.phase_pixels,
                  *reading.rectification.principal_point, *reading.rectification.centre]
        values.extend(value for row in reading.rectification.axis_matrix for value in row)
        values.extend(number for alternative in reading.alternatives
                      for number in (alternative.blocks_per_period,
                                     alternative.distance_blocks, alternative.support))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite reading field")
    for match in matches:
        numbers = [match.left_b, match.right_b, match.score, match.peak,
                   match.second_peak]
        if match.psr is not None:
            numbers.append(match.psr)
        if match.integer_residual is not None:
            numbers.append(match.integer_residual)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("non-finite match field")


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing probe output: {OUTPUT}")
    if sha256_file(PREREG) != EXPECTED_PREREG_SHA256:
        raise SystemExit("preregistration hash mismatch")
    if sha256_file(SELECTOR_PATH) != EXPECTED_SELECTOR_SHA256:
        raise SystemExit("selector source hash mismatch")
    if sha256_file(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise SystemExit("R4 freeze hash mismatch")
    if sha256_file(SIDECAR) != EXPECTED_SIDECAR_SHA256:
        raise SystemExit("R4 mask-sidecar hash mismatch")

    freeze_bytes = FREEZE.read_bytes()
    document = strict_json(freeze_bytes)
    if (document.get("labels_present") is not False
            or document.get("schema_version") != 1
            or document.get("artifact_id") != "7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0"
            or document.get("artifact_id") != canonical_artifact_id(document)):
        raise SystemExit("R4 freeze identity/schema mismatch")
    sidecar_record = document.get("mask_sidecar", {})
    if (sidecar_record.get("file") != SIDE_INFO["file"]
            or sidecar_record.get("sha256") != SIDE_INFO["sha256"]
            or sidecar_record.get("size_bytes") != SIDECAR.stat().st_size):
        raise SystemExit("R4 sidecar descriptor mismatch")

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
            left_reading = canonical_by_index[(match.left, match.left_plane)]
            right_reading = canonical_by_index[(match.right, match.right_plane)]
        except KeyError as exc:
            raise SystemExit("match references a missing canonical reading index") from exc
        if (match.left_axis != left_reading.axis
                or match.right_axis != right_reading.axis):
            raise SystemExit("match axis conflicts with canonical reading index")
    _finite_records(readings, matches)
    if len(document["frames"]) != 8 or len(readings) != 33 or len(matches) != 200:
        raise SystemExit("R4 selector input counts differ from pinned inventory")

    budget = selector.SearchBudget(
        max_search_nodes=4_096,
        max_compatibility_checks=1_000_000,
        max_objective_rows=20_000,
        max_axis_solves=512,
        max_pareto_comparisons=64,
        max_candidate_certificates=16,
        max_buffered_match_rows=1_024,
    )
    started = time.perf_counter()
    result = selector.select_joint_labels(readings, matches, budget=budget)
    elapsed = time.perf_counter() - started

    if (sha256_file(FREEZE) != EXPECTED_FREEZE_SHA256
            or sha256_file(SIDECAR) != EXPECTED_SIDECAR_SHA256
            or sha256_file(SELECTOR_PATH) != EXPECTED_SELECTOR_SHA256):
        raise SystemExit("a pinned input changed during the selector probe")
    no_partial = (result.status != "budget_exhausted"
                  or (result.candidate is None and not result.pareto
                      and result.certificate_sha256 is None))
    output = {
        "schema": "m4-g1d-r4-selector-probe-v1",
        "prereg_sha256": EXPECTED_PREREG_SHA256,
        "r4_freeze_sha256": EXPECTED_FREEZE_SHA256,
        "r4_artifact_id": document["artifact_id"],
        "sidecar_sha256": EXPECTED_SIDECAR_SHA256,
        "selector_sha256": EXPECTED_SELECTOR_SHA256,
        "probe_script_sha256": sha256_file(Path(__file__).resolve()),
        "counts": {"frames": len(document["frames"]),
                   "readings": len(readings), "matches": len(matches)},
        "budget": {name: getattr(budget, name) for name in budget.__dataclass_fields__},
        "elapsed_seconds": elapsed,
        "status": result.status,
        "reason": result.reason,
        "counters": {name: getattr(result.counters, name)
                     for name in result.counters.__dataclass_fields__},
        "candidate_count": len(result.pareto),
        "certificate_sha256": result.certificate_sha256,
        "no_partial_result_on_exhaustion": no_partial,
        "labels_present": False,
        "images_opened": False,
        "ground_truth_opened": False,
        "evaluator_called": False,
    }
    if result.status == "budget_exhausted" and not no_partial:
        raise SystemExit("selector violated whole-search exhaustion contract")
    rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if temp.exists():
        raise SystemExit(f"refusing to overwrite existing temporary output: {temp}")
    temp.write_text(rendered, encoding="utf-8", newline="\n")
    temp.replace(OUTPUT)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
