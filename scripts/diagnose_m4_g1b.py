"""Strict, label-blind census of the immutable M4 G1A R4 graph artifacts.

Reads only the R4 frozen JSON/sidecar and G1A diagnostic JSON. It never opens
the PNG inputs, capture metadata, GT, evaluator, or reference files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.reconstruction import frozen_image_runner


R4_FREEZE = Path("exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/freeze.json")
R4_FREEZE_SHA256 = "96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d"
R4_SIDECAR_SHA256 = "f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac"
G1A_EXACT = Path("exports/evaluation/m4-g1a-r4-exact.json")
G1A_EXACT_SHA256 = "4c0117015c9f21766c1bae671648c795d5d9cfa0b370a3098e96cc0da38bc681"
G1A_BASELINE = Path("exports/evaluation/m4-g1a-r4-baseline.json")
G1A_BASELINE_SHA256 = "b122a1f2989b1d90a7b43c84ab18dc601865f479e078e3007f205e4201eed3d2"
G1A_PROTECTED = Path("docs/reports/M4-G1A-protected-hashes.json")
G1A_PROTECTED_SHA256 = "72c51b6301c73706a4b73453a7bf3485016c3e754404c0131fcde3fbcda05d28"
G1A_SOURCE = Path("src/mc_photogrammetry/reconstruction/plane_graph.py")
G1A_SOURCE_SHA256 = "961459b5a9f7fcb640448cd74be804e03a1b2c1b24a4522fdf7fb89395bd1e16"
G1A_TESTS = Path("tests/test_plane_graph.py")
G1A_TESTS_SHA256 = "1b5e5b4554aa369015c18cd38dcd4160d2fd35f15ac00e7f837cbaab22d0aba6"
G1A_SCRIPT = Path("scripts/diagnose_m4_g1a.py")
G1A_SCRIPT_SHA256 = "fdf067a575ac5a6e9f91c581fa9bc39ae82978dbff29701a04ae3fd1cde46783"
G1A_PREREG = Path("docs/reports/M4-G1A-prereg-2026-09-13.md")
G1A_PREREG_SHA256 = "ac5ac010a0d707e7e0a7e43b6b0b223719d030cc506d163aa828be0f2e52e40f"
G1A_ARTIFACT_ID = "7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0"

EXPECTED_INPUTS = [
    {"basename": "view-canal-town-architectural-component-bbox-00.png", "width": 1280, "height": 720, "sha256": "48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c"},
    {"basename": "view-canal-town-architectural-component-bbox-01.png", "width": 1600, "height": 900, "sha256": "8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696"},
    {"basename": "view-canal-town-architectural-component-bbox-02.png", "width": 1920, "height": 1080, "sha256": "2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95"},
    {"basename": "view-canal-town-architectural-component-bbox-03.png", "width": 1920, "height": 1080, "sha256": "e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9"},
    {"basename": "view-canal-town-architectural-component-bbox-04.png", "width": 1280, "height": 720, "sha256": "8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f"},
    {"basename": "view-canal-town-architectural-component-bbox-05.png", "width": 1600, "height": 900, "sha256": "6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b"},
    {"basename": "view-canal-town-architectural-component-bbox-06.png", "width": 1600, "height": 900, "sha256": "53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923"},
    {"basename": "view-canal-town-architectural-component-bbox-07.png", "width": 1280, "height": 720, "sha256": "fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556"},
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_bytes(path: Path, expected: str) -> bytes:
    payload = (ROOT / path).read_bytes()
    actual = _sha256(payload)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: expected {expected}, got {actual}")
    return payload


def _reading_key(row: dict[str, Any]) -> tuple:
    """Mirror lattice_planes.reading_order_key over its serialized fields."""
    polygon = row["polygon"]
    return (
        row["axis"], row["side"], row["period_pixels"], tuple(row["phase_pixels"]),
        row["extent"], row["samples"], () if polygon is None else tuple(tuple(p) for p in polygon),
        row["structural_share"], row["confidence"],
        tuple((a["blocks_per_period"], a["distance_blocks"], a["support"])
              for a in row["alternatives"]),
    )


def _world_axis(axis: int, quadrant: int) -> int:
    if quadrant % 2 == 0:
        return axis
    return {0: 2, 1: 1, 2: 0}[axis]


def _group_key(match: dict[str, Any]) -> tuple:
    return (match["left"], match["left_plane"], match["right"],
            match["right_plane"], match["quadrant_of_right"])


def _node(frame: str, plane: int) -> tuple[str, int]:
    return (frame, int(plane))


def _semantic_edge_key(match: dict[str, Any]) -> tuple:
    """Compare active evidence independent of its quadrant-tagged record id."""
    return tuple(match[key] for key in (
        "left", "left_plane", "left_axis", "left_b", "right", "right_plane",
        "right_axis", "right_b", "psr", "peak", "second_peak", "overlap_cells",
        "integer_residual", "score",
    ))


def _components(nodes: set[tuple[str, int]], adjacency: dict[tuple[str, int], set[tuple[str, int]]]) -> list[set[tuple[str, int]]]:
    unseen = set(nodes)
    result = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[tuple[str, int]] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(adjacency[current] - component)
        result.append(component)
    return sorted(result, key=lambda c: min(c))


def _assignment_census(
    label: str,
    quadrants: dict[str, int],
    frames: list[str],
    matches: list[dict[str, Any]],
    intrinsic_domains: dict[tuple[str, int], list[float]],
) -> dict[str, Any]:
    active: list[tuple[int, dict[str, Any]]] = []
    incompatible = []
    for index, match in enumerate(matches):
        if ((quadrants[match["right"]] - quadrants[match["left"]]) % 4
                != match["quadrant_of_right"]):
            continue
        left_world = _world_axis(match["left_axis"], quadrants[match["left"]])
        right_world = _world_axis(match["right_axis"], quadrants[match["right"]])
        if left_world != right_world:
            incompatible.append(index)
        active.append((index, match))

    demands: dict[tuple[str, int], set[float]] = defaultdict(set)
    incidents: Counter[tuple[str, int]] = Counter()
    selected_last_writer: dict[tuple[str, int], float] = {}
    pair_groups: dict[tuple, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    axes: dict[int, dict[str, Any]] = {}
    for index, match in active:
        left = _node(match["left"], match["left_plane"])
        right = _node(match["right"], match["right_plane"])
        demands[left].add(float(match["left_b"]))
        demands[right].add(float(match["right_b"]))
        incidents[left] += 1
        incidents[right] += 1
        selected_last_writer[left] = float(match["left_b"])
        selected_last_writer[right] = float(match["right_b"])
        pair_groups[_group_key(match)].append((index, match))

    for world_axis in range(3):
        axis_edges = []
        axis_nodes: set[tuple[str, int]] = set()
        axis_domains: dict[tuple[str, int], set[float]] = defaultdict(set)
        adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
        axis_groups: dict[tuple, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, match in active:
            left_axis = _world_axis(match["left_axis"], quadrants[match["left"]])
            right_axis = _world_axis(match["right_axis"], quadrants[match["right"]])
            if left_axis != world_axis or right_axis != world_axis:
                continue
            left = _node(match["left"], match["left_plane"])
            right = _node(match["right"], match["right_plane"])
            axis_edges.append(index)
            axis_nodes.update((left, right))
            axis_domains[left].add(float(match["left_b"]))
            axis_domains[right].add(float(match["right_b"]))
            adjacency[left].add(right)
            adjacency[right].add(left)
            axis_groups[_group_key(match)].append((index, match))

        components = []
        for component in _components(axis_nodes, adjacency):
            demand_product = math.prod(len(axis_domains[node]) for node in component)
            alternative_product = math.prod(len(intrinsic_domains[node]) for node in component)
            component_groups = [
                rows for key, rows in axis_groups.items()
                if _node(key[0], key[1]) in component and _node(key[2], key[3]) in component
            ]
            components.append({
                "nodes": [{"frame": frame, "plane_index": plane,
                           "match_b_domain": sorted(axis_domains[(frame, plane)]),
                           "intrinsic_alternative_b_domain": intrinsic_domains[(frame, plane)]}
                          for frame, plane in sorted(component)],
                "node_count": len(component),
                "unique_pair_groups": len(component_groups),
                "match_record_count": sum(len(rows) for rows in component_groups),
                "match_demand_state_product": demand_product,
                "intrinsic_alternative_state_product": alternative_product,
            })
        axes[world_axis] = {
            "match_record_count": len(axis_edges),
            "unique_pair_group_count": len(axis_groups),
            "reading_count": len(axis_nodes),
            "multi_b_reading_count": sum(len(v) > 1 for n, v in axis_domains.items()),
            "components": components,
            "match_demand_state_product": math.prod(
                math.prod(len(axis_domains[n]) for n in component)
                for component in _components(axis_nodes, adjacency)),
        }

    selected_records = [
        {"frame": frame, "plane_index": plane, "blocks_per_period": b,
         "demand_domain": sorted(demands[(frame, plane)])}
        for (frame, plane), b in sorted(selected_last_writer.items())
    ]
    return {
        "name": label,
        "quadrants": [quadrants[frame] for frame in frames],
        "active_match_record_indices": [index for index, _ in active],
        "active_match_record_count": len(active),
        "active_unique_pair_quadrant_group_count": len(pair_groups),
        "duplicate_pair_quadrant_groups": [
            {"left": key[0], "left_plane": key[1], "right": key[2],
             "right_plane": key[3], "quadrant_of_right": key[4],
             "match_records": [{"index": index, "left_b": m["left_b"],
                                "right_b": m["right_b"], "score": m["score"]}
                               for index, m in rows]}
            for key, rows in sorted(pair_groups.items()) if len(rows) > 1
        ],
        "axis_incompatible_match_record_indices": incompatible,
        "active_readings": [
            {"frame": frame, "plane_index": plane,
             "match_incident_record_count": incidents[(frame, plane)],
             "match_b_domain": sorted(demands[(frame, plane)]),
             "intrinsic_alternative_b_domain": intrinsic_domains[(frame, plane)],
             "domain_size": len(demands[(frame, plane)])}
            for frame, plane in sorted(demands)
        ],
        "active_reading_count": len(demands),
        "multi_b_reading_count": sum(len(v) > 1 for v in demands.values()),
        "reading_domain_size_histogram": dict(sorted(Counter(map(len, demands.values())).items())),
        "axis_components": {str(axis): detail for axis, detail in axes.items()},
        "existing_last_writer_selected_b": selected_records,
    }


def build_census() -> dict[str, Any]:
    freeze_bytes = _checked_bytes(R4_FREEZE, R4_FREEZE_SHA256)
    exact_bytes = _checked_bytes(G1A_EXACT, G1A_EXACT_SHA256)
    baseline_bytes = _checked_bytes(G1A_BASELINE, G1A_BASELINE_SHA256)
    protected_bytes = _checked_bytes(G1A_PROTECTED, G1A_PROTECTED_SHA256)
    source_bytes = _checked_bytes(G1A_SOURCE, G1A_SOURCE_SHA256)
    tests_bytes = _checked_bytes(G1A_TESTS, G1A_TESTS_SHA256)
    diagnostic_bytes = _checked_bytes(G1A_SCRIPT, G1A_SCRIPT_SHA256)
    prereg_bytes = _checked_bytes(G1A_PREREG, G1A_PREREG_SHA256)
    freeze_path = ROOT / R4_FREEZE
    raw_freeze = json.loads(freeze_bytes.decode("utf-8"))
    sidecar_info = raw_freeze.get("mask_sidecar", {})
    sidecar_name = sidecar_info.get("file")
    if (not isinstance(sidecar_name, str) or Path(sidecar_name).name != sidecar_name
            or sidecar_info.get("sha256") != R4_SIDECAR_SHA256):
        raise ValueError("R4 mask sidecar descriptor mismatch")
    sidecar_path = R4_FREEZE.parent / sidecar_name
    sidecar_bytes = _checked_bytes(sidecar_path, R4_SIDECAR_SHA256)
    if len(sidecar_bytes) != sidecar_info.get("size_bytes"):
        raise ValueError("R4 mask sidecar length mismatch")
    document = frozen_image_runner.load_and_verify_frozen_image(freeze_path)
    if document["artifact_id"] != G1A_ARTIFACT_ID or document["labels_present"] is not False:
        raise ValueError("freeze identity/label gate mismatch")
    if document["inputs"] != EXPECTED_INPUTS:
        raise ValueError("ordered R4 image identity mismatch")
    exact = json.loads(exact_bytes.decode("utf-8"))
    baseline = json.loads(baseline_bytes.decode("utf-8"))
    if exact["freeze_sha256"] != _sha256(freeze_bytes) or exact["artifact_id"] != G1A_ARTIFACT_ID:
        raise ValueError("G1A exact diagnostic does not bind to this R4 freeze")
    if baseline["freeze_sha256"] != _sha256(freeze_bytes) or baseline["artifact_id"] != G1A_ARTIFACT_ID:
        raise ValueError("G1A baseline does not bind to this R4 freeze")
    if exact["inputs"] != EXPECTED_INPUTS or baseline["inputs"] != EXPECTED_INPUTS:
        raise ValueError("G1A diagnostic image identity mismatch")

    protected = json.loads(protected_bytes.decode("utf-8"))
    if protected.get("all_unchanged") is not True or protected.get("protected_count") != 20:
        raise ValueError("accepted G1A protected hash record is not intact")
    changed = []
    for rel, expected in protected["post"].items():
        path = ROOT / Path(rel)
        if not path.is_file() or _sha256(path.read_bytes()) != expected:
            changed.append(rel)
    if changed:
        raise ValueError(f"protected R1-R4/E1-E3/image hashes changed: {changed}")

    frame_records = document["frames"]
    frames = [str(frame["frame_id"]) for frame in frame_records]
    if frames != exact["solution"]["frames"]:
        raise ValueError("frame order differs from G1A exact diagnostic")
    intrinsic: dict[tuple[str, int], list[float]] = {}
    per_frame_readings = {}
    for frame in frame_records:
        name = str(frame["frame_id"])
        ordered = sorted(frame["readings"], key=_reading_key)
        per_frame_readings[name] = ordered
        for index, reading in enumerate(ordered):
            intrinsic[(name, index)] = sorted({float(item["blocks_per_period"])
                                               for item in reading["alternatives"]})
    matches = document["graph"]["matches"]

    all_demands: dict[tuple[str, int], set[float]] = defaultdict(set)
    all_incidents: Counter[tuple[str, int]] = Counter()
    all_groups: dict[tuple, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, match in enumerate(matches):
        left = _node(match["left"], match["left_plane"])
        right = _node(match["right"], match["right_plane"])
        if left not in intrinsic or right not in intrinsic:
            raise ValueError(f"match {index} references absent reading")
        all_demands[left].add(float(match["left_b"]))
        all_demands[right].add(float(match["right_b"]))
        all_incidents[left] += 1
        all_incidents[right] += 1
        all_groups[_group_key(match)].append((index, match))

    all_match = {
        "match_record_count": len(matches),
        "unique_pair_quadrant_group_count": len(all_groups),
        "duplicate_pair_quadrant_group_count": sum(len(v) > 1 for v in all_groups.values()),
        "duplicate_pair_quadrant_groups": [
            {"left": key[0], "left_plane": key[1], "right": key[2],
             "right_plane": key[3], "quadrant_of_right": key[4],
             "match_records": [{"index": index, "left_b": m["left_b"],
                                "right_b": m["right_b"], "score": m["score"]}
                               for index, m in rows]}
            for key, rows in sorted(all_groups.items()) if len(rows) > 1
        ],
        "matched_reading_count": len(all_demands),
        "multi_b_reading_count": sum(len(v) > 1 for v in all_demands.values()),
        "reading_domain_size_histogram": dict(sorted(Counter(map(len, all_demands.values())).items())),
        "readings": [
            {"frame": frame, "plane_index": plane,
             "axis_local": int(per_frame_readings[frame][plane]["axis"]),
             "intrinsic_alternative_b_domain": intrinsic[(frame, plane)],
             "match_demand_b_domain": sorted(all_demands[(frame, plane)]),
             "match_record_incident_count": all_incidents[(frame, plane)]}
            for frame, plane in sorted(all_demands)
        ],
    }

    best_vec = exact["solution"]["best"]["quadrants"]
    runner_vec = exact["solution"]["runner_up"]["quadrants"]
    best_q = dict(zip(frames, best_vec))
    runner_q = dict(zip(frames, runner_vec))
    best = _assignment_census("best", best_q, frames, matches, intrinsic)
    runner = _assignment_census("runner_up", runner_q, frames, matches, intrinsic)
    if best["active_readings"] != runner["active_readings"]:
        raise ValueError("best/runner-up b demands differ unexpectedly")
    best_indices = set(best["active_match_record_indices"])
    runner_indices = set(runner["active_match_record_indices"])
    best_semantic_edges = sorted(_semantic_edge_key(matches[index]) for index in best_indices)
    runner_semantic_edges = sorted(_semantic_edge_key(matches[index]) for index in runner_indices)
    if best_semantic_edges != runner_semantic_edges:
        raise ValueError("best/runner-up active match semantics differ beyond quadrant tags")

    expected_selected = exact["solution"]["selected_b"]
    observed_selected = best["existing_last_writer_selected_b"]
    observed_projection = [
        {"frame": row["frame"], "plane_index": row["plane_index"],
         "blocks_per_period": row["blocks_per_period"]}
        for row in observed_selected
    ]
    if expected_selected != observed_projection:
        raise ValueError("reconstructed best-assignment last-writer values differ from G1A record")

    freeze_solution_b = document["graph"]["solution"]["blocks_per_period"]
    return {
        "schema_version": 1,
        "diagnostic": "M4-G1B read-only b-domain census",
        "labels_present": False,
        "frozen": True,
        "inputs": {
            "r4_freeze_sha256": _sha256(freeze_bytes),
            "r4_mask_sidecar_sha256": R4_SIDECAR_SHA256,
            "r4_artifact_id": G1A_ARTIFACT_ID,
            "g1a_exact_sha256": _sha256(exact_bytes),
            "g1a_baseline_sha256": _sha256(baseline_bytes),
            "g1a_protected_record_sha256": _sha256(protected_bytes),
            "g1a_plane_graph_sha256": _sha256(source_bytes),
            "g1a_plane_graph_tests_sha256": _sha256(tests_bytes),
            "g1a_diagnostic_script_sha256": _sha256(diagnostic_bytes),
            "g1a_preregistration_sha256": _sha256(prereg_bytes),
            "protected_artifacts_verified": protected["protected_count"],
            "ordered_inputs": EXPECTED_INPUTS,
        },
        "graph_counts": {
            "frames": len(frames), "readings": len(intrinsic),
            "matches": len(matches), "refusals": len(document["graph"]["refusals"]),
        },
        "all_match_hypotheses": all_match,
        "best_assignment": best,
        "runner_up_assignment": runner,
        "best_runner_up_comparison": {
            "quadrant_frames_different": [frame for frame in frames
                                          if best_q[frame] != runner_q[frame]],
            "best_active_match_record_count": len(best_indices),
            "runner_up_active_match_record_count": len(runner_indices),
            "active_record_index_symmetric_difference_count": len(best_indices ^ runner_indices),
            "same_active_match_record_set": best_indices == runner_indices,
            "same_active_match_semantics_ignoring_quadrant_tag": True,
            "same_per_reading_b_domains": best["active_readings"] == runner["active_readings"],
            "best_cost": exact["solution"]["best"]["cost"],
            "runner_up_cost": exact["solution"]["runner_up"]["cost"],
            "cost_margin": exact["solution"]["best_runner_up_margin"],
        },
        "secondary_serialization_finding": {
            "freeze_solution_blocks_key_name": "axis",
            "solver_mapping_key_semantics": "(frame, plane_index)",
            "g1a_diagnostic_key_name": "plane_index",
            "freeze_serialized_item_count": len(freeze_solution_b),
            "finding": "frozen_image_runner._solution_record writes tuple key[1] as axis even though ComponentSolution.blocks_per_period is keyed by plane_index; diagnostic consumers must not treat the freeze field as an axis mapping or repair historical freeze bytes",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("exports/evaluation/m4-g1b-r4-b-census.json"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite census output: {output}")
    result = build_census()
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"output": str(output.relative_to(ROOT)),
                      "bytes": len(payload), "sha256": _sha256(payload),
                      "graph_counts": result["graph_counts"],
                      "all": {"multi_b_readings": result["all_match_hypotheses"]["multi_b_reading_count"],
                              "duplicate_groups": result["all_match_hypotheses"]["duplicate_pair_quadrant_group_count"]},
                      "best": {"active_matches": result["best_assignment"]["active_match_record_count"],
                               "readings": result["best_assignment"]["active_reading_count"],
                               "multi_b_readings": result["best_assignment"]["multi_b_reading_count"],
                               "axis_products": {axis: detail["match_demand_state_product"]
                                                 for axis, detail in result["best_assignment"]["axis_components"].items()}},
                      "runner_up_same_domains": result["best_runner_up_comparison"]["same_per_reading_b_domains"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
