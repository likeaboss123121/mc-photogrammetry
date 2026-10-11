"""Output-neutral cycle-consistency audit for selected image-derived matches."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_production as production_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import plane_graph, plane_placement

INTEGER_TOLERANCE = 0.15
MAX_SHIFT_CELLS = 32
MIN_RESULTANT = 0.90
AMBIGUOUS_MARGIN = 0.02


def _match_key(match) -> tuple:
    return (match.left, match.right, match.left_plane, match.right_plane,
            match.left_axis, match.right_axis, float(match.left_b),
            float(match.right_b), match.quadrant_of_right,
            tuple(match.cell_shift_rc or ()), -float(match.score))


def _edge_records(readings: tuple, matches: tuple,
                  quadrants: dict[str, int]) -> tuple[dict, ...]:
    index, _objects, _frames = plane_graph._canonical_reading_index(readings)
    records = []
    for number, match in enumerate(sorted(matches, key=_match_key)):
        if match.left not in quadrants or match.right not in quadrants:
            continue
        if ((quadrants[match.right] - quadrants[match.left]) % 4
                != match.quadrant_of_right):
            continue
        left = index[(match.left, match.left_plane)]
        right = index[(match.right, match.right_plane)]
        axis, left_sign = plane_graph.axis_transform(
            left.axis, quadrants[match.left])
        right_axis, right_sign = plane_graph.axis_transform(
            right.axis, quadrants[match.right])
        left_distance = left.distance_for(match.left_b)
        right_distance = right.distance_for(match.right_b)
        if right_axis != axis or left_distance is None or right_distance is None:
            continue
        delta = (left_sign * left.side * left_distance
                 - right_sign * right.side * right_distance)
        shift_world = None
        if match.cell_shift_rc is not None:
            row_shift, column_shift = match.cell_shift_rc
            tangent_labels = tuple(label for label in range(3)
                                   if label != left.axis)
            vector = [0, 0, 0]
            for local_label, amount in zip(
                    tangent_labels, (column_shift, row_shift)):
                world_label, sign = plane_graph.axis_transform(
                    local_label, quadrants[match.left])
                vector[world_label] = int(sign * amount)
            shift_world = tuple(vector)
        records.append({
            "id": number, "axis": int(axis), "left": match.left,
            "right": match.right, "delta": float(delta),
            "cell_shift_rc": (None if match.cell_shift_rc is None
                              else tuple(match.cell_shift_rc)),
            "cell_shift_world": shift_world,
            "left_plane": match.left_plane, "right_plane": match.right_plane,
            "left_axis": match.left_axis, "right_axis": match.right_axis,
        })
    return tuple(records)


def _alternate_paths(edges: tuple[dict, ...], excluded: int) -> list[tuple]:
    target = edges[excluded]
    adjacency = defaultdict(list)
    for position, edge in enumerate(edges):
        if position == excluded or edge["axis"] != target["axis"]:
            continue
        adjacency[edge["left"]].append((edge["right"], edge["delta"], position, 1))
        adjacency[edge["right"]].append((edge["left"], -edge["delta"], position, -1))
    for frame in adjacency:
        adjacency[frame].sort(key=lambda item: (item[0], item[2], item[3]))
    found = []

    def visit(frame, visited, delta, path):
        if frame == target["right"]:
            found.append((tuple(path), float(delta)))
            return
        for other, step, position, direction in adjacency.get(frame, ()):
            if other in visited:
                continue
            visit(other, visited | {other}, delta + step,
                  path + ((position, direction),))

    visit(target["left"], {target["left"]}, 0.0, ())
    return sorted(found, key=lambda item: (len(item[0]), item[0]))


def _circular_residual(value: float) -> float:
    return float(abs((value + 0.5) % 1.0 - 0.5))


def audit_match_cycles(readings: tuple, matches: tuple,
                       constraints: tuple = ()) -> dict:
    """Audit alternate measured paths without deleting or selecting evidence."""
    ordered = tuple(sorted(matches, key=_match_key))
    solution = plane_graph.solve_component(
        readings, ordered, integer_tolerance=INTEGER_TOLERANCE,
        min_resultant=MIN_RESULTANT, ambiguous_margin=AMBIGUOUS_MARGIN,
        quadrant_constraints=constraints)
    edges = _edge_records(readings, ordered, dict(solution.quadrants))
    cycles = []
    for excluded, edge in enumerate(edges):
        for path, path_delta in _alternate_paths(edges, excluded):
            if not path:
                continue
            edge_ids = tuple(sorted({edge["id"]}
                                    | {edges[position]["id"]
                                       for position, _direction in path}))
            residual = _circular_residual(edge["delta"] - path_delta)
            shifts = [edge["cell_shift_rc"]] + [
                edges[position]["cell_shift_rc"] for position, _direction in path]
            shift_closure = None
            if (edge["cell_shift_world"] is not None
                    and all(edges[position]["cell_shift_world"] is not None
                            for position, _direction in path)):
                path_shift = [0, 0, 0]
                for position, direction in path:
                    for axis, amount in enumerate(
                            edges[position]["cell_shift_world"]):
                        path_shift[axis] += direction * amount
                shift_closure = [edge["cell_shift_world"][axis]
                                 - path_shift[axis] for axis in range(3)]
            cycles.append({
                "axis": edge["axis"], "edge": edge["id"],
                "path": [[edges[position]["id"], direction]
                         for position, direction in path],
                "edge_ids": list(edge_ids), "path_delta": path_delta,
                "edge_delta": edge["delta"], "closure_residual": residual,
                "consistent": residual <= INTEGER_TOLERANCE,
                "cell_shifts": [None if shift is None else list(shift)
                                for shift in shifts],
                "cell_shift_closure_world": shift_closure,
            })
    unique_cycles = {}
    for cycle in cycles:
        key = (cycle["axis"], tuple(cycle["edge_ids"]))
        previous = unique_cycles.get(key)
        if previous is None or tuple(cycle["path"]) < tuple(previous["path"]):
            unique_cycles[key] = cycle
    cycles = [unique_cycles[key] for key in sorted(unique_cycles)]
    failing = [cycle for cycle in cycles if not cycle["consistent"]]
    common = (set(failing[0]["edge_ids"]) if failing else set())
    for cycle in failing[1:]:
        common &= set(cycle["edge_ids"])
    duplicates = defaultdict(list)
    for edge in edges:
        fingerprint = (edge["axis"], edge["left"], edge["right"],
                       edge["delta"], edge["cell_shift_rc"],
                       edge["cell_shift_world"])
        duplicates[fingerprint].append(edge["id"])
    correlated = [sorted(values) for values in duplicates.values()
                  if len(values) > 1]
    unique_inconsistent = sorted(common) if len(common) == 1 else []
    if not cycles:
        status, abstention = "abstained", "insufficient_independent_cycles"
    elif correlated:
        status, abstention = "abstained", "correlated_duplicate_paths"
    elif not failing:
        status, abstention = "consistent", ""
    elif len(common) != 1:
        status, abstention = "abstained", "equally_plausible_conflict"
    else:
        status, abstention = "inconsistent", ""
    return {
        "status": status, "abstention": abstention,
        "solver_gate": solution.reason or "assembled",
        "path_count": len(cycles), "closure_tolerance": INTEGER_TOLERANCE,
        "cycles": cycles, "uniquely_inconsistent_edges": unique_inconsistent,
        "correlated_edge_groups": sorted(correlated),
    }


def _snapshot(result) -> tuple:
    return (result.stage, result.status, result.reason, result.result_sha256,
            result.entry.graph.graph_sha256)


def diagnose_production_result(result) -> dict:
    if not isinstance(result, production_mod.ImageDerivedProductionResult):
        raise ValueError("expected an image-derived production result")
    before = _snapshot(result)
    replayed = production_mod.ImageDerivedProductionResult(**{
        field.name: getattr(result, field.name)
        for field in dataclasses.fields(production_mod.ImageDerivedProductionResult)})
    payload = {
        "schema": "mc-photogrammetry-image-derived-match-cycles-1",
        "normal_mode": {"stage": replayed.stage, "status": replayed.status,
                        "reason": replayed.reason,
                        "result_sha256": replayed.result_sha256,
                        "unchanged": False},
        "audit": {"status": "abstained", "abstention": replayed.reason,
                  "solver_gate": replayed.reason, "path_count": 0,
                  "closure_tolerance": INTEGER_TOLERANCE, "cycles": [],
                  "uniquely_inconsistent_edges": [],
                  "correlated_edge_groups": []},
    }
    if replayed.lattice is not None and replayed.lattice.status == "assembled":
        signatures = signatures_mod.derive_image_derived_signatures(
            session=result.entry.session, lattice_result=replayed.lattice)
        if (not isinstance(signatures, plane_placement.PlacementRefusal)
                and signatures.status == "assembled"):
            readings, evidence = graph_mod._ordered_universe(
                replayed.lattice, signatures)
            matches = plane_graph.match_plane_graph(
                readings, tuple(item.signature for item in evidence),
                max_shift_cells=MAX_SHIFT_CELLS)
            constraints = graph_mod._connection_constraints(
                result.entry.graph, result.entry.session, replayed.lattice)
            if not isinstance(constraints, plane_placement.PlacementRefusal):
                payload["audit"] = audit_match_cycles(
                    tuple(readings), tuple(matches), tuple(constraints))
    if before != _snapshot(result):
        raise AssertionError("cycle audit changed the normal-mode result")
    payload["normal_mode"]["unchanged"] = True
    return payload


def diagnose_image_derived_match_cycles(
    image_inputs: Iterable[str | Path], *, project_path: str | Path,
    recursive: bool = False,
) -> dict:
    result = production_mod.derive_image_output_from_paths(
        image_inputs, project_path=project_path, recursive=recursive)
    if isinstance(result, plane_placement.PlacementRefusal):
        return {"schema": "mc-photogrammetry-image-derived-match-cycles-1",
                "normal_mode": {"stage": "refusal", "status": "refused",
                                "reason": result.reason, "result_sha256": "",
                                "unchanged": True},
                "audit": {"status": "abstained",
                          "abstention": "refusal_" + result.reason,
                          "solver_gate": "", "path_count": 0,
                          "closure_tolerance": INTEGER_TOLERANCE, "cycles": [],
                          "uniquely_inconsistent_edges": [],
                          "correlated_edge_groups": []}}
    return diagnose_production_result(result)


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--project", required=True)
    parser.add_argument("--recursive", action="store_true")
    arguments = parser.parse_args(argv)
    payload = diagnose_image_derived_match_cycles(
        arguments.images, project_path=arguments.project,
        recursive=arguments.recursive)
    sys.stdout.write(canonical_json(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
