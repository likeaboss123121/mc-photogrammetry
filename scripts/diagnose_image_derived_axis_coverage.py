"""Output-neutral diagnostics for image-derived plane-axis coverage."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Iterable

from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_production as production_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import plane_graph, plane_placement

MAX_SHIFT_CELLS = 32
INTEGER_TOLERANCE = 0.15
MIN_RESULTANT = 0.90
AMBIGUOUS_MARGIN = 0.02

DIAGNOSTIC_SCHEMA = "mc-photogrammetry-image-derived-axis-diagnostic-2"

ABLATION_NOTE = (
    "The constraint_ablation section re-solves the identical captured readings "
    "and ordered matches with only quadrant_constraints changed to an empty tuple. "
    "It measures the contribution of the connection-graph yaw constraints to the "
    "component solver only; it is not camera accuracy, image causality or a "
    "fixture measurement.")


def _snapshot(result: production_mod.ImageDerivedProductionResult) -> tuple:
    return (result.stage, result.status, result.reason, result.detail,
            result.result_sha256, result.entry.graph.graph_sha256)


def _counts(values: Iterable[int]) -> dict[str, int]:
    counts = {str(axis): 0 for axis in range(3)}
    for value in values:
        counts[str(int(value))] += 1
    return counts


def _matched_counts(matches: tuple, quadrants: dict[str, int]) -> dict[str, int]:
    axes = []
    for match in matches:
        left_q = quadrants.get(match.left)
        right_q = quadrants.get(match.right)
        if left_q is None or right_q is None:
            continue
        if (right_q - left_q) % 4 != match.quadrant_of_right:
            continue
        axis, _sign = plane_graph.axis_transform(match.left_axis, left_q)
        axes.append(axis)
    return _counts(axes)


def _match_key(match) -> tuple:
    return (match.left, match.right, match.left_plane, match.right_plane,
            match.left_axis, match.right_axis, float(match.left_b),
            float(match.right_b), match.quadrant_of_right,
            tuple(match.cell_shift_rc or ()), -float(match.score),
            -float(match.psr), -float(match.peak), -float(match.second_peak))


def _solver_payload(solution) -> dict:
    return {
        "fired_gate": solution.reason or "assembled",
        "matched_axes": sorted(int(axis) for axis in solution.axis_resultants),
        "resultants": {str(axis): float(value) for axis, value in
                       sorted(solution.axis_resultants.items())},
        "residuals": {str(axis): float(value) for axis, value in
                      sorted(solution.axis_residuals.items())},
        "quadrants": {str(frame): int(value) for frame, value in
                      sorted(solution.quadrants.items())},
    }


def _endpoint_payload(reading, frame: str, plane: int) -> dict:
    evidence = getattr(reading, "region_evidence", None)
    source = None if evidence is None else evidence.source_region_id
    return {"frame": frame, "plane": int(plane),
            "source_region_id": "" if source is None else str(source)}


def _hypothesis_payload(diagnostic) -> dict:
    return {
        "yaw_class": int(diagnostic.yaw_class),
        "support": int(diagnostic.support),
        "spatial_bins": int(diagnostic.spatial_bins),
        "passed": bool(diagnostic.passed),
        "refusal": str(diagnostic.refusal),
    }


def _pair_evidence_payload(graph) -> tuple[list[dict], int]:
    """Per-pair status/reason/support copied from the carried connection graph."""
    records: list[dict] = []
    solved_edges = 0
    evidence = () if graph is None else tuple(getattr(graph, "evidence", ()) or ())
    for record in sorted(evidence, key=lambda item: (item.left_id, item.right_id)):
        if record.status == "solved":
            solved_edges += 1
        records.append({
            "left": str(record.left_id),
            "right": str(record.right_id),
            "status": str(record.status),
            "reason": str(record.reason),
            "support": int(record.support),
            "support_floor": int(record.support_floor),
            "match_count": int(record.match_count),
            "candidate_edges": int(record.candidate_edges),
            "mutual_survivors": int(record.mutual_survivors),
            "right_yaw_class": (None if record.right_yaw_class is None
                                else int(record.right_yaw_class)),
            "hypotheses": [_hypothesis_payload(item)
                           for item in record.hypotheses],
        })
    return records, solved_edges


def _constraint_rows(constraints) -> list[list]:
    return [[str(left), str(right), int(relative)]
            for left, right, relative in constraints]


def constraint_ablation(readings, matches, constraints=()) -> dict:
    """Solve one captured component with and without the graph yaw constraints.

    Both solves consume the identical readings and identical ordered matches and
    the fixed solver thresholds; the only difference is ``quadrant_constraints``.
    """
    ordered = tuple(matches)
    baseline = plane_graph.solve_component(
        readings, ordered,
        integer_tolerance=INTEGER_TOLERANCE,
        min_resultant=MIN_RESULTANT,
        ambiguous_margin=AMBIGUOUS_MARGIN,
        quadrant_constraints=tuple(constraints))
    ablated = plane_graph.solve_component(
        readings, ordered,
        integer_tolerance=INTEGER_TOLERANCE,
        min_resultant=MIN_RESULTANT,
        ambiguous_margin=AMBIGUOUS_MARGIN,
        quadrant_constraints=())
    baseline_payload = _solver_payload(baseline)
    ablated_payload = _solver_payload(ablated)
    return {"baseline": baseline_payload, "ablated": ablated_payload,
            "changed": baseline_payload != ablated_payload}


def _selected_match_diagnostics(readings: tuple, matches: tuple,
                                constraints: tuple, solution) -> list[dict]:
    """Describe selected evidence and deterministic leave-one-out sensitivity."""
    ordered = tuple(sorted(matches, key=_match_key))
    index, _by_object, _frames = plane_graph._canonical_reading_index(readings)
    annotated = plane_graph.annotate_integer_residuals(
        solution, readings, ordered)
    endpoints = []
    for match in ordered:
        endpoints.append(frozenset(((match.left, match.left_axis),
                                    (match.right, match.right_axis))))
    result = []
    for position, (match, residual) in enumerate(zip(ordered, annotated)):
        competitors = []
        for other_position, other in enumerate(ordered):
            if other_position != position and endpoints[position] & endpoints[other_position]:
                competitors.append({
                    "index": other_position, "left": other.left,
                    "right": other.right, "left_plane": other.left_plane,
                    "right_plane": other.right_plane,
                    "left_axis": other.left_axis, "right_axis": other.right_axis,
                    "score": float(other.score)})
        leave_one_out = ordered[:position] + ordered[position + 1:]
        loo_solution = plane_graph.solve_component(
            readings, leave_one_out,
            integer_tolerance=INTEGER_TOLERANCE,
            min_resultant=MIN_RESULTANT,
            ambiguous_margin=AMBIGUOUS_MARGIN,
            quadrant_constraints=constraints)
        left_reading = index[(match.left, match.left_plane)]
        right_reading = index[(match.right, match.right_plane)]
        result.append({
            "index": position,
            "left": _endpoint_payload(left_reading, match.left, match.left_plane),
            "right": _endpoint_payload(right_reading, match.right, match.right_plane),
            "left_axis": int(match.left_axis),
            "right_axis": int(match.right_axis),
            "left_b": float(match.left_b), "right_b": float(match.right_b),
            "quadrant_of_right": int(match.quadrant_of_right),
            "score": float(match.score), "psr": float(match.psr),
            "peak": float(match.peak), "second_peak": float(match.second_peak),
            "overlap_cells": int(match.overlap_cells),
            "cell_shift_rc": (None if match.cell_shift_rc is None
                              else list(match.cell_shift_rc)),
            "circular_phase_residual": (None if residual.integer_residual is None
                                        else float(residual.integer_residual)),
            "competing_candidates": competitors,
            "leave_one_out": _solver_payload(loo_solution),
        })
    return result


def _base(result: production_mod.ImageDerivedProductionResult) -> dict:
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "normal_mode": {
            "stage": result.stage, "status": result.status,
            "reason": result.reason, "detail": result.detail,
            "result_sha256": result.result_sha256,
            "graph_sha256": result.entry.graph.graph_sha256,
            "unchanged": False,
        },
        "thresholds": {
            "max_shift_cells": MAX_SHIFT_CELLS,
            "integer_tolerance": INTEGER_TOLERANCE,
            "min_resultant": MIN_RESULTANT,
            "ambiguous_margin": AMBIGUOUS_MARGIN,
        },
        "interpretation": ABLATION_NOTE,
        "reading_counts": {str(axis): 0 for axis in range(3)},
        "matched_pair_counts": {str(axis): 0 for axis in range(3)},
        "matched_axes": [], "resultants": {}, "residuals": {},
        "quadrants": {}, "selected_matches": [],
        "pair_evidence": [], "solved_edge_count": 0,
        "yaw_constraints": [], "yaw_constraint_count": 0,
        "constraint_ablation": None,
        "fired_gate": "",
    }


def _replay(result: production_mod.ImageDerivedProductionResult):
    return production_mod.ImageDerivedProductionResult(**{
        field.name: getattr(result, field.name)
        for field in dataclasses.fields(production_mod.ImageDerivedProductionResult)})


def diagnose_production_result(
    result: production_mod.ImageDerivedProductionResult,
) -> dict:
    """Inspect one normal-mode result without changing any carried record."""
    if not isinstance(result, production_mod.ImageDerivedProductionResult):
        raise ValueError("expected an image-derived production result")
    before = _snapshot(result)
    replayed = _replay(result)
    payload = _base(result)
    if replayed.lattice is None or replayed.lattice.status != "assembled":
        payload["fired_gate"] = (replayed.reason or
                                  ("entry_unavailable" if replayed.lattice is None
                                   else "lattice_" + replayed.lattice.reason))
    else:
        lattice = replayed.lattice
        readings = tuple(lattice.readings)
        payload["reading_counts"] = _counts(reading.axis for reading in readings)
        signatures = signatures_mod.derive_image_derived_signatures(
            session=result.entry.session, lattice_result=lattice)
        if isinstance(signatures, plane_placement.PlacementRefusal):
            payload["fired_gate"] = "signatures_" + signatures.reason
        elif signatures.status != "assembled":
            payload["fired_gate"] = "signatures_" + signatures.reason
        else:
            readings, evidence = graph_mod._ordered_universe(lattice, signatures)
            matches = plane_graph.match_plane_graph(
                readings, tuple(item.signature for item in evidence),
                max_shift_cells=MAX_SHIFT_CELLS)
            constraints = graph_mod._connection_constraints(
                result.entry.graph, result.entry.session, lattice)
            if isinstance(constraints, plane_placement.PlacementRefusal):
                payload["fired_gate"] = "connection_" + constraints.reason
            else:
                pair_evidence, solved_edges = _pair_evidence_payload(
                    result.entry.graph)
                payload["pair_evidence"] = pair_evidence
                payload["solved_edge_count"] = solved_edges
                payload["yaw_constraints"] = _constraint_rows(constraints)
                payload["yaw_constraint_count"] = len(constraints)
                if not matches:
                    payload["fired_gate"] = "no_matches"
                else:
                    solution = plane_graph.solve_component(
                        readings, matches,
                        integer_tolerance=INTEGER_TOLERANCE,
                        min_resultant=MIN_RESULTANT,
                        ambiguous_margin=AMBIGUOUS_MARGIN,
                        quadrant_constraints=constraints)
                    quadrants = {str(key): int(value) for key, value in
                                 sorted(solution.quadrants.items())}
                    payload["quadrants"] = quadrants
                    payload["matched_pair_counts"] = _matched_counts(
                        tuple(matches), quadrants)
                    payload["matched_axes"] = sorted(
                        int(axis) for axis in solution.axis_resultants)
                    payload["resultants"] = {
                        str(axis): float(value) for axis, value in
                        sorted(solution.axis_resultants.items())}
                    payload["residuals"] = {
                        str(axis): float(value) for axis, value in
                        sorted(solution.axis_residuals.items())}
                    payload["selected_matches"] = _selected_match_diagnostics(
                        tuple(readings), tuple(matches), tuple(constraints),
                        solution)
                    payload["constraint_ablation"] = constraint_ablation(
                        tuple(readings), tuple(matches), tuple(constraints))
                    payload["fired_gate"] = solution.reason or "assembled"
    after = _snapshot(result)
    if before != after:
        raise AssertionError("diagnostic changed the normal-mode result")
    payload["normal_mode"]["unchanged"] = True
    return payload


def diagnose_image_derived_axis_coverage(
    image_inputs: Iterable[str | Path], *, project_path: str | Path,
    recursive: bool = False,
) -> dict:
    """Run normal mode and return its deterministic, output-neutral diagnostic."""
    result = production_mod.derive_image_output_from_paths(
        image_inputs, project_path=project_path, recursive=recursive)
    if isinstance(result, plane_placement.PlacementRefusal):
        return {
            "schema": DIAGNOSTIC_SCHEMA,
            "normal_mode": {"stage": "refusal", "status": "refused",
                            "reason": result.reason,
                            "detail": result.detail, "result_sha256": "",
                            "graph_sha256": "", "unchanged": True},
            "thresholds": {
                "max_shift_cells": MAX_SHIFT_CELLS,
                "integer_tolerance": INTEGER_TOLERANCE,
                "min_resultant": MIN_RESULTANT,
                "ambiguous_margin": AMBIGUOUS_MARGIN},
            "interpretation": ABLATION_NOTE,
            "reading_counts": {str(axis): 0 for axis in range(3)},
            "matched_pair_counts": {str(axis): 0 for axis in range(3)},
            "matched_axes": [], "resultants": {}, "residuals": {},
            "quadrants": {}, "selected_matches": [],
            "pair_evidence": [], "solved_edge_count": 0,
            "yaw_constraints": [], "yaw_constraint_count": 0,
            "constraint_ablation": None,
            "fired_gate": "refusal_" + result.reason,
        }
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
    payload = diagnose_image_derived_axis_coverage(
        arguments.images, project_path=arguments.project,
        recursive=arguments.recursive)
    sys.stdout.write(canonical_json(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
