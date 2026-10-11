"""One-pass scalar partial-core diagnostic. Tests use synthetic stage stubs only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

# This venv's editable-install metadata may point at an older workspace. Insert
# this checkout's source before any package import so that code identity follows
# the packet's source pins.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mc_photogrammetry.pipeline.image_derived_entry import build_image_derived_entry
from mc_photogrammetry.pipeline.image_derived_lattice_measure import measure_image_derived_lattice
from mc_photogrammetry.pipeline.image_derived_signatures import derive_image_derived_signatures
from mc_photogrammetry.pipeline.image_derived_partial_joint_graph import derive_partial_image_derived_graph
from mc_photogrammetry.pipeline.image_derived_partial_axis_support import derive_partial_core_axis_support
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as endpoint_selection


INPUT_SEALS = (
    {"frame": "image-0000", "ordinal": 0, "path": "exports/datasets/pilot-batch-train-station-2026-09-12/scenes/train-station-spawn-208-96-orbit-a/captures/view-train-station-spawn-208-96-orbit-a-00.png", "bytes": 1267351, "sha256": "19E7399F449C339AC2DC38DD8F6B1C549E1616BE010B0682BB15CA26D8D99389"},
    {"frame": "image-0001", "ordinal": 1, "path": "exports/datasets/pilot-batch-train-station-2026-09-12/scenes/train-station-spawn-208-96-orbit-a/captures/view-train-station-spawn-208-96-orbit-a-01.png", "bytes": 770549, "sha256": "AA33D6B6DDEE52F071629C5EAC3116EC6F82467FF1285285122F4549E94B67E4"},
    {"frame": "image-0002", "ordinal": 2, "path": "exports/datasets/pilot-batch-train-station-2026-09-12/scenes/train-station-spawn-208-96-orbit-a/captures/view-train-station-spawn-208-96-orbit-a-02.png", "bytes": 960653, "sha256": "005E29A296BD096544100CA132D5ECC813451C60E7D9EF41A778323546EC3BF8"},
    {"frame": "image-0003", "ordinal": 3, "path": "exports/datasets/pilot-batch-train-station-2026-09-12/scenes/train-station-spawn-208-96-orbit-a/captures/view-train-station-spawn-208-96-orbit-a-03.png", "bytes": 947333, "sha256": "8E774CA86D4B305E1E177879D2BFD1529BA64D77EDAB5831412B3C433540E9AD"},
    {"frame": "image-0004", "ordinal": 4, "path": "exports/datasets/pilot-batch-train-station-2026-09-12/scenes/train-station-spawn-208-96-orbit-a/captures/view-train-station-spawn-208-96-orbit-a-04.png", "bytes": 982942, "sha256": "370F19E63486F7FA5D1A2835E3B621ED3F39FBD48B30B7447A2EC26E614B8148"},
    {"frame": "image-0005", "ordinal": 5, "path": "exports/datasets/pilot-batch-train-station-2026-09-12/scenes/train-station-spawn-208-96-orbit-a/captures/view-train-station-spawn-208-96-orbit-a-05.png", "bytes": 1634954, "sha256": "1E2319D0F01EF100E183AA8C8DE5FB0751F9A67078681E49938CC6EC7E3D92D2"},
)
ENDPOINT_BUDGET_CATEGORIES = frozenset({
    "endpoint_variables", "frames", "alternatives_per_endpoint", "input_match_rows",
    "registry_factors", "quadrant_assignments", "branch_nodes", "relation_visits",
    "solution_rows", "scalar_payload_bytes", "scalar_row_bytes", "proof_bytes", "record_bytes",
})
RELATION_ATTRIBUTIONS = (
    "preflight_visits", "incidence_visits", "partial_support_visits",
    "partial_update_visits", "partial_queue_visits", "partial_journal_visits",
    "partial_undo_visits", "partial_event_visits", "physical_relation_visits",
    "physical_cycle_relaxations",
    "arc_relation_visits", "arc_queue_visits", "arc_incidence_visits",
)

REJECTION_NAMES = ("partial_checks", "partial_prunes", "physical_feasible", "physical_rejected",
                   "retained_partial_exclusions", "omitted_partial_exclusions",
                   "retained_physical_exclusions", "omitted_physical_exclusions",
                   "retained_exclusions", "omitted_exclusions")
WITNESS_ROWS = 4
WITNESS_ROW_BYTES = 1024
REJECTION_BYTES = 8192
WITNESS_FACTOR_VISITS = 512
WITNESS_SOURCE_VISITS = 2048


def _encoded_size(value):
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _forced_witnesses(selection, proof, counts):
    """Project existing source references, never evaluate a physical relation.

    Endpoint and frame ordinals address the sealed certificate; only the two
    original source rows establishing each diameter are projected. Omitted rows
    remain evidence in the certificate, not reusable conflict-learning clauses.
    """
    events = proof["partial_exclusions"]
    if type(events) is not list or len(events) != counts["retained_partial_exclusions"]:
        raise ValueError("partial event census")
    def digest(value):
        return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if (proof["transcript_policy"] != endpoint_selection.partial_axes.TRANSCRIPT_POLICY
            or not digest(proof["physical_digest"])
            or proof["source_digest"] is not None and not digest(proof["source_digest"])):
        raise ValueError("witness provenance")
    rows, factor_visits, source_visits = [], 0, 0
    if events and type(selection) is not endpoint_selection.PartialEndpointSelectionCertificate:
        raise ValueError("witness requires source-bound certificate")
    if events and (not digest(selection.result_sha256) or not digest(proof["source_digest"])):
        raise ValueError("source certificate provenance")
    for ordinal, event in enumerate(events[:WITNESS_ROWS]):
        if type(event) is not list or len(event) != 6 or event[0] != "partial":
            raise ValueError("partial event shape")
        _, q_index, _, masks, factors, extrema = event
        group, low, high, lo, hi = extrema
        if (type(q_index) is not int or not 0 <= q_index < proof["counters"]["visited_assignments"]
                or type(low) is not float or type(high) is not float
                or not math.isfinite(low) or not math.isfinite(high)
                or high - low <= proof["physical_tolerance"]):
            raise ValueError("partial diameter")
        frames = selection.frame_ids
        bindings = selection.reading_bindings
        wanted = {lo[0], hi[0]}
        found = {}
        exhausted = False
        for factor in factors:
            if factor_visits == WITNESS_FACTOR_VISITS:
                exhausted = True
                break
            factor_visits += 1
            fid, u, a, v, b, source_rows = factor
            for source_row in source_rows:
                if source_visits == WITNESS_SOURCE_VISITS:
                    exhausted = True
                    break
                source_visits += 1
                if source_row not in wanted:
                    continue
                ref = lo if source_row == lo[0] else hi
                if (any(type(n) is not int for n in (fid, u, a, v, b, source_row))
                        or not 0 <= fid < proof["counters"]["registry_factors"]
                        or not 0 <= u < v < len(bindings)
                        or not 0 <= a < len(bindings[u][4]) or not 0 <= b < len(bindings[v][4])
                        or type(masks[u]) is not int or type(masks[v]) is not int
                        or masks[u] != 1 << a or masks[v] != 1 << b
                        or not 0 <= source_row < selection.candidate_count
                        or (bindings[u][0][0], bindings[v][0][0]) != tuple(group[:2])
                        or any(type(n) is not int for n in ref[:4])
                        or ref[1:4] != [a, b, group[2]]
                        or type(ref[4]) is not float or ref[4] != (low if source_row == lo[0] else high)):
                    raise ValueError("source witness join")
                found[source_row] = {"factor_index": fid, "endpoints": [u, v],
                    "selected_alternatives": [a, b], "source_row": source_row}
            if exhausted or wanted <= found.keys():
                break
        if exhausted:
            break
        if wanted != found.keys() or type(group[2]) is not int or group[2] not in range(3):
            raise ValueError("source witness missing")
        row = {"event_ordinal": ordinal, "quadrant_assignment_index": q_index,
               "physical_group": [frames.index(group[0]), frames.index(group[1]), group[2]],
               "extrema": [low, high], "extrema_source_rows": [lo[0], hi[0]],
               "factors": [found[i] for i in sorted(found)]}
        if _encoded_size(row) > WITNESS_ROW_BYTES:
            continue
        rows.append(row)
    return {"status": "complete" if len(rows) == len(events) else "capped",
            "binding": "existing_certificate_only_no_additional_source_replay",
            "certificate_sha256": getattr(selection, "result_sha256", None),
            "source_sha256": proof["source_digest"], "transcript_policy": proof["transcript_policy"],
            "physical_transcript_sha256": proof["physical_digest"], "rows": rows,
            "source_retained": len(events), "source_omitted": counts["omitted_partial_exclusions"],
            "observer_retained": len(rows), "observer_omitted": len(events) - len(rows),
            "factor_visits": factor_visits, "source_row_visits": source_visits}


def _endpoint_rejections(selection, proof, budget):
    """Diagnostic-only census from the one already-returned proof record."""
    try:
        counts = {name: proof["counters"][name] for name in REJECTION_NAMES}
        pruning = {name: proof["pruning"][name] for name in ("values_removed", "inconsistent_nodes")}
        reasons = proof["physical_reasons"]
        if (type(proof["physical_tolerance"]) is not float
                or proof["physical_tolerance"] != csp.plane_graph.SAME_AXIS_TOLERANCE
                or any(type(n) is not int or not 0 <= n <= budget.relation_visits for n in counts.values())
                or any(counts[name] > 128 for name in REJECTION_NAMES if name.startswith("retained_"))
                or type(pruning["values_removed"]) is not int
                or not 0 <= pruning["values_removed"] <= budget.endpoint_variables * budget.alternatives_per_endpoint * budget.branch_nodes
                or type(pruning["inconsistent_nodes"]) is not int
                or not 0 <= pruning["inconsistent_nodes"] <= budget.branch_nodes
                or counts["partial_prunes"] > counts["partial_checks"]
                or counts["retained_partial_exclusions"] + counts["omitted_partial_exclusions"] != counts["partial_prunes"]
                or counts["retained_physical_exclusions"] + counts["omitted_physical_exclusions"] != counts["physical_rejected"]
                or counts["retained_exclusions"] + counts["omitted_exclusions"] != proof["counters"]["excluded_assignments"]
                or counts["retained_partial_exclusions"] + counts["retained_physical_exclusions"] > 128
                or type(proof["physical_exclusions"]) is not list
                or len(proof["physical_exclusions"]) != counts["retained_physical_exclusions"]
                or counts["physical_feasible"] + counts["physical_rejected"] > proof["counters"]["physical_assignments"]
                or type(reasons) is not dict
                or any(name not in csp.PHYSICAL_REASON_CODES or name is None
                       or type(n) is not int or not 0 < n <= counts["physical_rejected"] for name, n in reasons.items())
                or sum(reasons.values()) != counts["physical_rejected"]):
            raise ValueError("rejection accounting")
        record = {"status": "complete", "reason": "", **counts, **pruning,
                  "physical_reasons": dict(reasons), "forced_witnesses": _forced_witnesses(selection, proof, counts)}
        if _encoded_size(record) > REJECTION_BYTES:
            raise ValueError("rejection report cap")
        return record
    except Exception:
        return {"status": "malformed", "reason": "invalid_rejection_evidence"}


def _endpoint_search(selection, budget):
    """Small exact whitelist, never the source proof or whole counters mapping."""
    csp.validate_budget(budget, partial=True)
    names = ("planned_assignments", "visited_assignments", "eligible_assignments",
             "branch_nodes", "physical_assignments", "relation_visits", "arc_factor_scans")
    empty = {name: None for name in names}
    empty.update(relation_visit_attribution={name: None for name in RELATION_ATTRIBUTIONS},
                 unattributed_relation_visits=None)
    if selection is None:
        return {"status": "not_reached", "reason": "no_selection", **empty}
    try:
        proof = selection.proof_record()
        counts = proof.get("counters") if type(proof) is dict else None
        if type(counts) is not dict:
            raise ValueError("missing counters")
        result = {name: counts[name] for name in names}
        attributed = {name: counts[name] for name in RELATION_ATTRIBUTIONS}
        caps = {"planned_assignments": budget.quadrant_assignments,
                "visited_assignments": budget.quadrant_assignments,
                "eligible_assignments": budget.quadrant_assignments,
                "branch_nodes": budget.branch_nodes,
                "physical_assignments": budget.branch_nodes,
                "relation_visits": budget.relation_visits, "arc_factor_scans": budget.relation_visits}
        for name, value in result.items():
            if value is None and name == "planned_assignments":
                continue
            if type(value) is not int or not 0 <= value <= caps[name]:
                raise ValueError("counter bound")
        if (any(type(value) is not int or value < 0 for value in attributed.values())
                or sum(attributed.values()) > result["relation_visits"]
                or result["eligible_assignments"] > result["visited_assignments"]
                or result["physical_assignments"] > result["branch_nodes"]
                or result["arc_factor_scans"] > attributed["arc_queue_visits"]
                or result["planned_assignments"] is not None
                   and result["visited_assignments"] > result["planned_assignments"]):
            raise ValueError("counter reconciliation")
        result["relation_visit_attribution"] = attributed
        result["unattributed_relation_visits"] = result["relation_visits"] - sum(attributed.values())
        rejections = _endpoint_rejections(selection, proof, budget)
        return {"status": rejections["status"], "reason": rejections["reason"], **result,
                "rejections": rejections}
    except Exception:
        return {"status": "malformed", "reason": "invalid_search_counters", **empty}


def _status_reason(value):
    if value is None:
        return {"status": "not_reached", "reason": ""}
    status = getattr(value, "status", None)
    reason = getattr(value, "reason", "")
    if status is None:
        status = "refused" if type(value).__name__ == "PlacementRefusal" else (
            "assembled" if not reason else "unavailable")
    record = {"status": str(status), "reason": str(reason or "")}
    if type(value).__name__ == "PlacementRefusal":
        record["result_type"] = "PlacementRefusal"
    return record


def _endpoint_budget(selection):
    """Expose only validated, whitelisted scalars from a budget-refusal proof."""
    empty = {"category": None, "used": None, "amount": None, "cap": None}
    if selection is None:
        return {"status": "not_reached", "reason": "no_selection", **empty}
    if getattr(selection, "status", None) != "budget_refused":
        return {"status": "not_reached", "reason": "selection_not_budget_refused", **empty}
    try:
        proof = selection.proof_record()
    except Exception:
        return {"status": "malformed", "reason": "proof_unavailable", **empty}
    denial = proof.get("denial") if type(proof) is dict else None
    if (type(denial) is not list or len(denial) != 4
            or type(denial[0]) is not str or denial[0] not in ENDPOINT_BUDGET_CATEGORIES
            or any(type(value) is not int or value < 0 for value in denial[1:])
            or denial[2] == 0 or denial[1] > denial[3]
            or denial[1] + denial[2] <= denial[3]):
        return {"status": "malformed", "reason": "invalid_denial", **empty}
    category, used, amount, cap = denial
    return {"status": "budget_refused", "reason": "", "category": category,
            "used": used, "amount": amount, "cap": cap}


def _elapsed(start):
    value = time.perf_counter() - start
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("nonfinite_stage_timing")
    return round(value, 6)


def _run_stage(stage, function, *args, **kwargs):
    started = time.perf_counter()
    try:
        value = function(*args, **kwargs)
        result = _status_reason(value)
    except Exception as error:
        value = None
        result = {"status": "raised", "reason": "stage_raised",
                  "error_type": type(error).__name__}
    return value, result, _elapsed(started)


def run_diagnostic(image_paths, project_path, report_path, *, budget=csp.EndpointBudget()):
    """Run each authorized stage once and write a scalar-only JSON report."""
    csp.validate_budget(budget, partial=True)
    paths = tuple(str(Path(path).resolve()) for path in image_paths)
    expected = tuple(str((REPO_ROOT / seal["path"]).resolve())
                     for seal in INPUT_SEALS)
    if paths != expected:
        raise ValueError("input_inventory_mismatch")

    entry, entry_status, entry_seconds = _run_stage(
        "entry", build_image_derived_entry, paths, project_path=project_path)
    connection_status = pose_status = {"status": "not_reached", "reason": ""}
    if entry is not None:
        connection_status = _status_reason(entry.graph)
        pose_status = _status_reason(entry.pose_graph)
        ready = (not entry.reason and connection_status["status"] == "ok"
                 and pose_status["status"] == "ok")
        entry_status = {"status": "ready" if ready else "unavailable",
                        "reason": "" if ready else str(entry.reason or connection_status["reason"] or pose_status["reason"])}
    lattice = signatures = graph = core_support = None
    lattice_status = signatures_status = graph_status = support_status = {"status": "not_reached", "reason": ""}
    lattice_seconds = signatures_seconds = graph_seconds = support_seconds = 0.0
    entry_ready = entry is not None and entry_status["status"] == "ready"
    if entry_ready:
        lattice, lattice_status, lattice_seconds = _run_stage(
            "lattice", measure_image_derived_lattice, session=entry.session, pose_graph=entry.pose_graph)
    if entry_ready and lattice is not None and getattr(lattice, "status", None) == "assembled":
        signatures, signatures_status, signatures_seconds = _run_stage(
            "signatures", derive_image_derived_signatures, session=entry.session, lattice_result=lattice)
    if (entry_ready and lattice is not None and getattr(lattice, "status", None) == "assembled"
            and signatures is not None and getattr(signatures, "status", None) == "assembled"):
        graph, graph_status, graph_seconds = _run_stage(
            "partial_graph", derive_partial_image_derived_graph,
            session=entry.session, lattice_result=lattice, signatures_result=signatures,
            **({"budget": budget} if type(budget) is csp.PartialEndpointBudget
               or budget != csp.EndpointBudget() else {}))
    if graph is not None and getattr(graph, "status", None) == "assembled":
        core_support, support_status, support_seconds = _run_stage(
            "partial_c1", derive_partial_core_axis_support, graph_result=graph)
    timings = {"entry_seconds": entry_seconds, "lattice_seconds": lattice_seconds,
               "signatures_seconds": signatures_seconds, "partial_graph_seconds": graph_seconds,
               "partial_c1_seconds": support_seconds}
    timings["total_seconds"] = round(sum(timings.values()), 6)

    selection = getattr(graph, "selection", None)
    resolved = unknown = 0
    per_axis = {str(axis): {"resolved_frame_count": 0, "unknown_frame_count": 0}
                for axis in range(3)}
    if graph is not None and getattr(graph, "status", None) == "assembled" and selection is not None:
        observation, = selection.observability
        for frame in graph.frame_ids:
            if frame == observation.anchor:  # the explicit global gauge is not a measured offset
                continue
            axes = observation.resolved_axes(frame)
            resolved += len(axes)
            unknown += 3 - len(axes)
            for axis in range(3):
                key = str(axis)
                per_axis[key]["resolved_frame_count"] += int(axis in axes)
                per_axis[key]["unknown_frame_count"] += int(axis not in axes)
    core = getattr(graph, "core_solution", None)
    record = {
        "schema": "mc-photogrammetry-partial-core-image-diagnostic-v1",
        "scope": "one-pass normal-mode partial-core diagnostic; scalar summaries only",
        "input_seals": [dict(seal) for seal in INPUT_SEALS],
        "stages": {"entry": {**entry_status, "terminal": getattr(entry, "terminal", "not_reached"),
                               "frame_count": len(entry.session.frames) if entry else 0},
                   "connection_graph": connection_status, "pose_graph": pose_status,
                   "lattice": lattice_status, "signatures": signatures_status,
                   "partial_graph": graph_status, "partial_c1": support_status},
        "selection_status": getattr(selection, "status", "not_reached"),
        "endpoint_budget": _endpoint_budget(selection),
        "endpoint_search_budget": {
            "profile": "partial-relations-524288-v1" if type(budget) is csp.PartialEndpointBudget else "legacy",
            "relation_visits": budget.relation_visits,
            "numerical_relation_visits": min(budget.relation_visits, csp.EndpointBudget().relation_visits)},
        "endpoint_search": _endpoint_search(selection, budget),
        "has_unique_assignment": getattr(selection, "status", None) == "unique",
        "unique_frame_count": len(getattr(selection, "frame_ids", ()))
        if getattr(selection, "status", None) == "unique" else 0,
        "core_frame_count": len(getattr(core, "frames", ())),
        "resolved_axis_count": resolved,
        "unknown_axis_count": unknown,
        "per_axis_observability": per_axis,
        "candidate_count": len(getattr(graph, "candidate_matches", ())),
        "reading_count": len(getattr(graph, "readings", ())),
        "factor_count": len(getattr(graph, "matches", ())),
        "partial_c1_coverage": {
            "core_reading_count": len(getattr(core_support, "core_reading_indices", ())),
            "available_reading_count": len(getattr(graph, "readings", ())),
            "core_factor_count": len(getattr(core_support, "source_rows", ())),
            "selected_factor_count": len(getattr(graph, "matches", ())),
        },
        "timings_seconds": timings,
    }
    destination = Path(report_path)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--project-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--partial-search-budget-profile", choices=("legacy", "relations-524288"),
                        default="legacy", help="Explicit partial search only; numerical cap stays 262144")
    args = parser.parse_args(argv)
    options = ({"budget": csp.PartialEndpointBudget()}
               if args.partial_search_budget_profile == "relations-524288" else {})
    record = run_diagnostic(args.image, args.project_output, args.report_output, **options)
    print(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
