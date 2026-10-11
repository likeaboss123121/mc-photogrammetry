"""Offline edge-phase influence analysis over the pinned R8 observation only.

This analyzer holds the captured full-fit translations and face scalars fixed.
For each active match it removes both endpoint phases and recomposes circular
phase metrics with the existing production helper. It does not rerun a solver,
fit, matcher, or image pipeline and does not estimate the effect of re-fitting
translations after deleting an edge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ARTIFACT_RELATIVE = Path(".tmp-geom-phase-capture-r8-20261004/observation.json")
ARTIFACT_SHA256 = "c3d2ee4ad3e803754d7297ab0c09b7994b17c24de345fda281c0b5cb62888ff8"
ARTIFACT_PAYLOAD_SHA256 = "db4b1100d34169570bd7878de05216c6d50a7df76d391087eedd84c7b7a94ddd"
OBSERVATION_SHA256 = "a9d15a99ce58a6b42c183dc4ba63662afe1ec5e12ed8609298e5883ac20fc3f1"
PHASE_CAPTURE_SHA256 = "5cbe0943a225abe565ad9f49bf7b0e136f305a992a52d48e51bdb3ed867b6535"
SOLVER_INPUT_BINDINGS_SHA256 = "25c8f1460311846cacfec929ec10f703ad8f47af9e8f41eeb5304bbe4f97c99a"
FAILED_ASSIGNMENT_SHA256 = "ab4cb09bf5e8fe00e0bca11ca6cf2732cf24fd086d6f717919369f4191525dcd"
INPUT_MATCH_IDENTITY_SHA256 = "28ad4ac48021c727c4b6e7bf283e9965514dc3016c3c16919a18367cf0868105"
TRACE_SHA256 = "738c9dc09dc17cf64a74e91472cb59a45bbc9917de479cd4f33f80551becd736"
GRAPH_SHA256 = "0a64c5e73b62b376f41e337a756c61af880d72cee1a90992001a42d00d9f39cd"
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_ACTIVE_EDGES = 64
MAX_PHASE_FACES = 128
MAX_FRAMES = 8
EXPECTED_INTEGER_TOLERANCE = 0.15
EXPECTED_MIN_RESULTANT = 0.90
EXPECTED_COMPATIBILITY_DECIMALS = 6
REPLAY_DECIMALS = 6
SOURCE_PINS = {
    "scripts/run_image_derived_graph_stage.py":
        "d5244bc4f5fb4648dadf22b14b66ef3694b60781d688864679a274f5058ada97",
    "src/mc_photogrammetry/pipeline/image_derived_graph.py":
        "32747546c934f8c218e43646cc995f310ff383e522c8a10f878e691f54954188",
    "src/mc_photogrammetry/reconstruction/plane_graph.py":
        "f697c2f4ffb512ebb3e7af8f9e497328b4fe828197f62d761b9b0f850ef8972f",
    "src/mc_photogrammetry/reconstruction/lattice_planes.py":
        "ddf95ff9f0dfb66f3bb2c080643b467652194611eaf6237cd28160d73e002d01",
}
INPUT_IDENTITY_SCHEMA = "mc-photogrammetry-failed-active-edge-aliases-v1"


class ArtifactError(ValueError):
    """Pinned or internally bound diagnostic artifact failed validation."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _expect(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ArtifactError(f"{label}_mismatch")


def verify_source_pins(repo_root: Path | None = None) -> None:
    """Refuse upstream drift before opening data or importing replay helpers."""
    root = repo_root or Path(__file__).resolve().parents[2]
    for relative, expected in SOURCE_PINS.items():
        try:
            raw = (root / relative).read_bytes()
        except OSError as exc:
            raise ArtifactError(f"source_unreadable:{relative}") from exc
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ArtifactError(f"source_sha256_mismatch:{relative}")


def _load_production_validators(source_root: Path):
    verify_source_pins(source_root.parent)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.reconstruction import plane_graph
    for module, relative in (
            (graph_mod, "pipeline/image_derived_graph.py"),
            (plane_graph, "reconstruction/plane_graph.py"),
            (graph_mod.lattice_planes, "reconstruction/lattice_planes.py")):
        _expect(Path(module.__file__).resolve(),
                (source_root / "mc_photogrammetry" / relative).resolve(),
                "loaded_source_path")
    return graph_mod, plane_graph


def validate_pinned_artifact(path: Path) -> dict[str, Any]:
    """Verify file, artifact, observer, nested-record and cross-record seals."""
    verify_source_pins()
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ArtifactError("artifact_unreadable") from exc
    if len(raw_bytes) > MAX_ARTIFACT_BYTES:
        raise ArtifactError("artifact_byte_cap")
    _expect(hashlib.sha256(raw_bytes).hexdigest(), ARTIFACT_SHA256,
            "artifact_file_sha256")
    try:
        artifact = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("artifact_json_invalid") from exc
    if type(artifact) is not dict:
        raise ArtifactError("artifact_root_invalid")
    body = dict(artifact)
    claimed_outer = body.pop("artifact_sha256", None)
    _expect(hashlib.sha256(_canonical(body)).hexdigest(), claimed_outer,
            "artifact_payload_digest")
    _expect(claimed_outer, ARTIFACT_PAYLOAD_SHA256, "artifact_payload_pin")

    observation = artifact.get("solver_failure_observation")
    if type(observation) is not dict:
        raise ArtifactError("solver_failure_observation_missing")
    obs_body = dict(observation)
    claimed_obs = obs_body.pop("observation_sha256", None)
    _expect(hashlib.sha256(_canonical(obs_body)).hexdigest(), claimed_obs,
            "observation_digest")
    _expect(claimed_obs, OBSERVATION_SHA256, "observation_pin")
    _expect(observation.get("schema"),
            "mc-photogrammetry-evaluation-graph-stage-observation-v1",
            "observation_schema")
    if (artifact.get("schema") != "mc-photogrammetry-evaluation-graph-stage-v1"
            or artifact.get("status") != "complete"
            or artifact.get("execution_complete") is not True):
        raise ArtifactError("artifact_completion_state")

    trace, graph = artifact.get("trace"), artifact.get("graph")
    binding, raw, projected = (observation.get(key)
                               for key in ("binding", "raw", "projected"))
    if not all(type(item) is dict for item in (trace, graph, binding, raw, projected)):
        raise ArtifactError("artifact_binding_shape")
    graph_payload = graph.get("payload")
    if type(graph_payload) is not dict:
        raise ArtifactError("graph_payload_shape")
    _expect(hashlib.sha256(_canonical(graph_payload)).hexdigest(),
            graph.get("result_sha256"), "graph_payload_digest")
    trace_body = dict(trace)
    claimed_trace = trace_body.pop("attribution_sha256", None)
    _expect(hashlib.sha256(_canonical(trace_body)).hexdigest(), claimed_trace,
            "trace_payload_digest")
    for value, expected, label in (
            (artifact.get("trace_sha256"), TRACE_SHA256, "trace_file_binding"),
            (trace.get("attribution_sha256"), TRACE_SHA256, "trace_digest"),
            (artifact.get("graph_sha256"), GRAPH_SHA256, "graph_file_binding"),
            (graph.get("result_sha256"), GRAPH_SHA256, "graph_digest"),
            (binding.get("graph_sha256"), GRAPH_SHA256, "observer_graph_binding"),
            (trace.get("graph_result_sha256"), GRAPH_SHA256, "trace_graph_binding"),
            (binding.get("entry_sha256"), artifact.get("entry_sha256"), "entry_binding"),
            (binding.get("input_sha256"),
             [row.get("sha256") for row in artifact.get("inputs", [])],
             "image_identity_binding"),
            (binding.get("lattice_sha256"), graph_payload.get("lattice_sha256"),
             "lattice_binding"),
            (binding.get("signatures_sha256"), graph_payload.get("signatures_sha256"),
             "signature_binding"),
            (projected, trace.get("solver_failure"), "projected_failure_binding")):
        _expect(value, expected, label)

    aliases = projected.get("active_edge_aliases")
    phase = observation.get("phase_capture")
    inputs = observation.get("solver_input_bindings")
    if not all(type(item) is dict for item in (aliases, phase, inputs)):
        raise ArtifactError("required_capture_absent")
    if len(aliases.get("edges", [])) > MAX_ACTIVE_EDGES:
        raise ArtifactError("active_edge_cap")
    if len(phase.get("faces", [])) > MAX_PHASE_FACES:
        raise ArtifactError("phase_face_cap")
    if len(raw.get("frames", [])) > MAX_FRAMES:
        raise ArtifactError("frame_cap")

    source_root = Path(__file__).resolve().parents[2] / "src"
    graph_mod, plane_graph = _load_production_validators(source_root)
    try:
        graph_mod.validate_failed_active_edge_aliases(aliases)
        graph_mod.validate_solver_phase_capture(phase)
        graph_mod.validate_solver_input_bindings(inputs)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ArtifactError("nested_record_validation") from exc
    _expect(phase.get("record_sha256"), PHASE_CAPTURE_SHA256,
            "phase_capture_digest")
    _expect(inputs.get("record_sha256"), SOLVER_INPUT_BINDINGS_SHA256,
            "solver_input_bindings_digest")
    try:
        identity = _ordered_match_identity_digest(inputs)
    except ArtifactError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ArtifactError("ordered_match_identity_shape") from exc
    _expect(identity, inputs.get("input_match_identity_sha256"),
            "ordered_match_identity_digest")
    _expect(identity, INPUT_MATCH_IDENTITY_SHA256, "ordered_match_identity_pin")

    # Shared assignment and full ordered input-match identity must all resolve to
    # the same failed best assignment; no row is reconstructed from imagery.
    for record, key in ((aliases, "failed_assignment_sha256"),
                        (phase, "source_sha256"),
                        (inputs, "failed_assignment_sha256")):
        _expect(record.get(key), FAILED_ASSIGNMENT_SHA256,
                f"{key}_cross_binding")
    for record in (aliases, inputs):
        _expect(record.get("input_match_identity_sha256"),
                INPUT_MATCH_IDENTITY_SHA256, "input_match_identity")
    _expect(inputs.get("input_match_count"), aliases.get("input_match_count"),
            "input_match_count")
    _expect(inputs.get("active_edge_count"), aliases.get("active_edge_count"),
            "active_edge_count")
    _expect(phase.get("frames"), inputs.get("frames"), "frame_binding")
    _expect(phase.get("assignments"), inputs.get("assignments"),
            "assignment_binding")
    _expect(raw.get("frames"), phase.get("frames"), "snapshot_frame_binding")
    _expect(raw.get("quadrants"), phase.get("assignments"),
            "snapshot_quadrant_binding")
    raw_resultants = {row["axis"]: row["value"]
                      for row in raw.get("axis_resultants", [])}
    raw_residuals = {row["axis"]: row["value"]
                     for row in raw.get("axis_residuals", [])}
    for axis_row in phase.get("axes", []):
        axis = axis_row[0]
        _expect(raw_resultants.get(axis), axis_row[1], "snapshot_resultant_binding")
        _expect(raw_residuals.get(axis), axis_row[2], "snapshot_residual_binding")

    settings = trace.get("effective_settings")
    if type(settings) is not dict:
        raise ArtifactError("effective_settings_missing")
    _expect(settings.get("integer_tolerance"), EXPECTED_INTEGER_TOLERANCE,
            "integer_tolerance")
    _expect(settings.get("min_resultant"), EXPECTED_MIN_RESULTANT,
            "min_resultant")
    _expect(phase.get("compatibility_decimals"), EXPECTED_COMPATIBILITY_DECIMALS,
            "phase_compatibility_decimals")
    try:
        _validate_edge_phase_bindings(phase, inputs, aliases, plane_graph)
    except ArtifactError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ArtifactError("edge_phase_binding_validation") from exc
    return artifact


def _ordered_match_identity_digest(inputs: dict) -> str:
    """Replay the full ordered scalar projection from retained input bindings.

    Every match, including inactive ones, is present. Resolve endpoint frame,
    plane and axis from its referenced captured reading, and selected b from the
    captured endpoint. This does not construct matches or infer missing values.
    """
    edges, reading_rows = inputs["edges"], inputs["readings"]
    if len(edges) != inputs["input_match_count"]:
        raise ArtifactError("ordered_match_identity_count")
    readings = {row[0]: row for row in reading_rows}
    if len(readings) != len(reading_rows):
        raise ArtifactError("ordered_match_identity_duplicate_reading")
    identities = []
    for position, edge in enumerate(edges):
        if type(edge[0]) is not int or edge[0] != position:
            raise ArtifactError("ordered_match_identity_order")
        left, right = readings[edge[1][0]], readings[edge[2][0]]
        identities.append([left[1], left[2], left[3], right[1], right[2],
                           right[3], edge[3], float(edge[1][2]), float(edge[2][2])])
    return hashlib.sha256(_canonical({"domain": "input_match_identity",
        "schema": INPUT_IDENTITY_SCHEMA, "payload": identities})).hexdigest()


def _validate_edge_phase_bindings(phase: dict, inputs: dict, aliases: dict,
                                  plane_graph) -> None:
    active = {row[0]: row for row in inputs["edges"] if row[4] is True}
    alias_ids = {row[0] for row in aliases["edges"]}
    if set(active) != alias_ids:
        raise ArtifactError("active_edge_match_index_binding")
    faces: dict[int, list[list]] = {}
    for row in phase["faces"]:
        faces.setdefault(row[0], []).append(row)
    if set(faces) != alias_ids or any(len(rows) != 2 for rows in faces.values()):
        raise ArtifactError("phase_face_match_coverage")
    readings = {row[0]: row for row in inputs["readings"]}
    quadrants = dict(zip(inputs["frames"], inputs["assignments"]))
    for match_index, rows in faces.items():
        edge = active[match_index]
        endpoints = (edge[1], edge[2])
        for face in rows:
            own, other = face[1:8], face[8:15]
            matched = False
            for endpoint_idx, endpoint in enumerate(endpoints):
                ref, alt_idx = endpoint[0], endpoint[1]
                reading = readings[ref]
                alt = reading[5][alt_idx]
                expected_own = [reading[1], reading[2], reading[3], alt[0],
                                alt[1], reading[4], endpoint[6]]
                other_endpoint = endpoints[1 - endpoint_idx]
                other_reading = readings[other_endpoint[0]]
                other_alt = other_reading[5][other_endpoint[1]]
                expected_other = [other_reading[1], other_reading[2],
                                  other_reading[3], other_alt[0], other_alt[1],
                                  other_reading[4], other_endpoint[6]]
                if own == expected_own:
                    matched = (other == expected_other
                               and face[15] == edge[3]
                               and face[17] == quadrants[face[1]]
                               and face[18] == quadrants[face[8]])
                    break
            if not matched:
                raise ArtifactError("phase_endpoint_value_binding")
            own_world = plane_graph.axis_transform(face[3], face[17])[0]
            other_world = plane_graph.axis_transform(face[10], face[18])[0]
            if own_world != other_world:
                raise ArtifactError("phase_world_axis_binding")


def recompute_full_set(phase: dict) -> list[dict[str, float | int]]:
    """Recompose captured full set with the unmodified production helper."""
    source_root = Path(__file__).resolve().parents[2] / "src"
    graph_mod, _plane_graph = _load_production_validators(source_root)
    scalars: dict[int, list[float]] = {axis: [] for axis in range(3)}
    plane_graph = _plane_graph
    for face in phase["faces"]:
        axis = plane_graph.axis_transform(face[3], face[17])[0]
        scalars[axis].append(face[16])
    expected = {row[0]: row for row in phase["axes"]}
    result = []
    for axis in range(3):
        resultant, residual = graph_mod._recompose_phase_rows(scalars[axis])
        stored = expected[axis]
        if (round(resultant, REPLAY_DECIMALS) != round(stored[3], REPLAY_DECIMALS)
                or round(residual, REPLAY_DECIMALS) != round(stored[4], REPLAY_DECIMALS)):
            raise ArtifactError("full_set_phase_recomposition_mismatch")
        result.append({"axis": axis, "resultant": resultant,
                       "residual": residual, "face_count": len(scalars[axis])})
    return result


def leave_one_edge_out(phase: dict) -> list[dict[str, float | int | bool]]:
    """Drop both endpoint phases per match and recompose; do not refit."""
    source_root = Path(__file__).resolve().parents[2] / "src"
    graph_mod, plane_graph = _load_production_validators(source_root)
    rows_by_edge: dict[int, list[list]] = {}
    for row in phase["faces"]:
        rows_by_edge.setdefault(row[0], []).append(row)
    baseline_faces: dict[int, list[list]] = {axis: [] for axis in range(3)}
    baseline_axes = {row[0]: row for row in phase["axes"]}
    for row in phase["faces"]:
        axis = plane_graph.axis_transform(row[3], row[17])[0]
        baseline_faces[axis].append(row)
    out = []
    for match_index in sorted(rows_by_edge):
        removed = rows_by_edge[match_index]
        if len(removed) != 2:
            raise ArtifactError("loo_edge_endpoint_count")
        axis_set = {plane_graph.axis_transform(row[3], row[17])[0]
                    for row in removed}
        if len(axis_set) != 1:
            raise ArtifactError("loo_edge_cross_axis")
        axis = next(iter(axis_set))
        removed_ids = {id(row) for row in removed}
        retained_scalars = [row[16] for row in baseline_faces[axis]
                            if id(row) not in removed_ids]
        resultant, residual = graph_mod._recompose_phase_rows(retained_scalars)
        out.append({
            "match_index": match_index,
            "world_axis": axis,
            "remaining_face_count": len(retained_scalars),
            "resultant": resultant,
            "residual": residual,
            "delta_resultant": resultant - baseline_axes[axis][3],
            "delta_residual": residual - baseline_axes[axis][4],
            "resultant_below_min": resultant < EXPECTED_MIN_RESULTANT,
            "residual_above_tolerance": residual > EXPECTED_INTEGER_TOLERANCE,
        })
    return out


def analyze(path: Path) -> dict[str, Any]:
    artifact = validate_pinned_artifact(path)
    observation = artifact["solver_failure_observation"]
    phase = observation["phase_capture"]
    full = recompute_full_set(phase)
    loo_first = leave_one_edge_out(phase)
    loo_second = leave_one_edge_out(phase)
    if loo_first != loo_second:
        raise ArtifactError("loo_nondeterministic")
    return {
        "schema": "mc-photogrammetry-plane-residual-attribution-v1",
        "status": "complete",
        "artifact_sha256": ARTIFACT_SHA256,
        "observation_sha256": OBSERVATION_SHA256,
        "phase_capture_sha256": PHASE_CAPTURE_SHA256,
        "solver_input_bindings_sha256": SOLVER_INPUT_BINDINGS_SHA256,
        "method": "fixed_baseline_face_phase_leave_one_match_out_recomposition",
        "refit_performed": False,
        "solver_rerun": False,
        "full_set_axes": full,
        "leave_one_edge_out": loo_first,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--validate-only", action="store_true",
                        help="validate pinned seals and cross-bindings without metrics")
    args = parser.parse_args()
    try:
        if args.validate_only:
            artifact = validate_pinned_artifact(args.repo_root / ARTIFACT_RELATIVE)
            result = {"schema": "mc-photogrammetry-plane-residual-attribution-v1",
                      "status": "validated",
                      "artifact_sha256": ARTIFACT_SHA256,
                      "phase_capture_sha256": PHASE_CAPTURE_SHA256,
                      "solver_input_bindings_sha256": SOLVER_INPUT_BINDINGS_SHA256,
                      "active_edges": artifact["solver_failure_observation"]
                                      ["solver_input_bindings"]["active_edge_count"]}
        else:
            result = analyze(args.repo_root / ARTIFACT_RELATIVE)
    except ArtifactError as exc:
        result = {"schema": "mc-photogrammetry-plane-residual-attribution-v1",
                  "status": "refused", "reason": str(exc)}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] in {"validated", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
