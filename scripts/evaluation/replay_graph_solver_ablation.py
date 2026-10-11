"""DS-R5F evaluation-only exact graph-solver replay and ablation utility.

This script is the one-shot runner for the DS-R5F task.  It writes exactly one
evaluation-only output, ``.tmp-ds-r5f-20261003.json``, and never edits a
production module, the frozen v6 artifact, the frozen project, an input PNG, or
any other file.  It does not create a scratch script, a scratch directory, or a
staging manifest: the utility itself is both the contained parent and the
contained worker.

Replay (worker) sequence, all at production defaults:

1. Load the frozen v6 artifact and confirm its own strict digest replays.
2. Recompute the four ordered PNG identities and require an exact match with the
   artifact's recorded input identities.
3. Re-enter the accepted entry -> lattice -> signatures chain from those PNGs.
   ``build_image_derived_entry`` is called with ``_persist_session`` temporarily
   replaced by a no-op so the frozen project is never rewritten; the entry
   result it returns (session, connection graph, pose graph) is otherwise the
   ordinary accepted composition.
4. Run the accepted graph attribution adapter with a non-invasive observer on
   ``plane_graph.solve_component``.  The observer stores the exact ``readings``,
   the exact post-same-axis-filter ``PlaneMatch`` tuple and the exact solver
   keyword arguments (including the image-derived quadrant constraints) that the
   production graph seam handed to the solver, then delegates to the real solver
   and returns its result unchanged.
5. Fail closed unless the recomputed session/lattice/signature/graph/policy
   digests, the complete ordered pair-trace record, the candidate counts, the
   selected match identity digest, the graph digest, the baseline reason and the
   per-axis diagnostics all reproduce the v6 artifact.  A mismatch writes one
   atomic refused record and returns nonzero without running any ablation.
6. Solver-only counterfactuals, reusing the captured readings, the captured
   image-derived quadrant constraints and the production gates: baseline;
   remove matches touching the +/-32 cell-shift boundary; remove matches
   incident to an ``active_edge_aliases.conflicting_endpoints`` endpoint; remove
   the union of the two.

Containment: the parent process creates the already-reviewed 8-GiB Job through
``scripts/run_image_derived_graph_stage.py::_native_execute`` (which loads the
reviewed ``.tmp-run-axis0-component-preflight-replacement.py`` Job helper) and
launches this same file as ``--worker`` with a 600-second deadline.  The worker
refuses before image processing if it is not actually inside a Job.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mc_photogrammetry.reconstruction import plane_graph  # noqa: E402

SCHEMA = "mc-photogrammetry-evaluation-graph-solver-ablation-v1"
OUTPUT_PATH = ROOT / ".tmp-ds-r5f-20261003.json"
ARTIFACT_PATH = ROOT / ".tmp-axis0-graph-stage-attribution-v6-20261003.json"
PROJECT_PATH = ROOT / ".tmp-axis0-graph-stage-attribution-v6-20261003.mcscan"
PNG_DIR = ROOT / ".tmp-axis0-component-preflight-replacement-20261002.png-inputs"
PNG_PATHS = tuple(PNG_DIR / f"image-{index:04d}.png" for index in range(4))
CONTAINED_RUNNER = ROOT / "scripts" / "run_image_derived_graph_stage.py"
JOB_HELPER = ROOT / ".tmp-run-axis0-component-preflight-replacement.py"

LIMIT_BYTES = 8 * 1024 ** 3
DEADLINE_SECONDS = 600
MAX_SHIFT_CELLS = 32
INTEGER_TOLERANCE = 0.15
MIN_RESULTANT = 0.90
AMBIGUOUS_MARGIN = 0.02
ABLATION_LABELS = ("baseline", "boundary", "alias_conflict", "union")
PRODUCER_PATHS = (
    Path(__file__),
    CONTAINED_RUNNER,
    JOB_HELPER,
    SOURCE_ROOT / "mc_photogrammetry/pipeline/image_derived_entry.py",
    SOURCE_ROOT / "mc_photogrammetry/pipeline/image_derived_lattice_measure.py",
    SOURCE_ROOT / "mc_photogrammetry/pipeline/image_derived_signatures.py",
    SOURCE_ROOT / "mc_photogrammetry/pipeline/image_derived_graph.py",
    SOURCE_ROOT / "mc_photogrammetry/pipeline/image_derived_graph_attribution.py",
    SOURCE_ROOT / "mc_photogrammetry/reconstruction/plane_graph.py",
    SOURCE_ROOT / "mc_photogrammetry/reconstruction/lattice_planes.py",
)


class ReplayMismatch(RuntimeError):
    """One recomputed evidence field disagrees with the frozen v6 artifact."""

    def __init__(self, field: str, detail: str = "") -> None:
        super().__init__(f"{field}: {detail}" if detail else field)
        self.field = field
        self.detail = detail


class ReplayUnavailable(RuntimeError):
    """A required replay stage did not complete without a typed refusal."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def jsonable(value: Any) -> Any:
    """Normalize tuples/None exactly through the strict canonical JSON codec."""
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False))


def seal(payload: Mapping[str, Any]) -> dict:
    """Return a copy carrying the strict digest of its own canonical payload."""
    value = dict(payload)
    value.pop("artifact_sha256", None)
    value["artifact_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return value


def verify_seal(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("artifact_sha256", None)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ReplayMismatch("artifact_sha256", "missing or malformed")
    if claimed != hashlib.sha256(canonical_json(payload)).hexdigest():
        raise ReplayMismatch("artifact_sha256", "digest does not replay")


def write_new(path: str | Path, value: Mapping[str, Any]) -> None:
    """Publish one sealed record atomically, refusing any existing target."""
    target = Path(path)
    temporary = target.with_name(target.name + ".partial")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite existing path {target}")
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing path {target}")
    os.replace(temporary, target)


def refusal_payload(reason: str, *, detail: str = "", field: str = "",
                    extra: Mapping[str, Any] | None = None) -> dict:
    value: dict[str, Any] = {"schema": SCHEMA, "status": "refused", "reason": reason}
    if field:
        value["field"] = field
    if detail:
        value["detail"] = detail
    if extra:
        value.update(extra)
    return seal(value)


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def require_equal(field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ReplayMismatch(field, f"actual={_short(actual)} expected={_short(expected)}")


def boundary_match_indices(matches: Sequence[plane_graph.PlaneMatch],
                           max_shift_cells: int) -> tuple[int, ...]:
    """Indices of matches whose row or column cell shift pins +/- max_shift_cells."""
    indices: list[int] = []
    for index, match in enumerate(matches):
        shift = getattr(match, "cell_shift_rc", None)
        if (type(shift) is not tuple or len(shift) != 2
                or any(type(component) is not int for component in shift)):
            raise ValueError(f"match {index} has a malformed cell_shift_rc")
        if abs(shift[0]) == max_shift_cells or abs(shift[1]) == max_shift_cells:
            indices.append(index)
    return tuple(indices)


def alias_conflict_match_indices(
    matches: Sequence[plane_graph.PlaneMatch],
    conflicting_endpoints: Sequence[Sequence[Any]],
) -> tuple[int, ...]:
    """Indices of matches incident to a conflicting (frame, plane) endpoint."""
    conflicts = {(str(frame), int(plane)) for frame, plane in conflicting_endpoints}
    indices: list[int] = []
    for index, match in enumerate(matches):
        if ((match.left, match.left_plane) in conflicts
                or (match.right, match.right_plane) in conflicts):
            indices.append(index)
    return tuple(indices)


def ablation_index_sets(
    matches: Sequence[plane_graph.PlaneMatch],
    conflicting_endpoints: Sequence[Sequence[Any]],
    max_shift_cells: int = MAX_SHIFT_CELLS,
) -> dict[str, tuple[int, ...]]:
    boundary = boundary_match_indices(matches, max_shift_cells)
    conflict = alias_conflict_match_indices(matches, conflicting_endpoints)
    union = tuple(sorted(set(boundary) | set(conflict)))
    return {"baseline": (), "boundary": boundary,
            "alias_conflict": conflict, "union": union}


def _axis_observations(solution: Any) -> tuple[int, ...] | None:
    alternatives = getattr(solution, "alternatives", None) or ()
    if not alternatives:
        return None
    counts = getattr(alternatives[0], "axis_observations", None)
    if counts is None:
        return None
    return tuple(counts)


def ablation_record(
    label: str,
    matches: Sequence[plane_graph.PlaneMatch],
    removed: Sequence[int],
    solution: Any,
    identity: Callable[[plane_graph.PlaneMatch], Sequence[Any]],
) -> dict:
    removed_tuple = tuple(removed)
    resultants = getattr(solution, "axis_resultants", None) or {}
    residuals = getattr(solution, "axis_residuals", None) or {}
    observations = _axis_observations(solution)
    axes = [
        [axis, resultants.get(axis), residuals.get(axis),
         None if observations is None else observations[axis]]
        for axis in range(3)
    ]
    return {
        "label": label,
        "removed_match_indices": list(removed_tuple),
        "removed_match_count": len(removed_tuple),
        "retained_match_count": len(matches) - len(removed_tuple),
        "removed_match_identities": [list(identity(matches[index]))
                                     for index in removed_tuple],
        "reason": str(getattr(solution, "reason", "")),
        "axis_resultants": [[axis, resultants[axis]] for axis in sorted(resultants)],
        "axis_residuals": [[axis, residuals[axis]] for axis in sorted(residuals)],
        "axis_observations": None if observations is None else list(observations),
        "axes": axes,
    }


def default_identity(match: plane_graph.PlaneMatch) -> Sequence[Any]:
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    return graph_mod._input_match_identity(match)


def run_ablations(
    *,
    readings: Sequence[Any],
    matches: Sequence[plane_graph.PlaneMatch],
    solver_kwargs: Mapping[str, Any],
    conflicting_endpoints: Sequence[Sequence[Any]],
    solver: Callable[..., Any],
    max_shift_cells: int = MAX_SHIFT_CELLS,
    baseline_solution: Any = None,
    identity: Callable[[plane_graph.PlaneMatch], Sequence[Any]] | None = None,
) -> list[dict]:
    """Solve the four preregistered counterfactuals with unchanged constraints."""
    identity_fn = identity or default_identity
    index_sets = ablation_index_sets(matches, conflicting_endpoints, max_shift_cells)
    records: list[dict] = []
    for label in ABLATION_LABELS:
        removed = index_sets[label]
        removed_set = set(removed)
        retained = tuple(match for index, match in enumerate(matches)
                         if index not in removed_set)
        if label == "baseline" and baseline_solution is not None:
            solution = baseline_solution
        else:
            solution = solver(readings, retained, **solver_kwargs)
        records.append(ablation_record(label, matches, removed, solution, identity_fn))
    return records


def validate_replay(artifact: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    """Raise ReplayMismatch unless every recomputed field matches the artifact."""
    graph = artifact.get("graph")
    if not isinstance(graph, dict):
        raise ReplayMismatch("graph", "artifact graph missing")
    payload = graph.get("payload")
    trace = artifact.get("trace")
    if not isinstance(payload, dict) or not isinstance(trace, dict):
        raise ReplayMismatch("graph", "artifact graph payload or trace missing")
    solver_failure = trace.get("solver_failure") or {}
    aliases = solver_failure.get("active_edge_aliases") or {}
    boundary = solver_failure.get("boundary_saturation") or {}
    require_equal("graph.result_sha256", evidence["graph_sha256"],
                  artifact.get("graph_sha256"))
    require_equal("graph.inner_result_sha256", evidence["graph_sha256"],
                  graph.get("result_sha256"))
    require_equal("trace.graph_result_sha256", evidence["graph_sha256"],
                  trace.get("graph_result_sha256"))
    require_equal("trace.record", evidence["trace"], trace)
    require_equal("trace.attribution_sha256", evidence["trace_sha256"],
                  artifact.get("trace_sha256"))
    require_equal("trace.attribution_record", evidence["trace_sha256"],
                  trace.get("attribution_sha256"))
    require_equal("graph.payload.session_sha256", evidence["session_sha256"],
                  payload.get("session_sha256"))
    require_equal("graph.payload.lattice_sha256", evidence["lattice_sha256"],
                  payload.get("lattice_sha256"))
    require_equal("graph.payload.signatures_sha256", evidence["signatures_sha256"],
                  payload.get("signatures_sha256"))
    require_equal("graph.payload.signature_digests", evidence["signature_digests"],
                  payload.get("signature_digests"))
    require_equal("graph.payload.reason", evidence["reason"], payload.get("reason"))
    require_equal("trace.reason", evidence["reason"], trace.get("reason"))
    require_equal("candidate_counts", evidence["candidate_counts"],
                  trace.get("candidate_counts"))
    require_equal("solver_failure.reason", evidence["reason"],
                  solver_failure.get("reason"))
    require_equal("solver_failure.axes", evidence["axes"], solver_failure.get("axes"))
    require_equal("selected_match_count", evidence["selected_match_count"],
                  aliases.get("input_match_count"))
    require_equal("selected_alias_identity", evidence["alias_identity_sha256"],
                  aliases.get("input_match_identity_sha256"))
    require_equal("selected_boundary_identity", evidence["boundary_identity_sha256"],
                  boundary.get("input_match_identity_sha256"))
    require_equal("boundary.kept_counts", evidence["boundary_kept_counts"],
                  boundary.get("kept_boundary_counts"))
    require_equal("boundary.max_shift_cells", evidence["max_shift_cells"],
                  boundary.get("max_shift_cells"))
    require_equal("inputs", evidence["inputs"], artifact.get("inputs"))
    require_equal("artifact_sha256", evidence["artifact_sha256"],
                  artifact.get("artifact_sha256"))
    require_equal("project_sha256", evidence["project_sha256"],
                  artifact.get("project_sha256"))
    require_equal("project.session_sha256", evidence["session_sha256"],
                  evidence["project_session_sha256"])


def input_identities(png_paths: Sequence[Path]) -> list[dict]:
    from PIL import Image
    identities: list[dict] = []
    for index, path in enumerate(png_paths):
        with Image.open(path) as image:
            image.verify()
            width, height = image.size
        identities.append({"frame_id": f"image-{index:04d}", "path": str(path),
                           "sha256": sha256_file(path), "width": width,
                           "height": height})
    return identities


def _require_stage(stage: str, value: Any) -> None:
    from mc_photogrammetry.reconstruction import plane_placement as placement
    if isinstance(value, placement.PlacementRefusal):
        raise ReplayUnavailable(
            f"{stage} refused: {value.reason}: {value.detail}")
    status = getattr(value, "status", None)
    if status != "assembled":
        raise ReplayUnavailable(f"{stage} status is {status!r}")


def recompute(png_paths: Sequence[Path], project_path: Path) -> dict:
    """Run the accepted chain once and capture the exact production solver call."""
    from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
    from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
    from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod

    original_persist = entry_mod._persist_session
    entry_mod._persist_session = lambda **kwargs: None
    try:
        entry = entry_mod.build_image_derived_entry(
            png_paths, project_path=project_path)
    finally:
        entry_mod._persist_session = original_persist
    if (getattr(getattr(entry, "graph", None), "status", None) != "ok"
            or getattr(getattr(entry, "pose_graph", None), "status", None) != "ok"):
        raise ReplayUnavailable("entry graph or pose graph is unavailable")
    lattice = lattice_mod.measure_image_derived_lattice(
        session=entry.session, pose_graph=entry.pose_graph)
    _require_stage("lattice", lattice)
    signatures = signatures_mod.derive_image_derived_signatures(
        session=entry.session, lattice_result=lattice)
    _require_stage("signatures", signatures)

    captured: dict[str, Any] = {}
    original_solve = plane_graph.solve_component

    def solve_observer(readings, matches, **kwargs):
        captured["readings"] = tuple(readings)
        captured["matches"] = tuple(matches)
        captured["kwargs"] = dict(kwargs)
        solution = original_solve(readings, matches, **kwargs)
        captured["solution"] = solution
        return solution

    plane_graph.solve_component = solve_observer
    try:
        result, attribution = attribution_mod.derive_image_derived_graph_attribution(
            session=entry.session, lattice_result=lattice,
            signatures_result=signatures, connection_graph=entry.graph,
            matcher_witness_sink=None, matcher_witness_payload_sink=None,
            witness_budget=None)
    finally:
        plane_graph.solve_component = original_solve
    if not isinstance(result, graph_mod.ImageDerivedGraphResult) or attribution is None:
        raise ReplayUnavailable("graph attribution produced no typed graph result")
    if "matches" not in captured or "solution" not in captured:
        raise ReplayUnavailable("production solver call was not observed")
    return {"entry": entry, "lattice": lattice, "signatures": signatures,
            "result": result, "attribution": attribution, "captured": captured,
            "solver": original_solve, "graph_mod": graph_mod}


def replay_evidence(artifact: Mapping[str, Any], recomputed: Mapping[str, Any],
                    identities: Sequence[Mapping[str, Any]],
                    project_path: Path) -> dict:
    graph_mod = recomputed["graph_mod"]
    result = recomputed["result"]
    attribution = recomputed["attribution"]
    captured = recomputed["captured"]
    matches = captured["matches"]
    failure = attribution.solver_failure
    if failure is None:
        raise ReplayUnavailable("graph attribution carried no solver failure")
    alias = failure.active_edge_aliases
    boundary = failure.boundary_saturation
    if alias is None or boundary is None:
        raise ReplayUnavailable("solver failure is missing an alias or boundary record")
    identity_rows = [graph_mod._input_match_identity(match) for match in matches]
    shifts = [match.cell_shift_rc for match in matches]
    kept_counts, _cardinality = graph_mod._boundary_saturation_counts(
        shifts, MAX_SHIFT_CELLS)
    from mc_photogrammetry.project.store import load_project
    loaded = load_project(project_path)
    if getattr(loaded, "schema_version", None) != 2:
        raise ReplayUnavailable("frozen project does not replay as schema 2")
    session_record = loaded.image_derived_session
    if not isinstance(session_record, dict):
        raise ReplayUnavailable("frozen project carries no image-derived session")
    if list(map(str, loaded.source_paths)) != [str(path) for path in PNG_PATHS]:
        raise ReplayUnavailable("frozen project source paths do not bind the PNGs")
    return {
        "graph_sha256": result.result_sha256,
        "trace": jsonable(attribution.to_record()),
        "trace_sha256": attribution.attribution_sha256,
        "session_sha256": result.session_sha256,
        "lattice_sha256": result.lattice_sha256,
        "signatures_sha256": result.signatures_sha256,
        "signature_digests": [item.signature_sha256 for item in result.signatures],
        "candidate_counts": {"pre_admission": attribution.pre_admission_candidates,
                             "post_admission": attribution.post_admission_candidates,
                             "post_same_axis": attribution.post_same_axis_candidates},
        "reason": result.reason,
        "axes": jsonable([list(row) for row in failure.axes]),
        "selected_match_count": len(matches),
        "alias_identity_sha256": graph_mod._alias_domain_digest(
            "input_match_identity", graph_mod.FAILED_ACTIVE_EDGE_ALIASES_SCHEMA,
            identity_rows),
        "boundary_identity_sha256": graph_mod._alias_domain_digest(
            graph_mod.BOUNDARY_SATURATION_INPUT_MATCH_IDENTITY_DOMAIN,
            graph_mod.BOUNDARY_SATURATION_SCHEMA, identity_rows),
        "boundary_kept_counts": [int(value) for value in kept_counts],
        "max_shift_cells": MAX_SHIFT_CELLS,
        "inputs": list(identities),
        "artifact_sha256": artifact.get("artifact_sha256"),
        "project_sha256": sha256_file(project_path),
        "project_session_sha256": session_record.get("session_sha256"),
    }


def producer_records() -> list[dict]:
    return [{"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in PRODUCER_PATHS]


def execute_replay(*, output_path: Path = OUTPUT_PATH, artifact_path: Path = ARTIFACT_PATH,
                   project_path: Path = PROJECT_PATH,
                   png_paths: Sequence[Path] = PNG_PATHS) -> dict:
    """Recompute, fail closed, and return the sealed ablation payload."""
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    verify_seal(artifact)
    identities = input_identities(png_paths)
    recomputed = recompute(png_paths, project_path)
    evidence = replay_evidence(artifact, recomputed, identities, project_path)
    validate_replay(artifact, evidence)

    attribution = recomputed["attribution"]
    captured = recomputed["captured"]
    failure = attribution.solver_failure
    conflicting = failure.active_edge_aliases.conflicting_endpoints
    matches = captured["matches"]
    records = run_ablations(
        readings=captured["readings"], matches=matches,
        solver_kwargs=captured["kwargs"], conflicting_endpoints=conflicting,
        solver=recomputed["solver"], max_shift_cells=MAX_SHIFT_CELLS,
        baseline_solution=captured["solution"])
    baseline = records[0]
    require_equal("ablation.baseline_reason", baseline["reason"], evidence["reason"])
    require_equal("ablation.baseline_axes", baseline["axes"], evidence["axes"])

    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "v6_artifact_path": str(Path(artifact_path).resolve()),
        "v6_artifact_sha256": artifact.get("artifact_sha256"),
        "project_path": str(Path(project_path).resolve()),
        "pngs": identities,
        "replay": {
            "session_sha256": evidence["session_sha256"],
            "lattice_sha256": evidence["lattice_sha256"],
            "signatures_sha256": evidence["signatures_sha256"],
            "policy_sha256": attribution.policy_sha256,
            "graph_result_sha256": evidence["graph_sha256"],
            "trace_sha256": evidence["trace_sha256"],
            "candidate_counts": evidence["candidate_counts"],
            "reason": evidence["reason"],
            "axes": evidence["axes"],
            "selected_match_count": evidence["selected_match_count"],
            "selected_match_identities": [
                list(default_identity(match)) for match in matches],
            "selected_alias_identity_sha256": evidence["alias_identity_sha256"],
            "selected_boundary_identity_sha256": evidence["boundary_identity_sha256"],
            "boundary_kept_counts": evidence["boundary_kept_counts"],
            "conflicting_endpoints": [[str(frame), int(plane)] for frame, plane
                                      in conflicting],
        },
        "effective_settings": {name: value for name, value
                               in attribution.effective_settings},
        "ablations": records,
        "provenance": {
            "sources": producer_records(),
            "containment": {
                "limit_bytes": LIMIT_BYTES,
                "deadline_seconds": DEADLINE_SECONDS,
                "contained_runner": str(CONTAINED_RUNNER.resolve()),
                "job_helper": str(JOB_HELPER.resolve()),
            },
        },
    }
    return seal(payload)


def _in_job() -> bool:
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                      ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    contained = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None,
                                 ctypes.byref(contained)):
        return False
    return bool(contained.value)


def worker_main() -> int:
    if not _in_job():
        write_new(OUTPUT_PATH, refusal_payload(
            "not_contained", detail="worker is not inside a Job"))
        return 1
    try:
        payload = execute_replay()
    except ReplayMismatch as error:
        write_new(OUTPUT_PATH, refusal_payload(
            "replay_mismatch", field=error.field, detail=error.detail))
        return 1
    except ReplayUnavailable as error:
        write_new(OUTPUT_PATH, refusal_payload(
            "replay_unavailable", detail=str(error)))
        return 1
    except FileExistsError:
        return 1
    except Exception as error:  # noqa: BLE001
        try:
            write_new(OUTPUT_PATH, refusal_payload(
                "replay_exception", detail=f"{type(error).__name__}: {error}"))
        except FileExistsError:
            return 1
        return 1
    write_new(OUTPUT_PATH, payload)
    return 0


def _load_contained_runner():
    spec = importlib.util.spec_from_file_location(
        "ds_r5f_contained_runner", CONTAINED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("contained runner is not loadable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_once() -> int:
    """Preflight every named output, then launch exactly one contained worker."""
    partial = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".partial")
    if OUTPUT_PATH.exists() or partial.exists():
        print(f"DS-R5F output already exists: {OUTPUT_PATH}", file=sys.stderr)
        return 1
    missing = [str(path) for path in (*PNG_PATHS, ARTIFACT_PATH, PROJECT_PATH)
               if not Path(path).is_file()]
    if missing:
        write_new(OUTPUT_PATH, refusal_payload(
            "missing_input", detail="; ".join(missing)))
        return 1
    try:
        runner = _load_contained_runner()
    except Exception as error:  # noqa: BLE001
        write_new(OUTPUT_PATH, refusal_payload(
            "containment_unavailable", detail=f"{type(error).__name__}: {error}"))
        return 1
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    code, reason, cleanup, elapsed = runner._native_execute(
        command, deadline_seconds=DEADLINE_SECONDS, root=ROOT,
        stdout_path=os.devnull, stderr_path=os.devnull)
    if code == 0:
        return 0
    if not OUTPUT_PATH.exists():
        write_new(OUTPUT_PATH, refusal_payload(
            "containment_failure", detail=reason,
            extra={"exit_code": int(code), "cleanup": cleanup,
                   "elapsed_seconds": elapsed}))
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main()
    return run_once()


if __name__ == "__main__":
    raise SystemExit(main())
