"""DS-R5G evaluation-only per-endpoint removal versus matched-size controls.

This script is the one-shot runner for the DS-R5G task.  It writes exactly one
evaluation-only output, ``.tmp-ds-r5g-20261003.json``, and never edits a
production module, the frozen v6 artifact, the frozen project, an input PNG, the
DS-R5F output or any other file.  It does not create a scratch script, a scratch
directory, or a staging manifest: the utility itself is both the contained
parent and the contained worker.

Replay (worker) sequence, all at production defaults:

1. Load the frozen v6 artifact and confirm its own strict digest replays.
2. Recompute the four ordered PNG identities and require an exact match with the
   artifact's recorded input identities.
3. Re-enter the accepted entry -> lattice -> signatures chain from those PNGs.
   ``build_image_derived_entry`` is called with ``_persist_session`` temporarily
   replaced by a no-op so the frozen project is never rewritten.
4. Run the accepted graph attribution adapter with a non-invasive observer on
   ``plane_graph.solve_component``.  The observer stores the exact ``readings``,
   the exact post-same-axis-filter ``PlaneMatch`` tuple and the exact solver
   keyword arguments (including the image-derived quadrant constraints) that the
   production graph seam handed to the solver, then delegates to the real solver
   and returns its result unchanged.
5. Fail closed through the DS-R5F replay checks: the recomputed
   session/lattice/signature/policy/graph digests, the complete ordered
   pair-trace record, the candidate counts, the selected match identity digests,
   the baseline reason and the per-axis diagnostics all have to reproduce the v6
   artifact.  A mismatch writes one atomic refused record, returns nonzero and
   runs no removal experiment.
6. Per-endpoint experiments.  For each of the eight conflicting
   ``(frame_id, plane)`` endpoints of the baseline failure:

   * **treatment** -- solve after removing only the matches incident to that
     endpoint;
   * **controls** -- 20 deterministic, without-replacement removals per endpoint,
     matching the treatment's removed-match count per baseline world axis.  The
     baseline world axis of a match is the baseline best assignment's own
     coordinate: ``plane_graph.axis_transform(match.left_axis,
     baseline_quadrants[match.left])``.  Controls are chosen only from matches
     not incident to the target endpoint, are stratified by that baseline axis
     and are ranked by a per-replicate hash key
     ``SHA-256(replicate_seed || canonical match identity)`` so the draw is
     deterministic, exactly stratified and without replacement.  The replicate
     seed is ``SHA-256(frame_id = plane_index # replicate_index)`` on the
     explicit UTF-8 text.

   A bin (endpoint, baseline world axis) that does not hold enough eligible
   controls is recorded as a typed ``control_unavailable``; the matching is never
   relaxed and no substitute row is fabricated.

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

SCHEMA = "mc-photogrammetry-evaluation-graph-endpoint-controls-v1"
OUTPUT_PATH = ROOT / ".tmp-ds-r5g-20261003.json"
ARTIFACT_PATH = ROOT / ".tmp-axis0-graph-stage-attribution-v6-20261003.json"
PROJECT_PATH = ROOT / ".tmp-axis0-graph-stage-attribution-v6-20261003.mcscan"
PNG_DIR = ROOT / ".tmp-axis0-component-preflight-replacement-20261002.png-inputs"
PNG_PATHS = tuple(PNG_DIR / f"image-{index:04d}.png" for index in range(4))
CONTAINED_RUNNER = ROOT / "scripts" / "run_image_derived_graph_stage.py"
JOB_HELPER = ROOT / ".tmp-run-axis0-component-preflight-replacement.py"
DS_R5F_UTILITY = ROOT / "scripts" / "evaluation" / "replay_graph_solver_ablation.py"
DS_R5F_TEST = ROOT / "tests" / "test_graph_solver_ablation.py"
REPORT_PATH = ROOT / "docs" / "reports" / "DS-R5G-conflict-endpoint-controls-2026-10-03.md"

LIMIT_BYTES = 8 * 1024 ** 3
DEADLINE_SECONDS = 600
MAX_SHIFT_CELLS = 32
CONTROLS_PER_ENDPOINT = 20
SEED_SEPARATOR = " # "
AXES = (0, 1, 2)
PRODUCER_PATHS = (
    Path(__file__),
    ROOT / "tests" / "test_graph_endpoint_controls.py",
    CONTAINED_RUNNER,
    JOB_HELPER,
    DS_R5F_UTILITY,
    DS_R5F_TEST,
    REPORT_PATH,
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


class ControlUnavailable(RuntimeError):
    """A control bin holds fewer eligible matches than the treatment removed."""

    def __init__(self, frame: str, plane: int, axis: int, required: int,
                 eligible: int) -> None:
        super().__init__(
            f"endpoint ({frame}, {plane}) axis {axis}: required {required} "
            f"eligible controls, only {eligible} are available")
        self.frame = frame
        self.plane = plane
        self.axis = axis
        self.required = required
        self.eligible = eligible


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
    value: dict[str, Any] = {"schema": SCHEMA, "status": "refused",
                             "reason": reason}
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


def default_identity(match: plane_graph.PlaneMatch) -> Sequence[Any]:
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    return graph_mod._input_match_identity(match)


def canonical_endpoint_identity(frame: object, plane: object) -> tuple[str, int]:
    """Strict ``(frame_id, plane_index)`` endpoint identity for seeding."""
    if type(frame) is not str or not frame:
        raise ValueError("conflicting endpoint frame must be a non-empty string")
    if type(plane) is not int or plane < 0:
        raise ValueError("conflicting endpoint plane index must be non-negative")
    return (frame, plane)


def canonical_endpoint_text(frame: str, plane: int) -> str:
    """Unambiguous UTF-8 identity text; frame ids cannot contain the separator."""
    return f"{canonical_endpoint_identity(frame, plane)[0]}{SEED_SEPARATOR}{plane}"


def replicate_seed(frame: str, plane: int, replicate_index: int) -> str:
    """``SHA-256`` of canonical endpoint identity plus the replicate index."""
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError("replicate index must be a non-negative integer")
    text = f"{canonical_endpoint_text(frame, plane)}{SEED_SEPARATOR}{replicate_index}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def endpoint_incident_indices(
    matches: Sequence[plane_graph.PlaneMatch], frame: str, plane: int,
) -> tuple[int, ...]:
    """Indices of matches incident to one exact ``(frame, plane)`` endpoint."""
    target = canonical_endpoint_identity(frame, plane)
    indices: list[int] = []
    for index, match in enumerate(matches):
        if ((match.left, match.left_plane) == target
                or (match.right, match.right_plane) == target):
            indices.append(index)
    return tuple(indices)


def baseline_match_axis(
    match: plane_graph.PlaneMatch,
    baseline_quadrants: Mapping[str, int],
) -> int:
    """World axis of one match in the baseline best assignment's coordinate.

    ``plane_graph.axis_transform(left_axis, baseline_quadrants[left])`` names the
    world axis the baseline assignment itself gives the left endpoint's local
    normal.  A match the baseline assignment did not activate can still be
    classified, because its left endpoint has one baseline quadrant regardless of
    the match's own relative-quadrant hypothesis.
    """
    frame = match.left
    if frame not in baseline_quadrants:
        raise ReplayUnavailable(
            f"match left frame {frame!r} is absent from the baseline quadrants")
    world, _sign = plane_graph.axis_transform(
        match.left_axis, baseline_quadrants[frame])
    return int(world)


def baseline_axis_counts(
    matches: Sequence[plane_graph.PlaneMatch],
    baseline_quadrants: Mapping[str, int],
    indices: Sequence[int],
) -> tuple[int, int, int]:
    """Per-``(0,1,2)`` baseline world-axis count of one index selection."""
    counts = [0, 0, 0]
    for index in indices:
        counts[baseline_match_axis(matches[index], baseline_quadrants)] += 1
    return (counts[0], counts[1], counts[2])


def control_candidate_indices(
    matches: Sequence[plane_graph.PlaneMatch],
    baseline_quadrants: Mapping[str, int],
    frame: str,
    plane: int,
) -> dict[int, tuple[int, ...]]:
    """Eligible control indices per baseline world axis.

    Eligibility is exactly "not incident to the target endpoint"; the candidate
    is grouped by the baseline assignment's own world axis for its left
    endpoint.  A candidate appears in exactly one bin, so a stratified draw of
    ``required`` per axis is automatically without replacement.
    """
    incident = set(endpoint_incident_indices(matches, frame, plane))
    grouped: dict[int, list[int]] = {axis: [] for axis in AXES}
    for index, match in enumerate(matches):
        if index in incident:
            continue
        grouped[baseline_match_axis(match, baseline_quadrants)].append(index)
    return {axis: tuple(grouped[axis]) for axis in AXES}


def control_removal_selection(
    matches: Sequence[plane_graph.PlaneMatch],
    baseline_quadrants: Mapping[str, int],
    frame: str,
    plane: int,
    required: Mapping[int, int],
    replicate_index: int,
    identity: Callable[[plane_graph.PlaneMatch], Sequence[Any]],
) -> tuple[int, ...]:
    """One deterministic, stratified, without-replacement control removal set.

    Every candidate index is ranked by
    ``SHA-256(replicate_seed || canonical match identity)`` and the lowest-ranked
    entries of each baseline world axis bin are taken, so the draw is
    reproducible, exactly matches the treatment's per-axis removal counts and
    never repeats a match.  A bin with too few eligible candidates raises
    :class:`ControlUnavailable` instead of relaxing the match.
    """
    seed = replicate_seed(frame, plane, replicate_index)
    eligible = control_candidate_indices(matches, baseline_quadrants, frame, plane)
    selected: list[int] = []
    for axis in AXES:
        wanted = int(required.get(axis, 0))
        candidates = eligible[axis]
        if wanted < 0:
            raise ValueError("required control counts must be non-negative")
        if wanted > len(candidates):
            raise ControlUnavailable(frame, plane, axis, wanted, len(candidates))
        ranked = sorted(
            candidates,
            key=lambda index: hashlib.sha256(
                seed.encode("ascii")
                + canonical_json(list(identity(matches[index])))).hexdigest())
        selected.extend(ranked[:wanted])
    return tuple(sorted(selected))


def _axis_observations(solution: Any) -> tuple[int, ...] | None:
    alternatives = getattr(solution, "alternatives", None) or ()
    if not alternatives:
        return None
    counts = getattr(alternatives[0], "axis_observations", None)
    if counts is None:
        return None
    return tuple(counts)


def axis_coverage(solution: Any) -> list[int]:
    """Sorted world axes the returned solution actually reports."""
    resultants = getattr(solution, "axis_resultants", None) or {}
    return sorted(int(axis) for axis in resultants)


def solution_axes(solution: Any) -> list[list[Any]]:
    """The DS-R5F per-axis ``[axis, resultant, residual, observations]`` rows."""
    resultants = getattr(solution, "axis_resultants", None) or {}
    residuals = getattr(solution, "axis_residuals", None) or {}
    observations = _axis_observations(solution)
    return [
        [axis, resultants.get(axis), residuals.get(axis),
         None if observations is None else observations[axis]]
        for axis in range(3)
    ]


def solution_record(
    label: str,
    matches: Sequence[plane_graph.PlaneMatch],
    removed: Sequence[int],
    solution: Any,
    identity: Callable[[plane_graph.PlaneMatch], Sequence[Any]],
) -> dict:
    removed_tuple = tuple(removed)
    axes = solution_axes(solution)
    return {
        "label": label,
        "status": "solved",
        "removed_match_indices": list(removed_tuple),
        "removed_match_count": len(removed_tuple),
        "retained_match_count": len(matches) - len(removed_tuple),
        "removed_match_identities": [list(identity(matches[index]))
                                     for index in removed_tuple],
        "reason": str(getattr(solution, "reason", "")),
        "axis_coverage": axis_coverage(solution),
        "axes": axes,
    }


def unavailable_record(
    label: str,
    matches: Sequence[plane_graph.PlaneMatch],
    removed: Sequence[int],
    error: ControlUnavailable,
) -> dict:
    removed_tuple = tuple(removed)
    return {
        "label": label,
        "status": "control_unavailable",
        "removed_match_indices": list(removed_tuple),
        "removed_match_count": len(removed_tuple),
        "retained_match_count": len(matches) - len(removed_tuple),
        "reason": "control_unavailable",
        "axis_coverage": [],
        "axes": [],
        "unavailable_axis": int(error.axis),
        "required_count": int(error.required),
        "eligible_count": int(error.eligible),
    }


def evaluate_removal(
    *,
    readings: Sequence[Any],
    matches: Sequence[plane_graph.PlaneMatch],
    solver_kwargs: Mapping[str, Any],
    solver: Callable[..., Any],
    removed: Sequence[int],
    label: str,
    identity: Callable[[plane_graph.PlaneMatch], Sequence[Any]],
) -> dict:
    """Solve after removing exactly one index set, reusing the captured inputs."""
    removed_set = set(removed)
    retained = tuple(match for index, match in enumerate(matches)
                     if index not in removed_set)
    solution = solver(readings, retained, **solver_kwargs)
    return solution_record(label, matches, removed, solution, identity)


def _incident_axis_counts(
    matches: Sequence[plane_graph.PlaneMatch],
    baseline_quadrants: Mapping[str, int],
    conflict: tuple[str, int],
) -> tuple[int, int, int]:
    frame, plane = conflict
    return baseline_axis_counts(
        matches, baseline_quadrants,
        endpoint_incident_indices(matches, frame, plane))


def run_endpoint_experiments(
    *,
    readings: Sequence[Any],
    matches: Sequence[plane_graph.PlaneMatch],
    solver_kwargs: Mapping[str, Any],
    conflicting_endpoints: Sequence[Sequence[Any]],
    baseline_quadrants: Mapping[str, int],
    solver: Callable[..., Any],
    identity: Callable[[plane_graph.PlaneMatch], Sequence[Any]] | None = None,
    controls_per_endpoint: int = CONTROLS_PER_ENDPOINT,
    baseline_solution: Any = None,
) -> list[dict]:
    """Treatment plus matched-size controls for every conflicting endpoint."""
    identity_fn = identity or default_identity
    if type(controls_per_endpoint) is not int or controls_per_endpoint < 1:
        raise ValueError("controls per endpoint must be a positive integer")
    experiments: list[dict] = []
    for raw_frame, raw_plane in conflicting_endpoints:
        frame, plane = canonical_endpoint_identity(raw_frame, raw_plane)
        incident = endpoint_incident_indices(matches, frame, plane)
        required = _incident_axis_counts(matches, baseline_quadrants, (frame, plane))
        if baseline_solution is not None and not incident:
            treatment = solution_record(
                "treatment", matches, incident, baseline_solution, identity_fn)
        else:
            treatment = evaluate_removal(
                readings=readings, matches=matches, solver_kwargs=solver_kwargs,
                solver=solver, removed=incident, label="treatment",
                identity=identity_fn)
        eligible = control_candidate_indices(matches, baseline_quadrants, frame, plane)
        replicate_rows: list[dict] = []
        for replicate_index in range(controls_per_endpoint):
            label = f"control-{replicate_index:04d}"
            seed = replicate_seed(frame, plane, replicate_index)
            try:
                removed = control_removal_selection(
                    matches, baseline_quadrants, frame, plane, dict(zip(AXES, required)),
                    replicate_index, identity_fn)
            except ControlUnavailable as error:
                record = unavailable_record(label, matches, (), error)
                record["seed_sha256"] = seed
                replicate_rows.append(record)
                continue
            record = evaluate_removal(
                readings=readings, matches=matches, solver_kwargs=solver_kwargs,
                solver=solver, removed=removed, label=label, identity=identity_fn)
            record["seed_sha256"] = seed
            replicate_rows.append(record)
        experiments.append({
            "endpoint": [frame, plane],
            "endpoint_identity_sha256": hashlib.sha256(
                canonical_endpoint_text(frame, plane).encode("utf-8")).hexdigest(),
            "incident_match_indices": list(incident),
            "treatment_required_axis_counts": list(required),
            "eligible_control_counts": [len(eligible[axis]) for axis in AXES],
            "treatment": treatment,
            "controls": replicate_rows,
        })
    return experiments


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


def validate_replay(artifact: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    """Raise ReplayMismatch unless every recomputed field matches the artifact.

    This is the DS-R5F fail-closed check set, reproduced verbatim so this
    utility can neither skip nor weaken it: both directions of the artifact
    self-digest, the graph result digest in all three carriers, the complete
    ordered pair-trace record, the source digests, the candidate counts, the
    selected match identity digests in both domains, the boundary kept counts,
    the baseline reason and the per-axis diagnostics are all compared.
    """
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
            for path in PRODUCER_PATHS if Path(path).is_file()]


def baseline_axis_requirement(
    baseline_quadrants: Mapping[str, int],
    observation_counts: Sequence[Any],
    matches: Sequence[plane_graph.PlaneMatch],
    alias_edges: Sequence[Sequence[Any]],
) -> tuple[dict[str, int], tuple[int, int, int], tuple[int, int, int]]:
    """Cross-check the baseline best assignment before any control draw.

    The baseline best quadrants are the recomputed solution's own quadrant
    mapping -- the artifact carries no ``selected_quadrants`` when the graph
    abstains, and the alias record's edges were derived from exactly this
    assignment.  Two independent conservations are re-derived and required:

    * every per-axis face count must be even, and halving it must equal the
      number of baseline-active matches on that world axis;
    * every alias edge maps its own ``(left_axis, baseline quadrant)`` to the
      same world axis the record stored, and its frame pair carries the baseline
      relative quadrant.

    Returns the quadrants, the per-axis baseline-active match counts and the
    per-axis alias edge counts.
    """
    quadrants: dict[str, int] = {}
    for frame, quadrant in baseline_quadrants.items():
        frame, _ = canonical_endpoint_identity(frame, 0)
        if type(quadrant) is not int or quadrant not in (0, 1, 2, 3):
            raise ReplayMismatch(
                "baseline.selected_quadrants",
                f"frame {frame!r} carries a non-canonical quadrant")
        quadrants[frame] = quadrant
    active = [index for index, match in enumerate(matches)
              if (match.left in quadrants and match.right in quadrants
                  and (quadrants[match.right] - quadrants[match.left]) % 4
                  == match.quadrant_of_right)]
    active_counts = baseline_axis_counts(matches, quadrants, active)
    face_counts = tuple(
        0 if count is None else int(count) for count in observation_counts)
    if any(count % 2 for count in face_counts):
        raise ReplayMismatch(
            "baseline_axis_observations",
            f"per-axis face counts {face_counts} are not all even")
    derived = tuple(count // 2 for count in face_counts)
    if derived != active_counts:
        raise ReplayMismatch(
            "baseline_active_match_axis_counts",
            f"baseline-active matches per axis {active_counts} disagree with the "
            f"artifact face counts {derived}")
    edge_counts = [0, 0, 0]
    for edge in alias_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 13:
            raise ReplayMismatch("alias_edges", "edge row is malformed")
        match_index, left_frame, left_plane, left_axis = (
            edge[0], edge[1], edge[2], edge[3])
        right_frame, right_plane, right_axis, quadrant, world = (
            edge[6], edge[7], edge[8], edge[11], edge[12])
        try:
            index = int(match_index)
            match = matches[index]
            canonical_endpoint_identity(left_frame, left_plane)
            canonical_endpoint_identity(right_frame, right_plane)
        except (IndexError, TypeError, ValueError) as error:
            raise ReplayMismatch("alias_edges", f"edge row is out of range: {error}")
        if index != match_index or (match.left, match.left_plane) != (
                left_frame, left_plane) or (match.right, match.right_plane) != (
                right_frame, right_plane):
            raise ReplayMismatch(
                "alias_edges", f"edge {index} does not bind its input match")
        if left_frame not in quadrants or right_frame not in quadrants:
            raise ReplayMismatch(
                "alias_edges", f"edge {index} names a frame outside the quadrants")
        if (quadrants[right_frame] - quadrants[left_frame]) % 4 != quadrant:
            raise ReplayMismatch(
                "alias_edges", f"edge {index} disagrees with its quadrants")
        baseline_world, _sign = plane_graph.axis_transform(
            left_axis, quadrants[left_frame])
        right_world, _sign = plane_graph.axis_transform(
            right_axis, quadrants[right_frame])
        if baseline_world != right_world or int(world) != int(baseline_world):
            raise ReplayMismatch(
                "alias_edges", f"edge {index} maps to a different world axis")
        edge_counts[int(baseline_world)] += 1
    if tuple(edge_counts) != active_counts:
        raise ReplayMismatch(
            "baseline_axis_edge_counts",
            f"alias edge counts {tuple(edge_counts)} disagree with baseline-active "
            f"matches {active_counts}")
    return quadrants, active_counts, (edge_counts[0], edge_counts[1], edge_counts[2])


def execute_replay(*, output_path: Path = OUTPUT_PATH,
                   artifact_path: Path = ARTIFACT_PATH,
                   project_path: Path = PROJECT_PATH,
                   png_paths: Sequence[Path] = PNG_PATHS,
                   controls_per_endpoint: int = CONTROLS_PER_ENDPOINT) -> dict:
    """Recompute, fail closed, then run the endpoint treatments and controls."""
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
    baseline_solution = captured["solution"]

    selected, active_counts, declared = baseline_axis_requirement(
        baseline_solution.quadrants,
        tuple(row[3] for row in evidence["axes"]),
        matches, failure.active_edge_aliases.edges)
    require_equal("baseline.active_match_axis_counts", active_counts,
                  tuple(int(value) for value
                        in failure.active_edge_aliases.axis_edge_counts))
    require_equal("baseline.artifact_axis_edge_counts", declared,
                  tuple(int(value) for value
                        in failure.active_edge_aliases.axis_edge_counts))

    experiments = run_endpoint_experiments(
        readings=captured["readings"], matches=matches,
        solver_kwargs=captured["kwargs"], conflicting_endpoints=conflicting,
        baseline_quadrants=selected, solver=recomputed["solver"],
        identity=default_identity, controls_per_endpoint=controls_per_endpoint,
        baseline_solution=None)

    baseline_record = solution_record(
        "baseline", matches, (), baseline_solution, default_identity)
    require_equal("baseline.record_reason", baseline_record["reason"],
                  evidence["reason"])
    require_equal("baseline.record_axes", baseline_record["axes"],
                  evidence["axes"])

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
            "baseline_quadrants": [[frame, selected[frame]]
                                   for frame in sorted(selected)],
            "baseline_axis_edge_counts": [int(value) for value
                                          in failure.active_edge_aliases.axis_edge_counts],
        },
        "effective_settings": {name: value for name, value
                               in attribution.effective_settings},
        "baseline_readings": [baseline_record],
        "experiments": experiments,
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
    except ControlUnavailable as error:
        write_new(OUTPUT_PATH, refusal_payload(
            "control_unavailable", detail=str(error)))
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
        "ds_r5g_contained_runner", CONTAINED_RUNNER)
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
        print(f"DS-R5G output already exists: {OUTPUT_PATH}", file=sys.stderr)
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
