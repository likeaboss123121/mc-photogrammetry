"""DS-IMG-SUPPORT-OWNERSHIP-R1 bounded evaluation harness (evaluation-only).

This harness runs one ordinary image-only Route B inference over the frozen
three-view Template 2 screenshots used by R10/R11 with the **accepted R4**
capture path enabled:

* ``build_image_derived_entry(..., capture_ingress=True)`` binds the typed
  ingress pin to the session frame (no side-channel ingress observer);
* ``derive_image_derived_signatures(..., signature_witness_sink=[...])`` captures
  the ordinary sampler's own ``SignatureSamplingWitnessV1``;
* ``derive_image_derived_graph_attribution(..., candidate_support_sink=...)``
  captures the complete pre-filter candidate-support record from the accepted
  session-bound ingress pins and the session-threaded signature witnesses.

Phase A serializes and seals every image-only record (inputs, ingress pins,
signature witnesses, candidate support, same-axis capture, graph/trace/solve
output) **before** any evaluator material is opened. Phase B then attempts the
evaluator-only projected-ownership join against a caller-supplied, separately
pinned ownership payload.

No real pinned evaluator ownership payload and no pinned inference-camera /
occlusion alignment are available for these inputs, so Phase B records a typed
unavailable and computes no ownership statistic. Synthetic ownership is never
substituted for a real payload and is never reported as a score. The harness
changes no gate, selection, threshold, matcher, solver or production output.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "mc-photogrammetry-support-ownership-r1-v1"
PHASE_A_SCHEMA = "mc-photogrammetry-support-ownership-r1-phase-a-v1"
PHASE_B_SCHEMA = "mc-photogrammetry-support-ownership-r1-phase-b-v1"
REPORT_SCHEMA = "mc-photogrammetry-support-ownership-r1-report-v1"
OWNERSHIP_PAYLOAD_SCHEMA = "mc-photogrammetry-projected-evaluator-ownership-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]

# The frozen three-view Template 2 Route B inputs used by the accepted R10/R11
# runs. Byte pins are the R10 preflight values and are re-verified before any
# stage runs.
DEFAULT_IMAGES = (
    ("templates 2/2026-09-02_13.22.14.png",
     "EB22541171E030FF94D6839A6626E7160C69AE3A780F07AEC0298322BE665D46"),
    ("templates 2/2026-09-02_13.24.24.png",
     "E2FD9FD6C83BF9005FE0973987DA9E46D213A4351EC91393CE578F8E6D3961AE"),
    ("templates 2/2026-09-02_13.24.33.png",
     "35D9DEBB206C2505F65D3BE29183F4A4F3D2BF3F9C05644CE75570F48381D1FC"),
)
# Exact R11 solver settings/constraints (the production defaults).
DEFAULT_SETTINGS = {
    "max_shift_cells": 32,
    "integer_tolerance": 0.15,
    "min_resultant": 0.90,
    "ambiguous_margin": 0.02,
}
DEFAULT_R10_ARTIFACT = ".tmp-geom-prefilter-capture-r10-20261004/observation.json"
R10_FILE_SHA256 = "BC957B684152AEEBB5C055EB12816643F91A506AB01543761D6461FEE4C624B4"
DEFAULT_ROOT = ".tmp-support-ownership-r1-20261004"


class HarnessError(RuntimeError):
    """A precondition, provenance or ordering failure (never a score)."""


class OwnershipPayloadError(RuntimeError):
    """A typed refusal while loading a projected evaluator ownership payload."""

    def __init__(self, reason: str, field: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.field = field


# ---------------------------------------------------------------------------
# Canonical JSON / digest helpers
# ---------------------------------------------------------------------------

def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: object, *, indent: int = 2) -> None:
    temporary = path.with_name(path.name + ".partial")
    data = json.dumps(payload, sort_keys=True, indent=indent,
                      ensure_ascii=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _array_record(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(record: object, *, field: str) -> np.ndarray:
    if not isinstance(record, dict):
        raise OwnershipPayloadError("ownership_payload_unavailable", field)
    dtype = record.get("dtype")
    shape = record.get("shape")
    encoded = record.get("base64")
    if (type(dtype) is not str or not isinstance(shape, list)
            or not all(type(item) is int and item >= 0 for item in shape)
            or type(encoded) is not str):
        raise OwnershipPayloadError("ownership_payload_unavailable", field)
    try:
        array = np.frombuffer(base64.b64decode(encoded, validate=True),
                              dtype=np.dtype(dtype)).reshape(tuple(shape))
    except (ValueError, TypeError) as error:
        raise OwnershipPayloadError(
            "ownership_payload_unavailable", field) from error
    digest = record.get("sha256")
    if digest != hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")).hexdigest():
        raise OwnershipPayloadError("byte_digest_mismatch", field)
    return array


# ---------------------------------------------------------------------------
# Phase A: accepted R4 capture over the ordinary image-only chain
# ---------------------------------------------------------------------------

@dataclass
class PhaseARun:
    """The in-process accepted R4 Phase A capture held until the freeze."""

    entry: object
    lattice: object
    signatures: object
    graph: object
    trace: object
    signature_witnesses: tuple
    candidate_support: object
    same_axis_candidate: object | None
    solver_snapshots: tuple
    project_path: Path
    elapsed_seconds: float

    @property
    def witness_by_signature(self) -> dict:
        from mc_photogrammetry.pipeline.candidate_support_provenance import (
            SourceTapWitnessBindingV1)
        return {item.signature_sha256: item for item in self.signature_witnesses
                if isinstance(item, SourceTapWitnessBindingV1)}


def _run_accepted_r4_chain(
    *,
    paths: tuple[Path, ...],
    project_path: Path,
    settings: dict,
) -> PhaseARun:
    """Run the ordinary image-only Route B chain with the accepted R4 sinks."""
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline.image_derived_entry import (
        build_image_derived_entry)
    from mc_photogrammetry.pipeline.image_derived_graph_attribution import (
        derive_image_derived_graph_attribution)
    from mc_photogrammetry.pipeline.image_derived_lattice_measure import (
        measure_image_derived_lattice)
    from mc_photogrammetry.pipeline.image_derived_signatures import (
        derive_image_derived_signatures)
    from mc_photogrammetry.reconstruction import plane_placement as placement

    started = time.monotonic()
    entry = build_image_derived_entry(
        [str(path) for path in paths], project_path=project_path,
        capture_ingress=True)
    if (getattr(entry.graph, "status", None) != "ok"
            or getattr(entry.pose_graph, "status", None) != "ok"):
        raise HarnessError(
            f"entry graph/pose graph unavailable: {getattr(entry, 'reason', '')}")
    lattice = measure_image_derived_lattice(
        session=entry.session, pose_graph=entry.pose_graph)
    if isinstance(lattice, placement.PlacementRefusal):
        raise HarnessError(f"lattice refusal: {lattice.reason}")
    if getattr(lattice, "status", None) != "assembled":
        raise HarnessError(
            f"lattice unavailable: {getattr(lattice, 'reason', '')}")

    signature_witnesses: list = []
    signatures = derive_image_derived_signatures(
        session=entry.session, lattice_result=lattice,
        signature_witness_sink=signature_witnesses)
    if isinstance(signatures, placement.PlacementRefusal):
        raise HarnessError(f"signature refusal: {signatures.reason}")
    if getattr(signatures, "status", None) != "assembled":
        raise HarnessError(
            f"signatures unavailable: {getattr(signatures, 'reason', '')}")

    candidate_support: list = []
    same_axis_candidates: list = []
    solver_snapshots: list = []
    graph, trace = derive_image_derived_graph_attribution(
        session=entry.session, lattice_result=lattice,
        signatures_result=signatures, connection_graph=entry.graph,
        matcher_witness_sink=[],
        matcher_witness_payload_sink=[],
        witness_budget=None,
        solver_failure_observer=solver_snapshots.append,
        same_axis_candidate_sink=same_axis_candidates.append,
        candidate_support_sink=candidate_support.append,
        signature_witnesses=signature_witnesses,
        same_axis_capture_axes=graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL,
        **settings)
    if isinstance(graph, placement.PlacementRefusal):
        raise HarnessError(f"graph refusal: {graph.reason}")
    if trace is None:
        raise HarnessError("graph attribution trace is unavailable")
    return PhaseARun(
        entry=entry, lattice=lattice, signatures=signatures, graph=graph,
        trace=trace, signature_witnesses=tuple(signature_witnesses),
        candidate_support=(candidate_support[0] if candidate_support else None),
        same_axis_candidate=(same_axis_candidates[0]
                             if same_axis_candidates else None),
        solver_snapshots=tuple(solver_snapshots), project_path=project_path,
        elapsed_seconds=time.monotonic() - started)


def _signature_witness_records(signature_witnesses: tuple) -> list:
    from mc_photogrammetry.pipeline.candidate_support_provenance import (
        SourceTapWitnessBindingV1)
    records = []
    for item in signature_witnesses:
        if isinstance(item, SourceTapWitnessBindingV1):
            record = item.to_record()
            record["outcome"] = "bound"
        else:
            record = dataclasses.asdict(item)
            record["outcome"] = "refused"
        records.append(record)
    return records


def _witness_coverage(signature_witnesses: tuple) -> dict:
    from mc_photogrammetry.pipeline.candidate_support_provenance import (
        SourceTapWitnessBindingV1)
    reasons: dict[str, int] = {}
    bound = 0
    for item in signature_witnesses:
        if isinstance(item, SourceTapWitnessBindingV1):
            bound += 1
        else:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
    return {"total": len(signature_witnesses), "bound": bound,
            "refused": len(signature_witnesses) - bound,
            "refused_by_reason": reasons}


def _r10_reference(path: Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return {"status": "artifact_unavailable", "path": None}
    actual = _file_sha256(Path(path))
    if actual.lower() != R10_FILE_SHA256.lower():
        return {"status": "pin_mismatch", "path": str(path),
                "expected_file_sha256": R10_FILE_SHA256,
                "actual_file_sha256": actual}
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    trace = record.get("trace") or {}
    solver_failure = trace.get("solver_failure") or {}
    active = solver_failure.get("active_edge_aliases") or {}
    graph_payload = (record.get("graph") or {}).get("payload") or {}
    return {
        "status": "loaded", "path": str(path), "file_sha256": actual,
        "graph_sha256": record.get("graph_sha256"),
        "trace_sha256": record.get("trace_sha256"),
        "project_sha256": record.get("project_sha256"),
        "reason": graph_payload.get("reason"),
        "candidate_counts": trace.get("candidate_counts"),
        "solver_axes": solver_failure.get("axes"),
        "first_failed_predicate": solver_failure.get("first_failed_predicate"),
        "failed_assignment_sha256": active.get("failed_assignment_sha256"),
        "input_match_count": active.get("input_match_count"),
        "active_edge_count": active.get("active_edge_count"),
    }


def _phase_a_payload(
    run: PhaseARun,
    *,
    images: tuple[tuple[str, str], ...],
    settings: dict,
    r10_reference: dict,
) -> dict:
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    from mc_photogrammetry.pipeline.image_derived_graph_attribution import (
        solver_failure_snapshot_record)

    ingress = []
    for frame in run.entry.session.frames:
        pin = getattr(frame, "ingress", None)
        ingress.append(None if pin is None else pin.to_record())
    trace_record = run.trace.to_record()
    graph_payload = graph_mod._payload(run.graph)
    support = (None if run.candidate_support is None
               else run.candidate_support.to_record())
    same_axis = (None if run.same_axis_candidate is None
                 else run.same_axis_candidate.to_record())
    capture_graph_sha = getattr(run.graph, "result_sha256", "")
    capture_trace_sha = getattr(run.trace, "attribution_sha256", "")
    reference = r10_reference if r10_reference.get("status") == "loaded" else {}
    solver_failure = trace_record.get("solver_failure") or {}
    comparison = {
        "r10": r10_reference,
        "capture_enabled": {
            "graph_sha256": capture_graph_sha,
            "trace_sha256": capture_trace_sha,
            "project_sha256": _file_sha256(run.project_path),
            "session_sha256": getattr(run.entry.session, "session_sha256", ""),
            "entry_sha256": getattr(run.entry, "entry_sha256", ""),
        },
        "graph_sha256_equal": bool(reference)
        and capture_graph_sha == reference.get("graph_sha256"),
        "trace_sha256_equal": bool(reference)
        and capture_trace_sha == reference.get("trace_sha256"),
        "semantic": {
            "reason_equal": bool(reference)
            and graph_payload.get("reason") == reference.get("reason"),
            "candidate_counts_equal": bool(reference)
            and trace_record.get("candidate_counts") == reference.get("candidate_counts"),
            "solver_axes_equal": bool(reference)
            and solver_failure.get("axes") == reference.get("solver_axes"),
            "active_edge_count_equal": bool(reference)
            and ((solver_failure.get("active_edge_aliases") or {}).get(
                "active_edge_count") == reference.get("active_edge_count")),
            "failed_assignment_equal": bool(reference)
            and ((solver_failure.get("active_edge_aliases") or {}).get(
                "failed_assignment_sha256") == reference.get(
                    "failed_assignment_sha256")),
        },
        "note": ("Provenance-enabled session/entry/graph/trace digests may differ "
                 "because the accepted R4 capture is additive and version-gated; "
                 "only the independent semantic fields are compared."),
    }
    return {
        "schema": PHASE_A_SCHEMA,
        "capture_ingress": True,
        "images": [{"path": str(path), "sha256": digest}
                   for path, digest in images],
        "settings": dict(settings),
        "stage_digests": {
            "entry_sha256": getattr(run.entry, "entry_sha256", ""),
            "session_sha256": getattr(run.entry.session, "session_sha256", ""),
            "lattice_sha256": getattr(run.lattice, "result_sha256", ""),
            "signatures_sha256": getattr(run.signatures, "result_sha256", ""),
            "graph_sha256": capture_graph_sha,
            "trace_sha256": capture_trace_sha,
        },
        "ingress": ingress,
        "signature_witnesses": {
            "coverage": _witness_coverage(run.signature_witnesses),
            "records": _signature_witness_records(run.signature_witnesses),
        },
        "candidate_support": support,
        "same_axis_candidate": same_axis,
        "graph": {"payload": graph_payload,
                  "result_sha256": capture_graph_sha},
        "trace": trace_record,
        "raw_solver_failure_bounded": [
            solver_failure_snapshot_record(snapshot)
            for snapshot in run.solver_snapshots],
        "comparison": comparison,
        "elapsed_seconds": run.elapsed_seconds,
    }


def freeze_phase_a(root: Path, phase_a: dict) -> dict:
    """Seal Phase A and write it; returns the recorded file/seal digests."""
    payload = dict(phase_a)
    payload.pop("phase_a_seal_sha256", None)
    seal = _digest(payload)
    payload["phase_a_seal_sha256"] = seal
    path = Path(root) / "phase_a.json"
    _atomic_write(path, payload)
    file_sha256 = _file_sha256(path)
    replayed = json.loads(path.read_text(encoding="utf-8"))
    replayed.pop("phase_a_seal_sha256", None)
    if _digest(replayed) != seal:
        raise HarnessError("Phase A seal does not replay after write")
    return {"path": str(path), "file_sha256": file_sha256,
            "phase_a_seal_sha256": seal}


# ---------------------------------------------------------------------------
# Phase B: evaluator-only projected-ownership join (real pinned payload only)
# ---------------------------------------------------------------------------

def load_projected_ownership(path: Path) -> dict:
    """Load one real pinned ``ProjectedEvaluatorOwnershipV1`` record per frame.

    Only a caller-supplied evaluator payload is accepted. No synthetic ownership
    is generated here, and a malformed or inconsistent payload is a typed
    refusal, never a defaulted owner.
    """
    from mc_photogrammetry.pipeline.candidate_support_provenance import (
        ProjectedEvaluatorOwnershipV1)
    from mc_photogrammetry.pipeline.contracts import FrameTransform

    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OwnershipPayloadError(
            "ownership_payload_unavailable", "payload") from error
    if not isinstance(record, dict) or record.get("schema") != OWNERSHIP_PAYLOAD_SCHEMA:
        raise OwnershipPayloadError("ownership_payload_unavailable", "schema")
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames:
        raise OwnershipPayloadError("ownership_payload_unavailable", "frames")
    loaded: dict[str, object] = {}
    for item in frames:
        if not isinstance(item, dict):
            raise OwnershipPayloadError("ownership_payload_unavailable", "frame")
        frame_id = item.get("frame_id")
        if type(frame_id) is not str or not frame_id or frame_id in loaded:
            raise OwnershipPayloadError("ownership_payload_unavailable", "frame_id")
        ownership = _decode_array(item.get("ownership"), field="ownership")
        usable = _decode_array(item.get("usable"), field="usable").astype(bool)
        if ownership.ndim != 2 or usable.shape != ownership.shape:
            raise OwnershipPayloadError("dimension_mismatch", "ownership")
        frame_size = item.get("frame_size")
        if (not isinstance(frame_size, list) or len(frame_size) != 2
                or not all(type(value) is int for value in frame_size)):
            raise OwnershipPayloadError("dimension_mismatch", "frame_size")
        transform_record = item.get("transform")
        if not isinstance(transform_record, dict):
            raise OwnershipPayloadError("transform_mismatch", "transform")
        try:
            transform = FrameTransform(
                source_size=tuple(transform_record["source_size"]),
                crop_origin=tuple(transform_record.get("crop_origin", (0, 0))),
                crop_size=(None if transform_record.get("crop_size") is None
                           else tuple(transform_record["crop_size"])),
                output_size=(None if transform_record.get("output_size") is None
                             else tuple(transform_record["output_size"])))
        except (KeyError, TypeError, ValueError) as error:
            raise OwnershipPayloadError(
                "transform_mismatch", "transform") from error
        camera_sha256 = item.get("camera_sha256")
        occlusion = item.get("occlusion_convention")
        if (type(camera_sha256) is not str or len(camera_sha256) != 64
                or type(occlusion) is not str or not occlusion):
            raise OwnershipPayloadError(
                "evaluator_camera_misalignment", "camera")
        keys = item.get("owner_plane_keys") or {}
        owner_plane_keys = {}
        for key, value in keys.items():
            try:
                owner_plane_keys[int(key)] = (None if value is None
                                              else tuple(value)
                                              if isinstance(value, list) else value)
            except (TypeError, ValueError) as error:
                raise OwnershipPayloadError(
                    "unresolved_physical_plane_identity", "owner_plane_keys") from error
        loaded[frame_id] = ProjectedEvaluatorOwnershipV1(
            ownership=ownership, usable=usable, frame_size=tuple(frame_size),
            camera_sha256=camera_sha256, transform=transform,
            occlusion_convention=occlusion, owner_plane_keys=owner_plane_keys)
    return loaded


def _project_support_rows(
    *,
    support_rows: list,
    witness_by_signature: dict,
    evaluator_by_frame: dict,
    inference_camera_sha256: str,
    occlusion_convention: str,
) -> dict:
    """Pure Phase B join: frozen source-tap witness -> projected ownership."""
    from mc_photogrammetry.pipeline.candidate_support_provenance import (
        SourceTapWitnessBindingV1, project_ownership_into_source_space)

    totals = {"rows": len(support_rows), "projected": 0, "kept": 0, "dropped": 0}
    by_disposition: dict[str, dict] = {}
    abstentions: dict[str, int] = {}
    states: dict[str, int] = {"pure": 0, "mixed": 0, "unknown": 0}
    for row in support_rows:
        disposition = row["disposition"]
        bucket = by_disposition.setdefault(
            disposition, {"rows": 0, "projected": 0, "pure": 0, "mixed": 0,
                          "unknown": 0, "abstained": 0})
        bucket["rows"] += 1
        totals["kept" if disposition == "kept" else "dropped"] += 1
        for digest in (row.get("left_signature_sha256"),
                       row.get("right_signature_sha256")):
            binding = witness_by_signature.get(digest or "")
            if not isinstance(binding, SourceTapWitnessBindingV1):
                abstentions["source_tap_witness_unavailable"] = abstentions.get(
                    "source_tap_witness_unavailable", 0) + 1
                bucket["abstained"] += 1
                continue
            evaluator = evaluator_by_frame.get(binding.frame_id)
            if evaluator is None:
                abstentions["source_projection_unavailable"] = abstentions.get(
                    "source_projection_unavailable", 0) + 1
                bucket["abstained"] += 1
                continue
            projection = project_ownership_into_source_space(
                binding=binding, evaluator=evaluator,
                camera_sha256=inference_camera_sha256,
                inference_transform=binding.ingress.transform,
                occlusion_convention=occlusion_convention)
            reason = getattr(projection, "reason", None)
            if reason is not None:
                abstentions[reason] = abstentions.get(reason, 0) + 1
                bucket["abstained"] += 1
                continue
            totals["projected"] += 1
            bucket["projected"] += 1
            for state in projection.cell_states:
                states[state] = states.get(state, 0) + 1
                bucket[state] = bucket.get(state, 0) + 1
    return {"totals": totals, "by_disposition": by_disposition,
            "cell_states": states, "abstentions": abstentions}


def execute_phase_b(
    *,
    phase_a: dict,
    ownership_payload: Path | None,
    witness_by_signature: dict,
    inference_camera_sha256: str = "",
    occlusion_convention: str = "",
    loader=load_projected_ownership,
) -> dict:
    """Attempt the evaluator-only join; refuse typed when it cannot be proven."""
    base = {"schema": PHASE_B_SCHEMA,
            "phase_a_seal_sha256": phase_a.get("phase_a_seal_sha256")}
    if ownership_payload is None:
        base.update({
            "status": "unavailable", "reason": "ownership_payload_unavailable",
            "field": "ownership_payload",
            "detail": ("no real pinned evaluator ownership payload was supplied; "
                       "no ownership statistic is computed and no synthetic "
                       "ownership is reported as a score")})
        return base
    if not Path(ownership_payload).is_file():
        base.update({
            "status": "unavailable", "reason": "ownership_payload_unavailable",
            "field": "path",
            "detail": "the pinned ownership payload path does not exist"})
        return base
    support = phase_a.get("candidate_support")
    if not isinstance(support, dict) or support.get("status") != "assembled":
        base.update({
            "status": "unavailable", "reason": "candidate_support_unavailable",
            "field": (support or {}).get("field") or "rows",
            "detail": ("Phase A candidate-support capture did not assemble, so "
                       "there are no frozen candidate rows to project")})
        return base
    if not inference_camera_sha256 or not occlusion_convention:
        base.update({
            "status": "unavailable",
            "reason": "evaluator_camera_misalignment"
            if not inference_camera_sha256 else "occlusion_unavailable",
            "field": "camera" if not inference_camera_sha256 else "occlusion",
            "detail": ("no pinned inference camera digest / occlusion convention "
                       "exists, so evaluator alignment cannot be proven")})
        return base
    try:
        evaluator_by_frame = loader(Path(ownership_payload))
    except OwnershipPayloadError as error:
        base.update({"status": "unavailable", "reason": error.reason,
                     "field": error.field})
        return base
    try:
        projection = _project_support_rows(
            support_rows=list(support.get("rows") or []),
            witness_by_signature=witness_by_signature,
            evaluator_by_frame=evaluator_by_frame,
            inference_camera_sha256=inference_camera_sha256,
            occlusion_convention=occlusion_convention)
    except Exception as error:  # noqa: BLE001 - typed refusal, never a score
        base.update({"status": "unavailable", "reason": "source_projection_unavailable",
                     "field": "join", "detail": type(error).__name__})
        return base
    base.update({"status": "assembled", "ownership_payload": str(ownership_payload),
                 "inference_camera_sha256": inference_camera_sha256,
                 "occlusion_convention": occlusion_convention,
                 "projection": projection,
                 "note": ("Renderer-conditioned appearance/visibility proxy only; "
                          "no feature-point truth, no accuracy claim, and no "
                          "production/promotion input.")})
    return base


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _resolve_images(images: tuple[tuple[str, str], ...]) -> tuple[tuple[Path, str], ...]:
    resolved = []
    for name, digest in images:
        path = Path(name)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise HarnessError(f"pinned input is missing: {path}")
        actual = _file_sha256(path)
        if actual.lower() != digest.lower():
            raise HarnessError(f"pinned input mismatch: {path}")
        resolved.append((path, digest))
    return tuple(resolved)


def run(
    *,
    root: Path,
    images: tuple[tuple[str, str], ...] = DEFAULT_IMAGES,
    settings: dict | None = None,
    ownership_payload: Path | None = None,
    r10_artifact: Path | None = None,
    inference_camera_sha256: str = "",
    occlusion_convention: str = "",
    chain=None,
    ownership_loader=load_projected_ownership,
) -> dict:
    """Run Phase A, freeze it, then run Phase B. Ordering is enforced here."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise HarnessError(f"the output root is not fresh: {root}")
    root.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_images(images)
    effective_settings = dict(DEFAULT_SETTINGS if settings is None else settings)
    reference = _r10_reference(r10_artifact)
    runner = chain or _run_accepted_r4_chain
    run_result = runner(
        paths=tuple(path for path, _ in resolved),
        project_path=root / "phase_a.mcscan",
        settings=effective_settings)
    phase_a = _phase_a_payload(
        run_result, images=tuple((str(path), digest) for path, digest in resolved),
        settings=effective_settings, r10_reference=reference)
    freeze = freeze_phase_a(root, phase_a)
    # Only after the freeze is recorded may evaluator material be opened.
    phase_b = execute_phase_b(
        phase_a=phase_a, ownership_payload=ownership_payload,
        witness_by_signature=run_result.witness_by_signature,
        inference_camera_sha256=inference_camera_sha256,
        occlusion_convention=occlusion_convention,
        loader=ownership_loader)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "capture": "accepted-r4-session-bound",
        "phase_a": {"file": freeze["path"],
                    "file_sha256": freeze["file_sha256"],
                    "phase_a_seal_sha256": freeze["phase_a_seal_sha256"],
                    "capture_ingress": True,
                    "stage_digests": phase_a["stage_digests"],
                    "signature_witness_coverage":
                        phase_a["signature_witnesses"]["coverage"],
                    "candidate_support":
                        None if phase_a["candidate_support"] is None
                        else {key: phase_a["candidate_support"].get(key)
                              for key in ("status", "reason", "field",
                                          "row_count", "record_sha256")},
                    "comparison": phase_a["comparison"]},
        "phase_b": phase_b,
    }
    report["report_sha256"] = _digest({key: value for key, value in report.items()
                                       if key != "report_sha256"})
    _atomic_write(root / "report.json", report)
    report["report_file_sha256"] = _file_sha256(root / "report.json")
    return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DS-IMG-SUPPORT-OWNERSHIP-R1 bounded evaluation harness")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="fresh output root for the Phase A freeze")
    parser.add_argument("--image", action="append", default=None,
                        help="pinned screenshot path (repeatable, in order)")
    parser.add_argument("--sha256", action="append", default=None,
                        help="expected screenshot SHA-256 (repeatable, in order)")
    parser.add_argument("--ownership-payload", default=None,
                        help="real pinned projected evaluator ownership payload")
    parser.add_argument("--r10-artifact", default=DEFAULT_R10_ARTIFACT,
                        help="retained R10 observation used only for comparison")
    parser.add_argument("--inference-camera-sha256", default="")
    parser.add_argument("--occlusion-convention", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    images = DEFAULT_IMAGES
    if args.image is not None or args.sha256 is not None:
        if (args.image is None or args.sha256 is None
                or len(args.image) != len(args.sha256)):
            print("--image and --sha256 must be supplied together and in equal "
                  "numbers", file=sys.stderr)
            return 2
        images = tuple(zip(args.image, args.sha256))
    report = run(
        root=Path(args.root), images=images,
        ownership_payload=(None if args.ownership_payload is None
                           else Path(args.ownership_payload)),
        r10_artifact=(None if args.r10_artifact is None
                      else Path(args.r10_artifact)),
        inference_camera_sha256=args.inference_camera_sha256,
        occlusion_convention=args.occlusion_convention)
    print(json.dumps({"status": report["status"],
                      "phase_a_seal_sha256": report["phase_a"]["phase_a_seal_sha256"],
                      "candidate_support": report["phase_a"]["candidate_support"],
                      "phase_b": {"status": report["phase_b"]["status"],
                                  "reason": report["phase_b"].get("reason")},
                      "report_file_sha256": report.get("report_file_sha256")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
