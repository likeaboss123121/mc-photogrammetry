"""Experimental image-only pipeline runner with a deterministic freeze format.

The public entry point accepts ordered RGB image paths and one output path.
It has no dataset, capture, reference, label, or manual-mask inputs. This module
is intentionally not wired into the normal CLI or reconstruction defaults.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import cv2
import numpy as np

from mc_photogrammetry.vision.gui_mask import GuiMaskConfig, create_gui_mask
from . import calibration as calibration_module
from . import lattice, lattice_planes, plane_graph, region_growth, supported
from .boundary_candidates import PlaneProposalResult
from .calibration import CalibrationResult, estimate_image_calibrations


SCHEMA_VERSION = 1
MAX_INPUT_FRAMES = 32
MAX_INPUT_PIXELS = 6_000_000
MAX_TOTAL_INPUT_PIXELS = 48_000_000
MAX_SIGNATURE_CELLS = 256
MAX_READINGS_PER_FRAME = 6
MAX_PROPOSALS_PER_FRAME = 32
MAX_SIDECAR_BYTES = 512 * 1024 * 1024
MAX_FREEZE_JSON_BYTES = 64 * 1024 * 1024
_FORBIDDEN_KEYS = frozenset({
    "gt", "ground_truth", "reference", "manifest", "capture", "annotation",
    "manual_mask", "camera", "pose", "bounds", "labels", "label_data",
})


class FrozenImageRunError(ValueError):
    """Invalid runner arguments or an unverifiable frozen artifact."""


class _FreezeBudgetExceeded(FrozenImageRunError):
    """Fatal writer refusal; never convert it to a partial-stage abstention."""


class _Sidecar:
    def __init__(self) -> None:
        self.data = bytearray()
        self.ranges: list[tuple[int, int]] = []

    def add(self, payload: bytes, *, shape: tuple[int, ...], dtype: str,
            encoding: str = "raw") -> dict[str, Any]:
        offset = len(self.data)
        if len(payload) > MAX_SIDECAR_BYTES - offset:
            raise _FreezeBudgetExceeded("mask sidecar exceeds the size cap")
        self.data.extend(payload)
        end = len(self.data)
        self.ranges.append((offset, end))
        return {
            "offset": offset,
            "length": len(payload),
            "shape": list(shape),
            "dtype": dtype,
            "encoding": encoding,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def _canonical_json(value: Any, *, max_bytes: int | None = None) -> bytes:
    """Preserve canonical bytes while bounding the writer buffer before growth."""
    limit = MAX_FREEZE_JSON_BYTES if max_bytes is None else max_bytes
    # Cheap lower bound rejects oversized strings before the encoder creates a
    # complete escaped token. UTF-8 and escape expansion are checked below.
    stack = [value]
    lower_bytes = 0
    while stack:
        item = stack.pop()
        lower_bytes += 1
        if isinstance(item, str):
            lower_bytes += len(item)
        elif isinstance(item, dict):
            lower_bytes += sum(len(str(key)) for key in item)
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        if lower_bytes > limit:
            raise _FreezeBudgetExceeded("freeze JSON exceeds the size cap")
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
    data = bytearray()
    for chunk in encoder.iterencode(value):
        for start in range(0, len(chunk), 65536):
            encoded = chunk[start:start + 65536].encode("utf-8")
            if len(encoded) > limit - len(data):
                raise _FreezeBudgetExceeded("freeze JSON exceeds the size cap")
            data.extend(encoded)
    return bytes(data)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish one complete file from a task-local temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:
    """Convert module dataclasses/numpy values to finite JSON primitives."""
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def _calibration_record(result: CalibrationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "focal_pixels": _json_value(result.focal_pixels),
        "yaw_degrees": _json_value(result.yaw_degrees),
        "pitch_degrees": _json_value(result.pitch_degrees),
        "roll_degrees": _json_value(result.roll_degrees),
        "principal_point": _json_value(result.principal_point),
        "width": int(result.width),
        "height": int(result.height),
        "inliers": int(result.inliers),
        "segments": int(result.segments),
        "median_residual": _json_value(result.median_residual),
        "per_axis_inliers": [int(v) for v in result.per_axis_inliers],
        "notes": list(result.notes),
    }


def _reading_record(reading: lattice_planes.PlaneReading) -> dict[str, Any]:
    rect = reading.rectification
    return {
        "frame": reading.frame,
        "axis": int(reading.axis),
        "side": _json_value(reading.side),
        "alternatives": [
            {"blocks_per_period": _json_value(value.blocks_per_period),
             "distance_blocks": _json_value(value.distance_blocks),
             "support": _json_value(value.support)}
            for value in reading.alternatives
        ],
        "period_pixels": _json_value(reading.period_pixels),
        "phase_pixels": _json_value(reading.phase_pixels),
        "extent": _json_value(reading.extent),
        "samples": int(reading.samples),
        "polygon": _json_value(reading.polygon),
        "coverage_cells": int(reading.coverage_cells),
        "structural_share": _json_value(reading.structural_share),
        "confidence": _json_value(reading.confidence),
        "rectification": {
            "axis_matrix": _json_value(rect.axis_matrix),
            "focal": _json_value(rect.focal),
            "principal_point": _json_value(rect.principal_point),
            "centre": _json_value(rect.centre),
        },
    }


def _match_record(match: plane_graph.PlaneMatch) -> dict[str, Any]:
    # Frozen schema-v1 is intentionally pinned to its original field set.
    # ``cell_shift_rc`` is an additive in-memory residual, not a serialized
    # absolute placement cue.
    return _json_value({
        "left": match.left,
        "right": match.right,
        "left_axis": match.left_axis,
        "right_axis": match.right_axis,
        "left_b": match.left_b,
        "right_b": match.right_b,
        "quadrant_of_right": match.quadrant_of_right,
        "psr": match.psr,
        "peak": match.peak,
        "second_peak": match.second_peak,
        "overlap_cells": match.overlap_cells,
        "integer_residual": match.integer_residual,
        "score": match.score,
        "left_plane": match.left_plane,
        "right_plane": match.right_plane,
    })


def _refusal_record(refusal: plane_graph.PlaneMatchRefusal) -> dict[str, Any]:
    return _json_value(asdict(refusal))


def _solution_record(solution: plane_graph.ComponentSolution) -> dict[str, Any]:
    return {
        "frames": list(solution.frames),
        "quadrants": {str(k): int(v) for k, v in sorted(solution.quadrants.items())},
        "translations": {
            str(k): _json_value(v) for k, v in sorted(solution.translations.items())
        },
        "axis_residuals": {str(k): _json_value(v)
                            for k, v in sorted(solution.axis_residuals.items())},
        "axis_resultants": {str(k): _json_value(v)
                             for k, v in sorted(solution.axis_resultants.items())},
        "scale_known": bool(solution.scale_known),
        "blocks_per_period": [
            {"frame": str(key[0]), "axis": int(key[1]), "value": _json_value(value)}
            for key, value in sorted(solution.blocks_per_period.items())
        ],
        "reason": str(solution.reason),
        "alternatives": [
            {"quadrants": list(item.quadrants), "cost": _json_value(item.cost),
             "resultants": _json_value(item.resultants),
             "axis_observations": list(item.axis_observations)}
            for item in solution.alternatives
        ],
    }


def _array_bytes(array: np.ndarray) -> tuple[bytes, tuple[int, ...], str]:
    value = np.asarray(array, dtype="<f4", order="C")
    return value.tobytes(order="C"), tuple(int(v) for v in value.shape), "float32-le"


def _code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "calibration": Path(calibration_module.__file__),
        "region_growth": Path(region_growth.__file__),
        "lattice": Path(lattice.__file__),
        "lattice_planes": Path(lattice_planes.__file__),
        "plane_graph": Path(plane_graph.__file__),
        "supported": Path(supported.__file__),
        "gui_mask": Path(create_gui_mask.__code__.co_filename),
    }
    return {name: _sha256(path.read_bytes()) for name, path in sorted(paths.items())}


def _configuration() -> dict[str, Any]:
    return {
        "input": {"max_frames": MAX_INPUT_FRAMES,
                  "max_pixels_per_frame": MAX_INPUT_PIXELS,
                  "order": "caller sequence preserved"},
        "gui_mask": asdict(GuiMaskConfig()),
        "calibration": {
            "grouping": "independent",
            "minimum_length": calibration_module.MINIMUM_LENGTH,
            "threshold": calibration_module.INLIER_PIXELS,
            "scales": list(calibration_module.DETECTION_SCALES),
            "fov_range": [30.0, 120.0],
            "coarse_segments": calibration_module.COARSE_SEGMENTS,
            "refine_optical_centre": True,
            "weight_by_contrast": False,
            "excluded_pixel_fill": 0,
        },
        "proposals": {"config": asdict(region_growth.DEFAULT_REGION_GROWTH_CONFIG),
                      "max_candidates": MAX_PROPOSALS_PER_FRAME},
        "readings": {"max_readings": MAX_READINGS_PER_FRAME,
                     "min_structural_share": lattice_planes.MIN_STRUCTURAL_SHARE,
                     "max_rectified_side": lattice_planes.MAX_RECTIFIED_SIDE},
        "signatures": {"max_cells": MAX_SIGNATURE_CELLS},
        "matching": {"max_shift_cells": 32, "max_structural_share": None},
        "solver": {"integer_tolerance": 0.15, "min_resultant": 0.90,
                   "ambiguous_margin": 0.02},
        "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
    }


def _valid_rgb(image: Any) -> bool:
    return (isinstance(image, np.ndarray) and image.ndim == 3
            and image.shape[2] == 3 and image.shape[0] > 0 and image.shape[1] > 0
            and image.dtype == np.uint8)


def _load_rgb(path: Path) -> tuple[bytes, np.ndarray | None, str | None]:
    try:
        payload = path.read_bytes()
    except OSError:
        return b"", None, "image_read_failed"
    try:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            return payload, None, "image_decode_failed"
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if image.shape[0] * image.shape[1] > MAX_INPUT_PIXELS:
            return payload, image, "image_pixel_cap"
        return payload, image, None
    except (cv2.error, ValueError, TypeError, MemoryError):
        return payload, None, "image_decode_failed"


def _abstain_proposal(reason: str) -> PlaneProposalResult:
    return PlaneProposalResult(status="abstain", reason=reason, proposals=(), stats={})


def _has_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).casefold() in _FORBIDDEN_KEYS
                   or _has_forbidden_key(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_forbidden_key(child) for child in value)
    return False


def _stage_counts(frames: list[dict[str, Any]], graph: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("image", "calibration", "proposals", "readings",
                                      "signatures", "matching", "solution")
    }
    for frame in frames:
        counts["image"][str(frame["image_reason"] or "ok")] += 1
        counts["calibration"][str(frame["calibration_reason"] or "ok")] += 1
        counts["proposals"][str(frame["proposal"]["reason"] or "ok")] += 1
        if frame["readings"]:
            counts["readings"]["ok"] += 1
        else:
            abstentions = frame["reading_abstentions"]
            reason = str(abstentions[0]["reason"]) if abstentions else "no_reading"
            counts["readings"][reason] += 1
        if frame["signatures"]:
            counts["signatures"]["ok"] += 1
        else:
            reason = (str(frame["signature_abstentions"][0]["reason"])
                      if frame["signature_abstentions"] else "no_signature")
            counts["signatures"][reason] += 1
    counts["matching"][str(graph["matching_reason"] or "ok")] += 1
    counts["solution"][str(graph["solution"]["reason"] or "ok")] += 1
    return {stage: dict(sorted(counter.items())) for stage, counter in counts.items()}


def _artifact_id(document: dict[str, Any]) -> str:
    # Only these two dictionaries are modified; a JSON round-trip would create
    # an unbounded intermediate serialization before the writer's size check.
    content = dict(document)
    content.pop("artifact_id", None)
    sidecar = content.get("mask_sidecar")
    if isinstance(sidecar, dict):
        content["mask_sidecar"] = {key: value for key, value in sidecar.items() if key != "file"}
    return _sha256(_canonical_json(content))


def run_frozen_image_pipeline(
    image_paths: Iterable[str | Path],
    freeze_path: str | Path,
) -> dict[str, Any]:
    """Run the frozen image-only stages and write JSON plus a binary sidecar.

    ``image_paths`` order is authoritative and retained in the artifact. The
    output path is never included in the artifact id. Only the passed image
    bytes are read as data; neighboring files are never inspected.
    """
    if isinstance(image_paths, (str, bytes, Path)):
        raise FrozenImageRunError("image_paths must be an ordered sequence of paths")
    try:
        path_values = list(islice(image_paths, MAX_INPUT_FRAMES + 1))
        if len(path_values) > MAX_INPUT_FRAMES:
            raise FrozenImageRunError(f"input path count must be between 1 and {MAX_INPUT_FRAMES}")
        if any(not isinstance(value, (str, Path)) for value in path_values):
            raise FrozenImageRunError("every image input must be a path")
        paths = [Path(value) for value in path_values]
    except (TypeError, OSError) as exc:
        raise FrozenImageRunError("image_paths is not an ordered path sequence") from exc
    if not paths or len(paths) > MAX_INPUT_FRAMES:
        raise FrozenImageRunError(f"input path count must be between 1 and {MAX_INPUT_FRAMES}")
    if any(not path.name or path.name in {".", ".."} for path in paths):
        raise FrozenImageRunError("every image path must have a basename")
    basenames = [path.name for path in paths]
    if len(set(basenames)) != len(basenames):
        raise FrozenImageRunError("input image basenames must be unique")
    destination = Path(freeze_path)
    if destination.suffix.lower() != ".json":
        raise FrozenImageRunError("freeze_path must name a .json file")

    loaded: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    usable_masks: dict[str, np.ndarray] = {}
    retained_pixels = 0
    for path in paths:
        payload, image, image_reason = _load_rgb(path)
        frame_id = path.name
        width = int(image.shape[1]) if image is not None else 0
        height = int(image.shape[0]) if image is not None else 0
        processable_image = image if image_reason is None else None
        if processable_image is not None:
            frame_pixels = width * height
            if frame_pixels > MAX_TOTAL_INPUT_PIXELS - retained_pixels:
                raise FrozenImageRunError("aggregate retained image pixels exceed the size cap")
            retained_pixels += frame_pixels
            images[frame_id] = processable_image
            usable_masks[frame_id] = ~create_gui_mask(height, width)
        loaded.append({
            "frame_id": frame_id,
            "input": {"basename": path.name, "width": width, "height": height,
                      "sha256": None if image_reason == "image_read_failed" else _sha256(payload)},
            "image_reason": image_reason,
            "image": processable_image,
            "usable": usable_masks.get(frame_id),
        })

    try:
        calibrated = estimate_image_calibrations(images) if images else {}
    except (cv2.error, ValueError, TypeError, IndexError, FloatingPointError,
            OverflowError, MemoryError):
        calibrated = {frame_id: None for frame_id in images}

    sidecar = _Sidecar()
    frame_records: list[dict[str, Any]] = []
    all_readings = []
    all_signatures = []
    for loaded_frame in loaded:
        frame_id = loaded_frame["frame_id"]
        image = loaded_frame["image"]
        usable = loaded_frame["usable"]
        calibration = calibrated.get(frame_id)
        calibration_reason = (None if calibration is not None else
                              ("insufficient_image_evidence" if image is not None
                               else "image_unavailable"))
        if image is None:
            proposal_result = _abstain_proposal(str(loaded_frame["image_reason"]))
            readings: tuple = ()
            reading_diagnostics: dict[str, Any] = {"abstentions": [{
                "frame": frame_id, "axis": None, "side": None,
                "reason": "no_plane_region", "detail": "image unavailable"}],
                "segments": 0, "low_structural_share": 0}
        else:
            try:
                proposal_result = region_growth.propose_image_planes_v2(
                    image, usable, calibration,
                    max_candidates=MAX_PROPOSALS_PER_FRAME)
            except (cv2.error, ValueError, TypeError, IndexError, FloatingPointError,
                    OverflowError, MemoryError):
                proposal_result = _abstain_proposal("proposal_stage_failed")
            if not isinstance(proposal_result, PlaneProposalResult):
                proposal_result = _abstain_proposal("invalid_proposal_result")
            elif (not isinstance(proposal_result.stats, dict)
                  or _has_forbidden_key(proposal_result.stats)):
                proposal_result = _abstain_proposal("invalid_proposal_stats")
            reading_diagnostics = {"abstentions": [], "segments": 0,
                                   "low_structural_share": 0}
            try:
                readings = lattice_planes.frame_plane_readings_from_proposals(
                    image, usable, calibration, proposal_result, frame=frame_id,
                    max_readings=MAX_READINGS_PER_FRAME,
                    max_rectified_side=lattice_planes.MAX_RECTIFIED_SIDE,
                    diagnostics=reading_diagnostics)
            except (cv2.error, ValueError, TypeError, IndexError, FloatingPointError,
                    OverflowError, MemoryError):
                readings = ()
                reading_diagnostics["abstentions"].append({
                    "frame": frame_id, "axis": None, "side": None,
                    "reason": "unmeasurable_plane", "detail": "reading_stage_failed"})

        proposal_records = []
        proposals = proposal_result.proposals[:MAX_PROPOSALS_PER_FRAME]
        if image is not None:
            for proposal in proposals:
                try:
                    mask = np.asarray(proposal.mask, dtype=bool)
                    if mask.shape != image.shape[:2]:
                        continue
                    # Intersect into a new array; the producer's mask remains untouched.
                    mask = mask & usable
                    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
                    mask_ref = sidecar.add(packed, shape=tuple(mask.shape), dtype="bool",
                                           encoding="packbits-little")
                    proposal_records.append({
                        "axis": int(proposal.axis), "side": _json_value(proposal.side),
                        "polygon": _json_value(proposal.polygon),
                        "strength": _json_value(proposal.strength),
                        "sources": list(proposal.sources),
                        "coverage": _json_value(proposal.coverage), "mask": mask_ref,
                    })
                except _FreezeBudgetExceeded:
                    raise
                except (ValueError, TypeError, IndexError):
                    continue

        reading_records = []
        signature_records = []
        signature_abstentions = []
        for reading_index, reading in enumerate(readings):
            reading_records.append(_reading_record(reading))
            all_readings.append(reading)
            for alternative_index, alternative in enumerate(reading.alternatives):
                try:
                    signature = lattice_planes.plane_signature(
                        image, reading, alternative.blocks_per_period,
                        usable=usable, max_cells=MAX_SIGNATURE_CELLS)
                    descriptors = {}
                    for name, value in (("cells", signature.cells),
                                        ("interior", signature.interior),
                                        ("coverage", signature.coverage)):
                        raw, shape, dtype = _array_bytes(value)
                        descriptors[name] = sidecar.add(raw, shape=shape, dtype=dtype)
                    signature_records.append({
                        "reading_index": reading_index,
                        "alternative_index": alternative_index,
                        "blocks_per_period": _json_value(signature.blocks_per_period),
                        "origin_uv": _json_value(signature.origin_uv),
                        "arrays": descriptors,
                    })
                    all_signatures.append(signature)
                except _FreezeBudgetExceeded:
                    raise
                except (cv2.error, ValueError, TypeError, IndexError, FloatingPointError,
                        OverflowError, MemoryError):
                    signature_abstentions.append({
                        "reading_index": reading_index,
                        "alternative_index": alternative_index,
                        "reason": "signature_unmeasurable",
                    })

        stats = {str(key): _json_value(value) for key, value in
                 sorted(proposal_result.stats.items()) if key != "elapsed_ms"}
        diagnostics = []
        for item in reading_diagnostics.get("abstentions", []):
            if hasattr(item, "__dataclass_fields__"):
                diagnostics.append(_json_value(asdict(item)))
            else:
                diagnostics.append(_json_value(item))
        frame_records.append({
            "frame_id": frame_id,
            "input": loaded_frame["input"],
            "image_reason": loaded_frame["image_reason"],
            "calibration_reason": calibration_reason,
            "calibration": _calibration_record(calibration),
            "proposal": {"status": str(proposal_result.status),
                         "reason": str(proposal_result.reason),
                         "stats": stats, "proposals": proposal_records},
            "reading_diagnostics": {
                "segments": int(reading_diagnostics.get("segments", 0)),
                "low_structural_share": int(
                    reading_diagnostics.get("low_structural_share", 0)),
            },
            "reading_abstentions": diagnostics,
            "readings": reading_records,
            "signature_abstentions": signature_abstentions,
            "signatures": signature_records,
        })

    refusals: list[plane_graph.PlaneMatchRefusal] = []
    matching_reason = None
    try:
        matches = plane_graph.match_plane_graph(
            all_readings, all_signatures, max_shift_cells=32,
            max_structural_share=None, refusals=refusals)
    except (ValueError, TypeError, IndexError, FloatingPointError, OverflowError,
            MemoryError):
        matches = ()
        matching_reason = "matching_stage_failed"
    try:
        solution = plane_graph.solve_component(
            all_readings, matches, integer_tolerance=0.15,
            min_resultant=0.90, ambiguous_margin=0.02)
        solution_record = _solution_record(solution)
    except (ValueError, TypeError, IndexError, FloatingPointError, OverflowError,
            MemoryError):
        solution_record = {
            "frames": sorted({str(reading.frame) for reading in all_readings}),
            "quadrants": {}, "translations": {}, "axis_residuals": {},
            "axis_resultants": {}, "scale_known": False,
            "blocks_per_period": [], "reason": "solver_stage_failed",
            "alternatives": [],
        }
    graph_record = {
        "matching_reason": matching_reason,
        "matches": [_match_record(match) for match in matches],
        "refusals": [_refusal_record(refusal) for refusal in refusals],
        "solution": solution_record,
    }
    raw_sidecar = bytes(sidecar.data)
    sidecar_digest = _sha256(raw_sidecar)
    sidecar_name = f"{destination.name}.{sidecar_digest}.masks.bin"
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "",
        "labels_present": False,
        "configuration": _configuration(),
        "code_hashes": _code_hashes(),
        "inputs": [item["input"] for item in loaded],
        "frames": frame_records,
        "graph": graph_record,
        "stage_reasons": _stage_counts(frame_records, graph_record),
        "mask_sidecar": {"file": sidecar_name,
                          "sha256": sidecar_digest,
                          "size_bytes": len(raw_sidecar)},
    }
    document["artifact_id"] = _artifact_id(document)
    _validate_keys(document)
    json_payload = _canonical_json(document, max_bytes=MAX_FREEZE_JSON_BYTES - 1) + b"\n"
    sidecar_path = destination.with_name(sidecar_name)
    _atomic_write(sidecar_path, raw_sidecar)
    _atomic_write(destination, json_payload)
    return document


_TOP_KEYS = {"schema_version", "artifact_id", "labels_present", "configuration",
             "code_hashes", "inputs", "frames", "graph", "stage_reasons",
             "mask_sidecar"}
_INPUT_KEYS = {"basename", "width", "height", "sha256"}
_FRAME_KEYS = {"frame_id", "input", "image_reason", "calibration_reason", "calibration",
               "proposal", "reading_diagnostics", "reading_abstentions", "readings",
               "signature_abstentions", "signatures"}
_CALIBRATION_KEYS = {"focal_pixels", "yaw_degrees", "pitch_degrees", "roll_degrees",
                     "principal_point", "width", "height", "inliers", "segments",
                     "median_residual", "per_axis_inliers", "notes"}
_PROPOSAL_KEYS = {"status", "reason", "stats", "proposals"}
_PROPOSAL_ITEM_KEYS = {"axis", "side", "polygon", "strength", "sources", "coverage", "mask"}
_READING_KEYS = {"frame", "axis", "side", "alternatives", "period_pixels", "phase_pixels",
                 "extent", "samples", "polygon", "coverage_cells", "structural_share",
                 "confidence", "rectification"}
_RECTIFICATION_KEYS = {"axis_matrix", "focal", "principal_point", "centre"}
_ALTERNATIVE_KEYS = {"blocks_per_period", "distance_blocks", "support"}
_DIAGNOSTIC_KEYS = {"frame", "axis", "side", "reason", "detail"}
_DIAGNOSTICS_KEYS = {"segments", "low_structural_share"}
_SIGNATURE_KEYS = {"reading_index", "alternative_index", "blocks_per_period", "origin_uv", "arrays"}
_ARRAYS_KEYS = {"cells", "interior", "coverage"}
_ARRAY_REF_KEYS = {"offset", "length", "shape", "dtype", "encoding", "sha256"}
_GRAPH_KEYS = {"matching_reason", "matches", "refusals", "solution"}
_MATCH_KEYS = {"left", "right", "left_axis", "right_axis", "left_b", "right_b",
               "quadrant_of_right", "psr", "peak", "second_peak", "overlap_cells",
               "integer_residual", "score", "left_plane", "right_plane"}
_REFUSAL_KEYS = {"left", "right", "left_axis", "right_axis", "quadrant_of_right",
                 "reason", "detail"}
_SOLUTION_KEYS = {"frames", "quadrants", "translations", "axis_residuals", "axis_resultants",
                  "scale_known", "blocks_per_period", "reason", "alternatives"}
_SOLUTION_BLOCK_KEYS = {"frame", "axis", "value"}
_SOLUTION_ALT_KEYS = {"quadrants", "cost", "resultants", "axis_observations"}
_SIDE_CAR_KEYS = {"file", "sha256", "size_bytes"}
_STAGE_NAMES = {"image", "calibration", "proposals", "readings", "signatures", "matching", "solution"}
_LEGACY_CODE_HASH_KEYS = frozenset({
    "runner", "calibration", "region_growth", "lattice_planes", "plane_graph", "gui_mask",
})
_CODE_HASH_KEYS = _LEGACY_CODE_HASH_KEYS | {"lattice", "supported"}
_CONFIG_KEYS = {
    "input": {"max_frames", "max_pixels_per_frame", "order"},
    "gui_mask": set(asdict(GuiMaskConfig())),
    "calibration": {"grouping", "minimum_length", "threshold", "scales", "fov_range",
                    "coarse_segments", "refine_optical_centre", "weight_by_contrast",
                    "excluded_pixel_fill"},
    "proposals": {"config", "max_candidates"},
    "readings": {"max_readings", "min_structural_share", "max_rectified_side"},
    "signatures": {"max_cells"},
    "matching": {"max_shift_cells", "max_structural_share"},
    "solver": {"integer_tolerance", "min_resultant", "ambiguous_margin"},
    "runtime": {"opencv", "numpy"},
}


def _expect_keys(value: Any, expected: set[str], where: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise FrozenImageRunError(f"invalid keys at {where}")


def _validate_keys(document: dict[str, Any]) -> None:
    if set(document) != _TOP_KEYS:
        raise FrozenImageRunError("unknown or missing top-level freeze key")

    def forbidden(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).casefold() in _FORBIDDEN_KEYS:
                    raise FrozenImageRunError(f"forbidden key: {key}")
                forbidden(child)
        elif isinstance(value, list):
            for child in value:
                forbidden(child)

    forbidden(document)
    _expect_keys(document["configuration"], set(_CONFIG_KEYS), "configuration")
    for name, expected in _CONFIG_KEYS.items():
        _expect_keys(document["configuration"][name], expected, f"configuration.{name}")
    _expect_keys(document["configuration"]["proposals"]["config"],
                 set(asdict(region_growth.DEFAULT_REGION_GROWTH_CONFIG)),
                 "configuration.proposals.config")
    code_hash_keys = (set(document["code_hashes"])
                      if isinstance(document["code_hashes"], dict) else set())
    if code_hash_keys not in (_LEGACY_CODE_HASH_KEYS, _CODE_HASH_KEYS):
        raise FrozenImageRunError("code_hashes has unsupported provenance keys")
    _expect_keys(document["code_hashes"], code_hash_keys, "code_hashes")
    _expect_keys(document["mask_sidecar"], _SIDE_CAR_KEYS, "mask_sidecar")
    if not isinstance(document["inputs"], list) or not isinstance(document["frames"], list):
        raise FrozenImageRunError("inputs and frames must be arrays")
    for index, item in enumerate(document["inputs"]):
        _expect_keys(item, _INPUT_KEYS, f"inputs[{index}]")
    for index, frame in enumerate(document["frames"]):
        where = f"frames[{index}]"
        _expect_keys(frame, _FRAME_KEYS, where)
        _expect_keys(frame["input"], _INPUT_KEYS, where + ".input")
        if frame["calibration"] is not None:
            _expect_keys(frame["calibration"], _CALIBRATION_KEYS, where + ".calibration")
        proposal = frame["proposal"]
        _expect_keys(proposal, _PROPOSAL_KEYS, where + ".proposal")
        for j, candidate in enumerate(proposal["proposals"]):
            _expect_keys(candidate, _PROPOSAL_ITEM_KEYS, f"{where}.proposal.proposals[{j}]")
            _expect_keys(candidate["mask"], _ARRAY_REF_KEYS, f"{where}.proposal.mask")
        _expect_keys(frame["reading_diagnostics"], _DIAGNOSTICS_KEYS, where + ".reading_diagnostics")
        for j, diagnostic in enumerate(frame["reading_abstentions"]):
            _expect_keys(diagnostic, _DIAGNOSTIC_KEYS, f"{where}.reading_abstentions[{j}]")
        for j, reading in enumerate(frame["readings"]):
            _expect_keys(reading, _READING_KEYS, f"{where}.readings[{j}]")
            _expect_keys(reading["rectification"], _RECTIFICATION_KEYS,
                         f"{where}.readings[{j}].rectification")
            for k, alternative in enumerate(reading["alternatives"]):
                _expect_keys(alternative, _ALTERNATIVE_KEYS,
                             f"{where}.readings[{j}].alternatives[{k}]")
        for j, abstention in enumerate(frame["signature_abstentions"]):
            _expect_keys(abstention, {"reading_index", "alternative_index", "reason"},
                         f"{where}.signature_abstentions[{j}]")
        for j, signature in enumerate(frame["signatures"]):
            _expect_keys(signature, _SIGNATURE_KEYS, f"{where}.signatures[{j}]")
            _expect_keys(signature["arrays"], _ARRAYS_KEYS, f"{where}.signatures[{j}].arrays")
            for name, descriptor in signature["arrays"].items():
                _expect_keys(descriptor, _ARRAY_REF_KEYS,
                             f"{where}.signatures[{j}].arrays.{name}")
    _expect_keys(document["graph"], _GRAPH_KEYS, "graph")
    for index, match in enumerate(document["graph"]["matches"]):
        _expect_keys(match, _MATCH_KEYS, f"graph.matches[{index}]")
    for index, refusal in enumerate(document["graph"]["refusals"]):
        _expect_keys(refusal, _REFUSAL_KEYS, f"graph.refusals[{index}]")
    solution = document["graph"]["solution"]
    _expect_keys(solution, _SOLUTION_KEYS, "graph.solution")
    for index, item in enumerate(solution["blocks_per_period"]):
        _expect_keys(item, _SOLUTION_BLOCK_KEYS, f"graph.solution.blocks_per_period[{index}]")
    for index, item in enumerate(solution["alternatives"]):
        _expect_keys(item, _SOLUTION_ALT_KEYS, f"graph.solution.alternatives[{index}]")
    _expect_keys(document["stage_reasons"], _STAGE_NAMES, "stage_reasons")
    for stage, counts in document["stage_reasons"].items():
        if not isinstance(counts, dict) or any(not isinstance(v, int) or v < 0
                                               for v in counts.values()):
            raise FrozenImageRunError(f"invalid stage counts for {stage}")


def _array_descriptors(document: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors = []
    for frame in document["frames"]:
        for proposal in frame["proposal"]["proposals"]:
            descriptors.append(proposal["mask"])
        for signature in frame["signatures"]:
            descriptors.extend(signature["arrays"].values())
    return descriptors


def _strict_json(payload: bytes) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FrozenImageRunError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise FrozenImageRunError(f"non-finite JSON constant is forbidden: {value}")

    try:
        parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_pairs,
                            parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenImageRunError("freeze is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise FrozenImageRunError("freeze root must be a JSON object")
    return parsed


def load_and_verify_frozen_image(path: str | Path) -> dict[str, Any]:
    """Strictly load a schema-v1 freeze and verify its ids and binary payloads."""
    try:
        return _load_and_verify_frozen_image(path)
    except FrozenImageRunError:
        raise
    except (TypeError, ValueError, KeyError, IndexError, OverflowError,
            MemoryError) as exc:
        raise FrozenImageRunError("freeze contains invalid schema values") from exc


def _load_and_verify_frozen_image(path: str | Path) -> dict[str, Any]:
    freeze_path = Path(path)
    try:
        if freeze_path.stat().st_size > MAX_FREEZE_JSON_BYTES:
            raise FrozenImageRunError("freeze JSON exceeds the size cap")
        document = _strict_json(freeze_path.read_bytes())
    except OSError as exc:
        raise FrozenImageRunError("freeze JSON could not be read") from exc
    _validate_keys(document)
    if document["schema_version"] != SCHEMA_VERSION or document["labels_present"] is not False:
        raise FrozenImageRunError("unsupported schema or labels_present is not false")
    if not _is_sha256(document["artifact_id"]) or document["artifact_id"] != _artifact_id(document):
        raise FrozenImageRunError("artifact id does not match canonical content")
    sidecar_info = document["mask_sidecar"]
    if not _is_sha256(sidecar_info["sha256"]):
        raise FrozenImageRunError("mask sidecar digest is invalid")
    expected_name = f"{freeze_path.name}.{sidecar_info['sha256']}.masks.bin"
    if (sidecar_info["file"] != expected_name
            or Path(sidecar_info["file"]).name != sidecar_info["file"]):
        raise FrozenImageRunError("mask sidecar path is invalid")
    if (not isinstance(sidecar_info["size_bytes"], int)
            or isinstance(sidecar_info["size_bytes"], bool)
            or sidecar_info["size_bytes"] < 0
            or sidecar_info["size_bytes"] > MAX_SIDECAR_BYTES):
        raise FrozenImageRunError("mask sidecar size is invalid")
    try:
        sidecar_path = freeze_path.with_name(expected_name)
        if sidecar_path.stat().st_size != sidecar_info["size_bytes"]:
            raise FrozenImageRunError("mask sidecar size differs from declared size")
        sidecar = sidecar_path.read_bytes()
    except OSError as exc:
        raise FrozenImageRunError("mask sidecar could not be read") from exc
    if len(sidecar) != sidecar_info["size_bytes"] or _sha256(sidecar) != sidecar_info["sha256"]:
        raise FrozenImageRunError("mask sidecar digest or length mismatch")
    descriptors = _array_descriptors(document)
    intervals = []
    for descriptor in descriptors:
        offset, length = descriptor["offset"], descriptor["length"]
        if (not isinstance(offset, int) or isinstance(offset, bool)
                or not isinstance(length, int) or isinstance(length, bool) or offset < 0
                or length < 0 or offset + length > len(sidecar)):
            raise FrozenImageRunError("array descriptor is outside sidecar bounds")
        raw = sidecar[offset:offset + length]
        if not _is_sha256(descriptor["sha256"]) or _sha256(raw) != descriptor["sha256"]:
            raise FrozenImageRunError("array payload digest mismatch")
        shape = descriptor["shape"]
        if (not isinstance(shape, list) or len(shape) != 2
                or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                       for dim in shape)):
            raise FrozenImageRunError("array descriptor shape is invalid")
        if descriptor["dtype"] == "bool" and descriptor["encoding"] == "packbits-little":
            area = math.prod(shape)
            if area > MAX_INPUT_PIXELS:
                raise FrozenImageRunError("boolean mask descriptor exceeds pixel cap")
            expected_length = (area + 7) // 8
        elif descriptor["dtype"] == "float32-le" and descriptor["encoding"] == "raw":
            if any(dim > MAX_SIGNATURE_CELLS for dim in shape):
                raise FrozenImageRunError("signature descriptor exceeds cell cap")
            expected_length = math.prod(shape) * 4
        else:
            raise FrozenImageRunError("array descriptor dtype or encoding is unsupported")
        if length != expected_length:
            raise FrozenImageRunError("array descriptor byte length disagrees with shape")
        intervals.append((offset, offset + length))
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start != cursor:
            raise FrozenImageRunError("sidecar ranges overlap or leave unreferenced bytes")
        cursor = end
    if cursor != len(sidecar):
        raise FrozenImageRunError("sidecar contains unreferenced bytes")
    for frame in document["frames"]:
        if frame["input"] not in document["inputs"]:
            raise FrozenImageRunError("frame input does not match input inventory")
        if frame["frame_id"] != frame["input"]["basename"]:
            raise FrozenImageRunError("frame id does not match input basename")
    return document
