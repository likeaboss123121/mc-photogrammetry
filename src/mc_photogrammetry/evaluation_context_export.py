"""Isolated contracts for the M4 evaluator-only context-GT export.

This module is intentionally outside the normal reconstruction and dataset
promotion flow. Its zero-job plan can export a larger staged-world region, and
its binder produces only F2A's separately labelled evaluation artifact.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from mc_photogrammetry.dataset.schemas import (
    CapturePlan,
    DatasetSchemaError,
    GroundTruthExport,
    RegionBounds,
)
from mc_photogrammetry.dataset.blocks import parse_ground_truth
from mc_photogrammetry.evaluation_frozen_planes import (
    FrozenPlaneEvaluationError,
    _CONTEXT_ARTIFACT_TYPE,
    _validate_context_ground_truth,
)


class ContextExportError(ValueError):
    """A context-only export or its provenance binding failed validation."""


def context_export_plan(
    *,
    dataset_id: str,
    scene_id: str,
    world_identity: str,
    region: RegionBounds,
    staging_token: str,
) -> CapturePlan:
    """Build an explicit zero-job plan for an evaluation-only GT export."""
    digest_input = json.dumps({
        "mode": "context_export_only",
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "world_identity": world_identity,
        "region": region.to_dict(),
        "staging_token": staging_token,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    plan_id = "context-" + hashlib.sha256(digest_input).hexdigest()[:24]
    try:
        return CapturePlan(
            plan_id=plan_id,
            scene_id=scene_id,
            dataset_id=dataset_id,
            world_identity=world_identity,
            region=region,
            jobs=(),
            staging_token=staging_token,
            notes=("evaluation-only context GT export; no screenshots or camera jobs",),
            context_export_only=True,
        )
    except DatasetSchemaError as error:
        raise ContextExportError(f"invalid context export plan: {error}") from error


def validate_context_export_status(
    status: Any,
    *,
    plan: CapturePlan,
) -> None:
    """Require the fresh mod sentinel to identify exactly this no-frame export."""
    if not isinstance(status, Mapping):
        raise ContextExportError("run_status must be an object")
    if (status.get("context_export_only") is not True
            or status.get("run_failed") is not False
            or status.get("all_frames_captured") is not False
            or status.get("ground_truth_exported") is not True
            or status.get("plan_id") != plan.plan_id
            or status.get("staging_token") != plan.staging_token
            or status.get("errors") != []):
        raise ContextExportError("run_status does not prove a successful context-only export")
    expected_context = {
        "scene_id": plan.scene_id,
        "world_identity": plan.world_identity,
        "region": plan.region.to_dict(),
        "camera_pose_applied": False,
        "result_count": 0,
    }
    if status.get("context_export") != expected_context:
        raise ContextExportError("run_status context identity/region/zero-pose contract differs")


def _context_truth(raw_export: Any) -> GroundTruthExport:
    try:
        return parse_ground_truth(raw_export)
    except (DatasetSchemaError, TypeError, ValueError, KeyError, IndexError) as error:
        raise ContextExportError(f"raw context ground truth is invalid: {error}") from error


def _require_complete_region(truth: GroundTruthExport, region: RegionBounds) -> None:
    if truth.region != region:
        raise ContextExportError("raw export region differs from the preregistered context bounds")
    volume = region.size[0] * region.size[1] * region.size[2]
    if len(truth.blocks) != volume:
        raise ContextExportError(
            f"raw context has {len(truth.blocks)} records for a {volume}-block region")
    unique_positions = {record.position for record in truth.blocks}
    if len(unique_positions) != volume:
        raise ContextExportError("raw context export contains missing or duplicate block positions")


def build_bound_context_document(
    raw_export: Any,
    *,
    capture_sha256: str,
    target_ground_truth_sha256: str,
    capture_document: Mapping[str, Any],
    ordered_images: Sequence[Mapping[str, Any]],
    target_ground_truth: GroundTruthExport,
    context_region: RegionBounds,
) -> dict[str, Any]:
    """Bind a complete raw mod export to F2A's immutable evaluation inputs."""
    context_truth = _context_truth(raw_export)
    _require_complete_region(context_truth, context_region)
    blocks = [
        {
            "pos": list(record.position),
            "name": record.name,
            "properties": dict(sorted(record.properties.items())),
        }
        for record in sorted(context_truth.blocks, key=lambda row: row.position)
    ]
    rows = []
    for row in ordered_images:
        if not isinstance(row, Mapping):
            raise ContextExportError("ordered image inventory contains a non-object")
        rows.append(dict(row))
    document = {
        "schema_version": 1,
        "artifact_type": _CONTEXT_ARTIFACT_TYPE,
        "scene_id": context_truth.scene_id,
        "world_identity": context_truth.world_identity,
        "capture_metadata_sha256": capture_sha256,
        "target_ground_truth_sha256": target_ground_truth_sha256,
        "ordered_images": rows,
        "target_region": target_ground_truth.region.to_dict(),
        "region": context_truth.region.to_dict(),
        "blocks": blocks,
    }
    try:
        _validate_context_ground_truth(
            document,
            capture_sha256=capture_sha256,
            target_sha256=target_ground_truth_sha256,
            capture_doc=capture_document,
            input_rows=rows,
            target_truth=target_ground_truth,
        )
    except FrozenPlaneEvaluationError as error:
        raise ContextExportError(f"bound context does not meet F2A: {error}") from error
    return document
