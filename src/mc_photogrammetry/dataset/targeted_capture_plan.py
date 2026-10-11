"""Evaluation-only, explicit targeted capture plans.

This module deliberately does not integrate with the batch spec or executor.
Its only production-facing projection is an ordered tuple of PNG paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from .hashing import digest_json
from .plan import validate_plan
from .schemas import CameraCapture, CapturePlan, DatasetSchemaError, RegionBounds

SCHEMA = "mc-photogrammetry-targeted-axis0-capture-plan-v1"
PRODUCER = "targeted-axis0-capture-plan-v1"
ROLES = ("axis0-primary", "axis0-overlap-a", "axis0-overlap-b", "context")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or value != value.lower() or any(c not in "0123456789abcdef" for c in value):
        raise DatasetSchemaError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise DatasetSchemaError(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class TargetQuality:
    min_pair_support_cells: int = 64
    min_pair_support_fraction: float = 0.5
    max_predicted_displacement_cells: float = 24.0
    min_structural_share: float = 0.15
    min_psr: float = 3.0

    def __post_init__(self) -> None:
        if isinstance(self.min_pair_support_cells, bool) or self.min_pair_support_cells != 64:
            raise DatasetSchemaError("min_pair_support_cells must be 64")
        expected = (0.5, 24.0, 0.15, 3.0)
        actual = tuple(_number(v, "quality target") for v in (
            self.min_pair_support_fraction, self.max_predicted_displacement_cells,
            self.min_structural_share, self.min_psr))
        if actual != expected:
            raise DatasetSchemaError("targeted capture quality thresholds are frozen")

    def to_record(self) -> dict[str, Any]:
        return {
            "min_pair_support_cells": self.min_pair_support_cells,
            "min_pair_support_fraction": self.min_pair_support_fraction,
            "max_predicted_displacement_cells": self.max_predicted_displacement_cells,
            "min_structural_share": self.min_structural_share,
            "min_psr": self.min_psr,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "TargetQuality":
        if not isinstance(value, Mapping) or set(value) != set(cls().to_record()):
            raise DatasetSchemaError("quality must contain exactly the v1 fields")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TargetedJob:
    role: str
    surface_id: str | None
    capture: CameraCapture
    predicted_support_cells: int
    predicted_support_fraction: float
    predicted_displacement_cells: float
    predicted_structural_share: float
    predicted_psr: float

    def to_record(self) -> dict[str, Any]:
        return {"role": self.role, "surface_id": self.surface_id,
                "capture": self.capture.to_dict(),
                "predicted_support_cells": self.predicted_support_cells,
                "predicted_support_fraction": self.predicted_support_fraction,
                "predicted_displacement_cells": self.predicted_displacement_cells,
                "predicted_structural_share": self.predicted_structural_share,
                "predicted_psr": self.predicted_psr}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "TargetedJob":
        keys = {"role", "surface_id", "capture", "predicted_support_cells",
                "predicted_support_fraction", "predicted_displacement_cells",
                "predicted_structural_share", "predicted_psr"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise DatasetSchemaError("targeted job must contain exactly the v1 fields")
        surface = value["surface_id"]
        if surface is not None and not isinstance(surface, str):
            raise DatasetSchemaError("surface_id must be a string or null")
        return cls(str(value["role"]), surface,
                   CameraCapture.from_dict(value["capture"]),
                   value["predicted_support_cells"],
                   value["predicted_support_fraction"],
                   value["predicted_displacement_cells"],
                   value["predicted_structural_share"], value["predicted_psr"])


@dataclass(frozen=True, slots=True)
class TargetedCapturePlan:
    schema: str
    producer: str
    dataset_id: str
    scene_id: str
    world_identity: str
    region: RegionBounds
    jobs: tuple[TargetedJob, ...]
    quality: TargetQuality
    source_sha256: str
    worker_template_sha256: str
    capture_mod_sha256: str
    predecessor_manifest_sha256: str
    predecessor_capture_sha256: str
    plan_sha256: str = ""

    def __post_init__(self) -> None:
        validate_targeted_capture_plan(self)
        computed = digest_json(self.to_record(include_digest=False))
        if self.plan_sha256 not in ("", computed):
            raise DatasetSchemaError("targeted capture plan digest mismatch")
        object.__setattr__(self, "plan_sha256", computed)

    def to_record(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {"schema": self.schema, "producer": self.producer,
                  "dataset_id": self.dataset_id, "scene_id": self.scene_id,
                  "world_identity": self.world_identity, "region": self.region.to_dict(),
                  "jobs": [job.to_record() for job in self.jobs],
                  "quality": self.quality.to_record(),
                  "source_sha256": self.source_sha256,
                  "worker_template_sha256": self.worker_template_sha256,
                  "capture_mod_sha256": self.capture_mod_sha256,
                  "predecessor_manifest_sha256": self.predecessor_manifest_sha256,
                  "predecessor_capture_sha256": self.predecessor_capture_sha256}
        if include_digest:
            result["plan_sha256"] = self.plan_sha256
        return result

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "TargetedCapturePlan":
        keys = {"schema", "producer", "dataset_id", "scene_id", "world_identity",
                "region", "jobs", "quality", "source_sha256",
                "worker_template_sha256", "capture_mod_sha256",
                "predecessor_manifest_sha256", "predecessor_capture_sha256",
                "plan_sha256"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise DatasetSchemaError("targeted plan must contain exactly the v1 fields")
        jobs = value["jobs"]
        if not isinstance(jobs, (list, tuple)):
            raise DatasetSchemaError("jobs must be a list")
        return cls(str(value["schema"]), str(value["producer"]),
                   str(value["dataset_id"]), str(value["scene_id"]),
                   str(value["world_identity"]), RegionBounds.from_dict(value["region"]),
                   tuple(TargetedJob.from_record(v) for v in jobs),
                   TargetQuality.from_record(value["quality"]),
                   str(value["source_sha256"]), str(value["worker_template_sha256"]),
                   str(value["capture_mod_sha256"]),
                   str(value["predecessor_manifest_sha256"]),
                   str(value["predecessor_capture_sha256"]), str(value["plan_sha256"]))


def validate_targeted_capture_plan(plan: TargetedCapturePlan) -> None:
    if plan.schema != SCHEMA or plan.producer != PRODUCER:
        raise DatasetSchemaError("unsupported targeted capture schema or producer")
    if not plan.dataset_id or not plan.scene_id or not plan.world_identity:
        raise DatasetSchemaError("dataset, scene, and world identities are required")
    for name in ("source_sha256", "worker_template_sha256", "capture_mod_sha256",
                 "predecessor_manifest_sha256", "predecessor_capture_sha256"):
        _digest(getattr(plan, name), name)
    if len(plan.jobs) != 4 or tuple(job.role for job in plan.jobs) != ROLES:
        raise DatasetSchemaError("plan requires three ordered axis-0 jobs and one context job")
    surface = plan.jobs[0].surface_id
    if not surface or any(job.surface_id != surface for job in plan.jobs[:3]):
        raise DatasetSchemaError("the three axis-0 jobs must bind one distinctive surface")
    if plan.jobs[3].surface_id is not None:
        raise DatasetSchemaError("the context job must not claim the target surface")
    quality = plan.quality
    for job in plan.jobs[:3]:
        if isinstance(job.predicted_support_cells, bool) \
                or job.predicted_support_cells < quality.min_pair_support_cells:
            raise DatasetSchemaError("axis-0 predicted support is below 64 cells")
        if _number(job.predicted_support_fraction, "support fraction") \
                < quality.min_pair_support_fraction:
            raise DatasetSchemaError("axis-0 predicted support fraction is below 0.5")
        if abs(_number(job.predicted_displacement_cells, "displacement")) \
                > quality.max_predicted_displacement_cells:
            raise DatasetSchemaError("axis-0 predicted displacement exceeds 24 cells")
        if _number(job.predicted_structural_share, "structural share") \
                < quality.min_structural_share:
            raise DatasetSchemaError("axis-0 structural share is below 0.15")
        if _number(job.predicted_psr, "PSR") < quality.min_psr:
            raise DatasetSchemaError("axis-0 predicted PSR is below 3")


def materialize_capture_plan(plan: TargetedCapturePlan) -> CapturePlan:
    """Convert the validated evaluation record into the existing mod plan type."""
    replay = TargetedCapturePlan.from_record(plan.to_record())
    capture = CapturePlan(
        plan_id=f"plan-{replay.plan_sha256[:16]}", scene_id=replay.scene_id,
        dataset_id=replay.dataset_id, world_identity=replay.world_identity,
        region=replay.region, jobs=tuple(item.capture for item in replay.jobs),
        staging_token=replay.source_sha256,
        notes=("evaluation-only targeted axis-0 capture; never production metadata",))
    validate_plan(capture, max_jobs=4)
    return capture


def png_only_handoff(plan: TargetedCapturePlan,
                     paths_by_frame: Mapping[str, str | Path]) -> tuple[str, ...]:
    """Return only ordered PNG paths; no capture metadata crosses this boundary."""
    replay = TargetedCapturePlan.from_record(plan.to_record())
    expected = tuple(job.capture.frame_id for job in replay.jobs)
    if set(paths_by_frame) != set(expected):
        raise DatasetSchemaError("PNG handoff must contain exactly the planned frames")
    result = tuple(str(paths_by_frame[frame]) for frame in expected)
    if any(Path(path).suffix.lower() != ".png" for path in result):
        raise DatasetSchemaError("production handoff accepts PNG paths only")
    return result
