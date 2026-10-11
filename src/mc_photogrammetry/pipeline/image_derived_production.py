"""Normal-mode production wiring from ordinary images to image-derived output."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mc_photogrammetry.domain.models import (
    SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
    BlockVolume,
    ScanProject,
)
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as geometry_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
from mc_photogrammetry.pipeline import image_derived_output as output_mod
from mc_photogrammetry.pipeline import image_derived_production_record as record_mod
from mc_photogrammetry.project.store import (
    ProjectStoreError,
    publish_project_no_clobber,
)
from mc_photogrammetry.reconstruction import plane_placement

__all__ = ["IMAGE_DERIVED_PRODUCTION_SCHEMA",
           "IMAGE_DERIVED_PRODUCTION_PRODUCER", "ImageDerivedProductionResult",
           "derive_image_output_from_paths"]

IMAGE_DERIVED_PRODUCTION_SCHEMA = "mc-photogrammetry-image-derived-production"
IMAGE_DERIVED_PRODUCTION_PRODUCER = "r3-w-image-derived-production-1"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _replay(value: object, expected: type):
    if not isinstance(value, expected):
        raise ValueError("production stage has a foreign type")
    return expected(**{field.name: getattr(value, field.name)
                       for field in dataclasses.fields(expected)})


def _payload(value: "ImageDerivedProductionResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "stage": value.stage, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "entry_sha256": value.entry_sha256,
        "lattice_sha256": value.lattice_sha256,
        "geometry_sha256": value.geometry_sha256,
        "output_sha256": value.output_sha256,
    }


@dataclass(frozen=True, slots=True)
class ImageDerivedProductionResult:
    schema: str
    producer: str
    stage: str
    status: str
    reason: str
    detail: str
    entry_sha256: str
    lattice_sha256: str
    geometry_sha256: str
    output_sha256: str
    entry: entry_mod.ImageDerivedEntryResult
    lattice: lattice_mod.ImageDerivedLatticeResult | None
    geometry: geometry_mod.ImageDerivedGeometryResult | None
    output: output_mod.ImageDerivedOutputResult | None
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_PRODUCTION_SCHEMA
                or self.producer != IMAGE_DERIVED_PRODUCTION_PRODUCER
                or self.stage not in ("entry", "lattice", "geometry", "output")
                or self.status not in (
                    "geometry_unavailable", "promoted", "review", "refused",
                    "budget_exhausted", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not _hex(self.entry_sha256)):
            raise ValueError("image-derived production result is malformed")
        entry = _replay(self.entry, entry_mod.ImageDerivedEntryResult)
        lattice = (None if self.lattice is None else
                   _replay(self.lattice, lattice_mod.ImageDerivedLatticeResult))
        geometry = (None if self.geometry is None else
                    _replay(self.geometry,
                            geometry_mod.ImageDerivedGeometryResult))
        output = (None if self.output is None else
                  _replay(self.output, output_mod.ImageDerivedOutputResult))
        if self.entry_sha256 != entry.entry_sha256:
            raise ValueError("production entry digest does not bind")
        expected_lattice = "" if lattice is None else lattice.result_sha256
        expected_geometry = "" if geometry is None else geometry.result_sha256
        expected_output = "" if output is None else output.result_sha256
        if (self.lattice_sha256 != expected_lattice
                or self.geometry_sha256 != expected_geometry
                or self.output_sha256 != expected_output):
            raise ValueError("production stage digests do not bind")
        if (self.stage == "entry" and any(
                item is not None for item in (lattice, geometry, output))
                or self.stage == "lattice" and (
                    lattice is None or geometry is not None or output is not None)
                or self.stage == "geometry" and (
                    lattice is None or geometry is None or output is not None)
                or self.stage == "output" and (
                    lattice is None or geometry is None or output is None)):
            raise ValueError("production stages are not a canonical prefix")
        if lattice is not None and lattice.session_sha256 != entry.session.session_sha256:
            raise ValueError("lattice does not bind the live entry session")
        if geometry is not None and geometry.lattice_sha256 != lattice.result_sha256:
            raise ValueError("geometry does not bind the lattice")
        if output is not None and output.geometry_sha256 != geometry.result_sha256:
            raise ValueError("output does not bind geometry")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("production result digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _make(entry, *, stage, status, reason="", detail="", lattice=None,
          geometry=None, output=None):
    return ImageDerivedProductionResult(
        IMAGE_DERIVED_PRODUCTION_SCHEMA, IMAGE_DERIVED_PRODUCTION_PRODUCER,
        stage, status, reason, detail, entry.entry_sha256,
        "" if lattice is None else lattice.result_sha256,
        "" if geometry is None else geometry.result_sha256,
        "" if output is None else output.result_sha256,
        entry, lattice, geometry, output)


def _project_name(target: Path, entry) -> str:
    if target.stem:
        return target.stem
    return Path(entry.source_paths[0]).stem


def _bind_entry_to_target(entry, target: Path):
    """Rebind one immutable entry identity to the requested final target.

    ``build_image_derived_entry`` persists its schema-2 checkpoint to a private
    temporary path and seals ``project_path``/``entry_sha256`` to that path.
    The durable schema-3 terminal must instead bind the requested final target,
    so the entry is rebuilt at this boundary through its own validating
    constructor, which replays every nested record and recomputes the digest
    without editing the frozen entry module.
    """
    if str(target) == entry.project_path:
        return entry
    return entry_mod.ImageDerivedEntryResult(
        schema=entry.schema, producer=entry.producer, terminal=entry.terminal,
        reason=entry.reason, detail=entry.detail, project_path=str(target),
        source_paths=entry.source_paths,
        connection_budgets=entry.connection_budgets,
        pose_budgets=entry.pose_budgets, session=entry.session,
        graph=entry.graph, pose_graph=entry.pose_graph)


def _result_envelope(entry, outcome):
    promotion = None
    volume = BlockVolume()
    if outcome.status == "promoted":
        promotion = outcome.output.promotion
        volume = promotion.to_block_volume()
    envelope = record_mod.build_production_envelope(
        stage=outcome.stage, status=outcome.status, reason=outcome.reason,
        detail=outcome.detail, entry_sha256=outcome.entry_sha256,
        lattice_sha256=outcome.lattice_sha256,
        geometry_sha256=outcome.geometry_sha256,
        output_sha256=outcome.output_sha256,
        session_sha256=entry.session.session_sha256,
        promotion=promotion, volume=volume,
        joint_lineage=() if outcome.geometry is None else outcome.geometry.joint_lineage)
    return envelope, volume


def _refusal_envelope(entry, *, stage, refusal, lattice_sha256="",
                      geometry_sha256="", output_sha256="", joint_lineage=()):
    return record_mod.build_production_envelope(
        stage=stage, status="refused", reason=refusal.reason,
        detail=record_mod.refusal_detail(refusal),
        entry_sha256=entry.entry_sha256, lattice_sha256=lattice_sha256,
        geometry_sha256=geometry_sha256, output_sha256=output_sha256,
        session_sha256=entry.session.session_sha256, promotion=None,
        refusal_sha256=refusal.refusal_sha256, volume=BlockVolume(), joint_lineage=joint_lineage)


def _publish(target: Path, entry, envelope, volume, outcome):
    """Publish the schema-3 terminal project without clobbering an existing file."""
    project = ScanProject(
        name=_project_name(target, entry),
        source_paths=list(entry.source_paths),
        volume=volume,
        schema_version=SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
        image_derived_session=entry.session_record,
        image_derived_production=envelope.to_record())
    try:
        publish_project_no_clobber(project, target)
    except ProjectStoreError:
        return record_mod.storage_refusal(target)
    return outcome


def derive_image_output_from_paths(
    image_inputs: Iterable[str | Path], *, project_path: str | Path,
    recursive: bool = False,
    max_frames: int = entry_mod.DEFAULT_MAX_FRAMES,
    max_pairs: int = entry_mod.DEFAULT_MAX_PAIRS,
    max_total_support: int = entry_mod.DEFAULT_MAX_TOTAL_SUPPORT,
    max_edges: int = entry_mod.DEFAULT_MAX_EDGES,
    max_cycle_checks: int = entry_mod.DEFAULT_MAX_CYCLE_CHECKS,
    rotation_tolerance_degrees: float =
        entry_mod.DEFAULT_ROTATION_TOLERANCE_DEGREES,
    direction_tolerance: float = entry_mod.DEFAULT_DIRECTION_TOLERANCE,
) -> ImageDerivedProductionResult | plane_placement.PlacementRefusal:
    """Run the accepted image-derived stages and persist one schema-3 terminal.

    A pre-existing project path is refused *before* any image is discovered or
    decoded, so a prior successful project is never replaced.  The schema-2 entry
    checkpoint is staged in a private temporary directory, and the schema-3
    terminal project is published to the requested path with atomic no-clobber
    semantics: a target race, a refusal, an exception or a save failure leaves the
    predecessor bytes exactly as they were.
    """
    target = Path(project_path)
    preexisting = record_mod.target_refusal(target)
    if preexisting is not None:
        return preexisting
    try:
        checkpoint = record_mod.checkpoint_path(target)
    except OSError:
        return record_mod.storage_refusal(target)
    try:
        entry = entry_mod.build_image_derived_entry(
            image_inputs, project_path=checkpoint, recursive=recursive,
            max_frames=max_frames, max_pairs=max_pairs,
            max_total_support=max_total_support, max_edges=max_edges,
            max_cycle_checks=max_cycle_checks,
            rotation_tolerance_degrees=rotation_tolerance_degrees,
            direction_tolerance=direction_tolerance)
        entry = _bind_entry_to_target(entry, target)
        if (len(entry.session.frames) < 2
                or entry.graph.status != "ok" or entry.pose_graph.status != "ok"):
            outcome = _make(entry, stage="entry", status="geometry_unavailable",
                            reason=entry.reason or ("single_camera_local_review" if len(entry.session.frames) == 1 else "entry_unavailable"),
                            detail=entry.detail)
            return _publish(target, entry, *_result_envelope(entry, outcome), outcome)
        lattice = lattice_mod.measure_image_derived_lattice(
            session=entry.session, pose_graph=entry.pose_graph)
        if isinstance(lattice, plane_placement.PlacementRefusal):
            return _publish(
                target, entry,
                _refusal_envelope(entry, stage="lattice", refusal=lattice),
                BlockVolume(), lattice)
        lattice_sha256 = getattr(lattice, "result_sha256", "")
        if lattice.status != "assembled":
            outcome = _make(entry, stage="lattice", status="geometry_unavailable",
                            reason="lattice_" + lattice.reason,
                            detail=lattice.detail, lattice=lattice)
            return _publish(target, entry, *_result_envelope(entry, outcome), outcome)
        completed_geometry = []
        geometry = geometry_mod._derive_image_derived_geometry_with_context(
            session=entry.session, lattice_result=lattice,
            connection_graph=entry.graph,
            _completed_stage_sink=lambda stage,records,refusal:completed_geometry.append((stage,records,refusal)))
        if isinstance(geometry, plane_placement.PlacementRefusal):
            return _publish(
                target, entry,
                _refusal_envelope(entry, stage="geometry", refusal=geometry,
                                  lattice_sha256=lattice_sha256,
                                  joint_lineage=geometry_mod._joint_lineage(completed_geometry[0][1])
                                    if completed_geometry else ()),
                BlockVolume(), geometry)
        geometry_sha256 = getattr(geometry, "result_sha256", "")
        if geometry.status != "assembled":
            outcome = _make(entry, stage="geometry", status="geometry_unavailable",
                            reason=geometry.reason, detail=geometry.detail,
                            lattice=lattice, geometry=geometry)
            return _publish(target, entry, *_result_envelope(entry, outcome), outcome)
        output = output_mod.derive_image_derived_output(
            geometry_result=geometry, session=entry.session)
        if isinstance(output, plane_placement.PlacementRefusal):
            return _publish(
                target, entry,
                _refusal_envelope(entry, stage="output", refusal=output,
                                  lattice_sha256=lattice_sha256,
                                  geometry_sha256=geometry_sha256,
                                  joint_lineage=geometry.joint_lineage),
                BlockVolume(), output)
        status = ("geometry_unavailable" if output.status == "unavailable"
                  else output.status)
        outcome = _make(entry, stage="output", status=status, reason=output.reason,
                        detail=output.detail, lattice=lattice, geometry=geometry,
                        output=output)
        return _publish(target, entry, *_result_envelope(entry, outcome), outcome)
    finally:
        record_mod.cleanup_checkpoint(checkpoint)
