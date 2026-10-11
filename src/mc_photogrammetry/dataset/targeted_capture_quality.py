"""Evaluation-only quality gate for a completed targeted capture."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from mc_photogrammetry.pipeline import image_derived_graph_attribution as trace_mod
from mc_photogrammetry.pipeline import image_derived_production as production_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import plane_graph, plane_placement
from .schemas import DatasetSchemaError
from .targeted_capture_plan import TargetedCapturePlan

SCHEMA = "mc-photogrammetry-targeted-capture-quality-v1"
PRODUCER = "targeted-capture-quality-v1"


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PairMeasurement:
    left: str
    right: str
    axis: int
    left_plane: int
    right_plane: int
    quadrant: int
    psr: float | str
    score: float
    overlap_cells: int
    valid_cells: int
    overlap_fraction: float
    cell_shift_rc: tuple[int, int]
    displacement_cells: int
    left_structural_share: float
    right_structural_share: float
    support_cells_pass: bool
    support_fraction_pass: bool
    displacement_pass: bool
    structural_share_pass: bool
    psr_pass: bool


@dataclass(frozen=True, slots=True)
class AxisMeasurement:
    axis: int
    pair_count: int
    minimum_support_cells: int
    minimum_support_fraction: float
    maximum_displacement_cells: int
    minimum_structural_share: float
    minimum_psr: float | str


@dataclass(frozen=True, slots=True)
class TargetedCaptureQualityReport:
    schema: str
    producer: str
    status: str
    reason: str
    detail: str
    plan_sha256: str
    frame_ids: tuple[str, ...]
    production_result_sha256: str
    graph_result_sha256: str
    graph_trace_sha256: str
    solver_reason: str
    pairs: tuple[PairMeasurement, ...]
    axes: tuple[AxisMeasurement, ...]
    report_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != SCHEMA or self.producer != PRODUCER
                or self.status not in ("pass", "fail", "insufficient_evidence")):
            raise ValueError("targeted quality report is malformed")
        digest = _sha(self.to_record(include_digest=False))
        if self.report_sha256 not in ("", digest):
            raise ValueError("targeted quality report digest mismatch")
        object.__setattr__(self, "report_sha256", digest)

    def to_record(self, *, include_digest: bool = True) -> dict[str, Any]:
        value = {"schema": self.schema, "producer": self.producer,
                 "status": self.status, "reason": self.reason, "detail": self.detail,
                 "plan_sha256": self.plan_sha256, "frame_ids": list(self.frame_ids),
                 "production_result_sha256": self.production_result_sha256,
                 "graph_result_sha256": self.graph_result_sha256,
                 "graph_trace_sha256": self.graph_trace_sha256,
                 "solver_reason": self.solver_reason,
                 "pairs": [asdict(row) for row in self.pairs],
                 "axes": [asdict(row) for row in self.axes]}
        if include_digest:
            value["report_sha256"] = self.report_sha256
        return value


def _paths(plan: TargetedCapturePlan, handoff: object) -> tuple[Path, ...]:
    if isinstance(handoff, Mapping) or not isinstance(handoff, Sequence) \
            or isinstance(handoff, (str, bytes)):
        raise DatasetSchemaError("production handoff must be a PNG-path sequence only")
    paths = tuple(Path(value) for value in handoff)
    expected = tuple(job.capture.frame_id for job in plan.jobs)
    if len(paths) != 4 or tuple(path.stem for path in paths) != expected:
        raise DatasetSchemaError("PNG handoff does not match the four planned frame IDs")
    for path in paths:
        if path.suffix.lower() != ".png" or not path.is_file():
            raise DatasetSchemaError(f"missing PNG handoff frame: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            raise DatasetSchemaError(f"unreadable PNG handoff frame: {path}") from error
    return paths


def _report(plan, frames, production, graph, trace):
    production_digest = getattr(production, "result_sha256", "")
    graph_digest = getattr(graph, "result_sha256", "")
    trace_digest = "" if trace is None else trace.attribution_sha256
    solver_reason = "" if graph is None else graph.reason
    target_frames = {job.capture.frame_id for job in plan.jobs[:3]}
    rows = []
    if trace is not None:
        for item in trace.pair_attribution:
            if not item.selected or item.left not in target_frames or item.right not in target_frames:
                continue
            required = (item.left_plane, item.right_plane, item.psr, item.score,
                        item.overlap_cells, item.valid_cells, item.overlap_fraction,
                        item.cell_shift_rc, item.left_structural_share,
                        item.right_structural_share)
            if any(value is None for value in required):
                continue
            displacement = max(abs(value) for value in item.cell_shift_rc)
            share = min(item.left_structural_share, item.right_structural_share)
            q = plan.quality
            encoded_psr = (item.psr if math.isfinite(item.psr)
                           else "positive_infinity")
            rows.append(PairMeasurement(
                item.left, item.right, item.left_axis, item.left_plane,
                item.right_plane, item.quadrant, encoded_psr, item.score,
                item.overlap_cells, item.valid_cells, item.overlap_fraction,
                item.cell_shift_rc, displacement, item.left_structural_share,
                item.right_structural_share,
                item.overlap_cells >= q.min_pair_support_cells,
                item.overlap_fraction >= q.min_pair_support_fraction,
                displacement <= q.max_predicted_displacement_cells,
                share >= q.min_structural_share, item.psr >= q.min_psr))
    rows.sort(key=lambda row: (row.left, row.right, row.axis, row.left_plane,
                               row.right_plane, row.quadrant))
    axes = []
    for axis in sorted({row.axis for row in rows}):
        group = [row for row in rows if row.axis == axis]
        minimum_psr = min(float("inf") if row.psr == "positive_infinity"
                          else row.psr for row in group)
        axes.append(AxisMeasurement(
            axis, len(group), min(row.overlap_cells for row in group),
            min(row.overlap_fraction for row in group),
            max(row.displacement_cells for row in group),
            min(min(row.left_structural_share, row.right_structural_share)
                for row in group),
            minimum_psr if math.isfinite(minimum_psr) else "positive_infinity"))
    ordered_target_frames = tuple(job.capture.frame_id for job in plan.jobs[:3])
    required_pairs = {tuple(sorted((ordered_target_frames[left],
                                    ordered_target_frames[right])))
                      for left, right in ((0, 1), (1, 2), (0, 2))}
    observed = {tuple(sorted((row.left, row.right))) for row in rows}
    if not rows or not required_pairs.issubset(observed):
        status, reason = "insufficient_evidence", "missing_axis0_pair_evidence"
    elif any(not all((row.support_cells_pass, row.support_fraction_pass,
                      row.displacement_pass, row.structural_share_pass,
                      row.psr_pass)) for row in rows):
        status, reason = "fail", "quality_target_failed"
    elif graph is None or graph.status != "assembled":
        status, reason = "insufficient_evidence", "graph_unavailable"
    else:
        status, reason = "pass", ""
    return TargetedCaptureQualityReport(
        SCHEMA, PRODUCER, status, reason, "", plan.plan_sha256, frames,
        production_digest, graph_digest, trace_digest, solver_reason,
        tuple(rows), tuple(axes))


def evaluate_targeted_capture_quality(
    *, plan: TargetedCapturePlan, png_handoff: object, project_path: str | Path,
) -> TargetedCaptureQualityReport:
    """Run ordinary-image production and inspect the same matcher trace."""
    accepted = TargetedCapturePlan.from_record(plan.to_record())
    paths = _paths(accepted, png_handoff)
    production = production_mod.derive_image_output_from_paths(
        paths, project_path=project_path)
    frames = tuple(path.stem for path in paths)
    if isinstance(production, plane_placement.PlacementRefusal) \
            or production.lattice is None:
        return _report(accepted, frames, production, None, None)
    signatures = signatures_mod.derive_image_derived_signatures(
        session=production.entry.session, lattice_result=production.lattice)
    if isinstance(signatures, plane_placement.PlacementRefusal):
        return _report(accepted, frames, production, None, None)
    graph, trace = trace_mod.derive_image_derived_graph_attribution(
        session=production.entry.session, lattice_result=production.lattice,
        signatures_result=signatures, connection_graph=production.entry.graph)
    if isinstance(graph, plane_placement.PlacementRefusal):
        return _report(accepted, frames, production, None, None)
    return _report(accepted, frames, production, graph, trace)
