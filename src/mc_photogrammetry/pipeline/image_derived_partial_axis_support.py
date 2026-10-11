"""Core-only C1 projection retaining the complete partial-graph authority."""
from __future__ import annotations

from dataclasses import dataclass

from . import image_derived_endpoint_selection as selection
from . import image_derived_graph as legacy
from . import image_derived_partial_joint_graph as graph_mod
from mc_photogrammetry.reconstruction import plane_axis_support, plane_placement as placement

SCHEMA = "mc-photogrammetry-partial-core-axis-support-v1"
PRODUCER = "m9-h-partial-core-axis-support-1"


def _source_support(graph):
    graph_mod._require_live(graph)
    if graph.status != "assembled":
        return "partial_graph_unavailable", None
    readings, signatures, matches = graph_mod._core_inputs(
        graph.readings, graph.signatures, graph.matches, graph.selection)
    core = graph.selection.observability[0].core
    # C1 only consumes direct source matches. Connectivity through an excluded
    # peripheral cannot be replaced with an invented core-to-core match.
    if legacy._production_solution_refusal(core, readings, matches, graph.core_solution) is not None:
        return "partial_core_direct_support_unavailable", None
    support = plane_axis_support.build_axis_support(graph.core_solution, readings, matches, signatures)
    if support.refusal is not None:
        return support.refusal, None
    if support.frames != core or support.global_anchor != graph.selection.observability[0].anchor:
        raise ValueError("partial core C1 gauge differs")
    return "", support


def _projection(graph):
    if graph.selection.status != "unique":
        return (), ()
    core = graph.selection.observability[0].core
    readings = tuple((index, reading.frame, reading.axis) for index, reading in enumerate(graph.readings)
                     if reading.frame in core)
    rows = tuple(rep.source_row for rep in graph.selection.representatives
                 if rep.match.left in core and rep.match.right in core)
    return readings, rows


def _payload(value):
    return dict(schema=value.schema, producer=value.producer, status=value.status,
        reason=value.reason, graph_sha256=value.graph.result_sha256,
        core_reading_indices=value.core_reading_indices, source_rows=value.source_rows,
        axis_support=None if value.axis_support is None else value.axis_support.to_record())


@dataclass(frozen=True, slots=True)
class PartialCoreAxisSupportResult:
    schema: str
    producer: str
    status: str
    reason: str
    graph: graph_mod.PartialJointGraphResult
    core_reading_indices: tuple
    source_rows: tuple
    axis_support: plane_axis_support.C1AxisSupport | None
    result_sha256: str = ""

    def __post_init__(self):
        if self.schema != SCHEMA or self.producer != PRODUCER or self.status not in ("assembled", "unavailable"):
            raise ValueError("partial core support identity differs")
        if (type(self.core_reading_indices) is not tuple or type(self.source_rows) is not tuple
                or any(type(row) is not tuple or len(row) != 3
                       or type(row[0]) is not int or type(row[1]) is not str
                       or type(row[2]) is not int for row in self.core_reading_indices)
                or any(type(index) is not int for index in self.source_rows)):
            raise ValueError("partial core source projection types differ")
        reason, expected = _source_support(self.graph)
        if (self.reason != reason or self.status != ("assembled" if expected is not None else "unavailable")
                or (self.axis_support is None) != (expected is None)
                or expected is not None and self.axis_support.to_record() != expected.to_record()
                or (self.core_reading_indices, self.source_rows) != _projection(self.graph)):
            raise ValueError("partial core support does not replay from complete graph")
        digest = selection._sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("partial core support digest differs")
        object.__setattr__(self, "result_sha256", digest)


def derive_partial_core_axis_support(*, graph_result):
    """Explicit supported consumer; never masquerades as whole-session C1."""
    try:
        reason, support = _source_support(graph_result)
        reading_indices, rows = _projection(graph_result)
        return PartialCoreAxisSupportResult(SCHEMA, PRODUCER,
            "assembled" if support is not None else "unavailable", reason,
            graph_result, reading_indices, rows, support)
    except graph_mod.source_mod._ERRORS as error:
        return legacy._refusal("provenance_mismatch", "partial_core_axis_support", type(error).__name__)
