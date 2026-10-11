"""Bind an accepted image-derived graph to the existing C1 producer."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_joint_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signature_mod
from mc_photogrammetry.reconstruction import lattice_planes, plane_axis_support
from mc_photogrammetry.reconstruction import plane_graph, plane_residue
from mc_photogrammetry.reconstruction import plane_placement as placement

__all__ = ["IMAGE_DERIVED_AXIS_SUPPORT_SCHEMA",
           "IMAGE_DERIVED_AXIS_SUPPORT_PRODUCER",
           "ImageDerivedAxisSupportResult", "derive_image_derived_axis_support"]

IMAGE_DERIVED_AXIS_SUPPORT_SCHEMA = "mc-photogrammetry-image-derived-axis-support-v2"
IMAGE_DERIVED_AXIS_SUPPORT_PRODUCER = "m9-h-image-derived-axis-support-2"
AXIS_SUPPORT_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _support_sha(value: plane_axis_support.C1AxisSupport | None) -> str:
    return "" if value is None else _sha(value.to_record())


def _payload(value: "ImageDerivedAxisSupportResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "graph_sha256": value.graph_sha256,
        "session_sha256": value.session_sha256,
        "lattice_sha256": value.lattice_sha256,
        "signatures_sha256": value.signatures_sha256,
        "correlation_algorithm_id": value.correlation_algorithm_id,
        "max_shift_cells": value.max_shift_cells,
        "frame_ids": list(value.frame_ids),
        "axis_support_sha256": _support_sha(value.axis_support),
    }


@dataclass(frozen=True, slots=True)
class ImageDerivedAxisSupportResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    graph_sha256: str
    session_sha256: str
    lattice_sha256: str
    signatures_sha256: str
    frame_ids: tuple[str, ...]
    correlation_algorithm_id: str
    max_shift_cells: int
    axis_support: plane_axis_support.C1AxisSupport | None
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_AXIS_SUPPORT_SCHEMA
                or self.producer != IMAGE_DERIVED_AXIS_SUPPORT_PRODUCER
                or self.terminal != AXIS_SUPPORT_TERMINAL
                or self.status not in ("assembled", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not all(_hex(value) for value in (
                    self.graph_sha256, self.session_sha256,
                    self.lattice_sha256, self.signatures_sha256))
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(value) is not str or not value for value in self.frame_ids)):
            raise ValueError("image-derived axis-support result is malformed")
        if (self.correlation_algorithm_id != plane_graph.PLANE_CORRELATION_ALGORITHM
                or type(self.max_shift_cells) is not int
                or self.max_shift_cells <= 0):
            raise ValueError("axis-support correlation provenance is unsupported")
        if self.status == "assembled":
            if (self.reason or self.detail
                    or not isinstance(self.axis_support,
                                      plane_axis_support.C1AxisSupport)
                    or self.axis_support.refusal is not None):
                raise ValueError("assembled axis support is inconsistent")
            replayed = plane_axis_support.C1AxisSupport(**{
                field.name: getattr(self.axis_support, field.name)
                for field in dataclasses.fields(plane_axis_support.C1AxisSupport)})
            if replayed.to_record() != self.axis_support.to_record():
                raise ValueError("axis support does not replay")
            for mapped in self.axis_support.mapped_cell_shifts:
                support = mapped.support
                if (support.schema != plane_residue.RELATIVE_SHIFT_SUPPORT_SCHEMA
                        or support.version
                        != plane_residue.RELATIVE_SHIFT_SUPPORT_VERSION
                        or support.correlation_algorithm_id
                        != self.correlation_algorithm_id):
                    raise ValueError("axis support carries foreign shift provenance")
        elif (not self.reason or self.axis_support is not None):
            raise ValueError("unavailable axis support must be atomic")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived axis-support digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = "") -> placement.PlacementRefusal:
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _replay_graph(value: object) -> graph_mod.ImageDerivedGraphResult | placement.PlacementRefusal:
    if not isinstance(value, graph_mod.ImageDerivedGraphResult):
        return _refusal("invalid_input", "graph_result")
    try:
        fields = {field.name: getattr(value, field.name)
                  for field in dataclasses.fields(graph_mod.ImageDerivedGraphResult)}
        graph_mod.ImageDerivedGraphResult(**fields)
        graph_mod._require_live(value)
        return value
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "graph_result",
                        type(error).__name__)


def _unavailable(graph: graph_mod.ImageDerivedGraphResult,
                 reason: str, detail: str) -> ImageDerivedAxisSupportResult:
    return ImageDerivedAxisSupportResult(
        IMAGE_DERIVED_AXIS_SUPPORT_SCHEMA, IMAGE_DERIVED_AXIS_SUPPORT_PRODUCER,
        AXIS_SUPPORT_TERMINAL, "unavailable", reason, detail,
        graph.result_sha256, graph.session_sha256, graph.lattice_sha256,
        graph.signatures_sha256, graph.frame_ids, graph.correlation_algorithm,
        graph.max_shift_cells, None)


def _reading_index(readings: tuple) -> tuple[dict[tuple[str, int], object], dict[int, tuple[str, int]]]:
    by_frame: dict[str, list[lattice_planes.PlaneReading]] = {}
    for reading in readings:
        if not isinstance(reading, lattice_planes.PlaneReading):
            raise ValueError("graph contains a non-reading")
        by_frame.setdefault(reading.frame, []).append(reading)
    indexed: dict[tuple[str, int], object] = {}
    reverse: dict[int, tuple[str, int]] = {}
    for frame, values in by_frame.items():
        ordered = sorted(values, key=lattice_planes.reading_order_key)
        if len({lattice_planes.reading_order_key(value) for value in ordered}) != len(ordered):
            raise ValueError("reading order is ambiguous")
        for index, reading in enumerate(ordered):
            indexed[(frame, index)] = reading
            reverse[id(reading)] = (frame, index)
    return indexed, reverse


def _bind_universe(graph: graph_mod.ImageDerivedGraphResult) -> tuple:
    readings = graph.readings
    evidence = graph.signatures
    matches = graph.matches
    solution = graph.component_solution
    if (not readings or not evidence or not matches
            or not isinstance(solution, plane_graph.ComponentSolution)
            or solution.reason):
        raise ValueError("assembled graph carries incomplete evidence")
    indexed, reverse = _reading_index(readings)
    expected = [(index, alternative_index)
                for index, reading in enumerate(readings)
                for alternative_index, _ in enumerate(reading.alternatives)]
    actual = [(item.canonical_index, item.alternative_index) for item in evidence]
    if actual != expected:
        raise ValueError("signature evidence is reordered or incomplete")
    raw = []
    for item in evidence:
        signature_mod.SignatureEvidence(**{
            field.name: getattr(item, field.name)
            for field in dataclasses.fields(signature_mod.SignatureEvidence)})
        reading = readings[item.canonical_index]
        alternative = reading.alternatives[item.alternative_index]
        if (item.signature.reading is not reading or item.frame_id != reading.frame
                or item.reading_axis != reading.axis
                or item.blocks_per_period != alternative.blocks_per_period):
            raise ValueError("signature evidence is foreign")
        raw.append(item.signature)
    for match in matches:
        if not isinstance(match, plane_graph.PlaneMatch):
            raise ValueError("graph contains a non-match")
        left = indexed.get((match.left, match.left_plane))
        right = indexed.get((match.right, match.right_plane))
        if (left is None or right is None or left.axis != match.left_axis
                or right.axis != match.right_axis
                or left.distance_for(match.left_b) is None
                or right.distance_for(match.right_b) is None):
            raise ValueError("match endpoint is foreign")
    reading_frames = tuple(sorted({reading.frame for reading in readings}))
    if (solution.frames != reading_frames
            or tuple(sorted(solution.quadrants)) != solution.frames
            or tuple(sorted(solution.translations)) != solution.frames
            or any(key not in reverse.values() for key in solution.blocks_per_period)):
        raise ValueError("component solution is foreign to the reading universe")
    return readings, evidence, matches, solution, tuple(raw)


def derive_image_derived_axis_support(
    *, graph_result: object,
) -> ImageDerivedAxisSupportResult | placement.PlacementRefusal:
    """Derive C1 from one exact accepted graph, without adding geometry."""
    graph = _replay_graph(graph_result)
    if isinstance(graph, placement.PlacementRefusal):
        return graph
    if graph.status != "assembled":
        return _unavailable(graph, "graph_unavailable", graph.reason)
    try:
        readings, _evidence, matches, solution, raw = _bind_universe(graph)
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "graph_result.binding",
                        type(error).__name__)
    try:
        support = plane_axis_support.build_axis_support(
            solution, readings, matches, raw)
        replayed = plane_axis_support.C1AxisSupport(**{
            field.name: getattr(support, field.name)
            for field in dataclasses.fields(plane_axis_support.C1AxisSupport)})
    except _ERRORS as error:
        return _refusal("invalid_input", "axis_support",
                        type(error).__name__)
    if replayed.to_record() != support.to_record():
        return _refusal("provenance_mismatch", "axis_support")
    if support.refusal is not None:
        return _unavailable(graph, support.refusal,
                            "C1 axis-support producer abstained")
    if (support.frames != solution.frames
            or support.global_anchor != solution.frames[0]):
        return _refusal("provenance_mismatch", "axis_support.binding")
    match_by_identity = {
        plane_axis_support.C1MatchIdentity.from_match(match): match
        for match in matches}
    for mapped in support.mapped_cell_shifts:
        match = match_by_identity.get(mapped.match_identity)
        if match is None:
            return _refusal("provenance_mismatch", "axis_support.match_identity")
        try:
            replayed_psr = plane_residue._psr_from_record(
                mapped.support.to_record()["psr"])
        except _ERRORS as error:
            return _refusal("provenance_mismatch", "axis_support.psr",
                            type(error).__name__)
        if replayed_psr != match.psr:
            return _refusal("provenance_mismatch", "axis_support.psr")
    return ImageDerivedAxisSupportResult(
        IMAGE_DERIVED_AXIS_SUPPORT_SCHEMA, IMAGE_DERIVED_AXIS_SUPPORT_PRODUCER,
        AXIS_SUPPORT_TERMINAL, "assembled", "", "", graph.result_sha256,
        graph.session_sha256, graph.lattice_sha256, graph.signatures_sha256,
        graph.frame_ids, graph.correlation_algorithm, graph.max_shift_cells,
        support)
