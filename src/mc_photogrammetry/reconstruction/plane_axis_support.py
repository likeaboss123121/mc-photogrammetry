"""C1 mapped integer-shift evidence and per-axis support topology.

This module records the discrete coordinate transform and which frame-axis
translations have relative evidence. It never solves a fractional phase or
adds cell shifts to numeric translations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph, plane_residue


C1_REFUSAL_REASONS = frozenset({
    "solution_abstained", "frame_missing", "reading_missing",
    "quadrant_invalid", "relative_quadrant_mismatch",
    "alternative_not_selected", "phase_shift_identity_mismatch",
    "cell_shift_absent", "nonfinite_support", "axis_transform_invalid",
    "signature_missing", "signature_ambiguous",
    "signature_identity_mismatch", "homography_invalid",
    "selected_provenance_missing", "disconnected_from_global_gauge",
    "no_constraints",
})
_EVIDENCE_CLASSES = ("normal_face", "lateral_shift")
_DIGEST_LENGTH = 64


def _finite(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _valid_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == _DIGEST_LENGTH
            and all(char in "0123456789abcdef" for char in value))


@dataclass(frozen=True, slots=True)
class C1MatchIdentity:
    """Stable full identity shared by C1 records for one admitted match."""

    left_frame: str
    right_frame: str
    left_plane: int
    right_plane: int
    left_axis: int
    right_axis: int
    left_b: float
    right_b: float
    quadrant_of_right: int

    def __post_init__(self) -> None:
        if (type(self.left_frame) is not str or not self.left_frame
                or type(self.right_frame) is not str or not self.right_frame
                or type(self.left_plane) is not int or self.left_plane < 0
                or type(self.right_plane) is not int or self.right_plane < 0
                or type(self.left_axis) is not int or self.left_axis not in (0, 1, 2)
                or type(self.right_axis) is not int or self.right_axis not in (0, 1, 2)
                or _finite(self.left_b) is None or self.left_b <= 0.0
                or _finite(self.right_b) is None or self.right_b <= 0.0
                or type(self.quadrant_of_right) is not int
                or self.quadrant_of_right not in (0, 1, 2, 3)):
            raise ValueError("C1 match identity is invalid")

    @classmethod
    def from_match(cls, match: plane_graph.PlaneMatch) -> "C1MatchIdentity":
        return cls(
            match.left, match.right, match.left_plane, match.right_plane,
            match.left_axis, match.right_axis, match.left_b,
            match.right_b, match.quadrant_of_right)

    def sort_key(self) -> tuple:
        return (self.left_frame, self.right_frame, self.left_plane,
                self.right_plane, self.left_axis, self.right_axis,
                self.left_b, self.right_b, self.quadrant_of_right)

    def to_record(self) -> dict[str, Any]:
        return {
            "left_frame": self.left_frame,
            "right_frame": self.right_frame,
            "left_plane": self.left_plane,
            "right_plane": self.right_plane,
            "left_axis": self.left_axis,
            "right_axis": self.right_axis,
            "left_b": self.left_b,
            "right_b": self.right_b,
            "quadrant_of_right": self.quadrant_of_right,
        }


@dataclass(frozen=True, slots=True)
class MappedCellShiftAxis:
    """One signed row/column integer shift mapped through the right view."""

    match_identity: C1MatchIdentity
    left_frame_quadrant: int
    right_frame_quadrant: int
    right_local_axis: str
    right_local_axis_label: int
    source_component: str
    right_grid_shift: int
    world_axis: int
    right_world_sign: int
    signed_world_integer: int
    left_signature_geometry_sha256: str
    right_signature_geometry_sha256: str
    cell_homography_sha256: str
    support: plane_residue.RelativeShiftSupport

    def __post_init__(self) -> None:
        if not isinstance(self.match_identity, C1MatchIdentity):
            raise ValueError("mapped shift requires a typed match identity")
        if (type(self.left_frame_quadrant) is not int
                or self.left_frame_quadrant not in (0, 1, 2, 3)
                or type(self.right_frame_quadrant) is not int
                or self.right_frame_quadrant not in (0, 1, 2, 3)
                or ((self.right_frame_quadrant - self.left_frame_quadrant) % 4
                    != self.match_identity.quadrant_of_right)):
            raise ValueError("mapped shift quadrants do not match the relative identity")
        if self.right_local_axis not in ("u", "v"):
            raise ValueError("mapped shift local axis must be u or v")
        tangent_labels = tuple(
            label for label in range(3) if label != self.match_identity.right_axis)
        expected_label = tangent_labels[0 if self.right_local_axis == "u" else 1]
        if (type(self.right_local_axis_label) is not int
                or self.right_local_axis_label != expected_label):
            raise ValueError("mapped shift local label disagrees with the lattice basis")
        expected_source = "column" if self.right_local_axis == "u" else "row"
        if self.source_component != expected_source:
            raise ValueError("mapped shift source component must preserve row=v/column=u")
        if (type(self.right_grid_shift) is not int
                or type(self.world_axis) is not int or self.world_axis not in (0, 1, 2)
                or type(self.right_world_sign) is not int
                or self.right_world_sign not in (-1, 1)
                or type(self.signed_world_integer) is not int
                or self.signed_world_integer
                != self.right_world_sign * self.right_grid_shift):
            raise ValueError("mapped shift integer/sign fields are inconsistent")
        expected_axis, expected_sign = plane_graph.axis_transform(
            self.right_local_axis_label, self.right_frame_quadrant)
        if (self.world_axis, self.right_world_sign) != (expected_axis, expected_sign):
            raise ValueError("mapped shift disagrees with the public axis transform")
        if not all(_valid_digest(value) for value in (
                self.left_signature_geometry_sha256,
                self.right_signature_geometry_sha256,
                self.cell_homography_sha256)):
            raise ValueError("mapped shift geometry identities must be SHA-256 digests")
        if (not isinstance(self.support, plane_residue.RelativeShiftSupport)
                or self.support.refusal is not None
                or _support_identity(self.support) != self.match_identity):
            raise ValueError("mapped shift must retain matching non-refused P1-B support")
        assert self.support.cell_shift_rc is not None
        expected_shift = (self.support.cell_shift_rc[1]
                          if self.right_local_axis == "u"
                          else self.support.cell_shift_rc[0])
        if self.right_grid_shift != expected_shift:
            raise ValueError("mapped shift does not copy its exact right-grid component")

    def to_record(self) -> dict[str, Any]:
        return {
            "match_identity": self.match_identity.to_record(),
            "left_frame_quadrant": self.left_frame_quadrant,
            "right_frame_quadrant": self.right_frame_quadrant,
            "right_local_axis": self.right_local_axis,
            "right_local_axis_label": self.right_local_axis_label,
            "source_component": self.source_component,
            "right_grid_shift": self.right_grid_shift,
            "world_axis": self.world_axis,
            "right_world_sign": self.right_world_sign,
            "signed_world_integer": self.signed_world_integer,
            "left_signature_geometry_sha256": self.left_signature_geometry_sha256,
            "right_signature_geometry_sha256": self.right_signature_geometry_sha256,
            "cell_homography_sha256": self.cell_homography_sha256,
            "support": self.support.to_record(),
        }


def _support_identity(value: plane_residue.RelativeShiftSupport) -> C1MatchIdentity | None:
    left_b = _finite(value.left_b)
    right_b = _finite(value.right_b)
    if left_b is None or right_b is None:
        return None
    try:
        return C1MatchIdentity(
            value.left_frame, value.right_frame, value.left_plane, value.right_plane,
            value.left_axis, value.right_axis, left_b, right_b,
            value.quadrant_of_right)
    except (ValueError, TypeError, OverflowError):
        return None


@dataclass(frozen=True, slots=True)
class MatchSupportRefusal:
    match_identity: C1MatchIdentity
    refusal: str
    support: plane_residue.RelativeShiftSupport

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity, C1MatchIdentity)
                or not isinstance(self.support, plane_residue.RelativeShiftSupport)
                or type(self.refusal) is not str
                or self.refusal not in {"cell_shift_absent", "nonfinite_support"}
                or _support_identity(self.support) != self.match_identity):
            raise ValueError("C1 lateral support refusal is invalid")
        if self.support.refusal == "cell_shift_absent":
            valid = (self.refusal == "cell_shift_absent"
                     and self.support.cell_shift_rc is None
                     and not self.support.nonfinite_fields)
        elif self.support.refusal == "nonfinite_derived_value":
            valid = (self.refusal == "nonfinite_support"
                     and bool(self.support.nonfinite_fields))
        else:
            valid = False
        if not valid:
            raise ValueError("C1 lateral refusal disagrees with copied P1-B refusal")

    @property
    def p1b_refusal(self) -> str:
        return str(self.support.refusal)

    def to_record(self) -> dict[str, Any]:
        return {"match_identity": self.match_identity.to_record(),
                "refusal": self.refusal, "p1b_refusal": self.p1b_refusal,
                "support": self.support.to_record()}


@dataclass(frozen=True, slots=True)
class NormalFaceConstraint:
    """Retained metric normal-face equation for one selected match."""

    match_identity: C1MatchIdentity
    left_frame_quadrant: int
    right_frame_quadrant: int
    world_axis: int
    left_world_sign: int
    right_world_sign: int
    left_side: float
    right_side: float
    left_distance_blocks: float
    right_distance_blocks: float
    delta_translation: float

    def __post_init__(self) -> None:
        if (not isinstance(self.match_identity, C1MatchIdentity)
                or type(self.left_frame_quadrant) is not int
                or self.left_frame_quadrant not in (0, 1, 2, 3)
                or type(self.right_frame_quadrant) is not int
                or self.right_frame_quadrant not in (0, 1, 2, 3)
                or ((self.right_frame_quadrant - self.left_frame_quadrant) % 4
                    != self.match_identity.quadrant_of_right)
                or type(self.world_axis) is not int or self.world_axis not in (0, 1, 2)
                or type(self.left_world_sign) is not int
                or self.left_world_sign not in (-1, 1)
                or type(self.right_world_sign) is not int
                or self.right_world_sign not in (-1, 1)):
            raise ValueError("normal-face constraint identity/sign fields are invalid")
        left_axis, left_sign = plane_graph.axis_transform(
            self.match_identity.left_axis, self.left_frame_quadrant)
        right_axis, right_sign = plane_graph.axis_transform(
            self.match_identity.right_axis, self.right_frame_quadrant)
        if (left_axis != right_axis
                or (self.world_axis, self.left_world_sign, self.right_world_sign)
                != (left_axis, left_sign, right_sign)):
            raise ValueError("normal-face transform disagrees with absolute quadrants")
        values = (self.left_side, self.right_side, self.left_distance_blocks,
                  self.right_distance_blocks, self.delta_translation)
        if any(_finite(value) is None for value in values):
            raise ValueError("normal-face constraint values must be finite")
        if (abs(self.left_side) != 1.0 or abs(self.right_side) != 1.0
                or self.left_distance_blocks <= 0.0
                or self.right_distance_blocks <= 0.0):
            raise ValueError("normal-face side/distance inputs are invalid")
        expected = (self.left_world_sign * self.left_side * self.left_distance_blocks
                    - self.right_world_sign * self.right_side
                    * self.right_distance_blocks)
        if self.delta_translation != expected:
            raise ValueError("normal-face delta disagrees with the metric equation")

    def to_record(self) -> dict[str, Any]:
        return {
            "match_identity": self.match_identity.to_record(),
            "left_frame_quadrant": self.left_frame_quadrant,
            "right_frame_quadrant": self.right_frame_quadrant,
            "world_axis": self.world_axis,
            "left_world_sign": self.left_world_sign,
            "right_world_sign": self.right_world_sign,
            "left_side": self.left_side,
            "right_side": self.right_side,
            "left_distance_blocks": self.left_distance_blocks,
            "right_distance_blocks": self.right_distance_blocks,
            "delta_translation": self.delta_translation,
        }


@dataclass(frozen=True, slots=True)
class AxisSupportComponent:
    world_axis: int
    frames: tuple[str, ...]
    gauge_anchor: str
    global_anchor: str
    match_identities: tuple[C1MatchIdentity, ...]
    normal_face_evidence_count: int
    lateral_shift_evidence_count: int
    contains_global_anchor: bool
    status: str

    def __post_init__(self) -> None:
        if (type(self.world_axis) is not int or self.world_axis not in (0, 1, 2)
                or not isinstance(self.frames, tuple) or not self.frames
                or any(type(frame) is not str or not frame for frame in self.frames)
                or tuple(sorted(set(self.frames))) != self.frames
                or self.gauge_anchor != self.frames[0]
                or type(self.global_anchor) is not str or not self.global_anchor
                or not isinstance(self.match_identities, tuple)
                or any(not isinstance(value, C1MatchIdentity)
                       for value in self.match_identities)
                or tuple(sorted(set(self.match_identities), key=lambda v: v.sort_key()))
                != self.match_identities
                or type(self.normal_face_evidence_count) is not int
                or type(self.lateral_shift_evidence_count) is not int
                or self.normal_face_evidence_count < 0
                or self.lateral_shift_evidence_count < 0
                or self.normal_face_evidence_count > len(self.match_identities)
                or self.lateral_shift_evidence_count > len(self.match_identities)
                or type(self.contains_global_anchor) is not bool
                or self.contains_global_anchor != (self.global_anchor in self.frames)):
            raise ValueError("axis support component fields are invalid")
        has_constraints = bool(self.match_identities)
        expected = ("no_constraints" if not has_constraints else
                    "supported" if self.contains_global_anchor else
                    "disconnected_from_global_gauge")
        if self.status != expected:
            raise ValueError("axis support component status disagrees with its evidence")

    def to_record(self) -> dict[str, Any]:
        return {
            "world_axis": self.world_axis,
            "frames": list(self.frames),
            "gauge_anchor": self.gauge_anchor,
            "global_anchor": self.global_anchor,
            "match_identities": [value.to_record() for value in self.match_identities],
            "normal_face_evidence_count": self.normal_face_evidence_count,
            "lateral_shift_evidence_count": self.lateral_shift_evidence_count,
            "contains_global_anchor": self.contains_global_anchor,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class FrameAxisSupport:
    frame: str
    world_axis: int
    axis_component_anchor: str
    connected_to_global_anchor: bool
    contributing_edge_count: int
    independent_frame_pair_count: int
    evidence_classes: tuple[str, ...]
    refusal: str | None

    def __post_init__(self) -> None:
        if (type(self.frame) is not str or not self.frame
                or type(self.world_axis) is not int or self.world_axis not in (0, 1, 2)
                or type(self.axis_component_anchor) is not str
                or not self.axis_component_anchor
                or type(self.connected_to_global_anchor) is not bool
                or type(self.contributing_edge_count) is not int
                or type(self.independent_frame_pair_count) is not int
                or self.contributing_edge_count < 0
                or self.independent_frame_pair_count < 0
                or not isinstance(self.evidence_classes, tuple)
                or any(value not in _EVIDENCE_CLASSES for value in self.evidence_classes)
                or tuple(value for value in _EVIDENCE_CLASSES
                         if value in self.evidence_classes) != self.evidence_classes
                or (self.refusal is not None and
                    (type(self.refusal) is not str
                     or self.refusal not in C1_REFUSAL_REASONS))):
            raise ValueError("frame-axis support fields are invalid")
        if self.contributing_edge_count == 0:
            if (self.independent_frame_pair_count != 0 or self.evidence_classes
                    or self.connected_to_global_anchor
                    or self.refusal != "no_constraints"):
                raise ValueError("unsupported frame axis must report no_constraints")
        elif (self.independent_frame_pair_count == 0 or not self.evidence_classes):
            raise ValueError("constrained frame axis has incomplete support counts")
        elif self.connected_to_global_anchor != (self.refusal is None):
            raise ValueError("frame-axis connectivity disagrees with refusal")
        elif (not self.connected_to_global_anchor
              and self.refusal != "disconnected_from_global_gauge"):
            raise ValueError("disconnected frame axis has an incorrect refusal")

    def to_record(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "world_axis": self.world_axis,
            "axis_component_anchor": self.axis_component_anchor,
            "connected_to_global_anchor": self.connected_to_global_anchor,
            "contributing_edge_count": self.contributing_edge_count,
            "independent_frame_pair_count": self.independent_frame_pair_count,
            "evidence_classes": list(self.evidence_classes),
            "refusal": self.refusal,
        }


@dataclass(frozen=True, slots=True)
class C1AxisSupport:
    refusal: str | None
    frames: tuple[str, ...]
    global_anchor: str | None
    normal_face_constraints: tuple[NormalFaceConstraint, ...]
    mapped_cell_shifts: tuple[MappedCellShiftAxis, ...]
    match_refusals: tuple[MatchSupportRefusal, ...]
    axis_components: tuple[AxisSupportComponent, ...]
    frame_support: tuple[FrameAxisSupport, ...]

    def __post_init__(self) -> None:
        if (self.refusal is not None and
                (type(self.refusal) is not str
                 or self.refusal not in C1_REFUSAL_REASONS)):
            raise ValueError("C1 result refusal is unknown")
        if not isinstance(self.frames, tuple):
            raise ValueError("C1 frame order must be a tuple")
        for values, kind in (
            (self.normal_face_constraints, NormalFaceConstraint),
            (self.mapped_cell_shifts, MappedCellShiftAxis),
            (self.match_refusals, MatchSupportRefusal),
            (self.axis_components, AxisSupportComponent),
            (self.frame_support, FrameAxisSupport),
        ):
            if not isinstance(values, tuple) or any(not isinstance(value, kind)
                                                    for value in values):
                raise ValueError("C1 result contains an invalid evidence tuple")
        if self.refusal is not None and any((self.frames, self.global_anchor,
                                             self.normal_face_constraints,
                                             self.mapped_cell_shifts,
                                             self.match_refusals,
                                             self.axis_components,
                                             self.frame_support)):
            raise ValueError("refused C1 result cannot contain partial support")
        if self.refusal is not None:
            return
        if (not self.frames or any(type(frame) is not str or not frame
                                   for frame in self.frames)
                or tuple(sorted(set(self.frames))) != self.frames
                or self.global_anchor != self.frames[0]
                or not self.normal_face_constraints):
            raise ValueError("successful C1 result requires ordered frames and selected constraints")
        normal_ids = tuple(value.match_identity for value in self.normal_face_constraints)
        if (len(set(normal_ids)) != len(normal_ids)
                or tuple(sorted(self.normal_face_constraints,
                                key=lambda v: v.match_identity.sort_key()))
                != self.normal_face_constraints):
            raise ValueError("normal-face constraints must be uniquely and deterministically ordered")
        mapped_keys = tuple((value.match_identity, value.right_local_axis)
                            for value in self.mapped_cell_shifts)
        if (len(set(mapped_keys)) != len(mapped_keys)
                or tuple(sorted(self.mapped_cell_shifts, key=lambda v: (
                    v.match_identity.sort_key(), v.world_axis, v.right_local_axis)))
                != self.mapped_cell_shifts):
            raise ValueError("mapped shifts must be unique and deterministically ordered")
        refusal_ids = tuple(value.match_identity for value in self.match_refusals)
        if (len(set(refusal_ids)) != len(refusal_ids)
                or tuple(sorted(self.match_refusals, key=lambda v: (
                    v.match_identity.sort_key(), v.refusal))) != self.match_refusals
                or set(refusal_ids) & {key[0] for key in mapped_keys}):
            raise ValueError("lateral mapped/refused match identities conflict")
        identities = set(normal_ids)
        if any(value.match_identity not in identities for value in self.mapped_cell_shifts):
            raise ValueError("mapped shift lacks its selected normal-face match")
        if any(value.match_identity not in identities for value in self.match_refusals):
            raise ValueError("lateral refusal lacks its selected normal-face match")
        for identity in identities:
            if identity.left_frame not in self.frames or identity.right_frame not in self.frames:
                raise ValueError("selected evidence endpoint is outside the result frames")
        for mapped in self.mapped_cell_shifts:
            if (mapped.match_identity.left_frame not in self.frames
                    or mapped.match_identity.right_frame not in self.frames):
                raise ValueError("mapped shift endpoint is outside the result frames")
        for refusal in self.match_refusals:
            if (refusal.match_identity.left_frame not in self.frames
                    or refusal.match_identity.right_frame not in self.frames):
                raise ValueError("refused shift endpoint is outside the result frames")
        mapped_by_identity: dict[C1MatchIdentity, list[MappedCellShiftAxis]] = {}
        for mapped in self.mapped_cell_shifts:
            mapped_by_identity.setdefault(mapped.match_identity, []).append(mapped)
        refusals_by_identity = {value.match_identity: value
                               for value in self.match_refusals}
        normals_by_identity = {value.match_identity: value
                               for value in self.normal_face_constraints}
        for identity, normal in normals_by_identity.items():
            pair = mapped_by_identity.get(identity, [])
            refusal = refusals_by_identity.get(identity)
            if refusal is not None:
                if pair:
                    raise ValueError("one selected match cannot be mapped and refused")
                continue
            if (len(pair) != 2
                    or {value.right_local_axis for value in pair} != {"u", "v"}):
                raise ValueError("selected normal match needs both mapped axes or one refusal")
            if any((value.left_frame_quadrant, value.right_frame_quadrant)
                   != (normal.left_frame_quadrant, normal.right_frame_quadrant)
                   for value in pair):
                raise ValueError("mapped shifts disagree with selected normal quadrants")
            first, second = pair
            if (first.support != second.support
                    or first.left_signature_geometry_sha256
                    != second.left_signature_geometry_sha256
                    or first.right_signature_geometry_sha256
                    != second.right_signature_geometry_sha256
                    or first.cell_homography_sha256 != second.cell_homography_sha256):
                raise ValueError("mapped u/v shifts must share exact support and geometry provenance")
        if set(mapped_by_identity) | set(refusals_by_identity) != identities:
            raise ValueError("every selected normal identity needs mapped or refused lateral evidence")
        component_keys = tuple((value.world_axis, value.frames)
                               for value in self.axis_components)
        if (tuple(sorted(self.axis_components,
                         key=lambda value: (value.world_axis, value.gauge_anchor)))
                != self.axis_components
                or len(set(component_keys)) != len(component_keys)
                or tuple(sorted(self.frame_support,
                                key=lambda value: (value.frame, value.world_axis)))
                != self.frame_support
                or len({(value.frame, value.world_axis)
                        for value in self.frame_support}) != len(self.frame_support)
                or {(value.frame, value.world_axis) for value in self.frame_support}
                != {(frame, axis) for frame in self.frames for axis in range(3)}):
            raise ValueError("C1 support components/frame records are not complete and ordered")
        edge_classes: dict[int, dict[C1MatchIdentity, set[str]]] = {
            axis: {} for axis in range(3)}
        normal_axis = {}
        for constraint in self.normal_face_constraints:
            edge_classes[constraint.world_axis].setdefault(
                constraint.match_identity, set()).add("normal_face")
            normal_axis[constraint.match_identity] = constraint.world_axis
        for mapped in self.mapped_cell_shifts:
            if normal_axis[mapped.match_identity] == mapped.world_axis:
                raise ValueError("lateral shift cannot constrain the mapped normal axis")
            edge_classes[mapped.world_axis].setdefault(
                mapped.match_identity, set()).add("lateral_shift")
        expected_components, expected_frames = _build_graph_records(
            self.frames, self.global_anchor,
            {axis: {identity: frozenset(classes)
                    for identity, classes in values.items()}
             for axis, values in edge_classes.items()})
        if self.axis_components != expected_components or self.frame_support != expected_frames:
            raise ValueError("C1 component/frame support records disagree with evidence edges")

    def to_record(self) -> dict[str, Any]:
        record = {
            "refusal": self.refusal,
            "frames": list(self.frames),
            "global_anchor": self.global_anchor,
            "normal_face_constraints": [value.to_record()
                                         for value in self.normal_face_constraints],
            "mapped_cell_shifts": [value.to_record()
                                   for value in self.mapped_cell_shifts],
            "match_refusals": [value.to_record() for value in self.match_refusals],
            "axis_components": [value.to_record() for value in self.axis_components],
            "frame_support": [value.to_record() for value in self.frame_support],
        }
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record


def _selected_alternative(
    reading: lattice_planes.PlaneReading,
    selected_b: Any,
    match_b: Any,
) -> tuple[lattice_planes.DistanceAlternative | None, str | None]:
    if selected_b is None:
        return None, "unselected"
    if _finite(selected_b) is None or selected_b <= 0.0:
        return None, "invalid_selection"
    selected = reading.matching_alternatives(selected_b)
    matched = reading.matching_alternatives(match_b)
    if len(selected) != 1:
        return None, "invalid_selection"
    if len(matched) != 1:
        return None, "unselected"
    if selected[0].blocks_per_period != matched[0].blocks_per_period:
        return None, "unselected"
    distance = _finite(selected[0].distance_blocks)
    return ((selected[0], None) if distance is not None and distance > 0.0
            else (None, "invalid_selection"))


def _signature_geometry_digest(
    signature: lattice_planes.PlaneSignature,
    plane_index: int,
) -> str | None:
    """Bind image-measurement and signature-grid provenance, not the grid map."""
    try:
        reading = signature.reading
        rect = reading.rectification
        cells = np.asarray(signature.cells)
        coverage = np.asarray(signature.coverage)
        origin = tuple(float(value) for value in signature.origin_uv)
        matrix = np.asarray(rect.axis_matrix, dtype=np.float64)
        focal = float(rect.focal)
        principal = tuple(float(value) for value in rect.principal_point)
        centre = tuple(float(value) for value in rect.centre)
        b_value = float(signature.blocks_per_period)
        period_pixels = float(reading.period_pixels)
        extent = float(reading.extent)
        samples = reading.samples
        distance = reading.distance_for(b_value)
        shape = tuple(int(value) for value in cells.shape)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (cells.ndim != 2 or cells.size == 0 or coverage.shape != cells.shape
            or len(shape) != 2 or min(shape) <= 0
            or len(origin) != 2 or not all(math.isfinite(v) for v in origin)
            or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not math.isfinite(focal) or focal <= 0.0
            or not all(math.isfinite(v) for v in principal + centre)
            or not math.isfinite(period_pixels) or period_pixels <= 0.0
            or not math.isfinite(extent) or extent <= 0.0
            or type(samples) is not int or samples <= 0
            or distance is None or not math.isfinite(float(distance))):
        return None
    try:
        image_homography = lattice_planes.rectification_homography(
            reading.axis, matrix, focal, principal, reading.side, float(distance))
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None
    if image_homography.shape != (3, 3) or not np.isfinite(image_homography).all():
        return None
    record = {
        "frame": reading.frame,
        "plane_index": plane_index,
        "reading_axis": reading.axis,
        "side": float(reading.side),
        "blocks_per_period": b_value,
        "distance_blocks": float(distance),
        "period_pixels": period_pixels,
        "extent": extent,
        "samples": samples,
        "origin_uv": list(origin),
        "grid_shape": list(shape),
        "axis_matrix": matrix.tolist(),
        "focal": focal,
        "principal_point": list(principal),
        "rectification_centre": list(centre),
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _normalized_homography_digest(matrix: np.ndarray) -> str | None:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        return None
    scale = float(value[2, 2])
    if abs(scale) > 1e-15:
        normalized = value / scale
    else:
        norm = float(np.linalg.norm(value))
        if not math.isfinite(norm) or norm <= 0.0:
            return None
        normalized = value / norm
        nonzero = normalized[np.abs(normalized) > 1e-15]
        if nonzero.size and nonzero[0] < 0.0:
            normalized = -normalized
    if not np.isfinite(normalized).all():
        return None
    canonical = np.asarray(normalized, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()


def _grid_homography_digest(
    left: lattice_planes.PlaneSignature,
    right: lattice_planes.PlaneSignature,
    relative_quadrant: int,
    left_plane_index: int,
    right_plane_index: int,
    matrix: np.ndarray,
) -> str | None:
    """Bind a block-grid map to its ordered readings, origins and hypothesis."""
    normalized_digest = _normalized_homography_digest(matrix)
    if (normalized_digest is None or type(relative_quadrant) is not int
            or relative_quadrant not in (0, 1, 2, 3)
            or type(left_plane_index) is not int or left_plane_index < 0
            or type(right_plane_index) is not int or right_plane_index < 0):
        return None
    try:
        left_origin = tuple(float(value) for value in left.origin_uv)
        right_origin = tuple(float(value) for value in right.origin_uv)
        left_shape = tuple(int(value) for value in np.asarray(left.cells).shape)
        right_shape = tuple(int(value) for value in np.asarray(right.cells).shape)
        left_b = float(left.blocks_per_period)
        right_b = float(right.blocks_per_period)
        left_frame, right_frame = left.reading.frame, right.reading.frame
        left_axis, right_axis = left.reading.axis, right.reading.axis
        if (len(left_origin) != 2 or len(right_origin) != 2
                or len(left_shape) != 2 or len(right_shape) != 2
                or not all(math.isfinite(value) for value in
                           (*left_origin, *right_origin, left_b, right_b))
                or type(left_frame) is not str or type(right_frame) is not str
                or type(left_axis) is not int or type(right_axis) is not int):
            return None
        normalized = np.asarray(matrix, dtype=np.float64)
        scale = float(normalized[2, 2])
        if abs(scale) <= 1e-15:
            return None
        normalized = normalized / scale
        record = {
            "mapping_version": "block-cell-grid-v2",
            "mapping_direction": "right-grid-to-left-grid",
            "left": {"frame": left_frame, "plane": left_plane_index,
                     "axis": left_axis, "b": left_b,
                     "shape": list(left_shape), "origin_uv": list(left_origin)},
            "right": {"frame": right_frame, "plane": right_plane_index,
                      "axis": right_axis, "b": right_b,
                      "shape": list(right_shape), "origin_uv": list(right_origin)},
            "relative_quadrant": relative_quadrant,
            "homography": normalized.tolist(),
            "normalized_homography_sha256": normalized_digest,
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("ascii")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _signature_for(
    reading: lattice_planes.PlaneReading,
    canonical_b: float,
    signatures: Sequence[lattice_planes.PlaneSignature],
) -> tuple[lattice_planes.PlaneSignature | None, str | None]:
    matches = []
    foreign_identity = False
    for signature in signatures:
        candidate_reading = signature.reading
        if (candidate_reading is not reading
                and candidate_reading.frame == reading.frame
                and lattice_planes.reading_order_key(candidate_reading)
                == lattice_planes.reading_order_key(reading)):
            candidate_alt = candidate_reading.matching_alternatives(
                signature.blocks_per_period)
            expected_alt = reading.matching_alternatives(canonical_b)
            if (len(candidate_alt) == 1 and len(expected_alt) == 1
                    and candidate_alt[0].blocks_per_period
                    == expected_alt[0].blocks_per_period):
                foreign_identity = True
            continue
        if candidate_reading is not reading:
            continue
        selected = reading.matching_alternatives(canonical_b)
        found = reading.matching_alternatives(signature.blocks_per_period)
        if (len(selected) == 1 and len(found) == 1
                and found[0].blocks_per_period == selected[0].blocks_per_period):
            matches.append(signature)
    if foreign_identity:
        return None, "signature_identity_mismatch"
    if not matches:
        return None, "signature_missing"
    if len(matches) > 1:
        return None, "signature_ambiguous"
    return matches[0], None


def _build_graph_records(
    frames: tuple[str, ...],
    global_anchor: str,
    edge_classes: Mapping[int, Mapping[C1MatchIdentity, frozenset[str]]],
) -> tuple[tuple[AxisSupportComponent, ...], tuple[FrameAxisSupport, ...]]:
    components: list[AxisSupportComponent] = []
    frame_records: list[FrameAxisSupport] = []
    for world_axis in range(3):
        edges = edge_classes[world_axis]
        adjacency: dict[str, set[str]] = {frame: set() for frame in frames}
        for identity in edges:
            adjacency[identity.left_frame].add(identity.right_frame)
            adjacency[identity.right_frame].add(identity.left_frame)
        unseen = set(frames)
        component_for_frame: dict[str, AxisSupportComponent] = {}
        while unseen:
            start = min(unseen)
            stack = [start]
            found: set[str] = set()
            while stack:
                frame = stack.pop()
                if frame in found:
                    continue
                found.add(frame)
                stack.extend(sorted(adjacency[frame] - found, reverse=True))
            unseen -= found
            identities = tuple(sorted(
                (identity for identity in edges
                 if identity.left_frame in found and identity.right_frame in found),
                key=lambda identity: identity.sort_key()))
            normal_count = sum(
                "normal_face" in edges[identity] for identity in identities)
            lateral_count = sum(
                "lateral_shift" in edges[identity] for identity in identities)
            contains_global = global_anchor in found
            status = ("no_constraints" if not identities else
                      "supported" if contains_global else
                      "disconnected_from_global_gauge")
            component = AxisSupportComponent(
                world_axis, tuple(sorted(found)), min(found), global_anchor, identities,
                int(normal_count), int(lateral_count), contains_global, status)
            components.append(component)
            for frame in found:
                component_for_frame[frame] = component

        for frame in frames:
            component = component_for_frame[frame]
            incident = tuple(identity for identity in edges
                             if frame in (identity.left_frame, identity.right_frame))
            classes = tuple(value for value in _EVIDENCE_CLASSES
                            if any(value in edges[identity] for identity in incident))
            pair_count = len({(identity.left_frame, identity.right_frame)
                              for identity in incident})
            connected = (component.contains_global_anchor
                         and bool(component.match_identities))
            refusal = ("no_constraints" if not incident else
                       "disconnected_from_global_gauge" if not connected else None)
            frame_records.append(FrameAxisSupport(
                frame, world_axis, component.gauge_anchor, connected,
                len(incident), pair_count, classes, refusal))
    components.sort(key=lambda value: (value.world_axis, value.gauge_anchor))
    frame_records.sort(key=lambda value: (value.frame, value.world_axis))
    return tuple(components), tuple(frame_records)


def build_axis_support(
    solution: plane_graph.ComponentSolution,
    readings: Sequence[lattice_planes.PlaneReading],
    matches: Sequence[plane_graph.PlaneMatch],
    signatures: Sequence[lattice_planes.PlaneSignature],
) -> C1AxisSupport:
    """Map accepted integer shifts and expose axis-wise support components.

    ``signatures`` must be the exact signature objects used to generate
    ``matches``. C1 hashes only their homography-relevant geometry and the
    resulting normalized right-to-left cell homography; these hashes identify
    inputs for later validation but are not evidence of geometric correctness.
    """
    def refused(reason: str) -> C1AxisSupport:
        return C1AxisSupport(
            refusal=reason, frames=(), global_anchor=None,
            normal_face_constraints=(), mapped_cell_shifts=(),
            match_refusals=(), axis_components=(), frame_support=())

    if not isinstance(solution, plane_graph.ComponentSolution):
        return refused("solution_abstained")
    if solution.reason:
        return refused("solution_abstained")
    if (not isinstance(solution.frames, tuple) or not solution.frames
            or any(type(frame) is not str or not frame for frame in solution.frames)
            or len(set(solution.frames)) != len(solution.frames)
            or tuple(sorted(solution.frames)) != solution.frames):
        return refused("frame_missing")
    frames = tuple(solution.frames)
    global_anchor = frames[0]
    if (not isinstance(solution.quadrants, Mapping)
            or not isinstance(solution.translations, Mapping)
            or not isinstance(solution.blocks_per_period, Mapping)):
        return refused("frame_missing")
    try:
        solution_quadrants = dict(solution.quadrants)
        solution_translations = dict(solution.translations)
        solution_blocks = dict(solution.blocks_per_period)
    except (TypeError, ValueError, OverflowError):
        return refused("frame_missing")
    for frame in frames:
        quadrant = solution_quadrants.get(frame)
        translation = solution_translations.get(frame)
        if quadrant is None or translation is None:
            return refused("frame_missing")
        if type(quadrant) is not int or quadrant not in (0, 1, 2, 3):
            return refused("quadrant_invalid")
        if (not isinstance(translation, tuple) or len(translation) != 3
                or any(_finite(value) is None for value in translation)):
            return refused("nonfinite_support")
    if (not isinstance(readings, Sequence) or not isinstance(matches, Sequence)
            or not isinstance(signatures, Sequence)):
        return refused("reading_missing")
    if any(not isinstance(signature, lattice_planes.PlaneSignature)
           or not isinstance(getattr(signature, "reading", None),
                             lattice_planes.PlaneReading)
           for signature in signatures):
        return refused("signature_identity_mismatch")

    by_frame: dict[str, list[lattice_planes.PlaneReading]] = {
        frame: [] for frame in frames}
    for reading in readings:
        if not isinstance(reading, lattice_planes.PlaneReading):
            return refused("reading_missing")
        if (type(reading.frame) is not str or reading.frame not in by_frame
                or type(reading.axis) is not int or reading.axis not in (0, 1, 2)
                or _finite(reading.side) not in (-1.0, 1.0)):
            return refused("frame_missing")
        by_frame[reading.frame].append(reading)
    read_by_index: dict[tuple[str, int], lattice_planes.PlaneReading] = {}
    index_by_object: dict[int, tuple[str, int]] = {}
    for frame, frame_readings in by_frame.items():
        try:
            ordered = sorted(frame_readings, key=lattice_planes.reading_order_key)
            order_keys = tuple(lattice_planes.reading_order_key(value) for value in ordered)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return refused("reading_missing")
        if len({id(value) for value in ordered}) != len(ordered):
            return refused("reading_missing")
        if len(set(order_keys)) != len(order_keys):
            return refused("reading_missing")
        for index, reading in enumerate(ordered):
            if id(reading) in index_by_object:
                return refused("reading_missing")
            read_by_index[(frame, index)] = reading
            index_by_object[id(reading)] = (frame, index)

    # Deduplicate byte-identical rows before they can add topology/support.
    unique: dict[C1MatchIdentity, plane_graph.PlaneMatch] = {}
    for match in matches:
        if not isinstance(match, plane_graph.PlaneMatch):
            return refused("phase_shift_identity_mismatch")
        try:
            identity = C1MatchIdentity.from_match(match)
        except (ValueError, TypeError, OverflowError):
            if (type(match.quadrant_of_right) is not int
                    or match.quadrant_of_right not in (0, 1, 2, 3)):
                return refused("quadrant_invalid")
            return refused("phase_shift_identity_mismatch")
        previous = unique.get(identity)
        if previous is not None:
            if previous != match:
                return refused("phase_shift_identity_mismatch")
            continue
        unique[identity] = match

    edge_classes: dict[int, dict[C1MatchIdentity, set[str]]] = {
        axis: {} for axis in range(3)}
    mapped: list[MappedCellShiftAxis] = []
    match_refusals: list[MatchSupportRefusal] = []
    normal_face_constraints: list[NormalFaceConstraint] = []
    quadrant_compatible_rows = 0
    selected_rows = 0
    missing_selected_provenance = False
    for identity in sorted(unique, key=lambda value: value.sort_key()):
        match = unique[identity]
        if match.left not in solution.quadrants or match.right not in solution.quadrants:
            return refused("frame_missing")
        if (match.left not in frames or match.right not in frames
                or type(match.left_plane) is not int or type(match.right_plane) is not int):
            return refused("frame_missing")
        left_q = solution.quadrants[match.left]
        right_q = solution.quadrants[match.right]
        # Solver input includes losing q/b hypotheses. They are filtered, not
        # reported as C1 errors; only selected rows can contribute evidence.
        if ((right_q - left_q) % 4) != match.quadrant_of_right:
            continue
        quadrant_compatible_rows += 1
        left = read_by_index.get((match.left, match.left_plane))
        right = read_by_index.get((match.right, match.right_plane))
        if left is None or right is None:
            return refused("reading_missing")
        if match.left_axis != left.axis or match.right_axis != right.axis:
            return refused("phase_shift_identity_mismatch")
        try:
            left_normal_world, left_normal_sign = plane_graph.axis_transform(
                left.axis, left_q)
            right_normal_world, right_normal_sign = plane_graph.axis_transform(
                right.axis, right_q)
        except ValueError:
            return refused("axis_transform_invalid")
        if left_normal_world != right_normal_world:
            return refused("axis_transform_invalid")

        left_key = (match.left, match.left_plane)
        right_key = (match.right, match.right_plane)
        if left_key not in solution_blocks or right_key not in solution_blocks:
            missing_selected_provenance = True
            continue
        left_selected = solution_blocks[left_key]
        right_selected = solution_blocks[right_key]
        left_alt, left_alt_status = _selected_alternative(
            left, left_selected, match.left_b)
        right_alt, right_alt_status = _selected_alternative(
            right, right_selected, match.right_b)
        if "invalid_selection" in {left_alt_status, right_alt_status}:
            return refused("selected_provenance_missing")
        if left_alt is None or right_alt is None:
            continue
        selected_rows += 1
        left_distance = _finite(left_alt.distance_blocks)
        right_distance = _finite(right_alt.distance_blocks)
        left_side = _finite(left.side)
        right_side = _finite(right.side)
        if (left_distance is None or right_distance is None
                or left_side is None or right_side is None):
            return refused("nonfinite_support")
        normal_delta = (left_normal_sign * left_side * left_distance
                        - right_normal_sign * right_side * right_distance)
        if not math.isfinite(normal_delta):
            return refused("nonfinite_support")
        normal_face_constraints.append(NormalFaceConstraint(
            identity, left_q, right_q, left_normal_world,
            left_normal_sign, right_normal_sign,
            left_side, right_side, left_distance, right_distance, normal_delta))
        edge_classes[left_normal_world].setdefault(identity, set()).add("normal_face")

        try:
            left_signature, left_sig_refusal = _signature_for(
                left, float(left_alt.blocks_per_period), signatures)
            right_signature, right_sig_refusal = _signature_for(
                right, float(right_alt.blocks_per_period), signatures)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return refused("signature_identity_mismatch")
        if left_sig_refusal is not None:
            return refused(left_sig_refusal)
        if right_sig_refusal is not None:
            return refused(right_sig_refusal)
        assert left_signature is not None and right_signature is not None
        left_geometry = _signature_geometry_digest(left_signature, match.left_plane)
        right_geometry = _signature_geometry_digest(right_signature, match.right_plane)
        if left_geometry is None or right_geometry is None:
            return refused("homography_invalid")
        try:
            cell_homography = plane_graph.cell_grid_homography(
                left_signature, right_signature, match.quadrant_of_right)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return refused("homography_invalid")
        homography_digest = _grid_homography_digest(
            left_signature, right_signature, match.quadrant_of_right,
            match.left_plane, match.right_plane, cell_homography)
        if homography_digest is None:
            return refused("homography_invalid")

        evidence = plane_residue.hypothesis_phase_shift_evidence(
            match, {(match.left, match.left_plane): left,
                    (match.right, match.right_plane): right})
        if evidence.refusal is not None:
            return refused("phase_shift_identity_mismatch")
        support = evidence.relative_shift
        if _support_identity(support) != identity:
            return refused("phase_shift_identity_mismatch")
        if support.refusal is not None:
            reason = ("cell_shift_absent" if support.refusal == "cell_shift_absent"
                      else "nonfinite_support")
            match_refusals.append(MatchSupportRefusal(
                identity, reason, support))
            continue
        if support.cell_shift_rc is None:
            match_refusals.append(MatchSupportRefusal(
                identity, "cell_shift_absent", support))
            continue

        left_tangent_labels = tuple(label for label in range(3)
                                    if label != left.axis)
        right_tangent_labels = tuple(label for label in range(3)
                                     if label != right.axis)
        left_lateral = {plane_graph.axis_transform(label, left_q)[0]
                        for label in left_tangent_labels}
        right_lateral = {
            plane_graph.axis_transform(label, right_q)[0]
            for label in right_tangent_labels}
        right_mapped = (
            ("u", right_tangent_labels[0], "column", support.cell_shift_rc[1]),
            ("v", right_tangent_labels[1], "row", support.cell_shift_rc[0]),
        )
        right_lateral_ordered = tuple(
            plane_graph.axis_transform(label, right_q)[0]
            for label in right_tangent_labels)
        if (len(set(right_lateral_ordered)) != 2
                or right_normal_world in right_lateral
                or left_lateral != right_lateral):
            return refused("axis_transform_invalid")
        for local_axis, local_label, component, grid_shift in right_mapped:
            world_axis, world_sign = plane_graph.axis_transform(local_label, right_q)
            mapped.append(MappedCellShiftAxis(
                identity, left_q, right_q, local_axis, local_label, component,
                grid_shift, world_axis, world_sign, world_sign * grid_shift,
                left_geometry, right_geometry, homography_digest, support))
            edge_classes[world_axis].setdefault(identity, set()).add("lateral_shift")

    immutable_edges = {
        axis: {identity: frozenset(classes) for identity, classes in values.items()}
        for axis, values in edge_classes.items()}
    components, frame_support = _build_graph_records(
        frames, global_anchor, immutable_edges)
    if selected_rows == 0:
        reason = ("no_constraints" if not unique else
                  "relative_quadrant_mismatch" if quadrant_compatible_rows == 0 else
                  "selected_provenance_missing" if missing_selected_provenance else
                  "alternative_not_selected")
        return refused(reason)
    normal_face_constraints.sort(key=lambda value: value.match_identity.sort_key())
    mapped.sort(key=lambda value: (
        value.match_identity.sort_key(), value.world_axis, value.right_local_axis))
    match_refusals.sort(key=lambda value: (
        value.match_identity.sort_key(), value.refusal))
    return C1AxisSupport(
        refusal=None, frames=frames, global_anchor=global_anchor,
        normal_face_constraints=tuple(normal_face_constraints),
        mapped_cell_shifts=tuple(mapped), match_refusals=tuple(match_refusals),
        axis_components=components, frame_support=frame_support)
