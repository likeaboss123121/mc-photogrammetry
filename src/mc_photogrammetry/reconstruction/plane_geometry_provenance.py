"""M4 in-memory geometry provenance bridge (narrow production seam).

Retains frozen-schema-v1-dropped, downstream-needed evidence: the rectified
per-segment **structural weights patch** the lattice measurement discards (with
its exact ``structural_share`` and
:class:`~lattice_planes.PeriodicEnergyPhaseEvidence`), and the exact P0
**split-source provenance** (accepted
:class:`~lattice_planes.SplitPhaseProducerResult` /
:class:`~lattice_planes.SplitPhaseProducerBudget` plus the held-out patch).
Both records are immutable explicit-array carriers that seal exact digests, and
:func:`bind_geometry_provenance` replays every carried record into one sealed
:class:`GeometryProvenanceBridge` or one typed
:class:`~plane_placement.PlacementRefusal` (never a partial bridge).
In-memory only: no image, no measurement, no producer, no synthetic default, no
IO, no manifest/F3/coordinate/direction/bound/reference, no CLI/schema/M6.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_placement as placement

__all__ = [
    "GEOMETRY_PROVENANCE_BRIDGE_VERSION", "GEOMETRY_PROVENANCE_REFUSAL_REASONS",
    "StructuralWeightsEvidence", "SplitSourceEvidence",
    "GeometryProvenanceBridge", "bind_geometry_provenance",
]

GEOMETRY_PROVENANCE_BRIDGE_VERSION = "m4-geometry-provenance-bridge-1"
#: Closed refusal vocabulary; a subset of
#: :data:`plane_placement.V2_PLACEMENT_REFUSAL_REASONS`.
GEOMETRY_PROVENANCE_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input", "missing_provenance", "provenance_mismatch", "replay_mismatch",
)

_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)

class _BridgeRefusal(Exception):
    """Internal control flow: one typed bridge refusal and its JSON detail."""

    def __init__(self, reason: str, detail: object = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = {} if detail is None else detail

def _refuse(reason: str, field: str) -> None:
    raise _BridgeRefusal(reason, {"field": field})

def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))

def _tuple_of(value: object, kind: type) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, kind) for item in value)

def _identity_key(item) -> tuple:
    return (item.frame, item.canonical_index, item.reading_axis)

def _structural_patch_sha256(rows: int, columns: int, raw: bytes) -> str:
    header = f"m4-structural-weights-patch-1:{rows}:{columns}:{len(raw)}".encode("ascii")
    return hashlib.sha256(header + raw).hexdigest()

def _structural_array(evidence: "StructuralWeightsEvidence") -> np.ndarray:
    return np.frombuffer(evidence.patch_float32_le, dtype="<f4").reshape(
        evidence.patch_rows, evidence.patch_columns)

def _structural_geometry_ok(evidence) -> bool:
    """True only for the exact square rectified sampling grid the reading used."""
    rows, columns, samples = (evidence.patch_rows, evidence.patch_columns,
                              evidence.samples)
    return (type(rows) is int and type(columns) is int and type(samples) is int
            and rows > 0 and rows == columns == samples)

def _structural_payload(evidence) -> list:
    return [evidence.frame, evidence.canonical_index, evidence.reading_axis,
            evidence.source_region_id, evidence.patch_rows, evidence.patch_columns,
            evidence.patch_sha256, float(evidence.structural_share),
            float(evidence.period_pixels), float(evidence.extent),
            int(evidence.samples), [float(value) for value in evidence.centre_uv],
            asdict(evidence.phase_evidence)]

def _split_payload(source) -> list:
    return [source.frame, source.canonical_index, source.reading_axis,
            source.result.result_sha256, source.budget.budget_sha256,
            source.patch_rows, source.patch_columns]

@dataclass(frozen=True, slots=True)
class StructuralWeightsEvidence:
    """One reading's exact retained rectified structural-weight patch."""

    frame: str
    canonical_index: int
    reading_axis: int
    source_region_id: str | None
    patch_rows: int
    patch_columns: int
    patch_float32_le: bytes
    structural_share: float
    period_pixels: float
    extent: float
    samples: int
    centre_uv: tuple[float, float]
    phase_evidence: lattice_planes.PeriodicEnergyPhaseEvidence
    patch_sha256: str = ""
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.patch_float32_le) is not bytes:
            raise ValueError("structural weights patch must be explicit bytes")
        if not _structural_geometry_ok(self):
            raise ValueError("structural weights patch geometry mismatch")
        digest = _structural_patch_sha256(self.patch_rows, self.patch_columns,
                                          self.patch_float32_le)
        if self.patch_sha256 not in ("", digest):
            raise ValueError("structural weights patch digest mismatch")
        object.__setattr__(self, "patch_sha256", digest)
        evidence = placement._sha256(_structural_payload(self))
        if self.evidence_sha256 not in ("", evidence):
            raise ValueError("structural weights evidence digest mismatch")
        object.__setattr__(self, "evidence_sha256", evidence)

def _replay_structural_weights(evidence: object) -> StructuralWeightsEvidence:
    if not isinstance(evidence, StructuralWeightsEvidence):
        _refuse("invalid_input", "structural_weights.entry")
    if not _structural_geometry_ok(evidence):
        _refuse("provenance_mismatch", "structural_weights.patch_geometry")
    array = _structural_array(evidence)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        _refuse("invalid_input", "structural_weights.patch")
    if _structural_patch_sha256(evidence.patch_rows, evidence.patch_columns,
                                evidence.patch_float32_le) != evidence.patch_sha256:
        _refuse("provenance_mismatch", "structural_weights.patch_sha256")
    share = lattice_planes._structural_share(array, evidence.period_pixels)
    if float(share) != float(evidence.structural_share):
        _refuse("provenance_mismatch", "structural_weights.structural_share")
    phase = lattice_planes._periodic_energy_phase_evidence(
        array, evidence.period_pixels, centre=evidence.centre_uv,
        extent=evidence.extent, samples=evidence.samples,
        source_region_id=evidence.source_region_id)
    if phase != evidence.phase_evidence:
        _refuse("provenance_mismatch", "structural_weights.phase_evidence")
    if placement._sha256(_structural_payload(evidence)) != evidence.evidence_sha256:
        _refuse("provenance_mismatch", "structural_weights.evidence_sha256")
    return evidence

@dataclass(frozen=True, slots=True)
class SplitSourceEvidence:
    """One reading's exact retained P0 held-out split source."""

    frame: str
    canonical_index: int
    reading_axis: int
    result: lattice_planes.SplitPhaseProducerResult
    budget: lattice_planes.SplitPhaseProducerBudget
    patch_float64_le: bytes
    patch_rows: int
    patch_columns: int
    source_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.patch_float64_le) is not bytes:
            raise ValueError("split-source patch must be explicit bytes")
        digest = placement._sha256(_split_payload(self))
        if self.source_sha256 not in ("", digest):
            raise ValueError("split-source digest mismatch")
        object.__setattr__(self, "source_sha256", digest)

    def to_source(self) -> tuple:
        return (self.result, np.frombuffer(self.patch_float64_le, dtype="<f8").reshape(
            self.patch_rows, self.patch_columns), self.budget)

def _split_patch_replays(result, raw: bytes, rows: int, columns: int, budget) -> bool:
    try:
        array = np.ascontiguousarray(
            np.frombuffer(raw, dtype="<f8").reshape(rows, columns))
        return bool(lattice_planes.validate_split_phase_evidence(
            result, array, geometry=result.geometry,
            partition_config=lattice_planes.SPLIT_PHASE_PARTITION_CONFIG,
            budget=budget))
    except _ERRORS:
        return False

def _replay_split_budget(budget: object) -> None:
    """Rebuild a carried split budget from its exact record before any use."""
    if not isinstance(budget, lattice_planes.SplitPhaseProducerBudget):
        _refuse("invalid_input", "split_sources.budget")
    try:
        record = budget.to_record()
        exact = lattice_planes.SplitPhaseProducerBudget(**record).to_record() == record
    except _ERRORS:
        exact = False
    if not exact:
        _refuse("provenance_mismatch", "split_sources.budget")

def _replay_split_source(source: object) -> SplitSourceEvidence:
    if not isinstance(source, SplitSourceEvidence):
        _refuse("invalid_input", "split_sources.entry")
    _replay_split_budget(source.budget)
    identity = getattr(source.result, "reading_identity", None)
    if identity != {"frame": source.frame,
                    "plane_index": source.canonical_index,
                    "reading_axis": source.reading_axis}:
        _refuse("replay_mismatch", "split_sources.result")
    if not _split_patch_replays(source.result, source.patch_float64_le,
                                source.patch_rows, source.patch_columns,
                                source.budget):
        _refuse("replay_mismatch", "split_sources.patch")
    if placement._sha256(_split_payload(source)) != source.source_sha256:
        _refuse("provenance_mismatch", "split_sources.source_sha256")
    return source

def _bridge_payload(*, universe: str, readings: tuple, structural_weights: tuple,
                    split_sources: tuple) -> list:
    return [universe,
            [[reading.frame, int(reading.axis),
              placement._sha256(reading.region_evidence.to_record()),
              placement._sha256(asdict(reading.phase_evidence))]
             for reading in readings],
            [item.evidence_sha256 for item in structural_weights],
            [item.source_sha256 for item in split_sources]]

@dataclass(frozen=True, slots=True)
class GeometryProvenanceBridge:
    """One sealed in-memory provenance bundle over an accepted reading universe."""

    reading_universe_sha256: str
    readings: tuple
    structural_weights: tuple
    split_sources: tuple
    bridge_sha256: str = ""

    def __post_init__(self) -> None:
        if (not _hex64(self.reading_universe_sha256) or not self.readings
                or not _tuple_of(self.readings, lattice_planes.PlaneReading)
                or not _tuple_of(self.structural_weights, StructuralWeightsEvidence)
                or not _tuple_of(self.split_sources, SplitSourceEvidence)):
            raise ValueError("geometry provenance bridge fields are malformed")
        digest = placement._sha256(_bridge_payload(
            universe=self.reading_universe_sha256, readings=self.readings,
            structural_weights=self.structural_weights,
            split_sources=self.split_sources))
        if self.bridge_sha256 not in ("", digest):
            raise ValueError("geometry provenance bridge digest mismatch")
        object.__setattr__(self, "bridge_sha256", digest)

    def split_sources_for_assembly(self) -> tuple:
        """Return the exact accepted V2-B split-source tuples, keyed in order."""
        return tuple(item.to_source() for item in self.split_sources)

def _make_bridge(*, readings: tuple, structural_weights: tuple,
                 split_sources: tuple) -> GeometryProvenanceBridge:
    return GeometryProvenanceBridge(
        reading_universe_sha256=v2a._canonical_universe_sha256(readings),
        readings=readings, structural_weights=structural_weights,
        split_sources=split_sources)

def _reading_index(readings: object) -> dict:
    if (not isinstance(readings, tuple) or not readings
            or any(not isinstance(reading, lattice_planes.PlaneReading)
                   for reading in readings)):
        _refuse("invalid_input", "readings")
    try:
        index = placement._readings_index(readings)
    except placement._InputRefusal as refusal:
        raise _BridgeRefusal(refusal.reason, refusal.detail) from refusal
    if (len(index) != len(readings)
            or len({id(reading) for reading in readings}) != len(readings)):
        _refuse("invalid_input", "readings.duplicate_identity")
    return index

def _bind_structural_weights_to_reading(evidence, reading) -> None:
    if (evidence.frame != reading.frame
            or evidence.reading_axis != int(reading.axis)
            or tuple(evidence.centre_uv) != tuple(
                float(value) for value in reading.rectification.centre)
            or float(evidence.period_pixels) != float(reading.period_pixels)
            or float(evidence.extent) != float(reading.extent)
            or int(evidence.samples) != int(reading.samples)
            or evidence.source_region_id != reading.region_evidence.source_region_id
            or float(evidence.structural_share) != float(reading.structural_share)
            or evidence.phase_evidence != reading.phase_evidence):
        _refuse("provenance_mismatch", "structural_weights.reading_binding")

def _bind_split_source_to_reading(source, reading) -> None:
    if (source.result.generation_evidence[0].source_region_id
            != reading.region_evidence.source_region_id):
        _refuse("provenance_mismatch", "split_sources.source_region_id")

def _replay_reading(reading) -> None:
    if (not isinstance(reading.region_evidence, lattice_planes.PlaneRegionEvidence)
            or not isinstance(reading.phase_evidence,
                              lattice_planes.PeriodicEnergyPhaseEvidence)):
        _refuse("missing_provenance", "readings.region_phase_evidence")
    try:
        rebuilt = replace(reading)
    except _ERRORS as error:
        raise _BridgeRefusal(
            "replay_mismatch", {"field": "readings",
                                "detail": type(error).__name__}) from error
    if rebuilt != reading:
        _refuse("replay_mismatch", "readings")

def _bind_records(records: tuple, index: dict, field: str, replay, cross) -> None:
    keys: set[tuple] = set()
    for record in records:
        replay(record)
        key = _identity_key(record)
        if key in keys:
            _refuse("provenance_mismatch", field + ".duplicate")
        keys.add(key)
        reading = index.get(key)
        if reading is None:
            _refuse("missing_provenance", field + ".reading")
        if cross is not None:
            cross(record, reading)
    if keys != set(index):
        _refuse("missing_provenance", field + ".coverage")

def bind_geometry_provenance(
    *, readings: tuple, structural_weights: tuple, split_sources: tuple,
) -> GeometryProvenanceBridge | placement.PlacementRefusal:
    """Replay, cross-bind and seal every carried record; typed refusal, no partial bridge."""
    try:
        index = _reading_index(readings)
        for reading in readings:
            _replay_reading(reading)
        if not structural_weights or not split_sources:
            _refuse("invalid_input", "provenance")
        _bind_records(structural_weights, index, "structural_weights",
                      _replay_structural_weights, _bind_structural_weights_to_reading)
        _bind_records(split_sources, index, "split_sources",
                      _replay_split_source, _bind_split_source_to_reading)
        return _make_bridge(
            readings=readings,
            structural_weights=tuple(sorted(structural_weights, key=_identity_key)),
            split_sources=tuple(sorted(split_sources, key=_identity_key)))
    except _BridgeRefusal as refusal:
        return placement._make_refusal(refusal.reason, refusal.detail)
    except placement._InputRefusal as refusal:
        return placement._make_refusal(refusal.reason, refusal.detail)
    except _ERRORS as error:
        return placement._make_refusal(
            "invalid_input", {"detail": type(error).__name__})
