"""Compose the accepted image-derived geometry seams into PlacementUniverse."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_axis_support as h_mod
from mc_photogrammetry.pipeline import image_derived_endpoint_provenance as o2_mod
from mc_photogrammetry.pipeline import image_derived_joint_graph as g_mod
from mc_photogrammetry.pipeline import image_derived_graph_policy_v5 as j_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
from mc_photogrammetry.pipeline import image_derived_phase_anchor as p2_mod
from mc_photogrammetry.pipeline import image_derived_phase_evidence as l_mod
from mc_photogrammetry.pipeline import image_derived_placement as d_mod
from mc_photogrammetry.pipeline import image_derived_region_support as q2_mod
from mc_photogrammetry.pipeline import image_derived_relative_translations as n2_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.pipeline import image_derived_shift_domains as k_mod
from mc_photogrammetry.pipeline import image_derived_signatures as f1_mod
from mc_photogrammetry.pipeline import image_derived_vector_domains as m2_mod
from mc_photogrammetry.reconstruction import plane_placement as placement

__all__ = ["IMAGE_DERIVED_GEOMETRY_SCHEMA", "IMAGE_DERIVED_GEOMETRY_PRODUCER",
           "ImageDerivedGeometryResult", "derive_image_derived_geometry"]

IMAGE_DERIVED_GEOMETRY_SCHEMA = "mc-photogrammetry-image-derived-geometry"
IMAGE_DERIVED_GEOMETRY_PRODUCER = "m9-r1-image-derived-geometry-2"
IMAGE_DERIVED_GEOMETRY_TERMINAL = "geometry_unavailable"
_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)
_STAGES = ("f1", "g", "h", "j", "k", "l", "m2", "n2", "o2", "p2", "q2")
_READY = {"f1": "assembled", "g": "assembled", "h": "assembled",
          "j": "bound", "k": "complete", "l": "complete",
          "m2": "complete", "n2": "complete", "o2": "complete",
          "p2": "determined", "q2": "assembled"}
_STAGE_TYPES = dict(zip(_STAGES, (f1_mod.ImageDerivedSignaturesResult,
    g_mod.ImageDerivedGraphResult, h_mod.ImageDerivedAxisSupportResult,
    j_mod.ImageDerivedPolicyBoundGraph, k_mod.ImageDerivedShiftDomainsResult,
    l_mod.ImageDerivedPhaseEvidenceResult, m2_mod.ImageDerivedVectorDomainsResult,
    n2_mod.ImageDerivedRelativeTranslationsResult, o2_mod.ImageDerivedEndpointProvenanceResult,
    p2_mod.ImageDerivedPhaseAnchorResult, q2_mod.ImageDerivedRegionSupportResult)))


def _joint_lineage(records):
    graph = dict(records).get("g")
    if graph is None:
        return ()
    if type(graph) is not g_mod.ImageDerivedGraphResult:
        raise ValueError("geometry contains a stale graph")
    cert = graph.selection
    return (graph.schema, graph.producer, graph.result_sha256,
            cert.schema, cert.producer, cert.result_sha256)


def _sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _digest(record: object) -> str:
    value = getattr(record, "result_sha256", None)
    if not _hex(value):
        raise ValueError("stage record lacks a result digest")
    return value


def _replay(record: object) -> object:
    if not dataclasses.is_dataclass(record) or isinstance(record, type):
        raise ValueError("stage record is foreign")
    return type(record)(**{field.name: getattr(record, field.name)
                           for field in dataclasses.fields(record)})


def _payload(value: "ImageDerivedGeometryResult") -> dict:
    return {
        "schema": value.schema, "producer": value.producer,
        "terminal": value.terminal, "status": value.status,
        "reason": value.reason, "detail": value.detail,
        "session_sha256": value.session_sha256,
        "lattice_sha256": value.lattice_sha256,
        "stage_digests": [[name, digest] for name, digest in value.stage_digests],
        "placement_keys": [name for name, _record in value.placement_records],
        "universe_sha256": ("" if value.universe is None
                            else value.universe.universe_sha256),
        "support_sha256": [item.support_sha256 for item in value.supports],
        "joint_lineage": value.joint_lineage,
    }


def _placement_mapping(records: dict[str, object]) -> tuple[tuple[str, object], ...]:
    f1, g, h, k, l = (records[name] for name in ("f1", "g", "h", "k", "l"))
    m2, n2, o2, p2, q2 = (
        records[name] for name in ("m2", "n2", "o2", "p2", "q2"))
    lattice = p2.lattice_result
    return (
        ("axis_support", h.axis_support),
        ("p1d_result", k.shift_domain_result),
        ("vector_result", m2.vector_domain_result),
        ("c2rb_result", n2.relative_translation_result),
        ("v2a_result", o2.endpoint_provenance_result),
        ("v2b_result", p2.phase_anchor_result),
        ("readings", g.readings),
        ("signatures", tuple(item.signature for item in f1.signatures)),
        ("matches", g.matches),
        ("phase_evidence", l.phase_evidence),
        ("usable_masks", tuple(item.mask for item in f1.usable_masks)),
        ("split_sources", lattice.bridge.split_sources_for_assembly()),
        ("anchor_budget", p2.anchor_policy.budget),
        ("support_budget", q2.policy.budget),
        ("comparison_tolerance", n2.relative_policy.comparison_tolerance),
        ("closure_tolerance", n2.relative_policy.closure_tolerance),
    )


def _same_value(actual: object, expected: object) -> bool:
    if isinstance(expected, tuple):
        return (isinstance(actual, tuple) and len(actual) == len(expected)
                and all(left is right for left, right in zip(actual, expected)))
    if type(expected) in (float, int, str):
        return type(actual) is type(expected) and actual == expected
    return actual is expected


@dataclass(frozen=True, slots=True)
class ImageDerivedGeometryResult:
    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    session_sha256: str
    lattice_sha256: str
    stage_digests: tuple[tuple[str, str], ...]
    records: tuple[tuple[str, object], ...]
    placement_records: tuple[tuple[str, object], ...]
    universe: placement.PlacementUniverse | None
    supports: tuple
    result_sha256: str = ""

    @property
    def joint_lineage(self):
        return _joint_lineage(self.records)

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_GEOMETRY_SCHEMA
                or self.producer != IMAGE_DERIVED_GEOMETRY_PRODUCER
                or self.terminal != IMAGE_DERIVED_GEOMETRY_TERMINAL
                or self.status not in ("assembled", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not _hex(self.session_sha256) or not _hex(self.lattice_sha256)
                or not isinstance(self.records, tuple)
                or not isinstance(self.stage_digests, tuple)
                or not isinstance(self.placement_records, tuple)
                or not isinstance(self.supports, tuple)):
            raise ValueError("image-derived geometry result is malformed")
        names = tuple(name for name, _record in self.records)
        if names != _STAGES[:len(names)] or len(set(names)) != len(names):
            raise ValueError("geometry stages are not a canonical prefix")
        if any(type(record) is not _STAGE_TYPES[name] for name, record in self.records):
            raise ValueError("geometry contains an obsolete or foreign stage")
        replayed = tuple((name, _replay(record)) for name, record in self.records)
        expected = tuple((name, _digest(record)) for name, record in replayed)
        if self.stage_digests != expected:
            raise ValueError("geometry stage digests do not bind")
        if self.status == "assembled":
            if (self.reason or self.detail or names != _STAGES
                    or not isinstance(self.universe, placement.PlacementUniverse)
                    or not self.placement_records or not self.supports
                    or tuple(name for name, _ in self.placement_records)
                    != d_mod.REQUIRED_PLACEMENT_KEYWORDS):
                raise ValueError("assembled image-derived geometry is inconsistent")
            replayed_universe = placement.PlacementUniverse(**{
                field.name: getattr(self.universe, field.name)
                for field in dataclasses.fields(placement.PlacementUniverse)})
            if replayed_universe.universe_sha256 != self.universe.universe_sha256:
                raise ValueError("placement universe does not replay")
            expected_mapping = _placement_mapping(dict(replayed))
            if (tuple(name for name, _ in expected_mapping)
                    != tuple(name for name, _ in self.placement_records)
                    or any(not _same_value(actual, expected)
                           for (_name, actual), (_other, expected) in zip(
                               self.placement_records, expected_mapping))):
                raise ValueError("placement records do not bind the stages")
            q2 = dict(replayed)["q2"]
            if (len(self.supports) != len(q2.support_evidence)
                    or any(actual is not evidence.support
                           for actual, evidence in zip(
                               self.supports, q2.support_evidence))):
                raise ValueError("region supports do not bind Q2")
        elif (not self.reason or self.universe is not None
              or self.placement_records or self.supports):
            raise ValueError("unavailable image-derived geometry must be atomic")
        digest = _sha(_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived geometry digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, error: object = ""):
    detail = {"field": field}
    if error:
        detail["detail"] = str(error)
    return placement._make_refusal(reason, detail)


def _make(session_sha, lattice_sha, records, *, status, reason="", detail="",
          placement_records=(), universe=None, supports=()):
    return ImageDerivedGeometryResult(
        IMAGE_DERIVED_GEOMETRY_SCHEMA, IMAGE_DERIVED_GEOMETRY_PRODUCER,
        IMAGE_DERIVED_GEOMETRY_TERMINAL, status, reason, detail,
        session_sha, lattice_sha,
        tuple((name, _digest(record)) for name, record in records),
        tuple(records), tuple(placement_records), universe, tuple(supports))


def _stop(session_sha, lattice_sha, records, stage, record):
    status = getattr(record, "status", "unavailable")
    return _make(session_sha, lattice_sha, records, status="unavailable",
                 reason=stage + "_" + status,
                 detail=str(getattr(record, "reason", "")))


def _completed_refusal(stage, records, refusal, sink):
    if sink is not None:
        sink(stage, tuple(records), refusal)
    return refusal


def _assemble_geometry_placement(session_sha, lattice, records, mapping, sink):
    if tuple(name for name, _ in mapping) != d_mod.REQUIRED_PLACEMENT_KEYWORDS:
        return _completed_refusal("placement", records,
            _refusal("provenance_mismatch", "placement_records.signature"), sink)
    universe = d_mod.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=dict(mapping))
    if isinstance(universe, placement.PlacementRefusal):
        return _completed_refusal("placement", records, universe, sink)
    if isinstance(universe, d_mod.ImageDerivedPlacementUnavailable):
        return _make(session_sha, lattice.result_sha256, records, status="unavailable",
                     reason="placement_" + universe.reason, detail=universe.detail)
    if not isinstance(universe, placement.PlacementUniverse):
        return _completed_refusal("placement", records,
            _refusal("provenance_mismatch", "placement_result"), sink)
    supports = tuple(item.support for item in dict(records)["q2"].support_evidence)
    return _make(session_sha, lattice.result_sha256, records, status="assembled",
                 placement_records=mapping, universe=universe, supports=supports)


def _derive_image_derived_geometry_with_context(
    *, session: object, lattice_result: object, connection_graph: object = None,
    _completed_stage_sink=None,
) -> ImageDerivedGeometryResult | placement.PlacementRefusal:
    """Run the accepted M9 seams once each and assemble their exact universe."""
    replayed_session = f1_mod._replay_session(session)
    if isinstance(replayed_session, placement.PlacementRefusal):
        return replayed_session
    checked_session, _frames = replayed_session
    if not isinstance(lattice_result, lattice_mod.ImageDerivedLatticeResult):
        return _refusal("invalid_input", "lattice_result")
    try:
        lattice = lattice_mod.ImageDerivedLatticeResult(**{
            field.name: getattr(lattice_result, field.name)
            for field in dataclasses.fields(lattice_mod.ImageDerivedLatticeResult)})
    except _ERRORS as error:
        return _refusal("provenance_mismatch", "lattice_result",
                        type(error).__name__)
    if lattice.session_sha256 != checked_session.session_sha256:
        return _refusal("provenance_mismatch", "session.lattice")
    records = []

    def accept(stage, record):
        if isinstance(record, placement.PlacementRefusal):
            return _completed_refusal(stage, records, record, _completed_stage_sink)
        try:
            if type(record) is not _STAGE_TYPES[stage]:
                raise ValueError("geometry stage is obsolete or foreign")
            replayed = _replay(record)
            if stage == "g":
                g_mod._require_live(record)
                replayed = record  # retain the exact validated live graph capability
        except _ERRORS as error:
            return _completed_refusal(stage, records,
                _refusal("provenance_mismatch", stage, type(error).__name__), _completed_stage_sink)
        records.append((stage, replayed))
        if replayed.status != _READY[stage]:
            return _stop(checked_session.session_sha256, lattice.result_sha256,
                         records, stage, replayed)
        if stage == "n2" and replayed.placement_readiness != "ready":
            return _stop(checked_session.session_sha256, lattice.result_sha256,
                         records, stage, replayed)
        return None

    f1 = f1_mod.derive_image_derived_signatures(
        session=session, lattice_result=lattice)
    stopped = accept("f1", f1)
    if stopped is not None: return stopped
    g = g_mod.derive_image_derived_graph(
        session=checked_session, lattice_result=lattice, signatures_result=f1,
        connection_graph=connection_graph,
        max_shift_cells=32, integer_tolerance=0.15,
        min_resultant=0.9, ambiguous_margin=0.02)
    stopped = accept("g", g)
    if stopped is not None: return stopped
    h = h_mod.derive_image_derived_axis_support(graph_result=g)
    stopped = accept("h", h)
    if stopped is not None: return stopped
    j = j_mod.bind_image_derived_graph_policy(graph_result=g)
    stopped = accept("j", j)
    if stopped is not None: return stopped
    k = k_mod.derive_image_derived_shift_domains(
        axis_support_result=h, policy_graph_result=j)
    stopped = accept("k", k)
    if stopped is not None: return stopped
    l = l_mod.derive_image_derived_phase_evidence(policy_graph_result=j)
    stopped = accept("l", l)
    if stopped is not None: return stopped
    m2 = m2_mod.derive_image_derived_vector_domains(
        shift_domains_result=k, phase_evidence_result=l)
    stopped = accept("m2", m2)
    if stopped is not None: return stopped
    n2 = n2_mod.derive_image_derived_relative_translations(
        vector_domains_result=m2)
    stopped = accept("n2", n2)
    if stopped is not None: return stopped
    o2 = o2_mod.derive_image_derived_endpoint_provenance(
        relative_translations_result=n2)
    stopped = accept("o2", o2)
    if stopped is not None: return stopped
    p2 = p2_mod.derive_image_derived_phase_anchor(
        endpoint_provenance_result=o2, lattice_result=lattice)
    stopped = accept("p2", p2)
    if stopped is not None: return stopped
    q2 = q2_mod.derive_image_derived_region_support(
        phase_anchor_result=p2, signatures_result=f1, session=session)
    stopped = accept("q2", q2)
    if stopped is not None: return stopped

    mapping = _placement_mapping(dict(records))
    return _assemble_geometry_placement(checked_session.session_sha256, lattice,
                                       records, mapping, _completed_stage_sink)


def derive_image_derived_geometry(*, session, lattice_result, connection_graph=None):
    """Public geometry inputs; completed refusal context remains internal."""
    return _derive_image_derived_geometry_with_context(
        session=session, lattice_result=lattice_result, connection_graph=connection_graph)
