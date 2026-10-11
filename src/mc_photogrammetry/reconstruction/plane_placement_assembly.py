"""M4 validating production :class:`PlacementUniverse` assembler.

This module is the narrow assembly seam that turns the already-accepted,
already-produced **in-memory** M4 records of one genuine mixed-normal reading
universe into exactly one
:class:`~mc_photogrammetry.reconstruction.plane_placement.PlacementUniverse`, or
one typed, atomic
:class:`~mc_photogrammetry.reconstruction.plane_placement.PlacementRefusal`.

It is a validating facade, not a producer.  It never re-solves a match, a shift
domain, a lateral vector domain, a C2-R-B translation class, a V2-A endpoint
assignment or a V2-B phase anchor; it never invents, transcribes, reseals or
reinterprets a record.  It re-runs the accepted replay validators over the
supplied records and refuses any contradiction before a universe exists.

What it consumes
----------------

Every input is an already-accepted record of an accepted seam, plus the exact
evidence tuple that seam was built from:

* ``axis_support`` -- the accepted C1 (P1) axis support;
* ``p1d_result`` -- the accepted P1-D complete shift-domain result;
* ``vector_result`` -- the accepted C2-R vector-domain result;
* ``c2rb_result`` -- the accepted C2-R-B relative-translation result;
* ``v2a_result`` -- the accepted V2-A endpoint-provenance result;
* ``v2b_result`` -- the accepted V2-B phase-anchor result;
* ``readings`` / ``signatures`` / ``matches`` / ``phase_evidence`` -- the
  canonical reading universe and its exact bound evidence;
* ``usable_masks`` -- one ``(reading_identity_sha256, mask)`` entry per accepted
  V2-A reading domain, in the caller's original usable-mask form;
* ``split_sources`` -- the original P0 held-out split sources the V2-B anchor was
  determined from;
* ``anchor_budget`` / ``support_budget`` -- the explicit accepted budgets;
* ``comparison_tolerance`` / ``closure_tolerance`` -- the explicit accepted
  replay tolerances.

Validation discipline
---------------------

Validation is ordered and atomic.  A malformed type, a missing reading identity,
an unbound digest, a foreign budget, an unresolved anchor axis, a non-ready
C2-R-B decision, an unsupported required frame/world axis or a replay
contradiction returns one :class:`PlacementRefusal` and never a partial or
repaired universe:

* every record is type-checked and the closed C2-R-B / V2-A / V2-B gates are
  re-applied (one connected, exhaustive, cycle-validated ``unique_validated``
  component; ``endpoint_unique``; all three world axes *determined*);
* the canonical reading-universe digest is recomputed from ``readings`` and
  bound to the vector, V2-A and V2-B records; the accepted cross-record source
  digests (C2-R-B -> vector, V2-A -> C2-R-B/vector, V2-B -> C2-R-B/vector/V2-A)
  and both tolerances are bound explicitly;
* ``anchor_budget`` and ``support_budget`` are first replayed through their own
  accepted constructors over their exact ``to_record()`` payloads: a frozen
  record built past ``__post_init__`` that carries a mutated cap under a stale
  ``budget_sha256`` is a typed refusal before either budget is bound or used;
* the V2-B budget digest is bound to the supplied ``anchor_budget`` -- a
  different budget is a typed refusal, never a silently replayed one;
* ``readings``, ``signatures``, ``matches`` and ``usable_masks`` are bound to
  the accepted reading identities, and the accepted replay validators
  (``validate_shift_domain_inputs``,
  ``validate_lateral_vector_domain_inputs``,
  ``validate_relative_translation_graph``,
  ``validate_endpoint_provenance_result``, ``validate_phase_anchor_replay``) are
  re-run over the original sources and budgets;
* every required ``(frame, world axis)`` translation must be supported by the
  accepted C1 axis support, so an unconstrained min-norm fill is never accepted
  as evidence;
* the whole-build integer translation gauge stays explicitly unresolved, exactly
  as the accepted universe carries it.

The module performs no IO and has no filesystem, path, network, process,
environment or dynamic-import access.  It accepts no manifest, F3 value,
coordinate, direction, bound, reference or other manual metadata, it defines no
CLI, it does not touch the persistence schema, and it does not wire the M6
orchestration seam.  It produces no ``ResolvedFacePlacement`` and no volume.
"""
from __future__ import annotations

import math

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_phase_anchor as v2b
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_region_support as v1
from mc_photogrammetry.reconstruction import plane_relative_translation
from mc_photogrammetry.reconstruction import plane_residue
from mc_photogrammetry.reconstruction import plane_shift_domain
from mc_photogrammetry.reconstruction import plane_vector_domain


__all__ = [
    "PLACEMENT_ASSEMBLY_PRODUCER_VERSION",
    "PLACEMENT_ASSEMBLY_REFUSAL_REASONS",
    "assemble_placement_universe",
]

PLACEMENT_ASSEMBLY_PRODUCER_VERSION = "m4-placement-universe-assembler-1"

#: The closed refusal vocabulary this assembler can emit.  It is a subset of the
#: accepted :data:`plane_placement.V2_PLACEMENT_REFUSAL_REASONS`, so a universe
#: refusal is always interchangeable with a placement refusal at the call site.
PLACEMENT_ASSEMBLY_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "missing_provenance",
    "provenance_mismatch",
    "replay_mismatch",
    "c2rb_gate_not_unique",
    "endpoint_gate_not_unique",
    "anchor_gate_not_unique",
    "anchor_axis_unresolved",
    "translation_unsupported",
)

_WORLD_AXES: tuple[int, ...] = (0, 1, 2)
_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    OverflowError,
    ZeroDivisionError,
    FloatingPointError,
)


class _AssemblyRefusal(Exception):
    """Internal control flow: one typed assembly refusal and its JSON detail."""

    def __init__(self, reason: str, detail: object = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = {} if detail is None else detail


def _refuse(reason: str, field: str) -> None:
    raise _AssemblyRefusal(reason, {"field": field})


def _is_digest(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _tolerance(value: object, field: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        _refuse("invalid_input", field)
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result) or result < 0.0:
        _refuse("invalid_input", field)
    return result


# ---------------------------------------------------------------------------
# Explicit input and identity binding
# ---------------------------------------------------------------------------

def _readings_index(readings: object) -> dict:
    """Return ``(frame, canonical_index, axis) -> reading`` for the universe.

    The canonical index is the reading's position among its frame's readings
    ordered by :func:`lattice_planes.reading_order_key`, exactly the accepted
    V2-A ``ReadingIdentity.canonical_index`` convention.
    """
    if not isinstance(readings, tuple) or not readings:
        _refuse("invalid_input", "readings")
    if any(not isinstance(reading, lattice_planes.PlaneReading)
           for reading in readings):
        _refuse("invalid_input", "readings")
    index = placement._readings_index(readings)
    if len(index) != len(readings):
        _refuse("invalid_input", "readings.duplicate")
    return index


def _bind_signatures(signatures: object, readings: tuple) -> None:
    """Bind every signature to exactly one original reading object."""
    if not isinstance(signatures, tuple) or not signatures:
        _refuse("invalid_input", "signatures")
    reading_ids = {id(reading) for reading in readings}
    seen: set[int] = set()
    for signature in signatures:
        if not isinstance(signature, lattice_planes.PlaneSignature):
            _refuse("invalid_input", "signatures")
        reading = signature.reading
        if not isinstance(reading, lattice_planes.PlaneReading) \
                or id(reading) not in reading_ids:
            _refuse("provenance_mismatch", "signatures.reading")
        if id(reading) in seen:
            _refuse("provenance_mismatch", "signatures.duplicate")
        seen.add(id(reading))
    if seen != reading_ids:
        _refuse("missing_provenance", "signatures.coverage")


def _bind_matches(matches: object, index: dict) -> None:
    """Bind every match endpoint to a canonical reading in the universe."""
    if not isinstance(matches, tuple) or not matches:
        _refuse("invalid_input", "matches")
    for match in matches:
        if not isinstance(match, plane_graph.PlaneMatch):
            _refuse("invalid_input", "matches")
        if type(match.left_plane) is not int or match.left_plane < 0 \
                or type(match.right_plane) is not int or match.right_plane < 0:
            _refuse("missing_provenance", "matches.plane_index")
        left = index.get((match.left, match.left_plane, match.left_axis))
        right = index.get((match.right, match.right_plane, match.right_axis))
        if left is None or right is None:
            _refuse("missing_provenance", "matches.endpoint")
        if left.axis != match.left_axis or right.axis != match.right_axis:
            _refuse("provenance_mismatch", "matches.endpoint_axis")


def _bind_usable_masks(usable_masks: object, v2a_result: object,
                       index: dict) -> None:
    """Bind exactly one usable mask to every accepted V2-A reading domain.

    The mask itself is carried unchanged into the universe; its content digest is
    not retained by any accepted upstream record, so the binding enforced here is
    identity, uniqueness, exact boolean-array shape and reading coverage.  The
    accepted V1 support rebuild re-derives the usable digest before any cell is
    consumed.
    """
    if not isinstance(usable_masks, tuple) or not usable_masks:
        _refuse("invalid_input", "usable_masks")
    domains: dict[str, tuple] = {}
    for domain in v2a_result.reading_domains:
        identity = domain.reading_identity
        domains[identity.identity_sha256] = (
            identity.frame, identity.canonical_index, identity.reading_axis)
    if not domains:
        _refuse("missing_provenance", "v2a_result.reading_domains")
    seen: set[str] = set()
    for entry in usable_masks:
        if not isinstance(entry, tuple) or len(entry) != 2:
            _refuse("invalid_input", "usable_masks.entry")
        digest, mask = entry
        if not _is_digest(digest):
            _refuse("invalid_input", "usable_masks.digest")
        if digest not in domains:
            _refuse("provenance_mismatch", "usable_masks.identity")
        if digest in seen:
            _refuse("provenance_mismatch", "usable_masks.duplicate")
        seen.add(digest)
        if not isinstance(mask, np.ndarray) or mask.dtype != bool \
                or mask.ndim != 2:
            _refuse("invalid_input", "usable_masks.mask")
        reading = index.get(domains[digest])
        if reading is None:
            _refuse("missing_provenance", "usable_masks.reading")
        evidence = reading.region_evidence
        if evidence is None:
            # Placement rebuilds each V1 support from the reading's original
            # region evidence; a reading without it can never be placed.
            _refuse("missing_provenance", "readings.region_evidence")
        if tuple(int(value) for value in mask.shape) \
                != tuple(int(value) for value in evidence.image_shape):
            _refuse("provenance_mismatch", "usable_masks.shape")
    if seen != set(domains):
        _refuse("missing_provenance", "usable_masks.coverage")


def _bind_split_sources(split_sources: object) -> None:
    """Type-check the original P0 split-source tuple before V2-B replay."""
    if not isinstance(split_sources, tuple) or not split_sources:
        _refuse("invalid_input", "split_sources")
    for source in split_sources:
        if (not isinstance(source, (tuple, list)) or len(source) not in (2, 3)
                or not isinstance(source[0],
                                  lattice_planes.SplitPhaseProducerResult)):
            _refuse("invalid_input", "split_sources.entry")


def _require_equal(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        _refuse("provenance_mismatch", field)


def _replay_budget(record: object, constructor: type, field: str) -> None:
    """Replay one carried budget through its own accepted constructor.

    ``V2BAnchorBudget`` and ``AlignedSupportBudget`` are frozen records whose
    ``__post_init__`` binds every declared cap into ``budget_sha256``.  A caller
    can still build one through ``object.__setattr__`` and leave the stale digest
    behind, so the assembler rebuilds each budget from its exact ``to_record()``
    payload -- the accepted constructor re-checks every cap type and the digest
    -- and requires the rebuilt record to reproduce the original byte for byte.
    Both budgets are replayed before either digest binding or use.
    """
    try:
        rebuilt = constructor(**record.to_record())
        exact = rebuilt.to_record() == record.to_record()
    except _ERRORS as error:
        raise _AssemblyRefusal(
            "provenance_mismatch",
            {"field": field, "detail": type(error).__name__}) from error
    if not exact:
        _refuse("provenance_mismatch", field)


def _bind_source_digests(
    *, vector_result: object, c2rb_result: object, v2a_result: object,
    v2b_result: object, p1d_result: object, anchor_budget: object,
    universe_sha256: str, comparison_tolerance: float,
    closure_tolerance: float,
) -> None:
    """Bind every retained cross-record digest, tolerance and budget."""
    _require_equal(vector_result.canonical_reading_universe_sha256,
                   universe_sha256, "vector_result.canonical_universe")
    _require_equal(v2a_result.canonical_universe_sha256, universe_sha256,
                   "v2a_result.canonical_universe")
    _require_equal(vector_result.p1d_input_sha256, p1d_result.input_sha256,
                   "vector_result.p1d_input")
    _require_equal(c2rb_result.vector_result_sha256, vector_result.result_sha256,
                   "c2rb_result.vector_result")
    _require_equal(c2rb_result.vector_input_sha256, vector_result.input_sha256,
                   "c2rb_result.vector_input")
    _require_equal(v2a_result.c2rb_result_sha256, c2rb_result.result_sha256,
                   "v2a_result.c2rb_result")
    _require_equal(v2a_result.vector_result_sha256, vector_result.result_sha256,
                   "v2a_result.vector_result")
    if c2rb_result.comparison_tolerance != comparison_tolerance:
        _refuse("replay_mismatch", "c2rb_result.comparison_tolerance")
    if c2rb_result.closure_tolerance != closure_tolerance:
        _refuse("replay_mismatch", "c2rb_result.closure_tolerance")
    if vector_result.comparison_tolerance != comparison_tolerance:
        _refuse("replay_mismatch", "vector_result.comparison_tolerance")
    prediction = v2b_result.prediction
    if prediction is None:
        # The closed V2-B gate below reports the exact typed reason.
        return
    _require_equal(prediction.canonical_universe_sha256, universe_sha256,
                   "v2b_result.canonical_universe")
    _require_equal(prediction.c2rb_result_sha256, c2rb_result.result_sha256,
                   "v2b_result.c2rb_result")
    _require_equal(prediction.vector_result_sha256, vector_result.result_sha256,
                   "v2b_result.vector_result")
    _require_equal(prediction.v2a_result_sha256, v2a_result.result_sha256,
                   "v2b_result.v2a_result")
    _require_equal(prediction.v2a_input_sha256, v2a_result.input_sha256,
                   "v2b_result.v2a_input")
    _require_equal(prediction.budget_sha256, anchor_budget.budget_sha256,
                   "v2b_result.anchor_budget")


def _recompute_universe_sha256(readings: tuple) -> str:
    try:
        return v2a._canonical_universe_sha256(readings)
    except v2a.V2AFailure as error:
        raise _AssemblyRefusal(
            "invalid_input", {"field": "readings.canonical_universe"}) from error


# ---------------------------------------------------------------------------
# Accepted replay validators and readiness invariants
# ---------------------------------------------------------------------------

def _replay(candidate: placement.PlacementUniverse) -> tuple:
    """Re-apply the accepted gates and replay validators over the candidate.

    ``plane_placement._replay_universe`` is the accepted composition of the
    closed C2-R-B / V2-A / V2-B gates and the three public replay validators
    (``validate_relative_translation_graph``,
    ``validate_endpoint_provenance_result``, ``validate_phase_anchor_replay``).
    Reusing it keeps this assembler and placement on one authority.
    """
    try:
        return placement._replay_universe(candidate)
    except placement._InputRefusal:
        raise
    except _ERRORS as error:
        raise _AssemblyRefusal(
            "invalid_input", {"stage": "replay",
                              "detail": type(error).__name__}) from error


def _validate_shift_and_vector(
    *, axis_support: object, p1d_result: object, vector_result: object,
    matches: tuple, signatures: tuple, phase_evidence: tuple, readings: tuple,
    comparison_tolerance: float,
) -> None:
    """Re-run the accepted P1-D and C2-R source validators over the inputs.

    These two public validators bind the shift-domain and vector-domain records
    to the original support, matches, signatures, readings and P1-B phase
    evidence.  ``_replay_universe`` re-derives the same records inside C2-R-B,
    but running them explicitly gives the assembler its own independent
    P1-D/P2 binding and a typed refusal at the exact stage that failed.
    """
    radii = {domain.max_shift_cells for domain in p1d_result.domains}
    if len(radii) != 1:
        _refuse("invalid_input", "p1d_result.max_shift_cells")
    try:
        shift_ok = plane_shift_domain.validate_shift_domain_inputs(
            p1d_result, axis_support, matches, signatures,
            readings=readings, max_shift_cells=radii.pop())
        vector_ok = plane_vector_domain.validate_lateral_vector_domain_inputs(
            vector_result, axis_support, p1d_result, matches, phase_evidence,
            readings=readings, signatures=signatures,
            comparison_tolerance=comparison_tolerance)
    except _ERRORS as error:
        raise _AssemblyRefusal(
            "replay_mismatch", {"stage": "p1d_vector",
                                "detail": type(error).__name__}) from error
    if not shift_ok:
        _refuse("replay_mismatch", "p1d_result")
    if not vector_ok:
        _refuse("replay_mismatch", "vector_result")


def _require_all_axis_support(certificate: object, supported: dict) -> None:
    """Require every certificate frame to be supported on all three world axes.

    Placement consumes a reading's normal axis and both tangent axes, so every
    required ``(frame, world axis)`` translation must be explicitly connected to
    the global C1 anchor with no refusal.  A least-squares min-norm fill is never
    evidence.
    """
    for frame, _translation in certificate.translations_dyadic_xyz:
        for axis in _WORLD_AXES:
            if not supported.get((frame, axis), False):
                raise _AssemblyRefusal(
                    "translation_unsupported", {"frame": frame, "axis": axis})


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def assemble_placement_universe(
    *,
    axis_support: plane_axis_support.C1AxisSupport,
    p1d_result: plane_shift_domain.ShiftDomainResult,
    vector_result: plane_vector_domain.C2RVectorDomainResult,
    c2rb_result: plane_relative_translation.C2RBResult,
    v2a_result: v2a.V2AEndpointResult,
    v2b_result: v2b.V2BAnchorResult,
    readings: tuple,
    signatures: tuple,
    matches: tuple,
    phase_evidence: tuple,
    usable_masks: tuple,
    split_sources: tuple,
    anchor_budget: v2b.V2BAnchorBudget,
    support_budget: v1.AlignedSupportBudget,
    comparison_tolerance: float,
    closure_tolerance: float,
) -> placement.PlacementUniverse | placement.PlacementRefusal:
    """Assemble one replay-validated universe, or one atomic typed refusal.

    Every input is an explicit already-produced in-memory record; no input has a
    default and nothing is fabricated.  On success the returned universe carries
    the caller's exact original objects and is safe to hand to
    :func:`plane_placement.resolve_face_placement`; on any malformed type,
    unbound identity/digest/budget, unresolved anchor axis, gate failure, replay
    contradiction or unsupported required axis the function returns one
    :class:`~plane_placement.PlacementRefusal` and no universe.
    """
    try:
        return _assemble(
            axis_support=axis_support, p1d_result=p1d_result,
            vector_result=vector_result, c2rb_result=c2rb_result,
            v2a_result=v2a_result, v2b_result=v2b_result,
            readings=readings, signatures=signatures, matches=matches,
            phase_evidence=phase_evidence, usable_masks=usable_masks,
            split_sources=split_sources, anchor_budget=anchor_budget,
            support_budget=support_budget,
            comparison_tolerance=comparison_tolerance,
            closure_tolerance=closure_tolerance)
    except _AssemblyRefusal as refusal:
        return placement._make_refusal(refusal.reason, refusal.detail)
    except placement._InputRefusal as refusal:
        return placement._make_refusal(refusal.reason, refusal.detail)
    except _ERRORS as error:
        return placement._make_refusal(
            "invalid_input", {"detail": type(error).__name__})


def _assemble(
    *,
    axis_support: object,
    p1d_result: object,
    vector_result: object,
    c2rb_result: object,
    v2a_result: object,
    v2b_result: object,
    readings: object,
    signatures: object,
    matches: object,
    phase_evidence: object,
    usable_masks: object,
    split_sources: object,
    anchor_budget: object,
    support_budget: object,
    comparison_tolerance: object,
    closure_tolerance: object,
) -> placement.PlacementUniverse:
    comparison = _tolerance(comparison_tolerance, "comparison_tolerance")
    closure = _tolerance(closure_tolerance, "closure_tolerance")

    if not isinstance(axis_support, plane_axis_support.C1AxisSupport):
        _refuse("invalid_input", "axis_support")
    if axis_support.refusal is not None:
        _refuse("missing_provenance", "axis_support.refusal")
    if not isinstance(p1d_result, plane_shift_domain.ShiftDomainResult) \
            or p1d_result.status != "complete":
        _refuse("invalid_input", "p1d_result")
    if not isinstance(vector_result, plane_vector_domain.C2RVectorDomainResult) \
            or vector_result.status != "complete":
        _refuse("invalid_input", "vector_result")
    if not isinstance(c2rb_result, plane_relative_translation.C2RBResult):
        _refuse("invalid_input", "c2rb_result")
    if not isinstance(v2a_result, v2a.V2AEndpointResult):
        _refuse("invalid_input", "v2a_result")
    if not isinstance(v2b_result, v2b.V2BAnchorResult):
        _refuse("invalid_input", "v2b_result")
    if not isinstance(anchor_budget, v2b.V2BAnchorBudget):
        _refuse("invalid_input", "anchor_budget")
    if not isinstance(support_budget, v1.AlignedSupportBudget):
        _refuse("invalid_input", "support_budget")
    if not isinstance(phase_evidence, tuple) or not phase_evidence \
            or any(not isinstance(evidence,
                                  plane_residue.HypothesisPhaseShiftEvidence)
                   for evidence in phase_evidence):
        _refuse("invalid_input", "phase_evidence")

    # Both explicit budgets replay through their accepted constructors before
    # either is digest-bound or used, so a constructor-bypassed stale digest is
    # refused here rather than silently carried into the candidate.
    _replay_budget(anchor_budget, v2b.V2BAnchorBudget, "anchor_budget")
    _replay_budget(support_budget, v1.AlignedSupportBudget, "support_budget")

    typed_readings = readings
    index = _readings_index(typed_readings)
    _bind_signatures(signatures, typed_readings)
    _bind_matches(matches, index)
    _bind_usable_masks(usable_masks, v2a_result, index)
    _bind_split_sources(split_sources)

    universe_sha256 = _recompute_universe_sha256(typed_readings)
    _bind_source_digests(
        vector_result=vector_result, c2rb_result=c2rb_result,
        v2a_result=v2a_result, v2b_result=v2b_result, p1d_result=p1d_result,
        anchor_budget=anchor_budget, universe_sha256=universe_sha256,
        comparison_tolerance=comparison, closure_tolerance=closure)

    if not v2a_result.reading_domains:
        _refuse("missing_provenance", "v2a_result.reading_domains")

    _validate_shift_and_vector(
        axis_support=axis_support, p1d_result=p1d_result,
        vector_result=vector_result, matches=matches, signatures=signatures,
        phase_evidence=phase_evidence, readings=typed_readings,
        comparison_tolerance=comparison)

    candidate = placement.PlacementUniverse(
        c2rb_result=c2rb_result, vector_result=vector_result,
        axis_support=axis_support, p1d_result=p1d_result, matches=matches,
        phase_evidence=phase_evidence, readings=typed_readings,
        signatures=signatures, v2a_result=v2a_result, v2b_result=v2b_result,
        split_sources=split_sources, anchor_budget=anchor_budget,
        support_budget=support_budget, usable_masks=usable_masks,
        comparison_tolerance=comparison, closure_tolerance=closure)

    _component, certificate, _translations, supported, _alpha, _binding = \
        _replay(candidate)
    _require_all_axis_support(certificate, supported)
    return candidate
