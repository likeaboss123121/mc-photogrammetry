"""Bounded image-derived placement assembly seam.

This is the narrow forward step between the accepted DS-M9-C image-derived
lattice measurement and the accepted M4 ``PlacementUniverse`` assembler.  It
consumes exactly one already-produced in-memory terminal - an assembled or
unavailable :class:`~mc_photogrammetry.pipeline.image_derived_lattice_measure.ImageDerivedLatticeResult`,
or the atomic
:class:`~mc_photogrammetry.reconstruction.plane_placement.PlacementRefusal` that
measurement may return instead - plus one explicit mapping of the already-accepted
assembly records, and nothing else.

What it owns
------------

#. **Propagation, never re-measurement.**  A ``PlacementRefusal`` from the
   lattice measurement is returned verbatim.  An ``status "unavailable"``
   lattice is propagated as one typed :class:`ImageDerivedPlacementUnavailable`
   with reason :data:`PLACEMENT_UNAVAILABLE_TERMINAL`; this seam never invents a
   reading, a bridge or a bundle to get past it.
#. **A validated acceptance gate.**  Only an accepted assembled lattice terminal
   advances: it must be an ``ImageDerivedLatticeResult`` instance that replays
   through its own declared dataclass fields and constructor, whose ``terminal``
   is ``geometry_unavailable``, whose ``status`` is ``"assembled"``, and which
   carries a non-empty sealed reading tuple and a provenance bridge.  A foreign
   object, a constructor-bypassed stand-in (``object.__new__`` plus
   ``object.__setattr__`` skips ``__post_init__``), a contradictory terminal, a
   malformed status, empty readings or a missing bridge is one typed
   ``geometry_unavailable`` outcome.
#. **An exact, explicit bundle.**  ``placement_records`` must be a ``Mapping``
   whose keys are exactly the required keyword-only names of
   :func:`~mc_photogrammetry.reconstruction.plane_placement_assembly.assemble_placement_universe`.
   Its ``readings`` must be the lattice's sealed reading universe by exact
   identity/order and canonical reading-universe digest, and its
   ``split_sources`` must be exactly the bridge's
   :meth:`~mc_photogrammetry.reconstruction.plane_geometry_provenance.GeometryProvenanceBridge.split_sources_for_assembly`
   materialisation.  A swapped or foreign bundle is one typed
   ``geometry_unavailable`` outcome.  An absent or non-conforming bundle is one
   typed ``missing_prerequisite`` outcome naming the accepted signature; no
   default, merge, subset or extra key is ever supplied.
#. **Unchanged delegation.**  When both gates pass, the accepted assembler is
   called exactly once as ``assemble_placement_universe(**dict(placement_records))``
   and its exact result - a ``PlacementUniverse`` or a ``PlacementRefusal`` - is
   returned.  Nothing is wrapped, copied, resealed or reinterpreted.

Typed honesty
-------------

:class:`ImageDerivedPlacementUnavailable` is a frozen record carrying only
``terminal`` (always :data:`PLACEMENT_UNAVAILABLE_TERMINAL`), one typed closed
``reason`` (:data:`PLACEMENT_UNAVAILABLE_REASONS`) and a ``detail`` string.  It
never carries readings, geometry, a universe or a placement.

Input discipline
----------------

The public call shape accepts exactly two in-memory arguments and has no
``**kwargs`` through which anything else could arrive.  The module performs no
IO, has no filesystem/path/network/process/environment access, defines no CLI,
does not touch the persistence schema, accepts no manifest, F3 value,
coordinate, direction, bound, reference or manual metadata, and fabricates no
V1/V2/V3 geometry, scale or default.
"""
from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from dataclasses import dataclass

from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.reconstruction import plane_placement_assembly as assembly

__all__ = [
    "PLACEMENT_UNAVAILABLE_TERMINAL",
    "PLACEMENT_UNAVAILABLE_REASONS",
    "REQUIRED_PLACEMENT_KEYWORDS",
    "ImageDerivedPlacementUnavailable",
    "assemble_image_derived_placement",
]

PLACEMENT_UNAVAILABLE_TERMINAL = "geometry_unavailable"
"""The typed terminal boundary carried by every unavailable placement outcome."""

PLACEMENT_UNAVAILABLE_REASONS: tuple[str, ...] = (
    "geometry_unavailable",
    "missing_prerequisite",
)
"""The closed ``reason`` vocabulary of :class:`ImageDerivedPlacementUnavailable`."""


def _required_placement_keywords() -> tuple[str, ...]:
    """The exact required keyword names of the accepted assembler, in order."""
    parameters = inspect.signature(
        assembly.assemble_placement_universe).parameters
    return tuple(
        name for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.KEYWORD_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty)


REQUIRED_PLACEMENT_KEYWORDS: tuple[str, ...] = _required_placement_keywords()
"""Every keyword ``assemble_placement_universe`` requires, captured at import.

The accepted assembler declares no defaults, so the bundle must carry exactly
these names; the seam reads the signature instead of transcribing it, and never
adds, drops or defaults one.
"""


@dataclass(frozen=True, slots=True)
class ImageDerivedPlacementUnavailable:
    """One typed, atomic unavailable placement outcome; it carries no geometry."""

    terminal: str
    reason: str
    detail: str

    def __post_init__(self) -> None:
        if self.terminal != PLACEMENT_UNAVAILABLE_TERMINAL:
            raise ValueError(
                "an image-derived placement terminal must be geometry_unavailable")
        if self.reason not in PLACEMENT_UNAVAILABLE_REASONS:
            raise ValueError(
                "image-derived placement unavailable reason is not in the closed enum")
        if type(self.detail) is not str:
            raise ValueError(
                "image-derived placement unavailable detail must be a string")


def _unavailable(reason: str, detail: str) -> ImageDerivedPlacementUnavailable:
    return ImageDerivedPlacementUnavailable(
        terminal=PLACEMENT_UNAVAILABLE_TERMINAL, reason=reason, detail=detail)


def _bundle_signature_detail() -> str:
    return ("placement_records must be a mapping with exactly the accepted "
            "assemble_placement_universe keywords: "
            + ", ".join(REQUIRED_PLACEMENT_KEYWORDS))


def _reject_lattice(detail: str) -> ImageDerivedPlacementUnavailable:
    return _unavailable("geometry_unavailable", detail)


def _replay_lattice(
    lattice_result: object,
) -> measure.ImageDerivedLatticeResult | None:
    """Rebuild a carried lattice through its declared fields and constructor.

    ``ImageDerivedLatticeResult`` is a frozen slots dataclass, so a stand-in can
    be installed with ``object.__new__`` plus ``object.__setattr__`` without
    ever running ``__post_init__``.  Rebuilding the declared dataclass fields
    through the accepted constructor re-runs the calibration replay, the sealed
    reading/bridge replay and the canonical result-digest check; a record that
    cannot replay is ``None`` and is refused rather than dereferenced.
    """
    try:
        fields = {
            field.name: getattr(lattice_result, field.name)
            for field in dataclasses.fields(measure.ImageDerivedLatticeResult)}
        return measure.ImageDerivedLatticeResult(**fields)
    except measure._ERRORS:
        return None


def _readings_bind(
    bundle_readings: object,
    lattice_result: measure.ImageDerivedLatticeResult,
) -> bool:
    """Whether the bundle readings are the lattice's sealed universe, in order.

    Every element must be the *same object* in the *same position* as
    ``lattice_result.readings``; a content-equal re-construction is refused
    because the accepted assembler binds signatures, masks and split sources by
    reading identity.  The canonical reading-universe digest is recomputed from
    the bundle readings and required to equal the sealed bridge's digest, so the
    bundle cannot smuggle a resealed or reordered universe.
    """
    bridge = lattice_result.bridge
    sealed = lattice_result.readings
    if (bridge is None or not isinstance(bundle_readings, tuple)
            or len(bundle_readings) != len(sealed)):
        return False
    if any(first is not second for first, second in zip(bundle_readings, sealed)):
        return False
    try:
        digest = v2a._canonical_universe_sha256(bundle_readings)
    except (v2a.V2AFailure, measure._ERRORS):
        return False
    return digest == bridge.reading_universe_sha256


def _split_entry_binds(actual: object, expected: tuple) -> bool:
    """Whether one bundle split-source triple is the bridge's, unchanged."""
    if not isinstance(actual, tuple) or len(actual) != len(expected):
        return False
    if actual[0] is not expected[0] or actual[2] is not expected[2]:
        return False
    patch, wanted = actual[1], expected[1]
    try:
        return (patch.dtype == wanted.dtype and patch.shape == wanted.shape
                and patch.tobytes() == wanted.tobytes())
    except AttributeError:
        return False


def _split_sources_bind(
    bundle_split_sources: object,
    lattice_result: measure.ImageDerivedLatticeResult,
) -> bool:
    """Whether the bundle split sources are the sealed bridge's, in order."""
    bridge = lattice_result.bridge
    if bridge is None or not isinstance(bundle_split_sources, tuple):
        return False
    try:
        expected = bridge.split_sources_for_assembly()
    except measure._ERRORS:
        return False
    if len(bundle_split_sources) != len(expected):
        return False
    return all(_split_entry_binds(actual, wanted)
               for actual, wanted in zip(bundle_split_sources, expected))


def assemble_image_derived_placement(
    *,
    lattice_result: object,
    placement_records: object | None = None,
) -> (placement.PlacementUniverse | placement.PlacementRefusal
      | ImageDerivedPlacementUnavailable):
    """Advance one accepted image-derived lattice into the accepted assembler.

    ``lattice_result`` is the terminal of the accepted image-derived lattice
    measurement; ``placement_records`` is the explicit mapping of already-accepted
    assembly records.  A ``PlacementRefusal`` lattice is propagated verbatim, an
    unavailable lattice becomes one typed ``geometry_unavailable`` outcome, and a
    foreign, constructor-bypassed or malformed lattice is rejected with one typed
    ``geometry_unavailable`` outcome.  An assembled lattice without a conforming
    bundle becomes one typed ``missing_prerequisite`` outcome; a bundle whose
    readings or split sources are not the lattice's sealed ones is rejected with
    one typed ``geometry_unavailable`` outcome.  Otherwise the assembler is called
    unchanged with ``**dict(placement_records)`` and its exact result is returned.
    """
    if isinstance(lattice_result, placement.PlacementRefusal):
        return lattice_result
    if not isinstance(lattice_result, measure.ImageDerivedLatticeResult):
        return _reject_lattice(
            "lattice_result is not an accepted image-derived lattice result")
    if _replay_lattice(lattice_result) is None:
        return _reject_lattice(
            "lattice_result is not a constructor-replayable image-derived "
            "lattice result")
    if lattice_result.terminal != measure.GEOMETRY_UNAVAILABLE:
        return _reject_lattice(
            "image-derived lattice terminal is not geometry_unavailable")
    if lattice_result.status == "unavailable":
        reason = lattice_result.reason or "geometry_unavailable"
        return _reject_lattice(
            f"the image-derived lattice is unavailable: {reason}")
    if lattice_result.status != "assembled":
        return _reject_lattice(
            f"image-derived lattice status {lattice_result.status!r} is malformed")
    readings = lattice_result.readings
    if not isinstance(readings, tuple) or not readings:
        return _reject_lattice(
            "an assembled image-derived lattice must carry sealed readings")
    if lattice_result.bridge is None:
        return _reject_lattice(
            "an assembled image-derived lattice must carry a provenance bridge")
    if placement_records is None:
        return _unavailable("missing_prerequisite", _bundle_signature_detail())
    if not isinstance(placement_records, Mapping):
        return _unavailable(
            "missing_prerequisite",
            "placement_records must be a Mapping of the accepted "
            "assemble_placement_universe keywords")
    present = set(placement_records)
    required = set(REQUIRED_PLACEMENT_KEYWORDS)
    if present != required:
        detail = _bundle_signature_detail()
        missing = [name for name in REQUIRED_PLACEMENT_KEYWORDS
                   if name not in present]
        unexpected = [key for key in placement_records if key not in required]
        if missing:
            detail += "; missing: " + ", ".join(missing)
        if unexpected:
            detail += "; unexpected: " + ", ".join(repr(key) for key in unexpected)
        return _unavailable("missing_prerequisite", detail)
    if not _readings_bind(placement_records["readings"], lattice_result):
        return _reject_lattice(
            "placement_records readings are not the sealed image-derived "
            "lattice reading universe in exact identity and order")
    if not _split_sources_bind(placement_records["split_sources"], lattice_result):
        return _reject_lattice(
            "placement_records split_sources are not the sealed image-derived "
            "lattice bridge split sources for assembly")
    return assembly.assemble_placement_universe(**dict(placement_records))
