"""Evaluation-only visible-denominator loss attribution over reachable checkpoints (A2).

This module is the A2 tranche of
``docs/reports/image-only-loss-attribution-design-2026-10-03.md`` sections 4-6.
It is the evaluator-only *measurable* layer that sits on top of the accepted A1
frontier (:mod:`mc_photogrammetry.evaluation_image_only_frontier`): A1 reports
**which** checkpoints a run reached, and A2 scores the reachable ones against a
frozen evaluator-only visible truth.

Contract
--------

* **Evaluation-only, pure and production-free.**  The module imports no
  production orchestration (``pipeline``, ``cli``, ``project.store``) and no
  reconstruction body, exporter or IO module; it never opens a path.  It accepts
  already-loaded truth/reference/alignment *objects* (or the runner's extracted
  production-record data) and returns the labelled attribution record.  It
  reuses the accepted evaluator semantics - :func:`evaluation.visible_metrics` for
  the G/E cell numbers and :func:`evaluation_pair_correspondence.metric_value` for
  every rate - and reuses A1's frontier record types rather than inventing new
  ones.
* **Five labelled spaces, never summed or interchanged.**  Geometry ``G``,
  face/pixel ownership ``O``, pair correspondence ``P``, candidate/palette ``C``
  and exact state ``E`` each declare a **fixed universe**: reference cells for
  G/C/E, frame pixels / ``(cell, slot, frame)`` face observations for O, and
  frame-pair / pixel-pair identities for P.  A camera diagnostic has no universe
  and no cell-loss equation.  No arithmetic here crosses a universe.
* **Typed unavailable, never a measured zero.**  A checkpoint whose record was
  not carried is ``reachable=False`` with a typed reason and ``metric=None``: it
  contributes no zero anywhere.  ``metric_value`` returns ``null`` on a zero
  denominator, so an undefined rate is never reported as ``0.0``.
* **Checkpoint reach is not metric availability.**  ``AttributionRow.reached``
  keeps its compatibility meaning (a metric record was produced); the serialized
  ``checkpoint_reached`` states *separately* whether the production checkpoint was
  reached under A1 frontier semantics, and is ``null`` when the row names no
  production checkpoint (an aggregate/evaluator-only row).  A reached but
  unscorable checkpoint - Route B's ``entry``/B0 P row is the canonical case - is
  ``checkpoint_reached: true`` with ``reached: false``, ``metric_value: null`` and
  a typed reason, so an unavailable metric is never mistaken for the checkpoint
  not having run, and a reached checkpoint never implies a measured metric.  A row
  whose substage has **no dedicated A1 checkpoint** - Route B's ``promoted``
  (state promotion) and ``terminal`` (decoded export) rows - serializes ``null``
  reach unless its own carried record proves that substage was reached, because
  A1's coarse ``output`` reach is not evidence that a promotion or export subrow
  ran; absence is never inferred as ``false`` where it may mean
  unattempted/unknown.  Presence is explicit (``record is not None``): a carried
  empty mapping proves the substage record exists (reach true, metric unavailable),
  while only a missing record is ``null``.
* **One primary E denominator.**  The primary exact-state denominator is **all**
  frozen visible ``V`` (>= 1 usable pixel), whether or not the truth state is in
  the run palette.  ``V inter palette`` is reported only as a separate conditional
  vocabulary row in the same universe.  The additive in-palette partition
  (:func:`in_palette_loss_partition`) separates geometry absence from label
  confusion over that same ``V_P``: every visible in-palette truth cell is exactly
  one of geometry-missed, candidate-covered wrong state or exact hit, and the two
  rates ``|C inter V_P| / |V_P|`` and ``exact / |C inter V_P|`` are typed ``null``
  when their own denominator is empty.  It only reads the scorer's already-aligned,
  already-validated memberships, never re-aligns or re-fits, and leaves every
  existing row and primary denominator byte-for-byte unchanged.  Route A's report
  builder emits those five additive rows only behind the default-off
  ``include_palette_loss_partition`` option; Route B owns no ``palette_label``
  checkpoint, so the option is rejected there with
  :data:`PALETTE_LOSS_PARTITION_ROUTE_B_REASON` and never claims unsupported
  coverage.
* **The candidate ceiling is the producer palette, checked per cell.**  A C
  candidate row may only count a visible cell whose truth is in the *aligned* run
  palette (``v in V_P``), so ``candidate_available_kK`` can never exceed
  ``palette_coverage``; the conditional exact-state numerator is likewise
  ``E inter V_P``.  Every retained ranked candidate must itself be a palette
  member: an out-of-palette candidate contradicts the run's own vocabulary and is a
  typed refusal, never silently filtered or reordered before the top-``K`` cut.
  The aligned palette and its ``V_P`` membership are in-memory-only evidence and
  are never serialized; the primary ``|V|`` denominator is untouched.
* **Non-monotone, reconciled in-universe.**  A transition joins two comparable
  reached checkpoints in one universe and satisfies
  ``next = previous + gained - lost``.  A ``G`` transition reconciles the geometry
  success set (``V inter candidate``) and is **unpartitioned**, because both
  endpoints already *are* the geometry count.  For a cell-level exact-state ``E``
  transition the loss is partitioned into disjoint ``geometry_lost`` (the cell left
  the scored set because its geometry/placement was lost) and ``state_lost``
  (geometry survived but the label was lost); a cell is counted in exactly one.  A
  P transition carries no cell partition, and a camera diagnostic carries none.
* **The reachable P entry is digest-bound, never faked.**  Route B's entry/B0
  checkpoint lives in the frozen frame-pair / pixel-pair universe, so it may only
  be scored (``truth_pairs_returned / truth_pairs``, plus the accepted
  returned-pair precision) from the *same* joint matcher the entry used
  (``joint_matching.match_and_solve_pair``) when the rederived match-array digest
  equals the accepted ``RelativeCameraEvidence.matches_sha256``.  An
  evaluator-only alternate matcher's aggregate (for example
  ``evaluation_pair_correspondence.evaluate_scene``) is never an achieved Route B
  P metric; without a digest-bound joint set the entry row is typed unavailable
  with :data:`ENTRY_MATCH_UNBOUND_REASON`.  A digest-bound aggregate is *still*
  never scored when the accepted frontier says entry was not reached: the builder
  fails closed to the typed ``entry_unreached`` row (and :func:`pair_metric_rows`
  refuses an explicitly unreached checkpoint), so no numeric P metric can be
  emitted with ``checkpoint_reached: false``.  Entry connectivity is never
  converted into accuracy and is never divided by ``|V|``.
* **No scale fit, ever.**  Alignment is the one permitted global integer
  translation plus lattice Y-quarter-turn supplied by
  ``evaluation.best_visible_alignment`` (or the run's own gauge).  This module
  never fits scale, reflection or a per-cell transform, and it reports the
  unresolved metric scale as a score-reducing caveat, not as ``unavailable``.
* **Run-native inputs, one transform.**  Every candidate map, proposal record,
  ranked candidate list and palette a builder receives is in the *run's own* frame.
  The builder applies the one permitted alignment exactly once, through
  :func:`align_candidate` / :func:`align_candidate_coordinates` for cells and
  :func:`align_candidate_rankings` for the ranked candidate lists, and
  :func:`align_palette_states` for the coordinate-free palette, and joins the result
  against reference-frame truth.  Geometry-only placement/volume records carry no
  ``BlockState``, so they are passed as :class:`GeometryCellEvidence` and scored by
  :func:`score_aligned_geometry_coordinates` (coordinates only) - never as a
  ``{coordinate: None}`` state map, which the state-rotating aligner cannot safely
  consume.  A caller must not pre-align the same records:
  a record transformed twice is a different, unaligned record and is never scored.
  Proposal coordinates and *every* ranked ``BlockState`` are transformed together
  so a directional candidate is never compared outside its aligned orientation, and
  every palette ``BlockState`` receives the *same* directional rotation (the
  alignment's translation cannot touch a coordinate-free palette), so palette
  coverage/E-conditional rows agree with the aligned candidate/E rows.

Route A and Route B are scored separately and never pooled.  Every number below
is labelled policy/availability or a labelled numerator over its declared
denominator; a status distribution is never accuracy, and the first unreached
checkpoint is a frontier indicator, not a causal attribution.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from mc_photogrammetry.domain.models import BlockCoordinate, BlockState
from mc_photogrammetry.evaluation import VisibleMetrics, visible_metrics
from mc_photogrammetry.evaluation_image_only_frontier import (
    FrontierCheckpoint,
    ImageOnlyFrontier,
    RecordKind,
    Route,
    canonical_json,
)
from mc_photogrammetry.evaluation_pair_correspondence import metric_value

__all__ = [
    "ATTRIBUTION_SCHEMA",
    "ATTRIBUTION_PRODUCER",
    "PRIMARY_VISIBILITY_MINIMUM_PIXELS",
    "REPORTED_VISIBILITY_MINIMUMS",
    "DEFAULT_CANDIDATE_DEPTHS",
    "REFERENCE_CELLS_UNIVERSE",
    "FRAME_PIXELS_UNIVERSE",
    "FACE_OBSERVATION_UNIVERSE",
    "PIXEL_PAIRS_UNIVERSE",
    "JOINT_MATCHER_SOURCE",
    "ENTRY_MATCH_UNBOUND_REASON",
    "PALETTE_LOSS_PARTITION_CHECKPOINT",
    "PALETTE_LOSS_PARTITION_ROUTE_B_REASON",
    "ALIGNMENT_TRANSFORM",
    "REQUIRED_ALIGNMENT_KEYS",
    "UNIVERSE_OF_SPACE",
    "KIND_OF_SPACE",
    "CELL_SPACES",
    "AttributionRefusal",
    "MetricSpace",
    "VisibleReference",
    "CellScore",
    "InPaletteLossPartition",
    "GeometryCellEvidence",
    "AttributionRow",
    "AttributionTransition",
    "AttributionSection",
    "ImageOnlyAttribution",
    "visible_reference_from_counts",
    "alignment_record",
    "alignment_from_record",
    "align_candidate",
    "align_candidate_coordinates",
    "align_candidate_rankings",
    "align_palette_states",
    "best_visible_alignment_for_coordinates",
    "score_aligned_cells",
    "score_aligned_geometry_coordinates",
    "palette_coverage",
    "cell_metric_rows",
    "palette_metric_rows",
    "exact_state_metric_rows",
    "in_palette_loss_partition",
    "in_palette_loss_partition_rows",
    "candidate_metric_rows",
    "label_metric_row",
    "EntryMatchBinding",
    "EntryPairEvidence",
    "pair_precision",
    "pair_metric_rows",
    "cell_transition",
    "attribution_unavailable",
    "completion_and_refusal_rates",
    "route_a_attribution",
    "route_b_attribution",
    "attribution_to_record",
    "attribution_digest",
    "build_attribution",
]

ATTRIBUTION_SCHEMA = "mc-photogrammetry-image-only-attribution-1"
"""Schema name carried by every canonical A2 attribution record."""

ATTRIBUTION_PRODUCER = "imgloss-a2-visible-attribution-1"
"""Producer identity carried by every canonical A2 attribution record."""

PRIMARY_VISIBILITY_MINIMUM_PIXELS = 1
"""The reviewed primary visibility threshold: a cell reached by >= 1 usable pixel."""

REPORTED_VISIBILITY_MINIMUMS = (1, 2, 5, 10)
"""Stricter pixel thresholds are separate labelled rows, never the primary row."""

DEFAULT_CANDIDATE_DEPTHS = (1, 2, 4, 8)
"""The accepted candidate-depth sweep ``K`` (design section 2.2 B11)."""

REFERENCE_CELLS_UNIVERSE = "frozen_reference_cells"
FRAME_PIXELS_UNIVERSE = "frozen_frame_pixels"
FACE_OBSERVATION_UNIVERSE = "cell_slot_frame_face_observations"
PIXEL_PAIRS_UNIVERSE = "frozen_pixel_pair_identities"

JOINT_MATCHER_SOURCE = "joint_matching.match_and_solve_pair"
"""The only matcher whose match arrays may be scored as a Route B achieved P row."""

ENTRY_MATCH_UNBOUND_REASON = "entry_match_identities_not_retained_or_digest_unbound"
"""Typed reason for a Route B entry P row with no digest-bound joint match set."""

PALETTE_LOSS_PARTITION_CHECKPOINT = "palette_label"
"""The only checkpoint the reviewed in-palette loss partition is defined over.

Route A's ``palette_label`` exact-state score is the one checkpoint that carries
the aligned run palette beside the frozen visible truth, so it is the only place
the reviewed five-row partition can be emitted.
"""

PALETTE_LOSS_PARTITION_ROUTE_B_REASON = "route_b_has_no_palette_label_checkpoint"
"""Typed reason the reviewed in-palette loss partition is never emitted for Route B.

Route B's reviewed exact-state rows live at ``raw_winner``/``promoted``/``terminal``
and its palette availability row is the C-space ``palette`` row; Route B owns no
``palette_label`` exact-state checkpoint, so the Route A-only partition is rejected
(or explicitly reported as not included) rather than fabricated, and no
unsupported Route B coverage is ever claimed.
"""

ALIGNMENT_TRANSFORM = (
    "candidate (x,y,z) rotated about +Y by quarter turns, then translated into the "
    "reference frame; no reflection, no scale fit, no per-cell transform"
)
"""The only permitted coordinate transform, reported verbatim beside every number."""

REQUIRED_ALIGNMENT_KEYS = ("rotation_quarter_turns", "translation", "transform")
"""An alignment record is only valid when it names all three."""


class AttributionRefusal(ValueError):
    """Raised when an attribution record cannot be built from the inputs given.

    The refusal is atomic: a caller never receives a partially scored section.
    """


class MetricSpace(Enum):
    """The five labelled spaces plus the camera diagnostic; never interchangeable."""

    GEOMETRY = "G"
    OWNERSHIP = "O"
    PAIR_CORRESPONDENCE = "P"
    CANDIDATE = "C"
    EXACT_STATE = "E"
    CAMERA_DIAGNOSTIC = "camera"


#: A cell metric may only ever be reported in one of these three spaces.
CELL_SPACES = (MetricSpace.GEOMETRY, MetricSpace.CANDIDATE, MetricSpace.EXACT_STATE)
#: 64-state vocabulary metadata keyed by the space's fixed universe.
UNIVERSE_OF_SPACE: dict[MetricSpace, str] = {
    MetricSpace.GEOMETRY: REFERENCE_CELLS_UNIVERSE,
    MetricSpace.CANDIDATE: REFERENCE_CELLS_UNIVERSE,
    MetricSpace.EXACT_STATE: REFERENCE_CELLS_UNIVERSE,
    MetricSpace.OWNERSHIP: FRAME_PIXELS_UNIVERSE,
    MetricSpace.PAIR_CORRESPONDENCE: PIXEL_PAIRS_UNIVERSE,
    MetricSpace.CAMERA_DIAGNOSTIC: "none",
}
#: The A1 record kind each labelled space carries; the spaces are never interchanged.
KIND_OF_SPACE: dict[MetricSpace, RecordKind] = {
    MetricSpace.GEOMETRY: RecordKind.GEOMETRY,
    MetricSpace.CANDIDATE: RecordKind.CANDIDATE,
    MetricSpace.EXACT_STATE: RecordKind.EXACT_STATE,
    MetricSpace.OWNERSHIP: RecordKind.OWNERSHIP,
    MetricSpace.PAIR_CORRESPONDENCE: RecordKind.PAIR_CORRESPONDENCE,
    MetricSpace.CAMERA_DIAGNOSTIC: RecordKind.CAMERA_DIAGNOSTIC,
}


def _is_token(value: Any) -> bool:
    return type(value) is str and bool(value) and not any(
        character.isspace() for character in value)


def _is_text(value: Any) -> bool:
    return type(value) is str


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_coordinate(value: Any) -> bool:
    return (isinstance(value, tuple) and len(value) == 3
            and all(_is_int(item) for item in value))


def _is_digest(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AttributionRefusal("a metric value must be a finite number or null")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise AttributionRefusal("a metric value must be finite or null")
    return number


def _canonical(value: Any) -> str:
    """Canonical, key-sorted, ASCII JSON for one attribution payload."""
    try:
        return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except AttributionRefusal:
        raise
    except (TypeError, ValueError) as error:
        raise AttributionRefusal("attribution payload is not canonical JSON") from error


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise AttributionRefusal("attribution floats must be finite")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise AttributionRefusal("attribution payload keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_record") and callable(value.to_record):
        return _json_value(value.to_record())
    raise AttributionRefusal("attribution payload is not a deterministic JSON value")


# ---------------------------------------------------------------------------
# Alignment and the frozen visible reference
# ---------------------------------------------------------------------------

def alignment_record(*, rotation_quarter_turns: int, translation: Sequence[int],
                     source: str) -> dict[str, Any]:
    """One validated, reportable alignment in the only permitted transform family."""
    if not _is_int(rotation_quarter_turns) or not 0 <= rotation_quarter_turns < 4:
        raise AttributionRefusal(
            "alignment quarter turns must be an integer in [0, 4)")
    if (isinstance(translation, (str, bytes)) or len(tuple(translation)) != 3
            or any(not _is_int(value) for value in translation)):
        raise AttributionRefusal("alignment translation must be three integers")
    if not _is_token(source):
        raise AttributionRefusal("alignment source must be a non-empty token")
    return {
        "rotation_quarter_turns": int(rotation_quarter_turns),
        "translation": [int(value) for value in translation],
        "transform": ALIGNMENT_TRANSFORM,
        "source": source,
    }


def alignment_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Re-validate one alignment record; a scale or reflection field is a refusal."""
    if not isinstance(record, Mapping):
        raise AttributionRefusal("an alignment must be a mapping")
    missing = [key for key in REQUIRED_ALIGNMENT_KEYS if key not in record]
    if missing:
        raise AttributionRefusal(f"alignment is missing {missing[0]}")
    if record.get("transform") != ALIGNMENT_TRANSFORM:
        raise AttributionRefusal(
            "alignment transform is not the permitted integer translation + Y-quarter-turn")
    for forbidden in ("scale", "scale_fit", "reflection", "per_cell", "stretch",
                      "scale_factor", "fit_scale"):
        if forbidden in record:
            raise AttributionRefusal(
                f"alignment carries a forbidden {forbidden} field")
    return alignment_record(
        rotation_quarter_turns=record["rotation_quarter_turns"],
        translation=record["translation"], source=str(record.get("source", "evaluator")))


def _rotate_coordinate(coordinate: BlockCoordinate, turns: int) -> BlockCoordinate:
    x, y, z = coordinate
    for _ in range(turns % 4):
        x, z = -z, x
    return x, y, z


def _translate(coordinate: BlockCoordinate, translation: Sequence[int]) -> BlockCoordinate:
    return tuple(coordinate[index] + int(translation[index]) for index in range(3))  # type: ignore[return-value]


def _rotate_state(state: BlockState, turns: int) -> BlockState:
    """Rotate a state's directional properties with the permitted Y-quarter-turn.

    Mirrors ``evaluation``'s own state transform (cardinal keys, ``facing``,
    ``axis``, ``rotation`` and compound ``shape`` tokens) so the scorer never
    compares a rotated coordinate carrying an unrotated orientation.
    """
    properties = dict(state.properties)
    order = ("north", "east", "south", "west")
    turns %= 4
    if not turns:
        return state

    def rotate(direction: str) -> str:
        return order[(order.index(direction) + turns) % 4] if direction in order else direction

    directional = {key: properties.pop(key) for key in order if key in properties}
    for key, value in directional.items():
        properties[rotate(key)] = value
    facing = properties.get("facing")
    if facing in order:
        properties["facing"] = rotate(facing)
    if turns % 2 and properties.get("axis") in {"x", "z"}:
        properties["axis"] = "z" if properties["axis"] == "x" else "x"
    if properties.get("rotation", "").isdigit():
        properties["rotation"] = str((int(properties["rotation"]) + 4 * turns) % 16)
    shape = properties.get("shape")
    if shape:
        properties["shape"] = "_".join(rotate(part) for part in shape.split("_"))
    return BlockState(state.name, tuple(sorted(properties.items())))


@dataclass(frozen=True, slots=True)
class VisibleReference:
    """The frozen evaluator-only visible truth and its pixel-count denominator.

    ``reference`` is every frozen reference cell; ``visible_pixels`` maps each cell
    the evaluator camera reached to the usable-pixel count that reached it.  The
    canonical ``V`` used by every cell metric is the set of cells with at least
    ``minimum_pixels`` usable pixels.  Nothing here is reachable from production:
    the caller loads it after reconstruction has finished.
    """

    reference: Mapping[BlockCoordinate, BlockState]
    visible_pixels: Mapping[BlockCoordinate, int]
    minimum_pixels: int = PRIMARY_VISIBILITY_MINIMUM_PIXELS
    reference_sha256: str = ""
    source: str = "evaluator"

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Mapping) or not isinstance(
                self.visible_pixels, Mapping):
            raise AttributionRefusal(
                "a visible reference needs reference cells and pixel counts")
        if not _is_int(self.minimum_pixels) or self.minimum_pixels < 1:
            raise AttributionRefusal("minimum_pixels must be a positive integer")
        if self.reference_sha256 and not _is_digest(self.reference_sha256):
            raise AttributionRefusal("reference digest must be lowercase SHA-256")
        if not _is_text(self.source):
            raise AttributionRefusal("visible reference source must be a non-empty string")

    @property
    def visible(self) -> dict[BlockCoordinate, BlockState]:
        """The canonical ``V``: reference cells reached by >= ``minimum_pixels``."""
        return {
            coordinate: self.reference[coordinate]
            for coordinate, count in self.visible_pixels.items()
            if coordinate in self.reference and int(count) >= self.minimum_pixels
        }

    def threshold(self, minimum_pixels: int) -> dict[BlockCoordinate, BlockState]:
        """The same universe at a stricter pixel threshold (a separate labelled row)."""
        if not _is_int(minimum_pixels) or minimum_pixels < 1:
            raise AttributionRefusal("a visibility threshold must be a positive integer")
        return {
            coordinate: self.reference[coordinate]
            for coordinate, count in self.visible_pixels.items()
            if coordinate in self.reference and int(count) >= minimum_pixels
        }

    def to_record(self) -> dict[str, Any]:
        counts = [int(value) for value in self.visible_pixels.values()]
        return {
            "source": self.source,
            "reference_sha256": self.reference_sha256,
            "reference_cells": len(self.reference),
            "pixel_counted_cells": len(self.visible_pixels),
            "minimum_pixels": self.minimum_pixels,
            "visible_cells": len(self.visible),
            "pixel_count_min": min(counts) if counts else None,
            "pixel_count_max": max(counts) if counts else None,
        }


def visible_reference_from_counts(
    reference: Mapping[BlockCoordinate, BlockState],
    visible_pixels: Mapping[BlockCoordinate, int],
    *,
    reference_sha256: str = "",
    source: str = "evaluator",
    minimum_pixels: int = PRIMARY_VISIBILITY_MINIMUM_PIXELS,
) -> VisibleReference:
    """Build the evaluator-only visible reference from raycast pixel counts."""
    counts: dict[BlockCoordinate, int] = {}
    for coordinate, count in visible_pixels.items():
        if coordinate not in reference:
            # A cell outside the reference cannot enter the denominator; the
            # raycast is over the reference, so this is a malformed join.
            raise AttributionRefusal(
                "visible-pixel counts name a cell outside the reference")
        if not _is_int(count) or count < 0:
            raise AttributionRefusal("visible-pixel counts must be non-negative integers")
        counts[coordinate] = int(count)
    return VisibleReference(
        reference=dict(reference), visible_pixels=counts,
        minimum_pixels=minimum_pixels, reference_sha256=reference_sha256,
        source=source)


# ---------------------------------------------------------------------------
# Cell scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CellScore:
    """One aligned candidate scored against the canonical visible denominator ``V``.

    ``matched`` is ``V inter candidate`` (geometry present); ``exact`` is the subset
    of ``matched`` whose label equals the truth; ``aligned`` is the whole aligned
    candidate; ``phantoms`` is ``candidate - reference``.  ``matched``/``exact``
    are carried as sets so an exact-state transition can partition its loss into
    disjoint geometry/state cell sets without re-deriving anything.  For a score
    built with a run palette, ``aligned_palette``/``palette_cells`` carry the
    aligned producer vocabulary and its exact visible membership ``V_P`` so a
    candidate row can enforce the vocabulary ceiling per cell.  Both are
    in-memory-only evidence: :meth:`to_record` never serializes palette contents.
    """

    space: MetricSpace
    checkpoint: str
    kind: RecordKind
    universe: str
    denominator: str
    alignment: Mapping[str, Any]
    visible_cells: int
    matched: frozenset[BlockCoordinate]
    exact: frozenset[BlockCoordinate]
    aligned_cells: int
    phantom_cells: int
    hidden_inferred_cells: int
    hidden_exact_cells: int
    palette_coverage: int | None = None
    conditional_denominator: int | None = None
    conditional_denominator_label: str = ""
    coordinate_truth: Mapping[BlockCoordinate, BlockState] | None = None
    """Reference truth keyed by reference-frame coordinate.

    ``matched``/``exact`` are sets of reference-frame coordinates (the alignment
    maps the candidate into the reference frame), and this mapping carries the
    truth for every canonical visible cell, so a candidate-availability row can ask
    whether the truth state sits among a cell's retained candidates without ever
    crossing into a different universe.  ``__post_init__`` replaces the caller's
    mapping with an immutable snapshot (a private copy behind a read-only view), so
    post-construction mutation through the exposed mapping or the constructor's
    source mapping can never silently stale the validated ``V_P`` evidence.
    """

    aligned_palette: frozenset[BlockState] | None = None
    """The aligned producer palette (run-native palette through the one permitted
    directional rotation).  In-memory membership evidence for the C ceiling; the
    raw palette is never serialized."""

    palette_cells: frozenset[BlockCoordinate] | None = None
    """``V_P``: the visible cells whose reference truth is in ``aligned_palette``.

    Carried explicitly so the C ceiling (``truth(v) in ranked(v)[:K]`` for
    ``v in V_P``) and the conditional exact-state numerator (``E inter V_P``) use
    the validated aligned membership rather than re-deriving it.  ``None`` is typed
    unknown (no palette was supplied) and is never treated as an empty palette.
    Palette evidence requires ``coordinate_truth``: the invariant recomputes this
    set exactly as ``{v: truth(v) in aligned_palette}`` from that truth context and
    refuses a missing context or any subset/superset mismatch, so a forged
    membership set can never reach the C/E rows.  A palette aggregate
    (``palette_coverage``/``conditional_denominator``) additionally requires this
    complete context; an aggregate-only record is refused rather than falling back
    to the whole ``E`` set.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.space, MetricSpace):
            raise AttributionRefusal("a cell score needs one MetricSpace")
        if self.space not in CELL_SPACES:
            raise AttributionRefusal("a cell score may only use a cell space")
        if not _is_token(self.checkpoint) or not _is_token(self.universe):
            raise AttributionRefusal("a cell score needs a checkpoint and a universe")
        if not isinstance(self.kind, RecordKind):
            raise AttributionRefusal("a cell score needs a RecordKind")
        if not isinstance(self.matched, frozenset) or not isinstance(self.exact, frozenset):
            raise AttributionRefusal("cell score memberships must be frozen sets")
        if not self.exact <= self.matched:
            raise AttributionRefusal("exact-state cells must be a subset of matched cells")
        if len(self.matched) > self.visible_cells:
            raise AttributionRefusal("matched cells cannot exceed the visible denominator")
        if self.conditional_denominator is not None and (
                not _is_int(self.conditional_denominator)
                or self.conditional_denominator < 0
                or self.conditional_denominator > self.visible_cells):
            raise AttributionRefusal("conditional denominator is out of range")
        if self.palette_coverage is not None and (
                not _is_int(self.palette_coverage)
                or self.palette_coverage < 0
                or self.palette_coverage > self.visible_cells):
            raise AttributionRefusal("palette coverage is out of range")
        if (self.aligned_palette is None) != (self.palette_cells is None):
            raise AttributionRefusal(
                "aligned palette and its visible membership are one evidence pair")
        # A palette aggregate (unconditional coverage or conditional denominator)
        # is only meaningful beside the complete palette context.  Without it the
        # conditional row used to fall back to the whole ``E`` set and could report
        # a rate above its own ``|V_P|`` ceiling; refuse the incomplete record here
        # so that fallback state is unconstructible.
        if (self.palette_coverage is not None
                or self.conditional_denominator is not None) \
                and self.aligned_palette is None:
            raise AttributionRefusal(
                "palette aggregates require the complete palette evidence "
                "(aligned vocabulary, exact V_P and visible truth context)")
        # ``coordinate_truth`` arrives as a caller-owned mapping that stays mutable
        # after construction: a later ``score.coordinate_truth[cell] = other`` (or a
        # mutation through the constructor's source mapping) could silently stale
        # the ``V_P`` membership that the C ceiling and the conditional E numerator
        # were validated against.  Replace it with an immutable snapshot - a
        # read-only view over a private copy - before the exact ``V_P``
        # recomputation below reads it.
        if self.coordinate_truth is not None:
            if not isinstance(self.coordinate_truth, Mapping):
                raise AttributionRefusal(
                    "the visible truth context must be a coordinate -> state mapping")
            object.__setattr__(
                self, "coordinate_truth",
                MappingProxyType(dict(self.coordinate_truth)))
        if self.aligned_palette is not None:
            if not isinstance(self.aligned_palette, frozenset) \
                    or not isinstance(self.palette_cells, frozenset):
                raise AttributionRefusal(
                    "aligned palette membership must be carried as frozen sets")
            # Palette evidence is only meaningful against the visible truth
            # context: without ``coordinate_truth`` there is nothing to validate
            # ``V_P`` against, so a forged or stale membership set could silently
            # mis-report the C ceiling and the conditional E numerator.  Fail
            # closed instead of trusting the carried set.
            if self.coordinate_truth is None:
                raise AttributionRefusal(
                    "palette evidence requires the visible truth context that "
                    "validates V_P")
            # ``V_P`` must be *exactly* the visible truth cells whose state is in
            # the aligned palette.  Recompute it from the frozen truth snapshot
            # rather than accepting a subset claim: a forged incomplete or
            # superset membership set must never reach the C/E rows.
            expected_palette_cells = frozenset(
                coordinate for coordinate, truth in self.coordinate_truth.items()
                if truth in self.aligned_palette)
            if self.palette_cells != expected_palette_cells:
                raise AttributionRefusal(
                    "V_P must be exactly the visible truth cells whose state is in "
                    "the aligned palette")
            # ``V_P`` is exactly the covered visible cells, so the unconditional
            # coverage count and the conditional denominator are both ``|V_P|``.
            if self.palette_coverage != len(self.palette_cells) \
                    or self.conditional_denominator != len(self.palette_cells):
                raise AttributionRefusal(
                    "palette coverage and the conditional denominator must equal |V_P|")

    @property
    def geometry_recall(self) -> dict[str, Any]:
        return metric_value(len(self.matched), self.visible_cells,
                            undefined_reason="no_visible_reference_cells")

    @property
    def geometry_precision(self) -> dict[str, Any]:
        return metric_value(len(self.matched), len(self.matched) + self.phantom_cells,
                            undefined_reason="no_placed_candidate_cells")

    @property
    def exact_state_over_visible(self) -> dict[str, Any]:
        return metric_value(len(self.exact), self.visible_cells,
                            undefined_reason="no_visible_reference_cells")

    @property
    def exact_state_over_palette(self) -> dict[str, Any] | None:
        if self.conditional_denominator is None:
            # No palette aggregate means no conditional row; an absent palette is
            # never replaced by the all-visible ``E`` rate.
            return None
        # The conditional numerator is the exact cells whose truth is *in* the
        # aligned palette (``E inter V_P``).  ``__post_init__`` guarantees the
        # complete palette evidence whenever the conditional denominator exists, so
        # there is no whole-``E`` fallback and the rate can never exceed ``|V_P|``.
        numerator = len(self.exact & self.palette_cells)
        return metric_value(numerator, self.conditional_denominator,
                            undefined_reason="no_visible_cell_in_run_palette")

    def to_record(self) -> dict[str, Any]:
        return {
            "space": self.space.value,
            "checkpoint": self.checkpoint,
            "kind": self.kind.value,
            "universe": self.universe,
            "denominator": self.denominator,
            "alignment": dict(self.alignment),
            "visible_cells": self.visible_cells,
            "matched_cells": len(self.matched),
            "exact_state_cells": len(self.exact),
            "aligned_candidate_cells": self.aligned_cells,
            "phantom_cells": self.phantom_cells,
            "hidden_inferred_cells": self.hidden_inferred_cells,
            "hidden_exact_state_cells": self.hidden_exact_cells,
            "palette_coverage": self.palette_coverage,
            "conditional_denominator": self.conditional_denominator,
            "conditional_denominator_label": self.conditional_denominator_label,
        }


@dataclass(frozen=True, slots=True)
class InPaletteLossPartition:
    """The additive in-palette partition of one exact-state checkpoint's ``V_P``.

    Every visible cell whose frozen truth is in the aligned run palette (``V_P``)
    falls into exactly one of three disjoint categories under the scorer's own
    frozen alignment and palette:

    * ``geometry_missed`` - no aligned candidate covers the cell, so no label could
      have been read there;
    * ``wrong_state`` - an aligned candidate covers the cell but its label differs
      from the frozen truth;
    * ``exact_hit`` - the aligned candidate's label equals the frozen truth.

    This separates geometry absence from label confusion, which the aggregate
    ``exact_state_recovery_in_palette`` numerator alone cannot do: a zero
    conditional exact-state rate may simply mean the candidate never covered the
    in-palette cells.  The two requested rates travel beside the counts:
    :attr:`candidate_opportunity` is ``|C inter V_P| / |V_P|`` (the share of
    in-palette cells the candidate actually covered) and
    :attr:`exact_state_in_candidate_opportunity` is ``exact / |C inter V_P|``, which
    is typed ``null`` when the opportunity denominator is empty.  The cell sets are
    in-memory-only evidence, exactly like ``CellScore``'s palette membership:
    :meth:`to_record` serializes counts and rates, never coordinates.
    """

    palette_cells: frozenset[BlockCoordinate]
    visible_cells: int
    geometry_missed: frozenset[BlockCoordinate]
    wrong_state: frozenset[BlockCoordinate]
    exact_hit: frozenset[BlockCoordinate]

    def __post_init__(self) -> None:
        if not _is_int(self.visible_cells) or self.visible_cells < 0:
            raise AttributionRefusal("the in-palette partition needs a visible count")
        for name in ("palette_cells", "geometry_missed", "wrong_state", "exact_hit"):
            value = getattr(self, name)
            if not isinstance(value, frozenset):
                raise AttributionRefusal(
                    f"the in-palette partition {name} must be a frozenset")
            if any(not _is_coordinate(cell) for cell in value):
                raise AttributionRefusal(
                    "in-palette partition cells must be integer triples")
        if len(self.palette_cells) > self.visible_cells:
            raise AttributionRefusal("V_P cannot exceed the visible denominator")
        if self.geometry_missed | self.wrong_state | self.exact_hit \
                != self.palette_cells:
            raise AttributionRefusal(
                "geometry_missed, wrong_state and exact_hit must partition V_P")
        if len(self.geometry_missed) + len(self.wrong_state) + len(self.exact_hit) \
                != len(self.palette_cells):
            raise AttributionRefusal("the in-palette partition must be disjoint")

    @property
    def candidate_covered(self) -> int:
        """``|C inter V_P|``: covered in-palette cells (wrong state + exact hit)."""
        return len(self.wrong_state) + len(self.exact_hit)

    @property
    def candidate_opportunity(self) -> dict[str, Any]:
        """``|C inter V_P| / |V_P|``; ``null`` when ``V_P`` is empty."""
        return metric_value(
            self.candidate_covered, len(self.palette_cells),
            undefined_reason="no_visible_cell_in_run_palette")

    @property
    def exact_state_in_candidate_opportunity(self) -> dict[str, Any]:
        """``exact / |C inter V_P|``; ``null`` when the opportunity is empty."""
        return metric_value(
            len(self.exact_hit), self.candidate_covered,
            undefined_reason="no_candidate_covered_palette_cell")

    def to_record(self) -> dict[str, Any]:
        return {
            "visible_cells": self.visible_cells,
            "visible_in_palette_cells": len(self.palette_cells),
            "candidate_covered_in_palette_cells": self.candidate_covered,
            "geometry_missed_cells": len(self.geometry_missed),
            "wrong_state_cells": len(self.wrong_state),
            "exact_state_hit_cells": len(self.exact_hit),
            "candidate_opportunity": self.candidate_opportunity,
            "exact_state_in_candidate_opportunity":
                self.exact_state_in_candidate_opportunity,
            "partition": (
                "geometry_missed + wrong_state + exact_state_hit = |V_P| "
                "(disjoint in-palette visible truth cells)"),
        }


def in_palette_loss_partition(score: CellScore) -> InPaletteLossPartition | None:
    """Partition one exact-state score's visible in-palette truth cells.

    The partition reads only the score's own frozen, already-aligned memberships -
    ``matched`` (``C``), ``exact`` (``E``) and the validated ``palette_cells``
    (``V_P``) - so it applies no second transform and no scale fit.  It returns
    ``None`` when the score carries no aligned palette evidence (typed unknown,
    never an empty ``V_P``), and refuses a score whose space is not exact state,
    because a geometry-only candidate carries no label that could be classified as
    a wrong state.  ``V_P`` is never mutated and the primary all-``V`` denominators
    are untouched.
    """
    if not isinstance(score, CellScore):
        raise AttributionRefusal("an in-palette loss partition needs a cell score")
    if score.space is not MetricSpace.EXACT_STATE:
        raise AttributionRefusal(
            "an in-palette loss partition needs an exact-state cell score, because a "
            "geometry-only candidate carries no state to score as a wrong state")
    if score.palette_cells is None or score.aligned_palette is None:
        return None
    palette_cells = score.palette_cells
    covered = score.matched & palette_cells
    exact_hit = score.exact & palette_cells
    return InPaletteLossPartition(
        palette_cells=frozenset(palette_cells), visible_cells=score.visible_cells,
        geometry_missed=frozenset(palette_cells - covered),
        wrong_state=frozenset(covered - exact_hit),
        exact_hit=frozenset(exact_hit))


def in_palette_loss_partition_rows(
    score: CellScore, *, record_digest: str = "", checkpoint: str | None = None,
    checkpoint_reached: bool | None = None,
) -> tuple[AttributionRow, ...]:
    """Additive rows for :func:`in_palette_loss_partition`, in the E universe.

    The five rows are new: three disjoint partition counts over ``|V_P|`` plus the
    two requested rates.  They are emitted only on explicit request and are never
    wired into the existing G/E/C row builders, so every existing row, denominator
    and canonical payload stays unchanged.  A score without aligned palette
    evidence is a typed refusal, never an assumed empty ``V_P``.
    """
    partition = in_palette_loss_partition(score)
    if partition is None:
        raise AttributionRefusal(
            "in-palette loss partition rows require the aligned palette membership")
    target = checkpoint if checkpoint is not None else score.checkpoint
    palette_denominator = "V inter run palette (conditional vocabulary diagnostic)"
    opportunity_denominator = (
        "candidate-covered visible cells whose truth is in the run palette")
    rows: list[AttributionRow] = []
    for metric, count in (
            ("in_palette_geometry_missed", len(partition.geometry_missed)),
            ("in_palette_wrong_state", len(partition.wrong_state)),
            ("in_palette_exact_state_hit", len(partition.exact_hit))):
        rows.append(AttributionRow(
            checkpoint=target, metric=metric, space=MetricSpace.EXACT_STATE,
            kind="partition", universe=REFERENCE_CELLS_UNIVERSE,
            denominator=palette_denominator, reached=True, typed_reason="",
            record_digest=record_digest,
            metric_value=metric_value(
                count, len(partition.palette_cells),
                undefined_reason="no_visible_cell_in_run_palette"),
            ceiling=None, conditional_denominator=None,
            conditional_denominator_label="", checkpoint_reached=checkpoint_reached))
    rows.append(AttributionRow(
        checkpoint=target, metric="candidate_opportunity_in_palette",
        space=MetricSpace.EXACT_STATE, kind="availability",
        universe=REFERENCE_CELLS_UNIVERSE, denominator=palette_denominator,
        reached=True, typed_reason="", record_digest=record_digest,
        metric_value=partition.candidate_opportunity, ceiling=None,
        conditional_denominator=None, conditional_denominator_label="",
        checkpoint_reached=checkpoint_reached))
    rows.append(AttributionRow(
        checkpoint=target, metric="exact_state_in_candidate_opportunity",
        space=MetricSpace.EXACT_STATE, kind="label",
        universe=REFERENCE_CELLS_UNIVERSE, denominator=opportunity_denominator,
        reached=True, typed_reason="", record_digest=record_digest,
        metric_value=partition.exact_state_in_candidate_opportunity, ceiling=None,
        conditional_denominator=partition.candidate_covered,
        conditional_denominator_label=palette_denominator,
        checkpoint_reached=checkpoint_reached))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class GeometryCellEvidence:
    """Coordinate-only geometry evidence for one Route B checkpoint.

    A Route B ``output.placement`` is a ``ResolvedFacePlacement`` and
    ``output.volume`` is a ``PlaneVolumeResult``; both carry exact world block
    coordinates and **no** block state.  They are not ``{coordinate: None}``
    mappings: :func:`align_candidate` consumes and rotates ``BlockState`` values, so
    a ``None`` value would either raise under a nonidentity Y-quarter-turn or
    fabricate label semantics.  This wrapper keeps the record as coordinates only,
    so :func:`score_aligned_geometry_coordinates` can apply the one permitted
    transform and score the geometry without inventing a state.  It is
    in-memory-only evidence and is never serialized.
    """

    cells: frozenset[BlockCoordinate]
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.cells, frozenset):
            raise AttributionRefusal(
                "geometry evidence must freeze its coordinates as a frozenset")
        if any(not _is_coordinate(cell) for cell in self.cells):
            raise AttributionRefusal(
                "geometry evidence coordinates must be integer triples")
        if not _is_text(self.source):
            raise AttributionRefusal("geometry evidence source must be a string")


def align_candidate(
    candidate: Mapping[BlockCoordinate, BlockState], alignment: Mapping[str, Any],
) -> dict[BlockCoordinate, BlockState]:
    """Map a run's candidate into the reference frame under the permitted alignment.

    This is the only coordinate transform the ledger is allowed to apply: integer
    translation plus lattice Y-quarter-turn, with directional state properties
    rotated to match.  A candidate/proposal map must be passed through this helper
    before it can be compared with reference-frame truth.
    """
    checked = alignment_from_record(alignment)
    return {
        _translate(_rotate_coordinate(coordinate, checked["rotation_quarter_turns"]),
                   checked["translation"]): _rotate_state(
                       state, checked["rotation_quarter_turns"])
        for coordinate, state in candidate.items()
    }


def align_candidate_coordinates(
    coordinates: Iterable[BlockCoordinate], alignment: Mapping[str, Any],
) -> frozenset[BlockCoordinate]:
    """Map run-native candidate **coordinates** into the reference frame exactly once.

    This is the coordinate-only half of :func:`align_candidate`: geometry-only
    placement/volume records carry no ``BlockState``, so only the integer
    translation plus the lattice Y-quarter-turn of the coordinate apply.  The
    directional state rotation is meaningless for a record with no states and is
    therefore not applied here.  A caller passes **run-native** coordinates; the
    one permitted transform is applied exactly once.
    """
    checked = alignment_from_record(alignment)
    turns = checked["rotation_quarter_turns"]
    translation = checked["translation"]
    aligned: set[BlockCoordinate] = set()
    for coordinate in coordinates:
        if not _is_coordinate(coordinate):
            raise AttributionRefusal(
                "candidate coordinates must be integer triples")
        aligned.add(_translate(_rotate_coordinate(coordinate, turns), translation))
    return frozenset(aligned)


def align_candidate_rankings(
    rankings: Mapping[BlockCoordinate, Sequence[BlockState]],
    alignment: Mapping[str, Any],
) -> dict[BlockCoordinate, tuple[BlockState, ...]]:
    """Map ranked run-native candidates into the reference frame exactly once.

    ``rankings`` is the run's own ``{cell: (candidate, ...)}`` proposal map: the
    ordered ranked candidate list each proposal retained in the run frame.  Every
    cell coordinate **and every ranked state** pass through the *same*
    :func:`align_candidate` semantics the cell scorers apply - one integer
    translation plus one lattice Y-quarter-turn, with the directional state
    properties rotated - so the result is keyed by reference-frame coordinates and
    carries reference-frame states, ready to join against
    ``CellScore.coordinate_truth``.  Ranking order (and therefore any position or
    cost association) is preserved: the returned tuple is the input sequence in
    order, through the same transform a single candidate map would receive.

    A caller passes **run-native** proposal records.  This helper applies the one
    permitted transform, so a second scorer must never transform its result again.
    """
    checked = alignment_from_record(alignment)
    aligned: dict[BlockCoordinate, tuple[BlockState, ...]] = {}
    for coordinate, states in rankings.items():
        ordered = tuple(states)
        if not ordered:
            continue
        target = next(iter(align_candidate({coordinate: ordered[0]}, checked)))
        aligned[target] = tuple(
            align_candidate({coordinate: state}, checked)[target]
            for state in ordered)
    return aligned


def align_palette_states(
    palette_states: Iterable[BlockState], alignment: Mapping[str, Any],
) -> frozenset[BlockState]:
    """Map a run's native palette into the reference frame under the permitted alignment.

    A palette is a set of ``BlockState`` records with no coordinates, so the
    alignment's **translation cannot touch it**: only the lattice Y-quarter-turn's
    directional state mapping applies, and it must be the *same* mapping
    :func:`align_candidate` applies to a candidate's states (``facing``,
    ``axis``, ``rotation``, compound ``shape`` and cardinal property keys).  A
    directional truth state is therefore never compared against an unrotated
    run-native palette, which would make the C palette-coverage denominator and the
    E in-palette conditional row disagree with the aligned candidate/E rows.
    A zero quarter-turn is the identity, and the result is a ``frozenset`` because
    palette membership is order-independent.

    ``palette_states`` must be **run-native**; exactly as for
    :func:`align_candidate`, this helper applies the one permitted transform, so a
    caller never passes an already reference-frame palette.
    """
    checked = alignment_from_record(alignment)
    turns = checked["rotation_quarter_turns"]
    return frozenset(_rotate_state(state, turns) for state in palette_states)


def best_visible_alignment_for_coordinates(
    reference: Mapping[BlockCoordinate, BlockState],
    candidate_coordinates: Iterable[BlockCoordinate],
    visible_reference: Iterable[BlockCoordinate],
) -> tuple[int, BlockCoordinate]:
    """The accepted alignment search specialised to coordinate-only evidence.

    Mirrors :func:`evaluation.best_visible_alignment`'s permitted transform family
    and candidate/vote search exactly, but ranks a hypothesis by its **coordinate**
    matches alone: a geometry-only placement/volume record carries no labels, so
    there is no state-match term to rank with and no ``BlockState`` to rotate.  The
    returned ``(quarter_turns, translation)`` maps a candidate coordinate ``c`` to
    ``translate(rotate(c, turns), translation)``, the same transform
    :func:`align_candidate_coordinates` applies.  The caller turns it into the
    single validated alignment record via :func:`alignment_record`; this helper
    never fits scale or reflection and never invents a state.
    """
    reference_coordinates = set(reference)
    for coordinate in reference_coordinates:
        if not _is_coordinate(coordinate):
            raise AttributionRefusal(
                "reference coordinates must be integer triples")
    candidate = frozenset(candidate_coordinates)
    for coordinate in candidate:
        if not _is_coordinate(coordinate):
            raise AttributionRefusal(
                "candidate coordinates must be integer triples")
    visible = set(visible_reference) & reference_coordinates
    if not reference_coordinates or not candidate:
        return 0, (0, 0, 0)
    reference_sample = sorted(reference_coordinates)[:20_000]
    best = (0, (0, 0, 0), -1)
    for turns in range(4):
        rotated = {_rotate_coordinate(coordinate, turns)
                   for coordinate in candidate}
        candidate_sample = sorted(rotated)[:2_000]
        votes = Counter(
            (reference_coordinate[0] - candidate_coordinate[0],
             reference_coordinate[1] - candidate_coordinate[1],
             reference_coordinate[2] - candidate_coordinate[2])
            for reference_coordinate in reference_sample
            for candidate_coordinate in candidate_sample
        )
        for translation, _count in votes.most_common(32):
            aligned = {_translate(coordinate, translation) for coordinate in rotated}
            coordinate_matches = len(visible & aligned)
            if coordinate_matches > best[2]:
                best = (turns, translation, coordinate_matches)
    return best[0], best[1]


def score_aligned_cells(
    *,
    reference: Mapping[BlockCoordinate, BlockState],
    candidate: Mapping[BlockCoordinate, BlockState],
    visible_reference: VisibleReference,
    alignment: Mapping[str, Any],
    checkpoint: str,
    space: MetricSpace,
    denominator: str = "all frozen visible V (primary, >= 1 usable pixel)",
    palette_states: Iterable[BlockState] | None = None,
    minimum_pixels: int | None = None,
) -> CellScore:
    """Score one aligned candidate against the canonical visible denominator ``V``.

    The alignment maps the **candidate** into the reference frame, exactly as the
    accepted evaluator does: candidate coordinates are rotated about +Y and then
    translated, and the directional block-state properties rotate with them.  The
    reference and the canonical ``V`` stay in their own frame.  ``palette_states``
    is the run's **native** palette; :func:`align_palette_states` applies the same
    directional rotation (the translation cannot affect a coordinate-free palette)
    before any membership count, so a directional truth state is never scored
    against an unrotated palette.  The aligned palette and its exact visible
    membership ``V_P`` (the visible cells whose truth is in it) are carried on the
    returned score as in-memory-only evidence, so the C candidate ceiling and the
    conditional exact-state numerator share one validated vocabulary; the raw
    palette is never serialized and the primary ``|V|`` denominator is unchanged.
    No scale, reflection or per-cell fit is
    possible.  Reusing the accepted
    ``visible_metrics`` is deliberate: the counts below are exactly its counts, and
    the extra membership sets exist only so an A2 transition can reconcile cell by
    cell.
    """
    alignment = alignment_from_record(alignment)
    turns = alignment["rotation_quarter_turns"]
    translation = alignment["translation"]
    threshold = visible_reference.minimum_pixels if minimum_pixels is None else minimum_pixels
    visible = visible_reference.threshold(threshold)
    aligned_candidate = align_candidate(candidate, alignment)
    visible_coordinates = set(visible)
    metrics: VisibleMetrics = visible_metrics(
        reference, aligned_candidate, visible_coordinates,
        translation=(0, 0, 0), rotation_quarter_turns=0)
    matched = {
        coordinate for coordinate in visible_coordinates
        if coordinate in aligned_candidate
    }
    exact = {
        coordinate for coordinate in matched
        if reference[coordinate] == aligned_candidate[coordinate]
    }
    coordinate_truth = {
        coordinate: reference[coordinate] for coordinate in visible_coordinates
    }
    coverage = None
    conditional = None
    label = ""
    aligned_palette = None
    palette_cells = None
    if palette_states is not None:
        palette = align_palette_states(palette_states, alignment)
        # ``V_P`` is the exact visible membership of the *aligned* producer
        # palette: the cells whose reference truth is a member of it.  The
        # unconditional ``|V|`` denominator is never reduced by it.
        palette_cells = frozenset(
            coordinate for coordinate in visible_coordinates
            if reference[coordinate] in palette)
        aligned_palette = frozenset(palette)
        coverage = len(palette_cells)
        conditional = coverage
        label = "V inter run palette (conditional vocabulary diagnostic)"
    if metrics.visible_reference_blocks != len(visible_coordinates) \
            or metrics.visible_coordinate_matches != len(matched) \
            or metrics.visible_exact_state_matches != len(exact):
        raise AttributionRefusal(
            "the accepted evaluator metrics disagree with the carried memberships")
    return CellScore(
        space=space, checkpoint=checkpoint, kind=KIND_OF_SPACE[space],
        universe=UNIVERSE_OF_SPACE[space], denominator=denominator,
        alignment=alignment, visible_cells=len(visible_coordinates),
        matched=frozenset(matched), exact=frozenset(exact),
        aligned_cells=metrics.aligned_candidate_blocks,
        phantom_cells=metrics.phantom_blocks,
        hidden_inferred_cells=metrics.hidden_inferred_blocks,
        hidden_exact_cells=metrics.hidden_exact_state_blocks,
        palette_coverage=coverage, conditional_denominator=conditional,
        conditional_denominator_label=label, coordinate_truth=coordinate_truth,
        aligned_palette=aligned_palette, palette_cells=palette_cells)


def score_aligned_geometry_coordinates(
    *,
    reference: Mapping[BlockCoordinate, BlockState],
    cells: Iterable[BlockCoordinate],
    visible_reference: VisibleReference,
    alignment: Mapping[str, Any],
    checkpoint: str,
    space: MetricSpace = MetricSpace.GEOMETRY,
    denominator: str = "all frozen visible V (primary, >= 1 usable pixel)",
    palette_states: Iterable[BlockState] | None = None,
    minimum_pixels: int | None = None,
) -> CellScore:
    """Score coordinate-only geometry evidence against the canonical visible ``V``.

    Placement (``ResolvedFacePlacement``) and volume (``PlaneVolumeResult``) records
    carry exact block coordinates but **no** ``BlockState``, so they must not be
    forced through the state-rotating :func:`align_candidate`.  This is the exact
    coordinate-only counterpart of :func:`score_aligned_cells`: it applies the same
    one permitted integer translation + lattice Y-quarter-turn to the coordinates
    only, uses the same frozen visible denominator and computes the same G counts
    ``visible_metrics`` would for a state-free candidate (``matched`` = ``V inter
    aligned``, ``phantom`` = ``aligned - reference``, hidden inferred = ``hidden
    inter aligned``).  The returned :class:`CellScore` carries the existing fields;
    ``exact``/``hidden_exact`` are empty because the record genuinely has no labels,
    so no fabricated state ever reaches an E row.

    ``palette_states`` is the run's native palette and is aligned with the *same*
    directional rotation :func:`score_aligned_cells` uses (the translation cannot
    touch a coordinate-free palette), so a geometry-backed C context shares the
    validated ``V_P`` membership of the state-bearing rows.
    """
    alignment = alignment_from_record(alignment)
    turns = alignment["rotation_quarter_turns"]
    translation = alignment["translation"]
    threshold = visible_reference.minimum_pixels if minimum_pixels is None else minimum_pixels
    visible = visible_reference.threshold(threshold)
    aligned = set(align_candidate_coordinates(cells, alignment))
    visible_coordinates = set(visible)
    reference_coordinates = set(reference)
    matched = visible_coordinates & aligned
    coordinate_truth = {
        coordinate: reference[coordinate] for coordinate in visible_coordinates
    }
    coverage = None
    conditional = None
    label = ""
    aligned_palette = None
    palette_cells = None
    if palette_states is not None:
        palette = align_palette_states(palette_states, alignment)
        palette_cells = frozenset(
            coordinate for coordinate in visible_coordinates
            if reference[coordinate] in palette)
        aligned_palette = frozenset(palette)
        coverage = len(palette_cells)
        conditional = coverage
        label = "V inter run palette (conditional vocabulary diagnostic)"
    return CellScore(
        space=space, checkpoint=checkpoint, kind=KIND_OF_SPACE[space],
        universe=UNIVERSE_OF_SPACE[space], denominator=denominator,
        alignment=alignment, visible_cells=len(visible_coordinates),
        matched=frozenset(matched), exact=frozenset(),
        aligned_cells=len(aligned),
        phantom_cells=len(aligned - reference_coordinates),
        hidden_inferred_cells=len((reference_coordinates - visible_coordinates) & aligned),
        hidden_exact_cells=0,
        palette_coverage=coverage, conditional_denominator=conditional,
        conditional_denominator_label=label, coordinate_truth=coordinate_truth,
        aligned_palette=aligned_palette, palette_cells=palette_cells)


def palette_coverage(
    *, reference: Mapping[BlockCoordinate, BlockState],
    visible_reference: VisibleReference, alignment: Mapping[str, Any],
    palette_states: Iterable[BlockState],
) -> tuple[int, int]:
    """The ``V inter palette`` conditional denominator and its covered count.

    Both numbers are drawn from the **same** reference-cell universe as ``V``;
    this is the vocabulary diagnostic, never the primary E denominator.
    ``palette_states`` is run-native and is mapped into the reference frame by
    :func:`align_palette_states` (directional rotation only; the translation cannot
    affect a coordinate-free palette) before membership is tested.
    """
    alignment = alignment_from_record(alignment)
    palette = align_palette_states(palette_states, alignment)
    visible = visible_reference.visible
    covered = sum(1 for state in visible.values() if state in palette)
    return covered, len(visible)


def cell_metric_rows(
    score: CellScore, *, kind: str, ceiling: dict[str, Any] | None = None,
    record_digest: str = "", checkpoint_reached: bool | None = None,
) -> tuple[AttributionRow, ...]:
    """The G/E rows for one scored cell checkpoint, in its declared universe."""
    if score.space is MetricSpace.GEOMETRY:
        rows = (
            _row_from_metric(score, "geometry_recall", score.geometry_recall,
                             kind=kind, ceiling=ceiling, record_digest=record_digest,
                             checkpoint_reached=checkpoint_reached),
            _row_from_metric(score, "geometry_precision", score.geometry_precision,
                             kind=kind, ceiling=None, record_digest=record_digest,
                             checkpoint_reached=checkpoint_reached),
        )
        return rows
    if score.space is MetricSpace.EXACT_STATE:
        rows = [
            _row_from_metric(score, "exact_state_recovery",
                             score.exact_state_over_visible, kind=kind,
                             ceiling=ceiling, record_digest=record_digest,
                             checkpoint_reached=checkpoint_reached),
        ]
        conditional = score.exact_state_over_palette
        if conditional is not None:
            rows.append(_row_from_metric(
                score, "exact_state_recovery_in_palette", conditional,
                kind=kind, ceiling=ceiling,
                record_digest=record_digest,
                denominator_label=score.conditional_denominator_label,
                conditional=True, checkpoint_reached=checkpoint_reached))
        return tuple(rows)
    raise AttributionRefusal("cell_metric_rows only scores a G or E checkpoint")


def palette_metric_rows(
    score: CellScore, *, kind: str, ceiling: dict[str, Any] | None = None,
    record_digest: str = "", checkpoint: str | None = None,
    checkpoint_reached: bool | None = None,
) -> tuple[AttributionRow, ...]:
    """The C rows for one checkpoint: palette membership over the same cell universe.

    ``palette_coverage`` is the unconditional availability share ``|V inter
    palette| / |V|``; ``palette_coverage_in_palette`` is the same count conditioned
    on its own denominator and is reported only as the labelled conditional
    vocabulary diagnostic.  The oracle ceiling for both is ``|V inter palette|``: a
    state the run palette lacks is unreachable at any candidate depth.
    """
    if score.palette_coverage is None or score.conditional_denominator is None:
        raise AttributionRefusal("palette coverage requires the run palette")
    coverage = metric_value(score.palette_coverage, score.visible_cells,
                            undefined_reason="no_visible_reference_cells")
    conditional = metric_value(score.palette_coverage, score.conditional_denominator,
                               undefined_reason="no_visible_cell_in_run_palette")
    ceiling = ceiling if ceiling is not None else _ceiling(
        score.palette_coverage, score.visible_cells)
    target = checkpoint if checkpoint is not None else score.checkpoint
    return (
        _row_from_metric(score, "palette_coverage", coverage, kind=kind,
                         ceiling=ceiling, record_digest=record_digest,
                         space=MetricSpace.CANDIDATE, checkpoint=target,
                         checkpoint_reached=checkpoint_reached),
        _row_from_metric(score, "palette_coverage_in_palette", conditional,
                         kind=kind, ceiling=ceiling, record_digest=record_digest,
                         denominator_label=score.conditional_denominator_label,
                         conditional=True, space=MetricSpace.CANDIDATE,
                         checkpoint=target, checkpoint_reached=checkpoint_reached),
    )


def exact_state_metric_rows(
    score: CellScore, *, kind: str, ceiling: dict[str, Any] | None = None,
    record_digest: str = "", checkpoint_reached: bool | None = None,
) -> tuple[AttributionRow, ...]:
    """The primary all-``V`` exact-state row plus its conditional palette row."""
    return cell_metric_rows(score, kind=kind, ceiling=ceiling,
                            record_digest=record_digest,
                            checkpoint_reached=checkpoint_reached)


def candidate_metric_rows(
    score: CellScore, *, depths: Sequence[int] = DEFAULT_CANDIDATE_DEPTHS,
    reachable_by_coordinate: Mapping[BlockCoordinate, Sequence[BlockState]] | None = None,
    kind: str = "availability_ceiling", record_digest: str = "",
    checkpoint: str | None = None, checkpoint_reached: bool | None = None,
) -> tuple[AttributionRow, ...]:
    """Candidate-availability rows over ``V`` for each depth ``K`` in ``depths``.

    The sweep reuses the accepted ``_candidate_availability`` semantics: the
    denominator is the pinned visible domain ``|V|``, a state the palette lacks is
    unreachable at every ``K`` and is charged to the palette, a cell with no
    candidate record is charged to geometry, and a cell whose truth state is
    present in the ranked list but loses the argmin is *available* (this is a
    ceiling, not a label).  ``reachable_by_coordinate[cell][:K]`` is the ordered
    candidate list the run actually retained, keyed by reference-frame coordinate
    (the same frame as ``CellScore.coordinate_truth``); a missing cell yields no
    candidates.

    The vocabulary ceiling is enforced **per cell**: the numerator is
    ``|{v in V_P : truth(v) in ranked(v)[:K]}|``, where ``V_P`` is the score's
    validated aligned palette membership (the visible cells whose truth is in the
    aligned producer palette), so the row can never exceed ``palette_coverage``.
    The score must carry that membership evidence; a C score without it is a typed
    refusal, never an assumed empty palette.  Every retained ranked candidate is
    validated against the aligned producer palette **before** any top-``K`` slice:
    an out-of-palette candidate contradicts the run's own palette, so it is a typed
    refusal rather than being filtered or reordered before the cut.
    """
    if score.space is not MetricSpace.CANDIDATE:
        raise AttributionRefusal("candidate metrics belong to the C space")
    if not depths or any(not _is_int(k) or k < 1 for k in depths):
        raise AttributionRefusal("candidate depths must be positive integers")
    if score.coordinate_truth is None:
        raise AttributionRefusal(
            "candidate availability needs the aligned truth keyed by candidate coordinate")
    if score.aligned_palette is None or score.palette_cells is None:
        raise AttributionRefusal(
            "candidate availability requires the aligned producer palette membership")
    ranked = {} if reachable_by_coordinate is None else reachable_by_coordinate
    for coordinate, states in ranked.items():
        for state in states:
            if state not in score.aligned_palette:
                raise AttributionRefusal(
                    "candidate ranking contradicts the aligned producer palette at "
                    f"{coordinate!r}")
    rows: list[AttributionRow] = []
    for k in depths:
        available = 0
        for coordinate, truth in score.coordinate_truth.items():
            if coordinate not in score.palette_cells:
                continue
            candidates = tuple(ranked.get(coordinate, ()))[: int(k)]
            if truth in candidates:
                available += 1
        rows.append(AttributionRow(
            checkpoint=checkpoint if checkpoint is not None else score.checkpoint,
            metric=f"candidate_available_k{k}",
            space=MetricSpace.CANDIDATE, kind=kind,
            universe=REFERENCE_CELLS_UNIVERSE,
            denominator="all frozen visible V (primary, >= 1 usable pixel)",
            reached=True, typed_reason="", record_digest=record_digest,
            metric_value=metric_value(available, score.visible_cells,
                                      undefined_reason="no_visible_reference_cells"),
            ceiling=None, conditional_denominator=None,
            conditional_denominator_label="",
            checkpoint_reached=checkpoint_reached))
    return tuple(rows)


def label_metric_row(
    *, checkpoint: str, space: MetricSpace, kind: str, numerator: int,
    denominator: int, denominator_label: str, record_digest: str = "",
    ceiling: dict[str, Any] | None = None,
    checkpoint_reached: bool | None = None,
) -> AttributionRow:
    """One typed label row (raw winner / promoted / decoded terminal) over ``V``."""
    if space not in CELL_SPACES:
        raise AttributionRefusal("a label row must use a cell space")
    if not _is_int(numerator) or not _is_int(denominator) or numerator < 0 \
            or denominator < 0 or numerator > denominator:
        raise AttributionRefusal("a label row needs a numerator within its denominator")
    return AttributionRow(
        checkpoint=checkpoint, metric="exact_state_recovery", space=space,
        kind=kind, universe=UNIVERSE_OF_SPACE[space],
        denominator=denominator_label, reached=True, typed_reason="",
        record_digest=record_digest,
        metric_value=metric_value(numerator, denominator,
                                  undefined_reason="no_visible_reference_cells"),
        ceiling=ceiling, conditional_denominator=None,
        conditional_denominator_label="", checkpoint_reached=checkpoint_reached)


def pair_precision(totals: Mapping[str, Any]) -> dict[str, Any]:
    """The accepted P-space precision value from an aggregated pair-totals record.

    The denominator is the *returned* pixel-pair identities; the numerator is the
    returned identities that landed on the same truth surface.  ``metric_value``
    yields ``null`` when nothing was returned, so an absent match pool is never a
    measured ``0.0``.  Availability/ceiling consumers can compare ``positive`` to
    ``truth_pairs_returned``, since returned-but-not-truth pairs are not a
    geometry failure.
    """
    returned = _pairs_count(totals, "returned")
    positive = _pairs_count(totals, "positive")
    return metric_value(positive, returned,
                        undefined_reason="no_returned_pixel_pairs")


def _pairs_count(totals: Mapping[str, Any], key: str) -> int:
    if not isinstance(totals, Mapping):
        raise AttributionRefusal("pair totals must be a mapping")
    value = totals.get(key)
    if not _is_int(value) or value < 0:
        raise AttributionRefusal(f"pair totals need a non-negative integer {key}")
    return value


@dataclass(frozen=True, slots=True)
class EntryMatchBinding:
    """The digest binding that alone makes a Route B entry match pool scoreable.

    ``matches_sha256`` is the digest rederived by replaying the *same* joint
    matcher (:data:`JOINT_MATCHER_SOURCE`) the entry used on the same returned
    session frames and cameras; ``accepted_matches_sha256`` is the accepted
    ``RelativeCameraEvidence.matches_sha256`` retained by the entry graph.  The
    binding holds only when the source is that exact joint matcher and the two
    digests are valid and equal.  Any other provenance - a missing or unbound
    digest, a mismatch, or an evaluator-only alternate matcher - is the typed
    :data:`ENTRY_MATCH_UNBOUND_REASON`.
    """

    source: str
    matches_sha256: str
    accepted_matches_sha256: str

    def __post_init__(self) -> None:
        if not _is_text(self.source) or not _is_text(self.matches_sha256) \
                or not _is_text(self.accepted_matches_sha256):
            raise AttributionRefusal(
                "an entry match binding needs three string fields")

    @property
    def bound(self) -> bool:
        """Whether this is the exact joint matcher's array digest, rederived."""
        return (self.source == JOINT_MATCHER_SOURCE
                and _is_digest(self.matches_sha256)
                and _is_digest(self.accepted_matches_sha256)
                and self.matches_sha256 == self.accepted_matches_sha256)

    @property
    def reason(self) -> str:
        """``""`` when bound, otherwise the typed unbound reason token."""
        return "" if self.bound else ENTRY_MATCH_UNBOUND_REASON

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "matches_sha256": self.matches_sha256,
            "accepted_matches_sha256": self.accepted_matches_sha256,
            "bound": self.bound,
        }


@dataclass(frozen=True, slots=True)
class EntryPairEvidence:
    """Digest-bound joint-match evidence for the Route B entry/B0 P row.

    ``totals`` is an aggregate (``truth_pairs``, ``truth_pairs_returned``,
    ``returned``, ``positive``) computed from the *same* joint matcher's match
    identities that :class:`EntryMatchBinding` binds to.  There is no way to
    construct this record without stating the source and both digests, and
    :func:`route_b_attribution` scores the entry row only when ``bound`` is true.
    An evaluator-only alternate matcher's aggregate is not this record's evidence
    and must be reported as a separate ``control`` diagnostic, never as achieved
    Route B P.
    """

    binding: EntryMatchBinding
    totals: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, EntryMatchBinding):
            raise AttributionRefusal("entry pair evidence needs an entry match binding")
        if not isinstance(self.totals, Mapping):
            raise AttributionRefusal("entry pair evidence totals must be a mapping")
        for key in ("truth_pairs", "truth_pairs_returned", "returned", "positive"):
            _pairs_count(self.totals, key)

    @property
    def bound(self) -> bool:
        return self.binding.bound

    @property
    def reason(self) -> str:
        return self.binding.reason

    def to_record(self) -> dict[str, Any]:
        return {
            "binding": self.binding.to_record(),
            "totals": dict(self.totals),
            "bound": self.bound,
        }


def pair_metric_rows(
    *, checkpoint: str, totals: Mapping[str, Any],
    binding: EntryMatchBinding, record_digest: str = "",
    checkpoint_reached: bool | None = None,
) -> tuple[AttributionRow, ...]:
    """The reachable P row: truth-pair recall (plus the accepted precision) over
    frozen frame-pair / pixel-pair identities, but only when digest-bound.

    ``totals`` is an already-computed aggregation of the joint matcher's pair
    scorecard (``truth_pairs``, ``truth_pairs_returned``, ``returned``,
    ``positive``).  The primary row is ``truth_pairs_returned / truth_pairs``:
    the share of jointly visible strict same-surface truth pixel pairs the run's
    admitted match pool actually returned.  It is never divided by ``|V|`` and is
    never substituted by entry connectivity.  A zero truth denominator yields
    ``metric_value``'s typed ``null`` with ``no_truth_keypoint_pairs``, never a
    measured zero.

    A ``totals`` aggregate may only be attached to the entry row when ``binding``
    proves it came from the *same* joint matcher whose array digest equals the
    accepted ``RelativeCameraEvidence.matches_sha256``; otherwise this raises an
    :class:`AttributionRefusal`.  That is the hard guard that keeps an
    evaluator-only alternate matcher's numbers out of the Route B attribution.

    ``checkpoint_reached=False`` states that the accepted frontier did **not**
    reach this checkpoint, so no metric may be scored at all: this also raises an
    :class:`AttributionRefusal` even when the aggregate is otherwise digest-bound,
    because a numeric P row with ``checkpoint_reached: false`` is a
    contradiction.  ``checkpoint_reached=None`` (no production-checkpoint context)
    and ``True`` (demonstrably reached) are both scoreable.
    """
    if not _is_token(checkpoint):
        raise AttributionRefusal("a pair row needs a checkpoint token")
    if not isinstance(binding, EntryMatchBinding):
        raise AttributionRefusal("a pair row needs an entry match binding")
    if checkpoint_reached is False:
        raise AttributionRefusal(
            "a pair row cannot be scored when the accepted frontier did not reach "
            "the checkpoint (entry_unreached)")
    if not binding.bound:
        raise AttributionRefusal(
            "an entry P row may only be scored from a digest-bound joint match set: "
            + ENTRY_MATCH_UNBOUND_REASON)
    precision = pair_precision(totals)
    recall = metric_value(
        _pairs_count(totals, "truth_pairs_returned"),
        _pairs_count(totals, "truth_pairs"),
        undefined_reason="no_truth_keypoint_pairs")
    denominator = "jointly visible strict same-surface truth pixel pairs"
    return (
        AttributionRow(
            checkpoint=checkpoint, metric="truth_pair_recall",
            space=MetricSpace.PAIR_CORRESPONDENCE, kind="label",
            universe=PIXEL_PAIRS_UNIVERSE, denominator=denominator, reached=True,
            typed_reason="", record_digest=record_digest, metric_value=recall,
            ceiling=None, conditional_denominator=None,
            conditional_denominator_label=precision["undefined_reason"] or "",
            checkpoint_reached=checkpoint_reached),
        AttributionRow(
            checkpoint=checkpoint, metric="positive_precision",
            space=MetricSpace.PAIR_CORRESPONDENCE, kind="availability",
            universe=PIXEL_PAIRS_UNIVERSE,
            denominator="returned pixel-pair identities", reached=True,
            typed_reason="", record_digest=record_digest, metric_value=precision,
            ceiling=None, conditional_denominator=None,
            conditional_denominator_label="",
            checkpoint_reached=checkpoint_reached),
    )


def _row_from_metric(
    score: CellScore, metric: str, value: dict[str, Any], *, kind: str,
    ceiling: dict[str, Any] | None, record_digest: str,
    denominator_label: str = "", conditional: bool = False,
    space: MetricSpace | None = None, checkpoint: str | None = None,
    checkpoint_reached: bool | None = None,
) -> AttributionRow:
    resolved = score.space if space is None else space
    denominator = denominator_label or score.denominator
    return AttributionRow(
        checkpoint=checkpoint if checkpoint is not None else score.checkpoint,
        metric=metric, space=resolved, kind=kind,
        universe=UNIVERSE_OF_SPACE[resolved], denominator=denominator, reached=True,
        typed_reason="", record_digest=record_digest, metric_value=value,
        ceiling=ceiling, conditional_denominator=(
            score.conditional_denominator if conditional else None),
        conditional_denominator_label=(
            score.conditional_denominator_label if conditional else ""),
        checkpoint_reached=checkpoint_reached)


# ---------------------------------------------------------------------------
# Rows, transitions and sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AttributionRow:
    """One labelled metric row: a metric in its declared universe or a typed null.

    ``reached`` is the compatibility flag for *metric availability*:
    ``reached=False`` is the fail-closed case where ``metric_value``, ``ceiling``
    and ``conditional_denominator`` are all ``None`` and ``typed_reason`` is
    non-empty, so a missing record can never be read as a measured zero.

    ``checkpoint_reached`` is a **separate** serialized statement of whether the
    production checkpoint this row is attributed to was reached (A1 frontier
    ``reachable`` semantics), independent of whether a metric could be scored.  It
    is ``True`` for a reached-but-unscorable checkpoint (a typed null metric), is
    ``False`` for a checkpoint the run never reached, and is ``None`` when the row
    names no production checkpoint (an aggregate/evaluator-only row) or names a
    substage A1 does not checkpoint separately and no carried record proves it
    (Route B ``promoted``/``terminal``), because that absence may mean
    unattempted/unknown and is never inferred as ``False``.  Presence is explicit
    record existence (``is not None``): a carried empty substage record proves reach
    even though its metric is unavailable.  A reached checkpoint
    never implies a measured metric and a null metric never implies the checkpoint
    did not run.
    """

    checkpoint: str
    metric: str
    space: MetricSpace
    kind: str
    universe: str
    denominator: str
    reached: bool
    typed_reason: str = ""
    record_digest: str = ""
    metric_value: dict[str, Any] | None = None
    ceiling: dict[str, Any] | None = None
    conditional_denominator: int | None = None
    conditional_denominator_label: str = ""
    checkpoint_reached: bool | None = None

    def __post_init__(self) -> None:
        if not _is_token(self.checkpoint) or not _is_token(self.metric):
            raise AttributionRefusal("a row needs a checkpoint and a metric token")
        if not isinstance(self.space, MetricSpace):
            raise AttributionRefusal("a row needs a MetricSpace")
        if not _is_token(self.kind):
            raise AttributionRefusal("a row needs a kind token")
        if not _is_token(self.universe):
            raise AttributionRefusal("a row needs a fixed universe token")
        if type(self.reached) is not bool:
            raise AttributionRefusal("a row reached flag must be a boolean")
        if self.checkpoint_reached is not None \
                and type(self.checkpoint_reached) is not bool:
            raise AttributionRefusal(
                "a row checkpoint_reached flag must be a boolean or null")
        if self.record_digest and not _is_digest(self.record_digest):
            raise AttributionRefusal("a row record digest must be lowercase SHA-256")
        if self.space is MetricSpace.CAMERA_DIAGNOSTIC and self.denominator != "none":
            raise AttributionRefusal(
                "a camera diagnostic has no denominator and no cell-loss equation")
        if self.reached:
            if self.metric_value is None or not isinstance(self.metric_value, Mapping):
                raise AttributionRefusal("a reached row needs a metric record")
            if "value" not in self.metric_value:
                raise AttributionRefusal("a metric record must carry a value field")
            if self.metric_value["value"] is not None:
                _finite(self.metric_value["value"])
            if self.typed_reason:
                raise AttributionRefusal("a reached row carries no typed reason")
        else:
            if not _is_token(self.typed_reason):
                raise AttributionRefusal("an unreached row needs a typed reason token")
            if self.metric_value is not None:
                raise AttributionRefusal(
                    "an unreached row must not carry a metric; a zero is never emitted")
            if self.ceiling is not None or self.conditional_denominator is not None:
                raise AttributionRefusal(
                    "an unreached row carries no ceiling or conditional denominator")
        if self.ceiling is not None:
            for key in ("numerator", "denominator", "value"):
                if key not in self.ceiling:
                    raise AttributionRefusal(f"a ceiling needs a {key} field")
            if self.ceiling["value"] is not None:
                _finite(self.ceiling["value"])

    @property
    def value(self) -> float | None:
        return None if self.metric_value is None else self.metric_value["value"]

    @property
    def numerator(self) -> int | None:
        return None if self.metric_value is None else self.metric_value["numerator"]

    @property
    def denominator_count(self) -> int | None:
        return None if self.metric_value is None else self.metric_value["denominator"]

    @property
    def undefined_reason(self) -> str | None:
        return None if self.metric_value is None else self.metric_value["undefined_reason"]

    @property
    def is_accuracy(self) -> bool:
        """Whether this row is a measured count over a declared denominator.

        Availability, ceiling and policy rows are explicitly *not* accuracy.
        """
        return self.reached and self.kind in ("label", "primary_visibility")

    def to_record(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "metric": self.metric,
            "space": self.space.value,
            "kind": self.kind,
            "universe": self.universe,
            "denominator": self.denominator,
            "reached": self.reached,
            "checkpoint_reached": self.checkpoint_reached,
            "typed_reason": self.typed_reason,
            "record_digest": self.record_digest,
            "metric_value": None if self.metric_value is None else dict(self.metric_value),
            "ceiling": None if self.ceiling is None else dict(self.ceiling),
            "conditional_denominator": self.conditional_denominator,
            "conditional_denominator_label": self.conditional_denominator_label,
            "is_accuracy": self.is_accuracy,
        }


def _ceiling(numerator: int, denominator: int) -> dict[str, Any]:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise AttributionRefusal("a ceiling numerator must lie within its denominator")
    return metric_value(numerator, denominator, undefined_reason="no_visible_reference_cells")


@dataclass(frozen=True, slots=True)
class AttributionTransition:
    """One reconciled gain/loss step between comparable checkpoints in one universe.

    ``next = previous + gained - lost`` holds inside one universe.  For a
    cell-level exact-state transition the loss is partitioned into the disjoint
    ``geometry_lost`` and ``state_lost`` sets; ``lost`` must equal their sum and a
    cell may not appear in both.  An O or P transition carries no cell partition,
    and a camera diagnostic carries no transition at all.
    """

    universe: str
    previous_checkpoint: str
    next_checkpoint: str
    space: MetricSpace
    previous: int | None
    gained: int | None
    lost: int | None
    net: int | None
    geometry_lost: int | None = None
    state_lost: int | None = None

    def __post_init__(self) -> None:
        if not _is_token(self.universe) or not _is_token(self.previous_checkpoint) \
                or not _is_token(self.next_checkpoint):
            raise AttributionRefusal("a transition needs a universe and two checkpoints")
        if self.previous_checkpoint == self.next_checkpoint:
            raise AttributionRefusal("a transition needs two distinct checkpoints")
        if not isinstance(self.space, MetricSpace):
            raise AttributionRefusal("a transition needs one MetricSpace")
        if self.space is MetricSpace.CAMERA_DIAGNOSTIC:
            raise AttributionRefusal("a camera diagnostic carries no transition")
        if self.universe != UNIVERSE_OF_SPACE[self.space]:
            raise AttributionRefusal("a transition must name its space's universe")
        for name in ("previous", "gained", "lost"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                raise AttributionRefusal(f"transition {name} must be a non-negative integer")
        if self.net is not None and (not _is_int(self.net)):
            raise AttributionRefusal("transition net must be an integer or null")
        if self.previous is None or self.gained is None or self.lost is None:
            raise AttributionRefusal(
                "a measured transition carries previous, gained and lost")
        if self.net is not None and self.net != self.gained - self.lost:
            raise AttributionRefusal("transition net must equal gained - lost")
        if self.geometry_lost is not None or self.state_lost is not None:
            if self.space is not MetricSpace.EXACT_STATE:
                raise AttributionRefusal(
                    "only a cell-level exact-state transition partitions its loss")
            if self.geometry_lost is None or self.state_lost is None:
                raise AttributionRefusal(
                    "an exact-state partition needs both geometry_lost and state_lost")
            if self.geometry_lost < 0 or self.state_lost < 0:
                raise AttributionRefusal("a partitioned loss must be non-negative")
            if self.geometry_lost + self.state_lost != self.lost:
                raise AttributionRefusal(
                    "geometry_lost and state_lost must partition lost exactly")
        if self.net != self.gained - self.lost:
            raise AttributionRefusal("transition net must equal gained - lost")

    def to_record(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "space": self.space.value,
            "previous_checkpoint": self.previous_checkpoint,
            "next_checkpoint": self.next_checkpoint,
            "previous": self.previous,
            "gained": self.gained,
            "lost": self.lost,
            "net": self.net,
            "geometry_lost": self.geometry_lost,
            "state_lost": self.state_lost,
            "identity": "next = previous + gained - lost",
            "loss_partition": (
                "geometry_lost + state_lost = lost (disjoint cell sets)"
                if self.geometry_lost is not None else None),
        }


def cell_transition(
    previous: CellScore, following: CellScore, *,
    partition_loss: bool = True,
) -> AttributionTransition:
    """Reconcile one same-universe cell transition.

    The reconciled success count is **this space's own primary count**: geometry
    ``G`` transitions reconcile ``V inter candidate`` (``matched``) cells, and
    exact-state ``E`` transitions reconcile the exact-state cells (``exact``), so
    ``next = previous + gained - lost`` holds on the metric the section actually
    reports.

    ``partition_loss`` is only meaningful for a cell-level exact-state transition:
    ``geometry_lost`` is a cell that matched at the earlier checkpoint and has no
    aligned candidate at the later one; ``state_lost`` is a cell that still has a
    candidate but whose label stopped matching.  The two sets are disjoint by
    construction and their union is ``lost``.  A ``G`` transition is unpartitioned
    (both endpoints already *are* the geometry count), so it never carries a
    ``geometry_lost``/``state_lost`` pair and never raises for lacking one.
    """
    if previous.checkpoint == following.checkpoint:
        raise AttributionRefusal("a transition needs two distinct checkpoints")
    if previous.universe != following.universe:
        raise AttributionRefusal("a transition may only join one comparable universe")
    if previous.space is not following.space:
        raise AttributionRefusal("a transition may only join one labelled space")
    if following.space is MetricSpace.GEOMETRY:
        # Both endpoints are geometry scores, so both success-set memberships *are*
        # ``matched``: an unpartitioned, same-universe G reconciliation.
        previous_cells = previous.matched
        following_cells = following.matched
    else:
        previous_cells = previous.exact
        following_cells = following.exact
    gained = following_cells - previous_cells
    lost = previous_cells - following_cells
    geometry_lost = None
    state_lost = None
    if partition_loss:
        if following.space is not MetricSpace.EXACT_STATE:
            raise AttributionRefusal(
                "only a cell-level exact-state transition partitions its loss")
        geometry_lost = len({cell for cell in lost if cell not in following.matched})
        state_lost = len({cell for cell in lost if cell in following.matched})
    return AttributionTransition(
        universe=previous.universe, space=previous.space,
        previous_checkpoint=previous.checkpoint,
        next_checkpoint=following.checkpoint,
        previous=len(previous_cells), gained=len(gained), lost=len(lost),
        net=len(gained) - len(lost), geometry_lost=geometry_lost,
        state_lost=state_lost)


@dataclass(frozen=True, slots=True)
class AttributionSection:
    """One labelled section (one space) of the attribution record.

    Sections are printed separately and are never summed: a G section, an O
    section, a P section, a C section and an E section answer five different
    questions.
    """

    space: MetricSpace
    universe: str
    rows: tuple[AttributionRow, ...] = ()
    transitions: tuple[AttributionTransition, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.space, MetricSpace):
            raise AttributionRefusal("a section needs one MetricSpace")
        if self.universe != UNIVERSE_OF_SPACE[self.space]:
            raise AttributionRefusal("a section must name its space's universe")
        if not isinstance(self.rows, tuple) or any(
                not isinstance(row, AttributionRow) for row in self.rows):
            raise AttributionRefusal("section rows must be attribution rows")
        if any(row.space is not self.space for row in self.rows):
            raise AttributionRefusal("a section may only carry its own space's rows")
        if not isinstance(self.transitions, tuple) or any(
                not isinstance(item, AttributionTransition) for item in self.transitions):
            raise AttributionRefusal("section transitions must be transition records")
        if any(item.space is not self.space for item in self.transitions):
            raise AttributionRefusal("a section may only carry its own space's transitions")

    def to_record(self) -> dict[str, Any]:
        return {
            "space": self.space.value,
            "universe": self.universe,
            "rows": [row.to_record() for row in self.rows],
            "transitions": [item.to_record() for item in self.transitions],
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ImageOnlyAttribution:
    """One route's canonical A2 attribution: labelled sections plus policy rates.

    ``sections`` are the five separate spaces; ``policy_rates`` are completion and
    refusal counts by terminal stage/status (labelled policy rates, never pooled
    accuracy); ``frontier`` is the A1 frontier the scored rows were built over.
    """

    route: Route
    sections: tuple[AttributionSection, ...]
    frontier: ImageOnlyFrontier
    visible_reference: VisibleReference
    policy_rates: Mapping[str, Any] | None = None
    caveats: tuple[str, ...] = ()
    schema: str = ATTRIBUTION_SCHEMA
    producer: str = ATTRIBUTION_PRODUCER

    def __post_init__(self) -> None:
        if self.schema != ATTRIBUTION_SCHEMA or self.producer != ATTRIBUTION_PRODUCER:
            raise AttributionRefusal("attribution schema or producer is unsupported")
        if not isinstance(self.route, Route):
            raise AttributionRefusal("an attribution needs a Route")
        if not isinstance(self.frontier, ImageOnlyFrontier) \
                or self.frontier.route is not self.route:
            raise AttributionRefusal("an attribution needs its own route's frontier")
        if not isinstance(self.visible_reference, VisibleReference):
            raise AttributionRefusal("an attribution needs a visible reference")
        if not isinstance(self.sections, tuple) or any(
                not isinstance(section, AttributionSection) for section in self.sections):
            raise AttributionRefusal("attribution sections must be section records")
        spaces = [section.space for section in self.sections]
        if len(set(spaces)) != len(spaces):
            raise AttributionRefusal("a space may appear in at most one section")
        if self.policy_rates is not None and not isinstance(self.policy_rates, Mapping):
            raise AttributionRefusal("policy rates must be a mapping or null")

    @property
    def section(self) -> dict[MetricSpace, AttributionSection]:
        return {item.space: item for item in self.sections}

    @property
    def frontier_stop(self) -> FrontierCheckpoint | None:
        return self.frontier.frontier_stop

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "producer": self.producer,
            "route": self.route.value,
            "visible_reference": self.visible_reference.to_record(),
            "sections": [section.to_record() for section in self.sections],
            "policy_rates": None if self.policy_rates is None else dict(self.policy_rates),
            "frontier": self.frontier.to_record(),
            "frontier_stop": (None if self.frontier_stop is None else {
                "stage": self.frontier_stop.stage,
                "kind": self.frontier_stop.kind.value,
                "universe": self.frontier_stop.universe.value,
                "reason": self.frontier_stop.reason,
            }),
            "caveats": list(self.caveats),
            "notes": (
                "five labelled spaces, fixed universes, typed unavailable; the first "
                "stop is a frontier indicator, never a causal attribution of loss"),
        }


def attribution_to_record(value: ImageOnlyAttribution) -> dict[str, Any]:
    if not isinstance(value, ImageOnlyAttribution):
        raise AttributionRefusal("attribution_to_record needs an attribution record")
    return value.to_record()


def attribution_digest(value: ImageOnlyAttribution | Mapping[str, Any]) -> str:
    payload = value.to_record() if isinstance(value, ImageOnlyAttribution) else value
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def attribution_unavailable(reason: str, *, stage: str = "", detail: str = "") -> dict[str, Any]:
    """One typed fail-closed section for a route that could not be measured."""
    if not _is_token(reason):
        raise AttributionRefusal("an unavailable attribution needs a typed reason token")
    return {
        "available": False,
        "stage": stage,
        "reason": reason,
        "detail": detail,
        "metric": None,
    }


def completion_and_refusal_rates(
    *, planned: int, reached: int, terminal_stage: str, frontier_stop: str | None,
) -> dict[str, Any]:
    """Per-route completion and refusal counts, as labelled policy rates.

    These are the share of planned checkpoints a run reached and the share it
    stopped at.  They are policy outcomes, **never** accuracy, and they are
    reported by terminal stage/status rather than pooled.
    """
    if not _is_int(planned) or planned < 0 or not _is_int(reached) \
            or reached < 0 or reached > planned:
        raise AttributionRefusal("completion counts must satisfy 0 <= reached <= planned")
    completion = metric_value(reached, planned, undefined_reason="no_planned_checkpoints")
    refusal = metric_value(planned - reached, planned,
                           undefined_reason="no_planned_checkpoints")
    return {
        "planned_checkpoints": planned,
        "reached_checkpoints": reached,
        "completion_rate": completion,
        "refusal_rate": refusal,
        "terminal_stage": terminal_stage,
        "frontier_stop": frontier_stop,
        "accounting": (
            "policy rates only: the share of planned checkpoints reached and the share "
            "stopped at; this is never accuracy and is never pooled across routes"),
    }


# ---------------------------------------------------------------------------
# Building a route attribution from a frontier + records
# ---------------------------------------------------------------------------

def _frontier_rows(frontier: ImageOnlyFrontier, expected: Sequence[str]) -> dict[str, FrontierCheckpoint]:
    rows = {item.stage: item for item in frontier.checkpoints}
    missing = [stage for stage in expected if stage not in rows]
    if missing:
        raise AttributionRefusal(
            f"the frontier does not carry the {missing[0]} checkpoint")
    return rows


def _section(space: MetricSpace, rows: Sequence[AttributionRow],
             transitions: Sequence[AttributionTransition], notes: str) -> AttributionSection:
    ordered_transitions = [item for item in transitions if item.space is space]
    return AttributionSection(
        space=space, universe=UNIVERSE_OF_SPACE[space], rows=tuple(rows),
        transitions=tuple(ordered_transitions), notes=notes)


def _terminal_stage(frontier: ImageOnlyFrontier) -> str:
    """The run's accepted terminal stage, never the last planned checkpoint."""
    if frontier.terminal_policy and "stage" in frontier.terminal_policy:
        return str(frontier.terminal_policy["stage"])
    return frontier.terminal_status


def route_a_attribution(
    *,
    frontier: ImageOnlyFrontier,
    visible_reference: VisibleReference,
    reference: Mapping[BlockCoordinate, BlockState],
    alignment: Mapping[str, Any],
    facade: Mapping[BlockCoordinate, BlockState] | None = None,
    palette: Sequence[BlockState] | None = None,
    palette_terminal: Mapping[BlockCoordinate, BlockState] | None = None,
    pair_totals: Mapping[str, Any] | None = None,
    visibility_minimums: Sequence[int] = REPORTED_VISIBILITY_MINIMUMS,
    include_palette_loss_partition: bool = False,
) -> ImageOnlyAttribution:
    """Score Route A's reachable checkpoints against the frozen visible truth.

    ``facade`` is the run's facade grid keyed by the run's own lattice coordinates;
    ``palette`` is the palette the run actually used (default 8-state or a normal
    CLI JAR set); ``palette_terminal`` is the decoded ``.litematic`` round trip.
    Route A's unresolved metric scale is reported as a score-reducing caveat on
    every G row, **not** as an unavailable geometry.  No camera, manifest,
    reference path, alignment or mask may enter here except as already-loaded
    evaluator objects supplied by the evaluation runner after reconstruction.

    ``include_palette_loss_partition`` is **default off** and additive: when true,
    exactly the reviewed five rows from
    :func:`in_palette_loss_partition_rows` are appended at the ``palette_label``
    checkpoint, and every pre-existing row, denominator and canonical payload is
    left byte-for-byte unchanged.  Requesting it without the reached
    ``palette_label`` exact-state score and its aligned run palette is a typed
    :class:`AttributionRefusal`, never an assumed empty ``V_P``.
    """
    if frontier.route is not Route.A:
        raise AttributionRefusal("route_a_attribution needs a route A frontier")
    if type(include_palette_loss_partition) is not bool:
        raise AttributionRefusal(
            "include_palette_loss_partition must be a boolean")
    if palette is not None and not tuple(palette):
        raise AttributionRefusal("a Route A palette cannot be empty")
    checkpoints = _frontier_rows(frontier, ("admission", "camera", "facade_grid",
                                            "palette_label", "terminal"))
    rows: list[AttributionRow] = []
    transitions: list[AttributionTransition] = []
    scores: dict[str, CellScore] = {}

    facade_present = facade is not None and bool(facade)
    if facade_present:
        facade_score = score_aligned_cells(
            reference=reference, candidate=facade, visible_reference=visible_reference,
            alignment=alignment, checkpoint="facade_grid",
            space=MetricSpace.GEOMETRY,
            palette_states=palette,
            minimum_pixels=PRIMARY_VISIBILITY_MINIMUM_PIXELS)
        scores["facade_grid"] = facade_score
        # The oracle ceiling for Route A geometry is the facade's own grid under
        # the permitted alignment; a geometry row has no palette ceiling.
        rows.extend(cell_metric_rows(
            facade_score, kind="geometry",
            ceiling=_ceiling(len(facade_score.matched), facade_score.visible_cells),
            record_digest=checkpoints["facade_grid"].record_digest,
            checkpoint_reached=checkpoints["facade_grid"].reachable))
        if palette is not None:
            rows.extend(palette_metric_rows(
                facade_score, kind="availability", checkpoint="palette_label",
                ceiling=_ceiling(facade_score.palette_coverage or 0,
                                 facade_score.visible_cells),
                record_digest=checkpoints["palette_label"].record_digest,
                checkpoint_reached=checkpoints["palette_label"].reachable))
            label_score = CellScore(
                space=MetricSpace.EXACT_STATE, checkpoint="palette_label",
                kind=RecordKind.EXACT_STATE, universe=REFERENCE_CELLS_UNIVERSE,
                denominator=facade_score.denominator, alignment=facade_score.alignment,
                visible_cells=facade_score.visible_cells,
                matched=facade_score.matched, exact=facade_score.exact,
                aligned_cells=facade_score.aligned_cells,
                phantom_cells=facade_score.phantom_cells,
                hidden_inferred_cells=facade_score.hidden_inferred_cells,
                hidden_exact_cells=facade_score.hidden_exact_cells,
                palette_coverage=facade_score.palette_coverage,
                conditional_denominator=facade_score.conditional_denominator,
                conditional_denominator_label=facade_score.conditional_denominator_label,
                coordinate_truth=facade_score.coordinate_truth,
                aligned_palette=facade_score.aligned_palette,
                palette_cells=facade_score.palette_cells)
            scores["palette_label"] = label_score
            rows.extend(exact_state_metric_rows(
                label_score, kind="label",
                ceiling=_ceiling(len(facade_score.matched), facade_score.visible_cells),
                record_digest=checkpoints["palette_label"].record_digest,
                checkpoint_reached=checkpoints["palette_label"].reachable))
            if include_palette_loss_partition:
                # Default-off additive diagnostic: the reviewed five rows read the
                # label score's own frozen, already-aligned palette memberships and
                # are appended beside the existing palette_label E rows.  Every
                # pre-existing row is left untouched.
                rows.extend(in_palette_loss_partition_rows(
                    label_score,
                    record_digest=checkpoints[
                        PALETTE_LOSS_PARTITION_CHECKPOINT].record_digest,
                    checkpoint=PALETTE_LOSS_PARTITION_CHECKPOINT,
                    checkpoint_reached=checkpoints[
                        PALETTE_LOSS_PARTITION_CHECKPOINT].reachable))
    else:
        reason = checkpoints["facade_grid"].reason or "no_visible_blocks"
        for space, metric, source in (
                (MetricSpace.GEOMETRY, "geometry_recall", "facade_grid"),
                (MetricSpace.GEOMETRY, "geometry_precision", "facade_grid"),
                (MetricSpace.CANDIDATE, "palette_coverage", "palette_label"),
                (MetricSpace.EXACT_STATE, "exact_state_recovery", "palette_label")):
            rows.append(_unreached_row(
                checkpoint=("facade_grid" if space is MetricSpace.GEOMETRY
                            else "palette_label"),
                metric=metric, space=space, reason=reason,
                checkpoint_reached=checkpoints[source].reachable))

    if include_palette_loss_partition and "palette_label" not in scores:
        # No reached palette_label exact-state score means no aligned palette
        # evidence exists; an explicit request is a typed refusal rather than a
        # silently empty V_P or a fabricated partition.
        raise AttributionRefusal(
            "the requested in-palette loss partition requires the reached "
            "palette_label exact-state score with the aligned run palette "
            "(palette_label_palette_evidence_unavailable)")

    if palette_terminal is not None:
        terminal_score = score_aligned_cells(
            reference=reference, candidate=palette_terminal,
            visible_reference=visible_reference, alignment=alignment,
            checkpoint="terminal", space=MetricSpace.EXACT_STATE,
            palette_states=palette)
        scores["terminal"] = terminal_score
        rows.extend(exact_state_metric_rows(
            terminal_score, kind="label",
            ceiling=_ceiling(len(terminal_score.matched), terminal_score.visible_cells),
            record_digest=checkpoints["terminal"].record_digest,
            checkpoint_reached=checkpoints["terminal"].reachable))
    else:
        rows.append(_unreached_row(
            checkpoint="terminal", metric="exact_state_recovery",
            space=MetricSpace.EXACT_STATE,
            reason=checkpoints["terminal"].reason or "export_unavailable",
            checkpoint_reached=checkpoints["terminal"].reachable))

    # The one comparable Route A cell transition: palette labels -> decoded export.
    if "palette_label" in scores and "terminal" in scores:
        transitions.append(cell_transition(scores["palette_label"], scores["terminal"]))

    # The camera row is a diagnostic: no denominator and no transition.
    camera = checkpoints["camera"]
    camera_row = AttributionRow(
        checkpoint="camera", metric="camera_diagnostic", space=MetricSpace.CAMERA_DIAGNOSTIC,
        kind="camera", universe="none", denominator="none",
        reached=camera.reachable, typed_reason="" if camera.reachable else camera.reason,
        record_digest=camera.record_digest, checkpoint_reached=camera.reachable,
        metric_value=(
            {"value": None, "numerator": None, "denominator": None,
             "undefined_reason": None, "diagnostic": True}
            if camera.reachable else None))

    all_rows = rows + [camera_row]
    # A section may only carry its own space's rows: the camera diagnostic lives in
    # its own section and is never mixed into a cell section.
    geometry_rows = [row for row in rows if row.space is MetricSpace.GEOMETRY]
    candidate_rows = [row for row in rows if row.space is MetricSpace.CANDIDATE]
    exact_rows = [row for row in rows if row.space is MetricSpace.EXACT_STATE]
    sections = (
        _section(MetricSpace.GEOMETRY, geometry_rows, transitions,
                 "geometry over frozen reference cells; unresolved metric scale reduces "
                 "the score and is reported as a caveat, never as unavailable"),
        _section(MetricSpace.OWNERSHIP, (), (),
                 "Route A emits no (cell, slot, frame) face-ownership record; "
                 "ownership is not scored and is never substituted by geometry"),
        _section(MetricSpace.PAIR_CORRESPONDENCE, (), (),
                 "Route A is single-view and emits no pair correspondence; the P "
                 "universe is frame-pair/pixel-pair identities, not reference cells"),
        _section(MetricSpace.CANDIDATE, candidate_rows, (),
                 "palette availability over the same visible cell universe as G and E; "
                 "availability, never accuracy"),
        _section(MetricSpace.EXACT_STATE, exact_rows, transitions,
                 "primary denominator is all frozen visible V; V inter palette is a "
                 "separate conditional vocabulary row"),
        _section(MetricSpace.CAMERA_DIAGNOSTIC, (camera_row,), (),
                 "camera diagnostics carry no cell denominator and no cell-loss equation"),
    )
    if pair_totals is not None:
        raise AttributionRefusal(
            "Route A carries no pair correspondence record to score")
    policy = completion_and_refusal_rates(
        planned=len(frontier.checkpoints),
        reached=len(frontier.reached_stages),
        terminal_stage=_terminal_stage(frontier),
        frontier_stop=None if frontier.frontier_stop is None
        else frontier.frontier_stop.stage)
    return ImageOnlyAttribution(
        route=Route.A, sections=sections, frontier=frontier,
        visible_reference=visible_reference, policy_rates=policy,
        caveats=(
            "Route A's camera scale_known=False: the facade's metric scale is "
            "unresolved, so the geometry score is reduced by an unknown scale and is "
            "reported with that caveat rather than as unavailable",
            "alignment is one global integer translation plus lattice Y-quarter-turn; "
            "no scale fit, reflection or per-cell alignment is permitted",
        ))


def _unreached_row(*, checkpoint: str, metric: str, space: MetricSpace,
                   reason: str, checkpoint_reached: bool | None = None) -> AttributionRow:
    """One fail-closed metric row: no metric value, never a measured zero.

    ``reached=False`` here means *no metric record was produced*.  The production
    checkpoint's own reach is stated separately by ``checkpoint_reached``: ``False``
    for a checkpoint the run never reached, ``True`` for a reached checkpoint whose
    record was carried but not scoreable by this row, and ``None`` when the row
    names no production checkpoint.
    """
    if not _is_text(reason):
        reason = "record_absent"
    if not _is_token(reason):
        raise AttributionRefusal(
            "an unreached row needs a single typed reason token")
    return AttributionRow(
        checkpoint=checkpoint, metric=metric, space=space, kind="unavailable",
        universe=UNIVERSE_OF_SPACE[space],
        denominator="all frozen visible V (primary, >= 1 usable pixel)"
        if space in CELL_SPACES else UNIVERSE_OF_SPACE[space],
        reached=False, typed_reason=reason, checkpoint_reached=checkpoint_reached)


def route_b_attribution(
    *,
    frontier: ImageOnlyFrontier,
    visible_reference: VisibleReference,
    reference: Mapping[BlockCoordinate, BlockState],
    alignment: Mapping[str, Any] | None,
    placement: Mapping[BlockCoordinate, BlockState] | GeometryCellEvidence | None = None,
    volume_cells: Mapping[BlockCoordinate, BlockState] | GeometryCellEvidence | None = None,
    promoted_cells: Mapping[BlockCoordinate, BlockState] | None = None,
    decoded_terminal: Mapping[BlockCoordinate, BlockState] | None = None,
    palette: Sequence[BlockState] | None = None,
    proposals: Mapping[BlockCoordinate, Any] | None = None,
    candidate_depths: Sequence[int] = DEFAULT_CANDIDATE_DEPTHS,
    entry_pair_evidence: EntryPairEvidence | None = None,
    entry_pair_unavailable_reason: str = "",
) -> ImageOnlyAttribution:
    """Score every reachable Route B checkpoint against the frozen visible truth.

    ``placement`` / ``volume_cells`` / ``promoted_cells`` / ``decoded_terminal`` are
    the run's own records at the corresponding checkpoints.  Placement and volume
    are **geometry-only** production records (``ResolvedFacePlacement`` cells and
    ``PlaneVolumeResult`` cells); a caller passes them as a
    :class:`GeometryCellEvidence` coordinate record, and this builder scores their
    G rows through :func:`score_aligned_geometry_coordinates` (coordinates only, no
    fabricated ``BlockState`` and no ``{coordinate: None}`` map).  A state-bearing
    record (for example a synthetic caller or the promoted/decoded exact-state
    records) is passed as a coordinate-to-``BlockState`` mapping and is scored
    through :func:`score_aligned_cells`.  ``proposals`` maps a cell to a
    ``FaceStateProposal``-like record exposing ``candidate_costs``, ``state`` and
    ``status``.  Every one of those records is **run-native**: this builder applies
    the one permitted alignment (``alignment``) exactly once, through
    :func:`align_candidate_coordinates` / :func:`align_candidate` for the cell
    records and :func:`align_candidate_rankings` for the ranked candidate lists, so a
    caller must never pre-align the same records (a doubly transformed map is a
    different, unaligned map and is never scored).  A stage whose record was not
    carried is typed unavailable and contributes no zero.  The latent
    (non-carried) checkpoints are reported as explicit unavailable rows so the
    record shape is complete without fabricating a measurement for a stage the run
    never reached.

    The C-space palette and candidate-availability rows are built on a genuine
    ``MetricSpace.CANDIDATE`` ``CellScore`` over the same frozen visible-cell
    universe as the G/E rows (never on a G-space base score), and each proposal's
    ranked candidates are transformed with their coordinate so the join against the
    reference-frame truth is exactly once.

    ``entry_pair_evidence`` is the digest-bound joint-match aggregate for the
    reachable ``entry``/B0 checkpoint.  The entry row is scored in the P space by
    :func:`pair_metric_rows` only when ``entry_pair_evidence.bound`` proves the
    aggregate came from the same joint matcher and its rederived match digest
    equals the accepted ``RelativeCameraEvidence.matches_sha256``.  Otherwise -
    including the evaluator-only alternate matcher that must be reported as a
    separate ``control`` - the entry row is typed unavailable with
    :data:`ENTRY_MATCH_UNBOUND_REASON` (or ``entry_pair_unavailable_reason`` when
    supplied) and no P row ever emits a fabricated zero.  Entry reach is checked
    *before* the binding: a frontier that did not reach entry serializes
    ``entry_unreached`` with ``checkpoint_reached: false`` and emits no numeric P
    metric even when a syntactically digest-bound aggregate is supplied.  Route B's
    entry starts no cell transition, because entry lives in the pixel-pair universe
    and the later checkpoints live in the reference-cell universe.

    Route B's ``promoted`` (state promotion) and ``terminal`` (decoded export)
    substages have no dedicated A1 checkpoint, so their ``checkpoint_reached`` is
    ``null`` unless their own carried record proves the substage was reached; A1's
    coarse ``output`` reach is never copied onto them, because it is not evidence
    that the promotion or export substage ran.  Presence is explicit record
    existence (``is not None``): a carried empty mapping proves the substage record
    exists and serializes reach ``true`` with a typed null metric, while a missing
    record is ``null``.
    """
    if frontier.route is not Route.B:
        raise AttributionRefusal("route_b_attribution needs a route B frontier")
    checkpoints = _frontier_rows(frontier, ("entry", "lattice", "geometry", "output"))
    rows: list[AttributionRow] = []
    transitions: list[AttributionTransition] = []
    scores: dict[str, CellScore] = {}

    def _cell_checkpoint(name: str, record: Any, space: MetricSpace, kind: str,
                         record_digest: str, checkpoint_reached: bool | None) -> CellScore | None:
        # Two admissible record shapes only: a coordinate-only geometry record
        # (placement/volume) for a G checkpoint, or a coordinate-to-BlockState map
        # for any cell checkpoint.  ``None`` and a carried empty record are
        # unavailable, never a fabricated zero.
        if isinstance(record, GeometryCellEvidence):
            if space is not MetricSpace.GEOMETRY:
                raise AttributionRefusal(
                    "coordinate-only geometry evidence may only score a G checkpoint")
            carried = bool(record.cells)
            coordinates = record.cells
        elif record is None:
            carried, coordinates = False, None
        elif isinstance(record, Mapping):
            carried = bool(record)
            coordinates = None
        else:
            raise AttributionRefusal(
                "a Route B cell record must be a state map or coordinate-only "
                "geometry evidence")
        if not carried:
            if name in ("placed", "volume_cell"):
                candidate_reason = checkpoints["geometry"].reason or "geometry_record_absent"
            else:
                candidate_reason = "record_absent"
            rows.append(_unreached_row(checkpoint=name, metric=(
                "geometry_recall" if space is MetricSpace.GEOMETRY
                else "exact_state_recovery"), space=space, reason=candidate_reason,
                checkpoint_reached=checkpoint_reached))
            return None
        if coordinates is not None:
            # Coordinate-only production geometry: align and score the coordinates
            # without ever manufacturing a ``BlockState`` for them.
            if alignment is None:
                raise AttributionRefusal(
                    "carried geometry evidence has no measured alignment")
            score = score_aligned_geometry_coordinates(
                reference=reference, cells=coordinates,
                visible_reference=visible_reference, alignment=alignment,
                checkpoint=name, space=space, palette_states=palette)
        else:
            if alignment is None:
                raise AttributionRefusal(
                    "carried state evidence has no measured alignment")
            score = score_aligned_cells(
                reference=reference, candidate=record,
                visible_reference=visible_reference,
                alignment=alignment, checkpoint=name, space=space,
                palette_states=palette)
        scores[name] = score
        if space is MetricSpace.GEOMETRY:
            rows.extend(cell_metric_rows(
                score, kind=kind,
                ceiling=_ceiling(len(score.matched), score.visible_cells),
                record_digest=record_digest,
                checkpoint_reached=checkpoint_reached))
        else:
            rows.extend(exact_state_metric_rows(
                score, kind=kind,
                ceiling=_ceiling(len(score.matched), score.visible_cells),
                record_digest=record_digest,
                checkpoint_reached=checkpoint_reached))
        return score

    _cell_checkpoint("placed", placement, MetricSpace.GEOMETRY, "geometry",
                     checkpoints["geometry"].record_digest,
                     checkpoints["geometry"].reachable)
    volume_score = _cell_checkpoint(
        "volume_cell", volume_cells, MetricSpace.GEOMETRY, "geometry",
        checkpoints["output"].record_digest,
        checkpoints["output"].reachable)
    raw_states = None if proposals is None else {
        cell: getattr(record, "state", None)
        for cell, record in proposals.items()
        if getattr(record, "state", None) is not None
    }
    _cell_checkpoint("raw_winner", raw_states, MetricSpace.EXACT_STATE,
                     "label", checkpoints["output"].record_digest,
                     checkpoints["output"].reachable)

    # B10 palette availability and B11 candidate availability over the same cell
    # universe as G and E.  A state the palette lacks is unreachable at every K and
    # is charged to the palette; a cell with no candidate record is charged to
    # geometry.  Both are availability ceilings, never a label.  The G base evidence
    # is run-native here (coordinate-only placement/volume from a real run, or a
    # state map from a synthetic caller); the C context below re-applies the one
    # permitted alignment to that same run-native evidence, so the C rows share the
    # frozen visible universe with the G/E rows without inheriting a G-space score or
    # a doubly transformed map.  It is a genuine C-space ``CellScore``: the strict
    # ``candidate_metric_rows`` C-space guard is never weakened.
    if volume_score is not None:
        base_evidence = volume_cells
    elif scores.get("placed") is not None:
        base_evidence = placement
    else:
        base_evidence = None
    output_reached = checkpoints["output"].reachable
    if palette is None:
        rows.append(_unreached_row(
            checkpoint="palette", metric="palette_coverage",
            space=MetricSpace.CANDIDATE, reason="palette_unknown",
            checkpoint_reached=output_reached))
        rows.append(_unreached_row(
            checkpoint="candidate_k", metric="candidate_available_k8",
            space=MetricSpace.CANDIDATE, reason="palette_unknown",
            checkpoint_reached=output_reached))
    elif base_evidence is None:
        reason = checkpoints["output"].reason or "output_record_absent"
        rows.append(_unreached_row(
            checkpoint="palette", metric="palette_coverage",
            space=MetricSpace.CANDIDATE, reason=reason,
            checkpoint_reached=output_reached))
        rows.append(_unreached_row(
            checkpoint="candidate_k", metric="candidate_available_k8",
            space=MetricSpace.CANDIDATE, reason=reason,
            checkpoint_reached=output_reached))
    else:
        if alignment is None:
            raise AttributionRefusal(
                "carried candidate evidence has no measured alignment")
        if isinstance(base_evidence, GeometryCellEvidence):
            # A geometry-backed C context: the palette/availability rows need only
            # the aligned vocabulary and the frozen truth, so the coordinate-only
            # scorer supplies them with no fabricated block state.
            candidate_score = score_aligned_geometry_coordinates(
                reference=reference, cells=base_evidence.cells,
                visible_reference=visible_reference, alignment=alignment,
                checkpoint="candidate_k", space=MetricSpace.CANDIDATE,
                palette_states=palette)
        else:
            candidate_score = score_aligned_cells(
                reference=reference, candidate=base_evidence,
                visible_reference=visible_reference, alignment=alignment,
                checkpoint="candidate_k", space=MetricSpace.CANDIDATE,
                palette_states=palette)
        rows.extend(palette_metric_rows(
            candidate_score, kind="availability", checkpoint="palette",
            record_digest=checkpoints["output"].record_digest,
            checkpoint_reached=output_reached))
        if proposals is not None:
            # The ranked candidates are run-frame records: their coordinates and
            # every ranked state are transformed together, once, before the join
            # against the C score's reference-frame ``coordinate_truth``.
            rows.extend(candidate_metric_rows(
                candidate_score, depths=candidate_depths,
                kind="availability_ceiling", checkpoint="candidate_k",
                reachable_by_coordinate=align_candidate_rankings(
                    {cell: tuple(state for state, _cost in
                                 getattr(record, "candidate_costs", ()))
                     for cell, record in proposals.items()},
                    alignment),
                record_digest=checkpoints["output"].record_digest,
                checkpoint_reached=output_reached))
        else:
            rows.append(_unreached_row(
                checkpoint="candidate_k", metric="candidate_available_k8",
                space=MetricSpace.CANDIDATE, reason="proposals_unavailable",
                checkpoint_reached=output_reached))

    # A1's coarse ``output`` reach is *not* evidence that a state-promotion or
    # decoded-export substage ran: A1 has no dedicated promotion/export checkpoint.
    # A substage row therefore states reach only when its own record proves it,
    # using **explicit record presence** (``is not None``, never mapping
    # truthiness): a carried empty mapping proves the substage record exists, so its
    # reach is true even though its metric is unavailable.  Only a missing record
    # serializes ``null`` rather than a fabricated ``false``, because absence may
    # mean unattempted/unknown.  ``volume_cell`` and ``raw_winner`` keep the A1
    # output record's own reach, because their evidence *is* that A1 output-stage
    # record.
    promoted_reached = True if promoted_cells is not None else None
    terminal_reached = True if decoded_terminal is not None else None
    _cell_checkpoint("promoted", promoted_cells, MetricSpace.EXACT_STATE, "label",
                     checkpoints["output"].record_digest, promoted_reached)
    _cell_checkpoint("terminal", decoded_terminal, MetricSpace.EXACT_STATE, "label",
                     checkpoints["output"].record_digest, terminal_reached)

    for previous_name, next_name in (("placed", "volume_cell"),
                                     ("volume_cell", "raw_winner"),
                                     ("raw_winner", "promoted"),
                                     ("promoted", "terminal")):
        previous = scores.get(previous_name)
        following = scores.get(next_name)
        if previous is not None and following is not None \
                and previous.space is following.space:
            # Only the G-to-G step (``placed -> volume_cell``) is unpartitioned:
            # both endpoints carry the geometry success set, so
            # ``geometry_lost/state_lost`` stays exact-state-only.  Every E-to-E
            # step (``raw_winner -> promoted``, ``promoted -> terminal``) keeps the
            # disjoint exact-state ``geometry_lost``/``state_lost`` partition.
            transitions.append(cell_transition(
                previous, following,
                partition_loss=previous.space is MetricSpace.EXACT_STATE))

    geometry_rows = [row for row in rows if row.space is MetricSpace.GEOMETRY]
    candidate_rows = [row for row in rows if row.space is MetricSpace.CANDIDATE]
    exact_rows = [row for row in rows if row.space is MetricSpace.EXACT_STATE]
    # The reachable entry/B0 checkpoint is the only P-universe row this route can
    # carry.  It may be scored only from a digest-bound joint match set; an
    # evaluator-only alternate matcher's aggregate is a separate control and is
    # never attached here as achieved Route B P.
    entry = checkpoints["entry"]
    if not entry.reachable:
        # Fail closed *before* consulting any evidence.  A digest-bound aggregate
        # can never produce a numeric P row when the accepted frontier says entry
        # was not reached: such a row would carry a measured metric beside
        # ``checkpoint_reached: false``, which is a contradiction.  This branch is
        # therefore first, not the ``bound`` check.
        pair_rows = (_unreached_row(
            checkpoint="entry", metric="truth_pair_recall",
            space=MetricSpace.PAIR_CORRESPONDENCE, reason="entry_unreached",
            checkpoint_reached=False),)
        pair_notes = (
            "P rows use the frame-pair/pixel-pair universe; the accepted frontier "
            "did not reach the entry/B0 checkpoint, so no entry P metric is scored "
            "even when a syntactically digest-bound joint match set is supplied: "
            "the row is typed unavailable with reason `entry_unreached` and "
            "`checkpoint_reached: false`, and it emits no numeric metric.  Entry "
            "connectivity is not a visible-truth pair count and is never divided "
            "by |V|")
    elif entry_pair_evidence is not None and entry_pair_evidence.bound:
        pair_rows = pair_metric_rows(
            checkpoint="entry", totals=entry_pair_evidence.totals,
            binding=entry_pair_evidence.binding,
            record_digest=entry.record_digest,
            checkpoint_reached=entry.reachable)
        pair_notes = (
            "P rows are digest-bound: the entry checkpoint is scored as "
            "truth_pairs_returned / truth_pairs over jointly visible strict "
            "same-surface truth pixel pairs, plus the accepted precision over "
            "returned pixel-pair identities, only from the same joint matcher "
            "(joint_matching.match_and_solve_pair) whose rederived match-array digest "
            "equals the accepted RelativeCameraEvidence.matches_sha256; entry "
            "connectivity and |V| are never substituted, and P is never added to or "
            "subtracted from G, O, C or E")
    else:
        if entry_pair_evidence is not None:
            reason = entry_pair_evidence.reason or ENTRY_MATCH_UNBOUND_REASON
        else:
            reason = entry_pair_unavailable_reason or ENTRY_MATCH_UNBOUND_REASON
        # The entry P metric is unavailable while the accepted frontier's entry
        # reach is true; the two facts are stated separately so a typed null metric
        # is never read as the checkpoint not having run (`entry.reachable` is True
        # on the corrected frozen Route B run).
        pair_rows = (_unreached_row(
            checkpoint="entry", metric="truth_pair_recall",
            space=MetricSpace.PAIR_CORRESPONDENCE, reason=reason,
            checkpoint_reached=entry.reachable),)
        pair_notes = (
            "P rows use the frame-pair/pixel-pair universe; the entry checkpoint is "
            "typed unavailable with no metric when there is no digest-bound joint "
            "match set: the retained RelativeCameraEvidence carries match counts and "
            "matches_sha256 but no pixel/keypoint-pair identities, and an "
            "evaluator-only alternate matcher's aggregate (reported separately as a "
            "control) is never attached here as an achieved Route B P metric; entry "
            "connectivity is not a visible-truth pair count and is never divided by "
            "|V|.  `checkpoint_reached` states the accepted frontier's entry reach "
            "separately from the null metric, so an unavailable P row on a reached "
            "checkpoint is not mistaken for a checkpoint that never ran")
    sections = (
        _section(MetricSpace.GEOMETRY, geometry_rows, transitions,
                 "geometry over frozen reference cells for every reachable Route B "
                 "checkpoint; the pose graph is a unit-edge gauge, so no scale is fitted"),
        _section(MetricSpace.OWNERSHIP, (), (),
                 "O rows need evaluator-derived (cell, slot, frame) surface ownership, "
                 "which the frozen A2 truth assets do not provide; ownership is reported "
                 "unavailable rather than substituted by a cell count"),
        _section(MetricSpace.PAIR_CORRESPONDENCE, pair_rows, (), pair_notes),
        _section(MetricSpace.CANDIDATE, candidate_rows, (),
                 "candidate/palette availability over the same cell universe as G and E; "
                 "a state the palette lacks is unreachable at every K"),
        _section(MetricSpace.EXACT_STATE, exact_rows, transitions,
                 "primary denominator is all frozen visible V; V inter palette is only a "
                 "separate conditional vocabulary row"),
    )
    policy = completion_and_refusal_rates(
        planned=len(frontier.checkpoints), reached=len(frontier.reached_stages),
        terminal_stage=_terminal_stage(frontier),
        frontier_stop=None if frontier.frontier_stop is None
        else frontier.frontier_stop.stage)
    return ImageOnlyAttribution(
        route=Route.B, sections=sections, frontier=frontier,
        visible_reference=visible_reference, policy_rates=policy,
        caveats=(
            "Route B's pose graph carries metric_scale_known=False; the ledger searches "
            "integer translation and rotation only and never derives a scale",
            "a Route B checkpoint is scored only when its own record was carried; the "
            "rest are typed unavailable and contribute no zero",
            "A1 has no dedicated state-promotion or decoded-export checkpoint: the "
            "Route B `promoted` and `terminal` rows serialize `checkpoint_reached: "
            "null` unless their own carried record proves that substage, so the coarse "
            "A1 `output` reach is never mapped onto promotion/export and an absent "
            "substage record is never inferred as `false`; presence is explicit "
            "record existence, so a carried empty record still proves reach while "
            "its metric stays unavailable",
        ))


def build_attribution(
    *, frontier: ImageOnlyFrontier, visible_reference: VisibleReference,
    reference: Mapping[BlockCoordinate, BlockState], alignment: Mapping[str, Any],
    **records: Any,
) -> ImageOnlyAttribution:
    """Dispatch to the route-specific builder for one A1 frontier.

    ``include_palette_loss_partition`` is a Route A-only opt-in: when true the
    dispatcher forwards it to :func:`route_a_attribution`, and a Route B frontier
    is a typed :class:`AttributionRefusal` with
    :data:`PALETTE_LOSS_PARTITION_ROUTE_B_REASON`, never a silent no-op that could be
    read as unsupported Route B coverage.
    """
    if not isinstance(frontier, ImageOnlyFrontier):
        raise AttributionRefusal("build_attribution needs an A1 frontier record")
    requested = records.pop("include_palette_loss_partition", False)
    if frontier.route is Route.B and requested:
        raise AttributionRefusal(
            "the in-palette loss partition is Route A's palette_label checkpoint "
            "only; Route B has no palette_label checkpoint: "
            + PALETTE_LOSS_PARTITION_ROUTE_B_REASON)
    if frontier.route is Route.A:
        return route_a_attribution(
            frontier=frontier, visible_reference=visible_reference, reference=reference,
            alignment=alignment, include_palette_loss_partition=requested, **records)
    return route_b_attribution(
        frontier=frontier, visible_reference=visible_reference, reference=reference,
        alignment=alignment, **records)
