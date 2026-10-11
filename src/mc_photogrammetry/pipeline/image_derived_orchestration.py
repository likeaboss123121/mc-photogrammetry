"""Bounded M6 image-derived orchestration facade.

This module is the narrow composition seam the M4/M5 chain was accepted for.  It
owns exactly one call shape - :func:`compose_image_derived_pipeline` - and turns
the already-accepted, digest-bound image-derived records into one immutable,
digest-bound :class:`PipelineOutcome`, or one typed, atomic
:class:`PipelineRefusal`.  It is a *facade*: it never re-solves a camera, a
match, a pose, a volume, a face observation, a proposal or a promotion, and it
never edits, reseals or reinterprets an accepted record.

What it consumes
----------------

Every input is an already-accepted record of an accepted seam, plus the exact
source records that seam was built from:

* :class:`~mc_photogrammetry.pipeline.image_derived_session.ImageDerivedSession`
  - the retained, immutable, digest-bound per-frame pixels and local cameras;
* :class:`~mc_photogrammetry.pipeline.image_derived_connections.ImageDerivedConnectionGraph`
  - the digest-bound relative-camera evidence graph for that session;
* :class:`~mc_photogrammetry.pipeline.image_derived_pose_graph.ImageDerivedPoseGraph`
  - the digest-bound relative gauge poses for that graph;
* :class:`~mc_photogrammetry.reconstruction.plane_volume.PlaneVolumeResult` plus
  its original ``PlacementUniverse``, ``AlignedRegionSupport`` records,
  ``ResolvedFacePlacement`` and both budgets - the accepted V3 geometry;
* :class:`~mc_photogrammetry.reconstruction.plane_face_observation.PlaneFaceObservationResult`
  plus its immutable per-frame rectification contexts - the accepted M5 plane
  homography observations;
* an explicit :class:`~mc_photogrammetry.blocks.appearance.PaletteColor` tuple
  and explicit proposal thresholds/budgets - the material palette a proposal
  call needs;
* an explicit :class:`~mc_photogrammetry.reconstruction.state_promotion.PromotionBudget`.

The facade itself derives the
:class:`~mc_photogrammetry.reconstruction.appearance_state_proposals.FaceStateProposalResult`
and the
:class:`~mc_photogrammetry.reconstruction.state_promotion.PromotionResult`
through the accepted seams, and calls the accepted
:func:`~mc_photogrammetry.pipeline.image_derived_delivery.deliver_image_derived_litematic`
seam **only** for an exact-state-eligible promotion.  A caller that already
holds a promotion or a delivery may pass ``expected_promotion`` /
``expected_delivery``; the facade then requires them to match the composed
records digest for digest instead of trusting them.

If no ``destination`` is given, composition stops after the promotion and the
outcome carries that promotion for the caller to deliver later under the same
eligibility gate.  The genuine recipe fixture has 15 unknown cells, so with the
default promotion settings its composed promotion is an ineligible review; a
caller must explicitly opt into the accepted ``allow_omitted_unknown`` path to
compose material states from it.  This module's own controls therefore prove the
gate rather than the successful delivery: an ineligible promotion is refused as
a typed review and the accepted delivery seam is never called for it, while the
accepted delivery suite covers the successful branch with its own genuinely
eligible promotion.

Validation discipline
---------------------

Every digest-bound input is replayed before any composition work, in a fixed
order, and a mismatch is a typed refusal rather than a silent input:

* the session is round-tripped through its accepted
  ``image_derived_session_to_record``/``image_derived_session_from_record``
  pair, and every frame's retained pixels are re-digested directly (a record
  cannot recompute pixels, so the producer-side pixel binding is checked here);
* the connection graph's and the pose graph's canonical payload digests are
  recomputed and compared with their stored digests, and their nested records
  are then rebuilt through their accepted constructors before an enclosing
  reseal is trusted: every connection budget and every evidence record (status
  and refusal reason, unit baseline direction, proper rotation, finite
  diagnostics, digest, and its session binding), every pose-graph budget and
  every gauge pose (proper rotation, finite position, its graph binding), and
  the components that must replay from those poses.  A stale nested record, a
  resealed enclosing digest over a constructor-bypassed record, or a nested
  budget whose own digest no longer replays is a typed refusal;
* the V3 volume's canonical payload digest is recomputed, and the volume plus
  every frame context is re-derived through the accepted M5 bridge builder,
  which reruns accepted V3 from the original universe/supports/placement/budgets
  and refuses a resealed or fresh result;
* the provided M5 observation result must equal the freshly re-derived one
  record for record, so a stale or swapped observation set cannot ride along;
* every frame rectification context, both budgets and the promotion budget are
  re-digested through their accepted constructors;
* the proposal result is re-derived from the authoritative M5 observations with
  the caller's palette and thresholds, and every proposal/observation digest is
  computed by the accepted adapter;
* a supplied ``expected_promotion`` must be an accepted, replay-valid
  ``PromotionResult`` whose ``volume_sha256`` and ``proposal_result_digest``
  bind exactly the composed volume and proposal result, and whose closed
  gauge/scale/normalization/``export_bounds`` facts replay;
* a supplied ``expected_delivery`` must bind that same promotion and its
  artifact must be the one the accepted delivery seam just wrote and verified.

Cross-record provenance is then enforced exactly: the graph's
``session_sha256`` must equal the session digest, the pose graph's
``graph_sha256``/``session_sha256`` must bind the composed graph and session,
the volume's ``placement_sha256`` must equal the placement record's own
``placement_sha256``, the M5 result must bind that volume and placement, the
frame contexts must cover the session frames exactly, and every V3 support
binding frame must be a session frame.  A coherent record built against a
*different* session, graph, volume or placement is therefore a typed
``source_mismatch`` rather than a plausible result.

Hard non-claims
---------------

The facade invents nothing and trusts no manual metadata.  It accepts no
camera position, scale, bound, translation, coordinate, direction, name or
F3 reference, and any non-empty ``manual_metadata`` mapping is refused with a
typed ``manual_metadata_forbidden`` before a single record is inspected.  Scale
and normalization stay explicitly unresolved end to end: ``metric_scale_known``
and ``scale_known`` must be literally ``False``, the promotion's
``scale_state``/``normalization_state`` must stay ``"unresolved"``, its
``export_bounds`` must stay ``None``, and the volume's own
``normalization_offset`` is copied, never inferred.

Boundary
--------

The module imports only the accepted image-derived seams, the accepted
reconstruction adapters, the block palette type, and the delivery adapter.  It
performs no IO of its own, imports no CLI/UI/project-store module, reads no
image or manifest, and adds no filesystem, process, network, environment or
dynamic-import surface beyond the explicit destination the accepted delivery
seam writes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields as _dataclass_fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.pipeline import image_derived_connections as connections_mod
from mc_photogrammetry.pipeline import image_derived_delivery as delivery_mod
from mc_photogrammetry.pipeline import image_derived_pose_graph as pose_graph_mod
from mc_photogrammetry.pipeline import image_derived_session as session_mod
from mc_photogrammetry.reconstruction import appearance_state_proposals as proposals_mod
from mc_photogrammetry.reconstruction import plane_face_observation as face_mod
from mc_photogrammetry.reconstruction import plane_placement as placement_mod
from mc_photogrammetry.reconstruction import plane_region_support as v1_mod
from mc_photogrammetry.reconstruction import plane_volume as volume_mod
from mc_photogrammetry.reconstruction import state_promotion as promotion_mod

__all__ = [
    "M6_ORCHESTRATION_SCHEMA",
    "M6_ORCHESTRATION_PRODUCER",
    "M6_MANUAL_METADATA_KEYS",
    "M6_REFUSAL_REASONS",
    "PipelineRefusal",
    "PipelineOutcome",
    "compose_image_derived_pipeline",
]

M6_ORCHESTRATION_SCHEMA = "m6-image-derived-orchestration-1"
"""Explicit orchestration outcome schema name."""

M6_ORCHESTRATION_PRODUCER = "m6-image-derived-orchestration-facade-1"
"""Producer identity carried by every outcome and refusal."""

M6_MANUAL_METADATA_KEYS: tuple[str, ...] = (
    "camera_position",
    "camera_positions",
    "scale",
    "bounds",
    "export_bounds",
    "coordinates",
    "direction",
    "directions",
    "translation",
    "manifest",
    "f3",
    "f3_reference",
    "annotation",
    "scene_transform",
)
"""Every manual-metadata key the facade declares forbidden.

The facade consumes no manual metadata at all; this closed tuple exists so a
host caller cannot quietly pass a coordinate, direction, bound, scale, manifest
or F3 reference through the one mapping the facade does accept.
"""

M6_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "manual_metadata_forbidden",
    "invalid_record",
    "digest_mismatch",
    "source_mismatch",
    "unresolved_scale",
    "proposal_refused",
    "promotion_refused",
    "delivery_refused",
    "artifact_mismatch",
)
"""Closed vocabulary of typed, atomic orchestration refusal reasons."""

_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    OverflowError,
    ZeroDivisionError,
)


class _Refused(Exception):
    """Internal control flow: one typed refusal with its machine-readable detail."""

    def __init__(self, reason: str, detail: Any = None) -> None:
        self.reason = reason
        self.detail = {} if detail is None else detail
        super().__init__(reason)


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise _Refused("invalid_record", {"field": "value"})
        return float(value)
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise _Refused("invalid_record", {"field": "value"})


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _detail(value: Any) -> dict:
    normalized = _json_value({} if value is None else value)
    if not isinstance(normalized, dict):
        normalized = {"value": normalized}
    return normalized


def _freeze_detail(value: Any) -> Any:
    """Recursively freeze an already-normalized JSON structure.

    Mappings become read-only proxies and sequences become tuples, so the stored
    ``detail`` can be neither item-assigned nor mutated in place.  The frozen
    structure still serializes and digests exactly like its plain form because
    :func:`_json_value` treats a mapping proxy as a mapping and a tuple as a list.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_detail(item)
                                 for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_detail(item) for item in value)
    return value


def _plain_detail(value: Any) -> Any:
    """Return a deep, mutable, JSON-able plain copy of a frozen detail structure."""
    if isinstance(value, Mapping):
        return {key: _plain_detail(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_detail(item) for item in value]
    return value


def _declared_fields(record: Any) -> dict:
    """Every declared dataclass field of ``record`` as a plain mapping."""
    return {field.name: getattr(record, field.name)
            for field in _dataclass_fields(record)}


# ---------------------------------------------------------------------------
# Typed refusal and composed outcome
# ---------------------------------------------------------------------------

_FORBIDDEN_REFUSAL_DETAIL_KEYS = frozenset(
    {"cells", "blocks", "artifact", "observations"})


@dataclass(frozen=True, slots=True)
class PipelineRefusal:
    """One typed, atomic orchestration refusal.  It carries no promoted state.

    ``detail`` is stored as a deep, immutable structure: nested mappings become
    read-only :class:`~types.MappingProxyType` proxies and nested sequences become
    tuples, so no part of the evidence can be mutated after construction.  It
    still rounds-trips canonically through :meth:`to_record` and its digest
    replays through the accepted canonical JSON helpers.
    """

    schema: str
    producer: str
    reason: str
    detail: Mapping[str, Any]
    refusal_sha256: str

    def __post_init__(self) -> None:
        if self.schema != M6_ORCHESTRATION_SCHEMA:
            raise ValueError("orchestration refusal schema is unsupported")
        if self.producer != M6_ORCHESTRATION_PRODUCER:
            raise ValueError("orchestration refusal producer is unsupported")
        if self.reason not in M6_REFUSAL_REASONS:
            raise ValueError("orchestration refusal reason is not in the closed enum")
        try:
            normalized = _json_value(self.detail)
        except _Refused as error:
            raise ValueError(
                "orchestration refusal detail must be a deterministic JSON object") from error
        if not isinstance(normalized, dict):
            raise ValueError("orchestration refusal detail must be a JSON object")
        if set(normalized) & _FORBIDDEN_REFUSAL_DETAIL_KEYS:
            raise ValueError("orchestration refusal must not embed partial records")
        object.__setattr__(self, "detail", _freeze_detail(normalized))
        expected = _sha256({"producer": self.producer, "reason": self.reason,
                            "detail": normalized})
        if self.refusal_sha256 != expected:
            raise ValueError("orchestration refusal digest mismatch")

    @classmethod
    def build(cls, *, reason: str, detail: Any = None) -> "PipelineRefusal":
        normalized = _detail(detail)
        reason = reason if reason in M6_REFUSAL_REASONS else "invalid_input"
        return cls(
            schema=M6_ORCHESTRATION_SCHEMA,
            producer=M6_ORCHESTRATION_PRODUCER,
            reason=reason,
            detail=normalized,
            refusal_sha256=_sha256({
                "producer": M6_ORCHESTRATION_PRODUCER,
                "reason": reason,
                "detail": normalized,
            }),
        )

    def to_record(self) -> dict:
        return {
            "schema": self.schema,
            "producer": self.producer,
            "reason": self.reason,
            "detail": _plain_detail(self.detail),
            "refusal_sha256": self.refusal_sha256,
        }


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """The immutable, digest-bound result of one composed image-derived pipeline.

    The record carries the accepted source records the pipeline was composed
    from, the freshly derived proposal and promotion records, the optional
    delivery result, and this outcome's own deterministic digest.  It retains
    no mutable volume or state and exposes the accepted records unchanged.

    Every carried record is bound twice: its own declared digest must equal the
    matching scalar field (so a missing or swapped record is rejected even when
    every scalar hash is a valid digest), and the outcome payload/digest binds a
    canonical digest of each record's own payload (so an internal change that
    keeps the declared scalar still changes :attr:`outcome_sha256`).

    The summary/provenance scalars are cross-bound to the records that own them:
    ``exact_state_eligible``, ``export_ready``, ``omitted_unknown``,
    ``scale_state`` and ``normalization_state`` must equal the carried
    promotion's own facts, ``unresolved`` must equal the carried pose graph's
    unresolved notes, and a carried delivery must bind the same promotion digest
    and its promoted block count.  A forged summary field therefore refuses even
    when the caller reseals ``outcome_sha256`` around it.
    """

    schema: str
    producer: str
    session_sha256: str
    graph_sha256: str
    pose_graph_sha256: str
    volume_sha256: str
    face_observation_sha256: str
    proposal_result_digest: str
    promotion_sha256: str
    delivery_artifact_sha256: str | None
    exact_state_eligible: bool
    delivered: bool
    export_ready: bool
    omitted_unknown: int
    normalization_state: str
    scale_state: str
    unresolved: tuple[str, ...]
    session: Any
    graph: Any
    pose_graph: Any
    volume: Any
    face_observations: Any
    proposals: Any
    promotion: Any
    delivery: Any
    outcome_sha256: str

    def __post_init__(self) -> None:
        if self.schema != M6_ORCHESTRATION_SCHEMA:
            raise ValueError("orchestration outcome schema is unsupported")
        if self.producer != M6_ORCHESTRATION_PRODUCER:
            raise ValueError("orchestration outcome producer is unsupported")
        for name in ("session_sha256", "graph_sha256", "pose_graph_sha256",
                     "volume_sha256", "face_observation_sha256",
                     "proposal_result_digest", "promotion_sha256"):
            if not session_mod._is_sha256(getattr(self, name)):
                raise ValueError(f"orchestration {name} is malformed")
        if self.delivery_artifact_sha256 is not None \
                and not session_mod._is_sha256(self.delivery_artifact_sha256):
            raise ValueError("orchestration delivery digest is malformed")
        if type(self.exact_state_eligible) is not bool \
                or type(self.delivered) is not bool:
            raise ValueError("orchestration flags must be boolean")
        if type(self.export_ready) is not bool:
            raise ValueError("orchestration export flag must be boolean")
        if type(self.omitted_unknown) is not int or self.omitted_unknown < 0:
            raise ValueError("orchestration omission count must be a non-negative integer")
        if self.scale_state != promotion_mod.PROMOTION_SCALE_STATE \
                or self.normalization_state != promotion_mod.PROMOTION_NORMALIZATION_STATE:
            raise ValueError("orchestration scale/normalization state must stay unresolved")
        if (not isinstance(self.unresolved, tuple)
                or any(not isinstance(note, str) for note in self.unresolved)):
            raise ValueError("orchestration unresolved notes must be a tuple of strings")
        if self.delivered != (self.delivery is not None):
            raise ValueError("orchestration delivery flag disagrees with its result")
        if self.delivered != (self.delivery_artifact_sha256 is not None):
            raise ValueError("orchestration delivery digest disagrees with its result")
        # Every carried record must be present, type-valid, and bound both to its
        # own declared digest scalar and to the enclosing outcome digest.  A
        # missing or swapped record is rejected here even when every scalar hash
        # is a valid, self-consistent digest.
        for label, record, expected_type in (
            ("session", self.session, session_mod.ImageDerivedSession),
            ("graph", self.graph, connections_mod.ImageDerivedConnectionGraph),
            ("pose_graph", self.pose_graph, pose_graph_mod.ImageDerivedPoseGraph),
            ("volume", self.volume, volume_mod.PlaneVolumeResult),
            ("face_observations", self.face_observations,
             face_mod.PlaneFaceObservationResult),
            ("proposals", self.proposals, proposals_mod.FaceStateProposalResult),
            ("promotion", self.promotion, promotion_mod.PromotionResult),
        ):
            if not isinstance(record, expected_type):
                raise ValueError(
                    f"orchestration carried {label} record is missing or type-invalid")
        if self.delivery is not None and not isinstance(
                self.delivery, delivery_mod.DeliveryResult):
            raise ValueError("orchestration carried delivery record is type-invalid")
        for label, declared, scalar in (
            ("session", self.session.session_sha256, self.session_sha256),
            ("graph", self.graph.graph_sha256, self.graph_sha256),
            ("pose_graph", self.pose_graph.pose_graph_sha256,
             self.pose_graph_sha256),
            ("volume", self.volume.volume_sha256, self.volume_sha256),
            ("face_observations", self.face_observations.result_sha256,
             self.face_observation_sha256),
            ("proposals", self.proposals.result_digest, self.proposal_result_digest),
            ("promotion", self.promotion.promotion_sha256, self.promotion_sha256),
        ):
            if declared != scalar:
                raise ValueError(
                    f"orchestration carried {label} record does not bind its "
                    "declared digest")
        if (None if self.delivery is None else self.delivery.artifact_sha256) \
                != self.delivery_artifact_sha256:
            raise ValueError(
                "orchestration carried delivery record does not bind its declared "
                "digest")
        # The summary/provenance scalars are not free-floating copies.  Each one
        # is cross-bound to the record that owns it, so a caller that forges a
        # summary field and reseals ``outcome_sha256`` still cannot describe a
        # promotion, pose graph or delivery that disagrees with the carried
        # records.
        if self.exact_state_eligible != self.promotion.exact_state_eligible:
            raise ValueError(
                "orchestration exact-state flag disagrees with its carried promotion")
        if self.export_ready != self.promotion.export_ready:
            raise ValueError(
                "orchestration export flag disagrees with its carried promotion")
        if self.omitted_unknown != self.promotion.omitted_unknown:
            raise ValueError(
                "orchestration omission count disagrees with its carried promotion")
        if self.scale_state != self.promotion.scale_state:
            raise ValueError(
                "orchestration scale state disagrees with its carried promotion")
        if self.normalization_state != self.promotion.normalization_state:
            raise ValueError(
                "orchestration normalization state disagrees with its carried promotion")
        if self.unresolved != tuple(self.pose_graph.unresolved):
            raise ValueError(
                "orchestration unresolved notes disagree with its carried pose graph")
        if self.delivery is not None:
            if self.delivery.promotion_sha256 != self.promotion.promotion_sha256:
                raise ValueError(
                    "orchestration delivery does not bind its carried promotion")
            if self.delivery.block_count != len(self.promotion.cells):
                raise ValueError(
                    "orchestration delivery block count disagrees with its carried "
                    "promotion")
        expected = _sha256(_outcome_payload(self))
        if self.outcome_sha256 != expected:
            raise ValueError("orchestration outcome digest mismatch")

    @classmethod
    def build(cls, **fields: Any) -> "PipelineOutcome":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "outcome_sha256", "")
        object.__setattr__(provisional, "outcome_sha256",
                           _sha256(_outcome_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _outcome_payload(self)
        record["outcome_sha256"] = self.outcome_sha256
        return record


def _to_record(record: Any) -> dict:
    return record.to_record()


def _record_binding(record: Any, encoder: Any) -> str:
    """Canonical digest of one carried record's own declared payload."""
    try:
        return _sha256(encoder(record))
    except _Refused as error:
        raise ValueError("orchestration carried record is malformed") from error
    except _ERRORS as error:
        raise ValueError("orchestration carried record is malformed") from error


def _outcome_payload(outcome: "PipelineOutcome") -> dict:
    delivery = outcome.delivery
    return {
        "schema": outcome.schema,
        "producer": outcome.producer,
        "session_sha256": outcome.session_sha256,
        "graph_sha256": outcome.graph_sha256,
        "pose_graph_sha256": outcome.pose_graph_sha256,
        "volume_sha256": outcome.volume_sha256,
        "face_observation_sha256": outcome.face_observation_sha256,
        "proposal_result_digest": outcome.proposal_result_digest,
        "promotion_sha256": outcome.promotion_sha256,
        "delivery_artifact_sha256": outcome.delivery_artifact_sha256,
        "exact_state_eligible": bool(outcome.exact_state_eligible),
        "delivered": bool(outcome.delivered),
        "export_ready": bool(outcome.export_ready),
        "omitted_unknown": int(outcome.omitted_unknown),
        "normalization_state": outcome.normalization_state,
        "scale_state": outcome.scale_state,
        "unresolved": list(outcome.unresolved),
        # The outcome digest binds every carried record's own canonical payload,
        # not only the scalar digests, so a swapped record changes the outcome.
        "record_bindings": {
            "session": _record_binding(
                outcome.session, session_mod.image_derived_session_to_record),
            "graph": _record_binding(outcome.graph, _to_record),
            "pose_graph": _record_binding(outcome.pose_graph, _to_record),
            "volume": _record_binding(outcome.volume, _to_record),
            "face_observations": _record_binding(
                outcome.face_observations, _to_record),
            "proposals": _record_binding(
                outcome.proposals, proposals_mod._result_digest_payload),
            "promotion": _record_binding(outcome.promotion, _to_record),
            "delivery": (None if delivery is None
                         else _record_binding(delivery, _to_record)),
        },
    }


# ---------------------------------------------------------------------------
# Manual metadata and input-type guards
# ---------------------------------------------------------------------------

def _check_manual_metadata(manual_metadata: Any) -> None:
    if manual_metadata is None:
        return
    if not isinstance(manual_metadata, Mapping):
        raise _Refused("invalid_input", {"field": "manual_metadata"})
    if manual_metadata:
        raise _Refused("manual_metadata_forbidden", {
            "keys": sorted(str(key) for key in manual_metadata)})


def _is_immutable_uint8(image: Any) -> bool:
    """A read-only ``uint8`` array whose buffer chain ends at immutable ``bytes``."""
    if not isinstance(image, np.ndarray) or image.dtype != np.dtype(np.uint8):
        return False
    base: Any = image
    seen: set[int] = set()
    while isinstance(base, np.ndarray):
        if id(base) in seen:
            return False
        seen.add(id(base))
        if base.flags.writeable:
            return False
        base = base.base
    return isinstance(base, bytes)


def _replay_budget(record: Any, module: Any, label: str) -> Any:
    """Re-digest one budget record from its own declared payload.

    Every accepted budget record owns a ``_payload`` method and its owning module
    owns the canonical ``_sha256``; replaying the pair proves the budget was not
    resealed past its accepted constructor.
    """
    try:
        if record.budget_sha256 != module._sha256(record._payload()):
            raise _Refused("digest_mismatch", {"field": label})
    except _Refused:
        raise
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": label,
                                          "detail": type(error).__name__}) from error
    return record


# ---------------------------------------------------------------------------
# Per-record replay
# ---------------------------------------------------------------------------

def _replay_session(session: Any) -> Any:
    if not isinstance(session, session_mod.ImageDerivedSession):
        raise _Refused("invalid_input", {"field": "session"})
    try:
        record = session_mod.image_derived_session_to_record(session)
        rebuilt = session_mod.image_derived_session_from_record(record)
    except session_mod.ImageDerivedSessionError as error:
        raise _Refused("invalid_record", {"field": "session",
                                          "detail": error.reason}) from error
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "session",
                                          "detail": type(error).__name__}) from error
    if rebuilt.session_sha256 != session.session_sha256:
        raise _Refused("digest_mismatch", {"field": "session"})
    frames = tuple(session.frames)
    frame_ids = tuple(frame.frame_id for frame in frames)
    if tuple(session.cameras) != frame_ids or tuple(session.local_cameras) != frame_ids:
        raise _Refused("source_mismatch", {"field": "session_cameras"})
    for frame in frames:
        image = frame.image
        if image is None:
            raise _Refused("invalid_record", {"field": "session_pixels",
                                              "frame_id": frame.frame_id})
        if not _is_immutable_uint8(image):
            raise _Refused("invalid_record", {"field": "session_pixels_immutable",
                                              "frame_id": frame.frame_id})
        try:
            digest = session_mod.image_sha256(image)
        except _ERRORS as error:
            raise _Refused("invalid_record", {"field": "session_pixels",
                                              "frame_id": frame.frame_id,
                                              "detail": type(error).__name__}) from error
        if digest != frame.image_sha256:
            raise _Refused("digest_mismatch", {"field": "session_pixels",
                                               "frame_id": frame.frame_id})
    return session


def _replay_connection_budget(budget: Any) -> None:
    """Re-run the accepted ``ConnectionBudgets`` constructor over a nested budget.

    The enclosing graph digest does not carry the nested ``budget_sha256``, so a
    budget resealed past its own constructor can only be caught here.
    """
    if not isinstance(budget, connections_mod.ConnectionBudgets):
        raise _Refused("invalid_record", {"field": "graph_budget"})
    try:
        rebuilt = connections_mod.ConnectionBudgets(
            max_pairs=budget.max_pairs,
            max_total_support=budget.max_total_support)
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "graph_budget",
                                          "detail": type(error).__name__}) from error
    if rebuilt.to_payload() != budget.to_payload():
        raise _Refused("digest_mismatch", {"field": "graph_budget"})


def _replay_connection_hypothesis(hypothesis: Any, index: int) -> Any:
    """Re-run the accepted ``YawClassDiagnostic`` constructor over one hypothesis.

    ``RelativeCameraEvidence.__post_init__`` only ``isinstance``-checks its nested
    diagnostics, so a diagnostic forged past its own ``__post_init__`` (a negative
    support, an out-of-range yaw class, a non-boolean ``passed`` flag, a
    non-string refusal or a non-finite/typed diagnostic value) would otherwise
    ride an evidence digest that was resealed to match it.  The record is rebuilt
    field for field through its accepted constructor and its canonical payload is
    compared field for field, so a malformed nested hypothesis refuses typed
    instead of being trusted.
    """
    if not isinstance(hypothesis, connections_mod.YawClassDiagnostic):
        raise _Refused("invalid_record", {"field": "graph_evidence_hypotheses",
                                          "index": index})
    try:
        rebuilt = connections_mod.YawClassDiagnostic(**_declared_fields(hypothesis))
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "graph_evidence_hypotheses",
                                          "index": index,
                                          "detail": type(error).__name__}) from error
    if (connections_mod._hypothesis_payload(rebuilt)
            != connections_mod._hypothesis_payload(hypothesis)):
        raise _Refused("digest_mismatch", {"field": "graph_evidence_hypotheses",
                                           "index": index})
    return rebuilt


def _replay_connection_evidence(records: Any, session: Any) -> None:
    """Replay every nested evidence record through its accepted constructor.

    Each record's own invariants (status/reason agreement, unit direction, proper
    rotation, finite diagnostics, digest) and its session binding are re-checked,
    so a stale nested record cannot ride an enclosing digest that was resealed to
    match it.  Every nested :class:`YawClassDiagnostic` is itself deeply
    reconstructed through its accepted constructor before the evidence payload is
    accepted, because the evidence/ graph constructors only ``isinstance``-check
    the diagnostics and would pass a constructor-bypassed one through.
    """
    if not isinstance(records, tuple):
        raise _Refused("invalid_record", {"field": "graph_evidence"})
    for record in records:
        if not isinstance(record, connections_mod.RelativeCameraEvidence):
            raise _Refused("invalid_record", {"field": "graph_evidence"})
        if not isinstance(record.hypotheses, tuple):
            raise _Refused("invalid_record", {"field": "graph_evidence_hypotheses"})
        rebuilt_hypotheses = tuple(
            _replay_connection_hypothesis(hypothesis, index)
            for index, hypothesis in enumerate(record.hypotheses))
        try:
            rebuilt = connections_mod.RelativeCameraEvidence(
                **{**_declared_fields(record), "hypotheses": rebuilt_hypotheses})
        except _ERRORS as error:
            raise _Refused("invalid_record", {"field": "graph_evidence",
                                              "detail": type(error).__name__}) from error
        if rebuilt.to_payload() != record.to_payload():
            raise _Refused("digest_mismatch", {"field": "graph_evidence"})
        if record.session_sha256 != session.session_sha256:
            raise _Refused("source_mismatch", {"field": "graph_evidence_session"})


def _replay_graph(graph: Any, session: Any) -> Any:
    if not isinstance(graph, connections_mod.ImageDerivedConnectionGraph):
        raise _Refused("invalid_input", {"field": "graph"})
    try:
        payload = connections_mod._graph_payload(graph)
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "graph",
                                          "detail": type(error).__name__}) from error
    if graph.graph_sha256 != connections_mod._digest(payload):
        raise _Refused("digest_mismatch", {"field": "graph"})
    # A resealed enclosing digest is not evidence that the nested records ever
    # passed their own constructors: rebuild the graph, its budget and every
    # evidence record through the accepted constructors before trusting bindings.
    try:
        connections_mod.ImageDerivedConnectionGraph(**_declared_fields(graph))
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "graph",
                                          "detail": type(error).__name__}) from error
    _replay_connection_budget(graph.budgets)
    _replay_connection_evidence(graph.evidence, session)
    if graph.session_sha256 != session.session_sha256:
        raise _Refused("source_mismatch", {"field": "graph_session"})
    if graph.status != "ok":
        raise _Refused("source_mismatch", {"field": "graph_status",
                                           "status": graph.status})
    session_ids = tuple(frame.frame_id for frame in session.frames)
    if tuple(graph.frame_ids) != session_ids:
        raise _Refused("source_mismatch", {"field": "graph_frames"})
    # The accepted builder refuses a graph that exceeds either declared cap before
    # returning it, so an ``ok`` graph must still satisfy the caps carried by its
    # own replayed budget.
    if (graph.pair_count > graph.budgets.max_pairs
            or sum(record.support for record in graph.evidence)
            > graph.budgets.max_total_support):
        raise _Refused("invalid_record", {"field": "graph_budget_caps"})
    return graph


def _replay_pose_budget(budget: Any) -> None:
    """Re-run the accepted ``PoseGraphBudgets`` constructor over a nested budget."""
    if not isinstance(budget, pose_graph_mod.PoseGraphBudgets):
        raise _Refused("invalid_record", {"field": "pose_graph_budget"})
    try:
        rebuilt = pose_graph_mod.PoseGraphBudgets(
            max_edges=budget.max_edges,
            max_cycle_checks=budget.max_cycle_checks,
            rotation_tolerance_degrees=budget.rotation_tolerance_degrees,
            direction_tolerance=budget.direction_tolerance)
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "pose_graph_budget",
                                          "detail": type(error).__name__}) from error
    if rebuilt.to_payload() != budget.to_payload():
        raise _Refused("digest_mismatch", {"field": "pose_graph_budget"})


def _replay_pose_records(poses: Any, graph: Any) -> None:
    """Replay every nested gauge pose and its source binding to the graph."""
    if not isinstance(poses, tuple):
        raise _Refused("invalid_record", {"field": "pose_graph_poses"})
    for pose in poses:
        if not isinstance(pose, pose_graph_mod.GaugeCameraPose):
            raise _Refused("invalid_record", {"field": "pose_graph_poses"})
        try:
            rebuilt = pose_graph_mod.GaugeCameraPose(**_declared_fields(pose))
        except _ERRORS as error:
            raise _Refused("invalid_record", {"field": "pose_graph_poses",
                                              "detail": type(error).__name__}) from error
        if rebuilt.to_payload() != pose.to_payload():
            raise _Refused("digest_mismatch", {"field": "pose_graph_poses"})
        if pose.graph_sha256 != graph.graph_sha256:
            raise _Refused("source_mismatch", {"field": "pose_graph_pose_binding"})


def _replay_pose_graph(pose_graph: Any, graph: Any, session: Any) -> Any:
    if not isinstance(pose_graph, pose_graph_mod.ImageDerivedPoseGraph):
        raise _Refused("invalid_input", {"field": "pose_graph"})
    try:
        payload = pose_graph_mod._pose_graph_payload(pose_graph)
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "pose_graph",
                                          "detail": type(error).__name__}) from error
    if pose_graph.pose_graph_sha256 != pose_graph_mod._digest(payload):
        raise _Refused("digest_mismatch", {"field": "pose_graph"})
    # Rebuild the enclosing result and every nested budget/pose record through
    # their accepted constructors, so a resealed enclosing digest cannot hide a
    # record that violates its own invariants or a stale source binding.
    try:
        pose_graph_mod.ImageDerivedPoseGraph(**_declared_fields(pose_graph))
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "pose_graph",
                                          "detail": type(error).__name__}) from error
    _replay_pose_budget(pose_graph.budgets)
    _replay_pose_records(pose_graph.poses, graph)
    if pose_graph.graph_sha256 != graph.graph_sha256:
        raise _Refused("source_mismatch", {"field": "pose_graph_graph"})
    if pose_graph.session_sha256 != session.session_sha256:
        raise _Refused("source_mismatch", {"field": "pose_graph_session"})
    if pose_graph.status != "ok":
        raise _Refused("source_mismatch", {"field": "pose_graph_status",
                                           "status": pose_graph.status})
    if tuple(pose_graph.frame_ids) != tuple(graph.frame_ids):
        raise _Refused("source_mismatch", {"field": "pose_graph_frames"})
    if pose_graph.metric_scale_known is not False:
        raise _Refused("unresolved_scale", {"field": "metric_scale_known"})
    if tuple(pose_graph.components) != tuple(graph.components):
        raise _Refused("source_mismatch", {"field": "pose_graph_components"})
    for pose in pose_graph.poses:
        if pose.graph_sha256 != graph.graph_sha256:
            raise _Refused("source_mismatch", {"field": "pose_graph_pose_binding"})
    # The accepted composer refuses a graph whose solved-edge or fundamental-cycle
    # count exceeds either declared cap before returning poses, so an ``ok`` pose
    # graph must still satisfy the caps carried by its own replayed budget.
    solved_edges = sum(1 for record in graph.evidence if record.status == "solved")
    fundamental_cycles = solved_edges - (len(graph.frame_ids) - len(graph.components))
    if (solved_edges > pose_graph.budgets.max_edges
            or fundamental_cycles > pose_graph.budgets.max_cycle_checks):
        raise _Refused("invalid_record", {"field": "pose_graph_budget_caps"})
    return pose_graph


def _replay_volume(volume: Any, placement: Any, session_ids: tuple[str, ...]) -> Any:
    if not isinstance(volume, volume_mod.PlaneVolumeResult):
        raise _Refused("invalid_input", {"field": "volume"})
    if not isinstance(placement, placement_mod.ResolvedFacePlacement):
        raise _Refused("invalid_input", {"field": "placement"})
    try:
        payload = volume_mod._volume_payload(volume)
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "volume",
                                          "detail": type(error).__name__}) from error
    if volume.volume_sha256 != volume_mod._sha256(payload):
        raise _Refused("digest_mismatch", {"field": "volume"})
    if volume.placement_sha256 != placement.placement_sha256:
        raise _Refused("source_mismatch", {"field": "volume_placement"})
    for binding in volume.support_bindings:
        if binding["frame"] not in session_ids:
            raise _Refused("source_mismatch", {"field": "volume_frame",
                                               "frame_id": binding["frame"]})
    return volume


def _replay_m5_result(result: Any, volume: Any, placement: Any,
                      frames: tuple[Any, ...]) -> Any:
    """Bind the provided M5 result to the volume, placement and frame contexts.

    The result's own frame bindings are compared field by field with the accepted
    contexts (frame id, upstream image digest, retained-pixel digest, usable-mask
    digest and the context digest), so a result produced from different frame
    evidence is a typed ``source_mismatch``.  The accepted constructor already
    replayed the result's own canonical digest.
    """
    if not isinstance(result, face_mod.PlaneFaceObservationResult):
        raise _Refused("invalid_input", {"field": "face_observations"})
    if result.volume_sha256 != volume.volume_sha256:
        raise _Refused("source_mismatch", {"field": "face_observations_volume"})
    if result.placement_sha256 != placement.placement_sha256:
        raise _Refused("source_mismatch", {"field": "face_observations_placement"})
    contexts = {context.frame_id: context for context in frames}
    if len(contexts) != len(frames):
        raise _Refused("invalid_record", {"field": "frame_context_duplicate"})
    if len(result.frame_bindings) != len(contexts):
        raise _Refused("source_mismatch", {"field": "face_observation_frames"})
    for binding in result.frame_bindings:
        context = contexts.get(binding.get("frame_id"))
        if context is None:
            raise _Refused("source_mismatch", {"field": "face_observation_frames"})
        if (binding.get("image_sha256") != context.image_sha256
                or binding.get("pixels_sha256") != context.pixels_sha256
                or binding.get("usable_mask_sha256") != context.usable_mask_sha256
                or binding.get("context_sha256") != context.context_sha256):
            raise _Refused("source_mismatch", {"field": "face_observation_frames"})
    return result


def _replay_frame_contexts(frames: Any, session_ids: tuple[str, ...]) -> tuple[Any, ...]:
    if (not isinstance(frames, Sequence) or isinstance(frames, (str, bytes))
            or not frames):
        raise _Refused("invalid_input", {"field": "frames"})
    ordered: list[Any] = []
    for context in frames:
        if not isinstance(context, face_mod.FrameRectificationContext):
            raise _Refused("invalid_input", {"field": "frames"})
        try:
            payload = face_mod._frame_context_payload(context)
        except _ERRORS as error:
            raise _Refused("invalid_record", {"field": "frame_context",
                                              "detail": type(error).__name__}) from error
        if context.context_sha256 != face_mod._sha256(payload):
            raise _Refused("digest_mismatch", {"field": "frame_context"})
        ordered.append(context)
    ids = [context.frame_id for context in ordered]
    if len(set(ids)) != len(ids):
        raise _Refused("invalid_record", {"field": "frame_context_duplicate"})
    if tuple(ids) != session_ids:
        raise _Refused("source_mismatch", {"field": "frame_context_frames"})
    return tuple(ordered)


def _replay_source_records(universe: Any, supports: Any, placement: Any,
                           placement_budget: Any, volume_budget: Any,
                           face_budget: Any) -> tuple[Any, ...]:
    if not isinstance(universe, placement_mod.PlacementUniverse):
        raise _Refused("invalid_input", {"field": "universe"})
    if (not isinstance(supports, Sequence) or isinstance(supports, (str, bytes))
            or not supports):
        raise _Refused("invalid_input", {"field": "supports"})
    prepared = tuple(supports)
    if any(not isinstance(support, v1_mod.AlignedRegionSupport)
           for support in prepared):
        raise _Refused("invalid_input", {"field": "supports"})
    if not isinstance(placement, placement_mod.ResolvedFacePlacement):
        raise _Refused("invalid_input", {"field": "placement"})
    if not isinstance(placement_budget, placement_mod.PlacementBudget):
        raise _Refused("invalid_input", {"field": "placement_budget"})
    if not isinstance(volume_budget, volume_mod.PlaneVolumeBudget):
        raise _Refused("invalid_input", {"field": "volume_budget"})
    if not isinstance(face_budget, face_mod.PlaneFaceObservationBudget):
        raise _Refused("invalid_input", {"field": "face_budget"})
    _replay_budget(placement_budget, placement_mod, "placement_budget")
    _replay_budget(volume_budget, volume_mod, "volume_budget")
    _replay_budget(face_budget, face_mod, "face_budget")
    return universe, prepared, placement, placement_budget, volume_budget, face_budget


def _replay_expected_promotion(expected: Any, volume: Any, authoritative: Any) -> None:
    """Require a caller-supplied promotion to bind the composed records exactly.

    The record is replayed through the accepted promotion module's own payload
    and constructor path (never through the delivery gate, so an ineligible but
    internally valid promotion is still a valid expectation), and its
    ``volume_sha256``, ``proposal_result_digest`` and ``promotion_sha256`` must
    equal the composed volume, proposal result and promotion.
    """
    if expected is None:
        return
    if not isinstance(expected, promotion_mod.PromotionResult):
        raise _Refused("invalid_input", {"field": "expected_promotion"})
    _replay_promotion_result(expected)
    if expected.volume_sha256 != volume.volume_sha256:
        raise _Refused("source_mismatch", {"field": "promotion_volume"})
    if expected.proposal_result_digest != authoritative.proposal_result_digest:
        raise _Refused("source_mismatch", {"field": "promotion_proposals"})
    if expected.promotion_sha256 != authoritative.promotion_sha256:
        raise _Refused("digest_mismatch", {"field": "promotion_digest"})


# ---------------------------------------------------------------------------
# Proposal and promotion composition
# ---------------------------------------------------------------------------

def _derive_proposals(observations: Any, palette: Any, *, max_cost: Any,
                      min_margin: Any, max_observations: Any,
                      max_candidate_evaluations: Any,
                      max_payload_bytes: Any) -> Any:
    if (not isinstance(palette, Sequence) or isinstance(palette, (str, bytes))
            or not palette
            or any(not isinstance(item, PaletteColor) for item in palette)):
        raise _Refused("invalid_input", {"field": "palette"})
    try:
        result = proposals_mod.propose_face_states(
            tuple(observations),
            tuple(palette),
            max_cost=max_cost,
            min_margin=min_margin,
            max_observations=max_observations,
            max_candidate_evaluations=max_candidate_evaluations,
            max_payload_bytes=max_payload_bytes,
        )
    except proposals_mod.AppearanceStateProposalError as error:
        raise _Refused("invalid_record", {"field": "proposals",
                                          "detail": str(error)}) from error
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "proposals",
                                          "detail": type(error).__name__}) from error
    if not isinstance(result, proposals_mod.FaceStateProposalResult):
        raise _Refused("invalid_record", {"field": "proposals"})
    if result.status == proposals_mod.REFUSED:
        raise _Refused("proposal_refused", {
            "reason": result.reason, "budget": result.budget})
    return result


def _replay_promotion_result(result: Any) -> Any:
    """Replay one accepted promotion record's own payload and digest.

    The record is rebuilt through the accepted module's own constructor path
    (``PromotionResult.build`` over every declared field) and its enclosing
    payload, ``payload_sha256`` and ``promotion_sha256`` are recomputed with the
    module's own canonical helpers, so a resealed or constructor-bypassed
    promotion cannot ride along.  Eligibility is deliberately *not* checked
    here: an ineligible but internally valid promotion is a composed outcome,
    and the delivery gate is what refuses it.
    """
    try:
        payload = promotion_mod._result_payload(result)
        encoded = promotion_mod._canonical_json(payload)
        if result.payload_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
            raise _Refused("digest_mismatch", {"field": "promotion_payload"})
        rebuilt = promotion_mod.PromotionResult.build(
            **promotion_mod._record_fields(result))
    except _Refused:
        raise
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "promotion",
                                          "detail": type(error).__name__}) from error
    if (rebuilt.promotion_sha256 != result.promotion_sha256
            or rebuilt.to_record() != result.to_record()):
        raise _Refused("digest_mismatch", {"field": "promotion"})
    return result


def _derive_promotion(volume: Any, proposals: Any, budget: Any,
                      allow_omitted_unknown: Any) -> Any:
    if not isinstance(budget, promotion_mod.PromotionBudget):
        raise _Refused("invalid_input", {"field": "promotion_budget"})
    if type(allow_omitted_unknown) is not bool:
        raise _Refused("invalid_input", {"field": "allow_omitted_unknown"})
    try:
        result = promotion_mod.promote_plane_states(
            volume, proposals, budget=budget,
            allow_omitted_unknown=allow_omitted_unknown)
    except _ERRORS as error:
        raise _Refused("invalid_record", {"field": "promotion",
                                          "detail": type(error).__name__}) from error
    if isinstance(result, promotion_mod.PromotionRefusal):
        raise _Refused("promotion_refused", {
            "kind": result.kind, "reason": result.reason})
    if not isinstance(result, promotion_mod.PromotionResult):
        raise _Refused("invalid_record", {"field": "promotion"})
    return _replay_promotion_result(result)


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def _deliver_exact_state(promotion: Any, *, destination: Any, name: Any,
                         description: Any, overwrite: Any, exporter: Any,
                         expected_delivery: Any) -> Any:
    """Call the accepted delivery seam for an eligible promotion, or refuse.

    The caller has already established that a delivery is requested and that the
    promotion is exact-state eligible; this helper owns the seam call, the typed
    refusal of the seam's own :class:`DeliveryRefusal`, and the independent
    re-check of a supplied ``expected_delivery``.
    """
    try:
        delivered = delivery_mod.deliver_image_derived_litematic(
            promotion, destination, name=name, description=description,
            overwrite=overwrite, exporter=exporter)
    except _ERRORS as error:
        raise _Refused("delivery_refused", {
            "field": "delivery", "detail": type(error).__name__}) from error
    if isinstance(delivered, delivery_mod.DeliveryRefusal):
        raise _Refused("delivery_refused", {
            "reason": delivered.reason, "detail": dict(delivered.detail)})
    if not isinstance(delivered, delivery_mod.DeliveryResult):
        raise _Refused("invalid_record", {"field": "delivery"})
    if expected_delivery is not None:
        if not isinstance(expected_delivery, delivery_mod.DeliveryResult):
            raise _Refused("invalid_input", {"field": "expected_delivery"})
        if (expected_delivery.promotion_sha256 != promotion.promotion_sha256
                or expected_delivery.artifact_sha256 != delivered.artifact_sha256
                or expected_delivery.block_count != delivered.block_count):
            raise _Refused("artifact_mismatch", {"field": "expected_delivery"})
    return delivered


def _requested_delivery(promotion: Any, *, destination: Any, name: Any,
                        description: Any, overwrite: Any, exporter: Any,
                        expected_delivery: Any) -> Any:
    """Call the accepted delivery seam, or return ``None`` when not requested.

    The seam is called only for an exact-state-eligible promotion; a requested
    delivery of an ineligible or review promotion is a typed
    :class:`PipelineRefusal`, never a partial artifact.
    """
    if destination is None:
        if expected_delivery is not None:
            raise _Refused("invalid_input", {"field": "expected_delivery"})
        return None
    if promotion.exact_state_eligible is not True:
        raise _Refused("promotion_refused", {
            "kind": "review", "reason": "not_exact_state_eligible",
            "omitted_unknown": promotion.omitted_unknown})
    return _deliver_exact_state(
        promotion, destination=destination, name=name, description=description,
        overwrite=overwrite, exporter=exporter, expected_delivery=expected_delivery)


def _check_destination(destination: Any, name: Any, description: Any,
                       overwrite: Any, exporter: Any) -> None:
    """Validate the delivery request's *form* before any composition work.

    Only request-shape facts are checked here (the overwrite flag, the
    destination type, the exporter type, the name/description syntax and the
    ability to derive a valid destination).  Destination *state* - an existing
    file, a directory collision - is the accepted delivery seam's own decision
    and is deliberately left to it, so a valid promotion is never reported as a
    malformed request.
    """
    if type(overwrite) is not bool:
        raise _Refused("invalid_input", {"field": "overwrite"})
    if destination is None:
        return
    if not isinstance(destination, Path):
        raise _Refused("invalid_input", {"field": "destination"})
    if exporter is not None and not isinstance(exporter, delivery_mod.LitematicExporter):
        raise _Refused("invalid_input", {"field": "exporter"})
    if not destination.name or destination.name in (".", ".."):
        raise _Refused("invalid_input", {"field": "destination_name"})
    for validated, label in ((delivery_mod._validate_name(name), "name"),
                             (delivery_mod._validate_description(description),
                              "description")):
        if isinstance(validated, delivery_mod.DeliveryRefusal):
            raise _Refused("invalid_input", {
                "field": label, "reason": validated.reason})


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

def compose_image_derived_pipeline(
    *,
    session: session_mod.ImageDerivedSession,
    graph: connections_mod.ImageDerivedConnectionGraph,
    pose_graph: pose_graph_mod.ImageDerivedPoseGraph,
    volume: volume_mod.PlaneVolumeResult,
    universe: placement_mod.PlacementUniverse,
    supports: Sequence[v1_mod.AlignedRegionSupport],
    placement: placement_mod.ResolvedFacePlacement,
    placement_budget: placement_mod.PlacementBudget,
    volume_budget: volume_mod.PlaneVolumeBudget,
    frames: Sequence[face_mod.FrameRectificationContext],
    face_observations: face_mod.PlaneFaceObservationResult,
    face_budget: face_mod.PlaneFaceObservationBudget,
    palette: Sequence[PaletteColor],
    promotion_budget: promotion_mod.PromotionBudget,
    max_cost: float,
    min_margin: float,
    max_observations: int,
    max_candidate_evaluations: int,
    max_payload_bytes: int,
    allow_omitted_unknown: bool = False,
    destination: Path | None = None,
    name: str = "",
    description: str = "",
    overwrite: bool = False,
    exporter: Any = None,
    expected_promotion: promotion_mod.PromotionResult | None = None,
    expected_delivery: delivery_mod.DeliveryResult | None = None,
    manual_metadata: Mapping[str, Any] | None = None,
) -> PipelineOutcome | PipelineRefusal:
    """Compose the accepted image-derived chain into one typed outcome.

    Every accepted record and source record is replayed and cross-bound exactly
    as documented in the module docstring before any composition work.  The
    proposal and promotion are then derived through the accepted seams.  When
    ``destination`` is given, the accepted delivery seam is called **only** for
    an exact-state-eligible promotion; otherwise the outcome is returned without
    a delivery and the caller may deliver it later under the same gate.

    ``manual_metadata`` must be empty (or ``None``).  ``expected_promotion`` and
    ``expected_delivery`` are optional independent re-checks: when supplied they
    must bind the composed records digest for digest.  Returns a
    :class:`PipelineRefusal` for every invalid, tampered, swapped,
    manual-metadata, review/abstention or refused input.
    """
    try:
        _check_manual_metadata(manual_metadata)
        checked_session = _replay_session(session)
        session_ids = tuple(frame.frame_id for frame in checked_session.frames)
        checked_graph = _replay_graph(graph, checked_session)
        checked_pose_graph = _replay_pose_graph(pose_graph, checked_graph,
                                                checked_session)
        checked_volume = _replay_volume(volume, placement, session_ids)
        checked_frames = _replay_frame_contexts(frames, session_ids)
        (checked_universe, checked_supports, checked_placement,
         checked_placement_budget, checked_volume_budget,
         checked_face_budget) = _replay_source_records(
            universe, supports, placement, placement_budget, volume_budget,
            face_budget)
        _replay_m5_result(face_observations, checked_volume, checked_placement,
                          checked_frames)
        _check_destination(destination, name, description, overwrite, exporter)

        # The accepted M5 bridge is the authority for the volume replay, the
        # context binding and the emitted observation records: it reruns
        # accepted V3 from the original sources and must reproduce the provided
        # observation result record for record.
        built = face_mod.build_plane_face_observations(
            checked_volume, universe=checked_universe, supports=checked_supports,
            placement=checked_placement, placement_budget=checked_placement_budget,
            volume_budget=checked_volume_budget, frames=checked_frames,
            budget=checked_face_budget)
        if isinstance(built, face_mod.PlaneFaceObservationRefusal):
            raise _Refused("invalid_record", {
                "field": "face_observations", "reason": built.reason})
        if not isinstance(built, face_mod.PlaneFaceObservationResult):
            raise _Refused("invalid_record", {"field": "face_observations"})
        if (built.result_sha256 != face_observations.result_sha256
                or built.to_record() != face_observations.to_record()):
            raise _Refused("digest_mismatch", {"field": "face_observations"})

        proposals = _derive_proposals(
            built.observations, palette, max_cost=max_cost, min_margin=min_margin,
            max_observations=max_observations,
            max_candidate_evaluations=max_candidate_evaluations,
            max_payload_bytes=max_payload_bytes)
        promotion = _derive_promotion(checked_volume, proposals, promotion_budget,
                                      allow_omitted_unknown)
        _replay_expected_promotion(expected_promotion, checked_volume, promotion)
        delivered = _requested_delivery(
            promotion, destination=destination, name=name,
            description=description, overwrite=overwrite, exporter=exporter,
            expected_delivery=expected_delivery)

        return PipelineOutcome.build(
            schema=M6_ORCHESTRATION_SCHEMA,
            producer=M6_ORCHESTRATION_PRODUCER,
            session_sha256=checked_session.session_sha256,
            graph_sha256=checked_graph.graph_sha256,
            pose_graph_sha256=checked_pose_graph.pose_graph_sha256,
            volume_sha256=checked_volume.volume_sha256,
            face_observation_sha256=built.result_sha256,
            proposal_result_digest=proposals.result_digest,
            promotion_sha256=promotion.promotion_sha256,
            delivery_artifact_sha256=(
                None if delivered is None else delivered.artifact_sha256),
            exact_state_eligible=bool(promotion.exact_state_eligible),
            delivered=delivered is not None,
            export_ready=bool(promotion.export_ready),
            omitted_unknown=int(promotion.omitted_unknown),
            normalization_state=promotion.normalization_state,
            scale_state=promotion.scale_state,
            unresolved=tuple(checked_pose_graph.unresolved),
            session=checked_session,
            graph=checked_graph,
            pose_graph=checked_pose_graph,
            volume=checked_volume,
            face_observations=built,
            proposals=proposals,
            promotion=promotion,
            delivery=delivered,
        )
    except _Refused as refusal:
        return PipelineRefusal.build(reason=refusal.reason, detail=refusal.detail)
    except _ERRORS as error:
        return PipelineRefusal.build(reason="invalid_record",
                                     detail={"detail": type(error).__name__})
