"""Derive deterministic plane signatures from an accepted image-derived lattice.

This module is an in-memory, all-or-nothing seam.  It replays the accepted
session and lattice terminal, derives the normal-mode usable mask
(``~create_gui_mask``) for every retained frame, and calls ``plane_signature``
once for every retained reading alternative.  It never matches signatures,
solves geometry, reads files, or accepts caller geometry or metadata.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

import cv2
import numpy as np

from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
from mc_photogrammetry.pipeline import image_derived_session as sessions
from mc_photogrammetry.pipeline import candidate_support_provenance as provenance
from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction import lattice_planes, region_growth
from mc_photogrammetry.reconstruction import plane_placement as placement
from mc_photogrammetry.vision.gui_mask import create_gui_mask

__all__ = [
    "IMAGE_DERIVED_SIGNATURES_SCHEMA", "IMAGE_DERIVED_SIGNATURES_PRODUCER",
    "SIGNATURES_TERMINAL", "UsableMask", "SignatureEvidence",
    "SignatureAbstention", "ImageDerivedSignaturesResult",
    "derive_image_derived_signatures",
]

IMAGE_DERIVED_SIGNATURES_SCHEMA = "mc-photogrammetry-image-derived-signatures"
IMAGE_DERIVED_SIGNATURES_PRODUCER = "m9-f1-image-derived-signatures-1"
SIGNATURES_TERMINAL = "geometry_unavailable"

_ERRORS = (AttributeError, TypeError, ValueError, IndexError, KeyError,
           OverflowError, ZeroDivisionError, FloatingPointError)
_DERIVATION_ERRORS = _ERRORS + (cv2.error, MemoryError, RuntimeError)


def _sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _array_payload(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _readonly_array(value: object) -> np.ndarray:
    source = np.asarray(value)
    contiguous = np.ascontiguousarray(source)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape)


@dataclass(frozen=True, slots=True)
class UsableMask:
    """One exact normal-mode usable mask, stored as immutable row-major bytes."""

    frame_id: str
    height: int
    width: int
    usable_bool_bytes: bytes
    mask_sha256: str = ""

    def __post_init__(self) -> None:
        if (type(self.frame_id) is not str or not self.frame_id
                or type(self.height) is not int or self.height <= 0
                or type(self.width) is not int or self.width <= 0
                or type(self.usable_bool_bytes) is not bytes
                or len(self.usable_bool_bytes) != self.height * self.width
                or any(value not in (0, 1) for value in self.usable_bool_bytes)):
            raise ValueError("usable mask fields are malformed")
        digest = _sha256([
            "m9-f1-usable-mask-1", self.frame_id, self.height, self.width,
            hashlib.sha256(self.usable_bool_bytes).hexdigest(),
        ])
        if self.mask_sha256 not in ("", digest):
            raise ValueError("usable mask digest mismatch")
        object.__setattr__(self, "mask_sha256", digest)

    @property
    def mask(self) -> np.ndarray:
        """A read-only boolean view of the exact retained mask."""
        return np.frombuffer(self.usable_bool_bytes, dtype=np.bool_).reshape(
            self.height, self.width)


def _footprint_payload(footprint: object) -> object:
    if footprint is None:
        return None
    if not isinstance(footprint, lattice_planes.PlaneRegionFootprint):
        raise ValueError("signature region footprint has an invalid type")
    return footprint.to_record()


def _signature_payload(evidence: "SignatureEvidence") -> list:
    signature = evidence.signature
    return [
        evidence.frame_id, evidence.canonical_index, evidence.reading_axis,
        evidence.alternative_index, float(evidence.blocks_per_period),
        _array_payload(signature.cells), _array_payload(signature.interior),
        _array_payload(signature.coverage),
        [float(value) for value in signature.origin_uv],
        _footprint_payload(signature.region_footprint),
    ]


@dataclass(frozen=True, slots=True)
class SignatureEvidence:
    """One immutable signature bound to one reading alternative."""

    frame_id: str
    canonical_index: int
    reading_axis: int
    alternative_index: int
    blocks_per_period: float
    signature: lattice_planes.PlaneSignature
    signature_sha256: str = ""

    def __post_init__(self) -> None:
        if (type(self.frame_id) is not str or not self.frame_id
                or type(self.canonical_index) is not int or self.canonical_index < 0
                or type(self.reading_axis) is not int
                or type(self.alternative_index) is not int or self.alternative_index < 0
                or not isinstance(self.signature, lattice_planes.PlaneSignature)):
            raise ValueError("signature evidence fields are malformed")
        reading = self.signature.reading
        value = float(self.blocks_per_period)
        if (reading.frame != self.frame_id or int(reading.axis) != self.reading_axis
                or not np.isfinite(value) or value <= 0.0
                or float(self.signature.blocks_per_period) != value):
            raise ValueError("signature evidence is not bound to its reading")
        arrays = (self.signature.cells, self.signature.interior,
                  self.signature.coverage)
        if any(not isinstance(array, np.ndarray) or array.flags.writeable
               for array in arrays):
            raise ValueError("signature arrays must be read-only")
        if (any(array.ndim != 2 or array.shape != arrays[0].shape
                for array in arrays)
                or arrays[0].size == 0
                or not np.issubdtype(arrays[0].dtype, np.floating)
                or not np.issubdtype(arrays[1].dtype, np.floating)
                or arrays[2].dtype != np.bool_
                or not np.isfinite(arrays[0]).all()
                or not np.isfinite(arrays[1]).all()
                or not isinstance(self.signature.origin_uv, tuple)
                or len(self.signature.origin_uv) != 2
                or any(type(item) not in (int, float) or not np.isfinite(item)
                       for item in self.signature.origin_uv)):
            raise ValueError("signature arrays or origin are malformed")
        digest = _sha256(_signature_payload(self))
        if self.signature_sha256 not in ("", digest):
            raise ValueError("signature evidence digest mismatch")
        object.__setattr__(self, "signature_sha256", digest)


@dataclass(frozen=True, slots=True)
class SignatureAbstention:
    """One frame or reading whose signatures could not be produced."""

    frame_id: str
    reason: str
    detail: str = ""
    canonical_index: int | None = None
    alternative_index: int | None = None

    def __post_init__(self) -> None:
        if (type(self.frame_id) is not str or not self.frame_id
                or type(self.reason) is not str or not self.reason
                or type(self.detail) is not str
                or (self.canonical_index is not None
                    and (type(self.canonical_index) is not int
                         or self.canonical_index < 0))
                or (self.alternative_index is not None
                    and (type(self.alternative_index) is not int
                         or self.alternative_index < 0))):
            raise ValueError("signature abstention fields are malformed")


def _masks_digest(masks: tuple[UsableMask, ...]) -> str:
    return _sha256([item.mask_sha256 for item in masks])


def _signatures_digest(signatures: tuple[SignatureEvidence, ...]) -> str:
    return _sha256([item.signature_sha256 for item in signatures])


def _result_payload(result: "ImageDerivedSignaturesResult") -> dict:
    return {
        "schema": result.schema, "producer": result.producer,
        "terminal": result.terminal, "status": result.status,
        "reason": result.reason, "detail": result.detail,
        "session_sha256": result.session_sha256,
        "lattice_sha256": result.lattice_sha256,
        "frame_ids": list(result.frame_ids),
        "masks_sha256": result.masks_sha256,
        "signatures_sha256": result.signatures_sha256,
        "abstentions": [[item.frame_id, item.reason, item.detail,
                          item.canonical_index, item.alternative_index]
                         for item in result.abstentions],
    }


@dataclass(frozen=True, slots=True)
class ImageDerivedSignaturesResult:
    """Atomic signature producer result; unavailable outcomes carry no partials."""

    schema: str
    producer: str
    terminal: str
    status: str
    reason: str
    detail: str
    session_sha256: str
    lattice_sha256: str
    frame_ids: tuple[str, ...]
    usable_masks: tuple[UsableMask, ...]
    signatures: tuple[SignatureEvidence, ...]
    abstentions: tuple[SignatureAbstention, ...]
    masks_sha256: str = ""
    signatures_sha256: str = ""
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != IMAGE_DERIVED_SIGNATURES_SCHEMA
                or self.producer != IMAGE_DERIVED_SIGNATURES_PRODUCER
                or self.terminal != SIGNATURES_TERMINAL
                or self.status not in ("assembled", "unavailable")
                or type(self.reason) is not str or type(self.detail) is not str
                or not _hex64(self.session_sha256)
                or not _hex64(self.lattice_sha256)
                or not isinstance(self.frame_ids, tuple) or not self.frame_ids
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or any(type(item) is not str or not item for item in self.frame_ids)
                or not isinstance(self.usable_masks, tuple)
                or any(not isinstance(item, UsableMask) for item in self.usable_masks)
                or not isinstance(self.signatures, tuple)
                or any(not isinstance(item, SignatureEvidence) for item in self.signatures)
                or not isinstance(self.abstentions, tuple)
                or any(not isinstance(item, SignatureAbstention)
                       for item in self.abstentions)):
            raise ValueError("image-derived signature result fields are malformed")
        # Frozen dataclasses can still be installed via object.__new__; replay
        # every nested helper before trusting its carried digest.
        for mask in self.usable_masks:
            UsableMask(**{field.name: getattr(mask, field.name)
                          for field in dataclasses.fields(UsableMask)})
        for evidence in self.signatures:
            SignatureEvidence(**{
                field.name: getattr(evidence, field.name)
                for field in dataclasses.fields(SignatureEvidence)})
            if evidence.frame_id not in self.frame_ids:
                raise ValueError("signature evidence names a foreign frame")
        for abstention in self.abstentions:
            SignatureAbstention(**{
                field.name: getattr(abstention, field.name)
                for field in dataclasses.fields(SignatureAbstention)})
            if abstention.frame_id not in self.frame_ids:
                raise ValueError("signature abstention names a foreign frame")
        if self.status == "assembled":
            if (self.reason or self.detail or self.abstentions or not self.signatures
                    or tuple(item.frame_id for item in self.usable_masks)
                    != self.frame_ids):
                raise ValueError("assembled signature result is inconsistent")
        elif (not self.reason or self.usable_masks or self.signatures
              or not self.abstentions):
            raise ValueError("unavailable signature result must be atomic")
        masks_digest = _masks_digest(self.usable_masks)
        signatures_digest = _signatures_digest(self.signatures)
        if self.masks_sha256 not in ("", masks_digest):
            raise ValueError("usable-mask collection digest mismatch")
        if self.signatures_sha256 not in ("", signatures_digest):
            raise ValueError("signature collection digest mismatch")
        object.__setattr__(self, "masks_sha256", masks_digest)
        object.__setattr__(self, "signatures_sha256", signatures_digest)
        digest = _sha256(_result_payload(self))
        if self.result_sha256 not in ("", digest):
            raise ValueError("image-derived signature result digest mismatch")
        object.__setattr__(self, "result_sha256", digest)


def _refusal(reason: str, field: str, detail: str = "") -> placement.PlacementRefusal:
    payload = {"field": field}
    if detail:
        payload["detail"] = detail
    return placement._make_refusal(reason, payload)


def _replay_session(session: object) -> tuple[sessions.ImageDerivedSession, tuple] | placement.PlacementRefusal:
    if not isinstance(session, sessions.ImageDerivedSession):
        return _refusal("invalid_input", "session")
    try:
        record = sessions.image_derived_session_to_record(session)
        replayed = sessions.image_derived_session_from_record(record)
        for frame in session.frames:
            dataclasses.replace(frame)
    except (sessions.ImageDerivedSessionError, *_ERRORS) as error:
        return _refusal("provenance_mismatch", "session", type(error).__name__)
    if (replayed.session_sha256 != session.session_sha256
            or tuple(frame.frame_id for frame in replayed.frames)
            != tuple(frame.frame_id for frame in session.frames)):
        return _refusal("provenance_mismatch", "session")
    return replayed, session.frames


def _replay_lattice(lattice_result: object) -> measure.ImageDerivedLatticeResult | placement.PlacementRefusal:
    if not isinstance(lattice_result, measure.ImageDerivedLatticeResult):
        return _refusal("invalid_input", "lattice_result")
    try:
        fields = {field.name: getattr(lattice_result, field.name)
                  for field in dataclasses.fields(measure.ImageDerivedLatticeResult)}
        return measure.ImageDerivedLatticeResult(**fields)
    except measure._ERRORS as error:
        return _refusal("provenance_mismatch", "lattice_result",
                        type(error).__name__)


def _unavailable(*, session_sha256: str, lattice_sha256: str,
                 frame_ids: tuple[str, ...], reason: str, detail: str,
                 abstentions: tuple[SignatureAbstention, ...],
                 ) -> ImageDerivedSignaturesResult:
    return ImageDerivedSignaturesResult(
        schema=IMAGE_DERIVED_SIGNATURES_SCHEMA,
        producer=IMAGE_DERIVED_SIGNATURES_PRODUCER,
        terminal=SIGNATURES_TERMINAL, status="unavailable", reason=reason,
        detail=detail, session_sha256=session_sha256,
        lattice_sha256=lattice_sha256, frame_ids=frame_ids,
        usable_masks=(), signatures=(), abstentions=abstentions)


def derive_image_derived_signatures(
    *, session: object, lattice_result: object,
    signature_witness_sink: list | None = None,
    max_cells: int = lattice_planes.MAX_SIGNATURE_CELLS,
    max_rectified_side: int = lattice_planes.MAX_RECTIFIED_SIDE,
    pixels_per_cell: int = lattice_planes.SIGNATURE_PIXELS_PER_CELL,
) -> ImageDerivedSignaturesResult | placement.PlacementRefusal:
    """Derive every normal-mode mask and reading-alternative signature.

    ``signature_witness_sink`` is an optional evaluation-only list. When given,
    the ordinary :func:`lattice_planes.plane_signature` call for every accepted
    reading alternative also receives a witness sink, and the emitted
    :class:`~...candidate_support_provenance.SourceTapWitnessBindingV1` (bound to
    the frame's ingress pin, usable mask, signature digest and sampler settings)
    or its typed refusal is appended. The default call passes no sink, allocates
    no witness and leaves ``signatures_sha256``/``result_sha256`` unchanged.
    """
    replayed_session = _replay_session(session)
    if isinstance(replayed_session, placement.PlacementRefusal):
        return replayed_session
    session_record, retained_frames = replayed_session
    lattice = _replay_lattice(lattice_result)
    if isinstance(lattice, placement.PlacementRefusal):
        return lattice
    frame_ids = tuple(frame.frame_id for frame in session_record.frames)
    if (lattice.session_sha256 != session_record.session_sha256
            or lattice.frame_ids != frame_ids):
        return _refusal("provenance_mismatch", "lattice_result.session_binding")
    if any(reading.frame not in frame_ids for reading in lattice.readings):
        return _refusal("provenance_mismatch", "lattice_result.readings.frame")
    if lattice.status != "assembled":
        return _unavailable(
            session_sha256=session_record.session_sha256,
            lattice_sha256=lattice.result_sha256, frame_ids=frame_ids,
            reason="geometry_unavailable", detail=lattice.reason,
            abstentions=tuple(SignatureAbstention(
                frame_id=frame_id, reason="geometry_unavailable",
                detail=lattice.reason) for frame_id in frame_ids))

    missing = tuple(frame.frame_id for frame in retained_frames if frame.image is None)
    if missing:
        return _unavailable(
            session_sha256=session_record.session_sha256,
            lattice_sha256=lattice.result_sha256, frame_ids=frame_ids,
            reason="image_unavailable", detail="retained pixels are unavailable",
            abstentions=tuple(SignatureAbstention(
                frame_id=frame_id, reason="image_unavailable",
                detail="retained pixels are unavailable") for frame_id in missing))

    masks: list[UsableMask] = []
    usable_by_frame: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    try:
        for frame in retained_frames:
            image = np.asarray(frame.image)
            height, width = image.shape[:2]
            usable = np.logical_not(create_gui_mask(height, width))
            raw = np.ascontiguousarray(usable, dtype=np.bool_).tobytes(order="C")
            record = UsableMask(frame.frame_id, height, width, raw)
            masks.append(record)
            usable_by_frame[frame.frame_id] = record.mask
            images[frame.frame_id] = image
    except _DERIVATION_ERRORS as error:
        return _unavailable(
            session_sha256=session_record.session_sha256,
            lattice_sha256=lattice.result_sha256, frame_ids=frame_ids,
            reason="signature_unavailable", detail=type(error).__name__,
            abstentions=(SignatureAbstention(
                frame_id=frame_ids[0], reason="mask_failure",
                detail=type(error).__name__),))

    signatures: list[SignatureEvidence] = []
    masks_sha256 = _masks_digest(tuple(masks))
    mask_sha256_by_frame = {item.frame_id: item.mask_sha256 for item in masks}
    # The accepted R4 session binding: the source-tap witness is bound to the
    # session frame's own ingress record, which is absent by default. No
    # side-channel override may substitute a pin that the session digest does
    # not carry.
    frame_ingress = {frame.frame_id: getattr(frame, "ingress", None)
                     for frame in retained_frames}
    capture_positions = {}
    observer = region_growth._attrition_state.get()
    if lattice_planes.lattice._period_support_state() is not None or observer is not None:
        for frame in frame_ids:
            ordered = sorted((r for r in lattice.readings if r.frame == frame),
                             key=lattice_planes.reading_order_key)
            capture_positions.update({id(r): index for index, r in enumerate(ordered)})
    if observer is not None:
        with region_growth._scalar_observation_guard(observer):
            for canonical_index, reading in enumerate(lattice.readings):
                plane_index = capture_positions[id(reading)]
                for alternative_index, alternative in enumerate(reading.alternatives):
                    observer.expect("signature", (reading.frame, plane_index,
                                    alternative_index, float(alternative.blocks_per_period)))
    try:
        for canonical_index, reading in enumerate(lattice.readings):
            capture_plane_index = capture_positions.get(id(reading), -1)
            if not reading.alternatives:
                raise ValueError("reading has no retained alternatives")
            for alternative_index, alternative in enumerate(reading.alternatives):
                witness_list = ([] if signature_witness_sink is not None else None)
                signature_kwargs: dict = {
                    "usable": usable_by_frame[reading.frame], "max_cells": max_cells}
                if witness_list is not None:
                    signature_kwargs["witness_sink"] = witness_list
                capture_identity = (reading.frame, capture_plane_index,
                                    alternative_index, float(alternative.blocks_per_period))
                capture_target = lattice_planes.lattice._period_support_signature(capture_identity)
                capture = lattice_planes.lattice._period_support_state()
                if capture_target:
                    capture.rectifications = []
                try:
                    with lattice_planes.lattice._period_support_context(
                            signature_target=capture_target, signature_identity=capture_identity):
                        signature = lattice_planes.plane_signature(
                            images[reading.frame], reading,
                            alternative.blocks_per_period, **signature_kwargs)
                finally:
                    if capture_target:
                        capture.rectifications = []
                signature = dataclasses.replace(
                    signature,
                    cells=_readonly_array(signature.cells),
                    interior=_readonly_array(signature.interior),
                    coverage=_readonly_array(signature.coverage))
                evidence = SignatureEvidence(
                    frame_id=reading.frame, canonical_index=canonical_index,
                    reading_axis=int(reading.axis),
                    alternative_index=alternative_index,
                    blocks_per_period=float(alternative.blocks_per_period),
                    signature=signature)
                signatures.append(evidence)
                if observer is not None:
                    with region_growth._scalar_observation_guard(observer):
                        identity = (reading.frame, capture_plane_index, alternative_index,
                                    float(alternative.blocks_per_period))
                        observer.finish("signature", identity, "built", axis=int(reading.axis),
                                        canonical_index=canonical_index, plane_index=capture_plane_index,
                                        alternative_index=alternative_index,
                                        blocks_per_period=float(alternative.blocks_per_period),
                                        rows=int(signature.cells.shape[0]), columns=int(signature.cells.shape[1]),
                                        covered_cells=int(np.count_nonzero(signature.coverage)))
                        observer.bind(("signature", id(signature)), identity)
                if capture_target:
                    with lattice_planes.lattice._period_support_context(signature_identity=capture_identity):
                        lattice_planes.lattice._period_support_event("signature_binding",
                            evidence=evidence, canonical_index=canonical_index,
                            mask_sha256=mask_sha256_by_frame[reading.frame], masks_sha256=masks_sha256)
                if witness_list is not None:
                    signature_witness_sink.append(_source_tap_witness_binding(
                        evidence=evidence, witnesses=witness_list,
                        ingress=frame_ingress.get(reading.frame),
                        mask_sha256=mask_sha256_by_frame.get(reading.frame, ""),
                        masks_sha256=masks_sha256, max_cells=max_cells,
                        max_rectified_side=max_rectified_side,
                        pixels_per_cell=pixels_per_cell))
    except _DERIVATION_ERRORS as error:
        if observer is not None:
            with region_growth._scalar_observation_guard(observer):
                failed = locals().get("capture_identity")
                for identity in observer.pending("signature"):
                    observer.finish("signature", identity,
                                    "signature_failure" if tuple(identity) == failed
                                    else "not_attempted_after_signature_failure",
                                    typed_reason="signature_failure", error_type=type(error).__name__)
        frame_id = getattr(locals().get("reading", None), "frame", frame_ids[0])
        return _unavailable(
            session_sha256=session_record.session_sha256,
            lattice_sha256=lattice.result_sha256, frame_ids=frame_ids,
            reason="signature_unavailable", detail=type(error).__name__,
            abstentions=(SignatureAbstention(
                frame_id=frame_id, reason="signature_failure",
                detail=type(error).__name__,
                canonical_index=locals().get("canonical_index"),
                alternative_index=locals().get("alternative_index")),))

    return ImageDerivedSignaturesResult(
        schema=IMAGE_DERIVED_SIGNATURES_SCHEMA,
        producer=IMAGE_DERIVED_SIGNATURES_PRODUCER,
        terminal=SIGNATURES_TERMINAL, status="assembled", reason="", detail="",
        session_sha256=session_record.session_sha256,
        lattice_sha256=lattice.result_sha256, frame_ids=frame_ids,
        usable_masks=tuple(masks), signatures=tuple(signatures), abstentions=())


def _source_tap_witness_binding(
    *, evidence: SignatureEvidence, witnesses: list, ingress: object,
    mask_sha256: str, masks_sha256: str, max_cells: int,
    max_rectified_side: int, pixels_per_cell: int,
) -> provenance.SourceTapWitnessBindingV1 | provenance.SourceTapWitnessRefusalV1:
    """Bind the ordinary sampler's own witness to the accepted evidence.

    A missing, malformed or budget-exhausted witness, or a missing/unusable
    ingress pin, is a typed refusal naming the frame/reading -- never a silent
    absence and never a witness rebuilt from evaluator pixels.
    """
    def refuse(reason: str, detail: str = ""):
        return provenance.SourceTapWitnessRefusalV1(
            frame_id=evidence.frame_id, canonical_index=evidence.canonical_index,
            alternative_index=evidence.alternative_index, reason=reason,
            detail=detail)

    if len(witnesses) != 1:
        return refuse(provenance.SOURCE_TAP_WITNESS_UNAVAILABLE, "witness_absent")
    captured = witnesses[0]
    if isinstance(captured, lattice.WitnessRefusalV1):
        return refuse(provenance.SOURCE_TAP_WITNESS_BUDGET_EXHAUSTED, captured.stage)
    if not isinstance(captured, lattice_planes.SignatureSamplingWitnessV1):
        return refuse(provenance.SOURCE_TAP_WITNESS_UNAVAILABLE, "witness_type")
    if not isinstance(ingress, provenance.FrameIngressProvenanceV1):
        return refuse(provenance.INGRESS_PROVENANCE_UNAVAILABLE, "ingress")
    try:
        return provenance.SourceTapWitnessBindingV1(
            frame_id=evidence.frame_id, canonical_index=evidence.canonical_index,
            alternative_index=evidence.alternative_index,
            signature_sha256=evidence.signature_sha256, ingress=ingress,
            mask_sha256=mask_sha256, masks_sha256=masks_sha256,
            max_cells=int(max_cells), max_rectified_side=int(max_rectified_side),
            pixels_per_cell=int(pixels_per_cell), witness=captured)
    except provenance.CandidateSupportError as error:
        return refuse(error.reason, error.field)
