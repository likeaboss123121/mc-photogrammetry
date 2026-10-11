"""Bounded M5 image-derived litematic delivery adapter.

This is the narrow, last seam of the image-derived M5 chain.  It consumes exactly
one accepted, immutable, digest-bound
:class:`~mc_photogrammetry.reconstruction.state_promotion.PromotionResult`, an
explicit destination :class:`~pathlib.Path`, an explicit litematic ``name`` and
``description``, an explicit ``overwrite`` flag and an explicit exporter, and
nothing else.  It never reads an image, manifest, camera record, F3 reference,
manual coordinate/direction/bound, project store, CLI or UI surface.

What the seam owns
------------------

* **Input discipline before any filesystem work.**  The promotion is re-validated
  at this boundary, not trusted: its schema and producer version, its closed
  gauge/scale/normalization facts, its ``export_bounds is None``, its cell tuple
  and every nested block are rebuilt through the accepted constructors
  (:meth:`PromotedBlock.build` and the module's own
  :func:`state_promotion._block_payload`/:func:`state_promotion._result_payload`
  digests), and each block's source bindings (``frame_ids``, ``provenance``,
  ``observation_digests``, ``proposal_digest``, ``evidence_digest``) are
  re-checked.  A constructor-bypassed or resealed record is therefore a typed
  refusal rather than an export.
* **Eligibility gates.**  ``export_ready`` must be literally ``True``,
  ``exact_state_eligible`` must be literally ``True``, ``omitted_unknown`` must
  be exactly ``0`` and the cell set must be nonempty with unique, canonically
  ordered coordinates.  Scale and normalization state must still be the explicit
  ``"unresolved"`` literals and ``export_bounds`` must still be ``None``, so the
  exporter derives normalization from the emitted blocks alone.
* **Atomic, typed refusal.**  Invalid/tampered promotion, review/ineligible,
  omitted, empty, invalid destination/name/description, an existing destination
  when ``overwrite`` is ``False``, and an exporter failure are all typed
  :class:`DeliveryRefusal` outcomes that create **no partial output**.  The export
  is written to a same-directory temporary path first, loaded back and verified,
  and only then published: ``overwrite=True`` replaces through ``os.replace``,
  while ``overwrite=False`` publishes through a same-directory ``os.link`` that
  fails with ``FileExistsError`` instead of replacing any destination entry that
  appears after validation.  A failed write, move, link or verification therefore
  leaves no destination file and removes its own temporary.
* **Deterministic delivery identity.**  On success the written artifact is loaded
  back and its *logical* content - the tight region span and the exact
  coordinate/state pairs - is compared against the promotion's fresh
  :meth:`PromotionResult.to_block_volume` blocks.  Only then is
  ``artifact_sha256`` computed from that destination-independent logical content,
  never from the artifact bytes and never from the exporter's wall-clock
  ``TimeCreated``/``TimeModified`` metadata, so two exports of the same
  promotion to two destinations have the same digest even though their gzip
  bytes differ.
* **Immutable result, no source retention.**  :class:`DeliveryResult` is a frozen
  record carrying only ``promotion_sha256``, ``artifact_sha256``, ``block_count``,
  ``bytes_written`` and a runtime-only ``output_path``.  It keeps no reference to
  the promotion, the exporter, the destination directory or any ``BlockVolume``;
  :meth:`DeliveryResult.to_record` omits the runtime path entirely.

The module imports the existing :class:`LitematicExporter` and calls its existing
``export`` method exactly once per accepted delivery, with ``bounds=None`` and a
fresh :meth:`PromotionResult.to_block_volume` volume.  It adds no CLI/UI or
project-store surface, no camera/manifest/F3/manual metadata, no image input and
no upstream replay.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mc_photogrammetry.export.litematic import LitematicExportError, LitematicExporter
from mc_photogrammetry.reconstruction import state_promotion as promotion

__all__ = [
    "DELIVERY_SCHEMA",
    "DELIVERY_PRODUCER_VERSION",
    "DELIVERY_EXPORT_BOUNDS",
    "DELIVERY_MAX_DENSE_CELLS",
    "DELIVERY_REFUSAL_REASONS",
    "DeliveryRefusal",
    "DeliveryResult",
    "deliver_image_derived_litematic",
    "verify_image_derived_litematic",
]

DELIVERY_SCHEMA = "m5-image-derived-delivery-1"
"""Explicit delivery result schema name."""

DELIVERY_PRODUCER_VERSION = "m5-image-derived-delivery-adapter-1"
"""Producer identity carried by this module."""

DELIVERY_EXPORT_BOUNDS = None
"""Every delivery exports with ``bounds=None``; a source bound is an ineligible input."""

DELIVERY_MAX_DENSE_CELLS = 4_194_304
"""Explicit cap on the exporter's dense region, refused *before* any filesystem work.

The exporter materializes one palette index per cell of the tight region, so a
promotion whose coordinate span is astronomically large would abort inside
``pack_palette_indices`` (after the destination directory existed).  Refusing it
here as ``invalid_promotion`` keeps the "no filesystem work before validation"
and "no partial output on refusal" properties true for every input.
"""

DELIVERY_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "invalid_promotion",
    "promotion_not_export_ready",
    "promotion_not_exact_state_eligible",
    "promotion_omitted_unknown",
    "promotion_empty",
    "invalid_destination",
    "invalid_name",
    "invalid_description",
    "destination_exists",
    "export_failed",
    "verify_failed",
)
"""The closed delivery refusal reason enum."""

_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    OverflowError,
    ZeroDivisionError,
)


# ---------------------------------------------------------------------------
# Canonical JSON and small exact helpers
# ---------------------------------------------------------------------------

def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError("value is not a deterministic JSON primitive")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hex64(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _record_fields(record: Any) -> dict:
    return {field.name: getattr(record, field.name)
            for field in dataclasses.fields(record)}


def _fullmatch(pattern: str, value: str) -> bool:
    return re.fullmatch(pattern, value) is not None


# ---------------------------------------------------------------------------
# Typed refusal
# ---------------------------------------------------------------------------

def _refusal_payload(value: "DeliveryRefusal") -> dict:
    return {
        "reason": value.reason,
        "detail_sha256": value.detail_sha256,
    }


@dataclass(frozen=True, slots=True)
class DeliveryRefusal:
    """One typed, atomic delivery refusal.  It carries no partial artifact."""

    reason: str
    """One of :data:`DELIVERY_REFUSAL_REASONS`."""

    detail: dict
    """The typed evidence for this outcome; never a partial artifact or block list."""

    detail_sha256: str
    refusal_sha256: str

    def __post_init__(self) -> None:
        if self.reason not in DELIVERY_REFUSAL_REASONS:
            raise ValueError("delivery refusal reason is not in the closed enum")
        normalized = _json_value(self.detail)
        if not isinstance(normalized, dict):
            raise ValueError("delivery refusal detail must be a JSON object")
        encoded = _canonical_json(normalized)
        if self.detail_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
            raise ValueError("delivery refusal detail digest mismatch")
        if "cells" in normalized or "blocks" in normalized \
                or "artifact" in normalized:
            raise ValueError("delivery refusal must not embed partial artifacts")
        if (not _hex64(self.refusal_sha256)
                or self.refusal_sha256 != _sha256(_refusal_payload(self))):
            raise ValueError("delivery refusal digest mismatch")
        object.__setattr__(self, "detail", normalized)

    @classmethod
    def build(cls, **fields: Any) -> "DeliveryRefusal":
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "refusal_sha256", "")
        object.__setattr__(provisional, "refusal_sha256",
                           _sha256(_refusal_payload(provisional)))
        cls.__post_init__(provisional)
        return provisional

    def to_record(self) -> dict:
        record = _refusal_payload(self)
        record["detail"] = dict(self.detail)
        record["refusal_sha256"] = self.refusal_sha256
        return record


def _refuse(reason: str, detail: Any = None) -> DeliveryRefusal:
    if reason not in DELIVERY_REFUSAL_REASONS:
        reason = "invalid_input"
    normalized = _json_value({} if detail is None else detail)
    detail_sha = hashlib.sha256(
        _canonical_json(normalized).encode("utf-8")).hexdigest()
    return DeliveryRefusal.build(reason=reason, detail=normalized,
                                 detail_sha256=detail_sha)


# ---------------------------------------------------------------------------
# Promotion validation
# ---------------------------------------------------------------------------

def _block_snapshot(block: Any) -> tuple[tuple[int, int, int], tuple[str, Any]]:
    """Rebuild one promoted block through its accepted constructor.

    A block built past ``PromotedBlock.__post_init__`` can carry a forged field
    (a repeated property key or a non-canonical property order, for example) while
    its stored digest still replays, so the block is rebuilt through
    :meth:`PromotedBlock.build` rather than merely re-recorded.
    """
    if not isinstance(block, promotion.PromotedBlock):
        raise ValueError("promoted cell is not a PromotedBlock")
    rebuilt = promotion.PromotedBlock.build(**_record_fields(block))
    if rebuilt.block_sha256 != block.block_sha256 \
            or rebuilt.to_record() != block.to_record():
        raise ValueError("promoted block digest does not replay")
    return block.coordinate, (block.state_name, tuple(block.state_properties))


def _validate_promotion(
    result: Any,
) -> tuple[str, tuple[tuple[tuple[int, int, int], tuple[str, Any]], ...]] | DeliveryRefusal:
    """Return the promotion digest and canonical block snapshots, or a refusal.

    The whole validation is performed here, before any destination or filesystem
    work: type, version, closed facts, eligibility, non-emptiness, uniqueness,
    canonical ordering, every nested block digest and every source binding.
    """
    if not isinstance(result, promotion.PromotionResult):
        return _refuse("invalid_input", {"field": "promotion"})
    if result.schema_version != promotion.PROMOTION_SCHEMA \
            or result.producer_version != promotion.PROMOTION_PRODUCER_VERSION:
        return _refuse("invalid_promotion", {"field": "promotion_version"})
    if result.gauge_state != promotion.PROMOTION_GAUGE_STATE \
            or tuple(result.gauge_axes) != promotion.PROMOTION_GAUGE_AXES:
        return _refuse("invalid_promotion", {"field": "promotion_gauge"})
    if result.scale_state != promotion.PROMOTION_SCALE_STATE:
        return _refuse("invalid_promotion", {"field": "scale_state"})
    if result.normalization_state != promotion.PROMOTION_NORMALIZATION_STATE:
        return _refuse("invalid_promotion", {"field": "normalization_state"})
    if result.export_bounds is not promotion.PROMOTION_EXPORT_BOUNDS:
        return _refuse("invalid_promotion", {"field": "export_bounds"})
    if result.export_ready is not True:
        return _refuse("promotion_not_export_ready", {
            "field": "export_ready", "export_ready": result.export_ready})
    if result.exact_state_eligible is not True:
        return _refuse("promotion_not_exact_state_eligible", {
            "field": "exact_state_eligible",
            "exact_state_eligible": result.exact_state_eligible})
    if type(result.omitted_unknown) is not int or result.omitted_unknown != 0:
        return _refuse("promotion_omitted_unknown", {
            "field": "omitted_unknown", "omitted_unknown": result.omitted_unknown})
    if not isinstance(result.cells, tuple):
        return _refuse("invalid_promotion", {"field": "cells"})
    if not result.cells:
        return _refuse("promotion_empty", {"field": "cells"})

    coordinates: list[tuple[int, int, int]] = []
    snapshots: list[tuple[tuple[int, int, int], tuple[str, Any]]] = []
    try:
        for block in result.cells:
            snapshots.append(_block_snapshot(block))
            coordinates.append(snapshots[-1][0])
    except _ERRORS as error:
        return _refuse("invalid_promotion", {
            "field": "cells", "detail": type(error).__name__})
    if coordinates != sorted(coordinates) or len(set(coordinates)) != len(coordinates):
        return _refuse("invalid_promotion", {"field": "cell_order"})

    # Replay the enclosing record's own payload and promotion digest.
    try:
        payload = promotion._result_payload(result)
    except _ERRORS as error:
        return _refuse("invalid_promotion", {
            "field": "payload", "detail": type(error).__name__})
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
        return _refuse("invalid_promotion", {"field": "payload"})
    if len(payload["cells"]) != len(snapshots):
        return _refuse("invalid_promotion", {"field": "payload_cells"})
    try:
        rebuilt = promotion.PromotionResult.build(**_record_fields(result))
    except _ERRORS as error:
        return _refuse("invalid_promotion", {
            "field": "promotion", "detail": type(error).__name__})
    if rebuilt.promotion_sha256 != result.promotion_sha256 \
            or rebuilt.to_record() != result.to_record():
        return _refuse("invalid_promotion", {"field": "promotion_digest"})
    if not _hex64(result.promotion_sha256):
        return _refuse("invalid_promotion", {"field": "promotion_sha256"})

    # The exporter materializes one dense palette index per region cell.  Refuse
    # an astronomically sparse promotion here rather than inside the exporter.
    low = [min(coordinate[axis] for coordinate in coordinates) for axis in range(3)]
    high = [max(coordinate[axis] for coordinate in coordinates) for axis in range(3)]
    span = 1
    for axis in range(3):
        span *= high[axis] - low[axis] + 1
        if span > DELIVERY_MAX_DENSE_CELLS:
            return _refuse("invalid_promotion", {
                "field": "dense_bounds", "limit": DELIVERY_MAX_DENSE_CELLS,
                "observed": span})
    return result.promotion_sha256, tuple(snapshots)


# ---------------------------------------------------------------------------
# Destination and metadata validation
# ---------------------------------------------------------------------------

def _validate_destination(destination: Any, overwrite: Any) -> Path | DeliveryRefusal:
    if type(overwrite) is not bool:
        return _refuse("invalid_input", {"field": "overwrite"})
    if not isinstance(destination, Path):
        return _refuse("invalid_destination", {"field": "destination"})
    if not destination.name or destination.name in (".", ".."):
        return _refuse("invalid_destination", {"field": "destination_name"})
    parent = destination.parent
    if parent.exists() and not parent.is_dir():
        return _refuse("invalid_destination", {"field": "destination_parent"})
    if destination.exists():
        if destination.is_dir():
            return _refuse("invalid_destination", {"field": "destination_is_directory"})
        if not overwrite:
            return _refuse("destination_exists", {"field": "destination"})
    return destination


def _validate_name(name: Any) -> str | DeliveryRefusal:
    if type(name) is not str:
        return _refuse("invalid_name", {"field": "name"})
    stripped = name.strip()
    if not stripped or len(stripped) > 128 \
            or not _fullmatch(r"[ -~]+", name):
        return _refuse("invalid_name", {"field": "name"})
    return name


def _validate_description(description: Any) -> str | DeliveryRefusal:
    if type(description) is not str:
        return _refuse("invalid_description", {"field": "description"})
    stripped = description.strip()
    if not stripped or len(stripped) > 512 \
            or not _fullmatch(r"[ -~]+", description):
        return _refuse("invalid_description", {"field": "description"})
    return description


# ---------------------------------------------------------------------------
# Artifact verification and the deterministic logical digest
# ---------------------------------------------------------------------------

def _read_artifact(path: Path) -> tuple[tuple[int, int, int], dict, tuple]:
    """Load one written artifact and return its canonical logical content.

    Returns ``(size, blocks, states)``: the tight region size, a mapping from the
    *region-local* integer coordinate to its palette index for every non-air
    block, and the parsed palette entries as ``(name, sorted property pairs)``.
    The exporter places the region at the tight lower bound, so a local
    coordinate plus that bound recovers the original block coordinate.
    """
    import nbtlib

    document = nbtlib.load(path, gzipped=True)
    regions = document["Regions"]
    if len(regions) != 1:
        raise ValueError("a delivered litematic must have exactly one region")
    region = next(iter(regions.values()))
    position = tuple(int(region["Position"][axis]) for axis in "xyz")
    if position != (0, 0, 0):
        raise ValueError("a delivered litematic must use a zero local position")
    size = tuple(int(region["Size"][axis]) for axis in "xyz")
    if any(dimension <= 0 for dimension in size):
        raise ValueError("a delivered litematic must have positive dimensions")

    states: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for entry in region["BlockStatePalette"]:
        name = str(entry["Name"])
        properties: list[tuple[str, str]] = []
        if "Properties" in entry:
            for key in entry["Properties"]:
                properties.append((str(key), str(entry["Properties"][key])))
        states.append((name, tuple(sorted(properties))))

    # Palette index zero is the implicit air state: the reader below skips every
    # cell that resolves to it.  If index zero is not literally ``minecraft:air``
    # a corrupted artifact could hide a real block in a gap cell and still match
    # the promotion's non-air block set, so this is checked before any cell
    # comparison rather than trusting the index-zero convention.
    if not states:
        raise ValueError("a delivered litematic must have a non-empty palette")
    if states[0][0] != "minecraft:air":
        raise ValueError("litematic palette index zero must be minecraft:air")

    bits = max(2, (len(states) - 1).bit_length())
    values = [int(value) for value in region["BlockStates"]]
    mask = (1 << 64) - 1
    blocks: dict[tuple[int, int, int], int] = {}
    index = 0
    for y in range(size[1]):
        for z in range(size[2]):
            for x in range(size[0]):
                bit_index = index * bits
                long_index, offset = divmod(bit_index, 64)
                if long_index >= len(values):
                    raise ValueError("block-state array is shorter than the region")
                value = (values[long_index] & mask) >> offset
                if offset + bits > 64:
                    if long_index + 1 >= len(values):
                        raise ValueError("block-state array is shorter than the region")
                    value |= (values[long_index + 1] & mask) << (64 - offset)
                palette_index = value & ((1 << bits) - 1)
                if palette_index >= len(states):
                    raise ValueError("block-state palette index is out of range")
                if palette_index != 0:
                    blocks[(x, y, z)] = palette_index
                index += 1
    return size, blocks, tuple(states)


def _logical_digest(
    promotion_sha256: str,
    blocks: tuple[tuple[tuple[int, int, int], tuple[str, Any]], ...],
) -> str:
    """Digest the destination-independent logical artifact content.

    The payload deliberately excludes the destination path, the file bytes, the
    gzip container, the palette index order and the exporter's wall-clock
    timestamps: it binds the promotion, the sorted coordinate set and the exact
    state at every coordinate, which is what the artifact means.
    """
    return _sha256({
        "promotion_sha256": promotion_sha256,
        "blocks": [
            {
                "coordinate": list(coordinate),
                "state": {
                    "name": state[0],
                    "properties": [list(item) for item in state[1]],
                },
            }
            for coordinate, state in blocks
        ],
        "block_count": len(blocks),
    })


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def _temporary_path(destination: Path) -> Path:
    """Create the same-directory temporary path, cleaning it up on close failure.

    ``mkstemp`` creates the file before the descriptor is closed, so a failing
    ``os.close`` must not leak a temporary: the partially-created path is removed
    before the ``OSError`` is re-raised to the caller's typed-refusal branch.
    """
    handle, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    temporary = Path(raw)
    try:
        os.close(handle)
    except OSError:
        try:
            temporary.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass
        raise
    return temporary


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    """Atomically publish ``temporary`` as ``destination`` without replacing it.

    A same-directory hard link is the no-clobber primitive: ``os.link`` creates the
    destination only when the name is still free and raises ``FileExistsError``
    when a competing entry appears after the caller's validation, so an existing or
    racing destination is never overwritten.  The two names share one inode and the
    caller removes the temporary afterwards, leaving the destination bytes intact.
    """
    os.link(temporary, destination)


def _deliver(
    promotion_result: Any,
    destination: Any,
    *,
    name: Any,
    description: Any,
    overwrite: Any,
    exporter: Any,
) -> "DeliveryResult | DeliveryRefusal":
    if not isinstance(exporter, LitematicExporter):
        return _refuse("invalid_input", {"field": "exporter"})

    validated = _validate_promotion(promotion_result)
    if isinstance(validated, DeliveryRefusal):
        return validated
    promotion_sha256, snapshots = validated

    destination_path = _validate_destination(destination, overwrite)
    if isinstance(destination_path, DeliveryRefusal):
        return destination_path
    validated_name = _validate_name(name)
    if isinstance(validated_name, DeliveryRefusal):
        return validated_name
    validated_description = _validate_description(description)
    if isinstance(validated_description, DeliveryRefusal):
        return validated_description

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return _refuse("export_failed", {
            "field": "destination_parent", "detail": type(error).__name__})
    try:
        temporary = _temporary_path(destination_path)
    except OSError as error:
        return _refuse("export_failed", {
            "field": "temporary", "detail": type(error).__name__})
    try:
        volume = promotion_result.to_block_volume()
        try:
            exporter.export(
                volume, temporary, name=validated_name,
                description=validated_description,
                bounds=DELIVERY_EXPORT_BOUNDS)
        except LitematicExportError as error:
            return _refuse("export_failed", {
                "field": "export", "detail": type(error).__name__})
        except _ERRORS as error:
            return _refuse("export_failed", {
                "field": "export", "detail": type(error).__name__})
        except Exception as error:  # pragma: no cover - hostile exporter only
            return _refuse("export_failed", {
                "field": "export", "detail": type(error).__name__})

        verified = verify_image_derived_litematic(promotion_result, temporary)
        if isinstance(verified, DeliveryRefusal):
            return _refuse("verify_failed", {"field": "artifact"})

        try:
            bytes_written = temporary.stat().st_size
        except OSError as error:
            return _refuse("verify_failed", {
                "field": "artifact", "detail": type(error).__name__})
        if bytes_written <= 0:
            return _refuse("verify_failed", {"field": "artifact_empty"})

        # Final publication.  ``overwrite=True`` keeps the explicit replace
        # semantics; ``overwrite=False`` links the verified temporary into place,
        # which fails with ``FileExistsError`` rather than replacing a destination
        # that appeared after ``_validate_destination``.
        try:
            if overwrite:
                os.replace(temporary, destination_path)
            else:
                _publish_no_clobber(temporary, destination_path)
        except FileExistsError as error:
            return _refuse("destination_exists", {
                "field": "destination", "detail": type(error).__name__})
        except OSError as error:
            return _refuse("export_failed", {
                "field": "placement", "detail": type(error).__name__})
        artifact_sha256, block_count = verified
    finally:
        # The temporary is removed on every path: ``os.replace`` has already
        # consumed it, while a successful no-clobber link leaves a second name
        # behind and a refused publication leaves the staged file behind.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover - best-effort cleanup
            pass

    return DeliveryResult(
        promotion_sha256=promotion_sha256,
        artifact_sha256=artifact_sha256,
        block_count=block_count,
        bytes_written=bytes_written,
        output_path=destination_path)


def verify_image_derived_litematic(
    promotion_result: promotion.PromotionResult,
    artifact: Path | str,
) -> "tuple[str, int] | DeliveryRefusal":
    """Re-validate one written artifact against the promotion that produced it.

    Returns ``(artifact_sha256, block_count)`` for an artifact whose tight region
    and exact coordinate/state pairs match the promotion's fresh
    :meth:`PromotionResult.to_block_volume` blocks, or a typed
    :class:`DeliveryRefusal` otherwise.  The digest is computed only from the
    logical content, so it is stable across destinations and exporter timestamps.
    """
    validated = _validate_promotion(promotion_result)
    if isinstance(validated, DeliveryRefusal):
        return validated
    promotion_sha256, snapshots = validated
    if not isinstance(artifact, Path):
        if type(artifact) is not str:
            return _refuse("verify_failed", {"field": "artifact"})
        artifact = Path(artifact)
    if not artifact.exists() or not artifact.is_file():
        return _refuse("verify_failed", {"field": "artifact_missing"})
    try:
        size, loaded_blocks, loaded_states = _read_artifact(artifact)
    except _ERRORS as error:
        return _refuse("verify_failed", {
            "field": "artifact", "detail": type(error).__name__})
    except Exception as error:  # pragma: no cover - nbtlib parse failure only
        return _refuse("verify_failed", {
            "field": "artifact", "detail": type(error).__name__})

    expected = {coordinate: state for coordinate, state in snapshots}
    low = tuple(min(coordinate[axis] for coordinate in expected) for axis in range(3))
    if size != _tight_size(snapshots):
        return _refuse("verify_failed", {"field": "artifact_bounds"})
    if len(loaded_blocks) != len(snapshots):
        return _refuse("verify_failed", {
            "field": "artifact_block_count",
            "expected": len(snapshots), "observed": len(loaded_blocks)})
    for local, palette_index in loaded_blocks.items():
        coordinate = (low[0] + local[0], low[1] + local[1], low[2] + local[2])
        expected_state = expected.get(coordinate)
        loaded_state = loaded_states[palette_index]
        if expected_state is None or loaded_state != expected_state:
            return _refuse("verify_failed", {
                "field": "artifact_state", "coordinate": list(coordinate)})

    ordered = tuple((coordinate, expected[coordinate])
                    for coordinate in sorted(expected))
    return _logical_digest(promotion_sha256, ordered), len(ordered)


def _tight_size(
    snapshots: tuple[tuple[tuple[int, int, int], tuple[str, Any]], ...],
) -> tuple[int, int, int]:
    low = [min(coordinate[axis] for coordinate, _state in snapshots)
           for axis in range(3)]
    high = [max(coordinate[axis] for coordinate, _state in snapshots)
            for axis in range(3)]
    return tuple(high[axis] - low[axis] + 1 for axis in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Immutable delivery result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """The immutable outcome of one accepted image-derived litematic delivery.

    The record carries only the promotion digest, the destination-independent
    logical ``artifact_sha256``, the delivered ``block_count``, the written
    ``bytes_written`` and the runtime-only ``output_path``.  It retains no
    reference to the promotion, the exporter, the temporary path or any mutable
    ``BlockVolume``, and :meth:`to_record` excludes the runtime path so the record
    is destination-independent by construction.
    """

    promotion_sha256: str
    artifact_sha256: str
    block_count: int
    bytes_written: int
    output_path: Path = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _hex64(self.promotion_sha256):
            raise ValueError("delivery promotion digest is malformed")
        if not _hex64(self.artifact_sha256):
            raise ValueError("delivery artifact digest is malformed")
        if type(self.block_count) is not int or self.block_count <= 0:
            raise ValueError("delivery block count must be a positive integer")
        if type(self.bytes_written) is not int or self.bytes_written <= 0:
            raise ValueError("delivery byte count must be a positive integer")
        if not isinstance(self.output_path, Path):
            raise ValueError("delivery output path must be a Path")

    def to_record(self) -> dict:
        """The serializable, destination-independent delivery record."""
        return {
            "promotion_sha256": self.promotion_sha256,
            "artifact_sha256": self.artifact_sha256,
            "block_count": self.block_count,
            "bytes_written": self.bytes_written,
        }


def deliver_image_derived_litematic(
    promotion_result: promotion.PromotionResult,
    destination: Path,
    *,
    name: str,
    description: str,
    overwrite: bool = False,
    exporter: LitematicExporter | None = None,
) -> DeliveryResult | DeliveryRefusal:
    """Deliver one accepted promotion as a verified litematic, or refuse atomically.

    ``promotion_result`` must be an accepted, replay-valid ``PromotionResult`` with
    ``export_ready is True``, ``exact_state_eligible is True``, zero omitted
    unknown cells, at least one block, unresolved scale/normalization state and
    ``export_bounds is None``.  Every nested block digest and source binding is
    re-validated before any filesystem work.  ``destination`` is the explicit
    output ``Path``, ``name``/``description`` are the explicit litematic metadata
    strings and ``overwrite`` is the explicit opt-in that permits replacing an
    existing destination.

    An invalid or tampered promotion, an ineligible/review/omitted/empty
    promotion, an invalid destination/name/description, an existing destination
    without ``overwrite``, a destination that appears before publication, a
    directory/temporary/placement failure and any exporter or verification failure
    all return a typed, atomic :class:`DeliveryRefusal` that leaves no destination
    file behind and removes its own temporary.  With ``overwrite=False`` the
    verified temporary is published through a same-directory ``os.link``, so a
    destination created after validation is refused rather than replaced; with
    ``overwrite=True`` the verified temporary is moved over the destination through
    ``os.replace``.  On success the artifact is written through a same-directory
    temporary path, loaded back and compared against the promotion's fresh
    :meth:`PromotionResult.to_block_volume` blocks, and a frozen
    :class:`DeliveryResult` is returned whose ``artifact_sha256`` is computed from
    the destination-independent logical content only.
    """
    try:
        return _deliver(
            promotion_result, destination, name=name, description=description,
            overwrite=overwrite,
            exporter=LitematicExporter() if exporter is None else exporter)
    except _ERRORS as error:
        return _refuse("invalid_input", {
            "field": "input", "detail": type(error).__name__})
    except OSError as error:
        # Backstop: every expected placement failure is named inside ``_deliver``
        # (and its temporary already removed in the ``finally``), so an ``OSError``
        # reaching this boundary is still a typed refusal rather than an escaped
        # exception.
        return _refuse("export_failed", {
            "field": "placement", "detail": type(error).__name__})
