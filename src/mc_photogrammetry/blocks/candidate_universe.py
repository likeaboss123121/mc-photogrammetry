"""Opt-in v2 inventory identities; no asset loading or render/recall guarantee.

Legal-state completeness is a provenance-bound declaration about an inventory.
Appearance support is separate, and neither declaration proves physical-truth
recall, source authenticity, or fidelity to Minecraft's renderer. External
archive, manifest, model and texture hashes identify supplied records/bytes;
this module does not open or authenticate those sources.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields
from typing import Any

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.domain.models import BlockState

UNIVERSE_SCHEMA = "mc-photogrammetry-candidate-universe"
UNIVERSE_VERSION = 2
COMPLETENESS_STATUSES = ("exhaustive", "certified_under_named_model", "unestablished")
SUPPORT_DISPOSITIONS = ("supported", "unsupported", "unknown")
_NAME = re.compile(r"[a-z0-9_.-]+:[a-z0-9_./-]+\Z")
_TOKEN = re.compile(r"[a-z0-9_.-]+\Z")

# Versioned source-only resource model: enough for a 16384-state inventory
# with modest per-state appearances. These are record ceilings, not throughput
# or recall claims; refusal is atomic and never returns a truncated inventory.
RESOURCE_CONTRACT_VERSION = 1
MAX_LEGAL_STATES = 16_384
MAX_RECORD_NODES = 2_000_000
MAX_RECORD_DEPTH = 32
MAX_STRING_CHARACTERS = 65_536
MAX_METADATA_BYTES = 16_000_000
MAX_CANONICAL_BYTES = 32_000_000


def resource_contract() -> dict:
    return {"version": RESOURCE_CONTRACT_VERSION, "max_legal_states": MAX_LEGAL_STATES,
            "max_record_nodes": MAX_RECORD_NODES, "max_record_depth": MAX_RECORD_DEPTH,
            "max_string_characters": MAX_STRING_CHARACTERS,
            "max_metadata_bytes": MAX_METADATA_BYTES,
            "max_canonical_bytes": MAX_CANONICAL_BYTES, "max_integer_bits": 64}


class _StateIds:
    """A bounded, lazy JSON array view; no sorting or canonicalization here."""

    def __init__(self, states: tuple) -> None:
        self.states = states

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self):
        for state in self.states:
            if type(state) is not BlockState or type(state.name) is not str:
                raise ValueError("legal state must be a BlockState")
            pairs = state.properties
            if type(pairs) is not tuple or len(pairs) > MAX_RECORD_NODES:
                raise ValueError("state properties exceed resource bounds")
            size = len(state.name) + (len(pairs) + 1 if pairs else 0)
            for pair in pairs:
                if (type(pair) is not tuple or len(pair) != 2
                        or any(type(token) is not str for token in pair)):
                    raise ValueError("state properties must be immutable string pairs")
                size += len(pair[0]) + len(pair[1]) + 1
                if size > MAX_STRING_CHARACTERS:
                    raise ValueError("state identity exceeds string resource bounds")
            if size > MAX_STRING_CHARACTERS:
                raise ValueError("state identity exceeds string resource bounds")
            suffix = ",".join(f"{key}={value}" for key, value in pairs)
            yield state.name + (f"[{suffix}]" if suffix else "")


def _contract_view(value: Any) -> Any:
    """Shallow record views only; large sequences and payloads remain aliased.

    Missing computed digests during construction occupy their exact eventual
    64-byte shape. No view invokes validation, serialization or nested replay.
    """
    kind = type(value)
    if (kind is dict and type(value.get("schema")) is str
            and value["schema"] == UNIVERSE_SCHEMA):
        if "legal_states" in value and "appearance_support" in value:
            _inventory_counts(value["legal_states"], value["appearance_support"])
        if "state_ids" in value:
            _inventory_counts(value["state_ids"], ())
    if kind is BlockState:
        return {"name": value.name, "properties": value.properties}
    if kind is PaletteColor:
        return {item.name: getattr(value, item.name) for item in fields(PaletteColor)}
    if kind in (InventoryCompleteness, LegalStateManifest, AppearanceSupport,
                CandidateUniverse):
        result = {item.name: getattr(value, item.name, "0" * 64) for item in fields(kind)}
        if kind is CandidateUniverse:
            _inventory_counts(value.legal_states, value.appearance_support)
            result.update(resource_contract=resource_contract(),
                          state_ids=_StateIds(value.legal_states),
                          appearance_completeness="unestablished")
            # Both dispositions have identical JSON overhead except this label.
            if (type(value.appearance_support) is tuple
                    and all(type(entry) is AppearanceSupport and entry.disposition == "supported"
                            for entry in value.appearance_support)):
                result["appearance_completeness"] = "exhaustive"
        return result
    return value


def _inventory_counts(states: Any, support: Any) -> None:
    for entries in (states, support):
        if type(entries) not in (tuple, list) or len(entries) > MAX_LEGAL_STATES:
            raise ValueError("inventory/support count exceeds resource bounds")


def _children(value: Any, depth: int):
    if type(value) is dict:
        for key, entry in value.items():
            yield key, depth
            yield entry, depth
    else:
        for entry in value:
            yield entry, depth


def _preflight(value: Any, *, max_bytes: int | None = None, project=None) -> tuple[int, int]:
    """Measure JSON nodes/bytes iteratively before copying, decoding or hashing.

    Count every occurrence (including shared aliases), mapping keys, containers
    and leaves. Root depth is zero. String budget counts ASCII-escaped JSON
    contents, including keys; canonical bytes also include quotes/separators.
    Exact primitive/container sizing matches ensure_ascii compact JSON. Integers
    are limited to 64 bits before conversion. Iterator frames keep memory bounded
    by depth instead of creating a stack proportional to the input sequence.
    """
    byte_limit = MAX_CANONICAL_BYTES if max_bytes is None else max_bytes
    nodes = total = metadata = 0
    active: set[int] = set()
    stack = [(iter(((value, 0),)), None)]
    while stack:
        iterator, owner = stack[-1]
        try:
            current, depth = next(iterator)
        except StopIteration:
            stack.pop()
            if owner is not None:
                active.remove(owner)
            continue
        nodes += 1
        if nodes > MAX_RECORD_NODES or depth > MAX_RECORD_DEPTH:
            raise ValueError("record node/depth resource bounds exceeded")
        raw = current
        current = _contract_view(raw)
        if current is raw and project is not None:
            current = project(raw)
        kind = type(current)
        if kind is str:
            if len(current) > MAX_STRING_CHARACTERS:
                raise ValueError("record string resource bounds exceeded")
            size = 0
            for character in current:
                code = ord(character)
                size += (2 if character in '"\\\b\f\n\r\t' else
                         6 if code < 32 or 127 <= code <= 65535 else
                         12 if code > 65535 else 1)
            metadata += size
            total += size + 2
        elif current is None or kind is bool:
            total += 4 if current is None or current is True else 5
        elif kind is int:
            if current.bit_length() > 64:
                raise ValueError("record integer resource bounds exceeded")
            total += len(str(current))
        elif kind is float:
            if not math.isfinite(current):
                raise ValueError("record numbers must be finite")
            total += len(repr(current))
        elif kind in (dict, list, tuple, _StateIds):
            count = len(current)
            # Refuse enormous containers before advancing their first member.
            minimum = count * (2 if kind is dict else 1)
            if nodes + minimum > MAX_RECORD_NODES:
                raise ValueError("record node resource bounds exceeded")
            owner_id = id(raw)
            if owner_id in active:
                raise ValueError("record cycles are forbidden")
            active.add(owner_id)
            total += 2 + max(0, count - 1) + (count if kind is dict else 0)
            if kind is dict:
                if any(type(key) is not str for key in current):
                    raise ValueError("record keys must be strings")
            stack.append((_children(current, depth + 1), owner_id))
        else:
            raise ValueError("record contains an unsupported object")
        if total > byte_limit or metadata > MAX_METADATA_BYTES:
            raise ValueError("record byte/metadata resource bounds exceeded")
    return nodes, total


def canonical_json(value: Any) -> str:
    """Deterministic JSON for contract records, rejecting nonfinite numbers."""
    _preflight(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def record_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, label: str) -> None:
    _preflight(value)
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} must not contain control characters")


def _digest(value: Any, label: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _strings(value: Any, label: str, *, nonempty: bool = False) -> None:
    if type(value) is not tuple or (nonempty and not value):
        raise ValueError(f"{label} must be an immutable tuple")
    for entry in value:
        _text(entry, label)
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{label} must be sorted and unique")


def _record(value: Any, keys: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} record has unexpected or missing fields")
    return value


def _replay(actual: dict, expected: dict, label: str) -> None:
    # JSON comparison preserves bool/int and int/float distinctions.
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"{label} record or digest does not replay canonically")


def _tuple_payload(value: Any) -> Any:
    if type(value) is tuple:
        return [_tuple_payload(entry) for entry in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("appearance payload must contain immutable finite primitives/tuples")


def _decode_tuple(value: Any) -> Any:
    if type(value) is list:
        return tuple(_decode_tuple(entry) for entry in value)
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise ValueError("appearance record contains an unsupported payload")


def canonical_state_id(state: BlockState) -> str:
    """Use the domain's canonical identity, requiring canonical input already.

    Redundant default properties are rejected rather than silently rewritten.
    This checks representation, not legality: legality comes from the declared
    version-matched manifest, never from PaletteColor or a catalogue API.
    """
    _preflight(state)
    _preflight(_StateIds((state,)))
    if type(state) is not BlockState or type(state.name) is not str:
        raise ValueError("legal state must be a BlockState")
    if _NAME.fullmatch(state.name) is None or type(state.properties) is not tuple:
        raise ValueError("state name/properties are not canonical")
    # The domain cache is not part of the record, but comparing a forged cache
    # must also be bounded before equality or sorting touches it.
    _preflight(state._identity)
    for pair in state.properties:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("state properties must be immutable pairs")
        if any(type(token) is not str or _TOKEN.fullmatch(token) is None for token in pair):
            raise ValueError("state property tokens are not canonical")
    if len({key for key, _ in state.properties}) != len(state.properties):
        raise ValueError("duplicate state property")
    if (state.properties != tuple(sorted(state.properties))
            or state._identity != (state.name, state.properties)):
        raise ValueError("state must use sorted, default-stripped canonical properties")
    suffix = ",".join(f"{key}={value}" for key, value in state.properties)
    return state.name + (f"[{suffix}]" if suffix else "")


def _state_record(state: BlockState) -> dict:
    canonical_state_id(state)
    return {"name": state.name, "properties": [list(pair) for pair in state.properties]}


def _state_from_record(record: Any) -> BlockState:
    record = _record(record, {"name", "properties"}, "state")
    _text(record["name"], "state name")
    if type(record["properties"]) is not list:
        raise ValueError("state properties record must be a list")
    pairs = record["properties"]
    if any(type(pair) is not list or len(pair) != 2 for pair in pairs):
        raise ValueError("state property record must contain pairs")
    state = BlockState(record["name"], tuple(tuple(pair) for pair in pairs))
    canonical_state_id(state)
    return state


def _rgb(value: Any) -> None:
    if (type(value) is not tuple or len(value) != 3
            or any(type(v) is not int or not 0 <= v <= 255 for v in value)):
        raise ValueError("appearance rgb must be three integer channels in [0, 255]")


def _vector(value: Any, *, size: int | None = None) -> None:
    if type(value) is not tuple or (size is not None and len(value) != size):
        raise ValueError("appearance vector has an invalid shape")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError("appearance vector must contain finite numbers")


def _slot_pairs(value: Any) -> None:
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("appearance slot mappings must contain immutable pairs")
        _text(pair[0], "appearance slot")
    names = tuple(pair[0] for pair in value)
    if names != tuple(sorted(set(names))):
        raise ValueError("appearance slot mappings must be sorted and unique")


def appearance_record(palette: PaletteColor) -> dict:
    """Bind every appearance field, retaining immutable payloads only."""
    _preflight(palette)
    if type(palette) is not PaletteColor:
        raise ValueError("supported appearance must be a PaletteColor")
    state = _state_record(palette.state)
    result = {item.name: _tuple_payload(getattr(palette, item.name))
              for item in fields(PaletteColor) if item.name != "state"}
    _rgb(palette.rgb)
    for item in fields(PaletteColor):
        if item.name not in ("state", "family", "full_cube"):
            if type(getattr(palette, item.name)) is not tuple:
                raise ValueError(f"appearance {item.name} must be an immutable tuple")
    _text(palette.family, "appearance family")
    if type(palette.full_cube) is not bool:
        raise ValueError("appearance full_cube must be a bool")
    for color in palette.face_rgbs:
        _rgb(color)
    for descriptor in palette.descriptors:
        _vector(descriptor)
    for name in ("face_map", "face_thumbnails", "face_corners"):
        _slot_pairs(getattr(palette, name))
    for _, color in palette.face_map:
        _rgb(color)
    for _, thumbnail in palette.face_thumbnails:
        _text(thumbnail, "appearance thumbnail record")
    for _, corners in palette.face_corners:
        if type(corners) is not tuple or len(corners) != 4:
            raise ValueError("appearance face must have four corners")
        for corner in corners:
            _vector(corner, size=3)
    for box in palette.shape_boxes:
        if type(box) is not tuple or len(box) != 2:
            raise ValueError("appearance box must contain lower and upper corners")
        for corner in box:
            _vector(corner, size=3)
        if any(low > high for low, high in zip(*box)):
            raise ValueError("appearance box lower corner exceeds upper corner")
    _strings(palette.covered_slots, "appearance covered slots")
    if any(type(face) is not tuple for face in palette.model_faces):
        raise ValueError("appearance model faces must be immutable records")
    return {"state": state, **result}


def _appearance_from_record(record: Any) -> PaletteColor:
    _preflight(record)
    keys = {item.name for item in fields(PaletteColor)}
    record = _record(record, keys, "appearance")
    palette = PaletteColor(
        **{key: _state_from_record(value) if key == "state" else _decode_tuple(value)
           for key, value in record.items()}
    )
    _replay(appearance_record(palette), record, "appearance")
    return palette


@dataclass(frozen=True, slots=True)
class InventoryCompleteness:
    """Inventory declaration only; a certificate names its limited model."""

    status: str
    named_model: str | None = None

    def __post_init__(self) -> None:
        _preflight(self)
        if type(self.status) is not str or self.status not in COMPLETENESS_STATUSES:
            raise ValueError("unsupported inventory completeness status")
        if self.status == "certified_under_named_model":
            _text(self.named_model, "completeness named_model")
        elif self.named_model is not None:
            raise ValueError("only certified completeness may name a model")

    def to_record(self) -> dict:
        _preflight(self)
        return {"status": self.status, "named_model": self.named_model}

    @classmethod
    def from_record(cls, record: Any) -> InventoryCompleteness:
        _preflight(record)
        return cls(**_record(record, {"status", "named_model"}, "completeness"))


@dataclass(frozen=True, slots=True)
class LegalStateManifest:
    """External manifest identity and explicit provenance; no authenticity claim."""

    minecraft_version: str
    source_manifest_sha256: str
    manifest_id: str
    generator: str
    generator_version: str
    provenance: tuple[str, ...]
    completeness: InventoryCompleteness

    def __post_init__(self) -> None:
        _preflight(self)
        for label in ("minecraft_version", "manifest_id", "generator", "generator_version"):
            _text(getattr(self, label), label)
        _digest(self.source_manifest_sha256, "source_manifest_sha256")
        _strings(self.provenance, "manifest provenance", nonempty=True)
        if type(self.completeness) is not InventoryCompleteness:
            raise ValueError("manifest must carry immutable inventory completeness")
        self.completeness.__post_init__()

    def to_record(self) -> dict:
        _preflight(self)
        return {"minecraft_version": self.minecraft_version,
                "source_manifest_sha256": self.source_manifest_sha256,
                "manifest_id": self.manifest_id, "generator": self.generator,
                "generator_version": self.generator_version,
                "provenance": list(self.provenance),
                "completeness": self.completeness.to_record()}

    @classmethod
    def from_record(cls, record: Any) -> LegalStateManifest:
        _preflight(record)
        record = _record(record, {item.name for item in fields(cls)}, "manifest")
        if type(record["provenance"]) is not list:
            raise ValueError("manifest provenance record must be a list")
        return cls(**{**record, "provenance": tuple(record["provenance"]),
                      "completeness": InventoryCompleteness.from_record(record["completeness"])})


@dataclass(frozen=True, slots=True)
class AppearanceSupport:
    """Exactly one explicit disposition for one canonical legal state."""

    state_id: str
    disposition: str
    exclusion_reasons: tuple[str, ...] = ()
    appearance: PaletteColor | None = None
    model_record_sha256: str | None = None
    texture_record_sha256: str | None = None
    appearance_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _preflight(self)
        _text(self.state_id, "support state_id")
        if type(self.disposition) is not str or self.disposition not in SUPPORT_DISPOSITIONS:
            raise ValueError("unsupported appearance disposition")
        _strings(self.exclusion_reasons, "exclusion reasons")
        if self.disposition == "supported":
            if self.exclusion_reasons:
                raise ValueError("supported appearance cannot carry exclusions")
            payload = appearance_record(self.appearance)
            if canonical_state_id(self.appearance.state) != self.state_id:
                raise ValueError("appearance state does not match disposition state_id")
            _digest(self.model_record_sha256, "model_record_sha256")
            _digest(self.texture_record_sha256, "texture_record_sha256")
        else:
            if not self.exclusion_reasons:
                raise ValueError("unsupported/unknown appearances require exclusion reasons")
            if any(value is not None for value in (
                    self.appearance, self.model_record_sha256, self.texture_record_sha256)):
                raise ValueError("unsupported/unknown appearances cannot retain support payloads")
            payload = None
        object.__setattr__(self, "appearance_sha256", record_sha256(payload))

    def to_record(self) -> dict:
        _preflight(self)
        return {"state_id": self.state_id, "disposition": self.disposition,
                "exclusion_reasons": list(self.exclusion_reasons),
                "appearance": (None if self.appearance is None
                               else appearance_record(self.appearance)),
                "model_record_sha256": self.model_record_sha256,
                "texture_record_sha256": self.texture_record_sha256,
                "appearance_sha256": self.appearance_sha256}

    @classmethod
    def from_record(cls, record: Any) -> AppearanceSupport:
        _preflight(record)
        record = _record(record, {item.name for item in fields(cls)}, "support")
        if type(record["exclusion_reasons"]) is not list:
            raise ValueError("exclusion reasons record must be a list")
        values = {key: value for key, value in record.items() if key != "appearance_sha256"}
        values["exclusion_reasons"] = tuple(record["exclusion_reasons"])
        if values["appearance"] is not None:
            values["appearance"] = _appearance_from_record(values["appearance"])
        result = cls(**values)
        _replay(result.to_record(), record, "support")
        return result


@dataclass(frozen=True, slots=True)
class CandidateUniverse:
    """A canonical legal inventory and its independently partial appearance support.

    Consumers must abstain when inventory completeness is unestablished or
    appearance support is incomplete/unknown. No retrieval, truncation, scoring,
    or assertion of physical-truth recall is implemented here.
    """

    minecraft_version: str
    source_archive_sha256: str
    manifest: LegalStateManifest
    decoder_version: str
    support_version: str
    decoder_settings: tuple[tuple[str, str], ...]
    legal_states: tuple[BlockState, ...]
    appearance_support: tuple[AppearanceSupport, ...]
    schema: str = UNIVERSE_SCHEMA
    version: int = UNIVERSE_VERSION
    legal_inventory_sha256: str = field(init=False)
    appearance_inventory_sha256: str = field(init=False)
    universe_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _preflight(self)
        if type(self.schema) is not str or self.schema != UNIVERSE_SCHEMA:
            raise ValueError("unsupported universe schema")
        if type(self.version) is not int or self.version != UNIVERSE_VERSION:
            raise ValueError("unsupported universe version")
        for label in ("minecraft_version", "decoder_version", "support_version"):
            _text(getattr(self, label), label)
        _digest(self.source_archive_sha256, "source_archive_sha256")
        if type(self.manifest) is not LegalStateManifest:
            raise ValueError("universe manifest must be immutable")
        if self.manifest.minecraft_version != self.minecraft_version:
            raise ValueError("manifest Minecraft version must match universe")
        self.manifest.__post_init__()
        LegalStateManifest.from_record(self.manifest.to_record())
        if type(self.decoder_settings) is not tuple:
            raise ValueError("decoder settings must be immutable pairs")
        for pair in self.decoder_settings:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("decoder settings must be immutable pairs")
            for value in pair:
                _text(value, "decoder setting")
        if (self.decoder_settings != tuple(sorted(self.decoder_settings))
                or len({key for key, _ in self.decoder_settings}) != len(self.decoder_settings)):
            raise ValueError("decoder settings must be sorted with unique keys")
        if type(self.legal_states) is not tuple or not self.legal_states:
            raise ValueError("legal inventory must be a nonempty immutable tuple")
        ids = tuple(canonical_state_id(state) for state in self.legal_states)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("legal inventory must be sorted and contain no duplicate states")
        if (type(self.appearance_support) is not tuple
                or any(type(entry) is not AppearanceSupport for entry in self.appearance_support)):
            raise ValueError("appearance support must be immutable disposition records")
        if tuple(entry.state_id for entry in self.appearance_support) != ids:
            raise ValueError("every legal state requires exactly one ordered support disposition")
        for entry in self.appearance_support:
            _strings(entry.exclusion_reasons, "exclusion reasons")
            AppearanceSupport.from_record(entry.to_record())
        object.__setattr__(self, "legal_inventory_sha256", record_sha256(
            {"minecraft_version": self.minecraft_version, "manifest": self.manifest.to_record(),
             "states": [_state_record(state) for state in self.legal_states]}))
        object.__setattr__(self, "appearance_inventory_sha256", record_sha256(
            [entry.to_record() for entry in self.appearance_support]))
        object.__setattr__(self, "universe_sha256", record_sha256(self._payload()))

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(canonical_state_id(state) for state in self.legal_states)

    @property
    def supported_palette(self) -> tuple[PaletteColor, ...]:
        return tuple(entry.appearance for entry in self.appearance_support
                     if entry.disposition == "supported")

    @property
    def appearance_completeness(self) -> str:
        return ("exhaustive" if all(entry.disposition == "supported"
                                    for entry in self.appearance_support) else "unestablished")

    @property
    def requires_abstention(self) -> bool:
        return (self.manifest.completeness.status != "exhaustive"
                or self.appearance_completeness != "exhaustive")

    def _payload(self) -> dict:
        _preflight(self)
        return {"schema": self.schema, "version": self.version,
                "resource_contract": resource_contract(),
                "minecraft_version": self.minecraft_version,
                "source_archive_sha256": self.source_archive_sha256,
                "manifest": self.manifest.to_record(), "decoder_version": self.decoder_version,
                "support_version": self.support_version,
                "decoder_settings": [list(pair) for pair in self.decoder_settings],
                "legal_states": [_state_record(state) for state in self.legal_states],
                "state_ids": list(self.state_ids),
                "appearance_support": [entry.to_record() for entry in self.appearance_support],
                "appearance_completeness": self.appearance_completeness,
                "legal_inventory_sha256": self.legal_inventory_sha256,
                "appearance_inventory_sha256": self.appearance_inventory_sha256}

    def to_record(self) -> dict:
        _preflight(self)
        return {**self._payload(), "universe_sha256": self.universe_sha256}

    @classmethod
    def from_record(cls, record: Any) -> CandidateUniverse:
        _preflight(record)
        keys = {item.name for item in fields(cls)} | {
            "state_ids", "appearance_completeness", "resource_contract"}
        record = _record(record, keys, "universe")
        _replay(record["resource_contract"], resource_contract(), "resource contract")
        _inventory_counts(record["legal_states"], record["appearance_support"])
        for label in ("decoder_settings", "legal_states", "state_ids", "appearance_support"):
            if type(record[label]) is not list:
                raise ValueError(f"universe {label} record must be a list")
        settings = record["decoder_settings"]
        if any(type(pair) is not list or len(pair) != 2 for pair in settings):
            raise ValueError("decoder setting record must contain pairs")
        values = {item.name: record[item.name] for item in fields(cls) if item.init}
        values.update(manifest=LegalStateManifest.from_record(record["manifest"]),
                      decoder_settings=tuple(tuple(pair) for pair in settings),
                      legal_states=tuple(_state_from_record(state)
                                         for state in record["legal_states"]),
                      appearance_support=tuple(AppearanceSupport.from_record(entry)
                                               for entry in record["appearance_support"]))
        result = cls(**values)
        _replay(result.to_record(), record, "universe")
        return result
