"""Pure, versioned admission contract for frozen world/topology splits.

This module adds an explicit, opt-in metadata validator on top of the legacy
topology-only split functions. It answers one question and nothing else: given a
bounded, closed, in-memory collection of already-finalized dataset rows, can a
proposed train/dev/test admission be accepted without any connected group of rows
spanning folds through a shared declared world identity, source fingerprint,
topology digest, or explicit lineage alias?

It never allocates a fold, reads a file, rebuilds an index, mutates a frozen
manifest, or authorizes training. The legacy :mod:`splits` module keeps its exact
``SPLIT_SALT``, ``scene_group`` and ``assign_split`` semantics; this contract only
binds and verifies them. Everything here is standard-library only, and every
record is immutable and canonical, so the same input content always yields the
same manifest regardless of caller ordering.

Optional world-lineage evidence is retained: an admitted manifest binds every
alias, its declared provenance and its exact row membership into the canonical
digest, the deserializer and replay. Provenance is a caller declaration and
metadata only - it is never treated as authentication of real-world lineage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from mc_photogrammetry.dataset.splits import scene_group

#: New schema namespace. Deliberately distinct from the legacy dataset schema so
#: a world-aware manifest can never be mistaken for ``index.json``.
SCHEMA = "mc-photogrammetry-world-topology-split-admission-v1"
#: The single policy this schema implements. Recorded so a reader can bind the
#: manifest to the exact admission rule rather than to a claimed free-form label.
POLICY = "declared-metadata-world-topology-fold-consistency-v1"
#: Declared metadata proves separation only for the labels the caller supplied.
IDENTITY_SCOPE = "declared_metadata"
#: Structural admission is not training authorization and not E6 eligibility.
TRAINING_AUTHORIZED = False
INDEPENDENT_WORLD_ELIGIBILITY = "not_established"
SPLITS = ("train", "dev", "test")

# Structural record bounds. They are contract limits, never host allowances, and
# are checked before any input-sized copy of the manifest is materialized.
MAX_ROWS = 4096
MAX_SOURCES = 256
MAX_LINEAGE_ALIASES = 1024
MAX_COMPONENTS = 4096
MAX_IDENTITY_CHARS = 256
MAX_IDENTITY_BYTES = 1024
MAX_CANONICAL_BYTES = 8 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_MANIFEST_KEYS = frozenset({
    "schema", "policy", "identity_scope", "training_authorized",
    "independent_world_eligibility", "source_count", "row_count",
    "component_count", "lineage_alias_count", "sources", "rows", "components",
    "lineage_aliases", "manifest_sha256",
})
_SOURCE_KEYS = frozenset({"source_index", "expected_row_ids"})
_ROW_KEYS = frozenset({
    "row_id", "source_index", "scene_id", "fold", "world_identity",
    "source_fingerprint", "topology_digest", "scene_group",
})
_ALIAS_KEYS = frozenset({"alias", "provenance", "row_ids"})
_COMPONENT_KEYS = frozenset({"component_id", "row_ids", "fold", "folds"})


class WorldSplitContractError(ValueError):
    """Malformed, incomplete, over-limit, or conflicting admission input."""


class WorldSplitFoldConflict(WorldSplitContractError):
    """A connected component spans more than one original frozen fold.

    ``row_ids`` is the stable, sorted union of every row in the conflicting
    components; no fold is ever chosen or reassigned to resolve the conflict.
    """

    def __init__(self, row_ids: Any, folds: Any, component_ids: Any = ()) -> None:
        self.row_ids = tuple(sorted(row_ids))
        self.folds = tuple(sorted(folds))
        self.component_ids = tuple(component_ids)
        super().__init__(
            "connected world/source/topology/lineage component spans folds "
            f"{list(self.folds)} across rows {list(self.row_ids)}")


def _identity(value: Any, label: str) -> str:
    """Return a strict identity without coercion, folding, or normalization."""
    if type(value) is not str:
        raise WorldSplitContractError(f"{label} must be a string")
    if not value:
        raise WorldSplitContractError(f"{label} must not be empty")
    if len(value) > MAX_IDENTITY_CHARS:
        raise WorldSplitContractError(
            f"{label} exceeds {MAX_IDENTITY_CHARS} characters")
    if value != value.strip():
        raise WorldSplitContractError(
            f"{label} must not have leading or trailing whitespace")
    for character in value:
        code = ord(character)
        if code < 0x20 or code == 0x7F or 0x80 <= code <= 0x9F:
            raise WorldSplitContractError(f"{label} contains a control character")
        if 0xD800 <= code <= 0xDFFF:
            raise WorldSplitContractError(
                f"{label} contains an unpaired surrogate")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise WorldSplitContractError(f"{label} is not valid Unicode") from error
    if len(encoded) > MAX_IDENTITY_BYTES:
        raise WorldSplitContractError(
            f"{label} exceeds {MAX_IDENTITY_BYTES} UTF-8 bytes")
    return value


def _digest(value: Any, label: str) -> str:
    if type(value) is not str:
        raise WorldSplitContractError(f"{label} must be a string")
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise WorldSplitContractError(
            f"{label} must be a lowercase 64-character hex digest")
    return value


def _fold(value: Any) -> str:
    if type(value) is not str or value not in SPLITS:
        raise WorldSplitContractError(f"fold must be one of {SPLITS}")
    return value


def _list(value: Any, label: str, maximum: int) -> list[Any]:
    """Reject a non-list or an over-cap raw list before it is copied or parsed."""
    if type(value) is not list:
        raise WorldSplitContractError(f"{label} must be a list")
    if len(value) > maximum:
        raise WorldSplitContractError(f"{label} exceeds its cap")
    return value


def _count(value: Any, label: str, maximum: int) -> int:
    """Require an exact bounded integer; bool/float/integer-like values refuse."""
    if type(value) is not int:
        raise WorldSplitContractError(f"{label} must be an exact integer")
    if value < 0 or value > maximum:
        raise WorldSplitContractError(f"{label} is outside its bounds")
    return value


def _string_tuple(value: Any, label: str, *, minimum: int) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise WorldSplitContractError(f"{label} must be an exact tuple")
    if len(value) < minimum:
        raise WorldSplitContractError(f"{label} needs at least {minimum} entries")
    if len(value) > MAX_ROWS:
        raise WorldSplitContractError(f"{label} exceeds the row cap")
    seen: set[str] = set()
    for item in value:
        _identity(item, label)
        if item in seen:
            raise WorldSplitContractError(f"{label} contains a duplicate entry")
        seen.add(item)
    return tuple(sorted(seen))


def _canonical_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise WorldSplitContractError(
            "admission record is not canonical JSON data") from error


def _canonical_digest(payload: Any, limit: int) -> str:
    """Hash canonical JSON incrementally, aborting before the byte cap.

    Serialization stops as soon as the running UTF-8 size would exceed ``limit``,
    so an over-cap manifest is never fully materialized just to be measured.
    """
    hasher = hashlib.sha256()
    size = 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
    try:
        for chunk in encoder.iterencode(payload):
            encoded = chunk.encode("utf-8")
            size += len(encoded)
            if size > limit:
                raise WorldSplitContractError(
                    "admission manifest exceeds the canonical byte cap")
            hasher.update(encoded)
    except WorldSplitContractError:
        raise
    except (TypeError, ValueError, UnicodeError) as error:
        raise WorldSplitContractError(
            "admission record is not canonical JSON data") from error
    return hasher.hexdigest()


def _expect_record(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    """Refuse a record before any key iteration, copy, or allocation.

    Only an exact built-in ``dict`` is accepted, so a foreign
    :class:`~collections.abc.Mapping` can never run adversarial ``__iter__``,
    ``keys``, or ``__len__`` logic ahead of the schema check. For an accepted
    exact dict the cardinality is compared first, so ``set(value)`` only ever
    materializes a bounded, schema-sized key set.
    """
    if type(value) is not dict:
        raise WorldSplitContractError(
            f"{label} must be an exact built-in dict record")
    if len(value) != len(keys) or set(value) != keys:
        raise WorldSplitContractError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _component_id(row_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical_bytes(
        {"schema": SCHEMA, "component": list(row_ids)})).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceScope:
    """One declared source index and the exact rows admitted from it."""

    source_index: str
    expected_row_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identity(self.source_index, "source_index")
        object.__setattr__(
            self, "expected_row_ids",
            _string_tuple(self.expected_row_ids, "expected_row_ids", minimum=1))

    def to_record(self) -> dict[str, Any]:
        return {"source_index": self.source_index,
                "expected_row_ids": list(self.expected_row_ids)}

    @classmethod
    def from_record(cls, value: Any) -> "SourceScope":
        record = _expect_record(value, _SOURCE_KEYS, "source scope")
        raw_ids = _list(record["expected_row_ids"], "source scope row ids", MAX_ROWS)
        return cls(record["source_index"], tuple(raw_ids))


@dataclass(frozen=True, slots=True)
class SplitRow:
    """One frozen dataset row, bound to its declared identity metadata.

    ``scene_group`` must equal the frozen :func:`splits.scene_group` of
    ``topology_digest``; a claimed group that disagrees is refused rather than
    recomputed silently.
    """

    row_id: str
    source_index: str
    scene_id: str
    fold: str
    world_identity: str
    source_fingerprint: str
    topology_digest: str
    scene_group: str

    def __post_init__(self) -> None:
        _identity(self.row_id, "row_id")
        _identity(self.source_index, "source_index")
        _identity(self.scene_id, "scene_id")
        _fold(self.fold)
        _identity(self.world_identity, "world_identity")
        _digest(self.source_fingerprint, "source_fingerprint")
        _digest(self.topology_digest, "topology_digest")
        _digest(self.scene_group, "scene_group")
        if self.scene_group != scene_group(self.topology_digest):
            raise WorldSplitContractError(
                "scene_group must be the frozen scene_group(topology_digest) binding")

    def to_record(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "source_index": self.source_index,
            "scene_id": self.scene_id,
            "fold": self.fold,
            "world_identity": self.world_identity,
            "source_fingerprint": self.source_fingerprint,
            "topology_digest": self.topology_digest,
            "scene_group": self.scene_group,
        }

    @classmethod
    def from_record(cls, value: Any) -> "SplitRow":
        record = _expect_record(value, _ROW_KEYS, "split row")
        return cls(record["row_id"], record["source_index"], record["scene_id"],
                   record["fold"], record["world_identity"],
                   record["source_fingerprint"], record["topology_digest"],
                   record["scene_group"])


@dataclass(frozen=True, slots=True)
class LineageAlias:
    """Explicit, provenance-backed assertion that rows share one world lineage.

    The alias and its provenance are caller-supplied declarations retained in the
    manifest as evidence metadata. They are not authentication: this contract
    never proves independent real-world lineage from a label or a note. The alias
    may only join rows that already exist in the request.
    """

    alias: str
    provenance: str
    row_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identity(self.alias, "lineage alias")
        _identity(self.provenance, "lineage provenance")
        object.__setattr__(
            self, "row_ids",
            _string_tuple(self.row_ids, "lineage row_ids", minimum=2))

    def to_record(self) -> dict[str, Any]:
        return {"alias": self.alias, "provenance": self.provenance,
                "row_ids": list(self.row_ids)}

    @classmethod
    def from_record(cls, value: Any) -> "LineageAlias":
        record = _expect_record(value, _ALIAS_KEYS, "lineage alias")
        raw_ids = _list(record["row_ids"], "lineage row ids", MAX_ROWS)
        return cls(record["alias"], record["provenance"], tuple(raw_ids))


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    """A closed, bounded, in-memory admission request.

    Validation is structural and internal: it checks that the rows exactly and
    once-only cover each declared source index, that no two rows in one source
    reuse a scene identity, and that every identity is strictly well formed. It is
    not authentication of external provenance.
    """

    schema: str
    sources: tuple[SourceScope, ...]
    rows: tuple[SplitRow, ...]
    lineage_aliases: tuple[LineageAlias, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise WorldSplitContractError(
                f"unsupported admission schema {self.schema!r}")
        if type(self.sources) is not tuple or not self.sources:
            raise WorldSplitContractError(
                "admission request needs a non-empty tuple of source scopes")
        if len(self.sources) > MAX_SOURCES:
            raise WorldSplitContractError("admission request exceeds the source cap")
        if any(type(item) is not SourceScope for item in self.sources):
            raise WorldSplitContractError(
                "sources must be exact SourceScope records")
        if type(self.rows) is not tuple or not self.rows:
            raise WorldSplitContractError(
                "admission request needs a non-empty tuple of rows")
        if len(self.rows) > MAX_ROWS:
            raise WorldSplitContractError("admission request exceeds the row cap")
        if any(type(item) is not SplitRow for item in self.rows):
            raise WorldSplitContractError("rows must be exact SplitRow records")
        if type(self.lineage_aliases) is not tuple:
            raise WorldSplitContractError("lineage_aliases must be an exact tuple")
        if len(self.lineage_aliases) > MAX_LINEAGE_ALIASES:
            raise WorldSplitContractError(
                "admission request exceeds the lineage alias cap")
        if any(type(item) is not LineageAlias for item in self.lineage_aliases):
            raise WorldSplitContractError(
                "lineage aliases must be exact LineageAlias records")
        self._validate_coverage()
        self._validate_lineage()

    def _validate_coverage(self) -> None:
        declared: dict[str, set[str]] = {}
        for scope in self.sources:
            if scope.source_index in declared:
                raise WorldSplitContractError("duplicate declared source index")
            declared[scope.source_index] = set(scope.expected_row_ids)
        seen: set[str] = set()
        scenes: dict[tuple[str, str], str] = {}
        observed = {name: set() for name in declared}
        for row in self.rows:
            if row.source_index not in declared:
                raise WorldSplitContractError(
                    f"row {row.row_id!r} claims undeclared source index "
                    f"{row.source_index!r}")
            if row.row_id in seen:
                raise WorldSplitContractError(f"duplicate row id {row.row_id!r}")
            seen.add(row.row_id)
            scene_key = (row.source_index, row.scene_id)
            previous = scenes.get(scene_key)
            if previous is not None:
                raise WorldSplitContractError(
                    f"scene {row.scene_id!r} of source index "
                    f"{row.source_index!r} appears in rows {previous!r} and "
                    f"{row.row_id!r}")
            scenes[scene_key] = row.row_id
            observed[row.source_index].add(row.row_id)
        for name, expected in declared.items():
            missing = sorted(expected - observed[name])
            foreign = sorted(observed[name] - expected)
            if missing:
                raise WorldSplitContractError(
                    f"source index {name!r} is missing declared rows {missing}")
            if foreign:
                raise WorldSplitContractError(
                    f"source index {name!r} carries foreign rows {foreign}")

    def _validate_lineage(self) -> None:
        known = {row.row_id for row in self.rows}
        aliases: set[str] = set()
        assigned: dict[str, str] = {}
        for item in self.lineage_aliases:
            if item.alias in aliases:
                raise WorldSplitContractError(
                    f"duplicate lineage alias {item.alias!r}")
            aliases.add(item.alias)
            for row_id in item.row_ids:
                if row_id not in known:
                    raise WorldSplitContractError(
                        f"lineage alias {item.alias!r} names unknown row {row_id!r}")
                if row_id in assigned:
                    raise WorldSplitContractError(
                        f"row {row_id!r} is bound to multiple lineage aliases")
                assigned[row_id] = item.alias


@dataclass(frozen=True, slots=True)
class SplitComponent:
    """One transitively connected component and its single original fold."""

    component_id: str
    row_ids: tuple[str, ...]
    fold: str
    folds: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.component_id, "component_id")
        if type(self.row_ids) is not tuple or not self.row_ids:
            raise WorldSplitContractError("a component needs a tuple of row ids")
        if len(self.row_ids) > MAX_ROWS:
            raise WorldSplitContractError("component row ids exceed the row cap")
        # Bound every membership identity before the sorted/set copy or the
        # component-id serialization below, so an over-long, control, or
        # invalid-Unicode member can never reach the canonical hash.
        for item in self.row_ids:
            _identity(item, "component row ids")
        if self.row_ids != tuple(sorted(set(self.row_ids))):
            raise WorldSplitContractError(
                "component row ids must be sorted and unique")
        if _component_id(self.row_ids) != self.component_id:
            raise WorldSplitContractError(
                "component id does not match its membership")
        if type(self.folds) is not tuple or not self.folds:
            raise WorldSplitContractError("a component needs its original folds")
        if any(type(item) is not str or item not in SPLITS for item in self.folds):
            raise WorldSplitContractError("component folds are invalid")
        if self.folds != tuple(sorted(set(self.folds))):
            raise WorldSplitContractError(
                "component folds must be sorted and unique")
        if len(self.folds) == 1:
            if self.fold != self.folds[0]:
                raise WorldSplitContractError(
                    "component fold must match its single original fold")
        elif self.fold:
            raise WorldSplitContractError(
                "a conflicting component has no single fold")

    def to_record(self) -> dict[str, Any]:
        return {"component_id": self.component_id, "row_ids": list(self.row_ids),
                "fold": self.fold, "folds": list(self.folds)}

    @classmethod
    def from_record(cls, value: Any) -> "SplitComponent":
        record = _expect_record(value, _COMPONENT_KEYS, "split component")
        # Check the raw list cardinality first, then validate every membership
        # identity in a validation-only loop before the tuple copy or the
        # component-id serialization in ``__post_init__``. The loop deliberately
        # builds no second collection, so no earlier member is copied while any
        # later member is still unchecked.
        raw_ids = _list(record["row_ids"], "component row ids", MAX_ROWS)
        for item in raw_ids:
            _identity(item, "component row ids")
        raw_folds = _list(record["folds"], "component folds", len(SPLITS))
        return cls(record["component_id"], tuple(raw_ids), record["fold"],
                   tuple(raw_folds))


def _derive(request: AdmissionRequest) -> tuple[
        tuple[SourceScope, ...], tuple[SplitRow, ...],
        tuple[SplitComponent, ...], tuple[LineageAlias, ...]]:
    """Recompute the canonical sources, rows, components and aliases.

    This is the single derivation used by :func:`admit` and by manifest
    validation, so a deserialized record cannot claim components or folds that do
    not follow from its own rows and lineage evidence.
    """
    rows = request.rows
    if len(rows) > MAX_ROWS:
        raise WorldSplitContractError("admission exceeds the row cap")
    parent = {row.row_id: row.row_id for row in rows}

    def find(item: str) -> str:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    # Transitive equality over every declared identity dimension. ``scene_group``
    # is included because it is the frozen binding of ``topology_digest``.
    for attribute in ("world_identity", "source_fingerprint", "topology_digest",
                      "scene_group"):
        buckets: dict[str, list[str]] = {}
        for row in rows:
            buckets.setdefault(getattr(row, attribute), []).append(row.row_id)
        for members in buckets.values():
            for other in members[1:]:
                union(members[0], other)
    for item in request.lineage_aliases:
        for other in item.row_ids[1:]:
            union(item.row_ids[0], other)

    grouped: dict[str, list[SplitRow]] = {}
    for row in rows:
        grouped.setdefault(find(row.row_id), []).append(row)
    internal: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for members in grouped.values():
        ordered = sorted(members, key=lambda row: row.row_id)
        folds = tuple(sorted({row.fold for row in ordered}))
        internal.append((tuple(row.row_id for row in ordered), folds))
    internal.sort(key=lambda entry: entry[0])
    if len(internal) > MAX_COMPONENTS:
        raise WorldSplitContractError("admission exceeds the component cap")
    conflicts = [entry for entry in internal if len(entry[1]) > 1]
    if conflicts:
        raise WorldSplitFoldConflict(
            {row_id for ids, _folds in conflicts for row_id in ids},
            {fold for _ids, folds in conflicts for fold in folds},
            [_component_id(ids) for ids, _folds in conflicts])
    components = tuple(
        SplitComponent(_component_id(ids), ids, folds[0], folds)
        for ids, folds in internal)
    sources = tuple(sorted(request.sources, key=lambda scope: scope.source_index))
    ordered_rows = tuple(
        sorted(rows, key=lambda row: (row.source_index, row.row_id)))
    aliases = tuple(sorted(
        request.lineage_aliases,
        key=lambda item: (item.alias, item.provenance, item.row_ids)))
    return sources, ordered_rows, components, aliases


def _manifest_digest(manifest: "AdmissionManifest") -> str:
    return _canonical_digest(manifest._body(), MAX_CANONICAL_BYTES)


@dataclass(frozen=True, slots=True)
class AdmissionManifest:
    """Canonical, immutable admission result.

    The manifest binds the policy and schema, the closed source-index scope, every
    row, every component membership and its original fold, and every declared
    lineage alias with its provenance and row membership. It deliberately states
    ``identity_scope=declared_metadata``, ``training_authorized=false`` and
    ``independent_world_eligibility=not_established``: declared labels and
    provenance notes do not prove world lineage, so this record is never a
    training export.
    """

    sources: tuple[SourceScope, ...]
    rows: tuple[SplitRow, ...]
    components: tuple[SplitComponent, ...]
    lineage_aliases: tuple[LineageAlias, ...] = ()
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if (type(self.sources) is not tuple or not self.sources
                or any(type(item) is not SourceScope for item in self.sources)):
            raise WorldSplitContractError("manifest sources are invalid")
        if len(self.sources) > MAX_SOURCES:
            raise WorldSplitContractError("manifest exceeds the source cap")
        if (type(self.rows) is not tuple or not self.rows
                or any(type(item) is not SplitRow for item in self.rows)):
            raise WorldSplitContractError("manifest rows are invalid")
        if len(self.rows) > MAX_ROWS:
            raise WorldSplitContractError("manifest exceeds the row cap")
        if (type(self.components) is not tuple or not self.components
                or any(type(item) is not SplitComponent for item in self.components)):
            raise WorldSplitContractError("manifest components are invalid")
        if len(self.components) > MAX_COMPONENTS:
            raise WorldSplitContractError("manifest exceeds the component cap")
        if (type(self.lineage_aliases) is not tuple
                or any(type(item) is not LineageAlias
                       for item in self.lineage_aliases)):
            raise WorldSplitContractError("manifest lineage aliases are invalid")
        if len(self.lineage_aliases) > MAX_LINEAGE_ALIASES:
            raise WorldSplitContractError("manifest exceeds the lineage alias cap")
        # Rederive from the record's own rows and evidence. A correctly resealed
        # digest is not enough: coverage, folds and equivalence closure must all
        # agree with a fresh derivation.
        request = AdmissionRequest(SCHEMA, self.sources, self.rows,
                                   self.lineage_aliases)
        sources, rows, components, aliases = _derive(request)
        if sources != self.sources:
            raise WorldSplitContractError(
                "manifest sources must be canonically ordered and unique")
        if rows != self.rows:
            raise WorldSplitContractError("manifest rows must be canonically ordered")
        if components != self.components:
            raise WorldSplitContractError(
                "manifest components do not match the derived equivalence closure")
        if aliases != self.lineage_aliases:
            raise WorldSplitContractError(
                "manifest lineage aliases must be canonically ordered")
        computed = _manifest_digest(self)
        if self.manifest_sha256 not in ("", computed):
            raise WorldSplitContractError(
                "manifest digest does not match its canonical contents")
        object.__setattr__(self, "manifest_sha256", computed)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def lineage_alias_count(self) -> int:
        return len(self.lineage_aliases)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "policy": POLICY,
            "identity_scope": IDENTITY_SCOPE,
            "training_authorized": TRAINING_AUTHORIZED,
            "independent_world_eligibility": INDEPENDENT_WORLD_ELIGIBILITY,
            "source_count": len(self.sources),
            "row_count": len(self.rows),
            "component_count": len(self.components),
            "lineage_alias_count": len(self.lineage_aliases),
            "sources": [scope.to_record() for scope in self.sources],
            "rows": [row.to_record() for row in self.rows],
            "components": [component.to_record() for component in self.components],
            "lineage_aliases": [alias.to_record()
                                for alias in self.lineage_aliases],
        }

    def to_record(self) -> dict[str, Any]:
        record = self._body()
        record["manifest_sha256"] = self.manifest_sha256
        return record

    @classmethod
    def from_record(cls, value: Any) -> "AdmissionManifest":
        record = _expect_record(value, _MANIFEST_KEYS, "admission manifest")
        if (record["schema"] != SCHEMA or record["policy"] != POLICY
                or record["identity_scope"] != IDENTITY_SCOPE
                or record["training_authorized"] is not TRAINING_AUTHORIZED
                or record["independent_world_eligibility"]
                != INDEPENDENT_WORLD_ELIGIBILITY):
            raise WorldSplitContractError(
                "unsupported admission manifest schema, policy or authority")
        # Exact integer counts and their caps are enforced before any list is
        # copied, sorted, or turned into dataclasses.
        _count(record["source_count"], "source_count", MAX_SOURCES)
        _count(record["row_count"], "row_count", MAX_ROWS)
        _count(record["component_count"], "component_count", MAX_COMPONENTS)
        _count(record["lineage_alias_count"], "lineage_alias_count",
               MAX_LINEAGE_ALIASES)
        raw_sources = _list(record["sources"], "manifest sources", MAX_SOURCES)
        raw_rows = _list(record["rows"], "manifest rows", MAX_ROWS)
        raw_components = _list(record["components"], "manifest components",
                               MAX_COMPONENTS)
        raw_aliases = _list(record["lineage_aliases"], "manifest lineage aliases",
                            MAX_LINEAGE_ALIASES)
        if (len(raw_sources) != record["source_count"]
                or len(raw_rows) != record["row_count"]
                or len(raw_components) != record["component_count"]
                or len(raw_aliases) != record["lineage_alias_count"]):
            raise WorldSplitContractError(
                "manifest declared counts do not match their records")
        sources = tuple(SourceScope.from_record(item) for item in raw_sources)
        rows = tuple(SplitRow.from_record(item) for item in raw_rows)
        components = tuple(
            SplitComponent.from_record(item) for item in raw_components)
        aliases = tuple(LineageAlias.from_record(item) for item in raw_aliases)
        manifest = cls(sources, rows, components, aliases,
                       record["manifest_sha256"])
        if manifest.manifest_sha256 != record["manifest_sha256"]:
            raise WorldSplitContractError(
                "manifest digest does not match its canonical contents")
        return manifest


def admit(request: AdmissionRequest) -> AdmissionManifest:
    """Validate a closed request and return the canonical admission manifest.

    Refusal is atomic: any malformed, incomplete, over-limit or fold-spanning
    input raises and no partial manifest is returned.
    """
    if type(request) is not AdmissionRequest:
        raise WorldSplitContractError("admit requires an exact AdmissionRequest")
    sources, rows, components, aliases = _derive(request)
    return AdmissionManifest(sources, rows, components, aliases)


def replay(request: AdmissionRequest) -> AdmissionManifest:
    """Recompute the manifest from the supplied rows alone."""
    return admit(request)


def verify_replay(request: AdmissionRequest,
                  claimed: AdmissionManifest | Mapping[str, Any]) -> AdmissionManifest:
    """Recompute from the request and compare against a claimed manifest.

    Claimed component ids, lineage evidence and digests are never trusted on their
    own: the result is always rederived from the supplied rows and must match
    byte for byte.
    """
    derived = admit(request)
    if isinstance(claimed, AdmissionManifest):
        record = claimed.to_record()
    elif isinstance(claimed, Mapping):
        record = AdmissionManifest.from_record(claimed).to_record()
    else:
        raise WorldSplitContractError(
            "claimed manifest must be an AdmissionManifest or a record mapping")
    if _canonical_bytes(record) != _canonical_bytes(derived.to_record()):
        raise WorldSplitContractError(
            "claimed manifest does not match the replayed derivation")
    return derived
