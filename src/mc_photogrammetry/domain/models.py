from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

BlockCoordinate = tuple[int, int, int]


class ObservationKind(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"


@dataclass(frozen=True, slots=True)
class BlockState:
    """A namespaced Minecraft block with optional state properties."""

    name: str
    properties: tuple[tuple[str, str], ...] = ()
    _identity: tuple[str, tuple[tuple[str, str], ...]] | None = field(
        init=False, repr=False, compare=False, default=None
    )
    """Name plus default-stripped properties, computed once.

    Equality and hashing both need the normalised form, and block states are the
    keys of every per-candidate score in the classifier - tens of millions of
    lookups in one reconstruction. Recomputing the normalisation on each of those
    dominated the runtime.
    """

    def __post_init__(self) -> None:
        if ":" not in self.name:
            raise ValueError("block names must use a namespace, for example minecraft:stone")
        if any(not key or not value for key, value in self.properties):
            raise ValueError("block state properties must have non-empty keys and values")
        object.__setattr__(self, "_identity", (self.name, self._normalized_properties()))

    def _normalized_properties(self) -> tuple[tuple[str, str], ...]:
        name = self.name.rsplit(":", 1)[-1].lower()
        normalized = dict(self.properties)
        defaults: dict[str, str] = {}
        if "stairs" in name:
            defaults = {"facing": "north", "half": "bottom", "shape": "straight", "waterlogged": "false"}
        elif "slab" in name:
            defaults = {"type": "bottom", "waterlogged": "false"}
        elif "fence" in name:
            defaults = {"east": "false", "north": "false", "south": "false", "west": "false", "waterlogged": "false"}
        elif name.endswith("_log") or name.endswith("_wood"):
            defaults = {"axis": "y"}
        for key, value in defaults.items():
            if normalized.get(key) == value:
                normalized.pop(key, None)
        return tuple(sorted(normalized.items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockState):
            return NotImplemented
        return self._identity == other._identity

    def __hash__(self) -> int:
        return hash(self._identity)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "properties": dict(self.properties)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BlockState:
        raw_properties = data.get("properties", {})
        if not isinstance(raw_properties, dict):
            raise ValueError("block state properties must be an object")
        properties = tuple(sorted((str(key), str(value)) for key, value in raw_properties.items()))
        return cls(name=str(data["name"]), properties=properties)


@dataclass(slots=True)
class BlockObservation:
    state: BlockState
    confidence: float = 1.0
    kind: ObservationKind = ObservationKind.OBSERVED
    frame_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.to_dict(),
            "confidence": self.confidence,
            "kind": self.kind.value,
            "frame_ids": self.frame_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BlockObservation:
        return cls(
            state=BlockState.from_dict(data["state"]),  # type: ignore[arg-type]
            confidence=float(data["confidence"]),
            kind=ObservationKind(str(data["kind"])),
            frame_ids=[str(frame_id) for frame_id in data.get("frame_ids", [])],
        )


@dataclass(slots=True)
class BlockVolume:
    """Sparse block storage; empty space is represented by absent coordinates."""

    blocks: dict[BlockCoordinate, BlockObservation] = field(default_factory=dict)

    def set(self, coordinate: BlockCoordinate, observation: BlockObservation) -> None:
        if len(coordinate) != 3 or any(not isinstance(value, int) for value in coordinate):
            raise ValueError("block coordinates must be a tuple of three integers")
        self.blocks[coordinate] = observation

    def get(self, coordinate: BlockCoordinate) -> BlockObservation | None:
        return self.blocks.get(coordinate)

    def items(self) -> Iterable[tuple[BlockCoordinate, BlockObservation]]:
        return self.blocks.items()

    def to_dict(self) -> dict[str, object]:
        entries = []
        for (x, y, z), observation in sorted(self.blocks.items()):
            entries.append({"coordinate": [x, y, z], "observation": observation.to_dict()})
        return {"blocks": entries}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BlockVolume:
        volume = cls()
        for entry in data.get("blocks", []):
            if not isinstance(entry, dict):
                raise ValueError("block entries must be objects")
            coordinate = tuple(int(value) for value in entry["coordinate"])
            if len(coordinate) != 3:
                raise ValueError("block coordinates must contain three values")
            volume.set(coordinate, BlockObservation.from_dict(entry["observation"]))  # type: ignore[arg-type]
        return volume


SCAN_PROJECT_LEGACY_SCHEMA_VERSION = 1
"""The original project payload: no camera/lattice/session record."""

SCAN_PROJECT_SCHEMA_VERSION = 2
"""The default schema for newly saved projects; adds the optional session record."""

IMAGE_DERIVED_SESSION_FIELD = "image_derived_session"
"""Schema-2 key carrying the canonical image-derived session record, or ``null``."""

SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION = 3
"""The durable image-derived schema: session record plus terminal envelope and volume."""

IMAGE_DERIVED_PRODUCTION_FIELD = "image_derived_production"
"""Schema-3 key carrying the canonical digest-bound terminal envelope."""

_LEGACY_PROJECT_KEYS = frozenset({
    "schema_version", "name", "minecraft_version", "source_paths", "volume"})
_SCHEMA2_PROJECT_KEYS = _LEGACY_PROJECT_KEYS | {IMAGE_DERIVED_SESSION_FIELD}
_SCHEMA3_PROJECT_KEYS = _SCHEMA2_PROJECT_KEYS | {IMAGE_DERIVED_PRODUCTION_FIELD}
_FORBIDDEN_PROJECT_KEYS = frozenset({
    "pixels", "pixel_data", "image", "images", "cameras", "camera_manifest",
    "f3", "coordinates", "directions", "bounds", "scene", "scene_transform",
    "scale", "translation", "metadata", "annotations",
})
"""Names a schema-2 project payload may never carry; image-derived evidence must
arrive only through the digest-bound session record."""


@dataclass(slots=True)
class ScanProject:
    """Serializable project state shared by the UI, pipeline, and exporters.

    ``image_derived_session`` is the canonical record produced by
    :func:`mc_photogrammetry.pipeline.image_derived_session.image_derived_session_to_record`
    (or ``None``). It is a plain JSON mapping so the domain layer stays free of
    the pipeline import it would otherwise need; :mod:`mc_photogrammetry.project.store`
    validates it through ``image_derived_session_from_record`` at the persistence
    boundary. The field carries no pixels, camera manifest, manual coordinate,
    direction, bound, scene metadata, or scale/translation claim.
    """

    name: str = "Untitled scan"
    minecraft_version: str = "1.21.1"
    volume: BlockVolume = field(default_factory=BlockVolume)
    source_paths: list[str] = field(default_factory=list)
    schema_version: int = SCAN_PROJECT_SCHEMA_VERSION
    image_derived_session: dict[str, object] | None = None
    image_derived_production: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        if self.schema_version == SCAN_PROJECT_LEGACY_SCHEMA_VERSION:
            if self.image_derived_session is not None:
                raise ValueError(
                    "schema-1 scan projects cannot carry an image-derived session")
            if self.image_derived_production is not None:
                raise ValueError(
                    "schema-1 scan projects cannot carry an image-derived production record")
            return {
                "schema_version": SCAN_PROJECT_LEGACY_SCHEMA_VERSION,
                "name": self.name,
                "minecraft_version": self.minecraft_version,
                "source_paths": self.source_paths,
                "volume": self.volume.to_dict(),
            }
        if self.schema_version == SCAN_PROJECT_SCHEMA_VERSION:
            if self.image_derived_production is not None:
                raise ValueError(
                    "schema-2 scan projects cannot carry an image-derived production record")
            return {
                "schema_version": SCAN_PROJECT_SCHEMA_VERSION,
                "name": self.name,
                "minecraft_version": self.minecraft_version,
                "source_paths": self.source_paths,
                "volume": self.volume.to_dict(),
                IMAGE_DERIVED_SESSION_FIELD: _session_record_copy(
                    self.image_derived_session),
            }
        if self.schema_version != SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported scan project schema: {self.schema_version}")
        if (self.image_derived_session is None
                or self.image_derived_production is None):
            raise ValueError(
                "schema-3 scan projects require an image-derived session and "
                "production record")
        return {
            "schema_version": SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
            "name": self.name,
            "minecraft_version": self.minecraft_version,
            "source_paths": self.source_paths,
            "volume": self.volume.to_dict(),
            IMAGE_DERIVED_SESSION_FIELD: _session_record_copy(
                self.image_derived_session),
            IMAGE_DERIVED_PRODUCTION_FIELD: _production_record_copy(
                self.image_derived_production),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ScanProject:
        if not isinstance(data, Mapping):
            raise ValueError("scan project root must be a mapping")
        schema_version = data.get("schema_version", 0)
        if type(schema_version) is not int:
            raise ValueError("scan project schema_version must be an integer")
        if schema_version == SCAN_PROJECT_LEGACY_SCHEMA_VERSION:
            if IMAGE_DERIVED_SESSION_FIELD in data:
                raise ValueError(
                    "schema-1 scan projects cannot carry an image-derived session")
            if IMAGE_DERIVED_PRODUCTION_FIELD in data:
                raise ValueError(
                    "schema-1 scan projects cannot carry an image-derived production record")
            _reject_unknown_project_keys(
                data, _LEGACY_PROJECT_KEYS, SCAN_PROJECT_LEGACY_SCHEMA_VERSION)
            return cls(
                name=str(data["name"]),
                minecraft_version=str(data["minecraft_version"]),
                source_paths=[str(path) for path in data.get("source_paths", [])],
                volume=BlockVolume.from_dict(data["volume"]),  # type: ignore[arg-type]
                schema_version=SCAN_PROJECT_LEGACY_SCHEMA_VERSION,
                image_derived_session=None,
                image_derived_production=None,
            )
        if schema_version == SCAN_PROJECT_SCHEMA_VERSION:
            _reject_unknown_project_keys(
                data, _SCHEMA2_PROJECT_KEYS, SCAN_PROJECT_SCHEMA_VERSION)
            return cls(
                name=str(data["name"]),
                minecraft_version=str(data["minecraft_version"]),
                source_paths=[str(path) for path in data.get("source_paths", [])],
                volume=BlockVolume.from_dict(data["volume"]),  # type: ignore[arg-type]
                schema_version=SCAN_PROJECT_SCHEMA_VERSION,
                image_derived_session=_session_record_copy(
                    data.get(IMAGE_DERIVED_SESSION_FIELD)),
            )
        if schema_version == SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION:
            _reject_unknown_project_keys(
                data, _SCHEMA3_PROJECT_KEYS,
                SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION)
            session = _session_record_copy(data.get(IMAGE_DERIVED_SESSION_FIELD))
            if session is None:
                raise ValueError(
                    "schema-3 scan projects require an image-derived session record")
            production = _production_record_copy(
                data.get(IMAGE_DERIVED_PRODUCTION_FIELD))
            if production is None:
                raise ValueError(
                    "schema-3 scan projects require an image-derived production record")
            return cls(
                name=str(data["name"]),
                minecraft_version=str(data["minecraft_version"]),
                source_paths=[str(path) for path in data.get("source_paths", [])],
                volume=BlockVolume.from_dict(data["volume"]),  # type: ignore[arg-type]
                schema_version=SCAN_PROJECT_PRODUCTION_SCHEMA_VERSION,
                image_derived_session=session,
                image_derived_production=production,
            )
        raise ValueError(f"unsupported scan project schema: {schema_version}")


def _session_record_copy(record: object) -> dict[str, object] | None:
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValueError("image_derived_session must be a record mapping or null")
    return dict(record)


def _production_record_copy(record: object) -> dict[str, object] | None:
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValueError(
            "image_derived_production must be a record mapping or null")
    return dict(record)


def _reject_unknown_project_keys(
    data: Mapping[str, object], allowed: frozenset[str], schema_version: int,
) -> None:
    forbidden = sorted(set(data) & _FORBIDDEN_PROJECT_KEYS)
    if forbidden:
        raise ValueError(
            f"scan project schema {schema_version} cannot carry {forbidden}")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"scan project schema {schema_version} has unexpected keys: {unknown}")
