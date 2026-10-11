"""Versioned schemas for the real-Minecraft capture dataset factory.

These records are the contract between three groups of code that cannot import
each other: the Python planner/finalizer, the Fabric client mod that performs a
capture, and any later evaluation reader. The mod side is Java, so every record
here round-trips through plain JSON and every reader rejects an unknown
``schema_version`` instead of guessing.

Ground-truth capture metadata (``CameraCapture``, ``GroundTruthExport``,
``capture.json``) is developer/evaluation scaffolding. It is never an input to
normal reconstruction: a plan is generated from explicit developer regions, and
the solved cameras produced from those screenshots must still stand on their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mc_photogrammetry.reconstruction.camera import CameraModel

SCHEMA_VERSION = 1
MOD_CONTRACT_VERSION = 1
GENERATOR_VERSION = "ds17r-1"

WEATHERS = ("clear", "rain", "thunder")
FAMILIES = ("house", "tower", "courtyard", "bridge", "landscape", "custom")

#: Hard cap on jobs in one plan. The mod refuses a plan beyond this so a bad
#: planner cannot drive an unbounded client session.
MAX_PLAN_JOBS = 64
#: Hard cap on the axis length of one exported evaluation region.
MAX_REGION_EDGE = 512
#: Hard cap on the block count of one exported region. The Fabric mod builds the
#: ground truth on the client tick, so a plan the Python side accepts must be one
#: the mod will still export; the two caps must agree.
MAX_REGION_VOLUME = 250_000


class DatasetSchemaError(ValueError):
    """Raised when a dataset record is malformed or has an unsupported version."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetSchemaError(f"{label} must be a JSON object")
    return value


def _int_tuple(value: Any, label: str, length: int = 3) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise DatasetSchemaError(f"{label} must contain {length} values")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise DatasetSchemaError(f"{label} must contain whole numbers") from error


def _float_tuple(value: Any, label: str, length: int = 3) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise DatasetSchemaError(f"{label} must contain {length} values")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise DatasetSchemaError(f"{label} must contain numbers") from error


def _check_version(data: Mapping[str, Any], label: str) -> None:
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DatasetSchemaError(f"{label} has unsupported schema_version {version!r}")


@dataclass(frozen=True, slots=True)
class RegionBounds:
    """Inclusive block bounds of one developer-declared evaluation region."""

    minimum: tuple[int, int, int]
    maximum: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.minimum) != 3 or len(self.maximum) != 3:
            raise DatasetSchemaError("region bounds must have three coordinates")
        if any(self.maximum[index] < self.minimum[index] for index in range(3)):
            raise DatasetSchemaError("region maximum must not be below its minimum")
        if any(self.maximum[index] - self.minimum[index] + 1 > MAX_REGION_EDGE
               for index in range(3)):
            raise DatasetSchemaError(
                f"region edge must not exceed {MAX_REGION_EDGE} blocks")
        volume = 1
        for index in range(3):
            volume *= self.maximum[index] - self.minimum[index] + 1
        if volume > MAX_REGION_VOLUME:
            raise DatasetSchemaError(
                f"region volume {volume} exceeds {MAX_REGION_VOLUME} blocks")

    @property
    def size(self) -> tuple[int, int, int]:
        return tuple(self.maximum[index] - self.minimum[index] + 1 for index in range(3))

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((self.minimum[index] + self.maximum[index]) / 2.0
                     for index in range(3))  # type: ignore[return-value]

    @property
    def diagonal(self) -> float:
        return sum(float(self.size[index]) ** 2 for index in range(3)) ** 0.5

    def contains(self, coordinate: tuple[int, int, int]) -> bool:
        return all(self.minimum[index] <= coordinate[index] <= self.maximum[index]
                   for index in range(3))

    def to_dict(self) -> dict[str, Any]:
        return {"min": list(self.minimum), "max": list(self.maximum)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegionBounds":
        data = _mapping(data, "region bounds")
        return cls(_int_tuple(data.get("min"), "region min"),
                   _int_tuple(data.get("max"), "region max"))


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    """Requested client options for one capture; actuals are recorded separately."""

    gamma: float = 0.0
    time_of_day: int = 6000
    weather: str = "clear"
    hud_hidden: bool = False
    resource_pack: str | None = None
    shader_pack: str | None = None

    def __post_init__(self) -> None:
        if self.weather not in WEATHERS:
            raise DatasetSchemaError(f"weather must be one of {WEATHERS}")
        if not 0 <= int(self.time_of_day) < 24000:
            raise DatasetSchemaError("time_of_day must be in [0, 24000)")
        # Minecraft's brightness option validates its value in [0, 1], so a request
        # outside that range can never be honoured; refusing it here keeps a plan
        # from promising a setting the client will silently ignore.
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise DatasetSchemaError("gamma must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "time_of_day": int(self.time_of_day),
            "weather": self.weather,
            "hud_hidden": bool(self.hud_hidden),
            "resource_pack": self.resource_pack,
            "shader_pack": self.shader_pack,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaptureSettings":
        data = _mapping(data, "capture settings")
        return cls(
            gamma=float(data.get("gamma", 0.0)),
            time_of_day=int(data.get("time_of_day", 6000)),
            weather=str(data.get("weather", "clear")),
            hud_hidden=bool(data.get("hud_hidden", False)),
            resource_pack=None if data.get("resource_pack") is None else str(data["resource_pack"]),
            shader_pack=None if data.get("shader_pack") is None else str(data["shader_pack"]),
        )


def capture_settings_from_payload(data: Mapping[str, Any], *,
                                  require_complete: bool = False,
                                  label: str = "capture settings") -> CaptureSettings:
    """Parse a JSON settings payload without permissive Python coercions.

    Batch specifications and command-line JSON are user-authored contracts, so
    accepting values such as ``"false"`` for ``hud_hidden`` or ``6.5`` for
    ``time_of_day`` would silently change their meaning.  Existing serialized
    plans continue to use :meth:`CaptureSettings.from_dict` for compatibility;
    new external requests should use this strict loader.
    """
    if not isinstance(data, Mapping):
        raise DatasetSchemaError(f"{label} must be a JSON object")
    allowed = {"gamma", "time_of_day", "weather", "hud_hidden",
               "resource_pack", "shader_pack"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise DatasetSchemaError(f"{label} has unsupported keys {unknown}")
    core = {"gamma", "time_of_day", "weather", "hud_hidden"}
    if require_complete:
        missing = sorted(core - set(data))
        if missing:
            raise DatasetSchemaError(
                f"{label} must contain all core keys {sorted(core)}; missing {missing}")

    gamma = data.get("gamma", 0.0)
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise DatasetSchemaError(f"{label}.gamma must be a number (not bool)")
    time_of_day = data.get("time_of_day", 6000)
    if isinstance(time_of_day, bool) or not isinstance(time_of_day, int):
        raise DatasetSchemaError(f"{label}.time_of_day must be an integer (not bool)")
    weather = data.get("weather", "clear")
    if not isinstance(weather, str):
        raise DatasetSchemaError(f"{label}.weather must be a string")
    hud_hidden = data.get("hud_hidden", False)
    if not isinstance(hud_hidden, bool):
        raise DatasetSchemaError(f"{label}.hud_hidden must be a boolean")

    def pack_value(key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise DatasetSchemaError(f"{label}.{key} must be a string or null")
        if not value.strip() or any(character in value for character in ("\r", "\n", "\0")):
            raise DatasetSchemaError(
                f"{label}.{key} must be a non-empty single-line string or null")
        return value

    return CaptureSettings(
        gamma=float(gamma),
        time_of_day=time_of_day,
        weather=weather,
        hud_hidden=hud_hidden,
        resource_pack=pack_value("resource_pack"),
        shader_pack=pack_value("shader_pack"),
    )


@dataclass(frozen=True, slots=True)
class CameraCapture:
    """One planned view: where the client is put, and what it should look at.

    ``camera`` is a ground-truth pose. It is recorded so a dataset can be
    re-captured and so finalization can check that the client actually honoured
    the request; production reconstruction is not allowed to read it.
    """

    frame_id: str
    camera: CameraModel
    resolution: tuple[int, int]
    target: tuple[float, float, float]
    settings: CaptureSettings = field(default_factory=CaptureSettings)
    settle_ticks: int = 20

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise DatasetSchemaError("frame_id must not be empty")
        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise DatasetSchemaError("resolution must be positive")
        if len(self.target) != 3:
            raise DatasetSchemaError("target must have three coordinates")
        if self.settle_ticks < 0:
            raise DatasetSchemaError("settle_ticks must not be negative")

    @property
    def filename(self) -> str:
        return f"{self.frame_id}.png"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frame_id": self.frame_id,
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "target": [float(value) for value in self.target],
            "settle_ticks": int(self.settle_ticks),
            "settings": self.settings.to_dict(),
            "camera": self.camera.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraCapture":
        data = _mapping(data, "camera capture")
        _check_version(data, "camera capture")
        resolution = _int_tuple(data.get("resolution"), "resolution", length=2)
        camera = CameraModel.from_dict(_mapping(data.get("camera"), "camera capture camera"))
        stored = camera.intrinsics
        if stored is not None and stored.image_size != resolution:
            raise DatasetSchemaError(
                "camera intrinsics size must match the declared resolution")
        return cls(
            frame_id=str(data.get("frame_id", "")),
            camera=camera,
            resolution=(int(resolution[0]), int(resolution[1])),
            target=_float_tuple(data.get("target"), "target"),
            settings=CaptureSettings.from_dict(_mapping(data.get("settings", {}),
                                                        "capture settings")),
            settle_ticks=int(data.get("settle_ticks", 20)),
        )


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """The armed.json payload: a bounded capture plan for one staged world.

    ``context_export_only`` is an evaluation-only, zero-job contract used to
    export a larger context region without capturing screenshots. Standard
    capture plans remain non-empty; the finalizer explicitly refuses this mode.
    """

    plan_id: str
    scene_id: str
    dataset_id: str
    world_identity: str
    region: RegionBounds
    jobs: tuple[CameraCapture, ...]
    staging_token: str = ""
    mod_contract_version: int = MOD_CONTRACT_VERSION
    notes: tuple[str, ...] = ()
    context_export_only: bool = False

    def __post_init__(self) -> None:
        if self.mod_contract_version != MOD_CONTRACT_VERSION:
            raise DatasetSchemaError(
                f"unsupported mod contract version {self.mod_contract_version!r}")
        for label, value in (("plan_id", self.plan_id), ("scene_id", self.scene_id),
                            ("dataset_id", self.dataset_id),
                            ("world_identity", self.world_identity)):
            if not value:
                raise DatasetSchemaError(f"{label} must not be empty")
        # The staging token is the SHA-256 fingerprint of the save the plan was
        # cut from. The mod recomputes it over the world it opened and refuses to
        # mutate anything when it differs, so a plan cannot leak onto a world that
        # merely has the same level name.
        if not self.staging_token:
            raise DatasetSchemaError("a plan must carry a staging_token")
        if not all(character in "0123456789abcdef" for character in self.staging_token.lower()):
            raise DatasetSchemaError("staging_token must be a lowercase hex digest")
        if len(self.staging_token) != 64:
            raise DatasetSchemaError("staging_token must be a 64-character SHA-256 digest")
        if type(self.context_export_only) is not bool:
            raise DatasetSchemaError("context_export_only must be a boolean")
        if self.context_export_only and self.jobs:
            raise DatasetSchemaError(
                "a context-export-only plan must not contain capture jobs")
        if not self.context_export_only and not self.jobs:
            raise DatasetSchemaError("a plan must contain at least one capture job")
        if len(self.jobs) > MAX_PLAN_JOBS:
            raise DatasetSchemaError(f"a plan must not exceed {MAX_PLAN_JOBS} jobs")
        frame_ids = [job.frame_id for job in self.jobs]
        if len(set(frame_ids)) != len(frame_ids):
            raise DatasetSchemaError("plan frame_id values must be unique")
        for job in self.jobs:
            if not set(job.frame_id).issubset(
                    set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")):
                raise DatasetSchemaError(
                    f"frame_id {job.frame_id!r} must be filename-safe")

    @property
    def filenames(self) -> tuple[str, ...]:
        return tuple(job.filename for job in self.jobs)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "mod_contract_version": int(self.mod_contract_version),
            "plan_id": self.plan_id,
            "scene_id": self.scene_id,
            "dataset_id": self.dataset_id,
            "world_identity": self.world_identity,
            "staging_token": self.staging_token,
            "region": self.region.to_dict(),
            "jobs": [job.to_dict() for job in self.jobs],
            "notes": list(self.notes),
        }
        if self.context_export_only:
            payload["context_export_only"] = True
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapturePlan":
        data = _mapping(data, "capture plan")
        _check_version(data, "capture plan")
        raw_jobs = data.get("jobs")
        if not isinstance(raw_jobs, (list, tuple)):
            raise DatasetSchemaError("capture plan jobs must be a list")
        context_export_only = data.get("context_export_only", False)
        if type(context_export_only) is not bool:
            raise DatasetSchemaError("capture plan context_export_only must be a boolean")
        if context_export_only and raw_jobs:
            raise DatasetSchemaError(
                "a context-export-only plan must not contain capture jobs")
        return cls(
            plan_id=str(data.get("plan_id", "")),
            scene_id=str(data.get("scene_id", "")),
            dataset_id=str(data.get("dataset_id", "")),
            world_identity=str(data.get("world_identity", "")),
            region=RegionBounds.from_dict(_mapping(data.get("region"), "capture plan region")),
            jobs=tuple(CameraCapture.from_dict(job) for job in raw_jobs),
            staging_token=str(data.get("staging_token", "")),
            mod_contract_version=int(data.get("mod_contract_version", 0)),
            notes=tuple(str(note) for note in data.get("notes", [])),
            context_export_only=context_export_only,
        )


@dataclass(frozen=True, slots=True)
class CaptureFrameResult:
    """What the mod actually observed for one job. Failures are recorded, not hidden."""

    frame_id: str
    filename: str
    captured: bool
    screenshot: str | None = None
    settled_ticks: int = 0
    actual: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frame_id": self.frame_id,
            "filename": self.filename,
            "captured": bool(self.captured),
            "screenshot": self.screenshot,
            "settled_ticks": int(self.settled_ticks),
            "actual": dict(self.actual),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaptureFrameResult":
        data = _mapping(data, "capture frame result")
        _check_version(data, "capture frame result")
        actual = data.get("actual", {})
        if not isinstance(actual, Mapping):
            raise DatasetSchemaError("capture frame actual values must be an object")
        return cls(
            frame_id=str(data.get("frame_id", "")),
            filename=str(data.get("filename", "")),
            captured=bool(data.get("captured", False)),
            screenshot=None if data.get("screenshot") is None else str(data["screenshot"]),
            settled_ticks=int(data.get("settled_ticks", 0)),
            actual=dict(actual),
            errors=tuple(str(error) for error in data.get("errors", [])),
        )


@dataclass(frozen=True, slots=True)
class BlockRecord:
    """One exact blockstate exported from the server world for a region."""

    position: tuple[int, int, int]
    name: str
    properties: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if ":" not in self.name:
            raise DatasetSchemaError("block names must be namespaced")
        if len(self.position) != 3:
            raise DatasetSchemaError("block positions must have three coordinates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": [int(value) for value in self.position],
            "name": self.name,
            "properties": {str(key): str(value)
                           for key, value in sorted(self.properties.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlockRecord":
        data = _mapping(data, "block record")
        properties = data.get("properties", {})
        if not isinstance(properties, Mapping):
            raise DatasetSchemaError("block properties must be an object")
        return cls(
            position=_int_tuple(data.get("pos"), "block position"),
            name=str(data.get("name", "")),
            properties={str(key): str(value) for key, value in properties.items()},
        )


@dataclass(frozen=True, slots=True)
class GroundTruthExport:
    """The mod-side ground truth for one scene: exact states over declared bounds."""

    scene_id: str
    world_identity: str
    region: RegionBounds
    blocks: tuple[BlockRecord, ...]
    mod_contract_version: int = MOD_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.mod_contract_version != MOD_CONTRACT_VERSION:
            raise DatasetSchemaError(
                f"unsupported mod contract version {self.mod_contract_version!r}")
        if not self.scene_id or not self.world_identity:
            raise DatasetSchemaError("ground truth needs a scene_id and world_identity")
        seen: set[tuple[int, int, int]] = set()
        for block in self.blocks:
            if not self.region.contains(block.position):
                raise DatasetSchemaError(
                    f"block {block.position} lies outside the declared region")
            if block.position in seen:
                raise DatasetSchemaError(f"duplicate block at {block.position}")
            seen.add(block.position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mod_contract_version": int(self.mod_contract_version),
            "scene_id": self.scene_id,
            "world_identity": self.world_identity,
            "region": self.region.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundTruthExport":
        data = _mapping(data, "ground truth")
        _check_version(data, "ground truth")
        raw_blocks = data.get("blocks")
        if not isinstance(raw_blocks, (list, tuple)):
            raise DatasetSchemaError("ground truth blocks must be a list")
        return cls(
            scene_id=str(data.get("scene_id", "")),
            world_identity=str(data.get("world_identity", "")),
            region=RegionBounds.from_dict(_mapping(data.get("region"), "ground truth region")),
            blocks=tuple(BlockRecord.from_dict(block) for block in raw_blocks),
            mod_contract_version=int(data.get("mod_contract_version", 0)),
        )


@dataclass(frozen=True, slots=True)
class SceneRecipe:
    """Versioned description of what a scene is and how it was produced."""

    seed: int
    family: str
    dimensions: tuple[int, int, int]
    palette: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    source: str = "procedural"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DatasetSchemaError(
                f"unsupported recipe schema version {self.schema_version!r}")
        if any(int(edge) <= 0 for edge in self.dimensions):
            raise DatasetSchemaError("recipe dimensions must be positive")
        for name in self.palette:
            if not name:
                raise DatasetSchemaError("palette entries must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "seed": int(self.seed),
            "family": self.family,
            "dimensions": [int(edge) for edge in self.dimensions],
            "palette": list(self.palette),
            "features": list(self.features),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneRecipe":
        data = _mapping(data, "scene recipe")
        _check_version(data, "scene recipe")
        return cls(
            seed=int(data.get("seed", 0)),
            family=str(data.get("family", "custom")),
            dimensions=_int_tuple(data.get("dimensions"), "recipe dimensions"),
            palette=tuple(str(name) for name in data.get("palette", [])),
            features=tuple(str(name) for name in data.get("features", [])),
            source=str(data.get("source", "procedural")),
            schema_version=int(data.get("schema_version", 0)),
        )


@dataclass(slots=True)
class GeneratedScene:
    """A concrete scene: its recipe, its blocks, its deterministic split."""

    scene_id: str
    recipe: SceneRecipe
    bounds: RegionBounds
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in ("train", "dev", "test"):
            raise DatasetSchemaError(f"unknown split {self.split!r}")
