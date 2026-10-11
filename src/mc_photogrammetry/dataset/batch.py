"""Deterministic, launch-free bulk capture queue for the single-scene factory.

The accepted factory in :mod:`mc_photogrammetry.dataset.plan` /
``inventory`` / ``finalize`` prepares and finalizes **one** scene at a time.  This
module turns a declarative JSON batch spec plus the read-only save inventory into
a bounded queue of scenes, each with the exact ``prepare_capture_dataset.py`` and
``finalize_capture_dataset.py`` argv a developer would otherwise type by hand.

What this module deliberately does **not** do:

* it never launches Minecraft, installs the mod, arms a plan, stages a world, or
  runs either factory script -- it only writes queue bookkeeping;
* it never opens a source save for writing (the only save access is a read-only
  content fingerprint);
* it never accepts cameras, F3/debug values, coordinates, poses or look-at
  targets as inputs.  This is advanced developer *data generation*; region bounds
  are the only geometry it carries, and they are explicit developer declarations;
* it never assigns the final train/dev/test split.  The exact topology is not
  knowable before finalization, so the queue dedupes on a **source-content
  identity** and the authoritative split stays with the existing finalizer
  (``finalize_scene`` keys it on the exported block topology).  A spec may declare
  a ``declared_split`` label, which is used only to refuse the same source content
  being requested for two different splits.

Capture settings are either explicitly fixed for every view or left to the
factory's deterministic per-view strata.  Nested task settings are the base and
nested variant keys override them; top-level task and variant pack fields then
override nested pack values, with a present ``null`` disabling an inherited pack.
Generated commands state the selected mode and the manifest records the effective
request.

Identity and immutability
-------------------------

``source_identity`` is :func:`~mc_photogrammetry.dataset.hashing.staging_fingerprint`
over the selected save tree (a SHA-256 over file contents), computed *before*
variants are expanded.  ``build_id`` is a salted digest of that identity, so every
variant of one build shares one build id.

The queue manifest is content-addressed: ``manifest_id`` is the digest of the
immutable plan payload (paths, seed, per-scene region/variant/command records).
Statuses and attempt evidence live in the same JSON document but are stripped
before hashing, so:

* re-planning the same spec yields the same ``manifest_id``;
* materializing twice is byte-idempotent and never rewrites a queue;
* a manifest whose content no longer hashes to its ``manifest_id`` is refused as
  corrupted/tampered instead of being resumed;
* status changes go through validated transitions and are written with an atomic
  temp-file replace, so an interrupted run cannot leave a half-written manifest.

Only :func:`materialize_queue` writes anything, and it writes queue directories,
``manifest.json`` and the per-scene ``regions/*.json`` files the existing
``prepare`` command consumes -- never a staged world, a dataset scene, or a
source save.
"""
from __future__ import annotations

import json
import os
import string
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from mc_photogrammetry.dataset.console import write_line
from mc_photogrammetry.dataset.hashing import (
    StagingLimitError,
    digest_json,
    staging_fingerprint,
)
from mc_photogrammetry.dataset.inventory import WorldSave, inventory_saves, safe_component
from mc_photogrammetry.dataset.plan import (
    MAX_VIEWS,
    MIN_VIEWS,
    landscape_region,
    spawn_from_save,
    spawn_position_from_save,
)
from mc_photogrammetry.dataset.schemas import (
    CaptureSettings,
    DatasetSchemaError,
    RegionBounds,
    capture_settings_from_payload,
)

BATCH_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
QUEUE_DIR_NAME = "queue"
REGIONS_DIR_NAME = "regions"
SCENES_DIR_NAME = "scenes"
STATUSES = ("pending", "running", "complete", "failed")
PYTHON_PROGRAM = "python"
BUILD_SALT = "mc-photogrammetry/batch-build/v1"
USAGE_LABEL = (
    "developer/evaluation data generation only; never a normal reconstruction "
    "input; no Minecraft launch"
)
SPLIT_AUTHORITY_NOTE = (
    "declared_split is a queue grouping used only to refuse source leakage; the "
    "authoritative train/dev/test split is the topology-keyed split computed by "
    "finalize_scene after ground truth exists"
)
PACK_NOTE = (
    "resource_pack/shader_pack names are propagated as requested plan data only; "
    "no runtime pack switch is performed, neither pack is installed or activated, "
    "and neither pack is claimed active"
)

#: Manual camera/geometry metadata the queue must never carry.  The factory's
#: normal path may not consume these, and the batch spec is not the place to
#: smuggle them in; explicit region bounds are the only geometry accepted.
FORBIDDEN_SPEC_KEYS = frozenset({
    "camera", "cameras", "pose", "poses", "position", "positions", "coords",
    "coordinates", "f3", "f3_debug", "debug", "yaw", "pitch", "roll", "rotation",
    "target", "targets", "view_target", "camera_position",
})

_ALLOWED_SETTINGS_KEYS = frozenset({
    "gamma", "time_of_day", "weather", "hud_hidden", "resource_pack", "shader_pack",
})
_ALLOWED_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "pending": frozenset({"running"}),
    "running": frozenset({"complete", "failed"}),
    "failed": frozenset({"running"}),
    "complete": frozenset(),
}
_FRAME_ID_CHARS = frozenset(string.ascii_letters + string.digits + "-_")


@dataclass(frozen=True, slots=True)
class SceneRequest:
    """One queued scene: a build (source identity) x region x variant."""

    scene_id: str
    task_id: str
    variant_id: str
    save_name: str
    world_identity: str
    source_identity: str
    build_id: str
    declared_split: str
    region_name: str
    region: RegionBounds
    seed: int
    views: int
    settings: CaptureSettings
    settings_mode: str = "stratified"


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """The immutable, content-addressed plan a queue manifest materializes."""

    manifest_id: str
    batch_id: str
    game_dir: Path
    dataset: Path
    staging_game_dir: Path
    queue_dir: Path
    seed: int
    scenes: tuple[SceneRequest, ...]


# --- spec loading and validation -------------------------------------------


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetSchemaError(f"{label} must be a JSON object")
    return value


def _reject_forbidden(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            if name.strip().lower() in FORBIDDEN_SPEC_KEYS:
                raise DatasetSchemaError(
                    f"batch spec field {path}{name!r} is refused: manual camera/F3/"
                    "coordinate metadata is never a factory input; use explicit "
                    "region min/max or a spawn-derived landscape instead")
            _reject_forbidden(item, f"{path}{name}.")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden(item, f"{path}[{index}].")


def load_batch_spec(path: Path | str) -> dict[str, Any]:
    """Read and validate the declarative batch spec; never touches a save."""
    spec_path = Path(path)
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DatasetSchemaError(f"cannot read batch spec {spec_path}: {error}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise DatasetSchemaError(f"batch spec {spec_path} is not valid JSON: {error}") from error
    payload = dict(_as_mapping(payload, "batch spec"))
    if payload.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise DatasetSchemaError(
            f"unsupported batch schema_version {payload.get('schema_version')!r}")
    _reject_forbidden(payload, "")
    if not str(payload.get("batch_id", "")).strip():
        raise DatasetSchemaError("batch spec needs a non-empty batch_id")
    return payload


def _optional_path(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise DatasetSchemaError(f"{label} must be a path string")
    text = str(value).strip()
    if not text:
        raise DatasetSchemaError(f"{label} must not be blank")
    return Path(text)


def _resolve_save(saves: Iterable[WorldSave], name: str) -> WorldSave:
    wanted = str(name).strip()
    matches = [save for save in saves if save.name.lower() == wanted.lower()]
    if not matches:
        available = ", ".join(save.name for save in list(saves)[:8]) or "none"
        raise DatasetSchemaError(
            f"no save named {wanted!r} in the inventory; available: {available}")
    if len(matches) > 1:
        raise DatasetSchemaError(
            f"save name {wanted!r} is ambiguous: {[save.name for save in matches]}")
    return matches[0]


def _overlaps(first: Path, second: Path) -> bool:
    a, b = first.resolve(), second.resolve()
    return a == b or a in b.parents or b in a.parents


def _validate_roots(*, game_dir: Path, dataset: Path, staging_game_dir: Path,
                    queue_dir: Path, saves: Iterable[WorldSave]) -> None:
    """Refuse any output root that overlaps the game dir or a source save."""
    roots = {"dataset": dataset, "staging_game_dir": staging_game_dir, "queue_dir": queue_dir}
    game = game_dir.resolve()
    for label, root in roots.items():
        if _overlaps(root, game):
            raise DatasetSchemaError(
                f"{label} {root.resolve()} overlaps the source game directory {game}; "
                "staging output must live outside every game directory")
    for save in saves:
        save_path = save.path.resolve()
        for label, root in roots.items():
            if _overlaps(root, save_path):
                raise DatasetSchemaError(
                    f"{label} {root.resolve()} overlaps source save {save_path}; "
                    "source saves are never written by the factory")
    if dataset.resolve() == staging_game_dir.resolve():
        raise DatasetSchemaError("dataset and staging game directory must be different paths")
    scenes_root = dataset.resolve() / SCENES_DIR_NAME
    staging = staging_game_dir.resolve()
    if staging == scenes_root or scenes_root in staging.parents:
        raise DatasetSchemaError("staging game directory must not sit inside <dataset>/scenes")


# --- identity and scene expansion ------------------------------------------


def build_identity_for(source_identity: str) -> str:
    """Salted, stable build id for a source-content fingerprint."""
    if not source_identity:
        raise DatasetSchemaError("a build identity needs a source-content identity")
    return digest_json({"salt": BUILD_SALT, "source_identity": str(source_identity)})


def scene_id_for(save_name: str, region_name: str) -> str:
    """The exact scene id ``prepare_capture_dataset.py`` derives from its inputs.

    Kept byte-identical to the factory script's ``_scene_id`` (lowercase and
    replace every non-alphanumeric character with ``-``) so a queued finalize
    command names the scene the prepare command created.  The result is then
    checked against the plan frame-id alphabet: the existing plan contract only
    allows ASCII alphanumerics, ``-`` and ``_`` in a frame id, and a scene id that
    cannot form one is refused here rather than queued into a guaranteed failure.
    """
    raw = f"{save_name}-{region_name}".lower()
    cleaned = "".join(character if character.isalnum() else "-" for character in raw)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise DatasetSchemaError("scene id became empty; rename the save or region")
    unsafe = sorted(set(cleaned) - _FRAME_ID_CHARS)
    if unsafe:
        raise DatasetSchemaError(
            f"scene id {cleaned!r} (from save {save_name!r}, region {region_name!r}) "
            f"contains characters the capture frame-id contract forbids: {unsafe}; "
            "rename the save or region to ASCII alphanumerics, '-' or '_'")
    return cleaned


def _pack(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetSchemaError(f"{label} must be a string or null")
    if not value.strip() or any(character in value for character in ("\r", "\n", "\0")):
        raise DatasetSchemaError(f"{label} must be a non-empty single-line name")
    return value


def _settings_for_variant(task: Mapping[str, Any], variant: Mapping[str, Any], *,
                          label: str) -> tuple[CaptureSettings, str]:
    """Resolve fixed settings with explicit variant and pack precedence.

    Nested settings merge key-by-key: task settings are the base and variant
    settings override them.  A present ``settings: null`` clears the inherited
    nested object to defaults.  Top-level pack fields then override nested pack
    values in this order: task field, variant field.  Presence is significant:
    an explicit null pack disables an inherited pack.
    """
    task_has_settings = "settings" in task
    variant_has_settings = "settings" in variant
    task_raw = task.get("settings")
    variant_raw = variant.get("settings")
    task_nested = (dict(_as_mapping(task_raw, f"{label} task settings"))
                   if task_raw is not None else {})
    if task_raw is not None:
        capture_settings_from_payload(task_nested, label=f"{label} task settings")
    if variant_has_settings and variant_raw is None:
        nested: dict[str, Any] = {}
    else:
        variant_nested = (dict(_as_mapping(variant_raw, f"{label} variant settings"))
                          if variant_raw is not None else {})
        if variant_raw is not None:
            capture_settings_from_payload(variant_nested,
                                          label=f"{label} variant settings")
        nested = {**task_nested, **variant_nested}

    # A top-level field wins even when its value is null.  Otherwise the
    # key-by-key nested merge supplies the effective value.
    effective: dict[str, Any] = dict(nested)
    for key in ("resource_pack", "shader_pack"):
        if key in task:
            effective[key] = _pack(task[key], f"{label} task {key}")
        if key in variant:
            effective[key] = _pack(variant[key], f"{label} variant {key}")

    unknown = sorted(set(effective) - _ALLOWED_SETTINGS_KEYS)
    if unknown:
        raise DatasetSchemaError(f"{label} settings has unsupported keys {unknown}")
    configured = (task_has_settings or variant_has_settings or
                  any(key in task or key in variant
                      for key in ("resource_pack", "shader_pack")))
    settings = capture_settings_from_payload(effective, label=f"{label} settings")
    return settings, "fixed" if configured else "stratified"


def _region_from_spec(raw: Any, label: str) -> tuple[str, RegionBounds]:
    raw = _as_mapping(raw, label)
    name = str(raw.get("name", "")).strip()
    if not name:
        raise DatasetSchemaError(f"{label} needs a name")
    safe_component(name)
    bounds = RegionBounds.from_dict({"min": raw.get("min"), "max": raw.get("max")})
    return name, bounds


def _landscape_region_from_spec(save: WorldSave, raw: Any,
                                label: str) -> tuple[str, RegionBounds]:
    raw = _as_mapping(raw, label)
    radius = int(raw.get("radius", 32))
    vertical_below = int(raw.get("vertical_below", 8))
    vertical_above = int(raw.get("vertical_above", 24))
    level_dat = save.path / "level.dat"
    surface_y = raw.get("surface_y")
    if surface_y is None:
        spawn_x, spawn_y, spawn_z = spawn_position_from_save(level_dat)
    else:
        spawn_x, spawn_z = spawn_from_save(level_dat)
        spawn_y = int(surface_y)
    bounds = landscape_region(spawn_x=spawn_x, spawn_z=spawn_z, surface_y=spawn_y,
                              radius=radius, vertical_below=vertical_below,
                              vertical_above=vertical_above)
    default_name = f"spawn-{spawn_x}-{spawn_z}"
    return str(raw.get("name", default_name)).strip() or default_name, bounds


def _variant_specs(task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = task.get("variants")
    if raw is None:
        raw = [{}]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise DatasetSchemaError("task variants must be a non-empty list")
    variants = []
    for entry in raw:
        variants.append(_as_mapping(entry, "task variant"))
    return variants


def _expand_task(task: Mapping[str, Any], *, save: WorldSave, source_identity: str,
                 default_seed: int, default_views: int) -> list[SceneRequest]:
    task = _as_mapping(task, "task")
    task_id = str(task.get("task_id", "")).strip()
    if not task_id:
        raise DatasetSchemaError("task needs a task_id")
    safe_component(task_id)
    declared_split = str(task.get("declared_split", "")).strip()
    if declared_split not in ("train", "dev", "test"):
        raise DatasetSchemaError(
            f"task {task_id!r} needs declared_split one of train/dev/test")
    task_seed = int(task.get("seed", default_seed))
    task_views = int(task.get("views", default_views))
    variants = _variant_specs(task)
    build_id = build_identity_for(source_identity)
    scenes: list[SceneRequest] = []
    seen_variant_ids: set[str] = set()
    for index, variant in enumerate(variants):
        variant_id = str(variant.get("variant_id") or
                         (f"v{index + 1}" if len(variants) > 1 else "v1")).strip()
        safe_component(variant_id)
        if variant_id in seen_variant_ids:
            raise DatasetSchemaError(
                f"task {task_id!r} repeats variant_id {variant_id!r}; variant ids "
                "must be unique within a task")
        seen_variant_ids.add(variant_id)
        seed = int(variant.get("seed", task_seed))
        views = int(variant.get("views", task_views))
        if not MIN_VIEWS <= views <= MAX_VIEWS:
            raise DatasetSchemaError(
                f"task {task_id!r} variant {variant_id!r}: views must be "
                f"{MIN_VIEWS}-{MAX_VIEWS}")
        region_spec = variant.get("region", task.get("region"))
        landscape_spec = variant.get("landscape", task.get("landscape"))
        if (region_spec is None) == (landscape_spec is None):
            raise DatasetSchemaError(
                f"task {task_id!r} variant {variant_id!r} needs exactly one of "
                "region or landscape")
        if region_spec is not None:
            region_name, bounds = _region_from_spec(
                region_spec, f"task {task_id!r} region")
        else:
            region_name, bounds = _landscape_region_from_spec(
                save, landscape_spec, f"task {task_id!r} landscape")
        if len(variants) > 1:
            # Every variant of a multi-variant task gets its own region name, and
            # therefore its own scene id, region file, staging directory and
            # commands, for explicit *and* landscape geometry. The bounds and the
            # source/build identity are identical across the variants, so they
            # still group as one build for split-leakage refusal.
            region_name = f"{region_name}-{variant_id}"
            safe_component(region_name)
        settings, settings_mode = _settings_for_variant(
            task, variant, label=f"task {task_id!r} variant {variant_id!r}")
        scenes.append(SceneRequest(
            scene_id=scene_id_for(save.name, region_name),
            task_id=task_id,
            variant_id=variant_id,
            save_name=save.name,
            world_identity=save.world_identity,
            source_identity=source_identity,
            build_id=build_id,
            declared_split=declared_split,
            region_name=region_name,
            region=bounds,
            seed=seed,
            views=views,
            settings=settings,
            settings_mode=settings_mode,
        ))
    return scenes


# --- duplicate / leakage validation ----------------------------------------


def _scene_key(scene: SceneRequest) -> str:
    return digest_json({
        "source_identity": scene.source_identity,
        "region": [list(scene.region.minimum), list(scene.region.maximum)],
        "variant": {
            "seed": scene.seed,
            "views": scene.views,
            "settings_mode": scene.settings_mode,
            "settings": (scene.settings.to_dict()
                         if scene.settings_mode == "fixed" else None),
        },
    })


def _read_existing_index(dataset: Path) -> dict[str, Any] | None:
    index_path = dataset / "index.json"
    if not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetSchemaError(
            f"existing dataset index {index_path} is unreadable: {error}") from error
    if not isinstance(payload, Mapping):
        raise DatasetSchemaError(f"existing dataset index {index_path} is not an object")
    return dict(payload)


def _validate_unique_and_leakage(scenes: Iterable[SceneRequest], dataset: Path) -> None:
    scene_list = list(scenes)
    scene_ids: dict[str, str] = {}
    scene_keys: dict[str, str] = {}
    build_splits: dict[str, set[str]] = {}
    for scene in scene_list:
        previous = scene_ids.get(scene.scene_id)
        if previous is not None:
            raise DatasetSchemaError(
                f"duplicate scene_id {scene.scene_id!r} (tasks {previous!r} and "
                f"{scene.task_id!r})")
        scene_ids[scene.scene_id] = scene.task_id
        key = _scene_key(scene)
        if key in scene_keys:
            raise DatasetSchemaError(
                f"scene {scene.scene_id!r} duplicates the same source/region/variant "
                f"work already requested by {scene_keys[key]!r}")
        scene_keys[key] = scene.scene_id
        build_splits.setdefault(scene.build_id, set()).add(scene.declared_split)
    for build_id, splits in sorted(build_splits.items()):
        if len(splits) > 1:
            raise DatasetSchemaError(
                f"source identity {build_id!r} (identical save content) is requested "
                f"under splits {sorted(splits)}; one build must not straddle "
                "train/dev/test")

    payload = _read_existing_index(dataset)
    if payload is None:
        return
    existing_scene_ids: set[str] = set()
    existing_splits: dict[str, str] = {}
    for entry in payload.get("scenes", []) or []:
        if not isinstance(entry, Mapping):
            continue
        existing_scene_ids.add(str(entry.get("scene_id", "")))
        token = str((entry.get("provenance") or {}).get("staging_token", ""))
        if token:
            existing_splits[build_identity_for(token)] = str(entry.get("split", ""))
    for scene_id in sorted(scene_ids):
        if scene_id in existing_scene_ids:
            raise DatasetSchemaError(
                f"scene {scene_id!r} already exists in {dataset / 'index.json'}; "
                "finalize refuses to clobber a completed scene")
    for scene in scene_list:
        previous_split = existing_splits.get(scene.build_id)
        if previous_split and previous_split != scene.declared_split:
            raise DatasetSchemaError(
                f"source identity {scene.source_identity!r} is already finalized under "
                f"split {previous_split!r} but this batch declares "
                f"{scene.declared_split!r}; identical source content must stay in one split")


# --- planning --------------------------------------------------------------


def plan_batch(spec: Mapping[str, Any], *, game_dir: Path | str | None = None,
               dataset: Path | str | None = None,
               staging_game_dir: Path | str | None = None,
               queue_dir: Path | str | None = None) -> BatchPlan:
    """Plan the whole batch deterministically without writing or launching anything."""
    spec = dict(_as_mapping(spec, "batch spec"))
    if spec.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise DatasetSchemaError(
            f"unsupported batch schema_version {spec.get('schema_version')!r}")
    _reject_forbidden(spec, "")
    batch_id = str(spec.get("batch_id", "")).strip()
    safe_component(batch_id)

    resolved_game = Path(game_dir) if game_dir is not None else _optional_path(
        spec.get("game_dir"), "game_dir")
    if resolved_game is None:
        raise DatasetSchemaError("batch spec or CLI must provide game_dir")
    resolved_dataset = Path(dataset) if dataset is not None else _optional_path(
        spec.get("dataset"), "dataset")
    if resolved_dataset is None:
        raise DatasetSchemaError("batch spec or CLI must provide dataset")
    resolved_staging = (Path(staging_game_dir) if staging_game_dir is not None
                        else _optional_path(spec.get("staging_game_dir"), "staging_game_dir")
                        or resolved_dataset / "staging-game")
    resolved_queue = (Path(queue_dir) if queue_dir is not None
                      else _optional_path(spec.get("queue_dir"), "queue_dir")
                      or resolved_dataset / QUEUE_DIR_NAME)

    raw_tasks = spec.get("tasks")
    if not isinstance(raw_tasks, (list, tuple)) or not raw_tasks:
        raise DatasetSchemaError("batch spec must list at least one task")
    default_seed = int(spec.get("seed", 0))
    default_views = int(spec.get("views", 6))

    saves = inventory_saves(resolved_game)
    selected: dict[str, WorldSave] = {}
    for raw_task in raw_tasks:
        task = _as_mapping(raw_task, "task")
        save_name = str(task.get("save", "")).strip()
        if not save_name:
            raise DatasetSchemaError("every task needs a save name")
        save = _resolve_save(saves, save_name)
        selected.setdefault(save.name, save)

    _validate_roots(game_dir=resolved_game, dataset=resolved_dataset,
                    staging_game_dir=resolved_staging, queue_dir=resolved_queue,
                    saves=selected.values())

    identities: dict[str, str] = {}
    for name, save in selected.items():
        try:
            identities[name] = staging_fingerprint(save.path)
        except StagingLimitError as error:
            raise DatasetSchemaError(
                f"cannot fingerprint source save {save.path}: {error}") from error

    scenes: list[SceneRequest] = []
    for raw_task in raw_tasks:
        task = _as_mapping(raw_task, "task")
        save_name = str(task.get("save", "")).strip()
        save = _resolve_save(saves, save_name)
        scenes.extend(_expand_task(task, save=save, source_identity=identities[save.name],
                                   default_seed=default_seed, default_views=default_views))

    _validate_unique_and_leakage(scenes, resolved_dataset)

    plan = BatchPlan(
        manifest_id="",
        batch_id=batch_id,
        game_dir=resolved_game.resolve(),
        dataset=resolved_dataset.resolve(),
        staging_game_dir=resolved_staging.resolve(),
        queue_dir=resolved_queue.resolve(),
        seed=default_seed,
        scenes=tuple(scenes),
    )
    return replace(plan, manifest_id=manifest_id_for(plan))


# --- manifest payloads -----------------------------------------------------


def region_payload(scene: SceneRequest) -> dict[str, Any]:
    """The ``--regions`` file content that reproduces this scene's bounds exactly."""
    return {
        "schema_version": 1,
        "regions": [{
            "name": scene.region_name,
            "views": int(scene.views),
            "min": [int(value) for value in scene.region.minimum],
            "max": [int(value) for value in scene.region.maximum],
        }],
    }


def scene_staging_dir(plan: BatchPlan, scene: SceneRequest) -> Path:
    return plan.staging_game_dir / scene.scene_id


def region_file_path(plan: BatchPlan, scene: SceneRequest) -> Path:
    return plan.queue_dir / REGIONS_DIR_NAME / f"{scene.scene_id}.json"


def scene_payload(plan: BatchPlan, scene: SceneRequest, *, status: str = "pending",
                  attempts: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    staging = scene_staging_dir(plan, scene)
    region_file = region_file_path(plan, scene)
    prepare_argv = [
        PYTHON_PROGRAM, "scripts/prepare_capture_dataset.py",
        "--game-dir", str(plan.game_dir),
        "--save", scene.save_name,
        "--dataset", str(plan.dataset),
        "--dataset-id", plan.batch_id,
        "--staging-game-dir", str(staging),
        "--regions", str(region_file),
        "--seed", str(scene.seed),
        "--views", str(scene.views),
        "--allow-overwrite",
    ]
    if scene.settings_mode == "fixed":
        fixed_settings = scene.settings.to_dict()
        prepare_argv.extend(("--fixed-settings-json", json.dumps(
            fixed_settings, ensure_ascii=False, separators=(",", ":"))))
        settings_effective_request: dict[str, Any] = fixed_settings
        settings_requested: dict[str, Any] | None = fixed_settings
    elif scene.settings_mode == "stratified":
        prepare_argv.append("--stratified-settings")
        settings_effective_request = {
            "strategy": "view_setting_strata",
            "view_count": int(scene.views),
        }
        settings_requested = None
    else:
        raise DatasetSchemaError(
            f"scene {scene.scene_id!r} has unknown settings mode {scene.settings_mode!r}")
    finalize_argv = [
        PYTHON_PROGRAM, "scripts/finalize_capture_dataset.py",
        "--dataset", str(plan.dataset),
        "--staging-game-dir", str(staging),
        "--scene", scene.scene_id,
        "--allow-overwrite",
    ]
    return {
        "scene_id": scene.scene_id,
        "task_id": scene.task_id,
        "variant_id": scene.variant_id,
        "save": scene.save_name,
        "world_identity": scene.world_identity,
        "source_identity": scene.source_identity,
        "build_id": scene.build_id,
        "declared_split": scene.declared_split,
        "region": {**scene.region.to_dict(), "name": scene.region_name},
        "seed": int(scene.seed),
        "views": int(scene.views),
        "settings_mode": scene.settings_mode,
        "settings_requested": settings_requested,
        "settings_effective_request": settings_effective_request,
        "resource_pack_requested": scene.settings.resource_pack,
        "shader_pack_requested": scene.settings.shader_pack,
        "resource_pack_applied": False,
        "shader_pack_applied": False,
        "pack_note": PACK_NOTE,
        "staging_game_dir": str(staging),
        "region_file": str(region_file),
        "prepare_argv": prepare_argv,
        "finalize_argv": finalize_argv,
        "status": status,
        "attempts": [dict(attempt) for attempt in attempts],
    }


def manifest_payload(plan: BatchPlan) -> dict[str, Any]:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "manifest_id": plan.manifest_id,
        "batch_id": plan.batch_id,
        "advanced_only": True,
        "usage": USAGE_LABEL,
        "split_authority": SPLIT_AUTHORITY_NOTE,
        "game_dir": str(plan.game_dir),
        "dataset": str(plan.dataset),
        "staging_game_dir": str(plan.staging_game_dir),
        "queue_dir": str(plan.queue_dir),
        "seed": plan.seed,
        "scenes": [scene_payload(plan, scene) for scene in plan.scenes],
    }


def _immutable_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in payload.items() if key != "manifest_id"}
    result["scenes"] = [
        {key: value for key, value in scene.items() if key not in ("status", "attempts")}
        for scene in result.get("scenes", [])
    ]
    return result


def manifest_id_for(plan: BatchPlan) -> str:
    """Digest of the immutable plan: paths, seed, scenes, and their commands."""
    return digest_json(_immutable_payload(manifest_payload(plan)))


def manifest_path(queue_dir: Path | str) -> Path:
    return Path(queue_dir) / MANIFEST_NAME


# --- atomic IO -------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_regions(plan: BatchPlan) -> None:
    for scene in plan.scenes:
        path = region_file_path(plan, scene)
        text = json.dumps(region_payload(scene), indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise DatasetSchemaError(
                    f"refusing to rewrite immutable region file {path}")
            continue
        _atomic_write_text(path, text)


def materialize_queue(plan: BatchPlan) -> Path:
    """Write the queue manifest and per-scene region files; byte-idempotent.

    Nothing else is written: no world is staged, no plan is armed, and no dataset
    scene is produced.  An existing manifest with a different ``manifest_id`` is
    refused rather than replaced, so a queue is immutable once materialized.
    """
    path = manifest_path(plan.queue_dir)
    if path.exists():
        existing = load_queue(plan.queue_dir)
        if existing.get("manifest_id") != plan.manifest_id:
            raise DatasetSchemaError(
                f"queue {plan.queue_dir} already holds manifest "
                f"{existing.get('manifest_id')!r}, not {plan.manifest_id!r}; refusing "
                "to replace an immutable queue")
        _write_regions(plan)
        return path
    _write_regions(plan)
    _write_json_atomic(path, manifest_payload(plan))
    return path


# --- loading, status transitions, resume -----------------------------------


def _validate_attempt(attempt: Any, label: str) -> None:
    attempt = _as_mapping(attempt, label)
    ordinal = attempt.get("attempt")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise DatasetSchemaError(f"{label} needs a positive integer attempt ordinal")
    if attempt.get("status") not in STATUSES:
        raise DatasetSchemaError(f"{label} has unknown status {attempt.get('status')!r}")
    if not isinstance(attempt.get("evidence"), Mapping):
        raise DatasetSchemaError(f"{label} evidence must be an object")


def _validate_attempt_history(scene_id: str, status: str, attempts: list[Any]) -> None:
    """Fully validate a scene's mutable attempt history.

    ``status`` and ``attempts`` are stripped from the content-addressed manifest
    digest, so they are the only mutable queue state and the digest cannot protect
    them.  This check is consequently the authority: ordinals must be exactly
    ``1..N``, the statuses must walk the legal transition ladder from ``pending``,
    and the recorded scene status must agree with the final attempt (with the sole
    exception of a pristine ``pending`` scene that has never been attempted).
    """
    if not attempts:
        if status != "pending":
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} has status {status!r} but no attempts; "
                "only a never-started pending scene may have an empty history")
        return
    current = "pending"
    for index, attempt in enumerate(attempts, start=1):
        ordinal = attempt["attempt"]
        if ordinal != index:
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} attempt {index} records ordinal "
                f"{ordinal!r}; attempt ordinals must be exactly 1..N")
        next_status = str(attempt["status"])
        if next_status not in _ALLOWED_TRANSITIONS[current]:
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} attempt {index} records illegal transition "
                f"{current!r} -> {next_status!r}")
        current = next_status
    if status != current:
        raise DatasetSchemaError(
            f"queue scene {scene_id!r} status {status!r} disagrees with the last "
            f"attempt status {current!r}")


def _validate_scene_settings_contract(scene: Mapping[str, Any]) -> None:
    """Check that a new manifest's declared mode matches its prepare argv.

    Legacy schema-v1 manifests have no ``settings_mode`` and remain loadable.
    New manifests must not claim one request in provenance while asking the
    prepare command to build a different plan.
    """
    if "settings_mode" not in scene:
        return
    scene_id = str(scene.get("scene_id", "<unknown>"))
    argv = scene.get("prepare_argv")
    if not isinstance(argv, (list, tuple)) or not all(isinstance(item, str) for item in argv):
        raise DatasetSchemaError(f"queue scene {scene_id!r} prepare_argv must be a list of strings")
    mode = scene.get("settings_mode")
    fixed_flags = [index for index, value in enumerate(argv)
                   if value == "--fixed-settings-json"]
    stratified_count = argv.count("--stratified-settings")
    if mode == "fixed":
        if len(fixed_flags) != 1 or stratified_count or fixed_flags[0] + 1 >= len(argv):
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} fixed mode must have exactly one complete "
                "--fixed-settings-json argument")
        try:
            raw = json.loads(argv[fixed_flags[0] + 1])
        except json.JSONDecodeError as error:
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} fixed settings argv is invalid JSON: {error}") from error
        requested = capture_settings_from_payload(
            raw, require_complete=True, label=f"queue scene {scene_id!r} argv settings")
        expected = requested.to_dict()
        if scene.get("settings_requested") != expected or scene.get(
                "settings_effective_request") != expected:
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} settings manifest and prepare argv disagree")
        if (scene.get("resource_pack_requested") != expected["resource_pack"]
                or scene.get("shader_pack_requested") != expected["shader_pack"]):
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} requested pack fields disagree with settings argv")
    elif mode == "stratified":
        if fixed_flags or stratified_count != 1:
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} stratified mode must have exactly one "
                "--stratified-settings argument")
        effective = scene.get("settings_effective_request")
        if (scene.get("settings_requested") is not None
                or not isinstance(effective, Mapping)
                or effective.get("strategy") != "view_setting_strata"
                or effective.get("view_count") != scene.get("views")):
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} stratified settings manifest is inconsistent")
        if scene.get("resource_pack_requested") is not None or scene.get(
                "shader_pack_requested") is not None:
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} stratified mode cannot request fixed packs")
    else:
        raise DatasetSchemaError(
            f"queue scene {scene_id!r} has unknown settings_mode {mode!r}")


def load_queue(queue_dir: Path | str) -> dict[str, Any]:
    """Load and fully validate a queue manifest, refusing any corrupted state."""
    path = manifest_path(queue_dir)
    if not path.is_file():
        raise DatasetSchemaError(f"no queue manifest at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetSchemaError(f"corrupted queue manifest {path}: {error}") from error
    payload = dict(_as_mapping(payload, "queue manifest"))
    if payload.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise DatasetSchemaError(
            f"queue manifest {path} has unsupported schema_version "
            f"{payload.get('schema_version')!r}")
    manifest_id = payload.get("manifest_id")
    if not isinstance(manifest_id, str) or len(manifest_id) != 64:
        raise DatasetSchemaError(f"queue manifest {path} has a malformed manifest_id")
    scenes = payload.get("scenes")
    if not isinstance(scenes, (list, tuple)) or not scenes:
        raise DatasetSchemaError(f"queue manifest {path} has no scenes")
    seen: set[str] = set()
    for scene in scenes:
        scene = _as_mapping(scene, "queue scene")
        scene_id = str(scene.get("scene_id", ""))
        if not scene_id:
            raise DatasetSchemaError(f"queue manifest {path} has a scene without scene_id")
        if scene_id in seen:
            raise DatasetSchemaError(f"queue manifest {path} repeats scene {scene_id!r}")
        seen.add(scene_id)
        if scene.get("status") not in STATUSES:
            raise DatasetSchemaError(
                f"queue scene {scene_id!r} has unknown status {scene.get('status')!r}")
        attempts = scene.get("attempts", [])
        if not isinstance(attempts, (list, tuple)):
            raise DatasetSchemaError(f"queue scene {scene_id!r} attempts must be a list")
        for index, attempt in enumerate(attempts):
            _validate_attempt(attempt, f"scene {scene_id!r} attempt {index + 1}")
        _validate_attempt_history(scene_id, str(scene.get("status")), list(attempts))
        for key in ("prepare_argv", "finalize_argv"):
            argv = scene.get(key)
            if not isinstance(argv, (list, tuple)) or not all(
                    isinstance(item, str) for item in argv):
                raise DatasetSchemaError(
                    f"queue scene {scene_id!r} {key} must be a list of strings")
        _validate_scene_settings_contract(scene)
    if digest_json(_immutable_payload(payload)) != manifest_id:
        raise DatasetSchemaError(
            f"queue manifest {path} content does not hash to its manifest_id "
            f"{manifest_id!r}; refusing to resume a tampered queue")
    return payload


def _find_scene(payload: Mapping[str, Any], scene_id: str) -> dict[str, Any]:
    for scene in payload.get("scenes", []):
        if scene.get("scene_id") == scene_id:
            return scene  # type: ignore[return-value]
    raise DatasetSchemaError(f"queue has no scene {scene_id!r}")


def set_status(queue_dir: Path | str, scene_id: str, status: str, *,
               evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Record a validated status transition and its attempt evidence, atomically."""
    if status not in STATUSES:
        raise DatasetSchemaError(
            f"unknown status {status!r}; expected one of {list(STATUSES)}")
    path = manifest_path(queue_dir)
    payload = load_queue(queue_dir)
    scene = _find_scene(payload, scene_id)
    current = str(scene["status"])
    if status == current:
        raise DatasetSchemaError(f"scene {scene_id!r} is already {status!r}")
    if status not in _ALLOWED_TRANSITIONS[current]:
        raise DatasetSchemaError(
            f"illegal status transition for {scene_id!r}: {current!r} -> {status!r}")
    record: dict[str, Any] = dict(evidence) if evidence is not None else {}
    if not isinstance(record, Mapping):
        raise DatasetSchemaError("attempt evidence must be a JSON object")
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DatasetSchemaError(f"attempt evidence is not JSON-serializable: {error}") from error
    scene["attempts"].append({
        "attempt": len(scene["attempts"]) + 1,
        "status": status,
        "evidence": record,
    })
    scene["status"] = status
    _write_json_atomic(path, payload)
    return payload


def resume_queue(queue_dir: Path | str) -> dict[str, Any]:
    """Reset stale ``running`` scenes and report what a resumed run should do.

    A manifest can only be left ``running`` by an interrupted process, so resume
    records that as a failed attempt (with the reason) and reports the scene as
    actionable.  ``complete`` scenes are never touched, and a second resume is a
    no-op, so resuming is safe to repeat.
    """
    payload = load_queue(queue_dir)
    reset: list[str] = []
    for scene in payload["scenes"]:
        if scene["status"] != "running":
            continue
        scene["attempts"].append({
            "attempt": len(scene["attempts"]) + 1,
            "status": "failed",
            "evidence": {"reason": "stale running entry reset on resume"},
        })
        scene["status"] = "failed"
        reset.append(scene["scene_id"])
    if reset:
        _write_json_atomic(manifest_path(queue_dir), payload)
    actionable = [scene["scene_id"] for scene in payload["scenes"]
                  if scene["status"] in ("pending", "failed")]
    complete = [scene["scene_id"] for scene in payload["scenes"]
                if scene["status"] == "complete"]
    return {"manifest_id": payload["manifest_id"], "reset": reset,
            "actionable": actionable, "complete": complete}


def queue_commands(payload: Mapping[str, Any], *, statuses: Iterable[str] = ("pending", "failed"),
                   scene_id: str | None = None) -> list[dict[str, Any]]:
    """The prepare/finalize argv rows a runner should execute, in manifest order."""
    wanted = set(statuses)
    rows = []
    for scene in payload.get("scenes", []):
        if scene_id is not None and scene.get("scene_id") != scene_id:
            continue
        if scene.get("status") not in wanted:
            continue
        rows.append({
            "scene_id": scene["scene_id"],
            "status": scene["status"],
            "prepare_argv": list(scene["prepare_argv"]),
            "finalize_argv": list(scene["finalize_argv"]),
        })
    return rows


def render_queue_lines(payload: Mapping[str, Any]) -> list[str]:
    """Deterministic, encoding-safe summary lines for a manifest or plan payload."""
    lines = [
        f"batch {payload.get('batch_id')} manifest {payload.get('manifest_id')}",
        f"dataset {payload.get('dataset')}",
        f"staging {payload.get('staging_game_dir')}",
        f"scenes {len(payload.get('scenes', []))}",
    ]
    for scene in payload.get("scenes", []):
        region = scene.get("region") or {}
        lines.append(
            f"{scene.get('scene_id')}  [{scene.get('status')}]  save={scene.get('save')!r} "
            f"world={scene.get('world_identity')!r} region={region.get('name')!r} "
            f"split={scene.get('declared_split')} attempts={len(scene.get('attempts', []))}")
    return lines


def write_queue_lines(payload: Mapping[str, Any], stream: Any = None) -> None:
    """Write :func:`render_queue_lines` through the encoding-safe console writer."""
    target = sys.stdout if stream is None else stream
    for line in render_queue_lines(payload):
        write_line(target, line)
