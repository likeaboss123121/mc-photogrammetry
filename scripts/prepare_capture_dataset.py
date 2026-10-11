"""Prepare a real-Minecraft capture dataset without ever launching the game.

This inventories the saves in a game directory read-only, copies the chosen
world into a staging game directory, and freezes a versioned capture plan plus a
scene recipe. It does not install the mod, does not arm anything unless asked
with ``--arm``, and never writes to the source save.

Examples
--------
List the saves the factory can see::

    python scripts/prepare_capture_dataset.py --list

Plan one explicit developer region::

    python scripts/prepare_capture_dataset.py --game-dir "%APPDATA%\\.minecraft" \\
        --save "Template 2" --dataset exports/datasets/ds17 --dataset-id ds17 \\
        --seed 7 --regions docs/reports/ds17-regions.json

Plan a landscape sample around spawn (the height defaults to the save's ``SpawnY``)::

    python scripts/prepare_capture_dataset.py --save "Template 6" \\
        --dataset exports/datasets/ds17 --dataset-id ds17 --seed 7 \\
        --landscape --views 6

Override the sample height explicitly (developer-only)::

    python scripts/prepare_capture_dataset.py --save "Template 6" \\
        --dataset exports/datasets/ds17 --dataset-id ds17 --seed 7 \\
        --landscape --surface-y 70 --views 6
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.dataset.console import write_line, write_safe  # noqa: E402
from mc_photogrammetry.dataset.finalize import (  # noqa: E402
    write_json,
    write_scene_plan,
    write_scene_recipe,
)
from mc_photogrammetry.dataset.hashing import sha256_file  # noqa: E402
from mc_photogrammetry.dataset.inventory import (  # noqa: E402
    default_game_dir,
    inventory_saves,
    safe_join,
    stage_world,
    staging_marker_payload,
    world_marker_path,
)
from mc_photogrammetry.dataset.options import (  # noqa: E402
    CAPTURE_OPTIONS,
    stage_capture_options,
)
from mc_photogrammetry.dataset.plan import (  # noqa: E402
    RegionRequest,
    build_plan,
    landscape_region,
    load_region_requests,
    spawn_from_save,
    spawn_position_from_save,
)
from mc_photogrammetry.dataset.schemas import (  # noqa: E402
    MOD_CONTRACT_VERSION,
    DatasetSchemaError,
    RegionBounds,
    SceneRecipe,
    capture_settings_from_payload,
)

MOD_DIR = "mc-photo-capture"


def _find_save(saves, name: str):
    lowered = name.lower()
    for save in saves:
        if save.name.lower() == lowered:
            return save
    raise SystemExit(f"no save named {name!r}; have {[save.name for save in saves]}")


def _scene_id(save_name: str, region_name: str) -> str:
    raw = f"{save_name}-{region_name}".lower()
    cleaned = "".join(character if character.isalnum() else "-" for character in raw)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise SystemExit("scene id became empty; rename the save or region")
    return cleaned


def _say(*values, sep: str = " ", end: str = "\n") -> None:
    """Print to ``sys.stdout`` even when the console cannot encode the text.

    ``print`` treats a ``UnicodeEncodeError`` as fatal, which would truncate a
    listing at the first save with a name the active code page lacks. The
    encoding-safe writer escapes only those characters and keeps every record.
    """
    write_safe(sys.stdout, sep.join(str(value) for value in values) + end)


def _print_saves(saves, stream=None) -> int:
    target = sys.stdout if stream is None else stream
    if not saves:
        write_line(target, "no saves found")
        return 0
    write_line(target, f"{'name':40} {'size MiB':>9}  {'modified':20}  level name")
    for save in saves:
        write_line(target, f"{save.name[:40]:40} {save.size_bytes / (1 << 20):9.1f}  "
                           f"{save.modified_utc:20}  {save.level_name or '-'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game-dir", type=Path, default=None,
                        help="Minecraft game directory to inventory read-only")
    parser.add_argument("--list", action="store_true", help="list saves and exit")
    parser.add_argument("--save", help="save directory name to capture")
    parser.add_argument("--dataset", type=Path, help="dataset root to create")
    parser.add_argument("--dataset-id", help="dataset identifier recorded in the index")
    parser.add_argument("--staging-game-dir", type=Path, default=None,
                        help="game directory that receives the staged world copy")
    parser.add_argument("--regions", type=Path, default=None,
                        help="JSON file of explicit developer evaluation regions")
    parser.add_argument("--landscape", action="store_true",
                        help="plan a landscape sample around the save spawn point")
    parser.add_argument("--surface-y", type=int, default=None,
                        help="developer override for the landscape centre height "
                             "(default: the save's SpawnY)")
    parser.add_argument("--radius", type=int, default=32,
                        help="landscape sample radius in blocks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", type=int, default=6, help="views per region (4-8)")
    settings_mode = parser.add_mutually_exclusive_group()
    settings_mode.add_argument(
        "--fixed-settings-json", metavar="JSON",
        help="one fixed CaptureSettings object for every planned view; requires all "
             "core keys gamma, time_of_day, weather, hud_hidden")
    settings_mode.add_argument(
        "--stratified-settings", action="store_true",
        help="assign the factory's deterministic per-view setting strata")
    parser.add_argument("--arm", action="store_true",
                        help="write armed.json into the staging game dir (never a live install)")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args(argv)

    # Validate this request before inventory, staging, or any file creation.  A
    # batch command always chooses one mode explicitly; legacy direct callers
    # with neither option retain the existing stratified behavior.
    fixed_settings = None
    if args.fixed_settings_json is not None:
        try:
            fixed_payload = json.loads(args.fixed_settings_json)
        except json.JSONDecodeError as error:
            raise SystemExit(f"--fixed-settings-json is not valid JSON: {error}") from error
        try:
            fixed_settings = capture_settings_from_payload(
                fixed_payload, require_complete=True, label="--fixed-settings-json")
        except DatasetSchemaError as error:
            raise SystemExit(str(error)) from error

    game_dir = args.game_dir or default_game_dir()
    saves = inventory_saves(game_dir)
    if args.list:
        return _print_saves(saves)
    if not args.save:
        raise SystemExit("pass --save <name> (or --list)")
    save = _find_save(saves, args.save)

    dataset_id = args.dataset_id or save.name.lower().replace(" ", "-")
    dataset = (args.dataset or ROOT / "exports" / "datasets" / dataset_id).resolve()
    staging = (args.staging_game_dir or dataset / "staging-game").resolve()
    if dataset == staging:
        raise SystemExit("dataset and staging game directory must not be the same path")
    scenes_root = dataset / "scenes"
    if scenes_root == staging or scenes_root in staging.parents:
        raise SystemExit("staging game directory must not sit inside <dataset>/scenes")

    if args.regions:
        region_requests = load_region_requests(args.regions)
    elif args.landscape:
        level_dat = save.path / "level.dat"
        if args.surface_y is None:
            # No developer override: centre the sample on the world's own SpawnY.
            # This is a smoke-capture convenience, not a chunk-heightmap sample, so
            # a missing/corrupt spawn fails loudly here instead of silently
            # assuming a height.
            try:
                spawn_x, surface_y, spawn_z = spawn_position_from_save(level_dat)
            except DatasetSchemaError as error:
                raise SystemExit(f"{error}; pass --surface-y <height> to override")
            _say(f"using SpawnY {surface_y} from {level_dat.name}; "
                 "pass --surface-y to override")
        else:
            spawn_x, spawn_z = spawn_from_save(level_dat)
            surface_y = args.surface_y
        bounds = landscape_region(spawn_x=spawn_x, spawn_z=spawn_z,
                                  surface_y=surface_y, radius=args.radius)
        region_requests = (RegionRequest(
            name=f"spawn-{spawn_x}-{spawn_z}", bounds=bounds, views=args.views),)
    else:
        raise SystemExit("pass --regions <file> or --landscape")

    staged_world = stage_world(save, staging, force=args.allow_overwrite)
    staged = staged_world.destination
    staging.mkdir(parents=True, exist_ok=True)
    _say(f"staged {save.path} -> {staged}")
    _say(f"source fingerprint {staged_world.source_fingerprint} "
         f"({staged_world.file_count} files, {staged_world.total_bytes} bytes)")

    # The capture runs on a hidden, unfocused desktop and must not disturb the
    # operator: the client would otherwise pause the integrated server, play
    # audio, or grab the mouse. Staged into the game directory (never the copied
    # world), preserving every other option.
    options_file = stage_capture_options(staging)
    staged_keys = ", ".join(f"{key}:{value}" for key, value in CAPTURE_OPTIONS.items())
    _say(f"staged client options {options_file} ({staged_keys})")

    arming: list[dict[str, object]] = []
    for request in region_requests:
        scene_id = _scene_id(save.name, request.name)
        if not isinstance(request.bounds, RegionBounds):
            raise SystemExit("region bounds did not parse")
        recipe = SceneRecipe(
            seed=args.seed,
            family="landscape" if args.landscape else "custom",
            dimensions=request.bounds.size,
            palette=(),
            features=("real-minecraft-capture", request.name),
            source="real-minecraft-capture",
        )
        plan = build_plan(
            dataset_id=dataset_id,
            scene_id=scene_id,
            world_identity=save.world_identity,
            region=request.bounds,
            seed=args.seed,
            view_count=request.views,
            target=request.target,
            staging_token=staged_world.staging_token,
            base_settings=fixed_settings,
        )
        write_scene_recipe(dataset, scene_id, recipe, force=args.allow_overwrite)
        plan_file = write_scene_plan(dataset, plan, force=args.allow_overwrite)
        staged_plan = safe_join(staging, MOD_DIR, "plans", f"{plan.plan_id}.json")
        write_json(staged_plan, plan.to_dict(), force=args.allow_overwrite)
        arming.append({
            "scene_id": scene_id,
            "plan_id": plan.plan_id,
            "plan_path": staged_plan.relative_to(staging).as_posix(),
            "plan_sha256": sha256_file(staged_plan),
            "world_identity": save.world_identity,
            "staging_token": plan.staging_token,
            "dataset_plan": str(plan_file),
        })
        _say(f"planned scene {scene_id}: {len(plan.jobs)} views -> {plan_file}")

    # The staging marker is what the mod reads before it mutates anything: it
    # carries the plan-required token plus the source fingerprint, so the client
    # refuses a world that is not this exact staged copy. The authoritative copy
    # lives in the copied world's own root - the mod resolves the real open save
    # root through the integrated server, because the save folder name need not
    # equal the level name - and the game-directory copy is provenance only. The
    # marker is written after the copy was verified, so it never perturbs the
    # staged fingerprint and the live tree is never rehashed.
    marker_payload = staging_marker_payload(
        token=staged_world.staging_token,
        source_fingerprint=staged_world.source_fingerprint,
        world_identity=save.world_identity,
        dataset_id=dataset_id,
        scene_id=str(arming[0]["scene_id"]) if arming else "",
        plan_id=str(arming[0]["plan_id"]) if arming else "",
    )
    world_marker = world_marker_path(staged)
    write_json(world_marker, marker_payload, force=args.allow_overwrite)
    _say(f"wrote in-world staging marker {world_marker}")

    provenance_marker = safe_join(staging, "config", MOD_DIR, "staging.json")
    write_json(provenance_marker, marker_payload, force=True)
    _say(f"wrote provenance marker {provenance_marker} (never proof of the open world)")

    if args.arm:
        if len(arming) != 1:
            raise SystemExit("--arm requires exactly one scene; one world copy, one plan")
        armed = dict(arming[0])
        armed["schema_version"] = 1
        armed["mod_contract_version"] = MOD_CONTRACT_VERSION
        armed["armed_by"] = "prepare_capture_dataset.py"
        armed["environment"] = {"python": platform.python_version()}
        armed_path = safe_join(staging, "config", MOD_DIR, "armed.json")
        write_json(armed_path, armed, force=args.allow_overwrite)
        _say(f"wrote {armed_path}; the mod stays inert until this file exists")

    _say(json.dumps({"dataset": str(dataset), "staging": str(staging),
                     "scenes": arming}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetSchemaError as error:
        raise SystemExit(f"refused: {error}")
