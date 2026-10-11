"""Finalize staged Minecraft captures into a hashed dataset scene and index.

Reads the mod's ``results.json``/``ground_truth.json`` from the staging game
directory, requires a real screenshot with a recorded hash for every planned
frame, converts the exported block states into ``reference.litematic``, and
writes ``capture.json`` plus a deterministic dataset ``index.json``.

Example
-------
    python scripts/finalize_capture_dataset.py --dataset exports/datasets/ds17

The default staging directory is ``<dataset>/staging-game``, matching
``prepare_capture_dataset.py``. Nothing here writes to a live ``.minecraft``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.dataset.finalize import (  # noqa: E402
    finalize_scene,
    write_dataset_index,
)
from mc_photogrammetry.dataset.schemas import DatasetSchemaError  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, required=True,
                        help="dataset root written by prepare_capture_dataset.py")
    parser.add_argument("--staging-game-dir", type=Path, default=None,
                        help="game directory holding the mod outputs and screenshots")
    parser.add_argument("--scene", action="append", default=None,
                        help="finalize only this scene id (repeatable)")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--provenance-mode", choices=("legacy", "report_only", "strict"),
                        default="legacy",
                        help="legacy reads old captures; report_only preserves incomplete evidence; strict requires certification")
    parser.add_argument("--provenance-expectation", type=Path, default=None)
    parser.add_argument("--provenance-expectation-sha256", default=None)
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    staging = (args.staging_game_dir or dataset / "staging-game").resolve()
    scenes_root = dataset / "scenes"
    if not scenes_root.is_dir():
        raise SystemExit(f"no scenes directory under {dataset}")

    wanted = set(args.scene) if args.scene else None
    finalized = []
    for scene in sorted(scenes_root.iterdir()):
        if not scene.is_dir() or not (scene / "plan.json").exists():
            continue
        if wanted is not None and scene.name not in wanted:
            continue
        try:
            entry = finalize_scene(dataset, scene.name, staging,
                                   force=args.allow_overwrite,
                                   provenance_mode=args.provenance_mode,
                                   provenance_expectation_path=args.provenance_expectation,
                                   provenance_expectation_sha256=args.provenance_expectation_sha256)
        except DatasetSchemaError as error:
            raise SystemExit(f"scene {scene.name}: {error}")
        finalized.append(entry["scene_id"])
        print(f"finalized {entry['scene_id']}: split={entry['split']} "
              f"captures={entry['captures']}")

    if not finalized:
        raise SystemExit("nothing to finalize")

    index = write_dataset_index(dataset, force=args.allow_overwrite)
    print(f"wrote {dataset / 'index.json'}: {index['summary']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetSchemaError as error:
        raise SystemExit(f"refused: {error}")
