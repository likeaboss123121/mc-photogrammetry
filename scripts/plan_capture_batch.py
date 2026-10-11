"""Plan and maintain a deterministic bulk capture queue without launching Minecraft.

Reads a declarative JSON batch spec plus the read-only save inventory, computes a
source-content identity for every selected save *before* expanding variants,
refuses duplicate/leaking source work, and writes a content-addressed queue
manifest with per-scene ``prepare``/``finalize`` argv, statuses, and attempt
evidence.  The default is a dry run; ``--materialize`` writes queue directories,
``manifest.json`` and the per-scene ``regions/*.json`` files only.

Nothing here stages a world, arms a plan, runs the factory scripts, or starts the
game.  Ground truth and capture metadata stay developer/evaluation-only.

Examples
--------
Dry-run a batch spec::

    python scripts/plan_capture_batch.py --spec docs/ds27-batch.json

Materialize the queue::

    python scripts/plan_capture_batch.py --spec docs/ds27-batch.json --materialize

Show status / print runnable commands / resume after an interruption::

    python scripts/plan_capture_batch.py --spec docs/ds27-batch.json --status
    python scripts/plan_capture_batch.py --spec docs/ds27-batch.json --commands
    python scripts/plan_capture_batch.py --spec docs/ds27-batch.json --resume
    python scripts/plan_capture_batch.py --spec docs/ds27-batch.json --mark house-v1 running

``--mark`` records a validated status transition; evidence is optional JSON::

    ... --mark house-v1 failed --evidence '{"stage": "finalize", "detail": "pose drift"}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.dataset.batch import (  # noqa: E402
    QUEUE_DIR_NAME,
    load_batch_spec,
    load_queue,
    manifest_path,
    manifest_payload,
    materialize_queue,
    plan_batch,
    queue_commands,
    resume_queue,
    set_status,
    write_queue_lines,
)
from mc_photogrammetry.dataset.console import write_safe  # noqa: E402
from mc_photogrammetry.dataset.hashing import canonical_json  # noqa: E402
from mc_photogrammetry.dataset.schemas import DatasetSchemaError  # noqa: E402


def _say(*values, sep: str = " ", end: str = "\n") -> None:
    write_safe(sys.stdout, sep.join(str(value) for value in values) + end)


def _queue_dir_from_spec(spec, args) -> Path:
    if args.queue_dir is not None:
        return Path(args.queue_dir)
    if spec.get("queue_dir"):
        return Path(str(spec["queue_dir"]))
    dataset = args.dataset if args.dataset is not None else spec.get("dataset")
    if not dataset:
        raise SystemExit("need --queue-dir, or a dataset in the spec or on the CLI")
    return Path(str(dataset)) / QUEUE_DIR_NAME


def _print_commands(rows) -> None:
    if not rows:
        _say("no actionable commands")
        return
    for row in rows:
        _say(f"{row['scene_id']} [{row['status']}]")
        _say("  prepare  " + json.dumps(row["prepare_argv"]))
        _say("  finalize " + json.dumps(row["finalize_argv"]))


def _parse_evidence(raw: str | None) -> dict:
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--evidence is not valid JSON: {error}")
    if not isinstance(payload, dict):
        raise SystemExit("--evidence must be a JSON object")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec", type=Path, required=True, help="declarative JSON batch spec")
    parser.add_argument("--game-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--staging-game-dir", type=Path, default=None)
    parser.add_argument("--queue-dir", type=Path, default=None)
    parser.add_argument("--materialize", action="store_true",
                        help="write the queue manifest and region files (default: dry run)")
    parser.add_argument("--status", action="store_true", help="print the materialized queue status")
    parser.add_argument("--commands", action="store_true",
                        help="print prepare/finalize argv for actionable scenes")
    parser.add_argument("--resume", action="store_true",
                        help="reset stale running scenes and report what to run")
    parser.add_argument("--mark", nargs=2, metavar=("SCENE_ID", "STATUS"),
                        help="record a validated status transition")
    parser.add_argument("--evidence", default=None, help="JSON object recorded with --mark")
    parser.add_argument("--json", action="store_true", help="emit the raw payload as JSON")
    args = parser.parse_args(argv)

    spec = load_batch_spec(args.spec)

    if args.mark:
        queue_dir = _queue_dir_from_spec(spec, args)
        scene_id, status = args.mark
        payload = set_status(queue_dir, scene_id, status,
                             evidence=_parse_evidence(args.evidence))
        if args.json:
            _say(canonical_json(payload))
        else:
            write_queue_lines(payload)
        return 0

    if args.resume:
        queue_dir = _queue_dir_from_spec(spec, args)
        result = resume_queue(queue_dir)
        if args.json:
            _say(canonical_json(result))
        else:
            _say(f"manifest {result['manifest_id']}")
            if result["reset"]:
                _say("reset stale running: " + ", ".join(result["reset"]))
            _say("actionable: " + (", ".join(result["actionable"]) or "none"))
            _say("complete: " + (", ".join(result["complete"]) or "none"))
        return 0

    if args.status:
        queue_dir = _queue_dir_from_spec(spec, args)
        payload = load_queue(queue_dir)
        if args.json:
            _say(canonical_json(payload))
        else:
            write_queue_lines(payload)
        return 0

    if args.commands and manifest_path(_queue_dir_from_spec(spec, args)).exists():
        payload = load_queue(_queue_dir_from_spec(spec, args))
        rows = queue_commands(payload)
        if args.json:
            _say(canonical_json(rows))
        else:
            _print_commands(rows)
        return 0

    plan = plan_batch(spec, game_dir=args.game_dir, dataset=args.dataset,
                      staging_game_dir=args.staging_game_dir, queue_dir=args.queue_dir)
    if args.materialize:
        path = materialize_queue(plan)
        payload = load_queue(plan.queue_dir)
        if args.json:
            _say(canonical_json(payload))
        else:
            _say(f"materialized {path}")
            write_queue_lines(payload)
    else:
        payload = manifest_payload(plan)
        if args.json:
            _say(canonical_json(payload))
        else:
            _say(f"dry run: {plan.manifest_id} (nothing written; pass --materialize)")
            write_queue_lines(payload)
    if args.commands:
        rows = queue_commands(payload) if args.materialize else [
            {"scene_id": scene["scene_id"], "status": scene["status"],
             "prepare_argv": scene["prepare_argv"], "finalize_argv": scene["finalize_argv"]}
            for scene in payload["scenes"]
        ]
        _print_commands(rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetSchemaError as error:
        raise SystemExit(f"refused: {error}")
