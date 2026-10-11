"""Safely execute a materialized capture queue through immutable scene workers.

This is the runner DS-27 deliberately left out: it takes the content-addressed
queue manifest written by ``plan_capture_batch.py`` and, for each actionable
scene, runs the queued ``prepare_capture_dataset.py`` with ``--arm``, stages the
pinned capture mod, initializes one disposable Prism worker per scene,
verifies the prepared plan/options/armed linkage, preflights the hidden desktop,
launches through :class:`~mc_photogrammetry.dataset.executor.PrismCaptureLauncher`,
and runs the queued ``finalize_capture_dataset.py``.  Scene status advances
through the existing validated ladder (``pending``/``failed`` -> ``running`` ->
``complete``/``failed``) with attempt evidence.

The default is a **dry run**: it prints the ordered stage list and the exact
prepare/capture/finalize commands, and constructs no worker, junction, desktop
controller, mod provisioner or launcher process, so it mutates nothing.
``--execute`` requires explicit worker, mod and Prism inputs:

* ``--worker-root`` / ``--worker-dir`` / ``--worker-instance-id`` /
  ``--worker-template`` describe a deterministic per-scene Prism worker family.
  Each worker gets an immutable ``minecraft`` junction to exactly one staging
  directory through :class:`~mc_photogrammetry.dataset.executor.WindowsJunctionAdapter`;
* ``--capture-mod`` and ``--capture-mod-sha256`` pin the exact capture jar that
  :class:`~mc_photogrammetry.dataset.executor.CaptureModProvisioner` verifies
  and copies into each scene's staging ``mods`` directory;
* ``--prism-executable`` (and optional ``--prism-root``) start the launcher.

The old per-scene ``--prism-instance-root``/``--prism-instance-map`` resolver is
dry-run-only for showing a command; ``--execute`` refuses it and uses the
immutable worker family under ``--worker-root``.

Examples
--------
Dry-run every actionable scene (also shows a worker command when its arguments
are supplied)::

    python scripts/run_capture_batch.py --spec docs/ds27-batch.json

Execute with deterministic per-scene workers under one explicit root::

    python scripts/run_capture_batch.py --spec docs/ds27-batch.json --execute \\
        --prism-executable "%LOCALAPPDATA%/Programs/PrismLauncher/prismlauncher.exe" \\
        --prism-root B:/prismRoot \\
        --worker-root B:/capture-workers --worker-dir mc-photo-capture \\
        --worker-instance-id mc-photo-capture --worker-template B:/prismInstances/template \\
        --capture-mod build/libs/mc-photo-capture-0.1.0.jar --capture-mod-sha256 <64 hex>

Resume a queue interrupted mid-scene, then continue past a failure::

    python scripts/run_capture_batch.py --spec ... --execute --resume \\
        --continue-on-failure --worker-root ... --worker-dir ... \\
        --worker-instance-id ... --worker-template ... \\
        --capture-mod ... --capture-mod-sha256 ... --prism-executable ...

A live run takes an exclusive ``<queue>/.executor.lock`` and writes a fsynced
``<queue>/executor-runs.jsonl`` ledger.  If a previous owner is known dead,
``--force-unlock`` reclaims a lock that cannot be proven stale (for example one
written by another host).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.dataset.batch import (  # noqa: E402
    QUEUE_DIR_NAME,
    load_batch_spec,
)
from mc_photogrammetry.dataset.console import write_safe  # noqa: E402
from mc_photogrammetry.dataset.executor import (  # noqa: E402
    BatchExecutor,
    CaptureModProvisioner,
    DisposablePrismWorker,
    ExecutorError,
    PrismCaptureLauncher,
    PrismInstance,
    WindowsDesktop2Controller,
    WindowsJunctionAdapter,
    render_report_lines,
    scene_worker_identity,
)
from mc_photogrammetry.dataset.hashing import canonical_json  # noqa: E402
from mc_photogrammetry.dataset.schemas import DatasetSchemaError  # noqa: E402


def _say(*values, sep: str = " ", end: str = "\n") -> None:
    write_safe(sys.stdout, sep.join(str(value) for value in values) + end)


def _dataset_dir(spec, args) -> Path:
    dataset = args.dataset if args.dataset is not None else spec.get("dataset")
    if not dataset:
        raise SystemExit("need --dataset, or a dataset in the spec or on the CLI")
    return Path(str(dataset))


def _queue_dir_from_spec(spec, args) -> Path:
    if args.queue_dir is not None:
        return Path(args.queue_dir)
    if spec.get("queue_dir"):
        return Path(str(spec["queue_dir"]))
    return _dataset_dir(spec, args) / QUEUE_DIR_NAME


def _staging_root(spec, args) -> Path:
    """The queue's ``dataset/staging-game`` root the junction adapter validates."""
    declared = spec.get("staging_game_dir")
    if declared:
        return Path(str(declared))
    return _dataset_dir(spec, args) / "staging-game"


def _build_worker(args, spec, scene_id: str) -> DisposablePrismWorker:
    """Construct a stable, scene-specific worker without filesystem I/O."""
    executable = Path(args.prism_executable) if args.prism_executable is not None else None
    prism_root = Path(args.prism_root) if args.prism_root is not None else None
    worker_dir, instance_id = scene_worker_identity(
        scene_id, scope=_staging_root(spec, args),
        worker_dir_prefix=str(args.worker_dir),
        instance_id_prefix=str(args.worker_instance_id))
    junction = WindowsJunctionAdapter(worker_root=Path(args.worker_root),
                                      target_root=_staging_root(spec, args))
    return DisposablePrismWorker(
        worker_root=Path(args.worker_root),
        worker_dir=worker_dir,
        instance_id=instance_id,
        template_dir=Path(args.worker_template),
        executable=executable,
        prism_root=prism_root,
        junction=junction,
    )


def _worker_args_supplied(args) -> bool:
    return all(value not in (None, "") for value in (
        args.worker_root, args.worker_dir, args.worker_instance_id, args.worker_template))


def _instance_resolver(args):
    """Build the dry-run scene -> :class:`PrismInstance` resolver the launcher calls.

    With ``--prism-instance-map`` the mapping is authoritative; otherwise the
    instance id is ``--prism-instance-template`` formatted with ``{scene_id}`` and
    resolved under ``--prism-instance-root``.  Nothing is created here: a missing
    instance is a refusal at launch time.  ``--execute`` never uses this path.
    """
    executable = Path(args.prism_executable)
    prism_root = Path(args.prism_root) if args.prism_root is not None else None
    if args.prism_instance_map is not None:
        map_path = Path(args.prism_instance_map)
        try:
            payload = json.loads(map_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise SystemExit(f"cannot read --prism-instance-map {map_path}: {error}")
        except json.JSONDecodeError as error:
            raise SystemExit(f"--prism-instance-map {map_path} is not valid JSON: {error}")
        if not isinstance(payload, dict) or not payload:
            raise SystemExit("--prism-instance-map must be a non-empty JSON object")
        root = Path(args.prism_instance_root) if args.prism_instance_root is not None else None

        def resolve(scene_id: str) -> PrismInstance:
            entry = payload.get(scene_id)
            if entry is None:
                raise DatasetSchemaError(
                    f"--prism-instance-map has no entry for scene {scene_id!r}")
            if isinstance(entry, str):
                instance_id, instance_dir = entry, None
            elif isinstance(entry, dict):
                instance_id = str(entry.get("instance_id") or entry.get("instance") or "")
                instance_dir = entry.get("instance_dir")
            else:
                raise DatasetSchemaError(
                    f"--prism-instance-map entry for {scene_id!r} must be a string or object")
            if not instance_id:
                raise DatasetSchemaError(
                    f"--prism-instance-map entry for {scene_id!r} has no instance id")
            if instance_dir is None:
                if root is None:
                    raise DatasetSchemaError(
                        f"--prism-instance-map entry for {scene_id!r} has no instance_dir "
                        "and no --prism-instance-root was given")
                instance_dir = root / instance_id
            return PrismInstance(str(instance_id), Path(str(instance_dir)),
                                 executable=executable, prism_root=prism_root)

        return resolve

    if args.prism_instance_root is None:
        raise SystemExit(
            "--prism-instance-root (or --prism-instance-map) is required to resolve "
            "the per-scene Prism instance")
    root = Path(args.prism_instance_root)
    template = str(args.prism_instance_template)

    def resolve(scene_id: str) -> PrismInstance:
        try:
            instance_id = template.format(scene_id=scene_id)
        except (KeyError, IndexError) as error:
            raise DatasetSchemaError(
                f"--prism-instance-template {template!r} cannot be formatted: {error}")
        return PrismInstance(instance_id, root / instance_id,
                             executable=executable, prism_root=prism_root)

    return resolve


def _require_execute_inputs(args) -> None:
    if args.prism_executable is None:
        raise SystemExit("--execute needs --prism-executable")
    required = [
        ("--worker-root", args.worker_root),
        ("--worker-dir", args.worker_dir),
        ("--worker-instance-id", args.worker_instance_id),
        ("--worker-template", args.worker_template),
        ("--capture-mod", args.capture_mod),
        ("--capture-mod-sha256", args.capture_mod_sha256),
    ]
    missing = [name for name, value in required if value in (None, "")]
    if missing:
        raise SystemExit("--execute requires " + ", ".join(missing))
    if args.prism_instance_map is not None or args.prism_instance_root is not None:
        raise SystemExit(
            "--execute uses deterministic per-scene workers; "
            "--prism-instance-map/--prism-instance-root are dry-run-only")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec", type=Path, required=True,
                        help="declarative JSON batch spec the queue was planned from")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--queue-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true",
                        help="actually run scenes (default: dry run)")
    parser.add_argument("--resume", action="store_true",
                        help="reset stale running scenes before executing")
    parser.add_argument("--scene", action="append", default=None,
                        help="run only this scene id (repeatable)")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="keep running later scenes after a failure")
    parser.add_argument("--force-unlock", action="store_true",
                        help="reclaim a queue lock whose owner cannot be proven dead")
    parser.add_argument("--prism-executable", type=Path, default=None)
    parser.add_argument("--prism-root", type=Path, default=None,
                        help="Prism application root passed as -d (optional)")
    parser.add_argument("--prism-instance-root", type=Path, default=None,
                        help="dry-run only: per-scene Prism instance directory")
    parser.add_argument("--prism-instance-template", default="{scene_id}",
                        help="dry-run only: instance folder name template")
    parser.add_argument("--prism-instance-map", type=Path, default=None,
                        help="dry-run only: JSON object mapping scene id -> instance")
    parser.add_argument("--worker-root", type=Path, default=None,
                        help="explicit root under which immutable per-scene workers live")
    parser.add_argument("--worker-dir", default=None,
                        help="sanitized prefix for each worker directory under --worker-root")
    parser.add_argument("--worker-instance-id", default=None,
                        help="sanitized prefix for each per-scene Prism instance id")
    parser.add_argument("--worker-template", type=Path, default=None,
                        help="template Prism instance the worker is initialized from")
    parser.add_argument("--capture-mod", type=Path, default=None,
                        help="exact capture mod jar to stage (no default)")
    parser.add_argument("--capture-mod-sha256", default=None,
                        help="SHA-256 pin of --capture-mod")
    parser.add_argument("--provenance-mode", choices=("report_only", "strict"),
                        default="report_only",
                        help="report_only preserves typed unavailable evidence; strict refuses uncertifiable runs")
    parser.add_argument("--python", default=None, help="interpreter for the factory scripts")
    parser.add_argument("--prepare-timeout", type=float, default=1800.0)
    parser.add_argument("--finalize-timeout", type=float, default=600.0)
    parser.add_argument("--capture-timeout", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    spec = load_batch_spec(args.spec)
    queue_dir = _queue_dir_from_spec(spec, args)

    # One interrupt event is shared by the executor's command runner, the Prism
    # launcher and the hidden-desktop controller, so a Ctrl-C tears down
    # whichever step is in flight.
    interrupt_event = threading.Event()
    launcher = None
    worker_for_scene = None
    provisioner = None

    if args.execute:
        _require_execute_inputs(args)
        worker_for_scene = lambda scene_id: _build_worker(args, spec, scene_id)
        provisioner = CaptureModProvisioner(Path(args.capture_mod),
                                            str(args.capture_mod_sha256))
        desktop = WindowsDesktop2Controller()
        launcher = PrismCaptureLauncher(
            lambda scene_id: worker_for_scene(scene_id).prism_instance(),
            poll_seconds=args.poll_seconds,
            desktop=desktop, interrupt=interrupt_event)
    elif _worker_args_supplied(args):
        # Dry run only: construct the worker to print its exact command, but run
        # no initialization, junction, desktop or mod I/O.
        worker_for_scene = lambda scene_id: _build_worker(args, spec, scene_id)
        launcher = PrismCaptureLauncher(
            lambda scene_id: worker_for_scene(scene_id).prism_instance(),
            poll_seconds=args.poll_seconds,
            interrupt=interrupt_event)
    elif args.prism_executable is not None:
        # Dry run only: show the per-scene command the legacy resolver would use.
        launcher = PrismCaptureLauncher(
            _instance_resolver(args), poll_seconds=args.poll_seconds,
            interrupt=interrupt_event)

    executor = BatchExecutor(
        queue_dir,
        launcher=launcher,
        worker_for_scene=worker_for_scene if args.execute else None,
        provisioner=provisioner if args.execute else None,
        python_executable=args.python,
        prepare_timeout=args.prepare_timeout,
        finalize_timeout=args.finalize_timeout,
        capture_timeout=args.capture_timeout,
        stop_on_failure=not args.continue_on_failure,
        dry_run=not args.execute,
        repo_root=ROOT,
        interrupt_event=interrupt_event,
        force_unlock=args.force_unlock,
        provenance_mode=args.provenance_mode,
    )

    if not args.execute:
        rows = executor.plan(args.scene)
        if args.json:
            _say(canonical_json({"dry_run": True, "queue_dir": str(queue_dir),
                                 "scenes": rows}))
        else:
            _say(f"dry run: {queue_dir} (nothing written or launched; pass --execute)")
            for row in rows:
                marker = "actionable" if row["actionable"] else "skip"
                _say(f"{row['scene_id']}  [{row['status']}]  {marker}")
                if not row["actionable"]:
                    continue
                _say("  stages   " + json.dumps(row["stages"]))
                _say("  prepare  " + json.dumps(row["prepare_argv"]))
                if row.get("capture_mod"):
                    _say("  mod      " + row["capture_mod"])
                if row.get("worker"):
                    _say("  worker   " + row["worker"])
                capture = row["capture"]
                _say("  capture  " + (json.dumps(capture) if capture is not None
                                      else "pass --worker-* (or --prism-executable/"
                                           "--prism-instance-root) to resolve the "
                                           "Prism command"))
                _say("  finalize " + json.dumps(row["finalize_argv"]))
        return 0

    report = executor.run(args.scene, resume=args.resume)
    if args.json:
        _say(canonical_json({
            "manifest_id": report.manifest_id,
            "dry_run": report.dry_run,
            "interrupted": report.interrupted,
            "outcomes": [outcome.as_dict() for outcome in report.outcomes],
        }))
    else:
        for line in render_report_lines(report):
            _say(line)
    return 1 if report.failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DatasetSchemaError, ExecutorError) as error:
        raise SystemExit(f"refused: {error}")
    except KeyboardInterrupt:
        raise SystemExit(130)
