"""Run the one F2B evaluator-only context GT export (no screenshot capture).

This dedicated entry point has no queue/finalizer/default-reconstruction path.
Without ``--execute`` it performs read-only preflight only. Execution stages an
immutable fresh save copy, arms a zero-job context-only plan, and uses the same
disposable Prism worker, Desktop 2 controller, and process-tree teardown as the
capture factory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.dataset.executor import (  # noqa: E402
    CaptureModProvisioner,
    CaptureRequest,
    DisposablePrismWorker,
    PrismCaptureLauncher,
    WindowsDesktop2Controller,
    WindowsJunctionAdapter,
    read_options_file,
    scene_worker_identity,
)
from mc_photogrammetry.dataset.hashing import (  # noqa: E402
    sha256_file,
    staging_fingerprint_for,
)
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
from mc_photogrammetry.dataset.schemas import (  # noqa: E402
    MOD_CONTRACT_VERSION,
    SCHEMA_VERSION,
    DatasetSchemaError,
    RegionBounds,
)
from mc_photogrammetry.evaluation_context_export import (  # noqa: E402
    ContextExportError,
    build_bound_context_document,
    context_export_plan,
    validate_context_export_status,
)
from mc_photogrammetry.evaluation_frozen_planes import (  # noqa: E402
    FrozenPlaneEvaluationError,
    _load_truth,
    _read_json,
    _strict_json,
    _verify_capture_images,
)
from mc_photogrammetry.reconstruction.frozen_image_runner import (  # noqa: E402
    load_and_verify_frozen_image,
)


SCENE_ID = "canal-town-architectural-component-bbox"
WORLD_IDENTITY = "Canal Town"
SOURCE_STAGING_TOKEN = "b4aeeb844025284939e51754be3eba6cce16a876d2500bb9d14ab4ba8a48e54e"
CAPTURE_SHA256 = "c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6"
TARGET_GROUND_TRUTH_SHA256 = "36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a"
R2_FREEZE_SHA256 = "3c4e8e1b63f73748bde1f072d97dbfb4276ebc90c957889123760edcc6a1c90f"
CONTEXT_REGION = RegionBounds((140, 60, 340), (190, 130, 400))
DEFAULT_CAPTURE = ROOT / "exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/capture.json"
DEFAULT_TARGET_GT = ROOT / "exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/ground_truth.json"
DEFAULT_FREEZE = ROOT / "exports/evaluation/frozen-image-run-r2-atom-a1-canal-town-2026-09-12/freeze.json"
DEFAULT_OUTPUT_ROOT = ROOT / "exports/evaluation/context-gt-f2b-canal-town-2026-09-12"
DEFAULT_EVALUATION_ROOT = ROOT / "exports/evaluation/frozen-plane-e3-context-canal-town-2026-09-12"
DEFAULT_WORKER_ROOT = Path("B:/prismInstances")
DEFAULT_WORKER_TEMPLATE = Path("B:/prismInstances/MCPhotoCapture-Smoke-DS20C")
DEFAULT_PRISM = Path(r"C:/Users/Likea/AppData/Local/Programs/PrismLauncher/prismlauncher.exe")
DEFAULT_MOD = ROOT / "tools/minecraft-capture-mod/build/libs/mc-photo-capture-0.1.0.jar"
DEFAULT_TIMEOUT_SECONDS = 2400.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refuse_overlap(path: Path, protected: Path, label: str) -> None:
    resolved, guard = path.resolve(), protected.resolve()
    if resolved == guard or guard in resolved.parents or resolved in guard.parents:
        raise ContextExportError(f"{label} overlaps protected source path {guard}")


def _scene(saves, save_name: str):
    matches = [save for save in saves if save.name == save_name]
    if len(matches) != 1:
        raise ContextExportError(
            f"expected exactly one source save named {save_name!r}; found {len(matches)}")
    return matches[0]


def _worker(
    *,
    scene_id: str,
    staging_root: Path,
    worker_root: Path,
    worker_template: Path,
    prism_executable: Path,
    script_dir: Path,
) -> tuple[DisposablePrismWorker, str]:
    worker_name, instance_id = scene_worker_identity(
        scene_id, scope=staging_root,
        worker_dir_prefix="MCPhotoCapture-Worker",
        instance_id_prefix="MCPhotoCapture-Worker",
    )
    worker = DisposablePrismWorker(
        worker_root=worker_root,
        worker_dir=worker_name,
        instance_id=instance_id,
        template_dir=worker_template,
        executable=prism_executable,
        junction=WindowsJunctionAdapter(
            worker_root=worker_root,
            target_root=staging_root,
            script_dir=script_dir,
        ),
    )
    return worker, worker_name


def _preflight(args) -> dict[str, Any]:
    game_dir = (args.game_dir or default_game_dir()).resolve()
    output_root = args.output_root.resolve()
    staging_root = output_root / "staging-game"
    evaluation_root = args.evaluation_root.resolve()
    worker_root = args.worker_root.resolve()
    worker_template = args.worker_template.resolve()
    prism = args.prism_executable.resolve()
    mod = args.capture_mod.resolve()
    source_save = _scene(inventory_saves(game_dir), "Canal Town")

    for candidate, label in ((output_root, "context output root"),
                             (evaluation_root, "E3 output root")):
        try:
            candidate.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ContextExportError(f"refusing existing output root {candidate}")
        _refuse_overlap(candidate, game_dir, label)
        _refuse_overlap(candidate, source_save.path, label)
    if not worker_root.is_dir():
        raise ContextExportError(f"Prism worker root is absent: {worker_root}")
    if not (worker_template / "instance.cfg").is_file() or not (
            worker_template / "mmc-pack.json").is_file():
        raise ContextExportError(f"Prism worker template is incomplete: {worker_template}")
    if not prism.is_file():
        raise ContextExportError(f"Prism executable is absent: {prism}")
    if not mod.is_file() or Path(mod.name).name != mod.name:
        raise ContextExportError(f"capture mod JAR is absent or unsafe: {mod}")
    actual_mod_sha256 = sha256_file(mod)
    if actual_mod_sha256 != args.capture_mod_sha256.lower():
        raise ContextExportError(
            f"capture mod hash {actual_mod_sha256} differs from explicit pin "
            f"{args.capture_mod_sha256.lower()}")

    prism_cfg = Path(os.environ.get("APPDATA", "")) / "PrismLauncher" / "prismlauncher.cfg"
    cfg_values = {}
    if prism_cfg.is_file():
        for line in prism_cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                cfg_values[key.strip()] = value.strip()
    instance_dir = cfg_values.get("InstanceDir")
    if not instance_dir or Path(instance_dir).resolve() != worker_root:
        raise ContextExportError(
            f"Prism InstanceDir {instance_dir!r} does not resolve to worker root {worker_root}")

    capture_path = args.capture.resolve()
    target_gt_path = args.target_ground_truth.resolve()
    freeze_path = args.freeze.resolve()
    if _sha256(capture_path) != CAPTURE_SHA256:
        raise ContextExportError("capture metadata hash differs from the preregistration")
    if _sha256(target_gt_path) != TARGET_GROUND_TRUTH_SHA256:
        raise ContextExportError("target ground-truth hash differs from the preregistration")
    if _sha256(freeze_path) != R2_FREEZE_SHA256:
        raise ContextExportError("R2 freeze hash differs from the preregistration")

    freeze = load_and_verify_frozen_image(freeze_path)
    capture_bytes, capture_document = _read_json(capture_path, "capture metadata")
    images, camera_rows, input_rows = _verify_capture_images(
        freeze, capture_document, capture_path)
    target_truth = _load_truth(target_gt_path)[1]
    if (capture_document.get("scene_id") != SCENE_ID
            or capture_document.get("world_identity") != WORLD_IDENTITY
            or target_truth.scene_id != SCENE_ID
            or target_truth.world_identity != WORLD_IDENTITY
            or capture_document.get("region") != target_truth.region.to_dict()):
        raise ContextExportError("capture/target scene, world, or target-region identity differs")
    if not all(CONTEXT_REGION.minimum[i] <= target_truth.region.minimum[i]
               and CONTEXT_REGION.maximum[i] >= target_truth.region.maximum[i]
               for i in range(3)):
        raise ContextExportError("preregistered context region does not contain target GT")
    if math_product(CONTEXT_REGION.size) > 250_000:
        raise ContextExportError("preregistered context region exceeds export cap")
    source_fingerprint = staging_fingerprint_for([source_save.path])
    if source_fingerprint != SOURCE_STAGING_TOKEN:
        raise ContextExportError(
            f"source save fingerprint changed: {source_fingerprint}")
    worker, worker_name = _worker(
        scene_id=SCENE_ID, staging_root=staging_root, worker_root=worker_root,
        worker_template=worker_template, prism_executable=prism,
        script_dir=output_root / "desktop-helper",
    )
    try:
        worker.worker_dir.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ContextExportError(
            f"derived per-scene worker already exists; refusing reuse/overwrite: "
            f"{worker.worker_dir}")
    # Pin the exact fixed capture camera inventory and reject if any camera would
    # fall outside the context export's requested block bounds.
    camera_positions = [tuple(row["position"]) for row in capture_document["cameras"]]
    camera_inside = [all(
        CONTEXT_REGION.minimum[axis] <= position[axis] <= CONTEXT_REGION.maximum[axis]
        for axis in range(3)) for position in camera_positions]
    if len(camera_positions) != 8 or not all(camera_inside):
        raise ContextExportError("capture camera inventory differs from preregistered eight views")

    return {
        "game_dir": game_dir,
        "source_save": source_save,
        "source_fingerprint": source_fingerprint,
        "output_root": output_root,
        "staging_root": staging_root,
        "evaluation_root": evaluation_root,
        "worker": worker,
        "worker_name": worker_name,
        "prism": prism,
        "mod": mod,
        "mod_sha256": actual_mod_sha256,
        "capture_path": capture_path,
        "capture_bytes": capture_bytes,
        "capture_document": capture_document,
        "target_gt_path": target_gt_path,
        "target_truth": target_truth,
        "freeze_path": freeze_path,
        "freeze": freeze,
        "input_rows": input_rows,
        "image_count": len(images),
        "camera_count": len(camera_rows),
        "camera_positions": camera_positions,
    }


def math_product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _write_new_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContextExportError(f"refusing to overwrite {path}")
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ContextExportError(f"refusing existing temporary path {temporary}")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _execute(args, preflight: dict[str, Any]) -> dict[str, Any]:
    game_dir: Path = preflight["game_dir"]
    save = preflight["source_save"]
    output_root: Path = preflight["output_root"]
    staging_root: Path = preflight["staging_root"]
    worker: DisposablePrismWorker = preflight["worker"]

    # stage_world is copy-only, refuses an existing destination, verifies the
    # source both before and after copy, and verifies the staged fingerprint.
    staged = stage_world(save, staging_root, force=False)
    if staged.staging_token != SOURCE_STAGING_TOKEN:
        raise ContextExportError("staged copy token differs from the capture source token")
    options_path = stage_capture_options(staging_root)
    options = read_options_file(options_path)
    if any(options.get(key) != value for key, value in CAPTURE_OPTIONS.items()):
        raise ContextExportError("staged options do not meet the muted/background-safe contract")

    plan = context_export_plan(
        dataset_id="m4-f2b-canal-town-context",
        scene_id=SCENE_ID,
        world_identity=WORLD_IDENTITY,
        region=CONTEXT_REGION,
        staging_token=staged.staging_token,
    )
    plan_payload = plan.to_dict()
    plans_dir = staging_root / "mc-photo-capture" / "plans"
    plan_path = plans_dir / f"{plan.plan_id}.json"
    staged_plan_path = plan_path
    plan_record_path = output_root / "context-plan.json"
    _write_new_json(plan_record_path, plan_payload)
    _write_new_json(plan_path, plan_payload)

    marker_payload = staging_marker_payload(
        token=staged.staging_token,
        source_fingerprint=staged.source_fingerprint,
        world_identity=WORLD_IDENTITY,
        dataset_id=plan.dataset_id,
        scene_id=SCENE_ID,
        plan_id=plan.plan_id,
    )
    marker_path = world_marker_path(staged.destination)
    _write_new_json(marker_path, marker_payload)
    _write_new_json(staging_root / "config/mc-photo-capture/staging.json", marker_payload)

    mod_jar = CaptureModProvisioner(preflight["mod"], preflight["mod_sha256"]).provision(
        staging_root)
    if sha256_file(mod_jar) != preflight["mod_sha256"]:
        raise ContextExportError("staged capture mod JAR hash differs from preflight pin")

    armed_path = staging_root / "config/mc-photo-capture/armed.json"
    armed_payload = {
        "schema_version": SCHEMA_VERSION,
        "mod_contract_version": MOD_CONTRACT_VERSION,
        "plan_id": plan.plan_id,
        "plan_path": plan_path.relative_to(staging_root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "world_identity": WORLD_IDENTITY,
        "staging_token": staged.staging_token,
        "armed_by": "run_context_ground_truth_export.py",
    }
    _write_new_json(armed_path, armed_payload)

    # The explicit identity is scope+scene-derived. The existing smoke worker and
    # its Train Station junction are never reused or retargeted.
    worker.ensure_ready()
    worker.ensure_target(staging_root)
    request = CaptureRequest(
        scene_id=SCENE_ID,
        save_name=save.name,
        world_identity=WORLD_IDENTITY,
        dataset=output_root,
        staging_game_dir=staging_root,
        plan_id=plan.plan_id,
        staging_token=staged.staging_token,
        plan_path=plan_record_path,
        staged_plan_path=staged_plan_path,
        armed_path=armed_path,
        output_dir=staging_root / "mc-photo-capture" / "out" / plan.plan_id,
        timeout_seconds=float(args.timeout_seconds),
        mod_jar=mod_jar,
        staging_marker=marker_path,
    )
    launcher = PrismCaptureLauncher(
        lambda scene_id: worker.prism_instance(),
        poll_seconds=float(args.poll_seconds),
        desktop=WindowsDesktop2Controller(),
    )
    outcome = launcher.launch(request)
    if not outcome.completed or outcome.evidence.get("teardown_confirmed") is not True:
        raise ContextExportError(
            f"the single context-only Prism run did not complete cleanly: {outcome.detail}")

    output = request.output_dir
    status_bytes = (output / "run_status.json").read_bytes()
    status = _strict_json(status_bytes, "context run status")
    validate_context_export_status(status, plan=plan)
    result_bytes = (output / "results.json").read_bytes()
    results = _strict_json(result_bytes, "context results")
    if results != []:
        raise ContextExportError("context-only run recorded one or more capture results")
    raw_gt_bytes = (output / "ground_truth.json").read_bytes()
    raw_gt = _strict_json(raw_gt_bytes, "raw context ground truth")
    if list(output.rglob("*.png")):
        raise ContextExportError("context-only run unexpectedly produced a screenshot PNG")
    if staging_fingerprint_for([save.path]) != SOURCE_STAGING_TOKEN:
        raise ContextExportError("source save fingerprint changed during context export")

    context_document = build_bound_context_document(
        raw_gt,
        capture_sha256=CAPTURE_SHA256,
        target_ground_truth_sha256=TARGET_GROUND_TRUTH_SHA256,
        capture_document=preflight["capture_document"],
        ordered_images=preflight["input_rows"],
        target_ground_truth=preflight["target_truth"],
        context_region=CONTEXT_REGION,
    )
    context_path = output_root / "context-ground-truth.json"
    _write_new_json(context_path, context_document)
    # Validate the exact bytes just published through the same F2A strict loader.
    context_bytes, _ = _read_json(context_path, "bound context ground truth")
    _validate_bound_file(context_bytes, preflight, context_path)
    return {
        "run_status": str(output / "run_status.json"),
        "result_count": len(results),
        "context_records": len(context_document["blocks"]),
        "context_region": CONTEXT_REGION.to_dict(),
        "context_artifact": str(context_path),
        "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "capture_outcome": dict(outcome.evidence),
        "worker_dir": str(worker.worker_dir),
        "staged_world": str(staged.destination),
        "staging_token": staged.staging_token,
    }


def _validate_bound_file(context_bytes: bytes, preflight: dict[str, Any], path: Path) -> None:
    # The evaluator will repeat this on the one scoring call. This check gates
    # artifact publication and ensures no producer/evaluator schema divergence.
    doc = _strict_json(context_bytes, "published context ground truth")
    _validate_context_ground_truth_local(
        doc, preflight=preflight,
    )


def _validate_context_ground_truth_local(document: dict[str, Any], *, preflight: dict[str, Any]) -> None:
    from mc_photogrammetry.evaluation_frozen_planes import _validate_context_ground_truth

    _validate_context_ground_truth(
        document,
        capture_sha256=CAPTURE_SHA256,
        target_sha256=TARGET_GROUND_TRUTH_SHA256,
        capture_doc=preflight["capture_document"],
        input_rows=preflight["input_rows"],
        target_truth=preflight["target_truth"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--target-ground-truth", type=Path, default=DEFAULT_TARGET_GT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--worker-template", type=Path, default=DEFAULT_WORKER_TEMPLATE)
    parser.add_argument("--prism-executable", type=Path, default=DEFAULT_PRISM)
    parser.add_argument("--capture-mod", type=Path, default=DEFAULT_MOD)
    parser.add_argument("--capture-mod-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true",
                        help="stage one disposable copy and perform exactly one context-only export")
    args = parser.parse_args(argv)
    try:
        preflight = _preflight(args)
        if args.execute:
            result = _execute(args, preflight)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            desktop_state = WindowsDesktop2Controller().preflight()
            print(json.dumps({
                "preflight_only": True,
                "source_fingerprint": preflight["source_fingerprint"],
                "source_save": str(preflight["source_save"].path),
                "context_region": CONTEXT_REGION.to_dict(),
                "context_volume": math_product(CONTEXT_REGION.size),
                "capture_scene_id": SCENE_ID,
                "camera_count": preflight["camera_count"],
                "ordered_image_count": preflight["image_count"],
                "worker_identity": preflight["worker_name"],
                "worker_dir_absent": not preflight["worker"].worker_dir.exists(),
                "prism_executable": str(preflight["prism"]),
                "capture_mod_sha256": preflight["mod_sha256"],
                "desktop_count": desktop_state.desktop_count,
                "operator_desktop_index": desktop_state.current_index,
                "outputs_absent": True,
                "launch_occurred": False,
                "stage_created": False,
            }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ContextExportError, DatasetSchemaError, FrozenPlaneEvaluationError,
            OSError, ValueError, TypeError, KeyError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
