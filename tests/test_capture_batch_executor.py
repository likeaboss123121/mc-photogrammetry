"""Focused tests for the safe sequential Prism batch executor (DS-31S).

Every test here runs without Minecraft, Prism, a real virtual desktop or a real
junction: the command runner, the capture launcher, the junction inspector and
the desktop controller are injected, and every worker/provisioner test uses a
temporary directory.  The tests pin the corrective implementation areas named by
DS-31S: the Windows junction adapter's validation/repoint script, the disposable
worker's template-safe initialization, the hidden-desktop preflight/move
contract, the capture-mod hash/copy contract, the exact lifecycle order, and the
CLI execute argument requirements.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from mc_photogrammetry.dataset import (
    CAPTURE_OPTIONS,
    LEDGER_NAME,
    LOCK_NAME,
    MOD_CONTRACT_VERSION,
    SCHEMA_VERSION,
    WORKER_MARKER_NAME,
    BatchExecutor,
    CaptureModProvisioner,
    CaptureOutcome,
    CommandResult,
    DatasetSchemaError,
    DesktopError,
    DesktopPreflight,
    DisposablePrismWorker,
    ExecutorError,
    JunctionError,
    LinkState,
    PrismCaptureLauncher,
    PrismInstance,
    ProvisionError,
    RunLedger,
    RunLock,
    SubprocessRunner,
    WindowsDesktop2Controller,
    WindowsJunctionAdapter,
    WorkerError,
    build_plan,
    load_queue,
    load_region_requests,
    materialize_queue,
    plan_batch,
    render_report_lines,
    set_status,
    stage_capture_options,
    staging_marker_payload,
    world_marker_path,
    write_json,
    write_report,
    write_scene_plan,
)
from mc_photogrammetry.dataset import executor as executor_module
from mc_photogrammetry.dataset.executor import (
    DESKTOP_MOVE_SCRIPT,
    DESKTOP_PREFLIGHT_SCRIPT,
    JUNCTION_ENSURE_SCRIPT,
    TARGET_DESKTOP_INDEX,
    CaptureRequest,
    _stop_process_tree,
    interpret_run_status,
    tail_text,
)
from mc_photogrammetry.dataset.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "a" * 64
SENTINEL = {"schema_version": SCHEMA_VERSION,
            "mod_contract_version": MOD_CONTRACT_VERSION,
            "plan_id": "plan-test",
            "staging_token": TOKEN,
            "run_failed": False,
            "ground_truth_exported": True,
            "errors": []}


# --- fixtures ---------------------------------------------------------------


def _level_dat(path: Path, *, level_name: str, spawn=(3, 71, -4)) -> None:
    import nbtlib
    from nbtlib import tag

    data = {
        "LevelName": tag.String(level_name),
        "LastPlayed": tag.Long(1_700_000_000_000),
        "SpawnX": tag.Int(spawn[0]),
        "SpawnY": tag.Int(spawn[1]),
        "SpawnZ": tag.Int(spawn[2]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbtlib.File(tag.Compound({"Data": tag.Compound(data)}), gzipped=True).save(path)


def _save(game: Path, name: str, *, level_name: str, payload: bytes = b"region") -> Path:
    world = game / "saves" / name
    (world / "region").mkdir(parents=True, exist_ok=True)
    (world / "region" / "r.0.0.mca").write_bytes(payload)
    _level_dat(world / "level.dat", level_name=level_name)
    return world


def _game(tmp_path: Path) -> Path:
    game = tmp_path / "game"
    _save(game, "Alpha", level_name="Alpha World", payload=b"alpha")
    _save(game, "Beta", level_name="Beta World", payload=b"beta")
    return game


def _region(name: str = "house", lo=(0, 60, 0), hi=(15, 75, 15)) -> dict:
    return {"name": name, "min": list(lo), "max": list(hi)}


def _spec(game: Path, dataset: Path, tasks: list, **over) -> dict:
    payload = {
        "schema_version": 1,
        "batch_id": "ds31-test",
        "game_dir": str(game),
        "dataset": str(dataset),
        "seed": 7,
        "views": 6,
        "tasks": tasks,
    }
    payload.update(over)
    return payload


def _two_scene_queue(tmp_path: Path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")},
        {"task_id": "b", "save": "Beta", "declared_split": "dev",
         "region": _region("yard")},
    ])
    plan = plan_batch(spec)
    materialize_queue(plan)
    assert [scene.scene_id for scene in plan.scenes] == ["alpha-house", "beta-yard"]
    return dataset / "queue", [scene.scene_id for scene in plan.scenes]


def _arg(argv, flag: str) -> str:
    return argv[list(argv).index(flag) + 1]


class _FakeFactory:
    """Synthesize exactly what ``prepare_capture_dataset.py --arm`` would write."""

    def __init__(self, queue_dir: Path, *, write_mod: bool = True):
        self.payload = load_queue(queue_dir)
        self.write_mod = bool(write_mod)
        self.by_staging = {
            str(Path(scene["staging_game_dir"]).resolve()): scene
            for scene in self.payload["scenes"]
        }
        self.prepared: list[str] = []

    def prepare(self, argv) -> None:
        staging = Path(_arg(argv, "--staging-game-dir")).resolve()
        scene = self.by_staging[str(staging)]
        dataset = Path(_arg(argv, "--dataset"))
        request = load_region_requests(Path(scene["region_file"]))[0]
        plan = build_plan(
            dataset_id=_arg(argv, "--dataset-id"),
            scene_id=scene["scene_id"],
            world_identity=scene["world_identity"],
            region=request.bounds,
            seed=int(_arg(argv, "--seed")),
            view_count=int(_arg(argv, "--views")),
            target=request.target,
            staging_token=TOKEN,
        )
        write_scene_plan(dataset, plan, force=True)
        staged_plan = staging / "mc-photo-capture" / "plans" / f"{plan.plan_id}.json"
        write_json(staged_plan, plan.to_dict(), force=True)
        write_json(staging / "config" / "mc-photo-capture" / "armed.json", {
            "schema_version": 1,
            "mod_contract_version": 1,
            "plan_id": plan.plan_id,
            "staging_token": TOKEN,
            "world_identity": plan.world_identity,
            "plan_sha256": sha256_file(staged_plan),
        }, force=True)
        # The accepted live-safety staging the preflight now demands: the client
        # options, the capture mod, and the in-world staging marker.
        stage_capture_options(staging)
        if self.write_mod:
            (staging / "mods").mkdir(parents=True, exist_ok=True)
            (staging / "mods" / "mc-photo-capture-0.1.0.jar").write_bytes(b"jar")
        write_json(world_marker_path(staging / "saves" / scene["save"]),
                   staging_marker_payload(
                       token=TOKEN, source_fingerprint=TOKEN,
                       world_identity=plan.world_identity,
                       dataset_id=_arg(argv, "--dataset-id"),
                       scene_id=scene["scene_id"], plan_id=plan.plan_id),
                   force=True)
        self.prepared.append(scene["scene_id"])


class _FakeRunner:
    def __init__(self, prepare_hook=None, failures=None):
        self.calls: list[dict] = []
        self.prepare_hook = prepare_hook
        self.failures = dict(failures or {})

    @property
    def scripts(self) -> list[str]:
        result = []
        for call in self.calls:
            joined = " ".join(call["argv"])
            result.append("prepare" if "prepare_capture_dataset.py" in joined else "finalize")
        return result

    def run(self, argv, *, cwd, timeout) -> CommandResult:
        argv = [str(item) for item in argv]
        self.calls.append({"argv": argv, "cwd": Path(cwd), "timeout": timeout})
        script = "prepare" if any(
            "prepare_capture_dataset.py" in item for item in argv) else "finalize"
        failure = self.failures.get(script)
        if failure is not None:
            return failure
        if script == "prepare" and self.prepare_hook is not None:
            self.prepare_hook(argv)
        return CommandResult(tuple(argv), 0, "ok", "", False, 0.01)


class _FakeLauncher:
    def __init__(self, completed: bool = True, detail: str = "capture complete"):
        self.requests: list[CaptureRequest] = []
        self.completed = completed
        self.detail = detail

    def launch(self, request: CaptureRequest) -> CaptureOutcome:
        self.requests.append(request)
        return CaptureOutcome(self.completed, self.detail, {"fake": True})


class _FakeJunction:
    """A create-once junction adapter that never touches a filesystem."""

    def __init__(self, order: list[str] | None = None):
        self.calls: list[tuple[Path, Path]] = []
        self.targets: dict[str, Path] = {}
        self.order = order

    def ensure(self, link: Path, target: Path) -> Path:
        key = os.path.normcase(str(link))
        current = self.targets.get(key)
        if current is not None and executor_module._path_key(current) != \
                executor_module._path_key(Path(target)):
            raise JunctionError("use a distinct per-scene worker")
        if current is None:
            self.calls.append((Path(link), Path(target)))
            self.targets[key] = Path(target)
        if self.order is not None:
            self.order.append("ensure-target")
        return Path(target)


class _FakeWorker:
    """The worker surface the executor uses, with no filesystem or Prism I/O."""

    def __init__(self, *, worker_dir: Path | None = None, order: list[str] | None = None):
        self.worker_dir = Path(worker_dir or "B:/workers/mc-photo-capture")
        self.ready = 0
        self.targets: list[Path] = []
        self.order = order

    def ensure_ready(self) -> dict:
        self.ready += 1
        if self.order is not None:
            self.order.append("worker-ready")
        return {"schema_version": 1}

    def ensure_target(self, staging_game_dir) -> Path:
        self.targets.append(Path(staging_game_dir))
        if self.order is not None:
            self.order.append("ensure-target")
        return Path(staging_game_dir)

    def prism_instance(self) -> PrismInstance:
        return PrismInstance("mc-photo-capture", self.worker_dir)


def _fake_provisioner_factory(source, sha256):
    class _Provisioner:
        def __init__(self):
            self.source = Path(source)
            self.expected_sha256 = str(sha256)

        def provision(self, staging_game_dir) -> Path:
            mods = Path(staging_game_dir) / "mods"
            mods.mkdir(parents=True, exist_ok=True)
            destination = mods / self.source.name
            destination.write_bytes(b"pinned")
            return destination

    return _Provisioner()


class _FakeDesktop:
    """A desktop controller that records the preflight/move calls the launcher makes."""

    def __init__(self, *, fail_preflight: bool = False, fail_move_at: int | None = None):
        self.preflights = 0
        self.moves = 0
        self.process_ids: list[int] = []
        self.fail_preflight = fail_preflight
        self.fail_move_at = fail_move_at

    def preflight(self) -> DesktopPreflight:
        self.preflights += 1
        if self.fail_preflight:
            raise DesktopError("no second desktop")
        return DesktopPreflight(2, 0)

    def move_windows(self, process_id: int) -> None:
        self.moves += 1
        self.process_ids.append(process_id)
        if self.fail_move_at is not None and self.moves == self.fail_move_at:
            raise DesktopError("current desktop changed")


def _template(tmp_path: Path, *, name: str = "capture-template") -> Path:
    template = tmp_path / name
    (template / "minecraft").mkdir(parents=True)
    (template / "instance.cfg").write_text(
        "[General]\nInstanceType=OneSix\nname=Template\niconKey=default\n",
        encoding="utf-8")
    (template / "mmc-pack.json").write_text('{"components": []}', encoding="utf-8")
    (template / "minecraft" / "options.txt").write_text(
        "template-do-not-touch", encoding="utf-8")
    (template / "extra.txt").write_text("keep out", encoding="utf-8")
    return template


class _FakeInspector:
    """A link inspector backed by a path -> :class:`LinkState` map."""

    def __init__(self, states=None):
        self.states = {os.path.normcase(str(path)): state
                       for path, state in (states or {}).items()}

    def inspect(self, path: Path) -> LinkState:
        return self.states.get(os.path.normcase(str(path)), LinkState("absent"))

    def set(self, path: Path, state: LinkState) -> None:
        self.states[os.path.normcase(str(path))] = state


class _ScriptRunner:
    """A command runner that answers with a caller-supplied handler."""

    def __init__(self, handler):
        self.calls: list[dict] = []
        self.handler = handler

    def run(self, argv, *, cwd, timeout) -> CommandResult:
        argv = [str(item) for item in argv]
        self.calls.append({"argv": argv, "cwd": Path(cwd), "timeout": timeout})
        return self.handler(argv)


def _executor(queue_dir, runner, launcher=None, **over):
    return BatchExecutor(queue_dir, runner=runner, launcher=launcher,
                         python_executable="py-test", repo_root=ROOT, **over)


# --- orchestration ----------------------------------------------------------


def test_executor_runs_each_scene_prepare_capture_finalize_in_order(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    runner = _FakeRunner(prepare_hook=factory.prepare)
    launcher = _FakeLauncher()
    report = _executor(queue, runner, launcher).run()

    assert report.dry_run is False
    assert report.completed == tuple(scene_ids)
    assert report.failed == () and report.skipped == ()
    assert factory.prepared == scene_ids
    assert runner.scripts == ["prepare", "finalize", "prepare", "finalize"]
    assert [request.scene_id for request in launcher.requests] == scene_ids
    assert [request.staging_token for request in launcher.requests] == [TOKEN, TOKEN]

    for call in runner.calls:
        assert call["argv"][0] == "py-test"
        assert call["cwd"] == ROOT
    prepare = runner.calls[0]["argv"]
    assert "--arm" in prepare and _arg(prepare, "--save") == "Alpha"
    payload = load_queue(queue)
    assert [scene["status"] for scene in payload["scenes"]] == ["complete", "complete"]
    assert [attempt["status"] for attempt in payload["scenes"][0]["attempts"]] == [
        "running", "complete"]
    assert payload["scenes"][0]["attempts"][-1]["evidence"]["stage"] == "finalize"


def test_dry_run_writes_nothing_and_never_calls_runner_or_launcher(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    runner = _FakeRunner()
    launcher = _FakeLauncher()
    report = _executor(queue, runner, launcher, dry_run=True).run()

    assert report.dry_run is True
    assert [item.status for item in report.outcomes] == ["pending", "pending"]
    assert runner.calls == [] and launcher.requests == []
    assert [scene["status"] for scene in load_queue(queue)["scenes"]] == [
        "pending", "pending"]


def test_dry_run_plan_rows_show_arm_and_both_factory_commands(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    rows = _executor(queue, _FakeRunner(), _FakeLauncher(), dry_run=True).plan()
    assert [row["scene_id"] for row in rows] == ["alpha-house", "beta-yard"]
    assert all(row["actionable"] for row in rows)
    assert "--arm" in rows[0]["prepare_argv"]
    assert "finalize_capture_dataset.py" in " ".join(rows[0]["finalize_argv"])
    assert rows[0]["stages"] == ["prepare --arm", "capture", "finalize"]


def test_prepare_failure_stops_before_capture_and_skips_later_scenes(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    failure = CommandResult(("prepare",), 2, "", "boom", False, 1.0)
    runner = _FakeRunner(prepare_hook=factory.prepare, failures={"prepare": failure})
    launcher = _FakeLauncher()
    report = _executor(queue, runner, launcher).run()

    assert report.failed == ("alpha-house",)
    assert report.skipped == ("beta-yard",)
    assert launcher.requests == []
    assert runner.scripts == ["prepare"]
    scene = load_queue(queue)["scenes"][0]
    assert scene["status"] == "failed"
    assert scene["attempts"][-1]["evidence"]["stage"] == "prepare"
    assert scene["attempts"][-1]["evidence"]["evidence"]["returncode"] == 2


def test_capture_failure_marks_failed_before_finalize(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    runner = _FakeRunner(prepare_hook=factory.prepare)
    launcher = _FakeLauncher(completed=False, detail="game menu on screen")
    report = _executor(queue, runner, launcher).run()

    assert report.failed == ("alpha-house",)
    assert runner.scripts == ["prepare"]
    scene = load_queue(queue)["scenes"][0]
    assert scene["attempts"][-1]["evidence"]["stage"] == "capture"
    assert "game menu" in scene["attempts"][-1]["evidence"]["detail"]


def test_finalize_failure_marks_failed_after_a_successful_capture(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    failure = CommandResult(("finalize",), 3, "", "pose drift", False, 2.0)
    runner = _FakeRunner(prepare_hook=factory.prepare, failures={"finalize": failure})
    launcher = _FakeLauncher()
    report = _executor(queue, runner, launcher).run()

    assert report.failed == ("alpha-house",)
    assert len(launcher.requests) == 1
    assert runner.scripts == ["prepare", "finalize"]
    scene = load_queue(queue)["scenes"][0]
    assert scene["attempts"][-1]["evidence"]["stage"] == "finalize"
    assert scene["attempts"][-1]["evidence"]["evidence"]["returncode"] == 3


def test_continue_on_failure_runs_the_remaining_scenes(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)

    class _FirstOnlyFails(_FakeRunner):
        def run(self, argv, *, cwd, timeout):
            if "prepare_capture_dataset.py" in " ".join(str(a) for a in argv) and not self.calls:
                argv = [str(item) for item in argv]
                self.calls.append({"argv": argv, "cwd": Path(cwd), "timeout": timeout})
                return CommandResult(tuple(argv), 1, "", "first only", False, 0.1)
            return super().run(argv, cwd=cwd, timeout=timeout)

    runner = _FirstOnlyFails(prepare_hook=factory.prepare)
    report = _executor(queue, runner, _FakeLauncher(), stop_on_failure=False).run()
    assert report.failed == ("alpha-house",)
    assert report.completed == ("beta-yard",)
    assert runner.scripts == ["prepare", "prepare", "finalize"]


def test_timeout_command_is_a_failure_with_timed_out_evidence(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    failure = CommandResult(("prepare",), -1, "", "", True, 1800.0)
    runner = _FakeRunner(prepare_hook=factory.prepare, failures={"prepare": failure})
    report = _executor(queue, runner, _FakeLauncher()).run()
    assert report.failed == ("alpha-house",)
    scene = load_queue(queue)["scenes"][0]
    assert scene["attempts"][-1]["evidence"]["detail"] == "prepare timed out"
    assert scene["attempts"][-1]["evidence"]["evidence"]["timed_out"] is True


# --- pre-launch refusal -----------------------------------------------------


def _prepared_scene(tmp_path):
    """Materialize a two-scene queue and synthesize scene one's prepare output."""
    queue, scene_ids = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    payload = load_queue(queue)
    scene = payload["scenes"][0]
    dataset = Path(payload["dataset"]).resolve()
    probe = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    factory.prepare(probe._prepare_argv(scene, dataset))
    return queue, scene, factory, dataset


def _armed_path(scene) -> Path:
    return Path(scene["staging_game_dir"]) / "config" / "mc-photo-capture" / "armed.json"


def test_missing_armed_marker_refuses_before_any_launch(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    launcher = _FakeLauncher()
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=launcher)
    _armed_path(scene).unlink()
    with pytest.raises(DatasetSchemaError, match="unarmed world"):
        executor._capture_request(scene, dataset)
    assert launcher.requests == []


def test_armed_token_mismatch_refuses_the_capture_request(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    armed = json.loads(_armed_path(scene).read_text(encoding="utf-8"))
    armed["staging_token"] = "b" * 64
    _armed_path(scene).write_text(json.dumps(armed), encoding="utf-8")
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="staging token"):
        executor._capture_request(scene, dataset)


def test_staged_plan_digest_mismatch_refuses_the_capture_request(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    armed = json.loads(_armed_path(scene).read_text(encoding="utf-8"))
    armed["plan_sha256"] = "0" * 64
    _armed_path(scene).write_text(json.dumps(armed), encoding="utf-8")
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="digest"):
        executor._capture_request(scene, dataset)


def test_a_bad_armed_marker_fails_the_scene_at_the_arm_stage(tmp_path):
    queue, _, _, _ = _prepared_scene(tmp_path)

    def prepare_then_break_token(argv):
        scene = load_queue(queue)["scenes"][0]
        factory = _FakeFactory(queue)
        factory.prepare(argv)
        armed = json.loads(_armed_path(scene).read_text(encoding="utf-8"))
        armed["plan_id"] = "someone-else"
        _armed_path(scene).write_text(json.dumps(armed), encoding="utf-8")

    launcher = _FakeLauncher()
    runner = _FakeRunner(prepare_hook=prepare_then_break_token)
    report = _executor(queue, runner, launcher).run(["alpha-house"])
    assert report.failed == ("alpha-house",)
    assert report.outcomes[0].stage == "arm"
    assert launcher.requests == []


# --- selection and resume ---------------------------------------------------


def test_complete_scenes_are_skipped_and_running_needs_resume(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    set_status(queue, scene_ids[0], "running")
    set_status(queue, scene_ids[0], "complete")
    set_status(queue, scene_ids[1], "running")

    factory = _FakeFactory(queue)
    runner = _FakeRunner(prepare_hook=factory.prepare)
    launcher = _FakeLauncher()
    report = _executor(queue, runner, launcher).run()

    assert report.completed == ()
    assert report.skipped == (scene_ids[0], scene_ids[1])
    assert "resume" in report.outcomes[1].detail
    assert launcher.requests == []

    resumed = _executor(queue, runner, launcher).run(resume=True)
    assert resumed.completed == (scene_ids[1],)
    assert resumed.skipped == (scene_ids[0],)
    assert len(launcher.requests) == 1


def test_unknown_scene_selection_is_refused(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    executor = _executor(queue, _FakeRunner(), _FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="no scene"):
        executor.run(["no-such-scene"])


def test_execute_without_a_launcher_is_an_executor_error(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    with pytest.raises(ExecutorError, match="launcher"):
        _executor(queue, _FakeRunner(), None).run()


# --- Prism instance contract ------------------------------------------------


def _instance(tmp_path: Path, *, instance_id="cap", game_dir=None) -> PrismInstance:
    instance_dir = tmp_path / instance_id
    (instance_dir).mkdir(parents=True, exist_ok=True)
    (instance_dir / "instance.cfg").write_text("[General]\nname=Capture\n", encoding="utf-8")
    (instance_dir / "minecraft").mkdir(exist_ok=True)
    return PrismInstance(instance_id, instance_dir,
                         executable=Path("C:/prism/prismlauncher.exe"),
                         prism_root=Path("B:/prismRoot"))


def test_prism_instance_verifies_its_game_directory(tmp_path):
    instance = _instance(tmp_path)
    expected = instance.instance_dir / "minecraft"
    assert instance.verify_game_dir(expected) == expected.resolve()
    with pytest.raises(DatasetSchemaError, match="wrong world"):
        instance.verify_game_dir(tmp_path / "somewhere-else")


def test_prism_instance_refuses_a_directory_without_instance_cfg(tmp_path):
    plain = tmp_path / "plain"
    (plain / "minecraft").mkdir(parents=True)
    instance = PrismInstance("plain", plain)
    with pytest.raises(DatasetSchemaError, match="instance.cfg"):
        instance.game_dir()


def test_prism_launch_argv_is_instance_scoped(tmp_path):
    instance = _instance(tmp_path)
    argv = instance.launch_argv(world="Train station")
    assert argv == (str(instance.executable), "-d", str(instance.prism_root),
                    "-l", "cap", "-w", "Train station")


# --- Prism launcher wait/terminate logic ------------------------------------


class _FakeProcess:
    def __init__(self, returncode: int | None = None):
        self.pid = 4242
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.waited = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waited += 1
        return self.returncode if self.returncode is not None else 0

    def terminate(self):
        self.terminated = True
        self.returncode = 1

    def kill(self):
        self.killed = True
        self.returncode = -9


def _launch_request(tmp_path: Path, *, timeout=5.0) -> tuple[CaptureRequest, Path]:
    instance_dir = tmp_path / "inst"
    instance_dir.mkdir()
    (instance_dir / "instance.cfg").write_text("", encoding="utf-8")
    game_dir = instance_dir / "minecraft"
    game_dir.mkdir()
    output = instance_dir / "mc-photo-capture" / "out" / "plan-test"
    output.mkdir(parents=True)
    request = CaptureRequest(
        scene_id="scene", save_name="Alpha", world_identity="Alpha World",
        dataset=tmp_path, staging_game_dir=game_dir, plan_id="plan-test",
        staging_token=TOKEN, plan_path=tmp_path / "plan.json",
        staged_plan_path=tmp_path / "staged.json", armed_path=tmp_path / "armed.json",
        output_dir=output, timeout_seconds=timeout)
    return request, output


def _launcher(tmp_path, process, *, status_writer=None, stopper=None, poll_seconds=0.25,
              desktop=None):
    instance = _instance(tmp_path, instance_id="inst")
    state = {"now": 0.0}

    def tick(seconds):
        state["now"] += seconds
        if status_writer is not None:
            status_writer()

    return PrismCaptureLauncher(
        lambda scene_id: instance,
        poll_seconds=poll_seconds,
        popener=lambda argv, cwd: process,
        stopper=stopper or (lambda proc, *, grace: (proc.terminate(), True)[1]),
        sleeper=tick,
        clock=lambda: state["now"],
        desktop=desktop,
    )


def test_prism_launcher_accepts_only_a_clean_run_status(tmp_path):
    request, output = _launch_request(tmp_path)
    writes = {"done": False}

    def write_status():
        if not writes["done"]:
            (output / "run_status.json").write_text(
                json.dumps({**SENTINEL, "plan_id": "plan-test"}), encoding="utf-8")
            writes["done"] = True

    process = _FakeProcess(returncode=None)
    stopped: list[bool] = []
    launcher = _launcher(tmp_path, process, status_writer=write_status,
                         stopper=lambda proc, *, grace: (stopped.append(True), True)[1])
    outcome = launcher.launch(request)
    assert outcome.completed is True
    assert stopped == [True]
    assert outcome.evidence["game_dir"] == str((tmp_path / "inst" / "minecraft").resolve())
    assert outcome.evidence["teardown_confirmed"] is True


def test_prism_launcher_reports_a_failed_run_status(tmp_path):
    request, output = _launch_request(tmp_path)
    writes = {"done": False}

    def write_status():
        if not writes["done"]:
            (output / "run_status.json").write_text(json.dumps({
                **SENTINEL, "plan_id": "plan-test", "run_failed": True,
                "ground_truth_exported": False, "errors": ["frame 01: screen open"]}),
                encoding="utf-8")
            writes["done"] = True

    process = _FakeProcess(returncode=None)
    launcher = _launcher(tmp_path, process, status_writer=write_status,
                         stopper=lambda proc, *, grace: True)
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert "screen open" in outcome.detail


def test_prism_launcher_clears_a_stale_sentinel_before_launch(tmp_path):
    """A previous run's sentinel must never satisfy a new capture."""
    request, output = _launch_request(tmp_path)
    (output / "run_status.json").write_text(json.dumps(SENTINEL), encoding="utf-8")
    process = _FakeProcess(returncode=None)
    writes = {"done": False}

    def write_status():
        if not writes["done"]:
            (output / "run_status.json").write_text(json.dumps(SENTINEL),
                                                    encoding="utf-8")
            writes["done"] = True

    launcher = _launcher(tmp_path, process, status_writer=write_status,
                         stopper=lambda proc, *, grace: True)
    outcome = launcher.launch(request)
    assert outcome.completed is True
    assert outcome.evidence["stale_sentinel_removed"] is True


def test_prism_launcher_refuses_a_backdated_sentinel(tmp_path):
    request, output = _launch_request(tmp_path)
    process = _FakeProcess(returncode=None)

    def write_status():
        path = output / "run_status.json"
        path.write_text(json.dumps(SENTINEL), encoding="utf-8")
        os.utime(path, (1.0, 1.0))

    launcher = PrismCaptureLauncher(
        lambda scene_id: _instance(tmp_path, instance_id="inst"),
        poll_seconds=0.25,
        popener=lambda argv, cwd: process,
        stopper=lambda proc, *, grace: True,
        sleeper=lambda seconds: write_status(),
        clock=lambda: 0.0,
    )
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert outcome.evidence["sentinel_fresh"] is False
    assert "predates" in outcome.detail


@pytest.mark.parametrize("mutate, needle", [
    ({"staging_token": "b" * 64}, "staging token"),
    ({"schema_version": 99}, "schema_version"),
    ({"mod_contract_version": 99}, "mod_contract_version"),
])
def test_prism_launcher_refuses_a_mismatched_sentinel_contract(tmp_path, mutate, needle):
    request, output = _launch_request(tmp_path)
    process = _FakeProcess(returncode=None)
    payload = {**SENTINEL, **mutate}

    def write_status():
        (output / "run_status.json").write_text(json.dumps(payload), encoding="utf-8")

    launcher = PrismCaptureLauncher(
        lambda scene_id: _instance(tmp_path, instance_id="inst"),
        poll_seconds=0.25,
        popener=lambda argv, cwd: process,
        stopper=lambda proc, *, grace: True,
        sleeper=lambda seconds: write_status(),
        clock=lambda: 0.0,
    )
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert needle in outcome.detail


def test_prism_launcher_fails_closed_when_teardown_is_unconfirmed(tmp_path):
    request, output = _launch_request(tmp_path)
    process = _FakeProcess(returncode=None)

    def write_status():
        (output / "run_status.json").write_text(json.dumps(SENTINEL), encoding="utf-8")

    launcher = PrismCaptureLauncher(
        lambda scene_id: _instance(tmp_path, instance_id="inst"),
        poll_seconds=0.25,
        popener=lambda argv, cwd: process,
        stopper=lambda proc, *, grace: False,
        sleeper=lambda seconds: write_status(),
        clock=lambda: 0.0,
    )
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert outcome.stage == "teardown"
    assert outcome.evidence["teardown_failed"] is True
    assert "orphan" in outcome.detail


def test_prism_launcher_reports_an_interrupt_as_a_stopped_failure(tmp_path):
    request, _ = _launch_request(tmp_path)
    process = _FakeProcess(returncode=None)
    interrupt = threading.Event()
    stopped: list[bool] = []
    launcher = PrismCaptureLauncher(
        lambda scene_id: _instance(tmp_path, instance_id="inst"),
        poll_seconds=0.25,
        popener=lambda argv, cwd: process,
        stopper=lambda proc, *, grace: (stopped.append(True), True)[1],
        sleeper=lambda seconds: interrupt.set(),
        clock=lambda: 0.0,
        interrupt=interrupt,
    )
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert outcome.stage == "interrupted"
    assert stopped == [True]


def test_prism_launcher_times_out_and_stops_the_process(tmp_path):
    request, _ = _launch_request(tmp_path, timeout=1.0)
    process = _FakeProcess(returncode=None)
    stopped: list[bool] = []
    launcher = _launcher(
        tmp_path, process, status_writer=None, poll_seconds=0.25,
        stopper=lambda proc, *, grace: stopped.append(True))
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert "no run_status.json" in outcome.detail
    assert outcome.evidence["timed_out"] is True
    assert stopped == [True]


def test_prism_launcher_refuses_an_unrelated_instance_before_starting(tmp_path):
    request, _ = _launch_request(tmp_path)
    other = _instance(tmp_path, instance_id="other")
    started: list[bool] = []
    launcher = PrismCaptureLauncher(lambda scene_id: other,
                                    popener=lambda argv, cwd: started.append(True),
                                    stopper=lambda proc, *, grace: None)
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert outcome.evidence["stage"] == "verify"
    assert started == []


# --- run-status interpretation ----------------------------------------------


@pytest.mark.parametrize("payload, completed", [
    ({"plan_id": "p", "run_failed": False, "ground_truth_exported": True}, True),
    ({"plan_id": "p", "run_failed": True, "ground_truth_exported": True}, False),
    ({"plan_id": "p", "run_failed": False, "ground_truth_exported": False}, False),
    ({"run_failed": False, "ground_truth_exported": True}, True),
])
def test_interpret_run_status(payload, completed):
    assert interpret_run_status(payload, plan_id="p").completed is completed


def test_interpret_run_status_rejects_a_foreign_plan():
    outcome = interpret_run_status(
        {"plan_id": "other", "run_failed": False, "ground_truth_exported": True},
        plan_id="p")
    assert outcome.completed is False
    assert "not 'p'" in outcome.detail


# --- command runner ---------------------------------------------------------


def test_subprocess_runner_runs_a_real_command_without_a_shell(tmp_path):
    result = SubprocessRunner().run(
        [sys.executable, "-c", "import sys; print('hello'); sys.exit(0)"],
        cwd=tmp_path, timeout=60)
    assert result.ok and result.returncode == 0
    assert "hello" in result.stdout


def test_subprocess_runner_reports_a_nonzero_exit(tmp_path):
    result = SubprocessRunner().run(
        [sys.executable, "-c", "import sys; sys.stderr.write('nope'); sys.exit(4)"],
        cwd=tmp_path, timeout=60)
    assert not result.ok and result.returncode == 4
    assert "nope" in result.stderr
    assert result.evidence()["returncode"] == 4


def test_tail_text_bounds_output():
    assert tail_text("abc", 10) == "abc"
    bounded = tail_text("x" * 100, 10)
    assert bounded.endswith("x" * 10) and "truncated" in bounded


# --- report rendering -------------------------------------------------------


def test_report_lines_and_encoding_safe_write(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    report = _executor(queue, _FakeRunner(), _FakeLauncher(), dry_run=True).run()
    lines = render_report_lines(report)
    assert lines[0].startswith("batch executor dry run")
    assert any(line.startswith("alpha-house") for line in lines)

    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", newline="")
    write_report(report, stream=stream)
    stream.flush()
    text = buffer.getvalue().decode("cp1252")
    assert "alpha-house" in text
    stream.detach()


# --- CLI --------------------------------------------------------------------


def _cli_module():
    path = ROOT / "scripts" / "run_capture_batch.py"
    spec = importlib.util.spec_from_file_location("ds31_run_capture_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_spec(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_is_a_dry_run_by_default_and_refuses_execute_without_prism(tmp_path, capsys):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")}])
    spec_path = _write_spec(tmp_path, spec)
    materialize_queue(plan_batch(spec))
    cli = _cli_module()

    assert cli.main(["--spec", str(spec_path)]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "--arm" in out and "alpha-house" in out
    assert load_queue(dataset / "queue")["scenes"][0]["status"] == "pending"

    with pytest.raises(SystemExit, match="prism-executable"):
        cli.main(["--spec", str(spec_path), "--execute"])
    assert load_queue(dataset / "queue")["scenes"][0]["status"] == "pending"


def test_cli_instance_map_resolves_a_per_scene_prism_command(tmp_path, capsys):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")}])
    spec_path = _write_spec(tmp_path, spec)
    materialize_queue(plan_batch(spec))
    map_path = tmp_path / "instances.json"
    map_path.write_text(json.dumps({
        "alpha-house": {"instance_id": "cap-alpha",
                        "instance_dir": str(tmp_path / "cap-alpha")}}), encoding="utf-8")
    cli = _cli_module()
    assert cli.main(["--spec", str(spec_path),
                     "--prism-executable", "C:/prism/prismlauncher.exe",
                     "--prism-instance-map", str(map_path)]) == 0
    out = capsys.readouterr().out
    assert "cap-alpha" in out and "prismlauncher.exe" in out and "-w" in out

    with pytest.raises(SystemExit, match="cannot read --prism-instance-map"):
        cli.main(["--spec", str(spec_path), "--prism-executable", "prism.exe",
                  "--prism-instance-map", str(tmp_path / "absent.json")])


def test_cli_executes_a_scene_end_to_end_with_injected_seams(tmp_path, monkeypatch, capsys):
    """The real CLI wiring with every external boundary faked: one worker scene."""
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")}])
    spec_path = _write_spec(tmp_path, spec)
    materialize_queue(plan_batch(spec))
    factory = _FakeFactory(dataset / "queue")
    runner = _FakeRunner(prepare_hook=factory.prepare)
    launcher = _FakeLauncher()
    worker = _FakeWorker()
    provisioner_source = tmp_path / "build" / "mc-photo-capture-0.1.0.jar"
    provisioner_source.parent.mkdir(parents=True, exist_ok=True)
    provisioner_source.write_bytes(b"pinned")
    cli = _cli_module()
    real_executor = cli.BatchExecutor

    def make_executor(queue_dir, **kwargs):
        kwargs["runner"] = runner
        kwargs["repo_root"] = ROOT
        return real_executor(queue_dir, **kwargs)

    monkeypatch.setattr(cli, "BatchExecutor", make_executor)
    monkeypatch.setattr(cli, "PrismCaptureLauncher", lambda *a, **k: launcher)
    monkeypatch.setattr(cli, "WindowsJunctionAdapter", lambda **k: object())
    monkeypatch.setattr(cli, "DisposablePrismWorker", lambda **k: worker)
    monkeypatch.setattr(cli, "WindowsDesktop2Controller", lambda: object())
    monkeypatch.setattr(cli, "CaptureModProvisioner", _fake_provisioner_factory)
    assert cli.main(["--spec", str(spec_path), "--execute",
                     "--prism-executable", "C:/prism/prismlauncher.exe",
                     "--worker-root", str(tmp_path / "workers"),
                     "--worker-dir", "mc-photo-capture",
                     "--worker-instance-id", "mc-photo-capture",
                     "--worker-template", str(tmp_path / "template"),
                     "--capture-mod", str(provisioner_source),
                     "--capture-mod-sha256", sha256_file(provisioner_source)]) == 0
    assert "complete: alpha-house" in capsys.readouterr().out
    assert load_queue(dataset / "queue")["scenes"][0]["status"] == "complete"
    assert worker.ready == 1 and len(worker.targets) == 1
    assert launcher.requests[0].mod_jar is not None


def test_cli_execute_requires_every_worker_and_mod_argument(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")}])
    spec_path = _write_spec(tmp_path, spec)
    materialize_queue(plan_batch(spec))
    cli = _cli_module()
    base = ["--spec", str(spec_path), "--execute",
            "--prism-executable", "C:/prism/prismlauncher.exe",
            "--worker-root", str(tmp_path / "workers"),
            "--worker-dir", "mc-photo-capture",
            "--worker-instance-id", "mc-photo-capture",
            "--worker-template", str(tmp_path / "template"),
            "--capture-mod", str(tmp_path / "mod.jar"),
            "--capture-mod-sha256", "0" * 64]

    required_flags = ["--worker-root", "--worker-dir", "--worker-instance-id",
                      "--worker-template", "--capture-mod", "--capture-mod-sha256"]
    for flag in required_flags:
        trimmed: list[str] = []
        skip_next = False
        for item in base:
            if skip_next:
                skip_next = False
                continue
            if item == flag:
                skip_next = True
                continue
            trimmed.append(item)
        with pytest.raises(SystemExit, match=flag.replace("--", "")):
            cli.main(trimmed)
    assert load_queue(dataset / "queue")["scenes"][0]["status"] == "pending"


def test_cli_execute_refuses_the_old_per_scene_instance_mapping(tmp_path):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")}])
    spec_path = _write_spec(tmp_path, spec)
    materialize_queue(plan_batch(spec))
    cli = _cli_module()
    with pytest.raises(SystemExit, match="dry-run-only"):
        cli.main(["--spec", str(spec_path), "--execute",
                  "--prism-executable", "C:/prism/prismlauncher.exe",
                  "--prism-instance-root", str(tmp_path / "instances"),
                  "--worker-root", str(tmp_path / "workers"),
                  "--worker-dir", "mc-photo-capture",
                  "--worker-instance-id", "mc-photo-capture",
                  "--worker-template", str(tmp_path / "template"),
                  "--capture-mod", str(tmp_path / "mod.jar"),
                  "--capture-mod-sha256", "0" * 64])


def test_cli_dry_run_with_worker_args_mutates_nothing(tmp_path, capsys):
    game = _game(tmp_path)
    dataset = tmp_path / "ds"
    spec = _spec(game, dataset, [
        {"task_id": "a", "save": "Alpha", "declared_split": "train",
         "region": _region("house")}])
    spec_path = _write_spec(tmp_path, spec)
    materialize_queue(plan_batch(spec))
    template = _template(tmp_path)
    template_state = {path: path.read_bytes() for path in template.rglob("*") if path.is_file()}
    worker_root = tmp_path / "workers"
    cli = _cli_module()
    assert cli.main(["--spec", str(spec_path),
                     "--prism-executable", "C:/prism/prismlauncher.exe",
                     "--worker-root", str(worker_root),
                     "--worker-dir", "mc-photo-capture",
                     "--worker-instance-id", "mc-photo-capture",
                     "--worker-template", str(template)]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "mc-photo-capture" in out
    assert not worker_root.exists()
    assert {path: path.read_bytes() for path in template.rglob("*")
            if path.is_file()} == template_state


# --- single-flight lock -----------------------------------------------------


def test_run_refuses_when_the_queue_lock_is_held(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    runner = _FakeRunner(prepare_hook=factory.prepare)
    executor = _executor(queue, runner, _FakeLauncher())
    executor.lock.acquire(manifest_id="someone-else")
    try:
        with pytest.raises(ExecutorError, match="already owned"):
            executor.run()
    finally:
        executor.lock.release()

    assert runner.calls == []
    assert [scene["status"] for scene in load_queue(queue)["scenes"]] == [
        "pending", "pending"]


def test_run_reclaims_a_lock_left_by_a_dead_local_process(tmp_path, monkeypatch):
    # Keep this hermetic on Windows: liveness is the injected boundary, so the
    # test never has to launch tasklist.exe to decide that a sentinel pid is dead.
    monkeypatch.setattr(executor_module, "_pid_alive", lambda pid: False)
    queue, scene_ids = _two_scene_queue(tmp_path)
    (queue / LOCK_NAME).write_text(json.dumps({
        "schema_version": 1, "pid": 2 ** 31 - 1, "host": socket.gethostname(),
        "created": "2000-01-01T00:00:00Z", "manifest_id": "x", "owner": "dead"}),
        encoding="utf-8")
    factory = _FakeFactory(queue)
    report = _executor(queue, _FakeRunner(prepare_hook=factory.prepare),
                       _FakeLauncher()).run()
    assert report.completed == tuple(scene_ids)
    assert not (queue / LOCK_NAME).exists()


def test_run_refuses_a_foreign_host_lock_unless_forced(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    foreign = {"schema_version": 1, "pid": 4242, "host": "some-other-host",
               "created": "2000-01-01T00:00:00Z", "manifest_id": "x", "owner": "remote"}
    (queue / LOCK_NAME).write_text(json.dumps(foreign), encoding="utf-8")
    factory = _FakeFactory(queue)
    with pytest.raises(ExecutorError, match="already owned"):
        _executor(queue, _FakeRunner(prepare_hook=factory.prepare),
                  _FakeLauncher()).run()
    assert (queue / LOCK_NAME).exists()

    forced = _executor(queue, _FakeRunner(prepare_hook=factory.prepare),
                       _FakeLauncher(), force_unlock=True).run()
    assert forced.completed == tuple(scene_ids)
    assert not (queue / LOCK_NAME).exists()


def test_dry_run_takes_no_lock_and_writes_no_ledger(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    _executor(queue, _FakeRunner(), _FakeLauncher(), dry_run=True).run()
    assert not (queue / LOCK_NAME).exists()
    assert not (queue / LEDGER_NAME).exists()


def test_run_writes_a_durable_ledger_and_releases_the_lock(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    _executor(queue, _FakeRunner(prepare_hook=factory.prepare), _FakeLauncher()).run()
    assert not (queue / LOCK_NAME).exists()
    records = RunLedger(queue).records()
    events = [record["event"] for record in records]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert events.count("scene_start") == 2 and events.count("scene_end") == 2
    assert [record["stage"] for record in records
            if record["event"] == "command"] == [
        "prepare", "finalize", "prepare", "finalize"]
    assert records[-1]["complete"] == list(scene_ids)


# --- interrupt safety -------------------------------------------------------


def test_interrupt_during_prepare_marks_failed_and_skips_the_rest(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    executor = _executor(queue, _FakeRunner(), _FakeLauncher())

    def prepare_then_interrupt(argv):
        factory.prepare(argv)
        executor.request_interrupt("operator stop")

    executor.runner.prepare_hook = prepare_then_interrupt
    report = executor.run()

    assert report.interrupted is True
    assert report.failed == (scene_ids[0],)
    assert report.outcomes[0].stage == "interrupted"
    assert report.skipped == (scene_ids[1],)
    payload = load_queue(queue)
    assert payload["scenes"][0]["status"] == "failed"
    assert payload["scenes"][1]["status"] == "pending"
    assert not (queue / LOCK_NAME).exists()


def test_abrupt_runner_failure_never_leaves_a_scene_running(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)

    class _ExplodingRunner(_FakeRunner):
        def run(self, argv, *, cwd, timeout):
            raise OSError("runner died")

    executor = _executor(queue, _ExplodingRunner(), _FakeLauncher())
    with pytest.raises(OSError):
        executor.run()
    assert load_queue(queue)["scenes"][0]["status"] == "failed"
    assert not (queue / LOCK_NAME).exists()


def test_subprocess_runner_interrupts_and_kills_the_child(tmp_path):
    interrupt = threading.Event()
    runner = SubprocessRunner(interrupt=interrupt)

    def trip():
        time.sleep(0.4)
        interrupt.set()

    thread = threading.Thread(target=trip)
    thread.start()
    try:
        started = time.monotonic()
        result = runner.run([sys.executable, "-c", "import time; time.sleep(60)"],
                            cwd=tmp_path, timeout=60)
    finally:
        thread.join()
    assert result.interrupted is True
    assert result.ok is False
    assert time.monotonic() - started < 30


def test_stop_process_tree_confirms_a_tree_kill_without_a_real_child():
    """The tree kill is judged by the injected taskkill result, not a real process."""
    process = _FakeProcess(returncode=None)
    seen: list[list[str]] = []

    def taskkill(argv, **kwargs):
        seen.append([str(item) for item in argv])
        process.terminate()
        return types.SimpleNamespace(returncode=0)

    assert _stop_process_tree(process, grace=5, taskkill=taskkill) is True
    assert process.poll() is not None
    assert seen[0][:2] == ["taskkill", "/PID"]

    denied = _FakeProcess(returncode=None)
    assert _stop_process_tree(denied, grace=5,
                              taskkill=lambda argv, **kw: types.SimpleNamespace(
                                  returncode=1)) is False
    assert denied.poll() is not None


# --- live-safety preflight --------------------------------------------------


def test_missing_capture_options_refuses_before_any_launch(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    (Path(scene["staging_game_dir"]) / "options.txt").unlink()
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="binding capture options"):
        executor._capture_request(scene, dataset)


def test_wrong_raw_mouse_option_refuses_before_any_launch(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    (Path(scene["staging_game_dir"]) / "options.txt").write_text(
        "pauseOnLostFocus:false\nrawMouseInput:true\nsoundCategory_master:0.0\n",
        encoding="utf-8")
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="binding capture options"):
        executor._capture_request(scene, dataset)


def test_missing_mod_jar_refuses_before_any_launch(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    for jar in (Path(scene["staging_game_dir"]) / "mods").glob("*.jar"):
        jar.unlink()
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="capture mod"):
        executor._capture_request(scene, dataset)


def test_missing_in_world_staging_marker_refuses_before_any_launch(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    marker = world_marker_path(Path(scene["staging_game_dir"]) / "saves" / scene["save"])
    marker.unlink()
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="in-world staging marker"):
        executor._capture_request(scene, dataset)


def test_in_world_marker_token_mismatch_refuses(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    marker = world_marker_path(Path(scene["staging_game_dir"]) / "saves" / scene["save"])
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["token"] = "b" * 64
    marker.write_text(json.dumps(payload), encoding="utf-8")
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="staging marker token"):
        executor._capture_request(scene, dataset)


def test_runtime_containment_refuses_paths_inside_the_live_game_dir(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    executor = _executor(queue, _FakeRunner(), _FakeLauncher())
    game = tmp_path / "game"
    payload = {"dataset": str(game / "ds"),
               "staging_game_dir": str(game / "ds" / "staging-game"),
               "game_dir": str(game), "scenes": []}
    with pytest.raises(ExecutorError, match="live game directory"):
        executor._check_runtime_containment(payload)


# --- sentinel interpretation contract ---------------------------------------


def test_interpret_run_status_checks_token_and_versions():
    good = {"schema_version": SCHEMA_VERSION, "mod_contract_version": MOD_CONTRACT_VERSION,
            "plan_id": "p", "staging_token": TOKEN, "run_failed": False,
            "ground_truth_exported": True}
    assert interpret_run_status(
        good, plan_id="p", staging_token=TOKEN, schema_version=SCHEMA_VERSION,
        mod_contract_version=MOD_CONTRACT_VERSION).completed is True
    assert interpret_run_status(
        {**good, "staging_token": "b" * 64}, plan_id="p",
        staging_token=TOKEN).completed is False
    assert interpret_run_status(
        {**good, "schema_version": 9}, schema_version=SCHEMA_VERSION).completed is False
    assert interpret_run_status(
        {**good, "mod_contract_version": 9},
        mod_contract_version=MOD_CONTRACT_VERSION).completed is False


# --- exact capture options (DS-31S item 5) ----------------------------------


def test_capture_options_are_exactly_the_three_accepted_values():
    assert CAPTURE_OPTIONS == {
        "pauseOnLostFocus": "false",
        "rawMouseInput": "false",
        "soundCategory_master": "0.0",
    }


@pytest.mark.parametrize("options", [
    "pauseOnLostFocus:true\nrawMouseInput:false\nsoundCategory_master:0.0\n",
    "pauseOnLostFocus:false\nrawMouseInput:false\nsoundCategory_master:1.0\n",
    "pauseOnLostFocus:false\nrawMouseInput:false\n",
])
def test_options_must_match_exactly_before_any_launch(tmp_path, options):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    (Path(scene["staging_game_dir"]) / "options.txt").write_text(options, encoding="utf-8")
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    with pytest.raises(DatasetSchemaError, match="binding capture options"):
        executor._capture_request(scene, dataset)


def test_correct_options_are_accepted_before_launch(tmp_path):
    queue, scene, factory, dataset = _prepared_scene(tmp_path)
    (Path(scene["staging_game_dir"]) / "options.txt").write_text(
        "pauseOnLostFocus:false\nrawMouseInput:false\nsoundCategory_master:0.0\n",
        encoding="utf-8")
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=_FakeLauncher())
    request = executor._capture_request(scene, dataset)
    assert request.mod_jar is not None


# --- canonical Windows path key (DS-31U) ------------------------------------

#: The junction target :class:`WindowsLinkInspector` returned, verbatim, for the
#: first real pilot (queue `pilot-batch-train-station-2026-09-12`, manifest
#: `25b8c792...`): the worker's ``minecraft`` junction carried the
#: extended-length ``\\?\`` spelling of the requested staging directory, so the
#: plain string comparison failed at the worker stage before Prism could start.
_OBSERVED_STAGING = (
    "C:\\Users\\Likea\\Documents\\mc photogrammetry project\\exports\\datasets"
    "\\pilot-batch-train-station-2026-09-12\\staging-game"
    "\\train-station-spawn-208-96-orbit-a")
_OBSERVED_JUNCTION_TARGET = "\\\\?\\" + _OBSERVED_STAGING


def test_path_key_treats_the_extended_drive_prefix_as_the_same_path():
    plain = "C:\\datasets\\staging\\scene-a"
    assert executor_module._path_key(plain) == \
        executor_module._path_key("\\\\?\\" + plain)


def test_path_key_treats_the_extended_unc_prefix_as_the_same_path():
    plain = "\\\\server\\share\\datasets\\scene-a"
    extended = "\\\\?\\UNC\\server\\share\\datasets\\scene-a"
    assert executor_module._path_key(plain) == executor_module._path_key(extended)


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_path_key_is_case_insensitive():
    assert executor_module._path_key("C:\\Data\\Scene-A") == \
        executor_module._path_key("\\\\?\\c:\\data\\scene-a")


@pytest.mark.skipif(os.name != "nt", reason="Windows path containment semantics")
def test_path_key_does_not_widen_containment():
    # A sibling whose name merely starts with the root name is not contained,
    # with or without the extended prefix on either side.
    assert not executor_module.path_contains("C:\\a\\b", "C:\\a\\bc")
    assert not executor_module.path_contains("\\\\?\\C:\\a\\b", "C:\\a\\bc")
    assert not executor_module.path_contains("C:\\a\\b", "\\\\?\\C:\\a\\bc")
    assert executor_module.path_contains("C:\\a\\b", "C:\\a\\b\\c")
    assert executor_module.path_contains("\\\\?\\C:\\a\\b", "C:\\a\\b\\c")
    assert not executor_module.path_contains(
        "\\\\server\\share\\a", "\\\\server\\share\\ab")
    assert executor_module.path_contains(
        "\\\\?\\UNC\\server\\share\\a", "\\\\server\\share\\a\\b")


def test_path_key_rejects_genuinely_different_paths():
    assert executor_module._path_key("C:\\a\\b") != executor_module._path_key(
        "C:\\a\\c")
    assert not executor_module.path_contains("C:\\a\\b", "C:\\a\\c")


def test_path_key_matches_the_exact_live_junction_target_observed():
    """The first pilot's LinkState target, verbatim, now matches its request."""
    state = LinkState("junction", Path(_OBSERVED_JUNCTION_TARGET))
    assert state.target is not None
    assert executor_module._path_key(state.target) == \
        executor_module._path_key(Path(_OBSERVED_STAGING))
    assert executor_module.path_contains(Path(_OBSERVED_STAGING).parent, state.target)
    assert executor_module._path_key(state.target) != \
        executor_module._path_key(Path(_OBSERVED_STAGING + "-b"))


# --- Windows junction adapter (DS-31S item 1) -------------------------------


def _junction(tmp_path, *, worker_root=None, target_root=None, states=None, handler=None):
    worker_root = Path(worker_root or (tmp_path / "workers"))
    target_root = Path(target_root or (tmp_path / "ds" / "staging-game"))
    link = worker_root / "mc-photo-capture" / "minecraft"
    inspector = _FakeInspector(states)
    if handler is None:
        def handler(argv):
            inspector.set(Path(argv[6]), LinkState("junction", Path(argv[7])))
            return CommandResult(tuple(argv), 0, "ok", "", False, 0.01)
    runner = _ScriptRunner(handler)
    adapter = WindowsJunctionAdapter(worker_root=worker_root, target_root=target_root,
                                     inspector=inspector, runner=runner,
                                     script_dir=tmp_path / "ps")
    return adapter, link, inspector, runner


def _staging_dir(target_root: Path, name: str = "alpha-house") -> Path:
    target = target_root / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def test_junction_adapter_creates_with_one_positional_powershell_script(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    adapter, link, inspector, runner = _junction(tmp_path, target_root=target_root)
    target = _staging_dir(target_root)
    assert adapter.ensure(link, target) == target
    assert len(runner.calls) == 1
    argv = runner.calls[0]["argv"]
    assert argv[0] == "powershell.exe"
    assert argv[1:4] == ["-NoProfile", "-ExecutionPolicy", "Bypass"]
    assert argv[4] == "-File"
    assert argv[6] == str(link) and argv[7] == str(target)
    assert argv[8] == str(adapter.worker_root)
    assert argv[9] == str(adapter.target_root)
    script = Path(argv[5]).read_text(encoding="utf-8")
    assert str(link) not in script and str(target) not in script
    assert str(adapter.worker_root) not in script
    assert str(adapter.target_root) not in script
    assert script == JUNCTION_ENSURE_SCRIPT
    assert "param(" in script
    assert "[Parameter(Mandatory = $true)][string]$WorkerRoot" in script
    assert "[Parameter(Mandatory = $true)][string]$TargetRoot" in script
    assert "Get-Item -LiteralPath $Minecraft" in script
    assert "New-Item -ItemType Junction -Path $Minecraft -Target $Target" in script
    assert "$created.Target" in script
    assert "created minecraft junction target does not equal the requested target" in script


def test_junction_script_reproves_containment_without_destructive_tokens():
    """The script, not just Python, must canonicalize and contained-check."""
    script = JUNCTION_ENSURE_SCRIPT
    assert "Get-NormalizedPath" in script
    assert "Test-PathUnderRoot" in script
    assert "Get-NormalizedPath $Minecraft" in script
    assert "Get-NormalizedPath $Target" in script
    assert "Get-NormalizedPath $WorkerRoot" in script
    assert "Get-NormalizedPath $TargetRoot" in script
    assert "worker junction is not strictly below the worker root" in script
    assert "junction target is outside the staging root" in script
    assert "$WorkerRoot -Candidate $Minecraft -Strict" in script
    assert "existing junction points outside the staging root" in script
    assert "created minecraft junction target does not equal the requested target" in script
    for token in ("Remove-Item", "Directory]::Delete", "Directory]::Remove",
                  "RemoveDirectory", "os.rmdir", "rmdir", "-Recurse"):
        assert token.lower() not in script.lower()


def test_junction_adapter_refuses_an_uncontained_existing_target_without_shell(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root)
    outside = tmp_path / "other-root" / "old-scene"
    link = tmp_path / "workers" / "mc-photo-capture" / "minecraft"
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        states={link: LinkState("junction", outside)})
    with pytest.raises(JunctionError, match="outside the queue staging root"):
        adapter.ensure(link, target)
    assert runner.calls == []


def test_junction_adapter_refuses_a_different_contained_existing_target(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    old = _staging_dir(target_root, "old-scene")
    new = _staging_dir(target_root, "new-scene")
    link = tmp_path / "workers" / "mc-photo-capture" / "minecraft"
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        states={link: LinkState("junction", old)})
    with pytest.raises(JunctionError, match="distinct per-scene worker"):
        adapter.ensure(link, new)
    assert runner.calls == []


def test_junction_adapter_accepts_an_extended_prefix_created_target(tmp_path):
    r"""The exact first-pilot failure: the created junction reports ``\\?\C:\...``."""
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root)
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        handler=lambda argv: (
            inspector.set(
                Path(argv[6]),
                LinkState("junction", Path("\\\\?\\" + str(Path(argv[7]))))),
            CommandResult(tuple(argv), 0, "ok", "", False, 0.01))[1])
    resolved = adapter.ensure(link, target)
    assert executor_module._path_key(resolved) == executor_module._path_key(target)
    assert len(runner.calls) == 1


def test_junction_adapter_accepts_a_contained_extended_prefix_existing_junction(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root, "old-scene")
    link = tmp_path / "workers" / "mc-photo-capture" / "minecraft"
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        states={link: LinkState("junction", Path("\\\\?\\" + str(target)))})
    resolved = adapter.ensure(link, target)
    assert executor_module._path_key(resolved) == executor_module._path_key(target)
    assert runner.calls == []


def test_junction_adapter_still_refuses_an_uncontained_extended_prefix_junction(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root)
    link = tmp_path / "workers" / "mc-photo-capture" / "minecraft"
    outside = tmp_path / "other-root" / "scene"
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        states={link: LinkState("junction", Path("\\\\?\\" + str(outside)))})
    with pytest.raises(JunctionError, match="outside the queue staging root"):
        adapter.ensure(link, target)
    assert runner.calls == []


def test_junction_adapter_refuses_a_worker_link_outside_the_root(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    adapter, link, inspector, runner = _junction(tmp_path, target_root=target_root)
    target = _staging_dir(target_root)
    with pytest.raises(JunctionError, match="worker root"):
        adapter.ensure(tmp_path / "elsewhere" / "minecraft", target)
    assert runner.calls == []


def test_junction_adapter_refuses_a_target_outside_the_staging_root(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    adapter, link, inspector, runner = _junction(tmp_path, target_root=target_root)
    with pytest.raises(JunctionError, match="staging root"):
        adapter.ensure(link, tmp_path / "outside-scene")
    assert runner.calls == []


@pytest.mark.parametrize("state, needle", [
    (LinkState("real-dir"), "real-dir"),
    (LinkState("symlink", Path("C:/elsewhere")), "symlink"),
    (LinkState("other-link", Path("C:/elsewhere")), "other-link"),
])
def test_junction_adapter_refuses_a_real_or_foreign_existing_link(tmp_path, state, needle):
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root)
    link = tmp_path / "workers" / "mc-photo-capture" / "minecraft"
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root, states={link: state})
    with pytest.raises(JunctionError, match=needle):
        adapter.ensure(link, target)
    assert runner.calls == []


def test_junction_adapter_refuses_an_uncontained_existing_junction(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root)
    link = tmp_path / "workers" / "mc-photo-capture" / "minecraft"
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        states={link: LinkState("junction", tmp_path / "other-root" / "scene")})
    with pytest.raises(JunctionError, match="outside the queue staging root"):
        adapter.ensure(link, target)
    assert runner.calls == []


def test_junction_adapter_refuses_when_powershell_fails(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root)
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        handler=lambda argv: CommandResult(tuple(argv), 1, "", "denied", False, 0.01))
    with pytest.raises(JunctionError, match="failed"):
        adapter.ensure(link, target)


def test_junction_adapter_verifies_the_resolved_target(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root)
    other = _staging_dir(target_root, "other")
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        handler=lambda argv: (inspector.set(Path(argv[6]), LinkState("junction", other)),
                              CommandResult(tuple(argv), 0, "ok", "", False, 0.01))[1])
    with pytest.raises(JunctionError, match="resolved to"):
        adapter.ensure(link, target)


def test_junction_adapter_refuses_when_no_junction_was_created(tmp_path):
    target_root = tmp_path / "ds" / "staging-game"
    target = _staging_dir(target_root)
    adapter, link, inspector, runner = _junction(
        tmp_path, target_root=target_root,
        handler=lambda argv: CommandResult(tuple(argv), 0, "ok", "", False, 0.01))
    with pytest.raises(JunctionError, match="did not produce a directory junction"):
        adapter.ensure(link, target)


# --- disposable Prism worker (DS-31S item 2) --------------------------------


def _worker(tmp_path, template, **over):
    junction = over.pop("junction", _FakeJunction())
    worker = DisposablePrismWorker(
        worker_root=over.pop("worker_root", tmp_path / "workers"),
        worker_dir="mc-photo-capture", instance_id="mc-photo-capture",
        template_dir=template, executable=Path("C:/prism/prismlauncher.exe"),
        prism_root=Path("B:/prismRoot"), junction=junction, **over)
    return worker, junction


def test_scene_worker_identity_is_stable_unique_and_bounded():
    first = executor_module.scene_worker_identity(
        "scene/a", scope="C:/capture/dataset-a/staging-game",
        worker_dir_prefix="capture-prefix", instance_id_prefix="capture-prefix")
    same = executor_module.scene_worker_identity(
        "scene/a", scope="C:/capture/dataset-a/staging-game",
        worker_dir_prefix="capture-prefix", instance_id_prefix="capture-prefix")
    colliding_slug = executor_module.scene_worker_identity(
        "scene?a", scope="C:/capture/dataset-a/staging-game",
        worker_dir_prefix="capture-prefix", instance_id_prefix="capture-prefix")
    bounded = executor_module.scene_worker_identity(
        "x" * 400, scope="C:/capture/dataset-a/staging-game",
        worker_dir_prefix="p" * 400, instance_id_prefix="p" * 400)

    assert first == same
    assert first != colliding_slug
    assert first != executor_module.scene_worker_identity(
        "scene/a", scope="C:/capture/dataset-b/staging-game",
        worker_dir_prefix="capture-prefix", instance_id_prefix="capture-prefix")
    assert all(len(value) <= 62 for value in first)
    assert all(len(value) <= 62 for value in bounded)
    assert all("/" not in value and "\\" not in value for value in (*first, *bounded))


def test_scene_worker_identity_rejects_empty_scene_id():
    with pytest.raises(WorkerError, match="scene id must not be empty"):
        executor_module.scene_worker_identity(
            "", scope="C:/capture/dataset/staging-game",
            worker_dir_prefix="worker", instance_id_prefix="worker")


def test_scene_worker_identity_requires_matching_prism_folder_prefixes():
    with pytest.raises(WorkerError, match="must normalize to the same name"):
        executor_module.scene_worker_identity(
            "scene-a", scope="C:/capture/dataset/staging-game",
            worker_dir_prefix="folder", instance_id_prefix="instance")


def _files(directory: Path) -> dict:
    return {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def _remove_worker_files(worker_dir: Path) -> None:
    """Delete exactly the known worker files in a test, then rmdir (never recurse)."""
    for name in ("instance.cfg", "mmc-pack.json", WORKER_MARKER_NAME):
        for candidate in (worker_dir / name, worker_dir / f"{name}.tmp"):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
    worker_dir.rmdir()


def test_worker_initialization_copies_only_template_instance_files(tmp_path):
    template = _template(tmp_path)
    before = _files(template)
    worker, _ = _worker(tmp_path, template)
    marker = worker.ensure_ready()
    worker_dir = tmp_path / "workers" / "mc-photo-capture"
    assert sorted(path.name for path in worker_dir.iterdir()) == sorted(
        ["instance.cfg", "mmc-pack.json", WORKER_MARKER_NAME])
    assert not (worker_dir / "minecraft").exists()
    assert not (worker_dir / "extra.txt").exists()
    cfg = (worker_dir / "instance.cfg").read_text(encoding="utf-8")
    assert "name=mc-photo-capture" in cfg
    assert "InstanceType=OneSix" in cfg
    match = re.search(r"uuid=([0-9a-f]{32})", cfg)
    assert match is not None
    assert marker["uuid"] == match.group(1)
    stored = json.loads((worker_dir / WORKER_MARKER_NAME).read_text(encoding="utf-8"))
    assert stored == marker
    assert stored["instance_id"] == "mc-photo-capture"
    assert _files(template) == before
    assert worker.prism_instance().instance_id == "mc-photo-capture"


def test_worker_initialization_assigns_a_fresh_uuid_each_time(tmp_path):
    template = _template(tmp_path)
    worker_dir = tmp_path / "workers" / "mc-photo-capture"
    first, _ = _worker(tmp_path, template)
    first.ensure_ready()
    uuid_one = json.loads((worker_dir / WORKER_MARKER_NAME).read_text())["uuid"]
    _remove_worker_files(worker_dir)
    second, _ = _worker(tmp_path, template)
    second.ensure_ready()
    uuid_two = json.loads((worker_dir / WORKER_MARKER_NAME).read_text())["uuid"]
    assert uuid_one != uuid_two


def test_worker_failed_initialization_removes_only_its_own_known_files(tmp_path):
    """A mid-copy failure cleans the two copies and the directory, not a tree."""
    template = _template(tmp_path)
    (template / "mmc-pack.json").unlink()
    (template / "mmc-pack.json").mkdir()
    worker, _ = _worker(tmp_path, template)
    worker_dir = tmp_path / "workers" / "mc-photo-capture"
    assert not worker_dir.exists()
    with pytest.raises(WorkerError, match="not a regular file"):
        worker.ensure_ready()
    assert not worker_dir.exists()


def test_worker_failed_initialization_removes_atomic_temporaries(tmp_path, monkeypatch):
    template = _template(tmp_path)
    worker, _ = _worker(tmp_path, template)
    worker_dir = tmp_path / "workers" / "mc-photo-capture"
    real_atomic_text = executor_module._atomic_write_text

    def failing_atomic_text(path, text):
        real_atomic_text(path.with_name(f"{path.name}.tmp"), text)
        raise OSError("simulated initialization failure")

    monkeypatch.setattr(executor_module, "_atomic_write_text", failing_atomic_text)
    with pytest.raises(OSError, match="simulated initialization failure"):
        worker.ensure_ready()
    assert not worker_dir.exists()


def test_worker_failed_initialization_never_recursively_deletes(tmp_path, monkeypatch):
    """An unexpected entry must survive; cleanup uses unlink plus non-recursive rmdir."""
    template = _template(tmp_path)
    worker, _ = _worker(tmp_path, template)
    worker_dir = tmp_path / "workers" / "mc-photo-capture"

    def failing_atomic_text(path, text):
        (worker_dir / "unexpected.bin").write_bytes(b"keep me")
        raise OSError("simulated initialization failure")

    monkeypatch.setattr(executor_module, "_atomic_write_text", failing_atomic_text)
    with pytest.raises(OSError, match="simulated initialization failure"):
        worker.ensure_ready()
    assert worker_dir.is_dir()
    assert (worker_dir / "unexpected.bin").read_bytes() == b"keep me"
    assert not (worker_dir / "instance.cfg").exists()
    assert not (worker_dir / "mmc-pack.json").exists()
    assert not (worker_dir / WORKER_MARKER_NAME).exists()


def test_discard_failed_initialize_refuses_a_directory_with_extra_entries(tmp_path):
    """Cleanup refuses a worker path that holds anything it did not create."""
    template = _template(tmp_path)
    worker, _ = _worker(tmp_path, template)
    worker_dir = tmp_path / "workers" / "mc-photo-capture"
    worker_dir.mkdir(parents=True)
    sentinel = worker_dir / "sentinel.bin"
    sentinel.write_bytes(b"not ours")
    worker._discard_failed_initialize(created=True)
    assert sentinel.read_bytes() == b"not ours"
    assert worker_dir.is_dir()


def test_ds31_sources_have_no_recursive_delete():
    """Static: DS-31 files must not contain a recursive remove anywhere."""
    recursive_delete = "rm" + "tree"
    recurse = "-" + "Recurse"
    paths = (
        ROOT / "src" / "mc_photogrammetry" / "dataset" / "executor.py",
        ROOT / "scripts" / "run_capture_batch.py",
        Path(__file__),
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert recursive_delete not in source, path
    assert recurse not in JUNCTION_ENSURE_SCRIPT


def test_worker_existing_marker_is_accepted_without_a_rewrite(tmp_path):
    template = _template(tmp_path)
    worker, _ = _worker(tmp_path, template)
    worker.ensure_ready()
    worker_dir = tmp_path / "workers" / "mc-photo-capture"
    marker_before = (worker_dir / WORKER_MARKER_NAME).read_bytes()
    cfg_before = (worker_dir / "instance.cfg").read_bytes()
    worker.ensure_ready()
    assert (worker_dir / WORKER_MARKER_NAME).read_bytes() == marker_before
    assert (worker_dir / "instance.cfg").read_bytes() == cfg_before


def test_worker_marker_accepts_extended_prefix_paths(tmp_path):
    r"""The marker may spell worker/template paths in the ``\\?\`` form."""
    template = _template(tmp_path)
    worker, _ = _worker(tmp_path, template)
    worker.ensure_ready()
    marker_path = tmp_path / "workers" / "mc-photo-capture" / WORKER_MARKER_NAME
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["worker_dir"] = "\\\\?\\" + payload["worker_dir"]
    payload["template_dir"] = "\\\\?\\" + payload["template_dir"]
    marker_path.write_text(json.dumps(payload), encoding="utf-8")
    worker.ensure_ready()
    assert json.loads(marker_path.read_text(encoding="utf-8"))[
        "worker_dir"].startswith("\\\\?\\")


def test_worker_refuses_a_template_equal_to_the_worker(tmp_path):
    with pytest.raises(WorkerError, match="same path"):
        DisposablePrismWorker(
            worker_root=tmp_path / "workers",
            worker_dir=tmp_path / "workers" / "template",
            instance_id="template", template_dir=tmp_path / "workers" / "template",
            junction=_FakeJunction())


def test_worker_refuses_an_out_of_root_directory(tmp_path):
    with pytest.raises(WorkerError, match="not under the worker root"):
        DisposablePrismWorker(
            worker_root=tmp_path / "workers",
            worker_dir=tmp_path / "elsewhere" / "mc-photo-capture",
            instance_id="mc-photo-capture", template_dir=_template(tmp_path),
            junction=_FakeJunction())


def test_worker_refuses_an_instance_id_that_is_not_the_folder_name(tmp_path):
    with pytest.raises(WorkerError, match="must equal the worker directory name"):
        DisposablePrismWorker(
            worker_root=tmp_path / "workers", worker_dir="mc-photo-capture",
            instance_id="something-else", template_dir=_template(tmp_path),
            junction=_FakeJunction())


def test_worker_refuses_an_unmarked_existing_directory(tmp_path):
    (tmp_path / "workers" / "mc-photo-capture").mkdir(parents=True)
    worker, _ = _worker(tmp_path, _template(tmp_path))
    with pytest.raises(WorkerError, match="has no"):
        worker.ensure_ready()


def test_worker_refuses_a_mismatched_marker(tmp_path):
    template = _template(tmp_path)
    worker, _ = _worker(tmp_path, template)
    worker.ensure_ready()
    marker_path = tmp_path / "workers" / "mc-photo-capture" / WORKER_MARKER_NAME
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["instance_id"] = "someone-else"
    marker_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkerError, match="names instance"):
        worker.ensure_ready()


def test_worker_target_is_immutable_after_first_ensure(tmp_path):
    template = _template(tmp_path)
    junction = _FakeJunction()
    worker, _ = _worker(tmp_path, template, junction=junction)
    worker.ensure_ready()
    worker.ensure_ready()
    one = _staging_dir(tmp_path / "ds", "one")
    two = _staging_dir(tmp_path / "ds", "two")
    worker.ensure_target(one)
    worker.ensure_target(one)
    with pytest.raises(JunctionError, match="distinct per-scene worker"):
        worker.ensure_target(two)
    assert junction.calls == [(worker.minecraft_path, one)]


def test_worker_refuses_to_ensure_target_without_a_junction_adapter(tmp_path):
    worker = DisposablePrismWorker(
        worker_root=tmp_path / "workers", worker_dir="mc-photo-capture",
        instance_id="mc-photo-capture", template_dir=_template(tmp_path))
    with pytest.raises(WorkerError, match="no junction adapter"):
        worker.ensure_target(_staging_dir(tmp_path / "ds"))


# --- capture mod provisioner (DS-31S item 4) --------------------------------


def _mod_source(tmp_path, data=b"the-capture-mod", name="mc-photo-capture-0.1.0.jar"):
    source = tmp_path / "build" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    return source, hashlib.sha256(data).hexdigest()


def test_provisioner_verifies_and_atomically_copies_the_mod(tmp_path):
    source, digest = _mod_source(tmp_path)
    staging = tmp_path / "staging"
    destination = CaptureModProvisioner(source, digest).provision(staging)
    assert destination == staging / "mods" / source.name
    assert destination.read_bytes() == b"the-capture-mod"
    assert sha256_file(destination) == digest
    assert sorted(path.name for path in (staging / "mods").iterdir()) == [source.name]


def test_provisioner_refuses_a_missing_source(tmp_path):
    _, digest = _mod_source(tmp_path)
    with pytest.raises(ProvisionError, match="missing"):
        CaptureModProvisioner(tmp_path / "absent.jar", digest).provision(tmp_path / "staging")


def test_provisioner_refuses_a_hash_mismatch_before_copying(tmp_path):
    source, _ = _mod_source(tmp_path)
    with pytest.raises(ProvisionError, match="pinned"):
        CaptureModProvisioner(source, "0" * 64).provision(tmp_path / "staging")
    assert not (tmp_path / "staging").exists()


def test_provisioner_refuses_a_bad_pin(tmp_path):
    source, _ = _mod_source(tmp_path)
    for pin in ("", "abc", "z" * 64, "0" * 63):
        with pytest.raises(ProvisionError, match="64 lowercase"):
            CaptureModProvisioner(source, pin)


def test_provisioner_refuses_a_source_outside_the_capture_glob(tmp_path):
    source, digest = _mod_source(tmp_path, name="random.jar")
    with pytest.raises(ProvisionError, match="does not match"):
        CaptureModProvisioner(source, digest).provision(tmp_path / "staging")


def test_provisioner_refuses_multiple_capture_jars(tmp_path):
    source, digest = _mod_source(tmp_path)
    mods = tmp_path / "staging" / "mods"
    mods.mkdir(parents=True)
    (mods / "mc-photo-capture-9.9.9.jar").write_bytes(b"other")
    with pytest.raises(ProvisionError, match="other capture jars"):
        CaptureModProvisioner(source, digest).provision(tmp_path / "staging")


# --- hidden-desktop controller (DS-31S item 3) ------------------------------

FORBIDDEN_DESKTOP_TOKENS = (
    "Switch-Desktop", "Move-ActiveWindow", "Get-ActiveWindowHandle",
    "SetForegroundWindow", "Activate", "Focus", "SendKeys",
    "mouse_event", "keybd_event", "SetCursorPos", "GetCursorPos",
)


def _desktop_handler(*, count="2", preflight_index="0", move_before=None,
                     move_index="0", move_error=None, move_stderr=""):
    def handler(argv):
        if Path(argv[5]).name == "desktop-preflight.ps1":
            stdout = f"count={count} index={preflight_index}\n"
        else:
            if move_error is not None:
                return CommandResult(tuple(argv), move_error, "", move_stderr, False, 0.01)
            before = preflight_index if move_before is None else move_before
            stdout = f"before={before} index={move_index} moved=1 stale=0\n"
        return CommandResult(tuple(argv), 0, stdout, "", False, 0.01)
    return handler


def _desktop(tmp_path, handler):
    runner = _ScriptRunner(handler)
    return WindowsDesktop2Controller(runner=runner, script_dir=tmp_path / "ps"), runner


def test_desktop_controller_preflights_and_moves_only_the_supplied_process_tree(tmp_path):
    controller, runner = _desktop(tmp_path, _desktop_handler())
    assert controller.preflight() == DesktopPreflight(2, 0)
    controller.move_windows(4242)
    assert len(runner.calls) == 2
    for call in runner.calls:
        assert call["argv"][0] == "powershell.exe"
        assert call["argv"][1:4] == ["-NoProfile", "-ExecutionPolicy", "Bypass"]
        assert call["argv"][4] == "-File"
    preflight_script = Path(runner.calls[0]["argv"][5]).read_text(encoding="utf-8")
    assert preflight_script == DESKTOP_PREFLIGHT_SCRIPT
    assert "Import-Module VirtualDesktop" in preflight_script
    assert "Get-DesktopCount" in preflight_script
    assert "Get-DesktopIndex (Get-CurrentDesktop)" in preflight_script
    assert "if ($index -eq 1)" in preflight_script
    assert "target capture desktop" in preflight_script
    move_script = Path(runner.calls[1]["argv"][5]).read_text(encoding="utf-8")
    assert move_script == DESKTOP_MOVE_SCRIPT
    assert runner.calls[1]["argv"][6] == "4242"
    assert "Move-Window -Desktop (Get-Desktop 1) -Hwnd ([IntPtr]$window.Handle)" in move_script
    assert "MainWindowHandle" in move_script
    assert "Get-CimInstance -ClassName Win32_Process" in move_script
    assert "ParentProcessId" in move_script
    assert "$targetIds.Add($ProcessId)" in move_script
    assert "foreach ($candidateId in @($targetIds | Sort-Object))" in move_script
    assert "foreach ($window in $windows)" in move_script
    assert "Get-Process -Name" not in move_script
    assert "Get-Process |" not in move_script
    assert "$window.StartTicks" in move_script
    assert "Get-WindowState $window.ProcessId $window.StartTicks $window.Handle" in move_script


def test_desktop_controller_accepts_only_confirmed_elementnotfound_stale_race(tmp_path):
    controller, runner = _desktop(
        tmp_path, lambda argv: CommandResult(
            tuple(argv), 0,
            "count=2 index=0\n" if Path(argv[5]).name == "desktop-preflight.ps1"
            else "before=0 index=0 moved=0 stale=1\n",
            "", False, 0.01))
    controller.preflight()
    controller.move_windows(4242)
    assert runner.calls[1]["argv"][6] == "4242"
    script = Path(runner.calls[1]["argv"][5]).read_text(encoding="utf-8")
    assert '$hresult -eq "8002802B"' in script
    assert "$stateAfterError -ne \"current\"" in script
    assert "if ($hresult -eq \"8002802B\" -and $stateAfterError -ne \"current\")" in script


def test_desktop_scripts_never_invoke_forbidden_switch_or_input_calls(tmp_path):
    controller, runner = _desktop(tmp_path, _desktop_handler())
    controller.preflight()
    controller.move_windows(4242)
    texts = [
        DESKTOP_PREFLIGHT_SCRIPT,
        DESKTOP_MOVE_SCRIPT,
        Path(runner.calls[0]["argv"][5]).read_text(encoding="utf-8"),
        Path(runner.calls[1]["argv"][5]).read_text(encoding="utf-8"),
    ]
    for text in texts:
        lowered = text.lower()
        for token in FORBIDDEN_DESKTOP_TOKENS:
            assert token.lower() not in lowered, token
    for call in runner.calls:
        joined = " ".join(call["argv"]).lower()
        for token in FORBIDDEN_DESKTOP_TOKENS:
            assert token.lower() not in joined, token


def test_desktop_controller_requires_a_second_desktop(tmp_path):
    controller, _ = _desktop(tmp_path, _desktop_handler(count="1"))
    with pytest.raises(DesktopError, match="at least two virtual desktops"):
        controller.preflight()


def test_desktop_controller_refuses_when_the_operator_is_on_the_target_desktop(tmp_path):
    controller, _ = _desktop(tmp_path, _desktop_handler(preflight_index="1"))
    with pytest.raises(DesktopError, match="target capture desktop"):
        controller.preflight()
    assert controller.preflight_state is None
    assert TARGET_DESKTOP_INDEX == 1


def test_desktop_controller_fails_when_the_module_is_unavailable(tmp_path):
    controller, _ = _desktop(
        tmp_path, lambda argv: CommandResult(tuple(argv), 1, "", "no module", False, 0.01))
    with pytest.raises(DesktopError, match="preflight failed"):
        controller.preflight()


@pytest.mark.parametrize(
    "move_before,move_index",
    [("0", "1"), ("1", "1")],
    ids=["changed-during-move", "changed-since-preflight"])
def test_desktop_controller_verifies_the_current_desktop_did_not_change(
        tmp_path, move_before, move_index):
    controller, _ = _desktop(tmp_path, _desktop_handler(preflight_index="0",
                                                        move_before=move_before,
                                                        move_index=move_index))
    controller.preflight()
    with pytest.raises(DesktopError, match="current desktop changed"):
        controller.move_windows(4242)


def test_desktop_controller_fails_closed_on_module_or_non_stale_move_error(tmp_path):
    controller, _ = _desktop(
        tmp_path, _desktop_handler(move_error=1,
                                   move_stderr="Move-Window failed HRESULT 0x80004005"))
    controller.preflight()
    with pytest.raises(DesktopError, match="moving the capture windows failed"):
        controller.move_windows(4242)


def test_desktop_controller_refuses_to_move_before_preflight(tmp_path):
    controller, runner = _desktop(tmp_path, _desktop_handler())
    with pytest.raises(DesktopError, match="preflight has not run"):
        controller.move_windows(4242)
    assert runner.calls == []


@pytest.mark.parametrize("process_id", [0, -1, True, "4242"])
def test_desktop_controller_requires_a_positive_integer_process_id(tmp_path, process_id):
    controller, runner = _desktop(tmp_path, _desktop_handler())
    controller.preflight()
    with pytest.raises(DesktopError, match="positive owning Prism process ID"):
        controller.move_windows(process_id)
    assert len(runner.calls) == 1


# --- launcher hidden-desktop integration ------------------------------------


def test_launcher_desktop_preflight_failure_never_starts_a_process(tmp_path):
    request, _ = _launch_request(tmp_path)
    started: list[bool] = []
    launcher = PrismCaptureLauncher(
        lambda scene_id: _instance(tmp_path, instance_id="inst"),
        poll_seconds=0.25,
        popener=lambda argv, cwd: started.append(True),
        stopper=lambda proc, *, grace: None,
        desktop=_FakeDesktop(fail_preflight=True))
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert outcome.stage == "desktop"
    assert started == []


def test_launcher_refuses_to_start_when_the_operator_is_on_the_target_desktop(tmp_path):
    request, _ = _launch_request(tmp_path)
    started: list[bool] = []
    controller = WindowsDesktop2Controller(
        runner=_ScriptRunner(_desktop_handler(preflight_index="1")),
        script_dir=tmp_path / "ps")
    launcher = PrismCaptureLauncher(
        lambda scene_id: _instance(tmp_path, instance_id="inst"),
        poll_seconds=0.25,
        popener=lambda argv, cwd: started.append(True),
        stopper=lambda proc, *, grace: True,
        desktop=controller)
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert outcome.stage == "desktop"
    assert "target capture desktop" in outcome.detail
    assert started == []


def test_launcher_moves_windows_after_popen_and_on_every_poll(tmp_path):
    request, output = _launch_request(tmp_path)
    writes = {"done": False}

    def write_status():
        if not writes["done"]:
            (output / "run_status.json").write_text(json.dumps(SENTINEL),
                                                    encoding="utf-8")
            writes["done"] = True

    desktop = _FakeDesktop()
    process = _FakeProcess(returncode=None)
    launcher = _launcher(tmp_path, process, status_writer=write_status, desktop=desktop,
                         stopper=lambda proc, *, grace: True)
    outcome = launcher.launch(request)
    assert outcome.completed is True
    assert desktop.preflights == 1
    assert desktop.moves >= 2
    assert desktop.process_ids == [process.pid] * desktop.moves


def test_launcher_fails_and_tears_down_when_a_poll_move_fails(tmp_path):
    request, _ = _launch_request(tmp_path)
    process = _FakeProcess(returncode=None)
    stopped: list[bool] = []
    launcher = _launcher(
        tmp_path, process, desktop=_FakeDesktop(fail_move_at=2),
        stopper=lambda proc, *, grace: (stopped.append(True), True)[1])
    outcome = launcher.launch(request)
    assert outcome.completed is False
    assert outcome.stage == "desktop"
    assert "current desktop changed" in outcome.detail
    assert stopped == [True]


# --- exact lifecycle order (DS-31S item 7) ----------------------------------


def test_executor_lifecycle_order_uses_a_distinct_worker_per_scene(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    order: list[str] = []
    factory = _FakeFactory(queue)
    base_prepare = factory.prepare

    def prepare(argv):
        order.append("prepare")
        base_prepare(argv)

    runner = _FakeRunner(prepare_hook=prepare)
    source, digest = _mod_source(tmp_path)
    inner_provisioner = CaptureModProvisioner(source, digest)

    class _RecordingProvisioner:
        source = inner_provisioner.source
        expected_sha256 = inner_provisioner.expected_sha256

        def provision(self, staging_game_dir):
            order.append("provision")
            return inner_provisioner.provision(staging_game_dir)

    template = _template(tmp_path)
    template_before = _files(template)
    worker_root = tmp_path / "workers"

    class _CheckingJunction:
        def ensure(self, link, target):
            order.append("ensure-target")
            assert (Path(link).parent / WORKER_MARKER_NAME).is_file()
            return Path(target)

    workers = {}

    def make_worker(scene_id):
        worker_dir, instance_id = executor_module.scene_worker_identity(
            scene_id, scope=tmp_path / "ds" / "staging-game",
            worker_dir_prefix="mc-photo-capture",
            instance_id_prefix="mc-photo-capture")
        worker = DisposablePrismWorker(
            worker_root=worker_root, worker_dir=worker_dir,
            instance_id=instance_id, template_dir=template,
            junction=_CheckingJunction())
        workers[scene_id] = worker
        return worker

    class _RecordingLauncher:
        def __init__(self):
            self.requests: list[CaptureRequest] = []

        def launch(self, request):
            order.append("launch")
            self.requests.append(request)
            return CaptureOutcome(True, "capture complete", {"fake": True})

    launcher = _RecordingLauncher()
    executor = BatchExecutor(queue, runner=runner, launcher=launcher,
                             worker_for_scene=make_worker,
                             provisioner=_RecordingProvisioner())
    report = executor.run()
    assert report.completed == tuple(scene_ids)
    assert order == ["prepare", "provision", "ensure-target", "launch"] * 2
    for scene in load_queue(queue)["scenes"]:
        staged = Path(scene["staging_game_dir"]) / "mods" / source.name
        assert staged.read_bytes() == source.read_bytes()
    assert _files(template) == template_before
    assert set(workers) == set(scene_ids)
    assert len({worker.worker_dir for worker in workers.values()}) == len(scene_ids)
    for worker in workers.values():
        marker = json.loads(worker.marker_path.read_text(encoding="utf-8"))
        assert marker["instance_id"] == worker.instance_id
        assert not worker.minecraft_path.exists()


def test_executor_rejects_a_fixed_worker_for_multiple_scenes_before_mutation(tmp_path):
    queue, scene_ids = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    runner = _FakeRunner(prepare_hook=factory.prepare)
    launcher = _FakeLauncher()
    worker = _FakeWorker()

    with pytest.raises(ExecutorError, match="one selected scene only"):
        BatchExecutor(queue, runner=runner, launcher=launcher, worker=worker).run()

    assert factory.prepared == []
    assert runner.scripts == []
    assert launcher.requests == []
    assert [scene["status"] for scene in load_queue(queue)["scenes"]] == [
        "pending" for _ in scene_ids]
    assert not (queue / LOCK_NAME).exists()


def test_executor_fails_the_scene_when_worker_initialization_is_refused(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue)
    runner = _FakeRunner(prepare_hook=factory.prepare)
    worker = DisposablePrismWorker(
        worker_root=tmp_path / "workers", worker_dir="mc-photo-capture",
        instance_id="mc-photo-capture", template_dir=tmp_path / "missing-template",
        junction=_FakeJunction())
    launcher = _FakeLauncher()
    report = BatchExecutor(queue, runner=runner, launcher=launcher,
                           worker=worker).run(["alpha-house"])
    assert report.failed == ("alpha-house",)
    assert report.outcomes[0].stage == "worker"
    assert launcher.requests == []
    assert not (tmp_path / "workers" / "mc-photo-capture").exists()


def test_executor_fails_the_scene_when_the_mod_pin_is_wrong(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    factory = _FakeFactory(queue, write_mod=False)
    runner = _FakeRunner(prepare_hook=factory.prepare)
    source, _ = _mod_source(tmp_path)
    launcher = _FakeLauncher()
    report = BatchExecutor(
        queue, runner=runner, launcher=launcher,
        provisioner=CaptureModProvisioner(source, "0" * 64)).run(["alpha-house"])
    assert report.failed == ("alpha-house",)
    assert report.outcomes[0].stage == "provision"
    assert launcher.requests == []


def test_dry_run_with_worker_and_provisioner_mutates_nothing(tmp_path):
    queue, _ = _two_scene_queue(tmp_path)
    source, digest = _mod_source(tmp_path)
    template = _template(tmp_path)
    template_before = _files(template)
    junction = _FakeJunction()
    worker = DisposablePrismWorker(
        worker_root=tmp_path / "workers", worker_dir="mc-photo-capture",
        instance_id="mc-photo-capture", template_dir=template, junction=junction)
    launcher = _FakeLauncher()
    executor = BatchExecutor(queue, runner=_FakeRunner(), launcher=launcher,
                             worker=worker,
                             provisioner=CaptureModProvisioner(source, digest),
                             dry_run=True)
    rows = executor.plan()
    assert rows[0]["stages"] == ["prepare --arm", "provision mod",
                                 "initialize/verify scene worker", "capture", "finalize"]
    report = executor.run()
    assert report.dry_run is True
    assert not (tmp_path / "workers").exists()
    assert junction.calls == []
    assert _files(template) == template_before
    for scene in load_queue(queue)["scenes"]:
        assert not (Path(scene["staging_game_dir"]) / "mods").exists()
