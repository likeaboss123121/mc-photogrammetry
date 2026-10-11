r"""Safe, sequential executor for a materialized capture queue.

DS-27's :mod:`mc_photogrammetry.dataset.batch` deliberately stops at
bookkeeping: it writes a content-addressed queue manifest with the exact
``prepare``/``finalize`` argv and never launches Minecraft.  This module is the
missing runner.  It walks the actionable scenes of one manifest **in manifest
order, one scene at a time**, and for each scene advances the validated status
ladder:

``pending``/``failed`` -> ``running``
    1. **prepare** - run the queued ``prepare_capture_dataset.py`` argv with
       ``--arm`` appended, so the staged world, plan and ``armed.json`` are
       created together;
    2. **verify** - the plan written by prepare, the staged plan digest, the
       armed marker's plan id/staging token/world identity, the mod's in-world
       staging marker, the accepted mute/unfocus client options and the mod jar
       must all agree with the queue record *before* any game process starts;
    3. **capture** - hand the scene to a :class:`CaptureLauncher`; the concrete
       :class:`PrismCaptureLauncher` starts Prism Launcher, waits for the mod's
       ``run_status.json`` sentinel, then stops the process tree;
    4. **finalize** - run the queued ``finalize_capture_dataset.py`` argv and
       require a zero exit code.
``running`` -> ``complete`` only after finalize succeeds; any failure records
``failed`` with the stage and bounded command output as attempt evidence, so the
scene can be retried after ``resume_queue``/``--resume``.

Safety invariants enforced here
-------------------------------
* **Sequential.** One scene is prepared, captured and finalized at a time; a
  second command or game process is never started concurrently.
* **Fail closed before launch.** A scene is never launched unless prepare exited
  zero *and* the plan/armed/staging linkage verifies.  A missing or mismatched
  marker fails the scene instead of being ignored.
* **Content-addressed commands.** The argv executed is the argv validated by
  ``load_queue``; only ``argv[0] == "python"`` is replaced with the configured
  interpreter, and the ``--dataset``/``--staging-game-dir``/``--save``/``--scene``
  arguments are re-checked against the queue record before execution.
* **Bounded.** Every command and the capture run has a timeout; a command that
  exceeds it is killed (with its process tree) and the scene fails with
  ``timed_out`` evidence.
* **No source-save write.** Only the queue manifest's status/attempt fields
  (through ``set_status``) and the dataset/staging outputs the factory scripts
  already own are touched.  Importing or dry-running this module launches
  nothing.
* **Dry run by default.** Importing or planning this module launches nothing.
* :class:`WindowsJunctionAdapter` - creates each worker's ``minecraft`` junction
  only when absent. It validates the worker path under an explicit
  ``worker_root`` and the target under the queue's ``dataset/staging-game``
  root. A safe exact-target junction is an idempotent no-op; a different target
  is refused. The PowerShell script repeats these checks and only creates a
  missing junction.
* :class:`DisposablePrismWorker` - one immutable worker instance per scene.
  It refuses a template equal to the worker, an out-of-root worker, or an
  existing but unmarked worker, and initializes a missing worker by copying only
  ``instance.cfg`` and ``mmc-pack.json`` (rewriting name/uuid and atomically
  writing a worker marker); the template's ``minecraft`` is never copied or
  mutated. Deterministic scene-specific names bind each worker to one staging
  directory.
* :class:`WindowsDesktop2Controller` - preflights a second virtual desktop
  (``Import-Module VirtualDesktop``, ``Get-DesktopCount >= 2``, and a refusal
  when the operator is already on the target capture desktop 1, remembering
  ``Get-DesktopIndex(Get-CurrentDesktop)``) and, after ``Popen`` and on every
  bounded poll, moves only nonzero ``MainWindowHandle`` windows belonging to
  the owning Prism PID and its live descendants with ``Move-Window -Desktop
  (Get-Desktop 1) -Hwnd`` while proving the current desktop index did not change. It never
  switches the active desktop, activates/focuses a window, or sends
  keyboard/mouse input.
* :class:`CaptureModProvisioner` - ``--execute`` requires an explicit
  ``--capture-mod`` path and ``--capture-mod-sha256``: the source hash must match
  exactly before an atomic copy to ``staging/mods/<source.name>``, the
  destination is re-hashed, and a missing/mismatched/multiple capture jar is
  refused.  There is no hardcoded default jar.

The post-prepare client-options check is unchanged and still requires exactly
``pauseOnLostFocus:false``, ``rawMouseInput:false`` and
``soundCategory_master:0.0`` before the launch.  A dry run constructs no
worker, junction, desktop controller, provisioner or launcher process, so it
mutates nothing.

Operational repair added by DS-31U (first live-probe failure)
-------------------------------------------------------------
The first real pilot stopped safely **before Prism**: the worker's
``minecraft`` junction was created correctly, but
:class:`WindowsLinkInspector` read its raw reparse target through
``os.readlink`` and reported the extended-length ``\\?\C:\...`` spelling while
the requested staging path was written ``C:\...``.  The plain string + case
comparison in :meth:`WindowsJunctionAdapter.ensure` therefore rejected an
otherwise exact match.  :func:`_path_key` now strips a leading ``\\?\`` (and
restores ``\\?\UNC\server\share`` to ``\\server\share``) before it makes the
path absolute and case-normalizes it, so both spellings of the same object are
equal for the link target equality/containment checks and the worker marker
comparisons, while a genuinely different path still fails.  Nothing else about
the pipeline changed and no live process was started by this repair.

DS-31V mutable-link proposal rejected by DS-33
----------------------------------------------
The interrupted DS-31V attempt to replace an existing worker junction is
withdrawn. A worker now belongs to one scene for its lifetime: an absent link
is created once, an exact existing target is accepted without a shell call, and
a changed target is refused with a requirement for that scene's distinct
worker. The runtime junction path contains no replacement or removal step.

"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat as stat_module
import subprocess
import sys
import tempfile
import threading
import time
import uuid as uuid_module
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from mc_photogrammetry.dataset.batch import (
    PYTHON_PROGRAM,
    load_queue,
    resume_queue,
    set_status,
)
from mc_photogrammetry.dataset.console import write_line
from mc_photogrammetry.dataset.hashing import canonical_json, is_link_like, sha256_file
from mc_photogrammetry.dataset.inventory import (
    WORLD_MARKER_NAME,
    safe_component,
    safe_join,
)
from mc_photogrammetry.dataset.options import CAPTURE_OPTIONS, OPTIONS_FILE_NAME
from mc_photogrammetry.dataset.schemas import (
    MOD_CONTRACT_VERSION,
    SCHEMA_VERSION,
    CapturePlan,
    DatasetSchemaError,
)

#: The factory's mod directory inside a staging game directory.
MOD_DIR = "mc-photo-capture"
#: The single-use arming marker the mod validates before mutating a world.
ARMED_NAME = "armed.json"
#: The mod's terminal sentinel: written after results.json, before disarm.
RUN_STATUS_NAME = "run_status.json"
#: Directories the mod and the client read inside a staged game directory.
MODS_DIR_NAME = "mods"
SAVES_DIR_NAME = "saves"
#: The capture mod jar the staged client must build against.
MOD_JAR_GLOB = "mc-photo-capture-*.jar"
#: The single-flight lock guarding one queue.
LOCK_NAME = ".executor.lock"
#: The append-only, fsynced run ledger inside the queue directory.
LEDGER_NAME = "executor-runs.jsonl"
#: The marker that proves a directory is a disposable worker the manager created.
WORKER_MARKER_NAME = ".mc-photo-capture-worker.json"
#: The only two template files a missing worker is allowed to copy.
WORKER_TEMPLATE_FILES = ("instance.cfg", "mmc-pack.json")
#: Every file (plus its atomic-write ``.tmp``) a failed initialization may remove.
WORKER_INIT_FILES = WORKER_TEMPLATE_FILES + (WORKER_MARKER_NAME,)
#: The worker's game-directory junction name inside a Prism instance folder.
MINECRAFT_DIR_NAME = "minecraft"
#: Schema version of the disposable-worker marker.
WORKER_MARKER_SCHEMA_VERSION = 1

DEFAULT_PREPARE_TIMEOUT = 1800.0
DEFAULT_FINALIZE_TIMEOUT = 600.0
DEFAULT_CAPTURE_TIMEOUT = 3600.0
DEFAULT_POLL_SECONDS = 2.0
DEFAULT_TERMINATE_GRACE = 20.0
DEFAULT_OUTPUT_LIMIT = 8000
#: How often an in-flight factory command is checked for timeout/interrupt.
DEFAULT_COMMAND_POLL_SECONDS = 0.25
#: How long one PowerShell helper (junction / desktop) may run.
DEFAULT_POWERSHELL_TIMEOUT = 120.0

_ACTIONABLE = ("pending", "failed")


class ExecutorError(RuntimeError):
    """Raised when the executor itself cannot continue (not a scene failure)."""


class JunctionError(ExecutorError):
    """Raised when a worker game-directory junction cannot be proven safe."""


class WorkerError(ExecutorError):
    """Raised when a disposable Prism worker is unsafe to initialize or use."""


class DesktopError(ExecutorError):
    """Raised when the hidden-desktop preflight/move contract cannot be met."""


class ProvisionError(ExecutorError):
    """Raised when the capture mod cannot be proven before it is staged."""


def _absolute(path: Path | str) -> Path:
    """A lexical absolute path; never follows a link (a junction must stay one)."""
    return Path(os.path.abspath(str(path)))


#: The Win32 extended-length prefixes.  ``os.readlink`` reports a directory
#: junction's raw reparse target with ``\\?\`` in front, so ``\\?\C:\x`` and
#: ``C:\x`` (and ``\\?\UNC\server\share`` and ``\\server\share``) name the same
#: object and must compare equal.
_EXTENDED_PATH_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "UNC\\"
_UNC_PATH_PREFIX = "\\\\"


def _canonical_windows_path(path: Path | str) -> str:
    r"""``path`` with any Win32 extended-length prefix rewritten to its plain form.

    The rewrite is purely lexical: it only drops a leading ``\\?\`` (restoring
    ``\\?\UNC\server\share`` to ``\\server\share``) and never resolves, follows
    or otherwise reinterprets the final component, so a link stays a link and
    containment is never widened.
    """
    text = str(path)
    if text.startswith(_EXTENDED_PATH_PREFIX):
        remainder = text[len(_EXTENDED_PATH_PREFIX):]
        if remainder[:len(_EXTENDED_UNC_PREFIX)].upper() == _EXTENDED_UNC_PREFIX:
            return _UNC_PATH_PREFIX + remainder[len(_EXTENDED_UNC_PREFIX):]
        return remainder
    return text


def _path_key(path: Path | str) -> str:
    r"""A case-normalized absolute path string for containment comparison.

    The two spellings Windows uses for one object - the plain ``C:\x`` and the
    extended-length ``\\?\C:\x`` that a junction's raw reparse target carries (or
    ``\\server\share`` and ``\\?\UNC\server\share``) - produce the same key, so
    equality and containment no longer depend on which spelling was observed.  A
    genuinely different path still produces a different key.
    """
    return os.path.normcase(str(_absolute(_canonical_windows_path(path))))


def path_contains(root: Path | str, candidate: Path | str) -> bool:
    """Whether ``candidate`` is ``root`` or sits beneath it (lexically)."""
    root_key = _path_key(root)
    candidate_key = _path_key(candidate)
    if candidate_key == root_key:
        return True
    return candidate_key.startswith(root_key.rstrip("\\/") + os.sep)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- bounded, interruptible command execution -------------------------------


def tail_text(text: str, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    """The last ``limit`` characters of ``text``, prefixed when truncated."""
    if len(text) <= limit:
        return text
    return "...[truncated]..." + text[-limit:]


def _stop_process_tree(process: Any, *, grace: float,
                       taskkill: Callable[..., Any] = subprocess.run) -> bool:
    """Terminate a process and its children; return whether the tree is confirmed dead.

    On Windows the launcher's Java child is killed with ``taskkill /T /F``.  On
    POSIX the child is started in its own session and its whole process group is
    signalled, so a launcher wrapper cannot strand the game.  The return value is
    the operational guarantee: ``False`` means the tree could not be proven gone
    (for example ``taskkill`` was denied) and the caller must fail closed.  The
    direct child is always terminated as a fallback before returning.
    """
    if process.poll() is not None:
        return True
    if os.name == "nt":
        tree_confirmed = False
        try:
            completed = taskkill(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, timeout=max(5.0, grace))
            tree_confirmed = getattr(completed, "returncode", 1) == 0
        except (OSError, subprocess.SubprocessError):
            tree_confirmed = False
        if not tree_confirmed:
            # The tree kill was refused; at least reap the launcher itself.
            try:
                process.terminate()
            except OSError:
                pass
    else:
        tree_confirmed = False
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            tree_confirmed = True
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except OSError:
                    process.kill()
            else:
                process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    return process.poll() is not None and tree_confirmed


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The bounded outcome of one external command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0
    interrupted: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.interrupted

    def evidence(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "returncode": int(self.returncode),
            "timed_out": bool(self.timed_out),
            "interrupted": bool(self.interrupted),
            "duration_seconds": round(float(self.duration_seconds), 3),
            "stdout": tail_text(self.stdout),
            "stderr": tail_text(self.stderr),
        }


class CommandRunner(Protocol):
    """Injectable boundary for running the factory scripts."""

    def run(self, argv: Sequence[str], *, cwd: Path, timeout: float) -> CommandResult:
        ...


class SubprocessRunner:
    """Run a command list with no shell, a fixed cwd, and a hard timeout.

    The argument vector is always a list, so a save/region name can never be
    reinterpreted as shell syntax.  Output is captured and bounded.  The child is
    polled so a timeout *or* an operator interrupt kills its whole process tree
    and is reported as evidence rather than raising out of the run loop.
    """

    def __init__(
        self,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        *,
        interrupt: Any = None,
        stop: Callable[..., bool] = _stop_process_tree,
        grace: float = DEFAULT_TERMINATE_GRACE,
        poll_seconds: float = DEFAULT_COMMAND_POLL_SECONDS,
    ) -> None:
        self.output_limit = int(output_limit)
        self._interrupt = interrupt
        self._stop = stop
        self.grace = float(grace)
        self.poll_seconds = float(poll_seconds)

    def _interrupted(self) -> bool:
        return self._interrupt is not None and self._interrupt.is_set()

    def _collect(self, process: Any) -> tuple[str, str, bool]:
        """Drain a stopped child's pipes; return bounded stdout/stderr and liveness."""
        try:
            stdout, stderr = process.communicate(timeout=max(1.0, self.grace))
        except (subprocess.TimeoutExpired, ValueError):
            stdout, stderr = "", ""
        return (
            tail_text(_decode(stdout), self.output_limit),
            tail_text(_decode(stderr), self.output_limit),
            process.poll() is not None,
        )

    def run(self, argv: Sequence[str], *, cwd: Path, timeout: float) -> CommandResult:
        command = [str(item) for item in argv]
        started = time.monotonic()
        kwargs: dict[str, Any] = {}
        if os.name != "nt":
            # A POSIX child gets its own process group so the whole tree can be
            # signalled; see :func:`_stop_process_tree`.
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(
                command, cwd=str(cwd), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", **kwargs,
            )
        except OSError as error:
            return CommandResult(
                argv=tuple(command), returncode=-2, stderr=str(error),
                duration_seconds=time.monotonic() - started,
            )

        deadline = started + float(timeout)
        while True:
            try:
                stdout, stderr = process.communicate(timeout=self.poll_seconds)
                return CommandResult(
                    argv=tuple(command), returncode=int(process.returncode or 0),
                    stdout=tail_text(stdout or "", self.output_limit),
                    stderr=tail_text(stderr or "", self.output_limit),
                    duration_seconds=time.monotonic() - started,
                )
            except subprocess.TimeoutExpired:
                pass
            if self._interrupted():
                self._stop(process, grace=self.grace)
                stdout, stderr, _ = self._collect(process)
                return CommandResult(
                    argv=tuple(command), returncode=-3, stdout=stdout, stderr=stderr,
                    interrupted=True, duration_seconds=time.monotonic() - started,
                )
            if time.monotonic() >= deadline:
                self._stop(process, grace=self.grace)
                stdout, stderr, _ = self._collect(process)
                return CommandResult(
                    argv=tuple(command), returncode=-1, stdout=stdout, stderr=stderr,
                    timed_out=True, duration_seconds=time.monotonic() - started,
                )


def _decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


# --- single-flight lock and durable ledger ----------------------------------


def _pid_alive(pid: int) -> bool:
    """Whether a local process id is still running; unknown counts as alive."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        # os.kill(pid, 0) terminates the target on Windows, so never use it here.
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return True
        return str(pid) in (completed.stdout or "")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


class RunLock:
    """An atomic single-flight lock over one capture queue.

    Only one executor may prepare/capture/tear down a queue at a time.  The lock
    is created with ``O_CREAT | O_EXCL`` and records the owner pid/host/start
    time plus a random token.  A lock held by a dead process on this host is
    stale and is reclaimed automatically; a lock from another host (whose
    liveness cannot be proven) is refused unless ``force`` is explicit.
    """

    def __init__(self, queue_dir: Path | str, *, force: bool = False) -> None:
        self.queue_dir = Path(queue_dir)
        self.force = bool(force)
        self._owner: str | None = None

    @property
    def path(self) -> Path:
        return self.queue_dir / LOCK_NAME

    @property
    def held(self) -> bool:
        return self._owner is not None

    def _read_existing(self) -> Any:
        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _stale(self, existing: Any) -> bool:
        if not isinstance(existing, Mapping):
            return False
        if existing.get("host") != socket.gethostname():
            return False
        pid = existing.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int):
            return False
        return not _pid_alive(pid)

    def _busy_message(self, existing: Any) -> str:
        if isinstance(existing, Mapping):
            return (f"queue {self.queue_dir} is already owned by pid "
                    f"{existing.get('pid')} on {existing.get('host')} "
                    f"(started {existing.get('created')}); refusing a concurrent run "
                    "(pass force_unlock only if that owner is known dead)")
        return (f"queue {self.queue_dir} has an unreadable lock {self.path}; refusing "
                "to guess whether a run is live (pass force_unlock to reclaim)")

    def acquire(self, *, manifest_id: str = "") -> dict[str, Any]:
        """Create the lock or raise :class:`ExecutorError` if it is held."""
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created": _utc_now(),
            "manifest_id": str(manifest_id),
            "owner": os.urandom(8).hex(),
        }
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                existing = self._read_existing()
                if self.force or (attempt == 0 and self._stale(existing)):
                    try:
                        os.unlink(self.path)
                    except OSError as error:
                        raise ExecutorError(
                            f"cannot reclaim stale lock {self.path}: {error}") from error
                    continue
                raise ExecutorError(self._busy_message(existing))
            except OSError as error:
                raise ExecutorError(f"cannot create executor lock {self.path}: {error}") from error
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
                stream.write(canonical_json(payload))
                stream.flush()
                os.fsync(stream.fileno())
            self._owner = payload["owner"]
            return payload
        raise ExecutorError(f"could not acquire executor lock {self.path}")

    def release(self) -> None:
        """Release the lock, but only while this owner still holds it."""
        if self._owner is None:
            return
        existing = self._read_existing()
        if isinstance(existing, Mapping) and existing.get("owner") == self._owner:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        self._owner = None


class RunLedger:
    """An append-only, fsynced JSONL receipt stream for live runs.

    The queue manifest only proves the current status; the ledger proves what
    actually ran, with the argv, exit codes, pids and timings, so an interrupted
    or crashed run can be reconstructed from disk.
    """

    def __init__(self, queue_dir: Path | str, *, name: str = LEDGER_NAME) -> None:
        self.path = Path(queue_dir) / name

    def append(self, record: Mapping[str, Any]) -> Path:
        line = canonical_json(dict(record)) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return self.path

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


# --- capture launcher boundary ---------------------------------------------


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    """Everything a launcher needs for one verified, armed scene."""

    scene_id: str
    save_name: str
    world_identity: str
    dataset: Path
    staging_game_dir: Path
    plan_id: str
    staging_token: str
    plan_path: Path
    staged_plan_path: Path
    armed_path: Path
    output_dir: Path
    timeout_seconds: float
    mod_jar: Path | None = None
    staging_marker: Path | None = None
    provenance_expectation: Mapping[str, Any] | None = None
    provenance_mode: str = "report_only"

    @property
    def run_status_path(self) -> Path:
        return self.output_dir / RUN_STATUS_NAME


@dataclass(frozen=True, slots=True)
class CaptureOutcome:
    """Whether one scene's capture produced a clean mod run."""

    completed: bool
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    #: Which pipeline stage the outcome belongs to; the executor records it.
    stage: str = "capture"


class CaptureLauncher(Protocol):
    """Injectable boundary for the Minecraft/Prism capture step."""

    def launch(self, request: CaptureRequest) -> CaptureOutcome:
        ...


def read_json_file(path: Path) -> Any | None:
    """Parse a JSON file, or return None when it is absent; malformed is an error."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise DatasetSchemaError(f"cannot read {path}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise DatasetSchemaError(f"{path} is not valid JSON: {error}") from error


def read_options_file(path: Path) -> dict[str, str]:
    """Parse a client ``options.txt`` into ``key -> value``; absent means empty."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise DatasetSchemaError(f"cannot read {path}: {error}") from error
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def interpret_run_status(
    payload: Mapping[str, Any], *,
    plan_id: str | None = None,
    staging_token: str | None = None,
    schema_version: int | None = None,
    mod_contract_version: int | None = None,
    provenance_expectation: Mapping[str, Any] | None = None,
) -> CaptureOutcome:
    """Judge the mod's terminal ``run_status.json`` payload.

    ``ground_truth_exported`` is the mod's own boolean proof that a complete
    export exists; it is false for a sentinel written on any failure path.  The
    finalizer re-validates every frame, but the executor refuses to spend a
    finalize run on a capture the mod already declared failed.

    When the caller supplies the expected identity/version, a sentinel that names
    another plan, another staging token, or another contract version is refused.
    """
    if not isinstance(payload, Mapping):
        return CaptureOutcome(False, "run_status.json is not a JSON object")
    if schema_version is not None and payload.get("schema_version") != schema_version:
        return CaptureOutcome(
            False, f"run_status.json declares schema_version "
                   f"{payload.get('schema_version')!r}, not {schema_version!r}")
    if mod_contract_version is not None and \
            payload.get("mod_contract_version") != mod_contract_version:
        return CaptureOutcome(
            False, f"run_status.json declares mod_contract_version "
                   f"{payload.get('mod_contract_version')!r}, not {mod_contract_version!r}")
    recorded_plan = payload.get("plan_id")
    if plan_id is not None and recorded_plan not in (None, plan_id):
        return CaptureOutcome(
            False, f"run_status.json names plan {recorded_plan!r}, not {plan_id!r}")
    if staging_token is not None and payload.get("staging_token") != staging_token:
        return CaptureOutcome(
            False, "run_status.json staging token does not match the armed scene")
    if provenance_expectation is not None:
        receipt = payload.get("runtime_render_provenance")
        if payload.get("runtime_render_provenance_required") is not True or \
                not isinstance(receipt, Mapping):
            return CaptureOutcome(False, "required runtime/render provenance receipt is missing")
        for key in ("attempt_nonce", "plan_sha256", "staging_token",
                    "expectation_sha256", "capture_mod_staged_sha256", "mode"):
            expected = provenance_expectation.get(key)
            if receipt.get(key) != expected:
                return CaptureOutcome(
                    False, f"runtime/render provenance {key} does not match this launch")
        if provenance_expectation.get("mode") == "strict" and receipt.get("certifiable") is not True:
            return CaptureOutcome(
                False, "runtime/render provenance is explicitly non-certifiable")
        sidecar_digest = receipt.get("sha256")
        if not isinstance(sidecar_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", sidecar_digest):
            return CaptureOutcome(False, "runtime/render provenance sidecar digest is invalid")
    errors = payload.get("errors")
    error_text = "; ".join(str(item) for item in errors) if isinstance(errors, list) else ""
    if payload.get("run_failed") is True:
        return CaptureOutcome(False, error_text or "the mod reported a failed run")
    if payload.get("ground_truth_exported") is not True:
        return CaptureOutcome(
            False, error_text or "the mod did not export ground truth")
    return CaptureOutcome(
        True, "capture complete",
        {"plan_id": recorded_plan, "staging_token": payload.get("staging_token")},
    )


@dataclass(frozen=True, slots=True)
class PrismInstance:
    """A Prism Launcher instance whose game directory is a staged capture root.

    Prism binds an instance to one game directory through
    ``<instance_dir>/minecraft`` (usually a directory junction). This value
    object and the launcher never mutate that link; the managed worker adapter
    creates it once before launch. :meth:`verify_game_dir` refuses to launch
    unless it resolves to the exact staging directory this scene prepared.
    """

    instance_id: str
    instance_dir: Path
    executable: Path | None = None
    prism_root: Path | None = None

    def game_dir(self) -> Path:
        if not (self.instance_dir / "instance.cfg").is_file():
            raise DatasetSchemaError(
                f"Prism instance {self.instance_id!r} has no instance.cfg at "
                f"{self.instance_dir}; refusing to launch a non-instance directory")
        minecraft = self.instance_dir / "minecraft"
        if not minecraft.exists():
            raise DatasetSchemaError(
                f"Prism instance {self.instance_id!r} has no minecraft directory at "
                f"{minecraft}")
        resolved = minecraft.resolve()
        if not resolved.is_dir():
            raise DatasetSchemaError(
                f"Prism instance {self.instance_id!r} game directory {minecraft} does "
                "not resolve to a directory")
        return resolved

    def verify_game_dir(self, expected: Path) -> Path:
        actual = self.game_dir()
        wanted = Path(expected).resolve()
        if actual != wanted:
            raise DatasetSchemaError(
                f"Prism instance {self.instance_id!r} game directory {actual} is not the "
                f"scene staging directory {wanted}; refusing to launch the wrong world")
        return actual

    def launch_argv(self, *, world: str | None = None) -> tuple[str, ...]:
        if self.executable is None:
            raise DatasetSchemaError("a Prism launch needs an executable path")
        argv = [str(self.executable)]
        if self.prism_root is not None:
            argv += ["-d", str(self.prism_root)]
        argv += ["-l", str(self.instance_id)]
        if world:
            argv += ["-w", str(world)]
        return tuple(argv)


# --- disposable worker game-directory junction ------------------------------

#: `Path.lstat()` classification of the worker's ``minecraft`` entry.
_LINK_ABSENT = "absent"
_LINK_JUNCTION = "junction"
_LINK_SYMLINK = "symlink"
_LINK_OTHER_LINK = "other-link"
_LINK_REAL_DIR = "real-dir"
_LINK_REAL_FILE = "real-file"


@dataclass(frozen=True, slots=True)
class LinkState:
    """What a path currently is, without ever following the final link."""

    kind: str
    target: Path | None = None


class LinkInspector(Protocol):
    """Injectable boundary for classifying the worker's ``minecraft`` entry."""

    def inspect(self, path: Path) -> LinkState:
        ...


class WindowsLinkInspector:
    """Classify a path as absent, a directory junction, another link or a real entry.

    Windows reports a directory junction as a reparse point whose tag is
    ``IO_REPARSE_TAG_MOUNT_POINT``; a symlink carries
    ``IO_REPARSE_TAG_SYMLINK``.  Only the former may be removed and replaced by
    :class:`WindowsJunctionAdapter`; a symlink or unknown reparse point is
    refused, and a real directory is never deleted.
    """

    def inspect(self, path: Path) -> LinkState:
        candidate = Path(path)
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            return LinkState(_LINK_ABSENT)
        except OSError as error:
            raise JunctionError(f"cannot inspect {candidate}: {error}") from error
        attributes = getattr(info, "st_file_attributes", 0)
        reparse_flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        junction_tag = getattr(stat_module, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)
        symlink_tag = getattr(stat_module, "IO_REPARSE_TAG_SYMLINK", 0xA000000C)
        is_symlink = stat_module.S_ISLNK(info.st_mode)
        if (attributes & reparse_flag) or is_symlink:
            target = self._target(candidate)
            tag = getattr(info, "st_reparse_tag", 0)
            if is_symlink or tag == symlink_tag:
                return LinkState(_LINK_SYMLINK, target)
            if tag == junction_tag:
                return LinkState(_LINK_JUNCTION, target)
            return LinkState(_LINK_OTHER_LINK, target)
        if stat_module.S_ISDIR(info.st_mode):
            return LinkState(_LINK_REAL_DIR)
        return LinkState(_LINK_REAL_FILE)

    def _target(self, path: Path) -> Path | None:
        try:
            raw = os.readlink(path)
        except OSError:
            try:
                return Path(path).resolve()
            except OSError:
                return None
        target = Path(raw)
        if not target.is_absolute():
            target = path.parent / target
        try:
            return target.resolve()
        except OSError:
            return target


#: The PowerShell script that validates and creates an absent worker junction.
#: The four paths are bound by name from positional ``-File`` arguments, never
#: interpolated into the script text, so a path with spaces or quotes is inert.
#: Containment is re-proved *inside* this process: the link must be strictly
#: below ``WorkerRoot`` and the requested/existing targets must be within
#: ``TargetRoot``. Existing exact-target junctions are a no-op; all other
#: existing entries fail closed. The script only creates an absent junction.
JUNCTION_ENSURE_SCRIPT = r"""param(
    [Parameter(Mandatory = $true)][string]$Minecraft,
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(Mandatory = $true)][string]$WorkerRoot,
    [Parameter(Mandatory = $true)][string]$TargetRoot
)
$ErrorActionPreference = "Stop"

function Get-NormalizedPath {
    param([string]$Path)
    $candidate = $Path
    if ($candidate.StartsWith("\\?\UNC\", [System.StringComparison]::OrdinalIgnoreCase)) {
        $candidate = "\\" + $candidate.Substring(8)
    } elseif ($candidate.StartsWith("\\?\", [System.StringComparison]::OrdinalIgnoreCase)) {
        $candidate = $candidate.Substring(4)
    }
    $full = [System.IO.Path]::GetFullPath($candidate)
    $root = [System.IO.Path]::GetPathRoot($full)
    if ($full.Length -gt $root.Length) {
        $full = $full.TrimEnd([char[]]@('\', '/'))
    }
    return $full
}

function Test-PathUnderRoot {
    param([string]$Root, [string]$Candidate, [switch]$Strict)
    $rootPath = Get-NormalizedPath $Root
    $candidatePath = Get-NormalizedPath $Candidate
    if (-not $Strict) {
        if ($candidatePath -ieq $rootPath) { return $true }
    }
    $prefix = $rootPath
    if (-not $prefix.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $prefix += [System.IO.Path]::DirectorySeparatorChar
    }
    return $candidatePath.StartsWith(
        $prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

$Minecraft = Get-NormalizedPath $Minecraft
$Target = Get-NormalizedPath $Target
$WorkerRoot = Get-NormalizedPath $WorkerRoot
$TargetRoot = Get-NormalizedPath $TargetRoot

if (-not (Test-PathUnderRoot -Root $WorkerRoot -Candidate $Minecraft -Strict)) {
    throw "worker junction is not strictly below the worker root"
}
if (-not (Test-PathUnderRoot -Root $TargetRoot -Candidate $Target)) {
    throw "junction target is outside the staging root"
}
if (-not (Test-Path -LiteralPath $Target -PathType Container)) {
    throw "junction target is not an existing directory"
}

$existing = Get-Item -LiteralPath $Minecraft -Force -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    if (-not ($existing.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "existing minecraft path is a real directory, not a junction"
    }
    if ($existing.LinkType -ne "Junction") {
        throw "existing minecraft path is a reparse point that is not a directory junction"
    }
    $existingTarget = [string](@($existing.Target) | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($existingTarget)) {
        throw "existing junction has no readable target"
    }
    if (-not [System.IO.Path]::IsPathRooted($existingTarget)) {
        $existingTarget = [System.IO.Path]::Combine(
            [System.IO.Path]::GetDirectoryName($Minecraft), $existingTarget)
    }
    $existingTarget = Get-NormalizedPath $existingTarget
    if (-not (Test-PathUnderRoot -Root $TargetRoot -Candidate $existingTarget)) {
        throw "existing junction points outside the staging root"
    }
    if ($existingTarget -ine $Target) {
        throw "existing junction points to a different staging target; use a distinct per-scene worker"
    }
    Write-Output $existingTarget
    exit 0
}

New-Item -ItemType Junction -Path $Minecraft -Target $Target | Out-Null
$created = Get-Item -LiteralPath $Minecraft -Force
if (-not ($created.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
    throw "created minecraft path is not a reparse point"
}
if ($created.LinkType -ne "Junction") {
    throw "created minecraft path is not a directory junction"
}
$createdTarget = [string](@($created.Target) | Select-Object -First 1)
if ([string]::IsNullOrWhiteSpace($createdTarget)) {
    throw "created minecraft junction has no readable target"
}
if (-not [System.IO.Path]::IsPathRooted($createdTarget)) {
    $createdTarget = [System.IO.Path]::Combine(
        [System.IO.Path]::GetDirectoryName($Minecraft), $createdTarget)
}
$createdTarget = Get-NormalizedPath $createdTarget
if ($createdTarget -ine $Target) {
    throw "created minecraft junction target does not equal the requested target"
}
Write-Output $createdTarget
"""


def _materialize_script(script_dir: Path | str | None, name: str, text: str) -> Path:
    directory = Path(script_dir) if script_dir is not None else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8", newline="")
    return path


class JunctionAdapter(Protocol):
    """Injectable boundary for create-once worker junctions."""

    def ensure(self, link: Path, target: Path) -> Path:
        ...


class WindowsJunctionAdapter:
    """Create a worker junction once and refuse any target change.

    Every call validates both ends before mutation: the link must sit
    under the explicit ``worker_root`` (checked lexically, because resolving a
    junction would follow it to its target), the target must sit under the
    queue's staging root, and an existing link must be a directory junction
    whose canonical target is exactly the requested target. A different target,
    real directory, symlink, unknown reparse point, or uncontained junction is
    refused. The PowerShell script repeats containment and target checks to
    close the race between inspection and creation; it never replaces a link.
    """

    def __init__(
        self,
        *,
        worker_root: Path | str,
        target_root: Path | str,
        inspector: LinkInspector | None = None,
        runner: CommandRunner | None = None,
        powershell: str = "powershell.exe",
        script_dir: Path | str | None = None,
        timeout: float = DEFAULT_POWERSHELL_TIMEOUT,
    ) -> None:
        self.worker_root = _absolute(worker_root)
        self.target_root = _absolute(target_root)
        self._inspector = inspector if inspector is not None else WindowsLinkInspector()
        self._runner = runner
        self.powershell = str(powershell)
        self.script_dir = Path(script_dir) if script_dir is not None else None
        self.timeout = float(timeout)
        self._script_path: Path | None = None

    @property
    def script_path(self) -> Path:
        if self._script_path is None:
            self._script_path = _materialize_script(
                self.script_dir, "ensure-minecraft-junction.ps1",
                JUNCTION_ENSURE_SCRIPT)
        return self._script_path

    def _runner_instance(self) -> CommandRunner:
        return self._runner if self._runner is not None else SubprocessRunner()

    def _argv(self, link: Path, target: Path) -> list[str]:
        return [self.powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(self.script_path), str(link), str(target),
                str(self.worker_root), str(self.target_root)]

    def ensure(self, link: Path, target: Path) -> Path:
        link = _absolute(link)
        target = _absolute(target)
        if _path_key(link) == _path_key(self.worker_root) \
                or not path_contains(self.worker_root, link):
            raise JunctionError(
                f"worker junction {link} is not under the worker root "
                f"{self.worker_root}")
        if not path_contains(self.target_root, target):
            raise JunctionError(
                f"junction target {target} is not under the queue staging root "
                f"{self.target_root}")
        self._validate_target(target)

        state = self._inspector.inspect(link)
        if state.kind == _LINK_JUNCTION:
            if state.target is None:
                raise JunctionError(
                    f"existing junction {link} has an unknown target; refusing it")
            if not path_contains(self.target_root, state.target):
                raise JunctionError(
                    f"existing junction {link} points outside the queue staging root "
                    f"({state.target})")
            if _path_key(state.target) == _path_key(target):
                return state.target
            raise JunctionError(
                f"existing junction {link} targets {state.target}, not {target}; "
                "use a distinct per-scene worker")
        elif state.kind != _LINK_ABSENT:
            raise JunctionError(
                f"existing minecraft path {link} is {state.kind!r}, not a directory "
                "junction; refusing it")

        result = self._runner_instance().run(
            self._argv(link, target), cwd=self.worker_root, timeout=self.timeout)
        if not result.ok:
            raise JunctionError(
                f"creating {link} for {target} failed "
                f"(returncode {result.returncode}): {tail_text(result.stderr, 500)}")
        after = self._inspector.inspect(link)
        if after.kind != _LINK_JUNCTION or after.target is None:
            raise JunctionError(
                f"ensuring {link} did not produce a directory junction "
                f"(found {after.kind!r})")
        if _path_key(after.target) != _path_key(target):
            raise JunctionError(
                f"junction {link} resolved to {after.target}, not the requested "
                f"staging target {target}")
        return after.target

    def _validate_target(self, target: Path) -> None:
        try:
            is_dir = target.is_dir()
        except OSError as error:
            raise JunctionError(f"cannot inspect junction target {target}: {error}") from error
        if not is_dir:
            raise JunctionError(
                f"junction target {target} is not an existing directory")
        if is_link_like(target):
            raise JunctionError(
                f"junction target {target} is itself a link/reparse point; refusing "
                "to point a worker at it")


# --- disposable Prism worker ------------------------------------------------


def scene_worker_identity(
    scene_id: str,
    *,
    scope: str | Path,
    worker_dir_prefix: str,
    instance_id_prefix: str,
) -> tuple[str, str]:
    """Return bounded, stable folder and Prism ids for one scene.

    Prefixes and the readable scene fragment are ASCII-sanitized and capped;
    a 20-hex digest of the canonical staging-root scope plus full UTF-8 scene
    id prevents both sanitized-name and cross-dataset collisions without
    allowing an untrusted id to create an oversized path component.
    """
    scene_key = str(scene_id)
    if not scene_key:
        raise WorkerError("scene id must not be empty")
    scope_key = _path_key(_absolute(scope))

    def label(value: str, *, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-_")
        return (cleaned or fallback)[:20]

    digest_input = f"{scope_key}\0{scene_key}".encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:20]
    scene_label = label(scene_key, fallback="scene")
    worker_label = label(worker_dir_prefix, fallback="worker")
    instance_label = label(instance_id_prefix, fallback="instance")
    if worker_label != instance_label:
        raise WorkerError(
            "worker directory and Prism instance prefixes must normalize to the same name")
    suffix = f"{scene_label}-{digest}"
    identity = f"{worker_label}-{suffix}"
    return identity, identity


def _rewrite_instance_cfg(text: str, *, name: str, uuid: str) -> str:
    """Rewrite ``name=`` and ``uuid=`` in a copied ``instance.cfg``.

    Every other line keeps its original position, order and line ending.  The
    newline style of the source is preserved; missing keys are appended.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    out: list[str] = []
    seen_name = False
    seen_uuid = False
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        key = body.partition("=")[0].strip().lower()
        if key == "name" and not seen_name:
            out.append(f"name={name}{ending}")
            seen_name = True
        elif key == "uuid" and not seen_uuid:
            out.append(f"uuid={uuid}{ending}")
            seen_uuid = True
        else:
            out.append(line)
    if not seen_name:
        out.append(f"name={name}{newline}")
    if not seen_uuid:
        out.append(f"uuid={uuid}{newline}")
    rendered = "".join(out)
    if rendered and not rendered.endswith(("\n", "\r")):
        rendered += newline
    return rendered


class DisposablePrismWorker:
    """One disposable Prism instance permanently bound to one scene.

    The worker is a real Prism instance directory under an explicit
    ``worker_root`` whose ``minecraft`` entry is a create-once junction to one
    scene's staging game directory. A missing worker is initialized from a
    template by copying **only** ``instance.cfg`` and ``mmc-pack.json`` (the
    copy's name/uuid are rewritten and a marker is written atomically); the
    template's ``minecraft`` directory is never copied or mutated.  An existing
    directory without a valid marker, a template equal to the worker, or a
    worker outside the root is refused.  A failed initialization removes only
    the files this class created (and their atomic-write temporaries) and then
    calls a non-recursive ``rmdir``; it never recursively deletes a tree.
    """

    def __init__(
        self,
        *,
        worker_root: Path | str,
        worker_dir: Path | str,
        instance_id: str,
        template_dir: Path | str,
        executable: Path | str | None = None,
        prism_root: Path | str | None = None,
        junction: JunctionAdapter | None = None,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self.worker_root = _absolute(worker_root)
        raw_dir = Path(str(worker_dir))
        candidate = raw_dir if raw_dir.is_absolute() else self.worker_root / raw_dir
        self.worker_dir = _absolute(candidate)
        if _path_key(self.worker_dir) == _path_key(self.worker_root) \
                or not path_contains(self.worker_root, self.worker_dir):
            raise WorkerError(
                f"worker directory {self.worker_dir} is not under the worker root "
                f"{self.worker_root}")
        self.template_dir = _absolute(template_dir)
        if _path_key(self.template_dir) == _path_key(self.worker_dir):
            raise WorkerError(
                "the worker template and the disposable worker are the same path")
        if path_contains(self.worker_dir, self.template_dir) \
                or path_contains(self.template_dir, self.worker_dir):
            raise WorkerError(
                "the worker template and the disposable worker may not contain "
                "each other")
        self.instance_id = str(instance_id)
        try:
            safe_component(self.instance_id)
        except DatasetSchemaError as error:
            raise WorkerError(f"unsafe worker instance id: {error}") from error
        if self.instance_id != self.worker_dir.name:
            raise WorkerError(
                f"worker instance id {self.instance_id!r} must equal the worker "
                f"directory name {self.worker_dir.name!r}; Prism resolves -l by "
                "folder name")
        self.executable = Path(executable) if executable is not None else None
        self.prism_root = Path(prism_root) if prism_root is not None else None
        self.junction = junction
        self._clock = clock

    @property
    def marker_path(self) -> Path:
        return self.worker_dir / WORKER_MARKER_NAME

    @property
    def minecraft_path(self) -> Path:
        return self.worker_dir / MINECRAFT_DIR_NAME

    def prism_instance(self) -> PrismInstance:
        """The launch target for the worker; never reads or mutates anything."""
        return PrismInstance(self.instance_id, self.worker_dir,
                             executable=self.executable, prism_root=self.prism_root)

    def describe(self) -> dict[str, Any]:
        return {"instance_id": self.instance_id,
                "worker_dir": str(self.worker_dir),
                "template_dir": str(self.template_dir),
                "minecraft": str(self.minecraft_path)}

    def _read_marker(self) -> dict[str, Any] | None:
        try:
            text = self.marker_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as error:
            raise WorkerError(
                f"cannot read worker marker {self.marker_path}: {error}") from error
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise WorkerError(
                f"worker marker {self.marker_path} is not valid JSON: {error}") from error
        if not isinstance(payload, Mapping):
            raise WorkerError(f"worker marker {self.marker_path} is not a JSON object")
        return dict(payload)

    def _validate_marker(self, marker: Mapping[str, Any]) -> None:
        if marker.get("schema_version") != WORKER_MARKER_SCHEMA_VERSION:
            raise WorkerError(
                f"worker marker {self.marker_path} has unsupported schema_version "
                f"{marker.get('schema_version')!r}")
        if marker.get("instance_id") != self.instance_id:
            raise WorkerError(
                f"worker marker {self.marker_path} names instance "
                f"{marker.get('instance_id')!r}, not {self.instance_id!r}")
        if _path_key(str(marker.get("worker_dir", ""))) != _path_key(self.worker_dir):
            raise WorkerError(
                f"worker marker {self.marker_path} names worker directory "
                f"{marker.get('worker_dir')!r}, not {self.worker_dir}")
        if _path_key(str(marker.get("template_dir", ""))) != _path_key(self.template_dir):
            raise WorkerError(
                f"worker marker {self.marker_path} names template "
                f"{marker.get('template_dir')!r}, not {self.template_dir}")

    def ensure_ready(self) -> dict[str, Any]:
        """Validate an initialized worker, or initialize a missing one once."""
        marker = self._read_marker()
        if marker is not None:
            self._validate_marker(marker)
            return marker
        if self.worker_dir.exists():
            raise WorkerError(
                f"existing worker directory {self.worker_dir} has no "
                f"{WORKER_MARKER_NAME} marker; refusing to use or overwrite it")
        return self._initialize()

    def _initialize(self) -> dict[str, Any]:
        template_instance = self.template_dir / "instance.cfg"
        if not template_instance.is_file() or is_link_like(template_instance):
            raise WorkerError(
                f"worker template {self.template_dir} has no regular instance.cfg")
        try:
            self.worker_dir.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise WorkerError(
                f"cannot create disposable worker {self.worker_dir}: {error}") from error
        created = True
        try:
            for name in WORKER_TEMPLATE_FILES:
                source = self.template_dir / name
                if not source.exists():
                    if name == "instance.cfg":
                        raise WorkerError(
                            f"worker template {self.template_dir} has no {name}")
                    continue
                if not source.is_file() or is_link_like(source):
                    raise WorkerError(
                        f"worker template {source} is not a regular file")
                _atomic_write_bytes(self.worker_dir / name, source.read_bytes())
            instance_text = (self.worker_dir / "instance.cfg").read_text(
                encoding="utf-8", errors="surrogateescape")
            new_uuid = uuid_module.uuid4().hex
            _atomic_write_text(
                self.worker_dir / "instance.cfg",
                _rewrite_instance_cfg(instance_text, name=self.instance_id,
                                      uuid=new_uuid))
            marker = {
                "schema_version": WORKER_MARKER_SCHEMA_VERSION,
                "instance_id": self.instance_id,
                "worker_dir": str(self.worker_dir),
                "template_dir": str(self.template_dir),
                "uuid": new_uuid,
                "created": self._clock(),
            }
            _atomic_write_text(self.marker_path, canonical_json(marker))
            return marker
        except BaseException:
            # Never leave a half-created, unmarked worker that would be refused
            # forever.  Cleanup is bounded to the exact files this method can
            # create (and their atomic-write temporaries) followed by a
            # non-recursive ``rmdir``; an unexpected entry makes the ``rmdir``
            # fail loudly enough to leave the tree for an operator instead of
            # walking and deleting it.
            self._discard_failed_initialize(created=created)
            raise

    def _discard_failed_initialize(self, *, created: bool) -> None:
        """Remove only the known files a failed initialization could have made.

        The worker directory must have been freshly created by this instance and
        still sit strictly below its worker root; a link/reparse point or a
        non-directory at the worker path is left untouched.  This method never
        recurses: it unlinks the two copied template files, the marker and each
        of their ``.tmp`` siblings, then calls ``rmdir``.  A leftover unknown
        entry therefore blocks removal, which is the safe outcome.
        """
        if not created:
            return
        if _path_key(self.worker_dir) == _path_key(self.worker_root) \
                or not path_contains(self.worker_root, self.worker_dir):
            return
        try:
            info = self.worker_dir.lstat()
        except OSError:
            return
        if stat_module.S_ISLNK(info.st_mode) or is_link_like(self.worker_dir):
            return
        if not stat_module.S_ISDIR(info.st_mode):
            return
        for name in WORKER_INIT_FILES:
            for candidate in (self.worker_dir / name,
                              self.worker_dir / f"{name}.tmp"):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
        try:
            self.worker_dir.rmdir()
        except OSError:
            pass

    def ensure_target(self, staging_game_dir: Path | str) -> Path:
        """Create or verify the one allowed ``minecraft`` target for this worker."""
        if self.junction is None:
            raise WorkerError("no junction adapter is configured for the worker")
        target = _absolute(staging_game_dir)
        return self.junction.ensure(self.minecraft_path, target)


# --- capture mod provisioning ----------------------------------------------


class CaptureModProvisioner:
    """Verify and atomically stage the capture mod before a launch.

    ``--execute`` must name the jar and its SHA-256 explicitly.  The source is
    hashed before anything is copied, the destination is copied through a
    same-directory temporary file and ``os.replace``, and the destination is
    re-hashed afterwards.  A missing source, a hash mismatch, or more than one
    capture jar in the staging ``mods`` directory is refused; there is no
    hardcoded default jar.
    """

    def __init__(self, source: Path | str, sha256: str,
                 *, copier: Callable[[Path, Path], Any] | None = None) -> None:
        self.source = _absolute(source)
        self.expected_sha256 = str(sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", self.expected_sha256):
            raise ProvisionError(
                "the capture mod pin must be 64 lowercase hexadecimal characters")
        self._copier = copier

    def _copy(self, source: Path, destination: Path) -> None:
        copier = self._copier if self._copier is not None else shutil.copyfile
        copier(source, destination)

    def _refuse_multiple(self, mods: Path, *, allow: str) -> None:
        jars = sorted(path.name for path in mods.glob(MOD_JAR_GLOB)
                      if path.is_file())
        strangers = [name for name in jars if name != allow]
        if strangers:
            raise ProvisionError(
                f"staging mods directory {mods} already holds other capture jars "
                f"{strangers}; refusing to guess which capture mod to run")

    def provision(self, staging_game_dir: Path | str) -> Path:
        source = self.source
        if not source.is_file() or is_link_like(source):
            raise ProvisionError(
                f"capture mod source {source} is missing or not a regular file")
        if not fnmatch.fnmatch(source.name, MOD_JAR_GLOB):
            raise ProvisionError(
                f"capture mod source name {source.name!r} does not match "
                f"{MOD_JAR_GLOB}")
        actual = sha256_file(source)
        if actual != self.expected_sha256:
            raise ProvisionError(
                f"capture mod source {source} hashes to {actual}, not the pinned "
                f"{self.expected_sha256}")
        mods = Path(staging_game_dir) / MODS_DIR_NAME
        try:
            mods.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ProvisionError(f"cannot create staging mods {mods}: {error}") from error
        self._refuse_multiple(mods, allow=source.name)
        destination = mods / source.name
        temporary = mods / f".{source.name}.tmp"
        try:
            self._copy(source, temporary)
            if sha256_file(temporary) != self.expected_sha256:
                raise ProvisionError(
                    f"copied capture mod {temporary} does not match the pinned hash")
            os.replace(temporary, destination)
        except OSError as error:
            raise ProvisionError(
                f"cannot stage capture mod {source} into {mods}: {error}") from error
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
        if sha256_file(destination) != self.expected_sha256:
            raise ProvisionError(
                f"staged capture mod {destination} does not match the pinned hash")
        self._refuse_multiple(mods, allow=source.name)
        return destination


# --- hidden-desktop controller ---------------------------------------------


@dataclass(frozen=True, slots=True)
class DesktopPreflight:
    """The proven hidden-desktop state a controller remembers before a launch."""

    desktop_count: int
    current_index: int


class DesktopController(Protocol):
    """Injectable boundary for the second-desktop preflight and window move."""

    def preflight(self) -> DesktopPreflight:
        ...

    def move_windows(self, process_id: int) -> None:
        ...


#: The virtual desktop hidden game windows are moved to; it is never the
#: operator's current desktop, so a preflight that finds the operator already
#: there must refuse rather than launch onto the visible target.
TARGET_DESKTOP_INDEX = 1

#: Preflight: the VirtualDesktop module must load, expose a second desktop, and
#: prove the operator is not already sitting on the target capture desktop.
DESKTOP_PREFLIGHT_SCRIPT = r"""$ErrorActionPreference = "Stop"
Import-Module VirtualDesktop -ErrorAction Stop
$count = Get-DesktopCount
if ($count -lt 2) {
    throw "the hidden capture desktop requires at least two virtual desktops"
}
$index = Get-DesktopIndex (Get-CurrentDesktop)
if ($index -eq 1) {
    throw "the operator is already on the target capture desktop; refusing to launch there"
}
Write-Output ("count=" + $count + " index=" + $index)
"""

#: Move only the supplied Prism process tree to desktop 1; never change the active one.
DESKTOP_MOVE_SCRIPT = r"""param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$ProcessId
)
$ErrorActionPreference = "Stop"
Import-Module VirtualDesktop -ErrorAction Stop
$before = [int](Get-DesktopIndex (Get-CurrentDesktop))

# Require the exact process launched by the executor. Build a descendant set
# from one OS process-table snapshot; each PID is reopened and checked live
# before its window handle is captured and immediately before moving it.
$owner = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
if ($null -eq $owner) {
    throw "the owning Prism process $ProcessId is no longer live"
}
$rows = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)
$targetIds = [System.Collections.Generic.HashSet[int]]::new()
[void]$targetIds.Add($ProcessId)
do {
    $added = $false
    foreach ($row in $rows) {
        $candidateId = [int]$row.ProcessId
        if ($targetIds.Contains([int]$row.ParentProcessId) -and
            $targetIds.Add($candidateId)) {
            $added = $true
        }
    }
} while ($added)

function Get-WindowState([int]$CandidateProcessId, [long]$ExpectedStartTicks,
                         [long]$ExpectedHandle) {
    $current = Get-Process -Id $CandidateProcessId -ErrorAction SilentlyContinue
    if ($null -eq $current) { return "vanished" }
    try {
        $startTicks = [long]$current.StartTime.ToUniversalTime().Ticks
        $currentHandle = [long]$current.MainWindowHandle
    } catch {
        $confirmed = Get-Process -Id $CandidateProcessId -ErrorAction SilentlyContinue
        if ($null -eq $confirmed) { return "vanished" }
        throw
    }
    if ($startTicks -ne $ExpectedStartTicks) { return "replaced" }
    if ($currentHandle -eq 0 -or $currentHandle -ne $ExpectedHandle) {
        return "stale-handle"
    }
    return "current"
}

$windows = @()
foreach ($candidateId in @($targetIds | Sort-Object)) {
    $process = Get-Process -Id $candidateId -ErrorAction SilentlyContinue
    if ($null -eq $process) { continue }
    try {
        $handle = [long]$process.MainWindowHandle
        if ($handle -eq 0) { continue }
        $startTicks = [long]$process.StartTime.ToUniversalTime().Ticks
    } catch {
        $confirmed = Get-Process -Id $candidateId -ErrorAction SilentlyContinue
        if ($null -eq $confirmed) { continue }
        throw
    }
    $windows += [pscustomobject]@{
        ProcessId = [int]$candidateId
        StartTicks = $startTicks
        Handle = $handle
    }
}

$moved = 0
$stale = 0
foreach ($window in $windows) {
    $state = Get-WindowState $window.ProcessId $window.StartTicks $window.Handle
    if ($state -ne "current") {
        $stale++
        continue
    }
    try {
        Move-Window -Desktop (Get-Desktop 1) -Hwnd ([IntPtr]$window.Handle)
        $moved++
    } catch {
        $hresult = ""
        $exception = $_.Exception
        while ($null -ne $exception) {
            $candidateHresult = '{0:X8}' -f ($exception.HResult -band 0xFFFFFFFF)
            if ($candidateHresult -eq "8002802B") {
                $hresult = $candidateHresult
                break
            }
            $exception = $exception.InnerException
        }
        $stateAfterError = Get-WindowState $window.ProcessId $window.StartTicks $window.Handle
        if ($hresult -eq "8002802B" -and $stateAfterError -ne "current") {
            $stale++
            continue
        }
        throw
    }
}
$after = [int](Get-DesktopIndex (Get-CurrentDesktop))
if ($after -ne $before) {
    throw "the operator's current desktop changed during the window move"
}
Write-Output ("before=" + $before + " index=" + $after +
    " moved=" + $moved + " stale=" + $stale)
"""


class WindowsDesktop2Controller:
    """Keep the capture client on virtual desktop 1 without disturbing desktop 0.

    ``preflight`` loads the VirtualDesktop module with the bypass execution
    policy, requires at least two desktops, refuses to launch when the operator
    is already on the target capture desktop (index 1), and remembers the
    operator's current desktop index. ``move_windows`` receives the spawned
    Prism PID and moves only nonzero ``MainWindowHandle`` windows owned by that
    process or its live descendants. It revalidates process start time and
    handle before each move. The exact stale-handle HRESULT is ignored only
    when revalidation confirms the process/handle is stale; all other failures
    remain fatal. The script asserts both pre-move and post-move desktop index
    match preflight. Nothing
    here switches, activates, focuses or sends input to a window.
    """

    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        powershell: str = "powershell.exe",
        script_dir: Path | str | None = None,
        timeout: float = DEFAULT_POWERSHELL_TIMEOUT,
        cwd: Path | str | None = None,
    ) -> None:
        self._runner = runner
        self.powershell = str(powershell)
        self.script_dir = Path(script_dir) if script_dir is not None else None
        self.timeout = float(timeout)
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._preflight: DesktopPreflight | None = None
        self._preflight_script: Path | None = None
        self._move_script: Path | None = None

    @property
    def preflight_script(self) -> Path:
        if self._preflight_script is None:
            self._preflight_script = _materialize_script(
                self.script_dir, "desktop-preflight.ps1", DESKTOP_PREFLIGHT_SCRIPT)
        return self._preflight_script

    @property
    def move_script(self) -> Path:
        if self._move_script is None:
            self._move_script = _materialize_script(
                self.script_dir, "desktop-move-windows.ps1", DESKTOP_MOVE_SCRIPT)
        return self._move_script

    @property
    def preflight_state(self) -> DesktopPreflight | None:
        return self._preflight

    def _runner_instance(self) -> CommandRunner:
        return self._runner if self._runner is not None else SubprocessRunner()

    def _run(self, script: Path, *arguments: str) -> CommandResult:
        argv = [self.powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(script), *arguments]
        try:
            return self._runner_instance().run(argv, cwd=self.cwd, timeout=self.timeout)
        except (OSError, subprocess.SubprocessError) as error:
            raise DesktopError(f"cannot run the desktop helper {script}: {error}") from error

    @staticmethod
    def _numbers(text: str, keys: Sequence[str]) -> list[int]:
        values: dict[str, str] = {}
        for token in re.split(r"\s+", (text or "").strip()):
            key, separator, value = token.partition("=")
            if separator:
                values[key.strip()] = value.strip()
        numbers: list[int] = []
        for key in keys:
            raw = values.get(key)
            if raw is None or not re.fullmatch(r"-?\d+", raw):
                raise DesktopError(
                    f"the desktop helper did not report {key!r}: {text!r}")
            numbers.append(int(raw))
        return numbers

    def preflight(self) -> DesktopPreflight:
        result = self._run(self.preflight_script)
        if not result.ok:
            raise DesktopError(
                "the hidden-desktop preflight failed "
                f"(returncode {result.returncode}): {tail_text(result.stderr, 500)}")
        count, index = self._numbers(result.stdout, ("count", "index"))
        if count < 2:
            raise DesktopError(
                f"the hidden capture desktop requires at least two virtual "
                f"desktops; found {count}")
        if index == TARGET_DESKTOP_INDEX:
            raise DesktopError(
                f"the operator's current desktop is the target capture desktop "
                f"{TARGET_DESKTOP_INDEX}; refusing to launch onto the visible "
                "target")
        self._preflight = DesktopPreflight(count, index)
        return self._preflight

    def move_windows(self, process_id: int) -> None:
        if self._preflight is None:
            raise DesktopError(
                "the hidden-desktop preflight has not run; refusing to move windows")
        if type(process_id) is not int or process_id <= 0:
            raise DesktopError("a positive owning Prism process ID is required")
        result = self._run(self.move_script, str(process_id))
        if not result.ok:
            raise DesktopError(
                "moving the capture windows failed "
                f"(returncode {result.returncode}): {tail_text(result.stderr, 500)}")
        before, index = self._numbers(result.stdout, ("before", "index"))
        if before != self._preflight.current_index or index != self._preflight.current_index:
            raise DesktopError(
                f"the operator's current desktop changed from "
                f"{self._preflight.current_index} to {before}->{index}; refusing to continue")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class PrismCaptureLauncher:
    """Launch Prism per scene, wait for the mod sentinel, stop the process tree.

    The launcher is deliberately thin and injectable: :func:`instance_for_scene`
    resolves a scene id to a :class:`PrismInstance`, the popener/sleeper/clock and
    the process stopper default to real implementations, and
    :class:`CaptureOutcome` can only report success when the mod's own
    ``run_status.json`` proves a clean export *and* the process tree is confirmed
    gone.  A sentinel left by a previous run is cleared before launch, so a stale
    success can never satisfy a new capture.
    """

    def __init__(
        self,
        instance_for_scene: Callable[[str], PrismInstance],
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        terminate_grace: float = DEFAULT_TERMINATE_GRACE,
        popener: Callable[..., Any] = subprocess.Popen,
        stopper: Callable[..., Any] = _stop_process_tree,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        interrupt: Any = None,
        desktop: DesktopController | None = None,
    ) -> None:
        self._instance_for_scene = instance_for_scene
        self.poll_seconds = float(poll_seconds)
        self.terminate_grace = float(terminate_grace)
        self._popener = popener
        self._stopper = stopper
        self._sleeper = sleeper
        self._clock = clock
        self._wall_clock = wall_clock
        self._interrupt = interrupt
        self._desktop = desktop
        self._popen_kwargs: dict[str, Any] = {}
        if os.name != "nt" and popener is subprocess.Popen:
            self._popen_kwargs["start_new_session"] = True

    def _interrupted(self) -> bool:
        return self._interrupt is not None and self._interrupt.is_set()

    def _move_windows(self, process: Any) -> DesktopError | None:
        """Move the game's own windows off the operator desktop, or report why not."""
        if self._desktop is None:
            return None
        try:
            process_id = getattr(process, "pid", None)
            if type(process_id) is not int or process_id <= 0:
                raise DesktopError("the spawned Prism process has no valid PID")
            self._desktop.move_windows(process_id)
        except (DesktopError, OSError, subprocess.SubprocessError) as error:
            return error if isinstance(error, DesktopError) else DesktopError(str(error))
        return None

    def describe(self, scene_id: str, *, world: str | None = None) -> dict[str, Any]:
        """The launch command a scene would use, without verifying or starting it."""
        instance = self._instance_for_scene(scene_id)
        try:
            argv = list(instance.launch_argv(world=world))
            command: list[str] | None = argv
        except DatasetSchemaError as error:
            command = None
            argv = [str(error)]
        return {"instance_id": instance.instance_id,
                "instance_dir": str(instance.instance_dir),
                "game_dir": str(instance.instance_dir / "minecraft"),
                "command": command,
                "note": None if command is not None else argv[0]}

    def _sentinel(self, path: Path) -> tuple[Any | None, str | None]:
        try:
            return read_json_file(path), None
        except DatasetSchemaError as error:
            return None, str(error)

    def launch(self, request: CaptureRequest) -> CaptureOutcome:
        try:
            instance = self._instance_for_scene(request.scene_id)
        except DatasetSchemaError as error:
            return CaptureOutcome(False, str(error), {"stage": "resolve"}, "resolve")
        try:
            game_dir = instance.verify_game_dir(request.staging_game_dir)
            argv = instance.launch_argv(world=request.save_name)
        except DatasetSchemaError as error:
            return CaptureOutcome(False, str(error), {"stage": "verify"}, "verify")

        if self._interrupted():
            return CaptureOutcome(
                False, "interrupted before launch",
                {"stage": "interrupted", "interrupted": True}, "interrupted")

        # The desktop preflight runs before any process is started: a machine
        # without a proven second desktop fails the scene here, never after a
        # client is already on the operator's screen.
        if self._desktop is not None:
            try:
                self._desktop.preflight()
            except (DesktopError, OSError, subprocess.SubprocessError) as error:
                return CaptureOutcome(
                    False, f"hidden-desktop preflight failed: {error}",
                    {"stage": "desktop", "argv": list(argv)}, "desktop")

        # A sentinel left by an earlier attempt must never satisfy this launch.
        stale_removed = False
        if request.run_status_path.exists():
            try:
                request.run_status_path.unlink()
            except OSError as error:
                return CaptureOutcome(
                    False, f"cannot clear the stale sentinel "
                           f"{request.run_status_path}: {error}",
                    {"stage": "preflight"}, "preflight")
            stale_removed = True
        started_wall = self._wall_clock()

        try:
            process = self._popener(
                list(argv), cwd=str(instance.instance_dir), **self._popen_kwargs)
        except OSError as error:
            return CaptureOutcome(
                False, f"could not start {argv[0]}: {error}", {"stage": "start"}, "start")

        # Immediately after Popen, and on every bounded poll, move only this
        # game's windows to desktop 1; a failed move or a changed current
        # desktop tears the client down and fails the scene.
        try:
            desktop_error = self._move_windows(process)
        except (DesktopError, OSError, subprocess.SubprocessError) as error:
            desktop_error = DesktopError(str(error))
        if desktop_error is not None:
            self._stopper(process, grace=self.terminate_grace)
            return CaptureOutcome(
                False, f"hidden-desktop move failed: {desktop_error}",
                {"stage": "desktop", "argv": list(argv),
                 "pid": getattr(process, "pid", None)}, "desktop")

        timed_out = False
        interrupted = False
        malformed: str | None = None
        status: Any = None
        terminated = False
        try:
            deadline = self._clock() + float(request.timeout_seconds)
            while True:
                if self._interrupted():
                    interrupted = True
                    break
                try:
                    desktop_error = self._move_windows(process)
                except (DesktopError, OSError, subprocess.SubprocessError) as error:
                    desktop_error = DesktopError(str(error))
                if desktop_error is not None:
                    return CaptureOutcome(
                        False, f"hidden-desktop move failed: {desktop_error}",
                        {"stage": "desktop", "argv": list(argv),
                         "pid": getattr(process, "pid", None)}, "desktop")
                status, malformed = self._sentinel(request.run_status_path)
                if status is not None or malformed is not None:
                    break
                if self._clock() >= deadline:
                    timed_out = True
                    break
                if process.poll() is not None:
                    # The launcher exited before the sentinel appeared; give the
                    # atomic writer one last chance, then fail.
                    status, malformed = self._sentinel(request.run_status_path)
                    if status is None and malformed is None:
                        return CaptureOutcome(
                            False, f"prism exited with code {process.returncode} before "
                                   "run_status.json appeared",
                            {"stage": "capture", "argv": list(argv),
                             "game_dir": str(game_dir), "returncode": process.returncode,
                             "teardown_confirmed": True, "stale_sentinel_removed": stale_removed},
                            "capture")
                    break
                self._sleeper(self.poll_seconds)
        finally:
            terminated = bool(self._stopper(process, grace=self.terminate_grace))

        evidence: dict[str, Any] = {
            "argv": list(argv),
            "game_dir": str(game_dir),
            "run_status": str(request.run_status_path),
            "pid": getattr(process, "pid", None),
            "teardown_confirmed": terminated,
            "stale_sentinel_removed": stale_removed,
            "owning_pid": os.getpid(),
        }
        if interrupted:
            return CaptureOutcome(
                False, "interrupted during capture",
                {**evidence, "stage": "interrupted", "interrupted": True}, "interrupted")
        if timed_out:
            return CaptureOutcome(
                False, f"capture produced no run_status.json within "
                       f"{request.timeout_seconds:.0f}s",
                {**evidence, "stage": "capture", "timed_out": True}, "capture")
        if malformed is not None:
            return CaptureOutcome(
                False, f"malformed run_status.json: {malformed}",
                {**evidence, "stage": "capture"}, "capture")

        try:
            run_status_mtime = os.stat(request.run_status_path).st_mtime
        except OSError as error:
            return CaptureOutcome(
                False, f"run_status.json disappeared after it was read: {error}",
                {**evidence, "stage": "capture"}, "capture")
        fresh = run_status_mtime + 1.0 >= started_wall
        evidence.update({"run_status_mtime": run_status_mtime, "sentinel_fresh": fresh})
        if not fresh:
            return CaptureOutcome(
                False, "run_status.json predates this launch; refusing a stale capture",
                {**evidence, "stage": "capture"}, "capture")

        outcome = interpret_run_status(
            status, plan_id=request.plan_id, staging_token=request.staging_token,
            schema_version=SCHEMA_VERSION, mod_contract_version=MOD_CONTRACT_VERSION,
            provenance_expectation=request.provenance_expectation)
        if not outcome.completed:
            return CaptureOutcome(
                False, outcome.detail, {**evidence, "stage": "capture"}, "capture")
        if not terminated:
            return CaptureOutcome(
                False, "capture finished but the process tree could not be confirmed "
                       "stopped; refusing to continue with a possible orphan client",
                {**evidence, "stage": "teardown", "teardown_failed": True}, "teardown")
        evidence.update(dict(outcome.evidence))
        return CaptureOutcome(True, "capture complete", evidence, "capture")


# --- batch execution --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SceneOutcome:
    scene_id: str
    status: str
    stage: str
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "status": self.status, "stage": self.stage,
                "detail": self.detail, "evidence": dict(self.evidence)}


@dataclass(frozen=True, slots=True)
class BatchReport:
    queue_dir: Path
    manifest_id: str
    dry_run: bool
    outcomes: tuple[SceneOutcome, ...]
    interrupted: bool = False

    @property
    def completed(self) -> tuple[str, ...]:
        return tuple(item.scene_id for item in self.outcomes if item.status == "complete")

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(item.scene_id for item in self.outcomes if item.status == "failed")

    @property
    def skipped(self) -> tuple[str, ...]:
        return tuple(item.scene_id for item in self.outcomes if item.status == "skipped")


def _require_argv(scene: Mapping[str, Any], key: str) -> list[str]:
    argv = scene.get(key)
    if not isinstance(argv, (list, tuple)) or not argv or not all(
            isinstance(item, str) for item in argv):
        raise DatasetSchemaError(f"scene {scene.get('scene_id')!r} {key} is malformed")
    return [str(item) for item in argv]


def _arg_value(argv: Sequence[str], flag: str) -> str | None:
    try:
        index = list(argv).index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


class BatchExecutor:
    """Run a materialized capture queue sequentially and safely."""

    def __init__(
        self,
        queue_dir: Path | str,
        *,
        launcher: CaptureLauncher | None = None,
        runner: CommandRunner | None = None,
        python_executable: str | None = None,
        prepare_timeout: float = DEFAULT_PREPARE_TIMEOUT,
        finalize_timeout: float = DEFAULT_FINALIZE_TIMEOUT,
        capture_timeout: float = DEFAULT_CAPTURE_TIMEOUT,
        stop_on_failure: bool = True,
        dry_run: bool = False,
        repo_root: Path | str | None = None,
        stream: Any = None,
        interrupt_event: Any = None,
        force_unlock: bool = False,
        ledger: RunLedger | None = None,
        lock: RunLock | None = None,
        worker: DisposablePrismWorker | None = None,
        worker_for_scene: Callable[[str], DisposablePrismWorker] | None = None,
        provisioner: CaptureModProvisioner | None = None,
        provenance_mode: str = "report_only",
    ) -> None:
        if worker is not None and worker_for_scene is not None:
            raise ExecutorError("configure worker or worker_for_scene, not both")
        self.queue_dir = Path(queue_dir)
        self.launcher = launcher
        self.interrupt_event = interrupt_event if interrupt_event is not None \
            else threading.Event()
        self.runner = runner if runner is not None else SubprocessRunner(
            interrupt=self.interrupt_event)
        self.python_executable = str(python_executable or sys.executable)
        self.prepare_timeout = float(prepare_timeout)
        self.finalize_timeout = float(finalize_timeout)
        self.capture_timeout = float(capture_timeout)
        self.stop_on_failure = bool(stop_on_failure)
        self.dry_run = bool(dry_run)
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.stream = stream if stream is not None else sys.stdout
        self.force_unlock = bool(force_unlock)
        self.lock = lock if lock is not None else RunLock(
            self.queue_dir, force=self.force_unlock)
        self.ledger = ledger if ledger is not None else RunLedger(self.queue_dir)
        self.worker = worker
        self.worker_for_scene = worker_for_scene
        self.provisioner = provisioner
        if provenance_mode not in ("report_only", "strict"):
            raise ExecutorError("provenance_mode must be report_only or strict")
        self.provenance_mode = provenance_mode
        self._active_provenance_expectation_path: Path | None = None
        self._active_provenance_expectation_sha256: str | None = None
        self._interrupt_reason = ""
        self._active_scene: str | None = None

    # -- interruption --------------------------------------------------------

    @property
    def interrupted(self) -> bool:
        return self.interrupt_event.is_set()

    @property
    def interrupt_reason(self) -> str:
        return self._interrupt_reason or "interrupted"

    def request_interrupt(self, reason: str = "interrupted") -> None:
        """Ask the run to stop as soon as the in-flight step can be torn down."""
        if not self.interrupt_event.is_set():
            self._interrupt_reason = str(reason)
        self.interrupt_event.set()

    def _install_interrupt_handlers(self) -> Callable[[], None]:
        """Route SIGINT/SIGTERM/SIGBREAK into a bounded, tearing-down stop."""
        if threading.current_thread() is not threading.main_thread():
            return lambda: None
        previous: dict[Any, Any] = {}

        def handler(signum: int, frame: Any) -> None:
            if self.interrupt_event.is_set():
                raise KeyboardInterrupt
            self.request_interrupt(f"signal {signum}")

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                previous[number] = signal.signal(number, handler)
            except (ValueError, OSError):
                continue

        def restore() -> None:
            for number, original in previous.items():
                try:
                    signal.signal(number, original)
                except (ValueError, OSError):
                    pass

        return restore

    # -- ledger --------------------------------------------------------------

    def _ledger(self, record: Mapping[str, Any]) -> None:
        try:
            self.ledger.append(record)
        except OSError as error:
            raise ExecutorError(
                f"cannot write the run ledger {self.ledger.path}: {error}") from error

    def _ledger_safe(self, record: Mapping[str, Any]) -> None:
        try:
            self.ledger.append(record)
        except OSError:
            pass

    # -- selection -----------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        return load_queue(self.queue_dir)

    def _scenes(self, payload: Mapping[str, Any],
                scene_ids: Iterable[str] | None) -> list[dict[str, Any]]:
        scenes = [dict(scene) for scene in payload.get("scenes", [])]
        by_id = {str(scene["scene_id"]): scene for scene in scenes}
        if scene_ids is None:
            return scenes
        wanted: list[str] = []
        for scene_id in scene_ids:
            if scene_id not in by_id:
                raise DatasetSchemaError(f"queue has no scene {scene_id!r}")
            if scene_id not in wanted:
                wanted.append(scene_id)
        return [by_id[scene_id] for scene_id in wanted]

    def plan(self, scene_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """The ordered, actionable rows a run would execute (no writing)."""
        payload = self._load()
        dataset = Path(str(payload["dataset"])).resolve()
        rows = []
        for scene in self._scenes(payload, scene_ids):
            status = str(scene.get("status"))
            if status not in _ACTIONABLE:
                rows.append({"scene_id": scene["scene_id"], "status": status,
                             "actionable": False, "stages": [], "prepare_argv": [],
                             "finalize_argv": [], "capture": None,
                             "staging_game_dir": scene.get("staging_game_dir")})
                continue
            rows.append({
                "scene_id": scene["scene_id"],
                "status": status,
                "actionable": True,
                "stages": self._stage_names(),
                "prepare_argv": self._prepare_argv(scene, dataset),
                "finalize_argv": self._finalize_argv(scene, dataset),
                "staging_game_dir": scene.get("staging_game_dir"),
                "capture": self._describe_capture(scene),
                "worker": (str(self._worker_for(str(scene["scene_id"])).worker_dir)
                           if self._has_worker() else None),
                "capture_mod": (str(self.provisioner.source)
                                if self.provisioner is not None else None),
            })
        return rows

    def _stage_names(self) -> list[str]:
        stages = ["prepare --arm"]
        if self.provisioner is not None:
            stages.append("provision mod")
        if self._has_worker():
            stages.append("initialize/verify scene worker")
        stages.append("capture")
        stages.append("finalize")
        return stages

    def _has_worker(self) -> bool:
        return self.worker is not None or self.worker_for_scene is not None

    def _worker_for(self, scene_id: str) -> DisposablePrismWorker:
        if self.worker_for_scene is not None:
            return self.worker_for_scene(scene_id)
        if self.worker is None:
            raise ExecutorError("no worker is configured")
        return self.worker

    def _describe_capture(self, scene: Mapping[str, Any]) -> dict[str, Any] | None:
        describe = getattr(self.launcher, "describe", None)
        if describe is None:
            return None
        try:
            return dict(describe(str(scene["scene_id"]), world=str(scene.get("save"))))
        except DatasetSchemaError as error:
            return {"command": None, "note": str(error)}

    # -- argv construction and re-validation ---------------------------------

    def _interpreter_argv(self, scene: Mapping[str, Any], key: str) -> list[str]:
        argv = _require_argv(scene, key)
        if argv[0] == PYTHON_PROGRAM:
            argv = [self.python_executable, *argv[1:]]
        return argv

    def _prepare_argv(self, scene: Mapping[str, Any], dataset: Path) -> list[str]:
        argv = self._interpreter_argv(scene, "prepare_argv")
        if "--arm" not in argv:
            argv = [*argv, "--arm"]
        self._check_argv_against_scene(argv, scene, dataset=dataset,
                                       script="prepare_capture_dataset.py",
                                       extra={"--save": str(scene.get("save", ""))})
        return argv

    def _finalize_argv(self, scene: Mapping[str, Any], dataset: Path) -> list[str]:
        argv = self._interpreter_argv(scene, "finalize_argv")
        self._check_argv_against_scene(argv, scene, dataset=dataset,
                                       script="finalize_capture_dataset.py",
                                       extra={"--scene": str(scene.get("scene_id", ""))})
        if not self.dry_run:
            argv.extend(["--provenance-mode", self.provenance_mode])
        if not self.dry_run and self._active_provenance_expectation_path is not None and \
                self._active_provenance_expectation_sha256 is not None:
            argv.extend(["--provenance-expectation",
                         str(self._active_provenance_expectation_path),
                         "--provenance-expectation-sha256",
                         self._active_provenance_expectation_sha256])
        return argv

    def _check_argv_against_scene(self, argv: Sequence[str], scene: Mapping[str, Any], *,
                                  dataset: Path, script: str,
                                  extra: Mapping[str, str]) -> None:
        if not any(script in part for part in argv):
            raise DatasetSchemaError(
                f"scene {scene.get('scene_id')!r} argv does not invoke {script}")
        found_dataset = _arg_value(argv, "--dataset")
        staging = _arg_value(argv, "--staging-game-dir")
        record_staging = scene.get("staging_game_dir")
        if found_dataset is None or Path(found_dataset).resolve() != dataset.resolve():
            raise DatasetSchemaError(
                f"scene {scene.get('scene_id')!r} {script} --dataset {found_dataset!r} is "
                f"not the queue dataset {dataset}")
        if staging is None or Path(staging).resolve() != Path(str(record_staging)).resolve():
            raise DatasetSchemaError(
                f"scene {scene.get('scene_id')!r} {script} --staging-game-dir {staging!r} "
                f"is not the queued staging directory {record_staging!r}")
        for flag, expected in extra.items():
            found = _arg_value(argv, flag)
            if found != expected:
                raise DatasetSchemaError(
                    f"scene {scene.get('scene_id')!r} {script} {flag} {found!r} is not "
                    f"the queued {expected!r}")

    # -- runtime containment -------------------------------------------------

    def _check_runtime_containment(self, payload: Mapping[str, Any]) -> None:
        """Re-prove at run time that nothing will touch the live game directory."""
        if not payload.get("game_dir"):
            return
        game = Path(str(payload["game_dir"])).resolve()
        roots: dict[str, Path] = {}
        if payload.get("dataset"):
            roots["dataset"] = Path(str(payload["dataset"])).resolve()
        if payload.get("staging_game_dir"):
            roots["staging root"] = Path(str(payload["staging_game_dir"])).resolve()
        for scene in payload.get("scenes", []):
            if scene.get("staging_game_dir"):
                roots[f"scene {scene.get('scene_id')!r} staging"] = Path(
                    str(scene["staging_game_dir"])).resolve()
        for label, path in roots.items():
            if path == game or game in path.parents:
                raise ExecutorError(
                    f"{label} {path} sits inside the live game directory {game}; "
                    "refusing to run a capture against live game data")
            if path in game.parents:
                raise ExecutorError(
                    f"{label} {path} contains the live game directory {game}; "
                    "refusing to run a capture against live game data")

    # -- verified arming -----------------------------------------------------
    def _capture_request(self, scene: Mapping[str, Any], dataset: Path, *,
                         expected_mod_jar: Path | None = None) -> CaptureRequest:
        scene_id = str(scene["scene_id"])
        staging = Path(str(scene["staging_game_dir"])).resolve()
        plan_file = safe_join(dataset, "scenes", scene_id, "plan.json")
        plan_payload = read_json_file(plan_file)
        if plan_payload is None:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: prepare did not write {plan_file}")
        plan = CapturePlan.from_dict(plan_payload)
        if plan.context_export_only:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: context-export-only plans must use the isolated "
                "evaluation exporter, not the capture queue")
        if plan.scene_id != scene_id:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: plan names {plan.scene_id!r}")
        if plan.world_identity != scene.get("world_identity"):
            raise DatasetSchemaError(
                f"scene {scene_id!r}: plan world {plan.world_identity!r} is not the "
                f"queued {scene.get('world_identity')!r}")

        staged_plan = safe_join(staging, MOD_DIR, "plans", f"{plan.plan_id}.json")
        if not staged_plan.is_file():
            raise DatasetSchemaError(
                f"scene {scene_id!r}: staged plan {staged_plan} is missing")
        armed_path = safe_join(staging, "config", MOD_DIR, ARMED_NAME)
        armed = read_json_file(armed_path)
        if not isinstance(armed, Mapping):
            raise DatasetSchemaError(
                f"scene {scene_id!r}: {armed_path} is missing or not a JSON object; "
                "refusing to launch an unarmed world")
        if armed.get("plan_id") != plan.plan_id:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: armed plan_id {armed.get('plan_id')!r} is not the "
                f"prepared {plan.plan_id!r}")
        if armed.get("staging_token") != plan.staging_token or not plan.staging_token:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: armed staging token does not match the prepared plan")
        if armed.get("world_identity") != plan.world_identity:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: armed world identity is not the prepared plan's")
        recorded_digest = armed.get("plan_sha256")
        actual_digest = sha256_file(staged_plan)
        if recorded_digest != actual_digest:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: staged plan digest {actual_digest} is not the armed "
                f"{recorded_digest!r}")

        # The mod only trusts the marker inside the copied world, so prove the
        # staged copy exists and is the exact world the plan was built from.
        staging_marker = safe_join(
            staging, SAVES_DIR_NAME, str(scene.get("save", "")), WORLD_MARKER_NAME)
        marker = read_json_file(staging_marker)
        if not isinstance(marker, Mapping):
            raise DatasetSchemaError(
                f"scene {scene_id!r}: in-world staging marker {staging_marker} is missing; "
                "the staged world is not proven")
        if marker.get("token") != plan.staging_token:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: in-world staging marker token is not the prepared "
                "plan's staging token")
        if marker.get("world_identity") != plan.world_identity:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: in-world staging marker world identity is not the "
                "prepared plan's")

        # The accepted DS-21/22 live-safety rule: the client must start unmuted,
        # unfocused-pausable and without taking the operator's mouse.
        options_file = safe_join(staging, OPTIONS_FILE_NAME)
        options = read_options_file(options_file)
        missing_options = {key: value for key, value in CAPTURE_OPTIONS.items()
                           if options.get(key) != value}
        if missing_options:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: {options_file} does not carry the binding capture "
                f"options {missing_options}; refusing to launch a client that could pause, "
                "take the mouse, or play audio")

        mods_dir = safe_join(staging, MODS_DIR_NAME)
        jars = sorted(mods_dir.glob(MOD_JAR_GLOB)) if mods_dir.is_dir() else []
        if not jars:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: no {MOD_JAR_GLOB} in {mods_dir}; refusing to launch "
                "without the capture mod")
        if len(jars) > 1:
            raise DatasetSchemaError(
                f"scene {scene_id!r}: {mods_dir} holds more than one capture mod "
                f"{[path.name for path in jars]}; refusing to guess which to run")
        if expected_mod_jar is not None:
            wanted_jar = Path(expected_mod_jar).resolve()
            if jars[0].resolve() != wanted_jar:
                raise DatasetSchemaError(
                    f"scene {scene_id!r}: staged capture mod {jars[0]} is not the "
                    f"provisioned {wanted_jar}")
        mod_jar = jars[0]
        staged_mod_sha = sha256_file(mod_jar)
        source_mod_sha: str | None = None
        source_mod_path: str | None = None
        provisioned_sha: str | None = None
        if self.provisioner is not None:
            source_mod_path = str(self.provisioner.source)
            source_mod_sha = sha256_file(self.provisioner.source)
            provisioned_sha = self.provisioner.expected_sha256
            if source_mod_sha != provisioned_sha or staged_mod_sha != provisioned_sha:
                raise DatasetSchemaError(
                    "capture mod source or staged bytes changed after provisioning")
        elif self.provenance_mode == "strict":
            raise DatasetSchemaError(
                "strict provenance requires an independently pinned capture-mod source")
        # Bind this launch attempt to the verified plan and staged mod. The
        # fresh nonce prevents a previous run or retry from satisfying the join.
        provenance_expectation = {
            "schema_version": 1,
            "mode": self.provenance_mode,
            "attempt_nonce": uuid_module.uuid4().hex,
            "plan_id": plan.plan_id,
            "plan_sha256": actual_digest,
            "dataset_plan_sha256": sha256_file(plan_file),
            "staging_token": plan.staging_token,
            "scene_id": scene_id,
            "world_identity": plan.world_identity,
            "staging_game_dir": str(staging),
            "capture_mod_sha256": staged_mod_sha,
            "capture_mod_source_sha256": source_mod_sha,
            "capture_mod_staged_sha256": staged_mod_sha,
            "capture_mod_expected_sha256": provisioned_sha,
            "capture_mod_source_path": source_mod_path,
            "capture_mod_staged_path": str(mod_jar),
        }
        if self.provenance_mode == "strict" and source_mod_sha is None:
            raise DatasetSchemaError("strict provenance source capture-mod hash is unavailable")
        expectation_path = safe_join(
            dataset, "scenes", scene_id, "runtime-render-provenance.expect.json")
        expectation_bytes = canonical_json(provenance_expectation).encode("utf-8")
        _atomic_write_text(expectation_path, expectation_bytes.decode("utf-8"))
        provenance_expectation["expectation_sha256"] = hashlib.sha256(
            expectation_bytes).hexdigest()
        self._active_provenance_expectation_path = expectation_path
        self._active_provenance_expectation_sha256 = provenance_expectation["expectation_sha256"]
        armed["runtime_render_provenance"] = provenance_expectation
        try:
            _atomic_write_text(armed_path, canonical_json(armed))
        except OSError as error:
            raise DatasetSchemaError(
                f"cannot persist provenance launch expectation to {armed_path}: {error}") from error
        self._ledger_safe({
            "event": "runtime_render_provenance_expectation",
            "scene_id": scene_id,
            "mode": self.provenance_mode,
            "attempt_nonce": provenance_expectation["attempt_nonce"],
            "plan_sha256": actual_digest,
            "capture_mod_source_path": source_mod_path,
            "capture_mod_source_sha256": source_mod_sha,
            "capture_mod_staged_path": str(mod_jar),
            "capture_mod_staged_sha256": staged_mod_sha,
            "expectation_path": str(expectation_path),
            "expectation_sha256": provenance_expectation["expectation_sha256"],
        })

        output_dir = safe_join(staging, MOD_DIR, "out", plan.plan_id)
        return CaptureRequest(
            scene_id=scene_id,
            save_name=str(scene.get("save", "")),
            world_identity=plan.world_identity,
            dataset=dataset,
            staging_game_dir=staging,
            plan_id=plan.plan_id,
            staging_token=plan.staging_token,
            plan_path=plan_file,
            staged_plan_path=staged_plan,
            armed_path=armed_path,
            output_dir=output_dir,
            timeout_seconds=self.capture_timeout,
            mod_jar=mod_jar,
            staging_marker=staging_marker,
            provenance_expectation=dict(provenance_expectation),
            provenance_mode=self.provenance_mode,
        )

    # -- status bookkeeping --------------------------------------------------

    def _mark(self, scene_id: str, status: str, evidence: Mapping[str, Any]) -> None:
        set_status(self.queue_dir, scene_id, status, evidence=dict(evidence))

    def _fail(self, scene_id: str, stage: str, detail: str,
              evidence: Mapping[str, Any] | None = None) -> SceneOutcome:
        record: dict[str, Any] = {"stage": stage, "detail": detail}
        if evidence:
            record["evidence"] = dict(evidence)
        self._mark(scene_id, "failed", record)
        self._ledger_safe({"event": "scene_end", "scene_id": scene_id, "status": "failed",
                           "stage": stage, "detail": detail, "time": _utc_now()})
        return SceneOutcome(scene_id, "failed", stage, detail, record)

    # -- the run loop --------------------------------------------------------

    def run(self, scene_ids: Iterable[str] | None = None, *,
            resume: bool = False) -> BatchReport:
        """Execute the selected actionable scenes sequentially, in manifest order.

        With ``resume`` a stale ``running`` scene is first reset to ``failed`` by
        :func:`~mc_photogrammetry.dataset.batch.resume_queue`, which is the only
        legal way back from an interrupted run.  A failed scene stops the batch
        unless ``stop_on_failure`` is false; all later scenes are then reported as
        skipped rather than silently dropped.
        """
        if self.dry_run:
            payload = self._load()
            manifest_id = str(payload.get("manifest_id", ""))
            selected = self._scenes(payload, scene_ids)
            outcomes = tuple(
                SceneOutcome(str(scene["scene_id"]), str(scene["status"]), "dry-run",
                             "would run: prepare --arm, capture, finalize")
                if str(scene["status"]) in _ACTIONABLE else
                SceneOutcome(str(scene["scene_id"]), str(scene["status"]), "skip",
                             f"status {scene['status']!r} is not actionable")
                for scene in selected)
            return BatchReport(self.queue_dir, manifest_id, True, outcomes)
        if self.launcher is None:
            raise ExecutorError("a capture launcher is required to execute a queue")

        payload = self._load()
        manifest_id = str(payload.get("manifest_id", ""))
        self._check_runtime_containment(payload)
        selected = self._scenes(payload, scene_ids)
        if self.worker is not None and len(selected) > 1:
            raise ExecutorError(
                "a fixed worker can serve one selected scene only; configure "
                "worker_for_scene for a multi-scene batch")

        self.lock.acquire(manifest_id=manifest_id)
        restore = self._install_interrupt_handlers()
        outcomes: list[SceneOutcome] = []
        stopped = False
        try:
            self._ledger({
                "event": "run_start", "manifest_id": manifest_id,
                "queue_dir": str(self.queue_dir), "pid": os.getpid(),
                "interrupt_event_owner": True, "time": _utc_now(),
                "scenes": [str(scene["scene_id"]) for scene in selected],
            })
            if resume:
                resume_queue(self.queue_dir)
                payload = self._load()
                selected = self._scenes(payload, scene_ids)
            for scene in selected:
                scene_id = str(scene["scene_id"])
                status = str(scene["status"])
                if self.interrupted:
                    outcomes.append(SceneOutcome(
                        scene_id, "skipped", "interrupted",
                        f"{self.interrupt_reason}; not started"))
                    continue
                if status not in _ACTIONABLE:
                    outcomes.append(SceneOutcome(
                        scene_id, "skipped", "select",
                        f"status {status!r} is not actionable"
                        + ("; pass --resume to reset a stale running scene"
                           if status == "running" else "")))
                    continue
                if stopped:
                    outcomes.append(SceneOutcome(
                        scene_id, "skipped", "stop",
                        "a previous scene failed and stop-on-failure is set"))
                    continue
                outcome = self._run_scene(scene_id)
                self._active_scene = None
                outcomes.append(outcome)
                if outcome.status == "failed" and self.stop_on_failure:
                    stopped = True
        except BaseException as error:
            # An abrupt exit must still never leave a scene ``running``.
            if self._active_scene is not None:
                try:
                    self._fail(self._active_scene, "interrupted",
                               f"aborted by {type(error).__name__}: {error}")
                except Exception:
                    pass
                self._active_scene = None
            raise
        finally:
            self._ledger_safe({
                "event": "run_end", "manifest_id": manifest_id,
                "interrupted": self.interrupted, "time": _utc_now(),
                "complete": [item.scene_id for item in outcomes if item.status == "complete"],
                "failed": [item.scene_id for item in outcomes if item.status == "failed"],
                "skipped": [item.scene_id for item in outcomes if item.status == "skipped"],
            })
            restore()
            self.lock.release()

        return BatchReport(self.queue_dir, manifest_id, False, tuple(outcomes),
                           interrupted=self.interrupted)

    def _ledger_command(self, scene_id: str, stage: str,
                        result: CommandResult) -> None:
        self._ledger_safe({
            "event": "command", "scene_id": scene_id, "stage": stage,
            "argv": list(result.argv), "returncode": result.returncode,
            "timed_out": bool(result.timed_out), "interrupted": bool(result.interrupted),
            "duration_seconds": round(float(result.duration_seconds), 3),
            "time": _utc_now(),
        })

    def _run_scene(self, scene_id: str) -> SceneOutcome:
        payload = self._load()
        scene = next((item for item in payload.get("scenes", [])
                      if item.get("scene_id") == scene_id), None)
        if scene is None:
            raise DatasetSchemaError(f"queue has no scene {scene_id!r}")
        dataset = Path(str(payload["dataset"])).resolve()

        self._mark(scene_id, "running", {"stage": "prepare", "executor": "batch-executor"})
        self._active_scene = scene_id
        self._ledger_safe({"event": "scene_start", "scene_id": scene_id,
                           "stage": "prepare", "time": _utc_now()})
        try:
            prepare_argv = self._prepare_argv(scene, dataset)
        except DatasetSchemaError as error:
            return self._fail(scene_id, "prepare", str(error))
        result = self.runner.run(prepare_argv, cwd=self.repo_root,
                                 timeout=self.prepare_timeout)
        self._ledger_command(scene_id, "prepare", result)
        if not result.ok:
            if result.interrupted or self.interrupted:
                return self._fail(scene_id, "interrupted", "prepare interrupted",
                                  result.evidence())
            detail = ("prepare timed out" if result.timed_out else
                      f"prepare exited {result.returncode}")
            return self._fail(scene_id, "prepare", detail, result.evidence())
        if self.interrupted:
            return self._fail(scene_id, "interrupted", self.interrupt_reason)

        # Exact live sequence: prepare -> provision mod -> initialize/verify the
        # scene's immutable worker -> validate options/arm -> desktop preflight and
        # launch (inside the launcher) -> sentinel -> teardown -> finalize.
        staging = Path(str(scene["staging_game_dir"]))
        mod_jar: Path | None = None
        if self.provisioner is not None:
            try:
                mod_jar = self.provisioner.provision(staging)
            except (ProvisionError, OSError) as error:
                return self._fail(scene_id, "provision", str(error))
            self._ledger_safe({
                "event": "provision", "scene_id": scene_id,
                "capture_mod": str(mod_jar), "time": _utc_now(),
            })
        if self._has_worker():
            try:
                worker = self._worker_for(scene_id)
                worker.ensure_ready()
                target = worker.ensure_target(staging)
            except (WorkerError, JunctionError, OSError) as error:
                return self._fail(scene_id, "worker", str(error))
            self._ledger_safe({
                "event": "worker", "scene_id": scene_id,
                "worker_dir": str(worker.worker_dir),
                "minecraft_target": str(target), "time": _utc_now(),
            })
        if self.interrupted:
            return self._fail(scene_id, "interrupted", self.interrupt_reason)

        try:
            request = self._capture_request(scene, dataset, expected_mod_jar=mod_jar)
        except DatasetSchemaError as error:
            return self._fail(scene_id, "arm", str(error))
        if self.interrupted:
            return self._fail(scene_id, "interrupted", self.interrupt_reason)

        try:
            capture = self.launcher.launch(request)
        except DatasetSchemaError as error:
            return self._fail(scene_id, "capture", str(error))
        self._ledger_safe({
            "event": "capture", "scene_id": scene_id, "stage": capture.stage,
            "completed": bool(capture.completed), "detail": capture.detail,
            "time": _utc_now(),
        })
        if self.interrupted or capture.stage == "interrupted":
            return self._fail(scene_id, "interrupted",
                              capture.detail or self.interrupt_reason, capture.evidence)
        if not capture.completed:
            return self._fail(scene_id, capture.stage or "capture",
                              capture.detail, capture.evidence)

        try:
            finalize_argv = self._finalize_argv(scene, dataset)
        except DatasetSchemaError as error:
            return self._fail(scene_id, "finalize", str(error))
        result = self.runner.run(finalize_argv, cwd=self.repo_root,
                                 timeout=self.finalize_timeout)
        self._ledger_command(scene_id, "finalize", result)
        if not result.ok:
            if result.interrupted or self.interrupted:
                return self._fail(scene_id, "interrupted", "finalize interrupted",
                                  result.evidence())
            detail = ("finalize timed out" if result.timed_out else
                      f"finalize exited {result.returncode}")
            return self._fail(scene_id, "finalize", detail, result.evidence())

        record = {"stage": "finalize", "executor": "batch-executor",
                  "capture": dict(capture.evidence)}
        self._mark(scene_id, "complete", record)
        self._ledger_safe({"event": "scene_end", "scene_id": scene_id,
                           "status": "complete", "stage": "finalize",
                           "time": _utc_now()})
        return SceneOutcome(scene_id, "complete", "finalize", "scene finalized", record)


def render_report_lines(report: BatchReport) -> list[str]:
    """Deterministic, encoding-safe summary lines for a batch report."""
    mode = "dry run" if report.dry_run else "run"
    lines = [f"batch executor {mode}: manifest {report.manifest_id}",
             f"queue {report.queue_dir}"]
    for outcome in report.outcomes:
        line = f"{outcome.scene_id}  [{outcome.status}]  stage={outcome.stage}"
        if outcome.detail:
            line += f"  {outcome.detail}"
        lines.append(line)
    lines.append("complete: " + (", ".join(report.completed) or "none"))
    lines.append("failed: " + (", ".join(report.failed) or "none"))
    lines.append("skipped: " + (", ".join(report.skipped) or "none"))
    if report.interrupted:
        lines.append("interrupted: yes")
    return lines


def write_report(report: BatchReport, stream: Any = None) -> None:
    """Write :func:`render_report_lines` through the encoding-safe writer."""
    target = sys.stdout if stream is None else stream
    for line in render_report_lines(report):
        write_line(target, line)
