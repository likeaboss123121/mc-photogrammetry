# DS-31 - safe sequential Prism batch executor

> **Status: DS-31 and DS-31R were rejected; DS-31S (corrective) was then
> repaired by DS-31T after lead review, and its first real pilot then exposed one
> more defect, repaired by DS-31U. The DS-31U revision is the current one.** The
> DS-31/DS-31R implementation below is retained for provenance and its tested
> lifecycle/lock/ledger behavior is kept, but it was rejected for live use
> because it had no production answer for the worker game-directory junction,
> the disposable Prism worker, the hidden-desktop isolation, or the
> capture-mod provisioning. DS-31S implemented those four areas, but three
> narrow defects survived lead review: the junction script did not re-prove
> containment itself, the failed worker initialization used a recursive delete,
> and the desktop preflight did not refuse the operator's own target desktop.
> See "DS-31S corrective implementation" and "DS-31T narrow lead-review repair"
> at the end of this report for the accepted contract and the exact evidence.
> The DS-31U section records the first live pilot's failed attempt and the
> canonical Windows path-key repair. The DS-31 text in between is the original
> submission and is **not** a current acceptance claim.

Date: 2026-09-12. Worker: DeepSeek DS-31. Contract source: the direct worker
prompt ("implement the missing safe sequential Prism batch executor") plus the
accepted DS-17/20C/21/22 factory and the accepted DS-27 launch-free queue.
Scope: the runner DS-27 deliberately omitted. No reconstruction, Java, evaluator,
threshold, existing artifact, or git-index change; no Minecraft launch, install,
staging, or arming was performed; no source save was opened for write.

## Why this is the missing piece

DS-27's `docs/reports/DS-27-bulk-capture-queue-2026-09-12.md` states the honest
limit plainly: "The queue never executes a command it prints. A runner (the lead)
advances statuses with `--mark`/`--resume`; nothing here can observe a real
capture." The queue already carries everything a runner needs - content-addressed
`prepare_argv`/`finalize_argv`, per-scene staging game dirs, validated statuses and
attempt evidence - but nothing turned a `pending` scene into a capture. DS-31 is
that runner: prepare -> arm/verify -> Prism capture -> finalize -> status, one
scene at a time.

## Deliverables

| Path | Purpose |
| --- | --- |
| `src/mc_photogrammetry/dataset/executor.py` (new) | `BatchExecutor`, bounded `SubprocessRunner`, `CaptureLauncher` protocol, `PrismInstance`/`PrismCaptureLauncher`, verified arming, status bookkeeping, report |
| `scripts/run_capture_batch.py` (new) | CLI: dry run by default; `--execute`, `--resume`, `--scene`, `--continue-on-failure`, Prism/instance configuration, timeouts, `--json` |
| `tests/test_capture_batch_executor.py` (new) | 34 focused tests (orchestration, pre-launch refusal, resume, Prism instance/launcher with fake processes, command runner, CLI) |
| `src/mc_photogrammetry/dataset/__init__.py` | Public exports for the executor surface (additive only) |
| `docs/reports/DS-31-batch-executor-2026-09-12.md` | This report |

## Per-scene pipeline

For each actionable scene (manifest order, `pending`/`failed` only):

1. **mark running** through DS-27's validated `set_status`, with
   `{"stage": "prepare", "executor": "batch-executor"}`.
2. **prepare** - run the queued `prepare_capture_dataset.py` argv with `--arm`
   appended and `argv[0] == "python"` replaced by the configured interpreter. This
   stages the world, freezes the plan and writes `armed.json` in one step.
3. **verify before launch** (see below); a failure here is stage `arm`, never a
   launch.
4. **capture** - hand a `CaptureRequest` to the injected `CaptureLauncher`. The
   concrete `PrismCaptureLauncher` starts Prism, waits for the mod's terminal
   `run_status.json`, and stops the process tree.
5. **finalize** - run the queued `finalize_capture_dataset.py` argv; only a zero
   exit marks the scene `complete`.

Any failure marks `failed` with the stage, a detail string, and bounded command
output (or launcher evidence) as attempt evidence, so `--resume` can retry it.
With the default `stop_on_failure`, later scenes are reported `skipped`, not
silently dropped.

## Safety invariants

* **Sequential.** One scene is prepared, captured and finalized at a time; the
  executor never starts a second command or game process concurrently.
* **Fail closed before launch.** `_capture_request` refuses the scene unless all
  of: the plan written by prepare parses and names the queued scene/world; the
  staged plan file exists and its SHA-256 equals the armed `plan_sha256`; the
  armed marker exists as a JSON object and its `plan_id`, non-blank
  `staging_token` and `world_identity` match the prepared plan. A missing or
  tampered marker is a scene failure, not a warning.
* **Wrong-world refusal.** `PrismInstance.verify_game_dir` resolves the
  instance's `minecraft` path (a directory junction in practice) and refuses to
  launch unless it is exactly the scene's staging directory. The executor never
  creates or repoints an instance link; the operator owns that, precisely because
  a Windows junction is not reported by `Path.is_symlink()` and a naive
  `rmtree` on it would delete the target tree.
* **Content-addressed commands.** The argv executed is the argv `load_queue`
  already validated; before execution `--dataset`, `--staging-game-dir`, `--save`
  and `--scene` are re-checked against the queue record.
* **Bounded.** Every factory command has a timeout (`--prepare-timeout`,
  `--finalize-timeout`) and the capture run has `--capture-timeout`; a timed-out
  command is killed and recorded with `timed_out: true`, and a timed-out capture
  stops the process tree and fails the scene.
* **Clean-capture gate.** The launcher accepts only `run_status.json` with
  `run_failed` not true and `ground_truth_exported is true` (the mod's own proof
  of a complete export, written atomically before disarm). A failed run_status
  fails the scene without spending a finalize run.
* **Dry run by default.** `BatchExecutor(dry_run=True)` and the CLI without
  `--execute` write nothing, run nothing, and launch nothing; they print the
  ordered stages plus the exact prepare/capture/finalize commands.
* **No source-save or queue rewrite.** Only `set_status`'s status/attempt fields
  and the dataset/staging outputs the factory scripts own are written; `--execute`
  additionally requires an explicit Prism executable, so a misconfigured run
  cannot silently fall back to anything.

## Prism adapter

`PrismInstance(instance_id, instance_dir, executable, prism_root)` builds the
launch command actually supported by the installed launcher
(`prismlauncher.exe -d <root> -l <instance> -w <world>`, verified with
`--help`), and `PrismCaptureLauncher` resolves one instance per scene via a
resolver. The CLI exposes two ways to supply it:

* `--prism-instance-root DIR` with `--prism-instance-template '{scene_id}'`
  (default), so scene `alpha-house` resolves to
  `<root>/alpha-house/minecraft`; or
* `--prism-instance-map FILE` mapping scene id -> instance id (or an
  `{"instance_id", "instance_dir"}` object) for explicit control.

The launcher polls for `<staging>/mc-photo-capture/out/<plan_id>/run_status.json`
at `--poll-seconds`, then terminates the process tree (`taskkill /T /F` on
Windows) with a bounded grace. `describe()` lets a dry run print the exact Prism
command without verifying or starting anything.

## CLI

```
python scripts/run_capture_batch.py --spec <batch.json>          # dry run
python scripts/run_capture_batch.py --spec <batch.json> --execute \
    --prism-executable <prismlauncher.exe> --prism-root B:/prismRoot \
    --prism-instance-root B:/prismInstances [--resume] [--scene ID] \
    [--continue-on-failure] [--json]
```

`--spec` locates the queue directory exactly as `plan_capture_batch.py` does
(spec `queue_dir`, else `--queue-dir`, else `<dataset>/queue`). Exit code is 1 if
any selected scene failed, else 0. Output goes through the DS-26 encoding-safe
console writer.

## Evidence

```
.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch_executor.py \
    -o addopts= -p no:cacheprovider -q
-> exit 0, 34 passed in 1.23s

.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch_executor.py \
    tests/test_capture_batch.py tests/test_capture_dataset.py \
    tests/test_capture_options.py tests/test_capture_console.py \
    tests/test_litematic.py tests/test_camera.py tests/test_project_store.py \
    -o addopts= -p no:cacheprovider -q
-> exit 0, 181 passed, 1 skipped in 9.30s
   (the skip is the pre-existing external-symlink staging case)

.\.venv\Scripts\python.exe -c "import mc_photogrammetry.dataset as d; ..."
-> every name in __all__ resolves; executor surface importable
```

Real pilot dry run against the materialized DS-27 queue (read-only, nothing
written or launched):

```
python scripts/run_capture_batch.py \
    --spec exports/datasets/pilot-batch-train-station-2026-09-12/batch-spec.json \
    --prism-executable <prismlauncher.exe> --prism-root B:/prismRoot \
    --prism-instance-root B:/prismInstances
-> exit 0; two actionable scenes (orbit-a 6 views seed 1702, orbit-b 8 views seed 2718)
   each shown as prepare(..., --allow-overwrite, --arm) /
   prismlauncher.exe -d B:/prismRoot -l <scene_id> -w "Train station" /
   finalize(... --scene <scene_id> --allow-overwrite)
```

The 34 new cases cover: exact prepare/capture/finalize ordering and argv
(interpreter substitution, appended `--arm`, re-validated `--dataset`/
`--staging-game-dir`/`--save`); dry run writing and launching nothing; prepare,
capture, finalize and timeout failures each failing at the right stage with
evidence and (by default) skipping later scenes; `--continue-on-failure`;
missing armed marker, token mismatch, plan-id mismatch and staged-plan digest
mismatch all refused before any launcher call; complete/running selection and
`resume=True`; unknown scene refusal; execute-without-launcher; the Prism
instance game-dir contract and `-l/-w` argv; the launcher's clean/failed/
timed-out/foreign-plan/unrelated-instance paths with fake processes and an
injected stopper; the real `SubprocessRunner` (zero and nonzero exits, bounded
tail); and the CLI dry run, execute-without-Prism refusal, `--prism-instance-map`
resolution/missing-file refusal, and a full injected execute run.

## Honest limits

* **No real launch was performed.** The publisher/mod contract is exercised with
  fake processes and synthesized `run_status.json`; whether a given Prism build
  honours `-d/-l/-w` exactly, and how many ticks the game takes to write the
  sentinel, are unverified here. The first live use should be one scene.
* **One Prism instance per scene is operator setup.** Because the executor
  refuses to create or repoint `minecraft` links, a two-scene batch needs two
  instances whose junctions point at the two staging directories (or an instance
  map). A single shared capture instance with a re-pointed junction is
  deliberately not automated; doing so safely requires reparse-point-aware
  deletion, which is not implemented or tested.
* **The capture step is fail-closed but not interrupt-proof.** A `Ctrl-C` while a
  scene is `running` leaves it `running`; that is exactly the state
  `resume_queue`/`--resume` resets to `failed`, so the documented recovery path
  is required rather than an automatic rewrite during interruption.
* **`taskkill /T /F` is Windows-specific.** Non-Windows falls back to
  `Popen.terminate()` on the launcher process only; a POSIX launcher wrapper that
  does not forward the signal could leave a Java child. This workspace is Windows.
* **The mod is not asked to quit.** The launcher stops the process tree once
  `run_status.json` proves the capture is complete; the staged world is
  throwaway and finalize reads only screenshots/JSON, but any in-flight world
  write is abandoned rather than flushed.
* **No production/promotion claim.** This is developer/evaluation data
  generation tooling; ground truth still never enters normal reconstruction, and
  the queue's declared split is still not the authoritative split.

## State left behind

Only new paths plus the additive `dataset/__init__.py` export block were touched:
`src/mc_photogrammetry/dataset/executor.py`,
`scripts/run_capture_batch.py`, `tests/test_capture_batch_executor.py`,
`docs/reports/DS-31-batch-executor-2026-09-12.md`, and the executor import/export
block in `src/mc_photogrammetry/dataset/__init__.py`. `batch.py`, `console.py`,
the factory scripts, the Java mod, the evaluator, thresholds, and every existing
report/artifact are unchanged. The pilot queue manifest and its scenes were read
only and remain `pending`; no source save, live `.minecraft`, staging world,
game process, or git stage/commit was created. No comms entry was appended by
this worker (shared coordination file preserved); the lead records the result.

# DS-31S - corrective implementation

Date: 2026-09-13. Worker: DeepSeek DS-31S. This replaces the rejected DS-31 and
DS-31R submissions. Scope: the four named implementation areas plus the CLI
contract; no new features and no launch, install, Minecraft, desktop, junction or
git action was performed.

## Why DS-31/DS-31R were rejected

DS-31 was rejected, and DS-31R (which added the single-flight lock and the
fsynced run ledger) remained rejected, because the pair did not implement the
production lifecycle a live capture needs. Concretely:

* **No worker/junction management.** The executor explicitly refused to create or
  repoint an instance's `minecraft` link, so a two-scene batch needed two
  operator-made Prism instances whose junctions already pointed at the two
  staging directories. There was no validated, reparse-point-aware way to point
  one worker at each scene, and no owner for the "existing `minecraft` is a real
  directory / a symlink / a foreign junction" cases.
* **No disposable worker.** There was no manager to initialize a worker from a
  template without mutating the template, to mark it, or to reuse one worker for
  every scene; `--prism-instance-root`/`--prism-instance-map` produced a
  *different* instance per scene.
* **No hidden-desktop isolation.** Nothing preflighted a second virtual desktop
  or moved only the game's own windows there, so a launched client sat on the
  operator's desktop and could steal focus/input.
* **No capture-mod provisioning.** The executor only globbed whatever
  `mc-photo-capture-*.jar` happened to be in the staging `mods` directory and
  launched with the first match; `--execute` had no `--capture-mod`/hash pin, so
  the exact jar that ran was unverified and effectively defaulted.
* **No enforced sequence.** There was no exact
  prepare -> provision -> worker -> validate/arm -> desktop -> launch order to
  pin, and the DS-31/DS-31R tests used real processes (`taskkill`) rather than
  pure fakes, so they were not hermetic.

## DS-31S deliverables

| Path | Change |
| --- | --- |
| `src/mc_photogrammetry/dataset/executor.py` | `LinkState`/`LinkInspector`/`WindowsLinkInspector`, `JunctionAdapter`/`WindowsJunctionAdapter` + one repoint script, `DisposablePrismWorker`, `CaptureModProvisioner`, `DesktopPreflight`/`DesktopController`/`WindowsDesktop2Controller` + preflight/move scripts, launcher desktop hook, executor provision/worker sequence, stricter mod-jar check |
| `scripts/run_capture_batch.py` | `--worker-root --worker-dir --worker-instance-id --worker-template --capture-mod --capture-mod-sha256`; `--execute` requires all six; execute uses one worker; per-scene instance options are dry-run-only |
| `tests/test_capture_batch_executor.py` | 105 tests (was 57); new hermetic tests for the four areas, the sequence, dry run, and the CLI contract |
| `src/mc_photogrammetry/dataset/__init__.py` | Additive exports for the new public surface |
| `docs/reports/DS-31-batch-executor-2026-09-12.md` | This rejection note and corrective section |

## (1) `WindowsJunctionAdapter`

`WindowsJunctionAdapter(worker_root=..., target_root=..., inspector=..., runner=...,
script_dir=..., timeout=...)` is the only component allowed to create or repoint a
worker `minecraft` entry.

* **Containment.** `repoint(link, target)` refuses a `link` that is not strictly
  under the explicit `worker_root` (checked lexically via `os.path.abspath` and a
  case-normalized prefix test, because resolving a junction would follow it to
  the target) and a `target` that is not under the explicit `target_root` (the
  queue's `dataset/staging-game`). The target must exist, be a directory and not
  itself be a link/reparse point.
* **Existing entry.** `WindowsLinkInspector` classifies the link from `lstat`:
  absent, `IO_REPARSE_TAG_MOUNT_POINT` junction, `IO_REPARSE_TAG_SYMLINK`
  symlink, other reparse point, real directory or real file. Only *absent* and
  *directory junction* are allowed; a real directory, a symlink and an unknown
  link are refused. A junction whose resolved target is not under `target_root`
  ("uncontained old target") is refused rather than removed.
* **Repoint script (DS-31S; corrected by DS-31T).** Exactly one script is
  invoked:
  `powershell.exe -NoProfile -ExecutionPolicy Bypass -File <script> <link> <target>`
  with both paths as positional arguments (never interpolated into the script
  text). The script `Get-Item -LiteralPath` reparse-checks any existing entry,
  `Remove-Item -LiteralPath` without `-Recurse` (the old target tree is never
  walked), `New-Item -ItemType Junction`, then `Resolve-Path -LiteralPath`
  compares the created link's `ProviderPath` with the target and throws on any
  mismatch. After the process, the Python side re-inspects the link and requires
  a junction whose target equals the requested absolute target.

  **Correction (DS-31T).** The DS-31S form above overclaimed. It passed only
  `<link> <target>` and the script performed **no** containment check of its
  own, so containment was enforced only in the Python process; and its final
  `Resolve-Path -LiteralPath $Minecraft` compared the link's own path with the
  target (PowerShell returns the link for a junction, not the junction target),
  which can never match, so a real repoint could not succeed. DS-31T replaces
  this with a four-argument script (link, target, worker root, target root)
  that canonicalizes all four paths and re-proves containment inside the same
  PowerShell process, immediately before removal, and verifies the created
  junction's own `Target` property. See the DS-31T section for the exact
  contract and evidence. The DS-31S behavior retained unchanged is
  `Get-Item -LiteralPath` reparse-checking, `Remove-Item -LiteralPath` without
  `-Recurse`, and `New-Item -ItemType Junction`, plus the Python-side
  re-inspection.

## (2) `DisposablePrismWorker`

`DisposablePrismWorker(worker_root, worker_dir, instance_id, template_dir,
executable, prism_root, junction)`:

* **Refusals.** Template equal to the worker, a worker outside `worker_root`, a
  worker and template that contain each other, an unsafe instance id, an
  instance id that is not the worker folder name (Prism resolves `-l` by folder),
  and an existing worker directory with no/invalid
  `.mc-photo-capture-worker.json` marker are all refused.
* **Initialization.** A missing worker is created by copying **only**
  `instance.cfg` and `mmc-pack.json` (regular, non-link template files) through a
  same-directory temporary file and `os.replace`. The copied `instance.cfg` has
  its `name=` rewritten to the instance id and a fresh `uuid4().hex` written
  (missing keys appended, every other line and the newline style preserved),
  then the marker is written atomically (canonical JSON) recording schema
  version, instance id, worker dir, template dir, uuid and creation time. The
  template's `minecraft` directory is never read, copied or written; a failed
  initialization removes the half-created worker rather than leaving an unmarked
  directory that could never be reused.

  **Correction (DS-31T).** DS-31S implemented that removal with
  `shutil.rmtree(ignore_errors=True)`, a recursive delete inside the
  junction-aware executor. DS-31T replaces it with a bounded, non-recursive
  cleanup that unlinks only `instance.cfg`, `mmc-pack.json`, the worker marker
  and their `.tmp` siblings and then calls `rmdir`, after proving the worker
  directory was the one this call freshly created and still sits in the worker
  root. No DS-31 file performs a recursive delete. See the DS-31T section.
* **Reuse.** `ensure_ready()` validates an existing marker and is idempotent;
  `repoint(staging)` points the same `minecraft` junction at each scene's staging
  directory before launch. The executor calls `ensure_ready()` then `repoint()`
  for every scene, so one worker serves the whole batch.

## (3) `WindowsDesktop2Controller`

* **Preflight.** `preflight()` runs one bypass-policy script that
  `Import-Module VirtualDesktop -ErrorAction Stop`, requires `Get-DesktopCount >=
  2`, and remembers `Get-DesktopIndex(Get-CurrentDesktop)`. The launcher calls it
  before `Popen`; a failure returns a `desktop`-stage `CaptureOutcome` **without
  starting a process**, and no scene can launch on a machine without a second
  desktop.

  **Correction (DS-31T).** DS-31S only required a second desktop and remembered
  the current index; it did not refuse the case where the operator was already
  on the target desktop 1, so the launcher could move the client onto the
  operator-visible desktop. DS-31T makes both the script and the Python
  preflight fail closed when the current index equals 1. See the DS-31T section.
* **Move.** After `Popen` and on every bounded poll, `move_windows()` runs one
  script that iterates `Get-Process -Name PrismLauncher,java,javaw`, ignores a
  zero `MainWindowHandle`, and calls
  `Move-Window -Desktop (Get-Desktop 1) -Hwnd $handle` for every nonzero handle;
  it then re-reads the current desktop index and throws if it changed, and the
  Python side requires that index to equal the remembered preflight index. A
  failed move or a changed index tears the process tree down and fails the scene
  at stage `desktop`.
* **Prohibited calls.** No script or argv contains `Switch-Desktop`,
  `Move-ActiveWindow`, `Get-ActiveWindowHandle`, `SetForegroundWindow`,
  `Activate`, `Focus`, `SendKeys`, or any mouse/keyboard API; a test audits both
  the scripts and the argv case-insensitively.

## (4) `CaptureModProvisioner`

`--execute` requires `--capture-mod` and `--capture-mod-sha256` (64 lowercase
hex). The provisioner refuses a missing/non-regular source, a source name outside
`mc-photo-capture-*.jar`, a source whose SHA-256 differs from the pin, a staging
`mods` directory that already holds another capture jar, and a destination whose
post-copy hash differs. On success it copies the source to a same-directory
temporary file, verifies the temporary, `os.replace`s it to
`staging/mods/<source.name>`, re-verifies the destination, and re-checks that
exactly the provisioned jar remains. There is no hardcoded default jar, and
`--execute` builds the provisioner only from explicit CLI arguments.

## Post-prepare options (unchanged contract)

The final pre-launch check still requires exactly `pauseOnLostFocus:false`,
`rawMouseInput:false` and `soundCategory_master:0.0`. A new test pins
`CAPTURE_OPTIONS` to that exact mapping and parametrizes a wrong pause, a wrong
master volume and a missing key, all of which refuse before any launcher call.

## CLI contract

```
python scripts/run_capture_batch.py --spec <batch.json>          # dry run
python scripts/run_capture_batch.py --spec <batch.json> --execute \
    --prism-executable <prismlauncher.exe> [--prism-root B:/prismRoot] \
    --worker-root B:/capture-workers --worker-dir mc-photo-capture \
    --worker-instance-id mc-photo-capture --worker-template B:/prismInstances/template \
    --capture-mod build/libs/mc-photo-capture-0.1.0.jar \
    --capture-mod-sha256 <64 lowercase hex> [--resume] [--scene ID] \
    [--continue-on-failure] [--json]
```

`--execute` refuses a missing `--prism-executable`, each of the six worker/mod
arguments when absent, and the old per-scene `--prism-instance-root`/
`--prism-instance-map` (dry-run-only, where it still renders the legacy command).
Execute constructs exactly one worker and one provisioner and uses them for every
scene. A dry run with the worker arguments renders the worker instance command but
creates no worker, junction, desktop controller script, provisioned mod, lock or
ledger.

## Exact sequence (DS-31S item 7)

For each actionable scene, in order: mark `running`; run prepare `--arm`;
provision the pinned mod; `ensure_ready()`/`repoint()` the one worker; validate
options/plan/armed/in-world marker; desktop preflight; `Popen`; move windows after
`Popen` and on every poll; wait for the fresh contract-matched
`run_status.json`; confirm teardown; run finalize. The executor records
`provision`/`worker` ledger receipts, and a failure at any step marks the scene
`failed` at that stage without reaching the next one.

## Evidence

```
.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch_executor.py \
    -o addopts= -p no:cacheprovider -q
-> exit 0, 105 passed in 3.97s

.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch_executor.py \
    tests/test_capture_batch.py tests/test_capture_dataset.py \
    tests/test_capture_options.py tests/test_capture_console.py \
    tests/test_litematic.py tests/test_camera.py tests/test_project_store.py \
    -o addopts= -p no:cacheprovider -q
-> exit 0, 252 passed, 1 skipped in 12.09s
   (the skip is the pre-existing external-symlink staging case)

.\.venv\Scripts\python.exe -m pytest -o addopts= -p no:cacheprovider -q
-> exit 0, 919 passed, 1 skipped in 186.85s
```

The 48 new cases (105 total, from 57) pin: the exact junction containment/
refusal matrix (worker out of root, target out of root, real dir, symlink,
unknown link, uncontained old junction, PowerShell failure, no junction created,
resolved-target mismatch); the one-script positional `-File` invocation and its
`Get-Item`/`Remove-Item` no-`-Recurse`/`New-Item -ItemType Junction`/
`Resolve-Path` contents; worker initialization copying only `instance.cfg` and
`mmc-pack.json`, a fresh uuid per initialization, an unmodified template, marker
idempotence and marker-mismatch refusal, template-equal/out-of-root/unmarked
refusals and per-scene repoint; provisioner success plus missing/hash-mismatch/
bad-pin/bad-name/multiple-jar refusals; the exact options mapping; the desktop
preflight count/index contract, exact move script, current-desktop invariant,
preflight-before-start failure, move-after-Popen/per-poll and failed-move
teardown, and a forbidden-token audit; the exact
prepare->provision->repoint->launch order with one worker for two scenes; worker
and provisioner failure stages; dry-run zero mutation; and the CLI
missing-argument/per-scene-refusal/dry-run/end-to-end behavior.

## Deviations from the literal instruction (recorded honestly)

* **`-File` with positional parameters.** The instruction named
  `powershell.exe -NoProfile -ExecutionPolicy Bypass` with the paths as
  positional arguments and one script. Windows PowerShell's `-Command <string>`
  cannot carry positional arguments safely: any token after the command string
  is re-parsed as command text, so a path containing a space is split (verified
  locally with `powershell.exe` 5.1). The script is therefore materialized once
  in a temp directory and invoked with `-File <script> <link> <target>`, which is
  the only form that keeps both paths positional and inert. This preserves the
  intent (bypass policy, one script, positional paths) while being correct for
  real paths.
* **Script files for the desktop helpers.** The desktop preflight/move scripts
  are likewise invoked with `-File` for the same reason; they take no arguments.

## Honest limits

* **No live verification.** No real junction, Prism instance, virtual desktop,
  Minecraft process or mod jar was touched. The scripts' PowerShell semantics are
  reasoned from the documented cmdlets and a small local argument-passing probe,
  not from a live capture; the first live use should still be one scene with the
  operator watching.
* **Junction classification is Windows-specific.** `WindowsLinkInspector` reads
  `st_reparse_tag`; on a non-Windows host it can only see symlinks and will refuse
  a junction. This workspace is Windows.
* **The worker folder is created but not otherwise managed.** Initialization from
  the template is one-shot; a worker whose copied `instance.cfg` is later removed
  fails at the Prism `instance.cfg`/`minecraft` check. Repair is an operator
  action (delete the marked worker directory and rerun).
* **Desktop isolation cannot be proven without the VirtualDesktop module.** If the
  module is absent or the machine has one desktop, `--execute` fails closed at the
  preflight rather than launching on the operator's desktop.
* **The mod hash pin is only as good as the operator's pin.** The provisioner
  proves the bytes match the supplied SHA-256; it does not independently know
  which jar is the "right" build.

## State left behind (DS-31S)

Only these paths were edited: `src/mc_photogrammetry/dataset/executor.py`,
`scripts/run_capture_batch.py`, `tests/test_capture_batch_executor.py`,
`src/mc_photogrammetry/dataset/__init__.py` (additive exports),
`docs/reports/DS-31-batch-executor-2026-09-12.md`, and this worker's appended
`AGENT_COMMS.md` note. `batch.py`, the factory scripts, the Java mod, the
evaluator, thresholds, and every existing report/artifact are unchanged. No real
junction, worker, desktop switch, Minecraft or Prism process, mod copy into live
staging, source-save write, or git stage/commit occurred. Every DS-31S test uses
temporary directories and injected fakes only; the only real subprocesses in the
file are the pre-existing short-lived Python children that unit-test
`SubprocessRunner` itself.

# DS-31T - narrow lead-review repair

Date: 2026-09-13. Worker: DeepSeek DS-31T. Scope: exactly the three defects
named by the lead review of DS-31S, plus their tests and this report/comms
record. No new features, no refactor, no launch/install/Minecraft/desktop
action, and no git stage/commit.

## Defect 1 - the junction script did not enforce containment itself

DS-31S validated the link and target in Python and passed only `<link> <target>`
to one PowerShell script. The script therefore had no in-process containment:
if the Python check were bypassed (or the entry changed between the check and
`Remove-Item`), it would still delete an existing reparse point pointing
anywhere. Its final `Resolve-Path -LiteralPath $Minecraft` comparison was also
wrong: PowerShell's `Resolve-Path` on a junction returns the link path, not the
junction target, so `$resolved -ne $wanted` was true for every real repoint and
the script could never report success.

DS-31T changes the contract to four positional arguments and moves containment
into the same process:

* `JUNCTION_REPOINT_SCRIPT` now declares `$Minecraft`, `$Target`, `$WorkerRoot`
  and `$TargetRoot` and receives all four from Python as inert `-File`
  arguments (never interpolated into the script text).
* The script canonicalizes all four paths (`Get-NormalizedPath`:
  `[System.IO.Path]::GetFullPath`, case-insensitive comparison), requires the
  link **strictly** below `$WorkerRoot`, and requires `$Target` within
  `$TargetRoot`.
* If an entry already exists, it keeps the `ReparsePoint` and
  `LinkType -eq "Junction"` checks, then re-reads that junction's own `Target`
  property, canonicalizes it, and requires it within `$TargetRoot`
  **immediately before** `Remove-Item -LiteralPath ... -Force` (no `-Recurse`).
* It creates the link with `New-Item -ItemType Junction`, then verifies the
  created junction's own `Target` property equals the requested target. The
  broken `Resolve-Path` cross-check is removed; the created `Target` comparison
  is the verification.
* Python still validates lexically first and still re-inspects the result
  afterwards, so both layers must agree.

New tests pin the four-argument argv (`argv[6]`/`argv[7]` link/target,
`argv[8]`/`argv[9]` worker/target roots), assert the script text carries the
canonicalization and both containment checks, assert the old-target check sits
between the `.Target` read and `Remove-Item`, and simulate a script-side refusal
(Python inspection passed, the script's re-read refused) surfacing as a
`JunctionError`.

Local, non-live check (temporary workspace directories only): the script parsed
cleanly under Windows PowerShell; create and repoint to the requested target
returned exit 0 and the created `Target` was correct; a link outside the worker
root, a target outside the target root, and an existing junction whose own
`Target` was outside the target root were all refused (exit 1) and the refused
junction was left untouched.

## Defect 2 - recursive cleanup in a failed worker initialization

DS-31S cleaned a failed `DisposablePrismWorker._initialize()` with
`shutil.rmtree(self.worker_dir, ignore_errors=True)`, a recursive delete inside
code whose entire purpose is to be reparse-point-aware.

DS-31T replaces it with `_discard_failed_initialize(created=...)`:

* it refuses to act unless the directory was freshly created by this call and
  still sits strictly under the worker root;
* it lstat-checks the directory, leaving a symlink/reparse point or a
  non-directory untouched;
* it unlinks only the files this method can create - `instance.cfg`,
  `mmc-pack.json`, `WORKER_MARKER_NAME`, and each of their `.tmp` atomic-write
  siblings - and then calls `rmdir()`. There is no recursion: an unexpected
  entry makes `rmdir` fail and the tree is left for an operator.

New tests force a mid-initialization failure before `mmc-pack.json` is copied,
force a failure that leaves an atomic temporary, and force a failure with an
unexpected entry present to prove the cleanup does not recursively delete it.
A static test scans `executor.py`, `run_capture_batch.py` and the test file for
any recursive-delete token and checks the junction script for `-Recurse`.

## Defect 3 - desktop preflight did not refuse the target desktop

The hidden capture desktop is index 1. DS-31S only required `Get-DesktopCount
>= 2` and remembered the current index, so an operator already working on
desktop 1 would have the client moved onto the desktop they are looking at.

DS-31T adds `TARGET_DESKTOP_INDEX = 1`; the preflight script now throws when
`$index -eq 1` **before** printing its state, and `WindowsDesktop2Controller.
preflight()` defensively refuses the same index in Python without storing a
preflight state. The move script's `$before`/`$after` equality check and the
Python `move_windows()` requirement that the index equal the remembered
preflight index are unchanged.

New tests drive a preflight report of index 1 through the controller (refused,
`preflight_state is None`) and through the launcher (a `desktop`-stage outcome
with no `Popen`).

## Evidence (DS-31T)

```
.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch_executor.py \
    -o addopts= -p no:cacheprovider -q
-> exit 0, 114 passed in 4.12s   (was 105; +9 cases)

.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch_executor.py \
    tests/test_capture_batch.py tests/test_capture_dataset.py \
    tests/test_capture_options.py tests/test_capture_console.py \
    tests/test_litematic.py tests/test_camera.py tests/test_project_store.py \
    -o addopts= -p no:cacheprovider -q
-> exit 0, 261 passed, 1 skipped in 13.04s
   (was 252 passed/1 skipped; the skip is the pre-existing external-symlink case)

.\.venv\Scripts\python.exe -m pytest -o addopts= -p no:cacheprovider -q
-> exit 0, 928 passed, 1 skipped in 210.74s   (was 919 passed/1 skipped)
   A second full run on the final revision, after the `Resolve-Path` tail
   correction, reproduced 928 passed, 1 skipped in 205.96s.
```

## Honest limits (DS-31T)

* **No live capture verification.** No Prism, Minecraft, mod jar, virtual
  desktop or real capture used the executor. The junction script was exercised
  only against temporary directories created and removed for the check; the
  desktop scripts were parsed and unit-tested with an injected runner, never
  run against a real VirtualDesktop module.
* **The junction script still cannot make the check-and-remove atomic.** The
  old junction's `Target` is re-read immediately before removal inside one
  PowerShell process, which is the strongest check available without holding a
  kernel handle; a determined concurrent actor could still race the final
  `Remove-Item`. In the intended single-operator flow the worker root is not
  writable by anything else.
* **`-File` with positional arguments remains the invocation.** Windows
  PowerShell `-Command` cannot carry positional arguments without re-parsing
  them, so the DS-31S deviation stands; DS-31T only extends it from two to four
  positional paths.

## State left behind (DS-31T)

Only these paths were edited: `src/mc_photogrammetry/dataset/executor.py`,
`tests/test_capture_batch_executor.py`,
`docs/reports/DS-31-batch-executor-2026-09-12.md`, and this worker's appended
`AGENT_COMMS.md` note. `scripts/run_capture_batch.py`,
`src/mc_photogrammetry/dataset/__init__.py`, `batch.py`, the factory scripts,
the Java mod, the evaluator, thresholds, and every existing artifact are
unchanged. No Prism/Minecraft process, virtual desktop, source-save write, or
git stage/commit occurred. The only junction objects created were four
temporary ones under a workspace scratch directory for the local script check;
all were removed with a non-recursive delete and the scratch directory was
deleted.

# DS-31U - first live-probe failure and canonical Windows path key

Date: 2026-09-13. Worker: DeepSeek DS-31U. Scope: repair exactly the one defect
the first real pilot exposed, plus its tests and this report/comms record. No
new features, no refactor, no launch/install/Minecraft/desktop action, no worker,
queue or artifact mutation, and no git stage/commit.

## The failed live attempt (observed, not inferred)

The first real pilot ran the executor against
`exports/datasets/pilot-batch-train-station-2026-09-12/queue` (manifest
`25b8c7926168b90bfd8cdb9505609980ae6f9cc00e18b3337be9116b3a347204`) for the
single selected scene `train-station-spawn-208-96-orbit-a`. The scene's
`executor-runs.jsonl` ledger records the attempt in full:

```
run_start  pid 16840  2026-09-12T20:29:23Z  scenes ["train-station-spawn-208-96-orbit-a"]
scene_start           2026-09-12T20:29:23Z  stage prepare
command  prepare  returncode 0  duration 9.672s  2026-09-12T20:29:33Z
provision  capture_mod .../staging-game/train-station-spawn-208-96-orbit-a/mods/mc-photo-capture-0.1.0.jar
scene_end  status failed  stage worker  2026-09-12T20:29:33Z
run_end    complete []  failed ["train-station-spawn-208-96-orbit-a"]  skipped []
```

The recorded failure detail is:

```
junction B:\prismInstances\MCPhotoCapture-Worker\minecraft resolved to
\\?\C:\Users\Likea\Documents\mc photogrammetry project\exports\datasets\pilot-batch-train-station-2026-09-12\staging-game\train-station-spawn-208-96-orbit-a,
not the requested staging target
C:\Users\Likea\Documents\mc photogrammetry project\exports\datasets\pilot-batch-train-station-2026-09-12\staging-game\train-station-spawn-208-96-orbit-a
```

**No Prism process was started.** The failure is at the `worker` stage, inside
`WindowsJunctionAdapter.repoint`, which runs *before* `_capture_request` and the
launcher; the ledger contains no `capture` event, and a `Get-Process` check for
`PrismLauncher`/`java`/`javaw`/`minecraft` after the run found none. The executor
stopped safely and left the scene `failed` (retryable through
`resume_queue`/`--resume`), the disposable worker
(`B:\prismInstances\MCPhotoCapture-Worker`, marker `instance_id`
`MCPhotoCapture-Worker`) valid, and its `minecraft` junction valid and pointing
at exactly the requested staging directory.

## Root cause

`_path_key` was `os.path.normcase(str(_absolute(path)))`. `os.path.abspath`
preserves the extended-length prefix, so a junction's raw reparse target read by
`os.readlink` as `\\?\C:\...` normalized to `\\?\c:\...`, which is a different
string from the requested `c:\...`. The post-repoint equality check therefore
raised even though the script had created the exact junction requested. The same
defect would have failed the pre-existing-junction containment check on the
retry, because `path_contains(target_root, "\\?\C:\...")` was also false.

## Repair

`_canonical_windows_path` strips a leading `\\?\` before the path is made
absolute and case-normalized: `\\?\C:\x` becomes `C:\x`, and
`\\?\UNC\server\share` becomes `\\server\share`. The rewrite is purely lexical -
it never resolves or follows the final component - so a link stays a link and no
containment is widened. `_path_key` is the single helper that applies it, so
every place that already used it (junction target equality, existing-junction
containment, `path_contains` for both the worker link and the target, and the
worker marker's `worker_dir`/`template_dir` comparisons) now treats both
spellings as the same object. `path_contains` keeps its exact-prefix rule, so
`C:\a\b` still does not contain `C:\a\bc`; a genuinely different path still
fails.

## Evidence (DS-31U)

```
.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch_executor.py \
    -o addopts= -p no:cacheprovider -q
-> exit 0, 124 passed in 3.62s   (was 114; +10 cases)

.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch_executor.py \
    tests/test_capture_batch.py tests/test_capture_dataset.py \
    tests/test_capture_options.py tests/test_capture_console.py \
    tests/test_litematic.py tests/test_camera.py tests/test_project_store.py \
    -o addopts= -p no:cacheprovider -q
-> exit 0, 271 passed, 1 skipped in 11.63s
   (was 261 passed/1 skipped; the skip is the pre-existing external-symlink case)
```

New tests: the drive-prefix equality, the UNC-prefix equality, mixed case, the
near-prefix escape (`C:\a\b` never contains `C:\a\bc`, with or without prefixes,
and the same for a UNC sibling), a genuinely different path, the **exact
observed LinkState target** from the pilot compared to its requested staging
path, an adapter repoint whose created junction reports the `\\?\` form, an
adapter that accepts a contained `\\?\`-spelled existing junction, an adapter
that still refuses an *uncontained* `\\?\`-spelled junction, and a worker marker
whose `worker_dir`/`template_dir` are written in the `\\?\` form.

Read-only confirmation against the real pilot junction (no mutation):
`WindowsLinkInspector().inspect("B:\prismInstances\MCPhotoCapture-Worker\minecraft")`
reports `kind == "junction"` with target
`\\?\C:\...\staging-game\train-station-spawn-208-96-orbit-a`; after the repair
that target's `_path_key` equals the requested staging path's and
`path_contains(staging root, target)` is true. The junction, worker marker,
queue manifest and ledger were read only.

## Honest limits (DS-31U)

* **The repair is not a live re-run.** No Prism/Minecraft launch was attempted,
  so the pilot scene has not been proven end to end; the next live run is what
  will exercise the fixed comparison against a real junction.
* **The PowerShell script already compared the plain form.** Windows
  PowerShell's `.Target` property returned `C:\...` (verified read-only against
  the pilot junction) while `os.readlink` returned `\\?\C:\...`; the script side
  needed no change and was left untouched.
* **The prefix strip is deliberately narrow.** Only the `\\?\` extended-length
  form and its `\\?\UNC\` spelling are canonicalized. A `\\?\Volume{GUID}\...`
  device path is not equated with a drive-letter path, and no other reparse
  semantics are reinterpreted.

## State left behind (DS-31U)

Only these paths were edited: `src/mc_photogrammetry/dataset/executor.py`,
`tests/test_capture_batch_executor.py`,
`docs/reports/DS-31-batch-executor-2026-09-12.md`, and this worker's appended
`AGENT_COMMS.md` note. `scripts/run_capture_batch.py`,
`src/mc_photogrammetry/dataset/__init__.py`, `batch.py`, the factory scripts, the
Java mod, the evaluator, thresholds, the existing pilot worker/junction/queue
manifest/ledger, and every other existing artifact are unchanged. No
Prism/Minecraft process, virtual desktop, junction creation/removal, source-save
write, or git stage/commit occurred.


## DS-33 - immutable per-scene workers

Date: 2026-09-12. Worker: Luna. Replaced the shared mutable minecraft-link design with stable per-scene workers. The interrupted DS-31V Directory.Delete proposal is rejected. Identity hashes canonical staging scope plus scene ID, with a bounded sanitized scene label; Prism instance ID equals its folder name. Link ensure is create-once, exact-target reuse is a no-op, and a different target fails closed without mutation. Script rechecks containment and has no junction replacement/removal operation. Fixed-worker multi-scene requests are rejected before mutation. No live worker/queue/artifact, PowerShell, junction/reparse point, external client/process, or git stage/commit was used.

Focused explicit node-ID selection (11 passed in 1.19s): test_cli_is_a_dry_run_by_default_and_refuses_execute_without_prism; test_cli_instance_map_resolves_a_per_scene_prism_command; test_cli_executes_a_scene_end_to_end_with_injected_seams; test_cli_execute_requires_every_worker_and_mod_argument; test_cli_execute_refuses_the_old_per_scene_instance_mapping; test_cli_dry_run_with_worker_args_mutates_nothing; test_scene_worker_identity_is_stable_unique_and_bounded; test_scene_worker_identity_rejects_empty_scene_id; test_scene_worker_identity_requires_matching_prism_folder_prefixes; test_executor_lifecycle_order_uses_a_distinct_worker_per_scene; test_executor_rejects_a_fixed_worker_for_multiple_scenes_before_mutation.

Safe executor command:
.venv\Scripts\python.exe -m pytest tests\test_capture_batch_executor.py --deselect=tests/test_capture_batch_executor.py::test_subprocess_runner_runs_a_real_command_without_a_shell --deselect=tests/test_capture_batch_executor.py::test_subprocess_runner_reports_a_nonzero_exit --deselect=tests/test_capture_batch_executor.py::test_subprocess_runner_interrupts_and_kills_the_child -o addopts= -p no:cacheprovider -q
Result: 125 passed, 3 deselected, 2.87s.

Dataset-focused command:
.venv\Scripts\python.exe -m pytest tests\test_capture_batch.py tests\test_capture_dataset.py tests\test_capture_options.py tests\test_capture_console.py tests\test_litematic.py tests\test_camera.py tests\test_project_store.py --deselect=tests/test_capture_batch.py::test_cli_materializes_a_unicode_identity_on_a_strict_cp1252_console --deselect=tests/test_capture_dataset.py::test_plan_is_identical_in_a_fresh_process --deselect=tests/test_capture_console.py::test_cli_list_exits_zero_with_strict_cp1252_stdout -o addopts= -p no:cacheprovider -q
Result: 144 passed, 1 skipped, 3 deselected, 6.57s. All deselected tests start external child processes.

Compile command: .venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\dataset\executor.py scripts\run_capture_batch.py tests\test_capture_batch_executor.py (exit 0). The executor suite includes the static junction-script containment/no-destructive-token audit. Lead independently reports queue dry-run manifest unchanged and no worker directories created.

SHA-256: executor.py e84936ad47eea142f77638f7a3275142e7c2ae4640a5b8cf09dcbd0444286bbd; run_capture_batch.py ede691290aa5c00131ee011555209044e54e82a8f10d41cf1c618f1360be03f7; test_capture_batch_executor.py cb84ee782ab83ceb016518ef08b07ee759c017ba3b063331d4da196eaeee2a42.

No source edits followed the passing runs; no staging or commit.
