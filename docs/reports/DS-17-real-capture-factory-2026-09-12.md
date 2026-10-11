# DS-17R (restart) - real Minecraft capture foundation

Date: 2026-09-12. Worker: DeepSeek DS-17R-restart. Contract source:
`AGENT_COMMS.md` 2026-09-12T11:45:45 (DS-17R) and 2026-09-12T11:50:00 (restart).
Scope: implement only the owned new paths; no existing source/test/world edit, no
mod install, no launch, no stage/commit/self-chain. Source worlds were never
opened for write. Result: Python factory + Fabric client mod implemented, all
focused tests green, `gradlew clean build` exits 0.

**Amendment 2026-09-12 (DS-17F).** The lead reproduced this build (56/56 Python
focused, Gradle 10/10 with 7/7 Java tests) but rejected launch and issued seven
bounded corrections; a follow-up correction superseded one claim about the
coordinate convention. All seven are now implemented, with the eye-coordinate
semantics preserved exactly as `camera_eye(camera) == position + eye_height`.
See "Corrections after lead review" below for the authoritative description and
the new evidence.

**Amendment 2026-09-12 (DS-17G, then DS-17H).** Two further launch rejections
followed source review. DS-17G's seven corrections are in "Corrections after
second lead review"; DS-17H then replaced the controller's dual state machine with
one explicit `CaptureDriver`, moved the staging proof into the copied world's own
root via `MinecraftServer#getSavePath(WorldSavePath#ROOT)`, and made
`build_plan`'s `staging_token` required. See "Corrections after third lead review"
at the end for the authoritative DS-17H description and its evidence.

**Amendment 2026-09-12 (DS-17J).** DS-17I was accepted by focused build/review but
launch was still withheld for missing setting variation and a landscape-usability
gap. Both are now implemented: the default planner stratifies gamma across the
client's whole supported `[0, 1]` range, cycles a fixed representative day/night
time set, and mixes HUD-hidden with HUD-visible captures; landscape preparation now
defaults to the save's `SpawnY`. See "Corrections after fifth lead review" at the
end. Java was not touched, so no Gradle build was run for this correction.

## Deliverables (all new; `git status --porcelain` shows only these untracked)

| Path | Lines | Purpose |
| --- | --- | --- |
| `src/mc_photogrammetry/dataset/schemas.py` | 487 | Versioned records: `SceneRecipe`, `RegionBounds`, `CaptureSettings`, `CameraCapture`, `CapturePlan`, `CaptureFrameResult`, `BlockRecord`, `GroundTruthExport`, `GeneratedScene` |
| `src/mc_photogrammetry/dataset/plan.py` | 364 | Target->pose inversion, deterministic orbit sweeps, region files, validation, `EYE_POSITION_TOLERANCE` |
| `src/mc_photogrammetry/dataset/finalize.py` | 367 | Capture validation, feet/eye pose verification, hashed scene files, dataset index |
| `src/mc_photogrammetry/dataset/inventory.py` | 154 | Read-only save inventory, path containment, staged copy |
| `src/mc_photogrammetry/dataset/blocks.py` | 83 | Exact blockstate JSON -> `BlockVolume` -> `reference.litematic` |
| `src/mc_photogrammetry/dataset/splits.py` | 74 | Scene-level SHA-256 train/dev/test split |
| `src/mc_photogrammetry/dataset/hashing.py` + `__init__.py` | 26 + 115 | Canonical JSON/hash helpers; public surface |
| `scripts/prepare_capture_dataset.py` | 213 | Inventory -> staged copy -> versioned plan (+ optional `--arm`) |
| `scripts/finalize_capture_dataset.py` | 76 | Staged results -> scene files -> `index.json` |
| `tests/test_capture_dataset.py` | 659 | 31 contract tests, including roll rejection and feet/eye tolerance |
| `tools/minecraft-capture-mod/` | ~1450 | Fabric 1.21.5 client mod, real Gradle wrapper, Java tests |
| `docs/reports/DS-17-real-capture-factory-2026-09-12.md` | this file | Report |

## Python contracts

**Plan/pose.** `camera_from_target` inverts the repo convention
`forward = (-cos(p)sin(y), -sin(p), cos(p)cos(y))` as `pitch = -asin(dy)`,
`yaw = atan2(-dx, dz)`, so `target` projects to the principal point exactly.
`CameraModel.position` is the player's **feet**, matching how the mod sets a
player: the shared `camera_eye(camera)` adds `eye_height`, and `camera_from_target`
documents and uses that. A seeded orbit
(`random.Random(f"{seed}|{scene_id}|{min}|{max}")`) picks angle, elevation,
resolution (1920x1080/1600x900/1280x720), FOV (60/70/80/90), gamma and weather once
per view, then widens the radius up to 12x until all 8 region corners project
inside the frame (`bounds_coverage == 1.0`); otherwise it raises. `validate_plan`
re-checks explicit intrinsics vs resolution, **rejects nonzero `roll_degrees`** (the
client has no roll control), target inside the region, finite projection matrix,
and a bounded camera distance. `CaptureSettings` declares
gamma/time/weather/HUD/resource-pack/shader; only the first four are applied by the
mod (see limits), and the pack names are recorded as *requested*, never as active.

**Caps (mod and Python must agree).** `MAX_PLAN_JOBS=64`,
`MAX_REGION_EDGE=512`, `MAX_REGION_VOLUME=250_000` blocks, `MAX_SETTLE_TICKS=600`.
The volume cap exists because the mod builds ground truth on the client tick.

**Split.** `assign_split(seed, family, volume_topology)` hashes a canonical digest
of occupied coordinates + exact states (never views) and buckets 70/15/15.
`split_for_scene` is a pure function of the concrete volume; `split_report`
refuses the same `scene_id` in two splits.

**Paths and overwrite.** `safe_join`/`safe_component` reject `..`, separators,
absolute and trailing-dot components. The finalizer refuses to write
`capture.json`/`ground_truth.json`/`reference.litematic` without `force`, refuses
a dataset root inside the staging game directory, and refuses a scene whose
ground-truth world identity differs from the plan.

**Finalization.** Every planned frame needs a `CaptureFrameResult` with
`captured=true` and a real screenshot file; the screenshot is copied into
`scenes/<id>/captures/`, hashed, and listed. Every captured frame must also record
`feet_position` and `eye_position`, and `validate_pose` rejects the scene unless
`feet_position` is within `EYE_POSITION_TOLERANCE = 0.25` blocks of the planned
`CameraModel.position` **and** `eye_position` is within the same tolerance of
`camera_eye(job.camera) == position + eye_height`. `ground_truth.json` converts to
`reference.litematic` pinned to the declared bounds (air dropped from the sparse
volume, kept in JSON). `index.json` is timestamp-free, so it is byte-identical
across rebuilds; it carries generator version, plan/world provenance, per-file
hashes, per-screenshot hashes, split, and `advanced_only: true` plus the explicit
usage label "developer/evaluation ground truth only; never normal reconstruction
input".

**Mod handshake.** `armed.json` lives at `<game>/config/mc-photo-capture/armed.json`
and names a plan by game-dir-relative `plan_path` plus `plan_sha256`; the planner
copies the frozen plan to `<game>/mc-photo-capture/plans/<plan_id>.json` and only
writes `armed.json` under explicit `--arm`. The mod writes
`<game>/mc-photo-capture/out/<plan_id>/results.json` and `ground_truth.json`.

## Fabric mod (tools/minecraft-capture-mod)

Versions: Fabric Loom `1.10.5`, Minecraft `1.21.5`, Yarn `1.21.5+build.1`,
Fabric Loader `0.16.10` (matches the locally installed
`fabric-loader-0.16.10-1.21.5`), Gradle wrapper `8.14.3` (wrapper jar/scripts
fetched from the official `gradle/gradle` tag `v8.14.3`; no local Gradle existed).
Java target 21 (`options.release = 21`), compiled with the installed JDK 24
because no JDK 21 is present; runtime Java >= 21 as the launcher profile provides.

Safety behavior. `CaptureController.load()` stays inert when `armed.json` is
absent, and otherwise validates the digest (`PlanAudit.verifyDigest`), schema and
contract version, bounded jobs/region, filename-safe frame ids, nonzero-roll
rejection, and contained plan path before arming. `begin()` returns before touching
any player state unless `client.isInSingleplayer()`, `client.getServer() != null`,
and the integrated server's `getSaveProperties().getLevelName()` equals the plan's
`world_identity`. Per job it submits bounded tasks to the **server thread** for the
player teleport (`TeleportTarget` to the requested feet plus yaw/pitch) and
`ServerWorld.setTimeOfDay`/`setWeather`; the client tick polls a completion flag and
never touches `ServerWorld`. FOV/gamma/HUD/window are client options. It waits for
the client feet to converge on the requested feet (bounded re-teleports), waits
`settle_ticks`, rejects a stale screenshot file, calls
`ScreenshotRecorder.saveScreenshot(runDirectory, "<frame>.png", framebuffer, cb)`,
and accepts success only after the callback signals and the final file is nonempty.
It then exports exact `BlockState` ids/properties from `getOverworld()` **on the
server thread**, writes `results.json`/`ground_truth.json` via temp+atomic replace,
restores the original FOV/gamma/HUD/window and `options.write()`, and atomically
disarms `armed.json` by a same-directory rename to `armed.json.consumed`
(`armed.json.rejected` on a rejected plan) followed by best-effort deletion. Aborts
(multiplayer, identity mismatch, tick limits) write per-frame failure results and a
ground-truth sentinel so the finalizer refuses the scene. No commands, no F3, no
coordinates.
The only hook is one client mixin `@Inject(method="tick", at=@At("TAIL"))` on
`MinecraftClient`; the generated refmap maps it to
`Lnet/minecraft/class_310;method_1574()V`, so no Fabric API module is required.

`PlanAudit` and `CapturePlumbing` are Minecraft-free so the safety rules are
unit-testable on a plain JVM: 16/16 cases pass (`PlanAuditTest` 8:
schema/contract versions, unsafe/duplicate frame ids, 64-job cap, 250k region cap,
digest mismatch, path traversal, nonzero roll; `CapturePlumbingTest` 8: feet+eye
arithmetic, tolerance accept/reject, JSON triple parsing, stale-file removal,
callback+nonempty acceptance, atomic JSON replace, atomic disarm).

## Evidence

```
gradlew clean build --offline --no-daemon --no-watch-fs --console=plain
  JAVA_HOME=C:\Program Files\Java\jdk-24
  GRADLE_USER_HOME=<mod>/.lead-gradle-ds17   (lead's cached wrapper + dependencies)
GRADLE_EXIT=0 ; BUILD SUCCESSFUL in 12s ; :compileJava :compileTestJava :test :build
PlanAuditTest          8 PASSED / 0 FAILED
CapturePlumbingTest    8 PASSED / 0 FAILED
build/test-results/test/TEST-...PlanAuditTest.xml         tests="8" failures="0" errors="0"
build/test-results/test/TEST-...CapturePlumbingTest.xml   tests="8" failures="0" errors="0"
artifacts: build/libs/mc-photo-capture-0.1.0.jar (38966 B)
           build/libs/mc-photo-capture-0.1.0-sources.jar (20186 B)
jar SHA-256 252E085923F7D1A7A00C8105644619F7D90C6FB8D2D61D119A144FC69AFB718D
sources SHA-256 2B2CA4E65790A9D7A3FEBED9E9C04E9B07388181C6B5F12D36A790FB97C952EC
```

The first offline attempt exposed three real API mismatches, fixed from the
compiler output and re-checked against the cached 1.21.5 merged jar (`javap`):
`TeleportTarget` needs `(ServerWorld, Vec3d, Vec3d, float, float,
Set<PositionFlag>, PostDimensionTransition)` (the constant `TeleportTarget.NO_OP`
is used), the screenshot callback delivers a `Text` rather than a `String`, and
`ResourcePackManager` exposes `getEnabledIds()` instead of a no-argument
`ResourceManager.getAllResources()`. No API research beyond those compile errors.

Python focused run (`.venv`, Python 3.10.11): `pytest tests/test_capture_dataset.py
tests/test_litematic.py tests/test_camera.py tests/test_project_store.py -q` ->
exit 0, 59 collected/passed (dataset 31, camera 19, litematic 8, project_store 1).
The dataset module alone is 31 passed; the two new cases cover roll rejection and
the recorded feet/eye contract, including the exact eye-height-sized swap error.

Script smoke (fixture game dir, staged copy, `--regions`, fabricated mod outputs
matching the contract): `prepare_capture_dataset.py` rc 0 -> plan
`plan-3b9de43c73426e91`, staged at
`<dataset>/staging-game/saves/Template 2`, plan copied to
`mc-photo-capture/plans/`; `finalize_capture_dataset.py` rc 0 -> scene
`template-2-house`, split `train`, `index.json` with hashes for recipe/plan/
capture/ground_truth/reference.litematic. The scratch tree was deleted afterwards.

## Corrections after lead review (2026-09-12T12:04 / 12:06)

The lead reproduced the build (56/56 Python focused, Gradle 10/10 with 7/7 Java
tests) and rejected launch approval, listing seven bounded defects. All seven are
now implemented. The 12:06 correction superseded only the mistaken eye-coordinate
claim and is applied verbatim: `CameraCapture.position` stays the player's
**feet**, the shared `camera_eye(camera)` still adds `eye_height`, and the
requested feet `(x,y,z)` and yaw derivation are preserved. The stray
`.apply_patch_probe.txt` was removed.

1. **Eye vs feet / client-only teleport.** `apply` no longer calls
   `player.refreshPositionAndAngles(job.x(), job.y(), job.z(), ...)`. The integrated
   server player is teleported on the server thread to the requested **feet** via
   `TeleportTarget`, the client's feet are then polled until they converge (up to
   `MAX_POSITION_ATTEMPTS = 3` bounded re-teleports), and `recordSuccess` writes both
   `actual.feet_position` and `actual.eye_position` from the rendered client state.
   No eye-height subtraction or reinterpretation was introduced. The finalizer's
   `validate_pose` compares `feet_position` to the requested feet and `eye_position`
   to `camera_eye(job.camera)` within the documented
   `EYE_POSITION_TOLERANCE = 0.25` blocks, recorded in `capture.json`.
2. **Server-thread ownership.** Teleport, time and weather are `server.execute(...)`
   tasks (`Action.POSITION` / `Action.SETTINGS`) whose actuals are re-read on the
   server thread immediately after the mutation; ground truth is `Action.GROUND_TRUTH`.
   The client tick only schedules and waits on volatile completion flags with bounded
   tick budgets (`SERVER_WAIT_TICKS`, `GROUND_TRUTH_WAIT_TICKS`), and records
   server-thread exceptions into the frame result. No `ServerWorld` read/write happens
   on the client thread.
3. **Screenshot completion.** Before each capture the stale target inside the
   contained `screenshots/` directory is deleted (`CapturePlumbing.removeStaleScreenshot`),
   and success requires the recorder callback to have signalled **and** the final file
   to be a non-empty regular file (`isScreenshotAccepted`). Callback error text is
   captured and recorded. Both helpers are pure and unit tested.
4. **Requested vs actual packs.** Resource/shader names are stored under
   `actual.requested`; the client's enabled pack ids are recorded under
   `actual.active_resource_packs`, and `resource_pack_applied` / `shader_pack_applied`
   are `false` until runtime switching exists. Nothing claims a request is active.
   FOV/gamma/window actuals remain separate keys from the requested plan values.
5. **Roll.** `validate_plan` raises on any nonzero `camera.roll_degrees`, and
   `PlanAudit` rejects it too (`camera roll is not supported by the client`).
6. **Atomic disarm.** `CapturePlumbing.disarmAtomic` renames `armed.json` to a
   same-directory `armed.json.consumed` / `.rejected` marker atomically and only then
   best-effort deletes the marker; a plain `deleteIfExists` is gone. A JVM file test
   covers it, including a pre-existing stale marker.
7. **Durable writes.** `results.json`, `ground_truth.json` and `last-error.json` go
   through `writeJsonAtomic` (temp sibling + `ATOMIC_MOVE` replace, plain-move
   fallback). A failed ground-truth export writes a `ground_truth_exported: false`
   sentinel plus a `run_status.json` failure record and then disarms safely, so the
   finalizer refuses the scene instead of accepting a partial run.

New focused coverage: `tests/test_capture_dataset.py` roll rejection and feet/eye
tolerance (including the 1.62-block swap and the at-tolerance boundary);
`PlanAuditTest.rejectsNonzeroRollBecauseTheClientCannotApplyIt`; and the eight
`CapturePlumbingTest` cases for eye-height arithmetic, tolerance schema, triples,
stale screenshots, callback acceptance, atomic JSON replace and atomic disarm.

## Not done (deliberately) and the exact remaining step

1. Not installed and not launched. Remaining step: copy
   `tools/minecraft-capture-mod/build/libs/mc-photo-capture-0.1.0.jar` into the
   staged game directory's `mods/` (or the local `.minecraft/mods/`), run
   `prepare_capture_dataset.py --arm` against the staged copy, open the staged
   world in singleplayer, and let the plan finish; then `finalize_capture_dataset.py`.
2. Resource-pack and shader fields are recorded as `actual.requested`, never as
   active; the enabled pack ids are read from the client into
   `actual.active_resource_packs`, `resource_pack_applied` / `shader_pack_applied`
   stay `false`, and no reload is triggered.
3. Landscape sampling defaults its centre height to the save's `SpawnY` through the
   new 3D reader `spawn_position_from_save`; a save whose spawn is missing or
   unreadable is refused clearly unless the developer passes `--surface-y`. True
   chunk-heightmap sampling is still not implemented: the height remains a spawn
   coordinate (or an explicit developer input), never a sampled terrain surface.

## Unvalidated risks (cannot be checked without launching, which is out of scope)

- Client teleports of ~40-60 blocks in one tick may trip the integrated server's
  "moved too quickly" correction and rubber-band the player. Mitigated, not
  eliminated: the server player is teleported to the exact requested feet, the
  client feet are polled and re-teleported up to 3 times, and the recorded
  `feet_position`/`eye_position` are checked against the request by the finalizer
  within 0.25 blocks, so a rubber-band becomes a recorded failure rather than
  silent bad ground truth. The first launch should still read the failure text.
- Screenshot timing/size after `setWindowedSize` can lag a frame or two; actual
  framebuffer dimensions are recorded for every capture, and the finalizer does
  not require an exact match, so the failure is visible rather than silent.
- No visual check that a screenshot shows the intended view (yaw/pitch sign
  convention is verified only analytically against the repo projection).
- No launch has exercised the server-thread task handoff, so the exact number of
  ticks an integrated-server teleport needs to reach the client is unmeasured;
  the budgets (`SERVER_WAIT_TICKS = 1200`, `GROUND_TRUTH_WAIT_TICKS = 600`) are
  generous but untested in situ.

## Corrections after second lead review (2026-09-12T12:16, DS-17G)

The lead's source review rejected launch again after the DS-17F corrections, even
though the build and tests passed: the build could not catch controller ordering.
Seven defects were enumerated and are all now implemented. No launch, install,
world mutation, dependency change, or out-of-scope path was touched.

**1. Critical: the initial POSITION was never scheduled.** `startApply()` set
`phase = APPLY` and the apply loop saw `pendingAction == null && serverActuals ==
null`, so it scheduled SETTINGS first and the teleport only happened after a
non-convergence retry. The order is now an explicit Minecraft-free state machine,
`JobFlow`: `WAIT_POSITION -> CONVERGE -> WAIT_SETTINGS -> SETTLE -> CAPTURE ->
WAIT_FILE -> DONE`. `startApply` resets the whole flow and every per-job field
(settle, file, server and convergence counters, pending ticket, cached screenshot
digest, cached screen position, world actuals), and `apply` only ever submits the
task the current stage declares through `JobFlow.requiredAction()`. The only way
into SETTLE is a CONVERGE that `confirmConvergence(true)` accepted; a terminal
convergence failure calls `recordFailure` + `advance()` and the stage stays at
CONVERGE, so the job is neither settled nor captured. `JobFlowTest` (5 cases)
fails on a missing, reordered, or repeated stage and on a failure-to-capture path.

**2. Server task targeted `getPlayerList().get(0)`.** `scheduleStage` now resolves
the integrated client player's UUID on the client thread and the server task looks
that exact player up with `getPlayerManager().getPlayer(uuid)`; an absent match
throws and the job is recorded as failed. Yaw is compared with a circular delta
(`PlanAudit.angleDeltaDegrees`, rounding before the modulo so a full turn lands on
0), pitch with an absolute delta, both against
`HEADING_TOLERANCE_DEGREES = 1.0`; the client records actual yaw/pitch (already
did) and `validate_pose` now enforces the same documented heading tolerance
alongside feet/eye. `JobFlow` and `PlanAudit` cover the pure parts.

**3. Level name was the only staged-copy proof.** `prepare_capture_dataset.py` now
stages through `stage_world()`, computes the source save's SHA-256 content
fingerprint, writes `<staging>/config/mc-photo-capture/staging.json` with
`schema_version`, `token` (= the fingerprint), `source_fingerprint`,
`world_identity`, `dataset_id`, `scene_id`, `plan_id`, and passes the same
`staging_token` into `CapturePlan` (schema/contract version 1 unchanged, because
both sides were changed atomically in this correction). The mod validates the
marker in `load()` and again in `begin()` after the singleplayer and level-name
checks, then recomputes the opened save's fingerprint with
`StagingMarker.fingerprint` and requires an exact match with the marker before
`recordOriginals` or any teleport. `PlanAudit` additionally rejects a plan whose
`staging_token` is absent, malformed, or the inert all-zero placeholder.
`StagingMarkerTest` (8 cases) covers token mismatch, world-identity mismatch,
unsupported marker schema, malformed JSON, the inert token, marker location, the
content-addressed fingerprint, and the screenshot digest helper.

**4. `copy_world_to_staging` had no caps and followed symlinks.** Staging is now
`preflight -> copytree -> source re-inventory -> copy fingerprint check`, all
raised as `DatasetSchemaError`. The preflight walks with `followlinks=False`,
refuses any symlink or Windows reparse point/junction before descending, and
enforces assigned caps `StagingLimits(max_files=200_000, max_bytes=4 GiB)`
(overridable per call). `session.lock` stays excluded. The copy's fingerprint must
equal the source's, and the source inventory (relative path + size + mtime_ns) is
taken again after the copy so a save that changed underneath is reported instead of
silently staged. Tests cover the over-cap file and byte refusals, the external
symlink refusal (skipped where the platform forbids creating symlinks), the
fingerprint stability/content-addressing, and a monkeypatched copy that mutates the
source mid-copy.

**5. Split identity included the capture seed.** `splits.py` now keys the split on
the topology digest alone: `scene_group(topology)` is the split identity,
`assign_split`/`split_for_scene` accept `seed`/`family` only for call
compatibility and deliberately ignore them, and the SPLIT_SALT moved to
`.../v2`. `finalize._index_entry` records both `topology_digest` and
`scene_group`, and `split_report` refuses the same `scene_id` in two splits and
the same `scene_group` in two splits, while counting an intentional recapture
inside one split as `repeated_groups`. Tests prove identical topology keeps one
split across six seeds and four families, that the three buckets are still
covered over 120 scenes, and that a cross-split group is rejected while an
in-split repeat is not.

**6. Screenshots were never content-verified.** The mod hashes the finished file
with `StagingMarker.fileSha256` and records `actual.screenshot_sha256`; the
finalizer requires a well-formed digest, verifies the source file against it
before copying, and verifies the copied file against the same digest after
copying, recording both `sha256` (copy) and `source_sha256` (source) in
`capture.json` and the index. A changed or missing digest is a refusal
(`test_finalize_refuses_a_screenshot_whose_bytes_changed`).

**7. `finish`/`abort` task-state races.** Each server task now owns a `Ticket`
object that only that task marks complete, so a late callback cannot be read as
the success of the next task; `finish` resets the ground-truth wait budget at
submission instead of counting from zero forever; `finished` makes every terminal
path idempotent; and `terminate` restores the client options and disarms exactly
once from success and failure alike. Ground truth can no longer be reported as
success: `failGroundTruth` clears the cached payload, writes the frame results and
`run_status.json` with `ground_truth_exported: false` (computed from an all-frames-
captured check rather than a cleared flag), writes the sentinel, rewrites the
status, then terminates. The DS-17F atomic temp+replace writes and same-directory
atomic disarm are unchanged.

### Evidence (DS-17G)

```
gradlew.bat --project-dir tools\minecraft-capture-mod clean build \
    --offline --no-daemon --no-watch-fs --console=plain
  JAVA_HOME=C:\Program Files\Java\jdk-24
  GRADLE_USER_HOME=<mod>/.lead-gradle-ds17   (lead's cached wrapper + dependencies)
GRADLE_EXIT=0 ; BUILD SUCCESSFUL in 12s ; :compileJava :compileTestJava :test
31 tests completed, 0 failed
  CapturePlumbingTest   8 tests, 0 failures
  JobFlowTest           5 tests, 0 failures   (new: stage order + failure-to-capture)
  PlanAuditTest        10 tests, 0 failures   (new: staging token, circular yaw)
  StagingMarkerTest     8 tests, 0 failures   (new: marker + fingerprint + digest)
artifacts: build/libs/mc-photo-capture-0.1.0.jar         53039 B
           build/libs/mc-photo-capture-0.1.0-sources.jar 27842 B
jar SHA-256     E3B069680957208BBED10DEA55F33E7D37B73CD5DD70C6FF9770D7EB8EC12E54
sources SHA-256 D7472B205911E09E2B0C3563EBE01B1BF0C4DF802DE49C2D3FBDF695C05C2FE0
```

Two first-attempt JVM failures were mine, in the new `JobFlowTest`, and were fixed
from the build output: `confirmConvergence(false)` deliberately leaves the stage at
CONVERGE (so `next()` is legal there), and the wrong-stage test was asserting from
WAIT_SETTINGS instead of SETTLE. No production code changed for either.

```
pytest tests/test_capture_dataset.py tests/test_litematic.py tests/test_camera.py \
       tests/test_project_store.py -q
PYTEST_EXIT=0 ; 68 passed, 1 skipped
  tests/test_capture_dataset.py alone: 39 collected, 38 passed, 1 skipped
  the skip is the external-symlink staging case: Windows in this sandbox does not
  permit creating symlinks without elevation, so the refusal path is not exercised
  here (the over-cap and mid-copy-change refusals are)
```

The same first-attempt run caught one real Python defect: the circular-yaw helper
returned `0.0` for `359.9` vs `-0.1` because floating point represents the
difference as exactly `360.0`; the fix rounds before the modulo in both languages
and the tests now use unambiguous wrap points (`10` vs `350`, `300` vs `-50`).

End-to-end fixture smoke (a scratch save named `Template 2` under a temp game
dir, `--regions`, `--seed 7`): `prepare_capture_dataset.py` rc 0 wrote
`staging-game/config/mc-photo-capture/staging.json` whose `token` equals the plan's
`staging_token` (`ec4341546cde...`), and `finalize_capture_dataset.py` rc 0 wrote
`index.json` with `summary {'scenes': 1, 'groups': 1, 'repeated_groups': {},
'by_split': {'train': 1, 'dev': 0, 'test': 0}}`, `topology_digest cbae570db2a0...`
and `scene_group 6a0180839bbf...`. The scratch tree was deleted afterwards.

## State left behind

Only files in the owned paths above were added or amended. DS-17G touched
`dataset/hashing.py`, `dataset/inventory.py`, `dataset/splits.py`,
`dataset/schemas.py`, `dataset/plan.py`, `dataset/finalize.py`,
`dataset/__init__.py`, `scripts/prepare_capture_dataset.py`,
`tests/test_capture_dataset.py`, the mod's `CaptureController`, `PlanAudit`,
`JobFlow` (new), `StagingMarker` (new), `JobFlowTest` (new), `StagingMarkerTest`
(new) and `PlanAuditTest`, plus this report. `PlanAudit.SUPPORTED_SCHEMA_VERSION`
and `SUPPORTED_MOD_CONTRACT_VERSION` remain 1 because the Python and Java sides
were changed together in this correction; `splits.SPLIT_SALT` moved to `v2`
because the split inputs changed. `.apply_patch_probe.txt` was already gone and
was re-checked as absent. Gradle caches now live in the owned
`<mod>/.lead-gradle-ds17` user home (used offline, no dependency was downloaded);
`tools/minecraft-capture-mod/build/libs/*.jar` was kept as the build artifact. No
source world, no live `.minecraft`, no staging world remains.

## Shared risks after DS-17G (still unvalidated without a launch)

- The stage machine is now enforced in code and tested, but the in-situ number of
  client ticks between a server teleport and convergence, and between the settings
  task and the settle window, is still unmeasured; the budgets
  (`SERVER_WAIT_TICKS = 1200`, `GROUND_TRUTH_WAIT_TICKS = 600`) are generous and
  bounded per ticket.
- The mod's staging fingerprint reads the whole staged save (bounded by
  `max_bytes = 4 GiB`) on the client thread when the world is entered. That is a
  one-time cost, but on a multi-GB save it will stall the first tick; the cap is
  the mitigation and a cheaper metadata-only variant would weaken the identity
  proof, so the content hash was kept.
- `validate_pose` now rejects a result that records no yaw/pitch, so any capture
  recorded by an older mod build is refused rather than accepted without a
  heading check.
- The external-symlink staging refusal is implemented but unexercised on this
  Windows sandbox (symlink creation is not permitted without elevation); a POSIX
  run or an elevated Windows run should exercise it.

## Corrections after third lead review (2026-09-12T12:30, DS-17H)

The lead accepted the DS-17G test results but rejected launch again after a
controller integration review: passing tests did not cover the order in which the
controller and its helper actually ran. Four exact defects were enumerated; all
four are implemented below. No launch, install, world mutation, dependency change,
or out-of-scope path was touched, and the earlier DS-17F/DS-17G corrections
(server-thread ownership, screenshot completion, requested-vs-actual packs, roll
rejection, atomic disarm, durable writes, screenshot digests, caps/symlinks, split
identity) are unchanged.

**1. Critical: `apply` ran a second state machine that could disagree with the
phase enum.** `CaptureController.apply` was replaced by one explicit, testable
driver, `CaptureDriver`, which owns the whole per-job sequence; the controller now
holds no stage state at all and the old `SETTLE` phase and its `flow.next()` call
are gone. The real call sequence the controller produces is exactly

```
start -> SCHEDULE_POSITION
position -> (CONVERGE) -> SCHEDULE_SETTINGS
settings -> SETTLE
settle -> INVOKE_CAPTURE
capture -> AWAIT_FILE
file -> DONE
```

The two specific bugs are gone by construction:

- A completed POSITION now hands its outcome to the driver, which records the
  `CONVERGE` stage before it decides anything, so the old "convergence reported
  from `WAIT_POSITION`" exception cannot occur.
- A completed SETTLE emits `INVOKE_CAPTURE`, which the controller consumes in the
  same tick; `capture(client)` now calls `jobDriver.onCaptureCompleted()` as soon
  as the writer starts, so the machine enters `AWAIT_FILE` instead of walking
  through capture without a screenshot.

A divergence during convergence emits only `SCHEDULE_POSITION`; the attempt ladder
is bounded by `MAX_POSITION_ATTEMPTS = 3`, and a terminal failure emits `FAIL_JOB`
and `advance()`s the job without ever reaching SETTLE, CAPTURE, or AWAIT_FILE.
`failJob` calls `jobDriver.fail()` once (idempotent: a driver that already failed
does not record a second `FAILED` stage).

`CaptureDriverTest` (6 cases) is Minecraft-free and asserts the emitted
`Effect`s and the `CaptureDriver.Stage` list the job actually entered, not just enum
successors: the exact order above with each stage once, divergence retrying
POSITION with no settings/capture, terminal convergence failure without capture,
a failure after capture not being able to report a screenshot, out-of-order and
duplicate stage calls throwing, and the one-attempt configuration. The earlier
`JobFlow`/`JobFlowTest` remain only as the position/settings action labels and
their own transition test; they are no longer consulted for controller order.

**2. The staged-copy proof resolved `gameDir/saves/<LevelName>` and hashed the
live tree.** Save folders commonly differ from `LevelName`, and Minecraft
creates/updates files (session lock, player data, region data) as soon as the copy
is opened, so a valid older copy could fail and a multi-GB hash on the client
thread could freeze the game. `begin()` now reads
`client.getServer().getSavePath(WorldSavePath.ROOT)` - the 1.21.5 integrated-server
save-path API, verified against the cached merged jar with `javap` - and calls the
new `StagingMarker.readAndValidateWorld(worldRoot, token, worldIdentity)`, which
reads `<world root>/.mc-photo-capture-staging.json`, requires the plan's token and
world identity, and never rehashes the world tree. `Python`'s
`prepare_capture_dataset.py` writes that marker into the verified copy's own root
(after copy verification, so it never perturbs the staged fingerprint) and keeps
the old `<game>/config/mc-photo-capture/staging.json` copy purely as provenance;
`CaptureController.load()` no longer treats the game-directory file as proof. New
Java cases cover the marker location, the folder-name != LevelName scenario, and
mutating ordinary world bytes after staging without invalidating the marker, and a
new Python case does the same through the real stager.

**3. `build_plan` still carried an implicit inert token.** `staging_token` is now a
required keyword-only parameter with no default, `_UNSTAGED_TOKEN` and the
`unstaged_token()` helper are deleted from the module and its public `__all__`, and
a blank token is refused with a clear error. The old test that produced a
placeholder plan now asserts that omitting the token raises `TypeError`, that a
blank token raises, and that `unstaged_token` is gone from the public surface.
Java still refuses all-zero/inert tokens in `PlanAudit.requiredDigest` and
`StagingMarker.parseAndValidate` (unchanged, still tested).

**4. Focused re-run and clean build.** Ran the focused Python set and
`gradlew clean build --offline` as specified; exact results below.

### Evidence (DS-17H)

```
gradlew.bat clean build --offline --no-daemon --no-watch-fs --console=plain
  JAVA_HOME=C:\Program Files\Java\jdk-24
  GRADLE_USER_HOME=<mod>/.lead-gradle-ds17
GRADLE_EXIT=0 ; BUILD SUCCESSFUL in 17s ; 10 actionable tasks executed
39 tests completed, 0 failed
  CaptureDriverTest     6 tests, 0 failures   (new: emitted effects + stage order)
  CapturePlumbingTest   8 tests, 0 failures
  JobFlowTest           5 tests, 0 failures
  PlanAuditTest        10 tests, 0 failures
  StagingMarkerTest    10 tests, 0 failures   (was 8; +world marker, +folder!=level)
artifacts: build/libs/mc-photo-capture-0.1.0.jar         58489 B
           build/libs/mc-photo-capture-0.1.0-sources.jar 31902 B
jar SHA-256     F26CA9DA2AAEEF8F5365336060E2EB556E55607929A4FEAA22F771E4C88EE847
                (identical across the two DS-17H builds)
sources SHA-256 C1FF3EE75342E4708C29D340CB2E0E552130875E839E4CDC70E7DDE2CFD41BEB
                on the first green run; the sources jar embeds entry timestamps, so
                a second green build produced the same 31902 bytes with a different
                digest (7BBC7881...). The compiled jar above was byte-identical.
```

The first attempt at this build failed at `:compileTestJava` with one error, mine
and fixed from the build output: the new Java test called
`Files.writeString(path, new byte[])`, which does not exist (`Files.write` is the
byte-array overload). The first `:test` run then failed one case, also mine: the
divergence test expected the log to start at the retry, but the harness log
correctly includes `start:SCHEDULE_POSITION` first. No production code changed for
either failure. One Python test also needed a fixture fix (`mkdir(exist_ok=True)`
after `_legacy_level_dat` already created the save dir, and the directory created
before the save file).

```
pytest tests/test_capture_dataset.py tests/test_litematic.py tests/test_camera.py \
       tests/test_project_store.py
PYTEST_EXIT=0 ; 67 passed, 1 skipped
  tests/test_capture_dataset.py alone: 39 passed, 1 skipped (40 collected)
  the skip is still the external-symlink staging case (Windows sandbox cannot
  create symlinks without elevation); the over-cap and mid-copy-change refusals run
```

End-to-end smoke against a throwaway game directory (a scratch save named
`Template 2`, `--regions`, `--seed 7`, scratch tree deleted afterwards):
`prepare_capture_dataset.py` rc 0 wrote
`staging-game/saves/Template 2/.mc-photo-capture-staging.json` whose `token`
equals the plan's `staging_token` (64 hex characters, `world_identity` =
`Template 2`) plus the provenance copy; `finalize_capture_dataset.py` rc 0 wrote
`index.json` with `summary {'scenes': 1, 'groups': 1, 'repeated_groups': {},
'by_split': {'train': 1, 'dev': 0, 'test': 0}}` and the same staging token in the
scene provenance.

## State left behind (DS-17H)

Only DS-17 paths were touched. This correction added
`tools/minecraft-capture-mod/src/main/java/com/mcphotogrammetry/capture/CaptureDriver.java`
and `.../src/test/java/com/mcphotogrammetry/capture/CaptureDriverTest.java`, and
amended `CaptureController.java`, `StagingMarker.java`, `StagingMarkerTest.java`,
`src/mc_photogrammetry/dataset/plan.py`, `.../inventory.py`, `.../__init__.py`,
`scripts/prepare_capture_dataset.py`, `tests/test_capture_dataset.py`, and this
report. `SUPPORTED_SCHEMA_VERSION`/`SUPPORTED_MOD_CONTRACT_VERSION` remain 1: the
marker payload fields are unchanged, only their location and validation rule
changed, and both sides moved together. The stray scratch script written while
building the smoke check (`tools/ds17h_smoke_check.py`) was deleted in the same
session and is absent; the smoke check ran inline and left nothing behind. Gradle
caches stayed in the owned `<mod>/.lead-gradle-ds17` and the build was fully
offline; no dependency was downloaded.

## Shared risks after DS-17H (still unvalidated without a launch)

- The driver now enforces the order in code and its effect stream is tested, but
  the in-situ tick counts (teleport -> convergence, settings -> settle expiry) are
  still unmeasured. The `SERVER_WAIT_TICKS = 1200` / `GROUND_TRUTH_WAIT_TICKS = 600`
  budgets are per ticket and generous, so a slow first tick records a failure
  rather than hanging.
- `getSavePath(WorldSavePath.ROOT)` is compiled against the cached 1.21.5 Yarn
  mapping and its shape was confirmed with `javap`, but whether the integrated
  server reports exactly the staged directory at runtime can only be seen in a
  launch. If it does not, the marker read fails closed (the job aborts before any
  mutation) rather than mutating the wrong world.
- The provenance copy at `<game>/config/mc-photo-capture/staging.json` is no longer
  read by the mod at all; a stale or edited copy there no longer blocks arming, and
  the in-world marker is the single authority.
- The scenario "world folder differs from LevelName" is proven for the pure marker
  read and the stager, not against a live integrated server.

## Corrections after fourth lead review (2026-09-12T12:42, DS-17I)

The lead accepted the DS-17H controller ordering but rejected launch again for lost
client settings. Four corrections were enumerated; all four are implemented below.
No launch, install, world mutation, dependency change, browsing, delegation,
stage/commit, or out-of-scope path was touched.

**1. Requested client settings were never applied.** `CaptureController` recorded
and restored FOV/gamma/HUD/window but had no call that set them, so every capture
ran with whatever the client already had. The driver now has a fourth stage and
effect: the required sequence is

```
POSITION -> CONVERGE -> APPLY_CLIENT_SETTINGS -> SCHEDULE_SETTINGS -> SETTLE
    -> INVOKE_CAPTURE -> AWAIT_FILE -> DONE
```

`CaptureDriver.onPositionApplied(converged)` enters `CLIENT_SETTINGS` and emits
`APPLY_CLIENT_SETTINGS`; the new `onClientSettingsApplied()` is the only path to
`SETTINGS_SCHEDULED`, so the client options can never be skipped or applied after
the world was mutated. The controller's `applyClientSettings` runs on the client
tick thread exactly once and sets `options.getFov()` (the integer round of the
plan's value), `options.getGamma()`, `options.hudHidden`, and
`Window.setWindowedSize(job.width(), job.height())`, logs the applied values, and
feeds the outcome back. The window resize and option changes therefore land before
the server settings task and the settle window, so they are in effect for the
screenshot. `CaptureDriverTest` is now 7 cases; the new
`clientSettingsAreAppliedExactlyOnceAfterConvergenceAndBeforeServerSettings`
asserts the emitted effect list, the recorded stage order (CONVERGE <
CLIENT_SETTINGS < SETTINGS_SCHEDULED < SETTLE < capture), and that a second
application throws.

**2. Result settings are recorded as request vs actual and validated.** Every
successful frame now writes `actual.requested` with fov (integer round), gamma,
hud_hidden, window and framebuffer width/height, time_of_day, weather, and the pack
names, alongside the flat actual readings, a derived `actual.weather`, and
`actual.daylight_cycle_frozen`. The finalizer gained `validate_settings` (exported
from `mc_photogrammetry.dataset`) and refuses a scene unless the recorded request
matches the plan **and** the actual matches the request for FOV (integer-round
contract), gamma (`GAMMA_TOLERANCE = 0.01`), HUD, framebuffer dimensions, and
time/weather (exact). The screenshot itself is checked too: `png_dimensions` parses
the PNG IHDR directly (a 24-byte read, no image library) and `finalize_scene`
refuses any frame whose PNG size is not the planned resolution, so a
`setWindowedSize` that did not take effect keeps the frame result but blocks scene
finalization.

**3. Captured time/weather could describe a pre-settle instant.** Applying the
capture time left `doDaylightCycle` on, so `timeOfDay` kept advancing during the
settle window while the cached `serverActuals` described the moment settings were
applied. The staged copy's daylight cycle is now disabled in the same server-thread
SETTINGS task, immediately before `setTimeOfDay`, and the frozen state is recorded
from `GameRules.DO_DAYLIGHT_CYCLE`. Rain/thunder gradients are set explicitly after
`setWeather`, because those gradients ramp at 0.01/tick and a sample taken in the
same tick would otherwise still report the previous weather. This mutates only the
throwaway staged copy the plan is armed against, never a source world; the plan is
consumed afterwards, so the rule is deliberately left in place and documented here.

**4. Tests.** New/updated coverage: Java `CaptureDriverTest` 7 cases (was 6);
Python `validate_settings` unit cases (matching settings pass, the FOV 70.4 → 70
round is accepted, a 10-way parametrised
`test_validate_settings_refuses_every_recorded_mismatch` covers every actual and
requested mismatch including a missing `requested` block,
`test_weather_contract_accepts_derived_flags_but_refuses_a_contradiction`), the
`png_dimensions` reader, and the end-to-end
`test_finalize_refuses_a_capture_whose_actual_settings_disagree` and
`test_finalize_refuses_a_screenshot_whose_pixels_do_not_match_the_plan`. The shared
mod-output fixture now writes real PNGs at the planned resolution and the full
request/actual object.

### Evidence (DS-17I)

```
gradlew.bat --project-dir tools\minecraft-capture-mod clean build \
    --offline --no-daemon --no-watch-fs --console=plain
  JAVA_HOME=C:\Program Files\Java\jdk-24
  GRADLE_USER_HOME=<mod>/.lead-gradle-ds17   (lead's cached wrapper + dependencies)
GRADLE_EXIT=0 ; BUILD SUCCESSFUL in 12s ; 10 actionable tasks executed
40 tests completed, 0 failed
  CaptureDriverTest     7 tests, 0 failures   (was 6; +client-settings ordering)
  CapturePlumbingTest   8 tests, 0 failures
  JobFlowTest           5 tests, 0 failures
  PlanAuditTest        10 tests, 0 failures
  StagingMarkerTest    10 tests, 0 failures
artifacts: build/libs/mc-photo-capture-0.1.0.jar         59716 B
           build/libs/mc-photo-capture-0.1.0-sources.jar 33689 B
jar SHA-256     C692202B81563AB4E1C664225515C836C939A57A416E96F355ECA9B1DE8BB845
sources SHA-256 AEC59882827587012F8886082CAFDB037B09641F53C0834B10F5E17043AA04AE
```

The build was green on the first attempt; no production code changed for a build
failure.

```
pytest tests/test_capture_dataset.py tests/test_litematic.py tests/test_camera.py \
       tests/test_project_store.py
PYTEST_EXIT=0 ; 82 passed, 1 skipped in 7.59s
  tests/test_capture_dataset.py alone: 54 passed, 1 skipped (55 collected)
  the skip is still the external-symlink staging case (this Windows sandbox cannot
  create symlinks without elevation)
```

The only first-attempt Python failures were mine in the new tests: the helper was
called as `_settings_result(tmp_path, job, plan)` instead of
`(tmp_path, plan, job)`, so twelve cases raised `AttributeError`; the call sites
were corrected and no production code changed.

**Schema note.** `CaptureSettings` now refuses gamma outside `[0, 1]` and the
planner draws gamma in `[0, 0.1]`. Minecraft's brightness option validates its value
in `[0, 1]` (read from the cached 1.21.5
`SimpleOption$DoubleSliderCallbacks.validate` bytecode), so the previous `[-1, 1]`
request range allowed a plan to demand a value the client silently ignores, which
the new actual-vs-requested check would then have refused forever.
`SUPPORTED_SCHEMA_VERSION` / `SUPPORTED_MOD_CONTRACT_VERSION` remain 1: no field
was renamed or removed with a different meaning, and both sides moved together in
this correction.

**Superseded by this correction:** the DS-17H-era statements that client
FOV/gamma/HUD/window were recorded but not applied, that the framebuffer size was
recorded but not required to match, and that `isRaining()` reflects a weather
request immediately after `setWeather`. The DS-17F/DS-17G/DS-17H structural fixes
(server-thread ownership, per-ticket task identity, atomic writes and disarm, the
in-world staging marker, staging caps/symlink refusal, topology-only split
identity, screenshot digests) are unchanged.

### State left behind (DS-17I)

Only DS-17 paths were touched: `CaptureController.java`, `CaptureDriver.java`,
`CaptureDriverTest.java`, `src/mc_photogrammetry/dataset/{schemas,plan,finalize,__init__}.py`,
`tests/test_capture_dataset.py`, and this report. No stray scratch file, staging
world, or source-world write was left behind; the Gradle build stayed fully offline
against the owned `<mod>/.lead-gradle-ds17` user home. The remaining launch/install
step is unchanged from "Not done (deliberately) and the exact remaining step"
above, and was not performed.

## Corrections after fifth lead review (2026-09-12T12:49, DS-17J)

The lead accepted the DS-17I build and review but withheld launch for missing capture
variation and a landscape-usability gap. Only those two changes were made; no Java
was touched, so no Gradle build was run, and no launch/install, world mutation,
dependency change, browsing, delegation, stage/commit, or out-of-scope path was
touched.

**1. Default plans sampled a narrow gamma band and never varied time or HUD.**
`build_plan` drew `gamma = uniform(0, 0.1)`, always used `time_of_day = 6000`, and
always left the HUD visible, so a dataset could not contain a dark/bright or
day/night or clean-screen contrast at all. The planner now uses
`view_setting_strata(view_count)` (new public helper):

* gamma is stratified evenly over the client's supported brightness range
  `[GAMMA_MIN, GAMMA_MAX] = [0, 1]` with `round(..., 4)`, so every allowed view
  count 4-8 contains both the `0.0` dark endpoint and the `1.0` bright endpoint and
  never repeats a level;
* time cycles the fixed representative set `TIME_OF_DAY_CHOICES = (0, 6000, 12000,
  18000)` - dawn, noon, dusk, midnight - so any 4-8 view plan covers all four and
  always includes both day (6000) and night (18000);
* HUD alternates hidden/visible, so every plan has both clean and ordinary-screen
  captures.

The seed only permutes which view receives which stratum (`rng.shuffle` over a
fixed multiset), so a reseed changes the orbit and the assignment order but can
never drop coverage. An explicit `base_settings` is copied verbatim into every job
and overrides the stratified variation entirely, while orbit generation is
unchanged. `WEATHER_CHOICES` remains a seeded draw (weather is not part of the
required coverage).

**2. Landscape sampling required a hand-typed height.** The CLI refused
`--landscape` without `--surface-y`. A new `spawn_position_from_save` reads
`SpawnX`/`SpawnY`/`SpawnZ` from `level.dat` and raises `DatasetSchemaError` with the
file and cause when the spawn is missing or the file is corrupt - it never invents a
height. `prepare_capture_dataset.py` uses it when `--surface-y` is omitted (printing
the resolved `SpawnY`) and keeps the existing `spawn_from_save` `(x, z)` reader
untouched for display/inventory callers; on failure the CLI adds
"pass --surface-y <height> to override". The explicit developer override still works
and is the only path that runs against a save with no `SpawnY` at all. This is a
smoke-capture convenience: it centres on the spawn block, it is **not** a
chunk-heightmap sample.

### Evidence (DS-17J)

```
pytest tests/test_capture_dataset.py -o addopts="" -q
PYTEST_EXIT=0 ; 60 passed, 1 skipped (61 collected; was 54 passed / 55 collected)
pytest tests/test_capture_dataset.py tests/test_litematic.py tests/test_camera.py \
       tests/test_project_store.py -o addopts="" -q
PYTEST_EXIT=0 ; 88 passed, 1 skipped in 5.89s (was 82 passed / 1 skipped)
  the skip is still the external-symlink staging case (this Windows sandbox cannot
  create symlinks without elevation)
```

Six new dataset cases, all passing:

* `test_every_allowed_view_count_covers_gamma_time_and_hud` - for every view count
  4-8: `{GAMMA_MIN, GAMMA_MAX} <= gammas`, one distinct level per view, all times
  equal `{0, 6000, 12000, 18000}`, and HUD `{False, True}`; the same assertions are
  made directly on `view_setting_strata`.
* `test_setting_coverage_survives_a_seed_change_while_the_orbit_moves` - same seed
  repeats exactly, seeds 11 and 12 keep the same gamma/time/HUD multiset while the
  camera positions differ.
* `test_the_seed_only_permutes_the_stratified_settings` - five seeds each assign a
  permutation of the fixed 6-view strata, and more than one distinct order appears.
* `test_explicit_base_settings_override_the_variation` - every job's settings equal
  the explicit `gamma=0.5, time=13000, weather=rain, hud_hidden=True` request.
* `test_spawn_y_reader_returns_the_full_spawn_and_refuses_a_missing_one` -
  `(7, 71, -9)` round trip; a save without `SpawnY` and a non-NBT file both raise
  `DatasetSchemaError` naming `SpawnY`.
* `test_prepare_landscape_defaults_to_spawn_y_and_honours_an_explicit_override` -
  runs the real CLI in-process: default writes a region `y in [63, 95]` from
  `SpawnY 71`; `--surface-y 70` writes `y in [62, 94]` against a save with no
  `SpawnY`; and the no-spawn/no-override run raises `SystemExit` mentioning
  `--surface-y` without creating any scene. The pre-existing byte-stability and
  fresh-process determinism tests still pass against the new stratum assignment.

The only first-attempt failure was mine in the new test: the seed-coverage case
compared full settings JSON and therefore tripped over the still-seeded weather
(`rain` vs `clear`); it now compares the stratified `(gamma, time, hud)` triple, and
no production code changed for it.

### State left behind (DS-17J)

Only DS-17 Python paths were touched: `src/mc_photogrammetry/dataset/plan.py`,
`.../__init__.py`, `scripts/prepare_capture_dataset.py`,
`tests/test_capture_dataset.py`, and this report. `finalize.py`, `schemas.py`,
`inventory.py`, `hashing.py`, `blocks.py`, `splits.py`, and every Java file are
unchanged, so schema/contract versions stay 1 and no rebuild was needed. No scratch
file, staging world, build artifact, or source-world write was left behind.

### Honest limits after DS-17J

* Setting variation is coverage, not photometric calibration: a plan now guarantees
  dark/bright, day/night, and clean/HUD screen coverage, but no exposure or
  white-balance ground truth is derived from it.
* The height is still `SpawnY` (or an explicit override), not a sampled terrain
  heightmap; a spawn inside a cave or on a roof centres the sample there.
* Resource-pack and shader switching remain unimplemented and are still recorded as
  requested-only (`resource_pack_applied` / `shader_pack_applied` stay `false`);
  true heightmap sampling, pack switching, launch/install, and any visual check of
  the captured views remain future work.

## DS-21 amendment (2026-09-12T13:23-05:00): staged `pauseOnLostFocus:false`

Binding context (lead, 2026-09-12 13:22 -05:00): interactive capture runs on a
hidden, unfocused Windows Desktop 2, so the client default `pauseOnLostFocus:true`
pauses the integrated server and the 13:12 Train Station frames 01-03 recorded the
Game Menu. DS-21 adds **preparation-time Python support only**; DS-20 still owns the
Java screen/pause/reconvergence guards, and no Java, reconstruction, evaluator,
threshold, or existing real artifact was touched.

**Implementation.** New `src/mc_photogrammetry/dataset/options.py` stages the client
option set (`CAPTURE_OPTIONS = {pauseOnLostFocus: "false"}`) into
`<staging game dir>/options.txt` via `stage_capture_options(game_dir)`; it is exported
from `mc_photogrammetry.dataset` together with `render_options`, `options_path`,
`OPTIONS_FILE_NAME`, `PAUSE_ON_LOST_FOCUS`, and `CAPTURE_OPTIONS`.
`scripts/prepare_capture_dataset.py` calls it on the staging game directory (never
the copied world, so the world fingerprint/staging token is unaffected) and prints
the staged path.

**Preservation and idempotence.** Only overridden keys are rewritten: the first
occurrence keeps its position and receives the staged value, later duplicates are
dropped, absent keys are appended in sorted order, and every unrelated line is kept
in order. An absent file becomes a minimal `pauseOnLostFocus:false` file (client
defaults then cover everything else); CRLF vs LF and the trailing-newline state of
an existing file are preserved byte-for-byte (raw byte read/write, no newline
translation). Re-running preparation is byte-idempotent and can never grow a
duplicate key.

**Focused evidence (new `tests/test_capture_options.py`, 10 cases).**
`.\.venv\Scripts\python.exe -m pytest tests/test_capture_options.py tests/test_capture_dataset.py -o addopts="" -q`
-> exit 0, **70 passed, 1 skipped** (the skip is the pre-existing external-symlink
staging case). The new cases cover: the constant set, absent-file creation,
existing-file preservation with in-place `true`->`false`, byte-idempotent re-runs,
duplicate-key collapse to one, missing-key append-once, CRLF/trailing-newline
preservation, `render_options` purity, directory-at-path refusal, and an in-process
CLI run proving `prepare_capture_dataset.py` stages the guard in `staging-game`
while preserving an earlier run's unrelated options. No commit was made.

## Corrections after DS-20 real-capture smoke (2026-09-12, DS-20)

**Focus note (superseded by the DS-20C correction at the end of this report).** The
focus gating and `GLFW.glfwFocusWindow` behaviour described in this section was wrong
for the binding 13:22 constraint: interactive capture runs on a hidden, *unfocused*
Windows Desktop 2, so focus is telemetry only. DS-20C removes the focus gate and the
focus-stealing calls while keeping focus in the recorded evidence. Where this section
says a frame requires a focused window, or that the guard re-focuses and retries, read
the DS-20C section instead.

Unit acceptance was superseded by real Prism/Java21 launch evidence. The staged
smoke dataset `exports/datasets/smoke-train-station-2026-09-12`, plan
`plan-770df5e17e759146`, run 13:12, produced four screenshots, four `captured:true`
results and an 18 MB ground truth, yet the finalizer refused frame00 and the visual
lead review found frames 01-03 showed the Game Menu. The 1-degree finalizer gate
was **not** weakened and no menu frame was accepted; the capture path was fixed.
(DS-21, above, additionally stages `pauseOnLostFocus:false` so the trigger is less
likely; this DS-20 Java guard still refuses the capture if a screen or pause appears
for any reason.)

### Exact smoke evidence (read-only, from the staged run)

Plan `scenes/train-station-spawn-208-96/plan.json`, 4 jobs, `settle_ticks = 20`.
`staging-game/mc-photo-capture/out/plan-770df5e17e759146/results.json`:

| Frame | Planned yaw | Actual yaw | Yaw err | Planned pitch | Actual pitch | Pitch err | Feet err | Bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `...-00` | 178.9284941986 | 180.27852 | **1.3500258** | 14.0271707258 | 13.727171 | 0.2999997 | ~0 | 826173 |
| `...-01` | 84.7673317230 | 84.767334 | ~2.3e-6 | 22.6536187748 | 22.653618 | ~2.8e-7 | ~0 | 190332 |
| `...-02` | -1.2163452858 | -1.2163453 | ~1.4e-8 | 21.6557308214 | 21.655731 | ~2.7e-7 | ~0 | 277695 |
| `...-03` | -87.0086814176 | -87.00868 | ~1.1e-6 | 13.3216970683 | 13.321697 | ~7e-8 | ~0 | 269662 |

The 1.35-degree frame00 error is exactly the planned-vs-recorded yaw difference and
exceeds the unchanged 1-degree finalizer gate; the position was exact, so only the
earlier pose check (taken before the resize and settle window) could have missed it.
`staging-game/logs/latest.log` records
`mc-photo-capture: settings -> SETTLE for view-...-01` at 13:12:20 immediately
followed by `[Server thread/INFO]: Saving and pausing game...`, and frames 01-03
were captured after that while the ordinary pause menu was on screen. The old
`results.json` carried no screen/pause/focus evidence at all, so the contaminated
frames were indistinguishable from clean ones except by eye.

### What changed (Java only; no Python, no threshold change)

**1. The pose is re-asserted after the client settings/resize and re-measured
before the screenshot.** `CaptureDriver` gained two stages and one guard stage:

```
POSITION_SCHEDULED -> CONVERGE -> CLIENT_SETTINGS -> REASSERT_SCHEDULED
    -> REASSERT_CONVERGE -> SETTINGS_SCHEDULED -> SETTLE -> GUARD
    -> CAPTURE_INVOKED -> AWAIT_FILE -> DONE
```

`onClientSettingsApplied()` now emits `FOCUS_WINDOW` followed by
`SCHEDULE_POSITION` (the same server-thread teleport path), so the requested
feet/yaw/pitch are re-applied and re-checked **after** the window resize and any
option change, with the same bounded ladder (`MAX_POSITION_ATTEMPTS = 3`,
`reassertAttempts`). A re-assertion that never re-converges is a terminal `FAIL_JOB`
and can never reach the settle window or capture. `onSettleElapsed()` no longer
invokes capture: it enters `GUARD` and emits `EVALUATE_GUARD`.

**2. One live capture guard decides the screenshot.** New Minecraft-free record
`CaptureGuard` (`ScreenKind NONE | GAME_MENU | UNEXPECTED`, `paused`, `focused`,
`poseConverged`, `poseDetail`). The controller measures it on the client tick with
`client.currentScreen`, `client.isPaused()`, `client.isWindowFocused()` and the same
feet/eye/yaw/pitch comparison; `ready()` is true only for no screen, unpaused,
focused, and pose converged. `CaptureDriver.onGuardEvaluated`:

* ready -> `INVOKE_CAPTURE` (the only way to capture);
* `UNEXPECTED` screen -> terminal `FAIL_JOB`, named in the failure text, never closed;
* paused with **no** screen -> terminal `FAIL_JOB` (nothing can clear it);
* otherwise -> the recovery effects it needs (`CLOSE_MENU` only for an ordinary
  `GameMenuScreen`, `FOCUS_WINDOW` when unfocused) **plus** `SCHEDULE_POSITION`, and
  a fresh settle window before the guard is consulted again;
* bounded by `DEFAULT_MAX_GUARD_ATTEMPTS = 3` (constructor-overridable), then terminal.

**3. Only the ordinary pause menu is closed.** `closeOrdinaryMenu` re-checks
`screen.getClass() == GameMenuScreen.class` at the moment of the close (the screen
can change between measuring and closing), calls `client.setScreen(null)`, and
refuses anything else with a recorded failure. `focusWindow` is a best-effort
`GLFW.glfwFocusWindow(handle)` after that close and after the resize, and
`isWindowFocused()` is re-read by the guard, so focus is never trusted.

**4. Fresh settle after recovery.** Recovery sets a driver `recovering` flag and
returns to `REASSERT_SCHEDULED`; the re-converged recovery path enters `SETTLE`
again without repeating the server time/weather task, and the controller resets
`waitTicks = settleTicks` on every `SETTLE` effect. The settle window that ran while
the menu was open is therefore void, not reused.

**5. The guard/pose state is recorded as actual evidence.** Every captured frame now
also writes `actual.screen_open`, `actual.screen_type`, `actual.ordinary_game_menu`,
`actual.paused`, `actual.focused`, `actual.pose_converged`, `actual.pose_guard_detail`,
numeric `pose_error_feet`, `pose_error_eye`, `pose_error_yaw_degrees`,
`pose_error_pitch_degrees`, and the counters `pose_attempts`, `pose_reassertions`,
`guard_attempts`, `recovery_cycles`, `menu_closes`, `focus_attempts` (alongside the
unchanged request/actual settings block and screenshot digest). The screenshot can
only be taken when that recorded guard state is `screen_open=false, paused=false,
focused=true, pose_converged=true`. The keys are additive; the Python finalizer was
**not** touched in this correction (per scope), so a future finalizer change can
additionally refuse a frame that carries a non-ready guard state.

### Evidence (DS-20)

```
gradlew.bat --project-dir tools\minecraft-capture-mod clean build \
    --offline --no-daemon --no-watch-fs --console=plain
  JAVA_HOME=C:\Program Files\Java\jdk-24
  GRADLE_USER_HOME=<mod>\.lead-gradle-ds17
GRADLE_EXIT=0 ; BUILD SUCCESSFUL in 12s ; 10 actionable tasks executed
49 tests completed, 0 failed
  CaptureDriverTest    13 tests, 0 failures   (was 7; +first-frame drift, +pause
                                              during settle, +unexpected screen,
                                              +bounded guard retry, +paused with no
                                              recoverable screen, +post-settings
                                              divergence)
  CaptureGuardTest      3 tests, 0 failures   (new: ready rule, screen classes,
                                              evidence description)
  CapturePlumbingTest   8 tests, 0 failures
  JobFlowTest           5 tests, 0 failures
  PlanAuditTest        10 tests, 0 failures
  StagingMarkerTest    10 tests, 0 failures
build/test-results/test/TEST-...CaptureDriverTest.xml   tests="13" failures="0" errors="0"
build/test-results/test/TEST-...CaptureGuardTest.xml    tests="3"  failures="0" errors="0"
artifacts: build/libs/mc-photo-capture-0.1.0.jar         66835 B
           build/libs/mc-photo-capture-0.1.0-sources.jar 39003 B
jar SHA-256     40E9908900B0D0C33BD8EDE5EA5D4E8F558A373EDA572F0BA0115C9F54D30247
sources SHA-256 D5751D95D4B2BF9C430D0D10F9EFD0A5618D0F1062A282557B3F5DB87CC1FAF1
```

The flow tests assert the emitted effect list and the exact stage order, not just
enum successors: the clean job is `start:SCHEDULE_POSITION` /
`position:APPLY_CLIENT_SETTINGS` / `client-settings:FOCUS_WINDOW` /
`client-settings:SCHEDULE_POSITION` / `position:SCHEDULE_SETTINGS` /
`settings:SETTLE` / `settle:EVALUATE_GUARD` / `guard:INVOKE_CAPTURE` /
`capture:AWAIT_FILE` / `file:DONE`, with POSITION scheduled exactly twice (pose plus
re-assertion). The DS-20 cases pin each defect:
`aFirstFrameDriftIsCaughtByTheGuardAndNeedsAReconvergenceBeforeCapture` (drift after
the settle emits only `SCHEDULE_POSITION`, zero `INVOKE_CAPTURE`, then a second
`SETTLE` and exactly one capture),
`aPauseDuringSettleClosesOnlyTheOrdinaryMenuAndRequiresAFreshSettle`
(`guard:CLOSE_MENU`, `guard:SCHEDULE_POSITION`, no capture while paused, two settles,
`SCHEDULE_SETTINGS` exactly once),
`anUnexpectedScreenIsRefusedAndNeverCapturedBehind` (`guard:FAIL_JOB`, zero
`CLOSE_MENU`, screen named in the failure),
`aGuardThatNeverBecomesReadyFailsAfterTheBoundedAttempts` (two evaluations then
terminal), `aPausedClientWithNoRecoverableScreenIsRefusedImmediately`, and
`aPostSettingsDivergenceRetriesAndNeverSettlesOrCaptures`.

The first `:test` run failed two of my new cases (assertions checked the *whole* log
for `position:SCHEDULE_SETTINGS`, which also appears on the initial settings path);
they now assert `SCHEDULE_SETTINGS` is emitted exactly once. No production code
changed for either failure.

### State left behind (DS-20)

Only DS-17-owned paths were touched: the mod's new
`src/main/java/com/mcphotogrammetry/capture/CaptureGuard.java` and
`src/test/java/com/mcphotogrammetry/capture/CaptureGuardTest.java`, and
`CaptureDriver.java`, `CaptureController.java`, `CaptureDriverTest.java`, plus this
report and the `AGENT_COMMS.md` completion note. No Python file was edited, so
`SUPPORTED_SCHEMA_VERSION` / `SUPPORTED_MOD_CONTRACT_VERSION` stay 1 and no focused
Python run was needed; the smoke dataset under `exports/datasets/` was read only and
not modified. No staging world was created, no game was launched or installed, no
source world was written, and the Gradle build stayed fully offline against the
owned `<mod>/.lead-gradle-ds17` user home. The rebuild artifact is
`tools/minecraft-capture-mod/build/libs/mc-photo-capture-0.1.0.jar` (66835 B,
40E99089...); the lead restages, recaptures, and finalizes.

### Honest limits after DS-20

* The guard is a client-tick measurement taken immediately before the screenshot; it
  cannot see a screen or pause that appears between that measurement and the
  recorder call in the same tick, but that window is now sub-tick rather than the
  20-tick settle window the smoke run captured through.
* If Minecraft keeps `isPaused()` true after `setScreen(null)` (for example a paused
  overlay), the next guard evaluation fails the frame rather than capturing it; this
  is fail-closed, not recovery.
* `GLFW.glfwFocusWindow` is best-effort: a compositor may refuse it, in which case
  the guard keeps refusing to capture and the job fails after the bounded attempts.
* The guard evidence is recorded but the Python finalizer does not yet read it (no
  Python change was in scope); frames are protected by construction because the
  screenshot is only invoked after the guard passed, and the recorded fields make
  any future violation auditable.
* No launch was performed for this correction, so the in-situ behaviour of the pause
  menu, the resize-induced focus loss, and the recovery tick counts remain
  unexercised here; the lead's next restage/recapture is the first live check.

## DS-20C correction (2026-09-12): focus is telemetry only

Binding background constraint (lead, 2026-09-12 13:22 -05:00, *background capture
constraint*): interactive Minecraft capture runs on Windows Desktop 2 with
`pauseOnLostFocus:false` staged; **focus is telemetry, not an acceptance requirement**;
every frame still requires no open screen, unpaused state, bounded pose convergence, and
uncontaminated pixels. The DS-20 guard gated capture on `isWindowFocused()` and emitted
best-effort `GLFW.glfwFocusWindow` calls. On a hidden, unfocused desktop `focused=false`
is the *normal* state, so that gate would fail every real capture and the focus attempts
were focus-stealing for no acceptance benefit. DS-20C removes the gate, the
focus-stealing calls, and the focus counter, and keeps focus as read-only evidence. It
supersedes the focus wording of the DS-20 section above.

**What changed (capture-mod Java and its tests only).**

1. `CaptureGuard.ready()` is now `screen == NONE && !paused && poseConverged`. The
   `focused` record component is unchanged and is still written verbatim into the frame
   evidence (`actual.focused`) and the guard description, but it cannot block or delay a
   capture. Focus is telemetry only.
2. `CaptureDriver.Effect.FOCUS_WINDOW` is removed. `onClientSettingsApplied()` emits
   only `SCHEDULE_POSITION` (the post-resize pose re-assertion is unchanged);
   `onGuardEvaluated` no longer adds a focus effect for an unfocused window. Recovery is
   emitted only for real contamination - an ordinary `GameMenuScreen` (`CLOSE_MENU`) or
   a pose that has not reconverged (`SCHEDULE_POSITION`) - and still requires a fresh
   settle and is still bounded by `maxGuardAttempts`.
3. `CaptureController` no longer imports `org.lwjgl.glfw.GLFW`; the `focusWindow`
   method, the `focusAttempts` counter, its `focus_attempts` evidence key, and the
   post-menu-close focus call are gone. The read-only `isWindowFocused()` probe is
   retained solely to populate the `focused` evidence field.
4. Preserved unchanged: ordinary `GameMenuScreen` close with the repeated exact-class
   re-check, refusal of any unexpected screen (terminal, never closed), the
   post-settings pose re-assertion, the fresh settle after recovery, bounded guard
   retries for actual contamination/drift, client-option restore, atomic disarm, and
   the staged-world identity guards.

**Tests (Java only).** New
`anUnfocusedCleanUnpausedConvergedFrameIsCapturedWithoutFocusRetry` drives the full clean
flow and asserts an unfocused guard emits exactly `guard:INVOKE_CAPTURE` with zero
`FOCUS_WINDOW`, zero recoveries, one guard attempt, and two position schedules (pose plus
post-settings re-assertion only).
`captureIsAllowedWithNoScreenUnpausedAndAConvergedPoseRegardlessOfFocus` pins the
readiness rule (unfocused + clean + unpaused + converged is ready; paused, drifted, or
screen-open states are refused, focused or not).
`aGuardThatNeverBecomesReadyFailsAfterTheBoundedAttempts` now uses a drifted pose as the
not-ready state, proving the bounded retry is for real contamination.
`ready()` in the driver test is unfocused on purpose, so every clean-path assertion also
proves that an unfocused window captures.

**Evidence (DS-20C).**

```
gradlew.bat --project-dir tools\minecraft-capture-mod clean build \
    --offline --no-daemon --no-watch-fs --console=plain
  JAVA_HOME=C:\Program Files\Java\jdk-24
  GRADLE_USER_HOME=<mod>\.lead-gradle-ds17
GRADLE_EXIT=0 ; BUILD SUCCESSFUL in 11s ; 10 actionable tasks executed
50 tests completed, 0 failed
  CaptureDriverTest    14 tests, 0 failures   (was 13; +unfocused clean capture)
  CaptureGuardTest      3 tests, 0 failures
  CapturePlumbingTest   8 tests, 0 failures
  JobFlowTest           5 tests, 0 failures
  PlanAuditTest        10 tests, 0 failures
  StagingMarkerTest    10 tests, 0 failures
build/test-results/test/TEST-...CaptureDriverTest.xml   tests="14" failures="0" errors="0"
artifacts: build/libs/mc-photo-capture-0.1.0.jar         66392 B
           build/libs/mc-photo-capture-0.1.0-sources.jar 39551 B
jar SHA-256     D5DB3F1F207EC986A60C63843A90E1D0570080CD7F33C92878E41E301BEE4CBC
sources SHA-256 0AF3E339104FCA596F8AE0D45B0FB1196271DCDA342F540E6DCF383F5FE15FEE
```

**State left behind (DS-20C).** Only the DS-20-owned capture-mod paths and this report
were touched: `CaptureGuard.java`, `CaptureDriver.java`, `CaptureController.java`,
`CaptureGuardTest.java`, `CaptureDriverTest.java`, plus this report and the
`AGENT_COMMS.md` completion note. No Python file was edited, so
`SUPPORTED_SCHEMA_VERSION` / `SUPPORTED_MOD_CONTRACT_VERSION` are unchanged and no
Python run was needed; the `focus_attempts` result key was removed, but no Python
consumer reads it (verified by search) and the remaining `focused` telemetry key is
unchanged. No staging world, launch, install, or source-world write occurred. No commit
was made; the lead restages, recaptures, and finalizes.

**Honest limits after DS-20C.** No launch was performed, so the first live check of the
unfocused capture path is the lead's next restage/recapture. `focused` may read `false`
on every frame; it is recorded but not a contract gate. Because focus no longer forces a
retry, a genuinely focus-related contamination (for example a lost-focus pause overlay)
is caught only through the screen/pause/pose signals, not through focus itself.

## DS-22 amendment (2026-09-12T13:37-05:00): capture client starts muted and never takes the mouse

Binding rule (lead, DS-22): **every dataset-factory Minecraft client must start
muted and must not steal mouse input.** DS-22 extends the DS-21 preparation-time
Python staging only; DS-20/DS-20C still own the Java screen/pause/reconvergence
guards, and no Java, reconstruction, evaluator, threshold, existing artifact, or
plan/schema file was touched.

**Implementation.** `CAPTURE_OPTIONS` in `src/mc_photogrammetry/dataset/options.py`
is now exactly:

```
pauseOnLostFocus:false      # unfocused window keeps the integrated server running
soundCategory_master:0.0    # client starts muted
rawMouseInput:false         # client never grabs/steers the physical mouse
```

New named constants `SOUND_CATEGORY_MASTER = "soundCategory_master"` and
`RAW_MOUSE_INPUT = "rawMouseInput"` are exported from `mc_photogrammetry.dataset`
beside `PAUSE_ON_LOST_FOCUS`/`CAPTURE_OPTIONS`. `stage_capture_options(staging)`
is unchanged in mechanism, so all three keys are staged together into
`<staging game dir>/options.txt` (never the copied world, so the staging
token/world fingerprint is unaffected) before the mod is armed.
`scripts/prepare_capture_dataset.py` now prints the full staged set
(`staged client options <path> (pauseOnLostFocus:false, soundCategory_master:0.0, rawMouseInput:false)`).

**Preservation and idempotence (unchanged contract).** Only the three named keys
are rewritten: the first occurrence of each keeps its position and receives the
staged value, later duplicates are dropped, absent keys are appended in sorted
order (`pauseOnLostFocus`, `rawMouseInput`, `soundCategory_master`), and every
unrelated line is kept in order. An absent/empty file becomes a minimal three-line
file; CRLF vs LF and the trailing-newline state of an existing file are preserved
byte-for-byte. Re-running preparation is byte-idempotent and can never grow a
duplicate key.

**Focused evidence (updated `tests/test_capture_options.py`, 11 cases).**

```
.\.venv\Scripts\python.exe -m pytest tests/test_capture_options.py tests/test_capture_dataset.py -o addopts="" -q
```

-> exit 0, **71 passed, 1 skipped** (the skip is the pre-existing external-symlink
staging case; the DS-21 count was 70 passed, +1 net new case). Cases cover: the
exact three-key constant set; absent-file creation with only the three guards;
existing-file preservation with all three guards replaced in place and a
byte-exact delta; absent guards appended in sorted order; byte-idempotent re-runs
with no unsafe `true`/nonzero value left behind; duplicate collapse to one line
per key; append-once keys; CRLF/trailing-newline preservation; `render_options`
purity; directory-at-path refusal; and an in-process CLI run proving
`prepare_capture_dataset.py` stages all three guards into `staging-game` while
preserving an earlier run's unrelated options. No commit/stage was made.

**Honest limits.** DS-22 stages the safe values; it cannot prove the client read
them, and `rawMouseInput:false` relies on the vanilla option name (no mod-side
re-assertion was added, since Java is out of scope). If Minecraft renames an
option, staging silently adds an ignored line and the client default applies for
that key. No launch was performed.

## DS-26 amendment (2026-09-12): `--list` survives a non-cp1252 save name

**Defect (reproduced).** Running the factory inventory against the local
`.minecraft` on this workspace's cp1252 console killed the whole listing:

```
scripts/prepare_capture_dataset.py --game-dir <game dir> --list
...
  File "...\scripts\prepare_capture_dataset.py", line 104, in _print_saves
    print(f"{save.name[:40]:40} {save.size_bytes / (1 << 20):9.1f}  "
  File "...\encodings\cp1252.py", line 19, in encode
UnicodeEncodeError: 'charmap' codec can't encode characters in position 74-85: character maps to <undefined>
-> exit 1
```

Position 74 is the start of the `level name` column, so the failing field was a
`level.dat` `LevelName`, not the directory. The run had already printed 24 of the
57 saves and aborted; every later save was invisible. The active console codec
was `cp1252` (`sys.stdout.encoding`) and the stream was strict, so a CJK world
name could not be written at all.

**Fix.** New `src/mc_photogrammetry/dataset/console.py` provides encoding-safe
terminal output, exported from `mc_photogrammetry.dataset` as `write_safe`,
`write_line`, `escape_for_encoding`, `is_encodable`, `stream_encoding` (and
`ESCAPE_ERRORS`). `write_safe` writes text verbatim when the stream's declared
codec can represent it, and otherwise replaces *only* the unrepresentable
characters with their deterministic Python escapes (`\u9ad8`, `\xXX`), including
lone surrogates. A stream that turns out stricter than its declared `encoding`
gets one ASCII-only retry instead of aborting the caller. `_print_saves` in
`scripts/prepare_capture_dataset.py` now takes an injectable stream and renders
one line per save through the helper; the other informational `print` calls in
that script were routed through the same writer, because staged paths can also
contain non-cp1252 characters. No save is dropped, reordered, or rewritten: only
the console rendering changes.

**Evidence.**

```
.\.venv\Scripts\python.exe -m pytest tests/test_capture_console.py tests/test_capture_dataset.py tests/test_capture_options.py -o addopts= -p no:cacheprovider -q
```

-> exit 0, **83 passed, 1 skipped** (the skip is the pre-existing external-symlink
staging case; the DS-22 count was 71 passed, +12 new DS-26 cases). The new
`tests/test_capture_console.py` covers: a strict cp1252 `TextIOWrapper` with
non-cp1252 save and `level.dat` names completing with one header plus one line per
save; inventory order preserved; an unchanged inventory (names, level names,
sizes, mtimes, and file set identical before and after rendering); UTF-8 streams
preserving the real Unicode names verbatim; `cp1252 == escape_for_encoding(utf8)`
so only unencodable characters change; byte-deterministic repeat renders on both
codecs; an empty inventory on a strict ASCII stream; the helper's per-character
escape/encodability/`write_line` contract; the stricter-than-declared stream retry;
and an end-to-end subprocess run of the real CLI with `PYTHONIOENCODING=cp1252`
returning exit 0 with the escaped record present.

Real re-run, output redirected to files rather than a console, on the same
57-save game directory:

```
scripts/prepare_capture_dataset.py --game-dir <game dir> --list
-> exit 0, stderr empty
-> 57 record lines (58 non-empty lines including the header); the pre-fix run died after 24
-> repeat run byte-identical
-> 1 unencodable LevelName, rendered escaped; the escape round-trips to the exact
   12-character CJK name, and nothing was skipped
```

**Honest limits.** The escape is a rendering choice for a byte console, not a
rename: a cp1252 console shows `\u9ad8...` where a UTF-8 console (for example
Windows Terminal or `chcp 65001` plus `PYTHONIOENCODING=utf-8`) shows the real
name, so scripts parsing the listing must treat escapes accordingly. If a switch
to UTF-8 is available, no escaping occurs and no information is lost. The change
touches no save on disk, no Java, no reconstruction, and no thresholds; no launch,
commit, or staging was performed.




