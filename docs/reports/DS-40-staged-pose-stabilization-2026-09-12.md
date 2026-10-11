# DS-40 - Staged capture pose stabilization

Date: 2026-09-12. Scope: capture-mod physics and run-status semantics only. No capture retry, client launch, reconstruction/evaluator edit, or source-save write.

## Failure diagnosis

Inspected failed plan `plan-17790993eaef47c3` in `exports/datasets/fresh-c4-canal-town-2026-09-12/staging-game/canal-town-architectural-component-bbox/mc-photo-capture/out/plan-17790993eaef47c3/results.json` and the staged world's `logs/latest.log`. All four frames were refused by the existing pose guard. During the 20-tick settle, frame 00 diverged by 0.471 blocks; frames 01 and 02 fell 14.554 blocks; frame 03 fell 3.059 blocks and opened the death screen (`likea fell from a high place`). `teleportPlayer` had used zero teleport velocity, but did not disable gravity or clear fall distance, so ordinary player physics moved the camera after each correct teleport. Reassertion retried the same unsafe settle.

`ground_truth.json` is a real 902,807-byte block export with the requested scene identity and region. The old `run_status.json` simultaneously reported `run_failed: true` and `ground_truth_exported: false`, while the log said ground truth was exported. Cause: `writeRunStatus` defined export success as “all frames captured and payload present,” although `finish` exports the staged world's blocks even when a frame failed.

## Change

`PoseStabilizer` provides a Minecraft-free, UUID-gated contract. The session starts only after level identity and the in-save staging marker validate. A server-player tick mixin reapplies no-gravity, zero velocity, zero fall distance, and invulnerability on the integrated server thread before each server-player tick. The client tick head and controller also apply the lock before local player physics. POSITION applies the lock around the exact server teleport. The helper has no position or rotation operation, preserving the requested feet/yaw/pitch and resulting eye pose. The session stops when the plan terminates.

The physics state is deliberately not restored: it affects only the staged-world copy used by the disposable capture worker; the original save is not opened for writing. `ground_truth_exported` now reports a successful non-sentinel payload write independently from frame success, and `all_frames_captured` reports the latter explicitly. `run_failed` remains true for failed frames, so the existing executor still rejects this run.

## Validation

- `tools/minecraft-capture-mod`: `gradlew.bat test --no-daemon --console=plain`, with task-scoped `JAVA_HOME=C:\Program Files\Java\jdk-22`: **54 tests, 0 failures/errors/skips**.
- `gradlew.bat build --no-daemon --console=plain` with the same JDK: **BUILD SUCCESSFUL**; remapped jar contains `PoseStabilizer`, both client/server tick mixins, and the mixin config.
- `.\.venv\Scripts\python.exe -m pytest tests\test_capture_batch_executor.py -k interpret_run_status -o addopts= -p no:cacheprovider -q`: **6 passed, 122 deselected**; failed runs remain rejected when ground truth was exported.
- Jar authority correction: the current rebuilt artifact `tools/minecraft-capture-mod/build/libs/mc-photo-capture-0.1.0.jar` is 71,226 bytes, SHA-256 `ff65f4110e8f8422af60c1bbfb00ca0eb0d44fb4f30b4e7cacef3fd5dc3f1419`. This supersedes the stale 71,219-byte / `9989614...` identity recorded in the earlier report copy; the artifact is still not copied into staging.

The staged failure artifacts were read only. No jar was copied into staging and no queue retry, staging, Minecraft/Prism launch, capture, git stage, or commit occurred. The staged log also contains unknown Forge-attribute warnings; they were not required to explain the measured vertical drift and are unchanged.
