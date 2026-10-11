# DS-43 - First-frame client convergence synchronization

Date: 2026-09-12. Scope: staged capture-mod POSITION completion and client-pose observation. No live retry/launch, source-world write, DSH, Defender, or junction operation.

## Diagnosis and change

Attempt 8 reported that the first server-thread teleport completed while the client still held the login pose (requested frame 00: 162.787/85.460/408.854; client: 124.498/65.063/277.118). Frames 01-03 converged on their first attempt, ground truth exported, and pose stabilization was active. In `CaptureController.apply()`, ticket completion was treated as proof that the client had consumed the network teleport: the code measured the old client pose in that same tick and returned `false` to `CaptureDriver`, which spent one of three POSITION attempts immediately.

After a successful POSITION server task, the controller now polls the unchanged feet/eye/yaw/pitch pose guard once per client tick for at most 20 ticks (one nominal second at 20 client ticks/sec). It advances to settings/reassertion as soon as a measurement passes the existing tolerances. It returns a failed attempt to the driver only after all 20 observations fail; the three-teleport cap and fail-closed terminal path are unchanged. Server-task errors bypass observation and fail on the completing tick. The same gate covers initial and post-settings POSITION operations. No world/player readiness heuristic, arbitrary sleep, pose tolerance, settings, or pose pin changed.

The observation window resets when a POSITION completes, cancels on frame advance/plan termination, and does not alter cleanup order: termination still stops the pose session, restores captured client options, and disarms. The client hook observes after vanilla tick processing, so each measurement sees network updates handled during that tick.

## Validation

- Focused Gradle test for `PositionObservationWindowTest`: **5 passed**.
- PowerShell, task-scoped JDK: `$env:JAVA_HOME = 'C:\Program Files\Java\jdk-22'; .\gradlew.bat test build --no-daemon --console=plain`: **59 tests, 0 failures/errors/skips; BUILD SUCCESSFUL**.
- Coverage pins delayed convergence with one teleport, timeout-only retry consumption through the unchanged three-attempt terminal failure, immediate server-task error handling, initial/reassert/later-frame flow, window cancellation, and pose-session stop at termination.
- Remapped artifact: `tools/minecraft-capture-mod/build/libs/mc-photo-capture-0.1.0.jar`, **74,882 bytes**, SHA-256 `72f3f6fdaf1b78fea6bc0d08cd91525b71543c546d9dd5548027acd9ac0eeb0d`. The jar was verified to contain `PositionObservationWindow.class`; it was not copied to staging.

No capture retry or live client test was run. This validation proves deterministic controller policy and build integration; network timing still requires a separately authorized fresh capture to observe in Minecraft.
