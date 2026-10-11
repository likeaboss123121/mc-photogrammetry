# M4 F2B / E3 context export preregistration

Status: preregistered before implementation or live export. This is an evaluator-only artifact and must not enter reconstruction, CLI, or UI inputs.

## Frozen inputs and region

- Scene: `canal-town-architectural-component-bbox`; world: `Canal Town`.
- Capture: `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/capture.json`, SHA-256 `c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6`.
- Target GT: adjacent `ground_truth.json`, SHA-256 `36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a`; target region inclusive `[154,66,361]..[170,88,376]`.
- R2 freeze: `exports/evaluation/frozen-image-run-r2-atom-a1-canal-town-2026-09-12/freeze.json`, SHA-256 `3c4e8e1b63f73748bde1f072d97dbfb4276ebc90c957889123760edcc6a1c90f`.
- The eight ordered image basenames, PNG hashes, and dimensions in R2 were verified against the capture record: 1280x720, 1600x900, 1920x1080, 1920x1080, 1280x720, 1600x900, 1600x900, 1280x720. They are frozen inputs; no screenshot recapture or replacement is permitted.
- Source save: `%APPDATA%\\.minecraft\\saves\\Canal Town`; expected staged-source token `b4aeeb844025284939e51754be3eba6cce16a876d2500bb9d14ab4ba8a48e54e`.
- Context region inclusive `[140,60,340]..[190,130,400]`; dimensions 51x71x61; volume 220,881, below the mod/evaluator 250,000 cap. It strictly contains the target region. All eight recorded camera feet are inside this context AABB and outside the target GT AABB. No coordinate or label adjustment will be made after export.

## Procedure and gates

1. Add a strictly explicit `context_export_only` zero-job plan mode. In the mod, this mode must pass normal plan digest, staging marker, world identity, and integrated-singleplayer checks, then enter the existing bounded `GroundTruthWriter` finish path without teleport, player/client option mutation, or screenshot code. Its terminal status must explicitly identify context-only success, zero frames, successful GT export, and no run failure. Ordinary empty-job plans remain rejected.
2. Use a new isolated output root, `exports/evaluation/context-gt-f2b-canal-town-2026-09-12`, with fail-if-present semantics. Re-inventory the exact source save, require its fingerprint to equal the capture's staged-source token, and copy it to a fresh disposable staging-game directory through the accepted `stage_world` copy/verification path. Do not reuse or modify the finalized capture's staging tree or source save.
3. Provision only the pinned capture-mod JAR, apply the accepted muted/background-safe options to the staged game directory, and launch through the existing disposable Prism worker plus `PrismCaptureLauncher` and `WindowsDesktop2Controller`. Require Desktop 2 to remain the hidden target; master volume 0; `pauseOnLostFocus=false`; `rawMouseInput=false`; no focus, activation, mouse, or keyboard automation. Verify plan/armed/staging identities and a fresh context-only sentinel before accepting output; require confirmed process-tree teardown. Abort on any identity, pose/world sentinel, option, or desktop mismatch. Do not call the screenshot recorder.
4. Convert only the new raw mod `ground_truth.json` into F2A's strict schema-v1 context artifact at `exports/evaluation/context-gt-f2b-canal-town-2026-09-12/context-ground-truth.json`. Bind exact capture and target-GT byte hashes, scene/world, target and context regions, ordered R2 image basenames/hashes/dimensions, and every exported exact block state. Validate it with F2A before scoring.
5. Invoke the unchanged E2 evaluator exactly once on immutable R2 with the explicit frozen capture, target GT, context artifact, pinned vanilla JAR, and palette cache. Write a distinct E3 output/report. Record known/unknown/excluded pixel counts and target-bounded E2 versus context-domain P/R/F1, preserve per-frame denominators, and keep `promotion=false` because exported context alone does not prove full boundary/occluder completeness. Verify R1, R2, E1, E2, capture, target GT, JAR, and cache hashes are unchanged.

No C4, graph, image, evaluator-threshold, reconstruction, normal UI/CLI, source-save, or finalized-capture changes are in scope. The source audit confirmed Python and Java currently reject zero-job plans and the controller unconditionally starts capture jobs; this mode is additive and opt-in. No live run is authorized until the implementation/tests pass and the concrete worker/template/launcher/options preflight succeeds.
