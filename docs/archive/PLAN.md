# Minecraft Build Reconstruction Plan

> **Historical plan — superseded on 2026-09-07.** Use [AGENT_HANDOFF.md](AGENT_HANDOFF.md)
> for the approved reconstruction plan, current status, acceptance metrics, and agent assignments.
> The text below is retained as history; its architecture, milestones, and completion expectations
> are not current instructions and may contradict the active handoff.

Build a Windows-first PySide6 desktop application that converts arbitrary screenshots, image sets, or video of existing Minecraft Java builds into a confidence-scored sparse block model, lets the user correct the result, and exports a Minecraft 1.21.x `.litematic`. The pipeline prefers multi-view geometry when enough overlapping frames exist, falls back to monocular depth for weak or single-image input, excludes GUI pixels rather than inpainting them, runs ONNX models through DirectML on AMD/Intel/NVIDIA GPUs with CPU fallback, and is explicit that unseen geometry is estimated rather than recovered.

## Phase 1: Foundation and stable contracts

1. Scaffold a Python 3.11 `uv` project with `src` layout, PySide6 desktop entry point, Ruff/Pyright/Pytest configuration, Windows packaging configuration, and CI for CPU-only unit tests. Keep model weights, extracted Minecraft assets, scans, and generated files out of source control.
2. Define core domain contracts: `ScanProject`, `SourceFrame`, `CameraModel`, `GuiMask`, `BlockObservation`, `BlockVolume`, `BlockState`, `Confidence`, `ReconstructionSettings`, and `ExportSettings`. Store occupied blocks sparsely by integer `(x, y, z)` with block state, confidence, contributing frame IDs, and observed/inferred status. Use a versioned project format.
3. Introduce replaceable interfaces for compute providers, depth estimation, camera solving, texture providers, block classification, reconstruction, and structure export. Extract vanilla assets from the user's selected Minecraft 1.21.x client JAR; do not redistribute Mojang textures.

## Phase 2: Input and GUI exclusion

4. Implement image, image-directory, and video ingestion. Retain timestamps, correct EXIF orientation, cap working resolution non-destructively, score blur/exposure, sample by visual change and motion, and deduplicate with perceptual hashes.
5. Build GUI filtering as exclusion masks, not image inpainting. Cover hotbar, crosshair, chat, debug text, scoreboard, boss bars, subtitles, status overlays, and menu frames. Show masks over frames and allow brush/rectangle corrections. Masked pixels must be ignored by feature matching, depth fusion, and texture classification.
6. Validate with fixtures across 16:9, ultrawide, multiple GUI scales, chat/debug overlays, shader/no-shader scenes, and short videos containing duplicate, blurred, menu, and useful frames.

## Phase 3: Reconstruction

7. Use ONNX Runtime with DirectML as the preferred Windows execution provider and CPU as the guaranteed fallback. Begin with a lightweight depth model after verifying upstream and weight licenses.
8. For overlapping images/video, estimate camera poses with a CPU-capable COLMAP or pycolmap integration after GUI masking. Reject weak reconstructions using registration and reprojection metrics. For single images or failed solves, estimate Minecraft axis vanishing points and FOV with user-adjustable hints.
9. Fuse multi-view depth/points into a sparse occupancy/evidence grid. For fallback input, unproject relative depth and fit grid scale, orientation, and origin using dominant orthogonal block planes and image edges. Track positive and free-space evidence and never silently claim occluded blocks as observed.
10. Provide explicit unseen-geometry policies: `surface only` by default, `conservative close` for small holes, and `solid fill` for closed exterior shells. Mark generated blocks as observed or inferred.
11. Validate against synthetic fixtures rendered from known small `.litematic` structures, then held-out vanilla and common-shader captures. Measure camera registration, occupied-block precision/recall, scale/orientation error, and CPU/DirectML runtime.

## Phase 4: Block identity and shader tolerance

12. Build a face-aware vanilla 1.21.x texture index from the selected JAR by resolving block-state variants and model inheritance. Initially restrict export to static full cubes and simple directional blocks whose state can be inferred reliably.
13. Classify visible block faces by rectifying projected patches, excluding GUI/occlusion pixels, normalizing illumination in perceptual color space, and comparing descriptors against the texture index. Fuse votes across faces and frames and preserve confidence.
14. Add palette constraints and bulk remapping in the UI. Keep `TextureProvider` and namespaced registries composable so resource packs and mods can be added later.

## Phase 5: Desktop workflow and corrections

15. Implement a single-window PySide6 workflow with source/frame strip, 2D mask and camera overlay, central VTK voxel preview, properties/palette inspector, and progress/log area. Use a spawned worker process with typed Qt messages for heavy reconstruction and cancellation.
16. Implement essential corrections: orbit/pan/zoom, block picking, add/erase, replace block type/state, box selection, confidence filtering, fill-policy switching, inferred-block visibility, and bounded undo/redo.
17. Add actionable failures for insufficient overlap, unstable camera solves, ambiguous grid scale, unsupported block states, missing models/assets, out-of-memory fallback, and cancellation.

## Phase 6: Litematic export and release

18. Implement a single-region Minecraft Java 1.21.x `.litematic` writer over `nbtlib` with normalized region coordinates, local palette compression, packed block-state arrays, metadata, timestamps, and exact configured patch data version.
19. Add an independent litematic reader/validator. Test palette bit boundaries, negative coordinates, non-cubic dimensions, directional properties, deterministic output, and round trips. Manually open release candidates in matching Minecraft and Litematica.
20. Package a Windows build with PyInstaller in folder-based form. First-run setup detects DirectML, downloads checksum-pinned model weights, asks for the Minecraft client JAR, and verifies disk space.

## Phase 7: Post-MVP extensions

21. Add world-download export behind the existing exporter interface after `.litematic` is stable. Generate or patch a matching-version Java world, place the model at a selected origin, set spawn safely, validate Anvil chunks and `level.dat`, and zip the save.
22. Add resource-pack ZIP and mod-JAR texture providers, broader model/state support, block entities, and optional Minecraft-specific models trained from legally generated synthetic scene/ground-truth pairs.

## Current implementation status

- Project scaffolding and Python packaging: complete.
- Sparse block model with confidence and observed/inferred provenance: complete.
- Atomic project persistence and CLI `init`/`inspect`: complete.
- Deterministic GUI masks and manual rectangle edits: complete.
- Image/video source manifest discovery and still-image deduplication: complete.
- Still-image decoding with EXIF orientation: complete.
- Quality scoring, perceptual deduplication, and sampled video adapter: complete; OpenCV remains required for actual video decoding.
- DirectML/CPU provider discovery boundary: complete.
- Evidence grid and explicit fill policies: complete.
- Deterministic single-image perspective grid-depth baseline with GUI exclusion: complete as an explicitly labeled best-effort depth estimate; hidden blocks are inferred. Image columns remain aligned and inferred depth is filled along camera rays.
- Visible-surface depth semantics corrected: each accepted image cell emits one estimated-depth surface block only; no blocks behind the visible hit are generated.
- Ground-truth litematic fixture comparison and report command: complete.
- Explicit Minecraft camera projection/unprojection boundary: complete; calibration metadata input remains next.
- Template benchmark batch: complete; all nine current screenshots have been processed and compared against the reference. Candidate geometry remains far smaller than the reference, so multi-view fusion is still required.
- Minecraft F3 camera convention and player eye-height correction: complete; yaw 0 points toward positive Z, yaw -90 toward positive X, and F3 feet positions are projected at +1.62 eye height.
- Calibrated SFM prototype: complete; nine-view feature triangulation produces a sparse shell, but strict filtering is currently preferred (24-26 points, with nonzero direct reference agreement) over relaxed filtering (42 points, zero exact state agreement). Camera-pose refinement is the next accuracy bottleneck.
- Compact Template 2 fixture added: three FOV-110 screenshots, exact bounds `(9, 12, 17)`, world origin `(-1008, 107, 8014)`, and 27-state reference palette. Initial calibrated SFM run produced one point, making it a focused correspondence/pose-refinement regression case.
- Compact Template 2 benchmark rerun after targeted F3-overlay masking: still one reconstructed point. This confirms calibrated projection/placement are valid, but the current feature matcher is not yet sufficient for dense Minecraft interior geometry.
- Calibrated visible-surface fusion benchmark: compact fixture produced 352 visible blocks in a `9x11x17` candidate at 36x48 sampling, with 156 direct coordinate matches (18.31%) and 207 best-translated matches (24.30%) against the 852-block reference. Material agreement remains 0 because the current fallback palette is not a reference texture classifier.
- Current best compact candidate: `exports/template2-visible-best.litematic` validates as `9x11x17` with 304 visible blocks, 184 best-translated coordinate matches (21.60%), and 140 direct coordinate matches (16.43%). Visible calibrated fusion is working; exact block materials and remaining surface coverage are not yet verified.
- Local vanilla texture provider: complete; extracts blockstate-backed texture samples from a user-selected client JAR without redistributing assets. The installed 1.21.5 JAR yields 495 valid samples. Textured compact candidate `exports/template2-visible-textured-valid.litematic` validates as `9x11x17` with 304 visible blocks, 184 aligned matches, 140 direct matches, and 8 exact state matches.
- Desktop folder workflow: complete; a folder containing calibrated screenshots, `cameras.json`, and `scene.json` now uses full scene-grid visible fusion, automatically loads a local 1.21.5/1.21.1 JAR when available, and favors one-view candidates for coverage while warning that background candidates require review.
- Desktop responsiveness: complete; reconstruction/export now runs in a background executor, reports loading/reconstruction/export progress through a Qt timer and progress bar, and supports cancellation between scene-grid loops. Headless GUI startup and the full suite pass after this change.
- Texture-index startup cache: complete; the installed 1.21.5 index loads in about 0.70s uncached and 0.05s from cache, with cache invalidation based on JAR size and modification time. This removes the repeated long setup delay in GUI runs.
- Launcher environment: complete; `.venv` now contains the runtime and test dependencies, launchers prefer it, GUI startup passes offscreen, the worker honors pre-cancellation, and `50` tests pass from the exact launcher interpreter.
- Final environment note: runtime imports and diagnostics remain clean after the latest launcher-message correction. A fresh long worker invocation was started, but the persistent PowerShell session stopped returning output, so its completion was not independently observed in this session.
- Texture cache schema invalidation: complete; caches from pre-variant and pre-face-texture extractors are rejected via schema version 2 and rebuilt automatically.
- Full scene-grid benchmark: `exports/template2-scene-grid.litematic` matches 788/852 reference coordinates (92.49% recall), but has 44.54% precision and 60.13% coordinate F1 because background pixels are still forced into block occupancy. Exact block-state accuracy is 1.14%. The requested 80% end-to-end accuracy is not yet reached; occupancy segmentation and face/state classification remain blocking work.
- Final current-file audit: `exports/templates 2.litematic` contains 1,769 decoded non-air blocks in the correct `9x12x17` zero-origin placement. Against `templates 2/Template 2.litematic`, it has 92.49% coordinate recall, 44.54% precision, 60.13% coordinate F1, and 1.14% exact state accuracy. This is verified data, not a claim of completion.
- Latest full audit after classifier experiments: `exports/templates 2.litematic` contains 1,769 decoded blocks, 92.49% coordinate recall, 44.54% precision, 60.13% F1, and 0.63% exact state accuracy. Texture-variation and consensus experiments were measured and reverted when they regressed results. The remaining 80% blocker is true face/texture/occupancy inference, not file generation.
- Surface-only occupancy pruning is now the default scene-grid rule. On Template 2 it reduces the candidate to 764 blocks and achieves 81.69% coordinate recall, 91.10% precision, and 86.14% coordinate F1. This exceeds the 80% geometry target without using the reference as an occupancy oracle.
- Surface-pruned fixture benchmark with reference-only palette and per-view appearance calibration reaches 56.32% exact state accuracy on overlapping candidate coordinates. Overall end-to-end 80% accuracy is not yet reached; face orientation, shader normalization, and state-specific texture matching remain required.
- Adjacency-aware state smoothing is implemented and measured as opt-in. On the current unrestricted shader fixture, strength `0.5` preserved 86.14% coordinate F1 but produced no state-accuracy improvement; because the classifier collapses states before neighborhood reasoning, multi-hypothesis face classification remains the next state-recognition change.
- Multi-hypothesis texture scoring is now active internally: each observation contributes rank-weighted votes from its top four palette states, and adjacency can add a bounded neighbor bonus without changing the project file schema. Reference-palette Template 2 candidate `exports/templates 2.litematic` remains at 764 blocks, 81.69% recall, 91.10% precision, 86.14% coordinate F1, and 47.70% exact state accuracy. Unrestricted final candidate has the same geometry but 0% exact state accuracy, confirming shader appearance calibration/state candidate selection remains the end-to-end blocker.
- Adjacency-aware label smoothing is implemented as a configurable conservative pass. On Template 2 it is neutral at strengths 0.0-1.0 because the current per-pixel classifier collapses candidates before neighborhood context can help; retaining top-two state hypotheses is the next state-classification design change.
- Final verified regeneration after block-center and face-visibility experiments: `exports/templates 2.litematic` contains 1,718 decoded blocks in `9x12x17` zero-origin placement, with 772/852 direct coordinate matches (90.61% recall), 44.94% precision, 60.08% coordinate F1, and 1.30% exact state accuracy. Projected polygon-face occlusion was measured and reverted because it reduced coordinate recall; center-point grid projection remains the production baseline until true face rasterization is implemented.
- State-aware model texture extraction and face-candidate matching: structurally complete and verified against the installed 1.21.5 JAR. The latest unconstrained candidate remains 1,718 blocks, 90.61% coordinate recall, 44.94% precision, 60.08% F1, and 0% exact state accuracy; fixture-only restriction to the ten reference states reaches 44.69% state accuracy. Shader-aware appearance normalization and real face/visibility inference are still required for 80% end-to-end accuracy.
- Per-state screenshot appearance profiles are now benchmarked with block-center and front-most visibility alignment. On Template 2, geometry remains 764 blocks at 86.14% coordinate F1; fixture-only exact state accuracy reaches 56.9% with adjacency strength 0.25. This improves material selection but remains below 80% without a ground-truth palette/model during unknown scans.
- Robust median patch sampling is now active for texture matching. In the fixture-only calibrated/reference-palette benchmark it improves exact state accuracy to 61.64% while preserving 86.14% coordinate F1; the unrestricted vanilla benchmark is still running before it is accepted as the normal default.
- Unrestricted median benchmark completed: terracotta dominance fell to 59 blocks, but exact state accuracy remained 0% because the unconstrained 1,100-state palette contains many appearance-confusable candidates. Median sampling is retained because it removes the pathological solid-terracotta failure; state-family/resource-pack calibration remains the next accuracy task.
- Final tuned texture candidate `exports/templates 2-reference-palette-final.litematic` validates as `9x12x17` with 764 blocks, 86.14% coordinate F1, and 57.33% exact state accuracy using the fixture-only reference palette/profile diagnostic. The requested 80% end-to-end state target remains unmet; the best general geometry target is met.
- Model-family priors are now active in the benchmark path: cube/log candidates are favored over unrelated plant/utility/other states while gradients remain possible. Unrestricted Template 2 candidate `exports/templates 2-family.litematic` retains 86.14% coordinate F1 and raises exact state accuracy to 1.29%. The prior reduces pathological candidates but cannot solve shader ambiguity alone.
- Property-aware variant extraction is now active: each blockstate variant keeps its parsed properties and common omitted defaults, while model-parent texture references are resolved. The installed 1.21.5 index exposes 6,175 variants and 840/852 reference states. The fixture-only variant benchmark reaches 63.79% exact state accuracy with 86.14% coordinate F1.
- The measured model-family prior is now promoted to the default `reconstruct_scene_surfaces` path, so normal CLI and GUI runs receive the same candidate ranking as the benchmark.
- Template finding: the supplied six screenshots plus reference litematic are sufficient for benchmarking. Exact per-view F3 camera position, yaw, pitch, and FOV are needed for calibrated block projection because the PNGs contain no embedded camera metadata.
- Template world transform recorded: local litematic `(0, 0, 0)` maps to world `(-236, 45, 248)` with exclusive size `(161, 47, 180)`; supplied maximum `(-76, 91, 427)` is treated as an inclusive block corner.
- Single-region `.litematic` writer and palette packing: implemented; full NBT round-trip awaits dependency installation.
- Independent litematic validation and CLI validation command: complete.
- Desktop screenshot workflow for baseline reconstruction and export: complete.
- Multi-view camera solving, learned depth inference, block classification, and interactive voxel editing: next stages.

## State accuracy plan

The current compact-fixture audit shows strong coordinate placement but weak block-state recognition. The latest candidate has 696 common coordinates with the reference, but only 16 exact state-name matches; the largest systematic error is `dark_oak_planks -> spruce_log` for 261 coordinates. The fix should be evaluated against decoded block states, not file size or palette overlap.

### Stage A: Make the normal GUI path reproducible

1. Add a reconstruction manifest to every project containing the exact Minecraft JAR path/hash, texture-index cache version, palette size, calibration mode, family-prior settings, patch radius, and reconstruction mode.
2. Make the GUI and benchmark call one shared reconstruction service so they cannot silently use different defaults.
3. Add a GUI option to select the Minecraft JAR and an explicit `calibrated diagnostic` mode. Never use a reference litematic automatically during a normal scan.
4. Acceptance: rerunning the same folder produces the same block-state map and records the configuration that produced it.

### Stage B: Correct state-index fidelity

1. Validate the texture cache schema/version before reuse; invalidate caches created by the old filename-based extractor or old property schema.
2. Preserve every blockstate variant, including parsed properties, model rotations, multipart applications, inherited texture variables, and default properties.
3. Resolve the actual model face map rather than only a set of texture paths. Keep separate north/south/east/west/up/down candidates and model family/shape metadata.
4. Add direct checks that all 852 reference blocks map to an available exact state where the installed client JAR supports that state.
5. Acceptance: at least 840/852 reference states remain representable, including the reference’s stair, slab, log-axis, fence-connection, carpet, torch, and potted-flower states.

### Stage C: Replace average-color classification

1. For each projected candidate block, determine which face is visible from the camera using the block center, camera direction, and neighboring occupancy.
2. Sample a face-aligned patch or several samples inside the projected face footprint, excluding GUI/F3 regions and rejecting patches that cross depth or material edges.
3. Compare normalized patch descriptors against the candidate face texture: robust RGB/chroma, luminance-normalized gradients, local binary patterns, and texture autocorrelation. Keep top-k state hypotheses instead of committing to one color winner.
4. Use family and model-shape compatibility as a gated prior: full cubes can compete with full cubes; logs, stairs, slabs, carpets, fences, and utility blocks require compatible projected shape evidence.
5. Fuse observations across screenshots with confidence-weighted votes. Use adjacency only as a bounded tie-breaker, never as a replacement for face evidence.
6. Acceptance: on the compact fixture, exact state accuracy must improve monotonically over the current 1.87% unrestricted baseline and the reference-palette diagnostic must exceed the current 63.79% result before a change is retained.

### Stage D: Learn shader appearance without ground-truth leakage

1. Keep the fixture-only reference appearance profile tool for diagnosis, but do not use reference blocks in normal scans.
2. Add an unknown-scan appearance estimator using robust scene samples and vanilla texture anchors, with exposure/gamma/shadow normalization and confidence reporting.
3. Generate legally reproducible synthetic Minecraft renders using the installed asset set and shader/lighting perturbations to train or calibrate descriptors. Store only derived descriptors/metadata, not redistributed Mojang assets.
4. Acceptance: compare unrestricted output against both supplied templates; target at least 80% exact state accuracy on the compact fixture and no regression below 80% coordinate F1.

### Stage E: Keep occupancy and UI usable

1. Preserve the current surface-only enclosed-block pruning, which achieved 86.14% coordinate F1 on Template 2.
2. Keep the GUI reconstruction on a worker thread with progress updates per scene-grid chunk and cancellation checks.
3. Add a preview report showing coordinate precision/recall diagnostics only in benchmark mode and confidence/uncertainty in normal mode.
4. Acceptance: the GUI remains responsive, cached texture loading stays near the measured 0.05 seconds, and a cancelled job leaves no partial export presented as complete.

### State-accuracy experiment order

Run experiments in this order and keep only measured wins:

1. Cache invalidation and shared GUI/benchmark configuration.
2. Exact blockstate/model/face metadata.
3. Face-aligned patch descriptors.
4. Shape/family gating.
5. Multi-view state fusion.
6. Unknown-scan shader normalization and synthetic training data.

Do not use the supplied reference palette or reference appearance profiles in the production GUI path; they are evaluation-only tools.

## Verification commands

```powershell
uv sync --extra dev
uv run ruff check .
uv run pyright
uv run pytest
```

For the desktop shell:

```powershell
uv sync --extra desktop
uv run mc-photogrammetry desktop
```

## Important decisions

- Source material is existing Minecraft Java builds, not real-world objects.
- Arbitrary screenshots and video are accepted, but overlapping multi-view footage receives the higher-accuracy path.
- The first interface is a Windows desktop app built with PySide6.
- Compute prefers ONNX Runtime DirectML and always supports CPU fallback.
- Initial recognition targets vanilla 1.21.x textures with tolerance for common shaders.
- GUI regions are masked from evidence, not inpainted.
- `.litematic` is the MVP output. World ZIP export follows later.
- Surface-only reconstruction is the honest default; hidden interiors are never presented as recovered facts.
- Mojang assets and model weights are not committed or redistributed without verified permission.
