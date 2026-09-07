# Historical handoff - superseded

Archived 2026-09-07. This file preserves the former chronological handoff for evidence only.
Use [the active AGENT_HANDOFF.md](../../AGENT_HANDOFF.md) for current decisions and assignments.
Old recommendations, completion claims, and metric definitions below may conflict or be obsolete.
Original SHA256: F11F89167CFD0808BDF8177245F3338B670E50FD1689D24F89B68F9EAB448588
The original file bytes follow the separator and its blank line unchanged.

---

# Agent Handoff — MC Photogrammetry

- Date: 2026-09-05
- Workspace: `c:\Users\Likea\Documents\mc photogrammetry project`
- State: Phases 0-2 measurement, palette, projection, face sampling, texture matching, pruning, AO, and runtime work are implemented. **The focus changed on 2026-09-07: see section 67.** Camera manifests are a test scaffold rather than part of the design, pose has to come from the images, and space carving is withdrawn in favour of lattice fitting. Read all of this before touching code.
- Companion docs: `PLAN.md` (long-form history + original phase plan), `README.md` (CLI commands).

## State of play (2026-09-07) - read section 67 first, then 44, 50, 54 and 55

Section 67 changes the direction the rest of this file was heading in. Everything
before it is still true as measurement; its recommendations about *what to build
next* are not, and the two places that pointed at space carving say so in place.

The rest of this file is chronological and long. The newest sections carry
the current picture; everything before them was measured against a surface metric
that section 31 corrects.

Steps and milestones in the plan sections carry a ✔️ once they are done. A step
that was attempted and stopped says so in place of the mark, with the section that
records why. Anything unmarked and unannotated has not been started.

| fixture | gamma | visible geometry F1 | state of placed | state of all visible |
|---|---|---|---|---|
| templates 2 | 1600% | **94.43%** | **85.64%** | 79.71% |
| templates 6 | 100% | 82.90% | 33.35% | 30.20% |
| templates 5 | 100% | 85.15% | 0.00% | 0.00% |
| templates 4 | 100% | 10.37% | 0.67% | 0.17% |

Quote **both** state columns. "State of placed" is survivorship-prone: anything
that discards hard cells flatters it (51). Where the visible error goes, measured
by 44 and 50: Template 2 is 75.7% partial blocks, Template 6 is 81% material.

Two problems remain, and they are separate:

1. **The geometry stage emits the region's bounding shell on every fixture**
   (35). Every region cell is the nearest along some ray in some frame, so the
   recovered "surface" is the region boundary by construction. It scores well on
   three fixtures only because their builds fill their boxes; Template 4, whose
   cameras sit in a sealed interior room, is where the same wrong answer stops
   flattering. Nothing recovers interior geometry anywhere. The fix is free-space
   carving from the camera positions (8n step 2), superseding 13's graph cut.
2. **Dim scenes are never un-lit** (30, 33, 34, 52, 59) - and 59 shows this is
   *the same problem as 1*, not a second one. The per-channel cast is a physical
   constant and the correction is sound: on Template 5's reference geometry it puts
   `diamond_block` at median rank **6** with **30.7%** top-1, beating the
   `pale_oak_log` the pipeline actually emits by 73,904 faces to 349. The pipeline
   fails because 97% of what reaches its classifier is centre-patch fallback from
   cells that are not surface (37). Fix the evidence and the lighting is already
   done. `CORRECT_LIGHT_CAST` ships off; on Template 6 it is cosmetic (52).

Template 2 needs neither and is the fixture where the pipeline works end to end.

Iterate with `scratchpad/crop.py`, which runs the real pipeline on a sub-box of a
big fixture: a diamond-dominated corner of Template 5 is two minutes rather than
ninety-five (37). **Both that harness and `pxbench.py` used to force
`adjacency_strength=1.0`**, so anything they called "shipped" before 2026-09-07 was
not; both now let the library default stand.

### Shipped defaults, and where each was measured

| constant | value | why |
|---|---|---|
| `ADJACENCY_STRENGTH` | **0.55** | 46; was 0.0, the worst value on both metrics |
| `SHAPE_WEIGHT` | 6.0 | 39; inert while the palette is all full cubes |
| `CAST_PERCENTILE` | **99.0** | 55; the maximum over-corrects on large scenes |
| `LIGHT_CAST` | (1.234, 1.002, 0.763) | 34; measured on three fixtures |
| `LIGHT_CAST_DAMPING` | 1.0 | 52; 0.8 suits Template 6, 1.0 suits Template 5 |
| `DIM_SPREAD_THRESHOLD` | 1.9 | 33; separates fullbright from lit |
| `CORRECT_LIGHT_CAST` | **off** | 52; cosmetic on Template 6, pending 55 on Template 5 |
| `STEREO_INTERIOR` | off | 43; net positive on Template 4 only |
| `ADOPT_BOOST` | 0.0 | 51; measured worth 0.2pp |
| `FALLBACK_RISK` / `UPWARD_RISK` | 3.0 / 11.0 | 46; both measured inert, kept as they cost nothing |
| `REQUIRE_RESOLVED_FACE` | off | 51; raises the rate and loses 213 correct blocks |
| `FIT_PARTIAL_SHAPES` | off | 64; opt-in, 4x the runtime for +4 blocks |
| `PROPOSAL_LIMIT` | **40** | 65; was 20, before the fit could read texture |
| `TEXTURE_WEIGHT` | **64.0** | 65; grain decides material outright, render is fallback |

Every one of these is a *measured operating point*, not a robust optimum. The
lighting constants in particular are sharp: one point of `CAST_PERCENTILE` either
side of 99 takes Template 5's crop from 48% to nothing.

---

## 1. Project scope (user's words, paraphrased)

A photogrammetry tool that takes an arbitrary number of images or a video of a Minecraft build or terrain, identifies each visible block's relative coordinates (shape) and type (blockstate), fuses multiple images into one best estimate, and exports a `.litematic` (works today) or a `world.zip` (not built yet).

User acceptance gate: **>= 80% exact block-state accuracy on the Template 2 fixture** (`templates 2/`) with no regression of coordinate F1. Exact state = `BlockState.__eq__` (name + non-default properties), measured over coordinates common to reference and candidate. Current archived geometry is `94.70%` coordinate F1; reports must also show whole-reference exact-state rate so missed coordinates remain visible.

---

## 2. Why the last ~10 iterations showed zero movement (verified root causes)

### 2a. The terminal was a Python REPL, not PowerShell
A bash heredoc (`python.exe - <<'PY'`) is invalid in PowerShell. It launched `python.exe -` reading stdin, i.e. an interactive REPL. Every later command (pytest, benchmark, probes, file writes) was typed INTO that REPL and swallowed; the tool reported "Command produced no output". The terminals named `python` / `py` in the terminal list are those REPLs.

Consequence: `benchmark_current.txt`, `pytest_reconstruction.txt`, `mismatch_probe.json`, `tmp_debug_summary.txt`, `benchmark_after_*.txt` are all STALE (pre-compaction). **No edit made in the last session was ever measured.**

### 2b. The benchmark ran on the 8-colour fallback palette
Every wrong label in the stale confusion report is a `DEFAULT_PALETTE` entry from `src/mc_photogrammetry/reconstruction/facade.py` (`dirt`, `oak_planks`, `white_concrete`, `stone`, `black_concrete`, `bricks`). The correct answers (`dark_oak_planks`, `birch_planks`, `dark_oak_log`) were never candidates, so exact-state accuracy had a hard ceiling near 0% regardless of scoring. Probable cause: `VanillaJarTextureProvider.load_jar` returned 0 (cache miss path or empty cache) and the scripts silently fell back to `DEFAULT_PALETTE`.

### 2c. Heuristics were tuned blind
`src/mc_photogrammetry/reconstruction/raycast.py` currently contains name-token wood/dirt hacks that were never validated against a real run. **User approved reverting them** (see §5).

### 2d. Historic numbers that calibrate expectations (from PLAN.md)
- Unrestricted JAR palette (6,175 states): ~1.3% exact state.
- Reference-palette + reference-derived per-image calibration (diagnostic; uses ground truth): 56–64%.
- Gap = palette confusability + lighting/shader shift + mean-colour-only descriptor. The classifier work in §6 Phase 2 targets exactly those three.

### 2e. Template 2 composition (important for strategy)
Of the 696 common coordinates in the stale run: ~398 `dark_oak_planks`, ~209 `birch_planks`, ~87 `dark_oak_log`. Getting those three states right, and not confusing them with the other ~6,000 palette states, is essentially the whole 80% target. Cameras are INSIDE a fully-lit room (FOV 110, poses in `templates 2/cameras.json`, bounds in `templates 2/scene.json`). Geometry currently comes from projecting the known AABB grid (fixture-specific; general footage needs Phase 4).

---

## 3. Environment facts (verified)

- Interpreter: `.venv\Scripts\python.exe` = Python 3.10.11. Installed: numpy 2.2.6, opencv-python 5.0.0.93, Pillow 12.3, nbtlib 2.0.4, PySide6 6.11.2, pytest 9.1.1. **No pycolmap.**
- The package is **not installed** in `.venv` (no dist-info, no `.pth`). Imports need `$env:PYTHONPATH='src'` (what `launch.ps1` does) or an editable install.
- **No git repository** in the workspace (no `.git`). Reverts are manual edits.
- PowerShell `>` and default `Out-File` write UTF-16LE; tools then see binary. Write result files from Python with `encoding='utf-8'`, or use `Tee-Object -Encoding utf8`.
- Terminal stdout capture has been unreliable in this environment even when commands ran. Treat a file written by Python and read back as the only proof.
- Minecraft JAR: `%APPDATA%\.minecraft\versions\1.21.5\1.21.5.jar`. Texture cache beside it: `1.21.5.mc-textures.json` (`cache_schema: 11`, 6,174 extracted variants in the current cache). The canonical full-cube palette is geometry-filtered, biome-tinted appearances are excluded, and visually indistinguishable states are deduplicated.
- `pyproject.toml`: hatchling backend, `[tool.pytest.ini_options] testpaths=["tests"]`, ruff/pyright configured. Optional extras: `desktop`, `dev`.

### Session start checklist
1. Close every terminal named `python` / `py` (or send `exit()` to each). Open a fresh PowerShell terminal.
2. Prove execution works: `.\.venv\Scripts\python.exe -c "from pathlib import Path; Path('exports/_ping.txt').write_text('ok', encoding='utf-8')"` then read `exports/_ping.txt`.
3. Make imports work without env vars (pick one):
   - `.\.venv\Scripts\python.exe -m pip install -e . --no-deps` (needs hatchling fetch), or
   - offline: create `.venv\Lib\site-packages\mc_photogrammetry.pth` containing the absolute path to `src`.
   Then add `pythonpath = ["src"]` under `[tool.pytest.ini_options]` in `pyproject.toml`.
4. `.\.venv\Scripts\python.exe -m pytest -q` from a shell with no `PYTHONPATH` must pass before anything else.

---

## 4. Current uncommitted state of the code (what to revert/keep)

`src/mc_photogrammetry/reconstruction/raycast.py`:
- `_palette_candidates`: `brown_tone` / `wood_like_rgb` / `warm_wood_like` detection with ±300/±260 distance bonuses for wood vs "earth" name tokens, extra `log`/`stairs` family bonuses, `"wood"` token. **Revert to plain face-distance + descriptor distance.**
- `_choose_state_with_local_family_support`: no-neighbour override returning the best wood candidate if within 6.0 of the best earth candidate; extra `wood_like`/`earth_like` bonuses inside `score()`. **Revert; replaced by Phase 2 step 12.**
- `reconstruct_scene_surfaces`: candidate `limit=8` (was 4). Keeping top-K around 5–8 is fine and is needed by step 11.
- `DEFAULT_FAMILY_PRIORS` (`log: 2.0`, `cube: 1.35`, …): flatten in step 6.

`tests/test_reconstruction.py`: three tests encode the hacks (`test_palette_candidates_prefers_wood_family_for_brown_pixels`, `..._birch_like_tones`, `test_choose_state_prefers_wood_family_when_family_support_outranks_earth`). Delete them with the hacks; replace with tests for the new classifier (face resolution, self-calibration, fusion).

Root-level debris to delete (all one-off, superseded by Phase 0): `tmp_*.py`, `tmp_*.txt`, `probe*.txt`, `probe_output.txt`, `benchmark_*.txt`, `pytest_*.txt`, `verification.txt`, `mismatch_probe.json`, `benchmark_summary.txt`. Keep `scripts/benchmark_templates.py` and upgrade it.

Other fixed/working pieces (do not touch unless a test says otherwise): Litematic writer/reader/validator, `compare_litematics`, camera projection (`project_points`, yaw/pitch conventions, eye height), `SceneTransform`, surface-only enclosed-block pruning, texture cache invalidation, GUI worker/progress/cancel.

---

## 5. User decisions (binding)

1. Revert the blind heuristics and rebuild a principled classifier.
2. Camera poses for arbitrary footage: **full structure-from-motion (pycolmap)**. F3-overlay OCR is only a documented fallback if SfM fails on repetitive Minecraft textures.
3. `world.zip` export only **after** the 80% exact-state gate is passed on Template 2.
4. Reference litematic, reference palette and `calibrate_appearance` (reference-based) are evaluation-only; never in the production CLI/GUI path.

---

## 6. Plan

A step marked ✔️ is done and verified; the section named beside it carries the
result. An unmarked step is outstanding, and where it was attempted and parked
rather than never started, it says so.

### Phase 0 — Measurement you can trust (blocks everything)
1. ✔️ Session start checklist above (kill REPLs, pythonpath, editable install, green pytest).
2. ✔️ Upgrade `scripts/benchmark_templates.py`:
   - `--report <path.json>`: written from Python, UTF-8, containing `palette_source` (`jar`|`default`|`reference`), `palette_size`, `sample_count`, calibration mode per image, coordinate recall/precision/F1, `exact_state_matches`, `common_coordinates`, `state_accuracy`, top-25 confusion pairs `(reference_state, candidate_state, count)`, per-reference-state accuracy.
   - Exit non-zero if the palette resolved to `DEFAULT_PALETTE` unless `--allow-default-palette`.
   - Extend `compare_litematics` (`src/mc_photogrammetry/export/compare.py`) to expose the confusion pairs and per-state counts instead of recomputing them in scripts.
3. ✔️ Delete the debris listed in §4. Save a repo memory note: run Python only via `python -c` or script files; never heredocs; write results from Python.

Status: **complete in implementation, but the current best result must be reproduced by the maintained benchmark before it is used as the acceptance artifact.** The benchmark now rejects fallback palettes, records provenance and gate status, and the unrestricted JAR baseline is archived under `exports/baselines/`.

Current reporting requirement: every winning run must include conditional common-coordinate accuracy, whole-reference exact-state rate, direct and best-translation coordinate metrics, best translation, per-state counts, effective classifier settings, palette/cache provenance, runtime, and test status.

### Phase 1 — Palette provenance + true baseline
4. ✔️ `VanillaJarTextureProvider.load_jar` (`src/mc_photogrammetry/blocks/textures.py`): treat an empty cached `samples` list as a cache miss and re-extract; expose the source (`cached`/`extracted`). Test: cache file with `"samples": []` → re-extraction. Add a visible warning in `src/mc_photogrammetry/ui/main_window.py` and `src/mc_photogrammetry/cli.py` when the palette falls back to `DEFAULT_PALETTE`.
5. ✔️ Run the Phase-0 benchmark once with the JAR palette; archive the JSON under `exports/baselines/`. This is the real unrestricted baseline. Every later step must beat the previous report.

Status: **complete.** Empty-cache re-extraction, cache source reporting, schema invalidation, fallback warnings, baseline archiving, and missing-JAR rejection are implemented and covered by tests.

### Phase 2 — Principled state classifier
6. ✔️ Revert the hacks in `_palette_candidates` and `_choose_state_with_local_family_support`; delete their tests. Flatten `DEFAULT_FAMILY_PRIORS` to ~1.0 for cube/log/planks/slab/stairs; keep only mild penalties for plant/utility/other.
7. ✔️ Per-face texture resolution: extend `_resolve_model_textures` to read model `elements[].faces` (and the `cube_all` / `cube_column` / `cube_bottom_top` / `cube` parent conventions) so each `TextureSample` records `face_textures: {up, down, north, south, east, west}` with per-face mean RGB and descriptor. Bump `CACHE_SCHEMA_VERSION` to 5. Test with a synthetic JAR (pattern in `tests/test_texture_cache.py`).
8. ✔️ Canonical appearance palette: build the classifier palette from placeable full-cube-ish states (families cube/log/planks/stairs/slab, concrete/terracotta/wool/stone-likes); drop item frames, doors, signs, rails, redstone parts, etc.; collapse states whose six face colours are identical into one canonical default-property state (e.g. `axis=y` log). Target ≈150–300 appearance classes. Keep the full 6,175-state index available for later shape-gated expansion.
9. ✔️ Face-aware sampling in `reconstruct_scene_surfaces`: for each front-most block compute visible faces via `dot(face_normal, eye − block_center) > 0` (eye = position + eye_height), project the face's four corners with `project_points`, sample a 3×3 grid inside the projected quad, skip faces smaller than ~6 px, score each visible face only against that face's texture. Remove the single centre-pixel patch path.
10. ✔️ Ground-truth-free exposure self-calibration (`src/mc_photogrammetry/reconstruction/texture_calibration.py`, new function beside `calibrate_appearance`): per image, classify with identity gain, take the top ~30% most confident faces, fit a robust per-channel gain (median of observed/expected), reclassify. Production path uses only this; the reference-based function stays diagnostic.
11. ✔️ Likelihood fusion: per block accumulate `−distance²/σ²` per candidate across all faces and views, weighted by projected face area and inverse depth; keep the top-K=5 candidates per block (extend the existing `candidate_scores` structure). Descriptor = median chroma + luminance + local contrast + gradient energy of the face patch vs. the same descriptor of the face texture.
12. ✔️ Bounded neighbour pass: one ICM iteration that may switch a block only among its own top-K when a 6-neighbour majority agrees and the likelihood gap is below a threshold. This replaces the name-token family logic entirely.
13. ✔️ Benchmark after each of steps 9–12; keep only measured wins. Done when unrestricted Template 2 exact-state >= 80% on common coordinates with coordinate F1 >= 86%, and `dark_oak_planks → dirt` no longer appears in the top-25 confusion pairs.

Status: **Phase A classifier work is implemented and measured.** The current code includes corrected projection handedness and vertical-FOV handling, rotated model-face baking, face-aware rectification, full-texture correlation, canonical palette coverage/pruning, ambient occlusion, self-calibration safeguards, lighting simulation with self-disable, vectorized scoring, and runtime improvements. The remaining exact-state gap is dominated by partial-model occlusion and fallback evidence, not another global color-distance parameter.

The next Phase 2 execution sequence is:

14. ✔️ Freeze and reproduce the current best Template 2 baseline from the maintained benchmark. Treat `exports/t2-ao.json` (`82.65%` conditional state accuracy, `94.70%` coordinate F1) as the strongest archived reproducible comparison until the handoff's later `85.35%` result is regenerated into a JSON artifact with effective settings and timings.
15. ✔️ Add diagnostic-only rejection counters to `face_sampling.py` and the reconstruction report: no front face, clipped/off-screen, minimum-area, visible-fraction, GUI-mask, uniformity, and centre-patch fallback. Record resolved-face versus fallback counts and upward-face evidence share.
16. ✔️ Extend `ModelAtlas` and texture-cache metadata to preserve arbitrary model elements, not only complete full cubes: transformed element bounds, world-facing polygons, UV order, texture/alpha metadata, variant rotations, and multipart composition. Start with slabs, carpets, and stairs; add fences/torches after the cuboid path is validated.
17. ✔️ Replace binary neighbor culling and reduced painter ordering with geometry-aware projected depth masks. Rasterize model polygons, compare per-pixel depths, and compute visible fractions from surviving pixels. Preserve the current centre-pixel coordinate pass as a fallback while validating the new occlusion path.
18. **Parked, not abandoned** (8p step 6, 22, 23, 29): shape selection reaches
    36.8% top-1 against a 40% floor, and 29 settled that more views do not fix
    it - the residual ambiguity is staircase *orientation*. Add shape-gated
    partial candidates. Keep the full-cube palette as the first pass; admit partial states only when projected silhouette, depth, visible area, cross-frame support, and texture evidence agree. Never admit a partial state merely because its color is close.
19. ✔️ Re-evaluate adjacency only after geometry is corrected. Compare no adjacency, the current setting, and bounded candidate-restricted ICM. Do not increase global coupling blindly because it erases isolated log inlays.
20. ✔️ Re-run Template 2 and Template 4 after each isolated geometry milestone. Require no coordinate-F1 regression, a measurable reduction in fallback/false-visible upward samples, nonzero partial-state accuracy, and improvement over the frozen baseline.

### Phase 3 — Second fixture (parallel with late Phase 2)
14. **Outstanding** - the fixture needs region cropping first (1,362,060 cells).
    Add `templates/` (exterior build, 6 screenshots, region 161×47×180, world origin (-236, 45, 248) per PLAN.md) to the benchmark report so the classifier is not tuned to one interior room. Verify first whether `templates/` has `cameras.json`; if not, poses must come from F3 values or Phase 4. Require no regression there. If exterior shading gaps are large, add Minecraft face-shading constants (top 1.0, N/S 0.8, E/W 0.6, bottom 0.5) to the expected face colour before self-calibration.

### Phase 4 — True photogrammetry: SfM camera poses

Status: **recommend closing without the spike; awaiting the user's word.** The
evidence is in 8o, 8q and 20: the existing SIFT path triangulates **1 cell** on
Template 2 and **6** on Template 4, because Minecraft's repeated textures defeat
descriptor matching, and COLMAP would meet the same wall - which is the fallback
step 15 already anticipated. Meanwhile F3-overlay poses are demonstrably good
enough: with correct visibility, two views of a genuinely exposed face agree at
median 0.540, and pose verification by wireframe overlay has been exact on every
fixture authored so far (Templates 4, 5 and 6). Authoring a manifest from the HUD
overlay takes one contact sheet per few frames and is recorded as the supported
method. The gate in 7 item 6 - pycolmap registering 3/3 frames under 2px - should
be struck rather than attempted.
15. Spike `pycolmap` (add as optional extra `sfm` in `pyproject.toml`; if no Windows wheel for 3.10, fall back to the COLMAP CLI via subprocess) on GUI-masked Template 2 frames and the exterior frames. Measure registered-image count and reprojection error. Decision point: if COLMAP fails on repetitive textures, implement F3-overlay OCR of `XYZ` / `Facing` to seed poses.
16. Grid fit: estimate block scale/orientation/origin from the sparse point cloud (dominant orthogonal planes + 1-block periodicity); generate `SceneTransform` bounds automatically (replaces hand-written `scene.json`); use point-cloud occupancy instead of the fixture's AABB-shell projection.
17. Video: feed `iter_video_frames` keyframes (`src/mc_photogrammetry/input/video.py`, already exists) through the same SfM path.

### Phase 5 — world.zip export (only after step 13 passes)
18. ✔️ Anvil exporter behind the existing exporter interface: `level.dat`, region `.mca` chunks, safe spawn, zip; validate with an independent round-trip read; document in README.

Status: **complete (25).** `export/anvil.py` plus `--format world` on the CLI,
verified by an independent reader and thirteen tests.

---

## 7. Verification (proof required at each gate)

1. Phase 0: benchmark JSON shows `palette_source: "jar"`, `palette_size > 100`; missing-JAR run exits non-zero; `python -m pytest -q` passes from a fresh shell with no `PYTHONPATH`; reports include source/configuration/timing metadata.
2. Phase 1: `test_empty_cache_triggers_reextraction` passes; baseline JSON is archived under `exports/baselines/`; cache schema and `texture_source` are recorded.
3. Phase 2: each isolated report is compared with the frozen current baseline; conditional state accuracy and whole-reference exact rate are both reported; coordinate F1 must not regress; partial-state accuracy must become measurable after geometry support; final conditional state accuracy must reach 80% and the whole-reference rate must also be reviewed explicitly.
4. Partial geometry: synthetic tests prove slab/carpet/stair polygons, variant rotations, multipart metadata, per-pixel depth ordering, clipping, and visible-fraction accounting; diagnostic counters show fallback and rejection causes.
5. Phase 3: same report on `templates/` shows no regression.
6. Phase 4: pycolmap registers 3/3 Template 2 frames with reprojection error < 2 px; auto-fitted `SceneTransform` matches `scene.json` within ±1 block.
7. Phase 5: `world.zip` loads in Minecraft 1.21.x; independent reader round-trips block states.

Never report a number that did not come from a file written by the run you just executed.

---

## 8. File map

- `src/mc_photogrammetry/reconstruction/raycast.py` — `reconstruct_scene_surfaces`, `_palette_candidates`, `_choose_state_with_local_family_support`, `_smooth_adjacent_states`, `_patch_descriptor`, `_remove_enclosed_blocks`. Phase 2 lives here.
- `src/mc_photogrammetry/reconstruction/face_sampling.py` — projected face quads, rectification, visibility fractions, uniformity filtering, and the next partial-model depth-mask work.
- `src/mc_photogrammetry/blocks/textures.py` — `VanillaJarTextureProvider.load_jar/palette`, `_model_variants`, `_resolve_model_textures`, `_model_family`, `_complete_properties`, cache load/save, `CACHE_SCHEMA_VERSION`.
- `src/mc_photogrammetry/blocks/model_atlas.py` — rotated full-cube face baking, UV/corner orientation, and the next arbitrary-element/multipart geometry extension.
- `src/mc_photogrammetry/reconstruction/camera.py` — `CameraModel` (yaw 0 → +Z, yaw −90 → +X, eye_height 1.62), `project_points` → `(x, y, depth) | None`, `unproject_pixel`, `load_camera_manifest`.
- `src/mc_photogrammetry/reconstruction/scene.py` — `SceneTransform(world_origin, size)`, `to_world/to_local`, `load_scene_transform`.
- `src/mc_photogrammetry/reconstruction/texture_calibration.py` — `AppearanceCalibration(scale, offset).apply`, `calibrate_appearance` (reference-based, diagnostic only).
- `src/mc_photogrammetry/reconstruction/facade.py` — `PaletteColor(state, rgb, face_rgbs, family, descriptors)`, `DEFAULT_PALETTE`, single-image fallback `reconstruct_facade`.
- `src/mc_photogrammetry/domain/models.py` — `BlockState` (default-property normalisation in `__eq__/__hash__`), `BlockObservation`, `BlockVolume`, `ScanProject`.
- `src/mc_photogrammetry/export/compare.py` — `load_litematic` → `LitematicSnapshot(blocks, palette, size, position, non_air_count)`, `compare_litematics`, `_best_translation`.
- `src/mc_photogrammetry/export/litematic.py` — `LitematicExporter.export(volume, destination, *, name, description)`.
- `src/mc_photogrammetry/cli.py`, `src/mc_photogrammetry/ui/main_window.py` — duplicated palette/JAR autodetect + `reconstruct_scene_surfaces` calls; unify into one service during Phase 1/2.
- `scripts/benchmark_templates.py` — the benchmark to upgrade in Phase 0.
- `templates 2/` — fixture: 3 PNGs, `cameras.json`, `scene.json`, `Template 2.litematic`, `location.md`.
- `src/mc_photogrammetry/reconstruction/lighting.py` — measured block-light simulation and self-disabling response fitting; not a current Template 2 accuracy lever.
- Tests: `tests/test_raycast.py` (end-to-end `reconstruct_scene_surfaces`), `tests/test_face_sampling.py` (rectification/visibility), `tests/test_model_atlas.py` (rotation/full-cube baking; extend for partial geometry), `tests/test_textures.py` + `tests/test_texture_cache.py` (JAR provider/cache), `tests/test_camera.py`, `tests/test_litematic.py`, `tests/test_surface_pruning.py`, `tests/test_adjacency.py`, `tests/test_lighting.py`, `tests/test_reconstruction.py`.

---

## 8b. Phase A results (2026-09-04) — READ BEFORE PLANNING MORE CLASSIFIER WORK

### What was measured, not assumed
- A synthetic renderer (`render` the reference litematic with baked textures from the known
  pose, then sample it with the production sampler) is the control that makes every claim
  below checkable. On the synthetic render, per-face texture correlation is **NCC +0.91**
  (median +0.93, 97% of faces above 0.5). On the real screenshots the same code gives
  **NCC -0.03**. The sampler is correct; the frames disagree with the reference.
- Sensitivity of that correlation to camera pose: 0.1° yaw -> 0.79, 0.25° yaw -> 0.58,
  **0.25° pitch -> 0.12**, 0.5° pitch -> -0.11. The usable basin is about ±0.1°.
  `templates 2/cameras.json` records positions to 0.1 blocks, which at 2-4 blocks is
  1-3° of angular error, an order of magnitude too coarse.
- Ruled out with evidence, so do not re-investigate: integer block offsets (±1 on all
  axes), focal length (whole-frame renders confirm hFOV 110 is right; 84° and 70° are
  visibly worse), all 8 dihedral face orientations, sub-texel cyclic shifts at 0.25-texel
  resolution (real peaks at 0.31 mean vs synthetic 0.98), yaw/pitch grid search at 0.02°
  over ±2°, and per-face 8-DOF quad refinement.

**Conclusion: texel-level texture matching cannot pay off until camera poses are
sub-0.1°. That is Phase 4 (SfM), so the plan's phase ordering was wrong.**

### Two structural bugs, now fixed
1. `cv2.warpPerspective` has no `INTER_AREA`. The old `_rectify_projected_face` asked for
   it, so warping a 280x280 face into 16x16 point-sampled it into noise. This is why the
   previous session's texture work measured *identical* to its baseline.
2. Blockstate variant model rotation (`"x"`/`"y"`) was discarded, so all three `dark_oak_log`
   axes collapsed into one palette entry with an unrotated face map. The fixture has 58
   `axis=x` and 31 `axis=z` logs and no `axis=y`, so those 89 blocks (10%) were
   unreachable. `dark_oak_log` went 0% -> 10-19% once rotation was baked in.

### Numbers (Template 2, coordinate F1 unchanged at 86.14% throughout)
| configuration | exact state |
|---|---|
| previous session (`t2-step13-acceptance.json`) | 46.70% |
| face pipeline, colour only, no adjacency | 26.4% |
| face pipeline + ICM adjacency 0.8 | 33.5% |
| ...with perceptual dedupe tolerance 24 | 36.5% |
| **labelling every block `dark_oak_planks`** | **up to 66.2%** |

That last row is the important one. The previous 46.70% is far below the trivial
majority-class score, and its per-state breakdown (dark_oak_planks 81.7%, every other
class 0.0%) shows it was close to a constant predictor. Neither version is near the gate.

### Why colour alone plateaus around a third
Top-K recall of the appearance cost on face-sampled blocks: top-1 16.7%, top-3 31.1%,
**top-5 59.8%**. The true state is usually in contention but rarely first, because with one
visible face and mean colour only, `birch_planks` / `stripped_birch_log` / `stripped_oak_log`
(and `dark_oak_planks` / `spruce_log` bark / `stripped_dark_oak_log`) are within a few colour
levels of each other. Separating them needs texture, which needs poses. Do not try to close
this with name-token priors; that is the blind heuristic the user already rejected (§5).

### Ideas measured and found not to help (do not retry blind)
- `calibrate_appearance_from_palette` in the face path: actively harmful. It fits a blue
  gain clamped at 0.5 and pushes expected colours off the observations. With per-face
  shading applied, an unmodified screenshot already lands on the texture colour
  (observed `[38,25,12]` vs expected `[40,26,12]`), so identity is the right default.
- Min-over-all-faces matching (the old behaviour): worse than slot-specific matching.
- Per-frame view weighting from median residual: neutral; frames do not separate on it.
- Outlier ceiling sweep 3-60: flat, best around 24.

### Files added or reworked
- `src/mc_photogrammetry/blocks/model_atlas.py` (new) — bakes world-oriented per-face
  textures and corner UV order, honouring variant `x`/`y` rotation and element face
  `rotation`. `FACE_SHADE` holds vanilla's per-direction brightness.
- `src/mc_photogrammetry/reconstruction/face_sampling.py` (new) — depth-buffered visible
  face extraction plus the two-stage rectification that fixes bug 1.
- `raycast.reconstruct_scene_surfaces` — three passes: unchanged centre-pixel buffer for
  coordinates, then face sampling, then a centre-patch fallback for blocks with no usable
  face. Costs are made relative to each observation's own best match and capped, so a
  contaminated view abstains instead of voting confidently.
- `textures.py` — cache schema 11, `face_corners`, rotation-aware `face_map`, and
  `canonical_palette(tolerance=...)` collapsing states this classifier cannot resolve.
- `vision/gui_mask.py` — the first-person held item was never masked and was contaminating
  floor samples.
- Tests: `tests/test_model_atlas.py`, `tests/test_face_sampling.py`. 72 pass.

### Open decision
The face pipeline is more principled and is the substrate Phase 4 needs, but it currently
scores **below** the previous number on the gate metric. Either accept that and go to
Phase 4, or restore the previous scoring first. The user has been asked.

---

## 8c. The projection was mirrored (2026-09-04, found by eye)

`project_points` computed `view_x = dx*cos(yaw) + dz*sin(yaw)`, which is the **negative** of
Minecraft's screen-right vector. Forward for yaw θ is `(-sin θ, 0, cos θ)`; screen right is
`forward × up = (-cos θ, 0, -sin θ)`, so facing south (+Z) puts **west** (−X) on the right of
frame. The dropped sign mirrored every render left to right.

A mirrored room still looks like a plausible room, which is why nothing caught it:
- Coordinate F1 is derived from the projected scene box, not from image content, so it stayed
  at 86.14% throughout and is **blind to this class of bug**. Do not treat coordinate F1 as
  evidence that the geometry is right.
- Every whole-frame render comparison I made looked "structurally similar" because the room is
  roughly symmetric about the mirror axis.
- The tell was a **landmark asymmetry**: the white carpet sits on the left of frame 1 in the
  photo and on the right in the render. The user spotted it in the review page.

Fixing it (`camera.py`, both `project_points` and `unproject_pixel`) moved exact block-state
accuracy from **33.48% to 39.94%**, and `dark_oak_planks` from 56.3% to 68.1%. Two regression
tests now pin the handedness: `test_screen_right_is_west_when_facing_south` and
`test_yaw_minus_90_faces_east_and_puts_south_on_the_right`.

**Everything in §8b measured through a scratch copy of the projection was measured mirrored.**
The sweeps were re-run afterwards through the production `project_points`: field of view 86°-122°,
scene origin offsets of ±1 on all three axes, eye height 0-1.82, and per-frame yaw/pitch at 0.1°
steps across ±1.2°. All stay flat within 0.02 of zero, so the recorded pose is still the best
available and the residual error is sub-block. The §8b conclusion about pose precision stands;
its individual numbers do not.

### Lesson for the next agent
Two independent checks would have caught this in minutes, and are worth keeping:
1. **Render the reference and compare landmarks by eye.** `scratchpad` held a renderer that draws
   every block's real model geometry (all elements, not just full cubes, honouring variant
   rotation). Asymmetric, brightly coloured blocks - carpet, wool, a bed - are the diagnostic.
2. **Never use coordinate F1 as a geometry check** while occupancy comes from projecting the known
   scene box. It cannot fall below ~86% no matter how wrong the camera is.

Still open: `dark_oak_fence` has no drawable geometry in that renderer (it is multipart, not a
`variants` blockstate), so the ceiling beam is missing from the twin.

---

## 8d. Sample the middle of a face, not all of it (2026-09-04)

An audit of which evidence path decides each block turned up something backwards: for
`dark_oak_planks`, the **centre-pixel fallback beat full-face sampling, 73.8% against 48.2%**.
A 5x5 patch at the block's projected centre tolerates a pose error of nearly half a block; a
whole-face rectification pulls in the neighbouring block as soon as the pose is off by a few
percent. So with poses this imprecise, the extra coverage of a full face is worth less than the
registration it costs.

`sample_visible_faces` already shrank each quad about its centre; the shrink factor was just far
too generous. Measured sweep, everything else held constant:

| face shrink | exact state |
|---|---|
| 0.9 | 38.65% |
| 0.8 (old default) | 39.51% |
| 0.65 | 41.52% |
| 0.5 | 41.95% |
| **0.35 (new default)** | **42.10%** |
| 0.25 | 41.09% |

With adjacency at 1.0 rather than 0.8 that becomes **42.67%**, which is the current number.
`FACE_SHRINK` in `raycast.py` and `DEFAULT_SHRINK` in `face_sampling.py` carry the value, and
the benchmark's `--adjacency-strength` default is now 1.0. The right way to set the shrink is
from how well the poses are known, so it should go back up once Phase 4 lands.

Also measured and found not to help: weighting frames by cross-view consensus (39.5% -> 39.4%),
and an edge-alignment pose objective (gradient energy landing on the projected block seams). The
edge objective is worth naming because it looked promising and is not: a deliberately wrong pose,
yaw off by 4-6 degrees, scores the same as the nominal one, because this scene has plank lines
every quarter block and so has edge energy everywhere.

### Running total on Template 2
| | exact state | coordinate F1 |
|---|---|---|
| previous session | 46.70% | 86.14% |
| Phase A rebuilt classifier | 33.48% | 86.14% |
| ...plus the mirror fix (§8c) | 39.94% | 86.14% |
| ...plus centre-of-face sampling | **42.67%** | 86.14% |

Still below the previous 46.70%, and far below the 66.2% a constant `dark_oak_planks` predictor
would score. `birch_planks` remains near 0%: it is read as `dark_oak_planks` (61) or as one of
the stripped logs that share its colour (72). The first of those is a large colour error that
sub-block jitter does not explain and is the next thing to chase.

### The diagnostic that found both fixes
`scratchpad/error_map.py` renders four panels per frame over the photo: the projected grid, the
reference block colours, what the classifier chose, and a green/red per-face score. Spatial
patterns in the red are what pointed at both the mirror and the sampling width. Keep it.

---

## 8e. Error budget: what actually stands between us and 100% (2026-09-04)

Measured with an oracle rather than reasoned about. The oracle replays the exact same visible
faces but hands the classifier the *true* colour each face should have; whatever it still gets
wrong is the palette's own ambiguity, and the rest is sampling. `scratchpad/error_budget.py`.

| cause | share of the 696 comparable blocks |
|---|---|
| correct today | 42.7% |
| **sampling / registration error** | **55.2%** |
| colour-degenerate candidates | 1.9% |
| palette cannot express the block | 0.3% |

**This corrects §8b and §8d.** I had said colour-degenerate lookalikes were the limiting factor.
They are not: given the correct per-face colour, 681 of 694 full-cube blocks (98.1%) are
*uniquely* identified, ties and all. Only 13 blocks are genuinely ambiguous, and they are all
log-axis or `dark_oak_wood` ties. `birch_planks` versus `stripped_birch_log` is NOT ambiguity -
with the right colour, birch wins outright. Those confusions are sampling failures wearing an
ambiguity costume.

### Where the sampling error sits
Per-block colour error against the block's own true face colour, in appearance units:

| block | n | median error | accuracy |
|---|---|---|---|
| `dark_oak_planks` | 272 | 7.1 | 68% |
| `dark_oak_log` | 66 | 6.9 | ~14% |
| `birch_planks` | 108 | 47.9 | ~0% |

An error around 7 is mostly survivable against a 338-entry palette; around 48 is fatal. Birch is
the floor, and its samples split hard **by frame**:

| frame | birch samples | median colour error |
|---|---|---|
| `13.22.14` | 13 | 7.0 |
| `13.24.24` | 74 | 47.9 |
| `13.24.33` | 29 | 133.8 |

Frame 1 samples birch correctly. Frames 2 and 3 do not. That is a per-frame geometry problem,
not a palette or descriptor problem, and it is where the next work belongs.

### A second, smaller and very fixable cause
Grouping face samples by what the reference says sits in front of the sampled face:

| what is in front | n | median error |
|---|---|---|
| nothing | 492 | 6.9 |
| `oak_stairs` | 23 | 60.4 |
| a full block that should have been culled | 14 | 67.5 |
| `white_carpet` | 8 | 111.0 |
| `gray_carpet` | 6 | 43.7 |
| `oak_slab` | 5 | 39.1 |

Only 8% of samples, but ruinous when it happens. The occupancy the sampler reasons about comes
from the projected scene box, so it knows nothing about carpets, stairs and slabs sitting on the
blocks it is sampling. The 14 "should have been culled" cases are a straight bug in the
visibility test worth chasing on its own.

### Priority order implied
1. Per-frame camera accuracy (frames 2 and 3). This is Phase 4 and it is worth ~55 points.
2. Occlusion from non-cube geometry, including the 14 mis-culled full blocks.
3. Palette ambiguity - worth 1.9 points. Not worth touching yet.

---

## 8f. Item 1 was the wrong diagnosis: it is block lighting, not camera pose (2026-09-04)

§8e attributed 55 points to "sampling / registration error" and implied camera pose. That was
tested and **pose is not the cause**.

### Oracle pose refinement
`scratchpad/pose_oracle.py` refines each frame's yaw, pitch and position by pattern search,
minimising colour error **against the ground-truth block colours** - an oracle no production
pipeline could have. Median colour error, nominal vs refined:

| frame | `dark_oak_planks` | `birch_planks` |
|---|---|---|
| `13.22.14` | 9.6 -> 8.3 | 14.0 -> 15.6 |
| `13.24.24` | 9.3 -> 7.3 | 57.4 -> 53.6 |
| `13.24.33` | 7.7 -> 6.3 | 133.8 -> 98.2 |

Corrections found were small (yaw within 1.2 deg, position within 0.25 blocks). Dark oak improves
slightly; **birch stays broken even with an oracle**. No pose fixes it, so do not spend Phase 4
expecting it to.

### What is actually happening
`birch_planks` sits in two places: 153 on the floor (local y=0) and **64 in the ceiling**
(y=11), whose undersides carry 0.5 face shading and very little torch light. Tracing individual
decisions:

- frame `13.22.14` reads a birch floor face as **(192,176,120)**; birch's texture is
  **(192,175,121)**. Exact.
- frame `13.24.24` reads birch floor faces as **(189,153,96)**, **(181,146,94)**, **(177,142,91)**.
  Red matches, green and blue are down - a warm cast. `stripped_oak_wood` and `stripped_oak_log`
  sit exactly there, and win. Birch's median rank across its own blocks is **30**.

So the observed colour of a face depends on Minecraft's per-block lighting, and torch light is
warm. A dark block barely moves; a pale block moves far enough to land on a different material.
That is the whole birch failure, and it is why `birch_planks` scores 0/209 in every configuration
tried.

### Why a per-frame white balance does not fix it
Implemented and measured: `_self_calibrate` matches each frame's faces against the palette, keeps
the most decisive, and takes the per-channel ratio as that frame's cast. **Neutral, 44.97% either
way.** The reason is diagnostic: the confident matches are dark oak walls, which already fit at
gain 1.01, so the estimate comes back neutral, while birch in the same frame needs about
(0.98, 0.87, 0.79). One gain cannot serve both, because the cast is **spatial** - a property of
where the torches are - not a property of the camera. The code is kept because footage with a
genuine global cast (shaders, a changed brightness setting) will need it, but it earns nothing here.

### What would actually fix it
Solve for a per-block light field jointly with the labels, the way intrinsic-image decomposition
separates reflectance from shading: neighbouring blocks share light, so a smooth light field plus
piecewise-constant materials is a well-posed problem. Comparing a face against its neighbours
rather than against an absolute constant is the cheaper approximation of the same idea. Both are
real work; neither is parameter tuning.

### Gains actually banked this round
| change | exact state |
|---|---|
| after §8d | 42.67% |
| + patch-uniformity gate (item 2) | 44.97% |
| + chroma weight 0.55 -> 0.75 | **45.69%** |

Item 2 is done. `MAX_FACE_VARIATION = 12.0` drops face samples whose quadrants disagree, which is
how an occluding carpet or stair is caught without needing it in the geometry; the threshold sits
where the retained samples' median colour error bottoms out (8.0 ungated, 6.4 gated).

### Measured and NOT worth keeping
- One-sided lighting gain (expected colour scaled into `[minimum_light, 1]`): slightly worse at
  every floor from 0.7 down to 0.05. `MINIMUM_LIGHT` stays 1.0, the code is inert.
- Weighting frames by cross-view consensus: 39.5% -> 39.4%.
- Weighting samples by internal consistency (`UNIFORMITY_SCALE`): +0.14 points, left inert at 1e9.
- Edge-alignment pose objective: a pose wrong by 4-6 deg of yaw scores the same as the right one.

---

## 8g. The field of view was read as horizontal; it is vertical (2026-09-04)

Found because the user said the preview renders did not look like the same perspective as the
photos. They were right, and it was the largest remaining bug.

`project_points` computed `focal = (width / 2) / tan(fov / 2)`. Minecraft's FOV setting is the
**vertical** field of view - it is passed to `Matrix4f.perspective` as the fovy - so the focal
length follows the image **height**. On a 2560x1369 frame that is a factor of 1.87.

A focal error is exact at the centre of frame and grows outward, so a wireframe overlay looks
convincing in the middle and drifts at the edges. §8b noted exactly that drift, then dismissed it
because a focal sweep found nothing. **The sweep was scored by per-face texture correlation, which
was flat at zero for every candidate because the projection was also mirrored.** A metric that
returns the same number for a deliberately wrong answer cannot arbitrate anything, and treating
its flatness as evidence was the mistake that hid this for the whole session.

### The metric that did work
Overlap (IoU) of the bright regions - white carpet and birch floor against dark walls - between a
full model render and the photo. `scratchpad/fov_check.py`. Sweeping the stored FOV:

| stored fov | focal | implied vertical FOV | mean IoU |
|---|---|---|---|
| 70 | 1828 | 41.1 | 0.232 |
| 110 (as used) | 896 | 74.7 | 0.346 |
| 130 | 597 | 97.8 | 0.490 |
| **140** | **466** | **111.5** | **0.558** |
| 150 | 343 | 126.8 | 0.468 |

The peak sits at a vertical FOV of about 110 degrees, which is exactly the setting recorded in
`cameras.json`. `_focal_length` in `camera.py` now derives focal from the height.

### What it unlocked
| step | coordinate F1 | exact state |
|---|---|---|
| before | 86.14% | 45.69% |
| focal fixed | **94.70%** | 55.53% |
| + full-face sampling (`FACE_SHRINK` 0.35 -> 1.0) | 94.70% | 58.48% |
| + full-texture matching (`TEXTURE_WEIGHT` 25) | 94.70% | **65.30%** |

Coordinate F1 had been stuck at 86.14% through every previous change; it moved the moment the
projection was right, which is the clearest sign of how much this was costing.

**`FACE_SHRINK` was a workaround for this bug.** It was set to 0.35 because a whole-face sample
dragged in neighbouring blocks; with the projection correct, the whole face is better, and
`dark_oak_log` goes from 5% to 67% because its axis is finally readable.

**Per-face texture matching now works**, which was the user's original proposal and the thing
§8b said was unreachable. Correlation against the candidate's own face texture, computed after
removing brightness: real frames now score mean NCC **+0.436**, median +0.531, 52% of faces above
0.5, against -0.016 before. It is worth +6.8 points and takes `birch_planks` from 0% to 21%,
because correlation is exactly what block lighting cannot disturb.

### Standing lesson
Both geometry bugs were found by a human looking at a render, and both had survived numeric
sweeps that were themselves broken. Before trusting any alignment metric, check that it scores a
deliberately wrong pose worse than the nominal one. The edge-alignment objective in §8f fails that
check and so did the focal sweep in §8b.

---

## 8h. Re-tuning after the focal fix, and where it leaves us (2026-09-04)

Every parameter in §8d and §8f was fitted against the broken projection, so all of them were
swept again. Results on Template 2, coordinate F1 constant at 94.70% throughout:

| parameter | swept | chosen | note |
|---|---|---|---|
| `FACE_SHRINK` | 0.35 - 1.0 | **1.0** | 0.35 was pure compensation for the bad projection |
| `MAX_FACE_VARIATION` | 8 - none | **12.0** | unchanged; none costs 3.4 points |
| `adjacency_strength` | 0 - 1.0 | **1.0** | now worth 11 points, was worth 7 |
| `CHROMA_WEIGHT` | 0.45 - 1.0 | **0.85** | see below |
| `TEXTURE_WEIGHT` | 0 - 200 | **40.0** | 0 costs 6.8 points |

`CHROMA_WEIGHT` measures flat from 0.8 to 1.0 (66.7 to 67.7, a spread of about eight blocks) and
0.95 is nominally the best. It is deliberately **not** set to the argmax: Template 2 is built
entirely from wood, so hue separates everything and brightness earns nothing. A scene of stone,
concrete or wool has no hue to work with and needs the brightness term that a weight of 1.0
discards. 0.85 sits inside the plateau and keeps that term.

### Current state
| | value |
|---|---|
| coordinate F1 | **94.70%** (recall 91.31%, precision 98.36%) |
| exact block state | **67.22%** |

Per state: `dark_oak_planks` 88.5%, `dark_oak_log` 70.8%, `birch_planks` 24.9%, and every
non-cube class still 0% because the palette has no candidate for them.

### Session arc, for context
| | exact state | coordinate F1 |
|---|---|---|
| previous session's number | 46.70% | 86.14% |
| Phase A rebuilt classifier | 33.48% | 86.14% |
| + mirrored projection fixed (§8c) | 39.94% | 86.14% |
| + centre-of-face sampling (§8d) | 42.67% | 86.14% |
| + patch-uniformity gate (§8f) | 44.97% | 86.14% |
| + chroma weight (§8f) | 45.69% | 86.14% |
| + field of view fixed (§8g) | 55.53% | **94.70%** |
| + full-face sampling (§8g) | 58.48% | 94.70% |
| + full-texture matching (§8g) | 65.30% | 94.70% |
| + re-tuning (this section) | **67.22%** | 94.70% |

### What is left between here and the 80% gate
1. **`birch_planks` at 24.9%**, 209 blocks. Still the largest single class of error. §8f showed
   why: warm torch light shifts a pale block onto a different material. Texture matching took it
   from 0% to 25% because correlation ignores brightness; the rest needs the light field.
2. **Non-cube blocks**, about 60 in the reference and all at 0%. They are not palette candidates
   at all. `scratchpad/full_render.py` already bakes every block model including stairs, slabs and
   carpets, so the geometry to do this exists; what is missing is candidate generation and a
   shape-aware sampler.
3. **The light-field decomposition** from §8f, still not attempted.

Item 2 is probably the better next move: it is bounded work against known geometry, worth up to
7 points, where item 3 is open-ended research.

---

## 8i. The preview was rendering the reference, not the reconstruction (2026-09-04)

The user loaded `exports/t2-final.litematic` in game, saw a floor of scattered wool, mangrove,
soul soil and redstone lamps, and asked how the render could look near perfect while the file
fell apart. **The render was of the reference litematic.** `scratchpad/visuals.py` and
`full_render.py` both take `templates 2/Template 2.litematic` as their subject; the panel was
built to check the projection against the photo and was never relabelled when it went into the
review page as "Reconstruction". Nothing in the pipeline was wrong - the deliverable was.

### The pipeline is clean, and this was verified rather than assumed
Round trip from classifier output to file and back:

```
classifier decided 791 blocks
litematic on disk holds 791 blocks, palette 52
EXPORT ROUND TRIP:  lost 0   invented 0   changed 0
```

So `BlockVolume` -> `LitematicExporter` -> `load_litematic` is exact. What the file contains is
precisely what the classifier chose, and at 67.22% exact state that is roughly one block in three
wrong. The scattered materials the user saw are the real top-of-list decisions: 30 `spruce_log`,
21 `stripped_oak_log`, 18 `oak_log`, 12 `mangrove_wood`, 9 `white_wool`, 8 `stripped_mangrove_log`,
7 `redstone_lamp`, 7 `white_concrete_powder`, 7 `soul_soil`, 6 `observer`.

The review page now carries four layers - photo, projected grid, **reference**, **reconstruction** -
so the two renders can be flipped against each other. `scratchpad/compare_three.py` produces them.

### Standing lesson
Never show a render of the ground truth in a panel that stands for the system's output. It hides
exactly the errors the page exists to expose, and it is a more dangerous failure than a wrong
number, because a picture is believed. Any preview of reconstruction quality must render the
exported artefact.

---

## 8j. Block-light simulation: built, correct, and worth nothing on Template 2 (2026-09-04)

`src/mc_photogrammetry/reconstruction/lighting.py` simulates Minecraft block light over the
reconstructed volume and feeds it back into classification as an EM loop: classify unlit, let the
decided blocks say where the light is, flood fill, re-classify against dimmed expected colours.
`tests/test_lighting.py`, 12 tests.

- Vanilla emission table. Copper bulbs dim as they oxidise (15/12/8/4) and only count while `lit`;
  candles scale with count; `soul_torch` must beat `torch` on longest-token match; redstone lamps
  and respawn anchors read their own state.
- Breadth-first propagation, one level lost per block, stopped by opaque cells. Non-solid emitters
  do not shadow themselves.
- Vanilla curve `f / (4 - 3f)`: level 14 is already 0.78, level 10 is 0.33.
- The response is **fitted**, not assumed, because the client's brightness slider and gamma sit
  between game light level and screenshot pixels.

Verified against the reference geometry of all three scenes: Template 2's four torches light 99.7%
of its air to a mean level 7.0; Template 4's 18 emitters reach mean 8.8; Template 3's 453 emitters
leave **88% of its air completely dark**, which matches its description.

### It does not help, and the measurements say why
| run | exact state |
|---|---|
| lighting off | 67.22% |
| lighting on | 67.22% |
| **oracle: the 4 real torches fed in by hand** | **67.22%** |

Identical to the digit, including per-state. Feeding in ground-truth light sources changes nothing,
so this is not a detection problem.

**These screenshots were taken with the brightness setting up.** Across a room whose simulated
levels span 0 to 12, observed brightness only spans 0.78 to 1.00, and the correlation between
simulated level and observed brightness is **0.124**. Vanilla's curve says level 0-3 should render
at 0.018; it actually renders at 0.782. There is almost no light variation in the picture for a
light model to explain.

Nor does light level move hue - the blue-to-red ratio stays 1.005 to 1.054 across every level band,
so the light in these frames is not warm-tinted.

### What birch's error actually is
Per-channel ratio of observed to unlit expected, by block:

| block | R | G | B |
|---|---|---|---|
| `dark_oak_planks` | 0.970 | 1.008 | 1.000 |
| `dark_oak_log` | 0.993 | 1.000 | 0.985 |
| **`birch_planks`** | **0.911** | **0.820** | **0.816** |

Birch is off in a way that is **material-specific, not level-specific**, and `192,175,121` scaled by
that lands on `175,143,99` - which is oak. The birch floor has oak stairs, slabs and planks sitting
on it at y=1, and a patch that lands wholly on one of those is perfectly uniform, so the
`MAX_FACE_VARIATION` gate cannot catch it. **Birch is an occlusion problem, not a lighting one.**

Together with §8f (oracle pose refinement does not fix birch) and §8h (per-frame white balance does
not fix birch), that is three separate hypotheses eliminated by measurement. The remaining one is
modelling the non-cube blocks so their cells are known occluders, which is item 2 of §8e.

### Kept anyway, and made self-disabling
`LightResponse.confidence` carries the correlation between simulated level and observed brightness;
the EM loop bails when it falls below `MINIMUM_LIGHT_EFFECT` (0.3). On Template 2 the loop exits
after one round having fitted a flat response, costing about 3 seconds and changing nothing. On a
genuinely dark scene it will engage. Template 3 is the test: 88% of its air is unlit and its 392
copper bulbs are full cubes the classifier can find on its own.

**Runtime note, correcting something said in passing during the session:** lighting is not what
makes a run slow. Measured, 122.0s with lighting off against 124.9s with it on. The two minutes are
the texture-matching term added in §8g.

---

## 8k. Runtime: 2m05s to 40s, and where the rest of it is (2026-09-04)

A reconstruction of Template 2 took just over two minutes, which made every A/B sweep a
ten-to-thirty minute wait. Profiling found two causes, neither of them the algorithm.

**`BlockState.__hash__` recomputed its normalisation 30.6 million times per run**, costing about
63 seconds. Block states are the keys of every per-candidate score in the classifier, and
`_normalized_properties` was re-deriving the default-stripped form on every lookup. It is now
computed once in `__post_init__` and stored in a `_identity` slot; equality and hashing both read
it. The class is frozen, so the cache cannot go stale.

**`_self_calibrate` re-scored every observation, three rounds over**, which was 158 seconds of a
228 second profile - three quarters of the run - to fit three numbers per frame. It now fits from
an evenly spaced sample of at most 250 observations per frame.

Sampling moved the answer by two blocks (67.22% to 66.97%), because a balance fitted from a sample
never lands exactly on identity and this fixture has no real cast to find. So the calibration now
snaps to identity when every channel is within `MINIMUM_COLOUR_CAST` (4%) of neutral, which is the
same self-disabling idea as the light response in §8j: only correct for something the data
actually shows.

| | runtime | exact state |
|---|---|---|
| before | 125s | 67.22% |
| cached identity + sampled balance | 40s | 66.97% |
| + snap near-neutral balance to identity | **40s** | **67.35%** |

Slightly better than where it started, at a third of the time.

### What is left, if it ever needs to be faster
`_observation_costs` is 50s of a 61s profiled run. It loops over all 338 palette entries in Python
per observation, calling `_appearance_distance` (2.1M calls) and `np.linalg.norm` (6.7M calls) on
three-element vectors. Vectorising it - one `(candidates, 3)` array of expected colours per face
slot, distances computed in a single array op - should take another large factor off. Nothing else
in the profile is close.

---

## 8l. Overnight run: palette repair, vectorisation, and pruning (2026-09-05)

### Three bugs in the palette, all costing real accuracy
1. **The family filter dropped 389 full-cube blocks.** `canonical_palette` required
   `family in {"cube", "log"}`, which excluded every full cube carrying more than one texture
   and no recognised name token - `stone`, `blackstone`, `basalt` among them. Template 4 is 43%
   `stone`, so its most common block was not a candidate at all. Membership is now decided by
   geometry: any full cube qualifies.
2. **`stone` lost its own appearance group to `infested_stone`** on an alphabetical tie-break.
   `_canonical_rank` now prefers the fewest properties, then the shortest name, because derived
   blocks name themselves after the base one and add to it.
3. **Biome-tinted blocks polluted the palette.** `spruce_leaves` ships a grey mask that the game
   recolours per biome, so its palette colour is exactly `stone`'s and it matched any grey block.
   `_is_biome_tinted` now excludes leaves, grass blocks, vines and the rest.

Template 4 palette coverage went from 40.2% to **99.7%**; Template 2 held at 91.8%. The dedupe
tolerance also had to drop from 12 to 4 - at 8 the palette loses `polished_blackstone_bricks`,
which is 21% of Template 4.

### The palette got bigger, so scoring had to get faster
338 candidates became 636, and a run went to 150s. `PaletteIndex` now stacks every candidate's
colours, descriptors, priors and texture patterns into arrays, and the whole pipeline carries
`(n_candidates,)` cost vectors instead of per-candidate dicts. **150s to 8.7s, about 17x.** That
is what made the rest of the night's sweeps possible at all.

### Pruning the palette to what the scene supports
A build uses a handful of materials but the palette offers every full cube in the game; the long
tail each win a few stray faces, and every one is a block wrong. After one full pass, candidates
winning less than `PALETTE_SUPPORT` (2%) of faces are dropped and the pass is repeated. Nothing
decides in advance what a scene may contain - support is measured. Worth **+17 points** on
Template 2 on its own. It converges in one round; a fixed share beat cumulative-coverage pruning.

### A simplicity prior for fixtures
`dark_oak_planks -> cartography_table` was the single largest confusion, 74 blocks. A cartography
table wears the same wood on its sides, so a wall seen only side-on genuinely cannot tell them
apart. Plain building material wears one distinct face (planks, stone) or two (a log's bark and
end, blackstone's top and side); three or more means a fixture. `SIMPLICITY_WEIGHT` charges only
for faces beyond the second, which settles that tie without touching the log-versus-planks call
that real texture evidence decides.

An earlier version of this prior used name length and had to be abandoned: it pushed
`dark_oak_log` (19 characters) toward `dark_oak_planks` (15) and collapsed logs from 67% to 11%.
Trading one class for another scores better and is not an improvement.

### Where Template 2 stands
| setting | value |
|---|---|
| palette tolerance | 4 (636 candidates, pruned to ~6) |
| `CHROMA_WEIGHT` | 0.75 |
| `TEXTURE_WEIGHT` | 40 |
| `SIMPLICITY_WEIGHT` | 3.0 |
| `PALETTE_SUPPORT` | 0.02 |
| adjacency | 0.55 |

**Coordinate F1 94.70%, exact state 79.43%** - `dark_oak_planks` 89%, `dark_oak_log` 69%,
`birch_planks` 68%. Adjacency at 1.0 scores marginally higher (79.4 vs 78.7 at 0.4) but erases the
log inlays, which are single blocks surrounded by planks; 0.55 keeps them and is the honest
setting.

---

## 8m. Evidence quality, partial blocks, and why Template 4 is still zero (2026-09-05)

> **Superseded in part by 8n.** The Template 4 diagnosis in this section, and the
> "264 unfaced blocks -> 90.0%" projection, are both wrong. Read 8n first.

Targets for this round were 90% exact state on Template 2 and 50% on Template 4.
**Neither was reached.** Template 2 went 82.65% -> **85.35%**; Template 4 is still
**0.00%**. What follows is what moved, what did not, and the measurements that say
where the remaining error actually lives.

### The thing that was actually wrong on Template 2

Birch was the worst class and no amount of colour or brightness modelling touched
it. Rendering the failing patches side by side with the passing ones settled it in
one look: the misclassified "birch" floor tiles are showing dark oak wood and
**pure white** pixels. Those are not sampling errors. The pixels genuinely are not
birch.

The reference puts `oak_stairs`, `oak_slab` or `white_carpet` directly on top of
six of the ten worst birch blocks. We never detect 74 of the reference's blocks,
and 63 of them are partial geometry:

| block | missed |
|---|---|
| `oak_stairs` | 20 |
| `white_carpet` | 16 |
| `dark_oak_fence` | 13 |
| `oak_slab` | 5 |
| `gray_carpet` | 5 |
| `torch` | 4 |
| (full cubes) | 11 |

These sit on a block, cover its upward face completely, and are absent from our
occupancy - so they can neither be measured nor culled as occluders. The existing
guard, `max_face_variation`, cannot catch them because a stair covers the block
below *uniformly*: the patch is a clean, confident read of the wrong material.
This is the failure mode the user called out early ("the partial blocks like
stairs, carpets, and fences are having trouble") and it is now measured rather
than suspected.

The exploitable asymmetry is directional. Things rest on top of blocks, so it is
upward faces that get covered:

| slot | sampled | covered in reference |
|---|---|---|
| up | 74 | **21.6%** |
| south | 49 | 8.2% |
| west | 279 | 2.9% |
| east | 201 | 1.0% |
| north | 120 | 0.0% |
| down | 124 | 0.0% |

### What was added

Two evidence-quality terms in the ICM smoothing in `raycast.py`. Both express the
same idea - a block whose evidence is of a kind that lies about its block should
defer to its neighbours - and both are keyed to physically measured causes rather
than tuned heuristics.

- **`UPWARD_RISK = 11.0`**. Coupling scales with the share of a block's evidence
  weight that came from upward faces, per the table above.
- **`FALLBACK_RISK = 3.0`**. The same, for the share coming from the pass-three
  centre-patch fallback. Those blocks were never resolved as a face, so the pixel
  standing in for them may belong to whatever is in front of them.

`upward_risk` is the first change all round that **fixed birch (76% -> 82%)
without trading away another class** - `dark_oak_log` held at 69%. Every previous
confidence mechanism collapsed the minority classes into `dark_oak_planks`.

Also added, currently off by default:

- **`FIT_EXPOSURE = False`**, with `EXPOSURE_FLOOR = 0.2` / `EXPOSURE_CEILING = 3.0`.
  `_self_calibrate` divided its fitted scale by its own mean, so it could express a
  colour cast but never overall brightness. The reverse-lighting idea was literally
  inexpressible before this. On Template 2 it trades planks for logs
  (`dark_oak_log` 67% -> 74%, net 84.58% vs 85.35%); on Template 4 it does nothing,
  for the reason in the next section.
- **`EVIDENCE_POWER = 0.0`**, **`AREA_POWER = 0.5`**, **`FIT_TOLERANCE = 0.0`**,
  **`MINIMUM_VISIBLE_FRACTION = 0.6`** - knobs added to test a hypothesis, each
  measured to be worth nothing, each left at its no-op default with the reasoning
  in its docstring.

Also fixed: the texture term charged a candidate with no pattern for a slot the
*full* penalty while patterned candidates paid `w*(1-ncc)` ~= `0.9w`, so "flat"
candidates were penalised rather than neutral as the comment claimed. The term is
now centred on the mean correlation, making it zero-sum. It did not change the
score at the optimum, but the code now does what it says.

### Measured and NOT worth keeping (do not retry)

| lever | result |
|---|---|
| per-slot shading correction | vanilla's constants are already right: observed/expected 0.92-1.03 after AO. Birch is equally well exposed at both heights (0.947). Brightness is not Template 2's problem. |
| outlier-fit down-weighting (`FIT_TOLERANCE`) | 82.65% -> 76.5%; `dark_oak_log` 69% -> 10% |
| area-power weighting (`AREA_POWER` > 0.5) | 82.65% -> 79.8% at 1.3; logs collapse |
| `minimum_face_area` 24 -> 1200 | 84.06% but logs 69% -> 51%; trades one class for another |
| `face_shrink` < 1.0 | strictly worse (0.92 -> 79.6%). **Shrinking breaks the texture correlation**, which is computed against the full 16px jar texture. Full-quad is correct, not merely convenient. |
| relaxing `max_face_variation` | 12 is right; 25 -> 82.65%, unbounded -> 80.08%. The rejected faces genuinely straddle occluders. |
| `minimum_visible_fraction` 0.6 -> 0.05 | no change; not the binding constraint |
| `TEXTURE_WEIGHT` re-swept after the centring fix | 40 still optimal; 55 -> 81.5%, 90 -> 77.3% |

### Where Template 2's remaining 4.6% is, precisely

| group | n | accuracy |
|---|---|---|
| blocks with a resolved face sample | 514 | **91.2%** |
| blocks with no face sample (centre-patch fallback) | 264 | **77.7%** |
| blocks whose every observed face is covered by unheld geometry | 27 | 66.7% |

**All 264 unfaced blocks are genuinely exposed in the reference - none are truly
buried**, and all 264 are already in the surface set. If they reached face-sample
quality, Template 2 lands on **exactly 90.0%**. That is the lever, and it is an
occupancy/visibility problem, not a tuning problem. Of the 264, the sampler does
return a face for 92 before the uniformity filter, but relaxing that filter loses
more than it gains, so the fix has to be better geometry rather than a looser
threshold.

Supporting diagnostics, if useful:

- Rank of the true block in the fused cost vector: `dark_oak_planks` median 0,
  `dark_oak_log` median 0 (87% already at rank 0), **`birch_planks` median 5** with
  only 40% inside the top 3. Birch was a representation failure, not a fusion one.
- Texture correlation does separate bark from planks - for true log faces a log
  outranks planks **83.7%** of the time - but the margin is only ~0.09 NCC, worth
  ~3.6 cost units at weight 40, against a coupling of ~5.5 from four plank
  neighbours. That is why logs sit on a cliff: `adjacency_strength` 0.55 -> 0.65
  drops `dark_oak_log` from 67% to 12%.

### Template 4: diagnosed, not fixed

Still 0.00% exact state, coordinate F1 49.10%. The reference has 6225 blocks; we
place 2044, overlapping 2030 - consistent with the user's note that Template 4
contains blocks that are never visible.

The zero is real and systematic, not a near miss:

| truth | predicted | n |
|---|---|---|
| `stone` | `gray_terracotta` | 626 |
| `stone` | `black_stained_glass` | 266 |
| `stone` | `tinted_glass` | 110 |
| `stone` | `coal_block` | 99 |
| `calcite` | `cherry_wood` | 99 |
| `smooth_basalt` | `cherry_wood` | 56 |

Reference census: `stone` (dominant), `calcite` 578, `amethyst_block` 575,
`blackstone` 466, `smooth_basalt` 414, `budding_amethyst` 46. Predicted census:
`gray_terracotta` 749, `black_stained_glass` 290, `cherry_wood` 254,
`tinted_glass` 199. **Every block reads far too dark**, and each one lands on
whatever the palette offers that is genuinely darker.

Two things ruled out:

1. **It is not misalignment.** All 343 integer offsets within +/-3 on each axis
   were tested; `(0, 0, 0)` is the best by F1 (49.10%, next best 41.63%). The
   region is correctly aligned, so this is appearance, not geometry.
2. **It is not fixable by fitting exposure.** `FIT_EXPOSURE` was built for exactly
   this and changes nothing here. The reason is structural and worth stating
   plainly: **the palette contains blocks of every brightness, so any exposure is
   perfectly explainable.** The calibration fits from the most confident matches;
   when dark observations match dark candidates confidently, the residual is ~1 and
   no correction is ever detected. It is a degenerate fixed point. Exposure is
   **unidentifiable from appearance alone** without an external anchor.

The lighting simulation cannot supply that anchor here either: with no emitters
inside the crop, `light_field` returns level 0 and the loop breaks out before any
un-lighting is applied - the field is level 0 for 94% of observed faces. Sky light
is not modelled, and the emitters are sparse or outside the crop.

### What Template 4 needs, for the implementation plan

The blocker is a single missing quantity: **how far below full brightness the
scene was captured**. Everything else is in place. Candidate anchors, in the order
I would try them:

1. **Take the light level as an input.** A per-fixture `scene.json` field (the
   in-game brightness/gamma setting, or a stated ambient light level) turns
   `brightness(level)` into a known gain and makes the existing un-lighting path
   correct rather than inert. This is the cheapest fix and the only one that is not
   a guess.
2. **Anchor on a known surface.** If any block in the scene can be identified
   independently - a light source's own texture, a sky pixel, a block the user
   labels - the gain follows from a single ratio.
3. **Sky-light propagation with boundary seeding** in `lighting.py`, so a cave open
   to the surface gets a non-zero field. Untried, and the modelling step most likely
   to make the simulated field informative.
4. **Break the degeneracy with a prior**, e.g. restricting the palette to blocks
   plausible for the biome or structure, so that "everything is dark grey" stops
   being a free explanation. Least principled; should come last.

Worth confirming with the user before building: the in-game brightness setting used
for those screenshots, and whether the geode contains any light source at all.

### Current settings

| setting | value |
|---|---|
| palette tolerance | 4 (636 candidates, pruned by measured support) |
| `CHROMA_WEIGHT` | 0.75 |
| `TEXTURE_WEIGHT` | 40 |
| `SIMPLICITY_WEIGHT` | 3.0 |
| `PALETTE_SUPPORT` | 0.02 |
| `UPWARD_RISK` | 11.0 |
| `FALLBACK_RISK` | 3.0 |
| `EVIDENCE_POWER` | 0.0 |
| `FIT_EXPOSURE` | False |
| adjacency | 0.55 |

**Template 2: coordinate F1 94.70%, exact state 85.35%** - `dark_oak_planks` 92%,
`birch_planks` 84%, `dark_oak_log` 67%.
**Template 4: coordinate F1 49.10%, exact state 0.00%.**
86 tests pass.

---

## 8n. Correction to 8m: there is no occupancy reconstruction (2026-09-05)

**Read this before acting on 8m.** Two of that section's conclusions are wrong, and
the second one is load-bearing for Template 4. Both were found by checking the
claims rather than by new modelling.

### The occupancy stage is degenerate on both fixtures

Pass one marks a cell "seen" if its centre is the nearest projected centre at any
pixel, then `_surface_of` drops cells whose six neighbours are all present. The
code's own comment admits the first half is nearly vacuous - a 1920x1080 frame has
two million pixels against a few thousand cells, so almost every cell wins a pixel
somewhere. What was not appreciated is what that does to the second half:

| template | region | marked seen | surface after pruning | bare bounding-box shell |
|---|---|---|---|---|
| 2 | 9x12x17 = 1836 | 1833 (99.8%) | 791 | 786 |
| 4 | 23x14x22 = 7084 | 7084 (**100.0%**) | 2044 | **2044 (identical)** |

`_surface_of` applied to a nearly-full set returns the region's bounding-box shell
and nothing else. On Template 4 the match is exact, and it decomposes exactly:
y=0 and y=13 hold 506 = 23x22 cells each, y=1..12 hold 86 = the 23x22 perimeter
each, 2*506 + 12*86 = 2044, which is precisely the number of blocks we place.

**We reconstruct the box around the scene, not the scene.** Template 2 scores well
because a room is approximately its own bounding box, so the shell and the true
surface nearly coincide - 791 against 786. That is a coincidence of the fixture,
not a working algorithm, and every accuracy number banked so far rests on it.

Template 4 is a cave interior: 6225 of 7084 cells are solid, the cameras sit inside
the hollow, and the visible surface is the rock/air boundary *inside* the region.
The bounding-box shell and that boundary are disjoint sets, so Template 4 was never
classifying cave walls at all.

### What this invalidates in 8m

1. **Template 4 is a geometry problem, not an appearance problem.** Section 8m
   concluded the opposite. The confusion tables, the "everything reads too dark"
   finding, the degenerate-exposure argument and the four proposed brightness
   anchors were all measured on faces of the bounding box - outward-facing shell
   cells pointing into solid rock or away from the cameras. They describe the
   classification of meaningless input. Do not build the exposure anchor on that
   evidence.
   - The alignment sweep in 8m is still valid: `(0,0,0)` is the correct offset.
   - `FIT_EXPOSURE` remains untested rather than disproved: a global gain sweep
     from 1.0 down to 0.15 moved Template 4's state accuracy not at all
     (0.00% throughout, F1 identical to two decimals), which is now explained by
     the geometry rather than by any property of the palette.
2. **The "264 unfaced blocks -> exactly 90.0%" projection is not achievable.**
   The arithmetic is right but the premise is not. Those blocks break down as:

   | why no face sample | n |
   |---|---|
   | every camera-facing side is genuinely covered by a real block | 133 |
   | reaches the rasteriser, then dropped by uniformity/area/HUD | 128 |
   | blocked by a cell we hold that the reference does not | 3 |

   Only 13 of our 791 held cells are absent from the reference, so phantom
   occupancy is not the cause. Half of the 264 are genuinely invisible to these
   seven screenshots, and most of the remainder are genuinely contaminated - the
   uniformity filter was measured to be earning its place. **The information is not
   in the images**, so no occupancy fix recovers them; only more viewpoints would.
   Template 2 at 85.35% is much closer to the ceiling for this fixture than 8m
   implies.

### What to build

The blocker for Template 4, and the real limit on everything after it, is that
visibility is never determined. The fix is a genuine surface-finding pass that
works when the cameras are inside the scene.

**Step 1 ✔️ - make the degeneracy visible in the test suite.** A regression test that
asserts the recovered surface of a synthetic hollow scene is not its bounding-box
shell would have caught this. Cheap, and it pins the behaviour before it changes.

**Step 2 - replace the centre-pixel carve with photo-consistency space carving.**
*Attempted, not landed, and now withdrawn:* 8o measured seven per-cell criteria and
13 landed M0 and M1 before M2 hit its kill criterion, but 32 shows Template 4 still
returns the bounding shell. **67 withdraws this step as the recommended route** -
carving needs many views and this program is required to work from few. The
diagnosis below stands; the proposed remedy does not.

The standard method for interior scenes, and Minecraft gives an unusually strong
version of it: a cell is *explained* when some palette block reproduces its
appearance across every view that can see it, which is a far sharper test than the
generic colour-variance one.

- Initialise every cell solid; the initial surface is today's shell.
- For each surface cell, rectify its camera-facing faces and score the best palette
  match, exactly as the existing pipeline already does.
- Carve cells whose best match is poor or whose views disagree, exposing the layer
  behind; recompute the surface; repeat to convergence.
- Cells containing a camera are air by construction, which seeds the carve - all
  seven Template 4 cameras and all three Template 2 cameras sit inside their region.

The per-face machinery this needs already exists (`sample_visible_faces`,
`PaletteIndex.costs`, the depth buffer). What is missing is the outer loop and a
carve criterion.

**Step 3 - blocked on step 2. Only then revisit Template 4's appearance.** Exposure, the light field
and the magenta cast should be re-measured against real cave-wall faces. Any
conclusion drawn before step 2 is unfounded.

**Step 4 - blocked on step 2. Re-tune Template 2.** Its constants were fitted against the shell. They
may not be optimal against a correctly carved surface, and `UPWARD_RISK` /
`FALLBACK_RISK` in particular are corrections for evidence quality whose balance
will shift once visibility is real.

### Standing lesson

An accuracy number is not evidence that the stage producing it works. Template 2's
94.70% coordinate F1 was read for several sessions as "geometry is solved", and it
was measuring a bounding box that happened to coincide with a room. The cheap check
that would have caught it - compare the recovered surface against the trivial
baseline, the bounding-box shell - costs one line and was never run.

---

## 8o. Which carve criterion actually works, measured (2026-09-05)

Section 8n said to build space carving and guessed at the criterion. Four were
tested against both fixtures, with the reference used only as an oracle to score
them. The answer is cross-view agreement, but only once visibility is handled -
which is what makes the carve necessarily iterative rather than a single pass.

New module `reconstruction/occupancy.py` holds the agreement machinery
(`normalised_pattern`, `face_patterns`, `face_agreement`, `visible_surface`) and
`bounding_shell`, the trivial baseline any future surface must be scored against.
**It is not yet wired into the pipeline** - `reconstruct_scene_surfaces` still uses
the old centre-pixel path, so nothing in the numbers below has changed yet.

### The corrected target

8m and the first pass of this work scored against "cells with any absent
neighbour", which counts cells on the region boundary as exposed because their
neighbours fall outside the region. That flatters the bounding-box shell enormously.
Scored against cells touching air *inside* the region:

| fixture | blocks | genuinely exposed | shell precision | shell recall | shell F1 |
|---|---|---|---|---|---|
| Template 2 | 852 | 643 | 73.9% | 90.4% | 81.3% |
| Template 4 | 6225 | 703 | **5.1%** | **14.8%** | **7.6%** |

This confirms 8n from the other direction: on Template 4 the shell we currently
emit has almost nothing to do with the cave wall. The real target is 703 cells out
of 7084.

### Criteria that do not work

| criterion | result |
|---|---|
| cross-view agreement, ignoring occlusion | Template 4 exposed 0.147 vs air 0.148 - no separation whatsoever |
| best palette-match cost per cell | Template 4 exposed 2.1 vs air 2.2; buried 3.9. Separates buried from visible, not surface from air |
| sparse SIFT triangulation (`sfm.py`) | returns **1 cell** on Template 2 and **6** on Template 4. Minecraft's repeated textures defeat descriptor matching; this module is not usable for seeding occupancy |
| dense plane-sweep, square windows | median depth error 6.75 blocks on Template 4 at a 129px window, 13.8% of rays within one block. Windows were not homography-warped, so this is a lower bound, but it is far from usable |

### The criterion that does work

The first row above is misleading, and the mistake is worth recording because it
nearly buried the result: `face_patterns` ignores occlusion **by design**, so a
face was being correlated across cameras that physically cannot see it, and the
contaminated views destroyed the correlation. Re-run with oracle visibility -
marching the reference occupancy to check each camera has clear line of sight -
genuinely exposed faces agree like this:

| percentile | Template 2 | Template 4 |
|---|---|---|
| p10 | -0.380 | 0.033 |
| p25 | -0.169 | 0.247 |
| **p50** | **0.173** | **0.540** |
| p75 | 0.571 | 0.792 |
| p90 | 0.853 | 0.923 |
| share > 0.3 | 40.0% | 72.5% |

**Cross-view agreement is a strong signal on Template 4 and a weak one on
Template 2.** That ordering is the opposite of every other measurement in this
project and makes sense: Template 4 is basalt, calcite and amethyst, which carry
real texture, and has seven cameras; Template 2 is flat wood planks seen by three.

So the poses are good enough for multi-view matching. They were never the blocker.

### The constraint that shapes the design

Oracle visibility also says how much of the surface this can ever decide:

| views that genuinely see the face | Template 2 | Template 4 |
|---|---|---|
| two or more (agreement is computable) | 275 | 276 |
| exactly one | 263 | 472 |
| none | 214 | 359 |
| total exposed faces | 752 | 1107 |

Only about a quarter of Template 4's exposed faces can ever be settled by
agreement. A finished system therefore needs agreement to fix a confident core and
something else - adjacency, the material prior, the existing smoothing - to carry
the rest. Designing as though agreement will decide every cell will not work.

### Consequences for the build in 8n

1. **Keep the iterative structure**, and for a stronger reason than 8n gave: the
   criterion only works with correct visibility, and correct visibility only comes
   from a correct solid set. Each carve round sharpens both.
2. **Use agreement, not palette cost**, as the carve criterion. Palette cost does
   separate buried cells from visible ones and may still be worth keeping as a
   secondary term, but it cannot find the air/surface boundary on its own.
3. **Bootstrapping needs care.** Starting from all-solid leaves only the faces
   adjacent to the camera cells exposed, and those are seen by one camera each, so
   agreement is undefined and the carve stalls on round one. Starting from all-air
   gives the contaminated no-occlusion measurement above. The workable shape is
   greedy: seed with the highest-agreement faces from the occlusion-free pass,
   where precision is decent even though recall is poor, then alternate
   recomputing visibility and admitting more faces.
4. **Do not expect Template 2 to improve.** Its agreement signal is weak, and its
   shell already scores 81.3% F1 against the true surface because a room nearly is
   its bounding box. The realistic goal is that Template 2 does not regress while
   Template 4 becomes possible at all.

---

## 8p. Phase 2 closeout, part one: gate passed, evidence instrumented, partial geometry (2026-09-05)

Working the §10 order. Steps 1-3 are done and covered by tests; steps 4-8 remain.
No accuracy regression: Template 2 holds at 85.35% state / 94.70% F1 throughout,
and the suite is 96 tests green.

### Step 1 ✔️ - baseline frozen, and the Phase 2 gate now formally passes

`scripts/benchmark_templates.py` was reporting far less than §10 requires. It now
records both accuracy denominators, direct versus best-translation metrics, block
counts, best translation, per-state seen/correct/missed counts, palette and cache
provenance, phase timings, environment, and **every uppercase classifier constant
in force at run time** - a report that records only a score cannot be reproduced
once a default moves, and several of these defaults have moved twice in a day.

`compare_litematics` gained `per_state_counts` and a `whole_reference_state_accuracy`
property, per §6 step 2's instruction to expose these from the comparison rather
than recompute them in scripts. Reporting both denominators matters because the
conditional figure rises when the reconstruction declines to place a block, so a
recall loss can otherwise read as an accuracy gain.

Archived at `exports/baselines/t2-phase2.json` and `t4-phase2.json`:

| | Template 2 | Template 4 |
|---|---|---|
| coordinate F1 | 94.70% | 49.10% |
| conditional state accuracy | **85.35%** | 0.00% |
| whole-reference exact rate | 77.93% | 0.00% |
| best translation | (0,0,0) | (0,0,0) |
| reconstruct seconds | 9.2 | 46.2 |

**Phase 2's acceptance gate (step 13) passes**: state accuracy 85.35% >= 80%,
coordinate F1 94.70% >= 86%, and no `dark_oak_planks -> dirt` confusion. The gate
was already met numerically; what was missing was an artifact proving it.

### Step 2 ✔️ - the fallback is the majority of the evidence, not an edge case

`sample_visible_faces` takes an optional `rejections` mapping and counts every
discard reason; `reconstruct_scene_surfaces` takes an optional `diagnostics` dict
and publishes those counts alongside the evidence composition. Template 2:

| discard reason | faces |
|---|---|
| back facing | 7119 |
| neighbour occupied | 5105 |
| off frame | 424 |
| behind camera | 309 |
| HUD mask | 233 |
| uniformity | 153 |
| visible fraction | 48 |
| **accepted** | **1000** |

| evidence composition | Template 2 |
|---|---|
| resolved face samples | 847 |
| centre-patch fallback samples | **2856** |
| blocks with a resolved face | 525 |
| blocks decided by fallback alone | 1308 |
| blocks seen only from above | 56 |
| mean upward evidence share | 0.031 |

**Three quarters of all evidence is centre-patch fallback**, which 8m measured at
77.7% accurate against 91.2% for resolved faces. That is the single largest quality
gap in the pipeline and it justifies `FALLBACK_RISK` after the fact. Note also the
5105 faces culled as "neighbour occupied" - under 8n's finding that occupancy is
the bounding-box shell, an unknown share of those culls are wrong.

### Step 3 ✔️ - partial block geometry

`ModelAtlas.bake()` deliberately handles only complete cubes, because the
classifier compares full-cube appearances; a model it refuses to represent cannot
act as an occluder either, which is why carpets and stairs could hide a floor tile
undetected. New alongside it, leaving `bake` untouched:

- `ModelGeometry` - every outward face, the bounding box, and **each element's own
  rotated extent**.
- `ElementFace` - world slot, block-local corners, normal, UV-cropped texture, and
  the model's `cullface` mapped through the variant rotation.
- `element_face_corners` - generalises `FACE_BASIS` from the unit cube to an
  arbitrary box by substituting the element's extent wherever the cube used 0 or 1,
  so a partial face's texture still lines up the way a full cube's does.
- `ModelAtlas.geometry()` - bakes every element regardless of shape, with UV
  cropping so a slab's side carries the lower half of its texture rather than the
  whole of it squashed.

Verified against the real 1.21.5 jar:

| model | height | full cube | covers |
|---|---|---|---|
| `oak_planks` | 1.00 | yes | all six |
| `oak_slab` | 0.50 | no | down |
| `oak_slab_top` | 0.50 | no | up |
| `white_carpet` | 0.06 | no | down |
| `oak_stairs` | 1.00 | **no** | down, east |
| `oak_stairs` (y=90) | 1.00 | no | down, south |
| `dark_oak_fence_post` | 1.00 | no | nothing |
| `torch` | 0.62 | no | nothing |

**Coverage must not be judged from the bounding box.** A staircase spans its cell
corner to corner, so bounds alone claim it hides every face, while the shape leaves
half the top open - the first implementation had exactly this bug. `covers()` and
`full_cube` now test the union of the elements on a sixteenth-of-a-block grid,
which is exact for vanilla models because that is the grid they are authored on.

Ten tests in `tests/test_model_geometry.py`, including the staircase bounding-box
trap and that `bake` still refuses partial models so the full-cube palette cannot
quietly widen.

### Step 4 ✔️ - shape persisted, appearance left alone

`TextureSample` gained `shape_boxes` (each element's own extent after the variant
rotation), `covered_slots`, a `shape_height` property and an `occludes(slot)`
helper. Cache schema 11 -> 12, round-tripped through the JSON cache.

Only the shape is stored, not the per-face textures: occlusion needs the boxes,
and keeping a texture per face per variant would bloat the cache for nothing.
`ModelAtlas.geometry()` therefore takes `include_textures=False`, which is what
extraction uses - full extraction of 6098 samples takes 9s, a cached reload 0.2s.

Verified against the real 1.21.5 jar, per variant rather than per block name:

| state | full cube | height | covers |
|---|---|---|---|
| `oak_slab type=bottom` | no | 0.50 | down |
| `oak_slab type=top` | no | 0.50 | up |
| `oak_slab type=double` | yes | 1.00 | all six |
| `oak_stairs facing=east half=bottom shape=straight` | no | 1.00 | down, east |
| `oak_stairs facing=north half=bottom shape=straight` | no | 1.00 | down, north |
| `oak_stairs ... shape=inner_left` | no | 1.00 | down, north, west |
| `oak_stairs ... shape=outer_left` | no | 1.00 | down |
| `oak_stairs ... half=top shape=straight` | no | 1.00 | up, east |

Facing, half and shape all resolve correctly. 5957 of 6098 samples carry shape;
1476 of them occlude downward without being full cubes - that is the population
that can hide a floor tile's upward face, which 8m measured as the largest single
source of Template 2's material errors.

Six tests in `tests/test_shape_persistence.py` cover the round trip, stale-schema
re-extraction, the staircase bounding-box trap, and - importantly - that partial
states still do **not** appear in `canonical_palette()`. Recording shape so a
carpet can occlude must not quietly make it a classifier candidate; that is step 6's
job, under its own evidence gate.

One pre-existing test, `test_texture_provider_reuses_jar_cache`, asserted the
literal string `"cache_schema": 11` and so broke on the bump. It now compares
against `CACHE_SCHEMA_VERSION`, since what is worth asserting is that the cache
records whatever schema the code writes, not a frozen number.

### Step 5 ✔️ - occlusion by depth, and a hook for shape

Two changes, one measurable now and one groundwork.

**Per-pixel depth ordering.** `sample_visible_faces` sorted whole faces by their
mean depth and painted them back to front into a quarter-resolution buffer. That
gets the order wrong wherever two faces interpenetrate on screen, which is what
neighbouring faces do at a grazing angle: with equal means, whichever was painted
last claimed every pixel. `_nearest_surface` now compares depth per pixel,
interpolating inverse depth across each quad - exact under perspective, because a
face is planar and its inverse depth is therefore affine in screen space.

**Shape-aware culling.** `_hidden_by` replaces the bare `neighbour in coordinates`
test. A face is hidden only when the neighbour's *opposing* face is solid: a
block's upward face is covered when whatever sits above it has a closed underside.
A bottom slab or a carpet above hides it; a top slab above does not, because
there is air between. `sample_visible_faces` takes `occluding_slots` to supply
that, and falls back to assuming any occupied neighbour occludes, which is exactly
right for full cubes.

**Measured.** Depth ordering is correct at any buffer scale; the scale only sets
how finely the visible share is measured.

| setting | state | logs | seconds |
|---|---|---|---|
| `depth_scale` 0.25 (default) | **85.35%** | 71% | 8.9 |
| `depth_scale` 0.5 | 84.96% | 67% | 9.7 |
| `depth_scale` 1.0 | 84.96% | 67% | 12.2 |
| `depth_scale` 1.0, `visibility_power` 1.0, `minimum_visible_fraction` 0.3 | 85.22% | 70% | 12.3 |

Full resolution is the more accurate measurement and scores one to three blocks
lower, all of them `dark_oak_log`: Template 2's log inlays are thin, and every
stricter visibility test in this project has dropped them. That is noise on this
fixture rather than a real ordering between the settings, but §10 step 8 says keep
only what does not regress, so the default stays at the cheaper grid. The accurate
setting is one parameter away when a fixture needs it.

`FaceSample` now carries `visible`, the rasterised share of the face the camera
sees, and `VISIBILITY_POWER` can weigh an observation by it rather than passing or
failing it at a threshold. It defaults to 0.0 because it did not beat the baseline
here; it is the more principled treatment and worth re-testing once occupancy is
real.

Ten tests in `tests/test_occlusion.py`, including the interpenetration case the
old algorithm fails, clipping at the frame edge, and visible-share accounting for
clear and half-covered faces.

**`occluding_slots` is not yet supplied by the pipeline.** Doing so needs the
neighbour's identity, which is not known until classification, and every block the
classifier can currently place is a full cube - for which the new test and the old
one agree exactly. It becomes live in step 6, when partial states can be placed.

### Multi-view merging: what was wrong, and what fixed it (2026-09-05)

Asked directly whether merging several photographs of the same block still has
problems. Measured on Template 2's raw appearance merge - no smoothing, no
fallback pass - over 506 blocks carrying a face sample:

| | |
|---|---|
| seen by 1 / 2 / 3 frames | 254 / 197 / 55 |
| seen by two or more | 49.8% |
| frames agreed | 54.4% of those, **94.2% correct** |
| frames split | 45.6% of those, **72.2% correct** merged |
| single view only | **53.1% correct** |
| merged answer | 68.6% |
| heaviest single view | 65.4% |
| some view was right (ceiling) | 72.5% |

Merging beats trusting the heaviest view by 3.2 points, so it earns its place, but
it delivers 68.6% of an available 72.5% - **about 4% of blocks are lost because a
correct view is outvoted by wrong ones.** Whether the cameras agree is the sharpest
confidence signal in the pipeline: 94.2% against 72.2% against 53.1%.

**What worked: reconciling exposure between frames.** `_reconcile_exposure` solves
one brightness gain per frame from the faces two cameras both saw, by least
squares over every pairwise log ratio, anchored so the gains cancel in the mean.
The distinction that makes this tractable is the one 8m and 8o got wrong in the
other direction: *absolute* exposure is unidentifiable, because the palette holds
a block for every brightness and any exposure explains the picture, but *relative*
exposure between two frames is directly measured wherever they overlap.

Template 2's three photographs differ by only **x0.9913, x1.014, x0.9948** - about
2.3% peak to peak. Correcting that gains **85.35% -> 85.99%**, with `birch_planks`
84% -> 87% and `dark_oak_log` held at 71%, no class traded away. Five blocks for a
two percent correction, because the wood tones in this build sit closer together
than that. On by default; nine tests in `tests/test_exposure_reconciliation.py`.

**What did not work, all measured and left at zero:**

| term | what it weights | best result |
|---|---|---|
| `DISAGREEMENT_RISK` | pulls blocks whose views disagree toward their neighbours | 85.86% at 0.5, monotonically worse above |
| `OBLIQUITY_POWER` | discounts a face seen edge-on, by the cosine to the camera | 85.86% at 0.5, 85.48% at 1.0 |
| `VISIBILITY_POWER` | discounts a partly occluded face by its visible share | 85.99% at 0.5, no gain |

The disagreement result is worth reading carefully, because the signal behind it is
real and strong. Agreement predicts correctness; using it to make split blocks
defer to their neighbours does not help, because the loss it was aimed at happens
*inside* the merge - a correct view being averaged away - and smoothing acts after
that, on a block whose neighbours are often no better informed. The measured
headroom is still there; increasing the coupling is the wrong lever for it. A
robust combination rule that lets a decisive view win, rather than one that
dilutes it, is the untried approach.

**Caveat on all three.** Template 2 has three cameras and only half its blocks are
seen twice, so multi-view refinements have little to work with here. Template 4 has
seven cameras and 31.3% multi-view coverage, but sits at 0.00% because of 8n's
occupancy shell, so it cannot arbitrate. These terms are built, tested and
defaulted off; they deserve re-measuring on a fixture that can actually exercise
them.

### Step 6, measured before building: the premise does not hold on Template 2

Section 10 states that the remaining exact-state loss is "primarily missing
partial-model geometry and occlusion". Measured against the reference, that is not
true of Template 2, and the numbers should be checked before more is built on it.

**The addressable population is tiny.** Across the whole fixture only **17 faces**
are genuinely covered by a partial block *and* carry an observation that survived
filtering. 8m already recorded the consequence from the other side: 27 blocks have
every observed face covered, and they run at 66.7%, so roughly **nine wrong blocks
out of 119** come from this cause - under 8% of the remaining error.

**The evidence for admitting a partial is weak.** For those 17 faces, comparing the
patch against the covering block versus the block beneath:

| signal | picks the covering block |
|---|---|
| colour distance | 70.6% |
| texture correlation | **52.9%** |

Texture is a coin flip. Colour is the only signal with any power, and admitting on
colour is precisely what step 6 forbids: *"never admit a partial state merely
because its colour is close."* By covering block, `oak_stairs` (n=10) reaches 80%
and `oak_slab` (n=3) 100%, but `white_carpet` (n=3) scores **0%** - which also
corrects an earlier reading in 8m that attributed the pure-white birch patches to
carpets.

That split is informative rather than merely discouraging: slabs and stairs raise
the visible surface by half a block, which is real geometric evidence, while a
carpet raises it by a sixteenth and offers none. Any admission gate should lean on
geometry and treat carpets as the hard case, not the easy one.

**Where step 6 does pay.** The 69 partial blocks in Template 2 are 8.1% of its
reference and are currently missing entirely, so they cost the whole-reference rate
(78.52%) rather than the conditional one (85.99%). Placing them correctly is worth
several points on that metric even though it barely moves conditional accuracy.

**Template 5 is the right fixture for this work**, once it produces a valid result:

| fixture | blocks | partial | largest partial classes |
|---|---|---|---|
| Template 2 | 852 | 69 (8.1%) | `oak_stairs` 21, `white_carpet` 19, `dark_oak_fence` 13 |
| Template 5 | 31862 | **1351** (4.2%) | `purpur_stairs` **717**, `snow` 467, `purpur_slab` 113 |

Twenty times the partial content, and dominated by stairs and slabs - the classes
that carry genuine geometric evidence - rather than by carpets and fences.

**Recommended order.** Build the admission gate as a geometric test - hypothesise
the cell occupied, sample the faces that shape would present, and require the
hypothesis to explain those pixels better than the currently classified block
behind them, with cross-frame support - rather than a colour test. Develop it
against Template 5 where there is enough partial content to measure, which means it
follows 8n's carve.

### Two bugs found by looking at the litematic (2026-09-05)

The user opened the best Template 2 export and pointed out two things immediately:
every partial block is missing, and the row of cobblestone in the ceiling is
absent. Both turned out to be real and neither was a classifier problem.

**Cobblestone was not in the palette at all.** `_appearance_key` de-duplicated
states by quantised mean face colour only, and `cobblestone` and `stone` both
quantise to `(32, 32, 32, ...)` at tolerance 4, so dedup kept the shorter name and
dropped cobblestone. Its seven ceiling blocks then had no candidate that could
ever win them, and `spruce_log` - a block absent from the scene entirely -
absorbed all seven.

The key predated full-texture matching. The classifier now compares whole
rectified textures and can easily tell a lumpy face from a smooth one, but only if
both survive into the palette. `_appearance_key` now folds in a coarse pattern
signature per face: the thumbnail pooled to a 4x4 grid, mean-removed, divided by
its own contrast and quantised, plus a quantised contrast term. Palette 636 -> 726
states, with `cobblestone`, `stone`, `andesite` and `mossy_cobblestone` all
present.

**That alone does not fix the row**, and the reason is worth recording: support
pruning is global and keeps roughly six candidates ranked by how many faces they
win. A material used for seven ceiling blocks cannot outrank one used for four
hundred, so cobblestone is pruned before it can win anything. Raising the floor
does not help - `palette_minimum` 6 is a sharp optimum (7 -> 85.48%, 8 -> 75.71%),
because admitting more candidates costs more in stolen faces than the minority
material recovers. **A locally-scoped palette, rather than a global top-N, is what
this needs.** Pruning by material instead of by state (83.68%) and ranking support
by decisiveness rather than by win count (84.45%) were both tried and are both
worse.

**The partial blocks are missing from occupancy, not from classification.** Of the
70 partial blocks in Template 2:

| | |
|---|---|
| in our occupancy set | **6** |
| we place anything there | 6 |
| have any face observation | 4 |
| we place nothing at all | **64** |

None of the 64 are in the surface set and none have a face sample. A carpet's cell
centre sits in the air above the carpet, a staircase's centre is in its notch, a
fence post is a thin column - so none of them win the centre-pixel frontmost test,
and `_surface_of` returns the bounding-box shell, which they do not lie on. The
classifier never gets the chance to reject them; the cells are never proposed.

This corrects the framing in the step 6 note above. That note measured whether a
covering block can be detected *from the face it covers* - 17 faces, texture a coin
flip - and concluded the evidence was weak. That is the wrong question. A staircase
in a room is a directly visible object with its own faces; recognising one from its
own geometry should be no harder than recognising any other block. The reason none
are placed is that nothing proposes their cells.

**So step 6 should propose, not filter.** For each empty cell adjacent to placed
geometry, hypothesise each partial candidate, project the faces that shape would
present, and accept the best if it explains those pixels better than leaving the
cell empty does. That works without waiting for 8n's carve, because it proposes
cells the occupancy stage cannot.

**Value if it lands.** Placing all 70 correctly moves the whole-reference rate from
78.52% to about 86.7%, and conditional accuracy from 85.99% to about 87%, before
counting the blocks behind them that stop being contaminated.

### Step 6 attempt one: proposing partial cells, and why it failed (2026-09-05)

Built the proposal pass described above - for every empty cell touching placed
geometry, hypothesise each partial candidate, render the faces that shape would
present, and accept it if it explains those pixels better than leaving the cell
empty. Measured on Template 2 against the reference. **It does not work.**

| outcome | cells |
|---|---|
| accepted | 172 |
| exactly right | **0** |
| occupied but wrong state | 28 |
| nothing should be there | **144** |

It placed `nether_brick_slab` (59) and `dark_oak_fence_gate` (38) - materials that
appear nowhere in the fixture.

**Why, measured rather than guessed.** For the 45 cells whose true state is in the
candidate list, the rank of that true state by the residual score:

| | |
|---|---|
| median rank | **177** of 1727 |
| rank 0 | 8.9% |
| within top 20 | 15.6% |
| within top 100 | 28.9% |

The truth is buried, so no acceptance rule can rescue this. The appearance model is
the problem, and the reason is plain in hindsight: candidates were scored by **mean
colour per face** against a flat-shaded render, with no texture correlation and no
lighting correction. That is precisely the "colour is close" test step 6 forbids,
and it degenerated exactly as the plan warned. The baseline made it worse - the
"empty" alternative was the current reconstruction's own crude render, which
explains the photograph poorly, so almost any roughly-coloured hypothesis beat it,
and nothing charged a cost for adding a block.

**What a second attempt needs.** The full-cube classifier works because it rectifies
each face to a square patch and correlates the whole texture at `TEXTURE_WEIGHT`
40, after exposure reconciliation. The proposal pass must reuse that machinery
rather than a simplified render:

1. Rectify each hypothesised face and score it with `PaletteIndex.costs`, texture
   included, not a per-face mean colour.
2. Charge an explicit cost for occupying a cell, so a hypothesis must beat empty by
   a margin proportional to the pixels it claims.
3. Require the candidate to win independently in two or more frames.
4. Shortlist by materials the full-cube pass already found in the scene, since
   builds reuse materials across shapes.

**Coverage limits found on the way.** Of the 74 cells holding a block we do not
place, only 45 have their true state in the partial palette at all.
`dark_oak_fence` (13 blocks) is absent because fences are multipart blockstates and
`blockstate_variants` handles only the `variants` form; `torch` (4) falls under the
`minimum_volume` floor of 0.05, being roughly 0.0098 of a cell. Multipart support
and a lower floor for thin blocks are both needed before those can be reached.

**Kept from this work:** `VanillaJarTextureProvider.partial_palette()` builds a
de-duplicated partial candidate list - 2049 states, 1727 with textured geometry -
keyed on shape *and* appearance so a slab and its upside-down twin stay distinct.
That is sound and reusable; only the scoring built on top of it was wrong.

### Steps 7 and 8 ✔️: adjacency re-evaluated, milestones benchmarked

**Step 7 asked for three comparisons, and the answer is to change nothing.**

| setting | state accuracy | notes |
|---|---|---|
| no adjacency | 74.04% | `dark_oak_log` 78%, `dark_oak_planks` 77% |
| plain ICM at 0.55 (current) | **85.99%** | the shipped setting |
| bounded ICM at 0.35 | 85.09% | |
| bounded ICM at 0.55 | **85.99%** | identical to plain |
| bounded ICM at 0.8 | 79.18% | `dark_oak_log` collapses to 9% |
| bounded ICM at 1.2 | 79.69% | logs gone entirely |

Two things worth recording. Adjacency is the single largest contributor in the
classifier - **worth 11.95 points on its own** - which is a strong argument against
weakening it while chasing anything else. And bounded, candidate-restricted ICM is
*exactly equivalent* to the plain form at the operating point and strictly worse
above it, so the `bounded_adjacency` option earns nothing at 0.55 and need not be
carried as a live tuning knob.

Note also that turning adjacency off *raises* `dark_oak_log` from 71% to 78%. The
log inlays are the class smoothing costs, and every attempt to strengthen coupling
in this project has erased them. That trade is now measured from both directions.

**Step 8 - both fixtures at the shipped defaults**, after the palette grew from 636
to 726 states with the pattern-aware appearance key:

| | Template 2 | Template 4 |
|---|---|---|
| coordinate recall | 91.31% | 32.61% |
| coordinate precision | 98.36% | 99.32% |
| coordinate F1 | 94.70% | 49.10% |
| conditional state accuracy | **85.99%** | 0.00% |
| whole-reference exact rate | 78.52% | 0.00% |
| reconstruct seconds | 11.8 | 57.0 |

Template 2 is unchanged by the palette growth and Template 4 remains blocked on
8n's occupancy shell. Coordinate F1 has not regressed on either fixture at any
point in Phase 2.

**Phase 2 status.** Steps 1, 2, 3, 4, 5, 7 and 8 are complete and tested. Step 6 was
attempted and failed, with the failure and its measured cause recorded above; it
needs a rebuild on `PaletteIndex.costs` rather than another tuning pass. The
acceptance gate from step 13 of the original plan passes and is archived at
`exports/baselines/t2-phase2.json`.

### Step 6 attempt two: proper scoring, a tenfold gain, still not usable

Rebuilt the proposal scoring on the machinery the full-cube classifier actually
uses - rectified face patches and texture correlation - instead of mean colour
against a flat render.

**Architecture that works.** The 1711 textured partial candidates fall into only
**34 distinct shapes**. Candidates sharing a shape present the same quads, so each
distinct face is rectified once per cell per frame and every candidate carrying it
is scored in one array operation, with per-face colour and centred texture
correlation stacked exactly as `PaletteIndex` stacks the full-cube palette. That
makes scoring 539 candidate cells against 1711 candidates tractable.

**Result, measured as the rank of the true state:**

| | attempt one (mean colour) | attempt two (rectified + texture) |
|---|---|---|
| median rank of truth | 177 | **18** |
| rank 0 | 8.9% | 15.8% |
| within top 5 | 8.9% | 34.2% |
| within top 20 | 15.6% | ~56% |

A tenfold improvement in median rank, and it confirms the diagnosis: the first
attempt failed because it compared mean colours, not because proposing cells is the
wrong idea. **It is still not good enough to place blocks** - accepting the top
candidate would be right about one time in six.

**Two further ideas, both measured, both worse:**

- *Support pruning*, the trick that works for the full-cube palette, fails here.
  Restricting to the best-supported candidates pushes the truth out of the allowed
  set entirely (median rank 10000 at every threshold tried). The partial states a
  build really uses are not the ones that win the most cells; the distractors are.
- *Occlusion testing* against a depth buffer of the placed geometry costs about
  seven points of rank-0 accuracy (15.8% -> 8.6%) at any tolerance, including one
  expressed in blocks rather than inverse depth to accommodate a carpet sitting a
  sixteenth of a block above its floor. The reason is 8n again: the placed geometry
  is the bounding-box shell, so its depths are not the scene's depths and the test
  rejects good hypotheses. **This becomes worth revisiting only after the carve.**

**What is left to try**, in the order I would try it: cut the candidate set, which
is dominated by near-duplicates that differ only in a texture the correlation
cannot separate; apply the exposure reconciliation to these patches as the main
path does; and reinstate occlusion once occupancy is trustworthy.

### What is left in Phase 2

6. Shape-gated partial candidates - **parked (23), and the re-measurement it asked
   for is done (29)**: twenty-one cameras score *worse* than three, because the
   residual ambiguity is staircase orientation rather than evidence quantity. A
   spatial prior on orientation is the remaining lever, not another appearance term.

Steps 1-5, 7 and 8 are complete, tested, and free of accuracy regression:
Template 2 holds at 85.99% state / 94.70% F1 with 121 tests green.

Step 6 is what makes steps 3 to 5 pay. The geometry is built, persisted, and the
occlusion path accepts it, but nothing supplies it yet, because a partial block
still cannot be placed and so still cannot occlude anything. 8n's carve remains
outstanding and still limits what occlusion reasoning can be worth, since the
occupancy underneath it is a bounding-box shell.

---

## 8q. The carve: five approaches measured, and a correction about which fixtures need it (2026-09-05)

### Only Template 4 has broken geometry

8n said the occupancy stage returns the bounding-box shell, which is true. It did
not check how *bad* that answer is per fixture. Scored against the cells that are
genuinely exposed to in-region air:

| fixture | shell precision | shell recall | shell F1 |
|---|---|---|---|
| Template 2 | 73.9% | 90.4% | **81.3%** |
| Template 5 | 92.7% | 79.4% | **85.5%** |
| Template 4 | 5.1% | 14.8% | **7.6%** |

**Template 5's geometry is largely correct.** Its diamond box is 245,298 cells of
hollow with a 25,160-cell shell, and that shell is 92.7% exposed reference blocks.
So Template 5's 0.00% state accuracy is **entirely a lighting and appearance
failure**, not an occupancy one - the reconstruction places the right cells and
calls `diamond_block` (98, 237, 228) `purple_terracotta` (118, 70, 86). The carve
would add its interior structures, worth about 20 points of recall, but would not
move state accuracy at all.

That makes Template 5 the right fixture for the hidden-emitter and reverse-lighting
work, and it does **not** need to wait for the carve. Only Template 4 does.

### Five carve criteria, all measured, none sufficient

| approach | result |
|---|---|
| cross-view agreement, no occlusion | Template 4 exposed 0.147 vs air 0.148 - no separation |
| best palette cost per cell | exposed 2.1 vs air 2.2; re-confirmed by a ray-ordered variant that accepted 7084 of 7084 cells at every threshold |
| sparse SIFT triangulation | 1 cell on Template 2, 6 on Template 4 |
| dense plane-sweep, 129px windows | median depth error 6.75 blocks, 13.8% of rays within one block |
| free space seeded from the walked path | covers 8.8% of Template 4's air, 0.6% of Template 5's; 30% of the cells it marks are really solid. Corridor wall F1 18.2% on Template 4 - better than the shell's 7.6%, still far too low |

The walked-path idea uses real information - the screenshots are sequential, so the
player moved between consecutive positions - and it is the only seed that beats the
shell on Template 4. It is simply too thin to bootstrap a flood fill.

**The one criterion with evidence behind it remains cross-view agreement**, which
reaches median 0.540 on Template 4's genuinely exposed faces *when visibility is
known* (8o). Every attempt to obtain that visibility without already having the
answer has failed. Space carving from an all-solid volume stalls because a camera
embedded in rock sits in a one-cell cavity whose faces are each seen by exactly one
camera, so agreement is undefined; starting from air gives the contaminated
measurement above.

**What has not been tried:** iterating from the shell rather than from all-solid or
all-air. The shell is a poor visibility model on Template 4 but not a random one,
and agreement computed against it may be good enough to carve the first layer,
after which visibility improves. That is the next thing to attempt, and it should
be judged by whether Template 4's surface F1 rises above 7.6% at all.

## 8r. Template 5 proves the lighting case, and rules out a global fix (2026-09-05)

Template 5's geometry is sound (8q: shell F1 85.5%), so its 0.00% state accuracy
isolates the appearance model with nothing else confounding it. Sampling 21,363
faces of known material against the jar gives the transfer function directly.

### The scene is dark, and unevenly so

| block | n | observed | expected | ratio |
|---|---|---|---|---|
| `purpur_block` | 2230 | (60, 38, 40) | (102, 76, 102) | 0.494 |
| `purpur_pillar` | 1160 | (60, 38, 38) | (138, 104, 138) | 0.359 |
| `end_stone_bricks` | 408 | (103, 92, 53) | (174, 179, 130) | 0.505 |
| `end_stone` | 315 | (99, 81, 43) | (220, 223, 158) | 0.362 |

Fitted per-channel gain across everything: **(0.306, 0.240, 0.197)**. The scene sits
near a quarter of full brightness, and **blue is attenuated hardest**, which is
precisely what drags cyan `diamond_block` (98, 237, 228) onto `purple_terracotta`
(118, 70, 86).

It is not uniform. Over 22,172 diamond faces the ratio runs p5 0.112 to p95 0.843 -
a **7.5x spread** - and varies systematically with position and direction:

| by height | ratio | | by slot | ratio |
|---|---|---|---|---|
| y 0-9 | 0.411 | | up | 0.590 |
| y 20-29 | 0.504 | | east | 0.416 |
| y 40-49 | 0.172 | | west | 0.312 |
| y 60-69 | 0.584 | | north | 0.296 |
| y 80-89 | 0.131 | | down | **0.131** |

Vanilla shading is already applied to the expected column, so that directional
spread is *additional*. Bright bands at y 20-29 and y 60-69 with dark bands between
are what local sources look like.

### Correcting it is worth about 53 points

Per-face classification accuracy on 19,427 faces whose true state is in the palette:

| correction | accuracy |
|---|---|
| as shipped | **4.67%** |
| chroma only, no texture | 3.92% |
| chroma only (weight 1.0) | 39.16% |
| global per-channel gain | 51.57% |
| per-slot per-channel gain | **58.23%** |

Those gains are fitted from the reference, so they are a ceiling rather than a
method - but they establish that the model form is right, and that texture matters
enormously (chroma alone without texture is 3.92%).

### A single global gain cannot do it, and cannot be found blind

Sweeping a scale along the fitted cast direction, split by class:

| scale | all | `diamond_block` | everything else |
|---|---|---|---|
| 1.00 | 12.29% | 13.69% | 4.66% |
| 0.60 | 28.12% | 29.32% | **21.57%** |
| 0.24 (truth) | 52.65% | 59.81% | 13.45% |
| 0.12 | 76.41% | 88.24% | 11.72% |
| 0.03 | 77.95% | 90.19% | 11.05% |

Overall accuracy climbs as the scale falls, but **only by collapsing onto the
majority material**: diamond reaches 90% while everything else decays from its own
peak of 21.6% at scale 0.60. No single scale serves both, which is the 7.5x spread
showing up as a modelling limit rather than as noise.

And no reference-free statistic recovers the truth. Distinct materials and
top-material share both choose 0.03-0.05, the degenerate answer; mean best-match
cost chooses 1.0, no correction at all. This repeats 8m's conclusion in a setting
where the geometry is known good: **absolute exposure is not identifiable from
appearance, and scene parsimony does not break the tie** - it prefers the
explanation where everything is one block.

### What this settles

The user's proposal is now the measured requirement rather than a hypothesis. A
global exposure correction is provably insufficient here; the variation is spatial
and directional, so what is needed is a **light field** - and since no emitter is
visible inside the crop, the sources have to be inferred from the residual, with
the one-level-per-block falloff as the signature that keeps the inference honest.

Template 5 is the fixture to build it on: geometry sound, one dominant material of
known appearance over 25,160 cells, deliberately dim, and 39 views. Suggested first
milestone: fit a per-cell light factor from the residual of a first classification
pass and check whether the fitted field has radial structure around plausible
emitter positions, rather than being noise.

## 11. Execution plan after Phase 2 (2026-09-05)

Written after a stretch of trial and error. Every item below has a falsifiable
milestone and a kill criterion, so an approach is abandoned on evidence rather than
on fatigue. Nothing here should be attempted without first re-reading the "measured
and not worth keeping" tables in 8f, 8j, 8m, 8o, 8p, 8q and 8r - most obvious ideas
have already been tried and recorded.

### Standing constraints, all measured

1. **Absolute exposure is not identifiable from appearance.** The palette holds a
   block for every brightness, and scene parsimony prefers "everything is one
   material" (8r). Any lighting work needs an external anchor or spatial structure.
2. **A global gain cannot fix a lit scene.** Template 5's brightness ratio spans
   7.5x and varies with height and face direction (8r).
3. **Cross-view agreement is the only carve criterion with evidence** (median 0.540
   with known visibility, 8o), and nothing yet bootstraps the visibility it needs.
4. **Adjacency is the biggest single classifier lever**, worth 11.95 points (8p).
   Do not weaken it while chasing something else.
5. **Texture on rectified patches is essential.** Chroma without texture scores
   3.92% on Template 5 (8r).

### Order of work

**0. ✔️ Ask the user two questions.** *Both answered:* gamma is bimodal, 0-100
   or 1600 with almost nothing between, and Templates 1-2 were shot fullbright
   while 3-5 were at 100%; and more screenshots are a development aid, not the
   remedy for an algorithm that underperforms on the data it has. Nearly free, and one of them collapses the
largest unknown in the project.
   - What in-game brightness/gamma setting were the screenshots taken at, and is
     any shader or resource pack in use?
   - Are more screenshots of Template 2 possible, particularly low angles that see
     under and behind the furniture?
   *Why it matters:* a stated brightness turns `brightness(level)` into a known
   gain and removes constraint 1 entirely. Template 2's remaining error is
   dominated by 272 of 778 blocks that three cameras never resolve (8p); more views
   is the only lever that reaches them, and it is a user action, not a code change.

**1. Open - the project's largest remaining prize. Lighting on Template 5.**
   *Where it stands (30, 33):* milestone A passed - the residual does have
   radial structure and trilateration finds sources - but the shipped gate
   never fires here, and the two palette-free replacements found in 33 detect
   the dim scene and recover its gain to within 15% while still buying under a
   point. The remaining move is 17's per-slot correction. The largest measured prize: correcting appearance is
   worth about 53 points of per-face accuracy on a fixture whose geometry is sound.
   - *Milestone A:* classify once, compute the residual ratio per cell, and test
     whether the field has **radial structure** - falloff consistent with one level
     per block around a few candidate positions - rather than being noise.
   - *Pass:* residual is better explained by a handful of fitted point sources than
     by a constant, by a margin, on held-out faces.
   - *Kill:* if the residual is spatially incoherent, stop. Fall back to asking for
     the brightness setting and applying a known global gain plus the per-slot term,
     which 8r measures at 51-58% per-face against 4.67% shipped.
   - *Milestone B:* feed fitted emitters through the existing `extra_emitters` hook
     and re-run. *Pass:* Template 5 state accuracy rises above 0.00% at all;
     target 25%.

**2. ✔️ Fix the exposure clamp.** `_reconcile_exposure` pins at its 2.0 ceiling across
   all 39 Template 5 frames (8r), so the fit is unreliable exactly where it matters.
   Widen the limit and record saturation as a confidence signal rather than
   silently clamping. Small, contained, and blocks nothing.

**3. ✔️ The carve, one bounded attempt.** *Concluded in 13:* M0 and M1 landed,
   M2 hit its kill criterion. 32 shows Template 4 still needs the free-space
   carve that 8n step 2 described and this attempt did not build. Only Template 4 needs it (8q).
   - *Approach:* iterate cross-view agreement from the shell rather than from
     all-solid or all-air - the one untried bootstrap.
   - *Pass:* Template 4 surface F1 rises above the shell's 7.6%.
   - *Kill:* if two iterations do not beat 7.6%, park the carve and record that
     interior reconstruction needs a different class of method (dense MVS with
     proper cost-volume regularisation, or depth priors), which is beyond a
     tuning-scale change.

**4. Parked on its own criterion (23, 29). Step 6, partial blocks.** Currently median rank 18 of 1711, 15.8% top-1 (8s).
   Three known levers, in order: cut the near-duplicate candidate set; apply the
   exposure correction from item 1 to the patches; reinstate occlusion once item 3
   settles.
   - *Pass:* top-1 above 60% on Template 2's 45 reachable cells, at which point
     placement becomes worth wiring in.
   - *Kill:* if top-1 stays below 40% after those three, park it - placing blocks
     at that accuracy makes the litematic worse, not better.
   - Independently: **multipart blockstate support** is needed for fences
     (13 blocks in Template 2) and a lower volume floor for torches. Small, separate,
     and useful regardless of whether the proposal pass ever lands.

**5. ✔️ Local palette scoping.** *Measured and rejected in 24.* Global support pruning cannot keep a material used
   for seven blocks against one used for four hundred, which is why the cobblestone
   row stays missing even though the block is now in the palette (8s). A palette
   scoped to a spatial neighbourhood would fix it. Worth ~1 point on Template 2 and
   more on varied builds.

**6. Recommendation written, awaiting the user's word. Close Phase 4 by
   decision, not by build.** The SIFT path triangulates 1 cell on
   Template 2 and 6 on Template 4, and F3-derived poses are already good enough for
   multi-view matching (8o, 8q). Recommend dropping the COLMAP spike and recording
   F3-overlay poses as the supported method.

**7. ✔️ Phase 5, `world.zip` export.** *Delivered in 25.* Unblocked - the Phase 2 gate passes. Independent
   of everything above, and the only remaining item that adds a user-facing
   capability rather than accuracy.

### Not on the list, and why

- **Template 2 to 95% by tuning.** Ruled out: 272 of 778 blocks have no face
  evidence and run at 78.7%; perfect classification of everything visible still
  caps at 92.5% (8p). Needs more photographs.
- **Templates 3 and the village fixture.** 1.08M and 1.36M cells respectively
  against Template 5's 277k, which already takes 56 minutes. Both need region
  cropping or a performance pass first (8q).
- **More palette-pruning variants.** Material-level grouping, margin-ranked support
  and threshold sweeps are all measured and worse (8s).

## 12. Plan: fix geometric reconstruction (2026-09-05)

Nothing about block identity is worth touching until this lands. The pipeline
currently emits the region's bounding-box shell - exactly, to the block, on
Templates 4 and 5 - and every classification result to date is a paint job on that
shell.

### The metric that hid it, fixed first

Coordinate F1 compares the cells we place against **every** reference block,
including ones buried where no camera could see them. On Template 2 that reads
94.70% for a bounding box. The honest measure of the geometry stage is **surface
F1**: the cells we place against the reference cells that are genuinely exposed to
air *inside the region*. Measured that way the shipped baseline is:

| fixture | surface precision | surface recall | **surface F1** |
|---|---|---|---|
| Template 2 | 73.9% | 90.4% | **81.3%** |
| Template 5 | 92.7% | 79.4% | **85.5%** |
| Template 4 | 5.1% | 14.8% | **7.6%** |

Template 2 and 5 score well by accident - a rectangular room and a rectangular box
each approximate their own bounding box. Template 4 is a cave, and the number
collapses. **Surface F1 is the number to move.**

### M0 ✔️ - Harness and regression guard

- A reusable evaluator: given a candidate solid set, report surface
  precision/recall/F1 per fixture, plus how many placed cells are any real block.
- Report surface F1 alongside coordinate F1 in `scripts/benchmark_templates.py`.
- A test asserting the recovered surface of a hollow synthetic scene is **not** its
  bounding shell, comparing against `occupancy.bounding_shell`.
- *Pass:* the three baselines above reproduce from the harness.

### M1 ✔️ (as a seed check only) - Free-space seed

Cameras stand in air. That is the only free space known without inference.

- Seed: the cell containing each camera's feet and eye.
- *Pass:* at least 95% of seeded cells are genuinely air in the reference. (The
  walked-path extension measured 30% wrong on Template 4 and is **not** included;
  8q.)
- *Kill:* if camera cells are not reliably air, the poses or the scene transform are
  wrong and that must be fixed first.

### M2 - stopped on its kill criterion (13) - Visibility-ordered space carving

The core. Every previous attempt failed on the same chicken-and-egg: the
consistency test needs to know which cameras can see a cell, which needs the
occupancy being solved. The standard resolution is to *order* the work so that
visibility is already settled when each cell is judged.

- Start from every cell solid, minus the M1 seeds.
- Sweep a plane along each of the six axis directions. For the +X sweep, judge
  cells on the plane using **only cameras positioned at smaller x**; any occluder
  between such a camera and the cell lies at smaller x and has therefore already
  been decided on this sweep.
- Visibility per cell comes from the current carved state by the existing depth
  rasteriser, not from an assumption.
- Consistency: rectify the cell's face toward each seeing camera and take the mean
  pairwise correlation of the normalised patches - the criterion measured at median
  **0.540** on Template 4's genuinely exposed faces with known visibility, against
  0.147 without (8o). Carve when agreement falls below a threshold.
- Cells seen by fewer than two cameras cannot be tested, so they stay solid. This
  is conservative by design: it leaves the reconstruction too solid rather than
  inventing holes.
- Repeat the six sweeps until a pass carves nothing.

Carving deep in one pass is expected rather than hoped for: within a single +X
sweep, a cell carved at x=6 lets the camera behind it see x=7 later in the same
sweep, so the free space propagates without needing many rounds.

- *Pass:* Template 4 surface F1 above 7.6%, and Template 5 places cells that are
  not on its bounding shell - meaning the end city, igloo and witch hut begin to
  appear at all.
- *Kill:* two complete sweep rounds with neither number moving. Then stop, record
  that interior reconstruction needs dense multi-view stereo with cost-volume
  regularisation rather than a carve, and say so rather than trying a seventh
  variant.

### M3 - not reached, blocked on M2 - Threshold and policy

- Sweep the agreement threshold; report the precision/recall curve per fixture.
- Choose one setting for all fixtures. *Constraint:* Template 2 must not fall below
  75% surface F1, since its 81.3% is the current accidental baseline.
- Decide the single-view policy by measurement: keep solid, or admit a weaker
  single-view test. Report the cost of each.

### M4 - not reached, blocked on M2 - Performance

Template 5 is 277,160 cells against 39 frames and already takes 56 minutes to paint
a shell. Six sweeps of rectification over that volume will not be free.

- Budget: a full carve of Template 4 in under 5 minutes, Template 5 under 45.
- Levers if needed: restrict sweeps to cells adjacent to known air, cache
  rectified patches per (cell, face, frame), and coarsen the agreement patch size.

### M5 - not reached, blocked on M2 - Integrate

- Replace `_surface_of(seen_everywhere)` with the carve; keep the old path behind a
  flag for comparison.
- Re-run all fixtures and report coordinate F1, **surface F1**, recall, precision.
- **Expect classification accuracy to drop initially.** The cell set changes, the
  classifier and its constants were tuned against the shell, and Template 2's
  85.99% was fitted to cells that partly will not exist any more. That is not a
  regression to chase; it is the point at which the painting work restarts on a
  correct canvas.

### M6 - not reached, blocked on M2 - Only then, revisit identity

Re-tune the classifier, then partial blocks, then lighting - in that order, against
the corrected geometry. Every measured result in 8f through 8s was obtained on the
shell and should be treated as provisional until re-measured.

## 13. Carve attempt: M0 ✔️ and M1 ✔️ land, M2 hits its kill criterion (2026-09-05)

> **Superseded in part by 31.** M0's surface metric counted air *inside the
> region* only. That reading is wrong on a build photographed from a sealed
> interior, and the table below is not comparable with anything measured after
> 31. Template 4's honest figure is 10.37%, not 7.6%.

> **M1 verified the seed, it did not build the carve.** All it establishes is
> that camera eye cells are genuinely air. Carving the free space *along each
> ray* - 8n step 2, third bullet - was never built, and 32 identifies it as the
> reason Template 4 still emits the bounding shell.

### Delivered

**M0.** `LitematicComparison` gained `reference_exposed`, `surface_matches` and
`surface_precision` / `surface_recall` / `surface_f1`, reported by
`scripts/benchmark_templates.py` alongside the coordinate figures. Surface F1
scores placed cells against the reference cells genuinely exposed to air *inside*
the region - the set a reconstruction can actually be asked to recover - rather
than against every buried block. Baselines reproduce:

| fixture | surface precision | surface recall | surface F1 | coordinate F1 |
|---|---|---|---|---|
| Template 2 | 73.8% | 90.8% | **81.5%** | 94.7% |
| Template 5 | 92.7% | 79.4% | **85.5%** | 88.2% |
| Template 4 | 5.1% | 14.8% | **7.6%** | 49.1% |

Four tests in `tests/test_surface_metrics.py` pin it, including one where a hollow
synthetic scene's bounding shell scores **surface F1 0.0 while coordinate F1 stays
above 0.5** - the exact failure that hid this bug behind 94.70%.

**M1.** Camera eye cells are 100% genuinely air on all three fixtures. Feet cells
are not (93.1% on Template 5) because the player stands on carpets and slabs, so
the seed uses eye positions only.

### M2 stopped: the criterion does not discriminate

Seven per-cell signals have now been measured on Template 4, comparing genuinely
exposed cells against air. None separates them:

| signal | exposed | air | verdict |
|---|---|---|---|
| mean pairwise agreement, no occlusion | 0.174 | 0.148 | marginal |
| **best-pair** agreement, no occlusion | 0.367 | **0.377** | inverted |
| best palette-match cost | 2.1 | 2.2 | none |
| first ray cell the palette explains | - | - | accepts 7084 of 7084 |
| sparse SIFT triangulation | - | - | 6 cells total |
| dense plane-sweep, 129px windows | - | - | 6.75 blocks median error |
| **face uniformity (seam test)** | 6.71 | **6.28** | inverted |

With *oracle* visibility, agreement on the same faces reaches median 0.540 (8o), so
the criterion itself is sound - obtaining the visibility it needs is what fails, and
seeding cannot bootstrap it because a camera in a one-cell cavity has every
neighbouring face seen by exactly one camera, leaving agreement undefined.

### Why, and what would actually work

Two properties of this domain defeat correspondence-based depth, and they explain
all seven results together:

1. **The block grid repeats.** A depth hypothesis one block off aliases onto a
   different block of the same material and matches nearly as well. This is
   textbook periodic ambiguity, and it is why the seam test fails too - a wrong
   depth in a cave often still lands on one block face, just the wrong one.
2. **The baselines are short relative to depth.** Template 4's seven cameras sit
   within a small cave, giving a few blocks of baseline against ten to twenty
   blocks of depth. Triangulation is weak before texture is even considered.

The per-cell signal is weak but **not zero**: at an agreement threshold of 0.7,
mean-pairwise keeps 6.3% of exposed cells against 0.9% of air, a seven-to-one
ratio. Every test so far has judged each cell independently, which throws that
away. The standard way to exploit a weak per-cell signal is **global
regularisation** - a graph cut or MRF over the voxel grid with the agreement score
as the data term and a smoothness term enforcing coherent surfaces, so isolated
aliased matches lose to a consistent surface and weak evidence aggregates along it.

That is the recommended next build. It is a genuine piece of work - a cost volume
over 7k cells for Template 4, 277k for Template 5, plus an optimiser - not a
variant of what has been tried. **Do not attempt an eighth per-cell criterion.**

## 14. Template 6, and the causal chain in one fixture (2026-09-05)

A ruined portal in a dark room. Four screenshots, region 16x21x23 = **7,728 cells**,
2,376 reference blocks. Manifests authored from the HUD overlays and **poses
verified by render overlay** - the portal, the netherrack mound, the prismarine,
hay and resin walls and the pink floor all align precisely.

This is now the best diagnostic fixture in the project: small, only four views,
gamma 100%, and an unusually distinctive palette - `hay_block` 399,
`white_concrete` 399, `pink_glazed_terracotta` 294, `prismarine` 266,
`netherrack` 263 - so material confusion is not a confound the way Template 2's
four browns are.

### Baseline

| | |
|---|---|
| coordinate F1 | 84.75% |
| surface F1 | 84.45% |
| exact state | **8.55%** |
| whole-reference | 6.90% |

### It shows the whole chain at once

| | |
|---|---|
| placed cells lying on the bounding shell | **99.6%** (2141 of 2150) |
| reference blocks *off* the shell - the portal | 458 |
| of those we place | **1** (0.2%) |

The room is rectangular, so once again the shell nearly coincides with the walls
and surface F1 flatters the geometry stage at 84.45%. The structure standing in the
middle of the room - netherrack 263, `stone_bricks` 63, obsidian 31, **lava 11 and
glowstone 7** - is missed in its entirety.

**And that is why the lighting fails.** The emitters *are* present in this scene and
plainly visible in the photographs as bright patches on the walls. They belong to
the interior structure, occupancy never proposes those cells, so `light_field`
receives no emitters, the lighting loop breaks out on its first check, no
un-lighting is applied, and every block reads far too dark:

| truth | predicted | n |
|---|---|---|
| `hay_block` | `dark_oak_planks` | 265 |
| `white_concrete` | `gray_terracotta` | 188 |
| `prismarine` | `cracked_deepslate_tiles` | 165 |
| `pink_glazed_terracotta` | `creaking_heart` | 120 |

The occupancy bug and the lighting failure are **not two problems**. The first
causes the second. 8r treated Template 5's lighting as independent of its geometry;
that was right about Template 5, whose emitters lie outside the crop, and wrong as
a general statement.

### Why this is the fixture to develop the carve on

Template 4's target is a cave wall - irregular, low contrast, seven cameras with
short baselines. Template 6's target is **a discrete object standing in an empty
room**: 458 cells of high-contrast structure surrounded by air, with the room's
walls already recovered. That is the easiest possible version of the problem the
carve has to solve, and it comes with an unambiguous pass criterion.

- *Pass:* recover a meaningful share of the 458 interior cells - and specifically
  the 7 glowstone and 11 lava, which unlock the lighting for free.
- *Kill:* if the carve cannot find a solid, well-lit, high-contrast object standing
  in the middle of an otherwise empty room from four views, it will not find a cave
  wall, and the graph-cut approach in 13 should be started instead of extended.

## 15. The geometry bug, precisely located (2026-09-05)

13 said the occupancy pass returns the bounding shell. That was the symptom. The
mechanism is one line further on, and knowing it changes what to fix.

### What actually happens

Tracing Template 6 through the pipeline:

| | cells |
|---|---|
| region | 7,728 |
| coordinates with any observation | **7,725** |
| coordinates surviving fusion (`merged`) | **7,725** |
| blocks in the exported volume | 2,150 |

**The volume is filled solid and then hollowed out.** Pass three's centre-patch
fallback reaches essentially every cell in the region, so every cell gets an
observation, gets classified, and gets placed - and `_remove_enclosed_blocks` then
prunes everything whose six neighbours are present, which is exactly the region's
bounding shell. Anything standing inside a room is interior to that solid block and
is pruned with the rest.

This explains why proposing extra cells changed nothing: they were already in
`merged`. It also explains the shape of every previous result - the shell is not
something the geometry stage *computes*, it is what is left after hollowing a
solid.

### Two options added, both defaulted off

**`REQUIRE_RESOLVED_FACE`** places a block only where a face was genuinely
rectified, rather than where a centre patch guessed. **`PROPOSE_COLOUR_GAP` /
`PROPOSE_AGREEMENT`** ask the cameras directly which cells they agree on, via
`occupancy.proposed_cells`, and sample them in a second pass with the original
surface as the occluder set - proposals are sampling targets, not occluders.
Letting them occlude collapsed coordinate F1 from 74.6% to 9.6% by hiding the walls
behind them.

| fixture | config | coord F1 | surface F1 | state | whole-ref | placed |
|---|---|---|---|---|---|---|
| Template 2 | shipped | 94.70% | 81.45% | 85.99% | 78.52% | 791 |
| Template 2 | resolved-face | 73.92% | **83.28%** | **89.92%** | 53.40% | 517 |
| Template 6 | shipped | 84.75% | **84.45%** | 8.55% | 6.90% | 2150 |
| Template 6 | resolved + propose | 74.14% | 75.94% | **10.82%** | 6.61% | 1538 |

`REQUIRE_RESOLVED_FACE` raises Template 2's surface F1 *and* its state accuracy to
**89.92%** - the best classification figure the project has recorded, with
`birch_planks` at 99% and `dark_oak_log` at 82% - but costs a third of the
coverage, and it hurts Template 6. Neither is a clear default, so both ship off and
the shipped numbers are unchanged.

### Colour is a better data term than pattern

Every agreement test before this used `normalised_pattern`, which subtracts the
mean and divides by the norm, deliberately discarding absolute colour. On
Template 6 the closest colour gap between two views separates far better than
pattern correlation does:

| | exposed | air |
|---|---|---|
| colour gap (median) | **6.4** | 15.6 |
| at gap <= 4 | keeps 39.7% | keeps 11.4% |
| pattern agreement at 0.5 | keeps 40.2% | keeps 16.1% |

At equal recall colour roughly halves the false-positive rate. Combining both gives
the first configuration that adds interior structure without losing surface F1
(39 of 247 interior cells at F1 84.6% against the shell's 84.5%).

### Where this leaves the geometry

The mechanism is understood and instrumented, and the pieces exist - a better data
term, a proposal path, and a filter that distinguishes measured cells from guessed
ones. What is still missing is a principled decision about **which cells are
solid**, rather than a filter applied after filling everything. The centre-patch
fallback should not be placing blocks at all; it exists to give the classifier
something to say about a cell, and it has been quietly deciding occupancy instead.

Next: remove occupancy from the fallback's hands entirely - decide the solid set
from face evidence and proposals, and let the fallback only *label* cells that
decision already accepted.

## 16. First real geometry improvement (2026-09-05)

### Why the pipeline filled the volume solid

Three individually reasonable steps compose into the bug. Pass one projects every
cell in the region. Pass three gives every cell no face resolved a centre-patch
answer, on the stated grounds that such blocks "still deserve an answer". Then
`surface_only` prunes enclosed blocks to leave a surface. Step three is correct if
the volume is the true solid - but step two has filled it, so what it leaves is the
region's bounding shell. Each step is defensible; nobody checked the composition.

### The fix that shipped

`occupancy.proposed_cells` asks the cameras directly which cells they agree on,
using **colour** as the primary signal rather than pattern. Two details mattered:

- **Proposals are sampling targets, not occluders.** They are sampled in a second
  pass with the original surface as the occluder set. Letting them occlude
  collapsed coordinate F1 from 74.6% to 9.6% by hiding the walls behind them.
- **Proposals are restored after the enclosed-block pruning.** They are in the
  filled volume like everything else, so the prune deletes them along with the
  filling; capturing them first and re-adding afterwards is what puts structure
  standing inside a room back into the output.

Defaults `PROPOSE_COLOUR_GAP = 6.0`, `PROPOSE_AGREEMENT = 0.5`.

### Measured, against the archived baselines

| fixture | metric | before | after |
|---|---|---|---|
| Template 6 | exact state | 8.55% | **16.34%** |
| Template 6 | whole-reference | 6.90% | **13.64%** |
| Template 6 | surface F1 | 84.45% | 82.95% |
| Template 4 | surface F1 | **7.57%** | **12.85%** |
| Template 4 | coordinate F1 | 49.10% | 52.94% |
| Template 2 | exact state | 85.99% | 85.64% |
| Template 2 | whole-reference | 78.52% | 78.40% |

Template 6 nearly doubles on both state accuracy and whole-reference rate.
**Template 4's surface F1 rises from 7.57% to 12.85%, passing the gate set in 12
and 13** - the first movement on that fixture's geometry in the project. Template 2
loses 0.12 points, which is noise on a fixture whose room genuinely is its own
bounding box.

Archived at `exports/baselines/t2-geo.json`, `t6-geo.json`, `t4-geo.json`.

### What this does not fix

The volume is still filled and hollowed; proposals are a patch over that, not a
replacement for deciding occupancy. Template 4 at 12.85% surface F1 is still mostly
wrong, and Template 6's own surface F1 dipped slightly - the proposals add real
structure and some false cells with it.

`REQUIRE_RESOLVED_FACE` is implemented and **defaults off**. It places blocks only
where a face was genuinely rectified, which raises Template 2 to **89.92% state and
83.28% surface F1** - the best classification figures recorded - but drops
whole-reference from 78.52% to 53.40% by placing 517 blocks instead of 791. In
absolute terms that is 455 correct blocks against 669, so it is worse where it
counts. It is the right tool once occupancy is decided properly rather than
filtered after the fact.

**Cost:** the proposal pass samples faces across the whole region, so Template 6
went from 39s to 70s and Template 4 from 57s to 70s. Template 5 at 277,160 cells
will be substantially slower and should be measured before being run routinely.

## 17. Template 6 diagnosed, and the light anchor found (2026-09-05)

### Why Template 6 scored so badly

Not geometry - its surface F1 is 82.95%. The scene is simply dark, and every block
reads as a darker one: `hay_block` (bright yellow) as `dark_oak_planks`,
`white_concrete` as `soul_soil`, `prismarine` (teal) as `black_stained_glass`.

Measured over 2,750 faces of known material:

| | |
|---|---|
| per-channel gain (observed/expected) | **(0.491, 0.405, 0.307)** |
| mean | 0.401 |
| implied vanilla light level | **about 11** |

Blue is attenuated hardest, as on Template 5. Correcting it lifts per-face
classification from **30.95% to 53.96%**, and what remains are genuine near misses
- `white_concrete` against `glass` and `diorite`, `hay_block` against `jungle_wood`.

### The anchor: emitters are self-lit

Absolute exposure is not identifiable from appearance because everything in a dim
room is darkened by an unknown amount (8m, 8r). **A block that makes its own light
is not.** It renders at its jar colour whatever the room is doing:

| | median ratio to jar colour |
|---|---|
| emitting faces | **0.86** |
| ordinary faces | **0.40** |

Glowstone reads (110, 90, 64) against a jar colour of (103, 79, 50). That is the
absolute reference the project has been missing, and it is exactly the user's
proposal: find the light sources, and the lighting maths runs in reverse.

`_detect_emitters` matches faces against the emitting entries of the palette at
gain 1.0. Restricting to *bright* emitters matters - a dim emitter is
indistinguishable from an ordinary dark block, and admitting them drops precision
from 60% to 26%. At `EMITTER_BRIGHTNESS` 120 and `EMITTER_COST` 5 it finds 6 of 12
emitter faces on Template 6 with 4 false positives.

### The response must not be fitted

With the field built, nothing changed - because `_measure_response` fits the light
curve from *already classified* blocks, and in a dim scene those are 16% correct,
so the fit collapses to `LightResponse`'s flat default and the field dims nothing.

The user confirmed the client runs at default gamma with no shaders, which means
the curve is not an unknown at all: it is Minecraft's own `f/(4-3f)`.
`VANILLA_LIGHT_RESPONSE` uses it directly.

### Result, and why it ships opt-in

| fixture | config | state | whole-reference |
|---|---|---|---|
| Template 6 | shipped | 16.34% | 13.64% |
| Template 6 | **emitter lighting** | **30.51%** | **25.46%** |
| Template 2 | shipped | 85.64% | 78.40% |
| Template 2 | emitter lighting | 80.26% | 73.47% |

Template 6 nearly doubles again. Template 2 regresses, because it is fullbright:
seventeen faces are mistaken for emitters and the resulting field dims a scene that
was already correctly exposed.

`LIGHT_MUST_HELP` was added to tell the two cases apart by comparing how well the
palette explains the faces with and without the field. It protects Template 2
correctly - but it also blocks Template 6, whose lit cost is *worse* (120.2 against
32.0) even though its accuracy nearly doubles. The reason is that only 4 of 18
emitters are found, so most cells sit at simulated level 0, get brightened 6.7x by
`MINIMUM_LIGHT_FACTOR`, and dominate the cost. The field is right near the sources
it found and badly wrong everywhere else.

So `DETECT_EMITTERS` and `VANILLA_LIGHT_RESPONSE` ship **off**, and Template 6's
30.51% is reachable with
`detect_emitters=True, vanilla_light_response=True, light_must_help=False`.

### The next step is clear

Find more of the emitters, or give the field an **ambient floor** instead of zero.
The measurement says the ambient sits near level 11, and a field of "level 11
everywhere, higher near the sources we found" would be close to the truth
everywhere rather than right in a few places and badly wrong elsewhere. That should
let the cost gate pass honestly, at which point the feature can default on.

## 18. Hidden light sources can be located by trilateration (2026-09-05)

The user's proposal, tested and working: a light source can be found even when
nothing can see it, because light is not a guess - it falls by exactly one level
per block, so an observed brightness *is* a distance measurement.

### The method

1. For each visible face, estimate the light level of the **air cell it looks
   into** - not of the block itself - by dividing the observed colour by the
   block's jar colour and inverting Minecraft's own curve `f/(4-3f)`.
2. Correct for ambient occlusion first. Smooth lighting darkens a face by up to
   half on its own, and reading that as darkness reads it as distance, which pushes
   the inferred source several blocks too far out.
3. Every lit air cell at level `L` then votes for the cells exactly `14 - L`
   propagation steps away, measured by breadth-first search through cells that pass
   light rather than by straight-line distance, so the vote bends around the
   geometry the way light does.
4. The cells with the most votes are where a source has to be.

The **14**, rather than 15, is the block-versus-air convention: the air beside a
level-15 emitter is level 14, so the source sits one step beyond the air cell's
distance. Getting this wrong put every candidate exactly two blocks out, which is
what made the error look like a broken method rather than an off-by-one.

### Result on Template 6

Restricting to air cells at level 13 or above - those near a source, and least
likely to be dominated by a different one:

| | |
|---|---|
| true emitters in the fixture | 18 |
| exact hits in the top 5 candidates | **4** |
| distance of the top ten to the nearest emitter | **0, 0, 0, 0, 1, 1, 1, 1, 1, 1** |

Every one of the ten best candidates is on or beside a real light source, and four
are exact. Several of those sources are entirely obscured from every camera.

### What it still needs

The estimate above uses the **reference** materials to get each face's expected
colour, so it is a proof that the inversion works, not yet a usable pass. In
production the expected colour comes from the classifier, which on a dim scene is
about 30% correct. The bootstrap is plausible rather than assumed: the cells this
method relies on are the *brightest* ones, which are also the best classified,
because they are the least darkened. The loop to build is

- classify -> estimate levels from the confident faces -> trilaterate sources ->
  run `light_field` -> un-light -> classify again,

which is the structure the lighting loop already has, with `extra_emitters` as the
injection point.

This also supplies what 17 identified as the missing piece. Detecting emitters by
their self-lit appearance found only 4 of 18 and left most cells at simulated level
0, where `MINIMUM_LIGHT_FACTOR` brightened them 6.7x and wrecked the explanatory
cost. Trilateration finds sources that are invisible, so the field it produces
covers the scene rather than a few patches around the sources that happened to be
in view.

## 19. Step 6: the problem is shape selection, not material (2026-09-05)

Step 6 is the last outstanding Phase 2 item. Attempt one scored candidates by mean
colour and failed outright; attempt two rebuilt the scoring on rectified patches
and texture correlation and brought the true state from median rank 177 to 18 of
1711 - better by an order of magnitude, still far from placeable.

Splitting the search changes the picture. The 1711 candidates carry only **34
distinct shapes**, and shape is far easier to recognise than the full state:

| rank of the true... | median | top 1 | top 3 | top 5 | top 10 |
|---|---|---|---|---|---|
| **shape** (of 34) | **2** | 17.8% | 66.7% | **93.3%** | **100%** |
| full state (of 1727) | 17 | 17.8% | 26.7% | 33.3% | 35.6% |

The true shape is **always** among the ten best of thirty-four, and inside the top
five in 93% of cells.

The two top-1 figures being identical at 17.8% is the informative part: a cell
whose shape is ranked first almost always has its full state ranked first as well.
**Once the shape is right the material follows**, which is expected - picking a
material from a rectified face is what the full-cube classifier already does well.
So step 6 is not a 1711-way problem at all. It is a 34-way one, and the remaining
work is entirely in choosing among a handful of shapes.

### What to build

Rank the shapes by appearance as now, keep the best five - which contain the truth
93% of the time - and separate them with the evidence step 6 was always specified
to use and this scoring ignores:

- **projected silhouette**: the shapes differ mostly in outline, and a carpet, a
  slab and a staircase cover very different fractions of a cell;
- **visible area**, for the same reason;
- **cross-frame support**: the same shape should win from more than one camera;
- **an explicit cost for occupying a cell**, so a hypothesis must beat leaving it
  empty by a margin rather than merely fitting.

That is a much smaller and better-posed problem than the one the previous two
attempts tried to solve, and the 93%-in-top-5 figure is the ceiling it is working
against rather than an unknown.

## 20. Trilateration finished and wired in (2026-09-05)

18 proved the inversion works against the reference. This makes it a real pass,
driven by the reconstruction's own output, and lands it in the pipeline.

### What shipped

`lighting.estimate_light_levels` reads the light level of the **air cell each face
looks into**, dividing out ambient occlusion first. `lighting.locate_sources` then
has every bright cell vote for the cells exactly `15 - L - 1` propagation steps
away - breadth-first through cells that pass light, so the vote bends around
geometry - and keeps the best few, spaced apart. `raycast` calls both inside the
existing lighting loop and feeds the result to `light_field` through
`extra_emitters`. Flag: `TRILATERATE_SOURCES`, off by default.

### Three bugs found on the way, all worth knowing

1. **Occupancy.** The estimator was handed `decided`, which after the fill-solid
   bug is nearly every cell in the region. With everything opaque there is no air
   for a face to be lit by and none for light to travel through, so it measured
   nothing. It now uses the cells with a resolved face - the surface actually seen.
2. **Sampling the useless cells.** Candidates were sorted by level and the top 400
   taken, which selects exactly the cells at level 15 - and a cell at full
   brightness carries *no* distance, since `15 - L - 1` is negative. Informative
   cells are the bright ones **short of** the maximum, so the range is now capped
   at 14.
3. **The circularity.** Estimating a level needs the block's true colour, and the
   classifier picks whichever block makes observed over expected come to one - so
   1023 of 1491 cells read level 15. The method still works because the minority
   that adjacency smoothing overrode carry honest measurements, and those are
   enough to locate sources.

### Result

| fixture | config | state | whole-reference |
|---|---|---|---|
| Template 6 | shipped | 16.34% | 13.64% |
| Template 6 | **trilateration** | **30.51%** | **25.46%** |
| Template 2 | shipped | 85.64% | 78.40% |
| Template 2 | trilateration | 80.26% | 73.47% |

Six sources located from the reconstruction's own output, with no reference
consulted, nearly doubling Template 6. Template 2 regresses because it is
fullbright: there is no darkness to undo, and correcting it damages a scene that
was already right.

### The gate does not work, and cannot

`LIGHT_MUST_HELP` compares how well the palette explains the faces with and without
the field. It protects Template 2 correctly and blocks Template 6 wrongly, and the
ambient floor added to rescue it makes the fit far better (cost 120 -> 37) while
making the answer worse (30.51% -> 20.32%).

That is not a tuning failure. **Minimising best-match cost always prefers leaving an
observation alone**, because the palette holds a darker block for every brightness -
the same unidentifiability recorded in 8m and 8r, now confirmed a third time from a
new direction. No statistic computed from appearance alone will separate a
fullbright room from a dim one.

`AMBIENT_LEVELS` therefore defaults to empty and `LIGHT_MUST_HELP` stays on, so the
shipped path is unchanged; a lit scene is reconstructed with
`trilaterate_sources=True, vanilla_light_response=True, light_must_help=False`.

### The one thing that would finish this

Whether a scene is fullbright is not recoverable from the images, but it is known
to whoever took them - the user stated it directly for every fixture. A
`scene.json` field naming the capture brightness would let the pipeline choose
correctly per fixture and turn this on by default. That is the last piece, and it
is data rather than inference.

## 21. Trilateration finished, and the dim-scene gate that makes it automatic

Trilateration now ships **on by default**, with the pipeline deciding for itself
whether a scene needs un-lighting.

### The gate

20 concluded no statistic from appearance could separate a fullbright room from a
dim one, because minimising best-match cost always prefers leaving an observation
alone. That was right about *that* statistic and wrong as a general claim.

`_scene_brightness` looks only at faces where the classifier's **own** choice
stands, and asks how bright they came out against the jar. This escapes the
degeneracy because the cost is weighted toward hue and texture rather than
brightness: in a dim scene the best match is usually the right material seen too
dark, and the leftover ratio measures the darkness.

It must be computed against the **unpruned** palette. Support pruning leaves a
handful of candidates, among which every face finds something of roughly the right
brightness, and the measurement collapses to 1.0 whatever the room is doing -
Template 6 read 0.958 pruned against 0.52 unpruned, which is the difference between
the gate never firing and working.

| fixture | measured brightness | gate | outcome |
|---|---|---|---|
| Template 2 | **0.985** | blocks | fullbright, nothing to undo |
| Template 6 | **0.52** | fires | un-lighting applied |
| Template 4 | 0.657 | fires | un-lighting applied |

`DIM_SCENE_THRESHOLD` 0.85 separates them cleanly, and the user's observation that
gamma is bimodal in practice - players sit at 0-100 or at 1600, almost never
between - is why one threshold suffices: the gate only has to make a binary call.

### Shipped defaults and results

`TRILATERATE_SOURCES`, `VANILLA_LIGHT_RESPONSE` on; `LIGHT_MUST_HELP` and
`AMBIENT_LEVELS` off, both superseded by the brightness gate.

| fixture | before | after |
|---|---|---|
| Template 6 exact state | 16.34% | **30.51%** |
| Template 6 whole-reference | 13.64% | **25.46%** |
| Template 2 exact state | 85.64% | 85.64% |
| Template 2 whole-reference | 78.40% | 78.40% |
| Template 4 exact state | 0.18% | 0.09% |

Template 6 nearly doubles; Template 2 is untouched because the gate correctly
declines to act; Template 4 moves within noise of zero, its geometry being the
problem there rather than its light. Archived as `t{2,4,6}-light.json`.

Cost: the lighting pass roughly doubles runtime on the dim fixtures - Template 6
from 39s to 100s - since locating sources runs a breadth-first search per bright
cell.

### What is left on this

- **More sources.** Six are located; Template 6 has eighteen. The trilateration
  keeps only well-separated peaks, and a second pass that subtracts the light
  already explained before looking again should find more.
- **The circularity is worked around, not solved.** Levels are estimated from
  materials the classifier chose, and 1023 of 1491 cells read level 15 as a result.
  It works because the minority that adjacency smoothing overrode carry honest
  measurements. Feeding the corrected classification back in for a second round is
  the obvious next step and is already supported by `lighting_rounds`.

## 22. Step 6: negative space more than doubles shape accuracy

19 reduced step 6 from a 1711-way problem to a 34-way one and left the question of
how to separate the top five shapes. Two candidates measured:

**Cross-view agreement does not work here.** The signal that finds occupancy on a
whole cell (median rank 2 by appearance) ranks shapes at median 13, and blending it
in makes matters worse. Different shapes place their faces within a fraction of a
block of each other, so agreement cannot tell them apart, and Template 2's three
cameras see most partial faces twice at best.

**Negative space works.** Looking at what the shape *fails* to cover was the
missing evidence, and the confusion table said so plainly - the errors are vertical
extent, `oak_stairs` read as a bottom slab seven times and `white_carpet` read as a
top slab seven times, with the height table showing 1.0 -> 0.5 twelve times and
0.06 -> 0.5 nine times. The face score never noticed that a shape stopping halfway
up leaves the rest of the cell showing the same material it claims to end at.

Projecting the whole cell, subtracting what the shape covers and measuring how far
the leftover is from the shape's own colour:

| weight on negative space | median rank | top 1 | top 3 | top 5 |
|---|---|---|---|---|
| 0 (faces only) | 2.0 | 17.8% | 66.7% | **93.3%** |
| 0.4 | 1.0 | 37.8% | 66.7% | 86.7% |
| 0.7 | 1.0 | 42.2% | 64.4% | 75.6% |
| **1.5** | **1.0** | **44.4%** | 60.0% | 64.4% |
| 4.0 | 2.0 | 37.8% | 55.6% | 57.8% |

Top-1 rises from 17.8% to **44.4%** and the median rank falls to one. The cost is
in the tail - top-5 falls from 93.3% to 64.4% - which is the right trade when the
top choice is what gets placed.

### Where that leaves step 6

12 set the gates at 60% top-1 to ship and 40% to abandon. **44.4% sits between
them**, and only one lever has been pulled - one that was not even among the three
that section listed. Still untried:

- **Cut the candidate set.** 34 shapes are offered to every cell regardless of
  whether the scene contains anything like them.
- **Exposure correction on the patches**, now that 21 does this for the main path.
- **Occlusion**, which 8s measured as harmful only because the depth buffer was the
  bounding shell; 16 has improved that.
- **Cross-frame support and an explicit occupancy cost**, both named in the
  original specification and still not implemented.

Placing at 44% would add roughly 31 correct blocks of Template 2's 70 missing
partials, worth about 3.6 points of whole-reference rate, against 39 wrong
placements costing coordinate precision. That is close to break-even, which is
another way of saying the gate at 60% is the right one to aim for before this
ships.

## 23. Step 6 taken to its conclusion, and parked on its own criterion

Every lever the specification named has now been measured on Template 2, against
the cells a production pass would actually propose - empty cells touching held
geometry, which is stricter than the set used in 22 and is why the headline figure
is 36.8% rather than 44.4%. The seven extra cells there sit away from the
reconstruction and would never be offered.

| lever | top-1 shape | note |
|---|---|---|
| appearance only | 17.8% | 19 |
| **+ negative space** | **36.8%** | the only thing that helps |
| + cross-view agreement | median rank 13 | shapes sit within a fraction of a block of each other |
| + shape support by scene | 36.8% | no change; the right shapes do not win repeatedly |
| + per-frame voting | 34.2% | three cameras see most partial faces twice at best |
| + occlusion against held geometry | 34.3% | and drops three cells entirely |

**Negative space is the real finding** and it is worth keeping: measuring what a
shape *fails* to cover, which still shows its own material when the shape stops too
early, doubles top-1. The confusion table explains why nothing else mattered - the
errors are vertical extent, and only negative space sees vertical extent.

### Parked, per 12

12 set the gates at 60% top-1 to ship and 40% to abandon, with the reasoning that
placing blocks below that makes the litematic worse rather than better. **36.8% is
below the kill line**, so step 6 is parked rather than shipped. Nothing about it is
enabled by default; `partial_palette`, `ModelGeometry` and the shape scoring all
remain available and tested.

Concretely, placing at 36.8% would add about 14 correct blocks of Template 2's 38
proposable partials and 24 wrong ones - it would cost coordinate precision for
roughly 1.6 points of whole-reference rate. That is the trade the gate exists to
refuse.

### What would change the answer

Not another scoring term. The two things that would move it are structural:

- **More views.** Template 2 has three cameras and 70 partial blocks. Template 5
  has **39 cameras and 1351 partials, 717 of them `purpur_stairs`** - enough that
  per-frame voting and cross-view agreement, both useless here, would have
  something to work with. Step 6 should be re-measured there before being written
  off, and that needs Template 5 to run in reasonable time.
- **Real occupancy.** Occlusion testing hurt because the held geometry is still
  largely the bounding shell, so its depths are not the scene's depths. This is the
  same limit recorded in 8s and it has not gone away.

## 24. Multipart blockstates, and a local-palette dead end (2026-09-06)

### Local palette scoping does not rescue the cobblestone row

8s proposed scoping the palette spatially so a material used for one corner of a
build could survive a ranking made over the whole of it. Implemented as
`LOCAL_SUPPORT_*` - support counted inside coarse regions and unioned with the
global survivors - and it changes nothing: cobblestone stays at 0%, and keeping
three per region costs `birch_planks` 56 points.

The reason is that pruning was never the obstacle. Cobblestone does not *win* those
seven ceiling cells even locally; it ranks 2nd to 87th behind `deepslate`, which is
a better match for a grey down-face. The pattern-aware key from 15 did help - the
palette now holds real cobblestone rather than mossy, its expected `down` colour is
a neutral (64, 64, 64) instead of a greenish (55, 59, 48), and two of the seven
cells now rank it third - but the scoring still prefers deepslate. Ambient
occlusion is already applied, so that is not the gap either. `LOCAL_SUPPORT_KEEP`
ships at 0.

### Multipart blockstates now reach the palette

Fences, walls, panes and bars are described as a set of pieces with conditions
rather than as a list of variants, so `blockstate_variants` found no `variants` key
and returned nothing - which is why `dark_oak_fence`, thirteen blocks of
Template 2, was absent from the palette entirely.

`blockstate_multipart` reads them, and `compose` unions the pieces that apply into
one shape. Two details that are easy to get wrong:

- **The off value is implied.** A fence names `north: true` and never
  `north: false`, so reading the domain literally yields the single combination in
  which every side connects. Booleans gain their `false`; wall-style properties
  gain `none`.
- **`OR` clauses** appear in real blockstates and each alternative has to be
  accepted separately.

Cache schema 12 -> 13. The volume floor on `partial_palette` also came down from
0.05 to 0.008, just under a torch, which occupies about a hundredth of its cell.

| | before | after |
|---|---|---|
| jar samples | 6098 | **8672** |
| partial palette | 2049 | **4367** |
| full-cube palette | 726 | 726 (unchanged) |
| Template 2 partials with a candidate | 45/70 (64%) | **69/70 (99%)** |
| Template 6 partials with a candidate | - | **76/76 (100%)** |

Only a potted flower remains unreachable. Nine tests in `tests/test_multipart.py`;
no accuracy regression, Template 2 holding at 85.64% state and 80.44% surface F1.

### It makes step 6 harder, not easier

Re-measuring 22 with the wider palette:

| | 34 shapes (before) | 52 shapes (now) |
|---|---|---|
| faces only, top-1 | 17.8% | 12.2% |
| best with negative space | **44.4%** | **28.6%** |

Coverage nearly doubled and accuracy fell by a third. Fences, walls and panes are
thin, hard to see, and add eighteen more shapes to choose between, so the cells
newly made reachable are also the hardest ones. Step 6 stays parked under 23's
floor - but it is now parked for the right reason, with almost every partial block
having a candidate rather than a third of them having none.

## 25. Phase 5 done: reconstructions export as a Minecraft world (2026-09-06)

A litematic needs a mod to look at. `export/anvil.py` writes a world.zip instead -
unzip it into the saves folder and the build is standing there.

`AnvilExporter.export(volume, path, name=..., origin=...)` groups the volume into
chunks and regions, writes each chunk as compressed NBT and each region with its
sector table, and adds a `level.dat` describing a void superflat with the spawn set
at the build. `origin` places the reconstruction at the world coordinates it came
from, so a scan can be written back where it was found.

Two details of the format produce a world that loads cleanly as **empty air** if
missed, and both are pinned by tests:

- **Packed indices must not span longs.** Since 1.16 each long holds as many whole
  entries as fit and the spare high bits go unused. The litematic format packs
  continuously, so reusing `pack_palette_indices` here would shift every entry
  after the first boundary and the chunk would decode as noise.
- **Region files are addressed in 4096-byte sectors and the two header sectors
  count**, so the first chunk begins at sector two. A location entry pointing at
  sector zero means the chunk is absent.

`read_world` parses a world back independently of the writer - walking the region
tables and unpacking the sections rather than trusting anything the exporter
remembered - so an export can be checked against what it meant.

Thirteen tests in `tests/test_anvil.py`, covering the round trip, block
properties, a requested origin, negative coordinates, a build spanning three
regions, a palette wide enough to force five-bit packing, and both packing traps.
Verified on a real result: Template 6's 2365 blocks export to a 4 KB world at
(100586, 167, 11680) and read back exactly.

The CLI gained `--format world` and `--origin X Y Z` on the existing `export`
command; `--format litematic` remains the default, so nothing changes for existing
use.

**Phase 5 is complete.** It was gated behind the Phase 2 acceptance gate, which
passed in 8p.

## 26. Trilateration follow-ups: source count is not the limit (2026-09-06)

21 left two follow-ups. Both are now done and neither moves the number.

**Finding sources one at a time.** `locate_sources` took several peaks from a
single vote, which finds the brightest corner of the scene repeatedly - every cell
near one lamp votes for it, so the runners-up are its own neighbours rather than
the next lamp along. It now finds one source, drops the cells that source accounts
for, and votes again. Correct, and worth keeping for that reason, but it locates
the same six on Template 6.

**More sources do not help.** Sweeping the cap from 4 to 24 against Template 6's
eighteen real emitters leaves state accuracy flat at 30.51% and slightly worse
above sixteen. Six sources already produce a field of about the right magnitude:
the scene measures 0.401 against the jar, the field puts most cells near level 11,
and `brightness(11)` is 0.407.

`SOURCE_CANDIDATES` stays at 6.

### What the residual actually is

Template 6's remaining confusions are still darkness, not material:

| truth | predicted | n |
|---|---|---|
| `prismarine` | `gray_wool` | 195 |
| `deepslate_iron_ore` | `netherite_block` | 149 |
| `white_concrete` | `soul_soil` | 148 |
| `pink_glazed_terracotta` | `netherrack` | 119 |

But the classes the light field reaches are now largely right - `hay_block` 86%
against 18.8% before the lighting work, `chiseled_resin_bricks` 87%. So the field
is correct where it applies and insufficient elsewhere, rather than wrong
everywhere.

The per-face measurement in 17 puts the ceiling for a perfect global correction at
51.67%, and a per-slot one at 53.96%, against 30.51% now. The gap is not more
sources; it is that a single flood-filled field cannot match a per-slot correction,
since a face is lit by its own side of a block. **The next thing to try is the
per-slot term**, which 17 measured as worth a further two points over a global
gain and which nothing in the pipeline currently applies.

## 27. The light field was never propagating (2026-09-06)

Chasing 26's per-slot term turned up something larger. Forcing an extreme slot
factor changed the output not at all, which meant it was not reaching the scoring.
It was: every face was already pinned at `MINIMUM_LIGHT_FACTOR`.

**The simulated field was level 0 in 7722 of Template 6's 7728 cells** - lit only
at the six source cells themselves. `light_field` was handed `decided` as its
opacity set, and after the fill-solid bug that is nearly every cell in the region,
so the whole volume counted as light-blocking and nothing propagated a single step.

So the un-lighting 21 shipped was never a spatial correction. It was a flat
brightening by the clamp, roughly 6.7x, which happened to suit a uniformly dim
room. That is why Template 6 improved and why nothing layered on top of it did
anything.

### Fixed, and rebuilt on the right mechanism

Light now propagates through **the surface actually resolved** rather than through
every cell the classifier had an opinion about. With correct physics the field
alone scores *worse* than the accident did, which is informative: six sources
propagating freely over-light the region, so the correction becomes too small,
where the scene wants a uniform one.

Two terms restore it, both measured on faces the classifier chose for itself,
where the leftover brightness is honest:

- **`GLOBAL_EXPOSURE`** applies how far under the jar the whole scene measures -
  0.52 on Template 6 - everywhere, not only where sources were found.
- **`SLOT_RESPONSE`** applies the directional bias that a field cannot express,
  since a field gives a cell one level and every face of it shares that level.
  Measured on Template 6 as down 0.363 against up 0.764.

| configuration | Template 6 state | whole-reference |
|---|---|---|
| propagating field only | 28.54% | 23.82% |
| + global exposure | 30.06% | 25.08% |
| **+ global + slot** | **30.51%** | **25.46%** |

The same 30.51% as before, reached by a mechanism that is actually doing what it
says. Templates 2 and 4 are unchanged - Template 2's dim gate correctly declines to
act on a fullbright room - so both terms ship on.

### Why this matters beyond the number

The components are now separable and individually measurable, where before a single
degenerate clamp was doing all the work and masking everything layered onto it. 17
puts a perfect global correction at 51.67% per-face and a per-slot one at 53.96%,
against 30.51% state now; the remaining gap is a real gap rather than a term that
silently does nothing.

## 28. Overnight session summary (2026-09-06)

Worked through the handoff while the user slept. Shipped numbers at the end:

| fixture | coord F1 | surface F1 | exact state | whole-reference | seconds |
|---|---|---|---|---|---|
| Template 2 | 93.64% | 80.44% | **85.64%** | 78.40% | 19 |
| Template 6 | 83.65% | 82.95% | **30.51%** | 25.46% | 99 |
| Template 4 | 52.94% | 12.85% | 0.09% | 0.03% | 105 |

Archived as `exports/baselines/t{2,4,6}-final.json`. 155 tests green.

### Done

- **Phase 5 complete (25).** Reconstructions export as a Minecraft world -
  `export/anvil.py`, `--format world` on the CLI, an independent reader, thirteen
  tests, README section. Verified on Template 6: 2365 blocks to a 4 KB world at
  its original coordinates, round-tripped exactly.
- **Multipart blockstates (24).** Fences, walls, panes and bars now reach the
  palette. Template 2's partial coverage went from 45/70 to **69/70**, Template 6
  to 76/76. Nine tests.
- **The light field was never propagating (27).** Level 0 in 7722 of 7728 cells,
  because it was handed every cell as opaque. The un-lighting had been a flat
  clamp all along. Fixed, and rebuilt on `GLOBAL_EXPOSURE` and `SLOT_RESPONSE`,
  both measured non-degenerately; same 30.51% by a mechanism that does what it
  says.
- **Trilateration follow-ups (26).** Sources are found one at a time now, dropping
  what each explains before looking again. Source count is not the limit - four to
  twenty-four all give 30.51%.
- **Proposal performance.** Batched projection and a budget that thins frames
  rather than the region, so a 277,160-cell fixture does not spend hours on it.
- **Phase 4 recommended closed** without the COLMAP spike, on the evidence in 8o,
  8q and 20.

### Measured and rejected

- **Local palette scoping (24)** does not rescue the cobblestone row. Pruning was
  never the obstacle: cobblestone does not win those cells even locally.
- **More light sources (26)**, **per-frame voting**, **shape support** and
  **occlusion** for step 6 (23) all neutral or worse.

### Where to pick up

1. **Template 5 has not completed a run.** It was still going after a long time
   with no output. Until it does, step 6 cannot be re-measured where it has enough
   views to work (39 cameras, 1351 partial blocks), which 23 named as the one thing
   that would change its answer.
2. **Template 4 remains the geometry problem.** Surface F1 12.85%; 13 records
   seven per-cell criteria that fail and recommends a graph cut over the voxel grid
   rather than an eighth.
3. **Template 6's lighting has headroom.** 17 puts a perfect global correction at
   51.67% per-face and per-slot at 53.96%, against 30.51% state. The exposure
   estimate converges to 0.524 where the truth is 0.401 - it is measured on faces
   that already matched well, so it is biased toward one. A less biased estimator
   is the next lever.
4. **Phase 3** needs the exterior fixture cropped; at 1,362,060 cells it is five
   times Template 5.

## 29. More views do not fix step 6 (2026-09-06)

23 parked shape selection at 36.8% top-1 and named one thing that would change the
answer: re-measuring where there are enough cameras for cross-frame evidence to
mean anything, since Template 2 offers only three. Template 5 cannot be run whole -
extrapolating from Template 6 it is several hours - but the experiment only needs a
corner of it. The densest 16-cube of staircases sits at local (4, 40, 40) with
**160 of them and twenty-one cameras** in reach.

| | Template 2, 3 cameras | Template 5 corner, 21 cameras |
|---|---|---|
| cells measured | 38 | **170** |
| faces only, top-1 | 17.8% | **7.1%** |
| + negative space, top-1 | **36.8%** | **28.2%** |
| + negative space, top-5 | 64.4% | 91.8% |

**Seven times the cameras and it is worse.** The hypothesis that Template 2's three
views were the constraint is wrong.

What the corner shows instead is what the difficulty really is. Its partial blocks
are 160 `purpur_stairs` and 16 `purpur_slab` - one material, so nothing turns on
telling materials apart - and of the 52 shapes, forty are staircase orientations:
four facings times two halves times five join shapes. Top-5 is 94.7% on appearance
alone, so the *kind* of block is nearly always identified; what fails is choosing
among orientations of the same block.

That is a different problem from the one step 6 was scoped around, and more views
do not touch it, because every camera sees the same ambiguity. Negative space helps
for the same reason it helped on Template 2 - it is the only term that notices
vertical extent - but it cannot separate a staircase facing north from one facing
east.

**Step 6 stays parked, for a better-understood reason.** Not "not enough evidence"
but "the residual ambiguity is orientation, and appearance does not carry it". The
lever, if anyone returns to this, is that a staircase's orientation is strongly
constrained by its neighbours: builds run stairs in flights, and a stair almost
always faces along the run and joins what it abuts. That is a spatial prior, like
the adjacency smoothing that is worth 11.95 points on the full-cube path, rather
than another appearance term.

## 30. Template 5 runs, and the dim-scene gate fails on it (2026-09-06)

Template 5 completed a full run for the first time: **86 minutes**, 277,160 cells
against 39 frames. Archived as `exports/baselines/t5-light.json`.

| | |
|---|---|
| coordinate F1 | 87.87% |
| surface F1 | 85.15% |
| exact state | **0.00%** |
| proposed cells added | 264 |

Geometry is essentially where it was - the shell still coincides with the diamond
box - and the classification is still nothing.

### The gate reads it as bright

`scene_brightness` came back **1.006**, so the dim-scene gate declined to act and
no lighting correction was applied at all. Yet 8r measured this fixture, against
the reference, at a gain of **0.401** with a 7.5x spread: it is emphatically dim.

The gate works by measuring brightness on faces where the classifier's own choice
stands, on the reasoning that a chroma- and texture-weighted cost picks the right
material seen too dark, leaving an honest ratio. That holds on Template 6, which
reads 0.52 against a truth of 0.401. It fails here: Template 5's palette offers
blocks that are genuinely dark *and* the right hue - `purple_terracotta` and
`gray_terracotta` for a dimmed cyan - so the classifier finds a match at full
brightness and the ratio comes back at one.

So the gate is not a general test. It detects a dim scene when the palette has no
convincing dark impostor for the materials present, and misses one when it does.
Template 6's vivid, well-separated palette is the friendly case; Template 5's
greys and browns are the hostile one.

### What that means

The lighting machinery is sound - 27 rebuilt it on correct propagation, and 17
measured a per-slot correction as worth 53.96% per-face on this very fixture -
but it never runs, because the trigger cannot see that it should. That is the
single blocker on Template 5, and it is a detection problem rather than a
correction one.

Two directions, neither tried:

- **Measure brightness on a subset that cannot be faked.** The faces whose label
  adjacency smoothing imposed were excluded deliberately, since the classifier did
  not choose them; but those are exactly the faces where a dark impostor was
  overruled, and 24 shows the smoothing does override often.
- **Ask.** The user notes gamma is bimodal in practice, so this is one bit per
  fixture. 21 proposed a `scene.json` field and the user thought it would not
  help; on the evidence here it would have, since no measurement taken from the
  images alone has yet distinguished Template 5 from a bright room.

## 31. What "visible" means, and what it says about Template 4 (2026-09-06)

Template 4 has been the standing geometry failure at surface F1 **12.85%**, with
13 proposing a graph cut to fix it. Before building that, the measure itself was
worth checking - and it was wrong, though not in the direction that would have
been convenient.

### The fixtures differ enormously in density

| fixture | bounding box | solid | density |
|---|---|---|---|
| templates 2 | 9x12x17 | 852 | 46.4% |
| templates 4 | 23x14x22 | 6225 | **87.9%** |
| templates 5 | 52x82x65 | 31862 | 11.5% |
| templates 6 | 16x21x23 | 2376 | 30.7% |

`compare_litematics` counted a block as exposed only when it had air beside it
*inside the region*, which on Template 4 - a build that fills 88% of its box -
discards most of the boundary. Counting air anywhere instead lifted Template 4
from 12.85% to 85.74% and every other fixture above 85% too, which looked like the
whole geometry problem dissolving.

**It was not.** Every fixture's cameras stand *inside* its bounding box, Template
4's sealed in an interior room. So the air around the outside of that mass is air
no photograph contains, and crediting a reconstruction for reaching it credits a
guess. The three readings of "exposed", by reference block count:

| fixture | air inside the region | air anywhere | air a camera reached |
|---|---|---|---|
| templates 2 | 643 | 838 | **838** |
| templates 4 | 703 | 2629 | **576** |
| templates 5 | 29383 | 31220 | **29381** |
| templates 6 | 1999 | 2165 | **1997** |

On an open build the distinction costs nothing - Template 2's 838 and Template
5's 29381 are the any-air figures almost exactly. On Template 4 it is the whole
question, and the truth is *fewer* blocks than either guess: 576 of 6225.

### The honest scores

Flooding the air from the camera cells and scoring against blocks that touch it:

| fixture | surface F1 | precision | recall |
|---|---|---|---|
| templates 2 | 94.43% | 95.82% | 93.08% |
| templates 5 | 85.15% | 91.78% | 79.42% |
| templates 6 | 82.90% | 76.45% | 90.54% |
| templates 4 | **10.37%** | 6.48% | 26.04% |

So Template 4 is worse than the recorded 12.85%, not better. It places 2316 cells
where 576 are visible - four times too many, in the wrong places - which is the
bounding-shell failure still standing on the one fixture whose interior the shell
does not resemble. 13's graph cut is still the right next move there.

The other three sit between 83% and 94%. **That is not evidence the geometry
stage works** - 35 shows it returns the bounding shell on all four fixtures, and
those three score well only because their builds fill their own boxes.

### What changed

`compare_litematics` takes `camera_cells` and floods the air from them
(`camera_reachable_air`); without them it falls back to air inside the region, the
older conservative reading, because nothing else says which side of the boundary
the photographer stood on. `cli.py compare` grows `--cameras` / `--scene` and now
prints surface F1. `tests/test_surface_metrics.py` pins all of it, including a
seven-cube with one air pocket and a camera in it - Template 4 in miniature -
which must report exactly six seeable blocks.

**Every surface F1 recorded before this section used the in-region reading.** They
are comparable with each other and not with anything measured after.

## 32. Template 4 returns the bounding shell, verbatim (2026-09-06)

With the measure settled (31), what Template 4's geometry stage actually emits is
no longer a mystery. Against `bounding_shell(reference.size)`:

- **100.0%** of the region's shell is present in the candidate
- **88.3%** of the candidate is that shell
- cells per y-level: 506 at y=0, 506 at y=13, and 96-124 at every level between

That is a hollow box with a floor, a lid, and a one-cell wall - the region's own
boundary and nothing else. The 272 cells beyond it are the proposals from 8p. The
bounding-shell failure that surface F1 was built to expose is not historical; it
is what Template 4 still does.

### Why here and not elsewhere

The architecture fills the volume solid from pass three's centre-patch fallback
and then hollows it with `_remove_enclosed_blocks`, which keeps cells adjacent to
air. On an open build the rays escape the region, leaving genuine air to carve
against, so what survives is roughly the real surface. Template 4's cameras are
sealed in an interior room: every ray terminates on a wall, the fallback fills
everything, no air is left anywhere, and the only cells with air beside them are
the ones on the region boundary. The shell is not a guess the stage makes - it is
the only thing left after hollowing something with no interior air.

### The fix this points at

The same insight that fixed the metric applies to the reconstruction: the air that
matters is the air a camera stood in. Every cell in front of a ray's first hit is
air, known directly from the per-frame depth buffers the stage already computes
(`per_frame_front`). Carving that free space *before* the fallback fills anything
would leave the room's walls standing and the mass behind them solid, which is
both the right answer and the one that makes `_remove_enclosed_blocks` hollow
against something real.

This supersedes 13's graph cut as the next move: a graph cut over a volume with no
free-space term would converge on the same shell.

## 33. Detecting a dim scene works; correcting one still does not (2026-09-06)

30 left the lighting gate broken in two places on Template 5: it cannot tell the
scene is dim, and even forced open its gain estimate reads 1.006 where the truth
is 0.401. Both halves ask the palette, and the palette is exactly what has a
convincing dark impostor for every material in that fixture. Two palette-free
statistics were tried in their place.

### Detection: brightness spread. It works.

A fullbright room paints every face of one material the same colour up to the
fixed face shading (up 1.0, down 0.5, N/S 0.8, E/W 0.6). A lit room does not - the
same material reads bright by a torch and dark across the room. So the *spread* of
shading-normalised brightness within one material separates the two, and needs no
notion of what the material should look like.

Grouped by true material, and again by quantised chromaticity (which light scales
but barely rotates, so it stands in for material with no palette at all), taking
the median across groups of each group's p90/p10:

| fixture | gamma | by chromaticity | by true material |
|---|---|---|---|
| templates 2 | 1600% | **1.35** | 1.46 |
| templates 4 | 100% | **2.37** | 2.52 |
| templates 6 | 100% | **2.45** | 2.70 |
| templates 5 | 100% | **3.17** | 3.04 |

The palette-free statistic tracks the ground-truth one and separates cleanly, with
the fullbright fixture at 1.35 and every gamma-100 fixture above 2.37 - a gap of
1.75x with nothing in it. A threshold near 1.9 classifies all four correctly,
including Template 5, which `scene_brightness` gets wrong. It rests on one bright
fixture, so it wants Template 1 before being trusted far.

### Magnitude: relight to the chromaticity ceiling. The estimate is good; the correction is not.

If most materials have at least one fully lit face somewhere in a scene, the
brightest member of a chromaticity group is that material's unlit appearance, and
each dimmer member's shortfall against it is the light factor - recovered from
pixels alone. Per-face top-1 against the reference, on reference geometry:

| fixture | faces | raw top-1 | relit top-1 | median gain |
|---|---|---|---|---|
| templates 5 | 92762 | 1.80% | **2.66%** | **0.458** |
| templates 2 | 783 | 41.63% | **39.59%** | 0.888 |
| templates 6 | 4004 | 8.34% | **8.57%** | 0.593 |
| templates 4 | 955 | 1.88% | **1.99%** | 0.589 |

**The gain estimate is the good news.** On Template 5 it recovers 0.458 against
8r's ground-truth 0.401 - within 15%, where the shipped `scene_brightness` reads
1.006 and is wrong by two and a half times. It reads 0.888 on the one fullbright
fixture, correctly declining to correct it. That is a working, palette-free
magnitude estimator, and it is the piece 30 said was missing.

**Applying it as a single per-face scalar is the bad news.** Template 5 gains less
than a point, Templates 4 and 6 two tenths, and Template 2 loses two. So the fault
is not in knowing how dark the scene is - that is now known to 15% - but in what
is done with the number. 17's measurement that a per-slot correction reaches
53.96% per-face on this very fixture says the correction has to know which way a
face points; a flat scalar does not, and re-testing a flat scalar is wasted work.

### Where that leaves it

Both halves of the gate now have a palette-free answer - spread for *whether*,
the chromaticity ceiling for *how much* - and neither is wired in, because
together they still only move Template 5 by a point. The blocker has moved from
measurement to application: feed the recovered gain into the per-slot correction
of 17 rather than multiplying a face by it, and re-measure. That is the next
concrete move, and it is a small one - both inputs already exist.

## 34. The colour cast, and why undoing brightness alone never worked (2026-09-06)

33 recovered a dim scene's magnitude to within 15% and gained under a point of
accuracy. The reason is in 17 and was mis-summarised there and in 33: the
correction that lifts Template 6 from 30.95% to 53.96% is **per-channel**,
(0.491, 0.405, 0.307), not a scalar. Minecraft block light is warm, so a dim room
is not merely darker - it is redder, and a dimmed white block scaled up by any
single number stays orange.

### The cast is a constant

Measured against the reference on every fixture, the per-channel gain normalised
by its own mean:

| fixture | gamma | gain | normalised shape |
|---|---|---|---|
| templates 2 | 1600% | 0.930 | (1.001, 1.003, 0.997) |
| templates 6 | 100% | 0.414 | (1.201, 1.020, 0.779) |
| templates 4 | 100% | 0.316 | (1.210, 1.019, 0.771) |
| templates 5 | 100% | 0.312 | (1.292, 0.968, 0.740) |

The three lit fixtures agree to a few percent and the fullbright one is flat. So
the cast is a constant, `LIGHT_CAST = (1.234, 1.002, 0.763)`, and its magnitude is
the only unknown.

### The magnitude: take the maximum, not a percentile

> **Superseded by 55.** The comparison below is sound but incomplete: every value
> was measured on whole fixtures, so the maximum was never compared against a
> percentile *at a comparable sample size*. A maximum grows with the number of
> faces it is taken over, and the default is now `CAST_PERCENTILE = 99.0`.

33 used the 95th percentile of a chromaticity group as that group's unlit
appearance and overestimated the magnitude badly. The plain maximum is far better:

| fixture | true | p95 | p99 | **max** | cost sweep |
|---|---|---|---|---|---|
| templates 6 | 0.414 | 0.592 | 0.487 | **0.410** | 0.525 |
| templates 4 | 0.316 | 0.572 | 0.456 | **0.331** | 0.350 |
| templates 5 | 0.312 | 0.457 | 0.406 | **0.262** | 1.000 |
| templates 2 | 0.930 | 0.880 | 0.738 | 0.674 | 0.350 |

Fitting the magnitude by palette cost is unreliable (Template 5 returns 1.000),
confirming 8r's warning even with only one free parameter. **Do not retry it.**

### Per-face top-1, on reference geometry

| fixture | raw | scalar (33) | cast, true magnitude | **cast, recovered** |
|---|---|---|---|---|
| templates 5 | 1.80% | 2.66% | 27.23% | **40.08%** |
| templates 4 | 1.88% | 1.78% | 38.43% | **37.59%** |
| templates 6 | 8.34% | 6.52% | 12.24% | **17.26%** |
| templates 2 | 41.63% | 35.25% | 0.00% | **0.00%** |

The recovered magnitude *beats* the true global one on Templates 5 and 6, because
it is measured per face and so carries the scene's actual spatial variation rather
than one average. Template 2 collapses to zero, which is what the spread gate from
33 exists to prevent - it measures 1.35 there against 2.37 and above on the lit
fixtures.

### Shipped, and it is close to neutral in the pipeline

`_scene_spread`, `_chroma_key` and `_uncast` are in `raycast.py` behind
`correct_light_cast` (default on) with `DIM_SPREAD_THRESHOLD = 1.9`. Thirteen
tests in `tests/test_light_cast.py`.

| fixture | shipped | with cast |
|---|---|---|
| templates 6 | 30.61% | 30.21% |
| templates 2 | 79.87% | 79.87% (gate declines, spread 1.35) |

Almost nothing, against a 9-point offline gain - and the reason is worth knowing.
**The pipeline already un-lights the scene twice.** `scene_gain` fired at 0.621 on
Template 6, and `_self_calibrate` fits a per-channel white balance per frame from
its own most confident faces, which absorbs most of a scene-wide cast. Stacking a
third correction over-corrects: Template 6 measured 26.48% with all three, 30.21%
with the other two disabled. `_uncast` now switches both off when it fires, so
whichever runs, runs alone.

So the offline 9 points are largely points the shipped pipeline was already
getting by other means, and this is a cleaner, measured, palette-free route to the
same place rather than a gain on top of it. Its value is on the fixture where the
existing route provably fails - Template 5, whose `scene_brightness` reads 1.006.

## 35. The bounding shell is not Template 4's problem - it is every fixture's (2026-09-06)

32 concluded that Template 4 emits the region's bounding shell because its cameras
are sealed in an interior room. The first half is right and the explanation is
wrong. Measured directly - project every region cell into every frame, keep the
nearest per pixel, union across frames, and take `_surface_of` that:

| fixture | cameras | region cells | seen | surface | is the shell |
|---|---|---|---|---|---|
| templates 2 | 3 | 1836 | 1833 (99.8%) | 791 | 99.1% |
| templates 6 | 4 | 7728 | 7725 (100%) | 2150 | 99.6% |
| templates 4 | 7 | 7084 | 7084 (100%) | 2044 | 100% |
| templates 5 | 39 | 277160 | 277160 (100%) | 25160 | 100% |

**Every cell of every region is the nearest cell along some ray in some frame.**
`seen_everywhere` is therefore the whole region on all four fixtures, and the
surface of the whole region is its bounding shell by definition. This is not a
property of Template 4's sealed room; it is what the stage does everywhere.

It survives into the shipped output, which is 88-99% shell on every fixture and
covers 99.7-100% of it.

### Why three fixtures score well anyway

| fixture | shell cells that are genuinely reference blocks |
|---|---|
| templates 2 | 98.7% |
| templates 6 | 89.5% |
| templates 4 | 99.3% |
| templates 5 | 100% |

The region is drawn tightly around the build, so its boundary really is mostly
build. The shell is an *almost entirely correct* set of blocks everywhere. What it
never recovers is anything inside - which is exactly the user's own reading of the
output: "a hollow box that doesn't have any internal geometry".

Template 4 differs only in where its cameras are. Its shell is 99.3% correct
blocks and still scores 10.37% surface F1, because with the photographer sealed
inside, the *visible* surface is the interior walls and the shell is precisely the
part no photograph contains. The other three are photographed from positions where
the outer boundary is what is visible, so the same wrong answer happens to be the
right one.

### Confirmed by eye

The user opened `t5-cast.litematic` and reported it independently and more
specifically than the measurement could: Template 5 contains **an igloo, a witch
hut and an end city**, and the output has none of them - "the only thing present
is the box surrounding it". Three whole structures, standing in the interior,
recovered as nothing.

That is worth recording because the numbers hide it. Template 5 scores 85.15%
visible geometry F1, which sounds like a reconstruction with detail problems. It
is not: it is a box, and the score is high because the box is where most of the
visible surface happens to be.

### What this means for the plan

- 31's "geometry is in reasonable shape everywhere except Template 4" is wrong.
  Geometry is the bounding shell everywhere; three fixtures are shaped like their
  bounding shells. Their 83-94% surface F1 is not evidence the stage works.
- 8n's standing lesson holds exactly as written, and applies more widely than it
  was written for.
- The work item is unchanged but now applies to every fixture: recover interior
  geometry. Free-space carving from the camera positions is the route 8n step 2
  described; M1 in 12 verified the seeds are sound and never built the carve.

## 36. Template 3 brought online (2026-09-06)

Template 3 shipped with 47 screenshots, a litematic and a `location.md`, and no
camera manifest, so nothing could be said about what was visible in it. It now has
`cameras.json` and `scene.json`.

### The poses

Every screenshot carries a four-line overlay giving position, facing and
yaw/pitch. `templates 4`, whose poses were verified by wireframe, copies those
values **verbatim** - position is the player's feet, `eye_height` stays at its
1.62 default, yaw and pitch as printed - so Template 3 is transcribed the same
way. All 47 read by tiling the overlay crops eight to a sheet.

### The region offset, which `location.md` does not give

`location.md` names corners at y -74 and y 5, an 80-tall selection, but the saved
litematic is only 53 tall: Litematica trimmed the empty layers, and which 53 is
not recorded anywhere. X and Z are unambiguous (122 and 111, matching the corners
exactly).

Solved by requiring the photographer to be standing in air. Sweeping every offset
in range and counting how many of the 94 camera cells - feet and eye for 47 frames
- land inside the region and inside a solid block:

| world_min y | cells inside | inside a block |
|---|---|---|
| **-47** | **94** | **0** |
| -48 | 94 | 2 |
| -69 | 78 | 2 |
| -74 | 74 | 2 |

`-47` is the only offset with every camera inside the region and none of them
buried, so the region is world (-1680, -47, 7603) to (-1558, 6, 7714) exclusive.

### Verification

Wireframe overlays on four frames put the projected block edges on the block edges
in the photographs. The quantitative check is better: 34 established that the
per-channel colour cast of Minecraft block light is a physical constant, so
recovering it is an independent test of whether faces are being paired with the
right reference blocks.

| fixture | normalised cast shape |
|---|---|
| templates 6 | (1.201, 1.020, 0.779) |
| templates 4 | (1.210, 1.019, 0.771) |
| **templates 3** | **(1.221, 1.019, 0.760)** |
| templates 5 | (1.292, 0.968, 0.740) |

Template 3 lands among the others to within 2% over 11,368 usable faces. Wrong
poses would pair faces with the wrong blocks and return noise, not a constant.

Its mean gain is 0.383, so it is a dim fixture as its `description.md` says, and
the spread gate should fire on it.

### Not yet run

The region is 122x53x111 = **717,606 cells**, about 2.6x Template 5, which takes
85 minutes. A full reconstruction is roughly three to four hours and has not been
started: the user's standing priority is a functional program before spending that
kind of runtime, and Template 4's geometry fault (35) would dominate the result
anyway.

## 37. Why the cast correction buys nothing: 97% of the evidence is junk (2026-09-06)

Template 5 finished with the correction on. **Still 0.00% state**, geometry
unchanged, and the gate fired correctly (spread 3.004). But every label changed:

| run | dominant confusion |
|---|---|
| t5-light | diamond_block -> purple_terracotta (6815), gray_terracotta (5972), cherry_wood (3274) |
| t5-cast | diamond_block -> **pale_oak_log (24313)**, blue_terracotta (747) |

The fixture is 25,160 diamond blocks out of 31,862, and after correction 24,313 of
them read as a single pale material with the palette pruned to 7 states. That is
the collapse 8r warned of - scene parsimony preferring one material - now in the
bright direction rather than the dark one, and consistent with the magnitude being
over-estimated on this fixture (0.262 recovered against 0.312 true).

### A crop harness, so this stops costing ninety-five minutes

Template 5 takes 95 minutes and Template 3 would take three to four hours, which
makes the two fixtures where the classifier actually fails useless for iteration.
`scratchpad/crop.py` runs the **real** pipeline, palette and screenshots on a
sub-box: the reference is cut, `scene.json` rewritten for that box, and the
cameras kept are the ones near enough to see in. A 20x20x20 diamond-dominated
crop of Template 5 is 992 cells, 18 cameras and **two minutes**.

### What the crop shows

| config | geometry F1 | state |
|---|---|---|
| cast off | 49.8% | 0.00% |
| cast on | 49.8% | 0.00% |
| cast on, no palette pruning | 49.8% | 0.23% |
| cast off, no palette pruning | 49.8% | 0.23% |

Nothing moves anything, and the diagnostics say why:

| | |
|---|---|
| blocks with a resolved face | 1,062 |
| blocks from fallback only | **6,938** |
| resolved face samples | 1,791 |
| centre-patch fallback samples | **61,689** |

**Ninety-seven percent of the evidence is the centre-patch fallback.** The volume
is filled solid (35), so all 8,000 cells of the crop demand an answer, and the
6,938 that are interior get one from whatever pixel they happen to project onto.
Those 61,689 meaningless samples swamp the 1,791 genuine rectified faces, and no
colour correction applied to the genuine ones can outvote them.

That is the whole story of Templates 3, 4 and 5: **the classifier is not being
given the scene to classify.** 34's correction is measured at 37-40% per-face on
reference geometry and is worth nothing on evidence that is 97% noise.

### Consequences

- `CORRECT_LIGHT_CAST` now defaults **off**. It is neutral on Template 2, -0.4pp
  on Template 6, and 0.00% either way on Template 5. The code and its thirteen
  tests stay, because the measurement behind it - the cast is a physical constant
  and its magnitude is recoverable from pixels - is sound and will matter once the
  evidence is real. Turning it on before then is tuning against noise.

  > **Superseded in part by 55.** "0.00% either way on Template 5" was true of the
  > calibration this section shipped, not of the correction. The ceiling was a
  > *maximum*, which is not scale-invariant, so the bigger the scene the harder it
  > over-corrected - which is the `pale_oak_log` collapse recorded below. At
  > percentile 99 the same fixture's crop goes from 0.00% to **48.44%**. The
  > conclusion that the correction cannot beat 97% fallback evidence still stands
  > for the *whole* fixture; the crop's geometry is no better and it works there.
- `DIM_SPREAD_THRESHOLD` is not robust. The same fixture reads 3.004 whole and
  1.775 on a crop, so the gate would decline on the crop. Do not tune it against
  the current evidence either.
- **Everything now points at one fix**, and it is the same one 8n named a session
  ago: carve free space so cells that are not surface stop producing observations.
  Nothing downstream - lighting, palette scope, adjacency, shape selection - can be
  honestly evaluated until then.

## 38. Two things the user saw in the output, and what they actually are (2026-09-06)

The user opened `t2-final` and `t6-final` and reported two faults. Both were
checked; one is not what it looks like.

### "No stairs or carpets or fences or anything" - the palette cannot say them

Not a classification failure. The classifier's vocabulary does not contain them:

| | |
|---|---|
| canonical palette entries | **726** |
| of which partial-shaped | **1** |
| `oak_stairs`, `white_carpet`, `dark_oak_fence`, `stone_brick_slab`, `torch` | **0 each** |

Families are cube 380, other 256, log 84, utility 5, slab 1. So the output contains
exactly **zero** partial blocks on both fixtures, against references that are 8.1%
and 3.2% partial:

| fixture | reference blocks | reference partials | candidate partials |
|---|---|---|---|
| templates 2 | 852 | 69 (21 oak_stairs, 19 white_carpet, 13 dark_oak_fence, 5 gray_carpet, 5 oak_slab, 4 torch) | **0** |
| templates 6 | 2376 | 76 (38 stone_brick_slab, 20 stone_brick_stairs, 18 iron_bars) | **0** |

Template 2's whole output is four states: `birch_planks`, `dark_oak_log`,
`dark_oak_planks`, `spruce_log`.

This is 6 step 8 working as designed - a canonical full-cube palette - with step 18
never landed, because 8p step 6 parked shape selection at 36.8% against a 40%
floor. **The geometry, cache, multipart reader and shape persistence for partial
blocks all exist** (8p steps 3-5, 24); nothing admits them to the palette. So the
8.1% of Template 2 that is stairs, carpets and fences is unreachable by
construction, and the honest ceiling on that fixture is 91.9% until step 18 lands.

### "Random floating blocks" - not a merge fault

The user's hypothesis was a hiccup merging multiple renders. Template 6 does have
them - 30 connected components, 85 blocks outside the largest, 382 placed cells
(16.2%) absent from the reference. But breaking false positives down by how many
frames saw the cell:

| views | correct | false positives | error rate |
|---|---|---|---|
| 1 | 789 | 74 | **8.6%** |
| 2 | 639 | 123 | **16.1%** |
| 3 | 370 | 129 | **25.9%** |
| 4 | 185 | 56 | **23.2%** |

**More views makes a cell more likely to be wrong.** A merge fault predicts the
opposite - single-view cells would be the unreliable ones and agreement would clean
them up. This is 35 again: the volume is filled solid, so a cell that does not
exist is still "seen" by every camera whose ray reaches it, and the more cameras
reach it the more consistently the fiction is confirmed. Cross-view agreement is
corroborating the degenerate geometry rather than the real geometry.

Every observation is `ObservationKind.OBSERVED` - none are marked inferred - so the
volume gives no signal that these are guesses.

This also condemns cross-view agreement as a carve criterion in its current form:
8o measured it as the only criterion with evidence, but that measurement assumed
known visibility, and with the visibility the pipeline actually has, agreement
points the wrong way.

## 39. The palette now contains every block, and it cannot pay off yet (2026-09-06)

The user directed that the palette include **all** blocks, skipping no textures and
no partial shapes. It does now, and measuring it found the next fault immediately.

### What was built

`VanillaJarTextureProvider.complete_palette()` - every sample, full cubes and
partial shapes alike, de-duplicated on **shape and appearance together**. Shape has
to be in the key: `oak_stairs` and `oak_slab` are cut from the same plank texture
and share an appearance key exactly, so keyed on looks alone one silently replaces
the other. No volume floor either - `partial_palette` drops shapes under a
hundredth of a cell, which is a matter for evidence weighting, not for deciding a
block is unnameable.

| | canonical | complete |
|---|---|---|
| entries | 726 | **6114** |
| full cube | 726 | 748 |
| partial | **1** | **5366** |

`oak_stairs` 24 orientations, `dark_oak_fence` 16, `iron_bars` 16, plus carpets,
slabs, torches, panes, and the biome-tinted blocks that were also being dropped.
`PaletteColor` now carries `full_cube`, `shape_boxes` and `covered_slots`.

Runtime is a non-issue: Template 2 goes 18s to 22s, not 8x, because scoring is
vectorised over the candidate axis.

### Shape has to cost something

Admitting partials with no shape term put a partial block on **760 of Template 2's
814 cells** and took state accuracy from 79.87% to **2.18%**. They are cut from the
same texture as the full block, so they match a face exactly as well, and the
palette holds 24 stair orientations against one planks entry.

`slot_coverage` measures how much of a cell face a shape actually fills - full cube
1.0, bottom slab 1.0 down / 0.0 up / 0.5 sides, carpet 1/16 sides, torch ~0, stair
inner corner 0.75 up - and `SHAPE_WEIGHT` charges a candidate for the part of the
face it does not explain. Eleven tests in `tests/test_complete_palette.py`.

| shape_weight | state | partials placed | partial cells correct |
|---|---|---|---|
| 0 | 2.18% | 760 | **0** |
| 4 | 27.18% | 505 | **0** |
| 10 | 47.69% | 240 | **0** |
| 16 | 55.51% | 125 | **0** |
| 30 | 65.77% | 3 | **0** |
| *canonical palette* | *79.87%* | *0* | *0* |

### Why it pays nothing: the cells are never proposed

**Zero partial cells are recovered at any weight**, and the reason is not the
palette:

| | |
|---|---|
| reference partial cells in Template 2 | 69 |
| of those, placed at all by the geometry | **8** |
| placed with any partial state | 3 |
| placed with the right block name | **0** |

Sixty-one of the sixty-nine cells never receive a block. Stairs, carpets and fences
are furniture standing *inside* the room, and 35 established that the geometry
stage returns the region's bounding shell and recovers no interior anywhere. A
vocabulary cannot name a block in a cell nothing proposes.

The complete palette is also strictly worse on the cells that *are* proposed -
65.77% against 79.87% even with partials suppressed - because it adds genuinely
confusable candidates for full faces too. A bottom slab seen from below fills that
face completely and pays no shape cost, so it competes with the full block on equal
and correct terms. Only shape evidence separates them, which is 8p step 6, parked
at 36.8%.

### Where it is left

`complete_palette()` ships, tested, and is not yet the default, because switching
to it today trades 14 points on Template 2 for nothing. It is the right vocabulary
and it becomes useful the moment two things land, in this order: the free-space
carve (35), so interior cells are proposed at all; then shape selection (8p step
6), so a proposed cell can tell a stair from the planks it is cut from.

## 40. The free-space carve: four approaches measured, none ships (2026-09-06)

Everything converged on this - 35 (the shell), 37 (97% fallback evidence), 38
(agreement points the wrong way), 39 (61 of 69 partial cells never proposed) - so
it was built. It does not work yet, and the reason is sharper than "the criterion
does not discriminate".

Scored throughout as visible surface F1 against camera-reachable exposure (31).

### 1. Carve from all solid, front to back

8n step 2's formulation: start solid, seed air at the camera cells, depth-buffer,
carve cells the views disagree about, repeat as carving exposes the next layer.

**It stalls after 34 cells.** With every cell solid a camera sits in a one-cell
pocket and can only ever see its six neighbours, so hardly anything gets the two
views it needs to be judged. Template 2 ran eight rounds carving 4-5 cells each and
halted with one cell judged and none carved. The all-solid start is a degenerate
initialisation, not a starting point.

### 2. Agreement from all air

The other extreme, and the one `occupancy.face_agreement` already implements: with
nothing solid, every cell is evaluable with no occlusion at all.

| fixture | best F1 | at threshold | current shell F1 |
|---|---|---|---|
| templates 2 | 41.2% | 0.1 | **94.4%** |
| templates 6 | 39.8% | 0.1-0.2 | **82.9%** |
| templates 4 | **18.4%** | 0.3 | 10.4% |

Better than the shell only on Template 4. Not a replacement.

### 3. Union of the two

The shell is a good *outer* surface and agreement might supply the missing
*interior*, so take both.

| fixture | precision | recall | F1 |
|---|---|---|---|
| templates 2 | 95.8 -> 85.3% | 93.1 -> **94.7%** | 94.4 -> 89.8% |
| templates 6 | 76.4 -> 45.0% | 90.5 -> **95.6%** | 82.9 -> 61.2% |
| templates 4 | 6.5 -> 8.3% | 26.0 -> **48.8%** | 10.4 -> **14.1%** |

Recall rises everywhere and substantially - Template 4 nearly doubles - so
agreement really is finding genuine interior surface. Precision falls further, so
it is finding at least as much that is not there.

### 4. Carve seeded from the union

The natural synthesis: give the carve a high-recall, low-precision starting set and
let occlusion-aware carving prune the phantoms.

| fixture | seed F1 | carved F1 | rounds |
|---|---|---|---|
| templates 6 | 61.2% | 61.3% | 2 (4243 -> 4235 cells) |
| templates 4 | 14.1% | 14.1% | 2-3 (3396 -> 3383) |
| templates 2 | 89.8% | **83.6%** | 10 (recall 94.7% -> 79.7%) |

It prunes **eight cells** on Template 6 and then converges, and on Template 2 what
it does prune is mostly real.

### Why agreement cannot carve, which is the finding

A phantom cell sitting in front of a true surface is not seen against a different
background in each view - **the rays pass through it and land on the true surface
behind, so every view samples very nearly the same pixels.** Its patches therefore
*agree*, strongly, and agreement cannot separate it from the real cell further
along the same rays. That is why precision is poor in approach 3, why carving
prunes almost nothing in approach 4, and why 38 found agreement rising with view
count on the degenerate volume. It is a property of the signal, not a threshold
that needs tuning.

Visibility ordering was the missing ingredient for making the *measurement* honest,
and it is not sufficient, because the ambiguity is along the ray rather than
between views.

### What would actually separate them

The discriminating quantity has to be sensitive to *depth*, not just to appearance:
a cell at the wrong depth reprojects to a quad that does not correspond between
views with a real baseline, so its rectified patches should decorrelate as the
baseline grows. `rectify_face` already warps each face's projected quad to a
square, so the machinery is present; what is missing is conditioning the agreement
on baseline and testing several candidate depths per ray, which is plain multi-view
stereo along the ray rather than a per-cell test. That is the next thing to try and
it is a bigger build than anything attempted here.

**Do not re-attempt approaches 1-4.** All four are measured above.

## 41. Ray competition, and the number that ends this line of attack (2026-09-06)

40 concluded that agreement cannot carve because a phantom in front of a wall
samples the same pixels as the wall. That was right about the mechanism and wrong
about the remedy: the phantom and the wall are not competing on an absolute scale,
they are competing **for the same pixels**, and only one can be what the camera
saw. So let the cells along each ray compete and keep the winner - the selection
rule ordinary stereo uses, needing no threshold at all.

`scratchpad/raywin.py` rasterises every camera-facing candidate face at 1/8
resolution in ascending order of agreement, so the best-agreeing face is left
holding each coarse pixel, and keeps cells that hold at least `minimum_wins`.

### It doubles Template 4 and halves the others

| fixture | shipped | ray winners (best) |
|---|---|---|
| templates 2 | **94.4%** | 44.3% |
| templates 6 | **82.9%** | 42.7% |
| templates 4 | 10.4% | **21.6%** |

### Keeping the shell and adding only interior winners

The targeted version, aimed at the missing igloo, witch hut and end city: keep what
ships, add only cells off the bounding shell that win rays.

| fixture | new interior cells | **their precision** | combined F1 |
|---|---|---|---|
| templates 2 | 202-326 | **8.4-9.5%** | 94.4% -> 86.1% |
| templates 6 | 308-987 | **11.0-12.2%** | 82.9% -> 78.6% |
| templates 4 | 913-1526 | **17.0-23.1%** | 10.4% -> **15.6%** |

### The number that settles it

**Interior cells proposed by cross-view agreement are 8-23% correct.** Four to ten
of every eleven are not there. That figure barely moves across a global threshold
(40), ray competition, `minimum_wins` from 2 to 64, or restriction to the interior,
and it is why every combination trades precision for recall at a losing rate.

The recall is real - Template 4 goes from 26.0% to 57.1%, Template 6 from 90.5% to
93.9% - so the information about interior structure genuinely is in these images.
It is buried under a false positive rate that no selection rule over this signal
has moved.

### Six approaches now measured, all recorded, none shipped

| # | approach | result |
|---|---|---|
| 1 | carve from all solid, front to back (40) | stalls at 34 cells |
| 2 | agreement from all air (40) | F1 41.2 / 39.8 / 18.4% |
| 3 | union with the shipped surface (40) | recall up, F1 down except Template 4 |
| 4 | carve seeded from that union (40) | prunes 8 cells, converges |
| 5 | ray winner-take-all | Template 4 x2, Templates 2 and 6 halved |
| 6 | shell plus interior winners only | interior precision 8-23% |

**Do not attempt a seventh variation on cross-view appearance agreement.** The
ceiling on that signal has been measured six ways.

### What is left, and it is a real build

Every approach above scores a cell by *how alike its views look*. None uses the
one thing that actually separates a phantom from a surface: **depth**. A cell at
the wrong depth reprojects to quads that do not correspond between views, and the
disagreement grows with baseline - so the test has to be conditioned on baseline
and swept over candidate depths along each ray, keeping the depth whose
correspondence is best rather than the cell whose patches look most alike. That is
plain multi-view stereo, and `rectify_face` already supplies the per-face
homography it needs.

One honest aside: switching Template 4 alone to ray winners would take it from
10.4% to 21.6% today. No non-circular rule for choosing per fixture exists, and
hand-picking one per fixture is not a reconstruction.

## 42. Stereo along the ray: the precision problem is solved, shipping is not (2026-09-06)

41 ended with interior proposals 8-23% correct however the agreement signal was
thresholded, competed or restricted, and named multi-view stereo as the only thing
left. Built: `reconstruction/stereo.py`, ten tests in `tests/test_stereo.py`.

### What makes it different

Three changes, all standard stereo practice and none previously tried here.

- **Uniqueness.** Accept a ray's best depth only when it beats the runner-up by a
  margin; otherwise leave the ray unknown. The painter pass keeps the runner-up
  per pixel, which a plain depth buffer discards. A missing block costs recall, a
  wrong one is unrecoverable, so this is the right way to be wrong.
- **Cross-camera consistency.** Two cameras must independently place a cell.
- **Colour as well as pattern.** `occupancy.face_agreement` correlates
  brightness-invariant patterns only; a phantom reprojecting onto another part of
  the same wall often keeps the pattern and shifts the tint.

### The precision problem is solved

Interior proposals, against the visible reference:

| margin | templates 4 | templates 6 | templates 2 |
|---|---|---|---|
| 0.20 | 63.6% (118 cells) | 63.7% (113) | 22.2% (27) |
| 0.25 | **79.8%** (84) | **75.3%** (73) | 31.2% (16) |
| 0.30 | **98.2%** (55) | **85.0%** (40) | 27.3% (11) |

Against 8-23% for all six methods in 40 and 41. The curve is monotonic and
well-behaved, which is what a calibrated stereo confidence looks like. Colour was
worth most of it: pattern alone reached 79.7% at margin 0.35, colour reaches 98.2%
at 0.30 with more cells.

### Geometry improves and exact state collapses

Wired in as a proposal source at margin 0.25, end to end:

| fixture | geometry F1 | exact state | new cells |
|---|---|---|---|
| templates 2 | 94.43 -> 94.17% | 79.87 -> 79.46% | 13 |
| templates 6 | 82.90 -> **83.09%** | 30.37 -> **3.40%** | 31 |
| templates 4 | 10.37 -> **11.88%** | 0.67 -> 0.00% | 37 |

Geometry moves exactly as the offline measurement predicted. **Template 6's exact
state falls by twenty-seven points on thirty-one added cells**, which those cells
cannot do directly - something downstream is being destabilised by their presence.
Palette pruning, the support prior and adjacency are the candidates.

`STEREO_INTERIOR` therefore ships **off**. The module, its tests and its measured
thresholds stand.

### A wrong hypothesis, tested and reverted

The second sampling pass calls `sample_visible_faces(proposed | surface, ...)` and
keeps every face it returns, so surface faces are rectified a second time under
different occlusion. Template 6 had no proposals before stereo, so that pass had
never run there, which looked like the explanation.

Filtering the second pass to the proposals' own faces **did not fix the stereo
path and cost 3 points on the baseline** (Template 6 33.46% -> 30.37%), so those
duplicate observations are load-bearing somehow. Reverted; baseline confirmed back
at Template 2 79.87% and Template 6 30.61%.

That the duplicates help is itself worth knowing, and unexplained.

### Where to pick up

The stereo cells are good. Find what they break. The cheapest probe is to inject
them with palette pruning off, then with the support prior off, then with adjacency
off, on the Template 5 crop harness (37) or Template 6 - each run is two minutes.
Twenty-seven points is a large enough signal to localise quickly.

## 43. The stereo collapse localised: it was palette pruning (2026-09-06)

42 left stereo cells improving geometry and destroying exact state - Template 6
from 33.46% to 3.40% on thirty-one added cells. Ablation on Template 6, each run
about a hundred seconds:

| config | exact state |
|---|---|
| stereo off | **33.46%** |
| stereo on | **3.40%** |
| stereo on, no palette pruning | 12.60% |
| stereo off, no palette pruning | 13.38% |
| stereo on, no support prior | 3.40% |
| stereo on, no adjacency | 4.11% |

Pruning is worth twenty points to the baseline on its own (33.46 against 13.38).
And with pruning disabled, stereo costs **0.78pp**; with it enabled, **30pp**. The
support prior and adjacency change nothing. So the damage is entirely the
interaction with pruning.

The mechanism: pruning keeps only seven to eleven candidates on Template 6, and
proposal observations vote in that decision. A handful of wrong votes removes a
material the build actually uses, and no later stage can put it back.

### The fix, and the two attempts that were wrong

Proposals now get a material but no vote on which materials exist. Template 6
recovers **3.40% -> 25.64%**, and Template 4's exact state *rises* (0.67% ->
1.72%) alongside its geometry.

Two related ideas were tried and are wrong:

- **Excluding proposals from self-calibration** by the same argument. Template 6
  measured **12.22%** against 32.69% - the per-frame white balance wants every
  face it can get, and uncertain ones still carry a frame's colour cast. Reverted.
- **Filtering the second sampling pass to the proposals' own faces**, so surface
  faces are not rectified twice under different occlusion. No effect with stereo
  on, and **32.69% -> 28.98%** with it off. Those duplicate observations are
  load-bearing. Reverted, and now commented in place so it is not retried.

The exclusion is also conditional on stereo actually contributing. Excluding only
the stereo cells leaves Template 6 at 3.40%, so `proposed_cells`' votes are harmful
too - but only in stereo's company, and excluding them unconditionally costs the
shipped path 0.77pp. With stereo off nothing changes at all.

### Where it stands

| fixture | geometry F1 | exact state |
|---|---|---|
| templates 2 | 94.43 -> 94.17% | 79.87 -> 79.46% |
| templates 6 | 82.90 -> **83.09%** | 33.46 -> 25.64% |
| templates 4 | 10.37 -> **11.88%** | 0.67 -> **1.72%** |

Stereo is now net positive on Template 4 in both measures and still costs Template
6 7.8 points of state for 0.19 of geometry, so `STEREO_INTERIOR` stays **off**.
Baselines are bit-for-bit unchanged with it off.

### The exposure fit, probed: 2.2 of the remaining 7.8 points

| config | exact state |
|---|---|
| stereo off | 33.46% |
| stereo on | 25.64% |
| stereo on, stereo cells excluded from the exposure fit | **25.64%** |
| stereo on, exposure fit disabled | **27.84%** |
| stereo off, exposure fit disabled | 33.30% |

Excluding the stereo cells from `_reconcile_exposure` changes **nothing**, while
disabling the fit outright recovers 2.2pp. So the exposure fit is not being
polluted by the proposals' own observations - it is reacting to a changed
observation set as a whole, and reacting badly. Turning it off is not the answer
either: it is worth almost nothing on the default path (33.30 against 33.46) but
14 was measured at +0.64pp on Template 2, so it stays.

**Every global fit is now measured.** Pruning was the large one and is fixed;
adjacency and the support prior are innocent; self-calibration and the duplicate
sampling pass must keep the proposals; exposure accounts for 2.2pp indirectly.
About 5.6 points on Template 6 remain unattributed, and no single stage holds
them - which suggests the residue is the proposals' own misclassifications
spreading through adjacency into neighbours that were previously right, rather
than another global fit to find.

## 44. Error attributed by rendering, and it changes the priorities (2026-09-06)

The user's idea, and it is the best instrument this project has had: we know the
camera poses and, for testing, we have the reference litematic. So render the
reference and the reconstruction from the same pose, difference the images, and
trace each differing pixel back into both volumes to find which block is
responsible and *why*.

Every metric before this counts **cells**, so a wrong block filling half the frame
and one glimpsed edge-on at forty blocks count the same. That is not how the
result looks and not how the error is distributed.

`scratchpad/errshot.py`. The renderer need not be faithful - both volumes go
through the same one, so its bias cancels in the difference. What matters is the
owner buffer recording which cell and face each pixel came from. Comparing the two
owner buffers at a pixel gives the cause directly:

| cause | meaning |
|---|---|
| missing | reference shows a block there, reconstruction shows none |
| phantom | reconstruction shows a block where the reference has air |
| material | both hit the same cell and disagree about the material |
| occlusion | the two hit *different* cells - geometry is in the wrong place |

### The first taxonomy was degenerate, and the user caught it

The causes were originally read off whether each render hit *anything*: missing
meant the reconstruction's ray hit nothing, phantom meant the reference's did.
Both sat at zero, which the user queried. They were **unreachable by
construction** - both volumes are closed boxes, so both rays almost always hit
something. Template 6 measured both renders missing on *the same* 26.3% of pixels.
All the real content of those two categories was hiding inside "occlusion".

The distinction that works is **depth**: compare where each ray stopped.

| cause | meaning |
|---|---|
| material | both rays stop at the same cell and disagree about the block |
| missing | ours stops *further* - we never placed what the reference has in front |
| phantom | ours stops *nearer* - we put something there that is not |
| displaced | same depth, different cells |
| silhouette | exactly one ray leaves the volume |

### Template 2

| cause | share of error | pixels |
|---|---|---|
| missing | **45.3%** | 199,703 |
| material | **44.5%** | 183,514 |
| phantom | 5.9% | 62,467 |
| displaced | 3.6% | 48,614 |
| silhouette | 0.7% | 6,913 |

The old catch-all's 54.9% resolves almost entirely into **missing**: blocks that
are simply not there, which is the carpets and stairs.

| block | share of visible error |
|---|---|
| white_carpet | **43.3%** |
| oak_stairs | **21.1%** |
| gray_carpet | 9.9% |
| dark_oak_planks | 9.3% |
| oak_slab | 4.3% |

**Partial blocks are 84.1% of all visible error, against 8.1% of cells.** Carpets
cover floors and stairs line walls, so they own screen area out of all proportion
to their cell count - and until 39 the palette could not name a single one of them.

### Template 6

| cause | share of error |
|---|---|
| material | **80.4%** |
| phantom | 12.5% |
| missing | 6.6% |
| displaced | 0.5% |

Partial blocks are only **4.4%** of Template 6's visible error, against 84.1% of
Template 2's - the two fixtures fail in genuinely different ways.

| block | share | worst confusion |
|---|---|---|
| white_concrete | 27.9% | -> soul_soil (15.3%) |
| pink_glazed_terracotta | 22.9% | -> netherrack (10.2%) |
| prismarine | 22.2% | -> gray_wool (13.0%) |
| deepslate_iron_ore | 12.3% | -> netherite_block (6.1%) |

Four bright materials read as four dark ones and account for 85% of the fixture's
visible error. That is the lighting fault of 17 and 34, stated in the units that
matter.

### What this changes

- **The complete palette (39) is worth far more than the cell counts implied.**
  8.1% of Template 2's cells, 84.1% of its visible error. It is now the highest
  value work in the project rather than a nice-to-have, and it is already built -
  what it needs is shape selection so the cells can be *named*, and interior
  geometry so they are proposed at all.
- **Template 6 is a lighting problem and nothing else**, at four fifths of its
  error, which retires any remaining doubt about where its effort belongs.
- **Cell-counting metrics understate thin blocks by an order of magnitude.** Every
  ranking in this document taken from `state_accuracy` or surface F1 should be
  re-read with that in mind.

This should be productionised out of the scratchpad; it is a permanent instrument,
not an experiment.

## 45. How the error instrument changes the plan (2026-09-06)

44 gave the project a way to measure what the output *looks like* rather than how
many cells it gets right. Four consequences, one of which invalidates earlier work.

### 1. Every constant was tuned against the wrong objective

`UPWARD_RISK`, `FALLBACK_RISK`, adjacency strength, `SHAPE_WEIGHT`, the palette
support thresholds - all fitted against `state_accuracy`, which counts cells and
so weights a carpet covering a floor the same as an edge-on face forty blocks
away. Some are likely sitting at values that are wrong by the standard that
matters. **Re-tuning against pixel-weighted error is the cheapest available work
and may be free accuracy sitting in mis-tuned constants.**

### 2. The fixtures need opposite things

| | templates 2 | templates 6 |
|---|---|---|
| missing | **45.3%** | 6.6% |
| material | 44.5% | **80.4%** |
| phantom | 5.9% | 12.5% |
| partial blocks | **84.1%** | 4.4% |

Template 2 needs thin blocks placed; Template 6 needs lighting. Stop looking for a
single change that lifts both.

### 3. Phantom error is visible for the first time, and it is 12.5% on Template 6

Cell metrics hid it entirely. It is the direct cost of proposals and stereo
additions, so the trade stereo makes - buying *missing*, paying in *phantom* - is
now measurable in one currency. That reframes 42 and 43: stereo attacks 45.3% of
Template 2's error and 6.6% of Template 6's, which is why it helps one and hurts
the other.

### 4. A closed loop, not an aggregate

The tool names the responsible blocks, so a fix can be checked against the thing it
was meant to move. `white_carpet` is 43.3% of Template 2's error; after a change,
look at that number rather than watching one aggregate wobble.

### The caveat, which matters

This instrument reads the reference litematic. It is a **diagnostic, not a
runtime signal**, and tuning constants against it on three fixtures risks
overfitting to those three. Fit on some, verify on held out ones. **Template 3's
manifest (36) is now more valuable than it looked**, because it is an unseen
fixture.

### Order of work

1. Re-tune the shipped constants against pixel error.
2. Shape selection, so the complete palette (39) can name partials - 84% of
   Template 2's error and the largest single lever in the project.
3. The cast correction (34) for Template 6.

Template 4 is unaffected by any of this: it places the wrong cells entirely, so
there is nothing to attribute. Its geometry remains a separate problem.

## 46. Re-tuning against pixel error: the adjacency default was the worst value (2026-09-06)

45's first item, and it found something that had nothing to do with pixel error.

### The risk constants are inert

`UPWARD_RISK` and `FALLBACK_RISK` were credited with +2.7pp in 8p. Swept against
pixel error on Template 2 they move nothing at all:

| config | relative error | state |
|---|---|---|
| shipped | 1.0000 | 79.87% |
| fallback_risk 0 | 1.0002 | 80.13% |
| fallback_risk 8 | 1.0000 | 79.87% |
| fallback_risk 20 | 1.0000 | 79.87% |
| upward_risk 0 | 1.0007 | 78.97% |
| upward_risk 25 | 1.0000 | 79.87% |

Whatever they were worth when 8p measured them, the pipeline has moved and they
are now doing nothing. Not removed - they cost nothing either - but do not spend
time on them.

### Three different adjacency defaults, and the shipped one was the worst

`reconstruct_scene_surfaces` defaulted to **0.0**, `benchmark_templates.py` passed
**0.55**, and every measurement in this handoff used **1.0**. Anyone calling the
library or the CLI without the flag was getting a configuration nobody had
measured.

| adjacency | relative pixel error | relative material error | exact state |
|---|---|---|---|
| **0.0** (was the library default) | 1.0575 | **1.1663** | **73.72%** |
| 0.40 | 1.0218 | 1.0509 | 84.87% |
| **0.55** (benchmark script) | 1.0211 | 1.0498 | **85.64%** |
| 0.60 | 1.0392 | 1.0905 | 78.85% |
| 1.0 (every measurement here) | **1.0000** | **1.0000** | 79.87% |

Off is the worst value on **both** metrics. `ADJACENCY_STRENGTH = 0.55` is now the
library and CLI default, worth **+11.9pp of exact state** against the old default
and **+5.8pp** against the 1.0 this handoff has been quoting. Template 6 is
insensitive (33.41-33.74% across the range), so it costs nothing there.

The landscape is sharp and not smooth - 0.55 gives 85.64% and 0.60 gives 78.85% -
so this is a measured operating point, not a robust optimum. **Re-sweep it after
any change to the palette or the classifier.**

### The two metrics genuinely disagree, and that is worth understanding

Pixel error is minimised at adjacency 1.0; exact state is maximised at 0.55. The
likely mechanism: smoothing makes a large flat wall uniform, so if its dominant
material is right the pixel error is low, while less smoothing lets more small
cells keep their own correct evidence at the cost of speckling the wall.

Neither metric is strictly superior. Pixel error weights a mistake by how much of
the image it occupies; exact state weights every block equally, which is what
someone rebuilding the schematic by hand would care about. **Report both.** The
choice made here favours exact state, because that is the project's stated gate
(6, step 13) - but the disagreement is real and a future decision could reasonably
go the other way.

## 47. The partial-block chain: proposal is the blocker, and the residual can find them (2026-09-06)

45 put shape selection next, on the grounds that partial blocks are 84.1% of
Template 2's visible error. Measured, that was the wrong target.

### Nothing is placed at those cells at all

Of the 69 partial cells in Template 2, **61 are air in the output**. Where are
they relative to what we do place?

| | |
|---|---|
| the cell *below* is placed (support present, thin block absent) | 30 (49%) |
| some horizontal neighbour is placed | 29 |
| **no neighbour placed at all** | 20 (33%) |

Of the 30 with a support: 19 `oak_stairs`, 5 `gray_carpet`, 5 `oak_slab`, 1
`white_carpet`. Running the **complete palette** with a shape cost places
**nothing at any of the 30** - the classifier picks `birch_planks` (22) and
`dark_oak_log` (5) at the *support* cell and never gets a candidate at the cell
above.

**So shape selection cannot be the blocker.** No palette entry, shape cost or
selection rule can name a block in a cell that no stage proposes. 39 said this and
this measures it exactly.

Stereo (42) is not the answer either: at margin 0.15 it proposes 8 of the 69, at
0.25 five, at 0.35 one.

### Why depth reasoning misses them

A carpet occupies the cell *above* the floor but is a sixteenth of a block tall,
so its top surface sits at essentially the same plane the floor's would. There is
no depth signal to find. Only the **texture** of that top face distinguishes a
carpeted floor from a bare one - which is exactly the evidence the classifier
already has and cannot act on, because it is being asked about the wrong cell.

### The residual, which needs no reference

44's attribution reads the reference litematic, so it can only ever be a
diagnostic. The same loop *without* the reference is a runtime signal: render the
reconstruction from a known pose, compare against the actual photograph, and
wherever they disagree something along that ray is wrong. A carpet we do not have
is a patch of floor rendered in the wrong colour.

`scratchpad/residual.py`. The renderer is flat and the photograph is lit, so
absolute difference is meaningless; what survives is each cell's disagreement
relative to the frame's median. Ranking Template 2's 645 rendered cells by it,
against the 52 placed cells that neighbour a missing partial:

| top N by residual | precision | recall |
|---|---|---|
| 25 | **36.0%** | 17.3% |
| 50 | 36.0% | 34.6% |
| 100 | 34.0% | **65.4%** |
| 200 | 20.0% | 76.9% |
| *random* | *8.1%* | - |

**Four and a half times chance**, and two thirds of them inside the top hundred
cells. That is a usable proposal source for exactly the error that dominates
Template 2, and unlike everything in 40 and 41 it does not depend on cross-view
agreement, so Minecraft's repetitiveness does not defeat it.

### Next

Close the loop: rank cells by residual, propose the neighbouring empty cell,
classify it against the complete palette with the shape cost, keep it if it lowers
the residual. That is a hill-climb on a signal available at runtime, and the first
thing in this project that attacks *missing* geometry rather than material.

## 48. The residual loop, and the renderer flaw that blocks it (2026-09-06)

47's plan: propose from the residual, classify, keep what lowers it. Built, and it
does not work yet - for a reason worth having found.

### Proposing works

Ranking Template 2's rendered faces by disagreement with the photograph and taking
the empty cell in front of each:

| proposals | are missing partials | are any reference block | partial recall |
|---|---|---|---|
| 40 | **37.5%** | 42.5% | 24.6% |
| 80 | 30.0% | 37.5% | 39.3% |
| 160 | 20.6% | 25.6% | 54.1% |

Against stereo's 8 of 69 and a proposal pass that reached these cells not at all.

### Injecting them all makes things worse

| config | rel error | state | geometry F1 | phantom |
|---|---|---|---|---|
| shipped | 1.0000 | **79.87%** | **94.43%** | 6% |
| canonical + residual 40 | 0.9938 | 78.81% | 93.74% | **26%** |
| canonical + residual 80 | 1.1701 | 78.12% | 92.38% | **35%** |
| complete + residual 80 | 1.2077 | 46.38% | 92.38% | 32% |

At 37% precision the other 63% stand *in front of* surfaces that were already
right, and phantom error goes from 6% to 26-35%. An accept test is not optional.

### A trap in the metric

`rel material` fell to **0.2852** on the worst configuration. That is not an
improvement: a phantom block covers a face that was previously scored as a
material error, so the pixels move category rather than improve. **Material energy
is only comparable between runs whose geometry is the same.** Use total error, or
compare causes, when geometry changes.

### Why the accept test fails: the renderer draws every block as a cube

`keep_helpful` renders the trial volume, keeps proposals whose own faces now agree
with the photograph, and measured **0 of 80 kept at a 0.6 ceiling, 5 at 1.0, none
of them real blocks.** Anti-correlated, not merely weak.

`errshot.render` draws every block as a **full cube** from its face colours. A
correctly placed carpet is drawn as a solid cube filling its cell, looks nothing
like the photograph, scores a high residual and is rejected. The filter throws away
precisely the right answers.

**Shape-accurate rendering is a prerequisite for any photo-versus-render loop.**
`PaletteColor.shape_boxes` and `TextureSample.shape_boxes` already carry the
geometry (39); the renderer needs to project each box's faces instead of the unit
cube.

### What this qualifies

44's attribution shares the renderer, so its magnitudes are inflated wherever thin
blocks are involved - a stair or a fence drawn as a full cube covers far more of
the frame than it should. Carpets are the exception and the largest term: a
carpet's top face *is* a full cell face, so viewed from above the cube rendering is
accurate. The 84.1% figure is therefore roughly right for the carpet share and
overstated for stairs and fences. The direction of 44's conclusion stands; the
exact split should be re-measured once the renderer is shape-aware.

## 49. Shape-accurate rendering, a coverage bug, and what still blocks partials (2026-09-06)

48 named shape-accurate rendering as the prerequisite. Built, and it exposed a
second bug in the shape cost written earlier the same day.

### The renderer draws boxes now

`errshot.render` projects each model box's faces instead of the unit cube, with
`shape_table` supplying `shape_boxes` per state. A face is only cullable by a
neighbour when it lies on the cell wall, so a slab's top face at y=0.5 stays
visible. 141,866 of 429,184 pixels change on one Template 2 frame.

44's numbers, re-measured honestly:

| fixture | partial share, cube render | partial share, box render |
|---|---|---|
| templates 2 | 84.1% | **76.3%** |
| templates 6 | 4.4% | **2.4%** |

Causes on Template 2 redistribute from missing 45%/material 44%/phantom 6% to
**material 36% / missing 26% / phantom 23%** - more honest, because where the
reference holds a thin block and we hold a full cube we really are placing too
much material. The direction of 44 stands: partial blocks dominate Template 2 and
are irrelevant on Template 6.

### `slot_coverage` measured the wrong thing

It asked whether a shape *touches the cell wall*. A carpet's top surface sits at
y=0.0625 and never touches the ceiling of its cell, so it scored **0.0 on `up`** -
the one face a camera ever sees it through - and paid the maximum shape penalty
there. A carpet could not win a cell under any weight, which is why residual
proposals that should have become carpets came back as `birch_planks` (34 of 80).

The right quantity is the **projected footprint** along the face's axis: how much
of that face the shape hides.

| shape | up / down | sides |
|---|---|---|
| carpet | **1.0** (was 0.0) | 0.0625 |
| bottom slab | **1.0** (was 0.0) | 0.5 |
| fence post | 0.0625 | **0.25** (was 0.0) |
| torch | 0.0156 | 0.0781 |

Coverage now depends on the axis, not the direction, which is correct: a shape
hides the same cross section whichever way you look along it. It deliberately no
longer separates a bottom slab from a full block seen from directly overhead -
they *do* hide the same area - leaving that call to the side faces and to depth.

**This is neutral for the shipped configuration**, whose canonical palette is all
full cubes, and correct for anything using the complete one. Tests rewritten;
185 pass.

### The complete palette is still not usable, and now for the opposite reason

| config | rel error | state | dominant cause |
|---|---|---|---|
| shipped (canonical) | 1.0000 | **85.64%** | material 36% |
| complete, shape_w 2 | 1.4804 | 23.08% | **silhouette 43%** |
| complete, shape_w 6 | 2.0814 | 2.69% | **silhouette 66%** |
| complete, shape_w 24 | 1.2770 | 51.67% | missing 34% |

Before the fix, partials were suppressed by an unearned penalty and never placed.
After it they are placed constantly and wrongly, and **silhouette error** - one
render hitting where the other does not - runs to two thirds of the total, because
a scene full of mistaken thin blocks is full of holes to see through.

Both failures have the same root: nothing chooses *which* shape. 8p step 6 parked
shape selection at 36.8% top-1 against a 40% floor, and with 5,366 partial
candidates colour and footprint alone cannot pick between them. **The complete
palette is blocked on shape selection, not on the palette or the cost.** That is
now measured from both sides.

## 50. The error instrument is now part of the package (2026-09-06)

44 said this should not stay in the scratchpad. `src/mc_photogrammetry/export/
attribution.py`, eleven tests in `tests/test_attribution.py`.

```python
from mc_photogrammetry.export.attribution import attribute_error

report = attribute_error(reference.blocks, candidate.blocks, scene, cameras,
                         frames, palette)
report.share()            # cause -> fraction of visible error
report.worst_blocks(5)    # the blocks costing the most
report.worst_confusions() # (wanted, got, share)
```

It builds its colour and shape tables from the **palette the run already holds**,
so it needs no texture provider - `PaletteColor` carries `face_map` and
`shape_boxes` since 39. Rendering is box-accurate (49).

Reproduced against the scratchpad version on real fixtures:

| fixture | partial share | causes | worst block |
|---|---|---|---|
| templates 2 | **75.7%** | material 35%, missing 27%, phantom 23%, displaced 11% | white_carpet 39.4% |
| templates 6 | **2.4%** | material 81%, phantom 13%, missing 5% | white_concrete 28.4% |

The tests pin the things that were got wrong along the way: that a block we never
placed reads as **missing** and not as material, that one we invented reads as
**phantom**, that blame names the *reference* block rather than ours, and that a
carpet is drawn as a carpet - the cube rendering that made the residual loop reject
its own correct answers is now a regression test.

## 51. The fallback cells: three dead ends, and a metric that was flattering us (2026-09-06)

The centre-patch fallback reaches nearly every cell in the region - 1293 of
Template 2's 1833, 6151 of Template 6's 7725 - and 37 showed it supplies 76-93% of
all evidence. Three ways of dealing with it, all measured, none shipped.

### Dropping them entirely: the rate rises, the answer gets worse

`require_resolved_face=True` places a block only where a face was really resolved.

| fixture | config | cells | **correct** | state given placed | state of all visible | geometry F1 |
|---|---|---|---|---|---|---|
| templates 2 | shipped | 814 | **668** | 85.64% | **79.71%** | 94.43% |
| templates 2 | resolved only | 540 | **455** | **89.39%** | 54.30% | 73.88% |
| templates 6 | shipped | 2365 | **603** | 33.35% | **30.20%** | 82.90% |
| templates 6 | resolved only | 1574 | **375** | 27.72% | 18.78% | 75.78% |

It drops 274 cells on Template 2 to shed 61 wrong ones. The rate rises because the
survivors are the easy cells; the reconstruction loses 213 correct blocks.

**This is a warning about the metric, not just the option.** "Conditional state
accuracy" - correct as a share of the cells we placed - is what this document has
quoted since 8b, and it is survivorship-prone: any change that discards hard cells
flatters it. `state|all`, against every visible reference block, is the honest one
and moves the opposite way here. **Report both whenever cell counts change.**

So the fallback cells carry the *geometry* (F1 94.43% against 73.88%) while
labelling it badly. Keep them and distrust their labels - which is what
`fallback_risk` was for.

### `fallback_risk` is inert, and the reason is structural

Swept 0 to 40, it changes nothing (46). The shares are populated - mean 0.705 on
Template 2, boost reaching 40.0 - so it is computed and passed correctly.

The mechanism: `_smooth_adjacent_states` scores `total = score + coupling *
neighbour_votes` **over the cell's own top-K candidates only**. Coupling can
re-rank what a cell already considered; it cannot introduce anything. A fallback
cell whose evidence never proposed the right answer is unreachable however hard its
neighbours push.

### Letting them adopt a neighbour's label: also worth nothing

`ADOPT_BOOST` admits the neighbours' majority as a candidate in its own right when
a cell's risk is high enough. Template 2 is unchanged to four decimals; Template 6
gains **0.22pp** at best. So the majority label was already in the cell's candidate
list nearly always - the top-K restriction was a real mechanism but not the binding
constraint.

Ships **off** at 0.0, with a test pinning the behaviour, because it is a genuine
capability that may matter once the evidence improves.

### What this leaves

The fallback cells' labels are not fixable by smoothing, from either direction.
They are wrong because the pixel under a cell's centre is not a picture of that
cell - and no amount of neighbourhood reasoning recovers information that was never
sampled. The way out is to stop needing them: 35's free-space carve, so cells that
are not surface stop demanding an answer at all.

## 52. The cast correction, re-judged by what it looks like (2026-09-06)

34 built the per-channel cast correction and 37 shipped it **off**, because exact
state said it was neutral to slightly negative. That judgement was made before the
error instrument existed. Template 6 is 81% material error (50), which is exactly
what this correction attacks, so it was worth re-asking.

| config | visible error | material error | exact state | geometry F1 |
|---|---|---|---|---|
| shipped, cast off | 1.0000 | 1.0000 | **33.35%** | 82.90% |
| cast, damping 1.0 | 0.6768 | 0.5840 | 32.96% | 82.90% |
| **cast, damping 0.8** | **0.6273** | **0.5431** | 33.13% | 82.90% |
| cast, damping 0.6 | 0.7745 | 0.7296 | 32.96% | 82.90% |

**A 37% cut in visible error and a 46% cut in material error**, with geometry
untouched and exact state within 0.2pp. Template 2 is bit-for-bit unchanged - the
spread gate correctly declines on a fullbright scene. `LIGHT_CAST_DAMPING` was
added because the magnitude estimator over-corrects (0.262 recovered against a true
0.312 on Template 5); 0.8 is the measured optimum.

### And it still ships off, because of *how* it improves

The confusions before and after:

| | worst confusions |
|---|---|
| off | white_concrete -> **soul_soil**, prismarine -> **gray_wool**, pink_glazed_terracotta -> **netherrack** |
| on | prismarine -> **pale_oak_log**, pink_glazed_terracotta -> **pale_oak_log**, deepslate_iron_ore -> **pale_oak_log** |

Before, bright materials read as dark ones. After, they read as one *pale* one.
The visible error falls because pale is closer to white concrete than soul soil is
- not because any block became right. It is the same collapse-to-a-single-material
that 37 found on Template 5, now in the bright direction.

So this is a **cosmetic** improvement: the reconstruction looks a third closer and
is no more correct. Left off, because the deliverable is a schematic someone
rebuilds block by block, and a wrong block is wrong however near its colour.

**This is a decision the user should make, not one to settle by default.** If the
goal is a reconstruction that reads correctly at a glance, turn it on:
`correct_light_cast=True, light_cast_damping=0.8`. If the goal is correct blocks,
leave it off. The numbers for both readings are above.

## 53. Attribution on the command line, and a CLI that had been broken (2026-09-06)

```
python -m mc_photogrammetry compare <reference> <candidate>     --cameras <cameras.json> --scene <scene.json>     --attribute --minecraft-jar <version.jar>
```

prints the surface scores and then:

```
Visible error by cause:
  material     35.5%  (123376 px)
  missing      26.6%  (112973 px)
  phantom      23.1%  (86344 px)
  displaced    11.1%  (71844 px)
  silhouette    3.8%  (31196 px)
Worst blocks:
   39.4%  minecraft:white_carpet
   23.9%  minecraft:oak_stairs
   15.0%  minecraft:dark_oak_planks
Worst confusions:
   38.2%  minecraft:white_carpet -> minecraft:dark_oak_planks
   14.8%  minecraft:oak_stairs -> minecraft:birch_planks
```

The candidate is shifted by the comparison's best translation first, so the
attribution and the coordinate scores describe the same alignment.

### The CLI had been raising `NameError` on every command

46 made `ADJACENCY_STRENGTH` the CLI default behind a guard that checked whether
*any* import from `raycast` existed. One did, so the import of the constant was
silently skipped and `build_parser` raised `NameError` before parsing anything.
Every test passed throughout, because they exercise the command bodies and never
the parser.

Fixed, and `tests/test_cli.py` now builds the parser, checks the subcommands are
reachable, and asserts the reconstruction default matches the measured constant.
Three tests that would have caught it the moment it was introduced.

**The lesson generalises:** a guard of the form "skip if something like this is
already present" is not a safe way to add an import, and a parser that no test
constructs is a parser that can be broken silently.

## 54. Template 5 diagnosed exactly, and moved off zero (2026-09-07)

Template 5 has read 0.00% exact state since it was first run. On its diamond crop
(992 cells, 18 cameras, two minutes), precisely why:

| | |
|---|---|
| `diamond_block` in the palette | **yes** |
| its jar colour | (98, 237, 228) - bright cyan |
| mean observed colour of its faces | **(49.5, 61.7, 46.5)** |
| its rank among candidates | **median 640** of ~726 |
| its top-1 rate | **1.5%** over 4616 faces |

Four times too dark *and the wrong hue*: observed green/blue is 1.33 where diamond's
is 1.04, because blue is attenuated hardest - exactly the cast of 34. Undoing that
cast at magnitude 0.31 turns (49.5, 61.7, 46.5) into roughly (129, 199, 197), which
has diamond's shape.

### Forcing the gate takes it from 0.00% to 24.33%

| config | state of placed | state of all visible | picks |
|---|---|---|---|
| shipped | **0.00%** | 0.00% | warped_stem 398, brown_stained_glass 184 |
| **cast forced, damping 1.0** | **24.33%** | **22.29%** | podzol 258, **diamond_block 210** |
| cast forced, damping 0.8 | 0.00% | 0.00% | mossy_cobblestone 271 |
| cast forced, damping 1.2 | 0.00% | 0.00% | pale_oak_log 849 |

`diamond_block` appears for the first time. The correction is **razor-sharp**:
either side of damping 1.0 collapses back to nothing, over-correction into
`pale_oak_log` and under-correction into `mossy_cobblestone`. That fragility is
itself a finding - the magnitude has to be right to a few percent.

### Local magnitude estimation: the obvious idea, and it is wrong

One end of Template 5 is lit and the other is not, so estimating each colour's
unlit brightness per region of the scene looks obviously better than one global
ceiling. Measured, it takes the crop from **24.33% back to 0.00%** at 16 blocks and
at 8. Fewer faces per group means a lower maximum and a correction that
under-shoots. Reverted; the ceiling stays global.

### The open question

The crop succeeds with the cast and the **full fixture fails** (37: it fired at
spread 3.004 and still returned 0.00%, collapsing to `pale_oak_log`). Locality is
now excluded as the explanation. What remains is that the ceiling is a **maximum**,
and a maximum is not scale-invariant: over the crop it is taken across 4,281 faces
and over the full fixture across 92,762, so the full fixture's is systematically
higher and over-corrects - which is exactly the `pale_oak_log` collapse.

**The next thing to try is a high percentile instead of the maximum**, which is
scale-stable. 34 measured p95 and p99 as worse, but only on whole fixtures, never
against a max taken over a comparable sample size.

### The gate reads differently depending on which faces it sees

Measured on shell faces, the spread statistic separates every fixture correctly:

| scene | spread | gate at 1.9 |
|---|---|---|
| templates 2 | **1.483** | declines, correctly |
| templates 5 crop (purpur) | 2.569 | fires |
| templates 5 crop (diamond) | 2.792 | fires |
| templates 6 | 2.934 | fires |
| templates 3 | 3.160 | fires |
| templates 4 | 3.393 | fires |

But the same crop reports **1.775** inside the pipeline, because there the
statistic is computed over a different face set. The gate is sound; what it is fed
is not consistent.

## 55. The ceiling was a maximum, and that was the whole problem (2026-09-07)

54 left one candidate explanation for why the cast correction rescues Template 5's
crop and fails on the whole fixture: the unlit-appearance ceiling is a **maximum**,
and a maximum is not scale-invariant. Over the crop it is taken across 4,281 faces
and over the full fixture across 92,762, so the larger scene's is systematically
higher, over-corrects, and collapses everything into `pale_oak_log`.

Replacing it with a percentile, on the crop that reads **0.00%** untreated:

| percentile | state of placed | state of all visible | `diamond_block` picks |
|---|---|---|---|
| 100 (the maximum) | 24.33% | 22.29% | 210 |
| 99.9 | 23.99% | 21.97% | 207 |
| 99.5 | 28.39% | 26.01% | 245 |
| **99.0** | **48.44%** | **44.37%** | **418** |
| 98.0 | 46.93% | 42.99% | 405 |
| 97.0 | **0.00%** | 0.00% | - |
| 95.0 and below | 0.00% | 0.00% | - |

**Zero to 48.44%**, and the window is narrow: one point of percentile either side
of 99 costs everything. The same sharpness showed in the damping sweep, where 0.8
and 1.2 both collapsed what 1.0 recovered. This correction has to be right to a few
percent, which is worth remembering before treating any of these constants as
robust.

`CAST_PERCENTILE = 99.0` is now the default. It is neutral or better elsewhere:

| fixture | relative visible error | exact state |
|---|---|---|
| templates 6, cast off | 1.0000 | 33.35% |
| templates 6, pct 100 | 0.6768 | 32.96% |
| **templates 6, pct 99** | **0.6650** | **33.19%** |
| templates 6, pct 99 + damping 0.8 | **0.6112** | 33.13% |
| templates 2, pct 99 | 1.0000 | 85.64% (gate declines) |

### What this does and does not settle

It explains 37's failure completely: the correction was right and its calibration
was wrong in a way that got worse the bigger the scene. 34's rejection of p95 and
p99 stands for the fixtures it measured, but it never compared them against a
maximum taken over a *comparable sample size*, which is the comparison that
matters.

`correct_light_cast` still ships **off**, pending the full-fixture run. The crop is
992 cells against 277,160 and the whole point of this section is that the two
scales behave differently.

## 56. Both exporters take a string now (2026-09-07)

`LitematicExporter.export` and `AnvilExporter.export` annotated `destination: Path`
and used `destination.parent`, so a `str` raised `AttributeError` **after** the
volume was built. On Template 5 that meant two and a half hours of reconstruction
thrown away at the final line.

Both now call `Path(destination)` first and both have a test for it. The type
error was mine; the trap was the code's, and it is the worst possible place for
one - a long run has no way to tell you it is about to waste itself.

A second lesson from the same run: a stray background job from an abandoned
experiment was still reconstructing Templates 3 and 5 the whole time, roughly
doubling the wall clock. **Stop background work explicitly when abandoning it** -
saying so in a message does not stop it.

## 57. The percentile does not transfer, and 54 tested locality in the wrong place (2026-09-07)

55 set `CAST_PERCENTILE = 99.0` on the strength of Template 5's crop going from
0.00% to 48.44%, and predicted it would carry to the whole fixture. It does not.

| Template 5, whole fixture, cast on at percentile 99 | |
|---|---|
| geometry F1 | 85.15% |
| exact state | **0.00%** |
| picks | **`pale_oak_log` 22,112**, blue_terracotta 780, purpur_block 140 |

Identical to the failure 37 recorded. Two and a half hours, no movement.

### The scale argument was wrong, and it points back at locality

55's reasoning was that a *maximum* grows with the number of faces it is taken
over. True - but a **percentile does not**. p99 is p99 whatever the sample size, so
if sample count were the whole story, percentile 99 would have transferred. It did
not, which means the crop and the fixture differ in their *distributions*, not
their sizes.

They do, and obviously in hindsight. The crop is a twenty-block box lit from one
place; the whole fixture is 52x82x65 with one end bright and the other dark, and
pooling both into one distribution puts p99 at a genuinely brighter face than any
dim region ever sees. Every face far from a light is then over-corrected, which is
exactly the pale collapse.

**So the crop works because it is local.** 54 tested locality and rejected it - on
the crop, where binning at 16 or 8 blocks can only fragment a region that is
already local into groups too small to hold a bright example. That was the wrong
fixture to test it on, and the conclusion it produced was the opposite of what the
same evidence supports at full scale.

`CAST_REGION` is restored, defaulting to **0** (one estimate for the whole scene,
unchanged behaviour), with the full fixture running at 16 to settle it.

### The honest position

`CAST_PERCENTILE = 99.0` still ships, because it is neutral-to-better everywhere it
has been measured (Template 6 0.6650 against 0.6768 relative error, Template 2
untouched) - but the reason given for it in 55 is wrong, and it has never been
shown to help a fixture at full scale. Treat the crop's 48.44% as a *ceiling
demonstration*: it proves the correction can recover `diamond_block`, not that the
current calibration does so on a real scene.

## 58. Locality also fails at full scale, and what actually decides it (2026-09-07)

57 restored `CAST_REGION` and predicted local estimation would carry Template 5.
It does not.

| Template 5, whole fixture, cast on | percentile 99, global | percentile 99, region 16 |
|---|---|---|
| geometry F1 | 85.15% | 85.15% |
| exact state | **0.00%** | **0.00%** |
| dominant pick | `pale_oak_log` 22,112 | `pale_oak_log` 21,830 |

Both hypotheses for the crop-versus-fixture gap are now refuted by measurement:
sample scale (55) and locality (57). Two runs of about ninety minutes each.

### What does decide it: the support vote

Every Template 5 output contains exactly **10 distinct states** against the
reference's 50, so palette pruning is doing most of the deciding. Testing that on
the crop:

| crop, cast on | exact state | distinct states | `diamond_block` placed |
|---|---|---|---|
| **pruning on** | **48.44%** | 14 | **1,023** |
| pruning off | **0.00%** | 30 | 5, with `pale_oak_log` at 2,295 |

**Pruning is what makes the crop work.** Without it the corrected colours favour
`pale_oak_log` there too - the same collapse as the full fixture. So the fixture's
failure is not that pruning *removed* `diamond_block`; it is that the support vote
picked `pale_oak_log` on 92,762 faces and `diamond_block` on 4,281.

That reframes the whole Template 5 problem. The cast correction leaves diamond and
pale oak close enough that whichever wins the first scoring pass wins the scene,
and the vote goes the wrong way at full scale. **This is not a calibration
problem** - three calibration knobs have now been swept - it is that two candidates
are nearly tied after correction and the tie is broken by a global popularity
contest.

### Where to pick up

Do **not** sweep `CAST_PERCENTILE`, `CAST_REGION` or `LIGHT_CAST_DAMPING` again;
all three are measured, and 8r's warning about fitting exposure by cost still
holds. The question to answer is why `pale_oak_log` beats `diamond_block` on the
corrected full-fixture faces when it loses on the crop's. That is answerable
offline, without a ninety-minute reconstruction: sample the reference geometry,
apply `_uncast`, and compare the two candidates' costs directly at both scales.
34's harness (`scratchpad/cast.py`) already does the sampling.

One structural note worth carrying: the crop reaches 48.44% with **worse** geometry
than the full fixture (49.8% against 85.15%). Geometry quality is not what
separates them.

## 59. Template 5 is an evidence problem, not a lighting problem (2026-09-07)

58 asked why `pale_oak_log` beats `diamond_block` at full scale and loses on a
crop. Answered offline, on reference geometry, with `_uncast` applied exactly as
the pipeline applies it:

| | crop corner | whole fixture |
|---|---|---|
| diamond faces scored | 4,616 | 74,253 |
| `diamond_block` median rank | 639 | **6** |
| `diamond_block` top-1 | 18.8% | **30.7%** |
| `pale_oak_log` median rank | 552 | 551 |
| `pale_oak_log` top-1 | 0.0% | **0.0%** |
| head to head | diamond 4,291 : pale 325 | **diamond 73,904 : pale 349** |
| mean cost | 16.29 / 23.48 | **11.94** / 23.97 |

**The correction works better at full scale, not worse.** Given the right faces,
`diamond_block` ranks sixth of seven hundred and wins the head-to-head against
`pale_oak_log` by two hundred to one. Pale oak is not a close second - it is top-1
on **zero** faces.

So the `pale_oak_log` collapse cannot come from the corrected colours. It comes
from the pipeline sampling **different faces**: 37 measured that 97% of what
reaches the classifier is centre-patch fallback from cells that are not surface,
and a centre pixel is not a picture of its cell. The classifier is not choosing
badly between diamond and pale oak; it is being shown something that is neither.

### What this settles

- **The lighting work on Template 5 is done.** 34's cast, 55's percentile and the
  gate are all sound, and 30.7% top-1 on correct geometry is the proof. Do not
  sweep `CAST_PERCENTILE`, `CAST_REGION` or `LIGHT_CAST_DAMPING` again.
- **Template 5's 0.00% is the same fault as Template 4's 10.37% geometry**: cells
  that are not surface still demand an answer. One fix serves both, and it is 8n
  step 2's free-space carve.
- 58's reading of the pruning result stands but is not the root cause. Pruning
  decides between candidates that are already wrong.

### Cost of getting here

Two full-fixture runs of about ninety minutes each (55's percentile, 57's
locality), both returning 0.00%, to test two hypotheses that a fifteen-minute
offline measurement refuted afterwards. **Measure the classifier on reference
geometry before spending a reconstruction on it** - `scratchpad/cast.py` and
`scratchpad/diamond_vs_pale.py` both do this, and either would have answered it
first.

## 60. Shape selection by rendering the candidate (2026-09-07)

22 scored shapes analytically - each visible face's colour plus a "negative space"
term - and reached 44.4% top-1 over **34** shapes. 23 parked it there. Every
attempt since has been about feeding that same score better evidence.

Shape-accurate rendering (49) allows a different question: draw each candidate into
the scene and compare the pixels it produces against the photograph. Negative space
stops being a term of its own, because the part of a cell a carpet does not fill
simply shows whatever is behind it. `scratchpad/shapefit.py`, measured on Template
2's 69 partial cells against a background of the reference with every partial cell
removed.

| configuration | shapes | median rank | top-1 | right family |
|---|---|---|---|---|
| bounding-box scoring | 797 | 60 | 7.2% | - |
| + silhouette mask | 797 | 64 | 8.7% | - |
| + gain matching | 797 | 50 | 8.7% | - |
| **+ restrict to build families** | **178** | **9** | **30.4%** | **60.9%** |
| + balance painted vs unpainted | 178 | 25 | **15.9%** | 47.8% |

### The three levers that worked

- **Score inside the cell's silhouette, not its bounding box.** A carpet fills a
  sixteenth of its cell, so most of the box is background whichever candidate is
  drawn and every score lands within noise of every other. Before this the winners
  were `fire`, `resin_clump` and `dead_tube_coral_fan` - shapes that draw almost
  nothing. After it, drawing nothing ranks median 557 and wins no cells at all.
- **Match the render's level to the photograph's.** The renderer is flat and the
  photograph is lit, so without a gain every candidate is penalised in proportion
  to how many pixels it paints. Median rank 64 to 50. The same trap `residual.py`
  already solved, not carried over.
- **Restrict candidates to the block families a build contains** - the lever 22
  named and never pulled. The full palette offers 797 partial shapes, hundreds of
  them coral fans, resin clumps and crop stems that no constructed scene holds, and
  each is another chance to beat the right answer by accident. 797 to 178 shapes
  takes top-1 from 8.7% to **30.4%** and the median rank from 50 to 9.

### The lever that measured worse

Weighting the painted and unpainted halves of the silhouette equally is the obvious
correction for the remaining flat-shape bias, and it **halves** the result. For a
large shape the unpainted sliver is a handful of noisy pixels handed half the vote,
and `(empty)` starts winning 13% of cells. Reverted, and commented in place.

### Where it stands

30.4% exact shape over 178 candidates is **54x chance**, against 22's 44.4% over 34
candidates at 15x. The two are not directly comparable and this is the stronger
signal, but 22's absolute number is still higher and 12's ship gate is 60%.

The useful decomposition is that **family accuracy is 60.9%** - carpets are
recognised as carpets and stairs as stairs - and the dominant remaining mistake is
`stairs -> carpet`, nine of twenty-seven. That is the flat-shape bias surviving at
smaller scale: a stair and a carpet agree about the bottom of the cell and the
scorer will not commit to the volume above it. Orientation, which 29 named as the
residual ambiguity, is now the *second* problem rather than the first.

## 61. The stairs-as-carpets bias, and what actually causes it (2026-09-07)

60 left family accuracy at 60.9% with `stairs -> carpet` the dominant mistake,
nine of twenty-seven, and read it as the scorer refusing to commit to volume. Three
mechanisms were argued and tested.

| configuration | median rank | top-1 | right family | stairs->carpet |
|---|---|---|---|---|
| 60's best | 9 | 30.4% | 60.9% | 9 |
| **+ gradient term, weight 0.25** | **7** | **30.4%** | **63.8%** | **8** |
| + gradient term, weight 0.6 | 9 | 27.5% | 59.4% | 9 |
| + local gain per cell window | 10 | 27.5% | **39.1%** | **17** |

**A gradient term helps a little.** Comparing Scharr magnitudes as well as colour
lifts family accuracy 2.9 points and the median rank from 9 to 7, at weight 0.25;
0.6 is too much. Kept.

**A local gain is invalid by construction, and measuring it was worth doing.**
Re-estimating the render-to-photograph gain inside each cell's own window looks
like the obvious fix for a shadowed recess. It takes family accuracy to **39.1%**
and doubles the stairs-as-carpets count. The reason is structural: where the cell
really is occupied, the background render and the photograph differ *because the
block is missing*, so the ratio between them is the error being measured, not a
lighting gain, and applying it amplifies the thing it was meant to remove.

### Looking at the failures, which should have come first

Rendering four failing cells side by side - photograph, true stair, chosen carpet,
empty - showed the cause immediately, and it is none of the three mechanisms
argued from. **The correct stair renders far too pale.** The photograph shows a
mid-brown, well-lit staircase; the flat render paints it light tan, and the
background behind it is nearly black. So the right answer paints a large region in
the wrong brightness and takes a large penalty, while the carpet leaves it dark and
happens to land closer.

The bias is not about area, edges or volume. It is that **the renderer has no
lighting model and the photograph has one**. A candidate is judged on pixels whose
brightness the renderer cannot predict, and the more of them it paints the worse it
does. A global gain cannot fix it because the mismatch is local - an interior
recess renders at full brightness - and a local gain cannot fix it for the reason
above.

### The ceiling, and what would raise it

**The shape scorer is capped by the renderer, not by the shape evidence.** The
project already has the machinery to fix this: `LightField`,
`VANILLA_LIGHT_RESPONSE` and the vanilla `f/(4-3f)` curve. Rendering candidates
through the simulated light field instead of flat colours is the change that would
matter, and it is the same work the lighting thread needs anyway.

Best measured configuration: silhouette masking, global gain, candidates restricted
to build families, gradient weight 0.25 - **30.4% exact shape, 63.8% family, median
rank 7 of 178**. Still under 12's 60% ship gate, still not wired into the pipeline,
and Template 2 remains at 79.71% of visible blocks with **zero** partial blocks
placed.

## 62. Shape selection: ambient occlusion, and searching only the free parameters (2026-09-07)

61 ended with the scorer capped by the renderer having no lighting model, and
predicted the fix was the simulated light field. It was simpler than that. Template
2 is **fullbright**, so block light is not what darkens its recesses - **ambient
occlusion** is, and Minecraft applies that whatever the gamma. `lighting.
ambient_occlusion` has been in the project since 8e.

`render_volume` takes `ambient=True` and shades each face by it.

| configuration | shapes | median rank | top-1 | right family | stairs->carpet |
|---|---|---|---|---|---|
| 61's best | 178 | 7 | 30.4% | 63.8% | 8 |
| **+ ambient occlusion** | 178 | **5** | **31.9%** | **68.1%** | **6** |
| **+ free parameters only** | **30** | **2** | **42.0%** | **73.9%** | **0** |

### Most of a block's state is not a free parameter

A fence's connections, a wall's shape, a stair's inner and outer corners: Minecraft
**derives** all of them from the neighbouring blocks. Searching them anyway offers
twenty-four states for one staircase where two are genuinely in question, and every
extra one is another chance to beat the right answer by accident. Only `facing`,
`half` and a slab's `type` are actually free.

Restricting to canonical states - derived properties at their off value - cuts 178
shapes to **30** and takes top-1 from 31.9% to **42.0%**, median rank 5 to 2. The
`stairs -> carpet` confusion that 60 and 61 chased is **gone entirely**; what
remains is stairs against slabs, both ways, which is a genuine half-height
ambiguity rather than a bias.

**A measurement trap worth recording.** Scoring only the cells whose truth is
already canonical reports **58.7%** - and drops 23 of Template 2's 69 cells,
which are exactly the hard ones. The honest number maps each truth to its canonical
twin, so all 69 are scored, and that is the 42.0% above. The derived properties are
filled in from neighbours after the shape is chosen, so the canonical twin is the
right target.

### Where it stands

**42.0% exact shape, 73.9% family, median rank 2 of 30.** Against 22's 44.4% over
34 shapes this is finally comparable, and it is about the same on top-1 with a
better median. 12's ship gate is 60%.

Still true: this runs on reference geometry with the true material supplied, it
chooses shape alone rather than shape and material jointly, and it is not wired
into the pipeline. Template 2 remains at 79.71% with zero partial blocks placed.

## 63. Shape and material fitted together, and wired in (2026-09-07)

`reconstruction/shape_fit.py`, eleven tests in `tests/test_shape_fit.py`,
`fit_partial_shapes` and `shape_palette` on `reconstruct_scene_surfaces`.

### Material is solved, not searched

Rendering every canonical entry is 676 renders a cell. Narrowing the materials by
the window's mean colour first looks obvious and **discards the truth 81% of the
time** - the window of an empty cell is dominated by whatever stands behind it, so
its mean colour is not the material's.

A shape decides which pixels belong to which *face slot*, so once it is drawn the
best material for it follows from those pixels directly. One render per shape (30),
then every material scored from the cached masks.

Joint choice on Template 2's 69 partial cells, no material given:

| | |
|---|---|
| right family | **69.6%** |
| right block name | **26.1%** |
| exact state | **20.3%** |

Against 62's 42.0% with the material handed over. Material is now the weak half.

### End to end, and the first partial blocks the project has placed

Residual proposals (47) feeding the fit, on Template 2:

| configuration | geometry F1 | state of visible | correct blocks | partials placed |
|---|---|---|---|---|
| baseline | **94.43%** | 79.71% | 668 | **0** |
| + 40 proposals + shape fit | 93.74% | **80.79%** | **677** | **63** |
| + 60 proposals + shape fit | 93.11% | 80.07% | 671 | 83 |

**+1.08pp and nine more correct blocks**, for 0.69pp of geometry precision. Forty
proposals is the operating point; sixty is worse.

### The mistake that nearly buried it

The first integration ran the *whole scene* on the complete palette so partial
states would be reachable, and measured **2.51%** - state accuracy destroyed, 832
partials placed for 60 proposals. That is 49's collapse exactly: 5,366 partial
candidates offered to every cell in the scene.

`shape_palette` separates them. The scene classifies on the canonical palette as it
always did; only the proposed cells see the complete one, inside the fit. Nothing
else changes.

### Where it stands

`FIT_PARTIAL_SHAPES` ships **off**, because it needs `extra_proposals` and the
residual proposal pass is still in the scratchpad. Wiring that in is what would
make the +1.08pp automatic.

Honest ceiling: the fit is right about the family 70% of the time and about the
exact block a quarter of the time, so most of the 63 blocks it places are wrong -
they are wrong in cells that were previously *empty*, which is why the score rises
anyway. Material selection inside the fit is the next thing worth attacking.

## 64. The thin-block chain, automatic in one pass (2026-09-07)

`reconstruction/residual.py` productionises 47's proposal pass, and
`fit_partial_shapes` now runs the whole chain by itself: render what will ship,
propose the empty cell in front of every disagreeing face, fit a shape and material
to each. Fifteen tests across `test_shape_fit.py`.

| Template 2 | geometry F1 | state of visible | correct | partials | runtime |
|---|---|---|---|---|---|
| shipped | **94.43%** | 79.71% | 668 | 0 | **19s** |
| automatic, limit 10 | 94.22% | 79.95% | 670 | 10 | 34s |
| **automatic, limit 20** | 94.14% | **80.07%** | **671** | 20 | 49s |
| automatic, limit 40 | 93.62% | 80.07% | 671 | 40 | 82s |
| automatic, limit 80 | 92.38% | 80.07% | 671 | 80 | 133s |

**The gain saturates at twenty proposals and the cost does not.** Forty and eighty
reach the same 80.07% while geometry falls from 94.14% to 92.38% and the run goes
from 49 to 133 seconds. `PROPOSAL_LIMIT = 20`.

### Two things had to be right, and both were wrong first

**Where the residual runs.** Placing it mid-pipeline, against `merged`, proposes
against the volume *before* pruning - which is filled solid, so the render is a box
and the residual describes that box. It found **three** cells where the finished
surface offers forty. The chain now runs at the very end, on the pruned volume that
will actually ship.

**What the owner buffer carries.** `render_volume`'s index held coordinates only,
and a proposal needs the face's normal to know which way to point. It now carries
`(cell, slot)`; `attribute_error` takes `[0]` and its numbers are unchanged
(35.5% / 26.6% / 23.1%, `white_carpet` 39.4%).

### Automatic is worth less than the two-pass version, and why

63 measured +1.08pp by passing the proposals in as `extra_proposals`; this
self-contained version gets **+0.36pp**. The difference is that `extra_proposals`
also makes them *sampling* targets, so the second sampling pass runs with them in
its occupancy - and 51 measured that pass as load-bearing for reasons still not
understood. Recovering the rest would mean reconstructing twice.

`FIT_PARTIAL_SHAPES` ships **off**: +0.36pp for 2.6x the runtime and 0.29pp of
geometry is not a default, but it is now a single flag rather than a scratchpad
pipeline.

## 65 ✔️ - the shape fit reads grain, and the two questions are separated

63 built the fit and it answered the shape well and the material badly: 69.6% right
family, 26.1% right block, 20.3% exact state on Template 2's 69 real partial cells.
The cause was visible in the code rather than the numbers. The fit painted each
candidate face with **one mean colour** from `face_map` and diffed it against
textured photograph pixels. A flat tan rectangle is as good a match for birch as for
oak, so the fit was never asked the question it was getting wrong.

The classifier already solves that question. What it needs is a rectified face, and
a *candidate shape supplies one*: the quad of each of its own boxes' faces is
exactly what the camera would have seen had the candidate been there. So
`_face_patches` projects each candidate face, rectifies the photograph through it,
and hands the patch to the same `PaletteIndex` with the same `_normalised_pattern`
correlation the main classifier uses. One `costs()` call per face scores every
material at once, so the cost is a few seconds, not a factor.

### The two verdicts must not be summed

Adding the texture cost into the render score improved the material and **cost 2.9pp
of family** - 69.6% -> 66.7% - because the grain then voted on the shape as well.
Silhouette and grain answer different questions and the fit now asks them
separately: the shape is chosen by the rendered difference alone, and only then the
material by grain within that shape. That recovered the family in full.

| | family | right block | exact state |
|---|---|---|---|
| 63, mean colour only | 69.6% | 26.1% | 20.3% |
| summed with texture | 66.7% | 39.1% | 30.4% |
| **split, shipped** | **69.6%** | **43.5%** | **33.3%** |

### The render term is not weak evidence about material, it is bad evidence

Sweeping `TEXTURE_WEIGHT` under the split, the right block name rose the entire way
and never turned over: 26.1% at 0, 31.9% at 0.3, 37.7% at 1, 40.6% at 4, 42.0% at
16, 43.5% at 64, and 43.5% at 1000. The optimum is at the limit, which means the
best thing the rendered difference can contribute to a material decision is
*nothing*. `TEXTURE_WEIGHT = 64.0` leaves it as the fallback for cells where no face
projects large enough to read (`MINIMUM_FACE_AREA = 12` pixels).

### The bottleneck moved to the proposals, and that moved the limit

End to end the gain was only one block, 671 -> 672, which is the useful part of the
result. Of 20 proposals, 9 land on cells the reference actually fills and 8 of those
are partial; the fit takes 4 of the 8. The ceiling at limit 20 was 8 blocks, not 69.

Because each proposal is now worth about twice what it was, the limit 63 chose no
longer holds. Against 668 correct and 93.64% F1 with the chain off:

| limit | correct | geometry F1 |
|---|---|---|
| 20 | 672 | 93.59% |
| **40** | **672** | **93.67%** |
| 80 | 675 | 92.90% |

Forty is the only setting that beats the baseline on *both* counts - the extra
twenty proposals correct as much geometry as they cost - so `PROPOSAL_LIMIT = 40`
and the old trade-off framing no longer applies. Eighty buys three more blocks for
three quarters of a point of F1 and nearly twice the runtime; it is available.

`FIT_PARTIAL_SHAPES` still ships **off**. At limit 40 it is now strictly better on
both metrics, but it is +4 blocks in 852 for 4x the runtime (21s -> 87s on Template
2), and runtime is the thing standing between this program and Template 1.

### What this says about the next step

The fit is no longer the weak link; the residual's 45% precision and its reach of
8 partial cells out of 69 is. Raising proposal recall - which is the free-space
carve by another name - is what the remaining 61 cells need.

*Superseded by 67:* the carve is withdrawn as the recommended route, because its
power scales with the number of views and this project has few. The remaining 61
cells are still the target; the proposed means is now lattice fitting.

218 tests green.

## 66 ✔️ - a theoretical-maximum reference, and the leak it exposed

Every accuracy figure in this project divides by the whole reference. That charges
the reconstruction for blocks buried under furniture, walled into a floor, or never
pointed at - and the fixtures deliberately contain them. `export/visibility.py`
casts the reference itself through the manifest's cameras and keeps the cells at
least one pixel lands on, writing a **theoretical maximum** litematic to
`theoretical_max/`. It is a drop-in reference: same extent, same coordinates, same
states, verified as an exact subset by round trip.

Available as `mc-photogrammetry visibility <reference> <output> --cameras --scene
--minecraft-jar [--min-pixels N]`. The output goes to a separate directory on
purpose - several harnesses take a fixture's reference with
`next(folder.glob("*.litematic"))`, and a second file in the fixture folder would
silently change which litematic they load.

### The instrument had a bug worth more than the instrument

The first version reported **33,243 pixels for a sealed cell** in Template 4. It
was caught by disagreeing with `compare.camera_reachable_air`, which floods air
from the camera cells - a method that shares no code with the raycast and is a
strict superset of what a ray can reach, so anything the raycast finds outside it
is a bug by construction. An independent forward ray march then confirmed the
cells were blocked by four solid blocks each.

The cause: **a face with one corner behind the eye was dropped entirely**, because
`project_points` returns `None` for it. That is not a conservative simplification.
It deletes exactly the near wall that should have occluded everything behind it,
and every fixture here is photographed from inside a room. 4.55% of Template 4's
candidate faces straddle the eye plane, 4.03% of Template 2's, 1.84% of Template
6's. The fix clips the face polygon against the near plane in camera space
(Sutherland-Hodgman) and keeps the visible part.

Two things were *guessed* and measured false, and are recorded because the guesses
were plausible:

* **OpenCV fixed-point overflow was not the cause.** A face a twentieth of a block
  from the eye projects past +/-80,000 pixels, and `fillConvexPoly` fills that
  correctly - measured, filling the whole frame. The clip to the image rectangle
  is kept because it keeps coordinates bounded by the image, not because it fixed
  anything.
* **Seam cracks were not the cause either.** Drawing each face's outline as well
  as its fill, so adjacent quads overlap rather than part, changed the disputed
  count on Template 2, 4 and 6 by exactly zero. It was removed rather than kept
  as insurance, because it widens every face by a pixel and would inflate the
  denominator it exists to make honest.

### Where it still disagrees with the flood fill, the raycast is right

After the fix, 3 disputed cells on Template 2, 14 on Template 4, 45 on Template 6.
All but one have a **partial block** as a neighbour. The flood fill asks whether a
neighbouring *cell* is air; a torch, a bud or a fence occupies its cell while
leaving nearly all of it empty. The amethyst wall behind a `soul_wall_torch` whose
cell is open to air on three sides is plainly visible, and the flood fill calls it
buried. One Template 4 cell at 5 pixels remains unexplained and is left standing
rather than tuned away.

### What the six fixtures actually expose

| template | blocks | cameras | reachable | share | cast |
|---|---|---|---|---|---|
| 1 (`templates`) | 566,680 | 9 | 14,191 | **2.50%** | 470s |
| 2 | 852 | 3 | 669 | 78.52% | 1.1s |
| 3 | 66,414 | 47 | 16,356 | 24.63% | 224s |
| 4 | 6,225 | 7 | 570 | **9.16%** | 2.9s |
| 5 | 31,862 | 39 | 29,406 | **92.29%** | 108s |
| 6 | 2,376 | 4 | 2,031 | 85.48% | 2.1s |

Three of these change how a result should be read:

* **Template 4 is 9% visible.** It is an amethyst geode photographed from inside
  the hollow, so 91% of its block count is the rock mass around the outside. Every
  Template 4 percentage ever quoted has divided by a number eleven times larger
  than the evidence supports.
* **Template 1 is 2.5% visible** from its nine cameras. Scoring a 566,680-block
  reference on nine photographs was never going to mean anything.
* **Template 5 is 92% visible.** Its poor results are confirmed *not* to be an
  exposure or lighting problem, which is what 51 and 55 concluded from the other
  direction. The evidence is there and the classifier is not taking it.

### The maximum is a ceiling on evidence, not a target

Scoring only against it would be a different mistake. On Template 2 the shipped run
gets 668 blocks right, of which **535 are reachable and 133 are not**: cells no
camera can see that the adjacency prior fills correctly because 95% of them sit in
a run of three or more identical neighbours. Those are real wins and a
maximum-only denominator throws them away. The honest pair is both numbers - what
the run recovers from evidence (535 of 669) and what it recovers by inference
beyond it (133) - and the gap that better inference could close is
`reachable - correct`, 134 blocks on Template 2.

## 67 - change of direction: the lattice, not the carve

This section supersedes the recommended next move in 8n step 2 and in 65. It comes
from the user, in two corrections, and both are scope facts rather than
measurements - so nothing below is a result, and the experiment that would make it
one is named at the end.

### The camera manifests are a test scaffold, not part of the design

Every fixture ships a `cameras.json` authored from the F3 overlay, and it is easy
to describe this program as one that reconstructs from *calibrated* views -
`reconstruct_from_calibrated_views` refuses to run without poses, which encourages
the reading. That reading is wrong, and the user was explicit about it.

The manifests exist only to make developing the algorithm easier. In production
there is almost never F3 data, because **the person running the reconstruction is
not the person who took the screenshots**. And the argument that settles it: anyone
with access to the world to take screenshots with the debug overlay open could
export a litematic of the build directly, which makes the entire program pointless.

So recovering pose from the images is part of the problem, not a Phase 4 nicety.
Any assessment that treats supplied poses as given is answering a question nobody
asked. This is the same principle as reconstructing from limited data: the input is
a handful of ordinary screenshots, and everything the fixtures add on top -
manifests, scene bounds, known FOV - is convenience the finished program must
derive or do without.

### Space carving was the wrong recommendation

66 closed by pointing at the free-space carve, as 8n step 2 and 65 do. That
recommendation is withdrawn. **Photo-consistency carving's power scales with the
number of views**, and it wants many; the project's binding constraint is that
there are few. Template 2 has three. Carving is not wrong, it is badly matched to
the data this program is required to work from, which is a better explanation of
six measured-out attempts than any of the six individual post-mortems gave.

### What replaces it

The user's proposal, in the terms it maps onto: **piecewise-planar reconstruction
with lattice fitting, verified by analysis-by-synthesis.** Treat a flat wall as one
surface rather than as forty independently classified faces; solve which integer
plane it lies on and how the block grid sits on it; render the hypothesis; compare
against the photograph; adjust the plane where it fails to explain the pixels, and
take those failures as the places the geometry actually changes.

Three things make it a better fit than anything tried so far:

* **It uses the constraint the program currently ignores.** Minecraft is
  axis-aligned and on an integer lattice. Nothing in the pipeline exploits this;
  every face is classified alone, with no notion of a surface larger than a block.
* **The search is small.** "Try a different position or size" sounds
  combinatorial and is not. A plane hypothesis is an axis, an integer coordinate
  and a 2D extent - a handful of integers. It is *not* a search over block
  layouts, which would be `palette^cells`. Materials stay a per-cell question, and
  become an easier one, because a misaligned grid currently smears two blocks into
  one patch and the classifier is then scored on the smear.
* **It works from one view.** That is the property carving does not have and this
  project needs.

The mechanism already exists in miniature. `shape_fit.py` renders candidates into
the scene and keeps whichever explains the pixels; this is the same loop with a
planar region as the unit instead of a cell. `residual.py` already finds where a
render disagrees with a photograph, which is the "difference in geometry" a plane
hypothesis has to stop at.

### The lattice also recovers the pose, which is why it is the keystone

Minecraft cameras never roll, so orientation is two degrees of freedom. Block edges
run along three orthogonal world axes, and their vanishing points give yaw, pitch
and focal length directly - Manhattan-world calibration. The lattice period then
gives scale, because a block is one metre. A litematic's origin is arbitrary, so
translation only has to be relative. Detecting the lattice therefore recovers
essentially all of `cameras.json` from the image alone: the geometry idea and the
pose problem are one problem.

There is already a measurement in the scratchpad pointing the same way.
`edge_align.py` found that **texture correlation has a basin about a tenth of a
degree wide**, which a search steps straight over, while **block seams collect far
more gradient and are dramatically more forgiving**. The robust signal in these
frames is the seams, and the seams are what a lattice fit runs on. Nothing in the
project has yet tried vanishing points or periodicity.

### Honest limits

It serves rooms and structures - Templates 2, 3 and 6 - and serves organic shapes
badly. Template 4 is a geode and Template 5 has terrain; neither offers many large
planes, and both still need occupancy reasoning of some kind. Partial blocks break
strict planarity, though the shape fit handles those once the grid is pinned. And
the scale ambiguity is real: a wall of 16 blocks at distance *d* projects
identically to 32 blocks at 2*d*, so the seam period gives a ratio and not an
absolute. Two things break it - a second view, and Minecraft's own 16x16 texel
grid, which fixes the texel-to-block ratio wherever texels are resolvable.

### The experiment that turns this into a result

**Estimate vanishing points from block seams on one Template 3 frame and compare
the recovered yaw, pitch and FOV against `cameras.json`.** That is the legitimate
use of the manifests - ground truth to develop against - and it is decisive. If the
recovered pose lands within a fraction of a degree, the manifest stops being
load-bearing and the geometry work gets its foundation in the same stroke. Not yet
run.

## 9. Open considerations

1. Palette scope for the 80% gate: Option A (recommended) canonical placeable full-block palette (~150–300 classes) first; Option B full 6,175-state palette with shape gating for stairs/slabs afterwards.
2. Template 2 is interior and evenly lit; the exterior fixture will expose face-shading/shadow gaps — plan for the shading constants in step 14.
3. pycolmap Windows availability for Python 3.10 decides between the Python binding and the CLI binary at step 15.

---

## 10. Current execution plan — exact-state improvement after Phase A (2026-09-05)

### Baseline to freeze

The current workspace contains a later Phase-A implementation than the original Phase 0-2 plan. Do not use the old `46.70%` result as the active plateau. The strongest archived reproducible report is `exports/t2-ao.json`:

- Conditional exact-state accuracy: `643 / 778 = 82.65%`
- Whole-reference exact-state rate: `643 / 852 = 75.47%`
- Coordinate recall: `91.31%`
- Coordinate precision: `98.36%`
- Coordinate F1: `94.70%`
- Palette size: `636`
- Palette coverage: approximately `91.8%` of Template 2 reference states

The handoff records a later `85.35%` Template 2 result, but no matching JSON report with effective settings and timings is currently present. Reproduce it from the maintained benchmark before treating it as the active gate result.

Every future report must distinguish:

- conditional state accuracy over common coordinates;
- exact-state rate over all reference blocks;
- direct-coordinate versus best-translation metrics;
- coordinate recall, precision, and F1;
- reference, candidate, and common block counts;
- best translation;
- per-state seen/correct/missed counts;
- palette/cache provenance;
- effective classifier settings and phase timings.

### Measured root cause

The remaining exact-state loss is primarily missing partial-model geometry and occlusion, not another global color or exposure parameter.

- Resolved-face blocks: `514`, approximately `91.2%` accurate.
- Centre-patch fallback blocks: `264`, approximately `77.7%` accurate.
- Blocks whose visible faces are covered by unmodelled geometry: `27`, approximately `66.7%` accurate.
- Reference blocks absent from current detected geometry include approximately `20 oak_stairs`, `16 white_carpet`, `13 dark_oak_fence`, `5 oak_slab`, `5 gray_carpet`, and `4 torch` blocks.
- Those partial blocks can cover a lower block's upward face uniformly, producing a clean but wrong sample that `MAX_FACE_VARIATION` cannot reject.

The current `ModelAtlas.bake()` and `canonical_palette()` paths intentionally retain only complete full-cube models. That makes the full-cube classifier reliable, but it also means stairs, carpets, slabs, fences, and torches cannot be used as occluders or classified candidates.

### Implementation order

1. ✔️ **Freeze/reproduce baseline.** Run only `scripts/benchmark_templates.py` from a clean shell. Archive the report with command/configuration, JAR/cache metadata, timing, tests, and both state denominators.
2. ✔️ **Instrument evidence quality.** Add counters to `face_sampling.py` and reconstruction for no front face, clipping/off-screen, minimum area, visible fraction, GUI mask, uniformity rejection, and centre-patch fallback. Report resolved-face/fallback counts and upward-face share.
3. ✔️ **Add arbitrary model geometry.** Extend `ModelAtlas` with an element-level geometry representation: transformed bounds, world-facing polygons, UV corner order, texture/alpha metadata, variant rotations, and multipart composition. Implement slabs, carpets, and stairs first; fences and torches afterward.
4. ✔️ **Persist partial geometry.** Extend `TextureSample` and cache schema with model geometry separately from full-cube face appearance. Do not automatically add partial states to the full-cube palette.
5. ✔️ **Use geometry-aware occlusion.** Replace binary neighbor culling and reduced painter assumptions with projected polygon depth masks and per-pixel visible fractions. Keep the centre-coordinate pass as fallback while validating the new occlusion path.
6. *Parked on its own criterion (23, 29).* **Add shape-gated partial candidates.** Admit a partial state only when its projected silhouette, depth, visible area, cross-frame support, and texture evidence agree. Never admit a partial state merely because its color is close.
7. ✔️ **Revisit adjacency.** Compare no adjacency, the current setting, and bounded candidate-restricted ICM only after partial geometry is working. Do not raise global coupling blindly; isolated log inlays are a known regression risk.
8. ✔️ **Benchmark each isolated milestone.** Keep only results that improve exact state without reducing coordinate F1. Re-run Template 4 after the first geometry milestone.
9. *In progress (30, 33); geometry is sound on every fixture but Template 4 (31, 32).*
   **Investigate residual material/light errors only after geometry.** Global white balance, face-shader constants, generic light-field tuning, face-shrink sweeps, and stronger global adjacency have already been measured and should not be retried without a new falsifiable hypothesis.

### Partial-geometry tests required before promotion

- A slab above a cube occludes only its true projected footprint.
- A carpet covers the lower cube's upward face while leaving side faces available.
- A stair hides only the appropriate portion and respects facing/half/shape rotations.
- A fence or torch occupies a thin depth silhouette and does not erase a whole cube face.
- A partial candidate enters the final decision only when its projected geometry agrees with observations.
- Per-pixel depth ordering beats a farther full cube for overlapping polygons.
- Near-plane and image-edge clipping preserve sufficiently large visible polygons.
- Diagnostic counters show the geometry path reduces fallback and false-visible upward samples.

### Do not repeat without new evidence

- More global exposure or white-balance fitting on Template 2.
- More Minecraft face-shader constants.
- More `FACE_SHRINK` sweeps now that the projection and FOV fixes are in place.
- Stronger global adjacency.
- Name-token wood/dirt/material priors.
- Texture-only tuning before face-to-pixel correspondence and partial occlusion are correct.

### Acceptance and scope

The original Phase 2 gate remains `>=80%` conditional exact-state accuracy with coordinate F1 `>=86%` and no `dark_oak_planks -> dirt` top confusion. The whole-reference exact-state rate must be reported alongside it so missed coordinates cannot disappear from the headline. Template 2 geometry is fixture-calibrated; general footage still requires Phase 4 SfM and grid fitting. `world.zip` remains blocked until the accepted exact-state gate is genuinely met.
