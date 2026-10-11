# DS-8 — Default image-only pipeline audit (read-only)

DeepSeek CLI DS-8, 2026-09-12. Read-only; owns this file only.
Revision: HEAD `c6dba2c` + uncommitted DS-3/DS-4F/DS-5/DS-7 and Luna/M4 edits; no DS-8-claimed path modified.
Method: `read`/`rg` over `cli.py`, `desktop.py`, `ui/main_window.py`, `reconstruction/` (camera, calibration,
sfm, registration, lattice, surfaces, multiview, facade, raycast), `input/`, `domain/models.py`,
`project/store.py`, `pipeline/`, `export/{litematic,validation}.py`, and all 54 test modules; one bounded
argparse+signature probe. No benchmark, suite, source/test edit, staging or workers. "Default" = CLI
`reconstruct` and `desktop`, mode `normal`.

## 1. Chain verdicts

| # | Link | Verdict | Anchor evidence |
|---|---|---|---|
| 1 | Input contract (discover/decode) | **implemented**, 2 defects | `input/media.py:42-81`, `input/image.py:14-29` |
| 2a | Intrinsics (focal/principal) | **implemented, disconnected** | `calibration.py` M3; no src consumer |
| 2b | Orientation (yaw/pitch/roll) | **implemented, disconnected** | `calibration.py:216-269`; `contracts.py:71-95` |
| 2c | Translation | **missing / advanced-only** | `registration.py:90-157` (stated plane links) |
| 2d | Scale | **missing / advanced-only** | `lattice.py:196` (stated length) |
| 3a | Normal reconstruction body | **placeholder** (fabricated lattice grid) | `facade.py:115-119,159` |
| 3b | Normal CLI entry | **disconnected — crashes** | `cli.py:282-285` vs parser `cli.py:47-68` |
| 3c | Metric reconstruction entry | **advanced-only** | `raycast.py:489-492`; `cli.py:310-315,351-356` |
| 3d | One production entry/orchestrator | **missing** | `pipeline/__init__.py` = protocols only |
| 4 | Project persistence/versioning | **schema 1 implemented; schema 2 missing** | `domain/models.py:141-171` |
| 5 | Review/uncertainty surface | **placeholder** | `contracts.py:119-128` unused; `main_window.py:166` |
| 6 | `.litematic` export | **implemented**, 2 provenance gaps | `export/litematic.py:39-151` |

## 2. Detail

### 2.1 Input — implemented; desktop truncates; keyframes/video disconnected
- `discover_media` (`input/media.py:42-81`): stable sort, `--recursive`, extension filter (`:9`),
  exact-sha256 dedupe (`:61-63`); never fills `SourceFrame.width/height`, and `describe_image`
  (`input/image.py:32-44`) has no caller. `load_image` (`input/image.py:14-29`) is RGB+EXIF.
- Disconnected: `iter_video_frames` (`input/video.py:13`) and `select_keyframes`/`perceptual_hash`
  (`input/frames.py:19-72`) are test-only (rg), so folder/video capture is not one pipeline.
- Defect A: normal desktop uses only `image_frames[0]` (`ui/main_window.py:28,73-78`) and records
  only that path (`:78`); extra screenshots vanish silently. Normal CLI takes one image (`cli.py:50`).
- Defect B: no dimensions/quality/`FrameTransform` survive discovery; `FrameObservation`
  (`contracts.py:41-50`) is never constructed in src.

### 2.2 Camera — M3 recovery implemented but unconsumed; translation/scale absent
- `calibration.py:1-30` (Manhattan-world focal/yaw/pitch search); `to_camera_solution:216-269` is
  lossless for measured optics: `position=None` (`:257`), `residual_pixels=median_residual` (`:262`),
  `confidence=0.0` (`:263`, deliberately not the inlier fraction), `scale_known=False` (`:269`), with
  explicit position/scale/yaw-quadrant `unresolved` strings (`:265-267`).
- Reachability: only `scripts/calibrate_frames.py:45-47,190,236` (diagnostic JSON). No
  `src/mc_photogrammetry` importer of `calibration` except `reconstruction/surfaces.py:108`, and
  `surfaces.py` itself has no src/script importer — both isolated. `CameraSolution` has no src
  consumer beyond `contracts.py:87-95` — the production seam is disconnected.
- `CameraModel` (`camera.py:14-24`) requires numeric `position`; `intrinsics_for:110-123` scales a
  stored calibration; `_focal_length:97-107` fixes the vertical-FOV convention. Manifests
  (`camera.py:83-94`, `scene.py:26-35`) are explicit developer/evaluation inputs.
- Translation only from *stated* shared-plane links (`registration.register_axis:90-157`, docstring
  `:1-31`); scale only from a *stated* length (`lattice.scale_from_known_length:196`); both script-only.
- Fallbacks in the normal body (`facade.py`): estimated horizon (`:90`), horizon-sweep depth
  (`:115-119`), fixed 4→32-block depth when a camera exists without scene (`:136`), and the pixel-grid
  coordinate `(column, rows-row-1, depth)` when no camera exists (`:159`). The last is the silent
  blocking fallback: pixel indices become lattice coordinates with no camera/focal/scale.

### 2.3 Reconstruction entry — normal CLI unrunnable; metric path developer-only
- Normal `reconstruct` (`cli.py:249-307`) passes `max_color_distance`/`min_palette_margin`/
  `min_state_agreement`/`min_surface_contrast` (`:282-285`). The parser (`:47-68`) defines none and the
  verified `reconstruct_facade` signature accepts none, so every `reconstruct` run raises before
  reconstruction. `tests/test_cli.py` only parses/gates; no test invokes this body.
- `--horizon` ("override normalized perspective horizon", `:61,286`) is manual geometry accepted in
  normal mode ungated — a §1 contract violation.
- `reconstruct-multiview` (`:309-348`) and `reconstruct-visible-multiview` (`:350-413`) require
  `--mode developer` + `--cameras` + `--scene` (`:310-315,351-356`); the latter imports
  `reconstruct_visible_surfaces` (`:21`) but never uses it (`:382`) — stale import.
- `reconstruct_scene_surfaces` (`raycast.py:489-492`) takes `cameras: dict[str, CameraModel]` and
  `scene: SceneTransform` positionally; `sfm.reconstruct_from_calibrated_views` (`sfm.py:23-45`) needs
  supplied poses. Nothing converts `CameraSolution`/`CameraEstimate` to either — bridge **missing**.
- Desktop normal (`ui/main_window.py:72-78`) calls the façade; the developer branch (`:35-69`) loads
  sidecars only when `developer_mode` and ≥2 frames and both files exist, else silently falls through.
- `pipeline/` is protocols only (`:1-23`); `Reconstructor`/`CameraSolver` are unimplemented, so no
  single entry accepts frames/assets/settings/review/progress as handoff §4 requires.

### 2.4 Persistence — schema 1 works; schema 2 absent, no migration
- `ScanProject` (`domain/models.py:141-171`): `schema_version=1` (`:149`) with name,
  `minecraft_version`, `source_paths`, `volume`; `from_dict:161-164` rejects every other version, so no
  camera/lattice/constraint/settings/evidence/review state exists (handoff §4 requires schema 2).
- `project/store.py:13-35` saves atomically and wraps errors; `tests/test_project_store.py` round-trips
  once, with no schema-rejection/migration case.
- Desktop is write-only: `main_window.py:83-85` saves an unnamed project; `desktop.py:9` passes
  `scan.mcscan` but `run_desktop` (`:91,99,108-111`) never loads it. Saved-project review is disconnected.

### 2.5 Review — placeholder contract plus one status string
- `BlockObservation` carries `confidence`/`kind`/`frame_ids` (`models.py:76-102`), but
  `reconstruct_scene_surfaces` returns a bare `BlockVolume`; `ReconstructionResult`
  `alternatives/diagnostics/review_items/timings/complete` (`contracts.py:119-128`) is never constructed.
- No CLI/GUI uncertain-cell surface; desktop only prints "Review low-confidence/background candidates
  before building." (`main_window.py:166`). `export/attribution.py` and `evaluation.py` are diagnostics.

### 2.6 Export — implemented, two provenance gaps
- `LitematicExporter.export` (`export/litematic.py:46-144`) writes gzipped single-region litematic
  (data version `4440`, schematic `7`, `:42`), packs palette indices (`:15-32`), tight bounds by
  default (`:67-68,146-151`), optional parent `bounds` (`:70-76`), properties preserved
  (`_state_tag:154-158`); `export/validation.py:10-59` checks invariants; `tests/test_litematic.py`
  covers packing/bounds/gzip. This is the one working end-to-end link (world ZIP also works).
- Gap 1: `ScanProject.minecraft_version` (`models.py:146`) never reaches the exporter — data-version
  consistency is not end-to-end. Gap 2: desktop always describes the export as "Calibrated visible
  scene surfaces" (`main_window.py:86`), including the uncalibrated first-image façade path.

## 3. Normal-mode manual-metadata gate check

| Input | Required? | Accepted normal? | Evidence |
|---|---|---|---|
| images/folder | yes | yes | `media.py:42-81`; `cli.py:240-247` |
| camera manifest | no | rejected CLI, ignored desktop | `cli.py:250-253`; `main_window.py:35` |
| scene/bounds manifest | no | rejected CLI, ignored desktop | `cli.py:250-253`; `main_window.py:35` |
| F3/OCR values | no | absent — no OCR module (rg) | n/a |
| `--horizon` override | no | **accepted, ungated** | `cli.py:61,286` |
| `export --origin X Y Z` | no | **accepted, ungated** (world) | `cli.py:147-154,497` |
| coordinates/directions/lengths/poses | no | developer commands + scripts only | `cli.py:310-356` |
| saved advanced constraints | no | impossible: schema 1 stores none | `models.py:141-171` |

Verdict: manifests/F3 are properly excluded; `--horizon` and `export --origin` are concrete ungated
manual-geometry inputs. Developer sidecars are correctly behind explicit `--mode developer`
(`cli.py:172-176`, `main_window.py:35`).

## 4. DS-8-P1 — image-only camera seam (smallest high-confidence package)

Earliest blocking fallback: normal mode never runs the implemented image-only camera solve and silently
substitutes the pixel-grid body (`facade.py:159`), while the only normal CLI entry cannot execute. P1
makes the solved camera mandatory in normal mode, persists it, and moves manual/pixel-grid paths behind
developer mode. Non-goals: automatic translation/scale, connected multi-view pose/SfM, exact-model
routing, renderer/threshold/GPU changes, and `CameraSolution` contract changes.

1. New `src/mc_photogrammetry/pipeline/image_only.py`
   - `solve_image_cameras(frames) -> CameraSolution`: thin delegator to existing
     `calibration.enhance`/`detect_segments`/`estimate_camera`/`to_camera_solution`; no algorithm
     change; `ValueError` on zero frames.
   - `camera_model_from_estimate(estimate, *, position) -> CameraModel`: lossless intrinsics
     (focal/principal/roll/yaw/pitch); `position` is an explicit caller-supplied local-frame origin,
     never inferred; missing intrinsics raise.
   - `LOCAL_FRAME_ORIGIN = (0.0, 0.0, 0.0)` named constant, so no call site can pass it off as
     recovered translation.
2. `domain/models.py`: `schema_version=2`; add `camera_solution: dict|None`, `unresolved: list[str]`,
   `scale_known: bool`, `review_items: list[str]`; `from_dict` migrates 1 (camera fields absent) and
   reads 2 while still rejecting others. Shared contract — propose in comms first
   (`implementation-next.md:29`).
3. `cli.py` `reconstruct` (`:249-307`): repair the call site (drop the four unknown kwargs or add them
   to the parser as settings), run `solve_image_cameras`, build
   `camera_model_from_estimate(..., position=LOCAL_FRAME_ORIGIN)`, pass it to `reconstruct_facade`;
   gate `--horizon` and manifests to `--mode developer` (extend `:250-253`); save schema 2 with the
   solution/`unresolved`/`review_items`; state unresolved scale/translation in the export description.
4. `ui/main_window.py` normal path (`:72-78`): solve cameras for **all** `image_frames`, record every
   source path, pass the solved camera; keep façade/`--horizon` only under `developer_mode`; never claim
   calibration at `:86` when `scale_known=False`.
5. Targeted tests (focused modules only): new `tests/test_image_only_entry.py` (adapter losslessness,
   position required, solution keys/`unresolved`/`scale_known`, normal output uses solved focal via
   `camera.project_points` vs pixel indices); `tests/test_cli.py` (temp-PNG normal `reconstruct` path
   creates `.mcscan`+`.litematic` without manifests; `--horizon` rejected normal/accepted developer;
   existing gates retained); `tests/test_project_store.py` (schema-1 migration, schema-2 round trip,
   unknown rejection); new `tests/test_main_window.py` (normal `_reconstruct_source` forwards every
   frame path and the solved camera; monkeypatched media/load/reconstruct, no Qt import).

## 5. Luna conflicts and boundaries

- Luna owns `reconstruction/shape_fit.py`, its tests, M5 diagnostics/profile runners and reports, and
  S-C annotations after L-D/L-E (`implementation-next.md:26-28`); P1 touches none.
- `domain/models.py` and `pipeline/` are shared contracts — announce schema 2 in `AGENT_COMMS.md` first.
  `reconstruction/camera.py` is Luna-adjacent shared camera/types (handoff §4): P1 only **reads**
  `CameraModel`, keeping the adapter in the new module. `raycast.py`/`residual.py`/`face_sampling.py`
  are untouched, preserving DS-3/DS-4F/DS-5 authority.
- `cli.py`, `ui/main_window.py`, `project/store.py`, new tests are unowned; stage named paths only.

## 6. Limits

- Static inspection plus one argparse/signature probe; the default pipeline was not executed, so P1's
  reconstruction quality is unmeasured. The `reconstruct` crash is certain from parser/namespace/
  signature evidence but should be reproduced with a temp image when P1 lands. Statuses reflect the
  working tree at inspection time and need re-checking after Luna's L-D/L-E/S-C edits.
