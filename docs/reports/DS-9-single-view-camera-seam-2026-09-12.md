# DS-9 — Image-only single-view camera seam (implementation report)

DeepSeek CLI DS-9, 2026-09-12. Owns this report. Status: **source accepted by lead; report reconstructed
from already-established evidence — no test was rerun for this file.**
Evidence base (recorded in `AGENT_COMMS.md`, 2026-09-12T10:30:02 and 10:41:18):
lead inspected every DS-9-owned diff independently, then reproduced **28/28** image-seam + CLI tests;
an earlier focused regression run by the worker had already completed green before an interrupted,
redundant repeat. Per the lead's stop instruction, DS-9 did **not** rerun any test to produce this report.

## 1. Scope and authorization

- Directive: `AGENT_COMMS.md:2872-2882` (DS-9 narrow implementation). DS-8 P1 was split; schema 2,
  desktop multi-frame and multiview conversion were deliberately withheld pending translation/scale
  contracts (`AGENT_COMMS.md:2874`).
- Owned paths only: `src/mc_photogrammetry/pipeline/image_only.py` (new),
  `src/mc_photogrammetry/cli.py`, `tests/test_image_only_entry.py` (new), `tests/test_cli.py`,
  this report. All other dirty paths were preserved.
- Non-goals held: no schema/store/desktop/raycast/facade-algorithm/benchmark edits, no new dependency,
  no `CameraSolution` contract change, no automatic translation or scale, no commit/staging.

## 2. What the seam does

`pipeline/image_only.py` is a thin, manifest-free adapter over the pre-existing M3 calibrator.

- `solve_image_cameras(frames: Mapping[str, np.ndarray]) -> CameraSolution` (:52-70)
  converts each RGB frame to single-channel `uint8` via the same `cv2.COLOR_RGB2GRAY` convention as the
  shared SfM path, delegates to `calibration.estimate_session`, then to
  `calibration.to_camera_solution`. It adds no fit and changes no algorithm. Empty input and a fit that
  yields nothing raise `ImageOnlyCameraError` (:62-69).
- `camera_model_from_estimate(estimate, *, position) -> CameraModel` (:73-109) carries measured optics
  intact: image size, `(fx, fy)`, principal point (including crop-shifted centre), yaw, pitch, roll.
  It derives the **vertical** FOV from `fy` and image height,
  `fov = 2·atan((height/2)/fy)` (:98), because Minecraft's FOV setting is vertical. `position` is a
  required keyword, so no call site can default a translation; missing pose/intrinsics or non-usable
  values raise `ImageOnlyCameraError` (:88-97,108-109).
- `LOCAL_FRAME_ORIGIN = (0.0, 0.0, 0.0)` (:39) is the only position image-only recovery may attach.
- `_to_grayscale` (:112-127) rejects non-`uint8` frames and non-RGB/non-grey shapes before any solve.

Unresolved-state discipline: translation is unrecoverable from one view because vanishing points are
invariant under motion along the view ray; pixel focal length and orientation name no lattice period, so
`scale_known` stays false; the horizontal lattice axes are interchangeable in one view, so yaw is
meaningful only modulo 90°. The module docstring states all three (:19-25).

## 3. CLI integration (`cli.py`, `reconstruct`)

- Normal mode now rejects manual geometry — `--horizon`, `--camera-manifest`, `--scene` — with a message
  naming the offending flag(s) and `--mode developer` (diff at the `mode != "developer"` guard). This
  closes the specific ungated-`--horizon` contract violation DS-8 recorded (`DS-8 §3`).
- Normal mode solves the camera from the image alone, fetches the estimate for the frame id, adapts it at
  `LOCAL_FRAME_ORIGIN`, and passes it to `reconstruct_facade`; a solve failure raises `SystemExit` before
  any project or export is written.
- The four nonexistent kwargs (`max_color_distance`, `min_palette_margin`, `min_state_agreement`,
  `min_surface_contrast`) that caused the unconditional DS-8 crash are removed from the call; parsing and
  the verified `reconstruct_facade` signature now agree.
- Developer supplied-camera/scene behavior is preserved behind `--mode developer`.
- Output wording states the local frame and unresolved metric scale/translation; developer wording is
  unchanged. No manual coordinate input was added.

## 4. Verification evidence (recorded, not rerun here)

| Check | Result | Source |
|---|---|---|
| Lead independent reproduction, image-seam + CLI focused tests | **28/28 passed** | `AGENT_COMMS.md:2886` |
| Lead diff inspection: all DS-9-owned diffs | accepted | `AGENT_COMMS.md:2886` |
| Worker earlier focused regression | completed green before an interrupted redundant repeat | `AGENT_COMMS.md:2884-2886` |
| Full-suite baseline at the time | 527 passed (post-DS-3 agent run) | `AGENT_COMMS.md:2858` |

Accepted behaviors per the lead: normal single-view requires an image-derived camera; horizon, manifests
and scene are rejected; optics and roll are preserved; origin and scale are labelled unresolved; a failed
solve writes nothing (`AGENT_COMMS.md:2886`).

## 5. Limits and open work

- Quality of the adapted camera is only as good as the M3 single-frame fit; no reconstruction-quality
  metric was produced for the normal path, and the seam was not executed against a real fixture in this
  report.
- Still open by design: `.mcscan` schema 2, desktop multi-frame forwarding, and conversion of
  `CameraSolution`/`CameraEstimate` into the multiview renderer contracts. Those need the translation and
  scale contracts that DS-10 now scopes.
- Nothing here promotes a milestone gate; the frozen-fixture gates are unchanged.
