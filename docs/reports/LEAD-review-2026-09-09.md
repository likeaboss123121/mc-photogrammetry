# Lead review: M4 image adapter, M5 fixes and M6 groundwork

GPT-6 Astra, 2026-09-09. Reviewed HEAD `8e51374`, including M4 `c27747b`/`94f6ebd`/`cc6f49d`,
M5 `73c2e63`/`8829c8c`, and M6 compute/cache/buffer additions. No implementation changes by the lead.

## Acceptance decisions

| Area | Decision |
|---|---|
| M4 image adapter | Useful implementation; semantic coverage and exported-coordinate mapping need correction. |
| M4 real pilot | Assisted assumed-grid diagnostic only; its 53 blocks are not validated reconstruction. |
| M4 A1 | Noise abstention improved; texture frequency still does not prove depth. Keep this cue out of B/C geometry. |
| M4-C | Conditionally authorized after the focused M4 corrections/checks below; no extra approval turn required. |
| M5 R2/R4 | Accept replayed exposure, all-masked, deletion and selected-state acceptance repairs; milestone remains open. |
| M5 R5 | Accept exact candidate cap/counting; default scheduling exhausts the budget before any partial shape. |
| M6 | GPU probes verified; caching/snapshot correctness needs fixes. Production GPU integration remains planned. |

## Verification

- Full `.venv` suite: **445 passed in 151.66 seconds**, using `python -m pytest -q -p no:cacheprovider -o addopts=`.
- Replayed M5 controls: empty-floor exposure gains 0.5/1/1.5 all return no insertion; an all-zero usable
  mask abstains; an extra slab returns explicit `None` deletion; three views with cap 1 perform exactly
  one candidate render and return no incomplete hypothesis.
- Re-ran `probe_wgpu_scoring.py` and `probe_wgpu_render_buffers.py` in `.venv-gpu` on the RX 9070 XT,
  D3D12. RMS score maximum absolute error: **1.9073486e-6**. The 37x29 buffer probe reports zero
  colour/depth/UV error. Context creation 1.180s, first scoring dispatch 0.121s, repeat 0.00124s;
  upload/readback approximately 0.032/0.056s. These are smoke timings, not application performance.
- Inspected the saved synthetic bundle, real pilot source overlay, reports and both scripts for the
  real-render transfer probe. That probe copies CPU-generated arrays; it is not GPU rasterization or
  independent renderer correctness. No new real-fixture reconstruction benchmark was run by the lead.

## M4 findings and corrections — Sonnet

### G1 — High: visible coverage includes foreground/holes and is normalized after clipping

`rasterize_cells` rasterizes every label independently, so a broad visible polygon keeps 100% coverage
even where an occluder overlays it. Specific labels only override it after crossing the 60% threshold.
Reproduction: one projected 100x100 cell, a full visible polygon and an occluder over its left half
returns `visible=1.0`, `foreground_occlusion=0.49505`, `declared=visible`. Actual unobscured coverage
is about 50%, below the declared 60% threshold.

Clipping compounds this: give the same projected cell an image only 10 pixels wide. It returns
`visible=1.0`, `outside_image_fraction=0.90099`, `declared=visible`. The score is coverage of the
remaining crop, although the contract describes coverage of the projected cell. Conflicting opening
and occlusion polygons are also resolved by hardcoded label priority, not evidence.

**Fix:** compose mutually exclusive semantic pixel regions first; subtract occlusion/openings from
visible support and retain conflicting labels as unresolved. Report visible-of-full-cell separately
from visible-of-in-frame-pixels, using the declared denominator for decisions. Retain partial coverage
and uncertainty. Avoid allocating a full offscreen raster just to measure projected area: a near-grazing
quad can be enormous even though only a tiny image crop is usable. Use bounded clipping/area evaluation.
Tests: half occluded under broad visible, opening/occlusion conflict, 90% outside image, extreme grazing.

### G2 — High: adapter and exporter transpose row/column axes

The adapter uses `u = origin_u + column*period`, `v = origin_v + row*period`. `plane_basis` maps u/v
to the remaining world axes in axis-index order. `footprint.cell_to_block` instead places **row on
the first world axis, column on the second**. Both pilot scripts pass these labels through unchanged.

With axis Z, unit period and zero origin, cell `(row=0,column=3)` has lateral low corner **(3,0)**
but exports block **(0,3,0)**. This transposes the footprint. Synthetic generation/reload uses the same
convention on both sides of its checks, and overlays draw the adapter grid rather than reprojecting
the exported block volume; neither catches this mismatch.

The synthetic script additionally substitutes face_coordinate=1 and integer origin zero for the
camera-relative plane distance/fractional origin. Arbitrary local export coordinates are fine, but
their transformation back to the reconstruction frame must be explicit and saved, not implied.

**Fix:** define one row/column-to-world-axis convention and a persisted export-to-reconstruction transform.
Test a nonsymmetric footprint's independently expected XYZ block set across axes/signs. Read the actual
litematic back and project its face corners through the saved transform/camera; compare with source
polygons. The overlay and scored geometry must originate from the read-back volume, not the pre-export grid.

### G3 — Medium: the real pilot silently chooses phase and does not save its full transform

`footprint_pilot_real.py` takes the first valid axis/side, constructs `lateral_origin=(0,0)` without a
phase anchor, and exports 53 blocks at the low-confidence estimated distance. `result.json` says
`status=resolved`. Marking the length low confidence does not resolve phase or establish exact block
coordinates. Arbitrary whole-build translation is permitted; fractional grid phase relative to the
photograph affects which block boundaries are being claimed and cannot be discarded as a gauge choice.

The real bundle saves summary cells/axis/distance, but not the complete transform, full per-cell
coverage, or source/calibration hashes. The synthetic bundle has a more complete persistence path.

**Fix:** share the persisted bundle schema, record axis/phase alternatives and their provenance, and
mark this result as an assumed-grid developer diagnostic. If phase/scale is unresolved, preserve the
continuous supported region; any discretized alternative must be explicitly identified as hypothetical.
Do not interpret 53 as recovered exact cells. No annotation retuning to reference truth.

### Conditional next step: C is a 2D boundary experiment

After G1/G2 independent controls pass and the corrected bundle reload/overlay works, Sonnet may proceed
directly to C. Correct G3's provenance/status before including the pilot in a report. No extra lead
approval checkpoint is required between these fixes and the already bounded C experiment.

- Freeze one correctly labelled visible-plane target per T2 frame from the source images, with HUD,
  foreground and openings distinguished. Freeze target masks/settings before scoring. Target masks are
  evaluation controls only, never proposal input; no fixture poses/reference blocks/F3 values in proposals.
- Run one automatic structural-boundary proposal configuration on each of the three frames. Reuse
  image-derived vanishing directions and line chains: maximum 12 chains/region and 8 boundary candidates.
  No period/seam occupancy gate, new dependency, generic SfM rerun, material tuning or threshold sweep.
- Compare all candidates over one fixed usable-pixel domain. Report per-frame region precision/recall/IoU,
  normalized boundary error and overlays, including failures and competing interpretations.
- The local continuation condition remains >=90% visible-region precision and >=50% recall on at least
  two of three targets. This is **2D footprint evidence**, not camera/block-scale recovery or production
  promotion. Unknown real phase/scale does not prevent this mask experiment; it prevents exact-cell claims.
- Stop after the three runs and return for review. If the condition fails, report the failure; do not
  invent another cue or make manual calibration mandatory. Normal/advanced boundaries remain unchanged.

## M5 findings and next assignment — Luna

The replayed R2/R4 regressions and cap-overrun are fixed. Preserve those fixes. Two integration gaps
remain before widening to real-scene partial fitting:

1. **High: default cap prevents any partial shape from being examined.** Actual local JAR enumeration
   gives 700 materials in the first full-cube shape group, followed by 57 bottom slabs, 57 top slabs,
   and other groups. The loop exhausts the default 256 candidate-render budget inside the first group,
   even with one view; with three views it covers fewer cubes. Thus the normal complete-palette partial
   pass never reaches its first partial shape. A tiny two-state palette cannot validate this scheduling.
   Allocate work across geometry hypotheses before material variants. Use reusable per-shape geometry
   and texture ranking, with deterministic bounded refinement and explicit unresolved cases. Keep air/
   current baselines. Do not fix this by increasing the cap or erasing rare materials from the vocabulary.
2. **Medium: supporting frames depend on optional diagnostics.** `fit_states` records actual support
   only in diagnostics; `raycast` falls back to every input frame when diagnostics is None. The CLI
   uses that ordinary path. Carry accepted edits' support through a required result channel or always
   capture it internally; enabling logging must not change reconstruction provenance.

Next deliverable: one bounded, deterministic scheduling control with a cube-heavy palette and known
slab/carpet targets. Show that at least those relevant partial geometries are evaluated within the
same cap, unaffected by palette ordering, while empty/correct-cube controls still pass. Log considered
shape families, candidate/render counts, accepted evidence and unresolved budget exits. No full fixture
sweep until this succeeds. Exact alpha/model-face material sampling and matched T2/T6 quality evidence
remain M5 acceptance requirements; the current one-representative-per-box-group material path is not
proof of arbitrary partial-model correctness.

## M6 corrections — Luna, bounded groundwork only

### C1 — Medium: oversized replacements leave ghost bytes in both LRU caches

In both `FramePyramidCache.put` and `BoundedArtifactCache.put`, rejecting an oversized replacement
removes the old entry without decrementing `_bytes`. Reproduction with a 10-byte artifact cache:
insert key A=4 bytes, replace A with 11 bytes => length 0, bytes_used=4. Insert B=8 bytes => B is
evicted despite fitting the empty cache; length remains 0, bytes_used=4.
Fix replacement accounting and independently test both caches, including subsequent valid insertion.

### C2 — Medium: read-only buffers alias caller storage and mutate its flags

`pack_render_result` uses `astype(copy=False)`/`ascontiguousarray` then `setflags(write=False)`.
For canonical contiguous inputs this changes the caller's original array flags, contrary to the
function contract. An alias created before packing can still mutate the supposedly immutable snapshot:
the lead probe changed an alias to 99 and `packed.colour` changed to 99. Frame pyramids use the same
pattern for their source level. This makes cached accepted buffers unstable under later edits.
Own immutable copies, or define/test explicit ownership transfer. Preserve source writability when
promising non-mutating packing, and test changes through pre-existing aliases.

These are small fixes, not a reason to start production GPU wiring. Existing CPU/GPU transfer/scoring
smoke evidence is useful, but the M6 bundle is not fully accepted while C1/C2 remain. The scoring
shader launches 64 threads/group but dispatches candidate_count groups; use ceiling(count/64) and
handle zero-candidate batches when preparing performance validation. No performance promotion is
supported by the three-vector timing alone.

No agent is reassigned to general renderer/CLI integration by another agent's report. Sonnet owns
the M4 fixes/C experiment; Luna owns M5 scheduling/provenance and these small M6 cache repairs.
M1/M2 review gates, schema migration and the 80%/90% production gates remain open/unchanged.
