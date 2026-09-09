# M4 direct review and recovery plan

> Current next step: [follow-up review](M4-AB-M5-followup-review-2026-09-08.md) requires an independent
> synthetic footprint-to-cell control before further real-image runs. M4-C remains paused; its scope below is planned.

Reviewer: GPT-6 Astra. Reviewed `32515413a7dd1a9eaa5bd7355d51e0da29067ae7` on 2026-09-08.
Current coding agents: Claude Sonnet 5 and GPT-5.6 Luna. Luna's user-authorized M5 work continues.
This review makes documentation changes only and supersedes the M4 hold/conclusions in the status report.

## Decision

Stop using per-cell seam strength as a binary occupancy detector. Recover a visible planar region,
its boundaries and holes first; estimate a shared block lattice on that region; then discretize it.
A solid wall does not need to show a seam at every block. A region's footprint must have evidence,
but evidence need not be a separately detectable texture seam in every cell.

Do not lower phase thresholds, try another brightness rule, or build a solver over the same telescoping
distance differences. Do not make length, plane identity and yaw quadrant mandatory production inputs.
They remain explicitly labelled diagnostic inputs or optional constraints behind advanced/developer mode.
User clarification: supplied manifests, F3/debug values, coordinates, directions and other explicit geometry
are for program refinement/evaluation. Normal operation must derive camera/geometry from visual scene
evidence, without accepting these inputs, parsing F3 values, or silently loading sidecars/saved overrides.
Advanced/developer opt-in must be explicit; reference assets remain confined to evaluation adapters.
Image-derived reconstruction remains the objective; neither feasibility nor impossibility has been proved.

## What the review actually verified

- Read `supported.py`, `surfaces.py`, `registration.py`, the supported-plane/registration/separation
  scripts, artifacts and reports; visually inspected the Template 2 views.
- Focused command: `.\.venv\Scripts\python.exe -m pytest tests/test_supported.py tests/test_registration.py tests/test_lattice.py -q -p no:cacheprovider`.
  **52 tests passed.** No full suite or expensive fixture sweep was needed for this review.
- `support_cells(full((64,64),128), ones((64,64)),16,(0,0), local_window_periods=1.5)` returns
  **16 contradicted cells, all score 0**. This is insufficient texture evidence, not evidence of holes.
- `plane_from_evidence(full((64,64),128), 2, axis_directions(0,0),32,(64,64),distance=4,extent=.25)`
  returns **None**. The assisted path unnecessarily requires a successful texture-period estimate.
- Replayed the AUC ranking expression with 55 positive and 114 negative identical scores: it returns
  **0.0 instead of 0.5** because ties do not receive averaged ranks. The reported 0.523/0.502 values
  are not thereby disproved; they must be rechecked with a correct evaluator and trustworthy labels.
- Registration artifacts contain 3 Y offsets, 3 X offsets and only **2 Z offsets**. One camera's Z
  remains unknown. Horizontal artifacts explicitly have `quadrants_simulated=true`. Sub-0.004-block
  residuals measure propagation of simulated constraints, not automatic recovery of all camera poses.
- The dominant occupancy-test view, `13.24.33`, looks **down** at the floor (pitch +39.6 degrees in
  this project's convention), with furniture/fences obscuring it. Reports calling it upward-looking
  are incorrect. No broad cross-scene conclusion follows from its 168 of 169 judged cells.

## Why the current approach stalled

1. **Unknown is being called absent.** `supported.py:246` maps every low score to `CONTRADICTED`.
   Textureless surfaces, texture aliases, shadows and foreground objects then become supposed openings.
   Keep weak evidence unknown. Foreground occlusion does not prove the plane behind it is missing.
2. **The representation loses what export needs.** Cell indices start at the rectification window's
   origin rather than an explicitly phase-aligned lattice origin. `SupportedPlane` stores neither the
   window centre/extent/sampling transform nor an invertible mapping to source pixels and 3D coordinates.
   Local phase estimates may diagnose drift, but independent per-cell phases cannot define one block grid.
3. **The assisted path is still gated on the failed automatic subtask.** A stated distance should allow
   a region and unknown phase to survive even when `measure_plane` fails. Region, scale, phase and
   occupancy need separately reported statuses. Missing texture must not delete valid user evidence.
4. **Evaluation labels are weaker than the claims.** `score_separation.py` uses box-default visibility,
   any reachable cell at a lateral coordinate and a 0.51-block plane tolerance. That is not a per-pixel
   check that the sampled ray sees this exact plane face. Occluded floor, a nearby slab and an opening
   can be conflated. It also lacks tie-correct AUC and persisted per-cell labels/projections.
5. **The small registration model is being mistaken for the full problem.** Its telescoping proof is
   correct for one distance per view/axis. It does not rule out independent corner/boundary observations,
   multiple planes per view, or image residuals. No general refinement infrastructure is needed now.
6. **Global yaw and relative alignment are different.** Choose an arbitrary global horizontal axis
   orientation for a local `.litematic`; true compass north is unnecessary. Relative quarter-turn choices
   between views can remain finite hypotheses. Perfectly symmetric evidence may leave them ambiguous,
   but failed generic feature matching does not prove all structural correspondence impossible.

## Sonnet's next assignment — three small, sequential deliverables

### A. Repair the evidence model and evaluator; no new occupancy cue

Own `supported.py`, `scripts/supported_plane.py`, `scripts/score_separation.py` and focused tests.

- Separate supported visible surface, unknown/weak evidence, foreground-occluded, outside-image and
  explicitly contradicted surface. Use an existing enum/dataclass extension; avoid a framework rewrite.
  A contradiction requires positive incompatible geometry evidence, not missing gradients.
- Preserve distance when period estimation fails; preserve a region when phase is unknown. Unknown
  phase prevents claiming exact block coordinates, but does not erase the plane observation.
- Persist the plane basis, side, distance/scale status, lattice-origin/phase status and rectification
  mapping. Make cell corners map back to image pixels and 3D points, including nonzero window centres.
  Use one consistent pixel-spacing convention (`linspace` endpoints versus sample count currently differ).
- Fix AUC ties and keep coverage/occlusion separate from occupancy labels. Use tiny independent synthetic
  scenes first: uniform wall, wall with a genuine opening, wall behind a foreground slab, and invalid
  image pixels. Verify the evaluator before interpreting another real-data AUC.
- Keep fixture reads in the scoring adapter. The current registration script loads references/manifests
  even with `--no-score`; it is a diagnostic harness, not a production input path to reuse unchanged.
- Publish the changed contract and tests in comms. Stop here if these controls fail; do not spend the
  real-image experiment budget debugging an unverified evaluator.

### B. Prove mask-to-cell construction independently of automatic segmentation

This is an **assisted diagnostic control**, not the production workflow or a new required user burden.
Run it through developer diagnostics. It does not authorize adding manual geometry fields to normal UI/CLI.

- Take an image-derived camera, a visible plane mask/polygon drawn from the image, and the existing
  optional known-length constraint. Use one clearly visible target plane per Template 2 image, selected
  and saved before scoring. Prefer the far wall visible in `13.24.24` over a nearly out-of-frame floor.
  Save image-only target annotations/hashes. Do not obtain masks, clicks or quadrants from the reference.
- Annotate only visible surface pieces, exclude foreground furniture/HUD, and mark any actual opening
  separately. A foreground occluder is an unknown footprint portion, not an opening. A region touching
  the screenshot boundary has an unobserved continuation, not a measured physical edge.
- Rectify this region with the known distance; establish one plane lattice from identifiable block
  intersections/boundaries or retain phase hypotheses. A generic length supplies distance, not phase:
  a clicked lattice intersection can also anchor phase only when explicitly identified as one.
- Rasterize the irregular footprint against that lattice with per-cell visible coverage and evidence.
  Do not fill its bounding rectangle. Cells completely occluded or outside the image stay unresolved;
  boundary cells keep partial coverage and alternatives rather than being silently dropped.
- Export a **geometry diagnostic** `.litematic` using a declared neutral block and save the footprint/
  transform/evidence alongside it. Its state accuracy is intentionally not a material result. Luna's
  M5 material stage can consume these supported faces without changing its palette work.
- Validate integer cell placement, holes and cell-to-image round trips on synthetic controls, then report
  the three fixed real regions. If supplied visible-region evidence still cannot produce correct cells,
  fix that path before attempting automatic masks.

### C. One automatic boundary proposal experiment, using the same fixed targets

Hypothesis: long structural boundaries and connected surface regions provide footprint information
that the distribution of interior texture gradients does not. This is untested, not a promised solution.
Here "automatic" describes boundary proposals only. If scale or other inputs are assisted, label the
whole run assisted; it cannot qualify as image-only reconstruction. Do not pass control-mask geometry,
reference-derived window bounds or reference camera poses into the proposal arm.

- Reuse detected multiscale line segments and image-derived vanishing directions. Merge collinear,
  nearby segments into long chains; prefer persistent chains and corner/termination structure over
  short repeated texture edges. Retain a small, deterministic candidate set (maximum 12 chains per
  region and 8 outer-boundary candidates). Log rejected/ambiguous alternatives.
- Form candidate plane polygons from compatible pairs of in-plane directions and observed line
  intersections. Score observed boundary alignment/termination support and coherent visible-region
  coverage, not the fraction of all texture energy landing on block seams. Use a weak simplicity prior
  only for ranking; it cannot create physical boundaries or occupancy without image evidence.
- Split at candidate structural boundaries; keep internal openings and foreground occlusion as distinct
  alternatives until evidence resolves them. Never treat every material-colour edge as a depth edge.
  A hand-drawn mask is used only as the control/evaluation target in this automatic arm.
- Give every candidate the same fixed usable-pixel domain for comparison. Where image boundaries do
  not distinguish masks or depths, return ambiguity with overlays. Do not force a single region by threshold.
- Freeze one settings configuration, then run the same three frames once. This is at most **six real
  frame runs total**: three assisted controls and three automatic proposals; no threshold sweep, new
  neural dependency, generic SfM rerun or full-fixture reconstruction benchmark in this assignment.

## Experiment acceptance and stopping criteria

- Synthetic controls must preserve a uniform supported region, retain real openings, mark occlusion
  unknown, ignore invalid pixels and map a nonzero-phase grid consistently. All-tied AUC must equal 0.5.
- Real-region reports need target area, visible-region precision/recall/IoU, boundary error at normalized
  resolution, cell precision/recall, unresolved counts, phase/scale alternatives and source-image overlays.
  Use correct full-model reference geometry only after predictions are frozen. Report each frame separately.
- To continue automatic integration: at least two of the three fixed targets must reach **90% visible
  region precision and 50% recall** with a nontrivial supported footprint; report the third failure too.
  These are local experiment criteria, not replacements for the project's 80%/90% production gates.
- If automatic boundaries fail, deliver the working assisted control and failure overlays, then return
  for lead review. Do not invent a seventh cue or make assisted inputs mandatory. No claim that the whole
  problem is impossible is supported by this bounded result.

## Ownership and next review

Sonnet owns these M4 modules/scripts and their local contracts. Luna continues the user-assigned M5
palette/material/partial-block work. Do not touch Luna's textures, renderer, `shape_fit.py`, CLI/UI or
benchmark files without a named coordination message. Shared type changes are proposed in comms first.
M1/M2 review issues remain tracked; this assignment does not silently accept or close them.

Review A and the mask-to-cell control before broadening scope. Keep the active handoff concise and
put detailed measurements here or in new linked reports. Preserve existing simulated-constraint
measurements with their provenance and partial-axis coverage, rather than rewriting them as image-only wins.
