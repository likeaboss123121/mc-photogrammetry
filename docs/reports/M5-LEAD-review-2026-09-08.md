# M5 palette integration review and next directives

> Follow-up: [latest review](M4-AB-M5-followup-review-2026-09-08.md) accepts R1 and the reviewed R3 gates;
> R2/R4/R5 remain open with new reproductions. The original review and measurements below are historical.

Reviewer: GPT-6 Astra, 2026-09-08. Reviewed `41ea653` and `ec98d59`, with HEAD at `3251541`.
The lead performed review, tests and in-memory probes; no implementation coding.
Sonnet's concurrent M4 work remains separate. Luna owns the corrections below.

## Decision

**M5 remains open.** Luna submitted a useful renderer/palette-support slice, as the completion message
itself states. It does not yet deliver supported-surface material recovery or validated partial-block
improvement. Keep canonical cubes in the first classifier and preserve baked metadata; correct the
unsafe fallback and newly exposed fitting path before accepting this slice for integration.

The user's latest input boundary also applies: supplied cameras, F3/debug values, directions and
geometric constraints are refinement/developer aids. Normal operation must derive geometry from
visual scene evidence; manual geometry requires explicit advanced/developer opt-in.

## Verified evidence

- Focused tests: `tests/test_shape_persistence.py`, `test_complete_palette.py`, `test_shape_fit.py`,
  `test_cli.py`: **42 passed**. These include the new metadata and parser checks.
- Full suite: `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=`:
  **383 passed in 149.27 seconds**. This was a live-workspace run; Sonnet began editing M4 during
  the review. It is not a claim of testing a clean checkout or verifying the earlier 393-test claim.
- Actual local 1.21.5 JAR, cache schema 15: 8,672 samples, 726 canonical entries and 6,163 complete
  entries. Every canonical state is present in the complete palette under current normalized equality.
- `palette_for_states` preserves model metadata when the requested normalized state exists. All 26
  non-air Template 2 reference palette entries resolve exactly. The fallback defect below therefore
  is not an explanation of the supplied Template 2 score.
- [First diagnostic](../../exports/m5-reference-t2.json): reference-assisted, 26 palette entries,
  814 candidate cells, 0/668 visible exact matches, geometry F1 92.6606%, conditional state accuracy 0%.
- [Corrected diagnostic](../../exports/m5-reference-t2-v2.json): reference-assisted, 6 full-cube palette
  entries, 814 candidates, **349/668 visible exact matches (52.2455%)**, geometry F1 **92.6606%**,
  **432/780 conditional state accuracy (55.3846%)**. Both use all decoded pixels without frozen masks.
  These are inspected existing artifacts, not freshly rerun benchmarks. Both set `FIT_PARTIAL_SHAPES=False`.
- The corrected artifact records source revision `0f5d510` plus dirty M5 source files and source hash
  `70cc6893e688ddf526974ac8dd90228bae5b054ece3800c20f81b3900f009103`; the first records `694b051`
  and hash `011ac359c507b83ea7b0b8835638946253d7aa60fcbf46e17036ec02c963d2d9`.
  Preserve that provenance; neither artifact is a clean-checkout, before/after M5 acceptance pair.

## Findings

### M5-R1 — High: fallback can label the wrong geometry as an exact state

`src/mc_photogrammetry/blocks/textures.py:314` falls back to the first sample with the same block name,
then replaces its state key with the requested state. Matching the name does not establish matching
geometry, orientation, textures or alpha. `include_partial=False` then tests that substituted sample.

Independent reproduction: populate a provider with only `oak_slab[type=bottom]`, with a Y extent of
0..0.5, and request `oak_slab[type=top]`. The function returns the requested top-slab key with the
bottom-slab boxes. The existing test exercises only an exact match and an entirely unknown name.
Real-JAR bottom/top/double states resolve correctly today; this defect concerns a missing requested
variant, precisely the incomplete-cache/accepted-edit case the new fallback claims to support.

**Directive:** use normalized exact-state resolution. If a state cannot be resolved, report it as
unavailable or bake that exact validated state from assets. Permit default completion only when it
preserves all supplied properties and the resolved geometry is justified. Never silently relabel
arbitrary same-name geometry. Add independent top/bottom/double and facing/rotation mismatch tests,
including missing variants, compatible defaults, and full-cube filtering after correct resolution.

### M5-R2 — High: the partial pass cannot keep an empty cell or reject a harmful edit

`shape_fit.py:113` excludes cubes; `fit_states` offers partial states only. It never compares air or
the existing scene. `raycast.py:1240` inserts every returned fit into the output. Residuals can arise
from material, lighting or imperfect geometry, so being a proposal is not proof a block belongs there.

Independent replay using the small stone-floor scene in `test_shape_fit.py`: render the floor with
no block at `(2,1,3)` and ask `fit_states` about that empty cell. It returns a bottom oak slab.
On the same 88-pixel cube-defined comparison domain, the unchanged floor has **0.0 mean RGB L2
error**, while the returned slab has **55.094715**. This is an insertion despite a perfect empty baseline.

**Directive:** include air for empty proposals and retain-current for existing cells, plus relevant
cube/partial alternatives. Compare all hypotheses over one usable, candidate-independent domain.
Accept only evidence-supported improvement; ties/insufficient evidence preserve the current scene.
Recompute visibility after accepted edits and preserve actual supporting frame IDs. Validate an empty
cell remains empty, a real slab is recovered, a correct cube remains a cube, and adjacent proposals
cannot ignore each other's occlusion. Existing slab-versus-carpet ordering must continue to pass.

This is a pre-existing fitting limitation newly exposed through the M5 CLI flag; the palette commit
did not create the algorithm, but its integration cannot be accepted as validated partial recovery.

### M5-R3 — High: normal UI/CLI still accepts the explicitly prohibited camera inputs

`ui/main_window.py:29-37` checks adjacent `cameras.json`/`scene.json` and loads them automatically.
`cli.py:73-85` defaults one multiview command to fixture paths and requires supplied camera/scene
paths in the visible-multiview command. There is no explicit advanced/developer mode gate.
Parser replay accepts `reconstruct-visible-multiview images --cameras c.json --scene s.json
--fit-partial-shapes` without any mode opt-in.

This predates the user's latest clarification. It is a current integration requirement, not a claim
that Luna knowingly violated a requirement before it existed.

**Directive:** Luna owns the UI/CLI mode gate; announce shared mode semantics in comms. Disable
implicit sidecar discovery in normal mode and gate manual/imported pose, intrinsics, direction, bounds
and geometry constraints behind explicit opt-in. Preserve existing calibrated tools as clearly labelled
developer diagnostics, with provenance. Evaluation/reference tools remain separate. Do not fabricate
image-derived camera recovery to keep an old command running; return explicit unresolved capability
where the automatic pipeline is not yet available. Normal mode cannot silently reuse advanced metadata.
Test default rejection/no sidecar reads, explicit diagnostic access and mode labelling. Do not use F3 OCR
as an automatic replacement for manifests. Keep unrelated export-only coordinate transformations intact.

### M5-R4 — Medium: exact visibility is not carried into material evidence

`shape_fit.py:366-370` computes a usable region for geometric scoring but discards it while extracting
texture patches. `_face_patches` uses axis-aligned box quads, averages all rectified pixels, rejects
partially off-frame quads wholesale and has no HUD/usable-mask input. Exact model polygons and alpha
are present in the palette, but this material path still cannot distinguish a visible face fragment
from foreground/HUD pixels over that face. This is an existing limitation affecting M5 acceptance.

**Directive after R1-R3:** carry fixed usable masks plus rendered face ownership, depth and alpha into
texture sampling and exposure fitting. Resolve material only from surviving pixels of the candidate
face. Keep unresolved evidence explicit. Add a synthetic wall partly hidden by a contrasting slab,
HUD/invalid pixels and a clipped-face control; changing excluded pixels must not change the material.
Coordinate renderer utilities within Luna's ownership and leave Sonnet's M4 modules untouched.

### M5-R5 — Medium: partial fitting has no submitted quality or runtime validation

The only new end-to-end artifacts have partial fitting disabled. CLI coverage checks parsing, not
that the fitting pass preserves scene quality or exports correct partial states. The full-suite pass
alone cannot establish the M5 deliverable of improved exact matches and partial-block recall.

Current complete-palette enumeration yields **33 shape groups and 706 material candidates**. The
implementation calls `render_result` inside its per-material loop. With the default 40 proposals and
three usable views, that loop can attempt **84,720 full-scene candidate renders**, excluding probes
and other work. This is an operation count, not a measured runtime or a worst-case wall-clock claim.
The module prose's "one render per shape" description does not match the current loop.

**Directive:** instrument actual candidate/render counts and elapsed time on a tiny controlled scene
before running complete palettes across real fixtures. Separate shape geometry rendering from material
texture scoring, caching/reusing geometry where valid; preserve rare-material alternatives. Do not
launch an expensive full-palette sweep to discover this bottleneck. Performance optimization follows
the correctness controls and must preserve the air/current comparison.

## Ordered assignment and acceptance

1. **Next patch:** R1 exact-state resolution and R3 mode boundary. Publish scope/contract in comms;
   keep Sonnet on M4. Demonstrate focused regressions and unchanged explicit diagnostic access.
2. **Next small fitting patch:** R2 with a tiny cube/slab/air palette, then R4 visibility-aware material
   evidence. Prove empty/current preservation and positive recovery before widening the vocabulary.
3. **Bounded evidence:** R5 instrumentation and one fixed synthetic holdout configuration. Run full
   palettes only after a declared render/work cap exists. No threshold or material-prior sweep.
4. **Later integration acceptance:** fixed-settings, matched baseline/candidate T2/T6 diagnostics with
   one reviewed evaluator/mask policy, partial fitting off/on, per-state and partial-family precision/
   recall, phantom counts, exact matches, supported frame evidence and runtime. Report image-derived
   versus assisted inputs explicitly. Do not compare the 52.25% artifact to an unmatched old baseline
   and call the difference a regression or an improvement.

M4-supported geometry is still needed for full M5 acceptance; these corrections can proceed without
waiting for M4. The 80% visible exact-state and 90% visible geometry-F1 production gates are unchanged.
No milestone is promoted by this review. Keep the active handoff concise and link reports for detail.
