# M5 lattice-cell state integration audit

Date: 2026-09-13  
Role: DeepSeek read-only/no-shell source audit, independently checked by GPT-6 Astra  
Status: design evidence; no implementation or promotion

## Result

Material and partial-state fitting exists, but it is disconnected from the
image-derived M4 path. Normal CLI and desktop reconstruction still call the
legacy facade. The M5 fitter is reachable only through the developer command
that requires supplied cameras and scene bounds. The shortest route to the
visible exact-state gate is therefore a shared face-observation contract and a
face-aware scorer over M4's future image-supported cells, after the V1 region
footprint and V2 placement contracts.

The current M5 render scheduler also remains a prerequisite. It allocates budget
per cell while a hypothesis consumes a complete set of views; incomplete view
sets are discarded. A budget smaller than the usable-view count can still spend
setup/probe work without evaluating a complete hypothesis. F2 must reserve
atomic full-view work before production integration.

## Verified reusable components

- `face_sampling.FaceSample` and `rectify_face_valid` provide rectified RGB,
  validity, coverage, and stable face-slot evidence.
- `raycast.PaletteIndex`, normalized texture patterns, face colors, and
  `slot_coverage` implement face-aware palette costs.
- `VanillaJarTextureProvider.palette_for_states` preserves exact block-state
  geometry; `complete_palette` includes partial blocks omitted by the old
  cube-only palette.
- `ModelAtlas` bakes blockstate variant rotations into face geometry.
- Lighting response and ambient-occlusion machinery are image/model based and
  reusable after validation on corrected geometry.
- `BlockState`, `BlockObservation`, `BlockVolume`, and the litematic exporter
  preserve one selected state for export.

## Verified disconnects and risks

- `raycast.reconstruct_scene_surfaces` owns the M5 `residual_proposals` and
  `fit_states` calls. The CLI reaches it only in developer mode with supplied
  cameras and scene bounds (`cli.py:399-443`, `raycast.py:1278-1313`).
- Normal CLI and desktop call `reconstruct_facade`; normal mode never invokes
  M5 (`cli.py:318`, `ui/main_window.py:76`).
- `PaletteColor`, the appearance record used by the fitter, lives in the legacy
  facade module. M4 must not depend on the facade to gain material scoring.
- `fit_states` divides remaining work across cells (`shape_fit.py:1045`) while
  incomplete view sets are rejected (`shape_fit.py:774,912`). F2 atomic
  scheduling/reporting remains open.
- Fitted observations currently write `confidence=0.0` (`raycast.py:1310`), so
  confidence cannot carry calibrated uncertainty.
- Canonical candidate filtering excludes derived connection/open properties;
  no post-fit neighbour-rule pass restores them (`shape_fit.py:103,130-170`).
- `BlockObservation` and `BlockVolume` hold one state per coordinate. There is no
  typed per-property uncertainty, alternative-state set, or conflict record.
- `TextureProvider` exposes only block resolution and JAR loading. It has no
  resource-pack overlay order, pack identity/hash, game data version, biome tint,
  or closest-vanilla decision contract.
- Existing matched M5 diagnostics use supplied cameras/bounds and reference
  scoring. They are evaluation evidence and cannot prove normal-mode behavior.

## Required contracts

1. A typed visible-face observation must connect an accepted M4 cell to world
   face slot, outward normal, supporting frame IDs, rectified pixels, usable
   coverage, and image-derived camera. V2 placement must own orientation.
2. A neutral appearance module must own the palette/state records and scoring
   interface so M4 does not import the legacy facade.
3. Scoring must keep shape and material verdicts separate and retain competing
   states/properties for review. Confidence remains diagnostic until calibrated.
4. A post-fit neighbour-rule pass must derive only game-determined properties
   for changed cells and neighbours. Visually unknowable properties remain typed
   uncertainty; `open` is not universally derivable.
5. Schema 2 must persist geometry alternatives, material/state alternatives,
   per-property uncertainty, conflict records, provenance, and review state while
   reading schema 1.
6. Known resource-pack/JAR input requires an explicit pack identity and overlay
   contract. Unknown-pack mode reports closest-vanilla alternatives rather than
   claiming an exact source state.

## Implementation order

### S1: face observation and neutral scorer

- Build the face record on V1/V2 region footprints and placements.
- Move or re-export appearance records outside `facade.py` without changing
  palette semantics.
- Score exact states with existing face textures, slot coverage, and model atlas.
- Test all face orientations/quadrants, palette-order invariance, masked pixels,
  partial model geometry, and missing metadata fallback.

### S2: atomic M5 scheduling

- Reserve a full usable-view set before starting a hypothesis; otherwise mark the
  cell unvisited without context/probe renders.
- Preserve F1 evidence retention and report complete hypotheses, render counts,
  unvisited cells, and budget exhaustion separately.
- Test caps below/equal/above view count, deterministic allocation, cancellation,
  and prohibition on partial-view acceptance.

### S3: alternatives and derived properties

- Return typed ranked state alternatives and separate shape/material evidence.
- Recompute deterministic neighbour-derived stair/fence/wall/pane/bar properties
  after accepted edits; retain non-derived properties as uncertainty.
- Persist in schema 2; export one reviewed/best state without inflating geometry.

### S4: M4 integration and gates

- Apply S1 scoring to exactly the supported cells emitted by V3, using only the
  frames that support each visible face.
- Wire the unified image-only entry to normal CLI and desktop; keep manifests,
  F3, supplied cameras, bounds, scene transforms, JAR/reference visibility, and
  manual geometry controls in explicit advanced/developer mode.
- Evaluate independently on T2, T6, and frozen small-view T3. Promotion requires
  at least 0.80 visible exact-state and 0.90 visible geometry F1 on each fixture.

No source, test, image, ground-truth, evaluator, or frozen artifact was modified
or executed for this audit.
