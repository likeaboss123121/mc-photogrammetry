# Production path audit — 2026-09-13

Status: verified lead/DeepSeek read-only audit; no implementation acceptance.

## Current executable paths

- Normal CLI `reconstruct` loads one image, estimates an image-derived camera with
  `pipeline.image_only.solve_image_cameras`, converts it to a local-frame camera,
  calls `reconstruction.facade.reconstruct_facade`, saves schema-1 `.mcscan`, and
  calls `LitematicExporter`.
- Normal desktop selects only the first discovered image and calls
  `reconstruct_facade` without a camera. It therefore uses the older pixel-grid
  coordinate fallback. It does not share the CLI camera path.
- Developer multiview paths consume supplied camera/scene manifests and call the
  calibrated SfM/raycast implementations. They are diagnostics, not evidence for
  normal-mode reconstruction.
- The M4 frozen image pipeline performs independent calibration, plane proposal,
  lattice measurement, graph matching and legacy component solving. It terminates
  in diagnostic JSON and has no adapter to `BlockVolume` or `.litematic`.
- `_joint_plane_selector` remains deliberately isolated. The frozen runner still
  calls `plane_graph.solve_component`; G1E must be proved before that changes.

## Verified integration defects

1. CLI and desktop do not call one production pipeline; desktop lacks image-derived
   camera recovery and multi-image use.
2. The normal CLI facade invents camera-ray depths from a fixed 4..32 range when
   metric scale is unresolved. This cannot establish block coordinates.
3. `facade.reconstruct_facade` records the fabricated frame ID `image-0000` instead
   of the actual source observation.
4. `FrameObservation`, `LatticeScene`, and `ReconstructionResult` exist as contracts
   but are not used by a production entry point.
5. M4 plane/graph output has no supported-surface-to-cell/volume adapter, so the
   image-derived research path cannot reach the already-working exporter.
6. `.mcscan` remains schema 1 and persists only source paths plus a volume; it has
   no cameras, lattice, constraints, alternatives, uncertainty, or review state.
7. `visibility` and `compare` accept supplied camera/scene data without an explicit
   developer/assisted mode label. Normal `reconstruct` and desktop sidecar loading
   are correctly gated.
8. CLI camera calibration and the frozen M4 runner use different calibration entry
   points (`estimate_session` versus `estimate_image_calibrations`).
9. M3 connected-camera research and M6 compute providers remain outside normal
   CLI/UI execution.

## Dependency order after G1E

### I1 — exact selector readiness

- Finish G1E isolated acceleration and parity proof.
- Select an explicit production budget only from measured Template-sized work.
- Wire the selector into a new diagnostic runner revision before changing normal
  CLI/UI. Preserve old frozen artifacts and typed whole-search abstention.

### I2 — lattice solution to visible cells

- Convert accepted plane identities, scales, translations, masks and supported
  boundaries into sparse visible lattice cells.
- Never fill bounding shells. Preserve holes, disconnected components, competing
  labels and unresolved depth.
- Produce `LatticeScene` and `BlockVolume` with per-cell observation provenance.
- Validate geometry on synthetic exact controls before any frozen evaluation run.

### I3 — one production entry point

- Add a callable accepting frames, assets, settings, review edits, progress and
  cancellation and returning `ReconstructionResult`.
- Use `FrameObservation` throughout; preserve crop/resize transforms and real IDs.
- Make CLI and desktop call this entry point. Normal multiview must use only image
  evidence. Supplied cameras/bounds remain developer-only and runs are assisted.
- Remove the fixed-depth facade result from production success; keep it only as an
  explicitly incomplete single-view fallback with uncertainty.

### I4 — persistence, appearance and export

- Migrate `.mcscan` with backward schema-1 loading and persisted cameras, lattice,
  alternatives, uncertainty, constraints and review edits.
- Run face-aware material/state fusion on reconstructed visible cells, then export
  the best estimate through the existing `LitematicExporter`.
- Emit review items for unresolved scale, pose, geometry or material state.

### I5 — hardening and gates

- Apply explicit developer/assisted gating to every camera/scene/reference command.
- Route renderer/scoring kernels through the compute provider after CPU parity.
- Run the preregistered T2, T6 and frozen-small-view T3 evaluation independently.
  Promotion requires geometry F1 >=0.90 and exact-state recovery >=0.80 on each.

## Immediate sequencing

G1E-A is active. After acceptance, prefer G1E-B budget/runner validation, then I2.
Desktop parity and mode hardening are independent but do not replace the missing
M4-to-volume bridge or improve the acceptance metrics; schedule them alongside I2
only when file ownership cannot conflict.
