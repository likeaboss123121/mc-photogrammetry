# M3 Production Camera Backlog Audit

Date: 2026-09-13
Role: DeepSeek read-only/no-shell audit
Status: current-source planning evidence; no implementation or evaluation performed

## Current normal paths

The CLI loads one image, estimates a single orientation/focal solution, forces camera position to the
local origin, and calls the legacy facade. The facade bins the image, chooses palette colours and
unprojects cell centres through a fabricated 4..32 depth range. It stamps observations with the
synthetic frame ID `image-0000`. The resulting litematic is not a connected block-scale reconstruction.

The desktop path is separate: it selects only the first discovered image and calls the facade without
a camera. Developer mode can consume supplied camera/scene sidecars, but that path is not normal-mode
evidence and must remain advanced-only.

The connected camera modules (`correspondence`, `connected_camera`, `joint_matching`,
`directed_matching`, `graph_tracks`) are currently diagnostics/test infrastructure and are not reached
by the normal CLI or desktop path.

## Proven pieces

- Single-view orientation/focal calibration passes the stored seam diagnostic; translation and scale
  remain unresolved by design.
- Independent per-frame calibration exists and can abstain on individual frames.
- Analytic pair pose handles four relative yaw classes, cheirality, seeded RANSAC, unequal intrinsics
  and typed degeneracies on synthetic controls.
- Image-only correspondence enforces the shared usable-mask boundary and deterministic typed results.
- Joint matching can solve repeated-descriptor synthetic cases when a unique geometric minority exists.
- DS-19/DS-23 support-first selection reaches 9/9 only with oracle correspondences. This demonstrates
  solver capability, not an image-only production result.

## Current failures

1. Repeating Minecraft textures make generic descriptor neighbours indistinguishable. PyCOLMAP and
   ratio/mutual filters either return false identity-like poses or remove the geometry-bearing evidence.
2. DS-24 image-derived variants produced only 1/1/0 usable results of nine and several false solves;
   the oracle selector does not transfer to these match sets.
3. The production pair solver's fixed residual-margin yaw rule can reject the true class even when its
   support and residual pass all other gates.
4. DS-25 graph-wide closure is not discriminative: T2 forms no cycles; T6's three tracks are false and
   pass a vacuous null-vector closure. Do not revive it as-is.
5. `estimate_session` pools focal by resolution, invalid when equal-sized images have different FOV.
6. No production component turns camera/plane solutions into a `BlockVolume`; schema 1 persists no
   cameras, lattice, alternatives or uncertainty.

Avoid generic descriptor retuning, current DS-25 closure, PyCOLMAP two-view pose, free-space carving,
fixture-pose substitution, F3/OCR, or relaxed gates. They do not address the measured failure.

## Minimal packages

### CP-1 — multi-image calibration entry

Create a shared normal-mode camera entry that calibrates every ordinary screenshot independently,
preserves real frame IDs and intrinsics/FOV, emits per-frame typed abstentions, and records unresolved
position/scale/yaw quadrant. CLI and desktop must both accept N images through this entry. No metadata
or same-resolution focal pooling.

Decisive tests: equal-resolution/different-FOV frames do not pool; one unmeasurable frame does not erase
the others; deterministic real frame IDs; no manifest/F3/coordinate/bounds imports or parameters.

### CP-2 — connected camera component graph

Consume per-frame calibration, blind pair matches and accepted analytic pair poses. Build deterministic
connected components, retain only solved edges, anchor one local gauge per component, propagate relative
quadrant/translation direction, and expose isolated frames and unsupported axes as typed abstentions.
Do not call unit baseline metric scale.

Decisive tests: chain, loop, disconnected and contradictory-edge synthetic graphs; frame permutation;
independent gauge equivalence; failed edges never connect components; scale remains unknown; exact
reprojection/cheirality against an analytic oracle.

### CP-3 — correspondence evidence suited to block repetition

Replace descriptor-only identity with image-derived plane/lattice regions, boundary junctions and
non-repeating structural marks. Pair evidence must keep competing relative-yaw hypotheses and abstain
when repeated surfaces supply no identity. DS-19/23 may remain an oracle for solver verification only.

V4 remains blocked until CP-1/CP-2 provide normal-mode connected cameras, CP-3 supplies sufficiently
pure image correspondences, and the M4 selector-to-volume bridge exists. Manual camera manifests, F3
coordinates/directions, bounds and reference schematics remain advanced/developer/evaluation-only.
