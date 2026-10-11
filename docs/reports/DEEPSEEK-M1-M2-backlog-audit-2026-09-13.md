# M1/M2 Current Backlog Audit

Date: 2026-09-13
Role: DeepSeek read-only/no-shell audit, pending lead acceptance of individual packages
Status: current-source backlog inventory; no evaluation or implementation performed

## M1 gaps

1. **Usable-mask contract:** reconstruction normalizes a missing frame to no usable pixels, while
   `export.visibility` still treats a missing mapping key as unmasked. The evaluator and reconstruction
   denominator must have identical missing-frame semantics.
2. **Frozen mask gate:** current derived GUI masks are explicitly provisional. Reviewed per-frame masks
   with immutable provenance/hashes remain required before promotion.
3. **Matched baseline:** stored R6 masked results predate the lighting correction; later lighting
   diagnostics are unmasked. They cannot form the required pre-M2/current comparison. T2 and T6 still
   require a single harness, evaluator, mask policy and settings record.
4. **Reporting closure:** directional state rotation and per-state breakdowns now exist, but the
   supported gravity/alignment restriction and process/native versus Python-only memory measurement
   must be explicit.
5. Shared `FrameObservation`/`LatticeScene`/`ReconstructionResult` production construction and schema
   migration remain planned integration work rather than evidence of M1 acceptance.

## M2 gaps

1. **Exact-model routing (high):** production `sample_visible_faces` calls omit `blocks` and `palette`
   (`raycast.py:725-744`), so sampling follows cube quads while residual/shape/visibility paths can use
   exact model faces (`face_sampling.py:300-305,374-405`).
2. **Texture ownership:** representative-material pixels are still gathered through the representative
   candidate's ownership mask instead of each candidate's exact render (`shape_fit.py:810-821`).
3. **Resolution contract:** visible-face sampling, residual rendering and shape fitting use scales 1.0,
   0.35 and 0.5 respectively. The claimed agreement is unproven; a scale-invariance contract or one
   shared resolution policy is required.
4. `render_volume` retains a zero-sized-buffer risk at degenerate scale; align it with the existing
   one-pixel floor used by the sampler.
5. M2's required matched T2/T6 baseline/candidate evidence is absent for the same reason as M1 item 3.

Already fixed: residual usable-mask forwarding, candidate-independent silhouette probes, fractional
sampler buffer flooring, cardinal/rail rotation alignment, and common mask-object forwarding when a
mapping is provided. Do not reopen these packages without contradictory evidence.

## Work packages

### W1 — mask semantics

Make a missing frame in `visibility` mean no usable pixels, matching pixel authority. Test that the
same incomplete mapping yields identical zero visibility in evaluator and reconstruction consumers.

### W2 — mask promotion

Freeze reviewed per-frame v2 masks and their derivation/configuration hashes. Provisional derived-mask
runs must be impossible to label frozen or promotable.

### W3 — exact-model sampling

Thread `blocks` and `palette` into both production `sample_visible_faces` calls. Preserve an explicitly
reported cube fallback. Test a cut-out/non-cube state whose exact ownership differs from its cube.

### W4 — texture and scale authority

Use each candidate's own exact ownership for texture evidence. Reconcile 1.0/0.35/0.5 render scales
or prove scale-invariant decisions on an analytic control.

### W5 — matched acceptance run

After W1-W4, score pre-M2 and current T2/T6 outputs in one reproducible run under the same evaluator,
frozen masks, settings, alignment policy and lighting ablation. Publish direct/aligned geometry and
exact-state metrics, per-state losses, wall time and process/native memory evidence.

### W6 — buffer floor

Give `render_volume` the same one-pixel minimum dimension and test degenerate scale.

Dependencies: W1/W2 precede W5; W3/W4 precede a meaningful current candidate; W6 is independent.
P0 spacing currently blocks P1/V2 directly. M2 acceptance blocks trustworthy production rendering
and later GPU promotion.

## Product boundary

Normal mode remains image-derived. Camera manifests, F3 values, coordinates, directions, bounds,
reference schematics and other supplied scene data remain advanced/developer/evaluation-only. Any
mask work must preserve this boundary and label assisted reporting explicitly.
