# E2 production observation bridge audit — 2026-10-03

## Scope and method

Read-only implementation audit. No production code, evaluation input, pin, mask,
threshold, corpus, artifact or dependency was changed, and no run was performed.
Findings are traced from source in
`src/mc_photogrammetry/reconstruction/raycast.py`,
`appearance_scoring.py`, `face_observation.py`, `face_sampling.py`,
`scripts/benchmark_templates.py` and
`scripts/evaluation/evaluate_texture_patch_labels.py`, read against
[texture-poc-e1-results-2026-10-03.md](texture-poc-e1-results-2026-10-03.md).

## 1. What production actually retains

The production observation record is the anonymous tuple appended to `raw`
(`raycast.py` lines 801, 879-883 and 909-913):

```
(frame_id, coordinate, rgb, descriptor, slot, weight, pattern, area)
```

`rgb` is the coverage-weighted mean of the retained valid pixels,
`descriptor` is `patch_descriptor(observed_patch, coverage)`, `pattern` is
`normalised_pattern(observed_patch, coverage)` for a rectified face and `None`
for a centre-patch fallback, and `area` is `0.0` for that fallback. The
rectified `patch`, the `valid` mask and the fractional `coverage` that
`sample_visible_faces` produced are never stored; they die at the call site in
the loop at lines 837-883. The fallback pass (lines 888-913) loses them too,
keeping only the median colour and a descriptor. Patch bytes are therefore
**not** recoverable from any production record, and neither is the per-cell
mask share (`valid_fraction`) that E1 records per patch.

Identity that survives inside one run: `frame_id`, `coordinate` and `slot` are
present per observation, and `raw` is de-duplicated on `(coordinate, slot)`
per frame by `seen_here` (lines 836-840). Identity that survives into the
trace: only aggregate. `stage_trace["coordinate_evidence"]` (line 970,
`_evidence_records`) keeps per-coordinate counts, frame ids, slot ids,
`resolved_area`, `total_weight`, and a `mean_rgb` — with no per-observation
split, no `(frame, slot)` key, and no sampled pixels. Crucially it is written
**before** `_uncast` and `_reconcile_exposure` (lines 972-1019), so its
`mean_rgb` is the *pre-correction* colour while the scored colour is the
*post-correction* one. Those are different numbers in the same trace.

Costs: `index.costs(...)` returns the full per-candidate vector and the
absolute best cost (`appearance_scoring.py` lines 200-272). It is called
repeatedly and thrown away: in `score()` (lines 1049-1079), again from
`_self_calibrate` (1760-1764), `_measure_response`, `_slot_brightness`
(2076-2079), `_scene_brightness` (2250-2253), `_explanatory_cost` (2288-2292)
and `_detect_emitters` (2335-2338). Only the last `score()` layer reaches
`observations`, and of that only `_accumulate_costs` produces the per-cell
`totals`/`weights` (lines 1311, 1939-1957), from which `merged` keeps the
production top-8 negated map (`_top_candidates`, lines 1334-1354, 1960-1968)
and the trace adds a deeper ordered copy. Per-observation cost vectors exist
only as transient locals; nothing keyed by `(frame, cell, slot)` is retained.

Rejections: `face_rejections` counts only the `low_contrast` and `uniformity`
drops raycast applies itself (lines 842-854). The sampler's own rejection
reasons are dropped because raycast passes no `rejections` mapping to
`sample_visible_faces` (lines 814-820), and the per-frame front pass discards
its reasons entirely. `diagnostics` expose only counts.

`FaceObservation` (`face_observation.py`) is the typed `(cell, slot,
frame_id)` seam with retained read-only `patch`, `valid` and `coverage`, and it
does preserve exactly what is missing. **The production raycast does not use
it**: it hand-rolls the tuple and calls `PaletteIndex.costs` directly.
`FaceObservation` is used only by `appearance_state_proposals` and by the E1
harness. So "does production preserve frame/cell/face identity and raw sampled
pixels?" answers: identity partially, pixels and masks no, cost vectors no.

## 2. Where E1 and production diverge already

E1 freezes frames × the reference's exact-model geometry
(`provider.palette_for_states(..., include_partial=True)`) sampled by the *same*
`sample_visible_faces`, but through a different input path and with different
post-processing:

- E1 builds its own `PatchSample` (frame, coordinate, slot, truth state,
  weight, area, depth, visible, facing, valid/coverage masks, patch bytes) and
  de-duplicates a `(frame, coordinate, slot)` group by
  `(valid_fraction, weight, -depth)` (`evaluate_texture_patch_labels.py`
  1017-1038), while production keeps the **first** occurrence per
  `(coordinate, slot)`.
- E1 refuses a patch below `MINIMUM_PATCH_VALID_FRACTION = 0.5`; production
  accepts any sample with one valid pixel and gates on `min_surface_contrast`
  and `max_face_variation` instead.
- E1 emits no centre-patch fallback observation at all; production adds one per
  uncovered cell per frame that saw it.
- E1 scores on the unpruned canonical full-cube palette
  (`provider.canonical_palette()`) with `use_pattern=true` and
  `TEXTURE_WEIGHT`, no per-view calibration, no light/cast correction, no
  simplicity or shape weight. Production scores after `_uncast`,
  `_reconcile_exposure` and `_self_calibrate`, and after up to two global
  support-prune rounds that rebuild the index and re-score
  (lines 1082-1146).
- E1's label state can be geometry-exact yet out of the canonical scoring
  vocabulary; production can never rank it either, but E1 counts it separately.
- E1's visible-cell denominator uses `provider.samples` while face ownership
  uses the reference-specific geometry palette (documented in the E1 module
  docstring and the R4 reconciliation).

Consequence: even with identical patch bytes, production's and E1's cost
orderings can differ legitimately (calibration and pruning), and the E1
per-patch top-K rows cannot be read as production candidate recall. That is
already stated in E1 §"Reconciliation status"; this audit locates the exact
mechanisms.

## 3. Minimal evaluation-only, output-neutral seam

The seam exists and is cheap to widen: `stage_trace` is a caller-owned mapping,
and nothing written into it is read back (module docstring; tests
`test_stage_trace_does_not_change_reconstruction_output_or_diagnostics` and
`test_stage_trace_candidate_limit_leaves_output_and_winner_identical`,
`tests/test_raycast.py` 475-490, 663-681). Adding records to it cannot change a
decision; only re-scoring or re-deciding would. So the minimal change is
**observation-level records inside the existing trace**, not a new code path,
not a new return value, and not a second reconstruction.

Shape of the extension (trace `schema` 1 → 2, additive; every existing key
keeps its meaning):

1. `observations`: one record per `raw` entry, after the cast/exposure/
   calibration passes and immediately before the existing `score()` call, in
   `raw` order, keyed `(coordinate, slot, frame_id)`.
   - identity and provenance: `coordinate`, `slot`, `frame_id`, and
     `source` ∈ {`resolved_face`, `centre_patch_fallback`} plus the fallback's
     `dominant_slot`;
   - sampler scalars as production used them: `area`, `depth`, `visible`,
     `facing`, `weight` (the post-uniformity weight actually passed in);
   - sampled-pixel evidence: `observed_patch` digest, `valid` fraction and
     `coverage` mean (and, behind an opt-in flag, the bytes themselves);
   - derived evidence: `rgb`, `descriptor` digest, `pattern` digest after the
     contrast/uniformity gates that produced them, so a replay can prove it
     recomputed the same inputs;
   - applied corrections: `light_factor`, `calibration` id/scale, `scene_gain`,
     `slot_factor`, `texture_weight_resolved`, so the scored `rgb`/`pattern` of
     each observation is reproducible rather than inferred.
2. `observation_costs`: one record per `score()` layer, keyed by the same
   `(coordinate, slot, frame_id)`, holding the observation's absolute best cost
   and a digest of its full cost vector under a declared index revision id.
   The digest proves row-by-row equality without serializing hundreds of floats
   per observation; a value-level sample (or full vectors behind the opt-in
   flag) settles any disputed row. The layers must be distinguishable, because
   `_self_calibrate`, `_slot_brightness`, `_scene_brightness` and
   `_explanatory_cost` each call `costs()` with different factors.
3. `aggregation`: the final per-cell `totals` and `weights` (or their exact
   float digests plus per-cell denominator counts), captured right after
   `_accumulate_costs` at line 1311, plus `view_weights`. This is the anchor:
   it is the exact production quantity from which `mean`, `_top_candidates`,
   the winner and smoothing were derived.
4. Header additions: `palette_revision` ids per prune round (so a cost digest
   can be attributed to the index that produced it) and the
   `stage_trace_candidate_limit` already present.

Boundedness: records are O(observations) small scalars and fixed-size digests;
the opt-in raw-bytes mode should be restricted to a caller-supplied coordinate
or visible-cell set so the trace stays a diagnostic, not a corpus.

Output-neutrality obligations, in order: nothing in the new code may be read
before `_emit_stage_trace`; no new ordering may affect `raw`, `observations`,
`merged`, the volume or `diagnostics`; digests must not mutate their inputs
(`np.ascontiguousarray`/`tobytes` only); the existing
`test_stage_trace_does_not_change_reconstruction_output_or_diagnostics`
signature and diagnostics equality must keep passing unchanged; and the E1
scorer-side constants (`TEXTURE_WEIGHT`, `AREA_POWER`, `UNIFORMITY_SCALE`) must
be read from `raycast`, not copied, as `observation_weight` already does.

## 4. Exact parity and denominator checks

Every row below is evaluator-side, joined post hoc against the sealed E1
artifact and the sealed stage trace, and each carries its own denominator.
None of them is a gate or a threshold change.

1. **Sampling parity (identity).** For the same fixture, frames, cameras,
   scene, usable masks and geometry mode, the set of
   `(frame_id, coordinate, slot)` keys in the production observation records
   versus E1's frozen `PatchSample.key`. Report intersection, E1-only, and
   production-only. Non-empty production-only sets are expected (fallback,
   first-occurrence dedup, production sample radius); non-empty E1-only sets
   are expected too (E1 keeps faces production's contrast/uniformity gate
   dropped). Denominator: each side's own key count, never a shared number.
2. **Patch equality (bytes).** On the intersection, `np.array_equal` of the
   rectified patch, `valid` and `coverage`, and of the `area`/`depth`/
   `visible`/`facing` scalars within 1e-6. Denominator: the intersection size,
   reported as the only domain where "E1 patch bytes equal production
   observations" is a factual claim.
3. **Cost vector agreement.** For each shared key and each declared score
   layer, the replayed cost vector's digest must equal the trace's; then the
   replayed `mean` must equal the trace's `totals/weights` within a tight
   relative tolerance (1e-9) and the replayed absolute best cost must match.
   Denominator: shared keys that formed a cost record in both (a cell can be
   missing from one side for the geometry reasons above).
4. **Selection parity (tie-aware).** Replay `_accumulate_costs` →
   `_top_candidates` → `raw_winner` → `_smooth_adjacent_states` →
   `_choose_state_with_local_family_support` → `_remove_enclosed_blocks` on the
   captured observations, with no re-scoring, and require: identical
   `top_candidates` set at K=8, identical `ranked_candidates` order to the
   traced depth, identical `raw_winner`, `post_smoothing_label`,
   `selected_state` and `final_state` per coordinate, and identical
   `diagnostics["merged_coordinates"]`. Ties at `OUTLIER_CEILING` must be
   handled by reproducing the same `argpartition` call, not by sorting — the
   K=8 vs ordered-K=8 five-cell delta in
   [visible-state-candidate-depth-2026-10-03.md](visible-state-candidate-depth-2026-10-03.md)
   is exactly this boundary and must be expected, not "fixed".
5. **Stage-chain parity.** The replayed chain's
   `available_palette`/`support`/`candidate`/`raw_winner`/`post_smoothing`/
   `placed` counts and per-stage loss cells must equal the sealed historical
   stage-trace rows for the same configuration, and the replayed replay must
   reproduce the E1-side `truth_in_canonical_palette` denominator
   (T6 1,920; T2 599) from the canonical vocabulary. Denominators stay
   separate: all-visible production cells (T6 2,007), truth-in-canonical-palette
   (T6 1,920), E1 scorer-usable patches (T6 1,866), E1 distinct labeled cells
   (T6 1,866). The E1 "scorer-usable" denominator is *not* the production
   visibility denominator and may not be substituted for it.
6. **Correction-chain parity.** Because `coordinate_evidence.mean_rgb` is
   pre-cast and the scored colour is post-cast, the replay must reproduce
   `_uncast` (when `scene_spread >= dim_spread_threshold`) and
   `_reconcile_exposure` gains from the recorded factors and match the scored
   `rgb`. A mismatch here invalidates any per-observation cost comparison even
   if patch bytes are equal.

## 5. Irreducible non-equivalence

These cannot be removed by instrumentation and must be reported as boundaries:

- **Geometry domain.** E1 samples the reference's own exact-model geometry;
  production samples only cells that survived the centre-pass surface,
  `_surface_of`, `proposed_cells`/stereo and `_remove_enclosed_blocks`, and it
  adds centre-patch fallback observations E1 never produces. Any cell E1 has
  and production lacks is a geometry/reconstruction difference, not an
  appearance difference, and vice versa. It is charged to geometry.
- **Vocabulary.** A truth state outside `provider.canonical_palette()` cannot
  be a production candidate at any K. E1 already counts it as
  out-of-vocabulary; the bridge must not add it to any candidate denominator.
- **Fallback observations.** Production's `centre_patch_fallback` rows have a
  real descriptor but no rectified patch and no pattern. They can be replayed
  in the aggregation but have no E1 counterpart; comparisons at those keys are
  undefined, not zero.
- **Post-sample corrections.** Light field, slot response, self-calibration,
  exposure reconciliation and the pattern `resolved = area/PATCH_SIZE²`
  weighting change the scored inputs. With the correction factors traced they
  are replayable, but they remain an input difference from E1, not an error.
- **Pruning/calibration asymmetry.** E1 ranks against the unpruned palette with
  no calibration. Identical patches may rank differently under production's
  pruned, calibrated index; those disagreements are legitimate scorer
  differences and must be reported as such, not as patch mismatch.
- **De-duplication and gating.** E1's largest-valid-fraction de-duplication and
  0.5 valid-fraction floor cannot be reconciled with production's
  first-occurrence and contrast/uniformity gates except by comparing on the
  intersection.

## 6. Recommended next bounded implementation task

**Task:** extend the stage trace with the observation-level records of §3 and
add an evaluation-only replay/parity reader for §4. No production behaviour
change, no threshold change, no corpus run by the implementer.

Owned paths:

- `src/mc_photogrammetry/reconstruction/raycast.py` — additive trace records
  only (`observations`, `observation_costs`, `aggregation`, header
  `palette_revision`), emitted inside `_emit_stage_trace`; `schema` 1 → 2 with
  v1 keys unchanged. No change to any decision, argument or ordering.
- `tests/test_raycast.py` — keep
  `test_stage_trace_does_not_change_reconstruction_output_or_diagnostics` and
  `test_stage_trace_candidate_limit_leaves_output_and_winner_identical`
  green unchanged; add: every `raw` observation has exactly one record; record
  count equals the `coordinate_evidence` observation count per coordinate; the
  recorded aggregate reproduces `totals/weights`; replaying
  `_accumulate_costs` + `_top_candidates` + `_smooth_adjacent_states` from the
  records reproduces the v1 `raw_winner`/`post_smoothing_label`/`final_state`
  for the same run; `json.dumps` round-trip equality; the opt-in raw-bytes mode
  is refused without an explicit key set.
- `scripts/evaluation/replay_production_observations.py` (new,
  evaluation-only) — loads the extended trace and the sealed E1 artifact,
  imports `raycast._accumulate_costs`, `_top_candidates`,
  `_smooth_adjacent_states` and `_choose_state_with_local_family_support` to
  replay selection, and emits the six metric families with their own
  denominators and a list of the irreducible boundaries it refused to compare.
- `tests/test_replay_production_observations.py` (new) — synthetic trace plus
  synthetic E1 artifact: intersection/E1-only/production-only accounting,
  byte-equality counting, degenerate `OUTLIER_CEILING` tie handling, and a
  fail-closed check that a mismatched cost digest is reported rather than
  silently re-scored.

Explicitly out of scope: `scripts/benchmark_templates.py` (its join semantics
are already sealed; a future task may add a passthrough flag), any new
reconstruction parameter, any change to `PALETTE_SUPPORT`, `TEXTURE_WEIGHT`,
`OUTLIER_CEILING`, `AREA_POWER` or `UNIFORMITY_SCALE`, any model training, and
any use of T2/T6 as training data. Stop after the parity report; do not lower a
threshold to make a parity row pass, and do not treat the E1 per-patch top-K
rows as a production baseline.
