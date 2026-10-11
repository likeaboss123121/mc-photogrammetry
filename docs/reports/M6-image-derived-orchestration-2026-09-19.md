# M6 image-derived orchestration facade — 2026-09-19

Bounded DS-M6 task: one new orchestration module, its test module, and this
report. No accepted source, schema, store, CLI/UI, or `AGENT_COMMS.md` file was
edited. This revision is the DS-M6 **final** repair round that answers Sol's two
remaining findings; the earlier four rejection blockers were repaired in the
prior revision and are retained below.

## Deliverables

| Path | SHA256 |
| --- | --- |
| `src/mc_photogrammetry/pipeline/image_derived_orchestration.py` | `3085850DFB168EDF01B22F76B1F4A16F39D7A8E1631462D34A7D870EA426407D` |
| `tests/test_image_derived_orchestration.py` | `1AE157F8DB975D66695610954E33A76DDE0511BC34E9B60F580ABFF6E8FC26B8` |
| `docs/reports/M6-image-derived-orchestration-2026-09-19.md` | this file; its SHA256 is taken by the reviewer from the delivered content |

## Final repair round

Two blockers were repaired in the scoped source/test paths only.

1. **Deep nested yaw-hypothesis reconstruction.**
   `RelativeCameraEvidence.__post_init__` (and the enclosing graph constructor)
   only `isinstance`-check `hypotheses`, so a `YawClassDiagnostic` forged past its
   own `__post_init__` - a negative `support`, an out-of-range `yaw_class`, a
   non-integer count, a non-boolean `passed`, a non-string `refusal`, or a
   non-finite/typed diagnostic value - used to ride an evidence digest that was
   resealed to match it. `_replay_connection_evidence` now deeply reconstructs
   every nested hypothesis through `connections_mod.YawClassDiagnostic`, compares
   the reconstructed canonical payload field for field with the stored one, and
   only then rebuilds and accepts the enclosing evidence record. A malformed
   nested diagnostic refuses typed (`invalid_record`); a well-formed one still
   composes.
2. **`PipelineOutcome` summary/provenance cross-binding.**
   `PipelineOutcome.__post_init__` now cross-binds every summary scalar to the
   record that owns it: `exact_state_eligible`, `export_ready`,
   `omitted_unknown`, `scale_state` and `normalization_state` must equal the
   carried `PromotionResult`'s own facts, `unresolved` must equal the carried
   `ImageDerivedPoseGraph.unresolved`, and a carried `DeliveryResult` must bind
   the same `promotion_sha256` and a `block_count` equal to the promoted cell
   count. A forged summary that reseals `outcome_sha256` around itself still
   refuses.

## Sol repair round (prior revision, retained)

All four blockers were repaired in the two scoped code paths only.

1. **Nested graph and pose-graph replay before an enclosing reseal is trusted.**
   `_replay_graph` now rebuilds the enclosing `ImageDerivedConnectionGraph`
   through its accepted constructor and then, after its own digest replay,
   re-runs `ConnectionBudgets` and every `RelativeCameraEvidence` through their
   accepted constructors. Each evidence record's invariants (status/refusal
   reason agreement, unit baseline direction, proper rotation, finite
   diagnostics, `evidence_sha256`) and its `session_sha256` binding are
   re-checked, and the graph's declared `max_pairs`/`max_total_support` caps are
   re-asserted against its own pair count and support sum.
   `_replay_pose_graph` likewise rebuilds `ImageDerivedPoseGraph`,
   `PoseGraphBudgets` and every `GaugeCameraPose` through their accepted
   constructors, re-checks each pose's graph binding, component replay and the
   declared `max_edges`/`max_cycle_checks` caps. A nested record that violates
   its own constructor, a nested budget whose `budget_sha256` no longer replays,
   a stale nested session/graph binding, and an incoherent component set now all
   refuse even when the enclosing digest was resealed to match.
2. **Every carried outcome record is bound and required.** `PipelineOutcome`
   now rejects a missing or type-invalid carried record (including `delivery`)
   and requires each record's own declared digest to equal its matching scalar
   field, so `PipelineOutcome.build` refuses a `None` or swapped record even when
   every scalar hash is a valid, self-consistent digest. The outcome payload and
   digest additionally carry a canonical `record_bindings` map over every
   carried record's own payload, so an internal change that keeps the declared
   scalar still changes `outcome_sha256`.
3. **`PipelineRefusal.detail` is deeply immutable and replayable.** Detail is
   normalized exactly as before (so the digest and `to_record` are unchanged in
   value), then frozen: nested mappings become `MappingProxyType` proxies and
   nested sequences become tuples. Item assignment and in-place mutation fail,
   `to_record()` emits a deep plain JSON copy, and `PipelineRefusal.build` over
   that copy reproduces the same `refusal_sha256`.
4. **Promotion replay no longer reaches into the delivery module.**
   `_replay_promotion_result` rebuilds through
   `promotion_mod._record_fields(result)` and `PromotionResult.build`, with no
   `delivery_mod._record_fields` reference left in the module.

## What the facade owns

`compose_image_derived_pipeline(...)` is the single entry point. It consumes only
accepted, digest-bound records plus the exact source records their seams were
built from:

* `ImageDerivedSession`, `ImageDerivedConnectionGraph`, `ImageDerivedPoseGraph`;
* `PlaneVolumeResult` plus the original `PlacementUniverse`,
  `AlignedRegionSupport` records, `ResolvedFacePlacement`, `PlacementBudget`;
* `PlaneFaceObservationResult` plus its immutable `FrameRectificationContext`
  records and `PlaneFaceObservationBudget`;
* an explicit `PaletteColor` tuple, the explicit proposal thresholds/budgets,
  and an explicit `PromotionBudget`.

It derives the `FaceStateProposalResult` and `PromotionResult` through the
accepted proposal and promotion seams, and calls
`deliver_image_derived_litematic` **only** for an exact-state-eligible
promotion. Without a `destination` it stops after the promotion and returns the
promotion on the outcome for a later, equally gated delivery. No camera, match,
pose, volume, observation or proposal is re-solved, and the entry point has no
parameter for a manifest, F3 reference, coordinate, direction, bound, scale,
annotation or scene transform. The public signature and `__all__` are unchanged.

## Validation and replay

Before any composition work, in a fixed order:

1. **Session** — round-tripped through the accepted
   `image_derived_session_to_record`/`image_derived_session_from_record` pair;
   every frame's retained pixels are re-checked as immutable `bytes`-backed
   `uint8` arrays and re-digested directly, because a record cannot recompute
   pixels.
2. **Connection graph / pose graph** — canonical payload digests recomputed and
   compared with the stored digests, then the enclosing result and every nested
   budget/evidence/pose record rebuilt through its accepted constructor and every
   session/graph/component/source binding and declared budget cap re-checked.
   Every nested `YawClassDiagnostic` is additionally reconstructed through its
   own constructor and payload-compared before the enclosing evidence is accepted.
3. **Volume and M5 result** — the volume's payload digest is recomputed and the
   accepted M5 bridge reruns accepted V3 from the original
   universe/supports/placement/budgets; the re-derived observation set must equal
   the provided one record for record. A resealed, fresh, swapped or tampered
   result is a typed refusal.
4. **Contexts and budgets** — every `FrameRectificationContext`, the placement,
   volume and face budgets and the promotion budget are re-digested through
   their accepted constructor/payload helpers.
5. **Cross-record provenance** — graph→session, pose-graph→graph/session,
   volume→placement, M5→volume/placement/frames, frame contexts→session frames,
   every V3 support-binding frame→session frames.
6. **Proposal and promotion** — derived through the accepted adapters; a
   supplied `expected_promotion` is replayed through the promotion module's own
   payload/constructor path and must bind the composed volume, proposal digest
   and promotion digest.
7. **Delivery** — the accepted delivery seam is called only for an
   exact-state-eligible promotion; an optional `expected_delivery` must bind the
   promotion and the artifact the seam just wrote.

Scale and normalization stay explicitly unresolved end to end
(`scale_known`/`metric_scale_known is False`, promotion `scale_state` and
`normalization_state` `"unresolved"`, `export_bounds is None`); the volume's own
`normalization_offset` is copied, never inferred.

## Typed immutable outcomes

* `PipelineOutcome` — frozen, digest-bound composition record carrying the
  accepted records, the derived proposal/promotion, the optional delivery, the
  closed eligibility flags and a `record_bindings` map over every carried
  record; `.to_record()` is deterministic and the digest replays from it.
* `PipelineRefusal` — frozen, digest-bound, closed reason vocabulary
  (`invalid_input`, `manual_metadata_forbidden`, `invalid_record`,
  `digest_mismatch`, `source_mismatch`, `unresolved_scale`, `proposal_refused`,
  `promotion_refused`, `delivery_refused`, `artifact_mismatch`), never embeds
  partial cells, blocks, artifacts or observations, and stores its detail as a
  deep immutable structure.

A non-empty `manual_metadata` mapping is refused with
`manual_metadata_forbidden` before any record is inspected.

## Tests

`tests/test_image_derived_orchestration.py` (**97 tests**) drives the real
accepted chain: the genuine V3 mixed-normal recipe from
`tests/test_plane_placement.py` supplies universe/supports/placement/volume and
the accepted M5 bridge supplies the observations; the accepted connection and
pose-graph seams supply the graph with the declared injectable `matcher` seam
(no SIFT, no disk IO).

Coverage:

* **Hostile tamper** — forged session pixels past the constructor, resealed
  session record, tampered graph/pose-graph/volume/M5/context payloads, a
  self-consistent M5 result over a resealed volume, and forged placement, volume
  and face budget digests.
* **Nested-reseal replay (blocker 1)** — a resealed graph over a
  constructor-bypassed evidence record (non-unit baseline), a tampered nested
  connection budget, a stale nested evidence session binding, a self-consistent
  budget whose caps the graph exceeds, a resealed pose graph over a
  constructor-bypassed pose rotation, a tampered nested pose budget, a
  self-consistent pose budget the source graph exceeds, and an incoherent
  component set.
* **Nested yaw-hypothesis replay (final repair 1)** — a well-formed injected
  `YawClassDiagnostic` replays and composes (positive control); ten forged nested
  diagnostics (negative/fractional `support`, negative `spatial_bins`,
  out-of-range and boolean `yaw_class`, non-boolean `passed`, non-string
  `refusal`, and string/boolean diagnostic values) refuse as `invalid_record`
  even when both the evidence digest and the enclosing graph digest are resealed
  around them.
* **Summary cross-binding (final repair 2)** — a mismatched
  `exact_state_eligible`, `export_ready`, `omitted_unknown`, `scale_state`,
  `normalization_state` or `unresolved` refuses both through
  `PipelineOutcome.build` and through `__post_init__` after `outcome_sha256` is
  resealed; a coherent delivery builds; and a delivery that binds another
  promotion digest or a forged `block_count` refuses.
* **Swapped inputs** — a coherent second session, a graph and pose graph built
  from a different session, a fully resealed volume binding a different
  placement, reordered frame contexts, and a swapped M5 result.
* **Outcome record binding (blocker 2)** — `record_bindings` covers all eight
  carried records; `PipelineOutcome.build` refuses every missing carried record;
  it refuses a swapped session/promotion that keeps a valid scalar hash; and a
  constructor-bypassed volume that keeps its declared digest still fails the
  outcome digest replay.
* **Refusal immutability (blocker 3)** — nested detail item assignment, nested
  proxy mutation and tuple mutation all fail; `to_record()` returns a deep plain
  copy; rebuilding from that copy reproduces the refusal digest.
* **Promotion replay source (blocker 4)** — no `delivery_mod._record_fields`
  reference remains and `promotion_mod._record_fields` is used; a promotion
  resealed past its constructor is refused through the promotion module as
  `invalid_record`.
* **Review / eligibility** — the recipe's default 15-unknown-cell promotion is a
  typed review; opted-in omission composes material states that remain
  `exact_state_eligible=False`; a proposal-level and a result-level abstention
  refuse; an ineligible promotion with a destination is refused as a typed
  review without touching the destination; the accepted delivery seam refuses
  the same ineligible promotion independently; delivery is never attempted for
  an ineligible promotion (spy seam, zero calls).
* **Manual metadata** — every declared forbidden key rejects with
  `manual_metadata_forbidden`, the guard precedes a broken record, the entry
  point consumes no forbidden key, and the module declares no CLI/UI/store
  surface.
* **Outcome discipline** — frozen/digest-bound outcome and refusal records,
  deterministic repeat composition, no mutation of any accepted input, exact
  entry-point signature.

## Checks run (`.venv`, direct shell)

* `py_compile` on the source and test modules: passed.
* Focused M6 module: **97 passed** (exit 0).
* Accepted adjacent suites — session (45), connections (24), pose graph (29),
  plane face observation (24), appearance state proposals (36), state promotion
  (43), image-derived delivery (81), plane volume (51), plane placement (310),
  litematic (8), project store (30): **681 passed** (exit 0).
* Fix sensitivity (temporary local revert of only the two repair hunks, restored
  and re-verified by hash): 22 of the new regressions failed without the fixes -
  all 10 forged-nested-hypothesis cases, the `exact_state_eligible`,
  `export_ready`, `omitted_unknown` and `unresolved` summary mismatches (build
  and resealed forms), and both delivery cross-binding cases - confirming the
  tests exercise the repaired guards; the two `scale_state`/`normalization_state`
  cases are additionally covered by the pre-existing unresolved-state guard.

## Accepted upstream hashes (verified unchanged)

| Module | SHA256 |
| --- | --- |
| `image_derived_session.py` | `4E842CF3B49C38CD6334254C48CE886F8129A43EC5F2EDF87F8676876C3A772A` |
| `image_derived_connections.py` | `A2D166BF711D6CD77C169F044DA5CA28D00073E66E60B9BFF992572FB56BA211` |
| `image_derived_pose_graph.py` | `B80B75034FF133AB8DD25EA328081863C97ABD7969C9D03E35B751E01447A74E` |
| `plane_face_observation.py` | `D86E64642B8CA6CCB3173D295B247BC06E56D443F345758F29B3FDB2D7364168` |
| `appearance_state_proposals.py` | `39ECFC70415472A7770A76FE492CC251D42233A43A557ECBF60305AE8846876A` |
| `state_promotion.py` | `EDA22E079162669CBB820DA647118AC263ED7FC37D40BBA1D41481621E53100B` |
| `image_derived_delivery.py` | `3EC50B8B934340C4A36EB3F8EBD2B8218CCD56B3471733145BB2F07CC21ABFA4` |

Independent integrity check of these seven accepted modules: 0 mismatches.

## Known coverage boundary

The genuine recipe retains 15 unknown cells, so its composed promotion is
ineligible and this module cannot reach its own successful-delivery branch
without fabricating support coverage, which was deliberately not done. The
facade's controls prove the gate (typed refusal; the accepted delivery seam is
never called for an ineligible promotion), and the accepted delivery suite
covers the successful branch with its own genuinely eligible promotion. A
production fixture with full support coverage is the natural next addition.

Production promotion/wiring remains blocked pending Sol review and the separate
M4 image-only producer integration.
