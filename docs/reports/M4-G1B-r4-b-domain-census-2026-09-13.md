# M4 G1B: R4 scale-conflict census and bounded selector proposal

Date: 2026-09-13

## Scope and freeze

This is a read-only graph-input census plus proposed semantics for review. It does not edit production solver code, change the graph objective, inspect PNG pixels or labels, run the evaluator, or run R5. The census script strict-loads only the R4 freeze/sidecar and the G1A baseline/exact JSON. It replays reading identity with reading_order_key and derives active match records from the serialized best and runner-up quadrant assignments.

The R4 freeze and sidecar hash to 96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d and f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac. The G1A exact and pre-edit baseline diagnostics hash to 4c0117015c9f21766c1bae671648c795d5d9cfa0b370a3098e96cc0da38bc681 and b122a1f2989b1d90a7b43c84ab18dc601865f479e078e3007f205e4201eed3d2. G1A solver source/tests/diagnostic/preregistration remain at their accepted hashes. The accepted G1A protected-hash record was checked against all 20 R1–R4/E1–E3/image artifacts; every current value matches. The strict census is deterministic: two in-memory reconstructions serialize to the same bytes as the published JSON.

## Measured domains and multiplicity

R4 has 8 frames, 33 indexed readings, 200 match records, and 71,216 refusals. Across all 200 candidate records, 32 readings are touched and 24 have more than one distinct match-demand blocks_per_period value. The size histogram for touched readings is 8 with one b, 8 with two, 5 with three, 5 with four, 4 with five, 1 with six, and 1 with seven. These are match-demand domains; the larger intrinsic reading-alternative domains are separately preserved in the JSON.

Grouping records by (left frame, left plane, right frame, right plane, quadrant_of_right) gives 198 groups from 200 records. Two groups are duplicated, both between frame 01 plane 4 and frame 03 plane 0, with relative quadrants 1 and 3. Each has two different b-pairs: (0.125, 0.5) and (0.25, 0.25). Thus a repeated pair key does not mean two copies of the same b evidence.

The best and runner-up quadrant vectors differ only at frame 02 (q=1 versus q=3) and have costs 60.04393342612463 and 60.04393342612466 (margin 2.842170943040401e-14). Each activates 45 match records in 44 unique pair/quadrant groups and touches the same 27 readings; 19 readings have multiple b demands. Their raw record-index sets differ by 18 because quadrant-tagged match rows differ, but the normalized reading-pair/b evidence and every per-reading b domain are identical when that quadrant tag is ignored. Both give one connected active reading component per world axis:

| World axis | Match records / unique pair groups | Reading nodes | Multi-b nodes | Product of per-node active match-demand domain sizes | Product of intrinsic alternative-domain sizes |
|---:|---:|---:|---:|---:|---:|
| 0 | 9 / 8 | 8 | 5 | 64 | 145,152 |
| 1 | 26 / 26 | 11 | 10 | 216,000 | 870,912,000 |
| 2 | 10 / 10 | 8 | 4 | 36 | 4,915,200 |

For the best assignment, the one duplicated group is the frame01-plane4 to frame03-plane0 group at quadrant_of_right=1. Across its 43 single-record groups, 18 readings receive different forced b values from distinct incident groups. One additional multi-valued reading is explained by the two b-pair alternatives in the duplicated group. Last-writer-wins therefore produces a single reported value for each of 27 readings while hiding 19 disagreements. duplicate_b detects the duplicated pair group, but not conflicts across different groups.

Intrinsic local support does not supply a strong default scale choice: the highest-support alternative is b=1 for 31 readings and b=0.5 for 2, but the top/second support ratio ranges from 1.0000965595 to 1.6549787169 (median 1.0679948157). This confirms that choosing a fixed b or applying a newly invented local support margin is not justified by this census.

The G1A exact solver still terminates ambiguous_quadrant; resultants are 0.498715/0.245562/0.411718, below the existing 0.90 minimum on every axis. No R5 is authorized.

## G1B-R joint-label selector design for review

The candidate variables are one quadrant q_f per frame (with the existing first-frame gauge fixed to zero) and one b_r per reading keyed by (frame_id, plane_index) under reading_order_key. A reading's domain is the sorted union of b values in its incident candidate records, intersected with its intrinsic alternatives. A full label assignment induces all records satisfying relative-quadrant, world-axis, and both endpoint-b constraints. Records are grouped by a fixed universe of 73 endpoint reading-pairs from R4; the 200 serialized rows are variants of those groups (46 groups have two variants and 27 have four). The selector does not independently choose, drop, or duplicate edges. Distinct same-orientation regions remain distinct through their reading-pair keys.

For each full assignment, compute the induced distinct pair-group set, covered frame/readings, per-axis group coverage, and unchanged geometry/appearance quantities. The objective implementation must preserve the existing least-squares translation, face resultant, residual, and mean-match-score definitions and existing terminal gates; G1B does not authorize threshold or weight changes. Structural coverage is reported and used by the following Pareto rule, not turned into an all-node, connectivity, or cycle prerequisite. Valid trees and disconnected evidence components therefore remain eligible if the unchanged solver semantics accept them.

Candidate A may dominate B only when all of these hold:

1. A's induced distinct pair-group set is a superset of B's (equality is allowed), its frame coverage is no worse, and for each world axis its induced group coverage is a superset.
2. On the same shared pair groups (the intersection of their induced groups, with the same world-axis assignment), A's existing geometry/appearance objectives are no worse: each comparable resultant is at least B's, each corresponding residual is no greater, and the mean score over every induced match record in that shared group set is at least B's. Refit translations and recompute resultants/residuals on that common evidence using the existing objective definitions; compare appearance by its existing score sum/count, not a group-normalized substitute or full-set means with different denominators.
3. For strict dominance, at least one structural or objective dimension is strictly better. Exact ties are retained in an ambiguity certificate, not resolved by traversal or lexical order.

Disjoint or tradeoff signatures are incomparable and cause abstention when they remain Pareto-maximal. Empty candidate sets, unresolved objective dimensions, or multiple incompatible Pareto maxima return typed b_unresolved; existing quadrant ambiguity remains ambiguous_quadrant, including the observed q=1 versus q=3 tie. Coverage alone never selects a candidate. This rule permits an evidence superset to dominate, without treating a merely different edge set as automatically inferior. A semantically false node that is exactly indistinguishable on all available image evidence cannot be rejected by this selector; the correct outcome is tie/abstention, not a fabricated confidence penalty.

Search is joint over every anchored q vector and the compatible b domains, not restricted to G1A's top two quadrant assignments. Decomposition is allowed only where no discrete variables are shared and the exact combination of sufficient statistics reproduces the joint candidate: pair-group and per-axis coverage sets, per-axis least-squares normal equations/face values, match-score sums/counts, and ambiguity certificates. Otherwise components remain in the joint search. Stable ordering is by frame id, plane index, numeric q/b values, then canonical pair key. Permuting input sequences must not change the result or serialized certificate.

## Deterministic work budget and profile gap

Budget OPEN: no numeric node/state cap is approved by G1B-R. The earlier 262,144/4,194,304 proposals were derived from R4 domain products and are withdrawn. A later implementation must define deterministic counters for expanded joint-label states, record-variant compatibility checks, per-axis least-squares rows/solves, and Pareto/certificate comparisons. Crossing any preregistered limit returns component_too_large for the whole component, with no partial solution. The limits must be selected only after profiling the implemented full selector, including bound construction, induced-set maintenance, shared-evidence objective recomputation, and ambiguity-certificate updates, against the documented end-to-end runtime target. A compatibility-only probe is not sufficient evidence for a cap. Keep a representative worst-case tie/incomparable workload in that profile, not just a strongly prunable case.

## Synthetic verification contract for a later implementation

Before any pixels-only run, compare against a small exhaustive joint-label oracle covering: a unique coherent candidate; no compatible full label; exact q+2 ambiguity; an accepted valid tree; disconnected evidence; a false-node competitor; a dense but geometrically worse candidate; repeated pair groups with distinct b variants; strict-superset dominance; disjoint and tradeoff incomparable candidates; and exact objective ties. Assert that all compatible records are induced and no candidate edge can be manually deleted. Include a case where a dense wrong signature loses on shared geometry/appearance, and a case where a perfectly indistinguishable false node remains unresolved. Prove any decomposition against the joint oracle, or test that shared variables force a joint solve.

On existing all-singleton-b fixtures, route to the unchanged legacy solver and assert field-exact parity. Test stable output under frame/reading/match permutation, repeated-run byte determinism, bounded memory, and a low deterministic budget that returns a typed whole-component abstention without partial alternatives. Record separate counts for each work counter in the later full-stage profile. A writer-schema guard for the separate plane_index/axis serialization issue remains a separately reviewed compatibility change.

## Census artifacts and hashes

- Strict census script: scripts/diagnose_m4_g1b.py, SHA-256 5df641aaa68640fd18440c8e6dc5f46ae99b7ae102fbbeccf2331c5003c1e8f4.
- Census JSON: exports/evaluation/m4-g1b-r4-b-census.json, 44,077 bytes, SHA-256 7d56a458402a0ce1ab6c68abc1792172ef9a26a90274081acddca225e4c4919c.
- Existing graph implementation remains unchanged: src/mc_photogrammetry/reconstruction/plane_graph.py, SHA-256 961459b5a9f7fcb640448cd74be804e03a1b2c1b24a4522fdf7fb89395bd1e16.
- src/mc_photogrammetry/reconstruction/frozen_image_runner.py is unchanged, SHA-256 17408ac780bf27bd25bd8819a02fcf534e77e940b741a075c0a3c7f5bcdab60e.

Secondary interface finding, not edited here: ComponentSolution.blocks_per_period is keyed by (frame, plane_index), while frozen_image_runner._solution_record serializes the second tuple member under "axis". The current R4 freeze has zero saved b rows because its saved result was component_too_large; the G1A diagnostic reconstructs the matches and correctly emits plane_index. Correcting the writer requires a separately reviewed additive schema/compatibility change.

The diagnostic script compiled and ran successfully; a second independent reconstruction matched the saved census byte-for-byte. No production test was run because no production code changed. R1–R4, E1–E3, the capture inputs, and labels remain untouched. No R5 was run.
