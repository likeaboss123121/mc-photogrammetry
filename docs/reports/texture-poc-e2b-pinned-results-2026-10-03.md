# Texture POC E2B pinned replay results - 2026-10-03

## Scope and integrity

Lead ran the E2A schema-v2 stage trace on the reviewed default supplied-camera
T2 and T6 fixtures, then replayed each trace against its sealed E1 R4 label
artifact with the evaluator-only [E2B reader](texture-poc-e2b-replay-2026-10-03.md).
No reconstruction settings, thresholds, palettes, masks, fixtures or source code
changed. This is conditional on supplied cameras and reference-assisted geometry,
not an image-only product score. It changes no acceptance gate.

Both new run reports confirm `fixture_pinned=true`, the expected fixture role,
no mismatched pin fields, and identical reference/camera/scene/JAR/frame/mask
identity to the prior reviewed stage-attribution runs. The baseline scores are
unchanged. Both per-fixture gates remain eligible but fail the exact-state gate.

| Fixture | Visible exact state | Geometry F1 | Runtime | Acceptance |
|---|---:|---:|---:|---|
| T2 | 480/667 (71.96%) | 91.88% | 104.8 s | exact-state gate not met |
| T6 | 350/2,007 (17.44%) | 85.39% | 430.2 s | exact-state and geometry gates not met |

## Stage losses

| Fixture | Canonical-palette truth | After support pruning | Candidate | Raw winner | Smoothed / placed |
|---|---:|---:|---:|---:|---:|
| T2 | 599/667 (89.81%) | 562/667 (84.26%) | 561/667 (84.11%) | 459/667 (68.82%) | 480/667 (71.96%) |
| T6 | 1,920/2,007 (95.67%) | 393/2,007 (19.58%) | 393/2,007 (19.58%) | 329/2,007 (16.39%) | 350/2,007 (17.44%) |

T6's dominant measured loss is still global support pruning: it removes 1,527
visible truth occurrences before candidate generation. Candidate availability
does not lose additional visible truth after support in this default run.
T2's largest measured loss remains candidate-to-raw-winner ranking (102 cells).
Smoothing recovers 21 net cells on each fixture. These are supplied-camera
checkpoints, not a decomposition of the image-only camera/geometry pipeline.

## Production trace integrity and E1 sample join

The E2B reader replayed every observation cost vector and effective weight:
aggregation totals and weights match exactly for **1,832/1,832 T2** coordinates
and **7,725/7,725 T6** coordinates. Its independent production `argpartition`
check reproduced raw top-8 candidate sets for **1,832/1,832 T2** and
**7,725/7,725 T6** coordinates, including the existing tie behavior.

| Fixture | E1 face patches | Production resolved faces | Fallbacks | Same face keys | E1-only | Production-only | Valid-mask digest match | Coverage digest match | Observed-patch digest match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 | 1,168 | 1,158 | 2,688 | 857 (73.4% of E1) | 311 | 301 | 712/857 (83.08%) | 0/857 | 0/857 |
| T6 | 4,025 | 4,113 | 19,887 | 2,664 (66.2% of E1) | 1,361 | 1,449 | 2,092/2,664 (78.53%) | 0/2,664 | 0/2,664 |

“Same face key” means only `(frame_id, coordinate, slot)` identity with
`source=resolved_face`; centre-patch fallbacks are counted separately, even if
their key collides. E1 and production therefore do **not** provide interchangeable
patch datasets. On joined keys, no E1 patch reconstructed under production's
float32 `observed_patch` contract matched the production digest; coverage matched
none. Valid masks matched for about 79-83%. This confirms that identity alone
does not establish pixel-level sample parity. Known E1/production differences
include exact reference geometry versus production geometry, duplicate selection
(E1 chooses the strongest-validity sample; production keeps the first), and
different rejection/fallback paths. The digest result identifies a real seam but
does not by itself isolate which difference caused it.

## State ranking on joined observations

These ranks use only the final post-pruning palette and only joined E1-labeled
face keys whose truth state survives in that palette:

| Fixture | Joined keys | Truth in final palette | OOV / pruned | Per-observation top-1 | E1-labeled cells in final palette | Per-cell top-8 membership | Raw winner | Final state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 | 857 | 802 | 55 | 680/802 (84.79%) | 492/577 | 492/492 (100%) | 423/492 (85.98%) | 425/492 (86.38%) |
| T6 | 2,664 | 588 | 2,076 | 586/588 (99.66%) | 359/1,866 | 359/359 (100%) | 309/359 (86.07%) | 326/359 (90.81%) |

Both final palettes contain six states. Thus per-observation top-8 and per-cell
top-8 are saturated because K=8 exceeds the final palette size; they are not
evidence of broad candidate recall. T6's 99.66% top-1 is heavily conditioned on
surviving support pruning: only 588 of 2,664 joined face patches, and 359 of
1,866 labeled cells, have truth in the final palette. It does not contradict
the 1,527-cell default support loss. T2's corresponding top-1 result is 84.79%
on 802 surviving joined face patches.

The current evidence does not justify training a general classifier on E1 crops
and treating it as a production scorer. First align labels with production's
actual sampled patches (or export bounded production patch bytes for a strictly
evaluation-only labeled subset); keep per-build holdouts. If testing a learned
model after that, T6 calls for pre-prune candidate preservation/recall, while T2
offers a narrower reranking test. The model's score must be compared on identical
patches against the current scorer and then on the end-to-end visible-state gate.
The factory still has only three source-build groups, all marked train, so its
current splits are exploratory.

## Artifacts

| Fixture | Benchmark report SHA-256 | Schema-v2 trace SHA-256 | Candidate Litematic SHA-256 |
|---|---|---|---|
| T2 | `07c43f702cbdd9a3f900789aeb3119bdac42be20a3b6866e2d9e72b9c25aefcf` | `4d94191272ba71cfaf99e3ea8dcf30015755660abf9aa1c768d84312043247f1` | `71d2df7e89199dceea197612a2653e5fa16f04ac61308c2dfc82e27f7a5a3a97` |
| T6 | `5c4935c0abdfac094fbc090bcbe1ae9f6978da9d01be79eacc6759ebe09dc942` | `a4b01febe1aceb6ffbb651900a90575e8dec2b3ae007c9fad52e84f4721cfe73` | `9b8b05f953a7f1579a98f23d4099b069dfa14a07fa1515e5df793cec8e794269` |

Output paths are the six new `.tmp-texture-e2b-{t2,t6}-v2-{report,trace,output}`
files in the repository root. T2 trace size is 17,730,214 bytes; T6 trace size is
95,321,595 bytes. The larger traces are evaluation diagnostics only. No model was
trained and no source code or fixture input changed during these runs.
