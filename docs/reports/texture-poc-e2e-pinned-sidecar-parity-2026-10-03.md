# E2E pinned production-patch parity — 2026-10-03

## Scope

Evaluation-only, supplied-camera T2/T6 runs with the reviewed fixtures, JSON
mask specs, and pinned vanilla 1.21.5 JAR. Each run used fresh output, report,
schema-v2 trace, and E2C sidecar paths. The E2D reader checked trace/sidecar
binding, replayed aggregation, and joined resolved-face observations to the
sealed E1 R4 artifacts. No production settings or source changed, and no model
was trained. This does not measure image-derived camera/geometry performance.

The first T2 invocation accidentally omitted `--minecraft-jar` and used the
default palette; it reported 0% and is invalid for comparison. It is excluded
here. The accepted reruns recorded the pinned JAR SHA-256
`522672ad20b460c02c2e39b6c5035ef6a849af28eb11ab2ac7eb293d395a8c11`, had
`fixture_identity.pinned=true`, and had no pin mismatches.

## Current pinned checkpoints

| Fixture | Visible exact state | Geometry F1 | Dominant measured state loss |
|---|---:|---:|---|
| T2 | 480/667 (71.96%) | 91.88% | candidate-to-raw-winner: 102 lost; smoothing recovers 21 net |
| T6 | 350/2,007 (17.44%) | 85.39% | support pruning: 1,527 visible truth states removed |

These reproduce the accepted E2B baselines. T2 is 8.04 percentage points below
the 80% exact-state gate; T6 is 62.56 points below. T6's surviving 393/2,007
truth states at support are all available as cell candidates; the local model
therefore needs to improve pre-prune candidate support as well as ranking.

## E1 identity and pixel parity

| Fixture | E1 patches | Production resolved faces | Identity joins | E1-only | Production-only | Raw patch exact | Observed patch exact | Valid-mask exact | Coverage exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 | 1,168 | 1,158 | 857 | 311 | 301 | 0/857 | 0/857 | 712/857 (83.08%) | 0/857 |
| T6 | 4,025 | 4,113 | 2,664 | 1,361 | 1,449 | 0/2,664 | 0/2,664 | 2,092/2,664 (78.53%) | 0/2,664 |

Join keys are exact `(frame_id, coordinate, slot)` identities; E2D excludes
centre-patch fallbacks. Aggregation totals and weights replay exactly for all
1,832 T2 and 7,725 T6 coordinates. Zero digest equality does **not** mean the
RGB crops are unrelated; hashes are byte-exact, so small resampling differences
also fail equality.

## Controlled sampler comparison

The E1 harness defaults to `face_sampling.DEFAULT_SHRINK = 0.35`, while
production uses `raycast.FACE_SHRINK = 1.0`. For each fixture I generated a
fresh sealed E1 artifact changing only `shrink=1.0`; all pinned inputs stayed
fixed. E2F then compared decoded E1 patches to the already-captured production
sidecars. These controlled artifacts are diagnostic, not training data.

| Fixture / E1 shrink | E1 patches | Joined | Raw digest exact | Valid digest exact | Coverage digest exact | Raw RGB MAE median / p90 | Both-valid raw MAE median / p90 | Coverage MAE median / p90 | Mean valid-pixel agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 / 0.35 | 1,168 | 857 | 0/857 | 712/857 (83.08%) | 0/857 | 4.520 / 9.736 | 4.462 / 7.396 | 0.03061 / 0.12405 | 98.67% |
| T2 / 1.0 | 1,184 | 859 | 0/859 | 686/859 (79.86%) | 0/859 | 0.105 / 3.261 | 0.096 / 0.677 | 0.02502 / 0.09811 | 98.49% |
| T6 / 0.35 | 4,025 | 2,664 | 0/2,664 | 2,092/2,664 (78.53%) | 0/2,664 | 4.847 / 13.030 | 4.560 / 12.519 | 0.04321 / 0.15586 | 98.72% |
| T6 / 1.0 | 4,035 | 2,673 | 2/2,673 | 1,989/2,673 (74.41%) | 2/2,673 | 0.154 / 4.203 | 0.114 / 0.797 | 0.03810 / 0.13040 | 98.49% |

RGB MAE is the median/p90 of per-face means over 16x16 patch pixels and RGB
channels, in channel-value units. Both-valid MAE includes pixels valid in both
artifacts. Mean valid-pixel agreement weights every mask pixel equally; median
per-face agreement is 100% for all four rows. Exact digests remain byte-level
and stricter: most masks agree on nearly every pixel while 16-25% of whole-mask
digests differ. Matching shrink reduces T2 median/p90 raw RGB MAE by about
43x/3x and T6 by about 31x/3x, but does not ensure exact patch or coverage
parity. A minority of rows retain large errors (T6 shrink-1 raw RGB MAE p99
16.96), so the aggregate similarity does not excuse per-face outliers.

At 1e-7 scalar tolerance on E1's original shrink-0.35 rows, T2 `area` and
`depth` match on 854/857 and `facing` on 857/857; T6 matches on 2,659/2,664 for
area/depth and 2,664/2,664 for facing. This supports close projected face
size/depth/orientation for joined keys, while remaining pixel ownership,
coverage, and the unmatched roughly one-third of sample identities still need
attention.

## Decision and next work

The E1/production pixel-domain gap is substantially reduced by aligning shrink,
but matching that setting is not sufficient for exact byte parity. E2F is
implemented and independently accepted: 53 focused replay tests pass, and its
numeric fields remain separate from exact digest parity. E2F was applied to all
four immutable E1/production comparisons above; no second reconstruction
benchmark was needed.

For the local model, the narrowest aligned training example is the *actual
production sidecar patch* joined to E1's exact state label by the face key;
do not use E1 patch pixels as if they were byte-identical production inputs.
Keep T2/T6 evaluation-only. The factory has only three source-build groups,
all indexed `train`, so a model trained there can be exploratory but cannot
establish held-out generalization. Add distinct source-build groups before a
promotion claim. At T2, test reranking where truth commonly survives to the
candidate stage. At T6, test whether a learned proposer can preserve correct
states before global support pruning. Keep the image-only camera/geometry gate
separate from both appearance experiments.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| T2 benchmark report | `f2b17596d9734c5496d41c9f799f8be02cacc418ea22e2f15e63921ff73163e9` |
| T2 E2A trace | `4d94191272ba71cfaf99e3ea8dcf30015755660abf9aa1c768d84312043247f1` |
| T2 E2C sidecar | `e864560888911ebe6b64776c02f83e663e753297df33c428d4cd6c2d04b58cec` |
| T6 benchmark report | `c239c2ab941e87d5a743090b75be8810a4c71733d0d1b14e9b8cc30f6e8c21b1` |
| T6 E2A trace | `a4b01febe1aceb6ffbb651900a90575e8dec2b3ae007c9fad52e84f4721cfe73` |
| T6 E2C sidecar | `503399678bec47f3b87c4f81b7bd25a90b5657978b6396794a3836c0451ff895` |
| T2 E1 shrink-1.0 artifact | `ad075db476f2bb60722de3dd2b52a7c2169933600df610a737f96949309d47c1` |
| T6 E1 shrink-1.0 artifact | `9109062af89daf176dd9bd5a628741ad6e5fedc93ec45317fc9b4d04df38f641` |
| E2F replay reader | `26323da2e881e46cab53c22b06a203de859f9b3f71fde28f56003975878a2674` |
| E2F focused tests | `6c59d5cbcc7eb4dee082c60744bf5e162f75a5d4d03f0caba00ad92acafa8c99` |
