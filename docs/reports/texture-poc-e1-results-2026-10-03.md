# Texture POC E1 results — 2026-10-03

## Scope and evidence

E1 generated evaluation-only exact-model face patches from the reviewed T2 and T6
screenshots, supplied camera manifests, scene transforms, reference Litematics,
vanilla 1.21.5 JAR, and frozen per-frame masks. It did not run the normal
image-derived camera/geometry pipeline, train a model, or change production code.
All camera/scene/JAR pins, reviewed frame/reference/mask pins, and artifact seals
validated. The five camera/scene/JAR digests match the immutable pins in the
[preregistration](texture-poc-e1-prereg-2026-10-03.md).

The R4 rerun now uses the same full `provider.samples` visibility palette as
`benchmark_templates.py`; face ownership still uses reference-specific exact
models. E1 T6 visibility now matches the production baseline's 2,007-cell domain.

The run initially refused before decoding because the harness requested
`timestamp.png.json` while the tracked mask is `timestamp.json`. R3 corrected
same-stem mapping, added dotted-timestamp and missing-spec regressions, and
passed 39 focused tests plus `py_compile`. Independent preflight then accepted
all seven frames and masks. No evaluation input, pin, mask, or production module
was changed.

| Fixture | Frames | Frozen face patches | Distinct labeled cells | Visible cells | Geometry-labelable | Canonical-vocabulary cells | Scorer-usable cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2 | 3 | 1,168 | 577 | 667 | 667 | 599 | 577 |
| T6 | 4 | 4,025 | 1,866 | 2,007 | 1,978 | 1,920 | 1,866 |

"Face patch" is one face observation, so cells with multiple visible faces
contribute multiple patches. The scorer successfully scored all patches; top-K
recall uses only truth-in-canonical-palette patches. The pattern term is enabled
with `use_pattern=true`, weight 40.0.

## Existing scorer on the frozen patches

| Fixture | Truth-in-vocabulary patches | Out-of-vocabulary patches | Top-1 | Top-8 | Top-16 | Top-32 |
|---|---:|---:|---:|---:|---:|---:|
| T2 | 958/1,168 | 210 | 363/958 (37.89%) | 737/958 (76.93%) | 745/958 (77.77%) | 754/958 (78.71%) |
| T6 | 3,813/4,025 | 212 | 522/3,813 (13.69%) | 1,010/3,813 (26.49%) | 1,068/3,813 (28.01%) | 1,078/3,813 (28.27%) |

These are per-face, per-patch ordered rankings against the full canonical
palette. They are **not** per-cell production candidate recall or exact-state
recovery. Production aggregates multiple observations by coordinate, applies
support/locality pruning, selects top-8 with `argpartition`, smooths adjacent
states, and places blocks; E1 does none of those operations. E1 also uses no
production per-view light/calibration correction, self-calibration, simplicity,
shape weights, or per-view weighting. Therefore the patch top-K rows cannot be
compared numerically as improvements or regressions against historical cell rows.

The authoritative supplied-camera baseline remains T2 480/667 exact (71.96%)
and T6 350/2,007 (17.44%). In the T6 no-global-pruning ablation, production
top-8 set membership is 788/2,007 (39.26%); the separately ordered per-cell
ranking is 793/1,920 (41.30%) at K=8, 850/1,920 (44.27%) at K=16, and
932/1,920 (48.54%) at K=32. Those are different units, denominators, and
selection stages than E1's per-patch rows. See the [stage attribution]
(visible-state-stage-attribution-2026-10-03.md) and [candidate-depth report]
(visible-state-candidate-depth-2026-10-03.md).

## Reconciliation status and decision

The original R3 E1 T6 artifact counted 2,003 visible cells and 1,916
canonical-vocabulary cells. A controlled replay changed only the visibility
palette: with the same reference blocks, cameras, scene, masks, and renderer,
the reference-only geometry palette counted 2,003 while `provider.samples`, the
palette used by `benchmark_templates.py`, counted 2,007. The four restored cells
are `(0,17,21)` hay block, `(9,18,22)` deepslate iron ore, and `(12,20,11)` /
`(12,20,12)` chiseled resin bricks. Their own exact model metadata matches
across palettes; the difference arises from other reference states and
palette-dependent name/occlusion lookup. R4 uses `provider.samples` for the
visible-cell domain and retains reference-specific geometry only for face-label
ownership. Its sealed T6 result is now 2,007 visible and 1,920 in-palette cells,
matching the historical production denominators. Original R3 artifacts remain
immutable evidence of the pre-correction harness.

This correction resolves the four-cell visible-domain mismatch in the renderer
replay, but does not establish that E1 face patches are identical to production
face observations or that the per-patch scorer reproduces per-cell aggregation.
Until those exact observations and candidate-cost vectors are compared, the
patch metrics establish scorer behavior only, not a production loss attribution.

Do not train on these pinned evaluation fixtures. The current factory has five
finalized scenes but only three source-build groups, all indexed as `train`;
leave-one-group-out is exploratory, not a designated held-out test. The next
experiment should bridge E1 labels to a production trace: record each actual
production face observation and its candidate cost vector by frame/cell/face,
then replay aggregation, pruning, top-K selection, and smoothing on the exact same
samples. The visible-cell denominator now matches; compare the replayed candidate
and label stages to the sealed historical artifacts. After that, evaluate a compact local model as
an auxiliary proposer/ranker on identical observations, split by source build;
retain end-to-end pinned visible-state and geometry metrics as separate gates.

## Sealed artifacts

The internal E1 seal was independently recomputed for each artifact. File SHA-256
and embedded artifact seal are recorded separately:

| Artifact | Patches | Embedded artifact seal | File SHA-256 |
|---|---:|---|---|
| R4 `.tmp-texture-poc-e1-patch-labels-20261003-r4-template-6.json` | 4,025 | `d6a487a3ae3888104e619b53df6424e1c4cd836d4d25a3969ca81dcdfaf56ae9` | `661D537D36ABAC7CE95A8309D02CD3F80DF1A786E2A7E0215ECC16247EFA0D30` |
| R4 `.tmp-texture-poc-e1-patch-labels-20261003-r4-template-2.json` | 1,168 | `1f1ad4a7a936908d3a89349f9c68557dfd3f35c6a2e9f58be73df1081ceb847a` | `FB9FB5F90EAD1C9B0926184322DBA8F057F5BA95B2951CB8C58B651F2B068874` |
| R3 `.tmp-texture-poc-e1-patch-labels-20261003-template-6.json` | 4,025 | `1641acce3ad5629669fdc051ce61573ef751da2f3ee5e26efe77d988de1dc90e` | `8E1C209D809AA5CF31614614588B315E10CA56509D2D6B1B68F92CC85C43576A` |
| R3 `.tmp-texture-poc-e1-patch-labels-20261003-template-2.json` | 1,168 | `c4134c2d86b0601aacb779a9dd8da190ae4c3560ecca763d8ce91368813170e7` | `3D780241C64110A0000CE5A888ED7CACD2210B06EA175153CCAFC1A039B5453A` |

Both artifacts are local evaluation evidence only; neither is training data.
