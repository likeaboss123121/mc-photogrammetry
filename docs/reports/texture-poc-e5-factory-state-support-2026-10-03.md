# E5 factory state-support / split feasibility audit (evaluation-only)

- Task ID: `texture-poc-e5-factory-state-support-2026-10-03`
- Worker: DeepSeek (Sol-equivalent) under the direct-shell Defender-safe profile
  in [docs/deepseek-worker.md](../../docs/deepseek-worker.md)
- Status: analysis-only read complete; **stopped for lead review**
- Verdict: **data support is too thin for a defensible held-out recognizer
  experiment today.** This report measures label support only; it does not
  measure recognizer performance.

## 1. Scope and owned paths

Owned and written path (only this file was written):

| Path | Role |
|---|---|
| `docs/reports/texture-poc-e5-factory-state-support-2026-10-03.md` | this report |

No other file was created, edited or overwritten. In particular, no code, test,
production module, configuration or data artifact was touched, no model was
trained, and no JSON/CSV output artifact was emitted. The audit is a read of the
four accepted sealed E3 artifacts only. The E4 report was consulted for
provenance, and screenshots, references, camera manifests, scenes, the JAR and
the T2/T6 fixtures were not read.

## 2. Method

### 2.1 Seal verification before trusting any record

All four artifacts were loaded with the accepted E4 reader
(`scripts/evaluation/audit_factory_texture_patch_consistency.py`,
`load_artifact`), which performs the two-pass admission contract: the artifact
self-seal and all non-array metadata are verified for every example, and every
per-example seal plus all three per-array (`patch`, `valid`, `coverage`) seals
are recomputed **before** any array is decoded. No refusal occurred: all four
artifact seals, all 8,516 example seals and all 25,548 array seals verified.

The reader returns `Observation` records whose `state_key` is the canonical
normalized exact BlockState `(name, properties)`. No label was trusted from an
unverified record.

### 2.2 Identity and grouping definitions

- **Exact state / class**: the normalized exact BlockState, i.e. block name
  *plus* its full property set (`minecraft:cobblestone_stairs` with
  `facing=... half=... shape=... waterlogged=...` is distinct from every other
  facing/half/shape variant). This is the `state_key`.
- **Build**: the E4 grouping pair `(scene_group, world_identity)`. Two builds
  exist, and the artifacts' declared folds are used only as recorded.
- **Capture**: one recorded screenshot, keyed by `screenshot_sha256`. This is
  the only globally unique capture key: `frame_id` repeats across the two
  artifacts of a build and `frame_key` includes the artifact prefix. The four
  artifacts contain 17 distinct screenshots and no screenshot is shared by two
  artifacts.
- **Leave-one-build-out (LOBO)**: for each of the two builds, train on the other
  build, test on the held-out build, and count the exact states in the test
  build that never occur in the train build ("OOV", out of vocabulary).

### 2.3 What this report deliberately does not do

- It does **not** infer label purity or correctness from repeatability. The E4
  consistency proxy shows that repeated exact faces look similar; it cannot show
  that they are correctly labelled, because a consistently occluded or
  mislabelled face agrees with itself by construction. Nothing here is a label
  quality gate.
- It does **not** estimate recognizer performance. It only enumerates how much
  supervision exists and how it partitions across builds and captures.
- It does not change, fit or tune any threshold, and it does not rerun the
  exporter, the E4 auditor, or any benchmark.

## 3. Inputs (as read) and seal results

| Artifact file | file SHA-256 | internal `artifact_sha256` | examples |
|---|---|---|---|
| `exports/texture-poc-e3/fresh-c4-canal-town-2026-09-12.json` | `9ec453beb82ee5a64e07074275e9142b3cd43e0e0fd994848ee9bd79d61cfef4` | `fb737108a659ed377ea201c66f69fcdf3ae46d224c99f651c9bfbc7524135fa7` | 1,303 |
| `exports/texture-poc-e3/fresh-c4-canal-town-framed-2026-09-12.json` | `206a92219923ccefbf2b3286d5282372bd0680b658b4df1f2979cf28fa04986a` | `97a3a64c06cf811e85ea9e0796de3a6348c10f5eb3324db17dc953af047537b0` | 2,333 |
| `exports/texture-poc-e3/pilot-batch-train-station-2026-09-12.json` | `a04b2db2dbca3503be55e6ff62740c22152c41f062e235f9c83acdb751f32114` | `1a469f6913b2fe3850bca0ea13fdf627cd367e952f4b4154c8667ca239694965` | 2,507 |
| `exports/texture-poc-e3/smoke-train-station-ds20c-2026-09-12.json` | `f71cce99ae0a819edf62ce3f0c9611925be8a9bb5a1d60cc5197c95d49313e44` | `fae14c754b1e3523c5838a3f6f6c7403a7d4cb0bae08a7aa7cff2d68329a4aaa` | 2,373 |

The file digests and internal seals are byte-identical to the values the lead
recorded when it accepted E4; the reader reported zero seal failures.

## 4. Corpus inventory

| Quantity | Value |
|---|---|
| artifacts | 4 |
| admitted examples | 8,516 |
| independent builds (`scene_group` = `world_identity`) | **2** |
| `scene_group` values | 2 |
| `world_identity` values | 2 |
| staging tokens (`build.staging_token`) | 2 |
| `scene_id` values | 3 |
| `dataset_id` values | 4 |
| captures (distinct `screenshot_sha256`) | 17 |
| distinct `frame_key` | 17 |
| distinct `frame_id` (not globally unique) | 13 |
| folds present | `train` only |
| exact states (classes) | 63 |
| distinct block names (property-insensitive) | 47 |

No screenshot digest appears in more than one artifact (0 of 17). Every example
belongs to exactly one build, and no build/fold leakage is introduced here
because the audit never invents a split.

## 5. Exact-state class frequencies

### 5.1 Overall class-size distribution

| Class size (examples) | classes | examples |
|---|---|---|
| 1 (singleton) | 4 | 4 |
| 2-4 | 11 | 29 |
| 5-9 | 7 | 45 |
| 10-49 | 18 | 376 |
| 50-99 | 4 | 316 |
| 100-999 | 17 | 4,506 |
| 1000+ | 2 | 3,240 |
| **total** | **63** | **8,516** |

The distribution is strongly concentrated and long-tailed: the single largest
state (`minecraft:dirt`, 1,788 examples) is 21.0% of the corpus, 4 states cover
52.6% of examples, 12 states cover 80.7% and 19 states cover 91.0%. Conversely
22 states (35% of classes) have fewer than 10 examples and together contribute
78 examples (0.9%).

### 5.2 Per-build frequencies (exact state)

| Build | examples | captures | artifacts | `scene_id` values | states | singletons | dominant state (share) |
|---|---|---|---|---|---|---|---|
| Train station `1757a5ad…` | 4,880 | 5 | 2 | 2 | 42 | 1 | `dirt` 1,772 (36.3%) |
| Canal Town `74f080aa…` | 3,636 | 12 | 2 | 1 | 25 | 3 | `cobbled_deepslate` 1,452 (39.9%) |

Full exact-state table (rows sorted by total examples). `captures` is the number
of distinct screenshots in which the exact state was observed; the Capture
columns are example counts by build.

| Exact state | Properties | Total | Train | Canal | captures |
|---|---|---|---|---|---|
| dirt | — | 1,788 | 1,772 | 16 | 13 |
| cobbled_deepslate | — | 1,452 | 0 | 1,452 | 12 |
| cobblestone | — | 653 | 167 | 486 | 17 |
| grass_block | snowy=false | 586 | 586 | 0 | 5 |
| smooth_quartz_slab | type=bottom,waterlogged=false | 520 | 0 | 520 | 12 |
| gravel | — | 437 | 437 | 0 | 5 |
| andesite | — | 340 | 147 | 193 | 16 |
| bricks | — | 323 | 323 | 0 | 5 |
| smooth_quartz | — | 205 | 0 | 205 | 10 |
| granite | — | 197 | 197 | 0 | 5 |
| green_concrete_powder | — | 197 | 197 | 0 | 5 |
| white_concrete | — | 177 | 0 | 177 | 6 |
| short_grass | — | 165 | 165 | 0 | 1 |
| stone | — | 151 | 73 | 78 | 15 |
| polished_granite | — | 138 | 138 | 0 | 5 |
| cobblestone_stairs | facing=west,half=top,shape=straight,waterlogged=false | 108 | 0 | 108 | 10 |
| jungle_planks | — | 107 | 107 | 0 | 5 |
| dead_horn_coral_block | — | 101 | 101 | 0 | 5 |
| dead_tube_coral_block | — | 101 | 101 | 0 | 5 |
| anvil | facing=south | 93 | 0 | 93 | 8 |
| dead_brain_coral_block | — | 84 | 84 | 0 | 4 |
| cobblestone_stairs | facing=east,half=top,shape=straight,waterlogged=false | 76 | 0 | 76 | 8 |
| polished_andesite | — | 63 | 0 | 63 | 11 |
| blackstone_slab | type=bottom,waterlogged=false | 38 | 38 | 0 | 2 |
| dark_oak_slab | type=bottom,waterlogged=false | 33 | 33 | 0 | 2 |
| polished_blackstone_brick_slab | type=bottom,waterlogged=false | 31 | 31 | 0 | 2 |
| fern | — | 28 | 28 | 0 | 1 |
| cobbled_deepslate_stairs | facing=east,half=bottom,shape=straight,waterlogged=false | 27 | 0 | 27 | 11 |
| stone_brick_slab | type=bottom,waterlogged=false | 25 | 0 | 25 | 5 |
| cobbled_deepslate_stairs | facing=south,half=bottom,shape=straight,waterlogged=false | 23 | 0 | 23 | 10 |
| cobbled_deepslate_stairs | facing=west,half=bottom,shape=straight,waterlogged=false | 23 | 0 | 23 | 11 |
| smooth_quartz_slab | type=double,waterlogged=false | 22 | 0 | 22 | 6 |
| cobbled_deepslate_stairs | facing=north,half=bottom,shape=straight,waterlogged=false | 21 | 0 | 21 | 10 |
| smooth_sandstone_stairs | facing=north,half=top,shape=straight,waterlogged=false | 21 | 21 | 0 | 1 |
| andesite_slab | type=bottom,waterlogged=false | 17 | 17 | 0 | 2 |
| brick_slab | type=bottom,waterlogged=false | 15 | 15 | 0 | 4 |
| andesite_stairs | facing=north,half=bottom,shape=straight,waterlogged=false | 11 | 11 | 0 | 3 |
| stone_slab | type=bottom,waterlogged=false | 11 | 11 | 0 | 2 |
| oak_wood | axis=y | 10 | 10 | 0 | 2 |
| smooth_quartz_stairs | facing=north,half=top,shape=straight,waterlogged=false | 10 | 0 | 10 | 3 |
| stone_stairs | facing=north,half=bottom,shape=straight,waterlogged=false | 10 | 10 | 0 | 2 |
| polished_granite_slab | type=bottom,waterlogged=false | 9 | 9 | 0 | 2 |
| andesite_stairs | facing=south,half=bottom,shape=straight,waterlogged=false | 8 | 8 | 0 | 3 |
| smooth_quartz_stairs | facing=south,half=top,shape=straight,waterlogged=false | 6 | 0 | 6 | 2 |
| smooth_quartz_stairs | facing=west,half=top,shape=straight,waterlogged=false | 6 | 0 | 6 | 3 |
| smooth_sandstone | — | 6 | 6 | 0 | 2 |
| polished_diorite | — | 5 | 5 | 0 | 3 |
| stone_stairs | facing=south,half=bottom,shape=straight,waterlogged=false | 5 | 5 | 0 | 2 |
| dead_fire_coral_block | — | 4 | 4 | 0 | 3 |
| smooth_sandstone_stairs | facing=west,half=top,shape=straight,waterlogged=false | 4 | 4 | 0 | 1 |
| smooth_quartz_stairs | facing=east,half=top,shape=straight,waterlogged=false | 3 | 0 | 3 | 1 |
| smooth_sandstone_stairs | facing=east,half=top,shape=straight,waterlogged=false | 3 | 3 | 0 | 1 |
| smooth_sandstone_stairs | facing=west,half=top,shape=inner_right,waterlogged=false | 3 | 3 | 0 | 1 |
| andesite_slab | type=top,waterlogged=false | 2 | 2 | 0 | 1 |
| blackstone_stairs | facing=west,half=top,shape=straight,waterlogged=false | 2 | 2 | 0 | 2 |
| brick_stairs | facing=south,half=top,shape=straight,waterlogged=false | 2 | 2 | 0 | 1 |
| polished_blackstone_bricks | — | 2 | 2 | 0 | 1 |
| sandstone | — | 2 | 2 | 0 | 2 |
| smooth_sandstone_stairs | facing=east,half=top,shape=inner_left,waterlogged=false | 2 | 2 | 0 | 1 |
| brick_stairs | facing=north,half=top,shape=straight,waterlogged=false | 1 | 1 | 0 | 1 |
| end_rod | facing=down | 1 | 0 | 1 | 1 |
| stone_bricks | — | 1 | 0 | 1 | 1 |
| stone_button | face=wall,facing=north,powered=false | 1 | 0 | 1 | 1 |

Nine block names carry property variants that split into separate exact states
(`smooth_quartz_slab` 2, `smooth_quartz_stairs` 4, `cobblestone_stairs` 2,
`cobbled_deepslate_stairs` 4, `andesite_stairs` 2, `stone_stairs` 2,
`andesite_slab` 2, `brick_stairs` 2, `smooth_sandstone_stairs` 5). Because the
task's unit is the exact state, these variants are separate classes even though
a name-level model would merge them.

### 5.3 Per-capture frequencies

| Build | `frame_id` | examples | exact states | top state (share) | screenshot |
|---|---|---|---|---|---|
| Train station | view-train-station-spawn-208-96-orbit-a-02 | 807 | 20 | dirt 380 (47.1%) | `005e29a296…` |
| Canal Town | view-canal-town-architectural-component-bbox-02 | 302 | 18 | cobbled_deepslate 74 (24.5%) | `0bf2002547…` |
| Canal Town | view-canal-town-architectural-component-bbox-03 | 358 | 17 | cobbled_deepslate 116 (32.4%) | `1472dfe2de…` |
| Train station | view-train-station-spawn-208-96-orbit-a-05 | 1,520 | 31 | dirt 446 (29.3%) | `1e2319d0f0…` |
| Canal Town | view-canal-town-architectural-component-bbox-02 | 274 | 14 | cobbled_deepslate 153 (55.8%) | `2d017c8f80…` |
| Train station | view-train-station-spawn-208-96-orbit-a-04 | 180 | 12 | dirt 83 (46.1%) | `370f19e634…` |
| Train station | view-train-station-spawn-208-96-01 | 852 | 24 | dirt 408 (47.9%) | `3ad1d7003c…` |
| Canal Town | view-canal-town-architectural-component-bbox-00 | 298 | 17 | cobbled_deepslate 92 (30.9%) | `48ba0ff58c…` |
| Canal Town | view-canal-town-architectural-component-bbox-06 | 287 | 14 | cobbled_deepslate 159 (55.4%) | `53c38526d3…` |
| Canal Town | view-canal-town-architectural-component-bbox-05 | 231 | 13 | cobbled_deepslate 140 (60.6%) | `6f844d5e36…` |
| Canal Town | view-canal-town-architectural-component-bbox-01 | 404 | 18 | cobbled_deepslate 137 (33.9%) | `8750bb1973…` |
| Canal Town | view-canal-town-architectural-component-bbox-04 | 205 | 12 | cobbled_deepslate 113 (55.1%) | `8797d70e0e…` |
| Canal Town | view-canal-town-architectural-component-bbox-00 | 313 | 17 | smooth_quartz_slab 83 (26.5%) | `8ea5dc8910…` |
| Train station | view-train-station-spawn-208-96-00 | 1,521 | 32 | dirt 455 (29.9%) | `ad3f59ab05…` |
| Canal Town | view-canal-town-architectural-component-bbox-01 | 330 | 16 | cobbled_deepslate 119 (36.1%) | `bcc01ac8fa…` |
| Canal Town | view-canal-town-architectural-component-bbox-03 | 413 | 17 | cobbled_deepslate 146 (35.4%) | `e71bfda3ec…` |
| Canal Town | view-canal-town-architectural-component-bbox-07 | 221 | 11 | cobbled_deepslate 138 (62.4%) | `fdd8a37094…` |

The four `bbox-00/01/02/03` `frame_id` values each appear twice because the two
Canal Town artifacts (`fresh-c4-canal-town` and `fresh-c4-canal-town-framed`)
render the same nominal viewpoints into different screenshots; each rendered
screenshot is its own capture here.

## 6. Build overlap and leave-one-build-out

Exact-state vocabulary overlap between the two builds:

| Quantity | Value |
|---|---|
| states in Train station only | 38 |
| states in Canal Town only | 21 |
| states in both builds | **4** (`andesite`, `cobblestone`, `dirt`, `stone`) |
| union | 63 |
| Jaccard | 0.0635 |

Name-level overlap (merging property variants) is also exactly those 4 names, so
the property split is not the cause of the mismatch.

| Held-out build | train build(s) | train states | test states | shared states | OOV states | OOV examples | OOV share | seen examples | seen share |
|---|---|---|---|---|---|---|---|---|---|
| Train station | Canal Town | 25 | 42 | 4 | 38 | 2,721 / 4,880 | 55.76% | 2,159 / 4,880 | 44.24% |
| Canal Town | Train station | 42 | 25 | 4 | 21 | 2,863 / 3,636 | 78.74% | 773 / 3,636 | 21.26% |

Shared-state support in both builds simultaneously:

| Minimum examples per build | shared exact states |
|---|---|
| ≥1 | 4 |
| ≥5 | 4 |
| ≥10 | 4 |
| ≥25 | 3 |
| ≥50 | 3 |
| ≥100 | 2 |

The two `dirt` counts are extremely asymmetric (1,772 Train vs 16 Canal), so
even the largest-vocabulary shared class has almost no held-out Canal Town
support. The residual held-out classes are dominated by build-specific material
(`cobbled_deepslate` 1,452 and `smooth_quartz_slab` 542 for Canal Town;
`grass_block` 586, `gravel` 437, `bricks` 323 for Train station).

## 7. Rare-class and capture concentration

### 7.1 Rare classes

| Threshold | classes | examples in those classes |
|---|---|---|
| exactly 1 example | 4 | 4 |
| ≤4 examples | 15 | 33 |
| <10 examples | 22 | 78 |
| ≤19 examples | 29 | 162 |

The 4 singletons are `brick_stairs facing=north,half=top,shape=straight`,
`end_rod facing=down`, `stone_bricks`, and `stone_button
face=wall,facing=north,powered=false`; 3 of the 4 are Canal Town-only. A closed
vocabulary of 63 exact states therefore cannot be trained from this corpus even
within a single build.

### 7.2 Capture concentration

| Quantity | Value |
|---|---|
| states observed by exactly 1 capture | 15 / 63 |
| states observed by ≥2 captures | 48 / 63 |
| per-state top-1 capture share, median | 0.50 |
| per-state top-1 capture share, mean | 0.55 |
| states where one capture supplies 100% of examples | 15 |
| states where one capture supplies ≥75% of examples | 18 |
| states present in exactly 1 build | 59 / 63 |
| states present in both builds | 4 / 63 |

Per build:

| Build | states | states in exactly 1 capture | states in ≥2 captures | examples in single-capture states |
|---|---|---|---|---|
| Train station | 42 | 11 | 31 | 233 |
| Canal Town | 25 | 4 | 21 | 6 |

Capture concentration is high: for 18 of 63 states a single screenshot supplies
at least three quarters of the supervision, and 15 states are never seen twice.
Any within-build evaluation would need capture-grouped folds to avoid leakage,
and the thin states would remain untestable.

## 8. Eligible labelled examples

"Eligible" here means structurally usable supervision that survived seal
admission; it is **not** a statement that training is authorized. Every E3
artifact declares `evaluation_only=true` and `training_permitted=false`, and the
E3 protocol excludes T2/T6.

| Denominator | examples |
|---|---|
| admitted examples (all labelled) | 8,516 |
| `valid_fraction` ≥ 0.5 | 8,516 |
| `valid_fraction` ≥ 0.75 | 8,321 |
| `valid_fraction` = 1.0 | 2,111 |
| in classes with ≥2 examples | 8,512 |
| in classes with ≥10 examples | 8,438 |
| in the 4 classes shared by both builds | 2,932 |
| per build: Train station admitted / ≥0.75 / ≥10-example classes / shared classes | 4,880 / 4,826 / 4,820 / 2,159 |
| per build: Canal Town admitted / ≥0.75 / ≥10-example classes / shared classes | 3,636 / 3,495 / 3,618 / 773 |

The mask quality is not the limiting factor: every admitted example has a valid
mask covering at least half of the patch, and 98% cover at least three quarters.
The limiting factors are the small number of independent builds and the tiny
shared vocabulary.

## 9. Feasibility assessment

**No defensible held-out recognizer experiment is possible from this corpus
today.** Concretely:

1. **Only 2 independent builds exist.** Every leave-one-build-out split has a
   single training build, so there is no way to separate a build effect from a
   model effect, no second training build to check stability, and no way to
   report a held-out spread. The artifacts are 4 files but only 2 builds
   (2 staging tokens, 2 `scene_group`s, 2 `world_identity`s).
2. **The build vocabularies barely overlap.** 4 shared exact states out of 63
   (Jaccard 0.0635). Holding out Train station leaves 38/42 states (55.8% of
   examples) unseen in training; holding out Canal Town leaves 21/25 states
   (78.7% of examples) unseen. A closed-set classifier can only be scored on the
   minority of held-out examples that belong to the 4 shared classes, and one of
   those (`dirt`) has just 16 Canal Town examples.
3. **The shared classes are also the least distinctive.** The 4 shared names are
   common terrain/building blocks; a cross-build accuracy computed only on them
   would not represent recognizer quality on the architectural classes that
   motivate the work.
4. **Rare and capture-concentrated classes dominate the tail.** 22 of 63 states
   have fewer than 10 examples, 15 are seen by exactly one capture, and 59 are
   present in only one build. Even a within-build grouped split could not
   evaluate those classes.
5. **All folds are `train`.** The factory declares no dev/test fold, and no
   build-held-out split has been registered. Any split would have to be created
   and preregistered; it cannot be inherited from the current factory.

What *is* possible today is a deliberately scoped diagnostic that would be
labeled as such: a 4-class (`andesite`, `cobblestone`, `dirt`, `stone`)
cross-build closed-set sanity check with explicit OOV abstention, or a
within-build capture-grouped split for one build. Neither supports a
generalization or performance claim, both would be dominated by the 56-79% OOV
mass if reported over all held-out examples, and the two builds differ in scene
type and lighting as well as block vocabulary. This report does not recommend
running either as a model experiment before more data exists.

### Distinguishing data support from recognizer performance

Everything in sections 4-8 is **data support**: how many labelled examples
exist, how they are distributed over exact states, builds and captures, and how
much of a held-out build's vocabulary is already represented in a training
build. None of it measures whether a recognizer can classify a patch. No
accuracy, precision, recall, calibration or ranking number is computed or
implied. Label support is necessary but not sufficient for a model experiment,
and label support says nothing about label correctness (section 2.3).

## 10. Minimum additional independent data

Ordered by what unblocks the next defensible experiment:

1. **Independent builds, not more frames.** At least **3 independent source
   builds** are required before a leave-one-build-out split has more than one
   training build; **4-5 builds** are recommended so a held-out result has any
   stability at all. Additional captures of the existing Canal Town or Train
   Station worlds do not create independence.
2. **A shared state vocabulary.** Additional builds must actually share exact
   states with the others. Current shared support is 4 states. A workable
   target is **at least 10 exact states shared across at least 3 builds, each
   with at least 50-100 examples per build**, so that every held-out build has
   non-trivial in-vocabulary mass. Adding more builds with disjoint vocabularies
   (the current pattern: Train station 38 exclusive states, Canal Town 21
   exclusive states) will not reduce OOV.
3. **Capture coverage per shared state.** Each shared state should be observed
   by **at least 3 distinct captures per build** (preferably distinct
   viewpoints/lighting), so that a capture-grouped split inside a build is
   possible and the state is not a single-screenshot artifact. Today 15 states
   are single-capture.
4. **Rare-class policy.** Either add coverage until the tail states reach a
   usable count, or preregister an explicit abstention/ignore policy. 22 states
   have fewer than 10 examples; they cannot be evaluated as classes now.
5. **Falsification criterion for the next audit.** Re-run this same
   seal-verified audit after adding builds. If a held-out build still has more
   than ~50% OOV examples, or fewer than ~10 exact states with ≥50 examples in
   at least two builds, a closed-set build-held-out experiment remains
   indefensible and the effort should go to coverage instead.
6. **Label-purity work remains separate.** Even after the support targets are
   met, E4's consistency proxy does not establish that repeated faces are
   correctly labelled. Region-clipping and occluder contamination must be
   quantified separately before any quality claim.

## 11. Limitations

- The audit reads only the four named E3 artifacts. It does not re-read source
  screenshots, so it cannot detect label errors introduced by the exporter; it
  only counts what the sealed exporter emitted.
- Exact-state identity is the normalized `(name, properties)` record. Name-level
  analyses (which merge property variants) are reported only as a supplementary
  check; the primary numbers are exact-state.
- `screenshot_sha256` is used as the capture identity. `frame_id` is not unique
  across artifacts (13 distinct for 17 captures), and the two Canal Town
  artifacts render the same nominal viewpoints into distinct screenshots.
- The corpus is 3 `dataset_id` groups within 2 builds, all `train`, all captured
  on one date by one pipeline; the build count, not the file count, is the
  independent-sample count.
- Counts are exact and deterministic for the sealed inputs read, and no
  threshold was fitted or tuned. No pass/fail model bar is defined here.
- This report does not authorize training. The artifacts themselves set
  `training_permitted=false`, and T2/T6 remain excluded.

## 12. Checks run

- Seal-verified load of all four E3 artifacts through the accepted E4 reader
  (`load_artifact`, two-pass admission): 4 artifact seals, 8,516 example seals
  and 25,548 array seals verified, 0 refusals. File and internal seal digests
  match the lead-accepted E4 table.
- All frequencies, overlap, LOBO, rare-class, capture-concentration and
  eligibility numbers were computed in one deterministic foreground read of the
  sealed artifacts. No JSON/CSV/other artifact was written; only this report was
  created.
- No production/runtime code, test, threshold, gate, exporter, benchmark or
  sealed artifact was modified, and no corpus/evaluator run was performed.

## 13. Hashes

| Artifact | SHA-256 |
|---|---|
| `exports/texture-poc-e3/fresh-c4-canal-town-2026-09-12.json` | `9ec453beb82ee5a64e07074275e9142b3cd43e0e0fd994848ee9bd79d61cfef4` |
| `exports/texture-poc-e3/fresh-c4-canal-town-framed-2026-09-12.json` | `206a92219923ccefbf2b3286d5282372bd0680b658b4df1f2979cf28fa04986a` |
| `exports/texture-poc-e3/pilot-batch-train-station-2026-09-12.json` | `a04b2db2dbca3503be55e6ff62740c22152c41f062e235f9c83acdb751f32114` |
| `exports/texture-poc-e3/smoke-train-station-ds20c-2026-09-12.json` | `f71cce99ae0a819edf62ce3f0c9611925be8a9bb5a1d60cc5197c95d49313e44` |

The SHA-256 of this report is recorded by the lead after review.

## 14. For lead review

1. This is a data-support audit, not a performance measurement. It makes no
   training, model-select, or generalization recommendation.
2. Recommendation: do not start a recognizer model experiment on the current
   factory corpus. Prioritize additional independent, vocabulary-sharing source
   builds and capture coverage, then re-run this audit against the section-10
   falsification criterion.
3. Open question for the lead: whether a narrow 4-class cross-build diagnostic
   or a within-build capture-grouped split is worth running explicitly as a
   bounded sanity check, clearly labeled as not a generalization result. This
   report does not recommend it.
4. Stop condition reached: no further action and no self-acceptance.
