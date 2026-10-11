# E6 balanced factory-build acquisition design (evaluation-only)

- Task ID: `texture-poc-e6-balanced-build-acquisition-design-2026-10-03`
- Contract: `AGENT_COMMS.md` 2026-10-03, `Lead -> DeepSeek - E6 balanced factory-build
  acquisition design` (heading at line 4909; body at line 4910).
- Revision contract: `AGENT_COMMS.md` 2026-10-03, `Lead review / E6 remediation required`
  and `Lead review / E6 alignment with existing-world acquisition`.
- Worker: DeepSeek (Sol-equivalent), read-only analysis. Owned path: this report only.
- Status: design complete and remediated; **stopped for lead review**. No self-acceptance.
- Revision (existing-world alignment): builds are now sourced **existing-world-first**. P0
  begins with a read-only inventory of already available saves/maps/landscapes and selects
  naturally suitable builds, which are captured first; manual Minecraft authoring is a
  bounded fallback only for a measured shared-state or world-count gap. The earlier
  approved changes are preserved unchanged: the E3 evaluation-only training prohibition
  (section 10.7), the blocked P3, the ≥10-state / ≥50-example / ≥3-build no-go threshold
  (section 13), and `LOBO` terminology. No user directory was scanned for this revision.
- Verdict: the current corpus (2 builds, 4 shared exact states, all `train`; E5) cannot
  support a held-out recognizer experiment. This report specifies the concrete
  acquisition plan that would make one defensible, and names the exact existing factory
  steps that would produce it.

## 1. Scope, inputs and owned paths

Owned and written path (only this file was written):

| Path | Role |
|---|---|
| `docs/reports/texture-poc-e6-balanced-build-acquisition-design-2026-10-03.md` | this design report |

Inputs read, as permitted by the task:

| Input | Use |
|---|---|
| `docs/reports/texture-poc-e5-factory-state-support-2026-10-03.md` | measured support/OOV baseline and minimum-data recommendation |
| `docs/reports/texture-poc-e3-factory-patch-dataset-export-2026-10-03.md` | exporter admission contract, artifact semantics, finalized-root workflow |
| `docs/reports/texture-poc-e4-cross-view-label-consistency-2026-10-03.md` | label-consistency limits, build/capture identity definitions |
| `src/mc_photogrammetry/dataset/plan.py`, `schemas.py`, `splits.py`, `batch.py` | planner caps/strata, region caps, topology split rule, batch-spec contract |
| `scripts/prepare_capture_dataset.py`, `scripts/run_capture_batch.py` | exact prepare/capture/finalize pipeline and CLI options |
| `src/mc_photogrammetry/dataset/executor.py` (headers/constants) | fail-closed launch preflight, per-stage timeouts, mod pinning |
| `docs/reports/DS-17-real-capture-factory-2026-09-12.md` | accepted factory + Fabric mod contract, caps, setting semantics |
| `exports/datasets/pilot-batch-train-station-2026-09-12/batch-spec.json` | one existing batch-spec example for field shape |

Not read: any screenshot, reference Litematic, camera manifest, scene payload, JAR, T2/T6
fixture, or sealed E3/E4 artifact. Not run: Minecraft, the launcher, any factory script,
any corpus/model job, any install/download. No source, test, config or data file was
edited. The original design pass had no shell/process tool available and executed no
command, hash or test; this report-only revision used only file read/edit tools and ran no
factory, corpus or Minecraft command (see section 15).

## 2. What the factory can already do (and cannot)

The accepted DS-17 factory is a **capture** factory, not a world generator. Read together
with the E3 exporter, the current capability boundary is:

**Can do**

- Read-only inventory of saves already present in a game directory (`inventory_saves`,
  `prepare_capture_dataset.py --list`). Existing saves, and the structures, maps and
  landscapes they contain, are the **first and preferred build source** under the
  existing-world-first rule (section 9); the user has explicitly authorized leveraging
  them for the factory.
- Stage a byte-verified copy of one save into a staging game directory
  (`stage_world`), with caps and symlink/reparse refusal.
- Freeze a capture plan: explicit developer `--regions` bounds or a spawn-centred
  `--landscape` sample, 4-8 views per region, per-view orbit pose fitting, FOV from
  `(60,70,80,90)`, resolution from `(1920x1080,1600x900,1280x720)`, framed so the region
  projects inside the frame.
- Apply per-view setting strata by default (gamma across `[0,1]`, time from
  `{0, 6000, 12000, 18000}`, HUD alternating hidden/visible, weather drawn from
  `("clear","clear","clear","rain")`), or one fully explicit fixed settings object per
  plan (`--fixed-settings-json`, requires `gamma`, `time_of_day`, `weather`,
  `hud_hidden`).
- Arm exactly one plan per staged copy and drive a pinned Fabric mod through a
  disposable Prism worker on a hidden desktop (`run_capture_batch.py --execute`), then
  finalize the scene (`finalize_capture_dataset.py`).
- Key every scene's split on its block **topology only** (occupied coordinates + exact
  states): 70/15/15 train/dev/test, and refuse the same topology group in two splits
  (`splits.py`).
- Export evaluation-only, sealed, exact-state patch labels from finalized roots
  (`export_factory_texture_patch_dataset.py`, E3), and audit repeated-key consistency
  (`audit_factory_texture_patch_consistency.py`, E4).

**Cannot do (relevant here)**

- It cannot author or generate worlds, templates or structures. The factory only
  inventories, stages, captures and finalizes. Builds are sourced from **already
  available Minecraft structures, saves, maps and landscapes**, which the user has
  explicitly authorized using. Manual building in Minecraft is a **bounded fallback**,
  used only for the shared-state or world-count gap that the existing inventory cannot
  fill (section 9); it is not the required path and the user is not asked to author the
  whole slate.
- It cannot choose the authoritative split. `batch.py`'s `declared_split` is only a
  queue grouping used to refuse the same source content in two declared splits; the real
  split is the topology-hash bucket computed at finalization.
- It cannot enforce a **world-level** holdout. The split group is the topology digest
  alone, so two different structures inside one save can land in different splits.
- It cannot switch resource/shader packs. They are recorded as requested only; E3
  admission additionally requires `active_resource_packs == ["vanilla"]`,
  `resource_pack_applied == false`, `shader_pack_applied == false`.
- It cannot guarantee that a placed block yields enough visible labelled patches; that
  depends on visible surface area, occlusion and the orbit, and is measured only after
  the E3 export.

Hard caps that bound any plan: `MIN_VIEWS=4`, `MAX_VIEWS=8`, `MAX_PLAN_JOBS=64`,
`MAX_REGION_EDGE=512`, `MAX_REGION_VOLUME=250_000` blocks, pose tolerance 0.25 blocks,
heading tolerance 1.0 degrees, nonzero roll rejected. Executor stage timeouts default to
prepare 1800 s, capture 3600 s, finalize 600 s.

## 3. Identity definitions (fixed for this design)

These are deliberately stricter than the code's minimum, because E5 showed that "four
files" was only two independent builds.

| Term | Definition | Notes |
|---|---|---|
| **Exact state / class** | normalized exact `BlockState` `(name, properties)` (E4/E5) | the unit of every quota below |
| **Material family** | block `name` ignoring properties (e.g. `stone_brick_stairs`) | reported separately; never substitutes for exact states |
| **World / source build** | one staging content fingerprint (`source_identity` / `build_id`) = one save | a world is the independence unit that the split code does *not* enforce |
| **Structure / split group** | one topology digest (`scene_group`), i.e. one distinct block arrangement | distinct structures always get distinct groups |
| **Independent build** | a unique `(world_identity, scene_group)` pair, **one structure per world** in this design | recaptures/variants of one structure are *not* builds |
| **Capture** | one distinct screenshot (`screenshot_sha256`), E5 definition | a scene can contribute many captures |
| **Capture session / scene** | one prepared + armed + finalized plan (one region, one settings object) | one Prism launch sequence |
| **Variant** | another capture session of the **same** region/structure (recapture) | same topology = same group = same split; this is how ≥3 captures per state are obtained without inflating the build count |

The single most important rule: **one structure per world**. If a world holds two
structures, the topology-only split can put them in different folds, and shared terrain,
lighting and world style leak across the evaluation boundary. The zero-code-change way to
prevent that is to put exactly one captured structure in each save. (A world-level split
key would need its own bounded factory contract; see section 11.)

## 4. Target number of independent builds

The binding unknown is the 70/15/15 topology-hash split, which cannot be predicted for a
build before it is captured. The plan therefore uses an **enrollment gate**: inventory the
available saves, select naturally suitable independent builds (authoring new ones only if
the inventory cannot fill the gap), finalize them, read each build's assigned split from
`index.json`, and keep going until the corpus holds the required groups. The plan below is
stated as a minimum and a target.

| Tier | Independent builds (structures/worlds) | What it buys | Gate |
|---|---:|---|---|
| Minimum viable | **6** | 6-fold leave-one-build-out; at least 5 training builds in every fold | every build a distinct save and distinct topology group |
| Recommended | **12** | 12-fold leave-one-build-out; expected ~8-9 train / ~2 dev / ~2 test under 70/15/15 | enroll from the existing inventory (fallback authoring only for uncovered gap) until **≥6 train, ≥2 dev, ≥2 test** distinct groups exist |
| Do not claim | <6 | a single held-out build is not a spread | refuse |

Both tiers require the same per-build quality gates (section 6). If the topology hash
refuses to put ≥2 groups in dev/test after 12 builds, either continue to the next
enrollment cohort or state the primary result as the preregistered 12-fold LOBO and treat
the hash split as a recorded secondary field. The fold assignment must be frozen before
any capture that will be scored; no fold may be changed after seeing a result.

## 5. Shared-core exact-state palette

The core exists to make a held-out build mostly in-vocabulary for a small closed set that
is **natural to build with**. It is not a quota that justifies adding odd blocks. Proposed
core (all full-cube or standard modelled blocks; all vanilla 1.21.5):

**Tier A - mandatory in every build (property-free, ubiquitous masonry/timber):**

| # | Exact state | Why natural |
|---|---|---|
| 1 | `minecraft:stone` | foundation, fill, retaining walls |
| 2 | `minecraft:cobblestone` | plinths, rustic walls, paths |
| 3 | `minecraft:stone_bricks` | formal walls, pillars, lintels |
| 4 | `minecraft:andesite` | quoins, kerbs, contrasting courses |
| 5 | `minecraft:oak_planks` | floors, doors, timber framing, crates |
| 6 | `minecraft:spruce_planks` | second timber tone, roofs, shutters |
| 7 | `minecraft:bricks` | chimneys, ovens, accent bands |
| 8 | `minecraft:smooth_stone` | sills, slabs, clean trim |

**Tier B - mandatory in at least two thirds of builds (low-property or one canonical
orientation):**

| # | Exact state | Why natural |
|---|---|---|
| 9 | `minecraft:oak_log axis=y` | upright structural posts |
| 10 | `minecraft:stone_brick_slab type=bottom,waterlogged=false` | sills, ledges, lintels |
| 11 | `minecraft:chiseled_stone_bricks` | decorative courses |
| 12 | `minecraft:polished_andesite` | steps, paving, trim |
| 13 | `minecraft:cobblestone_stairs facing=north,half=bottom,shape=straight,waterlogged=false` | actual stairs, in at least one canonical orientation |

Primary exact-state requirement: **≥10 exact states, each present in ≥3 builds**, drawn
from Tier A ∪ Tier B. With Tier A (8) plus any two of Tier B this is met in every build,
so the core is realistic rather than forced.

**Material-family (name-level) supplementary set** - reported but never a substitute:
the 13 names above plus `stone_brick_stairs`, `glass`, `cobblestone_slab`,
`oak_stairs`. Requirement: **≥10 material families shared across ≥3 builds**.

Why property-poor states dominate the exact core: E5 measured nine block names whose
property variants split into separate exact classes, so a staircase placed in four
facings plus shapes becomes many thin exact states. Requiring a specific stair facing in
every build would force the architecture; requiring a canonical orientation only where a
build naturally has one keeps the core honest while the family-level numbers capture the
rest. Any orientation-specific exact state that a build cannot reach 50 examples for is
demoted to "family support only", not padded with blocks.

## 6. Per-state and per-build quotas

Quotas are per **build** (one structure/world) and per **exact core state**. "Admitted"
means an E3 example that passed seal admission with `valid_fraction >= 0.75`.

| Gate | Minimum | Target | Rationale |
|---|---:|---:|---|
| distinct captures containing the state | **≥3** | ≥5 | E5 requires capture-grouped evaluation; a state seen once is untestable |
| admitted examples per core state | **≥50** | **≥100** | E5 recommendation; a moderate visible face yields tens of cells, so this is a few visible faces, not a whole building |
| admitted examples per build (all states) | **≥1,000** | ≥2,500 | keeps each fold's held-out mass non-trivial |
| capture sessions per build | **≥3** | 4 | gives gamma/time/weather variation and recapture headroom |
| finalized HUD-hidden frames per build | **≥16** | ≥24 | 3 sessions x 8 views, all HUD hidden |
| builds per exact core state | **≥3** | ≥6 | a held-out build must leave ≥2 training builds that still know the state |
| core exact states meeting the above | **≥10** | ≥13 | E5 falsification threshold |
| distinct material families shared | **≥10** | ≥14 | supplementary |

Corpus totals implied: minimum 10 states x 50 x 6 builds = **3,000 core examples**;
recommended 10 x 100 x 12 = **12,000 core examples**, plus tail states. The E3 artifact
cap (`MAX_EXAMPLES = 200_000`) is not a constraint at this scale.

## 7. Capture matrix (camera / gamma / render variation, vanilla POC)

Each build is captured as **one task with three fixed-settings variants** of the *same*
region, so all three share one topology group and one split, while varying render state.
This is expressible in the existing batch spec (`_settings_for_variant` merges task and
variant settings; `scene_payload` emits `--fixed-settings-json`).

Suggested matrix per build (task-level `hud_hidden: true`, `weather` inherited or
overridden):

| Variant | gamma | time_of_day | weather | views | purpose |
|---|---:|---:|---|---:|---|
| `day` | 1.0 | 6000 | clear | 8 | bright noon, full colour |
| `night` | 0.0 | 18000 | clear | 8 | dark endpoint / lighting invariance |
| `dusk-rain` | 0.5 | 12000 | rain | 8 | mid gamma, transition shading, wet tint |

Within a session all 8 views share gamma/time/weather but differ in camera pose, and the
planner additionally draws per-view FOV (60/70/80/90) and resolution
(1920x1080/1600x900/1280x720) from the seed. Different `seed` values per variant move the
orbit so the three sessions are genuinely distinct captures. All settings must be the
**fixed** mode, not the stratified default, because the stratified default alternates
HUD-visible views that E3 refuses; fixing `hud_hidden: true` makes every planned frame
labelable.

Vanilla-only constraint: the capture client must report `active_resource_packs ==
["vanilla"]`, `resource_pack_applied == false`, `shader_pack_applied == false`. "Render
variation" for this POC therefore means gamma, time, weather, FOV, resolution and camera
pose only - no resource packs and no shaders.

Two camera checks to run **before arming** (from the frozen plan JSON, no images):

- confirm the three sessions together exercise all four FOV choices and at least two
  resolutions; reseed and re-plan if a seed degenerates;
- confirm elevation spans near-horizontal (0.0 rad) to high (up to 1.4 rad) and at least
  one view per session is oblique to a large core-state surface, so grazing/rectification
  behaviour is represented.

## 8. Structure / template diversity brief

Independence of *content* is what the vocabulary overlap needs; diversity of *scene* is
what keeps the evaluation from being a single template test. The recommended 12-build
slate (3 per theme) is a **shopping list to match against the existing inventory first**:
prefer already-built structures, each in its own save, that naturally contain the Tier A
core plus enough Tier B. Only the themes the inventory cannot cover become candidates for
fallback authoring (section 9).

| Theme | Build briefs | Diversity contributed |
|---|---|---|
| Masonry | gatehouse; retaining wall with return stairs; bridge abutment | flat walls, corners, steps, thin trim |
| Timber frame | barn/house facade; covered walkway; watchtower | plank fields, log posts, overhangs, shadowed recesses |
| Mixed material | market stall; dock/pier; workshop with brick chimney | mixed palettes, partial occlusion, small faces |
| Terrain-integrated | terraced hillside cut; quarry face; causeway | irregular surroundings, slope-facing views, natural backdrop |

For every build: vary region size (tight to the structure, well under `MAX_REGION_EDGE` /
`MAX_REGION_VOLUME`), orientation (axis-aligned and 45-degree-facing structures), indoor
vs outdoor, and the amount of sky/background in frame. The `recon-canal-houses` failure
in E3 (sky-dominated framing, zero eligible faces) is the explicit anti-pattern: a build
that passes metadata admission but shows empty framing must be recaptured or excluded,
not counted.

No block is added to a structure solely to satisfy a quota. If a brief cannot naturally
host a core state, that state is recorded as Tier-B-missing for that build; a build that
misses a Tier A state or the ≥10-state shared requirement is not admitted.

## 9. Which existing factory steps generate / capture / finalize this data

The plan maps onto the accepted pipeline with no new production code. Builds come from
**already available Minecraft structures, saves, maps and landscapes**, which the user has
explicitly authorized using. The factory only inventories, stages, captures and finalizes;
it does not author or generate worlds. Manual Minecraft building is a **bounded fallback**,
used only for the shared-state or world-count gap that the existing inventory cannot fill.

| Stage | Existing step | What it does for E6 |
|---|---|---|
| 0. Inventory + select existing builds | `prepare_capture_dataset.py --list` / `inventory_saves`, then read-only inspection of the listed saves | read-only list of already available saves and the structures/landscapes they hold; select naturally suitable, independent builds whose regions contain the Tier A core. No writes. This report does not itself scan any user directory (section 15) |
| 1. Fill the gap (fallback only) | *manual Minecraft building by a human* | author a new structure, one per save, **only** where the inventory cannot fill the remaining shared-state or world-count gap; the factory does not build, and this is not the required path |
| 2. Plan queue | `plan_capture_batch.py` over a batch spec (`schema_version: 1`) | writes the content-addressed manifest, per-scene prepare/finalize argv and `regions/*.json`; never launches or stages |
| 3. Prepare / stage / arm | `prepare_capture_dataset.py` (invoked per scene by the executor) with `--regions`, `--fixed-settings-json`, `--views 8` | stages the copy, verifies the staging token, writes recipe/plan, writes the in-world marker, arms one scene |
| 4. Provision + capture | `run_capture_batch.py --execute` with `--capture-mod` + `--capture-mod-sha256`, `--prism-executable`, worker args | pins and copies the capture jar, creates one disposable worker, hidden-desktop preflight, launches Prism, waits for the mod sentinel, tears down |
| 5. Finalize | queued `finalize_capture_dataset.py` | validates poses/settings/PNG dimensions, writes `capture.json`/`ground_truth.json`/`reference.litematic` and the topology-split `index.json` |
| 6. Export labels | `export_factory_texture_patch_dataset.py` over the finalized dataset root with the pinned 1.21.5 JAR | sealed, evaluation-only exact-state patch artifact (`evaluation_only: true`, `training_permitted: false`); **never a training input** (section 10.7) |
| 7. Audit (optional) | `audit_factory_texture_patch_consistency.py`, then an E5-style support audit | consistency proxy and the machine-readable quota/falsification check |

A design-level batch-spec skeleton (illustrative; not written to disk):

```json
{
  "schema_version": 1,
  "batch_id": "e6-balanced-acquisition",
  "game_dir": "<source game dir>",
  "dataset": "<dataset root outside the game dir>",
  "seed": 1702,
  "views": 8,
  "tasks": [
    {
      "task_id": "b01-gatehouse",
      "declared_split": "train",
      "save": "E6 b01 gatehouse",
      "settings": {"hud_hidden": true, "weather": "clear"},
      "region": {"name": "gatehouse", "min": [0, 60, 0], "max": [31, 95, 31]},
      "variants": [
        {"variant_id": "day", "seed": 1101, "settings": {"gamma": 1.0, "time_of_day": 6000}},
        {"variant_id": "night", "seed": 1102, "settings": {"gamma": 0.0, "time_of_day": 18000}},
        {"variant_id": "dusk-rain", "seed": 1103,
         "settings": {"gamma": 0.5, "time_of_day": 12000, "weather": "rain"}}
      ]
    }
  ]
}
```

Field notes: `save` names an existing save selected in stage 0 (or, only as fallback, a
newly authored one, still one structure per save), and `region` bounds are chosen around
that build's structure. `declared_split` is queue bookkeeping only (it refuses the same
source content in two declared splits); the authoritative split is the topology hash at
finalization. The three variants share bounds and source identity, so they are recaptures
of one build, not three builds. Explicit `region` bounds are the only geometry the queue
accepts; camera/F3/coordinate keys are refused by contract.

## 10. Grouped split and evaluation gates

Proposed preregistration, before any scored capture:

1. **Fold unit = independent build**, and because each world holds exactly one structure,
   fold unit = world. Keep all captures and all variants of a build in one fold.
2. **Primary protocol = leave-one-build-out (LOBO; equivalent to leave-one-world-out
   here)** over the enrolled builds, which is what E3/E4/E5 already compute over
   `scene_group`; 6 folds minimum, 12 recommended. The topology-hash train/dev/test split
   is recorded and used only if it actually holds ≥1 (minimum) or ≥2 (recommended) groups
   per split.
3. **Scoring domain = the frozen ≥10-state core only.** Every held-out example whose exact
   state is outside the core is **OOV** and must be abstained on, never scored as a
   core-class error; its share of held-out examples is reported per build.
4. **Reporting unit = build and capture**, not patch. Report per-build held-out accuracy
   spread and per-capture spread; a single folded number is not a result.
5. **Leakage checks:** `split_report` refuses the same `scene_group` in two splits; on top
   of it, assert the world-level rule that no `world_identity` appears in two folds. E3's
   `check_split_leakage`/`build_group_manifest` cover the topology-group half of this.
6. No threshold, class list or fold may be changed after any scored result is seen.
7. **Training data is a separate, lead-approved artifact that does not exist yet.** The
   sealed E3 evaluation artifacts (`evaluation_only: true`, `training_permitted: false`;
   the four existing roots) and every future E3 evaluation artifact must **never** be used
   for model fitting. Fitting requires a separate training-data contract and exporter that
   this task neither implements nor authorizes; **that exporter/contract does not exist
   today.** It must admit only designated **world-grouped train folds**, keep
   `dev`/calibration examples in their own separately sealed artifact, exclude T2 and T6
   fixture data entirely, reserve an immutable world-held-out **test** set that is excluded
   from fitting, and bind its own frozen split manifest, label-purity gate, provenance and
   digests. Model fitting and evaluation stay blocked until that contract, the split
   manifest and the label-purity gate are reviewed and approved (P3).

## 11. Quotas, refusal criteria and what stays a gap

**Sourcing order (existing-world-first):** inventory the available saves/maps/landscapes,
select the naturally suitable independent builds, and capture those first. Only after that
produces a measured gap in shared exact states or independent world count does fallback
authoring become eligible, and then only for the specific missing themes.

**Refuse a build at export/audit time if any of:**

- its `scene_group` duplicates an earlier build (identical topology copied into a new
  save is the same group, not a new build);
- its `world_identity` already contributed a different structure to a different fold;
- any Tier A exact state is absent, or fewer than 10 exact states meet the ≥3-build
  requirement after export;
- a core state has <3 distinct captures or <50 admitted examples at
  `valid_fraction >= 0.75`;
- total admitted examples <1,000, or finalized HUD-hidden frames <16;
- export is empty (`empty label export`, the E3 failure class), or any core state is
  refused as `model_metadata_unavailable`;
- any frame has missing actual pose, HUD not hidden, PNG dimensions that disagree with the
  planned resolution, or active resource/shader packs.

**Recapture rule:** a build that misses a *state* quota is recaptured with an additional
variant session (new seed, same region) before any further build is selected from the
inventory and before any fallback world is authored, up to a bounded number of extra
sessions.

**Known gaps (proposals only, not implemented here):**

- *World-level split key.* The code groups by topology only. This design avoids the gap
  by one structure per world; a `(world_identity, topology)` group key would be a separate
  bounded contract.
- *Per-view HUD pinning with stratified gamma/time.* The stratified path alternates HUD
  visibility, and the fixed path freezes gamma/time for all views. This design works
  around it with three fixed variants; a small factory change could give stratified
  gamma/time with `hud_hidden` pinned true, saving ~1/3 of the launches.
- *A machine-checkable core-coverage report per build* would turn section 11's refusal
  list into a pre-registered audit step, analogous to E4/E5.
- *A training-data contract/export.* No approved exporter exists that admits only
  world-grouped train folds, separates `dev`/calibration, and reserves an immutable
  world-held-out test set. E3's `evaluation_only`/`training_permitted: false` artifacts are
  not training data and must never be reused as such.

## 12. Phased execution plan

| Phase | Work | Exit gate |
|---|---|---|
| P0 inventory + prototype | **read-only inventory** of already available saves/maps/landscapes; select naturally suitable independent builds, including at least one structure containing the full Tier A core; capture the 3-variant matrix on that build; export with E3 | inventory yields a candidate build list; every core state is labelable; measure examples-per-state-per-capture; if any core state <50 examples at 3 captures, revise the selection/briefs/regions before scaling |
| P1 minimum | capture + export the selected builds until **6** admitted builds; fill any remaining gap with fallback authoring only if the inventory cannot | 6 distinct worlds/groups; ≥10 shared exact states with ≥50 examples in ≥3 builds; per-build gates met |
| P2 recommended | keep enrolling from the inventory to **12** builds (fallback authoring only for themes the inventory cannot cover), enrollment gate on the hash split | ≥6 train / ≥2 dev / ≥2 test groups (or a frozen 12-fold LOBO); corpus ≥12,000 core examples |
| P3 fit/evaluate (**blocked**) | preregistered LOBO closed-set evaluation on the core with OOV abstention | **blocked**: no model fitting or evaluation until a separate lead-approved training-data contract/export exists (section 10.7), the world-grouped split manifest is frozen, and the label-purity gate is approved; E3 evaluation artifacts must never be used for fitting |

P0 is mandatory: it converts the example-per-state quotas from estimates into measured
yield for this specific sampler, palette and orbit behaviour.

## 13. Falsification / no-go criteria

Re-run the E5-style seal-verified support audit on the new corpus. Acquisition has failed
(do not train, go back to coverage) if any holds:

- a held-out build still has **>50% OOV examples** (E5 section 10);
- fewer than **10 exact states have ≥50 admitted examples in at least three independent
  builds** (the section 5/6 primary quota; two builds is not sufficient);
- fewer than **3 builds** share any core state;
- per-build admissible label yield collapses below the section 6 floors after the
  bounded recapture budget;
- capture metadata cannot be produced honestly (non-vanilla actives, missing actual pose,
  HUD visible) at scale.

E4 is explicit that repeated-key agreement is **not** correctness; region-clipping and
occluder contamination remain a separate, still-unmeasured label-purity gate, and no
accuracy claim may be made from consistency.

## 14. Explicit unknowns

1. **Examples per state per capture are unmeasured** for this palette/sampler; section 6's
   50/100 targets are design targets that P0 must confirm.
2. **Model labelability of the proposed core is unverified.** E3 labels only states with
   exact baked model polygons (`include_partial=True`); the palette must be checked
   against the pinned 1.21.5 JAR in P0 before it is frozen.
3. **Topology-hash split membership is unpredictable** for a build before it is captured;
   hence the enrollment gate and the LOBO fallback.
4. **The size and composition of the existing build inventory is unknown here.** This
   report does not scan user directories (section 15), so how many independent, naturally
   suitable builds, maps and landscapes already exist is unmeasured and is the first thing
   P0 measures. Fallback authoring cost is therefore conditional: zero if the inventory
   suffices, and only the unfilled shared-state/world-count gap would be hand-built. No
   generator is in scope and none may be added by this task.
5. **FOV/resolution coverage is seed-driven, not controllable**; the pre-arm plan
   inspection in section 7 is the only guarantee.
6. **Gamma/time variation is across sessions, not across views**, under fixed settings;
   whether three gamma levels are enough for the POC is untested.
7. **World-level holdout is convention, not code**, under the current topology-only
   split; a multi-structure world would silently break it.
8. **Label purity** (region clipping / omitted occluders, E4 section 6) is not addressed by
   acquisition design.
9. **Version pinning** (Minecraft 1.21.5, exact models, JAR/hash) means the frozen core
   palette and quotas are version-specific.

## 15. Non-actions and checks

- Read-only analysis: only this report was created. No source, test, config, data,
  artifact, fixture or world was created, edited or overwritten.
- No user directory, save folder, map or landscape was scanned, opened or listed by this
  task. The existing-world inventory in sections 9/11/12 is a design step for the factory
  operator, not an action performed here.
- No Minecraft launch, launcher automation, world/template/image generation, screenshot/
  reference/JAR read, corpus or model job, dependency install/download, background
  process, persistence, shell nesting, job/workflow/subagent, junction/reparse operation,
  recursive deletion, encoded command, generated executable, Defender/system change, or
  system-directory write occurred.
- The original design pass ran with no shell/process tool exposed, so it reported no
  command, hash or test result. This report-only revision touched only this file: no
  factory script, Minecraft launch, corpus/model job or user-directory scan was run. The
  report hash is computed by the lead with `Get-FileHash -Algorithm SHA256
  docs/reports/texture-poc-e6-balanced-build-acquisition-design-2026-10-03.md`.

## 16. For lead review

1. This is an acquisition **design**, not a measurement and not training authority. No
   model was trained and no accuracy/quality number is claimed.
2. Decision requested: accept the existing-world-first sourcing order (read-only inventory
   of available saves/maps/landscapes and selection of naturally suitable builds before
   any fallback authoring), the one-structure-per-world rule, the 6/12 build tiers, the
   ≥10 exact-state shared core with the section 6 quotas, and the P0 inventory + prototype
   gate.
3. The two code-level gaps in section 11 (world-level split key; stratified settings with
   HUD pinned hidden) are proposals only; if either is wanted, it needs its own bounded
   contract and Sol ownership.
4. Recommendation: run P0's read-only inventory first, select the naturally suitable
   builds, prototype yield on one of them, then decide whether the 12-build slate (or any
   fallback authoring) is needed.
5. Stop condition reached: no further action and no self-acceptance. This report-only
   revision applied the lead's existing-world alignment review and changed no code or data.
