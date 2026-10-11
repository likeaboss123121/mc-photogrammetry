# DS-R4-V2-ENTRY-REFUSAL-TRACE-2026-10-04

Task ID: `DS-R4-V2-ENTRY-REFUSAL-TRACE`. Owner: DeepSeek CLI worker
(Sol-equivalent, read-only source audit). Date: 2026-10-04.

Status: **READ-ONLY SOURCE/DOCS AUDIT COMPLETE — for lead review.** No source, test,
schema or runner file was edited. No pytest, py_compile, runner invocation, capture,
Minecraft/launcher, evaluator/truth/camera/mask/reference/litematic/scene input, screenshot
payload inspection, tuning, retry or implementation was performed. This report and the
`AGENT_COMMS.md` result are the only writes.

## 1. Question

Trace the exact R1 entry refusal from `scripts/evaluation/run_route_b_diagnostic.py`
`_default_api().entry` through `build_image_derived_entry` to the typed `unavailable` result
whose detail is:

```
image-derived session and relative camera evidence are established; V1/V2/V3/M5 geometry is unavailable from images alone in normal mode
```

Identify the precise branch/function and its prerequisites; separate a missing production
capability from an evaluator-only/advanced-mode condition; name the smallest synthetic
contract tests and the narrow automatic image-only geometry seam that takes no
manifest/camera/F3/direction/coordinate input; and say whether Route B would then reach
lattice/signatures/graph or hit another gate.

## 2. Exact trace (source-grounded)

### 2.1 Call path

| Step | Location | What happens |
|---|---|---|
| 1 | `scripts/evaluation/run_route_b_diagnostic.py:306-317` `_default_api()` | Binds `entry` to `mc_photogrammetry.pipeline.image_derived_entry.build_image_derived_entry` (with `lattice`/`signatures`/`graph` bound to the M9-C/M9-C/G-stage producers). |
| 2 | `run_route_b_diagnostic.py:412-414` `_run_single_pass` | Calls `api.entry(paths, project_path=..., capture_ingress=True)` and records `calls = ["entry"]`. |
| 3 | `image_derived_entry.py:396-553` `build_image_derived_entry` | Discovers/decodes images, builds the session (`build_image_derived_session`), the connection graph (`build_image_derived_connections`), the pose graph (`compose_image_derived_pose_graph`), persists the schema-2 checkpoint, and returns an `ImageDerivedEntryResult`. |
| 4 | `image_derived_entry.py:966-975` `_terminal_reason` | When `graph.status == "ok"` **and** `pose_graph.status == "ok"`, returns `("", "<the R1 detail string>")` (constant at lines 973-975). |
| 5 | `image_derived_entry.py:540-553` + `268-393` | `ImageDerivedEntryResult` is constructed with `terminal=GEOMETRY_UNAVAILABLE` (`:158-165`), `reason=""`, and that detail. `__post_init__` (`:373-376`) *requires* `(reason, detail)` to equal the derived terminal pair; `:320-321` *requires* `terminal == GEOMETRY_UNAVAILABLE`. |
| 6 | `run_route_b_diagnostic.py:390-398` `_status_refusal("entry", entry)` (`:415`) | `isinstance(entry, placement.PlacementRefusal)` is false; `getattr(entry, "status", None)` is `None`; `None != "assembled"`, so it raises `DiagnosticRefusal(reason = entry.reason or "unavailable", field="entry", detail=entry.detail)`. `entry.reason == ""`, so `reason == "unavailable"`. |
| 7 | `run_route_b_diagnostic.py:1066-1070` `main` | Serializes `{"status":"refused","reason":"unavailable","field":"entry","detail":"<R1 string>"}` to stderr and returns exit 1. |

### 2.2 Where R1's exact text is minted (two distinct acts)

- **The detail string** is produced only by `image_derived_entry._terminal_reason`
  (`image_derived_entry.py:973-975`). It is a hard-coded normal-mode terminal constant, not a
  computed measurement. It is reached **only** when the image chain fully succeeds at graph
  and pose-graph level (`graph.status == "ok"` and `pose_graph.status == "ok"`). Had either
  been refused, `_terminal_reason` (`:969-972`) would have returned that stage's own
  `reason`/`detail` instead. R1's `reason == "unavailable"` therefore *proves* both the
  connection graph and the pose graph were `ok` in the R1 run.
- **The refusal *type*** (`status:"refused"`, `field:"entry"`, `reason:"unavailable"`) is
  minted by the runner's `_status_refusal`, not by the entry. It appears because
  `ImageDerivedEntryResult` has no `status` field at all (`:301-313`) and `_status_refusal`
  reads `reason or "unavailable"` from the empty `reason`. Note the runner reports
  `"unavailable"`, not the entry's own terminal token `geometry_unavailable`, because it reads
  `reason` (empty) rather than `terminal`.

### 2.3 Prerequisites of the R1 detail

1. Three decodable still images producing a retained-pixel `ImageDerivedSession` with
   per-frame camera records (`build_image_derived_entry:453-502`).
2. `ImageDerivedConnectionGraph.status == "ok"` (`connections_mod.build_image_derived_connections`,
   `image_derived_entry.py:517-522`).
3. `ImageDerivedPoseGraph.status == "ok"` (`pose_graph_mod.compose_image_derived_pose_graph`,
   `image_derived_entry.py:524-529`).
4. `capture_ingress=True` succeeds (ingress pin built per frame, `:461-473`) — it did, because
   the entry returned normally rather than raising `ENTRY_SESSION_FAILED`.

## 3. The entry gate cannot pass with the default API

For the default `_default_api().entry == build_image_derived_entry`, `_status_refusal`
**always** refuses:

- `ImageDerivedEntryResult` has no `status` attribute (`image_derived_entry.py:301-313`);
  `getattr(entry, "status", None)` is `None`.
- `_status_refusal` accepts only `status == "assembled"` (`run_route_b_diagnostic.py:394-398`).
- The entry's contract forbids any other terminal: `__post_init__` requires
  `terminal == GEOMETRY_UNAVAILABLE` (`:320-321`). There is no branch of
  `build_image_derived_entry` that returns `status="assembled"`.

The runner's own injectable test seam masks this: `_fake_api` returns
`SimpleNamespace(status="assembled", ...)` for `entry` (`tests/test_route_b_diagnostic_runner.py:219-228`),
and every passing runner test injects that fake, so the default-API mismatch is never
exercised. The R1 refusal was therefore an **interface/contract mismatch in the runner's
generic stage-status adapter**, not a measured geometry result and not a downstream image
computation. (The runner's second, post-entry precondition — `entry.graph.status == "ok" and
entry.pose_graph.status == "ok"`, `run_route_b_diagnostic.py:418-422` — is *not* reached,
because the adapter refuses first; for R1 it would have passed, per §2.2.)

## 4. Missing production capability vs evaluator-only/advanced-mode condition

The R1 refusal is **neither** of the two conditions it could be confused with:

| Candidate explanation | Verdict | Source evidence |
|---|---|---|
| Evaluator-only condition (needs truth/reference/mask/camera/evaluator inputs) | **No** | The production path takes only ordinary image paths; there is no evaluator/truth/manifest token anywhere in `image_derived_entry.py`, `image_derived_lattice_measure.py`, `image_derived_geometry_orchestration.py` or the runner. The runner even asserts the banned tokens are absent from its own source (`tests/test_route_b_diagnostic_runner.py:306-311`). |
| Advanced/developer-mode condition (needs supplied camera/scene/F3/direction/coordinate) | **No** | Image-only geometry is implemented in normal mode: `measure_image_derived_lattice(session, pose_graph)` and `derive_image_derived_geometry(session, lattice_result, connection_graph)` take no manifest/camera/F3/direction/coordinate parameters, and `image_derived_production.derive_image_output_from_paths` already composes them from ordinary images (`image_derived_production.py:258-300`). |
| Genuine missing normal-mode geometry capability | **No, not at the algorithm level** | The V1 placement/support and V3/M5 composition seams exist and are wired (`image_derived_geometry_orchestration.py:245-307`, `image_derived_output.py`). The M4 audit records that they do run on real images and halt at geometry stage `g` (see §7), not that they are absent. |
| Missing entry contract / wiring the runner expects | **Yes** | The M7 entry is a contract-bound terminal (`geometry_unavailable`), and the runner's entry slot demands `status == "assembled"`. The production wiring handles the entry correctly by testing `graph.status`/`pose_graph.status` instead (`image_derived_production.py:253-257`); the runner does not. |

Naming caution: the entry's "V1/V2/V3/M5 geometry" means placement-universe (V1), aligned region
support (V2), plane volume (V3) and plane-face observation (M5). It is unrelated to the runner's
"V2" complete-support capture (`candidate_support_provenance_v2`). The R1 detail is the M7
entry's fixed scope terminal; it is not evidence that the runner's V2 target requires V1/V2/V3/M5
geometry.

## 5. The narrow automatic image-only geometry seam

The narrow seam already exists and is input-minimal:

- `measure_image_derived_lattice(*, session, pose_graph)` (`image_derived_lattice_measure.py:830-860`)
  is the first geometry step. Its public signature has exactly `session` and `pose_graph`
  (`:830-834`) — no manifest, camera, F3, direction, coordinate, bound, reference, scale or
  budget parameter — and it is the call the runner already makes at
  `run_route_b_diagnostic.py:425`. It returns `status="assembled"` when a complete
  image-derived reading/provenance bridge is sealed (`:943-952`), or typed
  `status="unavailable"`/`PlacementRefusal` otherwise.
- Beyond it, `derive_image_derived_geometry(*, session, lattice_result, connection_graph)`
  (`image_derived_geometry_orchestration.py:218-307`) is the full image-only V1/V2/V3/M5
  composition; `image_derived_production.derive_image_output_from_paths:271-273` is the
  existing production wiring.

So no new image-only geometry algorithm is required to "make automatic geometry available."
The minimal seam to unblock Route B is the runner's **entry acceptance**:

- **Recommended (runner-only, ~2 lines):** in `_run_single_pass`, replace the
  `_status_refusal("entry", entry)` call (`run_route_b_diagnostic.py:415-417`) with the same
  precondition the production wiring already uses (`image_derived_production.py:253-257`):
  accept the entry when its graph and pose graph are `ok`, treating `terminal ==
  geometry_unavailable` as the normal intermediate rather than a geometry refusal. This
  touches no production source and no geometry code.
- **Alternative (production-side adapter):** introduce an entry-stage wrapper in
  `_default_api()` that exposes an explicit assembled precondition record over the M7 entry.
  This is larger, duplicates `derive_image_output_from_paths`'s acceptance logic, and risks
  mislabeling the M7 terminal; not recommended.

Either change requires a fresh, separately authorized edit/review of the accepted runner;
this audit only names it.

## 6. Smallest synthetic contract tests (named, not written)

1. `test_entry_geometry_unavailable_terminal_is_an_accepted_precondition` — an
   `entry` fake shaped like a real `ImageDerivedEntryResult` (no `status`; `terminal ==
   "geometry_unavailable"`; `reason == ""`; `graph.status == "ok"`; `pose_graph.status ==
   "ok"`; session present) must **not** refuse at the entry stage and must reach the lattice
   call (`calls == ["entry", "lattice", ...]`). This is the single control that would have
   caught R1.
2. `test_entry_refusal_is_reported_when_graph_or_pose_graph_is_not_ok` — entry whose
   `graph.status == "unavailable"` still yields `DiagnosticRefusal(reason=graph.reason,
   field="entry", detail=graph.detail)`.
3. `test_status_refusal_filters_assembled_unavailable_and_missing_status` — unit-level:
   `_status_refusal` returns `None` only for `status == "assembled"`, maps
   `status == "unavailable"` to a typed refusal, and never treats a status-less entry as a
   geometry stage.
4. `test_lattice_seam_signature_takes_only_session_and_pose_graph` — assert
   `measure_image_derived_lattice`'s public parameters are exactly `session`/`pose_graph`
   (no manifest/camera/F3/direction/coordinate kwargs), proving the seam's input discipline.
5. `test_synthetic_lattice_signatures_graph_fires_v2_sink_once_on_abstention` — build
   synthetic retained-pixel frames/camera/pose graph as the existing
   `tests/test_image_derived_lattice_measure.py` and `tests/test_image_derived_graph.py`
   fixtures do, run lattice → signatures → graph with the V2 sink, and assert the capture
   fires exactly once on solver abstention (and that a solver *success* yields the runner's
   `capture_unavailable`).

Tests 1-3 are the "smallest" set for the refusal; 4-5 pin the seam and the next gate. All are
synthetic/in-memory and read no image payload.

## 7. Would Route B reach lattice/signatures/graph, or hit another gate?

After an entry-acceptance seam, the runner proceeds in order and meets these gates:

| Gate | Location | Condition to pass | Nature |
|---|---|---|---|
| Post-entry graph/pose check | `run_route_b_diagnostic.py:418-422` | `entry.graph.status == "ok"` and `entry.pose_graph.status == "ok"` | **Passes for R1** — R1's `reason == ""` proves both were `ok`. |
| Lattice | `:424-428` via `measure_image_derived_lattice` | `status == "assembled"` | Image-dependent. Can return `pose_unavailable`, `image_unavailable`, `no_lattice_reading`, `provenance_incomplete` (`image_derived_lattice_measure.py:875-936`), or an atomic `PlacementRefusal`. |
| Signatures | `:430-437` via `derive_image_derived_signatures` | `status == "assembled"`; requires lattice assembled (`image_derived_signatures.py:378-385`) | Image-dependent (`signature_unavailable`, `image_unavailable`, mask failure). |
| Graph attribution | `:439-450` via `derive_image_derived_graph_attribution` | lattice+signatures assembled (`image_derived_graph.py:5032-5035`); matcher yields matches (`:5062-5065`); no `PlacementRefusal`/`invalid_input` | Image-dependent. |
| V2 capture fire | `:451-455` | `len(captured) == 1`, i.e. the component solver **abstains exactly once** (`image_derived_graph.py:5082,5183-5198`) | Structural requirement. If the solver *succeeds*, the sink never fires and the runner refuses `capture_unavailable` ("the solver must abstain exactly once"). |
| Replay/count/identity | `:496-513`, `:763-810` | `row_count == 1_928` (`EXPECTED_ROW_COUNT`), page/order/whole-universe identity replay, ingress cross-binding | Fixed-run profile gate; a real run with any other admitted-universe count refuses `row_count_mismatch`. |

So the entry gate is the **first and currently unconditional** blocker, but it is not the only
one. Source reading alone cannot certify the lattice/signature/graph gates, because they
depend on the real image evidence. The strongest available documentary evidence is the
read-only M4 audit (`docs/reports/DS-R4-M4-production-integration-audit-2026-10-03.md` §2 G3,
§3): recorded normal/frozen runs **reached geometry stage `g`** (i.e. lattice and signatures
assembled) and the solver **abstained** there (`plane_integer_residual` / `ambiguous_quadrant`).
That abstention is exactly the condition that fires the V2 complete-support sink, which is
Route B's target. On that evidence the chain would likely reach graph and abstain, firing the
capture; but the same source also shows the geometry never assembled to V1/V2/V3/M5, and the
frozen 1,928-row count of these three R1 images is not established by source. Therefore:

- **Reaching** lattice/signatures/graph: yes, after the seam (they are simply the next calls).
- **Completing** the diagnostic: not guaranteed — the lattice/signatures assembly, the matcher,
  the solver-abstention requirement and the exact 1,928-row replay remain independent gates,
  each capable of refusing.

## 8. Boundaries and no-claims

- Read-only: source and docs only. No screenshot payload, evaluator, truth, camera, mask,
  litematic, scene or reference file was opened.
- No code, test, runner, capture, tuning, retry or implementation was performed.
- The M4-audit behavior cited in §7 is a recorded report of earlier runs, not a new run and not
  an accuracy/truth claim; the R1 verdict for these specific three PNGs remains unknown.
- The recommended entry-acceptance seam would change an accepted diagnostic runner and must
  itself be separately authorized, reviewed and tested; this report does not authorize it.
- Stopped for lead review.
