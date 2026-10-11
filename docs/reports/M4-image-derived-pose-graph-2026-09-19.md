# M4 image-derived relative pose graph composition (2026-09-19)

Task: bounded production seam requested 2026-09-19 - consume an accepted
image-derived connection graph and compose its solved relative-camera evidence
into deterministic, non-metric, connected-component gauge poses, refusing
contradictory cycles atomically.
Worker: `deepseek-flash`, bounded implementation.
Status: **Implemented and repaired twice; the edge composition convention was
bound exactly against the shipped matcher before implementation, Sol's first
finding - the source replay did not re-validate a solved edge's
`relative_rotation` as a proper rotation - is fixed and covered by adversarial
zero-budget regression, and Sol's final narrow finding - the SO(3)
orthonormality check ran `np.allclose(A.T @ A, I, atol=1e-6)` with NumPy's
default `rtol=1e-5`, silently widening the documented 1e-6 bound to 1.1e-5 - is
fixed with an explicit `rtol=0.0` and a determinant-preserving near-boundary
regression (`py_compile`, the suite (29 tests), and the accepted
image-derived-connections, image-derived-session and project-store suites all
pass: 128 combined); stopped for Sol review.**

Write scope used (exactly three routed paths):

- `src/mc_photogrammetry/pipeline/image_derived_pose_graph.py`
- `tests/test_image_derived_pose_graph.py`
- `docs/reports/M4-image-derived-pose-graph-2026-09-19.md` (this file)

No other module, test, script, `AGENT_COMMS.md` or `AGENT_HANDOFF.md` file was
created, edited or deleted. `AGENT_COMMS.md` was not modified by this worker
(the repair never opened it for writing). No git
stage/commit/reset/clean was run. The only side effect of the required
`py_compile`/`pytest` runs is refreshed standard `__pycache__` bytecode caches.

Artifacts:

| path | lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/pipeline/image_derived_pose_graph.py` | 986 | `B80B75034FF133AB8DD25EA328081863C97ABD7969C9D03E35B751E01447A74E` |
| `tests/test_image_derived_pose_graph.py` | 817 | `CF2C2B88FE37273EB71AEFF3B691C77BBBE99FEA9DAD22C4BE0C2FF161761256` |

The report's own digest is not self-referential and is recorded by the lead in
the run record.

## 1. Objective and boundary

The accepted `pipeline.image_derived_connections` seam emits, per solved
unordered frame pair, a matcher-claimed `relative_rotation` and unit
`baseline_direction`, plus the recovered `right_yaw_class`, bound to the
session digest; it deliberately composes no camera positions. This module is the
next narrow seam only: it consumes an accepted
`ImageDerivedConnectionGraph` whose `status` is `ok` and emits, per frame, an
immutable relative gauge pose on the frame's solved connected component. It is
**not** a reconstruction, a scale proposal, a bounds/scene proposal, or an
export/CLI/UI surface.

The module's imports are the standard library, `numpy`, and the accepted
connection-graph module. It never imports or consumes a `BlockVolume`,
`FaceObservation`, `ScanProject`, exporter, CLI, UI or manual-metadata path,
and it has no `**kwargs` through which one could arrive. Its only public entry
point is `compose_image_derived_pose_graph(graph, *, budgets)`.

## 2. Edge composition convention (bind exactly, or stop)

The task's guard was applied first. The accepted `RelativeCameraEvidence`
semantics were re-derived from the owning code and then probed against the
shipped matcher before any implementation:

- `PairPose.baseline_direction` is documented as `b = C_r - C_l` in the anchor
  frame declared by the **left** estimate, and `tests/test_connected_camera.py`
  asserts it equals the true **world** camera-to-camera unit vector
  (tolerance 1 degree). A probe reproduced this exactly for left yaw 85 degrees
  and showed that rotating the true baseline by the left estimate's own
  world-to-camera matrix misses it by 126.87 degrees.

- A single view fixes yaw only modulo 90 degrees, so each frame's declared
  world frame differs from the shared session frame by an unresolved per-frame
  yaw gauge `k` in `Z_4`. The probe used true yaws 85/95/105 degrees
  (representatives 85/5/15, so `k = 0/1/1`) and found that the shipped matcher's
  `(b, c)` baseline reproduces the true world direction to **90.0 degrees**
  unless it is rotated by `Ry90(-(k_b - k_a))`.

- `relative_rotation` therefore does **not** transport baselines between left
  estimate frames: it is gauge-invariant (`R_ij = R_j R_i^T`), and the probe
  measured `R_ab^T b_bc` at 100.10 degrees and `R_ab b_bc` at 80.12 degrees from
  the true direction. The DS-15 report's `R(i,j)^T b(j,k)` transport is
  frame-inconsistent for the accepted evidence (DS-15's closure diagnostic was
  never exercised - its eligibility set was empty).

- `right_yaw_class` is exactly the missing relative gauge:
  `class(i, j) = k_j - k_i (mod 4)`. With it, `Ry90(-class(i,j)) b(j,k)`
  matched the true world direction to **0.0 degrees**, and a three-camera
  triangle's scaled closure residual was **2.1e-15** with a **0.0 degree**
  rotation-cycle error.

**Decision.** A composition from `relative_rotation` and
`baseline_direction` *alone* cannot bind (stop condition met); but the accepted
evidence record already carries `right_yaw_class`, its documented purpose is
that quadrant, and using it makes the convention bind exactly. The seam
therefore consumes the full accepted `RelativeCameraEvidence`, not two isolated
fields, and the convention it implements is:

1. build the component's canonical spanning forest and accumulate
   `gauge[i] = k_i - k_anchor (mod 4)` along the tree from `right_yaw_class`;
2. transport each edge baseline into the anchor's declared frame with
   `Ry90(-90 * gauge[edge.left_id])`, flipping it when the edge is walked
   right-to-left;
3. propagate `position[child] = position[parent] + transported_unit_baseline`
   (a unit-edge gauge embedding: no magnitude, no metric scale);
4. propagate `rotation[child] = R @ rotation[parent]` from the identity anchor.

The result is anchored at the component's first frame (identity rotation, zero
position) and is deterministic for a given graph and budget. It is a *relative*
record: the residual global orientation of the anchor's own estimate is not
carried by the graph and is never invented, and the positions live in the
anchor's declared yaw gauge, not in world coordinates.

## 3. Public contract

### 3.1 `GaugeCameraPose` (immutable)

One record per source frame, in the graph's declared frame order:

- `frame_id` - the frame;
- `component_id` - the canonical index of the solved component (singleton
  components for isolated frames included);
- `anchor_frame_id` - the component's first frame (`frames[0]`);
- `rotation` - the gauge-to-camera 3x3 rotation, validated orthonormal with
  determinant `+1` within `1e-6` (an absolute entry-wise bound: the
  orthonormality check forces `rtol=0.0`);
- `position` - three finite numbers: the unit-edge gauge position;
- `graph_sha256` - provenance binding to the accepted source graph.

### 3.2 `ImageDerivedPoseGraph` (immutable)

- `status` `"ok"` with exactly one pose per frame in canonical frame order, the
  canonical solved components, the copied session `unresolved` facts, and
  `metric_scale_known` hard-frozen `False`; or
- `status` `"refused"` with a typed `reason`, a `detail`, and **no** poses and
  **no** components - never a partial result;
- `graph_sha256` and `session_sha256` source provenance and a self-computing
  `pose_graph_sha256` over the canonical record.

### 3.3 `PoseGraphBudgets` (explicit, self-validating)

`make_pose_graph_budgets` requires every field explicitly (no hidden defaults)
and the record self-computes a `budget_sha256`:

- `max_edges` - the **solved-edge budget**, checked against the number of solved
  evidence edges before any tree or cycle work;
- `max_cycle_checks` - the **cycle-check budget**: each fundamental cycle
  consumes one check for its rotation/quadrant closure, and its direction
  closure consumes one check per Caratheodory subset examined, so the budget
  bounds both the cycle count and the total elementary work. A fundamental-cycle
  lower bound is checked before any tree work;
- `rotation_tolerance_degrees` - largest accepted angle between a cycle's
  composed rotation and the identity;
- `direction_tolerance` - largest accepted distance from the origin to the
  convex hull of a cycle's signed transported unit directions.

`0` is legal and deliberately strict.

### 3.4 Cycle checks

For every non-tree solved edge (chord), in canonical order:

- its quadrant cycle must close modulo 4
  (`gauge[right] - gauge[left] == right_yaw_class`) and its composed rotation
  must be the identity within `rotation_tolerance_degrees`;
- the origin must lie in the convex hull of the cycle's signed, transported unit
  directions within `direction_tolerance`. Positive (unknown) edge lengths then
  close the loop. The residual is the exact hull distance, computed by
  Caratheodory subset enumeration (subsets of size two, three and four, each
  with an exact KKT/simplex solve) - **no averaging** is used and no length is
  estimated.

Even a fully consistent cycle does not close in the unit-edge position gauge
when its true edge lengths differ, which is why the direction test is the
scale-free positive closure rather than a positional one. A consistent cycle is
therefore admitted; a contradictory one refuses the whole result.

### 3.5 Typed refusals

Graph reasons (`POSE_GRAPH_REFUSAL_REASONS`):

- `pose_graph_source_invalid` - the source graph or an evidence record does not
  replay its own bindings, **or** a solved edge's `relative_rotation` is not a
  finite `3x3` proper rotation (orthonormal and determinant `+1` within the
  explicit `1e-6` absolute tolerance);
- `pose_graph_source_refused` - the source graph's `status` is not `ok`;
- `pose_graph_budget_exhausted` - the solved-edge or cycle-check budget was
  exceeded;
- `pose_graph_cycle_rotation_inconsistent` - a cycle's rotation or quadrant does
  not close;
- `pose_graph_cycle_direction_inconsistent` - a cycle's directions admit no
  positive closure.

A malformed call (a non-graph, a non-budget) raises
`ImageDerivedPoseGraphError` with `pose_graph_input_invalid`, never a raw
`ValueError`.

## 4. Atomic invalid-before-budget behavior

`compose_image_derived_pose_graph` orders its decisions so a malformed input can
never be reported as a budget stop:

1. type checks on the graph and budgets (raised);
2. **source replay**: the graph and every evidence record are rebuilt through
   their own constructors with `dataclasses.replace`, so schema/version, status,
   canonical pair order, replayed components, forbidden claims and all digests
   are re-checked by the modules that own them; then the graph status, the
   per-record session binding, and every solved edge's `relative_rotation` are
   verified;
3. the **solved-edge** budget;
4. the fundamental-cycle lower bound of the **cycle-check** budget;
5. the canonical spanning forest and the cycle checks, with the elementary
   check counter charged as each closure and subset is evaluated.

Any refusal returns a result with empty `poses` and empty `components`, so no
prefix of a graph is ever emitted. The session's unresolved position/scale/yaw
facts and `scale_known=False` are copied into every result.

### 4.1 Repaired SO(3) source-replay gap (Sol's finding)

The accepted `RelativeCameraEvidence` constructor
(`_require_rotation_matrix`) guarantees only that `relative_rotation` is a
finite `3x3` matrix - it does **not** require orthonormality or determinant
`+1`. Before this repair the composer therefore trusted a finite `3x3` matrix:

- `diag(2, 1, 1)` replayed cleanly through `dataclasses.replace` (it is finite
  and `3x3`), and with ordinary budgets `_compose_poses` fed it to
  `GaugeCameraPose`, whose `__post_init__` raised a **raw `ValueError`**
  (`pose rotation must be orthonormal`) straight out of
  `compose_image_derived_pose_graph`; and
- with all budgets at zero the malformed edge was reported as
  `pose_graph_budget_exhausted`, because the solved-edge budget was consulted
  before any rotation validation ever ran.

Both violate the module's own contract that source-data conditions are typed
refusals, never raised and never a budget stop. The repair adds an explicit
SO(3) re-validation to source replay, before any budget:

- a shared `_rotation_problem(value, label)` helper checks finite `3x3`,
  `AᵀA = I` within the module's explicit `_ROTATION_TOLERANCE = 1e-6`, and
  `det(A) = +1` within the same tolerance (so reflections are rejected by the
  determinant test even though they are orthonormal);
- `_source_problem` applies it to every `status == "solved"` edge and returns
  `POSE_GRAPH_SOURCE_INVALID` with a diagnostic naming the edge and the failed
  property; `GaugeCameraPose._require_rotation` now shares the same helper, so
  the two checks cannot drift apart;
- the composition call is wrapped in a defensive seal mapping any residual
  data-derived `TypeError`/`ValueError` (e.g. a propagated rotation drifting
  outside tolerance on an extreme chain) to `POSE_GRAPH_SOURCE_INVALID` rather
  than letting it escape.

Every refused result still carries empty `poses` and empty `components`. The
accepted convention, reverse transpose/direction handling, canonical BFS forest,
cycle/direction checks, digests, replay, immutable output, unresolved
scale/manual metadata and module boundaries are unchanged.

### 4.2 Repaired SO(3) tolerance widening (Sol's final finding)

The shared `_rotation_problem` helper computed the orthonormality test as
`np.allclose(A.T @ A, np.eye(3), atol=_ROTATION_TOLERANCE)` and left NumPy's
default `rtol=1e-5` in force. `np.allclose` admits an entry when
`|AᵀA - I| <= atol + rtol*|I|`, so the effective entry-wise bound was
`1e-6 + 1e-5 = 1.1e-5`, not the documented explicit `1e-6`. A finite `3x3`
matrix that is determinant-preserving - so only the orthonormality test can
reject it - and whose Gram defect lies between `1e-6` and `1.1e-5` was silently
admitted as a proper rotation, then propagated into the gauge poses.

The repair is exactly `rtol=0.0` on that comparison (determinant `+1` was
already an absolute `1e-6` check), so the declared bound and the enforced bound
are now the same. A new regression constructs the determinant-preserving
`A = diag(1 + 4e-6, 1/(1 + 4e-6), 1)`: `det(A) = 1` to machine precision while
`max|AᵀA - I| = (1 + 4e-6)^2 - 1 = 8.000008e-6` sits above `1e-6` and below the
old `1.1e-5` slack. Via the real matcher seam it builds an `ok` graph that
composes at **all budgets zero** to `POSE_GRAPH_SOURCE_INVALID` with empty
poses/components and `orthonormal` in the detail. A companion valid
near-boundary control uses `A = diag(1 + 4e-7, 1/(1 + 4e-7), 1)`, whose Gram
defect of `~8e-7` is just inside the bound, and must still compose `ok`. The
existing explicit-tolerance control (`1e-9` admitted, `1e-3` refused) is
retained unchanged.

## 5. Determinism

- Components are the accepted graph's canonical solved components (ordered by
  first frame, isolated frames as singletons); the anchor is `frames[0]`.
- Each component's spanning forest is a breadth-first tree whose neighbour
  lists are sorted by the source graph's declared frame order, so tree edges and
  positions are order-invariant.
- Chords are checked in canonical pair order; classification order does not
  affect the outcome (any contradiction refuses the whole result).
- `budget_sha256`, `pose_graph_sha256` and every pose payload are SHA-256 over
  canonical sorted-key JSON, so two runs over equal graphs and budgets are
  byte-identical. This is controlled both by re-running the composer on one
  graph and by two independently built graphs with identical evidence.

## 6. Controls (`tests/test_image_derived_pose_graph.py`, 29 tests)

Every test injects the accepted connection adapter's declared `matcher` seam, so
no SIFT runs and no pixels are read; the evidence graph is produced by the real
`build_image_derived_connections` over a session whose only monkeypatched piece
is `image_only.solve_image_cameras`.

- **Chain composition** - `(a,b)` and `(b,c)` give `a` identity/zero, `b` at
  `b_ab` with `R_ab`, and `c` at `b_ab + b_bc` with `R_bc R_ab`; every tree step
  is exactly one unit long.
- **Gauge transport (convention lock)** - with `right_yaw_class = 1` on both
  chain edges, `c` is placed at `(1, 0, 1)` (the `(b,c)` baseline rotated by
  `Ry90(-1)`), not the untransported `(2, 0, 0)`. This is the regression that
  pins the bound convention.
- **Disconnected components** - `((a,b),(c,d))` anchors both components
  independently with the right `component_id`/`anchor_frame_id`.
- **Isolated frames** - an all-refused graph emits identity/zero poses with
  singleton components, including at a zero cycle budget.
- **Cycle-consistent** - a triangle of unit directions that close with positive
  lengths and rotations that multiply to the identity composes without
  refusal, even at a `1e-12` direction tolerance.
- **Longer consistent cycle** - a four-frame loop whose origin lies in the
  convex hull of four signed unit directions is admitted at exactly 12
  elementary checks (one rotation plus eleven Caratheodory subsets) and refused
  at 11, exercising the size-four face.
- **Cycle-inconsistent (rotation)** - a 20-degree loop rotation refuses with
  `pose_graph_cycle_rotation_inconsistent` and no poses/components.
- **Cycle-inconsistent (direction)** - closing rotations but three signed unit
  directions confined to one quadrant refuse with
  `pose_graph_cycle_direction_inconsistent` and no poses/components.
- **Edge budget** - one solved edge over a budget of one refuses before any
  composition with `pose_graph_budget_exhausted`.
- **Cycle budget** - a triangle is admitted at five elementary checks, refused
  at four, and refused immediately at zero (fundamental-cycle lower bound); a
  chain and an isolated graph are admitted at zero.
- **Budget self-validation** - negative caps, negative/over-range tolerances and
  a wrong digest all raise; the record exposes a 64-character `budget_sha256`.
- **Malformed graph** - a tampered `graph_sha256` and a tampered evidence field
  both refuse as `pose_graph_source_invalid` even at a zero edge budget.
- **Non-SO(3) solved rotation (Sol's finding, adversarial zero-budget
  regression)** - a graph built through the real matcher seam with
  `relative_rotation = diag(2, 1, 1)` is a self-consistent `ok` graph (finite
  `3x3` passes the accepted evidence constructor and the replay digest), yet
  composes at **all budgets zero** to `pose_graph_source_invalid` with empty
  poses/components and `orthonormal` in the detail - never
  `pose_graph_budget_exhausted` and never a raw `ValueError`.
- **Reflection determinant** - `diag(1, 1, -1)` is orthonormal but has
  determinant `-1`, so it is refused as `pose_graph_source_invalid` with
  `determinant +1` in the detail, pinning the explicit determinant check.
- **Explicit SO(3) tolerance** - a `1e-9` orthonormality defect on an otherwise
  valid rotation composes, while a `1e-3` defect refuses as
  `pose_graph_source_invalid`, showing the check is the declared `1e-6`
  tolerance and not exact equality.
- **Near-boundary Gram defect inside the bound (valid control)** - the
  determinant-preserving `diag(1 + 4e-7, 1/(1 + 4e-7), 1)`, whose Gram defect is
  `~8e-7 < 1e-6`, still composes `ok` and its rotation is emitted unchanged,
  pinning that `rtol=0.0` did not make the check stricter than declared.
- **Near-boundary Gram defect outside the bound (Sol's final finding,
  adversarial zero-budget regression)** - the determinant-preserving
  `diag(1 + 4e-6, 1/(1 + 4e-6), 1)` has `det = 1` and `max|AᵀA - I| = 8.000008e-6`
  (above `1e-6`, below NumPy's old `1.1e-5` slack). Through the real matcher seam
  it builds a self-consistent `ok` graph, yet composes at **all budgets zero** to
  `pose_graph_source_invalid` with empty poses/components and `orthonormal` in
  the detail - never admitted, never a budget stop, never a raw `ValueError`.
- **Non-finite rotation** - a `NaN` rotation injected past the constructor is
  still turned into `pose_graph_source_invalid` by the `dataclasses.replace`
  replay rather than a raw `ValueError`.
- **Refused source** - a `refused` connection graph refuses as
  `pose_graph_source_refused` with no poses/components.
- **Input contract** - a non-graph and a non-budget raise
  `ImageDerivedPoseGraphError` with `pose_graph_input_invalid`.
- **Determinism** - one graph re-run and a second, independently built graph
  give equal `pose_graph_sha256`, equal records and equal pose payloads.
- **No-scale / no-geometry** - `metric_scale_known is False`, the session
  `unresolved` facts and both source digests are preserved, and the recursive
  record keys contain no `image`/`pixels`/`world`/`bounds`/`transform`/`block`/
  `volume`/`face`/`observation`/`translation`/`manifest`/`litematic`/`export`/
  `cli` keys; every position is three finite numbers.
- **Pose invariants** - a non-orthonormal rotation and a non-finite position are
  refused by the pose constructor.
- **Real-matcher binding** - the shipped `match_feature_frames` solves a
  three-camera scene whose declared yaw representatives are 85/5/15 degrees
  (true 85/95/105); all three edges solve with classes `[1, 1, 0]`, and the
  composer admits the triangle at a `1e-12` direction tolerance and a `1e-9`
  degree rotation tolerance. Instrumenting the same scene measured a **0.0
  degree** rotation cycle and a **9.0e-17** direction hull distance, so the
  convention binds exactly on production matcher output.

## 7. Evidence (exact commands and results)

Workspace interpreter `.venv\Scripts\python.exe`, Python 3.10.11, NumPy 2.2.6,
OpenCV 5.0.0:

```
.venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\pipeline\image_derived_pose_graph.py tests\test_image_derived_pose_graph.py
    -> exit 0
.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_image_derived_pose_graph.py
    -> 29 passed
.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_image_derived_connections.py tests/test_image_derived_session.py tests/test_project_store.py
    -> 99 passed (connections 24 + session 45 + store 30)
```

Combined required run: **128 passed** (pose graph 29 + connections 24 + session 45
+ store 30).

Additional adversarial confirmation (not part of the graded suites): at all-zero
budgets, `diag(2, 1, 1)`, `diag(0, 1, 1)`, a reflection `diag(1, 1, -1)`,
`3*I` and a shear matrix each returned `refused` /
`pose_graph_source_invalid` with `poses = 0` and `components = 0`, with no
raised exception; a valid rotation still composes `ok`.

Boundary checks:

```
imports in the new module: stdlib, numpy, and
  mc_photogrammetry.pipeline.image_derived_connections only
  (unchanged by the repair)
forbidden-token scan of the new module and its tests:
  only docstring non-claims and the test's forbidden-key assertion set
```

Scope check: only the three routed paths were written. `AGENT_COMMS.md` was not
modified by this worker; `AGENT_HANDOFF.md` was not touched. No git
stage/commit/reset/clean was run.

Ruff is not installed in the workspace interpreter, so no lint run is claimed.

## 8. Limits and non-claims

1. This is a bounded relative composition, not a reconstruction. It computes no
   metric scale, no translation magnitude, no world coordinate, no bound and no
   scene transform; `metric_scale_known` stays `False`, positions are unit-edge
   gauge only, and the session's unresolved facts are preserved.
2. The emitted pose is a gauge record, not a rigid world transform: positions
   live in the component anchor's declared world-yaw frame while rotations are
   relative to the anchor camera frame. The residual global orientation of the
   anchor's own estimate is not carried by the graph and is never invented.
3. The composition depends on the accepted evidence's `right_yaw_class`; the
   guard check showed that `relative_rotation` and `baseline_direction` alone
   cannot bind, and that DS-15's `R(i,j)^T` transport is frame-inconsistent for
   the accepted evidence.
4. The direction closure tests whether the origin lies in the convex hull of the
   cycle's signed unit directions (positive-length closure). It does not
   estimate edge lengths, and Caratheodory subset enumeration is exact but
   charged to the cycle-check budget, so an over-large cycle refuses typed
   rather than running unbounded.
5. A solver claim is evidence, not validated truth: a graph whose cycles happen
   to close is admitted even if a pair is a false solve. The composer
   re-adjudicates nothing the matcher already decided.
6. Production wiring (normal-mode orchestration, state fusion, `.litematic`
   promotion) remains out of scope and blocked pending separate acceptance.
