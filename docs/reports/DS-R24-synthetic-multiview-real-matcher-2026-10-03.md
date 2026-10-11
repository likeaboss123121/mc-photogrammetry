# DS-R24 synthetic four-view real-matcher experiment — 2026-10-03

Bounded foreground experiment authorized by the DS-R24 entry in
`AGENT_COMMS.md` (2026-10-03 11:48 UTC). It extends the accepted DS-R23
two-frame experiment to one **pre-registered** four-view layout. The objective
is to render four fixed synthetic views in memory, build one coherent synthetic
connection/pose graph whose only injected content is the oracle relative yaw
used for solver conditioning, measure the lattice and derive signatures for
real, then run the **unpatched** real `plane_graph.match_plane_graph` and the
**unpatched** real `plane_graph.solve_component` on identical evidence, plus a
same-input empty-yaw-constraint ablation.

No source or test file was or will be edited. No real image, capture, fixture,
reference or mask file was read, hashed or processed. No adaptive view changes,
benchmark, live diagnostic, Minecraft/v5 path, gate/threshold change,
background/persistence or system change is involved. This report is the only
repository addition.

This section is the **pre-registration**. It was written before any view was
rendered, and the rendered frames, graph, lattice, signatures, matcher and
solver below are produced only from these fixed constants.

## Pre-registered fixed four-view layout

Renderer and scene are the existing procedural controls, used unchanged:

- room `ROOM = (0.0, 12.0, 0.0, 6.0, 0.0, 12.0)` (x, y, z bounds);
- material function `mixed_material` (the `(bx*3 + by*5) % 3` palette);
- `render_room(camera, 360, 240, ROOM, materials=mixed_material)`;
- per-frame calibration via the existing `calibration(camera, width=360,
  height=240)` helper (focal from the vertical FOV, yaw modulo 90, pitch, no
  roll, centred principal point).

Frame size: **360 × 240** for every frame. Frame order is exactly
`f0, f1, f2, f3`. Every camera uses the default `eye_height = 1.62`, so the
optical centre is `position + (0, 1.62, 0)`.

| order | frame | position (x, y, z) | yaw° | pitch° | fov° | eye (x, y, z) |
|-------|-------|--------------------|------|--------|------|----------------|
| 0 | f0 | (3.6, 0.7, 3.6) | 10.0 | −25.0 | 70.0 | (3.6, 2.32, 3.6) |
| 1 | f1 | (5.0, 0.7, 3.6) | 10.0 | −25.0 | 70.0 | (5.0, 2.32, 3.6) |
| 2 | f2 | (3.6, 0.7, 5.0) | 10.0 | −25.0 | 70.0 | (3.6, 2.32, 5.0) |
| 3 | f3 | (5.0, 0.7, 5.0) | 10.0 | −25.0 | 70.0 | (5.0, 2.32, 5.0) |

f0 and f3 are the two accepted DS-R23 camera positions; f1 and f2 are the other
two corners of the same axis-aligned 1.4 × 1.4 rectangle. The four positions
were chosen before rendering to give an assembled lattice and a direction set
that positively spans each fundamental cycle; no view was changed after any
result was seen.

## Pre-registered coherent synthetic oracle graph

The connection graph is the accepted adapter
`build_image_derived_connections(session, budgets, matcher=oracle)` with the
**oracle matcher seeded solely from the pre-registered camera geometry**. It is
not the real matcher; the real matcher is left untouched and runs later on the
real lattice/signatures.

- Pair order is canonical over `f0..f3`: `f0→f1, f0→f2, f0→f3, f1→f2, f1→f3,
  f2→f3` (6 pairs).
- Because every pre-registered camera shares the same yaw (10.0°) and pitch
  (−25.0°), the true world-to-camera rotations are equal, so the oracle relative
  rotation `R_right @ R_leftᵀ` is exactly the identity for every pair.
- Because the shared yaw is 10.0° ≡ 10.0° (mod 90°), the oracle relative yaw
  class `(k_right − k_left) mod 4` is **0** for every pair.
- The oracle unit baseline direction `b = unit(eye_right − eye_left)` in the
  left view's declared frame (all gauges 0) is, over `(x, y, z)`:

  | pair | oracle unit baseline |
  |------|----------------------|
  | f0→f1 | (1, 0, 0) |
  | f0→f2 | (0, 0, 1) |
  | f0→f3 | (0.7071067811865476, 0, 0.7071067811865476) |
  | f1→f2 | (−0.7071067811865476, 0, 0.7071067811865476) |
  | f1→f3 | (0, 0, 1) |
  | f2→f3 | (1, 0, 0) |

Each oracle pair reports `status="solved"`, `support=6`, `support_floor=8`,
`raw_keypoints=(6,6)`, `candidate_edges=12`, `mutual_survivors=6`,
`median_selected_residual_degrees=0.4`, `parallax_degrees=5.0`,
`cheirality_fraction=1.0`, `singular_ratio=0.01`, `ambiguity_margin=3`, and no
hypotheses, matching the accepted DS-R22B/R22C fixture record shape.

Composition uses the accepted
`compose_image_derived_pose_graph` with fixed budgets `max_edges=6`,
`max_cycle_checks=100`, `rotation_tolerance_degrees=2.0`,
`direction_tolerance=0.05`. The true per-cycle direction combination has
strictly positive length weights (`|AB|·b_AB + |BC|·b_BC = |CA|·b_CA`), so the
origin lies in the convex hull and direction closure is residual 0.

The oracle graph's only role is to supply
`plane_graph.solve_component(..., quadrant_constraints=...)`; signature
derivation, the real matcher, connection-constraint extraction, the solver and
the ablation are all real, unpatched code.

## Pre-registered fixed evidence constants

| constant | fixed value | source |
|----------|-------------|--------|
| matcher shift radius | `MAX_SHIFT_CELLS = 32` | `scripts/diagnose_image_derived_axis_coverage.py` |
| integer tolerance | `INTEGER_TOLERANCE = 0.15` | same |
| minimum resultant | `MIN_RESULTANT = 0.90` | same |
| ambiguity margin | `AMBIGUOUS_MARGIN = 0.02` | same |
| lattice rectified-side ceiling | `LATTICE_MAX_RECTIFIED_SIDE = 512` | `image_derived_lattice_measure.py` |

Ablation rule (fixed in advance): the ablated solve receives the **identical**
ordered reading object and the **identical** ordered match objects as the
baseline, with only `quadrant_constraints` changed from the connection graph
rows to `()`.

Stop rule (fixed in advance): if the measured lattice is not `assembled`, the
run reports that and stops; no view, camera, scene or constant may be tuned
after observing a result.

## Execution results

### Exact commands and artifacts

Working directory: repository root. Python: project venv, 3.10.11.

Pre-solver inspection over the same pre-registered views (used only to read the
measured readings and the graph constraints before wiring the solver; it changed
no view, camera or constant):

```powershell
& .\.venv\Scripts\python.exe "$env:TEMP\ds_r24_inspect.py"
```

Final driver, run twice (run 1 and run 2), each writing its own UTF-8/LF
summary file:

```powershell
& .\.venv\Scripts\python.exe "$env:TEMP\ds_r24_probe.py" "$env:TEMP\ds_r24_out1.json"
& .\.venv\Scripts\python.exe "$env:TEMP\ds_r24_probe.py" "$env:TEMP\ds_r24_out2.json"
```

| artifact | location | bytes | SHA-256 |
|----------|----------|-------|---------|
| final driver | `%TEMP%\ds_r24_probe.py` | 17358 | `B1A73BC63D0B2DD13D1528431CFFFF148CFCE4E4580DC350A22782BDC077921D` |
| inspection driver | `%TEMP%\ds_r24_inspect.py` | 3114 | `B6B887EA6D47D0946D5AAAA710DCBD06E7154468BDA716E68BF3D41B8A91BB63` |
| run 1 summary | `%TEMP%\ds_r24_out1.json` | 15888 | `4A8CB9CD582C3200C8A8F11CFC294AF6853CD86927AD15CD1BB9D93168283E50` |
| run 2 summary | `%TEMP%\ds_r24_out2.json` | 15888 | `4A8CB9CD582C3200C8A8F11CFC294AF6853CD86927AD15CD1BB9D93168283E50` |

A first driver invocation (same pre-registered layout) exited 1 because the
literal full-graph baseline call raised; the driver was then revised only to
record that raised outcome in the summary and to add the applicable-subset and
empty-constraint runs. No view, camera, FOV, frame order or solver/matcher
constant changed, and the final driver above is the one that produced these
results.

### Session / graph / lattice provenance

| item | value |
|------|-------|
| session digest | `0010ff4c329b6ecb79837f9e7f2d4a637fb4e69f9a8c4eb9607b7e793f1eee7b` |
| connection graph status | `ok` |
| connection graph digest | `1b380e1d802ac0a3af06836a535b14b583d9156eea920ef3e9af6bcfe288aac7` |
| pose graph status | `ok` (component `[f0, f1, f2, f3]`) |
| pose graph digest | `b971fda391d92d867a67bd69e1bd76e1b0288a97768e66cf65d2b79ee9dee7ce` |
| lattice status | `assembled` (reason empty; 0 abstentions) |
| lattice graph digest | `1b380e1d802ac0a3af06836a535b14b583d9156eea920ef3e9af6bcfe288aac7` |
| lattice result digest | `e2b6908df04e2a2ba685feb35f25bfe609277d556f5f3d0ffca3781d39d01aad` |
| readings / signatures | 6 readings, 16 signatures |

Pose rows confirm the oracle graph is internally coherent: all four rotations
are identity, gauge closure is trivial, and the unit-edge positions are
`f0 (0,0,0)`, `f1 (1,0,0)`, `f2 (0,0,1)`, `f3 (0.7071067811865476, 0,
0.7071067811865476)`, i.e. the true blocked baselines of the pre-registered
rectangle. The pose graph composed with no cycle refusal.

**The pre-registered view `f1` produced no lattice reading.** The measured
lattice therefore contains only `f0`, `f2`, `f3` (2 axes): `f0` and `f2` each
measured axis 0 ×1 and axis 2 ×1, and `f3` measured axis 1 ×1 and axis 2 ×1.
The lattice's `frame_ids` still name all four session frames, and no lattice
abstention record was emitted for `f1`. This is a real outcome of the fixed
layout; the layout was not changed.

### Per-frame readings (measured lattice)

| frame | axis | side | structural_share | retained b alternatives |
|-------|------|------|------------------|-------------------------|
| f0 | 0 | −1 | 0.6467179242714372 | 1.0, 0.5 |
| f0 | 2 | +1 | 0.7731308927998708 | 1.0, 0.25, 0.5, 0.125 |
| f2 | 0 | −1 | 0.6058246346531586 | 1.0, 0.5 |
| f2 | 2 | +1 | 0.7463785883279832 | 1.0, 0.5, 0.25 |
| f3 | 1 | +1 | 0.1276797457356359 | 1.0 |
| f3 | 2 | +1 | 0.7445951786825442 | 1.0, 0.25, 0.5, 0.125 |

No reading for `f1`. Axes seen: 0, 1, 2, but never all three in one frame pair.

### Per-frame signatures (real derivation)

`derive_image_derived_signatures` status `assembled`; 16 signatures total (one
per reading alternative).

| frame | axis 0 | axis 1 | axis 2 | total |
|-------|--------|--------|--------|-------|
| f0 | 2 | 0 | 4 | 6 |
| f2 | 2 | 0 | 3 | 5 |
| f3 | 0 | 1 | 4 | 5 |

### Real matcher (`match_plane_graph`, `max_shift_cells=32`)

Trace counters over the 3 measured frames: 340 pre-admission candidates
(85 signature pairs × 4 quadrants), 116 pair-correlation admissions, 15 after
the same-axis equality filter, 340 pair rows.

Refusals: **325 total** — `axis_mismatch` 192, `same_axis_mismatch` 101,
`no_signal` 17, `repetition_unresolved` 12, `insufficient_overlap` 3
(224 pre-admission refusals + 116 admissions = 340; 101 post-admission drops +
15 kept = 116).

| frame pair | refusals | admitted |
|------------|----------|----------|
| f0→f2 | 113 | 7 |
| f0→f3 | 116 | 4 |
| f2→f3 | 96 | 4 |

Per-reason refusal axis distributions (left axis / right axis):

| reason | left axes {0,1,2} | right axes {0,1,2} |
|--------|-------------------|--------------------|
| axis_mismatch | 68 / 0 / 124 | 24 / 44 / 124 |
| same_axis_mismatch | 25 / 0 / 76 | 13 / 0 / 88 |
| no_signal | 16 / 0 / 1 | 5 / 0 / 12 |
| repetition_unresolved | 2 / 0 / 10 | 2 / 0 / 10 |
| insufficient_overlap | 2 / 0 / 1 | 1 / 0 / 2 |

**Admitted matches: 15.** Left endpoint axes {0: 7, 2: 8}; right endpoint axes
{0: 3, 2: 12}; relative quadrants {0: 4, 1: 4, 2: 3, 3: 4}. All 15 are
cross-frame identities admitted under an unconstrained quadrant hypothesis;
most are wrong-axis or wrong-quadrant candidates.

| # | left | right | Lax | Rax | Lp | Rp | Lb | Rb | q | psr | score | overlap | shift rc |
|---|------|-------|-----|-----|----|----|----|----|---|-----|-------|---------|----------|
| 1 | f0 | f2 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | 0 | 4.2497 | 0.4681 | 256 | (−8, −3) |
| 2 | f0 | f2 | 0 | 2 | 0 | 1 | 1.0 | 1.0 | 1 | 5.4265 | 0.5595 | 241 | (11, 21) |
| 3 | f0 | f2 | 0 | 2 | 0 | 1 | 1.0 | 1.0 | 3 | 6.3337 | 0.6492 | 246 | (11, −10) |
| 4 | f0 | f2 | 2 | 0 | 1 | 0 | 1.0 | 0.5 | 3 | 6.4392 | 0.5339 | 507 | (5, 2) |
| 5 | f0 | f2 | 2 | 0 | 1 | 0 | 1.0 | 1.0 | 1 | 6.3316 | 0.6628 | 2612 | (−4, −1) |
| 6 | f0 | f2 | 2 | 2 | 1 | 1 | 1.0 | 0.5 | 2 | 9.3932 | 0.7531 | 1195 | (0, 2) |
| 7 | f0 | f2 | 2 | 2 | 1 | 1 | 1.0 | 1.0 | 0 | 44.6780 | 0.8445 | 4990 | (0, 0) |
| 8 | f0 | f3 | 0 | 2 | 0 | 1 | 1.0 | 0.5 | 1 | 5.4247 | 0.6562 | 31 | (3, 31) |
| 9 | f0 | f3 | 0 | 2 | 0 | 1 | 1.0 | 1.0 | 3 | 4.7922 | 0.5506 | 267 | (−20, 21) |
| 10 | f0 | f3 | 2 | 2 | 1 | 1 | 1.0 | 0.5 | 2 | 7.5252 | 0.7114 | 575 | (1, −2) |
| 11 | f0 | f3 | 2 | 2 | 1 | 1 | 1.0 | 1.0 | 0 | 7.9208 | 0.7208 | 2415 | (1, 3) |
| 12 | f2 | f3 | 0 | 2 | 0 | 1 | 0.5 | 0.5 | 3 | 5.9557 | 0.6174 | 305 | (−1, −4) |
| 13 | f2 | f3 | 0 | 2 | 0 | 1 | 0.5 | 1.0 | 1 | 12.2004 | 0.6452 | 340 | (1, 1) |
| 14 | f2 | f3 | 2 | 2 | 1 | 1 | 0.5 | 1.0 | 0 | 6.1003 | 0.6769 | 1179 | (0, −3) |
| 15 | f2 | f3 | 2 | 2 | 1 | 1 | 1.0 | 0.5 | 2 | 7.6401 | 0.6905 | 584 | (1, −2) |

### Graph constraints (`_connection_constraints`)

Status `ok`. Six solved rows, all from the coherent oracle graph, all with
relative yaw 0:

`f0→f1:0, f0→f2:0, f0→f3:0, f1→f2:0, f1→f3:0, f2→f3:0`.

Because `f1` has no reading, three of those rows name a frame that is foreign to
the solver's measured reading universe `{f0, f2, f3}`.

### Real solver (`solve_component`; thresholds 0.15 / 0.90 / 0.02)

**Literal full-graph baseline (all 6 rows) — the unpatched solver rejects it:**

- raised `ValueError`: `quadrant constraint is malformed or names a foreign frame`.

This is an observed property of the unpatched code path: `solve_component`
validates every constraint frame against the frames present in the reading
universe, and `f1` is absent. No product or test change was made; this is
reported as a finding.

**Conditioned on the applicable rows (`f0→f2:0, f0→f3:0, f2→f3:0`), identical
readings and matches:**

- gate (`reason`): **`insufficient_axis_coverage`**
- matched world axes: `[0, 2]`
- resultants: `{0: 1.0, 2: 0.004952028968268453}`
- residuals: `{0: 3.8981718325193755e-17, 2: 0.2755459512031173}`
- quadrants: `{f0: 0, f2: 0, f3: 0}`; retained alternatives 1; `scale_known` false
- translations: f0 (0,0,0), f2 (12.347917083967229, 0, 9.53924165427111),
  f3 (0, 0, 3.7905318795019425)
- world-axis matched counts `{0: 1, 2: 3}`

Only two world axes carry any active match under the oracle gauge, so the
three-axis requirement is unreachable and the solver abstains before the
resultant threshold is consulted; axis 2's resultant is nevertheless very low
(0.00495), i.e. the admitted axis-2 identities selected under the oracle gauge
are mutually inconsistent.

**Same-input empty-yaw-constraint ablation (`quadrant_constraints = ()`),
identical ordered reading and match objects:**

- gate (`reason`): **`ambiguous_quadrant`**
- matched world axes: `[0, 2]`
- resultants: `{0: 1.0, 2: 0.9999999999999999}`
- residuals: `{0: 3.8981718325193755e-17, 2: 9.483978957577537e-15}`
- quadrants: `{f0: 0, f2: 0, f3: 2}`; retained alternatives 8; `scale_known` false
- translations: f0 (0,0,0), f2 (12.347917083967229, 0, 3.8698967957148525),
  f3 (0, 0, 52.09901055154954)
- world-axis matched counts `{0: 1, 2: 3}`
- ablation changed the typed outcome: `true`

So the oracle gauge and the empty gauge produce different typed results: the
oracle forces `f3` to quadrant 0 and leaves axis 2 unresolved/ inconsistent
(`insufficient_axis_coverage`), while dropping the gauge lets the solver pick
`f3 = 2`, which makes axis 2 numerically clean but leaves a rival assignment
within the 0.02 margin (`ambiguous_quadrant`). The `f3 = 2` assignment violates
the oracle relative gauge, so the cleaner numbers are not evidence that the
oracle graph is wrong — they are evidence that the matcher's admitted identities
support more than one quadrant reading of the same sparse two-axis evidence.

### Determinism

- In-process: a second real matcher call produced an equal match tuple, and a
  second `solve_component` call with the same applicable rows produced an equal
  payload; baseline and ablation provably reused the identical reading object
  and identical ordered match object (`same_reading_object = true`,
  `same_match_object = true`, `solution_repeat_equal = true`,
  `matches_repeat_equal = true`).
- Cross-process: two independent invocations (run 1, run 2) produced
  byte-identical UTF-8 summaries, SHA-256
  `4A8CB9CD582C3200C8A8F11CFC294AF6853CD86927AD15CD1BB9D93168283E50` each.
- Compact canonical probe digest (excluding the digest field itself):
  `c8ce3d45574f9f64773748dd055a504d146d9d94dc382590accdc685a97b3c96`.

### Limitations

- Synthetic geometry only: a procedural ray-cast room rendered from constants
  in memory. No real image, capture, fixture, reference or mask file was read,
  hashed or processed. This is not real-camera or real-fixture accuracy and not
  an image-causality claim.
- The camera solution is injected through the session's declared solver seam
  (the accepted fixture construction); the connection/pose graph is a synthetic
  oracle used only to bind relative yaw.
- The pre-registered `f1` view produced no lattice reading, so the measured
  component is `{f0, f2, f3}` over 2 axes. A three-axis solve is therefore
  unreachable by construction and both reported gates are typed abstentions,
  not matcher or solver defects.
- The literal full-graph constraint set includes rows naming the reading-less
  `f1`, and the unpatched solver rejects that set with `ValueError`. The
  applicable-subset conditioning (3 of 6 rows) is a clearly labelled post-hoc
  reduction forced by that outcome, not a change to the pre-registered layout;
  the layout, cameras, FOV, frame order and all fixed constants are unchanged.
- The oracle relative yaw 0 is coherent with the pre-registered identical camera
  orientations and the pose graph closes every cycle, but it is an injected
  oracle value, not an image-derived yaw. All reported solver behaviour is
  conditioned on it.
- The real matcher admits 15 identities spanning every relative quadrant,
  including cross-axis pairs; this experiment does not establish a matcher
  rejection rate, and the DS-R24 layout is too sparse to distinguish matcher
  error from solver under-determination.
- No real-media, benchmark, live diagnostic, Minecraft/v5 path, gate/threshold
  change, background/persistence or system modification was involved.

Reported for independent lead/Jev review. No self-acceptance.

---

Report body SHA-256 (UTF-8 bytes of everything above this separator):
`282FB774CF3C51C795220B7E21C4676DB9F7C9671A50EE1C0BAF5B141A04DE17`.
