# M5 F2 atomic scheduling â€” implementation report

Task: `m5-f2-atomic-scheduling-2026-09-19` (bugfix, medium risk)
Repair: `m5-f2-repair-2026-09-19` (bugfix, medium risk)
Date: 2026-09-19
Worker: DeepSeek/Sol-equivalent bounded developer
Status: implemented, repaired and locally verified; independent review pending

## Result

`fit_states` / `_fit_states_sequential` schedule render work in whole
usable-view sets. A candidate hypothesis is only started when every one of its
usable views fits the remaining candidate-render cap; otherwise the cell is
reported as explicitly `budget_unvisited` and its share rolls to the next cell.
Because a hypothesis can no longer be entered with a partial view set, a
prefix-scored candidate can no longer enter the comparison (previously it was
rendered, counted, then rejected). `max_candidate_renders == 0` now performs no
background, context, probe, ownership, current or candidate render and no
palette setup, returning a `budget_unvisited` record for every requested cell.

The repair task closed two scheduler defects that survived the first attempt:

1. **Leftover starvation.** The multi-cell path divided the cap with
   `ceil(remaining / cells_left)` before any cell's usable-view count was known.
   A cell whose whole view set exceeded that fair share was refused, and if the
   later cell also could not fit, the budget was simply left unspent. After
   every cell has had its fair-share turn, unspent budget is now offered back to
   the refused cells in order.
2. **Aggregate cap-reached false negative.** The multi-cell
   `shape_fit_candidate_render_cap_reached` only tested `remaining == 0` or a
   `budget_unvisited` cell status, so a refused representative/refinement with a
   leftover balance reported `False` while every per-cell run reported `True`.
   The aggregate now also mirrors `shape_fit_unresolved_budget_exits`.

## Owned paths and hashes

Scope was exactly the three files below; no other file was edited and no git
mutation was performed.

| Path | Baseline (prior attempt) | After repair (SHA-256) |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/shape_fit.py` | `95790E0A7E3176D0D6E825B39BC2584F794413FF0546BCBB291942028EA34F07` | `0647E71B84F8A2EFD6FEA0E9B1042D2769F344AF4AD7996AEF58A41859DBFD6E` |
| `tests/test_shape_fit.py` | `4EC1B5877D01D6DF0DB7E88D640343271896DD31D5594B1DCFBA6D1579C0F618` | `F32C32A9513C2E2C22A45DB82314B9AAF7B50CDF59E039461441231E54C2B557` |
| `docs/reports/M5-F2-atomic-scheduling-2026-09-19.md` | prior revision of this file | this file |

The baseline `tests/test_shape_fit.py` hash was independently reconstructed by
removing this repair's two added tests and matches the prior attempt's recorded
value exactly; the `shape_fit.py` line count also matches the prior attempt
(1273 after repair minus 72 net added lines = 1201).

Line counts after repair: `shape_fit.py` 1273, `tests/test_shape_fit.py` 1062.

## Behavior implemented

1. **Atomic reservation.** `reserve_usable_views(len(views))` gates:
   - the per-cell context renders,
   - each geometry representative hypothesis,
   - each material-refinement hypothesis.
   It returns true only when `candidate_renders + len(views) <= cap` (or the cap
   is `None`). Inside a reserved hypothesis the inner view loop has no cap
   checks, so exactly `len(views)` candidate renders are spent or none are.
2. **Discovery is separate work.** The full-cube probe render, which is the only
   way to learn a cell's exact usable-view set, still runs when a candidate
   budget remains. It is counted only in `shape_fit_probe_renders` and is never
   charged as candidate work. Context renders were moved to *after* the
   reservation so a refused cell pays no context preparation.
3. **`cap=0` is inert.** An early return before `candidate_states`, palette-index
   construction or any frame loop emits `shape_fit_cell_results` entries with
   `status="budget_unvisited"`, `usable_views=0`, `candidate_renders=0`, all
   render counters `0`, `shape_fit_completed_hypotheses=0` and
   `shape_fit_budget_unvisited_cells=len(cells)`.
4. **Later exhaustion skips preparation.** At the top of each cell, a fully
   spent cap marks the cell `budget_unvisited` without any render. A refusal
   with cap remaining does not break the search, so a cheaper eligible cell can
   still use the scarce budget (`test_uneven_view_costs_...`).
5. **Leftover recovery (repair).** The multi-cell path keeps its fair-share
   first pass unchanged: `allocation = ceil(remaining / cells_left)`, a refused
   cell rolls its share forward, and the existing
   `test_two_cells_two_views_cap_two_completes_one_hypothesis_only` behaviour is
   preserved. Cells refused with budget still unspent are recorded, and a
   second pass offers each of them the whole remaining budget in sorted order.
   The atomic reservation still refuses a view set that does not fit, so the
   recovery can only spend the cap on complete hypotheses.
6. **Aggregate cap-reached (repair).** The aggregate flag is now true when the
   shared cap was fully spent, any cell was left `budget_unvisited`, or any
   `shape_fit_unresolved_budget_exits` entry exists. Per-cell results and list
   diagnostics come from each cell's decisive attempt, so a superseded refusal
   that the recovery later resolved does not survive; render counters and stage
   seconds still sum every attempt (a refused cell always spent zero candidate
   renders, so per-cell `candidate_renders` continue to sum to the aggregate).
7. **Diagnostics.** Scalar keys, faithful in the single-cell and aggregated
   multi-cell paths:
   - `shape_fit_completed_hypotheses` â€” count of hypotheses rendered over their
     entire usable-view set.
   - `shape_fit_budget_unvisited_cells` â€” count of cells whose final status is
     budget-refused.
   - `shape_fit_unresolved_budget_exits` entries carry `usable_views` and
     `remaining_candidate_renders`, and the phase `reserve_usable_view_set`
     distinguishes a cell-level refusal from `shape_representative` /
     `material_refinement`.
   - `shape_fit_candidate_render_cap_reached` follows the rules in (6).

## Exact boundary evidence (focused)

`test_atomic_view_reservation_below_equal_and_above_cap` uses one geometry with
two materials and two usable views (one `SLAB` shape, `aaa_slab`/`bbb_slab`):

| cap | candidate renders | completed hypotheses | cell status |
| --- | --- | --- | --- |
| 1 (below one view set) | 0 | 0 | `budget_unvisited` |
| 2 (equal) | 2 | 1 | `partly_tested` (refinement refused) |
| 3 (one over) | 2 | 1 | `partly_tested` (refinement refused atomically) |
| 4 (full) | 4 | 2 | `accepted` |

**Regression 1 â€” two-view then three-view, cap 2.**
`test_two_view_then_three_view_cap_two_does_not_starve_the_budget` places the
two-view cell first (`(0,0,0)`) and the three-view cell second (`(4,0,0)`):

| | cell `(0,0,0)` (2 views) | cell `(4,0,0)` (3 views) | candidate renders | completed | cap reached |
| --- | --- | --- | --- | --- | --- |
| before repair | `budget_unvisited`, 0 | `budget_unvisited`, 0 | 0 | 0 | `True` |
| after repair | `accepted`, 2 | `budget_unvisited`, 0 | 2 | 1 | `True` |

The pre-repair rows were reproduced in a scratch process that reimplemented the
prior multi-cell fair-share loop against the current sequential path and ran the
same fixtures; no workspace file was changed.

**Regression 2 â€” two-cell cap 7 aggregate cap-reached.**
`test_two_cell_cap_seven_reports_the_shared_cap_as_reached` uses two two-view
cells and a three-shape palette; both cells exhaust their allocation and are
refused an atomic representative hypothesis, leaving one render unspent:

| | per-cell renders | statuses | unresolved exits | aggregate cap reached |
| --- | --- | --- | --- | --- |
| before repair | `[4, 2]`, total 6 | `partly_tested`, `partly_tested` | 2 | **`False`** |
| after repair | `[4, 2]`, total 6 | `partly_tested`, `partly_tested` | 2 | **`True`** |

Other pinned cases:

- `test_candidate_render_cap_below_view_count_reserves_no_partial_hypothesis`:
  three usable views, cap 1 â†’ 0 candidate renders, `context_renders == 0`,
  `probe_renders >= 1`, `budget_unvisited`.
- `test_zero_cap_reports_every_cell_budget_unvisited_without_rendering`: two
  cells, cap 0 â†’ all render counters 0, `considered_shapes == []`, two
  `budget_unvisited` records.
- `test_two_cells_two_views_cap_two_completes_one_hypothesis_only`: cap 2 over
  two cells and two views completes exactly one hypothesis (2 renders for one
  cell); the other is `budget_unvisited`; per-cell renders sum to the aggregate.
  The recovery pass deliberately preserves this first-refused/roll-forward
  result.
- `test_uneven_view_costs_let_the_cheaper_cell_use_a_scarce_cap`: cap 1; the
  two-view cell is refused and the one-view cell is reserved and decided.

## Checks performed

Focused:

```
.venv\Scripts\python.exe -m pytest tests/test_shape_fit.py -q -o addopts=''
-> 45 passed in 0.84s
```

Adjacent regressions:

```
.venv\Scripts\python.exe -m pytest tests/test_shape_fit.py tests/test_residual.py \
  tests/test_pixel_authority.py tests/test_face_sampling.py \
  "tests/test_cli.py::test_visible_multiview_exposes_opt_in_partial_shape_fitting" -q -o addopts=''
-> 88 passed in 1.76s
```

Compile:

```
.venv\Scripts\python.exe -m py_compile src/mc_photogrammetry/reconstruction/shape_fit.py
-> exit 0
```

Full suite (dirty working tree, unrelated uncommitted work present):

```
.venv\Scripts\python.exe -m pytest -o addopts='' -q
-> 1699 passed, 1 skipped in 459.46s (0:07:39)
```

The single skip is the pre-existing platform symlink-permission test. The count
is the prior 1697 plus this repair's two regression tests.

## Diagnostics semantics preserved

`shape_fit_seconds`, stage seconds, `shape_fit_render_work`, per-cell `status`,
`considered_shapes`, `material_refinements`, `representative_fallbacks`,
`unresolved_budget_exits` and `supporting_frames` keep their prior meaning.
`partly_tested` still overrides `accepted`/`retained` when any budget exit
exists for the cell, so the equal-cap case above reports a completed hypothesis
with an unfinished refinement rather than a clean acceptance. The only
intentional cross-attempt aggregation change is that list and cell-result
diagnostics reflect the decisive attempt per cell instead of concatenating a
superseded refusal.

## Limitations and honest caveats

- Reservation is applied to candidate and context work, not to the probe render
  that defines the usable-view set. With `0 < cap < views`, a cell still pays its
  probe renders (its own counter) before being refused; only `cap == 0` avoids
  all discovery. This is deliberate: a one-view cell cannot be identified as the
  cheaper eligible task without the probe, which L-D4 requires.
- Recovery is a second scheduling pass, not a global packing optimum. The
  fair-share pass is order-deterministic, and the first deferred cell may spend
  leftover budget that a later, cheaper deferred cell could have used; the
  bound is still one or more complete hypotheses, never a partial one.
- A recovered cell re-runs its background/probe discovery (counted in the
  non-candidate counters) and sees the scene after later cells' edits, because
  it is revisited after the first pass rather than in sorted order.
- Global background renders for the frames still occur whenever `cap > 0`, even
  if every cell is later refused. They are not per-cell preparation and remain
  in `shape_fit_background_renders`.
- `MAX_MATERIAL_REFINEMENTS_PER_SHAPE`, texture weighting, air/current baselines
  and acceptance thresholds were not changed.
- No production path was rewired: `fit_states` is still reached only through the
  existing `raycast.reconstruct_scene_surfaces` developer path. No real-photo
  fixture was run; all evidence is synthetic render-vs-photo fixtures, and a
  textual report cannot establish image-space ground truth.
- The full-suite run includes unrelated uncommitted workspace changes, so it
  measures the tree as-is; no failure was attributable to this change.

## Unresolved concerns / next step

Independent review should re-derive the cap below/equal/above table and the two
regression rows, confirm `cap=0` performs no palette setup, and decide whether
the leftover recovery should choose the deferred cell by view cost rather than
sorted order before M5 production integration.
## Manager clarification and acceptance — 2026-09-19

Astra conditionally accepted the repair after independent verification. The active wording “reserve before candidate/context/probe” is clarified as follows: while positive candidate budget remains, separately counted background/probe discovery may establish the exact usable-view set, because ownership, occlusion, masks, and minimum-pixel visibility cannot be known from projection alone. The complete set is then reserved before candidate renders and associated context preparation. A zero budget, or an exhausted budget before a new cell, performs no palette setup or rendering. Recovery revalidates changed context and counts repeated discovery. This is a bounded fitter acceptance; M5 production integration and real-image promotion remain open.