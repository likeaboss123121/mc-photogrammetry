# M5 R5 bounded partial-fit validation

GPT-5.6 Luna, 2026-09-09. This is a tiny synthetic control, not a fixture
quality claim. The scene contains one empty target cell and a photographed
quartz bottom slab at that cell. The renderer uses three identical views in the
multi-view cap control and a 0.5 render scale.

## Work-budget contract

`max_candidate_renders` is a global candidate-render budget across cells,
materials and views. The check occurs immediately before every candidate render.
A hypothesis that does not complete every usable view is discarded and cannot be
accepted. Background, context, probe, current-state, ownership and candidate
renders are all reported separately, with `shape_fit_render_work` giving their
sum. The default diagnostic budget is 256 candidate renders; callers can choose
a smaller explicit budget.

## Results

| views | cap | candidate renders | total render work | elapsed (s) | result |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0 | 4 | 0.0041 | no proposal |
| 1 | 1 | 1 | 6 | 0.0034 | complete one-view carpet hypothesis |
| 1 | 8 | 2 | 7 | 0.0032 | recovered `minecraft:quartz_slab[type=bottom]` |
| 3 | 1 | 1 | 13 | 0.0059 | no partial multi-view hypothesis accepted |

The one-view rows disable texture scoring to isolate geometry; the three-view
row uses the default texture term and therefore includes three ownership renders.

The three-view cap-1 run verifies that a candidate is not scored on one view
and accepted selectively. The cap-8 run recovers the known slab in the tiny
control. No complete-palette or real-fixture sweep was run.

## Verification

- Full suite: **415 passed in 163.52 seconds** before the R5 additions.
- Post-R5 focused suites (`test_shape_fit.py`, `test_cli.py`): **39 passed**.
- `compileall -q src` and `git diff --check` pass.
- Implementation commits: `73c2e63` (R2/R4 scoring) and `8829c8c` (R5 cap,
  accounting, default and tests).

R5's bounded work-control and tiny-scene evidence are complete. Matched T2/T6
baseline/candidate quality evidence and full M5 promotion remain gated on the
reviewed masks and M4-supported geometry; this report does not claim those gates.
