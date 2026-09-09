# M5 shape-balanced partial fitting

> Lead follow-up: partial-shape scheduling, material selection, and palette-order controls implemented. Matched Template 2 evidence is evaluation-only; see [the diagnostic](M5-t2-matched-diagnostic.md).

GPT-5.6 Luna, 2026-09-09.

Astra's M5 review found that a complete palette could spend the default 256
candidate-render budget inside roughly 700 full-cube materials before the first
slab, carpet, or other partial geometry was examined. `fit_states` now performs
a deterministic representative render for every geometry hypothesis before
material refinement. Full-cube material variants remain in the vocabulary, but
they no longer monopolise the bounded geometry-search budget. Shape order is
stable and puts non-full geometry before the full-cube group, independent of
palette insertion order.

After geometry representatives are scored, rectified texture evidence ranks
materials within each shape. At most the best material refinement is rendered
per shape under the same global cap; air and the current state remain explicit
baselines. Diagnostics record considered representative states, material
refinements, render counts, and unresolved budget exits. A cube-heavy synthetic
control with 100 cube materials and slab/carpet states reaches both partial
shapes within a three-render cap.

`FitStateResult` remains a `dict` for compatibility while carrying the exact
supporting frames for each accepted edit. The raycast integration consumes that
required result channel directly, so enabling optional diagnostics cannot change
block provenance or silently fall back to every input frame.

Focused shape-fit, raycast, and CLI tests pass. The workspace currently has 483 collected tests, and the final full `.venv` run passes all 483. No fixture sweep
or production-quality claim is made by this bounded control.

## L1/L2 material-selection correction

The representative geometry probe is no longer inserted as a competing final
material result. Each geometry now produces one final material decision from
its own rendered scores, with texture cost used when available and a deterministic
RGB prior used when texture scoring is disabled. The representative remains a
fallback when material patches are missing. Refinement is capped at four materials
per shape, preserving the global render budget and palette-order independence.

Forward, reversed, and shuffled two-material slab controls all select the
evidence-matching material; cube-heavy slab/carpet/cube controls and multiview cap
controls remain green. The final full `.venv` run passes all 483 collected tests.




