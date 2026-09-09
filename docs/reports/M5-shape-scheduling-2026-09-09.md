# M5 shape-balanced partial fitting

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

Focused shape-fit, raycast, and CLI tests pass. The workspace currently has 461
collected tests, and the final full `.venv` run passes all 461. No fixture sweep
or production-quality claim is made by this bounded control.
