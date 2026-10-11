# DS-7 — benchmark no-frozen-mask Policy C implementation — 2026-09-12

Implements the DS-6 recommendation (Policy C) only. Owned paths: `scripts/benchmark_templates.py`,
`tests/test_benchmark_templates.py`, this report. No reconstruction/visibility module, mask, threshold,
metric, CLI, M5 or historical artifact changed; no commit/staging/chaining; no benchmark fixture run.

## 1. Migration (exactly DS-6 §4)

`scripts/benchmark_templates.py`:

* `:39` new import `from mc_photogrammetry.vision.gui_mask import create_gui_mask` (single new dependency
  on an existing production module; no new package).
* `:87-96` keep `usable_masks = _load_usable_masks(...)`; record `mask_origin = "frozen"` when the loader
  returned a mapping, else `"derived"` and derive once:
  `{name: ~create_gui_mask(image.shape[0], image.shape[1]) for name, image in frames.items()}`.
  `_load_usable_masks` is untouched, so loader `None` (no paths) / `{}`-to-`None` semantics and its
  `None`/coverage contract are unchanged (`:306-316` still).
* `:97-100` one policy string reused everywhere: `"frozen supplied masks"` /
  `"derived production GUI mask (provisional)"`. Derived runs are never labelled frozen or all-decoded.
* `:130-150` unchanged call sites: the *same* `usable_masks` object is passed to
  `reconstruct_scene_surfaces(..., usable_masks=usable_masks)` and
  `visible_cells(..., usable_masks=usable_masks)`, so evaluator denominator and reconstruction sampling
  now share one object in both modes.
* `:197`, `:210-220`, `:228` report now carries `effective_settings.usable_mask_policy`,
  `usable_masks.policy`, new `usable_masks.origin` and `visibility.mask_policy` from that one string;
  `usable_pixels` is populated from the map in both modes (previously `{}` for unfrozen runs).

**Reconstruction invariance.** With no frozen paths, `raycast._pixel_authority` (`:474-475`) previously
derived `~create_gui_mask(h, w)` itself; the script now passes exactly those arrays, and `_pixel_authority`
`np.asarray(supplied, dtype=bool)` (`:481`) keeps them by value/identity. Unfrozen reconstruction evidence
is therefore unchanged; only the evaluator denominator and report labels change, matching DS-6 §3. Frozen
`--usable-mask` runs still take the authoritative loader branch, whose policy string and counts are
byte-identical to before.

**Not changed:** `_load_usable_masks` body/docstring, `visible_cells`, `raycast`, `evaluation_mask_sensitivity.py:41`
(still the explicit all-decoded control using `None`), M5 loaders, metric formulas, CLI, thresholds.

## 2. Focused tests (`tests/test_benchmark_templates.py`)

* `test_frozen_mask_spec_supports_compact_polygon_overlays`, `test_no_mask_paths_are_not_an_authoritative_empty_mapping`
  unchanged (pin loader semantics).
* `test_run_without_mask_paths_shares_one_derived_gui_mapping[None|{}]` (replaces
  `test_run_without_mask_paths_forwards_none`): both consumers receive the identical mapping object and,
  per frame, the identical array equal to `~create_gui_mask(8, 10)` (also asserted non-trivial, i.e. not
  all-true).
* `test_derived_run_reports_provisional_gui_policy_and_counts`: `origin == "derived"`, policy contains
  "derived production GUI mask" and "provisional", excludes "frozen"/"all decoded", `usable_pixels` equals
  the derived `~create_gui_mask(8, 10)` sum and is non-empty, and all three policy fields agree.
* `test_run_with_mask_paths_forwards_one_frozen_mapping`: both consumers receive the same loaded mapping
  object (identity preserved), excluded pixel honored, `origin == "frozen"`, frozen policy string and
  `usable_pixels` correct in all three report fields.
* Shared helper drives `run` past both consumers into the report with bounded stubs; it does not run or
  assert a reconstruction result, as authorized.

## 3. Results

* `tests/test_benchmark_templates.py` — 6 passed.
* `tests/test_visibility.py` — 22 passed.
* `tests/test_pixel_authority.py` — 9 passed.
* Combined run — 37 passed. No full suite, no benchmark fixture run (per scope).

## 4. Limits / open items

* Derived masks depend on `GuiMaskConfig` defaults: deterministic for a tracked revision, but not a
  hashable frozen fixture — DS-6's Policy A remains the long-term gate once frozen masks exist per
  template/frame.
* Old unfrozen **visible** numbers are not comparable to newly derived-policy runs (denominator changed
  from all decoded pixels to the production GUI domain); reconstruction-side unfrozen numbers reproduce.
* M5 diagnostic runners keep their own hardcoded `"frozen supplied JSON masks"` string and are untouched.
* No reconstruction-side behavior was re-derived by an actual output run here; the invariance argument is
  by the shared-array identity path in `_pixel_authority`, with the contract pinned by
  `tests/test_pixel_authority.py`.
