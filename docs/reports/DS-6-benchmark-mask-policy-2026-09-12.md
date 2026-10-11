# DS-6 — benchmark no-frozen-mask policy decision audit — 2026-09-12

Read-only audit; only this report was written and no code, test, metric, threshold, annotation or frozen artifact changed.

## 1. Traced current behaviour

**Loader — `scripts/benchmark_templates._load_usable_masks` (`:291-355`).**
`if not paths: return None` (`:306-307`); non-empty `paths` must cover exactly
the frame set or `ValueError` (`:308-316`); each mask is a shape-checked bool
array from a white=usable PNG or tracked JSON rect/polygon spec (`:326-353`).
`{}` and `None` both collapse to `None`, so "no frozen mask" is one sentinel.

**`run` — one load, two consumers (`:86`, `:130-135`, `:145-151`).** The loaded
mapping (or `None`) is passed unconditionally to `reconstruct_scene_surfaces(...,
usable_masks=...)` and to `visible_cells(..., usable_masks=...)`.

**Reconstruction — `raycast._pixel_authority` (`:452-486`), normalised once at
`:606`.** Supplied mapping is authoritative: missing frame → all-false, wrong
shape → `ValueError` (`:476-483`). `None` derives `usable[f] =
~create_gui_mask(h, w)` (`:474-475`). The map feeds sampling as its inverse
exclusion mask (`:674` occupancy, `:724-726` face sampling), the fallback
full-patch check (`:797-812`), and residual proposals / partial fitting
(`:1292-1297`).

**Evaluator — `export.visibility.visible_cells` (`:225-341`).** `usable_mask =
None if usable_masks is None else usable_masks.get(frame_id)` (`:263`), shape
checked (`:264-269`), ANDed onto `owned` after ownership render (`:276-277` model
path, `:333-334` analytic path). `None` means **no mask at all**. With no paths
the report says `usable_mask_policy = "all decoded pixels (provisional)"`
(`:183`), `usable_masks.policy = "all decoded image pixels; no frozen mask was
supplied"` (`:197`), `visibility.mask_policy` likewise (`:213`), `usable_pixels`
= `{}`.

**The mismatch (confirmed).** For an unfrozen run reconstruction scores
`~create_gui_mask` pixels while the evaluator denominator counts every rendered
pixel including HUD; the report's "all decoded pixels" describes only the
evaluator side (DS-4F M1 backlog:
`DS-4-pixel-authority-2026-09-12.md:134-138`,
`DS-2-render-sampling-audit-2026-09-12.md:113-123,218-221`).

**Normal-mode scope.** The CLI passes no `usable_masks`
(`src/mc_photogrammetry/cli.py:382-397`), so production already runs on the
derived GUI domain; the benchmark is the only caller that can supply frozen
masks. M5 diagnostic runners always supply masks
(`scripts/m5_matched_t2_diagnostic.py:128-131,168,284`), a separate path.

**Pinned tests.** `tests/test_benchmark_templates.py:30-38` (`{}`→`None`),
`:45-95` (forwards `None` without paths, mapping with paths);
`tests/test_pixel_authority.py:154-181` (`{}`≠`None`), `:184-198` (shape
rejection), `:201-219` (`None` derives GUI); `tests/test_visibility.py:86-102`
(mask restricts denominator, shape checked);
`scripts/evaluation_mask_sensitivity.py:41` uses `{}`→`None` as the explicit
all-decoded control.

## 2. The three candidate policies

**A — reject a run with no frozen masks.** `run` raises before `load_image`
(guard at `:86-87`); only the frozen branch survives.

**B — all-true authority.** `run` builds
`{name: np.ones(image.shape[:2], bool)}` and passes it to both consumers; the
GUI heuristic stops gating reconstruction.

**C — derived production GUI authority shared with the evaluator.** `run`
derives once, when no paths exist, `usable = {name: ~create_gui_mask(h, w)}`,
passes that same mapping to `reconstruct_scene_surfaces` **and** `visible_cells`,
and labels it derived, not frozen.

## 3. Effects

| Dimension | A reject | B all-true | C shared derived GUI |
|---|---|---|---|
| Reconstruction evidence | unchanged from today's frozen branch | adds HUD/hotbar/crosshair/held-item pixels (`gui_mask.py:30-52`); fallback check (`raycast.py:797-812`) admits HUD-contaminated patches | byte-identical to today's unfrozen reconstruction (`raycast.py:474-475` derives the same map) |
| Visible denominator | only frozen-masked denominators exist | all decoded pixels: matches current report strings, not current reconstruction | GUI-masked; HUD removed from evidence and denominator |
| Reproducibility | highest: every number tied to a hashable frozen fixture | independent of `create_gui_mask`, but not evaluator-frozen or hashable | deterministic per tracked revision; `_source_provenance` (`:358-383`) already hashes `src/`; mutable only if `GuiMaskConfig` defaults change |
| Historical comparability | unfrozen runs become unrunnable | label preserved, but reconstruction domain changes (GUI→all), so old unfrozen numbers do not reproduce | reconstruction reproduces today's unfrozen runs; old unfrozen **visible** metrics change with the denominator and must be re-quoted |
| Normal-mode scope | benchmark-only; CLI untouched | benchmark-only; domain diverges further from CLI production | benchmark evidence domain becomes the CLI production domain; closes the M1 mismatch for unfrozen runs |

A also needs new frozen mask fixtures for every template/frame before any
unfrozen template can be scored, and must not be enforced inside
`_load_usable_masks`, or `evaluation_mask_sensitivity.py:41`'s all-decoded
control (`{}`→`None`) and the `m5_*` loaders break: more than the smallest
migration, and it blocks runs. B is the smallest edit but manufactures evidence
from HUD pixels, contradicting the DS-4F authority invariant, and leaves
benchmark runs measuring a domain production never uses — the divergence DS-6
exists to resolve.

## 4. Recommendation — Policy C

C makes the evaluator's denominator the *same* boolean authority the
reconstructor sampled, without inventing pixels (B) or blocking runs (A);
preserves every current unfrozen reconstruction result; and changes only the
visible denominator and report label, matching what the CLI already does. A
remains the long-term gate once frozen fixtures exist; C stops the report from
lying meanwhile.

### Smallest migration (no algorithm/threshold change)

1. `scripts/benchmark_templates.py:86`: keep `_load_usable_masks`; when it
   returns `None`, derive `usable_masks = {name: ~create_gui_mask(image.shape[0],
   image.shape[1]) for name, image in frames.items()}`, and record a
   `mask_origin` flag (`frozen` / `derived`).
2. Pass that one mapping to both `reconstruct_scene_surfaces` (`:134`) and
   `visible_cells` (`:150`); reconstruction output stays identical because
   `_pixel_authority` derives the same arrays for `None`.
3. Rewrite the three policy strings (`:183`, `:197`, `:213`) to distinguish
   "frozen supplied masks" from "derived production GUI mask (provisional)", and
   populate `usable_pixels` for the derived mapping too. Never call derived frozen.
4. Do not alter `_load_usable_masks`'s `None`/`{}` semantics or
   `reconstruct_scene_surfaces`'s `None` contract (`test_pixel_authority.py:154-219`).
5. Leave `evaluation_mask_sensitivity.py:41` as the explicit all-decoded
   control: it calls `visible_cells` with `None` and must stay that way.

### Smallest tests

* **T1 (extends `test_benchmark_templates.py:45-79`).** Add a `visible_cells`
  spy; with no mask paths both consumers receive the *same* derived mapping
  (identity per frame array), equal to `~create_gui_mask(8, 10)`. Replaces
  `test_run_without_mask_paths_forwards_none` (`:82-84`).
* **T2.** With a JSON mask path, both consumers still receive the loaded mapping
  by identity and the report policy says frozen.
* **T3 (actual output, no spies).** A synthetic frame whose only contrasting
  patch lies inside a GUI region yields no reconstructed block and zero
  `visible_cells` count under the derived policy; the same frame with an explicit
  all-true mask yields both.
* **T4 (report).** Derived runs report the derived policy string and non-empty
  `usable_pixels`; no derived run is labelled "all decoded pixels".
* Keep `tests/test_benchmark_templates.py:30-38` unchanged — it pins the loader.

## 5. Frozen-mask gate unchanged

With `--usable-mask` paths, `_load_usable_masks` already returns the
authoritative mapping and `run` already forwards it to both consumers, where
`raycast._pixel_authority` (`:476-483`) and `visible_cells` (`:263-269`,
`:276-277`, `:333-334`) enforce it. Policy C touches only the `None` branch, so
the "frozen supplied masks" policy string, `usable_pixels`, M5 frozen runners
(`m5_matched_t2_diagnostic.py:284`, `m5_profile_t2_cell.py:122`) and every
frozen-gate run, metric and threshold are unchanged. No full-suite, benchmark,
commit or staging action was taken; this is a read-only decision audit.
