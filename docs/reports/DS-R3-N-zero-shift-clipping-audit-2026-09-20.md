# DS-R3-N zero-shift clipping audit (read-only)

Date: 2026-09-20. Scope: DS-R3-N only. No source/comms edits, no shell/process automation.
Preserved: `max_shift_cells=32` (`image_derived_graph.py:257`), `integer_tolerance=0.15`,
`min_resultant=0.90`, `ambiguous_margin=0.02` (`image_derived_graph.py:257-258`).

## Exact functions

- `plane_graph.cell_grid_homography` (`plane_graph.py:369-394`) bakes `origin_uv` into `H`.
- `lattice_planes.warp_signature` (`lattice_planes.py:1790-1828`) produces `warped_coverage`.
- `plane_graph.match_plane_pair` (`plane_graph.py:489-506`): fixed `overlap = warped_coverage & right.coverage`; refusal `insufficient_overlap` at `overlap_cells < MIN_OVERLAP_CELLS=9` (`:83`, `:492`).
- `plane_graph._correlation_surface` (`:397-430`) whitens the two maps **pre-clipped to that fixed zero-shift intersection**.
- `plane_graph._peak_statistics` (`:433-448`) integer `argmax` over that surface.
- `plane_shift_domain._shift_support` (`:815-827`), `_make_domain` (`:952-999`), `build_shift_domains` (`:1035`).

## Verdict: defect confirmed (structural, not hypothetical)

1. **Hard clip.** `S(s) ⊆ I` by construction: `_shift_support(I, s)` requires `I[r,c] AND I[r+sr,c+sc]`, so no shift can obtain support outside the zero-shift intersection. `_make_domain` drops every shift with `count < 9`, so the retained domain is confined to `|sr| < rows(I)`, `|sc| < columns(I)`.
2. **Admission clip.** `match_plane_pair` refuses the whole pair on `count(I) < 9` *before* any nonzero shift is examined (`:492`), although a shifted alignment can exist with support `>= 9`. This is the Template 2/6 mechanism: a falsely refused or short domain reaches `insufficient_axis_coverage` (`plane_graph.py:901`) / `plane_integer_residual`.
3. Therefore the declared radius 32 can never be exhausted, and `admitted_top_shift_rc` is the argmax of a surface computed on `I` alone.
4. **Secondary bias.** Raw surface values are not normalized by the shifted support `count(S(s))` that is already computed (`ShiftDomainEntry.overlap_cells`), so a 9-cell shift competes against a 900-cell shift on unequal samples.

Refutation of the benign reading: the fixed intersection is deliberate (`M4-P1-D-complete-shift-domain-contract-2026-09-13.md:33-43`) and `(0,0)` is required in every domain (`plane_shift_domain.py:312`) — which is exactly why the domain inherits the zero-shift footprint.

## Smallest repair / capture condition

Smallest admission-preserving repair: replace refusal-on-fixed-`I` with a shifted-support gate — keep the analytic maximal intersection boundary as the full search square, and refuse only when `max_s count(S(s)) < MIN_OVERLAP_CELLS`. Leave `max_shift_cells`, all three gates, `MIN_OVERLAP_CELLS`, the `_correlation_surface` primitive and its digest untouched (P1-D never ranks by surface value).

Cheapest capture-only option (diagnostic-first, no behavior change): in `_peak_statistics`/`_make_domain`, additionally retain the argmax under support-normalized values and whether zero-shift `count(I)` was binding. This is additive evidence, not a selector.

## Focused tests

- Zero-shift `count(I) = 8` with a shifted `count(S(s)) = 30` must not refuse and must retain `s`.
- `max_s count(S(s))` boundary: support extent smaller than `2*32+1` proves the domain clip; exact equality case retained.
- Normalized-vs-raw argmax disagreement on unequal supports.
- Regression: `max_shift_cells=32` and `.90/.15/.02` unchanged; existing hashes byte-identical.
- Existing `tests/test_plane_shift_domain.py:296-309` already pins the support floor and `(0,0)` presence.
