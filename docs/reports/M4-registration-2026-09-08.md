# M4 — cross-view translation registration, Template 2

**Claude Opus 5, 2026-09-08.** Follows
[M4-supported-plane-2026-09-08.md](M4-supported-plane-2026-09-08.md), which
established that a stated length places a plane to 0.1-0.4% from one image
alone. This is the next bullet of M4: "jointly fit translations ... using
... plane positions", built and validated on the same fixture.

Code: [`registration.py`](../../src/mc_photogrammetry/reconstruction/registration.py),
[`register_planes.py`](../../scripts/register_planes.py), 9 tests in
[`test_registration.py`](../../tests/test_registration.py). Full suite: 387
tests green.

## The result

All three Template 2 cameras' positions, recovered relative to one anchor view,
using nothing but each view's own Manhattan-world calibration, a stated length
per plane, a stated fact that two views see the same plane, and a stated yaw
quadrant. The manifest is read only in the right-hand column, after every
number on the left was already fixed:

| view | X (recovered) | X (manifest) | Y (recovered) | Y (manifest) | Z (recovered) | Z (manifest) |
|---|---|---|---|---|---|---|
| 13.22.14 (anchor) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 13.24.24 | 1.697 | 1.700 | −0.001 | 0.000 | — | 11.300 |
| 13.24.33 | 4.702 | 4.700 | 7.800 | 7.800 | 1.996 | 2.000 |

Every recovered coordinate is within 0.004 blocks of the manifest. The one
missing cell is honest, not a bug: 13.24.24 never asserted a plane on axis 2
at the requested side (it looks at the opposite wall), so it stays unlinked
on that axis rather than being guessed.

Reproduce:

```
python scripts/register_planes.py "templates 2" \
    --calibration docs/reports/M3-calibration-templates-2.json \
    --axis 1 --side -1 --simulate-annotations \
    --link 2026-09-02_13.22.14.png,2026-09-02_13.24.24.png \
    --link 2026-09-02_13.24.24.png,2026-09-02_13.24.33.png \
    --link 2026-09-02_13.22.14.png,2026-09-02_13.24.33.png \
    --anchor 2026-09-02_13.22.14.png \
    --out docs/reports/M4-registration-templates-2.json
```
(and equivalently for `--axis 0 --side 1` and `--axis 2 --side 1`, with
`--simulate-quadrant` added since the horizontal axes need a stated quarter-turn
per view - see below.)

## Why this needed a third kind of stated input

A single image fixes a plane's shape, never its position in a shared frame. Two
images only become comparable once something says they show the *same* surface.
That something cannot be appearance - `supported.py` measured the obvious
appearance cue (phase agreement) at chance across four decades of scale, so
building automatic cross-view correspondence on the same kind of signal would
fail for the same reason repeated textures always defeat it.

So plane identity across views is a **stated fact**, exactly like the stated
length that already replaced automatic scale: cheap to give (a person looks at
two photographs and says "that's the same floor"), immune to repetition, and
needing no feature matching. `SharedPlaneLink` is the whole mechanism:
`register_axis` chains these links through a graph, fixes one view as an anchor
at zero, and solves everyone else's position along that axis by simple
differencing - `eye_a + side_a*distance_a == eye_b + side_b*distance_b`, because
both name the same real plane coordinate.

The two horizontal axes need one more fact the vertical axis does not.
`axis_directions` only recovers yaw modulo 90 degrees from a single view - one
view cannot tell its "axis 0" from a wall rotated a quarter-turn from it. So a
second stated fact, `--quadrant`, supplies the missing quarter-turn per view:
also something a person reads off two photographs at a glance ("these are
looking roughly the same way" or "rotate this one a quarter-turn"), and also
immune to the repetition that defeats appearance matching.

Both `--simulate-annotations` (from the earlier report) and the new
`--simulate-quadrant` are labelled scaffolding in the artifact: they stand in
for what a person would state, by reading it off the reference and manifest
respectively. Nothing downstream of the stated points, lengths, links or
quadrants reads either file again until the final scoring column.

## A documented, proven limitation

With one observation per (view, axis), a wrong `SharedPlaneLink` cannot be
caught by this module. The proof is in `registration.py` and exercised by
`test_a_link_asserting_two_unrelated_planes_are_the_same_is_not_caught`: a
link's equation is a plain difference of two fixed per-view numbers, and any
cycle those differences form sums to zero identically for *any* values -
including values from a link that is flatly false. A bad assertion produces a
registration that is internally self-consistent and still wrong, the same way
`scale_from_known_length` states plainly that it cannot check whether two
points genuinely lie on one plane. Catching a wrong link needs evidence from
outside the link graph - a third, independent measurement, or (in this report)
the reference used only to score.

`register_axis` also refuses, rather than silently keeps-the-last, when a view
supplies two observations on one axis - the dict-keyed-by-view shortcut would
otherwise drop one measurement without saying so.

## What this does and does not establish

**Does:** camera translation, recovered without ever reading the manifest
during estimation, to millimetre-scale (0.004 block) agreement with it, from
three ordinary screenshots and the kind of facts a person can state by looking
at them.

**Does not:** say which *cells* of a registered plane are occupied. Position and
occupancy are different questions - this report answers the first. The
per-cell cue remains the open problem from the previous report: phase agreement
scored at chance regardless of scale, and no replacement has been measured yet.
A registered plane is therefore not yet an export-ready surface; treating its
full rectangular extent as occupied would be exactly the "rectangular extent is
not proof of occupancy" mistake M4 exists to avoid.

## Honest gaps

- **Automatic quadrant resolution is not built.** `--simulate-quadrant` reads
  the manifest as a stand-in for a person's judgement; nothing here derives the
  quarter-turn from the images.
- **No least-squares reconciliation of over-determined systems.** A connected
  component with more links than views uses a spanning tree and reports the
  rest as redundant; it does not currently average conflicting redundant links
  into a refined estimate, because - see the limitation above - two views with
  one observation each cannot conflict in a way this module can detect.
- **Joint refinement across views is not built.** Each plane's distance still
  comes from one view alone; nothing here yet combines multiple views' looks at
  the same plane into a single, better estimate.
