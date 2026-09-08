# CPU PyCOLMAP relative pose — a clean negative, and why

- **Agent:** Claude Opus 5 · **Date:** 2026-09-08 · **Directive:** lead review, item 4
- **Outcome:** **feature-based two-view pose does not work on these fixtures.** 0 of 41 resolved
  pairs are usable. Stopping here as directive 6 requires, with the pairwise evidence below.
- **Environment:** Terra's isolated `.venv-sfm`, Python 3.12.10, PyCOLMAP 4.2.0, CPU.
- **Code:** [`scripts/relative_pose.py`](../../scripts/relative_pose.py),
  [`scripts/overlay_mask.py`](../../scripts/overlay_mask.py)
- **Artifacts:** `M3-relative-pose-templates-{2,6,3}.json`, one per fixture, with every pair.
- Configurations were **declared in `AGENT_COMMS.md` before any of them ran**, along with the
  expectation that this would struggle.

## Two findings, the second decisive

### 1. The F3 overlay is a watermark, and it stopped everything

Every Template 2 pair first came back as `WATERMARK` — PyCOLMAP's name for a static overlay — and
yielded no pose at all, despite 232 to 262 geometric inliers per pair. The debug overlay is drawn
identically in every screenshot of a session, so hundreds of features match at *exactly* the same
coordinates, which is indistinguishable from a watermark because it is one.

`scripts/overlay_mask.py` excludes pixels whose value is identical across every view. It finds them
from the frames rather than assuming where the overlay sits, and it removes no evidence: a pixel that
never changes between viewpoints has no parallax and cannot contribute to pose. It masks **11.8%** of
each Template 2 frame, 1.0% of Template 6, 0.5% of Template 3.

That this is needed at all is an artefact of the fixtures. Production screenshots carry no debug
overlay, so masking moves the experiment *toward* the real case.

With the overlay masked, pairs resolve. They are also wrong.

### 2. The recovered rotation is the identity

41 pairs resolved across the three fixtures. Sorted by how far apart the two views actually are:

| views apart | rotation error | views apart | rotation error |
|---|---|---|---|
| 88.5° | 88.59° | 112.3° | 112.36° |
| 90.7° | 90.68° | 123.8° | 123.80° |
| 94.1° | 93.96° | 150.8° | 150.63° |

**The error equals the separation.** That is not noise; it is the signature of a recovered rotation of
almost exactly identity — PyCOLMAP reporting that the cameras did not rotate, for pairs facing up to
150° apart.

The mechanism is Minecraft's repetition. A wall of identical blocks offers identical descriptors, so
each feature matches its look-alike at roughly the same screen position. Those correspondences are
mutually consistent, so they pass geometric verification in numbers — 100 to 300 inliers — and the
consistent solution they support is "nothing moved".

**No configuration flag catches this.** `PANORAMIC`, `PLANAR` and `DEGENERATE` exist precisely to
report an unrecoverable pose, and these pairs are none of them: they come back `CALIBRATED` or
`UNCALIBRATED` with a healthy inlier count. A caller cannot distinguish this failure from success by
anything the library returns.

## Per fixture

| fixture | views | pairs resolved | median rotation error | median direction error |
|---|---|---|---|---|
| templates 2 | 3 | 2–3 of 3 | 70.5° | 45.8° |
| templates 6 | 4 | 1 of 6 | 97.5° | 32.0° |
| templates 3 / t3-sparse-8 | 8 | 10–11 of 28 | 94.0° | 49.8° |

Across all three, **0 of 41 resolved pairs land within 5° of the true relative rotation.** The three
closest pairs available anywhere — 28.1° apart, in Template 3, with 258 to 298 inliers — are wrong by
46.9°, 48.3° and 49.8°.

`strict` (tighter ratio, cross-check) resolved the most pairs on Template 2 and was no more accurate.
`dense` found more features and no more truth. The three configurations differ in how many wrong
answers they produce, not in whether they produce right ones.

## Why the fixtures make this worse

Measured from the manifests, as evaluation only:

| fixture | closest pair | median separation | pairs under 60° |
|---|---|---|---|
| templates 2 | 66.4° | 124.3° | 0 of 3 |
| templates 6 | 73.5° | 109.2° | 0 of 6 |
| templates 3 / t3-sparse-8 | 28.1° | 96.0° | 6 of 28 |

These captures are sets of distinct viewpoints, not overlapping sweeps. Two views 176° apart share no
scene at all. That compounds the repetition problem but does not cause it — the 28°-apart pairs, which
overlap heavily, fail too.

## The smallest useful user constraint

Not more views, and not a separation limit: even the closest pairs fail, so a rule like "keep views
within 30°" would not rescue this. The failure is that **appearance cannot establish correspondence in
a world built from repeated identical blocks**, and no amount of matching tuning changes that.

What would help, in increasing order of burden on the user:

1. **A known length** — one wall stated to be *n* blocks. That fixes scale directly and is what the
   milestone already permits, and it does not require correspondence at all.
2. **One corresponded point pair** between two views — a corner the user clicks in both. A single
   unambiguous correspondence anchors what repetition destroys.
3. **A capture instruction** — overlapping sweeps rather than distinct viewpoints. This changes the
   data rather than the algorithm and is the least attractive, since the project's premise is
   reconstruction from whatever screenshots exist.

## What this means for sequencing

The lattice route does not need correspondence. It reads a plane's distance from the block grid
within a single view, which is why it measured Template 2's floors to 0.2% while this returns
identity rotations. On the evidence here, feature-based SfM is the wrong instrument for Minecraft
interiors, and the per-view lattice with a user-supplied scale is the better line.

Recommend the lead treat PyCOLMAP as **evaluated and set aside** for these fixtures rather than left
open, and that M4's surface work proceed from single-view lattice evidence plus cross-view *geometric*
consistency, rather than waiting on correspondences that repetition prevents.

## Not claimed

- Incremental mapping and bundle adjustment were not run. With no credible pairwise geometry there is
  nothing for them to start from, and directive 6 says to stop rather than continue.
- Nothing here shows PyCOLMAP is at fault. It is being asked for correspondence in a scene engineered
  to defeat it.
- Templates 1, 4 and 5 were not tested.
