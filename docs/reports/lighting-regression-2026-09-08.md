# Dim-fixture material regression — 2026-09-08

## Finding

The current Template 6 pipeline preserved its 1,983 direct coordinate matches
but fell from 605 to zero direct exact-state matches. Its global-lighting gate
reported `scene_brightness=0.983`, so it treated a dim capture as fullbright and
selected dark materials for every correctly placed cell.

The gate was circular: `_scene_brightness` compared a face with the material the
classifier had already selected. With a broad Minecraft palette, dark blocks can
always ratify a dim observation. The existing `_scene_spread` detector instead
groups observed pixels by chroma and measures brightness variation inside each
group, without consulting a candidate material.

## Fixed-settings comparison

Both diagnostics use the current cache schema 15, supplied Template camera/scene
fixtures, the local 1.21.5 JAR, and `adjacency_strength=0.55`. They are
diagnostics, not image-only promotion evidence.

| Fixture | Correction | Direct exact states | Direct coordinate matches | Behaviour |
| --- | --- | ---: | ---: | --- |
| Template 2 | enabled | 655 | 780 | Spread 1.665, below 1.9 gate; no correction applied. |
| Template 6 | disabled | 0 | 1,983 | Self-labelled brightness 0.983 prevented correction. |
| Template 6 | enabled | 613 | 1,983 | Spread 2.364; correction applied. |

The Template 6 corrected run took 199.7 seconds to reconstruct and 279.4
seconds total; the earlier false fullbright run was faster because it exited the
lighting pass before source/location refinement. Template 2 took 35.0 seconds to
reconstruct and 60.7 seconds total with no correction applied.

The default is now `CORRECT_LIGHT_CAST=True`. The maintained benchmark supports
`--no-correct-light-cast` for an explicit ablation and records the setting in its
effective settings.
