# Evaluation-mask review — 2026-09-08

The former `vanilla-hud-2560x1369-v1.json` is retained as a historical artifact,
not as the current evaluator definition. Its rectangles exclude 39.001752% of
every 2560x1369 frame. Visual inspection confirms that the old chat and held-item
rectangles cover scene geometry: Template 2 frame `2026-09-02_13.22.14.png` at
pixel `(400,1000)` is a table, not an overlay.

## Reviewed v2 policy

Each source frame has a separate schema-2 JSON in
`docs/evaluation_masks/templates-2/` or `docs/evaluation_masks/templates-6/`.
The exclusions were manually drawn after visual inspection of that frame. They
contain only the observed debug text, crosshair, hotbar, held hand, and the
screenshot-notification toast where it is present. The v2 files do not call or
derive from `scripts/overlay_mask.py`; unchanged scene pixels are never treated
as an overlay.

The held hand is represented as a polygon, so its surrounding floor and wall stay
available. The red coverage thumbnails under `docs/evaluation_masks/coverage/`
are generated from the frozen JSON by
`scripts/render_evaluation_mask_overlays.py`; they are a review aid, not mask
input.

| Fixture | Frame | v2 excluded pixels |
|---|---|---:|
| Template 2 | 13.22.14 | 6.362422% |
| Template 2 | 13.24.24 | 6.026268% |
| Template 2 | 13.24.33 | 7.694143% |
| Template 6 | 18.01.09 | 6.509770% |
| Template 6 | 18.01.14 | 8.079118% |
| Template 6 | 18.01.18 | 8.735391% |
| Template 6 | 18.01.22 | 8.735391% |

## Denominator sensitivity

This is visibility-only sensitivity, not a reconstruction quality claim. The
reference litematic, supplied cameras and scene bounds were held fixed. The
shared renderer was intentionally not supplied a palette for this probe, so its
analytic fallback makes the result cheap to replay; the next matched quality run
must use the exact palette renderer and these same v2 masks.

| Fixture | Mask policy | visible reference cells | visible reference pixels |
|---|---|---:|---:|
| Template 2 | all decoded pixels | 623 | 10,469,687 |
| Template 2 | historical v1 | 566 | 6,413,307 |
| Template 2 | reviewed v2 | 621 | 9,808,034 |
| Template 6 | all decoded pixels | 2,123 | 14,018,560 |
| Template 6 | historical v1 | 2,076 | 8,551,076 |
| Template 6 | reviewed v2 | 2,123 | 12,894,984 |

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation_mask_sensitivity.py "templates 2" `
  --reference "templates 2\Template 2.litematic" --cameras "templates 2\cameras.json" `
  --scene "templates 2\scene.json" --v1 docs\evaluation_masks\vanilla-hud-2560x1369-v1.json `
  --v2 docs\evaluation_masks\templates-2
```

The command was run from the project root for `templates 2` and `templates 6`,
with their explicit `cameras.json`, `scene.json`, and reference litematics. The
measured result shows why v1 cannot be the common denominator:
it drops 57 Template 2 visible cells and 38.7% of its analytic visible pixels.
