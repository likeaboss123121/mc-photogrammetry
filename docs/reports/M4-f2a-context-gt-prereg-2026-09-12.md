# M4 F2A context-ground-truth preregistration

This note freezes the evaluator-only contract and test plan before implementation. Final report path: `docs/reports/M4-f2a-context-gt-evaluator-2026-09-12.md`.

## Context artifact schema v1

The evaluator accepts a separate optional context JSON with exactly these top-level keys: `schema_version`, `artifact_type`, `scene_id`, `world_identity`, `capture_metadata_sha256`, `target_ground_truth_sha256`, `ordered_images`, `target_region`, `region`, and `blocks`. `schema_version` is integer 1 and `artifact_type` is `mc-photogrammetry-context-ground-truth`. Each ordered image has exactly `basename`, `width`, `height`, and `sha256`; order and values must match the freeze-verified image sequence. Both regions use exact `{min,max}` integer triplets. Block rows use exact `{pos,name,properties}` fields and the existing ground-truth record constraints.

The context scene/world IDs must match both capture and target GT. Its capture and target-GT digests bind exact bytes. Its target region must equal the target GT region; its own inclusive region must contain the target region and pass the existing `MAX_REGION_VOLUME`/edge caps. Every target record must be present with identical exact state in context. The context may extend beyond `capture.region`; it does not change the capture or target files. Unknown keys, malformed types, duplicate/out-of-region records, bad hashes, mismatched ordered frames/dimensions, or incomplete target-state inclusion fail closed.

## Scoring semantics

With context supplied, render the context block map once per image using the same capture camera, palette/shape/alpha/depth renderer, screenshot dimensions, and frozen default GUI mask. For usable pixels, context ownership `>= 0` defines known pixels; context ownership `< 0` defines unknown pixels. GUI-masked pixels are excluded separately. The primary full-cube plane masks are positives. All other context-owned pixels, including partial/non-full-cube owners, are known negatives. Intersect both prediction and truth scoring with the known-pixel mask so unknown pixels cannot become false positives or positives. Report known, unknown, GUI-excluded, full-cube-positive, and known-negative pixel counts per frame and in aggregate. Keep the all-face comparison provisional and on the same known context domain.

Without context, keep current target-bounded E1/E2 scoring behavior and thresholds; mark it explicitly as bounded/provisional. Context output reports target and context provenance. Promotion remains false: a finite context render does not prove boundary or exterior-occluder completeness.

## Planned changes and tests

Change only the evaluator module, its explicit evaluator wrapper for an optional context path, focused evaluator tests, this preregistration/final report, and a concise comms entry. Tests cover strict schema and region containment/volume; capture/target hash, image order/hash/dimension, scene/world mismatch; target-record preservation; no-owner symmetric exclusion; partial-owner known negatives; deterministic output; context provenance; and AST separation from production reconstruction imports. Synthetic tests only. No real labels, C4/threshold changes, R2 scoring, capture, or Minecraft actions are planned.
