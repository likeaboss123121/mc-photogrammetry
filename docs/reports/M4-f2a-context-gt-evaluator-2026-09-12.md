# M4 F2A: evaluation-only context ground truth

Implemented the preregistered optional context-ground-truth contract in the frozen-plane evaluator. The context artifact is separate from target `ground_truth.json` and is accepted only through the explicit evaluator API/`--context-ground-truth` option. No capture, Minecraft action, real context artifact, R2 scoring, or evaluator-label run occurred.

## Contract and scoring

Schema v1 has exact top-level keys for artifact type/version, scene/world identity, capture metadata SHA-256, target-GT SHA-256, ordered image `{basename,width,height,sha256}` rows, target region, context region, and block records. The evaluator first verifies the label-free freeze and image bytes, then loads target GT and the optional context artifact. It rejects unknown/missing keys and malformed types, mismatched scene/world/capture/target/image bindings, reordered or changed image hashes/dimensions, target-region mismatch, context bounds that fail to contain the target, over-cap regions/block inventories, duplicate/out-of-region blocks, or any target block whose exact state is absent from context. The context region can extend beyond `capture.region`; it is not required to equal it.

Context pixels are rendered through the existing `attribution.render_result` path with capture cameras, exact block shapes/models/alpha, depth ownership, original screenshot dimensions, and the freeze's default GUI mask. For primary full-cube-only scoring, usable pixels with a nearest context owner are known. Full-cube face-owner pixels form positive planes; all other context-owned pixels, including partial/non-full-cube owners, are known negatives. Usable no-owner pixels are unknown and excluded from both proposal and truth scoring; GUI-masked pixels are counted separately. Each frame and the aggregate expose image, usable, known-owned, unknown, excluded-unknown, excluded-GUI, full-cube-positive, and other-known-negative counts. The provisional all-face comparison uses the same known domain.

With no context artifact, the existing target-bounded scoring path and masks are preserved and labeled `target_bounded_legacy_provisional`. Output records target and context provenance independently. Promotion remains false in both modes: finite context does not prove boundary or exterior-occluder completeness.

## Validation

Synthetic tests cover equal and expanded context regions, containment and volume limits, strict keys/types, target-state preservation, capture/target/image hash bindings, ordered images, dimensions, scene/world identity, symmetric unknown exclusion, partial-owner known negatives, deterministic context artifact IDs, provenance output, and production-import separation. Existing camera-space and partial-model/alpha renderer controls remain in the same focused suite.

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_frozen_plane_evaluation.py tests\test_capture_dataset.py -o addopts= -p no:cacheprovider -q
```

Result: **121 passed, 1 skipped** (existing platform-specific skip), in 21.90 seconds. `py_compile` passed for the evaluator and explicit wrapper.

## Files and integrity

- Evaluator: `src/mc_photogrammetry/evaluation_frozen_planes.py`, SHA-256 `4f233a3ff4d4cd5cbe289201b9cfcc6e147a9da0e2418ec5ca13740a23bdc4ed`.
- Explicit wrapper: `scripts/evaluate_frozen_planes.py`, SHA-256 `3c2311c655074b975f3c7c2fc06f256d451ca9d9ea692a556e7878a85b572270`.
- Focused tests: `tests/test_frozen_plane_evaluation.py`, SHA-256 `b9bfa1eb64df46a4afcaf95cf46a5a0d665f07dc0ca6a21097cfad5707f89dd`.
- Preregistration: `docs/reports/M4-f2a-context-gt-prereg-2026-09-12.md`, SHA-256 `62ff07f9482de8fff1b64be43a98e293acd1841071fa4a06e612698bcd25c044`.

R1/R2 freezes and sidecars, E1/E2 results, capture/target-GT, vanilla JAR, and palette cache retain their recorded hashes. No production reconstruction imports or threshold/configuration changes were added.
