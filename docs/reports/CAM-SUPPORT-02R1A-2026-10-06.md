# CAM-SUPPORT-02R1A evaluation-only masked segment support

CAM_SUPPORT_EXECUTION_STATE: FAILED

Preregistration SHA-256: `cecb8a69c61c6873bf8a1380c2330d8b1eaf335a8fab68e4cf58fd5e68b0e377`.

R1A supersedes R1 before inference and freezes the same 12 R1 RGB pixels, recipes, strata, case order, runtime, detector treatment, thresholds and five acceptance gates. RGB bytes are embedded and hash-checked so the attempt never rebuilds renderer cameras before scoring.

Recipe values include camera/world parameters used only as renderer/evaluator metadata; they are present in `renderer_registry`, not absent. The image-only arm adapter accepts only a read-only RGB array and frame ID. The separate evaluator truth registry contains identity-to-recipe references only; evaluator camera objects are reconstructed after all 24 arm outputs are persisted and their seals validated. No truth is available to an arm.

The single authorized attempt ended FAILED at the aggregate JSON byte cap after 14 arm traces (seven paired cases); no evaluator truth was opened and no scoring ran. The FAILED JSON is preserved. This report records the realized failure; the preregistration paragraph above describes the intended adapter boundary and truth-order gate.
