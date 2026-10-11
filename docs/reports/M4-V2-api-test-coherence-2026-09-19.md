# M4 V2 face placement: API/test coherence recovery (2026-09-19)

Task: `m4-v2-api-test-coherence-2026-09-19` (follow-up to the interrupted repair)
Status: **implemented locally; pending Astra acceptance; production promotion remains blocked**.

The interrupted repair left `plane_placement.py` on the new single-universe `PlacementUniverse` API while the focused tests were stale. The current worktree now has a coherent test/API pair. The test fixture builds one phase-0 universe containing 15 readings across five frames and all three normal axes. It exercises one connected, cycle-validated C2-R-B component, endpoint uniqueness, all-axis determination, support replay, tangent arithmetic, resealed contradiction rejection, truthful atomic budgets, permutation invariance, and module-boundary isolation.

The suite intentionally does **not** claim a replay-valid signed/nonzero V2-B fixture. Its module documentation records that limitation; signed arithmetic is covered only by independent formula oracles. No signed/nonzero production placement evidence is asserted.

## Verification

- `python -m py_compile src/mc_photogrammetry/reconstruction/plane_placement.py`: pass.
- `pytest tests/test_plane_placement.py -o addopts= -q`: **298 passed**.
- Adjacent phase-anchor, relative-translation, region-support, and endpoint-provenance tests: **253 passed**.
- No `_scratch*` probe files remain in `tests/`.
- Scope: only the V2 placement source/test/report were considered; no volume, CLI, persistence, exporter, or litematic wiring was changed by this recovery.

Current SHA-256:

- `src/mc_photogrammetry/reconstruction/plane_placement.py`: `8C8FD20BD707A535F0B02056ACF922EE264D1AD3ACF2B723F7728A46455E49EA`
- `tests/test_plane_placement.py`: `BE011923F4896B61826EA8091E075A9BC888C5A437E6744FBEE84B2E4BE2CB5D`

These checks establish internal coherence only. Astra must still inspect the six prior findings, especially support provenance binding, exact tangent placement, budget accounting, resealed-cell rejection, and reachability of signed/nonzero translations. Do not route V3 or production integration until that review accepts the branch or amends the contract.
