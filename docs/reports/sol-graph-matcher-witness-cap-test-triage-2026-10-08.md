# Matcher witness cap assertion triage

The failing assertion used the right signature's unpadded 80×80 grid as the planned warp entry count. `match_plane_pair` constructs a common canvas padded by `max_shift_cells=32` on every side, and `warp_signature` budgets that target canvas. Its planned count is therefore `(80 + 64) × (80 + 64) = 20,736`. The existing pair-witness budget test already expects that expression. The source refusal stage and counters are consistent; only the envelope test assertion was stale.

The corrected assertion checks the complete `(stage, limit, planned, actual)` tuple and retains the returned-match equality check. The named test passed 1/1; the full envelope and pair-witness modules passed 13/13. No production source, cap, or image-run change occurred.
