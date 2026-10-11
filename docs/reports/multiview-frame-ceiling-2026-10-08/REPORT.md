# Multiview diagnostic frame ceiling removal

Implemented for review; production pipeline unchanged. No screenshot inference, camera metadata, truth, ground-truth schematic, or scored reconstruction was accessed.

- 3?8 frames retain v4 all-pairs generation and original RGB/descriptor/output budgets. SHA-verified prechange source reconstructed exactly: `90a92e9eb345a260a52ea0a765609159652a1019be76c525d27690c0c5254eb2`. Synthetic3/4/8 canonical outputs are byte-identical; see `legacy-byte-equivalence.json`.
- 9?32 frames use v5 and the deterministic `id-order-cyclic-distance-2-v1` graph. Each frame connects to its next two cyclic neighbors. At32 frames:64 pairs,32 three-view cliques, at most864 model combinations. Frame IDs remain opaque; ordering is a scheduling convention, not camera or overlap evidence. Unscheduled pairs have unknown support. This schedule can miss an overlap that an exhaustive search would find.
- V5 hard ceilings:32 frames,1MP/frame,96MB copied RGB,700 features/frame,64 pairs,62,720,000 bidirectional descriptor comparisons,3 models/pair,700 reciprocal matches/pair,1,000 cycle model combinations,16MB canonical output. Exceeding a ceiling refuses; evidence is not truncated.
- V5 work ledger names the schedule and reports omitted pair count (432 of496 at32 frames). Replay independently recomputes pairs, scheduled triangles, descriptor cost, omitted count, accepted model combinations, and track-to-accepted-edge evidence. An unscheduled edge cannot support a track.
- Existing sealed IMAGE-03 artifacts were untouched. v4 replay remains supported; v5 is restricted to9?32 frames.

Validation:150 focused tests pass (108 existing+42 new). New checks cover all counts9?32, deterministic32-view textured support across every frame,32 full-resolution input frames, >32 rejection, exact budget refusals, v5 persistence/source binding, resealed schedule/accounting/ledger attacks, and v4 schema preservation. Tests generate pixels; no screenshots run.

Review files: `implementation.diff`, `legacy-byte-equivalence.json`. `baseline-v4.py` is the SHA-verified prechange snapshot used for the byte comparison, preserved here for audit. Detailed history remains outside the active handoff.
