# M4 V2-B P0 split-phase producer — implementation report

Date: 2026-09-14

Status: **implementation complete — revision 2 remediation after a Luna REJECT;
stopped for lead/Luna review.** P0 only. No V2-B module, no calibration
promotion, no placement/wiring, no runner/CLI/schema/renderer/evaluator/export
change, and no git mutation. Revision 1 was rejected by Luna for accepting a
forged `budget_exhausted` result; the reproduced counterexample, the constructor
repair, the resealed regressions, and the `complete`/`refused` audit are recorded
in §7.

Assignment: `AGENT_COMMS.md` 2026-09-14 04:59:36 -05:00 ("P0 thirteenth contract
ACCEPTED; preflight-only implementation resumes"). Accepted contracts:

| Contract | SHA-256 | Lines |
| --- | --- | --- |
| `docs/reports/M4-V2B-P0-split-phase-producer-contract-2026-09-14.md` | `F7FA9295196EB02B284416934C50C4B895438D8EADB0BDB5353863E5E8F79172` | 171 |
| `docs/reports/M4-V2B-physical-phase-anchor-contract-2026-09-14.md` | `F7C336E848BD3C082C5A1BE914C91B222F1FB611B658269518497F275426787E` | 214 |

Resume point: partial source `lattice_planes.py` SHA-256
`17FDBF8847D49EB87166773C7F176F66946E89AF862299180C1F3A1E262B1414`.
Owned files: `src/mc_photogrammetry/reconstruction/lattice_planes.py`, new
`tests/test_split_phase_producer.py`, and this report. Nothing else was edited.

## 1. Defects fixed (owned scope)

1. **Coordinate-only `G`/`D`/`X` digests.** `_split_role_sha256` hashed the
   rectified patch *values* of the role's lane block. It is replaced by
   `_split_coordinate_sha256(local_axis, lane_start, lane_stop, height, width)`,
   which enumerates the full ordered 2D lane coordinate set (row-major for `u`,
   column-major for `v`) from shape and validated partition config alone and
   never reads the patch. At a fixed partition a value mutation cannot move any
   `G_a_sha256`/`D_a_sha256`/`X_a_sha256`/`lane_set.coordinate_sha256`, while a
   lane-coordinate change moves them even when the reduced axis bins do not.
2. **`payload_bytes` framing.** Planned `payload_bytes` was
   `48 + 24*(H+W)`, omitting the frozen payload version bytes. It is now
   `2 * len(SPLIT_SEAM_PROFILE_PAYLOAD_VERSION) + 48 + 24*(H+W)` = `618` for the
   `12x10` control (the contract's `2*21 + 48 + 24*22`).
3. **`source_records = L + 3`.** Each present lane set plus one generation record
   plus two seam records (previously `L + 4`).
4. **`replay_checks = L + 4`.** One result envelope plus the `source_records`
   terms (previously also `L + 4` by coincidence — now explicit and distinct).
5. **`_split_observed_counters` removed.** The synthetic `+10` replay-checks
   inflation and the whole runtime-exhaustion stage are gone:
   `_split_first_over_cap` is now evaluated once, on the exact `planned_work`,
   before any role/coordinate/evidence work. `budget_stage` is exactly
   `None`/`"planned"`; `_split_runtime_trigger` and the `"runtime"` branch are
   deleted. A `budget_exhausted` result keeps the exact over-cap `planned_work`
   with `runtime_work is None`; `complete` has `runtime_work == planned_work`;
   a `refused` result carries work only when computed, in-cap, and identical to
   the plan.
6. **Orientation-1 lane-plan construction bug.** `SplitPhaseLanePlan`
   re-derived the `G`/`D`/guard blocks and then required them to already be in
   ascending order, so every orientation-1 plan raised `axis_lane_overlap`. The
   check now sorts the three blocks by start and requires exact contiguous cover
   of `[0, N)`.
7. **Missing patch.** A `None` in-memory patch now maps to the typed
   `p0_patch_unavailable` refusal instead of a generic reconstruction refusal.
8. **Calibration sample tuples.** `SplitPhaseCalibration` wrongly required
   non-empty `calibration_sample_sha256s`/`evaluation_sample_sha256s`; the
   contract requires non-empty only for `null_control_ids`/`seam_control_ids`.

## 2. Producer / validator / record completion

- `produce_split_phase_evidence(...)` keeps the frozen signature and returns the
  typed `SplitPhaseProducerResult`, never bytes, with no IO. Precedence is
  domain/type validation, built-in-config check, deterministic lane plan, exact
  planned-count preflight, then roles/coordinates, counts/digests, `G`-only
  phase evidence, `D`-only embedded payloads, and a full replay self-check.
- `validate_split_phase_evidence(record, P, *, geometry, partition_config,
  budget) -> bool` returns `True` only for a `complete` record whose rebuilt
  canonical bytes match and `False` (no raise) for every missing/mismatched/
  non-complete case. `p0_patch_unavailable`/patch mismatch resolve to `False`.
- `SplitPhaseProducerResult` branch validation now enforces the exact lane-set
  role pattern (four with `g = 0`, six with `g > 0`), evidence element types,
  `lane_set.coordinate_sha256` cross-links to the result role digests, the
  generation/seam `partition_sha256`/`D_a_sha256` cross-links, `budget_stage`
  exactly `None`/`"planned"` (so `"runtime"` raises `budget_stage_invalid`), and
  `runtime_work == planned_work` on `complete`/`refused`.

## 3. Tests (`tests/test_split_phase_producer.py`, new)

20 isolated synthetic tests, no image/manifest/label/evaluator/graph/runner and
no `plane_residue` symbol (23 after the revision-2 regressions of §7, which add
the resealed `budget_exhausted` constructor controls):

- complete branch, exact aggregate counters (`payload_bytes`, `source_records`,
  `replay_checks`, `d_profile_terms`, `profile_terms`), `to_record()` field
  order, non-self-referential `result_sha256`, frozen payload framing;
- validator accept + mismatched patch/shape/missing `P`/wrong geometry/
  non-complete rejection;
- planned-only budget staging for every exhaustible cap at exact-limit
  (success), one-over (success), and one-below (atomic `budget_exhausted`,
  `budget_stage="planned"`, `runtime_work=None`, exact over-cap plan, no
  partial evidence), plus inline/digest transport switch, `work_over_cap`,
  `budget_stage_invalid`, `p0_patch_unavailable`, and
  `partition_config_not_builtin`;
- coordinate-only role digests are value-blind but lane-coordinate sensitive;
- the required **per-axis `D_a -> G_a`** controls (`D_u` only leaves the axis-`u`
  G phase/profile/digest fixed and lets axis `v` move; `D_v` only symmetric) and
  the simultaneous **`D_u ∩ D_v`** control (both G components and the whole
  `generation_evidence_sha256` fixed; both `D` value digests, `patch_sha256`,
  `producer_input_sha256`, `result_sha256` changed; every coordinate digest and
  `partition_sha256` fixed);
- selector freeze (both orientations, six-lane-set guard branch, min-lane shape
  (ii), domain errors), canonical dyadic zero, textureless typed abstention, and
  an AST isolation gate proving no `plane_residue` import/reference.

## 4. Commands and results

| Command | Result |
| --- | --- |
| `python -m py_compile src\mc_photogrammetry\reconstruction\lattice_planes.py tests\test_split_phase_producer.py` | clean |
| `python -m pytest tests\test_split_phase_producer.py -q -p no:cacheprovider -o addopts=` | **23 passed** in 0.31 s |
| `python -m pytest tests\test_split_phase_producer.py tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_lattice.py tests\test_supported.py -q -p no:cacheprovider -o addopts=` | **197 passed** in 56.84 s |
| `python -m pytest tests\test_frozen_image_runner.py tests\test_plane_residue.py tests\test_region_growth.py tests\test_joint_plane_selector.py tests\test_plane_axis_support.py -q -p no:cacheprovider -o addopts=` | **179 passed** in 40.85 s |

Interpreter: `.venv\Scripts\python.exe` (CPython 3.10.11).

## 5. Frozen hashes after this task

| Path | SHA-256 | Lines |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/lattice_planes.py` | `82ADA0CCC0F92B580342AA35F8333D1013910F7CFA91EADA0ED7D5E7C95831E7` | 3807 |
| `tests/test_split_phase_producer.py` | `E5E17B9CCC8203394EBE1F9F78601176F33C041F34620F214D1B02457B7A9B31` | 630 |
| `docs/reports/M4-V2B-P0-split-phase-producer-implementation-2026-09-14.md` | this file | — |

Revision-2 hashes; the rejected revision-1 bytes were
`62F55BCCAF7C4B6BAAA3191EE6CF4EC5364A03DF0DDE16B015264F19976CDEAF` (3778
lines) and `3E59E3BE6AFC282FA208C14AE808333A96D544906FCE18BF141A64713C5E2063`
(536 lines), retained here only as the remediation chain record. The two frozen
contracts are unchanged: P0
`F7FA9295196EB02B284416934C50C4B895438D8EADB0BDB5353863E5E8F79172` and V2-B
`F7C336E848BD3C082C5A1BE914C91B222F1FB611B658269518497F275426787E`.

Built-in partition config digest (computed, never caller-supplied):
`SPLIT_PHASE_PARTITION_CONFIG_SHA256 =
b2225d8f2ca16bf01189826870511e025e80c108ac3c1122c57ed4f7ba6b214f`.

## 6. Limitations and stop boundary

- The production producer accepts only the literal built-in
  `SPLIT_PHASE_PARTITION_CONFIG` (`g = 0`, four lane sets); the six-lane-set
  guard branch is exercised only through the pure
  `lane_role`/`split_phase_lane_plan` functions and the lane-set builder, as the
  contract requires. Alternative configs never reach the producer or validator.
- `max_inline_patch_samples` is a transport selector: when `H*W` exceeds it the
  patch is referenced by digest and `inline_patch_samples` is `0`, so the
  smallest legal cap still succeeds rather than exhausting.
- Coordinate digests enumerate the full ordered 2D lane coordinate set in
  canonical JSON; cost is bounded by the caller's `max_patch_samples` cap, and
  P0 is not wired into any production path.
- `plane_residue`, `supported`, `plane_graph`, `plane_axis_support`, `lattice`,
  the legacy full-patch APIs (`_periodic_energy_measurements`,
  `_periodic_energy_phase_evidence`, `_phase_pixels`), and every other module
  are untouched. No `plane_residue` import exists.
- No V2-B implementation, no `SplitPhaseCalibration` fitting/promotion, no
  placement, no runner/CLI/schema/renderer/evaluator/export wiring, no capture/
  label/reference access, no launcher/client automation, no git mutation, and no
  security/system change. Defender-safe prohibitions unchanged.

## 7. Luna REJECT remediation (revision 2)

Luna rejected revision 1 for a constructor-side forgery in
`SplitPhaseProducerResult`: a resealed `budget_exhausted` envelope with
`runtime_work = None`, `budget_stage = "planned"`, `producer_input_sha256` set,
an **under-cap** `planned_work` (all eleven counters within the caller budget),
and the refusal detail `{"nonsense": true}` constructed without `ValueError`.
Reproduced on `.venv` with the frozen `12x10` control and `1_000_000` caps
(`planned_work` in cap, status accepted, detail retained verbatim), confirming
the branch shape check alone was insufficient.

Root cause: `_validate_branch`'s `budget_exhausted` arm validated only `status`,
the single refusal `reason`, non-`None` provenance, `budget_stage = "planned"`,
`runtime_work is None`, and the absent evidence/geometry/patch/coordinate fields.
It never bound the refusal detail to the frozen four-key shape or to
`planned_work`, so any detail and any (including under-cap) plan was accepted.

Repair (owned source only, no schema, field, or name change): the result
constructor now enforces the §6/§105 invariant exactly. On `budget_exhausted` the
sole refusal detail must be a `dict` with exactly
`{"budget_stage", "cap", "limit", "observed"}`; `budget_stage` must be exactly
`"planned"`; `cap` must be one of the eleven frozen cap field names and therefore
map one-to-one via `_SPLIT_CAP_TO_WORK` to a `planned_work` counter; `limit` must
be an exact (non-`bool`) `int >= 1`, the budget legal domain; `observed` must be
an exact (non-`bool`) `int` equal to that counter; and `observed > limit` must
hold. Any violation raises `ValueError("split_not_reconstructible")`. The
`complete`/`refused` branch shapes, `runtime_work == planned_work`,
`budget_stage` domain, evidence/coordinate absences, and every record field and
name are unchanged.

Regressions (§3, all resealed through `_make_result`, which recomputes
`result_sha256`, so a rejection proves the branch/detail invariant rather than a
stale digest): `test_forged_budget_exhausted_detail_shapes_are_rejected` covers
arbitrary, missing, and extra keys, wrong/unknown/unhashable/non-`str` `cap`,
wrong `limit` type, `bool` `limit`, illegal zero `limit`, wrong and `bool`
`observed`, `observed == limit`, `observed < limit`, a zero
`guard_samples_per_axis` counter for which no over-cap detail is representable,
and wrong detail `budget_stage`;
`test_luna_under_cap_forgery_is_rejected` reseals the exact under-cap
`{"nonsense": true}` counterexample plus `runtime_work`/`budget_stage` tampering;
`test_all_genuine_producer_exhaustions_reseal_and_are_accepted` reseals each
genuine producer over-cap refusal (all nine representable caps, excluding the
`max_inline_patch_samples` transport selector and the zero guard counter) and
requires byte-identical `to_record()`.

Audit of the analogous `complete`/`refused` invariants, against the frozen
contract only: `complete` already requires every mandatory field, the exact
four/six lane-set role pattern, generation/seam cross-links, `refusals == ()`,
and `runtime_work == planned_work`; moreover every `complete` envelope is gated
by the mandatory byte-identical `validate_split_phase_evidence` replay, which
recomputes the selector, coordinates, counts, digests, `G` phase evidence, and
the embedded `D` payload from the supplied patch and budget, thereby enforcing
`planned <= cap`. A `refused` envelope carries no evidence, geometry, patch,
lane, or coordinate field, and its `planned_work == runtime_work` is enforced.
For both branches the in-cap requirement is not decidable from a result alone
because the budget object is not a result field; enforcing it in the constructor
would require a schema/signature argument, which the frozen contract forbids. No
additional `complete`/`refused` repair is therefore decisively required within
the owned scope, and none was made.

Commands: `.venv\Scripts\python.exe` (CPython 3.10.11), `py_compile` clean;
isolated `23 passed` in 0.31 s; focused regressions `197 passed` in 56.84 s and
`179 passed` in 40.85 s (§4). Only the three owned paths were edited; the two
frozen contract hashes are unchanged.

Stop for lead/Luna review.
