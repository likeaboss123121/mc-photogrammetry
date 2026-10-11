#!/usr/bin/env python3
"""DS-IMG-SAMEAXIS-R10-COUNTERFACTUAL-REPLAY-IMPLEMENT-20261004.

Bounded, evaluation-only offline replay of the accepted R10 same-axis
pre-filter capture (`.tmp-geom-prefilter-capture-r10-20261004/observation.json`).

What this module does
---------------------
It rebuilds the *exact* 38 post-same-axis `PlaneMatch` solver inputs from the
pinned R10 artifact alone -- ``input_identities``, the 15 reading alternative
tables, the 38 uniquely matched ``trace.pairs[selected=true]`` rows and the
complete candidate rows -- then calls the *unchanged* public
``plane_graph.solve_component`` on them.  It never re-matches, re-filters,
re-signatures, re-solves upstream evidence, or reads an image.

Fail-closed reconstruction
--------------------------
Every count, identity join and digest is replayed before a single solver call:

* the record must replay through the production
  ``image_derived_graph.validate_same_axis_candidates`` validator;
* the record's ``failed_assignment_sha256`` and
  ``input_match_identity_sha256`` must equal the projected
  ``trace.solver_failure.active_edge_aliases`` binding;
* the 38 ``selected`` trace rows must be pairwise unique and their
  ``(left, left_plane, right, right_plane, quadrant)`` key set must equal the
  38 identity key set exactly;
* each identity's reading reference must exist in the 15-row table with a
  matching local axis;
* the reconstructed 38 endpoints are cross-checked field-by-field against the
  artifact's own ``solver_failure_observation.solver_input_bindings.edges``.

Plane-index remap
-----------------
The R10 reading table omits the single *unreferenced* reading
``image-0000`` plane ``3``.  A synthetic `PlaneReading` universe built from the
15 rows therefore compresses that frame's canonical indices, so an explicit
``recorded_plane -> canonical_index`` remap is computed by sorting with the
production ``lattice_planes.reading_order_key``.  The remap is asserted to be a
bijection per frame and is reported in full.  Synthetic ``period_pixels``
carries the recorded plane index purely as a deterministic tie-breaker; it is
not solver-relevant.

Solver-relevant fields (everything else is documented as inert)
--------------------------------------------------------------
`PlaneMatch`: ``left``, ``right``, ``left_plane``, ``right_plane``,
``left_axis``, ``right_axis``, ``left_b``, ``right_b``, ``quadrant_of_right``
and ``score``.  `PlaneReading`: ``frame``, ``axis``, ``side`` and
``alternatives`` (``blocks_per_period``, ``distance_blocks``, ``support``).
``psr``, ``peak``, ``second_peak``, ``overlap_cells``, ``cell_shift_rc``,
``period_pixels``, ``phase_pixels``, ``extent``, ``samples``, ``polygon``,
``coverage_cells``, ``structural_share``, ``confidence``, ``rectification``,
``region_evidence`` and ``phase_evidence`` are never read by
``solve_component``.  ``peak``/``second_peak`` are exact for the 10 captured
matches and a documented ``0.0`` placeholder for the other 28 because the trace
does not serialize them.

Control gate
------------
The directive requires an *unmodified, unpinned* solver control that reproduces
R10's returned assignment ``[0, 3, 1]``, reason ``plane_integer_residual``, the
recorded per-axis resultants/residuals/observations, and an empty accepted
match set.  The gate is checked with a stated tight tolerance.  If it fails the
harness fails closed: it records ``status="baseline_unreconstructed"``,
``control_gate.passed=false``, ``treatments.interpretable=false`` and
``treatments.arms={}``, and it does **not** call ``build_treatments`` or execute
any C1-C5 counterfactual arm (no treatment solver call at all).  Only the
passing pinned baseline and the observed unpinned control -- the two mandated
control runs -- are retained as diagnostics.  No harmful-pruning verdict is
asserted or derivable from a failed control.

R11 exact-context control (DS-IMG-SOLVER-CONTEXT-R11-REPLAY-CONTROL-20261004)
-----------------------------------------------------------------------------
The R10 capture does not serialize the quadrant constraints the producing call
actually passed, which is why its unpinned control cannot reconstruct R10.  The
accepted R11 contained run captured those exact arguments as a production
``solve_context`` record.  ``replay_r11_control`` therefore loads *two* pinned
observation files: R10 for the 38 reconstructed solver inputs and R11 for the
exact captured call.  Before a single solver call it

* reconciles the shared R10/R11 graph, trace, source and capture inputs
  field-by-field (``reconcile_shared_inputs``);
* replays the R11 ``solve_context`` through the production
  ``image_derived_graph.validate_solver_solve_context`` validator, binds its
  digest to the pinned value, and binds its ordered frame universe to the R10
  reconstruction's own solver frame universe; and
* runs exactly one unpruned, context-pinned control call carrying R11's exact
  ``quadrant_constraints`` and the three captured thresholds -- no C1-C5 or any
  other treatment arm (``treatments.arms`` is empty by construction).

The control must reproduce R10's recorded assignment ``[0, 3, 1]``, reason
``plane_integer_residual``, per-axis resultants/residuals/observations and
translations.  On any mismatch the harness fails closed: it records
``status="control_mismatch"`` with the exact mismatch list and executes no
further solver call.  Tolerances are never relaxed and no pruning loss is
inferred.

R11 preregistered C1-C5 arms
(DS-IMG-SOLVER-CONTEXT-R11-C1-C5-COUNTERFACTUAL-20261004)
---------------------------------------------------------
``replay_r11_arms`` consumes three sealed artifacts -- the R10 capture, the R11
capture and the Lead-accepted R11 baseline replay -- and reconciles all three
*before* any treatment.  The accepted baseline is never rerun: its sealed
control outcome is the reference the arms are measured against.  The replay then
executes exactly the preregistered C1-C5 one-for-one substitutions, once each,
through the *identical* captured thresholds and ordered quadrant constraints, so
every arm is a single exact-context solver call.  Each substituted row must
still be a dropped ``same_axis_mismatch`` candidate in its declared group and
must replace a survivor of that same group; otherwise nothing is solved and the
replay fails closed.  The report records the unchanged solver reason and
assignment, per-axis resultants/residuals/observations, active edge counts and
axis coverage, translations, and exact deltas against the sealed control.  The
frozen ``.90`` resultant / ``.15`` residual gates appear only as per-axis
diagnostics and are never relaxed: a gate pass is feasibility evidence for this
single fixture, never candidate truth or product accuracy, and the overall
geometry keeps its unchanged solver-reason label.  Any pin or control mismatch
stops the replay before a single arm is solved.

R11 axes-0/2 + C3 combined arms
(DS-IMG-SOLVER-CONTEXT-R11-AXES02-C3-COUNTERFACTUAL-20261004)
-------------------------------------------------------------
``replay_r11_axes02_c3`` consumes four sealed artifacts -- the R10 capture, the
R11 capture, the accepted R11 baseline control and the accepted C1-C5 arm
replay -- and *consumes* the last one only for its sealed C3 outcome.  The
sealed C1-C5 artifact, its C3 arm and the baseline control are never rebuilt or
rerun.  After every file/seal pin, the R10<->R11<->baseline reconciliation and
the sealed C3 provenance replay, all eight frozen combined substitutions are
planned and validated *before* any solver call: each candidate must still be a
dropped ``same_axis_mismatch`` row in its declared group, must replace a
survivor of that group, and must replay that survivor's exact
``(left, left_plane, right, right_plane, quadrant)`` edge identity -- otherwise
the remaining survivor's edge would be duplicated.  Both endpoints must also
map to the declared world axis and be active under the sealed C3 ``[0, 3, 1]``
assignment, and the combined active edge set must stay duplicate-free.  Only
then does the replay make exactly eight context-pinned solver calls (one per
arm) under the identical captured thresholds and ordered constraints, and
record each arm's unchanged reason, assignment, per-axis
resultants/residuals/observations, active edge counts, axis coverage,
translations and exact deltas against the sealed C3 outcome.  Any pin or
semantic mismatch stops the replay before a single arm is solved, and the
``.90``/``.15`` gates remain diagnostics only.

R11 exhaustive axes-0/2 + C3 replay
(DS-IMG-SOLVER-CONTEXT-R11-AXES02-EXHAUSTIVE-COUNTERFACTUAL-20261004)
------------------------------------------------------------------
``replay_r11_axes02_exhaustive`` consumes five sealed artifacts -- the R10
capture, the R11 capture, the accepted R11 baseline control, the accepted
C1-C5 arm replay and the accepted axes-0/2 + C3 eight-arm replay.  The accepted
baseline, C1-C5 and axes-0/2 artifacts are consumed, never rebuilt or rerun.

Before a single solver call the harness enumerates *every* captured dropped
``same_axis_mismatch`` row that maps both endpoints to world axis 0 or 2 under
the sealed ``[0, 3, 1]`` assignment, is active there, replaces a survivor of its
own group and leaves no duplicate active edge key against the remaining active
matches.  Candidate edge identity may differ from the replaced survivor's -- the
exact-identity restriction belonged only to the earlier rank-1 choices.  The
enumeration must reproduce the Lead's independently audited 201 valid row/target
pairs (164 world-axis-0 + 37 world-axis-2, groups 0/3/8 = 81/53/30 and groups
5/6 = 26/11, group 3 split 25->16 / 28->17); any other count is reported and
stops the replay before any solver call rather than being silently filtered.

Every enumerated substitution is then combined with the sealed C3 rows
(203->19/group 4 and 383->33/group 7) and solved exactly once under the identical
captured R11 thresholds/ordered constraints: 201 context-pinned calls.  All
potential cross-axis pair keys are validated solver-free before the first call
so the single final combined arm can be run only if the measured-best axis-0 and
axis-2 winners are pairable.  The winners are ranked by measured outcome --
moved-axis gate pass (resultant >= .90 and residual <= .15) first, then measured
resultant descending, measured residual ascending, captured candidate score
descending, flat row ascending and survivor index ascending -- never by
candidate score alone, and a confidence-ranked fallback pair is never
substituted for the measured winners.  If no valid measured pair exists the
replay stops at 201 calls and reports why.  Each arm records its group/row/score/
delta provenance, unchanged reason, assignment, per-axis resultants/residuals/
observations, translations, active edge count, axis coverage and exact deltas
against the sealed C3 outcome.  The ``.90``/``.15`` gates remain per-axis
diagnostics only and are never relaxed or treated as accuracy.

R11 axes-0 x axis-2 pair scan
(DS-IMG-R11-AXES02-PAIR-SCAN-20261004)
--------------------------------------
``replay_r11_axes02_pair_scan`` consumes six sealed artifacts -- the R10 capture,
the R11 capture, the accepted R11 baseline control, the accepted C1-C5 arm
replay, the accepted axes-0/2 + C3 replay and the accepted exhaustive axes-0/2
replay.  None of them is rebuilt or rerun.  After every file/seal pin, the
R10<->R11<->baseline reconciliation, the sealed C1-C5 C3 provenance replay and
the exhaustive artifact's own cross-binds, the harness solver-free re-derives
the exact 164 axis-0 x 37 axis-2 = 6,068 duplicate-free combined substitutions
(each adding its axis-0 and axis-2 one-for-one replacement to the sealed C3
rows), recomputes the sorted pair-key digest independently and requires it to
equal both the pinned value and the digest the accepted exhaustive artifact
recorded.  Every pair's combined active edge set must be duplicate-free before
the first solver call; a mismatch stops the replay with zero calls.

Exactly 6,068 context-pinned solver calls then run once each under the identical
captured R11 thresholds and ordered quadrant constraints -- no retries, no
baseline/C1-C5/C3/axes-0-2/exhaustive rerun.  Each pair records its axis-0 and
axis-2 flat row/group/target/score, the exact replaced survivors' scores and the
per-axis score deltas against them (a correlation-score proxy only), the
assignment, acceptance/reason, resultants/residuals, active count and duplicate
status.  The pair that the accepted exhaustive run froze as ``A020-308`` is
identified in-scan and its measured outcome is cross-checked against that sealed
final arm.

Reporting is exact six-dimensional per-axis dominance -- maximise both score
deltas, both resultants and minimise both residuals, equivalently maximise the
unchanged ``resultant - .90`` and ``.15 - residual`` gate margins per moved
axis.  Both the all-arm frontier and, when non-empty, the solver-accepted
frontier (the named accepted-arm frontier ``frontier_accepted_solver``) are
reported, with the gate-accepted frontier (``frontier_accepted_gate``) kept as a
separate diagnostic, plus the globally highest-correlation-score pair and its
result.  The primary result remains the accepted-pair count and fraction, where
"accepted" is the unchanged solver's own ``reason == ''`` / ``accepted true``
outcome on the whole arm.  There is no pooled score, no weighted scalar and no
selected or promoted winner; the ``.90``/``.15`` gates remain diagnostics only
and axis 1 is reported separately as the sealed constant.

No-truth disclaimer: this is a single-fixture diagnostic replay of serialized
provenance.  It is not accuracy, correspondence truth, camera or geometry
evidence, and it changes no production behaviour or decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.reconstruction import lattice_planes as lattice_planes
from mc_photogrammetry.reconstruction import plane_graph as plane_graph

SCHEMA = "mc-photogrammetry-same-axis-counterfactual-replay-v1"
INPUT_SCHEMA = "mc-photogrammetry-evaluation-graph-stage-v1"
CAPTURE_SCHEMA_V2 = "mc-photogrammetry-same-axis-candidate-capture-v2"
INPUT_FILE_SHA256 = (
    "BC957B684152AEEBB5C055EB12816643F91A506AB01543761D6461FEE4C624B4")
DEFAULT_INPUT = (
    Path(".tmp-geom-prefilter-capture-r10-20261004") / "observation.json")
DEFAULT_OUTPUT = (
    Path(".tmp-geom-counterfactual-r10-20261004") / "counterfactual_replay.json")

#: DS-IMG-SOLVER-CONTEXT-R11-REPLAY-CONTROL-20261004 pinned second input: the
#: accepted R11 contained capture that serialized the producing call's exact
#: ``solve_context``.  Both observation digests and the captured context digest
#: are pinned to the verified Lead-acceptance values in AGENT_COMMS.md.
R11_INPUT_FILE_SHA256 = (
    "924D388D5ED41287EB88A6B0B709B8228C2EAD8422F9957946EFC6A734F75A92")
R11_CONTEXT_SHA256 = (
    "7e18498e82cd76d589e5673dca75e463d4efd6e0bd39a29cc8f1a8427aa24439")
R11_CONTEXT_STATUS = "complete"
DEFAULT_INPUT_R11 = (
    Path(".tmp-geom-solver-context-r11r2-20261004") / "observation.json")
DEFAULT_OUTPUT_R11 = (
    Path(".tmp-geom-counterfactual-r11-20261004") / "counterfactual_replay.json")
R11_CONTROL_SCHEMA = (
    "mc-photogrammetry-same-axis-counterfactual-replay-r11-control-v1")

#: DS-IMG-SOLVER-CONTEXT-R11-C1-C5-COUNTERFACTUAL-20261004 pins: the accepted,
#: sealed R11 baseline replay whose control outcome the arms are measured
#: against, and the fresh single-run output root for the preregistered arms.
R11_BASELINE_FILE_SHA256 = (
    "E0E7F9C8F46FF4B97E94E5CA8F2E612C46D926D773B006AD605E5BED27A7DD5C")
R11_BASELINE_SEAL_SHA256 = (
    "fc94cd3700c0106d3769f928a3461fea5fe8716054b68abfe333f697ef0a5b4f")
R11_BASELINE_TASK_ID = "DS-IMG-SOLVER-CONTEXT-R11-REPLAY-CONTROL-20261004"
DEFAULT_INPUT_R11_BASELINE = (
    Path(".tmp-geom-counterfactual-r11-20261004") / "counterfactual_replay.json")
ARMS_SCHEMA = (
    "mc-photogrammetry-same-axis-counterfactual-replay-r11-arms-v1")
ARMS_TASK_ID = "DS-IMG-SOLVER-CONTEXT-R11-C1-C5-COUNTERFACTUAL-20261004"
DEFAULT_OUTPUT_ARMS = (
    Path(".tmp-geom-counterfactual-r11-arms-20261004")
    / "counterfactual_replay.json")

#: DS-IMG-SOLVER-CONTEXT-R11-AXES02-C3-COUNTERFACTUAL-20261004 pins: the
#: Lead-accepted sealed R11 C1-C5 arm replay (consumed only for its sealed C3
#: outcome) and the fresh single-run output root for the eight frozen
#: axes-0/2 + C3 combined substitutions.  The C1-C5 artifact (and hence its C3
#: arm) is never rebuilt or rerun.
R11_C1_C5_FILE_SHA256 = (
    "CFC12BF5A6CBD36374175F2B0A28EE254519D119A094C15A9347A2D7583E87DC")
R11_C1_C5_SEAL_SHA256 = (
    "a7a7ba4363f56dcf4e9f30c956ae20d1fecf3bb4761f629ab2387fef5fc85556")
R11_C1_C5_TASK_ID = "DS-IMG-SOLVER-CONTEXT-R11-C1-C5-COUNTERFACTUAL-20261004"
DEFAULT_INPUT_R11_C1_C5 = (
    Path(".tmp-geom-counterfactual-r11-arms-20261004")
    / "counterfactual_replay.json")
AXES02_C3_SCHEMA = (
    "mc-photogrammetry-same-axis-counterfactual-replay-r11-axes02-c3-v1")
AXES02_C3_TASK_ID = (
    "DS-IMG-SOLVER-CONTEXT-R11-AXES02-C3-COUNTERFACTUAL-20261004")
DEFAULT_OUTPUT_AXES02_C3 = (
    Path(".tmp-geom-counterfactual-r11-axes02-20261004")
    / "counterfactual_replay.json")

#: DS-IMG-SOLVER-CONTEXT-R11-AXES02-EXHAUSTIVE-COUNTERFACTUAL-20261004 pins:
#: the Lead-accepted sealed axes-0/2 + C3 eight-arm replay (cross-consumed as a
#: sealed reference alongside C1-C5) and the fresh single-run output root for the
#: exhaustive one-for-one axes-0/2 replay.  Neither the accepted baseline, the
#: C1-C5 replay nor any accepted axes-0/2 arm is rebuilt or rerun.
R11_AXES02_FILE_SHA256 = (
    "0ED1FF3328F8059FA041F0EDB3D8ABC539E67FDE630E347539F8103D95547149")
R11_AXES02_SEAL_SHA256 = (
    "661257618e6635cdead5696259896af9120f04848757374162b30130610ce249")
R11_AXES02_TASK_ID = (
    "DS-IMG-SOLVER-CONTEXT-R11-AXES02-C3-COUNTERFACTUAL-20261004")
DEFAULT_INPUT_R11_AXES02 = (
    Path(".tmp-geom-counterfactual-r11-axes02-20261004")
    / "counterfactual_replay.json")
AXES02_EXHAUSTIVE_SCHEMA = (
    "mc-photogrammetry-same-axis-counterfactual-replay-r11-axes02-exhaustive-v1")
AXES02_EXHAUSTIVE_TASK_ID = (
    "DS-IMG-SOLVER-CONTEXT-R11-AXES02-EXHAUSTIVE-COUNTERFACTUAL-20261004")
DEFAULT_OUTPUT_AXES02_EXHAUSTIVE = (
    Path(".tmp-geom-counterfactual-r11-axes02-exhaustive-20261004")
    / "counterfactual_replay.json")

#: The frozen exhaustive scope: every captured dropped ``same_axis_mismatch``
#: row that maps both endpoints to the declared world axis under the sealed
#: ``[0, 3, 1]`` assignment, is active there, replaces a survivor of its own
#: group and leaves no duplicate active edge key.  Groups 0/3/8 map to world
#: axis 0 and groups 5/6 to world axis 2; group 2 has no dropped alternative.
EXHAUSTIVE_AXIS_GROUPS: Mapping[int, int] = {0: 0, 3: 0, 8: 0, 5: 2, 6: 2}
EXHAUSTIVE_EXPECTED_PER_GROUP_AXIS: Mapping[tuple[int, int], int] = {
    (0, 0): 81, (3, 0): 53, (8, 0): 30, (5, 2): 26, (6, 2): 11}
EXHAUSTIVE_EXPECTED_PER_GROUP_TARGET: Mapping[tuple[int, int], int] = {
    (0, 4): 81, (3, 16): 25, (3, 17): 28, (5, 23): 26, (6, 26): 11,
    (8, 37): 30}
EXHAUSTIVE_EXPECTED_AXIS_COUNTS: Mapping[int, int] = {0: 164, 2: 37}
EXHAUSTIVE_EXPECTED_TOTAL = 201
EXHAUSTIVE_AUTHORIZED_SOLVER_CALLS = 202
EXHAUSTIVE_SELECTION_RULE = (
    "measured arms ranked by (1) moved-axis gate pass (resultant >= .90 and "
    "residual <= .15) first, (2) moved-axis measured resultant descending, "
    "(3) moved-axis measured residual ascending, (4) captured candidate score "
    "descending, (5) candidate flat row ascending, (6) replaced survivor index "
    "ascending; the final combined arm uses the measured-best valid axis-0 and "
    "axis-2 substitutions and runs only after the pair passes the solver-free "
    "duplicate-key check")

#: The sealed C3 substitution the combined arms are built on, exactly as frozen
#: by the accepted C1-C5 run: ``(flat_row, survivor_match, group, world_axis)``.
C3_SUBSTITUTIONS: tuple[tuple[int, int, int, int], ...] = (
    (203, 19, 4, 1), (383, 33, 7, 1))

#: The eight frozen combined arms.  Each adds exactly two one-for-one
#: axes-0/2 substitutions to C3; every entry is ``(flat_row, survivor_match,
#: group, world_axis)``.  Groups 0/3/8 map to world axis 0 and groups 5/6 to
#: world axis 2 under the sealed ``[0, 3, 1]`` assignment.  The group-3 target
#: is forced: row 151 replays survivor 16's edge identity and row 188 replays
#: survivor 17's, so the wrong target would leave a duplicate active edge.
AXES02_ARMS: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "A50-295": ((50, 4, 0, 0), (295, 23, 5, 2)),
    "A50-304": ((50, 4, 0, 0), (304, 26, 6, 2)),
    "A151-295": ((151, 16, 3, 0), (295, 23, 5, 2)),
    "A151-304": ((151, 16, 3, 0), (304, 26, 6, 2)),
    "A188-295": ((188, 17, 3, 0), (295, 23, 5, 2)),
    "A188-304": ((188, 17, 3, 0), (304, 26, 6, 2)),
    "A424-295": ((424, 37, 8, 0), (295, 23, 5, 2)),
    "A424-304": ((424, 37, 8, 0), (304, 26, 6, 2)),
}

#: DS-IMG-R11-AXES02-PAIR-SCAN-20261004 pins: the Lead-accepted sealed exhaustive
#: axes-0/2 replay is consumed as a sealed cross-reference -- its frozen A020-308
#: final arm is the already accepted pair the scan identifies in-scan, and its
#: solver-free pair digest is independently re-derived rather than trusted --
#: and the fresh single-run output root holds the 164 x 37 = 6,068 axis-0 x
#: axis-2 pairs, each combined with the sealed C3 substitution, solved once.
R11_AXES02_EXHAUSTIVE_FILE_SHA256 = (
    "2FE78B7ABF5AC25831F84624C91A31291B59EA09023DFB06BC1D314B4555B27A")
R11_AXES02_EXHAUSTIVE_SEAL_SHA256 = (
    "c274d77ecd0475809423523e7d76c17c24d2c0319e2e089e3718781ea8c9c2ac")
R11_AXES02_EXHAUSTIVE_TASK_ID = AXES02_EXHAUSTIVE_TASK_ID
DEFAULT_INPUT_R11_AXES02_EXHAUSTIVE = (
    Path(".tmp-geom-counterfactual-r11-axes02-exhaustive-20261004")
    / "counterfactual_replay.json")
PAIR_SCAN_SCHEMA = (
    "mc-photogrammetry-same-axis-counterfactual-replay-r11-axes02-pair-scan-v1")
PAIR_SCAN_TASK_ID = "DS-IMG-R11-AXES02-PAIR-SCAN-20261004"
DEFAULT_OUTPUT_PAIR_SCAN = (
    Path(".tmp-geom-r11-axes02-pair-scan-20261004")
    / "counterfactual_replay.json")
PAIR_SCAN_EXPECTED_AXIS_COUNTS: Mapping[int, int] = {0: 164, 2: 37}
PAIR_SCAN_EXPECTED_PAIRS = 6068
PAIR_SCAN_AUTHORIZED_SOLVER_CALLS = 6068
#: The already accepted measured-best combined pair, exactly as frozen by the
#: accepted exhaustive run: ``(axis0_row, axis0_target, axis2_row,
#: axis2_target)``.
PAIR_SCAN_ACCEPTED_REFERENCE_PAIR: tuple[int, int, int, int] = (20, 4, 308, 26)
PAIR_SCAN_ACCEPTED_REFERENCE_LABEL = "A020-308"
PAIR_SCAN_PAIR_KEY_SHA256 = (
    "8cdc32850ad53bc3388ba68e2e4170fe204bf7256ff6570a4b02a36b7f63fd55")
#: The preregistered single-arm ranking line from the original task text.  It is
#: recorded for provenance and explicitly superseded by the Lead clarification
#: (and by the later definitive 6-D Pareto contract below): no weighted scalar
#: "best" arm, no pooled score and no bottleneck scalar is selected or promoted.
PAIR_SCAN_SUPERSEDED_RANKING_RULE = (
    "superseded preregistered single-arm line: choose accepted arms first; rank "
    "by minimum resultant descending, maximum residual ascending, sum of "
    "candidate scores descending, then row indices ascending")
PAIR_SCAN_PRIMARY_RESULT = (
    "accepted-pair count and fraction over the exact 6,068 context-pinned solver "
    "calls")
#: The definitive 6-D Pareto contract (Lead restart note, superseding both the
#: earlier weighted/bottleneck proposal and the 2-D pooled constants that were
#: present in source).  Every objective is stated as *maximise*; ``negated_*``
#: entries are the two residual objectives held with their sign flipped so that
#: one uniform componentwise dominance test applies.  No pooled score, no
#: weighted scalar and no newly selected winner.
#:
#: DS-IMG-R11-PAIR-SCAN-ACCEPTANCE-ERRATUM (2026-10-04): the two report strings
#: below are wording corrections only.  ``frontier_accepted_solver`` is the named
#: accepted-arm frontier; ``frontier_accepted_gate`` is preserved separately as a
#: gate-accepted diagnostic.  The frozen pair-scan artifact was sealed before
#: this erratum and therefore still embeds the earlier gate-primary
#: ``accepted_definition``/``rule`` text; it is not modified or resealed, and the
#: two acceptance sets and their 4-member frontiers coincide on this fixture, so
#: every recorded count is unchanged.
PAIR_SCAN_OBJECTIVES: tuple[str, ...] = (
    "score_delta_axis0",
    "score_delta_axis2",
    "resultant0",
    "resultant2",
    "negated_residual0",
    "negated_residual2",
)
PAIR_SCAN_PARETO_RULE = (
    "exact six-dimensional per-axis dominance over all 6,068 arms: maximise "
    "axis-0 score_delta_vs_replaced_survivor, axis-2 "
    "score_delta_vs_replaced_survivor, resultant0 and resultant2; minimise "
    "residual0 and residual2.  Equivalently maximise the unchanged gate margins "
    "(resultant - .90) and (.15 - residual) on each moved axis.  Axis 1 is the "
    "sealed C3 substitution and constant, reported separately.  The all-arm "
    "frontier is reported, together with the named accepted-arm frontier "
    "``frontier_accepted_solver`` restricted to solver-accepted (reason == '') "
    "arms; the gate-accepted frontier ``frontier_accepted_gate`` is preserved "
    "separately as a diagnostic.  No weighted scalar, no pooled score and no "
    "best arm is selected or promoted")
PAIR_SCAN_AXIS_OBJECTIVES = (
    "per moved axis: maximise score_delta_vs_replaced_survivor and the "
    "unchanged gate margins (resultant - .90) and (.15 - residual)")
#: Binding acceptance definition after DS-IMG-R11-PAIR-SCAN-ACCEPTANCE-ERRATUM:
#: primary accepted = the unchanged solver's normal accepted outcome on the whole
#: arm (``reason == ""`` / ``accepted true``); the moved-axis per-axis gates are a
#: separate diagnostic so a per-axis-only pass is never called solver-accepted.
PAIR_SCAN_ACCEPTED_DEFINITION = (
    "primary accepted pair = the unchanged solver returned its normal accepted "
    "outcome on the whole arm (reason == '' / accepted true), including the "
    "sealed axis-1 C3 row.  The moved-axis unchanged per-axis gate (resultant >= "
    "min_resultant and residual <= integer_tolerance) is reported independently "
    "as a diagnostic so a per-moved-axis-only gate pass is never labelled "
    "solver-accepted; the sealed axis-1 C3 row is constant and is reported "
    "separately")

#: Absolute tolerance for the recorded preregistered axis-1 predictions.  Those
#: are hand-derived design values and are reported verbatim; a mismatch is
#: reported exactly and never rationalised away.
PREREGISTERED_ABS_TOLERANCE = 1e-6

#: The R10/R11 fields that must be byte-equal before any cross-artifact replay:
#: the graph result, the trace (and its source digests), and the whole
#: same-axis capture record.  The artifact self-digest, entry digest, telemetry
#: and the failure observation legitimately differ (R11 adds the context).
RECONCILED_KEYS = (
    "schema", "graph", "graph_sha256", "trace", "trace_sha256",
    "same_axis_candidates", "inputs", "project_sha256",
    "signature_identities", "reason", "status")

#: Absolute tolerance for the R10 control's per-axis resultants/residuals and
#: translations.  The reconstruction reproduces R10 to ~1e-15, so 1e-9 is tight.
CONTROL_ABS_TOLERANCE = 1e-9

NO_TRUTH_DISCLAIMER = (
    "Evaluation-only diagnostic replay of serialized R10 provenance. Not "
    "accuracy, correspondence truth, camera or geometry evidence; no production "
    "behaviour, threshold, matcher, filter or solver decision is changed.")

#: Preregistered axis-1 (resultant, residual) predictions from the accepted
#: feasibility audit.  Design evidence only -- never an expected-value override.
PREREGISTERED_AXIS1 = {
    "C1": (0.736, None),
    "C2": (0.204, None),
    "C3": (0.989176, 0.020419),
    "C4": (0.957187, 0.040839),
    "C5": (0.974688, 0.031305),
}

#: arm -> ordered one-for-one substitutions ``(candidate flat row, match index)``
#: plus the expected same-axis group index of each candidate row.
SUBSTITUTIONS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "C1": ((203, 19, 4),),
    "C2": ((383, 33, 7),),
    "C3": ((203, 19, 4), (383, 33, 7)),
    "C4": ((204, 19, 4), (384, 33, 7)),
    "C5": ((206, 19, 4), (384, 33, 7)),
}


class ReplayValidationError(ValueError):
    """The pinned artifact cannot be reconstructed fail-closed."""


class BaselineReconstructionError(ReplayValidationError):
    """The unpinned control does not reproduce R10's recorded solver outcome."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of one file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic, ASCII, NaN-free JSON encoding used for sealing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def seal_payload(payload: Mapping[str, Any]) -> str:
    """SHA-256 seal over every field except the ``seal_sha256`` field itself."""
    unsigned = {key: value for key, value in payload.items()
                if key != "seal_sha256"}
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReadingUniverse:
    """The reconstructed reading universe plus the explicit plane remap."""

    readings: tuple[Any, ...]
    frames: tuple[str, ...]
    canonical_index: Mapping[tuple[str, int], int]
    recorded_plane: Mapping[tuple[str, int], int]
    reading_by_recorded: Mapping[tuple[str, int], Any]

    def canonical(self, frame: str, recorded_plane: int) -> int:
        try:
            return self.canonical_index[(frame, recorded_plane)]
        except KeyError as error:  # pragma: no cover - guarded by callers
            raise ReplayValidationError(
                "recorded plane is absent from the reading universe") from error


@dataclass(frozen=True)
class ExactSolveContext:
    """The R11 captured ``solve_component`` arguments, replay-ready.

    ``constraints`` preserves the capture's exact row order (which the digest
    binds); ``frames`` is the captured solver frame universe, cross-bound to the
    R10 reconstruction before use.  ``record_sha256`` is the production
    ``solver_solve_context`` digest, already re-derived by the production
    validator and additionally pinned to the Lead-accepted value.
    """

    integer_tolerance: float
    min_resultant: float
    ambiguous_margin: float
    constraints: tuple[tuple[str, str, int], ...]
    frames: tuple[str, ...]
    record_sha256: str

    def call_arguments(self) -> dict[str, Any]:
        """The exact keyword arguments the producing call handed the solver."""
        return {
            "integer_tolerance": self.integer_tolerance,
            "min_resultant": self.min_resultant,
            "ambiguous_margin": self.ambiguous_margin,
            "quadrant_constraints": self.constraints,
        }

    def to_report(self) -> dict[str, Any]:
        return {
            "record_sha256": self.record_sha256,
            "status": R11_CONTEXT_STATUS,
            "integer_tolerance": self.integer_tolerance,
            "min_resultant": self.min_resultant,
            "ambiguous_margin": self.ambiguous_margin,
            "constraints": [
                {"left": left, "right": right, "yaw": yaw}
                for left, right, yaw in self.constraints],
            "frames": list(self.frames),
        }


@dataclass(frozen=True)
class AxisSubstitution:
    """One validated exhaustive one-for-one world-axis-0/2 substitution.

    ``match`` is the fully rebuilt candidate ``PlaneMatch``.  ``score``/``delta``
    are the captured candidate-row values purely for provenance and later
    tie-breaking; ``flat_row``/``target_match`` identify the substitution.
    """

    world_axis: int
    group: int
    flat_row: int
    target_match: int
    score: float
    delta: float
    match: Any

    @property
    def arm_id(self) -> str:
        return f"A{self.world_axis}-{self.flat_row:03d}-{self.target_match:02d}"

    def key(self) -> tuple[int, int, int, int]:
        return (self.flat_row, self.target_match, self.group, self.world_axis)


def _synthetic_rectification() -> Any:
    """Inert rectification context; ``solve_component`` never reads it."""
    return lattice_planes.PlaneRectification(
        axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        focal=1.0,
        principal_point=(0.0, 0.0),
        centre=(0.0, 0.0),
    )


def _reading_from_row(row: Sequence[Any]) -> Any:
    reading_index, frame, plane, axis, side, alternatives = row
    if type(reading_index) is not int or reading_index < 0:
        raise ReplayValidationError("reading row index is malformed")
    if type(frame) is not str or not frame:
        raise ReplayValidationError("reading row frame is malformed")
    if type(plane) is not int or plane < 0:
        raise ReplayValidationError("reading row plane is malformed")
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ReplayValidationError("reading row axis is malformed")
    if type(side) is not int or side not in (-1, 1):
        raise ReplayValidationError("reading row side is malformed")
    return lattice_planes.PlaneReading(
        frame=frame,
        axis=int(axis),
        side=int(side),
        alternatives=tuple(
            lattice_planes.DistanceAlternative(
                float(entry[0]), float(entry[1]), float(entry[2]))
            for entry in alternatives),
        # Synthetic tie-breaker only: the recorded plane index orders one frame's
        # readings deterministically because every other order-key field is
        # constant.  Never solver-relevant.
        period_pixels=float(plane),
        phase_pixels=(0.0, 0.0),
        extent=0.0,
        samples=0,
        polygon=None,
        coverage_cells=0,
        structural_share=0.0,
        confidence=0.0,
        rectification=_synthetic_rectification(),
    )


def build_reading_universe(record: Mapping[str, Any]) -> ReadingUniverse:
    """Rebuild the 15 referenced readings and the explicit canonical remap."""
    rows = record.get("readings")
    if type(rows) is not list or not rows:
        raise ReplayValidationError("reading table is missing or empty")
    readings: list[Any] = []
    for row in rows:
        readings.append(_reading_from_row(row))

    by_frame: dict[str, list[tuple[Any, int, Any]]] = {}
    for row, reading in zip(rows, readings):
        frame = row[1]
        plane = int(row[2])
        if any(known == plane for _key, known, _reading in by_frame.get(frame, [])):
            raise ReplayValidationError(
                "reading table repeats a frame/plane pair")
        by_frame.setdefault(frame, []).append(
            (lattice_planes.reading_order_key(reading), plane, reading))

    canonical_index: dict[tuple[str, int], int] = {}
    recorded_plane: dict[tuple[str, int], int] = {}
    reading_by_recorded: dict[tuple[str, int], Any] = {}
    for frame, items in by_frame.items():
        ordered = sorted(items, key=lambda item: item[0])
        keys = [item[0] for item in ordered]
        if len(set(keys)) != len(keys):
            raise ReplayValidationError(
                "synthetic reading order keys are ambiguous inside a frame")
        for position, (_key, plane, reading) in enumerate(ordered):
            canonical_index[(frame, plane)] = position
            recorded_plane[(frame, position)] = plane
            reading_by_recorded[(frame, plane)] = reading

    for frame in by_frame:
        planes = sorted(plane for (name, plane) in canonical_index
                        if name == frame)
        expected = list(range(len(planes)))
        observed = sorted(canonical_index[(frame, plane)] for plane in planes)
        if observed != expected:
            raise ReplayValidationError(
                "recorded plane -> canonical index remap is not a bijection")
    frames = tuple(sorted(by_frame))
    return ReadingUniverse(
        readings=tuple(readings),
        frames=frames,
        canonical_index=canonical_index,
        recorded_plane=recorded_plane,
        reading_by_recorded=reading_by_recorded,
    )


def select_trace_rows(
    record: Mapping[str, Any], trace: Mapping[str, Any],
) -> Mapping[tuple[str, int, str, int, int], Mapping[str, Any]]:
    """Return the unique ``(left, left_plane, right, right_plane, quadrant)`` map."""
    pairs = trace.get("pairs")
    if type(pairs) is not list:
        raise ReplayValidationError("trace.pairs is missing")
    selected = [pair for pair in pairs
                if isinstance(pair, dict) and pair.get("selected") is True]
    identities = record.get("input_identities")
    if type(identities) is not list:
        raise ReplayValidationError("input_identities is missing")
    if len(selected) != len(identities):
        raise ReplayValidationError(
            "selected trace row count does not equal the input match count")
    joined: dict[tuple[str, int, str, int, int], Mapping[str, Any]] = {}
    for pair in selected:
        key = (pair.get("left"), pair.get("left_plane"), pair.get("right"),
               pair.get("right_plane"), pair.get("quadrant"))
        if None in key:
            raise ReplayValidationError("selected trace row key is malformed")
        if key in joined:
            raise ReplayValidationError(
                "selected trace rows are not pairwise unique")
        joined[key] = pair
    identity_keys = {
        (row[0], int(row[1]), row[3], int(row[4]), int(row[6]))
        for row in identities}
    if len(identity_keys) != len(identities):
        raise ReplayValidationError("input identities are not pairwise unique")
    if identity_keys != set(joined):
        raise ReplayValidationError(
            "selected trace rows do not join the identity set exactly")
    return joined


def _kept_candidate_overrides(
    record: Mapping[str, Any],
) -> Mapping[int, Sequence[Any]]:
    """Map each retained match index to its exact candidate row."""
    candidates = record.get("candidates")
    groups = record.get("groups")
    if type(candidates) is not list or type(groups) is not list:
        raise ReplayValidationError("candidate rows or groups are missing")
    claims = {group[0]: tuple(group[5]) for group in groups}
    overrides: dict[int, Sequence[Any]] = {}
    for row in candidates:
        if row[17] != "kept":
            continue
        match_index = int(row[1])
        if match_index in overrides:
            raise ReplayValidationError("a retained match index is repeated")
        if match_index not in claims.get(int(row[0]), ()):
            raise ReplayValidationError(
                "a retained candidate row is not claimed by its group")
        overrides[match_index] = row
    return overrides


def reconstruct_matches(
    record: Mapping[str, Any], trace: Mapping[str, Any],
    universe: ReadingUniverse,
) -> list[Any]:
    """Rebuild the exact 38 ``PlaneMatch`` inputs in solver-input order."""
    identities = record["input_identities"]
    joined = select_trace_rows(record, trace)
    overrides = _kept_candidate_overrides(record)
    matches: list[Any] = []
    for index, identity in enumerate(identities):
        left_frame, left_plane, left_axis = (
            identity[0], int(identity[1]), int(identity[2]))
        right_frame, right_plane, right_axis = (
            identity[3], int(identity[4]), int(identity[5]))
        relative = int(identity[6])
        left_b = float(identity[7])
        right_b = float(identity[8])
        left_reading = universe.reading_by_recorded.get((left_frame, left_plane))
        right_reading = universe.reading_by_recorded.get(
            (right_frame, right_plane))
        if left_reading is None or right_reading is None:
            raise ReplayValidationError(
                "an input identity references a missing reading")
        if left_reading.axis != left_axis or right_reading.axis != right_axis:
            raise ReplayValidationError(
                "an input identity axis contradicts its reading")
        pair = joined[(left_frame, left_plane, right_frame, right_plane, relative)]
        peak = 0.0
        second_peak = 0.0
        override = overrides.get(index)
        if override is not None:
            # Candidate row layout: ... score(8) psr_value(9) psr_infinite(10)
            # peak(11) second_peak(12) overlap_cells(13) ...  Only peak and
            # second_peak are copied; neither is read by solve_component.
            if (override[11] is None or override[12] is None):
                raise ReplayValidationError(
                    "a retained candidate row lacks peak fields")
            peak = float(override[11])
            second_peak = float(override[12])
        shift = pair.get("cell_shift_rc")
        matches.append(plane_graph.PlaneMatch(
            left=left_frame,
            right=right_frame,
            left_axis=left_axis,
            right_axis=right_axis,
            left_b=left_b,
            right_b=right_b,
            quadrant_of_right=relative,
            psr=float(pair["psr"]),
            peak=peak,
            second_peak=second_peak,
            overlap_cells=int(pair["overlap_cells"]),
            integer_residual=None,
            score=float(pair["score"]),
            left_plane=universe.canonical(left_frame, left_plane),
            right_plane=universe.canonical(right_frame, right_plane),
            cell_shift_rc=(int(shift[0]), int(shift[1])) if shift else None,
        ))
    return matches


def candidate_match(
    record: Mapping[str, Any], universe: ReadingUniverse, flat_index: int,
) -> tuple[Any, Sequence[Any]]:
    """Build one substitution match from a complete candidate row."""
    candidates = record.get("candidates")
    if type(candidates) is not list or not (0 <= flat_index < len(candidates)):
        raise ReplayValidationError("candidate flat row index is out of range")
    row = candidates[flat_index]
    (group_index, match_index, _row_index, left_ref, left_alt, right_ref,
     right_alt, relative, score, psr_value, psr_infinite, peak, second_peak,
     overlap_cells, shift_row, shift_col, delta, disposition,
     reason) = row
    if disposition != "dropped" or match_index != -1:
        raise ReplayValidationError(
            "a substitution candidate row must be a dropped row")
    if reason != plane_graph.SAME_AXIS_DISPOSITION_DROPPED:
        raise ReplayValidationError(
            "a substitution candidate row carries no mismatch reason")
    table = {item[0]: item for item in record["readings"]}
    left_row = table.get(left_ref)
    right_row = table.get(right_ref)
    if left_row is None or right_row is None:
        raise ReplayValidationError("a candidate row references a missing reading")
    if left_alt >= len(left_row[5]) or right_alt >= len(right_row[5]):
        raise ReplayValidationError("a candidate alternative index is out of range")
    left_frame, left_plane, left_axis = (
        left_row[1], int(left_row[2]), int(left_row[3]))
    right_frame, right_plane, right_axis = (
        right_row[1], int(right_row[2]), int(right_row[3]))
    left_distance = float(left_row[5][left_alt][1])
    right_distance = float(right_row[5][right_alt][1])
    _mapped, sigma = plane_graph.axis_transform(right_axis, int(relative))
    expected_delta = float(
        float(left_row[4]) * left_distance
        - float(sigma) * float(right_row[4]) * right_distance)
    if not math.isclose(float(delta), expected_delta, rel_tol=0.0, abs_tol=1e-12):
        raise ReplayValidationError("a candidate row delta does not replay")
    match = plane_graph.PlaneMatch(
        left=left_frame,
        right=right_frame,
        left_axis=left_axis,
        right_axis=right_axis,
        left_b=float(left_row[5][left_alt][0]),
        right_b=float(right_row[5][right_alt][0]),
        quadrant_of_right=int(relative),
        psr=math.inf if psr_infinite else float(psr_value),
        peak=float(peak),
        second_peak=float(second_peak),
        overlap_cells=int(overlap_cells),
        integer_residual=None,
        score=float(score),
        left_plane=universe.canonical(left_frame, left_plane),
        right_plane=universe.canonical(right_frame, right_plane),
        cell_shift_rc=(int(shift_row), int(shift_col)),
    )
    return match, row


def substitute(matches: Sequence[Any], replacements: Mapping[int, Any]) -> list[Any]:
    """Apply strict one-for-one substitutions, preserving length and order."""
    result = list(matches)
    if len(set(replacements)) != len(replacements):
        raise ReplayValidationError("substitution indices are not unique")
    for index, replacement in replacements.items():
        if not (0 <= index < len(result)):
            raise ReplayValidationError("substitution index is out of range")
        result[index] = replacement
    if len(result) != len(matches):
        raise ReplayValidationError("substitution changed the match count")
    return result


def pinned_constraints(
    frames: Sequence[str], target: Sequence[int],
) -> tuple[tuple[str, str, int], ...]:
    """Deterministic pairwise quadrant constraints that pin one assignment."""
    if len(frames) != len(target):
        raise ReplayValidationError("pin target length does not match frames")
    return tuple(
        (frames[left], frames[right], (int(target[right]) - int(target[left])) % 4)
        for left in range(len(frames))
        for right in range(left + 1, len(frames)))


def _active_indices(matches: Sequence[Any], frames: Sequence[str],
                    quadrants: Sequence[int]) -> list[int]:
    position = {frame: index for index, frame in enumerate(frames)}
    return [
        index for index, match in enumerate(matches)
        if (quadrants[position[match.right]] - quadrants[position[match.left]]) % 4
        == match.quadrant_of_right]


def run_arm(
    readings: Sequence[Any], matches: Sequence[Any], frames: Sequence[str],
    *, target: Sequence[int] | None,
    context: ExactSolveContext | None = None,
) -> dict[str, Any]:
    """Call the unchanged public solver once and project its observable outcome.

    ``target`` selects the deterministic all-pairs ``pinned_constraints`` pin of
    the R10 baseline, ``None`` an unconstrained control, and ``context`` the R11
    exact captured call (its own ordered constraints and thresholds).  At most
    one of ``target``/``context`` may be supplied, so a single call is always
    one solver invocation with one explicitly known argument set.
    """
    if context is not None:
        if target is not None:
            raise ReplayValidationError(
                "a solver call cannot be both target-pinned and context-pinned")
        if tuple(context.frames) != tuple(frames):
            raise ReplayValidationError(
                "the exact context frame universe is foreign to the solver call")
        constraints = context.constraints
        call_kwargs = context.call_arguments()
        mode = "exact_context"
        pinned = True
    else:
        constraints = () if target is None else pinned_constraints(frames, target)
        call_kwargs = {"quadrant_constraints": constraints}
        mode = "pinned" if target is not None else "unpinned"
        pinned = target is not None
    started = time.perf_counter()
    solution = plane_graph.solve_component(
        readings, matches, **call_kwargs)
    elapsed = time.perf_counter() - started
    quadrants = [int(solution.quadrants[frame]) for frame in frames]
    active = _active_indices(matches, frames, quadrants)
    keys = [(matches[index].left, matches[index].left_plane,
             matches[index].right, matches[index].right_plane,
             matches[index].quadrant_of_right) for index in active]
    observations = None
    if solution.alternatives:
        observations = [int(value)
                        for value in solution.alternatives[0].axis_observations]
    outcome = {
        "mode": mode,
        "pinned": pinned,
        "reason": solution.reason,
        "accepted": solution.reason == "",
        "assignment": quadrants,
        "resultants": {str(axis): float(solution.axis_resultants[axis])
                       for axis in sorted(solution.axis_resultants)},
        "residuals": {str(axis): float(solution.axis_residuals[axis])
                      for axis in sorted(solution.axis_residuals)},
        "observations": observations,
        "active_match_indices": active,
        "active_match_count": len(active),
        "duplicate_b": len(keys) != len(set(keys)),
        "translations": {
            frame: [float(value) for value in solution.translations[frame]]
            for frame in frames},
        "elapsed_seconds": elapsed,
    }
    if context is not None:
        outcome["exact_context"] = context.to_report()
    return outcome


def recorded_outcome(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a stage artifact's own recorded solver outcome."""
    observation = artifact.get("solver_failure_observation")
    if not isinstance(observation, dict):
        raise ReplayValidationError("solver_failure_observation is missing")
    raw = observation.get("raw")
    if not isinstance(raw, dict):
        raise ReplayValidationError("solver_failure_observation.raw is missing")
    phase = observation.get("phase_capture")
    translations: dict[str, list[float]] = {}
    if isinstance(phase, dict) and isinstance(phase.get("translations"), list):
        translations = {str(frame): [float(value) for value in row]
                        for frame, row in phase["translations"]}
    return {
        "reason": raw.get("reason"),
        "assignment": [int(value) for value in raw["best_alternative_quadrants"]],
        "resultants": {str(item["axis"]): float(item["value"])
                       for item in raw["axis_resultants"]},
        "residuals": {str(item["axis"]): float(item["value"])
                      for item in raw["axis_residuals"]},
        "observations": [int(value)
                         for value in raw["best_alternative_axis_observations"]],
        "translations": translations,
    }


def r10_expected(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for the shared recorded-outcome extractor."""
    return recorded_outcome(artifact)


def control_mismatches(arm: Mapping[str, Any], expected: Mapping[str, Any],
                       tolerance: float) -> list[str]:
    """Return the concrete reasons an arm fails the R10 control gate."""
    problems: list[str] = []
    if arm.get("reason") != expected["reason"]:
        problems.append(
            f"reason {arm.get('reason')!r} != {expected['reason']!r}")
    if list(arm.get("assignment", [])) != list(expected["assignment"]):
        problems.append(
            f"assignment {arm.get('assignment')} != {expected['assignment']}")
    if list(arm.get("observations") or []) != list(expected["observations"]):
        problems.append(
            f"observations {arm.get('observations')} != {expected['observations']}")
    for name in ("resultants", "residuals"):
        observed = arm.get(name) or {}
        reference = expected[name]
        for axis, value in reference.items():
            if axis not in observed:
                problems.append(f"{name}[{axis}] is missing")
                continue
            if not math.isclose(float(observed[axis]), float(value),
                                rel_tol=0.0, abs_tol=tolerance):
                problems.append(
                    f"{name}[{axis}] {observed[axis]!r} != {value!r}")
    for frame, value in expected["translations"].items():
        observed = (arm.get("translations") or {}).get(frame)
        if observed is None:
            problems.append(f"translations[{frame}] is missing")
            continue
        if any(not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
               for a, b in zip(observed, value)):
            problems.append(f"translations[{frame}] {observed} != {value}")
    return problems


def validate_record_bindings(
    record: Mapping[str, Any], trace: Mapping[str, Any],
) -> None:
    """Replay the record and cross-bind its shared digests to the trace."""
    graph_mod.validate_same_axis_candidates(record)
    if record.get("schema") != CAPTURE_SCHEMA_V2:
        raise ReplayValidationError("capture schema is not the v2 aggregate")
    projected = trace.get("solver_failure")
    if not isinstance(projected, dict):
        raise ReplayValidationError("trace.solver_failure is missing")
    aliases = projected.get("active_edge_aliases")
    if not isinstance(aliases, dict) or aliases.get("status") != "complete":
        raise ReplayValidationError(
            "active_edge_aliases binding is absent or not complete")
    if record.get("failed_assignment_sha256") != aliases.get(
            "failed_assignment_sha256"):
        raise ReplayValidationError("failed assignment digest does not bind")
    if record.get("input_match_identity_sha256") != aliases.get(
            "input_match_identity_sha256"):
        raise ReplayValidationError("input identity digest does not bind")


def cross_check_solver_bindings(
    artifact: Mapping[str, Any], record: Mapping[str, Any],
    universe: ReadingUniverse, matches: Sequence[Any],
) -> None:
    """Independently compare the 38 endpoints with the artifact's own binding."""
    observation = artifact.get("solver_failure_observation")
    if not isinstance(observation, dict):
        raise ReplayValidationError("solver_failure_observation is missing")
    bindings = observation.get("solver_input_bindings")
    if not isinstance(bindings, dict) or bindings.get("status") != "complete":
        raise ReplayValidationError("solver_input_bindings is not complete")
    edges = bindings.get("edges")
    if type(edges) is not list or len(edges) != len(matches):
        raise ReplayValidationError(
            "solver_input_bindings edge count does not match")
    table = {row[0]: row for row in record["readings"]}
    for index, (edge, match) in enumerate(zip(edges, matches)):
        match_index, left, right, relative, _active = edge
        if int(match_index) != index:
            raise ReplayValidationError("solver binding edge order does not replay")
        if int(relative) != match.quadrant_of_right:
            raise ReplayValidationError("solver binding quadrant does not replay")
        for endpoint, frame, plane, axis, selected_b in (
                (left, match.left, match.left_plane, match.left_axis, match.left_b),
                (right, match.right, match.right_plane, match.right_axis,
                 match.right_b)):
            reading_ref, alternative, bound_b, bound_distance, _support, side, \
                _sign, bound_axis = endpoint
            row = table.get(reading_ref)
            if row is None:
                raise ReplayValidationError("solver binding reading is missing")
            if (str(frame) != row[1] or int(plane) != universe.canonical(
                    row[1], int(row[2]))):
                raise ReplayValidationError(
                    "solver binding endpoint does not replay")
            if int(axis) != int(row[3]) or int(bound_axis) != int(row[3]):
                raise ReplayValidationError("solver binding axis does not replay")
            if int(side) != int(row[4]):
                raise ReplayValidationError("solver binding side does not replay")
            if not math.isclose(float(selected_b), float(bound_b),
                                rel_tol=0.0, abs_tol=1e-15):
                raise ReplayValidationError("solver binding b does not replay")
            if not math.isclose(float(bound_distance),
                                float(row[5][alternative][1]),
                                rel_tol=0.0, abs_tol=1e-12):
                raise ReplayValidationError(
                    "solver binding alternative distance does not replay")


def _group_claims(record: Mapping[str, Any]) -> Mapping[int, tuple[int, ...]]:
    return {int(group[0]): tuple(int(value) for value in group[5])
            for group in record["groups"]}


def build_treatments(
    record: Mapping[str, Any], universe: ReadingUniverse,
    baseline: Sequence[Any],
) -> dict[str, Any]:
    """Build and run every one-for-one counterfactual arm.

    Evaluation-only, and reached only when the unpinned control gate passes:
    ``replay_artifact`` never calls this after a control mismatch.
    """
    claims = _group_claims(record)
    arms: dict[str, Any] = {}
    for name, specification in SUBSTITUTIONS.items():
        replacements: dict[int, Any] = {}
        provenance: list[dict[str, Any]] = []
        for flat_index, match_index, group_index in specification:
            if match_index not in claims.get(group_index, ()):
                raise ReplayValidationError(
                    f"{name} replacement index is not a survivor of its group")
            match, row = candidate_match(record, universe, flat_index)
            if int(row[0]) != group_index:
                raise ReplayValidationError(
                    f"{name} candidate row is not in the declared group")
            replacements[match_index] = match
            provenance.append({
                "candidate_flat_row": flat_index,
                "candidate_group": group_index,
                "replaces_match_index": match_index,
                "candidate_score": float(row[8]),
                "candidate_delta": float(row[16]),
            })
        treatment = substitute(baseline, replacements)
        arms[name] = {
            "provenance": provenance,
            "preregistered_axis1": PREREGISTERED_AXIS1.get(name),
            "pinned": run_arm(universe.readings, treatment, universe.frames,
                              target=(0, 3, 1)),
            "unpinned": run_arm(universe.readings, treatment, universe.frames,
                                target=None),
        }
    return arms


@dataclass(frozen=True)
class R10Reconstruction:
    """Fail-closed, replay-ready R10 inputs plus their validation record."""

    artifact: Mapping[str, Any]
    record: Mapping[str, Any]
    trace: Mapping[str, Any]
    universe: ReadingUniverse
    matches: tuple[Any, ...]
    validation: Mapping[str, Any]


def load_r10_reconstruction(
    artifact_path: Path, *, expected_sha256: str = INPUT_FILE_SHA256,
) -> R10Reconstruction:
    """Load, validate and reconstruct the pinned R10 solver inputs.

    Shared by the R10 replay and the R11 exact-context control so both consume
    byte-identical reconstructed inputs; every count, digest and binding is
    replayed before any solver call.
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise ReplayValidationError(f"pinned artifact is absent: {artifact_path}")
    observed_sha256 = sha256_file(artifact_path).upper()
    if observed_sha256 != expected_sha256.upper():
        raise ReplayValidationError(
            "artifact file SHA-256 does not match the pinned value")
    with open(artifact_path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("schema") != INPUT_SCHEMA:
        raise ReplayValidationError("artifact schema is not the R10 stage schema")
    record = artifact.get("same_axis_candidates")
    trace = artifact.get("trace")
    if not isinstance(record, dict) or not isinstance(trace, dict):
        raise ReplayValidationError("capture record or trace is missing")

    validation: dict[str, Any] = {"artifact_file_sha256": observed_sha256}
    validate_record_bindings(record, trace)
    validation["record_replay"] = "pass"
    validation["runner_cross_binding"] = "pass"

    graph_payload = artifact.get("graph", {}).get("payload", {})
    if graph_payload.get("matches") != []:
        raise ReplayValidationError("R10 graph payload carries matches")
    if graph_payload.get("component_solution") is not None:
        raise ReplayValidationError("R10 graph payload carries a component solution")
    validation["graph_matches_empty"] = True

    if record.get("input_match_count") != 38:
        raise ReplayValidationError("R10 input match count is not 38")
    if record.get("group_count") != 9 or record.get("row_count") != 448:
        raise ReplayValidationError("R10 group/row counts are not 9/448")
    if len(record.get("readings", [])) != 15:
        raise ReplayValidationError("R10 reading table is not 15 rows")

    universe = build_reading_universe(record)
    matches = reconstruct_matches(record, trace, universe)
    if len(matches) != 38:
        raise ReplayValidationError("reconstruction did not yield 38 matches")
    validation["selected_trace_rows"] = 38
    validation["identity_join_bijective"] = True
    validation["reading_count"] = len(universe.readings)
    validation["candidate_row_count"] = len(record["candidates"])
    validation["plane_index_remap"] = {
        f"{frame}:{plane}": index
        for (frame, plane), index in sorted(universe.canonical_index.items())}
    validation["plane_index_remap_identity"] = all(
        universe.canonical_index[(frame, plane)] == plane
        for (frame, plane) in universe.canonical_index)
    cross_check_solver_bindings(artifact, record, universe, matches)
    validation["solver_input_binding_cross_check"] = "pass"
    return R10Reconstruction(
        artifact=artifact, record=record, trace=trace, universe=universe,
        matches=tuple(matches), validation=validation)


def replay_artifact(
    artifact_path: Path, *, expected_sha256: str = INPUT_FILE_SHA256,
    tolerance: float = CONTROL_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Run the full fail-closed replay and return the sealed output payload."""
    artifact_path = Path(artifact_path)
    reconstruction = load_r10_reconstruction(
        artifact_path, expected_sha256=expected_sha256)
    artifact = reconstruction.artifact
    record = reconstruction.record
    universe = reconstruction.universe
    matches = reconstruction.matches
    validation = dict(reconstruction.validation)
    observed_sha256 = validation["artifact_file_sha256"]

    expected = recorded_outcome(artifact)
    pinned_baseline = run_arm(universe.readings, matches, universe.frames,
                              target=(0, 3, 1))
    control = run_arm(universe.readings, matches, universe.frames, target=None)
    problems = control_mismatches(control, expected, tolerance)
    pinned_problems = control_mismatches(pinned_baseline, expected, tolerance)
    gate = {
        "mode": "unpinned",
        "tolerance_absolute": tolerance,
        "passed": not problems,
        "mismatches": problems,
        "pinned_baseline_passed": not pinned_problems,
        "pinned_baseline_mismatches": pinned_problems,
    }

    if gate["passed"]:
        treatments = build_treatments(record, universe, matches)
        interpretable = True
        status = "complete"
        notes = [
            "Unmodified, unpinned control reproduces R10 exactly; counterfactual "
            "arms are interpretable as bounded feasibility evidence."]
    else:
        # Fail closed on the required control: the unpinned optimum does not
        # reproduce R10's recorded assignment, so the producing call carried
        # quadrant constraints the pinned artifact does not serialize.  No C1-C5
        # arm may be built or solved, and no harmful-pruning verdict follows.
        treatments = {}
        interpretable = False
        status = "baseline_unreconstructed"
        notes = [
            "The R10 capture record and its 38 solver inputs replay exactly; the "
            "pinned [0, 3, 1] assignment reproduces every recorded R10 per-axis "
            "metric and translation.",
            "The required unmodified, unpinned control does NOT reproduce R10: the "
            "unconstrained optimum is a different assignment, so the producing call "
            "carried quadrant constraints that the pinned artifact does not "
            "serialize.",
            "Per the task stop condition no counterfactual arm was built or solved "
            "(no C1-C5 solver call); treatments.arms is empty and only the passing "
            "pinned baseline and the observed unpinned control are retained as "
            "diagnostics.",
        ]

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "no_truth_disclaimer": NO_TRUTH_DISCLAIMER,
        "input": {
            "path": str(artifact_path),
            "file_sha256": observed_sha256,
            "expected_file_sha256": expected_sha256.upper(),
            "artifact_sha256": artifact.get("artifact_sha256"),
            "record_sha256": record.get("record_sha256"),
            "failed_assignment_sha256": record.get("failed_assignment_sha256"),
            "input_match_identity_sha256": record.get(
                "input_match_identity_sha256"),
            "producer_plane_graph_sha256": (
                "3ed45cc7f8b0460e26ce45c6b2f7bf16493cb772d3f97dedaf9cfe2657584b47"),
        },
        "validation": validation,
        "control_gate": gate,
        "control_unpinned": control,
        "pinned_baseline": pinned_baseline,
        "r10_expected": expected,
        "treatments": {
            "interpretable": interpretable,
            "arms": treatments,
        },
        "notes": notes,
    }
    payload["seal_sha256"] = seal_payload(payload)
    return payload


def _r11_raw_solve_context(
    artifact: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return one artifact's failure observation and raw solve-context record."""
    observation = artifact.get("solver_failure_observation")
    if not isinstance(observation, dict):
        raise ReplayValidationError("R11 solver_failure_observation is missing")
    raw = observation.get("raw")
    if not isinstance(raw, dict):
        raise ReplayValidationError("R11 solver_failure_observation.raw is missing")
    record = raw.get("solve_context")
    if not isinstance(record, dict):
        raise ReplayValidationError("R11 solve_context is absent or malformed")
    return observation, record


def exact_solve_context(
    artifact: Mapping[str, Any], *,
    expected_frames: Sequence[str] | None = None,
    expected_context_sha256: str = R11_CONTEXT_SHA256,
) -> ExactSolveContext:
    """Validate the R11 capture and return it as replay-ready solver inputs.

    The production ``validate_solver_solve_context`` replays the schema, basis,
    status, numeric value shapes, ordered constraint rows, frame universe,
    digest, byte budget and (when supplied) the frame cross-binding; the digest
    is then additionally pinned to the Lead-accepted value.  A refused or
    incomplete capture is never turned into solver inputs.
    """
    _observation, record = _r11_raw_solve_context(artifact)
    if record.get("status") != R11_CONTEXT_STATUS:
        raise ReplayValidationError(
            "R11 solve context is not an exact complete capture")
    try:
        graph_mod.validate_solver_solve_context(record, expected_frames)
    except ValueError as error:
        raise ReplayValidationError(
            "R11 solve context failed production replay validation") from error
    digest = record.get("record_sha256")
    if type(digest) is not str or digest.lower() != expected_context_sha256.lower():
        raise ReplayValidationError(
            "R11 solve context digest does not match the pinned value")
    constraints = tuple(
        (str(row["left"]), str(row["right"]), int(row["yaw"]))
        for row in record["constraints"])
    frames = tuple(str(frame) for frame in record["frames"])
    if expected_frames is not None and frames != tuple(expected_frames):
        raise ReplayValidationError(
            "R11 solve context frame universe is foreign to the replay")
    return ExactSolveContext(
        integer_tolerance=float(record["integer_tolerance"]),
        min_resultant=float(record["min_resultant"]),
        ambiguous_margin=float(record["ambiguous_margin"]),
        constraints=constraints,
        frames=frames,
        record_sha256=digest,
    )


def reconcile_shared_inputs(
    r10_artifact: Mapping[str, Any], r11_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the R10 and R11 stage runs share graph/trace/source inputs.

    The R11 run adds only the solve-context diagnostic and its own artifact /
    entry / telemetry digests.  Everything that defines the 38 solver inputs --
    graph result, trace (and its source digests), the whole same-axis capture
    record and the pinned stage inputs -- must therefore be identical before a
    cross-artifact control may run.
    """
    mismatches = [key for key in RECONCILED_KEYS
                  if r10_artifact.get(key) != r11_artifact.get(key)]
    r10_trace = r10_artifact.get("trace")
    r11_trace = r11_artifact.get("trace")
    r10_sources = (
        r10_trace.get("source_digests") if isinstance(r10_trace, dict) else None)
    r11_sources = (
        r11_trace.get("source_digests") if isinstance(r11_trace, dict) else None)
    if r10_sources != r11_sources:
        mismatches.append("trace.source_digests")
    if mismatches:
        raise ReplayValidationError(
            "R10/R11 shared graph/trace/source inputs differ: "
            + ", ".join(sorted(set(mismatches))))
    record = r10_artifact.get("same_axis_candidates")
    record = record if isinstance(record, dict) else {}

    def _observation_digest(artifact: Mapping[str, Any]) -> Any:
        observation = artifact.get("solver_failure_observation")
        return (observation.get("observation_sha256")
                if isinstance(observation, dict) else None)

    return {
        "reconciled_keys": list(RECONCILED_KEYS),
        "graph_sha256": r10_artifact.get("graph_sha256"),
        "trace_sha256": r10_artifact.get("trace_sha256"),
        "source_digests": r10_sources,
        "project_sha256": r10_artifact.get("project_sha256"),
        "record_sha256": record.get("record_sha256"),
        "failed_assignment_sha256": record.get("failed_assignment_sha256"),
        "input_match_identity_sha256": record.get("input_match_identity_sha256"),
        "input_match_count": record.get("input_match_count"),
        "observation_sha256": {
            "r10": _observation_digest(r10_artifact),
            "r11": _observation_digest(r11_artifact),
        },
    }


def replay_r11_control(
    r10_path: Path, r11_path: Path, *,
    expected_r10_sha256: str = INPUT_FILE_SHA256,
    expected_r11_sha256: str = R11_INPUT_FILE_SHA256,
    expected_context_sha256: str = R11_CONTEXT_SHA256,
    tolerance: float = CONTROL_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Run exactly one R11 exact-context, unpruned control and seal the result.

    R10 supplies the 38 reconstructed solver inputs, R11 supplies the exact
    captured call.  The two are reconciled and the R11 context is replayed and
    frame-bound *before* the single solver call.  No treatment arm is ever
    built, so ``treatments.arms`` is empty by construction.
    """
    r10_path = Path(r10_path)
    r11_path = Path(r11_path)
    if not r11_path.is_file():
        raise ReplayValidationError(f"pinned R11 artifact is absent: {r11_path}")
    observed_r11_sha256 = sha256_file(r11_path).upper()
    if observed_r11_sha256 != expected_r11_sha256.upper():
        raise ReplayValidationError(
            "R11 artifact file SHA-256 does not match the pinned value")
    with open(r11_path, "r", encoding="utf-8") as handle:
        r11_artifact = json.load(handle)
    if r11_artifact.get("schema") != INPUT_SCHEMA:
        raise ReplayValidationError("R11 artifact schema is not the stage schema")

    reconstruction = load_r10_reconstruction(
        r10_path, expected_sha256=expected_r10_sha256)
    r10_artifact = reconstruction.artifact
    universe = reconstruction.universe
    matches = reconstruction.matches
    validation = dict(reconstruction.validation)
    validation["r11_artifact_file_sha256"] = observed_r11_sha256
    validation["r11_context_replay"] = "pass"
    validation["r11_context_frame_binding"] = "pass"
    validation["r10_r11_reconciliation"] = "pass"

    # Ordering is deliberate: reconcile and validate both pinned inputs first,
    # then make the one authorized solver call.
    reconciliation = reconcile_shared_inputs(r10_artifact, r11_artifact)
    context = exact_solve_context(
        r11_artifact, expected_frames=universe.frames,
        expected_context_sha256=expected_context_sha256)

    expected = recorded_outcome(r10_artifact)
    r11_recorded = recorded_outcome(r11_artifact)
    control = run_arm(universe.readings, matches, universe.frames,
                      target=None, context=context)
    problems = control_mismatches(control, expected, tolerance)
    r11_problems = control_mismatches(control, r11_recorded, tolerance)
    passed = not problems and not r11_problems
    status = "complete" if passed else "control_mismatch"

    if passed:
        notes = [
            "R10 and R11 share byte-identical graph, trace, source and "
            "same-axis capture inputs; the R11 solve-context record replays "
            "through the production validator and its frames bind to the R10 "
            "reconstruction's solver frame universe.",
            "Exactly one unpruned solver call was made, carrying R11's exact "
            "captured quadrant constraints and thresholds; it reproduces R10's "
            "recorded [0, 3, 1] assignment, plane_integer_residual reason, "
            "per-axis resultants/residuals/observations and translations.",
            "No C1-C5 or other treatment arm was built or solved; no pruning "
            "loss is inferred and no tolerance was relaxed.",
        ]
    else:
        notes = [
            "The R11 exact context did not reproduce R10's recorded outcome; "
            "per the task stop condition the exact mismatch list is reported "
            "and no further solver call was made.",
            "Tolerances were not relaxed and no pruning loss is inferred or "
            "derivable from this control.",
        ]

    payload: dict[str, Any] = {
        "schema": R11_CONTROL_SCHEMA,
        "task_id": "DS-IMG-SOLVER-CONTEXT-R11-REPLAY-CONTROL-20261004",
        "status": status,
        "no_truth_disclaimer": NO_TRUTH_DISCLAIMER,
        "inputs": {
            "r10": {
                "path": str(r10_path),
                "file_sha256": validation["artifact_file_sha256"],
                "expected_file_sha256": expected_r10_sha256.upper(),
                "artifact_sha256": r10_artifact.get("artifact_sha256"),
                "observation_sha256": (
                    r10_artifact["solver_failure_observation"].get(
                        "observation_sha256")),
                "record_sha256": reconstruction.record.get("record_sha256"),
                "failed_assignment_sha256": reconstruction.record.get(
                    "failed_assignment_sha256"),
                "input_match_identity_sha256": reconstruction.record.get(
                    "input_match_identity_sha256"),
            },
            "r11": {
                "path": str(r11_path),
                "file_sha256": observed_r11_sha256,
                "expected_file_sha256": expected_r11_sha256.upper(),
                "artifact_sha256": r11_artifact.get("artifact_sha256"),
                "entry_sha256": r11_artifact.get("entry_sha256"),
                "observation_sha256": (
                    r11_artifact["solver_failure_observation"].get(
                        "observation_sha256")),
                "solve_context": context.to_report(),
            },
        },
        "reconciliation": reconciliation,
        "validation": validation,
        "control": control,
        "r10_expected": expected,
        "r11_recorded": r11_recorded,
        "control_gate": {
            "mode": "exact_context_pinned_unpruned",
            "tolerance_absolute": tolerance,
            "passed": passed,
            "mismatches": problems,
            "r11_recorded_mismatches": r11_problems,
        },
        "solver_calls_executed": 1,
        "treatments": {
            "authorized": False,
            "interpretable": False,
            "arms": {},
            "arms_executed": 0,
        },
        "notes": notes,
    }
    payload["seal_sha256"] = seal_payload(payload)
    return payload


def _validate_r11_baseline_payload(
    payload: Mapping[str, Any], *,
    expected_seal_sha256: str = R11_BASELINE_SEAL_SHA256,
) -> Mapping[str, Any]:
    """Fail-closed checks on an already-loaded sealed R11 baseline replay.

    These are the acceptance conditions the Lead recorded for the baseline, so a
    file that does not replay its own seal, is not the accepted control task, or
    unexpectedly carries arms can never become the arm reference.
    """
    recorded_seal = payload.get("seal_sha256")
    recomputed = seal_payload(payload)
    if recorded_seal != recomputed:
        raise ReplayValidationError(
            "R11 baseline replay seal does not replay over its own payload")
    if str(recorded_seal).lower() != expected_seal_sha256.lower():
        raise ReplayValidationError(
            "R11 baseline replay seal does not match the pinned value")
    if payload.get("schema") != R11_CONTROL_SCHEMA:
        raise ReplayValidationError(
            "R11 baseline replay schema is not the accepted control schema")
    if payload.get("task_id") != R11_BASELINE_TASK_ID:
        raise ReplayValidationError(
            "R11 baseline replay task id is not the accepted control task")
    if payload.get("status") != "complete":
        raise ReplayValidationError(
            "R11 baseline replay is not a completed control")
    gate = payload.get("control_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ReplayValidationError(
            "R11 baseline replay control gate did not pass")
    if gate.get("mismatches") or gate.get("r11_recorded_mismatches"):
        raise ReplayValidationError(
            "R11 baseline replay control gate retains mismatches")
    if payload.get("solver_calls_executed") != 1:
        raise ReplayValidationError(
            "R11 baseline replay is not exactly one solver call")
    treatments = payload.get("treatments")
    if (not isinstance(treatments, dict) or treatments.get("arms") != {}
            or treatments.get("arms_executed") != 0):
        raise ReplayValidationError(
            "R11 baseline replay unexpectedly carries treatment arms")
    inputs = payload.get("inputs")
    if (not isinstance(inputs, dict) or not isinstance(inputs.get("r10"), dict)
            or not isinstance(inputs.get("r11"), dict)):
        raise ReplayValidationError(
            "R11 baseline replay input bindings are missing")
    if str(inputs["r10"].get("file_sha256", "")).upper() != INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 baseline replay R10 input digest does not match the pin")
    if (str(inputs["r11"].get("file_sha256", "")).upper()
            != R11_INPUT_FILE_SHA256):
        raise ReplayValidationError(
            "R11 baseline replay R11 input digest does not match the pin")
    context_record = inputs["r11"].get("solve_context")
    if (not isinstance(context_record, dict)
            or str(context_record.get("record_sha256", "")).lower()
            != R11_CONTEXT_SHA256.lower()):
        raise ReplayValidationError(
            "R11 baseline replay solve-context digest does not match the pin")
    return payload


def load_r11_baseline(
    path: Path, *,
    expected_sha256: str = R11_BASELINE_FILE_SHA256,
    expected_seal_sha256: str = R11_BASELINE_SEAL_SHA256,
) -> Mapping[str, Any]:
    """Load and fail-closed-validate the accepted sealed R11 baseline replay."""
    path = Path(path)
    if not path.is_file():
        raise ReplayValidationError(
            f"pinned R11 baseline replay is absent: {path}")
    observed_sha256 = sha256_file(path).upper()
    if observed_sha256 != expected_sha256.upper():
        raise ReplayValidationError(
            "R11 baseline replay file SHA-256 does not match the pinned value")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ReplayValidationError("R11 baseline replay is not a JSON object")
    _validate_r11_baseline_payload(
        payload, expected_seal_sha256=expected_seal_sha256)
    return payload


def reconcile_arms_inputs(
    reconstruction: R10Reconstruction, r11_artifact: Mapping[str, Any],
    context: ExactSolveContext, baseline: Mapping[str, Any], *,
    r11_file_sha256: str,
) -> dict[str, Any]:
    """Reconcile the R10 capture, R11 capture and sealed R11 control first.

    Nothing is solved here.  The accepted baseline's own reconciliation, its
    context binding and its recorded control outcome must all replay exactly
    against freshly loaded R10/R11 artifacts before any arm may run.
    """
    fresh = reconcile_shared_inputs(reconstruction.artifact, r11_artifact)
    if baseline.get("reconciliation") != fresh:
        raise ReplayValidationError(
            "the sealed R11 control's reconciliation does not replay")
    inputs = baseline.get("inputs") or {}
    r10_binding = inputs.get("r10") or {}
    r11_binding = inputs.get("r11") or {}
    if (str(r10_binding.get("file_sha256", "")).upper()
            != reconstruction.validation["artifact_file_sha256"]):
        raise ReplayValidationError(
            "the sealed R11 control's R10 file binding does not replay")
    if (str(r10_binding.get("record_sha256", ""))
            != str(reconstruction.record.get("record_sha256"))):
        raise ReplayValidationError(
            "the sealed R11 control's R10 record binding does not replay")
    if str(r11_binding.get("file_sha256", "")).upper() != r11_file_sha256.upper():
        raise ReplayValidationError(
            "the sealed R11 control's R11 file binding does not replay")
    if r11_binding.get("solve_context") != context.to_report():
        raise ReplayValidationError(
            "the sealed R11 control's solve-context binding does not replay")
    baseline_validation = baseline.get("validation")
    if (not isinstance(baseline_validation, dict)
            or baseline_validation.get("artifact_file_sha256")
            != reconstruction.validation["artifact_file_sha256"]):
        raise ReplayValidationError(
            "the sealed R11 control's validation does not replay")
    control = baseline.get("control")
    if not isinstance(control, dict):
        raise ReplayValidationError(
            "the sealed R11 control outcome is missing")
    recorded = recorded_outcome(reconstruction.artifact)
    if baseline.get("r10_expected") != recorded:
        raise ReplayValidationError(
            "the sealed R11 control's R10 expectation does not replay")
    if control.get("exact_context") != context.to_report():
        raise ReplayValidationError(
            "the sealed R11 control's exact context does not replay")
    if control_mismatches(control, recorded, CONTROL_ABS_TOLERANCE):
        raise ReplayValidationError(
            "the sealed R11 control does not reproduce the recorded R10 "
            "outcome")
    return {
        "reconciled_keys": list(RECONCILED_KEYS),
        "fresh": fresh,
        "sealed_baseline": baseline.get("reconciliation"),
        "sealed_baseline_file_sha256": baseline.get("inputs", {})
        .get("r10", {}).get("file_sha256"),
        "sealed_baseline_seal_sha256": baseline.get("seal_sha256"),
        "sealed_control_reproduced_recorded_outcome": True,
        "sealed_context_matches_capture": True,
    }


def _axis_diagnostics(
    outcome: Mapping[str, Any], context: ExactSolveContext,
) -> dict[str, Any]:
    """Per-axis ``.90`` resultant / ``.15`` residual diagnostics (never a gate)."""
    min_resultant = float(context.min_resultant)
    integer_tolerance = float(context.integer_tolerance)
    observations = [int(value) for value in (outcome.get("observations") or [])]
    per_axis: dict[str, Any] = {}
    for axis in sorted(int(key) for key in outcome["resultants"]):
        resultant = float(outcome["resultants"][str(axis)])
        residual = float(outcome["residuals"][str(axis)])
        face_count = observations[axis] if axis < len(observations) else None
        per_axis[str(axis)] = {
            "resultant": resultant,
            "residual": residual,
            "resultant_at_or_above_min": resultant >= min_resultant,
            "residual_at_or_below_tolerance": residual <= integer_tolerance,
            "axis_gate_pass": (resultant >= min_resultant
                               and residual <= integer_tolerance),
            "face_observation_count": face_count,
            "active_edge_count": None if face_count is None else face_count // 2,
        }
    coverage = [axis for axis, value in enumerate(observations) if value > 0]
    return {
        "diagnostic_only": True,
        "min_resultant": min_resultant,
        "integer_tolerance": integer_tolerance,
        "per_axis": per_axis,
        "axis_coverage": coverage,
        "axis_coverage_complete": sorted(coverage) == [0, 1, 2],
        "active_match_count": int(outcome["active_match_count"]),
        "active_match_indices": list(outcome["active_match_indices"]),
    }


def _arm_deltas(
    outcome: Mapping[str, Any], control: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact deltas of one arm's observable outcome against the sealed control."""
    resultants: dict[str, float] = {}
    residuals: dict[str, float] = {}
    for axis in sorted(set(map(str, outcome["resultants"]))
                       | set(map(str, control["resultants"])), key=int):
        resultants[axis] = (float(outcome["resultants"].get(axis, 0.0))
                            - float(control["resultants"].get(axis, 0.0)))
        residuals[axis] = (float(outcome["residuals"].get(axis, 0.0))
                           - float(control["residuals"].get(axis, 0.0)))
    observations = [
        int(a) - int(b) for a, b in zip(
            outcome.get("observations") or [],
            control.get("observations") or [])]
    translations: dict[str, Any] = {}
    for frame, value in (control.get("translations") or {}).items():
        observed = (outcome.get("translations") or {}).get(frame)
        translations[frame] = (
            None if observed is None
            else [float(a) - float(b) for a, b in zip(observed, value)])
    return {
        "resultants": resultants,
        "residuals": residuals,
        "observations": observations,
        "translations": translations,
        "active_match_count": (int(outcome["active_match_count"])
                               - int(control["active_match_count"])),
        "assignment_changed": (list(outcome["assignment"])
                               != list(control["assignment"])),
    }


def _preregistered_check(
    name: str, outcome: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Compare one arm's axis-1 metrics with the preregistered prediction."""
    predicted = PREREGISTERED_AXIS1.get(name)
    if predicted is None:
        return None
    predicted_resultant, predicted_residual = predicted
    if "1" not in outcome["resultants"]:
        return {
            "axis": 1,
            "predicted_resultant": predicted_resultant,
            "predicted_residual": predicted_residual,
            "measured_resultant": None,
            "measured_residual": None,
            "resultant_delta": None,
            "residual_delta": None,
            "matched": False,
            "tolerance_absolute": PREREGISTERED_ABS_TOLERANCE,
            "note": "axis 1 is absent from the arm outcome",
        }
    measured_resultant = float(outcome["resultants"]["1"])
    measured_residual = float(outcome["residuals"]["1"])
    resultant_matched = math.isclose(
        measured_resultant, predicted_resultant, rel_tol=0.0,
        abs_tol=PREREGISTERED_ABS_TOLERANCE)
    residual_matched = (
        predicted_residual is None
        or math.isclose(measured_residual, predicted_residual, rel_tol=0.0,
                        abs_tol=PREREGISTERED_ABS_TOLERANCE))
    return {
        "axis": 1,
        "predicted_resultant": predicted_resultant,
        "predicted_residual": predicted_residual,
        "measured_resultant": measured_resultant,
        "measured_residual": measured_residual,
        "resultant_delta": measured_resultant - predicted_resultant,
        "residual_delta": (None if predicted_residual is None
                           else measured_residual - predicted_residual),
        "matched": bool(resultant_matched and residual_matched),
        "tolerance_absolute": PREREGISTERED_ABS_TOLERANCE,
    }


def build_context_arms(
    record: Mapping[str, Any], universe: ReadingUniverse,
    baseline_matches: Sequence[Any], context: ExactSolveContext,
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Run exactly one context-pinned solver call per preregistered arm.

    Only the frozen C1-C5 one-for-one substitutions are built.  Each substituted
    row must still be a dropped ``same_axis_mismatch`` candidate in its declared
    group and must replace a survivor of that same group; otherwise nothing is
    solved and the replay fails closed.  The baseline is never rerun: ``control``
    is the accepted sealed control outcome consumed for the deltas.
    """
    claims = _group_claims(record)
    frames = tuple(universe.frames)
    assignment = list(control.get("assignment") or [])
    if len(assignment) != len(frames):
        raise ReplayValidationError(
            "the sealed control assignment is not frame-aligned")
    quadrants = {frame: int(value) for frame, value in zip(frames, assignment)}
    arms: dict[str, Any] = {}
    for name, specification in SUBSTITUTIONS.items():
        replacements: dict[int, Any] = {}
        provenance: list[dict[str, Any]] = []
        for flat_index, match_index, group_index in specification:
            if match_index not in claims.get(group_index, ()):
                raise ReplayValidationError(
                    f"{name} replacement index is not a survivor of its group")
            match, row = candidate_match(record, universe, flat_index)
            if int(row[0]) != group_index:
                raise ReplayValidationError(
                    f"{name} candidate row is not in the declared group")
            if (row[17] != "dropped"
                    or row[18] != plane_graph.SAME_AXIS_DISPOSITION_DROPPED):
                raise ReplayValidationError(
                    f"{name} candidate row is not a dropped same-axis mismatch")
            # The group is world-axis 1 only if both substituted endpoints map
            # to one world axis under the sealed control assignment; this is
            # recorded (never assumed) so the axis-0/2 invariance is evidenced.
            left_world, _left_sign = plane_graph.axis_transform(
                match.left_axis, quadrants[match.left])
            right_world, _right_sign = plane_graph.axis_transform(
                match.right_axis, quadrants[match.right])
            replacements[match_index] = match
            provenance.append({
                "candidate_flat_row": flat_index,
                "candidate_group": group_index,
                "replaces_match_index": match_index,
                "candidate_disposition": row[17],
                "candidate_reason": row[18],
                "candidate_left_axis": int(match.left_axis),
                "candidate_right_axis": int(match.right_axis),
                "candidate_world_axis": int(left_world),
                "candidate_right_world_axis": int(right_world),
                "candidate_same_axis_under_control": left_world == right_world,
                "candidate_active_under_control": (
                    (quadrants[match.right] - quadrants[match.left]) % 4
                    == match.quadrant_of_right),
                "candidate_score": float(row[8]),
                "candidate_delta": float(row[16]),
            })
        treatment = substitute(baseline_matches, replacements)
        outcome = run_arm(universe.readings, treatment, universe.frames,
                          target=None, context=context)
        arms[name] = {
            "provenance": provenance,
            "preregistered_axis1": PREREGISTERED_AXIS1.get(name),
            "preregistered_check": _preregistered_check(name, outcome),
            "overall_geometry_label": outcome["reason"],
            "outcome": outcome,
            "diagnostics": _axis_diagnostics(outcome, context),
            "deltas_vs_sealed_control": _arm_deltas(outcome, control),
        }
    return arms


def replay_r11_arms(
    r10_path: Path, r11_path: Path, baseline_path: Path, *,
    expected_r10_sha256: str = INPUT_FILE_SHA256,
    expected_r11_sha256: str = R11_INPUT_FILE_SHA256,
    expected_context_sha256: str = R11_CONTEXT_SHA256,
    expected_baseline_sha256: str = R11_BASELINE_FILE_SHA256,
    expected_baseline_seal_sha256: str = R11_BASELINE_SEAL_SHA256,
    tolerance: float = CONTROL_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Run the preregistered C1-C5 arms once each against the sealed control.

    Ordering is deliberate: the accepted sealed baseline, the R10 capture and the
    R11 context capture are all pinned and reconciled *before* any treatment
    call, and the baseline is never rerun.  Exactly five context-pinned solver
    calls follow (one per arm); no other arm is built or solved.
    """
    baseline = load_r11_baseline(
        baseline_path, expected_sha256=expected_baseline_sha256,
        expected_seal_sha256=expected_baseline_seal_sha256)
    baseline_file_sha256 = sha256_file(Path(baseline_path)).upper()

    r11_path = Path(r11_path)
    if not r11_path.is_file():
        raise ReplayValidationError(f"pinned R11 artifact is absent: {r11_path}")
    observed_r11_sha256 = sha256_file(r11_path).upper()
    if observed_r11_sha256 != expected_r11_sha256.upper():
        raise ReplayValidationError(
            "R11 artifact file SHA-256 does not match the pinned value")
    with open(r11_path, "r", encoding="utf-8") as handle:
        r11_artifact = json.load(handle)
    if r11_artifact.get("schema") != INPUT_SCHEMA:
        raise ReplayValidationError("R11 artifact schema is not the stage schema")

    reconstruction = load_r10_reconstruction(
        r10_path, expected_sha256=expected_r10_sha256)
    context = exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames,
        expected_context_sha256=expected_context_sha256)

    reconciliation = reconcile_arms_inputs(
        reconstruction, r11_artifact, context, baseline,
        r11_file_sha256=observed_r11_sha256)
    control = baseline["control"]

    arms = build_context_arms(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        context, control)

    preregistered = {name: arms[name]["preregistered_check"] for name in arms}
    deviations = [name for name, check in preregistered.items()
                  if check is not None and not check["matched"]]
    validation = dict(reconstruction.validation)
    validation["r11_artifact_file_sha256"] = observed_r11_sha256
    validation["r11_context_replay"] = "pass"
    validation["r11_context_frame_binding"] = "pass"
    validation["r10_r11_reconciliation"] = "pass"
    validation["sealed_baseline_reconciliation_replays"] = "pass"
    validation["sealed_control_reproduced_recorded_outcome"] = "pass"
    validation["baseline_rerun"] = False

    payload: dict[str, Any] = {
        "schema": ARMS_SCHEMA,
        "task_id": ARMS_TASK_ID,
        "status": "complete",
        "no_truth_disclaimer": NO_TRUTH_DISCLAIMER,
        "inputs": {
            "r10": {
                "path": str(r10_path),
                "file_sha256": validation["artifact_file_sha256"],
                "expected_file_sha256": expected_r10_sha256.upper(),
                "artifact_sha256": reconstruction.artifact.get(
                    "artifact_sha256"),
                "observation_sha256": reconstruction.artifact[
                    "solver_failure_observation"].get("observation_sha256"),
                "record_sha256": reconstruction.record.get("record_sha256"),
                "failed_assignment_sha256": reconstruction.record.get(
                    "failed_assignment_sha256"),
                "input_match_identity_sha256": reconstruction.record.get(
                    "input_match_identity_sha256"),
            },
            "r11": {
                "path": str(r11_path),
                "file_sha256": observed_r11_sha256,
                "expected_file_sha256": expected_r11_sha256.upper(),
                "artifact_sha256": r11_artifact.get("artifact_sha256"),
                "entry_sha256": r11_artifact.get("entry_sha256"),
                "observation_sha256": r11_artifact[
                    "solver_failure_observation"].get("observation_sha256"),
                "solve_context": context.to_report(),
            },
            "r11_baseline": {
                "path": str(baseline_path),
                "file_sha256": baseline_file_sha256,
                "expected_file_sha256": expected_baseline_sha256.upper(),
                "seal_sha256": baseline.get("seal_sha256"),
                "expected_seal_sha256": expected_baseline_seal_sha256.lower(),
            },
        },
        "reconciliation": reconciliation,
        "validation": validation,
        "sealed_control": {
            "file_sha256": baseline_file_sha256,
            "seal_sha256": baseline.get("seal_sha256"),
            "reason": control.get("reason"),
            "assignment": control.get("assignment"),
            "accepted": control.get("accepted"),
            "resultants": control.get("resultants"),
            "residuals": control.get("residuals"),
            "observations": control.get("observations"),
            "translations": control.get("translations"),
            "active_match_count": control.get("active_match_count"),
            "active_match_indices": control.get("active_match_indices"),
            "duplicate_b": control.get("duplicate_b"),
            "diagnostics": _axis_diagnostics(control, context),
        },
        "arms": arms,
        "arm_order": list(SUBSTITUTIONS),
        "preregistered_check": {
            "tolerance_absolute": PREREGISTERED_ABS_TOLERANCE,
            "arms": preregistered,
            "deviations": deviations,
            "all_matched": not deviations,
        },
        "solver_calls_executed": len(arms),
        "solver_calls_expected": len(SUBSTITUTIONS),
        "baseline_rerun": False,
        "notes": [
            "The accepted R11 baseline replay, the R10 capture and the R11 "
            "context capture were pinned and reconciled before any treatment; "
            "the accepted baseline control was consumed, never rerun.",
            "Exactly the preregistered C1-C5 one-for-one substitutions were "
            "solved once each with the identical captured thresholds "
            "(.15/.90/.02) and ordered quadrant constraints; no other arm, "
            "retune or threshold change was made.",
            "The .90 resultant / .15 residual gates are per-axis diagnostics "
            "only and were not relaxed.  A passing axis is feasibility "
            "evidence for this single fixture, not candidate truth or product "
            "accuracy, and the overall geometry keeps the unchanged solver "
            "reason label recorded above.",
        ],
    }
    payload["seal_sha256"] = seal_payload(payload)
    return payload


def _match_identity_key(match: Any) -> tuple[Any, ...]:
    """The active-edge identity key the solver's ``duplicate_b`` flag uses."""
    return (match.left, match.left_plane, match.right, match.right_plane,
            match.quadrant_of_right)


def _world_axis_for(
    local_axis: int, frame: str, quadrants: Mapping[str, int],
) -> int:
    """Map one endpoint's local axis to its world axis under an assignment."""
    world, _sign = plane_graph.axis_transform(
        int(local_axis), int(quadrants[frame]))
    return int(world)


def _assert_no_active_duplicates(
    matches: Sequence[Any], frames: Sequence[str],
    quadrants: Mapping[str, int],
) -> None:
    """Refuse a substitution set that leaves two identical active edge keys."""
    if _active_keys_have_duplicates(matches, frames, quadrants):
        raise ReplayValidationError(
            "a combined arm leaves a duplicate active edge")


def _active_keys_have_duplicates(
    matches: Sequence[Any], frames: Sequence[str],
    quadrants: Mapping[str, int],
) -> bool:
    """Whether two active matches share the solver's edge-identity key."""
    ordered = [int(quadrants[frame]) for frame in frames]
    active = _active_indices(matches, frames, ordered)
    keys = [_match_identity_key(matches[index]) for index in active]
    return len(keys) != len(set(keys))


def _validate_r11_c1_c5_payload(
    payload: Mapping[str, Any], *,
    expected_seal_sha256: str = R11_C1_C5_SEAL_SHA256,
) -> Mapping[str, Any]:
    """Fail-closed checks on the already-loaded sealed C1-C5 arm replay.

    The artifact is consumed only as the source of the sealed C3 reference, so a
    file that does not replay its own seal, is not the accepted C1-C5 task, does
    not carry exactly the frozen five arms, or does not bind the pinned R10/R11
    context and accepted baseline can never define the arm reference.
    """
    recorded_seal = payload.get("seal_sha256")
    recomputed = seal_payload(payload)
    if recorded_seal != recomputed:
        raise ReplayValidationError(
            "R11 C1-C5 replay seal does not replay over its own payload")
    if str(recorded_seal).lower() != expected_seal_sha256.lower():
        raise ReplayValidationError(
            "R11 C1-C5 replay seal does not match the pinned value")
    if payload.get("schema") != ARMS_SCHEMA:
        raise ReplayValidationError(
            "R11 C1-C5 replay schema is not the accepted arms schema")
    if payload.get("task_id") != R11_C1_C5_TASK_ID:
        raise ReplayValidationError(
            "R11 C1-C5 replay task id is not the accepted arms task")
    if payload.get("status") != "complete":
        raise ReplayValidationError(
            "R11 C1-C5 replay is not a completed arms run")
    if payload.get("solver_calls_executed") != 5:
        raise ReplayValidationError(
            "R11 C1-C5 replay is not exactly five solver calls")
    if payload.get("solver_calls_expected") != 5:
        raise ReplayValidationError(
            "R11 C1-C5 replay does not expect five solver calls")
    if payload.get("baseline_rerun") is not False:
        raise ReplayValidationError(
            "R11 C1-C5 replay reports a baseline rerun")
    arm_order = payload.get("arm_order")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or arm_order != list(SUBSTITUTIONS):
        raise ReplayValidationError(
            "R11 C1-C5 replay does not carry the frozen C1-C5 arm order")
    if list(arms) != list(SUBSTITUTIONS):
        raise ReplayValidationError(
            "R11 C1-C5 replay arms are not the frozen C1-C5 set")
    for name, arm in arms.items():
        outcome = arm.get("outcome") if isinstance(arm, dict) else None
        if not isinstance(outcome, dict) or outcome.get("mode") != "exact_context":
            raise ReplayValidationError(
                f"R11 C1-C5 replay arm {name} is not an exact-context outcome")
        if outcome.get("pinned") is not True:
            raise ReplayValidationError(
                f"R11 C1-C5 replay arm {name} is not context-pinned")
        exact = outcome.get("exact_context")
        if (not isinstance(exact, dict)
                or str(exact.get("record_sha256", "")).lower()
                != R11_CONTEXT_SHA256.lower()):
            raise ReplayValidationError(
                f"R11 C1-C5 replay arm {name} does not bind the pinned context")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise ReplayValidationError("R11 C1-C5 replay validation is missing")
    if str(validation.get("artifact_file_sha256", "")).upper() != INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 C1-C5 replay R10 file digest does not match the pin")
    if (str(validation.get("r11_artifact_file_sha256", "")).upper()
            != R11_INPUT_FILE_SHA256):
        raise ReplayValidationError(
            "R11 C1-C5 replay R11 file digest does not match the pin")
    if validation.get("baseline_rerun") is not False:
        raise ReplayValidationError(
            "R11 C1-C5 replay validation reports a baseline rerun")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ReplayValidationError("R11 C1-C5 replay inputs are missing")
    r10_binding = inputs.get("r10")
    r11_binding = inputs.get("r11")
    baseline_binding = inputs.get("r11_baseline")
    if not all(isinstance(item, dict)
               for item in (r10_binding, r11_binding, baseline_binding)):
        raise ReplayValidationError(
            "R11 C1-C5 replay input bindings are incomplete")
    if str(r10_binding.get("file_sha256", "")).upper() != INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 C1-C5 replay R10 input digest does not match the pin")
    if str(r11_binding.get("file_sha256", "")).upper() != R11_INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 C1-C5 replay R11 input digest does not match the pin")
    context_record = r11_binding.get("solve_context")
    if (not isinstance(context_record, dict)
            or str(context_record.get("record_sha256", "")).lower()
            != R11_CONTEXT_SHA256.lower()):
        raise ReplayValidationError(
            "R11 C1-C5 replay solve-context digest does not match the pin")
    if (str(baseline_binding.get("file_sha256", "")).upper()
            != R11_BASELINE_FILE_SHA256):
        raise ReplayValidationError(
            "R11 C1-C5 replay baseline file digest does not match the pin")
    if (str(baseline_binding.get("seal_sha256", "")).lower()
            != R11_BASELINE_SEAL_SHA256.lower()):
        raise ReplayValidationError(
            "R11 C1-C5 replay baseline seal does not match the pin")
    sealed_control = payload.get("sealed_control")
    if not isinstance(sealed_control, dict):
        raise ReplayValidationError(
            "R11 C1-C5 replay sealed control is missing")
    if (str(sealed_control.get("file_sha256", "")).upper()
            != R11_BASELINE_FILE_SHA256):
        raise ReplayValidationError(
            "R11 C1-C5 replay sealed control file digest does not match the pin")
    if (str(sealed_control.get("seal_sha256", "")).lower()
            != R11_BASELINE_SEAL_SHA256.lower()):
        raise ReplayValidationError(
            "R11 C1-C5 replay sealed control seal does not match the pin")
    return payload


def load_r11_c1_c5(
    path: Path, *,
    expected_sha256: str = R11_C1_C5_FILE_SHA256,
    expected_seal_sha256: str = R11_C1_C5_SEAL_SHA256,
) -> Mapping[str, Any]:
    """Load and fail-closed-validate the accepted sealed C1-C5 arm replay."""
    path = Path(path)
    if not path.is_file():
        raise ReplayValidationError(
            f"pinned R11 C1-C5 replay is absent: {path}")
    observed_sha256 = sha256_file(path).upper()
    if observed_sha256 != expected_sha256.upper():
        raise ReplayValidationError(
            "R11 C1-C5 replay file SHA-256 does not match the pinned value")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ReplayValidationError("R11 C1-C5 replay is not a JSON object")
    _validate_r11_c1_c5_payload(
        payload, expected_seal_sha256=expected_seal_sha256)
    return payload


def _validate_r11_axes02_payload(
    payload: Mapping[str, Any], *,
    expected_seal_sha256: str = R11_AXES02_SEAL_SHA256,
) -> Mapping[str, Any]:
    """Fail-closed checks on the already-loaded accepted axes-0/2 replay.

    The artifact is consumed as a sealed cross-reference for the exhaustive
    replay's reconciliation and sealed C3 outcome, so a file that does not
    replay its own seal, is not the accepted axes-0/2 task, does not carry the
    frozen eight arms bound to the pinned context, or does not bind the pinned
    R10/R11/C1-C5 artifacts can never define the reference.
    """
    recorded_seal = payload.get("seal_sha256")
    recomputed = seal_payload(payload)
    if recorded_seal != recomputed:
        raise ReplayValidationError(
            "R11 axes-0/2 replay seal does not replay over its own payload")
    if str(recorded_seal).lower() != expected_seal_sha256.lower():
        raise ReplayValidationError(
            "R11 axes-0/2 replay seal does not match the pinned value")
    if payload.get("schema") != AXES02_C3_SCHEMA:
        raise ReplayValidationError(
            "R11 axes-0/2 replay schema is not the accepted combined schema")
    if payload.get("task_id") != R11_AXES02_TASK_ID:
        raise ReplayValidationError(
            "R11 axes-0/2 replay task id is not the accepted combined task")
    if payload.get("status") != "complete":
        raise ReplayValidationError(
            "R11 axes-0/2 replay is not a completed arms run")
    if payload.get("solver_calls_executed") != 8:
        raise ReplayValidationError(
            "R11 axes-0/2 replay is not exactly eight solver calls")
    if payload.get("solver_calls_expected") != 8:
        raise ReplayValidationError(
            "R11 axes-0/2 replay does not expect eight solver calls")
    if payload.get("baseline_rerun") is not False:
        raise ReplayValidationError(
            "R11 axes-0/2 replay reports a baseline rerun")
    if payload.get("c3_rerun") is not False:
        raise ReplayValidationError(
            "R11 axes-0/2 replay reports a C3 rerun")
    arm_order = payload.get("arm_order")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or arm_order != list(AXES02_ARMS):
        raise ReplayValidationError(
            "R11 axes-0/2 replay does not carry the frozen eight-arm order")
    if set(arms) != set(AXES02_ARMS):
        raise ReplayValidationError(
            "R11 axes-0/2 replay arms are not the frozen eight-arm set")
    for name, arm in arms.items():
        outcome = arm.get("outcome") if isinstance(arm, dict) else None
        if not isinstance(outcome, dict) or outcome.get("mode") != "exact_context":
            raise ReplayValidationError(
                f"R11 axes-0/2 replay arm {name} is not an exact-context outcome")
        if outcome.get("pinned") is not True:
            raise ReplayValidationError(
                f"R11 axes-0/2 replay arm {name} is not context-pinned")
        exact = outcome.get("exact_context")
        if (not isinstance(exact, dict)
                or str(exact.get("record_sha256", "")).lower()
                != R11_CONTEXT_SHA256.lower()):
            raise ReplayValidationError(
                f"R11 axes-0/2 replay arm {name} does not bind the pinned context")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise ReplayValidationError("R11 axes-0/2 replay validation is missing")
    if str(validation.get("artifact_file_sha256", "")).upper() != INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 axes-0/2 replay R10 file digest does not match the pin")
    if (str(validation.get("r11_artifact_file_sha256", "")).upper()
            != R11_INPUT_FILE_SHA256):
        raise ReplayValidationError(
            "R11 axes-0/2 replay R11 file digest does not match the pin")
    if validation.get("baseline_rerun") is not False:
        raise ReplayValidationError(
            "R11 axes-0/2 replay validation reports a baseline rerun")
    if validation.get("c3_rerun") is not False:
        raise ReplayValidationError(
            "R11 axes-0/2 replay validation reports a C3 rerun")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ReplayValidationError("R11 axes-0/2 replay inputs are missing")
    r10_binding = inputs.get("r10")
    r11_binding = inputs.get("r11")
    baseline_binding = inputs.get("r11_baseline")
    c1_c5_binding = inputs.get("r11_c1_c5")
    if not all(isinstance(item, dict) for item in (
            r10_binding, r11_binding, baseline_binding, c1_c5_binding)):
        raise ReplayValidationError(
            "R11 axes-0/2 replay input bindings are incomplete")
    if str(r10_binding.get("file_sha256", "")).upper() != INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 axes-0/2 replay R10 input digest does not match the pin")
    if str(r11_binding.get("file_sha256", "")).upper() != R11_INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 axes-0/2 replay R11 input digest does not match the pin")
    context_record = r11_binding.get("solve_context")
    if (not isinstance(context_record, dict)
            or str(context_record.get("record_sha256", "")).lower()
            != R11_CONTEXT_SHA256.lower()):
        raise ReplayValidationError(
            "R11 axes-0/2 replay solve-context digest does not match the pin")
    if (str(baseline_binding.get("file_sha256", "")).upper()
            != R11_BASELINE_FILE_SHA256):
        raise ReplayValidationError(
            "R11 axes-0/2 replay baseline file digest does not match the pin")
    if (str(c1_c5_binding.get("file_sha256", "")).upper()
            != R11_C1_C5_FILE_SHA256):
        raise ReplayValidationError(
            "R11 axes-0/2 replay C1-C5 file digest does not match the pin")
    if (str(c1_c5_binding.get("seal_sha256", "")).lower()
            != R11_C1_C5_SEAL_SHA256.lower()):
        raise ReplayValidationError(
            "R11 axes-0/2 replay C1-C5 seal does not match the pin")
    sealed_c3 = payload.get("sealed_c3")
    if not isinstance(sealed_c3, dict):
        raise ReplayValidationError(
            "R11 axes-0/2 replay sealed C3 reference is missing")
    if (str(sealed_c3.get("file_sha256", "")).upper()
            != R11_C1_C5_FILE_SHA256):
        raise ReplayValidationError(
            "R11 axes-0/2 replay sealed C3 file digest does not match the pin")
    if (str(sealed_c3.get("seal_sha256", "")).lower()
            != R11_C1_C5_SEAL_SHA256.lower()):
        raise ReplayValidationError(
            "R11 axes-0/2 replay sealed C3 seal does not match the pin")
    provenance = sealed_c3.get("provenance")
    if not isinstance(provenance, list):
        raise ReplayValidationError(
            "R11 axes-0/2 replay sealed C3 provenance is missing")
    frozen = sorted(
        (int(entry["candidate_flat_row"]), int(entry["candidate_group"]),
         int(entry["replaces_match_index"]))
        for entry in provenance)
    expected = sorted((int(row), int(group), int(target))
                      for row, target, group, _axis in C3_SUBSTITUTIONS)
    if frozen != expected:
        raise ReplayValidationError(
            "R11 axes-0/2 replay sealed C3 is not the frozen substitution")
    return payload


def load_r11_axes02(
    path: Path, *,
    expected_sha256: str = R11_AXES02_FILE_SHA256,
    expected_seal_sha256: str = R11_AXES02_SEAL_SHA256,
) -> Mapping[str, Any]:
    """Load and fail-closed-validate the accepted sealed axes-0/2 replay."""
    path = Path(path)
    if not path.is_file():
        raise ReplayValidationError(
            f"pinned R11 axes-0/2 replay is absent: {path}")
    observed_sha256 = sha256_file(path).upper()
    if observed_sha256 != expected_sha256.upper():
        raise ReplayValidationError(
            "R11 axes-0/2 replay file SHA-256 does not match the pinned value")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ReplayValidationError(
            "R11 axes-0/2 replay is not a JSON object")
    _validate_r11_axes02_payload(
        payload, expected_seal_sha256=expected_seal_sha256)
    return payload


def sealed_c3_reference(c1_c5_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the sealed C3 arm (the accepted reference the arms measure)."""
    arms = c1_c5_payload.get("arms")
    c3 = arms.get("C3") if isinstance(arms, dict) else None
    if not isinstance(c3, dict):
        raise ReplayValidationError("the sealed C1-C5 replay carries no C3 arm")
    outcome = c3.get("outcome")
    if not isinstance(outcome, dict):
        raise ReplayValidationError("the sealed C3 outcome is missing")
    return {
        "arm": "C3",
        "outcome": outcome,
        "diagnostics": c3.get("diagnostics"),
        "provenance": c3.get("provenance"),
        "preregistered_check": c3.get("preregistered_check"),
        "overall_geometry_label": c3.get("overall_geometry_label"),
    }


def plan_combined_arm(
    record: Mapping[str, Any], universe: ReadingUniverse,
    baseline_matches: Sequence[Any], quadrants: Mapping[str, int],
    specification: Sequence[tuple[int, int, int, int]], *, label: str,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Validate one arm's frozen one-for-one substitutions (solver-free).

    Every constraint the Lead froze is checked here: the candidate is a dropped
    ``same_axis_mismatch`` row of its declared group, the replaced match index is
    a survivor of that group, the candidate replays the survivor's exact edge
    identity (otherwise the remaining survivor's edge is duplicated), and both
    endpoints map to the declared world axis and are active under the sealed
    assignment.
    """
    claims = _group_claims(record)
    baseline_keys = {index: _match_identity_key(match)
                     for index, match in enumerate(baseline_matches)}
    replacements: dict[int, Any] = {}
    provenance: list[dict[str, Any]] = []
    for flat_index, match_index, group_index, world_axis in specification:
        if match_index not in claims.get(group_index, ()):
            raise ReplayValidationError(
                f"{label} replacement index is not a survivor of its group")
        match, row = candidate_match(record, universe, flat_index)
        if int(row[0]) != group_index:
            raise ReplayValidationError(
                f"{label} candidate row is not in the declared group")
        if (row[17] != "dropped"
                or row[18] != plane_graph.SAME_AXIS_DISPOSITION_DROPPED):
            raise ReplayValidationError(
                f"{label} candidate row is not a dropped same-axis mismatch")
        if _match_identity_key(match) != baseline_keys[match_index]:
            raise ReplayValidationError(
                f"{label} candidate row does not replay its declared survivor "
                "edge identity, which would leave a duplicate active edge")
        left_world = _world_axis_for(match.left_axis, match.left, quadrants)
        right_world = _world_axis_for(match.right_axis, match.right, quadrants)
        same_axis = left_world == right_world
        if not same_axis or left_world != int(world_axis):
            raise ReplayValidationError(
                f"{label} candidate row does not map both endpoints to the "
                "declared world axis")
        active = ((int(quadrants[match.right]) - int(quadrants[match.left])) % 4
                  == int(match.quadrant_of_right))
        if not active:
            raise ReplayValidationError(
                f"{label} candidate row is not active under the sealed "
                "assignment")
        if match_index in replacements:
            raise ReplayValidationError(
                f"{label} arm replaces one survivor more than once")
        replacements[match_index] = match
        provenance.append({
            "candidate_flat_row": int(flat_index),
            "candidate_group": int(group_index),
            "replaces_match_index": int(match_index),
            "candidate_disposition": row[17],
            "candidate_reason": row[18],
            "candidate_left_axis": int(match.left_axis),
            "candidate_right_axis": int(match.right_axis),
            "candidate_world_axis": int(left_world),
            "candidate_right_world_axis": int(right_world),
            "candidate_same_axis_under_control": bool(same_axis),
            "candidate_active_under_control": bool(active),
            "candidate_key_replays_survivor": True,
            "declared_world_axis": int(world_axis),
            "candidate_score": float(row[8]),
            "candidate_delta": float(row[16]),
        })
    return replacements, provenance


def build_axes02_c3_arms(
    readings: Sequence[Any], matches: Sequence[Any], frames: Sequence[str],
    context: ExactSolveContext, plans: Mapping[str, Any],
    c3_replacements: Mapping[int, Any], c3_outcome: Mapping[str, Any],
    quadrants: Mapping[str, int],
) -> dict[str, Any]:
    """Run exactly one context-pinned solver call per frozen combined arm.

    ``plans`` and the C3 replacements were already validated before this point,
    so no arm is solved until every row/group/survivor/duplicate constraint of
    all eight arms has held.  The sealed C3 outcome is consumed, never rebuilt.
    """
    arms: dict[str, Any] = {}
    for name, specification in AXES02_ARMS.items():
        plan = plans[name]
        combined = dict(c3_replacements)
        combined.update(plan["replacements"])
        treatment = substitute(matches, combined)
        _assert_no_active_duplicates(treatment, frames, quadrants)
        outcome = run_arm(readings, treatment, frames, target=None,
                          context=context)
        arms[name] = {
            "frozen_axes02_rows": [int(entry[0]) for entry in specification],
            "frozen_axes02_pair": [int(specification[0][0]),
                                   int(specification[1][0])],
            "includes_c3": True,
            "c3_rows": [int(entry[0]) for entry in C3_SUBSTITUTIONS],
            "provenance": plan["provenance"],
            "overall_geometry_label": outcome["reason"],
            "outcome": outcome,
            "diagnostics": _axis_diagnostics(outcome, context),
            "deltas_vs_sealed_c3": _arm_deltas(outcome, c3_outcome),
        }
    return arms


def replay_r11_axes02_c3(
    r10_path: Path, r11_path: Path, baseline_path: Path, c1_c5_path: Path, *,
    expected_r10_sha256: str = INPUT_FILE_SHA256,
    expected_r11_sha256: str = R11_INPUT_FILE_SHA256,
    expected_context_sha256: str = R11_CONTEXT_SHA256,
    expected_baseline_sha256: str = R11_BASELINE_FILE_SHA256,
    expected_baseline_seal_sha256: str = R11_BASELINE_SEAL_SHA256,
    expected_c1_c5_sha256: str = R11_C1_C5_FILE_SHA256,
    expected_c1_c5_seal_sha256: str = R11_C1_C5_SEAL_SHA256,
    tolerance: float = CONTROL_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Run the eight frozen axes-0/2 + C3 arms once each against sealed C3.

    Ordering is deliberate.  The accepted baseline, the sealed C1-C5 replay, the
    R10 capture and the R11 context capture are all pinned, reconciled and
    cross-bound *before* anything else, then every frozen substitution of all
    eight arms is validated (row, group, survivor, duplicate and world-axis
    constraints) before the first of exactly eight context-pinned solver calls.
    C1-C5 is consumed, so neither the baseline control nor the C3 arm is rerun.
    """
    baseline = load_r11_baseline(
        baseline_path, expected_sha256=expected_baseline_sha256,
        expected_seal_sha256=expected_baseline_seal_sha256)
    baseline_file_sha256 = sha256_file(Path(baseline_path)).upper()

    c1_c5 = load_r11_c1_c5(
        c1_c5_path, expected_sha256=expected_c1_c5_sha256,
        expected_seal_sha256=expected_c1_c5_seal_sha256)
    c1_c5_file_sha256 = sha256_file(Path(c1_c5_path)).upper()

    r11_path = Path(r11_path)
    if not r11_path.is_file():
        raise ReplayValidationError(f"pinned R11 artifact is absent: {r11_path}")
    observed_r11_sha256 = sha256_file(r11_path).upper()
    if observed_r11_sha256 != expected_r11_sha256.upper():
        raise ReplayValidationError(
            "R11 artifact file SHA-256 does not match the pinned value")
    with open(r11_path, "r", encoding="utf-8") as handle:
        r11_artifact = json.load(handle)
    if r11_artifact.get("schema") != INPUT_SCHEMA:
        raise ReplayValidationError("R11 artifact schema is not the stage schema")

    reconstruction = load_r10_reconstruction(
        r10_path, expected_sha256=expected_r10_sha256)
    context = exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames,
        expected_context_sha256=expected_context_sha256)

    reconciliation = reconcile_arms_inputs(
        reconstruction, r11_artifact, context, baseline,
        r11_file_sha256=observed_r11_sha256)

    # Consume-only cross-binds of the sealed C1-C5 replay.  Nothing below
    # rebuilds or reruns C1-C5 (and never the sealed C3 arm).
    if c1_c5.get("reconciliation") != reconciliation:
        raise ReplayValidationError(
            "the sealed C1-C5 replay's reconciliation does not replay")
    if (c1_c5["validation"].get("artifact_file_sha256")
            != reconstruction.validation["artifact_file_sha256"]):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's R10 file binding does not replay")
    if (str(c1_c5["inputs"]["r10"].get("record_sha256", ""))
            != str(reconstruction.record.get("record_sha256"))):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's R10 record binding does not replay")
    if c1_c5["inputs"]["r11"].get("solve_context") != context.to_report():
        raise ReplayValidationError(
            "the sealed C1-C5 replay's solve-context binding does not replay")
    if (str(c1_c5["inputs"]["r11_baseline"].get("file_sha256", "")).upper()
            != baseline_file_sha256):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's baseline file binding does not replay")
    if (str(c1_c5["sealed_control"].get("file_sha256", "")).upper()
            != baseline_file_sha256):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's sealed control binding does not replay")
    sealed_c3 = sealed_c3_reference(c1_c5)
    frozen_c3 = sorted(
        (int(entry["candidate_flat_row"]), int(entry["candidate_group"]),
         int(entry["replaces_match_index"]))
        for entry in sealed_c3["provenance"])
    expected_c3 = sorted((int(row), int(group), int(target))
                         for row, target, group, _axis in C3_SUBSTITUTIONS)
    if frozen_c3 != expected_c3:
        raise ReplayValidationError(
            "the sealed C3 arm is not the frozen C3 substitution")

    frames = tuple(reconstruction.universe.frames)
    outcome_c3 = sealed_c3["outcome"]
    assignment = list(outcome_c3.get("assignment") or [])
    if len(assignment) != len(frames):
        raise ReplayValidationError(
            "the sealed C3 outcome assignment is not frame-aligned")
    quadrants = {frame: int(value) for frame, value in zip(frames, assignment)}

    # Every frozen substitution of every arm is validated before any solver call.
    c3_replacements, _c3_provenance = plan_combined_arm(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        quadrants, C3_SUBSTITUTIONS, label="C3")
    plans: dict[str, Any] = {}
    for name, specification in AXES02_ARMS.items():
        replacements, provenance = plan_combined_arm(
            reconstruction.record, reconstruction.universe,
            reconstruction.matches, quadrants, specification, label=name)
        plans[name] = {"replacements": replacements, "provenance": provenance}

    arms = build_axes02_c3_arms(
        reconstruction.universe.readings, reconstruction.matches, frames,
        context, plans, c3_replacements, outcome_c3, quadrants)

    validation = dict(reconstruction.validation)
    validation["r11_artifact_file_sha256"] = observed_r11_sha256
    validation["r11_context_replay"] = "pass"
    validation["r11_context_frame_binding"] = "pass"
    validation["r10_r11_reconciliation"] = "pass"
    validation["sealed_baseline_reconciliation_replays"] = "pass"
    validation["sealed_control_reproduced_recorded_outcome"] = "pass"
    validation["sealed_c1_c5_reconciliation_replays"] = "pass"
    validation["sealed_c3_provenance_replays"] = "pass"
    validation["frozen_substitutions_validated_before_any_arm"] = "pass"
    validation["baseline_rerun"] = False
    validation["c3_rerun"] = False

    payload: dict[str, Any] = {
        "schema": AXES02_C3_SCHEMA,
        "task_id": AXES02_C3_TASK_ID,
        "status": "complete",
        "no_truth_disclaimer": NO_TRUTH_DISCLAIMER,
        "inputs": {
            "r10": {
                "path": str(r10_path),
                "file_sha256": validation["artifact_file_sha256"],
                "expected_file_sha256": expected_r10_sha256.upper(),
                "artifact_sha256": reconstruction.artifact.get(
                    "artifact_sha256"),
                "observation_sha256": reconstruction.artifact[
                    "solver_failure_observation"].get("observation_sha256"),
                "record_sha256": reconstruction.record.get("record_sha256"),
                "failed_assignment_sha256": reconstruction.record.get(
                    "failed_assignment_sha256"),
                "input_match_identity_sha256": reconstruction.record.get(
                    "input_match_identity_sha256"),
            },
            "r11": {
                "path": str(r11_path),
                "file_sha256": observed_r11_sha256,
                "expected_file_sha256": expected_r11_sha256.upper(),
                "artifact_sha256": r11_artifact.get("artifact_sha256"),
                "entry_sha256": r11_artifact.get("entry_sha256"),
                "observation_sha256": r11_artifact[
                    "solver_failure_observation"].get("observation_sha256"),
                "solve_context": context.to_report(),
            },
            "r11_baseline": {
                "path": str(baseline_path),
                "file_sha256": baseline_file_sha256,
                "expected_file_sha256": expected_baseline_sha256.upper(),
                "seal_sha256": baseline.get("seal_sha256"),
                "expected_seal_sha256": expected_baseline_seal_sha256.lower(),
            },
            "r11_c1_c5": {
                "path": str(c1_c5_path),
                "file_sha256": c1_c5_file_sha256,
                "expected_file_sha256": expected_c1_c5_sha256.upper(),
                "seal_sha256": c1_c5.get("seal_sha256"),
                "expected_seal_sha256": expected_c1_c5_seal_sha256.lower(),
                "task_id": c1_c5.get("task_id"),
            },
        },
        "reconciliation": reconciliation,
        "validation": validation,
        "sealed_c3": {
            "file_sha256": c1_c5_file_sha256,
            "seal_sha256": c1_c5.get("seal_sha256"),
            "reason": outcome_c3.get("reason"),
            "assignment": outcome_c3.get("assignment"),
            "accepted": outcome_c3.get("accepted"),
            "resultants": outcome_c3.get("resultants"),
            "residuals": outcome_c3.get("residuals"),
            "observations": outcome_c3.get("observations"),
            "translations": outcome_c3.get("translations"),
            "active_match_count": outcome_c3.get("active_match_count"),
            "active_match_indices": outcome_c3.get("active_match_indices"),
            "duplicate_b": outcome_c3.get("duplicate_b"),
            "provenance": sealed_c3["provenance"],
            "diagnostics": sealed_c3["diagnostics"],
        },
        "arms": arms,
        "arm_order": list(AXES02_ARMS),
        "c3_substitutions": [
            {"candidate_flat_row": int(row), "replaces_match_index": int(target),
             "candidate_group": int(group), "world_axis": int(axis)}
            for row, target, group, axis in C3_SUBSTITUTIONS],
        "solver_calls_executed": len(arms),
        "solver_calls_expected": len(AXES02_ARMS),
        "baseline_rerun": False,
        "c3_rerun": False,
        "notes": [
            "The accepted R11 baseline control and the accepted sealed C1-C5 arm "
            "replay were pinned, reconciled and cross-bound before any treatment; "
            "both the baseline control and the C3 arm were consumed, never rerun.",
            "All eight frozen axes-0/2 + C3 substitutions were validated (row, "
            "group, survivor, duplicate-edge and declared-world-axis constraints) "
            "before the first of exactly eight context-pinned solver calls, each "
            "under the identical captured thresholds (.15/.90/.02) and ordered "
            "quadrant constraints.",
            "Deltas are measured against the sealed C3 outcome.  The .90 "
            "resultant / .15 residual gates are per-axis diagnostics only and "
            "were not relaxed; a passing axis is feasibility evidence for this "
            "single fixture, not candidate truth or product accuracy, and the "
            "overall geometry keeps the unchanged solver reason label.",
        ],
    }
    payload["seal_sha256"] = seal_payload(payload)
    return payload


def enumerate_valid_axis_substitutions(
    record: Mapping[str, Any], universe: ReadingUniverse,
    baseline_matches: Sequence[Any], frames: Sequence[str],
    quadrants: Mapping[str, int],
) -> tuple[tuple[AxisSubstitution, ...], dict[str, Any]]:
    """Solver-free enumeration of the frozen exhaustive substitution scope.

    A captured row is valid only when it is a dropped ``same_axis_mismatch``
    candidate of an eligible group, maps both endpoints to the declared world
    axis (0 or 2) and is active under the sealed assignment, replaces a survivor
    of its own group, and leaves no duplicate active edge key against the
    remaining active baseline matches.  Candidate edge identity may legitimately
    differ from the replaced survivor's.  The measured counts are compared with
    the Lead-frozen values; a different count is reported exactly and stops the
    replay before any solver call instead of being silently filtered.
    """
    candidates = record.get("candidates")
    if type(candidates) is not list:
        raise ReplayValidationError("candidate rows are missing")
    claims = _group_claims(record)
    unit = list(baseline_matches)
    substitutions: list[AxisSubstitution] = []
    per_group_axis: dict[tuple[int, int], int] = {}
    per_group_target: dict[tuple[int, int], int] = {}
    for flat_index in range(len(candidates)):
        row = candidates[flat_index]
        if (row[17] != "dropped"
                or row[18] != plane_graph.SAME_AXIS_DISPOSITION_DROPPED):
            continue
        group_index = int(row[0])
        world_axis = EXHAUSTIVE_AXIS_GROUPS.get(group_index)
        if world_axis is None:
            continue
        match, _row = candidate_match(record, universe, flat_index)
        left_world = _world_axis_for(match.left_axis, match.left, quadrants)
        right_world = _world_axis_for(match.right_axis, match.right, quadrants)
        if left_world != right_world or left_world not in (0, 2):
            continue
        if left_world != int(world_axis):
            continue
        if ((int(quadrants[match.right]) - int(quadrants[match.left])) % 4
                != int(match.quadrant_of_right)):
            continue
        for target_match in sorted(claims.get(group_index, ())):
            treatment = list(unit)
            treatment[target_match] = match
            if _active_keys_have_duplicates(treatment, frames, quadrants):
                continue
            substitutions.append(AxisSubstitution(
                world_axis=int(world_axis), group=group_index,
                flat_row=int(flat_index), target_match=int(target_match),
                score=float(row[8]), delta=float(row[16]), match=match))
            key = (group_index, int(world_axis))
            per_group_axis[key] = per_group_axis.get(key, 0) + 1
            target_key = (group_index, int(target_match))
            per_group_target[target_key] = per_group_target.get(target_key, 0) + 1

    observed_axis = {
        axis: sum(1 for sub in substitutions if sub.world_axis == axis)
        for axis in sorted(EXHAUSTIVE_EXPECTED_AXIS_COUNTS)}
    report = {
        "expected_axis_counts": {str(axis): count for axis, count
                                 in EXHAUSTIVE_EXPECTED_AXIS_COUNTS.items()},
        "observed_axis_counts": {str(axis): count
                                 for axis, count in observed_axis.items()},
        "expected_per_group_axis": {
            f"{group}:{axis}": count for (group, axis), count
            in EXHAUSTIVE_EXPECTED_PER_GROUP_AXIS.items()},
        "observed_per_group_axis": {
            f"{group}:{axis}": count for (group, axis), count
            in sorted(per_group_axis.items())},
        "expected_per_group_target": {
            f"{group}:{target}": count for (group, target), count
            in EXHAUSTIVE_EXPECTED_PER_GROUP_TARGET.items()},
        "observed_per_group_target": {
            f"{group}:{target}": count for (group, target), count
            in sorted(per_group_target.items())},
        "expected_total": EXHAUSTIVE_EXPECTED_TOTAL,
        "observed_total": len(substitutions),
        "validated_before_any_solver_call": True,
    }
    if (observed_axis != dict(EXHAUSTIVE_EXPECTED_AXIS_COUNTS)
            or per_group_axis != dict(EXHAUSTIVE_EXPECTED_PER_GROUP_AXIS)
            or per_group_target != dict(EXHAUSTIVE_EXPECTED_PER_GROUP_TARGET)
            or len(substitutions) != EXHAUSTIVE_EXPECTED_TOTAL):
        raise ReplayValidationError(
            "the harness's exhaustive prevalidation does not reproduce the "
            f"frozen {EXHAUSTIVE_EXPECTED_TOTAL} valid substitutions: axis "
            f"counts {observed_axis}, per group/axis "
            f"{sorted(per_group_axis.items())}, per group/target "
            f"{sorted(per_group_target.items())}, total {len(substitutions)}")
    return tuple(substitutions), report


def plan_exhaustive_arms(
    record: Mapping[str, Any], universe: ReadingUniverse,
    baseline_matches: Sequence[Any], frames: Sequence[str],
    quadrants: Mapping[str, int], c3_replacements: Mapping[int, Any],
) -> dict[str, Any]:
    """Validate every exhaustive substitution and pair key before any solve.

    Every enumerated arm must stay duplicate-free once the sealed C3 rows are
    applied, and every potential cross-axis pair must form a duplicate-free
    combined active set.  The result therefore proves pairability before the
    first solver call, so the final 202nd call can be admitted without ever
    falling back to a confidence-ranked pair.
    """
    substitutions, report = enumerate_valid_axis_substitutions(
        record, universe, baseline_matches, frames, quadrants)
    base_c3 = substitute(baseline_matches, c3_replacements)
    for sub in substitutions:
        treatment = list(base_c3)
        treatment[sub.target_match] = sub.match
        if _active_keys_have_duplicates(treatment, frames, quadrants):
            raise ReplayValidationError(
                f"exhaustive arm {sub.arm_id} leaves a duplicate active edge "
                "once the sealed C3 substitutions are applied")
    axis0 = [sub for sub in substitutions if sub.world_axis == 0]
    axis2 = [sub for sub in substitutions if sub.world_axis == 2]
    valid_pairs: list[tuple[int, int, int, int]] = []
    for first in axis0:
        for second in axis2:
            treatment = list(base_c3)
            treatment[first.target_match] = first.match
            treatment[second.target_match] = second.match
            if not _active_keys_have_duplicates(treatment, frames, quadrants):
                valid_pairs.append((first.flat_row, first.target_match,
                                    second.flat_row, second.target_match))
    pair_report = {
        "pairs_checked": len(axis0) * len(axis2),
        "pairs_valid": len(valid_pairs),
        "all_pairs_valid": len(valid_pairs) == len(axis0) * len(axis2),
        "valid_pairs_sha256": hashlib.sha256(
            canonical_json(sorted(valid_pairs)).encode("utf-8")).hexdigest(),
    }
    report["combined_arms_duplicate_free"] = len(substitutions)
    report["combined_arms_duplicate_free_all"] = True
    report["pair_prevalidation"] = pair_report
    return {
        "substitutions": substitutions,
        "report": report,
        "pair_report": pair_report,
        "valid_pair_keys": frozenset(valid_pairs),
    }


def _exhaustive_sort_key(arm: Mapping[str, Any]) -> tuple[Any, ...]:
    """The Lead-frozen measured ranking for one moved-axis arm."""
    diagnostic = arm["diagnostics"]["per_axis"][str(arm["world_axis"])]
    return (
        0 if diagnostic["axis_gate_pass"] else 1,
        -float(diagnostic["resultant"]),
        float(diagnostic["residual"]),
        -float(arm["candidate_score"]),
        int(arm["candidate_flat_row"]),
        int(arm["candidate_target_match"]),
    )


def select_best_axis_substitution(
    substitutions: Sequence[AxisSubstitution], arms: Mapping[str, Any],
    world_axis: int,
) -> AxisSubstitution:
    """Return the measured-best substitution of one axis by the frozen rule."""
    eligible = [sub for sub in substitutions if sub.world_axis == int(world_axis)]
    if not eligible:
        raise ReplayValidationError(
            f"no valid axis-{world_axis} substitution can be selected")
    return min(eligible,
               key=lambda sub: _exhaustive_sort_key(arms[sub.arm_id]))


def build_exhaustive_arms(
    readings: Sequence[Any], baseline_matches: Sequence[Any],
    frames: Sequence[str], context: ExactSolveContext,
    c3_replacements: Mapping[int, Any], c3_outcome: Mapping[str, Any],
    quadrants: Mapping[str, int],
    substitutions: Sequence[AxisSubstitution],
) -> dict[str, Any]:
    """Run exactly one context-pinned solver call per exhaustive substitution.

    Every substitution was validated before this point, so no arm is solved
    until all 201 row/group/survivor/duplicate constraints and pair keys held.
    The sealed C3 outcome is consumed, never rebuilt.
    """
    base_c3 = substitute(baseline_matches, c3_replacements)
    arms: dict[str, Any] = {}
    for sub in substitutions:
        treatment = list(base_c3)
        treatment[sub.target_match] = sub.match
        _assert_no_active_duplicates(treatment, frames, quadrants)
        outcome = run_arm(readings, treatment, frames, target=None,
                          context=context)
        arms[sub.arm_id] = {
            "arm_id": sub.arm_id,
            "world_axis": sub.world_axis,
            "candidate_flat_row": sub.flat_row,
            "candidate_group": sub.group,
            "candidate_target_match": sub.target_match,
            "candidate_score": sub.score,
            "candidate_delta": sub.delta,
            "candidate_key_replays_survivor": (
                _match_identity_key(sub.match)
                == _match_identity_key(baseline_matches[sub.target_match])),
            "includes_c3": True,
            "c3_rows": [int(entry[0]) for entry in C3_SUBSTITUTIONS],
            "overall_geometry_label": outcome["reason"],
            "outcome": outcome,
            "diagnostics": _axis_diagnostics(outcome, context),
            "deltas_vs_sealed_c3": _arm_deltas(outcome, c3_outcome),
        }
    return arms


def _arm_summary(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, JSON-safe summary of one exhaustive arm."""
    outcome = arm["outcome"]
    diagnostics = arm["diagnostics"]
    moved = diagnostics["per_axis"][str(arm["world_axis"])]
    return {
        "arm_id": arm["arm_id"],
        "world_axis": int(arm["world_axis"]),
        "candidate_flat_row": int(arm["candidate_flat_row"]),
        "candidate_group": int(arm["candidate_group"]),
        "candidate_target_match": int(arm["candidate_target_match"]),
        "candidate_score": float(arm["candidate_score"]),
        "candidate_delta": float(arm["candidate_delta"]),
        "candidate_key_replays_survivor": bool(
            arm["candidate_key_replays_survivor"]),
        "reason": outcome["reason"],
        "assignment": outcome["assignment"],
        "resultants": outcome["resultants"],
        "residuals": outcome["residuals"],
        "observations": outcome["observations"],
        "translations": outcome["translations"],
        "active_match_count": outcome["active_match_count"],
        "duplicate_b": outcome["duplicate_b"],
        "axis_coverage": diagnostics["axis_coverage"],
        "moved_axis_resultant": float(moved["resultant"]),
        "moved_axis_residual": float(moved["residual"]),
        "moved_axis_gate_pass": bool(moved["axis_gate_pass"]),
        "all_axis_gate_pass": all(
            value["axis_gate_pass"]
            for value in diagnostics["per_axis"].values()),
    }


def _summarize_exhaustive_arms(
    arms: Mapping[str, Any], substitutions: Sequence[AxisSubstitution],
) -> dict[str, Any]:
    """Best/worst and threshold-pass summary for each moved axis."""
    summary: dict[str, Any] = {}
    all_axis_passes: list[str] = []
    for world_axis in (0, 2):
        eligible = [sub for sub in substitutions
                    if sub.world_axis == world_axis]
        ordered = sorted(eligible,
                         key=lambda sub: _exhaustive_sort_key(arms[sub.arm_id]))
        passing = [arm for arm in arms.values()
                   if arm["world_axis"] == world_axis
                   and arm["diagnostics"]["per_axis"][
                       str(world_axis)]["axis_gate_pass"]]
        complete = [arm for arm in arms.values()
                    if arm["world_axis"] == world_axis
                    and all(value["axis_gate_pass"] for value
                            in arm["diagnostics"]["per_axis"].values())]
        all_axis_passes.extend(arm["arm_id"] for arm in complete)
        summary[str(world_axis)] = {
            "valid_count": len(eligible),
            "moved_axis_gate_pass_count": len(passing),
            "moved_axis_gate_pass_arms": sorted(arm["arm_id"]
                                                for arm in passing),
            "all_axis_gate_pass_count": len(complete),
            "all_axis_gate_pass_arms": sorted(arm["arm_id"]
                                              for arm in complete),
            "best": _arm_summary(arms[ordered[0].arm_id]),
            "worst": _arm_summary(arms[ordered[-1].arm_id]),
        }
    summary["threshold_passes"] = {
        "moved_axis": {
            "0": summary["0"]["moved_axis_gate_pass_arms"],
            "2": summary["2"]["moved_axis_gate_pass_arms"],
        },
        "all_axis": sorted(all_axis_passes),
        "definition": (
            "per-axis .90 resultant / .15 residual diagnostics only, never an "
            "accuracy claim or a preregistered pass/fail gate"),
    }
    return summary


def replay_r11_axes02_exhaustive(
    r10_path: Path, r11_path: Path, baseline_path: Path, c1_c5_path: Path,
    axes02_path: Path, *,
    expected_r10_sha256: str = INPUT_FILE_SHA256,
    expected_r11_sha256: str = R11_INPUT_FILE_SHA256,
    expected_context_sha256: str = R11_CONTEXT_SHA256,
    expected_baseline_sha256: str = R11_BASELINE_FILE_SHA256,
    expected_baseline_seal_sha256: str = R11_BASELINE_SEAL_SHA256,
    expected_c1_c5_sha256: str = R11_C1_C5_FILE_SHA256,
    expected_c1_c5_seal_sha256: str = R11_C1_C5_SEAL_SHA256,
    expected_axes02_sha256: str = R11_AXES02_FILE_SHA256,
    expected_axes02_seal_sha256: str = R11_AXES02_SEAL_SHA256,
    tolerance: float = CONTROL_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Run the 201 exhaustive one-for-one arms plus one measured-best pair.

    Ordering is deliberate.  The accepted baseline, C1-C5 and axes-0/2
    artifacts are pinned and reconciled, the R10/R11 inputs are reconstructed,
    and every enumerated row/target substitution and every cross-axis pair key
    is validated *before* the first of the 201 context-pinned solver calls.
    The final 202nd call is admitted only for the measured-best axis-0 and
    axis-2 winners when their pair passes the prevalidated key check; otherwise
    the replay stops at 201 and reports why.  Neither the baseline, C1-C5, C3
    nor any accepted axes-0/2 arm is rebuilt or rerun.
    """
    baseline = load_r11_baseline(
        baseline_path, expected_sha256=expected_baseline_sha256,
        expected_seal_sha256=expected_baseline_seal_sha256)
    baseline_file_sha256 = sha256_file(Path(baseline_path)).upper()

    c1_c5 = load_r11_c1_c5(
        c1_c5_path, expected_sha256=expected_c1_c5_sha256,
        expected_seal_sha256=expected_c1_c5_seal_sha256)
    c1_c5_file_sha256 = sha256_file(Path(c1_c5_path)).upper()

    axes02 = load_r11_axes02(
        axes02_path, expected_sha256=expected_axes02_sha256,
        expected_seal_sha256=expected_axes02_seal_sha256)
    axes02_file_sha256 = sha256_file(Path(axes02_path)).upper()

    r11_path = Path(r11_path)
    if not r11_path.is_file():
        raise ReplayValidationError(f"pinned R11 artifact is absent: {r11_path}")
    observed_r11_sha256 = sha256_file(r11_path).upper()
    if observed_r11_sha256 != expected_r11_sha256.upper():
        raise ReplayValidationError(
            "R11 artifact file SHA-256 does not match the pinned value")
    with open(r11_path, "r", encoding="utf-8") as handle:
        r11_artifact = json.load(handle)
    if r11_artifact.get("schema") != INPUT_SCHEMA:
        raise ReplayValidationError("R11 artifact schema is not the stage schema")

    reconstruction = load_r10_reconstruction(
        r10_path, expected_sha256=expected_r10_sha256)
    context = exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames,
        expected_context_sha256=expected_context_sha256)

    reconciliation = reconcile_arms_inputs(
        reconstruction, r11_artifact, context, baseline,
        r11_file_sha256=observed_r11_sha256)

    # Consume-only cross-binds of the sealed C1-C5 and axes-0/2 replays.
    if c1_c5.get("reconciliation") != reconciliation:
        raise ReplayValidationError(
            "the sealed C1-C5 replay's reconciliation does not replay")
    if (c1_c5["validation"].get("artifact_file_sha256")
            != reconstruction.validation["artifact_file_sha256"]):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's R10 file binding does not replay")
    if (str(c1_c5["inputs"]["r10"].get("record_sha256", ""))
            != str(reconstruction.record.get("record_sha256"))):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's R10 record binding does not replay")
    if c1_c5["inputs"]["r11"].get("solve_context") != context.to_report():
        raise ReplayValidationError(
            "the sealed C1-C5 replay's solve-context binding does not replay")
    if (str(c1_c5["inputs"]["r11_baseline"].get("file_sha256", "")).upper()
            != baseline_file_sha256):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's baseline file binding does not replay")

    if axes02.get("reconciliation") != reconciliation:
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's reconciliation does not replay")
    if (axes02["validation"].get("artifact_file_sha256")
            != reconstruction.validation["artifact_file_sha256"]):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's R10 file binding does not replay")
    if (str(axes02["inputs"]["r10"].get("record_sha256", ""))
            != str(reconstruction.record.get("record_sha256"))):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's R10 record binding does not replay")
    if axes02["inputs"]["r11"].get("solve_context") != context.to_report():
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's solve-context binding does not replay")
    if (str(axes02["inputs"]["r11_baseline"].get("file_sha256", "")).upper()
            != baseline_file_sha256):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's baseline binding does not replay")
    if (str(axes02["inputs"]["r11_c1_c5"].get("file_sha256", "")).upper()
            != c1_c5_file_sha256):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's C1-C5 file binding does not replay")
    sealed_c3 = sealed_c3_reference(c1_c5)
    frozen_c3 = sorted(
        (int(entry["candidate_flat_row"]), int(entry["candidate_group"]),
         int(entry["replaces_match_index"]))
        for entry in sealed_c3["provenance"])
    expected_c3 = sorted((int(row), int(group), int(target))
                         for row, target, group, _axis in C3_SUBSTITUTIONS)
    if frozen_c3 != expected_c3:
        raise ReplayValidationError(
            "the sealed C3 arm is not the frozen C3 substitution")
    axes02_c3 = axes02.get("sealed_c3")
    if not isinstance(axes02_c3, dict):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay carries no sealed C3 reference")
    for key in ("reason", "assignment", "resultants", "residuals",
                "observations", "active_match_count", "active_match_indices",
                "duplicate_b", "translations"):
        if axes02_c3.get(key) != sealed_c3["outcome"].get(key):
            raise ReplayValidationError(
                "the sealed axes-0/2 C3 reference does not replay the accepted "
                f"C3 outcome ({key})")
    if axes02_c3.get("provenance") != sealed_c3["provenance"]:
        raise ReplayValidationError(
            "the sealed axes-0/2 C3 provenance does not replay")
    if axes02_c3.get("diagnostics") != sealed_c3["diagnostics"]:
        raise ReplayValidationError(
            "the sealed axes-0/2 C3 diagnostics do not replay")
    if (str(axes02_c3.get("file_sha256", "")).upper()
            != c1_c5_file_sha256):
        raise ReplayValidationError(
            "the sealed axes-0/2 C3 file binding does not replay")

    frames = tuple(reconstruction.universe.frames)
    outcome_c3 = sealed_c3["outcome"]
    assignment = list(outcome_c3.get("assignment") or [])
    if len(assignment) != len(frames):
        raise ReplayValidationError(
            "the sealed C3 outcome assignment is not frame-aligned")
    quadrants = {frame: int(value) for frame, value in zip(frames, assignment)}

    c3_replacements, _c3_provenance = plan_combined_arm(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        quadrants, C3_SUBSTITUTIONS, label="C3")
    plans = plan_exhaustive_arms(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        frames, quadrants, c3_replacements)
    substitutions = plans["substitutions"]

    arms = build_exhaustive_arms(
        reconstruction.universe.readings, reconstruction.matches, frames,
        context, c3_replacements, outcome_c3, quadrants, substitutions)

    best0 = select_best_axis_substitution(substitutions, arms, 0)
    best2 = select_best_axis_substitution(substitutions, arms, 2)
    pair_key = (best0.flat_row, best0.target_match,
                best2.flat_row, best2.target_match)
    pair_prevalidated = pair_key in plans["valid_pair_keys"]

    final_arm: dict[str, Any] | None = None
    if pair_prevalidated:
        replacements = dict(c3_replacements)
        replacements[best0.target_match] = best0.match
        replacements[best2.target_match] = best2.match
        treatment = substitute(reconstruction.matches, replacements)
        if _active_keys_have_duplicates(treatment, frames, quadrants):
            raise ReplayValidationError(
                "the measured-best axis pair contradicts the solver-free pair "
                "prevalidation")
        final_outcome = run_arm(
            reconstruction.universe.readings, treatment, frames, target=None,
            context=context)
        final_arm = {
            "arm_id": f"A{best0.flat_row:03d}-{best2.flat_row:03d}",
            "axis0_arm_id": best0.arm_id,
            "axis2_arm_id": best2.arm_id,
            "candidate_flat_rows": [best0.flat_row, best2.flat_row],
            "candidate_targets": [best0.target_match, best2.target_match],
            "includes_c3": True,
            "c3_rows": [int(entry[0]) for entry in C3_SUBSTITUTIONS],
            "pair_key_prevalidated": True,
            "overall_geometry_label": final_outcome["reason"],
            "outcome": final_outcome,
            "diagnostics": _axis_diagnostics(final_outcome, context),
            "deltas_vs_sealed_c3": _arm_deltas(final_outcome, outcome_c3),
            "summary": None,
        }
        final_arm["summary"] = _arm_summary({
            "arm_id": final_arm["arm_id"],
            "world_axis": best0.world_axis,
            "candidate_flat_row": best0.flat_row,
            "candidate_group": best0.group,
            "candidate_target_match": best0.target_match,
            "candidate_score": best0.score,
            "candidate_delta": best0.delta,
            "candidate_key_replays_survivor": False,
            "outcome": final_outcome,
            "diagnostics": final_arm["diagnostics"],
        })
        status = "complete"
        solver_calls = len(arms) + 1
    else:
        status = "stopped_at_201_no_valid_pair"
        solver_calls = len(arms)

    summary = _summarize_exhaustive_arms(arms, substitutions)
    summary["final_combined"] = (
        final_arm["summary"] if final_arm is not None else None)
    summary["final_combined_pair_key_prevalidated"] = pair_prevalidated
    summary["best_axis0_arm"] = best0.arm_id
    summary["best_axis2_arm"] = best2.arm_id
    summary["selection_rule"] = EXHAUSTIVE_SELECTION_RULE
    summary["stopped_at_201"] = final_arm is None

    validation = dict(reconstruction.validation)
    validation["r11_artifact_file_sha256"] = observed_r11_sha256
    validation["r11_context_replay"] = "pass"
    validation["r11_context_frame_binding"] = "pass"
    validation["r10_r11_reconciliation"] = "pass"
    validation["sealed_baseline_reconciliation_replays"] = "pass"
    validation["sealed_control_reproduced_recorded_outcome"] = "pass"
    validation["sealed_c1_c5_reconciliation_replays"] = "pass"
    validation["sealed_axes02_reconciliation_replays"] = "pass"
    validation["sealed_c3_provenance_replays"] = "pass"
    validation["exhaustive_prevalidation_reproduces_201"] = "pass"
    validation["all_201_substitutions_validated_before_any_arm"] = "pass"
    validation["all_cross_axis_pairs_validated_solver_free"] = "pass"
    validation["measured_best_pair_key_prevalidated"] = bool(pair_prevalidated)
    validation["baseline_rerun"] = False
    validation["c3_rerun"] = False
    validation["axes02_rerun"] = False
    validation["c1_c5_rerun"] = False

    payload: dict[str, Any] = {
        "schema": AXES02_EXHAUSTIVE_SCHEMA,
        "task_id": AXES02_EXHAUSTIVE_TASK_ID,
        "status": status,
        "no_truth_disclaimer": NO_TRUTH_DISCLAIMER,
        "inputs": {
            "r10": {
                "path": str(r10_path),
                "file_sha256": validation["artifact_file_sha256"],
                "expected_file_sha256": expected_r10_sha256.upper(),
                "artifact_sha256": reconstruction.artifact.get(
                    "artifact_sha256"),
                "observation_sha256": reconstruction.artifact[
                    "solver_failure_observation"].get("observation_sha256"),
                "record_sha256": reconstruction.record.get("record_sha256"),
                "failed_assignment_sha256": reconstruction.record.get(
                    "failed_assignment_sha256"),
                "input_match_identity_sha256": reconstruction.record.get(
                    "input_match_identity_sha256"),
            },
            "r11": {
                "path": str(r11_path),
                "file_sha256": observed_r11_sha256,
                "expected_file_sha256": expected_r11_sha256.upper(),
                "artifact_sha256": r11_artifact.get("artifact_sha256"),
                "entry_sha256": r11_artifact.get("entry_sha256"),
                "observation_sha256": r11_artifact[
                    "solver_failure_observation"].get("observation_sha256"),
                "solve_context": context.to_report(),
            },
            "r11_baseline": {
                "path": str(baseline_path),
                "file_sha256": baseline_file_sha256,
                "expected_file_sha256": expected_baseline_sha256.upper(),
                "seal_sha256": baseline.get("seal_sha256"),
                "expected_seal_sha256": expected_baseline_seal_sha256.lower(),
            },
            "r11_c1_c5": {
                "path": str(c1_c5_path),
                "file_sha256": c1_c5_file_sha256,
                "expected_file_sha256": expected_c1_c5_sha256.upper(),
                "seal_sha256": c1_c5.get("seal_sha256"),
                "expected_seal_sha256": expected_c1_c5_seal_sha256.lower(),
                "task_id": c1_c5.get("task_id"),
            },
            "r11_axes02": {
                "path": str(axes02_path),
                "file_sha256": axes02_file_sha256,
                "expected_file_sha256": expected_axes02_sha256.upper(),
                "seal_sha256": axes02.get("seal_sha256"),
                "expected_seal_sha256": expected_axes02_seal_sha256.lower(),
                "task_id": axes02.get("task_id"),
            },
        },
        "reconciliation": reconciliation,
        "validation": validation,
        "sealed_c3": {
            "file_sha256": c1_c5_file_sha256,
            "seal_sha256": c1_c5.get("seal_sha256"),
            "reason": outcome_c3.get("reason"),
            "assignment": outcome_c3.get("assignment"),
            "accepted": outcome_c3.get("accepted"),
            "resultants": outcome_c3.get("resultants"),
            "residuals": outcome_c3.get("residuals"),
            "observations": outcome_c3.get("observations"),
            "translations": outcome_c3.get("translations"),
            "active_match_count": outcome_c3.get("active_match_count"),
            "active_match_indices": outcome_c3.get("active_match_indices"),
            "duplicate_b": outcome_c3.get("duplicate_b"),
            "provenance": sealed_c3["provenance"],
            "diagnostics": sealed_c3["diagnostics"],
        },
        "prevalidation": plans["report"],
        "substitutions": [
            {
                "arm_id": sub.arm_id,
                "world_axis": sub.world_axis,
                "candidate_group": sub.group,
                "candidate_flat_row": sub.flat_row,
                "candidate_target_match": sub.target_match,
                "candidate_score": sub.score,
                "candidate_delta": sub.delta,
            }
            for sub in substitutions],
        "arms": arms,
        "arm_order": [sub.arm_id for sub in substitutions],
        "selection": {
            "rule": EXHAUSTIVE_SELECTION_RULE,
            "best_axis0_arm": best0.arm_id,
            "best_axis2_arm": best2.arm_id,
            "best_axis0": _arm_summary(arms[best0.arm_id]),
            "best_axis2": _arm_summary(arms[best2.arm_id]),
            "pair_key": list(pair_key),
            "pair_key_prevalidated": bool(pair_prevalidated),
            "final_arm": (
                None if final_arm is None
                else {key: value for key, value in final_arm.items()
                      if key != "summary"}),
            "final_arm_summary": (
                None if final_arm is None else final_arm["summary"]),
        },
        "summary": summary,
        "solver_calls_authorized": EXHAUSTIVE_AUTHORIZED_SOLVER_CALLS,
        "solver_calls_executed": solver_calls,
        "solver_calls_expected": EXHAUSTIVE_AUTHORIZED_SOLVER_CALLS,
        "baseline_rerun": False,
        "c3_rerun": False,
        "axes02_rerun": False,
        "c1_c5_rerun": False,
        "notes": [
            "The accepted R11 baseline control, the accepted sealed C1-C5 arm "
            "replay and the accepted sealed axes-0/2 eight-arm replay were "
            "pinned, reconciled and cross-bound before any treatment; none of "
            "them, and not the sealed C3 arm, was rerun.",
            "All 201 valid captured one-for-one world-axis-0/2 substitutions "
            "(164 axis-0 + 37 axis-2) were validated (dropped same-axis row, "
            "declared group, group survivor, active world axis under [0, 3, 1] "
            "and duplicate-free active edge set) before the first of exactly "
            "201 context-pinned solver calls, each combined with the sealed C3 "
            "rows under the identical captured thresholds (.15/.90/.02) and "
            "ordered quadrant constraints.",
            "Candidate edge identity was allowed to differ from the replaced "
            "survivor's, per the Lead clarification; only duplicate active "
            "edge keys were refused.  All cross-axis pair keys were validated "
            "solver-free before the first call.",
            "The single final combined arm uses the measured-best axis-0 and "
            "axis-2 substitutions by the frozen measured ranking and was run "
            "only after its pair passed the prevalidated duplicate-key check; "
            "no confidence-ranked fallback pair was ever substituted.",
            "Deltas are measured against the sealed C3 outcome.  The .90 "
            "resultant / .15 residual gates are per-axis diagnostics only and "
            "were not relaxed; a passing axis is feasibility evidence for this "
            "single fixture, not candidate truth or product accuracy, and the "
            "overall geometry keeps the unchanged solver reason label.",
        ],
    }
    payload["seal_sha256"] = seal_payload(payload)
    return payload


def _validate_r11_axes02_exhaustive_payload(
    payload: Mapping[str, Any], *,
    expected_seal_sha256: str = R11_AXES02_EXHAUSTIVE_SEAL_SHA256,
) -> Mapping[str, Any]:
    """Fail-closed checks on the already-loaded accepted exhaustive replay.

    The artifact is consumed as a sealed cross-reference: its own prevalidated
    6,068-pair digest and its frozen ``A020-308`` final arm anchor the pair scan.
    A file that does not replay its own seal, is not the accepted exhaustive
    task, does not bind the pinned R10/R11/baseline/C1-C5/axes-0/2 artifacts, or
    whose prevalidation/final arm differ from the frozen values can never define
    the reference.
    """
    recorded_seal = payload.get("seal_sha256")
    recomputed = seal_payload(payload)
    if recorded_seal != recomputed:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay seal does not replay over its own "
            "payload")
    if str(recorded_seal).lower() != expected_seal_sha256.lower():
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay seal does not match the pinned value")
    if payload.get("schema") != AXES02_EXHAUSTIVE_SCHEMA:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay schema is not the accepted "
            "exhaustive schema")
    if payload.get("task_id") != R11_AXES02_EXHAUSTIVE_TASK_ID:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay task id is not the accepted task")
    if payload.get("status") != "complete":
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay is not a completed run")
    if payload.get("solver_calls_executed") != 202:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay is not exactly 202 solver calls")
    if payload.get("solver_calls_expected") != 202:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay does not expect 202 solver calls")
    if payload.get("solver_calls_authorized") != 202:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay does not authorize 202 solver calls")
    for flag in ("baseline_rerun", "c3_rerun", "axes02_rerun", "c1_c5_rerun"):
        if payload.get(flag) is not False:
            raise ReplayValidationError(
                f"R11 axes-0/2 exhaustive replay reports {flag}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay inputs are missing")
    r10_binding = inputs.get("r10")
    r11_binding = inputs.get("r11")
    baseline_binding = inputs.get("r11_baseline")
    c1_c5_binding = inputs.get("r11_c1_c5")
    axes02_binding = inputs.get("r11_axes02")
    if not all(isinstance(item, dict) for item in (
            r10_binding, r11_binding, baseline_binding, c1_c5_binding,
            axes02_binding)):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay input bindings are incomplete")
    if str(r10_binding.get("file_sha256", "")).upper() != INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay R10 digest does not match the pin")
    if str(r11_binding.get("file_sha256", "")).upper() != R11_INPUT_FILE_SHA256:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay R11 digest does not match the pin")
    context_record = r11_binding.get("solve_context")
    if (not isinstance(context_record, dict)
            or str(context_record.get("record_sha256", "")).lower()
            != R11_CONTEXT_SHA256.lower()):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay solve-context digest does not match "
            "the pin")
    if (str(baseline_binding.get("file_sha256", "")).upper()
            != R11_BASELINE_FILE_SHA256):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay baseline digest does not match the "
            "pin")
    if (str(c1_c5_binding.get("file_sha256", "")).upper()
            != R11_C1_C5_FILE_SHA256):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay C1-C5 digest does not match the pin")
    if (str(axes02_binding.get("file_sha256", "")).upper()
            != R11_AXES02_FILE_SHA256):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay axes-0/2 digest does not match the "
            "pin")
    if (str(axes02_binding.get("seal_sha256", "")).lower()
            != R11_AXES02_SEAL_SHA256.lower()):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay axes-0/2 seal does not match the pin")

    prevalidation = payload.get("prevalidation")
    if not isinstance(prevalidation, dict):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay prevalidation is missing")
    if prevalidation.get("observed_total") != EXHAUSTIVE_EXPECTED_TOTAL:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay does not carry the frozen 201 "
            "substitutions")
    if prevalidation.get("observed_axis_counts") != {
            str(axis): count for axis, count
            in EXHAUSTIVE_EXPECTED_AXIS_COUNTS.items()}:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay axis counts are not the frozen "
            "164/37")
    pair_prevalidation = prevalidation.get("pair_prevalidation")
    if not isinstance(pair_prevalidation, dict):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay pair prevalidation is missing")
    if (pair_prevalidation.get("pairs_checked") != PAIR_SCAN_EXPECTED_PAIRS
            or pair_prevalidation.get("pairs_valid") != PAIR_SCAN_EXPECTED_PAIRS
            or pair_prevalidation.get("all_pairs_valid") is not True):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay does not carry the frozen 6,068 "
            "duplicate-free pairs")
    if (str(pair_prevalidation.get("valid_pairs_sha256", "")).lower()
            != PAIR_SCAN_PAIR_KEY_SHA256.lower()):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay pair-key digest does not match the "
            "pin")

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay selection is missing")
    if list(selection.get("pair_key") or []) != list(
            PAIR_SCAN_ACCEPTED_REFERENCE_PAIR):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay pair key is not the frozen "
            "A020-308 pair")
    if selection.get("pair_key_prevalidated") is not True:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay pair key was not prevalidated")
    final_arm = selection.get("final_arm")
    if not isinstance(final_arm, dict):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay final arm is missing")
    if final_arm.get("arm_id") != PAIR_SCAN_ACCEPTED_REFERENCE_LABEL:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay final arm is not A020-308")
    if list(final_arm.get("candidate_flat_rows") or []) != [
            PAIR_SCAN_ACCEPTED_REFERENCE_PAIR[0],
            PAIR_SCAN_ACCEPTED_REFERENCE_PAIR[2]]:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay final arm rows are not the frozen "
            "pair")
    if list(final_arm.get("candidate_targets") or []) != [
            PAIR_SCAN_ACCEPTED_REFERENCE_PAIR[1],
            PAIR_SCAN_ACCEPTED_REFERENCE_PAIR[3]]:
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay final arm targets are not the "
            "frozen pair")
    if not isinstance(payload.get("sealed_c3"), dict):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay sealed C3 reference is missing")
    return payload


def load_r11_axes02_exhaustive(
    path: Path, *,
    expected_sha256: str = R11_AXES02_EXHAUSTIVE_FILE_SHA256,
    expected_seal_sha256: str = R11_AXES02_EXHAUSTIVE_SEAL_SHA256,
) -> Mapping[str, Any]:
    """Load and fail-closed-validate the accepted sealed exhaustive replay."""
    path = Path(path)
    if not path.is_file():
        raise ReplayValidationError(
            f"pinned R11 axes-0/2 exhaustive replay is absent: {path}")
    observed_sha256 = sha256_file(path).upper()
    if observed_sha256 != expected_sha256.upper():
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay file SHA-256 does not match the "
            "pinned value")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ReplayValidationError(
            "R11 axes-0/2 exhaustive replay is not a JSON object")
    _validate_r11_axes02_exhaustive_payload(
        payload, expected_seal_sha256=expected_seal_sha256)
    return payload


def _pair_scan_pair_id(first: AxisSubstitution,
                       second: AxisSubstitution) -> str:
    """The stable ``A<axis0 row>-<axis2 row>`` label of one combined pair."""
    return f"A{first.flat_row:03d}-{second.flat_row:03d}"


def plan_pair_scan(
    plans: Mapping[str, Any], baseline_matches: Sequence[Any],
    frames: Sequence[str], quadrants: Mapping[str, int],
    c3_replacements: Mapping[int, Any],
) -> dict[str, Any]:
    """Solver-free enumeration of the exact 6,068 axis-0 x axis-2 pairs.

    ``plans`` is the already-validated exhaustive plan (201 substitutions and
    their pair keys).  Each axis-0 substitution is combined with each axis-2
    substitution on top of the sealed C3 rows and the combined active edge set
    must be duplicate-free.  The sorted pair-key digest is recomputed here and
    required to equal both the pinned value and the digest the accepted
    exhaustive run recorded.  Nothing is solved.
    """
    substitutions = list(plans["substitutions"])
    axis0 = [sub for sub in substitutions if sub.world_axis == 0]
    axis2 = [sub for sub in substitutions if sub.world_axis == 2]
    expected0 = PAIR_SCAN_EXPECTED_AXIS_COUNTS[0]
    expected2 = PAIR_SCAN_EXPECTED_AXIS_COUNTS[2]
    if len(axis0) != expected0 or len(axis2) != expected2:
        raise ReplayValidationError(
            "the pair scan's eligible axis substitution counts are not the "
            f"frozen {expected0} axis-0 / {expected2} axis-2: "
            f"{len(axis0)}/{len(axis2)}")
    base_c3 = substitute(baseline_matches, c3_replacements)
    pairs: list[tuple[AxisSubstitution, AxisSubstitution]] = []
    keys: list[tuple[int, int, int, int]] = []
    for first in axis0:
        for second in axis2:
            treatment = list(base_c3)
            treatment[first.target_match] = first.match
            treatment[second.target_match] = second.match
            if _active_keys_have_duplicates(treatment, frames, quadrants):
                raise ReplayValidationError(
                    f"pair {_pair_scan_pair_id(first, second)} leaves a "
                    "duplicate active edge once the sealed C3 substitutions "
                    "are applied")
            pairs.append((first, second))
            keys.append((first.flat_row, first.target_match,
                         second.flat_row, second.target_match))
    if len(pairs) != PAIR_SCAN_EXPECTED_PAIRS:
        raise ReplayValidationError(
            "the pair scan did not enumerate the frozen "
            f"{PAIR_SCAN_EXPECTED_PAIRS} pairs: {len(pairs)}")
    if len(set(keys)) != len(keys):
        raise ReplayValidationError("the pair scan's active pair keys repeat")
    digest = hashlib.sha256(
        canonical_json(sorted(keys)).encode("utf-8")).hexdigest()
    exhaustive_digest = str(
        plans["report"]["pair_prevalidation"]["valid_pairs_sha256"]).lower()
    report = {
        "expected_axis_counts": {
            str(axis): count for axis, count
            in PAIR_SCAN_EXPECTED_AXIS_COUNTS.items()},
        "observed_axis_counts": {"0": len(axis0), "2": len(axis2)},
        "expected_pairs": PAIR_SCAN_EXPECTED_PAIRS,
        "observed_pairs": len(pairs),
        "duplicate_free_pairs": len(pairs),
        "duplicate_free_all": True,
        "pair_keys_unique": True,
        "pair_key_sha256": digest,
        "expected_pair_key_sha256": PAIR_SCAN_PAIR_KEY_SHA256.lower(),
        "exhaustive_recorded_pair_key_sha256": exhaustive_digest,
        "pair_key_sha256_matches_pin": digest == PAIR_SCAN_PAIR_KEY_SHA256.lower(),
        "pair_key_sha256_matches_exhaustive": digest == exhaustive_digest,
        "validated_before_any_solver_call": True,
    }
    if not report["pair_key_sha256_matches_pin"]:
        raise ReplayValidationError(
            "the pair scan's independently re-derived pair-key digest does not "
            "match the pinned value")
    if not report["pair_key_sha256_matches_exhaustive"]:
        raise ReplayValidationError(
            "the pair scan's pair-key digest does not match the accepted "
            "exhaustive replay's recorded digest")
    return {
        "pairs": tuple(pairs),
        "keys": tuple(keys),
        "digest": digest,
        "report": report,
    }


def _pair_scan_axis_provenance(
    sub: AxisSubstitution, matches: Sequence[Any],
) -> dict[str, Any]:
    """One moved axis's substitution provenance plus its replaced-survivor delta."""
    survivor = matches[sub.target_match]
    return {
        "flat_row": int(sub.flat_row),
        "group": int(sub.group),
        "target_match": int(sub.target_match),
        "candidate_score": float(sub.score),
        "candidate_delta": float(sub.delta),
        "replaced_survivor_score": float(survivor.score),
        "score_delta_vs_replaced_survivor": float(sub.score - survivor.score),
        "candidate_key_replays_survivor": (
            _match_identity_key(sub.match) == _match_identity_key(survivor)),
    }


def _pair_scan_gate(
    resultants: Mapping[str, float], residuals: Mapping[str, float],
    context: ExactSolveContext,
) -> dict[str, Any]:
    """Per-axis ``.90``/``.15`` gate diagnostics plus the unchanged margins."""
    min_resultant = float(context.min_resultant)
    integer_tolerance = float(context.integer_tolerance)
    passes: dict[str, bool] = {}
    margins: dict[str, float] = {}
    for axis in ("0", "1", "2"):
        if axis not in resultants or axis not in residuals:
            raise ReplayValidationError(
                f"a pair outcome is missing axis {axis}")
        passes[axis] = (float(resultants[axis]) >= min_resultant
                        and float(residuals[axis]) <= integer_tolerance)
        margins[f"axis{axis}_resultant_minus_min"] = (
            float(resultants[axis]) - min_resultant)
        margins[f"axis{axis}_tolerance_minus_residual"] = (
            integer_tolerance - float(residuals[axis]))
    return {
        "axis0_pass": bool(passes["0"]),
        "axis1_pass": bool(passes["1"]),
        "axis2_pass": bool(passes["2"]),
        "moved_axes_pass": bool(passes["0"] and passes["2"]),
        "all_axes_pass": bool(passes["0"] and passes["1"] and passes["2"]),
        "margins": margins,
    }


def _pair_scan_row(
    index: int, first: AxisSubstitution, second: AxisSubstitution,
    matches: Sequence[Any], outcome: Mapping[str, Any],
    context: ExactSolveContext,
) -> dict[str, Any]:
    """Compact, JSON-safe record of one combined pair's exact solver outcome."""
    resultants = {str(axis): float(value)
                  for axis, value in outcome["resultants"].items()}
    residuals = {str(axis): float(value)
                 for axis, value in outcome["residuals"].items()}
    axis0 = _pair_scan_axis_provenance(first, matches)
    axis2 = _pair_scan_axis_provenance(second, matches)
    gate = _pair_scan_gate(resultants, residuals, context)
    return {
        "index": int(index),
        "pair_id": _pair_scan_pair_id(first, second),
        "axis0": axis0,
        "axis2": axis2,
        "score_delta_axis0": axis0["score_delta_vs_replaced_survivor"],
        "score_delta_axis2": axis2["score_delta_vs_replaced_survivor"],
        "includes_c3": True,
        "c3_rows": [int(entry[0]) for entry in C3_SUBSTITUTIONS],
        "reason": outcome["reason"],
        "accepted": bool(outcome["accepted"]),
        "assignment": [int(value) for value in outcome["assignment"]],
        "resultants": resultants,
        "residuals": residuals,
        "observations": [int(value)
                         for value in (outcome.get("observations") or [])],
        "active_match_count": int(outcome["active_match_count"]),
        "duplicate_b": bool(outcome["duplicate_b"]),
        "gate": gate,
    }


def build_pair_scan_rows(
    readings: Sequence[Any], matches: Sequence[Any], frames: Sequence[str],
    context: ExactSolveContext, c3_replacements: Mapping[int, Any],
    quadrants: Mapping[str, int],
    pairs: Sequence[tuple[AxisSubstitution, AxisSubstitution]],
) -> list[dict[str, Any]]:
    """Run exactly one pinned solver call per prevalidated pair, once each.

    Every pair's duplicate-free active edge set was proven before the first
    call, so no pair is solved until the full 6,068-pair prevalidation held.
    There are no retries and no baseline/C1-C5/C3 rerun: one call per pair.
    """
    base_c3 = substitute(matches, c3_replacements)
    rows: list[dict[str, Any]] = []
    for index, (first, second) in enumerate(pairs):
        treatment = list(base_c3)
        treatment[first.target_match] = first.match
        treatment[second.target_match] = second.match
        _assert_no_active_duplicates(treatment, frames, quadrants)
        outcome = run_arm(readings, treatment, frames, target=None,
                          context=context)
        rows.append(_pair_scan_row(
            index, first, second, matches, outcome, context))
    return rows


def pair_scan_objective_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    """The exact 6-D objective vector; every component is *maximised*."""
    return (
        float(row["score_delta_axis0"]),
        float(row["score_delta_axis2"]),
        float(row["resultants"]["0"]),
        float(row["resultants"]["2"]),
        -float(row["residuals"]["0"]),
        -float(row["residuals"]["2"]),
    )


def _exact_dominates(
    first: Sequence[float], second: Sequence[float],
) -> bool:
    """Exact componentwise 6-D dominance (no tolerance, one strict component)."""
    strictly_better = False
    for value, reference in zip(first, second):
        if value < reference:
            return False
        if value > reference:
            strictly_better = True
    return strictly_better


def pareto_frontier(
    vectors: Mapping[str, Sequence[float]],
) -> list[str]:
    """All exactly non-dominated ids in one 6-D maximisation space.

    Sorting by descending lexicographic objective order guarantees that any
    dominator of the current vector has already been processed (dominance implies
    lexicographic precedence), so the single pass over the maintained frontier is
    exact.  Equal vectors do not dominate each other and both remain.
    """
    for pair_id, vector in vectors.items():
        for value in vector:
            if not math.isfinite(float(value)):
                raise ReplayValidationError(
                    f"pair {pair_id} has a non-finite Pareto objective")
    ordered = sorted(
        vectors,
        key=lambda pair_id: (
            tuple(-float(value) for value in vectors[pair_id]), pair_id))
    frontier: list[tuple[str, tuple[float, ...]]] = []
    for pair_id in ordered:
        vector = tuple(float(value) for value in vectors[pair_id])
        if any(_exact_dominates(member, vector)
               for _member_id, member in frontier):
            continue
        frontier = [(member_id, member)
                    for member_id, member in frontier
                    if not _exact_dominates(vector, member)]
        frontier.append((pair_id, vector))
    return sorted(pair_id for pair_id, _vector in frontier)


def _pair_scan_frontier_members(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    frontier_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Serialize one frontier's members with their six objectives."""
    members: list[dict[str, Any]] = []
    for pair_id in frontier_ids:
        row = rows_by_id[pair_id]
        members.append({
            "pair_id": pair_id,
            "objectives": dict(zip(
                PAIR_SCAN_OBJECTIVES, pair_scan_objective_vector(row))),
            "axis0_flat_row": row["axis0"]["flat_row"],
            "axis0_target_match": row["axis0"]["target_match"],
            "axis0_score_delta": row["score_delta_axis0"],
            "axis2_flat_row": row["axis2"]["flat_row"],
            "axis2_target_match": row["axis2"]["target_match"],
            "axis2_score_delta": row["score_delta_axis2"],
            "resultants": row["resultants"],
            "residuals": row["residuals"],
            "gate_moved_axes_pass": row["gate"]["moved_axes_pass"],
            "reason": row["reason"],
            "accepted": row["accepted"],
        })
    return members


def replay_r11_axes02_pair_scan(
    r10_path: Path, r11_path: Path, baseline_path: Path, c1_c5_path: Path,
    axes02_path: Path, exhaustive_path: Path, *,
    expected_r10_sha256: str = INPUT_FILE_SHA256,
    expected_r11_sha256: str = R11_INPUT_FILE_SHA256,
    expected_context_sha256: str = R11_CONTEXT_SHA256,
    expected_baseline_sha256: str = R11_BASELINE_FILE_SHA256,
    expected_baseline_seal_sha256: str = R11_BASELINE_SEAL_SHA256,
    expected_c1_c5_sha256: str = R11_C1_C5_FILE_SHA256,
    expected_c1_c5_seal_sha256: str = R11_C1_C5_SEAL_SHA256,
    expected_axes02_sha256: str = R11_AXES02_FILE_SHA256,
    expected_axes02_seal_sha256: str = R11_AXES02_SEAL_SHA256,
    expected_exhaustive_sha256: str = R11_AXES02_EXHAUSTIVE_FILE_SHA256,
    expected_exhaustive_seal_sha256: str = R11_AXES02_EXHAUSTIVE_SEAL_SHA256,
    tolerance: float = CONTROL_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Scan the exact 6,068 axis-0 x axis-2 pairs, once each, under pinned R11.

    Ordering is deliberate.  All six sealed artifacts are pinned by file and
    seal, the accepted baseline/C1-C5/axes-0/2/exhaustive replays are reconciled
    though the sealed C3 provenance and exhaustive cross-binds, the R10/R11
    inputs are reconstructed, and the full 6,068-pair duplicate-free
    prevalidation plus the independently re-derived pair-key digest must hold
    *before* the first of exactly 6,068 context-pinned solver calls.  No
    baseline, C1-C5, C3, axes-0/2 or exhaustive arm is rebuilt or rerun and no
    call is retried.
    """
    baseline = load_r11_baseline(
        baseline_path, expected_sha256=expected_baseline_sha256,
        expected_seal_sha256=expected_baseline_seal_sha256)
    baseline_file_sha256 = sha256_file(Path(baseline_path)).upper()

    c1_c5 = load_r11_c1_c5(
        c1_c5_path, expected_sha256=expected_c1_c5_sha256,
        expected_seal_sha256=expected_c1_c5_seal_sha256)
    c1_c5_file_sha256 = sha256_file(Path(c1_c5_path)).upper()

    axes02 = load_r11_axes02(
        axes02_path, expected_sha256=expected_axes02_sha256,
        expected_seal_sha256=expected_axes02_seal_sha256)
    axes02_file_sha256 = sha256_file(Path(axes02_path)).upper()

    exhaustive = load_r11_axes02_exhaustive(
        exhaustive_path, expected_sha256=expected_exhaustive_sha256,
        expected_seal_sha256=expected_exhaustive_seal_sha256)
    exhaustive_file_sha256 = sha256_file(Path(exhaustive_path)).upper()

    r11_path = Path(r11_path)
    if not r11_path.is_file():
        raise ReplayValidationError(f"pinned R11 artifact is absent: {r11_path}")
    observed_r11_sha256 = sha256_file(r11_path).upper()
    if observed_r11_sha256 != expected_r11_sha256.upper():
        raise ReplayValidationError(
            "R11 artifact file SHA-256 does not match the pinned value")
    with open(r11_path, "r", encoding="utf-8") as handle:
        r11_artifact = json.load(handle)
    if r11_artifact.get("schema") != INPUT_SCHEMA:
        raise ReplayValidationError("R11 artifact schema is not the stage schema")

    reconstruction = load_r10_reconstruction(
        r10_path, expected_sha256=expected_r10_sha256)
    context = exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames,
        expected_context_sha256=expected_context_sha256)

    reconciliation = reconcile_arms_inputs(
        reconstruction, r11_artifact, context, baseline,
        r11_file_sha256=observed_r11_sha256)

    # Consume-only cross-binds of the accepted C1-C5, axes-0/2 and exhaustive
    # replays.  None of them is rebuilt or rerun.
    if c1_c5.get("reconciliation") != reconciliation:
        raise ReplayValidationError(
            "the sealed C1-C5 replay's reconciliation does not replay")
    if (c1_c5["validation"].get("artifact_file_sha256")
            != reconstruction.validation["artifact_file_sha256"]):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's R10 file binding does not replay")
    if (str(c1_c5["inputs"]["r10"].get("record_sha256", ""))
            != str(reconstruction.record.get("record_sha256"))):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's R10 record binding does not replay")
    if c1_c5["inputs"]["r11"].get("solve_context") != context.to_report():
        raise ReplayValidationError(
            "the sealed C1-C5 replay's solve-context binding does not replay")
    if (str(c1_c5["inputs"]["r11_baseline"].get("file_sha256", "")).upper()
            != baseline_file_sha256):
        raise ReplayValidationError(
            "the sealed C1-C5 replay's baseline file binding does not replay")

    if axes02.get("reconciliation") != reconciliation:
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's reconciliation does not replay")
    if (axes02["validation"].get("artifact_file_sha256")
            != reconstruction.validation["artifact_file_sha256"]):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's R10 file binding does not replay")
    if (str(axes02["inputs"]["r10"].get("record_sha256", ""))
            != str(reconstruction.record.get("record_sha256"))):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's R10 record binding does not replay")
    if axes02["inputs"]["r11"].get("solve_context") != context.to_report():
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's solve-context binding does not replay")
    if (str(axes02["inputs"]["r11_baseline"].get("file_sha256", "")).upper()
            != baseline_file_sha256):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's baseline binding does not replay")
    if (str(axes02["inputs"]["r11_c1_c5"].get("file_sha256", "")).upper()
            != c1_c5_file_sha256):
        raise ReplayValidationError(
            "the sealed axes-0/2 replay's C1-C5 file binding does not replay")

    if exhaustive.get("reconciliation") != reconciliation:
        raise ReplayValidationError(
            "the sealed exhaustive replay's reconciliation does not replay")
    if (exhaustive["validation"].get("artifact_file_sha256")
            != reconstruction.validation["artifact_file_sha256"]):
        raise ReplayValidationError(
            "the sealed exhaustive replay's R10 file binding does not replay")
    if (str(exhaustive["inputs"]["r10"].get("record_sha256", ""))
            != str(reconstruction.record.get("record_sha256"))):
        raise ReplayValidationError(
            "the sealed exhaustive replay's R10 record binding does not replay")
    if exhaustive["inputs"]["r11"].get("solve_context") != context.to_report():
        raise ReplayValidationError(
            "the sealed exhaustive replay's solve-context binding does not "
            "replay")
    if (str(exhaustive["inputs"]["r11_baseline"].get("file_sha256", "")).upper()
            != baseline_file_sha256):
        raise ReplayValidationError(
            "the sealed exhaustive replay's baseline binding does not replay")
    if (str(exhaustive["inputs"]["r11_c1_c5"].get("file_sha256", "")).upper()
            != c1_c5_file_sha256):
        raise ReplayValidationError(
            "the sealed exhaustive replay's C1-C5 binding does not replay")
    if (str(exhaustive["inputs"]["r11_axes02"].get("file_sha256", "")).upper()
            != axes02_file_sha256):
        raise ReplayValidationError(
            "the sealed exhaustive replay's axes-0/2 binding does not replay")

    sealed_c3 = sealed_c3_reference(c1_c5)
    frozen_c3 = sorted(
        (int(entry["candidate_flat_row"]), int(entry["candidate_group"]),
         int(entry["replaces_match_index"]))
        for entry in sealed_c3["provenance"])
    expected_c3 = sorted((int(row), int(group), int(target))
                         for row, target, group, _axis in C3_SUBSTITUTIONS)
    if frozen_c3 != expected_c3:
        raise ReplayValidationError(
            "the sealed C3 arm is not the frozen C3 substitution")
    axes02_c3 = axes02.get("sealed_c3")
    for key in ("reason", "assignment", "resultants", "residuals",
                "observations", "active_match_count", "active_match_indices",
                "duplicate_b", "translations"):
        if axes02_c3.get(key) != sealed_c3["outcome"].get(key):
            raise ReplayValidationError(
                "the sealed axes-0/2 C3 reference does not replay the accepted "
                f"C3 outcome ({key})")
    exhaustive_c3 = exhaustive.get("sealed_c3")
    for key in ("reason", "assignment", "resultants", "residuals",
                "observations", "active_match_count", "active_match_indices",
                "duplicate_b", "translations"):
        if exhaustive_c3.get(key) != sealed_c3["outcome"].get(key):
            raise ReplayValidationError(
                "the sealed exhaustive C3 reference does not replay the "
                f"accepted C3 outcome ({key})")

    frames = tuple(reconstruction.universe.frames)
    outcome_c3 = sealed_c3["outcome"]
    assignment = list(outcome_c3.get("assignment") or [])
    if len(assignment) != len(frames):
        raise ReplayValidationError(
            "the sealed C3 outcome assignment is not frame-aligned")
    quadrants = {frame: int(value) for frame, value in zip(frames, assignment)}

    c3_replacements, _c3_provenance = plan_combined_arm(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        quadrants, C3_SUBSTITUTIONS, label="C3")
    plans = plan_exhaustive_arms(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        frames, quadrants, c3_replacements)
    pair_plan = plan_pair_scan(
        plans, reconstruction.matches, frames, quadrants, c3_replacements)

    # The accepted exhaustive run's own recorded pair digest must equal the
    # independent re-derivation before any solver call.
    if (str(exhaustive["prevalidation"]["pair_prevalidation"][
            "valid_pairs_sha256"]).lower() != pair_plan["digest"]):
        raise ReplayValidationError(
            "the accepted exhaustive replay's pair-key digest does not replay "
            "against the pair scan's independent re-derivation")

    rows = build_pair_scan_rows(
        reconstruction.universe.readings, reconstruction.matches, frames,
        context, c3_replacements, quadrants, pair_plan["pairs"])
    if len(rows) != PAIR_SCAN_EXPECTED_PAIRS:
        raise ReplayValidationError(
            "the pair scan did not execute the frozen 6,068 calls")

    rows_by_id = {row["pair_id"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ReplayValidationError("the pair scan's pair ids repeat")

    reference_id = (
        f"A{PAIR_SCAN_ACCEPTED_REFERENCE_PAIR[0]:03d}-"
        f"{PAIR_SCAN_ACCEPTED_REFERENCE_PAIR[2]:03d}")
    reference_row = rows_by_id.get(reference_id)
    if reference_row is None:
        raise ReplayValidationError(
            "the accepted A020-308 pair is absent from the pair scan")
    if (reference_row["axis0"]["flat_row"],
            reference_row["axis0"]["target_match"],
            reference_row["axis2"]["flat_row"],
            reference_row["axis2"]["target_match"]) != (
                PAIR_SCAN_ACCEPTED_REFERENCE_PAIR):
        raise ReplayValidationError(
            "the accepted A020-308 pair does not replay its frozen key")
    sealed_final_arm = exhaustive["selection"]["final_arm"]
    sealed_final_outcome = sealed_final_arm["outcome"]
    reference_problems: list[str] = []
    for key in ("reason", "assignment", "resultants", "residuals",
                "observations", "active_match_count", "duplicate_b"):
        if reference_row.get(key) != sealed_final_outcome.get(key):
            reference_problems.append(
                f"{key} {reference_row.get(key)!r} != "
                f"{sealed_final_outcome.get(key)!r}")
    if reference_problems:
        raise ReplayValidationError(
            "the in-scan A020-308 outcome contradicts the sealed exhaustive "
            f"final arm: {reference_problems}")

    axis1_values = sorted({
        (row["resultants"]["1"], row["residuals"]["1"]) for row in rows})
    assignment_variants = sorted({
        tuple(row["assignment"]) for row in rows})
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason_counts[str(row["reason"])] = reason_counts.get(
            str(row["reason"]), 0) + 1
    accepted_gate = [row for row in rows if row["gate"]["moved_axes_pass"]]
    accepted_solver = [row for row in rows if row["accepted"]]
    duplicate_rows = [row for row in rows if row["duplicate_b"]]

    vectors = {row["pair_id"]: pair_scan_objective_vector(row) for row in rows}
    frontier_all = pareto_frontier(vectors)
    frontier_accepted_gate = pareto_frontier({
        row["pair_id"]: vectors[row["pair_id"]] for row in accepted_gate})
    frontier_accepted_solver = pareto_frontier({
        row["pair_id"]: vectors[row["pair_id"]] for row in accepted_solver})

    highest = min(
        rows,
        key=lambda row: (
            -(row["axis0"]["candidate_score"]
              + row["axis2"]["candidate_score"]),
            -row["axis0"]["candidate_score"],
            -row["axis2"]["candidate_score"],
            row["pair_id"]))

    total = float(PAIR_SCAN_EXPECTED_PAIRS)
    summary = {
        "primary_result": PAIR_SCAN_PRIMARY_RESULT,
        "accepted_definition": PAIR_SCAN_ACCEPTED_DEFINITION,
        "accepted_gate_pair_count": len(accepted_gate),
        "accepted_gate_pair_fraction": len(accepted_gate) / total,
        "accepted_solver_pair_count": len(accepted_solver),
        "accepted_solver_pair_fraction": len(accepted_solver) / total,
        "axis0_gate_pass_pair_count": sum(
            1 for row in rows if row["gate"]["axis0_pass"]),
        "axis2_gate_pass_pair_count": sum(
            1 for row in rows if row["gate"]["axis2_pass"]),
        "all_axes_gate_pass_pair_count": sum(
            1 for row in rows if row["gate"]["all_axes_pass"]),
        "duplicate_pair_count": len(duplicate_rows),
        "duplicate_pair_ids": [row["pair_id"] for row in duplicate_rows],
        "reason_counts": dict(sorted(reason_counts.items())),
        "assignment_variants": [list(value) for value in assignment_variants],
        "axis1_constant_across_all_pairs": len(axis1_values) == 1,
        "axis1_values": [{"resultant": value[0], "residual": value[1]}
                         for value in axis1_values],
        "sealed_c3_axis1_resultant": float(
            outcome_c3["resultants"]["1"]),
        "sealed_c3_axis1_residual": float(outcome_c3["residuals"]["1"]),
        "frontier_all_size": len(frontier_all),
        "frontier_all_fraction": len(frontier_all) / total,
        "frontier_accepted_gate_size": len(frontier_accepted_gate),
        "frontier_accepted_gate_fraction": (
            len(frontier_accepted_gate) / total),
        "frontier_accepted_solver_size": len(frontier_accepted_solver),
        "frontier_accepted_solver_fraction": (
            len(frontier_accepted_solver) / total),
        "highest_score_pair_id": highest["pair_id"],
        "highest_score_pair_axis0_score": highest["axis0"]["candidate_score"],
        "highest_score_pair_axis2_score": highest["axis2"]["candidate_score"],
        "highest_score_pair_total_score": (
            highest["axis0"]["candidate_score"]
            + highest["axis2"]["candidate_score"]),
        "accepted_reference_pair_id": reference_id,
        "accepted_reference_pair_gate_moved_axes_pass": (
            reference_row["gate"]["moved_axes_pass"]),
        "accepted_reference_pair_solver_accepted": reference_row["accepted"],
        "pareto_rule": PAIR_SCAN_PARETO_RULE,
        "superseded_ranking_rule": PAIR_SCAN_SUPERSEDED_RANKING_RULE,
    }

    validation = dict(reconstruction.validation)
    validation["r11_artifact_file_sha256"] = observed_r11_sha256
    validation["r11_context_replay"] = "pass"
    validation["r11_context_frame_binding"] = "pass"
    validation["r10_r11_reconciliation"] = "pass"
    validation["sealed_baseline_reconciliation_replays"] = "pass"
    validation["sealed_control_reproduced_recorded_outcome"] = "pass"
    validation["sealed_c1_c5_reconciliation_replays"] = "pass"
    validation["sealed_axes02_reconciliation_replays"] = "pass"
    validation["sealed_exhaustive_reconciliation_replays"] = "pass"
    validation["sealed_c3_provenance_replays"] = "pass"
    validation["pair_scan_prevalidation_reproduces_6068"] = "pass"
    validation["all_6068_pairs_duplicate_free_before_any_solver"] = "pass"
    validation["pair_key_digest_re_derived_and_matched"] = "pass"
    validation["all_6068_calls_context_pinned_once"] = "pass"
    validation["accepted_reference_pair_cross_checked_against_sealed_final_arm"] = (
        "pass")
    validation["baseline_rerun"] = False
    validation["c3_rerun"] = False
    validation["axes02_rerun"] = False
    validation["exhaustive_rerun"] = False
    validation["c1_c5_rerun"] = False

    payload: dict[str, Any] = {
        "schema": PAIR_SCAN_SCHEMA,
        "task_id": PAIR_SCAN_TASK_ID,
        "status": "complete",
        "no_truth_disclaimer": NO_TRUTH_DISCLAIMER,
        "inputs": {
            "r10": {
                "path": str(r10_path),
                "file_sha256": validation["artifact_file_sha256"],
                "expected_file_sha256": expected_r10_sha256.upper(),
                "artifact_sha256": reconstruction.artifact.get(
                    "artifact_sha256"),
                "observation_sha256": reconstruction.artifact[
                    "solver_failure_observation"].get("observation_sha256"),
                "record_sha256": reconstruction.record.get("record_sha256"),
                "failed_assignment_sha256": reconstruction.record.get(
                    "failed_assignment_sha256"),
                "input_match_identity_sha256": reconstruction.record.get(
                    "input_match_identity_sha256"),
            },
            "r11": {
                "path": str(r11_path),
                "file_sha256": observed_r11_sha256,
                "expected_file_sha256": expected_r11_sha256.upper(),
                "artifact_sha256": r11_artifact.get("artifact_sha256"),
                "entry_sha256": r11_artifact.get("entry_sha256"),
                "observation_sha256": r11_artifact[
                    "solver_failure_observation"].get("observation_sha256"),
                "solve_context": context.to_report(),
            },
            "r11_baseline": {
                "path": str(baseline_path),
                "file_sha256": baseline_file_sha256,
                "expected_file_sha256": expected_baseline_sha256.upper(),
                "seal_sha256": baseline.get("seal_sha256"),
                "expected_seal_sha256": expected_baseline_seal_sha256.lower(),
            },
            "r11_c1_c5": {
                "path": str(c1_c5_path),
                "file_sha256": c1_c5_file_sha256,
                "expected_file_sha256": expected_c1_c5_sha256.upper(),
                "seal_sha256": c1_c5.get("seal_sha256"),
                "expected_seal_sha256": expected_c1_c5_seal_sha256.lower(),
                "task_id": c1_c5.get("task_id"),
            },
            "r11_axes02": {
                "path": str(axes02_path),
                "file_sha256": axes02_file_sha256,
                "expected_file_sha256": expected_axes02_sha256.upper(),
                "seal_sha256": axes02.get("seal_sha256"),
                "expected_seal_sha256": expected_axes02_seal_sha256.lower(),
                "task_id": axes02.get("task_id"),
            },
            "r11_axes02_exhaustive": {
                "path": str(exhaustive_path),
                "file_sha256": exhaustive_file_sha256,
                "expected_file_sha256": expected_exhaustive_sha256.upper(),
                "seal_sha256": exhaustive.get("seal_sha256"),
                "expected_seal_sha256": (
                    expected_exhaustive_seal_sha256.lower()),
                "task_id": exhaustive.get("task_id"),
            },
        },
        "reconciliation": reconciliation,
        "validation": validation,
        "sealed_c3": {
            "file_sha256": c1_c5_file_sha256,
            "seal_sha256": c1_c5.get("seal_sha256"),
            "reason": outcome_c3.get("reason"),
            "assignment": outcome_c3.get("assignment"),
            "accepted": outcome_c3.get("accepted"),
            "resultants": outcome_c3.get("resultants"),
            "residuals": outcome_c3.get("residuals"),
            "observations": outcome_c3.get("observations"),
            "active_match_count": outcome_c3.get("active_match_count"),
            "active_match_indices": outcome_c3.get("active_match_indices"),
            "duplicate_b": outcome_c3.get("duplicate_b"),
            "provenance": sealed_c3["provenance"],
        },
        "prevalidation": {
            "exhaustive_substitutions": plans["report"],
            "pair_scan": pair_plan["report"],
        },
        "pareto": {
            "objectives": list(PAIR_SCAN_OBJECTIVES),
            "rule": PAIR_SCAN_PARETO_RULE,
            "axis_objectives": PAIR_SCAN_AXIS_OBJECTIVES,
            "dominance": (
                "exact componentwise six-dimensional dominance, no tolerance "
                "and at least one strictly better objective; equal vectors do "
                "not dominate each other"),
            "accepted_definition": PAIR_SCAN_ACCEPTED_DEFINITION,
            "frontier_all": {
                "size": len(frontier_all),
                "fraction": len(frontier_all) / total,
                "pair_ids": frontier_all,
                "members": _pair_scan_frontier_members(
                    rows_by_id, frontier_all),
            },
            "frontier_accepted_gate": {
                "size": len(frontier_accepted_gate),
                "fraction": len(frontier_accepted_gate) / total,
                "accepted_count": len(accepted_gate),
                "pair_ids": frontier_accepted_gate,
                "members": _pair_scan_frontier_members(
                    rows_by_id, frontier_accepted_gate),
            },
            "frontier_accepted_solver": {
                "size": len(frontier_accepted_solver),
                "fraction": len(frontier_accepted_solver) / total,
                "accepted_count": len(accepted_solver),
                "pair_ids": frontier_accepted_solver,
                "members": _pair_scan_frontier_members(
                    rows_by_id, frontier_accepted_solver),
            },
        },
        "highest_score_pair": {
            "definition": (
                "globally highest total captured candidate score "
                "(axis-0 candidate score + axis-2 candidate score), tie-broken "
                "by axis-0 score, axis-2 score and pair id; correlation-score "
                "proxy only, never truth probability and never a selected "
                "winner"),
            "row": highest,
        },
        "accepted_reference_pair": {
            "label": PAIR_SCAN_ACCEPTED_REFERENCE_LABEL,
            "frozen_key": list(PAIR_SCAN_ACCEPTED_REFERENCE_PAIR),
            "row": reference_row,
            "matches_sealed_exhaustive_final_arm": True,
            "sealed_final_arm_outcome": {
                key: sealed_final_outcome.get(key)
                for key in ("reason", "assignment", "resultants", "residuals",
                            "observations", "active_match_count",
                            "duplicate_b")},
            "sealed_exhaustive_file_sha256": exhaustive_file_sha256,
        },
        "pairs": rows,
        "pair_order": [row["pair_id"] for row in rows],
        "summary": summary,
        "solver_calls_authorized": PAIR_SCAN_AUTHORIZED_SOLVER_CALLS,
        "solver_calls_executed": len(rows),
        "solver_calls_expected": PAIR_SCAN_EXPECTED_PAIRS,
        "retries": 0,
        "baseline_rerun": False,
        "c3_rerun": False,
        "axes02_rerun": False,
        "exhaustive_rerun": False,
        "c1_c5_rerun": False,
        "notes": [
            "Six sealed artifacts were pinned and reconciled before any "
            "treatment: R10 capture, R11 capture, the accepted R11 baseline "
            "control, the accepted C1-C5 arm replay, the accepted axes-0/2+C3 "
            "replay and the accepted exhaustive axes-0/2 replay.  None of them "
            "and not the sealed C3 arm was rebuilt or rerun.",
            "The exact 164 axis-0 x 37 axis-2 = 6,068 combined substitutions "
            "(each adding its one-for-one axis-0 and axis-2 replacement to the "
            "sealed C3 rows) were enumerated solver-free, every combined active "
            "edge set was proven duplicate-free, and the sorted pair-key digest "
            "was independently re-derived and required to equal both the pinned "
            "value and the digest the accepted exhaustive artifact recorded -- "
            "all before the first solver call.",
            "Exactly 6,068 context-pinned solver calls then ran once each under "
            "the identical captured R11 thresholds (.15/.90/.02) and ordered "
            "quadrant constraints; there were no retries and no baseline, "
            "C1-C5, C3, axes-0/2 or exhaustive rerun.",
            "Reporting uses the definitive exact six-dimensional per-axis "
            "Pareto contract: maximise both score deltas and both resultants, "
            "minimise both residuals (equivalently maximise the unchanged "
            "resultant - .90 and .15 - residual gate margins per moved axis).  "
            "Both the all-arm frontier and the named solver-accepted accepted-arm "
            "frontier frontier_accepted_solver are reported, with the "
            "gate-accepted frontier frontier_accepted_gate preserved separately "
            "as a diagnostic; there is no pooled score, no weighted scalar and no "
            "selected or promoted winner.",
            "The globally highest-correlation-score pair and the already "
            "accepted A020-308 pair are identified in-scan; A020-308's measured "
            "outcome is cross-checked field-by-field against the sealed "
            "exhaustive final arm.",
            "Axis 1 is the sealed C3 row and is constant; its resultants and "
            "residuals are reported separately.  The .90/.15 gates remain "
            "per-axis diagnostics only, never an accuracy claim.",
        ],
    }
    payload["seal_sha256"] = seal_payload(payload)
    return payload


def write_sealed(output_path: Path, payload: Mapping[str, Any]) -> Path:
    """Write the sealed payload to a fresh output root, fail-closed."""
    output_path = Path(output_path)
    root = output_path.parent
    if root.exists():
        raise ReplayValidationError(
            f"output root already exists and must be fresh: {root}")
    root.mkdir(parents=True)
    output_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=("r10-replay", "r11-control", "r11-arms",
                                 "r11-axes02-c3", "r11-axes02-exhaustive",
                                 "r11-axes02-pair-scan"),
                        default="r10-replay")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--r11-input", default=str(DEFAULT_INPUT_R11))
    parser.add_argument("--r11-baseline-input",
                        default=str(DEFAULT_INPUT_R11_BASELINE))
    parser.add_argument("--r11-c1-c5-input",
                        default=str(DEFAULT_INPUT_R11_C1_C5))
    parser.add_argument("--r11-axes02-input",
                        default=str(DEFAULT_INPUT_R11_AXES02))
    parser.add_argument("--r11-axes02-exhaustive-input",
                        default=str(DEFAULT_INPUT_R11_AXES02_EXHAUSTIVE))
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-sha256", default=INPUT_FILE_SHA256)
    parser.add_argument("--expected-r11-sha256", default=R11_INPUT_FILE_SHA256)
    parser.add_argument("--expected-context-sha256", default=R11_CONTEXT_SHA256)
    parser.add_argument("--expected-baseline-sha256",
                        default=R11_BASELINE_FILE_SHA256)
    parser.add_argument("--expected-baseline-seal-sha256",
                        default=R11_BASELINE_SEAL_SHA256)
    parser.add_argument("--expected-c1-c5-sha256",
                        default=R11_C1_C5_FILE_SHA256)
    parser.add_argument("--expected-c1-c5-seal-sha256",
                        default=R11_C1_C5_SEAL_SHA256)
    parser.add_argument("--expected-axes02-sha256",
                        default=R11_AXES02_FILE_SHA256)
    parser.add_argument("--expected-axes02-seal-sha256",
                        default=R11_AXES02_SEAL_SHA256)
    parser.add_argument("--expected-exhaustive-sha256",
                        default=R11_AXES02_EXHAUSTIVE_FILE_SHA256)
    parser.add_argument("--expected-exhaustive-seal-sha256",
                        default=R11_AXES02_EXHAUSTIVE_SEAL_SHA256)
    parser.add_argument("--tolerance", type=float, default=CONTROL_ABS_TOLERANCE)
    arguments = parser.parse_args(argv)

    if arguments.mode == "r11-axes02-pair-scan":
        destination_path = (Path(arguments.output) if arguments.output
                            else DEFAULT_OUTPUT_PAIR_SCAN)
        payload = replay_r11_axes02_pair_scan(
            Path(arguments.input), Path(arguments.r11_input),
            Path(arguments.r11_baseline_input),
            Path(arguments.r11_c1_c5_input),
            Path(arguments.r11_axes02_input),
            Path(arguments.r11_axes02_exhaustive_input),
            expected_r10_sha256=arguments.expected_sha256,
            expected_r11_sha256=arguments.expected_r11_sha256,
            expected_context_sha256=arguments.expected_context_sha256,
            expected_baseline_sha256=arguments.expected_baseline_sha256,
            expected_baseline_seal_sha256=(
                arguments.expected_baseline_seal_sha256),
            expected_c1_c5_sha256=arguments.expected_c1_c5_sha256,
            expected_c1_c5_seal_sha256=arguments.expected_c1_c5_seal_sha256,
            expected_axes02_sha256=arguments.expected_axes02_sha256,
            expected_axes02_seal_sha256=(
                arguments.expected_axes02_seal_sha256),
            expected_exhaustive_sha256=arguments.expected_exhaustive_sha256,
            expected_exhaustive_seal_sha256=(
                arguments.expected_exhaustive_seal_sha256),
            tolerance=arguments.tolerance)
        destination = write_sealed(destination_path, payload)
        summary = {
            "schema": payload["schema"],
            "task_id": payload["task_id"],
            "status": payload["status"],
            "output": str(destination),
            "seal_sha256": payload["seal_sha256"],
            "solver_calls_authorized": payload["solver_calls_authorized"],
            "solver_calls_executed": payload["solver_calls_executed"],
            "solver_calls_expected": payload["solver_calls_expected"],
            "retries": payload["retries"],
            "baseline_rerun": payload["baseline_rerun"],
            "c3_rerun": payload["c3_rerun"],
            "axes02_rerun": payload["axes02_rerun"],
            "exhaustive_rerun": payload["exhaustive_rerun"],
            "prevalidation": payload["prevalidation"]["pair_scan"],
            "summary": payload["summary"],
            "pareto_frontiers": {
                "all": payload["pareto"]["frontier_all"]["pair_ids"],
                "accepted_gate": payload["pareto"][
                    "frontier_accepted_gate"]["pair_ids"],
                "accepted_solver": payload["pareto"][
                    "frontier_accepted_solver"]["pair_ids"],
            },
            "highest_score_pair_id": payload["summary"][
                "highest_score_pair_id"],
            "accepted_reference_pair_id": payload["summary"][
                "accepted_reference_pair_id"],
        }
        print(canonical_json(summary))
        return 0

    if arguments.mode == "r11-axes02-exhaustive":
        destination_path = (Path(arguments.output) if arguments.output
                            else DEFAULT_OUTPUT_AXES02_EXHAUSTIVE)
        payload = replay_r11_axes02_exhaustive(
            Path(arguments.input), Path(arguments.r11_input),
            Path(arguments.r11_baseline_input),
            Path(arguments.r11_c1_c5_input),
            Path(arguments.r11_axes02_input),
            expected_r10_sha256=arguments.expected_sha256,
            expected_r11_sha256=arguments.expected_r11_sha256,
            expected_context_sha256=arguments.expected_context_sha256,
            expected_baseline_sha256=arguments.expected_baseline_sha256,
            expected_baseline_seal_sha256=(
                arguments.expected_baseline_seal_sha256),
            expected_c1_c5_sha256=arguments.expected_c1_c5_sha256,
            expected_c1_c5_seal_sha256=arguments.expected_c1_c5_seal_sha256,
            expected_axes02_sha256=arguments.expected_axes02_sha256,
            expected_axes02_seal_sha256=(
                arguments.expected_axes02_seal_sha256),
            tolerance=arguments.tolerance)
        destination = write_sealed(destination_path, payload)
        summary = {
            "schema": payload["schema"],
            "task_id": payload["task_id"],
            "status": payload["status"],
            "output": str(destination),
            "seal_sha256": payload["seal_sha256"],
            "solver_calls_authorized": payload["solver_calls_authorized"],
            "solver_calls_executed": payload["solver_calls_executed"],
            "solver_calls_expected": payload["solver_calls_expected"],
            "baseline_rerun": payload["baseline_rerun"],
            "c3_rerun": payload["c3_rerun"],
            "prevalidation": {
                "observed_axis_counts": payload["prevalidation"][
                    "observed_axis_counts"],
                "observed_total": payload["prevalidation"]["observed_total"],
                "pair_prevalidation": payload["prevalidation"][
                    "pair_prevalidation"],
            },
            "selection": {
                "best_axis0_arm": payload["selection"]["best_axis0_arm"],
                "best_axis2_arm": payload["selection"]["best_axis2_arm"],
                "pair_key_prevalidated": payload["selection"][
                    "pair_key_prevalidated"],
                "best_axis0": payload["selection"]["best_axis0"],
                "best_axis2": payload["selection"]["best_axis2"],
                "final_arm_summary": payload["selection"][
                    "final_arm_summary"],
            },
            "sealed_c3": {
                "reason": payload["sealed_c3"]["reason"],
                "assignment": payload["sealed_c3"]["assignment"],
                "resultants": payload["sealed_c3"]["resultants"],
                "residuals": payload["sealed_c3"]["residuals"],
                "observations": payload["sealed_c3"]["observations"],
            },
            "threshold_passes": payload["summary"]["threshold_passes"],
        }
        print(canonical_json(summary))
        return 0

    if arguments.mode == "r11-axes02-c3":
        destination_path = (Path(arguments.output) if arguments.output
                            else DEFAULT_OUTPUT_AXES02_C3)
        payload = replay_r11_axes02_c3(
            Path(arguments.input), Path(arguments.r11_input),
            Path(arguments.r11_baseline_input),
            Path(arguments.r11_c1_c5_input),
            expected_r10_sha256=arguments.expected_sha256,
            expected_r11_sha256=arguments.expected_r11_sha256,
            expected_context_sha256=arguments.expected_context_sha256,
            expected_baseline_sha256=arguments.expected_baseline_sha256,
            expected_baseline_seal_sha256=(
                arguments.expected_baseline_seal_sha256),
            expected_c1_c5_sha256=arguments.expected_c1_c5_sha256,
            expected_c1_c5_seal_sha256=arguments.expected_c1_c5_seal_sha256,
            tolerance=arguments.tolerance)
        destination = write_sealed(destination_path, payload)
        summary = {
            "schema": payload["schema"],
            "task_id": payload["task_id"],
            "status": payload["status"],
            "output": str(destination),
            "seal_sha256": payload["seal_sha256"],
            "solver_calls_executed": payload["solver_calls_executed"],
            "baseline_rerun": payload["baseline_rerun"],
            "c3_rerun": payload["c3_rerun"],
            "sealed_c3": {
                "reason": payload["sealed_c3"]["reason"],
                "assignment": payload["sealed_c3"]["assignment"],
                "resultants": payload["sealed_c3"]["resultants"],
                "residuals": payload["sealed_c3"]["residuals"],
                "observations": payload["sealed_c3"]["observations"],
            },
            "arms": {
                name: {
                    "frozen_axes02_rows": arm["frozen_axes02_rows"],
                    "reason": arm["outcome"]["reason"],
                    "assignment": arm["outcome"]["assignment"],
                    "resultants": arm["outcome"]["resultants"],
                    "residuals": arm["outcome"]["residuals"],
                    "observations": arm["outcome"]["observations"],
                    "active_match_count": arm["outcome"]["active_match_count"],
                    "duplicate_b": arm["outcome"]["duplicate_b"],
                    "diagnostics": arm["diagnostics"]["per_axis"],
                    "deltas_vs_sealed_c3": arm["deltas_vs_sealed_c3"],
                }
                for name, arm in payload["arms"].items()},
        }
        print(canonical_json(summary))
        return 0

    if arguments.mode == "r11-arms":
        destination_path = (Path(arguments.output) if arguments.output
                            else DEFAULT_OUTPUT_ARMS)
        payload = replay_r11_arms(
            Path(arguments.input), Path(arguments.r11_input),
            Path(arguments.r11_baseline_input),
            expected_r10_sha256=arguments.expected_sha256,
            expected_r11_sha256=arguments.expected_r11_sha256,
            expected_context_sha256=arguments.expected_context_sha256,
            expected_baseline_sha256=arguments.expected_baseline_sha256,
            expected_baseline_seal_sha256=(
                arguments.expected_baseline_seal_sha256),
            tolerance=arguments.tolerance)
        destination = write_sealed(destination_path, payload)
        summary = {
            "schema": payload["schema"],
            "task_id": payload["task_id"],
            "status": payload["status"],
            "output": str(destination),
            "seal_sha256": payload["seal_sha256"],
            "solver_calls_executed": payload["solver_calls_executed"],
            "baseline_rerun": payload["baseline_rerun"],
            "preregistered_check": payload["preregistered_check"],
            "arms": {
                name: {
                    "reason": arm["outcome"]["reason"],
                    "assignment": arm["outcome"]["assignment"],
                    "resultants": arm["outcome"]["resultants"],
                    "residuals": arm["outcome"]["residuals"],
                    "observations": arm["outcome"]["observations"],
                    "diagnostics": arm["diagnostics"]["per_axis"],
                    "deltas_vs_sealed_control": (
                        arm["deltas_vs_sealed_control"]),
                }
                for name, arm in payload["arms"].items()},
        }
        print(canonical_json(summary))
        return 0

    if arguments.mode == "r11-control":
        destination_path = (Path(arguments.output) if arguments.output
                            else DEFAULT_OUTPUT_R11)
        payload = replay_r11_control(
            Path(arguments.input), Path(arguments.r11_input),
            expected_r10_sha256=arguments.expected_sha256,
            expected_r11_sha256=arguments.expected_r11_sha256,
            expected_context_sha256=arguments.expected_context_sha256,
            tolerance=arguments.tolerance)
        destination = write_sealed(destination_path, payload)
        summary = {
            "schema": payload["schema"],
            "task_id": payload["task_id"],
            "status": payload["status"],
            "output": str(destination),
            "seal_sha256": payload["seal_sha256"],
            "control_gate": payload["control_gate"],
            "control": payload["control"],
            "solver_calls_executed": payload["solver_calls_executed"],
            "treatments": payload["treatments"],
        }
        print(canonical_json(summary))
        return 0 if payload["status"] == "complete" else 2

    destination_path = (Path(arguments.output) if arguments.output
                        else DEFAULT_OUTPUT)
    payload = replay_artifact(
        Path(arguments.input), expected_sha256=arguments.expected_sha256,
        tolerance=arguments.tolerance)
    destination = write_sealed(destination_path, payload)
    summary = {
        "schema": payload["schema"],
        "status": payload["status"],
        "output": str(destination),
        "seal_sha256": payload["seal_sha256"],
        "control_gate": payload["control_gate"],
        "pinned_baseline": payload["pinned_baseline"],
    }
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
