"""Evaluation-only Phase A template-match progress scorecard and gates.

This module owns the **pure** Phase A measurement contract for the bounded
ordinary-image Template 2 Route-B diagnostic:

* the ordered T2 source pin profile (the three accepted screenshot byte
  SHA-256 values, taken from
  ``docs/reports/image-only-route-b-stage-loss-run-2026-10-04.md:41-43``);
* the five exact terminal states and the refusal-to-terminal classifier;
* the five resource limits the run harness must enforce before it launches
  (these are deliberately *not* the 512 MiB sidecar allowance);
* the separate measurement universes - candidate rows, logical cell pairs,
  unique ``(frame_id, source_pixel_index)`` dependencies with per-frame counts
  and union coverage, and per-row/per-cell footprint availability;
* the UNKNOWN sentinel policy (a missing payload or the sampler's ``-1`` border
  sentinel is UNKNOWN, never zero and never a fabricated dependency); and
* the freeze-before-evaluator gate and the Phase B source-bound evidence gate.

It performs no inference, no evaluator/truth/pixel access, no file I/O and no
NumPy allocation. Availability, coverage and union counts computed here are
availability/coverage counters, **not** accuracy, correctness or ownership
scores; no percentage is an accuracy percentage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = [
    "SCHEMA",
    "FREEZE_SCHEMA",
    "PHASE_B_AUTHORIZATION_SCHEMA",
    "PHASE_B_ALIGNMENT_UNAVAILABLE",
    "T2_PIN_REFERENCE",
    "T2_PIN_PROFILE",
    "T2_PIN_SHA256S",
    "T2_INGRESS_SOURCE_SIDE",
    "TERMINAL_CAPTURED_SOLVER_ABSTENTION",
    "TERMINAL_ASSEMBLED_NO_SUPPORT_CAPTURE",
    "TERMINAL_EARLY_REFUSAL_NO_CAPTURE",
    "TERMINAL_PROFILE_DRIFT_NO_CAPTURE",
    "TERMINAL_RESOURCE_ABORTED_NO_CAPTURE",
    "TERMINAL_STATES",
    "SOLVER_ASSEMBLED_NO_CAPTURE",
    "LEGACY_CAPTURE_REFUSAL_REASON",
    "PROFILE_DRIFT_REASONS",
    "UNKNOWN",
    "PRESENT",
    "REFUSED",
    "DISPOSITION_KEPT",
    "DISPOSITION_DROPPED",
    "PHASE_B_EVIDENCE_FIELDS",
    "PHASE_B_SOURCE_KINDS",
    "PHASE_B_MANUAL_MARKERS",
    "PHASE_B_JOIN_REQUIREMENTS",
    "SOURCE_FOOTPRINT_REASON",
    "DOWNSTREAM_METRIC_KEYS",
    "FREEZE_ARTIFACTS_BY_TERMINAL",
    "FREEZE_ARTIFACT_CAPS",
    "MAX_SCORE_ROWS",
    "MAX_PROJECTED_DEPENDENCIES",
    "MAX_LOGICAL_PAIR_ELEMENTS",
    "ResourceLimits",
    "TemplateMatchProgressError",
    "check_pin_profile",
    "classify_refusal",
    "classify_outcome",
    "candidate_row_universe",
    "logical_cell_pair_universe",
    "dependency_universe",
    "support_grid_availability",
    "source_screenshot_footprint",
    "downstream_metrics",
    "build_progress_scorecard",
    "scorecard_digest",
    "artifact_freeze_record",
    "assert_phase_a_frozen",
    "phase_b_gate",
    "phase_b_join_contract",
]

SCHEMA = "mc-photogrammetry-template-match-progress-scorecard-v2"
FREEZE_SCHEMA = "mc-photogrammetry-template-match-progress-freeze-v2"
PHASE_B_AUTHORIZATION_SCHEMA = (
    "mc-photogrammetry-template-match-progress-phase-b-authorization-v1")
# Phase B is *categorically* unavailable in this task. No real adapter joins a
# revalidated exact freeze to same-pass source-bound evidence, so the only
# possible Phase B answer is this typed refusal. A valid freeze alone never
# authorizes Phase B.
PHASE_B_ALIGNMENT_UNAVAILABLE = "phase_b_alignment_unavailable"
# The legacy route-B runner's own refusal reason when the complete-support sink
# did not fire exactly once. The wrapper (not the runner) classifies the
# assembled-without-capture terminal from the same-pass graph checkpoint.
LEGACY_CAPTURE_REFUSAL_REASON = "capture_unavailable"
# Source screenshot-footprint metrics stay null/unavailable until a persisted
# endpoint witness + CSR join exists. The grid counter below is a
# signature-grid availability counter only and is never a source footprint.
SOURCE_FOOTPRINT_REASON = "endpoint_witness_csr_join_unavailable"

# ---------------------------------------------------------------------------
# Accepted T2 source pins (report lines 41-43). These are the *byte* SHA-256
# values of the three ordinary screenshots; a path match, filename, size-only
# match or arbitrary PNG never qualifies.
# ---------------------------------------------------------------------------
T2_PIN_REFERENCE = "docs/reports/image-only-route-b-stage-loss-run-2026-10-04.md:41-43"
T2_PIN_PROFILE = (
    ("templates 2/2026-09-02_13.22.14.png",
     "eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46",
     1609624),
    ("templates 2/2026-09-02_13.24.24.png",
     "e2fd9fd6c83bf9005fe0973987da9e46d213a4351ec91393ce578f8e6d3961ae",
     1461426),
    ("templates 2/2026-09-02_13.24.33.png",
     "35d9debb206c2505f65d3be29183f4a4f3d2bf3f9c05644ce75570f48381d1fc",
     1336191),
)
T2_PIN_SHA256S = tuple(item[1] for item in T2_PIN_PROFILE)
# The accepted T2 ingress decoded side (2560x1369) is recorded so a later audit
# can see the profile the persistence source cap was split for. It is profile
# metadata only; it is never an input to the Phase A pass.
T2_INGRESS_SOURCE_SIDE = (2560, 1369)

# ---------------------------------------------------------------------------
# Exact terminal states. ``captured_solver_abstention`` is the only terminal
# that means a complete support capture exists. The other four never claim any
# capture, retry or invented inventory.
# ---------------------------------------------------------------------------
TERMINAL_CAPTURED_SOLVER_ABSTENTION = "captured_solver_abstention"
TERMINAL_ASSEMBLED_NO_SUPPORT_CAPTURE = "assembled_no_support_capture"
TERMINAL_EARLY_REFUSAL_NO_CAPTURE = "early_refusal_no_capture"
TERMINAL_PROFILE_DRIFT_NO_CAPTURE = "profile_drift_no_capture"
TERMINAL_RESOURCE_ABORTED_NO_CAPTURE = "resource_aborted_no_capture"
TERMINAL_STATES = (
    TERMINAL_CAPTURED_SOLVER_ABSTENTION,
    TERMINAL_ASSEMBLED_NO_SUPPORT_CAPTURE,
    TERMINAL_EARLY_REFUSAL_NO_CAPTURE,
    TERMINAL_PROFILE_DRIFT_NO_CAPTURE,
    TERMINAL_RESOURCE_ABORTED_NO_CAPTURE,
)

# The runner's typed reason for "the component solver assembled, so the
# complete-support sink correctly never fired". It is distinct from a genuine
# missing/duplicate capture, which stays an early refusal.
SOLVER_ASSEMBLED_NO_CAPTURE = "solver_assembled_no_capture"

# Refusals that mean the produced capture does not match the accepted frozen
# 1,928-row V2 profile. They are classified as profile drift, not as a generic
# early refusal, because a capture *was* produced but is not the accepted one.
PROFILE_DRIFT_REASONS = frozenset({
    "universe_count_mismatch",
    "row_count_mismatch",
    "universe_identity_mismatch",
    "profile_drift",
})

# ---------------------------------------------------------------------------
# UNKNOWN / availability sentinels and disposition names.
# ---------------------------------------------------------------------------
UNKNOWN = "unknown"
PRESENT = "present"
REFUSED = "refused"
DISPOSITION_KEPT = "kept"
DISPOSITION_DROPPED = "dropped"

# Bounded scorecard traversal budgets. A universe that cannot be walked inside
# its budget is reported UNKNOWN with a typed reason, never partially summed and
# never labelled complete.
MAX_SCORE_ROWS = 100_000
MAX_PROJECTED_DEPENDENCIES = 6_000_000
MAX_LOGICAL_PAIR_ELEMENTS = 6_000_000

# ---------------------------------------------------------------------------
# Phase B evidence contract. Phase B stays coded but gated: only actual
# source-bound measurement evidence for every field authorizes it. Digest-only
# or pin fields are insufficient, and manual F3/coordinate/direction/bounds
# metadata never qualifies.
# ---------------------------------------------------------------------------
PHASE_B_EVIDENCE_FIELDS = (
    "camera", "intrinsics", "pose", "transform", "world_domain",
    "occlusion", "ownership",
)
PHASE_B_SOURCE_KINDS = ("ordinary_image_pass", "production_capture")
PHASE_B_MANUAL_MARKERS = (
    "manual", "f3", "annotation", "coordinate", "direction", "bounds",
    "camera_manifest", "advanced", "digest", "pin",
)
# What a *real* adapter would have to join before Phase B could ever be
# authorized. Recorded as data so the contract is explicit; nothing in this task
# satisfies it, and ``phase_b_gate`` therefore refuses categorically.
PHASE_B_JOIN_REQUIREMENTS = (
    "revalidated_exact_freeze",
    "same_pass_frames",
    "same_pass_source",
    "camera",
    "intrinsics",
    "pose",
    "transform",
    "world_domain",
    "occlusion",
    "ownership",
)
# The exact terminal-specific artifact set that a Phase A freeze may cover.
# Nothing else may be present: a missing, duplicate, foreign or extra artifact
# refuses the freeze. ``freeze.json`` seals the set and is never a member.
FREEZE_ARTIFACTS_BY_TERMINAL = {
    TERMINAL_CAPTURED_SOLVER_ABSTENTION: (
        "record.json", "route_b.mcscan", "diagnostics.json", "scorecard.json"),
    TERMINAL_ASSEMBLED_NO_SUPPORT_CAPTURE: ("diagnostics.json", "refusal.json"),
    TERMINAL_EARLY_REFUSAL_NO_CAPTURE: ("diagnostics.json", "refusal.json"),
    TERMINAL_PROFILE_DRIFT_NO_CAPTURE: ("diagnostics.json", "refusal.json"),
    TERMINAL_RESOURCE_ABORTED_NO_CAPTURE: ("diagnostics.json", "refusal.json"),
}
# Per-artifact serialized caps. A freeze refuses an artifact above its own typed
# cap, so an over-cap record/scorecard/journal/refusal can never be sealed.
FREEZE_ARTIFACT_CAPS = {
    "record.json": 32 * 1024 * 1024,
    "route_b.mcscan": 4 * 1024 * 1024,
    "diagnostics.json": 2 * 1024 * 1024,
    "scorecard.json": 4 * 1024 * 1024,
    "refusal.json": 64 * 1024,
}
DOWNSTREAM_METRIC_KEYS = (
    "geometry", "placement", "ownership", "cell_pairs", "exact_state",
)


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """The five pre-launch limits the run harness must be able to enforce.

    ``total_output_bytes`` caps everything the run may leave behind; the
    per-artifact caps are enforced at their own publication points and are
    deliberately independent of the accepted 512 MiB matcher sidecar
    ``payload_bytes`` allowance, which is a serialized-record field cap, not a
    process-memory or on-disk budget.
    """

    wall_seconds: float = 900.0
    process_tree_bytes: int = 8 * 1024 * 1024 * 1024
    total_output_bytes: int = 128 * 1024 * 1024
    record_bytes: int = 32 * 1024 * 1024
    project_bytes: int = 4 * 1024 * 1024
    diagnostics_bytes: int = 2 * 1024 * 1024
    stream_bytes: int = 4 * 1024 * 1024

    def record(self) -> dict:
        return {
            "wall_seconds": self.wall_seconds,
            "process_tree_bytes": self.process_tree_bytes,
            "total_output_bytes": self.total_output_bytes,
            "record_bytes": self.record_bytes,
            "project_bytes": self.project_bytes,
            "diagnostics_bytes": self.diagnostics_bytes,
            "stream_bytes": self.stream_bytes,
        }

    def declared_budget_bytes(self) -> int:
        """Declared worst-case output of the bounded run.

        The per-artifact caps sum to far less than ``total_output_bytes``; the
        harness refuses before launch if that declaration cannot fit.
        """
        return (self.record_bytes + self.project_bytes + self.diagnostics_bytes
                + self.stream_bytes)


class TemplateMatchProgressError(ValueError):
    """One typed, bounded scorecard/gate refusal."""

    def __init__(self, reason: str, field: str = "", detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.field = field
        self.detail = detail


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_of(payload: object, *, drop: str) -> str:
    body = {key: value for key, value in payload.items() if key != drop}
    return _digest(body)


# ---------------------------------------------------------------------------
# Pin identity
# ---------------------------------------------------------------------------

def check_pin_profile(pins, expected=None) -> tuple[str, ...]:
    """Ordered same-pass ingress identity gate against an accepted pin profile.

    ``pins`` is the ordered same-pass entry ingress sequence. Each item may be a
    :class:`FrameIngressProvenanceV1` (or anything exposing ``byte_sha256``) or a
    bare 64-character hex string. ``expected`` defaults to the accepted T2 byte
    SHA-256 pins; the Phase A harness always passes that fixed profile. The
    comparison is positional and exact: a permuted, missing, extra or changed
    pin refuses ``input_pin_mismatch``. An arbitrary PNG (path match, filename
    match, size match or no pin at all) can therefore never satisfy the gate.
    """
    profile = T2_PIN_SHA256S if expected is None else tuple(expected)
    if not profile or any(not _hex64(value) for value in profile):
        raise TemplateMatchProgressError(
            "input_pin_mismatch", "expected", "the expected pin profile is invalid")
    values = []
    for position, pin in enumerate(pins):
        value = getattr(pin, "byte_sha256", pin)
        if not _hex64(value):
            raise TemplateMatchProgressError(
                "input_pin_mismatch", f"pin.{position}",
                "the same-pass ingress pin is not a lowercase byte sha256")
        values.append(value)
    hashes = tuple(values)
    if len(hashes) != len(profile):
        raise TemplateMatchProgressError(
            "input_pin_mismatch", "pin_count",
            f"expected {len(profile)} ordered pins, got {len(hashes)}")
    for position, (actual, wanted) in enumerate(zip(hashes, profile)):
        if actual != wanted:
            raise TemplateMatchProgressError(
                "input_pin_mismatch", f"pin.{position}",
                f"ordered ingress pin {position} is not the accepted pin")
    return hashes


# ---------------------------------------------------------------------------
# Terminal classification
# ---------------------------------------------------------------------------

def classify_refusal(reason: str, field: str = "") -> str:
    """Map one typed runner refusal to its exact Phase A terminal state.

    This is the *legacy* mapping only. The assembled-without-capture terminal is
    deliberately excluded: only the wrapper's :func:`classify_outcome`, reading
    the same-pass graph checkpoint, may classify it. A bare
    ``capture_unavailable`` is therefore an early refusal here.
    """
    if reason in PROFILE_DRIFT_REASONS:
        return TERMINAL_PROFILE_DRIFT_NO_CAPTURE
    if reason == "resource_aborted":
        return TERMINAL_RESOURCE_ABORTED_NO_CAPTURE
    return TERMINAL_EARLY_REFUSAL_NO_CAPTURE


def classify_outcome(reason: str, field: str = "",
                     *, graph_checkpoint: Mapping | None = None) -> str:
    """Wrapper-only terminal classification from the same-pass evidence.

    The legacy route-B runner reports ``capture_unavailable`` whenever its
    complete-support sink did not fire exactly once, without inventing a Phase A
    terminal. The wrapper alone decides ``assembled_no_support_capture`` by
    reading the *same-pass* graph checkpoint the runner emitted immediately
    before that refusal: the terminal is assembled-without-capture only when the
    graph checkpoint says the component solver ``assembled``. There is no second
    graph/matcher/solver pass. A missing or non-assembled checkpoint stays an
    early refusal, never a fabricated capture.

    ``graph_checkpoint`` is the already-persisted graph stage record (its
    ``status``/``reason`` fields); it is evidence, not a re-run.
    """
    if reason in PROFILE_DRIFT_REASONS:
        return TERMINAL_PROFILE_DRIFT_NO_CAPTURE
    if reason == "resource_aborted":
        return TERMINAL_RESOURCE_ABORTED_NO_CAPTURE
    if (reason == LEGACY_CAPTURE_REFUSAL_REASON and field == "capture"
            and type(graph_checkpoint) is dict
            and graph_checkpoint.get("status") == "assembled"
            and graph_checkpoint.get("stage") == "graph"):
        return TERMINAL_ASSEMBLED_NO_SUPPORT_CAPTURE
    if reason == SOLVER_ASSEMBLED_NO_CAPTURE:
        # Back-compat with the superseded runner reason; still gated on the
        # same-pass graph checkpoint so a caller cannot fabricate the terminal.
        if (type(graph_checkpoint) is dict
                and graph_checkpoint.get("status") == "assembled"
                and graph_checkpoint.get("stage") == "graph"):
            return TERMINAL_ASSEMBLED_NO_SUPPORT_CAPTURE
        return TERMINAL_EARLY_REFUSAL_NO_CAPTURE
    return TERMINAL_EARLY_REFUSAL_NO_CAPTURE


# ---------------------------------------------------------------------------
# Universe 1: candidate rows
# ---------------------------------------------------------------------------

def _capture_rows(capture_record: Mapping) -> list:
    """Index rows joined to their sidecar support payload, in page order.

    The canonical V2 capture keeps the ordered admitted rows in ``index_pages``
    and each present row's support payload in ``sidecar_pages``. The join is by
    ``admitted_row_index``; a ``present`` row whose sidecar payload is absent or
    whose payload omits ``values`` stays UNKNOWN downstream rather than being
    counted as zero.
    """
    if type(capture_record) is not dict:
        raise TemplateMatchProgressError(
            "scorecard_input_malformed", "capture")
    pages = capture_record.get("index_pages")
    if type(pages) is not list:
        raise TemplateMatchProgressError(
            "scorecard_input_malformed", "capture.index_pages")
    rows: list = []
    for page in pages:
        if type(page) is not dict or type(page.get("rows")) is not list:
            raise TemplateMatchProgressError(
                "scorecard_input_malformed", "capture.index_pages.rows")
        rows.extend(page["rows"])
    if len(rows) > MAX_SCORE_ROWS:
        raise TemplateMatchProgressError(
            "scorecard_over_cap", "capture.rows",
            f"{len(rows)} rows exceed the {MAX_SCORE_ROWS} scorecard budget")
    sidecar: dict = {}
    sidecar_pages = capture_record.get("sidecar_pages")
    if type(sidecar_pages) is list:
        for page in sidecar_pages:
            entries = page.get("entries") if type(page) is dict else None
            if type(entries) is not list:
                continue
            for entry in entries:
                if type(entry) is not dict:
                    continue
                index = entry.get("admitted_row_index")
                if type(index) is int:
                    sidecar[index] = entry.get("payload")
    enriched: list = []
    for row in rows:
        if type(row) is not dict:
            enriched.append(row)
            continue
        joined = dict(row)
        index = row.get("admitted_row_index")
        if index in sidecar and type(sidecar[index]) is dict:
            joined["support"] = sidecar[index]
        enriched.append(joined)
    return enriched


def _row_state(row: object) -> str:
    """One row's support availability: present, refused or UNKNOWN.

    ``present`` requires a support payload dict; ``refused`` requires a typed
    ``support_refusal_reason`` string. A row with neither is UNKNOWN - a missing
    payload is never turned into zero, a refusal or a fabricated disposition.
    """
    if type(row) is not dict:
        return UNKNOWN
    support = row.get("support")
    if type(support) is dict:
        return PRESENT
    reason = row.get("support_refusal_reason")
    if type(reason) is str and reason:
        return REFUSED
    return UNKNOWN


def _row_disposition(row: object) -> str:
    if type(row) is not dict:
        return UNKNOWN
    disposition = row.get("disposition")
    if disposition in (DISPOSITION_KEPT, DISPOSITION_DROPPED):
        return disposition
    return UNKNOWN


def candidate_row_universe(rows: Sequence) -> dict:
    """All admitted rows with kept/dropped x support cross-tabs.

    Denominators are printed explicitly: ``all_admitted`` (the complete captured
    row universe) and ``conditional_support_present`` (the subset whose support
    payload is present). These are availability denominators, never accuracy.
    """
    cross_tab: dict = {}
    counts = {PRESENT: 0, REFUSED: 0, UNKNOWN: 0}
    dispositions = {DISPOSITION_KEPT: 0, DISPOSITION_DROPPED: 0, UNKNOWN: 0}
    kept_present = 0
    for row in rows:
        state = _row_state(row)
        disposition = _row_disposition(row)
        counts[state] += 1
        dispositions[disposition] += 1
        bucket = cross_tab.setdefault(disposition, {PRESENT: 0, REFUSED: 0,
                                                    UNKNOWN: 0})
        bucket[state] += 1
        if disposition == DISPOSITION_KEPT and state == PRESENT:
            kept_present += 1
    admitted = len(rows)
    return {
        "admitted_total": admitted,
        "kept": dispositions[DISPOSITION_KEPT],
        "dropped": dispositions[DISPOSITION_DROPPED],
        "disposition_unknown": dispositions[UNKNOWN],
        "support_present": counts[PRESENT],
        "support_refused": counts[REFUSED],
        "support_unknown": counts[UNKNOWN],
        "kept_with_support_present": kept_present,
        "cross_tab": {key: dict(sorted(value.items()))
                      for key, value in sorted(cross_tab.items())},
        "denominators": {
            "all_admitted": admitted,
            "conditional_support_present": counts[PRESENT],
        },
    }


# ---------------------------------------------------------------------------
# Universe 2: logical cell pairs
# ---------------------------------------------------------------------------

def _payload_pairs(payload: object):
    """Return the declared logical-pair rows, or ``None`` when UNKNOWN."""
    if type(payload) is not dict:
        return None
    array = payload.get("logical_pairs")
    if type(array) is not dict:
        return None
    values = array.get("values")
    if type(values) is not list:
        return None
    return values


def logical_cell_pair_universe(rows: Sequence) -> dict:
    """Declared and unique logical cell pairs over the admitted universe.

    A logical pair's identity is **(candidate ordinal, four coordinates)**: the
    same four coordinates on two different candidates are two distinct logical
    pairs and never collapse into one. The candidate ordinal is the row's
    ``admitted_row_index`` when present (the canonical admitted ordinal) and the
    page-order position otherwise.

    A row whose payload (or payload ``values``) is absent contributes to
    ``rows_unknown_payload`` and stays UNKNOWN; it is never counted as zero
    pairs. The sampler's ``-1`` border sentinel is an UNKNOWN entry, not a pair.
    """
    declared = 0
    unique: set = set()
    unknown_entries = 0
    rows_with_pairs = 0
    rows_unknown_payload = 0
    rows_without_payload = 0
    for position, row in enumerate(rows):
        if type(row) is not dict or _row_state(row) != PRESENT:
            rows_without_payload += 1
            continue
        values = _payload_pairs(row.get("support"))
        if values is None:
            rows_unknown_payload += 1
            continue
        rows_with_pairs += 1
        declared += len(values)
        if declared > MAX_LOGICAL_PAIR_ELEMENTS:
            return {
                "state": UNKNOWN,
                "reason": "logical_pair_budget_exhausted",
                "declared_pairs": None,
                "unique_pairs": None,
                "unknown_entries": None,
                "rows_with_pairs": rows_with_pairs,
                "rows_unknown_payload": rows_unknown_payload,
                "rows_without_payload": rows_without_payload,
            }
        ordinal = _candidate_ordinal(row, position)
        for pair in values:
            if (type(pair) is not list or len(pair) != 4
                    or any(type(item) is not int for item in pair)):
                unknown_entries += 1
                continue
            if any(item < 0 for item in pair):
                unknown_entries += 1
                continue
            unique.add((ordinal, pair[0], pair[1], pair[2], pair[3]))
    return {
        "state": "known",
        "identity": "candidate_ordinal_and_four_coordinates",
        "declared_pairs": declared,
        "unique_pairs": len(unique),
        "unknown_entries": unknown_entries,
        "rows_with_pairs": rows_with_pairs,
        "rows_unknown_payload": rows_unknown_payload,
        "rows_without_payload": rows_without_payload,
    }


def _candidate_ordinal(row: object, position: int) -> int:
    """Canonical admitted ordinal of one row: its admitted index or position."""
    if type(row) is dict:
        index = row.get("admitted_row_index")
        if type(index) is int and index >= 0:
            return index
    return position


# ---------------------------------------------------------------------------
# Universe 3: unique (frame_id, source_pixel_index) dependencies
# ---------------------------------------------------------------------------

def _dependency_arrays(entry):
    """Yield the bounded dependency arrays of one decoded bound entry."""
    witness = getattr(entry, "witness", None)
    if witness is None:
        return
    cells = getattr(witness, "cell_source_indices", None)
    if cells is not None:
        yield "cell_source_indices", cells
    for name in ("rgb_rectification", "coverage_rectification",
                 "region_rectification"):
        rectification = getattr(witness, name, None)
        if rectification is None:
            continue
        taps = getattr(rectification, "tap_source_indices", None)
        if taps is not None:
            yield f"{name}.tap_source_indices", taps


def dependency_universe(section, *, ingress_by_frame=None) -> dict:
    """Project decoded persisted witnesses into unique source dependencies.

    The decoder is the real :func:`decode_section` (never a stub): a canonical
    section plus the record's own root pins decodes into immutable typed
    witnesses, and this function projects their declared source dependencies into
    unique ``(frame_id, source_pixel_index)`` sets with per-frame counts and a
    union total. The sampler's ``-1`` border sentinel is UNKNOWN and is counted
    separately, never added as a dependency. An absent or refused section is
    UNKNOWN, and a dependency budget overrun is UNKNOWN, never a partial sum.
    """
    from mc_photogrammetry.pipeline import signature_sampling_persistence as persistence

    if section is None:
        return {"state": UNKNOWN, "reason": "sampling_section_unavailable"}
    try:
        decoded = persistence.decode_section(
            section, ingress_by_frame=ingress_by_frame, require_ingress=True)
    except persistence.SamplingPersistenceError as error:
        return {"state": UNKNOWN, "reason": error.reason, "field": error.field}
    if decoded.status != persistence.SAMPLING_PERSISTENCE_STATUS_COMPLETE:
        return {"state": UNKNOWN, "reason": decoded.reason or "section_refused"}

    cell_frames: dict = {}
    tap_frames: dict = {}
    combined_pairs: set = set()
    declared_cells = 0
    declared_taps = 0
    unknown_cells = 0
    unknown_taps = 0
    scanned = 0
    projected_entries = 0
    entries_total = len(decoded.entries)
    bound_entries = 0
    refused_entries = 0
    persistence_refused_entries = 0
    partial_entries = 0
    entries_with_payload = 0
    entries_without_payload = 0
    for entry in decoded.entries:
        outcome = getattr(entry, "outcome", "")
        if outcome == persistence.SAMPLING_PERSISTENCE_OUTCOME_BOUND:
            bound_entries += 1
        elif outcome == persistence.SAMPLING_PERSISTENCE_OUTCOME_REFUSED:
            refused_entries += 1
        elif outcome == (
                persistence.SAMPLING_PERSISTENCE_OUTCOME_PERSISTENCE_REFUSED):
            persistence_refused_entries += 1
        if not getattr(entry, "has_payload", False):
            entries_without_payload += 1
            continue
        entries_with_payload += 1
        projected_entries += 1
        entry_unknown = 0
        frame_id = entry.frame_id
        for name, array in _dependency_arrays(entry):
            size = int(getattr(array, "size", 0))
            scanned += size
            if scanned > MAX_PROJECTED_DEPENDENCIES:
                return {
                    "state": UNKNOWN,
                    "reason": "dependency_budget_exhausted",
                    "projected_entries": projected_entries,
                    "entries_total": entries_total,
                    "entries_with_payload": entries_with_payload,
                    "entries_without_payload": entries_without_payload,
                }
            target = cell_frames if name == "cell_source_indices" else tap_frames
            is_cell = name == "cell_source_indices"
            values = array.tolist()
            bucket = target.setdefault(frame_id, set())
            for value in values:
                if value == -1:
                    if is_cell:
                        unknown_cells += 1
                    else:
                        unknown_taps += 1
                    entry_unknown += 1
                    continue
                pixel = int(value)
                bucket.add(pixel)
                combined_pairs.add((frame_id, pixel))
            if is_cell:
                declared_cells += size
            else:
                declared_taps += size
        # A *partial* witness is one whose payload decoded but whose source
        # dependencies include at least one UNKNOWN sentinel. It stays in the
        # denominators rather than being silently dropped.
        if entry_unknown:
            partial_entries += 1

    def summarise(frames: dict, declared: int, unknown: int) -> dict:
        per_frame = {frame: len(values) for frame, values in frames.items()}
        return {
            "per_frame": dict(sorted(per_frame.items())),
            "frames_total": len(per_frame),
            "union_total": sum(per_frame.values()),
            "declared": declared,
            "unknown_sentinel": unknown,
        }

    cell_union = sum(len(values) for values in cell_frames.values())
    tap_union = sum(len(values) for values in tap_frames.values())
    return {
        "state": UNKNOWN if (entries_without_payload or partial_entries
                              or refused_entries or persistence_refused_entries) else "known",
        "projected_entries": projected_entries,
        "entries_total": entries_total,
        "bound_entries": bound_entries,
        "refused_entries": refused_entries,
        "persistence_refused_entries": persistence_refused_entries,
        "partial_entries": partial_entries,
        "entries_with_payload": entries_with_payload,
        "entries_without_payload": entries_without_payload,
        "denominators": {
            # Every decoded entry counts, including refused, persistence-refused,
            # skipped (no payload) and partial (sentinel-bearing) witnesses, so
            # coverage can never claim completeness from bound entries alone.
            "all_entries": entries_total,
            "entries_with_payload": entries_with_payload,
            "bound_entries": bound_entries,
            "partial_entries": partial_entries,
        },
        "cell_dependencies": summarise(cell_frames, declared_cells,
                                       unknown_cells),
        "tap_dependencies": summarise(tap_frames, declared_taps,
                                      unknown_taps),
        # Kept separate: the cell channel and the rectification-tap channel each
        # keep their own union total; neither is folded into the other.
        "channel_totals_separate": {"cell": cell_union, "tap": tap_union},
        # The *actual* union of (frame_id, source_pixel_index) pairs across both
        # channels. A pixel named by both a cell and a tap is one pair here, so
        # this is never larger than the sum of the separate channel totals.
        "union_pairs": len(combined_pairs),
        "union_total": len(combined_pairs),
    }


# ---------------------------------------------------------------------------
# Signature-grid availability and source screenshot footprint
# ---------------------------------------------------------------------------

def support_grid_availability(rows: Sequence) -> dict:
    """Per-row and per-cell **signature-grid** availability.

    This counts the matcher's own signature-grid warp counters: whether each
    target cell's ``warp_source_rows``/``warp_source_columns`` entry names a real
    source pixel. It is deliberately renamed from a footprint metric: it is a
    *support-grid availability* counter over the signature grid, **not** a source
    screenshot footprint, coverage or accuracy measure.

    A payload's warp sampling maps use the sampler's ``-1`` sentinel for a target
    cell whose source footprint is UNKNOWN; those cells are counted unknown, not
    as available and not as zero. A missing payload is a row-level UNKNOWN.
    """
    per_row: dict = {}
    counts = {PRESENT: 0, REFUSED: 0, UNKNOWN: 0}
    available_cells = 0
    unknown_cells = 0
    for row in rows:
        state = _row_state(row)
        counts[state] += 1
        index = row.get("admitted_row_index") if type(row) is dict else None
        per_row[str(index)] = state
        if state != PRESENT:
            continue
        payload = row.get("support")

        def _values(name):
            array = payload.get(name) if type(payload) is dict else None
            values = array.get("values") if type(array) is dict else None
            return values if type(values) is list else None

        # One target cell is one (row, column) map position: the cell is
        # available only when both warp maps name a real source pixel; either
        # ``-1`` makes that cell UNKNOWN.
        row_map = _values("warp_source_rows")
        column_map = _values("warp_source_columns")
        if row_map is None or column_map is None:
            continue
        for row_values, column_values in zip(row_map, column_map):
            if type(row_values) is not list or type(column_values) is not list:
                continue
            for source_row, source_column in zip(row_values, column_values):
                if source_row == -1 or source_column == -1:
                    unknown_cells += 1
                elif (type(source_row) is int and source_row >= 0
                      and type(source_column) is int and source_column >= 0):
                    available_cells += 1
    return {
        "metric": "support_grid_availability",
        "note": ("signature-grid warp counter only; not a source screenshot "
                 "footprint, coverage or accuracy measure"),
        "per_row": per_row,
        "rows_total": len(rows),
        "present": counts[PRESENT],
        "refused": counts[REFUSED],
        "unknown": counts[UNKNOWN],
        "cells_available": available_cells,
        "cells_unknown": unknown_cells,
        "denominators": {
            "all_admitted": len(rows),
            "conditional_support_present": counts[PRESENT],
        },
    }


def source_screenshot_footprint(*, endpoint_witness_csr_join=None) -> dict:
    """Source screenshot-footprint metrics: null/unavailable without a join.

    A real source screenshot footprint can only be measured from a *persisted
    endpoint witness joined to a CSR* (compressed-sparse-row) source tap map
    bound to the same pass. This task persists no such join, so every footprint
    metric is ``null`` with a typed unavailable state. No grid counter, pixel
    count, digest or pin is substituted for the missing join.
    """
    return {
        "metric": "source_screenshot_footprint",
        "state": "unavailable",
        "reason": SOURCE_FOOTPRINT_REASON,
        "available": False,
        "join_present": endpoint_witness_csr_join is not None,
        "metrics": {
            "source_pixels": None,
            "union_source_pixels": None,
            "per_frame_source_pixels": None,
            "source_screenshot_count": None,
            "covered_fraction": None,
        },
    }


# Back-compat alias kept private: the old name was a misnomer and must not be
# used for source footprint claims.
footprint_availability = support_grid_availability


# ---------------------------------------------------------------------------
# Downstream metrics
# ---------------------------------------------------------------------------

def downstream_metrics(terminal: str) -> dict:
    """Null downstream metrics for every Phase A terminal.

    Phase A never runs the evaluator and the component solver abstained at
    ``captured_solver_abstention``; geometry, placement, ownership, cell-pair and
    exact-state metrics therefore stay ``null`` (UNKNOWN), never a measured zero
    and never an accuracy percentage.
    """
    if terminal not in TERMINAL_STATES:
        raise TemplateMatchProgressError(
            "scorecard_input_malformed", "terminal", str(terminal))
    metrics = {key: None for key in DOWNSTREAM_METRIC_KEYS}
    metrics["state"] = "unavailable"
    metrics["reason"] = ("phase_a_evaluation_only" if terminal
                         == TERMINAL_CAPTURED_SOLVER_ABSTENTION
                         else "no_complete_support_capture")
    return metrics


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def build_progress_scorecard(*, capture_record, section=None,
                             ingress_by_frame=None,
                             terminal=TERMINAL_CAPTURED_SOLVER_ABSTENTION,
                             endpoint_witness_csr_join=None) -> dict:
    """Build the bounded Phase A scorecard over the separate universes.

    ``terminal`` must be one of the five exact terminals. A non-capture terminal
    still gets the declared (empty/UNKNOWN) universes so an early refusal is
    preserved as evidence rather than silently absent.
    """
    if terminal not in TERMINAL_STATES:
        raise TemplateMatchProgressError(
            "scorecard_input_malformed", "terminal", str(terminal))
    rows = _capture_rows(capture_record)
    scorecard = {
        "schema": SCHEMA,
        "terminal": terminal,
        "complete_support_capture": (
            terminal == TERMINAL_CAPTURED_SOLVER_ABSTENTION),
        "candidate_rows": candidate_row_universe(rows),
        "logical_cell_pairs": logical_cell_pair_universe(rows),
        "unique_dependencies": dependency_universe(
            section, ingress_by_frame=ingress_by_frame),
        # Signature-grid warp counters, explicitly *not* a source footprint.
        "support_grid_availability": support_grid_availability(rows),
        # Source screenshot-footprint metrics stay null/unavailable: no persisted
        # endpoint witness + CSR join exists in this task.
        "source_screenshot_footprint": source_screenshot_footprint(
            endpoint_witness_csr_join=endpoint_witness_csr_join),
        "downstream": downstream_metrics(terminal),
        "note": ("Availability/coverage counters only. No accuracy, ownership, "
                 "geometry, correspondence or correctness claim; -1 and missing "
                 "payloads are UNKNOWN. 'support_grid_availability' is a "
                 "signature-grid counter, never a source footprint."),
    }
    scorecard["scorecard_sha256"] = scorecard_digest(scorecard)
    return scorecard


def scorecard_digest(scorecard: Mapping) -> str:
    if type(scorecard) is not dict:
        raise TemplateMatchProgressError(
            "scorecard_input_malformed", "scorecard")
    return _sha256_of(scorecard, drop="scorecard_sha256")


# ---------------------------------------------------------------------------
# Freeze-before-evaluator gate
# ---------------------------------------------------------------------------

def _freeze_relative_path(path: object, name: str) -> str:
    """Validate one artifact path as a unique relative, root-contained path."""
    text = str(path).replace("\\", "/")
    if (not text or ":" in text or text.startswith("/")
            or text.startswith("//")):
        raise TemplateMatchProgressError(
            "freeze_path_invalid", name,
            "artifact path must be relative to the frozen root")
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise TemplateMatchProgressError(
            "freeze_path_invalid", name,
            "artifact path must not contain empty, '.' or '..' components")
    if parts[-1] != name:
        raise TemplateMatchProgressError(
            "freeze_path_invalid", name,
            "artifact path basename must equal its artifact name")
    return text


def artifact_freeze_record(artifacts, *, root=None,
                           terminal=TERMINAL_CAPTURED_SOLVER_ABSTENTION,
                           limits=None) -> dict:
    """Freeze the exact terminal-specific Phase A artifact identity.

    ``artifacts`` is an iterable of ``(name, relative_path, sha256, size)`` rows,
    each already measured by the harness with a bounded, stable read. The record
    is digest-sealed so an evaluator loader can revalidate it. Rejected:

    * a terminal that is not one of the five exact terminals;
    * a missing or extra artifact (the set must equal
      :data:`FREEZE_ARTIFACTS_BY_TERMINAL` exactly);
    * a duplicate artifact name or duplicate relative path;
    * an absolute, drive-qualified, ``..``-bearing or non-root-contained path;
    * a non-typed size, a negative size, or a size above the artifact's own
      typed cap in :data:`FREEZE_ARTIFACT_CAPS`; and
    * a non-64-hex digest.
    """
    if terminal not in FREEZE_ARTIFACTS_BY_TERMINAL:
        raise TemplateMatchProgressError(
            "freeze_input_malformed", "terminal", str(terminal))
    expected = tuple(FREEZE_ARTIFACTS_BY_TERMINAL[terminal])
    caps = dict(FREEZE_ARTIFACT_CAPS)
    if limits is not None:
        def _cap(name, default):
            return int(getattr(limits, name, default))
        caps["record.json"] = _cap("record_bytes", caps["record.json"])
        caps["route_b.mcscan"] = _cap("project_bytes", caps["route_b.mcscan"])
        caps["diagnostics.json"] = _cap("diagnostics_bytes",
                                        caps["diagnostics.json"])
        caps["scorecard.json"] = _cap("scorecard_bytes",
                                      caps["scorecard.json"])
    entries = []
    seen_names: set = set()
    seen_paths: set = set()
    for item in artifacts:
        if type(item) is dict:
            name = item.get("name")
            path = item.get("path")
            sha256 = item.get("sha256")
            size = item.get("size")
        else:
            try:
                name, path, sha256, size = item
            except (TypeError, ValueError) as error:
                raise TemplateMatchProgressError(
                    "freeze_input_malformed", "artifacts") from error
        name = str(name)
        if name not in expected:
            raise TemplateMatchProgressError(
                "freeze_artifact_foreign", name,
                "artifact is not in the terminal's exact artifact set")
        if name in seen_names:
            raise TemplateMatchProgressError(
                "freeze_artifact_duplicate", name)
        relative = _freeze_relative_path(path, name)
        if relative in seen_paths:
            raise TemplateMatchProgressError(
                "freeze_artifact_duplicate", relative)
        if not _hex64(sha256):
            raise TemplateMatchProgressError(
                "freeze_input_malformed", name, "digest is not lowercase hex64")
        if type(size) is not int or size < 0:
            raise TemplateMatchProgressError(
                "freeze_input_malformed", name, "size must be a nonnegative int")
        cap = caps.get(name)
        if type(cap) is int and size > cap:
            raise TemplateMatchProgressError(
                "freeze_artifact_over_cap", name,
                f"{size} > {cap} bytes")
        seen_names.add(name)
        seen_paths.add(relative)
        entries.append({"name": name, "path": relative, "sha256": sha256,
                        "size": int(size), "cap_bytes": cap})
    missing = [name for name in expected if name not in seen_names]
    if missing:
        raise TemplateMatchProgressError(
            "freeze_artifact_missing", ",".join(missing))
    entries.sort(key=lambda entry: expected.index(entry["name"]))
    record = {
        "schema": FREEZE_SCHEMA,
        "terminal": terminal,
        "artifact_set": "terminal_exact",
        "root": None if root is None else str(root),
        "artifacts": entries,
    }
    record["freeze_sha256"] = _sha256_of(record, drop="freeze_sha256")
    return record


def assert_phase_a_frozen(freeze, *, read_sha256, read_size=None, root=None,
                          is_reparse=None, read_binding=None) -> dict:
    """Revalidate a Phase A freeze record before any evaluator loader runs.

    Every artifact must belong to the freeze's exact terminal-specific artifact
    set with a unique, relative, root-contained, non-reparse path and a typed
    size inside its own cap. Each artifact's current byte digest (and, when
    supplied, size) must equal the frozen value; the injected ``read_sha256``
    performs the actual bounded read, so this module stays I/O-free. A missing,
    resized, digest-changed, reparse, foreign or extra artifact refuses
    ``phase_a_not_frozen``.

    ``read_binding`` (optional) must return ``{"ok": True, ...}``; the harness
    passes a callback that cross-validates the record/scorecard/journal bindings
    against the frozen digests. A binding mismatch refuses ``phase_a_not_frozen``.
    """
    if type(freeze) is not dict or freeze.get("schema") != FREEZE_SCHEMA:
        raise TemplateMatchProgressError(
            "phase_a_not_frozen", "freeze", "freeze record is missing")
    if freeze.get("freeze_sha256") != _sha256_of(freeze, drop="freeze_sha256"):
        raise TemplateMatchProgressError(
            "phase_a_not_frozen", "freeze_sha256", "freeze record is tampered")
    terminal = freeze.get("terminal")
    if terminal not in FREEZE_ARTIFACTS_BY_TERMINAL:
        raise TemplateMatchProgressError(
            "phase_a_not_frozen", "terminal", "unknown freeze terminal")
    expected = tuple(FREEZE_ARTIFACTS_BY_TERMINAL[terminal])
    artifacts = freeze.get("artifacts")
    if type(artifacts) is not list or not artifacts:
        raise TemplateMatchProgressError(
            "phase_a_not_frozen", "freeze.artifacts")
    base = freeze.get("root") if root is None else str(root)
    seen_names: set = set()
    seen_paths: set = set()
    for item in artifacts:
        if type(item) is not dict or not _hex64(item.get("sha256")):
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", "freeze.artifacts")
        name = str(item.get("name"))
        if name not in expected:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", name, "foreign artifact in freeze")
        if name in seen_names:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", name, "duplicate artifact in freeze")
        relative = _freeze_relative_path(item.get("path"), name)
        if relative in seen_paths:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", relative, "duplicate artifact path")
        seen_names.add(name)
        seen_paths.add(relative)
        if type(item.get("size")) is not int or item["size"] < 0:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", name, "artifact size is not a typed int")
        cap = FREEZE_ARTIFACT_CAPS[name]
        if item["size"] > cap:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", name, "artifact exceeds its typed cap")
        target = _join_frozen_path(base, relative)
        if is_reparse is not None and is_reparse(target):
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", name, "artifact became a reparse point")
        try:
            actual = read_sha256(target)
        except FileNotFoundError as error:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", name, "artifact is missing") from error
        if actual != item["sha256"]:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", name,
                "artifact digest changed after the freeze")
        if read_size is not None and int(read_size(target)) != item["size"]:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", name,
                "artifact size changed after the freeze")
    missing = [name for name in expected if name not in seen_names]
    if missing:
        raise TemplateMatchProgressError(
            "phase_a_not_frozen", ",".join(missing),
            "terminal artifact set is incomplete")
    if read_binding is not None:
        binding = read_binding(freeze, base)
        if type(binding) is not dict or binding.get("ok") is not True:
            raise TemplateMatchProgressError(
                "phase_a_not_frozen", "binding",
                "record/scorecard/journal binding is not consistent with the "
                "freeze")
    return {"status": "frozen", "terminal": terminal,
            "artifact_count": len(artifacts),
            "freeze_sha256": freeze["freeze_sha256"]}


def _join_frozen_path(base, relative: str) -> str:
    """Join a frozen root and a validated relative path, staying inside root."""
    import os as _os

    relative = relative.replace("/", _os.sep)
    if base is None:
        return relative
    base = _os.path.abspath(str(base))
    candidate = _os.path.abspath(_os.path.join(base, relative))
    if candidate != base and not candidate.startswith(base + _os.sep):
        raise TemplateMatchProgressError(
            "phase_a_not_frozen", relative, "artifact escapes the frozen root")
    return candidate


# ---------------------------------------------------------------------------
# Phase B source-bound evidence gate
# ---------------------------------------------------------------------------

def phase_b_join_contract() -> dict:
    """The join a *real* Phase B adapter must provide before authorization.

    This is descriptive data only: no adapter in this task satisfies it, and
    :func:`phase_b_gate` is therefore categorically unavailable. A freeze alone
    never authorizes Phase B.
    """
    return {
        "requirements": list(PHASE_B_JOIN_REQUIREMENTS),
        "evidence_fields": list(PHASE_B_EVIDENCE_FIELDS),
        "note": ("a real adapter must join a revalidated exact Phase A freeze to "
                 "same-pass frame/source/camera/intrinsics/pose/transform/"
                 "world-domain/occlusion/ownership evidence; digest-only, "
                 "pin-only and manual metadata never qualify"),
    }


def phase_b_gate(evidence=None, *, adapter_join=None) -> dict:
    """Phase B is categorically unavailable; this never returns authorized.

    The only positive authorization path would be a real adapter joining a
    revalidated exact freeze to same-pass source-bound evidence for every field
    in :data:`PHASE_B_EVIDENCE_FIELDS`. No such adapter exists in this task, so
    every call refuses the typed :data:`PHASE_B_ALIGNMENT_UNAVAILABLE`, whether
    or not an ``adapter_join`` claim is supplied. Manual F3/coordinate/direction/
    bounds/camera-manifest metadata is rejected first with its own typed refusal.

    Fabricated positive authorization is impossible by construction: there is no
    accepted-input branch that returns ``status == "authorized"``.
    """
    if type(evidence) is dict:
        if evidence.get("manual_metadata"):
            raise TemplateMatchProgressError(
                "phase_b_manual_metadata_rejected", "manual_metadata")
        for field, record in evidence.items():
            if type(record) is not dict:
                continue
            source = str(record.get("source", "")).lower()
            if any(token in source for token in PHASE_B_MANUAL_MARKERS):
                raise TemplateMatchProgressError(
                    "phase_b_manual_metadata_rejected", field)
    missing = []
    if type(adapter_join) is not dict:
        missing.append("adapter_join")
    else:
        if adapter_join.get("freeze_revalidated") is not True:
            missing.append("revalidated_exact_freeze")
        if adapter_join.get("same_pass") is not True:
            missing.append("same_pass")
        joined = adapter_join.get("fields")
        if type(joined) is not dict:
            missing.append("fields")
        else:
            for field in PHASE_B_EVIDENCE_FIELDS:
                record = joined.get(field)
                if (type(record) is not dict
                        or record.get("binding") != "source_bound"
                        or record.get("source") not in PHASE_B_SOURCE_KINDS
                        or not _hex64(record.get("evidence_sha256"))
                        or type(record.get("detail")) is not dict
                        or not record["detail"]):
                    missing.append(field)
    # Even a structurally complete claim is not a *real* adapter join: this task
    # implements none, so authorization remains categorically unavailable.
    missing.append("real_adapter")
    raise TemplateMatchProgressError(
        PHASE_B_ALIGNMENT_UNAVAILABLE, "adapter_join",
        "phase B is unavailable: no real adapter joins a revalidated exact "
        "freeze to same-pass source-bound evidence (missing: "
        + ",".join(sorted(set(missing))) + ")")
