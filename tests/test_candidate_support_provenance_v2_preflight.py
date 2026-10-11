"""Focused controls for the additive, default-off V2 complete-support contract.

Scope: the accepted ``DS-R4-COMPLETE-SUPPORT-DESIGN-2026-10-04.md`` design and
its assignment ``DS-R4-V2-PREFLIGHT-IMPL``. These tests use synthetic,
in-memory material only: synthetic index rows, synthetic digest sections,
synthetic sidecar payloads and the synthetic ceiling preflight. No real
screenshot, truth, camera manifest, mask, evaluator payload, litematic or JAR is
opened, no capture is run, and no reconstruction is rerun.

The default path is proved byte-for-byte neutral by hashing the shipped V1
module and the graph module and by checking that no V2 capture is wired into
them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.pipeline import candidate_support_provenance as csp
from mc_photogrammetry.pipeline import candidate_support_provenance_v2 as v2
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline.contracts import FrameTransform
from mc_photogrammetry.reconstruction import plane_graph
from test_candidate_support_provenance import (
    _HEX,
    _HEX_B,
    _binding,
    _match,
    _observation,
    _payload_row,
    _pixels,
    _png_bytes,
    _readings_for,
    _refusal_row,
)

# Byte-for-byte preservation pins. V1 is unchanged. The graph module changes
# only by the authorized additive, default-off V2 diagnostic seam
# (DS-R4-V2-ROUTEB-DIAGNOSTIC-WIRING), so its new source is pinned here too and
# the default-off parity is proved behaviorally below.
V1_SOURCE_SHA256 = (
    "A8BD3102652720513ED85E2D3B7E2C81A79A01A7621C41238B09F6E741604562")
GRAPH_SOURCE_SHA256 = (
    "32747546C934F8C218E43646CC995F310FF383E522C8A10F878E691F54954188")

_ASSIGNMENT = hashlib.sha256(b"assignment").hexdigest()


# ---------------------------------------------------------------------------
# Synthetic V2 material helpers
# ---------------------------------------------------------------------------

def _small_payload(pairs: int = 6) -> v2.MatcherSupportPayloadV1:
    logical = np.zeros((pairs, 4), dtype=np.int64)
    warp = np.zeros((2, 2), dtype=np.int64)
    for array in (logical, warp):
        array.flags.writeable = False
    return v2.MatcherSupportPayloadV1(
        logical_pairs=logical, warp_source_rows=warp,
        warp_source_columns=warp, source_shape=(2, 2), target_shape=(2, 2))


def _index_row(
    index: int,
    *,
    disposition: str,
    state: str,
    reason: str | None = None,
    payload_sha: str | None = None,
    overlap: int = 0,
    group_index: int = 0,
) -> v2.CandidateSupportIndexRowV2:
    kept = disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT
    return v2.CandidateSupportIndexRowV2(
        admitted_row_index=index, group_index=group_index, left="a", right="b",
        left_axis=1, right_axis=1, quadrant_of_right=0, left_plane=0,
        right_plane=0, left_alternative_index=0, right_alternative_index=0,
        left_b=1.0, right_b=1.0, delta=0.0, tolerance=0.15,
        disposition=disposition, match_index=(index if kept else None),
        overlap_cells=overlap, cell_shift_rc=(0, 0), support_state=state,
        support_refusal_reason=reason, support_payload_sha256=payload_sha)


def _universe(count: int, *, present: int = 0, refused: int | None = None,
              group_stride: int = 1):
    """A synthetic universe of ``count`` mixed kept/dropped admitted rows.

    ``present`` leading rows carry a real sidecar payload; every other row is a
    typed row-local refusal (``refused`` overrides that code).
    """
    payload = _small_payload()
    rows: list[v2.CandidateSupportIndexRowV2] = []
    entries: list[v2.MatcherSupportSidecarEntryV2] = []
    sections: dict[str, list] = {section: [] for section in v2.DIGEST_SECTIONS}
    for index in range(count):
        kept = index % 2 == 0
        disposition = (plane_graph.SAME_AXIS_DISPOSITION_KEPT if kept
                       else plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
        is_present = index < present
        row = _index_row(
            index, disposition=disposition,
            state=v2.SUPPORT_STATE_PRESENT if is_present else v2.SUPPORT_STATE_REFUSED,
            reason=(None if is_present
                    else (refused or "matcher_witness_unavailable")),
            payload_sha=payload.payload_sha256 if is_present else None,
            overlap=len(payload.logical_pairs) if is_present else 0,
            group_index=index // group_stride)
        rows.append(row)
        if is_present:
            entries.append(v2.MatcherSupportSidecarEntryV2(
                admitted_row_index=index, payload=payload))
        sections["match_sha256"].append(_HEX)
        sections["envelope_sha256"].append(_HEX if is_present else None)
        sections["left_signature_sha256"].append(_HEX if is_present else None)
        sections["right_signature_sha256"].append(_HEX_B if is_present else None)
        sections["matcher_witness_sha256"].append(_HEX if is_present else None)
        sections["input_match_identity_sha256"].append(
            v2._identity_digest([v2._row_identity(row)]))
    return tuple(rows), sections, tuple(entries)


def _assemble(rows, sections, entries, *, expected_row_count=None, **overrides):
    return v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=expected_row_count,
        plan=v2.CompleteSupportCapPlanV1(**overrides))


def _plan_from_proposals(preflight) -> v2.CompleteSupportCapPlanV1:
    proposed = dict(preflight.proposed_caps)
    return v2.CompleteSupportCapPlanV1(
        index_page_max_bytes=proposed["INDEX_PAGE_MAX_BYTES"],
        digest_page_max_bytes=proposed["DIGEST_PAGE_MAX_BYTES"],
        sidecar_page_max_bytes=proposed["SIDECAR_PAGE_MAX_BYTES"],
        sidecar_entry_max_raw_bytes=proposed["SIDECAR_ENTRY_MAX_RAW_BYTES"],
        sidecar_entry_max_bytes=proposed["SIDECAR_ENTRY_MAX_BYTES"],
        sidecar_aggregate_max_bytes=proposed["SIDECAR_AGGREGATE_MAX_BYTES"],
        root_max_bytes=proposed["ROOT_MAX_BYTES"],
        universe_max_rows=proposed["UNIVERSE_MAX_ROWS"])


def _collector_material(*, witnesses=True):
    kept = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    dropped = _match("a", "b", left_plane=1, right_plane=1, score=0.5)
    rows = (
        _payload_row(kept, index=0, disposition="kept", post_index=0),
        _refusal_row(dropped, index=1, disposition="same_axis_mismatch",
                     post_index=None),
    )
    matches = (kept, dropped)
    dispositions = ("kept", "same_axis_mismatch")
    bindings = (
        _binding("a", _pixels(101), _png_bytes((1, 2, 3)), signature_sha256=_HEX),
        _binding("b", _pixels(102), _png_bytes((4, 5, 6)),
                 signature_sha256=_HEX_B),
    )
    return (matches, rows, dispositions,
            bindings if witnesses else ())


# ---------------------------------------------------------------------------
# Preflight: deterministic ceiling derivation and cap proposals
# ---------------------------------------------------------------------------

def test_preflight_is_deterministic_and_derives_page_counts_at_ceiling():
    first = v2.preflight_complete_support()
    second = v2.preflight_complete_support()
    assert first.to_record() == second.to_record()
    assert first.declared_universe_cap == v2.UNIVERSE_MAX_ROWS == 4_096
    assert first.schema == v2.PREFLIGHT_SCHEMA
    assert first.preflight_status in (v2.PREFLIGHT_OK, v2.PREFLIGHT_REFUSED)
    assert first.preflight_refusal_reason in (
        "", v2.PREFLIGHT_CAP_CONFIGURATION_INVALID,
        v2.PREFLIGHT_SYNTHETIC_SIZE_OVER_CAP, v2.PREFLIGHT_AGGREGATE_UNBOUNDED)

    ceiling = first.declared_universe_cap
    rows_per_page = first.derived("max_rows_per_index_page")
    digests_per_page = first.derived("max_digests_per_digest_page")
    entries_per_page = first.derived("max_entries_per_sidecar_page")
    assert rows_per_page > 0 and digests_per_page > 0 and entries_per_page > 0
    assert first.derived("index_page_count") == -(-ceiling // rows_per_page)
    per_section = -(-ceiling // digests_per_page)
    assert first.derived("digest_page_count_per_section") == per_section
    assert first.derived("digest_page_count") == 6 * per_section
    assert first.derived("sidecar_page_count") == -(-ceiling // entries_per_page)

    # Every measured canonical size is positive and each page fits its envelope
    # plus at least one element.
    for name, value in first.measured_bytes:
        assert value > 0, name
    assert (first.measurement("index_page_worst_case_bytes")
            > first.measurement("index_page_envelope_bytes"))
    assert (first.measurement("digest_page_worst_case_bytes")
            > first.measurement("digest_page_envelope_bytes"))
    assert (first.measurement("sidecar_page_worst_case_bytes")
            > first.measurement("sidecar_page_envelope_bytes"))
    assert first.cap_invariant_ok is True


def test_preflight_default_refuses_placeholder_caps_and_proposes_measured_caps():
    default = v2.preflight_complete_support()
    # The placeholder aggregate cap cannot hold the worst-case complete sidecar.
    assert default.preflight_status == v2.PREFLIGHT_REFUSED
    assert default.preflight_refusal_reason == v2.PREFLIGHT_SYNTHETIC_SIZE_OVER_CAP
    assert "sidecar_aggregate" in default.detail

    proposed = _plan_from_proposals(default)
    accepted = v2.preflight_complete_support(proposed)
    assert accepted.preflight_status == v2.PREFLIGHT_OK
    assert accepted.preflight_refusal_reason == ""
    assert accepted.cap_invariant_ok is True
    assert accepted.measurement("sidecar_entry_worst_case_bytes") <= (
        proposed.sidecar_entry_max_bytes)
    assert accepted.measurement("index_page_worst_case_bytes") <= (
        proposed.index_page_max_bytes)
    assert accepted.measurement("digest_page_worst_case_bytes") <= (
        proposed.digest_page_max_bytes)
    assert accepted.measurement("sidecar_page_worst_case_bytes") <= (
        proposed.sidecar_page_max_bytes)
    assert accepted.measurement("sidecar_aggregate_worst_case_bytes") <= (
        proposed.sidecar_aggregate_max_bytes)
    assert accepted.measurement("root_worst_case_bytes") <= proposed.root_max_bytes


def test_preflight_cap_configuration_invalid_and_aggregate_unbounded():
    invalid = v2.preflight_complete_support(v2.CompleteSupportCapPlanV1(
        sidecar_page_max_bytes=2_000, sidecar_entry_max_bytes=3_000))
    assert invalid.preflight_status == v2.PREFLIGHT_REFUSED
    assert invalid.preflight_refusal_reason == v2.PREFLIGHT_CAP_CONFIGURATION_INVALID
    assert invalid.cap_invariant_ok is False

    unbounded = v2.preflight_complete_support(v2.CompleteSupportCapPlanV1(
        sidecar_page_max_bytes=2_000, sidecar_entry_max_bytes=1_000))
    assert unbounded.preflight_status == v2.PREFLIGHT_REFUSED
    assert unbounded.preflight_refusal_reason == v2.PREFLIGHT_AGGREGATE_UNBOUNDED
    assert unbounded.derived("max_entries_per_sidecar_page") == 0


def test_preflight_refuses_when_index_page_or_root_cap_is_too_small():
    default = v2.preflight_complete_support()
    base = _plan_from_proposals(default).as_record()
    tiny_index = v2.CompleteSupportCapPlanV1(**{**base, "index_page_max_bytes": 10})
    refused = v2.preflight_complete_support(tiny_index)
    assert refused.preflight_status == v2.PREFLIGHT_REFUSED
    assert refused.preflight_refusal_reason == v2.PREFLIGHT_SYNTHETIC_SIZE_OVER_CAP
    assert "index_page" in refused.detail

    tiny_root = v2.CompleteSupportCapPlanV1(**{**base, "root_max_bytes": 1_000})
    root_refused = v2.preflight_complete_support(tiny_root)
    assert root_refused.preflight_status == v2.PREFLIGHT_REFUSED
    assert root_refused.preflight_refusal_reason == (
        v2.PREFLIGHT_SYNTHETIC_SIZE_OVER_CAP)
    assert "root" in root_refused.detail


def test_preflight_reads_no_candidate_material_and_imports_no_evaluator():
    import inspect

    parameters = list(inspect.signature(v2.preflight_complete_support).parameters)
    assert parameters == ["plan"]
    source = Path(v2.__file__).read_text(encoding="utf-8").casefold()
    for token in ("renderedtruthevidence", "rendererownershipwitness",
                  "evaluation_frozen_planes", "physical_plane_key",
                  "image_derived_graph"):
        assert token not in source, token


# ---------------------------------------------------------------------------
# Identity: full universe versus the unchanged kept-row subset
# ---------------------------------------------------------------------------

def test_full_universe_identity_is_distinct_from_the_kept_subset():
    rows, sections, entries = _universe(5, present=2)
    capture = _assemble(rows, sections, entries, expected_row_count=5)
    assert capture.status == "complete", capture.root.reason
    root = capture.root
    assert root.whole_universe_identity_sha256 == (
        v2.complete_support_universe_digest(rows))
    kept = [v2._row_identity(row) for row in rows
            if row.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT]
    assert root.input_match_identity_sha256 == csp._identity_digest(kept)
    assert root.whole_universe_identity_sha256 != root.input_match_identity_sha256

    # A caller-declared full-universe digest that disagrees refuses globally.
    tampered = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5, plan=v2.CompleteSupportCapPlanV1(),
        expected_whole_universe_identity_sha256="0" * 64)
    assert tampered.status == "refused"
    assert tampered.root.reason == "universe_identity_mismatch"
    assert tampered.row_count == 0 and tampered.index_pages == ()


def test_repeated_capture_of_the_same_universe_is_identical():
    rows, sections, entries = _universe(40, present=3)
    first = _assemble(rows, sections, entries, expected_row_count=40)
    second = _assemble(rows, sections, entries, expected_row_count=40)
    assert first.status == second.status == "complete"
    assert first.root.record_sha256 == second.root.record_sha256
    assert json.dumps(first.to_record(), sort_keys=True) == json.dumps(
        second.to_record(), sort_keys=True)


def test_page_and_section_binding_tamper_fails_closed():
    rows, sections, entries = _universe(5, present=2)
    capture = _assemble(rows, sections, entries, expected_row_count=5)
    assert capture.status == "complete", capture.root.reason

    dropped = _assemble(rows, sections, entries, expected_row_count=5)
    object.__setattr__(dropped, "digest_pages", dropped.digest_pages[:-1])
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(dropped)

    flipped = _assemble(rows, sections, entries, expected_row_count=5)
    object.__setattr__(flipped.digest_pages[0], "page_sha256", "0" * 64)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(flipped)

    stale = _assemble(rows, sections, entries, expected_row_count=5)
    object.__setattr__(stale.root, "whole_universe_identity_sha256", "0" * 64)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(stale)


def _sparse_universe(count: int, present_indices):
    """A synthetic universe whose *present* support rows are sparse.

    Every admitted row index in ``range(count)`` exists exactly once, so the
    index pages and digest sections stay the dense admitted order. Only the rows
    whose index is in ``present_indices`` carry a real sidecar payload and become
    sidecar entries; every other row is a typed row-local refusal. The present
    admitted indices are therefore not the sidecar tuple ordinals.
    """
    wanted = set(present_indices)
    assert all(0 <= index < count for index in wanted)
    payload = _small_payload()
    rows: list[v2.CandidateSupportIndexRowV2] = []
    entries: list[v2.MatcherSupportSidecarEntryV2] = []
    sections: dict[str, list] = {section: [] for section in v2.DIGEST_SECTIONS}
    for index in range(count):
        kept = index % 2 == 0
        disposition = (plane_graph.SAME_AXIS_DISPOSITION_KEPT if kept
                       else plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
        is_present = index in wanted
        row = _index_row(
            index, disposition=disposition,
            state=v2.SUPPORT_STATE_PRESENT if is_present else v2.SUPPORT_STATE_REFUSED,
            reason=None if is_present else "matcher_witness_unavailable",
            payload_sha=payload.payload_sha256 if is_present else None,
            overlap=len(payload.logical_pairs) if is_present else 0,
            group_index=index)
        rows.append(row)
        if is_present:
            entries.append(v2.MatcherSupportSidecarEntryV2(
                admitted_row_index=index, payload=payload))
        sections["match_sha256"].append(_HEX)
        sections["envelope_sha256"].append(_HEX if is_present else None)
        sections["left_signature_sha256"].append(_HEX if is_present else None)
        sections["right_signature_sha256"].append(_HEX_B if is_present else None)
        sections["matcher_witness_sha256"].append(_HEX if is_present else None)
        sections["input_match_identity_sha256"].append(
            v2._identity_digest([v2._row_identity(row)]))
    return tuple(rows), sections, tuple(entries)


def _sidecar_page_cap(entries, count: int) -> int:
    """A declared page cap that fits exactly ``count`` worst-case entries.

    The tests pass the real entry objects, so the cap is measured from canonical
    encodings rather than assumed.
    """
    envelope = v2._sidecar_page_size((), 0, 0)
    per_entry = v2._sidecar_page_size(entries[:1], 0, entries[0].admitted_row_index) - envelope
    return envelope + count * per_entry + 64


def _synthetic_sidecar_page(page_index, first, entries):
    """A directly constructed page with a self-consistent content digest.

    This is the tamper-model seam: the caller assembles pages by hand (including
    pages the packer could never emit) and recomputes the page digest so only
    ``validate_complete_support_capture`` can reject the capture.
    """
    page = v2.MatcherSupportSidecarPageV2(
        page_index=page_index, first_admitted_row_index=first,
        entries=tuple(entries))
    object.__setattr__(
        page, "page_sha256", v2._sidecar_page_content_digest(page))
    return page


def _sparse_capture_with_pages(rows, sections, entries, pages, **overrides):
    """A hand-built capture over ``rows``/``sections`` with explicit sidecar pages.

    Index, digest and root commitments are real (only the sidecar pagination is
    caller-controlled), but ``CompleteSupportCaptureV2.__post_init__`` is
    deliberately bypassed: the whole point of these controls is to reach
    ``validate_complete_support_capture`` with a self-consistent page set that
    only the capture-level sparse replay rules can refuse. The root is signed
    after construction so its own digest replay is not the rejection reason.
    """
    plan = v2.CompleteSupportCapPlanV1(**{**overrides, "universe_max_rows": 64})
    index_pages = v2._pack_index_pages(rows, plan.index_page_max_bytes)
    digest_pages = v2._pack_digest_pages(sections, plan.digest_page_max_bytes)
    root = v2.CandidateSupportCaptureRootV2(
        status="complete", basis=v2.COMPLETE_SUPPORT_BASIS, reason=None, field=None,
        failed_assignment_sha256=v2._sha(
            {"domain": "failed_best_assignment", "payload": []}),
        whole_universe_identity_sha256=v2.complete_support_universe_digest(rows),
        input_match_identity_sha256=v2._identity_digest(
            [v2._row_identity(row) for row in rows
             if row.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT]),
        row_count=len(rows), declared_universe_cap=plan.universe_max_rows,
        index_pages=tuple(
            v2.CandidateSupportIndexPageBindingV2(
                page.page_index, page.page_sha256, page.row_count)
            for page in index_pages),
        sidecar_pages=tuple(
            v2.CandidateSupportSidecarPageBindingV2(
                page.page_index, page.page_sha256, page.entry_count)
            for page in pages),
        digest_sections=tuple(
            v2._digest_section_from_pages(
                section,
                tuple(page for page in digest_pages if page.section == section))
            for section in v2.DIGEST_SECTIONS),
        ingress=(), cap_plan=plan)
    object.__setattr__(root, "cap_plan_sha256", v2._digest(
        v2.CAP_PLAN_DIGEST_DOMAIN, plan.as_record()))
    object.__setattr__(root, "record_sha256", v2._digest(
        v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
        v2._root_payload(root, include_digest=False)))
    capture = object.__new__(v2.CompleteSupportCaptureV2)
    object.__setattr__(capture, "root", root)
    object.__setattr__(capture, "index_pages", index_pages)
    object.__setattr__(capture, "digest_pages", digest_pages)
    object.__setattr__(capture, "sidecar_pages", tuple(pages))
    object.__setattr__(capture, "schema", v2.COMPLETE_SUPPORT_CAPTURE_SCHEMA)
    return capture


def test_sparse_present_rows_pack_and_replay_in_and_across_pages():
    # Admitted indices 0, 3, 8: a same-page gap (0 -> 3 on page 0) and a
    # cross-page gap (3 -> 8 between pages). The sidecar tuple ordinals 0, 1, 2
    # are not the admitted indices, so an ordinal-based packer breaks here.
    rows, sections, entries = _sparse_universe(12, (0, 3, 8))
    two_per_page = _sidecar_page_cap(entries, 2)
    capture = _assemble(rows, sections, entries, expected_row_count=12,
                        sidecar_page_max_bytes=two_per_page)
    assert capture.status == "complete", capture.root.reason
    v2.validate_complete_support_capture(capture)
    assert len(capture.sidecar_pages) == 2

    flat = [entry for page in capture.sidecar_pages for entry in page.entries]
    assert [entry.admitted_row_index for entry in flat] == [0, 3, 8]
    for page in capture.sidecar_pages:
        assert page.first_admitted_row_index == page.entries[0].admitted_row_index
    assert [page.first_admitted_row_index
            for page in capture.sidecar_pages] == [0, 8]
    assert [entry.admitted_row_index for entry in capture.sidecar_pages[1].entries] == [8]

    # Exactly the present rows, in admitted order, bind one-to-one.
    present = [row for page in capture.index_pages for row in page.rows
               if row.support_state == v2.SUPPORT_STATE_PRESENT]
    assert [row.admitted_row_index for row in present] == [0, 3, 8]
    for row, entry in zip(present, flat):
        assert row.admitted_row_index == entry.admitted_row_index
        assert row.support_payload_sha256 == entry.payload_sha256


def test_sparse_single_page_gap_and_adjacent_present_rows_pack_and_replay():
    # Same-page-only gap (2, 5) and a present row that starts after a refused
    # first row (1, 2, 3): the first sidecar page does not start at index 0.
    for present in ((2, 5), (1, 2, 3), (0, 1, 3, 9)):
        rows, sections, entries = _sparse_universe(12, present)
        capture = _assemble(rows, sections, entries, expected_row_count=12)
        assert capture.status == "complete", capture.root.reason
        v2.validate_complete_support_capture(capture)
        flat = [entry for page in capture.sidecar_pages for entry in page.entries]
        assert [entry.admitted_row_index for entry in flat] == list(present)
        assert capture.sidecar_pages[0].first_admitted_row_index == present[0]


def test_sparse_sidecar_reorder_and_misbinding_refuse_at_replay():
    # Admitted present rows 3, 5 and 7. Every page below is individually
    # self-consistent (its declared first index matches its own first entry and
    # its content digest replays), so only capture-level replay can reject it.
    rows, sections, entries = _sparse_universe(10, (3, 5, 7))
    at_three, at_five, at_seven = entries

    # Reordered across pages: page 0 covers rows 5 and 7, then page 1 returns to
    # row 3. Each page replays alone, but the flattened sidecar order is no
    # longer the present-row order.
    reordered_across_pages = _sparse_capture_with_pages(
        rows, sections, entries,
        (_synthetic_sidecar_page(0, 5, (at_five, at_seven)),
         _synthetic_sidecar_page(1, 3, (at_three,))))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(reordered_across_pages)

    # A page that declares an admitted index which is not the first present row
    # it covers (here row 7 while row 5 is still outstanding) while staying
    # internally consistent: its first entry is row 7, so the page's own first
    # index is 7, but the next present row the capture must cover is 5.
    misbound_first = _sparse_capture_with_pages(
        rows, sections, entries,
        (_synthetic_sidecar_page(0, 7, (at_seven,)),
         _synthetic_sidecar_page(1, 5, (at_five,)),
         _synthetic_sidecar_page(2, 3, (at_three,))))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(misbound_first)


def test_sparse_sidecar_duplicate_and_misbound_first_index_cannot_be_built():
    # The page constructor is the first guard: a page whose declared first index
    # is not its own first entry's admitted index is rejected at construction.
    rows, sections, entries = _sparse_universe(10, (3, 5, 7))
    at_three, at_five, at_seven = entries
    with pytest.raises(ValueError):
        _synthetic_sidecar_page(0, 5, (at_three,))
    with pytest.raises(ValueError):
        _synthetic_sidecar_page(0, 3, (at_five, at_three))
    with pytest.raises(ValueError):
        _synthetic_sidecar_page(0, 3, (at_three, at_three))

    # The same shape with the correct declared first index replays over the
    # sparse present rows.
    page = _synthetic_sidecar_page(0, 3, (at_three, at_five, at_seven))
    assert page.first_admitted_row_index == 3
    accepted = _sparse_capture_with_pages(
        rows, sections, entries,
        (_synthetic_sidecar_page(0, 3, (at_three,)),
         _synthetic_sidecar_page(1, 5, (at_five,)),
         _synthetic_sidecar_page(2, 7, (at_seven,))))
    v2.validate_complete_support_capture(accepted)


def test_sparse_sidecar_overflow_page_refuses_with_value_error():
    # Present rows 3, 5 and 7 are fully covered by page 0, then a self-consistent
    # extra page starts beyond the last present row. The flattened order stays
    # strictly increasing, so replay reaches the present-row lookup with no
    # present row left; it must fail closed with ValueError instead of leaking an
    # IndexError from present[len(entries)].
    rows, sections, entries = _sparse_universe(10, (3, 5, 7))
    at_three, at_five, at_seven = entries
    extra = v2.MatcherSupportSidecarEntryV2(
        admitted_row_index=9, payload=at_seven.payload)
    overflow = _sparse_capture_with_pages(
        rows, sections, entries,
        (_synthetic_sidecar_page(0, 3, (at_three, at_five, at_seven)),
         _synthetic_sidecar_page(1, 9, (extra,))))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(overflow)


# ---------------------------------------------------------------------------
# Null slots keep their ordered positions across page boundaries
# ---------------------------------------------------------------------------

def test_null_digest_slots_keep_their_positions_across_pages():
    rows, sections, entries = _universe(5, present=0)
    # Alternate digest/null in every nullable section so each page holds exactly
    # two ordered slots and a null lands on a page boundary.
    for name in v2.DIGEST_SECTIONS:
        if name != "input_match_identity_sha256":
            sections[name] = [_HEX, None, _HEX, None, _HEX]
    envelope = max(
        v2._page_size(v2._digest_page_payload(
            v2._DigestPageView(name, 0, 0, ()), include_digest=False),
            "page_sha256")
        for name in v2.DIGEST_SECTIONS)
    digest_page_cap = envelope + 72
    capture = _assemble(rows, sections, entries, expected_row_count=5,
                        digest_page_max_bytes=digest_page_cap)
    assert capture.status == "complete", capture.root.reason
    for name in v2.DIGEST_SECTIONS:
        pages = [page for page in capture.digest_pages if page.section == name]
        # Pages are consecutive slices: each page starts where the previous ended.
        expected_first = 0
        for page in pages:
            assert page.first_admitted_row_index == expected_first
            expected_first += page.digest_count
        flattened = [value for page in pages for value in page.digests]
        assert flattened == list(sections[name])
    # A null at an interior page boundary keeps its ordered slot and is not
    # collapsed or renumbered by the page split.
    envelope_pages = [page for page in capture.digest_pages
                      if page.section == "envelope_sha256"]
    assert [page.first_admitted_row_index for page in envelope_pages] == [0, 2, 4]
    assert [page.digest_count for page in envelope_pages] == [2, 2, 1]
    assert [page.digests for page in envelope_pages] == [
        (_HEX, None), (_HEX, None), (_HEX,)]
    for digest_section in v2.DIGEST_SECTIONS:
        total = sum(page.digest_count for page in capture.digest_pages
                    if page.section == digest_section)
        assert total == capture.root.row_count


# ---------------------------------------------------------------------------
# Exactly 1,928 rows round-trip; one extra row refuses atomically
# ---------------------------------------------------------------------------

def test_exactly_1928_rows_round_trip_with_every_disposition():
    rows, sections, entries = _universe(1_928, present=8)
    capture = _assemble(rows, sections, entries, expected_row_count=1_928)
    assert capture.status == "complete", capture.root.reason
    root = capture.root
    assert root.row_count == 1_928
    assert sum(binding.row_count for binding in root.index_pages) == 1_928
    assert len(root.index_pages) >= 1
    indexed = [row for page in capture.index_pages for row in page.rows]
    assert [row.admitted_row_index for row in indexed] == list(range(1_928))
    assert [row.disposition for row in indexed] == [
        row.disposition for row in rows]
    for section in v2.DIGEST_SECTIONS:
        assert sum(page.digest_count for page in capture.digest_pages
                   if page.section == section) == 1_928
    present = [row for row in indexed if row.support_state == v2.SUPPORT_STATE_PRESENT]
    assert len(present) == 8
    assert all(row.support_payload_sha256 for row in present)
    assert all(row.support_refusal_reason for row in indexed if row not in present)
    v2.validate_complete_support_capture(capture)


def test_1929_rows_refuse_atomically_on_count_mismatch():
    rows, sections, entries = _universe(1_929, present=0)
    capture = _assemble(rows, sections, entries, expected_row_count=1_928)
    assert capture.status == "refused"
    assert capture.root.reason == "universe_count_mismatch"
    assert capture.root.field == "row_count"
    assert capture.row_count == 0
    assert capture.index_pages == () and capture.digest_pages == ()
    assert capture.sidecar_pages == ()


# ---------------------------------------------------------------------------
# Order and identity violations refuse the whole capture
# ---------------------------------------------------------------------------

def test_order_violations_refuse_with_zero_pages():
    rows, sections, entries = _universe(5, present=1)
    duplicated = (rows[0], rows[0], rows[2], rows[3], rows[4])
    capture = v2.build_complete_support_capture(
        rows=duplicated, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5, plan=v2.CompleteSupportCapPlanV1())
    assert capture.status == "refused"
    assert capture.root.reason == "universe_order_mismatch"
    assert capture.row_count == 0 and capture.index_pages == ()

    gap = (rows[0], rows[2], rows[3], rows[4])
    gap_capture = v2.build_complete_support_capture(
        rows=gap, digest_sections={k: v[:4] for k, v in sections.items()},
        sidecar_entries=entries, expected_row_count=4,
        plan=v2.CompleteSupportCapPlanV1())
    assert gap_capture.status == "refused"
    assert gap_capture.root.reason == "universe_order_mismatch"


# ---------------------------------------------------------------------------
# Row-local typed support refusal versus capture-global refusal
# ---------------------------------------------------------------------------

def test_row_local_support_refusal_leaves_the_capture_complete():
    matches, rows, dispositions, bindings = _collector_material(witnesses=False)
    capture = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, rows,
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2)
    assert capture.status == "complete", capture.root.reason
    assert capture.row_count == 2
    assert capture.sidecar_pages == ()
    indexed = [row for page in capture.index_pages for row in page.rows]
    assert [row.disposition for row in indexed] == list(dispositions)
    assert indexed[0].support_state == v2.SUPPORT_STATE_REFUSED
    assert indexed[0].support_refusal_reason == "source_tap_witness_unavailable"
    assert indexed[1].support_state == v2.SUPPORT_STATE_REFUSED
    assert indexed[1].support_refusal_reason == "matcher_witness_unavailable"
    assert indexed[1].disposition == "same_axis_mismatch"
    assert indexed[1].match_index is None
    assert capture.root.reason is None
    assert capture.root.record_sha256


def test_present_and_refused_rows_coexist_in_one_complete_capture():
    matches, rows, dispositions, bindings = _collector_material(witnesses=True)
    capture = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, rows,
        _observation(matches, dispositions),
        ingress=tuple(item.ingress for item in bindings),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2)
    assert capture.status == "complete", capture.root.reason
    index_rows = [row for page in capture.index_pages for row in page.rows]
    assert [row.support_state for row in index_rows] == [
        v2.SUPPORT_STATE_PRESENT, v2.SUPPORT_STATE_REFUSED]
    present = index_rows[0]
    entry = capture.sidecar_pages[0].entries[0]
    assert entry.admitted_row_index == present.admitted_row_index
    assert entry.payload_sha256 == present.support_payload_sha256
    assert all(row.support_refusal_reason is None
               for row in index_rows if row.support_state == v2.SUPPORT_STATE_PRESENT)


def test_row_local_raw_and_canonical_entry_caps_keep_the_capture_complete():
    matches, rows, dispositions, bindings = _collector_material(witnesses=True)
    raw_capped = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, rows,
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2,
        plan=v2.CompleteSupportCapPlanV1(sidecar_entry_max_raw_bytes=1))
    assert raw_capped.status == "complete"
    assert raw_capped.row_count == 2
    assert raw_capped.sidecar_pages == ()
    assert [row.support_refusal_reason for page in raw_capped.index_pages
            for row in page.rows][0] == "support_payload_over_cap"

    canonical_capped = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, rows,
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2,
        plan=v2.CompleteSupportCapPlanV1(sidecar_entry_max_bytes=10))
    assert canonical_capped.status == "complete"
    assert canonical_capped.row_count == 2
    assert canonical_capped.sidecar_pages == ()
    assert [row.support_refusal_reason for page in canonical_capped.index_pages
            for row in page.rows][0] == "support_payload_over_cap"


def test_row_local_payload_tamper_refuses_only_that_row():
    matches, rows, dispositions, bindings = _collector_material(witnesses=True)
    # Break the envelope/witness digest binding on the payload row only.
    object.__setattr__(rows[0].payload.envelope, "matcher_witness_sha256",
                       "0" * 64)
    capture = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, rows,
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2)
    assert capture.status == "complete", capture.root.reason
    index_rows = [row for page in capture.index_pages for row in page.rows]
    assert index_rows[0].support_refusal_reason == "witness_digest_mismatch"
    assert index_rows[0].disposition == "kept"
    assert capture.sidecar_pages == ()


def test_capture_global_cap_failures_return_zero_rows_and_no_pages():
    rows, sections, entries = _universe(5, present=2)
    cases = (
        ("index_page_over_cap", {"index_page_max_bytes": 10}),
        ("digest_page_over_cap", {"digest_page_max_bytes": 10}),
        ("sidecar_page_over_cap", {"sidecar_page_max_bytes": 10}),
        ("sidecar_aggregate_over_cap", {"sidecar_aggregate_max_bytes": 1}),
        ("root_over_cap", {"root_max_bytes": 100}),
    )
    for reason, overrides in cases:
        capture = _assemble(rows, sections, entries, expected_row_count=5,
                            **overrides)
        assert capture.status == "refused", reason
        assert capture.root.reason == reason, (reason, capture.root.reason)
        assert capture.row_count == 0
        assert capture.index_pages == ()
        assert capture.digest_pages == ()
        assert capture.sidecar_pages == ()


# ---------------------------------------------------------------------------
# Default-off parity: V1 and the default graph path are byte-for-byte unchanged
# ---------------------------------------------------------------------------

def test_v1_source_is_unchanged_and_graph_is_the_pinned_additive_seam():
    assert hashlib.sha256(Path(csp.__file__).read_bytes()).hexdigest() == (
        V1_SOURCE_SHA256.casefold())
    assert hashlib.sha256(Path(graph_mod.__file__).read_bytes()).hexdigest() == (
        GRAPH_SOURCE_SHA256.casefold())


def test_v2_capture_seam_is_default_off_and_not_public():
    import inspect

    source = Path(graph_mod.__file__).read_text(encoding="utf-8")
    # The V2 collector is reachable only through the additive graph seam.
    assert "collect_route_b_diagnostic_support" in source
    graph_parameters = inspect.signature(
        graph_mod._derive_image_derived_graph).parameters
    assert graph_parameters["candidate_support_sink"].default is None
    assert graph_parameters["complete_support_sink"].default is None
    # The public producer and the production orchestration entry never expose the
    # V2 diagnostic sink, so the default path cannot collect V2 material.
    public_parameters = inspect.signature(
        graph_mod.derive_image_derived_graph).parameters
    assert "complete_support_sink" not in public_parameters
    assert not any("v2" in name or "complete_support" in name
                   for name in public_parameters)


def test_v1_default_row_cap_and_atomic_refusal_are_unchanged():
    assert csp.CANDIDATE_SUPPORT_MAX_ROWS == 1_024
    matches = tuple(_match("a", "b", left_plane=0, right_plane=0, score=0.9)
                    for _ in range(1_025))
    observation = _observation(matches, ("kept",) * 1_025)
    refused = csp.failed_candidate_support_provenance(
        None, matches, (), [None] * 1_025, observation)
    assert refused.status == "refused"
    assert refused.reason == "budget_exhausted"
    assert refused.field == "row_count"
    assert refused.row_count == 0 and refused.rows == ()


def test_v1_small_universe_still_assembles_after_v2_import():
    kept = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    rows = (_payload_row(kept, index=0, disposition="kept", post_index=0),)
    witness = _binding("a", _pixels(103), _png_bytes((9, 9, 9)))
    other = _binding("b", _pixels(104), _png_bytes((10, 10, 10)),
                     signature_sha256=_HEX_B)
    record = csp.failed_candidate_support_provenance(
        _readings_for((kept,)), (kept,), (witness, other), rows,
        _observation((kept,), ("kept",)),
        ingress=(witness.ingress, other.ingress),
        failed_assignment_sha256=_ASSIGNMENT)
    assert record.status == "assembled", record.reason
    assert record.row_count == 1
    csp.validate_candidate_support_provenance(record)


# ---------------------------------------------------------------------------
# Adversarial declared-bound controls (DS-R4-V2-BOUNDS-REMEDIATION)
# ---------------------------------------------------------------------------

def _row_kwargs(**overrides) -> dict:
    values = dict(
        admitted_row_index=0, group_index=0, left="a", right="b", left_axis=1,
        right_axis=1, quadrant_of_right=0, left_plane=0, right_plane=0,
        left_alternative_index=0, right_alternative_index=0, left_b=1.0,
        right_b=1.0, delta=0.0, tolerance=0.15,
        disposition=plane_graph.SAME_AXIS_DISPOSITION_KEPT, match_index=0,
        overlap_cells=0, cell_shift_rc=(0, 0),
        support_state=v2.SUPPORT_STATE_PRESENT, support_refusal_reason=None,
        support_payload_sha256=_HEX)
    values.update(overrides)
    return values


def _legal_payload() -> v2.MatcherSupportPayloadV1:
    logical = np.zeros((2, 4), dtype=np.int64)
    warp = np.zeros((2, 2), dtype=np.int64)
    for array in (logical, warp):
        array.flags.writeable = False
    return v2.MatcherSupportPayloadV1(
        logical_pairs=logical, warp_source_rows=warp,
        warp_source_columns=warp, source_shape=(2, 2), target_shape=(2, 2))


def _entry_with(*, source_shape, target_shape, pairs=0,
                warp_value=0) -> v2.MatcherSupportSidecarEntryV2:
    logical = np.zeros((pairs, 4), dtype=np.int64)
    warp_rows = np.full(target_shape, warp_value, dtype=np.int64)
    warp_columns = np.full(target_shape, warp_value, dtype=np.int64)
    for array in (logical, warp_rows, warp_columns):
        array.flags.writeable = False
    payload = v2.MatcherSupportPayloadV1(
        logical_pairs=logical, warp_source_rows=warp_rows,
        warp_source_columns=warp_columns, source_shape=source_shape,
        target_shape=target_shape)
    return v2.MatcherSupportSidecarEntryV2(admitted_row_index=0, payload=payload)


def _raw_ingress(*, frame_id="a", source_label="a.png", byte_length=1,
                 decoded_shape=(2, 2), transform=None
                 ) -> csp.FrameIngressProvenanceV1:
    return csp.FrameIngressProvenanceV1(
        frame_id=frame_id, source_label=source_label, byte_sha256=_HEX,
        byte_length=byte_length, decoded_shape=decoded_shape,
        decoded_dtype="|u1", decoded_pixel_sha256=_HEX_B,
        transform=transform or FrameTransform(source_size=(2, 2)))


def test_missing_expected_row_count_fails_closed_atomically():
    rows, sections, entries = _universe(5, present=2)
    capture = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        plan=v2.CompleteSupportCapPlanV1())
    assert capture.status == "refused"
    assert capture.root.reason == "universe_count_mismatch"
    assert capture.root.field == "expected_row_count"
    assert capture.row_count == 0 and capture.index_pages == ()
    assert capture.digest_pages == () and capture.sidecar_pages == ()

    matches, witness_rows, dispositions, bindings = _collector_material(
        witnesses=True)
    collected = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, witness_rows,
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT)
    assert collected.status == "refused"
    assert collected.root.field == "expected_row_count"
    assert collected.row_count == 0


def test_candidates_dispositions_and_witness_lengths_are_checked_before_zip():
    matches, witness_rows, dispositions, bindings = _collector_material(
        witnesses=True)
    # Bypass the observation constructor so a non-parallel record is possible.
    mismatched = object.__new__(plane_graph.SameAxisCandidateObservation)
    object.__setattr__(mismatched, "candidates", tuple(matches))
    object.__setattr__(mismatched, "dispositions", ("kept",))
    refused = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, witness_rows, mismatched,
        ingress=(), failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2)
    assert refused.status == "refused"
    assert refused.root.field == "dispositions"
    assert refused.row_count == 0

    witness_refused = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, witness_rows[:1],
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2)
    assert witness_refused.status == "refused"
    assert witness_refused.root.field == "matcher_witness_rows"


def test_index_rows_reject_every_field_beyond_its_declared_bound():
    base = _row_kwargs()
    # The declared-maximal legal row is accepted and stays inside every bound.
    maximal = v2.CandidateSupportIndexRowV2(**_row_kwargs(
        left="z" * v2.INDEX_ROW_MAX_FRAME_ID_BYTES,
        right="z" * v2.INDEX_ROW_MAX_FRAME_ID_BYTES,
        left_axis=-v2.INDEX_ROW_MAX_ABS_INT,
        cell_shift_rc=(-v2.INDEX_ROW_MAX_ABS_INT, -v2.INDEX_ROW_MAX_ABS_INT)))
    assert v2._row_field_over_cap(maximal) == ""
    cases = (
        ("left", {"left": "z" * (v2.INDEX_ROW_MAX_FRAME_ID_BYTES + 1)}),
        # ``ensure_ascii`` expands each ``é`` to a six-character escape, so a
        # short character string can still encode beyond the declared byte bound.
        ("right", {"right": "\u00e9" * (v2.INDEX_ROW_MAX_FRAME_ID_BYTES // 3)}),
        ("left_axis", {"left_axis": v2.INDEX_ROW_MAX_ABS_INT + 1}),
        ("right_axis", {"right_axis": -(v2.INDEX_ROW_MAX_ABS_INT + 1)}),
        ("overlap_cells", {"overlap_cells": v2.INDEX_ROW_MAX_ABS_INT + 1}),
        ("match_index", {"match_index": v2.INDEX_ROW_MAX_ABS_INT + 1}),
        ("cell_shift_rc", {"cell_shift_rc": (0, v2.INDEX_ROW_MAX_ABS_INT + 1)}),
        ("delta", {"delta": float("inf")}),
        ("tolerance", {"tolerance": float("nan")}),
    )
    for field, override in cases:
        with pytest.raises(v2.CandidateSupportError) as info:
            v2.CandidateSupportIndexRowV2(**{**base, **override})
        assert info.value.reason == "index_row_over_cap", field
        assert info.value.field == field, (field, info.value.field)


def test_preflight_index_row_measurement_bounds_the_declared_maximum():
    preflight = v2.preflight_complete_support()
    row = v2._synthetic_index_row(preflight.declared_universe_cap)
    assert v2._row_field_over_cap(row) == ""
    assert len(v2._canonical_bytes(row.to_record())) == preflight.measurement(
        "index_row_worst_case_bytes")
    assert (v2._serialized_string_bytes(row.left)
            == v2.INDEX_ROW_MAX_FRAME_ID_BYTES)
    assert (v2._serialized_string_bytes(row.right)
            == v2.INDEX_ROW_MAX_FRAME_ID_BYTES)
    assert row.cell_shift_rc == (-v2.INDEX_ROW_MAX_ABS_INT,
                                 -v2.INDEX_ROW_MAX_ABS_INT)


def test_sidecar_payloads_reject_shapes_and_elements_beyond_the_declared_bounds():
    assert v2._payload_over_declared_bounds(_legal_payload()) == ""
    with pytest.raises(v2.CandidateSupportError) as shape_info:
        _entry_with(source_shape=(v2.PAYLOAD_MAX_SHAPE_DIM + 1, 1),
                    target_shape=(1, 1))
    assert shape_info.value.reason == "support_payload_over_cap"
    assert shape_info.value.field == "shape"

    with pytest.raises(v2.CandidateSupportError) as element_info:
        _entry_with(source_shape=(2, 2), target_shape=(1, 1),
                    warp_value=v2.PAYLOAD_MAX_ABS_ELEMENT + 1)
    assert element_info.value.reason == "support_payload_over_cap"
    assert element_info.value.field == "warp_source_rows"


def test_sidecar_preflight_measurement_bounds_all_adversarial_entries():
    preflight = v2.preflight_complete_support()
    ceiling = preflight.measurement("sidecar_entry_worst_case_bytes")
    raw_cap = v2.SIDECAR_ENTRY_MAX_RAW_BYTES
    assert ceiling > 0
    adversarial = (
        _entry_with(source_shape=(1, 1), target_shape=(1, 1)),
        _entry_with(source_shape=(v2.PAYLOAD_MAX_SHAPE_DIM,
                                  v2.PAYLOAD_MAX_SHAPE_DIM),
                    target_shape=(1, 1)),
        _entry_with(source_shape=(2, 2), target_shape=(1, 1_024),
                    warp_value=-v2.PAYLOAD_MAX_ABS_ELEMENT),
        _entry_with(source_shape=(2, 2), target_shape=(1_024, 1),
                    warp_value=v2.PAYLOAD_MAX_ABS_ELEMENT),
        _entry_with(source_shape=(2, 2), target_shape=(32, 32),
                    warp_value=-v2.PAYLOAD_MAX_ABS_ELEMENT),
        # Many nested rows encode more brackets/commas than a wide grid with the
        # same element count, so ``(cells, 1)`` is the worst declared shape.
        _entry_with(source_shape=(v2.PAYLOAD_MAX_SHAPE_DIM,
                                  v2.PAYLOAD_MAX_SHAPE_DIM),
                    target_shape=(8_192, 1),
                    warp_value=-v2.PAYLOAD_MAX_ABS_ELEMENT),
        _entry_with(source_shape=(v2.PAYLOAD_MAX_SHAPE_DIM,
                                  v2.PAYLOAD_MAX_SHAPE_DIM),
                    target_shape=(64, 128),
                    warp_value=-v2.PAYLOAD_MAX_ABS_ELEMENT),
    )
    for entry in adversarial:
        assert entry.payload.payload_bytes() <= raw_cap
        assert (len(v2._canonical_bytes(entry.to_record())) <= ceiling)
    synthetic = v2._synthetic_sidecar_entry(raw_cap, v2.UNIVERSE_MAX_ROWS - 1)
    assert len(v2._canonical_bytes(synthetic.to_record())) == ceiling
    assert synthetic.payload.target_shape == (raw_cap // 16, 1)
    assert v2._payload_over_declared_bounds(synthetic.payload) == ""
    assert synthetic.payload.payload_bytes() == raw_cap


def test_preflight_digest_envelope_uses_the_longest_section_name():
    preflight = v2.preflight_complete_support()
    bound = preflight.declared_universe_cap - 1
    envelopes = {
        section: v2._page_size(v2._digest_page_payload(
            v2._DigestPageView(section, bound, bound, ()), include_digest=False),
            "page_sha256")
        for section in v2.DIGEST_SECTIONS}
    longest = max(envelopes, key=lambda name: envelopes[name])
    assert longest == max(v2.DIGEST_SECTIONS, key=len)
    measured = preflight.measurement("digest_page_envelope_bytes")
    assert measured == envelopes[longest]
    assert measured == max(envelopes.values())


def test_preflight_ingress_and_root_include_full_maximal_bindings():
    preflight = v2.preflight_complete_support()
    record = v2._synthetic_ingress_record()
    record_bytes = len(v2._canonical_bytes(record.to_record()))
    assert preflight.measurement("ingress_record_worst_case_bytes") == record_bytes
    tuple_bytes = len(v2._canonical_bytes(
        [record.to_record()] * v2.INGRESS_MAX_RECORDS))
    assert preflight.measurement("ingress_tuple_worst_case_bytes") == tuple_bytes
    assert tuple_bytes > record_bytes

    counts = dict(preflight.derived_counts)
    plan = v2.CompleteSupportCapPlanV1()
    arguments = (
        counts["index_page_count"], counts["sidecar_page_count"],
        counts["max_rows_per_index_page"], counts["max_entries_per_sidecar_page"],
        counts["max_digests_per_digest_page"], preflight.declared_universe_cap)
    without = v2._synthetic_root(*arguments, plan, ())
    with_ingress = v2._synthetic_root(
        *arguments, plan, (record,) * v2.INGRESS_MAX_RECORDS)
    assert (len(v2._canonical_bytes(with_ingress.to_record()))
            > len(v2._canonical_bytes(without.to_record())))


def test_capture_refuses_mutated_oversized_index_row_atomically():
    rows, sections, entries = _universe(5, present=2)
    mutated = list(rows)
    object.__setattr__(mutated[2], "left",
                       "z" * (v2.INDEX_ROW_MAX_FRAME_ID_BYTES + 1))
    capture = v2.build_complete_support_capture(
        rows=tuple(mutated), digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5)
    assert capture.status == "refused"
    assert capture.root.reason == "index_row_over_cap"
    assert capture.root.field.startswith("row.2.left")
    assert capture.row_count == 0 and capture.index_pages == ()
    assert capture.digest_pages == () and capture.sidecar_pages == ()


def test_collector_refuses_oversized_support_value_row_locally(monkeypatch):
    matches, witness_rows, dispositions, bindings = _collector_material(
        witnesses=True)
    logical = np.zeros((0, 4), dtype=np.int64)
    warp = np.zeros((1, 1), dtype=np.int64)
    for array in (logical, warp):
        array.flags.writeable = False
    oversized = v2.MatcherSupportPayloadV1(
        logical_pairs=logical, warp_source_rows=warp, warp_source_columns=warp,
        source_shape=(v2.PAYLOAD_MAX_SHAPE_DIM + 1, 1), target_shape=(1, 1))
    monkeypatch.setattr(v2, "_support_payload", lambda row: oversized)
    capture = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, witness_rows,
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2)
    assert capture.status == "complete", capture.root.reason
    assert capture.row_count == 2
    assert capture.sidecar_pages == ()
    index_rows = [row for page in capture.index_pages for row in page.rows]
    assert all(row.support_state == v2.SUPPORT_STATE_REFUSED
               for row in index_rows)
    assert index_rows[0].support_refusal_reason == "support_payload_over_cap"
    assert index_rows[1].support_refusal_reason == "matcher_witness_unavailable"


# A sentinel that must never reach a refusal field through ``str(error)``.
_RAW_FAILURE_SENTINEL = "SECRET-7f3a9c-raw-payload-do-not-echo"


def _row_site_raw_raiser(*args, **kwargs):
    """Inject a raw ``ValueError`` while the collector walks one admitted row."""
    raise ValueError(_RAW_FAILURE_SENTINEL)


def _assembly_site_raw_raiser(*args, **kwargs):
    """Inject a raw ``TypeError`` while the collector packs the capture pages."""
    raise TypeError(_RAW_FAILURE_SENTINEL)


def _sidecar_site_raw_raiser(*args, **kwargs):
    """Inject a raw ``ValueError`` at the row-local sidecar entry construction."""
    raise ValueError(_RAW_FAILURE_SENTINEL)


def _raw_failure_refusal(monkeypatch, attribute, raiser):
    """Inject one raw failure at ``v2.<attribute>`` and return its refusal field.

    Every raw refusal stays atomic (zero rows, no pages) and still validates; the
    returned field is the diagnostic-only localization code under test.
    """
    monkeypatch.setattr(v2, attribute, raiser)
    matches, witness_rows, dispositions, bindings = _collector_material(
        witnesses=True)
    capture = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, witness_rows,
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2)
    assert capture.status == "refused"
    assert capture.root.reason == "universe_identity_mismatch"
    assert capture.row_count == 0 and capture.index_pages == ()
    assert capture.digest_pages == () and capture.sidecar_pages == ()
    v2.validate_complete_support_capture(capture)
    return capture.root.field


def test_collector_raw_row_failure_field_localizes_operation_and_position(
        monkeypatch):
    field = _raw_failure_refusal(
        monkeypatch, "_alternative_index", _row_site_raw_raiser)
    # Not the bare exception class: the innermost module, operation and source
    # line name the failing site, and the exact admitted row is appended.
    assert field != "ValueError"
    class_name, qualified, location = field.split(":")
    assert class_name == "ValueError"
    assert qualified.endswith("._row_site_raw_raiser")
    assert qualified.count(".") >= 1
    assert location.endswith("@row0")
    line_component = location[: -len("@row0")]
    assert line_component.isdigit() and int(line_component) > 0
    # No exception text, no path and no image-derived value can enter the field.
    assert _RAW_FAILURE_SENTINEL not in field
    assert "/" not in field and "\\" not in field
    assert 0 < v2._serialized_string_bytes(field) <= v2.REFUSAL_FIELD_MAX_BYTES


def test_collector_raw_assembly_failure_field_localizes_without_position(
        monkeypatch):
    field = _raw_failure_refusal(
        monkeypatch, "_pack_index_pages", _assembly_site_raw_raiser)
    assert field != "TypeError"
    class_name, qualified, location = field.split(":")
    assert class_name == "TypeError"
    assert qualified.endswith("._assembly_site_raw_raiser")
    # A capture/validation assembly phase has no admitted row, so no position.
    assert "@row" not in field
    assert location.isdigit() and int(location) > 0
    assert _RAW_FAILURE_SENTINEL not in field
    assert "/" not in field and "\\" not in field
    assert 0 < v2._serialized_string_bytes(field) <= v2.REFUSAL_FIELD_MAX_BYTES


def test_collector_raw_sidecar_failure_field_localizes_row_construction(
        monkeypatch):
    # The row-local sidecar construction/serialization site sits after
    # ``_resolve_row_support``'s own local payload catch, so its raw failure must
    # still escape to the broad arm and be localized with the admitted row.
    field = _raw_failure_refusal(
        monkeypatch, "MatcherSupportSidecarEntryV2", _sidecar_site_raw_raiser)
    class_name, qualified, location = field.split(":")
    assert class_name == "ValueError"
    assert qualified.endswith("._sidecar_site_raw_raiser")
    assert location.endswith("@row0")
    assert _RAW_FAILURE_SENTINEL not in field
    assert "/" not in field and "\\" not in field
    assert 0 < v2._serialized_string_bytes(field) <= v2.REFUSAL_FIELD_MAX_BYTES


def test_collector_raw_failure_fields_distinguish_row_and_assembly_sites():
    with pytest.MonkeyPatch.context() as patch:
        row_field = _raw_failure_refusal(
            patch, "_alternative_index", _row_site_raw_raiser)
    with pytest.MonkeyPatch.context() as patch:
        assembly_field = _raw_failure_refusal(
            patch, "_pack_index_pages", _assembly_site_raw_raiser)
    assert row_field != assembly_field
    assert row_field != "ValueError" and assembly_field != "TypeError"
    assert row_field.endswith("@row0") and "@row" not in assembly_field


def test_collector_typed_candidate_support_error_field_is_unchanged(monkeypatch):
    def _typed_site_error(*args, **kwargs):
        raise v2.CandidateSupportError("index_row_over_cap", "typed_control_field")

    monkeypatch.setattr(v2, "_alternative_index", _typed_site_error)
    matches, witness_rows, dispositions, bindings = _collector_material(
        witnesses=True)
    capture = v2.collect_complete_candidate_support(
        _readings_for(matches), matches, bindings, witness_rows,
        _observation(matches, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=2)
    # The typed arm is untouched: the project reason/field pair is preserved
    # verbatim and is never rewritten into a raw localization code.
    assert capture.status == "refused"
    assert capture.root.reason == "index_row_over_cap"
    assert capture.root.field == "typed_control_field"
    assert capture.row_count == 0 and capture.index_pages == ()
    v2.validate_complete_support_capture(capture)


def test_collector_raw_failure_field_canonicalizes_custom_exception_subclass(
        monkeypatch):
    # A custom subclass can carry a caller-controlled ``__name__`` (here a
    # path-like ``C:/sentinel``); the persisted field must expose only the fixed
    # canonical built-in category, never the subclass name.
    leaky_error = type("C:/sentinel", (ValueError,), {})

    def _leaky_row_site(*args, **kwargs):
        raise leaky_error(_RAW_FAILURE_SENTINEL)

    field = _raw_failure_refusal(
        monkeypatch, "_alternative_index", _leaky_row_site)
    assert field.startswith("ValueError:")
    assert "C:/sentinel" not in field
    assert "/" not in field and "\\" not in field
    class_name, qualified, location = field.split(":")
    assert class_name == "ValueError"
    # The module/function/line and admitted-row localization is retained.
    assert qualified.endswith("._leaky_row_site")
    assert location.endswith("@row0")
    line_component = location[: -len("@row0")]
    assert line_component.isdigit() and int(line_component) > 0
    assert _RAW_FAILURE_SENTINEL not in field
    assert 0 < v2._serialized_string_bytes(field) <= v2.REFUSAL_FIELD_MAX_BYTES


def test_build_capture_raw_field_canonicalizes_custom_exception_subclass(
        monkeypatch):
    # The sibling pre-built packer arm emits the same refusal field; it must
    # canonicalize a custom subclass name identically.
    leaky_error = type("C:/sentinel", (ValueError,), {})

    def _leaky_assembly(*args, **kwargs):
        raise leaky_error(_RAW_FAILURE_SENTINEL)

    monkeypatch.setattr(v2, "_assemble_complete_support", _leaky_assembly)
    rows, sections, entries = _universe(2, present=1)
    capture = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=2)
    assert capture.status == "refused"
    assert capture.root.reason == "universe_identity_mismatch"
    assert capture.root.field == "ValueError"
    assert "C:/sentinel" not in capture.root.field
    assert "/" not in capture.root.field and "\\" not in capture.root.field
    assert _RAW_FAILURE_SENTINEL not in capture.root.field
    assert capture.row_count == 0 and capture.index_pages == ()
    v2.validate_complete_support_capture(capture)


def test_capture_refuses_excessive_ingress_atomically():
    rows, sections, entries = _universe(5, present=2)
    valid = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5, ingress=(_raw_ingress(),))
    assert valid.status == "complete", valid.root.reason

    too_many = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5,
        ingress=tuple(_raw_ingress()
                      for _ in range(v2.INGRESS_MAX_RECORDS + 1)))
    assert too_many.status == "refused"
    assert too_many.root.reason == "ingress_over_cap"
    assert too_many.root.field == "ingress_count"
    assert too_many.row_count == 0 and too_many.index_pages == ()
    assert too_many.sidecar_pages == () and too_many.digest_pages == ()

    long_id = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5,
        ingress=(_raw_ingress(
            frame_id="z" * (v2.INGRESS_MAX_FRAME_ID_BYTES + 1)),))
    assert long_id.status == "refused"
    assert long_id.root.reason == "ingress_over_cap"
    assert long_id.root.field.endswith("frame_id")

    huge_transform = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5,
        ingress=(_raw_ingress(transform=FrameTransform(
            source_size=(2, 2),
            crop_origin=(v2.INGRESS_MAX_ABS_INT + 1, 0))),))
    assert huge_transform.status == "refused"
    assert huge_transform.root.reason == "ingress_over_cap"
    assert huge_transform.root.field.endswith("crop_origin")


# ---------------------------------------------------------------------------
# Adversarial controls (DS-R4-V2-ROOT-CAPS-REMEDIATION)
# ---------------------------------------------------------------------------

def _ingress_with_transform(
        transform: FrameTransform) -> csp.FrameIngressProvenanceV1:
    """One legal pin with its transform replaced by an arbitrary tuple shape.

    ``FrameIngressProvenanceV1`` pins its transform at construction, but
    ``object.__setattr__`` can still replace the slot afterwards. This models a
    tampered or foreign record that the V2 bound check must refuse without ever
    reading image/truth/camera material.
    """
    record = _raw_ingress()
    object.__setattr__(record, "transform", transform)
    return record


def _refuses_ingress_field(record, expected_field: str) -> None:
    rows, sections, entries = _universe(3, present=0)
    capture = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=3, ingress=(record,))
    assert capture.status == "refused"
    assert capture.root.reason == "ingress_over_cap"
    assert capture.root.field == f"ingress.0.{expected_field}"
    assert capture.row_count == 0 and capture.index_pages == ()
    assert capture.digest_pages == () and capture.sidecar_pages == ()


def test_ingress_transform_arity_and_dimension_bounds():
    dim = v2.INGRESS_MAX_SHAPE_DIM
    abs_int = v2.INGRESS_MAX_ABS_INT
    maximal = _raw_ingress(
        decoded_shape=(dim, dim),
        transform=FrameTransform(
            source_size=(dim, dim), crop_origin=(-abs_int, abs_int),
            crop_size=(dim, dim), output_size=(dim, dim)))
    assert v2._ingress_record_over_cap(maximal) == ""

    # Luna's finding: with an explicit two-value output_size the other size
    # tuples and crop_origin can be arbitrarily long and still construct a legal
    # V1 pin. They must now be refused by arity.
    over_arity = (
        ("source_size", _raw_ingress(transform=FrameTransform(
            source_size=(2, 2, 2), output_size=(2, 2)))),
        ("crop_origin", _raw_ingress(transform=FrameTransform(
            source_size=(2, 2), crop_origin=(0, 0, 0), output_size=(2, 2)))),
        ("crop_size", _raw_ingress(transform=FrameTransform(
            source_size=(2, 2), crop_size=(2, 2, 2), output_size=(2, 2)))),
        ("source_size", _raw_ingress(transform=FrameTransform(
            source_size=(2,), output_size=(2, 2)))),
        ("output_size", _ingress_with_transform(FrameTransform(
            source_size=(2, 2), output_size=(2, 2, 2)))),
    )
    for field, record in over_arity:
        assert v2._ingress_record_over_cap(record) == field
        _refuses_ingress_field(record, field)

    # Size dimensions must be positive and bounded by INGRESS_MAX_SHAPE_DIM,
    # not the much wider INGRESS_MAX_ABS_INT.
    over_dimension = (
        ("source_size", _raw_ingress(transform=FrameTransform(
            source_size=(dim + 1, 2), output_size=(2, 2)))),
        ("source_size", _raw_ingress(transform=FrameTransform(
            source_size=(0, 2), output_size=(2, 2)))),
        ("source_size", _raw_ingress(transform=FrameTransform(
            source_size=(abs_int, 2), output_size=(2, 2)))),
        ("crop_size", _raw_ingress(transform=FrameTransform(
            source_size=(2, 2), crop_size=(dim + 1, 2), output_size=(2, 2)))),
        ("crop_size", _raw_ingress(transform=FrameTransform(
            source_size=(2, 2), crop_size=(0, 2), output_size=(2, 2)))),
        ("crop_origin", _raw_ingress(transform=FrameTransform(
            source_size=(2, 2), crop_origin=(abs_int + 1, 0),
            output_size=(2, 2)))),
        ("output_size", _ingress_with_transform(FrameTransform(
            source_size=(2, 2), output_size=(dim + 1, 2)))),
        ("output_size", _ingress_with_transform(FrameTransform(
            source_size=(2, 2), output_size=(0, 2)))),
    )
    for field, record in over_dimension:
        assert v2._ingress_record_over_cap(record) == field
        _refuses_ingress_field(record, field)


def test_ingress_decoded_shape_arity_and_bounds():
    dim = v2.INGRESS_MAX_SHAPE_DIM
    assert v2._decoded_shape_over_cap((dim, dim, 3)) is False
    assert v2._decoded_shape_over_cap((dim, dim)) is False
    assert v2._decoded_shape_over_cap((dim, dim, 3, 3)) is True
    assert v2._decoded_shape_over_cap((0, dim)) is True
    assert v2._decoded_shape_over_cap((dim + 1, dim)) is True
    assert v2._decoded_shape_over_cap((dim, dim, 4)) is True
    legal = _raw_ingress(decoded_shape=(dim, dim, 3),
                         transform=FrameTransform(source_size=(dim, dim)))
    assert v2._ingress_record_over_cap(legal) == ""
    tampered = _raw_ingress()
    object.__setattr__(tampered, "decoded_shape", (dim, dim, 3, 3))
    assert v2._ingress_record_over_cap(tampered) == "decoded_shape"
    _refuses_ingress_field(tampered, "decoded_shape")


def test_preflight_ingress_measurement_matches_the_declared_maximum():
    preflight = v2.preflight_complete_support()
    record = v2._synthetic_ingress_record()
    assert v2._ingress_record_over_cap(record) == ""
    assert record.decoded_shape == (
        v2.INGRESS_MAX_SHAPE_DIM, v2.INGRESS_MAX_SHAPE_DIM, 3)
    assert record.transform.source_size == (
        v2.INGRESS_MAX_SHAPE_DIM, v2.INGRESS_MAX_SHAPE_DIM)
    assert record.transform.crop_origin == (
        -v2.INGRESS_MAX_ABS_INT, -v2.INGRESS_MAX_ABS_INT)
    assert (len(v2._canonical_bytes(record.to_record()))
            == preflight.measurement("ingress_record_worst_case_bytes"))
    # No legal record can exceed this canonical width: a source size at the
    # wider INGRESS_MAX_ABS_INT bound is refused, and the three-dimensional
    # decoded shape is the largest arity the V1 shape contract permits.
    wider = _raw_ingress(transform=FrameTransform(
        source_size=(v2.INGRESS_MAX_ABS_INT, 2), output_size=(2, 2)))
    assert v2._ingress_record_over_cap(wider) == "source_size"


def test_refused_root_reason_and_field_are_bounded():
    # A planned refusal bounds its field and keeps a bounded reason code; the
    # over-long field is truncated rather than raising, so the refusal stays
    # atomic.
    plan = v2.CompleteSupportCapPlanV1()
    capture = v2.refused_complete_support("universe_identity_mismatch",
                                          "z" * 10_000, plan=plan)
    root = capture.root
    assert root.status == "refused"
    assert 0 < v2._serialized_string_bytes(root.field) <= v2.REFUSAL_FIELD_MAX_BYTES
    assert root.reason in v2.CAPTURE_REFUSAL_REASONS
    assert root.cap_plan is not None and root.cap_plan_sha256
    assert root.to_record()["cap_plan"] == plan.as_record()
    assert len(v2._canonical_bytes(root.to_record())) <= v2.ROOT_MAX_BYTES
    v2.validate_complete_support_capture(capture)

    # Unknown reasons and non-string/empty fields sanitize instead of raising,
    # so an internal refusal stays atomic.
    assert v2.refused_complete_support(
        "not_a_reason", "", plan=plan).root.reason in v2.CAPTURE_REFUSAL_REASONS
    assert v2.refused_complete_support(
        "row_cap_exceeded", "", plan=plan).root.field == "record"
    assert v2.refused_complete_support(
        "row_cap_exceeded", 123, plan=plan).root.field == "record"

    # A planless refusal is exactly the fixed configuration-invalid pair, no
    # matter what reason/field a caller supplies.
    coerced = v2.refused_complete_support("row_cap_exceeded", "row_count")
    assert coerced.root.reason == v2.PLANLESS_REFUSAL_REASON
    assert coerced.root.field == v2.PLANLESS_REFUSAL_FIELD
    assert coerced.root.cap_plan is None and coerced.root.cap_plan_sha256 == ""
    assert (len(v2._canonical_bytes(coerced.root.to_record()))
            <= v2.PLANLESS_REFUSAL_ROOT_MAX_BYTES)
    v2.validate_complete_support_capture(coerced)

    planless_kwargs = dict(
        status="refused", basis=None, reason=v2.PLANLESS_REFUSAL_REASON,
        field=v2.PLANLESS_REFUSAL_FIELD, failed_assignment_sha256="",
        whole_universe_identity_sha256="", input_match_identity_sha256="",
        row_count=0, declared_universe_cap=0, index_pages=(), sidecar_pages=(),
        digest_sections=(), ingress=(), cap_plan=None)
    # The fixed pair constructs and validates ...
    v2.CandidateSupportCaptureRootV2(**planless_kwargs)
    # ... but any other planless code is rejected at construction.
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(
            **{**planless_kwargs, "reason": "row_cap_exceeded",
               "field": "row_count"})
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(
            **{**planless_kwargs, "reason": "not_a_reason"})
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(
            **{**planless_kwargs,
               "field": "z" * (v2.REFUSAL_FIELD_MAX_BYTES + 1)})
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(**{**planless_kwargs, "field": ""})
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(**{**planless_kwargs, "field": 7})
    # A planless refusal may not claim a plan digest, and a plan-carrying refusal
    # must carry a real plan with a replaying digest.
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(
            **{**planless_kwargs, "cap_plan_sha256": "0" * 64})
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(
            **{**planless_kwargs, "cap_plan": "not-a-plan"})
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(
            **{**planless_kwargs, "cap_plan": v2.CompleteSupportCapPlanV1(),
               "cap_plan_sha256": "0" * 64})


def test_planless_refusal_is_restricted_to_the_fixed_configuration_pair():
    # A planless root mutated to an ordinary refusal code cannot validate even
    # when both its own digests are recomputed consistently.
    def resign(root):
        object.__setattr__(root, "record_sha256", v2._digest(
            v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
            v2._root_payload(root, include_digest=False)))

    reason_swap = v2.refused_complete_support("row_cap_exceeded", "row_count")
    assert reason_swap.root.reason == v2.PLANLESS_REFUSAL_REASON
    object.__setattr__(reason_swap.root, "reason", "row_cap_exceeded")
    resign(reason_swap.root)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(reason_swap)

    field_swap = v2.refused_complete_support("row_cap_exceeded", "row_count")
    object.__setattr__(field_swap.root, "field", "row_count")
    resign(field_swap.root)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(field_swap)

    # Every capture-level reason other than the fixed pair is refused when the
    # root is planless, including reasons that are otherwise legal.
    for reason in v2.CAPTURE_REFUSAL_REASONS:
        capture = v2.refused_complete_support("row_cap_exceeded", "row_count")
        object.__setattr__(capture.root, "reason", reason)
        resign(capture.root)
        if reason == v2.PLANLESS_REFUSAL_REASON:
            v2.validate_complete_support_capture(capture)
        else:
            with pytest.raises(ValueError):
                v2.validate_complete_support_capture(capture)

    # A planned refusal with the same ordinary reason still validates.
    planned = v2.refused_complete_support("row_cap_exceeded", "row_count",
                                          plan=v2.CompleteSupportCapPlanV1())
    assert planned.root.reason == "row_cap_exceeded"
    v2.validate_complete_support_capture(planned)


def test_refusal_preserves_the_resolved_cap_plan_and_stays_bounded():
    plan = v2.CompleteSupportCapPlanV1(
        index_page_max_bytes=200_000, digest_page_max_bytes=200_000,
        sidecar_page_max_bytes=200_000, sidecar_entry_max_raw_bytes=131_072,
        sidecar_entry_max_bytes=150_000, sidecar_aggregate_max_bytes=1_000_000,
        root_max_bytes=300_000, universe_max_rows=8)
    refused = v2.refused_complete_support("row_cap_exceeded", "row_count",
                                          plan=plan)
    root = refused.root
    assert root.cap_plan is not None
    assert root.cap_plan.as_record() == plan.as_record()
    assert root.cap_plan_sha256 == v2._digest(
        v2.CAP_PLAN_DIGEST_DOMAIN, plan.as_record())
    assert root.to_record()["cap_plan"] == plan.as_record()
    assert len(v2._canonical_bytes(root.to_record())) <= v2.ROOT_MAX_BYTES
    v2.validate_complete_support_capture(refused)

    # A substituted plan or a stale plan digest cannot be re-signed into
    # acceptance, because the validator replays the plan digest itself.
    swapped = v2.refused_complete_support("row_cap_exceeded", "row_count",
                                          plan=plan)
    object.__setattr__(swapped.root, "cap_plan",
                       v2.CompleteSupportCapPlanV1(universe_max_rows=64))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(swapped)

    stale = v2.refused_complete_support("row_cap_exceeded", "row_count",
                                        plan=plan)
    object.__setattr__(stale.root, "cap_plan_sha256", "0" * 64)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(stale)


def test_invalid_plan_resolution_yields_the_planless_refusal():
    rows, sections, entries = _universe(5, present=1)
    for bad_plan in ("not-a-plan", 7, object()):
        capture = v2.build_complete_support_capture(
            rows=rows, digest_sections=sections, sidecar_entries=entries,
            expected_row_count=5, plan=bad_plan)
        assert capture.status == "refused"
        assert capture.root.reason == "cap_configuration_invalid"
        assert capture.root.field == "cap_plan"
        assert capture.root.cap_plan is None
        assert capture.root.cap_plan_sha256 == ""
        assert (len(v2._canonical_bytes(capture.root.to_record()))
                <= v2.PLANLESS_REFUSAL_ROOT_MAX_BYTES)
        v2.validate_complete_support_capture(capture)

    # A resolved plan that merely fails a cap refuses with that plan preserved.
    planned = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5, plan=v2.CompleteSupportCapPlanV1(root_max_bytes=1))
    assert planned.status == "refused"
    assert planned.root.reason == "root_over_cap"
    assert planned.root.cap_plan is not None
    assert len(v2._canonical_bytes(planned.root.to_record())) <= v2.ROOT_MAX_BYTES
    v2.validate_complete_support_capture(planned)


def test_preflight_measures_the_maximal_refused_roots():
    preflight = v2.preflight_complete_support()
    measured = preflight.measurement("refused_root_worst_case_bytes")
    planless = preflight.measurement("planless_refused_root_worst_case_bytes")
    assert measured > 0 and planless > 0

    maximal_plan = v2.CompleteSupportCapPlanV1(
        index_page_max_bytes=v2.CAP_PLAN_MAX_ABS_INT,
        digest_page_max_bytes=v2.CAP_PLAN_MAX_ABS_INT,
        sidecar_page_max_bytes=v2.CAP_PLAN_MAX_ABS_INT,
        sidecar_entry_max_raw_bytes=v2.CAP_PLAN_MAX_ABS_INT,
        sidecar_entry_max_bytes=v2.CAP_PLAN_MAX_ABS_INT,
        sidecar_aggregate_max_bytes=v2.CAP_PLAN_MAX_ABS_INT,
        root_max_bytes=v2.CAP_PLAN_MAX_ABS_INT,
        universe_max_rows=v2.CAP_PLAN_MAX_ABS_INT)
    planned = v2.CandidateSupportCaptureRootV2(
        status="refused", basis=None,
        reason=max(v2.CAPTURE_REFUSAL_REASONS, key=len),
        field="z" * v2.REFUSAL_FIELD_MAX_BYTES,
        failed_assignment_sha256="", whole_universe_identity_sha256="",
        input_match_identity_sha256="", row_count=0, declared_universe_cap=0,
        index_pages=(), sidecar_pages=(), digest_sections=(), ingress=(),
        cap_plan=maximal_plan)
    assert len(v2._canonical_bytes(planned.to_record())) == measured
    assert measured <= v2.ROOT_MAX_BYTES

    planless_root = v2.CandidateSupportCaptureRootV2(
        status="refused", basis=None, reason=v2.PLANLESS_REFUSAL_REASON,
        field=v2.PLANLESS_REFUSAL_FIELD,
        failed_assignment_sha256="", whole_universe_identity_sha256="",
        input_match_identity_sha256="", row_count=0, declared_universe_cap=0,
        index_pages=(), sidecar_pages=(), digest_sections=(), ingress=(),
        cap_plan=None)
    assert len(v2._canonical_bytes(planless_root.to_record())) == planless
    assert planless <= v2.PLANLESS_REFUSAL_ROOT_MAX_BYTES
    assert measured > planless
    assert measured < preflight.measurement("root_worst_case_bytes")


def _complete_universe(count: int = 3, present: int = 1):
    rows, sections, entries = _universe(count, present=present)
    capture = _assemble(rows, sections, entries, expected_row_count=count)
    assert capture.status == "complete", capture.root.reason
    v2.validate_complete_support_capture(capture)
    return capture


def test_content_replay_rejects_index_and_digest_page_mutation():
    # Index page: a row field mutated behind the frozen cached page hash.
    index_tamper = _complete_universe()
    object.__setattr__(index_tamper.index_pages[0].rows[0], "overlap_cells",
                       index_tamper.index_pages[0].rows[0].overlap_cells + 1)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(index_tamper)

    # Index page: the page payload replaced by a reordered tuple of its own rows.
    reordered = _complete_universe()
    object.__setattr__(reordered.index_pages[0], "rows",
                       tuple(reversed(reordered.index_pages[0].rows)))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(reordered)

    # Digest page: one digest value mutated in place.
    digest_tamper = _complete_universe()
    page = next(item for item in digest_tamper.digest_pages
                if item.section == "match_sha256")
    object.__setattr__(page, "digests", ("0" * 64,) + page.digests[1:])
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(digest_tamper)

    # Digest page: the cached page sha is rewritten but the content is not.
    sha_tamper = _complete_universe()
    object.__setattr__(sha_tamper.digest_pages[0], "page_sha256", "0" * 64)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(sha_tamper)

    # An untouched builder-produced capture still replays.
    v2.validate_complete_support_capture(_complete_universe())


def test_content_replay_rejects_sidecar_array_and_entry_mutation():
    # Sidecar page: one component array swapped for a different same-shape,
    # read-only array; the stale cached component digest must not mask it.
    array_tamper = _complete_universe(count=3, present=1)
    entry = array_tamper.sidecar_pages[0].entries[0]
    swapped = np.ones(entry.payload.warp_source_rows.shape, dtype=np.int64)
    swapped.flags.writeable = False
    object.__setattr__(entry.payload, "warp_source_rows", swapped)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(array_tamper)

    # Sidecar entry: a stale component-array digest is recomputed and rejected.
    digest_tamper = _complete_universe(count=3, present=1)
    object.__setattr__(digest_tamper.sidecar_pages[0].entries[0],
                       "logical_pairs_sha256", "0" * 64)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(digest_tamper)

    # Sidecar entry: the payload's canonical digest is recomputed from current
    # arrays and re-bound to the index row.
    payload_tamper = _complete_universe(count=3, present=1)
    entry = payload_tamper.sidecar_pages[0].entries[0]
    object.__setattr__(entry, "payload_sha256", "0" * 64)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(payload_tamper)

    # Sidecar entry: a recorded shape/count commitment is replayed.
    shape_tamper = _complete_universe(count=3, present=1)
    entry = shape_tamper.sidecar_pages[0].entries[0]
    object.__setattr__(entry, "source_shape", (1, 1))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(shape_tamper)

    overlap_tamper = _complete_universe(count=3, present=1)
    entry = overlap_tamper.sidecar_pages[0].entries[0]
    object.__setattr__(entry, "overlap_cells", entry.overlap_cells + 1)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(overlap_tamper)

    # Sidecar entry: an entire same-shape payload substituted for another.
    payload_swap = _complete_universe(count=3, present=1)
    entry = payload_swap.sidecar_pages[0].entries[0]
    replacement = _small_payload(pairs=1)
    object.__setattr__(entry, "payload", replacement)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(payload_swap)

    # The untouched builder-produced capture still replays.
    v2.validate_complete_support_capture(
        _complete_universe(count=3, present=1))


def test_preflight_sidecar_entry_uses_the_last_admitted_index():
    preflight = v2.preflight_complete_support()
    raw_cap = v2.SIDECAR_ENTRY_MAX_RAW_BYTES
    last = v2._synthetic_sidecar_entry(raw_cap, v2.UNIVERSE_MAX_ROWS - 1)
    first = v2._synthetic_sidecar_entry(raw_cap, 0)
    assert last.admitted_row_index == v2.UNIVERSE_MAX_ROWS - 1
    assert (len(v2._canonical_bytes(last.to_record()))
            == preflight.measurement("sidecar_entry_worst_case_bytes"))
    assert (len(v2._canonical_bytes(last.to_record()))
            > len(v2._canonical_bytes(first.to_record())))
    page = v2._make_sidecar_page((last,), 0, v2.UNIVERSE_MAX_ROWS - 1)
    assert (len(v2._canonical_bytes(page.to_record()))
            <= preflight.measurement("sidecar_page_worst_case_bytes"))


def test_root_persists_and_hashes_the_full_cap_plan():
    rows, sections, entries = _universe(5, present=2)
    plan = v2.CompleteSupportCapPlanV1(
        index_page_max_bytes=200_000, digest_page_max_bytes=200_000,
        sidecar_page_max_bytes=300_000, sidecar_entry_max_raw_bytes=131_072,
        sidecar_entry_max_bytes=300_000, sidecar_aggregate_max_bytes=2_000_000,
        root_max_bytes=1_000_000, universe_max_rows=8)
    capture = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=5, plan=plan)
    assert capture.status == "complete", capture.root.reason
    root = capture.root
    assert root.cap_plan.as_record() == plan.as_record()
    assert root.cap_plan_sha256 == v2._digest(
        v2.CAP_PLAN_DIGEST_DOMAIN, plan.as_record())
    record = root.to_record()
    assert record["cap_plan"] == plan.as_record()
    assert set(record["cap_plan"]) == {
        "index_page_max_bytes", "digest_page_max_bytes", "sidecar_page_max_bytes",
        "sidecar_entry_max_raw_bytes", "sidecar_entry_max_bytes",
        "sidecar_aggregate_max_bytes", "root_max_bytes", "universe_max_rows"}
    # The builder writes exactly the plan it was given.
    assert record["declared_universe_cap"] == plan.universe_max_rows
    v2.validate_complete_support_capture(capture)


def test_tampered_or_missing_cap_plan_cannot_validate():
    rows, sections, entries = _universe(5, present=2)

    missing = _assemble(rows, sections, entries, expected_row_count=5)
    assert missing.status == "complete"
    object.__setattr__(missing.root, "cap_plan", None)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(missing)

    swapped = _assemble(rows, sections, entries, expected_row_count=5)
    object.__setattr__(swapped.root, "cap_plan",
                       v2.CompleteSupportCapPlanV1(universe_max_rows=64))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(swapped)

    # Recomputing only the root record digest still leaves the stale cap-plan
    # digest, so a tampered plan cannot be re-signed into acceptance.
    resigned = _assemble(rows, sections, entries, expected_row_count=5)
    object.__setattr__(resigned.root, "cap_plan",
                       v2.CompleteSupportCapPlanV1(universe_max_rows=64))
    object.__setattr__(resigned.root, "record_sha256", v2._digest(
        v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
        v2._root_payload(resigned.root, include_digest=False)))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(resigned)

    # A stale root that no longer matches its own bound plan digest refuses.
    stale_plan = _assemble(rows, sections, entries, expected_row_count=5)
    object.__setattr__(stale_plan.root, "cap_plan_sha256", "0" * 64)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(stale_plan)


def test_manually_constructed_over_cap_captures_cannot_validate():
    rows, sections, entries = _universe(5, present=2)
    capture = _assemble(rows, sections, entries, expected_row_count=5)
    assert capture.status == "complete"
    base_plan = capture.root.cap_plan

    def manual(plan):
        root = dataclasses.replace(capture.root, cap_plan=plan,
                                   cap_plan_sha256="", record_sha256="")
        rebuilt = object.__new__(v2.CompleteSupportCaptureV2)
        object.__setattr__(rebuilt, "root", root)
        object.__setattr__(rebuilt, "index_pages", capture.index_pages)
        object.__setattr__(rebuilt, "digest_pages", capture.digest_pages)
        object.__setattr__(rebuilt, "sidecar_pages", capture.sidecar_pages)
        object.__setattr__(rebuilt, "schema", v2.COMPLETE_SUPPORT_CAPTURE_SCHEMA)
        return rebuilt

    strict = (
        dataclasses.replace(base_plan, index_page_max_bytes=10),
        dataclasses.replace(base_plan, digest_page_max_bytes=10),
        dataclasses.replace(base_plan, sidecar_page_max_bytes=10),
        dataclasses.replace(base_plan, sidecar_entry_max_raw_bytes=1),
        dataclasses.replace(base_plan, sidecar_entry_max_bytes=1),
        dataclasses.replace(base_plan, sidecar_aggregate_max_bytes=1),
        dataclasses.replace(base_plan, root_max_bytes=1),
    )
    for plan in strict:
        with pytest.raises(ValueError):
            v2.validate_complete_support_capture(manual(plan))

    # The same manually rebuilt capture validates only when every declared cap is
    # restored, so the refusal above is the cap check and not a construction bug.
    v2.validate_complete_support_capture(manual(base_plan))


def test_cap_plan_fields_are_replay_validated_after_tampering():
    rows, sections, entries = _universe(3, present=0)
    assert entries == ()
    capture = _assemble(rows, sections, entries, expected_row_count=3)
    assert capture.status == "complete", capture.root.reason
    v2.validate_complete_support_capture(capture)

    def resign(root):
        # Recompute both digests from the tampered object so the only remaining
        # defect is the plan value itself.
        object.__setattr__(root, "cap_plan_sha256", v2._digest(
            v2.CAP_PLAN_DIGEST_DOMAIN, root.cap_plan.as_record()))
        object.__setattr__(root, "record_sha256", v2._digest(
            v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
            v2._root_payload(root, include_digest=False)))

    for name in v2.CAP_PLAN_FIELDS:
        for bad in (0, -1, 1.5, "1", v2.CAP_PLAN_MAX_ABS_INT + 1):
            tampered = _assemble(rows, sections, entries, expected_row_count=3)
            object.__setattr__(tampered.root.cap_plan, name, bad)
            resign(tampered.root)
            # Both digests are internally consistent, so only the plan-value
            # re-validation can reject the fully re-signed capture.
            assert tampered.root.cap_plan_sha256 == v2._digest(
                v2.CAP_PLAN_DIGEST_DOMAIN, tampered.root.cap_plan.as_record())
            assert tampered.root.record_sha256 == v2._digest(
                v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
                v2._root_payload(tampered.root, include_digest=False))
            with pytest.raises(ValueError):
                v2.validate_complete_support_capture(tampered)

    # A fully re-signed invalid plan on a planned refusal is also rejected.
    refused = v2.refused_complete_support("row_cap_exceeded", "row_count",
                                          plan=v2.CompleteSupportCapPlanV1())
    object.__setattr__(refused.root.cap_plan, "sidecar_entry_max_bytes", 0)
    resign(refused.root)
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(refused)

    # A re-signed invalid plan is a plan-resolution failure, so a request that
    # carries it refuses planless instead of assembling under it.
    bad_plan = v2.CompleteSupportCapPlanV1()
    object.__setattr__(bad_plan, "sidecar_page_max_bytes", 0)
    assert v2._resolve_plan(bad_plan) is None
    refused_build = v2.build_complete_support_capture(
        rows=rows, digest_sections=sections, sidecar_entries=entries,
        expected_row_count=3, plan=bad_plan)
    assert refused_build.status == "refused"
    assert refused_build.root.reason == v2.PLANLESS_REFUSAL_REASON
    assert refused_build.root.field == v2.PLANLESS_REFUSAL_FIELD
    assert refused_build.root.cap_plan is None
    v2.validate_complete_support_capture(refused_build)

    # Untouched builder-produced captures still validate.
    v2.validate_complete_support_capture(
        _assemble(rows, sections, entries, expected_row_count=3))

    # A plan whose canonical record has a missing or extra serialized field is
    # rejected even though it is still an instance of the plan class.
    class _ExtraFieldPlan(v2.CompleteSupportCapPlanV1):
        def as_record(self):
            return {**super().as_record(), "extra_cap": 1}

    class _MissingFieldPlan(v2.CompleteSupportCapPlanV1):
        def as_record(self):
            record = super().as_record()
            del record["universe_max_rows"]
            return record

    for malformed in (_ExtraFieldPlan(), _MissingFieldPlan()):
        assert v2._cap_plan_field_over_cap(malformed) == "plan"
        assert v2._resolve_plan(malformed) is None
        refused_plan = v2.refused_complete_support(
            "row_cap_exceeded", "row_count", plan=malformed)
        assert refused_plan.root.reason == v2.PLANLESS_REFUSAL_REASON
        assert refused_plan.root.field == v2.PLANLESS_REFUSAL_FIELD
        assert refused_plan.root.cap_plan is None
        v2.validate_complete_support_capture(refused_plan)
        # A refused root cannot carry the malformed plan directly either.
        with pytest.raises(ValueError):
            v2.CandidateSupportCaptureRootV2(
                status="refused", basis=None, reason="row_cap_exceeded",
                field="row_count", failed_assignment_sha256="",
                whole_universe_identity_sha256="",
                input_match_identity_sha256="", row_count=0,
                declared_universe_cap=0, index_pages=(), sidecar_pages=(),
                digest_sections=(), ingress=(), cap_plan=malformed)


def test_subclass_plan_as_record_spoof_cannot_hide_invalid_live_caps():
    """Luna HOLD #5: ``as_record`` is overridable; live cap attributes are not.

    A ``CompleteSupportCapPlanV1`` subclass can return a default-valid serialized
    mapping while its real sidecar caps are zero. Runtime enforcement and the
    cap-plan digest bind the live attributes, not the overridable mapping, so such
    a plan must be refused in construction, resolution and replay even after both
    digests are recomputed from the spoofed mapping.
    """
    rows, sections, entries = _universe(3, present=0)
    assert entries == ()
    default_record = v2.CompleteSupportCapPlanV1().as_record()

    class _SpoofedPlan(v2.CompleteSupportCapPlanV1):
        def as_record(self):
            # Always emits default-valid values, hiding the live caps set below.
            return v2.CompleteSupportCapPlanV1().as_record()

    spoof = _SpoofedPlan()
    for name in ("sidecar_page_max_bytes", "sidecar_entry_max_raw_bytes",
                 "sidecar_entry_max_bytes", "sidecar_aggregate_max_bytes"):
        object.__setattr__(spoof, name, 0)
    # The spoofed record claims a valid default plan while the live caps are 0.
    assert spoof.as_record() == default_record
    assert spoof.sidecar_page_max_bytes == 0

    # The predicate and resolver validate the exact class and live attributes,
    # not the overridable mapping.
    assert v2._cap_plan_field_over_cap(spoof) == "plan"
    assert v2._resolve_plan(spoof) is None

    # A request carrying the spoof cannot assemble under it; it refuses planless.
    refused = v2.refused_complete_support("row_cap_exceeded", "row_count",
                                          plan=spoof)
    assert refused.status == "refused"
    assert refused.root.reason == v2.PLANLESS_REFUSAL_REASON
    assert refused.root.field == v2.PLANLESS_REFUSAL_FIELD
    assert refused.root.cap_plan is None
    v2.validate_complete_support_capture(refused)

    # A refused root cannot carry the spoofed plan directly either.
    with pytest.raises(ValueError):
        v2.CandidateSupportCaptureRootV2(
            status="refused", basis=None, reason="row_cap_exceeded",
            field="row_count", failed_assignment_sha256="",
            whole_universe_identity_sha256="",
            input_match_identity_sha256="", row_count=0,
            declared_universe_cap=0, index_pages=(), sidecar_pages=(),
            digest_sections=(), ingress=(), cap_plan=spoof)

    # A fully re-signed complete capture cannot hide the spoof either: both
    # digests are recomputed from the spoofed mapping, so only the live-attribute
    # revalidation can reject it (Luna's zero-sidecar repro).
    tampered = _assemble(rows, sections, entries, expected_row_count=3)
    assert tampered.status == "complete"
    object.__setattr__(tampered.root, "cap_plan", spoof)
    object.__setattr__(tampered.root, "cap_plan_sha256", v2._digest(
        v2.CAP_PLAN_DIGEST_DOMAIN, spoof.as_record()))
    object.__setattr__(tampered.root, "record_sha256", v2._digest(
        v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
        v2._root_payload(tampered.root, include_digest=False)))
    # Both digests are internally consistent with the spoofed record, so
    # acceptance would require trusting the overridable mapping.
    assert tampered.root.cap_plan_sha256 == v2._digest(
        v2.CAP_PLAN_DIGEST_DOMAIN, tampered.root.cap_plan.as_record())
    assert tampered.root.record_sha256 == v2._digest(
        v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
        v2._root_payload(tampered.root, include_digest=False))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(tampered)

    # The same re-signed spoof on a planned refusal is rejected too.
    planned = v2.refused_complete_support(
        "row_cap_exceeded", "row_count", plan=v2.CompleteSupportCapPlanV1())
    assert planned.root.cap_plan is not None
    object.__setattr__(planned.root, "cap_plan", spoof)
    object.__setattr__(planned.root, "cap_plan_sha256", v2._digest(
        v2.CAP_PLAN_DIGEST_DOMAIN, spoof.as_record()))
    object.__setattr__(planned.root, "record_sha256", v2._digest(
        v2.COMPLETE_SUPPORT_DIGEST_DOMAIN,
        v2._root_payload(planned.root, include_digest=False)))
    with pytest.raises(ValueError):
        v2.validate_complete_support_capture(planned)

    # Untouched builder-produced captures still validate.
    v2.validate_complete_support_capture(
        _assemble(rows, sections, entries, expected_row_count=3))


# ---------------------------------------------------------------------------
# DS-R4-V2-ROUTEB-DIAGNOSTIC-WIRING: frozen Route B profile and graph seam
# ---------------------------------------------------------------------------

def test_route_b_diagnostic_profile_is_the_accepted_plan_and_preflights_ok():
    plan = v2.ROUTE_B_DIAGNOSTIC_CAP_PLAN
    assert v2.ROUTE_B_DIAGNOSTIC_EXPECTED_ROW_COUNT == 1_928
    assert plan.as_record() == {
        "index_page_max_bytes": 262_144,
        "digest_page_max_bytes": 262_144,
        "sidecar_page_max_bytes": 237_568,
        "sidecar_entry_max_raw_bytes": 131_072,
        "sidecar_entry_max_bytes": 233_472,
        "sidecar_aggregate_max_bytes": 536_870_912,
        "root_max_bytes": 512_000,
        "universe_max_rows": 1_928}
    # The fixed-run profile is separate from, and does not alter, the shipped
    # placeholder defaults.
    assert plan != v2.CompleteSupportCapPlanV1()
    assert v2.CompleteSupportCapPlanV1().universe_max_rows == v2.UNIVERSE_MAX_ROWS
    preflight = v2.preflight_complete_support(plan)
    assert preflight.preflight_status == v2.PREFLIGHT_OK
    assert preflight.preflight_refusal_reason == ""
    assert preflight.cap_invariant_ok is True
    assert preflight.declared_universe_cap == 1_928
    assert preflight.measurement("sidecar_entry_worst_case_bytes") <= (
        plan.sidecar_entry_max_bytes)
    assert preflight.measurement("sidecar_page_worst_case_bytes") <= (
        plan.sidecar_page_max_bytes)
    assert preflight.measurement("index_page_worst_case_bytes") <= (
        plan.index_page_max_bytes)
    assert preflight.measurement("digest_page_worst_case_bytes") <= (
        plan.digest_page_max_bytes)
    assert preflight.measurement("sidecar_aggregate_worst_case_bytes") <= (
        plan.sidecar_aggregate_max_bytes)
    assert preflight.measurement("root_worst_case_bytes") <= plan.root_max_bytes


def _route_b_universe(count, present):
    """A synthetic Route B diagnostic universe over the shipped sink material.

    The first ``present`` admitted candidates are kept and carry real support
    payloads; every other admitted candidate is dropped with a typed row-local
    refusal. The returned ``matches`` tuple is the solver's returned post-filter
    tuple, in the same admitted order the collector sees.
    """
    candidates = tuple(
        _match("a", "b", left_plane=0, right_plane=0,
               score=0.9 - index * 1e-9)
        for index in range(count))
    dispositions = tuple(
        plane_graph.SAME_AXIS_DISPOSITION_KEPT if index < present
        else plane_graph.SAME_AXIS_DISPOSITION_DROPPED for index in range(count))
    rows = []
    for index, (candidate, disposition) in enumerate(zip(candidates, dispositions)):
        if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT:
            rows.append(_payload_row(candidate, index=index,
                                     disposition=disposition, post_index=index))
        else:
            rows.append(_refusal_row(candidate, index=index,
                                     disposition=disposition, post_index=None))
    matches = candidates[:present]
    bindings = (
        _binding("a", _pixels(211), _png_bytes((1, 2, 3)), signature_sha256=_HEX),
        _binding("b", _pixels(212), _png_bytes((4, 5, 6)),
                 signature_sha256=_HEX_B),
    )
    return candidates, matches, tuple(rows), dispositions, bindings


def _canonical_round_trip(capture):
    """Serialize, re-parse and re-validate one capture; returns its record."""
    encoded = v2._canonical_bytes(capture.to_record())
    assert encoded == v2._canonical_bytes(capture.to_record())
    record = json.loads(encoded.decode("utf-8"))
    assert record["schema"] == v2.COMPLETE_SUPPORT_CAPTURE_SCHEMA
    assert record["root"]["record_sha256"] == capture.root.record_sha256
    assert v2._canonical_bytes(record) == encoded
    v2.validate_complete_support_capture(capture)
    return record


def test_route_b_diagnostic_collector_captures_the_full_ordered_universe():
    candidates, matches, rows, dispositions, bindings = _route_b_universe(
        1_928, 2)
    reads = _readings_for(candidates)
    observation = _observation(candidates, dispositions)
    capture = v2.collect_route_b_diagnostic_support(
        reads, matches, bindings, rows, observation, ingress=(),
        failed_assignment_sha256=_ASSIGNMENT)
    assert capture.status == "complete", capture.root.reason
    assert capture.root.row_count == 1_928
    assert capture.root.declared_universe_cap == 1_928
    indexed = [row for page in capture.index_pages for row in page.rows]
    assert [row.admitted_row_index for row in indexed] == list(range(1_928))
    assert [row.disposition for row in indexed] == list(dispositions)
    for section in v2.DIGEST_SECTIONS:
        assert sum(page.digest_count for page in capture.digest_pages
                   if page.section == section) == 1_928
    present = [row for row in indexed if row.support_state == v2.SUPPORT_STATE_PRESENT]
    refused = [row for row in indexed if row.support_state == v2.SUPPORT_STATE_REFUSED]
    assert len(present) == 2 and len(refused) == 1_926
    assert all(row.support_payload_sha256 and row.support_refusal_reason is None
               for row in present)
    assert all(row.support_refusal_reason in v2.SUPPORT_REFUSAL_REASONS
               and row.support_payload_sha256 is None for row in refused)
    assert capture.sidecar_pages[0].entry_count == 2
    # A full ordered universe round-trips through canonical serialization.
    record = _canonical_round_trip(capture)
    assert record["root"]["whole_universe_identity_sha256"] == (
        v2.complete_support_universe_digest(indexed))

    # A caller-declared whole-universe identity that disagrees refuses atomically
    # instead of assembling under a foreign identity.
    wrong_identity = v2.collect_route_b_diagnostic_support(
        reads, matches, bindings, rows, observation, ingress=(),
        failed_assignment_sha256=_ASSIGNMENT,
        expected_whole_universe_identity_sha256="0" * 64)
    assert wrong_identity.status == "refused"
    assert wrong_identity.root.reason == "universe_identity_mismatch"
    assert wrong_identity.root.field == "whole_universe_identity_sha256"
    assert wrong_identity.row_count == 0 and wrong_identity.index_pages == ()
    assert wrong_identity.digest_pages == () and wrong_identity.sidecar_pages == ()
    v2.validate_complete_support_capture(wrong_identity)


def test_route_b_diagnostic_collector_refuses_short_and_over_cap_universes():
    # A short universe cannot satisfy the frozen 1,928-row expectation.
    candidates, matches, rows, dispositions, bindings = _route_b_universe(3, 1)
    short = v2.collect_route_b_diagnostic_support(
        _readings_for(candidates), matches, bindings, rows,
        _observation(candidates, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT)
    assert short.status == "refused"
    assert short.root.reason == "universe_count_mismatch"
    assert short.root.field == "row_count"
    assert short.row_count == 0 and short.index_pages == ()
    assert short.digest_pages == () and short.sidecar_pages == ()
    # A planned refusal preserves and hashes the plan that produced it.
    assert short.root.cap_plan is not None
    v2.validate_complete_support_capture(short)

    # The frozen plan's row ceiling refuses a 1,929-row universe atomically.
    candidates, matches, rows, dispositions, bindings = _route_b_universe(1_929, 1)
    over_cap = v2.collect_route_b_diagnostic_support(
        _readings_for(candidates), matches, bindings, rows,
        _observation(candidates, dispositions), ingress=(),
        failed_assignment_sha256=_ASSIGNMENT, expected_row_count=1_929)
    assert over_cap.status == "refused"
    assert over_cap.root.reason == "row_cap_exceeded"
    assert over_cap.root.field == "row_count"
    assert over_cap.row_count == 0 and over_cap.index_pages == ()
    v2.validate_complete_support_capture(over_cap)


def test_route_b_diagnostic_profile_cap_boundaries_refuse_atomically():
    candidates, matches, rows, dispositions, bindings = _route_b_universe(8, 2)
    reads = _readings_for(candidates)
    observation = _observation(candidates, dispositions)
    base = v2.ROUTE_B_DIAGNOSTIC_CAP_PLAN.as_record()
    cases = (
        ("index_page_over_cap", {"index_page_max_bytes": 10}),
        ("digest_page_over_cap", {"digest_page_max_bytes": 10}),
        ("sidecar_page_over_cap", {"sidecar_page_max_bytes": 10}),
        ("sidecar_aggregate_over_cap", {"sidecar_aggregate_max_bytes": 1}),
        ("root_over_cap", {"root_max_bytes": 100}),
    )
    for reason, overrides in cases:
        plan = v2.CompleteSupportCapPlanV1(**{**base, **overrides})
        capture = v2.collect_route_b_diagnostic_support(
            reads, matches, bindings, rows, observation, ingress=(),
            failed_assignment_sha256=_ASSIGNMENT, expected_row_count=8, plan=plan)
        assert capture.status == "refused", reason
        assert capture.root.reason == reason, (reason, capture.root.reason)
        assert capture.row_count == 0 and capture.index_pages == ()
        assert capture.digest_pages == () and capture.sidecar_pages == ()
        assert capture.root.cap_plan is not None
        v2.validate_complete_support_capture(capture)


def _graph_seam_material(count, present):
    candidates, matches, rows, dispositions, bindings = _route_b_universe(
        count, present)
    observation = _observation(candidates, dispositions)
    solution = plane_graph.ComponentSolution(
        frames=("a", "b"), quadrants={"a": 0, "b": 0},
        translations={"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)},
        axis_residuals={}, axis_resultants={}, scale_known=False,
        blocks_per_period={}, reason="plane_integer_residual", alternatives=())
    return matches, rows, dispositions, bindings, observation, solution


def _graph_seam_inputs():
    from test_image_derived_lattice_measure import (
        _build_session, _pose_graph, m9c)
    from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod

    patch = pytest.MonkeyPatch()
    # ``ingress=True`` binds the accepted builder's own per-frame ingress pins to
    # the session, so the seam can be shown to carry the session identity into
    # the capture rather than a side-channel pin.
    session = _build_session(patch, ingress=True)
    lattice = m9c.measure_image_derived_lattice(
        session=session, pose_graph=_pose_graph(session))
    signatures = signatures_mod.derive_image_derived_signatures(
        session=session, lattice_result=lattice)
    return patch, {"session": session, "lattice_result": lattice,
                   "signatures_result": signatures}


def _install_graph_seam_fakes(patch, matches, rows, observation, solution):
    def fake_match(_readings, _signatures, **kwargs):
        candidate_sink = kwargs.get("candidate_observation_sink")
        if candidate_sink is not None:
            candidate_sink.append(observation)
        witness_sink = kwargs.get("pre_filter_witness_sink")
        if witness_sink is not None:
            witness_sink.extend(rows)
        return matches

    patch.setattr(plane_graph, "match_plane_graph", fake_match)
    patch.setattr(plane_graph, "solve_component", lambda *args, **kwargs: solution)
    # The synthetic candidate identities do not exist in the real lattice
    # readings, so the two purely derived row values are fixed here; the seam
    # plumbing under test is the ingress/witness/identity binding, not these
    # replayable derived integers.
    patch.setattr(v2, "_alternative_index", lambda *args, **kwargs: 0)
    patch.setattr(v2, "_delta", lambda *args, **kwargs: 0.0)


def test_graph_complete_support_seam_collects_output_neutrally():
    patch, inputs = _graph_seam_inputs()
    try:
        matches, rows, dispositions, bindings, observation, solution = (
            _graph_seam_material(1_928, 2))
        _install_graph_seam_fakes(patch, matches, rows, observation, solution)
        plain = graph_mod._derive_image_derived_graph(**inputs)
        captured: list = []
        with_sink = graph_mod._derive_image_derived_graph(
            **inputs, complete_support_sink=captured.append,
            signature_witnesses=bindings)
        # Default-off parity: the diagnostic changes nothing about the returned
        # production result when it is attached.
        assert with_sink == plain
        assert len(captured) == 1
        capture = captured[0]
        assert isinstance(capture, v2.CompleteSupportCaptureV2)
        assert capture.status == "complete", capture.root.reason
        assert capture.root.row_count == 1_928
        assert capture.root.cap_plan.as_record() == (
            v2.ROUTE_B_DIAGNOSTIC_CAP_PLAN.as_record())
        indexed = [row for page in capture.index_pages for row in page.rows]
        assert [row.admitted_row_index for row in indexed] == list(range(1_928))
        assert [row.disposition for row in indexed] == list(dispositions)
        assert capture.sidecar_pages[0].entry_count == 2
        # The accepted session's own ingress pins are carried into the root.
        session_ingress = tuple(
            frame.ingress for frame in inputs["session"].frames
            if getattr(frame, "ingress", None) is not None)
        assert session_ingress
        assert capture.root.ingress == session_ingress
        _canonical_round_trip(capture)
    finally:
        patch.undo()


def test_graph_complete_support_seam_refuses_atomically_on_count_mismatch():
    patch, inputs = _graph_seam_inputs()
    try:
        matches, rows, dispositions, bindings, observation, solution = (
            _graph_seam_material(3, 1))
        _install_graph_seam_fakes(patch, matches, rows, observation, solution)
        captured: list = []
        result = graph_mod._derive_image_derived_graph(
            **inputs, complete_support_sink=captured.append,
            signature_witnesses=bindings)
        assert graph_mod._derive_image_derived_graph(**inputs) == result
        assert len(captured) == 1
        capture = captured[0]
        assert capture.status == "refused"
        assert capture.root.reason == "universe_count_mismatch"
        assert capture.root.field == "row_count"
        assert capture.row_count == 0 and capture.index_pages == ()
        assert capture.digest_pages == () and capture.sidecar_pages == ()
        assert capture.root.cap_plan is not None
        v2.validate_complete_support_capture(capture)
    finally:
        patch.undo()

