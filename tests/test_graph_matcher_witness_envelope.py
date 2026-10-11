from dataclasses import replace

import pytest

from scripts.probe_image_derived_shift_overlap import build_fixture
from mc_photogrammetry.reconstruction import lattice, plane_graph


def _run(*, budget=None, sink=None):
    left, right, _known = build_fixture("zero_overlap")
    sink = [] if sink is None else sink
    matches = plane_graph.match_plane_graph(
        (left.reading, right.reading), (left, right), max_shift_cells=32,
        witness_sink=sink, witness_budget=budget,
        signature_sha256_by_identity={id(left): "a" * 64, id(right): "b" * 64})
    return left, right, matches, sink


def test_final_match_envelopes_join_exactly_in_deterministic_order():
    _left, _right, matches, first = _run()
    assert len(matches) == len(first) > 0
    assert all(isinstance(row, plane_graph.GraphMatcherWitnessEnvelopeV1)
               for row in first)
    for match, row in zip(matches, first):
        assert (row.left, row.right, row.left_plane, row.right_plane,
                row.quadrant, row.shift_rc, row.overlap_cells) == (
                    match.left, match.right, match.left_plane, match.right_plane,
                    match.quadrant_of_right, match.cell_shift_rc,
                    match.overlap_cells)
    assert [row.envelope_sha256 for row in first] == [
        row.envelope_sha256 for row in _run()[3]]


def test_same_axis_filtered_candidates_publish_no_envelope(monkeypatch):
    original = plane_graph._apply_same_axis_filter
    monkeypatch.setattr(plane_graph, "_apply_same_axis_filter",
                        lambda matches, readings, refusals: [])
    _left, _right, matches, sink = _run()
    assert matches == () and sink == []
    monkeypatch.setattr(plane_graph, "_apply_same_axis_filter", original)


def test_envelope_tamper_and_final_cap_refusal_fail_closed():
    _left, right, matches, sink = _run()
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(sink[0], score=sink[0].score + 1.0)
    refused = []
    _left, _right, capped_matches, refused = _run(
        budget=lattice.WitnessBudgetV1(max_pair_entries=0), sink=refused)
    assert capped_matches == matches
    assert len(refused) == len(matches)
    assert all(isinstance(row, lattice.WitnessRefusalV1) for row in refused)
    # The witness covers the padded common canvas, not just the right grid.
    expected_entries = ((right.cells.shape[0] + 2 * 32)
                        * (right.cells.shape[1] + 2 * 32))
    assert all((row.stage, row.limit, row.planned, row.actual)
               == ("warp_entries", 0, expected_entries, 0) for row in refused)


def test_missing_signature_hash_binding_is_rejected():
    left, right, _known = build_fixture("zero_overlap")
    with pytest.raises(ValueError, match="does not bind final match"):
        plane_graph.match_plane_graph(
            (left.reading, right.reading), (left, right), witness_sink=[],
            signature_sha256_by_identity={})
