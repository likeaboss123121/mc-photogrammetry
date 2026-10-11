"""DS-IMG-CANDIDATE-SUPPORT-PROVENANCE-IMPL-R4 focused synthetic controls.

These controls cover the accepted R4 report's section 8 A-D seams and its
section 3 test matrix (a)-(v):

* **A** single-read ingress chain: one physical read per screenshot, hash +
  de-duplication + decode from the same sequential buffer with no aggregate
  byte cache, plus typed ingress refusal and default output/session parity;
* **B** full-universe pre-filter matcher-witness publication (kept and dropped
  rows each carrying their own witness or typed refusal);
* **C** the ordinary sampler's own ``SignatureSamplingWitnessV1`` bound to the
  ingress pin, usable mask, signature digest and sampler settings;
* **D** the strict bounded fail-closed candidate-support record and the pure
  Phase-B projected-ownership join.

Only synthetic, in-memory material is used: tiny locally generated PNG bytes,
synthetic ``uint8`` arrays, synthetic readings/signatures and synthetic
evaluator ownership grids. No real screenshot, truth, camera manifest, mask,
evaluator payload, litematic or JAR is opened, and no production CLI, evaluator
or solver run is made.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.input import media as media_mod
from mc_photogrammetry.input.image import ImageDecodeError, load_image_bytes
from mc_photogrammetry.pipeline import candidate_support_provenance as csp
from mc_photogrammetry.pipeline import frame_ingress as ingress_mod
from mc_photogrammetry.pipeline import image_derived_entry as entry
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.pipeline.contracts import FrameTransform
from mc_photogrammetry.reconstruction import lattice, lattice_planes, plane_graph
from test_image_derived_graph_same_axis_candidate_capture import _candidate_fixture

_HEX = "a" * 64
_HEX_B = "b" * 64
_MATCH_OVERLAP = 6


# ---------------------------------------------------------------------------
# Synthetic image / PNG helpers
# ---------------------------------------------------------------------------

def _png_bytes(colour, *, size=(12, 12)) -> bytes:
    from PIL import Image

    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = colour
    image[: height // 2, : width // 2] = tuple(value // 3 for value in colour)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _pixels(seed: int, *, shape=(32, 32, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _profile_reading(frame: str = "f", *, axis: int = 2):
    """The minimal synthetic ``PlaneReading`` accepted by ``plane_signature``."""
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=8.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=17,
        polygon=None, coverage_cells=4, structural_share=0.5, confidence=0.8,
        rectification=lattice_planes.PlaneRectification(
            tuple(tuple(float(value) for value in row) for row in np.eye(3)),
            12.0, (16.0, 16.0), (0.0, 0.0)))


def _make_ingress(frame_id: str, pixels: np.ndarray, data: bytes,
                  *, transform: FrameTransform | None = None
                  ) -> ingress_mod.FrameIngressProvenanceV1:
    from mc_photogrammetry.pipeline import image_derived_session as session_mod

    height, width = pixels.shape[:2]
    return ingress_mod.build_frame_ingress_provenance(
        frame_id=frame_id, source_label=f"{frame_id}.png", encoded_bytes=data,
        decoded_pixels=pixels,
        transform=transform or FrameTransform(source_size=(width, height)),
        decoded_pixel_sha256=session_mod.image_sha256(pixels))


def _signature_witness(pixels: np.ndarray, reading=None):
    """One real ``SignatureSamplingWitnessV1`` from the ordinary sampler call."""
    sink: list = []
    lattice_planes.plane_signature(
        pixels, reading or _profile_reading("f"), 1.0, witness_sink=sink)
    assert len(sink) == 1
    return sink[0]


def _border_rectification(pixels: np.ndarray, value: int
                          ) -> lattice.RectificationWitnessV1:
    """A real rectification witness whose first tap is forced to ``value``.

    The ordinary tap table already emits ``-1`` for every tap that falls outside
    the decoded source frame (``lattice.rectification_witness_v1``); this helper
    makes the sentinel reachable for a range check by writing exactly one tap.
    """
    source = np.asarray(pixels, dtype=np.float32)[..., 0]
    xs = np.full((1, 2), 1.0, dtype=np.float32)
    ys = np.full((1, 2), 1.0, dtype=np.float32)
    xs[0, 0] = -0.25
    witness = lattice.rectification_witness_v1(
        xs, ys, np.ones(xs.shape, dtype=bool), source_shape=source.shape)
    indices = np.asarray(witness.tap_source_indices, dtype=np.int64).copy()
    indices[0] = value
    indices.flags.writeable = False
    return replace(witness, tap_source_indices=indices, witness_sha256="")


def _binding(frame_id: str, pixels: np.ndarray, data: bytes,
             signature_sha256: str = _HEX,
             *, transform: FrameTransform | None = None
             ) -> csp.SourceTapWitnessBindingV1:
    ingress = _make_ingress(frame_id, pixels, data, transform=transform)
    witness = _signature_witness(pixels, _profile_reading(frame_id))
    return csp.SourceTapWitnessBindingV1(
        frame_id=frame_id, canonical_index=0, alternative_index=0,
        signature_sha256=signature_sha256, ingress=ingress,
        mask_sha256=_HEX_B, masks_sha256=_HEX,
        max_cells=lattice_planes.MAX_SIGNATURE_CELLS,
        max_rectified_side=lattice_planes.MAX_RECTIFIED_SIDE,
        pixels_per_cell=lattice_planes.SIGNATURE_PIXELS_PER_CELL,
        witness=witness)


# ---------------------------------------------------------------------------
# Synthetic candidate-support material
# ---------------------------------------------------------------------------

def _match(left, right, *, left_plane, right_plane, score, left_b=1.0,
           right_b=1.0, quadrant=0, left_axis=1, right_axis=1,
           shift=(0, 0)):
    return plane_graph.PlaneMatch(
        left=left, right=right, left_axis=left_axis, right_axis=right_axis,
        left_b=left_b, right_b=right_b, quadrant_of_right=quadrant, psr=5.0,
        peak=1.0, second_peak=0.1, overlap_cells=_MATCH_OVERLAP,
        integer_residual=None, score=score, left_plane=left_plane,
        right_plane=right_plane, cell_shift_rc=shift)


def _warp_witness(target=(6, 6), source_shape=(4, 4)):
    """A real, well-formed ``WarpSignatureWitnessV1`` over an explicit grid.

    The map is an integer translation whose taps stay inside ``source_shape`` for
    every target cell, so it is a genuine non-identity, asymmetric warp witness
    that the matcher-support payload can range-check.
    """
    rows, columns = target
    source_rows, source_columns = source_shape
    homography = np.eye(3, dtype=np.float64)
    homography.flags.writeable = False
    grid_rows, grid_columns = np.mgrid[0:rows, 0:columns]
    mapped_u = grid_columns.astype(np.float64)
    mapped_v = grid_rows.astype(np.float64)
    valid = ((mapped_v < source_rows) & (mapped_u < source_columns))
    source_row_index = np.clip(np.rint(mapped_v), 0, source_rows - 1).astype(np.int64)
    source_column_index = np.clip(
        np.rint(mapped_u), 0, source_columns - 1).astype(np.int64)
    for value in (mapped_u, mapped_v, valid, source_row_index, source_column_index):
        value.flags.writeable = False
    return lattice_planes.WarpSignatureWitnessV1(
        "mc-photogrammetry-warp-signature-witness-v1", homography,
        source_shape, target, mapped_u, mapped_v, valid, source_row_index,
        source_column_index)


def _pair_witness(seed: int = 5, *, target=(6, 6), shift=(0, 0),
                  source_shape=(4, 4)):
    """A real ``MatcherPairWitnessV1`` whose logical pairs conserve support.

    Only pairs with real (non-sentinel) indices are kept, so the witness is
    bounded and range-checkable just like a matcher witness whose coverage never
    touches a border.
    """
    warp = _warp_witness(target, source_shape)
    pairs = []
    for row in range(target[0]):
        for column in range(target[1]):
            if not bool(warp.geometric_valid[row, column]):
                continue
            pairs.append((int(warp.source_rows[row, column]),
                          int(warp.source_columns[row, column]), row, column))
            if len(pairs) == _MATCH_OVERLAP:
                break
        if len(pairs) == _MATCH_OVERLAP:
            break
    assert len(pairs) == _MATCH_OVERLAP, "fixture needs enough real pairs"
    pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 4)
    pair_array.flags.writeable = False
    homography = np.eye(3, dtype=np.float64)
    homography.flags.writeable = False
    return plane_graph.MatcherPairWitnessV1(
        "mc-photogrammetry-matcher-pair-witness-v1",
        plane_graph.PLANE_CORRELATION_ALGORITHM, 32, 0, homography,
        warp.witness_sha256, target, tuple(int(v) for v in shift), len(pairs),
        pair_array, warp_witness=warp)


def _sha_of(match: plane_graph.PlaneMatch) -> str:
    return hashlib.sha256(json.dumps(
        plane_graph._indexed_match_payload(match), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _payload_row(match: plane_graph.PlaneMatch, *, index: int, disposition: str,
                 post_index: int | None, target=(6, 6), shift=(0, 0),
                 source_shape=(4, 4)):
    witness = _pair_witness(target=target, shift=shift, source_shape=source_shape)
    envelope = plane_graph.GraphMatcherWitnessEnvelopeV1(
        "mc-photogrammetry-graph-matcher-witness-envelope-v1", _sha_of(match),
        _HEX, _HEX_B, match.left, match.right, match.left_plane,
        match.right_plane, match.left_axis, match.right_axis,
        match.quadrant_of_right, shift, witness.selected_support, 5.0,
        float(match.score), witness.witness_sha256)
    return plane_graph.PreFilterMatcherWitnessRowV1(
        admitted_row_index=index, disposition=disposition,
        post_match_index=post_index, match_sha256=_sha_of(match),
        envelope_sha256=envelope.envelope_sha256,
        left_signature_sha256=_HEX, right_signature_sha256=_HEX_B,
        payload=plane_graph.GraphMatcherWitnessPayloadV1(envelope, witness))


def _refusal_row(match, *, index, disposition, post_index,
                 reason="budget_exhausted"):
    refusal = lattice.WitnessRefusalV1(
        "refused", reason, "matcher_pair_entries", 0, 99, 0)
    return plane_graph.PreFilterMatcherWitnessRowV1(
        admitted_row_index=index, disposition=disposition,
        post_match_index=post_index, match_sha256=_sha_of(match),
        envelope_sha256=None, left_signature_sha256=None,
        right_signature_sha256=None, payload=None, refusal=refusal,
        refusal_reason=reason)


def _observation(candidates, dispositions):
    return plane_graph.SameAxisCandidateObservation(
        tuple(candidates), tuple(dispositions))


def _readings_for(matches):
    """Readings carrying every ``(frame, plane, b)`` the matches reference.

    Each plane in a frame gets a distinct canonical order key (period/phase), so
    the accepted ``_canonical_reading_index`` can order them without ambiguity.
    """
    wanted: dict[tuple, list] = {}
    for match in matches:
        wanted.setdefault((match.left, match.left_plane), []).append(match.left_b)
        wanted.setdefault((match.right, match.right_plane), []).append(match.right_b)
    frames: dict[str, list] = {}
    for (frame, plane), values in wanted.items():
        while len(frames.setdefault(frame, [])) <= plane:
            frames[frame].append(None)
        frames[frame][plane] = values
    readings: list = []
    for frame in sorted(frames):
        for plane, values in enumerate(frames[frame]):
            unique = sorted(set(values or [1.0]))
            alternatives = tuple(
                lattice_planes.DistanceAlternative(float(b), 1.0, 1.0)
                for b in unique)
            readings.append(lattice_planes.PlaneReading(
                frame=frame, axis=1, side=1.0, alternatives=alternatives,
                period_pixels=8.0 + 4.0 * plane,
                phase_pixels=(0.0, float(plane)), extent=1.0,
                samples=17, polygon=None, coverage_cells=4,
                structural_share=0.6, confidence=0.8,
                rectification=lattice_planes.PlaneRectification(
                    tuple(tuple(float(value) for value in row) for row in np.eye(3)),
                    12.0, (16.0, 16.0), (0.0, 0.0))))
    return readings


def _collect(matches, rows, dispositions, *, witnesses=(), readings=None,
             ingress=None):
    observations = _observation(matches, dispositions)
    if ingress is None:
        # The strict validator requires a pin for every frame a row names; a
        # caller that supplies witnesses already has their pins.
        ingress = tuple(item.ingress for item in witnesses
                        if isinstance(item, csp.SourceTapWitnessBindingV1))
    return csp.failed_candidate_support_provenance(
        readings if readings is not None else _readings_for(matches),
        tuple(matches), tuple(witnesses), tuple(rows), observations,
        ingress=tuple(ingress),
        failed_assignment_sha256=hashlib.sha256(b"assignment").hexdigest())


def _controlled_pair(left_signature, right_signature, *, quadrant,
                     max_shift_cells, refusals=None, diagnostic_sink=None,
                     witness_sink=None, **_kwargs):
    """A deterministic matcher stand-in that publishes a real pair witness.

    It mirrors ``test_image_derived_graph_same_axis_candidate_capture``'s
    controlled matcher but also honours ``witness_sink`` so the full-universe
    pre-filter publication can be exercised.
    """
    left = left_signature.reading
    right = right_signature.reading
    key = (left.frame, right.frame, float(left_signature.blocks_per_period),
           float(right_signature.blocks_per_period), int(quadrant))
    score = (0.9 if key in {
        ("a", "b", 1.0, 1.0, 0), ("a", "b", 0.5, 1.0, 0),
        ("a", "c", 1.0, 1.0, 0), ("b", "c", 1.0, 1.0, 0)} else None)
    admitted = score is not None
    if diagnostic_sink is not None:
        diagnostic_sink.pairs.append(plane_graph.PlanePairTrace(
            left.frame, right.frame, left.axis, right.axis, int(quadrant),
            admitted, "" if admitted else "no_signal",
            None, None, None, None, None, None,
            psr=5.0 if admitted else None, score=score,
            overlap_cells=_MATCH_OVERLAP if admitted else None,
            cell_shift_rc=(0, 0) if admitted else None))
    if not admitted:
        return None
    witness = _pair_witness()
    if witness_sink is not None:
        witness_sink.append(witness)
    return plane_graph.PlaneMatch(
        left=left.frame, right=right.frame, left_axis=left.axis,
        right_axis=right.axis,
        left_b=float(left_signature.blocks_per_period),
        right_b=float(right_signature.blocks_per_period),
        quadrant_of_right=int(quadrant), psr=5.0, peak=1.0, second_peak=0.1,
        overlap_cells=_MATCH_OVERLAP, integer_residual=None, score=score,
        cell_shift_rc=(0, 0))


def _real_matcher_pass(monkeypatch):
    readings, signatures = _candidate_fixture()
    monkeypatch.setattr(plane_graph, "match_plane_pair", _controlled_pair)
    observations: list = []
    rows: list = []
    hashes = {id(signature): f"{index + 1:064x}"
              for index, signature in enumerate(signatures)}
    matches = plane_graph.match_plane_graph(
        readings, signatures, candidate_observation_sink=observations,
        pre_filter_witness_sink=rows, signature_sha256_by_identity=hashes)
    return readings, signatures, matches, observations[0], rows


# ---------------------------------------------------------------------------
# (a) output neutrality of the capture seam
# ---------------------------------------------------------------------------

def test_default_matcher_and_signature_paths_are_output_neutral(monkeypatch):
    """(a) No sink allocates nothing and changes no returned digest."""
    readings, signatures = _candidate_fixture()
    monkeypatch.setattr(plane_graph, "match_plane_pair", _controlled_pair)
    control = plane_graph.match_plane_graph(readings, signatures)
    _readings, _signatures, captured, _observation_, _rows = _real_matcher_pass(
        monkeypatch)
    # The pre-filter publication changes neither the returned match tuple nor the
    # default allocation path.
    assert captured == control
    # A no-sink collector call over the same material still refuses atomically
    # rather than fabricating a partial record.
    refused = csp.failed_candidate_support_provenance(
        readings, captured, (), (), None)
    assert refused.status == "refused" and refused.row_count == 0
    # The witness sink changes no signature array on the ordinary signature path.
    pixels = _pixels(40)
    reading = _profile_reading("f")
    plain = lattice_planes.plane_signature(pixels, reading, 1.0)
    sink: list = []
    observed = lattice_planes.plane_signature(
        pixels, reading, 1.0, witness_sink=sink)
    assert len(sink) == 1
    assert np.array_equal(observed.cells, plain.cells)
    assert np.array_equal(observed.coverage, plain.coverage)


# ---------------------------------------------------------------------------
# (b) pre-filter universe and dropped-candidate witness publication
# ---------------------------------------------------------------------------

def test_pre_filter_universe_covers_kept_and_dropped_with_own_witnesses(monkeypatch):
    readings, signatures, matches, observation, rows = _real_matcher_pass(monkeypatch)
    assert len(observation.candidates) == len(rows) == 4
    assert all(row.admitted_row_index == index for index, row in enumerate(rows))
    assert tuple(row.disposition for row in rows) == observation.dispositions
    dropped = [row for row in rows if row.disposition ==
               plane_graph.SAME_AXIS_DISPOSITION_DROPPED]
    kept = [row for row in rows if row.disposition ==
            plane_graph.SAME_AXIS_DISPOSITION_KEPT]
    assert dropped and kept and len(dropped) + len(kept) == 4
    for row in rows:
        assert row.payload is not None or row.refusal_reason is not None
    assert all(row.post_match_index is None for row in dropped)
    envelopes = [row.envelope_sha256 for row in kept if row.envelope_sha256]
    assert len(set(envelopes)) == len(envelopes)


def test_dropped_candidate_carries_its_own_refusal_not_the_survivor_witness():
    kept = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    dropped = _match("a", "b", left_plane=1, right_plane=1, score=0.5)
    rows = (
        _payload_row(kept, index=0, disposition="kept", post_index=0),
        _refusal_row(dropped, index=1, disposition="same_axis_mismatch",
                     post_index=None),
    )
    witnesses = (
        _binding("a", _pixels(1), _png_bytes((1, 2, 3)), signature_sha256=_HEX),
        _binding("b", _pixels(2), _png_bytes((4, 5, 6)),
                 signature_sha256=_HEX_B))
    record = _collect(
        (kept, dropped), rows, ("kept", "same_axis_mismatch"), witnesses=witnesses)
    assert record.status == "assembled", record.reason
    refusal_row = record.rows[1]
    assert refusal_row.admitted_row_index == 1
    assert refusal_row.match_index is None
    assert refusal_row.matcher_refusal_reason == "budget_exhausted"
    assert refusal_row.support is None
    assert refusal_row.support_refusal_reason == "budget_exhausted"
    assert refusal_row.matcher_witness_sha256 is None


# ---------------------------------------------------------------------------
# (c) duplicate 9-scalar identity control
# ---------------------------------------------------------------------------

def test_duplicate_nine_scalar_identity_rows_resolve_by_exact_keys():
    first = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    second = _match("a", "b", left_plane=0, right_plane=0, score=0.7)
    # Same nine scalars, different score and therefore different match digest.
    assert (first.left, first.left_plane, first.left_axis, first.right,
            first.right_plane, first.right_axis, first.quadrant_of_right,
            first.left_b, first.right_b) == (
        second.left, second.left_plane, second.left_axis, second.right,
        second.right_plane, second.right_axis, second.quadrant_of_right,
        second.left_b, second.right_b)
    rows = (
        _payload_row(first, index=0, disposition="kept", post_index=0),
        _refusal_row(second, index=1, disposition="same_axis_mismatch",
                     post_index=None),
    )
    witness = _binding("a", _pixels(3), _png_bytes((7, 8, 9)))
    other = _binding("b", _pixels(4), _png_bytes((10, 11, 12)),
                     signature_sha256=_HEX_B)
    record = _collect((first, second), rows, ("kept", "same_axis_mismatch"),
                      witnesses=(witness, other))
    assert record.status == "assembled", record.reason
    assert len(record.rows) == 2
    assert record.rows[0].match_sha256 != record.rows[1].match_sha256
    assert record.rows[0].admitted_row_index == 0
    assert record.rows[1].admitted_row_index == 1
    assert record.rows[0].input_match_identity == record.rows[1].input_match_identity


# ---------------------------------------------------------------------------
# (d) one-to-one join and atomic refusal
# ---------------------------------------------------------------------------

def test_duplicate_join_key_refuses_atomically():
    match = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    payload_row = _payload_row(match, index=0, disposition="kept", post_index=0)
    witness = _binding("a", _pixels(5), _png_bytes((1, 1, 1)))
    other = _binding("b", _pixels(6), _png_bytes((2, 2, 2)),
                     signature_sha256=_HEX_B)
    record = _collect((match, match), (payload_row, payload_row),
                      ("kept", "kept"), witnesses=(witness, other))
    assert record.status == "refused"
    assert record.row_count == 0 and record.rows == ()
    assert record.reason


# ---------------------------------------------------------------------------
# (e)-(f) identity and signature join
# ---------------------------------------------------------------------------

def test_rows_bind_identity_and_endpoint_signature_digests():
    left = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    rows = (_payload_row(left, index=0, disposition="kept", post_index=0),)
    left_binding = _binding("a", _pixels(7), _png_bytes((3, 3, 3)))
    right_binding = _binding("b", _pixels(8), _png_bytes((4, 4, 4)),
                             signature_sha256=_HEX_B)
    record = _collect((left,), rows, ("kept",),
                      witnesses=(left_binding, right_binding))
    assert record.status == "assembled", record.reason
    row = record.rows[0]
    assert row.left_signature_sha256 == _HEX
    assert row.right_signature_sha256 == _HEX_B
    assert left_binding.signature_sha256 == row.left_signature_sha256
    assert right_binding.signature_sha256 == row.right_signature_sha256
    assert row.input_match_identity == (
        left.left, left.left_plane, left.left_axis, left.right,
        left.right_plane, left.right_axis, left.quadrant_of_right,
        float(left.left_b), float(left.right_b))
    assert row.input_match_identity_sha256 == csp._identity_digest(
        [list(row.input_match_identity)])
    assert row.left_alternative_index == 0 and row.right_alternative_index == 0


# ---------------------------------------------------------------------------
# (g)-(h) orientation, shapes and support conservation
# ---------------------------------------------------------------------------

def test_support_conservation_and_left_source_right_target_orientation():
    match = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    row = _payload_row(match, index=0, disposition="kept", post_index=0,
                       target=(7, 5))
    assert row.payload is not None
    witness = row.payload.witness
    envelope = row.payload.envelope
    assert len(witness.logical_pairs) == match.overlap_cells == _MATCH_OVERLAP
    assert len(witness.logical_pairs) == envelope.overlap_cells
    warp = witness.warp_witness
    assert warp is not None
    assert warp.source_shape != warp.target_shape
    assert warp.target_shape == (7, 5)
    assert witness.selected_support == len(witness.logical_pairs)
    pairs = witness.logical_pairs
    assert pairs[:, 0].max() < warp.source_shape[0]
    assert pairs[:, 1].max() < warp.source_shape[1]
    assert pairs[:, 2].max() < warp.target_shape[0]
    assert pairs[:, 3].max() < warp.target_shape[1]


def test_endpoint_swap_changes_recorded_orientation_and_shapes():
    forward = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    swapped = _match("b", "a", left_plane=0, right_plane=0, score=0.9)
    row_a = _payload_row(forward, index=0, disposition="kept", post_index=0,
                         target=(7, 5))
    row_b = _payload_row(swapped, index=0, disposition="kept", post_index=0,
                         target=(5, 7))
    assert (row_a.payload.witness.warp_witness.source_shape,
            row_a.payload.witness.warp_witness.target_shape) != (
        row_b.payload.witness.warp_witness.source_shape,
        row_b.payload.witness.warp_witness.target_shape)
    assert row_a.envelope_sha256 != row_b.envelope_sha256
    assert row_a.payload.envelope.left == "a"
    assert row_b.payload.envelope.left == "b"


# ---------------------------------------------------------------------------
# (i)-(j) retrievable bounded payload, over-cap and tamper refusal
# ---------------------------------------------------------------------------

def test_support_payload_is_retrievable_and_digests_recompute():
    match = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    row = _payload_row(match, index=0, disposition="kept", post_index=0)
    payload = row.payload.witness
    rebuilt = csp._support_payload(row)
    assert rebuilt is not None
    assert rebuilt.logical_pairs.shape == payload.logical_pairs.shape
    assert np.array_equal(rebuilt.warp_source_rows,
                          payload.warp_witness.source_rows)
    assert np.array_equal(rebuilt.warp_source_columns,
                          payload.warp_witness.source_columns)
    assert rebuilt.payload_bytes() <= csp.CANDIDATE_SUPPORT_MAX_SUPPORT_BYTES
    again = csp.MatcherSupportPayloadV1(
        logical_pairs=payload.logical_pairs,
        warp_source_rows=payload.warp_witness.source_rows,
        warp_source_columns=payload.warp_witness.source_columns,
        source_shape=payload.warp_witness.source_shape,
        target_shape=payload.warp_witness.target_shape)
    assert again.payload_sha256 == rebuilt.payload_sha256


def test_over_cap_support_refuses_atomically():
    match = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    rows = (_payload_row(match, index=0, disposition="kept", post_index=0),)
    left_binding = _binding("a", _pixels(9), _png_bytes((5, 5, 5)))
    right_binding = _binding("b", _pixels(10), _png_bytes((6, 6, 6)),
                             signature_sha256=_HEX_B)
    record = _collect((match,), rows, ("kept",),
                      witnesses=(left_binding, right_binding))
    assert record.status == "assembled", record.reason
    refused = csp.failed_candidate_support_provenance(
        _readings_for((match,)), (match,), (left_binding, right_binding),
        rows, _observation((match,), ("kept",)),
        failed_assignment_sha256=hashlib.sha256(b"x").hexdigest(),
        max_support_bytes=1)
    assert refused.status == "refused"
    assert refused.row_count == 0 and refused.rows == ()
    assert refused.reason == csp.SUPPORT_PAYLOAD_OVER_CAP


def test_tampered_record_refuses_atomically():
    match = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    rows = (_payload_row(match, index=0, disposition="kept", post_index=0),)
    left_binding = _binding("a", _pixels(11), _png_bytes((7, 7, 7)))
    right_binding = _binding("b", _pixels(12), _png_bytes((8, 8, 8)),
                             signature_sha256=_HEX_B)
    record = _collect((match,), rows, ("kept",),
                      witnesses=(left_binding, right_binding))
    assert record.status == "assembled", record.reason
    tampered = replace(record.rows[0])
    object.__setattr__(tampered, "match_index", 41)
    with pytest.raises(ValueError):
        replace(record, rows=(tampered,))
    stale = replace(record)
    object.__setattr__(stale, "record_sha256", "f" * 64)
    with pytest.raises(ValueError):
        csp.validate_candidate_support_provenance(stale)


# ---------------------------------------------------------------------------
# (k) shared identity digest with the sibling observers
# ---------------------------------------------------------------------------

def test_identity_digest_matches_the_shared_observer_domain():
    kept = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    rows = (_payload_row(kept, index=0, disposition="kept", post_index=0),)
    witness = _binding("a", _pixels(13), _png_bytes((9, 9, 9)))
    other = _binding("b", _pixels(14), _png_bytes((10, 10, 10)),
                     signature_sha256=_HEX_B)
    record = _collect((kept,), rows, ("kept",), witnesses=(witness, other))
    assert record.status == "assembled", record.reason
    assert csp.INPUT_MATCH_IDENTITY_DOMAIN == "input_match_identity"
    expected = hashlib.sha256(json.dumps(
        {"domain": csp.INPUT_MATCH_IDENTITY_DOMAIN,
         "schema": csp.INPUT_MATCH_IDENTITY_SCHEMA,
         "payload": [list(record.rows[0].input_match_identity)]},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert record.input_match_identity_sha256 == expected


# ---------------------------------------------------------------------------
# (l)-(n) source-tap witness capture, binding and sampler settings
# ---------------------------------------------------------------------------

def test_source_tap_witness_matches_consumed_shape_and_valid_indices():
    pixels = _pixels(20)
    data = _png_bytes((1, 2, 3))
    binding = _binding("f", pixels, data)
    witness = binding.witness
    assert witness.rgb_rectification.source_shape == pixels.shape[:2]
    flat = pixels.shape[0] * pixels.shape[1]
    assert np.all(witness.cell_source_indices >= -1)
    assert np.all(witness.cell_source_indices < flat)
    for rectification in (witness.rgb_rectification,
                          witness.coverage_rectification,
                          witness.region_rectification):
        if rectification is not None:
            assert rectification.source_shape == pixels.shape[:2]
            assert np.all(rectification.tap_source_indices < flat)
    assert binding.witness_sha256 == witness.witness_sha256
    assert binding.ingress.decoded_pixel_sha256 == binding.ingress.decoded_pixel_sha256


def test_source_tap_witness_allows_exactly_the_minus_one_border_sentinel():
    """``-1`` is the documented border sentinel and must bind in every rectification."""
    pixels = _pixels(27)
    data = _png_bytes((4, 5, 6))
    ingress = _make_ingress("f", pixels, data)
    witness = _signature_witness(pixels, _profile_reading("f"))
    sentinel = _border_rectification(pixels, -1)
    assert -1 in sentinel.tap_source_indices
    witness = replace(
        witness, rgb_rectification=sentinel, coverage_rectification=sentinel,
        region_rectification=sentinel, witness_sha256="")
    # The forced tap lives inside the decoded frame's area, so the untouched
    # upper bound cannot be what accepts it.
    assert sentinel.tap_source_indices.max() < pixels.shape[0] * pixels.shape[1]
    binding = csp.SourceTapWitnessBindingV1(
        frame_id="f", canonical_index=0, alternative_index=0,
        signature_sha256=_HEX, ingress=ingress, mask_sha256=_HEX_B,
        masks_sha256=_HEX, max_cells=1, max_rectified_side=1,
        pixels_per_cell=1, witness=witness)
    assert binding.witness is witness
    assert -1 in binding.witness.rgb_rectification.tap_source_indices
    assert -1 in binding.witness.coverage_rectification.tap_source_indices
    assert -1 in binding.witness.region_rectification.tap_source_indices
    assert binding.witness_sha256 == witness.witness_sha256


@pytest.mark.parametrize("field", ("rgb", "coverage", "region"))
def test_source_tap_witness_rejects_below_sentinel_or_at_source_area(field):
    """Anything below ``-1`` or at/above ``rows * columns`` refuses typed."""
    pixels = _pixels(28)
    data = _png_bytes((7, 8, 9))
    ingress = _make_ingress("f", pixels, data)
    base = _signature_witness(pixels, _profile_reading("f"))
    flat = pixels.shape[0] * pixels.shape[1]
    for value in (-2, flat, flat + 1):
        witness = replace(
            base, region_rectification=base.rgb_rectification,
            witness_sha256="")
        witness = replace(
            witness, **{field + "_rectification": _border_rectification(pixels, value)},
            witness_sha256="")
        with pytest.raises(csp.CandidateSupportError) as error:
            csp.SourceTapWitnessBindingV1(
                frame_id="f", canonical_index=0, alternative_index=0,
                signature_sha256=_HEX, ingress=ingress, mask_sha256=_HEX_B,
                masks_sha256=_HEX, max_cells=1, max_rectified_side=1,
                pixels_per_cell=1, witness=witness)
        assert error.value.reason == csp.SOURCE_TAP_WITNESS_INDEX_OUT_OF_RANGE
        assert error.value.field == "tap_source_indices"
    # The untouched original witness still binds, so only the forced tap decides.
    csp.SourceTapWitnessBindingV1(
        frame_id="f", canonical_index=0, alternative_index=0,
        signature_sha256=_HEX, ingress=ingress, mask_sha256=_HEX_B,
        masks_sha256=_HEX, max_cells=1, max_rectified_side=1,
        pixels_per_cell=1, witness=base)


def test_shape_mismatch_and_transform_disagreement_refuse_typed():
    pixels = _pixels(21)
    data = _png_bytes((2, 3, 4))
    # A pin whose consumed shape disagrees with the witness shape refuses.
    small = _pixels(99, shape=(16, 16, 3))
    ingress = _make_ingress("f", small, data)
    witness = _signature_witness(pixels, _profile_reading("f"))
    assert witness.rgb_rectification.source_shape != ingress.source_shape
    with pytest.raises(csp.CandidateSupportError) as error:
        csp.SourceTapWitnessBindingV1(
            frame_id="f", canonical_index=0, alternative_index=0,
            signature_sha256=_HEX, ingress=ingress, mask_sha256=_HEX_B,
            masks_sha256=_HEX, max_cells=1, max_rectified_side=1,
            pixels_per_cell=1, witness=witness)
    assert error.value.reason == csp.SOURCE_TAP_WITNESS_SHAPE_MISMATCH
    # A transform whose output size disagrees with the decoded shape refuses as a
    # frame-transform problem, never silently defaulting to identity.
    with pytest.raises(ingress_mod.IngressProvenanceError) as error:
        _make_ingress("f", pixels, data,
                      transform=FrameTransform(source_size=(999, 999)))
    assert error.value.reason == ingress_mod.FRAME_TRANSFORM_UNAVAILABLE


def test_budget_exhausted_sampler_refuses_never_empty_witness():
    pixels = _pixels(23)
    sink: list = []
    lattice_planes.plane_signature(
        pixels, _profile_reading("f"), 1.0, witness_sink=sink,
        witness_budget=lattice.WitnessBudgetV1(0, 0, 0, 0))
    assert len(sink) == 1
    assert isinstance(sink[0], lattice.WitnessRefusalV1)
    binding = signatures_mod._source_tap_witness_binding(
        evidence=SimpleNamespace(frame_id="f", canonical_index=0,
                                 alternative_index=0, signature_sha256=_HEX),
        witnesses=sink, ingress=_make_ingress("f", pixels, _png_bytes((3, 3, 3))),
        mask_sha256=_HEX_B, masks_sha256=_HEX, max_cells=1,
        max_rectified_side=1, pixels_per_cell=1)
    assert isinstance(binding, csp.SourceTapWitnessRefusalV1)
    assert binding.reason == csp.SOURCE_TAP_WITNESS_BUDGET_EXHAUSTED
    missing = signatures_mod._source_tap_witness_binding(
        evidence=SimpleNamespace(frame_id="f", canonical_index=0,
                                 alternative_index=0, signature_sha256=_HEX),
        witnesses=[], ingress=_make_ingress("f", pixels, _png_bytes((4, 4, 4))),
        mask_sha256=_HEX_B, masks_sha256=_HEX, max_cells=1,
        max_rectified_side=1, pixels_per_cell=1)
    assert missing.reason == csp.SOURCE_TAP_WITNESS_UNAVAILABLE
    no_ingress = signatures_mod._source_tap_witness_binding(
        evidence=SimpleNamespace(frame_id="f", canonical_index=0,
                                 alternative_index=0, signature_sha256=_HEX),
        witnesses=[_signature_witness(pixels, _profile_reading("f"))],
        ingress=None, mask_sha256=_HEX_B, masks_sha256=_HEX, max_cells=1,
        max_rectified_side=1, pixels_per_cell=1)
    assert no_ingress.reason == ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE


# ---------------------------------------------------------------------------
# (o) no evaluator material in Phase A
# ---------------------------------------------------------------------------

def test_phase_a_record_carries_no_evaluator_or_truth_material():
    pixels = _pixels(24)
    data = _png_bytes((5, 6, 7))
    binding = _binding("f", pixels, data)
    record = binding.to_record()
    ingress = binding.ingress.to_record()
    forbidden = ("truth", "physical_plane_key", "owner", "ownership",
                 "evaluator", "camera_manifest", "renderer")
    for payload in (record, ingress):
        text = json.dumps(payload, sort_keys=True).casefold()
        assert not any(token in text for token in forbidden)
    assert binding.signature_sha256 == _HEX
    # The Phase A provenance module imports no evaluator/truth module.
    source = Path(csp.__file__).read_text(encoding="utf-8").casefold()
    for token in ("renderedtruthevidence", "rendererownershipwitness",
                  "evaluation_frozen_planes", "physical_plane_key"):
        assert token not in source


# ---------------------------------------------------------------------------
# (p)-(q) Phase B projects ownership; alignment abstentions
# ---------------------------------------------------------------------------

def _projected_ownership(rows, columns, *, camera=_HEX, convention="z-buffer-v1"):
    ownership = np.full((rows, columns), -1, dtype=np.int64)
    ownership[2:4, 2:5] = 0
    ownership[2:4, 5:8] = 1
    usable = np.ones((rows, columns), dtype=bool)
    return csp.ProjectedEvaluatorOwnershipV1(
        ownership=ownership, usable=usable, frame_size=(columns, rows),
        camera_sha256=camera, transform=FrameTransform(source_size=(columns, rows)),
        occlusion_convention=convention,
        owner_plane_keys={-1: None, 0: ("x", 1, 1, 1), 1: ("z", -1, 1, 1)})


def test_phase_b_projects_ownership_from_saved_indices():
    pixels = _pixels(25)
    data = _png_bytes((8, 9, 10))
    binding = _binding("f", pixels, data)
    rows, columns = pixels.shape[:2]
    evaluator = _projected_ownership(rows, columns)
    projection = csp.project_ownership_into_source_space(
        binding=binding, evaluator=evaluator,
        camera_sha256=evaluator.camera_sha256,
        inference_transform=evaluator.transform,
        occlusion_convention=evaluator.occlusion_convention)
    assert isinstance(projection, csp.OwnershipProjectionV1)
    assert set(projection.cell_states) <= {"pure", "mixed", "unknown"}
    witness = binding.witness
    flat = evaluator.ownership.reshape(-1)
    for cell, state in enumerate(projection.cell_states):
        indices = witness.cell_source_indices[
            witness.cell_source_ptr[cell]:witness.cell_source_ptr[cell + 1]]
        observed = {int(flat[int(index)]) for index in indices if int(index) >= 0}
        if state == "pure":
            assert len(observed) == 1


def test_phase_b_marks_the_minus_one_sentinel_unknown_without_indexing_it():
    """A saved ``-1`` source tap is unknown; the join never reads ``flat[-1]``."""
    pixels = _pixels(29)
    data = _png_bytes((13, 14, 15))
    ingress = _make_ingress("f", pixels, data)
    witness = _signature_witness(pixels, _profile_reading("f"))
    sentinel = _border_rectification(pixels, -1)
    # Cell 0's own dependency list carries the saved border sentinel. The last
    # flat position (position 1023 in the 32x32 frame) owns plane 1, so an
    # accidental ``flat[-1]`` read would resolve cell 0 to ``("z", -1, 1, 1)``
    # instead of the typed ``out_of_range_indices`` unknown.
    members = np.asarray(witness.cell_source_indices, dtype=np.int64).copy()
    assert members[witness.cell_source_ptr[0]] != -1
    members[witness.cell_source_ptr[0]] = -1
    members.flags.writeable = False
    rows, columns = pixels.shape[:2]
    ownership = np.full((rows, columns), -1, dtype=np.int64)
    ownership[-1, -1] = 1
    assert int(ownership.reshape(-1)[-1]) == 1
    evaluator = csp.ProjectedEvaluatorOwnershipV1(
        ownership=ownership, usable=np.ones((rows, columns), dtype=bool),
        frame_size=(columns, rows), camera_sha256=_HEX,
        transform=FrameTransform(source_size=(columns, rows)),
        occlusion_convention="z-buffer-v1",
        owner_plane_keys={-1: None, 0: ("x", 1, 1, 1), 1: ("z", -1, 1, 1)})
    witness = replace(
        witness, rgb_rectification=sentinel, coverage_rectification=sentinel,
        region_rectification=sentinel,
        cell_source_indices=members, witness_sha256="")
    binding = csp.SourceTapWitnessBindingV1(
        frame_id="f", canonical_index=0, alternative_index=0,
        signature_sha256=_HEX, ingress=ingress, mask_sha256=_HEX_B,
        masks_sha256=_HEX, max_cells=1, max_rectified_side=1,
        pixels_per_cell=1, witness=witness)
    projection = csp.project_ownership_into_source_space(
        binding=binding, evaluator=evaluator, camera_sha256=evaluator.camera_sha256,
        inference_transform=evaluator.transform,
        occlusion_convention=evaluator.occlusion_convention)
    assert isinstance(projection, csp.OwnershipProjectionV1)
    # Only cell 0 depends on the saved ``-1`` membership, and the join named the
    # out-of-range cause instead of resolving ``flat[-1]``.
    sentinel_cells = [
        cell for cell in range(len(witness.cell_rectangles))
        if -1 in witness.cell_source_indices[
            witness.cell_source_ptr[cell]:witness.cell_source_ptr[cell + 1]]]
    assert sentinel_cells == [0]
    assert projection.cell_states[0] == "unknown"
    assert projection.cell_plane_keys[0] is None
    assert dict(projection.reason_counts)["out_of_range_indices"] == 1
    # The synthetic plane 1 stored at flat position -1 must never be resolved by
    # the sentinel cell or anywhere else in the projection.
    assert ("z", -1, 1, 1) not in projection.cell_plane_keys
    assert len(projection.cell_states) == len(witness.cell_rectangles)


def test_phase_b_alignment_mismatches_return_typed_abstentions():
    pixels = _pixels(26)
    binding = _binding("f", pixels, _png_bytes((11, 12, 13)))
    rows, columns = pixels.shape[:2]
    evaluator = _projected_ownership(rows, columns)
    wrong_camera = csp.project_ownership_into_source_space(
        binding=binding, evaluator=evaluator, camera_sha256=_HEX_B,
        inference_transform=evaluator.transform,
        occlusion_convention=evaluator.occlusion_convention)
    assert wrong_camera.reason == "evaluator_camera_misalignment"
    wrong_transform = csp.project_ownership_into_source_space(
        binding=binding, evaluator=evaluator, camera_sha256=evaluator.camera_sha256,
        inference_transform=FrameTransform(source_size=(999, 999)),
        occlusion_convention=evaluator.occlusion_convention)
    assert wrong_transform.reason == "transform_mismatch"
    wrong_occlusion = csp.project_ownership_into_source_space(
        binding=binding, evaluator=evaluator, camera_sha256=evaluator.camera_sha256,
        inference_transform=evaluator.transform, occlusion_convention="other-v1")
    assert wrong_occlusion.reason == "occlusion_unavailable"
    wrong_size = _projected_ownership(rows, columns)
    object.__setattr__(wrong_size, "frame_size", (columns + 1, rows))
    oversized = csp.project_ownership_into_source_space(
        binding=binding, evaluator=wrong_size, camera_sha256=evaluator.camera_sha256,
        inference_transform=evaluator.transform,
        occlusion_convention=evaluator.occlusion_convention)
    assert oversized.reason == "dimension_mismatch"


# ---------------------------------------------------------------------------
# (r)-(s) publication and row/disposition identity through the real matcher
# ---------------------------------------------------------------------------

def test_real_matcher_publishes_a_row_for_every_admitted_candidate(monkeypatch):
    readings, signatures, matches, observation, rows = _real_matcher_pass(monkeypatch)
    assert len(rows) == len(observation.candidates)
    for position, (candidate, disposition) in enumerate(zip(
            observation.candidates, observation.dispositions)):
        row = rows[position]
        assert row.admitted_row_index == position
        assert row.disposition == disposition
        if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT:
            assert row.post_match_index is not None
            assert matches[row.post_match_index] is candidate
        else:
            assert row.post_match_index is None


def test_real_matcher_dropped_row_never_aliases_a_survivor(monkeypatch):
    _readings, _signatures, matches, observation, rows = _real_matcher_pass(
        monkeypatch)
    dropped = [row for row in rows
               if row.disposition == plane_graph.SAME_AXIS_DISPOSITION_DROPPED]
    kept = [row for row in rows
            if row.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT]
    assert dropped and kept and len(dropped) + len(kept) == 4
    # Each row names its own admitted match: the dropped rows' match digests are
    # distinct from the survivors' and their post-filter index is a typed absence.
    survivor_hashes = {row.match_sha256 for row in kept}
    for row in dropped:
        assert row.post_match_index is None
        assert row.match_sha256 not in survivor_hashes
    assert len({row.match_sha256 for row in rows}) == len(rows)


def test_dropped_payload_row_keeps_its_own_envelope_and_witness():
    """A dropped row's own minted payload is never a survivor's object."""
    kept = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    dropped = _match("a", "b", left_plane=1, right_plane=1, score=0.5)
    kept_row = _payload_row(kept, index=0, disposition="kept", post_index=0)
    dropped_row = _payload_row(
        dropped, index=1, disposition="same_axis_mismatch", post_index=None,
        target=(5, 5))
    assert dropped_row.payload is not None
    assert dropped_row.payload.envelope.envelope_sha256 != (
        kept_row.payload.envelope.envelope_sha256)
    assert dropped_row.payload.envelope.match_sha256 == dropped_row.match_sha256
    assert dropped_row.match_sha256 != kept_row.match_sha256


def test_pre_filter_row_unpacks_nested_refusal_without_aliasing_the_survivor():
    """A refused dropped row carries its own refusal, never a survivor's witness.

    Regression for the nested ``(witness, left, right)`` unpack: the refusal is
    inside the stored triple, so a check on the container would miss it and then
    read the survivor witness fields (or crash). The dropped row must be a typed
    refusal with no envelope and no matcher witness digest.
    """
    kept = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    dropped = _match("a", "b", left_plane=1, right_plane=1, score=0.5)
    witness = _pair_witness()
    refusal = lattice.WitnessRefusalV1(
        "refused", "budget_exhausted", "matcher_pair_entries", 0,
        _MATCH_OVERLAP, 0)
    left_signature, right_signature = object(), object()
    candidate_witnesses = {
        id(kept): (witness, left_signature, right_signature),
        id(dropped): (refusal, left_signature, right_signature),
    }
    filtered_index = {id(kept): 0}
    hashes = {id(left_signature): _HEX, id(right_signature): _HEX_B}
    kept_row = plane_graph._pre_filter_witness_row(
        kept, admitted_row_index=0, filtered_index=filtered_index,
        candidate_witnesses=candidate_witnesses,
        signature_sha256_by_identity=hashes)
    dropped_row = plane_graph._pre_filter_witness_row(
        dropped, admitted_row_index=1, filtered_index=filtered_index,
        candidate_witnesses=candidate_witnesses,
        signature_sha256_by_identity=hashes)
    assert kept_row.disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT
    assert kept_row.payload is not None and kept_row.envelope_sha256 is not None
    assert kept_row.payload.witness is witness
    assert dropped_row.disposition == plane_graph.SAME_AXIS_DISPOSITION_DROPPED
    assert dropped_row.payload is None
    assert dropped_row.refusal is refusal
    assert dropped_row.refusal_reason == "budget_exhausted"
    assert dropped_row.envelope_sha256 is None
    assert dropped_row.to_record()["matcher_witness_sha256"] is None
    assert dropped_row.to_record()["envelope_sha256"] is None
    # The refused row's match digest is its own dropped candidate's, not the
    # survivor's, so the two rows can never alias.
    assert dropped_row.match_sha256 != kept_row.match_sha256


def _refusing_pair(refusing_key):
    """Controlled matcher that mints a budget refusal for one admitted key."""
    def pair(left_signature, right_signature, *, quadrant, max_shift_cells,
             refusals=None, diagnostic_sink=None, witness_sink=None,
             witness_budget=None, **_kwargs):
        left = left_signature.reading
        right = right_signature.reading
        key = (left.frame, right.frame,
               float(left_signature.blocks_per_period),
               float(right_signature.blocks_per_period), int(quadrant))
        score = (0.9 if key in {
            ("a", "b", 1.0, 1.0, 0), ("a", "b", 0.5, 1.0, 0),
            ("a", "c", 1.0, 1.0, 0), ("b", "c", 1.0, 1.0, 0)} else None)
        if score is None:
            return None
        if diagnostic_sink is not None:
            diagnostic_sink.pairs.append(plane_graph.PlanePairTrace(
                left.frame, right.frame, left.axis, right.axis, int(quadrant),
                True, "", None, None, None, None, None, None,
                psr=5.0, score=score, overlap_cells=_MATCH_OVERLAP,
                cell_shift_rc=(0, 0)))
        if witness_sink is not None:
            if key == refusing_key:
                witness_sink.append(lattice.WitnessRefusalV1(
                    "refused", "budget_exhausted", "matcher_pair_entries", 0,
                    _MATCH_OVERLAP, 0))
            else:
                witness_sink.append(_pair_witness())
        return plane_graph.PlaneMatch(
            left=left.frame, right=right.frame, left_axis=left.axis,
            right_axis=right.axis,
            left_b=float(left_signature.blocks_per_period),
            right_b=float(right_signature.blocks_per_period),
            quadrant_of_right=int(quadrant), psr=5.0, peak=1.0, second_peak=0.1,
            overlap_cells=_MATCH_OVERLAP, integer_residual=None, score=score,
            cell_shift_rc=(0, 0))
    return pair


def test_real_matcher_refused_dropped_row_never_aliases_a_survivor(monkeypatch):
    readings, signatures = _candidate_fixture()
    monkeypatch.setattr(
        plane_graph, "match_plane_pair",
        _refusing_pair(("a", "b", 0.5, 1.0, 0)))
    observations: list = []
    rows: list = []
    hashes = {id(signature): f"{index + 1:064x}"
              for index, signature in enumerate(signatures)}
    plane_graph.match_plane_graph(
        readings, signatures, candidate_observation_sink=observations,
        pre_filter_witness_sink=rows, signature_sha256_by_identity=hashes)
    refused = [row for row in rows if row.refusal_reason == "budget_exhausted"]
    payloads = [row for row in rows if row.payload is not None]
    assert len(refused) == 1 and len(payloads) == 3
    refused_row = refused[0]
    assert refused_row.disposition == plane_graph.SAME_AXIS_DISPOSITION_DROPPED
    assert refused_row.payload is None and refused_row.envelope_sha256 is None
    assert refused_row.to_record()["matcher_witness_sha256"] is None
    # The refused dropped row never carries a survivor's envelope or witness.
    for row in payloads:
        assert row.envelope_sha256 is not None
        assert row.envelope_sha256 != refused_row.envelope_sha256
        assert row.to_record()["matcher_witness_sha256"] is not None


# ---------------------------------------------------------------------------
# (g)/(h) logical pairs are real cells: the border sentinel is never selected
# ---------------------------------------------------------------------------

def _raw_pair_witness(pairs, *, target=(6, 6), source_shape=(4, 4)):
    """A ``MatcherPairWitnessV1`` over ``pairs`` with a real warp witness bound."""
    warp = _warp_witness(target, source_shape)
    pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 4)
    pair_array.flags.writeable = False
    homography = np.eye(3, dtype=np.float64)
    homography.flags.writeable = False
    return plane_graph.MatcherPairWitnessV1(
        "mc-photogrammetry-matcher-pair-witness-v1",
        plane_graph.PLANE_CORRELATION_ALGORITHM, 32, 0, homography,
        warp.witness_sha256, target, (0, 0), len(pair_array), pair_array,
        warp_witness=warp)


def _payload_with_pairs(pairs, *, source_shape=(4, 4), target_shape=(6, 6)):
    logical = np.asarray(pairs, dtype=np.int64).reshape(-1, 4)
    logical = np.frombuffer(logical.tobytes(order="C"),
                            dtype=np.int64).reshape(logical.shape)
    logical.flags.writeable = False
    rows = np.zeros(target_shape, dtype=np.int64)
    columns = np.zeros(target_shape, dtype=np.int64)
    for value in (rows, columns):
        value.flags.writeable = False
    return csp.MatcherSupportPayloadV1(
        logical_pairs=logical, warp_source_rows=rows,
        warp_source_columns=columns, source_shape=source_shape,
        target_shape=target_shape)


def test_logical_pairs_reject_sentinel_and_out_of_grid_but_accept_zero():
    # ``0`` is a real cell index, not a second sentinel.
    valid = _raw_pair_witness([(0, 0, 0, 0)])
    assert valid.logical_pairs.tolist() == [[0, 0, 0, 0]]
    for bad in ((-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, -1),
                (4, 0, 0, 0), (0, 4, 0, 0), (0, 0, 6, 0), (0, 0, 0, 6)):
        with pytest.raises(ValueError, match="logical pair"):
            _raw_pair_witness([bad])


def test_support_payload_and_validator_reject_sentinel_and_out_of_grid_pairs():
    assert _payload_with_pairs([(0, 0, 0, 0)]).logical_pairs.tolist() == [
        [0, 0, 0, 0]]
    for bad in ((-1, 0, 0, 0), (0, 0, -1, 0), (4, 0, 0, 0),
                (0, 4, 0, 0), (0, 0, 6, 0), (0, 0, 0, 6)):
        with pytest.raises(csp.CandidateSupportError):
            _payload_with_pairs([bad])

    match = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    rows = (_payload_row(match, index=0, disposition="kept", post_index=0),)
    left_binding = _binding("a", _pixels(41), _png_bytes((1, 1, 2)))
    right_binding = _binding("b", _pixels(42), _png_bytes((3, 3, 4)),
                             signature_sha256=_HEX_B)
    record = _collect((match,), rows, ("kept",),
                      witnesses=(left_binding, right_binding))
    assert record.status == "assembled", record.reason
    support = record.rows[0].support
    assert support is not None
    # A post-construction tamper that reintroduces the warp border sentinel into a
    # selected logical pair still fails closed in the strict validator.
    tampered = np.array(support.logical_pairs, dtype=np.int64)
    tampered[0, 0] = -1
    tampered.flags.writeable = False
    object.__setattr__(support, "logical_pairs", tampered)
    with pytest.raises(ValueError, match="sentinel logical pair"):
        csp.validate_candidate_support_provenance(record)


# ---------------------------------------------------------------------------
# (t)-(v) ingress chain over one shared read
# ---------------------------------------------------------------------------

def _write_png(path: Path, colour, *, size=(12, 12)) -> Path:
    path.write_bytes(_png_bytes(colour, size=size))
    return path


def test_discover_media_streams_one_buffer_per_retained_candidate():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = _write_png(root / "a.png", (1, 2, 3))
        duplicate = root / "a_dup.png"
        duplicate.write_bytes(first.read_bytes())
        _write_png(root / "b.png", (4, 5, 6))
        seen: list = []
        live: list = []

        def sink(path, frame, data):
            # No other callback is in flight: the previous buffer was released.
            assert live == []
            live.append(True)
            try:
                seen.append((Path(path).name, frame.frame_id))
                assert isinstance(data, bytes)
            finally:
                live.pop()

        manifest = media_mod.discover_media([root], ingress_sink=sink)
        assert duplicate.exists()
    assert [Path(frame.source_path).name for frame in manifest.frames] == [
        "a.png", "b.png"]
    assert [row[0] for row in seen] == ["a.png", "b.png"]
    # There is no aggregate byte cache: discovery's public interface carries only
    # the synchronous sink, never a retained buffer mapping.
    import inspect

    parameters = inspect.signature(media_mod.discover_media).parameters
    assert "byte_cache" not in parameters
    assert list(parameters) == ["paths", "recursive", "ingress_sink"]


def test_single_read_chain_reads_each_file_once_and_is_content_only(
        monkeypatch, tmp_path):
    _write_png(tmp_path / "a.png", (10, 20, 30))
    _write_png(tmp_path / "b.png", (40, 50, 60))
    opens: dict[str, int] = {}
    reads: dict[str, int] = {}
    original_open = Path.open
    original_read_bytes = Path.read_bytes

    def counting_open(self, mode="r", *args, **kwargs):
        opens[str(self.resolve())] = opens.get(str(self.resolve()), 0) + 1
        return original_open(self, mode, *args, **kwargs)

    def counting_read_bytes(self, *args, **kwargs):
        reads[str(self.resolve())] = reads.get(str(self.resolve()), 0) + 1
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    captured: dict = {}

    def sink(path, frame, data):
        captured[frame.frame_id] = (hashlib.sha256(data).hexdigest(), len(data))

    manifest = media_mod.discover_media([tmp_path], ingress_sink=sink)
    monkeypatch.undo()
    for frame in manifest.frames:
        resolved = str(Path(frame.source_path).resolve())
        # Exactly one physical read: one ``read_bytes`` (which opens the file
        # once) and no second open/read of the same source anywhere.
        assert reads[resolved] == 1
        assert opens[resolved] == 1
        digest, length = captured[frame.frame_id]
        assert digest == frame.content_hash
        assert length == len((tmp_path / Path(frame.source_path).name).read_bytes())


def test_duplicate_bytes_are_deduplicated_by_the_shared_streamed_hash():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        colour = (1, 2, 3)
        payload = _png_bytes(colour)
        (root / "a.png").write_bytes(payload)
        (root / "a_dup.png").write_bytes(payload)
        calls: list = []
        manifest = media_mod.discover_media(
            [root], ingress_sink=lambda path, frame, data: calls.append(frame.frame_id))
    assert len(manifest.frames) == 1
    assert calls == ["image-0000"]


def test_streamed_and_default_discovery_agree_on_order_dedup_and_hash():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_png(root / "a.png", (1, 2, 3))
        colour = (4, 5, 6)
        payload = _png_bytes(colour)
        (root / "b.png").write_bytes(payload)
        (root / "b_dup.png").write_bytes(payload)
        _write_png(root / "c.png", (7, 8, 9))
        streamed = media_mod.discover_media(
            [root], ingress_sink=lambda path, frame, data: None)
        default = media_mod.discover_media([root])
    assert [Path(f.source_path).name for f in streamed.frames] == [
        Path(f.source_path).name for f in default.frames]
    assert [f.content_hash for f in streamed.frames] == [
        f.content_hash for f in default.frames]
    assert [f.frame_id for f in streamed.frames] == [
        f.frame_id for f in default.frames]


def test_ingress_pin_binds_byte_hash_pixel_digest_and_transform():
    pixels = _pixels(30)
    data = _png_bytes((7, 8, 9))
    from mc_photogrammetry.pipeline import image_derived_session as session_mod

    ingress = _make_ingress("f", pixels, data)
    assert ingress.byte_sha256 == hashlib.sha256(data).hexdigest()
    assert ingress.byte_length == len(data)
    assert ingress.decoded_shape == pixels.shape
    assert ingress.decoded_pixel_sha256 == session_mod.image_sha256(pixels)
    assert ingress.transform.resolved_output_size == (pixels.shape[1], pixels.shape[0])
    record = ingress.to_record()
    assert ingress_mod.FrameIngressProvenanceV1.from_record(record) == ingress


def test_ingress_pin_typed_refusals_for_missing_bytes_and_bad_shape():
    with pytest.raises(ingress_mod.IngressProvenanceError) as missing:
        ingress_mod.build_frame_ingress_provenance(
            frame_id="f", source_label="f.png", encoded_bytes=b"",
            decoded_pixels=_pixels(31),
            transform=FrameTransform(source_size=(32, 32)),
            decoded_pixel_sha256=_HEX)
    assert missing.value.reason == ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE
    with pytest.raises(ingress_mod.IngressProvenanceError) as bad:
        ingress_mod.build_frame_ingress_provenance(
            frame_id="f", source_label="f.png", encoded_bytes=b"xyz",
            decoded_pixels=np.zeros((0, 0, 3), dtype=np.uint8),
            transform=FrameTransform(source_size=(0, 0)),
            decoded_pixel_sha256=_HEX)
    assert bad.value.reason == ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE
    with pytest.raises(ingress_mod.IngressProvenanceError) as digest:
        ingress_mod.build_frame_ingress_provenance(
            frame_id="f", source_label="f.png", encoded_bytes=b"xyz",
            decoded_pixels=_pixels(32),
            transform=FrameTransform(source_size=(32, 32)),
            decoded_pixel_sha256="not-a-digest")
    assert digest.value.reason == ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE


def test_image_decode_bytes_rejects_non_buffer_input():
    with pytest.raises(ImageDecodeError):
        load_image_bytes(b"", suffix=".png")
    with pytest.raises(ImageDecodeError):
        load_image_bytes(b"not an image", suffix=".png")


def test_ingress_pin_rejects_a_transform_sized_for_another_frame():
    pixels = _pixels(33)
    with pytest.raises(ingress_mod.IngressProvenanceError) as error:
        _make_ingress("f", pixels, _png_bytes((1, 1, 1)),
                      transform=FrameTransform(source_size=(64, 64)))
    assert error.value.reason == ingress_mod.FRAME_TRANSFORM_UNAVAILABLE


def test_ingress_pin_is_immutable_and_keeps_a_stable_hash():
    pixels = _pixels(36)
    ingress = _make_ingress("f", pixels, _png_bytes((3, 4, 5)))
    digest = ingress.__hash__()
    record = ingress.to_record()

    for name, value in (
            ("frame_id", "other"),
            ("source_label", "other.png"),
            ("byte_sha256", _HEX_B),
            ("byte_length", 1),
            ("decoded_shape", (1, 1, 3)),
            ("decoded_dtype", "<f4"),
            ("decoded_pixel_sha256", _HEX_B),
            ("transform", FrameTransform(source_size=(1, 1))),
            ("schema", "other-schema"),
            ("decoder", "other-decoder")):
        with pytest.raises(AttributeError):
            setattr(ingress, name, value)
    with pytest.raises(AttributeError):
        del ingress.frame_id

    assert ingress.to_record() == record
    assert ingress.__hash__() == digest
    rebuilt = ingress_mod.FrameIngressProvenanceV1.from_record(record)
    assert rebuilt == ingress
    assert hash(rebuilt) == digest
    # The pin is usable as a hash key and cannot be silently re-keyed.
    assert {ingress: "pinned"}[rebuilt] == "pinned"


def test_ingress_pin_from_record_rejects_noncanonical_integers():
    ingress = _make_ingress("f", _pixels(37), _png_bytes((6, 7, 8)))
    record = ingress.to_record()

    def mutated(path, value):
        payload = json.loads(json.dumps(record))
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return payload

    # Each value is one an eager ``int(...)`` coercion would have accepted
    # silently (the coerced result is a canonical/valid pin) or a bool, which is
    # an ``int`` subclass. All must now refuse.
    cases = (
        (["decoded_shape", 2], 3.0, ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE),
        (["decoded_shape", 0], True, ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE),
        (["decoded_shape"], "3232", ingress_mod.INGRESS_PROVENANCE_UNAVAILABLE),
        (["transform", "source_size", 1], "32", ingress_mod.FRAME_TRANSFORM_UNAVAILABLE),
        (["transform", "source_size", 0], 32.0, ingress_mod.FRAME_TRANSFORM_UNAVAILABLE),
        (["transform", "source_size"], None, ingress_mod.FRAME_TRANSFORM_UNAVAILABLE),
        (["transform", "crop_origin"], [True, True], ingress_mod.FRAME_TRANSFORM_UNAVAILABLE),
        (["transform", "crop_size"], [32.0, 32.0], ingress_mod.FRAME_TRANSFORM_UNAVAILABLE),
        (["transform", "output_size"], [32, "32"], ingress_mod.FRAME_TRANSFORM_UNAVAILABLE),
        (["byte_length"], True, ingress_mod.BYTE_DIGEST_MISMATCH),
        (["byte_length"], 12.5, ingress_mod.BYTE_DIGEST_MISMATCH),
    )
    for path, value, reason in cases:
        with pytest.raises(ingress_mod.IngressProvenanceError) as error:
            ingress_mod.FrameIngressProvenanceV1.from_record(mutated(path, value))
        assert error.value.reason == reason, path

    # The canonical record still round-trips byte-identically.
    assert ingress_mod.FrameIngressProvenanceV1.from_record(record) == ingress
    assert (ingress_mod.FrameIngressProvenanceV1.from_record(record).to_record()
            == record)


def test_session_ingress_round_trips_and_binds_the_same_frame(monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_session as session_mod

    _patch_camera_solver(monkeypatch)
    pixels = _pixels(34)
    data = _png_bytes((2, 2, 2))
    ingress = _make_ingress("f", pixels, data)
    session = session_mod.build_image_derived_session(
        {"f": pixels}, source_paths={"f": "/label/f.png"}, ingress={"f": ingress})
    assert session.frames[0].ingress == ingress
    record = session_mod.image_derived_session_to_record(session)
    assert record["frames"][0]["ingress"]["byte_sha256"] == ingress.byte_sha256
    rebuilt = session_mod.image_derived_session_from_record(record)
    assert rebuilt.session_sha256 == session.session_sha256
    assert rebuilt.frames[0].ingress == ingress


def test_session_without_ingress_carries_no_ingress_field_and_same_digest(
        monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_session as session_mod

    _patch_camera_solver(monkeypatch)
    pixels = _pixels(35)
    plain = session_mod.build_image_derived_session({"f": pixels})
    payload = session_mod.image_derived_session_to_record(plain)
    assert "ingress" not in payload["frames"][0]
    padded = session_mod.build_image_derived_session({"f": pixels}, ingress={})
    assert padded.session_sha256 == plain.session_sha256


def _patch_camera_solver(monkeypatch):
    """A synthetic camera solution so a tiny image can build a session."""
    from mc_photogrammetry.pipeline.contracts import (
        CameraEstimate, CameraIntrinsics, CameraPose, CameraSolution)
    from mc_photogrammetry.pipeline import image_only

    estimate = CameraEstimate(
        intrinsics=CameraIntrinsics((12, 12), (8.0, 8.0), (6.0, 6.0)),
        pose=CameraPose(position=None, yaw_degrees=14.0, pitch_degrees=17.0,
                        roll_degrees=3.5),
        residual_pixels=0.4, confidence=0.0)

    def fake_solve(frames):
        unresolved: list[str] = []
        for frame_id in frames:
            unresolved.append(f"{frame_id}: position (vanishing points fix no translation)")
            unresolved.append(f"{frame_id}: scale (needs the lattice period)")
            unresolved.append(f"{frame_id}: yaw quadrant (recovered modulo 90 degrees)")
        return CameraSolution(
            cameras={frame_id: estimate for frame_id in frames},
            unresolved=tuple(unresolved), scale_known=False)

    monkeypatch.setattr(image_only, "solve_image_cameras", fake_solve)


def test_validate_candidate_support_record_replays_and_is_fail_closed():
    match = _match("a", "b", left_plane=0, right_plane=0, score=0.9)
    rows = (_payload_row(match, index=0, disposition="kept", post_index=0),)
    witness = _binding("a", _pixels(36), _png_bytes((3, 4, 5)))
    other = _binding("b", _pixels(37), _png_bytes((6, 7, 8)),
                     signature_sha256=_HEX_B)
    record = _collect((match,), rows, ("kept",), witnesses=(witness, other),
                      ingress=(witness.ingress, other.ingress))
    assert record.status == "assembled", record.reason
    csp.validate_candidate_support_provenance(record)
    assert record.record_sha256 == csp.candidate_support_record_digest(record)
    stale = replace(record)
    object.__setattr__(stale, "record_sha256", "f" * 64)
    with pytest.raises(ValueError):
        csp.validate_candidate_support_provenance(stale)
    bad_count = replace(record)
    object.__setattr__(bad_count, "row_count", 9)
    with pytest.raises(ValueError):
        csp.validate_candidate_support_provenance(bad_count)


def test_accepted_r4_seams_keep_only_the_session_bound_ingress_and_sinks():
    """The unaccepted side-channel ingress hooks stay removed (Lead amendment)."""
    import inspect

    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod

    entry_parameters = inspect.signature(
        entry.build_image_derived_entry).parameters
    signature_parameters = inspect.signature(
        signatures_mod.derive_image_derived_signatures).parameters
    graph_parameters = inspect.signature(
        graph_mod._derive_image_derived_graph).parameters
    # The three rejected side channels must not reappear on any seam.
    assert "ingress_observer" not in entry_parameters
    assert "ingress_by_frame" not in signature_parameters
    assert "candidate_support_ingress" not in graph_parameters
    # The accepted R4 surface is preserved: session-bound capture plus the
    # ordinary witness/support sinks.
    assert entry_parameters["capture_ingress"].default is False
    assert signature_parameters["signature_witness_sink"].default is None
    assert graph_parameters["candidate_support_sink"].default is None
    assert graph_parameters["signature_witnesses"].default is None


def test_entry_capture_ingress_runs_the_single_read_chain_end_to_end(
        monkeypatch, tmp_path):
    """The real entry runs with the ingress observer on, once per file."""
    from test_image_derived_entry import (
        _patch_matcher, _patch_solver, _write_png)

    _write_png(tmp_path / "a.png", (10, 20, 30))
    _write_png(tmp_path / "b.png", (40, 50, 60))
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    result = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "captured.mcscan",
        capture_ingress=True)
    assert len(result.session.frames) == 2
    for frame in result.session.frames:
        pin = frame.ingress
        assert pin is not None
        assert pin.frame_id == frame.frame_id
        assert pin.decoded_pixel_sha256 == frame.image_sha256
        assert pin.decoded_shape == frame.image_shape
        assert pin.byte_length > 0
        assert len(pin.byte_sha256) == 64
        assert pin.transform.resolved_output_size == (
            frame.image_shape[1], frame.image_shape[0])


def test_entry_capture_ingress_is_output_neutral_and_default_carries_none(
        monkeypatch, tmp_path):
    from test_image_derived_entry import (
        _patch_matcher, _patch_solver, _write_png)

    _write_png(tmp_path / "a.png", (10, 20, 30))
    _write_png(tmp_path / "b.png", (40, 50, 60))
    _patch_solver(monkeypatch)
    _patch_matcher(monkeypatch)
    control = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "control.mcscan")
    observed = entry.build_image_derived_entry(
        [tmp_path], project_path=tmp_path / "observed.mcscan",
        capture_ingress=True)
    # The default path carries no ingress pin and stays byte-for-byte neutral on
    # decoded pixels, shapes and digests.
    assert all(frame.ingress is None for frame in control.session.frames)
    assert observed.session.frames[0].ingress is not None
    for left, right in zip(control.session.frames, observed.session.frames):
        assert left.image_shape == right.image_shape
        assert left.image_sha256 == right.image_sha256
        assert np.array_equal(left.image, right.image)
    # The pin is additive metadata that changes the version-gated session digest;
    # the graph binds that session digest, so it differs only through the pin.
    assert control.session.session_sha256 != observed.session.session_sha256
    session_mod = __import__(
        "mc_photogrammetry.pipeline.image_derived_session",
        fromlist=["image_derived_session_to_record"])
    control_payload = session_mod.image_derived_session_to_record(control.session)
    observed_payload = session_mod.image_derived_session_to_record(observed.session)
    for term in ("frames", "cameras", "components", "unresolved", "scale_known",
                 "local_cameras"):
        if term == "frames":
            for left_frame, right_frame in zip(control_payload["frames"],
                                               observed_payload["frames"]):
                assert {key: value for key, value in left_frame.items()
                        if key != "ingress"} == {
                    key: value for key, value in right_frame.items()
                    if key != "ingress"}
        else:
            assert control_payload[term] == observed_payload[term]
