from dataclasses import replace

import numpy as np
import pytest

from mc_photogrammetry import evaluation_surface_support as subject
from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry import evaluation_signature_sampling as sampling_mod
from mc_photogrammetry import synthetic_targeted_capture_preflight as preflight_mod
from mc_photogrammetry.export import attribution
from mc_photogrammetry.pipeline.image_derived_signatures import SignatureEvidence
from mc_photogrammetry.reconstruction import lattice, lattice_planes, plane_graph
from test_synthetic_targeted_capture_preflight import _evidence, _frozen
from test_plane_graph import make_reading


def _inputs():
    mask = np.ones((8, 8), dtype=bool)
    left_truth = _evidence(mask)
    right_truth = replace(_evidence(mask), truth_sha256="d" * 64,
                          evidence_sha256="")
    evidences = []
    for frame in ("left", "right"):
        reading = make_reading(frame, 0, 1.0, ((1.0, 2.0, 1.0),))
        cells = _frozen(np.arange(16, dtype=np.float32).reshape(4, 4))
        interior = _frozen(np.zeros((4, 4), dtype=np.float32))
        coverage = _frozen(np.ones((4, 4), dtype=bool))
        signature = lattice_planes.PlaneSignature(
            reading, 1.0, cells, interior, coverage, (0.0, 0.0))
        evidences.append(SignatureEvidence(frame, 0, 0, 0, 1.0, signature))
    match = plane_graph.PlaneMatch(
        "left", "right", 0, 0, 1.0, 1.0, 0, 4.0, 1.0, 0.2, 9, None,
        0.8, 0, 0, (0, 0))
    return left_truth, right_truth, evidences[0], evidences[1], match


def _evaluate(**changes):
    left, right, left_sig, right_sig, match = _inputs()
    values = dict(left_truth=left, right_truth=right,
                  left_signature=left_sig, right_signature=right_sig,
                  match=match, homography=np.eye(3),
                  correlation_algorithm_id=plane_graph.PLANE_CORRELATION_ALGORITHM,
                  max_shift_cells=32)
    values.update(changes)
    return subject.evaluate_surface_support(**values)


def test_exact_sampling_dependency_gap_is_typed_atomic_and_bound():
    result = _evaluate()
    assert result.status == "refused"
    assert result.reason == "unavailable_exact_dependencies"
    assert result.missing_dependencies == (
        "rectification_nonzero_source_taps", "gradient_neighbor_source_taps",
        "cell_aggregation_source_taps", "renderer_subface_plane_identity")
    assert result.left_signature_sha256 and result.right_signature_sha256
    assert result.match_sha256 and result.homography_sha256
    assert result.whole_reading_overlap_cells == 9
    assert result.to_record()["report_sha256"] == result.report_sha256
    assert "surface_support_cells" not in result.to_record()


def test_refusal_is_deterministic_and_does_not_mutate_inputs():
    left, right, left_sig, right_sig, match = _inputs()
    before = (left.rgb.tobytes(), right.ownership.tobytes(),
              left_sig.signature.cells.tobytes())
    kwargs = dict(left_truth=left, right_truth=right,
                  left_signature=left_sig, right_signature=right_sig,
                  match=match, homography=np.eye(3),
                  correlation_algorithm_id=plane_graph.PLANE_CORRELATION_ALGORITHM,
                  max_shift_cells=32)
    first = subject.evaluate_surface_support(**kwargs)
    second = subject.evaluate_surface_support(**kwargs)
    assert first == second
    assert before == (left.rgb.tobytes(), right.ownership.tobytes(),
                      left_sig.signature.cells.tobytes())


def test_budget_and_provenance_fail_closed_before_surface_counts():
    budget = _evaluate(max_source_pixels=1)
    assert budget.reason == "budget_exhausted"
    mismatch = _evaluate(correlation_algorithm_id="legacy")
    assert mismatch.reason == "provenance_mismatch"
    bad_match = replace(_inputs()[-1], left_plane=2)
    endpoint = _evaluate(match=bad_match)
    assert endpoint.reason == "provenance_mismatch"
    for result in (budget, mismatch, endpoint):
        assert "surface_support_cells" not in result.to_record()


def test_positive_infinity_is_canonical_tag_not_nonfinite_json():
    match = replace(_inputs()[-1], psr=float("inf"))
    result = _evaluate(match=match)
    assert result.whole_reading_psr == "positive_infinity"
    assert result.reason == "unavailable_exact_dependencies"


def _payload_fixture(*, right_owner=0, unknown=False):
    left, right, left_sig, right_sig, _match = _inputs()
    # Both rendered frames originate from the same immutable truth artifact.
    right = replace(right, truth_sha256=left.truth_sha256,
                    camera=replace(right.camera, position=(13., 5., 5.)),
                    evidence_sha256="")
    owners = []
    sampling = []
    for truth, signature, plane_offset in ((left, left_sig, 0),
                                           (right, right_sig, right_owner)):
        ownership = np.zeros(truth.ownership.shape, dtype=np.int64)
        if unknown:
            ownership.flat[0] = -1
        ownership.flags.writeable = False
        truth = replace(truth, ownership=ownership, evidence_sha256="")
        row = attribution.RendererOwnerRowV1(
            0, (0, 0, 0), "minecraft:stone", "box", 0, "east",
            ((1., 0., 0.), (1., 1., 0.), (1., 1., 1.), (1., 0., 1.)),
            (1., 0., 0.), "a" * 64, "", (0, 1, 1 + plane_offset, 1))
        renderer = attribution.RendererOwnershipWitnessV1(
            "mc-photogrammetry-renderer-ownership-witness-v1", 1., .5,
            "strict-nearer-inverse-depth-v1", "face-order-stable-v1", (row,))
        binding = truth_mod.bind_renderer_ownership_witness(truth, renderer)
        owners.append(truth_mod.RenderedTruthOwnershipWitnessPayloadV1(binding, renderer))
        maps = np.arange(16, dtype=np.float32).reshape(4, 4) % 8
        map_y = np.arange(16, dtype=np.float32).reshape(4, 4) // 8
        rect = lattice.rectification_witness_v1(
            maps, map_y, np.ones((4, 4), dtype=bool), source_shape=(8, 8))
        ptr = np.arange(17, dtype=np.int64); ptr.flags.writeable = False
        indices = np.arange(16, dtype=np.int64); indices.flags.writeable = False
        witness = lattice_planes.SignatureSamplingWitnessV1(
            "mc-photogrammetry-signature-sampling-witness-v1", rect, rect, None,
            "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64,
            1, 1, 1, tuple((0, 1, 0, 1) for _ in range(16)),
            tuple((0, 1, 0, 1) for _ in range(16)), ptr, indices)
        sb = sampling_mod.SignatureSamplingBindingV1(
            sampling_mod.SCHEMA, signature.frame_id, signature.canonical_index,
            signature.alternative_index, signature.signature_sha256,
            truth.evidence_sha256, witness.witness_sha256)
        sampling.append(sampling_mod.SignatureSamplingWitnessPayloadV1(sb, witness))
        if signature is left_sig: left = truth
        else: right = truth
    pairs = np.asarray([[r, c, r, c] for r in range(3) for c in range(3)], dtype=np.int64)
    pairs.flags.writeable = False
    homography = np.eye(3); homography.flags.writeable = False
    warp_rows = []
    lattice_planes.warp_signature(
        left_sig.signature, homography, right_sig.signature.cells.shape,
        witness_sink=warp_rows)
    matcher = plane_graph.MatcherPairWitnessV1(
        "mc-photogrammetry-matcher-pair-witness-v1",
        plane_graph.PLANE_CORRELATION_ALGORITHM, 32, 0, homography,
        warp_rows[0].witness_sha256, (132, 132), (0, 0), 9, pairs,
        warp_witness=warp_rows[0])
    envelope = plane_graph.GraphMatcherWitnessEnvelopeV1(
        "mc-photogrammetry-graph-matcher-witness-envelope-v1", "7" * 64,
        left_sig.signature_sha256, right_sig.signature_sha256,
        "left", "right", 0, 0, 0, 0, 0, (0, 0), 9, 4., .8,
        matcher.witness_sha256)
    match_payload = plane_graph.GraphMatcherWitnessPayloadV1(envelope, matcher)
    public = preflight_mod.SyntheticPreflightEvidenceBundleV1(
        "mc-photogrammetry-synthetic-preflight-evidence-bundle-v1",
        ("left", "right"), (left.evidence_sha256, right.evidence_sha256),
        tuple(row.binding.binding_sha256 for row in owners),
        tuple(row.binding.report_sha256 for row in sampling),
        (envelope.envelope_sha256,), "8" * 64)
    bundle = preflight_mod.SyntheticPreflightWitnessPayloadV1(
        public, tuple(owners), tuple(sampling), (match_payload,))
    return bundle, left, right, left_sig, right_sig


def test_payload_evaluator_conserves_pure_surface_pairs_and_is_deterministic():
    values = _payload_fixture()
    first = subject.evaluate_surface_support_from_payload(
        payload=values[0], left_truth=values[1], right_truth=values[2],
        left_signature=values[3], right_signature=values[4])
    second = subject.evaluate_surface_support_from_payload(
        payload=values[0], left_truth=values[1], right_truth=values[2],
        left_signature=values[3], right_signature=values[4])
    assert first == second and first.status == "complete"
    assert first.pair_count == first.same_surface_pairs == 9
    assert first.different_surface_pairs == first.mixed_pairs == first.unknown_pairs == 0
    assert first.surfaces[0].surface_support_cells == 9
    assert first.surfaces[0].surface_share_of_pair_support == 1.0
    assert first.surfaces[0].left_endpoint_cells == 9
    assert first.surfaces[0].right_endpoint_cells == 9
    assert first.surfaces[0].left_surface_canvas_cells == 16
    assert first.surfaces[0].right_surface_canvas_cells == 16
    assert first.surfaces[0].surface_overlap_fraction == 9 / 16


def test_payload_evaluator_unknown_precedence_and_atomic_cap():
    values = _payload_fixture(unknown=True)
    result = subject.evaluate_surface_support_from_payload(
        payload=values[0], left_truth=values[1], right_truth=values[2],
        left_signature=values[3], right_signature=values[4])
    assert result.unknown_pairs == 1 and result.same_surface_pairs == 8
    refused = subject.evaluate_surface_support_from_payload(
        payload=values[0], left_truth=values[1], right_truth=values[2],
        left_signature=values[3], right_signature=values[4], max_pair_cells=8)
    assert refused.status == "refused" and refused.reason == "budget_exhausted"


def test_payload_evaluator_distinguishes_certified_physical_planes():
    values = _payload_fixture(right_owner=1)
    result = subject.evaluate_surface_support_from_payload(
        payload=values[0], left_truth=values[1], right_truth=values[2],
        left_signature=values[3], right_signature=values[4])
    assert result.different_surface_pairs == 9
    assert result.same_surface_pairs == result.mixed_pairs == result.unknown_pairs == 0
    assert result.surfaces == ()


def test_payload_evaluator_refuses_join_tamper_without_counts():
    values = _payload_fixture()
    cells = values[3].signature.cells.copy()
    cells[0, 0] += 1
    cells.flags.writeable = False
    changed = replace(values[3], signature=replace(
        values[3].signature, cells=cells), signature_sha256="")
    result = subject.evaluate_surface_support_from_payload(
        payload=values[0], left_truth=values[1], right_truth=values[2],
        left_signature=changed, right_signature=values[4])
    assert result.status == "refused" and result.reason == "provenance_mismatch"


def test_aggregate_payload_replays_each_nested_carrier_tamper():
    for group, digest_name in (("ownership", "witness_sha256"),
                               ("signatures", "witness_sha256"),
                               ("matches", "witness_sha256")):
        values = _payload_fixture()
        payload = values[0]
        row = getattr(payload, group)[0]
        object.__setattr__(row.witness, digest_name, "0" * 64)
        with np.testing.assert_raises(ValueError):
            preflight_mod.SyntheticPreflightWitnessPayloadV1(
                payload.public_bundle, payload.ownership,
                payload.signatures, payload.matches)
    payload = _payload_fixture()[0]
    payload.matches[0].witness.logical_pairs.flags.writeable = True
    with np.testing.assert_raises(ValueError):
        preflight_mod.SyntheticPreflightWitnessPayloadV1(
            payload.public_bundle, payload.ownership,
            payload.signatures, payload.matches)


def test_pair_coordinates_are_checked_per_dimension_without_linear_aliasing():
    values = _payload_fixture()
    payload = values[0]
    pairs = payload.matches[0].witness.logical_pairs.copy()
    pairs[0, :2] = (-1, values[3].signature.cells.shape[1])
    pairs.flags.writeable = False
    with pytest.raises(ValueError, match="sentinel logical pair"):
        replace(payload.matches[0].witness,
                logical_pairs=pairs, witness_sha256="")
    pairs = payload.matches[0].witness.logical_pairs.copy()
    pairs[0, :2] = (0, values[3].signature.cells.shape[1])
    pairs.flags.writeable = False
    with pytest.raises(ValueError, match="outside its grid"):
        replace(payload.matches[0].witness,
                logical_pairs=pairs, witness_sha256="")


def test_stale_warp_signature_and_truth_mutations_fail_replay():
    values = _payload_fixture()
    payload = values[0]
    warp = payload.matches[0].witness.warp_witness
    warp.source_rows.flags.writeable = True
    warp.source_rows[0, 0] += 1
    warp.source_rows.flags.writeable = False
    result = subject.evaluate_surface_support_from_payload(
        payload=payload, left_truth=values[1], right_truth=values[2],
        left_signature=values[3], right_signature=values[4])
    assert result.status == "refused" and result.reason == "provenance_mismatch"

    values = _payload_fixture()
    cells = values[3].signature.cells
    cells.flags.writeable = True; cells[0, 0] += 1; cells.flags.writeable = False
    stale_signature = subject.evaluate_surface_support_from_payload(
        payload=values[0], left_truth=values[1], right_truth=values[2],
        left_signature=values[3], right_signature=values[4])
    assert stale_signature.status == "refused"

    values = _payload_fixture()
    rgb = values[1].rgb
    rgb.flags.writeable = True; rgb[0, 0, 0] += 1; rgb.flags.writeable = False
    stale_truth = subject.evaluate_surface_support_from_payload(
        payload=values[0], left_truth=values[1], right_truth=values[2],
        left_signature=values[3], right_signature=values[4])
    assert stale_truth.status == "refused"
