from dataclasses import replace

import cv2
import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice, lattice_planes


def _replay(witness, source):
    flat = np.asarray(source, dtype=np.float32).reshape(-1)
    result = np.zeros(witness.map_x.size, dtype=np.float32)
    for index in range(len(result)):
        start, end = witness.tap_ptr[index:index + 2]
        for source_index, weight in zip(
                witness.tap_source_indices[start:end],
                witness.tap_weight_numerators[start:end]):
            if source_index >= 0:
                result[index] += flat[source_index] * int(weight) / 1024.0
    return result.reshape(witness.map_x.shape)


def test_convert_maps_taps_match_remap_for_integer_fractional_and_border():
    source = np.arange(25, dtype=np.float32).reshape(5, 5)
    xs = np.asarray([[2.0, 1.25, -0.25, 4.75]], dtype=np.float32)
    ys = np.asarray([[3.0, 2.5, 1.0, 4.75]], dtype=np.float32)
    valid = np.ones(xs.shape, dtype=bool)
    witness = lattice.rectification_witness_v1(
        xs, ys, valid, source_shape=source.shape)
    expected = cv2.remap(source, xs, ys, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    np.testing.assert_array_equal(_replay(witness, source), expected)
    assert -1 in witness.tap_source_indices
    assert all(sum(witness.tap_weight_numerators[witness.tap_ptr[i]:
                                                witness.tap_ptr[i + 1]]) == 1024
               for i in range(xs.size))


def test_rectify_no_sink_is_identical_and_sink_records_float_maps():
    image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    axis = np.eye(3)
    plain = lattice.rectify_plane(image, 2, axis, 12.0, samples=8,
                                  principal_point=(8.0, 8.0))
    sink = []
    observed = lattice.rectify_plane(image, 2, axis, 12.0, samples=8,
                                     principal_point=(8.0, 8.0),
                                     witness_sink=sink)
    np.testing.assert_array_equal(observed, plain)
    assert len(sink) == 1 and sink[0].map_x.dtype == np.float32
    assert sink[0].depth_valid.dtype == np.bool_


def test_rectification_witness_digest_tamper_backend_and_caps_fail_closed():
    xs = np.asarray([[0.5]], dtype=np.float32)
    witness = lattice.rectification_witness_v1(
        xs, xs, np.ones_like(xs, dtype=bool), source_shape=(2, 2))
    assert witness.witness_sha256 == lattice.rectification_witness_v1(
        xs, xs, np.ones_like(xs, dtype=bool), source_shape=(2, 2)).witness_sha256
    bad_map = witness.map_x.copy(); bad_map.flags.writeable = False
    bad_map_view = bad_map.copy(); bad_map_view[0, 0] = 1.0
    bad_map_view.flags.writeable = False
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(witness, map_x=bad_map_view)
    with pytest.raises(ValueError, match="malformed"):
        replace(witness, opencv_version="foreign")
    sink = []
    source = np.ones((2, 2), dtype=np.float32)
    observed = lattice.rectify_plane(
        source, 2, np.eye(3), 2.0, samples=2, witness_sink=sink,
        witness_budget=lattice.WitnessBudgetV1(0, 0, 0, 0))
    assert observed.shape == (2, 2)
    assert sink == [lattice.WitnessRefusalV1(
        "refused", "budget_exhausted", "rectified_samples", 0, 4, 0)]


def test_signature_sampling_witness_records_gradient_and_cell_dependencies():
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    reading = lattice_planes.PlaneReading(
        frame="f", axis=2, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=8.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=17,
        polygon=None, coverage_cells=4, structural_share=.5, confidence=.8,
        rectification=lattice_planes.PlaneRectification(
            tuple(tuple(float(v) for v in row) for row in np.eye(3)), 12.0,
            (16.0, 16.0), (0.0, 0.0)))
    plain = lattice_planes.plane_signature(image, reading, 1.0)
    sink = []
    observed = lattice_planes.plane_signature(
        image, reading, 1.0, witness_sink=sink)
    np.testing.assert_array_equal(observed.cells, plain.cells)
    assert len(sink) == 1
    witness = sink[0]
    assert witness.gradient_edge_order == 1
    assert len(witness.cell_rectangles) == observed.cells.size
    assert witness.cell_source_ptr[-1] == len(witness.cell_source_indices)
    assert witness.region_rectification is None


def test_signature_observer_aggregate_caps_are_atomic_at_boundary():
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    reading = lattice_planes.PlaneReading(
        frame="f", axis=2, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 1.0, 1.0),),
        period_pixels=8.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=17,
        polygon=None, coverage_cells=4, structural_share=.5, confidence=.8,
        rectification=lattice_planes.PlaneRectification(
            tuple(tuple(float(v) for v in row) for row in np.eye(3)), 12.0,
            (16.0, 16.0), (0.0, 0.0)))
    plain = lattice_planes.plane_signature(image, reading, 1.0)
    probe = []
    lattice_planes.plane_signature(image, reading, 1.0, witness_sink=probe)
    samples = (probe[0].rgb_rectification.map_x.size
               + probe[0].coverage_rectification.map_x.size)
    exact = lattice.WitnessBudgetV1(samples, samples * 4, 100_000, 2_000_000)
    for sample_cap, expected in ((samples - 1, "refused"),
                                 (samples, "witness"), (samples + 1, "witness")):
        sink = []
        observed = lattice_planes.plane_signature(
            image, reading, 1.0, witness_sink=sink,
            witness_budget=replace(exact, max_rectified_samples=sample_cap))
        np.testing.assert_array_equal(observed.cells, plain.cells)
        assert len(sink) == 1
        if expected == "refused":
            assert isinstance(sink[0], lattice.WitnessRefusalV1)
            assert sink[0].stage == "aggregate_rectified_samples"
            assert sink[0].limit == samples - 1 and sink[0].planned == samples
        else:
            assert isinstance(sink[0], lattice_planes.SignatureSamplingWitnessV1)
    for budget, stage in (
        (replace(exact, max_cell_memberships=0), "aggregate_cell_memberships"),
        (replace(exact, max_payload_bytes=0), "aggregate_payload_bytes"),
    ):
        sink = []
        observed = lattice_planes.plane_signature(
            image, reading, 1.0, witness_sink=sink, witness_budget=budget)
        np.testing.assert_array_equal(observed.cells, plain.cells)
        assert len(sink) == 1 and isinstance(sink[0], lattice.WitnessRefusalV1)
        assert sink[0].stage == stage and sink[0].actual == 0
