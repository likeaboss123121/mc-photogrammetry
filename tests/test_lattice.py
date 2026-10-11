"""Measuring the block grid on a plane once the camera orientation is known.

The distance a period implies is only as good as the period, so these check the
measurement against signals whose spacing is set by construction, then check the
rectification against a plane placed by construction.
"""
from typing import get_type_hints

import numpy as np
import pytest

from mc_photogrammetry.reconstruction.calibration import axis_directions
from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction.lattice import (
    PeriodEstimate,
    measure_plane,
    TEXELS_PER_BLOCK,
    estimate_period,
    period_hypotheses,
    plane_basis,
    rectify_plane,
    rectified_pixel_step,
    rectified_pixel_to_plane_coordinates,
    plane_coordinates_to_camera_point,
    camera_point_to_pixel,
)


def _striped(size, period, phase=0.0, along=0):
    """A patch of narrow bright seams at a known spacing."""
    coordinate = np.arange(size)
    line = ((coordinate - phase) % period < 1.0).astype(np.float32)
    field = np.repeat(line[None, :], size, axis=0)
    return field if along == 0 else field.T


@pytest.mark.parametrize("along", (0, 1))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("values,expected", (
    ([255, 0], [255]), ([1, 0], [1]), ([0, 255], [255]),
    ([0, 255, 0, 1, 0], [255, 255, 1, 1]),
))
def test_integer_profile_preserves_signed_edge_magnitudes(along, weighted, values, expected):
    patch = np.tile(np.array(values, dtype=np.uint8), (4, 1))
    if along:
        patch = patch.T
    weights = np.full(patch.shape, 2., dtype=np.float64) if weighted else None
    before = patch.tobytes()
    weight_before = weights.tobytes() if weighted else None
    actual = lattice._profile(patch, along, weights)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, lattice._profile(patch.astype(np.float64), along, weights))
    np.testing.assert_array_equal(actual, lattice._profile(255 - patch, along, weights))
    np.testing.assert_array_equal(actual, lattice._profile(np.repeat(patch[..., None], 3, axis=2), along, weights))
    assert actual.dtype == np.float64 and patch.tobytes() == before
    assert not weighted or weights.tobytes() == weight_before


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("along", (0, 1))
@pytest.mark.parametrize("weighted", (False, True))
def test_float_profile_is_exactly_the_original_arithmetic(dtype, along, weighted):
    patch = np.random.default_rng(440).normal(size=(17, 19)).astype(dtype)
    weights = np.random.default_rng(441).random(patch.shape).astype(dtype) if weighted else None
    before = patch.tobytes()
    gradient = np.abs(np.diff(patch, axis=1 - along))
    if weights is None:
        expected = gradient.mean(axis=along)
    else:
        mask = weights[:, :-1] if along == 0 else weights[:-1, :]
        total = mask.sum(axis=along)
        covered = (gradient * mask).sum(axis=along)
        expected = np.divide(covered, total, out=np.zeros_like(covered), where=total > 0)
    actual = lattice._profile(patch, along, weights)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype == dtype and patch.tobytes() == before


@pytest.mark.parametrize("along", (0, 1))
def test_uint8_period_phase_and_hypotheses_equal_float_patch(along):
    patch = (_striped(256, 16., phase=3., along=along) * 255).astype(np.uint8)
    before = patch.tobytes()
    integer = lattice.estimate_period(patch, along=along)
    floating = lattice.estimate_period(patch.astype(np.float64), along=along)
    assert integer is not None and integer == floating
    assert integer.period_pixels == pytest.approx(16., rel=.02)
    assert lattice.period_hypotheses(integer, extent=1., samples=256) == lattice.period_hypotheses(
        floating, extent=1., samples=256)
    assert patch.tobytes() == before


def test_uint8_harmonic_reconciliation_and_coarse_support_equal_float():
    patch = (np.maximum(_striped(256, 16., along=0),
                        _striped(256, 32., along=1)) * 255).astype(np.uint8)
    integer = lattice._plane_period_with_direction(patch)
    floating = lattice._plane_period_with_direction(patch.astype(np.float64))
    assert integer is not None and integer == floating
    assert integer[0].period_pixels == pytest.approx(16., rel=.02)
    textured = (_texelled(512, 4.) * 160).astype(np.uint8)
    integer = lattice.estimate_period(textured)
    floating = lattice.estimate_period(textured.astype(np.float64))
    assert integer is not None and integer == floating
    assert integer.period_pixels == pytest.approx(4., rel=.05) and integer.coarse_support > .5


def test_a_known_stripe_spacing_is_recovered():
    for period in (11.0, 17.0, 32.0, 64.0):
        found = estimate_period(_striped(512, period), along=0)
        assert found is not None, f"no period found at spacing {period}"
        assert found.period_pixels == pytest.approx(period, rel=0.02)


def test_the_period_is_measured_across_the_seams_not_along_them():
    # Stripes running one way must be read on the axis that crosses them; the
    # other axis carries no variation and must decline to answer.
    patch = _striped(256, 16.0, along=0)
    across = estimate_period(patch, along=0)
    assert across is not None and across.period_pixels == pytest.approx(16.0, rel=0.05)
    assert estimate_period(patch, along=1) is None


def test_a_blank_patch_yields_nothing_rather_than_a_number():
    assert estimate_period(np.zeros((128, 128), np.float32)) is None
    assert estimate_period(np.full((128, 128), 0.5, np.float32)) is None


def test_a_period_below_the_floor_is_refused():
    # Two-pixel stripes are indistinguishable from resampling noise.
    assert estimate_period(_striped(256, 2.0), minimum_period=4.0) is None


def test_both_the_block_and_the_texel_reading_are_kept():
    """A period of `p` is block seams, or texel edges sixteen times closer.

    One image cannot separate them, so the milestone requires both to survive.
    """
    base = estimate_period(_striped(512, 32.0))
    assert base is not None
    enriched = period_hypotheses(base, extent=2.0, samples=512)
    assert len(enriched.hypotheses) == 2
    block, texel = enriched.hypotheses
    assert block.interpretation.startswith("one block")
    assert texel.interpretation.startswith("one texel")
    ratio = block.plane_distance / texel.plane_distance
    assert ratio == pytest.approx(TEXELS_PER_BLOCK, rel=1e-6)
    assert enriched.best is block


def test_the_distance_a_period_implies_inverts_correctly():
    # The measured period is close to 32 pixels, and its scale must use the
    # endpoint-inclusive 4/511 world-unit grid interval.
    base = estimate_period(_striped(512, 32.0))
    enriched = period_hypotheses(base, extent=2.0, samples=512)
    expected = 1.0 / (base.period_pixels * rectified_pixel_step(2.0, 512))
    assert enriched.best.plane_distance == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("samples", (2, 3, 7, 64, 512))
def test_endpoint_inclusive_pixel_step_matches_linspace(samples):
    extent = 1.375
    coordinates = np.linspace(-extent, extent, samples)
    step = rectified_pixel_step(extent, samples)
    assert coordinates[0] == -extent
    assert coordinates[-1] == extent
    assert step == pytest.approx(float(np.diff(coordinates)[0]), rel=1e-14, abs=1e-14)
    assert step == pytest.approx(2.0 * extent / (samples - 1), rel=1e-14)


@pytest.mark.parametrize("extent,samples", ((0.0, 2), (1.0, 1), (float("inf"), 4),
                                               (True, 4), (1.0, True), (1.0, 2.5)))
def test_endpoint_inclusive_pixel_step_rejects_invalid_dimensions(extent, samples):
    with pytest.raises(ValueError):
        rectified_pixel_step(extent, samples)


def test_plane_window_annotation_declares_the_four_value_contract():
    from mc_photogrammetry.reconstruction.lattice import plane_window

    assert get_type_hints(plane_window)["return"] == (tuple[float, float, float, float] | None)


@pytest.mark.parametrize("samples", (3, 17, 64, 512))
def test_period_hypotheses_use_exact_endpoint_spacing(samples):
    period_pixels = 5.25
    extent = 1.375
    base = PeriodEstimate(period_pixels=period_pixels, phase_pixels=0.0,
                          strength=0.8)
    found = period_hypotheses(base, extent=extent, samples=samples)
    expected_distance = 1.0 / (period_pixels * rectified_pixel_step(extent, samples))
    assert found.best.plane_distance == pytest.approx(expected_distance, rel=1e-14)


def test_the_plane_basis_is_orthonormal_and_excludes_its_own_normal():
    axes = axis_directions(23.0, -14.0)
    for axis in range(3):
        normal, first, second = plane_basis(axis, axes)
        assert abs(float(normal @ first)) < 1e-12
        assert abs(float(normal @ second)) < 1e-12
        assert abs(float(first @ second)) < 1e-12


def test_an_unknown_axis_is_refused():
    with pytest.raises(ValueError):
        plane_basis(3, axis_directions(0.0, 0.0))


def test_rectifying_a_plane_recovers_the_grid_that_was_drawn_on_it():
    """End to end: draw a known grid on a known plane, then measure it back.

    The wall is rendered by inverse mapping - every image pixel is traced to the
    plane and shaded by its distance to the nearest world grid line - so the
    picture is exact and carries none of the aliasing that scattering points
    along the lines would introduce.
    """
    pytest.importorskip("cv2")
    width, height = 800, 800
    focal, distance = 500.0, 6.0
    axes = axis_directions(0.0, 0.0)
    normal, first, second = plane_basis(2, axes)

    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    rays = np.stack([(xs - width / 2.0) / focal,
                     -(ys - height / 2.0) / focal,
                     np.ones_like(xs, dtype=float)], axis=2)
    # The plane is normal . X = distance; scale each ray until it lands there.
    denominator = rays @ normal
    valid = denominator > 1e-6
    scale = np.where(valid, distance / np.where(valid, denominator, 1.0), 0.0)
    points = rays * scale[..., None]
    u, v = points @ first, points @ second
    near = lambda t: np.minimum(t % 1.0, 1.0 - (t % 1.0))
    image = np.where(valid & ((near(u) < 0.04) | (near(v) < 0.04)), 1.0, 0.0)
    image = image.astype(np.float32)

    patch = rectify_plane(image, 2, axes, focal, extent=1.0, samples=512)
    assert patch is not None
    found = estimate_period(patch, along=0)
    assert found is not None, "no grid recovered from the rectified plane"
    enriched = period_hypotheses(found, extent=1.0, samples=512)
    assert enriched.best.plane_distance == pytest.approx(distance, rel=0.1)


def test_a_plane_behind_the_camera_rectifies_to_nothing():
    axes = axis_directions(180.0, 0.0)
    image = np.zeros((64, 64), np.float32)
    # World +Z now points away from the camera, so its plane is entirely behind.
    assert rectify_plane(image, 2, axes, 100.0, samples=32) is None


def _texelled(size, texel_period):
    """A real texture's profile: strong texel edges, block seams a shade stronger.

    Inside a block the texture's own detail gives high-contrast edges every
    texel. A block boundary is only where one texture ends and the next begins,
    so it stands out a little but does not dominate - which is why the texel
    spacing is the period actually measured on close-range walls.
    """
    coordinate = np.arange(size)
    fine = ((coordinate % texel_period) < 1.0).astype(np.float32)
    coarse = ((coordinate % (texel_period * TEXELS_PER_BLOCK)) < 1.0).astype(np.float32)
    return np.repeat((fine + 0.5 * coarse)[None, :], size, axis=0)


def test_a_peak_at_sixteen_times_the_period_identifies_texels():
    """The one thing that separates texels from block seams in a single image.

    If the fine spacing is texels, block seams sit sixteen of them apart and
    leave their own correlation peak. Reading the fine spacing as blocks would
    put the surface sixteen times too far away.
    """
    found = estimate_period(_texelled(1024, 4.0))
    assert found is not None
    assert found.period_pixels == pytest.approx(4.0, rel=0.05)
    assert found.coarse_support > 0.5, "the block-scale seam left no peak"
    ranked = period_hypotheses(found, extent=1.0, samples=1024)
    assert ranked.best.interpretation.startswith("one texel")


def test_plain_seams_with_nothing_at_sixteen_times_read_as_blocks():
    found = estimate_period(_striped(1024, 40.0))
    assert found is not None
    assert found.coarse_support < 0.5
    ranked = period_hypotheses(found, extent=1.0, samples=1024)
    assert ranked.best.interpretation.startswith("one block")


def test_both_readings_survive_whichever_is_ranked_first():
    # The milestone requires the alternative to be retained, not discarded.
    for patch in (_texelled(1024, 4.0), _striped(1024, 40.0)):
        ranked = period_hypotheses(estimate_period(patch), extent=1.0, samples=1024)
        kinds = {h.interpretation.split()[1] for h in ranked.hypotheses}
        assert kinds == {"block", "texel"}


def test_the_patch_is_sampled_finely_enough_for_texels_by_default():
    """Too coarse a patch does not lose precision, it returns a wrong answer.

    A texel eight blocks away lands under `MINIMUM_PERIOD` at 512 samples, so
    the estimator skips it for the first harmonic and reports half the distance.
    The default sampling follows the source resolution to keep that from
    happening silently.
    """
    from mc_photogrammetry.reconstruction.lattice import OVERSAMPLE, sampling_for

    assert sampling_for(1.0, 479.3) == pytest.approx(2 * 479.3 * OVERSAMPLE, abs=1)
    # Bounded at both ends so a degenerate focal cannot ask for a huge buffer.
    assert sampling_for(1.0, 1e-6) == 64
    assert sampling_for(1000.0, 5000.0) == 8192


def test_the_default_sampling_is_what_rectify_uses():
    from mc_photogrammetry.reconstruction.lattice import sampling_for

    pytest.importorskip("cv2")
    axes = axis_directions(0.0, 0.0)
    image = np.zeros((200, 200), np.float32)
    patch = rectify_plane(image, 2, axes, 120.0, extent=0.5)
    assert patch is not None
    assert patch.shape[0] == sampling_for(0.5, 120.0)


def test_two_in_plane_readings_are_reconciled_to_the_fundamental():
    """A plane has one spacing, so an integer disagreement is a harmonic lock.

    Measured on Template 2, one in-plane direction returns the true period and
    the other exactly twice it, with the roles swapping between views of the
    same floor. The shorter reading is the right one, because a harmonic is a
    multiple of the fundamental and never a fraction.
    """
    from mc_photogrammetry.reconstruction.lattice import plane_period

    size = 1024
    across = _striped(size, 8.0)              # varies along the columns
    down = _striped(size, 16.0, along=1).T    # varies along the rows, twice as coarse
    patch = np.maximum(across, down.T if down.shape != across.shape else down)
    # Build it explicitly instead: columns every 8, rows every 16.
    columns = ((np.arange(size) % 8) < 1.0).astype(np.float32)
    rows = ((np.arange(size) % 16) < 1.0).astype(np.float32)
    patch = np.maximum(np.repeat(columns[None, :], size, axis=0),
                       np.repeat(rows[:, None], size, axis=1))
    found = plane_period(patch)
    assert found is not None
    assert found.period_pixels == pytest.approx(8.0, rel=0.05)


def test_readings_that_are_not_harmonics_keep_the_stronger_one():
    """8 and 13 are not an integer multiple, so neither is the other's harmonic.

    The correction must not fire there - picking the shorter would be a guess -
    so the cleaner signal wins instead. The noisy direction here has a genuinely
    weaker autocorrelation peak, which is what `strength` measures.
    """
    from mc_photogrammetry.reconstruction.lattice import plane_period

    size, rng = 1024, np.random.default_rng(5)
    columns = ((np.arange(size) % 13) < 1.0).astype(np.float32)
    rows = ((np.arange(size) % 8) < 1.0).astype(np.float32)
    patch = np.repeat(columns[None, :], size, axis=0)
    # The row-wise signal is buried in noise, so its period is barely there.
    patch = patch + 0.25 * np.repeat(rows[:, None], size, axis=1)
    patch = patch + rng.normal(0.0, 0.6, size=(size, size)).astype(np.float32)

    found = plane_period(patch)
    assert found is not None
    assert found.period_pixels == pytest.approx(13.0, rel=0.05)


def test_the_window_follows_the_masked_region_rather_than_the_plane_centre():
    """A proposed region is rarely at the plane's centre.

    Rectifying a fixed window about the centre either misses it or gives it a
    few pixels in a corner, which is why every region proposed on Template 2
    first came back with no period at all.
    """
    from mc_photogrammetry.reconstruction.lattice import plane_window

    axes = axis_directions(0.0, 0.0)

    def window_for(rows, columns):
        mask = np.zeros((400, 400), np.float32)
        mask[rows[0]:rows[1], columns[0]:columns[1]] = 1.0
        found = plane_window(mask, 2, axes, 250.0)
        assert found is not None
        return found

    middle = window_for((180, 220), (180, 220))
    left = window_for((180, 220), (40, 100))
    right = window_for((180, 220), (300, 360))
    top = window_for((40, 100), (180, 220))

    # Signs are left to the projection's own handedness - in Minecraft world +X
    # falls on screen left - so what is asserted is that the window moves with
    # the region and in opposite directions for opposite sides.
    assert abs(middle[0]) < 0.05 and abs(middle[1]) < 0.05
    assert left[0] * right[0] < 0, "opposite sides gave the same-signed window"
    assert abs(left[0]) > 0.1 and abs(right[0]) > 0.1
    assert abs(top[1]) > 0.1, "the window did not follow the region vertically"
    assert 0.0 < middle[2] < 1.0
    assert middle[3] in (1.0, -1.0)


def test_an_empty_mask_has_no_window():
    from mc_photogrammetry.reconstruction.lattice import plane_window

    axes = axis_directions(0.0, 0.0)
    assert plane_window(np.zeros((50, 50), np.float32), 2, axes, 100.0) is None


def test_plane_window_and_coordinates_share_normalized_incidence_and_keep_mapping_values():
    from dataclasses import fields
    from mc_photogrammetry.reconstruction.lattice import (
        GRAZING_INCIDENCE, PlaneRayAbstention, PlaneRayMap, _map_rays_to_unit_plane,
        plane_coordinates, plane_window,
    )

    # The estimate remains a public period/phase/support record; winning profile
    # direction is private measurement context and does not change its fields.
    assert [field.name for field in fields(type(estimate_period(_striped(64, 8))))] == [
        "period_pixels", "phase_pixels", "strength", "block_supports",
        "coarse_support", "hypotheses",
    ]

    matrix = np.asarray(((1.0, 0.02, 0.0), (0.0, 1.0, 0.01), (0.0, 0.0, 1.0)))
    points = np.asarray(((30.0, 35.0), (70.0, 65.0)))
    result = _map_rays_to_unit_plane(points, 2, matrix, 100.0, (100, 100), None,
                                     require_all=True)
    assert isinstance(result, PlaneRayMap)
    assert np.all(result.incidence[result.stable] > GRAZING_INCIDENCE)
    pixels = np.atleast_2d(points).astype(float)
    rays = np.stack(((pixels[:, 0] - 50.0) / 100.0,
                     -(pixels[:, 1] - 50.0) / 100.0,
                     np.ones(len(points))), axis=1)
    projection = rays @ matrix[2]
    side = 1.0 if np.count_nonzero(projection > 0) >= np.count_nonzero(projection < 0) else -1.0
    old_points = rays / (projection * side)[:, None]
    expected = np.stack((old_points @ matrix[0], old_points @ matrix[1]), axis=1)
    np.testing.assert_allclose(result.coordinates, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(plane_coordinates(
        points, 2, matrix, 100.0, (100, 100)), expected, rtol=0.0, atol=1e-12)

    fronto = np.zeros((100, 100), dtype=np.float32)
    fronto[30:70, 30:70] = 1.0
    window = plane_window(fronto, 2, np.eye(3), 100.0)
    assert window is not None and window[3] == 1.0


def test_near_grazing_window_and_vanishing_line_constraints_abstain_consistently():
    from mc_photogrammetry.reconstruction.lattice import (
        PlaneRayAbstention, PlaneRayMap, _map_rays_to_unit_plane,
        plane_coordinates, plane_window,
    )

    # Axis zero is horizontal in this camera. A narrow stripe at the image
    # centre carries only near-parallel rays; spanning the vanishing line gives
    # a window on the dominant side but cannot define a two-point constraint.
    narrow = np.zeros((100, 100), dtype=np.float32)
    narrow[20:80, 49:52] = 1.0
    assert plane_window(narrow, 0, np.eye(3), 100.0) is None
    near = _map_rays_to_unit_plane(((49.0, 50.0), (51.0, 50.0)), 0,
                                   np.eye(3), 100.0, (100, 100), None,
                                   require_all=True)
    assert isinstance(near, PlaneRayAbstention)
    assert plane_coordinates(((49.0, 50.0), (51.0, 50.0)), 0,
                             np.eye(3), 100.0, (100, 100)) is None

    crossing = np.zeros((20, 100), dtype=np.float32)
    crossing[:, :] = 1.0
    crossing_window = plane_window(crossing, 0, np.eye(3), 100.0,
                                   principal_point=(50.0, 50.0))
    assert crossing_window is not None
    mapped = _map_rays_to_unit_plane(
        np.column_stack((np.arange(100, dtype=float), np.full(100, 10.0))),
        0, np.eye(3), 100.0, (100, 100), (50.0, 50.0), require_all=False)
    assert isinstance(mapped, PlaneRayMap)
    assert np.all(mapped.incidence[mapped.stable] * mapped.side > 0.05)
    assert plane_coordinates(((10.0, 10.0), (90.0, 10.0)), 0,
                             np.eye(3), 100.0, (100, 100),
                             principal_point=(50.0, 50.0)) is None


def test_anisotropic_period_winner_carries_private_direction_without_public_shape_change():
    from mc_photogrammetry.reconstruction.lattice import _plane_period_with_direction

    height = width = 128
    x = (((np.arange(width) % 8) < 1).astype(np.float32) * 80.0)
    y = (((np.arange(height) % 13) < 1).astype(np.float32) * 45.0)
    horizontal = y[:, None] + x[None, :]
    x_wide_period = (((np.arange(width) % 13) < 1).astype(np.float32) * 45.0)
    y_narrow_period = (((np.arange(height) % 8) < 1).astype(np.float32) * 80.0)
    vertical = y_narrow_period[:, None] + x_wide_period[None, :]
    first = _plane_period_with_direction(horizontal)
    second = _plane_period_with_direction(vertical)
    assert first is not None and second is not None
    first_estimate, first_direction = first
    second_estimate, second_direction = second
    assert first_direction == 0
    assert second_direction == 1
    assert first_estimate.period_pixels == pytest.approx(estimate_period(horizontal, 0).period_pixels)
    assert second_estimate.period_pixels == pytest.approx(estimate_period(vertical, 1).period_pixels)


def test_measure_plane_reports_the_extent_it_actually_used():
    """The caller must convert the period at the scale it was measured at.

    `measure_plane` may replace the requested extent with the region's own
    half-span, so it returns the extent alongside the sampling; using the
    requested one instead scales every distance by their ratio.
    """
    pytest.importorskip("cv2")
    axes = axis_directions(0.0, 0.0)
    image = np.zeros((300, 300), np.float32)
    image[:, ::9] = 1.0
    mask = np.zeros((300, 300), np.float32)
    mask[100:200, 100:200] = 1.0
    found, samples, extent = measure_plane(image, 2, axes, 200.0, mask=mask)
    assert samples > 0
    assert extent != 1.0, "the extent should have become the region's own span"
    assert extent > 0.0


def test_a_stated_length_gives_the_distance_to_the_plane():
    """One image fixes a plane's shape and never its size.

    A wall twice as far away with blocks twice as big is the same picture, so
    distance has to come from something outside the image. A length the user
    states is the cheapest such thing: no correspondence, no second view, and
    nothing that Minecraft's repetition can defeat.
    """
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length

    width, height, focal = 800, 800, 500.0
    axes = axis_directions(0.0, 0.0)
    normal, first, second = plane_basis(2, axes)

    # Two points four blocks apart on a wall six blocks away, projected by hand.
    distance, apart = 6.0, 4.0
    a = normal * distance
    b = normal * distance + first * apart
    def pixel(point):
        return (width / 2.0 + point[0] * focal / point[2],
                height / 2.0 - point[1] * focal / point[2])

    found = scale_from_known_length(pixel(a), pixel(b), apart, 2, axes, focal,
                                    (width, height))
    assert found.resolved and found.status == "resolved"
    assert found.distance == pytest.approx(distance, rel=1e-6)


def test_the_stated_length_scales_the_answer_linearly():
    # Saying two points are twice as far apart puts the wall twice as far away.
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length

    axes = axis_directions(0.0, 0.0)
    args = ((520.0, 400.0), (620.0, 400.0), 2, axes, 500.0, (800, 800))
    one = scale_from_known_length(args[0], args[1], 3.0, *args[2:])
    two = scale_from_known_length(args[0], args[1], 6.0, *args[2:])
    assert one.resolved and two.resolved
    assert two.distance == pytest.approx(2.0 * one.distance, rel=1e-9)


def test_a_nonsense_length_is_refused_rather_than_scaled():
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length

    axes = axis_directions(0.0, 0.0)
    zero = scale_from_known_length((400.0, 400.0), (500.0, 400.0), 0.0, 2, axes,
                                   500.0, (800, 800))
    assert not zero.resolved and zero.status == "invalid_length"
    # The same point twice states nothing about scale.
    same = scale_from_known_length((400.0, 400.0), (400.0, 400.0), 3.0, 2, axes,
                                   500.0, (800, 800))
    assert not same.resolved and same.status == "ill_conditioned"


def test_a_known_distance_settles_the_block_versus_texel_reading():
    """The ambiguity this module has carried since it was written.

    Both readings are consistent with one image and differ by sixteen. A stated
    length picks the true one, and the loser is kept so the ambiguity stays in
    the record rather than being quietly deleted.
    """
    from mc_photogrammetry.reconstruction.lattice import resolve_hypotheses

    ranked = period_hypotheses(estimate_period(_striped(512, 32.0)),
                               extent=2.0, samples=512)
    block, texel = ranked.hypotheses[0], ranked.hypotheses[1]
    assert block.plane_distance != pytest.approx(texel.plane_distance)

    settled = resolve_hypotheses(ranked, texel.plane_distance)
    assert settled.best.interpretation == texel.interpretation
    assert len(settled.hypotheses) == 2, "the alternative must be retained"

    other = resolve_hypotheses(ranked, block.plane_distance)
    assert other.best.interpretation == block.interpretation


def test_a_distance_agreeing_with_neither_reading_changes_no_ranking():
    # A length that supports neither hypothesis is not evidence for the closer
    # one; the ordering is left exactly as the image left it.
    from mc_photogrammetry.reconstruction.lattice import resolve_hypotheses

    ranked = period_hypotheses(estimate_period(_striped(512, 32.0)),
                               extent=2.0, samples=512)
    settled = resolve_hypotheses(ranked, 10_000.0)
    assert [h.interpretation for h in settled.hypotheses] == \
           [h.interpretation for h in ranked.hypotheses]


def test_a_constraint_that_cannot_exist_carries_status_and_no_distance():
    """The lead's reproduction: two points on opposite sides of a vanishing line.

    Their rays meet the plane on opposite sides of the eye, so no single plane in
    front of the camera contains both. This previously returned 0.2 - a plausible
    number nothing downstream could tell from a measured one.
    """
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length

    axes = axis_directions(0.0, 0.0)
    found = scale_from_known_length((300.0, 400.0), (500.0, 400.0), 2.0, 0, axes,
                                    500.0, (800, 800))
    assert not found.resolved
    assert found.distance is None
    assert found.status == "inconsistent"
    assert "opposite sides" in found.detail


def test_a_grazing_ray_is_refused_rather_than_extrapolated():
    # Nearly parallel to the plane, so the intersection races to infinity and a
    # pixel of pointing error moves it without bound.
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length

    axes = axis_directions(0.0, 0.0)
    found = scale_from_known_length((401.0, 400.0), (403.0, 400.0), 2.0, 0, axes,
                                    500.0, (800, 800))
    assert not found.resolved
    assert found.status in {"inconsistent", "ill_conditioned"}


def test_non_finite_input_is_refused():
    from mc_photogrammetry.reconstruction.lattice import scale_from_known_length

    axes = axis_directions(0.0, 0.0)
    for bad in ((float("nan"), 400.0), (float("inf"), 400.0)):
        found = scale_from_known_length(bad, (500.0, 400.0), 2.0, 2, axes,
                                        500.0, (800, 800))
        assert not found.resolved and found.status in {"non_finite", "inconsistent"}
    assert not scale_from_known_length((400.0, 400.0), (500.0, 400.0),
                                       float("nan"), 2, axes, 500.0,
                                       (800, 800)).resolved


def test_rectified_cell_mapping_inverts_rectify_planes_own_construction():
    """Round-trip a rectified-patch pixel back to a world-facing pixel through
    the new mapping functions, checked against `rectify_plane`'s own internal
    grid and pinhole formulas directly - not against a second copy of them.

    This is the "invertible rectification-to-image/3D mapping" the per-cell
    evaluation scripts need: given a cell in the rectified patch, where in the
    original photograph, and where in camera space at the plane's real
    distance, did that content actually come from.
    """
    axes = axis_directions(37.0, -12.0, 5.0)
    axis, side, distance = 1, -1.0, 6.3
    extent, samples = 2.0, 512
    centre = (0.3, -0.1)
    focal = 500.0
    principal_point = (960.0, 540.0)
    normal, first, second = plane_basis(axis, axes)

    grid_u = np.linspace(centre[0] - extent, centre[0] + extent, samples)
    grid_v = np.linspace(centre[1] - extent, centre[1] + extent, samples)

    rng = np.random.default_rng(3)
    for _ in range(40):
        row = int(rng.integers(0, samples))
        column = int(rng.integers(0, samples))
        u_true, v_true = grid_u[column], grid_v[row]

        point_unit = side * normal + u_true * first + v_true * second
        depth = point_unit[2]
        if depth <= 1e-6:
            continue
        px_true = principal_point[0] + point_unit[0] * focal / depth
        py_true = principal_point[1] - point_unit[1] * focal / depth

        u, v = rectified_pixel_to_plane_coordinates(
            (column, row), extent=extent, samples=samples, centre=centre)
        assert u == pytest.approx(u_true)
        assert v == pytest.approx(v_true)

        camera_point = plane_coordinates_to_camera_point(axis, axes, side, distance, u, v)
        px, py = camera_point_to_pixel(camera_point, focal, principal_point)
        assert px == pytest.approx(px_true, abs=1e-6)
        assert py == pytest.approx(py_true, abs=1e-6)


def test_camera_point_to_pixel_refuses_a_point_behind_the_camera():
    from mc_photogrammetry.reconstruction.lattice import camera_point_to_pixel
    assert camera_point_to_pixel(np.array([1.0, 0.0, -2.0]), 500.0, (960.0, 540.0)) is None
