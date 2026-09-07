"""Measuring the block grid on a plane once the camera orientation is known.

The distance a period implies is only as good as the period, so these check the
measurement against signals whose spacing is set by construction, then check the
rectification against a plane placed by construction.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.calibration import axis_directions
from mc_photogrammetry.reconstruction.lattice import (
    TEXELS_PER_BLOCK,
    estimate_period,
    period_hypotheses,
    plane_basis,
    rectify_plane,
)


def _striped(size, period, phase=0.0, along=0):
    """A patch of narrow bright seams at a known spacing."""
    coordinate = np.arange(size)
    line = ((coordinate - phase) % period < 1.0).astype(np.float32)
    field = np.repeat(line[None, :], size, axis=0)
    return field if along == 0 else field.T


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
    # One output pixel spans 2*extent/samples = 4/512 world units at unit
    # distance, so a 32-pixel period is 0.25 world units, and reading that as one
    # block puts the plane four units away.
    base = estimate_period(_striped(512, 32.0))
    enriched = period_hypotheses(base, extent=2.0, samples=512)
    assert enriched.best.plane_distance == pytest.approx(4.0, rel=0.03)


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
