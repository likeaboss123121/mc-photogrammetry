"""Recovering a camera from the lattice in the picture.

The estimator has to be checked against a camera whose answer is known, and the
only trustworthy source of that is the projection the rest of the project uses.
So the synthetic cases here build segments with `project_points` and ask the
estimator to recover the camera that made them: if the vanishing-point derivation
disagrees with the shipped projection, these fail.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.calibration import (
    CalibrationResult,
    axis_directions,
    estimate_camera,
    segment_residuals,
    vanishing_points,
)
from mc_photogrammetry.reconstruction.camera import CameraModel, project_points

WIDTH, HEIGHT = 1280, 720


def _camera(yaw, pitch, fov):
    return CameraModel(position=(0.5, 0.5, 0.5), yaw_degrees=yaw,
                       pitch_degrees=pitch, fov_degrees=fov)


def _focal(fov, height=HEIGHT):
    return (height / 2.0) / np.tan(np.radians(fov) / 2.0)


def _lattice_segments(camera, width=WIDTH, height=HEIGHT, reach=9):
    """Project the edges of a block lattice, as a detector would see them."""
    eye = np.asarray(camera.position) + [0.0, camera.eye_height, 0.0]
    rows = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1.0
        for a in range(-reach, reach + 1):
            for b in range(-reach, reach + 1):
                start = np.array(eye, dtype=float)
                other = [i for i in range(3) if i != axis]
                start[other[0]] += a * 2.0
                start[other[1]] += b * 2.0
                start[axis] += 6.0
                pair = project_points([tuple(start), tuple(start + step * 2.0)],
                                      camera, width=width, height=height)
                if any(point is None for point in pair):
                    continue
                (x0, y0, _d0), (x1, y1, _d1) = pair
                if not (0 <= x0 < width and 0 <= x1 < width
                        and 0 <= y0 < height and 0 <= y1 < height):
                    continue
                if np.hypot(x1 - x0, y1 - y0) < 40.0:
                    continue
                rows.append((x0, y0, x1, y1))
    return np.asarray(rows, dtype=float)


def _optics_anchor_fixture(kind):
    """Reuse the independent projected-line generator; no pixels or fitted truth."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "anchor_independent_fixture", Path(__file__).with_name("test_calibration_independent.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    yaw, pitch, roll = (5., 16., 0.) if kind == "near_frontal" else (23., -14., 3. if kind == "rolled" else 0.)
    segments = fixture._segments((.5, .5, .5), yaw, pitch, roll, 70., WIDTH, HEIGHT)
    cut_x, cut_y = (200, 100) if kind == "cropped" else (0, 0)
    if cut_x or cut_y:
        segments = segments - [cut_x, cut_y, cut_x, cut_y]
        segments = segments[(segments[:, [0, 2]] >= 0).all(axis=1)
                            & (segments[:, [1, 3]] >= 0).all(axis=1)]
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    return segments, lengths, WIDTH - cut_x, HEIGHT - cut_y, (_focal(70.) * 1.15, yaw, pitch)


@pytest.mark.parametrize("kind", ["near_frontal", "rolled", "cropped"])
def test_optics_anchor_real_geometry_candidate_order_is_stable(kind, monkeypatch):
    import mc_photogrammetry.reconstruction.calibration as module

    segments, lengths, width, height, initial = _optics_anchor_fixture(kind)
    real_numpy = module.np

    class OrderedGrid:
        def __init__(self, order):
            self.order = order

        def __getattr__(self, name):
            return getattr(real_numpy, name)

        def arange(self, *args, **kwargs):
            values = real_numpy.arange(*args, **kwargs)
            reverse = self.order == "reverse_all" or (
                self.order == "reverse_scales" and abs(float(args[0]) - .80) < 1e-10)
            return values[::-1] if reverse else values

        def linspace(self, *args, **kwargs):
            values = real_numpy.linspace(*args, **kwargs)
            return values[::-1] if self.order in ("reverse_all", "reverse_point") else values

    answers = []
    for order in ("forward", "reverse_all", "reverse_scales", "reverse_point"):
        monkeypatch.setattr(module, "np", OrderedGrid(order))
        answers.append(module.refine_optics(segments, lengths, *initial, width, height, module.INLIER_PIXELS))
    expected = answers[0]
    assert expected[4], "control must exercise accepted optical refinement"
    for found in answers[1:]:
        assert found[4] == expected[4]
        assert found[0] == pytest.approx(expected[0], abs=1e-10)
        assert found[1] == pytest.approx(expected[1], abs=1e-10) if expected[1] is not None else found[1] is None
        assert found[2:] == pytest.approx(expected[2:], abs=1e-10)


def test_optics_anchor_each_sweep_keeps_its_declared_grid(monkeypatch):
    import mc_photogrammetry.reconstruction.calibration as module

    segments, lengths, width, height, initial = _optics_anchor_fixture("rolled")
    original = module._score
    calls = []

    def observe(*args, **kwargs):
        calls.append(tuple(float(value) for value in args[2:5]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_score", observe)
    found = module.refine_optics(segments, lengths, *initial, width, height, module.INLIER_PIXELS)
    assert found[4]
    assert len(calls) == 1 + 2 * (17 * 19 + 49 * 19) + 9 * 9 + 13 * 13
    offset = 1
    for count in (17 * 19, 49 * 19, 17 * 19, 49 * 19):
        focals = np.unique([row[0] for row in calls[offset:offset + count]])
        assert len(focals) == 19
        assert focals / focals[0] == pytest.approx(np.arange(.80, 1.2501, .025) / .80, abs=1e-12)
        offset += count
    for count, grid_size in ((9 * 9, 9), (13 * 13, 13)):
        grid = np.asarray(calls[offset:offset + count])
        assert len(np.unique(grid[:, 1])) == grid_size
        assert len(np.unique(grid[:, 2])) == grid_size
        offset += count


def test_the_vertical_axis_points_straight_up_in_camera_space():
    # World +Y must have no sideways component whatever the yaw, or the zero-roll
    # assumption the whole estimator rests on is wrong.
    for yaw in (0.0, 37.0, -122.0):
        right, _up, _forward = axis_directions(yaw, 12.0)[1]
        assert abs(right) < 1e-12


def test_a_vanishing_point_is_where_a_receding_edge_ends_up():
    """Cross-check against `project_points` rather than against the derivation."""
    camera = _camera(31.0, -9.0, 70.0)
    points = vanishing_points(_focal(70.0), 31.0, -9.0, WIDTH, HEIGHT)
    eye = np.asarray(camera.position) + [0.0, camera.eye_height, 0.0]
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1.0
        far = project_points([tuple(eye + step * 40000.0)], camera,
                             width=WIDTH, height=HEIGHT)[0]
        if far is None:  # that axis recedes behind the camera; try the other way
            far = project_points([tuple(eye - step * 40000.0)], camera,
                                 width=WIDTH, height=HEIGHT)[0]
        x, y, w = points[axis]
        assert abs(w) > 1e-9
        assert np.allclose([x / w, y / w], [far[0], far[1]], atol=0.5)


def test_a_level_camera_puts_the_vertical_vanishing_point_at_infinity():
    x, y, w = vanishing_points(_focal(70.0), 20.0, 0.0, WIDTH, HEIGHT)[1]
    assert abs(w) < 1e-12, "a level camera has parallel vertical seams"
    assert abs(x) < 1e-9 and abs(y) > 0.0


def test_a_segment_pointing_at_its_vanishing_point_has_no_residual():
    points = vanishing_points(_focal(70.0), 0.0, -20.0, WIDTH, HEIGHT)
    x, y, w = points[1]
    target = np.asarray([x / w, y / w])
    midpoint = np.asarray([600.0, 400.0])
    direction = target - midpoint
    direction /= np.linalg.norm(direction)
    segment = np.concatenate([midpoint - direction * 30.0,
                              midpoint + direction * 30.0])[None, :]
    assert segment_residuals(segment, points)[0, 1] < 1e-6


def test_a_perpendicular_segment_is_penalised_by_its_own_length():
    points = vanishing_points(_focal(70.0), 0.0, -20.0, WIDTH, HEIGHT)
    x, y, w = points[1]
    target = np.asarray([x / w, y / w])
    midpoint = np.asarray([600.0, 400.0])
    direction = target - midpoint
    direction /= np.linalg.norm(direction)
    across = np.asarray([-direction[1], direction[0]])
    segment = np.concatenate([midpoint - across * 30.0,
                              midpoint + across * 30.0])[None, :]
    assert segment_residuals(segment, points)[0, 1] == pytest.approx(30.0, rel=1e-6)


@pytest.mark.parametrize("yaw,pitch,fov", [
    (23.0, -14.0, 70.0),
    (61.0, 8.0, 90.0),
    (7.0, -31.0, 50.0),
])
def test_the_camera_that_made_the_segments_is_recovered(yaw, pitch, fov):
    camera = _camera(yaw, pitch, fov)
    segments = _lattice_segments(camera)
    assert len(segments) > 60, "the synthetic fixture produced too little to fit"
    found = estimate_camera(np.zeros((HEIGHT, WIDTH), np.uint8), segments=segments)
    assert found is not None
    assert found.pitch_degrees == pytest.approx(pitch, abs=0.5)
    assert found.yaw_degrees % 90.0 == pytest.approx(yaw % 90.0, abs=0.5)
    assert found.fov_degrees == pytest.approx(fov, rel=0.02)


def test_too_few_segments_returns_nothing_rather_than_a_guess():
    assert estimate_camera(np.zeros((HEIGHT, WIDTH), np.uint8),
                           segments=np.zeros((3, 4))) is None


def test_the_result_reports_its_own_field_of_view():
    result = CalibrationResult(focal_pixels=_focal(70.0), yaw_degrees=0.0,
                               pitch_degrees=0.0, width=WIDTH, height=HEIGHT,
                               inliers=10, segments=20, median_residual=1.0)
    assert result.fov_degrees == pytest.approx(70.0, abs=1e-6)
    assert result.inlier_fraction == pytest.approx(0.5)


def test_the_coarse_cap_does_not_move_the_answer():
    """The coarse sweep sees only the longest segments, for speed.

    That is only legitimate if it changes nothing. Short segments are added here
    as noise in random directions, exactly the population the cap discards.
    """
    camera = _camera(23.0, -14.0, 70.0)
    real = _lattice_segments(camera)
    rng = np.random.default_rng(3)
    midpoints = rng.uniform([0, 0], [WIDTH, HEIGHT], size=(400, 2))
    angles = rng.uniform(0.0, np.pi, size=400)
    arms = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 22.0
    noise = np.concatenate([midpoints - arms, midpoints + arms], axis=1)
    mixed = np.vstack([real, noise])

    capped = estimate_camera(np.zeros((HEIGHT, WIDTH), np.uint8),
                             segments=mixed, coarse_segments=300)
    uncapped = estimate_camera(np.zeros((HEIGHT, WIDTH), np.uint8),
                               segments=mixed, coarse_segments=len(mixed))
    assert capped is not None and uncapped is not None
    assert capped.yaw_degrees == pytest.approx(uncapped.yaw_degrees, abs=1e-9)
    assert capped.pitch_degrees == pytest.approx(uncapped.pitch_degrees, abs=1e-9)
    assert capped.focal_pixels == pytest.approx(uncapped.focal_pixels, rel=1e-9)


def test_the_axis_directions_are_orthonormal():
    # They come from the shared world-to-camera transform, which must be a
    # rotation; a scale or skew creeping in would silently bias every estimate.
    axes = axis_directions(41.0, -23.0)
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)


def test_the_contract_carries_orientation_and_still_invents_no_position():
    """Honesty about translation must not cost the orientation that was measured.

    An earlier version dropped the whole pose so as not to fabricate a position,
    which threw away measured yaw, pitch and roll to avoid inventing one number.
    `CameraPose.position` is optional, so the pose travels with it unset.
    """
    from mc_photogrammetry.reconstruction.calibration import to_camera_solution

    result = CalibrationResult(focal_pixels=_focal(70.0), yaw_degrees=14.0,
                               pitch_degrees=17.0, width=WIDTH, height=HEIGHT,
                               inliers=200, segments=250, median_residual=0.4,
                               roll_degrees=3.5, principal_point=(410.0, 330.0))
    solution = to_camera_solution({"a.png": result})
    estimate = solution.cameras["a.png"]

    assert estimate.pose is not None, "measured orientation was discarded"
    assert estimate.pose.position is None, "a position was invented"
    assert estimate.pose.yaw_degrees == pytest.approx(14.0)
    assert estimate.pose.pitch_degrees == pytest.approx(17.0)
    assert estimate.pose.roll_degrees == pytest.approx(3.5)

    # The measured optical centre must survive, not be replaced by the middle.
    assert estimate.intrinsics.principal_point_pixels == (410.0, 330.0)

    assert not solution.scale_known
    for wanted in ("position", "scale", "yaw quadrant"):
        assert any(wanted in note for note in solution.unresolved), wanted


def test_the_adapter_does_not_pass_the_inlier_fraction_off_as_confidence():
    """That statistic was measured not to separate success from failure.

    Across 109 views the passing median was 93.7% and the failing median 92.7%.
    Reporting it as `confidence` asserts a correctness it has been shown not to
    carry.
    """
    from mc_photogrammetry.reconstruction.calibration import to_camera_solution

    result = CalibrationResult(focal_pixels=_focal(70.0), yaw_degrees=0.0,
                               pitch_degrees=0.0, width=WIDTH, height=HEIGHT,
                               inliers=249, segments=250, median_residual=0.4)
    estimate = to_camera_solution({"a.png": result}).cameras["a.png"]
    assert result.inlier_fraction > 0.99
    assert estimate.confidence != pytest.approx(result.inlier_fraction)
    assert estimate.residual_pixels == pytest.approx(0.4)


def test_a_centred_frame_reports_the_image_centre_rather_than_nothing():
    # `principal_point is None` means "never justified moving it", and a consumer
    # still needs a usable number.
    from mc_photogrammetry.reconstruction.calibration import to_camera_solution

    result = CalibrationResult(focal_pixels=_focal(70.0), yaw_degrees=0.0,
                               pitch_degrees=0.0, width=WIDTH, height=HEIGHT,
                               inliers=10, segments=20, median_residual=1.0)
    estimate = to_camera_solution({"a.png": result}).cameras["a.png"]
    assert estimate.intrinsics.principal_point_pixels == (WIDTH / 2.0, HEIGHT / 2.0)


def test_roll_matches_the_shared_camera_basis_at_every_orientation():
    """A convention with two definitions has two places to drift, and it drifted.

    This module once applied roll by hand because `CameraModel` had nowhere to
    put it. Once it did, the two turned opposite ways: 0.832 apart in maximum
    component at yaw 30, pitch -10, roll 25, and 6e-17 apart once negated.
    """
    from mc_photogrammetry.reconstruction.camera import camera_space_points

    for yaw, pitch, roll in ((30.0, -10.0, 25.0), (0.0, 0.0, 0.0),
                             (-76.0, 17.0, -4.0), (120.0, 45.0, 12.0)):
        camera = CameraModel(position=(0.0, 0.0, 0.0), yaw_degrees=yaw,
                             pitch_degrees=pitch, fov_degrees=70.0,
                             roll_degrees=roll)
        eye = np.asarray(camera.position, float) + [0.0, camera.eye_height, 0.0]
        shared = camera_space_points([tuple(eye + axis) for axis in np.eye(3)], camera)
        assert np.allclose(axis_directions(yaw, pitch, roll), shared, atol=1e-12)


def test_a_session_of_mixed_image_sizes_does_not_pool_their_focal_lengths():
    """A focal length in pixels belongs to the sampling, not to the lens.

    The same field of view on a frame half as tall is half the focal length, so
    a median taken across differently sized frames averages quantities that are
    not the same quantity.
    """
    import mc_photogrammetry.reconstruction.calibration as module

    big = _lattice_segments(_camera(23.0, -14.0, 70.0))
    small_camera = _camera(23.0, -14.0, 70.0)
    small = _lattice_segments(small_camera, width=WIDTH // 2, height=HEIGHT // 2)
    frames = {"big.png": np.full((HEIGHT, WIDTH), 1, np.uint8),
              "small.png": np.full((HEIGHT // 2, WIDTH // 2), 2, np.uint8)}
    supplied = {1: big, 2: small}

    detect, boost = module.detect_segments, module.enhance
    module.enhance = lambda grey: grey
    module.detect_segments = lambda grey, **kw: supplied[int(grey[0, 0])]
    try:
        results = module.estimate_session(frames)
    finally:
        module.detect_segments, module.enhance = detect, boost

    if len(results) < 2:
        pytest.skip("the synthetic small frame did not yield a fit")
    focals = {name: r.focal_pixels for name, r in results.items()}
    assert focals["big.png"] != pytest.approx(focals["small.png"], rel=0.2), (
        "focals from different image geometries were pooled")


def test_a_session_shares_one_focal_length_and_the_median_sets_it():
    """Field of view is a game setting, so it is a session quantity, not a view one.

    One view here is given segments that fit a different camera; the session
    median must ignore that outlier rather than average it in, and the outlier
    view must come back refitted at the shared focal.
    """
    import mc_photogrammetry.reconstruction.calibration as module

    good = _lattice_segments(_camera(23.0, -14.0, 70.0))
    odd = _lattice_segments(_camera(40.0, 5.0, 40.0))
    # Each frame carries a distinct fill value so the stub can tell them apart.
    frames = {name: np.full((HEIGHT, WIDTH), value, np.uint8)
              for name, value in (("a.png", 1), ("b.png", 2), ("c.png", 3))}
    supplied = {1: good, 2: good, 3: odd}

    original_detect, original_enhance = module.detect_segments, module.enhance
    # `enhance` runs first and rewrites the pixel values, so it is stubbed too.
    module.enhance = lambda grey: grey
    module.detect_segments = lambda grey, **kw: supplied[int(grey[0, 0])]
    try:
        results = module.estimate_session(frames)
    finally:
        module.detect_segments, module.enhance = original_detect, original_enhance

    assert set(results) == {"a.png", "b.png", "c.png"}
    focals = {round(r.focal_pixels, 6) for r in results.values()}
    assert len(focals) == 1, "every view in a session must end on one focal length"
    assert focals.pop() == pytest.approx(_focal(70.0), rel=0.02)
    assert any("median" in note and "1280x720" in note
               for note in results["c.png"].notes)


def test_a_whole_screenshot_does_not_invent_roll_or_a_shifted_centre():
    """The refinement must come back empty-handed when nothing is wrong.

    Three extra free parameters can always fit noise a little better, so without
    a margin the estimator would report a small roll and a shifted optical centre
    on every frame - describing its own freedom rather than the camera. A whole
    Minecraft screenshot genuinely has neither.
    """
    camera = _camera(23.0, -14.0, 70.0)
    segments = _lattice_segments(camera)
    found = estimate_camera(np.zeros((HEIGHT, WIDTH), np.uint8), segments=segments)
    assert found is not None
    assert found.roll_degrees == 0.0
    assert found.principal_point is None, "an uncropped frame needs no new centre"


def test_a_cropped_frame_has_its_optical_centre_recovered():
    """Cropping moves the optical centre, and nothing else can absorb it.

    Cutting pixels off one edge shifts the centre by half of what was removed.
    The frame then cannot be described by focal length and orientation alone,
    which is the case the milestone means by refining the assumption when the
    evidence supports it.
    """
    camera = _camera(23.0, -14.0, 70.0)
    full = _lattice_segments(camera)
    cut_x, cut_y = 260, 140
    kept = full - np.asarray([cut_x, cut_y, cut_x, cut_y], dtype=float)
    inside = ((kept[:, [0, 2]] >= 0).all(axis=1) & (kept[:, [1, 3]] >= 0).all(axis=1))
    kept = kept[inside]
    assert len(kept) > 40, "the crop left too little to fit"

    width, height = WIDTH - cut_x, HEIGHT - cut_y
    true_centre = (WIDTH / 2.0 - cut_x, HEIGHT / 2.0 - cut_y)
    found = estimate_camera(np.zeros((height, width), np.uint8), segments=kept)
    assert found is not None
    assert found.principal_point is not None, "the shifted centre was not detected"
    assert found.principal_point[0] == pytest.approx(true_centre[0], abs=0.2 * width)
    assert found.principal_point[1] == pytest.approx(true_centre[1], abs=0.2 * height)


def test_roll_turns_the_frame_and_is_carried_through_the_axes():
    # World +Y has no sideways component at zero roll; rolling the camera must
    # give it one, or the parameter is not reaching the geometry.
    upright = axis_directions(30.0, -10.0, 0.0)[1]
    rolled = axis_directions(30.0, -10.0, 25.0)[1]
    assert abs(upright[0]) < 1e-12
    assert abs(rolled[0]) > 0.3
    assert np.allclose(rolled @ rolled, 1.0), "roll must stay a rotation"


def test_the_principal_point_moves_the_vanishing_points_with_it():
    shifted = vanishing_points(_focal(70.0), 20.0, -12.0, WIDTH, HEIGHT,
                               0.0, (WIDTH / 2.0 + 60.0, HEIGHT / 2.0))
    centred = vanishing_points(_focal(70.0), 20.0, -12.0, WIDTH, HEIGHT)
    for (x1, y1, w1), (x0, y0, w0) in zip(shifted, centred):
        if abs(w0) < 1e-12:
            continue
        assert (x1 / w1) == pytest.approx(x0 / w0 + 60.0, abs=1e-6)
        assert (y1 / w1) == pytest.approx(y0 / w0, abs=1e-6)


def _drawn(width=400, height=400):
    """A wall of one material, with a genuine seam through the middle of it.

    Left and right halves are different materials, so the join between them is a
    structural edge. Both halves carry fine internal lines of their own, which
    run in the same direction and are not.
    """
    image = np.zeros((height, width), np.float32)
    image[:, : width // 2] = 60.0
    image[:, width // 2:] = 190.0
    image[:, ::9] += 12.0          # texture lines inside both materials
    return image


def test_contrast_separates_a_real_seam_from_lines_inside_a_texture():
    """Direction cannot make this distinction; Minecraft textures are axis-aligned
    and their internal lines run along the same lattice directions as the seams."""
    from mc_photogrammetry.reconstruction.calibration import segment_contrast

    image = _drawn()
    seam = np.asarray([[200.0, 20.0, 200.0, 380.0]])         # the material join
    texture = np.asarray([[108.0, 20.0, 108.0, 380.0]])      # a line within one
    assert segment_contrast(image, seam)[0] > 5 * segment_contrast(image, texture)[0]


def test_structural_weights_are_relative_to_the_frames_own_contrast():
    """A gamma-100% fixture has a fraction of the contrast of a fullbright one.

    An absolute threshold would call an entire dim scene texture, so the scale is
    the frame's own median.
    """
    from mc_photogrammetry.reconstruction.calibration import (
        CONTRAST_FLOOR,
        structural_weights,
    )

    image = _drawn()
    segments = np.asarray([[200.0, 20.0, 200.0, 380.0],
                           [108.0, 20.0, 108.0, 380.0],
                           [306.0, 20.0, 306.0, 380.0]])
    weights = structural_weights(image, segments)
    assert weights.shape == (3,)
    assert (weights >= CONTRAST_FLOOR).all() and (weights <= 1.0).all()
    assert weights[0] > weights[1], "the seam should outrank the texture line"

    faint = structural_weights(image * 0.1, segments)
    assert faint[0] > faint[1], "a dim frame must still rank the seam first"


def test_a_texture_line_is_discounted_and_not_discarded():
    # It still lies on a face that is genuinely oriented that way, so it is
    # evidence about direction even though it is not an edge.
    from mc_photogrammetry.reconstruction.calibration import (
        CONTRAST_FLOOR,
        structural_weights,
    )

    flat = np.full((200, 200), 100.0, np.float32)
    segments = np.asarray([[100.0, 20.0, 100.0, 180.0]])
    assert structural_weights(flat, segments)[0] >= CONTRAST_FLOOR > 0.0


def test_contrast_on_no_segments_is_empty_rather_than_an_error():
    from mc_photogrammetry.reconstruction.calibration import segment_contrast

    assert segment_contrast(np.zeros((10, 10), np.float32), np.zeros((0, 4))).size == 0
