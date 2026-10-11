"""Recover a camera's orientation and focal length from the build in the picture.

Every fixture ships a `cameras.json` read off the F3 overlay, and that is
evaluation scaffolding: in production nobody has the debug screen, because the
person reconstructing a build is not the person who photographed it. So pose has
to come out of the pixels.

Minecraft makes this unusually tractable. The world is a lattice aligned to three
orthogonal axes, so almost every edge in frame runs along one of them, and the
three directions converge on three vanishing points. Their geometry fixes the
focal length and the orientation - the classical Manhattan-world calibration.

Two properties of the game narrow it further:

* **The camera never rolls.** Orientation is yaw and pitch alone.
* **The vertical axis is vertical.** With zero roll and a centred principal point
  the vertical vanishing point sits on the image's centre column, and the two
  horizontal ones share a horizon row.

This does not fit three free vanishing points and then hope they are orthogonal.
It searches `(focal, yaw, pitch)` directly and derives the three vanishing points
from that, so every hypothesis considered is a physically consistent camera and
the orthogonality constraint cannot be violated by construction.

What it cannot do from one view is name the axes. Yaw is recoverable only modulo
90 degrees, because the two horizontal lattice directions are interchangeable
without semantic cues, and no image evidence distinguishes north from east.
`CalibrationResult.yaw_degrees` is therefore reported on `[0, 90)` and the caller
is responsible for resolving the quadrant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

MINIMUM_LENGTH = 40.0
"""Shortest segment worth trusting, in pixels at full resolution.

Minecraft's F3 overlay is text about ten pixels tall, and its strokes are the
main non-structural lines in frame. Requiring forty removes essentially all of
them without touching block seams, which run the height of a wall.
"""

INLIER_PIXELS = 3.0
"""Endpoint displacement, in pixels, within which a segment supports a direction.

The residual is expressed as how far a segment's endpoints sit off the line from
its midpoint to the vanishing point, so it is directly comparable to the seam
displacement the milestone is graded on.
"""

DETECTION_SCALES = (1.0, 0.5)
"""Resolutions to detect at. Coarse passes recover seams that fine ones fragment."""

COARSE_SEGMENTS = 300
"""How many of the longest segments the coarse sweep considers.

The exhaustive stage dominates the runtime and does not need every segment: on
the M3 frame, fitting to the 306 segments over 120px returned the same camera as
fitting to all 2151. Long edges pin a vanishing point; short ones mostly vote for
whichever direction they happen to lie near.
"""

REFINE_MARGIN = 0.02
"""Fractional score gain required before roll or an off-centre principal point is believed.

Three extra parameters can always fit noise a little better, so without a margin
the refinement would always "find" a small roll and a shifted centre and would be
reporting its own freedom rather than the camera. A whole screenshot genuinely has
neither, and the refinement must come back empty-handed on one.
"""

ROLL_LIMIT = 4.0
"""Degrees of roll searched either side of level.

Minecraft's camera cannot roll at all. This exists for frames that have been
rotated or cropped after capture, which the image scope admits, so the range is
deliberately narrow - a wide one would mostly offer new ways to alias with yaw.
"""

PRINCIPAL_LIMIT = 0.12
"""How far the principal point may move, as a fraction of width and height.

Cropping is what moves it: cutting pixels off one edge shifts the optical centre
by half of what was removed. A frame cropped harder than this is better handled
by being told about the crop.
"""

GAMMA = 0.45
CLAHE_CLIP = 3.0
CLAHE_GRID = 16


@dataclass(frozen=True)
class CalibrationResult:
    """An estimated camera, with the evidence that produced it."""

    focal_pixels: float
    yaw_degrees: float
    pitch_degrees: float
    width: int
    height: int
    inliers: int
    segments: int
    median_residual: float
    per_axis_inliers: tuple[int, int, int] = (0, 0, 0)
    roll_degrees: float = 0.0
    principal_point: tuple[float, float] | None = None
    """Optical centre, or ``None`` where the image centre was not improved upon.

    ``None`` is the honest value for a whole screenshot and is not the same as
    "the centre was measured and happens to be the middle": it records that the
    evidence never justified moving it.
    """

    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fov_degrees(self) -> float:
        """Vertical field of view, which is the quantity Minecraft's slider sets."""
        return float(np.degrees(2.0 * np.arctan(self.height / 2.0 / self.focal_pixels)))

    @property
    def inlier_fraction(self) -> float:
        return self.inliers / self.segments if self.segments else 0.0


def estimate_session(
    frames: "dict[str, np.ndarray]",
    *,
    threshold: float = INLIER_PIXELS,
    minimum_length: float = MINIMUM_LENGTH,
    **kwargs,
) -> "dict[str, CalibrationResult]":
    """Calibrate a set of frames that share one field of view.

    Per-view fitting alone got 96 of 109 fixture views inside the gate, and the
    thirteen failures were not identifiable from anything the fit reported -
    inlier fraction on the failures (92.7% median) was indistinguishable from the
    successes (93.7%). A method that is right seven times in eight and cannot say
    which time is which is difficult to use, since production has no manifest to
    check against.

    Focal length is what rescues it, because it is not a per-view quantity at
    all. Minecraft's field of view is a setting, so every screenshot from one
    session shares it, and the *median* estimate across views is unaffected by
    however badly a few of them fitted. Measured across all six fixtures the
    median came out at exactly the true focal every time, including on the two
    fixtures whose individual estimates ranged over hundreds of pixels.

    So: fit each view freely, take the median focal as the session's, then refit
    every view with focal held there. The second pass searches two parameters
    instead of three and is correspondingly quicker.

    **The median is only meaningful across comparable image geometries.** A focal
    length in pixels is a property of the sampling, not of the lens: the same
    field of view on a frame half the height is half the focal length, and a crop
    changes the height without changing the field of view at all. Pooling raw
    pixel focals across differently sized or differently cropped frames therefore
    averages quantities that are not the same quantity. Frames are grouped by
    dimensions and each group gets its own median; a group of one keeps its own
    estimate rather than borrowing a stranger's.

    Only image evidence is used. Nothing here consults a manifest.
    """
    first: dict[str, CalibrationResult] = {}
    cached: dict[str, np.ndarray] = {}
    for frame_id, grey in frames.items():
        cached[frame_id] = detect_segments(enhance(grey),
                                           minimum_length=minimum_length)
        found = estimate_camera(grey, segments=cached[frame_id],
                                threshold=threshold, **kwargs)
        if found is not None:
            first[frame_id] = found
    if not first:
        return {}

    groups: dict[tuple[int, int], list[float]] = {}
    for result in first.values():
        groups.setdefault((result.width, result.height), []).append(result.focal_pixels)
    medians = {shape: float(np.median(values)) for shape, values in groups.items()}

    refined: dict[str, CalibrationResult] = {}
    for frame_id, result in first.items():
        shape = (result.width, result.height)
        session_focal = medians[shape]
        found = estimate_camera(frames[frame_id], segments=cached[frame_id],
                                threshold=threshold, focal=session_focal, **kwargs)
        if found is None:
            continue
        shifted = abs(result.focal_pixels - session_focal) / session_focal
        note = (f"focal fixed to the median {session_focal:.1f}px of the "
                f"{len(groups[shape])} frame(s) at {shape[0]}x{shape[1]}; "
                f"this view's own estimate differed by {shifted:.1%}")
        if len(groups[shape]) == 1:
            note = ("only frame at this size, so its own focal stands rather than "
                    "one pooled from frames of a different geometry")
        # Every measured field is carried across. Rebuilding the result and
        # forgetting the refined roll and optical centre threw away the very
        # refinement that had just been computed, before any consumer saw it.
        refined[frame_id] = CalibrationResult(
            focal_pixels=found.focal_pixels, yaw_degrees=found.yaw_degrees,
            pitch_degrees=found.pitch_degrees, width=found.width,
            height=found.height, inliers=found.inliers, segments=found.segments,
            median_residual=found.median_residual,
            per_axis_inliers=found.per_axis_inliers,
            roll_degrees=found.roll_degrees,
            principal_point=found.principal_point,
            notes=found.notes + (note,),
        )
    return refined


def estimate_image_calibrations(
    frames_rgb: "dict[str, np.ndarray]",
) -> "dict[str, CalibrationResult | None]":
    """Estimate each frame independently from RGB pixels and the default GUI mask.

    This is the public image-only seam for callers that must not provide a
    camera manifest or a hand-authored mask. It deliberately does not call
    :func:`estimate_session`: equal-sized frames may have different FOVs, so
    pooling their focal lengths would add an unsupported shared-setting
    assumption. Pixels excluded by the default GUI mask are zero-filled before
    line detection, matching the existing vision convention. The resulting
    boundary segments from that fill are accepted by the current calibration
    model; excluded source pixel values themselves cannot affect the fit.

    Invalid or unmeasurable frames are retained in the result map with ``None``
    so a multi-frame caller can record a typed per-frame abstention and continue.
    """
    if not isinstance(frames_rgb, dict):
        raise TypeError("frames_rgb must be a mapping of frame ids to RGB arrays")
    if any(not isinstance(frame_id, str) or not frame_id for frame_id in frames_rgb):
        raise TypeError("frame ids must be non-empty strings")
    import cv2
    from mc_photogrammetry.vision.gui_mask import create_gui_mask

    results: dict[str, CalibrationResult | None] = {}
    for frame_id in sorted(frames_rgb):
        image = frames_rgb[frame_id]
        if (not isinstance(frame_id, str) or not frame_id
                or not isinstance(image, np.ndarray) or image.ndim != 3
                or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0
                or image.dtype != np.uint8
                or not np.isfinite(image).all()):
            results[frame_id] = None
            continue
        try:
            rgb = np.asarray(image, dtype=np.uint8)
            grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            usable = ~create_gui_mask(grey.shape[0], grey.shape[1])
            grey = grey.copy()
            grey[~usable] = 0
            results[frame_id] = estimate_camera(grey)
        except (cv2.error, TypeError, ValueError, IndexError, FloatingPointError,
                OverflowError):
            results[frame_id] = None
    return results


def to_camera_solution(results: "dict[str, CalibrationResult]"):
    """Express these estimates in the shared `CameraSolution` contract.

    Now lossless for everything that was measured. `CameraPose.position` is
    optional, so orientation travels as a real pose with the position left
    unset, rather than the whole pose being dropped to avoid inventing one -
    which is what an earlier version did, discarding measured yaw and pitch in
    order to be honest about an unmeasured translation.

    Three things stay explicitly unresolved and are named per frame rather than
    left for a consumer to infer:

    * **Position.** Vanishing points are unchanged by moving along a ray.
    * **Scale.** `scale_known` is False; the lattice period would supply it.
    * **The yaw quadrant.** One view cannot tell which horizontal lattice axis is
      which, so yaw is meaningful only modulo 90 degrees.

    `confidence` is deliberately **not** the inlier fraction. That statistic was
    measured across 109 views and does not separate a good estimate from a bad
    one - 93.7% median on the passes against 92.7% on the failures - so passing
    it as confidence would assert a correctness it has been shown not to carry.
    The inlier fraction is reported in `residual_pixels`' company as a
    diagnostic instead.
    """
    from mc_photogrammetry.pipeline.contracts import (
        CameraEstimate,
        CameraIntrinsics,
        CameraPose,
        CameraSolution,
    )

    cameras, unresolved = {}, []
    for frame_id, result in results.items():
        centre = result.principal_point or (result.width / 2.0, result.height / 2.0)
        cameras[frame_id] = CameraEstimate(
            intrinsics=CameraIntrinsics(
                image_size=(result.width, result.height),
                focal_length_pixels=(result.focal_pixels, result.focal_pixels),
                principal_point_pixels=(float(centre[0]), float(centre[1])),
            ),
            pose=CameraPose(
                position=None,
                yaw_degrees=result.yaw_degrees,
                pitch_degrees=result.pitch_degrees,
                roll_degrees=result.roll_degrees,
            ),
            residual_pixels=result.median_residual,
            confidence=0.0,
        )
        unresolved.append(f"{frame_id}: position (vanishing points fix no translation)")
        unresolved.append(f"{frame_id}: scale (needs the lattice period)")
        unresolved.append(f"{frame_id}: yaw quadrant (recovered modulo 90 degrees)")
    return CameraSolution(cameras=cameras, unresolved=tuple(unresolved),
                          scale_known=False)


def enhance(grey: np.ndarray) -> np.ndarray:
    """Lift a dim capture so its seams are detectable.

    Fixtures shot at gamma 100% mean about 34 with a 99th percentile of 88, and a
    detector run on them finds roughly a third of the segments it finds after
    correction. Gamma first, then local equalisation.
    """
    import cv2

    table = (np.linspace(0.0, 1.0, 256) ** GAMMA * 255.0).astype(np.uint8)
    lifted = cv2.LUT(grey, table)
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                           tileGridSize=(CLAHE_GRID, CLAHE_GRID)).apply(lifted)


def detect_segments(
    grey: np.ndarray,
    *,
    scales: Sequence[float] = DETECTION_SCALES,
    minimum_length: float = MINIMUM_LENGTH,
) -> np.ndarray:
    """Line segments as ``(x0, y0, x1, y1)`` rows, in full-resolution pixels."""
    import cv2

    detector = cv2.createLineSegmentDetector()
    gathered = []
    for scale in scales:
        source = grey if scale == 1.0 else cv2.resize(
            grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        found = detector.detect(source)[0]
        if found is None:
            continue
        gathered.append(found.reshape(-1, 4) / scale)
    if not gathered:
        return np.zeros((0, 4))
    segments = np.vstack(gathered)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    return segments[lengths >= minimum_length]


CONTRAST_OFFSET = 2.5
"""How far either side of a segment its contrast is sampled, in pixels."""

CONTRAST_SAMPLES = 12
"""Points along a segment at which contrast is measured."""

CONTRAST_FLOOR = 0.25
"""Weight given to a segment with no contrast across it at all.

Not zero. A texture line that happens to run along a lattice direction is still
evidence about that direction - it is drawn on a face that is genuinely oriented
that way. Discarding it outright throws away most of what a dim frame offers,
which is why this discounts rather than filters.
"""


def segment_contrast(
    image: np.ndarray,
    segments: np.ndarray,
    *,
    offset: float = CONTRAST_OFFSET,
    samples: int = CONTRAST_SAMPLES,
) -> np.ndarray:
    """How strongly each segment separates what lies on either side of it.

    A structural edge - a block seam, a silhouette, the join between two
    surfaces - has different material or different vanilla face shading on its
    two sides, so sampling across it finds a step. A line inside a single block's
    texture has the same material and the same shading on both sides, so it does
    not.

    Direction alone cannot make this distinction, because Minecraft's textures
    are themselves axis-aligned and their internal lines run along the very same
    lattice directions as the seams. Contrast across the line is independent of
    direction, which is what makes it worth measuring.

    Returned as the mean absolute difference in intensity, in levels.
    """
    grey = image if image.ndim == 2 else image.mean(axis=2)
    height, width = grey.shape[:2]
    if not len(segments):
        return np.zeros(0)

    fractions = np.linspace(0.15, 0.85, samples)[None, :]
    x0, y0 = segments[:, 0][:, None], segments[:, 1][:, None]
    x1, y1 = segments[:, 2][:, None], segments[:, 3][:, None]
    xs = x0 + fractions * (x1 - x0)
    ys = y0 + fractions * (y1 - y0)

    dx, dy = segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]
    length = np.maximum(np.hypot(dx, dy), 1e-9)
    # The unit normal to each segment, along which the two sides are sampled.
    nx, ny = (-dy / length)[:, None], (dx / length)[:, None]

    def sample(px, py):
        rows = np.clip(np.rint(py).astype(int), 0, height - 1)
        columns = np.clip(np.rint(px).astype(int), 0, width - 1)
        return grey[rows, columns]

    near = sample(xs + nx * offset, ys + ny * offset)
    far = sample(xs - nx * offset, ys - ny * offset)
    return np.abs(near.astype(float) - far.astype(float)).mean(axis=1)


def structural_weights(
    image: np.ndarray,
    segments: np.ndarray,
    *,
    floor: float = CONTRAST_FLOOR,
    **kwargs,
) -> np.ndarray:
    """Per-segment confidence that a line is structure rather than texture.

    Scaled against the frame's own median contrast rather than an absolute level,
    because a fixture shot at gamma 100% has a fraction of the contrast of one
    shot at fullbright and a fixed threshold would classify a whole scene as
    texture. Bounded into ``[floor, 1]``.
    """
    contrast = segment_contrast(image, segments, **kwargs)
    if not contrast.size:
        return contrast
    # The median is the scale of choice, but it goes to zero exactly when the
    # frame is mostly texture - a thin line inside a texture has the same
    # material on both sides, so its contrast is genuinely nothing, and a scene
    # full of them has a zero median. That is the case this most needs to work
    # in, so the mean stands in rather than the whole frame being called
    # structural by a divide-by-nothing guard.
    reference = float(np.median(contrast))
    if reference <= 1e-6:
        reference = float(np.mean(contrast))
    if reference <= 1e-6:
        return np.ones(contrast.shape)
    scaled = contrast / (2.0 * reference)
    return np.clip(scaled, floor, 1.0)


def axis_directions(yaw_degrees: float, pitch_degrees: float,
                    roll_degrees: float = 0.0) -> np.ndarray:
    """The three world axes as camera-space directions ``(right, up, forward)``.

    Every part of this defers to `camera.camera_space_points`, including the
    roll. An earlier version applied roll here by hand, because `CameraModel`
    had nowhere to put it; it now does, and the two disagreed by 0.832 in
    maximum component at yaw 30, pitch -10, roll 25 - the hand-rolled rotation
    turned the opposite way. Negating it reproduced the shared basis to 6e-17,
    which is the whole argument against keeping a second copy of a convention.
    """
    from mc_photogrammetry.reconstruction.camera import CameraModel, camera_space_points

    camera = CameraModel(position=(0.0, 0.0, 0.0), yaw_degrees=yaw_degrees,
                         pitch_degrees=pitch_degrees, fov_degrees=70.0,
                         roll_degrees=roll_degrees)
    eye = np.asarray(camera.position, dtype=float) + np.asarray(
        [0.0, camera.eye_height, 0.0])
    return camera_space_points([tuple(eye + axis) for axis in np.eye(3)], camera)


def vanishing_points(
    focal: float, yaw_degrees: float, pitch_degrees: float,
    width: int, height: int,
    roll_degrees: float = 0.0,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """The three vanishing points, homogeneous, as ``(x, y, w)`` rows.

    A direction parallel to the image plane vanishes at infinity, which happens
    for the vertical axis whenever the pitch is level. Homogeneous coordinates
    carry that case instead of dividing by zero.

    ``principal_point`` defaults to the image centre, which is where a whole
    screenshot puts it. A *cropped* one does not: cutting pixels off one side
    moves the optical centre by half that amount, and the frame is no longer
    describable by focal length and orientation alone.
    """
    centre_x, centre_y = principal_point or (width / 2.0, height / 2.0)
    points = []
    for right, up, forward in axis_directions(yaw_degrees, pitch_degrees,
                                              roll_degrees):
        points.append((centre_x * forward + right * focal,
                       centre_y * forward - up * focal,
                       forward))
    return np.asarray(points, dtype=float)


def segment_residuals(segments: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Endpoint displacement, in pixels, of each segment against each direction.

    How far a segment's ends sit off the line joining its midpoint to the
    vanishing point. In pixels, so it is the same quantity a seam-displacement
    gate is written in, and it scales with the segment's own length - a long wall
    edge pointing slightly wrong is a worse error than a short one.
    """
    if not len(segments):
        return np.zeros((0, len(points)))
    midpoints = np.stack([(segments[:, 0] + segments[:, 2]) / 2.0,
                          (segments[:, 1] + segments[:, 3]) / 2.0], axis=1)
    deltas = np.stack([segments[:, 2] - segments[:, 0],
                       segments[:, 3] - segments[:, 1]], axis=1)
    lengths = np.linalg.norm(deltas, axis=1)
    unit = deltas / np.maximum(lengths, 1e-9)[:, None]

    out = np.empty((len(segments), len(points)))
    for index, (x, y, w) in enumerate(points):
        if abs(w) > 1e-12:
            toward = np.asarray([x / w, y / w])[None, :] - midpoints
        else:
            toward = np.broadcast_to(np.asarray([x, y]), midpoints.shape)
        span = np.linalg.norm(toward, axis=1)
        cross = np.abs(unit[:, 0] * toward[:, 1] - unit[:, 1] * toward[:, 0])
        sine = cross / np.maximum(span, 1e-9)
        out[:, index] = sine * lengths / 2.0
    return out


def _score(segments, lengths, focal, yaw, pitch, width, height, threshold,
           roll=0.0, principal_point=None):
    residuals = segment_residuals(segments, vanishing_points(
        focal, yaw, pitch, width, height, roll, principal_point))
    best = residuals.min(axis=1)
    inside = best <= threshold
    # Weighted by length: a long structural edge is stronger evidence than a
    # short texture line, and Minecraft frames are full of the latter.
    return float((lengths[inside] * (1.0 - best[inside] / threshold)).sum())


def refine_optics(
    segments: np.ndarray,
    lengths: np.ndarray,
    focal: float,
    yaw: float,
    pitch: float,
    width: int,
    height: int,
    threshold: float,
    *,
    margin: float = REFINE_MARGIN,
    roll_limit: float = ROLL_LIMIT,
    principal_limit: float = PRINCIPAL_LIMIT,
):
    """Search roll and principal point, and keep them only if they earn their keep.

    The main fit assumes a level camera and an optical centre at the middle of
    the frame. Both are true of a whole Minecraft screenshot and neither is true
    of a cropped or rotated one, which the image scope admits - cutting pixels
    off an edge shifts the optical centre by half of what was removed, and the
    frame stops being describable by focal length and orientation alone.

    So they are searched, and then *disbelieved by default*. Three more free
    parameters will always fit the noise a little better; the refinement is
    accepted only when it improves the score by `margin`, which on an uncropped
    frame it does not. Yaw and pitch are re-polished afterwards, because moving
    the optical centre moves where the vanishing points should sit.

    Returns ``(roll, principal_point, yaw, pitch, improved, focal)``. The focal
    length comes back because it is refitted alongside the roll.
    """
    base = _score(segments, lengths, focal, yaw, pitch, width, height, threshold)
    if base <= 0.0:
        return 0.0, None, yaw, pitch, False, focal

    # All four of these trade off against one another, so none can be settled
    # while the others are held at values chosen without it. The main search
    # assumes a level camera with a centred lens, and absorbs any real roll or
    # crop into a wrong focal - on a 3 degree synthetic roll it returned 588px
    # for a true 514 - after which the correct roll scores *worse* than zero.
    # Fixing them in a set order merely moves which one gets absorbed, so they
    # are alternated instead, each sweep re-fitting the focal alongside.
    centred = (width / 2.0, height / 2.0)
    span_x, span_y = principal_limit * width, principal_limit * height
    scales = np.arange(0.80, 1.2501, 0.025)

    value, roll, point = base, 0.0, centred
    for _round in range(2):
        # Each candidate grid belongs to the sweep's starting camera. Updating
        # its anchor after a better candidate makes the neighborhood depend on
        # enumeration order and permits compounded focal steps within one grid.
        roll_focal_anchor = focal
        for candidate_roll in np.arange(-roll_limit, roll_limit + 1e-9, 0.5):
            for scale in scales:
                found = _score(segments, lengths, roll_focal_anchor * scale, yaw, pitch,
                               width, height, threshold, candidate_roll, point)
                if found > value:
                    value, roll, focal = found, float(candidate_roll), roll_focal_anchor * scale

        point_focal_anchor = focal
        for dx in np.linspace(-span_x, span_x, 7):
            for dy in np.linspace(-span_y, span_y, 7):
                candidate = (width / 2.0 + dx, height / 2.0 + dy)
                for scale in scales:
                    found = _score(segments, lengths, point_focal_anchor * scale, yaw, pitch,
                                   width, height, threshold, roll, candidate)
                    if found > value:
                        value, point, focal = found, candidate, point_focal_anchor * scale

    if value <= base * (1.0 + margin):
        return 0.0, None, yaw, pitch, False, focal

    # Moving the optical centre moves where the vanishing points belong, so the
    # orientation is worth re-polishing against the new optics.
    for span, step in ((1.0, 0.25), (0.3, 0.05)):
        yaw_anchor, pitch_anchor = yaw, pitch
        for candidate_yaw in np.arange(yaw_anchor - span, yaw_anchor + span + 1e-9, step):
            for candidate_pitch in np.arange(pitch_anchor - span, pitch_anchor + span + 1e-9, step):
                found = _score(segments, lengths, focal, candidate_yaw,
                               candidate_pitch, width, height, threshold,
                               roll, point)
                if found > value:
                    value, yaw, pitch = found, float(candidate_yaw), float(candidate_pitch)

    if point == centred:
        point = None
    return roll, point, yaw, pitch, True, focal


def estimate_camera(
    grey: np.ndarray,
    *,
    minimum_length: float = MINIMUM_LENGTH,
    threshold: float = INLIER_PIXELS,
    scales: Sequence[float] = DETECTION_SCALES,
    fov_range: tuple[float, float] = (30.0, 120.0),
    segments: np.ndarray | None = None,
    coarse_segments: int = COARSE_SEGMENTS,
    focal: float | None = None,
    refine_optical_centre: bool = True,
    weight_by_contrast: bool = False,
) -> CalibrationResult | None:
    """Estimate focal length, yaw and pitch from one greyscale frame.

    Coarse-to-fine over the three parameters. The search is deliberately over the
    camera rather than over vanishing points: every hypothesis is then a real
    camera, and no stage can produce three vanishing points that are not mutually
    orthogonal.
    """
    height, width = grey.shape[:2]
    if segments is None:
        segments = detect_segments(enhance(grey), scales=scales,
                                   minimum_length=minimum_length)
    if len(segments) < 12:
        return None
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    if weight_by_contrast and grey.size:
        # Discount lines that separate nothing: they are texture drawn on a face,
        # not the face's own edge. Whether that helps is a measurement, not an
        # assumption, which is why it is off by default.
        lengths = lengths * structural_weights(grey, segments)

    def focal_for(fov: float) -> float:
        return (height / 2.0) / np.tan(np.radians(fov) / 2.0)

    # The coarse sweep runs on the longest segments only. Measured on the M3
    # frame: restricting to segments over 120px returned the identical camera as
    # using all 2151, because a long structural edge constrains a vanishing point
    # far more tightly than a short texture line. This is a speed change resting
    # on that result, not a change of answer - `test_the_coarse_cap_does_not_move
    # _the_answer` pins it.
    order = np.argsort(-lengths)
    coarse = segments[order[:coarse_segments]]
    coarse_lengths = lengths[order[:coarse_segments]]

    focals = ([float(focal)] if focal is not None
              else [focal_for(f) for f in np.arange(fov_range[0], fov_range[1] + 1, 5.0)])

    best = None
    # Yaw only needs [0, 90): a quarter turn swaps the two horizontal axes and
    # leaves the set of vanishing points identical.
    for yaw in np.arange(0.0, 90.0, 2.0):
        for pitch in np.arange(-88.0, 88.1, 2.0):
            for candidate in focals:
                value = _score(coarse, coarse_lengths, candidate, yaw, pitch,
                               width, height, threshold)
                if best is None or value > best[0]:
                    best = (value, candidate, yaw, pitch)
    # Re-score the coarse winner against everything, so the refinement below
    # compares like with like.
    best = (_score(segments, lengths, best[1], best[2], best[3],
                   width, height, threshold), best[1], best[2], best[3])

    scales = [1.0] if focal is not None else list(np.arange(0.94, 1.0601, 0.01))
    for span, step in ((2.0, 0.5), (0.6, 0.15), (0.2, 0.05)):
        _value, centre_focal, yaw, pitch = best
        for candidate_yaw in np.arange(yaw - span, yaw + span + 1e-9, step):
            for candidate_pitch in np.arange(pitch - span, pitch + span + 1e-9, step):
                for scale in scales:
                    candidate_focal = centre_focal * scale
                    value = _score(segments, lengths, candidate_focal, candidate_yaw,
                                   candidate_pitch, width, height, threshold)
                    if value > best[0]:
                        best = (value, candidate_focal, candidate_yaw, candidate_pitch)

    _value, focal, yaw, pitch = best

    roll, principal_point = 0.0, None
    if refine_optical_centre:
        roll, principal_point, yaw, pitch, improved, refined_focal = refine_optics(
            segments, lengths, focal, yaw, pitch, width, height, threshold)
        if improved:
            focal = refined_focal
        else:
            roll, principal_point = 0.0, None

    residuals = segment_residuals(segments, vanishing_points(
        focal, yaw, pitch, width, height, roll, principal_point))
    owner = residuals.argmin(axis=1)
    closest = residuals.min(axis=1)
    inside = closest <= threshold
    counts = tuple(int(((owner == axis) & inside).sum()) for axis in range(3))
    return CalibrationResult(
        focal_pixels=float(focal),
        yaw_degrees=float(yaw % 90.0),
        pitch_degrees=float(pitch),
        width=int(width),
        height=int(height),
        inliers=int(inside.sum()),
        segments=int(len(segments)),
        median_residual=float(np.median(closest[inside])) if inside.any() else float("nan"),
        per_axis_inliers=counts,
        roll_degrees=float(roll),
        principal_point=principal_point,
        notes=("yaw is modulo 90 degrees: one view cannot name the horizontal axes",)
        + (() if principal_point is None else
           (f"optical centre moved to {principal_point[0]:.0f},"
            f"{principal_point[1]:.0f} and roll to {roll:.2f} deg; "
            "the frame is cropped or rotated",)),
    )
