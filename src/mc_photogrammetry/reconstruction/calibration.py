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
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fov_degrees(self) -> float:
        """Vertical field of view, which is the quantity Minecraft's slider sets."""
        return float(np.degrees(2.0 * np.arctan(self.height / 2.0 / self.focal_pixels)))

    @property
    def inlier_fraction(self) -> float:
        return self.inliers / self.segments if self.segments else 0.0


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


def axis_directions(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    """The three world axes as camera-space directions ``(right, up, forward)``.

    Matches `camera.project_points`: screen right is `view_x`, screen up is
    `rotated_y`, and `depth` is forward.
    """
    yaw, pitch = np.radians(yaw_degrees), np.radians(pitch_degrees)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.asarray([
        [-cy, -sy * sp, -sy * cp],   # world +X
        [0.0, cp, -sp],              # world +Y
        [-sy, cy * sp, cy * cp],     # world +Z
    ])


def vanishing_points(
    focal: float, yaw_degrees: float, pitch_degrees: float,
    width: int, height: int,
) -> np.ndarray:
    """The three vanishing points, homogeneous, as ``(x, y, w)`` rows.

    A direction parallel to the image plane vanishes at infinity, which happens
    for the vertical axis whenever the pitch is level. Homogeneous coordinates
    carry that case instead of dividing by zero.
    """
    points = []
    for right, up, forward in axis_directions(yaw_degrees, pitch_degrees):
        points.append((width / 2.0 * forward + right * focal,
                       height / 2.0 * forward - up * focal,
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


def _score(segments, lengths, focal, yaw, pitch, width, height, threshold):
    residuals = segment_residuals(segments, vanishing_points(
        focal, yaw, pitch, width, height))
    best = residuals.min(axis=1)
    inside = best <= threshold
    # Weighted by length: a long structural edge is stronger evidence than a
    # short texture line, and Minecraft frames are full of the latter.
    return float((lengths[inside] * (1.0 - best[inside] / threshold)).sum())


def estimate_camera(
    grey: np.ndarray,
    *,
    minimum_length: float = MINIMUM_LENGTH,
    threshold: float = INLIER_PIXELS,
    scales: Sequence[float] = DETECTION_SCALES,
    fov_range: tuple[float, float] = (30.0, 120.0),
    segments: np.ndarray | None = None,
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

    def focal_for(fov: float) -> float:
        return (height / 2.0) / np.tan(np.radians(fov) / 2.0)

    best = None
    # Yaw only needs [0, 90): a quarter turn swaps the two horizontal axes and
    # leaves the set of vanishing points identical.
    grids = (
        (np.arange(0.0, 90.0, 2.0), np.arange(-88.0, 88.1, 2.0),
         np.array([focal_for(f) for f in np.arange(fov_range[0], fov_range[1] + 1, 5.0)])),
    )
    for yaws, pitches, focals in grids:
        for yaw in yaws:
            for pitch in pitches:
                for focal in focals:
                    value = _score(segments, lengths, focal, yaw, pitch,
                                   width, height, threshold)
                    if best is None or value > best[0]:
                        best = (value, focal, yaw, pitch)

    for span, step in ((2.0, 0.5), (0.6, 0.15), (0.2, 0.05)):
        _value, focal, yaw, pitch = best
        for candidate_yaw in np.arange(yaw - span, yaw + span + 1e-9, step):
            for candidate_pitch in np.arange(pitch - span, pitch + span + 1e-9, step):
                for scale in np.arange(0.94, 1.0601, 0.01):
                    candidate_focal = focal * scale
                    value = _score(segments, lengths, candidate_focal, candidate_yaw,
                                   candidate_pitch, width, height, threshold)
                    if value > best[0]:
                        best = (value, candidate_focal, candidate_yaw, candidate_pitch)

    _value, focal, yaw, pitch = best
    residuals = segment_residuals(segments, vanishing_points(
        focal, yaw, pitch, width, height))
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
        notes=("yaw is modulo 90 degrees: one view cannot name the horizontal axes",),
    )
